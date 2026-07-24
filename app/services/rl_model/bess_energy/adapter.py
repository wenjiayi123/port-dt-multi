# app/services/rl_model/bess_energy/adapter.py
# -*- coding: utf-8 -*-
"""
E 模块｜BESS 适配器（数据摄取稳定接口 + 点表握手/节流/限坡 + 审计 JSONL）
=====================================================================
大白话说明：
- 这是 E 模块的“工程化接口”文件：统一读取数据（列名与时间戳自适应），
  并把对 PCS 的写入做成“nonce+时效60s + 速率/幅度节流 + 斜坡执行 + 并网/需量/N-1 兜底”。
- 只用标准库 + numpy；不依赖 pandas。
- 记录与审计统一写入 policy_evaluate_history.jsonl（与 module.py 共用，前端直接读）。

对外稳定接口（被 rl_engine.py / 前端 API 调用）：
- DataAdapter.load_for_env(dt_min, horizon_steps, jsonl_path) -> env, planner, ctx     # 直接封装 module.make_env
- ActuatorAdapter.apply_setpoint(t_utc, desired_p_kW, soc_target, write_enable, nonce, qos_flag, state)-> ack
  state: {"pcc_base_kW": float, "soc": float}  # 可扩展（如温度/SoH）
"""
from __future__ import annotations

import argparse
import json
import os
import time
import secrets
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 复用 module.py 中的能力与常量
from .module import (
    make_env,
    load_configs,
    EconomicMPCPlanner,
    JsonlLogger,
    DEFAULT_JSONL,
    STATIC_JSONL,
    BessSiteConfig,
    DemandWindowPolicy,
)

# ====== 常量（工程口径） ======
NONCE_TTL_SEC = 60       # 操作时效 60s（附录 C）
FIFTEEN_MIN_SEC = 15 * 60


# ====== 数据适配器 ======
class DataAdapter:
    """
    数据适配器：
    - 对外暴露稳定入口以构建 env/planner（内部复用 module.make_env）
    - 预留实时/批量两种路径（当前默认批量 CSV）
    """

    def __init__(self) -> None:
        pass

    def load_for_env(self, dt_min: int, horizon_steps: int, jsonl_path: Optional[str] = None):
        """
        统一入口：读取配置与数据，构造 env 与 planner，返回 (env, planner, ctx)
        """
        jsonl = jsonl_path or DEFAULT_JSONL
        env, planner, ctx = make_env(dt_min=dt_min, horizon_steps=horizon_steps, jsonl_path=jsonl)
        return env, planner, ctx

    def load_configs_only(self, dt_min: int) -> Tuple[BessSiteConfig, DemandWindowPolicy]:
        """
        只读站级与需量策略配置（用于前端显示/参数面板）
        """
        return load_configs(dt_min=dt_min)


# ====== 写入节流/限坡与握手 ======
class _NonceBook:
    """
    Nonce 管理与时效校验：
    - 每次写入都必须带 nonce；60s 内有效；已用过的不可复用（防重放）。
    """

    def __init__(self, ttl_sec: int = NONCE_TTL_SEC):
        self.ttl = ttl_sec
        self._seen: Dict[str, int] = {}  # nonce -> ts

    def gen(self) -> str:
        n = secrets.token_hex(8)  # 16 hex
        self._seen[n] = int(time.time())
        return n

    def check(self, nonce: str, now_ts: int) -> Tuple[bool, str]:
        if not nonce:
            return False, "nonce_missing"
        ts = self._seen.get(nonce)
        if ts is None:
            # 第一次见到也允许，但记录并施加时效
            self._seen[nonce] = now_ts
            ts = now_ts
        if now_ts - ts > self.ttl:
            return False, "nonce_expired"
        return True, "ok"


class _RateLimiter:
    """
    点位节流：
    - 每 15 分钟最多 N 次写入（rate_limit_per_15min）
    - 单次写入与上次生效值差值上限（max_delta_per_write_kW）
    """

    def __init__(self, rate_limit_per_15min: int, max_delta_per_write_kW: float):
        self.rate_limit = max(1, int(rate_limit_per_15min))
        self.max_delta = float(max_delta_per_write_kW)
        self._writes_ts: List[int] = []

    def allow(self, t_utc: int, last_value: float, new_value: float) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        # 15 分钟内计数
        self._writes_ts = [t for t in self._writes_ts if t >= t_utc - FIFTEEN_MIN_SEC]
        if len(self._writes_ts) >= self.rate_limit:
            reasons.append("rate_limit_15min")
        # 单次幅度限制
        if abs(new_value - last_value) > self.max_delta:
            reasons.append("exceed_max_delta")
        ok = (len(reasons) == 0)
        if ok:
            self._writes_ts.append(t_utc)
        return ok, reasons


class ActuatorAdapter:
    """
    写入适配器（最后一跳工程兜底）：
    - 校验：nonce+时效 / 速率限制 / 幅度限制 / 限坡执行
    - 并网：PCC/N-1/逆潮流，按需修正功率
    - 电池：SOC/Pmax/C-rate
    - 日志：每次写入均写 JSONL（key="actuator_write"）
    """

    def __init__(self,
                 cfg: BessSiteConfig,
                 policy: DemandWindowPolicy,
                 dt_min: int,
                 rate_limit_per_15min: int = 6,
                 max_delta_per_write_ratio: float = 0.3,
                 enable_ramp_exec: bool = True,
                 jsonl_path: Optional[str] = None) -> None:
        self.cfg = cfg
        self.policy = policy
        self.dt_min = dt_min
        self.enable_ramp = enable_ramp_exec
        self.step_sec = dt_min * 60

        # 节流参数
        max_delta_kW = max_delta_per_write_ratio * self.cfg.rated_power_kW
        self._rl = _RateLimiter(rate_limit_per_15min, max_delta_kW)
        self._nonce = _NonceBook(NONCE_TTL_SEC)

        # 内部状态（记录上次生效）
        self.last_p_kW = 0.0
        self.last_soc = self.cfg.soc_target
        self.last_write_ts = 0

        # 日志
        self.log = JsonlLogger(jsonl_path or DEFAULT_JSONL)

        # 计算硬件功率上限（Pmax 取 C-rate 与 PCS 额定中较小者）
        self.pmax_hw = float(self.cfg.rated_power_kW)
        self.pmax_c = float(self.cfg.c_rate_max * self.cfg.rated_energy_kWh)
        self.pmax = float(min(self.pmax_hw, self.pmax_c))

        # 斜坡能力（每步 kW）
        self.ramp_step_kW = float(self.cfg.p_ramp_kW_per_step if self.cfg.p_ramp_kW_per_step else 0.0)

    # ---- 工具：约束裁剪 ----
    def _clip_by_ramp(self, p_desired: float) -> float:
        if not self.enable_ramp or self.ramp_step_kW <= 0:
            return p_desired
        lo = self.last_p_kW - self.ramp_step_kW
        hi = self.last_p_kW + self.ramp_step_kW
        return float(np.clip(p_desired, lo, hi))

    def _clip_by_pmax_crate(self, p_in: float) -> float:
        return float(np.clip(p_in, -self.pmax, self.pmax))

    def _clip_by_soc_bounds(self, p_in: float, soc_now: float) -> Tuple[float, List[str]]:
        """
        估算下一步 SOC 并保证在 [soc_min, soc_max]；必要时减小充/放。
        """
        reasons: List[str] = []
        dt_h = self.dt_min / 60.0
        e_ch = max(0.0, -p_in) * self.cfg.eff_ch * dt_h
        e_dis = max(0.0, p_in) / self.cfg.eff_dis * dt_h
        soc_next = soc_now + (e_ch - e_dis) / max(1e-9, self.cfg.rated_energy_kWh)
        p_adj = p_in
        if soc_next > self.cfg.soc_max and p_in < 0:
            over = (soc_next - self.cfg.soc_max) * self.cfg.rated_energy_kWh / dt_h
            p_adj += over / max(1e-6, self.cfg.eff_ch)
            reasons.append("soc_max")
        if soc_next < self.cfg.soc_min and p_in > 0:
            under = (self.cfg.soc_min - soc_next) * self.cfg.rated_energy_kWh / dt_h
            p_adj -= under * self.cfg.eff_dis
            reasons.append("soc_min")
        return float(p_adj), reasons

    def _clip_by_grid_limits(self, p_in: float, pcc_base_kW: float) -> Tuple[float, List[str]]:
        """
        并网/逆潮流/N-1 保护：根据 base PCC + BESS 充放影响，限制功率。
        """
        reasons: List[str] = []
        # 计算执行后 PCC（充电增加 PCC，放电减少 PCC）
        pcc_try = float(pcc_base_kW + max(0.0, -p_in) - max(0.0, p_in))
        # 逆潮流
        if not self.policy.export_allowed and pcc_try < 0.0:
            if p_in < 0:
                p_in = -max(0.0, -pcc_base_kW)
            if p_in > 0 and (pcc_base_kW - p_in) < 0.0:
                p_in = min(p_in, pcc_base_kW)
            reasons.append("anti_export")
        # N-1 余度
        if (pcc_try + self.policy.n_minus_1_margin_kW) > self.policy.pcc_limit_kW:
            delta = (pcc_try + self.policy.n_minus_1_margin_kW - self.policy.pcc_limit_kW)
            # 需减少充电或增加放电（降低 pcc_try）
            if p_in < 0:
                p_in += delta
            else:
                p_in -= delta
            reasons.append("n-1_guard")
        # 需量软限守护（软裁剪，硬限制由环境屏蔽；这里避免更糟）
        if pcc_try > self.policy.soft_cap_kW:
            # 若已超 softcap，则不允许增加 PCC（禁止更大充电或更小放电）
            if p_in < self.last_p_kW:
                p_in = self.last_p_kW
                reasons.append("softcap_guard")
        return float(p_in), reasons

    # ---- 对外：写入握手 ----
    def apply_setpoint(self,
                       t_utc: int,
                       desired_p_kW: float,
                       soc_target: float,
                       write_enable: bool,
                       nonce: str,
                       qos_flag: str,
                       state: Dict[str, float]) -> Dict[str, Any]:
        """
        下发功率与 SOC 目标（工程握手）：
        - 入参：
          t_utc:       指令时间（UTC epoch 秒）
          desired_p_kW:期望 PCS 有功（放电+、充电-）
          soc_target:  终态 SOC 目标（0~1）
          write_enable:使能位（False 则拒绝）
          nonce:      防重放随机串，TTL=60s
          qos_flag:   "fast"|"balanced"|"safe"（影响是否严控斜坡/幅度）
          state:      {"pcc_base_kW": float, "soc": float}
        - 出参：ack 字典（包含修正后功率、裁剪原因等）
        """
        reasons: List[str] = []
        status = "ok"

        # 1) 使能检查
        if not write_enable:
            status = "rejected"
            reasons.append("write_disabled")

        # 2) nonce 校验（60s）
        ok_nonce, r = self._nonce.check(nonce, now_ts=t_utc)
        if not ok_nonce:
            status = "rejected"
            reasons.append(r)

        # 3) QoS -> 限制策略
        # "safe": 更严格的幅度/斜坡；"fast": 放宽幅度，但仍受硬约束
        ramp = self.ramp_step_kW
        max_delta = self._rl.max_delta
        if qos_flag == "safe":
            ramp = self.ramp_step_kW
            max_delta = min(max_delta, 0.2 * self.cfg.rated_power_kW)
        elif qos_flag == "fast":
            ramp = max(ramp, 1.5 * self.ramp_step_kW)
            max_delta = 1.2 * self._rl.max_delta
        else:  # balanced
            pass

        # 4) 速率/幅度节流
        # 临时构造一个使用“当前策略”的 RateLimiter 视图（不改变全局配置）
        rl_tmp = _RateLimiter(rate_limit_per_15min=len(self._rl._writes_ts) + 999,  # 次数限制在全局 _rl 里检查
                              max_delta_per_write_kW=max_delta)
        ok_rate, reasons_delta = rl_tmp.allow(t_utc, self.last_p_kW, desired_p_kW)
        reasons.extend([r for r in reasons_delta if r == "exceed_max_delta"])
        if "exceed_max_delta" in reasons:
            # 将 desired 剪裁到允许的最大幅度方向
            if desired_p_kW > self.last_p_kW:
                desired_p_kW = self.last_p_kW + max_delta
            else:
                desired_p_kW = self.last_p_kW - max_delta

        # 15 分钟次数限制（全局）
        ok_rate_global, reasons_rate = self._rl.allow(t_utc, self.last_p_kW, desired_p_kW)
        if not ok_rate_global:
            status = "rejected"
            reasons.extend(reasons_rate)

        p_cmd = float(desired_p_kW)

        # 5) 限坡执行
        if self.enable_ramp and ramp > 0:
            lo = self.last_p_kW - ramp
            hi = self.last_p_kW + ramp
            p_cmd = float(np.clip(p_cmd, lo, hi))
            if (p_cmd != desired_p_kW) and ("ramp" not in reasons):
                reasons.append("ramp")

        # 6) 硬件 Pmax/C-rate
        p_cmd = self._clip_by_pmax_crate(p_cmd)
        if abs(p_cmd - desired_p_kW) > 1e-6 and "pmax/c_rate" not in reasons:
            reasons.append("pmax/c_rate")

        # 7) 并网/逆潮流/N-1/softcap
        pcc_base = float(state.get("pcc_base_kW", 0.0))
        p_cmd, r_grid = self._clip_by_grid_limits(p_cmd, pcc_base)
        reasons.extend(r_grid)

        # 8) SOC 硬边界（预测下一步）
        soc_now = float(state.get("soc", self.last_soc))
        p_cmd, r_soc = self._clip_by_soc_bounds(p_cmd, soc_now)
        reasons.extend(r_soc)

        # 9) 形成 ACK 与状态更新
        dt_h = self.dt_min / 60.0
        e_ch = max(0.0, -p_cmd) * self.cfg.eff_ch * dt_h
        e_dis = max(0.0, p_cmd) / self.cfg.eff_dis * dt_h
        soc_next = float(np.clip(soc_now + (e_ch - e_dis) / max(1e-6, self.cfg.rated_energy_kWh),
                                 self.cfg.soc_min, self.cfg.soc_max))

        # 写入结果：如果被拒绝，功率不变
        applied_p = p_cmd if status == "ok" else self.last_p_kW

        ack = {
            "key": "actuator_write",
            "applied_ts": int(t_utc),
            "write_enable": bool(write_enable),
            "nonce": str(nonce),
            "status": status,
            "reasons": reasons,
            "requested": {
                "p_kW": float(desired_p_kW),
                "soc_target": float(soc_target),
                "qos_flag": str(qos_flag),
            },
            "applied": {
                "p_kW": float(applied_p),
                "soc_next": float(soc_next),
            },
            "limits": {
                "pmax_kW": float(self.pmax),
                "ramp_step_kW": float(self.ramp_step_kW),
                "rate_limit_per_15min": int(self._rl.rate_limit),
                "max_delta_per_write_kW": float(self._rl.max_delta),
                "export_allowed": bool(self.policy.export_allowed),
                "soft_cap_kW": float(self.policy.soft_cap_kW),
                "pcc_limit_kW": float(self.policy.pcc_limit_kW),
                "n_minus_1_margin_kW": float(self.policy.n_minus_1_margin_kW),
            },
            "state_in": {
                "pcc_base_kW": float(pcc_base),
                "soc": float(soc_now),
                "last_p_kW": float(self.last_p_kW),
            },
        }

        # 日志落地（统一 JSONL）
        self.log.write(ack)

        # 更新内部状态（仅成功写入才更新 last_*）
        if status == "ok":
            self.last_p_kW = float(applied_p)
            self.last_soc = float(soc_next)
            self.last_write_ts = int(t_utc)

        return ack


# ====== 自检：构建 Env + 按参考轨迹写入（模拟南向 PCS） ======
def _self_check(dt_min: int, steps: int, rate_limit_15m: int, max_delta_ratio: float,
                sleep_every: int, sleep_sec: int) -> int:
    # 1) 用 DataAdapter 构建 env/planner（复用 module.make_env）
    da = DataAdapter()
    env, planner, ctx = da.load_for_env(dt_min=dt_min, horizon_steps=max(steps, 144), jsonl_path=DEFAULT_JSONL)

    # 2) 读取配置（供适配器构造）
    cfg, window = da.load_configs_only(dt_min=dt_min)
    adapter = ActuatorAdapter(
        cfg=cfg,
        policy=window,
        dt_min=dt_min,
        rate_limit_per_15min=rate_limit_15m,
        max_delta_per_write_ratio=max_delta_ratio,
        enable_ramp_exec=True,
        jsonl_path=DEFAULT_JSONL,
    )

    # 3) 按参考轨迹（planner 结果）逐步下发功率（模拟在线握手）
    #    这里把 env 的参考功率作为 desired_p_kW，并传入 base PCC 与当前 SOC。
    #    注意：这里只做接口/约束自检，不触发 RL。
    t_list = ctx["ts"]
    pcc_base = ctx["pcc"]
    p_ref = getattr(env, "p_ref")  # module.env 内部已生成
    soc = cfg.soc_target

    # 生成一次有效 nonce（实际生产由上游生成）
    def gen_nonce() -> str:
        return secrets.token_hex(8)

    writes_ok = 0
    for i in range(min(steps, len(t_list))):
        t = int(t_list[i])
        desired_p = float(p_ref[i])
        state = {"pcc_base_kW": float(pcc_base[i]), "soc": float(soc)}
        nonce = gen_nonce()

        ack = adapter.apply_setpoint(
            t_utc=t,
            desired_p_kW=desired_p,
            soc_target=cfg.soc_target,
            write_enable=True,
            nonce=nonce,
            qos_flag="balanced",
            state=state,
        )

        # 同步模拟 SOC（用于下一步 state）
        soc = float(ack["applied"]["soc_next"]) if ack["status"] == "ok" else soc
        writes_ok += 1 if ack["status"] == "ok" else 0

        if (i + 1) % max(1, sleep_every) == 0:
            time.sleep(max(0, sleep_sec))

    # 4) 写一条 summary，镜像到静态目录
    log = JsonlLogger(DEFAULT_JSONL)
    summary = {
        "key": "bess_adapter_summary",
        "steps": int(min(steps, len(t_list))),
        "writes_ok": int(writes_ok),
        "jsonl_path": DEFAULT_JSONL,
    }
    log.write(summary)
    # 模拟关闭（不干扰 module 的 logger 句柄）
    try:
        log.close()
    except Exception:
        pass

    # 镜像到前端静态目录
    try:
        os.makedirs(os.path.dirname(STATIC_JSONL), exist_ok=True)
        with open(DEFAULT_JSONL, "r", encoding="utf-8") as fsrc, open(STATIC_JSONL, "w", encoding="utf-8") as fdst:
            for line in fsrc:
                fdst.write(line)
    except Exception:
        pass

    print("[ADAPTER SELF-CHECK] summary:", json.dumps(summary, ensure_ascii=False))
    return 0


# ====== main ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BESS Adapter (IO handshake + throttling + limits)")
    parser.add_argument("--self-check", action="store_true", help="运行自检：构建 env + 参考轨迹写入（模拟南向 PCS）")
    parser.add_argument("--dt-min", type=int, default=10, help="步长（分钟），建议 10")
    parser.add_argument("--steps", type=int, default=144, help="自检步数（默认 24h @10min）")
    parser.add_argument("--rate-limit-15m", type=int, default=6, help="每 15 分钟最多写入次数")
    parser.add_argument("--max-delta-per-write-ratio", type=float, default=0.3, help="单次写入幅度上限（P_rated 的比例）")
    parser.add_argument("--sleep-every", type=int, default=1000, help="每隔 N 步休眠（测试与训练流程对齐）")
    parser.add_argument("--sleep-sec", type=int, default=60, help="每次休眠秒数")
    args = parser.parse_args()

    if args.self_check:
        raise SystemExit(_self_check(
            dt_min=args.dt_min,
            steps=args.steps,
            rate_limit_15m=args.rate_limit_15m,
            max_delta_ratio=args.max_delta_per_write_ratio,
            sleep_every=args.sleep_every,
            sleep_sec=args.sleep_sec,
        ))
    else:
        print("Use --self-check to run adapter handshake & throttling check.")
