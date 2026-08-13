# -*- coding: utf-8 -*-
"""
HVAC 冷站/末端设定点联动 —— 策略编排器（module）
================================================
目标：
- 提供面向服务的 Python API：plan() / decide() / build_write_jobs()
- 将 MPC 参考、Residual 微调、安全屏蔽、需量窗口对齐 串成可复用编排
- 统一输出 JSONL（与 api.py 一致），写点真正下发留给 adapter.py
- 仅用标准库 + 少量 numpy；CSV/JSON 读取走 api.py 的容错口径

对接关系：
- 被平台服务 import 使用：from app.services.rl_model.hvac_cooling.module import CoolingRLModule
- 也可命令行自检：python -m app.services.rl_model.hvac_cooling.module --self-test
- 读取配置：
  * demand_window_config.json（步长、需量软上限、罚金、权重等）
  * plant_master.json（设定点上下界/爬坡、设备能力等）
- 若缺失配置或数据，将启用工程兜底（不会中断控制链）

输出：
- JSONL 统一写到 artifacts/policy_evaluate_history.jsonl
- 关键键：kind(plan/decision)、plan[]、inputs/reference/residual/proposed/masks/final_action/command_payload
- 新增：write_jobs（供 adapter 真正下发）

注意：
- 本文件不直接“写点”，仅生成 write_jobs 与 command_payload；第三个文件 adapter.py 执行南向写入
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

try:
    import numpy as np  # 少量使用
except Exception:
    class _NP:
        def clip(self, a, a_min, a_max):  # 简化兜底
            return max(a_min, min(a_max, a))
        def mean(self, x): return sum(x)/max(1, len(x))
        def array(self, x): return x
    np = _NP()  # type: ignore

# 复用 api.py 中的实现，避免重复逻辑/口径分叉
from .api import (
    MODULE_NAME, DEFAULT_OUT, STATE_PATH, ARTIFACT_DIR,
    load_data_with_fallback, append_jsonl,
    SetpointPlanner, ResidualPolicy, SafetyShield, EffEstimator, DemandWindow,
    load_state, save_state, to_series, dew_point_C, now_utc_iso,
    build_command_payload
)


class CoolingRLModule:
    """
    编排器：把“计划(MPC参考) -> 残差 -> 安全屏蔽 -> 需量对齐 -> 写点任务”拼装起来。
    供服务层直接 import 调用。
    """
    def __init__(self, data_dir: str = "/mnt/data", out_path: str = DEFAULT_OUT, state_path: str = STATE_PATH):
        self.data_dir = data_dir
        self.out_path = out_path
        self.state_path = state_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 读取数据（带兜底）
        self.data = load_data_with_fallback(self.data_dir)
        self.demand_cfg = self.data.get("demand_cfg", {})
        self.plant_cfg = self.data.get("plant_master", {})

        # 步长（默认 15 分钟，对齐需量口径；可被配置覆盖）
        self.step_min = int(self.demand_cfg.get("granularity_min", 15))

        # 效率估计（有表优先，无表回经验式）
        self.eff_est = EffEstimator(self.data.get("effmap", []))

        # 设定点规划器 + 残差策略 + 安全屏蔽
        self.planner = SetpointPlanner(self.plant_cfg, self.demand_cfg, self.eff_est)
        self.residual = ResidualPolicy(self.planner.delta)
        self.shield = SafetyShield(self.plant_cfg, self.demand_cfg)

    # ---------- 工具 ----------
    def _series_inputs(self):
        price_series = to_series(self.data.get("price", []), "ts", ["price_yuan_per_kwh"])
        ef_series    = to_series(self.data.get("ef", []),    "ts", ["ef_kg_per_kwh"])
        load_series  = to_series(self.data.get("loadf", []), "ts", ["load_kw"])
        weather_series = to_series(self.data.get("weather", []), "ts", ["DB_C", "WB_C", "RH_pct"])
        tel_series     = to_series(self.data.get("telemetry", []), "ts", ["PCC_kW"])
        return price_series, ef_series, load_series, weather_series, tel_series

    def _compute_demand_context(self, state: Dict[str, Any], pcc_kw_now: float, now_ts: datetime) -> Dict[str, Any]:
        # 需量窗口/罚金参数
        soft_cap = float(self.demand_cfg.get("soft_cap_kW",
                         self.demand_cfg.get("limits", {}).get("plant_soft_cap_kw", 12500.0)))
        penalty = float(self.demand_cfg.get("penalty_yuan_per_kW", 80.0))
        window_min = int(self.demand_cfg.get("granularity_min", 15))

        # recent_kw 更新（跨回合持久化）
        recent = state.get("recent_kw", [])
        def _parse_iso(ts_s: str) -> datetime:
            try:
                return datetime.strptime(ts_s, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return now_ts - timedelta(minutes=60)
        recent = [(_parse_iso(ts), float(kw)) for ts, kw in recent if isinstance(ts, str)]
        recent = [(ts, kw) for ts, kw in recent if (now_ts - ts).total_seconds() <= 3600]
        recent.append((now_ts, float(pcc_kw_now)))
        state["recent_kw"] = [(ts.strftime("%Y-%m-%dT%H:%M:%S"), kw) for ts, kw in recent]

        dw = DemandWindow(soft_cap, penalty, window_min)
        p_roll = dw.rolling_avg_kw(recent)
        tight = (p_roll >= soft_cap * 0.98)  # 需量紧张阈值

        return {
            "p_roll_kw": float(p_roll),
            "p_cap_kw": float(soft_cap),
            "demand_tight": bool(tight),
            "soft_penalty_yuan": float(dw.soft_penalty(p_roll)),
        }

    # ---------- 业务主流程 ----------
    def plan(self, start_ts: datetime | None = None) -> Dict[str, Any]:
        """生成 24h 参考轨迹（MPC 启发式），并写入 JSONL 记录"""
        price_series, ef_series, load_series, weather_series, _ = self._series_inputs()
        start = start_ts or datetime.utcnow()

        state = load_state(self.state_path)
        last_targets = state.get("last_targets", {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0})
        plan = self.planner.plan_24h(start, self.step_min, price_series, ef_series, load_series, weather_series, last_targets)

        rec = {
            "ts": now_utc_iso(),
            "module": MODULE_NAME,
            "kind": "plan",
            "step_min": self.step_min,
            "plan_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "plan": plan,
            "source_files": self.data.get("paths", {}),
            "audit": {"version": 1, "from": "module.plan"}
        }
        append_jsonl(self.out_path, rec)
        return rec

    def decide(self) -> Dict[str, Any]:
        """执行当前一步决策：参考 -> 残差 -> 屏蔽 -> 需量对齐 -> 写点载荷/任务"""
        price_series, ef_series, load_series, weather_series, tel_series = self._series_inputs()
        now_ts = datetime.utcnow()

        # 当前传感
        pcc_kw_now = float(tel_series[-1][1]) if tel_series else 0.0
        if weather_series:
            db, wb, rh = weather_series[-1][1]
        else:
            db, wb, rh = 30.0, 25.0, 70.0
        dp = dew_point_C(db, rh)

        # 状态与需量上下文
        state = load_state(self.state_path)
        dem_ctx = self._compute_demand_context(state, pcc_kw_now, now_ts)

        # 参考轨迹第一点（若无计划则即时生成）
        eff = self.eff_est
        plan_now = self.planner.plan_24h(now_ts, self.step_min, price_series, ef_series, load_series, weather_series,
                                         state.get("last_targets", {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0}))
        ref0 = plan_now[0] if plan_now else {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0,
                                             "price": 0.8, "ef": 0.7, "db_C": db, "rh_pct": rh}

        # 残差（DR/需量紧张时自适应减半，api.ResidualPolicy 已含逻辑）
        ctx = {
            "ref": ref0,
            "price": ref0.get("price", 0.8),
            "ef": ref0.get("ef", 0.7),
            "db_C": db,
            "rh_pct": rh,
            "dr_mode": False,                    # 预留 DR 信号注入
            "demand_tight": dem_ctx["demand_tight"],
        }
        telemetry_rows = self.data.get("telemetry") or []
        if telemetry_rows:
            ctx["state_features"] = telemetry_rows[-1]
        d = self.residual.decide(ctx)

        proposed = {
            "CHWS_set": float(np.clip(ref0["CHWS_set"] + d["dCHWS"], self.planner.min_chws, self.planner.max_chws)),
            "SAT_set":  float(np.clip(ref0["SAT_set"] + d["dSAT"],   self.planner.min_sat,  self.planner.max_sat)),
            "SP_set":   float(np.clip(ref0["SP_set"]  + d["dSP"],    self.planner.min_sp,   self.planner.max_sp)),
        }

        # 安全屏蔽（露点/最小流量/需量紧张/速率限制）
        safety_ctx = {
            "dew_point_C": dp,
            "CHW_flow": 0.0,  # 若 telemetry 含 CHW_flow，可在 adapter 注入更准确值
            "G_min": 0.3,     # 额定 30% 兜底
            "demand_tight": dem_ctx["demand_tight"]
        }
        last_targets = state.get("last_targets", {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0})
        final_targets, masks = self.shield.apply(last_targets, proposed, safety_ctx)

        # BAS 写点载荷（不直接下发）
        cmd_payload = build_command_payload(final_targets, ttl_s=60)
        write_jobs = self.build_write_jobs(final_targets)

        # 更新状态（跨回合）
        state["last_targets"] = final_targets
        save_state(self.state_path, state)

        rec = {
            "ts": now_utc_iso(),
            "module": MODULE_NAME,
            "kind": "decision",
            "inputs": {
                "pcc_kw_now": pcc_kw_now,
                "db_C": db, "wb_C": wb, "rh_pct": rh, "dew_point_C": round(dp, 2),
                "demand_window": dem_ctx
            },
            "reference": ref0,
            "residual": d,
            "proposed": proposed,
            "masks": masks,
            "final_action": final_targets,
            "command_payload": cmd_payload,      # 轻量聚合（BAS PID handoff）
            "write_jobs": write_jobs,            # adapter 会消费这个数组真正下发
            "source_files": self.data.get("paths", {}),
            "audit": {"version": 3, "from": "module.decide", "rl_backend": self.residual.backend, "inference": self.residual.last_audit}
        }
        append_jsonl(self.out_path, rec)
        return rec

    # ---------- 写点任务编排 ----------
    def build_write_jobs(self, final_targets: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        生成“写点任务”数组，由 adapter.py 执行南向写入：
        - 每个任务 = {point, value, ttl_s, nonce, limits, priority, topic}
        - 附带速率/步进限制（由 plant/demand 配置推导）
        - 统一口径：BAS 侧 PID 执行，AI 设定点只给 set_cmd
        """
        # ramp 限制读取（按 15min 口径）
        ramp = self.demand_cfg.get("ramp_limits", {})
        chws_max_delta = float(ramp.get("chws_C_per_15min",
                          self.plant_cfg.get("setpoints", {}).get("chws_C", {}).get("ramp_C_per_15min", 0.5)))
        sat_max_delta = float(ramp.get("sat_C_per_15min",
                         self.plant_cfg.get("setpoints", {}).get("sat_C", {}).get("ramp_C_per_15min", 0.6)))
        sp_max_delta  = float(self.plant_cfg.get("setpoints", {}).get("static_pressure_Pa", {}).get("ramp_Pa_per_15min", 50))

        ttl_s = 60
        nonce = self._gen_nonce()

        jobs = [
            {
                "point": "CHWS_set_cmd",
                "value": float(final_targets["CHWS_set"]),
                "ttl_s": ttl_s,
                "nonce": nonce,
                "limits": {"max_delta_per_write": chws_max_delta, "rate_limit_per_15min": 1},
                "priority": "energy_optimized",
                "topic": "BAS/HVAC/ChillerPlant"
            },
            {
                "point": "SAT_set_cmd",
                "value": float(final_targets["SAT_set"]),
                "ttl_s": ttl_s,
                "nonce": nonce,
                "limits": {"max_delta_per_write": sat_max_delta, "rate_limit_per_15min": 1},
                "priority": "energy_optimized",
                "topic": "BAS/HVAC/AHU"
            },
            {
                "point": "SP_set_cmd",
                "value": float(final_targets["SP_set"]),
                "ttl_s": ttl_s,
                "nonce": nonce,
                "limits": {"max_delta_per_write": sp_max_delta, "rate_limit_per_15min": 1},
                "priority": "energy_optimized",
                "topic": "BAS/HVAC/AHU"
            },
        ]
        return jobs

    @staticmethod
    def _gen_nonce() -> str:
        # 简化版 nonce；adapter 会在南向通道再做签名/过期校验
        import uuid, time, hashlib
        raw = f"{uuid.uuid4()}-{time.time()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------- CLI：便于一键自检 / 调用 ----------------
def main():
    parser = argparse.ArgumentParser(description="HVAC cooling orchestrator (module)")
    parser.add_argument("--data-dir", type=str, default="/mnt/data", help="数据目录（默认 /mnt/data）")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="JSONL 输出文件（默认 artifacts/policy_evaluate_history.jsonl）")
    parser.add_argument("--plan", action="store_true", help="只生成 24h 参考轨迹")
    parser.add_argument("--decide", action="store_true", help="只执行当前一步决策")
    parser.add_argument("--self-test", action="store_true", help="自检（plan + decide）")
    args = parser.parse_args()

    mod = CoolingRLModule(data_dir=args.data_dir, out_path=args.out, state_path=STATE_PATH)

    try:
        if args.self_test:
            rec_plan = mod.plan()
            rec_dec  = mod.decide()
            print("SELF-TEST OK:", json.dumps({
                "plan_len": len(rec_plan.get("plan", [])),
                "write_jobs": len(rec_dec.get("write_jobs", [])),
                "final": rec_dec.get("final_action", {})
            }, ensure_ascii=False))
            return 0
        if args.plan:
            rec = mod.plan()
            print("PLAN OK:", len(rec.get("plan", [])))
            return 0
        if args.decide:
            rec = mod.decide()
            print("DECIDE OK:", json.dumps({"final": rec.get("final_action", {}), "jobs": rec.get("write_jobs", [])}, ensure_ascii=False))
            return 0
        # 默认：执行一次决策
        rec = mod.decide()
        print("DECIDE OK:", json.dumps({"final": rec.get("final_action", {}), "jobs": rec.get("write_jobs", [])}, ensure_ascii=False))
        return 0
    except Exception as e:
        print("ERROR:", repr(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
