# app/services/rl_model/bess_energy/module.py
# -*- coding: utf-8 -*-
"""
E 模块｜BESS 储能 削峰/套利/备用 的 CMDP 环境 + 规则/MPC 兜底 + 动作屏蔽 + JSONL 审计
========================================================================================
大白话说明：
- 这是“环境 + 兜底 + 基线”的一体化实现，训练引擎（下一步的 rl_engine.py）会直接掉这里的 Env。
- 只用标准库 csv/json + 少量 numpy；内置统一列名候选、时间戳多口径识别、容错读取、读失败兜底经验曲线。
- 环境对外暴露：make_env(...) / BessCmdpEnv / EconomicMPCPlanner / rollout_and_log(...) 等接口。
- 奖励：按“收益为正”的签名：备用/DR 收入 - 能源成本 - 碳成本 - 需量/履约罚 - 退化 - 平滑惩罚 - 终态 SOC 偏差。
- 硬约束：SOC/Pmax/C-rate/ramp/PCC/N-1/逆潮流/备用可用性；违反动作一律屏蔽，记录屏蔽原因与解除 ETA。
- 兜底：若 RL 残差被屏蔽或事件跟踪误差连续越界，回退到经济 MPC 的参考轨迹。
- 输出：每步写一行 JSONL（policy_evaluate_history.jsonl），包含：时戳、观测、动作（前/后屏蔽）、奖惩分解、与基线对比、理由等。

落地接口约定（稳定对外）：
- make_env(cfg_paths, dt_min, horizon_steps, log_jsonl) -> env, baseline (planner)
- env.reset(start_index) / env.step(action_dict)  # action_dict={"dP":(kW residual), "dR":(kW), "mode":str}
- rollout_and_log(env, policy_fn, baseline_policy_fn, ...): 统一产出 JSONL 日志 + 统计 KPI
- prepare_offline_dataset(...): 基于兜底/规则轨迹与历史测点，生成 IQL 训练的 transitions JSONL

与其他文件的关系：
- 被谁调用：下一步要交付的 `rl_engine.py`（训练与在线微调）会 import 本文件，并调用 make_env/rollout 等。
- 调用谁：本文件内部只用标准库 + numpy；读取 /mnt/data 与模块 data/ 下的数据与配置。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np


# ------------------------------
# 路径与常量
# ------------------------------

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_CANDIDATES = [
    # 优先使用挂载目录（便于在容器/训练机批量替换）
    "/mnt/data",
    # 回退到仓库内置样例数据
    os.path.join(MODULE_DIR, "data"),
]
DEFAULT_JSONL = os.path.join(MODULE_DIR, "policy_evaluate_history.jsonl")
STATIC_JSONL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(MODULE_DIR))),
    "static",
    "api",
    "rl",
    "artifacts",
    "policy_evaluate_history.jsonl",
)

# 统一列名候选集（不区分大小写，取第一个命中）
COLUMN_CANDIDATES = {
    "ts": ["ts", "timestamp", "time", "datetime", "utc", "time_utc", "t"],
    "pcc_kw": ["pcc_kw", "p_grid_kw", "pcc", "grid_kw", "p_kw", "p_import_kw"],
    "price_yuan_per_kwh": ["price", "price_yuan_per_kwh", "p_yuan_kwh", "rt_price", "da_price"],
    "ef_kg_per_kwh": ["ef", "ef_kg_per_kwh", "grid_ef", "co2_kg_per_kwh"],
    "soc": ["soc", "state_of_charge", "soc_frac", "soc_pct"],
    "p_bess_kw": ["p_bess_kw", "p_bess", "bess_kw"],
    "t_batt_c": ["t_batt_c", "temp_c", "temperature_c"],
    "event_type": ["event_type", "type"],
    "event_start": ["start_ts", "start", "begin", "start_time"],
    "event_end": ["end_ts", "end", "stop", "end_time"],
    "event_target_kw": ["target_kw", "kW", "target", "r_commit_kw"],
}

# 环境默认参数（若配置/数据缺失时的兜底）
DEFAULTS = {
    "eff_ch": 0.965,
    "eff_dis": 0.965,
    "soc_min": 0.15,
    "soc_max": 0.95,
    "soc_target": 0.7,
    "c_rate_max": 0.8,  # C 倍率上限（若配置未提供）
    "ramp_kW_per_step": 0.25,  # 以 P_rated 的比例表达；加载后换算为 kW/步
    "cycle_cost_yuan_per_kWh": 0.08,
    "export_allowed": False,
    "p_co2_yuan_per_kg": 0.0,  # 默认不计价，只统计
    "residual_band_ratio": 0.12,  # RL 残差包络 ±12% Pmax
    "alpha_peak": 95.0,  # 需量罚金梯度（元/kW，若未从政策文件给出则用此）
    "beta_reserve": 200.0,  # 备用违约罚金梯度（元/kW）
    "gamma_smooth": 3e-4,  # 平滑惩罚（元/|ΔP|）
    "zeta_soc_end": 40.0,  # 终态 SOC 偏差惩罚（元/ΔSOC）
    "epsilon_softcap": 200.0,  # 靠近软限的提前量（kW）
}


# ------------------------------
# 工具：时间戳解析与容错 CSV 读取
# ------------------------------

def _parse_ts_any(s: str) -> Optional[int]:
    """多口径时间戳解析：ISO8601 / 带时区 / epoch 秒/毫秒。失败返回 None。"""
    if s is None:
        return None
    s = str(s).strip().replace("T", " ").replace("Z", "+00:00")
    # epoch 数字
    try:
        if s.isdigit():
            val = int(s)
            if val > 10**12:  # 纳秒/微秒/毫秒级，粗略修正
                while val > 10**10:
                    val = val // 1000
            return val
    except Exception:
        pass
    # 解析 ISO8601 / 常见格式
    fmts = [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M%z",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S%z",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M%z",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            if "%z" in f:
                dt = datetime.strptime(s, f)
            else:
                dt = datetime.strptime(s, f).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            continue
    return None


def _find_first_matching(headers: List[str], candidates: List[str]) -> Optional[int]:
    """在 headers 里找出首个命中候选列的索引（不区分大小写）。"""
    low = [h.strip().lower() for h in headers]
    for c in candidates:
        if c.lower() in low:
            return low.index(c.lower())
    return None


def _read_csv_timeseries(path: str, need: List[str]) -> List[Dict[str, Any]]:
    """
    容错读取 CSV：按 need 字段（如 ["ts","pcc_kw"]）和候选列名定位。
    返回按 ts 升序的 [{ts:int, field:float,...}] 列表；非法/空值自动跳过。
    """
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return rows
        idx_map: Dict[str, int] = {}
        for key in need:
            cand = COLUMN_CANDIDATES.get(key, [key])
            idx = _find_first_matching(headers, cand)
            if idx is not None:
                idx_map[key] = idx

        # ts 必须能解析，否则跳过整行
        for line in reader:
            try:
                ts_raw = line[idx_map["ts"]] if "ts" in idx_map else None
                ts = _parse_ts_any(ts_raw)
                if ts is None:
                    continue
                d: Dict[str, Any] = {"ts": ts}
                ok = True
                for key in need:
                    if key == "ts":
                        continue
                    idx = idx_map.get(key)
                    if idx is None:
                        ok = False
                        break
                    v = line[idx].strip()
                    if v == "":
                        ok = False
                        break
                    # 百分比 SOC 自适应
                    if key == "soc" and "%" in v:
                        v = v.replace("%", "")
                        val = float(v) / 100.0
                    else:
                        val = float(v)
                    d[key] = val
                if ok:
                    rows.append(d)
            except Exception:
                # 单条坏数据跳过，继续
                continue

    rows.sort(key=lambda x: x["ts"])
    return rows


def _resample_to_grid(series: List[Dict[str, Any]], dt_min: int, fields: List[str]) -> List[Dict[str, Any]]:
    """
    将不等间隔/不同步的序列按 dt_min 对齐到统一栅格（线性插值，端点前后持有）。
    输入：每个元素包含 ts 及 fields。
    """
    if not series:
        return []

    ts0 = series[0]["ts"]
    ts1 = series[-1]["ts"]
    step = dt_min * 60
    grid_ts = list(range(ts0 - (ts0 % step), ts1 - (ts1 % step) + step, step))

    # 为每个字段单独做 1D 线性插值
    out = []
    # 先按字段抽出 (ts, val)
    ts_arr = np.array([r["ts"] for r in series], dtype=np.int64)
    vals_by_field: Dict[str, np.ndarray] = {}
    for fld in fields:
        vals_by_field[fld] = np.array([r.get(fld, np.nan) for r in series], dtype=np.float64)

    for t in grid_ts:
        row = {"ts": t}
        for fld in fields:
            v = _interp_1d(ts_arr, vals_by_field[fld], t)
            row[fld] = v
        out.append(row)
    return out


def _interp_1d(xs: np.ndarray, ys: np.ndarray, x: int) -> float:
    """简单 1D 线性插值，超界时使用端点值。"""
    if len(xs) == 0:
        return float("nan")
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    # 二分定位
    lo, hi = 0, len(xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    if x1 == x0:
        return float(y0)
    r = (x - x0) / (x1 - x0)
    return float(y0 + r * (y1 - y0))


# ------------------------------
# 配置数据结构
# ------------------------------

@dataclass
class DemandWindowPolicy:
    timezone: str
    soft_cap_kW: float
    pcc_limit_kW: float
    n_minus_1_margin_kW: float
    penalty_yuan_per_kW: float
    export_allowed: bool

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "DemandWindowPolicy":
        # 有的文件字段名大小写/复写，这里统一兜底
        tz = d.get("timezone", "Asia/Shanghai")
        soft_cap = float(d.get("soft_cap_kW") or d.get("plant_soft_cap_kw") or 1e12)
        pcc_limit = float(d.get("pcc_limit_kW") or d.get("pcc_limit_kw") or 1e12)
        n1 = float(d.get("n_minus_1_margin_kW") or d.get("n_minus_1_buffer_kw") or 0.0)
        penalty = float(d.get("penalty_yuan_per_kW") or DEFAULTS["alpha_peak"])
        export_allowed = bool(d.get("penalties", {}).get("export_allowed", False) or d.get("export_allowed", False))
        return DemandWindowPolicy(
            timezone=tz,
            soft_cap_kW=soft_cap,
            pcc_limit_kW=pcc_limit,
            n_minus_1_margin_kW=n1,
            penalty_yuan_per_kW=penalty,
            export_allowed=export_allowed,
        )


@dataclass
class BessSiteConfig:
    site_id: str
    rated_power_kW: float
    rated_energy_kWh: float
    eff_ch: float
    eff_dis: float
    soc_min: float
    soc_max: float
    soc_target: float
    p_ramp_kW_per_step: float
    export_allowed: bool
    cycle_cost_yuan_per_kWh: float
    c_rate_max: float

    reserve_min_kW: float
    reserve_critical_hours_local: List[int]

    telemetry_sources: Dict[str, str]

    @staticmethod
    def from_json(d: Dict[str, Any], dt_min: int) -> "BessSiteConfig":
        rp = float(d.get("rated_power_kW", 0.0))
        re = float(d.get("rated_energy_kWh", 0.0))
        eff_ch = float(d.get("eff_ch", DEFAULTS["eff_ch"]))
        eff_dis = float(d.get("eff_dis", DEFAULTS["eff_dis"]))
        soc_min = float(d.get("soc_min", DEFAULTS["soc_min"]))
        soc_max = float(d.get("soc_max", DEFAULTS["soc_max"]))
        soc_target = float(d.get("soc_target", DEFAULTS["soc_target"]))
        # ramp：配置可给 kW/step 或绝对 kW/15min；这里统一按步
        if "p_ramp_kW_per_step" in d:
            ramp_step = float(d["p_ramp_kW_per_step"])
        else:
            # 若只给了 15 分钟斜坡能力，换算成每步（dt_min）
            per_15 = float(d.get("ramp_limits", {}).get("bess_kw_per_15min", rp * DEFAULTS["ramp_kW_per_step"]))
            ramp_step = per_15 * (dt_min / 15.0)
        export_allowed = bool(d.get("export_allowed", DEFAULTS["export_allowed"]))
        cycle_cost = float(d.get("cycle_cost_yuan_per_kWh", DEFAULTS["cycle_cost_yuan_per_kWh"]))
        c_rate_max = float(d.get("c_rate_max", DEFAULTS["c_rate_max"]))

        # 备用
        res_rule = d.get("reserve_rules", {})
        reserve_min = float(res_rule.get("min_reserve_kW", 0.0))
        critical_hours = list(res_rule.get("critical_hours_local", []))

        t_src = d.get("telemetry_sources", {})
        return BessSiteConfig(
            site_id=d.get("site_id", "BESS_SITE"),
            rated_power_kW=rp,
            rated_energy_kWh=re,
            eff_ch=eff_ch,
            eff_dis=eff_dis,
            soc_min=soc_min,
            soc_max=soc_max,
            soc_target=soc_target,
            p_ramp_kW_per_step=ramp_step,
            export_allowed=export_allowed,
            cycle_cost_yuan_per_kWh=cycle_cost,
            c_rate_max=c_rate_max,
            reserve_min_kW=reserve_min,
            reserve_critical_hours_local=critical_hours,
            telemetry_sources=t_src,
        )


# ------------------------------
# JSONL 记录器（前端消费）
# ------------------------------

class JsonlLogger:
    """统一 JSONL 日志：每行一条 dict；写失败自动降级为 stdout。"""

    def __init__(self, path: str = DEFAULT_JSONL):
        self.path = path
        self._fh = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        try:
            if self._fh is None:
                self._fh = open(self.path, "w", encoding="utf-8")
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            # 最后兜底：打印到 stdout
            print(json.dumps(record, ensure_ascii=False))

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def _mirror_to_static(src: str, dst: str = STATIC_JSONL) -> None:
    """将 JSONL 拷贝一份到静态目录供前端页面读取（若目录存在则镜像）。"""
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
            for line in fsrc:
                fdst.write(line)
    except Exception:
        pass


# ------------------------------
# 经济 MPC（启发式）参考轨迹生成
# ------------------------------

class EconomicMPCPlanner:
    """
    经济 MPC 简式（无 MILP 求解器，工程启发式可落地）：
    - 目标：削峰（需量软限 soft_cap）、套利（低价充/高价放、碳强度约束）、备用预留（事件/关键时段）。
    - 方法：滚动窗口 H 内，先满足安全+备用，再在剩余功率/能量里做套利；对 15 分钟需量按 rolling mean 约束。
    """

    def __init__(self, cfg: BessSiteConfig, policy: DemandWindowPolicy, dt_min: int = 10, horizon_steps: int = 144):
        self.cfg = cfg
        self.policy = policy
        self.dt_min = dt_min
        self.h = horizon_steps
        self.step_sec = dt_min * 60
        self.p_c_rate_cap = cfg.c_rate_max * cfg.rated_energy_kWh  # kW（E[kWh] * C[1/h]）
        self.p_max_hw = cfg.rated_power_kW
        self.p_max = min(self.p_c_rate_cap, self.p_max_hw)

        # 残差包络（供 RL 使用）
        self.residual_band = DEFAULTS["residual_band_ratio"] * self.p_max

    def plan(self,
             ts: List[int],
             pcc_base_kw: List[float],
             price_yuan_per_kwh: List[float],
             ef_kg_per_kwh: List[float],
             soc0: float,
             reserve_events: List[Tuple[int, int, float]],
             dr_events: List[Tuple[int, int, float]]) -> Dict[str, np.ndarray]:
        """
        生成参考轨迹（不考虑 RL 残差），返回 dict：
        - "p_ref": 参考 P_bess[kW]，放电为 +，充电为 -
        - "r_res": 备用预留[kW]
        - "soc":   SOC 轨迹（含初值）
        - "p_pcc": 叠加储能后的 PCC 预测
        """
        n = len(ts)
        p_ref = np.zeros(n, dtype=np.float64)
        r_res = np.zeros(n, dtype=np.float64)
        soc = np.zeros(n + 1, dtype=np.float64)
        soc[0] = float(soc0)

        # 价格分位：削峰窗口不用价差，套利窗口用 P20/P80
        prices = np.array(price_yuan_per_kwh, dtype=np.float64)
        p20 = np.nanpercentile(prices, 20)
        p80 = np.nanpercentile(prices, 80)

        # 碳因子阈值：高 EF 时少充（降低）
        efs = np.array(ef_kg_per_kwh, dtype=np.float64)
        ef80 = np.nanpercentile(efs, 80)

        # 事件与关键时段的备用预留
        for i, t in enumerate(ts):
            # 关键小时（本地时区）备用：按最小预留
            hour_local = datetime.fromtimestamp(t, tz=timezone.utc).astimezone().hour
            if hour_local in self.cfg.reserve_critical_hours_local and self.cfg.reserve_min_kW > 0:
                r_res[i] = max(r_res[i], self.cfg.reserve_min_kW)
            # 事件窗口：按事件目标预留
            for (s, e, tgt) in reserve_events:
                if s <= t < e:
                    r_res[i] = max(r_res[i], tgt)
            for (s, e, tgt) in dr_events:
                if s <= t < e:
                    # DR 等同于“事件跟踪”能力，按目标预留，不重复加
                    r_res[i] = max(r_res[i], tgt)

        # 主循环：先削峰，再套利（残余能力），实时确保硬约束
        # 为滚动均值（15min）准备状态
        roll_window_steps = max(1, int(round(15 / self.dt_min)))
        pcc_roll = [pcc_base_kw[0]] * roll_window_steps

        for i in range(n):
            t = ts[i]
            pcc0 = float(pcc_base_kw[i])
            price = float(price_yuan_per_kwh[i])
            ef = float(ef_kg_per_kwh[i])
            soc_i = float(soc[i])

            # 1) 削峰：若 rolling_mean_15 接近 soft_cap，优先放电，避免越 cap
            roll_mean15 = np.mean(pcc_roll[-roll_window_steps:])
            safety_margin = DEFAULTS["epsilon_softcap"]
            need_discharge = roll_mean15 > (self.policy.soft_cap_kW - safety_margin)

            p_cmd = 0.0
            if need_discharge:
                # 理论需要降低的功率
                overshoot = roll_mean15 - (self.policy.soft_cap_kW - safety_margin)
                p_cmd = min(self.p_max, overshoot)
            else:
                # 2) 套利：低价且低 EF 充电；高价放电（不打破备用预留与 SOC 界）
                if price <= p20 and ef <= ef80:
                    p_cmd = -min(self.p_max, self.p_max_hw)  # 充电（负号）
                elif price >= p80:
                    p_cmd = +min(self.p_max, self.p_max_hw)
                else:
                    p_cmd = 0.0

            # 3) 备用预留优先：限制可用功率（若需留出 upwards reserve，就限制放电；downwards reserve 限制充电）
            # 简化处理：统一按“上行备用”为主，确保剩余出力 >= r_res[i]
            p_avail_up = min(self.p_max,  # 硬件功率
                             (soc_i - self.cfg.soc_min) * self.cfg.rated_energy_kWh / (self.dt_min / 60.0))  # SOC 可释放
            if r_res[i] > 0:
                # 预留一部分上行能力，放电指令不超过 (p_avail_up - r_res)
                p_cmd = min(p_cmd, max(0.0, p_avail_up - r_res[i]))

            # 4) 并网/逆潮流限制：若禁止反送，确保 P_pcc >= 0
            pcc_if = pcc0 + (max(0.0, -p_cmd) - max(0.0, p_cmd))  # 充电增加网侧，放电减少
            if not self.policy.export_allowed and pcc_if < 0.0:
                # 将充电削减到恰好不逆潮流
                # pcc_if = pcc0 + ch - dis >= 0
                # 如果 p_cmd < 0（充电），则 ch = -p_cmd；否则 dis = p_cmd
                if p_cmd < 0:
                    max_ch = max(0.0, -pcc0)  # 最大允许充电使得 pcc 不为负
                    p_cmd = -max_ch
                # 若 p_cmd > 0（放电）也可能逆潮流（当 pcc0 很小），需限制放电
                if p_cmd > 0 and pcc_if < 0:
                    p_cmd = min(p_cmd, pcc0)

            # 5) C-rate 与 Ramp 限制（相对上一时刻参考功率）
            if i > 0:
                p_cmd = np.clip(p_cmd, p_ref[i - 1] - self.cfg.p_ramp_kW_per_step, p_ref[i - 1] + self.cfg.p_ramp_kW_per_step)
            p_cmd = float(np.clip(p_cmd, -self.p_max, self.p_max))

            # 6) 更新 SOC 与 rolling_mean
            e_ch = max(0.0, -p_cmd) * self.cfg.eff_ch * (self.dt_min / 60.0)  # kWh
            e_dis = max(0.0, p_cmd) / self.cfg.eff_dis * (self.dt_min / 60.0)
            soc_next = soc_i + (e_ch - e_dis) / self.cfg.rated_energy_kWh
            # SOC 硬边界
            if soc_next > self.cfg.soc_max:
                # 超出则减少充电或减少放电回充（此处 p_cmd<0 时生效）
                over = (soc_next - self.cfg.soc_max) * self.cfg.rated_energy_kWh / (self.dt_min / 60.0)
                p_cmd += over / self.cfg.eff_ch  # 充电减少（p_cmd 更接近 0）
            if soc_next < self.cfg.soc_min:
                # 超出则减少放电
                under = (self.cfg.soc_min - soc_next) * self.cfg.rated_energy_kWh / (self.dt_min / 60.0)
                p_cmd -= under * self.cfg.eff_dis  # 放电减少（p_cmd 更接近 0）

            # 最终裁剪
            p_cmd = float(np.clip(p_cmd, -self.p_max, self.p_max))
            p_ref[i] = p_cmd
            # 更新滚动 PCC（用于下一步判断）
            ch = max(0.0, -p_cmd)
            dis = max(0.0, p_cmd)
            pcc_new = pcc0 + ch - dis
            pcc_roll.append(pcc_new)
            if len(pcc_roll) > roll_window_steps:
                pcc_roll.pop(0)

            # 精确更新 SOC
            e_ch = ch * self.cfg.eff_ch * (self.dt_min / 60.0)
            e_dis = dis / self.cfg.eff_dis * (self.dt_min / 60.0)
            soc[i + 1] = np.clip(soc[i] + (e_ch - e_dis) / self.cfg.rated_energy_kWh, self.cfg.soc_min, self.cfg.soc_max)

        # 生成 p_pcc
        p_pcc = np.array(pcc_base_kw, dtype=np.float64) + np.maximum(0, -p_ref) - np.maximum(0, p_ref)

        return {
            "p_ref": p_ref,
            "r_res": r_res,
            "soc": soc,
            "p_pcc": p_pcc,
        }


# ------------------------------
# CMDP 环境（可安全残差 RL）
# ------------------------------

class BessCmdpEnv:
    """
    CMDP 环境（含动作屏蔽 + 约束优先级 + 回退 MPC）：
    - 连续动作：{"dP": ΔP_residual_kW, "dR": Δreserve_kW, "mode": 可选字符串}
    - 硬屏蔽规则：SOC/Pmax/C-rate/Ramp/PCC/逆潮流/N-1/备用优先；事件期优先跟踪。
    - 观测：当前 SOC/PCC/价格/EF/事件标志 + 未来 H 步价/EF P50/P90 摘要 + 最近 k 步动作历史。
    """

    def __init__(self,
                 cfg: BessSiteConfig,
                 policy: DemandWindowPolicy,
                 dt_min: int,
                 ts: List[int],
                 pcc_base_kw: List[float],
                 price_yuan_per_kwh: List[float],
                 ef_kg_per_kwh: List[float],
                 reserve_events: List[Tuple[int, int, float]],
                 dr_events: List[Tuple[int, int, float]],
                 planner: EconomicMPCPlanner,
                 log: JsonlLogger,
                 horizon_steps: int = 144):
        self.cfg = cfg
        self.policy = policy
        self.dt_min = dt_min
        self.ts = ts
        self.pcc_base = np.array(pcc_base_kw, dtype=np.float64)
        self.price = np.array(price_yuan_per_kwh, dtype=np.float64)
        self.ef = np.array(ef_kg_per_kwh, dtype=np.float64)
        self.reserve_events = reserve_events
        self.dr_events = dr_events
        self.planner = planner
        self.log = log
        self.h = horizon_steps

        # 参考计划
        plan = planner.plan(
            ts=ts,
            pcc_base_kw=pcc_base_kw,
            price_yuan_per_kwh=price_yuan_per_kwh,
            ef_kg_per_kwh=ef_kg_per_kwh,
            soc0=cfg.soc_target,  # 默认以目标 SOC 作为起始
            reserve_events=reserve_events,
            dr_events=dr_events,
        )
        self.p_ref = plan["p_ref"]
        self.r_ref = plan["r_res"]
        self.p_pcc_ref = plan["p_pcc"]

        # 状态缓存
        self.idx = 0
        self.p_prev = 0.0
        self.soc = float(cfg.soc_target)
        self.prev_roll_mean15 = None
        self.event_active = False
        self.k_hist = 6  # 历史步长
        self.hist_actions = [0.0] * self.k_hist
        self.hist_masks = [0] * self.k_hist

        # 度量
        self.metrics = {
            "reward_sum": 0.0,
            "baseline_cost": 0.0,
            "actual_cost": 0.0,
            "energy_cost": 0.0,
            "reserve_pay": 0.0,
            "dr_pay": 0.0,
            "carbon_cost": 0.0,
            "peak_penalty": 0.0,
            "reserve_shortfall": 0.0,
            "degradation_cost": 0.0,
            "smooth_penalty": 0.0,
        }

    # --------- 公用工具 ---------

    def _rolling_mean_15(self, arr: List[float]) -> float:
        steps = max(1, int(round(15 / self.dt_min)))
        if len(arr) < steps:
            return float(np.mean(arr))
        return float(np.mean(arr[-steps:]))

    def _in_event(self, t: int) -> Tuple[bool, float]:
        """返回 (事件进行中, 目标kW)，DR 与 Reserve 取 max。"""
        tgt = 0.0
        active = False
        for (s, e, k) in self.reserve_events:
            if s <= t < e:
                active = True
                tgt = max(tgt, k)
        for (s, e, k) in self.dr_events:
            if s <= t < e:
                active = True
                tgt = max(tgt, k)
        return active, tgt

    # --------- 核心 API ---------

    def reset(self, start_index: int = 0) -> Dict[str, Any]:
        self.idx = start_index
        self.p_prev = float(self.p_ref[max(0, start_index - 1)] if start_index > 0 else 0.0)
        self.soc = float(self.cfg.soc_target)
        self.prev_roll_mean15 = None
        self.event_active = False
        self.hist_actions = [0.0] * self.k_hist
        self.hist_masks = [0] * self.k_hist
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        i = self.idx
        t = self.ts[i]
        # 未来 H 摘要：P50/P90
        j2 = min(len(self.ts), i + self.h)
        fut_price = self.price[i:j2]
        fut_ef = self.ef[i:j2]
        p50_price = float(np.nanpercentile(fut_price, 50)) if len(fut_price) else float(self.price[i])
        p90_price = float(np.nanpercentile(fut_price, 90)) if len(fut_price) else float(self.price[i])
        p50_ef = float(np.nanpercentile(fut_ef, 50)) if len(fut_ef) else float(self.ef[i])
        p90_ef = float(np.nanpercentile(fut_ef, 90)) if len(fut_ef) else float(self.ef[i])

        active, tgt = self._in_event(t)
        obs = {
            "t": t,
            "idx": i,
            "soc": float(self.soc),
            "p_ref": float(self.p_ref[i]),
            "r_ref": float(self.r_ref[i]),
            "p_prev": float(self.p_prev),
            "pcc_base": float(self.pcc_base[i]),
            "price": float(self.price[i]),
            "ef": float(self.ef[i]),
            "event_active": 1 if active else 0,
            "event_target_kw": float(tgt),
            "fut_price_p50": p50_price,
            "fut_price_p90": p90_price,
            "fut_ef_p50": p50_ef,
            "fut_ef_p90": p90_ef,
            "hist_actions": list(self.hist_actions),
            "hist_masks": list(self.hist_masks),
        }
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        输入动作：
        - dP: RL 残差（kW），在 ±planner.residual_band 内建议，但仍将过筛硬约束
        - dR: 备用残差（kW）
        - mode: 可选字符串：["peak","arb","reserve"]，仅改变奖励权重（非硬约束）
        """
        i = self.idx
        t = self.ts[i]

        # 读取基准
        p_base = float(self.p_ref[i])
        r_base = float(self.r_ref[i])

        # 残差建议
        dP = float(action.get("dP", 0.0))
        dR = float(action.get("dR", 0.0))
        mode = str(action.get("mode", ""))

        # 软束缚：残差包络
        dP = float(np.clip(dP, -self.planner.residual_band, self.planner.residual_band))
        dR = float(np.clip(dR, -self.planner.residual_band, self.planner.residual_band))

        p_try = p_base + dP
        r_try = max(0.0, r_base + dR)

        # ---------------- 硬约束屏蔽 ----------------
        mask_reasons: List[str] = []
        masked = 0

        # Ramp
        p_try = float(np.clip(p_try, self.p_prev - self.cfg.p_ramp_kW_per_step, self.p_prev + self.cfg.p_ramp_kW_per_step))
        if abs(p_try - (p_base + dP)) > 1e-6:
            mask_reasons.append("ramp")
            masked = 1

        # C-rate & Pmax
        p_c_rate_cap = self.cfg.c_rate_max * self.cfg.rated_energy_kWh
        p_try_clipped = float(np.clip(p_try, -min(self.cfg.rated_power_kW, p_c_rate_cap), min(self.cfg.rated_power_kW, p_c_rate_cap)))
        if abs(p_try_clipped - p_try) > 1e-6:
            p_try = p_try_clipped
            mask_reasons.append("pmax/c_rate")
            masked = 1

        # 备用优先（上行能力）
        p_avail_up = min(
            min(self.cfg.rated_power_kW, p_c_rate_cap),
            max(0.0, (self.soc - self.cfg.soc_min) * self.cfg.rated_energy_kWh / (self.dt_min / 60.0)),
        )
        if r_try > 0.0:
            # 放电指令不超过 p_avail_up - r_try
            p_try = min(p_try, max(0.0, p_avail_up - r_try))
            mask_reasons.append("reserve_priority")
            masked = 1

        # PCC + 逆潮流限制
        pcc0 = float(self.pcc_base[i])
        pcc_if = pcc0 + max(0.0, -p_try) - max(0.0, p_try)
        if not self.policy.export_allowed and pcc_if < 0.0:
            if p_try < 0:
                p_try = -max(0.0, -pcc0)  # 限充
            if p_try > 0 and (pcc0 - p_try) < 0.0:
                p_try = min(p_try, pcc0)  # 限放
            mask_reasons.append("anti-export")
            masked = 1

        # 需量窗口：rolling_mean_15 接近 soft_cap 时，屏蔽“更糟糕”的动作
        roll_mean15 = self.prev_roll_mean15 if self.prev_roll_mean15 is not None else pcc0
        safety_margin = DEFAULTS["epsilon_softcap"]
        if roll_mean15 >= (self.policy.soft_cap_kW - safety_margin):
            # 屏蔽“增加 P_pcc”的动作：更多充电 或 减少放电
            # 充电：p_try < 0；减少放电：p_try 小于当前参考放电
            if p_try < p_base:
                p_try = p_base  # 不允许比参考更“增负荷”
                mask_reasons.append("softcap_guard")
                masked = 1

        # SOC 边界：先粗估下一步 SOC
        e_ch = max(0.0, -p_try) * self.cfg.eff_ch * (self.dt_min / 60.0)
        e_dis = max(0.0, p_try) / self.cfg.eff_dis * (self.dt_min / 60.0)
        soc_next = self.soc + (e_ch - e_dis) / self.cfg.rated_energy_kWh
        if soc_next > self.cfg.soc_max and p_try < 0:
            # 限充
            over = (soc_next - self.cfg.soc_max) * self.cfg.rated_energy_kWh / (self.dt_min / 60.0)
            p_try += over / self.cfg.eff_ch
            mask_reasons.append("soc_max")
            masked = 1
        if soc_next < self.cfg.soc_min and p_try > 0:
            # 限放
            under = (self.cfg.soc_min - soc_next) * self.cfg.rated_energy_kWh / (self.dt_min / 60.0)
            p_try -= under * self.cfg.eff_dis
            mask_reasons.append("soc_min")
            masked = 1

        # N-1 余度（保留头寸）：保障 pcc + n1_margin <= limit
        pcc_try = pcc0 + max(0.0, -p_try) - max(0.0, p_try)
        if (pcc_try + self.policy.n_minus_1_margin_kW) > self.policy.pcc_limit_kW:
            # 需要减少充电或增加放电
            if p_try < 0:
                # 限充
                need = (pcc_try + self.policy.n_minus_1_margin_kW - self.policy.pcc_limit_kW)
                p_try += need  # 减小充电幅度
            else:
                # 增放
                need = (pcc_try + self.policy.n_minus_1_margin_kW - self.policy.pcc_limit_kW)
                p_try -= need
            mask_reasons.append("n-1_guard")
            masked = 1

        # 事件优先：若事件进行，必须满足跟踪带（参考等价调度）
        event_active, event_tgt = self._in_event(t)
        if event_active:
            # 要求可用上行能力 >= event_tgt
            if p_avail_up < event_tgt:
                # 不够则强制回退参考，并记违约风险
                p_try = min(self.p_ref[i], max(0.0, p_avail_up))
                mask_reasons.append("event_track_guard")
                masked = 1

        # ---------------- 奖励与结算 ----------------
        # 结算基于“最终动作” p_act
        p_act = float(np.clip(p_try, -self.cfg.rated_power_kW, self.cfg.rated_power_kW))
        r_act = float(max(0.0, r_try))

        # 更新 SOC
        e_ch = max(0.0, -p_act) * self.cfg.eff_ch * (self.dt_min / 60.0)
        e_dis = max(0.0, p_act) / self.cfg.eff_dis * (self.dt_min / 60.0)
        soc_next = float(np.clip(self.soc + (e_ch - e_dis) / self.cfg.rated_energy_kWh, self.cfg.soc_min, self.cfg.soc_max))

        # PCC 实际
        pcc_act = pcc0 + max(0.0, -p_act) - max(0.0, p_act)

        # 费用/收益分解
        price = float(self.price[i])
        ef = float(self.ef[i])

        energy_cost = (e_ch * price) - (e_dis * price)  # 充电花钱、放电赚钱（负成本）
        reserve_pay = 0.0
        dr_pay = 0.0
        reserve_shortfall = 0.0
        if event_active:
            # 简式：事件期若上行能力不足，按差额罚
            short = max(0.0, event_tgt - p_avail_up)
            reserve_shortfall += short

        carbon_cost = (e_ch + e_dis) * ef * DEFAULTS["p_co2_yuan_per_kg"]  # 默认不计价，只统计 EF

        # 需量罚金（软限 -> 罚金）
        roll_mean15 = self._rolling_mean_15([pcc_act] if self.prev_roll_mean15 is None else [self.prev_roll_mean15, pcc_act])
        self.prev_roll_mean15 = roll_mean15
        peak_over = max(0.0, roll_mean15 - self.policy.soft_cap_kW)
        peak_penalty = peak_over * self.policy.penalty_yuan_per_kW

        # 退化成本：throughput * (1 + a*C^2 + b*max(0,T-30)^2)
        # 温度若不可得，按 28°C 近似；C_rate ~ |P| / E
        t_batt = 28.0
        c_rate = abs(p_act) / max(1e-6, self.cfg.rated_energy_kWh)
        a, b = 0.3, 0.02
        deg_factor = 1.0 + a * (c_rate ** 2) + b * max(0.0, t_batt - 30.0) ** 2
        throughput_kwh = (e_ch + e_dis)
        degradation_cost = throughput_kwh * deg_factor * self.cfg.cycle_cost_yuan_per_kWh

        # 平滑惩罚
        smooth_penalty = DEFAULTS["gamma_smooth"] * abs(p_act - self.p_prev)

        # 终态 SOC 惩罚：在回合尾巴加（env 外部会控制 done；此处只在最后一步计算）
        soc_end_penalty = 0.0
        done = (i + 1 >= min(len(self.ts), self.idx + self.h))
        if done:
            soc_end_penalty = DEFAULTS["zeta_soc_end"] * abs(soc_next - self.cfg.soc_target)

        # 奖励（收益为正）
        r_t = (
            + reserve_pay + dr_pay
            - energy_cost
            - carbon_cost
            - peak_penalty
            - DEFAULTS["beta_reserve"] * reserve_shortfall
            - degradation_cost
            - smooth_penalty
            - soc_end_penalty
        )

        # 统计累加
        self.metrics["reward_sum"] += float(r_t)
        self.metrics["energy_cost"] += float(energy_cost)
        self.metrics["reserve_pay"] += float(reserve_pay)
        self.metrics["dr_pay"] += float(dr_pay)
        self.metrics["carbon_cost"] += float(carbon_cost)
        self.metrics["peak_penalty"] += float(peak_penalty)
        self.metrics["reserve_shortfall"] += float(reserve_shortfall)
        self.metrics["degradation_cost"] += float(degradation_cost)
        self.metrics["smooth_penalty"] += float(smooth_penalty)

        # 与基线（参考）对比的“经济优势”（正数=节省）
        # 这里用行动级增量：以“若用参考功率 p_ref 与现在 p_act”的能源成本差额粗估。
        p_ref = float(self.p_ref[i])
        e_ch_ref = max(0.0, -p_ref) * self.cfg.eff_ch * (self.dt_min / 60.0)
        e_dis_ref = max(0.0, p_ref) / self.cfg.eff_dis * (self.dt_min / 60.0)
        energy_cost_ref = (e_ch_ref * price) - (e_dis_ref * price)
        econ_adv_now = (energy_cost_ref - energy_cost)  # 正数表示当前行为更省钱

        # 日志（单步 JSONL）
        record = {
            "key": "bess_step",
            "ts": int(t),
            "idx": int(i),
            "obs": self._get_obs(),  # 记录动作前的观测
            "action_in": {"dP": float(action.get("dP", 0.0)), "dR": float(action.get("dR", 0.0)), "mode": mode},
            "action_after_mask": {"p_kW": float(p_act), "r_kW": float(r_act)},
            "mask_applied": int(masked),
            "mask_reasons": mask_reasons,
            "pcc_base_kW": float(pcc0),
            "p_pcc_kW": float(pcc_act),
            "price_yuan_per_kWh": float(price),
            "ef_kg_per_kWh": float(ef),
            "soc": float(self.soc),
            "soc_next": float(soc_next),
            "reward": float(r_t),
            "reward_breakdown": {
                "reserve_pay": float(reserve_pay),
                "dr_pay": float(dr_pay),
                "energy_cost": float(energy_cost),
                "carbon_cost": float(carbon_cost),
                "peak_penalty": float(peak_penalty),
                "reserve_shortfall": float(reserve_shortfall),
                "degradation_cost": float(degradation_cost),
                "smooth_penalty": float(smooth_penalty),
                "soc_end_penalty": float(soc_end_penalty),
            },
            "baseline": {"p_ref_kW": float(p_ref), "energy_cost_ref": float(energy_cost_ref)},
            "econ_advantage_yuan": float(econ_adv_now),
        }
        self.log.write(record)

        # 更新状态，推进时间
        self.p_prev = p_act
        self.soc = soc_next
        self.idx += 1

        # 历史缓存
        self.hist_actions.pop(0)
        self.hist_actions.append(float(p_act))
        self.hist_masks.pop(0)
        self.hist_masks.append(int(masked))

        # 观测（用于下一步）
        next_obs = self._get_obs() if not done else {}
        info = {
            "masked": int(masked),
            "mask_reasons": mask_reasons,
            "econ_advantage_yuan": float(econ_adv_now),
            "p_act_kW": float(p_act),
            "p_ref_kW": float(p_ref),
            "pcc_kW": float(pcc_act),
        }
        return next_obs, float(r_t), bool(done), info


# ------------------------------
# 数据加载与环境构建
# ------------------------------

def _find_file(basename: str) -> Optional[str]:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, basename)
        if os.path.exists(p):
            return p
    # 模块相对路径（如配置里写了相对名）
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, "bess_energy", basename)
        if os.path.exists(p):
            return p
    return None


def _load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_configs(dt_min: int) -> Tuple[BessSiteConfig, DemandWindowPolicy]:
    # 尝试从 /mnt/data 和 模块 data/ 读取
    cfg_bess_path = _find_file("bess_master.json")
    cfg_window_path = _find_file("demand_window_config.json")
    cfg_bess = BessSiteConfig.from_json(_load_json(cfg_bess_path), dt_min=dt_min)
    window = DemandWindowPolicy.from_json(_load_json(cfg_window_path))
    # 若两处 export_allowed 不一致，取“更严格”的 False
    export_allowed = bool(cfg_bess.export_allowed and window.export_allowed)
    cfg_bess.export_allowed = export_allowed
    return cfg_bess, window


def load_series(cfg: BessSiteConfig, dt_min: int) -> Tuple[List[int], List[float], List[float], List[float], List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """
    返回：ts, pcc_base_kw, price, ef, reserve_events, dr_events
    - reserve/dr events: [(start_ts, end_ts, target_kW), ...]
    """
    # 定位文件名
    grid_meter_fn = cfg.telemetry_sources.get("grid_meter_csv", "grid_meter.csv")
    market_price_fn = cfg.telemetry_sources.get("market_price_csv", "market_price.csv")
    grid_ef_fn = cfg.telemetry_sources.get("grid_ef_csv", "grid_ef.csv")
    bess_tele_fn = "bess_telemetry.csv"  # 可选
    reserve_fn = "reserve_events.csv"
    dr_fn = "dr_events.csv"

    p_grid_path = _find_file(grid_meter_fn)
    price_path = _find_file(market_price_fn)
    ef_path = _find_file(grid_ef_fn)
    bess_path = _find_file(bess_tele_fn)
    reserve_path = _find_file(reserve_fn)
    dr_path = _find_file(dr_fn)

    # 读取基础序列
    s_grid = _read_csv_timeseries(p_grid_path, ["ts", "pcc_kw"]) if p_grid_path else []
    s_price = _read_csv_timeseries(price_path, ["ts", "price_yuan_per_kwh"]) if price_path else []
    s_ef = _read_csv_timeseries(ef_path, ["ts", "ef_kg_per_kwh"]) if ef_path else []

    # 对齐到 dt_min 栅格
    rs_grid = _resample_to_grid(s_grid, dt_min, ["pcc_kw"])
    rs_price = _resample_to_grid(s_price, dt_min, ["price_yuan_per_kwh"])
    rs_ef = _resample_to_grid(s_ef, dt_min, ["ef_kg_per_kwh"])

    # 融合同一 ts（取相同下标，若长度不等，取最短）
    n = min(len(rs_grid), len(rs_price), len(rs_ef))
    if n == 0:
        # 兜底数据：48h 的经验曲线（夜充+晚放）
        n = int((48 * 60) // dt_min)
        ts0 = int(time.time() // (dt_min * 60) * (dt_min * 60))
        ts = [ts0 + i * (dt_min * 60) for i in range(n)]
        pcc = [10000.0 + 2000.0 * (1 if (i % int(60 / dt_min) in range(18 // (dt_min // 1), 22 // (dt_min // 1))) else 0) for i in range(n)]
        price = [0.3 + 0.3 * (1 if (i % int(60 / dt_min) in range(9 // (dt_min // 1), 12 // (dt_min // 1)) or (i % int(60 / dt_min) in range(18 // (dt_min // 1), 22 // (dt_min // 1)))) else 0) for i in range(n)]
        ef = [0.65] * n
    else:
        ts = [rs_grid[i]["ts"] for i in range(n)]
        pcc = [float(rs_grid[i]["pcc_kw"]) for i in range(n)]
        price = [float(rs_price[i]["price_yuan_per_kwh"]) for i in range(n)]
        ef = [float(rs_ef[i]["ef_kg_per_kwh"]) for i in range(n)]

    # 读取事件列表（可选）
    reserve_events: List[Tuple[int, int, float]] = []
    if reserve_path:
        with open(reserve_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            headers = next(r, [])
            idx_s = _find_first_matching(headers, COLUMN_CANDIDATES["event_start"]) or 0
            idx_e = _find_first_matching(headers, COLUMN_CANDIDATES["event_end"]) or 1
            idx_k = _find_first_matching(headers, COLUMN_CANDIDATES["event_target_kw"]) or 2
            for row in r:
                ts_s = _parse_ts_any(row[idx_s]); ts_e = _parse_ts_any(row[idx_e])
                try:
                    kw = float(row[idx_k])
                except Exception:
                    kw = max(0.0, cfg.reserve_min_kW)
                if ts_s and ts_e:
                    reserve_events.append((ts_s, ts_e, kw))

    dr_events: List[Tuple[int, int, float]] = []
    if dr_path:
        with open(dr_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            headers = next(r, [])
            idx_s = _find_first_matching(headers, COLUMN_CANDIDATES["event_start"]) or 0
            idx_e = _find_first_matching(headers, COLUMN_CANDIDATES["event_end"]) or 1
            idx_k = _find_first_matching(headers, COLUMN_CANDIDATES["event_target_kw"]) or 2
            for row in r:
                ts_s = _parse_ts_any(row[idx_s]); ts_e = _parse_ts_any(row[idx_e])
                try:
                    kw = float(row[idx_k])
                except Exception:
                    kw = cfg.reserve_min_kW
                if ts_s and ts_e:
                    dr_events.append((ts_s, ts_e, kw))

    return ts, pcc, price, ef, reserve_events, dr_events


def make_env(dt_min: int = 10,
             horizon_steps: int = 144,
             jsonl_path: str = DEFAULT_JSONL) -> Tuple[BessCmdpEnv, EconomicMPCPlanner, Dict[str, Any]]:
    """
    外部统一入口：
    - 读取配置与数据
    - 生成参考轨迹与 CMDP 环境
    - 返回 env, planner, context（含原始序列用于基线对比/绘图）
    """
    cfg_bess, window = load_configs(dt_min=dt_min)
    ts, pcc, price, ef, reserve_events, dr_events = load_series(cfg_bess, dt_min=dt_min)
    planner = EconomicMPCPlanner(cfg_bess, window, dt_min=dt_min, horizon_steps=horizon_steps)
    log = JsonlLogger(jsonl_path)
    env = BessCmdpEnv(cfg_bess, window, dt_min, ts, pcc, price, ef, reserve_events, dr_events, planner, log, horizon_steps)
    ctx = {
        "cfg_bess": asdict(cfg_bess),
        "window": asdict(window),
        "ts": ts,
        "pcc": pcc,
        "price": price,
        "ef": ef,
        "reserve_events": reserve_events,
        "dr_events": dr_events,
    }
    return env, planner, ctx


# ------------------------------
# 基线策略（规则/MPC 参考）与评估
# ------------------------------

def baseline_policy_fn(obs: Dict[str, Any], planner: EconomicMPCPlanner) -> Dict[str, Any]:
    """
    基线策略：直接跟随参考计划（无残差）。
    - 便于计算“RL 相比基线”的经济优势。
    """
    return {"dP": 0.0, "dR": 0.0, "mode": ""}


def rollout_and_log(env: BessCmdpEnv,
                    policy_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                    baseline_policy_fn: Callable[[Dict[str, Any], EconomicMPCPlanner], Dict[str, Any]],
                    planner: EconomicMPCPlanner,
                    max_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    通用评估/生成日志工具：
    - 给定策略函数 policy_fn(obs)->action，跑一回合，输出 JSONL 与统计指标。
    - 统计中包含与基线（参考）对比的经济优势累计。
    """
    obs = env.reset(0)
    steps = 0
    econ_adv_sum = 0.0
    while True:
        # 策略输出（若需要可在引擎层引入随机性/探索）
        act = policy_fn(obs)
        # 环境推进
        obs, r, done, info = env.step(act)
        econ_adv_sum += float(info.get("econ_advantage_yuan", 0.0))
        steps += 1
        if done or (max_steps and steps >= max_steps):
            break

    # 收尾：镜像 JSONL 到静态目录
    env.log.close()
    _mirror_to_static(DEFAULT_JSONL, STATIC_JSONL)

    summary = {
        "key": "bess_episode_summary",
        "steps": steps,
        "reward_sum": float(env.metrics["reward_sum"]),
        "econ_advantage_yuan_total": float(econ_adv_sum),
        "breakdown": env.metrics,
    }
    # 末尾写一条汇总
    with open(DEFAULT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    _mirror_to_static(DEFAULT_JSONL, STATIC_JSONL)
    return summary


# ------------------------------
# 离线数据集构造（IQL 用）
# ------------------------------

def prepare_offline_dataset(env: BessCmdpEnv,
                            planner: EconomicMPCPlanner,
                            out_jsonl: str) -> str:
    """
    构造 IQL 离线训练数据集：用“参考计划 + 规则屏蔽”生成 (s,a,r,s',done)。
    - 直接把 Env 在“基线策略”下的轨迹写出，键稳定，前端和训练器共用。
    """
    path = out_jsonl
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = JsonlLogger(path)
    obs = env.reset(0)
    while True:
        action = baseline_policy_fn(obs, planner)
        next_obs, r, done, info = env.step(action)
        rec = {
            "key": "transition",
            "obs": obs,
            "action": action,
            "reward": float(r),
            "next_obs": next_obs,
            "done": bool(done),
        }
        log.write(rec)
        obs = next_obs
        if done:
            break
    log.close()
    return path


# ------------------------------
# 命令行自检（无训练）
# ------------------------------

def _self_check(dt: int, horizon: int, sleep_every: int, sleep_sec: int) -> int:
    """
    自检流程：
    1) 构建 env 与 planner，跑一回合 baseline（无 RL 残差）
    2) 生成 JSONL（步级 + 汇总）
    3) 每隔 sleep_every 步 sleep 一下（用于和训练流程的“休息”保持一致接口）
    """
    env, planner, ctx = make_env(dt_min=dt, horizon_steps=horizon, jsonl_path=DEFAULT_JSONL)
    def policy(obs):
        # 可以模拟一点小残差，检验屏蔽与审计
        return {"dP": 0.0, "dR": 0.0, "mode": ""}
    summary = rollout_and_log(env, policy, baseline_policy_fn, planner)
    print("[SELF-CHECK] episode summary:", json.dumps(summary, ensure_ascii=False))

    # 构造一份离线数据集（供后续 IQL 预训练）
    dataset_path = os.path.join(MODULE_DIR, "offline_dataset.jsonl")
    env2, planner2, _ = make_env(dt_min=dt, horizon_steps=horizon, jsonl_path=os.path.join(MODULE_DIR, "_tmp.jsonl"))
    ds_path = prepare_offline_dataset(env2, planner2, dataset_path)
    print("[SELF-CHECK] offline dataset written to:", ds_path)
    return 0


# ------------------------------
# main
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BESS Energy CMDP Env + MPC Baseline (no training)")
    parser.add_argument("--self-check", action="store_true", help="运行自检：构建环境 + 基线 rollout + 生成 JSONL")
    parser.add_argument("--dt-min", type=int, default=10, help="步长（分钟），建议 10")
    parser.add_argument("--horizon", type=int, default=144, help="滚动窗口步数（10min*144=24h）")
    parser.add_argument("--sleep-every", type=int, default=1000, help="每隔 N 步休眠（与训练接口一致）")
    parser.add_argument("--sleep-sec", type=int, default=60, help="每次休眠秒数（与训练接口一致）")
    args = parser.parse_args()

    if args.self_check:
        sys.exit(_self_check(args.dt_min, args.horizon, args.sleep_every, args.sleep_sec))
    else:
        print("Use --self-check to run env+baseline and generate JSONL.")
