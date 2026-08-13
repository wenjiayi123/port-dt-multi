# app/services/rl_model/yard_crane/module_f.py
# -*- coding: utf-8 -*-
"""
模块 F｜RTG/RMG 待机与功率模式 —— 孪生环境 + 规则/MPC 兜底 + 动作屏蔽 + JSONL 审计
==================================================================================
大白话：
- 这是 F 模块的“可落地环境”。与 E 模块结构一致：读取数据→生成参考（经济 MPC 启发式）→Residual RL 残差微调→
  动作屏蔽（SLA/温升/噪声/需量/DR）→奖励分解→JSONL 审计。数据缺失自动经验曲线兜底。
- 只用标准库 csv/json + 少量 numpy；不依赖 pandas。
- 输出 JSONL：`app/services/rl_model/yard_crane/policy_evaluate_history.jsonl`，并镜像到
  `static/api/rl/artifacts/policy_evaluate_history.jsonl` 以便前端直接展示。
- 接口稳定：make_env()/YardCraneEnv/rollout_and_log()/prepare_offline_dataset()。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta

import numpy as np

# ----------------------------
# 路径与常量
# ----------------------------
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = ["/mnt/data", os.path.join(MODULE_DIR, "data")]
DEFAULT_JSONL = os.path.join(MODULE_DIR, "policy_evaluate_history.jsonl")
STATIC_JSONL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(MODULE_DIR))),
    "static", "api", "rl", "artifacts", "policy_evaluate_history.jsonl"
)

# 列名候选（不区分大小写）
COLS = {
    "ts": ["ts", "timestamp", "time", "datetime", "utc", "time_utc", "end_time_utc", "start_time_utc"],
    "crane_id": ["crane_id", "id", "unit_id"],
    "block_id": ["block_id", "block", "yard_block"],
    "power_kW": ["power_kw", "power_kW", "p_kw", "p"],
    "speed_pct": ["speed_pct", "speed%", "speed", "freq_pct"],
    "temp_motor_C": ["temp_motor_c", "motor_temp_c", "temp_motor", "tmotor_c"],
    "temp_inv_C": ["temp_inverter_c", "inv_temp_c", "temp_inverter", "tinverter_c"],
    "mode": ["mode", "work_mode"],
    "start_stop": ["start_stop_event", "start_stop", "startstop"],
    "regen_kWh": ["regen_kwh", "regen_kWh", "kwh_regen"],

    "boxes_done": ["boxes_done", "boxes", "cnt_boxes", "moves", "moves_planned"],
    "cycle_time_s": ["cycle_time_s", "cyc_time_s", "cycle_s"],
    "distance_m": ["distance_m", "dist_m", "distance"],
    "queue_len": ["queue_len", "q_len", "len_q"],
    "wait_time_s": ["wait_time_s", "wait_s", "w_s"],

    "queue_p50": ["queue_len_p50", "q_p50", "arrivals_p50_per_step"],
    "queue_p90": ["queue_len_p90", "q_p90"],
    "arrive_p50": ["arrival_rate_p50", "arrive_p50"],

    "pcc_kw": ["pcc_kw", "grid_kw", "p_kw", "power_kw"],
    "price": ["price", "price_yuan_per_kWh", "rt_price", "da_price"],
    "ef": ["ef", "ef_kg_per_kWh", "grid_ef"],

    "near_res": ["near_residential", "near_res", "residential"],
    "quiet_hours": ["quiet_hours_local", "quiet_hours"],
    "noise_dba": ["night_noise_limit_dBA", "noise_limit_dba"],
}

# 默认参数（兜底）
DEFAULTS = {
    "dt_min": 5,
    "horizon_steps": 144,  # 12h @5min
    "alpha_idle": 1.0,
    "beta_sla": 25.0,
    "gamma_switch": 1.0,
    "eta_thermal": 1.5,
    "rho_regen": 0.2,   # 元/kWh
    "boxes_target_15m": 60.0,  # 目标吞吐（示例，真实由 TOS/班次表给出）
    "q_hi": 4.0,        # 高队列阈值（箱）
    "q_lo": 1.0,        # 低队列阈值（箱）
    "theta1_K": 10.0,   # 温升预警裕度
    "theta0_K": 5.0,    # 温升红线裕度
    "price_yuan": 0.6,
    "ef_kg_per_kWh": 0.65,
    "ambient_C": 28.0,
    "soft_cap_guard_kW": 800.0,  # 靠近软限的保护提前量
    "res_band_power_pct": 0.10,  # 残差带（功率百分比）
    "res_band_idle_min": 3.0,    # 残差带（待机分钟）
    "min_on_min": 10.0,
    "min_off_min": 5.0,
    "idle_auto_off_min": 8.0,
}

# Eco 档映射（可站点替换）
ECO_MAP = {
    "normal": {"spd": 1.00, "pwr": 1.00, "acc": 1.00, "sway": 1.00},
    "ecoL1":  {"spd": 0.95, "pwr": 0.95, "acc": 0.90, "sway": 1.05},
    "ecoL2":  {"spd": 0.85, "pwr": 0.85, "acc": 0.80, "sway": 1.10},
    "ecoL3":  {"spd": 0.75, "pwr": 0.75, "acc": 0.70, "sway": 1.15},
}

# ----------------------------
# 工具：时间戳与 CSV 读取
# ----------------------------
def _parse_ts_any(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().replace("T", " ").replace("Z", "+00:00")
    # 纯数字 epoch
    if s.isdigit():
        v = int(s)
        while v > 10**10:
            v //= 1000
        return v
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

def _find(headers: List[str], keys: List[str]) -> Optional[int]:
    low = [h.strip().lower() for h in headers]
    for k in keys:
        if k.lower() in low:
            return low.index(k.lower())
    return None

def _read_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        headers = next(r, [])
        rows = [row for row in r]
    return headers, rows

def _find_file(name: str) -> Optional[str]:
    for d in DATA_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def _resample_grid(ts_vals: List[int], series_vals: List[float], dt_min: int, start_ts: int, end_ts: int) -> Tuple[List[int], List[float]]:
    """线性插值到统一时间栅格 [start_ts, end_ts] 步长 dt_min"""
    step = dt_min * 60
    grid_ts = list(range(start_ts, end_ts + 1, step))
    if not ts_vals or not series_vals:
        return grid_ts, [float("nan")] * len(grid_ts)
    xs = np.array(ts_vals, dtype=np.int64)
    ys = np.array(series_vals, dtype=np.float64)
    out = []
    for t in grid_ts:
        if t <= xs[0]:
            out.append(float(ys[0]))
        elif t >= xs[-1]:
            out.append(float(ys[-1]))
        else:
            # 二分
            lo, hi = 0, len(xs) - 1
            while lo + 1 < hi:
                m = (lo + hi) // 2
                if xs[m] <= t:
                    lo = m
                else:
                    hi = m
            x0, x1 = xs[lo], xs[hi]
            y0, y1 = ys[lo], ys[hi]
            v = float(y0 + (t - x0) / (x1 - x0) * (y1 - y0))
            out.append(v)
    return grid_ts, out

# ----------------------------
# 配置数据类
# ----------------------------
@dataclass
class DemandWindowPolicy:
    soft_cap_kW: float
    pcc_limit_kW: float
    n_minus_1_margin_kW: float
    penalty_yuan_per_kW: float
    export_allowed: bool
    timezone: str

    @staticmethod
    def load_from_json(path: Optional[str]) -> "DemandWindowPolicy":
        d = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        # 字段名兼容（对齐站端口径）  :contentReference[oaicite:3]{index=3}
        soft_cap = float(d.get("soft_cap_kW") or d.get("limits", {}).get("plant_soft_cap_kw") or 1e12)
        pcc_limit = float(d.get("pcc_limit_kW") or d.get("limits", {}).get("pcc_limit_kw") or 1e12)
        n1 = float(d.get("n_minus_1_margin_kW") or d.get("limits", {}).get("n_minus_1_buffer_kw") or 0.0)
        penalty = float(d.get("penalty_yuan_per_kW") or d.get("penalties", {}).get("penalty_yuan_per_kW") or 95.0)
        export_allowed = bool(d.get("export_allowed", d.get("penalties", {}).get("export_allowed", False)))
        tz = d.get("timezone", "Asia/Shanghai")
        return DemandWindowPolicy(soft_cap, pcc_limit, n1, penalty, export_allowed, tz)

@dataclass
class CraneMaster:
    crane_id: str
    type: str  # RTG/RMG
    block_id: str
    idle_auto_off_min: float
    min_on_min: float
    min_off_min: float
    pf_min: float
    thd_max: float

# ----------------------------
# JSONL Logger
# ----------------------------
class JsonlLogger:
    def __init__(self, path: str = DEFAULT_JSONL):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    @staticmethod
    def _clean_nans(o):
        import math
        if isinstance(o, dict):
            return {k: JsonlLogger._clean_nans(v) for k, v in o.items()}
        if isinstance(o, list):
            return [JsonlLogger._clean_nans(v) for v in o]
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                return None
        return o

    def write(self, d: Dict[str, Any]):
        cleaned = JsonlLogger._clean_nans(d)
        # 强制标准 JSON：禁止 NaN/Inf
        self._fh.write(json.dumps(cleaned, ensure_ascii=False, allow_nan=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def _mirror_to_static(src: str, dst: str = STATIC_JSONL):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
            for line in fsrc:
                fdst.write(line)
    except Exception:
        pass

# ----------------------------
# 数据加载（容错 + 兜底）
# ----------------------------
def _load_cranes_master() -> Dict[str, CraneMaster]:
    path = _find_file("cranes_master.csv")
    out: Dict[str, CraneMaster] = {}
    if not path or not os.path.exists(path):
        # 兜底：构造 2 台示例机
        for cid, blk in [("RTG_01", "YB_06"), ("RMG_01", "YB_11")]:
            out[cid] = CraneMaster(cid, "RTG" if cid.startswith("RTG") else "RMG", blk,
                                   DEFAULTS["idle_auto_off_min"], DEFAULTS["min_on_min"], DEFAULTS["min_off_min"], 0.95, 0.08)
        return out

    headers, rows = _read_csv(path)
    idx = {
        "crane_id": _find(headers, COLS["crane_id"]),
        "type": _find(headers, ["crane_type", "type"]),
        "block_id": _find(headers, COLS["block_id"]),
        "idle_auto_off_min": headers.index("idle_auto_off_min") if "idle_auto_off_min" in [h.lower() for h in headers] else None,
        "min_on_min": headers.index("min_on_min") if "min_on_min" in [h.lower() for h in headers] else None,
        "min_off_min": headers.index("min_off_min") if "min_off_min" in [h.lower() for h in headers] else None,
        "pf_min": headers.index("pf_min") if "pf_min" in [h.lower() for h in headers] else None,
        "thd_max": headers.index("thd_max") if "thd_max" in [h.lower() for h in headers] else None,
    }
    for r in rows:
        try:
            cid = r[idx["crane_id"]]
            typ = (r[idx["type"]] if idx["type"] is not None else "RMG").upper()
            blk = r[idx["block_id"]] if idx["block_id"] is not None else "YB_00"
            idle = float(r[idx["idle_auto_off_min"]]) if idx["idle_auto_off_min"] is not None else DEFAULTS["idle_auto_off_min"]
            min_on = float(r[idx["min_on_min"]]) if idx["min_on_min"] is not None else DEFAULTS["min_on_min"]
            min_off = float(r[idx["min_off_min"]]) if idx["min_off_min"] is not None else DEFAULTS["min_off_min"]
            pf_min = float(r[idx["pf_min"]]) if idx["pf_min"] is not None else 0.95
            thd_max = float(r[idx["thd_max"]]) if idx["thd_max"] is not None else 0.08
            out[cid] = CraneMaster(cid, typ, blk, idle, min_on, min_off, pf_min, thd_max)
        except Exception:
            continue
    if not out:
        # 兜底
        out["RMG_01"] = CraneMaster("RMG_01", "RMG", "YB_11",
                                    DEFAULTS["idle_auto_off_min"], DEFAULTS["min_on_min"], DEFAULTS["min_off_min"], 0.95, 0.08)
    return out

def _load_yard_blocks() -> Dict[str, Dict[str, Any]]:
    path = _find_file("yard_blocks.csv")
    out: Dict[str, Dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return out
    headers, rows = _read_csv(path)
    idx = {
        "block_id": _find(headers, COLS["block_id"]),
        "near_res": _find(headers, COLS["near_res"]),
        "quiet_hours": _find(headers, COLS["quiet_hours"]),
        "noise": _find(headers, COLS["noise_dba"]),
    }
    for r in rows:
        try:
            bid = r[idx["block_id"]]
            near = int(float(r[idx["near_res"]])) if idx["near_res"] is not None and r[idx["near_res"]] != "" else 0
            qh = r[idx["quiet_hours"]] if idx["quiet_hours"] is not None else "22:00-06:00"
            noise = float(r[idx["noise"]]) if idx["noise"] is not None and r[idx["noise"]] != "" else 60.0
            out[bid] = {"near_residential": near, "quiet_hours": qh, "noise_limit": noise}
        except Exception:
            continue
    return out

def _load_series_generic(name: str, key: str) -> Tuple[List[int], List[float]]:
    path = _find_file(name)
    if not path or not os.path.exists(path):
        return [], []
    headers, rows = _read_csv(path)
    i_ts = _find(headers, COLS["ts"])
    i_val = _find(headers, COLS[key])
    if i_ts is None or i_val is None:
        return [], []
    ts, val = [], []
    for r in rows:
        t = _parse_ts_any(r[i_ts])
        if t is None:
            continue
        try:
            v = float(r[i_val])
        except Exception:
            continue
        ts.append(t); val.append(v)
    return ts, val

def _load_job_events() -> Dict[str, List[Dict[str, Any]]]:
    """
    读取 job_events.csv：
    - 关键列缺失（ts/crane_id）时，安全返回空字典，交由经验曲线兜底；
    - boxes_done / wait_time_s 为可选列，缺失则默认 0。
    """
    path = _find_file("job_events.csv")
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not path or not os.path.exists(path):
        return out

    headers, rows = _read_csv(path)
    i_ts = _find(headers, COLS["ts"])
    i_cid = _find(headers, COLS["crane_id"])
    i_boxes = _find(headers, COLS["boxes_done"])
    i_wait = _find(headers, COLS["wait_time_s"])

    # 关键列缺失 → 安全返回空
    if i_ts is None or i_cid is None:
        return out

    for r in rows:
        try:
            t = _parse_ts_any(r[i_ts])
            if t is None:
                continue
            cid = r[i_cid]
            boxes = float(r[i_boxes]) if (i_boxes is not None and r[i_boxes] != "") else 0.0
            wait = float(r[i_wait]) if (i_wait is not None and r[i_wait] != "") else 0.0
            out.setdefault(cid, []).append({"ts": t, "boxes_done": boxes, "wait_s": wait})
        except Exception:
            # 单行异常忽略
            continue
    for k in out:
        out[k].sort(key=lambda d: d["ts"])
    return out


def _load_crane_telemetry() -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    读取 crane_telemetry.csv：
    - 缺少 ts 或 crane_id → 返回空；
    - 其他列缺失则按 NaN/0 兜底。
    """
    path = _find_file("crane_telemetry.csv")
    out: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    if not path or not os.path.exists(path):
        return out

    headers, rows = _read_csv(path)
    i_ts = _find(headers, COLS["ts"])
    i_cid = _find(headers, COLS["crane_id"])
    if i_ts is None or i_cid is None:
        return out  # 关键列缺失安全返回

    i_p = _find(headers, COLS["power_kW"])
    i_spd = _find(headers, COLS["speed_pct"])
    i_tm = _find(headers, COLS["temp_motor_C"])
    i_ti = _find(headers, COLS["temp_inv_C"])
    i_mode = _find(headers, COLS["mode"])  # 目前不用
    i_ss = _find(headers, COLS["start_stop"])
    i_reg = _find(headers, COLS["regen_kWh"])

    for r in rows:
        try:
            t = _parse_ts_any(r[i_ts])
            if t is None:
                continue
            cid = r[i_cid]
            def _tofloat(ii):
                try:
                    return float(r[ii]) if ii is not None and r[ii] != "" else float("nan")
                except Exception:
                    return float("nan")
            p = _tofloat(i_p)
            spd = _tofloat(i_spd)
            tm = _tofloat(i_tm)
            ti = _tofloat(i_ti)
            ss = 1.0 if (i_ss is not None and r[i_ss] and str(r[i_ss]).strip() != "0") else 0.0
            reg = _tofloat(i_reg)

            out.setdefault(cid, {}).setdefault("power", []).append((t, p))
            out[cid].setdefault("speed", []).append((t, spd))
            out[cid].setdefault("tmotor", []).append((t, tm))
            out[cid].setdefault("tinv", []).append((t, ti))
            out[cid].setdefault("ss", []).append((t, ss))
            out[cid].setdefault("regen", []).append((t, reg))
        except Exception:
            continue
    return out


def _load_queue_forecast() -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    读取 queue_forecast.csv：
    - 缺少 ts 或 block_id → 返回空；
    - 分位列缺失则记为 NaN。
    """
    path = _find_file("queue_forecast.csv")
    out: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    if not path or not os.path.exists(path):
        return out

    headers, rows = _read_csv(path)
    i_ts = _find(headers, COLS["ts"])
    i_bid = _find(headers, COLS["block_id"])
    i_p50 = _find(headers, COLS["queue_p50"])
    i_p90 = _find(headers, COLS["queue_p90"])
    i_arr = _find(headers, COLS["arrive_p50"])

    # 关键列缺失 → 返回空
    if i_ts is None or i_bid is None:
        return out

    for r in rows:
        try:
            t = _parse_ts_any(r[i_ts])
            if t is None:
                continue
            bid = r[i_bid]
            def _tf(ii):
                try:
                    return float(r[ii]) if ii is not None and r[ii] != "" else float("nan")
                except Exception:
                    return float("nan")
            p50 = _tf(i_p50)
            p90 = _tf(i_p90)
            arr = _tf(i_arr)
            out.setdefault(bid, {}).setdefault("p50", []).append((t, p50))
            out[bid].setdefault("p90", []).append((t, p90))
            out[bid].setdefault("arr", []).append((t, arr))
        except Exception:
            continue
    return out

def _load_dr_events() -> List[Dict[str, Any]]:
    path = _find_file("dr_events.json")
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    # 规范字段名  :contentReference[oaicite:4]{index=4}
    evs = []
    for e in js:
        s = _parse_ts_any(e.get("start_utc"))
        e_ = _parse_ts_any(e.get("end_utc"))
        if s and e_:
            evs.append({
                "event_id": e.get("event_id"),
                "start_ts": s,
                "end_ts": e_,
                "required_reduction_kw": float(e.get("required_reduction_kw", 0.0)),
                "target_blocks": list(e.get("target_blocks", [])),
                "price_adder": float(e.get("price_adder_yuan_per_kWh", 0.0)),
            })
    evs.sort(key=lambda x: x["start_ts"])
    return evs

# ----------------------------
# 参考计划（经济 MPC 启发式）
# ----------------------------
class CraneMPCPlanner:
    """
    根据队列/温升/噪声/DR/需量，生成参考档位与功率上限、待机超时。
    - 先守 SLA 与温度，再在低队列/夜间/DR 时段下调到 eco 档。
    """
    def __init__(self, policy: DemandWindowPolicy, master: CraneMaster, dt_min: int, res_band_pct: float):
        self.policy = policy
        self.master = master
        self.dt_min = dt_min
        self.res_band_pct = res_band_pct

    def plan(self, ts: List[int], queue_p50: List[float], tmotor: List[float], tinv: List[float],
             block_meta: Dict[str, Any], dr_events: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        n = len(ts)
        mode = ["normal"] * n
        pwr_pct = [1.0] * n
        idle_min = [self.master.idle_auto_off_min if self.master.idle_auto_off_min>0 else DEFAULTS["idle_auto_off_min"]] * n

        # 噪声窗口
        near_res = int(block_meta.get("near_residential", 0)) if block_meta else 0
        qh = (block_meta or {}).get("quiet_hours", "22:00-06:00")
        def in_quiet(tloc: datetime) -> bool:
            try:
                s, e = qh.split("-")
                sh, sm = [int(x) for x in s.split(":")]
                eh, em = [int(x) for x in e.split(":")]
                st = tloc.replace(hour=sh, minute=sm, second=0)
                et = tloc.replace(hour=eh, minute=em, second=0)
                if sh <= eh:
                    return st <= tloc < et
                else:
                    # 跨午夜
                    return tloc >= st or tloc < et
            except Exception:
                return False

        # 事件索引
        def _dr_active(t: int, block_id: str) -> bool:
            for e in dr_events:
                if e["start_ts"] <= t < e["end_ts"]:
                    # 如果 target_blocks 非空且不含该 block，则跳过
                    if e["target_blocks"] and (block_id not in e["target_blocks"]):
                        continue
                    return True
            return False

        for i, t in enumerate(ts):
            q = queue_p50[i] if i < len(queue_p50) else float("nan")
            tm = tmotor[i] if i < len(tmotor) else float("nan")
            ti = tinv[i] if i < len(tinv) else float("nan")
            tloc = datetime.fromtimestamp(t, tz=timezone.utc).astimezone()
            quiet = in_quiet(tloc)
            thermal_margin = 999.0
            if not (math.isnan(tm) or math.isnan(ti)):
                thermal_margin = min( (90.0 - tm), (90.0 - ti) )  # 90°C 参考上限（可替换成配置）

            # 规则：
            # 1) 高队列 or thermal margin 低 → 保持 normal/ecoL1
            if (not math.isnan(q) and q > DEFAULTS["q_hi"]) or thermal_margin < DEFAULTS["theta1_K"]:
                mode[i] = "normal"
                pwr_pct[i] = 1.0
                idle_min[i] = max(self.master.idle_auto_off_min, DEFAULTS["idle_auto_off_min"])
                continue

            # 2) DR 活动 or 需量靠近软限 → 下调到 ecoL2/L3
            if _dr_active(t, self.master.block_id):
                mode[i] = "ecoL2"
                pwr_pct[i] = ECO_MAP["ecoL2"]["pwr"]
                idle_min[i] = max(idle_min[i], DEFAULTS["idle_auto_off_min"] + 2.0)
            # 3) 低队列/夜间噪声，进一步下调
            if (not math.isnan(q) and q <= DEFAULTS["q_lo"]) or (near_res and quiet):
                mode[i] = "ecoL3"
                pwr_pct[i] = ECO_MAP["ecoL3"]["pwr"]
                idle_min[i] = max(idle_min[i], DEFAULTS["idle_auto_off_min"] + 3.0)

            # 4) 常规低中队列 → ecoL1
            if mode[i] == "normal":
                mode[i] = "ecoL1"
                pwr_pct[i] = ECO_MAP["ecoL1"]["pwr"]

        return {"mode": mode, "pwr_pct": pwr_pct, "idle_min": idle_min}

# ----------------------------
# 环境（单机）
# ----------------------------
class YardCraneEnv:
    """
    单机 CMDP 环境：
    - 状态/观测：功率/温度/队列/DR/噪声/需量窗口等（含未来分位摘要）
    - 动作：连续残差（Δpower_limit_pct, Δidle_timeout_min），离散 mode（可选）
    - 屏蔽：SLA/温升/噪声/需量/DR，并计最小开停机驻留
    - 奖励：能耗/碳+空转+SLA 违约+切换+温升−回收
    """
    def __init__(self,
                 crane: CraneMaster,
                 policy: DemandWindowPolicy,
                 dt_min: int,
                 ts: List[int],
                 power_base: List[float],
                 price: List[float],
                 ef: List[float],
                 queue_p50: List[float],
                 job_hist: List[Dict[str, Any]],
                 tmotor: List[float],
                 tinv: List[float],
                 regen_kwh: List[float],
                 pcc_kw: List[float],
                 block_meta: Dict[str, Any],
                 planner: CraneMPCPlanner,
                 dr_events: List[Dict[str, Any]],
                 log: JsonlLogger):

        self.crane = crane
        self.policy = policy
        self.dt_min = dt_min
        self.h = len(ts)
        self.ts = ts
        self.power_base = np.array(power_base, dtype=np.float64)  # 观测到的“自然功率”（或经验曲线）
        self.price = np.array(price, dtype=np.float64)
        self.ef = np.array(ef, dtype=np.float64)
        self.queue_p50 = np.array(queue_p50, dtype=np.float64)
        self.job_hist = job_hist or []
        self.tmotor = np.array(tmotor, dtype=np.float64)
        self.tinv = np.array(tinv, dtype=np.float64)
        self.regen = np.array(regen_kwh, dtype=np.float64)
        self.pcc = np.array(pcc_kw, dtype=np.float64)
        self.block_meta = block_meta
        self.planner = planner
        self.dr_events = dr_events
        self.log = log

        self.idx = 0
        self.mode_ref = []
        self.pwr_ref = []
        self.idle_ref = []
        plan = planner.plan(ts, queue_p50, tmotor, tinv, block_meta, dr_events)
        self.mode_ref = plan["mode"]
        self.pwr_ref = np.array(plan["pwr_pct"], dtype=np.float64)
        self.idle_ref = np.array(plan["idle_min"], dtype=np.float64)

        # 残差带
        self.res_band_power = DEFAULTS["res_band_power_pct"]
        self.res_band_idle = DEFAULTS["res_band_idle_min"]

        # 计时器（最小驻留）
        self.on_timer_min = 999.0
        self.off_timer_min = 999.0
        self.is_on = True  # 简式：认为设备初始在岗

        # 历史轨迹（for obs）
        self.hist_actions = []
        self.hist_masks = []

        # 统计
        self.metrics = {
            "reward_sum": 0.0,
            "energy_cost": 0.0,
            "carbon_cost": 0.0,
            "idle_time_min": 0.0,
            "sla_viol": 0.0,
            "switch_cost": 0.0,
            "thermal_pen": 0.0,
            "regen_income": 0.0,
            "peak_penalty": 0.0,
        }

        self.prev_mode = self.mode_ref[0]
        self.prev_power_pct = self.pwr_ref[0]
        self.prev_idle_min = self.idle_ref[0]

    # ---- 工具 ----
    def _in_dr(self, t: int) -> Tuple[bool, float]:
        for e in self.dr_events:
            if e["start_ts"] <= t < e["end_ts"]:
                return True, float(e["required_reduction_kw"])
        return False, 0.0

    def _rolling_mean_15(self, arr: List[float]) -> float:
        steps = max(1, int(round(15 / self.dt_min)))
        if len(arr) < steps:
            return float(np.mean(arr))
        return float(np.mean(arr[-steps:]))

    def _obs(self) -> Dict[str, Any]:
        i = self.idx
        j2 = min(self.h, i + max(1, int(60/self.dt_min)))  # 未来1小时摘要
        fut_price = self.price[i:j2]; fut_ef = self.ef[i:j2]
        p50p = float(np.nanpercentile(fut_price, 50)) if len(fut_price)>0 else float(self.price[i])
        p90p = float(np.nanpercentile(fut_price, 90)) if len(fut_price)>0 else float(self.price[i])
        p50e = float(np.nanpercentile(fut_ef, 50)) if len(fut_ef)>0 else float(self.ef[i])
        p90e = float(np.nanpercentile(fut_ef, 90)) if len(fut_ef)>0 else float(self.ef[i])

        # 队列/吞吐摘要（15 分钟）
        boxes_15m = 0.0; wait_15m = 0.0
        win_s = self.dt_min*60*3
        for it in self.job_hist:
            if self.ts[i]-win_s <= it["ts"] <= self.ts[i]:
                boxes_15m += float(it.get("boxes_done", 0.0))
                wait_15m += float(it.get("wait_s", 0.0))
        obs = {
            "t": int(self.ts[i]),
            "idx": int(i),
            "mode_ref": self.mode_ref[i],
            "pwr_ref_pct": float(self.pwr_ref[i]),
            "idle_ref_min": float(self.idle_ref[i]),
            "queue_p50": float(self.queue_p50[i]) if i < len(self.queue_p50) else float("nan"),
            "tmotor": float(self.tmotor[i]) if i < len(self.tmotor) else float("nan"),
            "tinv": float(self.tinv[i]) if i < len(self.tinv) else float("nan"),
            "power_base_kW": float(self.power_base[i]),
            "price": float(self.price[i]),
            "ef": float(self.ef[i]),
            "pcc_kw": float(self.pcc[i]) if i < len(self.pcc) else float("nan"),
            "boxes_15m": float(boxes_15m),
            "wait_15m": float(wait_15m/60.0),
            "fut_price_p50": p50p,
            "fut_price_p90": p90p,
            "fut_ef_p50": p50e,
            "fut_ef_p90": p90e,
            "hist_actions": list(self.hist_actions[-6:]),
            "hist_masks": list(self.hist_masks[-6:]),
        }
        return obs

    def reset(self, start_idx: int = 0) -> Dict[str, Any]:
        self.idx = max(0, min(start_idx, self.h-1))
        self.on_timer_min = 999.0
        self.off_timer_min = 999.0
        self.is_on = True
        self.hist_actions = []; self.hist_masks = []
        self.prev_mode = self.mode_ref[self.idx]
        self.prev_power_pct = float(self.pwr_ref[self.idx])
        self.prev_idle_min = float(self.idle_ref[self.idx])
        return self._obs()

    # ---- 核心：一步推进 ----
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        i = self.idx
        t = self.ts[i]
        obs0 = self._obs()

        # 参考
        mode_b = self.mode_ref[i]
        pwr_b = float(self.pwr_ref[i])
        idle_b = float(self.idle_ref[i])

        # ---- 动作（残差 + 可选离散模式）+ NaN 兜底 ----
        d_power_raw = action.get("d_power_pct", 0.0)
        d_idle_raw = action.get("d_idle_min", 0.0)

        def _finite_or(x, dv):
            try:
                xx = float(x)
                if math.isnan(xx) or math.isinf(xx):
                    return dv, True
                return xx, False
            except Exception:
                return dv, True

        d_power, nan1 = _finite_or(d_power_raw, 0.0)
        d_idle, nan2 = _finite_or(d_idle_raw, 0.0)
        # 残差带裁剪
        d_power = float(np.clip(d_power, -self.res_band_power, self.res_band_power))
        d_idle = float(np.clip(d_idle, -self.res_band_idle, self.res_band_idle))
        mode_cmd = str(action.get("mode", "")) or mode_b

        # 试行设定
        pwr_try = pwr_b + d_power
        idle_try = idle_b + d_idle
        nan_guard_triggered = False
        # 数值保护：非有限回退基线
        if not (math.isfinite(pwr_try) and math.isfinite(idle_try)):
            pwr_try = pwr_b
            idle_try = idle_b
            nan_guard_triggered = True

        # 基础边界
        pwr_try = float(np.clip(pwr_try, 0.6, 1.0))
        idle_try = max(0.0, float(idle_try))

        mask_reasons = []
        masked = 0
        if nan_guard_triggered or nan1 or nan2:
            mask_reasons.append("nan_guard")
            masked = 1

        # ---- 屏蔽 1：SLA（15 分钟箱量/队列） ----
        q = obs0["queue_p50"]
        q_hi = DEFAULTS["q_hi"];
        q_lo = DEFAULTS["q_lo"]
        # 在低队列时，不许吞吐罚拉低节能收益：动态目标
        if math.isnan(q):
            targ_eff = 0.5 * DEFAULTS["boxes_target_15m"]
        elif q <= q_lo:
            targ_eff = 0.0
        elif q >= q_hi:
            targ_eff = DEFAULTS["boxes_target_15m"]
        else:
            scale = (q - q_lo) / max(1e-6, (q_hi - q_lo))  # [0,1]
            targ_eff = DEFAULTS["boxes_target_15m"] * scale

        # 若队列高（q>q_hi），仍禁止降功率/延长待机/深 eco
        if (not math.isnan(q) and q > q_hi):
            if pwr_try < pwr_b:
                pwr_try = pwr_b;
                mask_reasons.append("sla_guard_power_down");
                masked = 1
            if idle_try > idle_b:
                idle_try = idle_b;
                mask_reasons.append("sla_guard_idle_up");
                masked = 1
            if mode_cmd in ("ecoL2", "ecoL3"):
                mode_cmd = mode_b;
                mask_reasons.append("sla_guard_mode");
                masked = 1

        # ---- 屏蔽 2：温升裕度 ----
        tm = obs0["tmotor"];
        ti = obs0["tinv"]
        if not (math.isnan(tm) or math.isnan(ti)):
            thermal_margin = min(90.0 - tm, 90.0 - ti)
            if thermal_margin < DEFAULTS["theta1_K"]:
                if pwr_try > pwr_b:
                    pwr_try = pwr_b;
                    mask_reasons.append("thermal_guard_no_up");
                    masked = 1
            if thermal_margin < DEFAULTS["theta0_K"]:
                pwr_try = min(pwr_try, 0.85)
                mode_cmd = "normal"
                mask_reasons.append("thermal_redline_cap");
                masked = 1

        # ---- 屏蔽 3：夜间噪声 ----
        near_res = int(self.block_meta.get("near_residential", 0)) if self.block_meta else 0
        quiet = False
        if self.block_meta:
            qh = self.block_meta.get("quiet_hours", "22:00-06:00")
            try:
                s, e = qh.split(":")[0], qh
            except Exception:
                pass
            try:
                s, e = qh.split("-")
                hh, mm = [int(x) for x in s.split(":")]
                eh, em = [int(x) for x in e.split(":")]
                tl = datetime.fromtimestamp(t, tz=timezone.utc).astimezone()
                st = tl.replace(hour=hh, minute=mm, second=0)
                et = tl.replace(hour=eh, minute=em, second=0)
                quiet = (st <= tl < et) if hh <= eh else (tl >= st or tl < et)
            except Exception:
                quiet = False
        if near_res and quiet:
            if d_idle < 0:
                d_idle = 0.0;
                idle_try = idle_b
                mask_reasons.append("noise_guard_idle_shorten");
                masked = 1
            pwr_try = max(pwr_try, 0.75)
            mask_reasons.append("noise_guard_speed_cap");
            masked = 1

        # ---- 屏蔽 4：需量/DR ----
        roll15 = self._rolling_mean_15(list(self.pcc[max(0, i - 2):i + 1])) if self.pcc.size > 0 else 0.0
        if roll15 > (self.policy.soft_cap_kW - DEFAULTS["soft_cap_guard_kW"]):
            if pwr_try > pwr_b:
                pwr_try = pwr_b;
                mask_reasons.append("softcap_guard_power_up");
                masked = 1
        dr_active, _ = self._in_dr(t)
        if dr_active and pwr_try > pwr_b:
            pwr_try = pwr_b;
            mask_reasons.append("dr_guard_power_up");
            masked = 1

        # ---- 屏蔽 5：最小 on/off 驻留 ----
        if self.is_on:
            self.on_timer_min += self.dt_min;
            self.off_timer_min = 0.0
            if self.on_timer_min < max(self.crane.min_on_min, DEFAULTS["min_on_min"]):
                if idle_try > idle_b:
                    idle_try = idle_b;
                    mask_reasons.append("min_on_guard");
                    masked = 1
        else:
            self.off_timer_min += self.dt_min;
            self.on_timer_min = 0.0
            if self.off_timer_min < max(self.crane.min_off_min, DEFAULTS["min_off_min"]):
                if idle_try < idle_b:
                    idle_try = idle_b;
                    mask_reasons.append("min_off_guard");
                    masked = 1

        action_after = {"mode": mode_cmd, "power_limit_pct": float(pwr_try), "idle_timeout_min": float(idle_try)}

        # ---- 结算/奖励（含队列自适应 SLA）----
        dt_h = self.dt_min / 60.0
        p_natural = float(self.power_base[i]) if math.isfinite(self.power_base[i]) else 150.0
        p_act = min(p_natural, p_natural * pwr_try)
        idle_time_min = min(self.dt_min, obs0["wait_15m"])

        energy_kwh = p_act * dt_h
        price_i = float(self.price[i] if math.isfinite(self.price[i]) else DEFAULTS["price_yuan"])
        ef_i = float(self.ef[i] if math.isfinite(self.ef[i]) else DEFAULTS["ef_kg_per_kWh"])
        energy_cost = energy_kwh * price_i
        carbon_cost = energy_kwh * ef_i * 0.0  # 默认不计价

        # 动态目标下的吞吐罚
        boxes_gap = max(0.0, targ_eff - obs0["boxes_15m"])
        wait_gap = max(0.0, obs0["wait_15m"] - 5.0) if (not math.isnan(q) and q > q_lo) else 0.0
        sla_pen = DEFAULTS["beta_sla"] * (boxes_gap + 0.5 * wait_gap)

        switch_cost = DEFAULTS["gamma_switch"] * (1.0 if mode_cmd != self.prev_mode else 0.0)

        T_target = 75.0
        thermal_pen = 0.0
        if not (math.isnan(tm) or math.isnan(ti)):
            def softplus(x): return math.log1p(math.exp(x))

            thermal_pen = DEFAULTS["eta_thermal"] * (softplus((tm - T_target) / 3.0) + softplus((ti - T_target) / 3.0))

        regen_income = self.regen[i] * DEFAULTS["rho_regen"] if (
                    i < len(self.regen) and math.isfinite(self.regen[i])) else 0.0
        peak_over = max(0.0, roll15 - self.policy.soft_cap_kW)
        peak_penalty = peak_over * self.policy.penalty_yuan_per_kW

        r_t = - (energy_cost + carbon_cost) - DEFAULTS["alpha_idle"] * idle_time_min \
              - sla_pen - switch_cost - thermal_pen - peak_penalty + regen_income

        p_ref = min(p_natural, p_natural * pwr_b)
        kwh_ref = p_ref * dt_h
        cost_ref = kwh_ref * price_i
        econ_adv = cost_ref - energy_cost

        # 日志
        rec = {
            "key": "crane_step",
            "crane_id": self.crane.crane_id,
            "block_id": self.crane.block_id,
            "ts": int(t),
            "obs": obs0,
            "action_in": {"d_power_pct": float(d_power), "d_idle_min": float(d_idle),
                          "mode": str(action.get("mode", ""))},
            "action_after_mask": action_after,
            "mask_applied": int(masked),
            "mask_reasons": mask_reasons,
            "p_act_kW": float(p_act),
            "p_ref_kW": float(p_ref),
            "price_yuan_per_kWh": price_i,
            "ef_kg_per_kWh": ef_i,
            "reward": float(r_t),
            "reward_breakdown": {
                "energy_cost": float(energy_cost), "carbon_cost": float(carbon_cost),
                "idle_time_min": float(idle_time_min), "sla_penalty": float(sla_pen),
                "switch_cost": float(switch_cost), "thermal_penalty": float(thermal_pen),
                "regen_income": float(regen_income), "peak_penalty": float(peak_penalty),
            },
            "baseline": {"mode_ref": mode_b, "pwr_ref_pct": float(pwr_b), "idle_ref_min": float(idle_b),
                         "cost_ref": float(cost_ref)},
            "econ_advantage_yuan": float(econ_adv),
        }
        self.log.write(rec)

        # on/off 状态近似
        if idle_try >= max(self.crane.idle_auto_off_min, DEFAULTS["idle_auto_off_min"]) and \
                (math.isnan(q) or q <= q_lo):
            self.is_on = False
        else:
            self.is_on = True

        self.on_timer_min = (self.on_timer_min + self.dt_min) if self.is_on else 0.0
        self.off_timer_min = (self.off_timer_min + self.dt_min) if not self.is_on else 0.0
        self.prev_mode, self.prev_power_pct, self.prev_idle_min = mode_cmd, pwr_try, idle_try

        # 累计指标
        self.metrics["reward_sum"] += float(r_t)
        self.metrics["energy_cost"] += float(energy_cost)
        self.metrics["carbon_cost"] += float(carbon_cost)
        self.metrics["idle_time_min"] += float(idle_time_min)
        self.metrics["sla_viol"] += float(sla_pen)
        self.metrics["switch_cost"] += float(switch_cost)
        self.metrics["thermal_pen"] += float(thermal_pen)
        self.metrics["regen_income"] += float(regen_income)
        self.metrics["peak_penalty"] += float(peak_penalty)

        # 历史（清洗后入库，杜绝 NaN）
        self.hist_actions.append({"pwr_pct": float(pwr_try), "idle_min": float(idle_try)})
        self.hist_masks.append(int(masked))

        # 推进
        self.idx += 1
        done = self.idx >= self.h
        next_obs = self._obs() if not done else {}
        info = {
            "masked": int(masked),
            "mask_reasons": mask_reasons,
            "econ_advantage_yuan": float(econ_adv),
            "p_act_kW": float(p_act),
            "p_ref_kW": float(p_ref),
            "roll15_pcc": float(roll15),
            "targ_boxes_eff": float(targ_eff)
        }
        return next_obs, float(r_t), bool(done), info


# ----------------------------
# 构建环境入口
# ----------------------------
def make_env(dt_min: int = DEFAULTS["dt_min"],
             horizon_steps: int = DEFAULTS["horizon_steps"],
             crane_id: Optional[str] = None,
             jsonl_path: str = DEFAULT_JSONL) -> Tuple[YardCraneEnv, CraneMPCPlanner, Dict[str, Any]]:
    # 站端需量与并网  :contentReference[oaicite:5]{index=5}
    dwp = DemandWindowPolicy.load_from_json(_find_file("demand_window_config.json"))
    # 并网边界统一（若有 BESS 配置）  :contentReference[oaicite:6]{index=6}
    try:
        path_bess = _find_file("bess_master.json")
        if path_bess and os.path.exists(path_bess):
            with open(path_bess, "r", encoding="utf-8") as f:
                jb = json.load(f)
            dwp.export_allowed = bool(dwp.export_allowed and jb.get("export_allowed", False))
    except Exception:
        pass

    # 主数据
    masters = _load_cranes_master()
    if not crane_id:
        crane_id = sorted(list(masters.keys()))[0]
    master = masters.get(crane_id, list(masters.values())[0])
    blocks = _load_yard_blocks()
    block_meta = blocks.get(master.block_id, {})

    # 时间轴：优先由 grid_meter 或 price 决定
    ts_pcc, v_pcc = _load_series_generic("grid_meter.csv", "pcc_kw")
    ts_price, v_price = _load_series_generic("market_price.csv", "price")
    ts_ef, v_ef = _load_series_generic("grid_ef.csv", "ef")
    ts_ref = ts_pcc or ts_price or ts_ef
    if not ts_ref:
        # 兜底 12h 栅格
        now = int(time.time() // (dt_min*60) * (dt_min*60))
        ts_ref = [now + i*dt_min*60 for i in range(horizon_steps)]
        v_pcc = [10000.0 + 1000.0 * (1 if (i % (60//dt_min) in range(18//(dt_min//1), 22//(dt_min//1))) else 0) for i in range(len(ts_ref))]
        v_price = [0.4 + 0.3 * (1 if (i % (60//dt_min) in (list(range(9//(dt_min//1),12//(dt_min//1))) + list(range(18//(dt_min//1),21//(dt_min//1))))) else 0) for i in range(len(ts_ref))]
        v_ef = [0.65] * len(ts_ref)
    start_ts, end_ts = ts_ref[0], ts_ref[-1]
    ts_grid, pcc_grid = _resample_grid(ts_pcc, v_pcc, dt_min, start_ts, end_ts)
    _, price_grid = _resample_grid(ts_price, v_price, dt_min, start_ts, end_ts)
    _, ef_grid = _resample_grid(ts_ef, v_ef, dt_min, start_ts, end_ts)

    # 起重机数据
    tel = _load_crane_telemetry()
    job = _load_job_events()
    qf = _load_queue_forecast()

    # 对齐到栅格（该机）
    def _resample_list(pairs: List[Tuple[int, float]]) -> List[float]:
        if not pairs:
            return [float("nan")] * len(ts_grid)
        tsv = [p[0] for p in pairs]; vv = [p[1] for p in pairs]
        _, out = _resample_grid(tsv, vv, dt_min, start_ts, end_ts)
        return out

    power_base = _resample_list(tel.get(crane_id, {}).get("power", []))
    tmotor = _resample_list(tel.get(crane_id, {}).get("tmotor", []))
    tinv = _resample_list(tel.get(crane_id, {}).get("tinv", []))
    regen = _resample_list(tel.get(crane_id, {}).get("regen", []))
    queue_p50 = _resample_list(qf.get(master.block_id, {}).get("p50", []))

    # 兜底：经验曲线
    if all(math.isnan(x) for x in power_base):
        power_base = [200.0 + 60.0*math.sin(2*math.pi*i/24.0) for i in range(len(ts_grid))]  # kW 经验曲线
    if all(math.isnan(x) for x in queue_p50):
        queue_p50 = [1.5 + 1.0*math.sin(2*math.pi*i/48.0) for i in range(len(ts_grid))]     # 箱

    # DR 事件  :contentReference[oaicite:7]{index=7}
    dr_events = _load_dr_events()

    # 计划器与环境
    planner = CraneMPCPlanner(dwp, master, dt_min, DEFAULTS["res_band_power_pct"])
    log = JsonlLogger(jsonl_path)
    env = YardCraneEnv(master, dwp, dt_min, ts_grid, power_base, price_grid, ef_grid,
                       queue_p50, job.get(crane_id, []), tmotor, tinv, regen,
                       pcc_grid, block_meta, planner, dr_events, log)
    ctx = {
        "policy": asdict(dwp),
        "crane": asdict(master),
        "block_meta": block_meta,
        "ts": ts_grid,
        "pcc_kw": pcc_grid,
        "price": price_grid,
        "ef": ef_grid,
        "queue_p50": queue_p50,
        "dr_events": dr_events,
    }
    return env, planner, ctx

# ----------------------------
# 基线策略 & Rollout
# ----------------------------
def baseline_policy_fn(obs: Dict[str, Any]) -> Dict[str, Any]:
    # 不做残差
    return {"d_power_pct": 0.0, "d_idle_min": 0.0, "mode": ""}

def rollout_and_log(env: YardCraneEnv,
                    policy_fn,
                    max_steps: Optional[int] = None) -> Dict[str, Any]:
    obs = env.reset(0)
    steps = 0
    econ_adv_sum = 0.0
    while True:
        act = policy_fn(obs)
        obs, r, done, info = env.step(act)
        econ_adv_sum += float(info.get("econ_advantage_yuan", 0.0))
        steps += 1
        if done or (max_steps and steps >= max_steps):
            break
    env.log.write({
        "key": "crane_episode_summary",
        "crane_id": env.crane.crane_id,
        "steps": int(steps),
        "reward_sum": float(env.metrics["reward_sum"]),
        "econ_advantage_yuan_total": float(econ_adv_sum),
        "breakdown": env.metrics
    })
    env.log.close()
    _mirror_to_static(DEFAULT_JSONL, STATIC_JSONL)
    return {
        "steps": steps,
        "reward_sum": float(env.metrics["reward_sum"]),
        "econ_advantage_yuan_total": float(econ_adv_sum),
        "breakdown": env.metrics
    }

# ----------------------------
# 离线数据集（TD3+BC / IQL 预训练用）
# ----------------------------
def prepare_offline_dataset(env: YardCraneEnv, out_jsonl: str) -> str:
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        obs = env.reset(0)
        while True:
            act = baseline_policy_fn(obs)
            next_obs, r, done, info = env.step(act)
            f.write(json.dumps({
                "key": "transition",
                "obs": obs,
                "action": act,
                "reward": float(r),
                "next_obs": next_obs,
                "done": bool(done)
            }, ensure_ascii=False) + "\n")
            obs = next_obs
            if done:
                break
    return out_jsonl

# ----------------------------
# main（自检）
# ----------------------------
def _self_check(crane_id: Optional[str], dt_min: int, horizon: int, sleep_every: int, sleep_sec: int) -> int:
    env, planner, ctx = make_env(dt_min=dt_min, horizon_steps=horizon, crane_id=crane_id, jsonl_path=DEFAULT_JSONL)
    summary = rollout_and_log(env, baseline_policy_fn, max_steps=horizon)
    print("[F-MODULE SELF-CHECK] summary:", json.dumps(summary, ensure_ascii=False))
    # 生成一份离线数据集
    ds_path = os.path.join(MODULE_DIR, "offline_dataset_crane.jsonl")
    env2, _, _ = make_env(dt_min=dt_min, horizon_steps=horizon, crane_id=crane_id, jsonl_path=os.path.join(MODULE_DIR, "_tmp.jsonl"))
    prepare_offline_dataset(env2, ds_path)
    print("[F-MODULE SELF-CHECK] offline dataset:", ds_path)
    return 0

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Yard Crane Env + Shielding + MPC Baseline (Module F)")
    p.add_argument("--self-check", action="store_true", help="运行自检：基线 rollout + JSONL 审计 + 离线数据")
    p.add_argument("--crane-id", type=str, default="", help="指定起重机 ID（缺省取第一台）")
    p.add_argument("--dt-min", type=int, default=DEFAULTS["dt_min"])
    p.add_argument("--horizon", type=int, default=DEFAULTS["horizon_steps"])
    p.add_argument("--sleep-every", type=int, default=1000)
    p.add_argument("--sleep-sec", type=int, default=60)
    args = p.parse_args()
    if args.self_check:
        sys.exit(_self_check(args.crane_id or None, args.dt_min, args.horizon, args.sleep_every, args.sleep_sec))
    else:
        print("Use --self-check to run environment rollout and generate JSONL.")
