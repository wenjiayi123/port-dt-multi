# -*- coding: utf-8 -*-
"""
HVAC 冷站/末端设定点联动 —— API 入口（IQL 残差升级版）
==================================================
新增能力：
- 引入 IQL（Implicit Q-Learning）离线策略的推理后端 IQLPolicy（纯 numpy 实现 MLP 前向）
- ResidualPolicy 自动选择后端：优先 IQL，失败时退回启发式
- 动作幅度 δ、安全域、DR/需量紧张折减与动作屏蔽兼容
- 输出审计新增 rl_backend，便于确认实际使用的 RL 算法

说明：
- IQL 模型文件：policy.bin（JSON），位置：app/services/rl_model/hvac_cooling/policy.bin
  期望结构：
  {
    "arch": {"layers": [N_in, H1, H2, N_out]},
    "norm": {"mean": {...}, "std": {...}},         # 可选；键名见 IQLPolicy.DEFAULT_INPUT_KEYS
    "weights": [ {"W": [[...]], "b": [...]}, ...], # len = len(layers)-1
    "output_keys": ["dCHWS", "dSAT", "dSP"]        # 可选；默认就是这三个顺序
  }
- 若 policy.bin 不存在或解析失败：自动回退启发式残差（与上一版一致）

数据口径/需量/步长/设定点边界仍读取：
- demand_window_config.json（步长、软上限、罚金等）  # 引用：需求口径来源
- plant_master.json（上下界/爬坡等）                    # 引用：设备与设定点边界
"""

from __future__ import annotations

import os
import sys
import csv
import json
import argparse
import time
import math
import uuid
import hashlib
import errno
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    class _NP:
        def clip(self, a, a_min, a_max):
            return max(a_min, min(a_max, a))
        def mean(self, x): return sum(x) / max(1, len(x))
        def array(self, x): return x
        def dot(self, A, x):
            # 仅处理二维*一维的简单情形
            return [sum(a_i * x_i for a_i, x_i in zip(a_row, x)) for a_row in A]
    np = _NP()  # type: ignore

# ---------- 常量与默认 ----------
MODULE_NAME = "hvac_cooling"
BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
STATE_PATH = os.path.join(ARTIFACT_DIR, "hvac_cooling_state.json")
DEFAULT_OUT = os.path.join(ARTIFACT_DIR, "policy_evaluate_history.jsonl")
POLICY_BIN = os.path.join(BASE_DIR, "policy.bin")         # IQL 策略权重（JSON）
POLICY_META = os.path.join(BASE_DIR, "policy_meta.json")  # 可选的补充元数据

DEFAULT_RESIDUAL_DELTA = {
    "CHWS_C": 0.5,
    "SAT_C": 0.5,
    "SP_Pa": 80,
    "VFD_pct": 5,
}

HEADER_CANDIDATES = {
    "ts": ["ts", "timestamp", "time", "datetime", "date_time", "DateTime", "采集时间", "时间"],
    "PCC_kW": ["pcc_kw", "grid_kw", "total_kw", "plant_kw", "PCC_kW", "总功率kW", "总有功功率kW"],
    "CHWS_C": ["chws", "chw_supply_C", "chilled_water_supply_temp_C", "供冷水温(出水)", "CHWS_C"],
    "CHWR_C": ["chwr", "chw_return_C", "chilled_water_return_temp_C", "回水温", "CHWR_C"],
    "CHW_flow": ["chw_flow", "chw_flow_m3_h", "chw_flow_m3h", "chw_flow_kg_s", "一次侧流量", "CHW_flow"],
    "SAT_C": ["sat", "supply_air_temp_C", "送风温", "SAT_C"],
    "SP_Pa": ["static_pressure", "sp", "supply_static_pressure_Pa", "静压", "SP_Pa"],
    "valve_p90": ["valve_p90", "valve_pct_p90", "vav_p90", "阀位P90"],
    "DB_C": ["db", "dry_bulb_C", "outdoor_db_C", "室外干球", "DB_C"],
    "WB_C": ["wb", "wet_bulb_C", "outdoor_wb_C", "室外湿球", "WB_C"],
    "RH_pct": ["rh", "relative_humidity_pct", "室外相对湿度%", "RH_pct"],
    "zone_temp_max": ["zone_temp_max", "区域最高温", "T_zone_max"],
    "zone_rh_max": ["zone_rh_max", "区域最高湿度", "RH_zone_max"],
}
PRICE_HEADER_CANDS = {"ts": ["ts", "timestamp", "time", "datetime"], "price_yuan_per_kwh": ["price", "elec_price_yuan_per_kWh", "yuan_per_kWh", "电价", "rmb_per_kwh"]}
EF_HEADER_CANDS    = {"ts": ["ts", "timestamp", "time", "datetime"], "ef_kg_per_kwh":     ["ef", "kgco2_per_kWh", "ef_kg_per_kwh", "排放因子kg/kWh"]}
WEATHER_HEADER_CANDS = {"ts": ["ts", "timestamp", "time", "datetime"], "DB_C": ["db", "dry_bulb_C", "DB_C"], "WB_C": ["wb", "wet_bulb_C", "WB_C"], "RH_pct": ["rh", "RH_pct", "relative_humidity_pct"]}
LOAD_HEADER_CANDS    = {"ts": ["ts", "timestamp", "time", "datetime"], "load_kw": ["load_kw", "q_cool_kw", "plant_kw_forecast", "cooling_load_kw", "冷站负荷kW"]}
EFFMAP_HEADER_CANDS  = {"CHWS_C": ["chws_C", "CHWS_C", "chws"], "WB_C": ["wb_C", "WB_C", "wb"], "COP": ["cop", "COP"], "EIR": ["eir", "EIR"]}

# ---------- 工具 ----------
def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def now_utc_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_ts_any(s: str) -> Optional[datetime]:
    if s is None: return None
    s = str(s).strip()
    if not s: return None
    if s.isdigit() and len(s) in (10, 13):
        try:
            sec = int(s[:10]); return datetime.utcfromtimestamp(sec)
        except Exception:
            pass
    fmts = ["%Y-%m-%d %H:%M:%S","%Y/%m/%d %H:%M:%S","%Y-%m-%d %H:%M","%Y/%m/%d %H:%M","%Y-%m-%dT%H:%M:%S","%Y-%m-%dT%H:%M:%SZ","%Y/%m/%d","%Y-%m-%d"]
    for fmt in fmts:
        try: return datetime.strptime(s, fmt)
        except Exception: continue
    return None

def softplus(x: float, k: float = 6.0) -> float:
    try: return math.log1p(math.exp(k * x)) / k
    except OverflowError: return x if x > 0 else 0.0

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None: return default
        if isinstance(x, (float, int)): return float(x)
        s = str(x).strip().replace(",", "")
        if s == "": return default
        return float(s)
    except Exception:
        return default

def dew_point_C(db_C: float, rh_pct: float) -> float:
    rh = max(1e-6, min(100.0, rh_pct)) / 100.0
    a, b = 17.62, 243.12
    gamma = (a * db_C) / (b + db_C) + math.log(rh)
    return (b * gamma) / (a - gamma)

def gen_nonce() -> str:
    raw = f"{uuid.uuid4()}-{time.time()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def clip_rate(prev: float, target: float, max_delta: float) -> float:
    if target > prev: return min(target, prev + max_delta)
    else: return max(target, prev - max_delta)

# ---------- 文件读取 ----------
def find_header(row_headers: List[str], cands: List[str]) -> Optional[str]:
    lower = {h.lower(): h for h in row_headers}
    for name in cands:
        if name.lower() in lower: return lower[name.lower()]
    return None

def read_csv_generic(path: str, colmap: Dict[str, List[str]], limit_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.isfile(path): return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try: headers = next(reader)
        except StopIteration: return []
        headers = [h.strip() for h in headers]
        idx_map: Dict[str, int] = {}
        for canon, cands in colmap.items():
            h = find_header(headers, cands)
            if h is not None: idx_map[canon] = headers.index(h)
        count = 0
        for row in reader:
            if len(row) != len(headers):
                row = (row + [""] * (len(headers) - len(row)))[: len(headers)]
            rec: Dict[str, Any] = {}
            for canon, idx in idx_map.items():
                rec[canon] = row[idx]
            out.append(rec)
            count += 1
            if limit_rows is not None and count >= limit_rows: break
    return out

def read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except Exception: return {}

def load_data_with_fallback(data_dir: str) -> Dict[str, Any]:
    base_dir = data_dir if os.path.isdir(data_dir) else os.path.join(BASE_DIR, "data")
    hvac_t_path = os.path.join(base_dir, "hvac_telemetry.csv")
    weather_path = os.path.join(base_dir, "weather_forecast.csv")
    price_path = os.path.join(base_dir, "market_price.csv")
    ef_path = os.path.join(base_dir, "grid_ef.csv")
    load_path = os.path.join(base_dir, "load_forecast.csv")
    demand_cfg_path = os.path.join(base_dir, "demand_window_config.json")
    plant_master_path = os.path.join(base_dir, "plant_master.json")
    effmap_path = os.path.join(base_dir, "plant_efficiency_map.csv")

    telemetry = read_csv_generic(hvac_t_path, HEADER_CANDIDATES, limit_rows=None)
    weather   = read_csv_generic(weather_path, WEATHER_HEADER_CANDS, limit_rows=None)
    price     = read_csv_generic(price_path, PRICE_HEADER_CANDS, limit_rows=None)
    ef        = read_csv_generic(ef_path, EF_HEADER_CANDS, limit_rows=None)
    loadf     = read_csv_generic(load_path, LOAD_HEADER_CANDS, limit_rows=None)
    demand_cfg = read_json(demand_cfg_path)
    plant_master = read_json(plant_master_path)
    effmap    = read_csv_generic(effmap_path, EFFMAP_HEADER_CANDS, limit_rows=None)

    return {
        "telemetry": telemetry, "weather": weather, "price": price, "ef": ef, "loadf": loadf,
        "demand_cfg": demand_cfg, "plant_master": plant_master, "effmap": effmap,
        "base_dir": base_dir,
        "paths": {
            "telemetry": hvac_t_path, "weather": weather_path, "price": price_path, "ef": ef_path,
            "loadf": load_path, "demand_cfg": demand_cfg_path, "plant_master": plant_master_path,
            "effmap": effmap_path,
        }
    }

# ---------- 需量窗口 ----------
class DemandWindow:
    def __init__(self, p_cap_soft: float, penalty_yuan_per_kw: float, window_min: int = 15):
        self.p_cap = float(p_cap_soft)
        self.penalty_rate = float(penalty_yuan_per_kw)
        self.window_min = int(window_min)

    def rolling_avg_kw(self, recent_kw_series: List[Tuple[datetime, float]]) -> float:
        if not recent_kw_series: return 0.0
        cutoff = (recent_kw_series[-1][0] - timedelta(minutes=self.window_min))
        vals = [kw for ts, kw in recent_kw_series if ts > cutoff]
        if not vals: return 0.0
        return float(sum(vals) / len(vals))

    def soft_penalty(self, p_roll: float, delta_p: float = 50.0) -> float:
        excess = (p_roll - self.p_cap) / max(1.0, delta_p)
        return softplus(excess) * self.penalty_rate

# ---------- 效率映射 ----------
class EffEstimator:
    def __init__(self, effmap_rows: List[Dict[str, Any]]):
        self.table: List[Tuple[float, float, float]] = []
        for r in effmap_rows:
            try:
                chws = safe_float(r.get("CHWS_C")); wb = safe_float(r.get("WB_C")); cop = safe_float(r.get("COP"))
                if chws > 0 and cop > 0: self.table.append((chws, wb, cop))
            except Exception: continue

    def cop(self, chws_C: float, wb_C: float) -> float:
        if self.table:
            best = None; best_d = 1e9
            for chws, wb, cop in self.table:
                d = abs(chws - chws_C) + abs(wb - wb_C)
                if d < best_d: best_d, best = d, cop
            return max(1.5, float(best))
        cop = 4.5 - 0.2 * (chws_C - 7.0) - 0.05 * max(0.0, wb_C - 25.0)
        return max(1.8, min(6.0, cop))

# ---------- 24h 参考轨迹（MPC 启发式，保持不变） ----------
class SetpointPlanner:
    def __init__(self, plant_cfg: Dict[str, Any], demand_cfg: Dict[str, Any], eff: EffEstimator):
        self.plant = plant_cfg or {}; self.demand = demand_cfg or {}; self.eff = eff
        sp_limits = self.plant.get("setpoints", {})
        self.min_chws = float(sp_limits.get("chws_C", {}).get("min", 6.0))
        self.max_chws = float(sp_limits.get("chws_C", {}).get("max", 9.0))
        self.min_sat  = float(sp_limits.get("sat_C", {}).get("min", 12.0))
        self.max_sat  = float(sp_limits.get("sat_C", {}).get("max", 18.0))
        self.min_sp   = float(sp_limits.get("static_pressure_Pa", {}).get("min", 500))
        self.max_sp   = float(sp_limits.get("static_pressure_Pa", {}).get("max", 1200))
        self.ramp_chws = float(sp_limits.get("chws_C", {}).get("ramp_C_per_15min", 0.5))
        self.ramp_sat  = float(sp_limits.get("sat_C", {}).get("ramp_C_per_15min", 0.6))
        self.ramp_sp   = float(sp_limits.get("static_pressure_Pa", {}).get("ramp_Pa_per_15min", 50))
        w_cfg = self.demand.get("weights", {})
        self.w_cost = float(w_cfg.get("cost_weight", 1.0))
        self.w_carbon = float(w_cfg.get("carbon_weight", 0.4))
        self.w_comfort = float(w_cfg.get("comfort_weight", 0.8))
        self.delta = dict(DEFAULT_RESIDUAL_DELTA)

    def plan_24h(self, start_ts: datetime, step_min: int,
                 price_path: List[Tuple[datetime, float]],
                 ef_path: List[Tuple[datetime, float]],
                 load_path: List[Tuple[datetime, float]],
                 weather_path: List[Tuple[datetime, float, float, float]],
                 last_targets: Dict[str, float]) -> List[Dict[str, Any]]:
        horizon = 24 * 60 // step_min
        times = [start_ts + timedelta(minutes=i * step_min) for i in range(horizon)]
        def interp(ts, series):
            if not series: return 0.0
            best = min(series, key=lambda x: abs((x[0]-ts).total_seconds()))
            return float(best[1])
        def interp_weather(ts, series):
            if not series: return (30.0, 25.0, 70.0)
            best = min(series, key=lambda x: abs((x[0]-ts).total_seconds()))
            return (float(best[1]), float(best[2]), float(best[3]))
        out: List[Dict[str, Any]] = []
        prev_chws = float(last_targets.get("CHWS_set", 7.5))
        prev_sat  = float(last_targets.get("SAT_set", 14.0))
        prev_sp   = float(last_targets.get("SP_set", 800.0))
        all_prices = [p for _, p in price_path] or [0.8]
        all_efs    = [e for _, e in ef_path] or [0.7]
        all_loads  = [l for _, l in load_path] or [6000.0]
        def _med(v):
            s = sorted(v); mid = len(s)//2
            return (s[mid-1] + s[mid]) / 2 if len(s) >= 2 else s[0]
        p_med, e_med, l_med = _med(all_prices), _med(all_efs), _med(all_loads)
        def score(x, m):
            if m <= 0: return 0.0
            return (x - m) / max(1e-3, m)

        for ts in times:
            price = interp(ts, price_path) or 0.8
            ef    = interp(ts, ef_path) or 0.7
            load_kw = max(0.0, interp(ts, load_path))
            db, wb, rh = interp_weather(ts, weather_path)
            dp = dew_point_C(db, rh)
            cost_score = self.w_cost * score(price, p_med) + self.w_carbon * score(ef, e_med)
            load_score = score(load_kw, l_med)
            humidity_risk = max(0.0, (dp - 18.0) / 6.0)

            base_chws = 7.5 + 0.6 * (-load_score) + 0.8 * cost_score - 0.5 * humidity_risk
            base_sat  = 14.0 + 0.5 * (-load_score) + 0.7 * cost_score - 0.6 * humidity_risk
            base_sp   = 800.0 + 80.0 * max(0.0, load_score) - 60.0 * max(0.0, -load_score)

            tgt_chws = clip_rate(prev_chws, float(np.clip(base_chws, self.min_chws, self.max_chws)), self.ramp_chws)
            tgt_sat  = clip_rate(prev_sat,  float(np.clip(base_sat,  self.min_sat,  self.max_sat)),  self.ramp_sat)
            tgt_sp   = clip_rate(prev_sp,   float(np.clip(base_sp,   self.min_sp,   self.max_sp)),   self.ramp_sp)
            prev_chws, prev_sat, prev_sp = tgt_chws, tgt_sat, tgt_sp

            reasons = []
            if cost_score > 0.2: reasons.append("high_cost_or_ef")
            if humidity_risk > 0.2: reasons.append("dewpoint_risk")
            if load_score < -0.2: reasons.append("light_load_saving")
            if load_score >  0.2: reasons.append("heavy_load_capacity")

            out.append({
                "ts": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "CHWS_set": round(tgt_chws, 2),
                "SAT_set":  round(tgt_sat, 2),
                "SP_set":   float(int(tgt_sp)),
                "reasons": reasons,
                "db_C": round(db, 2), "wb_C": round(wb, 2), "rh_pct": round(rh, 1),
                "price": round(price, 4), "ef": round(ef, 4), "load_kw": round(load_kw, 1)
            })
        return out

# ---------- IQL 残差策略（推理） ----------
class IQLPolicy:
    """
    纯 numpy 的 MLP 前向推理：
    - 输入：固定特征向量（可被 policy.bin 的 norm.mean/std 归一）
    - 输出：残差 [dCHWS, dSAT, dSP]（随后会按 δ 安全域裁剪）
    - 若 policy.bin 缺失/结构不符 -> 抛异常，由 ResidualPolicy 捕获并回退到启发式
    """
    DEFAULT_INPUT_KEYS = [
        "price", "ef", "db_C", "rh_pct", "dew_point_C",
        "sat_ref", "chws_ref", "sp_ref",
        "demand_tight", "dr_mode"
    ]
    DEFAULT_OUTPUT_KEYS = ["dCHWS", "dSAT", "dSP"]

    def __init__(self, policy_path: str = POLICY_BIN, meta_path: str = POLICY_META):
        if not os.path.isfile(policy_path):
            raise FileNotFoundError("policy.bin not found")
        with open(policy_path, "r", encoding="utf-8") as f:
            try:
                pol = json.load(f)
            except Exception as e:
                raise RuntimeError(f"policy.bin parse error: {e}")

        arch = pol.get("arch", {})
        self.layers = arch.get("layers", [])
        if not self.layers or len(self.layers) < 2:
            raise RuntimeError("invalid arch.layers in policy.bin")
        ws = pol.get("weights", [])
        if len(ws) != (len(self.layers) - 1):
            raise RuntimeError("weights length mismatch arch")

        # 加载权重
        self.W: List[Any] = []
        self.b: List[Any] = []
        for i, layer in enumerate(ws):
            W = layer.get("W"); b = layer.get("b")
            if W is None or b is None: raise RuntimeError(f"missing W/b at layer {i}")
            self.W.append(np.array(W))
            self.b.append(np.array(b))

        # 归一化参数（可选）
        self.mean = pol.get("norm", {}).get("mean", {})
        self.std  = pol.get("norm", {}).get("std", {})
        # 键名
        self.input_keys = pol.get("input_keys", self.DEFAULT_INPUT_KEYS)
        self.output_keys = pol.get("output_keys", self.DEFAULT_OUTPUT_KEYS)

    def _norm(self, x: Dict[str, float]) -> List[float]:
        vec = []
        for k in self.input_keys:
            v = float(x.get(k, 0.0))
            m = safe_float(self.mean.get(k), 0.0)
            s = safe_float(self.std.get(k), 1.0)
            if s <= 0: s = 1.0
            vec.append((v - m) / s)
        return vec

    @staticmethod
    def _relu(v: Any):
        return np.array([max(0.0, float(x)) for x in v])

    def forward(self, feat: Dict[str, float]) -> Dict[str, float]:
        x = np.array(self._norm(feat))
        for i in range(len(self.W)):
            x = np.dot(self.W[i], x) + self.b[i]
            if i < len(self.W) - 1:
                x = self._relu(x)
        # 输出对齐
        out = {}
        for i, k in enumerate(self.output_keys):
            out[k] = float(x[i]) if i < len(x) else 0.0
        return out

# ---------- 残差策略（IQL 优先，启发式兜底） ----------
class ResidualPolicy:
    def __init__(self, delta: Dict[str, float]):
        self.delta = delta or DEFAULT_RESIDUAL_DELTA
        self.backend = "heuristic_fallback"
        self.iql: Optional[IQLPolicy] = None
        # 尝试加载 IQL 策略
        try:
            self.iql = IQLPolicy(POLICY_BIN, POLICY_META)
            self.backend = "IQL"
        except Exception:
            self.iql = None
            self.backend = "heuristic_fallback"

    def decide(self, context: Dict[str, Any]) -> Dict[str, float]:
        ref = context.get("ref", {})
        price = safe_float(context.get("price"), 0.8)
        ef    = safe_float(context.get("ef"), 0.7)
        db    = safe_float(context.get("db_C"), 30.0)
        rh    = safe_float(context.get("rh_pct"), 70.0)
        dp    = dew_point_C(db, rh)
        sat_ref = safe_float(ref.get("SAT_set"), 14.0)
        chws_ref= safe_float(ref.get("CHWS_set"), 7.5)
        sp_ref  = safe_float(ref.get("SP_set"), 800.0)
        dr_mode = bool(context.get("dr_mode", False))
        demand_tight = bool(context.get("demand_tight", False))

        # IQL 推理（若可用）
        if self.iql is not None:
            try:
                feat = {
                    "price": price, "ef": ef, "db_C": db, "rh_pct": rh, "dew_point_C": dp,
                    "sat_ref": sat_ref, "chws_ref": chws_ref, "sp_ref": sp_ref,
                    "demand_tight": 1.0 if demand_tight else 0.0,
                    "dr_mode": 1.0 if dr_mode else 0.0
                }
                y = self.iql.forward(feat)
                # 将网络输出裁剪到 δ 安全域；DR/需量紧张折减
                scale = 0.5 if (dr_mode or demand_tight) else 1.0
                return {
                    "dCHWS": float(np.clip(y.get("dCHWS", 0.0), -self.delta["CHWS_C"], self.delta["CHWS_C"]) * scale),
                    "dSAT":  float(np.clip(y.get("dSAT",  0.0), -self.delta["SAT_C"],  self.delta["SAT_C"])  * scale),
                    "dSP":   float(np.clip(y.get("dSP",   0.0), -self.delta["SP_Pa"],  self.delta["SP_Pa"])  * scale),
                }
            except Exception:
                # 若推理异常，回退启发式
                pass

        # 启发式残差（兜底，与上一版一致）
        cost_s = (price - 0.8) / max(0.2, 0.8)
        ef_s   = (ef    - 0.7) / max(0.2, 0.7)
        dew_risk = max(0.0, (dp - (sat_ref - 2.5)) / 3.0)
        k_cost, k_dew = 0.35, 0.6
        res_chws = +k_cost * (max(0.0, cost_s) + max(0.0, ef_s)) - k_dew * dew_risk
        res_sat  = +k_cost * (max(0.0, cost_s) + max(0.0, ef_s)) - k_dew * dew_risk
        res_sp   = -k_cost * (max(0.0, cost_s) + max(0.0, ef_s)) + 0.8 * k_dew * dew_risk
        scale = 0.5 if (dr_mode or demand_tight) else 1.0
        return {
            "dCHWS": float(np.clip(res_chws * self.delta["CHWS_C"], -self.delta["CHWS_C"], self.delta["CHWS_C"]) * scale),
            "dSAT":  float(np.clip(res_sat  * self.delta["SAT_C"],  -self.delta["SAT_C"],  self.delta["SAT_C"])  * scale),
            "dSP":   float(np.clip(res_sp   * self.delta["SP_Pa"],   -self.delta["SP_Pa"],  self.delta["SP_Pa"])  * scale),
        }

# ---------- 安全屏蔽 ----------
class SafetyShield:
    def __init__(self, plant_cfg: Dict[str, Any], demand_cfg: Dict[str, Any]):
        self.plant = plant_cfg or {}; self.demand = demand_cfg or {}
        sp_limits = self.plant.get("setpoints", {})
        self.min_chws = float(sp_limits.get("chws_C", {}).get("min", 6.0))
        self.max_chws = float(sp_limits.get("chws_C", {}).get("max", 9.0))
        self.min_sat  = float(sp_limits.get("sat_C", {}).get("min", 12.0))
        self.max_sat  = float(sp_limits.get("sat_C", {}).get("max", 18.0))
        self.min_sp   = float(sp_limits.get("static_pressure_Pa", {}).get("min", 500))
        self.max_sp   = float(sp_limits.get("static_pressure_Pa", {}).get("max", 1200))
        self.ramp_chws = float(sp_limits.get("chws_C", {}).get("ramp_C_per_15min", 0.5))
        self.ramp_sat  = float(sp_limits.get("sat_C", {}).get("ramp_C_per_15min", 0.6))
        self.ramp_sp   = float(sp_limits.get("static_pressure_Pa", {}).get("ramp_Pa_per_15min", 50))
        limits = self.demand.get("limits", {})
        self.pcc_limit_kw = float(limits.get("pcc_limit_kw", self.demand.get("pcc_limit_kW", 14000.0)))
        self.soft_cap_kw  = float(self.demand.get("soft_cap_kW", limits.get("plant_soft_cap_kw", 12500.0)))

    def apply(self, prev_targets: Dict[str, float], proposed: Dict[str, float], context: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        reasons: List[str] = []
        chws = float(np.clip(proposed.get("CHWS_set", prev_targets.get("CHWS_set", 7.5)), self.min_chws, self.max_chws))
        sat  = float(np.clip(proposed.get("SAT_set",  prev_targets.get("SAT_set", 14.0)), self.min_sat,  self.max_sat))
        sp   = float(np.clip(proposed.get("SP_set",   prev_targets.get("SP_set", 800.0)), self.min_sp,   self.max_sp))
        dp = float(context.get("dew_point_C", 16.0))
        if (sat - dp) < 2.5:
            sat = max(sat, dp + 2.6); reasons.append("dewpoint_protection")
        chw_flow = float(context.get("CHW_flow", 0.0))
        g_min = float(context.get("G_min", 0.3))
        if chw_flow > 0 and chw_flow < g_min:
            chws = max(chws, prev_targets.get("CHWS_set", chws)); reasons.append("min_flow_protection")
        if bool(context.get("demand_tight", False)):
            chws = max(chws, prev_targets.get("CHWS_set", chws))
            sp   = min(sp,   prev_targets.get("SP_set", sp))
            sat  = max(sat,  prev_targets.get("SAT_set", sat))
            reasons.append("demand_window_protection")
        chws = clip_rate(prev_targets.get("CHWS_set", chws), chws, self.ramp_chws)
        sat  = clip_rate(prev_targets.get("SAT_set",  sat),  sat,  self.ramp_sat)
        sp   = clip_rate(prev_targets.get("SP_set",   sp),   sp,   self.ramp_sp)
        return {"CHWS_set": round(chws, 2), "SAT_set": round(sat, 2), "SP_set": float(int(sp))}, reasons

# ---------- 状态持久化 ----------
def load_state(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {"last_targets": {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0}, "recent_kw": []}
    with open(path, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except Exception: return {"last_targets": {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0}, "recent_kw": []}

def save_state(path: str, state: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ---------- JSONL 输出 ----------
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------- 辅助 ----------
def to_series(rows: List[Dict[str, Any]], ts_key: str, val_keys: List[str]) -> List[Tuple[datetime, Any]]:
    out: List[Tuple[datetime, Any]] = []
    for r in rows:
        ts = parse_ts_any(str(r.get(ts_key, "")).strip())
        if ts is None: continue
        if len(val_keys) == 1: out.append((ts, safe_float(r.get(val_keys[0]))))
        else: out.append((ts, tuple(safe_float(r.get(k)) for k in val_keys)))
    out.sort(key=lambda x: x[0])
    return out

def compute_demand_context(state: Dict[str, Any], demand_cfg: Dict[str, Any], pcc_kw_now: float, now_ts: datetime) -> Dict[str, Any]:
    recent = state.get("recent_kw", [])
    def _p(ts_s: str):
        try: return datetime.strptime(ts_s, "%Y-%m-%dT%H:%M:%S")
        except Exception: return now_ts - timedelta(minutes=60)
    recent = [(_p(ts), float(kw)) for ts, kw in recent if isinstance(ts, str)]
    recent = [(ts, kw) for ts, kw in recent if (now_ts - ts).total_seconds() <= 3600]
    recent.append((now_ts, float(pcc_kw_now)))
    state["recent_kw"] = [(ts.strftime("%Y-%m-%dT%H:%M:%S"), kw) for ts, kw in recent]

    soft_cap = float(demand_cfg.get("soft_cap_kW", demand_cfg.get("limits", {}).get("plant_soft_cap_kw", 12500.0)))
    penalty  = float(demand_cfg.get("penalty_yuan_per_kW", 80.0))
    window_min = int(demand_cfg.get("granularity_min", 15))
    dw = DemandWindow(soft_cap, penalty, window_min)
    p_roll = dw.rolling_avg_kw(recent)
    tight = (p_roll >= soft_cap * 0.98)
    return {"p_roll_kw": float(p_roll), "p_cap_kw": float(soft_cap), "demand_tight": bool(tight), "soft_penalty_yuan": float(dw.soft_penalty(p_roll))}

def build_command_payload(final_targets: Dict[str, float], ttl_s: int = 60) -> Dict[str, Any]:
    nonce = gen_nonce()
    expires_at = (datetime.utcnow() + timedelta(seconds=ttl_s)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"write_enable": True, "nonce": nonce, "expires_at": expires_at,
            "cmd": {"CHWS_set_cmd": final_targets["CHWS_set"], "SAT_set_cmd": final_targets["SAT_set"], "SP_set_cmd": final_targets["SP_set"]} }

# ---------- 业务流程：计划/决策/自检 ----------
def run_plan_once(data_dir: str, out_path: str) -> Dict[str, Any]:
    data = load_data_with_fallback(data_dir)
    plant = data["plant_master"]; demand_cfg = data["demand_cfg"]
    price_series = to_series(data["price"], "ts", ["price_yuan_per_kwh"])
    ef_series    = to_series(data["ef"],    "ts", ["ef_kg_per_kwh"])
    load_series  = to_series(data["loadf"], "ts", ["load_kw"])
    weather_series = to_series(data["weather"], "ts", ["DB_C", "WB_C", "RH_pct"])

    step_min = int(demand_cfg.get("granularity_min", 15)); start_ts = datetime.utcnow()
    eff = EffEstimator(data["effmap"]); state = load_state(STATE_PATH)
    last_targets = state.get("last_targets", {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0})
    planner = SetpointPlanner(plant, demand_cfg, eff)
    plan = planner.plan_24h(start_ts, step_min, price_series, ef_series, load_series, weather_series, last_targets)
    record = {"ts": now_utc_iso(), "module": MODULE_NAME, "kind": "plan", "step_min": step_min,
              "plan_start": start_ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "plan": plan, "source_files": data["paths"],
              "audit": {"version": 2, "from": "api.plan"}}
    append_jsonl(out_path, record); return record

def run_decide_once(data_dir: str, out_path: str) -> Dict[str, Any]:
    data = load_data_with_fallback(data_dir)
    plant = data["plant_master"]; demand_cfg = data["demand_cfg"]
    tel_rows = data["telemetry"]; tel_ts_series = to_series(tel_rows, "ts", ["PCC_kW"])
    pcc_kw_now = tel_ts_series[-1][1] if tel_ts_series else 0.0
    weather_now = to_series(data["weather"], "ts", ["DB_C", "WB_C", "RH_pct"])
    if weather_now: db, wb, rh = weather_now[-1][1]
    else: db, wb, rh = 30.0, 25.0, 70.0
    dp = dew_point_C(db, rh)

    state = load_state(STATE_PATH); now_ts = datetime.utcnow()
    dem_ctx = compute_demand_context(state, demand_cfg, pcc_kw_now, now_ts)

    step_min = int(demand_cfg.get("granularity_min", 15))
    last_targets = state.get("last_targets", {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0})
    price_series = to_series(data["price"], "ts", ["price_yuan_per_kwh"])
    ef_series    = to_series(data["ef"],    "ts", ["ef_kg_per_kwh"])
    load_series  = to_series(data["loadf"], "ts", ["load_kw"])
    weather_series = to_series(data["weather"], "ts", ["DB_C", "WB_C", "RH_pct"])
    eff = EffEstimator(data["effmap"])
    planner = SetpointPlanner(plant, demand_cfg, eff)
    plan = planner.plan_24h(now_ts, step_min, price_series, ef_series, load_series, weather_series, last_targets)
    ref0 = plan[0] if plan else {"CHWS_set": 7.5, "SAT_set": 14.0, "SP_set": 800.0, "price": 0.8, "ef": 0.7, "db_C": db, "rh_pct": rh}

    residual = ResidualPolicy(planner.delta)
    ctx = {"ref": ref0, "price": ref0.get("price", 0.8), "ef": ref0.get("ef", 0.7),
           "db_C": db, "rh_pct": rh, "dr_mode": False, "demand_tight": dem_ctx["demand_tight"]}
    d = residual.decide(ctx)

    proposed = {"CHWS_set": float(np.clip(ref0["CHWS_set"] + d["dCHWS"], planner.min_chws, planner.max_chws)),
                "SAT_set":  float(np.clip(ref0["SAT_set"]  + d["dSAT"],   planner.min_sat,  planner.max_sat)),
                "SP_set":   float(np.clip(ref0["SP_set"]   + d["dSP"],    planner.min_sp,   planner.max_sp))}

    shield = SafetyShield(plant, demand_cfg)
    safety_ctx = {"dew_point_C": dp, "CHW_flow": safe_float((tel_rows[-1] if tel_rows else {}).get("CHW_flow"), 0.0),
                  "G_min": 0.3, "demand_tight": dem_ctx["demand_tight"]}
    final_targets, reasons = shield.apply(last_targets, proposed, safety_ctx)
    cmd_payload = build_command_payload(final_targets, ttl_s=60)

    state["last_targets"] = final_targets; save_state(STATE_PATH, state)

    record = {
        "ts": now_utc_iso(), "module": MODULE_NAME, "kind": "decision",
        "inputs": {"pcc_kw_now": pcc_kw_now, "db_C": db, "wb_C": wb, "rh_pct": rh, "dew_point_C": round(dp, 2), "demand_window": dem_ctx},
        "reference": ref0, "residual": d, "proposed": proposed, "masks": reasons,
        "final_action": final_targets, "command_payload": cmd_payload, "source_files": data["paths"],
        "audit": {"version": 2, "from": "api.decide", "rl_backend": residual.backend}
    }
    append_jsonl(out_path, record)
    return record

def run_self_test(data_dir: str, out_path: str) -> Dict[str, Any]:
    plan_rec = run_plan_once(data_dir, out_path)
    dec_rec  = run_decide_once(data_dir, out_path)
    print_backend = dec_rec.get("audit", {}).get("rl_backend", "unknown")
    return {"ok": True, "plan_len": len(plan_rec.get("plan", [])), "decision_keys": list(dec_rec.keys()), "rl_backend": print_backend}

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="HVAC cooling setpoint linkage API (IQL-enabled)")
    parser.add_argument("--data-dir", type=str, default="/mnt/data", help="数据目录（优先 /mnt/data），回退 hvac_cooling/data")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="JSONL 输出路径")
    parser.add_argument("--plan", action="store_true", help="生成 24h 参考轨迹（写 JSONL）")
    parser.add_argument("--decide", action="store_true", help="当前一步决策（写 JSONL）")
    parser.add_argument("--self-test", action="store_true", help="自检（plan+decide）")
    args = parser.parse_args()

    ensure_dir(ARTIFACT_DIR)
    try:
        if args.self_test:
            r = run_self_test(args.data_dir, args.out)
            print("SELF-TEST OK:", json.dumps(r, ensure_ascii=False)); return 0
        if args.plan:
            r = run_plan_once(args.data_dir, args.out)
            print("PLAN OK:", len(r.get("plan", []))); return 0
        if args.decide:
            r = run_decide_once(args.data_dir, args.out)
            print("DECIDE OK:", json.dumps({"final": r.get("final_action", {}), "rl_backend": r.get("audit",{}).get("rl_backend")}, ensure_ascii=False)); return 0
        r = run_decide_once(args.data_dir, args.out)
        print("DECIDE OK:", json.dumps({"final": r.get("final_action", {}), "rl_backend": r.get("audit",{}).get("rl_backend")}, ensure_ascii=False))
        return 0
    except Exception as e:
        print("ERROR:", repr(e)); return 2

if __name__ == "__main__":
    sys.exit(main())
