# -*- coding: utf-8 -*-
"""
模块 G｜岸桥（QC）作业节拍与待机 —— 数字孪生 + 计划层（规则/MPC 简式）+ 执行层屏蔽（Shielding）
----------------------------------------------------------------
- 仅用 Python 标准库 + 少量 numpy；CSV/JSON 鲁棒摄取（列名候选集 + 时间戳多口径解析）
- 统一 JSONL 输出：qc_step / qc_episode_summary / policy_update（预留）
- 自检命令：
  python -m app.services.rl_model.port_G_qc_mvp.module_g --self-check --qc-id "" --dt-min 5 --horizon 144 \
    --sleep-every 1000 --sleep-sec 60
"""

import os, sys, csv, json, time, math, argparse, random
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
import numpy as np

# ------------------------------
# 路径与默认文件
# ------------------------------
MODULE_DIR = os.path.dirname(__file__)
DATA_DIR_DEFAULT = os.path.join(MODULE_DIR, "data")
DEFAULT_JSONL = os.path.join(MODULE_DIR, "policy_evaluate_history.jsonl")
DQ_JSONL = os.path.join(MODULE_DIR, "data_quality.jsonl")

# ------------------------------
# 配置与默认参数（与规范一致）
# ------------------------------
DEFAULTS = {
    "dt_min": 5,
    "horizon": 144,  # 12小时
    "price_yuan": 0.85,
    "ef_kg_per_kWh": 0.50,
    "rho_regen": 0.2,  # 回收kWh的等价价值 (¥/kWh)
    "alpha_idle": 1.0,   # IdleEnergy权重（单位约等于 ¥/kWh）
    "beta_sla": 40.0,    # SLA/吞吐罚（船时/里程碑风险最高）
    "gamma_switch": 1.0, # 切换成本
    "eta_thermal": 1.0,
    "zeta_peak": 1.0,    # 峰罚系数（需量罚金梯度对齐时会被外部覆盖）
    "soft_cap_guard_kW": 200.0,  # 接近软上限的缓冲带
    "thermal_theta1_K": 10.0,
    "thermal_theta0_K": 5.0,
    "T_target": 75.0,    # 热惩罚目标温度
    # SLA 动态目标（队列→目标放大），参考 F 模块：q_lo=1, q_hi=4
    "q_lo": 1.0,
    "q_hi": 4.0,
    # Eco 档 → 功率上限映射（可站点标定）
    "eco_power_pct": {
        "normal": 1.00, "ecoL1": 0.95, "ecoL2": 0.85, "ecoL3": 0.75
    },
    # Standby 档
    "idle_auto_off_min": 8.0,  # 软关断阈值
    # 残差带（文件3/3在线微调会用到，这里只用于参数合法性裁剪）
    "res_band_power": 0.10,     # ±10% 额定
    "res_band_pace":  0.10,     # ±10% 节拍
    "res_band_idle":  5.0,      # ±5 min
    # 需量软上限（如无 grid_meter 则使用 rated 合理估）
    "pcc_soft_cap_kW": None,
    "penalty_yuan_per_kW": 0.0,  # 若站端提供需量罚金梯度可覆盖
}

# 列名候选集（鲁棒摄取用）
COLS = {
    "ts": ["ts_utc", "timestamp", "time_utc", "ts", "time"],
    "qc_id": ["qc_id", "crane_id"],
    "state": ["state"],
    "mode": ["mode"],
    "hoist_speed": ["hoist_speed%", "hoist_speed"],
    "trolley_speed": ["trolley_speed%", "trolley_speed"],
    "gantry_speed": ["gantry_speed%", "gantry_speed"],
    "power_kW": ["power_kW", "p_kw", "p"],
    "energy_kWh": ["energy_kWh", "e_kwh"],
    "temp_motor": ["temp_motor_C", "temp_motor"],
    "temp_inverter": ["temp_inverter_C", "temp_inverter"],
    "start_stop_event": ["start_stop_event", "start_stop"],
    "regen_kWh": ["regen_kWh", "regen"],
    "cycle_time_s_median": ["cycle_time_s_median", "cycle_time_s"],
    "moves_5min": ["moves_5min", "moves"],
    "bay_id": ["bay_id"],
    "interference_flag": ["interference_flag", "interference"],
    "wind_mps": ["wind_mps", "wind_ms", "wind"],
    "sway_deg": ["sway_deg", "sway"],
    # price/ef
    "price_p50": ["price_yuan_kWh_p50", "price_p50"],
    "price_p90": ["price_p90"],
    "ef_p50": ["ef_kg_kWh_p50", "ef_p50"],
    "ef_p90": ["ef_p90"],
    # forecast
    "gmph_p50": ["gmph_p50"],
    "gmph_p90": ["gmph_p90"],
    "queue_p50": ["queue_len_p50", "queue_p50", "queue"],
    "queue_p90": ["queue_len_p90", "queue_p90"],
    "next_lashing_utc": ["next_lashing_utc"],
    "next_hatch_utc": ["next_hatch_utc"],
    # PCC
    "pcc_kW": ["P_pcc_kW", "pcc_kw", "pcc"],
    # master / vessel plan
    "rated_kW": ["rated_kW"],
    "hoist_kW": ["hoist_kW"],
    "trolley_kW": ["trolley_kW"],
    "gantry_kW": ["gantry_kW"],
    "regen_capable": ["regen_capable"],
    "eco_levels": ["eco_levels"],
    "min_on_min": ["min_on_min"],
    "min_off_min": ["min_off_min"],
    "accel_limit": ["accel_limit"],
    "jerk_limit": ["jerk_limit"],
    "sway_limit_deg": ["sway_limit_deg"],
    "wind_cutout_mps": ["wind_cutout_mps"],
    "temp_redline_C": ["temp_redline_C"],
    # vessel
    "vessel_id": ["vessel_id"],
    "berth_id": ["berth_id"],
    "ata_utc": ["ata_utc"],
    "atd_utc_plan": ["atd_utc_plan"],
    "target_gmph": ["target_gmph"],
    "qc_assigned": ["qc_assigned"],
    "bays_seq": ["bays_seq"],
    "lashing_windows": ["lashing_windows"],
}

# ------------------------------
# 通用：时间戳解析 + CSV 读取 + 取列工具
# ------------------------------
def _log_dq(kind: str, msg: str, payload: Optional[Dict[str, Any]] = None):
    """数据质量日志 JSONL：读取失败/字段缺失/数值容错"""
    try:
        os.makedirs(os.path.dirname(DQ_JSONL), exist_ok=True)
        with open(DQ_JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": int(time.time()), "kind": kind, "msg": msg, "payload": payload or {}}, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _parse_ts_any(s: Any) -> Optional[int]:
    """多口径时间戳解析：支持 epoch秒/毫秒、ISO、'YYYY-mm-dd HH:MM:SS'、带Z/+08:00"""
    if s is None or s == "":
        return None
    try:
        # 数字 epoch
        if isinstance(s, (int, float)):
            x = float(s)
            if x > 1e12:  # 毫秒
                return int(x / 1000)
            return int(x)
        # 字符串
        st = str(s).strip()
        if st.isdigit():
            x = int(st)
            if x > 1e12:
                return int(x / 1000)
            return x
        # ISO
        try:
            dt = datetime.fromisoformat(st.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception:
            pass
        # 常见格式
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                if fmt.endswith("%z") and ("+" not in st and "-" not in st[-6:]):
                    # 无显式时区时按本地再转UTC
                    dt = datetime.strptime(st, fmt.replace("%z",""))
                    return int(dt.replace(tzinfo=timezone.utc).timestamp())
                dt = datetime.strptime(st, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except Exception:
                continue
    except Exception as e:
        _log_dq("ts_parse_error", str(e), {"raw": s})
    return None

def _read_csv_dicts(path: str) -> List[Dict[str, Any]]:
    """读取 CSV 为字典列表；自动去BOM/空白；失败则返回空并记 DQ"""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        _log_dq("file_missing", f"CSV not found: {path}")
        return rows
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            header = next(rd, None)
            if header is None:
                return rows
            header = [h.strip("\ufeff ").strip() for h in header]
            for r in rd:
                d = {}
                for i, h in enumerate(header):
                    if i < len(r):
                        d[h] = r[i].strip()
                rows.append(d)
    except Exception as e:
        _log_dq("csv_read_error", str(e), {"file": path})
    return rows

def _col(r: Dict[str, Any], keys: List[str], default=None):
    """从一行中按候选列取值"""
    for k in keys:
        if k in r and r[k] not in ("", None):
            return r[k]
    return default

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default

def _json_loads_safe(s: str, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

# ------------------------------
# JSONL 记录器（强制清洗 NaN/Inf）
# ------------------------------
class JsonlLogger:
    def __init__(self, path: str = DEFAULT_JSONL):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    @staticmethod
    def _clean_nans(o):
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
        self._fh.write(json.dumps(cleaned, ensure_ascii=False, allow_nan=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

# ------------------------------
# 数据加载（最小可用集合）
# ------------------------------
def load_qc_master(data_root: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(data_root, "qc_master.csv")
    rows = _read_csv_dicts(path)
    out = {}
    for r in rows:
        qc = _col(r, COLS["qc_id"])
        if not qc:
            continue
        out[qc] = {
            "qc_id": qc,
            "rated_kW": _to_float(_col(r, COLS["rated_kW"]), 400.0),
            "hoist_kW": _to_float(_col(r, COLS["hoist_kW"]), 150.0),
            "trolley_kW": _to_float(_col(r, COLS["trolley_kW"]), 150.0),
            "gantry_kW": _to_float(_col(r, COLS["gantry_kW"]), 100.0),
            "regen_capable": _to_int(_col(r, COLS["regen_capable"]), 1),
            "eco_levels": (_col(r, COLS["eco_levels"]) or "L1|L2|L3"),
            "min_on_min": _to_float(_col(r, COLS["min_on_min"]), 10.0),
            "min_off_min": _to_float(_col(r, COLS["min_off_min"]), 5.0),
            "accel_limit": _to_float(_col(r, COLS["accel_limit"]), 0.3),
            "jerk_limit": _to_float(_col(r, COLS["jerk_limit"]), 1.0),
            "sway_limit_deg": _to_float(_col(r, COLS["sway_limit_deg"]), 4.0),
            "wind_cutout_mps": _to_float(_col(r, COLS["wind_cutout_mps"]), 14.0),
            "temp_redline_C": _to_float(_col(r, COLS["temp_redline_C"]), 95.0),
        }
    if not out:
        _log_dq("master_empty", "qc_master empty, use fallback one")
        out["QC_01"] = {
            "qc_id": "QC_01",
            "rated_kW": 400.0,
            "hoist_kW": 150.0,
            "trolley_kW": 150.0,
            "gantry_kW": 100.0,
            "regen_capable": 1,
            "eco_levels": "L1|L2|L3",
            "min_on_min": 10.0,
            "min_off_min": 5.0,
            "accel_limit": 0.3,
            "jerk_limit": 1.0,
            "sway_limit_deg": 4.0,
            "wind_cutout_mps": 14.0,
            "temp_redline_C": 95.0,
        }
    return out

def load_vessel_plan(data_root: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(data_root, "vessel_plan.csv")
    rows = _read_csv_dicts(path)
    out = {}
    for r in rows:
        vid = _col(r, COLS["vessel_id"])
        if not vid:
            continue
        out[vid] = {
            "vessel_id": vid,
            "berth_id": _col(r, COLS["berth_id"]),
            "ata_utc": _parse_ts_any(_col(r, COLS["ata_utc"])),
            "atd_utc_plan": _parse_ts_any(_col(r, COLS["atd_utc_plan"])),
            "target_gmph": _to_float(_col(r, COLS["target_gmph"]), 30.0),
            "qc_assigned": _col(r, COLS["qc_assigned"]) or "",
            "bays_seq": _json_loads_safe(_col(r, COLS["bays_seq"]), []),
            "lashing_windows": _json_loads_safe(_col(r, COLS["lashing_windows"]), []),
        }
    return out

def load_qc_telemetry(data_root: str) -> List[Dict[str, Any]]:
    path = os.path.join(data_root, "qc_telemetry.csv")
    rows = _read_csv_dicts(path)
    out = []
    for r in rows:
        ts = _parse_ts_any(_col(r, COLS["ts"]))
        if ts is None:
            continue
        out.append({
            "ts": ts,
            "qc_id": _col(r, COLS["qc_id"]),
            "state": _col(r, COLS["state"]) or "idle",
            "mode": _col(r, COLS["mode"]) or "ecoL1",
            "hoist_speed": _to_float(_col(r, COLS["hoist_speed"]), 0.0),
            "trolley_speed": _to_float(_col(r, COLS["trolley_speed"]), 0.0),
            "gantry_speed": _to_float(_col(r, COLS["gantry_speed"]), 0.0),
            "power_kW": _to_float(_col(r, COLS["power_kW"]), 0.0),
            "energy_kWh": _to_float(_col(r, COLS["energy_kWh"]), 0.0),
            "temp_motor": _to_float(_col(r, COLS["temp_motor"]), float("nan")),
            "temp_inverter": _to_float(_col(r, COLS["temp_inverter"]), float("nan")),
            "start_stop_event": _to_int(_col(r, COLS["start_stop_event"]), 0),
            "regen_kWh": _to_float(_col(r, COLS["regen_kWh"]), 0.0),
            "cycle_time_s_median": _to_float(_col(r, COLS["cycle_time_s_median"]), 0.0),
            "moves_5min": _to_float(_col(r, COLS["moves_5min"]), 0.0),
            "bay_id": _col(r, COLS["bay_id"]),
            "interference_flag": _to_int(_col(r, COLS["interference_flag"]), 0),
            "wind_mps": _to_float(_col(r, COLS["wind_mps"]), float("nan")),
            "sway_deg": _to_float(_col(r, COLS["sway_deg"]), float("nan")),
        })
    out.sort(key=lambda x: x["ts"])
    return out

def load_qc_jobs(data_root: str) -> List[Dict[str, Any]]:
    path = os.path.join(data_root, "qc_jobs.csv")
    rows = _read_csv_dicts(path)
    out = []
    for r in rows:
        st = _parse_ts_any(_col(r, COLS["ts"]) or _col(r, ["start_utc"]))
        et = _parse_ts_any(_col(r, ["end_utc"]))
        out.append({
            "job_id": _col(r, ["job_id"]) or "",
            "vessel_id": _col(r, COLS["vessel_id"]),
            "qc_id": _col(r, COLS["qc_id"]),
            "bay_id": _col(r, COLS["bay_id"]),
            "start_utc": st, "end_utc": et,
            "move_type": _col(r, ["move_type"]) or "",
            "moves": _to_float(_col(r, ["moves"]), 0.0),
            "avg_cycle_time_s": _to_float(_col(r, ["avg_cycle_time_s"]), 0.0),
        })
    return out

def load_tos_forecast(data_root: str) -> List[Dict[str, Any]]:
    path = os.path.join(data_root, "tos_forecast.csv")
    rows = _read_csv_dicts(path)
    out = []
    for r in rows:
        ts = _parse_ts_any(_col(r, COLS["ts"]))
        if ts is None:
            continue
        out.append({
            "ts": ts,
            "vessel_id": _col(r, COLS["vessel_id"]),
            "bay_id": _col(r, COLS["bay_id"]),
            "gmph_p50": _to_float(_col(r, COLS["gmph_p50"]), float("nan")),
            "gmph_p90": _to_float(_col(r, COLS["gmph_p90"]), float("nan")),
            "queue_p50": _to_float(_col(r, COLS["queue_p50"]), float("nan")),
            "queue_p90": _to_float(_col(r, COLS["queue_p90"]), float("nan")),
            "next_lashing_utc": _parse_ts_any(_col(r, COLS["next_lashing_utc"])),
            "next_hatch_utc": _parse_ts_any(_col(r, COLS["next_hatch_utc"])),
        })
    out.sort(key=lambda x: x["ts"])
    return out

def load_price_ef(data_root: str) -> Tuple[List[Tuple[int,float]], List[Tuple[int,float]]]:
    p_rows = _read_csv_dicts(os.path.join(data_root, "market_price.csv"))
    e_rows = _read_csv_dicts(os.path.join(data_root, "grid_ef.csv"))
    price = []
    for r in p_rows:
        ts = _parse_ts_any(_col(r, COLS["ts"]))
        if ts is None:
            continue
        v = _to_float(_col(r, COLS["price_p50"]), DEFAULTS["price_yuan"])
        price.append((ts, v))
    price.sort(key=lambda x:x[0])
    ef = []
    for r in e_rows:
        ts = _parse_ts_any(_col(r, COLS["ts"]))
        if ts is None:
            continue
        v = _to_float(_col(r, COLS["ef_p50"]), DEFAULTS["ef_kg_per_kWh"])
        ef.append((ts, v))
    ef.sort(key=lambda x:x[0])
    return price, ef

def load_meteo(data_root: str) -> List[Tuple[int, float, float]]:
    rows = _read_csv_dicts(os.path.join(data_root, "meteo_sea.csv"))
    out = []
    for r in rows:
        ts = _parse_ts_any(_col(r, COLS["ts"]))
        if ts is None:
            continue
        out.append((ts, _to_float(_col(r, ["wind_mps"]), float("nan")), _to_float(_col(r, ["gust_mps"]), float("nan"))))
    out.sort(key=lambda x:x[0])
    return out

def load_pcc(data_root: str) -> List[Tuple[int, float]]:
    rows = _read_csv_dicts(os.path.join(data_root, "grid_meter.csv"))
    out = []
    for r in rows:
        ts = _parse_ts_any(_col(r, COLS["ts"]))
        if ts is None:
            continue
        out.append((ts, _to_float(_col(r, COLS["pcc_kW"]), float("nan"))))
    out.sort(key=lambda x:x[0])
    return out

# ------------------------------
# 序列对齐与插值（阶梯保持）
# ------------------------------
def _align_series(ts_vec: List[int], series: List[Tuple[int, float]], default: float) -> np.ndarray:
    """将任意时间序列(时刻,值)对齐到 ts_vec（阶梯保持，上一个可用值；无则用默认）"""
    out = np.full((len(ts_vec),), default, dtype=float)
    if not series:
        return out
    j = 0
    cur = default
    for i, t in enumerate(ts_vec):
        while j < len(series) and series[j][0] <= t:
            cur = series[j][1]
            j += 1
        out[i] = cur
    return out

# ------------------------------
# 基线计划层（简式）：按 gmph/队列/里程碑 生成参考 mode/power/idle
# ------------------------------
def _baseline_ref_for_step(gmph_target: float, queue_p50: float) -> Tuple[str, float, float]:
    """
    返回 (mode_ref, pwr_ref_pct, idle_ref_min)
    - 队列低 & gmph 低 → 深节能（ecoL2/L3）+ 更长 idle
    - 队列高 或 gmph 高 → normal/ecoL1 + 较短 idle
    """
    q = queue_p50
    if math.isnan(q):
        q = DEFAULTS["q_lo"]
    if gmph_target >= 40 or q >= DEFAULTS["q_hi"]:
        return ("normal", DEFAULTS["eco_power_pct"]["normal"], max(3.0, DEFAULTS["idle_auto_off_min"] - 3.0))
    if gmph_target >= 32 or q >= (DEFAULTS["q_lo"] + DEFAULTS["q_hi"]) / 2:
        return ("ecoL1", DEFAULTS["eco_power_pct"]["ecoL1"], DEFAULTS["idle_auto_off_min"])
    if gmph_target >= 26:
        return ("ecoL2", DEFAULTS["eco_power_pct"]["ecoL2"], DEFAULTS["idle_auto_off_min"] + 2.0)
    return ("ecoL3", DEFAULTS["eco_power_pct"]["ecoL3"], DEFAULTS["idle_auto_off_min"] + 3.0)

# ------------------------------
# 环境类：推进一步、屏蔽、奖励、日志
# ------------------------------
class QCEnv:
    """
    大白话：
    - 这个类把“数据 → 参考计划 → 屏蔽/奖励 → JSONL”串起来
    - 后续 RL 引擎会用 make_env() 得到这个 env，在它上面做离线/在线训练
    """

    def __init__(self, dt_min: int, horizon_steps: int, qc_id: Optional[str], data_root: str, jsonl_path: str = DEFAULT_JSONL):
        self.dt_min = dt_min
        self.h = horizon_steps
        self.data_root = data_root
        self.log = JsonlLogger(jsonl_path)

        # --- 加载数据 ---
        self.master = load_qc_master(data_root)
        # QC 选择：指定或首个
        self.qc_id = qc_id or (list(self.master.keys())[0] if self.master else "QC_01")
        self.qc_meta = self.master.get(self.qc_id, {})
        self.vessel_plan = load_vessel_plan(data_root)
        self.tele = [r for r in load_qc_telemetry(data_root) if (r.get("qc_id")==self.qc_id or not r.get("qc_id"))]
        self.jobs = [r for r in load_qc_jobs(data_root) if (r.get("qc_id")==self.qc_id or not r.get("qc_id"))]
        self.fcst = load_tos_forecast(data_root)
        self.price_series, self.ef_series = load_price_ef(data_root)
        self.meteo = load_meteo(data_root)
        self.pcc_series = load_pcc(data_root)

        # --- 统一时间轴 ---
        # 选择一个“起点”：先用 telemetry 有数据的第一条，否则用市场/气象
        t0 = None
        for src in (self.tele, [{"ts": x[0]} for x in self.price_series], [{"ts": x[0]} for x in self.meteo]):
            if src:
                t0 = src[0]["ts"] if isinstance(src[0], dict) else src[0][0]
                break
        if t0 is None:
            # 如果都没有，用“现在”向前回推
            t0 = int(time.time()) - 24*3600
        # 对齐到整 dt 边界
        step_sec = self.dt_min * 60
        t0 = (t0 // step_sec) * step_sec
        self.ts = [t0 + i*step_sec for i in range(self.h)]

        # --- 对齐外生量 ---
        self.price = _align_series(self.ts, self.price_series, DEFAULTS["price_yuan"])
        self.ef = _align_series(self.ts, self.ef_series, DEFAULTS["ef_kg_per_kWh"])
        # meteo: wind/gust
        wind_series = [(t, w) for (t,w,_) in self.meteo]
        self.wind = _align_series(self.ts, wind_series, float("nan"))
        # pcc
        self.pcc = _align_series(self.ts, self.pcc_series, float("nan"))

        # --- 对齐观测/预测 ---
        # telemetry 基础功率；没有则用额定功率的一定比例（经验曲线）
        tele_map = {r["ts"]: r for r in self.tele}
        rated = float(self.qc_meta.get("rated_kW", 400.0))
        base = []
        moves_5 = []
        tmotor = []
        tinv = []
        sway = []
        interf = []
        regen = []
        state = []
        mode = []
        wind_m = []
        for t in self.ts:
            r = tele_map.get(t)
            if r:
                base.append(_to_float(r.get("power_kW"), 0.0))
                moves_5.append(_to_float(r.get("moves_5min"), 0.0))
                tmotor.append(_to_float(r.get("temp_motor"), float("nan")))
                tinv.append(_to_float(r.get("temp_inverter"), float("nan")))
                sway.append(_to_float(r.get("sway_deg"), float("nan")))
                interf.append(int(r.get("interference_flag", 0)))
                regen.append(_to_float(r.get("regen_kWh"), 0.0))
                state.append(r.get("state") or "idle")
                mode.append(r.get("mode") or "ecoL1")
                wind_m.append(_to_float(r.get("wind_mps"), float("nan")))
            else:
                # 经验兜底：无遥测时取额定功率的 12% 作为“自然功率”
                base.append(0.12 * rated)
                moves_5.append(0.0)
                tmotor.append(float("nan"))
                tinv.append(float("nan"))
                sway.append(float("nan"))
                interf.append(0)
                regen.append(0.0)
                state.append("idle")
                mode.append("ecoL1")
                wind_m.append(float("nan"))
        # 优先用 meteo 的风；遥测风为空才用遥测
        for i in range(self.h):
            if not math.isnan(self.wind[i]):
                wind_m[i] = self.wind[i]
        self.power_base = np.array(base, dtype=float)
        self.moves_5min = np.array(moves_5, dtype=float)
        self.tmotor = np.array(tmotor, dtype=float)
        self.tinv = np.array(tinv, dtype=float)
        self.sway = np.array(sway, dtype=float)
        self.interf = np.array(interf, dtype=int)
        self.regen = np.array(regen, dtype=float)
        self.state = state
        self.mode_hist = mode
        self.wind = np.array(wind_m, dtype=float)

        # 预测：tos_forecast（按时刻最近不超时的值）
        fc_map = {r["ts"]: r for r in self.fcst}
        gmph_p50 = []
        queue_p50 = []
        next_lashing = []
        next_hatch = []
        last_fc = None
        for t in self.ts:
            # 取最近不晚于 t 的一条
            cand = None
            if t in fc_map:
                cand = fc_map[t]
            else:
                # 简化：遍历到 t 为止的最后一条
                # （数据不大，此处不额外优化。）
                for r in self.fcst:
                    if r["ts"] <= t:
                        cand = r
                    else:
                        break
            last_fc = cand or last_fc
            if last_fc:
                gmph_p50.append(_to_float(last_fc.get("gmph_p50"), float("nan")))
                queue_p50.append(_to_float(last_fc.get("queue_p50"), float("nan")))
                next_lashing.append(_to_int(last_fc.get("next_lashing_utc"), 0))
                next_hatch.append(_to_int(last_fc.get("next_hatch_utc"), 0))
            else:
                gmph_p50.append(float("nan"))
                queue_p50.append(float("nan"))
                next_lashing.append(0)
                next_hatch.append(0)
        self.gmph_p50 = np.array(gmph_p50, dtype=float)
        self.queue_p50 = np.array(queue_p50, dtype=float)
        self.next_lashing = np.array(next_lashing, dtype=int)
        self.next_hatch = np.array(next_hatch, dtype=int)

        # 基线参考（计划层）
        self.mode_ref = []
        self.pwr_ref = []
        self.idle_ref = []
        for i in range(self.h):
            gm = self.gmph_p50[i]
            if math.isnan(gm):
                gm = float(np.nanmean(self.gmph_p50)) if np.any(~np.isnan(self.gmph_p50)) else 30.0
            q = self.queue_p50[i]
            if math.isnan(q):
                q = DEFAULTS["q_lo"]
            m, p, idle = _baseline_ref_for_step(gm, q)
            self.mode_ref.append(m)
            self.pwr_ref.append(p)
            self.idle_ref.append(idle)
        self.pwr_ref = np.array(self.pwr_ref, dtype=float)
        self.idle_ref = np.array(self.idle_ref, dtype=float)

        # 状态计时器与执行状态
        self.idx = 0
        self.is_on = True
        self.on_timer_min = 0.0
        self.off_timer_min = 0.0
        self.prev_mode = self.mode_ref[0] if self.mode_ref else "ecoL1"
        self.prev_idle = self.idle_ref[0] if self.idle_ref.size>0 else DEFAULTS["idle_auto_off_min"]
        self.prev_pwr_pct = self.pwr_ref[0] if self.pwr_ref.size>0 else 0.95

        # 指标累计
        self.metrics = {
            "reward_sum": 0.0,
            "energy_cost": 0.0,
            "carbon_cost": 0.0,
            "idle_energy": 0.0,
            "sla_viol": 0.0,
            "switch_cost": 0.0,
            "thermal_pen": 0.0,
            "peak_penalty": 0.0,
            "regen_income": 0.0,
            "econ_adv_yuan_total": 0.0,
        }

        # 需量口径（若站端未提供 soft cap，则按额定功率*并行台数给出保守上限）
        if DEFAULTS["pcc_soft_cap_kW"] is None:
            DEFAULTS["pcc_soft_cap_kW"] = 0.9 * max(self.qc_meta.get("rated_kW", 400.0), 400.0)

    # ---- 工具：滚动15分钟均值 ----
    def _rolling_mean_15(self, arr: List[float]) -> float:
        # Δt=5 min → 3个点
        if not arr:
            return 0.0
        return float(np.nanmean(np.array(arr[-3:], dtype=float)))

    # ---- 工具：判断 lashing/hatch 是否“临近”或“进行中” ----
    def _proc_window(self, t: int) -> Tuple[bool, bool]:
        # 简化：若 next_lashing_utc/next_hatch_utc 在 [t, t+30min) 视为“临近”；若恰好等于 t 视为“进行中”
        i = self.idx
        lash = self.next_lashing[i] if i < len(self.next_lashing) else 0
        hatch = self.next_hatch[i] if i < len(self.next_hatch) else 0
        near = False
        active = False
        if lash:
            if t <= lash < t + self.dt_min*60:
                active = True
            elif t <= lash < t + 30*60:
                near = True
        if hatch:
            if t <= hatch < t + self.dt_min*60:
                active = True
            elif t <= hatch < t + 30*60:
                near = True
        return near, active

    # ---- 观测打包（前端直用） ----
    def _obs(self) -> Dict[str, Any]:
        i = self.idx
        t = self.ts[i]
        obs = {
            "t": int(t),
            "idx": int(i),
            "mode_ref": self.mode_ref[i],
            "pwr_ref_pct": float(self.pwr_ref[i]),
            "idle_ref_min": float(self.idle_ref[i]),
            "gmph_target": float(self.gmph_p50[i] if math.isfinite(self.gmph_p50[i]) else 30.0),
            "queue_p50": float(self.queue_p50[i] if math.isfinite(self.queue_p50[i]) else DEFAULTS["q_lo"]),
            "tmotor": float(self.tmotor[i]) if math.isfinite(self.tmotor[i]) else None,
            "tinv": float(self.tinv[i]) if math.isfinite(self.tinv[i]) else None,
            "power_base_kW": float(self.power_base[i]),
            "price": float(self.price[i]),
            "ef": float(self.ef[i]),
            "pcc_kw": float(self.pcc[i]) if math.isfinite(self.pcc[i]) else None,
            "moves_5min": float(self.moves_5min[i]),
            "sway_deg": float(self.sway[i]) if math.isfinite(self.sway[i]) else None,
            "wind_mps": float(self.wind[i]) if math.isfinite(self.wind[i]) else None,
            "fut_gmph_p50": float(self.gmph_p50[i]),
            "fut_queue_p50": float(self.queue_p50[i] if math.isfinite(self.queue_p50[i]) else DEFAULTS["q_lo"]),
            "hist_actions": [
                {"pwr_pct": float(self.prev_pwr_pct), "idle_min": float(self.prev_idle)}
            ],
            "hist_masks": []
        }
        return obs

    # ---- 一步推进 ----
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        i = self.idx
        t = self.ts[i]
        obs0 = self._obs()

        # 计划层参考
        mode_b = self.mode_ref[i]
        pwr_b = float(self.pwr_ref[i])
        idle_b = float(self.idle_ref[i])

        # 残差动作（在线微调会填；自检默认0）
        d_power = float(np.clip(float(action.get("d_power_pct", 0.0)), -DEFAULTS["res_band_power"], DEFAULTS["res_band_power"]))
        d_idle  = float(np.clip(float(action.get("d_idle_min", 0.0)), -DEFAULTS["res_band_idle"],  DEFAULTS["res_band_idle"]))
        mode_cmd = str(action.get("mode", "")) or mode_b

        pwr_try = pwr_b + d_power
        idle_try = max(0.0, idle_b + d_idle)

        # --- 强屏蔽规则（与规范一致） ---
        mask_reasons = []
        masked = 0

        # 温升/风/摆/干涉限
        tm = self.tmotor[i]
        ti = self.tinv[i]
        thermal_margin = float("inf")
        t_red = float(self.qc_meta.get("temp_redline_C", 95.0))
        if math.isfinite(tm) and math.isfinite(ti):
            thermal_margin = min(t_red - tm, t_red - ti)

        wind = self.wind[i]
        sway = self.sway[i]
        inter = int(self.interf[i])

        # 里程碑窗口
        near_proc, active_proc = self._proc_window(t)

        # SLA 风险（动态目标基于队列）
        gmph_targ = obs0["gmph_target"]
        q = obs0["queue_p50"]
        if q <= DEFAULTS["q_lo"]:
            gmph_eff = 0.0
        elif q >= DEFAULTS["q_hi"]:
            gmph_eff = gmph_targ
        else:
            scale = (q - DEFAULTS["q_lo"]) / max(1e-6, (DEFAULTS["q_hi"] - DEFAULTS["q_lo"]))
            gmph_eff = gmph_targ * scale

        # 实际（近窗） moves → gmph 近似（5min 窗）*12
        gmph_real = float(self.moves_5min[i] * 12.0)

        # 1) 风/摆/干涉/互锁
        if (math.isfinite(wind) and wind >= float(self.qc_meta.get("wind_cutout_mps", 14.0))) or \
           (math.isfinite(sway) and math.isfinite(self.qc_meta.get("sway_limit_deg", 4.0)) and sway >= self.qc_meta.get("sway_limit_deg", 4.0)) or \
           inter == 1:
            # 强制 normal + 不许降额/待机
            mode_cmd = "normal"
            pwr_try = max(pwr_try, DEFAULTS["eco_power_pct"]["normal"])
            idle_try = min(idle_try, DEFAULTS["idle_auto_off_min"])
            mask_reasons.append("safety_guard(wind/sway/interference)")
        # 2) lashing/hatch 互锁
        if active_proc:
            # 进行中：严禁深待机，确保跟随
            mode_cmd = "normal"
            idle_try = min(idle_try, DEFAULTS["idle_auto_off_min"])
            mask_reasons.append("proc_guard(active)")
        elif near_proc:
            # 临近：禁止大幅 pace_down / 深待机，要求提前唤醒
            idle_try = min(idle_try, DEFAULTS["idle_auto_off_min"])
            mask_reasons.append("proc_guard(near)")

        # 3) SLA 风险：gmph_real < gmph_eff → 禁止降额/延长待机/深 eco
        if gmph_real < gmph_eff - 1e-6:
            if pwr_try < pwr_b:
                pwr_try = pwr_b
                mask_reasons.append("sla_guard_power_down")
            if idle_try > idle_b:
                idle_try = idle_b
                mask_reasons.append("sla_guard_idle_up")
            if mode_cmd in ("ecoL2", "ecoL3"):
                mode_cmd = mode_b
                mask_reasons.append("sla_guard_mode")

        # 4) 温升保护
        if math.isfinite(thermal_margin):
            if thermal_margin < DEFAULTS["thermal_theta1_K"]:
                # 禁止升功率
                if pwr_try > pwr_b:
                    pwr_try = pwr_b
                    mask_reasons.append("thermal_guard_no_up")
            if thermal_margin < DEFAULTS["thermal_theta0_K"]:
                # 强限功率至85%，回到 normal
                pwr_try = min(pwr_try, 0.85)
                mode_cmd = "normal"
                mask_reasons.append("thermal_redline_cap")

        # 5) 需量/DR（此处只有 soft cap；DR 待 rl_engine 接入）
        roll15 = self._rolling_mean_15(list(self.pcc[max(0, i-2):i+1])) if self.pcc.size>0 else 0.0
        soft_cap = float(DEFAULTS["pcc_soft_cap_kW"])
        if roll15 > (soft_cap - DEFAULTS["soft_cap_guard_kW"]):
            if pwr_try > pwr_b:
                pwr_try = pwr_b
                mask_reasons.append("softcap_guard_power_up")

        # 6) min on/off 驻留
        if self.is_on:
            self.on_timer_min += self.dt_min; self.off_timer_min = 0.0
            if self.on_timer_min < max(self.qc_meta.get("min_on_min", 10.0), 10.0):
                if idle_try > idle_b:
                    idle_try = idle_b
                    mask_reasons.append("min_on_guard")
        else:
            self.off_timer_min += self.dt_min; self.on_timer_min = 0.0
            if self.off_timer_min < max(self.qc_meta.get("min_off_min", 5.0), 5.0):
                if idle_try < idle_b:
                    idle_try = idle_b
                    mask_reasons.append("min_off_guard")

        masked = 1 if mask_reasons else 0

        action_after = {"mode": mode_cmd, "power_limit_pct": float(np.clip(pwr_try, 0.6, 1.0)), "idle_timeout_min": float(max(0.0, idle_try))}

        # --- 运行结算 ---
        dt_h = self.dt_min / 60.0
        p_nat = float(self.power_base[i]) if math.isfinite(self.power_base[i]) else 0.12 * float(self.qc_meta.get("rated_kW", 400.0))
        p_act = min(p_nat, p_nat * action_after["power_limit_pct"])
        energy_kwh = p_act * dt_h
        price_i = float(self.price[i] if math.isfinite(self.price[i]) else DEFAULTS["price_yuan"])
        ef_i = float(self.ef[i] if math.isfinite(self.ef[i]) else DEFAULTS["ef_kg_per_kWh"])

        # IdleEnergy 近似：若 idle_timeout_min >= 阈值 且 队列低，则按 base*dt 计入 idle
        idle_energy = 0.0
        if action_after["idle_timeout_min"] >= DEFAULTS["idle_auto_off_min"] and q <= DEFAULTS["q_lo"]:
            idle_energy = p_nat * dt_h

        energy_cost = energy_kwh * price_i
        carbon_cost = energy_kwh * ef_i * 0.0  # 默认不计价（可接碳价后置非零）

        # SLA 罚：gmph_eff - gmph_real（仅非低队列区间才施加）
        sla_gap = 0.0
        if q > DEFAULTS["q_lo"]:
            sla_gap = max(0.0, gmph_eff - gmph_real)
        sla_pen = DEFAULTS["beta_sla"] * sla_gap

        switch_cost = DEFAULTS["gamma_switch"] * (1.0 if mode_cmd != self.prev_mode else 0.0)

        # 热惩罚（softplus）
        def softplus(x):
            return math.log1p(math.exp(x))
        thermal_pen = 0.0
        if math.isfinite(tm) and math.isfinite(ti):
            thermal_pen = DEFAULTS["eta_thermal"] * (softplus((tm-DEFAULTS["T_target"])/3.0) + softplus((ti-DEFAULTS["T_target"])/3.0))

        # 峰罚
        peak_over = max(0.0, roll15 - soft_cap)
        peak_penalty = peak_over * float(DEFAULTS["penalty_yuan_per_kW"])

        # 回收收益
        regen_income = float(self.regen[i])*DEFAULTS["rho_regen"] if (i < len(self.regen) and math.isfinite(self.regen[i])) else 0.0

        r_t = - (energy_cost + carbon_cost) \
              - DEFAULTS["alpha_idle"] * idle_energy \
              - sla_pen - switch_cost - thermal_pen - peak_penalty \
              + regen_income

        # 基线成本（仅能量与价格），用于“相对基线经济优势”口径
        p_ref = min(p_nat, p_nat * pwr_b)
        kwh_ref = p_ref * dt_h
        cost_ref = kwh_ref * price_i
        econ_adv = cost_ref - energy_cost

        # --- 日志逐步 ---
        rec = {
            "key": "qc_step",
            "qc_id": self.qc_id,
            "vessel_id": None,
            "bay_id": None,
            "ts": int(t),
            "obs": obs0,
            "action_in": {"d_power_pct": float(d_power), "d_idle_min": float(d_idle), "mode": str(action.get("mode",""))},
            "action_after_mask": action_after,
            "mask_applied": int(masked),
            "mask_reasons": mask_reasons,
            "p_act_kW": float(p_act),
            "p_ref_kW": float(p_ref),
            "price_yuan_per_kWh": float(price_i),
            "ef_kg_per_kWh": float(ef_i),
            "reward": float(r_t),
            "reward_breakdown": {
                "energy_cost": float(energy_cost),
                "carbon_cost": float(carbon_cost),
                "idle_energy": float(idle_energy),
                "sla_viol": float(sla_pen),
                "switch_cost": float(switch_cost),
                "thermal_penalty": float(thermal_pen),
                "peak_penalty": float(peak_penalty),
                "regen_income": float(regen_income),
            },
            "baseline": {"mode_ref": mode_b, "pwr_ref_pct": float(pwr_b), "idle_ref_min": float(idle_b), "cost_ref": float(cost_ref)},
            "econ_advantage_yuan": float(econ_adv),
        }
        self.log.write(rec)

        # --- 状态推进 + 计数累计 ---
        self.prev_mode = action_after["mode"]
        self.prev_pwr_pct = action_after["power_limit_pct"]
        self.prev_idle = action_after["idle_timeout_min"]

        # on/off 状态近似
        if action_after["idle_timeout_min"] >= DEFAULTS["idle_auto_off_min"] and q <= DEFAULTS["q_lo"]:
            self.is_on = False
        else:
            self.is_on = True

        self.metrics["reward_sum"] += float(r_t)
        self.metrics["energy_cost"] += float(energy_cost)
        self.metrics["carbon_cost"] += float(carbon_cost)
        self.metrics["idle_energy"] += float(idle_energy)
        self.metrics["sla_viol"] += float(sla_pen)
        self.metrics["switch_cost"] += float(switch_cost)
        self.metrics["thermal_pen"] += float(thermal_pen)
        self.metrics["peak_penalty"] += float(peak_penalty)
        self.metrics["regen_income"] += float(regen_income)
        self.metrics["econ_adv_yuan_total"] += float(econ_adv)

        self.idx += 1
        done = self.idx >= self.h
        next_obs = self._obs() if not done else {}
        info = {
            "masked": int(masked),
            "mask_reasons": mask_reasons,
            "econ_advantage_yuan": float(econ_adv),
            "roll15_pcc": float(roll15),
            "gmph_real": float(gmph_real),
            "gmph_eff": float(gmph_eff),
        }
        return next_obs, float(r_t), bool(done), info

    def close(self):
        self.log.close()

# ------------------------------
# 工厂方法：供 RL 引擎与自检使用
# ------------------------------
def make_env(dt_min: int = DEFAULTS["dt_min"], horizon_steps: int = DEFAULTS["horizon"], qc_id: Optional[str] = None,
             data_root: str = DATA_DIR_DEFAULT, jsonl_path: str = DEFAULT_JSONL) -> Tuple[QCEnv, Dict[str, Any]]:
    env = QCEnv(dt_min=dt_min, horizon_steps=horizon_steps, qc_id=qc_id, data_root=data_root, jsonl_path=jsonl_path)
    ctx = {
        "dt_min": dt_min,
        "horizon": horizon_steps,
        "data_root": data_root,
        "jsonl": jsonl_path,
        "qc_id": env.qc_id
    }
    return env, ctx

# ------------------------------
# 自检：推进 H 步，输出 episode 汇总
# ------------------------------
def _self_check(qc_id: Optional[str], dt_min: int, horizon: int, sleep_every: int, sleep_sec: int, data_root: str) -> int:
    env, ctx = make_env(dt_min=dt_min, horizon_steps=horizon, qc_id=qc_id, data_root=data_root, jsonl_path=DEFAULT_JSONL)
    steps = horizon
    for k in range(steps):
        # 自检不加残差：验证“计划层+屏蔽+奖励/日志”是否工作
        _, _, done, _ = env.step({})
        if done:
            break
        if (k+1) % max(1, sleep_every) == 0:
            time.sleep(max(0, sleep_sec))
    # 回合总结
    summary = {
        "key": "qc_episode_summary",
        "steps": int(env.idx),
        "reward_sum": float(env.metrics["reward_sum"]),
        "econ_advantage_yuan_total": float(env.metrics["econ_adv_yuan_total"]),
        "breakdown": {
            "energy_cost": float(env.metrics["energy_cost"]),
            "carbon_cost": float(env.metrics["carbon_cost"]),
            "idle_energy": float(env.metrics["idle_energy"]),
            "sla_viol": float(env.metrics["sla_viol"]),
            "switch_cost": float(env.metrics["switch_cost"]),
            "thermal_pen": float(env.metrics["thermal_pen"]),
            "peak_penalty": float(env.metrics["peak_penalty"]),
            "regen_income": float(env.metrics["regen_income"]),
        }
    }
    env.log.write(summary)
    env.close()
    print("[G-MODULE SELF-CHECK] summary:", json.dumps(summary, ensure_ascii=False))
    return 0

# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="QC Module G - Digital Twin + Planner + Shielding")
    parser.add_argument("--self-check", action="store_true", help="run self-check rollout (no RL residual)")
    parser.add_argument("--qc-id", type=str, default="", help="QC id; default: first in qc_master")
    parser.add_argument("--dt-min", type=int, default=DEFAULTS["dt_min"])
    parser.add_argument("--horizon", type=int, default=DEFAULTS["horizon"])
    parser.add_argument("--data-root", type=str, default=DATA_DIR_DEFAULT, help="data folder (default: port_G_qc_mvp/data)")
    parser.add_argument("--sleep-every", type=int, default=1000)
    parser.add_argument("--sleep-sec", type=int, default=60)
    args = parser.parse_args()

    if args.self_check:
        sys.exit(_self_check(args.qc_id or None, args.dt_min, args.horizon, args.sleep_every, args.sleep_sec, args.data_root))
    else:
        print("Use --self-check for a quick rollout, or import make_env() from rl_engine_g.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())
