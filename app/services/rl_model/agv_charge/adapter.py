# app/services/rl_model/agv_charge/adapter.py
# -*- coding: utf-8 -*-
"""
AGV/无人集卡：充/换电调度 数据适配器（从 9 个原始文件 -> 训练/仿真/上线统一数据）
v1.7（全 CSV 无依赖兜底）

【本版新增/变化】
- 8 个 CSV（除 JSON 外）均支持“纯 Python 解析 + 无 pandas 管线”：
  - vehicles_master / chargers_master：直接构造 VehicleSpec / ChargerSpec（不创建 DataFrame）
  - market_price / grid_ef / grid_meter：构造对齐后的时间序列字典（aligned_*），无需 DataFrame
  - vehicle_state：构建按时间步对齐的“车辆状态快照”映射（_veh_state_at）
  - charge_sessions：列表结构，推断 t 时刻动作无需 DataFrame
  - tos_jobs：列表结构（用于延迟/SLA 统计）
- evaluate_policy 回放历史与 KPI：优先用 pandas；若 pandas 依旧异常，自动降级为 JSON/CSV 写出。

【谁调用/被谁调用】
- train_bc_iql.py：sample_transitions()（下一步我会给）
- module.py：evaluate_policy()/to_actuation_payload()
- server.py/dispatch_api.py：读取 artifacts/ 输出前端与南向下发

运行要求（正常路径）：python>=3.10, pandas>=2.2, numpy>=2.0, pyyaml
在你环境若 pandas 出现底层异常，本文件可完全绕开 pandas 跑通自检。
"""

from __future__ import annotations

import argparse
import json
import io
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable, List

import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from dateutil import parser as dtparser  # v1.8: 用 dateutil 解析时间，避免 pandas 触发崩溃

LOG = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, "INFO"),
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

# ----------------------------
# 版本兼容性（正常路径才用到）
# ----------------------------

def _parse_ver(v: str) -> Tuple[int, int, int]:
    parts = []
    for x in str(v).split(".")[:3]:
        try:
            parts.append(int("".join(ch for ch in x if ch.isdigit())))
        except Exception:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)  # type: ignore


def ensure_numpy_pandas_compat() -> None:
    pv = _parse_ver(pd.__version__)
    nv = _parse_ver(np.__version__)
    if nv[0] >= 2 and pv < (2, 2, 0):
        LOG.warning("Your numpy/pandas combo may be problematic: numpy %s, pandas %s", np.__version__, pd.__version__)

# ----------------------------
# 常量与默认配置
# ----------------------------
DEFAULT_DT_MIN = 5
ARTIFACT_DIRNAME = "artifacts"


@dataclass
class VehicleSpec:
    vehicle_id: str
    battery_kwh: float
    p_charge_max_kw: float
    soc_min: float
    soc_max: float
    soc_target: float
    can_swap: bool = False


@dataclass
class ChargerSpec:
    charger_id: str
    station_id: str
    type: str  # "charger" or "swap"
    max_power_kw: float
    concurrency: int = 1
    ramp_kw_per_min: float = 9999.0


@dataclass
class GridConfig:
    pcc_limit_kw: float
    demand_limit_kw: Optional[float]
    demand_window_min: int
    demand_penalty_per_kw: float
    safety_n_1_margin_kw: float = 0.0


@dataclass
class AdapterConfig:
    dt_min: int = DEFAULT_DT_MIN
    low_soc_threshold: float = 0.3
    target_soc: float = 0.8
    max_c_rate: float = 1.0
    reward_alpha_delay: float = 5.0
    reward_beta_peak: float = 2.0
    reward_gamma_switch: float = 0.1
    reward_eta_degrade: float = 0.05
    p_CO2_yuan_per_kg: float = 0.0
    time_col: str = "time"
    vehicle_id_col: str = "vehicle_id"
    station_id_col: str = "station_id"
    charger_id_col: str = "charger_id"
    price_col: str = "price_yuan_per_kwh"
    ef_col: str = "ef_kg_per_kwh"
    grid_kw_col: str = "pcc_kw"
    available_col: str = "available"
    next_task_eta_min_col: str = "next_task_eta_min"
    priority_col: str = "priority"
    can_swap_col: str = "can_swap"
    location_col: str = "location"
    soc_col: str = "soc"
    temp_col: str = "temp"

# ----------------------------
# 文件读取：稳健 CSV + Excel 识别 + 纯 Python
# ----------------------------

def _is_excel_like(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(8)
        return head[:2] == b"PK" or head.startswith(b"\xD0\xCF\x11\xE0")
    except Exception:
        return False


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """尽最大可能用 pandas 读；失败则抛错（由上层接纯 Python 兜底）。"""
    if not path.exists():
        raise FileNotFoundError(path)
    # Excel 头识别
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if head[:2] == b"PK" or head.startswith(b"\xD0\xCF\x11\xE0"):
            return pd.read_excel(path, engine=None)
    except Exception:
        pass

    errors = []
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, engine="c", encoding=enc, low_memory=False)
        except Exception as e:
            errors.append(f"C/{enc}:{e}")
    for enc in encodings:
        try:
            return pd.read_csv(path, engine="python", sep=None, encoding=enc, on_bad_lines="skip")
        except Exception as e:
            errors.append(f"PY/{enc}:{e}")
    for enc in encodings:
        try:
            return pd.read_csv(path, engine="python", sep=None, encoding=enc, on_bad_lines="skip", dtype=str)
        except Exception as e:
            errors.append(f"PY_STR/{enc}:{e}")

    # 原始字节 -> 文本 -> StringIO
    try:
        raw = open(path, "rb").read()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1", errors="replace")
        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        buf = io.StringIO(text)
        try:
            return pd.read_csv(buf, engine="python", sep=None, on_bad_lines="skip")
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf)
    except Exception as e:
        errors.append(f"RAW:{e}")

    # 强制 read_excel
    try:
        return pd.read_excel(path, engine=None)
    except Exception as e:
        errors.append(f"XLX:{e}")

    raise RuntimeError(f"pandas read failed: {path}; tail={errors[-3:]}")

def _read_csv_rows_plain(path: Path, encodings=("utf-8", "utf-8-sig", "gbk", "latin-1")) -> List[Dict[str, str]]:
    """纯 Python：csv.reader -> 字典列表（字符串）"""
    last = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                r = csv.reader(f)
                rows = list(r)
            if not rows:
                return []
            header = [h.strip() for h in rows[0]]
            out = []
            for row in rows[1:]:
                d = {}
                for i, h in enumerate(header):
                    if i < len(row):
                        d[h] = row[i]
                out.append(d)
            return out
        except Exception as e:
            last = e
    raise RuntimeError(f"plain csv read failed for {path}: {last}")
# ---- 轻量时间索引：当 pandas.DatetimeIndex 创建失败时的替身 ----
class PyTimeIndex(list):
    """
    仅实现当前代码用到的最小接口：
    - 切片/下标：继承 list 即可
    - get_loc(ts): 返回等于 ts 的位置索引（找不到时报 ValueError）
    """
    def get_loc(self, ts):
        # Datetime values match the internally constructed timeline exactly.
        return self.index(ts)


# ----------------------------
# 时间/对齐辅助（无 pandas 路径）
# ----------------------------

def _to_dt(x) -> datetime:
    """
    稳健的时间解析（纯 Python，不碰 pandas）：
    - 先处理已是 datetime/np.datetime64 的情况
    - 再尝试把数字/数字字符串按“秒级 epoch”解析
    - 否则用 dateutil.parser 解析常见时间字符串（含 ISO 格式）
    """
    if isinstance(x, datetime):
        return x
    # numpy datetime64 -> python datetime
    try:
        import numpy as _np  # 局部导入，避免全局副作用
        if isinstance(x, _np.datetime64):
            # 转为毫秒再构造
            ts_ms = _np.datetime64(x, 'ms').astype('int64')
            return datetime.utcfromtimestamp(float(ts_ms) / 1000.0)
    except Exception:
        pass

    # 纯数字的 epoch（秒）
    try:
        if isinstance(x, (int, float)):
            return datetime.utcfromtimestamp(float(x))
        s = str(x).strip()
        if s.isdigit() or (s.replace('.', '', 1).isdigit() and s.count('.') < 2):
            return datetime.utcfromtimestamp(float(s))
    except Exception:
        pass

    # 通用字符串解析（ISO/常见格式）
    try:
        return dtparser.parse(str(x))
    except Exception as e:
        raise ValueError(f"cannot parse datetime from value={x!r}") from e


def _build_time_range_from_lists(lists: List[List[Tuple[datetime, float]]], dt_min: int) -> List[datetime]:
    """从若干 (t,val) 序列推导统一时间轴；若都为空，用当日 12h"""
    ts = []
    for L in lists:
        if L:
            ts.append(L[0][0])
            ts.append(L[-1][0])
    if not ts:
        start = datetime(2025, 1, 1, 0, 0, 0)
        end = start + timedelta(hours=12)
    else:
        start, end = min(ts), max(ts)
    step = timedelta(minutes=dt_min)
    out = []
    t = start
    while t <= end:
        out.append(t)
        t += step
    return out

def _align_series_ffill(points: List[Tuple[datetime, float]], index: List[datetime]) -> Dict[datetime, float]:
    """把离散点 (t,val) 前向填充到 index 上。若无点，统一给 NaN。"""
    out = {}
    if not points:
        for t in index:
            out[t] = float("nan")
        return out
    points = sorted(points, key=lambda x: x[0])
    j = 0
    cur = points[0][1]
    for t in index:
        while j + 1 < len(points) and points[j + 1][0] <= t:
            j += 1
            cur = points[j][1]
        out[t] = float(cur)
    return out

# ----------------------------
# Grid config 兼容解析
# ----------------------------

def _parse_grid_config_compat(gcfg: dict) -> GridConfig:
    if ("pcc_limit_kw" in gcfg) or ("demand_window_min" in gcfg) or ("demand_penalty_per_kw" in gcfg):
        return GridConfig(
            pcc_limit_kw=float(gcfg.get("pcc_limit_kw", 50000.0)),
            demand_limit_kw=float(gcfg["demand_limit_kw"]) if "demand_limit_kw" in gcfg and gcfg.get("demand_limit_kw") is not None else None,
            demand_window_min=int(gcfg.get("demand_window_min", 900)),
            demand_penalty_per_kw=float(gcfg.get("demand_penalty_per_kw", 0.0)),
            safety_n_1_margin_kw=float(gcfg.get("safety_n_1_margin_kw", 0.0)),
        )
    feeders = gcfg.get("feeders", [])
    if feeders:
        pcc = float(sum(float(f.get("p_limit_kW", 0.0)) for f in feeders))
        soft = float(sum(float(f.get("soft_cap_kW", 0.0)) for f in feeders))
        n1 = float(sum(float(f.get("n_minus_1_margin_kW", 0.0)) for f in feeders))
        dwin = gcfg.get("demand_window", {})
        gran = int(dwin.get("granularity_min", 15))
        pen = float(gcfg.get("demand_penalty_yuan_per_kW", gcfg.get("demand_penalty_per_kw", 0.0)))
        return GridConfig(
            pcc_limit_kw=pcc,
            demand_limit_kw=(soft if soft > 0 else None),
            demand_window_min=gran,
            demand_penalty_per_kw=pen,
            safety_n_1_margin_kw=n1,
        )
    raise KeyError("Unsupported port_grid_config schema")

# ----------------------------
# 主类
# ----------------------------

class AGVChargeAdapter:
    """统一读取→状态构造→奖励/KPI→基线→自检→下发表"""

    def __init__(self, base_dir: Optional[Path] = None, config_path: Optional[Path] = None):
        ensure_numpy_pandas_compat()
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.artifact_dir = self.base_dir / ARTIFACT_DIRNAME
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = Path(config_path) if config_path else (self.base_dir / "config.yaml")
        self.cfg = self._load_config(cfg_path)

        self.dt_min = self.cfg.dt_min
        self.rule = f"{self.dt_min}min"

        # 结构化对象（车辆/桩/电网）
        self.vehicles: Dict[str, VehicleSpec] = {}
        self.chargers: Dict[str, ChargerSpec] = {}
        self.grid: Optional[GridConfig] = None

        # pandas 正常管线
        self.price_df: Optional[pd.DataFrame] = None
        self.ef_df: Optional[pd.DataFrame] = None
        self.grid_meter_df: Optional[pd.DataFrame] = None
        self.vehicle_state_df: Optional[pd.DataFrame] = None
        self.charge_sessions_df: Optional[pd.DataFrame] = None
        self.tos_jobs_df: Optional[pd.DataFrame] = None

        # 无 pandas 管线（对齐后）
        self._index: Optional[List[datetime]] = None
        self._price_pts: List[Tuple[datetime, float]] = []
        self._ef_pts: List[Tuple[datetime, float]] = []
        self._grid_pts: List[Tuple[datetime, float]] = []
        self._price_aligned: Dict[datetime, float] = {}
        self._ef_aligned: Dict[datetime, float] = {}
        self._grid_aligned: Dict[datetime, float] = {}

        # 车辆时序 & 会话 & 作业（无 pandas 兜底）
        self._veh_rows: List[Dict[str, str]] = []
        self._veh_state_at: Dict[datetime, Dict[str, Dict[str, float]]] = {}
        self._sessions: List[Dict[str, object]] = []
        self._jobs: List[Dict[str, object]] = []

        self._time_index: Optional[pd.DatetimeIndex] = None  # 兼容旧接口

    # ---------- 公共接口 ----------

    def load_all(self) -> None:
        LOG.info("Loading data from %s", self.data_dir)

        # 1) vehicles_master.csv
        vpath = self.data_dir / "vehicles_master.csv"
        self.vehicles.clear()
        try:
            vdf = _safe_read_csv(vpath)
            for _, row in vdf.iterrows():
                vid = str(row.get(self.cfg.vehicle_id_col, row.get("vehicle_id")))
                if pd.isna(vid):
                    continue
                can_swap_val = row.get(self.cfg.can_swap_col, row.get("can_swap", 0))
                try:
                    cbool = bool(int(can_swap_val)) if str(can_swap_val).strip() != "" else False
                except Exception:
                    cbool = False
                self.vehicles[vid] = VehicleSpec(
                    vehicle_id=vid,
                    battery_kwh=float(row.get("battery_kwh", 200.0)),
                    p_charge_max_kw=float(row.get("p_charge_max_kw", 150.0)),
                    soc_min=float(row.get("soc_min", 0.1)),
                    soc_max=float(row.get("soc_max", 1.0)),
                    soc_target=float(row.get("soc_target", 0.8)),
                    can_swap=cbool,
                )
        except Exception as e:
            LOG.warning("pandas failed on vehicles_master.csv: %s; use pure csv.", e)
            rows = _read_csv_rows_plain(vpath)
            for r in rows:
                vid = str(r.get(self.cfg.vehicle_id_col, r.get("vehicle_id", ""))).strip()
                if not vid:
                    continue
                def tn(v, d):
                    try: return float(v)
                    except: return float(d)
                def ti(v, d):
                    try: return int(v)
                    except: return int(d)
                self.vehicles[vid] = VehicleSpec(
                    vehicle_id=vid,
                    battery_kwh=tn(r.get("battery_kwh", 200.0), 200.0),
                    p_charge_max_kw=tn(r.get("p_charge_max_kw", 150.0), 150.0),
                    soc_min=tn(r.get("soc_min", 0.1), 0.1),
                    soc_max=tn(r.get("soc_max", 1.0), 1.0),
                    soc_target=tn(r.get("soc_target", 0.8), 0.8),
                    can_swap=bool(ti(r.get(self.cfg.can_swap_col, r.get("can_swap", "0")), 0)),
                )
        LOG.info("Loaded vehicles: %d", len(self.vehicles))

        # 2) chargers_master.csv
        cpath = self.data_dir / "chargers_master.csv"
        self.chargers.clear()
        try:
            cdf = _safe_read_csv(cpath)
            c_station = self.cfg.station_id_col if self.cfg.station_id_col in cdf.columns else "station_id"
            c_type = "type" if "type" in cdf.columns else "charger_type"
            c_chid = self.cfg.charger_id_col if self.cfg.charger_id_col in cdf.columns else "charger_id"
            for _, row in cdf.iterrows():
                cid = str(row.get(c_chid, row.get("id", row.get("name"))))
                self.chargers[cid] = ChargerSpec(
                    charger_id=cid,
                    station_id=str(row.get(c_station, "S01")),
                    type=str(row.get(c_type, "charger")).lower(),
                    max_power_kw=float(row.get("max_power_kw", row.get("p_max_kw", 180.0))),
                    concurrency=int(row.get("concurrency", row.get("connector_count", 1))),
                    ramp_kw_per_min=float(row.get("ramp_kw_per_min", row.get("ramp_rate_kw_per_min", 9999.0))),
                )
        except Exception as e:
            LOG.warning("pandas failed on chargers_master.csv: %s; use pure csv.", e)
            rows = _read_csv_rows_plain(cpath)
            for r in rows:
                cid = str(r.get(self.cfg.charger_id_col, r.get("charger_id", r.get("id", r.get("name", ""))))).strip()
                if not cid:
                    continue
                def tn(v, d):
                    try: return float(v)
                    except: return float(d)
                def ti(v, d):
                    try: return int(v)
                    except: return int(d)
                self.chargers[cid] = ChargerSpec(
                    charger_id=cid,
                    station_id=str(r.get(self.cfg.station_id_col, r.get("station_id", r.get("site", "S01")))),
                    type=str(r.get("type", r.get("charger_type", "charger"))).lower(),
                    max_power_kw=tn(r.get("max_power_kw", r.get("p_max_kw", 180.0)), 180.0),
                    concurrency=ti(r.get("concurrency", r.get("connector_count", 1)), 1),
                    ramp_kw_per_min=tn(r.get("ramp_kw_per_min", r.get("ramp_rate_kw_per_min", 9999.0)), 9999.0),
                )
        LOG.info("Loaded chargers: %d", len(self.chargers))

        # 3) port_grid_config.json
        gcfg = json.loads((self.data_dir / "port_grid_config.json").read_text(encoding="utf-8"))
        self.grid = _parse_grid_config_compat(gcfg)

        # 4) market_price.csv -> self._price_pts 或 price_df
        mpath = self.data_dir / "market_price.csv"
        try:
            mdf = _safe_read_csv(mpath)
            tcol = self.cfg.time_col if self.cfg.time_col in mdf.columns else "time"
            mdf[tcol] = pd.to_datetime(mdf[tcol])
            if self.cfg.price_col not in mdf.columns:
                for cand in ["price", "elec_price", "yuan_per_kwh", "p"]:
                    if cand in mdf.columns:
                        mdf.rename(columns={cand: self.cfg.price_col}, inplace=True)
                        break
            self.price_df = mdf[[tcol, self.cfg.price_col]].sort_values(tcol).set_index(tcol)
        except Exception as e:
            LOG.warning("pandas failed on market_price.csv: %s; use pure csv.", e)
            rows = _read_csv_rows_plain(mpath)
            pts = []
            for r in rows:
                try:
                    t = _to_dt(r.get("time", r.get(self.cfg.time_col)))
                    v = float(r.get(self.cfg.price_col, r.get("price", r.get("elec_price", r.get("yuan_per_kwh", r.get("p", "nan"))))))
                    pts.append((t, v))
                except Exception:
                    continue
            pts.sort(key=lambda x: x[0])
            self._price_pts = pts

        # 5) grid_ef.csv
        epath = self.data_dir / "grid_ef.csv"
        try:
            edf = _safe_read_csv(epath)
            tcol = self.cfg.time_col if self.cfg.time_col in edf.columns else "time"
            edf[tcol] = pd.to_datetime(edf[tcol])
            if self.cfg.ef_col not in edf.columns:
                for cand in ["ef", "emission_factor", "kgCO2_per_kwh"]:
                    if cand in edf.columns:
                        edf.rename(columns={cand: self.cfg.ef_col}, inplace=True)
                        break
            self.ef_df = edf[[tcol, self.cfg.ef_col]].sort_values(tcol).set_index(tcol)
        except Exception as e:
            LOG.warning("pandas failed on grid_ef.csv: %s; use pure csv.", e)
            rows = _read_csv_rows_plain(epath)
            pts = []
            for r in rows:
                try:
                    t = _to_dt(r.get("time", r.get(self.cfg.time_col)))
                    v = float(r.get(self.cfg.ef_col, r.get("ef", r.get("emission_factor", r.get("kgCO2_per_kwh", "nan")))))
                    pts.append((t, v))
                except Exception:
                    continue
            pts.sort(key=lambda x: x[0])
            self._ef_pts = pts

        # 6) grid_meter.csv
        gmpath = self.data_dir / "grid_meter.csv"
        try:
            gmdf = _safe_read_csv(gmpath)
            tcol = self.cfg.time_col if self.cfg.time_col in gmdf.columns else "time"
            gmdf[tcol] = pd.to_datetime(gmdf[tcol])
            if self.cfg.grid_kw_col not in gmdf.columns:
                for cand in ["pcc_power_kw", "total_kw", "power_kw", "kw"]:
                    if cand in gmdf.columns:
                        gmdf.rename(columns={cand: self.cfg.grid_kw_col}, inplace=True)
                        break
            self.grid_meter_df = gmdf[[tcol, self.cfg.grid_kw_col]].sort_values(tcol).set_index(tcol)
        except Exception as e:
            LOG.warning("pandas failed on grid_meter.csv: %s; use pure csv.", e)
            rows = _read_csv_rows_plain(gmpath)
            pts = []
            for r in rows:
                try:
                    t = _to_dt(r.get("time", r.get(self.cfg.time_col)))
                    v = float(r.get(self.cfg.grid_kw_col, r.get("pcc_power_kw", r.get("total_kw", r.get("power_kw", r.get("kw", "nan"))))))
                    pts.append((t, v))
                except Exception:
                    continue
            pts.sort(key=lambda x: x[0])
            self._grid_pts = pts

        # 7) vehicle_state.csv
        vspath = self.data_dir / "vehicle_state.csv"
        try:
            vsdf = _safe_read_csv(vspath)
            tcol = self.cfg.time_col if self.cfg.time_col in vsdf.columns else "time"
            vsdf[tcol] = pd.to_datetime(vsdf[tcol])
            for need in [self.cfg.vehicle_id_col, self.cfg.soc_col]:
                if need not in vsdf.columns:
                    raise KeyError("vehicle_state.csv missing required columns")
            for col, default in [(self.cfg.available_col, 1), (self.cfg.next_task_eta_min_col, 0),
                                 (self.cfg.priority_col, 0), (self.cfg.temp_col, 25.0)]:
                if col not in vsdf.columns:
                    vsdf[col] = default
            self.vehicle_state_df = vsdf.set_index(tcol)
        except Exception as e:
            LOG.warning("pandas failed on vehicle_state.csv: %s; use pure csv.", e)
            rows = _read_csv_rows_plain(vspath)
            self._veh_rows = rows  # 先存原始行，稍后对齐

        # 8) charge_sessions.csv
        cspath = self.data_dir / "charge_sessions.csv"
        try:
            csdf = _safe_read_csv(cspath)
            if "start_time" in csdf.columns and "end_time" in csdf.columns:
                csdf["start_time"] = pd.to_datetime(csdf["start_time"])
                csdf["end_time"] = pd.to_datetime(csdf["end_time"])
            elif self.cfg.time_col in csdf.columns and "duration_min" in csdf.columns:
                csdf["start_time"] = pd.to_datetime(csdf[self.cfg.time_col])
                csdf["end_time"] = csdf["start_time"] + pd.to_timedelta(csdf["duration_min"], unit="m")
            else:
                csdf["start_time"] = pd.to_datetime("1970-01-01")
                csdf["end_time"] = pd.to_datetime("1970-01-01")
            self.charge_sessions_df = csdf
        except Exception as e:
            LOG.warning("pandas failed on charge_sessions.csv: %s; use pure csv.", e)
            rows = _read_csv_rows_plain(cspath)
            sess = []
            for r in rows:
                try:
                    if r.get("start_time") and r.get("end_time"):
                        st = _to_dt(r["start_time"]); et = _to_dt(r["end_time"])
                    elif r.get(self.cfg.time_col) and r.get("duration_min"):
                        st = _to_dt(r[self.cfg.time_col]); et = st + timedelta(minutes=float(r["duration_min"]))
                    else:
                        continue
                    vid = str(r.get(self.cfg.vehicle_id_col, r.get("vehicle_id", ""))).strip()
                    powkw = r.get("avg_power_kw", r.get("power_kw", r.get("p_set_kw", "")))
                    ekwh = r.get("energy_kwh")
                    if powkw:
                        p = float(powkw)
                    elif ekwh:
                        dur_h = max((et - st).total_seconds()/3600.0, 1e-6)
                        p = float(ekwh) / dur_h
                    else:
                        p = float("nan")
                    sess.append({"start_time": st, "end_time": et, "vehicle_id": vid, "avg_power_kw": p})
                except Exception:
                    continue
            self._sessions = sess

        # 9) tos_jobs.csv
        tjpath = self.data_dir / "tos_jobs.csv"
        try:
            tjdf = _safe_read_csv(tjpath)
            for c in ["due_time", "start_time", "finish_time"]:
                if c in tjdf.columns:
                    tjdf[c] = pd.to_datetime(tjdf[c])
            self.tos_jobs_df = tjdf
        except Exception as e:
            LOG.warning("pandas failed on tos_jobs.csv: %s; use pure csv.", e)
            rows = _read_csv_rows_plain(tjpath)
            jobs = []
            for r in rows:
                jobs.append({
                    "due_time": _to_dt(r.get("due_time")) if r.get("due_time") else None,
                    "start_time": _to_dt(r.get("start_time")) if r.get("start_time") else None,
                    "finish_time": _to_dt(r.get("finish_time")) if r.get("finish_time") else None,
                })
            self._jobs = jobs

        # ==== 统一时间轴 & 对齐 ====
        self._build_and_align_timebase()
        self._emit_data_quality_report()

    # ---------- 训练/评估接口 ----------

    def sample_transitions(self, start: Optional[datetime] = None, end: Optional[datetime] = None,
                           include_actions_from_history: bool = True) -> Iterable[Dict]:
        tidx = self._slice_time_index(start, end)
        for t in tidx:
            obs = self._build_observation_at(t)
            act = self._infer_action_from_sessions(t) if include_actions_from_history else None
            reward, info = self._calc_reward_at(t, action_power_map=act or {})
            yield {"obs": obs, "action": act, "reward": reward, "done": False, "info": info, "time": t}

    def evaluate_policy(self, policy_fn, horizon_hours: int = 12) -> Dict:
        tidx = self._time_index[:int(np.ceil(horizon_hours * 60 / self.dt_min))]
        records = []
        for t in tidx:
            obs = self._build_observation_at(t)
            action = policy_fn(obs) or {}
            action = self._project_to_feasible(t, action)
            reward, info = self._calc_reward_at(t, action_power_map=action)
            # ===== 计算前端需要的三个额外字段 =====
            # 1) Convert the hourly delay penalty to minutes.
            latency_min = float(info.get("delay_penalty", 0.0)) * 60.0

            # 2) 削峰能力（kW）：给一个直观口径——距需量上限的“剩余余量”
            #    如果设置了需量上限，就用 (上限 - 预测功率)，否则给 0
            limit = getattr(self.grid, "demand_limit_kw", None)
            predicted_kw = float(info.get("grid_kw", 0.0)) + float(info.get("total_power_kw", 0.0))
            peak_reduction_kW = float(max(0.0, (limit - predicted_kw))) if limit is not None else 0.0

            # 3) 策略“熵”（探索度的近似）：正在充电的车辆占比的二元熵
            n_veh = max(1, len(obs.get("vehicles", [])))
            k_on = sum(1 for p in (action or {}).values() if p > 1e-6)
            p_on = k_on / n_veh
            # 防 0/1（避免 log(0)）
            eps = 1e-9
            entropy = float(-(p_on * np.log(p_on + eps) + (1 - p_on) * np.log(1 - p_on + eps)))

            # 写入记录（前端的字段别名有兼容，这里直接提供命中的键名）
            records.append({
                "time": t,
                "action": action,
                "reward": float(reward),
                "latency_min": latency_min,  # 回合长度/延迟（分钟）
                "peak_reduction_kW": peak_reduction_kW,  # 削峰能力（kW）
                "entropy": entropy,  # 策略熵/探索度
                **info
            })

        # KPI 与回放落盘（优先 pandas，失败则 JSON/CSV）
        kpi = self._aggregate_kpi_safe(records)
        hist_path = self.artifact_dir / "policy_evaluate_history.parquet"
        try:
            df = pd.DataFrame(records).set_index("time")
            df.to_parquet(hist_path)
            # 同步写一份 JSONL，供前端直接读取
            jsonl_path = self.artifact_dir / "policy_evaluate_history.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in records:
                    r2 = dict(r)
                    r2["time"] = r2["time"].isoformat()
                    f.write(json.dumps(r2, ensure_ascii=False) + "\n")

        except Exception:
            # 降级
            hist_path = self.artifact_dir / "policy_evaluate_history.jsonl"
            with open(hist_path, "w", encoding="utf-8") as f:
                for r in records:
                    r2 = dict(r); r2["time"] = r2["time"].isoformat()
                    f.write(json.dumps(r2, ensure_ascii=False) + "\n")
        LOG.info("Saved rollout history -> %s", hist_path)
        return {"kpi": kpi, "history_path": str(hist_path)}

    # ---------- 规则基线策略 ----------

    def rule_baseline_policy(self):
        def _fn(obs: Dict) -> Dict[str, float]:
            grid_room = obs["grid_room_kw"]
            price = obs["price_yuan_per_kwh"]
            ef = obs["ef_kg_per_kwh"]
            appetite = 0.5 * (1.0 / (1.0 + price)) + 0.5 * (1.0 / (1.0 + ef))
            cand = [v for v in obs["vehicles"]
                    if (v["available"] > 0) and (v["soc"] < max(v["soc_target"], self.cfg.low_soc_threshold))]
            cand.sort(key=lambda x: (x["soc"], -x["priority"]))
            action, used = {}, 0.0
            for v in cand:
                pmax = min(v["p_charge_max_kw"], v["battery_kwh"] * self.cfg.max_c_rate)
                want = float(np.clip(appetite * pmax, 0, pmax))
                remain = max(0.0, grid_room - used)
                if remain <= 1e-6:
                    break
                pset = float(min(want, remain))
                if pset > 1e-6:
                    action[v["vehicle_id"]] = pset
                    used += pset
            return action
        return _fn

    # ---------- 下发打包 ----------

    def to_actuation_payload(self, action_map: Dict[str, float], effective_time: datetime, duration_min=None) -> Dict:
        end_ts = effective_time + timedelta(minutes=duration_min or self.dt_min)
        first_station = None
        first_charger = None
        if self.chargers:
            first_charger = list(self.chargers.values())[0]
            first_station = first_charger.station_id
        cmds = []
        for vid, pkw in action_map.items():
            cmds.append({
                "vehicle_id": vid,
                "target_station_id": first_station or "S01",
                "target_charger_id": first_charger.charger_id if first_charger else "S01-C01",
                "setpoint_kw": float(max(0.0, pkw)),
                "mode": "charge",
                "workorder": {"type": "go_charge", "time_window": [effective_time.isoformat(), end_ts.isoformat()]},
            })
        return {"version": "v1", "effective_time": effective_time.isoformat(),
                "duration_min": int(duration_min or self.dt_min), "commands": cmds}

    # ---------- 自检 ----------

    def self_check(self) -> Dict:
        self.load_all()
        policy = self.rule_baseline_policy()
        res = self.evaluate_policy(policy, horizon_hours=12)
        t0 = self._time_index[0]
        payload = self.to_actuation_payload(policy(self._build_observation_at(t0)), effective_time=t0, duration_min=self.dt_min)
        (self.artifact_dir / "sample_dispatch_payload.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report = {"kpi": res["kpi"], "history_path": res["history_path"],
                  "sample_payload_path": str(self.artifact_dir / "sample_dispatch_payload.json")}
        (self.artifact_dir / "adapter_self_check_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOG.info("Self-check done.")
        return report

    # ---------- 内部实现（时间轴与对齐） ----------

    def _build_and_align_timebase(self) -> None:
        """根据可用数据构建统一时间轴，并前向对齐（pandas 存在则走 pandas；否则走纯 Python）"""
        # 先尝试 pandas 时间范围
        starts, ends = [], []
        for df in [self.price_df, self.ef_df, self.grid_meter_df]:
            if df is not None and not df.empty:
                starts.append(df.index.min().to_pydatetime())
                ends.append(df.index.max().to_pydatetime())

        # 若 pandas 三个都不可用，改用纯 Python点列的范围
        lists = []
        if not starts:
            if self._price_pts: lists.append(self._price_pts)
            if self._ef_pts: lists.append(self._ef_pts)
            if self._grid_pts: lists.append(self._grid_pts)
            self._index = _build_time_range_from_lists(lists, self.dt_min)
        else:
            start, end = min(starts), max(ends)
            step = timedelta(minutes=self.dt_min)
            idx = []
            t = start
            while t <= end:
                idx.append(t)
                t += step
            self._index = idx

        # pandas 版本的时间轴（兼容原接口）
        # Some NumPy/Pandas combinations can fail while constructing DatetimeIndex.
        # 这里兜底为纯 Python 索引，保证后续逻辑正常跑。
        try:
            self._time_index = pd.DatetimeIndex(self._index)
        except Exception as e:
            LOG.warning("pandas DatetimeIndex failed: %s; fallback to PyTimeIndex.", e)
            self._time_index = PyTimeIndex(self._index)  # 纯 Python 索引

        # 对齐三个时序：优先 pandas；否则纯 Python对齐
        if self.price_df is not None:
            self.price_df = self.price_df.reindex(self._time_index).ffill()
        else:
            self._price_aligned = _align_series_ffill(self._price_pts, self._index)

        if self.ef_df is not None:
            self.ef_df = self.ef_df.reindex(self._time_index).ffill()
        else:
            self._ef_aligned = _align_series_ffill(self._ef_pts, self._index)

        if self.grid_meter_df is not None:
            self.grid_meter_df = self.grid_meter_df.reindex(self._time_index).ffill()
        else:
            self._grid_aligned = _align_series_ffill(self._grid_pts, self._index)

        # 车辆时序对齐：pandas 优先，否则纯 Python构造每步快照
        if self.vehicle_state_df is not None:
            # groupby(vehicle)->resample->ffill
            vs = self.vehicle_state_df.reset_index().rename(columns={self.cfg.time_col: "time"})
            vs["time"] = pd.to_datetime(vs["time"])
            vs = vs[(vs["time"] >= self._time_index[0]) & (vs["time"] <= self._time_index[-1])]
            vs = vs.set_index("time").groupby(self.cfg.vehicle_id_col).apply(lambda g: g.resample(self.rule).ffill())
            vs.index = vs.index.set_names([self.cfg.vehicle_id_col, "time"])
            self.vehicle_state_df = vs
        else:
            # 纯 Python：为每个时间步生成 map: vid -> 状态
            # 先把每车的时间序列整理
            by_vid: Dict[str, List[Dict[str, object]]] = {}
            for r in self._veh_rows:
                try:
                    t = _to_dt(r.get("time", r.get(self.cfg.time_col)))
                    vid = str(r.get(self.cfg.vehicle_id_col, r.get("vehicle_id", ""))).strip()
                    if not vid:
                        continue
                    item = {
                        "time": t,
                        "soc": float(r.get(self.cfg.soc_col, r.get("soc", "nan"))),
                        "available": int(float(r.get(self.cfg.available_col, r.get("available", 1)))) if r.get(self.cfg.available_col, r.get("available")) is not None else 1,
                        "eta_min": float(r.get(self.cfg.next_task_eta_min_col, r.get("next_task_eta_min", 0))),
                        "priority": int(float(r.get(self.cfg.priority_col, r.get("priority", 0)))),
                        "temp": float(r.get(self.cfg.temp_col, r.get("temp", 25.0))),
                    }
                    by_vid.setdefault(vid, []).append(item)
                except Exception:
                    continue
            for vid in by_vid:
                by_vid[vid] = sorted(by_vid[vid], key=lambda x: x["time"])

            # 前向填充到统一时间轴
            last: Dict[str, Dict[str, float]] = {}
            for t in self._index:
                snap = {}
                for vid, seq in by_vid.items():
                    # 推进该车序列
                    while seq and seq[0]["time"] <= t:
                        cur = seq.pop(0)
                        last[vid] = {
                            "soc": float(cur["soc"]),
                            "available": int(cur["available"]),
                            "eta_min": float(cur["eta_min"]),
                            "priority": int(cur["priority"]),
                            "temp": float(cur["temp"]),
                        }
                    if vid in last:
                        snap[vid] = last[vid].copy()
                self._veh_state_at[t] = snap

    # ---------- 内部实现（观测/动作/奖励/KPI） ----------

    def _emit_data_quality_report(self) -> None:
        rep = {
            "vehicles": len(self.vehicles),
            "chargers": len(self.chargers),
            "time_steps": int(len(self._time_index or [])),
            "price_has_null": (self.price_df.isna().any().any() if self.price_df is not None else False),
            "ef_has_null": (self.ef_df.isna().any().any() if self.ef_df is not None else False),
            "grid_has_null": (self.grid_meter_df.isna().any().any() if self.grid_meter_df is not None else False),
            "pipeline": "pandas" if self.price_df is not None and self.ef_df is not None and self.grid_meter_df is not None else "pure-python-mixed",
            "latest_time": self._index[-1].isoformat() if self._index else None,
            "has_vehicle_states": bool((self.vehicle_state_df is not None and not self.vehicle_state_df.empty) or self._veh_state_at),
            "has_charge_sessions": bool((self.charge_sessions_df is not None and not self.charge_sessions_df.empty) or self._sessions),
            "has_tos_jobs": bool((self.tos_jobs_df is not None and not self.tos_jobs_df.empty) or self._jobs),
        }
        (self.artifact_dir / "data_quality_quickcheck.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _slice_time_index(self, start: Optional[datetime], end: Optional[datetime]) -> List[datetime]:
        idx = self._index
        if start is not None:
            idx = [t for t in idx if t >= _to_dt(start)]
        if end is not None:
            idx = [t for t in idx if t <= _to_dt(end)]
        return idx

    def _val_price(self, t: datetime) -> float:
        if self.price_df is not None:
            return float(self.price_df.loc[pd.Timestamp(t), self.cfg.price_col])
        return float(self._price_aligned.get(t, float("nan")))

    def _val_ef(self, t: datetime) -> float:
        if self.ef_df is not None:
            return float(self.ef_df.loc[pd.Timestamp(t), self.cfg.ef_col])
        return float(self._ef_aligned.get(t, float("nan")))

    def _val_grid(self, t: datetime) -> float:
        if self.grid_meter_df is not None:
            return float(self.grid_meter_df.loc[pd.Timestamp(t), self.cfg.grid_kw_col])
        return float(self._grid_aligned.get(t, float("nan")))

    def _window_grid_max(self, t: datetime, win_steps: int) -> float:
        if self.grid_meter_df is not None:
            idx = self._time_index.get_loc(pd.Timestamp(t))
            start = max(0, idx - win_steps + 1)
            window = self._time_index[start: idx + 1]
            return float(self.grid_meter_df.loc[window, self.cfg.grid_kw_col].max())
        # 纯 python
        idx = self._index.index(t)
        start = max(0, idx - win_steps + 1)
        return max(self._grid_aligned.get(self._index[i], 0.0) for i in range(start, idx + 1))

    def _build_observation_at(self, t: datetime) -> Dict:
        price = self._val_price(t)
        ef = self._val_ef(t)
        grid_kw = self._val_grid(t)
        grid_room = max(0.0, (self.grid.pcc_limit_kw - self.grid.safety_n_1_margin_kw) - grid_kw)

        vehicles = []
        if self.vehicle_state_df is not None and not self.vehicle_state_df.empty:
            try:
                snap = self.vehicle_state_df.xs(pd.Timestamp(t), level="time", drop_level=False)
            except Exception:
                snap = pd.DataFrame(columns=self.vehicle_state_df.columns)
            for vid, row in snap.reset_index().groupby(self.cfg.vehicle_id_col):
                spec = self.vehicles.get(str(vid))
                if not spec:
                    continue
                soc = float(row[self.cfg.soc_col].iloc[-1]) if not row.empty else spec.soc_target - 0.1
                vehicles.append({
                    "vehicle_id": str(vid),
                    "soc": float(np.clip(soc, 0.0, 1.0)),
                    "soc_target": float(spec.soc_target),
                    "p_charge_max_kw": float(spec.p_charge_max_kw),
                    "battery_kwh": float(spec.battery_kwh),
                    "available": int(row[self.cfg.available_col].iloc[-1]) if self.cfg.available_col in row else 1,
                    "priority": int(row[self.cfg.priority_col].iloc[-1]) if self.cfg.priority_col in row else 0,
                    "eta_min": float(row[self.cfg.next_task_eta_min_col].iloc[-1]) if self.cfg.next_task_eta_min_col in row else 0.0,
                    "temp": float(row[self.cfg.temp_col].iloc[-1]) if self.cfg.temp_col in row else 25.0,
                })
        else:
            # 纯 python快照
            snap = self._veh_state_at.get(t, {})
            for vid, spec in self.vehicles.items():
                st = snap.get(vid, {"soc": max(spec.soc_min, spec.soc_target - 0.1),
                                    "available": 1, "eta_min": 0.0, "priority": 0, "temp": 25.0})
                vehicles.append({
                    "vehicle_id": vid,
                    "soc": float(np.clip(st.get("soc", spec.soc_target - 0.1), 0.0, 1.0)),
                    "soc_target": float(spec.soc_target),
                    "p_charge_max_kw": float(spec.p_charge_max_kw),
                    "battery_kwh": float(spec.battery_kwh),
                    "available": int(st.get("available", 1)),
                    "priority": int(st.get("priority", 0)),
                    "eta_min": float(st.get("eta_min", 0.0)),
                    "temp": float(st.get("temp", 25.0)),
                })

        return {
            "time": t,
            "price_yuan_per_kwh": price,
            "ef_kg_per_kwh": ef,
            "grid_kw": grid_kw,
            "grid_room_kw": grid_room,
            "vehicles": vehicles,
            "constraints": {
                "pcc_limit_kw": self.grid.pcc_limit_kw,
                "demand_limit_kw": self.grid.demand_limit_kw,
                "window_min": self.grid.demand_window_min,
                "n_1_margin_kw": self.grid.safety_n_1_margin_kw,
                "dt_min": self.dt_min,
            },
        }

    def _infer_action_from_sessions(self, t: datetime) -> Dict[str, float] | None:
        if self.charge_sessions_df is not None and not self.charge_sessions_df.empty:
            df = self.charge_sessions_df
            mask = (pd.to_datetime(df["start_time"]) <= pd.Timestamp(t)) & (pd.to_datetime(df["end_time"]) > pd.Timestamp(t))
            act = {}
            for _, r in df[mask].iterrows():
                vid = str(r.get(self.cfg.vehicle_id_col, r.get("vehicle_id")))
                pkw = r.get("avg_power_kw", np.nan)
                try:
                    pkw = float(pkw)
                except Exception:
                    pkw = np.nan
                if np.isnan(pkw):
                    if "energy_kwh" in r and "end_time" in r and "start_time" in r:
                        dur_h = max((pd.to_datetime(r["end_time"]) - pd.to_datetime(r["start_time"])).total_seconds()/3600.0, 1e-6)
                        pkw = float(r["energy_kwh"]) / dur_h
                    else:
                        continue
                act[vid] = float(max(0.0, pkw))
            return act

        # 纯 python
        act = {}
        for r in self._sessions:
            st, et = r["start_time"], r["end_time"]
            if st <= t < et:
                vid = str(r.get("vehicle_id", ""))
                pkw = float(r.get("avg_power_kw", 0.0))
                if vid:
                    act[vid] = max(0.0, pkw)
        return act

    def _project_to_feasible(self, t: datetime, action: Dict[str, float]) -> Dict[str, float]:
        obs = self._build_observation_at(t)
        vehicles = {v["vehicle_id"]: v for v in obs["vehicles"]}
        out = {}
        for vid, p in action.items():
            v = vehicles.get(vid)
            if not v:
                continue
            if v["available"] <= 0 or v["soc"] >= v["soc_target"]:
                continue
            pmax = min(v["p_charge_max_kw"], v["battery_kwh"] * self.cfg.max_c_rate)
            out[vid] = float(np.clip(p, 0.0, pmax))
        total = sum(out.values())
        room = obs["grid_room_kw"]
        if total > room + 1e-6 and total > 0:
            scale = room / total
            for vid in list(out.keys()):
                out[vid] = float(out[vid] * scale)
        return out

    def _calc_reward_at(self, t: datetime, action_power_map: Dict[str, float]):
        dt_h = self.dt_min / 60.0
        price = self._val_price(t)
        ef = self._val_ef(t)
        grid_kw = self._val_grid(t)

        energy_kwh = sum(max(0.0, p) * dt_h for p in action_power_map.values())
        elec_cost = energy_kwh * price
        carbon_cost = energy_kwh * ef * float(getattr(self.cfg, "p_CO2_yuan_per_kg", 0.0))

        # 延迟（近似）
        delay_penalty = 0.0
        for v in self._build_observation_at(t)["vehicles"]:
            if v["eta_min"] > 0 and v["soc"] < self.cfg.target_soc:
                delay_penalty += float(v["eta_min"]) / 60.0

        # 峰值/需量
        peak_penalty = 0.0
        if self.grid and self.grid.demand_limit_kw:
            win = int(np.ceil(self.grid.demand_window_min / self.dt_min))
            peak_obs = self._window_grid_max(t, win)
            predicted_peak = max(peak_obs, grid_kw + sum(action_power_map.values()))
            over = predicted_peak - float(self.grid.demand_limit_kw)
            if over > 0:
                peak_penalty = np.log1p(np.exp(over)) * float(self.grid.demand_penalty_per_kw)

        switch_penalty = float(len([p for p in action_power_map.values() if p > 1e-6])) * self.cfg.reward_gamma_switch

        degrade = 0.0
        for vid, p in action_power_map.items():
            spec = self.vehicles.get(vid)
            if not spec:
                continue
            # 取当前温度（若无则 25）
            obs_v = next((x for x in self._build_observation_at(t)["vehicles"] if x["vehicle_id"] == vid), None)
            temp = obs_v["temp"] if obs_v else 25.0
            c_rate = (p / max(spec.battery_kwh, 1e-6))
            temp_factor = max(0.0, (temp - 25.0) / 10.0)
            degrade += max(0.0, c_rate - 0.5) * (1.0 + temp_factor) * dt_h

        reward = -elec_cost - carbon_cost \
                 - self.cfg.reward_alpha_delay * delay_penalty \
                 - self.cfg.reward_beta_peak * peak_penalty \
                 - self.cfg.reward_eta_degrade * degrade \
                 - switch_penalty

        info = {
            "energy_kwh": energy_kwh,
            "elec_cost": elec_cost,
            "carbon_cost": carbon_cost,
            "delay_penalty": delay_penalty,
            "peak_penalty": peak_penalty,
            "switch_penalty": switch_penalty,
            "degrade_penalty": degrade,
            "grid_kw": grid_kw,
            "price_yuan_per_kwh": price,
            "ef_kg_per_kwh": ef,
            "total_power_kw": sum(action_power_map.values()),
        }
        return float(reward), info

    def module_a_data_profile(self) -> Dict:
        """给前端/接口一个稳定的数据口径摘要，便于说明模块A当前吃到的真实数据情况。"""
        latest_t = self._index[-1].isoformat() if self._index else None
        profile = {
            "pipeline": "pandas" if self.price_df is not None and self.ef_df is not None and self.grid_meter_df is not None else "pure-python-mixed",
            "vehicles": len(self.vehicles),
            "chargers": len(self.chargers),
            "time_steps": int(len(self._index or [])),
            "latest_time": latest_t,
            "has_price_series": bool(self.price_df is not None or self._price_aligned),
            "has_ef_series": bool(self.ef_df is not None or self._ef_aligned),
            "has_grid_series": bool(self.grid_meter_df is not None or self._grid_aligned),
            "has_vehicle_states": bool((self.vehicle_state_df is not None and not self.vehicle_state_df.empty) or self._veh_state_at),
            "has_charge_sessions": bool((self.charge_sessions_df is not None and not self.charge_sessions_df.empty) or self._sessions),
            "has_tos_jobs": bool((self.tos_jobs_df is not None and not self.tos_jobs_df.empty) or self._jobs),
            "dt_min": int(self.dt_min),
            "pcc_limit_kw": float(self.grid.pcc_limit_kw) if self.grid else None,
            "demand_limit_kw": float(self.grid.demand_limit_kw) if (self.grid and self.grid.demand_limit_kw is not None) else None,
        }
        (self.artifact_dir / "module_a_data_profile.json").write_text(
            json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return profile

    def summarize_latest_window(self, horizon_hours: int = 12) -> Dict:
        """输出适合首页/接口直接消费的模块A摘要，不要求真实策略执行即可生成。"""
        if self._time_index is None or len(self._time_index) == 0:
            self.load_all()
        steps = max(1, int(np.ceil(horizon_hours * 60 / self.dt_min)))
        tidx = list(self._time_index[-steps:])
        if not tidx:
            return {"ok": False, "reason": "no_time_index"}

        price_vals = [self._val_price(t) for t in tidx]
        ef_vals = [self._val_ef(t) for t in tidx]
        grid_vals = [self._val_grid(t) for t in tidx]
        obs0 = self._build_observation_at(tidx[-1])
        vehicle_count = len(obs0.get("vehicles", []))
        low_soc_count = sum(1 for v in obs0.get("vehicles", []) if float(v.get("soc", 1.0)) < float(v.get("soc_target", self.cfg.target_soc)))

        def _clean_stats(vals):
            arr = np.array([float(v) for v in vals if v is not None and not np.isnan(v)], dtype=float)
            if arr.size == 0:
                return {"min": None, "max": None, "mean": None}
            return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}

        summary = {
            "ok": True,
            "window_hours": int(horizon_hours),
            "time_start": tidx[0].isoformat(),
            "time_end": tidx[-1].isoformat(),
            "vehicle_count": vehicle_count,
            "low_soc_count": low_soc_count,
            "price_stats": _clean_stats(price_vals),
            "ef_stats": _clean_stats(ef_vals),
            "grid_stats": _clean_stats(grid_vals),
            "data_profile": self.module_a_data_profile(),
        }
        (self.artifact_dir / "module_a_latest_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary

    def _aggregate_kpi_safe(self, records: List[Dict]) -> Dict:
        # 优先 pandas 聚合；失败则纯 Python 聚合
        try:
            df = pd.DataFrame(records)
            energy = float(df["energy_kwh"].sum(skipna=True)) if "energy_kwh" in df else 0.0
            cost = float(df.get("elec_cost", 0).sum(skipna=True) + df.get("carbon_cost", 0).sum(skipna=True))
            peak_kw = float(df.get("total_power_kw", 0).max(skipna=True) + df.get("grid_kw", 0).max(skipna=True))
            delay = float(df.get("delay_penalty", 0).sum(skipna=True))
            peak_pen = float(df.get("peak_penalty", 0).sum(skipna=True))
            kpi = {"total_energy_kwh": energy, "total_cost_yuan": cost,
                   "peak_kw_with_charging": peak_kw, "acc_delay_hours": delay,
                   "acc_peak_penalty_yuan": peak_pen}
        except Exception:
            energy = sum(float(r.get("energy_kwh", 0.0)) for r in records)
            cost = sum(float(r.get("elec_cost", 0.0)) + float(r.get("carbon_cost", 0.0)) for r in records)
            peak_kw = max(float(r.get("total_power_kw", 0.0)) + float(r.get("grid_kw", 0.0)) for r in records) if records else 0.0
            delay = sum(float(r.get("delay_penalty", 0.0)) for r in records)
            peak_pen = sum(float(r.get("peak_penalty", 0.0)) for r in records)
            kpi = {"total_energy_kwh": energy, "total_cost_yuan": cost,
                   "peak_kw_with_charging": peak_kw, "acc_delay_hours": delay,
                   "acc_peak_penalty_yuan": peak_pen}

        (self.artifact_dir / "kpi_summary.json").write_text(
            json.dumps(kpi, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return kpi

    # ---------- 配置/通用 ----------

    def _load_config(self, cfg_path: Path) -> AdapterConfig:
        if not cfg_path.exists():
            LOG.warning("config.yaml not found at %s, using defaults.", cfg_path)
            return AdapterConfig()
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        base = AdapterConfig()
        for k, v in raw.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

# ----------------------------
# CLI
# ----------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AGV charge/swap adapter: load & self-check")
    p.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parent),
                   help="directory that contains config.yaml and data/")
    p.add_argument("--config", type=str, default=None, help="path to config.yaml")
    p.add_argument("--self-check", action="store_true", help="run full self-check")
    return p


def main():
    ensure_numpy_pandas_compat()
    args = _build_argparser().parse_args()
    adapter = AGVChargeAdapter(base_dir=Path(args.base_dir), config_path=Path(args.config) if args.config else None)
    if args.self_check:
        adapter.self_check()
    else:
        adapter.load_all()
        t0 = adapter._time_index[0]
        obs = adapter._build_observation_at(t0)
        (adapter.artifact_dir / "sample_observation.json").write_text(
            json.dumps({"time": t0.isoformat(), **{k: v for k, v in obs.items() if k != "time"}},
                       indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOG.info("Loaded data. Sample observation saved.")


if __name__ == "__main__":
    main()
