# -*- coding: utf-8 -*-
"""
Shore+BESS 模块 · 适配器/基线/数据集构建（扩展：奖励 + 经济分解 + 对照基线）
位置: app/services/rl_model/shore_bess/adapter.py

说明：
- 仅依赖标准库 + numpy；数据摄取只用 csv/json。
- baseline_dispatch：逐步写出 reward_yuan_step 与完整经济分解、BESS throughput/C-rate、P_other 估计、无BESS对照。
- offline_dataset：info.reward_components 同步写入。
- metrics：输出全分解合计与对“无BESS规则基线”的节省 advantage_yuan。
"""
from __future__ import annotations
import os, sys, csv, json, math, time, argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta, timezone
import numpy as np

# -------------------- 路径与常量 --------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_CANDIDATES = [os.path.join(HERE, "data"), "/mnt/data"]
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
UNIFIED_JSONL = os.path.join(ARTIFACTS_DIR, "shore_bess_outputs.jsonl")
DT_MIN_DEFAULT = 10  # 步长默认 10 min

# 列名别名（统一口径）
ALIAS = {
    "timestamp": ["ts","time","timestamp","datetime","date_time","DateTime"],
    "berth_id": ["berth_id","berth","berthId","berth_name"],
    "call_id": ["call_id","vessel_call_id","call"],
    "eta": ["eta","ETA","arrive_ts","start_ts"],
    "etd": ["etd","ETD","depart_ts","end_ts"],
    "p_req_min": ["p_req_min","P_req_min","pmin_kw","p_req_baseline_kw"],
    "p_req_p50": ["p_req_p50","P_req_p50","p50_kw"],
    "p_req_p90": ["p_req_p90","P_req_p90","p90_kw"],
    "priority": ["priority","prio","weight","is_critical"],
    "pcc_kw": ["p_pcc_kw","pcc_kw","p_total_kw","PCC_kW"],
    "p_bess_kw": ["p_bess_kw","P_bess_kW","bess_kw","bess_power_kw"],
    "soc": ["soc","SOC","soc_frac"],
    "temp_c": ["temp_c","temperature_c","tempC","batt_temp_c"],
    "shore_kw": ["p_shore_kw","P_shore_kW","shore_power_kw","kw"],
    "price_yuan_per_kwh": ["price_yuan_per_kwh","price","ele_price","cny_per_kwh"],
    "ef_kg_per_kwh": ["ef_kg_per_kwh","grid_ef","ef","kg_per_kwh"],
    "cap_kw": ["cap_kw","shore_cap_kw","cap_kW","max_kw","max_kW"],
    "export_allowed": ["export_allowed","allow_export","export"],
    "pf_min": ["pf_min","pf_limit","power_factor_min"],
}

# -------------------- 通用工具 --------------------
def _tz_of_asia_shanghai() -> timezone:
    return timezone(timedelta(hours=8))

def parse_ts_any(s: str, assume_tz: timezone) -> datetime:
    if not s:
        raise ValueError("empty timestamp")
    ss = s.strip().replace("T"," ")
    if ss.endswith("Z"):
        ss = ss[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=assume_tz)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    fmts = ["%Y-%m-%d %H:%M:%S%z","%Y-%m-%d %H:%M%z","%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S","%Y/%m/%d %H:%M"]
    for f in fmts:
        try:
            dt = datetime.strptime(ss, f)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=assume_tz)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    raise ValueError(f"cannot parse timestamp: {s}")

def ts_to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00","Z")

def find_file(filename: str) -> Optional[str]:
    for d in DATA_DIR_CANDIDATES:
        fp = os.path.join(d, filename)
        if os.path.exists(fp):
            return fp
    return None

def read_csv_flex(filepath: str) -> Tuple[List[str], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        headers = list(r.fieldnames or [])
        for line in r:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in line.items()})
    return headers, rows

def map_first_match(d: Dict[str, str], key: str, default=None):
    for k in ALIAS.get(key, [key]):
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default

def to_float_safe(x, default=None):
    try:
        if x in (None, ""): return default
        return float(x)
    except Exception:
        return default

# -------------------- 数据类 --------------------
@dataclass
class Berth:
    berth_id: str
    cap_kw: float

@dataclass
class BESSConfig:
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
    reserve_min_kW: float
    reserve_critical_hours_local: List[int] = field(default_factory=list)

@dataclass
class DemandConfig:
    pcc_limit_kw: float
    soft_cap_kw: float
    penalty_yuan_per_kW: float
    export_allowed: bool
    co2_price_yuan_per_kg: float = 0.0
    alpha_under_supply: float = 120.0
    beta_peak: float = 90.0
    eta_deg: Optional[float] = None   # None → 使用 BESS cycle_cost
    rho_reserve: float = 20.0

# -------------------- 适配器主体 --------------------
class ShoreBESSAdapter:
    def __init__(self, dt_min: int = DT_MIN_DEFAULT):
        self.dt_min = dt_min
        self.timezone_local = _tz_of_asia_shanghai()
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def write_jsonl(self, key: str, payload: Dict[str, Any]):
        rec = {"key": key, **payload}
        with open(UNIFIED_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- 配置 ----------
    def load_configs(self) -> Tuple[BESSConfig, DemandConfig, Dict[str, Berth]]:
        # BESS
        bess_cfg_path = find_file("bess_master.json")
        if bess_cfg_path and os.path.exists(bess_cfg_path):
            with open(bess_cfg_path, "r", encoding="utf-8") as f:
                bj = json.load(f)
        else:
            bj = {
                "rated_power_kW": 10000, "rated_energy_kWh": 40000,
                "eff_ch": 0.95, "eff_dis": 0.95,
                "soc_min": 0.2, "soc_max": 0.9, "soc_target": 0.6,
                "p_ramp_kW_per_step": 2000, "export_allowed": False,
                "cycle_cost_yuan_per_kWh": 0.2,
                "reserve_rules": {"min_reserve_kW": 2000, "critical_hours_local": [19,20,21,22]},
                "timezone": "Asia/Shanghai"
            }
        self.timezone_local = _tz_of_asia_shanghai()
        reserve = bj.get("reserve_rules", {})
        bess_cfg = BESSConfig(
            rated_power_kW=float(bj.get("rated_power_kW", 10000)),
            rated_energy_kWh=float(bj.get("rated_energy_kWh", 40000)),
            eff_ch=float(bj.get("eff_ch", 0.95)),
            eff_dis=float(bj.get("eff_dis", 0.95)),
            soc_min=float(bj.get("soc_min", 0.2)),
            soc_max=float(bj.get("soc_max", 0.9)),
            soc_target=float(bj.get("soc_target", 0.6)),
            p_ramp_kW_per_step=float(bj.get("p_ramp_kW_per_step", 2000)),
            export_allowed=bool(bj.get("export_allowed", False)),
            cycle_cost_yuan_per_kWh=float(bj.get("cycle_cost_yuan_per_kWh", 0.2)),
            reserve_min_kW=float(reserve.get("min_reserve_kW", 2000.0)),
            reserve_critical_hours_local=list(reserve.get("critical_hours_local", []))
        )

        # Demand
        dw_path = find_file("demand_window_config.json")
        if dw_path and os.path.exists(dw_path):
            with open(dw_path, "r", encoding="utf-8") as f:
                dj = json.load(f)
        else:
            dj = {
                "limits": {"pcc_limit_kw": 20000.0, "plant_soft_cap_kw": 18000.0},
                "penalties": {"penalty_yuan_per_kW": 80.0, "export_allowed": False},
                "timezone": "Asia/Shanghai"
            }
        limits = dj.get("limits", {})
        pens = dj.get("penalties", {})
        rewardj = dj.get("reward", {})  # 可选
        demand_cfg = DemandConfig(
            pcc_limit_kw=float(limits.get("pcc_limit_kw", 20000.0)),
            soft_cap_kw=float(limits.get("plant_soft_cap_kw", 18000.0)),
            penalty_yuan_per_kW=float(pens.get("penalty_yuan_per_kW", 80.0)),
            export_allowed=bool(pens.get("export_allowed", False)),
            co2_price_yuan_per_kg=float(rewardj.get("co2_price_yuan_per_kg", pens.get("co2_price_yuan_per_kg", 0.0))),
            alpha_under_supply=float(rewardj.get("alpha_under_supply", 120.0)),
            beta_peak=float(rewardj.get("beta_peak", pens.get("penalty_yuan_per_kW", 80.0))),
            eta_deg=rewardj.get("eta_deg", None),
            rho_reserve=float(rewardj.get("rho_reserve", 20.0)),
        )
        if demand_cfg.eta_deg is None:
            demand_cfg.eta_deg = bess_cfg.cycle_cost_yuan_per_kWh

        # 泊位能力
        berths: Dict[str, Berth] = {}
        b_path = find_file("berths_master.csv")
        if b_path and os.path.exists(b_path):
            headers, rows = read_csv_flex(b_path)
            for r in rows:
                berth = map_first_match(r, "berth_id", None)
                cap = to_float_safe(map_first_match(r, "cap_kw", None), default=None)
                if berth and cap:
                    berths[str(berth)] = Berth(berth_id=str(berth), cap_kw=float(cap))
        if not berths:
            for k in ["B1","B2","B3","B4"]:
                berths[k] = Berth(berth_id=k, cap_kw=6000.0)
        return bess_cfg, demand_cfg, berths

    # ---------- 数据加载 ----------
    def load_ship_calls(self) -> List[Dict[str, Any]]:
        fp = find_file("ship_calls.csv")
        if not fp: return []
        _, rows = read_csv_flex(fp)
        out = []
        for r in rows:
            berth = map_first_match(r, "berth_id")
            if not berth: continue
            try:
                eta = parse_ts_any(map_first_match(r, "eta"), self.timezone_local)
                etd = parse_ts_any(map_first_match(r, "etd"), self.timezone_local)
            except Exception:
                continue
            out.append({
                "berth_id": str(berth),
                "call_id": map_first_match(r, "call_id", ""),
                "eta": eta, "etd": etd,
                "p_req_min": to_float_safe(map_first_match(r, "p_req_min"), 0.0),
                "p_req_p50": to_float_safe(map_first_match(r, "p_req_p50"), 0.0),
                "p_req_p90": to_float_safe(map_first_match(r, "p_req_p90"), 0.0),
                "priority": to_float_safe(map_first_match(r, "priority"), 1.0),
            })
        return out

    def load_market_price(self) -> List[Tuple[datetime, float]]:
        fp = find_file("market_price.csv")
        out = []
        if fp and os.path.exists(fp):
            _, rows = read_csv_flex(fp)
            for r in rows:
                ts_raw = map_first_match(r, "timestamp")
                if not ts_raw: continue
                try:
                    ts = parse_ts_any(ts_raw, self.timezone_local)
                except Exception:
                    continue
                price = to_float_safe(map_first_match(r, "price_yuan_per_kwh"), None)
                if price is None: continue
                out.append((ts, float(price)))
        if not out:
            now = datetime.now(timezone.utc)
            out = [(now - timedelta(days=1), 0.9), (now + timedelta(days=1), 0.9)]
        return sorted(out, key=lambda x: x[0])

    def load_grid_ef(self) -> List[Tuple[datetime, float]]:
        fp = find_file("grid_ef.csv")
        out = []
        if fp and os.path.exists(fp):
            _, rows = read_csv_flex(fp)
            for r in rows:
                ts_raw = map_first_match(r, "timestamp")
                if not ts_raw: continue
                try:
                    ts = parse_ts_any(ts_raw, self.timezone_local)
                except Exception:
                    continue
                ef = to_float_safe(map_first_match(r, "ef_kg_per_kwh"), None)
                if ef is None: continue
                out.append((ts, float(ef)))
        if not out:
            now = datetime.now(timezone.utc)
            out = [(now - timedelta(days=1), 0.6), (now + timedelta(days=1), 0.6)]
        return sorted(out, key=lambda x: x[0])

    def load_grid_meter(self) -> List[Tuple[datetime, float]]:
        fp = find_file("grid_meter.csv")
        if not fp: return []
        _, rows = read_csv_flex(fp)
        out = []
        for r in rows:
            ts_raw = map_first_match(r, "timestamp")
            if not ts_raw: continue
            try:
                ts = parse_ts_any(ts_raw, self.timezone_local)
            except Exception:
                continue
            p = to_float_safe(map_first_match(r, "pcc_kw"), None)
            if p is None: continue
            out.append((ts, float(p)))
        return sorted(out, key=lambda x: x[0])

    def load_bess_tel(self) -> List[Dict[str, Any]]:
        fp = find_file("bess_telemetry.csv")
        if not fp: return []
        _, rows = read_csv_flex(fp)
        out = []
        for r in rows:
            ts_raw = map_first_match(r, "timestamp")
            if not ts_raw: continue
            try:
                ts = parse_ts_any(ts_raw, self.timezone_local)
            except Exception:
                continue
            out.append({
                "ts": ts,
                "p_bess_kw": to_float_safe(map_first_match(r, "p_bess_kw"), 0.0),
                "soc": to_float_safe(map_first_match(r, "soc"), None),
                "temp_c": to_float_safe(map_first_match(r, "temp_c"), None),
            })
        return sorted(out, key=lambda x: x["ts"])

    def load_shore_tel(self, berths: Dict[str, Berth]) -> List[Dict[str, Any]]:
        fp = find_file("shore_power_telemetry.csv")
        if not fp: return []
        headers, rows = read_csv_flex(fp)
        narrow_possible = any(h in headers for h in ALIAS["berth_id"]) and any(h in headers for h in ALIAS["shore_kw"])
        out = []
        if narrow_possible:
            for r in rows:
                ts_raw = map_first_match(r, "timestamp")
                if not ts_raw: continue
                try:
                    ts = parse_ts_any(ts_raw, self.timezone_local)
                except Exception:
                    continue
                berth_id = str(map_first_match(r, "berth_id"))
                p = to_float_safe(map_first_match(r, "shore_kw"), None)
                if berth_id and p is not None:
                    out.append({"ts": ts, "berth_id": berth_id, "p_shore_kw": float(p)})
        else:
            berth_cols: List[Tuple[str, str]] = []
            heads_lower = [h.lower() for h in headers]
            for b in berths.values():
                candidates = [
                    b.berth_id, b.berth_id.lower(),
                    f"{b.berth_id}_kw", f"{b.berth_id.lower()}_kw",
                    f"berth_{b.berth_id}", f"berth_{b.berth_id.lower()}",
                    f"{b.berth_id}_p", f"{b.berth_id}_power",
                ]
                for i, h in enumerate(headers):
                    hl = heads_lower[i]
                    if any(c.lower() in hl for c in candidates):
                        berth_cols.append((h, b.berth_id))
            for r in rows:
                ts_raw = map_first_match(r, "timestamp")
                if not ts_raw: continue
                try:
                    ts = parse_ts_any(ts_raw, self.timezone_local)
                except Exception:
                    continue
                for col, bid in berth_cols:
                    p = to_float_safe(r.get(col), None)
                    if p is None: continue
                    out.append({"ts": ts, "berth_id": bid, "p_shore_kw": float(p)})
        out.sort(key=lambda x: (x["ts"], x["berth_id"]))
        return out

    # ---------- 时间网格与需求 ----------
    def build_time_grid(self, start_utc: datetime, end_utc: datetime) -> List[datetime]:
        grid = []
        t = start_utc
        while t <= end_utc:
            grid.append(t)
            t += timedelta(minutes=self.dt_min)
        return grid

    @staticmethod
    def stepwise_value_at(ts: datetime, series: List[Tuple[datetime, float]]) -> float:
        if not series: return 0.0
        lo = None
        for t, v in series:
            if t <= ts: lo = (t, v)
            else: break
        if lo is None: return series[0][1]
        return lo[1]

    def build_berth_demand_series(
        self, grid: List[datetime], ship_calls: List[Dict[str, Any]], berths: Dict[str, Berth]
    ) -> Dict[str, Dict[datetime, Dict[str, float]]]:
        demand: Dict[str, Dict[datetime, Dict[str, float]]] = {b: {} for b in berths.keys()}
        for ts in grid:
            for b in berths.keys():
                demand[b][ts] = {"p_req_min": 0.0, "p_req_p50": 0.0, "p_req_p90": 0.0, "shore_required": 0.0, "priority": 1.0}
        for sc in ship_calls:
            b = sc["berth_id"]
            if b not in demand: continue
            for ts in grid:
                if sc["eta"] <= ts < sc["etd"]:
                    d = demand[b][ts]
                    d["p_req_min"] = max(d["p_req_min"], sc["p_req_min"] or 0.0)
                    d["p_req_p50"] = max(d["p_req_p50"], sc["p_req_p50"] or d["p_req_min"])
                    d["p_req_p90"] = max(d["p_req_p90"], sc["p_req_p90"] or d["p_req_p50"])
                    d["shore_required"] = 1.0
                    d["priority"] = max(d["priority"], sc.get("priority", 1.0) or 1.0)
        return demand

    def rolling_mean_15(self, past2: List[Tuple[datetime, float]], now_ts: datetime, dt_min: int) -> float:
        """
        工程近似：Δt=10min 时，用最近两点做线性插值构造 15min 均值
        """
        if len(past2) == 0: return 0.0
        if len(past2) == 1: return past2[0][1]
        t0, p0 = past2[-1]
        t1, p1 = past2[-2]
        if t1 > t0: t0, p0, t1, p1 = t1, p1, t0, p0
        p_linear_avg = 0.5 * (p0 + p1)
        avg = (10.0/15.0) * p_linear_avg + (5.0/15.0) * p1
        return avg

    # ---------- 基线生成（含经济分解/奖励） ----------
    def build_reference_trajectory(
        self,
        grid: List[datetime],
        berths: Dict[str, Berth],
        demand: Dict[str, Dict[datetime, Dict[str, float]]],
        bess_cfg: BESSConfig,
        demand_cfg: DemandConfig,
        price_series: List[Tuple[datetime, float]],
        ef_series: List[Tuple[datetime, float]],
        pcc_meter: List[Tuple[datetime, float]] = None,
        init_soc: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if pcc_meter is None: pcc_meter = []
        soc = float(init_soc if init_soc is not None else 0.6)
        soc = min(max(soc, bess_cfg.soc_min), bess_cfg.soc_max)

        prices = [v for _, v in price_series] or [0.9]
        p_lo = float(np.quantile(prices, 0.3))
        p_hi = float(np.quantile(prices, 0.7))

        prev_p_bess = 0.0
        past_points: List[Tuple[datetime, float]] = []
        out: List[Dict[str, Any]] = []
        E_rated = bess_cfg.rated_energy_kWh
        P_max = bess_cfg.rated_power_kW
        dt_h = self.dt_min / 60.0

        def _price_at(ts: datetime) -> float: return self.stepwise_value_at(ts, price_series)
        def _ef_at(ts: datetime) -> float: return self.stepwise_value_at(ts, ef_series)

        # 估计“其他负荷”= PCC表计 - 岸电汇总（若有）
        est_other: Dict[datetime, float] = {t: 0.0 for t in grid}
        if pcc_meter:
            shore_hist = self.load_shore_tel(berths)
            if shore_hist:
                sh_map: Dict[Tuple[datetime, str], float] = {}
                for rec in shore_hist:
                    ts = rec["ts"]
                    idx = int(round((ts - grid[0]).total_seconds() / 60.0 / self.dt_min))
                    if 0 <= idx < len(grid):
                        gts = grid[idx]
                        sh_map[(gts, rec["berth_id"])] = rec["p_shore_kw"]
                shore_sum: Dict[datetime, float] = {t: 0.0 for t in grid}
                for (ts, bid), v in sh_map.items():
                    shore_sum[ts] = shore_sum.get(ts, 0.0) + v
                pcc_map = {t: self.stepwise_value_at(t, pcc_meter) for t in grid}
                for t in grid:
                    est_other[t] = max(0.0, pcc_map[t] - shore_sum.get(t, 0.0))

        for ts in grid:
            # 1) 泊位最小保供
            P_shore_t: Dict[str, float] = {}
            sum_shore_min = 0.0
            for b in berths.values():
                need = demand.get(b.berth_id, {}).get(ts, {"p_req_min": 0.0, "shore_required": 0.0, "priority": 1.0})
                p_need = float(need["p_req_min"] if need["shore_required"] else 0.0)
                p_set = min(max(p_need, 0.0), b.cap_kw)
                P_shore_t[b.berth_id] = p_set
                sum_shore_min += p_set

            # 2) 备用需求（关键时段抬高）
            hour_local = ts.astimezone(self.timezone_local).hour
            base_res = bess_cfg.reserve_min_kW
            r_res = max(base_res, 0.25 * P_max) if hour_local in bess_cfg.reserve_critical_hours_local else base_res

            p_other = est_other.get(ts, 0.0)
            pcc_no_bess = sum_shore_min + p_other

            price = _price_at(ts)
            ef = _ef_at(ts)

            # 3) BESS 可用充放能力与备用
            soc_usable_dis = max(0.0, (soc - bess_cfg.soc_min)) * E_rated / dt_h
            soc_usable_ch  = max(0.0, (bess_cfg.soc_max - soc)) * E_rated / dt_h
            avail_dis = min(P_max, soc_usable_dis * bess_cfg.eff_dis)
            avail_ch  = min(P_max, soc_usable_ch  / bess_cfg.eff_ch)
            avail_dis_after_res = max(0.0, avail_dis - r_res)

            # 4) 规则化基线：削峰优先 + 价差套利 + 斜坡
            target_pcc = min(demand_cfg.soft_cap_kw, demand_cfg.pcc_limit_kw)
            p_bess_cmd = 0.0
            if pcc_no_bess > target_pcc:
                need_dis = min(avail_dis_after_res, pcc_no_bess - target_pcc)
                p_bess_cmd = +need_dis
            if abs(p_bess_cmd) < 1e-6:
                if price <= p_lo and avail_ch > 0.0 and soc < bess_cfg.soc_target * 1.02:
                    p_bess_cmd = -min(avail_ch, 0.5 * P_max)
                elif price >= p_hi and avail_dis_after_res > 0.0 and soc > bess_cfg.soc_target * 0.98:
                    p_bess_cmd = +min(avail_dis_after_res, 0.5 * P_max)

            ramp = bess_cfg.p_ramp_kW_per_step
            p_bess = max(min(p_bess_cmd, prev_p_bess + ramp), prev_p_bess - ramp)

            export_allowed = (bess_cfg.export_allowed and demand_cfg.export_allowed)
            pcc_now = pcc_no_bess - p_bess
            if (not export_allowed) and (pcc_now < 0.0):
                if p_bess < 0:
                    p_bess = -min(abs(p_bess), pcc_no_bess)  # 避免反送
                pcc_now = pcc_no_bess - p_bess

            past_points.append((ts, pcc_now))
            if len(past_points) > 2: past_points = past_points[-2:]
            p_roll15 = self.rolling_mean_15(past_points, ts, self.dt_min)

            # 若滚动均值逼近上限，强制加放电/禁充电
            if p_roll15 > demand_cfg.pcc_limit_kw + 1e-6:
                p_bess = max(p_bess, min(P_max, prev_p_bess + ramp, avail_dis_after_res + (p_bess if p_bess>0 else 0.0)))
                pcc_now = pcc_no_bess - p_bess
                past_points[-1] = (ts, pcc_now)
                p_roll15 = self.rolling_mean_15(past_points, ts, self.dt_min)

            # 5) SOC 与 throughput（kWh）
            e_throughput_kWh = 0.0
            if p_bess >= 0:
                e_out = p_bess * dt_h
                e_cell = e_out / bess_cfg.eff_dis
                soc = min(bess_cfg.soc_max, soc - e_cell / E_rated)
                e_throughput_kWh = e_cell
            else:
                e_in = (-p_bess) * dt_h
                e_cell = e_in * bess_cfg.eff_ch
                soc = max(bess_cfg.soc_min, soc + e_cell / E_rated)
                e_throughput_kWh = e_cell
            prev_p_bess = p_bess

            # 6) 经济分解与奖励
            energy_kWh = max(pcc_now, 0.0) * dt_h
            ele_cost = energy_kWh * price
            co2_kg = energy_kWh * ef
            co2_cost = co2_kg * float(demand_cfg.co2_price_yuan_per_kg or 0.0)

            delta = p_roll15 - demand_cfg.pcc_limit_kw
            if delta <= 0:
                peak_penalty = 0.0
            else:
                smooth = 0.02 * demand_cfg.pcc_limit_kw
                peak_penalty = math.log1p(max(delta, 0.0) / (smooth + 1e-9)) * demand_cfg.penalty_yuan_per_kW

            reserve_short_kW = max(0.0, r_res - avail_dis)
            reserve_penalty = reserve_short_kW * dt_h * float(demand_cfg.rho_reserve or 0.0)

            under_supply_kWh = 0.0  # 参考基线不缺供
            under_supply_penalty = under_supply_kWh * float(demand_cfg.alpha_under_supply or 0.0)

            deg_cost = e_throughput_kWh * float(demand_cfg.eta_deg or 0.0)

            total_cost = ele_cost + co2_cost + peak_penalty + deg_cost + reserve_penalty + under_supply_penalty
            reward = -total_cost

            # “无BESS规则基线”对照（近似）：认为 PCC≈pcc_no_bess
            p_roll15_rule_est = pcc_no_bess
            peak_penalty_rule = 0.0
            if p_roll15_rule_est > demand_cfg.pcc_limit_kw:
                smooth = 0.02 * demand_cfg.pcc_limit_kw
                peak_penalty_rule = math.log1p((p_roll15_rule_est - demand_cfg.pcc_limit_kw) / (smooth + 1e-9)) * demand_cfg.penalty_yuan_per_kW
            energy_rule_kWh = max(pcc_no_bess, 0.0) * dt_h
            cost_rule = energy_rule_kWh * price + peak_penalty_rule

            flags = {
                "UnderSupply": int(under_supply_kWh > 1e-9),
                "ReserveShort": int(reserve_short_kW > 1e-9),
                "ExportBlocked": int((not export_allowed) and pcc_now < 0.0),
                "PeakNearCap": int(p_roll15 > 0.95 * demand_cfg.pcc_limit_kw),
            }

            # 7) C-rate（按额定能量）
            c_rate = abs(p_bess) / max(1e-9, E_rated)

            out.append({
                "ts": ts_to_iso_z(ts),
                "P_shore": P_shore_t,
                "P_bess_kW": round(p_bess, 3),
                "SOC": round(soc, 4),
                "r_res_kW": round(r_res, 3),
                "P_pcc_kW": round(pcc_now, 3),
                "P_roll15_kW": round(p_roll15, 3),
                "P_other_kW_est": round(p_other, 3),

                # 市场因子
                "price_yuan_per_kWh": round(price, 6),
                "ef_kg_per_kWh": round(ef, 6),

                # —— 逐步经济分解 / 奖励 ——
                "energy_kWh_step": round(energy_kWh, 6),
                "ele_cost_yuan_step": round(ele_cost, 6),
                "co2_kg_step": round(co2_kg, 6),
                "co2_cost_yuan_step": round(co2_cost, 6),
                "peak_penalty_yuan_step": round(peak_penalty, 6),
                "deg_cost_yuan_step": round(deg_cost, 6),
                "reserve_short_kW_step": round(reserve_short_kW, 6),
                "reserve_penalty_yuan_step": round(reserve_penalty, 6),
                "under_supply_kWh_step": round(under_supply_kWh, 6),
                "under_supply_penalty_yuan_step": round(under_supply_penalty, 6),
                "total_cost_yuan_step": round(total_cost, 6),
                "reward_yuan_step": round(reward, 6),

                # —— 兼容旧前端 ——（语义不变）
                "cost_yuan": round(ele_cost, 6),
                "deg_cost_yuan": round(deg_cost, 6),
                "penalty_yuan": round(peak_penalty, 6),
                "co2_kg": round(co2_kg, 6),

                # BESS 统计
                "bess_throughput_kWh_step": round(e_throughput_kWh, 6),
                "bess_c_rate": round(c_rate, 6),

                # “无BESS规则基线”估计（近似）
                "P_roll15_rule_est_kW": round(p_roll15_rule_est, 3),
                "peak_penalty_rule_yuan_step": round(peak_penalty_rule, 6),
                "cost_rule_yuan_step": round(cost_rule, 6),

                "flags": flags,
            })
        return out

    # ---------- 离线数据集 ----------
    def build_offline_dataset(self, traj: List[Dict[str, Any]], berths: Dict[str, Berth]) -> List[Dict[str, Any]]:
        berth_order = list(berths.keys())
        dataset = []
        for i, st in enumerate(traj):
            obs = {
                "SOC": st["SOC"],
                "P_bess_kW": st["P_bess_kW"],
                "r_res_kW": st["r_res_kW"],
                "P_pcc_kW": st["P_pcc_kW"],
                "P_roll15_kW": st["P_roll15_kW"],
                "price_yuan_per_kWh": st["price_yuan_per_kWh"],
                "ef_kg_per_kWh": st["ef_kg_per_kWh"],
            }
            for b in berth_order:
                obs[f"P_shore_{b}_kW"] = float(st["P_shore"].get(b, 0.0))
            act = [0.0 for _ in berth_order] + [0.0, 0.0]   # 基线残差=0
            reward = float(st["reward_yuan_step"])
            dataset.append({
                "ts": st["ts"],
                "obs": obs,
                "act": act,
                "reward": reward,
                "done": False,
                "info": {
                    "co2_kg": st["co2_kg"],
                    "reward_components": {
                        "ele_cost_yuan": st["ele_cost_yuan_step"],
                        "co2_cost_yuan": st["co2_cost_yuan_step"],
                        "peak_penalty_yuan": st["peak_penalty_yuan_step"],
                        "deg_cost_yuan": st["deg_cost_yuan_step"],
                        "reserve_penalty_yuan": st["reserve_penalty_yuan_step"],
                        "under_supply_penalty_yuan": st["under_supply_penalty_yuan_step"],
                        "total_cost_yuan": st["total_cost_yuan_step"]
                    }
                }
            })
        return dataset

    # ---------- 主导出 ----------
    def export_all(self, start_utc: datetime, end_utc: datetime) -> Dict[str, Any]:
        bess_cfg, demand_cfg, berths = self.load_configs()
        price = self.load_market_price()
        ef = self.load_grid_ef()
        pcc = self.load_grid_meter()
        ship_calls = self.load_ship_calls()

        grid = self.build_time_grid(start_utc, end_utc)
        demand = self.build_berth_demand_series(grid, ship_calls, berths)

        bess_tel = self.load_bess_tel()
        init_soc = None
        if bess_tel:
            for rec in reversed(bess_tel):
                if rec.get("soc") is not None:
                    init_soc = float(rec["soc"]); break

        traj = self.build_reference_trajectory(
            grid=grid, berths=berths, demand=demand,
            bess_cfg=bess_cfg, demand_cfg=demand_cfg,
            price_series=price, ef_series=ef,
            pcc_meter=pcc, init_soc=init_soc
        )

        # baseline_dispatch（逐步）
        for st in traj:
            self.write_jsonl("baseline_dispatch", st)

        # metrics（汇总 & 对照基线）
        sums = {
            "ele_cost": 0.0, "co2_cost": 0.0, "deg_cost": 0.0, "peak_penalty": 0.0,
            "reserve_penalty": 0.0, "under_supply_penalty": 0.0, "total_cost": 0.0,
            "co2_kg": 0.0, "energy_kWh": 0.0, "peak_roll15_kW": 0.0,
            "rule_cost": 0.0, "rule_peak_penalty": 0.0
        }
        for st in traj:
            sums["ele_cost"] += st["ele_cost_yuan_step"]
            sums["co2_cost"] += st["co2_cost_yuan_step"]
            sums["deg_cost"] += st["deg_cost_yuan_step"]
            sums["peak_penalty"] += st["peak_penalty_yuan_step"]
            sums["reserve_penalty"] += st["reserve_penalty_yuan_step"]
            sums["under_supply_penalty"] += st["under_supply_penalty_yuan_step"]
            sums["total_cost"] += st["total_cost_yuan_step"]
            sums["co2_kg"] += st["co2_kg_step"]
            sums["energy_kWh"] += st["energy_kWh_step"]
            sums["peak_roll15_kW"] = max(sums["peak_roll15_kW"], st["P_roll15_kW"])
            sums["rule_cost"] += st["cost_rule_yuan_step"]
            sums["rule_peak_penalty"] += st["peak_penalty_rule_yuan_step"]

        advantage_yuan = sums["rule_cost"] - sums["total_cost"]
        summary = {
            "window": {"start": ts_to_iso_z(grid[0]), "end": ts_to_iso_z(grid[-1])},
            "kpis": {
                "cost_ref_yuan": round(sums["total_cost"], 3),
                "cost_rule_yuan": round(sums["rule_cost"], 3),
                "advantage_yuan": round(advantage_yuan, 3),
                "ele_cost_yuan": round(sums["ele_cost"], 3),
                "co2_cost_yuan": round(sums["co2_cost"], 3),
                "deg_cost_yuan": round(sums["deg_cost"], 3),
                "peak_penalty_yuan": round(sums["peak_penalty"], 3),
                "reserve_penalty_yuan": round(sums["reserve_penalty"], 3),
                "under_supply_penalty_yuan": round(sums["under_supply_penalty"], 3),
                "co2_kg": round(sums["co2_kg"], 3),
                "energy_kWh": round(sums["energy_kWh"], 3),
                "peak_ref_roll15_kW": round(sums["peak_roll15_kW"], 3),
            }
        }
        self.write_jsonl("metrics", summary)

        # offline_dataset（预训练）
        offline = self.build_offline_dataset(traj, berths)
        for row in offline:
            self.write_jsonl("offline_dataset", row)

        return summary

# -------------------- CLI --------------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Shore+BESS Adapter · 数据适配/基线/数据集构建")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--dt", type=int, default=DT_MIN_DEFAULT)
    ap.add_argument("--self-check", action="store_true")
    return ap.parse_args()

def main():
    args = _parse_args()
    adapter = ShoreBESSAdapter(dt_min=args.dt)

    if args.start:
        start_utc = parse_ts_any(args.start, _tz_of_asia_shanghai())
    else:
        today_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        start_utc = today_utc
    if args.end:
        end_utc = parse_ts_any(args.end, _tz_of_asia_shanghai())
    else:
        end_utc = start_utc + timedelta(hours=24)

    summary = adapter.export_all(start_utc, end_utc)
    print("[adapter] OK. Wrote JSONL to:", UNIFIED_JSONL)
    print("[adapter] Summary:", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
