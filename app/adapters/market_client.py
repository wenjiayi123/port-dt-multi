# -*- coding: utf-8 -*-
"""
app/adapters/market_client.py

【文件用途（大白话）】
- 对接“电力公司需量/现货、碳价/绿证、DR 事件、边际排放因子”等外部市场与政策信号。
- 未配置真实 REST 时使用可复现模拟器；真实接口失败时默认报错，不静默降级。
- 输出稳定、归一化的结构，供成本优化/DR仿真/碳核算/报表使用。
- 自带审计落盘：快照存到 data/objects/audit/evt-market-*.json，便于回放/追责。

【落地接入】
- 编辑 data/objects/config/market_client.json（本文件首次运行自动生成模板）。
- base_url/token/各资源path 按现场实际填写即可切换真实接口；时区、货币、单位都可在配置里覆盖。

"""

from __future__ import annotations
import os
import csv
import json
import time
import uuid
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

# 可选依赖：没有 requests 也能跑（会走模拟）
try:
    import requests  # type: ignore
except Exception:
    requests = None

# ---------------------------------------------------------------------
# 目录与配置文件
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIT_DIR = os.path.join(ROOT_DIR, "data", "objects", "audit")
CONFIG_DIR = os.path.join(ROOT_DIR, "data", "objects", "config")
FACTORS_DIR = os.path.join(ROOT_DIR, "data", "factors")
GRID_FACTORS_CSV = os.path.join(FACTORS_DIR, "grid_factors.csv")
CONFIG_FILE = os.path.join(CONFIG_DIR, "market_client.json")
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class _Cfg:
    """配置加载与默认模板。支持环境变量覆盖。"""
    def __init__(self, path: str = CONFIG_FILE):
        self.path = path
        if not os.path.exists(self.path):
            self._write_default()
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # 环境变量覆盖（可选）
        if os.getenv("MARKET_BASE_URL"):
            self.data["base_url"] = os.getenv("MARKET_BASE_URL")
        if os.getenv("MARKET_TOKEN"):
            self.data.setdefault("auth", {})["token"] = os.getenv("MARKET_TOKEN")

    def _write_default(self):
        default = {
            "region": "CN-EXAMPLE",
            "currency": "CNY",
            "base_url": "",  # 真实市场 REST 根地址（空则走模拟）
            "auth": {"token": "REPLACE_WITH_MARKET_TOKEN"},
            "paths": {
                "day_ahead": "/api/market/v1/day_ahead",       # 参数: date=YYYY-MM-DD
                "real_time": "/api/market/v1/real_time",       # 参数: from,to (ISO)
                "demand_charge": "/api/market/v1/demand_charge", # 参数: month=YYYY-MM
                "demand_limit": "/api/market/v1/demand_limit",   # 当前/本月kW上限
                "dr_events": "/api/market/v1/dr_events",       # 参数: from,to (ISO)
                "carbon_price": "/api/market/v1/carbon_price", # 参数: from,to (ISO) or date
                "grid_factor": "/api/market/v1/grid_factor",   # 参数: from,to (ISO)
                "rec_price": "/api/market/v1/rec_price"        # 参数: date
            },
            "timeout_sec": 4,
            "fallback_mock": False,
            "mock_seed": 4317,
            "tz_offset_min": 0  # 若现场返回本地时区，可设偏移（分钟）统一转UTC
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 归一化数据结构
# ---------------------------------------------------------------------
@dataclass
class SpotPrice:
    ts_utc: str
    price: float            # 电价（货币/kWh）
    currency: str

@dataclass
class DemandChargeWindow:
    window: str             # 例如 "peak"|"partial_peak"|"valley"
    start_hhmm: str         # 本地时段，如 "09:00"
    end_hhmm: str           # 本地时段，如 "11:00"
    price_per_kw: float     # 需量罚金（货币/kW）

@dataclass
class DemandLimit:
    month: str              # YYYY-MM
    contract_kw: float      # 合同需量/本月需量上限（kW）
    measured_peak_kw: float # 截止当前已发生的最大需量
    penalty_per_kw: float   # 超契约罚金（货币/kW）

@dataclass
class DREvent:
    event_id: str
    start_utc: str
    end_utc: str
    shed_target_kw: float   # 要求削减的 kW（基于基线）
    incentive_per_kwh: float# 奖励（货币/kWh）
    program: str            # 事件/项目名称

@dataclass
class CarbonPrice:
    ts_utc: str
    cny_per_ton: float

@dataclass
class GridFactor:
    ts_utc: str
    kg_co2e_per_kwh: float

@dataclass
class RECPrice:
    date: str               # YYYY-MM-DD
    cny_per_mwh: float


# ---------------------------------------------------------------------
# 市场客户端
# ---------------------------------------------------------------------
class MarketClient:
    """
    - 对接日内/日前/实时电价、需量/需量罚金、DR事件、碳价、绿证、边际排放因子；
    - 真实接口优先；无配置/不通 → 模拟；
    - 提供 compose_signals() 聚合出优化/RL需要的连续信号（价格/碳因子/DR标志/需量上限）。
    """
    def __init__(self, config_path: str = CONFIG_FILE):
        self.cfg = _Cfg(config_path)
        random.seed(self.cfg.data.get("mock_seed", 4317))

    def source_status(self) -> Dict[str, Any]:
        configured = bool(str(self.cfg.data.get("base_url") or "").strip())
        return {
            "adapter": "market_rest",
            "mode": "live_rest" if self._use_real() else "engineering_simulator",
            "configured": configured,
            "http_runtime_available": requests is not None,
            "fallback_on_live_error": bool(self.cfg.data.get("fallback_mock", False)),
            "region": self.cfg.data.get("region"),
            "currency": self.cfg.data.get("currency"),
        }

    # ------------------------- 公共查询 -------------------------
    def day_ahead_price(self, date: datetime) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("day_ahead", params={"date": date.strftime("%Y-%m-%d")})
                return [self._norm_price(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        # 24h × 1h/15min 间隔
        return [asdict(p) for p in self._mock_price_series(date, freq_min=60)]

    def real_time_price(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("real_time", params={"from": _ts_iso(start), "to": _ts_iso(end)})
                return [self._norm_price(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(p) for p in self._mock_price_series(start, end=end, freq_min=15)]

    def demand_charge(self, month: str) -> List[Dict[str, Any]]:
        """返回月度需量价窗口（峰/平/谷等）的罚金标准（货币/kW）。"""
        if self._use_real():
            try:
                items = self._http_get("demand_charge", params={"month": month})
                return [self._norm_demand_charge(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(w) for w in self._mock_demand_windows()]

    def demand_limit(self, month: str) -> Dict[str, Any]:
        if self._use_real():
            try:
                x = self._http_get_one("demand_limit", params={"month": month})
                return asdict(self._norm_demand_limit(x))
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return asdict(self._mock_demand_limit(month))

    def dr_events(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("dr_events", params={"from": _ts_iso(start), "to": _ts_iso(end)})
                return [self._norm_dr(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(e) for e in self._mock_dr_events(start, end)]

    def carbon_price(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("carbon_price", params={"from": _ts_iso(start), "to": _ts_iso(end)})
                return [self._norm_carbon_price(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(c) for c in self._mock_carbon_price(start, end)]

    def grid_factor(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """电网边际排放因子（kgCO2e/kWh），优先真实接口；其次尝试读取 data/factors/grid_factors.csv；否则模拟。"""
        if self._use_real():
            try:
                items = self._http_get("grid_factor", params={"from": _ts_iso(start), "to": _ts_iso(end)})
                return [self._norm_grid_factor(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        # CSV 尝试
        csv_items = self._read_grid_factors_csv(start, end)
        if csv_items:
            return csv_items
        # 模拟
        return [asdict(g) for g in self._mock_grid_factor(start, end)]

    def rec_price(self, date: datetime) -> Dict[str, Any]:
        if self._use_real():
            try:
                x = self._http_get_one("rec_price", params={"date": date.strftime("%Y-%m-%d")})
                return asdict(self._norm_rec_price(x, date))
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return asdict(self._mock_rec_price(date))

    # ------------------------- 聚合信号 -------------------------
    def compose_signals(self, start: datetime, end: datetime, prefer: str = "RT") -> Dict[str, Any]:
        """
        生成优化/RL可直接使用的连续信号：
        - price[t]：电价（currency/kWh）
        - carbon_factor[t]：kgCO2e/kWh
        - dr_flag[t]：0/1
        - demand_cap_kw：合同上限
        - penalty_per_kw：月超契约罚金
        """
        if prefer.upper() == "RT":
            price = self.real_time_price(start, end)
            if not price:
                price = self.day_ahead_price(start)
        else:
            price = self.day_ahead_price(start)

        gf = self.grid_factor(start, end)
        cp = self.carbon_price(start, end)
        dr = self.dr_events(start, end)
        dl = self.demand_limit(start.strftime("%Y-%m"))

        # 归一化到同一时间索引（以 price 为主）
        idx = [p["ts_utc"] for p in price]
        gf_by_ts = {g["ts_utc"]: g for g in gf}
        cp_by_ts = {c["ts_utc"]: c for c in cp}
        dr_flags = self._dr_to_flags(dr, idx)

        out: List[Dict[str, Any]] = []
        for p in price:
            ts = p["ts_utc"]
            out.append({
                "ts_utc": ts,
                "price": p["price"],
                "currency": p.get("currency", self.cfg.data.get("currency","CNY")),
                "kg_co2e_per_kwh": gf_by_ts.get(ts, {"kg_co2e_per_kwh": 0.58}).get("kg_co2e_per_kwh", 0.58),
                "carbon_price_cny_per_ton": cp_by_ts.get(ts, {"cny_per_ton": 60.0}).get("cny_per_ton", 60.0),
                "dr_flag": 1 if ts in dr_flags else 0
            })

        return {
            "series": out,
            "demand_cap_kw": float(dl.get("contract_kw", 0.0)),
            "penalty_per_kw": float(dl.get("penalty_per_kw", 0.0)),
            "measured_peak_kw": float(dl.get("measured_peak_kw", 0.0)),
            "currency": self.cfg.data.get("currency", "CNY"),
            "region": self.cfg.data.get("region", "CN-EXAMPLE"),
            "_provenance": self.source_status(),
        }

    # ------------------------- HTTP & 归一化 -------------------------
    def _use_real(self) -> bool:
        base = self.cfg.data.get("base_url", "").strip()
        return bool(base) and (requests is not None)

    def _http_get(self, key: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = self.cfg.data.get("base_url", "").rstrip("/")
        path = self.cfg.data.get("paths", {}).get(key, "")
        if not base or not path:
            raise RuntimeError("market base_url 或 path 未配置")
        url = f"{base}{path}"
        headers = {"Accept": "application/json"}
        token = self.cfg.data.get("auth", {}).get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        timeout = int(self.cfg.data.get("timeout_sec", 4))
        r = requests.get(url, params=params, headers=headers, timeout=timeout)  # type: ignore
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
        return [data]

    def _http_get_one(self, key: str, params: Dict[str, Any]) -> Dict[str, Any]:
        items = self._http_get(key, params)
        return items[0] if items else {}

    def _apply_tz(self, iso_str: str) -> str:
        try:
            off = int(self.cfg.data.get("tz_offset_min", 0))
            if off == 0:
                return iso_str
            dt = datetime.fromisoformat(iso_str.replace("Z","+00:00"))
            return _ts_iso(dt - timedelta(minutes=off))
        except Exception:
            return iso_str

    def _norm_price(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(SpotPrice(
            ts_utc=self._apply_tz(str(x.get("ts_utc") or x.get("ts") or _ts_iso(_utc_now()))),
            price=float(x.get("price") or x.get("value") or 0.7),
            currency=str(x.get("currency") or self.cfg.data.get("currency","CNY"))
        ))

    def _norm_demand_charge(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(DemandChargeWindow(
            window=str(x.get("window") or x.get("period") or "peak"),
            start_hhmm=str(x.get("start_hhmm") or x.get("start") or "09:00"),
            end_hhmm=str(x.get("end_hhmm") or x.get("end") or "11:00"),
            price_per_kw=float(x.get("price_per_kw") or x.get("penalty") or 40.0)
        ))

    def _norm_demand_limit(self, x: Dict[str, Any]) -> DemandLimit:
        return DemandLimit(
            month=str(x.get("month") or _utc_now().strftime("%Y-%m")),
            contract_kw=float(x.get("contract_kw") or 5000.0),
            measured_peak_kw=float(x.get("measured_peak_kw") or 4200.0),
            penalty_per_kw=float(x.get("penalty_per_kw") or 60.0)
        )

    def _norm_dr(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(DREvent(
            event_id=str(x.get("event_id") or x.get("id") or uuid.uuid4()),
            start_utc=self._apply_tz(str(x.get("start_utc") or x.get("start") or _ts_iso(_utc_now()))),
            end_utc=self._apply_tz(str(x.get("end_utc") or x.get("end") or _ts_iso(_utc_now()+timedelta(hours=2)))),
            shed_target_kw=float(x.get("shed_target_kw") or 1500.0),
            incentive_per_kwh=float(x.get("incentive_per_kwh") or 1.0),
            program=str(x.get("program") or "DR-PROGRAM-A")
        ))

    def _norm_carbon_price(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(CarbonPrice(
            ts_utc=self._apply_tz(str(x.get("ts_utc") or x.get("ts") or _ts_iso(_utc_now()))),
            cny_per_ton=float(x.get("cny_per_ton") or x.get("price") or 60.0)
        ))

    def _norm_grid_factor(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(GridFactor(
            ts_utc=self._apply_tz(str(x.get("ts_utc") or x.get("ts") or _ts_iso(_utc_now()))),
            kg_co2e_per_kwh=float(x.get("kg_co2e_per_kwh") or x.get("factor") or 0.58)
        ))

    def _norm_rec_price(self, x: Dict[str, Any], date: datetime) -> RECPrice:
        return RECPrice(
            date=(x.get("date") or date.strftime("%Y-%m-%d")),
            cny_per_mwh=float(x.get("cny_per_mwh") or x.get("price") or 120.0)
        )

    # ------------------------- CSV & flags -------------------------
    def _read_grid_factors_csv(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """尝试读取 data/factors/grid_factors.csv；容错字段名：timestamp/ts, kg_* or factor."""
        if not os.path.exists(GRID_FACTORS_CSV):
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(GRID_FACTORS_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("timestamp") or row.get("ts") or row.get("time")
                    if not ts:
                        continue
                    try:
                        dt = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                    except Exception:
                        continue
                    if dt < start or dt > end:
                        continue
                    # 找一个看起来像因子的列
                    k = None
                    for cand in row.keys():
                        lc = cand.lower()
                        if "kg" in lc or "factor" in lc:
                            k = cand
                            break
                    val = float(row.get(k, 0.58)) if k else 0.58
                    out.append({"ts_utc": _ts_iso(dt), "kg_co2e_per_kwh": val})
        except Exception:
            return []
        return out

    def _dr_to_flags(self, dr_list: List[Dict[str, Any]], ts_list: List[str]) -> set:
        flags = set()
        if not dr_list:
            return flags
        ts_dt = [datetime.fromisoformat(ts.replace("Z","+00:00")) for ts in ts_list]
        for e in dr_list:
            s = datetime.fromisoformat(e["start_utc"].replace("Z","+00:00"))
            t = datetime.fromisoformat(e["end_utc"].replace("Z","+00:00"))
            for i, dt in enumerate(ts_dt):
                if s <= dt <= t:
                    flags.add(ts_list[i])
        return flags

    # ------------------------- 模拟器 -------------------------
    def _mock_price_series(self, start: datetime, end: Optional[datetime] = None, freq_min: int = 60) -> List[SpotPrice]:
        if end is None:
            end = (start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        out: List[SpotPrice] = []
        cur = start.replace(second=0, microsecond=0, tzinfo=timezone.utc)
        while cur <= end:
            # 峰谷价型（举例）：谷 0.45，平 0.68，峰 0.95，尖峰 1.25（CNY/kWh）
            h = cur.hour
            if 0 <= h < 7:
                price = 0.45
            elif 7 <= h < 10:
                price = 0.95
            elif 10 <= h < 15:
                price = 0.68
            elif 15 <= h < 21:
                price = 1.10 if h in (18,19) else 0.95
            else:
                price = 0.68
            # 轻微扰动
            price = round(price * random.uniform(0.97, 1.03), 4)
            out.append(SpotPrice(ts_utc=_ts_iso(cur), price=price, currency=self.cfg.data.get("currency","CNY")))
            cur += timedelta(minutes=freq_min)
        return out

    def _mock_demand_windows(self) -> List[DemandChargeWindow]:
        return [
            DemandChargeWindow(window="peak", start_hhmm="09:00", end_hhmm="11:00", price_per_kw=45.0),
            DemandChargeWindow(window="peak", start_hhmm="18:00", end_hhmm="20:00", price_per_kw=60.0),
            DemandChargeWindow(window="valley", start_hhmm="00:00", end_hhmm="07:00", price_per_kw=10.0),
            DemandChargeWindow(window="flat", start_hhmm="10:00", end_hhmm="15:00", price_per_kw=25.0),
        ]

    def _mock_demand_limit(self, month: str) -> DemandLimit:
        return DemandLimit(month=month, contract_kw=5000.0, measured_peak_kw=4230.0, penalty_per_kw=65.0)

    def _mock_dr_events(self, start: datetime, end: datetime) -> List[DREvent]:
        out: List[DREvent] = []
        span_h = int((end - start).total_seconds() // 3600)
        # 以晚高峰为中心随机 0~2 个事件
        n = random.randint(0, 2)
        for i in range(n):
            st = (start + timedelta(hours=random.randint(16, 20))).replace(minute=0, second=0, microsecond=0)
            ed = st + timedelta(hours=random.choice([1, 2]))
            out.append(DREvent(
                event_id=f"DR-{int(time.time())}-{i}",
                start_utc=_ts_iso(st),
                end_utc=_ts_iso(ed),
                shed_target_kw=random.randint(800, 1800),
                incentive_per_kwh=round(random.uniform(0.6, 1.2), 2),
                program=random.choice(["CPP","TOU-DR","EDRP"])
            ))
        return out

    def _mock_carbon_price(self, start: datetime, end: datetime) -> List[CarbonPrice]:
        out: List[CarbonPrice] = []
        cur = start.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        while cur <= end:
            # 模拟：60–80 CNY/t 随机波动
            val = round(random.uniform(60.0, 80.0), 2)
            out.append(CarbonPrice(ts_utc=_ts_iso(cur), cny_per_ton=val))
            cur += timedelta(hours=1)
        return out

    def _mock_grid_factor(self, start: datetime, end: datetime) -> List[GridFactor]:
        out: List[GridFactor] = []
        cur = start.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        while cur <= end:
            # 简例：夜间 0.55，白天 0.62，晚高峰 0.7（kgCO2e/kWh）
            h = cur.hour
            if 0 <= h < 7:
                f = 0.55
            elif 7 <= h < 10:
                f = 0.62
            elif 10 <= h < 15:
                f = 0.60
            elif 15 <= h < 21:
                f = 0.70 if h in (18,19) else 0.64
            else:
                f = 0.60
            f = round(f * random.uniform(0.98, 1.02), 4)
            out.append(GridFactor(ts_utc=_ts_iso(cur), kg_co2e_per_kwh=f))
            cur += timedelta(hours=1)
        return out

    def _mock_rec_price(self, date: datetime) -> RECPrice:
        # 示例：REC 价格 80–180 CNY/MWh 区间
        return RECPrice(date=date.strftime("%Y-%m-%d"), cny_per_mwh=round(random.uniform(80.0, 180.0), 2))

    # ------------------------- 审计 -------------------------
    def snapshot(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> Dict[str, Any]:
        """抓取或模拟一个区间的市场快照，供落盘/回放使用。"""
        if start is None:
            start = _utc_now().replace(minute=0, second=0, microsecond=0)
        if end is None:
            end = start + timedelta(hours=24)
        out = {
            "region": self.cfg.data.get("region"),
            "currency": self.cfg.data.get("currency", "CNY"),
            "ts_utc": _ts_iso(_utc_now()),
            "day_ahead": self.day_ahead_price(start),
            "real_time": self.real_time_price(start, end),
            "demand_charge": self.demand_charge(start.strftime("%Y-%m")),
            "demand_limit": self.demand_limit(start.strftime("%Y-%m")),
            "dr_events": self.dr_events(start, end),
            "carbon_price": self.carbon_price(start, end),
            "grid_factor": self.grid_factor(start, end),
            "rec_price": self.rec_price(start),
            "signals": self.compose_signals(start, end),
        }
        return out

    def save_audit(self, payload: Dict[str, Any]) -> str:
        ts = int(time.time())
        fn = os.path.join(AUDIT_DIR, f"evt-market-{ts}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return fn


# ---------------------------------------------------------------------
# 便捷自测：
#   python -c "from app.adapters.market_client import demo_self_test; demo_self_test()"
# ---------------------------------------------------------------------
def demo_self_test() -> None:
    cli = MarketClient()
    snap = cli.snapshot()
    path = cli.save_audit(snap)

    rt = snap.get("real_time", [])
    dr = snap.get("dr_events", [])
    dl = snap.get("demand_limit", {})
    gf = snap.get("grid_factor", [])
    print("[MARKET] snapshot@{}  rt_points={}  dr_events={}  contract_kw={}  peak_measured_kw={}  grid_pts={}".format(
        snap.get("ts_utc"), len(rt), len(dr), dl.get("contract_kw"), dl.get("measured_peak_kw"), len(gf)
    ))
    print("证据文件：", path)
