# ============================================
# app/services/schedule.py
# --------------------------------------------
# 外部/作业驱动统一服务（天气 / 船期(AIS/TOS) / 分时电价&碳因子 / 作业驱动序列）
#
# 用途：
# - 给 Forecast/ Twin/ RL 提供“可落地”的驱动输入；
# - /external/* 接口直接透传本服务的只读数据；
# - 真实接入时，只需替换 _fetch_*_from_source() 三个函数的读取实现。
#
# 数据口径（务必按此对接）：
# 1) weather(...) -> list[{ts, drybulb_C, rh_pct, wind_mps}]
# 2) vessels(...) -> list[{eta, etd, vessel, service, teu, berth, draft_m, call_id}]
# 3) tou_tariff(date, port_code) -> dict{date, blocks:[{start,end,price_yuan_per_kwh,carbon_g_per_kwh,label}]}
# 4) load_drivers(start,end,port_code,assets) -> list[{ts, price_yuan_per_kwh, carbon_g_per_kwh, ambient_C, vessel_demand_index}]
#
# 版权: 直连/示例输出，便于后期落地替换真实数据源。
# ============================================

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import math
import random
import csv
from pathlib import Path

_TZ = timezone.utc
_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# ----------- 内部小工具 ----------- #
def _iso(dt: datetime) -> str:
    return dt.astimezone(_TZ).isoformat()

def _parse_iso(s: str) -> datetime:
    # 支持 'Z' 结尾与无时区
    if s.endswith("Z"):
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(_TZ)
    dt = datetime.fromisoformat(s)
    return (dt if dt.tzinfo else dt.replace(tzinfo=_TZ)).astimezone(_TZ)

def _iter_minutes(start: datetime, end: datetime, step_min: int = 1):
    t = start
    delta = timedelta(minutes=step_min)
    while t <= end:
        yield t
        t += delta

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _day_key(dt: datetime) -> str:
    return dt.astimezone(_TZ).strftime("%Y-%m-%d")

# ----------- 对接配置（真实落地时替换以下取数逻辑） ----------- #
@dataclass
class SourceConfig:
    # 可配置真实文件/API 位置；示例为 data/external 下的 CSV
    weather_csv_dir: Path = _DATA_ROOT / "external" / "weather"
    vessels_csv_dir: Path = _DATA_ROOT / "external" / "vessels"
    tou_csv_dir: Path      = _DATA_ROOT / "external" / "tariff"

class ScheduleService:
    """
    统一对外暴露：
      - weather(start,end,lat,lon)
      - vessels(start,end,port_code)
      - tou_tariff(date,port_code)
      - load_drivers(start,end,port_code,assets)

    真实接入：改写 _fetch_*_from_source() 读取真实 CSV / DB / API；字段口径保持不变。
    """

    def __init__(self, cfg: Optional[SourceConfig] = None, seed: int = 20251006) -> None:
        self.cfg = cfg or SourceConfig()
        random.seed(seed)  # 保证演示可重复

    # ============ 天气 ============ #
    def weather(self, start: str, end: str, lat: float = 31.2304, lon: float = 121.4737) -> List[Dict[str, Any]]:
        """
        返回 [ {ts, drybulb_C, rh_pct, wind_mps}, ... ]，步长=10分钟。
        若本地有 CSV：优先按日读取；否则用可解释的日周期模型合成。
        CSV 字段：ts,drybulb_C,rh_pct,wind_mps
        """
        t0 = _parse_iso(start); t1 = _parse_iso(end)
        rows: List[Dict[str, Any]] = []

        # 优先尝试真实文件
        rows = self._fetch_weather_from_source(t0, t1)
        if rows:
            return rows

        # —— 合成：日变化 + 随机扰动（可落地为“无数据兜底”）—— #
        for t in _iter_minutes(t0, t1, 10):
            # 温度：最低 18℃，最高 31℃（可按季节修正）
            angle = 2 * math.pi * ((t.hour + t.minute/60) / 24.0)
            base_C = 24 + 6 * math.sin(angle - math.pi/2)  # 清晨最低，中午最高
            ambient = base_C + 0.8 * math.sin(2*angle) + random.uniform(-0.6, 0.6)
            rh = _clamp(65 + 15*math.sin(angle+math.pi/3) + random.uniform(-5,5), 35, 95)
            wind = _clamp(3.0 + 1.5*math.sin(angle*1.5) + random.uniform(-0.8, 0.8), 0.2, 9.0)
            rows.append({
                "ts": _iso(t),
                "drybulb_C": round(ambient, 2),
                "rh_pct": round(rh, 1),
                "wind_mps": round(wind, 2),
            })
        return rows

    def _fetch_weather_from_source(self, t0: datetime, t1: datetime) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            # 以“按天 CSV”组织：data/external/weather/2025-10-06.csv
            cur = t0
            while cur.date() <= t1.date():
                p = (self.cfg.weather_csv_dir / f"{_day_key(cur)}.csv")
                if p.exists():
                    with p.open("r", encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            ts = _parse_iso(r["ts"])
                            if t0 <= ts <= t1:
                                out.append({
                                    "ts": _iso(ts),
                                    "drybulb_C": float(r.get("drybulb_C", 24.0)),
                                    "rh_pct": float(r.get("rh_pct", 70.0)),
                                    "wind_mps": float(r.get("wind_mps", 3.0)),
                                })
                cur += timedelta(days=1)
        except Exception:
            return []
        return out

    # ============ 船期（AIS/TOS） ============ #
    def vessels(self, start: str, end: str, port_code: str = "CN_DEMO") -> List[Dict[str, Any]]:
        """
        返回 [ {eta, etd, vessel, service, teu, berth, draft_m, call_id}, ... ]
        优先 CSV；无则依据“工作日密集、周末稀疏”的规则合成。
        """
        t0 = _parse_iso(start); t1 = _parse_iso(end)
        rows = self._fetch_vessels_from_source(t0, t1, port_code)
        if rows:
            return rows

        out: List[Dict[str, Any]] = []
        # 合成：每天 6~12 条靠泊；工作日更多
        cur = t0.replace(hour=0, minute=0, second=0, microsecond=0)
        while cur <= t1:
            day_factor = 1.2 if cur.weekday() < 5 else 0.8
            n_calls = int(6 * day_factor + random.randint(0, int(6 * day_factor)))
            for i in range(n_calls):
                eta = cur + timedelta(minutes=random.randint(0, 22*60))
                dur_h = _clamp(random.gauss(10, 3), 4, 24)
                etd = eta + timedelta(hours=dur_h)
                if etd < t0 or eta > t1:
                    continue
                teu = int(_clamp(random.gauss(4500, 1200), 800, 18000))
                berth = f"B{random.randint(1, 10)}"
                call_id = f"{eta.strftime('%Y%m%d%H%M')}-{berth}-{random.randint(100,999)}"
                out.append({
                    "eta": _iso(eta),
                    "etd": _iso(etd),
                    "vessel": f"MV DEMO-{random.randint(100,999)}",
                    "service": random.choice(["FE3","AE10","PSW","AEX","CIMEX"]),
                    "teu": teu,
                    "berth": berth,
                    "draft_m": round(_clamp(random.gauss(12.0, 1.5), 8.0, 16.5), 1),
                    "call_id": call_id
                })
            cur += timedelta(days=1)
        out.sort(key=lambda x: x["eta"])
        return out

    def _fetch_vessels_from_source(self, t0: datetime, t1: datetime, port_code: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            # 以“按月 CSV”组织：data/external/vessels/2025-10_CN_DEMO.csv
            p = (self.cfg.vessels_csv_dir / f"{t0.strftime('%Y-%m')}_{port_code}.csv")
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        eta = _parse_iso(r["eta"]); etd = _parse_iso(r["etd"])
                        if not (t0 <= etd and eta <= t1):
                            continue
                        out.append({
                            "eta": _iso(eta),
                            "etd": _iso(etd),
                            "vessel": r.get("vessel") or "UNKNOWN",
                            "service": r.get("service") or "",
                            "teu": int(r.get("teu") or 0),
                            "berth": r.get("berth") or "",
                            "draft_m": float(r.get("draft_m") or 0),
                            "call_id": r.get("call_id") or f"{eta:%Y%m%d%H%M}"
                        })
        except Exception:
            return []
        return out

    # ============ 分时电价 & 碳因子 ============ #
    def tou_tariff(self, date: str, port_code: str = "CN_DEMO") -> Dict[str, Any]:
        """
        返回：
        {
          "date":"YYYY-MM-DD",
          "blocks":[
             {"start":"HH:MM","end":"HH:MM","price_yuan_per_kwh":1.10,"carbon_g_per_kwh":120,"label":"valley|flat|peak"},
             ...
          ]
        }
        优先 CSV；无则给出“峰 10-15,19-21；谷 23-07；其余平”的规则。
        """
        # 先尝试外部文件
        from_file = self._fetch_tou_from_source(date, port_code)
        if from_file:
            return from_file

        # 默认口径（可与 server._tou_bucket 对齐）
        return {
            "date": date,
            "blocks": [
                {"start":"00:00","end":"07:00","price_yuan_per_kwh":0.68,"carbon_g_per_kwh":110,"label":"valley"},
                {"start":"07:00","end":"10:00","price_yuan_per_kwh":0.90,"carbon_g_per_kwh":120,"label":"flat"},
                {"start":"10:00","end":"15:00","price_yuan_per_kwh":1.20,"carbon_g_per_kwh":140,"label":"peak"},
                {"start":"15:00","end":"19:00","price_yuan_per_kwh":0.95,"carbon_g_per_kwh":125,"label":"flat"},
                {"start":"19:00","end":"21:00","price_yuan_per_kwh":1.25,"carbon_g_per_kwh":145,"label":"peak"},
                {"start":"21:00","end":"23:00","price_yuan_per_kwh":0.92,"carbon_g_per_kwh":122,"label":"flat"},
                {"start":"23:00","end":"24:00","price_yuan_per_kwh":0.66,"carbon_g_per_kwh":108,"label":"valley"}
            ]
        }

    def _fetch_tou_from_source(self, date: str, port_code: str) -> Dict[str, Any]:
        try:
            p = (self.cfg.tou_csv_dir / f"{date}_{port_code}.csv")
            if not p.exists():
                return {}
            blocks = []
            with p.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    blocks.append({
                        "start": r["start"],
                        "end": r["end"],
                        "price_yuan_per_kwh": float(r.get("price_yuan_per_kwh", 1.0)),
                        "carbon_g_per_kwh": float(r.get("carbon_g_per_kwh", 120.0)),
                        "label": r.get("label") or "flat"
                    })
            return {"date": date, "blocks": blocks}
        except Exception:
            return {}

    # ============ 融合为“作业驱动时间序列” ============ #
    def load_drivers(
        self,
        start: str,
        end: str,
        port_code: str = "CN_DEMO",
        assets: Optional[List[str]] = None,
        step_min: int = 1
    ) -> List[Dict[str, Any]]:
        """
        将天气/船期/分时价融合成统一驱动序列：
        返回 [ {ts, price_yuan_per_kwh, carbon_g_per_kwh, ambient_C, vessel_demand_index}, ... ]
        - vessel_demand_index ∈ [0, 1.0+]：靠泊密度和船舶规模的指数化表征（>1 代表超密集）。
        - price/carbon 来自 TOU；ambient 来自天气。
        """
        t0 = _parse_iso(start); t1 = _parse_iso(end)

        # 取各个源
        wx = self.weather(start, end)
        vs = self.vessels(start, end, port_code=port_code)
        tou = self.tou_tariff(_day_key(t0), port_code=port_code)

        # 预计算：把天气/TOU映射为时间段查找
        wx_by_minute = self._index_weather(wx)
        price_carbon_fn = self._price_carbon_lookup(tou)

        # 船期转“活跃度曲线”（按在港时段叠加权重）
        demand_idx = self._vessel_demand_index(t0, t1, vs, step_min)

        # 组装输出
        out: List[Dict[str, Any]] = []
        for t in _iter_minutes(t0, t1, step_min):
            ts = _iso(t)
            amb = wx_by_minute.get(ts) or wx_by_minute.get(self._align_10min(ts), 24.0)
            price, carbon = price_carbon_fn(t)
            out.append({
                "ts": ts,
                "price_yuan_per_kwh": round(float(price), 4),
                "carbon_g_per_kwh": round(float(carbon), 2),
                "ambient_C": round(float(amb), 2),
                "vessel_demand_index": round(float(demand_idx.get(ts, 0.0)), 4)
            })
        return out

    # ------- 辅助：天气索引 ------- #
    def _index_weather(self, wx: List[Dict[str, Any]]) -> Dict[str, float]:
        out = {}
        for r in wx:
            try:
                out[str(r["ts"])] = float(r.get("drybulb_C", 24.0))
            except Exception:
                continue
        return out

    def _align_10min(self, ts: str) -> str:
        dt = _parse_iso(ts)
        m = (dt.minute // 10) * 10
        return _iso(dt.replace(minute=m, second=0, microsecond=0))

    # ------- 辅助：TOU 查找函数 ------- #
    def _price_carbon_lookup(self, tou: Dict[str, Any]):
        blocks = tou.get("blocks") or []
        def _fn(t: datetime):
            hhmm = f"{t.hour:02d}:{t.minute:02d}"
            for b in blocks:
                if b["start"] <= hhmm < b["end"]:
                    return b["price_yuan_per_kwh"], b["carbon_g_per_kwh"]
            # 落不到块：取平段
            return 0.95, 125.0
        return _fn

    # ------- 辅助：船期 → 需求指数曲线 ------- #
    def _vessel_demand_index(
        self, t0: datetime, t1: datetime, calls: List[Dict[str, Any]], step_min: int
    ) -> Dict[str, float]:
        """
        把 (ETA, ETD, TEU, BERTH) 映射为一个“靠泊强度指数”曲线：
        - 在港每条船贡献 ~ log(1+TEU/1000) 的强度；
        - 同一时刻多船累加；并做平滑（考虑装卸节拍）。
        """
        base: Dict[str, float] = {}
        for c in calls:
            try:
                eta = _parse_iso(c["eta"]); etd = _parse_iso(c["etd"])
                teu = float(c.get("teu", 0) or 0)
                weight = math.log1p(teu / 1000.0)  # 800TEU≈0.59，18000TEU≈2.94
                cur = max(eta, t0); end = min(etd, t1)
                while cur <= end:
                    k = _iso(cur)
                    base[k] = base.get(k, 0.0) + weight
                    cur += timedelta(minutes=step_min)
            except Exception:
                continue

        # 轻微平滑（移动平均 3 点）
        keys = sorted(base.keys())
        smoothed: Dict[str, float] = {}
        for i, k in enumerate(keys):
            v = base[k]
            if i > 0: v += base[keys[i-1]]
            if i+1 < len(keys): v += base[keys[i+1]]
            smoothed[k] = v / (3 if 0 < i < len(keys)-1 else 2)

        # 归一化 [0, 1.5]
        if not smoothed:
            return {}
        vmax = max(smoothed.values()) or 1.0
        return {k: 1.5 * (v / vmax) for k, v in smoothed.items()}
