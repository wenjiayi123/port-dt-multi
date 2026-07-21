# -*- coding: utf-8 -*-
"""
app/adapters/ais_tide_client.py

【文件用途（大白话）】
- 统一对接 AIS（船舶当前位置/航迹/ETA 粗估）与 潮汐/潮流（高度/流速/涨落）。
- 未配置真实 REST 时使用可复现模拟器；真实接口失败时默认报错，不静默降级。
- 输出稳定、归一化结构，供 TOS/调度/能耗优化/RL 使用（如靠离泊节拍、岸电功率滚动优化等）。
- 自带审计落盘：快照保存到 data/objects/audit/evt-ais-*.json，便于回放/追责。

【落地接入】
- 首次运行会自动生成 data/objects/config/ais_tide_client.json 的模板。
- 把 base_url/token/各资源路径（ais_live/ais_track/tide 等）填上现场参数后即可切换真实接口。
- 若现场返回本地时间，可通过 tz_offset_min（分钟）统一转换为 UTC。

【与哪些文件关联】
- 可被 app/adapters/tos_client.py（船期）、app/services/schedule.py（作业驱动预测）引用；
- 可被 app/services/optimize.py / rl_env_pro.py 用来对接「AIS 驱动靠离泊节拍 + 岸电功率滚动优化」；
- 审计输出与现有黑匣子一致：data/objects/audit/evt-ais-*.json。

"""

from __future__ import annotations
import os
import json
import time
import uuid
import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

# 可选：没有 requests 也能跑（会走模拟）
try:
    import requests  # type: ignore
except Exception:
    requests = None

# ---------------------------------------------------------------------
# 目录/配置
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIT_DIR = os.path.join(ROOT_DIR, "data", "objects", "audit")
CONFIG_DIR = os.path.join(ROOT_DIR, "data", "objects", "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "ais_tide_client.json")
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class _Cfg:
    """配置加载与默认模板；支持环境变量覆盖。"""
    def __init__(self, path: str = CONFIG_FILE):
        self.path = path
        if not os.path.exists(self.path):
            self._write_default()
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # 环境变量覆盖（可选）
        if os.getenv("AIS_BASE_URL"):
            self.data["ais"]["base_url"] = os.getenv("AIS_BASE_URL")
        if os.getenv("AIS_TOKEN"):
            self.data["ais"].setdefault("auth", {})["token"] = os.getenv("AIS_TOKEN")
        if os.getenv("TIDE_BASE_URL"):
            self.data["tide"]["base_url"] = os.getenv("TIDE_BASE_URL")
        if os.getenv("TIDE_TOKEN"):
            self.data["tide"].setdefault("auth", {})["token"] = os.getenv("TIDE_TOKEN")

    def _write_default(self):
        default = {
            "site_code": "INTL-MEGA-PORT",
            "tz_offset_min": 0,            # 若现场返回本地时间，可通过该偏移（分钟）统一转为 UTC
            "prefer_live_radius_km": 25,   # live_ships 默认搜索半径
            "ais": {
                "base_url": "",            # 真实 AIS 提供商 REST 根地址（空则走模拟）
                "auth": {"token": "REPLACE_WITH_AIS_TOKEN"},
                "paths": {
                    "live": "/api/ais/v1/live",       # 参数: lat,lon,radius_km
                    "track": "/api/ais/v1/track",     # 参数: mmsi, hours
                },
                "timeout_sec": 4,
            },
            "tide": {
                "base_url": "",            # 真实潮汐 REST 根地址（空则走模拟）
                "auth": {"token": "REPLACE_WITH_TIDE_TOKEN"},
                "paths": {
                    "series": "/api/tide/v1/series"   # 参数: lat,lon, from, to
                },
                "timeout_sec": 4,
            },
            "fallback_mock": False,
            "mock_seed": 2233,
            # 港池参考：港口中心（便于 live_ships 模拟/默认定位）
            "port_center": {"lat": 22.555, "lon": 114.055}  # 示例：华南某港
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 数据结构（归一化）
# ---------------------------------------------------------------------
@dataclass
class AISShip:
    mmsi: str
    vessel_name: str
    lat: float
    lon: float
    sog: float              # 航速（kn）
    cog: float              # 航向（度）
    nav_status: str         # UnderWay, AtAnchor, Moored, Restricted...（统一为 TitleCase）
    last_utc: str

@dataclass
class AISTrackPoint:
    mmsi: str
    ts_utc: str
    lat: float
    lon: float
    sog: float
    cog: float

@dataclass
class ETAEstimate:
    mmsi: str
    eta_utc: str
    dist_nm: float          # 距港中心的海里距离（估算）
    sog: float
    reason: str             # 估计说明（方法/假设）

@dataclass
class TidePoint:
    ts_utc: str
    height_m: float         # 潮位高度（相对基准面）
    current_ms: float       # 潮流流速（m/s，若无则 0）
    phase: str              # flood / ebb / slack

# ---------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------
class AISTideClient:
    """
    - AIS：附近船舶、单船航迹、基于距港+SOG 的 ETA 粗估；
    - Tide：潮汐/潮流时间序列；
    - compose_context：供靠泊/岸电优化的统一上下文。
    """
    def __init__(self, config_path: str = CONFIG_FILE):
        self.cfg = _Cfg(config_path)
        random.seed(self.cfg.data.get("mock_seed", 2233))

    def source_status(self) -> Dict[str, Any]:
        ais_configured = bool(str(self.cfg.data.get("ais", {}).get("base_url") or "").strip())
        tide_configured = bool(str(self.cfg.data.get("tide", {}).get("base_url") or "").strip())
        return {
            "adapter": "ais_tide_rest",
            "ais_mode": "live_rest" if self._use_real_ais() else "engineering_simulator",
            "tide_mode": "live_rest" if self._use_real_tide() else "engineering_simulator",
            "ais_configured": ais_configured,
            "tide_configured": tide_configured,
            "http_runtime_available": requests is not None,
            "fallback_on_live_error": bool(self.cfg.data.get("fallback_mock", False)),
            "site_code": self.cfg.data.get("site_code"),
        }

    # ------------------ AIS: 附近船舶（按圆形区域） ------------------
    def live_ships(self, center_lat: Optional[float]=None, center_lon: Optional[float]=None, radius_km: Optional[float]=None) -> List[Dict[str, Any]]:
        if center_lat is None or center_lon is None:
            c = self.cfg.data.get("port_center", {})
            center_lat = float(c.get("lat", 0.0))
            center_lon = float(c.get("lon", 0.0))
        if radius_km is None:
            radius_km = float(self.cfg.data.get("prefer_live_radius_km", 25.0))

        if self._use_real_ais():
            try:
                items = self._http_get("ais", "live", params={"lat": center_lat, "lon": center_lon, "radius_km": radius_km})
                return [self._norm_ship(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(s) for s in self._mock_live_ships(center_lat, center_lon, radius_km)]

    # ------------------ AIS: 航迹（近 N 小时轨迹） ------------------
    def track(self, mmsi: str, hours: int = 6) -> List[Dict[str, Any]]:
        if self._use_real_ais():
            try:
                items = self._http_get("ais", "track", params={"mmsi": mmsi, "hours": hours})
                return [self._norm_track(mmsi, x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(p) for p in self._mock_track(mmsi, hours)]

    # ------------------ AIS: ETA 粗估（基于距港 + SOG） ------------------
    def eta_estimate(self, ships: List[Dict[str, Any]], center_lat: Optional[float]=None, center_lon: Optional[float]=None) -> List[Dict[str, Any]]:
        if center_lat is None or center_lon is None:
            c = self.cfg.data.get("port_center", {})
            center_lat = float(c.get("lat", 0.0))
            center_lon = float(c.get("lon", 0.0))
        out: List[Dict[str, Any]] = []
        for s in ships:
            nm = self._haversine_nm(center_lat, center_lon, s["lat"], s["lon"])
            sog = max(0.1, float(s.get("sog", 0.1)))
            # ETA = 距离 / 速度（小时） → UTC
            hrs = nm / sog
            eta = _utc_now() + timedelta(hours=hrs)
            out.append(asdict(ETAEstimate(
                mmsi=s["mmsi"], eta_utc=_ts_iso(eta),
                dist_nm=round(nm, 2), sog=round(sog, 2),
                reason="dist_nm/sog heuristic"
            )))
        # 近→远排序
        out.sort(key=lambda x: x["dist_nm"])
        return out

    # ------------------ 潮汐/潮流 时间序列 ------------------
    def tide_series(self, lat: Optional[float]=None, lon: Optional[float]=None, start: Optional[datetime]=None, end: Optional[datetime]=None) -> List[Dict[str, Any]]:
        if lat is None or lon is None:
            c = self.cfg.data.get("port_center", {})
            lat = float(c.get("lat", 0.0))
            lon = float(c.get("lon", 0.0))
        if start is None:
            start = _utc_now().replace(minute=0, second=0, microsecond=0)
        if end is None:
            end = start + timedelta(hours=24)

        if self._use_real_tide():
            try:
                items = self._http_get("tide", "series", params={"lat": lat, "lon": lon, "from": _ts_iso(start), "to": _ts_iso(end)})
                return [self._norm_tide(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(p) for p in self._mock_tide_series(start, end)]

    # ------------------ 统一上下文（供靠泊/岸电优化等） ------------------
    def compose_context(self, hours_ahead: int = 24) -> Dict[str, Any]:
        c = self.cfg.data.get("port_center", {})
        lat, lon = float(c.get("lat", 0.0)), float(c.get("lon", 0.0))
        ships = self.live_ships(lat, lon)
        eta = self.eta_estimate(ships, lat, lon)
        start = _utc_now().replace(minute=0, second=0, microsecond=0)
        tide = self.tide_series(lat, lon, start=start, end=start + timedelta(hours=hours_ahead))
        return {
            "site": self.cfg.data.get("site_code"),
            "ts_utc": _ts_iso(_utc_now()),
            "center": {"lat": lat, "lon": lon},
            "ships": ships,
            "eta_estimate": eta,
            "tide_series": tide,
            "_provenance": self.source_status(),
        }

    # -----------------------------------------------------------------
    # HTTP & 归一化
    # -----------------------------------------------------------------
    def _use_real_ais(self) -> bool:
        base = self.cfg.data.get("ais", {}).get("base_url", "").strip()
        return bool(base) and (requests is not None)

    def _use_real_tide(self) -> bool:
        base = self.cfg.data.get("tide", {}).get("base_url", "").strip()
        return bool(base) and (requests is not None)

    def _http_get(self, which: str, key: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        node = self.cfg.data.get(which, {})
        base = node.get("base_url", "").rstrip("/")
        path = node.get("paths", {}).get(key, "")
        if not base or not path:
            raise RuntimeError(f"{which} base_url 或 path 未配置")
        url = f"{base}{path}"
        headers = {"Accept": "application/json"}
        token = node.get("auth", {}).get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        timeout = int(node.get("timeout_sec", 4))
        r = requests.get(url, params=params, headers=headers, timeout=timeout)  # type: ignore
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
        return [data]

    def _apply_tz_offset(self, iso_str: str) -> str:
        try:
            off = int(self.cfg.data.get("tz_offset_min", 0))
            if off == 0:
                return iso_str
            dt = datetime.fromisoformat(iso_str.replace("Z","+00:00"))
            return _ts_iso(dt - timedelta(minutes=off))
        except Exception:
            return iso_str

    def _norm_ship(self, x: Dict[str, Any]) -> Dict[str, Any]:
        ns = (x.get("nav_status") or x.get("status") or "UnderWay").title()
        return asdict(AISShip(
            mmsi=str(x.get("mmsi") or x.get("id") or uuid.uuid4()),
            vessel_name=str(x.get("vessel_name") or x.get("name") or "UNKNOWN"),
            lat=float(x.get("lat") or 0.0),
            lon=float(x.get("lon") or 0.0),
            sog=float(x.get("sog") or x.get("speed") or 0.0),
            cog=float(x.get("cog") or x.get("course") or 0.0),
            nav_status=ns,
            last_utc=self._apply_tz_offset(str(x.get("last_utc") or x.get("ts") or _ts_iso(_utc_now())))
        ))

    def _norm_track(self, mmsi: str, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(AISTrackPoint(
            mmsi=str(mmsi),
            ts_utc=self._apply_tz_offset(str(x.get("ts_utc") or x.get("ts") or _ts_iso(_utc_now()))),
            lat=float(x.get("lat") or 0.0),
            lon=float(x.get("lon") or 0.0),
            sog=float(x.get("sog") or 0.0),
            cog=float(x.get("cog") or 0.0)
        ))

    def _norm_tide(self, x: Dict[str, Any]) -> Dict[str, Any]:
        ph = (x.get("phase") or "slack").lower()
        if ph not in ("flood", "ebb", "slack"):
            ph = "slack"
        return asdict(TidePoint(
            ts_utc=self._apply_tz_offset(str(x.get("ts_utc") or x.get("ts") or _ts_iso(_utc_now()))),
            height_m=float(x.get("height_m") or x.get("height") or 0.0),
            current_ms=float(x.get("current_ms") or x.get("current") or 0.0),
            phase=ph
        ))

    # -----------------------------------------------------------------
    # 模拟器（高拟真）：附近船舶/航迹/潮汐
    # -----------------------------------------------------------------
    def _mock_live_ships(self, lat0: float, lon0: float, radius_km: float) -> List[AISShip]:
        out: List[AISShip] = []
        n = random.randint(6, 14)
        for i in range(n):
            # 在圈内随机一些点（简单近似）
            r = radius_km * math.sqrt(random.random())   # 均匀分布在圆面积
            theta = random.random() * 2 * math.pi
            dlat = (r / 111.0) * math.cos(theta)
            dlon = (r / (111.0 * math.cos(math.radians(lat0)))) * math.sin(theta)
            lat = lat0 + dlat
            lon = lon0 + dlon
            sog = max(0.1, random.gauss(9.0, 3.0))       # kn
            cog = random.random() * 360
            ns = random.choice(["UnderWay", "Moored", "AtAnchor"])
            out.append(AISShip(
                mmsi=f"{412000000 + random.randint(0,99999)}",
                vessel_name=random.choice(["MSC AURORA","COSCO STAR","MAERSK HORIZON","CMA CGM JADE","EMIRATES SKY"]),
                lat=round(lat, 5), lon=round(lon, 5),
                sog=round(sog, 2), cog=round(cog, 1),
                nav_status=ns, last_utc=_ts_iso(_utc_now()-timedelta(minutes=random.randint(0,15)))
            ))
        return out

    def _mock_track(self, mmsi: str, hours: int) -> List[AISTrackPoint]:
        pts: List[AISTrackPoint] = []
        c = self.cfg.data.get("port_center", {})
        lat0, lon0 = float(c.get("lat", 0.0)), float(c.get("lon", 0.0))
        # 以港口为中心，生成一段入港相关的曲线轨迹
        start = _utc_now() - timedelta(hours=hours)
        seg = int(hours * 12)  # 5 min 一个点
        for i in range(seg+1):
            t = start + timedelta(minutes=5*i)
            # 圆弧靠近港口
            ang = 2*math.pi*(1 - i/(seg+1))
            rad_km = 30 * (i/(seg+1))  # 由远到近
            lat = lat0 + (rad_km/111.0)*math.cos(ang)
            lon = lon0 + (rad_km/(111.0*math.cos(math.radians(lat0))))*math.sin(ang)
            sog = max(0.1, 12.0 - 10.0*(i/(seg+1)))  # 逐步减速
            cog = (math.degrees(ang) + 360) % 360
            pts.append(AISTrackPoint(
                mmsi=mmsi, ts_utc=_ts_iso(t),
                lat=round(lat,5), lon=round(lon,5),
                sog=round(sog,2), cog=round(cog,1)
            ))
        return pts

    def _mock_tide_series(self, start: datetime, end: datetime) -> List[TidePoint]:
        out: List[TidePoint] = []
        cur = start
        # 简单正弦拟合（半日潮 ~12.4h），叠加一些噪声与流速估计
        period = 12.4  # 小时
        base = random.uniform(1.0, 1.8)  # 潮差基幅（m）
        mean = random.uniform(0.8, 1.2)  # 平均潮位（m）
        while cur <= end:
            hrs = (cur - start).total_seconds() / 3600.0
            phase = 2*math.pi*hrs/period
            height = mean + base * math.sin(phase) + random.uniform(-0.05, 0.05)
            # 流速 ~ 高度变化率的绝对值（简化）
            dh = (base * (2*math.pi/period) * math.cos(phase))
            current = abs(dh) * 0.6 + random.uniform(0.0, 0.1)  # m/s
            # 相位判断：上升为 flood，下降 ebb，接近 0 为 slack
            if abs(dh) < 0.03:
                ph = "slack"
            else:
                ph = "flood" if dh > 0 else "ebb"
            out.append(TidePoint(
                ts_utc=_ts_iso(cur),
                height_m=round(height, 3),
                current_ms=round(current, 3),
                phase=ph
            ))
            cur += timedelta(minutes=15)
        return out

    # -----------------------------------------------------------------
    # 工具：距离估算/角度
    # -----------------------------------------------------------------
    def _haversine_nm(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """返回两点球面距离（海里）。"""
        R_km = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        km = R_km * c
        return km * 0.539957  # km -> nm

    # -----------------------------------------------------------------
    # 审计/自测
    # -----------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        ctx = self.compose_context(hours_ahead=24)
        return ctx

    def save_audit(self, payload: Dict[str, Any]) -> str:
        ts = int(time.time())
        fn = os.path.join(AUDIT_DIR, f"evt-ais-{ts}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return fn


# ---------------------------------------------------------------------
# 便捷自测（不依赖 server）：
#   python -c "from app.adapters.ais_tide_client import demo_self_test; demo_self_test()"
# ---------------------------------------------------------------------
def demo_self_test() -> None:
    cli = AISTideClient()
    snap = cli.snapshot()
    path = cli.save_audit(snap)

    ships = snap.get("ships", [])
    tide = snap.get("tide_series", [])
    eta  = snap.get("eta_estimate", [])
    print("[AIS/TIDE] snapshot@{} ships={} tide_pts={} eta_items={}".format(
        snap.get("ts_utc"), len(ships), len(tide), len(eta)
    ))
    print("证据文件：", path)
