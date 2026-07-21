# ============================================
# app/adapters/schedule_sources.py
# --------------------------------------------
# 真实港口数据对接口径 + 显式开启的工程模拟数据源
#
# 设计目标（直白点）：
# 1) 一份类：ScheduleSources
#    - weather(start,end,lat,lon)     -> [{ts,temp_c,humidity_pct,wind_mps}]
#    - tide(start,end,port_code)      -> [{ts,height_m}]
#    - vessels(start,end,port_code)   -> [{imo,vessel_name,eta,etd,berth,quay_cranes,moves,teus,service}]
#    - tou_tariff(date,port_code)     -> [{start_ts,end_ts,tier,price_cny_per_kwh}]
#    - load_drivers(start,end,port_code,assets) -> dict(供 ForecastService 使用)
#
# 2) 真/模拟双通道：
#    - 默认不生成数据；工程模拟需 PORT_DT_ENABLE_ENGINEERING_SIMULATORS=1
#    - 若设置 PORTDT_REAL=1 且配置了 PORTDT_BASE_URL/PORTDT_API_KEY 等，即调用真实接口
#
# 3) 现场上线只需要：
#    - 把 BASE_URL、API KEY 换成你们的（或电力公司/气象/海事局/AIS 服务）
#    - 如果字段名不同，仅在 _map_* 函数做一次字段映射即可
#
# 4) 与本项目其它模块的关系：
#    - server.py 的 /external/* 直连本类（你上轮已添加）
#    - /api/forecast/*?use_drivers=1 会把本类生成的 drivers 传给 ForecastService
#      目前 ForecastService 只消费 "workload_boost" 规则（start,end,ratio）
#      （未来你也可以扩展更多字段，ForecastService 那边加个小钩子即可）
# ============================================

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


# -----------------------
# 工具：时间/随机数可复现
# -----------------------
def _to_dt(s: str) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def _seed(*xs: Any) -> int:
    # 根据关键入参生成稳定种子，保证相同窗口/港口返回一致数据（便于回放/对账）
    s = "|".join(str(x) for x in xs)
    return abs(hash(s)) % (2**31)


# -----------------------
# 港口参数表（可扩展）
# -----------------------
_PORTS = {
    # 仅示例：经纬度有助于模拟天气与潮汐，真实数据会覆盖
    "CN_DEMO": {"name": "Demo Port", "lat": 31.2304, "lon": 121.4737, "tz": "UTC+8"},
    "CN_SGH":  {"name": "Shanghai",  "lat": 31.2304, "lon": 121.4737, "tz": "UTC+8"},
    "SG_SIN":  {"name": "Singapore", "lat": 1.3521,  "lon": 103.8198, "tz": "UTC+8"},
    "NL_RTM":  {"name": "Rotterdam", "lat": 51.9244, "lon": 4.4777,   "tz": "UTC+1"},
    "CN_SZX":  {"name": "Shenzhen",  "lat": 22.5431, "lon": 114.0579, "tz": "UTC+8"},
}


# ===================================================
# 主类：真实/模拟数据统一在这里
# ===================================================
class ScheduleSources:
    def __init__(self) -> None:
        # 开关：是否启用真实接口
        self.real_enabled = os.getenv("PORTDT_REAL", "0") == "1"
        self.simulation_enabled = os.getenv(
            "PORT_DT_ENABLE_ENGINEERING_SIMULATORS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        # 真实接口基础配置（按你们现场改即可；默认不生效）
        self.base_url = os.getenv("PORTDT_BASE_URL", "https://api.your-port.example/")
        self.api_key  = os.getenv("PORTDT_API_KEY", "")
        # 电力公司/碳价等（如需）
        self.power_url  = os.getenv("PORTDT_POWER_URL", self.base_url)
        self.weather_url= os.getenv("PORTDT_WEATHER_URL", self.base_url)
        self.ais_url    = os.getenv("PORTDT_AIS_URL", self.base_url)

    def source_status(self) -> Dict[str, Any]:
        if self.real_enabled:
            return {"mode": "live_rest", "measured": True, "configured": True}
        if self.simulation_enabled:
            return {"mode": "engineering_simulator", "measured": False, "production": False}
        return {"mode": "unavailable", "measured": False, "reason": "schedule adapter is not configured"}

    # ---------------------------
    # 统一 HTTP GET（仅在 real_enabled 时用）
    # ---------------------------
    def _http_get(self, base: str, path: str, params: Dict[str, Any]) -> Any:
        """
        现场替换点：若你们使用统一 API 网关，这里改 base/path/鉴权即可。
        返回必须为可 JSON 反序列化对象。
        """
        if not self.real_enabled:
            return None
        params = {k: v for k, v in (params or {}).items() if v is not None}
        q = urlencode(params)
        url = urljoin(base, path)
        if q:
            if "?" in url:
                url = f"{url}&{q}"
            else:
                url = f"{url}?{q}"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data)

    # ===================================================
    # 天气：每小时1点（真实：接气象API；模拟：按纬度/季节生成）
    # ===================================================
    def weather(self, start: str, end: str, lat: float, lon: float) -> List[Dict[str, Any]]:
        start_dt, end_dt = _to_dt(start), _to_dt(end)
        if self.real_enabled:
            try:
                raw = self._http_get(
                    self.weather_url,
                    "/weather/hourly",
                    {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "lat": lat, "lon": lon},
                )
                return self._map_weather(raw)
            except Exception:
                raise
        if not self.simulation_enabled:
            return []
        # --- 模拟 ---
        rnd = random.Random(_seed("weather", start_dt, end_dt, lat, lon))
        out = []
        t = start_dt.replace(minute=0, second=0, microsecond=0)
        while t <= end_dt:
            # 简化的日变化 + 季节因子（北半球）
            doy = int(t.timetuple().tm_yday)
            season = math.cos(2 * math.pi * (doy - 200) / 365.0)  # 夏季温高
            daily = math.cos(2 * math.pi * (t.hour - 15) / 24.0)  # 15点最高温
            base_temp = 18 + 10 * (-season)  # 冬季低、夏季高
            temp = base_temp + 5 * (-daily) + rnd.uniform(-0.8, 0.8)
            humi = max(35, min(95, 70 + rnd.uniform(-15, 15)))
            wind = max(0.0, min(12.0, 3 + 2 * rnd.random() + (1.0 if humi > 80 else 0.0)))
            out.append({"ts": _iso(t), "temp_c": round(temp, 1), "humidity_pct": round(humi, 1), "wind_mps": round(wind, 1), "_source": "engineering_simulation"})
            t += timedelta(hours=1)
        return out

    def _map_weather(self, raw: Any) -> List[Dict[str, Any]]:
        """真实接口字段映射（按现场改）；期望输出如 weather() 所述。"""
        try:
            arr = []
            for r in raw or []:
                arr.append({
                    "ts": r.get("time") or r.get("ts"),
                    "temp_c": float(r.get("temp_c") or r.get("temperature") or 20.0),
                    "humidity_pct": float(r.get("humidity") or 70.0),
                    "wind_mps": float(r.get("wind_mps") or r.get("wind_speed") or 3.0),
                })
            return arr
        except Exception:
            return []

    # ===================================================
    # 潮汐：半日潮（~12h25m）；真实：海事/水文站；模拟：余弦+噪声
    # ===================================================
    def tide(self, start: str, end: str, port_code: str) -> List[Dict[str, Any]]:
        start_dt, end_dt = _to_dt(start), _to_dt(end)
        if self.real_enabled:
            try:
                raw = self._http_get(
                    self.base_url,
                    "/hydro/tide",
                    {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "port": port_code},
                )
                return self._map_tide(raw)
            except Exception:
                raise
        if not self.simulation_enabled:
            return []
        # --- 模拟 ---
        rnd = random.Random(_seed("tide", start_dt, end_dt, port_code))
        out = []
        t = start_dt.replace(minute=0, second=0, microsecond=0)
        # 半日潮周期 12h25m
        period = 12 * 3600 + 25 * 60
        amp = 1.8 if port_code.startswith("CN") else 1.2
        while t <= end_dt:
            phi = ((t - start_dt).total_seconds() % period) / period * 2 * math.pi
            h = amp * math.cos(phi) + rnd.uniform(-0.1, 0.1)
            out.append({"ts": _iso(t), "height_m": round(h, 2), "_source": "engineering_simulation"})
            t += timedelta(minutes=30)
        return out

    def _map_tide(self, raw: Any) -> List[Dict[str, Any]]:
        try:
            arr = []
            for r in raw or []:
                arr.append({
                    "ts": r.get("time") or r.get("ts"),
                    "height_m": float(r.get("height_m") or r.get("height") or 0.0),
                })
            return arr
        except Exception:
            return []

    # ===================================================
    # 船期（AIS 近似）：真实：TOS/AIS；模拟：按泊位/箱量/岸桥数生成
    # ===================================================
    def vessels(self, start: str, end: str, port_code: str) -> List[Dict[str, Any]]:
        start_dt, end_dt = _to_dt(start), _to_dt(end)
        if self.real_enabled:
            try:
                raw = self._http_get(
                    self.ais_url,
                    "/vessels/schedule",
                    {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "port": port_code},
                )
                return self._map_vessels(raw)
            except Exception:
                raise
        if not self.simulation_enabled:
            return []
        # --- 模拟 ---
        rnd = random.Random(_seed("vessels", start_dt, end_dt, port_code))
        # 估计窗口长度与靠泊数量
        hours = max(1.0, (end_dt - start_dt).total_seconds() / 3600.0)
        count = max(1, int(hours / 6) + (1 if rnd.random() > 0.5 else 0))
        out = []
        for i in range(count):
            # 随机靠泊时间与作业时长（2.5h ~ 10h）
            eta = start_dt + timedelta(hours=rnd.uniform(0, hours * 0.8))
            work_h = rnd.uniform(2.5, 10.0)
            etd = eta + timedelta(hours=work_h)
            # 箱量/吊机分配
            teus = int(rnd.uniform(800, 3500))
            moves = int(teus * rnd.uniform(0.9, 1.1))
            cranes_n = 2 + int(rnd.random() * 3)  # 2~4台岸桥
            berth = f"B{1 + int(rnd.random() * 8):02d}"
            cranes = [f"QC-{j:02d}" for j in range(1, cranes_n + 1)]
            imo = 9000000 + rnd.randint(1000, 9999)
            out.append({
                "imo": imo,
                "vessel_name": f"MV-{imo}",
                "eta": _iso(eta),
                "etd": _iso(etd),
                "berth": berth,
                "quay_cranes": cranes,
                "moves": moves,
                "teus": teus,
                "service": "FE1",
                "_source": "engineering_simulation",
            })
        # 按 ETA 排序
        out.sort(key=lambda x: x["eta"])
        return out

    def _map_vessels(self, raw: Any) -> List[Dict[str, Any]]:
        try:
            arr = []
            for r in raw or []:
                arr.append({
                    "imo": r.get("imo") or r.get("ship_imo"),
                    "vessel_name": r.get("vessel_name") or r.get("name"),
                    "eta": r.get("eta") or r.get("arrive_time"),
                    "etd": r.get("etd") or r.get("depart_time"),
                    "berth": r.get("berth") or r.get("berth_code"),
                    "quay_cranes": r.get("quay_cranes") or r.get("qcs") or [],
                    "moves": int(r.get("moves") or r.get("moves_est") or 0),
                    "teus": int(r.get("teus") or r.get("teu") or 0),
                    "service": r.get("service") or "UNK",
                })
            return arr
        except Exception:
            return []

    # ===================================================
    # 峰谷电价（TOU）：真实：电网/售电侧；模拟：常见三段/四段式
    # ===================================================
    def tou_tariff(self, date: str, port_code: str) -> List[Dict[str, Any]]:
        # 真实：按日返回每个时段分段电价
        if self.real_enabled:
            try:
                raw = self._http_get(
                    self.power_url,
                    "/power/tou",
                    {"date": date, "port": port_code},
                )
                return self._map_tou(raw)
            except Exception:
                raise
        if not self.simulation_enabled:
            return []
        # --- 模拟 ---
        rnd = random.Random(_seed("tou", date, port_code))
        # 简化：谷 23:00–7:00，峰 10:00–15:00、19:00–21:00，其余平
        base = 0.72 if port_code.startswith("CN") else 0.28  # CNY/kWh 或 EUR/kWh 仅示例
        tiers = [
            ("00:00", "07:00", "valley", base * 0.7),
            ("07:00", "10:00", "flat",   base * 1.0),
            ("10:00", "15:00", "peak",   base * 1.5),
            ("15:00", "19:00", "flat",   base * 1.1),
            ("19:00", "21:00", "peak",   base * 1.6),
            ("21:00", "23:00", "flat",   base * 1.0),
            ("23:00", "24:00", "valley", base * 0.7),
        ]
        out = []
        d0 = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        for hhmm_s, hhmm_e, tier, price in tiers:
            sdt = d0.replace(hour=int(hhmm_s[:2]), minute=int(hhmm_s[3:]), second=0, microsecond=0)
            edt = d0.replace(hour=int(hhmm_e[:2]), minute=int(hhmm_e[3:]), second=0, microsecond=0)
            out.append({
                "start_ts": _iso(sdt),
                "end_ts": _iso(edt),
                "tier": tier,
                "price_cny_per_kwh": round(price * (0.95 + 0.1 * rnd.random()), 4),
                "_source": "engineering_simulation",
            })
        return out

    def _map_tou(self, raw: Any) -> List[Dict[str, Any]]:
        try:
            arr = []
            for r in raw or []:
                arr.append({
                    "start_ts": r.get("start_ts") or r.get("start"),
                    "end_ts": r.get("end_ts") or r.get("end"),
                    "tier": r.get("tier") or r.get("period"),
                    "price_cny_per_kwh": float(r.get("price_cny_per_kwh") or r.get("price") or 0.6),
                })
            return arr
        except Exception:
            return []

    # ===================================================
    # 生成预测“驱动”：把船期等转成 ForecastService 能吃的 workload_boost
    # ===================================================
    def load_drivers(
        self,
        start: str,
        end: str,
        port_code: str = "CN_DEMO",
        assets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        返回示例：
        {
          "workload_boost": [
             {"start":"2025-10-06T08:00:00Z","end":"2025-10-06T10:30:00Z","ratio":1.15},
             {"start":"...","end":"...","ratio":0.92}
          ],
          "meta": {"port":"CN_DEMO","notes":"...","count_vessels":3}
        }
        """
        if not self.real_enabled and not self.simulation_enabled:
            return {
                "workload_boost": [],
                "meta": {"available": False, "source": "unavailable", "port": port_code},
            }
        start_dt, end_dt = _to_dt(start), _to_dt(end)
        port = _PORTS.get(port_code, _PORTS["CN_DEMO"])

        # 1) 拉船期（真实或模拟）
        ships = self.vessels(start, end, port_code)

        # 2) 估算作业强度 -> 生成“负荷放大窗口”
        boosts = []  # [{start,end,ratio}]
        for v in ships:
            try:
                eta = _to_dt(v.get("eta"))
                etd = _to_dt(v.get("etd"))
                if eta > end_dt or etd < start_dt:
                    continue  # 不在窗口内
                # 裁剪到窗口
                s = max(eta, start_dt)
                e = min(etd, end_dt)

                cranes = max(1, len(v.get("quay_cranes") or []))
                moves = max(200, int(v.get("moves") or v.get("teus") or 800))
                # 简化：单位时间强度 ~ 吊机数 * moves 密度
                hours = max(0.5, (e - s).total_seconds() / 3600.0)
                density = moves / hours
                # 把强度映射到负荷放大因子（1.05 ~ 1.35）
                ratio = 1.05 + min(0.30, (cranes * density) / 20000.0)

                boosts.append({"start": _iso(s), "end": _iso(e), "ratio": round(ratio, 3)})
            except Exception:
                continue

        # 3) 夜间/低温/台风等也可影响（此处给示例：夜间按少量抑制，天气示例留接口）
        # 你也可以在此叠加 weather/tide/tou 的影响
        #   - 夜间 00:00-06:00，非靠泊窗口：ratio ~ 0.93
        #   - 价格峰段（若你愿意让负荷“自抑制”）：ratio ~ 0.97
        try:
            tou = self.tou_tariff(start.split("T")[0], port_code)
        except Exception:
            tou = []
        # 夜间抑制：只加在“没有船期”的时段，避免过度叠加
        t = start_dt
        while t < end_dt:
            window = (t, min(end_dt, t + timedelta(hours=2)))
            # 如果 2 小时窗口里没有靠泊，给一个轻微抑制
            hit = False
            for b in boosts:
                bs, be = _to_dt(b["start"]), _to_dt(b["end"])
                if not (window[1] <= bs or window[0] >= be):
                    hit = True
                    break
            if not hit and (0 <= t.hour < 6):
                boosts.append({"start": _iso(window[0]), "end": _iso(window[1]), "ratio": 0.93})
            t += timedelta(hours=2)

        # 4) 合并/去重（简单相邻合并）
        boosts = self._merge_boosts(boosts)

        return {
            "workload_boost": boosts,
            "meta": {
                "port": port_code,
                "count_vessels": len(ships),
                "source": self.source_status()["mode"],
                "notes": "workload_boost 由船期/夜间等生成，预测服务将使用它调整未来负荷",
            },
        }

    def _merge_boosts(self, boosts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not boosts:
            return []
        # 先按开始时间排序
        arr = sorted(boosts, key=lambda b: b["start"])
        out: List[Dict[str, Any]] = []
        cur = arr[0].copy()
        for b in arr[1:]:
            cs, ce, cr = _to_dt(cur["start"]), _to_dt(cur["end"]), float(cur["ratio"])
            bs, be, br = _to_dt(b["start"]), _to_dt(b["end"]), float(b["ratio"])
            # 如果时间上重叠/相邻且 ratio 很接近，则合并
            if bs <= ce + timedelta(minutes=5) and abs(br - cr) <= 0.04:
                cur["end"] = _iso(max(ce, be))
                cur["ratio"] = round((cr + br) / 2.0, 3)
            else:
                out.append(cur)
                cur = b.copy()
        out.append(cur)
        return out
