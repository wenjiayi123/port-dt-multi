from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/multiport", tags=["multiport"])

# 模块根目录：.../app/services/multiport/
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 统一文件命名：summary_{slug}.json
DEFAULT_PORTS = ["Shanghai", "Singapore", "Rotterdam"]

def _slug(port: str) -> str:
    return "".join(ch for ch in port if ch.isalnum()).lower()

def _round(x: float, nd: int = 0) -> float:
    return round(x, nd)

def _default_payload(port: str) -> Dict[str, Any]:
    """
    生成“跨港口概要”示例数据，结构与现有 summary_snapshot.json 对齐并扩展：
      meta: {port, currency, window}
      kpis: {throughput_teu, vessel_calls, truck_turn_min, crane_uptime_pct, shore_power_h,
            energy_kwh, peak_kw, bill_cny, co2_ton}
      trend: [{t, throughput_teu, energy_kwh, bill_cny, co2_ton}]
    数值可直接替换为真实数据；本示例会根据港口名做轻微扰动。
    """
    port_name = port.strip()
    seed = (sum(ord(c) for c in port_name) % 7) + 1
    mul = 1.0 + seed * 0.03
    cur = "CNY"
    # 规模基线（上港 > 新加坡 > 鹿特丹 只是示例，无业务含义）
    base = {
        "shanghai": dict(teu=4200000, calls=1650, energy=9.6e7, peak=10500, bill=1.8e8, co2=5200),
        "singapore": dict(teu=3600000, calls=1500, energy=8.1e7, peak=9800, bill=1.55e8, co2=4700),
        "rotterdam": dict(teu=2400000, calls=1100, energy=5.2e7, peak=8200, bill=1.02e8, co2=3600),
    }.get(_slug(port_name), dict(teu=2500000, calls=1200, energy=5.5e7, peak=8000, bill=1.0e8, co2=3500))

    throughput_teu = int(base["teu"] * mul)
    vessel_calls = int(base["calls"] * mul)
    energy_kwh = int(base["energy"] * mul)
    peak_kw = int(base["peak"] * (0.95 + 0.1 * (seed / 8)))
    bill_cny = int(base["bill"] * (0.95 + 0.1 * (seed / 8)))
    co2_ton = int(base["co2"] * (0.9 + 0.12 * (seed / 8)))

    # 运营效率类
    truck_turn_min = _round(52 - seed * 1.7, 1)
    crane_uptime_pct = _round(96 - seed * 0.6, 1)
    shore_power_h = _round(2800 + seed * 60, 0)

    # 趋势 30 天（相对 t：-29…0）
    trend: List[Dict[str, Any]] = []
    for i in range(30):
        t = i - 29
        factor = 0.92 + (i % 7) * 0.02 + seed * 0.005
        trend.append({
            "t": t,
            "throughput_teu": int(throughput_teu / 30 * factor),
            "energy_kwh": int(energy_kwh / 30 * (0.95 + (seed * 0.01))),
            "bill_cny": int(bill_cny / 30 * factor),
            "co2_ton": _round(co2_ton / 30 * factor, 1),
        })

    return {
        "meta": {"port": port_name, "currency": cur, "window": "P30D"},
        "kpis": {
            "throughput_teu": throughput_teu,
            "vessel_calls": vessel_calls,
            "truck_turn_min": truck_turn_min,
            "crane_uptime_pct": crane_uptime_pct,
            "shore_power_h": shore_power_h,
            "energy_kwh": energy_kwh,
            "peak_kw": peak_kw,
            "bill_cny": bill_cny,
            "co2_ton": co2_ton
        },
        "trend": trend
    }

def _read_or_create(port: str) -> Dict[str, Any]:
    """
    读取 data/summary_{slug}.json；若不存在：
      1) 若存在老文件 summary_snapshot.json，则回退读取（兼容历史，当作“Shanghai”）；
      2) 否则按默认规则生成，并写入本模块 data/ 目录。
    """
    slug = _slug(port)
    f = DATA_DIR / f"summary_{slug}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))

    legacy = DATA_DIR / "summary_snapshot.json"
    if legacy.exists() and slug in {"shanghai", "default"}:
        try:
            return json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:
            pass

    payload = _default_payload(port)
    try:
        f.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return payload

def _list_ports() -> List[str]:
    ports: List[str] = []
    for p in DATA_DIR.glob("summary_*.json"):
        name = p.stem.replace("summary_", "")
        if name:
            ports.append(name.capitalize())
    if not ports:
        return DEFAULT_PORTS
    return sorted(set(ports))

@router.get("/ports")
def list_ports() -> Dict[str, Any]:
    """列出可用港口（由 data/summary_*.json 推断，若空则返回默认清单）"""
    return {"ports": _list_ports()}

@router.get("/summary")
def get_summary(
    port: str = Query("Shanghai", description="港口名：Shanghai/Singapore/Rotterdam…"),
) -> Dict[str, Any]:
    """
    返回指定港口的多指标概要：
    - 优先读 data/summary_{slug}.json
    - 兼容 data/summary_snapshot.json（老文件，视为 Shanghai）
    - 若文件缺失，则自动生成示例 JSON 并写入模块目录
    """
    try:
        payload = _read_or_create(port)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load summary failed: {e}")
    return payload

@router.get("/compare")
def compare(
    ports: str = Query("Shanghai,Singapore", description="用逗号分隔的港口名列表")
) -> Dict[str, Any]:
    """批量返回多个港口的 summary，便于前端做对比/切换。"""
    result: Dict[str, Any] = {}
    for name in [p.strip() for p in ports.split(",") if p.strip()]:
        try:
            result[name] = _read_or_create(name)
        except Exception as e:
            result[name] = {"error": str(e)}
    return {"items": result}

# 说明：
# 1) 该路由不依赖外部系统，数据均在本模块 data/ 下；真实落地时替换同名 JSON 即可；
# 2) 结构与现有 summary_snapshot.json 对齐并向后兼容；
# 3) 在 app/server.py 中 include 本路由即可生效：
#    from app.services.multiport.service import router as multiport_router
#    app.include_router(multiport_router)
