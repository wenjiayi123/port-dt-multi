# app/services/energyx/api.py
from pathlib import Path
from functools import lru_cache
import json
import math
import random
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter(tags=["energyx"])

# data 目录：app/services/energyx/data
DATA_DIR = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=16)
def _load_json(filename: str) -> Dict[str, Any]:
    """
    从 data 目录加载 JSON 配置。
    真实环境你只需要把同名文件替换为真实港口数据即可。
    """
    registered = {item.name: item for item in DATA_DIR.glob("*.json") if item.is_file()}
    path = registered.get(str(filename))
    if path is None:
        raise FileNotFoundError("registered EnergyX input not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _generate_curve_from_config(cfg: Dict[str, Any], seed: str) -> List[Dict[str, float]]:
    """
    根据 JSON 配置生成一条 0~1 的“出清曲线”（q=相对电量，price=电价）。
    配置示例：market_curves_CNYTN_day_ahead.json / real_time.json
    """
    points = int(cfg.get("points", 60)) or 60
    base_price = float(cfg.get("base_price", 0.4))
    peak_price = float(cfg.get("peak_price", max(base_price + 0.4, 0.8)))
    sin_amp = float(cfg.get("sin_amp", 0.06))
    noise = float(cfg.get("noise", 0.01))
    min_price = float(cfg.get("min_price", 0.05))
    shape = float(cfg.get("shape", 1.6))  # 曲线陡峭程度（1~3）

    rnd = random.Random(seed)
    curve: List[Dict[str, float]] = []
    for i in range(points):
        q = i / (points - 1) if points > 1 else 0.0  # [0,1]
        # 基础：从 base_price 平滑抬升到 peak_price
        base_line = base_price + (peak_price - base_price) * (q ** shape)
        price = base_line + sin_amp * math.sin(q * 2 * math.pi) + rnd.uniform(-noise, noise)
        price = max(min_price, price)
        curve.append({"qty": round(q, 4), "price": round(price, 4)})
    return curve


def _generate_curve_fallback(market: str, region: str, h: int) -> List[Dict[str, float]]:
    """
    找不到配置或解析失败时使用的兜底曲线（保持原来的 demo 效果）。
    """
    rnd = random.Random(f"{market}:{region}:{h}")
    curve: List[Dict[str, float]] = []
    for i in range(60):
        q = i / 59.0                     # 0~1
        price = 0.35 + 0.45 * q + 0.08 * math.sin(q * 6.28) + rnd.uniform(-0.01, 0.01)
        curve.append({"qty": round(q, 4), "price": round(max(0.05, price), 4)})
    return curve


def _pick_clearing_point(curve: List[Dict[str, float]], cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    """
    从曲线中挑选一个“出清点”：
    - 若配置里给了 clearing_qty / clearing_index 就按配置来；
    - 否则默认在 60% 的量附近取一个点。
    """
    if not curve:
        return {"qty": 0.0, "price": 0.0}

    if cfg is not None:
        if "clearing_index" in cfg:
            idx = max(0, min(int(cfg["clearing_index"]), len(curve) - 1))
            return curve[idx]
        if "clearing_qty" in cfg:
            target_q = float(cfg["clearing_qty"])
            # 找到离 target_q 最近的点
            idx = min(range(len(curve)), key=lambda i: abs(curve[i]["qty"] - target_q))
            return curve[idx]

    # 默认：60% 量
    idx = int(0.6 * (len(curve) - 1))
    return curve[idx]


def _benefit_from_config(region: str) -> Dict[str, float]:
    """
    从 benefit_inputs_{region}.json 计算：
    - 电费 energy_yuan
    - 需量费 demand_yuan
    - 碳费 carbon_yuan
    - 碳排放 carbon_kg
    - 总用电量 energy_kwh
    """
    cfg = _load_json(f"benefit_inputs_{region}.json")

    load = list(cfg.get("load_kwh_by_hour", []))
    price = list(cfg.get("price_y_per_kwh_by_hour", []))
    ci = list(cfg.get("carbon_intensity_kg_per_kwh_by_hour", []))

    if not (load and price and ci and len(load) == len(price) == len(ci)):
        raise ValueError("benefit_inputs JSON arrays length mismatch")

    energy_kwh = float(sum(load))
    energy_yuan = float(sum(l * p for l, p in zip(load, price)))
    carbon_kg = float(sum(l * c for l, c in zip(load, ci)))

    demand_yuan = float(cfg.get("demand_yuan", 0.0))
    carbon_price = float(cfg.get("carbon_price_y_per_kg", 0.0))
    carbon_yuan = carbon_kg * carbon_price

    return {
        "energy_yuan": round(energy_yuan, 2),
        "demand_yuan": round(demand_yuan, 2),
        "carbon_yuan": round(carbon_yuan, 2),
        "carbon_kg": round(carbon_kg, 2),
        "energy_kwh": round(energy_kwh, 1),
    }


def _benefit_fallback(h: int) -> Dict[str, float]:
    """
    兜底：保留原来 demo 中的大致数值，方便没有配置文件时也能跑。
    """
    energy_kwh = 18000 + 800 * h / 24.0
    price_y = 0.75
    demand_yuan = 3200
    carbon_price = 0.35  # ¥/kg
    carbon_kg = energy_kwh * 0.12  # 假设 0.12 kg/kWh

    components = {
        "energy_yuan": round(energy_kwh * price_y, 2),
        "demand_yuan": round(demand_yuan, 2),
        "carbon_yuan": round(carbon_kg * carbon_price, 2),
        "carbon_kg": round(carbon_kg, 2),
        "energy_kwh": round(energy_kwh, 1),
    }
    return components


@router.get("/clearing_curve")
def clearing_curve(
    market: str = Query("day_ahead"),
    region: str = Query("CNYTN"),
    h: int = Query(24, ge=1, le=168),
) -> JSONResponse:
    """
    出清曲线接口：
    - market: 日前 / 实时（day_ahead / real_time）
    - region: 港口/区域编码（如 CNYTN），会映射到 data/market_curves_{region}_{market}.json
    - h: 预留参数（小时窗口），目前只参与兜底逻辑的随机种子
    """
    cfg: Dict[str, Any] | None = None
    try:
        cfg = _load_json(f"market_curves_{region}_{market}.json")
        curve = _generate_curve_from_config(cfg, seed=f"{market}:{region}")
        source = "config"
    except Exception:
        curve = _generate_curve_fallback(market, region, h)
        source = "fallback"

    clearing = _pick_clearing_point(curve, cfg)
    return JSONResponse(
        {
            "market": market,
            "region": region,
            "curve": curve,
            "clearing": clearing,
            "source": source,
            "ts": int(time.time()),
        }
    )


@router.get("/breakdown")
def benefit_breakdown(
    region: str = Query("CNYTN"),
    h: int = Query(24, ge=1, le=168),
) -> JSONResponse:
    """
    收益 / 碳减排分解接口：
    - region: 港口/区域编码，映射到 benefit_inputs_{region}.json
    - h: 预留参数（小时数），目前仅在 fallback 中用于缩放演示数据
    """
    try:
        components = _benefit_from_config(region)
        source = "config"
    except Exception:
        components = _benefit_fallback(h)
        source = "fallback"

    total_yuan = (
        components["energy_yuan"] + components["demand_yuan"] + components["carbon_yuan"]
    )
    return JSONResponse(
        {
            "components": components,
            "total_yuan": round(total_yuan, 2),
            "h": h,
            "region": region,
            "source": source,
            "ts": int(time.time()),
        }
    )
