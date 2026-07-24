from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import math, random

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 13579)

def _clamp(x, a, b): return a if x < a else b if x > b else x

def _tou_price(h: float) -> float:
    """
    分时电价（¥/kWh，演示口径）：
    00-06 低谷 0.50；06-18 平段 0.80；18-21 高峰 1.30；21-24 平段 0.75；
    DR(14-16) 价格外溢（×1.25）。
    """
    base = 0.5 if 0 <= h < 6 else 0.8 if 6 <= h < 18 else 1.3 if 18 <= h < 21 else 0.75
    if 14 <= h < 16: base *= 1.25
    return base

def _grid_ef(h: float) -> float:
    """
    电网排放因子（kg/kWh，演示口径），中午可再生占比高 → 略低。
    """
    # 0.55 ± 0.08 的日内波动
    return 0.55 - 0.08*math.sin(2*math.pi*(h-12)/24)

def _qc_baseload(h: float) -> float:
    # QC 岸桥：白班峰，换班扰动
    v = 55 + 22*math.exp(-((h-10.5)**2)/(2*2.0**2)) + 18*math.exp(-((h-15.0)**2)/(2*2.0**2))
    if int(h) in (7,8,12,13,17,18): v += 10
    return v

def _yc_baseload(h: float) -> float:
    # YC 场桥：傍晚/夜间
    return 40 + 20*math.exp(-((h-19.0)**2)/(2*2.5**2)) + 12*math.exp(-((h-22.0)**2)/(2*2.0**2))

def _hvac_baseload(h: float) -> float:
    # 冷站：午后温控
    return 280 + 80*math.exp(-((h-15.0)**2)/(2*2.5**2))

def _other_baseload(h: float) -> float:
    return 60 + 10*math.sin(2*math.pi*h/24)

def _bess_dispatch_kw(h: float) -> float:
    """
    BESS/岸电：模拟调度功率（+为向电网输出/削减购电，-为充电/购电）。
    - 00-06 充电：-40 kW
    - 14-22 放电：+60 kW，18-21 提升到 +80 kW
    - 其余：0~10 kW 漫游
    """
    if 0 <= h < 6: return -40
    if 18 <= h < 21: return 80
    if 14 <= h < 22: return 60
    return 5*math.sin(2*math.pi*h/24)

def simulate_savings(asset: str, start: datetime, end: datetime,
                     seed: Optional[int]=None) -> Dict[str, float]:
    """
    计算“今日累计”节省金额/减排：
    - 对 QC/YC/HVAC 等“用电负荷类”：在高价/DR/负荷高时段按比例降负荷，节约=ΔkWh×电价，减排=ΔkWh×EF。
    - 对 BESS/shore 等“储能/岸电类”：按策略在高价时段放电（记作节约/收益），计算同上（不把低价充电的成本计负值，演示口径）。
    以上均为分钟积分。
    """
    rng = _rng(seed)
    tz = start.tzinfo or timezone.utc
    day0 = start.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    day1 = day0 + timedelta(days=1)
    t_end = min(end, day1)
    if t_end <= day0:
        return {"cny": 0.0, "co2": 0.0}

    # 逐分钟积分
    total_cny = 0.0
    total_co2 = 0.0

    asset_l = asset.lower()
    is_qc   = asset_l.startswith(("qc","g_","port_g"))
    is_yc   = asset_l.startswith(("yc","f_","port_f"))
    is_bess = asset_l.startswith(("bess","shore"))
    is_hvac = asset_l.startswith(("hvac","plant","cool"))

    t = day0
    while t < t_end:
        h = t.hour + t.minute/60.0
        price = _tou_price(h)
        ef    = _grid_ef(h)

        if is_bess:
            # 只在放电时计“正收益”与“减排”；充电段不计负值（演示）
            p = _bess_dispatch_kw(h)  # kW
            if p > 0:
                kwh = p/60.0
                total_cny += kwh * price * 0.95   # 假设 95% 效率折损
                total_co2 += kwh * ef
        else:
            # 负荷基线
            if is_qc:   base = _qc_baseload(h)
            elif is_yc: base = _yc_baseload(h)
            elif is_hvac: base = _hvac_baseload(h)
            else:       base = _other_baseload(h)

            # 降负荷比例：高价/DR/高负荷更激进
            frac = 0.02
            if 14 <= h < 16: frac += 0.08     # DR
            if 18 <= h < 21: frac += 0.10     # 价格高峰
            if base > (80 if is_qc or is_yc else 320): frac += 0.05
            frac += rng.uniform(-0.01, 0.015) # 噪声
            frac = _clamp(frac, 0.0, 0.25)    # 最高减 25%

            save_kw = base * frac
            kwh = save_kw/60.0
            total_cny += kwh * price
            total_co2 += kwh * ef

        t += timedelta(minutes=1)

    # 圆整到可读口径
    total_cny = round(total_cny, 2)     # 两位小数
    total_co2 = round(total_co2, 0)     # 取整 kg
    return {"cny": total_cny, "co2": total_co2}
