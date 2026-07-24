from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Optional
import math, random

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 271828)

def _gauss(x, mu, sigma):  # 简单高斯凸起
    return math.exp(-((x-mu)**2)/(2*sigma*sigma))

def _sigmoid(z):
    return 1/(1+math.exp(-z))

def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def simulate_peak_risk(asset: str, now: datetime, horizon_end: datetime,
                       seed: Optional[int]=None) -> Dict[str, float]:
    rng = _rng(seed)
    asset_l = asset.lower()
    is_qc   = asset_l.startswith(("qc","g_","port_g"))
    is_yc   = asset_l.startswith(("yc","f_","port_f"))
    is_bess = asset_l.startswith(("bess","shore"))
    is_hvac = asset_l.startswith(("hvac","plant","cool"))

    # —— 基线（工作日>周末） —— #
    weekday = now.weekday()  # 0=Mon..6=Sun
    base = 0.18 if weekday < 5 else 0.12
    if is_qc:   base += 0.08
    if is_yc:   base += 0.06
    if is_bess: base += 0.04
    if is_hvac: base += 0.05

    # —— 小时型谱 —— #
    h = now.hour + now.minute/60.0
    # 价格高峰 18-21
    price_peak = 0.25 * (1 if 18 <= h <= 21 else 0)
    # DR 14-16
    dr_bump    = 0.18 * (1 if 14 <= h <= 16 else 0)
    # QC 白班两峰 + 换班扰动
    qc_shape = 0.0
    if is_qc:
        qc_shape += 0.28 * _gauss(h, 10.5, 2.0)
        qc_shape += 0.22 * _gauss(h, 15.0, 2.0)
        if int(h) in (7,8,12,13,17,18): qc_shape += 0.07
    # YC 傍晚夜间
    yc_shape = (0.22 * _gauss(h, 19.0, 2.5) + 0.10 * _gauss(h, 22.0, 2.0)) if is_yc else 0.0
    # HVAC 午后
    hvac_shape = (0.18 * _gauss(h, 15.0, 2.5)) if is_hvac else 0.0
    # BESS: DR/峰价段动作多
    bess_shape = (0.12 if is_bess and 14 <= h <= 22 else 0.0)

    # —— 潮汐（12h）靠泊影响：极值附近更“紧” —— #
    phase = ((now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()/3600.0) % 12
    tide = 0.10 * (math.sin(2*math.pi*phase)**2) * (1.0 if is_qc else 0.4)

    # —— 外生冲击：作业堆积/天气 —— #
    backlog = 0.0
    if rng.random() < (0.15 if is_qc or is_yc else 0.08):
        backlog = rng.uniform(0.06, 0.15)  # 有时段突然拥挤
    weather = 0.0
    if rng.random() < 0.10:
        weather = rng.uniform(0.04, 0.10)  # 阵风或高温

    # 汇总“当下”强度（logit 输入）
    z_now = (base + price_peak + dr_bump + qc_shape + yc_shape + hvac_shape
             + bess_shape + tide + backlog + weather
             + rng.uniform(-0.03, 0.03))
    p_now = _clamp01(_sigmoid(3.0*(z_now - 0.35)))  # 平移缩放到(0..1)

    # —— 随着时距增长，风险上浮一些（不减性） —— #
    # 远期加入“更多不确定性 + 价格峰/DR 窗口潜在靠近”的溢价
    p15 = _clamp01(p_now + 0.02 + rng.uniform(-0.01, 0.01))
    p30 = _clamp01(max(p15, p_now + 0.07 + rng.uniform(-0.02, 0.02)))
    p60 = _clamp01(max(p30, p_now + 0.13 + rng.uniform(-0.03, 0.03)))

    return {"m15": p15, "m30": p30, "m60": p60}
