from __future__ import annotations
from datetime import datetime
from typing import Dict, Optional
import random, math

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 424242)

def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def simulate_dq(asset: str, start: datetime, end: datetime,
                seed: Optional[int]=None) -> Dict[str, float]:
    """
    港口现实感建模：
    - 换班窗口(7–8/12–13/17–18) -> missing/stale 上升
    - DR(14–16)、晚高峰(18–21) -> outlier 上升（工况/策略切换）
    - 周末整体更好
    - 资产差异：QC/YC 现场链路更脆弱（missing↑）；BESS/shore 设备自带缓存（stale↑，missing↓）；
              HVAC/plant SCADA 越界报警多（outlier↑）
    """
    rng = _rng(seed)
    h = start.hour + start.minute/60.0
    weekday = start.weekday()  # 0=Mon..6=Sun
    week_factor = 0.9 if weekday >= 5 else 1.0  # 周末更“绿”

    asset_l = asset.lower()
    is_qc   = asset_l.startswith(("qc","g_","port_g"))
    is_yc   = asset_l.startswith(("yc","f_","port_f"))
    is_bess = asset_l.startswith(("bess","shore"))
    is_hvac = asset_l.startswith(("hvac","plant","cool"))

    # 基线（单位：比例）
    missing = 0.010  # 1.0%
    stale   = 0.012
    outlier = 0.015

    # 资产差异
    if is_qc or is_yc:
        missing += 0.006
        stale   += 0.004
    if is_bess:
        missing -= 0.004
        stale   += 0.006
    if is_hvac:
        outlier += 0.010

    # 换班窗口
    if int(h) in (7,8,12,13,17,18):
        missing += 0.010
        stale   += 0.008

    # DR 与晚高峰
    if 14 <= h < 16:
        outlier += 0.015
    if 18 <= h < 21:
        outlier += 0.020

    # 轻微日内平滑（正弦）
    missing *= (0.9 + 0.1*math.sin(2*math.pi*(h/24))**2)
    stale   *= (0.9 + 0.1*math.cos(2*math.pi*(h/24))**2)
    outlier *= (0.9 + 0.1*math.sin(2*math.pi*((h-6)/24))**2)

    # 周末更好
    missing *= week_factor
    stale   *= week_factor
    outlier *= (0.95 if weekday >= 5 else 1.0)

    # 随机扰动
    missing = _clamp01(missing + rng.uniform(-0.003, 0.003))
    stale   = _clamp01(stale   + rng.uniform(-0.003, 0.003))
    outlier = _clamp01(outlier + rng.uniform(-0.004, 0.004))

    # 合理上/下限（避免太夸张）
    missing = max(0.0, min(0.06, missing))
    stale   = max(0.0, min(0.07, stale))
    outlier = max(0.0, min(0.10, outlier))

    return {"missing": round(missing, 4),
            "stale":   round(stale,   4),
            "outlier": round(outlier, 4)}
