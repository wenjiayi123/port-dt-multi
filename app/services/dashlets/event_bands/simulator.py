# app/services/dashlets/event_bands/simulator.py
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 42)

def _clamp(t: datetime, a: datetime, b: datetime) -> datetime:
    return max(a, min(b, t))

def simulate_event_bands(asset: str, start: datetime, end: datetime,
                         seed: Optional[int]=None) -> List[Dict[str, Any]]:
    """
    生成一组“事件时间带”：
    - 电价高峰/低谷（演示规则）
    - DR 事件（随机 1~2 段）
    - 作业高强度（按设备类型）
    - 潮汐极值（QC 才显示）
    - 策略执行窗口（示意）
    """
    rng = _rng(seed)
    start = start.replace(second=0, microsecond=0)
    end   = end.replace(second=0, microsecond=0)
    bands: List[Dict[str,Any]] = []

    # 1) 电价 TOU：高峰 18:00-21:00；低谷 00:00-06:00
    cur = start
    while cur < end:
        day0 = cur.replace(hour=0, minute=0, second=0, microsecond=0)
        peak0, peak1 = day0.replace(hour=18), day0.replace(hour=21)
        valley0, valley1 = day0, day0.replace(hour=6)
        bands += [
            {"t0": _clamp(peak0, start, end), "t1": _clamp(peak1, start, end),
             "label":"电价高峰", "kind":"price_peak", "color":"rgba(245,158,11,0.12)"},
            {"t0": _clamp(valley0, start, end), "t1": _clamp(valley1, start, end),
             "label":"低谷", "kind":"price_valley", "color":"rgba(96,165,250,0.10)"},
        ]
        cur = day0 + timedelta(days=1)

    # 2) DR：14:00-16:00 随机 1~2 段，各 30~45 分钟
    cur = start
    while cur < end:
        base = cur.replace(hour=14, minute=0, second=0, microsecond=0)
        for _ in range(rng.randint(1,2)):
            off = rng.randint(0, 120)
            dur = rng.randint(30, 45)
            t0 = base + timedelta(minutes=off)
            t1 = t0 + timedelta(minutes=dur)
            bands.append({"t0": _clamp(t0, start, end), "t1": _clamp(t1, start, end),
                          "label":"DR", "kind":"dr", "color":"rgba(239,68,68,0.12)"})
        cur = (cur.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))

    # 3) 作业高强度：QC 白天两段；YC 傍晚一段；其余随机一段
    cur = start
    while cur < end:
        day0 = cur.replace(hour=0, minute=0, second=0, microsecond=0)
        if asset.lower().startswith("qc"):
            b1, e1 = day0.replace(hour=8),  day0.replace(hour=12)
            b2, e2 = day0.replace(hour=13), day0.replace(hour=17)
            bands += [
                {"t0": _clamp(b1, start, end), "t1": _clamp(e1, start, end),
                 "label":"作业↑", "kind":"ops_high", "color":"rgba(34,197,94,0.10)"},
                {"t0": _clamp(b2, start, end), "t1": _clamp(e2, start, end),
                 "label":"作业↑", "kind":"ops_high", "color":"rgba(34,197,94,0.10)"},
            ]
        elif asset.lower().startswith("yc"):
            b, e = day0.replace(hour=17, minute=30), day0.replace(hour=20)
            bands.append({"t0": _clamp(b, start, end), "t1": _clamp(e, start, end),
                          "label":"作业↑", "kind":"ops_high", "color":"rgba(34,197,94,0.10)"})
        else:
            b = day0.replace(hour=rng.randint(9,16), minute=rng.choice([0,15,30,45]))
            e = b + timedelta(minutes=rng.randint(30,60))
            bands.append({"t0": _clamp(b, start, end), "t1": _clamp(e, start, end),
                          "label":"作业↑", "kind":"ops_high", "color":"rgba(34,197,94,0.10)"})
        cur = day0 + timedelta(days=1)

    # 4) 潮汐极值（QC 才显示）：每 ~12h 一次，20 分钟短带
    if asset.lower().startswith("qc"):
        t = start.replace(hour=1, minute=0, second=0, microsecond=0)
        while t < end:
            t0, t1 = t, t + timedelta(minutes=20)
            bands.append({"t0": _clamp(t0, start, end), "t1": _clamp(t1, start, end),
                          "label":"潮汐极值", "kind":"tide", "color":"rgba(148,163,184,0.10)"})
            t += timedelta(hours=12)

    # 5) 策略执行窗口（示意）
    if (end - start) >= timedelta(minutes=60):
        t0 = start + (end - start)/2 - timedelta(minutes=20)
        t1 = t0 + timedelta(minutes=30)
        bands.append({"t0": _clamp(t0, start, end), "t1": _clamp(t1, start, end),
                      "label":"策略执行", "kind":"policy", "color":"rgba(99,102,241,0.12)"})

    bands = [b for b in bands if b["t1"] > b["t0"]]
    bands.sort(key=lambda b: b["t0"])
    return bands
