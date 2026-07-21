from __future__ import annotations
from typing import Iterable
import statistics

def _quantile(a, p: float) -> float:
    if not a:
        return 0.0
    a = sorted(a)
    pos = (len(a) - 1) * p
    lo = int(pos)
    hi = min(len(a) - 1, lo + 1)
    if lo == hi:
        return float(a[lo])
    h = pos - lo
    return float(a[lo] * (1 - h) + a[hi] * h)

def cvar95(delta_kwh: Iterable[float]) -> float:
    """对 delta_kwh（正值=更差/更耗电）取损失正尾的 CVaR@95。"""
    losses = sorted([float(x) for x in delta_kwh if x > 0.0])
    if not losses:
        return 0.0
    # Tail: worst 5%
    k = max(1, int(len(losses) * 0.05))
    tail = losses[-k:]
    return float(sum(tail) / len(tail))

def mape_from_deltas(delta_kwh: Iterable[float], ref_kwh: float) -> float:
    """将绝对偏差相对 job_kwh 作为 MAPE 的近似（演示用）。"""
    if ref_kwh <= 0:
        return 0.0
    arr = [abs(float(x)) for x in delta_kwh]
    if not arr:
        return 0.0
    return statistics.mean(arr) / float(ref_kwh)
