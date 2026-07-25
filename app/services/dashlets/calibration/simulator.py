from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import random, math

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 20251021)

def simulate_calibration(asset: str, start: datetime, end: datetime,
                         seed: Optional[int]=None) -> Dict[str, Any]:
    """
    造一段“预测 vs 真实”序列并计算四个指标：
    - mape: 平均绝对百分比误差（0~1）
    - cover: 覆盖率（真实是否落在 P10~P90 之间的比例）
    - bias: P50 - 真实 的均值（kW）
    - sigma: 残差标准差（kW）
    备注：完全独立的模拟器，将来替换为真实源时保持字段名即可。
    """
    rng = _rng(seed)
    # Up to 360 minute-resolution points match the six-hour UI window.
    n = max(60, min(360, int((end - start).total_seconds() // 60)))
    base = rng.uniform(40, 90)                          # 起始功率
    slope = rng.uniform(0.05, 0.25)                     # 斜率 kW/min
    band  = rng.uniform(6, 14)                          # 预测带半幅（≈ P90-P10 的一半）
    noise = rng.uniform(1.5, 4.5)                       # 真实噪声幅度

    p50, p10, p90, y = [], [], [], []
    for i in range(n):
        v = base + slope * i
        p50.append(v)
        p10.append(v - band)
        p90.append(v + band)
        # 真实值：在 p50 周围抖动，并注入轻微偏置（不同设备给不同 sign）
        sign = -1.0 if asset.lower().startswith(("qc", "bess", "shore")) else 1.0
        y.append(v + sign * rng.uniform(0.2, 1.2) + rng.gauss(0, noise))

    # 计算指标
    abs_perc = []
    resid = []
    inside = 0
    for i in range(n):
        yi, fi = y[i], p50[i]
        if abs(yi) > 1e-6:
            abs_perc.append(abs(yi - fi) / abs(yi))
        resid.append(fi - yi)
        if p10[i] <= yi <= p90[i]:
            inside += 1

    mape  = sum(abs_perc) / len(abs_perc) if abs_perc else 0.0
    cover = inside / n
    bias  = sum(resid) / len(resid)
    mean  = sum(resid) / len(resid)
    var   = sum((r - mean)**2 for r in resid) / max(1, len(resid)-1)
    sigma = math.sqrt(max(0.0, var))

    # 稍作裁剪，避免极端值
    mape  = max(0.0, min(0.5, mape))
    cover = max(0.0, min(1.0, cover))
    return {"mape": mape, "cover": cover, "bias": bias, "sigma": sigma}
