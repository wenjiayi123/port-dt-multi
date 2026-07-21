from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Optional
import random, math

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 20251102)

def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def _hour_frac(t: datetime) -> float:
    return t.hour + t.minute/60.0

def _phase_and_traffic(now: datetime) -> (str, float):
    """
    简易阶段推进：
    - 每天 0~8 点 canary（5%）
    - 8~16 点 ramp（20~50%，随小时上升）
    - 16~24 点 stable（70~100%，晚高峰可收缩些）
    """
    h = _hour_frac(now)
    if h < 8:   return "canary", 0.05
    if h < 16:  return "ramp", min(0.5, 0.2 + (h-8)*0.04)
    # 晚高峰适度收缩，避免风险
    base = 0.9 - (0.1 if 18 <= h <= 21 else 0.0)
    return "stable", max(0.7, base)

def simulate_rollout_snapshot(asset: str, now: datetime,
                              seed: Optional[int]=None) -> Dict[str, object]:
    rng = _rng(seed)
    asset_l = asset.lower()
    is_qc   = asset_l.startswith(("qc","g_","port_g"))
    is_yc   = asset_l.startswith(("yc","f_","port_f"))
    is_bess = asset_l.startswith(("bess","shore"))
    is_hvac = asset_l.startswith(("hvac","plant","cool"))

    # 基线（不同资产的固有难度）
    mape_base  = 0.10  # 能源/功率类 MAPE 基线
    guard_base = 0.010 # 策略被 Guard 拦截比例
    sla_base   = 0.006 # SLA 违约比例
    if is_qc:   mape_base += 0.02; guard_base += 0.005
    if is_yc:   mape_base += 0.015
    if is_bess: guard_base += 0.004
    if is_hvac: mape_base += 0.01; sla_base += 0.004

    h = _hour_frac(now)
    weekday = now.weekday()  # 0..6

    # 时段影响：DR(14-16)、峰价(18-21) 加重；周末更好
    bump = 0.0
    if 14 <= h < 16: bump += 0.03
    if 18 <= h < 21: bump += 0.04
    if weekday >= 5: bump -= 0.01

    # 潮汐/作业节律（QC 明显）
    if is_qc:
        bump += 0.02 * math.exp(-((h-10.5)**2)/(2*2.0**2))
        bump += 0.02 * math.exp(-((h-15.0)**2)/(2*2.0**2))
        if int(h) in (7,8,12,13,17,18): bump += 0.01

    # 阶段与流量
    phase, traffic = _phase_and_traffic(now)

    # 指标当前值（加入随机与轻微随 traffic 的改善/恶化）
    # traffic 越大，数据更稳定（有时也可能更暴露问题，这里设为小幅改善）
    mape  = max(0.0, mape_base  + bump + rng.uniform(-0.015, 0.015) - 0.02*traffic)
    guard = max(0.0, guard_base + 0.5*bump + rng.uniform(-0.004, 0.004) - 0.01*traffic)
    sla   = max(0.0, sla_base   + 0.4*bump + rng.uniform(-0.003, 0.003) - 0.008*traffic)

    # 门槛（策略配置）：能耗MAPE<=0.18，Guard<=2.5%，SLA<=1.5%
    thresholds = {
        "mape_energy_max":         0.18,
        "guard_block_rate_max":    0.025,
        "sla_violation_rate_max":  0.015,
    }

    # 圆整
    out = {
        "phase": phase,
        "traffic_pct": round(_clamp01(traffic), 3),
        "mape": round(mape, 3),
        "guard_block_rate": round(guard, 3),
        "sla_violation_rate": round(sla, 3),
        "thresholds": thresholds,
    }
    return out
