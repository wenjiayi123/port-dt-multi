from __future__ import annotations
from datetime import datetime
from typing import Dict, Optional
import random

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 662607)

def _make_job_id(rng: random.Random, now: datetime) -> str:
    # Stable job identifier compatible with external approval systems.
    ts = now.strftime("%Y%m%d-%H%M%S")
    tail = "".join(rng.choice("0123456789ABCDEF") for _ in range(6))
    return f"JOB-{ts}-{tail}"

def simulate_approvals(asset: str, now: datetime, seed: Optional[int]=None) -> Dict[str, object]:
    """
    规则：
    - 换班窗口(7–8/12–13/17–18) ↑
    - DR(14–16) ↑、晚高峰(18–21) ↑
    - 工作日 > 周末
    - 资产差异：QC/YC 更容易产生“调度/功率类”审批；BESS 在 DR/峰价段更集中；HVAC 午后设定点修改多
    """
    rng = _rng(seed)
    h = now.hour + now.minute/60.0
    weekday = now.weekday()  # 0..6
    week_mul = 1.0 if weekday < 5 else 0.6

    asset_l = asset.lower()
    is_qc   = asset_l.startswith(("qc","g_","port_g"))
    is_yc   = asset_l.startswith(("yc","f_","port_f"))
    is_bess = asset_l.startswith(("bess","shore"))
    is_hvac = asset_l.startswith(("hvac","plant","cool"))

    # 基线待办（0~2）
    base = rng.randint(0, 2)

    # 换班窗口
    if int(h) in (7,8,12,13,17,18):
        base += rng.randint(1, 2)

    # DR/峰价
    if 14 <= h < 16:
        base += rng.randint(1, 2)
    if 18 <= h < 21:
        base += rng.randint(1, 2)

    # 资产差异
    if is_qc or is_yc:
        base += 1
    if is_bess and (14 <= h < 22):
        base += 1
    if is_hvac and (12 <= h < 18):
        base += 1

    # 工作日加权
    pending = int(round(base * week_mul))
    # 限幅，避免太夸张
    pending = max(0, min(12, pending))

    last_job = _make_job_id(rng, now) if pending > 0 else None
    return {"pending": pending, "last_job": last_job}
