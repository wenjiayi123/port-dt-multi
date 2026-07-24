from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 123)

def simulate_action_markers(asset: str, start: datetime, end: datetime,
                            seed: Optional[int]=None) -> List[Dict[str, Any]]:
    """
    造几类“动作/保护”标记：
    - act_charge / act_discharge：充放电动作（或设定点上/下调）
    - setpoint：目标设定（温度/照度/阈值等）
    - guard：策略被 Guard 拦截/削减
    规则：每 20~40 分钟随机 1 个点；QC/储能偏充放电，HVAC 偏 setpoint。
    """
    rng = _rng(seed)
    cursor = start.replace(second=0, microsecond=0)
    items: List[Dict[str,Any]] = []

    weights = {"act_charge":0.30, "act_discharge":0.30, "setpoint":0.25, "guard":0.15}
    if asset.lower().startswith(("bess","shore","qc","port","g_","f_")):
        weights.update(act_charge=0.35, act_discharge=0.35, guard=0.2, setpoint=0.1)
    if asset.lower().startswith(("hvac","plant","cool")):
        weights.update(setpoint=0.5, act_charge=0.2, act_discharge=0.2, guard=0.1)

    kinds = list(weights.keys())
    def pick_kind():
        r = rng.random(); acc = 0.0
        for k in kinds:
            acc += weights[k]
            if r <= acc: return k
        return kinds[-1]

    while cursor < end:
        step = rng.randint(20, 40)  # 分钟
        cursor += timedelta(minutes=step)
        if cursor >= end: break
        k = pick_kind()
        lbl = {"act_charge":"充电/上调","act_discharge":"放电/下调",
               "setpoint":"设定点","guard":"Guard"}[k]
        col = {"act_charge":"#60a5fa","act_discharge":"#f97316",
               "setpoint":"#a78bfa","guard":"#f87171"}[k]
        meta: Dict[str,Any] = {}
        if k in ("act_charge","act_discharge"):
            meta["power_kw"] = round(rng.uniform(20,120),1)
        if k == "setpoint":
            meta["target"] = round(rng.uniform(0.2,0.8),2)
        if k == "guard":
            meta["reason"] = rng.choice(["越上限","越下限","速率限制","SLA 保护"])
        items.append({"ts": cursor, "kind": k, "label": lbl, "color": col, "meta": meta})

    return items
