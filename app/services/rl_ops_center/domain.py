# app/services/rl_ops_center/domain.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------- OPE ----------
@dataclass
class OPEPolicyRow:
    policy: str
    mape: float
    cvar95: float
    viol_ppm: float
    n: int

@dataclass
class OPEOverview:
    ts: str
    summary: Dict[str, Any]
    leaderboard: List[OPEPolicyRow]

# ---------- 守护栏 ----------
@dataclass
class GuardRule:
    level: str          # hard/soft
    rule: str           # 规则表达式
    reason: Optional[str] = None

@dataclass
class GuardVerifyResult:
    ok: bool
    strategy_id: str
    violations: List[Dict[str, Any]]

# ---------- 可观测性 ----------
@dataclass
class Signals:
    metrics: Dict[str, float]    # reward_drift / action_entropy / policy_div / coverage_gap
    thresholds: Dict[str, float] # *_max / *_min

# ---------- 实验 ----------
@dataclass
class Experiment:
    id: str
    phase: str          # stable / canary / paused ...
    traffic: float      # 0~1
    mape: Optional[float] = None
    guard: Optional[float] = None
    sla: Optional[float] = None
    updated_at: Optional[str] = None

# ---------- 因果 ----------
@dataclass
class CausalATE:
    metric: str
    ate: float
    ci: Tuple[float, float]
    method: str
    sample: int

@dataclass
class CausalCATE:
    seg: str
    eff: float
    ci: Tuple[float, float]

@dataclass
class CausalResult:
    metric: str
    ate: float
    ci: Tuple[float, float]
    method: str
    sample: int
    cate: List[CausalCATE]
