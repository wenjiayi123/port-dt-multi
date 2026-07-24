from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 场景工厂
@dataclass
class ScenarioItem:
    name: str
    tags: List[str]
    pass_rate: float
    last_run: Optional[str] = None
    cases: Optional[int] = None

@dataclass
class ScenarioSeries:
    ts: List[str]
    rate: List[float]

@dataclass
class ScenariosPayload:
    items: List[Dict[str, Any]]
    pass_series: Dict[str, Any]
    duration_hist: List[int]

# 韧性演练
@dataclass
class DrillItem:
    name: str
    window: str
    result: str
    duration_min: int

@dataclass
class DrillsPayload:
    state: str
    items: List[Dict[str, Any]]
    sla_series: Dict[str, List[int]]

# 数据契约
@dataclass
class ContractItem:
    feature: str
    source: str
    freshness_min: int
    null_rate: float
    schema_ok: bool
    status: str
