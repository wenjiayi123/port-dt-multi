# app/services/mas_orchestrator/domain.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---- 基础指标 ----
@dataclass
class KPI:
    throughput_teu: float
    wait_p95_h: float
    peak_kw: float
    energy_kwh: float
    carbon_kg: float

# ---- 设备/体 ----
@dataclass
class AgentQC:
    id: str
    status: str
    job: Optional[str] = None

@dataclass
class AgentYC:
    id: str
    status: str

@dataclass
class AgentAGV:
    id: str
    status: str  # enroute / charging / idle ...

@dataclass
class AgentBESS:
    id: str
    soc: float          # 0~1
    mode: str           # charge / discharge / standby

@dataclass
class AgentShore:
    id: str
    power_kw: float     # 当前岸电功率

# ---- 图谱 ----
@dataclass
class GraphNode:
    id: str
    name: str
    category: str       # vessel/qc/yc/agv/bess/shore

@dataclass
class GraphEdge:
    source: str
    target: str
    value: float = 1.0

@dataclass
class Graph:
    nodes: List[Dict[str, Any]]  # 为了兼容前端，保持 dict 形态
    edges: List[Dict[str, Any]]

# ---- 时间线（甘特） ----
@dataclass
class TimelineItem:
    name: str
    category: str
    start: int          # min
    end: int            # min

@dataclass
class Timeline:
    categories: List[str]
    items: List[Dict[str, Any]]

# ---- 冲突 ----
@dataclass
class Conflict:
    type: str           # resource/power/...
    detail: str
    severity: str       # info/warn/bad
    proposal: Optional[str] = None

# ---- API 载体（可选） ----
@dataclass
class OverviewResponse:
    ts: str
    kpis: Dict[str, Any]
    agents: Dict[str, Any]
    graph: Dict[str, Any]
    timeline: Dict[str, Any]
    conflicts: List[Dict[str, Any]]

@dataclass
class ProposeRequest:
    horizon_min: int = 120

@dataclass
class ProposeResponse:
    plan_id: str
    horizon_min: int
    actions: List[Dict[str, Any]]

@dataclass
class SimulateRequest:
    scenario: str = "dense_berthing"

@dataclass
class SimulateResponse:
    status: str
    scenario: str

@dataclass
class DispatchResponse:
    status: str
    job_id: str
