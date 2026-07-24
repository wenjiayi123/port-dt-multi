from __future__ import annotations
from typing import List, Dict
from pydantic import BaseModel, Field

class PolicyItem(BaseModel):
    id: str
    mape: float = Field(..., description="Mean Absolute Percentage Error vs job_kwh")
    cvar95_kwh: float = Field(..., description="CVaR@95 of positive loss tail (kWh)")
    violations_ppm: int = Field(..., description="Shield violations per million")
    n: int = Field(..., description="Samples")

class Leaderboard(BaseModel):
    sample_total: int
    items: List[PolicyItem]

class SafetyRule(BaseModel):
    rule: str
    ppm: int

class SafetySummary(BaseModel):
    updated_at: str
    cvar95_kwh: float
    guard_pass_rate: float
    violations_ppm: int
    rules: List[SafetyRule]

class HistBin(BaseModel):
    bin: str
    count: int

class ActionsHist(BaseModel):
    hist: List[HistBin]
