# app/services/mas_orchestrator/api.py
from __future__ import annotations
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from .service import OrchestratorService

router = APIRouter()
_svc = OrchestratorService()

@router.get("/overview", summary="协同编排 - 概览（KPI/Agents/Graph/Timeline/Conflicts）")
def overview() -> Dict[str, Any]:
    """
    返回前端所需的全量概览数据。
    若 data/*.json 缺失/为空，会自动使用 service 内置样本。
    """
    return _svc.get_overview()

@router.post("/propose", summary="生成编排建议（演示版）")
def propose(payload: Dict[str, Any] = Body(default={"horizon_min": 120})) -> Dict[str, Any]:
    """
    payload: { "horizon_min": number }  # 预测/计划时域（分钟）
    """
    try:
        horizon = int(payload.get("horizon_min", 120))
        if horizon <= 0 or horizon > 24 * 60:
            raise ValueError("horizon_min out of range")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid horizon_min")
    return _svc.propose(horizon)

@router.post("/simulate", summary="场景仿真（占位/连通性测试）")
def simulate(payload: Dict[str, Any] = Body(default={"scenario": "dense_berthing"})) -> Dict[str, Any]:
    """
    payload: { "scenario": string }
    """
    scenario = str(payload.get("scenario", "dense_berthing")) or "dense_berthing"
    return _svc.simulate(scenario)

@router.post("/dispatch", summary="下发计划（演示版）")
def dispatch() -> Dict[str, Any]:
    """
    演示态返回 ok，后续可接入执行器/工控总线。
    """
    return _svc.dispatch()

# 便于存活探针/联调
@router.get("/ping", summary="健康检查")
def ping() -> Dict[str, str]:
    return {"pong": "mas"}
