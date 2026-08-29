# app/services/rl_ops_center/api.py
from __future__ import annotations
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query
from .service import RLOpsService

router = APIRouter()
_svc = RLOpsService()

# 便于就绪探针
@router.get("/ping")
def ping() -> Dict[str, str]:
    return {"pong": "rlops"}

# --- OPE ---
@router.get("/overview", summary="OPE 概览（可与原 Platform 对齐）")
def overview() -> Dict[str, Any]:
    return _svc.overview()

@router.post("/ope/eval", summary="OPE 评测能力检查")
def ope_eval(payload: Dict[str, Any] = Body(default={"metric": "delta_kWh"})) -> Dict[str, Any]:
    return _svc.ope_eval(payload)

# --- 守护栏 ---
@router.get("/policies", summary="守护栏规则列表")
def list_policies() -> Dict[str, Any]:
    return _svc.list_policies()

@router.post("/policies/verify", summary="守护栏干跑校验")
def verify_policy(payload: Dict[str, Any] = Body(default={"strategy_id": "demo"})) -> Dict[str, Any]:
    sid = str(payload.get("strategy_id", "demo")).strip()
    if not sid:
        raise HTTPException(status_code=400, detail="strategy_id is required")
    return _svc.verify_policy(sid)

# --- 可观测性 ---
@router.get("/signals", summary="RL 可观测性黄金信号")
def signals(algorithm: str | None = Query(None)) -> Dict[str, Any]:
    normalized = algorithm.strip().lower() if algorithm else None
    return _svc.signals(normalized or None)

# --- 实验 ---
@router.get("/experiments", summary="实验/策略列表")
def experiments() -> Dict[str, Any]:
    return _svc.experiments()

@router.post("/rollback", summary="请求回滚某实验/策略")
def rollback(payload: Dict[str, Any] = Body(default={"id": "rl-canary-B"})) -> Dict[str, Any]:
    id_ = str(payload.get("id", "")).strip()
    if not id_:
        raise HTTPException(status_code=400, detail="id is required")
    return _svc.rollback(id_)

# --- 因果 ---
@router.post("/causal/estimate", summary="因果估计（ATE/CATE）")
def causal_estimate(payload: Dict[str, Any] = Body(default={"metric": "delta_kWh", "segment": None})) -> Dict[str, Any]:
    metric = str(payload.get("metric", "delta_kWh"))
    seg = payload.get("segment")
    if seg is not None:
        seg = str(seg)
    return _svc.causal_estimate(metric, seg)
