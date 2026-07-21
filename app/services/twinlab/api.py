from __future__ import annotations
from typing import Any, Dict
from fastapi import APIRouter, Body, Query
from .service import TwinLabService

router = APIRouter()
_svc = TwinLabService()

@router.get("/scenarios", summary="场景工厂：列表与趋势")
def scenarios() -> Dict[str, Any]:
    return _svc.scenarios()

@router.post("/scenarios/run", summary="场景批量回放")
def scenarios_run() -> Dict[str, Any]:
    return _svc.scenarios_run()

@router.get("/report", summary="生成 TwinLab 报告（演示）")
def report() -> Dict[str, Any]:
    return _svc.report()

@router.get("/drills", summary="韧性演练：计划/结果/SLA")
def drills() -> Dict[str, Any]:
    return _svc.drills()

@router.post("/drills/trigger", summary="触发演练态（shadow/freeze/rollback）")
def drills_trigger(payload: Dict[str, Any] = Body(default={"state":"shadow"})) -> Dict[str, Any]:
    return _svc.drills_trigger(str(payload.get("state","shadow")))

@router.get("/contracts", summary="数据契约：特征健康度")
def contracts() -> Dict[str, Any]:
    return _svc.contracts()

@router.post("/contracts/verify", summary="数据契约：校验")
def contracts_verify() -> Dict[str, Any]:
    return _svc.contracts_verify()

@router.get("/drills/trigger", summary="触发演练态（兼容GET）")
def drills_trigger_get(state: str = Query("shadow")) -> Dict[str, Any]:
    return _svc.drills_trigger(state)
