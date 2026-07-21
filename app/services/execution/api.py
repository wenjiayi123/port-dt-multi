from __future__ import annotations

import asyncio
import os
import re
from dataclasses import asdict
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from app.adapters.actuators import Command, PortSouthboundGateway
from app.services.rl_training.trainer import TRAINING_MANAGER


router = APIRouter(prefix="/api/actuators", tags=["site-actuators"])
gateway = PortSouthboundGateway()
SAFE_PARAMETER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
CONTROL_FIELDS = {"bess_kw", "service_factor", "flexible_load_command"}


def _public_result(result: Any) -> Dict[str, Any]:
    payload = asdict(result)
    evidence_path = payload.pop("evidence_path", None)
    details = dict(payload.get("details") or {})
    detail_path = details.pop("evidence_path", None)
    payload["details"] = details
    payload["evidence_id"] = os.path.basename(str(evidence_path or detail_path)) if (evidence_path or detail_path) else None
    return payload


def _result(result: Any) -> JSONResponse:
    payload = _public_result(result)
    status = 202 if result.status == "PENDING" else (200 if result.status in {"EXECUTED", "ROLLEDBACK"} else 409)
    return JSONResponse(payload, status_code=status)


@router.get("/capabilities")
async def actuator_capabilities() -> JSONResponse:
    data = gateway.cfg.data
    routes = data.get("routing") or {}
    channels = sorted({
        str(route.get("channel") or "unavailable")
        for group in (routes.get("asset") or {}, routes.get("type") or {})
        for route in group.values()
        if isinstance(route, dict)
    })
    token_env = str((data.get("security") or {}).get("confirmation_token_env") or "PORT_DT_SECOND_CHANNEL_TOKEN")
    return JSONResponse({
        "enabled": data.get("enabled") is True,
        "mode": "site_configured" if data.get("enabled") is True else "fail_closed",
        "reason": data.get("reason") if data.get("enabled") is not True else None,
        "config_source": "PORT_DT_ACTUATOR_CONFIG" if os.getenv("PORT_DT_ACTUATOR_CONFIG") else "unconfigured_default",
        "whitelisted_asset_count": len(data.get("whitelist") or {}),
        "configured_channels": channels,
        "site_constraint_names": sorted((data.get("constraints") or {}).keys()),
        "second_channel_secret_configured": len(os.getenv(token_env, "")) >= 32,
        "two_person_confirmation_required": True,
        "requester_confirmer_must_differ": True,
        "audit_evidence": "atomic_json_mode_0600",
        "rollback_endpoint_available": True,
        "policy": "every command is staged for a different human confirmer; no one-request execution",
    })


@router.post("/stage")
async def stage_manual_command(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    parameters = payload.get("parameters")
    if not all(str(payload.get(name) or "").strip() for name in ("asset_id", "asset_type", "action", "requested_by")):
        raise HTTPException(status_code=422, detail="asset_id, asset_type, action and requested_by are required")
    if not isinstance(parameters, dict) or not parameters:
        raise HTTPException(status_code=422, detail="parameters must be a non-empty object")
    command = Command(
        asset_id=str(payload["asset_id"]),
        asset_type=str(payload["asset_type"]),
        action=str(payload["action"]),
        parameters=dict(parameters),
        priority=int(payload.get("priority") or 5),
        requested_by=str(payload["requested_by"]),
        idempotency_key=str(payload.get("idempotency_key") or "") or None,
        two_channel_required=True,
        constraints_check={"source": "manual_staged_command", "independent_site_review_required": True},
    )
    return _result(await asyncio.to_thread(gateway.dispatch, command))


@router.post("/rl-stage")
async def stage_rl_recommendation(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    job_id = str(payload.get("job_id") or "").strip()
    state = payload.get("state")
    control_field = str(payload.get("control_field") or "").strip()
    parameter_name = str(payload.get("setpoint_parameter") or "").strip()
    if not job_id or not isinstance(state, dict):
        raise HTTPException(status_code=422, detail="job_id and canonical state are required")
    if control_field not in CONTROL_FIELDS or not SAFE_PARAMETER.fullmatch(parameter_name):
        raise HTTPException(status_code=422, detail="control_field or setpoint_parameter is invalid")
    if not all(str(payload.get(name) or "").strip() for name in ("asset_id", "asset_type", "action", "requested_by")):
        raise HTTPException(status_code=422, detail="asset_id, asset_type, action and requested_by are required")
    try:
        prediction = await asyncio.to_thread(TRAINING_MANAGER.predict, job_id, {"state": state})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown model run: {job_id}") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    safety = prediction.get("safety_envelope") or {}
    if safety.get("human_review_eligible") is not True:
        raise HTTPException(status_code=409, detail={"reason": "recommendation failed software safety envelope", "safety_envelope": safety})
    control = prediction["decoded_control"]
    parameters = dict(payload.get("parameters") or {})
    parameters[parameter_name] = control[control_field]
    command = Command(
        asset_id=str(payload["asset_id"]),
        asset_type=str(payload["asset_type"]),
        action=str(payload["action"]),
        parameters=parameters,
        priority=int(payload.get("priority") or 5),
        requested_by=str(payload["requested_by"]),
        idempotency_key=str(payload.get("idempotency_key") or "") or None,
        two_channel_required=True,
        model_version=job_id,
        constraints_check={
            "source": "verified_rl_recommendation",
            "algorithm": prediction.get("algorithm"),
            "dataset_id": prediction.get("dataset_id"),
            "dataset_sha256": prediction.get("dataset_sha256"),
            "control": control,
            "safety_envelope": safety,
        },
    )
    response = await asyncio.to_thread(gateway.dispatch, command)
    payload_result = _public_result(response)
    payload_result["recommendation"] = prediction
    status_code = 202 if response.status == "PENDING" else (200 if response.status in {"EXECUTED", "ROLLEDBACK"} else 409)
    return JSONResponse(payload_result, status_code=status_code)


@router.post("/{command_id}/confirm")
async def confirm_command(command_id: str, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    confirmer = str(payload.get("confirmer") or "").strip()
    token = str(payload.get("channel_token") or "")
    if not confirmer or not token:
        raise HTTPException(status_code=422, detail="confirmer and channel_token are required")
    return _result(await asyncio.to_thread(gateway.confirm, command_id, confirmer, token))


@router.post("/{command_id}/rollback")
async def rollback_command(command_id: str, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    approved_by = str(payload.get("approved_by") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    token = str(payload.get("channel_token") or "")
    if not approved_by or not reason or not token:
        raise HTTPException(status_code=422, detail="approved_by, reason and channel_token are required")
    return _result(await asyncio.to_thread(gateway.rollback, command_id, reason, approved_by, token))
