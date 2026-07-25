from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.services.business_benchmark import (
    load_verified_report as load_business_benchmark,
)
from app.services.mobile_api.benchmark import (
    load_verified_report as load_workflow_benchmark,
)
from app.services.mobile_api.workflow import (
    IdempotencyConflict,
    MobileWorkflowStore,
    utc_now,
)
from app.services.rl_training.trainer import TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(
    os.getenv(
        "PORT_DT_MOBILE_WORKFLOW_DIR",
        str(ROOT / "data/runtime/mobile"),
    )
)
STORE = MobileWorkflowStore(RUNTIME_ROOT)
router = APIRouter(prefix="/api/mobile", tags=["shared-web-mobile-contract"])


def _reports() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return load_business_benchmark(), load_workflow_benchmark()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"shared Web/mobile evidence is unavailable: {exc}",
        ) from exc


def _business_candidate(report: dict[str, Any]) -> dict[str, Any]:
    test = report["test"]
    improvements = test["improvements"]
    uncertainty = test["uncertainty"]
    berth = improvements["berth_utilization_relative_improvement_percent"]
    berth_point_gain = (
        test["coordinated_policy"]["berth_utilization"]
        - test["baseline"]["berth_utilization"]
    ) * 100.0
    waiting = improvements["average_waiting_time_reduction_percent"]
    cost = improvements["scenario_energy_cost_reduction_percent"]
    wait_ci = uncertainty["average_waiting_time_reduction_percent"]
    cost_ci = uncertainty["scenario_energy_cost_reduction_percent"]
    return {
        "id": report["benchmark_id"] + ":coordinated",
        "title": "多设备协同调度 · 固定留出集候选",
        "summary": (
            "由共享后端读取固定数字孪生基准；结果来自2025年8,760个小时留出步的"
            "时序留出测试，不是港口实测或在线生产收益。"
        ),
        "priorityHint": "默认推荐 · 证据完整 · 仍需人工确认",
        "congestionIndex": {
            "low": 0,
            "high": 0,
            "unit": "n/a",
        },
        "conflictRisk": {
            "low": 0,
            "high": 0,
            "unit": "n/a",
        },
        "safetyMargin": {
            "low": 0,
            "high": 0,
            "unit": "n/a",
        },
        "rewardDelta": {
            "low": round(cost_ci["ci_low"], 2),
            "high": round(cost_ci["ci_high"], 2),
            "unit": "%",
            "prefix": "+",
        },
        "effects": [
            {
                "type": "berth",
                "targetName": "泊位协同",
                "impact": (
                    f"泊位利用率提升 {berth_point_gain:.2f} 个百分点"
                    f"（相对 +{berth:.2f}%）"
                ),
                "severity": "medium",
            },
            {
                "type": "vessel",
                "targetName": "待泊船舶",
                "impact": f"平均等待时长降低 {waiting:.2f}%",
                "severity": "medium",
            },
            {
                "type": "system",
                "targetName": "岸电/储能/柔性负载",
                "impact": f"情景用电成本降低 {cost:.2f}%",
                "severity": "medium",
            },
        ],
        "counterfactuals": [
            {
                "metricName": "泊位利用率",
                "currentRange": f"{test['coordinated_policy']['berth_utilization']:.4f}",
                "baselineRange": f"{test['baseline']['berth_utilization']:.4f}",
                "delta": f"+{berth:.2f}%",
                "direction": "up",
            },
            {
                "metricName": "平均待泊时间",
                "currentRange": f"{test['coordinated_policy']['average_waiting_hours']:.2f}h",
                "baselineRange": f"{test['baseline']['average_waiting_hours']:.2f}h",
                "delta": f"-{waiting:.2f}%",
                "direction": "down",
            },
            {
                "metricName": "情景用电成本",
                "currentRange": f"{test['coordinated_policy']['scenario_energy_cost']:.2f}",
                "baselineRange": f"{test['baseline']['scenario_energy_cost']:.2f}",
                "delta": f"-{cost:.2f}%",
                "direction": "down",
            },
        ],
        "relatedAlerts": [],
        "baselinePolicyId": report["benchmark_id"] + ":baseline",
        "dataset_id": report["dataset"]["dataset_id"],
        "dataset_sha256": report["dataset"]["sha256"],
        "dataset_split": "test",
        "test_rows": test["rows"],
        "evidence_level": report["evidence_level"],
        "production_dispatch": False,
    }


def _registered_model_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    registry = TRAINING_MANAGER.model_registry().list()
    for record in registry.get("models", []):
        evaluation = record.get("evaluation") or {}
        metrics = evaluation.get("metrics") or {}
        if evaluation.get("available") is not True:
            continue
        violation_rate = metrics.get("guardrail_violation_rate")
        if violation_rate is None:
            continue
        algorithm = str(record.get("algorithm") or "").upper()
        reward = metrics.get("reward")
        candidates.append(
            {
                "id": str(record.get("job_id")),
                "title": f"{algorithm} · 已登记留出测试模型",
                "summary": "指标来自共享后端模型登记与独立测试集评测；未生产下发。",
                "priorityHint": "已登记模型 · 需人工复核",
                "congestionIndex": {"low": 0, "high": 0, "unit": "%"},
                "conflictRisk": {
                    "low": round(float(violation_rate) * 100, 3),
                    "high": round(float(violation_rate) * 100, 3),
                    "unit": "%",
                },
                "safetyMargin": {
                    "low": round((1 - float(violation_rate)) * 100, 3),
                    "high": round((1 - float(violation_rate)) * 100, 3),
                    "unit": "%",
                },
                "rewardDelta": {
                    "low": float(reward or 0),
                    "high": float(reward or 0),
                    "unit": " reward",
                    "prefix": "+" if float(reward or 0) >= 0 else "",
                },
                "effects": [
                    {
                        "type": "system",
                        "targetName": "chronological test split",
                        "impact": f"guardrail violation rate={float(violation_rate):.4f}",
                        "severity": "medium",
                    }
                ],
                "counterfactuals": [],
                "relatedAlerts": [],
                "baselinePolicyId": None,
                "dataset_id": record.get("dataset_id"),
                "dataset_sha256": record.get("dataset_sha256"),
                "dataset_split": "test",
                "evidence_level": "registered_heldout_model",
                "production_dispatch": False,
            }
        )
    return candidates


def _system_alerts() -> list[dict[str, Any]]:
    audit = STORE.verify()
    alerts: list[dict[str, Any]] = []
    if audit["valid"] is not True:
        alerts.append(
            {
                "id": "mobile-audit-chain-invalid",
                "title": "跨端审计链校验失败",
                "detail": "共享后端已阻止将当前审计记录作为有效执行证据。",
                "severity": "critical",
                "createdAt": utc_now(),
                "source": "shared_backend_audit_verifier",
            }
        )
    if os.getenv("PORT_DT_LIVE_DATA_VERIFIED", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        alerts.append(
            {
                "id": "mobile-public-evidence-boundary",
                "title": "当前为公开数据离线推演",
                "detail": "未配置已验证的港口实时数据适配器；页面数据不得表述为生产态势。",
                "severity": "info",
                "createdAt": utc_now(),
                "source": "system_provenance",
            }
        )
    return alerts


@router.get("/status")
async def mobile_status() -> JSONResponse:
    business, workflow = _reports()
    return JSONResponse(
        {
            "schema_version": "port_dt_shared_mobile_api_v1",
            "system_id": "digital_twin_ai_port_dual_frontend_v1",
            "backend_id": "port-dt-multi",
            "frontends": ["web", "flutter_mobile"],
            "shared_backend_verified": True,
            "algorithms": ["sac", "ppo", "td3", "dqn", "a2c", "tqc", "mpc"],
            "business_benchmark": {
                "benchmark_id": business["benchmark_id"],
                "dataset_id": business["dataset"]["dataset_id"],
                "dataset_sha256": business["dataset"]["sha256"],
                "test_rows": business["test"]["rows"],
                "test_period": business["dataset"]["test_period"],
                "claims_percent": business[
                    "resume_claims_rounded_percent"
                ],
                "berth_utilization_point_gain": round(
                    (
                        business["test"]["coordinated_policy"][
                            "berth_utilization"
                        ]
                        - business["test"]["baseline"]["berth_utilization"]
                    )
                    * 100.0,
                    3,
                ),
                "measured_port_kpi": False,
            },
            "mobile_workflow_benchmark": {
                "benchmark_id": workflow["benchmark_id"],
                "operations": workflow["operations"]["total"],
                "results": workflow["results"],
            },
            "execution_boundary": {
                "mobile_production_dispatch": False,
                "two_person_actuator_gate": "/api/actuators/capabilities",
                "public_replay_is_dry_run": True,
            },
            "audit": STORE.verify(),
            "updated_at": utc_now(),
        }
    )


@router.get("/situation")
async def mobile_situation() -> JSONResponse:
    business, _workflow = _reports()
    daily = business["test"]["daily_paired_metrics"]
    trend = [
        round(
            float(item["average_waiting_time_reduction_percent"]),
            3,
        )
        for item in daily
        if int(item["rows"]) == 24
    ][-12:]
    audit = STORE.verify()
    claims = business["resume_claims_rounded_percent"]
    return JSONResponse(
        {
            "stabilityLevel": "stable" if audit["valid"] else "critical",
            "systemScore": 100 if audit["valid"] else 0,
            "strategyPressure": 0,
            "constraintHeadroom": 100,
            "riskIntervalLow": 0,
            "riskIntervalHigh": 0,
            "trendPoints": trend,
            "summaryText": (
                "共享后端证据链有效；曲线为固定留出集逐日待泊时间改善率，"
                "不是实时港口态势。"
                if audit["valid"]
                else "共享后端审计链异常，所有执行证据失效关闭。"
            ),
            "refreshAt": utc_now(),
            "dataSource": "public_replay",
            "live_data_verified": False,
            "trend_metric": "daily_waiting_time_reduction_percent",
            "business_benchmark_id": business["benchmark_id"],
            "businessBenchmark": {
                "datasetId": business["dataset"]["dataset_id"],
                "testRows": business["test"]["rows"],
                "berthImprovementPercent": claims[
                    "berth_utilization_relative_improvement_percent"
                ],
                "waitReductionPercent": claims[
                    "average_waiting_time_reduction_percent"
                ],
                "costReductionPercent": claims[
                    "scenario_energy_cost_reduction_percent"
                ],
                "measuredPortKpi": False,
            },
        }
    )


@router.get("/alerts")
async def mobile_alerts() -> JSONResponse:
    items = _system_alerts()
    return JSONResponse(
        {
            "items": items,
            "source": "shared_backend_system_evidence",
            "live_data_verified": False,
            "count": len(items),
        }
    )


@router.websocket("/alerts/ws")
async def mobile_alerts_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        for item in _system_alerts():
            await websocket.send_json(item)
        while True:
            await asyncio.sleep(30)
            await websocket.send_json(
                {
                    "id": "shared-backend-heartbeat",
                    "title": "共享后端心跳",
                    "detail": "Web/移动端共享服务在线；此心跳不是生产港口告警。",
                    "severity": "info",
                    "createdAt": utc_now(),
                    "source": "service_health",
                }
            )
    except WebSocketDisconnect:
        return


@router.get("/strategy/candidates")
async def mobile_strategy_candidates() -> JSONResponse:
    business, _workflow = _reports()
    items = [_business_candidate(business), *_registered_model_candidates()]
    return JSONResponse(
        {
            "items": items,
            "count": len(items),
            "source": "shared_backend_verified_evidence",
            "generated_values": False,
            "production_dispatch": False,
        }
    )


@router.post("/strategy/decisions")
async def mobile_strategy_decision(
    payload: dict[str, Any] = Body(...),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> JSONResponse:
    if not idempotency_key:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key header is required",
        )
    try:
        receipt, replayed = STORE.record_decision(payload, idempotency_key)
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    status_code = 409 if receipt["accepted"] is False else 202
    return JSONResponse(
        {**receipt, "idempotent_replay": replayed},
        status_code=status_code,
    )


@router.get("/strategy/decisions/{request_id}")
async def mobile_strategy_receipt(request_id: str) -> JSONResponse:
    try:
        return JSONResponse(STORE.get_receipt(request_id))
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="unknown mobile strategy decision",
        ) from exc


@router.post("/strategy/replan")
async def mobile_replan_review(
    payload: dict[str, Any] = Body(...),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> JSONResponse:
    business, _workflow = _reports()
    key = idempotency_key or (
        "mobile-replan-"
        + str(payload.get("source_alert_id") or "manual")
        + "-"
        + str(payload.get("trigger") or "replan")
    )
    decision_payload = {
        **payload,
        "target_policy_id": business["benchmark_id"] + ":coordinated",
        "requested_by": str(payload.get("requested_by") or "mobile_operator"),
        "production_dispatch": False,
        "decision_type": "replan_review_request",
    }
    try:
        receipt, replayed = STORE.record_decision(decision_payload, key)
    except (IdempotencyConflict, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(
        {
            "accepted": True,
            "request_id": receipt["request_id"],
            "status": "review_required",
            "candidate_policy_id": receipt["policy_id"],
            "production_dispatch": False,
            "idempotent_replay": replayed,
            "message": "replan review request recorded; no command was sent",
        },
        status_code=202,
    )


@router.post("/audit/events")
async def mobile_audit_event(
    payload: dict[str, Any] = Body(...),
) -> JSONResponse:
    try:
        event = STORE.append_client_audit(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(
        {
            "accepted": True,
            "event_id": event["event_id"],
            "server_time": event["at"],
            "event_hash": event["event_hash"],
            "chain_valid": STORE.verify()["valid"],
        },
        status_code=201,
    )


@router.get("/audit/verify")
async def mobile_audit_verify() -> JSONResponse:
    return JSONResponse(STORE.verify())


@router.get("/business-benchmark")
async def mobile_business_benchmark() -> JSONResponse:
    business, workflow = _reports()
    return JSONResponse(
        {
            "business": business,
            "mobile_workflow": workflow,
            "claim_boundary": (
                "business improvements are system-level digital-twin test "
                "results; workflow rates are deterministic local integration "
                "results, not field SLAs"
            ),
        }
    )
