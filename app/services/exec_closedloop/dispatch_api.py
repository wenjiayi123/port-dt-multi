from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query

from .dispatch import DispatchService

router = APIRouter(prefix="/api/exec", tags=["exec_closedloop"])


class _FallbackTelemetry:
    def list_assets(self) -> List[Dict[str, Any]]:
        return [
            {"id": "QC-01", "type": "yard_crane", "zone": "yard-a"},
            {"id": "YARD-LIGHT-A1", "type": "lighting", "zone": "yard-a"},
            {"id": "SHORE-02", "type": "shore_power", "zone": "berth-2"},
            {"id": "BESS-01", "type": "bess", "zone": "energy-station"},
            {"id": "AGV-CH-01", "type": "agv_charge", "zone": "yard-b"},
        ]


class _FallbackRlPanel:
    def simulate(self, strategy: Dict[str, Any], horizon_min: int = 360, step_min: int = 1) -> Dict[str, Any]:
        actions = strategy.get("actions") or []
        scope = strategy.get("scope") or {}
        asset_ids = list(scope.get("asset_ids") or [])
        scope_size = len(asset_ids) if asset_ids else max(1, len(actions))

        peak_reduction = 0.0
        delta_kwh = 0.0
        delta_carbon = 0.0
        contributors: List[Dict[str, Any]] = []

        for idx, act in enumerate(actions):
            cmd = str((act or {}).get("cmd") or "").strip().lower()
            asset = str((act or {}).get("asset") or asset_ids[idx] if idx < len(asset_ids) else f"ASSET-{idx+1}")
            pct = float((act or {}).get("percent") or 0.0)
            kw_delta = float((act or {}).get("kW_delta") or 0.0)

            if cmd in ("reduce", "lighting_dim", "setpoint", "shore_power"):
                local_peak = max(kw_delta, pct * 120.0, 18.0)
                local_kwh = -max(kw_delta * 0.6, pct * 35.0, 12.0)
                local_carbon = -abs(local_kwh) * 0.42
            elif cmd == "charge":
                local_peak = -8.0
                local_kwh = 15.0
                local_carbon = 6.0
            elif cmd == "discharge":
                local_peak = 22.0
                local_kwh = -18.0
                local_carbon = -7.0
            else:
                local_peak = 3.0
                local_kwh = -4.0
                local_carbon = -1.2

            peak_reduction += local_peak
            delta_kwh += local_kwh
            delta_carbon += local_carbon
            contributors.append(
                {
                    "asset_id": asset,
                    "cmd": cmd or "unknown",
                    "delta_kWh": round(local_kwh, 2),
                    "delta_carbon_kg": round(local_carbon, 2),
                    "peak_reduction_kW": round(local_peak, 2),
                }
            )

        dispatch_ready = bool(actions) and scope_size > 0
        risk_flags: List[str] = []
        if peak_reduction < 1.0:
            risk_flags.append("peak_reduction_low")
        if not dispatch_ready:
            risk_flags.append("dispatch_not_ready")

        return {
            "summary": {
                "delta_kWh": round(delta_kwh, 2),
                "delta_carbon_kg": round(delta_carbon, 2),
                "peak_reduction_kW": round(peak_reduction, 2),
                "window": strategy.get("window") or {},
                "scope_size": scope_size,
                "adjusted_asset_count": max(1, len(actions)) if actions else 0,
                "dispatch_ready": dispatch_ready,
            },
            "feasibility": {
                "ok": dispatch_ready,
                "risk_flags": risk_flags,
            },
            "contributors": contributors,
            "baseline": {
                "total_kWh": round(1500.0 + scope_size * 40.0, 2),
                "peak_kW": round(680.0 + scope_size * 15.0, 2),
            },
            "simulated": {
                "total_kWh": round(1500.0 + scope_size * 40.0 + delta_kwh, 2),
                "peak_kW": round(680.0 + scope_size * 15.0 - peak_reduction, 2),
            },
        }


_dispatch_service: Optional[DispatchService] = None


def _svc() -> DispatchService:
    global _dispatch_service
    if _dispatch_service is None:
        _dispatch_service = DispatchService(
            telemetry=_FallbackTelemetry(),
            rlpanel=_FallbackRlPanel(),
            twin=None,
        )
    return _dispatch_service


@router.get("/health")
async def exec_health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "exec_closedloop.dispatch_api",
        "mode": "fallback-demo",
        "capabilities": ["dispatch", "list", "get", "cancel", "summary"],
    }


@router.post("/dispatch")
async def dispatch_strategy(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    strategy = dict(payload.get("strategy") or payload)
    operator = str(payload.get("operator") or "web-ui")
    dry_run = bool(payload.get("dry_run", True))
    enforce_guardrails = bool(payload.get("enforce_guardrails", True))
    guardrail_min_peak_kw = float(payload.get("guardrail_min_peak_kw", 1.0))
    notes = payload.get("notes")
    return _svc().dispatch(
        strategy=strategy,
        operator=operator,
        dry_run=dry_run,
        enforce_guardrails=enforce_guardrails,
        guardrail_min_peak_kw=guardrail_min_peak_kw,
        notes=notes,
    )


@router.get("/list")
async def list_dispatch_jobs(limit: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    data = _svc().list_history(limit=limit)
    items = data.get("items") or []
    return {
        **data,
        "items": [
            {
                **item,
                "brief": {
                    "status": item.get("status"),
                    "title": item.get("strategy_title") or item.get("strategy_id"),
                    "source": ((item.get("readiness") or {}).get("source_readiness_label") or "unknown"),
                    "dispatch_ready": bool((item.get("readiness") or {}).get("dispatch_ready", False)),
                },
            }
            for item in items
        ],
    }


@router.get("/get/{job_id}")
async def get_dispatch_job(job_id: str) -> Dict[str, Any]:
    items = (_svc().list_history(limit=500).get("items") or [])
    for item in items:
        if item.get("job_id") == job_id:
            return item
    raise HTTPException(status_code=404, detail="job not found")


@router.post("/cancel/{job_id}")
async def cancel_dispatch_job(job_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    operator = str((payload or {}).get("operator") or "web-ui")
    result = _svc().cancel(job_id=job_id, operator=operator)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error") or "cancel failed")
    return result


@router.get("/summary")
async def exec_summary() -> Dict[str, Any]:
    items = (_svc().list_history(limit=50).get("items") or [])
    if not items:
        return {
            "ok": True,
            "recent_job": None,
            "totals": {
                "total": 0,
                "approved_like": 0,
                "rejected": 0,
                "cancelled": 0,
            },
            "homepage_hint": "当前还没有执行工单，首页会显示‘待审批=—，最近工单=—’。",
        }

    latest = items[0]
    approved_like = sum(1 for x in items if x.get("status") == "DRY_RUN_RECORDED")
    rejected = sum(1 for x in items if x.get("status") == "REJECTED")
    cancelled = sum(1 for x in items if x.get("status") == "CANCELLED")

    readiness = latest.get("readiness") or {}
    estimate = latest.get("estimate") or {}

    return {
        "ok": True,
        "recent_job": {
            "job_id": latest.get("job_id"),
            "status": latest.get("status"),
            "strategy_title": latest.get("strategy_title"),
            "created_at": latest.get("created_at"),
            "source_readiness_label": readiness.get("source_readiness_label"),
            "dispatch_ready": readiness.get("dispatch_ready"),
            "peak_reduction_kW": estimate.get("peak_reduction_kW"),
        },
        "totals": {
            "total": len(items),
            "approved_like": approved_like,
            "rejected": rejected,
            "cancelled": cancelled,
        },
        "homepage_hint": (
            f"最近工单状态={latest.get('status')}；"
            f"数据源就绪态={readiness.get('source_readiness_label') or 'unknown'}；"
            f"dispatch_ready={bool(readiness.get('dispatch_ready', False))}。"
        ),
    }
