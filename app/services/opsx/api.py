"""OpsX engineering-simulator routes.

The simulator remains useful for integration development, but it is disabled by
default so an open-source deployment cannot mistake generated telemetry for a
live rollout control plane. Set ``PORT_DT_ENABLE_ENGINEERING_SIMULATORS=1`` to
opt in deliberately.
"""
import os
from datetime import datetime, timedelta
from typing import Any, Dict
from fastapi import APIRouter, Body, HTTPException, Query

router = APIRouter()


def _require_simulator_opt_in() -> None:
    enabled = os.getenv("PORT_DT_ENABLE_ENGINEERING_SIMULATORS", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        raise HTTPException(
            status_code=503,
            detail="OpsX production backend is not configured; engineering simulator is disabled by default.",
        )


def _simulation(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {**payload, "_provenance": {"mode": "engineering_simulation", "production": False}}

# ---------- 工具：安全导入 ----------
def _try_import(path: str, name: str):
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None

# ========== 1) Rollout ==========
@router.get("/rollout/status")
def rollout_status() -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.rollout_control.simulator", "get_status")
    if fn: return _simulation(fn())
    # 兜底
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")

@router.post("/rollout/traffic")
def rollout_traffic(pct: float = Body(..., embed=True)) -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.rollout_control.simulator", "set_traffic")
    if fn:
        fn(pct)
        return _simulation({"ok": True, "traffic_pct": pct})
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")

@router.post("/rollout/rollback")
def rollout_rollback() -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.rollout_control.simulator", "do_rollback")
    if fn: fn()
    return _simulation({"ok": True, "rolled_back_at": datetime.utcnow().isoformat()})

# ========== 2) 质量门槛 ==========
@router.get("/gates")
def gates() -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.quality_gate.simulator", "get_gates")
    if fn:
        return _simulation(fn())
    # 兜底（找不到模拟器时）
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")

# ========== 3) 时间线 ==========
@router.get("/timeline")
def timeline(horizon_min: int = Query(60, ge=1, le=240)) -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.timeline.simulator", "get_timeline")
    if fn:
        return _simulation(fn(horizon_min=horizon_min))
    # 兜底（找不到模块时）
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")


# ========== 4) 策略画像 ==========
@router.get("/profile")
def profile() -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.profile_card.simulator", "get_profile")
    if fn:
        return _simulation(fn())
    # 兜底
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")


# ========== 5) 多目标权衡 ==========
@router.post("/objective/try")
def objective_try(weights: Dict[str, float] = Body(...)) -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.objective_weight.simulator", "try_objective")
    if fn:
        return _simulation(fn(weights))
    # 兜底（万一模块未创建）
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")


# ========== 6) 守护栏预演 ==========
@router.post("/guard/dryrun")
def guard_dryrun(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.guardrails_dryrun.simulator", "dryrun")
    sid = str((payload or {}).get("strategy_id", "demo"))
    if fn:
        return _simulation(fn(sid))
    # 兜底（未找到模块时）
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")


# ========== 7) 运维健康 ==========
@router.get("/health")
def health() -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.ops_health.simulator", "get_health")
    if fn:
        return _simulation(fn())
    # 兜底（未创建模块时）
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")

# ========== 8) 审计导出 ==========
@router.post("/audit/make")
def audit_make() -> Dict[str, Any]:
    _require_simulator_opt_in()
    fn = _try_import("app.services.opsx.audit_export.simulator", "make_report")
    if fn:
        return _simulation(fn())
    # 兜底
    raise HTTPException(status_code=503, detail="OpsX simulator module is unavailable")
