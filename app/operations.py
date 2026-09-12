from __future__ import annotations

import hmac
import hashlib
import json
import copy
import os
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


def is_production() -> bool:
    return os.getenv("PORT_DT_ENV", "development").strip().lower() == "production"


def cors_origins() -> list[str]:
    configured = [item.strip() for item in os.getenv("PORT_DT_CORS_ORIGINS", "").split(",") if item.strip()]
    if configured:
        if is_production() and any(item == "*" or not item.startswith("https://") for item in configured):
            return []
        return configured
    return [] if is_production() else [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    ]


class RuntimeMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.requests: Counter[tuple[str, str, int]] = Counter()
        self.duration: Dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        with self.lock:
            self.requests[(method, route, int(status))] += 1
            values = self.duration[(method, route)]
            values[0] += 1
            values[1] += float(seconds)

    def prometheus(self) -> str:
        def esc(value: str) -> str:
            return value.replace("\\", "\\\\").replace('"', '\\"')

        lines = [
            "# HELP port_dt_uptime_seconds Process uptime in seconds.",
            "# TYPE port_dt_uptime_seconds gauge",
            f"port_dt_uptime_seconds {time.time() - self.started_at:.6f}",
            "# HELP port_dt_http_requests_total HTTP requests by method, route and status.",
            "# TYPE port_dt_http_requests_total counter",
        ]
        with self.lock:
            for (method, route, status), count in sorted(self.requests.items()):
                lines.append(f'port_dt_http_requests_total{{method="{esc(method)}",route="{esc(route)}",status="{status}"}} {count}')
            lines.extend([
                "# HELP port_dt_http_request_duration_seconds_sum Total HTTP request duration.",
                "# TYPE port_dt_http_request_duration_seconds_sum counter",
                "# HELP port_dt_http_request_duration_seconds_count HTTP request duration observations.",
                "# TYPE port_dt_http_request_duration_seconds_count counter",
            ])
            for (method, route), (count, total) in sorted(self.duration.items()):
                labels = f'method="{esc(method)}",route="{esc(route)}"'
                lines.append(f"port_dt_http_request_duration_seconds_sum{{{labels}}} {total:.9f}")
                lines.append(f"port_dt_http_request_duration_seconds_count{{{labels}}} {int(count)}")
        return "\n".join(lines) + "\n"


RUNTIME_METRICS = RuntimeMetrics()


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: Dict[str, deque[float]] = defaultdict(deque)

    def allow(self, identity: str, *, limit: int, now: float | None = None) -> tuple[bool, int]:
        timestamp = time.monotonic() if now is None else float(now)
        cutoff = timestamp - 60.0
        with self.lock:
            bucket = self.events[identity]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            remaining = max(0, int(limit) - len(bucket))
            if len(bucket) >= int(limit):
                return False, 0
            bucket.append(timestamp)
            return True, max(0, remaining - 1)


RATE_LIMITER = SlidingWindowRateLimiter()
READINESS_BASE_CACHE: Dict[str, Any] = {"at": 0.0, "checks": None}
READINESS_BASE_LOCK = threading.Lock()


def rate_limit_per_minute() -> int:
    try:
        value = int(os.getenv("PORT_DT_RATE_LIMIT_RPM", "600"))
    except ValueError:
        value = 600
    return max(1, min(60_000, value))


def max_request_bytes() -> int:
    try:
        value = int(os.getenv("PORT_DT_MAX_REQUEST_BYTES", str(10 * 1024 * 1024)))
    except ValueError:
        value = 10 * 1024 * 1024
    return max(1024, min(1024 * 1024 * 1024, value))


def api_keys() -> list[str]:
    """Return only keys with enough entropy-bearing length for production use."""
    return [item.strip() for item in os.getenv("PORT_DT_API_KEYS", "").split(",") if len(item.strip()) >= 32]


def admin_api_keys() -> list[str]:
    return [item.strip() for item in os.getenv("PORT_DT_ADMIN_API_KEYS", "").split(",") if len(item.strip()) >= 32]


def requires_admin(request: Request) -> bool:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return False
    path = request.url.path.rstrip("/")
    return (
        path == "/api/rl/datasets/upload"
        or path == "/api/rl/metrics/clear"
        or path == "/api/rl/artifacts/upload"
        or path.startswith("/api/rl/models")
        or path.startswith("/api/exec")
        or path.startswith("/api/actuators")
        or "/dispatch" in path
    )


class OperationsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", "").strip()[:128] or str(uuid.uuid4())
        request.state.request_id = request_id
        protected = request.url.path.startswith(("/api", "/external")) or request.url.path == "/metrics"
        if is_production() and protected:
            response: Response | None = None
            matched_index: int | None = None
            keys = api_keys()
            admins = admin_api_keys()
            supplied = request.headers.get("X-API-Key", "")
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > max_request_bytes():
                response = JSONResponse({"detail": "request body exceeds production limit", "request_id": request_id}, status_code=413)
            elif not keys and not admins:
                response = JSONResponse({"detail": "production API authentication is not configured", "request_id": request_id}, status_code=503)
            else:
                configured_keys = [*admins, *keys]
                matched_index = next(
                    (
                        index
                        for index, expected in enumerate(configured_keys)
                        if hmac.compare_digest(supplied, expected)
                    ),
                    None,
                )
            if response is None and matched_index is None:
                response = JSONResponse({"detail": "invalid or missing API key", "request_id": request_id}, status_code=401)
            elif response is None:
                is_admin = any(hmac.compare_digest(supplied, expected) for expected in admins)
                request.state.auth_role = "admin" if is_admin else "operator"
                # The stable rate-limit bucket is the configured key slot, not
                # a digest of credential material. This avoids treating an API
                # key like a password hashed with a fast general-purpose hash.
                identity = f"{'admin' if is_admin else 'operator'}:{matched_index}"
                allowed, remaining = RATE_LIMITER.allow(
                    identity,
                    limit=rate_limit_per_minute(),
                )
                if not allowed:
                    response = JSONResponse({"detail": "production API rate limit exceeded", "request_id": request_id}, status_code=429)
                    response.headers["Retry-After"] = "60"
                elif requires_admin(request) and not is_admin:
                    response = JSONResponse({"detail": "administrator API key required", "request_id": request_id}, status_code=403)
                else:
                    response = await call_next(request)
                response.headers["X-RateLimit-Limit"] = str(rate_limit_per_minute())
                response.headers["X-RateLimit-Remaining"] = str(remaining)
            assert response is not None
        else:
            response = await call_next(request)
        duration = time.perf_counter() - started
        route_object = request.scope.get("route")
        route = getattr(route_object, "path", None) or "__unmatched__"
        RUNTIME_METRICS.observe(request.method, route, response.status_code, duration)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if protected:
            response.headers["Cache-Control"] = "no-store"
        if is_production():
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if is_production() and protected:
            response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
        return response


def _verified_site_json(env_name: str, *, kind: str) -> Dict[str, Any]:
    configured = os.getenv(env_name, "").strip()
    if not configured:
        return {"ok": False, "status": "not_configured", "env": env_name}
    path = Path(configured).expanduser()
    if not path.is_file():
        return {"ok": False, "status": "file_missing", "env": env_name, "artifact_id": path.name}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"ok": False, "status": "invalid_json", "env": env_name, "artifact_id": path.name, "error": "attestation JSON is invalid"}
    if not isinstance(payload, dict):
        return {"ok": False, "status": "invalid_payload", "env": env_name, "artifact_id": path.name}
    common = bool(payload.get("site_id") and payload.get("approved") is True)
    try:
        if kind == "graph":
            kind_ok = bool(
                payload.get("source_mode") == "authorized_site"
                and (payload.get("entities") or payload.get("nodes"))
                and payload.get("approved_by")
            )
        elif kind == "calibration":
            from app.services.twin_schema.service import TwinSchemaService

            calibration_validation = TwinSchemaService.validate_calibration(payload)
            kind_ok = bool(
                payload.get("measured_outcomes") is True
                and payload.get("validation_status") == "pass"
                and int(payload.get("validation_rows") or 0) > 0
                and payload.get("approved_by")
                and calibration_validation.get("production_gate_eligible") is True
            )
        elif kind == "shadow":
            from app.services.site_shadow_acceptance import SiteShadowAcceptanceService

            shadow_validation = SiteShadowAcceptanceService.validate_evidence(payload)
            kind_ok = bool(
                payload.get("measured_incumbent_baseline") is True
                and payload.get("acceptance_status") == "pass"
                and int(payload.get("shadow_cycles") or 0) >= 35
                and int(payload.get("operational_days") or 0) >= 7
                and float(payload.get("guardrail_violation_rate") or 0.0) == 0.0
                and payload.get("approved_by")
                and shadow_validation.get("production_gate_eligible") is True
            )
        else:
            kind_ok = False
    except (TypeError, ValueError):
        kind_ok = False
    return {
        "ok": bool(common and kind_ok),
        "status": "verified" if common and kind_ok else "evidence_incomplete",
        "env": env_name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "site_id": payload.get("site_id"),
        "kind": kind,
    }


def _repository_readiness_checks() -> Dict[str, Dict[str, Any]]:
    from app.services.rl_training.datasets import dataset_quality_report, load_port_dataset
    from app.services.rl_training.trainer import TRAINING_MANAGER

    now = time.monotonic()
    with READINESS_BASE_LOCK:
        cached = READINESS_BASE_CACHE.get("checks")
        if cached is not None and now - float(READINESS_BASE_CACHE.get("at") or 0.0) < 30.0:
            return copy.deepcopy(cached)
    checks: Dict[str, Dict[str, Any]] = {}
    try:
        quality = dataset_quality_report(load_port_dataset("public_port_ops_v1", TRAINING_MANAGER.data_root))
        checks["canonical_dataset"] = {"ok": quality["training_eligible"], "status": quality["status"], "sha256": quality["dataset_sha256"]}
    except Exception:
        checks["canonical_dataset"] = {
            "ok": False,
            "error": "canonical dataset readiness check failed; inspect server logs",
        }
    runtime = TRAINING_MANAGER.capabilities().get("runtime") or {}
    checks["rl_runtime"] = {"ok": bool(runtime.get("available")), **runtime}
    with READINESS_BASE_LOCK:
        READINESS_BASE_CACHE.update(at=now, checks=copy.deepcopy(checks))
    return checks


def readiness_report() -> Dict[str, Any]:
    checks = _repository_readiness_checks()
    origins = cors_origins()
    keys = api_keys()
    admins = admin_api_keys()
    checks["cors"] = {"ok": bool(origins), "origins": origins, "status": "https_allowlist_configured" if is_production() and origins else "development_same_origin_defaults" if origins else "not_configured"}
    checks["api_authentication"] = {"ok": (not is_production()) or bool(keys or admins), "mode": "role_separated_api_key" if is_production() else "development_open"}
    checks["privileged_api_key"] = {"ok": (not is_production()) or bool(admins), "status": "separate_admin_key_configured" if admins else "not_required_in_development", "required_for_research_api": False}
    checks["api_rate_limit"] = {"ok": rate_limit_per_minute() > 0, "status": "per_key_sliding_window", "requests_per_minute_per_key": rate_limit_per_minute()}
    checks["request_body_limit"] = {"ok": max_request_bytes() > 0, "status": "declared_body_limit_enabled", "max_bytes": max_request_bytes(), "basis": "Content-Length rejected before route execution"}
    checks["security_headers"] = {"ok": True, "status": "application_middleware_enabled", "headers": ["X-Request-ID", "X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy", "Permissions-Policy", "Cache-Control", "HSTS", "CSP"]}
    checks["production_mode"] = {"ok": is_production(), "mode": "production" if is_production() else "development"}
    checks["tls_termination"] = {"ok": os.getenv("PORT_DT_TLS_TERMINATION_ATTESTED", "").strip().lower() == "true", "attestation": "operator_configured"}
    checks["secret_manager"] = {"ok": os.getenv("PORT_DT_SECRET_MANAGER_ATTESTED", "").strip().lower() == "true", "attestation": "operator_configured"}
    checks["twin_graph"] = {**_verified_site_json("PORT_DT_TWIN_GRAPH_PATH", kind="graph"), "required_for_research_api": False}
    checks["site_calibration"] = {**_verified_site_json("PORT_DT_TWIN_CALIBRATION_PATH", kind="calibration"), "required_for_research_api": False}
    checks["shadow_acceptance"] = {**_verified_site_json("PORT_DT_SHADOW_ACCEPTANCE_PATH", kind="shadow"), "required_for_research_api": False}
    from app.services.site_execution_acceptance import SiteExecutionAcceptanceService

    execution_readiness = SiteExecutionAcceptanceService().readiness()
    execution_artifacts = execution_readiness["configured_artifacts"]
    checks["site_execution_acceptance"] = {
        "ok": execution_readiness["boundary"]["site_execution_accepted"],
        "status": "verified" if execution_readiness["boundary"]["site_execution_accepted"] else "evidence_incomplete",
        "env": "PORT_DT_EXECUTION_ACCEPTANCE_PATH",
        "artifact_id": execution_artifacts.get("evidence_artifact_id"),
        "sha256": execution_artifacts.get("evidence_sha256"),
        "site_id": execution_artifacts.get("site_id"),
        "kind": "execution",
        "blockers": execution_artifacts.get("blockers") or [],
        "required_for_research_api": False,
    }
    from app.services.port_call_collaboration import PortCallCollaborationService

    collaboration_readiness = PortCallCollaborationService().readiness()
    collaboration_artifact = collaboration_readiness["configured_artifact"]
    checks["port_call_collaboration"] = {
        "ok": collaboration_readiness["boundary"]["site_collaboration_accepted"],
        "status": "verified" if collaboration_readiness["boundary"]["site_collaboration_accepted"] else "evidence_incomplete",
        "env": "PORT_DT_PORT_CALL_COLLABORATION_PATH",
        "artifact_id": collaboration_artifact.get("artifact_id"),
        "sha256": collaboration_artifact.get("sha256"),
        "site_id": collaboration_artifact.get("site_id"),
        "kind": "port_call_collaboration",
        "blockers": collaboration_artifact.get("blockers") or [],
        "required_for_research_api": False,
    }
    from app.services.maritime_interoperability import MaritimeInteroperabilityService

    interoperability_readiness = MaritimeInteroperabilityService().readiness()
    interoperability_artifact = interoperability_readiness["configured_artifact"]
    checks["maritime_interoperability"] = {
        "ok": interoperability_readiness["boundary"]["site_interoperability_accepted"],
        "status": "verified" if interoperability_readiness["boundary"]["site_interoperability_accepted"] else "evidence_incomplete",
        "env": "PORT_DT_MARITIME_INTEROPERABILITY_PATH",
        "artifact_id": interoperability_artifact.get("artifact_id"),
        "sha256": interoperability_artifact.get("sha256"),
        "site_id": interoperability_artifact.get("site_id"),
        "kind": "maritime_interoperability",
        "blockers": interoperability_artifact.get("blockers") or [],
        "required_for_research_api": False,
    }
    from app.services.forecast_uncertainty import ForecastUncertaintyService

    forecast_readiness = ForecastUncertaintyService().readiness()
    forecast_artifact = forecast_readiness["configured_artifact"]
    checks["forecast_uncertainty"] = {
        "ok": forecast_readiness["boundary"]["site_forecast_service_accepted"],
        "status": "verified" if forecast_readiness["boundary"]["site_forecast_service_accepted"] else "evidence_incomplete",
        "env": "PORT_DT_FORECAST_UNCERTAINTY_PATH",
        "artifact_id": forecast_artifact.get("artifact_id"),
        "sha256": forecast_artifact.get("sha256"),
        "site_id": forecast_artifact.get("site_id"),
        "kind": "forecast_uncertainty",
        "blockers": forecast_artifact.get("blockers") or [],
        "required_for_research_api": False,
    }
    from app.services.business_benefit_attribution import BusinessBenefitAttributionService

    benefit_readiness = BusinessBenefitAttributionService().readiness()
    benefit_artifact = benefit_readiness["configured_artifact"]
    checks["business_benefit_attribution"] = {
        "ok": benefit_readiness["boundary"]["realized_business_benefit_verified"],
        "status": "verified" if benefit_readiness["boundary"]["realized_business_benefit_verified"] else "evidence_incomplete",
        "env": "PORT_DT_BUSINESS_BENEFIT_ATTRIBUTION_PATH",
        "artifact_id": benefit_artifact.get("artifact_id"),
        "sha256": benefit_artifact.get("sha256"),
        "site_id": benefit_artifact.get("site_id"),
        "kind": "business_benefit_attribution",
        "blockers": benefit_artifact.get("blockers") or [],
        "required_for_research_api": False,
    }
    from app.services.end_to_end_coordination import EndToEndCoordinationService

    coordination_readiness = EndToEndCoordinationService().readiness()
    coordination_artifact = coordination_readiness["configured_artifact"]
    checks["end_to_end_coordination"] = {
        "ok": coordination_readiness["boundary"]["site_end_to_end_coordination_accepted"],
        "status": "verified" if coordination_readiness["boundary"]["site_end_to_end_coordination_accepted"] else "evidence_incomplete",
        "env": "PORT_DT_END_TO_END_COORDINATION_PATH",
        "artifact_id": coordination_artifact.get("artifact_id"),
        "sha256": coordination_artifact.get("sha256"),
        "site_id": coordination_artifact.get("site_id"),
        "kind": "end_to_end_coordination",
        "blockers": coordination_artifact.get("blockers") or [],
        "required_for_research_api": False,
    }
    from app.services.production_continuity import ProductionContinuityService

    continuity_readiness = ProductionContinuityService().readiness()
    continuity_artifact = continuity_readiness["configured_artifact"]
    checks["production_continuity"] = {
        "ok": continuity_readiness["boundary"]["site_continuity_accepted"],
        "status": "verified" if continuity_readiness["boundary"]["site_continuity_accepted"] else "evidence_incomplete",
        "env": "PORT_DT_PRODUCTION_CONTINUITY_PATH",
        "artifact_id": continuity_artifact.get("artifact_id"),
        "sha256": continuity_artifact.get("sha256"),
        "site_id": continuity_artifact.get("site_id"),
        "kind": "production_continuity",
        "blockers": continuity_artifact.get("blockers") or [],
        "required_for_research_api": False,
    }
    from app.services.operating_model_governance import OperatingModelGovernanceService

    operating_model_readiness = OperatingModelGovernanceService().readiness()
    operating_model_artifact = operating_model_readiness["configured_artifact"]
    checks["operating_model_governance"] = {
        "ok": operating_model_readiness["boundary"]["site_operating_model_accepted"],
        "status": "verified" if operating_model_readiness["boundary"]["site_operating_model_accepted"] else "evidence_incomplete",
        "env": "PORT_DT_OPERATING_MODEL_GOVERNANCE_PATH",
        "artifact_id": operating_model_artifact.get("artifact_id"),
        "sha256": operating_model_artifact.get("sha256"),
        "site_id": operating_model_artifact.get("site_id"),
        "kind": "operating_model_governance",
        "blockers": operating_model_artifact.get("blockers") or [],
        "required_for_research_api": False,
    }
    evidence_site_ids = {
        checks[name].get("site_id")
        for name in (
            "twin_graph", "site_calibration", "shadow_acceptance",
            "site_execution_acceptance", "port_call_collaboration", "maritime_interoperability", "forecast_uncertainty",
            "business_benefit_attribution",
            "end_to_end_coordination",
            "production_continuity", "operating_model_governance",
        )
        if checks[name].get("ok")
    }
    checks["site_evidence_consistency"] = {
        "ok": len(evidence_site_ids) == 1 and all(
            checks[name].get("ok")
            for name in (
                "twin_graph", "site_calibration", "shadow_acceptance",
                "site_execution_acceptance", "port_call_collaboration", "maritime_interoperability", "forecast_uncertainty",
                "business_benefit_attribution",
                "end_to_end_coordination",
                "production_continuity", "operating_model_governance",
            )
        ),
        "site_ids": sorted(str(item) for item in evidence_site_ids),
        "requirement": "all technical, continuity and operating-model evidence must bind to the same site_id",
    }
    open_source_ready = all(checks[name]["ok"] for name in ("canonical_dataset", "rl_runtime"))
    production_ready = open_source_ready and all(
        checks[name]["ok"]
        for name in (
            "production_mode", "cors", "api_authentication", "privileged_api_key",
            "tls_termination", "secret_manager", "twin_graph", "site_calibration",
            "shadow_acceptance", "site_execution_acceptance", "port_call_collaboration", "maritime_interoperability",
            "forecast_uncertainty",
            "business_benefit_attribution",
            "end_to_end_coordination",
            "production_continuity", "operating_model_governance",
            "site_evidence_consistency",
        )
    )
    return {
        "status": "ready" if open_source_ready else "not_ready",
        "open_source_runtime_ready": open_source_ready,
        "production_site_ready": production_ready,
        "checks": checks,
        "boundary": "production_site_ready requires verified technical evidence, thirty-day continuity with service targets and recovery drills, a four-week named operating model with duty separation and escalation, production auth/CORS, TLS and secret-manager attestations",
    }


def configure_operations(app: FastAPI) -> None:
    app.add_middleware(OperationsMiddleware)

    @app.get("/health/live", tags=["operations"])
    async def health_live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/health/ready", tags=["operations"])
    def health_ready() -> JSONResponse:
        report = readiness_report()
        return JSONResponse(report, status_code=200 if report["open_source_runtime_ready"] else 503)

    @app.get("/metrics", tags=["operations"], include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(RUNTIME_METRICS.prometheus(), media_type="text/plain; version=0.0.4")
