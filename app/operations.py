from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


def is_production() -> bool:
    return os.getenv("PORT_DT_ENV", "development").strip().lower() == "production"


def cors_origins() -> list[str]:
    configured = [item.strip() for item in os.getenv("PORT_DT_CORS_ORIGINS", "").split(",") if item.strip()]
    if configured:
        if is_production() and any(item == "*" or not item.startswith(("https://", "http://")) for item in configured):
            return []
        return configured
    return [] if is_production() else ["http://127.0.0.1:8000", "http://localhost:8000"]


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
            keys = api_keys()
            admins = admin_api_keys()
            supplied = request.headers.get("X-API-Key", "")
            if not keys and not admins:
                response: Response = JSONResponse({"detail": "production API authentication is not configured", "request_id": request_id}, status_code=503)
            elif any(hmac.compare_digest(supplied, expected) for expected in admins):
                request.state.auth_role = "admin"
                response = await call_next(request)
            elif not any(hmac.compare_digest(supplied, expected) for expected in keys):
                response = JSONResponse({"detail": "invalid or missing API key", "request_id": request_id}, status_code=401)
            elif requires_admin(request):
                response = JSONResponse({"detail": "administrator API key required", "request_id": request_id}, status_code=403)
            else:
                request.state.auth_role = "operator"
                response = await call_next(request)
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
        return response


def readiness_report() -> Dict[str, Any]:
    from app.services.rl_training.datasets import dataset_quality_report, load_port_dataset
    from app.services.rl_training.trainer import TRAINING_MANAGER

    checks: Dict[str, Dict[str, Any]] = {}
    try:
        quality = dataset_quality_report(load_port_dataset("public_port_ops_v1", TRAINING_MANAGER.data_root))
        checks["canonical_dataset"] = {"ok": quality["training_eligible"], "status": quality["status"], "sha256": quality["dataset_sha256"]}
    except Exception as exc:
        checks["canonical_dataset"] = {"ok": False, "error": str(exc)}
    runtime = TRAINING_MANAGER.capabilities().get("runtime") or {}
    checks["rl_runtime"] = {"ok": bool(runtime.get("available")), **runtime}
    origins = cors_origins()
    keys = api_keys()
    admins = admin_api_keys()
    checks["cors"] = {"ok": bool(origins), "origins": origins}
    checks["api_authentication"] = {"ok": (not is_production()) or bool(keys or admins), "mode": "role_separated_api_key" if is_production() else "development_open"}
    checks["privileged_api_key"] = {"ok": (not is_production()) or bool(admins), "required_for_research_api": False}
    checks["twin_graph"] = {"ok": bool(os.getenv("PORT_DT_TWIN_GRAPH_PATH", "").strip()), "required_for_research_api": False}
    checks["site_calibration"] = {"ok": bool(os.getenv("PORT_DT_TWIN_CALIBRATION_PATH", "").strip()), "required_for_research_api": False}
    open_source_ready = all(checks[name]["ok"] for name in ("canonical_dataset", "rl_runtime"))
    production_ready = open_source_ready and all(checks[name]["ok"] for name in ("cors", "api_authentication", "privileged_api_key", "twin_graph", "site_calibration"))
    return {
        "status": "ready" if open_source_ready else "not_ready",
        "open_source_runtime_ready": open_source_ready,
        "production_site_ready": production_ready,
        "checks": checks,
        "boundary": "production_site_ready requires site-specific graph, calibration, authentication and CORS configuration",
    }


def configure_operations(app: FastAPI) -> None:
    app.add_middleware(OperationsMiddleware)

    @app.get("/health/live", tags=["operations"])
    async def health_live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/health/ready", tags=["operations"])
    async def health_ready() -> JSONResponse:
        report = readiness_report()
        return JSONResponse(report, status_code=200 if report["open_source_runtime_ready"] else 503)

    @app.get("/metrics", tags=["operations"], include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(RUNTIME_METRICS.prometheus(), media_type="text/plain; version=0.0.4")
