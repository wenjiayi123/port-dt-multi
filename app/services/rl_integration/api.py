"""Health and configuration endpoints for the RL panel linkage.

This module is intentionally scoped to the top-menu RL panel.  It reports
whether the local Xiaoyi assistant is reachable, whether the current FastAPI app
has the RL routes mounted, and whether the desktop sailing simulator can be
started from the configured Godot project.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.error import URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(prefix="/api/rl/integration", tags=["rl-integration"])

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"


DEFAULT_CONFIG: Dict[str, Any] = {
    "port_dt_multi": {
        "name": "port-dt-multi",
        "health_route": "/api/rl/integration/health",
        "rl_panel_route": "/rl-panel",
    },
    "xiaoyi_ai": {
        "name": "小懿AI",
        "base_url": "http://127.0.0.1:8010",
        "health_path": "/health",
        "chat_path": "/api/chat",
        "project_path": "",
        "start_command": "",
    },
    "sailing_simulator": {
        "name": "航行模拟器",
        "project_path": "",
        "project_file": "",
        "godot_executable": "",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_config() -> Dict[str, Any]:
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        raw = {}

    cfg = _deep_merge(DEFAULT_CONFIG, raw if isinstance(raw, dict) else {})
    xiaoyi_url = os.getenv("XIAOYI_AI_BASE_URL")
    xiaoyi_project = os.getenv("XIAOYI_AI_PROJECT")
    xiaoyi_start = os.getenv("XIAOYI_AI_START_COMMAND")
    godot_exec = os.getenv("SAILING_SIM_GODOT")
    sailing_project = os.getenv("SAILING_SIM_PROJECT")

    if xiaoyi_url:
        cfg["xiaoyi_ai"]["base_url"] = xiaoyi_url.rstrip("/")
    if xiaoyi_project:
        cfg["xiaoyi_ai"]["project_path"] = str(Path(xiaoyi_project).expanduser())
    if xiaoyi_start:
        cfg["xiaoyi_ai"]["start_command"] = xiaoyi_start
    if godot_exec:
        cfg["sailing_simulator"]["godot_executable"] = godot_exec
    if sailing_project:
        project_path = Path(sailing_project).expanduser()
        cfg["sailing_simulator"]["project_path"] = str(project_path)
        cfg["sailing_simulator"]["project_file"] = str(project_path / "project.godot")
    return cfg


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _route_exists(request: Request, path: str, methods: Optional[Iterable[str]] = None) -> bool:
    wanted_methods = {m.upper() for m in methods or []}
    for route in request.app.routes:
        if getattr(route, "path", None) != path:
            continue
        route_methods = {m.upper() for m in getattr(route, "methods", set()) or set()}
        if not wanted_methods or route_methods.intersection(wanted_methods):
            return True
    return False


def _probe_http(url: str, timeout_sec: float = 0.55) -> Dict[str, Any]:
    req = UrlRequest(url, headers={"User-Agent": "port-dt-multi-rl-integration/1.0"})
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            status_code = int(getattr(resp, "status", 200))
            ok = 200 <= status_code < 300
            return {"ok": ok, "status_code": status_code, "error": None}
    except URLError:
        return {"ok": False, "status_code": None, "error": "integration endpoint unavailable"}
    except Exception:
        return {"ok": False, "status_code": None, "error": "integration probe failed"}


def _probe_xiaoyi_service(
    base_url: str,
    health_path: str,
    chat_path: str,
    timeout_sec: float = 0.8,
) -> Dict[str, Any]:
    """Reject a health-only service unless it also exposes POST /api/chat."""
    health_url = base_url.rstrip("/") + health_path
    health = _probe_http(health_url, timeout_sec=timeout_sec)
    schema_url = base_url.rstrip("/") + "/openapi.json"
    schema: Dict[str, Any] = {
        "ok": False,
        "status_code": None,
        "error": "health check failed",
        "post_supported": False,
    }
    if health.get("ok"):
        req = UrlRequest(schema_url, headers={"User-Agent": "port-dt-multi-rl-integration/1.0"})
        try:
            with urlopen(req, timeout=timeout_sec) as resp:
                status_code = int(getattr(resp, "status", 200))
                document = json.loads(resp.read().decode("utf-8"))
                methods = (document.get("paths") or {}).get(chat_path) or {}
                post_supported = isinstance(methods, dict) and "post" in methods
                schema = {
                    "ok": 200 <= status_code < 300 and post_supported,
                    "status_code": status_code,
                    "error": None if post_supported else f"POST {chat_path} missing from OpenAPI",
                    "post_supported": post_supported,
                }
        except Exception:
            schema["error"] = "integration schema endpoint unavailable"
    chat_capable = bool(health.get("ok") and schema.get("ok"))
    return {
        "ok": chat_capable,
        "chat_capable": chat_capable,
        "identity": "xiaoyi_chat_service" if chat_capable else "health_only_not_xiaoyi",
        "reason": None if chat_capable else (schema.get("error") or health.get("error")),
        "health": health,
        "schema": schema,
    }


def _path_status(path_value: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser()
    configured = bool(str(path_value or "").strip())
    return {"configured": configured, "artifact_id": path.name if configured else None, "exists": configured and path.exists()}


@router.get("/config", summary="RL 面板联动配置")
async def rl_integration_config() -> JSONResponse:
    cfg = _load_config()
    desktop_enabled = os.getenv("PORT_DT_ENABLE_DESKTOP_INTEGRATIONS", "").strip().lower() in {"1", "true", "yes", "on"}
    return JSONResponse(
        {
            "updated_at": _utc_now(),
            "scope": "top-menu RL panel linkage",
            "desktop_integrations_enabled": desktop_enabled,
            "config": {
                "port_dt_multi": cfg["port_dt_multi"],
                "xiaoyi_ai": {
                    "name": cfg["xiaoyi_ai"].get("name"),
                    "base_url": cfg["xiaoyi_ai"].get("base_url"),
                    "health_path": cfg["xiaoyi_ai"].get("health_path"),
                    "chat_path": cfg["xiaoyi_ai"].get("chat_path"),
                    "project_configured": bool(cfg["xiaoyi_ai"].get("project_path")),
                    "start_command_configured": bool(cfg["xiaoyi_ai"].get("start_command")),
                },
                "sailing_simulator": {
                    "name": cfg["sailing_simulator"].get("name"),
                    "project_configured": bool(cfg["sailing_simulator"].get("project_path")),
                    "godot_configured": bool(cfg["sailing_simulator"].get("godot_executable")),
                },
            },
        }
    )


@router.get("/health", summary="RL 面板联动健康检查")
async def rl_integration_health(request: Request) -> JSONResponse:
    cfg = _load_config()
    desktop_enabled = os.getenv("PORT_DT_ENABLE_DESKTOP_INTEGRATIONS", "").strip().lower() in {"1", "true", "yes", "on"}
    xiaoyi_cfg = cfg["xiaoyi_ai"]
    sailing_cfg = cfg["sailing_simulator"]

    xiaoyi_base_url = str(xiaoyi_cfg.get("base_url", "")).rstrip("/")
    xiaoyi_health_path = str(xiaoyi_cfg.get("health_path", "/health"))
    xiaoyi_chat_path = str(xiaoyi_cfg.get("chat_path", "/api/chat"))
    xiaoyi_health_url = xiaoyi_base_url + xiaoyi_health_path
    xiaoyi_probe = _probe_xiaoyi_service(
        xiaoyi_base_url,
        xiaoyi_health_path,
        xiaoyi_chat_path,
    ) if desktop_enabled else {
        "ok": False,
        "chat_capable": False,
        "identity": "disabled",
        "reason": "desktop integrations disabled",
        "health": {"ok": False, "status_code": None, "error": "desktop integrations disabled"},
        "schema": {"ok": False, "status_code": None, "error": "desktop integrations disabled"},
    }
    xiaoyi_project = _path_status(str(xiaoyi_cfg.get("project_path", "")))
    xiaoyi_online = bool(xiaoyi_probe.get("chat_capable"))

    rl_routes = {
        "/rl-panel": _route_exists(request, "/rl-panel", {"GET"}),
        "/api/rl/train/baselines": _route_exists(request, "/api/rl/train/baselines", {"GET"}),
        "/api/rl/train/start": _route_exists(request, "/api/rl/train/start", {"POST"}),
        "/api/rl/train/status": _route_exists(request, "/api/rl/train/status", {"GET", "POST"}),
        "/api/rl/train/metrics": _route_exists(request, "/api/rl/train/metrics", {"GET"}),
        "/api/rl/actions/registry": _route_exists(request, "/api/rl/actions/registry", {"GET"}),
        "/api/rl/actions/resolve": _route_exists(request, "/api/rl/actions/resolve", {"POST"}),
        "/api/assistant/actions/execute": _route_exists(request, "/api/assistant/actions/execute", {"POST"}),
    }
    optional_routes = {
        "/api/rl/strategies": _route_exists(request, "/api/rl/strategies", {"GET"}),
        "/api/rl/simulate": _route_exists(request, "/api/rl/simulate", {"POST"}),
        "/api/rl/dispatch": _route_exists(request, "/api/rl/dispatch", {"POST"}),
        "/api/rlops/policies/verify": _route_exists(request, "/api/rlops/policies/verify", {"POST"}),
        "/api/xiaoyi/status": _route_exists(request, "/api/xiaoyi/status", {"GET"}),
        "/api/xiaoyi/launch": _route_exists(request, "/api/xiaoyi/launch", {"POST"}),
        "/api/sailing/status": _route_exists(request, "/api/sailing/status", {"GET"}),
        "/api/sailing/launch": _route_exists(request, "/api/sailing/launch", {"POST"}),
        "/api/sailing/actions/execute": _route_exists(request, "/api/sailing/actions/execute", {"POST"}),
    }
    rl_online = all(rl_routes.values())

    sailing_project = _path_status(str(sailing_cfg.get("project_file", "")))
    sailing_root = _path_status(str(sailing_cfg.get("project_path", "")))
    godot_executable = _path_status(str(sailing_cfg.get("godot_executable", "")))
    sailing_launchable = bool(sailing_project["exists"] and godot_executable["exists"])

    systems = {
        "port_dt_multi": {
            "name": cfg["port_dt_multi"].get("name", "port-dt-multi"),
            "online": True,
            "label": "port-dt-multi在线",
            "health_route": cfg["port_dt_multi"].get("health_route"),
            "rl_panel_route": cfg["port_dt_multi"].get("rl_panel_route"),
        },
        "xiaoyi_ai": {
            "name": xiaoyi_cfg.get("name", "小懿AI"),
            "online": xiaoyi_online,
            "label": "小懿在线" if xiaoyi_online else "小懿未启动",
            "base_url": xiaoyi_base_url,
            "health_url": xiaoyi_health_url,
            "chat_url": xiaoyi_base_url + xiaoyi_chat_path,
            "chat_capable": xiaoyi_online,
            "identity_check": xiaoyi_probe.get("identity"),
            "project": xiaoyi_project,
            "start_command_configured": bool(xiaoyi_cfg.get("start_command")),
            "status_code": (xiaoyi_probe.get("health") or {}).get("status_code"),
            "error": xiaoyi_probe.get("reason"),
            "probe": xiaoyi_probe,
            "routes": {
                "/api/xiaoyi/status": _route_exists(request, "/api/xiaoyi/status", {"GET"}),
                "/api/xiaoyi/launch": _route_exists(request, "/api/xiaoyi/launch", {"POST"}),
            },
        },
        "rl_interface": {
            "name": "RL接口",
            "online": rl_online,
            "label": "RL接口在线" if rl_online else "RL接口缺失",
            "routes": rl_routes,
            "optional_routes": optional_routes,
        },
        "sailing_simulator": {
            "name": sailing_cfg.get("name", "航行模拟器"),
            "launchable": sailing_launchable,
            "label": "航行模拟器可启动" if sailing_launchable else "航行模拟器不可启动",
            "project_root": sailing_root,
            "project_file": sailing_project,
            "godot_executable": godot_executable,
            "control_mode": "launch_and_preset_scene",
            "routes": {
                "/api/sailing/status": _route_exists(request, "/api/sailing/status", {"GET"}),
                "/api/sailing/launch": _route_exists(request, "/api/sailing/launch", {"POST"}),
                "/api/sailing/actions/execute": _route_exists(request, "/api/sailing/actions/execute", {"POST"}),
            },
        },
    }

    overall_ready = rl_online
    return JSONResponse(
        {
            "ok": overall_ready,
            "updated_at": _utc_now(),
            "scope": "top-menu RL panel linkage",
            "summary": {
                "xiaoyi": systems["xiaoyi_ai"]["label"],
                "rl": systems["rl_interface"]["label"],
                "sailing": systems["sailing_simulator"]["label"],
            },
            "desktop_integration_ready": bool(xiaoyi_online and sailing_launchable),
            "desktop_integrations_enabled": desktop_enabled,
            "systems": systems,
        }
    )
