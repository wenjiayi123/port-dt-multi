"""Local Xiaoyi AI service launcher and status endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from app.services.rl_integration.api import _load_config


router = APIRouter(prefix="/api/xiaoyi", tags=["xiaoyi-ai"])

_xiaoyi_process: Optional[subprocess.Popen[Any]] = None
_last_launch: Dict[str, Any] = {}
_action_log: List[Dict[str, Any]] = []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _cfg() -> Dict[str, Any]:
    return _load_config().get("xiaoyi_ai", {})


def _desktop_enabled() -> bool:
    return os.getenv("PORT_DT_ENABLE_DESKTOP_INTEGRATIONS", "").strip().lower() in {"1", "true", "yes", "on"}


def _path_info(value: Any) -> Dict[str, Any]:
    path = Path(str(value or "")).expanduser()
    configured = bool(str(value or "").strip())
    return {"configured": configured, "artifact_id": path.name if configured else None, "exists": configured and path.exists()}


def _probe_http(url: str, timeout_sec: float = 0.55) -> Dict[str, Any]:
    req = UrlRequest(url, headers={"User-Agent": "port-dt-multi-xiaoyi-launcher/1.0"})
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            status_code = int(getattr(resp, "status", 200))
            return {"ok": 200 <= status_code < 300, "status_code": status_code, "error": None}
    except URLError as exc:
        return {"ok": False, "status_code": None, "error": str(getattr(exc, "reason", exc))}
    except Exception as exc:
        return {"ok": False, "status_code": None, "error": str(exc)}


def _append_log(action_id: str, status: str, detail: Dict[str, Any]) -> None:
    _action_log.insert(0, {"ts": _utc_now(), "action_id": action_id, "status": status, "detail": detail})
    del _action_log[80:]


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _process_state() -> Dict[str, Any]:
    global _xiaoyi_process
    if _xiaoyi_process is None:
        return {"tracked": False, "running": False, "pid": None, "returncode": None}
    returncode = _xiaoyi_process.poll()
    return {
        "tracked": True,
        "running": returncode is None and _is_pid_alive(_xiaoyi_process.pid),
        "pid": _xiaoyi_process.pid,
        "returncode": returncode,
    }


def xiaoyi_status() -> Dict[str, Any]:
    cfg = _cfg()
    enabled = _desktop_enabled()
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    health_path = str(cfg.get("health_path") or "/health")
    health_url = base_url + health_path
    probe = _probe_http(health_url) if enabled else {"ok": False, "status_code": None, "error": "desktop integrations disabled"}
    project_value = str(cfg.get("project_path") or "")
    project = _path_info(project_value)
    run_script = _path_info(Path(project_value) / "run.sh" if project_value else "")
    launchable = bool(enabled and project["exists"] and run_script["exists"] and cfg.get("start_command"))
    online = bool(probe.get("ok"))
    return {
        "ok": online,
        "enabled": enabled,
        "updated_at": _utc_now(),
        "name": cfg.get("name", "小懿AI"),
        "online": online,
        "launchable": launchable,
        "label": "小懿在线" if online else ("小懿可启动" if launchable else "小懿不可启动"),
        "base_url": base_url,
        "health_url": health_url,
        "chat_url": base_url + str(cfg.get("chat_path") or "/api/chat"),
        "project": project,
        "run_script": run_script,
        "start_command_configured": bool(cfg.get("start_command")),
        "probe": probe,
        "process": _process_state(),
        "last_launch": _last_launch,
    }


def launch_xiaoyi(payload: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
    global _xiaoyi_process, _last_launch
    payload = payload or {}
    if not _desktop_enabled():
        return {"type": "xiaoyi_service_launch", "status": "failed", "error": "desktop integrations disabled"}
    status = xiaoyi_status()
    cfg = _cfg()
    command = str(cfg.get("start_command") or "")
    packet: Dict[str, Any] = {
        "type": "xiaoyi_service_launch",
        "dry_run": dry_run,
        "command_configured": bool(command),
        "base_url": status.get("base_url"),
        "health_url": status.get("health_url"),
        "note": "启动小懿本地 FastAPI 服务；启动后健康检查为 /health，问答接口为 /api/chat。",
    }
    if status["online"]:
        packet.update({"status": "already_online", "health": status})
        _append_log("start_xiaoyi_ai", "already_online", packet)
        return packet
    if not status["launchable"]:
        packet.update({"status": "failed", "error": "小懿项目、run.sh 或 start_command 不可用", "health": status})
        _append_log("start_xiaoyi_ai", "failed", packet)
        return packet
    if dry_run:
        packet["status"] = "ready_to_launch"
        return packet

    process = _process_state()
    if process["running"] and not bool(payload.get("force_new")):
        packet.update({"status": "already_running", "pid": process["pid"]})
        _append_log("start_xiaoyi_ai", "already_running", packet)
        return packet

    env = os.environ.copy()
    env["PORT_DT_LINKAGE_SOURCE"] = str(payload.get("source") or "port-dt-multi")
    try:
        _xiaoyi_process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(cfg.get("project_path") or ""),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        packet.update({"status": "launched", "pid": _xiaoyi_process.pid})
        _last_launch = {"ts": _utc_now(), "pid": _xiaoyi_process.pid, "source": payload.get("source") or "port-dt-multi"}
        for _ in range(20):
            health = xiaoyi_status()
            if health["online"]:
                packet.update({"status": "online", "health": health})
                break
            time.sleep(0.35)
        _append_log("start_xiaoyi_ai", str(packet.get("status") or "launched"), packet)
        return packet
    except Exception as exc:
        packet.update({"status": "failed", "error": str(exc)})
        _append_log("start_xiaoyi_ai", "failed", packet)
        return packet


def execute_xiaoyi_action(action_id: str, payload: Optional[Dict[str, Any]] = None, dry_run: bool = True) -> Dict[str, Any]:
    if action_id != "start_xiaoyi_ai":
        return {"type": "xiaoyi_action", "status": "failed", "error": f"未知小懿动作：{action_id}"}
    return launch_xiaoyi(payload, dry_run=dry_run)


@router.get("/status", summary="小懿AI状态")
async def get_xiaoyi_status() -> JSONResponse:
    return JSONResponse(xiaoyi_status())


@router.post("/launch", summary="启动小懿AI")
async def post_xiaoyi_launch(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    dry_run = bool(payload.get("dry_run", False))
    confirm = bool(payload.get("confirm", False))
    if not dry_run and not confirm:
        return JSONResponse(
            {
                "ok": False,
                "status": "confirmation_required",
                "human_confirmation": {"required": True, "provided": False},
                "preview": launch_xiaoyi(payload, dry_run=True),
            }
        )
    result = launch_xiaoyi(payload, dry_run=dry_run)
    return JSONResponse({"ok": result.get("status") not in {"failed"}, "result": result, "status": xiaoyi_status()})


@router.get("/logs", summary="小懿AI启动日志")
async def get_xiaoyi_logs(limit: int = 30) -> JSONResponse:
    limit = max(1, min(int(limit or 30), 80))
    return JSONResponse({"updated_at": _utc_now(), "items": _action_log[:limit]})
