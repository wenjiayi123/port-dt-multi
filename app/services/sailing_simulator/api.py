"""Desktop Godot sailing simulator integration.

The Godot project does not expose an HTTP control server yet, so this service
keeps the first linkage layer honest: launch the project, load the configured
main scene, run the existing RL smoke test, and return explicit staged actions
for controls that still need a Godot-side command bridge.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from app.services.rl_integration.api import _load_config


router = APIRouter(prefix="/api/sailing", tags=["sailing-simulator"])

MAIN_SCENE = "res://main.tscn"
SMOKE_SCRIPT = "res://tools/ship_rl_smoke_test.gd"
CONTROL_MODE = "launch_and_preset_scene"

_sailing_process: Optional[subprocess.Popen[Any]] = None
_last_launch: Dict[str, Any] = {}
_action_log: List[Dict[str, Any]] = []


SAILING_ACTIONS: List[Dict[str, Any]] = [
    {
        "id": "open_sailing_simulator",
        "label": "打开航行模拟器",
        "description": "启动由环境变量配置的 Godot 航行模拟器主场景。",
        "keywords": ["打开航行模拟器", "启动航行模拟器", "打开模拟器", "启动模拟器", "模拟器启动", "打开Godot模拟器", "启动Godot模拟器", "打开船舶模拟器", "启动船舶模拟器", "航行模拟器", "航行沙盘"],
        "button_label": "打开 Godot 航行模拟器",
        "preset": "main_scene",
        "requires_human_confirm": True,
    },
    {
        "id": "start_navigation_demo",
        "label": "启动航线演示",
        "description": "启动 Godot 航行模拟器并加载主场景；当前作为航线演示预设入口。",
        "keywords": ["启动航线演示", "开始航线演示", "启动航行演示", "开始航行演示", "演示航线", "路线演示", "开始导航演示", "自动航行"],
        "button_label": "启动航线演示",
        "preset": "route_demo",
        "requires_human_confirm": True,
    },
    {
        "id": "switch_ship_view",
        "label": "切换船舶视角",
        "description": "Godot 端尚无 HTTP 控制入口，先启动模拟器并标记视角切换动作。",
        "keywords": ["切换船舶视角", "切到船舶视角", "查看船舶视角", "切换驾驶视角", "跟随船舶", "跟随船", "切换视角", "换船", "切换船"],
        "button_label": "切换船舶视角",
        "preset": "ship_view",
        "requires_human_confirm": True,
    },
    {
        "id": "run_sailing_rl_smoke_test",
        "label": "运行 RL 航行场景 smoke test",
        "description": "用 Godot headless 执行 res://tools/ship_rl_smoke_test.gd。",
        "keywords": ["运行rl航行场景smoketest", "航行smoke test", "rl航行测试", "ship rl smoke", "运行航行测试", "运行模拟器测试", "模拟器smoke", "航行场景测试", "rl场景测试", "烟雾测试"],
        "button_label": "运行 smoke test",
        "preset": "headless_smoke_test",
        "requires_human_confirm": True,
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sailing_cfg() -> Dict[str, Any]:
    return _load_config().get("sailing_simulator", {})


def _desktop_enabled() -> bool:
    return os.getenv("PORT_DT_ENABLE_DESKTOP_INTEGRATIONS", "").strip().lower() in {"1", "true", "yes", "on"}


def _path_info(value: Any) -> Dict[str, Any]:
    path = Path(str(value or "")).expanduser()
    configured = bool(str(value or "").strip())
    return {"configured": configured, "artifact_id": path.name if configured else None, "exists": configured and path.exists()}


def _append_log(action_id: str, status: str, detail: Dict[str, Any]) -> None:
    _action_log.insert(
        0,
        {
            "ts": _utc_now(),
            "action_id": action_id,
            "status": status,
            "detail": detail,
        },
    )
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
    global _sailing_process
    if _sailing_process is None:
        return {"tracked": False, "running": False, "pid": None, "returncode": None}
    returncode = _sailing_process.poll()
    running = returncode is None and _is_pid_alive(_sailing_process.pid)
    return {
        "tracked": True,
        "running": running,
        "pid": _sailing_process.pid,
        "returncode": returncode,
    }


def _launch_command(preset: str = "main_scene", scene: str = MAIN_SCENE) -> List[str]:
    cfg = _sailing_cfg()
    executable = str(cfg.get("godot_executable") or "")
    project_path = str(cfg.get("project_path") or "")
    command = [executable, "--path", project_path]
    if scene:
        command.append(scene)
    return command


def _smoke_command() -> List[str]:
    cfg = _sailing_cfg()
    executable = str(cfg.get("godot_executable") or "")
    project_path = str(cfg.get("project_path") or "")
    return [executable, "--headless", "--path", project_path, "--script", SMOKE_SCRIPT]


def sailing_status() -> Dict[str, Any]:
    cfg = _sailing_cfg()
    enabled = _desktop_enabled()
    project_value = str(cfg.get("project_path") or "")
    project_root = _path_info(project_value)
    project_file = _path_info(cfg.get("project_file"))
    godot_executable = _path_info(cfg.get("godot_executable"))
    smoke_script = _path_info(Path(project_value) / "tools" / "ship_rl_smoke_test.gd" if project_value else "")
    process = _process_state()
    launchable = bool(enabled and project_root["exists"] and project_file["exists"] and godot_executable["exists"])
    return {
        "ok": launchable,
        "enabled": enabled,
        "updated_at": _utc_now(),
        "name": cfg.get("name", "航行模拟器"),
        "control_mode": CONTROL_MODE,
        "launchable": launchable,
        "label": "航行模拟器可启动" if launchable else "航行模拟器不可启动",
        "project_root": project_root,
        "project_file": project_file,
        "godot_executable": godot_executable,
        "main_scene": MAIN_SCENE,
        "smoke_script": smoke_script,
        "process": process,
        "last_launch": _last_launch,
        "actions": SAILING_ACTIONS,
    }


def launch_sailing_simulator(payload: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
    global _sailing_process, _last_launch
    payload = payload or {}
    if not _desktop_enabled():
        return {"type": "godot_launch", "status": "failed", "error": "desktop integrations disabled"}
    status = sailing_status()
    preset = str(payload.get("preset") or "main_scene")
    scene = str(payload.get("scene") or MAIN_SCENE)
    force_new = bool(payload.get("force_new"))
    command = _launch_command(preset=preset, scene=scene)
    launch_packet: Dict[str, Any] = {
        "type": "godot_launch",
        "dry_run": dry_run,
        "preset": preset,
        "scene": scene,
        "command_artifacts": [Path(item).name for item in command if "/" in item or "\\" in item],
        "control_mode": CONTROL_MODE,
        "note": "Godot 端暂未开放 HTTP 控制；当前联动执行启动和预设主场景加载。",
    }
    if not status["launchable"]:
        launch_packet.update({"status": "failed", "error": "航行模拟器项目或 Godot 可执行文件不存在", "status_detail": status})
        _append_log("open_sailing_simulator", "failed", launch_packet)
        return launch_packet
    if dry_run:
        launch_packet["status"] = "ready_to_launch"
        return launch_packet

    process = _process_state()
    if process["running"] and not force_new:
        launch_packet.update({"status": "already_running", "pid": process["pid"]})
        _append_log("open_sailing_simulator", "already_running", launch_packet)
        return launch_packet

    cfg = _sailing_cfg()
    env = os.environ.copy()
    env["PORT_DT_LINKAGE_SOURCE"] = str(payload.get("source") or "port-dt-multi")
    env["PORT_DT_SAILING_PRESET"] = preset
    try:
        _sailing_process = subprocess.Popen(
            command,
            cwd=str(cfg.get("project_path") or ""),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        launch_packet.update({"status": "launched", "pid": _sailing_process.pid})
        _last_launch = {
            "ts": _utc_now(),
            "preset": preset,
            "scene": scene,
            "pid": _sailing_process.pid,
            "source": payload.get("source") or "port-dt-multi",
        }
        _append_log("open_sailing_simulator", "launched", launch_packet)
        return launch_packet
    except Exception as exc:
        launch_packet.update({"status": "failed", "error": str(exc)})
        _append_log("open_sailing_simulator", "failed", launch_packet)
        return launch_packet


def run_sailing_smoke_test(payload: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
    payload = payload or {}
    command = _smoke_command()
    timeout_sec = float(payload.get("timeout_sec") or 35)
    packet: Dict[str, Any] = {
        "type": "godot_headless_smoke_test",
        "dry_run": dry_run,
        "command_artifacts": [Path(item).name for item in command if "/" in item or "\\" in item],
        "script": SMOKE_SCRIPT,
        "timeout_sec": timeout_sec,
    }
    status = sailing_status()
    if not status["launchable"]:
        packet.update({"status": "failed", "error": "航行模拟器项目或 Godot 可执行文件不存在", "status_detail": status})
        _append_log("run_sailing_rl_smoke_test", "failed", packet)
        return packet
    if dry_run:
        packet["status"] = "ready_to_run"
        return packet
    try:
        completed = subprocess.run(
            command,
            cwd=str(_sailing_cfg().get("project_path") or ""),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        ok = completed.returncode == 0 and "SHIP_RL_OK" in output
        packet.update(
            {
                "status": "passed" if ok else "failed",
                "returncode": completed.returncode,
                "ok_marker": "SHIP_RL_OK" in output,
                "output_tail": output[-2400:],
            }
        )
        _append_log("run_sailing_rl_smoke_test", packet["status"], packet)
        return packet
    except subprocess.TimeoutExpired as exc:
        packet.update({"status": "timeout", "error": str(exc)})
        _append_log("run_sailing_rl_smoke_test", "timeout", packet)
        return packet
    except Exception as exc:
        packet.update({"status": "failed", "error": str(exc)})
        _append_log("run_sailing_rl_smoke_test", "failed", packet)
        return packet


def _action_by_id(action_id: str) -> Optional[Dict[str, Any]]:
    return next((item for item in SAILING_ACTIONS if item["id"] == action_id), None)


def execute_sailing_action(action_id: str, payload: Optional[Dict[str, Any]] = None, dry_run: bool = True) -> Dict[str, Any]:
    payload = payload or {}
    action = _action_by_id(action_id)
    if not action:
        return {"type": "sailing_action", "status": "failed", "error": f"未知航行模拟器动作：{action_id}"}

    if action_id == "run_sailing_rl_smoke_test":
        result = run_sailing_smoke_test(payload, dry_run=dry_run)
    elif action_id == "switch_ship_view":
        launch = launch_sailing_simulator({**payload, "preset": "ship_view"}, dry_run=dry_run)
        result = {
            "type": "sailing_staged_control",
            "status": launch.get("status") if dry_run else ("staged_no_http_control" if launch.get("status") in {"launched", "already_running"} else launch.get("status")),
            "dry_run": dry_run,
            "launch": launch,
            "control_mode": CONTROL_MODE,
            "godot_side_needed": True,
            "manual_fallback": {
                "panel_key": "X",
                "action": "在 Godot 航行模拟器内切换船队/受控船舶视角",
            },
        }
        _append_log(action_id, str(result["status"]), result)
    else:
        preset = str(action.get("preset") or "main_scene")
        result = launch_sailing_simulator({**payload, "preset": preset}, dry_run=dry_run)
        result["action_id"] = action_id
        result["action_label"] = action.get("label")
        if action_id == "start_navigation_demo":
            result["preset_note"] = "已加载 main.tscn；Godot 端 ShipScenarioController 提供开放水域/港口航线预设。"
        _append_log(action_id, str(result.get("status")), result)
    return result


@router.get("/status", summary="航行模拟器状态")
async def get_sailing_status() -> JSONResponse:
    return JSONResponse(sailing_status())


@router.post("/launch", summary="启动 Godot 航行模拟器")
async def post_sailing_launch(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    dry_run = bool(payload.get("dry_run", False))
    confirm = bool(payload.get("confirm", False))
    if not dry_run and not confirm:
        return JSONResponse(
            {
                "ok": False,
                "status": "confirmation_required",
                "human_confirmation": {"required": True, "provided": False},
                "preview": launch_sailing_simulator(payload, dry_run=True),
            },
            status_code=200,
        )
    result = launch_sailing_simulator(payload, dry_run=dry_run)
    return JSONResponse({"ok": result.get("status") not in {"failed"}, "result": result, "status": sailing_status()})


@router.get("/actions/registry", summary="航行模拟器动作注册表")
async def sailing_action_registry() -> JSONResponse:
    return JSONResponse(
        {
            "updated_at": _utc_now(),
            "control_mode": CONTROL_MODE,
            "count": len(SAILING_ACTIONS),
            "actions": SAILING_ACTIONS,
        }
    )


@router.post("/actions/execute", summary="执行航行模拟器动作")
async def post_sailing_action_execute(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    action_id = str(payload.get("action_id") or "").strip()
    if not action_id:
        raise HTTPException(status_code=400, detail="缺少 action_id")
    action = _action_by_id(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"未知航行模拟器动作：{action_id}")
    dry_run = bool(payload.get("dry_run", True))
    if not dry_run and action.get("requires_human_confirm") and not bool(payload.get("confirm")):
        return JSONResponse(
            {
                "ok": False,
                "status": "confirmation_required",
                "action": action,
                "human_confirmation": {"required": True, "provided": False},
                "preview": execute_sailing_action(action_id, payload, dry_run=True),
            }
        )
    result = execute_sailing_action(action_id, payload, dry_run=dry_run)
    return JSONResponse(
        {
            "ok": result.get("status") not in {"failed", "timeout"},
            "updated_at": _utc_now(),
            "action": action,
            "dry_run": dry_run,
            "execution": result,
            "status": sailing_status(),
        }
    )


@router.get("/logs", summary="航行模拟器动作日志")
async def get_sailing_logs(limit: int = 30) -> JSONResponse:
    limit = max(1, min(int(limit or 30), 80))
    return JSONResponse({"updated_at": _utc_now(), "items": _action_log[:limit]})
