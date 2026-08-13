"""Allow-listed action registry for Xiaoyi-to-port-operations commands.

The registry converts Xiaoyi's recognized intent or a raw user instruction
into a bounded command packet.  Mission actions are read-only navigation into
the evidence-bound copilot; training, desktop launch and policy actions keep
their existing confirmation and dry-run gates.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from app.services.sailing_simulator.api import execute_sailing_action
from app.services.xiaoyi_ai.api import execute_xiaoyi_action


router = APIRouter(prefix="/api/rl/actions", tags=["rl-actions"])


DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "algorithm": "sac",
    "objective": "multi_objective",
    "scenario": "mapped_dataset",
    "asset_group": "all_port",
    "horizon_min": 720,
    "step_min": 5,
    "total_steps": 240000,
    "batch_size": 256,
    "learning_rate": 0.0003,
    "gamma": 0.995,
    "tau": 0.005,
    "entropy_coef": 0.02,
    "replay_buffer": 120000,
    "seed": 42,
    "demand_cap_kw": 520,
    "guardrail_mode": "strict",
    "reward_weights": {"cost": 0.24, "carbon": 0.22, "peak": 0.18, "safety": 0.20},
}


RL_ACTIONS: List[Dict[str, Any]] = [
    {
        "id": "open_ops_copilot",
        "label": "打开小懿运营副驾",
        "category": "xiaoyi_mission",
        "description": "打开小懿任务工作台；默认读取当前孪生态势，不执行生产动作。",
        "intent_aliases": ["open_ops_copilot", "open_xiaoyi_copilot", "打开运营副驾", "打开小懿工作台"],
        "keywords": ["小懿运营副驾", "小懿工作台", "运营副驾", "智能副驾"],
        "route": "/ops-copilot?mission=situation&from=xiaoyi",
        "button_selector": None,
        "button_label": None,
        "button_sequence": [],
        "backend_request": None,
        "execution": {"type": "open_route", "dry_run_default": False},
        "requires_panel": False,
        "requires_human_confirm": False,
    },
    {
        "id": "summarize_current_situation",
        "label": "小懿研判当前态势",
        "category": "xiaoyi_mission",
        "description": "读取回放、数据质量、预测、模型和安全门，形成带哈希的当班态势。",
        "intent_aliases": ["summarize_current_situation", "current_situation", "研判当前态势", "总结当前态势"],
        "keywords": ["当前态势", "现在情况", "当班情况", "态势研判", "系统现状"],
        "route": "/ops-copilot?mission=situation&auto=1&from=xiaoyi",
        "button_selector": None,
        "button_label": None,
        "button_sequence": [],
        "backend_request": None,
        "execution": {"type": "open_route", "dry_run_default": False},
        "requires_panel": False,
        "requires_human_confirm": False,
    },
    {
        "id": "review_twin_forecast",
        "label": "小懿解释未来风险",
        "category": "xiaoyi_mission",
        "description": "解释后端预测区间、工程阈值风险及尚未接入的现场校准证据。",
        "intent_aliases": ["review_twin_forecast", "forecast_risk", "解释未来风险", "查看未来预测"],
        "keywords": ["未来风险", "预测风险", "预测峰值", "未来六小时", "p10", "p50", "p90"],
        "route": "/ops-copilot?mission=forecast&auto=1&from=xiaoyi",
        "button_selector": None,
        "button_label": None,
        "button_sequence": [],
        "backend_request": None,
        "execution": {"type": "open_route", "dry_run_default": False},
        "requires_panel": False,
        "requires_human_confirm": False,
    },
    {
        "id": "explain_current_strategy",
        "label": "小懿解释当前策略",
        "category": "xiaoyi_mission",
        "description": "解释当前模型、相对基线变化、硬约束、安全投影和声明边界。",
        "intent_aliases": ["explain_current_strategy", "strategy_explain", "解释当前策略", "为什么这样调度"],
        "keywords": ["策略解释", "当前策略", "为什么这样调度", "模型决策", "策略依据"],
        "route": "/ops-copilot?mission=strategy&auto=1&from=xiaoyi",
        "button_selector": None,
        "button_label": None,
        "button_sequence": [],
        "backend_request": None,
        "execution": {"type": "open_route", "dry_run_default": False},
        "requires_panel": False,
        "requires_human_confirm": False,
    },
    {
        "id": "triage_monitoring",
        "label": "小懿执行告警分诊",
        "category": "xiaoyi_mission",
        "description": "联动异常、漂移和准入门，生成检查与安全回退顺序。",
        "intent_aliases": ["triage_monitoring", "alert_triage", "告警分诊", "异常分诊"],
        "keywords": ["告警分诊", "异常分诊", "漂移处理", "怎么处理告警", "安全回退"],
        "route": "/ops-copilot?mission=triage&auto=1&from=xiaoyi",
        "button_selector": None,
        "button_label": None,
        "button_sequence": [],
        "backend_request": None,
        "execution": {"type": "open_route", "dry_run_default": False},
        "requires_panel": False,
        "requires_human_confirm": False,
    },
    {
        "id": "prepare_shift_handoff",
        "label": "小懿准备交接班",
        "category": "xiaoyi_mission",
        "description": "生成包含上下文哈希、模型/数据版本、风险和缺失字段的交接预览。",
        "intent_aliases": ["prepare_shift_handoff", "shift_handoff", "准备交接班", "生成交接摘要"],
        "keywords": ["交接班", "交班摘要", "下一班", "班次交接", "交接留痕"],
        "route": "/ops-copilot?mission=handoff&auto=1&from=xiaoyi",
        "button_selector": None,
        "button_label": None,
        "button_sequence": [],
        "backend_request": None,
        "execution": {"type": "open_route", "dry_run_default": False},
        "requires_panel": False,
        "requires_human_confirm": False,
    },
    {
        "id": "prepare_strategy_dry_run",
        "label": "小懿准备策略预演",
        "category": "xiaoyi_mission",
        "description": "只进入策略dry-run准备；监控门阻断时保持安全基线，不下发设备指令。",
        "intent_aliases": ["prepare_strategy_dry_run", "strategy_dry_run", "准备策略预演", "策略演练"],
        "keywords": ["策略预演", "策略演练", "dryrun", "dry-run", "预演一下", "仿真执行"],
        "route": "/ops-copilot?mission=dry_run&auto=1&from=xiaoyi",
        "button_selector": None,
        "button_label": None,
        "button_sequence": [],
        "backend_request": None,
        "execution": {"type": "open_route", "dry_run_default": False},
        "requires_panel": False,
        "requires_human_confirm": False,
    },
    {
        "id": "start_rl_training",
        "label": "启动强化学习训练",
        "category": "rl_training",
        "description": "打开强化学习面板并触发训练控制台的启动训练按钮。",
        "intent_aliases": ["start_rl_training", "rl_train_start", "start_training", "开始训练", "启动训练"],
        "keywords": ["启动训练", "开始训练", "强化学习训练", "rl训练", "训练模型", "训练策略", "开始rl"],
        "route": "/rl-panel?action=start_rl_training&from=xiaoyi",
        "button_selector": "#btnStartTrain",
        "button_label": "启动训练",
        "button_sequence": ["#btnStartTrain"],
        "backend_request": {
            "method": "POST",
            "path": "/api/rl/train/start",
            "body": {"config": DEFAULT_TRAIN_CONFIG, "source": "xiaoyi_action_registry"},
        },
        "execution": {"type": "backend_api_or_frontend_click", "dry_run_default": True},
        "requires_panel": True,
        "requires_human_confirm": True,
    },
    {
        "id": "view_rl_training_status",
        "label": "查看强化学习训练状态",
        "category": "rl_training",
        "description": "查询当前 RL 训练任务的 step、reward、entropy、policy 版本和训练日志。",
        "intent_aliases": ["view_rl_training_status", "rl_train_status", "training_status", "查看训练状态", "查询训练状态"],
        "keywords": ["查看训练状态", "查询训练状态", "训练状态", "训练指标", "训练日志", "查看rl", "reward", "entropy", "policy版本", "策略版本"],
        "route": "/rl-panel?action=view_rl_training_status&from=xiaoyi",
        "button_selector": "#btnPollTrainStatus",
        "button_label": "查看状态",
        "button_sequence": ["#btnPollTrainStatus"],
        "backend_request": {
            "method": "GET",
            "path": "/api/rl/train/status",
        },
        "execution": {"type": "backend_status_query_or_frontend_poll", "dry_run_default": False},
        "requires_panel": True,
        "requires_human_confirm": False,
    },
    {
        "id": "stop_rl_training",
        "label": "停止/暂停强化学习训练",
        "category": "rl_training",
        "description": "映射到训练控制台的暂停按钮；当前 RL 面板使用暂停/继续按钮承接停止类口令。",
        "intent_aliases": ["stop_rl_training", "pause_rl_training", "stop_training", "停止训练", "暂停训练"],
        "keywords": ["停止训练", "暂停训练", "终止训练", "停下训练", "停止rl", "暂停rl"],
        "route": "/rl-panel?action=stop_rl_training",
        "button_selector": "#btnPauseTrain",
        "button_label": "暂停",
        "button_sequence": ["#btnPauseTrain"],
        "backend_request": None,
        "execution": {"type": "frontend_click", "dry_run_default": True},
        "requires_panel": True,
        "requires_human_confirm": True,
    },
    {
        "id": "open_rl_panel",
        "label": "打开强化学习面板",
        "category": "navigation",
        "description": "打开顶部菜单栏里的强化学习面板。",
        "intent_aliases": ["open_rl_panel", "show_rl_panel", "打开强化学习面板", "进入强化学习"],
        "keywords": ["打开强化学习", "进入强化学习", "强化学习面板", "rl面板", "训练面板", "策略面板"],
        "route": "/rl-panel",
        "button_selector": None,
        "button_label": None,
        "button_sequence": [],
        "backend_request": None,
        "execution": {"type": "open_route", "dry_run_default": False},
        "requires_panel": False,
        "requires_human_confirm": False,
    },
    {
        "id": "run_policy_test",
        "label": "运行训练后策略测试",
        "category": "policy_test",
        "description": "读取最新训练产物，在独立时序留出集上执行确定性多窗口评测，评测完成后才返回轨迹。",
        "intent_aliases": ["run_policy_test", "policy_test", "simulate_policy", "策略测试", "运行策略测试", "训练后策略测试"],
        "keywords": ["策略测试", "训练后测试", "测试策略", "运行测试", "仿真策略", "先模拟", "策略仿真", "读取最新policy", "policy artifact"],
        "route": "/rl-panel?action=run_policy_test&from=xiaoyi",
        "button_selector": "#btnEvaluateTrain",
        "button_label": "留出集评测",
        "button_sequence": ["#btnEvaluateTrain"],
        "backend_request": {
            "method": "POST",
            "path": "/api/rl/train/{job_id}/evaluate",
            "body": {"episodes": 10},
        },
        "execution": {"type": "backend_policy_test_or_frontend_click", "dry_run_default": False},
        "requires_panel": True,
        "requires_human_confirm": False,
    },
    {
        "id": "verify_policy_for_online",
        "label": "验证策略能否上线",
        "category": "safety_dry_run",
        "description": "读取模型登记、校验哈希、留出集评测、多种子和守护栏证据；只给出上线门禁结果，不执行设备指令。",
        "intent_aliases": [
            "verify_policy_for_online",
            "verify_policy",
            "can_policy_go_live",
            "验证这个策略能不能上线",
            "验证策略能否上线",
            "验证最新策略能否上线",
            "验证策略上线",
        ],
        "keywords": [
            "验证这个策略能不能上线",
            "验证最新策略能否上线",
            "最新策略能否上线",
            "策略能否上线",
            "能不能上线",
            "上线验证",
            "验证策略",
            "只做演练验证",
            "不进入生产",
            "安全校验",
            "守护栏",
            "dry-run下发",
            "dry run下发",
            "能上线吗",
        ],
        "route": "/rl-panel?action=verify_policy_for_online&from=xiaoyi",
        "button_selector": "#btnVerifyDryRun",
        "button_label": "验证上线(dry-run)",
        "button_sequence": ["#btnLoad", "[data-simid]:first", "#btnVerifyDryRun"],
        "backend_request": {
            "method": "GET",
            "path": "/api/rl/models/{job_id}/readiness",
        },
        "execution": {"type": "model_registry_readiness_check", "dry_run_default": False},
        "requires_panel": True,
        "requires_human_confirm": True,
    },
    {
        "id": "start_xiaoyi_ai",
        "label": "启动小懿AI",
        "category": "assistant_linkage",
        "description": "启动桌面小懿AI本地服务，并让 /health 与 /api/chat 可用。",
        "intent_aliases": ["start_xiaoyi_ai", "start_xiaoyi", "启动小懿", "打开小懿AI", "启动小懿AI"],
        "keywords": ["启动小懿", "打开小懿", "启动小懿ai", "打开小懿ai", "拉起小懿", "小懿服务", "启动ai助手"],
        "route": "/integration-hub?action=start_xiaoyi_ai&from=xiaoyi",
        "button_selector": "#btnXiaoyiStart",
        "button_label": "启动小懿AI",
        "button_sequence": ["#btnXiaoyiStart"],
        "backend_request": {"method": "POST", "path": "/api/xiaoyi/launch", "body": {"confirm": True}},
        "execution": {"type": "local_assistant_launch", "dry_run_default": True},
        "linked_system": "xiaoyi_ai",
        "requires_panel": False,
        "requires_human_confirm": True,
    },
    {
        "id": "open_sailing_simulator",
        "label": "打开航行模拟器",
        "category": "desktop_linkage",
        "description": "启动由环境变量配置的 Godot 航行模拟器主场景。",
        "intent_aliases": [
            "open_sailing_simulator",
            "open_navigation_simulator",
            "打开航行模拟器",
            "启动航行模拟器",
            "打开模拟器",
            "启动模拟器",
            "打开Godot模拟器",
            "启动Godot模拟器",
            "打开船舶模拟器",
            "启动船舶模拟器",
        ],
        "keywords": [
            "打开航行模拟器",
            "启动航行模拟器",
            "打开航行",
            "启动航行",
            "航行模拟器",
            "打开模拟器",
            "启动模拟器",
            "模拟器启动",
            "godot航行",
            "打开godot",
            "启动godot",
            "打开godot模拟器",
            "启动godot模拟器",
            "船舶模拟器",
            "航行沙盘",
        ],
        "route": "/integration-hub?action=open_sailing_simulator&from=xiaoyi",
        "button_selector": "#btnSailingLaunch",
        "button_label": "打开 Godot 航行模拟器",
        "button_sequence": ["#btnSailingLaunch"],
        "backend_request": {"method": "POST", "path": "/api/sailing/launch", "body": {"preset": "main_scene", "confirm": True}},
        "execution": {"type": "desktop_launch", "dry_run_default": True},
        "linked_system": "sailing_simulator",
        "requires_panel": False,
        "requires_human_confirm": True,
    },
    {
        "id": "start_navigation_demo",
        "label": "启动航线演示",
        "category": "desktop_linkage",
        "description": "打开航行模拟器并加载航线演示预设；Godot 端无 HTTP 控制时先执行主场景预设加载。",
        "intent_aliases": ["start_navigation_demo", "navigation_demo", "开始航行演示", "启动航行演示", "启动航线演示", "开始航线演示", "演示航线", "启动路线演示"],
        "keywords": ["开始航行演示", "启动航行演示", "启动航线演示", "开始航线演示", "开始导航演示", "演示航线", "路线演示", "航行演示", "导航演示", "自动航行", "航线演示"],
        "route": "/integration-hub?action=start_navigation_demo&from=xiaoyi",
        "button_selector": "#btnSailingRouteDemo",
        "button_label": "启动航线演示",
        "button_sequence": ["#btnSailingRouteDemo"],
        "backend_request": {"method": "POST", "path": "/api/sailing/actions/execute", "body": {"action_id": "start_navigation_demo", "confirm": True}},
        "execution": {"type": "desktop_launch_with_demo_intent", "dry_run_default": True},
        "linked_system": "sailing_simulator",
        "requires_panel": False,
        "requires_human_confirm": True,
    },
    {
        "id": "switch_ship_view",
        "label": "切换船舶视角",
        "category": "desktop_linkage",
        "description": "切换 Godot 航行模拟器船队/受控船舶视角；当前先启动模拟器并标记视角切换动作。",
        "intent_aliases": ["switch_ship_view", "switch_sailing_view", "切换船舶视角", "切换视角", "换船", "切到船舶视角", "查看船舶视角"],
        "keywords": ["切换船舶视角", "切到船舶视角", "查看船舶视角", "切换驾驶视角", "跟随船舶", "跟随船", "切换视角", "换船", "切换船", "船舶视角", "受控船舶"],
        "route": "/integration-hub?action=switch_ship_view&from=xiaoyi",
        "button_selector": "#btnSailingSwitchView",
        "button_label": "切换船舶视角",
        "button_sequence": ["#btnSailingSwitchView"],
        "backend_request": {"method": "POST", "path": "/api/sailing/actions/execute", "body": {"action_id": "switch_ship_view", "confirm": True}},
        "execution": {"type": "desktop_launch_then_staged_control", "dry_run_default": True},
        "linked_system": "sailing_simulator",
        "requires_panel": False,
        "requires_human_confirm": True,
    },
    {
        "id": "run_sailing_rl_smoke_test",
        "label": "运行 RL 航行场景 smoke test",
        "category": "desktop_linkage",
        "description": "通过 Godot headless 执行航行模拟器 tools/ship_rl_smoke_test.gd。",
        "intent_aliases": ["run_sailing_rl_smoke_test", "sailing_smoke_test", "ship_rl_smoke_test", "运行RL航行场景smoketest", "运行航行smoketest", "运行模拟器测试"],
        "keywords": ["运行rl航行场景smoketest", "rl航行场景smoketest", "航行smoketest", "航行 smoke test", "ship rl smoke", "航行测试", "rl航行测试", "运行航行测试", "运行模拟器测试", "模拟器smoke", "航行场景测试", "rl场景测试", "烟雾测试"],
        "route": "/integration-hub?action=run_sailing_rl_smoke_test&from=xiaoyi",
        "button_selector": "#btnSailingSmoke",
        "button_label": "运行 smoke test",
        "button_sequence": ["#btnSailingSmoke"],
        "backend_request": {"method": "POST", "path": "/api/sailing/actions/execute", "body": {"action_id": "run_sailing_rl_smoke_test", "confirm": True}},
        "execution": {"type": "godot_headless_smoke_test", "dry_run_default": True},
        "linked_system": "sailing_simulator",
        "requires_panel": False,
        "requires_human_confirm": True,
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _public_action(action: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": action["id"],
        "label": action["label"],
        "category": action["category"],
        "description": action["description"],
        "intent_aliases": action["intent_aliases"],
        "keywords": action["keywords"],
        "route": action["route"],
        "button_selector": action.get("button_selector"),
        "button_label": action.get("button_label"),
        "button_sequence": action.get("button_sequence", []),
        "backend_request": action.get("backend_request"),
        "execution": action.get("execution", {}),
        "linked_system": action.get("linked_system"),
        "requires_panel": bool(action.get("requires_panel")),
        "requires_human_confirm": bool(action.get("requires_human_confirm")),
    }


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _score_action(action: Dict[str, Any], text: str, intent: str) -> Tuple[int, List[str]]:
    text_norm = _norm(text)
    intent_norm = _norm(intent)
    score = 0
    reasons: List[str] = []

    action_id_norm = _norm(action["id"])
    if intent_norm and intent_norm == action_id_norm:
        score += 120
        reasons.append("intent_exact_action_id")

    for alias in action.get("intent_aliases", []):
        alias_norm = _norm(alias)
        if not alias_norm:
            continue
        if intent_norm and intent_norm == alias_norm:
            score += 90
            reasons.append("intent_alias")
        if text_norm and alias_norm in text_norm:
            score += 55
            reasons.append("alias_in_instruction")

    for keyword in action.get("keywords", []):
        keyword_norm = _norm(keyword)
        if keyword_norm and keyword_norm in text_norm:
            score += 24 + min(len(keyword_norm), 10)
            reasons.append("keyword:" + keyword)

    if text_norm and action_id_norm in text_norm:
        score += 70
        reasons.append("action_id_in_instruction")

    return score, reasons


def resolve_action(instruction: str = "", intent: str = "") -> Dict[str, Any]:
    ranked: List[Dict[str, Any]] = []
    for action in RL_ACTIONS:
        score, reasons = _score_action(action, instruction, intent)
        if score <= 0:
            continue
        item = _public_action(action)
        item["score"] = score
        item["match_reasons"] = reasons
        ranked.append(item)

    ranked.sort(key=lambda item: (-int(item["score"]), item["id"]))
    if ranked:
        return {"matched": True, "action": ranked[0], "candidates": ranked[:4]}

    fallback = _public_action(next(action for action in RL_ACTIONS if action["id"] == "open_rl_panel"))
    fallback["score"] = 0
    fallback["match_reasons"] = ["fallback_open_rl_panel"]
    return {"matched": False, "action": fallback, "candidates": [fallback]}


def _absolute_url(request: Request, path: str) -> str:
    return str(request.base_url).rstrip("/") + path


def _request_json(method: str, url: str, body: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = UrlRequest(url, data=data, method=method.upper(), headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return {"ok": True, "status_code": int(getattr(resp, "status", 200)), "data": json.loads(raw) if raw else {}}
    except URLError as exc:
        return {"ok": False, "status_code": None, "error": str(getattr(exc, "reason", exc))}
    except Exception as exc:
        return {"ok": False, "status_code": None, "error": str(exc)}


def _execute_backend_action(request: Request, action: Dict[str, Any], payload: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    action_id = str(action["id"])
    if action_id == "start_xiaoyi_ai":
        return execute_xiaoyi_action(action_id, payload, dry_run=dry_run)
    if action_id in {"open_sailing_simulator", "start_navigation_demo", "switch_ship_view", "run_sailing_rl_smoke_test"}:
        return execute_sailing_action(action_id, payload, dry_run=dry_run)

    if action_id == "open_rl_panel" or action.get("category") == "xiaoyi_mission":
        route = str(action.get("route") or "/rl-panel")
        return {
            "type": "open_route",
            "status": "ready",
            "url": _absolute_url(request, route),
            "dry_run": dry_run,
            "production_action_executed": False,
        }

    backend_request = action.get("backend_request")
    if dry_run or not backend_request:
        return {
            "type": "frontend_click" if not backend_request else "backend_api",
            "status": "ready_to_execute",
            "dry_run": True,
            "route": action.get("route"),
            "button_selector": action.get("button_selector"),
            "button_sequence": action.get("button_sequence", []),
            "backend_request": backend_request,
        }

    if action_id == "view_rl_training_status":
        path = str(backend_request.get("path") or "/api/rl/train/status")
        if payload.get("job_id"):
            path += "?" + urlencode({"job_id": str(payload["job_id"])})
        result = _request_json(backend_request.get("method", "GET"), _absolute_url(request, path))
        return {"type": "backend_status_query", "status": "queried" if result.get("ok") else "failed", "result": result}

    if action_id == "verify_policy_for_online":
        training_result = _request_json("GET", _absolute_url(request, "/api/rl/train/status"))
        training_data = training_result.get("data") or {}
        training_status = training_data.get("status") if isinstance(training_data.get("status"), dict) else training_data
        training_state = str(training_status.get("status") or "IDLE").upper()
        job_id = str(payload.get("job_id") or training_status.get("job_id") or "").strip()
        if not job_id or training_state not in {"COMPLETED", "EVALUATED"}:
            return {
                "type": "model_registry_readiness_check",
                "status": "training_incomplete",
                "current_training": training_status,
                "production_deployment_approved": False,
                "production_boundary": "no model promotion or equipment command was executed",
            }
        readiness = _request_json("GET", _absolute_url(request, f"/api/rl/models/{job_id}/readiness"))
        model = _request_json("GET", _absolute_url(request, f"/api/rl/models/{job_id}"))
        readiness_data = readiness.get("data") or {}
        return {
            "type": "model_registry_readiness_check",
            "status": "promotion_gate_passed" if readiness.get("ok") and readiness_data.get("ready_for_champion_alias") else "blocked",
            "job_id": job_id,
            "training_preflight": training_result,
            "model": model,
            "readiness": readiness,
            "production_deployment_approved": False,
            "production_boundary": "this action reads evidence gates only; it cannot promote a model or execute equipment commands",
        }

    if action_id == "run_policy_test":
        artifact_status = _request_json("GET", _absolute_url(request, "/api/rl/train/status"))
        status_payload = artifact_status.get("data") or {}
        current = status_payload.get("status") if isinstance(status_payload.get("status"), dict) else status_payload
        job_id = str(payload.get("job_id") or current.get("job_id") or "").strip()
        if not job_id or str(current.get("status") or "").upper() not in {"COMPLETED", "EVALUATED"}:
            return {"type": "heldout_policy_evaluation", "status": "training_incomplete", "latest_policy_artifact": artifact_status}
        episodes = max(5, min(50, int(payload.get("episodes") or (backend_request.get("body") or {}).get("episodes") or 10)))
        result = _request_json("POST", _absolute_url(request, f"/api/rl/train/{job_id}/evaluate"), {"episodes": episodes})
        evaluation = result.get("data") or {}
        return {
            "type": "heldout_policy_evaluation",
            "status": "executed" if result.get("ok") else "failed",
            "job_id": job_id,
            "episodes": episodes,
            "latest_policy_artifact": artifact_status,
            "result": result,
            "test_metrics": {
                "metrics": evaluation.get("metrics"),
                "uncertainty": evaluation.get("uncertainty"),
                "render": evaluation.get("render"),
            },
        }

    if action_id == "start_rl_training":
        body = dict(backend_request.get("body") or {})
        overrides = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        if overrides:
            cfg = dict(DEFAULT_TRAIN_CONFIG)
            cfg.update(overrides)
            body["config"] = cfg
        result = _request_json(backend_request.get("method", "POST"), _absolute_url(request, backend_request.get("path", "/api/rl/train/start")), body)
        return {"type": "backend_api", "status": "executed" if result.get("ok") else "failed", "result": result}

    return {
        "type": "frontend_click",
        "status": "ready_to_execute",
        "dry_run": True,
        "route": action.get("route"),
        "button_selector": action.get("button_selector"),
        "button_sequence": action.get("button_sequence", []),
    }


def get_action_by_id(action_id: str) -> Optional[Dict[str, Any]]:
    return next((item for item in RL_ACTIONS if item["id"] == action_id), None)


def public_action(action: Dict[str, Any]) -> Dict[str, Any]:
    return _public_action(action)


def action_url(request: Request, route: str) -> str:
    return _absolute_url(request, route)


def execute_registered_action(request: Request, action: Dict[str, Any], payload: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    return _execute_backend_action(request, action, payload, dry_run=dry_run)


@router.get("/registry", summary="RL 联动动作注册表")
def action_registry() -> JSONResponse:
    return JSONResponse(
        {
            "updated_at": _utc_now(),
            "scope": "Xiaoyi mission navigation plus gated RL and desktop linkage",
            "count": len(RL_ACTIONS),
            "actions": [_public_action(action) for action in RL_ACTIONS],
        }
    )


@router.post("/resolve", summary="把小懿意图或自然语言指令映射为动作")
def action_resolve(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    instruction = str(payload.get("instruction") or payload.get("text") or payload.get("question") or "")
    xiaoyi = payload.get("xiaoyi") if isinstance(payload.get("xiaoyi"), dict) else {}
    intent = str(payload.get("intent") or payload.get("xiaoyi_intent") or xiaoyi.get("intent") or "")
    result = resolve_action(instruction=instruction, intent=intent)
    return JSONResponse(
        {
            "updated_at": _utc_now(),
            "instruction": instruction,
            "intent": intent,
            **result,
        }
    )


@router.post("/execute", summary="执行或预演 RL 联动动作")
def action_execute(request: Request, payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    action_id = str(payload.get("action_id") or "").strip()
    if not action_id:
        resolved = resolve_action(
            instruction=str(payload.get("instruction") or payload.get("text") or ""),
            intent=str(payload.get("intent") or payload.get("xiaoyi_intent") or ""),
        )
        action_id = str((resolved.get("action") or {}).get("id") or "")

    action = get_action_by_id(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"未知动作：{action_id}")

    dry_run = bool(payload.get("dry_run", (action.get("execution") or {}).get("dry_run_default", True)))
    public = _public_action(action)
    execution = _execute_backend_action(request, action, payload, dry_run=dry_run)
    return JSONResponse(
        {
            "updated_at": _utc_now(),
            "ok": execution.get("status") not in {"failed"},
            "dry_run": dry_run,
            "action": public,
            "execution": execution,
            "ui_command": {
                "open_url": _absolute_url(request, str(public.get("route") or "/rl-panel")),
                "button_selector": public.get("button_selector"),
                "button_sequence": public.get("button_sequence", []),
                "button_label": public.get("button_label"),
            },
        }
    )


@router.post("/from-xiaoyi", summary="接收小懿识别结果并生成可执行动作")
def action_from_xiaoyi(request: Request, payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    xiaoyi = payload.get("xiaoyi") if isinstance(payload.get("xiaoyi"), dict) else {}
    instruction = str(payload.get("instruction") or payload.get("text") or payload.get("question") or xiaoyi.get("question") or xiaoyi.get("answer") or "")
    intent = str(payload.get("intent") or payload.get("xiaoyi_intent") or xiaoyi.get("intent") or "")
    resolved = resolve_action(instruction=instruction, intent=intent)
    action = next(item for item in RL_ACTIONS if item["id"] == (resolved.get("action") or {}).get("id"))
    execute_payload = {
        **payload,
        "action_id": (resolved.get("action") or {}).get("id"),
        "dry_run": payload.get("dry_run", (action.get("execution") or {}).get("dry_run_default", True)),
    }
    execution = _execute_backend_action(request, action, execute_payload, dry_run=bool(execute_payload["dry_run"]))
    return JSONResponse(
        {
            "updated_at": _utc_now(),
            "instruction": instruction,
            "intent": intent,
            "matched": resolved["matched"],
            "action": resolved["action"],
            "candidates": resolved["candidates"],
            "execution": execution,
            "ui_command": {
                "open_url": _absolute_url(request, str(resolved["action"].get("route") or "/rl-panel")),
                "button_selector": resolved["action"].get("button_selector"),
                "button_sequence": resolved["action"].get("button_sequence", []),
                "button_label": resolved["action"].get("button_label"),
            },
        }
    )
