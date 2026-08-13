"""app.services.copilot.api
--------------------------------
Ops Copilot 后端接口。

- 暴露 /api/copilot/ask GET 接口，供前端 “Ops Copilot（知识 + 对接）” 面板使用。
- 当前实现：优先调用桌面“小懿AI”服务；不可达时回退到本地 JSON 知识库。
- 对前端保持原有协议，避免影响页面结构和下游执行闭环。
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import JSONResponse
import json

try:
    import httpx
except Exception:  # pragma: no cover - optional adapter dependency
    httpx = None  # type: ignore

# 注意：在 app/server.py 中已经通过
#   app.include_router(copilot_router, prefix="/api/copilot", tags=["copilot"])
# 注册了路由，所以这里不要再加前缀。
router = APIRouter()

# 本地知识库文件路径：app/services/copilot/data/knowledge_base.json
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "knowledge_base.json"


def _load_knowledge_items() -> List[Dict[str, Any]]:
    """从本地 JSON 文件读取知识条目列表。

    JSON 允许两种结构：
    1) { "items": [ ... ] }
    2) [ ... ]
    """
    try:
        with DATA_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        # 没有数据文件时，返回空列表；前端会显示“未命中知识项”
        return []
    except json.JSONDecodeError:
        # JSON 解析失败时也返回空列表，避免打断接口
        return []

    if isinstance(raw, dict):
        items = raw.get("items", [])
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    # 仅保留字典类型条目，避免脏数据
    return [x for x in items if isinstance(x, dict)]


def _normalize_scope_values(scopes: Iterable[str]) -> List[str]:
    return [str(s).strip().lower() for s in scopes if s is not None]


def _score_item(query: str, item: Dict[str, Any]) -> float:
    """非常轻量级的打分逻辑：适合 demo / 单机部署。

    规则：
    - query 直接出现在 title/snippet/keywords 中：加较高权重
    - query 按空格 / / 分词后，每个 token 匹配到时加分
    """
    q = (query or "").strip().lower()
    if not q:
        return 0.0

    title = str(item.get("title", "")).lower()
    snippet = str(item.get("snippet", "")).lower()
    keywords = " ".join(str(k) for k in item.get("keywords", [])).lower()
    haystack = " ".join([title, snippet, keywords])

    score = 0.0

    # 整体包含
    if q in haystack:
        score += 5.0

    # 简单分词：空格 + 斜杠
    for token in q.replace("／", " ").replace("/", " ").split():
        if not token:
            continue
        if token in haystack:
            score += 2.0 + min(len(token), 8) * 0.2

    return score


def _rank_items(query: str, scope: str = "all", top_k: int = 8) -> List[Dict[str, Any]]:
    items = _load_knowledge_items()
    if not items:
        return []

    scope_norm = scope.strip().lower()
    if scope_norm and scope_norm != "all":
        filtered: List[Dict[str, Any]] = []
        for it in items:
            type_norm = str(it.get("type", "")).strip().lower()
            scopes_norm = _normalize_scope_values(it.get("scopes", []))
            if scope_norm == type_norm or scope_norm in scopes_norm:
                filtered.append(it)
        items = filtered or items

    scored: List[Dict[str, Any]] = []
    for it in items:
        s = _score_item(query, it)
        if s <= 0.0:
            continue
        it_copy = dict(it)
        it_copy["_score"] = float(s)
        scored.append(it_copy)

    if scored:
        scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return scored[:top_k]
    return items[:top_k]


def _public_item(it: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": it.get("id") or "",
        "type": it.get("type") or "item",
        "title": it.get("title") or "",
        "snippet": it.get("snippet") or "",
        "link": it.get("link") or "#",
        "score": round(float(it.get("_score", 0.0)), 3),
        "keywords": it.get("keywords") or [],
    }


def _intent_from(query: str, scope: str) -> str:
    q = (query or "").lower()
    scope_norm = (scope or "").lower()
    if scope_norm == "alert" or any(k in q for k in ["告警", "报警", "异常", "超标", "风险", "预警", "alarm", "alert"]):
        return "alert_explain"
    if scope_norm == "sop" or any(k in q for k in ["sop", "流程", "怎么处理", "处置", "步骤"]):
        return "sop_draft"
    if scope_norm == "compliance" or any(k in q for k in ["合规", "审计", "碳", "esg", "报表"]):
        return "compliance_answer"
    if scope_norm == "device" or any(k in q for k in ["设备", "岸桥", "场桥", "冷站", "bess", "岸电", "agv"]):
        return "device_diagnosis"
    return "ops_answer"


def _risk_from(severity: str, evidence_count: int) -> str:
    sev = (severity or "").lower()
    if sev in {"critical", "high", "red"}:
        return "高"
    if sev in {"major", "medium", "orange"}:
        return "中高"
    if evidence_count <= 1:
        return "中"
    return "中低"


def _steps_for(intent: str, query: str, scope: str) -> List[Dict[str, Any]]:
    base = [
        {"step": "确认现场上下文", "owner": "值班调度", "eta_min": 2, "detail": "核对设备、时窗、告警等级、当前负荷和是否已有执行工单。"},
        {"step": "拉取证据与相似案例", "owner": "Ops Copilot", "eta_min": 1, "detail": "从 SOP、告警、设备、协议和合规知识库召回相关条目。"},
    ]
    if intent == "alert_explain":
        base += [
            {"step": "解释告警成因", "owner": "能源/设备工程师", "eta_min": 5, "detail": "区分真实越限、传感器漂移、策略切换、外部作业扰动四类原因。"},
            {"step": "执行分级处置", "owner": "值班长", "eta_min": 10, "detail": "先执行无损动作，再进入限功率、切负荷或人工审批。"},
        ]
    elif intent == "sop_draft":
        base += [
            {"step": "生成 SOP 草案", "owner": "Ops Copilot", "eta_min": 3, "detail": "输出目标、适用范围、触发条件、操作步骤、回滚条件和记录要求。"},
            {"step": "人工确认与发布", "owner": "运行经理", "eta_min": 15, "detail": "确认不影响安全、吞吐和合规边界后再下发。"},
        ]
    elif intent == "compliance_answer":
        base += [
            {"step": "对齐合规口径", "owner": "ESG / Audit", "eta_min": 6, "detail": "确认 Scope、碳因子、分摊边界和审计证据链。"},
            {"step": "生成审计备注", "owner": "Ops Copilot", "eta_min": 2, "detail": "输出可复制的审计说明和证据引用。"},
        ]
    else:
        base += [
            {"step": "形成处置建议", "owner": "Ops Copilot", "eta_min": 4, "detail": "给出风险、建议动作、依赖模块和回看指标。"},
            {"step": "转交执行闭环", "owner": "Execution / OpsX", "eta_min": 8, "detail": "需要动作时进入 dry-run、审批、执行、回执和审计链路。"},
        ]
    base.append({"step": "复盘与沉淀", "owner": "MLOps / 知识库", "eta_min": 30, "detail": "把最终结论、失败动作和有效处置沉淀到 playbook。"})
    return base


_PLAYBOOKS: List[Dict[str, Any]] = [
    {"id": "peak_demand", "title": "合同需量越峰解释", "scope": "alert", "severity": "major", "query": "未来 15 分钟需量峰值抬升，应该怎么解释并处置？"},
    {"id": "shore_harmonics", "title": "岸电谐波告警处置", "scope": "alert", "severity": "critical", "query": "岸电 THDi 超标告警出现，先检查哪些原因？"},
    {"id": "hvac_setpoint", "title": "冷站供水温度建议", "scope": "sop", "severity": "medium", "query": "未来 60 分钟建议下调供水温度 0.3°C，应该如何生成 SOP？"},
    {"id": "typhoon_mode", "title": "台风模式启停", "scope": "sop", "severity": "critical", "query": "台风红色预警下港区需要启动哪些安全流程？"},
    {"id": "tos_degradation", "title": "TOS 性能下降", "scope": "alert", "severity": "major", "query": "TOS 响应变慢且消息队列堆积，如何解释并降级运行？"},
    {"id": "carbon_audit", "title": "碳盘查审计说明", "scope": "compliance", "severity": "medium", "query": "港区年度碳排放盘查需要保留哪些审计证据？"},
]


def _xiaoyi_config() -> Dict[str, Any]:
    base_url = (os.getenv("XIAOYI_AI_BASE_URL") or "http://127.0.0.1:8010").rstrip("/")
    return {
        "engine": "小懿AI",
        "provider": "desktop-xiaoyi",
        "base_url": base_url,
        "health_url": base_url + "/health",
        "chat_url": base_url + "/api/chat",
        "configured": httpx is not None,
        "model": "XiaoyiAI local RAG",
        "fallback": "Ops Copilot local knowledge base",
        "httpx_available": httpx is not None,
    }


def _xiaoyi_mode(scope: str, mode: str, query: str) -> str:
    q = (query or "").lower()
    scope_norm = (scope or "").lower()
    mode_norm = (mode or "").lower()
    if scope_norm == "sop" or "sop" in mode_norm or any(k in q for k in ["sop", "步骤", "处置", "怎么处理", "应急"]):
        return "sop"
    if scope_norm == "alert" or any(k in q for k in ["告警", "异常", "超标", "风险", "故障"]):
        return "sop"
    if scope_norm == "compliance" or mode_norm in {"audit_note", "handoff"}:
        return "ops"
    if mode_norm in {"brief", "briefing"}:
        return "brief"
    return "ops"


def _probe_xiaoyi_status() -> Dict[str, Any]:
    cfg = _xiaoyi_config()
    if httpx is None:
        return {**cfg, "configured": False, "online": False, "status": "missing_httpx", "reason": "httpx 不可用，无法连接小懿AI。"}
    try:
        with httpx.Client(timeout=2.0) as client:
            health_response = client.get(cfg["health_url"])
            health_response.raise_for_status()
            health = health_response.json()
            openapi_response = client.get(cfg["base_url"] + "/openapi.json")
            openapi_response.raise_for_status()
            openapi = openapi_response.json()
        paths = openapi.get("paths") if isinstance(openapi, dict) else {}
        chat_route = paths.get("/api/chat") if isinstance(paths, dict) else None
        chat_capable = isinstance(chat_route, dict) and "post" in chat_route
        if not chat_capable:
            return {
                **cfg,
                "configured": True,
                "online": False,
                "chat_capable": False,
                "status": "health_only_not_xiaoyi",
                "health": health,
                "reason": "目标地址健康检查可达，但没有POST /api/chat；不得标记为小懿在线。",
            }
        return {
            **cfg,
            "configured": True,
            "online": True,
            "chat_capable": True,
            "status": "ready",
            "health": health,
            "identity_check": "health_plus_openapi_chat_route",
        }
    except Exception as exc:
        return {
            **cfg,
            "configured": True,
            "online": False,
            "chat_capable": False,
            "status": "offline",
            "reason": f"小懿AI 未在线或不可达：{str(exc)[:180]}",
        }


def _call_xiaoyi(
    *,
    query: str,
    scope: str,
    mode: str,
    top_k: int,
) -> Dict[str, Any]:
    cfg = _xiaoyi_config()
    if httpx is None:
        return {"ok": False, "status": "fallback", "reason": "httpx 不可用，无法调用小懿AI。", "config": cfg}

    body = {
        "question": query,
        "mode": _xiaoyi_mode(scope=scope, mode=mode, query=query),
        "top_k": max(1, min(int(top_k or 5), 10)),
    }
    started = time.perf_counter()
    timeout_seconds = max(5.0, min(180.0, float(os.getenv("XIAOYI_AI_TIMEOUT_SECONDS") or 65.0)))
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(cfg["chat_url"], json=body)
            resp.raise_for_status()
            data = resp.json()
        answer = str(data.get("answer") or "") if isinstance(data, dict) else ""
        if not answer.strip():
            raise ValueError("小懿响应缺少完整answer字段")
        return {
            "ok": True,
            "status": "xiaoyi",
            "engine_execution": "external_xiaoyi_chat_api",
            "config": cfg,
            "request": body,
            "parsed": data,
            "usage": data.get("usage") if isinstance(data, dict) else {},
            "generation": {
                "provider": data.get("generation_provider"),
                "model": data.get("generation_model"),
                "fallback": data.get("generation_fallback"),
                "grounded": data.get("grounded"),
                "completion_status": data.get("completion_status"),
            } if isinstance(data, dict) else {},
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "response_validated": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "fallback",
            "engine_execution": "local_evidence_fallback",
            "reason": f"小懿AI 调用失败，已退回本地 Copilot：{str(exc)[:220]}",
            "config": cfg,
            "request": body,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "response_validated": False,
        }


def _public_xiaoyi_evidence(raw: Dict[str, Any]) -> Dict[str, Any]:
    source = str(raw.get("source") or "xiaoyi")
    return {
        "id": raw.get("id") or "",
        "type": "xiaoyi",
        "title": raw.get("title") or source,
        "snippet": raw.get("snippet") or "",
        "link": "#",
        "score": round(float(raw.get("score", 0.0) or 0.0), 3),
        "keywords": [source],
        "source": source,
    }


def _xiaoyi_evidence_items(parsed: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    evidence = parsed.get("evidence") if isinstance(parsed, dict) else []
    if not isinstance(evidence, list):
        return []
    return [_public_xiaoyi_evidence(x) for x in evidence if isinstance(x, dict)][:limit]


def _first_answer_line(answer: str) -> str:
    for line in (answer or "").splitlines():
        text = line.strip()
        if text:
            return text[:180]
    return "小懿AI 已生成港航运营答案。"


def _steps_from_xiaoyi_answer(answer: str, intent: str, query: str, scope: str) -> List[Dict[str, Any]]:
    lines = [x.strip() for x in (answer or "").splitlines()]
    steps: List[Dict[str, Any]] = []
    in_steps = False
    owners = ["值班调度", "设备/能源工程师", "能源/安全负责人", "值班长", "Ops Copilot", "审计/合规"]

    for line in lines:
        if "处置步骤" in line or "Response steps" in line:
            in_steps = True
            continue
        if in_steps and (line.startswith("人工确认") or line.startswith("Human confirmation")):
            break
        if not in_steps:
            continue

        text = line
        for marker in [". ", "、", "."]:
            head, sep, tail = text.partition(marker)
            if sep and head.strip().isdigit():
                text = tail.strip()
                break
        if not text or len(text) < 4:
            continue
        steps.append({
            "step": text[:26],
            "owner": owners[min(len(steps), len(owners) - 1)],
            "eta_min": 2 + min(len(steps) * 3, 18),
            "detail": text,
        })
        if len(steps) >= 8:
            break

    return steps or _steps_for(intent, query, scope)


def _actions_from_xiaoyi(primary_title: str) -> List[Dict[str, Any]]:
    title = primary_title or "小懿证据"
    return [
        {"priority": "P0", "action": "按小懿建议先确认告警对象、时窗和风险等级", "owner": "值班长", "guardrail": "人工确认", "handoff": "OpsX"},
        {"priority": "P1", "action": f"核对小懿命中证据：{title}", "owner": "设备/能源工程师", "guardrail": "证据不足不执行", "handoff": "Twin / Monitoring"},
        {"priority": "P2", "action": "把小懿问答结果写入审计包并进入 dry-run 承接", "owner": "Ops Copilot", "guardrail": "仅建议，不直接生产下发", "handoff": "Execution"},
    ]


def _mission_prompt(query: str, context: Dict[str, Any], mission_id: str) -> str:
    source = context.get("source") or {}
    forecast = context.get("forecast") or {}
    monitoring = context.get("monitoring") or {}
    policy = context.get("policy") or {}
    business = context.get("business_value") or {}
    compact = {
        "context_sha256": context.get("context_sha256"),
        "state": context.get("overall_state"),
        "source": {
            "mode": source.get("mode"),
            "artifact": source.get("artifact_id"),
            "sha256": source.get("sha256"),
            "samples": source.get("sample_count"),
            "measured": source.get("measured"),
            "production": source.get("production"),
        },
        "data_quality": context.get("data_quality"),
        "forecast": {
            "model": forecast.get("model"),
            "horizon_min": forecast.get("horizon_min"),
            "p10_kw": forecast.get("peak_p10_kw"),
            "p50_kw": forecast.get("peak_p50_kw"),
            "p90_kw": forecast.get("peak_p90_kw"),
            "cap_kw": forecast.get("engineering_cap_kw"),
            "exceedance_probability": forecast.get("peak_probability"),
            "site_calibrated": forecast.get("site_calibration_available"),
        },
        "monitoring": monitoring,
        "policy": {
            "algorithm": policy.get("algorithm"),
            "implementation": policy.get("implementation"),
            "model_sha256": policy.get("model_sha256"),
            "dataset_id": policy.get("dataset_id"),
            "dataset_sha256": policy.get("dataset_sha256"),
            "hard_guardrail_passed": policy.get("hard_guardrail_passed"),
            "production_authority": policy.get("production_authority"),
        },
        "business_projection": {
            "avoided_energy_cost_cny": business.get("avoided_energy_cost_cny"),
            "avoided_carbon_kg": business.get("avoided_carbon_kg"),
            "financial_audit_ready": business.get("financial_audit_ready"),
        },
        "missing_site_factors": context.get("missing_site_factors"),
    }
    prompt = (
        "你是小懿AI港口一线副驾。仅根据下列后端上下文；不得补造现场数据、事故、工单、收益或执行结果。"
        "明确区分公开回放、预测投影和现场实测。准入门阻断时先保持上一稳定策略或FCFS/MPC基线。"
        "回答：结论、依据、检查项、dry-run、人工确认。"
        f"\n任务:{mission_id}\n问题:{query[:240]}\n上下文:"
        + json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    # Xiaoyi's public request contract caps the question at 2,000 chars.  The
    # packet is intentionally compact, but never send an invalid oversized
    # request if a future backend adds fields.
    return prompt if len(prompt) <= 1980 else prompt[:1978] + "…"


def _context_grounding(answer: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether Xiaoyi actually used the supplied runtime packet.

    A successful HTTP response proves invocation, not grounding.  We only use
    the generated answer as the frontline answer if at least two independent
    runtime anchors are repeated; otherwise the deterministic evidence-bound
    composer takes over while the raw call remains visible in the audit packet.
    """
    monitoring = context.get("monitoring") or {}
    policy = context.get("policy") or {}
    forecast = context.get("forecast") or {}
    anchors = [
        str(context.get("context_sha256") or "")[:12],
        str(monitoring.get("admission_decision") or ""),
        str(policy.get("algorithm") or "").upper(),
        f"{float(monitoring.get('drift_psi')):.3f}" if monitoring.get("drift_psi") is not None else "",
        f"{float(forecast.get('peak_p50_kw')):,.0f}" if forecast.get("peak_p50_kw") is not None else "",
    ]
    answer_upper = answer.upper()
    matched = [anchor for anchor in anchors if anchor and anchor.upper() in answer_upper]
    prompt_echo_detected = "上下文:{" in answer or "上下文：{" in answer
    return {
        "passed": len(matched) >= 2 and not prompt_echo_detected,
        "matched_anchors": matched,
        "required_anchor_count": 2,
        "prompt_echo_detected": prompt_echo_detected,
    }


@router.get(
    "/ask",
    summary="Ops Copilot：小懿知识检索",
    description="优先调用桌面小懿AI返回证据；不可达时回退到本地知识库。",
)
def copilot_ask(
    query: str = Query(..., min_length=1, description="用户提问内容，可为中文或英文"),
    scope: str = Query(
        "all",
        description="知识类别：all / sop / alert / device / protocol / compliance 等",
    ),
    top_k: int = Query(8, ge=1, le=32, description="返回的最大条目数量"),
) -> JSONResponse:
    xiaoyi_result = _call_xiaoyi(query=query, scope=scope, mode="brief", top_k=top_k)
    if xiaoyi_result.get("ok"):
        parsed = xiaoyi_result.get("parsed") or {}
        resp_items = _xiaoyi_evidence_items(parsed, limit=top_k)
        return JSONResponse(
            {
                "items": resp_items,
                "count": len(resp_items),
                "scope": scope,
                "engine": "xiaoyi_ai",
                "confidence": parsed.get("confidence"),
                "fallback": False,
            }
        )

    selected = _rank_items(query=query, scope=scope, top_k=top_k)
    resp_items = [_public_item(it) for it in selected]
    return JSONResponse(
        {
            "items": resp_items,
            "count": len(resp_items),
            "scope": scope,
            "engine": "local_copilot_fallback",
            "fallback": True,
            "reason": xiaoyi_result.get("reason"),
        }
    )


@router.get(
    "/context",
    summary="Ops Copilot：运行上下文",
    description="给独立 Ops Copilot 工作台提供演示级上下文、连接器和待处置摘要。",
)
def copilot_context(
    request: Request,
    port: str = Query("CNSHA", description="港口代码"),
    asset_id: str = Query("qc-01", description="当前资产"),
    mission: str = Query("situation", description="小懿任务模式"),
) -> JSONResponse:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    xiaoyi_status = _probe_xiaoyi_status()
    assistant_status = "ready" if xiaoyi_status.get("online") else "fallback"
    mission_control = getattr(request.app.state, "xiaoyi_mission_control", None)
    if mission_control is None:
        return JSONResponse(
            {
                "port": port,
                "status": "unavailable",
                "updated_at": now,
                "signals": [],
                "connectors": [],
                "mission_modes": [],
                "reason": "xiaoyi mission context service is not registered",
                "production_authority": False,
            }
        )
    context = mission_control.build_context(
        asset_id=asset_id,
        mission_id=mission,
    )
    overall = str(context.get("overall_state") or "unavailable")
    if overall in {"data_unavailable", "unavailable"}:
        status = "unavailable"
    elif "review" in overall or "risk" in overall:
        status = "review"
    else:
        status = "ready"
    return JSONResponse(
        {
            "port": port,
            "asset_id": asset_id,
            "mission": mission,
            "status": status,
            "updated_at": now,
            "overall_state": overall,
            "context_sha256": context.get("context_sha256"),
            "signals": context.get("signals") or [],
            "connectors": [
                {
                    "name": "小懿AI生成服务",
                    "endpoint": xiaoyi_status.get("chat_url"),
                    "status": assistant_status,
                    "chat_capable": bool(xiaoyi_status.get("chat_capable")),
                    "reason": xiaoyi_status.get("reason"),
                },
                {"name": "孪生态势上下文", "endpoint": "/api/copilot/context", "status": "ready"},
                {"name": "小懿任务推理", "endpoint": "/api/copilot/mission", "status": "ready"},
                {"name": "交接班留痕", "endpoint": "/api/copilot/handoff", "status": "confirm-required"},
                {"name": "OpsX审计", "endpoint": "/api/opsx/*", "status": "ready"},
            ],
            "mission_modes": mission_control.mission_modes(),
            "context": context,
            "handoff_links": [
                {"label": "策略执行", "href": "/#strategy-exec-module"},
                {"label": "OpsX 审计", "href": "/#opsx-section"},
                {"label": "实时孪生", "href": "/#twin3d-section"},
            ],
            "production_authority": False,
        }
    )


@router.get(
    "/playbooks",
    summary="Ops Copilot：常用问题与处置 playbook",
)
def copilot_playbooks() -> JSONResponse:
    return JSONResponse({"items": _PLAYBOOKS, "count": len(_PLAYBOOKS)})


@router.get(
    "/llm/status",
    summary="Ops Copilot：小懿AI 状态",
)
def copilot_llm_status() -> JSONResponse:
    cfg = _probe_xiaoyi_status()
    return JSONResponse(
        {
            **cfg,
            "active_mode": "xiaoyi_ai" if cfg.get("online") else "local_copilot_fallback",
            "env_hint": {
                "base_url": "XIAOYI_AI_BASE_URL，默认 http://127.0.0.1:8010",
                "run": "设置 PORT_DT_ENABLE_DESKTOP_INTEGRATIONS=1、XIAOYI_AI_PROJECT 和 XIAOYI_AI_START_COMMAND",
            },
        }
    )


_MISSION_DEFAULT_QUESTIONS = {
    "situation": "小懿，请基于当前数字孪生后端状态给本班一个简明态势判断。",
    "forecast": "小懿，请解释当前预测峰值、区间和工程阈值风险，并指出缺失的现场证据。",
    "strategy": "小懿，请解释当前策略为什么这样决策、相对基线改善什么、风险和边界是什么。",
    "triage": "小懿，请根据当前异常、漂移和策略准入门给出一线分诊与回退顺序。",
    "handoff": "小懿，请把当前态势、模型版本、未决风险和下一班检查项整理成交接摘要。",
    "dry_run": "小懿，请判断当前是否适合进入策略预演；只给dry-run与人工审批步骤。",
}


@router.post(
    "/mission",
    summary="小懿数字孪生任务推理",
    description="先读取后端孪生/预测/监控/策略上下文，再真实调用小懿；不可达时使用同一上下文做本地证据兜底。",
)
def copilot_mission(
    request: Request,
    payload: Dict[str, Any] = Body(default={}),
) -> JSONResponse:
    mission_control = getattr(request.app.state, "xiaoyi_mission_control", None)
    if mission_control is None:
        return JSONResponse(
            {
                "ok": False,
                "status": "mission_context_unavailable",
                "production_authority": False,
            },
            status_code=503,
        )

    mission_id = str(payload.get("mission") or "situation").strip().lower()
    allowed = {row["id"] for row in mission_control.mission_modes()}
    if mission_id not in allowed:
        mission_id = "situation"
    query = str(payload.get("query") or "").strip() or _MISSION_DEFAULT_QUESTIONS[mission_id]
    asset_id = str(payload.get("asset_id") or payload.get("asset_group") or "qc-01")
    if asset_id in {"all_port", "all", "port"}:
        asset_id = "qc-01"
    cap_kw = float(payload.get("cap_kw") or 36_000.0)
    horizon_min = max(15, min(360, int(payload.get("horizon_min") or 60)))
    context = mission_control.build_context(
        asset_id=asset_id,
        mission_id=mission_id,
        cap_kw=cap_kw,
        horizon_min=horizon_min,
        step_min=5,
    )

    engine = str(payload.get("engine") or "xiaoyi_ai")
    uses_xiaoyi = engine in {"xiaoyi_ai", "auto", "external_llm"}
    llm_result: Dict[str, Any] = {
        "ok": False,
        "status": "local_evidence_fallback",
        "engine_execution": "local_evidence_fallback",
        "reason": "调用方选择本地证据模式。",
        "config": _xiaoyi_config(),
        "latency_ms": 0.0,
    }
    if uses_xiaoyi:
        llm_result = _call_xiaoyi(
            query=_mission_prompt(query, context, mission_id),
            scope="alert" if mission_id == "triage" else "all",
            mode="handoff" if mission_id == "handoff" else "ops",
            top_k=max(1, min(10, int(payload.get("top_k") or 6))),
        )
    parsed = llm_result.get("parsed") if llm_result.get("ok") else {}
    generated_answer = str((parsed or {}).get("answer") or "").strip()
    grounding = _context_grounding(generated_answer, context) if generated_answer else {
        "passed": False,
        "matched_anchors": [],
        "required_anchor_count": 2,
    }
    answer = generated_answer if grounding.get("passed") else ""
    if not answer:
        answer = mission_control.local_fallback_answer(context, query)
    answer_source = "xiaoyi_context_grounded" if grounding.get("passed") else "backend_evidence_guardrail"

    actions = mission_control.recommended_actions(context, mission_id)
    action_cards = [
        {
            **row,
            "action": row.get("label"),
            "owner": "值班调度 / 专业工程师",
            "guardrail": "人工确认" if row.get("human_confirmation") else "只读导航",
            "handoff": row.get("href"),
        }
        for row in actions
    ]
    sop_steps = [
        {
            "step": str(row.get("label")),
            "owner": "值班调度" if index == 0 else "专业工程师",
            "eta_min": 2 + index * 3,
            "detail": str(row.get("reason") or ""),
            "status": "blocked" if str(row.get("execution") or "").startswith("blocked") else "ready",
        }
        for index, row in enumerate(actions[:5])
    ]
    system_evidence = [
        {
            "id": str(row.get("id") or "runtime"),
            "type": "runtime",
            "title": str(row.get("name") or "运行上下文"),
            "snippet": f"{row.get('value')} · {row.get('source')}",
            "link": "#",
            "score": 1.0,
            "source": row.get("source"),
        }
        for row in context.get("signals") or []
    ]
    xiaoyi_evidence = _xiaoyi_evidence_items(parsed or {}, limit=6) if llm_result.get("ok") else []
    evidence = system_evidence + xiaoyi_evidence
    risk = "高" if (context.get("monitoring") or {}).get("new_policy_suggestions_allowed") is False else "中低"
    headline = _first_answer_line(answer)
    invocation_id = "xy-mission-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    handoff_preview = mission_control.handoff_preview(
        context,
        answer=answer,
        operator=str(payload.get("operator") or ""),
        shift=str(payload.get("shift") or ""),
    )
    audit_packet = {
        "invocation_id": invocation_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mission": mission_id,
        "query": query,
        "context_sha256": context.get("context_sha256"),
        "engine_requested": engine,
        "engine_executed": llm_result.get("engine_execution"),
        "generation": llm_result.get("generation") or {},
        "xiaoyi_called": bool(llm_result.get("ok")),
        "answer_source": answer_source,
        "context_grounding": grounding,
        "latency_ms": llm_result.get("latency_ms"),
        "evidence_count": len(evidence),
        "production_authority": False,
        "human_in_loop": True,
    }
    return JSONResponse(
        {
            "ok": True,
            "status": (
                "xiaoyi_answer"
                if grounding.get("passed")
                else ("xiaoyi_called_guarded_fallback" if llm_result.get("ok") else "local_evidence_fallback")
            ),
            "summary": {
                "headline": headline,
                "risk_level": risk,
                "intent": mission_id,
                "primary_reference": "hash-addressed runtime context",
                "operator_note": answer,
            },
            "actions": action_cards,
            "sop_steps": sop_steps,
            "evidence": evidence,
            "llm": {
                "engine": "小懿AI" if grounding.get("passed") else "小懿调用 + 后端证据护栏",
                "status": llm_result.get("status"),
                "ok": bool(llm_result.get("ok")),
                "true_xiaoyi_called": bool(llm_result.get("ok")),
                "engine_execution": llm_result.get("engine_execution"),
                "reason": llm_result.get("reason"),
                "latency_ms": llm_result.get("latency_ms"),
                "response_validated": bool(llm_result.get("response_validated")),
                "answer_source": answer_source,
                "context_grounding": grounding,
                "chat_url": (llm_result.get("config") or {}).get("chat_url"),
                "usage": llm_result.get("usage") or {},
                "generation": llm_result.get("generation") or {},
            },
            "xiaoyi": {
                "ok": bool(llm_result.get("ok")),
                "answer": generated_answer if llm_result.get("ok") else "",
                "used_as_frontline_answer": bool(grounding.get("passed")),
                "confidence": (parsed or {}).get("confidence"),
                "next_questions": (parsed or {}).get("next_questions") or [],
            },
            "context": context,
            "context_sha256": context.get("context_sha256"),
            "mission_modes": mission_control.mission_modes(),
            "handoff_preview": handoff_preview,
            "audit_packet": audit_packet,
            "production_authority": False,
        }
    )


@router.post(
    "/handoff",
    summary="小懿交接班预览与确认留痕",
    description="默认只返回预览；confirm=true才追加写入运行时审计日志，始终不执行生产动作。",
)
def copilot_handoff(
    request: Request,
    payload: Dict[str, Any] = Body(default={}),
) -> JSONResponse:
    mission_control = getattr(request.app.state, "xiaoyi_mission_control", None)
    if mission_control is None:
        return JSONResponse({"ok": False, "status": "mission_context_unavailable"}, status_code=503)
    context = mission_control.build_context(
        asset_id=str(payload.get("asset_id") or "qc-01"),
        mission_id="handoff",
    )
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        answer = mission_control.local_fallback_answer(context, _MISSION_DEFAULT_QUESTIONS["handoff"])
    packet = mission_control.handoff_preview(
        context,
        answer=answer,
        operator=str(payload.get("operator") or ""),
        shift=str(payload.get("shift") or ""),
    )
    result = mission_control.persist_handoff(packet, confirm=bool(payload.get("confirm")))
    return JSONResponse(
        {
            "ok": True,
            **result,
            "human_confirmation_required": True,
            "production_action_executed": False,
        }
    )


@router.post(
    "/brief",
    summary="Ops Copilot：生成解释、SOP 与审计包",
    description="优先调用桌面小懿AI生成解释、SOP 与审计包；不可达时回退到本地知识库。",
)
def copilot_brief(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    query = str(payload.get("query") or "").strip()
    if not query:
        query = "这段波动意味着什么？请给出 SOP 和告警解释。"
    scope = str(payload.get("scope") or "all")
    severity = str(payload.get("severity") or "medium")
    port = str(payload.get("port") or "CNYTN")
    asset_group = str(payload.get("asset_group") or "all_port")
    mode = str(payload.get("mode") or "explain_sop")
    engine = str(payload.get("engine") or "xiaoyi_ai")
    top_k = int(payload.get("top_k") or 8)
    top_k = max(1, min(top_k, 16))

    evidence = [_public_item(it) for it in _rank_items(query=query, scope=scope, top_k=top_k)]
    intent = _intent_from(query, scope)
    risk = _risk_from(severity, len(evidence))
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    primary = evidence[0] if evidence else {"title": "未命中知识条目", "snippet": "建议补充真实港口 SOP、告警、设备和合规文档后再进入生产使用。"}

    if intent == "alert_explain":
        headline = "该问题更像告警/异常解释，需要先确认是否真实越限，再进入分级处置。"
    elif intent == "sop_draft":
        headline = "该问题适合生成 SOP 草案，需保留触发条件、操作边界、回滚条件和人工确认。"
    elif intent == "compliance_answer":
        headline = "该问题涉及合规/审计口径，重点是边界、证据和可追溯说明。"
    elif intent == "device_diagnosis":
        headline = "该问题更偏设备诊断，需要把设备状态、负荷曲线和维护记录放到同一张证据链里看。"
    else:
        headline = "该问题适合由运营副驾生成解释、建议动作和下游模块承接。"

    actions = [
        {"priority": "P0", "action": "先冻结高风险自动动作", "owner": "值班长", "guardrail": "需要人工确认", "handoff": "OpsX"},
        {"priority": "P1", "action": f"围绕“{primary.get('title')}”核对现场证据", "owner": "设备/能源工程师", "guardrail": "证据不足不执行", "handoff": "Twin / Monitoring"},
        {"priority": "P2", "action": "生成 dry-run 处置建议并写入审计备注", "owner": "Ops Copilot", "guardrail": "仅建议，不直接生产下发", "handoff": "Execution"},
    ]

    llm_result: Dict[str, Any] = {
        "ok": False,
        "status": "local_rag",
        "reason": "未请求小懿AI。",
        "config": _xiaoyi_config(),
    }
    xiaoyi_answer = ""
    xiaoyi_confidence = ""
    xiaoyi_next_questions: List[str] = []
    uses_xiaoyi = engine in {"xiaoyi_ai", "auto", "external_llm"}

    if uses_xiaoyi:
        llm_result = _call_xiaoyi(query=query, scope=scope, mode=mode, top_k=top_k)
        parsed = llm_result.get("parsed") if llm_result.get("ok") else None
        if isinstance(parsed, dict):
            xiaoyi_answer = str(parsed.get("answer") or "")
            xiaoyi_confidence = str(parsed.get("confidence") or "")
            next_questions = parsed.get("next_questions")
            if isinstance(next_questions, list):
                xiaoyi_next_questions = [str(x) for x in next_questions[:4]]

            xiaoyi_evidence = _xiaoyi_evidence_items(parsed, limit=top_k)
            if xiaoyi_evidence:
                evidence = xiaoyi_evidence
                primary = evidence[0]
            intent = str(parsed.get("intent") or intent)
            risk = _risk_from(severity, len(evidence))
            headline = _first_answer_line(xiaoyi_answer)
            actions = _actions_from_xiaoyi(str(primary.get("title") or "小懿证据"))
            sop_steps = _steps_from_xiaoyi_answer(xiaoyi_answer, intent, query, scope)
        else:
            sop_steps = _steps_for(intent, query, scope)
    else:
        sop_steps = _steps_for(intent, query, scope)

    audit_packet = {
        "query_id": "cp-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3],
        "generated_at": now,
        "port": port,
        "asset_group": asset_group,
        "intent": intent,
        "mode": mode,
        "engine": engine,
        "llm_status": llm_result.get("status"),
        "llm_model": (llm_result.get("config") or {}).get("model"),
        "assistant": "小懿AI" if llm_result.get("ok") else "local_copilot_fallback",
        "assistant_confidence": xiaoyi_confidence or None,
        "scope": scope,
        "severity": severity,
        "risk_level": risk,
        "evidence_count": len(evidence),
        "human_in_loop": True,
    }

    return JSONResponse(
        {
            "summary": {
                "headline": headline,
                "risk_level": risk,
                "intent": intent,
                "primary_reference": primary.get("title"),
                "operator_note": (
                    xiaoyi_answer
                    if llm_result.get("ok")
                    else (
                        "小懿AI 当前不可达，已使用 Ops Copilot 本地规则库生成；请检查脱敏联动状态和环境变量配置。"
                        if uses_xiaoyi
                        else f"{headline}\n\n已按 Ops Copilot 本地知识库生成兜底答案。"
                    )
                ),
            },
            "actions": actions,
            "sop_steps": sop_steps,
            "evidence": evidence,
            "llm": {
                "engine": "小懿AI" if uses_xiaoyi else "Ops Copilot local",
                "status": llm_result.get("status"),
                "ok": bool(llm_result.get("ok")),
                "reason": llm_result.get("reason"),
                "configured": (llm_result.get("config") or {}).get("configured"),
                "online": bool(llm_result.get("ok")),
                "model": (llm_result.get("config") or {}).get("model"),
                "fallback": (llm_result.get("config") or {}).get("fallback"),
                "usage": llm_result.get("usage") or {},
            },
            "xiaoyi": {
                "ok": bool(llm_result.get("ok")),
                "answer": xiaoyi_answer,
                "confidence": xiaoyi_confidence,
                "next_questions": xiaoyi_next_questions,
                "request": llm_result.get("request") or {},
            },
            "audit_packet": audit_packet,
            "handoff": [
                {"target": "Twin 3D", "href": "/#twin3d-section", "reason": "回看曲线、场景和设备状态"},
                {"target": "Strategy / Execution", "href": "/#strategy-exec-module", "reason": "生成 dry-run 和审批流"},
                {"target": "OpsX", "href": "/#opsx-section", "reason": "记录审计 trace 与运行治理"},
            ],
        }
    )
