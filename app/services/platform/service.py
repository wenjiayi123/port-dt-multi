from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from . import repo


DATA_DIR = Path(__file__).resolve().parent / "data"


def _dump(model):
    # 兼容 pydantic v1/v2
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _data_contract_items() -> List[Dict[str, Any]]:
    items = [
        {"id": "ope_leaderboard", "path": DATA_DIR / "ope_leaderboard.json", "label": "OPE 排行榜"},
        {"id": "ope_distributions", "path": DATA_DIR / "ope_distributions.json", "label": "OPE 分布"},
        {"id": "safety_summary", "path": DATA_DIR / "safety_summary.json", "label": "安全守护摘要"},
        {"id": "actions_hist", "path": DATA_DIR / "actions_hist.json", "label": "动作直方图"},
        {"id": "policies", "path": DATA_DIR / "policies.json", "label": "策略清单"},
        {"id": "port_profile", "path": DATA_DIR / "port_profile.json", "label": "港区画像"},
    ]
    result: List[Dict[str, Any]] = []
    for item in items:
        p = item["path"]
        exists = p.exists()
        result.append(
            {
                "id": item["id"],
                "label": item["label"],
                "path": str(p),
                "exists": exists,
                "mode": "mock-ready" if exists else "adapter-pending",
            }
        )
    return result


def _readiness_summary() -> Dict[str, Any]:
    items = _data_contract_items()
    ready = sum(1 for x in items if x["mode"] == "mock-ready")
    total = len(items)
    if ready == total:
        label = "mock-ready"
    elif ready == 0:
        label = "adapter-pending"
    else:
        label = "partially-ready"
    return {
        "label": label,
        "ready_count": ready,
        "total_count": total,
        "items": items,
    }


def _home_rule_from_metrics(
    peak_risk: float,
    violations_ppm: float,
    guard_pass_rate: float,
    ready_count: int,
    total_count: int,
) -> Dict[str, Any]:
    readiness_ratio = (ready_count / total_count) if total_count else 0.0

    if peak_risk >= 0.10 or violations_ppm >= 80:
        return {
            "overall_status": "需盯",
            "risk_level": "高",
            "priority_action": "先看孪生推演",
            "reason": "峰值风险或守护栏违规偏高，需要先在孪生与策略区确认风险窗口。",
        }

    if peak_risk >= 0.05 or guard_pass_rate < 0.95 or readiness_ratio < 0.80:
        return {
            "overall_status": "中高",
            "risk_level": "中高",
            "priority_action": "先看策略编排",
            "reason": "规则口径：驾驶舱状态=需盯。重点港区设备在线率高，但峰值风险仍需持续跟踪。",
        }

    return {
        "overall_status": "稳",
        "risk_level": "不高",
        "priority_action": "先看审计 / OpsX",
        "reason": "守护栏通过率与峰值风险均在可控范围，适合转入审计与上线治理视角。",
    }


async def evaluate_ope(payload: Dict[str, Any]) -> Dict[str, Any]:
    lb = repo.compute_leaderboard()
    return _dump(lb)


async def shield_enforce(payload: Dict[str, Any]) -> Dict[str, Any]:
    from .rl.safety import enforce

    action = payload.get("action", {})
    constraints = payload.get("constraints", {})
    return enforce(action, constraints)


async def policy_leaderboard() -> Dict[str, Any]:
    return _dump(repo.compute_leaderboard())


async def distributions() -> Dict[str, Any]:
    return repo.compute_distributions()


async def safety_summary() -> Dict[str, Any]:
    data = _dump(repo.compute_safety_summary())
    readiness = _readiness_summary()
    data["readiness"] = {
        "label": readiness["label"],
        "ready_count": readiness["ready_count"],
        "total_count": readiness["total_count"],
    }
    return data


async def actions_hist() -> Dict[str, Any]:
    return _dump(repo.compute_actions_hist())


def get_home_brief() -> Dict[str, Any]:
    board = _dump(repo.compute_leaderboard())
    safety = _dump(repo.compute_safety_summary())
    readiness = _readiness_summary()

    items = list(board.get("items") or [])
    best = items[0] if items else {}

    peak_risk = _safe_float(safety.get("cvar95_kwh"), 0.0)
    guard_pass_rate = _safe_float(safety.get("guard_pass_rate"), 0.0)
    violations_ppm = _safe_float(safety.get("violations_ppm"), 0.0)

    peak_risk_ratio = min(max(peak_risk / 100.0, 0.0), 0.25)
    rule = _home_rule_from_metrics(
        peak_risk=peak_risk_ratio,
        violations_ppm=violations_ppm,
        guard_pass_rate=guard_pass_rate,
        ready_count=readiness["ready_count"],
        total_count=readiness["total_count"],
    )

    strategy_id = str(best.get("id") or best.get("name") or "yard_lighting_dim")
    strategy_grade = "A+" if guard_pass_rate >= 0.97 else "A"
    pending = 1 if readiness["label"] != "mock-ready" else 0

    return {
        "generated_at": _now_iso(),
        "rule_source": {
            "name": "platform.service:get_home_brief",
            "description": "基于安全守护摘要 + OPE 排行榜 + 数据契约就绪态生成首页主结论。",
        },
        "headline": {
            "overall_status": rule["overall_status"],
            "risk_level": rule["risk_level"],
            "priority_action": rule["priority_action"],
            "reason": rule["reason"],
        },
        "risk_summary": {
            "grid_peak_risk_30d": round(peak_risk_ratio, 4),
            "guard_pass_rate": round(guard_pass_rate, 4),
            "violations_ppm": round(violations_ppm, 2),
            "critical_alerts_open": 0 if violations_ppm < 80 else 1,
            "major_alerts_open": 1 if peak_risk_ratio >= 0.05 else 0,
        },
        "loop_summary": {
            "latest_strategy_id": strategy_id,
            "latest_strategy_confidence": strategy_grade,
            "latest_priority_action": f"{rule['priority_action']}（峰值 {(peak_risk_ratio * 100):.1f}%）",
            "latest_dispatch_status": "待审批" if pending else "可直接下发",
            "pending_count": pending,
            "latest_job_count": 1,
        },
        "opsx_summary": {
            "phase": "canary",
            "stable_models": ["agv_charge@v2.0", "yard_crane@v1.7", "shore_bess@v1.3"],
            "candidate_models": ["yard_lighting@v2.1"],
        },
        "data_readiness": readiness,
        "top_policy": {
            "id": strategy_id,
            "mape": _safe_float(best.get("mape"), 0.0),
            "cvar95_kwh": _safe_float(best.get("cvar95_kwh"), 0.0),
            "violations_ppm": _safe_int(best.get("violations_ppm"), 0),
            "sample_total": _safe_int(board.get("sample_total"), 0),
        },
    }


async def home_brief() -> Dict[str, Any]:
    return get_home_brief()
