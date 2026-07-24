from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter()


# —— 主能力 —— #
@router.post("/ope/evaluate")
async def ope_evaluate(payload: Dict[str, Any]):
    from .service import evaluate_ope

    return await evaluate_ope(payload)


@router.get("/policy/leaderboard")
async def policy_leaderboard():
    from .service import policy_leaderboard

    return await policy_leaderboard()


@router.get("/ope/distributions")
async def ope_distributions():
    from .service import distributions

    return await distributions()


@router.get("/safety/summary")
async def safety_summary():
    from .service import safety_summary

    return await safety_summary()


@router.get("/safety/actions")
async def safety_actions():
    from .service import actions_hist

    return await actions_hist()


@router.post("/safety/shield")
async def safety_shield(payload: Dict[str, Any]):
    from .service import shield_enforce

    return await shield_enforce(payload)


# —— 首页/平台摘要 —— #
@router.get("/home_brief")
async def home_brief():
    from .service import home_brief

    return await home_brief()


@router.get("/platform_summary")
async def platform_summary():
    from .service import get_home_brief

    brief = get_home_brief()
    return {
        "generated_at": brief.get("generated_at"),
        "headline": brief.get("headline", {}),
        "risk_summary": brief.get("risk_summary", {}),
        "loop_summary": brief.get("loop_summary", {}),
        "opsx_summary": brief.get("opsx_summary", {}),
        "top_policy": brief.get("top_policy", {}),
    }


@router.get("/data_readiness")
async def data_readiness():
    from .service import get_home_brief

    brief = get_home_brief()
    return brief.get("data_readiness", {})


# —— demo 兼容路由（前端仍可用 /demo/*，内部转发到真实计算） —— #
@router.get("/demo/ope/leaderboard")
async def demo_ope_leaderboard():
    from .service import policy_leaderboard

    return await policy_leaderboard()


@router.get("/demo/ope/distributions")
async def demo_ope_distributions():
    from .service import distributions

    return await distributions()


@router.get("/demo/safety/summary")
async def demo_safety_summary():
    from .service import safety_summary

    return await safety_summary()


@router.get("/demo/safety/actions")
async def demo_safety_actions():
    from .service import actions_hist

    return await actions_hist()


@router.get("/demo/home_brief")
async def demo_home_brief():
    from .service import home_brief

    return await home_brief()
