from typing import Any, Dict

async def explain_with_counterfactual(payload: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: SHAP Top-K + 反事实 Δ
    return {"ok": True, "explain": {}}

async def get_change_radar() -> Dict[str, Any]:
    # TODO: “自上次上线有什么变化”雷达
    return {"ok": True, "radar": {}}
