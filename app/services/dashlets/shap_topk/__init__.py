from __future__ import annotations
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query
from .simulator import simulate_shap_topk

router = APIRouter()

@router.get("/shap_topk")
async def get_shap_topk(
    asset: str = Query(..., description="设备ID，如 QC-01"),
    k: int = Query(5, ge=3, le=10, description="返回前K个特征"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, Any]:
    """
    返回形如：
      { "asset": "...", "items": [ {"name":"电价(18-21)","contribution":-5.4}, ... ] }
    约定：contribution<0 代表节电（绿色），>0 代表增耗（红色）。
    """
    items = simulate_shap_topk(asset=asset, k=k, seed=seed)
    return {"asset": asset, "items": items}
