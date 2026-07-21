from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException
from .simulator import simulate_residual_heatmap

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: {s}") from e

@router.get("/residual_heatmap")
async def get_residual_heatmap(
    asset: str = Query(..., description="设备ID，如 QC-01"),
    start: str = Query(..., description="ISO 起始时间"),
    end:   str = Query(..., description="ISO 结束时间"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, Any]:
    """
    返回 7×24 的残差强度网格（0~1）。0=误差小(绿)，1=误差大(红)。
    行=周几(0..6)，列=小时(0..23)；前端只用数值，不看标签。
    """
    t0 = _parse_iso(start); t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")
    grid = simulate_residual_heatmap(asset=asset, start=t0, end=t1, seed=seed)
    return {"asset": asset, "grid": grid}
