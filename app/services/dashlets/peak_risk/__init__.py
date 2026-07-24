from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Optional
from fastapi import APIRouter, Query, HTTPException
from .simulator import simulate_peak_risk

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z","+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: {s}") from e

@router.get("/peak_risk")
async def get_peak_risk(
    asset: str = Query(..., description="设备ID，如 QC-01"),
    start: str = Query(..., description="ISO 起始时间（视作“现在”）"),
    end:   str = Query(..., description="ISO 结束时间（用于上下文，不强约束）"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, float]:
    """
    返回未来 15/30/60 分钟“越峰概率”（0~1）。
    约定与前端一致：字段 m15/m30/m60，数值 0..1。
    """
    t0 = _parse_iso(start); t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")
    return simulate_peak_risk(asset=asset, now=t0, horizon_end=t1, seed=seed)
