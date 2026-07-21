from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException
from .simulator import simulate_calibration

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z","+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: {s}") from e

@router.get("/calibration")
async def get_calibration(
    asset: str = Query(..., description="设备ID，如 QC-01"),
    start: str = Query(..., description="ISO 起始时间"),
    end: str   = Query(..., description="ISO 结束时间"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, Any]:
    t0 = _parse_iso(start); t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")
    m = simulate_calibration(asset=asset, start=t0, end=t1, seed=seed)
    # 前端字段：mape(0~1), cover(0~1), bias(kW), sigma(kW)
    return dict(asset=asset, **m)
