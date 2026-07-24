from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Optional
from fastapi import APIRouter, Query, HTTPException
from .simulator import simulate_dq

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z","+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: {s}") from e

@router.get("/dq_lights")
async def get_dq_lights(
    asset: str = Query(..., description="设备ID，如 QC-01"),
    start: str = Query(..., description="ISO 起始时间"),
    end:   str = Query(..., description="ISO 结束时间"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, float]:
    """
    返回最近窗口的数据质量比例（0~1）：
      { missing: 缺失率, stale: 陈旧率, outlier: 越界率 }
    """
    t0 = _parse_iso(start); t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")
    return simulate_dq(asset=asset, start=t0, end=t1, seed=seed)
