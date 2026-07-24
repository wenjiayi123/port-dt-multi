from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException
from .simulator import simulate_savings

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z","+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: {s}") from e

@router.get("/savings")
async def get_savings(
    asset: str = Query(..., description="设备ID，如 QC-01"),
    start: str = Query(..., description="ISO 起始时间（用于确定“今天”的时区天界）"),
    end:   str = Query(..., description="ISO 结束时间（用于限定窗口，通常=预测窗尾）"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, Any]:
    """
    返回“今日累计”的节省金额(¥)与减排(kg CO₂e)：
      { "asset": "...", "cny": 1234.56, "co2": 789.0 }
    计算窗口：从 start 所在“本地日”的 00:00 ~ min(end, 当日24:00)（以 start 的时区为准）。
    """
    t0 = _parse_iso(start); t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")
    result = simulate_savings(asset=asset, start=t0, end=t1, seed=seed)
    # 前端只读 cny/co2；其余明细字段保留便于调试
    return dict(asset=asset, cny=result["cny"], co2=result["co2"])
