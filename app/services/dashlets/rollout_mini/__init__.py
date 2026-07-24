from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Optional
from fastapi import APIRouter, Query, HTTPException
from .simulator import simulate_rollout_snapshot

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z","+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: {s}") from e

@router.get("/rollout_mini")
async def get_rollout_mini(
    asset: str = Query("QC-01", description="设备ID(可选)"),
    start: str = Query(..., description="ISO 起始时间（视作“现在”）"),
    end:   str = Query(..., description="ISO 结束时间（上下文，不强约束）"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, object]:
    t0 = _parse_iso(start); t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")
    snap = simulate_rollout_snapshot(asset=asset, now=t0, seed=seed)
    # 前端使用的字段：phase/traffic_pct/mape/guard_block_rate/sla_violation_rate/thresholds
    return snap
