from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException
from .simulator import simulate_action_markers

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: {s}") from e

@router.get("/action_markers")
async def get_action_markers(
    asset: str = Query(..., description="设备ID，如 QC-01"),
    start: str = Query(..., description="ISO 起始时间"),
    end: str = Query(..., description="ISO 结束时间"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, Any]:
    t0 = _parse_iso(start)
    t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")
    items = simulate_action_markers(asset=asset, start=t0, end=t1, seed=seed)
    out = [{
        "ts": it["ts"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kind": it["kind"],        # 'act_charge' | 'act_discharge' | 'guard' | 'setpoint'
        "label": it.get("label",""),
        "color": it.get("color"),
        "meta": it.get("meta",{}),
    } for it in items]
    return {"asset": asset, "markers": out}
