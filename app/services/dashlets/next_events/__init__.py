from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException
from .simulator import simulate_next_events

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z","+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: %s" % s) from e

@router.get("/next_events")
async def get_next_events(
    asset: str = Query(..., description="设备ID，如 QC-01"),
    start: str = Query(..., description="ISO 起始（视作“现在”）"),
    end:   str = Query(..., description="ISO 结束（上下文，不强约束）"),
    horizon_min: int = Query(60, ge=5, le=240, description="向前看的分钟数（5~240）"),
    seed: Optional[int] = Query(None, description="随机种子（可选）"),
) -> Dict[str, Any]:
    t0 = _parse_iso(start); t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")
    items = simulate_next_events(asset=asset, now=t0, horizon_min=horizon_min, seed=seed)
    # 统一 ts 为 Z 结尾的 UTC，保留 ts_local 作为可读时间
    out=[]
    for ev in items:
        ts = ev["ts"].astimezone(timezone.utc).isoformat().replace("+00:00","Z")
        row = {k:v for k,v in ev.items() if k!="ts"}
        row["ts"]=ts
        out.append(row)
    return {"asset": asset, "items": out}
