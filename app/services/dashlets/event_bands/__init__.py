# app/services/dashlets/event_bands/__init__.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException

from .simulator import simulate_event_bands

router = APIRouter()

def _parse_iso(s: str) -> datetime:
    """把 '2025-11-02T05:10:00Z' 或 '2025-11-02T05:10:00+00:00' 解析成 aware 的 UTC 时间"""
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO time: {s}") from e

@router.get("/event_bands")
async def get_event_bands(
    asset: str = Query(..., description="设备ID，例如 QC-01"),
    start: str = Query(..., description="ISO 起始时间（含Z）"),
    end: str = Query(..., description="ISO 结束时间（含Z）"),
    seed: Optional[int] = Query(None, description="可选：随机种子，便于复现"),
) -> Dict[str, Any]:
    """
    大白话：
    - 前端给你“设备 + 起止时间”
    - 我们用模拟器造出一堆彩色时间段（电价峰/低谷、DR、作业高强度、潮汐、策略执行……）
    - 再把时间都转成 '...Z' 字符串返回，前端直接画底色条和顶部小标签
    """
    t0 = _parse_iso(start)
    t1 = _parse_iso(end)
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="end must be after start")

    bands = simulate_event_bands(asset=asset, start=t0, end=t1, seed=seed)
    out = [{
        "t0": b["t0"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "t1": b["t1"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "label": b.get("label", ""),
        "kind": b.get("kind", "generic"),
        "color": b.get("color", "rgba(148,163,184,0.12)"),
    } for b in bands]
    return {"asset": asset, "bands": out}
