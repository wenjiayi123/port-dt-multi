from fastapi import APIRouter, HTTPException, Query

from .service import asset_series, assets, profile, replacement_contract, snapshot
from .workspace import workspace

router = APIRouter(prefix="/api/public-port-demo", tags=["public-parameter-presentation"])


@router.get("/workspace")
def get_workspace(at: str | None = Query(None, description="固定演示时刻，含时区")):
    try:
        return workspace(at)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=f"presentation workspace unavailable: {exc}") from exc


@router.get("/profile")
def get_profile():
    return profile()


@router.get("/snapshot")
def get_snapshot(at: str | None = Query(None, description="固定历史演示时刻，含时区；留空则演示时钟推进")):
    try:
        return snapshot(at)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=f"presentation source unavailable: {exc}") from exc


@router.get("/replacement-contract")
def get_replacement_contract():
    return replacement_contract()


@router.get("/assets")
def get_assets():
    try:
        return assets()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=f"presentation groups unavailable: {exc}") from exc


@router.get("/asset-series/{asset_id}")
def get_asset_series(asset_id: str, at: str | None = None):
    try:
        return asset_series(asset_id, at)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=f"presentation group series unavailable: {exc}") from exc
