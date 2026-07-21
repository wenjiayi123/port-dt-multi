from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from .service import TwinSchemaService


router = APIRouter(prefix="/api", tags=["twin-schema"])
service = TwinSchemaService()


@router.get("/twin-models")
async def twin_models() -> JSONResponse:
    try:
        return JSONResponse(service.models())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/twin-graph")
async def twin_graph() -> JSONResponse:
    try:
        return JSONResponse(service.configured_graph())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/twin-graph/validate")
async def validate_twin_graph(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    result = service.validate_graph(payload)
    return JSONResponse(result, status_code=200 if result["valid"] else 422)


@router.get("/twin-calibration")
async def twin_calibration() -> JSONResponse:
    try:
        return JSONResponse(service.configured_calibration())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/twin-calibration/validate")
async def validate_twin_calibration(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    result = service.validate_calibration(payload)
    return JSONResponse(result, status_code=200 if result["valid"] else 422)
