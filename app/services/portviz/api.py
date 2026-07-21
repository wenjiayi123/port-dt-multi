from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from .source import get_source


router = APIRouter()
_SRC = get_source()


@router.get("/bootstrap", summary="PortViz geometry and source provenance", tags=["portviz"])
def bootstrap() -> JSONResponse:
    return JSONResponse(_SRC.get_bootstrap())


@router.get("/stream", summary="PortViz data frame with source provenance", tags=["portviz"])
def stream(since: Optional[int] = Query(None, description="previous frame timestamp in milliseconds")) -> JSONResponse:
    return JSONResponse(_SRC.next_frame(since=since))
