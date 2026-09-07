from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .business_runtime import BUSINESS_POLICY_REGISTRY, MODULES

router = APIRouter(prefix="/business-v7", tags=["business-real-rl"])


class PredictRequest(BaseModel):
    observation: list[float] = Field(min_length=1, max_length=110)
    expected_model_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


@router.get("/evidence")
def evidence():
    try:
        return {"modules": [BUSINESS_POLICY_REGISTRY.evidence(m) for m in MODULES], "active_champions": [BUSINESS_POLICY_REGISTRY.evidence(m, champion=True) for m in MODULES], "production_authority": False}
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/{module}/predict")
def predict(module: str, request: PredictRequest):
    try:
        return BUSINESS_POLICY_REGISTRY.predict(module, request.observation, expected_model_sha256=request.expected_model_sha256)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc
