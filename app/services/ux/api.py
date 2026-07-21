from fastapi import APIRouter
from typing import Any, Dict

router = APIRouter()

@router.post("/explain")
async def explain(payload: Dict[str, Any]):
    from .service import explain_with_counterfactual
    return await explain_with_counterfactual(payload)

@router.get("/change-radar")
async def change_radar():
    from .service import get_change_radar
    return await get_change_radar()
