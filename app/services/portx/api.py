from fastapi import APIRouter
from typing import Any, Dict

router = APIRouter()

@router.get("/kpi/dashboard")
async def kpi_dashboard():
    from .service import get_kpi_dashboard
    return await get_kpi_dashboard()

@router.get("/kpi/contribution")
async def kpi_contribution():
    from .service import get_kpi_contribution
    return await get_kpi_contribution()

@router.get("/pareto")
async def pareto():
    from .service import get_pareto_frontier
    return await get_pareto_frontier()
