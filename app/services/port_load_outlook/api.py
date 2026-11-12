from datetime import date
from typing import Literal
from fastapi import APIRouter,HTTPException
from .summer_forecast import forecast

router=APIRouter()


@router.get("/api/forecast/port-load/{site}/{day}",tags=["forecast"])
def port_load_outlook(site:Literal["shanghai","los_angeles","funchal"],day:date,assess_peak:bool=False):
    try:return forecast(site,day,assess_peak=assess_peak)
    except ValueError as exc:
        if "指纹" in str(exc):raise HTTPException(503,"预测数据完整性核验失败") from exc
        raise HTTPException(422,str(exc)) from exc
    except OSError as exc:raise HTTPException(503,"预测档案暂不可用") from exc
