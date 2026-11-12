from datetime import date
import json
from pathlib import Path
import subprocess
import sys
from threading import BoundedSemaphore
from fastapi import APIRouter,HTTPException

router=APIRouter()
_slot=BoundedSemaphore(1)
CASES=("shanghai","industrial_mix","industrial_mix_price_shift","phase_minus3","phase_plus6","week_switch","unlisted_tariff")


@router.get("/api/forecast/external-load/{case}/{day}",tags=["forecast"])
def external_load_replay(case:str,day:date):
    if case not in CASES or not date(2026,6,1)<=day<=date(2026,9,23):raise HTTPException(422,"工况或完整 UTC 周窗口不在可用范围")
    if not _slot.acquire(blocking=False):raise HTTPException(429,"本机负荷压力复算正在运行")
    try:
        p=Path(__file__).resolve().parents[2]/"scripts/gullmate_external_load/native_worker.py"
        try:r=subprocess.run([sys.executable,"-X","utf8",p,case,day.isoformat()],capture_output=True,encoding="utf-8",timeout=80)
        except subprocess.TimeoutExpired as e:raise HTTPException(504,"原生计算超时") from e
        if r.returncode:raise HTTPException(503,"负荷或策略原件未通过核验")
        d=json.loads(r.stdout)
        if d["week"]!=day.isoformat() or d["case"]!=case or d["production_authority"] is not False:raise HTTPException(503,"回执身份不符")
        return d
    finally:_slot.release()
