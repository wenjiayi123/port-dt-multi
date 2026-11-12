"""Read-only on-demand native Shore+BESS replay for the locally installed controller."""
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
from threading import BoundedSemaphore

from fastapi import APIRouter,HTTPException

router=APIRouter()
_worker=BoundedSemaphore(1)


@router.get("/shore-storage/weekly/{week}")
def weekly_replay(week:date, include_frames:bool=False):
    if not _worker.acquire(blocking=False):raise HTTPException(429,"岸电复核正在使用本机仿真资源")
    try:
        script=Path(__file__).resolve().parents[3]/"scripts/gullmate_shore/infer_week.py"
        cmd=[sys.executable,"-X","utf8",str(script),week.isoformat()]
        try:r=subprocess.run(cmd,capture_output=True,encoding="utf-8",timeout=75)
        except subprocess.TimeoutExpired as exc:raise HTTPException(504,"岸电周重放超时") from exc
        if r.returncode:
            if "日期不在支持" in r.stderr or "周窗口越过冻结分区" in r.stderr:
                raise HTTPException(422,"日期无法形成分区内完整周窗口")
            raise HTTPException(503,"本机岸电模型或数据不可用")
        sim=json.loads(r.stdout)
        if sim["week"]!=week.isoformat() or sim["production_authority"] is not False:raise HTTPException(503,"仿真回执范围不符")
        trace=sim["methods"]["selected"]["hours"]
        if len(trace)!=168:raise HTTPException(503,"逐小时回执不完整")
        extra = {}
        if include_frames:
            from .shore_scene_frames import scene_frames
            try:
                extra["frames"] = scene_frames(sim["methods"])
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(503, "孪生逐时场景回执不可用") from exc
        return {**extra,"schema":"twin.shore-storage-week.v1","week":sim["week"],"part":sim["part"],"hours":len(trace),
                "weight_sha256":sim["weight_sha256"],"runtime_sha256":sim["runtime_sha256"],
                "methods":{k:v["metrics"] for k,v in sim["methods"].items()},
                "carbon_delta_by_component":{"battery_kg":sum(-h["bess_kw"]*h["factor"] for h in trace),
                                             "flex_kg":sum(h["flex_kw"]*h["factor"] for h in trace)},
                "source":"native TwinLab replay","production_authority":False}
    finally:_worker.release()
