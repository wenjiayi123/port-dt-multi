from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from pathlib import Path
import json, os, datetime
import re

router = APIRouter(prefix="/api/rl", tags=["rl-artifacts"])

# 约定：默认模型为 agv_charge；可通过 /api/rl/model/{model}/artifacts/* 指定不同模型
BASE = Path(__file__).resolve().parent
# 如果当前目录下就有 rl_model，则用当前目录；否则退回到上一级（app/services）
SERVICES_ROOT = BASE if (BASE / "rl_model").exists() else BASE.parent
MODEL_ROOT = SERVICES_ROOT / "rl_model"
DEFAULT_MODEL = "agv_charge"
DEFAULT_ART = (MODEL_ROOT / DEFAULT_MODEL / "artifacts").resolve()
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _art_dir(model: str) -> Path:
    if not MODEL_ID.fullmatch(str(model)) or not MODEL_ROOT.is_dir():
        raise HTTPException(status_code=404, detail="artifact model not found")
    registered = {
        child.name: (child / "artifacts").resolve()
        for child in MODEL_ROOT.iterdir()
        if child.is_dir() and (child / "artifacts").is_dir()
    }
    artifact_dir = registered.get(str(model))
    if artifact_dir is None:
        raise HTTPException(status_code=404, detail="artifact model not found")
    return artifact_dir

def _registered_artifact(dirpath: Path, requested: str) -> Path | None:
    if not dirpath.is_dir():
        return None
    files = {
        item.relative_to(dirpath).as_posix(): item.resolve()
        for item in dirpath.rglob("*")
        if item.is_file()
    }
    return files.get(str(requested).lstrip("/"))

def _safe_in(dirpath: Path, fp: Path) -> bool:
    try:
        return fp.resolve().is_relative_to(dirpath.resolve())
    except Exception:
        return False

@router.get("/artifacts/{path:path}")
async def get_artifact_default(path: str):
    fp = _registered_artifact(DEFAULT_ART, path)
    if fp is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(fp)

@router.get("/model/{model}/artifacts/{path:path}")
async def get_artifact_by_model(model: str, path: str):
    art = _art_dir(model)
    fp = _registered_artifact(art, path)
    if fp is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(fp)

@router.get("/metrics/history")
async def rl_metrics_history():
    """
    返回训练/评估历史。
    优先读取 DEFAULT_ART/policy_evaluate_history.jsonl（原样 text），
    若不存在返回空 JSON 数组（前端已做降级 parse）。
    """
    cand = [
        DEFAULT_ART / "policy_evaluate_history.jsonl",
    ]
    for p in cand:
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            # 直接以 text/plain 返回，前端用 r.text() 解析行
            return PlainTextResponse(txt, media_type="text/plain")
    return JSONResponse([])

@router.get("/rollout/status")
async def rollout_status():
    """
    可选：提供一个迷你 Rollout 状态，便于前端展示（若你不需要可忽略）。
    """
    art = DEFAULT_ART
    info = {}
    meta_file = art / "policy_meta.json"
    if meta_file.exists():
        try:
            info = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            info = {}
    mtime = None
    try:
        mtime = datetime.datetime.utcfromtimestamp((art / "policy.bin").stat().st_mtime).isoformat() + "Z"
    except Exception:
        mtime = None

    return JSONResponse({
        "phase": "stable",
        "candidate_version": info.get("candidate_version") or "policy.bin",
        "stable_version": info.get("stable_version") or "policy.bin",
        "traffic_pct": info.get("traffic_pct", 1.0),
        "updated_at": info.get("updated_at") or mtime or datetime.datetime.utcnow().isoformat() + "Z",
        "metrics": info.get("metrics", {}),
        "thresholds": info.get("thresholds", {})
    })
