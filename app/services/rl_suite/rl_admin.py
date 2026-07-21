from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, FileResponse
from pathlib import Path
import threading, time, random, json, io, csv, zipfile, os, datetime

router = APIRouter(prefix="/api/rl", tags=["rl-admin"])

BASE = Path(__file__).resolve().parent              # .../app/services
MODEL_ROOT = (BASE / "rl_model").resolve()
DEFAULT_MODEL = "agv_charge"

def _art_dir(model: str) -> Path:
    return (MODEL_ROOT / model / "artifacts").resolve()

def _safe_in(dirpath: Path, fp: Path) -> bool:
    try:
        return str(fp.resolve()).startswith(str(dirpath.resolve()))
    except Exception:
        return False

def _jsonl_file(model: str) -> Path:
    return _art_dir(model) / "policy_evaluate_history.jsonl"

def _policy_meta(model: str) -> Path:
    return _art_dir(model) / "policy_meta.json"

# ---- 1.1 列出可用模型 ----
@router.get("/models")
async def list_models():
    models = []
    if not MODEL_ROOT.exists():
        return JSONResponse(models)
    for d in sorted(p.name for p in MODEL_ROOT.iterdir() if p.is_dir()):
        art = _art_dir(d)
        if art.exists():
            models.append({"id": d, "artifacts": str(art.relative_to(Path.cwd()))})
    return JSONResponse(models)

# ---- 1.2 历史 latest（JSON）/ CSV 导出 / 清空 ----
def _read_jsonl_tail(fp: Path, limit: int = 1000):
    if not fp.exists():
        return []
    try:
        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = lines[-limit:] if limit>0 else lines
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return out
    except Exception:
        return []

@router.get("/metrics/latest")
async def metrics_latest(model: str = DEFAULT_MODEL, limit: int = 1000):
    fp = _jsonl_file(model)
    return JSONResponse(_read_jsonl_tail(fp, limit))

@router.get("/metrics/csv")
async def metrics_csv(model: str = DEFAULT_MODEL, limit: int = 2000):
    rows = _read_jsonl_tail(_jsonl_file(model), limit)
    buf = io.StringIO()
    w = csv.writer(buf)
    header = ["ts","step","reward","peak_reduction_kW","episode_len","latency_min","entropy"]
    w.writerow(header)
    for r in rows:
        w.writerow([
            r.get("ts") or r.get("time") or r.get("timestamp") or "",
            r.get("step") or r.get("episode") or "",
            r.get("reward") or r.get("returns") or r.get("episode_reward") or r.get("return"),
            r.get("peak_reduction_kW") or r.get("peak_kW_delta"),
            r.get("episode_len") or r.get("len") or r.get("steps"),
            r.get("latency_min"),
            r.get("entropy") or r.get("policy_entropy"),
        ])
    data = buf.getvalue().encode("utf-8")
    return StreamingResponse(io.BytesIO(data), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{model}_history.csv"'})

@router.post("/metrics/clear")
async def metrics_clear(model: str = DEFAULT_MODEL):
    fp = _jsonl_file(model)
    art = _art_dir(model)
    art.mkdir(parents=True, exist_ok=True)
    if fp.exists() and fp.stat().st_size > 0:
        bak = fp.with_suffix(".jsonl.bak." + datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))
        fp.rename(bak)
    fp.write_text("", encoding="utf-8")
    return JSONResponse({"ok": True, "message": "cleared", "file": str(fp)})

# ---- 1.3 产物上传（zip/jsonl/png）与列出文件 ----
@router.post("/artifacts/upload")
async def artifacts_upload(model: str = DEFAULT_MODEL, file: UploadFile = File(...)):
    art = _art_dir(model); art.mkdir(parents=True, exist_ok=True)
    name = file.filename or "upload.bin"
    data = await file.read()
    if name.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(art)
            return JSONResponse({"ok": True, "message": "zip extracted", "dir": str(art)})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"zip error: {e}")
    else:
        # 允许覆盖同名文件
        dst = art / name
        with open(dst, "wb") as f:
            f.write(data)
        return JSONResponse({"ok": True, "file": str(dst)})

@router.get("/artifacts/list")
async def artifacts_list(model: str = DEFAULT_MODEL):
    art = _art_dir(model)
    if not art.exists():
        return JSONResponse({"files": []})
    files = []
    for p in sorted(art.glob("*")):
        if p.is_file():
            files.append({
                "name": p.name,
                "size": p.stat().st_size,
                "mtime": datetime.datetime.utcfromtimestamp(p.stat().st_mtime).isoformat() + "Z"
            })
    return JSONResponse({"files": files})

# ---- 1.4 训练报告（概要） ----
@router.get("/report")
async def rl_report(model: str = DEFAULT_MODEL, limit: int = 2000):
    rows = _read_jsonl_tail(_jsonl_file(model), limit)
    if not rows:
        return JSONResponse({"ok": True, "n": 0, "summary": {}})
    def num(x):
        try:
            return float(x)
        except Exception:
            return None
    rewards = [num(r.get("reward") or r.get("returns") or r.get("episode_reward") or r.get("return")) for r in rows]
    rewards = [x for x in rewards if x is not None]
    peak = [num(r.get("peak_reduction_kW") or r.get("peak_kW_delta")) for r in rows if r.get("peak_reduction_kW") or r.get("peak_kW_delta")]
    now = datetime.datetime.utcnow().isoformat() + "Z"
    meta_path = _policy_meta(model)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    summary = {
        "model": model,
        "rows": len(rows),
        "reward": {
            "last": rewards[-1] if rewards else None,
            "mean": sum(rewards)/len(rewards) if rewards else None,
            "min": min(rewards) if rewards else None,
            "max": max(rewards) if rewards else None,
            "last100_mean": (sum(rewards[-100:])/min(100,len(rewards))) if rewards else None
        },
        "peak_reduction_kW": {
            "mean": (sum(peak)/len(peak)) if peak else None
        },
        "updated_at": now,
        "policy_meta": meta
    }
    return JSONResponse({"ok": True, "summary": summary})

# ---- 1.5 短训（演示数据生成） ----
TRAINERS = {}  # model -> {"thread": Thread, "stop": bool}

def _trainer_fn(model: str, period_sec: float = 1.2):
    """简单写入演示 JSONL 行；不依赖外部库。"""
    art = _art_dir(model); art.mkdir(parents=True, exist_ok=True)
    fp = _jsonl_file(model)
    step = 0
    base = random.uniform(0.0, 0.5)
    while True:
        st = TRAINERS.get(model)
        if not st or st.get("stop"):
            break
        step += 1
        reward = base + 0.2 * random.random() + 0.05 * (step/100.0)
        peak_kw = max(0.0, 5.0 + random.gauss(0, 1.0) + 0.02*step)
        ep_len = 20 + int(5*random.random())
        entropy = max(0.01, 0.8 - 0.003*step + random.uniform(-0.02,0.02))
        row = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "step": step,
            "reward": round(float(reward), 6),
            "peak_reduction_kW": round(float(peak_kw), 3),
            "episode_len": ep_len,
            "entropy": round(float(entropy), 5)
        }
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
        time.sleep(period_sec)

@router.post("/train/start")
async def train_start(model: str = DEFAULT_MODEL, period_sec: float = 1.2):
    st = TRAINERS.get(model)
    if st and not st.get("stop"):
        return JSONResponse({"ok": True, "message": "already running"})
    TRAINERS[model] = {"stop": False}
    th = threading.Thread(target=_trainer_fn, args=(model, period_sec), daemon=True)
    TRAINERS[model]["thread"] = th
    th.start()
    return JSONResponse({"ok": True, "message": "started"})

@router.post("/train/stop")
async def train_stop(model: str = DEFAULT_MODEL):
    st = TRAINERS.get(model)
    if not st:
        return JSONResponse({"ok": True, "message": "not running"})
    st["stop"] = True
    return JSONResponse({"ok": True, "message": "stopping"})

@router.get("/train/status")
async def train_status(model: str = DEFAULT_MODEL):
    st = TRAINERS.get(model, {})
    running = bool(st and not st.get("stop") and st.get("thread") and st["thread"].is_alive())
    fp = _jsonl_file(model)
    n = 0
    if fp.exists():
        try:
            n = sum(1 for _ in open(fp, "r", encoding="utf-8", errors="ignore"))
        except Exception:
            n = 0
    return JSONResponse({"running": running, "rows": n, "file": str(fp)})
