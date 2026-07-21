from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .datasets import dataset_quality_report, import_dataset, list_datasets, load_port_dataset
from .trainer import TRAINING_MANAGER


router = APIRouter(prefix="/api/rl", tags=["rl-training-real"])


@router.get("/engine/capabilities")
async def capabilities() -> JSONResponse:
    return JSONResponse(TRAINING_MANAGER.capabilities())


@router.get("/datasets")
async def datasets() -> JSONResponse:
    items = list_datasets(TRAINING_MANAGER.data_root)
    return JSONResponse({"datasets": items, "count": len(items)})


@router.get("/datasets/{dataset_id}/quality")
async def dataset_quality(dataset_id: str) -> JSONResponse:
    try:
        dataset = load_port_dataset(dataset_id, TRAINING_MANAGER.data_root)
        return JSONResponse(dataset_quality_report(dataset))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    mapping_json: str = Form("{}"),
    metadata_json: str = Form("{}"),
) -> JSONResponse:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="only CSV datasets are accepted")
    tmp_path: Optional[Path] = None
    try:
        mapping = json.loads(mapping_json)
        metadata = json.loads(metadata_json)
        if not isinstance(mapping, dict) or not isinstance(metadata, dict):
            raise ValueError("mapping_json and metadata_json must be JSON objects")
        with NamedTemporaryFile(prefix="port-rl-upload-", suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            max_bytes = max(1, int(os.getenv("PORT_DT_MAX_DATASET_UPLOAD_MB", "50"))) * 1024 * 1024
            received = 0
            while chunk := await file.read(1024 * 1024):
                received += len(chunk)
                if received > max_bytes:
                    raise HTTPException(status_code=413, detail=f"dataset exceeds {max_bytes // (1024 * 1024)} MiB limit")
                tmp.write(chunk)
        result = import_dataset(tmp_path, dataset_id, mapping, metadata, TRAINING_MANAGER.data_root)
        return JSONResponse(result, status_code=201)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@router.post("/train/{job_id}/control")
async def control_training(job_id: str, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    try:
        return JSONResponse(TRAINING_MANAGER.control(job_id, str(payload.get("action") or "")))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown training job: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/train/{job_id}/history")
async def training_history(job_id: str, limit: int = 1000) -> JSONResponse:
    try:
        return JSONResponse(TRAINING_MANAGER.history(job_id, limit))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown training job: {job_id}") from exc


@router.post("/train/{job_id}/evaluate")
async def evaluate_training(job_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
    try:
        result = await asyncio.to_thread(TRAINING_MANAGER.evaluate, job_id, int((payload or {}).get("episodes") or 10))
        return JSONResponse(result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown training job: {job_id}") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/benchmarks/summary")
async def benchmark_summary(dataset_id: Optional[str] = None) -> JSONResponse:
    return JSONResponse(TRAINING_MANAGER.benchmark_summary(dataset_id))


@router.get("/models")
async def list_models() -> JSONResponse:
    return JSONResponse(TRAINING_MANAGER.model_registry().list())


@router.post("/models/sync")
async def sync_models() -> JSONResponse:
    return JSONResponse(TRAINING_MANAGER.model_registry().refresh())


@router.get("/models/{job_id}")
async def get_model(job_id: str) -> JSONResponse:
    try:
        return JSONResponse(TRAINING_MANAGER.model_registry().get(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown model run: {job_id}") from exc


@router.get("/models/{job_id}/readiness")
async def model_readiness(job_id: str) -> JSONResponse:
    try:
        config = json.loads((TRAINING_MANAGER.run_dir(job_id) / "config.json").read_text(encoding="utf-8"))
        benchmark = TRAINING_MANAGER.benchmark_summary(config.get("dataset_id"))
        return JSONResponse(TRAINING_MANAGER.model_registry().readiness(job_id, benchmark))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"unknown model run: {job_id}") from exc


@router.post("/models/{job_id}/alias")
async def set_model_alias(job_id: str, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    try:
        config = json.loads((TRAINING_MANAGER.run_dir(job_id) / "config.json").read_text(encoding="utf-8"))
        benchmark = TRAINING_MANAGER.benchmark_summary(config.get("dataset_id"))
        return JSONResponse(TRAINING_MANAGER.model_registry().set_alias(
            job_id,
            str(payload.get("alias") or ""),
            approved_by=str(payload.get("approved_by") or ""),
            reason=str(payload.get("reason") or ""),
            benchmark=benchmark,
        ))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown model run: {job_id}") from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/models/rollback")
async def rollback_model(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    try:
        registry = TRAINING_MANAGER.model_registry()
        target = registry.rollback_target()
        if not target:
            raise ValueError("no rollback alias is available")
        config = json.loads((TRAINING_MANAGER.run_dir(target) / "config.json").read_text(encoding="utf-8"))
        return JSONResponse(registry.rollback(
            approved_by=str(payload.get("approved_by") or ""),
            reason=str(payload.get("reason") or ""),
            benchmark=TRAINING_MANAGER.benchmark_summary(config.get("dataset_id")),
        ))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/train/{job_id}/evaluation")
async def get_evaluation(job_id: str) -> JSONResponse:
    try:
        run_dir = TRAINING_MANAGER.run_dir(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"unknown training job: {job_id}") from exc
    path = run_dir / "evaluation.json"
    trace_path = run_dir / "evaluation_trajectory.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="evaluation has not been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if trace_path.exists():
        payload["render"] = json.loads(trace_path.read_text(encoding="utf-8"))
    return JSONResponse(payload)


@router.post("/train/{job_id}/predict")
async def predict_control(job_id: str, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    try:
        return JSONResponse(await asyncio.to_thread(TRAINING_MANAGER.predict, job_id, payload))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown training job: {job_id}") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
