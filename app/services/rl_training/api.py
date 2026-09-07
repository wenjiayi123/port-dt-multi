from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .datasets import (
    dataset_quality_report,
    import_dataset,
    list_datasets,
    load_port_dataset,
    site_replacement_readiness_report,
)
from .profiles import list_profiles, load_profile
from .trainer import TRAINING_MANAGER
from .business_api import router as business_rl_router


router = APIRouter(prefix="/api/rl", tags=["rl-training-real"])
router.include_router(business_rl_router)
REPO_ROOT = Path(__file__).resolve().parents[3]
REGULATORY_EVIDENCE_ROOT = REPO_ROOT / "evidence/v4/regulatory_delay"
INTEGRATED_EVIDENCE_ROOT = REPO_ROOT / "evidence/v5/integrated_business"
INTEGRATED_GUARDRAIL_ROOT = REPO_ROOT / "evidence/v5/deterministic_guardrails"
COORDINATED_EVIDENCE_ROOT = REPO_ROOT / "evidence/v6/coordinated_business"


@router.get("/engine/capabilities")
async def capabilities() -> JSONResponse:
    return JSONResponse(TRAINING_MANAGER.capabilities())


@router.get("/datasets")
async def datasets() -> JSONResponse:
    items = list_datasets(TRAINING_MANAGER.data_root)
    return JSONResponse({"datasets": items, "count": len(items)})


@router.get("/port-profiles")
async def port_profiles() -> JSONResponse:
    items = list_profiles()
    return JSONResponse({"profiles": items, "count": len(items)})


@router.get("/port-profiles/{profile_id}")
async def port_profile(profile_id: str) -> JSONResponse:
    try:
        return JSONResponse(load_profile(profile_id))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/quality")
async def dataset_quality(dataset_id: str) -> JSONResponse:
    try:
        dataset = load_port_dataset(dataset_id, TRAINING_MANAGER.data_root)
        return JSONResponse(dataset_quality_report(dataset))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/site-readiness")
async def dataset_site_readiness(dataset_id: str) -> JSONResponse:
    try:
        dataset = load_port_dataset(dataset_id, TRAINING_MANAGER.data_root)
        return JSONResponse(site_replacement_readiness_report(dataset))
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
async def benchmark_summary(
    dataset_id: Optional[str] = None,
    environment_version: Optional[str] = None,
    business_profile_id: Optional[str] = None,
) -> JSONResponse:
    return JSONResponse(
        TRAINING_MANAGER.benchmark_summary(
            dataset_id,
            environment_version=environment_version,
            business_profile_id=business_profile_id,
        )
    )


@router.get("/regulatory-resilience/evidence")
async def regulatory_resilience_evidence() -> JSONResponse:
    """Return hash-gated V4 evidence without exposing local absolute paths."""
    pointer_path = REGULATORY_EVIDENCE_ROOT / "latest.json"
    if not pointer_path.exists():
        raise HTTPException(
            status_code=404, detail="regulatory resilience evidence is unavailable"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    report_path = (REPO_ROOT / str(pointer.get("report_path") or "")).resolve()
    evidence_root = REGULATORY_EVIDENCE_ROOT.resolve()
    if not report_path.is_relative_to(evidence_root) or not report_path.is_file():
        raise HTTPException(status_code=409, detail="regulatory evidence pointer is invalid")
    observed_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if observed_sha256 != pointer.get("report_sha256"):
        raise HTTPException(
            status_code=409, detail="regulatory evidence hash gate failed"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    legacy = dict(report.get("legacy_preservation") or {})
    legacy.pop("sha256_before", None)
    legacy.pop("sha256_after", None)
    report["legacy_preservation"] = legacy

    forward_path_value = str(pointer.get("forward_challenge_path") or "")
    if not forward_path_value:
        raise HTTPException(
            status_code=409, detail="independent forward challenge is unavailable"
        )
    forward_path = (REPO_ROOT / forward_path_value).resolve()
    if not forward_path.is_relative_to(evidence_root) or not forward_path.is_file():
        raise HTTPException(
            status_code=409, detail="forward challenge pointer is invalid"
        )
    forward_sha256 = hashlib.sha256(forward_path.read_bytes()).hexdigest()
    if forward_sha256 != pointer.get("forward_challenge_sha256"):
        raise HTTPException(
            status_code=409, detail="forward challenge hash gate failed"
        )
    forward_challenge = json.loads(forward_path.read_text(encoding="utf-8"))
    if forward_challenge.get("status") != "PASS":
        raise HTTPException(
            status_code=409, detail="independent forward challenge is blocked"
        )
    return JSONResponse(
        {
            "schema": "port-dt-regulatory-resilience-api.v2",
            "status": report.get("status"),
            "report_sha256": observed_sha256,
            "evidence_path": str(report_path.relative_to(REPO_ROOT)),
            "forward_challenge_status": forward_challenge.get("status"),
            "forward_challenge_sha256": forward_sha256,
            "forward_challenge_path": str(forward_path.relative_to(REPO_ROOT)),
            "production_authority": False,
            "report": report,
            "forward_challenge": forward_challenge,
        }
    )


@router.get("/integrated-business/evidence")
async def integrated_business_evidence() -> JSONResponse:
    """Return the hash-gated V5 offline champion and deterministic gate proof."""
    champion_path = INTEGRATED_EVIDENCE_ROOT / "offline_champion.json"
    guardrail_pointer_path = INTEGRATED_GUARDRAIL_ROOT / "latest.json"
    if not champion_path.is_file() or not guardrail_pointer_path.is_file():
        raise HTTPException(
            status_code=404, detail="integrated business evidence is unavailable"
        )
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    if (
        champion.get("status") != "ADMITTED_OFFLINE_CHAMPION"
        or champion.get("production_authority") is not False
    ):
        raise HTTPException(status_code=409, detail="integrated champion is blocked")
    report_path = (REPO_ROOT / str(champion.get("report_path") or "")).resolve()
    evidence_root = INTEGRATED_EVIDENCE_ROOT.resolve()
    if not report_path.is_relative_to(evidence_root) or not report_path.is_file():
        raise HTTPException(
            status_code=409, detail="integrated evidence pointer is invalid"
        )
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if report_sha256 != champion.get("report_sha256"):
        raise HTTPException(
            status_code=409, detail="integrated evidence hash gate failed"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    admission = report.get("admission") or {}
    if admission.get("passed") is not True or admission.get(
        "production_authority"
    ) is not False:
        raise HTTPException(
            status_code=409, detail="integrated business admission is blocked"
        )
    training = report.get("training") or {}
    if training.get("selected_job_id") != champion.get("selected_job_id"):
        raise HTTPException(
            status_code=409, detail="integrated selected job binding failed"
        )

    model_path = (
        REPO_ROOT / str(champion.get("selected_model_path") or "")
    ).resolve()
    model_root = (REPO_ROOT / "data/rl/runs").resolve()
    if (
        not model_path.is_relative_to(model_root)
        or not model_path.is_file()
        or hashlib.sha256(model_path.read_bytes()).hexdigest()
        != champion.get("selected_model_sha256")
    ):
        raise HTTPException(
            status_code=409, detail="integrated champion model hash gate failed"
        )

    dataset_root = (REPO_ROOT / "data/rl/datasets").resolve()
    verified_datasets: Dict[str, Dict[str, Any]] = {}
    for label, report_key, id_key, sha_key in (
        ("training", "dataset", "dataset_id", "dataset_sha256"),
        (
            "forward",
            "final_evaluation_dataset",
            "final_evaluation_dataset_id",
            "final_evaluation_dataset_sha256",
        ),
    ):
        dataset_evidence = report.get(report_key) or {}
        artifact_path = (
            REPO_ROOT / str(dataset_evidence.get("artifact") or "")
        ).resolve()
        expected_sha256 = champion.get(sha_key)
        if (
            dataset_evidence.get("dataset_id") != champion.get(id_key)
            or dataset_evidence.get("sha256") != expected_sha256
            or not artifact_path.is_relative_to(dataset_root)
            or not artifact_path.is_file()
            or hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            != expected_sha256
        ):
            raise HTTPException(
                status_code=409,
                detail=f"integrated {label} dataset hash gate failed",
            )
        verified_datasets[label] = {
            "dataset_id": champion.get(id_key),
            "sha256": expected_sha256,
            "rows": int(dataset_evidence.get("rows") or 0),
        }
    report["legacy_preservation"] = {
        "checked_artifact_count": int(
            (report.get("legacy_preservation") or {}).get(
                "checked_artifact_count"
            )
            or 0
        ),
        "preserved": (report.get("legacy_preservation") or {}).get(
            "preserved"
        )
        is True,
    }

    guardrail_pointer = json.loads(
        guardrail_pointer_path.read_text(encoding="utf-8")
    )
    guardrail_report_path = (
        REPO_ROOT / str(guardrail_pointer.get("report_path") or "")
    ).resolve()
    guardrail_root = INTEGRATED_GUARDRAIL_ROOT.resolve()
    if (
        guardrail_pointer.get("status") != "PASS"
        or not guardrail_report_path.is_relative_to(guardrail_root)
        or not guardrail_report_path.is_file()
        or hashlib.sha256(guardrail_report_path.read_bytes()).hexdigest()
        != guardrail_pointer.get("report_sha256")
    ):
        raise HTTPException(
            status_code=409, detail="deterministic business guardrail proof failed"
        )
    guardrail_report = json.loads(
        guardrail_report_path.read_text(encoding="utf-8")
    )
    return JSONResponse(
        {
            "schema": "port-dt-integrated-business-api.v1",
            "status": champion["status"],
            "report_sha256": report_sha256,
            "selected_job_id": champion.get("selected_job_id"),
            "selected_model_sha256": champion.get("selected_model_sha256"),
            "business_score": champion.get("business_score"),
            "training_dataset": verified_datasets["training"],
            "forward_dataset": verified_datasets["forward"],
            "guardrail_replay": {
                "status": guardrail_pointer.get("status"),
                "report_sha256": guardrail_pointer.get("report_sha256"),
                "rows_replayed": guardrail_pointer.get("rows_replayed"),
                "challenge_checks": (
                    guardrail_report.get("challenge_suite") or {}
                ).get("checks"),
            },
            "production_authority": False,
            "report": report,
        }
    )


@router.get("/coordinated-business/evidence")
async def coordinated_business_evidence() -> JSONResponse:
    """Return the hash-bound V6 coordinated offline champion, if admitted."""
    champion_path = COORDINATED_EVIDENCE_ROOT / "offline_champion.json"
    if not champion_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="coordinated business champion is unavailable or not admitted",
        )
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    if (
        champion.get("status") != "ADMITTED_OFFLINE_CHAMPION"
        or champion.get("production_authority") is not False
    ):
        raise HTTPException(status_code=409, detail="coordinated champion is blocked")

    evidence_root = COORDINATED_EVIDENCE_ROOT.resolve()
    report_path = (REPO_ROOT / str(champion.get("report_path") or "")).resolve()
    if not report_path.is_relative_to(evidence_root) or not report_path.is_file():
        raise HTTPException(status_code=409, detail="coordinated evidence pointer is invalid")
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if report_sha256 != champion.get("report_sha256"):
        raise HTTPException(status_code=409, detail="coordinated evidence hash gate failed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    admission = report.get("admission") or {}
    training = report.get("training") or {}
    if (
        admission.get("passed") is not True
        or admission.get("promoted") is not True
        or admission.get("production_authority") is not False
        or training.get("selected_job_id") != champion.get("selected_job_id")
    ):
        raise HTTPException(status_code=409, detail="coordinated admission binding failed")

    model_path = (REPO_ROOT / str(champion.get("selected_model_path") or "")).resolve()
    model_root = (REPO_ROOT / "data/rl/runs").resolve()
    if (
        not model_path.is_relative_to(model_root)
        or not model_path.is_file()
        or hashlib.sha256(model_path.read_bytes()).hexdigest()
        != champion.get("selected_model_sha256")
    ):
        raise HTTPException(status_code=409, detail="coordinated model hash gate failed")

    verified_datasets: Dict[str, Dict[str, Any]] = {}
    dataset_root = (REPO_ROOT / "data/rl/datasets").resolve()
    for label, report_key, id_key, sha_key in (
        ("training", "dataset", "dataset_id", "dataset_sha256"),
        (
            "forward",
            "final_evaluation_dataset",
            "final_evaluation_dataset_id",
            "final_evaluation_dataset_sha256",
        ),
    ):
        dataset_evidence = report.get(report_key) or {}
        artifact_path = (REPO_ROOT / str(dataset_evidence.get("artifact") or "")).resolve()
        expected_sha256 = champion.get(sha_key)
        if (
            dataset_evidence.get("dataset_id") != champion.get(id_key)
            or dataset_evidence.get("sha256") != expected_sha256
            or not artifact_path.is_relative_to(dataset_root)
            or not artifact_path.is_file()
            or hashlib.sha256(artifact_path.read_bytes()).hexdigest() != expected_sha256
        ):
            raise HTTPException(
                status_code=409,
                detail=f"coordinated {label} dataset hash gate failed",
            )
        verified_datasets[label] = {
            "dataset_id": champion.get(id_key),
            "sha256": expected_sha256,
            "rows": int(dataset_evidence.get("rows") or 0),
        }

    legacy = dict(report.get("legacy_preservation") or {})
    report["legacy_preservation"] = {
        "checked_artifact_count": int(legacy.get("checked_artifact_count") or 0),
        "preserved": legacy.get("preserved") is True,
    }
    return JSONResponse(
        {
            "schema": "port-dt-coordinated-business-api.v1",
            "status": champion["status"],
            "report_sha256": report_sha256,
            "selected_job_id": champion.get("selected_job_id"),
            "selected_model_sha256": champion.get("selected_model_sha256"),
            "business_score_vs_fcfs": champion.get("business_score_vs_fcfs"),
            "business_score_vs_fixed_rule": champion.get(
                "business_score_vs_fixed_rule"
            ),
            "champion_metrics": champion.get("champion_metrics"),
            "training_dataset": verified_datasets["training"],
            "forward_dataset": verified_datasets["forward"],
            "production_authority": False,
            "report": report,
        }
    )


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
        # run_dir validates one path component and enforces root containment.
        # codeql[py/path-injection]
        config = json.loads((TRAINING_MANAGER.run_dir(job_id) / "config.json").read_text(encoding="utf-8"))
        benchmark = TRAINING_MANAGER.benchmark_summary(config.get("dataset_id"))
        return JSONResponse(TRAINING_MANAGER.model_registry().readiness(job_id, benchmark))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"unknown model run: {job_id}") from exc


@router.post("/models/{job_id}/alias")
async def set_model_alias(job_id: str, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    try:
        # codeql[py/path-injection]
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
    # `run_dir` is already validated and both filenames are fixed constants.
    # codeql[py/path-injection]
    if not path.exists():
        raise HTTPException(status_code=404, detail="evaluation has not been run")
    # codeql[py/path-injection]
    payload = json.loads(path.read_text(encoding="utf-8"))
    # codeql[py/path-injection]
    if trace_path.exists():
        # codeql[py/path-injection]
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
