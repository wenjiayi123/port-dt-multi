"""Pinned public-replay evaluation with durable, recoverable receipts."""

from datetime import datetime, timezone
from app.services.durable_file import publish
from app.services.file_lock import exclusive_lock
import hashlib
import json
import os
from pathlib import Path
import re
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    publish(temporary, path)


class StaleContext(ValueError):
    pass


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,100}$")
    context_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    scenario: str = Field(default="high_density_berthing", max_length=80)
    horizon_min: int = Field(default=60, ge=1, le=360)
    step_min: int = Field(default=15, ge=1, le=60)


class AgentRuntimeReceipts:
    def __init__(self, runtime, telemetry, root, *, ttl=300, clock=time.time):
        self.runtime, self.telemetry = runtime, telemetry
        self.root, self.ttl, self.clock = Path(root), ttl, clock

    def version(self):
        state = self.runtime.status()
        if state.get("available") is not True:
            raise ValueError("native runtime policy unavailable")
        model = state.get("model", {})
        source = self.telemetry.source_status()
        if not model.get("loaded_model_sha256") or not source.get("sha256"):
            raise ValueError("loaded model/data identity missing")
        fields = {
            "loaded_model_sha256": model["loaded_model_sha256"],
            "declared_model_sha256": model.get("model_sha256"),
            "config_sha256": model.get("config_sha256"),
            "policy_dataset_sha256": model.get("dataset_sha256"),
            "telemetry_sha256": source["sha256"],
        }
        return {"source_version": digest(fields), **fields}

    def capture(self):
        before = self.version()
        snapshot = self.telemetry.agent_snapshot_context()
        if before != self.version():
            raise StaleContext("source changed while capturing context")
        body = {
            "schema": "agent-port-context.v1",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "created_unix": self.clock(),
            "max_age_seconds": self.ttl,
            "source_context": before,
            "snapshot": snapshot,
            "production_authority": False,
            "scope": "captured calibrated public replay, not measured terminal telemetry",
        }
        key = digest(body)
        save(self.root / "contexts" / (key + ".json"), body)
        return {"context_id": key, **body}

    def context(self, context_id, *, require_current=True):
        if not re.fullmatch(r"[a-f0-9]{64}", context_id):
            raise ValueError("invalid context identifier")
        path = self.root / "contexts" / (context_id + ".json")
        if not path.is_file():
            raise StaleContext("context missing; capture a new observation")
        value = json.loads(path.read_text())
        if digest(value) != context_id:
            raise ValueError("context checksum mismatch")
        if require_current and (
            self.clock() - value["created_unix"] > value["max_age_seconds"]
            or self.clock() < value["created_unix"]
            or value["source_context"] != self.version()
        ):
            raise StaleContext(
                "context expired or source changed; capture a new observation"
            )
        return value

    def get(self, request_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", request_id):
            raise ValueError("invalid request identifier")
        path = self.root / "receipts" / (request_id + ".json")
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
        checksum = value.pop("receipt_sha256")
        if digest(value) != checksum:
            raise ValueError("receipt checksum mismatch")
        return {**value, "receipt_sha256": checksum}

    def evaluate(self, payload):
        self.root.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(self.root / "calculation.lock"):
            existing = self.get(payload["request_id"])
            if existing:
                if existing["request_sha256"] != digest(payload):
                    raise StaleContext(
                        "request identifier already bound to a different calculation"
                    )
                return {"duplicate": True, "receipt": existing}
            context = self.context(payload["context_id"])
            allowed = {
                r["id"]
                for r in self.runtime.coverage()["scenarios"]
                if r["state"] != "contract_only"
            }
            if payload["scenario"] not in allowed:
                raise ValueError("unregistered calculation scenario")
            raw = self.runtime.series(
                horizon_min=payload["horizon_min"],
                step_min=payload["step_min"],
                scenario=payload["scenario"],
                replay_anchor_position=context["snapshot"]["replay_anchor_position"],
            )
            if (
                raw.get("available") is not True
                or raw.get("production_authority") is not False
            ):
                raise ValueError(
                    "native calculation unavailable or authority contract invalid"
                )
            self.context(payload["context_id"])
            if (
                raw.get("policy", {}).get("loaded_model_sha256")
                != context["source_context"]["loaded_model_sha256"]
            ):
                raise StaleContext("calculation used a different loaded policy")
            receipt = {
                "schema": "agent-port-evaluation.v1",
                "request_id": payload["request_id"],
                "request_sha256": digest(payload),
                "parameters": payload,
                "context_id": payload["context_id"],
                "source_context": context["source_context"],
                "replay_anchor_position": context["snapshot"]["replay_anchor_position"],
                "observation_captured_at": context["captured_at"],
                "raw": raw,
                "production_authority": False,
                "time_basis": "pinned replay projection uses a fixed computational clock, not wall-clock predictions",
            }
            receipt["receipt_sha256"] = digest(receipt)
            save(self.root / "receipts" / (payload["request_id"] + ".json"), receipt)
            return {"duplicate": False, "receipt": receipt}


def build_router(service):
    router = APIRouter(prefix="/api/v3/runtime/agent", tags=["agent-runtime-receipts"])

    def local(request):
        if (
            not request.client
            or request.client.host not in {"127.0.0.1", "::1", "localhost"}
            or request.headers.get("origin")
        ):
            raise HTTPException(403, "local backend tool calls only")

    @router.get("/context")
    def capture(request: Request):
        local(request)
        try:
            return service.capture()
        except StaleContext as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.post("/evaluations")
    def evaluate(payload: EvaluationRequest, request: Request):
        local(request)
        try:
            return service.evaluate(payload.model_dump())
        except StaleContext as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/evaluations/{request_id}")
    def read(request_id: str, request: Request):
        local(request)
        try:
            result = service.get(request_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if result is None:
            raise HTTPException(404, "completed receipt not found")
        try:
            service.context(result["context_id"])
            current = True
        except StaleContext:
            current = False
        return {"receipt": result, "context_current": current, "readback_only": True}

    return router
