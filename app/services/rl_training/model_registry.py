from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .datasets import file_sha256
from .identifiers import resolve_child_dir, validate_identifier


ALIASES = {"candidate", "champion", "archive"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ModelRegistry:
    """File-backed local registry with aliases, evidence gates and audit log."""

    def __init__(self, run_root: Path, registry_path: Path) -> None:
        self.run_root = Path(run_root)
        self.registry_path = Path(registry_path)
        self.audit_path = self.registry_path.with_name("model_registry_audit.jsonl")

    def _state(self) -> Dict[str, Any]:
        return _read(self.registry_path, {"schema_version": 1, "models": {}, "aliases": {}})

    def _run_dir(self, job_id: str) -> Path:
        return resolve_child_dir(self.run_root, job_id, field="job_id")

    def _audit(self, event: Dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": _now(), **event}, ensure_ascii=False) + "\n")

    def sync(self, job_id: str) -> Dict[str, Any]:
        job_id = validate_identifier(job_id, field="job_id")
        run_dir = self._run_dir(job_id)
        status = _read(run_dir / "status.json", {})
        config = _read(run_dir / "config.json", {})
        manifest = _read(run_dir / "manifest.json", {})
        evaluation = _read(run_dir / "evaluation.json", None)
        if not status or not config:
            raise KeyError(job_id)
        model_path = run_dir / "model.zip"
        actual_model_sha = file_sha256(model_path) if model_path.exists() else None
        expected_model_sha = manifest.get("model_sha256")
        artifact_verified = (
            actual_model_sha == expected_model_sha
            if expected_model_sha
            else config.get("algorithm") == "mpc" and manifest.get("controller_only") is True
        )
        record = {
            "job_id": job_id,
            "algorithm": config.get("algorithm"),
            "dataset_id": config.get("dataset_id"),
            "dataset_sha256": config.get("dataset_fingerprint"),
            "seed": config.get("seed"),
            "status": status.get("status"),
            "created_at": status.get("created_at"),
            "updated_at": _now(),
            "implementation": manifest.get("implementation"),
            "artifact": {
                "artifact_id": "model.zip" if model_path.exists() else ("controller-manifest" if config.get("algorithm") == "mpc" else None),
                "expected_sha256": expected_model_sha,
                "actual_sha256": actual_model_sha,
                "verified": artifact_verified,
                "controller_only": bool(manifest.get("controller_only")),
            },
            "evaluation": {
                "available": bool(evaluation),
                "episodes": (evaluation or {}).get("episodes"),
                "metrics": (evaluation or {}).get("metrics"),
                "uncertainty": (evaluation or {}).get("uncertainty"),
                "evaluated_at": (evaluation or {}).get("evaluated_at"),
            },
        }
        state = self._state()
        state.setdefault("models", {})[job_id] = record
        state["updated_at"] = _now()
        _write(self.registry_path, state)
        self._write_model_card(run_dir, record, manifest)
        return {**record, "aliases": sorted(name for name, target in state.get("aliases", {}).items() if target == job_id)}

    def _write_model_card(self, run_dir: Path, record: Dict[str, Any], manifest: Dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "model": record,
            "intended_use": "offline port operations decision support and held-out evaluation",
            "not_intended_use": [
                "autonomous actuator dispatch",
                "safety-critical control without site validation",
                "performance claims outside the recorded dataset and configuration",
            ],
            "training_data": manifest.get("split"),
            "limitations": [
                "public example data includes explicitly derived engineering fields",
                "software guardrails do not replace PLC/BMS interlocks or operator authority",
                "results require multiple seeds before comparative claims",
            ],
            "generated_at": _now(),
        }
        _write(run_dir / "model_card.json", payload)
        evaluation = record["evaluation"]
        text = (
            f"# Model card: {record['job_id']}\n\n"
            f"- Algorithm: `{record.get('algorithm')}`\n"
            f"- Implementation: `{record.get('implementation')}`\n"
            f"- Dataset: `{record.get('dataset_id')}` (`{record.get('dataset_sha256')}`)\n"
            f"- Seed: `{record.get('seed')}`\n"
            f"- Artifact verified: `{record['artifact']['verified']}`\n"
            f"- Held-out evaluation episodes: `{evaluation.get('episodes') or 0}`\n\n"
            "## Intended use\n\nOffline decision support, reproducible research and integration testing.\n\n"
            "## Prohibited claim\n\nThis artifact is not approved for autonomous equipment control.\n"
        )
        (run_dir / "MODEL_CARD.md").write_text(text, encoding="utf-8")

    def list(self) -> Dict[str, Any]:
        state = self._state()
        items = []
        for job_id, record in state.get("models", {}).items():
            items.append({**self._public_record(record), "aliases": sorted(name for name, target in state.get("aliases", {}).items() if target == job_id)})
        items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return {"models": items, "aliases": state.get("aliases", {}), "count": len(items), "updated_at": state.get("updated_at")}

    def get(self, job_id: str) -> Dict[str, Any]:
        job_id = validate_identifier(job_id, field="job_id")
        state = self._state()
        record = state.get("models", {}).get(job_id)
        if not record:
            raise KeyError(job_id)
        return {**self._public_record(record), "aliases": sorted(name for name, target in state.get("aliases", {}).items() if target == job_id)}

    @staticmethod
    def _public_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Remove absolute paths left by older local registry versions."""
        public = json.loads(json.dumps(record))
        artifact = public.get("artifact") or {}
        legacy_path = artifact.pop("path", None)
        if legacy_path and not artifact.get("artifact_id"):
            artifact["artifact_id"] = Path(str(legacy_path)).name
        public["artifact"] = artifact
        return public

    def refresh(self) -> Dict[str, Any]:
        synced = []
        errors = []
        for path in sorted(self.run_root.glob("*/status.json")):
            try:
                synced.append(self.sync(path.parent.name)["job_id"])
            except Exception as exc:
                errors.append({"job_id": path.parent.name, "error": str(exc)})
        return {"synced": synced, "errors": errors, "count": len(synced)}

    def readiness(self, job_id: str, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        job_id = validate_identifier(job_id, field="job_id")
        record = self.get(job_id)
        blockers: list[str] = []
        warnings: list[str] = []
        evaluation = record["evaluation"]
        if not record["artifact"]["verified"]:
            blockers.append("artifact checksum is missing or does not match")
        if not evaluation["available"]:
            blockers.append("chronological holdout evaluation is missing")
        if int(evaluation.get("episodes") or 0) < 5:
            blockers.append("at least 5 held-out evaluation episodes are required")
        if not evaluation.get("uncertainty"):
            blockers.append("evaluation uncertainty intervals are missing")
        group = next((item for item in benchmark.get("algorithms", []) if item.get("id") == record.get("algorithm")), {})
        if record.get("algorithm") != "mpc" and not group.get("multi_seed_ready"):
            blockers.append("at least 3 distinct evaluated seeds are required")
        violation_rate = (evaluation.get("metrics") or {}).get("guardrail_violation_rate")
        threshold = float(os.getenv("PORT_DT_MAX_GUARDRAIL_VIOLATION_RATE", "0.05"))
        if violation_rate is None:
            blockers.append("guardrail violation rate is missing")
        elif float(violation_rate) > threshold:
            blockers.append(f"guardrail violation rate exceeds {threshold:.3f}")
        manifest = _read(self._run_dir(job_id) / "manifest.json", {})
        split = manifest.get("split") or {}
        quality = split.get("quality") or {}
        if quality.get("training_eligible") is not True:
            blockers.append("training dataset quality gate evidence is missing or failed")
        if quality.get("status") == "warn":
            warnings.extend(quality.get("warnings") or [])
        enabled = os.getenv("PORT_DT_ALLOW_MODEL_PROMOTION", "").strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            blockers.append("model promotion is disabled by PORT_DT_ALLOW_MODEL_PROMOTION")
        return {
            "job_id": job_id,
            "ready_for_champion_alias": not blockers,
            "production_deployment_approved": False,
            "blockers": blockers,
            "warnings": warnings,
            "criteria": {
                "minimum_evaluation_episodes": 5,
                "minimum_distinct_seeds": 0 if record.get("algorithm") == "mpc" else 3,
                "deterministic_controller_seed_exception": record.get("algorithm") == "mpc",
                "maximum_guardrail_violation_rate": threshold,
                "explicit_promotion_opt_in": enabled,
            },
        }

    def set_alias(self, job_id: str, alias: str, *, approved_by: str, reason: str, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        job_id = validate_identifier(job_id, field="job_id")
        alias = str(alias).strip().lower()
        if alias not in ALIASES:
            raise ValueError("alias must be candidate, champion, or archive")
        if not approved_by.strip() or not reason.strip():
            raise ValueError("approved_by and reason are required")
        self.get(job_id)
        readiness = self.readiness(job_id, benchmark)
        if alias == "champion" and not readiness["ready_for_champion_alias"]:
            raise ValueError("promotion blocked: " + "; ".join(readiness["blockers"]))
        state = self._state()
        previous = state.setdefault("aliases", {}).get(alias)
        if alias == "champion" and previous and previous != job_id:
            state["aliases"]["rollback"] = previous
        state["aliases"][alias] = job_id
        state["updated_at"] = _now()
        _write(self.registry_path, state)
        self._audit({"event": "alias_set", "alias": alias, "job_id": job_id, "previous": previous, "approved_by": approved_by, "reason": reason})
        return {"alias": alias, "job_id": job_id, "previous": previous, "rollback": state["aliases"].get("rollback"), "readiness": readiness}

    def rollback(self, *, approved_by: str, reason: str, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        state = self._state()
        target = state.get("aliases", {}).get("rollback")
        if not target:
            raise ValueError("no rollback alias is available")
        return self.set_alias(target, "champion", approved_by=approved_by, reason=reason, benchmark=benchmark)

    def rollback_target(self) -> Optional[str]:
        target = self._state().get("aliases", {}).get("rollback")
        return validate_identifier(target, field="job_id") if target else None
