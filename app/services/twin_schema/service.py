from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = APP_ROOT / "twin_models"
SITE_CALIBRATION_EVIDENCE_SCHEMA = "site_twin_calibration_evidence.v2"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso8601(value: Any, field: str) -> None:
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be ISO-8601") from exc


class TwinSchemaService:
    def __init__(self, model_root: Path = DEFAULT_MODEL_ROOT) -> None:
        self.model_root = Path(model_root)

    def models(self) -> Dict[str, Any]:
        items = []
        for path in sorted(self.model_root.glob("*.json")):
            payload = _read_json(path)
            self.validate_model(payload)
            items.append({
                "id": payload["@id"],
                "display_name": payload.get("displayName"),
                "artifact_id": path.name,
                "sha256": _sha256(path),
                "contents": payload.get("contents", []),
            })
        ids = {item["id"] for item in items}
        for item in items:
            for content in item["contents"]:
                if content.get("@type") == "Relationship" and content.get("target") not in ids:
                    raise ValueError(f"unknown relationship target: {content.get('target')}")
        return {"schema": "DTDL-compatible JSON-LD", "models": items, "count": len(items)}

    @staticmethod
    def validate_model(payload: Dict[str, Any]) -> None:
        if payload.get("@context") != "dtmi:dtdl:context;2":
            raise ValueError("model @context must be dtmi:dtdl:context;2")
        if payload.get("@type") != "Interface" or not str(payload.get("@id", "")).startswith("dtmi:"):
            raise ValueError("model must be a DTDL Interface with a dtmi @id")
        contents = payload.get("contents")
        if not isinstance(contents, list):
            raise ValueError("model contents must be a list")
        names = [item.get("name") for item in contents if isinstance(item, dict)]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("model content names must be present and unique")

    def validate_graph(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        known_models = {item["id"] for item in self.models()["models"]}
        entities = payload.get("entities")
        relationships = payload.get("relationships")
        if not isinstance(entities, list) or not isinstance(relationships, list):
            raise ValueError("graph requires entities and relationships lists")
        errors: list[str] = []
        ids: list[str] = []
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict) or not entity.get("id"):
                errors.append(f"entity[{index}] requires id")
                continue
            ids.append(str(entity["id"]))
            if entity.get("model") not in known_models:
                errors.append(f"entity[{index}] references unknown model")
            if not isinstance(entity.get("properties"), dict):
                errors.append(f"entity[{index}] properties must be an object")
            source = entity.get("source")
            if not isinstance(source, dict) or not source.get("type") or not source.get("observed_at"):
                errors.append(f"entity[{index}] requires source.type and source.observed_at")
            else:
                try:
                    _iso8601(source["observed_at"], f"entity[{index}].source.observed_at")
                except ValueError:
                    errors.append(f"entity[{index}].source.observed_at must be ISO-8601")
        if len(ids) != len(set(ids)):
            errors.append("entity ids must be unique")
        entity_ids = set(ids)
        for index, relationship in enumerate(relationships):
            if not isinstance(relationship, dict):
                errors.append(f"relationship[{index}] must be an object")
                continue
            if relationship.get("source") not in entity_ids or relationship.get("target") not in entity_ids:
                errors.append(f"relationship[{index}] source and target must reference known entities")
            if not relationship.get("name"):
                errors.append(f"relationship[{index}] requires name")
        return {
            "valid": not errors,
            "errors": errors,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "source": "configured_graph_not_generated",
        }

    def configured_graph(self) -> Dict[str, Any]:
        raw_path = os.getenv("PORT_DT_TWIN_GRAPH_PATH", "").strip()
        if not raw_path:
            raise FileNotFoundError("PORT_DT_TWIN_GRAPH_PATH is not configured")
        path = Path(raw_path).expanduser().resolve()
        payload = _read_json(path)
        result = self.validate_graph(payload)
        if not result["valid"]:
            raise ValueError("; ".join(result["errors"]))
        return {**payload, "validation": result, "source_id": path.name, "sha256": _sha256(path)}

    @staticmethod
    def validate_calibration(payload: Dict[str, Any]) -> Dict[str, Any]:
        required = ("dataset_sha256", "model_version", "parameters", "validation_window", "metrics", "thresholds", "approved_by", "provenance")
        missing = [name for name in required if not payload.get(name)]
        errors = ["missing field: " + name for name in missing]
        window = payload.get("validation_window") or {}
        if not isinstance(window, dict) or not window.get("start_at") or not window.get("end_at"):
            errors.append("validation_window requires start_at and end_at")
        else:
            for name in ("start_at", "end_at"):
                try:
                    _iso8601(window[name], "validation_window." + name)
                except ValueError:
                    errors.append("validation_window timestamp must be ISO-8601")
        for name in ("parameters", "metrics", "thresholds", "provenance"):
            if payload.get(name) is not None and not isinstance(payload.get(name), dict):
                errors.append(name + " must be an object")
        metrics = payload.get("metrics") or {}
        thresholds = payload.get("thresholds") or {}
        threshold_checks = {}
        for name, threshold in thresholds.items():
            if name not in metrics:
                errors.append(f"threshold {name} has no measured metric")
                continue
            try:
                threshold_checks[name] = float(metrics[name]) <= float(threshold)
            except (TypeError, ValueError):
                errors.append(f"metric and threshold {name} must be numeric")
        if any(value is False for value in threshold_checks.values()):
            errors.append("one or more calibration acceptance thresholds failed")
        evidence_v2 = payload.get("schema_version") == SITE_CALIBRATION_EVIDENCE_SCHEMA
        production_gate_eligible = False
        if evidence_v2:
            for name in (
                "site_id",
                "dataset_id",
                "training_window",
                "training_rows",
                "validation_rows",
                "threshold_checks",
                "error_decomposition",
                "data_quality",
                "boundary",
                "evidence_digest",
            ):
                if payload.get(name) in (None, "", [], {}):
                    errors.append("missing field: " + name)
            if not _SHA256.fullmatch(str(payload.get("dataset_sha256") or "")):
                errors.append("dataset_sha256 must be a lowercase SHA-256 digest")
            training_window = payload.get("training_window") or {}
            try:
                train_start = datetime.fromisoformat(str(training_window["start_at"]).replace("Z", "+00:00"))
                train_end = datetime.fromisoformat(str(training_window["end_at"]).replace("Z", "+00:00"))
                validation_start = datetime.fromisoformat(str(window["start_at"]).replace("Z", "+00:00"))
                validation_end = datetime.fromisoformat(str(window["end_at"]).replace("Z", "+00:00"))
                if train_start.tzinfo is None or train_end.tzinfo is None or validation_start.tzinfo is None or validation_end.tzinfo is None:
                    raise ValueError("timezone missing")
                if train_end >= validation_start or train_end <= train_start or validation_end <= validation_start:
                    errors.append("training and validation windows must be chronological and non-overlapping")
            except (KeyError, TypeError, ValueError):
                errors.append("training_window and validation_window must contain timezone-aware timestamps")
            try:
                if int(payload.get("training_rows") or 0) < 48:
                    errors.append("training_rows must be at least 48")
                if int(payload.get("validation_rows") or 0) < 24:
                    errors.append("validation_rows must be at least 24")
            except (TypeError, ValueError):
                errors.append("training_rows and validation_rows must be integers")
            provided_checks = payload.get("threshold_checks") or {}
            if provided_checks != threshold_checks:
                errors.append("threshold_checks do not match metrics and thresholds")
            expected_status = "pass" if threshold_checks and all(threshold_checks.values()) else "fail"
            if payload.get("validation_status") != expected_status:
                errors.append("validation_status does not match threshold results")
            digest_payload = dict(payload)
            provided_digest = str(digest_payload.pop("evidence_digest", ""))
            canonical = json.dumps(
                digest_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != provided_digest:
                errors.append("evidence_digest does not match calibration content")
            boundary = payload.get("boundary") or {}
            if boundary.get("dispatch_allowed") is not False or boundary.get("production_authority") is not False:
                errors.append("calibration evidence cannot grant dispatch or production authority")
            source = payload.get("source") or {}
            provenance = payload.get("provenance") or {}
            approved_contract = bool(
                payload.get("approved") is True
                and payload.get("measured_outcomes") is True
                and payload.get("approved_by")
                and source.get("evidence_class") == "authorized_site_export"
                and provenance.get("change_ticket")
                and str(payload.get("approved_by")).strip().lower()
                != str(source.get("owner") or "").strip().lower()
                and boundary.get("site_calibrated") is True
                and boundary.get("measured_outcomes_verified") is True
                and boundary.get("independent_validation_passed") is True
                and boundary.get("independent_approval_verified") is True
            )
            if payload.get("approved") is True and not approved_contract:
                errors.append("approved calibration requires authorized measured outcomes, change ticket and approval boundary")
            production_gate_eligible = bool(not errors and approved_contract)
        return {
            "valid": not errors,
            "errors": errors,
            "threshold_checks": threshold_checks,
            "evidence_type": "site_calibration_evidence_v2" if evidence_v2 else "legacy_site_supplied_calibration",
            "production_gate_eligible": production_gate_eligible,
        }

    def configured_calibration(self) -> Dict[str, Any]:
        raw_path = os.getenv("PORT_DT_TWIN_CALIBRATION_PATH", "").strip()
        if not raw_path:
            raise FileNotFoundError("PORT_DT_TWIN_CALIBRATION_PATH is not configured")
        path = Path(raw_path).expanduser().resolve()
        payload = _read_json(path)
        validation = self.validate_calibration(payload)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        return {**payload, "validation": validation, "source_id": path.name, "sha256": _sha256(path)}
