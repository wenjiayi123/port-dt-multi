from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATASET_SCHEMA = "site_execution_commissioning_dataset.v1"
EVIDENCE_SCHEMA = "site_execution_acceptance_evidence.v1"
ACTUATOR_CONFIG_SCHEMA = "site_actuator_config.v2"
EVIDENCE_CLASSES = ("authorized_site_commissioning_export", "contract_test_only")
REQUIRED_SCENARIOS = (
    "safe_command_readback",
    "out_of_bounds_block",
    "expired_command_block",
    "duplicate_command_block",
    "lost_acknowledgement_block",
    "independent_interlock_trip",
    "emergency_stop",
    "rollback_restore",
)
BLOCK_ONLY_SCENARIOS = {
    "out_of_bounds_block",
    "expired_command_block",
    "duplicate_command_block",
    "lost_acknowledgement_block",
    "independent_interlock_trip",
    "emergency_stop",
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PLACEHOLDERS = {"unknown", "unset", "todo", "replace", "replace_me", "n/a", "none"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _canonical_digest(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _parse_timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(_clean(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone is required")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


class SiteExecutionAcceptanceService:
    """Validate actuator configuration and site commissioning evidence.

    The service never dispatches a command. Browser input can only exercise the
    contract. Production binding additionally requires an authorized source,
    three independent approvals, a fixed commissioning scenario matrix, and
    an exact digest match to the configured actuator file.
    """

    @staticmethod
    def validate_actuator_config(payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {
                "valid": False,
                "errors": ["actuator config must be an object"],
                "production_gate_eligible": False,
                "config_sha256": None,
            }
        if payload.get("schema_version") != ACTUATOR_CONFIG_SCHEMA:
            errors.append(f"schema_version must equal {ACTUATOR_CONFIG_SCHEMA}")
        site_id = _clean(payload.get("site_id"))
        mode = _clean(payload.get("mode"))
        if not _IDENTIFIER.fullmatch(site_id) or "replace" in site_id.lower():
            errors.append("site_id must be a stable identifier")
        if mode not in {"contract_test", "authorized_site"}:
            errors.append("mode must be contract_test or authorized_site")
        whitelist = payload.get("whitelist") if isinstance(payload.get("whitelist"), dict) else {}
        routing = payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
        asset_routes = routing.get("asset") if isinstance(routing.get("asset"), dict) else {}
        constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
        asset_constraints = constraints.get("asset") if isinstance(constraints.get("asset"), dict) else {}
        if not whitelist:
            errors.append("whitelist must contain at least one asset")
        permitted_channels = {"dry_run"} if mode == "contract_test" else {"http", "opcua", "modbus"}
        for asset_id, actions in whitelist.items():
            if not _IDENTIFIER.fullmatch(_clean(asset_id)):
                errors.append(f"whitelist asset {asset_id} is not a stable identifier")
                continue
            if not isinstance(actions, list) or not actions or any(not _IDENTIFIER.fullmatch(_clean(action)) for action in actions):
                errors.append(f"whitelist asset {asset_id} requires stable action identifiers")
                continue
            if len(actions) != len(set(actions)):
                errors.append(f"whitelist asset {asset_id} contains duplicate actions")
            route = asset_routes.get(asset_id) if isinstance(asset_routes.get(asset_id), dict) else {}
            channel = _clean(route.get("channel")).lower()
            if channel not in permitted_channels:
                errors.append(f"asset {asset_id} requires a supported exact route; channel {channel or 'missing'} is not eligible")
            interlock_id = _clean(route.get("interlock_id"))
            if not _IDENTIFIER.fullmatch(interlock_id) or "replace" in interlock_id.lower():
                errors.append(f"asset {asset_id} route requires an independent interlock_id")
            readback_mode = route.get("readback_mode")
            allowed_readback_modes = (
                {"contract_receipt"}
                if mode == "contract_test"
                else {"signed_gateway_receipt"}
                if channel == "http"
                else {"independent_read"}
            )
            if readback_mode not in allowed_readback_modes:
                errors.append(
                    f"asset {asset_id} route requires a mode- and channel-appropriate readback_mode"
                )
            if route.get("rollback_supported") is not True:
                errors.append(f"asset {asset_id} route must declare rollback_supported=true")
            endpoint = _clean(route.get("endpoint"))
            if mode == "authorized_site":
                if channel == "http":
                    parsed = urlparse(endpoint)
                    if (
                        parsed.scheme != "https"
                        or not parsed.netloc
                        or parsed.hostname in {"localhost", "127.0.0.1"}
                        or str(parsed.hostname or "").endswith(".invalid")
                    ):
                        errors.append(f"asset {asset_id} HTTP endpoint must be a non-local HTTPS URL")
                elif channel == "opcua" and not endpoint.startswith("opc.tcp://"):
                    errors.append(f"asset {asset_id} OPC UA endpoint must use opc.tcp")
                elif channel == "modbus" and (not endpoint or endpoint in {"localhost", "127.0.0.1"}):
                    errors.append(f"asset {asset_id} Modbus endpoint must be a non-local host")
            rules_by_action = asset_constraints.get(asset_id) if isinstance(asset_constraints.get(asset_id), dict) else {}
            for action in actions:
                rules = rules_by_action.get(action) if isinstance(rules_by_action.get(action), dict) else {}
                if not rules:
                    errors.append(f"asset {asset_id} action {action} requires parameter constraints")
                    continue
                for parameter, rule in rules.items():
                    if not _IDENTIFIER.fullmatch(_clean(parameter)) or not isinstance(rule, dict):
                        errors.append(f"asset {asset_id} action {action} has an invalid constraint rule")
                        continue
                    if rule.get("required") is not True:
                        errors.append(f"asset {asset_id} action {action} parameter {parameter} must be required")
                    allowed = rule.get("allowed")
                    has_range = rule.get("min") is not None and rule.get("max") is not None
                    if not has_range and not (isinstance(allowed, list) and allowed):
                        errors.append(f"asset {asset_id} action {action} parameter {parameter} needs min/max or allowed values")
                    if has_range:
                        try:
                            minimum = _finite(rule["min"])
                            maximum = _finite(rule["max"])
                            if maximum <= minimum:
                                errors.append(f"asset {asset_id} action {action} parameter {parameter} has an invalid range")
                        except (TypeError, ValueError):
                            errors.append(f"asset {asset_id} action {action} parameter {parameter} range must be finite")
        security = payload.get("security") if isinstance(payload.get("security"), dict) else {}
        for flag in (
            "require_two_channel",
            "require_constraints",
            "require_verified_readback",
            "require_independent_interlock",
            "require_rollback",
        ):
            if security.get(flag) is not True:
                errors.append(f"security.{flag} must be true")
        token_env = _clean(security.get("confirmation_token_env"))
        if not _ENV_NAME.fullmatch(token_env):
            errors.append("security.confirmation_token_env must be a stable environment-variable name")
        try:
            ttl = int(security.get("command_ttl_seconds"))
            if not 15 <= ttl <= 300:
                raise ValueError("ttl outside bounds")
        except (TypeError, ValueError):
            errors.append("security.command_ttl_seconds must be between 15 and 300")
        config_sha256 = _canonical_digest(payload)
        production_gate_eligible = bool(
            not errors
            and mode == "authorized_site"
            and payload.get("enabled") is True
        )
        return {
            "valid": not errors,
            "errors": errors,
            "production_gate_eligible": production_gate_eligible,
            "config_sha256": config_sha256,
            "site_id": site_id or None,
            "mode": mode or None,
            "asset_count": len(whitelist),
            "action_count": sum(len(value) for value in whitelist.values() if isinstance(value, list)),
        }

    @staticmethod
    def validate_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {"valid": False, "errors": ["execution evidence must be an object"], "production_gate_eligible": False}
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        for field in (
            "site_id", "run_id", "dataset_sha256", "actuator_config_sha256",
            "shadow_acceptance_digest", "source", "tests", "scenario_counts",
            "metrics", "acceptance_status", "approved_by", "provenance",
            "boundary", "evidence_digest",
        ):
            if payload.get(field) in (None, "", [], {}):
                errors.append("missing field: " + field)
        for field in ("dataset_sha256", "actuator_config_sha256", "shadow_acceptance_digest"):
            if not _SHA256.fullmatch(_clean(payload.get(field))):
                errors.append(f"{field} must be a lowercase SHA-256 digest")
        tests = payload.get("tests") if isinstance(payload.get("tests"), list) else []
        test_ids: set[str] = set()
        counts: Counter[str] = Counter()
        for index, row in enumerate(tests):
            if not isinstance(row, dict):
                errors.append(f"tests[{index}] must be an object")
                continue
            test_id = _clean(row.get("test_id"))
            scenario = _clean(row.get("scenario"))
            if not _IDENTIFIER.fullmatch(test_id) or test_id in test_ids:
                errors.append(f"tests[{index}].test_id must be stable and unique")
            test_ids.add(test_id)
            if scenario not in REQUIRED_SCENARIOS:
                errors.append(f"tests[{index}].scenario is not in the fixed matrix")
            counts[scenario] += 1
            if row.get("passed") is not True:
                errors.append(f"tests[{index}] did not pass")
            if scenario in BLOCK_ONLY_SCENARIOS and row.get("equipment_command_sent") is not False:
                errors.append(f"tests[{index}] block scenario sent an equipment command")
            if scenario == "safe_command_readback" and row.get("readback_verified") is not True:
                errors.append(f"tests[{index}] safe command lacks verified readback")
            if scenario == "rollback_restore" and (
                row.get("rollback_verified") is not True or row.get("readback_verified") is not True
            ):
                errors.append(f"tests[{index}] rollback lacks restoration readback")
            if scenario in {"independent_interlock_trip", "emergency_stop"} and row.get("interlock_verified") is not True:
                errors.append(f"tests[{index}] lacks independent interlock evidence")
            if not _SHA256.fullmatch(_clean(row.get("source_reference_sha256"))):
                errors.append(f"tests[{index}] source reference digest is invalid")
            if row.get("requester_confirmer_distinct") is not True:
                errors.append(f"tests[{index}] does not prove separate requester and confirmer")
        missing_scenarios = sorted(set(REQUIRED_SCENARIOS) - set(counts))
        if missing_scenarios:
            errors.append("missing fixed commissioning scenarios: " + ", ".join(missing_scenarios))
        repeated_scenarios = sorted(name for name, count in counts.items() if count != 1)
        if repeated_scenarios:
            errors.append("fixed commissioning matrix requires exactly one record per scenario: " + ", ".join(repeated_scenarios))
        supplied_counts = payload.get("scenario_counts") if isinstance(payload.get("scenario_counts"), dict) else {}
        if supplied_counts != dict(sorted(counts.items())):
            errors.append("scenario_counts does not match test records")
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        derived_metrics = {
            "scenario_pass_rate": float(sum(row.get("passed") is True for row in tests) / len(tests)) if tests else 0.0,
            "unsafe_command_execution_count": sum(
                row.get("equipment_command_sent") is not False
                for row in tests if row.get("scenario") in BLOCK_ONLY_SCENARIOS
            ),
            "readback_coverage_rate": float(sum(
                row.get("readback_verified") is True
                for row in tests if row.get("scenario") in {"safe_command_readback", "rollback_restore"}
            ) / 2.0),
            "interlock_coverage_rate": float(sum(
                row.get("interlock_verified") is True
                for row in tests if row.get("scenario") in {"independent_interlock_trip", "emergency_stop"}
            ) / 2.0),
            "rollback_success_rate": float(sum(
                row.get("rollback_verified") is True
                for row in tests if row.get("scenario") == "rollback_restore"
            )),
        }
        for name, value in derived_metrics.items():
            try:
                if not math.isclose(float(metrics.get(name)), value, rel_tol=1e-9, abs_tol=1e-9):
                    errors.append(f"metric {name} does not match test records")
            except (TypeError, ValueError):
                errors.append(f"metric {name} must be numeric")
        expected_status = "pass" if (
            not missing_scenarios
            and derived_metrics["scenario_pass_rate"] == 1.0
            and derived_metrics["unsafe_command_execution_count"] == 0
            and derived_metrics["readback_coverage_rate"] == 1.0
            and derived_metrics["interlock_coverage_rate"] == 1.0
            and derived_metrics["rollback_success_rate"] >= 1.0
        ) else "fail"
        if payload.get("acceptance_status") != expected_status:
            errors.append("acceptance_status does not match the fixed commissioning matrix")
        digest_payload = dict(payload)
        supplied_digest = _clean(digest_payload.pop("evidence_digest", ""))
        if _canonical_digest(digest_payload) != supplied_digest:
            errors.append("evidence_digest does not match execution evidence content")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        approvers = payload.get("approved_by") if isinstance(payload.get("approved_by"), dict) else {}
        reviewer_values = [_clean(approvers.get(name)) for name in ("operations", "maritime_safety", "controls_engineering")]
        owner = _clean(source.get("owner")).lower()
        approval_contract = bool(
            payload.get("approved") is True
            and payload.get("site_commissioning_verified") is True
            and source.get("evidence_class") == "authorized_site_commissioning_export"
            and provenance.get("source_attestation") is True
            and provenance.get("change_ticket")
            and provenance.get("actuator_config_production_eligible") is True
            and all(reviewer_values)
            and len({value.lower() for value in reviewer_values}) == 3
            and all(value.lower() != owner for value in reviewer_values)
            and expected_status == "pass"
            and boundary.get("site_execution_accepted") is True
            and boundary.get("production_authority") is False
            and boundary.get("direct_model_dispatch") is False
        )
        if payload.get("approved") is True and not approval_contract:
            errors.append("approved execution evidence requires authorized commissioning, an eligible config, three independent reviewers and a change ticket")
        return {
            "valid": not errors,
            "errors": errors,
            "production_gate_eligible": bool(not errors and approval_contract),
            "metrics": derived_metrics,
            "scenario_counts": dict(sorted(counts.items())),
        }

    def readiness(self) -> Dict[str, Any]:
        config_path_text = _clean(os.getenv("PORT_DT_ACTUATOR_CONFIG"))
        evidence_path_text = _clean(os.getenv("PORT_DT_EXECUTION_ACCEPTANCE_PATH"))
        configured: Dict[str, Any] = {
            "config_configured": bool(config_path_text),
            "evidence_configured": bool(evidence_path_text),
            "verified": False,
            "production_gate_eligible": False,
            "config_artifact_id": Path(config_path_text).name if config_path_text else None,
            "evidence_artifact_id": Path(evidence_path_text).name if evidence_path_text else None,
            "config_sha256": None,
            "evidence_sha256": None,
            "site_id": None,
            "blockers": [],
        }
        if not config_path_text:
            configured["blockers"].append("actuator_config_not_configured")
        if not evidence_path_text:
            configured["blockers"].append("execution_acceptance_artifact_not_configured")
        config_payload: Dict[str, Any] | None = None
        evidence_payload: Dict[str, Any] | None = None
        config_validation: Dict[str, Any] = {"valid": False, "production_gate_eligible": False, "errors": []}
        evidence_validation: Dict[str, Any] = {"valid": False, "production_gate_eligible": False, "errors": []}
        try:
            if config_path_text:
                config_path = Path(config_path_text).expanduser()
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                config_validation = self.validate_actuator_config(config_payload)
                configured["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
                if not config_validation["valid"]:
                    configured["blockers"].extend(config_validation["errors"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            configured["blockers"].append("actuator_config_invalid")
        try:
            if evidence_path_text:
                evidence_path = Path(evidence_path_text).expanduser()
                evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
                evidence_validation = self.validate_evidence(evidence_payload)
                configured["evidence_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                if not evidence_validation["valid"]:
                    configured["blockers"].extend(evidence_validation["errors"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            configured["blockers"].append("execution_acceptance_artifact_invalid")
        if config_payload and evidence_payload:
            configured["site_id"] = evidence_payload.get("site_id")
            if config_payload.get("site_id") != evidence_payload.get("site_id"):
                configured["blockers"].append("execution_site_id_mismatch")
            if config_validation.get("config_sha256") != evidence_payload.get("actuator_config_sha256"):
                configured["blockers"].append("actuator_config_digest_mismatch")
        eligible = bool(
            config_validation.get("production_gate_eligible")
            and evidence_validation.get("production_gate_eligible")
            and not configured["blockers"]
        )
        configured["verified"] = bool(config_validation.get("valid") and evidence_validation.get("valid") and not configured["blockers"])
        configured["production_gate_eligible"] = eligible
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "actuator_config_schema": ACTUATOR_CONFIG_SCHEMA,
            "configured_artifacts": configured,
            "contract": {
                "required_scenarios": list(REQUIRED_SCENARIOS),
                "supported_site_channels": ["HTTPS control gateway", "OPC UA with readback", "Modbus with readback"],
                "mqtt_boundary": "asynchronous receipt correlation is not implemented; MQTT is not production eligible",
                "required_config_controls": [
                    "exact asset/action allowlist",
                    "required parameter limits",
                    "short command expiry",
                    "separate confirmation channel",
                    "verified device readback",
                    "independent interlock",
                    "verified rollback",
                ],
            },
            "boundary": {
                "site_execution_accepted": eligible,
                "direct_model_dispatch": False,
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "现场执行放行证据已验证" if eligible else "待接入执行配置与现场联锁验收",
                "reason": (
                    "配置摘要、现场调试矩阵、读回、联锁、回退和三方审批均已验证；模型仍不能直接下发，实际执行必须逐单双人确认。"
                    if eligible
                    else "合同预检不能替代现场调试；必须提供授权执行配置、设备读回、独立联锁、回退和三方审批证据。"
                ),
            },
        }

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        operations_approved_by: str | None = None,
        maritime_safety_approved_by: str | None = None,
        controls_engineering_approved_by: str | None = None,
        change_ticket: str | None = None,
    ) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("schema_version") != DATASET_SCHEMA:
            errors.append({"code": "schema_version", "field": "schema_version", "message": f"must equal {DATASET_SCHEMA}"})
        site_id = _clean(payload.get("site_id"))
        run_id = _clean(payload.get("run_id"))
        for field, value in (("site_id", site_id), ("run_id", run_id)):
            if not _IDENTIFIER.fullmatch(value):
                errors.append({"code": "identifier", "field": field, "message": "stable identifier is required"})
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        evidence_class = _clean(source.get("evidence_class"))
        owner = _clean(source.get("owner"))
        timezone_name = _clean(source.get("timezone"))
        for field in ("source_system", "owner", "license", "timezone", "extracted_at", "evidence_class"):
            if not _clean(source.get(field)):
                errors.append({"code": "source_metadata", "field": f"source.{field}", "message": "field is required"})
        if owner.lower() in _PLACEHOLDERS or _clean(source.get("license")).lower() in _PLACEHOLDERS:
            errors.append({"code": "source_placeholder", "field": "source", "message": "placeholder governance metadata is rejected"})
        if evidence_class and evidence_class not in EVIDENCE_CLASSES:
            errors.append({"code": "evidence_class", "field": "source.evidence_class", "message": "unsupported evidence class"})
        try:
            ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError):
            errors.append({"code": "timezone", "field": "source.timezone", "message": "valid IANA timezone is required"})
        extracted_at: datetime | None = None
        try:
            extracted_at = _parse_timestamp(source.get("extracted_at"))
        except (TypeError, ValueError):
            errors.append({"code": "extracted_at", "field": "source.extracted_at", "message": "timezone-aware timestamp is required"})
        shadow_digest = _clean(payload.get("shadow_acceptance_digest"))
        if not _SHA256.fullmatch(shadow_digest):
            errors.append({"code": "shadow_binding", "field": "shadow_acceptance_digest", "message": "lowercase SHA-256 digest is required"})
        actuator_config = payload.get("actuator_config") if isinstance(payload.get("actuator_config"), dict) else {}
        config_validation = self.validate_actuator_config(actuator_config)
        if not config_validation["valid"]:
            errors.extend({"code": "actuator_config", "field": "actuator_config", "message": message} for message in config_validation["errors"])
        if actuator_config.get("site_id") != site_id:
            errors.append({"code": "site_binding", "field": "actuator_config.site_id", "message": "must match dataset site_id"})
        raw_tests = payload.get("tests") if isinstance(payload.get("tests"), list) else []
        records: List[Dict[str, Any]] = []
        test_ids: set[str] = set()
        for index, row in enumerate(raw_tests):
            if not isinstance(row, dict):
                errors.append({"code": "test_type", "field": "tests", "message": "test must be an object", "row_index": index})
                continue
            before = len(errors)
            test_id = _clean(row.get("test_id"))
            scenario = _clean(row.get("scenario"))
            asset_id = _clean(row.get("asset_id"))
            action = _clean(row.get("action"))
            source_reference = _clean(row.get("source_reference"))
            for field, value in (("test_id", test_id), ("asset_id", asset_id), ("action", action), ("source_reference", source_reference)):
                if not _IDENTIFIER.fullmatch(value):
                    errors.append({"code": "identifier", "field": field, "message": "stable identifier is required", "row_index": index})
            if test_id in test_ids:
                errors.append({"code": "duplicate_test", "field": "test_id", "message": "test_id must be unique", "row_index": index})
            test_ids.add(test_id)
            if scenario not in REQUIRED_SCENARIOS:
                errors.append({"code": "scenario", "field": "scenario", "message": "scenario is not in fixed commissioning matrix", "row_index": index})
            try:
                started = _parse_timestamp(row.get("started_at"))
                ended = _parse_timestamp(row.get("ended_at"))
                if ended <= started or (extracted_at and ended > extracted_at):
                    raise ValueError("invalid interval")
            except (TypeError, ValueError):
                started = ended = None
                errors.append({"code": "test_interval", "field": "started_at/ended_at", "message": "valid interval before extraction is required", "row_index": index})
            requester = _clean(row.get("requester"))
            confirmer = _clean(row.get("confirmer"))
            if not _IDENTIFIER.fullmatch(requester) or not _IDENTIFIER.fullmatch(confirmer) or requester.lower() == confirmer.lower():
                errors.append({"code": "separate_review", "field": "requester/confirmer", "message": "stable and distinct identities are required", "row_index": index})
            for field in ("passed", "equipment_command_sent", "readback_verified", "rollback_verified", "interlock_verified"):
                if type(row.get(field)) is not bool:
                    errors.append({"code": "boolean", "field": field, "message": "boolean is required", "row_index": index})
            if scenario in BLOCK_ONLY_SCENARIOS and row.get("equipment_command_sent") is not False:
                errors.append({"code": "unsafe_execution", "field": "equipment_command_sent", "message": "block scenario must not send a command", "row_index": index})
            if len(errors) == before and started and ended:
                records.append({
                    "test_id": test_id,
                    "scenario": scenario,
                    "asset_id": asset_id,
                    "action": action,
                    "started_at": _iso_utc(started),
                    "ended_at": _iso_utc(ended),
                    "passed": row["passed"],
                    "equipment_command_sent": row["equipment_command_sent"],
                    "readback_verified": row["readback_verified"],
                    "rollback_verified": row["rollback_verified"],
                    "interlock_verified": row["interlock_verified"],
                    "requester_confirmer_distinct": True,
                    "source_reference_sha256": _canonical_digest(source_reference),
                })
        counts = Counter(row["scenario"] for row in records)
        missing = sorted(set(REQUIRED_SCENARIOS) - set(counts))
        if missing:
            errors.append({"code": "scenario_coverage", "field": "tests", "message": "missing: " + ", ".join(missing)})
        repeated = sorted(name for name, count in counts.items() if count != 1)
        if repeated:
            errors.append({"code": "scenario_cardinality", "field": "tests", "message": "exactly one record is required for: " + ", ".join(repeated)})
        if errors or len(records) != len(raw_tests):
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "received_tests": len(raw_tests),
                "accepted_tests": 0,
                "evidence": None,
                "boundary": {"site_execution_accepted": False, "dispatch_allowed": False, "production_authority": False},
            }
        records.sort(key=lambda row: (row["started_at"], row["test_id"]))
        normalized_source = {
            "source_system": _clean(source.get("source_system")),
            "owner": owner,
            "license": _clean(source.get("license")),
            "timezone": timezone_name,
            "extracted_at": _iso_utc(extracted_at) if extracted_at else None,
            "evidence_class": evidence_class,
        }
        normalized_input = {
            "schema_version": DATASET_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "source": normalized_source,
            "shadow_acceptance_digest": shadow_digest,
            "actuator_config_sha256": config_validation["config_sha256"],
            "tests": records,
        }
        dataset_sha256 = _canonical_digest(normalized_input)
        metrics = {
            "scenario_pass_rate": float(sum(row["passed"] for row in records) / len(records)),
            "unsafe_command_execution_count": sum(row["equipment_command_sent"] for row in records if row["scenario"] in BLOCK_ONLY_SCENARIOS),
            "readback_coverage_rate": float(sum(row["readback_verified"] for row in records if row["scenario"] in {"safe_command_readback", "rollback_restore"}) / 2.0),
            "interlock_coverage_rate": float(sum(row["interlock_verified"] for row in records if row["scenario"] in {"independent_interlock_trip", "emergency_stop"}) / 2.0),
            "rollback_success_rate": float(sum(row["rollback_verified"] for row in records if row["scenario"] == "rollback_restore")),
        }
        status = "pass" if (
            not missing and metrics["scenario_pass_rate"] == 1.0
            and metrics["unsafe_command_execution_count"] == 0
            and metrics["readback_coverage_rate"] == 1.0
            and metrics["interlock_coverage_rate"] == 1.0
            and metrics["rollback_success_rate"] >= 1.0
        ) else "fail"
        site_commissioning = bool(
            evidence_class == "authorized_site_commissioning_export"
            and source_verified
            and config_validation["production_gate_eligible"]
            and any(row["scenario"] == "safe_command_readback" and row["equipment_command_sent"] for row in records)
            and any(row["scenario"] == "rollback_restore" and row["equipment_command_sent"] for row in records)
        )
        reviewers = {
            "operations": _clean(operations_approved_by),
            "maritime_safety": _clean(maritime_safety_approved_by),
            "controls_engineering": _clean(controls_engineering_approved_by),
        }
        ticket = _clean(change_ticket)
        approval_metadata_valid = bool(
            ticket and _IDENTIFIER.fullmatch(ticket)
            and all(_IDENTIFIER.fullmatch(value) for value in reviewers.values())
            and len({value.lower() for value in reviewers.values()}) == 3
            and all(value.lower() != owner.lower() for value in reviewers.values())
        )
        approved = bool(site_commissioning and status == "pass" and approval_metadata_valid)
        if evidence_class == "contract_test_only":
            warnings.append({"code": "contract_only", "message": "执行样例只验证配置和联锁验收合同，不构成现场调试、设备读回或生产放行。"})
        elif not source_verified:
            warnings.append({"code": "source_attestation_missing", "message": "现场调试导出未完成来源证明，不得形成执行放行证据。"})
        if site_commissioning and not approval_metadata_valid:
            warnings.append({"code": "triple_approval_invalid", "message": "运营、海事安全和控制工程审批人必须彼此独立且不同于数据责任主体，并绑定变更单。"})
        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "dataset_sha256": dataset_sha256,
            "actuator_config_sha256": config_validation["config_sha256"],
            "shadow_acceptance_digest": shadow_digest,
            "source": normalized_source,
            "tests": records,
            "scenario_counts": dict(sorted(counts.items())),
            "metrics": metrics,
            "acceptance_status": status,
            "site_commissioning_verified": site_commissioning,
            "approved": approved,
            "approved_by": {name: value or None for name, value in reviewers.items()},
            "provenance": {
                "source_attestation": site_commissioning,
                "actuator_config_production_eligible": config_validation["production_gate_eligible"],
                "change_ticket": ticket or None,
                "algorithm_implementation": "app.services.site_execution_acceptance.SiteExecutionAcceptanceService",
            },
            "boundary": {
                "site_execution_accepted": approved,
                "direct_model_dispatch": False,
                "per_command_human_confirmation_required": True,
                "dispatch_allowed": False,
                "production_authority": False,
                "claim": "approved_site_execution_acceptance_evidence" if approved else "contract_or_unapproved_execution_candidate",
            },
        }
        evidence["evidence_digest"] = _canonical_digest(evidence)
        validation = self.validate_evidence(evidence)
        if not validation["valid"]:
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": [{"code": "evidence_validation", "field": "evidence", "message": message} for message in validation["errors"]],
                "warnings": warnings,
                "received_tests": len(raw_tests),
                "accepted_tests": 0,
                "evidence": None,
                "boundary": {"site_execution_accepted": False, "dispatch_allowed": False, "production_authority": False},
            }
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "received_tests": len(raw_tests),
            "accepted_tests": len(records),
            "evidence": evidence,
            "boundary": dict(evidence["boundary"]),
        }
