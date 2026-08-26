from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATASET_SCHEMA = "production_continuity_dataset.v1"
EVIDENCE_SCHEMA = "production_continuity_evidence.v1"
EVIDENCE_CLASSES = ("authorized_site_continuity_export", "contract_test_only")
COMPONENTS = (
    "api_gateway",
    "decision_service",
    "digital_twin",
    "data_ingestion",
    "message_queue",
    "evidence_store",
    "telemetry_store",
    "operator_ui",
)
COMPONENT_LABELS = {
    "api_gateway": "应用程序接口网关",
    "decision_service": "决策建议服务",
    "digital_twin": "数字孪生运行时",
    "data_ingestion": "现场数据接入",
    "message_queue": "事件消息队列",
    "evidence_store": "审计证据存储",
    "telemetry_store": "遥测时序存储",
    "operator_ui": "运营人员界面",
}
DRILL_LIMITS = {
    "process_restart": {"rto_minutes_max": 5, "rpo_minutes_max": 0},
    "zone_failover": {"rto_minutes_max": 15, "rpo_minutes_max": 1},
    "database_restore": {"rto_minutes_max": 60, "rpo_minutes_max": 15},
    "queue_backlog_recovery": {"rto_minutes_max": 30, "rpo_minutes_max": 0},
    "credential_rotation": {"rto_minutes_max": 15, "rpo_minutes_max": 0},
    "safe_degradation": {"rto_minutes_max": 5, "rpo_minutes_max": 0},
}
SLO_THRESHOLDS = {
    "availability_percent_min": 99.9,
    "error_rate_percent_max": 0.1,
    "latency_p95_ms_max": 1000.0,
    "data_freshness_p95_seconds_max": 120.0,
    "audit_delivery_percent_min": 100.0,
}
MINIMUM_HOURS = {"contract_test_only": 168, "authorized_site_continuity_export": 720}
BINDING_FIELDS = (
    "end_to_end_coordination_evidence_digest",
    "execution_acceptance_evidence_digest",
    "forecast_uncertainty_evidence_digest",
    "business_benefit_attribution_evidence_digest",
)
REVIEWER_ROLES = ("service_owner", "site_reliability", "continuity_cybersecurity")
SEVERITIES = ("P0", "P1", "P2", "P3")

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PLACEHOLDERS = {"unknown", "unset", "todo", "replace", "replace_me", "n/a", "none"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _named(value: Any) -> bool:
    text = _clean(value)
    return bool(text and text.lower() not in _PLACEHOLDERS)


def _parse(value: Any) -> datetime:
    text = _clean(value)
    if not text:
        raise ValueError("timestamp required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone required")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("finite number required")
    return number


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProductionContinuityService:
    """Validate continuous site service evidence without granting control authority."""

    @staticmethod
    def _error(errors: List[Dict[str, Any]], code: str, field: str, message: str, row_index: int | None = None) -> None:
        item: Dict[str, Any] = {"code": code, "field": field, "message": message}
        if row_index is not None:
            item["row_index"] = row_index
        errors.append(item)

    @classmethod
    def _normalize(cls, raw_payload: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        payload = deepcopy(raw_payload) if isinstance(raw_payload, dict) else {}
        errors: List[Dict[str, Any]] = []
        if _clean(payload.get("schema_version")) != DATASET_SCHEMA:
            cls._error(errors, "schema_version", "schema_version", f"must equal {DATASET_SCHEMA}")
        site_id, run_id = _clean(payload.get("site_id")), _clean(payload.get("run_id"))
        for field, value in (("site_id", site_id), ("run_id", run_id)):
            if not _IDENTIFIER.fullmatch(value):
                cls._error(errors, "identifier", field, "stable identifier is required")

        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        for field in ("source_system", "owner", "license"):
            if not _named(source.get(field)):
                cls._error(errors, "source_metadata", f"source.{field}", "is required and cannot be a placeholder")
        evidence_class = _clean(source.get("evidence_class"))
        if evidence_class not in EVIDENCE_CLASSES:
            cls._error(errors, "evidence_class", "source.evidence_class", f"must be one of {EVIDENCE_CLASSES}")
        timezone_name = _clean(source.get("timezone"))
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            cls._error(errors, "timezone", "source.timezone", "IANA timezone is required")
        try:
            extracted_at = _parse(source.get("extracted_at"))
        except (TypeError, ValueError):
            extracted_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
            cls._error(errors, "extracted_at", "source.extracted_at", "timezone-aware timestamp is required")

        bindings = payload.get("bindings") if isinstance(payload.get("bindings"), dict) else {}
        normalized_bindings: Dict[str, str] = {}
        if set(bindings) != set(BINDING_FIELDS):
            cls._error(errors, "binding_fields", "bindings", "keys must exactly match the four upstream evidence bindings")
        for field in BINDING_FIELDS:
            value = _clean(bindings.get(field))
            if not _SHA256.fullmatch(value):
                cls._error(errors, "binding_digest", f"bindings.{field}", "lowercase SHA-256 digest is required")
            normalized_bindings[field] = value

        window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
        try:
            start_at, end_at = _parse(window.get("start_at")), _parse(window.get("end_at"))
            if end_at <= start_at or (end_at - start_at).total_seconds() % 3600:
                raise ValueError
        except (TypeError, ValueError):
            start_at = end_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
            cls._error(errors, "window", "window", "positive whole-hour window is required")
        if window.get("step_minutes") != 60:
            cls._error(errors, "step_minutes", "window.step_minutes", "must equal sixty minutes")
        hours = int((end_at - start_at).total_seconds() // 3600)
        minimum = MINIMUM_HOURS.get(evidence_class, MINIMUM_HOURS["contract_test_only"])
        if hours < minimum:
            cls._error(errors, "window_length", "window", f"at least {minimum} continuous hours are required")
        if extracted_at < end_at or extracted_at - end_at > timedelta(hours=24):
            cls._error(errors, "source_cutoff", "source.extracted_at", "export must be at or after window end and no more than twenty-four hours later")

        records_payload = payload.get("hourly_records")
        if not isinstance(records_payload, list):
            records_payload = []
            cls._error(errors, "hourly_records", "hourly_records", "array is required")
        normalized_records: List[Dict[str, Any]] = []
        seen_hours: set[str] = set()
        for index, raw in enumerate(records_payload):
            before = len(errors)
            raw = raw if isinstance(raw, dict) else {}
            try:
                hour_start = _parse(raw.get("hour_start"))
                if hour_start < start_at or hour_start >= end_at or (hour_start - start_at).total_seconds() % 3600:
                    raise ValueError
            except (TypeError, ValueError):
                hour_start = start_at
                cls._error(errors, "hour_start", "hour_start", "aligned hour inside the window is required", index)
            hour_key = _iso(hour_start)
            if hour_key in seen_hours:
                cls._error(errors, "duplicate_hour", "hour_start", "hour must be unique", index)
            seen_hours.add(hour_key)
            receipt_id = _clean(raw.get("collection_receipt_id"))
            if not _IDENTIFIER.fullmatch(receipt_id):
                cls._error(errors, "collection_receipt", "collection_receipt_id", "stable receipt is required", index)
            component_rows = raw.get("components") if isinstance(raw.get("components"), list) else []
            normalized_components: List[Dict[str, Any]] = []
            component_ids: set[str] = set()
            for component in component_rows:
                component = component if isinstance(component, dict) else {}
                component_id = _clean(component.get("component_id"))
                if component_id not in COMPONENTS or component_id in component_ids:
                    cls._error(errors, "component_coverage", "components", "each fixed component must appear exactly once", index)
                    continue
                component_ids.add(component_id)
                try:
                    uptime = _finite(component.get("uptime_minutes"))
                    requests = int(component.get("request_count"))
                    errors_count = int(component.get("error_count"))
                    latency = _finite(component.get("latency_p95_ms"))
                    freshness = _finite(component.get("data_freshness_p95_seconds"))
                    audit_expected = int(component.get("audit_events_expected"))
                    audit_delivered = int(component.get("audit_events_delivered"))
                    if not 0 <= uptime <= 60 or requests < 1 or not 0 <= errors_count <= requests or latency < 0 or freshness < 0 or audit_expected < 1 or not 0 <= audit_delivered <= audit_expected:
                        raise ValueError
                except (TypeError, ValueError):
                    uptime, requests, errors_count, latency, freshness, audit_expected, audit_delivered = 0.0, 0, 0, 0.0, 0.0, 0, 0
                    cls._error(errors, "component_metrics", "components", "valid bounded hourly service metrics are required", index)
                source_reference = _clean(component.get("source_reference"))
                if not _IDENTIFIER.fullmatch(source_reference):
                    cls._error(errors, "component_source", "source_reference", "stable source reference is required", index)
                normalized_components.append({
                    "component_id": component_id,
                    "uptime_minutes": uptime,
                    "request_count": requests,
                    "error_count": errors_count,
                    "latency_p95_ms": latency,
                    "data_freshness_p95_seconds": freshness,
                    "audit_events_expected": audit_expected,
                    "audit_events_delivered": audit_delivered,
                    "source_reference": source_reference,
                })
            if component_ids != set(COMPONENTS):
                cls._error(errors, "component_coverage", "components", "all eight fixed components are required every hour", index)
            if len(errors) == before:
                normalized_records.append({"hour_start": hour_key, "collection_receipt_id": receipt_id, "components": sorted(normalized_components, key=lambda row: row["component_id"])})
        expected_hours = {_iso(start_at + timedelta(hours=index)) for index in range(hours)}
        if seen_hours != expected_hours or len(normalized_records) != hours:
            cls._error(errors, "continuous_coverage", "hourly_records", "every hour in the window must have one complete record")

        drills_payload = payload.get("drills") if isinstance(payload.get("drills"), list) else []
        normalized_drills: List[Dict[str, Any]] = []
        drill_types: set[str] = set()
        for index, raw in enumerate(drills_payload):
            raw = raw if isinstance(raw, dict) else {}
            drill_type = _clean(raw.get("drill_type"))
            drill_id = _clean(raw.get("drill_id"))
            if drill_type not in DRILL_LIMITS or drill_type in drill_types or not _IDENTIFIER.fullmatch(drill_id):
                cls._error(errors, "drill_contract", "drills", "each fixed drill needs a unique type and stable identifier", index)
                continue
            drill_types.add(drill_type)
            try:
                started, completed = _parse(raw.get("started_at")), _parse(raw.get("completed_at"))
                rto, rpo = _finite(raw.get("rto_minutes")), _finite(raw.get("rpo_minutes"))
                if not start_at <= started < completed <= extracted_at or rto < 0 or rpo < 0:
                    raise ValueError
            except (TypeError, ValueError):
                started = completed = start_at
                rto = rpo = -1.0
                cls._error(errors, "drill_measurement", "drills", "valid drill timing and recovery measurements are required", index)
            refs = {field: _clean(raw.get(field)) for field in ("execution_receipt_id", "restore_receipt_id", "approved_by")}
            if any(not _IDENTIFIER.fullmatch(value) for value in refs.values()):
                cls._error(errors, "drill_receipt", "drills", "execution, restore and approval references are required", index)
            normalized_drills.append({"drill_id": drill_id, "drill_type": drill_type, "started_at": _iso(started), "completed_at": _iso(completed), "rto_minutes": rto, "rpo_minutes": rpo, "result": _clean(raw.get("result")), **refs})
        if drill_types != set(DRILL_LIMITS):
            cls._error(errors, "drill_coverage", "drills", "all six fixed resilience drills are required")

        backups_payload = payload.get("backups") if isinstance(payload.get("backups"), list) else []
        normalized_backups: List[Dict[str, Any]] = []
        backup_days: set[str] = set()
        for index, raw in enumerate(backups_payload):
            raw = raw if isinstance(raw, dict) else {}
            backup_id, checksum = _clean(raw.get("backup_id")), _clean(raw.get("sha256"))
            try:
                created, restored = _parse(raw.get("created_at")), _parse(raw.get("restore_tested_at"))
                day = created.date().isoformat()
                if not start_at <= created < end_at or not created < restored <= extracted_at:
                    raise ValueError
            except (TypeError, ValueError):
                created = restored = start_at
                day = "invalid"
                cls._error(errors, "backup_timing", "backups", "backup and later restore-test timestamps are required", index)
            if day in backup_days:
                cls._error(errors, "backup_day", "backups", "one restore-tested backup per UTC day is required", index)
            backup_days.add(day)
            restore_receipt = _clean(raw.get("restore_receipt_id"))
            if not _IDENTIFIER.fullmatch(backup_id) or not _SHA256.fullmatch(checksum) or not _IDENTIFIER.fullmatch(restore_receipt) or raw.get("immutable") is not True or raw.get("encrypted") is not True:
                cls._error(errors, "backup_contract", "backups", "stable id, checksum, restore receipt, immutability and encryption are required", index)
            normalized_backups.append({"backup_id": backup_id, "created_at": _iso(created), "restore_tested_at": _iso(restored), "sha256": checksum, "immutable": raw.get("immutable") is True, "encrypted": raw.get("encrypted") is True, "restore_receipt_id": restore_receipt})
        expected_backup_days = math.ceil(hours / 24)
        if len(normalized_backups) < expected_backup_days:
            cls._error(errors, "backup_coverage", "backups", f"at least {expected_backup_days} daily restore-tested backups are required")

        incidents_payload = payload.get("incidents") if isinstance(payload.get("incidents"), list) else []
        normalized_incidents: List[Dict[str, Any]] = []
        incident_ids: set[str] = set()
        for index, raw in enumerate(incidents_payload):
            raw = raw if isinstance(raw, dict) else {}
            incident_id, severity = _clean(raw.get("incident_id")), _clean(raw.get("severity"))
            if not _IDENTIFIER.fullmatch(incident_id) or incident_id in incident_ids or severity not in SEVERITIES:
                cls._error(errors, "incident_identity", "incidents", "unique incident and fixed severity are required", index)
                continue
            incident_ids.add(incident_id)
            try:
                times = [_parse(raw.get(field)) for field in ("detected_at", "acknowledged_at", "mitigated_at", "recovered_at", "closed_at")]
                if times != sorted(times) or not start_at <= times[0] <= times[-1] <= extracted_at:
                    raise ValueError
            except (TypeError, ValueError):
                times = [start_at] * 5
                cls._error(errors, "incident_timeline", "incidents", "ordered detection through closure timestamps are required", index)
            refs = {field: _clean(raw.get(field)) for field in ("commander_id", "work_order_id", "root_cause_reference", "postmortem_reference")}
            if any(not _IDENTIFIER.fullmatch(value) for value in refs.values()):
                cls._error(errors, "incident_receipts", "incidents", "commander, work order, root cause and postmortem references are required", index)
            normalized_incidents.append({"incident_id": incident_id, "severity": severity, **dict(zip(("detected_at", "acknowledged_at", "mitigated_at", "recovered_at", "closed_at"), map(_iso, times))), **refs})

        changes_payload = payload.get("changes") if isinstance(payload.get("changes"), list) else []
        normalized_changes: List[Dict[str, Any]] = []
        for index, raw in enumerate(changes_payload):
            raw = raw if isinstance(raw, dict) else {}
            refs = {field: _clean(raw.get(field)) for field in ("change_id", "approval_receipt_id", "canary_receipt_id", "health_receipt_id", "rollback_receipt_id")}
            try:
                deployed_at = _parse(raw.get("deployed_at"))
            except (TypeError, ValueError):
                deployed_at = start_at
                cls._error(errors, "change_time", "changes", "deployment timestamp is required", index)
            if any(not _IDENTIFIER.fullmatch(value) for value in refs.values()) or raw.get("rollback_ready") is not True:
                cls._error(errors, "change_contract", "changes", "approval, canary, health and rollback receipts are required", index)
            normalized_changes.append({**refs, "deployed_at": _iso(deployed_at), "rollback_ready": raw.get("rollback_ready") is True})
        if not normalized_changes:
            cls._error(errors, "change_coverage", "changes", "at least one canary and rollback verified change is required")

        normalized = {
            "schema_version": DATASET_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "source": {"source_system": _clean(source.get("source_system")), "owner": _clean(source.get("owner")), "license": _clean(source.get("license")), "timezone": timezone_name, "extracted_at": _iso(extracted_at), "evidence_class": evidence_class},
            "bindings": normalized_bindings,
            "window": {"start_at": _iso(start_at), "end_at": _iso(end_at), "step_minutes": 60},
            "hourly_records": sorted(normalized_records, key=lambda row: row["hour_start"]),
            "drills": sorted(normalized_drills, key=lambda row: row["drill_type"]),
            "backups": sorted(normalized_backups, key=lambda row: row["created_at"]),
            "incidents": sorted(normalized_incidents, key=lambda row: row["incident_id"]),
            "changes": sorted(normalized_changes, key=lambda row: row["change_id"]),
        }
        return normalized, errors

    @staticmethod
    def _evaluate(normalized: Dict[str, Any]) -> Dict[str, Any]:
        component_metrics = []
        for component_id in COMPONENTS:
            rows = [component for hour in normalized["hourly_records"] for component in hour["components"] if component["component_id"] == component_id]
            uptime = sum(row["uptime_minutes"] for row in rows)
            total_minutes = 60 * len(rows)
            requests = sum(row["request_count"] for row in rows)
            errors = sum(row["error_count"] for row in rows)
            audit_expected = sum(row["audit_events_expected"] for row in rows)
            audit_delivered = sum(row["audit_events_delivered"] for row in rows)
            metrics = {
                "component_id": component_id,
                "availability_percent": 100.0 * uptime / max(total_minutes, 1),
                "error_rate_percent": 100.0 * errors / max(requests, 1),
                "latency_p95_ms_max_observed": max((row["latency_p95_ms"] for row in rows), default=math.inf),
                "data_freshness_p95_seconds_max_observed": max((row["data_freshness_p95_seconds"] for row in rows), default=math.inf),
                "audit_delivery_percent": 100.0 * audit_delivered / max(audit_expected, 1),
                "observation_hours": len(rows),
            }
            metrics["slo_pass"] = bool(
                metrics["availability_percent"] >= SLO_THRESHOLDS["availability_percent_min"]
                and metrics["error_rate_percent"] <= SLO_THRESHOLDS["error_rate_percent_max"]
                and metrics["latency_p95_ms_max_observed"] <= SLO_THRESHOLDS["latency_p95_ms_max"]
                and metrics["data_freshness_p95_seconds_max_observed"] <= SLO_THRESHOLDS["data_freshness_p95_seconds_max"]
                and metrics["audit_delivery_percent"] >= SLO_THRESHOLDS["audit_delivery_percent_min"]
            )
            component_metrics.append(metrics)
        drill_results = []
        for drill in normalized["drills"]:
            limit = DRILL_LIMITS[drill["drill_type"]]
            passed = drill["result"] == "pass" and drill["rto_minutes"] <= limit["rto_minutes_max"] and drill["rpo_minutes"] <= limit["rpo_minutes_max"]
            drill_results.append({**drill, "limit": limit, "pass": passed})
        metrics = {
            "continuous_hours": len(normalized["hourly_records"]),
            "component_count": len(component_metrics),
            "component_slo_pass_count": sum(row["slo_pass"] for row in component_metrics),
            "drill_pass_count": sum(row["pass"] for row in drill_results),
            "drill_count": len(drill_results),
            "restore_tested_backup_count": len(normalized["backups"]),
            "closed_incident_count": len(normalized["incidents"]),
            "verified_change_count": len(normalized["changes"]),
            "collection_receipt_coverage": sum(bool(row["collection_receipt_id"]) for row in normalized["hourly_records"]) / max(len(normalized["hourly_records"]), 1),
        }
        threshold_checks = {
            "continuous_window": metrics["continuous_hours"] >= MINIMUM_HOURS[normalized["source"]["evidence_class"]],
            "component_coverage": metrics["component_count"] == len(COMPONENTS),
            "all_component_slos": metrics["component_slo_pass_count"] == len(COMPONENTS),
            "all_resilience_drills": metrics["drill_pass_count"] == len(DRILL_LIMITS),
            "daily_restore_tests": metrics["restore_tested_backup_count"] >= math.ceil(metrics["continuous_hours"] / 24),
            "incident_closure": all(row["closed_at"] and row["postmortem_reference"] for row in normalized["incidents"]),
            "canary_and_rollback": metrics["verified_change_count"] >= 1 and all(row["rollback_ready"] for row in normalized["changes"]),
            "collection_receipts": metrics["collection_receipt_coverage"] == 1.0,
        }
        return {"component_metrics": component_metrics, "drill_results": drill_results, "metrics": metrics, "threshold_checks": threshold_checks, "continuity_status": "pass" if all(threshold_checks.values()) else "blocked", "algorithm": "hourly-continuity-slo-incident-backup-drill-gate-v1"}

    @classmethod
    def validate_evidence(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {"valid": False, "errors": ["continuity evidence must be an object"], "production_gate_eligible": False}
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        body = deepcopy(payload)
        evidence_digest = _clean(body.pop("evidence_digest", ""))
        if not _SHA256.fullmatch(evidence_digest) or _digest(body) != evidence_digest:
            errors.append("evidence_digest mismatch")
        source_input = payload.get("source_input") if isinstance(payload.get("source_input"), dict) else {}
        normalized, normalization_errors = cls._normalize(source_input)
        if normalization_errors or normalized != source_input:
            errors.append("source_input is not a valid normalized continuity contract")
        else:
            expected = cls._evaluate(normalized)
            if payload.get("dataset_sha256") != _digest(normalized):
                errors.append("dataset_sha256 mismatch")
            for field in ("component_metrics", "drill_results", "metrics", "threshold_checks", "continuity_status", "algorithm"):
                if payload.get(field) != expected[field]:
                    errors.append(f"{field} does not reproduce from source_input")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        reviewers = payload.get("approved_by") if isinstance(payload.get("approved_by"), list) else []
        reviewer_ids = [_clean(row.get("reviewer_id")) for row in reviewers if isinstance(row, dict)]
        roles = Counter(row.get("role") for row in reviewers if isinstance(row, dict))
        owner = _clean(source.get("owner")).lower()
        approval_contract = bool(
            not errors
            and source.get("evidence_class") == "authorized_site_continuity_export"
            and source.get("source_verified") is True
            and payload.get("continuity_status") == "pass"
            and payload.get("approved") is True
            and roles == Counter(REVIEWER_ROLES)
            and len(reviewer_ids) == len(REVIEWER_ROLES)
            and len(set(reviewer_ids)) == len(REVIEWER_ROLES)
            and all(_IDENTIFIER.fullmatch(item) and item.lower() != owner for item in reviewer_ids)
            and _IDENTIFIER.fullmatch(_clean(payload.get("change_ticket")))
            and boundary.get("site_continuity_accepted") is True
            and boundary.get("field_slo_claim_eligible") is True
            and boundary.get("automatic_failover_authority") is False
            and boundary.get("dispatch_allowed") is False
            and boundary.get("production_authority") is False
        )
        if payload.get("approved") is True and not approval_contract:
            errors.append("approved continuity evidence requires authorized measurements, clean fixed gates and three independent reviewers")
        return {"valid": not errors, "errors": errors, "production_gate_eligible": bool(not errors and approval_contract)}

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        service_owner_approved_by: str | None = None,
        site_reliability_approved_by: str | None = None,
        continuity_cybersecurity_approved_by: str | None = None,
        change_ticket: str | None = None,
    ) -> Dict[str, Any]:
        normalized, errors = self._normalize(payload)
        rejected_boundary = {"site_continuity_accepted": False, "field_slo_claim_eligible": False, "automatic_failover_authority": False, "dispatch_allowed": False, "production_authority": False}
        if errors:
            return {"schema_version": EVIDENCE_SCHEMA, "valid": False, "errors": errors, "warnings": [], "dataset_sha256": None, "evidence": None, "boundary": rejected_boundary}
        evaluated = self._evaluate(normalized)
        source = normalized["source"]
        source_attested = bool(source_verified and source["evidence_class"] == "authorized_site_continuity_export")
        reviewers = [
            {"role": "service_owner", "reviewer_id": _clean(service_owner_approved_by)},
            {"role": "site_reliability", "reviewer_id": _clean(site_reliability_approved_by)},
            {"role": "continuity_cybersecurity", "reviewer_id": _clean(continuity_cybersecurity_approved_by)},
        ]
        reviewer_ids = [row["reviewer_id"] for row in reviewers]
        owner = source["owner"].lower()
        ticket = _clean(change_ticket)
        reviewers_valid = bool(all(_IDENTIFIER.fullmatch(item) and item.lower() != owner for item in reviewer_ids) and len(set(reviewer_ids)) == 3 and _IDENTIFIER.fullmatch(ticket))
        approved = bool(source_attested and evaluated["continuity_status"] == "pass" and reviewers_valid)
        boundary = {
            "site_continuity_accepted": approved,
            "field_slo_claim_eligible": approved,
            "automatic_failover_authority": False,
            "dispatch_allowed": False,
            "production_authority": False,
            "claim": "approved_site_continuity_evidence" if approved else "contract_or_unapproved_continuity_evidence",
            "reason": "连续实测、服务目标、事件、恢复演练、备份还原和独立复核均已通过；故障切换与业务控制仍由现场授权系统执行。" if approved else "合同样例、进程运行时间或单次健康检查不能替代现场连续服务、事件闭环和灾备恢复证据。",
        }
        warnings = [] if approved else [{"code": "contract_only", "message": "连续性合同计算不构成现场服务等级承诺或自动故障切换授权。"}]
        evidence: Dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": normalized["site_id"],
            "run_id": normalized["run_id"],
            "dataset_sha256": _digest(normalized),
            "source": {**source, "source_verified": source_attested},
            "bindings": normalized["bindings"],
            "source_input": normalized,
            **evaluated,
            "approved": approved,
            "approved_by": reviewers if approved else [],
            "change_ticket": ticket if approved else "",
            "boundary": boundary,
        }
        evidence["evidence_digest"] = _digest(evidence)
        validation = self.validate_evidence(evidence)
        if not validation["valid"]:
            return {"schema_version": EVIDENCE_SCHEMA, "valid": False, "errors": [{"code": "evidence_validation", "field": "evidence", "message": item} for item in validation["errors"]], "warnings": warnings, "dataset_sha256": evidence["dataset_sha256"], "evidence": evidence, "boundary": boundary}
        return {"schema_version": EVIDENCE_SCHEMA, "valid": True, "errors": [], "warnings": warnings, "dataset_sha256": evidence["dataset_sha256"], "evidence": evidence, "boundary": boundary}

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_PRODUCTION_CONTINUITY_PATH"))
        artifact: Dict[str, Any] = {"mode": "unconfigured", "configured": False, "verified": False, "artifact_id": None, "sha256": None, "blockers": ["production_continuity_artifact_not_configured"]}
        if raw_path:
            path = Path(raw_path).expanduser()
            artifact.update(mode="configured_invalid", configured=True, artifact_id=path.name, blockers=["production_continuity_artifact_invalid"])
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validation = self.validate_evidence(payload)
                eligible = validation["production_gate_eligible"]
                metrics = payload.get("metrics") or {}
                artifact.update(mode="verified_site_artifact" if eligible else "configured_invalid", verified=eligible, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), site_id=payload.get("site_id"), run_id=payload.get("run_id"), continuous_hours=metrics.get("continuous_hours"), component_slo_pass_count=metrics.get("component_slo_pass_count"), drill_pass_count=metrics.get("drill_pass_count"), backup_count=metrics.get("restore_tested_backup_count"), approved=payload.get("approved") is True, blockers=[] if eligible else list(validation["errors"]) or ["site_continuity_not_approved"])
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        accepted = artifact.get("verified") is True
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "configured_artifact": artifact,
            "contract": {"components": [{"component_id": item, "label": COMPONENT_LABELS[item]} for item in COMPONENTS], "slo_thresholds": dict(SLO_THRESHOLDS), "drill_limits": deepcopy(DRILL_LIMITS), "minimum_hours": dict(MINIMUM_HOURS), "required_bindings": list(BINDING_FIELDS), "required_reviewer_roles": list(REVIEWER_ROLES)},
            "boundary": {"site_continuity_accepted": accepted, "field_slo_claim_eligible": accepted, "automatic_failover_authority": False, "dispatch_allowed": False, "production_authority": False, "site_status": "现场连续运行证据已验证" if accepted else "待接入现场连续服务与恢复证据", "reason": "现场连续测量、服务目标、事件闭环、备份恢复与故障演练已验证；执行权限仍由现场系统持有。" if accepted else "进程健康检查、公开回放和合同样例不能替代连续运行、值班响应与灾备恢复证据。"},
        }
