from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATASET_SCHEMA = "site_shadow_observation.v1"
EVIDENCE_SCHEMA = "site_shadow_acceptance_evidence.v2"
EVIDENCE_CLASSES = ("authorized_site_shadow_export", "contract_test_only")
MINIMUM_CYCLES = 35
MINIMUM_OPERATIONAL_DAYS = 7
THRESHOLDS = {
    "guardrail_violation_rate": 0.0,
    "data_quality_failure_rate": 0.01,
    "recommendation_unavailable_rate": 0.01,
    "action_infeasible_rate": 0.02,
    "operator_review_uncovered_rate": 0.20,
    "operator_rejection_rate": 0.20,
    "projected_energy_degradation_rate": 0.05,
    "projected_throughput_degradation_rate": 0.05,
    "projected_delay_degradation_rate": 0.05,
    "p95_recommendation_latency_ms": 2000.0,
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PLACEHOLDERS = {"unknown", "unset", "todo", "replace", "replace_me", "n/a", "none"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _parse_timestamp(value: Any) -> datetime:
    text = _clean(value)
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp requires an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_digest(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 1.0


def _percentile_95(values: List[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("inf")
    rank = 0.95 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower))


def _cycle_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(records)
    reviewed = [row for row in records if row.get("operator_disposition") in {"accepted", "rejected"}]
    available = [row for row in records if row.get("recommendation_available") is True]
    return {
        "guardrail_violation_rate": _rate(
            sum(int(row.get("guardrail_violation_count") or 0) > 0 for row in records), total
        ),
        "data_quality_failure_rate": _rate(
            sum(row.get("data_quality_passed") is not True for row in records), total
        ),
        "recommendation_unavailable_rate": _rate(
            sum(row.get("recommendation_available") is not True for row in records), total
        ),
        "action_infeasible_rate": _rate(
            sum(row.get("action_feasible") is not True for row in available), len(available)
        ),
        "operator_review_uncovered_rate": _rate(total - len(reviewed), total),
        "operator_rejection_rate": _rate(
            sum(row.get("operator_disposition") == "rejected" for row in reviewed), len(reviewed)
        ),
        "projected_energy_degradation_rate": _rate(
            sum(row.get("projected_energy_non_degraded") is not True for row in records), total
        ),
        "projected_throughput_degradation_rate": _rate(
            sum(row.get("projected_throughput_non_degraded") is not True for row in records), total
        ),
        "projected_delay_degradation_rate": _rate(
            sum(row.get("projected_delay_non_degraded") is not True for row in records), total
        ),
        "p95_recommendation_latency_ms": _percentile_95(
            [float(row["recommendation_latency_ms"]) for row in available]
        ),
    }


def _consecutive_days(days: List[date]) -> bool:
    return all(right - left == timedelta(days=1) for left, right in zip(days, days[1:]))


class SiteShadowAcceptanceService:
    """Build read-only, tamper-evident site shadow acceptance evidence."""

    @staticmethod
    def _error(
        errors: List[Dict[str, Any]],
        code: str,
        field: str,
        message: str,
        *,
        row_index: int | None = None,
    ) -> None:
        item: Dict[str, Any] = {"code": code, "field": field, "message": message}
        if row_index is not None:
            item["row_index"] = row_index
        errors.append(item)

    @staticmethod
    def validate_rollback_drill(payload: Dict[str, Any] | None) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {"verified": False, "errors": ["rollback drill evidence must be an object"]}
        reference = _clean(payload.get("command_id") or payload.get("drill_id"))
        if not _IDENTIFIER.fullmatch(reference):
            errors.append("rollback drill requires a stable command_id or drill_id")
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        execution_passed = any(
            isinstance(row, dict) and row.get("ok") is True and row.get("rollback") is not True
            for row in results
        )
        rollback_passed = any(
            isinstance(row, dict) and row.get("ok") is True and row.get("rollback") is True
            for row in results
        )
        if not execution_passed:
            errors.append("rollback drill has no successful execution result")
        if not rollback_passed:
            errors.append("rollback drill has no successful rollback result")
        timestamps = payload.get("timestamps") if isinstance(payload.get("timestamps"), dict) else {}
        try:
            executed_at = _finite(timestamps.get("executed_at"))
            rolledback_at = _finite(timestamps.get("rolledback_at"))
            if rolledback_at <= executed_at:
                errors.append("rollback completion must occur after execution")
        except (TypeError, ValueError):
            executed_at = 0.0
            rolledback_at = 0.0
            errors.append("rollback drill requires numeric executed_at and rolledback_at timestamps")
        approvals = payload.get("approvals") if isinstance(payload.get("approvals"), list) else []
        rollback_approval = next(
            (
                row for row in approvals
                if isinstance(row, dict)
                and row.get("type") == "rollback"
                and _clean(row.get("by"))
                and _clean(row.get("reason"))
            ),
            None,
        )
        if rollback_approval is None:
            errors.append("rollback drill requires a named rollback approval and reason")
        return {
            "verified": not errors,
            "errors": errors,
            "reference": reference or None,
            "evidence_sha256": _canonical_digest(payload),
            "executed_at": executed_at or None,
            "rolledback_at": rolledback_at or None,
            "approved_by": _clean((rollback_approval or {}).get("by")) or None,
        }

    @staticmethod
    def validate_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {
                "valid": False,
                "errors": ["shadow evidence must be an object"],
                "production_gate_eligible": False,
            }
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        for field in (
            "site_id", "run_id", "dataset_sha256", "candidate_policy_version",
            "incumbent_policy_version", "comparison_window", "source", "cycles",
            "operational_days", "metrics", "thresholds", "threshold_checks",
            "provenance", "boundary", "evidence_digest",
        ):
            if payload.get(field) in (None, "", [], {}):
                errors.append("missing field: " + field)
        if not _SHA256.fullmatch(_clean(payload.get("dataset_sha256"))):
            errors.append("dataset_sha256 must be a lowercase SHA-256 digest")
        records = payload.get("cycles") if isinstance(payload.get("cycles"), list) else []
        if len(records) < MINIMUM_CYCLES or int(payload.get("shadow_cycles") or 0) != len(records):
            errors.append(f"shadow evidence requires at least {MINIMUM_CYCLES} cycle records")
        dates: List[date] = []
        cycle_ids: set[str] = set()
        for index, row in enumerate(records):
            if not isinstance(row, dict):
                errors.append(f"cycles[{index}] must be an object")
                continue
            cycle_id = _clean(row.get("cycle_id"))
            if not _IDENTIFIER.fullmatch(cycle_id) or cycle_id in cycle_ids:
                errors.append(f"cycles[{index}].cycle_id must be stable and unique")
            cycle_ids.add(cycle_id)
            try:
                started = _parse_timestamp(row.get("started_at"))
                ended = _parse_timestamp(row.get("ended_at"))
                if ended <= started:
                    raise ValueError("invalid interval")
                dates.append(date.fromisoformat(_clean(row.get("operational_day"))))
            except (TypeError, ValueError):
                errors.append(f"cycles[{index}] requires a valid interval and operational_day")
            if row.get("side_effect") is not False:
                errors.append(f"cycles[{index}] side_effect must be false")
            for field in ("incumbent_reference_sha256", "candidate_receipt_sha256"):
                if not _SHA256.fullmatch(_clean(row.get(field))):
                    errors.append(f"cycles[{index}].{field} must be a SHA-256 digest")
        unique_days = sorted(set(dates))
        if len(unique_days) < MINIMUM_OPERATIONAL_DAYS or not _consecutive_days(unique_days):
            errors.append(
                f"shadow evidence requires at least {MINIMUM_OPERATIONAL_DAYS} consecutive operational days"
            )
        if int(payload.get("operational_days") or 0) != len(unique_days):
            errors.append("operational_days does not match cycle records")
        derived_metrics = _cycle_metrics(records) if records else {}
        supplied_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        for name, value in derived_metrics.items():
            try:
                if not math.isclose(float(supplied_metrics.get(name)), value, rel_tol=1e-9, abs_tol=1e-9):
                    errors.append(f"metric {name} does not match cycle records")
            except (TypeError, ValueError):
                errors.append(f"metric {name} must be numeric")
        if payload.get("thresholds") != THRESHOLDS:
            errors.append("shadow thresholds do not match the fixed acceptance contract")
        derived_checks = {
            name: derived_metrics.get(name, float("inf")) <= limit
            for name, limit in THRESHOLDS.items()
        }
        if payload.get("threshold_checks") != derived_checks:
            errors.append("threshold_checks do not match cycle metrics and fixed thresholds")
        expected_status = "pass" if derived_checks and all(derived_checks.values()) else "fail"
        if payload.get("acceptance_status") != expected_status:
            errors.append("acceptance_status does not match threshold results")
        try:
            if not math.isclose(
                float(payload.get("guardrail_violation_rate")),
                float(derived_metrics.get("guardrail_violation_rate")),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                errors.append("guardrail_violation_rate does not match cycle records")
        except (TypeError, ValueError):
            errors.append("guardrail_violation_rate must be numeric")
        digest_payload = dict(payload)
        provided_digest = _clean(digest_payload.pop("evidence_digest", ""))
        if _canonical_digest(digest_payload) != provided_digest:
            errors.append("evidence_digest does not match shadow evidence content")
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        if (
            boundary.get("dispatch_allowed") is not False
            or boundary.get("production_authority") is not False
            or boundary.get("side_effects") is not False
            or boundary.get("candidate_business_impact") != "projected_not_measured"
        ):
            errors.append("shadow evidence cannot grant dispatch, authority, side effects or measured candidate impact")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        approvers = payload.get("approved_by") if isinstance(payload.get("approved_by"), dict) else {}
        owner = _clean(source.get("owner")).lower()
        operations_reviewer = _clean(approvers.get("operations"))
        safety_reviewer = _clean(approvers.get("safety"))
        approval_contract = bool(
            payload.get("approved") is True
            and payload.get("measured_incumbent_baseline") is True
            and source.get("evidence_class") == "authorized_site_shadow_export"
            and provenance.get("source_attestation") is True
            and provenance.get("change_ticket")
            and isinstance(provenance.get("rollback_drill"), dict)
            and provenance["rollback_drill"].get("verified") is True
            and provenance["rollback_drill"].get("reference")
            and _SHA256.fullmatch(_clean(provenance["rollback_drill"].get("evidence_sha256")))
            and operations_reviewer
            and safety_reviewer
            and operations_reviewer.lower() != safety_reviewer.lower()
            and operations_reviewer.lower() != owner
            and safety_reviewer.lower() != owner
            and expected_status == "pass"
            and boundary.get("site_shadow_accepted") is True
            and boundary.get("dual_approval_verified") is True
        )
        if payload.get("approved") is True and not approval_contract:
            errors.append("approved shadow evidence requires an attested source, passing cycles, rollback reference and two independent reviewers")
        production_gate_eligible = bool(not errors and approval_contract)
        return {
            "valid": not errors,
            "errors": errors,
            "threshold_checks": derived_checks,
            "production_gate_eligible": production_gate_eligible,
            "evidence_type": "site_shadow_acceptance_evidence_v2",
        }

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_SHADOW_ACCEPTANCE_PATH"))
        configured: Dict[str, Any] = {
            "mode": "unconfigured",
            "configured": False,
            "verified": False,
            "production_gate_eligible": False,
            "artifact_id": None,
            "sha256": None,
            "blockers": ["shadow_acceptance_artifact_not_configured"],
        }
        if raw_path:
            path = Path(raw_path).expanduser()
            configured.update(
                mode="configured_invalid",
                configured=True,
                artifact_id=path.name,
                blockers=["shadow_acceptance_artifact_invalid"],
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validation = self.validate_evidence(payload)
                configured.update(
                    mode="verified_site_artifact" if validation["valid"] else "configured_invalid",
                    verified=bool(validation["valid"]),
                    production_gate_eligible=bool(validation["production_gate_eligible"]),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    site_id=payload.get("site_id"),
                    acceptance_status=payload.get("acceptance_status"),
                    shadow_cycles=payload.get("shadow_cycles"),
                    operational_days=payload.get("operational_days"),
                    approved=payload.get("approved") is True,
                    blockers=[] if validation["production_gate_eligible"] else list(validation["errors"]),
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        accepted = bool(configured.get("production_gate_eligible"))
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "configured_artifact": configured,
            "contract": {
                "minimum_cycles": MINIMUM_CYCLES,
                "minimum_consecutive_operational_days": MINIMUM_OPERATIONAL_DAYS,
                "comparison": "measured_incumbent_vs_candidate_projection",
                "candidate_impact_claim": "projected_not_measured",
                "required_evidence": [
                    "measured incumbent source reference",
                    "candidate recommendation receipt",
                    "data quality and guardrail result",
                    "action feasibility and latency",
                    "operator disposition",
                    "two independent approvals and rollback drill reference",
                ],
                "fixed_thresholds": dict(THRESHOLDS),
            },
            "boundary": {
                "site_shadow_accepted": accepted,
                "measured_incumbent_verified": accepted,
                "candidate_business_impact_measured": False,
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "现场影子验收证据已验证" if accepted else "待接入连续现场影子周期",
                "reason": (
                    "现行策略实测基线、连续影子周期、固定安全门、回退演练引用和双人审批均已验证；本模块仍不发送设备指令。"
                    if accepted
                    else "合同样例不能替代现场影子运行；必须接入授权现行策略实测结果、候选建议回执、连续周期、回退演练和双人审批。"
                ),
            },
        }

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        operations_approved_by: str | None = None,
        safety_approved_by: str | None = None,
        change_ticket: str | None = None,
        rollback_drill_reference: str | None = None,
        rollback_drill_evidence: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        if not isinstance(payload, dict):
            payload = {}
            self._error(errors, "dataset_type", "$", "payload must be a JSON object")
        if _clean(payload.get("schema_version")) != DATASET_SCHEMA:
            self._error(errors, "schema_version", "schema_version", f"must equal {DATASET_SCHEMA}")
        site_id = _clean(payload.get("site_id"))
        run_id = _clean(payload.get("run_id"))
        for field, value in (("site_id", site_id), ("run_id", run_id)):
            if not _IDENTIFIER.fullmatch(value):
                self._error(errors, "identifier", field, "must be a stable 3-128 character identifier")

        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        source_system = _clean(source.get("source_system"))
        owner = _clean(source.get("owner"))
        license_name = _clean(source.get("license"))
        timezone_name = _clean(source.get("timezone"))
        evidence_class = _clean(source.get("evidence_class"))
        for field, value in (
            ("source_system", source_system), ("owner", owner), ("license", license_name),
            ("timezone", timezone_name), ("evidence_class", evidence_class),
        ):
            if not value:
                self._error(errors, "source_metadata", f"source.{field}", "field is required")
        if owner.lower() in _PLACEHOLDERS or license_name.lower() in _PLACEHOLDERS:
            self._error(errors, "source_placeholder", "source.owner/license", "placeholder governance metadata is rejected")
        if evidence_class and evidence_class not in EVIDENCE_CLASSES:
            self._error(errors, "evidence_class", "source.evidence_class", f"must be one of {', '.join(EVIDENCE_CLASSES)}")
        zone: ZoneInfo | None = None
        try:
            zone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError):
            self._error(errors, "timezone", "source.timezone", "must be a valid IANA timezone")
        extracted_at: datetime | None = None
        try:
            extracted_at = _parse_timestamp(source.get("extracted_at"))
        except (TypeError, ValueError):
            self._error(errors, "extracted_at", "source.extracted_at", "must be a timezone-aware timestamp")

        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        candidate_version = _clean(policy.get("candidate_policy_version"))
        incumbent_version = _clean(policy.get("incumbent_policy_version"))
        if not _IDENTIFIER.fullmatch(candidate_version) or not _IDENTIFIER.fullmatch(incumbent_version):
            self._error(errors, "policy_version", "policy", "candidate and incumbent policy versions must be stable identifiers")
        if candidate_version == incumbent_version and candidate_version:
            self._error(errors, "policy_identity", "policy", "candidate and incumbent versions must differ")
        if policy.get("recommendation_only") is not True:
            self._error(errors, "shadow_mode", "policy.recommendation_only", "shadow candidate must be recommendation-only")
        for field in ("calibration_evidence_digest", "twin_graph_digest"):
            if not _SHA256.fullmatch(_clean(policy.get(field))):
                self._error(errors, "policy_binding", f"policy.{field}", "must be a lowercase SHA-256 digest")

        rows = payload.get("cycles")
        if not isinstance(rows, list):
            self._error(errors, "cycles", "cycles", "cycles must be an array")
            rows = []
        elif len(rows) > 100000:
            self._error(errors, "cycle_limit", "cycles", "a shadow bundle may contain at most one hundred thousand cycles")
        records: List[Dict[str, Any]] = []
        cycle_ids: set[str] = set()
        for index, raw in enumerate(rows):
            if not isinstance(raw, dict):
                self._error(errors, "cycle_type", "cycles", "cycle must be an object", row_index=index)
                continue
            before = len(errors)
            cycle_id = _clean(raw.get("cycle_id"))
            asset_group = _clean(raw.get("asset_group"))
            scenario = _clean(raw.get("scenario"))
            for field, value in (("cycle_id", cycle_id), ("asset_group", asset_group), ("scenario", scenario)):
                if not _IDENTIFIER.fullmatch(value):
                    self._error(errors, "identifier", field, "stable identifier is required", row_index=index)
            if cycle_id in cycle_ids:
                self._error(errors, "duplicate_cycle", "cycle_id", "cycle_id must be unique", row_index=index)
            cycle_ids.add(cycle_id)
            started: datetime | None = None
            ended: datetime | None = None
            try:
                started = _parse_timestamp(raw.get("started_at"))
                ended = _parse_timestamp(raw.get("ended_at"))
                if ended <= started or ended - started > timedelta(hours=24):
                    raise ValueError("invalid cycle interval")
                if extracted_at and ended > extracted_at:
                    raise ValueError("cycle after extraction")
            except (TypeError, ValueError):
                self._error(errors, "cycle_interval", "started_at/ended_at", "cycle interval must be timezone-aware, positive, at most one day, and before extraction", row_index=index)
            incumbent = raw.get("incumbent") if isinstance(raw.get("incumbent"), dict) else {}
            candidate = raw.get("candidate") if isinstance(raw.get("candidate"), dict) else {}
            guardrail = raw.get("guardrail") if isinstance(raw.get("guardrail"), dict) else {}
            review = raw.get("review") if isinstance(raw.get("review"), dict) else {}
            numeric: Dict[str, float] = {}
            for scope, values, fields in (
                ("incumbent", incumbent, ("actual_energy_kwh", "actual_throughput_teu", "actual_delay_minutes")),
                ("candidate", candidate, ("projected_energy_kwh", "projected_throughput_teu", "projected_delay_minutes", "recommendation_latency_ms")),
            ):
                for field in fields:
                    try:
                        numeric[f"{scope}.{field}"] = _finite(values.get(field))
                        if numeric[f"{scope}.{field}"] < 0.0:
                            raise ValueError("negative")
                    except (TypeError, ValueError):
                        self._error(errors, "numeric", f"{scope}.{field}", "finite nonnegative value is required", row_index=index)
            incumbent_reference = _clean(incumbent.get("source_reference"))
            receipt_id = _clean(candidate.get("recommendation_receipt_id"))
            if not _IDENTIFIER.fullmatch(incumbent_reference):
                self._error(errors, "incumbent_reference", "incumbent.source_reference", "stable measured source reference is required", row_index=index)
            if not _IDENTIFIER.fullmatch(receipt_id):
                self._error(errors, "candidate_receipt", "candidate.recommendation_receipt_id", "stable recommendation receipt is required", row_index=index)
            if candidate.get("recommendation_available") is not True or candidate.get("action_feasible") is not True:
                # A failed cycle is accepted as evidence, but will fail the fixed rate gates as appropriate.
                pass
            if raw.get("data_quality_passed") not in {True, False}:
                self._error(errors, "data_quality", "data_quality_passed", "boolean is required", row_index=index)
            try:
                violation_count = int(guardrail.get("violation_count"))
                if violation_count < 0:
                    raise ValueError("negative")
            except (TypeError, ValueError):
                violation_count = 0
                self._error(errors, "guardrail", "guardrail.violation_count", "nonnegative integer is required", row_index=index)
            disposition = _clean(review.get("disposition"))
            if disposition not in {"accepted", "rejected", "not_reviewed"}:
                self._error(errors, "operator_disposition", "review.disposition", "must be accepted, rejected, or not_reviewed", row_index=index)
            if raw.get("side_effect") is not False:
                self._error(errors, "side_effect", "side_effect", "shadow cycle must declare side_effect=false", row_index=index)
            if len(errors) == before and started and ended and zone:
                actual_energy = numeric["incumbent.actual_energy_kwh"]
                actual_throughput = numeric["incumbent.actual_throughput_teu"]
                actual_delay = numeric["incumbent.actual_delay_minutes"]
                projected_energy = numeric["candidate.projected_energy_kwh"]
                projected_throughput = numeric["candidate.projected_throughput_teu"]
                projected_delay = numeric["candidate.projected_delay_minutes"]
                records.append({
                    "cycle_id": cycle_id,
                    "started_at": _iso_utc(started),
                    "ended_at": _iso_utc(ended),
                    "operational_day": started.astimezone(zone).date().isoformat(),
                    "asset_group": asset_group,
                    "scenario": scenario,
                    "data_quality_passed": raw.get("data_quality_passed") is True,
                    "recommendation_available": candidate.get("recommendation_available") is True,
                    "action_feasible": candidate.get("action_feasible") is True,
                    "recommendation_latency_ms": numeric["candidate.recommendation_latency_ms"],
                    "guardrail_violation_count": violation_count,
                    "operator_disposition": disposition,
                    "projected_energy_non_degraded": projected_energy <= actual_energy,
                    "projected_throughput_non_degraded": projected_throughput >= actual_throughput,
                    "projected_delay_non_degraded": projected_delay <= actual_delay,
                    "incumbent_measurement_complete": True,
                    "incumbent_reference_sha256": _canonical_digest(incumbent_reference),
                    "candidate_receipt_sha256": _canonical_digest(receipt_id),
                    "side_effect": False,
                })

        records.sort(key=lambda row: (row["started_at"], row["cycle_id"]))
        days = sorted({date.fromisoformat(row["operational_day"]) for row in records})
        if len(records) < MINIMUM_CYCLES:
            self._error(errors, "minimum_cycles", "cycles", f"at least {MINIMUM_CYCLES} valid shadow cycles are required")
        if len(days) < MINIMUM_OPERATIONAL_DAYS or not _consecutive_days(days):
            self._error(errors, "operational_continuity", "cycles", f"at least {MINIMUM_OPERATIONAL_DAYS} consecutive operational days are required")
        if errors or len(records) != len(rows):
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "received_cycles": len(rows),
                "accepted_cycles": 0,
                "dataset_sha256": None,
                "evidence": None,
                "boundary": {
                    "site_shadow_accepted": False,
                    "dispatch_allowed": False,
                    "production_authority": False,
                    "claim": "shadow_dataset_rejected",
                },
            }

        normalized_source = {
            "source_system": source_system,
            "owner": owner,
            "license": license_name,
            "timezone": timezone_name,
            "extracted_at": _iso_utc(extracted_at) if extracted_at else None,
            "evidence_class": evidence_class,
        }
        normalized_input = {
            "schema_version": DATASET_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "source": normalized_source,
            "policy": {
                "candidate_policy_version": candidate_version,
                "incumbent_policy_version": incumbent_version,
                "calibration_evidence_digest": _clean(policy.get("calibration_evidence_digest")),
                "twin_graph_digest": _clean(policy.get("twin_graph_digest")),
                "recommendation_only": True,
            },
            "cycles": records,
        }
        dataset_sha256 = _canonical_digest(normalized_input)
        metrics = _cycle_metrics(records)
        checks = {name: metrics[name] <= limit for name, limit in THRESHOLDS.items()}
        status = "pass" if all(checks.values()) else "fail"
        measured_incumbent = bool(evidence_class == "authorized_site_shadow_export" and source_verified)
        operations_reviewer = _clean(operations_approved_by)
        safety_reviewer = _clean(safety_approved_by)
        ticket = _clean(change_ticket)
        rollback_reference = _clean(rollback_drill_reference)
        rollback_validation = self.validate_rollback_drill(rollback_drill_evidence)
        if rollback_validation.get("reference"):
            if rollback_reference and rollback_reference != rollback_validation["reference"]:
                rollback_validation = {
                    **rollback_validation,
                    "verified": False,
                    "errors": [*rollback_validation["errors"], "rollback reference does not match evidence"],
                }
            rollback_reference = _clean(rollback_validation["reference"])
        approval_metadata_valid = bool(
            operations_reviewer and safety_reviewer and ticket and rollback_reference
            and all(_IDENTIFIER.fullmatch(value) for value in (operations_reviewer, safety_reviewer, ticket, rollback_reference))
            and operations_reviewer.lower() != safety_reviewer.lower()
            and operations_reviewer.lower() != owner.lower()
            and safety_reviewer.lower() != owner.lower()
            and rollback_validation.get("verified") is True
        )
        approved = bool(measured_incumbent and status == "pass" and approval_metadata_valid)
        if evidence_class == "contract_test_only":
            warnings.append({"code": "contract_only", "message": "影子样例只验证计算与验收合同，不构成现场现行策略基线、候选收益或上线许可。"})
        elif not source_verified:
            warnings.append({"code": "source_attestation_missing", "message": "授权现场影子导出尚未完成来源证明，不得形成现场验收证据。"})
        if measured_incumbent and (operations_reviewer or safety_reviewer or ticket or rollback_reference) and not approval_metadata_valid:
            warnings.append({"code": "dual_approval_invalid", "message": "运营与安全审批人必须彼此独立、均不同于数据责任主体，并绑定变更单和已验证的回退演练证据。"})
        if status != "pass":
            warnings.append({"code": "acceptance_threshold_failed", "message": "至少一项固定影子验收门未通过，不得形成现场验收候选。"})

        start = min(_parse_timestamp(row["started_at"]) for row in records)
        end = max(_parse_timestamp(row["ended_at"]) for row in records)
        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "dataset_sha256": dataset_sha256,
            "source": normalized_source,
            "candidate_policy_version": candidate_version,
            "incumbent_policy_version": incumbent_version,
            "policy_bindings": normalized_input["policy"],
            "comparison_window": {"start_at": _iso_utc(start), "end_at": _iso_utc(end)},
            "cycles": records,
            "shadow_cycles": len(records),
            "operational_days": len(days),
            "scenario_counts": dict(sorted(Counter(row["scenario"] for row in records).items())),
            "asset_group_counts": dict(sorted(Counter(row["asset_group"] for row in records).items())),
            "metrics": metrics,
            "thresholds": dict(THRESHOLDS),
            "threshold_checks": checks,
            "acceptance_status": status,
            "guardrail_violation_rate": metrics["guardrail_violation_rate"],
            "measured_incumbent_baseline": measured_incumbent,
            "candidate_business_impact_measured": False,
            "approved": approved,
            "approved_by": {"operations": operations_reviewer or None, "safety": safety_reviewer or None},
            "approval_required": True,
            "provenance": {
                "type": "site_shadow_observation" if measured_incumbent else "contract_test_or_unverified_export",
                "source_attestation": measured_incumbent,
                "source_system": source_system,
                "cycle_reference_digest": _canonical_digest([
                    [row["cycle_id"], row["incumbent_reference_sha256"], row["candidate_receipt_sha256"]]
                    for row in records
                ]),
                "algorithm_implementation": "app.services.site_shadow_acceptance.SiteShadowAcceptanceService",
                "change_ticket": ticket or None,
                "rollback_drill": {
                    "verified": rollback_validation.get("verified") is True,
                    "reference": rollback_reference or None,
                    "evidence_sha256": rollback_validation.get("evidence_sha256"),
                    "executed_at": rollback_validation.get("executed_at"),
                    "rolledback_at": rollback_validation.get("rolledback_at"),
                    "approved_by": rollback_validation.get("approved_by"),
                },
            },
            "boundary": {
                "site_shadow_accepted": approved,
                "measured_incumbent_verified": measured_incumbent,
                "candidate_business_impact": "projected_not_measured",
                "dual_approval_verified": approved,
                "dispatch_allowed": False,
                "production_authority": False,
                "side_effects": False,
                "claim": "approved_site_shadow_acceptance_evidence" if approved else "contract_or_unapproved_shadow_candidate",
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
                "received_cycles": len(rows),
                "accepted_cycles": 0,
                "dataset_sha256": dataset_sha256,
                "evidence": None,
                "boundary": {"site_shadow_accepted": False, "dispatch_allowed": False, "production_authority": False},
            }
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "received_cycles": len(rows),
            "accepted_cycles": len(records),
            "dataset_sha256": dataset_sha256,
            "evidence": evidence,
            "boundary": dict(evidence["boundary"]),
        }
