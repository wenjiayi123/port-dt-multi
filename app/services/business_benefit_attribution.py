from __future__ import annotations

import hashlib
import json
import math
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np


DATASET_SCHEMA = "business_benefit_attribution_dataset.v1"
EVIDENCE_SCHEMA = "business_benefit_attribution_evidence.v1"
EVIDENCE_CLASSES = ("authorized_site_benefit_export", "contract_test_only")
DESIGN_METHODS = (
    "randomized_controlled_trial",
    "stepped_wedge_rollout",
    "matched_difference_in_differences",
)
METRICS: Dict[str, Dict[str, str]] = {
    "energy_cost_cny": {"label": "能源费用", "unit": "cny", "direction": "lower"},
    "waiting_time_minutes": {"label": "平均等待时间", "unit": "minute", "direction": "lower"},
    "berth_utilization_ratio": {"label": "泊位利用率", "unit": "ratio", "direction": "higher"},
    "throughput_teu": {"label": "集装箱吞吐量", "unit": "teu", "direction": "higher_noninferiority"},
    "carbon_emissions_kg": {"label": "碳排放量", "unit": "kilogram", "direction": "lower"},
    "unplanned_downtime_minutes": {"label": "设备非计划停机", "unit": "minute", "direction": "lower"},
    "maintenance_cost_cny": {"label": "设备维护费用", "unit": "cny", "direction": "lower"},
    "regulatory_delay_minutes": {"label": "监管作业延误", "unit": "minute", "direction": "lower"},
}
RAW_OUTCOME_FIELDS = (
    "energy_consumption_kwh",
    "tariff_cny_per_kwh",
    "carbon_factor_kg_per_kwh",
    "waiting_time_minutes",
    "berth_productive_minutes",
    "berth_available_minutes",
    "throughput_teu",
    "unplanned_downtime_minutes",
    "maintenance_cost_cny",
    "regulatory_delay_minutes",
)
MATCHING_FACTORS = (
    "cargo_teu",
    "vessel_size_teu",
    "weather_severity_ratio",
    "tide_m",
    "shift_index",
    "equipment_availability_ratio",
)
SOURCE_REFERENCES = (
    "energy_meter_reference",
    "tos_operations_reference",
    "emissions_factor_reference",
    "maintenance_reference",
    "regulatory_reference",
    "safety_reference",
)
BINDING_FIELDS = (
    "forecast_uncertainty_evidence_digest",
    "shadow_acceptance_evidence_digest",
    "execution_acceptance_evidence_digest",
    "collaboration_evidence_digest",
)
THRESHOLDS = {
    "primary_relative_effect_ci95_low_min_percent": 0.0,
    "throughput_relative_effect_ci95_low_min_percent": -2.0,
    "max_matching_standardized_difference": 0.25,
    "execution_receipt_coverage_min": 1.0,
    "outcome_source_coverage_min": 1.0,
    "candidate_execution_compliance_min": 1.0,
    "safety_incident_count_max": 0,
    "concurrent_intervention_count_max": 0,
}
MINIMUM_PAIRS = {"contract_test_only": 12, "authorized_site_benefit_export": 30}

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PLACEHOLDERS = {"unknown", "unset", "todo", "replace", "replace_me", "n/a", "none"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _named(value: Any) -> bool:
    text = _clean(value)
    return bool(text and text.lower() not in _PLACEHOLDERS)


def _parse_timestamp(value: Any) -> datetime:
    text = _clean(value)
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp requires an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


def _digest(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _metric_values(outcomes: Dict[str, float]) -> Dict[str, float]:
    available = outcomes["berth_available_minutes"]
    return {
        "energy_cost_cny": outcomes["energy_consumption_kwh"] * outcomes["tariff_cny_per_kwh"],
        "waiting_time_minutes": outcomes["waiting_time_minutes"],
        "berth_utilization_ratio": outcomes["berth_productive_minutes"] / available,
        "throughput_teu": outcomes["throughput_teu"],
        "carbon_emissions_kg": outcomes["energy_consumption_kwh"] * outcomes["carbon_factor_kg_per_kwh"],
        "unplanned_downtime_minutes": outcomes["unplanned_downtime_minutes"],
        "maintenance_cost_cny": outcomes["maintenance_cost_cny"],
        "regulatory_delay_minutes": outcomes["regulatory_delay_minutes"],
    }


def _benefit_effect(metric_id: str, raw_difference_in_differences: float) -> float:
    return -raw_difference_in_differences if METRICS[metric_id]["direction"] == "lower" else raw_difference_in_differences


class BusinessBenefitAttributionService:
    """Build execution-bound, paired business-benefit attribution evidence.

    This does not upgrade an offline counterfactual into a field KPI. A field
    claim needs pre-registered design, actual execution receipts, later measured
    outcomes, a concurrent comparator and three independent reviewers.
    """

    @staticmethod
    def _error(
        errors: List[Dict[str, Any]],
        code: str,
        field: str,
        message: str,
        *,
        row_index: int | None = None,
        pair_id: str | None = None,
    ) -> None:
        item: Dict[str, Any] = {"code": code, "field": field, "message": message}
        if row_index is not None:
            item["row_index"] = row_index
        if pair_id is not None:
            item["pair_id"] = pair_id
        errors.append(item)

    @staticmethod
    def _window(design: Dict[str, Any], name: str, errors: List[Dict[str, Any]]) -> tuple[datetime | None, datetime | None]:
        row = design.get(name) if isinstance(design.get(name), dict) else {}
        try:
            start = _parse_timestamp(row.get("start_at"))
            end = _parse_timestamp(row.get("end_at"))
            if end <= start:
                raise ValueError
            return start, end
        except (TypeError, ValueError):
            BusinessBenefitAttributionService._error(
                errors,
                "experiment_window",
                f"design.{name}",
                "window requires timezone-aware start_at before end_at",
            )
            return None, None

    @staticmethod
    def _bootstrap_summary(
        pair_effects: List[Dict[str, Any]],
        metric_id: str,
        *,
        seed: int,
        samples: int = 1000,
    ) -> Dict[str, Any]:
        benefits = np.asarray([row["effects"][metric_id]["benefit_absolute"] for row in pair_effects], dtype=np.float64)
        comparator = np.asarray([abs(row["effects"][metric_id]["incumbent_post"]) for row in pair_effects], dtype=np.float64)
        denominator = max(float(comparator.mean()), 1e-9)
        relative = benefits / np.maximum(comparator, 1e-9) * 100.0
        rng = np.random.default_rng(seed)
        boot_absolute = np.zeros(samples, dtype=np.float64)
        boot_relative = np.zeros(samples, dtype=np.float64)
        for index in range(samples):
            selected = rng.integers(0, len(pair_effects), size=len(pair_effects))
            boot_absolute[index] = float(benefits[selected].mean())
            boot_relative[index] = float(benefits[selected].mean() / max(float(comparator[selected].mean()), 1e-9) * 100.0)
        definition = METRICS[metric_id]
        return {
            "metric_id": metric_id,
            "label": definition["label"],
            "unit": definition["unit"],
            "direction": definition["direction"],
            "pair_count": len(pair_effects),
            "mean_incumbent_post": float(comparator.mean()),
            "mean_benefit_absolute": float(benefits.mean()),
            "mean_benefit_relative_percent": float(benefits.mean() / denominator * 100.0),
            "median_pair_relative_percent": float(np.median(relative)),
            "ci95_absolute": {
                "low": float(np.percentile(boot_absolute, 2.5)),
                "high": float(np.percentile(boot_absolute, 97.5)),
            },
            "ci95_relative_percent": {
                "low": float(np.percentile(boot_relative, 2.5)),
                "high": float(np.percentile(boot_relative, 97.5)),
            },
            "uncertainty_method": "paired_cluster_bootstrap_over_complete_difference_in_differences_pairs",
            "bootstrap_samples": samples,
        }

    @staticmethod
    def _threshold_checks(
        summaries: List[Dict[str, Any]],
        primary_metric: str,
        data_quality: Dict[str, Any],
    ) -> Dict[str, bool]:
        by_metric = {row["metric_id"]: row for row in summaries}
        return {
            "primary_effect": by_metric[primary_metric]["ci95_relative_percent"]["low"] > THRESHOLDS["primary_relative_effect_ci95_low_min_percent"],
            "throughput_noninferiority": by_metric["throughput_teu"]["ci95_relative_percent"]["low"] >= THRESHOLDS["throughput_relative_effect_ci95_low_min_percent"],
            "matching_balance": data_quality["max_matching_standardized_difference"] <= THRESHOLDS["max_matching_standardized_difference"],
            "execution_receipt_coverage": data_quality["execution_receipt_coverage"] >= THRESHOLDS["execution_receipt_coverage_min"],
            "outcome_source_coverage": data_quality["outcome_source_coverage"] >= THRESHOLDS["outcome_source_coverage_min"],
            "candidate_execution_compliance": data_quality["candidate_execution_compliance"] >= THRESHOLDS["candidate_execution_compliance_min"],
            "safety": data_quality["safety_incident_count"] <= THRESHOLDS["safety_incident_count_max"],
            "no_concurrent_intervention": data_quality["concurrent_intervention_count"] <= THRESHOLDS["concurrent_intervention_count_max"],
        }

    @staticmethod
    def validate_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {"valid": False, "errors": ["business benefit evidence must be an object"], "production_gate_eligible": False}
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        required = (
            "site_id", "run_id", "dataset_sha256", "source", "bindings", "design",
            "metric_contract", "unit_receipts", "pair_effects", "metric_summaries",
            "data_quality", "thresholds", "threshold_checks", "attribution_status",
            "measured_outcomes", "approved_by", "change_ticket", "approved", "boundary",
            "evidence_digest",
        )
        for field in required:
            if field not in payload:
                errors.append(f"missing {field}")
        evidence_digest = _clean(payload.get("evidence_digest"))
        body = deepcopy(payload)
        body.pop("evidence_digest", None)
        if not _SHA256.fullmatch(evidence_digest) or _digest(body) != evidence_digest:
            errors.append("evidence_digest mismatch")
        metric_contract = payload.get("metric_contract") if isinstance(payload.get("metric_contract"), list) else []
        expected_contract = [
            {"metric_id": metric_id, **definition}
            for metric_id, definition in METRICS.items()
        ]
        if metric_contract != expected_contract or payload.get("metric_contract_digest") != _digest(metric_contract):
            errors.append("metric contract or digest mismatch")
        unit_receipts = payload.get("unit_receipts") if isinstance(payload.get("unit_receipts"), list) else []
        pair_effects = payload.get("pair_effects") if isinstance(payload.get("pair_effects"), list) else []
        summaries = payload.get("metric_summaries") if isinstance(payload.get("metric_summaries"), list) else []
        if payload.get("unit_receipts_digest") != _digest(unit_receipts):
            errors.append("unit_receipts_digest mismatch")
        if payload.get("pair_effects_digest") != _digest(pair_effects):
            errors.append("pair_effects_digest mismatch")
        if {row.get("metric_id") for row in summaries if isinstance(row, dict)} != set(METRICS):
            errors.append("metric_summaries must contain all eight fixed metrics")
        if payload.get("metric_summaries_digest") != _digest(summaries):
            errors.append("metric_summaries_digest mismatch")
        if payload.get("thresholds") != THRESHOLDS:
            errors.append("fixed thresholds mismatch")
        design = payload.get("design") if isinstance(payload.get("design"), dict) else {}
        primary_metric = _clean(design.get("primary_metric"))
        if primary_metric not in METRICS or primary_metric == "throughput_teu":
            errors.append("primary metric is not eligible")
        data_quality = payload.get("data_quality") if isinstance(payload.get("data_quality"), dict) else {}
        try:
            expected_checks = BusinessBenefitAttributionService._threshold_checks(summaries, primary_metric, data_quality)
        except (KeyError, TypeError, ValueError):
            expected_checks = {}
            errors.append("metrics or data quality cannot reproduce threshold checks")
        if payload.get("threshold_checks") != expected_checks or not expected_checks or not all(expected_checks.values()):
            errors.append("threshold checks failed or were altered")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        bindings = payload.get("bindings") if isinstance(payload.get("bindings"), dict) else {}
        if any(not _SHA256.fullmatch(_clean(bindings.get(field))) for field in BINDING_FIELDS):
            errors.append("upstream evidence bindings are incomplete")
        pair_count = int(data_quality.get("complete_pair_count") or 0)
        if pair_count < MINIMUM_PAIRS["authorized_site_benefit_export"] or len(pair_effects) != pair_count or len(unit_receipts) != pair_count * 4:
            errors.append("authorized pair or unit-receipt coverage is incomplete")
        reviewers = payload.get("approved_by") if isinstance(payload.get("approved_by"), list) else []
        reviewer_ids = [_clean(row.get("reviewer_id")) for row in reviewers if isinstance(row, dict)]
        reviewer_roles = {row.get("role") for row in reviewers if isinstance(row, dict)}
        expected_roles = {"business_owner", "operations_assurance", "causal_methods"}
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        production_gate_eligible = bool(
            not errors
            and source.get("evidence_class") == "authorized_site_benefit_export"
            and source.get("source_verified") is True
            and design.get("method") in DESIGN_METHODS
            and payload.get("attribution_status") == "pass"
            and payload.get("measured_outcomes") is True
            and len(reviewer_ids) == 3
            and len(set(reviewer_ids)) == 3
            and reviewer_roles == expected_roles
            and all(_IDENTIFIER.fullmatch(item) for item in reviewer_ids)
            and all(item.lower() != _clean(source.get("owner")).lower() for item in reviewer_ids)
            and _IDENTIFIER.fullmatch(_clean(payload.get("change_ticket")))
            and payload.get("approved") is True
            and boundary.get("realized_business_benefit_verified") is True
            and boundary.get("field_kpi_claim_eligible") is True
            and boundary.get("automatic_resource_commitment_allowed") is False
            and boundary.get("dispatch_allowed") is False
            and boundary.get("production_authority") is False
        )
        return {"valid": not errors, "errors": errors, "production_gate_eligible": production_gate_eligible}

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_BUSINESS_BENEFIT_ATTRIBUTION_PATH"))
        artifact: Dict[str, Any] = {
            "mode": "unconfigured",
            "configured": False,
            "verified": False,
            "artifact_id": None,
            "sha256": None,
            "blockers": ["business_benefit_attribution_artifact_not_configured"],
        }
        if raw_path:
            path = Path(raw_path).expanduser()
            artifact.update(
                mode="configured_invalid",
                configured=True,
                artifact_id=path.name,
                blockers=["business_benefit_attribution_artifact_invalid"],
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validation = self.validate_evidence(payload)
                eligible = validation["production_gate_eligible"]
                design = payload.get("design") or {}
                summaries = payload.get("metric_summaries") or []
                primary = next((row for row in summaries if row.get("metric_id") == design.get("primary_metric")), {})
                artifact.update(
                    mode="verified_site_artifact" if eligible else "configured_invalid",
                    verified=eligible,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    site_id=payload.get("site_id"),
                    method=design.get("method"),
                    primary_metric=design.get("primary_metric"),
                    complete_pair_count=(payload.get("data_quality") or {}).get("complete_pair_count"),
                    primary_relative_effect_percent=primary.get("mean_benefit_relative_percent"),
                    primary_ci95_low_percent=(primary.get("ci95_relative_percent") or {}).get("low"),
                    attribution_status=payload.get("attribution_status"),
                    approved=payload.get("approved") is True,
                    blockers=[] if eligible else list(validation["errors"]) or ["field_benefit_not_approved"],
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        verified = artifact.get("verified") is True
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "configured_artifact": artifact,
            "contract": {
                "methods": list(DESIGN_METHODS),
                "metrics": [
                    {"metric_id": metric_id, **definition}
                    for metric_id, definition in METRICS.items()
                ],
                "required_bindings": list(BINDING_FIELDS),
                "matching_factors": list(MATCHING_FACTORS),
                "required_outcome_sources": list(SOURCE_REFERENCES),
                "minimum_complete_pairs": deepcopy(MINIMUM_PAIRS),
                "fixed_thresholds": dict(THRESHOLDS),
                "required_cells_per_pair": ["candidate/pre", "candidate/post", "incumbent/pre", "incumbent/post"],
                "uncertainty_method": "paired cluster bootstrap over complete difference-in-differences pairs",
            },
            "boundary": {
                "realized_business_benefit_verified": verified,
                "field_kpi_claim_eligible": verified,
                "measured_outcomes_verified": verified,
                "automatic_resource_commitment_allowed": False,
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "现场业务收益归因证据已验证" if verified else "待接入现场执行与同期对照结果",
                "reason": (
                    "预注册方案、实际执行、后验计量、同期对照、置信区间和三方独立复核均已验证；系统仍不自动承诺资源或控制设备。"
                    if verified
                    else "公开数据反事实、影子预测或合同样例不能替代现场实际执行、同期对照和后验计量。"
                ),
            },
        }

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        business_owner_approved_by: str | None = None,
        operations_assurance_approved_by: str | None = None,
        causal_methods_approved_by: str | None = None,
        change_ticket: str | None = None,
    ) -> Dict[str, Any]:
        payload = deepcopy(payload) if isinstance(payload, dict) else {}
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        if _clean(payload.get("schema_version")) != DATASET_SCHEMA:
            self._error(errors, "schema_version", "schema_version", f"must equal {DATASET_SCHEMA}")
        site_id = _clean(payload.get("site_id"))
        run_id = _clean(payload.get("run_id"))
        for field, value in (("site_id", site_id), ("run_id", run_id)):
            if not _IDENTIFIER.fullmatch(value):
                self._error(errors, "identifier", field, "must be a stable 3-128 character identifier")

        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        for field in ("source_system", "owner", "license"):
            if not _named(source.get(field)):
                self._error(errors, "source_metadata", f"source.{field}", "is required and cannot be a placeholder")
        evidence_class = _clean(source.get("evidence_class"))
        if evidence_class not in EVIDENCE_CLASSES:
            self._error(errors, "evidence_class", "source.evidence_class", f"must be one of {EVIDENCE_CLASSES}")
        timezone_name = _clean(source.get("timezone"))
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            self._error(errors, "timezone", "source.timezone", "must be an IANA timezone")
        try:
            extracted_at = _parse_timestamp(source.get("extracted_at"))
        except (TypeError, ValueError):
            extracted_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
            self._error(errors, "extracted_at", "source.extracted_at", "requires an explicit timezone")

        bindings = payload.get("bindings") if isinstance(payload.get("bindings"), dict) else {}
        normalized_bindings: Dict[str, str] = {}
        for field in BINDING_FIELDS:
            value = _clean(bindings.get(field))
            if not _SHA256.fullmatch(value):
                self._error(errors, "binding_digest", f"bindings.{field}", "must be a lowercase SHA-256 digest")
            normalized_bindings[field] = value

        design = payload.get("design") if isinstance(payload.get("design"), dict) else {}
        method = _clean(design.get("method"))
        if method not in DESIGN_METHODS:
            self._error(errors, "design_method", "design.method", f"must be one of {DESIGN_METHODS}")
        primary_metric = _clean(design.get("primary_metric"))
        if primary_metric not in METRICS or primary_metric == "throughput_teu":
            self._error(errors, "primary_metric", "design.primary_metric", "must be a fixed benefit metric other than throughput non-inferiority")
        protocol_id = _clean(design.get("protocol_id"))
        assignment_reference = _clean(design.get("assignment_reference"))
        interference_scope = _clean(design.get("interference_scope"))
        for field, value in (
            ("protocol_id", protocol_id),
            ("assignment_reference", assignment_reference),
            ("interference_scope", interference_scope),
        ):
            if not _IDENTIFIER.fullmatch(value):
                self._error(errors, "design_identifier", f"design.{field}", "stable design identifier is required")
        protocol_sha256 = _clean(design.get("protocol_sha256"))
        analysis_plan_sha256 = _clean(design.get("analysis_plan_sha256"))
        for field, value in (("protocol_sha256", protocol_sha256), ("analysis_plan_sha256", analysis_plan_sha256)):
            if not _SHA256.fullmatch(value):
                self._error(errors, "design_digest", f"design.{field}", "must be a lowercase SHA-256 digest")
        pre_start, pre_end = self._window(design, "pre_window", errors)
        post_start, post_end = self._window(design, "post_window", errors)
        if pre_end and post_start and pre_end >= post_start:
            self._error(errors, "window_overlap", "design", "pre and post windows must not overlap")
        try:
            protocol_registered_at = _parse_timestamp(design.get("protocol_registered_at"))
            assignment_at = _parse_timestamp(design.get("assignment_at"))
            if post_start and not protocol_registered_at < assignment_at <= post_start:
                raise ValueError
        except (TypeError, ValueError):
            protocol_registered_at = assignment_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
            self._error(errors, "preregistration", "design", "protocol_registered_at must precede assignment_at, which must not follow the post window start")

        units = payload.get("units")
        if not isinstance(units, list):
            self._error(errors, "units", "units", "units must be an array")
            units = []
        elif len(units) > 100000:
            self._error(errors, "unit_limit", "units", "an attribution bundle may contain at most one hundred thousand units")
        normalized_units: List[Dict[str, Any]] = []
        unit_ids: set[str] = set()
        execution_ids: set[str] = set()
        cells: Dict[str, Dict[tuple[str, str], Dict[str, Any]]] = {}
        for row_index, raw in enumerate(units):
            if not isinstance(raw, dict):
                self._error(errors, "unit_type", "units", "unit must be an object", row_index=row_index)
                continue
            before = len(errors)
            pair_id = _clean(raw.get("pair_id"))
            unit_id = _clean(raw.get("unit_id"))
            cluster_id = _clean(raw.get("cluster_id"))
            for field, value in (("pair_id", pair_id), ("unit_id", unit_id), ("cluster_id", cluster_id)):
                if not _IDENTIFIER.fullmatch(value):
                    self._error(errors, "unit_identifier", field, "stable identifier is required", row_index=row_index, pair_id=pair_id or None)
            if unit_id in unit_ids:
                self._error(errors, "duplicate_unit", "unit_id", "unit identifier must be unique", row_index=row_index, pair_id=pair_id)
            unit_ids.add(unit_id)
            group = _clean(raw.get("group"))
            period = _clean(raw.get("period"))
            if group not in {"candidate", "incumbent"} or period not in {"pre", "post"}:
                self._error(errors, "cell", "group/period", "group must be candidate or incumbent and period must be pre or post", row_index=row_index, pair_id=pair_id)
            try:
                started_at = _parse_timestamp(raw.get("started_at"))
                ended_at = _parse_timestamp(raw.get("ended_at"))
                if not started_at < ended_at <= extracted_at:
                    raise ValueError
                if period == "pre" and pre_start and pre_end and not (pre_start <= started_at and ended_at <= pre_end):
                    raise ValueError
                if period == "post" and post_start and post_end and not (post_start <= started_at and ended_at <= post_end):
                    raise ValueError
            except (TypeError, ValueError):
                started_at = ended_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
                self._error(errors, "unit_time", "started_at/ended_at", "unit interval must belong to its declared experiment window", row_index=row_index, pair_id=pair_id)

            factors_payload = raw.get("matching_factors") if isinstance(raw.get("matching_factors"), dict) else {}
            if set(factors_payload) != set(MATCHING_FACTORS):
                self._error(errors, "matching_factor_fields", "matching_factors", "matching-factor keys must exactly match the fixed contract", row_index=row_index, pair_id=pair_id)
            factors: Dict[str, float] = {}
            for field in MATCHING_FACTORS:
                try:
                    factors[field] = _finite(factors_payload.get(field))
                except (TypeError, ValueError):
                    self._error(errors, "matching_factor_value", f"matching_factors.{field}", "finite numeric value is required", row_index=row_index, pair_id=pair_id)
            if factors.get("cargo_teu", 0.0) < 0.0 or factors.get("vessel_size_teu", 0.0) <= 0.0:
                self._error(errors, "matching_factor_range", "matching_factors", "cargo and vessel size must be physically valid", row_index=row_index, pair_id=pair_id)
            for field in ("weather_severity_ratio", "equipment_availability_ratio"):
                if field in factors and not 0.0 <= factors[field] <= 1.0:
                    self._error(errors, "matching_factor_range", f"matching_factors.{field}", "ratio must be from zero to one", row_index=row_index, pair_id=pair_id)
            if "tide_m" in factors and not -20.0 <= factors["tide_m"] <= 20.0:
                self._error(errors, "matching_factor_range", "matching_factors.tide_m", "tide is outside physical bounds", row_index=row_index, pair_id=pair_id)

            outcomes_payload = raw.get("outcomes") if isinstance(raw.get("outcomes"), dict) else {}
            if set(outcomes_payload) != set(RAW_OUTCOME_FIELDS):
                self._error(errors, "outcome_fields", "outcomes", "outcome keys must exactly match the fixed contract", row_index=row_index, pair_id=pair_id)
            outcomes: Dict[str, float] = {}
            for field in RAW_OUTCOME_FIELDS:
                try:
                    outcomes[field] = _finite(outcomes_payload.get(field))
                    if outcomes[field] < 0.0:
                        raise ValueError
                except (TypeError, ValueError):
                    self._error(errors, "outcome_value", f"outcomes.{field}", "finite nonnegative outcome is required", row_index=row_index, pair_id=pair_id)
            if outcomes.get("berth_available_minutes", 0.0) <= 0.0 or outcomes.get("berth_productive_minutes", 0.0) > outcomes.get("berth_available_minutes", 0.0):
                self._error(errors, "berth_outcome", "outcomes", "productive berth minutes must not exceed positive available minutes", row_index=row_index, pair_id=pair_id)

            refs_payload = raw.get("source_references") if isinstance(raw.get("source_references"), dict) else {}
            if set(refs_payload) != set(SOURCE_REFERENCES):
                self._error(errors, "outcome_source_fields", "source_references", "all fixed measured-outcome source references are required", row_index=row_index, pair_id=pair_id)
            refs: Dict[str, str] = {}
            for field in SOURCE_REFERENCES:
                refs[field] = _clean(refs_payload.get(field))
                if not _IDENTIFIER.fullmatch(refs[field]):
                    self._error(errors, "outcome_source", f"source_references.{field}", "stable source reference is required", row_index=row_index, pair_id=pair_id)

            decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
            execution_receipt_id = _clean(decision.get("execution_receipt_id"))
            actual_plan_reference = _clean(decision.get("actual_plan_reference"))
            if not _IDENTIFIER.fullmatch(execution_receipt_id) or execution_receipt_id in execution_ids:
                self._error(errors, "execution_receipt", "decision.execution_receipt_id", "unique actual execution receipt is required", row_index=row_index, pair_id=pair_id)
            execution_ids.add(execution_receipt_id)
            if not _IDENTIFIER.fullmatch(actual_plan_reference):
                self._error(errors, "actual_plan", "decision.actual_plan_reference", "actual plan reference is required", row_index=row_index, pair_id=pair_id)
            try:
                executed_at = _parse_timestamp(decision.get("executed_at"))
                if not started_at <= executed_at <= ended_at:
                    raise ValueError
            except (TypeError, ValueError):
                executed_at = started_at
                self._error(errors, "execution_time", "decision.executed_at", "execution time must fall inside the unit interval", row_index=row_index, pair_id=pair_id)
            candidate_executed = decision.get("candidate_executed") is True
            recommendation_receipt_id = _clean(decision.get("recommendation_receipt_id"))
            human_approval_receipt_id = _clean(decision.get("human_approval_receipt_id"))
            recommendation_digest = _clean(decision.get("recommendation_digest"))
            if group == "candidate" and period == "post":
                if not candidate_executed:
                    self._error(errors, "candidate_execution", "decision.candidate_executed", "candidate post cell must represent an actually executed accepted recommendation", row_index=row_index, pair_id=pair_id)
                for field, value in (
                    ("recommendation_receipt_id", recommendation_receipt_id),
                    ("human_approval_receipt_id", human_approval_receipt_id),
                ):
                    if not _IDENTIFIER.fullmatch(value):
                        self._error(errors, "candidate_receipt", f"decision.{field}", "stable receipt is required for candidate post execution", row_index=row_index, pair_id=pair_id)
                if not _SHA256.fullmatch(recommendation_digest):
                    self._error(errors, "candidate_digest", "decision.recommendation_digest", "recommendation digest is required for candidate post execution", row_index=row_index, pair_id=pair_id)
            elif candidate_executed or recommendation_receipt_id or human_approval_receipt_id or recommendation_digest:
                self._error(errors, "treatment_contamination", "decision", "only the candidate post cell may execute a candidate recommendation", row_index=row_index, pair_id=pair_id)

            other_interventions = raw.get("other_intervention_ids")
            if not isinstance(other_interventions, list) or any(not _IDENTIFIER.fullmatch(_clean(item)) for item in other_interventions):
                self._error(errors, "other_interventions", "other_intervention_ids", "must be an array of stable identifiers", row_index=row_index, pair_id=pair_id)
                other_interventions = []
            try:
                safety_incidents = int(raw.get("safety_incident_count"))
                if isinstance(raw.get("safety_incident_count"), bool) or safety_incidents < 0:
                    raise ValueError
            except (TypeError, ValueError):
                safety_incidents = 0
                self._error(errors, "safety_incidents", "safety_incident_count", "must be a nonnegative integer", row_index=row_index, pair_id=pair_id)
            row_assignment_reference = _clean(raw.get("assignment_reference"))
            if row_assignment_reference != assignment_reference:
                self._error(errors, "assignment_binding", "assignment_reference", "unit must bind the declared assignment reference", row_index=row_index, pair_id=pair_id)

            cell_key = (group, period)
            if pair_id in cells and cell_key in cells[pair_id]:
                self._error(errors, "duplicate_cell", "group/period", "pair may contain each cell exactly once", row_index=row_index, pair_id=pair_id)
            if len(errors) == before:
                normalized = {
                    "pair_id": pair_id,
                    "unit_id": unit_id,
                    "cluster_id": cluster_id,
                    "group": group,
                    "period": period,
                    "started_at": _iso(started_at),
                    "ended_at": _iso(ended_at),
                    "assignment_reference": row_assignment_reference,
                    "matching_factors": factors,
                    "outcomes": outcomes,
                    "derived_metrics": _metric_values(outcomes),
                    "source_references": refs,
                    "decision": {
                        "actual_plan_reference": actual_plan_reference,
                        "execution_receipt_id": execution_receipt_id,
                        "executed_at": _iso(executed_at),
                        "candidate_executed": candidate_executed,
                        "recommendation_receipt_id": recommendation_receipt_id,
                        "human_approval_receipt_id": human_approval_receipt_id,
                        "recommendation_digest": recommendation_digest,
                    },
                    "other_intervention_ids": [_clean(item) for item in other_interventions],
                    "safety_incident_count": safety_incidents,
                }
                normalized_units.append(normalized)
                cells.setdefault(pair_id, {})[cell_key] = normalized

        required_cells = {("candidate", "pre"), ("candidate", "post"), ("incumbent", "pre"), ("incumbent", "post")}
        minimum_pairs = MINIMUM_PAIRS.get(evidence_class, MINIMUM_PAIRS["contract_test_only"])
        complete_pair_ids: List[str] = []
        all_cluster_ids: set[str] = set()
        for pair_id, pair_cells in sorted(cells.items()):
            if set(pair_cells) != required_cells:
                self._error(errors, "incomplete_pair", "units", "pair requires candidate/incumbent pre/post cells", pair_id=pair_id)
                continue
            candidate_clusters = {pair_cells[("candidate", period)]["cluster_id"] for period in ("pre", "post")}
            incumbent_clusters = {pair_cells[("incumbent", period)]["cluster_id"] for period in ("pre", "post")}
            if len(candidate_clusters) != 1 or len(incumbent_clusters) != 1 or candidate_clusters == incumbent_clusters:
                self._error(errors, "cluster_binding", "cluster_id", "each group must retain one distinct cluster across pre and post", pair_id=pair_id)
                continue
            clusters = candidate_clusters | incumbent_clusters
            if clusters & all_cluster_ids:
                self._error(errors, "cluster_reuse", "cluster_id", "clusters may not be reused across independent pairs", pair_id=pair_id)
                continue
            all_cluster_ids.update(clusters)
            complete_pair_ids.append(pair_id)
        if len(complete_pair_ids) < minimum_pairs:
            self._error(errors, "complete_pairs", "units", f"at least {minimum_pairs} complete independent pairs are required")
        if len(normalized_units) != len(units):
            errors.append({"code": "unit_rejection", "field": "units", "message": "all submitted units must pass validation"})

        normalized_source = {
            "source_system": _clean(source.get("source_system")),
            "owner": _clean(source.get("owner")),
            "license": _clean(source.get("license")),
            "timezone": timezone_name,
            "extracted_at": _iso(extracted_at),
            "evidence_class": evidence_class,
            "source_verified": bool(source_verified and evidence_class == "authorized_site_benefit_export"),
        }
        normalized_design = {
            "method": method,
            "primary_metric": primary_metric,
            "protocol_id": protocol_id,
            "protocol_sha256": protocol_sha256,
            "analysis_plan_sha256": analysis_plan_sha256,
            "protocol_registered_at": _iso(protocol_registered_at),
            "assignment_reference": assignment_reference,
            "assignment_at": _iso(assignment_at),
            "interference_scope": interference_scope,
            "pre_window": {"start_at": _iso(pre_start) if pre_start else None, "end_at": _iso(pre_end) if pre_end else None},
            "post_window": {"start_at": _iso(post_start) if post_start else None, "end_at": _iso(post_end) if post_end else None},
        }
        normalized_input = {
            "schema_version": DATASET_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "source": {key: value for key, value in normalized_source.items() if key != "source_verified"},
            "bindings": normalized_bindings,
            "design": normalized_design,
            "units": sorted(normalized_units, key=lambda row: (row["pair_id"], row["group"], row["period"])),
        }
        dataset_sha256 = _digest(normalized_input)
        if errors:
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "dataset_sha256": None,
                "evidence": None,
                "boundary": {
                    "realized_business_benefit_verified": False,
                    "field_kpi_claim_eligible": False,
                    "automatic_resource_commitment_allowed": False,
                    "dispatch_allowed": False,
                    "production_authority": False,
                    "claim": "benefit_attribution_dataset_rejected",
                },
            }

        pair_effects: List[Dict[str, Any]] = []
        for pair_id in complete_pair_ids:
            pair_cells = cells[pair_id]
            candidate_pre = pair_cells[("candidate", "pre")]
            candidate_post = pair_cells[("candidate", "post")]
            incumbent_pre = pair_cells[("incumbent", "pre")]
            incumbent_post = pair_cells[("incumbent", "post")]
            effects: Dict[str, Any] = {}
            for metric_id in METRICS:
                candidate_change = candidate_post["derived_metrics"][metric_id] - candidate_pre["derived_metrics"][metric_id]
                incumbent_change = incumbent_post["derived_metrics"][metric_id] - incumbent_pre["derived_metrics"][metric_id]
                raw_did = candidate_change - incumbent_change
                benefit = _benefit_effect(metric_id, raw_did)
                effects[metric_id] = {
                    "candidate_pre": candidate_pre["derived_metrics"][metric_id],
                    "candidate_post": candidate_post["derived_metrics"][metric_id],
                    "incumbent_pre": incumbent_pre["derived_metrics"][metric_id],
                    "incumbent_post": incumbent_post["derived_metrics"][metric_id],
                    "raw_difference_in_differences": raw_did,
                    "benefit_absolute": benefit,
                    "benefit_relative_to_incumbent_post_percent": benefit / max(abs(incumbent_post["derived_metrics"][metric_id]), 1e-9) * 100.0,
                }
            pair_effects.append({
                "pair_id": pair_id,
                "candidate_cluster_id": candidate_post["cluster_id"],
                "incumbent_cluster_id": incumbent_post["cluster_id"],
                "candidate_recommendation_digest": candidate_post["decision"]["recommendation_digest"],
                "candidate_execution_receipt_id": candidate_post["decision"]["execution_receipt_id"],
                "effects": effects,
            })

        post_candidate = [cells[pair_id][("candidate", "post")] for pair_id in complete_pair_ids]
        post_incumbent = [cells[pair_id][("incumbent", "post")] for pair_id in complete_pair_ids]
        standardized_differences: Dict[str, float] = {}
        for factor in MATCHING_FACTORS:
            candidate_values = np.asarray([row["matching_factors"][factor] for row in post_candidate], dtype=np.float64)
            incumbent_values = np.asarray([row["matching_factors"][factor] for row in post_incumbent], dtype=np.float64)
            pooled_scale = math.sqrt((float(candidate_values.var()) + float(incumbent_values.var())) / 2.0)
            standardized_differences[factor] = abs(float(candidate_values.mean() - incumbent_values.mean())) / max(pooled_scale, 1e-9)
        concurrent_count = sum(len(row["other_intervention_ids"]) for row in normalized_units)
        safety_count = sum(row["safety_incident_count"] for row in normalized_units)
        candidate_post_count = len(post_candidate)
        data_quality = {
            "complete_pair_count": len(complete_pair_ids),
            "unit_receipt_count": len(normalized_units),
            "required_cells_complete": len(normalized_units) == len(complete_pair_ids) * 4,
            "execution_receipt_coverage": sum(bool(row["decision"]["execution_receipt_id"]) for row in normalized_units) / len(normalized_units),
            "outcome_source_coverage": sum(all(row["source_references"].values()) for row in normalized_units) / len(normalized_units),
            "candidate_execution_compliance": sum(row["decision"]["candidate_executed"] for row in post_candidate) / max(candidate_post_count, 1),
            "candidate_post_execution_count": sum(row["decision"]["candidate_executed"] for row in post_candidate),
            "concurrent_intervention_count": concurrent_count,
            "safety_incident_count": safety_count,
            "matching_standardized_differences": standardized_differences,
            "max_matching_standardized_difference": max(standardized_differences.values()),
            "chronological_pre_post_windows": True,
            "pre_registered_before_assignment": True,
            "independent_cluster_pairs": True,
        }
        summaries = [
            self._bootstrap_summary(
                pair_effects,
                metric_id,
                seed=int(hashlib.sha256(f"{dataset_sha256}:{metric_id}".encode("utf-8")).hexdigest()[:16], 16),
            )
            for metric_id in METRICS
        ]
        threshold_checks = self._threshold_checks(summaries, primary_metric, data_quality)
        attribution_status = "pass" if all(threshold_checks.values()) else "fail"
        measured_outcomes = bool(
            evidence_class == "authorized_site_benefit_export"
            and source_verified
            and attribution_status == "pass"
        )
        reviewer_rows = [
            {"role": "business_owner", "reviewer_id": _clean(business_owner_approved_by)},
            {"role": "operations_assurance", "reviewer_id": _clean(operations_assurance_approved_by)},
            {"role": "causal_methods", "reviewer_id": _clean(causal_methods_approved_by)},
        ]
        reviewer_ids = [row["reviewer_id"] for row in reviewer_rows]
        ticket = _clean(change_ticket)
        owner = normalized_source["owner"]
        approval_metadata_valid = bool(
            all(_IDENTIFIER.fullmatch(item) for item in reviewer_ids)
            and len(set(reviewer_ids)) == 3
            and all(item.lower() != owner.lower() for item in reviewer_ids)
            and _IDENTIFIER.fullmatch(ticket)
        )
        approved = bool(measured_outcomes and approval_metadata_valid)
        if evidence_class == "contract_test_only":
            warnings.append({"code": "contract_only", "message": "合同样例只验证成对归因计算，不构成现场已实现收益。"})
        elif not source_verified:
            warnings.append({"code": "source_attestation_missing", "message": "现场执行、计量和同期对照来源尚未完成授权证明。"})
        if attribution_status != "pass":
            warnings.append({"code": "attribution_gate_failed", "message": "主指标置信区间、吞吐非劣、匹配平衡、执行覆盖、安全或同期干预门禁未通过。"})
        if measured_outcomes and any(reviewer_ids + [ticket]) and not approval_metadata_valid:
            warnings.append({"code": "independent_approval_invalid", "message": "业务责任人、运营保证和因果方法复核人必须互异、独立于数据责任主体，并绑定稳定变更单。"})
        if method == "matched_difference_in_differences":
            warnings.append({"code": "parallel_trends_boundary", "message": "匹配双重差分仍依赖平行趋势和无未记录同期干预假设；该方法不能证明所有未观测混杂均已消除。"})
        metric_contract = [
            {"metric_id": metric_id, **definition}
            for metric_id, definition in METRICS.items()
        ]
        boundary = {
            "realized_business_benefit_verified": approved,
            "field_kpi_claim_eligible": approved,
            "measured_outcomes_verified": measured_outcomes,
            "causal_design_gate_passed": attribution_status == "pass",
            "offline_counterfactual_relabelled_as_field_kpi": False,
            "automatic_resource_commitment_allowed": False,
            "dispatch_allowed": False,
            "production_authority": False,
            "claim": "approved_field_benefit_attribution" if approved else "contract_or_unapproved_benefit_attribution",
            "reason": (
                "现场实际执行、后验计量、同期对照、预注册分析和三方独立复核均已通过；结果只支持本协议、本窗口和本场站的收益声明。"
                if approved
                else "公开数据反事实、合同样例或缺少独立复核的现场数据不能形成已实现收益声明。"
            ),
        }
        evidence: Dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "dataset_sha256": dataset_sha256,
            "source": normalized_source,
            "bindings": normalized_bindings,
            "design": normalized_design,
            "metric_contract": metric_contract,
            "metric_contract_digest": _digest(metric_contract),
            "unit_receipts": normalized_input["units"],
            "unit_receipts_digest": _digest(normalized_input["units"]),
            "pair_effects": pair_effects,
            "pair_effects_digest": _digest(pair_effects),
            "metric_summaries": summaries,
            "metric_summaries_digest": _digest(summaries),
            "data_quality": data_quality,
            "thresholds": dict(THRESHOLDS),
            "threshold_checks": threshold_checks,
            "attribution_status": attribution_status,
            "measured_outcomes": measured_outcomes,
            "approved_by": reviewer_rows if approved else [],
            "change_ticket": ticket if approved else "",
            "approved": approved,
            "boundary": boundary,
        }
        evidence["evidence_digest"] = _digest(evidence)
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "dataset_sha256": dataset_sha256,
            "evidence": evidence,
            "boundary": boundary,
        }
