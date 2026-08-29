from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np


DATASET_SCHEMA = "forecast_uncertainty_dataset.v1"
EVIDENCE_SCHEMA = "forecast_uncertainty_evidence.v1"
EVIDENCE_CLASSES = ("authorized_site_forecast_export", "contract_test_only")
TARGETS: Dict[str, Dict[str, Any]] = {
    "vessel_eta_minutes": {
        "label": "船舶预计到港时间",
        "unit": "minute",
        "kind": "continuous",
        "features": ("distance_nm", "speed_knots", "channel_wait_minutes", "wind_mps", "current_knots"),
        "bounds": (0.0, 10080.0),
    },
    "berth_duration_minutes": {
        "label": "泊位作业时长",
        "unit": "minute",
        "kind": "continuous",
        "features": ("container_moves", "crane_count", "crane_productivity", "labor_availability", "yard_congestion_ratio"),
        "bounds": (0.0, 20160.0),
    },
    "quay_crane_productivity_mph": {
        "label": "岸桥作业效率",
        "unit": "move_per_hour",
        "kind": "continuous",
        "features": ("crane_age_years", "wind_mps", "labor_availability", "yard_congestion_ratio", "equipment_availability"),
        "bounds": (0.0, 100.0),
    },
    "yard_congestion_ratio": {
        "label": "堆场拥堵程度",
        "unit": "ratio",
        "kind": "continuous",
        "features": ("yard_occupancy_ratio", "gate_queue_trucks", "planned_discharge_units", "rail_backlog_units", "horizontal_transport_availability"),
        "bounds": (0.0, 1.0),
    },
    "equipment_failure_probability": {
        "label": "设备故障风险",
        "unit": "probability",
        "kind": "binary_probability",
        "features": ("runtime_hours", "vibration_mm_s", "temperature_c", "fault_count_24h", "maintenance_overdue_hours"),
        "bounds": (0.0, 1.0),
    },
    "weather_stoppage_probability": {
        "label": "天气停工风险",
        "unit": "probability",
        "kind": "binary_probability",
        "features": ("wind_mps", "wave_height_m", "visibility_m", "precipitation_mm_h", "lightning_distance_km"),
        "bounds": (0.0, 1.0),
    },
    "regulatory_delay_minutes": {
        "label": "监管作业延误",
        "unit": "minute",
        "kind": "continuous",
        "features": ("inspection_load", "document_completeness_ratio", "dangerous_goods_indicator", "authority_queue", "exception_count"),
        "bounds": (0.0, 10080.0),
    },
    "energy_load_kw": {
        "label": "港区能源负荷",
        "unit": "kilowatt",
        "kind": "continuous",
        "features": ("vessel_calls", "crane_moves", "reefer_count", "shore_power_kw", "yard_occupancy_ratio", "ambient_c"),
        "bounds": (0.0, 1_000_000.0),
    },
}
THRESHOLDS = {
    "normalized_mae_max": 0.20,
    "coverage_80_min": 0.65,
    "coverage_95_min": 0.85,
    "normalized_width_80_max": 0.75,
    "normalized_width_95_max": 1.10,
    "max_standardized_feature_shift_max": 1.50,
    "brier_score_max": 0.22,
    "expected_calibration_error_max": 0.20,
}
MINIMUM_ROWS = {
    "contract_test_only": {"training": 24, "calibration": 12, "test": 12},
    "authorized_site_forecast_export": {"training": 72, "calibration": 24, "test": 24},
}

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


def _conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(residuals, dtype=np.float64))
    index = min(len(ordered) - 1, max(0, int(math.ceil((len(ordered) + 1) * coverage)) - 1))
    return float(ordered[index])


class ForecastUncertaintyService:
    """Build auditable multi-target forecasts and chronological interval evidence.

    A browser call can only exercise the calculation contract. Production
    eligibility requires authorized, pre-issued site records, measured outcomes,
    three independent reviewers and a versioned offline artifact.
    """

    @staticmethod
    def _error(
        errors: List[Dict[str, Any]],
        code: str,
        field: str,
        message: str,
        *,
        target_id: str | None = None,
        row_index: int | None = None,
    ) -> None:
        item: Dict[str, Any] = {"code": code, "field": field, "message": message}
        if target_id is not None:
            item["target_id"] = target_id
        if row_index is not None:
            item["row_index"] = row_index
        errors.append(item)

    @staticmethod
    def _window(split: Dict[str, Any], name: str, errors: List[Dict[str, Any]]) -> tuple[datetime | None, datetime | None]:
        row = split.get(name) if isinstance(split.get(name), dict) else {}
        try:
            start = _parse_timestamp(row.get("start_at"))
            end = _parse_timestamp(row.get("end_at"))
            if end <= start:
                raise ValueError("end must follow start")
            return start, end
        except (TypeError, ValueError):
            ForecastUncertaintyService._error(
                errors,
                "forecast_window",
                f"split.{name}",
                "window requires timezone-aware start_at before end_at",
            )
            return None, None

    @staticmethod
    def _fit_ridge(matrix: np.ndarray, target: np.ndarray, alpha: float) -> Dict[str, Any]:
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        if np.any(scales < 1e-9):
            raise ValueError("each training feature must vary")
        standardized = (matrix - means) / scales
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
        penalty[0, 0] = 0.0
        try:
            fitted = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        except np.linalg.LinAlgError:
            fitted = np.linalg.lstsq(design.T @ design + penalty, design.T @ target, rcond=None)[0]
        coefficients = fitted[1:] / scales
        intercept = float(fitted[0] - np.dot(means, coefficients))
        return {
            "intercept": intercept,
            "coefficients": coefficients,
            "training_feature_mean": means,
            "training_feature_std": scales,
        }

    @staticmethod
    def _predict(model: Dict[str, Any], matrix: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        predicted = float(model["intercept"]) + matrix @ np.asarray(model["coefficients"], dtype=np.float64)
        return np.clip(predicted, bounds[0], bounds[1])

    @staticmethod
    def _ece(observed: np.ndarray, predicted: np.ndarray) -> float:
        result = 0.0
        for index in range(5):
            low, high = index / 5.0, (index + 1) / 5.0
            mask = (predicted >= low) & ((predicted < high) if index < 4 else (predicted <= high))
            if np.any(mask):
                result += float(mask.mean()) * abs(float(predicted[mask].mean()) - float(observed[mask].mean()))
        return float(result)

    @staticmethod
    def validate_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {"valid": False, "errors": ["forecast evidence must be an object"], "production_gate_eligible": False}
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        required = (
            "site_id", "run_id", "dataset_sha256", "source", "target_contract", "split",
            "row_counts", "models", "metrics_by_target", "test_receipts", "calibration_status",
            "measured_outcomes", "live_service_level_verified", "approved_by", "change_ticket",
            "approved", "boundary", "evidence_digest",
        )
        for field in required:
            if field not in payload:
                errors.append(f"missing {field}")
        digest = _clean(payload.get("evidence_digest"))
        body = deepcopy(payload)
        body.pop("evidence_digest", None)
        if not _SHA256.fullmatch(digest) or _digest(body) != digest:
            errors.append("evidence_digest mismatch")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        counts = payload.get("row_counts") if isinstance(payload.get("row_counts"), dict) else {}
        metrics = payload.get("metrics_by_target") if isinstance(payload.get("metrics_by_target"), list) else []
        target_ids = {row.get("target_id") for row in metrics if isinstance(row, dict)}
        if target_ids != set(TARGETS):
            errors.append("metrics_by_target must contain all eight fixed targets")
        target_contract = payload.get("target_contract") if isinstance(payload.get("target_contract"), list) else []
        expected_contract = [
            {
                "target_id": target_id,
                "label": definition["label"],
                "unit": definition["unit"],
                "kind": definition["kind"],
                "features": list(definition["features"]),
            }
            for target_id, definition in TARGETS.items()
        ]
        if target_contract != expected_contract or payload.get("target_contract_digest") != _digest(target_contract):
            errors.append("target contract or target_contract_digest mismatch")
        models = payload.get("models") if isinstance(payload.get("models"), list) else []
        if {row.get("target_id") for row in models if isinstance(row, dict)} != set(TARGETS):
            errors.append("models must contain all eight fixed targets")
        if payload.get("model_bundle_digest") != _digest(models):
            errors.append("model_bundle_digest mismatch")
        receipts = payload.get("test_receipts") if isinstance(payload.get("test_receipts"), list) else []
        if payload.get("test_receipts_digest") != _digest(receipts):
            errors.append("test_receipts_digest mismatch")
        if payload.get("thresholds") != THRESHOLDS:
            errors.append("fixed thresholds mismatch")
        split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
        try:
            train_start = _parse_timestamp(split["training_window"]["start_at"])
            train_end = _parse_timestamp(split["training_window"]["end_at"])
            calibration_start = _parse_timestamp(split["calibration_window"]["start_at"])
            calibration_end = _parse_timestamp(split["calibration_window"]["end_at"])
            test_start = _parse_timestamp(split["test_window"]["start_at"])
            test_end = _parse_timestamp(split["test_window"]["end_at"])
            if not train_start < train_end < calibration_start < calibration_end < test_start < test_end:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append("chronological split windows are invalid")
        minimums = MINIMUM_ROWS["authorized_site_forecast_export"]
        for target_id in TARGETS:
            target_counts = counts.get(target_id) if isinstance(counts.get(target_id), dict) else {}
            for split_name, minimum in minimums.items():
                if int(target_counts.get(split_name) or 0) < minimum:
                    errors.append(f"{target_id} {split_name} rows below authorized minimum")
            target_receipts = [row for row in receipts if isinstance(row, dict) and row.get("target_id") == target_id]
            if len(target_receipts) != int(target_counts.get("test") or 0):
                errors.append(f"{target_id} test receipt count mismatch")
        for row in metrics:
            if not isinstance(row, dict) or row.get("target_id") not in TARGETS:
                continue
            target_id = row["target_id"]
            definition = TARGETS[target_id]
            values = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            checks = row.get("threshold_checks") if isinstance(row.get("threshold_checks"), dict) else {}
            try:
                expected_checks = {
                    "normalized_mae": _finite(values.get("normalized_mae")) <= THRESHOLDS["normalized_mae_max"],
                    "coverage_80": _finite(values.get("coverage_80")) >= THRESHOLDS["coverage_80_min"],
                    "coverage_95": _finite(values.get("coverage_95")) >= THRESHOLDS["coverage_95_min"],
                    "normalized_width_80": _finite(values.get("normalized_interval_width_80")) <= THRESHOLDS["normalized_width_80_max"],
                    "normalized_width_95": _finite(values.get("normalized_interval_width_95")) <= THRESHOLDS["normalized_width_95_max"],
                    "feature_shift": _finite(values.get("max_standardized_feature_shift")) <= THRESHOLDS["max_standardized_feature_shift_max"],
                }
                if definition["kind"] == "binary_probability":
                    expected_checks.update(
                        brier_score=_finite(values.get("brier_score")) <= THRESHOLDS["brier_score_max"],
                        expected_calibration_error=_finite(values.get("expected_calibration_error")) <= THRESHOLDS["expected_calibration_error_max"],
                    )
            except (TypeError, ValueError):
                errors.append(f"{target_id} metrics contain non-finite or missing values")
                continue
            if checks != expected_checks or not all(expected_checks.values()) or row.get("status") != "pass":
                errors.append(f"{target_id} fixed threshold checks failed or were altered")
        reviewers = payload.get("approved_by") if isinstance(payload.get("approved_by"), list) else []
        expected_roles = {"operations_planning", "model_risk", "maritime_safety"}
        reviewer_ids = [_clean(row.get("reviewer_id")) for row in reviewers if isinstance(row, dict)]
        reviewer_roles = {row.get("role") for row in reviewers if isinstance(row, dict)}
        data_quality = payload.get("data_quality") if isinstance(payload.get("data_quality"), dict) else {}
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        production_gate_eligible = bool(
            not errors
            and source.get("evidence_class") == "authorized_site_forecast_export"
            and source.get("source_verified") is True
            and payload.get("calibration_status") == "pass"
            and payload.get("measured_outcomes") is True
            and payload.get("live_service_level_verified") is True
            and len(reviewer_ids) == 3
            and len(set(reviewer_ids)) == 3
            and reviewer_roles == expected_roles
            and all(_IDENTIFIER.fullmatch(item) for item in reviewer_ids)
            and all(item.lower() != _clean(source.get("owner")).lower() for item in reviewer_ids)
            and _IDENTIFIER.fullmatch(_clean(payload.get("change_ticket")))
            and payload.get("approved") is True
            and data_quality.get("chronological_training_calibration_test") is True
            and data_quality.get("pre_issue_feature_snapshot_verified") is True
            and data_quality.get("prequential_timing_verified") is True
            and int(data_quality.get("target_count") or 0) == len(TARGETS)
            and boundary.get("site_forecast_service_accepted") is True
            and boundary.get("forecast_advisory_only") is True
            and boundary.get("automatic_resource_commitment_allowed") is False
            and boundary.get("dispatch_allowed") is False
            and boundary.get("production_authority") is False
        )
        return {"valid": not errors, "errors": errors, "production_gate_eligible": production_gate_eligible}

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_FORECAST_UNCERTAINTY_PATH"))
        artifact: Dict[str, Any] = {
            "mode": "unconfigured",
            "configured": False,
            "verified": False,
            "artifact_id": None,
            "sha256": None,
            "blockers": ["forecast_uncertainty_artifact_not_configured"],
        }
        if raw_path:
            path = Path(raw_path).expanduser()
            artifact.update(
                mode="configured_invalid",
                configured=True,
                artifact_id=path.name,
                blockers=["forecast_uncertainty_artifact_invalid"],
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validation = self.validate_evidence(payload)
                eligible = validation["production_gate_eligible"]
                artifact.update(
                    mode="verified_site_artifact" if eligible else "configured_invalid",
                    verified=eligible,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    site_id=payload.get("site_id"),
                    target_count=len(payload.get("metrics_by_target") or []),
                    interval_80_pass_count=sum(bool(row.get("threshold_checks", {}).get("coverage_80")) for row in payload.get("metrics_by_target") or []),
                    interval_95_pass_count=sum(bool(row.get("threshold_checks", {}).get("coverage_95")) for row in payload.get("metrics_by_target") or []),
                    calibration_status=payload.get("calibration_status"),
                    approved=payload.get("approved") is True,
                    blockers=[] if eligible else list(validation["errors"]) or ["forecast_service_not_approved"],
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        accepted = artifact.get("verified") is True
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "configured_artifact": artifact,
            "contract": {
                "targets": [
                    {
                        "target_id": target_id,
                        "label": definition["label"],
                        "unit": definition["unit"],
                        "kind": definition["kind"],
                        "features": list(definition["features"]),
                    }
                    for target_id, definition in TARGETS.items()
                ],
                "fixed_thresholds": dict(THRESHOLDS),
                "minimum_rows": deepcopy(MINIMUM_ROWS),
                "chronological_training_calibration_test_required": True,
                "pre_issue_feature_snapshot_required": True,
                "interval_method": "chronological split conformal absolute residual calibration",
                "interval_levels": [0.80, 0.95],
            },
            "boundary": {
                "site_forecast_service_accepted": accepted,
                "live_service_level_verified": accepted,
                "forecast_advisory_only": True,
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "现场预测与区间校准证据已验证" if accepted else "待接入现场预测留出证据",
                "reason": (
                    "八类现场预测均完成预签发时序核验、留出评估、区间校准和三方独立复核；预测仍只提供建议。"
                    if accepted
                    else "运行态曲线或合同样例不能替代现场预签发预测、后验实测结果、时间留出评估和独立复核。"
                ),
            },
        }

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        operations_planning_approved_by: str | None = None,
        model_risk_approved_by: str | None = None,
        maritime_safety_approved_by: str | None = None,
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

        split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
        train_start, train_end = self._window(split, "training_window", errors)
        calibration_start, calibration_end = self._window(split, "calibration_window", errors)
        test_start, test_end = self._window(split, "test_window", errors)
        if train_end and calibration_start and train_end >= calibration_start:
            self._error(errors, "window_overlap", "split", "training and calibration windows must not overlap")
        if calibration_end and test_start and calibration_end >= test_start:
            self._error(errors, "window_overlap", "split", "calibration and test windows must not overlap")

        series = payload.get("series")
        if not isinstance(series, list):
            self._error(errors, "series", "series", "series must be an array")
            series = []
        if len(series) != len(TARGETS):
            self._error(errors, "target_coverage", "series", "all eight fixed forecast targets are required exactly once")
        normalized_series: List[Dict[str, Any]] = []
        seen_targets: set[str] = set()
        accepted_rows = 0
        for series_index, raw_series in enumerate(series):
            if not isinstance(raw_series, dict):
                self._error(errors, "series_type", "series", "series entry must be an object", row_index=series_index)
                continue
            target_id = _clean(raw_series.get("target_id"))
            if target_id not in TARGETS or target_id in seen_targets:
                self._error(errors, "target_id", "series.target_id", "target must be fixed and unique", target_id=target_id or None)
                continue
            seen_targets.add(target_id)
            definition = TARGETS[target_id]
            try:
                horizon = int(raw_series.get("forecast_horizon_minutes"))
                if horizon < 5 or horizon > 10080:
                    raise ValueError
            except (TypeError, ValueError):
                horizon = 0
                self._error(errors, "forecast_horizon", "series.forecast_horizon_minutes", "must be from five minutes to seven days", target_id=target_id)
            try:
                ridge_alpha = _finite(raw_series.get("ridge_alpha", 0.1))
                if ridge_alpha <= 0.0 or ridge_alpha > 100.0:
                    raise ValueError
            except (TypeError, ValueError):
                ridge_alpha = 0.1
                self._error(errors, "ridge_alpha", "series.ridge_alpha", "must be greater than zero and at most one hundred", target_id=target_id)
            if raw_series.get("features") != list(definition["features"]):
                self._error(errors, "feature_contract", "series.features", "features must equal the fixed target contract in order", target_id=target_id)
            rows = raw_series.get("rows")
            if not isinstance(rows, list):
                self._error(errors, "rows", "series.rows", "rows must be an array", target_id=target_id)
                rows = []
            if len(rows) > 100000:
                self._error(errors, "row_limit", "series.rows", "one target may contain at most one hundred thousand rows", target_id=target_id)
            normalized_rows: List[Dict[str, Any]] = []
            identifiers: set[str] = set()
            counts: Counter[str] = Counter()
            for row_index, raw_row in enumerate(rows):
                if not isinstance(raw_row, dict):
                    self._error(errors, "row_type", "series.rows", "row must be an object", target_id=target_id, row_index=row_index)
                    continue
                before = len(errors)
                forecast_id = _clean(raw_row.get("forecast_id"))
                source_reference = _clean(raw_row.get("source_reference"))
                if not _IDENTIFIER.fullmatch(forecast_id) or forecast_id in identifiers:
                    self._error(errors, "forecast_id", "forecast_id", "must be a unique stable identifier per target", target_id=target_id, row_index=row_index)
                identifiers.add(forecast_id)
                if not _IDENTIFIER.fullmatch(source_reference):
                    self._error(errors, "source_reference", "source_reference", "stable source reference is required", target_id=target_id, row_index=row_index)
                timestamps: Dict[str, datetime] = {}
                for field in ("feature_snapshot_at", "issued_at", "target_at", "observed_at"):
                    try:
                        timestamps[field] = _parse_timestamp(raw_row.get(field))
                    except (TypeError, ValueError):
                        self._error(errors, "timestamp", field, "timezone-aware timestamp is required", target_id=target_id, row_index=row_index)
                if len(timestamps) == 4:
                    if not timestamps["feature_snapshot_at"] <= timestamps["issued_at"] < timestamps["target_at"] <= timestamps["observed_at"] <= extracted_at:
                        self._error(errors, "forecast_timing", "series.rows", "requires feature_snapshot_at <= issued_at < target_at <= observed_at <= extracted_at", target_id=target_id, row_index=row_index)
                    actual_horizon = (timestamps["target_at"] - timestamps["issued_at"]).total_seconds() / 60.0
                    if abs(actual_horizon - horizon) > 1e-6:
                        self._error(errors, "forecast_horizon", "target_at", "target_at must match the declared forecast horizon", target_id=target_id, row_index=row_index)
                feature_values: Dict[str, float] = {}
                feature_payload = raw_row.get("features") if isinstance(raw_row.get("features"), dict) else {}
                if set(feature_payload) != set(definition["features"]):
                    self._error(errors, "feature_fields", "features", "feature keys must exactly match the fixed target contract", target_id=target_id, row_index=row_index)
                for feature in definition["features"]:
                    try:
                        feature_values[feature] = _finite(feature_payload.get(feature))
                    except (TypeError, ValueError):
                        self._error(errors, "feature_value", f"features.{feature}", "finite numeric value is required", target_id=target_id, row_index=row_index)
                try:
                    observed_value = _finite(raw_row.get("observed_value"))
                    lower, upper = definition["bounds"]
                    if observed_value < lower or observed_value > upper:
                        raise ValueError
                    if definition["kind"] == "binary_probability" and observed_value not in {0.0, 1.0}:
                        self._error(errors, "binary_outcome", "observed_value", "risk targets require a realized zero-or-one outcome", target_id=target_id, row_index=row_index)
                except (TypeError, ValueError):
                    observed_value = 0.0
                    self._error(errors, "observed_value", "observed_value", "observed value is outside the target bounds", target_id=target_id, row_index=row_index)
                row_split: str | None = None
                target_at = timestamps.get("target_at")
                if target_at and train_start and train_end and train_start <= target_at <= train_end:
                    row_split = "training"
                elif target_at and calibration_start and calibration_end and calibration_start <= target_at <= calibration_end:
                    row_split = "calibration"
                elif target_at and test_start and test_end and test_start <= target_at <= test_end:
                    row_split = "test"
                elif target_at:
                    self._error(errors, "outside_windows", "target_at", "target must belong to exactly one declared window", target_id=target_id, row_index=row_index)
                if len(errors) == before and row_split:
                    normalized_rows.append({
                        "forecast_id": forecast_id,
                        "feature_snapshot_at": _iso(timestamps["feature_snapshot_at"]),
                        "issued_at": _iso(timestamps["issued_at"]),
                        "target_at": _iso(timestamps["target_at"]),
                        "observed_at": _iso(timestamps["observed_at"]),
                        "features": feature_values,
                        "observed_value": observed_value,
                        "source_reference": source_reference,
                        "split": row_split,
                    })
                    counts[row_split] += 1
            minimums = MINIMUM_ROWS.get(evidence_class, MINIMUM_ROWS["contract_test_only"])
            for split_name, minimum in minimums.items():
                if counts[split_name] < minimum:
                    self._error(errors, f"{split_name}_rows", "series.rows", f"at least {minimum} valid {split_name} rows are required", target_id=target_id)
            accepted_rows += len(normalized_rows)
            normalized_series.append({
                "target_id": target_id,
                "forecast_horizon_minutes": horizon,
                "ridge_alpha": ridge_alpha,
                "features": list(definition["features"]),
                "rows": sorted(normalized_rows, key=lambda row: row["target_at"]),
            })
        if seen_targets != set(TARGETS):
            self._error(errors, "target_coverage", "series", "target identifiers must exactly match the fixed eight-target contract")

        normalized_source = {
            "source_system": _clean(source.get("source_system")),
            "owner": _clean(source.get("owner")),
            "license": _clean(source.get("license")),
            "timezone": timezone_name,
            "extracted_at": _iso(extracted_at),
            "evidence_class": evidence_class,
            "source_verified": bool(source_verified and evidence_class == "authorized_site_forecast_export"),
        }
        normalized_input = {
            "schema_version": DATASET_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "source": {key: value for key, value in normalized_source.items() if key != "source_verified"},
            "split": {
                "training_window": {"start_at": _iso(train_start) if train_start else None, "end_at": _iso(train_end) if train_end else None},
                "calibration_window": {"start_at": _iso(calibration_start) if calibration_start else None, "end_at": _iso(calibration_end) if calibration_end else None},
                "test_window": {"start_at": _iso(test_start) if test_start else None, "end_at": _iso(test_end) if test_end else None},
            },
            "series": sorted(normalized_series, key=lambda row: row["target_id"]),
        }
        dataset_sha256 = _digest(normalized_input)
        if errors or accepted_rows != sum(len(row.get("rows") or []) for row in series if isinstance(row, dict)):
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "dataset_sha256": None,
                "evidence": None,
                "boundary": {
                    "site_forecast_service_accepted": False,
                    "live_service_level_verified": False,
                    "forecast_advisory_only": True,
                    "dispatch_allowed": False,
                    "production_authority": False,
                    "claim": "forecast_dataset_rejected",
                },
            }

        metrics_by_target: List[Dict[str, Any]] = []
        model_evidence: List[Dict[str, Any]] = []
        receipts: List[Dict[str, Any]] = []
        row_counts: Dict[str, Dict[str, int]] = {}
        try:
            for target_series in normalized_input["series"]:
                target_id = target_series["target_id"]
                definition = TARGETS[target_id]
                features = definition["features"]
                parts = {
                    name: [row for row in target_series["rows"] if row["split"] == name]
                    for name in ("training", "calibration", "test")
                }
                matrices = {
                    name: np.asarray([[row["features"][field] for field in features] for row in rows], dtype=np.float64)
                    for name, rows in parts.items()
                }
                observed = {
                    name: np.asarray([row["observed_value"] for row in rows], dtype=np.float64)
                    for name, rows in parts.items()
                }
                model = self._fit_ridge(matrices["training"], observed["training"], target_series["ridge_alpha"])
                calibration_prediction = self._predict(model, matrices["calibration"], definition["bounds"])
                test_prediction = self._predict(model, matrices["test"], definition["bounds"])
                residuals = np.abs(observed["calibration"] - calibration_prediction)
                q80 = _conformal_quantile(residuals, 0.80)
                q95 = _conformal_quantile(residuals, 0.95)
                lower_bound, upper_bound = definition["bounds"]
                lower80, upper80 = np.clip(test_prediction - q80, lower_bound, upper_bound), np.clip(test_prediction + q80, lower_bound, upper_bound)
                lower95, upper95 = np.clip(test_prediction - q95, lower_bound, upper_bound), np.clip(test_prediction + q95, lower_bound, upper_bound)
                absolute_error = np.abs(observed["test"] - test_prediction)
                scale = max(
                    float(np.ptp(observed["training"])),
                    abs(float(np.mean(observed["training"]))) * 0.10,
                    1e-9,
                )
                train_means = matrices["training"].mean(axis=0)
                train_scales = np.maximum(matrices["training"].std(axis=0), 1e-9)
                shifts = np.abs((matrices["test"].mean(axis=0) - train_means) / train_scales)
                values: Dict[str, Any] = {
                    "mae": float(absolute_error.mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(observed["test"] - test_prediction)))),
                    "normalized_mae": float(absolute_error.mean() / scale),
                    "coverage_80": float(np.mean((observed["test"] >= lower80) & (observed["test"] <= upper80))),
                    "coverage_95": float(np.mean((observed["test"] >= lower95) & (observed["test"] <= upper95))),
                    "average_interval_width_80": float(np.mean(upper80 - lower80)),
                    "average_interval_width_95": float(np.mean(upper95 - lower95)),
                    "normalized_interval_width_80": float(np.mean(upper80 - lower80) / scale),
                    "normalized_interval_width_95": float(np.mean(upper95 - lower95) / scale),
                    "calibration_residual_quantile_80": q80,
                    "calibration_residual_quantile_95": q95,
                    "max_standardized_feature_shift": float(np.max(shifts)),
                }
                if definition["kind"] == "binary_probability":
                    values["brier_score"] = float(np.mean(np.square(test_prediction - observed["test"])))
                    values["expected_calibration_error"] = self._ece(observed["test"], test_prediction)
                checks = {
                    "normalized_mae": values["normalized_mae"] <= THRESHOLDS["normalized_mae_max"],
                    "coverage_80": values["coverage_80"] >= THRESHOLDS["coverage_80_min"],
                    "coverage_95": values["coverage_95"] >= THRESHOLDS["coverage_95_min"],
                    "normalized_width_80": values["normalized_interval_width_80"] <= THRESHOLDS["normalized_width_80_max"],
                    "normalized_width_95": values["normalized_interval_width_95"] <= THRESHOLDS["normalized_width_95_max"],
                    "feature_shift": values["max_standardized_feature_shift"] <= THRESHOLDS["max_standardized_feature_shift_max"],
                }
                if definition["kind"] == "binary_probability":
                    checks.update(
                        brier_score=values["brier_score"] <= THRESHOLDS["brier_score_max"],
                        expected_calibration_error=values["expected_calibration_error"] <= THRESHOLDS["expected_calibration_error_max"],
                    )
                target_status = "pass" if all(checks.values()) else "fail"
                metrics_by_target.append({
                    "target_id": target_id,
                    "label": definition["label"],
                    "unit": definition["unit"],
                    "kind": definition["kind"],
                    "forecast_horizon_minutes": target_series["forecast_horizon_minutes"],
                    "metrics": values,
                    "threshold_checks": checks,
                    "status": target_status,
                })
                model_evidence.append({
                    "target_id": target_id,
                    "model_version": f"ridge-{target_id}-{dataset_sha256[:12]}",
                    "algorithm": "ridge_regression",
                    "features": list(features),
                    "ridge_alpha": target_series["ridge_alpha"],
                    "intercept": float(model["intercept"]),
                    "coefficients": {name: float(model["coefficients"][index]) for index, name in enumerate(features)},
                    "training_feature_mean": {name: float(model["training_feature_mean"][index]) for index, name in enumerate(features)},
                    "training_feature_std": {name: float(model["training_feature_std"][index]) for index, name in enumerate(features)},
                    "calibration_method": "chronological_split_conformal_absolute_residual",
                })
                row_counts[target_id] = {name: len(parts[name]) for name in parts}
                for index, row in enumerate(parts["test"]):
                    receipts.append({
                        "target_id": target_id,
                        "forecast_id": row["forecast_id"],
                        "issued_at": row["issued_at"],
                        "target_at": row["target_at"],
                        "observed_at": row["observed_at"],
                        "prediction": float(test_prediction[index]),
                        "interval_80": [float(lower80[index]), float(upper80[index])],
                        "interval_95": [float(lower95[index]), float(upper95[index])],
                        "observed_value": float(observed["test"][index]),
                        "source_reference": row["source_reference"],
                    })
        except ValueError as exc:
            self._error(errors, "model_fit", "series.rows", str(exc))
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "dataset_sha256": None,
                "evidence": None,
                "boundary": {
                    "site_forecast_service_accepted": False,
                    "live_service_level_verified": False,
                    "forecast_advisory_only": True,
                    "dispatch_allowed": False,
                    "production_authority": False,
                    "claim": "forecast_model_fit_rejected",
                },
            }

        calibration_status = "pass" if all(row["status"] == "pass" for row in metrics_by_target) else "fail"
        measured_outcomes = bool(
            evidence_class == "authorized_site_forecast_export"
            and source_verified
            and calibration_status == "pass"
        )
        reviewer_rows = [
            {"role": "operations_planning", "reviewer_id": _clean(operations_planning_approved_by)},
            {"role": "model_risk", "reviewer_id": _clean(model_risk_approved_by)},
            {"role": "maritime_safety", "reviewer_id": _clean(maritime_safety_approved_by)},
        ]
        reviewer_ids = [row["reviewer_id"] for row in reviewer_rows]
        owner = _clean(source.get("owner"))
        ticket = _clean(change_ticket)
        approval_metadata_valid = bool(
            all(_IDENTIFIER.fullmatch(item) for item in reviewer_ids)
            and len(set(reviewer_ids)) == 3
            and all(item.lower() != owner.lower() for item in reviewer_ids)
            and _IDENTIFIER.fullmatch(ticket)
        )
        approved = bool(measured_outcomes and approval_metadata_valid)
        if evidence_class == "contract_test_only":
            warnings.append({"code": "contract_only", "message": "合同样例只验证计算与时序合同，不构成现场预测服务水平。"})
        elif not source_verified:
            warnings.append({"code": "source_attestation_missing", "message": "授权现场导出未完成来源证明，只能形成未验证候选。"})
        if calibration_status != "pass":
            warnings.append({"code": "holdout_threshold_failed", "message": "至少一个预测目标未通过留出误差、覆盖率、区间宽度或漂移门禁。"})
        if measured_outcomes and any(reviewer_ids + [ticket]) and not approval_metadata_valid:
            warnings.append({"code": "independent_approval_invalid", "message": "运营计划、模型风险和海事安全复核人必须互异、独立于数据责任主体，并绑定稳定变更单。"})
        warnings.append({
            "code": "time_series_coverage_boundary",
            "message": "按时间留出的经验覆盖率不是未来分布不变的保证；数据漂移后必须重新校准并重新审批。",
        })
        target_contract = [
            {
                "target_id": target_id,
                "label": definition["label"],
                "unit": definition["unit"],
                "kind": definition["kind"],
                "features": list(definition["features"]),
            }
            for target_id, definition in TARGETS.items()
        ]
        boundary = {
            "site_forecast_service_accepted": approved,
            "live_service_level_verified": measured_outcomes,
            "forecast_advisory_only": True,
            "automatic_resource_commitment_allowed": False,
            "dispatch_allowed": False,
            "production_authority": False,
            "claim": "accepted_site_forecast_evidence" if approved else "contract_or_unapproved_forecast_evidence",
            "reason": (
                "八类现场预测通过时间留出误差和区间校准门禁，并完成三方独立复核；结果仍只作为调度建议输入。"
                if approved
                else "合同样例、未签发预测、未观测结果或缺少独立审批都不能形成现场预测服务水平。"
            ),
        }
        evidence: Dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "dataset_sha256": dataset_sha256,
            "source": normalized_source,
            "target_contract": target_contract,
            "target_contract_digest": _digest(target_contract),
            "split": normalized_input["split"],
            "row_counts": row_counts,
            "models": model_evidence,
            "model_bundle_digest": _digest(model_evidence),
            "metrics_by_target": metrics_by_target,
            "test_receipts": receipts,
            "test_receipts_digest": _digest(receipts),
            "thresholds": dict(THRESHOLDS),
            "calibration_status": calibration_status,
            "data_quality": {
                "chronological_training_calibration_test": True,
                "pre_issue_feature_snapshot_verified": True,
                "prequential_timing_verified": True,
                "target_count": len(metrics_by_target),
                "accepted_rows": accepted_rows,
                "unique_forecast_ids_per_target": True,
            },
            "measured_outcomes": measured_outcomes,
            "live_service_level_verified": measured_outcomes,
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
