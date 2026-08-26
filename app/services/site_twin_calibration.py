from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np


DATASET_SCHEMA = "site_twin_calibration_dataset.v1"
EVIDENCE_SCHEMA = "site_twin_calibration_evidence.v2"
FEATURES = ("throughput_teu", "vessel_arrivals", "ambient_c", "tide_m")
TARGET = "observed_power_kw"
EVIDENCE_CLASSES = ("authorized_site_export", "contract_test_only")
THRESHOLDS = {
    "normalized_mae": 0.12,
    "absolute_bias_ratio": 0.05,
    "one_minus_r2": 0.25,
    "max_standardized_feature_shift": 1.5,
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


class SiteTwinCalibrationService:
    """Fit a bounded port-power twin from chronological site observations.

    The public API never turns a browser-submitted bundle into approved site
    evidence. An authorized offline export must be processed with an explicit
    source attestation and then independently approved before the existing
    production-readiness gate can consume the resulting versioned artifact.
    """

    def configured_evidence(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_TWIN_CALIBRATION_PATH"))
        if not raw_path:
            raise FileNotFoundError("PORT_DT_TWIN_CALIBRATION_PATH is not configured")
        path = Path(raw_path).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        from app.services.twin_schema.service import TwinSchemaService

        validation = TwinSchemaService.validate_calibration(payload)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        return {
            **payload,
            "artifact_id": path.name,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "validation": validation,
        }

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_TWIN_CALIBRATION_PATH"))
        configured: Dict[str, Any] = {
            "mode": "unconfigured",
            "configured": False,
            "verified": False,
            "artifact_id": None,
            "sha256": None,
            "blockers": ["calibration_artifact_not_configured"],
        }
        if raw_path:
            path = Path(raw_path).expanduser()
            configured.update(
                mode="configured_invalid",
                configured=True,
                artifact_id=path.name,
                blockers=["calibration_artifact_invalid"],
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                from app.services.twin_schema.service import TwinSchemaService

                validation = TwinSchemaService.validate_calibration(payload)
                configured.update(
                    mode="verified_site_artifact" if validation["valid"] else "configured_invalid",
                    verified=bool(validation["valid"]),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    site_id=payload.get("site_id"),
                    validation_status=payload.get("validation_status"),
                    measured_outcomes=payload.get("measured_outcomes") is True,
                    approved=payload.get("approved") is True,
                    blockers=[] if validation["valid"] else list(validation["errors"]),
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        site_calibrated = bool(
            configured.get("verified")
            and configured.get("measured_outcomes")
            and configured.get("approved")
            and configured.get("validation_status") == "pass"
        )
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "configured_artifact": configured,
            "contract": {
                "target": TARGET,
                "features": list(FEATURES),
                "minimum_training_rows": 48,
                "minimum_validation_rows": 24,
                "chronological_holdout_required": True,
                "required_row_fields": [
                    "timestamp",
                    "asset_group",
                    TARGET,
                    *FEATURES,
                    "source_reference",
                ],
                "fixed_thresholds": dict(THRESHOLDS),
            },
            "boundary": {
                "site_calibrated": site_calibrated,
                "measured_outcomes_verified": bool(configured.get("measured_outcomes")),
                "independent_approval_verified": bool(configured.get("approved")),
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "现场标定证据已验证" if site_calibrated else "待接入港口标定样本",
                "reason": (
                    "现场测量、独立验证与审批证据已通过；生产执行仍由独立门禁决定。"
                    if site_calibrated
                    else "合同样例或仅配置文件不构成现场标定；必须使用授权测量样本、独立验证窗和审批证据。"
                ),
            },
        }

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
    def _window(
        split: Dict[str, Any],
        name: str,
        errors: List[Dict[str, Any]],
    ) -> tuple[datetime | None, datetime | None]:
        payload = split.get(name) if isinstance(split.get(name), dict) else {}
        try:
            start = _parse_timestamp(payload.get("start_at"))
            end = _parse_timestamp(payload.get("end_at"))
            if end <= start:
                raise ValueError("end must be after start")
            return start, end
        except (TypeError, ValueError):
            SiteTwinCalibrationService._error(
                errors,
                "calibration_window",
                f"split.{name}",
                "window requires timezone-aware start_at before end_at",
            )
            return None, None

    @staticmethod
    def _fit_ridge(matrix: np.ndarray, target: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        if np.any(scales < 1e-9):
            raise ValueError("training features must vary")
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
        return intercept, coefficients.astype(np.float64)

    @staticmethod
    def _metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        residual = predicted - observed
        mean_observed = max(1e-9, float(np.mean(np.abs(observed))))
        ss_res = float(np.sum(np.square(residual)))
        ss_tot = float(np.sum(np.square(observed - np.mean(observed))))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return {
            "mae_kw": float(np.mean(np.abs(residual))),
            "rmse_kw": float(np.sqrt(np.mean(np.square(residual)))),
            "normalized_mae": float(np.mean(np.abs(residual)) / mean_observed),
            "absolute_bias_ratio": float(abs(np.mean(residual)) / mean_observed),
            "r2": float(r2),
            "one_minus_r2": float(max(0.0, 1.0 - r2)),
        }

    @staticmethod
    def _bootstrap_interval(
        observed: np.ndarray,
        predicted: np.ndarray,
        *,
        seed: int,
        samples: int = 500,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(samples):
            indices = rng.integers(0, len(observed), size=len(observed))
            values.append(SiteTwinCalibrationService._metrics(observed[indices], predicted[indices])["normalized_mae"])
        return {
            "metric": "normalized_mae",
            "method": "paired_validation_row_bootstrap",
            "samples": samples,
            "confidence": 0.95,
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        approved_by: str | None = None,
        change_ticket: str | None = None,
    ) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        if not isinstance(payload, dict):
            payload = {}
            self._error(errors, "dataset_type", "$", "payload must be a JSON object")
        if _clean(payload.get("schema_version")) != DATASET_SCHEMA:
            self._error(errors, "schema_version", "schema_version", f"must equal {DATASET_SCHEMA}")
        site_id = _clean(payload.get("site_id"))
        dataset_id = _clean(payload.get("dataset_id"))
        for field, value in (("site_id", site_id), ("dataset_id", dataset_id)):
            if not _IDENTIFIER.fullmatch(value):
                self._error(errors, "identifier", field, "must be a stable 3-128 character identifier")

        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        source_system = _clean(source.get("source_system"))
        owner = _clean(source.get("owner"))
        license_name = _clean(source.get("license"))
        evidence_class = _clean(source.get("evidence_class"))
        timezone_name = _clean(source.get("timezone"))
        for field, value in (
            ("source_system", source_system),
            ("owner", owner),
            ("license", license_name),
            ("timezone", timezone_name),
            ("evidence_class", evidence_class),
        ):
            if not value:
                self._error(errors, "source_metadata", f"source.{field}", "field is required")
        if owner.lower() in _PLACEHOLDERS or license_name.lower() in _PLACEHOLDERS:
            self._error(errors, "source_placeholder", "source.owner/license", "placeholder governance metadata is rejected")
        if evidence_class and evidence_class not in EVIDENCE_CLASSES:
            self._error(errors, "evidence_class", "source.evidence_class", f"must be one of {', '.join(EVIDENCE_CLASSES)}")
        if timezone_name:
            try:
                ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError:
                self._error(errors, "timezone", "source.timezone", "must be a valid IANA timezone")
        extracted_at: datetime | None = None
        try:
            extracted_at = _parse_timestamp(source.get("extracted_at"))
        except (TypeError, ValueError):
            self._error(errors, "extracted_at", "source.extracted_at", "must be a timezone-aware timestamp")

        split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
        train_start, train_end = self._window(split, "training_window", errors)
        validation_start, validation_end = self._window(split, "validation_window", errors)
        if train_end and validation_start and train_end >= validation_start:
            self._error(errors, "window_overlap", "split", "training and validation windows must not overlap")

        model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
        if model.get("target") != TARGET:
            self._error(errors, "model_target", "model.target", f"must equal {TARGET}")
        if model.get("features") != list(FEATURES):
            self._error(errors, "model_features", "model.features", f"must equal {list(FEATURES)} in declared order")
        try:
            ridge_alpha = _finite(model.get("ridge_alpha", 0.1))
            if ridge_alpha <= 0.0 or ridge_alpha > 100.0:
                raise ValueError("ridge alpha is outside bounds")
        except (TypeError, ValueError):
            ridge_alpha = 0.1
            self._error(errors, "ridge_alpha", "model.ridge_alpha", "must be greater than zero and at most one hundred")

        rows = payload.get("rows")
        if not isinstance(rows, list):
            self._error(errors, "rows", "rows", "rows must be an array")
            rows = []
        elif len(rows) > 100000:
            self._error(errors, "row_limit", "rows", "a calibration bundle may contain at most one hundred thousand rows")
        normalized_rows: List[Dict[str, Any]] = []
        unique_keys: set[tuple[str, str]] = set()
        split_counts: Counter[str] = Counter()
        asset_split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        for index, raw in enumerate(rows):
            if not isinstance(raw, dict):
                self._error(errors, "row_type", "rows", "row must be an object", row_index=index)
                continue
            before = len(errors)
            asset_group = _clean(raw.get("asset_group"))
            source_reference = _clean(raw.get("source_reference"))
            if not _IDENTIFIER.fullmatch(asset_group):
                self._error(errors, "asset_group", "asset_group", "stable asset group identifier is required", row_index=index)
            if not _IDENTIFIER.fullmatch(source_reference):
                self._error(errors, "source_reference", "source_reference", "stable source reference is required", row_index=index)
            timestamp: datetime | None = None
            try:
                timestamp = _parse_timestamp(raw.get("timestamp"))
                if extracted_at and timestamp > extracted_at:
                    self._error(errors, "future_observation", "timestamp", "observation cannot be after source extraction", row_index=index)
            except (TypeError, ValueError):
                self._error(errors, "timestamp", "timestamp", "timezone-aware timestamp is required", row_index=index)
            numeric: Dict[str, float] = {}
            for field in (TARGET, *FEATURES):
                try:
                    numeric[field] = _finite(raw.get(field))
                except (TypeError, ValueError):
                    self._error(errors, "numeric", field, "finite numeric value is required", row_index=index)
            for field in (TARGET, "throughput_teu", "vessel_arrivals"):
                if field in numeric and numeric[field] < 0.0:
                    self._error(errors, "nonnegative", field, "value must be nonnegative", row_index=index)
            if "ambient_c" in numeric and not -50.0 <= numeric["ambient_c"] <= 70.0:
                self._error(errors, "ambient_range", "ambient_c", "ambient temperature is outside physical bounds", row_index=index)
            if "tide_m" in numeric and not -20.0 <= numeric["tide_m"] <= 20.0:
                self._error(errors, "tide_range", "tide_m", "tide is outside physical bounds", row_index=index)
            row_split: str | None = None
            if timestamp and train_start and train_end and train_start <= timestamp <= train_end:
                row_split = "training"
            elif timestamp and validation_start and validation_end and validation_start <= timestamp <= validation_end:
                row_split = "validation"
            elif timestamp:
                self._error(errors, "outside_windows", "timestamp", "row must belong to exactly one declared window", row_index=index)
            if timestamp:
                key = (asset_group, _iso_utc(timestamp))
                if key in unique_keys:
                    self._error(errors, "duplicate_observation", "asset_group/timestamp", "observation key must be unique", row_index=index)
                unique_keys.add(key)
            if len(errors) == before and timestamp and row_split:
                normalized = {
                    "timestamp": _iso_utc(timestamp),
                    "asset_group": asset_group,
                    TARGET: numeric[TARGET],
                    **{name: numeric[name] for name in FEATURES},
                    "source_reference": source_reference,
                    "split": row_split,
                }
                normalized_rows.append(normalized)
                split_counts[row_split] += 1
                asset_split_counts[asset_group][row_split] += 1

        if split_counts["training"] < 48:
            self._error(errors, "training_rows", "rows", "at least forty-eight valid training rows are required")
        if split_counts["validation"] < 24:
            self._error(errors, "validation_rows", "rows", "at least twenty-four valid validation rows are required")
        for asset_group, counts in asset_split_counts.items():
            if counts["training"] < 12 or counts["validation"] < 6:
                self._error(errors, "asset_coverage", "rows", f"asset group {asset_group} requires at least twelve training and six validation rows")

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
            "dataset_id": dataset_id,
            "source": normalized_source,
            "split": {
                "training_window": {
                    "start_at": _iso_utc(train_start) if train_start else None,
                    "end_at": _iso_utc(train_end) if train_end else None,
                },
                "validation_window": {
                    "start_at": _iso_utc(validation_start) if validation_start else None,
                    "end_at": _iso_utc(validation_end) if validation_end else None,
                },
            },
            "model": {"target": TARGET, "features": list(FEATURES), "ridge_alpha": ridge_alpha},
            "rows": sorted(normalized_rows, key=lambda row: (row["timestamp"], row["asset_group"])),
        }
        dataset_sha256 = _canonical_digest(normalized_input)
        if errors or len(normalized_rows) != len(rows):
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "received_rows": len(rows),
                "accepted_rows": 0,
                "dataset_sha256": None,
                "evidence": None,
                "boundary": {
                    "site_calibrated": False,
                    "measured_outcomes_verified": False,
                    "dispatch_allowed": False,
                    "production_authority": False,
                    "claim": "calibration_dataset_rejected",
                },
            }

        training = [row for row in normalized_rows if row["split"] == "training"]
        validation = [row for row in normalized_rows if row["split"] == "validation"]
        train_x = np.asarray([[row[name] for name in FEATURES] for row in training], dtype=np.float64)
        train_y = np.asarray([row[TARGET] for row in training], dtype=np.float64)
        validation_x = np.asarray([[row[name] for name in FEATURES] for row in validation], dtype=np.float64)
        validation_y = np.asarray([row[TARGET] for row in validation], dtype=np.float64)
        validation_predicted = np.zeros(len(validation), dtype=np.float64)
        asset_models: Dict[str, Any] = {}
        try:
            for asset_group in sorted(asset_split_counts):
                train_indices = np.asarray([
                    index for index, row in enumerate(training)
                    if row["asset_group"] == asset_group
                ])
                validation_indices = np.asarray([
                    index for index, row in enumerate(validation)
                    if row["asset_group"] == asset_group
                ])
                group_x = train_x[train_indices]
                group_y = train_y[train_indices]
                intercept, coefficients = self._fit_ridge(group_x, group_y, ridge_alpha)
                group_means = group_x.mean(axis=0)
                group_scales = np.maximum(group_x.std(axis=0), 1e-9)
                validation_predicted[validation_indices] = np.maximum(
                    0.0,
                    intercept + validation_x[validation_indices] @ coefficients,
                )
                asset_models[asset_group] = {
                    "intercept_kw": float(intercept),
                    "coefficients": {
                        name: float(coefficients[index])
                        for index, name in enumerate(FEATURES)
                    },
                    "training_feature_mean": {
                        name: float(group_means[index])
                        for index, name in enumerate(FEATURES)
                    },
                    "training_feature_std": {
                        name: float(group_scales[index])
                        for index, name in enumerate(FEATURES)
                    },
                    "training_rows": int(len(train_indices)),
                    "validation_rows": int(len(validation_indices)),
                }
        except ValueError as exc:
            self._error(errors, "feature_variation", "rows", str(exc))
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "received_rows": len(rows),
                "accepted_rows": 0,
                "dataset_sha256": None,
                "evidence": None,
                "boundary": {
                    "site_calibrated": False,
                    "measured_outcomes_verified": False,
                    "dispatch_allowed": False,
                    "production_authority": False,
                    "claim": "calibration_dataset_rejected",
                },
            }
        metrics = self._metrics(validation_y, validation_predicted)
        train_means = train_x.mean(axis=0)
        train_scales = np.maximum(train_x.std(axis=0), 1e-9)
        shifts = np.abs((validation_x.mean(axis=0) - train_means) / train_scales)
        metrics["max_standardized_feature_shift"] = float(np.max(shifts))
        threshold_checks = {name: metrics[name] <= limit for name, limit in THRESHOLDS.items()}
        validation_status = "pass" if all(threshold_checks.values()) else "fail"

        by_asset = []
        for asset_group in sorted({row["asset_group"] for row in validation}):
            indices = np.asarray([index for index, row in enumerate(validation) if row["asset_group"] == asset_group])
            by_asset.append({
                "asset_group": asset_group,
                "rows": int(len(indices)),
                **self._metrics(validation_y[indices], validation_predicted[indices]),
            })
        q_low, q_high = np.quantile(train_x[:, 0], [1 / 3, 2 / 3])
        regimes = []
        for name, mask in (
            ("low_throughput", validation_x[:, 0] <= q_low),
            ("medium_throughput", (validation_x[:, 0] > q_low) & (validation_x[:, 0] <= q_high)),
            ("high_throughput", validation_x[:, 0] > q_high),
        ):
            if np.any(mask):
                regimes.append({"regime": name, "rows": int(mask.sum()), **self._metrics(validation_y[mask], validation_predicted[mask])})

        measured_outcomes = bool(
            evidence_class == "authorized_site_export" and source_verified and validation_status == "pass"
        )
        reviewer = _clean(approved_by)
        ticket = _clean(change_ticket)
        approval_metadata_valid = bool(
            reviewer
            and ticket
            and _IDENTIFIER.fullmatch(reviewer)
            and _IDENTIFIER.fullmatch(ticket)
            and reviewer.lower() != owner.lower()
        )
        approved = bool(measured_outcomes and approval_metadata_valid)
        if evidence_class == "contract_test_only":
            warnings.append({
                "code": "contract_only",
                "message": "标定样例仅验证计算合同，不构成现场测量、现场精度或生产许可。",
            })
        elif not source_verified:
            warnings.append({
                "code": "source_attestation_missing",
                "message": "授权现场导出尚未完成来源证明，因此结果只能作为未验证候选。",
            })
        if measured_outcomes and (reviewer or ticket) and not approval_metadata_valid:
            warnings.append({
                "code": "independent_approval_invalid",
                "message": "审批人和变更单必须是稳定标识，且审批人不能与数据责任主体相同。",
            })
        if validation_status != "pass":
            warnings.append({
                "code": "validation_threshold_failed",
                "message": "独立验证窗未通过预声明阈值，不得形成现场标定候选。",
            })

        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": site_id,
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_sha256,
            "model_version": f"site-twin-ridge-{dataset_sha256[:12]}",
            "source": normalized_source,
            "parameters": {
                "algorithm": "ridge_regression",
                "scope": "separate_response_model_per_asset_group",
                "target": TARGET,
                "ridge_alpha": ridge_alpha,
                "asset_group_models": asset_models,
            },
            "training_window": normalized_input["split"]["training_window"],
            "validation_window": normalized_input["split"]["validation_window"],
            "training_rows": len(training),
            "validation_rows": len(validation),
            "metrics": {name: float(value) for name, value in metrics.items()},
            "thresholds": dict(THRESHOLDS),
            "threshold_checks": threshold_checks,
            "validation_status": validation_status,
            "uncertainty": self._bootstrap_interval(
                validation_y,
                validation_predicted,
                seed=int(dataset_sha256[:16], 16),
            ),
            "error_decomposition": {
                "by_asset_group": by_asset,
                "by_operating_regime": regimes,
            },
            "drift_diagnostic": {
                "standardized_mean_shift": {name: float(shifts[index]) for index, name in enumerate(FEATURES)},
                "maximum": float(np.max(shifts)),
                "threshold": THRESHOLDS["max_standardized_feature_shift"],
                "passed": threshold_checks["max_standardized_feature_shift"],
            },
            "data_quality": {
                "received_rows": len(rows),
                "accepted_rows": len(normalized_rows),
                "training_rows": len(training),
                "validation_rows": len(validation),
                "unique_asset_timestamp_keys": len(unique_keys) == len(rows),
                "asset_group_counts": {
                    asset: dict(sorted(counts.items()))
                    for asset, counts in sorted(asset_split_counts.items())
                },
                "chronological_holdout": True,
            },
            "provenance": {
                "type": "site_measurement" if measured_outcomes else "contract_test_or_unverified_export",
                "source_system": source_system,
                "source_reference_digest": _canonical_digest(sorted(row["source_reference"] for row in normalized_rows)),
                "algorithm_implementation": "app.services.site_twin_calibration.SiteTwinCalibrationService",
                "change_ticket": ticket or None,
            },
            "measured_outcomes": measured_outcomes,
            "approved": approved,
            "approved_by": reviewer or None,
            "approval_required": True,
            "boundary": {
                "site_calibrated": approved,
                "measured_outcomes_verified": measured_outcomes,
                "independent_validation_passed": validation_status == "pass",
                "independent_approval_verified": approved,
                "dispatch_allowed": False,
                "production_authority": False,
                "claim": (
                    "approved_site_calibration_evidence"
                    if approved
                    else "contract_validation_or_unapproved_calibration_candidate"
                ),
            },
        }
        evidence["evidence_digest"] = _canonical_digest(evidence)
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "received_rows": len(rows),
            "accepted_rows": len(normalized_rows),
            "dataset_sha256": dataset_sha256,
            "evidence": evidence,
            "boundary": dict(evidence["boundary"]),
        }
