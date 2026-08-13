"""Data-driven short-term load forecasting.

The model is fitted per asset from telemetry on every request.  It uses a
regularized autoregression and empirical residual quantiles.  There are no
device defaults, synthetic day shapes, or random perturbations: insufficient
history produces an empty forecast for that asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _clean_history(rows: Sequence[Any]) -> Tuple[List[datetime], np.ndarray]:
    cleaned: List[Tuple[datetime, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        try:
            value = float(row.get("kW", row.get("value", row.get("v"))))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        timestamp = _parse_timestamp(row.get("ts", row.get("timestamp")))
        if timestamp is None:
            timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
        cleaned.append((timestamp, max(0.0, value)))
    cleaned.sort(key=lambda item: item[0])
    return [item[0] for item in cleaned], np.asarray([item[1] for item in cleaned], dtype=np.float64)


def _sampling_minutes(timestamps: Sequence[datetime], requested_step_min: int) -> int:
    gaps = [
        (timestamps[index] - timestamps[index - 1]).total_seconds() / 60.0
        for index in range(1, len(timestamps))
        if timestamps[index] > timestamps[index - 1]
    ]
    observed = int(round(float(np.median(gaps)))) if gaps else max(1, int(requested_step_min))
    return max(max(1, int(requested_step_min)), max(1, observed))


def _feature(window: Sequence[float], step_index: int, total_steps: int) -> np.ndarray:
    trend = float(step_index) / max(1.0, float(total_steps))
    return np.asarray([1.0, *reversed(window), trend], dtype=np.float64)


def _fit_autoregression(values: np.ndarray) -> Optional[Tuple[np.ndarray, int, float, float]]:
    if len(values) < 18:
        return None
    lag = min(12, max(3, len(values) // 6))
    if len(values) < 2 * lag + 6:
        return None

    features: List[np.ndarray] = []
    targets: List[float] = []
    total = len(values) - lag
    for index in range(lag, len(values)):
        features.append(_feature(values[index - lag : index], index - lag, total))
        targets.append(float(values[index]))
    matrix = np.vstack(features)
    target = np.asarray(targets, dtype=np.float64)

    # Ridge regularization stabilizes collinear lag features. The intercept is
    # intentionally not regularized.
    penalty = max(1e-6, float(np.var(values)) * 1e-4)
    regularizer = np.eye(matrix.shape[1], dtype=np.float64) * penalty
    regularizer[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(matrix.T @ matrix + regularizer, matrix.T @ target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]

    residuals = target - matrix @ coefficients
    if len(residuals) < 8:
        return coefficients, lag, 0.0, 0.0
    return (
        coefficients,
        lag,
        float(np.quantile(residuals, 0.10)),
        float(np.quantile(residuals, 0.90)),
    )


def _explicit_multiplier(drivers: Optional[Dict[str, Any]], index: int) -> float:
    if not isinstance(drivers, dict):
        return 1.0
    raw = drivers.get("load_multiplier", 1.0)
    if isinstance(raw, list):
        raw = raw[index] if index < len(raw) else (raw[-1] if raw else 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, value) if np.isfinite(value) else 1.0


@dataclass
class ForecastService:
    telemetry: Any
    factors: Optional[Any] = None
    schedule: Optional[Any] = None

    def _stability_cap(self, asset_id: str, values: np.ndarray) -> Tuple[float, str]:
        """Return a traceable physical/data envelope for recursive inference.

        A recursively evaluated autoregression can be numerically valid while
        leaving the operating range it was fitted on. Prefer the registered
        engineering asset rating; otherwise use a robust envelope derived only
        from the observed history. This is an inference guard, not a substitute
        prediction or a site-calibration claim.
        """
        rated_kw: Optional[float] = None
        try:
            for asset in self.telemetry.list_assets() or []:
                if str(asset.get("id")) != str(asset_id):
                    continue
                candidate = float(asset.get("rated_kw"))
                if np.isfinite(candidate) and candidate > 0:
                    rated_kw = candidate
                break
        except Exception:
            rated_kw = None
        observed_max = float(np.max(values))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q99 = float(np.quantile(values, 0.99))
        robust_cap = max(observed_max * 1.15, q99 + max(6.0 * mad, 0.15 * max(q99, 1.0)))
        if rated_kw is not None:
            # Never clip a source point merely because an engineering registry
            # is stale; surface that mismatch through data-quality monitoring.
            return max(rated_kw, observed_max * 1.01), "registered_asset_rating"
        return robust_cap, "robust_observed_history_envelope"

    def forecast_load(
        self,
        asset_ids: List[str],
        horizon_min: int = 360,
        step_min: int = 1,
        drivers: Optional[Dict[str, Any]] = None,
        scenario: Optional[str] = None,
        return_quantiles: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        output: Dict[str, List[Dict[str, Any]]] = {}

        for asset_id in asset_ids:
            try:
                recent = self.telemetry.get_recent_power(asset_id) or []
            except Exception:
                output[asset_id] = []
                continue
            timestamps, values = _clean_history(recent)
            fitted = _fit_autoregression(values)
            if fitted is None:
                output[asset_id] = []
                continue

            effective_step_min = _sampling_minutes(timestamps, step_min)
            points = int(horizon_min // effective_step_min)
            if points < 1:
                output[asset_id] = []
                continue

            coefficients, lag, residual_q10, residual_q90 = fitted
            window = list(values[-lag:])
            stability_cap_kw, stability_guard = self._stability_cap(asset_id, values)
            last_timestamp = timestamps[-1] if timestamps else datetime.now(timezone.utc)
            rows: List[Dict[str, Any]] = []
            for index in range(points):
                features = _feature(window[-lag:], len(values) + index - lag, len(values) + points)
                multiplier = _explicit_multiplier(drivers, index)
                unconstrained_p50 = float(features @ coefficients) * multiplier
                if not np.isfinite(unconstrained_p50):
                    unconstrained_p50 = stability_cap_kw
                p50 = min(stability_cap_kw, max(0.0, unconstrained_p50))
                guard_applied = abs(p50 - unconstrained_p50) > 1e-9
                timestamp = last_timestamp + timedelta(minutes=(index + 1) * effective_step_min)
                row: Dict[str, Any] = {
                    "ts": timestamp.isoformat(),
                    "kW": round(p50, 6),
                    "p50": round(p50, 6),
                    "model": "ridge_autoregression",
                    "model_step_min": effective_step_min,
                    "scenario": scenario,
                    "stability_guard": stability_guard,
                    "stability_cap_kw": round(stability_cap_kw, 6),
                    "guard_applied": guard_applied,
                }
                if return_quantiles and residual_q10 != residual_q90:
                    row["p10"] = round(min(stability_cap_kw, max(0.0, p50 + residual_q10)), 6)
                    row["p90"] = round(min(stability_cap_kw, max(0.0, p50 + residual_q90)), 6)
                rows.append(row)
                window.append(p50)
            output[asset_id] = rows
        return output
