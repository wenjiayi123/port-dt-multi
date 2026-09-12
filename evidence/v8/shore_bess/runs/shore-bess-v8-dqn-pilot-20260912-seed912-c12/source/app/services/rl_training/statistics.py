from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np


def bootstrap_summary(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260720,
) -> Dict[str, Any]:
    """Summarize repeated evaluation values with a deterministic bootstrap CI."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "ci_low": None, "ci_high": None}
    confidence = min(0.999, max(0.50, float(confidence)))
    if array.size == 1:
        low = high = float(array[0])
    else:
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, array.size, size=(max(200, int(resamples)), array.size))
        means = np.mean(array[indices], axis=1)
        alpha = (1.0 - confidence) / 2.0
        low, high = (float(value) for value in np.quantile(means, [alpha, 1.0 - alpha]))
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "method": "percentile_bootstrap_of_mean",
        "resamples": 0 if array.size == 1 else max(200, int(resamples)),
    }


def summarize_metric_rows(rows: Iterable[Mapping[str, Any]], *, seed: int = 20260720) -> Dict[str, Dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    keys = sorted({key for row in materialized for key, value in row.items() if isinstance(value, (int, float))})
    return {
        key: bootstrap_summary((float(row[key]) for row in materialized if isinstance(row.get(key), (int, float))), seed=seed)
        for key in keys
    }
