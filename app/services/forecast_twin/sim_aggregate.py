"""Aggregate genuine digital-twin outputs without display-oriented fallbacks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple


def _safe_list_assets(di: Any, limit: int) -> List[Dict[str, Any]]:
    try:
        assets = di.telemetry.list_assets() or []
    except Exception:
        return []
    return [item for item in assets if isinstance(item, dict) and (item.get("id") or item.get("asset_id"))][
        :limit
    ]


def _classify(asset_id: str) -> str:
    value = (asset_id or "").lower()
    for prefix in ("qc", "yc", "agv", "wh", "cs", "ps", "yard"):
        if value.startswith(prefix):
            return prefix
    return "misc"


def _run_single(
    di: Any,
    asset_id: str,
    scenario: str,
    horizon_min: int,
    step_min: int,
) -> Tuple[List[Any], List[float], List[float], List[float], int]:
    """Return timestamps and model-produced p10/p50/p90 for one asset.

    Forecast substitution and percentage uncertainty bands are intentionally not
    allowed: a simulation request must be backed by the configured twin adapter.
    """
    try:
        data = di.twin.run(
            asset_id=asset_id,
            horizon_min=horizon_min,
            step_min=step_min,
            scenario=scenario,
            use_drivers=True,
        ) or {}
    except Exception:
        return [], [], [], [], step_min

    plan = [point for point in (data.get("plan") or []) if isinstance(point, dict)]
    if not plan:
        return [], [], [], [], step_min

    p50 = [float(point.get("p50", point.get("kW", 0.0)) or 0.0) for point in plan]
    has_p10 = all(point.get("p10") is not None for point in plan)
    has_p90 = all(point.get("p90") is not None for point in plan)
    p10 = [float(point["p10"]) for point in plan] if has_p10 else []
    p90 = [float(point["p90"]) for point in plan] if has_p90 else []
    timestamps = [point.get("ts") for point in plan]
    effective_step = int((data.get("window") or {}).get("step_min_effective") or step_min)
    return timestamps, p10, p50, p90, effective_step


def aggregate_sim(
    di: Any,
    scenario: str = "baseline",
    horizon_min: int = 360,
    step_min: int = 1,
    limit: int = 50,
) -> Dict[str, Any]:
    assets = _safe_list_assets(di, limit)
    series_p10: List[List[float]] = []
    series_p50: List[List[float]] = []
    series_p90: List[List[float]] = []
    timestamp_candidates: List[List[Any]] = []
    items: List[Dict[str, Any]] = []
    effective_steps: List[int] = []

    for asset in assets:
        asset_id = str(asset.get("id") or asset.get("asset_id"))
        timestamps, p10, p50, p90, effective_step = _run_single(di, asset_id, scenario, horizon_min, step_min)
        if not p50:
            continue
        series_p50.append(p50)
        timestamp_candidates.append(timestamps)
        effective_steps.append(effective_step)
        if p10:
            series_p10.append(p10)
        if p90:
            series_p90.append(p90)
        items.append(
            {
                "id": asset_id,
                "type": _classify(asset_id),
                "avgKW": round(sum(p50) / len(p50), 3),
                "uncertainty_available": bool(p10 and p90),
            }
        )

    if not series_p50:
        return {
            "available": False,
            "reason": "No configured twin adapter returned a simulation plan",
            "scenario": scenario,
            "agg": {"p10": [], "p50": [], "p90": []},
            "ts": [],
            "p10": [],
            "p50": [],
            "p90": [],
            "assets": [],
            "count": 0,
            "_source": "twin_adapter_unavailable",
        }

    length = min(len(values) for values in series_p50)
    agg_p50 = [round(sum(values[i] for values in series_p50), 6) for i in range(length)]
    quantiles_complete = len(series_p10) == len(series_p50) and len(series_p90) == len(series_p50)
    if quantiles_complete:
        quantile_length = min(
            length,
            *(len(values) for values in series_p10),
            *(len(values) for values in series_p90),
        )
        length = quantile_length
        agg_p50 = agg_p50[:length]
        agg_p10 = [round(sum(values[i] for values in series_p10), 6) for i in range(length)]
        agg_p90 = [round(sum(values[i] for values in series_p90), 6) for i in range(length)]
    else:
        agg_p10 = []
        agg_p90 = []

    timestamps = (timestamp_candidates[0] if timestamp_candidates else [])[:length]
    effective_step_min = max(effective_steps) if effective_steps else step_min
    if not timestamps or any(value is None for value in timestamps):
        start = datetime.now(timezone.utc)
        timestamps = [(start + timedelta(minutes=i * max(1, effective_step_min))).isoformat() for i in range(length)]
        timestamp_source = "generated_axis_from_requested_step"
    else:
        timestamp_source = "twin_adapter"

    start_iso = timestamps[0] if timestamps else None
    end_iso = timestamps[-1] if timestamps else None
    total_kwh_p50 = round(sum(agg_p50) * (effective_step_min / 60.0), 3)
    aggregate = {
        "p10": agg_p10,
        "p50": agg_p50,
        "p90": agg_p90,
        "sum_kw_p50": round(agg_p50[-1], 3) if agg_p50 else 0.0,
        "total_kWh_p50": total_kwh_p50,
    }
    return {
        "available": True,
        "scenario": scenario,
        "updated": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start": start_iso,
            "end": end_iso,
            "horizon_min": horizon_min,
            "step_min": step_min,
            "step_min_effective": effective_step_min,
        },
        "agg": aggregate,
        "ts": timestamps,
        "p10": agg_p10,
        "p50": agg_p50,
        "p90": agg_p90,
        "assets": items,
        "count": len(items),
        "uncertainty_available": quantiles_complete,
        "timestamp_source": timestamp_source,
        "_source": "twin_adapter",
    }
