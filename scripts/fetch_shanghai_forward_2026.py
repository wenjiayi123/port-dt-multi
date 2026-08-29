"""Build the fresh 2026 Shanghai public forward-challenge dataset.

This artifact is deliberately separate from the 2024-2025 training benchmark.
It uses later official reporting periods plus pinned public reanalysis so a
candidate chosen on historical training/validation data can be evaluated once
on a temporally forward window. It is not Shanghai terminal telemetry.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.rl_training.datasets import utc_now, write_extended_rows
from scripts import fetch_shanghai_public_dataset as historical


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_PATH = ROOT / "data/public_sources/shanghai_port_mot_2026_forward.json"
SNAPSHOT_PATH = ROOT / "data/public_sources/shanghai_yangshan_reanalysis_2026_01_05.csv"
DATASET_ID = "public_cn_sha_forward_2026m05_v1"
DATASET_ROOT = ROOT / "data/rl/datasets"
START_DATE = "2026-01-01"
END_DATE = "2026-05-31"
EXPECTED_HOURS = 151 * 24


def refresh_snapshot() -> dict[str, Any]:
    weather = historical._get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": str(historical.LATITUDE),
            "longitude": str(historical.LONGITUDE),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "temperature_2m,wind_speed_10m,precipitation,cloud_cover",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
            "models": "era5",
            "cell_selection": "nearest",
        },
    )
    marine = historical._get_json(
        "https://marine-api.open-meteo.com/v1/marine",
        {
            "latitude": str(historical.LATITUDE),
            "longitude": str(historical.LONGITUDE),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "wave_height,sea_level_height_msl,ocean_current_velocity",
            "timezone": "UTC",
        },
    )
    series = {
        "temperature_2m_c": historical._aligned(weather, "temperature_2m"),
        "wind_speed_10m_mps": historical._aligned(weather, "wind_speed_10m"),
        "precipitation_mm": historical._aligned(weather, "precipitation"),
        "cloud_cover_percent": historical._aligned(weather, "cloud_cover"),
        "wave_height_m": historical._aligned(marine, "wave_height"),
        "sea_level_height_msl": historical._aligned(marine, "sea_level_height_msl"),
        "ocean_current_velocity_kmh": historical._aligned(marine, "ocean_current_velocity"),
    }
    timestamps = sorted(set.intersection(*(set(values) for values in series.values())))
    if len(timestamps) != EXPECTED_HOURS:
        raise RuntimeError(
            f"2026 Shanghai forward snapshot expected {EXPECTED_HOURS} aligned hours; got {len(timestamps)}"
        )
    null_counts: dict[str, int] = defaultdict(int)
    fields = ("timestamp", *series)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_PATH.with_suffix(".building.csv")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for timestamp in timestamps:
            row: dict[str, Any] = {"timestamp": timestamp + ":00Z"}
            for field, values in series.items():
                value = values[timestamp]
                if value is None:
                    null_counts[field] += 1
                row[field] = "" if value is None else value
            writer.writerow(row)
    tmp.replace(SNAPSHOT_PATH)
    return {
        "rows": len(timestamps),
        "sha256": historical._sha256(SNAPSHOT_PATH),
        "null_counts": dict(null_counts),
        "weather_grid": {"latitude": weather.get("latitude"), "longitude": weather.get("longitude")},
        "marine_grid": {"latitude": marine.get("latitude"), "longitude": marine.get("longitude")},
    }


def read_snapshot() -> list[dict[str, Any]]:
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"missing forward snapshot: {SNAPSHOT_PATH}; run with --refresh")
    rows: list[dict[str, Any]] = []
    with SNAPSHOT_PATH.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = {"timestamp": str(raw["timestamp"])}
            for name, value in raw.items():
                if name != "timestamp":
                    row[name] = None if value in (None, "") else float(value)
            rows.append(row)
    if len(rows) != EXPECTED_HOURS:
        raise RuntimeError(f"forward snapshot must contain {EXPECTED_HOURS} hours")
    return rows


def period_totals(payload: dict[str, Any]) -> dict[str, float]:
    observations = list(payload.get("observations") or [])
    if len(observations) != 4:
        raise RuntimeError("2026 forward challenge requires four pinned official observations")
    previous = 0.0
    totals: dict[str, float] = {}
    for observation in observations:
        cumulative = float(observation["cumulative_teu_10000"]) * 10_000.0
        increment = cumulative - previous
        if increment <= 0:
            raise RuntimeError(f"non-increasing official anchor: {observation}")
        totals[str(observation["period"])] = increment
        previous = cumulative
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Shanghai 2026 public forward challenge")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    refresh = refresh_snapshot() if args.refresh else None
    anchors = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    source_rows = read_snapshot()
    totals = period_totals(anchors)
    built_rows = list(historical._build_rows(source_rows, totals))
    expected_teu = sum(totals.values())
    observed_teu = sum(float(row["throughput_teu"]) for row in built_rows)
    if abs(observed_teu - expected_teu) > 1.0:
        raise RuntimeError("hourly allocation does not preserve official totals")
    accessed_at = datetime.now(timezone.utc).date().isoformat()
    metadata = {
        "title": "Shanghai 2026 official aggregate + Yangshan public forward reanalysis",
        "created_at": utc_now(),
        "provenance_type": "public_official_aggregate_plus_public_reanalysis_and_declared_derivatives",
        "evidence_tier": "public_forward_challenge_not_site_telemetry",
        "owner": "port-dt-multi maintainers",
        "timezone": "UTC",
        "intended_use": "single-use temporal forward challenge after candidate selection on the 2024-2025 training/validation split",
        "license": "PRC Ministry of Transport public information; Open-Meteo attribution and upstream model terms; code MIT",
        "port_profile_id": "cn_sha_public_benchmark_v3",
        "environment_version": "port_ops_v3_forward_challenge",
        "measured_columns": [],
        "official_aggregate_columns": ["throughput_teu"],
        "public_reanalysis_columns": ["ambient_c", "wind_speed_mps", "wave_height_m", "tide_m", "current_speed_mps"],
        "derived_columns": [
            "hourly_throughput_teu_distribution", "vessel_arrivals", "base_load_kw",
            "price_per_kwh", "carbon_kg_per_kwh", "berth_occupancy_ratio",
            "yard_occupancy_ratio", "crane_availability_ratio",
            "equipment_availability_ratio", "channel_congestion_ratio",
            "reefer_load_kw", "pilot_tug_availability_ratio", "closure_flag"
        ],
        "unavailable_factors": ["visibility_km", "terminal_metering", "equipment_control_actions", "site_tariff", "site_carbon_intensity"],
        "independent_source_observations": len(source_rows) + 4,
        "source_observation_counts": {"official_port_reporting_periods": 4, "aligned_public_reanalysis_hours": len(source_rows)},
        "snapshot": {"artifact_id": SNAPSHOT_PATH.name, "sha256": historical._sha256(SNAPSHOT_PATH), "rows": len(source_rows), "refresh_result": refresh},
        "sources": [
            {
                "publisher": anchors["publisher"], "role": "official_Shanghai_container_throughput_forward_anchors",
                "artifact_id": ANCHOR_PATH.name, "sha256": historical._sha256(ANCHOR_PATH),
                "publication_urls": [item["publication_url"] for item in anchors["observations"]],
                "source_urls": [item["source_url"] for item in anchors["observations"]], "accessed_at": accessed_at
            },
            {
                "publisher": "Open-Meteo / Copernicus ERA5", "role": "hourly_public_weather_reanalysis_near_Yangshan",
                "url": "https://archive-api.open-meteo.com/v1/archive", "license_url": historical.OPEN_METEO_ATTRIBUTION, "accessed_at": accessed_at
            },
            {
                "publisher": "Open-Meteo marine model aggregation", "role": "hourly_public_marine_model_near_Yangshan",
                "url": "https://marine-api.open-meteo.com/v1/marine", "license_url": historical.OPEN_METEO_ATTRIBUTION, "accessed_at": accessed_at
            }
        ],
        "split_policy": {"role": "forward_challenge_only", "candidate_selection_allowed": False, "shuffle": False},
        "warning": "Not Shanghai terminal telemetry. Aggregate throughput and environmental model data cannot validate site power/control economics; equipment-level results remain engineering replay until authorized field data passes calibration and shadow-operation gates."
    }
    result = write_extended_rows(DATASET_ID, built_rows, metadata, DATASET_ROOT)
    print(json.dumps({"dataset": result, "official_total_teu": expected_teu, "allocated_total_teu": round(observed_teu, 3), "snapshot_sha256": historical._sha256(SNAPSHOT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
