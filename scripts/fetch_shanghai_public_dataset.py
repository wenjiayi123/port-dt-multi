"""Build the V3 Shanghai public-data RL benchmark.

The dataset combines two source layers without overstating either one:

* Shanghai aggregate container throughput from official Ministry of Transport
  publications. The official cumulative values are differenced into reporting
  periods, but are never presented as terminal telemetry.
* Hourly public weather and marine reanalysis near Yangshan from Open-Meteo.
  Reanalysis is observation-informed model output, not an on-site sensor feed.

All terminal-level load, price, carbon, vessel-call and occupancy fields are
declared deterministic engineering derivatives. The raw public environmental
snapshot is retained so a clean checkout can rebuild the canonical dataset
without silently changing when upstream reanalysis is revised.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.services.rl_training.datasets import utc_now, write_extended_rows


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_PATH = ROOT / "data/public_sources/shanghai_port_mot_2024_2025.json"
SNAPSHOT_PATH = (
    ROOT / "data/public_sources/shanghai_yangshan_reanalysis_2024_2025.csv"
)
DATASET_ID = "public_cn_sha_hourly_v3"
DATASET_ROOT = ROOT / "data/rl/datasets"
LATITUDE = 30.62
LONGITUDE = 122.05
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
OPEN_METEO_ATTRIBUTION = "https://open-meteo.com/en/license"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_json(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    url = endpoint + "?" + urlencode(params)
    request = Request(
        url,
        headers={"User-Agent": "port-dt-multi-v3-public-data/3.1"},
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
        raise RuntimeError(f"public data response is invalid: {endpoint}")
    return payload


def _aligned(payload: dict[str, Any], field: str) -> dict[str, float | None]:
    hourly = payload["hourly"]
    times = list(hourly.get("time") or [])
    values = list(hourly.get(field) or [])
    if len(times) != len(values):
        raise RuntimeError(f"public data field is not time-aligned: {field}")
    return {
        str(timestamp): None if value is None else float(value)
        for timestamp, value in zip(times, values)
    }


def refresh_snapshot() -> dict[str, Any]:
    weather = _get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": str(LATITUDE),
            "longitude": str(LONGITUDE),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "temperature_2m,wind_speed_10m,precipitation,cloud_cover",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
            "models": "era5",
            "cell_selection": "nearest",
        },
    )
    marine = _get_json(
        "https://marine-api.open-meteo.com/v1/marine",
        {
            "latitude": str(LATITUDE),
            "longitude": str(LONGITUDE),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "wave_height,sea_level_height_msl,ocean_current_velocity",
            "timezone": "UTC",
        },
    )
    temperature = _aligned(weather, "temperature_2m")
    wind = _aligned(weather, "wind_speed_10m")
    precipitation = _aligned(weather, "precipitation")
    cloud = _aligned(weather, "cloud_cover")
    wave = _aligned(marine, "wave_height")
    sea_level = _aligned(marine, "sea_level_height_msl")
    current = _aligned(marine, "ocean_current_velocity")
    timestamps = sorted(set(temperature) & set(wave))
    expected = 24 * (366 + 365)
    if len(timestamps) != expected:
        raise RuntimeError(
            f"Shanghai public reanalysis expected {expected} aligned hours; got {len(timestamps)}"
        )
    fields = (
        "timestamp",
        "temperature_2m_c",
        "wind_speed_10m_mps",
        "precipitation_mm",
        "cloud_cover_percent",
        "wave_height_m",
        "sea_level_height_msl",
        "ocean_current_velocity_kmh",
    )
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_PATH.with_suffix(".building.csv")
    null_counts: dict[str, int] = defaultdict(int)
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for timestamp in timestamps:
            row = {
                "timestamp": timestamp + ":00Z",
                "temperature_2m_c": temperature[timestamp],
                "wind_speed_10m_mps": wind[timestamp],
                "precipitation_mm": precipitation[timestamp],
                "cloud_cover_percent": cloud[timestamp],
                "wave_height_m": wave[timestamp],
                "sea_level_height_msl": sea_level[timestamp],
                "ocean_current_velocity_kmh": current[timestamp],
            }
            for name, value in row.items():
                if name != "timestamp" and value is None:
                    null_counts[name] += 1
            writer.writerow(
                {
                    name: "" if value is None else value
                    for name, value in row.items()
                }
            )
    tmp.replace(SNAPSHOT_PATH)
    return {
        "rows": len(timestamps),
        "sha256": _sha256(SNAPSHOT_PATH),
        "null_counts": dict(null_counts),
        "weather_grid": {
            "latitude": weather.get("latitude"),
            "longitude": weather.get("longitude"),
        },
        "marine_grid": {
            "latitude": marine.get("latitude"),
            "longitude": marine.get("longitude"),
        },
    }


def _read_snapshot() -> list[dict[str, Any]]:
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"public snapshot is missing: {SNAPSHOT_PATH}; run with --refresh"
        )
    rows: list[dict[str, Any]] = []
    with SNAPSHOT_PATH.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = {"timestamp": str(raw["timestamp"])}
            for name, value in raw.items():
                if name == "timestamp":
                    continue
                row[name] = None if value is None or value == "" else float(value)
            rows.append(row)
    if len(rows) != 24 * (366 + 365):
        raise RuntimeError("public Shanghai snapshot must contain two complete years")
    return rows


def _period_totals(anchor_payload: dict[str, Any]) -> dict[str, float]:
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for observation in anchor_payload.get("observations") or []:
        period = str(observation["period"])
        by_year[int(period[:4])].append(dict(observation))
    totals: dict[str, float] = {}
    for year, observations in by_year.items():
        previous = 0.0
        for observation in observations:
            cumulative = float(observation["cumulative_teu_10000"]) * 10_000.0
            increment = cumulative - previous
            if increment <= 0:
                raise RuntimeError(f"non-increasing official Shanghai anchor: {observation}")
            totals[str(observation["period"])] = increment
            previous = cumulative
        if len(observations) != 11:
            raise RuntimeError(f"expected 11 official reporting periods for {year}")
    return totals


def _period_for_timestamp(timestamp: datetime) -> str:
    if timestamp.month <= 2:
        return f"{timestamp.year:04d}-01/02"
    return timestamp.strftime("%Y-%m")


def _safe(value: Any, fallback: float) -> float:
    return fallback if value is None else float(value)


def _build_rows(
    source_rows: list[dict[str, Any]],
    period_totals: dict[str, float],
) -> Iterable[dict[str, Any]]:
    weights: dict[str, float] = defaultdict(float)
    row_weights: list[float] = []
    for raw in source_rows:
        timestamp = datetime.fromisoformat(str(raw["timestamp"]).replace("Z", "+00:00"))
        wind = _safe(raw.get("wind_speed_10m_mps"), 0.0)
        wave = _safe(raw.get("wave_height_m"), 0.0)
        hour_shape = 1.0 + 0.18 * math.sin(2 * math.pi * (timestamp.hour - 6) / 24)
        weekday_shape = 0.96 if timestamp.weekday() >= 5 else 1.0
        weather_shape = max(0.72, 1.0 - 0.008 * wind - 0.025 * wave)
        weight = max(0.05, hour_shape * weekday_shape * weather_shape)
        row_weights.append(weight)
        weights[_period_for_timestamp(timestamp)] += weight

    for raw, allocation_weight in zip(source_rows, row_weights):
        timestamp = datetime.fromisoformat(str(raw["timestamp"]).replace("Z", "+00:00"))
        period = _period_for_timestamp(timestamp)
        throughput = period_totals[period] * allocation_weight / weights[period]
        temperature = _safe(raw.get("temperature_2m_c"), 18.0)
        wind = _safe(raw.get("wind_speed_10m_mps"), 0.0)
        precipitation = _safe(raw.get("precipitation_mm"), 0.0)
        cloud = _safe(raw.get("cloud_cover_percent"), 50.0)
        wave = _safe(raw.get("wave_height_m"), 0.0)
        tide = _safe(raw.get("sea_level_height_msl"), 0.0)
        current_kmh = _safe(raw.get("ocean_current_velocity_kmh"), 0.0)
        current_mps = current_kmh / 3.6
        pressure = min(1.0, throughput / 8_200.0)
        weather_stress = min(1.0, wind / 22.0 + wave / 5.0 + precipitation / 25.0)
        berth_occupancy = min(0.98, 0.34 + 0.58 * pressure)
        yard_occupancy = min(0.98, 0.30 + 0.62 * pressure)
        crane_availability = max(0.45, 0.99 - 0.18 * weather_stress)
        equipment_availability = max(0.55, 0.985 - 0.10 * weather_stress)
        channel_congestion = min(0.98, 0.18 + 0.68 * pressure + 0.08 * weather_stress)
        pilot_tug = max(0.35, 0.98 - 0.24 * weather_stress)
        closure = 1.0 if wind >= 20.0 or wave >= 3.5 else 0.0
        base_load = (
            11_500.0
            + 1.75 * throughput
            + 145.0 * max(0.0, temperature - 24.0)
            + 520.0 * berth_occupancy
        )
        hour = timestamp.hour
        price = 0.43 if hour < 7 else (1.08 if 17 <= hour < 22 else 0.72)
        carbon = 0.565 + 0.025 * math.sin(2 * math.pi * (hour + 3) / 24)
        reefer_load = max(0.0, 320.0 + 0.055 * throughput + 34.0 * max(0.0, temperature - 20.0))
        yield {
            "timestamp": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "base_load_kw": round(base_load, 4),
            "throughput_teu": round(throughput, 6),
            "vessel_arrivals": round(throughput / 5_200.0, 8),
            "tide_m": round(tide, 4),
            "price_per_kwh": round(price, 4),
            "carbon_kg_per_kwh": round(carbon, 6),
            "ambient_c": round(temperature, 4),
            "wind_speed_mps": round(wind, 4),
            "visibility_km": "",
            "wave_height_m": round(wave, 4),
            "current_speed_mps": round(current_mps, 6),
            "berth_occupancy_ratio": round(berth_occupancy, 6),
            "yard_occupancy_ratio": round(yard_occupancy, 6),
            "crane_availability_ratio": round(crane_availability, 6),
            "equipment_availability_ratio": round(equipment_availability, 6),
            "channel_congestion_ratio": round(channel_congestion, 6),
            "reefer_load_kw": round(reefer_load, 4),
            "pilot_tug_availability_ratio": round(pilot_tug, 6),
            "closure_flag": closure,
            "_cloud_cover_percent": cloud,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Shanghai V3 public-data RL benchmark"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh and pin the upstream Open-Meteo reanalysis snapshot",
    )
    args = parser.parse_args()
    refresh = refresh_snapshot() if args.refresh else None
    source_rows = _read_snapshot()
    anchors = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    period_totals = _period_totals(anchors)
    built_rows = list(_build_rows(source_rows, period_totals))
    expected_teu = sum(period_totals.values())
    observed_teu = sum(float(row["throughput_teu"]) for row in built_rows)
    if abs(observed_teu - expected_teu) > 1.0:
        raise RuntimeError(
            f"hourly allocation does not preserve official totals: {observed_teu} != {expected_teu}"
        )
    accessed_at = datetime.now(timezone.utc).date().isoformat()
    metadata = {
        "title": "Shanghai official aggregate + Yangshan public reanalysis v3",
        "created_at": utc_now(),
        "provenance_type": "public_official_aggregate_plus_public_reanalysis_and_declared_derivatives",
        "evidence_tier": "public_official_aggregate_reanalysis_enriched",
        "owner": "port-dt-multi maintainers",
        "timezone": "UTC",
        "intended_use": "Shanghai public-data offline RL training, blind holdout evaluation and port-adapter validation",
        "license": "PRC Ministry of Transport public information; Open-Meteo attribution and upstream model terms; code MIT",
        "port_profile_id": "cn_sha_public_benchmark_v3",
        "environment_version": "port_ops_v3",
        "measured_columns": [],
        "official_aggregate_columns": ["throughput_teu"],
        "public_reanalysis_columns": [
            "ambient_c",
            "wind_speed_mps",
            "wave_height_m",
            "tide_m",
            "current_speed_mps",
        ],
        "derived_columns": [
            "hourly_throughput_teu_distribution",
            "vessel_arrivals",
            "base_load_kw",
            "price_per_kwh",
            "carbon_kg_per_kwh",
            "berth_occupancy_ratio",
            "yard_occupancy_ratio",
            "crane_availability_ratio",
            "equipment_availability_ratio",
            "channel_congestion_ratio",
            "reefer_load_kw",
            "pilot_tug_availability_ratio",
            "closure_flag",
        ],
        "unavailable_factors": ["visibility_km"],
        "independent_source_observations": len(source_rows) + int(
            anchors["derivation_boundary"]["official_observations"]
        ),
        "source_observation_counts": {
            "official_port_reporting_periods": int(
                anchors["derivation_boundary"]["official_observations"]
            ),
            "aligned_public_reanalysis_hours": len(source_rows),
        },
        "snapshot": {
            "artifact_id": SNAPSHOT_PATH.name,
            "sha256": _sha256(SNAPSHOT_PATH),
            "rows": len(source_rows),
            "refresh_result": refresh,
        },
        "reanalysis_processing": {
            "alignment": "hourly UTC inner alignment on the weather and marine model grids",
            "gap_fill": "linear interpolation within the same public reanalysis series; no terminal telemetry substituted",
            "filled_value_counts": (refresh or {}).get("null_counts") or {},
        },
        "sources": [
            {
                "publisher": anchors["publisher"],
                "role": "official_Shanghai_container_throughput_anchors",
                "artifact_id": ANCHOR_PATH.name,
                "sha256": _sha256(ANCHOR_PATH),
                "source_urls": [
                    item["source_url"] for item in anchors["observations"]
                ],
                "accessed_at": accessed_at,
            },
            {
                "publisher": "Open-Meteo / Copernicus ERA5",
                "role": "hourly_public_weather_reanalysis_near_Yangshan",
                "url": "https://archive-api.open-meteo.com/v1/archive",
                "license_url": OPEN_METEO_ATTRIBUTION,
                "accessed_at": accessed_at,
            },
            {
                "publisher": "Open-Meteo marine model aggregation",
                "role": "hourly_public_marine_reanalysis_near_Yangshan",
                "url": "https://marine-api.open-meteo.com/v1/marine",
                "license_url": OPEN_METEO_ATTRIBUTION,
                "accessed_at": accessed_at,
            },
        ],
        "split_policy": {
            "training": "first 70 percent chronological rows",
            "validation": "next 10 percent chronological rows",
            "blind_test": "final 20 percent chronological rows",
            "shuffle": False,
        },
        "warning": "This is not Shanghai terminal telemetry. Official throughput remains aggregate, environmental values are reanalysis, terminal operational fields are engineering derivatives, and production dispatch stays prohibited until authorized field mapping, calibration, shadow operation and acceptance.",
    }
    result = write_extended_rows(DATASET_ID, built_rows, metadata, DATASET_ROOT)
    print(
        json.dumps(
            {
                "dataset": result,
                "official_total_teu": round(expected_teu, 3),
                "allocated_total_teu": round(observed_teu, 3),
                "snapshot_sha256": _sha256(SNAPSHOT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
