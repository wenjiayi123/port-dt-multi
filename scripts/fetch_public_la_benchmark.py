"""Build a high-frequency public benchmark for the Los Angeles port region.

Measured public inputs:
- U.S. DOT/BTS monthly Port of Los Angeles TEU totals.
- NOAA CO-OPS verified six-minute Los Angeles water levels.
- NOAA CO-OPS six-minute Santa Monica air temperature and wind.

The remaining canonical controls and operational factors are deterministic
engineering derivatives and are labelled per field in the dataset metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.services.rl_training.datasets import utc_now, write_extended_rows


BTS_URL = "https://data.bts.gov/resource/rd72-aq8r.json?$limit=500"
NOAA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_WATER_STATION = "9410660"
NOAA_MET_STATION = "9410840"


def get_json(url: str, attempts: int = 4) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "port-dt-multi-open-source/2.0"})
            with urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"public data request failed: {url}") from last_error


def noaa_month(product: str, station: str, begin: datetime, end: datetime) -> Dict[str, Dict[str, Any]]:
    params = {
        "product": product,
        "application": "port-dt-multi",
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "station": station,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
    }
    if product == "water_level":
        params["datum"] = "MLLW"
    payload = get_json(NOAA_API + "?" + urlencode(params))
    if payload.get("error"):
        raise RuntimeError(f"NOAA {product} error: {payload['error']}")
    return {str(item["t"]): item for item in payload.get("data") or []}


def measured_value(records: Dict[str, Dict[str, Any]], timestamp: datetime, field: str) -> float | None:
    item = records.get(timestamp.strftime("%Y-%m-%d %H:%M"))
    if not item:
        return None
    raw = item.get(field)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fill_short_gaps(values: list[float | None], max_gap: int = 10) -> list[float | None]:
    output = list(values)
    last_index: int | None = None
    last_value: float | None = None
    for index, value in enumerate(output):
        if value is not None:
            if last_index is not None and index - last_index - 1 <= max_gap:
                next_value = value
                gap = index - last_index
                for offset in range(1, gap):
                    weight = offset / gap
                    output[last_index + offset] = float(last_value + weight * (next_value - last_value))
            last_index = index
            last_value = value
    return output


def bts_monthly_teu(year: int) -> Dict[int, float]:
    payload = get_json(BTS_URL)
    monthly: Dict[int, float] = {}
    for item in payload:
        raw_date = str(item.get("port") or "")
        try:
            parsed = datetime.strptime(raw_date, "%m/%d/%Y")
        except ValueError:
            continue
        if parsed.year == year and item.get("los_angeles_ca") is not None:
            monthly[parsed.month] = float(item["los_angeles_ca"])
    if set(monthly) != set(range(1, 13)):
        missing = sorted(set(range(1, 13)) - set(monthly))
        raise RuntimeError(f"BTS monthly TEU data is missing months for {year}: {missing}")
    return monthly


def build_rows(
    year: int,
) -> tuple[list[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    monthly_teu = bts_monthly_teu(year)
    min_teu = min(monthly_teu.values())
    max_teu = max(monthly_teu.values())
    rows: list[Dict[str, Any]] = []
    measured_counts = {"water_level": 0, "air_temperature": 0, "wind": 0}
    interpolated_counts = {"water_level": 0, "air_temperature": 0, "wind": 0}
    for month in range(1, 13):
        begin = datetime(year, month, 1, tzinfo=timezone.utc)
        next_month = (
            datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            if month == 12
            else datetime(year, month + 1, 1, tzinfo=timezone.utc)
        )
        end = next_month - timedelta(days=1)
        water = noaa_month("water_level", NOAA_WATER_STATION, begin, end)
        temperature = noaa_month("air_temperature", NOAA_MET_STATION, begin, end)
        wind = noaa_month("wind", NOAA_MET_STATION, begin, end)
        stamps: list[datetime] = []
        current = begin
        while current < next_month:
            stamps.append(current)
            current += timedelta(minutes=6)
        raw_tide_values = [measured_value(water, stamp, "v") for stamp in stamps]
        raw_ambient_values = [
            measured_value(temperature, stamp, "v") for stamp in stamps
        ]
        raw_wind_values = [measured_value(wind, stamp, "s") for stamp in stamps]
        tide_values = fill_short_gaps(raw_tide_values)
        ambient_values = fill_short_gaps(raw_ambient_values)
        wind_values = fill_short_gaps(raw_wind_values)
        eligible_indices = [
            index
            for index in range(len(stamps))
            if tide_values[index] is not None and ambient_values[index] is not None
        ]
        for name, raw_values, completed_values in (
            ("water_level", raw_tide_values, tide_values),
            ("air_temperature", raw_ambient_values, ambient_values),
            ("wind", raw_wind_values, wind_values),
        ):
            raw_count = sum(
                raw_values[index] is not None for index in eligible_indices
            )
            completed_count = sum(
                completed_values[index] is not None for index in eligible_indices
            )
            measured_counts[name] += raw_count
            interpolated_counts[name] += completed_count - raw_count
        if sum(value is not None for value in tide_values) < len(stamps) * 0.95:
            raise RuntimeError(f"NOAA water-level coverage below 95% for {year}-{month:02d}")
        if sum(value is not None for value in ambient_values) < len(stamps) * 0.90:
            raise RuntimeError(f"NOAA air-temperature coverage below 90% for {year}-{month:02d}")
        weights = []
        for stamp in stamps:
            work_wave = 1.0 + 0.20 * math.sin(2 * math.pi * (stamp.hour - 5) / 24)
            weekday = 0.94 if stamp.weekday() >= 5 else 1.0
            weights.append(work_wave * weekday)
        weight_sum = sum(weights)
        monthly_total = monthly_teu[month]
        throughput_scale = (monthly_total - min_teu) / max(1.0, max_teu - min_teu)
        for index, stamp in enumerate(stamps):
            tide = tide_values[index]
            ambient = ambient_values[index]
            wind_speed = wind_values[index]
            if tide is None or ambient is None:
                continue
            throughput = monthly_total * weights[index] / weight_sum
            # Vessel arrivals are an explicit TEU-per-call proxy because a
            # terminal-level call stream is not present in the BTS table.
            vessel_arrivals = throughput / 6500.0
            berth_occupancy = min(0.98, 0.46 + 0.42 * throughput_scale + 0.06 * weights[index])
            yard_occupancy = min(0.98, 0.50 + 0.36 * throughput_scale + 0.05 * weights[index])
            congestion = min(1.0, 0.30 + 0.55 * throughput_scale + 0.06 * weights[index])
            reefer_load = 160.0 + 140.0 * throughput_scale + 22.0 * weights[index]
            base_load = 1700.0 + 1050.0 * throughput_scale + 360.0 * weights[index] + reefer_load
            price = 0.11 if stamp.hour < 16 else (0.28 if 16 <= stamp.hour < 21 else 0.15)
            carbon = 0.225
            rows.append(
                {
                    "timestamp": stamp.isoformat().replace("+00:00", "Z"),
                    "base_load_kw": round(base_load, 5),
                    "throughput_teu": round(throughput, 8),
                    "vessel_arrivals": round(vessel_arrivals, 10),
                    "tide_m": round(tide, 5),
                    "price_per_kwh": price,
                    "carbon_kg_per_kwh": carbon,
                    "ambient_c": round(ambient, 4),
                    "wind_speed_mps": "" if wind_speed is None else round(wind_speed, 4),
                    "berth_occupancy_ratio": round(berth_occupancy, 6),
                    "yard_occupancy_ratio": round(yard_occupancy, 6),
                    "channel_congestion_ratio": round(congestion, 6),
                    "reefer_load_kw": round(reefer_load, 5),
                }
            )
    return rows, measured_counts, interpolated_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public Los Angeles six-minute benchmark")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--dataset-id", default="public_us_la_6min_v1")
    args = parser.parse_args()
    if not 2019 <= args.year <= 2021:
        parser.error("BTS rd72-aq8r currently provides complete 2019-2021 monthly series")
    rows, measured_counts, interpolated_counts = build_rows(args.year)
    accessed_at = datetime.now(timezone.utc).date().isoformat()
    metadata = {
        "title": f"Los Angeles public six-minute benchmark {args.year}",
        "created_at": utc_now(),
        "provenance_type": "public_official_measurements_plus_declared_engineering_derivatives",
        "evidence_tier": "public_measured_enriched",
        "owner": "port-dt-multi maintainers",
        "timezone": "UTC",
        "intended_use": "reproducible offline RL benchmark and port-adapter validation",
        "license": "U.S. public domain source data; code MIT",
        "port_profile_id": "us_la_public_benchmark_v2",
        "environment_version": "port_ops_v2",
        "measured_columns": ["tide_m", "ambient_c", "wind_speed_mps"],
        "column_treatments": {
            "tide_m": "NOAA measured with declared short-gap linear interpolation",
            "ambient_c": "NOAA measured with declared short-gap linear interpolation",
            "wind_speed_mps": "NOAA measured with declared short-gap linear interpolation",
            "throughput_teu": "BTS monthly aggregate with deterministic six-minute allocation",
        },
        "official_aggregate_columns": ["throughput_teu"],
        "derived_columns": [
            "base_load_kw",
            "throughput_teu_six_minute_distribution",
            "vessel_arrivals",
            "price_per_kwh",
            "carbon_kg_per_kwh",
            "berth_occupancy_ratio",
            "yard_occupancy_ratio",
            "channel_congestion_ratio",
            "reefer_load_kw",
        ],
        "unavailable_factors": [
            "visibility_km",
            "wave_height_m",
            "current_speed_mps",
            "crane_availability_ratio",
            "equipment_availability_ratio",
            "pilot_tug_availability_ratio",
            "closure_flag",
        ],
        "independent_source_observations": sum(measured_counts.values()) + 12,
        "source_observation_counts": {**measured_counts, "monthly_teu": 12},
        "short_gap_interpolation_counts": interpolated_counts,
        "short_gap_interpolation_max_steps": 10,
        "sources": [
            {
                "publisher": "U.S. Department of Transportation, Bureau of Transportation Statistics",
                "dataset_id": "rd72-aq8r",
                "role": "monthly_port_of_los_angeles_teu",
                "url": BTS_URL,
                "license_url": "https://www.usa.gov/publicdomain/label/1.0/",
                "accessed_at": accessed_at,
            },
            {
                "publisher": "NOAA Center for Operational Oceanographic Products and Services",
                "station_id": NOAA_WATER_STATION,
                "role": "verified_six_minute_water_level",
                "url": NOAA_API,
                "accessed_at": accessed_at,
            },
            {
                "publisher": "NOAA Center for Operational Oceanographic Products and Services",
                "station_id": NOAA_MET_STATION,
                "role": "six_minute_air_temperature_and_wind",
                "url": NOAA_API,
                "accessed_at": accessed_at,
            },
        ],
        "warning": "This is a public benchmark, not terminal telemetry. Site equipment, berth, yard, tariff, carbon, call and control-limit fields require operator-approved replacement before field use.",
    }
    result = write_extended_rows(args.dataset_id, rows, metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
