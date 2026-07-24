"""Build the reproducible public-data example used by the RL engine.

Public-source inputs:
- Singapore MPA monthly container throughput via data.gov.sg.
- Singapore MPA monthly container-vessel arrivals via data.gov.sg.

The operational columns not published by those sources are deterministic,
documented engineering derivatives. They are not represented as live port data.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

from app.services.rl_training.datasets import utc_now, write_canonical_rows


MPA_DATASET_ID = "d_da030f7028200d19ffcbe4a2d71af39c"
MPA_URL = f"https://data.gov.sg/api/action/datastore_search?resource_id={MPA_DATASET_ID}&limit=500"
MPA_CONTAINER_ARRIVALS_DATASET_ID = "d_8f264219109e61fffa87ac64dd5a9a65"
MPA_CONTAINER_ARRIVALS_URL = (
    "https://data.gov.sg/api/action/datastore_search?"
    f"resource_id={MPA_CONTAINER_ARRIVALS_DATASET_ID}&limit=5000"
)


def get_json(url: str):
    request = Request(url, headers={"User-Agent": "port-dt-multi-open-source/1.0"})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def main() -> None:
    mpa = get_json(MPA_URL)
    monthly = {
        row["month"]: float(row["container_throughput"]) * 1000.0
        for row in mpa["result"]["records"]
    }
    arrivals_payload = get_json(MPA_CONTAINER_ARRIVALS_URL)
    monthly_container_arrivals = {
        row["month"]: float(row["number_of_vessels"])
        for row in arrivals_payload["result"]["records"]
        if str(row.get("vessel_type", "")).strip().lower() == "container"
    }
    # Six complete historical years make the chronological test boundary
    # meaningful while avoiding preliminary 2026 observations.
    begin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, 23, tzinfo=timezone.utc)
    accessed_at = datetime.now(timezone.utc).date().isoformat()
    required_months = {
        f"{year:04d}-{month:02d}"
        for year in range(begin.year, end.year + 1)
        for month in range(1, 13)
    }
    missing = sorted(
        required_months - set(monthly) | required_months - set(monthly_container_arrivals)
    )
    if missing:
        raise RuntimeError(f"MPA public inputs are missing months: {', '.join(missing)}")
    values = [monthly[key] for key in sorted(required_months)]
    min_teu, max_teu = min(values), max(values)
    rows = []
    month_start = begin
    while month_start <= end:
        month_key = month_start.strftime("%Y-%m")
        monthly_teu = monthly[month_key]
        monthly_arrivals = monthly_container_arrivals[month_key]
        throughput_scale = (monthly_teu - min_teu) / max(1.0, max_teu - min_teu)
        next_month = (
            datetime(month_start.year + 1, 1, 1, tzinfo=timezone.utc)
            if month_start.month == 12
            else datetime(
                month_start.year, month_start.month + 1, 1, tzinfo=timezone.utc
            )
        )
        hours: list[datetime] = []
        current = month_start
        while current < next_month:
            hours.append(current)
            current += timedelta(hours=1)
        throughput_weights = []
        arrival_weights = []
        for current in hours:
            daily_wave = 1.0 + 0.26 * math.sin(
                2 * math.pi * (current.hour - 6) / 24
            )
            weekend_factor = 0.91 if current.weekday() >= 5 else 1.0
            throughput_weights.append(daily_wave * weekend_factor)
            arrival_weights.append(
                1.0
                + 0.18 * math.cos(2 * math.pi * (current.hour - 8) / 24)
            )
        throughput_weight_sum = sum(throughput_weights)
        arrival_weight_sum = sum(arrival_weights)
        for current, throughput_weight, arrival_weight in zip(
            hours, throughput_weights, arrival_weights
        ):
            hour = current.hour
            throughput_teu = monthly_teu * throughput_weight / throughput_weight_sum
            vessel_arrivals = (
                monthly_arrivals * arrival_weight / arrival_weight_sum
            )
            # Canonical tide remains a declared harmonic stress input; it is
            # not used as an observed Singapore tide or business-KPI driver.
            tide = 0.8 * math.sin(2 * math.pi * (hour + 1) / 12.42)
            daily_wave = throughput_weight / (
                0.91 if current.weekday() >= 5 else 1.0
            )
            base_load_kw = (
                1850.0
                + 950.0 * throughput_scale
                + 410.0 * daily_wave
            )
            price = 0.72 if hour < 7 else (1.34 if 17 <= hour < 22 else 1.02)
            carbon = 0.46 + 0.05 * math.sin(
                2 * math.pi * (hour + 2) / 24
            )
            ambient = 28.0 + 3.2 * math.sin(
                2 * math.pi * (hour - 8) / 24
            )
            rows.append(
                {
                    "timestamp": current.isoformat().replace("+00:00", "Z"),
                    "base_load_kw": round(base_load_kw, 4),
                    "throughput_teu": round(throughput_teu, 4),
                    "vessel_arrivals": round(vessel_arrivals, 6),
                    "tide_m": round(tide, 4),
                    "price_per_kwh": round(price, 4),
                    "carbon_kg_per_kwh": round(carbon, 6),
                    "ambient_c": round(ambient, 4),
                }
            )
        month_start = next_month
    metadata = {
        "title": "Public port operations driver example v1",
        "created_at": utc_now(),
        "provenance_type": "public_official_inputs_plus_deterministic_engineering_derivatives",
        "owner": "port-dt-multi maintainers",
        "timezone": "UTC",
        "intended_use": "integration testing, reproducible RL research, and port data-adapter validation",
        "license": "Singapore Open Data Licence; code MIT",
        "geographically_coherent_single_port_series": False,
        "official_input_geography_coherent": True,
        "official_source_columns": [
            "monthly_container_throughput",
            "monthly_container_vessel_arrivals",
        ],
        "derived_columns": [
            "base_load_kw",
            "hourly_throughput_teu",
            "hourly_vessel_arrivals",
            "tide_m",
            "price_per_kwh",
            "carbon_kg_per_kwh",
            "ambient_c",
        ],
        "sources": [
            {
                "publisher": "Maritime and Port Authority of Singapore",
                "dataset_id": MPA_DATASET_ID,
                "role": "monthly_container_throughput_anchor",
                "accessed_at": accessed_at,
                "license_url": "https://data.gov.sg/open-data-licence",
                "url": MPA_URL,
            },
            {
                "publisher": "Maritime and Port Authority of Singapore",
                "dataset_id": MPA_CONTAINER_ARRIVALS_DATASET_ID,
                "role": "monthly_container_vessel_arrivals_anchor",
                "accessed_at": accessed_at,
                "license_url": "https://data.gov.sg/open-data-licence",
                "url": MPA_CONTAINER_ARRIVALS_URL,
            },
        ],
        "warning": "This is an integration/reproducibility dataset, not a production telemetry export. Replace it with a mapped port dataset for deployment.",
    }
    result = write_canonical_rows("public_port_ops_v1", rows, metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
