"""Build the reproducible public-data example used by the RL engine.

Official inputs:
- Singapore MPA monthly container throughput via data.gov.sg.
- NOAA CO-OPS hourly tide predictions for station 9414290.

The operational columns not published by those sources are deterministic,
documented engineering derivatives. They are not represented as live port data.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.services.rl_training.datasets import utc_now, write_canonical_rows


MPA_DATASET_ID = "d_da030f7028200d19ffcbe4a2d71af39c"
MPA_URL = f"https://data.gov.sg/api/action/datastore_search?resource_id={MPA_DATASET_ID}&limit=500"
NOAA_STATION = "9414290"


def get_json(url: str):
    request = Request(url, headers={"User-Agent": "port-dt-multi-open-source/1.0"})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def main() -> None:
    mpa = get_json(MPA_URL)
    monthly = {row["month"]: float(row["container_throughput"]) * 1000.0 for row in mpa["result"]["records"]}
    # A fixed historical window makes the checked-in example reproducible.
    begin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 31, tzinfo=timezone.utc)
    query = urlencode(
        {
            "product": "predictions",
            "application": "port_dt_multi_open_source",
            "begin_date": begin.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "datum": "MLLW",
            "station": NOAA_STATION,
            "time_zone": "gmt",
            "units": "metric",
            "interval": "h",
            "format": "json",
        }
    )
    noaa_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?" + query
    tide_payload = get_json(noaa_url)
    accessed_at = datetime.now(timezone.utc).date().isoformat()
    tides = {item["t"]: float(item["v"]) for item in tide_payload["predictions"]}
    values = list(monthly.values())
    min_teu, max_teu = min(values), max(values)
    rows = []
    current = begin
    while current <= end:
        month_key = current.strftime("%Y-%m")
        monthly_teu = monthly[month_key]
        throughput_scale = (monthly_teu - min_teu) / max(1.0, max_teu - min_teu)
        hour = current.hour
        weekday = current.weekday()
        tide = tides[current.strftime("%Y-%m-%d %H:%M")]
        # Explicit deterministic derivatives for exercising the adapter/env.
        daily_wave = 0.55 + 0.30 * math.sin(2 * math.pi * (hour - 6) / 24)
        shift_wave = 0.12 * math.sin(2 * math.pi * hour / 8)
        weekend_factor = 0.90 if weekday >= 5 else 1.0
        throughput_teu = max(40.0, monthly_teu / (30.4375 * 24) * weekend_factor * (daily_wave + 0.55))
        vessel_arrivals = max(1.0, throughput_teu / 1150.0 + 0.35 * abs(tide))
        base_load_kw = 1850.0 + 950.0 * throughput_scale + 410.0 * daily_wave + 55.0 * tide
        price = 0.72 if hour < 7 else (1.34 if 17 <= hour < 22 else 1.02)
        carbon = 0.46 + 0.05 * math.sin(2 * math.pi * (hour + 2) / 24)
        ambient = 27.0 + 4.5 * math.sin(2 * math.pi * (hour - 8) / 24)
        rows.append(
            {
                "timestamp": current.isoformat().replace("+00:00", "Z"),
                "base_load_kw": round(base_load_kw, 4),
                "throughput_teu": round(throughput_teu, 4),
                "vessel_arrivals": round(vessel_arrivals, 4),
                "tide_m": round(tide, 4),
                "price_per_kwh": round(price, 4),
                "carbon_kg_per_kwh": round(carbon, 6),
                "ambient_c": round(ambient, 4),
            }
        )
        current += timedelta(hours=1)
    metadata = {
        "title": "Public port operations driver example v1",
        "created_at": utc_now(),
        "provenance_type": "public_official_inputs_plus_deterministic_engineering_derivatives",
        "owner": "port-dt-multi maintainers",
        "timezone": "UTC",
        "intended_use": "integration testing, reproducible RL research, and port data-adapter validation",
        "license": "Singapore Open Data Licence and US public-domain NOAA data; code MIT",
        "official_source_columns": ["container_throughput", "tide_m"],
        "derived_columns": ["base_load_kw", "throughput_teu", "vessel_arrivals", "price_per_kwh", "carbon_kg_per_kwh", "ambient_c"],
        "sources": [
            {
                "publisher": "Maritime and Port Authority of Singapore",
                "dataset_id": MPA_DATASET_ID,
                "accessed_at": accessed_at,
                "license_url": "https://data.gov.sg/open-data-licence",
                "url": MPA_URL,
            },
            {
                "publisher": "NOAA CO-OPS",
                "station": NOAA_STATION,
                "accessed_at": accessed_at,
                "terms_url": "https://tidesandcurrents.noaa.gov/disclaimers.html",
                "url": noaa_url,
            },
        ],
        "warning": "This is an integration/reproducibility dataset, not a production telemetry export. Replace it with a mapped port dataset for deployment.",
    }
    result = write_canonical_rows("public_port_ops_v1", rows, metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
