"""Verified public inputs and explicit engineering supplementation for one port."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(os.getenv("SHANGHAI_LAB_DATA_DIR", str(ROOT / "data/public/shanghai_lab_v1_20261031")))
SCENARIOS = ("normal", "arrival_surge", "berth_outage", "grid_derating", "weather_closure", "communications_loss")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_dataset(folder=FOLDER):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest.get("schema") != "shanghai-public-lab-dataset.v1"
        or manifest.get("port_id") != "CNSHA"
        or manifest.get("production_authority") is not False
    ):
        raise ValueError("Shanghai dataset identity or authority mismatch")
    for name, entry in manifest["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or file_sha(path) != entry["sha256"]:
            raise ValueError("frozen source integrity failed: " + name)
    daily = list(csv.DictReader((folder / "shanghai_daily.csv").open()))
    if len(daily) != 2557 or len({r["date"] for r in daily}) != len(daily):
        raise ValueError("daily coverage or uniqueness failed")
    first = date(2019, 1, 1)
    for i, row in enumerate(daily):
        if row["date"] != (first + timedelta(days=i)).isoformat() or row["portid"] != "port1188":
            raise ValueError("daily continuity or port binding failed")
        for key in ["portcalls", "portcalls_container"]:
            value = float(row[key])
            if not math.isfinite(value) or value < 0 or value != int(value):
                raise ValueError("invalid provider count")
    weather = list(csv.DictReader((folder / "yangshan_weather_2025.csv").open()))
    expected = datetime(2025, 1, 1)
    if len(weather) != 8760:
        raise ValueError("weather coverage failed")
    for i, row in enumerate(weather):
        if datetime.fromisoformat(row["time"]) != expected + timedelta(hours=i):
            raise ValueError("weather continuity failed")
        for field in ["temperature_2m", "wind_speed_10m", "surface_pressure", "shortwave_radiation"]:
            if not row[field] or not math.isfinite(float(row[field])):
                raise ValueError("missing/nonfinite weather; no silent imputation")
    return manifest, daily, weather, json.loads((folder / "assumptions.json").read_text())


def _field(unit, evidence, source, replacement, **extra):
    return {
        "unit": unit,
        "evidence_class": evidence,
        "source_id": source,
        "live_replacement": replacement,
        "null_policy": "unknown_or_unavailable_is_null; zero_is_valid",
        **extra,
    }


def contract():
    E = "declared_engineering_assumption"
    return {
        "schema": "shanghai-unified-trial.v1",
        "port_id": "CNSHA",
        "timezone": "Asia/Shanghai",
        "horizon_hours": 24,
        "scenario_ids": list(SCENARIOS),
        "required_context": [
            "trial_id",
            "port_id",
            "dataset_id",
            "dataset_sha256",
            "event_time",
            "revision",
            "snapshot_sha256",
        ],
        "field_contracts": {
            "public_context.container_calls": _field(
                "calls/provider_day",
                "public_provider_derived",
                "imf_portwatch_shanghai",
                "AIS/VTS port-call events; geographic boundary reconciliation",
            ),
            "public_context.annual_container_teu": _field(
                "TEU/year", "official_aggregate", "shanghai_2025_official", "official annual statistics; context only"
            ),
            "weather[].temperature_c": _field(
                "degC",
                "public_reanalysis",
                "yangshan_era5_2025",
                "approved local weather observation",
                grid_not_sensor=True,
            ),
            "weather[].wind_ms": _field(
                "m/s",
                "public_reanalysis",
                "yangshan_era5_2025",
                "approved local weather observation",
                grid_not_sensor=True,
            ),
            "weather[].radiation_wm2": _field(
                "W/m2", "public_reanalysis", "yangshan_era5_2025", "local irradiance sensor", grid_not_sensor=True
            ),
            "jobs[].arrival_hour": _field(
                "h since event_time", E, "scenario_recipe.v1", "TOS planned ETA and revision history"
            ),
            "jobs[].duration_hours": _field(
                "h", E, "scenario_recipe.v1", "work quantity and calibrated equipment productivity"
            ),
            "jobs[].deadline_hour": _field(
                "h since event_time", E, "scenario_recipe.v1", "contractual service deadline"
            ),
            "jobs[].cranes": _field("cranes", E, "scenario_recipe.v1", "TOS job-resource requirements"),
            "jobs[].shore_kw": _field("kW", E, "scenario_recipe.v1", "vessel shore-power request and metering"),
            "resources": _field(
                "named capacities; see assumptions",
                E,
                "assumptions.json",
                "operator-approved inventory and availability events",
            ),
            "energy.base_kw": _field(
                "kW", "simulation_output", "scenario_recipe.v1", "synchronized revenue and sub-meter observations"
            ),
            "energy.pv_kw": _field(
                "kW",
                "simulation_output",
                "scenario_recipe.v1",
                "PV inverter telemetry",
                formula="min(PV peak, irradiance/1000*PV peak*0.85)",
            ),
            "energy.price_cny_kwh": _field(
                "CNY/kWh",
                E,
                "assumptions.json",
                "site retail/market contract; current prices are not Shanghai tariff quotations",
            ),
            "energy.grid_carbon_kg_kwh": _field(
                "kgCO2e/kWh", E, "assumptions.json", "applicable approved accounting factors"
            ),
            "battery": _field(
                "kWh,kW,fraction,CNY/kWh", E, "assumptions.json", "BMS/EMS parameters with site calibration"
            ),
        },
        "live_envelope": {
            "required": [
                "source_system",
                "source_record_id",
                "asset_id",
                "event_time",
                "received_at",
                "revision",
                "value",
                "unit",
                "quality",
                "evidence_class",
            ],
            "checks": [
                "port/asset mapping",
                "units",
                "timezone",
                "quality",
                "freshness",
                "revisions",
                "reconciliation",
                "physical constraints",
            ],
            "note": "Same business contract supports new adapters; field calibration, identity and acceptance remain required.",
        },
        "production_authority": False,
    }


def make_case(day="2025-10-15", scenario="normal", seed=17, *, dataset=None):
    if scenario not in SCENARIOS or type(seed) is not int or not 0 <= seed <= 100000:
        raise ValueError("invalid scenario or seed")
    day_value = date.fromisoformat(day)
    if day_value.year != 2025:
        raise ValueError("this frozen trial supports 2025 only")
    manifest, daily, weather, assumptions = dataset or verified_dataset()
    observed = next(r for r in daily if r["date"] == day)
    normalizer = mean(float(r["portcalls_container"]) for r in daily if r["date"] < "2024-01-01")
    ratio = float(observed["portcalls_container"]) / normalizer
    day_weather = [r for r in weather if r["time"].startswith(day)]
    if len(day_weather) != 24:
        raise ValueError("incomplete weather day")
    n_jobs = max(6, min(12, round(8 * ratio))) + (3 if scenario == "arrival_surge" else 0)
    jobs = []
    for i in range(n_jobs):
        perturbation = int(hashlib.sha256(f"{seed}:{day}:{i}".encode()).hexdigest()[:8], 16)
        arrival = int(i * 12 / n_jobs)
        duration = 2 + perturbation % 3
        jobs.append(
            {
                "id": f"SIM-CNSHA-{day.replace('-', '')}-{seed}-{i + 1:02}",
                "entity_class": "engineered_vessel_call_not_real_identity",
                "arrival_hour": arrival,
                "duration_hours": duration,
                "deadline_hour": min(24, arrival + duration + 7),
                "cranes": 2 + (perturbation // 3) % 2,
                "shore_kw": 300 + (perturbation // 11) % 5 * 100,
                "teu": duration * 2 * 30,
                "priority": 1 + (perturbation // 19) % 3,
            }
        )
    resources = dict(assumptions["resources"])
    resources["berth_capacity"] = [
        resources["berths"] - (1 if scenario == "berth_outage" and 6 <= h < 14 else 0) for h in range(24)
    ]
    resources["channel_open"] = [not (scenario == "weather_closure" and 6 <= h < 10) for h in range(24)]
    equip, econ = assumptions["equipment"], assumptions["economics"]
    weather_values = [
        {
            "hour": i,
            "temperature_c": float(r["temperature_2m"]),
            "wind_ms": float(r["wind_speed_10m"]),
            "radiation_wm2": float(r["shortwave_radiation"]),
        }
        for i, r in enumerate(day_weather)
    ]
    prices = [
        econ["off_peak_cny_per_kwh"]
        if h < 6 or h >= 22
        else econ["peak_cny_per_kwh"]
        if 8 <= h < 11 or 18 <= h < 21
        else econ["shoulder_cny_per_kwh"]
        for h in range(24)
    ]
    body = {
        "schema": "shanghai-unified-trial.v1",
        "trial_id": f"sha-v1-{day}-{scenario}-{seed}",
        "port_id": "CNSHA",
        "site_id": "CNSHA-LAB-YANGSHAN",
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": digest(manifest),
        "event_time": day + "T00:00:00+08:00",
        "revision": 1,
        "scenario": scenario,
        "seed": seed,
        "data_mode": "retrospective_public_calibrated_engineering_trial",
        "quality": "disconnected" if scenario == "communications_loss" else "valid_for_engineering_trial",
        "public_context": {
            "provider_day": day,
            "container_calls": int(observed["portcalls_container"]),
            "annual_container_teu": 55063000,
            "annual_usage": "context only; published in 2026; not input to historical forecasting",
            "reference_calls_mean_2019_2023": normalizer,
        },
        "jobs": jobs,
        "resources": resources,
        "weather": weather_values,
        "energy": {
            "base_kw": [
                round(3200 + 700 * min(1.7, ratio) + max(0, r["temperature_c"] - 20) * 45, 4) for r in weather_values
            ],
            "pv_kw": [
                round(min(equip["pv_peak_kw"], max(0, r["radiation_wm2"]) / 1000 * equip["pv_peak_kw"] * 0.85), 4)
                for r in weather_values
            ],
            "price_cny_kwh": prices,
            "grid_limit_kw": [
                equip["grid_limit_kw"] * (0.60 if scenario == "grid_derating" and 8 <= h < 20 else 1) for h in range(24)
            ],
            "grid_carbon_kg_kwh": econ["grid_carbon_kg_per_kwh"],
            "crane_kw": 160,
        },
        "battery": {
            "capacity_kwh": equip["battery_capacity_kwh"],
            "power_kw": equip["battery_power_kw"],
            "efficiency": equip["battery_efficiency"],
            "initial_soc": equip["battery_initial_soc"],
            "min_soc": equip["battery_min_soc"],
            "max_soc": equip["battery_max_soc"],
            "wear_cny_kwh": econ["battery_wear_cny_per_kwh"],
        },
        "delay_cny_vessel_hour": econ["delay_cny_per_vessel_hour"],
        "recipe": {
            "version": "scenario_recipe.v1",
            "job_count": "clip(round(8*container_calls/2019-2023 mean),6,12); surge adds3",
            "identity": "all individual jobs and resources are declared engineering supplements",
            "weather_usage": "realized reanalysis drives exogenous replay; an online controller must not read future weather",
            "official_annual_usage": "context only; no conversion from annual totals to measured hourly telemetry",
        },
        "contract": contract(),
        "production_authority": False,
    }
    body["snapshot_sha256"] = digest(body)
    validate_case(body)
    return body


def validate_case(case, *, require_usable=False):
    if (
        case.get("schema") != "shanghai-unified-trial.v1"
        or case.get("port_id") != "CNSHA"
        or case.get("production_authority") is not False
    ):
        raise ValueError("trial schema, port or authority mismatch")
    if case.get("snapshot_sha256") != digest({k: v for k, v in case.items() if k != "snapshot_sha256"}):
        raise ValueError("trial content hash mismatch")
    if case.get("contract") != contract():
        raise ValueError("field units or provenance contract mismatch")
    if datetime.fromisoformat(case["event_time"]).utcoffset() != timedelta(hours=8) or case["revision"] != 1:
        raise ValueError("clock or revision mismatch")
    if require_usable and case["quality"] != "valid_for_engineering_trial":
        raise ValueError("source quality prevents decision")
    if not 1 <= len(case["jobs"]) <= 30 or len({j["id"] for j in case["jobs"]}) != len(case["jobs"]):
        raise ValueError("duplicate job identity")
    for job in case["jobs"]:
        if (
            any(
                type(job[k]) is not int or job[k] < 0
                for k in ("arrival_hour", "duration_hours", "deadline_hour", "cranes", "shore_kw", "teu", "priority")
            )
            or job["cranes"] < 1
            or job["teu"] < 1
            or job["priority"] < 1
        ):
            raise ValueError("invalid engineered job quantities")
        if (
            not job["id"].startswith("SIM-CNSHA-")
            or not 0 <= job["arrival_hour"] < job["deadline_hour"] <= 24
            or not 1 <= job["duration_hours"] <= 8
        ):
            raise ValueError("invalid engineered job")
    for key in ["base_kw", "pv_kw", "price_cny_kwh", "grid_limit_kw"]:
        values = case["energy"][key]
        if len(values) != 24 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid hourly energy input: " + key)
    if (
        len(case["weather"]) != 24
        or len(case["resources"]["berth_capacity"]) != 24
        or len(case["resources"]["channel_open"]) != 24
    ):
        raise ValueError("incomplete shared horizon")
    r = case["resources"]
    if any(
        type(r[k]) is not int or not 1 <= r[k] <= 100
        for k in ("berths", "cranes", "pilots", "tugs", "channel_concurrency")
    ):
        raise ValueError("invalid resource quantities")
    if any(type(v) is not int or not 0 <= v <= r["berths"] for v in r["berth_capacity"]) or any(
        type(v) is not bool for v in r["channel_open"]
    ):
        raise ValueError("invalid resource availability")
    for h, weather in enumerate(case["weather"]):
        if (
            weather["hour"] != h
            or any(
                type(weather[k]) not in (int, float) or not math.isfinite(weather[k])
                for k in ("temperature_c", "wind_ms", "radiation_wm2")
            )
            or weather["wind_ms"] < 0
            or weather["radiation_wm2"] < 0
        ):
            raise ValueError("invalid weather observation")
    b = case["battery"]
    numbers = [
        b[k] for k in ("capacity_kwh", "power_kw", "efficiency", "min_soc", "max_soc", "initial_soc", "wear_cny_kwh")
    ]
    numbers += [case["energy"]["grid_carbon_kg_kwh"], case["energy"]["crane_kw"], case["delay_cny_vessel_hour"]]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in numbers):
        raise ValueError("invalid physical/economic coefficient")
    if not (
        0 < b["efficiency"] <= 1
        and 0 <= b["min_soc"] <= b["initial_soc"] <= b["max_soc"] <= 1
        and b["power_kw"] > 0
        and b["capacity_kwh"] > 0
    ):
        raise ValueError("invalid battery contract")
    return True
