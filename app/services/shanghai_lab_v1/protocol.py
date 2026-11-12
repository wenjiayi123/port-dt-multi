"""Transport, independent resource checks, and an auditable energy ledger."""

from __future__ import annotations

import hashlib
import json
import math

from .data import digest, validate_case


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON"))
    )


def packed(value, prefix):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {prefix + "_json": text, prefix + "_transport_sha256": hashlib.sha256(text.encode()).hexdigest()}


def unpack(payload, prefix):
    text = payload.get(prefix + "_json")
    if not isinstance(text, str) or not 1 <= len(text.encode()) <= 200000:
        raise ValueError("missing or oversized transport: " + prefix)
    if payload.get(prefix + "_transport_sha256") != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError("transport hash mismatch: " + prefix)
    return strict_json(text)


def inputs(payload):
    case = unpack(payload, "snapshot")
    validate_case(case, require_usable=True)
    return case


def validate_plan(case, plan):
    validate_case(case, require_usable=True)
    if plan.get("source_snapshot_sha256") != case["snapshot_sha256"] or plan.get("production_authority") is not False:
        raise ValueError("plan snapshot or authority mismatch")
    jobs = {j["id"]: j for j in case["jobs"]}
    rows = plan.get("scheduled", [])
    if len(rows) != len(jobs) or {r["job_id"] for r in rows} != set(jobs):
        raise ValueError("partial or duplicate schedule")
    berth_occupancy = [[0] * 24 for _ in range(case["resources"]["berths"])]
    cranes, shore, starts, completed = [0] * 24, [0] * 24, [0] * 24, [0] * 24
    wait, late = 0, 0
    for row in rows:
        job = jobs[row["job_id"]]
        start, end, berth = row["start_hour"], row["end_hour"], row["berth"]
        if (
            any(type(v) is not int for v in (start, end, berth))
            or not job["arrival_hour"] <= start < end <= 24
            or end - start != job["duration_hours"]
            or not 0 <= berth < len(berth_occupancy)
        ):
            raise ValueError("invalid task timing or berth")
        if not case["resources"]["channel_open"][start]:
            raise ValueError("channel closure violated")
        starts[start] += 1
        completed[end - 1] += job["teu"]
        wait += start - job["arrival_hour"]
        late += max(0, end - job["deadline_hour"])
        for hour in range(start, end):
            if berth >= case["resources"]["berth_capacity"][hour]:
                raise ValueError("unavailable berth used")
            berth_occupancy[berth][hour] += 1
            cranes[hour] += job["cranes"]
            shore[hour] += job["shore_kw"]
    for h in range(24):
        r = case["resources"]
        if (
            any(b[h] > 1 for b in berth_occupancy)
            or cranes[h] > r["cranes"]
            or starts[h] > min(r["channel_concurrency"], r["pilots"], r["tugs"] // 2)
        ):
            raise ValueError("resource contention")
    loads = [case["energy"]["base_kw"][h] + shore[h] + cranes[h] * case["energy"]["crane_kw"] for h in range(24)]
    return {
        "feasible": True,
        "wait_vessel_hours": wait,
        "late_vessel_hours": late,
        "processed_teu": sum(completed),
        "shore_kwh": sum(shore),
        "crane_hours": sum(cranes),
        "load_kw": loads,
        "shore_kw": shore,
        "cranes": cranes,
        "starts": starts,
        "berth_occupancy": berth_occupancy,
        "completed_teu": completed,
        "assumptions": "Hourly slot model; berth/pilot/tug/channel/crane constraints; no vessel dynamics or field clearance",
    }


def energy_ledger(case, plan, actions=None):
    operations = validate_plan(case, plan)
    b = case["battery"]
    stored = b["capacity_kwh"] * b["initial_soc"]
    initial = stored
    rows = []
    actions = actions or [0.0] * 24  # positive AC charge, negative AC discharge
    if len(actions) != 24:
        raise ValueError("incomplete energy actions")
    for h, action in enumerate(actions):
        if type(action) not in (int, float) or not math.isfinite(action) or abs(action) > b["power_kw"] + 1e-7:
            raise ValueError("battery action outside physical bound")
        charge, discharge = max(0, action), max(0, -action)
        stored += charge * b["efficiency"] - discharge / b["efficiency"]
        if not b["min_soc"] * b["capacity_kwh"] - 1e-6 <= stored <= b["max_soc"] * b["capacity_kwh"] + 1e-6:
            raise ValueError("battery state outside physical bound")
        load, pv = operations["load_kw"][h], case["energy"]["pv_kw"][h]
        curtailment = max(0.0, pv + discharge - load - charge)
        grid = load + charge - discharge - pv + curtailment
        if grid < -1e-6 or grid > case["energy"]["grid_limit_kw"][h] + 1e-6:
            raise ValueError("grid constraint violated")
        cost = grid * case["energy"]["price_cny_kwh"][h]
        wear = (charge + discharge) * b["wear_cny_kwh"]
        rows.append(
            {
                "hour": h,
                "load_kw": load,
                "pv_kw": pv,
                "charge_kw": charge,
                "discharge_kw": discharge,
                "grid_kw": grid,
                "curtailment_kw": curtailment,
                "stored_kwh": stored,
                "electricity_cny": cost,
                "wear_cny": wear,
                "carbon_kg": grid * case["energy"]["grid_carbon_kg_kwh"],
            }
        )
    if abs(stored - initial) > 1e-5:
        raise ValueError("terminal state of charge must equal initial inventory")
    totals = {key: sum(r[key] for r in rows) for key in ["electricity_cny", "wear_cny", "carbon_kg"]}
    totals.update(
        {
            "grid_kwh": sum(r["grid_kw"] for r in rows),
            "battery_charge_kwh": sum(r["charge_kw"] for r in rows),
            "battery_discharge_kwh": sum(r["discharge_kw"] for r in rows),
            "peak_grid_kw": max(r["grid_kw"] for r in rows),
            "delay_cny": operations["late_vessel_hours"] * case["delay_cny_vessel_hour"],
            "terminal_soc_difference_kwh": stored - initial,
        }
    )
    totals["total_cny"] = totals["electricity_cny"] + totals["wear_cny"] + totals["delay_cny"]
    return {
        "schema": "shanghai-energy-ledger.v1",
        "source_snapshot_sha256": case["snapshot_sha256"],
        "plan_sha256": digest(plan),
        "operations": operations,
        "hourly": rows,
        "totals": totals,
        "production_authority": False,
        "evidence_class": "simulation_output",
    }


def verify_energy(case, plan, result):
    if result.get("production_authority") is not False or result.get("evidence_class") != "simulation_output":
        raise ValueError("energy authority or evidence class mismatch")
    rows = result.get("hourly", [])
    if len(rows) != 24:
        raise ValueError("missing hourly energy ledger")
    rebuilt = energy_ledger(case, plan, [r["charge_kw"] - r["discharge_kw"] for r in rows])
    if (
        result.get("source_snapshot_sha256") != rebuilt["source_snapshot_sha256"]
        or result.get("plan_sha256") != rebuilt["plan_sha256"]
    ):
        raise ValueError("energy plan or snapshot mismatch")
    for i, row in enumerate(rebuilt["hourly"]):
        for key, value in row.items():
            given = rows[i].get(key)
            if type(given) not in (int, float) or not math.isfinite(given) or abs(given - value) > 1e-5:
                raise ValueError("energy balance mismatch: " + key)
    for key, value in rebuilt["totals"].items():
        given = result["totals"].get(key)
        if type(given) not in (int, float) or not math.isfinite(given) or abs(value - given) > 1e-5:
            raise ValueError("energy total mismatch: " + key)
    return rebuilt
