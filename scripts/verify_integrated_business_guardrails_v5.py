"""Replay deterministic V5 business guardrails over every chronological data row."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.rl_training.business_guardrails import (
    assess_integrated_business_constraints,
)
from app.services.rl_training.datasets import (
    FACTOR_COLUMNS,
    NUMERIC_COLUMNS,
    PORT_WIDE_COLUMNS,
    load_port_dataset,
)
from app.services.rl_training.profiles import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "evidence/v5/deterministic_guardrails"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def state_for(dataset: Any, index: int) -> dict[str, float]:
    state = {
        name: float(dataset.values[index, position])
        for position, name in enumerate(NUMERIC_COLUMNS)
    }
    state.update(
        {
            name: float(dataset.factor_values[index, position])
            for position, name in enumerate(FACTOR_COLUMNS)
            if dataset.factor_availability[index, position] > 0.5
        }
    )
    state.update(
        {
            name: float(dataset.port_wide_values[index, position])
            for position, name in enumerate(PORT_WIDE_COLUMNS)
            if dataset.port_wide_availability[index, position] > 0.5
        }
    )
    return state


def main() -> None:
    dataset = load_port_dataset("public_cn_sha_integrated_scenario_v5")
    profile = load_profile("cn_sha_integrated_scenario_v5")
    control = {
        "bess_kw": 0.0,
        "flexible_load_command": 0.0,
        "yard_flow_command": 0.0,
        "reefer_service_ratio": 0.5,
        "maintenance_reserve_ratio": 0.5,
        "shore_power_allocation_ratio": 0.5,
        "marine_service_allocation_ratio": 0.5,
    }
    counts: Counter[str] = Counter()
    violation_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    clearance_values = []
    representative: dict[str, Any] = {}
    for index in range(dataset.rows):
        result = assess_integrated_business_constraints(
            state=state_for(dataset, index),
            decoded_control=control,
            demand_cap_kw=profile["assets"]["demand_cap_kw"],
            port_profile=profile,
        )
        counts[result["status"]] += 1
        clearance_values.append(result["estimates"]["under_keel_clearance_m"])
        for item in result["violations"]:
            violation_counts[item["code"]] += 1
        for item in result["warnings"]:
            warning_counts[item["code"]] += 1
        representative.setdefault(
            result["status"],
            {
                "row_index": index,
                "timestamp": dataset.timestamps[index],
                "result": result,
            },
        )

    safe_state = state_for(dataset, int(max(range(dataset.rows), key=lambda index: dataset.values[index, 3])))
    safe_state.update(
        tide_m=3.0,
        channel_chart_depth_m=18.0,
        planning_vessel_draft_m=13.0,
        squat_allowance_m=0.3,
        yard_occupancy_ratio=0.70,
        dangerous_goods_workload_ratio=0.03,
        reefer_temperature_risk_ratio=0.10,
        equipment_failure_risk_ratio=0.10,
        base_load_kw=20000.0,
        shore_power_demand_kw=3000.0,
        shore_power_connection_ratio=0.70,
        closure_flag=0.0,
        wind_speed_mps=5.0,
        wave_height_m=0.5,
        pilot_tug_availability_ratio=0.9,
        channel_capacity_ratio=0.8,
        forecast_uncertainty_ratio=0.2,
    )
    challenge_specs = {
        "safe_reference": ({}, {}),
        "under_keel_clearance": ({"tide_m": -6.0}, {}),
        "channel_closure": ({"closure_flag": 1.0}, {}),
        "weather_stop": ({"wind_speed_mps": 25.0}, {}),
        "dangerous_goods_yard": (
            {
                "yard_occupancy_ratio": 0.92,
                "dangerous_goods_workload_ratio": 0.10,
            },
            {"yard_flow_command": 0.25},
        ),
        "reefer_safety_reserve": (
            {"reefer_temperature_risk_ratio": 0.85},
            {"reefer_service_ratio": 0.1},
        ),
        "maintenance_safety_reserve": (
            {"equipment_failure_risk_ratio": 0.80},
            {"maintenance_reserve_ratio": 0.1},
        ),
        "integrated_demand_cap": (
            {"base_load_kw": 35500.0, "shore_power_demand_kw": 8000.0},
            {"shore_power_allocation_ratio": 1.0},
        ),
        "missing_hydrographic_input": (
            {"channel_chart_depth_m": None},
            {},
        ),
    }
    challenges = {}
    for name, (state_changes, control_changes) in challenge_specs.items():
        challenge_state = deepcopy(safe_state)
        challenge_state.update(state_changes)
        challenge_control = {**control, **control_changes}
        challenges[name] = assess_integrated_business_constraints(
            state=challenge_state,
            decoded_control=challenge_control,
            demand_cap_kw=profile["assets"]["demand_cap_kw"],
            port_profile=profile,
        )
    challenge_checks = {
        "safe_reference_not_blocked": challenges["safe_reference"]["status"]
        != "blocked",
        **{
            f"{name}_blocked": result["status"] == "blocked"
            and result["dispatch_allowed"] is False
            for name, result in challenges.items()
            if name != "safe_reference"
        },
    }
    passed = all(challenge_checks.values())
    run_id = "deterministic-guardrails-v5-" + datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    run_dir = EVIDENCE_ROOT / "runs" / run_id
    report = {
        "schema": "port-dt-deterministic-business-guardrails-evidence.v1",
        "run_id": run_id,
        "status": "PASS" if passed else "FAIL",
        "generated_at": now(),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "artifact": str(dataset.path),
            "sha256": dataset.fingerprint,
            "rows_replayed": dataset.rows,
            "independent_source_observations": int(
                dataset.metadata.get("independent_source_observations") or 0
            ),
            "public_observation_boundary": dataset.metadata.get("warning"),
        },
        "chronological_replay": {
            "status_counts": dict(sorted(counts.items())),
            "violation_code_counts": dict(sorted(violation_counts.items())),
            "warning_code_counts": dict(sorted(warning_counts.items())),
            "under_keel_clearance_m": {
                "minimum": min(clearance_values),
                "maximum": max(clearance_values),
                "blocked_window_count": violation_counts[
                    "UNDER_KEEL_CLEARANCE"
                ],
            },
            "representative_rows": representative,
            "interpretation": "Blocked rows are the intended fail-closed result, not failed software tests.",
        },
        "challenge_suite": {
            "checks": challenge_checks,
            "passed": passed,
            "results": challenges,
        },
        "non_rl_responsibilities": [
            "source completeness and unit validation",
            "maritime and customs release authority",
            "weather and channel closure",
            "under-keel-clearance acceptance",
            "pilot tug and channel capacity confirmation",
            "dangerous-goods segregation",
            "reefer minimum service reserve",
            "maintenance minimum safety reserve",
            "integrated electrical demand cap",
            "human approval execution rollback and audit",
        ],
        "integration_boundary": {
            "recommendation_only": True,
            "dispatch_allowed": False,
            "production_authority": False,
            "next_stage": "existing end-to-end coordination, independent approval and actuator interlock services",
        },
    }
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    latest = {
        "schema": "port-dt-deterministic-business-guardrails-latest.v1",
        "run_id": run_id,
        "status": report["status"],
        "report_path": str(report_path.relative_to(ROOT)),
        "report_sha256": sha256(report_path),
        "dataset_sha256": dataset.fingerprint,
        "rows_replayed": dataset.rows,
        "production_authority": False,
        "updated_at": now(),
    }
    write_json(EVIDENCE_ROOT / "latest.json", latest)
    history_path = EVIDENCE_ROOT / "history_index.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(latest, ensure_ascii=False) + "\n")
    print(json.dumps(latest | {"checks": challenge_checks}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
