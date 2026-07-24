from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

from app.services.rl_training.statistics import bootstrap_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/business_kpi_benchmark_v1.json"
DEFAULT_REPORT = ROOT / "data/rl/business_kpi_benchmark_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line, raw in enumerate(csv.DictReader(handle), 2):
            try:
                rows.append(
                    {
                        "timestamp": str(raw["timestamp"]),
                        "base_load_kw": float(raw["base_load_kw"]),
                        "throughput_teu": float(raw["throughput_teu"]),
                        "vessel_arrivals": float(raw["vessel_arrivals"]),
                        "tide_m": float(raw["tide_m"]),
                        "price_per_kwh": float(raw["price_per_kwh"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid canonical row at line {line}") from exc
    if len(rows) < 240:
        raise ValueError("business benchmark requires at least 240 chronological rows")
    timestamps = [row["timestamp"] for row in rows]
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        raise ValueError("dataset timestamps must be strictly increasing")
    return rows


def _weighted_mean(values: Iterable[tuple[float, float]]) -> float:
    pairs = list(values)
    denominator = sum(weight for _value, weight in pairs)
    return sum(value * weight for value, weight in pairs) / max(denominator, 1e-9)


def _settled_energy_cost(
    rows: list[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[float, float, list[dict[str, Any]]]:
    baseline_cost = sum(
        float(row["base_load_kw"]) * float(row["price_per_kwh"])
        for row in rows
    )
    capacity = float(policy["bess_capacity_kwh"])
    power_limit = float(policy["bess_power_kw"])
    initial_soc = float(policy["bess_initial_soc"])
    minimum_energy = capacity * float(policy["bess_min_soc"])
    maximum_energy = capacity * float(policy["bess_max_soc"])
    efficiency = float(policy["bess_roundtrip_leg_efficiency"])
    shift_fraction = float(policy["flexible_load_shift_fraction"])
    low_price = float(policy["low_price_threshold"])
    high_price = float(policy["high_price_threshold"])
    reference_price = float(policy["terminal_energy_reference_price"])
    energy = capacity * initial_soc
    deferred_kwh = 0.0
    candidate_cost = 0.0
    hourly: list[dict[str, Any]] = []
    for row in rows:
        load = float(row["base_load_kw"])
        price = float(row["price_per_kwh"])
        grid_load = load
        shifted_kwh = 0.0
        bess_kw = 0.0
        if price >= high_price:
            shifted_kwh = load * shift_fraction
            grid_load -= shifted_kwh
            deferred_kwh += shifted_kwh
            discharge_kw = min(
                power_limit,
                grid_load,
                max(0.0, (energy - minimum_energy) * efficiency),
            )
            grid_load -= discharge_kw
            energy -= discharge_kw / efficiency
            bess_kw = -discharge_kw
        elif price <= low_price:
            restored_kwh = min(deferred_kwh, load * shift_fraction)
            grid_load += restored_kwh
            deferred_kwh -= restored_kwh
            charge_kw = min(
                power_limit,
                max(0.0, (maximum_energy - energy) / efficiency),
            )
            grid_load += charge_kw
            energy += charge_kw * efficiency
            bess_kw = charge_kw
            shifted_kwh = -restored_kwh
        hourly_cost = max(0.0, grid_load) * price
        candidate_cost += hourly_cost
        hourly.append(
            {
                "timestamp": row["timestamp"],
                "baseline_cost": load * price,
                "candidate_cost": hourly_cost,
                "bess_kw": bess_kw,
                "deferred_load_kwh": deferred_kwh,
                "soc": energy / capacity,
                "shifted_load_kwh": shifted_kwh,
            }
        )
    terminal_energy_delta = capacity * initial_soc - energy
    terminal_settlement = (
        deferred_kwh + terminal_energy_delta
    ) * reference_price
    candidate_cost += terminal_settlement
    if hourly:
        hourly[-1]["candidate_cost"] += terminal_settlement
        hourly[-1]["terminal_settlement"] = terminal_settlement
        hourly[-1]["terminal_deferred_load_kwh"] = deferred_kwh
        hourly[-1]["terminal_bess_energy_delta_kwh"] = terminal_energy_delta
    return baseline_cost, candidate_cost, hourly


def _evaluate_split(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    compute_uncertainty: bool = True,
) -> dict[str, Any]:
    baseline = config["baseline"]
    policy = config["coordinated_policy"]
    crane_rate = float(policy["crane_productivity_teu_per_hour"])
    cranes_per_berth = int(policy["cranes_per_berth"])
    baseline_buffer = float(
        baseline["berth_uncertainty_buffer_hours_per_vessel"]
    )
    candidate_buffer = float(
        policy["berth_uncertainty_buffer_hours_per_vessel"]
    )
    productive_hours = sum(
        float(row["throughput_teu"]) / (crane_rate * cranes_per_berth)
        for row in rows
    )
    arrivals = sum(float(row["vessel_arrivals"]) for row in rows)
    baseline_utilization = productive_hours / (
        productive_hours + arrivals * baseline_buffer
    )
    candidate_utilization = productive_hours / (
        productive_hours + arrivals * candidate_buffer
    )
    train_throughput = [float(row["throughput_teu"]) for row in train_rows]
    throughput_mean = fmean(train_throughput)
    throughput_std = max(pstdev(train_throughput), 1e-9)
    wait_base = float(policy["congestion_wait_base_hours"])
    throughput_coefficient = float(
        policy["throughput_pressure_coefficient_hours"]
    )
    tide_coefficient = float(policy["tide_pressure_coefficient_hours"])
    baseline_wait = _weighted_mean(
        (
            wait_base
            + throughput_coefficient
            * max(
                0.0,
                (float(row["throughput_teu"]) - throughput_mean)
                / throughput_std,
            )
            + tide_coefficient * abs(float(row["tide_m"]))
            + baseline_buffer / 2.0,
            float(row["vessel_arrivals"]),
        )
        for row in rows
    )
    candidate_wait = _weighted_mean(
        (
            wait_base
            + throughput_coefficient
            * max(
                0.0,
                (float(row["throughput_teu"]) - throughput_mean)
                / throughput_std,
            )
            + tide_coefficient * abs(float(row["tide_m"]))
            + candidate_buffer / 2.0,
            float(row["vessel_arrivals"]),
        )
        for row in rows
    )
    rows_by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_day[str(row["timestamp"])[:10]].append(row)
    baseline_cost = 0.0
    candidate_cost = 0.0
    hourly_cost: list[dict[str, Any]] = []
    for day_rows in rows_by_day.values():
        day_baseline, day_candidate, day_hourly = _settled_energy_cost(
            day_rows,
            policy,
        )
        baseline_cost += day_baseline
        candidate_cost += day_candidate
        hourly_cost.extend(day_hourly)
    daily: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "rows": 0.0,
            "baseline_cost": 0.0,
            "candidate_cost": 0.0,
            "productive_hours": 0.0,
            "arrivals": 0.0,
            "baseline_wait_numerator": 0.0,
            "candidate_wait_numerator": 0.0,
        }
    )
    hourly_by_timestamp = {
        str(item["timestamp"]): item for item in hourly_cost
    }
    for row in rows:
        day = str(row["timestamp"])[:10]
        cost_row = hourly_by_timestamp[str(row["timestamp"])]
        common_wait = (
            wait_base
            + throughput_coefficient
            * max(
                0.0,
                (float(row["throughput_teu"]) - throughput_mean)
                / throughput_std,
            )
            + tide_coefficient * abs(float(row["tide_m"]))
        )
        weight = float(row["vessel_arrivals"])
        daily[day]["rows"] += 1.0
        daily[day]["baseline_cost"] += float(cost_row["baseline_cost"])
        daily[day]["candidate_cost"] += float(cost_row["candidate_cost"])
        daily[day]["productive_hours"] += float(row["throughput_teu"]) / (
            crane_rate * cranes_per_berth
        )
        daily[day]["arrivals"] += weight
        daily[day]["baseline_wait_numerator"] += (
            common_wait + baseline_buffer / 2.0
        ) * weight
        daily[day]["candidate_wait_numerator"] += (
            common_wait + candidate_buffer / 2.0
        ) * weight
    daily_rows: list[dict[str, Any]] = []
    for day, values in sorted(daily.items()):
        daily_productive = values["productive_hours"]
        daily_arrivals = values["arrivals"]
        daily_baseline_utilization = daily_productive / (
            daily_productive + daily_arrivals * baseline_buffer
        )
        daily_candidate_utilization = daily_productive / (
            daily_productive + daily_arrivals * candidate_buffer
        )
        daily_baseline_wait = (
            values["baseline_wait_numerator"] / max(daily_arrivals, 1e-9)
        )
        daily_candidate_wait = (
            values["candidate_wait_numerator"] / max(daily_arrivals, 1e-9)
        )
        daily_rows.append(
            {
                "date": day,
                "rows": int(values["rows"]),
                "berth_utilization_relative_improvement_percent": (
                    daily_candidate_utilization
                    / daily_baseline_utilization
                    - 1.0
                )
                * 100.0,
                "average_waiting_time_reduction_percent": (
                    1.0 - daily_candidate_wait / daily_baseline_wait
                )
                * 100.0,
                "scenario_energy_cost_reduction_percent": (
                    1.0
                    - values["candidate_cost"]
                    / max(values["baseline_cost"], 1e-9)
                )
                * 100.0,
            }
        )
    metric_values = {
        "berth_utilization_relative_improvement_percent": (
            candidate_utilization / baseline_utilization - 1.0
        )
        * 100.0,
        "average_waiting_time_reduction_percent": (
            1.0 - candidate_wait / baseline_wait
        )
        * 100.0,
        "scenario_energy_cost_reduction_percent": (
            1.0 - candidate_cost / baseline_cost
        )
        * 100.0,
    }
    complete_daily_rows = [
        row for row in daily_rows if int(row["rows"]) == 24
    ]
    uncertainty = (
        {
            metric: bootstrap_summary(
                [float(row[metric]) for row in complete_daily_rows],
                seed=20260724,
            )
            for metric in metric_values
        }
        if compute_uncertainty
        else {}
    )
    return {
        "rows": len(rows),
        "start_at": rows[0]["timestamp"],
        "end_at": rows[-1]["timestamp"],
        "baseline": {
            "berth_utilization": baseline_utilization,
            "average_waiting_hours": baseline_wait,
            "scenario_energy_cost": baseline_cost,
        },
        "coordinated_policy": {
            "berth_utilization": candidate_utilization,
            "average_waiting_hours": candidate_wait,
            "scenario_energy_cost": candidate_cost,
        },
        "improvements": metric_values,
        "uncertainty": uncertainty,
        "daily_paired_metrics": daily_rows,
        "energy_balance": {
            "terminal_settlement_included": True,
            "terminal_settlement_frequency": "daily",
            "service_energy_omitted": False,
            "throughput_changed": False,
        },
    }


def build_report(
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    dataset_path = ROOT / str(config["dataset_path"])
    rows = _read_rows(dataset_path)
    split = config["split"]
    if split["method"] == "chronological_calendar_boundaries_no_shuffle":
        train_boundary = str(split["train_end_exclusive"])
        validation_boundary = str(split["validation_end_exclusive"])
        train_end = next(
            index
            for index, row in enumerate(rows)
            if row["timestamp"] >= train_boundary
        )
        validation_end = next(
            index
            for index, row in enumerate(rows)
            if row["timestamp"] >= validation_boundary
        )
    else:
        train_end = round(len(rows) * float(split["train_ratio"]))
        validation_end = round(
            len(rows)
            * (
                float(split["train_ratio"])
                + float(split["validation_ratio"])
            )
        )
    train_rows = rows[:train_end]
    validation_rows = rows[train_end:validation_end]
    test_rows = rows[validation_end:]
    validation = _evaluate_split(validation_rows, train_rows, config)
    test = _evaluate_split(test_rows, train_rows, config)
    sensitivity_runs: list[dict[str, Any]] = []
    sensitivity = config.get("sensitivity") or {}
    for buffer_hours, bess_power_kw, shift_fraction in product(
        sensitivity.get(
            "berth_uncertainty_buffer_hours_per_vessel",
            [config["coordinated_policy"]["berth_uncertainty_buffer_hours_per_vessel"]],
        ),
        sensitivity.get(
            "bess_power_kw",
            [config["coordinated_policy"]["bess_power_kw"]],
        ),
        sensitivity.get(
            "flexible_load_shift_fraction",
            [config["coordinated_policy"]["flexible_load_shift_fraction"]],
        ),
    ):
        scenario_config = deepcopy(config)
        scenario_policy = scenario_config["coordinated_policy"]
        scenario_policy["berth_uncertainty_buffer_hours_per_vessel"] = float(
            buffer_hours
        )
        scenario_policy["bess_power_kw"] = float(bess_power_kw)
        scenario_policy["flexible_load_shift_fraction"] = float(
            shift_fraction
        )
        evaluated = _evaluate_split(
            test_rows,
            train_rows,
            scenario_config,
            compute_uncertainty=False,
        )
        sensitivity_runs.append(
            {
                "parameters": {
                    "berth_uncertainty_buffer_hours_per_vessel": float(
                        buffer_hours
                    ),
                    "bess_power_kw": float(bess_power_kw),
                    "flexible_load_shift_fraction": float(shift_fraction),
                },
                "improvements": evaluated["improvements"],
            }
        )
    rounding = config["claim_rounding"]
    resume_claims = {
        metric: round(float(value), int(rounding[metric]))
        for metric, value in test["improvements"].items()
    }
    return {
        "schema_version": config["schema_version"],
        "benchmark_id": config["benchmark_id"],
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "evidence_level": (
            "public_input_driven_fixed_digital_twin_counterfactual"
        ),
        "dataset": {
            "dataset_id": config["dataset_id"],
            "path": str(dataset_path.relative_to(ROOT)),
            "sha256": _sha256(dataset_path),
            "rows": len(rows),
            "split_method": split["method"],
            "split_sizes": {
                "train": len(train_rows),
                "validation": len(validation_rows),
                "test": len(test_rows),
            },
            "test_period": {
                "start": test_rows[0]["timestamp"],
                "end": test_rows[-1]["timestamp"],
            },
        },
        "evidence_sha256": {
            str(config_path.relative_to(ROOT)): _sha256(config_path),
            str(dataset_path.relative_to(ROOT)): _sha256(dataset_path),
            "app/services/business_benchmark.py": _sha256(Path(__file__)),
        },
        "baseline": config["baseline"],
        "coordinated_policy": config["coordinated_policy"],
        "metric_definitions": config["metric_definitions"],
        "validation": validation,
        "test": test,
        "sensitivity": {
            "predeclared_scenarios": len(sensitivity_runs),
            "parameters": sensitivity,
            "ranges": {
                metric: {
                    "min": min(
                        run["improvements"][metric]
                        for run in sensitivity_runs
                    ),
                    "max": max(
                        run["improvements"][metric]
                        for run in sensitivity_runs
                    ),
                }
                for metric in test["improvements"]
            },
            "runs": sensitivity_runs,
        },
        "resume_claims_rounded_percent": resume_claims,
        "attribution": {
            "berth_utilization_and_waiting": (
                "The reported berth and waiting differences are mechanically "
                "induced by the predeclared 4 h versus 2 h uncertainty-buffer "
                "scenario. The buffer was not learned from outcomes and the "
                "comparison is not a causal estimate."
            ),
            "energy_cost": (
                "The reported cost difference is produced by the disclosed "
                "deterministic BESS and flexible-load schedule, including "
                "daily terminal energy settlement; it is not evidence of a "
                "trained RL policy outperforming the baseline."
            ),
            "throughput": (
                "Throughput is held identical between baseline and candidate; "
                "the benchmark does not claim incremental handled volume."
            ),
        },
        "claim_text": {
            "zh": (
                "在固定公开数据驱动、参数预声明的数字孪生情景对照中，"
                "相对静态FCFS与固定能源时刻表，协调情景使泊位利用率相对提升"
                f"{resume_claims['berth_utilization_relative_improvement_percent']:.0f}%，"
                "平均待泊时间缩短"
                f"{resume_claims['average_waiting_time_reduction_percent']:.0f}%，"
                "情景用电成本降低"
                f"{resume_claims['scenario_energy_cost_reduction_percent']:.0f}%。"
            )
        },
        "evidence_boundary": config["evidence_boundary"],
        "release_gate": {
            "passed": all(
                math.isfinite(float(value))
                for value in test["improvements"].values()
            ),
            "production_claim_allowed": False,
            "measured_port_kpi": False,
            "wording_required": (
                "固定公开数据驱动、参数预声明的数字孪生情景对照，"
                "非港口实测KPI或因果效果"
            ),
        },
    }


def load_verified_report(
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    report = _load_json(report_path)
    errors: list[str] = []
    for relative, expected in report.get("evidence_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            errors.append(relative)
    if errors:
        raise ValueError(
            "business benchmark evidence changed: " + ", ".join(errors)
        )
    if report.get("release_gate", {}).get("passed") is not True:
        raise ValueError("business benchmark release gate is not PASS")
    return report
