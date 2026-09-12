#!/usr/bin/env python3
"""Train-only, perfect-foresight LP bounds for the unchanged V3 Shore+BESS assets.

This diagnostic is an optimistic bound, never a learned-policy result.  It
omits thermal/SOH nonlinearities and the V8 FIFO deferral deadline, and records
that limitation explicitly.  Historical held-out rows are not loaded into any
optimizer, feasibility comparison, normalization, or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rl_model.shore_bess.v3_environment import (  # noqa: E402
    ShoreBESSEnv,
    fixed_window_starts,
    load_config,
    load_public_dataset,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_train_slice(dataset: Any) -> slice:
    """Full official source months only; no May 2025 validation rows."""
    stop = next(
        i for i, timestamp in enumerate(dataset.timestamps)
        if timestamp >= "2025-05-01T00:00:00Z"
    )
    if stop < 168 or dataset.timestamps[stop - 1] != "2025-04-30T23:00:00Z":
        raise ValueError("Expected a complete training period through April 2025")
    return slice(0, stop)


def make_environment(dataset: Any, train_slice: slice, start: int) -> ShoreBESSEnv:
    env = ShoreBESSEnv(
        dataset, train_slice, config=load_config(), normalization_slice=train_slice,
        training=False, episode_steps=168,
    )
    env.reset(options={"start_index": start})
    return env


def solve_bound(
    dataset: Any,
    train_slice: slice,
    start: int,
    *,
    objective: str,
    carbon_guard: bool,
    bess_enabled: bool,
) -> dict[str, Any]:
    env = make_environment(dataset, train_slice, start)
    hours = env.episode_steps
    variable_count = 5 * hours + 1
    def index(block: int, t: int) -> int:
        return block * hours + t
    contexts = []
    for t in range(hours):
        env._step = t
        contexts.append(env._row_context())
    env.close()
    def column(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in contexts], dtype=np.float64)
    base = column("base_load_kw")
    price = column("price_cny_per_kwh")
    carbon_factor = column("carbon_kg_per_kwh")
    auxiliary = column("auxiliary_shore_kw")
    reserve = column("reserve_required_kw")
    # Variables: grid-side charge, discharge, signed flex, SOC, backlog, peak.
    # Constant initial SOH is an explicitly optimistic relaxation.
    available_power = env.power_kw * column("equipment_availability_ratio") * env.soh_initial
    bounds: list[tuple[float, float]] = []
    for t in range(hours):
        upper = min(available_power[t], env.hard_pcc_limit_kw - base[t])
        bounds.append((0.0, max(0.0, upper) if bess_enabled else 0.0))
    for t in range(hours):
        upper = max(0.0, available_power[t] - reserve[t])
        bounds.append((0.0, upper if bess_enabled else 0.0))
    bounds.extend((-float(x) * env.flex_limit, float(x) * env.flex_limit) for x in auxiliary)
    for t in range(hours):
        remaining = hours - t - 1
        reachable = min(
            remaining * env.power_kw * min(env.charge_eff, env.discharge_eff) / env.energy_kwh,
            0.18 * remaining / (hours - 1),
        )
        bounds.append((max(env.soc_min, env.soc_initial - reachable),
                       min(env.soc_max, env.soc_initial + reachable)))
    bounds.extend(
        (0.0, min(env.max_backlog_kwh, (hours - t - 1) * float(auxiliary[t]) * env.flex_limit))
        for t in range(hours)
    )
    # Every experiment forbids increased maximum demand.
    bounds.append((0.0, min(float(base.max()), env.hard_pcc_limit_kw)))
    inequalities: list[dict[int, float]] = []
    inequality_rhs: list[float] = []
    equalities: list[dict[int, float]] = []
    equality_rhs: list[float] = []
    for t in range(hours):
        row = {index(3, t): 1.0, index(0, t): -env.charge_eff / env.energy_kwh,
               index(1, t): 1.0 / (env.discharge_eff * env.energy_kwh)}
        if t:
            row[index(3, t - 1)] = -1.0
        equalities.append(row)
        equality_rhs.append(env.soc_initial if t == 0 else 0.0)
        row = {index(4, t): 1.0, index(2, t): 1.0}
        if t:
            row[index(4, t - 1)] = -1.0
        equalities.append(row)
        equality_rhs.append(0.0)
        inequalities.append({index(0, t): 1.0, index(1, t): -1.0,
                             index(2, t): 1.0, variable_count - 1: -1.0})
        inequality_rhs.append(-float(base[t]))
        ramp = {index(1, t): 1.0, index(0, t): -1.0}
        if t:
            ramp.update({index(1, t - 1): -1.0, index(0, t - 1): 1.0})
        inequalities.extend([ramp, {i: -v for i, v in ramp.items()}])
        inequality_rhs.extend([env.ramp_kw, env.ramp_kw])
    carbon = np.zeros(variable_count)
    carbon[:hours] = carbon_factor
    carbon[hours:2 * hours] = -carbon_factor
    carbon[2 * hours:3 * hours] = carbon_factor
    cost = np.zeros(variable_count)
    cost[:hours] = price + env.cycle_cost
    cost[hours:2 * hours] = -price + env.cycle_cost
    cost[2 * hours:3 * hours] = price
    cost[-1] = float(env.config["grid"]["demand_charge_cny_per_kw_month"]) * hours / (24.0 * 30.4375)
    if carbon_guard:
        inequalities.append({i: float(v) for i, v in enumerate(carbon) if v})
        inequality_rhs.append(0.0)
    if objective == "carbon":
        inequalities.append({i: float(v) for i, v in enumerate(cost) if v})
        inequality_rhs.append(float(cost[-1] * base.max()))

    def matrix(rows: list[dict[int, float]]) -> Any:
        result = lil_matrix((len(rows), variable_count))
        for row_number, row in enumerate(rows):
            for column_number, value in row.items():
                result[row_number, column_number] = value
        return result.tocsr()

    optimum = linprog(
        cost if objective == "cost" else carbon,
        A_ub=matrix(inequalities), b_ub=inequality_rhs,
        A_eq=matrix(equalities), b_eq=equality_rhs, bounds=bounds, method="highs",
    )
    if not optimum.success:
        raise RuntimeError(f"LP failed at train start {start}: {optimum.message}")
    solution = optimum.x
    net = base + solution[:hours] - solution[hours:2 * hours] + solution[2 * hours:3 * hours]
    baseline_cost = float(base @ price + cost[-1] * base.max())
    baseline_carbon = float(base @ carbon_factor)
    cost_saving = -float(cost @ solution - cost[-1] * base.max())
    carbon_saving = -float(carbon @ solution)
    # Replaying a perfect-foresight schedule does NOT make it causal or learned.
    replay = make_environment(dataset, train_slice, start)
    for t in range(hours):
        action = np.asarray([
            (solution[hours + t] - solution[t]) / env.power_kw,
            solution[2 * hours + t] / (auxiliary[t] * env.flex_limit),
        ], dtype=np.float32)
        replay.step(action)
    metrics = replay.totals
    replay.close()
    return {
        "start_index": start,
        "start_timestamp": dataset.timestamps[start],
        "end_timestamp": dataset.timestamps[start + hours - 1],
        "objective": objective, "carbon_non_regression_required": carbon_guard,
        "peak_non_regression_required": True, "bess_enabled": bess_enabled,
        "cost_reduction_percent": 100.0 * cost_saving / baseline_cost,
        "carbon_reduction_percent": 100.0 * carbon_saving / baseline_carbon,
        "peak_reduction_percent": 100.0 * float(base.max() - net.max()) / float(base.max()),
        "weekly_cost_saving_cny": cost_saving, "weekly_carbon_saving_kg": carbon_saving,
        "charge_kwh": float(solution[:hours].sum()),
        "discharge_kwh": float(solution[hours:2 * hours].sum()),
        "flex_shift_kwh": float(np.abs(solution[2 * hours:3 * hours]).sum() / 2.0),
        "max_backlog_kwh": float(solution[4 * hours:5 * hours].max()),
        "simultaneous_charge_discharge_kwh": float(np.minimum(solution[:hours], solution[hours:2 * hours]).sum()),
        "historical_environment_replay": metrics,
        "replay_cost_reduction_percent": 100.0 * (baseline_cost - metrics["total_cost_cny"]) / baseline_cost,
        "replay_carbon_reduction_percent": 100.0 * (baseline_carbon - metrics["carbon_kg"]) / baseline_carbon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.windows <= 20:
        parser.error("--windows must be between 1 and 20")
    dataset = load_public_dataset()
    train = grouped_train_slice(dataset)
    starts = fixed_window_starts(train.stop, 168, args.windows)
    modes = [("cost", False, True), ("cost", True, True),
             ("carbon", True, True), ("cost", True, False)]
    rows = []
    for start in starts:
        for objective, guard, bess in modes:
            row = solve_bound(dataset, train, start, objective=objective,
                              carbon_guard=guard, bess_enabled=bess)
            rows.append(row)
            print(json.dumps({k: row[k] for k in (
                "start_timestamp", "objective", "carbon_non_regression_required", "bess_enabled",
                "cost_reduction_percent", "carbon_reduction_percent", "peak_reduction_percent",
            )}), flush=True)
    sources = [Path(__file__).resolve(), ROOT / "config/shore_bess_v3.json",
               ROOT / "data/rl/datasets/public_cn_sha_hourly_v3.csv",
               ROOT / "app/services/rl_model/shore_bess/v3_environment.py"]
    generated_at = datetime.now(timezone.utc)
    output = args.output or ROOT / "evidence/v8/shore_bess/audits" / generated_at.strftime("train-feasibility-%Y%m%dT%H%M%SZ.json")
    if output.exists():
        raise FileExistsError(f"Refusing to replace prior audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "port-dt-shore-bess-v8-train-feasibility.v1",
        "generated_at": generated_at.isoformat(), "scope": "train_only_optimistic_perfect_foresight_lp_bound",
        "is_learned_policy_result": False, "production_authority": False,
        "split": {"train_rows": train.stop, "train_start": dataset.timestamps[0],
                  "train_end": dataset.timestamps[train.stop - 1],
                  "validation_start": "2025-05-01T00:00:00Z", "validation_end": "2025-07-31T23:00:00Z",
                  "previously_inspected_test_start": "2025-08-01T00:00:00Z",
                  "test_never_used_in_this_audit": True,
                  "method": "official source month grouped; chronological"},
        "relaxations": ["perfect knowledge of each training window", "constant initial SOH power limit",
                        "thermal nonlinearities omitted", "V8 FIFO maximum deferral age omitted",
                        "LP schedule replay uses historical V3 constraints; not a V8 admission result"],
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "asset": load_config()["asset"], "window_hours": 168, "windows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SHORE_BESS_TRAIN_FEASIBILITY:PASS {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")


if __name__ == "__main__":
    main()
