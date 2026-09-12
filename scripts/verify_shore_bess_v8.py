#!/usr/bin/env python3
"""Independently verify V8 evidence; integrity PASS never promotes a candidate.

Default: inspect evidence/v8/shore_bess/latest.json. --report accepts a retained
report directly. --replay also reloads the selected policy and replays its
persisted selected/final validation and, when present, test/forward windows using a
source-matched current environment. It does not open new held-out evaluations.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rl_model.shore_bess.v8_public_artifacts import resolve_public_artifact, model_exports, inside

METRICS = ("total_cost_cny", "carbon_kg", "peak_kw")
SAFETY = ("guardrail_violation_rate", "terminal_soc_error", "terminal_flex_backlog_kwh",
          "shore_sla_violation_kwh", "reserve_shortfall_kwh", "flex_deadline_violation_kwh",
          "physical_power_violations")
REUSED_REPORT_KIND = "formal_admission_of_immutable_pilot_checkpoints"
REUSED_SELECTION_PROTOCOL = "recompute every checkpoint on all validation windows, then freeze seed before test/forward"
LEDGER_PHASE = "after_environment_step_before_replay_storage_and_optimizer"


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchored_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "empty artifact path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"artifact escapes repository: {relative}")
    require(path.is_file(), f"missing artifact: {relative}")
    return path


def verify_hash(root: Path, relative: str, expected: str) -> Path:
    require(isinstance(expected, str) and len(expected) == 64, f"invalid SHA-256: {relative}")
    try:
        return resolve_public_artifact(root, relative, expected)
    except (ValueError, OSError) as exc:
        raise VerificationError(str(exc)) from exc


def same(actual: Any, expected: Any, label: str, *, atol: float = 1e-8) -> None:
    """Compare structures without accepting omitted/extra evidence fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{label}: keys differ")
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}", atol=atol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label}: list differs")
        for i, value in enumerate(expected):
            same(actual[i], value, f"{label}[{i}]", atol=atol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, f"{label}: differs")
    elif isinstance(expected, (float, int)):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool), f"{label}: not numeric")
        require(np.isfinite(actual) and np.isfinite(expected)
                and np.isclose(actual, expected, rtol=1e-10, atol=atol), f"{label}: numeric mismatch")
    else:
        require(actual == expected, f"{label}: differs")


def official_source_period(timestamp: str) -> str:
    """MOT's pinned Shanghai series has one shared January/February anchor."""
    stamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    require(stamp.utcoffset() == timedelta(0), "source period timestamp is not UTC")
    return f"{stamp.year:04d}-01/02" if stamp.month in (1, 2) else f"{stamp.year:04d}-{stamp.month:02d}"


def validate_evaluation(evaluation: dict, *, split_rows: int | None = None) -> None:
    starts, rows = evaluation["starts"], evaluation["rows"]
    hours = evaluation["episode_hours"]
    require(hours == 168, "evaluation episode must be 168 hours")
    require(bool(rows) and len(rows) == len(starts), "evaluation rows/start count differs")
    require(all(type(start) is int and start >= 0 for start in starts), "invalid evaluation starts")
    require(starts == sorted(set(starts)), "evaluation starts repeat or are unordered")
    require(all(b - a >= hours for a, b in zip(starts, starts[1:])), "evaluation windows overlap")
    if split_rows is not None:
        require(starts[-1] + hours < split_rows, "evaluation crosses partition boundary")
    require(evaluation["window_overlap"] is False, "window overlap declaration differs")
    if "windows" in evaluation:
        require(len(evaluation["windows"]) == len(starts), "window provenance count differs")
        for start, window in zip(starts, evaluation["windows"], strict=True):
            require(window["start_index"] == start, "window provenance start differs")
            first = datetime.fromisoformat(window["first_timestamp"].replace("Z", "+00:00"))
            last = datetime.fromisoformat(window["last_timestamp"].replace("Z", "+00:00"))
            require(last - first == timedelta(hours=167), "window timestamps do not span 168 hours")
            require(window["source_month"] == window["first_timestamp"][:7], "source-month label differs")
            if "source_period" in window:
                require(window["source_period"] == official_source_period(window["first_timestamp"]),
                        "official source-period label differs")
    keys = set(rows[0])
    for row in rows:
        require(set(row) == keys, "evaluation row metric schema drift")
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                    for v in row.values()), "non-finite or nonnumeric evaluation metric")
        require(set(METRICS) <= set(row) and all(row[k] > 0 for k in METRICS), "invalid business totals")
        components = ("energy_cost_cny", "degradation_cost_cny", "demand_charge_cny")
        if any(key in row for key in components):
            require(all(key in row for key in components), "missing total-cost accounting component")
            same(row["total_cost_cny"], sum(row[key] for key in components), "cost accounting identity")
    expected = {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}
    same(evaluation["mean"], expected, "evaluation.mean")


def verify_window_provenance(evaluation: dict, timestamps: list[str], offset: int) -> None:
    """Check calendar cluster membership against pinned data, not report labels."""
    if "windows" not in evaluation:
        return  # Historical archives predate the additive cluster schema.
    for start, window in zip(evaluation["starts"], evaluation["windows"], strict=True):
        require(window["first_timestamp"] == timestamps[offset + start]
                and window["last_timestamp"] == timestamps[offset + start + 167],
                "window timestamps differ from pinned dataset")


def source_month_clusters(values: list[float], groups: list[str]) -> dict:
    """Independent weighted cluster bootstrap; whole groups retain all rows."""
    array = np.asarray(values, dtype=np.float64)
    require(len(array) == len(groups) and len(array) > 0 and np.isfinite(array).all(),
            "invalid cluster values/groups")
    require(all(isinstance(group, str) and group for group in groups), "invalid source-month group")
    unique = sorted(set(groups))
    labels = np.asarray(groups)
    sums = np.asarray([array[labels == group].sum() for group in unique])
    counts = np.asarray([np.count_nonzero(labels == group) for group in unique])
    samples = np.random.default_rng(20260912).integers(0, len(unique), size=(5000, len(unique)))
    estimates = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    low, high = np.quantile(estimates, [.025, .975])
    return {"clustered_ci_low": float(low), "clustered_ci_high": float(high),
            "cluster_count": len(unique), "cluster_confidence": .95,
            "cluster_method": "percentile_bootstrap_of_source_month_groups",
            "cluster_resamples": 5000,
            "cluster_window_counts": dict(zip(unique, map(int, counts), strict=True)),
            "cluster_boundary": "UTC month of window start; a cross-month window is assigned to its starting month; few-cluster engineering uncertainty, not field confidence"}


def source_period_clusters(values: list[float], groups: list[str]) -> dict:
    result = source_month_clusters(values, groups)
    result.update(cluster_method="percentile_bootstrap_of_official_source_period_groups",
                  cluster_group_field="source_period",
                  cluster_boundary="Official MOT period of window start; January/February share YYYY-01/02. Cross-period windows belong to their starting period, an approximation. Few-cluster engineering uncertainty, not field confidence.")
    return result


def contained_period_sensitivity(candidate: dict, reference: dict) -> dict:
    """Additional uncertainty diagnostic; never changes the recorded admission."""
    if "windows" not in candidate:
        return {"status": "UNAVAILABLE_HISTORICAL_ARCHIVE_WITHOUT_WINDOW_TIMESTAMPS"}
    require(candidate["windows"] == reference["windows"], "sensitivity windows differ")
    included = [i for i, window in enumerate(candidate["windows"])
                if official_source_period(window["first_timestamp"]) == official_source_period(window["last_timestamp"])]
    excluded = [candidate["starts"][i] for i in range(len(candidate["windows"])) if i not in included]
    result = {"status": "COMPUTED" if included else "NO_CONTAINED_WINDOWS", "admission_reinterpreted": False,
              "estimand": "Mean weekly gain among windows fully contained in one official source period; differs from the all-window mean.",
              "all_business_windows_retained": True, "included_start_indices": [candidate["starts"][i] for i in included],
              "cross_period_start_indices_excluded_from_sensitivity_only": excluded,
              "selection_rule": "Timestamps only, independent of returns or candidate metrics; January/February remain one period.",
              "metrics": {}}
    if included:
        groups = [official_source_period(candidate["windows"][i]["first_timestamp"]) for i in included]
        for metric in METRICS:
            values = [100. * (reference["rows"][i][metric] - candidate["rows"][i][metric]) /
                      max(abs(reference["rows"][i][metric]), 1e-9) for i in included]
            result["metrics"][metric] = {"contained_window_mean": float(np.mean(values)),
                                          **source_period_clusters(values, groups)}
            result["metrics"][metric]["cluster_method"] = "percentile_bootstrap_of_contained_official_source_period_groups"
            result["metrics"][metric]["cluster_boundary"] = "Only windows fully contained in one official MOT period; January/February share an anchor. This subset sensitivity is not a confidence interval for the all-window mean or a field confidence claim."
    return result


def verify_zero_reference(reference: dict, values: np.ndarray, offset: int, physical: dict) -> None:
    """Recompute idle baseline bills directly from pinned rows, without the env."""
    demand_rate = physical["grid"]["demand_charge_cny_per_kw_month"] * 168 / (24 * 30.4375)
    for start, row in zip(reference["starts"], reference["rows"], strict=True):
        window = values[offset + start:offset + start + 168].astype(np.float64)
        require(len(window) == 168, "baseline crosses available data")
        base = window[:, 0]
        require(np.all(base >= 0) and np.all(base <= physical["grid"]["hard_pcc_limit_kw"]),
                "zero-action baseline requires site-limit intervention; explicit replay is required")
        energy_cost = float(np.sum(base * window[:, 4]))
        carbon = float(np.sum(base * window[:, 5]))
        peak = float(base.max())
        for key, expected in {"energy_cost_cny": energy_cost, "carbon_kg": carbon,
                              "peak_kw": peak, "degradation_cost_cny": 0.,
                              "demand_charge_cny": peak * demand_rate,
                              "total_cost_cny": energy_cost + peak * demand_rate,
                              "bess_throughput_kwh": 0., "aux_shift_kwh": 0.}.items():
            same(row[key], expected, "zero baseline " + key, atol=1e-5)


def comparisons(candidate: dict, reference: dict) -> dict:
    require(candidate["starts"] == reference["starts"], "paired windows differ")
    require(candidate.get("windows") == reference.get("windows"), "paired window timestamps differ")
    output = {}
    for metric in METRICS:
        gains = np.asarray([100.0 * (base[metric] - row[metric]) / max(abs(base[metric]), 1e-9)
                            for row, base in zip(candidate["rows"], reference["rows"], strict=True)])
        n = len(gains)
        require(n > 0, "no paired metric rows")
        if n == 1:
            low = high = float(gains[0])
        else:
            indices = np.random.default_rng(20260912).integers(0, n, size=(5000, n))
            low, high = (float(x) for x in np.quantile(np.mean(gains[indices], axis=1), [.025, .975]))
        output[metric] = {"n": n, "mean": float(gains.mean()),
                          "std": float(gains.std(ddof=1)) if n > 1 else 0.0,
                          "median": float(np.median(gains)), "ci_low": low, "ci_high": high,
                          "confidence": .95, "method": "percentile_bootstrap_of_mean",
                          "resamples": 5000 if n > 1 else 0, "per_window_percent": gains.tolist(),
                          "minimum_percent": float(gains.min())}
        if "windows" in candidate:
            period_schema = any("source_period" in window for window in candidate["windows"])
            field = "source_period" if period_schema else "source_month"
            groups = [window[field] for window in candidate["windows"]]
            summarize = source_period_clusters if period_schema else source_month_clusters
            output[metric].update(summarize(gains.tolist(), groups))
    return output


def declared_business_gates(evaluation: dict, comparison: dict, gate: dict, *, require_month_ci: bool = False) -> dict[str, bool]:
    """Recompute the report's frozen criteria, without changing admission policy."""
    require(gate["all_three_95ci_lower_bounds_strictly_positive"] is True
            and gate["every_window_carbon_non_regression"] is True, "unsupported weakened gate protocol")
    checks = {}
    for metric, threshold in zip(METRICS, ("cost_reduction_percent_min", "carbon_reduction_percent_min",
                                          "peak_reduction_percent_min"), strict=True):
        checks[metric + "_minimum_mean_gain"] = comparison[metric]["mean"] >= gate[threshold]
        checks[metric + "_positive_95ci"] = comparison[metric]["ci_low"] > 0.0
        if require_month_ci:
            group_name = "source_period" if "formal_source_period_cluster_95ci_lower_bounds_strictly_positive" in gate else "source_month"
            require(gate[f"formal_{group_name}_cluster_95ci_lower_bounds_strictly_positive"] is True,
                    "formal source cluster gate is disabled")
            checks[metric + f"_positive_{group_name}_95ci"] = comparison[metric]["clustered_ci_low"] > 0.0
            checks[metric + f"_{group_name}_count"] = comparison[metric]["cluster_count"] >= gate[f"formal_{group_name}_cluster_count_min"]
    checks["every_window_carbon_non_regression"] = comparison["carbon_kg"]["minimum_percent"] >= 0.0
    for metric in SAFETY + ("terminal_soc_absolute_error", "terminal_energy_debt_kwh"):
        if metric not in SAFETY and metric not in evaluation["mean"]:
            continue
        tolerance = gate["terminal_absolute_tolerance"] if metric.startswith("terminal") else gate["safety_tolerance"]
        checks[metric] = all(metric in row and abs(row[metric]) <= tolerance for row in evaluation["rows"])
    checks["maximum_flex_age_within_12_hours"] = all(
        "max_flex_age_hours" in row and row["max_flex_age_hours"] <= 12 for row in evaluation["rows"])
    return checks


def convergence(curve: list[dict], gate: dict) -> dict:
    tail = curve[-3:]
    stability = {}
    for metric, floor in gate["tail_range_floors_pp"].items():
        values = [row["comparison"][metric]["mean"] for row in tail]
        spread = float(np.ptp(values)) if values else None
        tolerance = max(floor, gate["tail_relative_range_max"] * abs(float(np.mean(values)))) if values else floor
        stability[metric] = {"range_pp": spread, "tolerance_pp": tolerance,
                             "passed": len(tail) == 3 and spread <= tolerance}
    business = len(tail) == 3 and all(all(row["gates"].values()) for row in tail)
    return {"passed": business and all(row["passed"] for row in stability.values()),
            "tail_checkpoints": len(tail), "all_tail_business_gates_passed": business, "stability": stability}


def checkpoint_rank(row: dict) -> tuple:
    return (sum(not value for value in row["gates"].values()),
            -row["comparison"]["carbon_kg"]["ci_low"], -row["comparison"]["total_cost_cny"]["ci_low"])


def verify_optimizer_proof(parameters: dict, reported_updates: int, sb3_updates: int) -> None:
    require(type(reported_updates) is int and reported_updates > 0, "no real optimizer step count")
    require(parameters.get("gradient_optimizer_step_calls") == reported_updates, "missing/mismatched actual optimizer hook count")
    components = parameters.get("optimizer_step_calls_by_component")
    require(isinstance(components, dict) and bool(components), "missing optimizer component proof")
    require(all(type(n) is int and n >= 0 for n in components.values()), "invalid optimizer component count")
    require(sum(components.values()) == reported_updates, "optimizer component sum differs")
    require(parameters["_n_updates"] == sb3_updates and sb3_updates > 0, "SB3 update counter differs")
    if "critic" in components:
        require(components["critic"] == sb3_updates, "critic optimizer steps disagree with SB3 iterations")
        require(0 < components.get("actor", 0) <= sb3_updates, "invalid actor update count")
    elif "DQN" in parameters.get("policy_class", ""):
        require(components.get("policy") == sb3_updates, "DQN optimizer count differs")


def verify_n_step_snapshot(parameters: dict, config: dict, step: int, phase: str) -> None:
    """Cross-check recorded replay statistics without counting storage as learning."""
    n_step = config["n_step"]
    require(type(n_step) is int and n_step > 1 and config["algorithm"] == "stable_baselines3.TD3",
            "unregistered n-step algorithm")
    require(config.get("algorithm_variant") == f"td3_n_step_{n_step}_off_policy_uncorrected" and config["gamma"] == 1.,
            "n-step variant or undiscounted target declaration differs")
    declared = config["replay_buffer_config"]
    require(declared["class"].split(".")[-1] == "NStepReplayBuffer", "n-step replay class differs")
    same(declared["kwargs"], {"n_step": n_step, "gamma": 1.}, "n-step replay configuration")
    require(parameters.get("replay_buffer_class", "").split(".")[-1] == "NStepReplayBuffer", "actual replay class differs")
    same(parameters.get("replay_buffer_kwargs"), declared["kwargs"], "actual replay kwargs")
    approximation = {
        "return": "sum of observed behavior rewards with undiscounted endpoint bootstrap",
        "off_policy_importance_correction": False, "sac_intermediate_entropy_terms_included": False,
        "vanilla_algorithm_equivalence_claimed": False,
    }
    same(declared["approximation"], {
        "off_policy_importance_correction": False, "behavior_policy_intermediate_rewards": True,
        "variable_tail_bootstrap_discount": "gamma=1, including shorter terminal and budget-end tails",
        "sac_intermediate_entropy_terms_included": False, "vanilla_algorithm_equivalence_claimed": False,
        "boundary": "Experimental off-policy multi-step TD3; no importance correction. Not unchanged vanilla TD3.",
    }, "off-policy approximation declaration")
    snapshot = parameters.get("replay_buffer_snapshot")
    require(isinstance(snapshot, dict), "missing n-step replay snapshot")
    same(snapshot.get("approximation"), approximation, "recorded replay approximation")
    require(snapshot.get("class") == "NStepReplayBuffer" and snapshot.get("n_step") == n_step
            and snapshot.get("gamma") == 1. and snapshot.get("n_envs") == 1
            and snapshot.get("handle_timeout_termination") is True and snapshot.get("optimize_memory_usage") is False,
            "n-step replay target or memory semantics differ")
    require(snapshot.get("statistics_scope") == "since_last_replay_buffer_reset" and snapshot.get("reset_count") == 0,
            "n-step replay statistics were reset during training")
    require(phase in {"initial_zero_updates", "post_rollout_and_optimizer", "training_end_after_storage_only_flush"}
            and snapshot.get("checkpoint_phase") == phase, "n-step checkpoint phase differs")
    count_names = ("observed_transitions", "emitted_transitions", "pending_transitions", "discarded_pending_transitions",
                   "resident_transitions", "resident_position", "observed_terminal_events", "observed_timeout_events",
                   "emitted_terminal_endpoints", "emitted_timeout_endpoints", "resident_terminal_endpoints",
                   "resident_timeout_endpoints", "sampled_batches", "sampled_transition_draws",
                   "sampled_terminal_endpoints", "sampled_timeout_endpoints")
    require(all(type(snapshot.get(name)) is int and snapshot[name] >= 0 for name in count_names), "invalid replay counter")
    observed, emitted, pending = (snapshot[key] for key in ("observed_transitions", "emitted_transitions", "pending_transitions"))
    require(observed == step == parameters["num_timesteps"], "replay observed steps differ from model collection counter")
    require(snapshot["discarded_pending_transitions"] == 0 and pending < n_step and observed == emitted + pending,
            "n-step observed/emitted/pending conservation differs")
    require(snapshot.get("transition_accounting_balanced") is True
            and snapshot.get("emitted_horizon_accounting_balanced") is True, "replay conservation flag differs")
    capacity = config["algorithm_parameters"]["buffer_size"]
    require(snapshot["resident_transitions"] == min(capacity, emitted)
            and snapshot["resident_position"] == emitted % capacity
            and snapshot["resident_full"] is (emitted >= capacity), "replay ring-buffer accounting differs")
    for prefix, count in (("emitted", emitted), ("resident", min(capacity, emitted)),
                          ("sampled", snapshot["sampled_transition_draws"])):
        histogram = snapshot.get(prefix + "_horizon_counts")
        require(isinstance(histogram, dict) and all(isinstance(key, str) and key.isdigit() and str(int(key)) == key
                and 1 <= int(key) <= n_step and type(value) is int and value > 0 for key, value in histogram.items()),
                "invalid n-step horizon histogram")
        require(sum(histogram.values()) == count, prefix + " horizon histogram count differs")
        if prefix != "emitted":
            require(set(histogram) <= set(snapshot["emitted_horizon_counts"]), "sampled/stored horizon was never emitted")
        if prefix == "resident":
            require(all(count <= snapshot["emitted_horizon_counts"][key] for key, count in histogram.items()),
                    "resident horizon count exceeds lifetime emission")
    iterations = parameters["_n_updates"]
    require(snapshot["sampled_batches"] == iterations == parameters["optimizer_step_calls_by_component"]["critic"],
            "actual replay sample batches differ from critic optimizer iterations")
    require(snapshot["sampled_transition_draws"] == iterations * config["algorithm_parameters"]["batch_size"],
            "actual replay draws differ from learner batch accounting")
    for prefix, count, suffix in (("observed", observed, "events"), ("emitted", emitted, "endpoints"),
                                  ("resident", min(capacity, emitted), "endpoints"),
                                  ("sampled", snapshot["sampled_transition_draws"], "endpoints")):
        require(snapshot[f"{prefix}_terminal_{suffix}"] + snapshot[f"{prefix}_timeout_{suffix}"] <= count,
                "terminal/timeout replay accounting exceeds transition count")
    for ending in ("terminal", "timeout"):
        require(snapshot[f"emitted_{ending}_endpoints"] <= n_step * snapshot[f"observed_{ending}_events"],
                "more terminal tails than observed episode endings")
        require(snapshot[f"resident_{ending}_endpoints"] <= snapshot[f"emitted_{ending}_endpoints"],
                "resident terminal tails exceed emitted tails")
    require(snapshot.get("sampling_count_basis") == "replay draws with replacement, not unique transitions or optimizer updates",
            "replay sampling count claim differs")
    require(snapshot.get("flush_claim_boundary") == "Storage only. Final emitted tails are not asserted to have been sampled or learned.",
            "replay storage is incorrectly claimed as learning")
    flush_counts = snapshot.get("flush_counts")
    require(isinstance(flush_counts, dict) and all(name in {"episode_end", "collection_boundary", "training_budget_end"}
            and type(count) is int and count > 0 for name, count in flush_counts.items()), "invalid replay flush statistics")
    require(flush_counts.get("episode_end", 0) == snapshot["observed_terminal_events"] + snapshot["observed_timeout_events"],
            "episode ending did not flush pending replay origins")
    if snapshot.get("last_flush") is not None:
        require(snapshot["last_flush"].get("storage_only_no_optimizer_update_performed") is True,
                "buffer flush claims optimizer activity")
    if phase == "initial_zero_updates":
        require(step == iterations == emitted == pending == 0 and snapshot["last_flush"] is None and flush_counts == {},
                "initial replay snapshot contains prior learning or observations")
    elif phase == "training_end_after_storage_only_flush":
        require(pending == 0 and flush_counts.get("training_budget_end", 0) == 1, "final replay tails were not explicitly flushed")
        last_flush = snapshot["last_flush"]
        require(last_flush["reason"] == "training_budget_end" and last_flush["observed_transitions"] == observed,
                "final flush provenance differs")
        require(snapshot.get("storage_only_flush_emitted") == last_flush["newly_stored_transitions"],
                "final flush emission count differs")
        optimizer_counts = {"optimizer_step_calls": parameters["gradient_optimizer_step_calls"],
                            "sb3_update_counter": parameters["_n_updates"],
                            "optimizer_step_calls_by_component": parameters["optimizer_step_calls_by_component"]}
        same(snapshot.get("optimizer_steps_before_flush"), optimizer_counts, "pre-flush optimizer counts")
        same(snapshot.get("optimizer_steps_after_flush"), optimizer_counts, "post-flush optimizer counts")
    else:
        require("training_budget_end" not in flush_counts, "intermediate checkpoint claims final replay flush")


def archive_proof(path: Path, row: dict) -> None:
    """Full ZIP SHA is checked by caller; independently inspect tensors/Adam steps."""
    import torch
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, f"corrupt ZIP member: {path.name}")
        data = json.loads(archive.read("data"))
        require(data.get("num_timesteps") == row["step"], "archive timestep counter differs")
        require(data.get("_n_updates") == row["sb3_update_counter"], "archive SB3 update counter differs")
        require(data.get("_v8_optimizer_step_calls") == row["optimizer_updates"], "archive missing actual optimizer counter")
        same(data.get("_v8_optimizer_steps_by_component"), row["parameters"]["optimizer_step_calls_by_component"], "archive optimizer components")
        require(("replay_buffer_snapshot" in row["parameters"]) == ("_v8_replay_snapshot" in data),
                "archive replay snapshot was omitted or invented in report parameters")
        if "replay_buffer_snapshot" in row["parameters"]:
            same(data.get("_v8_replay_snapshot"), row["parameters"]["replay_buffer_snapshot"], "archive replay snapshot")
        if "gamma" in row["parameters"]:
            same(data.get("gamma"), row["parameters"]["gamma"], "archive discount factor")
            for key, expected in row["parameters"]["policy_kwargs"].items():
                same(data.get("policy_kwargs", {}).get(key), expected, "archive network configuration " + key)
        state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
        weight_hash = hashlib.sha256()
        for key, tensor in sorted(state.items()):
            array = tensor.detach().cpu().numpy()
            require(np.isfinite(array).all(), "model contains non-finite weights")
            weight_hash.update(key.encode("utf-8")); weight_hash.update(str(array.dtype).encode("ascii"))
            weight_hash.update(str(array.shape).encode("ascii")); weight_hash.update(array.tobytes(order="C"))
        require(weight_hash.hexdigest() == row["weights_sha256"], "policy tensor digest differs")
        names = {"policy": "policy.optimizer.pth", "actor": "actor.optimizer.pth",
                 "critic": "critic.optimizer.pth", "entropy": "ent_coef_optimizer.pth"}
        for name, count in row["parameters"]["optimizer_step_calls_by_component"].items():
            require(name in names and names[name] in archive.namelist(), f"missing optimizer state: {name}")
            optimizer = torch.load(io.BytesIO(archive.read(names[name])), map_location="cpu", weights_only=True)
            steps = [int(v["step"].item() if hasattr(v["step"], "item") else v["step"])
                     for v in optimizer["state"].values() if "step" in v]
            require(bool(steps) and max(steps) == count, f"Adam state step does not corroborate hook count: {name}")
            learning_rates = row["parameters"].get("optimizer_learning_rates")
            if isinstance(learning_rates, dict):
                same([float(group["lr"]) for group in optimizer["param_groups"]], learning_rates[name],
                     f"archive {name} optimizer learning rates")


def validate_split(description: dict, timestamps: list[str], first: str, stop: str) -> slice:
    start = datetime.fromisoformat(first)
    end = datetime.fromisoformat(stop)
    expected = [str((start + timedelta(hours=i)).isoformat()) + "Z"
                for i in range(int((end - start).total_seconds() / 3600))]
    a, b = description["start_row"], description["stop_row_exclusive"]
    require(type(a) is int and type(b) is int and 0 <= a < b <= len(timestamps), "invalid split row bounds")
    require(timestamps[a:b] == expected, "split is not complete isolated source months")
    require(description["rows"] == b - a == len(expected), "split row count differs")
    require(description["first_timestamp"] == expected[0] and description["last_timestamp"] == expected[-1], "split timestamp labels differ")
    return slice(a, b)


def verify_reused_accounting(report: dict) -> None:
    """A second evaluation of fixed weights cannot count as another training run."""
    require(report.get("training_reused") is True and report["manifest"].get("training_reused") is True,
            "missing reused-training declaration")
    require(report.get("new_training_environment_steps") == 0 and report.get("new_optimizer_updates") == 0,
            "immutable adoption claims new training work")
    require(report.get("reused_training_environment_steps") == report["total_environment_steps"],
            "reused environment-step accounting differs")
    same(report.get("training_accounting"), {
        "count_as_new_training_run": False, "environment_steps_executed_by_this_run": 0,
        "optimizer_updates_executed_by_this_run": 0,
        "referenced_environment_steps": report["total_environment_steps"],
        "referenced_optimizer_updates": report["total_optimizer_updates"],
    }, "reused training accounting")


def verify_reused_training(report: dict, run_dir: Path, root: Path) -> list[dict]:
    """Bind all copied checkpoints to independently verified original pilot runs."""
    root, run_dir = root.resolve(), run_dir.resolve()
    verify_reused_accounting(report)
    origins = report.get("source_training_runs")
    require(isinstance(origins, list) and len(origins) == 3, "adoption requires exactly three original pilot runs")
    require([row["seed"] for row in origins] == report["config"]["seeds"], "adopted source seeds differ")
    source_ids = [row["run_id"] for row in origins]
    require(len(set(source_ids)) == 3 and report["config"].get("reused_training_run_ids") == source_ids,
            "adopted source run identity differs")
    formal_config = {key: value for key, value in report["config"].items()
                     if key not in {"seeds", "pilot", "reused_training_run_ids"}}
    protected, summaries, first_protocol = {}, [], None
    for origin, result in zip(origins, report["results"], strict=True):
        source_path = verify_hash(root, origin["report_path"], origin["report_sha256"])
        require(source_path != run_dir / "report.json", "adoption references itself")
        source = read_json(source_path)
        require(source.get("report_kind") != REUSED_REPORT_KIND and not source.get("training_reused", False),
                "adoption source cannot itself reuse training")
        require(source["run_id"] == origin["run_id"] and source["config"]["pilot"] is True
                and source["status"] == "PILOT_VALIDATION_ONLY" and source["evaluations"] == {},
                "adoption source was not a sealed validation-only pilot")
        require(source["config"]["seeds"] == [origin["seed"]] and len(source["results"]) == 1,
                "adoption source must contain the declared single seed")
        require(all(source["checks"].get(name) is True for name in (
            "real_optimizer_updates", "weights_changed", "source_unchanged_during_training",
            "config_and_dataset_files_unchanged", "historical_pointers_preserved", "no_training_rendering")),
            "adoption source training integrity check failed")
        # Recursion is bounded: sources are explicitly forbidden from being adoptions.
        verified = verify_report(source_path, root=root)
        summaries.append({"run_id": source["run_id"], "report_sha256": verified["report_sha256"],
                          "integrity_status": verified["integrity_status"]})
        require(source["generated_at"] <= report["manifest"]["started_at"], "pilot was completed after admission began")
        same({key: value for key, value in source["config"].items() if key not in {"seeds", "pilot"}},
             formal_config, "immutable source training configuration")
        protocol = {"versions": source["manifest"]["versions"], "source_sha256": source["manifest"]["source_sha256"],
                    "input_files_sha256": source["input_files_sha256"],
                    "train": source["manifest"]["train"], "validation": source["manifest"]["validation"]}
        if first_protocol is None:
            first_protocol = protocol
        else:
            same(protocol, first_protocol, "independent pilot protocol compatibility")
        for name in ("train", "validation", "versions", "dataset_id", "dataset_sha256"):
            same(source["manifest"][name], report["manifest"][name], "adopted source " + name)
        same(source["input_files_sha256"], report["manifest"]["input_files_sha256"], "adopted historical inputs")
        formal_sources = dict(report["manifest"]["source_sha256"])
        require("scripts/admit_shore_bess_v8.py" in formal_sources, "admission source was not frozen")
        formal_sources.pop("scripts/admit_shore_bess_v8.py")
        same(formal_sources, source["manifest"]["source_sha256"], "adopted training sources")
        source_dir = source_path.parent
        for name in ("config", "manifest"):
            path = verify_hash(root, origin[name + "_path"], origin[name + "_sha256"])
            require(path == source_dir / (name + ".json"), "source mirror belongs to another run")
            same(read_json(path), source[name], "adopted " + name + " snapshot")
        original = source["results"][0]
        for origin_key, result_key in (("environment_steps", "steps"), ("optimizer_updates", "optimizer_updates"),
                                        ("sb3_update_counter", "sb3_update_counter"), ("initial_weights_sha256", "initial_weights_sha256")):
            same(origin[origin_key], original[result_key], "source training accounting " + origin_key)
        for name in ("seed", "steps", "optimizer_updates", "sb3_update_counter", "initial_weights_sha256",
                     "final_weights_sha256", "parameters", "render_calls"):
            same(result[name], original[name], "reused seed training fact " + name)
        require(result.get("training_reused") is True and result.get("source_training_run_id") == origin["run_id"],
                "seed lacks original training identity")
        source_seed = source_dir / f"seed_{origin['seed']}"
        formal_seed = run_dir / f"seed_{origin['seed']}"
        require(digest(source_seed / "monitor.csv") == digest(formal_seed / "monitor.csv"), "copied training monitor changed")
        original_ledger, copied_ledger = original.get("training_episode_ledger"), result.get("training_episode_ledger")
        require((original_ledger is None) == (copied_ledger is None), "adoption omitted or invented a training ledger")
        if original_ledger is not None:
            expected_ledger = {**original_ledger, "path": str((formal_seed / "training_episodes.csv").relative_to(root)),
                               "training_reused": True, "source_path": original_ledger["path"],
                               "source_sha256": original_ledger["sha256"]}
            same(copied_ledger, expected_ledger, "immutable training episode ledger descriptor")
            same(report["manifest"].get("training_episode_ledger_contract"),
                 source["manifest"].get("training_episode_ledger_contract"), "adopted training ledger contract")
        require(read_json(formal_seed / "initial_validation.json").get("reconstructed_from_source_seed_without_learning") is True,
                "adopted initial control was not declared as zero-update reconstruction")
        source_curve, formal_curve = read_json(source_seed / "curve.json"), read_json(formal_seed / "curve.json")
        require(len(source_curve) == len(formal_curve) == len(origin["checkpoints"]), "adoption omitted original checkpoints")
        for old, new, link in zip(source_curve, formal_curve, origin["checkpoints"], strict=True):
            same(link, {"step": old["step"], "source_model_path": old["model_path"],
                        "source_model_sha256": old["model_sha256"], "source_weights_sha256": old["weights_sha256"],
                        "formal_model_path": new["model_path"], "formal_model_sha256": new["model_sha256"]},
                 "adopted checkpoint mapping")
            old_path = verify_hash(root, link["source_model_path"], link["source_model_sha256"])
            new_path = verify_hash(root, link["formal_model_path"], link["formal_model_sha256"])
            require(inside(root, link["source_model_path"]).is_relative_to(source_seed)
                    and inside(root, link["formal_model_path"]).is_relative_to(formal_seed),
                    "checkpoint is outside its seed evidence directory")
            require(digest(old_path) == digest(new_path), "adoption modified original checkpoint bytes")
            for name in ("step", "optimizer_updates", "sb3_update_counter", "weights_sha256", "parameters"):
                same(new[name], old[name], "immutable checkpoint fact " + name)
            if "checkpoint_phase" in old or "checkpoint_phase" in new:
                same(new.get("checkpoint_phase"), old.get("checkpoint_phase"), "immutable checkpoint phase")
            same(new.get("optimizer", {}), old.get("optimizer", {}), "immutable optimizer diagnostics")
            require(new.get("source_training_run_id") == origin["run_id"], "checkpoint source identity differs")
        protected[origin["report_path"]] = origin["report_sha256"]
        protected.update(source["evidence_files_sha256"])
    same(report.get("source_training_files_sha256"), protected, "exhaustive original training file inventory")
    for name, sha in protected.items():
        verify_hash(root, name, sha)
    return summaries


def verify_training_ledger(result: dict, config: dict, manifest: dict, seed_dir: Path, root: Path) -> dict:
    """Check every completed rollout against bills, learner phase and Monitor.

    Recorded unsafe episodes remain valid historical evidence when faithfully
    recorded; this audit reports them and never changes admission decisions.
    """
    descriptor = result.get("training_episode_ledger")
    if descriptor is None:
        require("training_episode_ledger_contract" not in manifest, "declared training ledger is missing")
        return {"status": "UNAVAILABLE_HISTORICAL_REPORT_WITHOUT_LEDGER", "seed": result["seed"]}
    require(manifest.get("training_episode_ledger_contract") == descriptor.get("record_phase") == LEDGER_PHASE,
            "training ledger observation phase differs")
    require(descriptor.get("partial_final_episode_included") is False, "partial episode claimed as complete")
    require(descriptor.get("metrics_source") == "unaltered numeric environment info.episode_metrics from physical training rollouts",
            "training ledger metric source differs")
    path = verify_hash(root, descriptor["path"], descriptor["sha256"])
    require(path == seed_dir.resolve() / "training_episodes.csv", "training ledger is outside the seed directory")
    if descriptor.get("training_reused"):
        verify_hash(root, descriptor["source_path"], descriptor["source_sha256"])
        require(descriptor["source_sha256"] == descriptor["sha256"], "copied training ledger bytes changed")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    hours, steps = config["episode_hours"], result["steps"]
    require(len(rows) == descriptor["completed_episodes"] == steps // hours, "completed training episodes were omitted or duplicated")
    prefix = "episode_metrics."
    identity_fields = {"episode_index", "environment_steps", "optimizer_step_calls_at_observation",
                       "sb3_update_counter_at_observation", "record_phase"}
    required_metrics = {"energy_cost_cny", "degradation_cost_cny", "demand_charge_cny", "total_cost_cny",
        "financial_delta_cny", "carbon_delta_kg", "carbon_kg", "peak_kw", "training_reward",
        "guardrail_violations", "guardrail_violation_rate", "projection_count", "projection_rate",
        "nonzero_bess_actions", "nonzero_bess_action_rate", "nonzero_flex_actions", "nonzero_flex_action_rate",
        "terminal_soc_error", "terminal_flex_backlog_kwh", "shore_sla_violation_kwh", "reserve_shortfall_kwh",
        "flex_deadline_violation_kwh", "physical_power_violations", "max_flex_age_hours",
        "soc_min_observed", "soc_max_observed", "temperature_max_c"}
    require(fields is not None and len(set(fields)) == len(fields) and identity_fields <= set(fields)
            and all(name in identity_fields or name.startswith(prefix) for name in fields)
            and {prefix + name for name in required_metrics} <= set(fields), "training ledger metric schema differs")
    with (seed_dir / "monitor.csv").open(encoding="utf-8") as handle:
        monitor = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    require(len(monitor) == len(rows), "training Monitor/ledger completed episode counts differ")
    parameters = result["parameters"]
    algorithm = config["algorithm"].split(".")[-1]
    exact_counter_protocol = algorithm in {"SAC", "TD3"} and parameters.get("train_freq", {}).get("unit") == "step"
    metrics = []
    max_reward_error = max_monitor_error = 0.
    previous_updates = previous_calls = -1
    for index, (row, monitor_row) in enumerate(zip(rows, monitor, strict=True), 1):
        require(None not in row and all(value is not None for value in row.values()), "malformed training ledger row")
        observed = int(row["environment_steps"])
        calls, updates = int(row["optimizer_step_calls_at_observation"]), int(row["sb3_update_counter_at_observation"])
        require(int(row["episode_index"]) == index and observed == index * hours,
                "training episode index or collection step is not contiguous")
        require(row["record_phase"] == LEDGER_PHASE, "training ledger row phase differs")
        require(previous_updates <= updates <= result["sb3_update_counter"]
                and previous_calls <= calls <= result["optimizer_updates"], "training ledger optimizer counters regress or exceed final state")
        if exact_counter_protocol:
            frequency = parameters["train_freq"]["frequency"]
            require(type(frequency) is int and frequency > 0, "invalid training rollout frequency")
            completed_rollouts = (observed - 1) // frequency
            training_rollouts = max(0, completed_rollouts - parameters["learning_starts"] // frequency)
            gradient_steps = parameters["gradient_steps"]
            expected_updates = training_rollouts * (frequency if gradient_steps == -1 else gradient_steps)
            expected_calls = (expected_updates + expected_updates // parameters["policy_delay"] if algorithm == "TD3"
                              else expected_updates * len(parameters["optimizer_step_calls_by_component"]))
            require(updates == expected_updates and calls == expected_calls, "training ledger pre-optimizer phase counter mismatch")
        previous_updates, previous_calls = updates, calls
        values = {name[len(prefix):]: float(row[name]) for name in fields if name.startswith(prefix)}
        require(all(np.isfinite(value) for value in values.values()), "non-finite training ledger metric")
        same(values["total_cost_cny"], sum(values[key] for key in
             ("energy_cost_cny", "degradation_cost_cny", "demand_charge_cny")), "training cost accounting identity")
        for count_key, rate_key in (("guardrail_violations", "guardrail_violation_rate"),
              ("projection_count", "projection_rate"), ("nonzero_bess_actions", "nonzero_bess_action_rate"),
              ("nonzero_flex_actions", "nonzero_flex_action_rate")):
            require(0 <= values[count_key] <= hours and values[count_key].is_integer(), "training event count is invalid")
            same(values[rate_key], values[count_key] / hours, "training rate denominator " + rate_key)
        require(0 <= values["physical_power_violations"] <= values["guardrail_violations"]
                and values["physical_power_violations"].is_integer(), "physical violation was omitted from guard count")
        reward = -(values["financial_delta_cny"] + config["carbon_price_cny_per_kg_constraint_multiplier"] *
                   values["carbon_delta_kg"]) / config["normalization"]["reward_scale"] - 100. * values["guardrail_violations"]
        error = abs(values["training_reward"] - reward)
        same(values["training_reward"], reward, "training reward telescoping accounting", atol=1e-7)
        require(int(monitor_row["l"]) == hours, "training Monitor episode length differs")
        monitor_error = abs(float(monitor_row["r"]) - values["training_reward"])
        require(monitor_error <= 5.1e-7, "training reward differs from independent Monitor return")
        max_reward_error, max_monitor_error = max(max_reward_error, error), max(max_monitor_error, monitor_error)
        metrics.append(values)
    snapshot = parameters.get("replay_buffer_snapshot")
    if snapshot is not None:
        require(snapshot["observed_terminal_events"] == len(rows) and snapshot["observed_timeout_events"] == 0,
                "training ledger terminal events differ from replay snapshot")
    summaries = {key: {"min": float(min(row[key] for row in metrics)), "max": float(max(row[key] for row in metrics)),
                       "mean": float(np.mean([row[key] for row in metrics]))} for key in sorted(required_metrics)} if metrics else {}
    asset = config.get("physical_config", {}).get("asset", {})
    unsafe = [i for i, row in enumerate(metrics, 1) if any(abs(row[key]) > (1e-6 if key.startswith("terminal_") else 1e-12)
              for key in SAFETY) or row["max_flex_age_hours"] > 12. + 1e-6
              or row["soc_min_observed"] < asset.get("soc_min", 0.) - 1e-6
              or row["soc_max_observed"] > asset.get("soc_max", 1.) + 1e-6
              or row["temperature_max_c"] > asset.get("temperature_trip_c", float("inf")) + 1e-6]
    return {"status": "PASS", "seed": result["seed"], "path": descriptor["path"], "sha256": descriptor["sha256"],
            "completed_episodes": len(rows), "completed_episode_steps": len(rows) * hours,
            "remaining_partial_episode_steps": steps - len(rows) * hours,
            "record_phase": LEDGER_PHASE, "exact_pre_optimizer_counters_recomputed": exact_counter_protocol,
            "max_training_reward_identity_error": max_reward_error, "max_monitor_return_rounding_error": max_monitor_error,
            "unsafe_completed_episode_indices": unsafe, "metrics": summaries, "admission_reinterpreted": False,
            "claim_boundary": "Full recorded training episodes only; the final partial episode is not claimed to have a complete physical ledger. Monitor return is the V8 training reward, not legacy reward."}


def verify_report(report_path: Path, *, root: Path = ROOT, replay: bool = False, pointer: dict | None = None) -> dict:
    report_path = report_path.resolve()
    require(report_path.is_relative_to(root.resolve()), "report must reside in the repository evidence tree")
    report = read_json(report_path)
    require(report["schema"] == "port-shore-bess-v8-report.v1", "unsupported report schema")
    reused = report.get("report_kind") == REUSED_REPORT_KIND
    require(reused or not report.get("training_reused", False), "unknown reused-training protocol")
    run_dir = report_path.parent
    config, manifest = report["config"], report["manifest"]
    same(read_json(run_dir / "config.json"), config, "config snapshot")
    same(read_json(run_dir / "manifest.json"), manifest, "manifest snapshot")
    same(read_json(run_dir / "selection.json"), report["selection"], "selection snapshot")
    for obj in (report, manifest):
        require(obj.get("simulation_mode") is True and all(obj.get(key) is False for key in
                ("production_authority", "dispatch_allowed", "live_data_verified")), "production/field authority boundary violated")
    require(manifest["teacher_actions_used"] is False and manifest["warm_start_used"] is False, "teacher or warm start used")
    require(manifest["normalization_fit_split"] == "train_only", "scaler contamination")
    require(manifest["selection_protocol"] == (REUSED_SELECTION_PROTOCOL if reused else "validation_only_then_frozen_checkpoint_and_seed"),
            "selection protocol differs")
    require(manifest["test_status"] == ("not_opened" if config["pilot"] else "sealed_until_selection_previously_used_benchmark")
            and manifest["forward_status"] == ("not_loaded" if config["pilot"] else "sealed_until_selection_previously_used_benchmark"),
            "held-out access protocol differs")
    require(config.get("optimizer_update_count_basis") == "actual PyTorch optimizer step post-hook calls", "missing actual-update-count evidence protocol")
    n_step_protocol = config.get("n_step", 1) > 1
    if n_step_protocol:
        require("app/services/rl_model/shore_bess/v8_replay.py" in manifest["source_sha256"], "n-step replay source was not frozen")
    require(config["physical_config"]["control_authority"] == "recommendation_only", "physical config authority differs")
    if not config["discrete"]:
        require(config["action_mapping"].get("economic_teacher_used") is False, "continuous mapping contains economic teacher")
    evidence = report.get("evidence_files_sha256")
    require(isinstance(evidence, dict) and bool(evidence), "missing exhaustive evidence_files_sha256 anchors")
    expected_files = {str(path.relative_to(root)) for path in run_dir.rglob("*")
                      if path.is_file() and path.suffix in {".json", ".csv", ".zip"}
                      and "source" not in path.relative_to(run_dir).parts and path != report_path}
    # Raw training archives remain local; their original identities are still
    # exhaustive evidence entries when an equivalent public copy is present.
    prefix = str(run_dir.relative_to(root)) + "/"
    expected_files.update(name for item in model_exports(root) for name in item["source_paths"]
                          if name.startswith(prefix) and "/source/" not in name)
    require(set(evidence) == expected_files, "evidence inventory is incomplete or contains unregistered artifacts")
    for name, sha in evidence.items():
        verify_hash(root, name, sha)
    require(bool(manifest["source_sha256"]), "missing source snapshot hashes")
    current_source_drift = []
    for name, sha in manifest["source_sha256"].items():
        snapshot = verify_hash(run_dir / "source", name, sha)
        current = root / name
        if not current.is_file() or digest(current) != sha:
            current_source_drift.append(name)
    trainer = anchored_path(run_dir / "source", "scripts/train_shore_bess_v8.py")
    assignments = {}
    for node in ast.parse(trainer.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"GATE", "EPISODE_HOURS"}:
                    assignments[target.id] = ast.literal_eval(node.value)
    same(config["admission_gate"], assignments.get("GATE"), "gate matches frozen training source")
    same(config["episode_hours"], assignments.get("EPISODE_HOURS"), "episode matches frozen training source")
    physical_sources = [name for name in manifest["source_sha256"]
                        if name.startswith("config/") and name.endswith(".json")]
    require(len(physical_sources) == 1, "ambiguous frozen physical config")
    same(config["physical_config"], read_json(run_dir / "source" / physical_sources[0]), "physical config snapshot")
    selection = report["selection"]
    require(selection["selection_access_to_test"] is False and selection["selection_access_to_forward"] is False,
            "held-out data contaminated selection")
    require(manifest["started_at"] <= selection["frozen_at"] <= report["generated_at"], "selection freeze chronology differs")
    from app.services.rl_training.datasets import load_port_dataset
    dataset = load_port_dataset(manifest["dataset_id"])
    require(digest(dataset.path) == manifest["dataset_sha256"] == dataset.fingerprint, "historical dataset hash differs")
    anchored_inputs = manifest.get("input_files_sha256")
    expected_inputs = None
    if anchored_inputs is not None:
        expected_inputs = {str(dataset.path.resolve().relative_to(root.resolve())), str(dataset.path.with_suffix(".meta.json").resolve().relative_to(root.resolve())),
                           physical_sources[0]}
        require(set(anchored_inputs) == expected_inputs, "historical input inventory differs")
        for name, sha in anchored_inputs.items():
            verify_hash(root, name, sha)
    train = validate_split(manifest["train"], dataset.timestamps, "2024-01-01T00:00:00", "2025-05-01T00:00:00")
    validation = validate_split(manifest["validation"], dataset.timestamps, "2025-05-01T00:00:00", "2025-08-01T00:00:00")
    require(train.stop == validation.start, "training/validation source months overlap")
    reference = read_json(run_dir / "validation_reference.json")
    validate_evaluation(reference, split_rows=validation.stop - validation.start)
    verify_window_provenance(reference, dataset.timestamps, validation.start)
    verify_zero_reference(reference, dataset.values, validation.start, config["physical_config"])
    require(reference["starts"] == manifest["validation_starts"], "validation windows changed")
    if reused:
        require(config["pilot"] is False and reference["starts"] == list(range(0, validation.stop - validation.start - 168, 168)),
                "formal adoption omitted full-validation windows")
    gate = config["admission_gate"]
    cluster_protocol = any(f"formal_{group}_cluster_95ci_lower_bounds_strictly_positive" in gate
                           for group in ("source_month", "source_period"))
    if cluster_protocol:
        require("windows" in reference, "new cluster protocol requires provenance for every window")
        if "formal_source_period_cluster_95ci_lower_bounds_strictly_positive" in gate:
            require(all("source_period" in window for window in reference["windows"]),
                    "official-period protocol requires January/February-aware cluster labels")
        require(anchored_inputs is not None and "input_files_sha256" in report,
                "new protocol requires initial and final input SHA inventory")
    results = report["results"]
    seeds = [row["seed"] for row in results]
    require(seeds == config["seeds"] and len(set(seeds)) == len(seeds), "seed list differs or repeats")
    require(bool(seeds) and (config["pilot"] or len(seeds) >= 3), "formal report requires three independent seeds")
    ledger_summaries = []
    for result in results:
        seed_dir = run_dir / f"seed_{result['seed']}"
        same(read_json(seed_dir / "result.json"), result, "seed result snapshot")
        curve = read_json(seed_dir / "curve.json")
        require(bool(curve), "no optimizer checkpoints")
        if n_step_protocol:
            require(curve[-1].get("checkpoint_phase") == "training_end_after_storage_only_flush"
                    and all(row.get("checkpoint_phase") == "post_rollout_and_optimizer" for row in curve[:-1]),
                    "checkpoint replay phases do not match collection chronology")
        previous_step = previous_updates = -1
        for row in curve:
            require(row["step"] > previous_step and row["optimizer_updates"] > previous_updates, "checkpoint steps/updates do not increase")
            previous_step, previous_updates = row["step"], row["optimizer_updates"]
            verify_optimizer_proof(row["parameters"], row["optimizer_updates"], row["sb3_update_counter"])
            same(row["parameters"]["gamma"], config["gamma"], "configured training discount")
            same(row["parameters"]["policy_kwargs"]["net_arch"], config["network"], "configured network size")
            path = verify_hash(root, row["model_path"], row["model_sha256"])
            archive_proof(path, row)
            if n_step_protocol:
                verify_n_step_snapshot(row["parameters"], config, row["step"], row.get("checkpoint_phase"))
            else:
                require("replay_buffer_snapshot" not in row["parameters"], "undeclared multi-step replay in one-step configuration")
            validate_evaluation(row["evaluation"], split_rows=validation.stop - validation.start)
            verify_window_provenance(row["evaluation"], dataset.timestamps, validation.start)
            comp = comparisons(row["evaluation"], reference)
            same(row["comparison"], comp, "checkpoint comparison")
            same(row["gates"], declared_business_gates(row["evaluation"], comp, gate,
                 require_month_ci=cluster_protocol and not config["pilot"]), "checkpoint gates")
        same(result["selected"], min(curve, key=checkpoint_rank), "validation-only checkpoint selection")
        same(result["convergence"], convergence(curve, gate), "convergence")
        require(result["steps"] == curve[-1]["step"] >= config["requested_steps_per_seed"], "final training steps differ")
        require(result["optimizer_updates"] == curve[-1]["optimizer_updates"], "final optimizer proof differs")
        same(result["parameters"], curve[-1]["parameters"], "final optimizer and training parameters")
        verify_optimizer_proof(result["parameters"], result["optimizer_updates"], result["sb3_update_counter"])
        require(result["final_weights_sha256"] == curve[-1]["weights_sha256"] != result["initial_weights_sha256"], "network weights did not change")
        initial = read_json(seed_dir / "initial_validation.json")
        require(initial["weights_sha256"] == result["initial_weights_sha256"], "initial weights provenance differs")
        require(initial["parameters"].get("num_timesteps") == 0
                and initial["parameters"].get("_n_updates") == 0
                and initial["parameters"].get("gradient_optimizer_step_calls") == 0,
                "untrained control contains prior training updates")
        if n_step_protocol:
            verify_n_step_snapshot(initial["parameters"], config, 0, "initial_zero_updates")
        validate_evaluation(initial["evaluation"], split_rows=validation.stop - validation.start)
        verify_window_provenance(initial["evaluation"], dataset.timestamps, validation.start)
        same(initial["comparison"], comparisons(initial["evaluation"], reference), "untrained control comparison")
        require(result["model_sha256"] == result["selected"]["model_sha256"], "selected copy differs")
        verify_hash(root, result["model_path"], result["model_sha256"])
        require(selection["per_seed_model_sha256"][str(result["seed"])] == result["model_sha256"], "selection seed hash differs")
        ledger_summaries.append(verify_training_ledger(result, config, manifest, seed_dir, root))
    require(len({row["initial_weights_sha256"] for row in results}) == len(results), "independent seeds share identical initial network weights")
    chosen = min(results, key=lambda row: (not row["convergence"]["passed"], *checkpoint_rank(row["selected"])))
    require(report["selected_seed"] == selection["selected_seed"] == chosen["seed"], "seed selection is not validation-only")
    require(selection["model_path"] == chosen["model_path"] and selection["model_sha256"] == chosen["model_sha256"], "global selected model differs")
    require(set(selection["per_seed_model_sha256"]) == {str(seed) for seed in seeds}, "selection contains extra or missing seeds")
    require(report["total_optimizer_updates"] == sum(row["optimizer_updates"] for row in results), "total optimizer updates differ")
    require(report["total_environment_steps"] == sum(row["steps"] for row in results), "total environment steps differ")
    verified_origins = verify_reused_training(report, run_dir, root) if reused else []
    validation_passed = all(all(row["selected"]["gates"].values()) and row["convergence"]["passed"] for row in results)
    partitions = report["evaluations"]
    validation_rejected = reused and not validation_passed
    require(set(partitions) == (set() if config["pilot"] or validation_rejected else {"test", "forward"}),
            "missing/unexpected held-out partitions")
    if reused:
        same(report.get("holdout_access"), ({"test": "not_opened_validation_rejected", "forward": "not_loaded_validation_rejected"}
             if validation_rejected else {label: "evaluated_after_full_validation_selection_freeze" for label in ("test", "forward")}),
             "adopted held-out access protocol")
        require(report.get("selection_sha256") == digest(run_dir / "selection.json"), "adopted frozen selection hash differs")
        if validation_rejected:
            require(not any((run_dir / name).exists() for name in
                    ("test_evaluation.json", "forward_evaluation.json", "forward_input_manifest.json")),
                    "rejected validation run contains held-out access artifacts")
    for label, partition in partitions.items():
        same(read_json(run_dir / f"{label}_evaluation.json"), partition, f"{label} snapshot")
        require(partition["status"] == "previously_used_chronological_benchmark_not_fresh_blind_data", "false fresh-blind claim")
        source = load_port_dataset(partition["dataset_id"])
        require(digest(source.path) == partition["dataset_sha256"] == source.fingerprint, f"{label} dataset hash differs")
        if label == "test":
            require(source.fingerprint == dataset.fingerprint, "test does not use pinned history")
            split = validate_split(partition["split"], source.timestamps, "2025-08-01T00:00:00", "2026-01-01T00:00:00")
            require(split.start == validation.stop, "test overlaps validation source months")
        else:
            validate_split(partition["split"], dataset.timestamps + source.timestamps,
                           "2026-01-01T00:00:00", "2026-06-01T00:00:00")
            if anchored_inputs is not None:
                forward_manifest = read_json(run_dir / "forward_input_manifest.json")
                require(forward_manifest["dataset_id"] == source.dataset_id, "forward input dataset differs")
                require(forward_manifest["opened_after_selection_sha256"] == digest(run_dir / "selection.json"),
                        "forward input manifest does not anchor the frozen selection")
                forward_inputs = {str(source.path.resolve().relative_to(root.resolve())), str(source.path.with_suffix(".meta.json").resolve().relative_to(root.resolve()))}
                require(set(forward_manifest["input_files_sha256"]) == forward_inputs, "forward input inventory differs")
                expected_inputs.update(forward_inputs)
                for name, sha in forward_manifest["input_files_sha256"].items():
                    verify_hash(root, name, sha)
                    require(report["input_files_sha256"].get(name) == sha, "forward final input hash differs")
        ref = partition["reference"]
        validate_evaluation(ref, split_rows=partition["split"]["rows"])
        reference_offset = partition["split"]["start_row"] if label == "test" else 0
        verify_window_provenance(ref, source.timestamps, reference_offset)
        verify_zero_reference(ref, source.values, reference_offset, config["physical_config"])
        require(ref["starts"] == list(range(0, partition["split"]["rows"] - 168, 168)), "held-out windows were selectively omitted")
        require([row["seed"] for row in partition["per_seed"]] == seeds, "held-out seeds differ")
        for row in partition["per_seed"]:
            validate_evaluation(row["evaluation"], split_rows=partition["split"]["rows"])
            verify_window_provenance(row["evaluation"], source.timestamps, reference_offset)
            comp = comparisons(row["evaluation"], ref)
            same(row["comparison"], comp, f"{label} comparison")
            same(row["gates"], declared_business_gates(row["evaluation"], comp, gate,
                 require_month_ci=cluster_protocol), f"{label} business gates")
    if anchored_inputs is not None:
        final_inputs = report["input_files_sha256"]
        require(set(final_inputs) == expected_inputs, "final input inventory is incomplete or contains undeclared inputs")
        for name, sha in final_inputs.items():
            verify_hash(root, name, sha)
            if name in anchored_inputs:
                require(anchored_inputs[name] == sha, "initial/final input fingerprints differ")
    checks = dict(report["checks"])
    recomputed = {"formal_run": not config["pilot"], "three_independent_seeds": len(seeds) >= 3,
                  "all_seeds_validation_passed": all(all(row["selected"]["gates"].values()) for row in results),
                  "all_seeds_last_three_stable": all(row["convergence"]["passed"] for row in results),
                  "real_optimizer_updates": all(row["optimizer_updates"] > 0 for row in results),
                  "weights_changed": all(row["initial_weights_sha256"] != row["final_weights_sha256"] for row in results),
                  "no_training_rendering": all(row["render_calls"] == 0 for row in results),
                  "test_all_seed_gates": "test" in partitions and all(all(row["gates"].values()) for row in partitions["test"]["per_seed"]),
                  "forward_all_seed_gates": "forward" in partitions and all(all(row["gates"].values()) for row in partitions["forward"]["per_seed"])}
    if anchored_inputs is not None:
        recomputed["config_and_dataset_files_unchanged"] = True
    if reused:
        recomputed.update(source_training_evidence_preserved=True, frozen_selection_preserved=True)
    for key, expected in recomputed.items():
        same(checks.get(key), expected, "admission check " + key)
    # Historical runtime assertions are anchored declarations, not proof that
    # today's source/pointer state must remain identical forever.
    require(set(checks) == set(recomputed) | {"source_unchanged_during_training", "historical_pointers_preserved"}, "unknown/missing report checks")
    admitted = all(checks.values())
    expected_status = ("PILOT_VALIDATION_ONLY" if config["pilot"] else "VALIDATION_REJECTED" if validation_rejected
                       else "ADMITTED_OFFLINE_RL" if admitted else "CANDIDATE_NOT_ADMITTED")
    require(report["status"] == expected_status, "report status conflicts with its declared gates")
    if reused:
        require(type(report["promoted"]) is bool and (not report["promoted"] or admitted), "unqualified candidate was promoted")
        same(report.get("promotion"), {"candidate_qualified": admitted, "promoted": report["promoted"],
             "reason": "qualified vacant champion slot" if report["promoted"] else "validation/benchmark gate failed" if not admitted
             else "existing champion retained pending paired incumbent comparison"}, "admission and champion promotion distinction")
    else:
        require(report["promoted"] is admitted, "report promotion conflicts with its declared gates")
    if pointer is not None:
        for key in ("run_id", "status"):
            require(pointer[key] == report[key], f"pointer {key} differs")
        require(pointer["report_sha256"] == digest(report_path), "pointer report hash differs")
        require(pointer["model_path"] == chosen["model_path"] and pointer["model_sha256"] == chosen["model_sha256"], "pointer model differs")
        require(pointer["production_authority"] is False, "pointer enables production")
    replay_result = replay_selected(report, root, current_source_drift) if replay else None
    sensitivity = {"validation": contained_period_sensitivity(chosen["selected"]["evaluation"], reference)}
    for label, partition in partitions.items():
        selected_evaluation = next(row["evaluation"] for row in partition["per_seed"] if row["seed"] == chosen["seed"])
        sensitivity[label] = contained_period_sensitivity(selected_evaluation, partition["reference"])
    return {"schema": "shore-bess-v8-independent-verification.v1", "integrity_status": "PASS",
            "report_path": str(report_path.relative_to(root)), "report_sha256": digest(report_path),
            "run_id": report["run_id"], "reported_status": report["status"], "admitted_offline": admitted,
            "failed_admission_checks": [key for key, passed in checks.items() if not passed],
            "verified_seed_count": len(seeds), "verified_optimizer_updates": report["total_optimizer_updates"],
            "training_reused": reused, "verified_original_training_runs": verified_origins,
            "new_optimizer_updates": 0 if reused else report["total_optimizer_updates"],
            "verified_anchored_artifacts": len(evidence), "current_source_drift": current_source_drift,
            "replay": replay_result, "production_authority": False,
            "training_episode_ledgers": ledger_summaries,
            "contained_official_period_sensitivity": sensitivity,
            "limitations": ["Hashes anchor recorded evidence; no signed wall-clock attestation is available.",
                            "No-teacher and no-warm-start declarations are source/manifest evidence, not third-party attestation.",
                            "Start-period clusters approximate cross-period dependence. Contained-period sensitivity has a different estimand and never changes admission."]}


def replay_selected(report: dict, root: Path, source_drift: list[str]) -> dict:
    require(not source_drift, "--replay requires current sources to match every frozen source snapshot")
    import torch
    import stable_baselines3 as sb3
    from sb3_contrib import MaskablePPO
    from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8Env, ShoreBESSV8SACEnv
    from app.services.rl_training.datasets import PortDataset, load_port_dataset
    torch.set_num_threads(1)
    config, manifest = report["config"], report["manifest"]
    classes = {"stable_baselines3.DQN": sb3.DQN, "stable_baselines3.PPO": sb3.PPO,
               "stable_baselines3.SAC": sb3.SAC, "stable_baselines3.TD3": sb3.TD3,
               "sb3_contrib.MaskablePPO": MaskablePPO}
    require(config["algorithm"] in classes, "unsupported replay algorithm")
    selected_path = report["selection"]["model_path"]
    resolved_path = verify_hash(root, selected_path, report["selection"]["model_sha256"])
    model = classes[config["algorithm"]].load(resolved_path, device="cpu")
    history = load_port_dataset(manifest["dataset_id"])
    train = slice(manifest["train"]["start_row"], manifest["train"]["stop_row_exclusive"])
    output = {}
    # A rejected pilot still has a reproducible validation policy. Replaying its
    # existing validation windows never opens any held-out policy evaluations.
    run_dir = inside(root, report["selection"]["model_path"]).parent.parent
    selected = next(row for row in report["results"] if row["seed"] == report["selected_seed"])
    validation = {"split": manifest["validation"], "reference": read_json(run_dir / "validation_reference.json"),
                  "per_seed": [{"seed": selected["seed"], "evaluation": selected["selected"]["evaluation"]}]}
    final_checkpoint = read_json(run_dir / f"seed_{selected['seed']}" / "curve.json")[-1]
    final_validation = {**validation, "per_seed": [{"seed": selected["seed"], "evaluation": final_checkpoint["evaluation"]}]}
    loaded_path = selected_path
    for label, partition in {"validation": validation, "final_validation": final_validation, **report["evaluations"]}.items():
        model_path = final_checkpoint["model_path"] if label == "final_validation" else selected_path
        model_sha = final_checkpoint["model_sha256"] if label == "final_validation" else report["selection"]["model_sha256"]
        resolved_path = verify_hash(root, model_path, model_sha)
        if model_path != loaded_path:
            model = classes[config["algorithm"]].load(resolved_path, device="cpu")
            loaded_path = model_path
        source = history
        if label == "forward":
            forward = load_port_dataset(partition["dataset_id"])
            source = PortDataset(dataset_id="verified_v8_forward_composite", path=forward.path,
                timestamps=history.timestamps + forward.timestamps, values=np.vstack((history.values, forward.values)),
                metadata={"sha256": "replay_composite"}, factor_values=np.vstack((history.factor_values, forward.factor_values)),
                factor_availability=np.vstack((history.factor_availability, forward.factor_availability)))
        description = partition["split"]
        split = slice(description["start_row"], description["stop_row_exclusive"])
        expected = next(row["evaluation"] for row in partition["per_seed"] if row["seed"] == report["selected_seed"])
        env_class = ShoreBESSV8Env if config["discrete"] else ShoreBESSV8SACEnv
        for control, evaluation in ((True, expected), (False, partition["reference"])):
            for start, stored in zip(evaluation["starts"], evaluation["rows"], strict=True):
                env = env_class(source, split, config=config["physical_config"], normalization_slice=train,
                                episode_steps=168, carbon_price=config["carbon_price_cny_per_kg_constraint_multiplier"],
                                discrete=config["discrete"], training=False, record_trace=False)
                try:
                    for key, value in config["normalization"].items():
                        same(float(getattr(env, key)), value, "train scaler " + key)
                    obs, info = env.reset(options={"start_index": start})
                    require(info["start_index"] == start, "replay start was silently changed")
                    reward_sum = 0.0
                    for _ in range(168):
                        if control:
                            kwargs = {"action_masks": env.action_masks()} if isinstance(model, MaskablePPO) else {}
                            action = model.predict(obs, deterministic=True, **kwargs)[0]
                        else:
                            if config["discrete"]:
                                idle = np.flatnonzero(np.all(np.asarray(config["action_lattice"]) == 0, axis=1))
                                require(len(idle) == 1, "expected exactly one idle lattice action")
                                action = int(idle[0])
                            else:
                                action = np.zeros(2, dtype=np.float32)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        reward_sum += reward
                    require(terminated and not truncated, "replay episode did not terminate")
                    totals = env.totals
                    totals["learning_reward"] = reward_sum
                    for key, value in stored.items():
                        same(float(totals[key]), value, f"{label} replay {start}/{key}", atol=1e-5)
                finally:
                    env.close()
        output[label] = {"selected_seed": report["selected_seed"], "windows": len(expected["starts"]),
                         "checkpoint_step": final_checkpoint["step"] if label == "final_validation" else selected["selected"]["step"],
                         "model_path": model_path, "model_sha256": model_sha,
                         "loaded_model_path": str(resolved_path.relative_to(root)), "loaded_model_sha256": digest(resolved_path),
                         "zero_baseline_replayed": True, "all_persisted_metrics_reproduced": True}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    try:
        pointer = None
        if args.report is None:
            pointer = read_json(ROOT / "evidence/v8/shore_bess/latest.json")
            path = verify_hash(ROOT, pointer["report_path"], pointer["report_sha256"])
        else:
            path = args.report if args.report.is_absolute() else ROOT / args.report
        result = verify_report(path, pointer=pointer, replay=args.replay)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        print("SHORE_BESS_V8_INTEGRITY:PASS")
    except Exception as exc:
        print(json.dumps({"integrity_status": "FAIL", "reason": str(exc), "admission_not_modified": True}, ensure_ascii=False))
        print("SHORE_BESS_V8_INTEGRITY:FAIL")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
