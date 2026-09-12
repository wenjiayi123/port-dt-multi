"""Replay a frozen V8 actor against causal rules and action-channel ablations.

This script does not train, select or promote policies. It writes an append-only
comparison run tied to an existing formal report and its selected model hash.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.services.rl_model.shore_bess.v8_baselines import RuleBaseline, fit_training_hourly_profile
from app.services.rl_model.shore_bess.v8_environment import LATTICE, ShoreBESSV8Env, ShoreBESSV8SACEnv
from app.services.rl_training.datasets import file_sha256, load_port_dataset
from app.services.shore_bess_v8_evidence import BOUNDARY, ShoreBESSV8EvidenceService, verify_checkpoint_counters
from scripts.train_shore_bess_v8 import ROOT, business_gates, compare, joined_forward, official_source_period, write_json

ARMS = ("learned_policy", "idle", "carbon_aware_rule", "historical_peak_valley_quantized",
        "learned_no_bess_coordinate", "learned_no_flex_coordinate")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def ablate_action(action: Any, arm: str, discrete: bool):
    """Zero one requested neural coordinate; keep the shared safety projection."""
    if arm not in {"learned_no_bess_coordinate", "learned_no_flex_coordinate"}:
        return action
    index = 0 if arm == "learned_no_bess_coordinate" else 1
    if discrete:
        command = LATTICE[int(np.asarray(action).item())].copy()
        command[index] = 0.0
        matches = np.flatnonzero(np.all(LATTICE == command, axis=1))
        if len(matches) != 1:
            raise ValueError("ablated action is missing from the frozen lattice")
        return int(matches[0])
    command = np.asarray(action, dtype=np.float32).copy()
    if command.shape != (2,) or not np.isfinite(command).all():
        raise ValueError("finite two-dimensional neural coordinates required")
    command[index] = 0.0
    return command


def verify_formal_report(report_path: Path, expected_sha256: str | None) -> tuple[dict, str]:
    service = ShoreBESSV8EvidenceService(ROOT)
    report_path = service._path(relative(report_path))
    actual = file_sha256(report_path)
    if expected_sha256 is None:
        for pointer_name in ("offline_champion.json", "latest.json"):
            pointer_path = ROOT / "evidence/v8/shore_bess" / pointer_name
            if pointer_path.is_file():
                pointer = json.loads(pointer_path.read_text())
                if pointer.get("report_path") == relative(report_path):
                    expected_sha256 = pointer["report_sha256"]
                    break
    if expected_sha256 is None or actual != expected_sha256:
        raise ValueError("formal report requires a matching saved pointer or explicit --report-sha256")
    report = json.loads(report_path.read_text())
    if (report.get("schema") != "port-shore-bess-v8-report.v1"
            or report.get("status") not in {"ADMITTED_OFFLINE_RL", "CANDIDATE_NOT_ADMITTED"}
            or report["config"].get("pilot") is not False):
        raise ValueError("comparison requires a completed formal V8 report")
    for name, value in (("config.json", report["config"]), ("manifest.json", report["manifest"]),
                        ("selection.json", report["selection"])):
        if json.loads(report_path.with_name(name).read_text()) != value:
            raise ValueError("formal report artifact changed: " + name)
    selection = report["selection"]
    if selection.get("selection_access_to_test") is not False or selection.get("selection_access_to_forward") is not False:
        raise ValueError("formal actor was not frozen before evaluation")
    selected = next(row for row in report["results"] if row["seed"] == report["selected_seed"])
    if (selection["selected_seed"] != selected["seed"]
            or selection["model_path"] != selected["model_path"]
            or selection["model_sha256"] != selected["model_sha256"]):
        raise ValueError("selected actor differs from the frozen selection")
    for path, expected in report["manifest"]["source_sha256"].items():
        service._verify(path, expected)
        if file_sha256(report_path.parent / "source" / path) != expected:
            raise ValueError("frozen source snapshot changed: " + path)
    for path, expected in report.get("evidence_files_sha256", {}).items():
        service._verify(path, expected)
    for row in report["results"]:
        service._verify(row["model_path"], row["model_sha256"])
    if report.get("report_kind") == "formal_admission_of_immutable_pilot_checkpoints":
        service._verify_reused_training(report, report_path)
    if set(report["evaluations"]) != {"test", "forward"}:
        raise ValueError("comparison cannot open partitions absent from the formal report")
    for partition in report["evaluations"].values():
        for row in partition["per_seed"]:
            service._summary(partition, row)
            if row["gates"] != business_gates(row["evaluation"], row["comparison"], require_month_ci=True):
                raise ValueError("formal source-month/weekly admission gates differ from evaluation")
    return report, actual


def evaluate_arm(factory, actor, arm: str, starts: list[int], *, discrete: bool,
                 masked: bool, trace_path: Path | None = None) -> dict:
    rows, traces, requested_counts, windows = [], [], {}, []
    for window_index, start in enumerate(starts):
        env = factory()
        try:
            observation, reset_info = env.reset(options={"start_index": start})
            if reset_info["start_index"] != start:
                raise ValueError("comparison window was silently changed")
            accounts = {key: 0.0 for key in (
                "meter_energy_kwh", "baseline_energy_kwh", "charge_kwh", "discharge_kwh",
                "flex_deferred_kwh", "flex_repaid_kwh", "bess_energy_cost_delta_cny",
                "flex_energy_cost_delta_cny", "bess_carbon_delta_kg", "flex_carbon_delta_kg",
                "baseline_energy_cost_cny", "baseline_carbon_kg", "baseline_peak_kw",
                "learning_reward", "nonzero_executed_bess_steps", "nonzero_executed_flex_steps")}
            while True:
                options = {"action_masks": env.action_masks()} if masked else {}
                neural_action, _ = actor.predict(observation, deterministic=True, **options)
                action = ablate_action(neural_action, arm, discrete)
                key = (str(int(np.asarray(action).item())) if discrete else
                       ",".join("idle_band" if abs(value) <= env.action_deadband else "positive" if value > 0 else "negative"
                                for value in np.asarray(action)))
                requested_counts[key] = requested_counts.get(key, 0) + 1
                next_observation, reward, terminated, truncated, info = env.step(action)
                ctx = info["context"]
                bess, flex = info["final_action"]["bess_kw"], info["final_action"]["flex_kw"]
                price, carbon = ctx["price_cny_per_kwh"], ctx["carbon_kg_per_kwh"]
                accounts["meter_energy_kwh"] += info["pcc_kw"]
                accounts["baseline_energy_kwh"] += ctx["base_load_kw"]
                accounts["charge_kwh"] += max(-bess, 0)
                accounts["discharge_kwh"] += max(bess, 0)
                accounts["flex_deferred_kwh"] += max(-flex, 0)
                accounts["flex_repaid_kwh"] += max(flex, 0)
                accounts["bess_energy_cost_delta_cny"] -= bess * price
                accounts["flex_energy_cost_delta_cny"] += flex * price
                accounts["bess_carbon_delta_kg"] -= bess * carbon
                accounts["flex_carbon_delta_kg"] += flex * carbon
                accounts["baseline_energy_cost_cny"] += ctx["base_load_kw"] * price
                accounts["baseline_carbon_kg"] += ctx["base_load_kw"] * carbon
                accounts["baseline_peak_kw"] = max(accounts["baseline_peak_kw"], ctx["base_load_kw"])
                accounts["learning_reward"] += reward
                accounts["nonzero_executed_bess_steps"] += float(abs(bess) > 1e-6)
                accounts["nonzero_executed_flex_steps"] += float(abs(flex) > 1e-6)
                if window_index == 0:
                    traces.append({"window_start": start, "hour": env._step - 1,
                                   "observation": observation.tolist(),
                                   "neural_or_rule_action": np.asarray(neural_action).tolist(),
                                   "requested_action_after_ablation": np.asarray(action).tolist(),
                                   "learning_reward": reward, **info})
                observation = next_observation
                if terminated or truncated:
                    break
            totals = {key: float(value) for key, value in env.totals.items() if isinstance(value, (int, float, np.number))}
            accounts["energy_reconciliation_residual_kwh"] = (
                accounts["meter_energy_kwh"] - accounts["baseline_energy_kwh"] - accounts["charge_kwh"]
                + accounts["discharge_kwh"] - accounts["flex_repaid_kwh"] + accounts["flex_deferred_kwh"])
            accounts["carbon_reconciliation_residual_kg"] = (
                totals["carbon_kg"] - accounts["baseline_carbon_kg"]
                - accounts["bess_carbon_delta_kg"] - accounts["flex_carbon_delta_kg"])
            demand_rate = env.config["grid"]["demand_charge_cny_per_kw_month"] * env.episode_steps / (24 * 30.4375)
            accounts["demand_cost_delta_cny"] = (totals["peak_kw"] - accounts["baseline_peak_kw"]) * demand_rate
            accounts["financial_reconciliation_residual_cny"] = (
                totals["total_cost_cny"] - accounts["baseline_energy_cost_cny"] - accounts["baseline_peak_kw"] * demand_rate
                - accounts["bess_energy_cost_delta_cny"] - accounts["flex_energy_cost_delta_cny"]
                - totals["degradation_cost_cny"] - accounts["demand_cost_delta_cny"])
            for key in ("energy_reconciliation_residual_kwh", "carbon_reconciliation_residual_kg", "financial_reconciliation_residual_cny"):
                if abs(accounts[key]) > 1e-6:
                    raise ValueError("executed command accounting does not reconcile: " + key)
            totals.update(accounts)
            if not all(np.isfinite(value) for value in totals.values()):
                raise ValueError("comparison contains non-finite metrics")
            rows.append(totals)
            first_timestamp = str(env.timestamps[start])
            windows.append({"start_index": start, "first_timestamp": first_timestamp,
                            "last_timestamp": str(env.timestamps[start + env.episode_steps - 1]),
                            "source_month": first_timestamp[:7],
                            "source_period": official_source_period(first_timestamp)})
        finally:
            env.close()
    if trace_path is not None:
        write_json(trace_path, {"arm": arm, "window_start": starts[0], "actual_steps": len(traces), "trace": traces})
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {"mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
            "rows": rows, "starts": starts, "episode_hours": env.episode_steps,
            "windows": windows,
            "requested_action_counts": requested_counts, "window_overlap": False,
            "requested_action_histogram_kind": "lattice_indices" if discrete else "neural_coordinate_sign_and_idle_band",
            "attribution_boundary": "Executed-command arithmetic decomposition; peak-cost effects are joint. Ablations zero requested coordinates while retaining safety recovery."}


def run_comparisons(report_path: Path, expected_sha256: str | None = None, run_id: str | None = None) -> dict:
    import torch
    torch.set_num_threads(1)
    report, report_sha = verify_formal_report(report_path, expected_sha256)
    cfg, manifest, selection = report["config"], report["manifest"], report["selection"]
    history = load_port_dataset(manifest["dataset_id"])
    forward_id = report["evaluations"]["forward"]["dataset_id"]
    forward = load_port_dataset(forward_id)
    if (history.fingerprint != manifest["dataset_sha256"]
            or forward.fingerprint != report["evaluations"]["forward"]["dataset_sha256"]):
        raise ValueError("comparison dataset differs from formal evaluation")
    train = slice(manifest["train"]["start_row"], manifest["train"]["stop_row_exclusive"])
    profile = fit_training_hourly_profile(history, train, config=cfg["physical_config"])
    policies = {"idle": RuleBaseline(profile, kind="idle"),
                "carbon_aware_rule": RuleBaseline(profile, kind="carbon_aware"),
                "historical_peak_valley_quantized": RuleBaseline(profile, kind="legacy_peak_valley")}
    actor = ShoreBESSV8EvidenceService._actor(str(ROOT / selection["model_path"]), selection["model_sha256"], cfg["algorithm"])
    frozen = next(row for row in report["results"] if row["seed"] == report["selected_seed"])["selected"]
    verify_checkpoint_counters(actor, frozen)
    comparison_id = run_id or "comparators-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(comparison_id).name != comparison_id or comparison_id in {".", ".."}:
        raise ValueError("comparison run id must be a single directory name")
    output = ROOT / "evidence/v8/shore_bess/comparisons" / comparison_id
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / "app/services/rl_model/shore_bess/v8_baselines.py",
                    ROOT / "app/services/rl_model/shore_bess/v8_environment.py",
                    ROOT / "app/services/rl_model/shore_bess/v3_environment.py"]
    source_hashes = {relative(path): file_sha256(path) for path in source_paths}
    for path in source_paths:
        destination = output / "source" / relative(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    protected = {relative(path): file_sha256(path) for directory in ("evidence/v3/shore_bess", "evidence/v7/shore_bess", "evidence/v8/shore_bess")
                 for path in (ROOT / directory).glob("*.json")}
    write_json(output / "manifest.json", {"comparison_id": comparison_id,
               "formal_report_path": relative(report_path), "formal_report_sha256": report_sha,
               "selected_seed": report["selected_seed"], "selected_model_sha256": selection["model_sha256"],
               "source_sha256": source_hashes, "protected_pointer_sha256": protected,
               "rule_provenance": {name: rule.provenance() for name, rule in policies.items()},
               "training_updates": 0, "selection_updates": 0, "promotion_attempts": 0, **BOUNDARY})
    partitions = {}
    for label, dataset in (("test", history), ("forward", joined_forward(history, forward))):
        partition = report["evaluations"][label]
        split_meta = partition["split"]
        split = slice(split_meta["start_row"], split_meta["stop_row_exclusive"])
        starts = partition["reference"]["starts"]
        if any(right - left < cfg["episode_hours"] for left, right in zip(starts, starts[1:])):
            raise ValueError("formal evaluation windows overlap")
        arms = {}
        for name in ARMS:
            learned = name.startswith("learned")
            discrete = cfg["discrete"] if learned else True
            env_cls = ShoreBESSV8Env if discrete else ShoreBESSV8SACEnv
            factory = lambda: env_cls(dataset, split, config=cfg["physical_config"], normalization_slice=train,
                episode_steps=cfg["episode_hours"], carbon_price=cfg["carbon_price_cny_per_kg_constraint_multiplier"],
                discrete=discrete, training=False, record_trace=False)
            arms[name] = evaluate_arm(factory, actor if learned else policies[name], name, starts,
                discrete=discrete, masked=learned and cfg["algorithm"] == "sb3_contrib.MaskablePPO",
                trace_path=output / f"{label}_{name}_first_window_trace.json")
            print(json.dumps({"partition": label, "arm": name,
                              "cost_cny": arms[name]["mean"]["total_cost_cny"],
                              "carbon_kg": arms[name]["mean"]["carbon_kg"],
                              "physical_violations": arms[name]["mean"]["physical_power_violations"]}), flush=True)
        selected_evaluation = next(row for row in partition["per_seed"] if row["seed"] == report["selected_seed"])["evaluation"]
        for actual, expected in zip(arms["learned_policy"]["rows"], selected_evaluation["rows"], strict=True):
            for key in ("total_cost_cny", "carbon_kg", "peak_kw", "terminal_soc_error", "terminal_flex_backlog_kwh", "physical_power_violations"):
                if not np.isclose(actual[key], expected[key], rtol=1e-12, atol=1e-6):
                    raise ValueError("selected learned policy replay differs from saved formal result: " + key)
        for name, evaluation in arms.items():
            evaluation["comparison_vs_idle"] = compare(evaluation, arms["idle"])
            evaluation["comparison_vs_carbon_aware_rule"] = compare(evaluation, arms["carbon_aware_rule"])
            evaluation["business_and_safety_gates_vs_idle"] = business_gates(evaluation, evaluation["comparison_vs_idle"], require_month_ci=True)
        partitions[label] = {"split": split_meta, "dataset_sha256": partition["dataset_sha256"],
                             "benchmark_status": partition["status"], "arms": arms,
                             "learned_replay_matches_frozen_report": True}
        write_json(output / f"{label}_comparisons.json", partitions[label])
    checks = {"source_unchanged": all(file_sha256(ROOT / path) == expected for path, expected in source_hashes.items()),
              "formal_report_unchanged": file_sha256(report_path) == report_sha,
              "all_pointers_unchanged": all(file_sha256(ROOT / path) == expected for path, expected in protected.items()),
              "no_training_selection_or_promotion": True,
              "same_persisted_windows_all_arms": True, "reconciled_executed_command_accounting": True}
    result = {"schema": "port-shore-bess-v8-comparators.v1", "comparison_id": comparison_id,
              "formal_report_path": relative(report_path), "formal_report_sha256": report_sha,
              "selected_model_sha256": selection["model_sha256"], "selected_seed": report["selected_seed"],
              "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "partitions": partitions, "training_steps": 0, "optimizer_updates": 0,
              "claim_boundary": "Previously used public engineering benchmarks. Rules are conventional comparators; channel ablations do not retrain policies. No measured port savings or production authority.",
              "artifact_sha256": {relative(path): file_sha256(path) for path in sorted(output.glob("*.json"))},
              **BOUNDARY}
    write_json(output / "report.json", result)
    if not all(checks.values()):
        raise RuntimeError("comparison source or protected evidence changed during replay")
    return {"status": result["status"], "report_path": relative(output / "report.json"),
            "report_sha256": file_sha256(output / "report.json"), "comparison_id": comparison_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--report-sha256")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(run_comparisons(args.report.resolve(), args.report_sha256, args.run_id)), flush=True)


if __name__ == "__main__":
    main()
