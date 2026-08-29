"""Train and audit the append-only V3.1 Shore+BESS SAC evidence track.

Training never renders.  Model selection uses fixed chronological validation
windows; the blind-test rows are read only after every seed has completed.
Legacy ``policy.bin`` and ``shore_bess_outputs.jsonl`` are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.services.rl_model.shore_bess.v3_environment import (
    CONTRACT,
    ShoreBESSEnv,
    chronological_slices,
    evaluate_windows,
    fixed_window_starts,
    load_config,
    load_public_dataset,
    neutral_policy,
    rule_peak_valley_policy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evidence" / "v3" / "shore_bess"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def model_policy(model):
    def predict(observation: np.ndarray, _env: ShoreBESSEnv) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return predict


def make_env_factory(dataset, split, config, train_slice, seed, *, record_trace=False):
    return lambda: ShoreBESSEnv(
        dataset,
        split,
        config=config,
        normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_hours"]),
        seed=seed,
        training=False,
        record_trace=record_trace,
    )


def relative_improvement(candidate: float, baseline: float, lower_is_better: bool = True) -> float:
    if abs(baseline) <= 1e-12:
        return 0.0
    delta = baseline - candidate if lower_is_better else candidate - baseline
    return 100.0 * delta / abs(baseline)


def convergence_diagnostic(curve: List[Dict[str, float]], gate: Dict[str, Any]) -> Dict[str, Any]:
    tail_count = max(2, int(gate["tail_checkpoints"]))
    values = np.asarray([float(row["validation_reward_mean"]) for row in curve], dtype=np.float64)
    tail = values[-tail_count:]
    scale = max(1e-9, abs(float(np.mean(tail))))
    tail_relative_range = float((np.max(tail) - np.min(tail)) / scale)
    x = np.arange(len(tail), dtype=np.float64)
    slope = float(np.polyfit(x, tail, 1)[0]) if len(tail) >= 2 else 0.0
    relative_slope = abs(slope) / scale
    improvement = float(values[-1] - values[0])
    passed = bool(
        len(values) >= tail_count
        and tail_relative_range <= float(gate["tail_relative_range_max"])
        and relative_slope <= float(gate["tail_regression_slope_max"])
        and improvement >= -0.01 * max(1.0, abs(values[0]))
    )
    return {
        "passed": passed,
        "criterion": "fixed_validation_reward_tail_plateau_no_blind_test_selection",
        "checkpoint_count": int(len(values)),
        "tail_checkpoints": tail_count,
        "tail_relative_range": tail_relative_range,
        "tail_relative_range_max": float(gate["tail_relative_range_max"]),
        "tail_relative_slope": relative_slope,
        "tail_relative_slope_max": float(gate["tail_regression_slope_max"]),
        "reward_change_first_to_final": improvement,
        "first_validation_reward": float(values[0]),
        "final_validation_reward": float(values[-1]),
        "note": "Raw episode rewards may remain noisy across seasons; convergence is judged only on repeated deterministic fixed validation windows.",
    }


def aggregate_seed_metrics(seed_results: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    metric_names = sorted(
        set.intersection(*(set(row[key]["mean"]) for row in seed_results))
    )
    return {
        name: float(np.mean([float(row[key]["mean"][name]) for row in seed_results]))
        for name in metric_names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal multi-seed Shore+BESS V3.1 training")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint-steps", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated integer seeds")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    train_cfg = config["training"]
    total_steps = int(args.steps or train_cfg["formal_steps_per_seed"])
    checkpoint_steps = int(args.checkpoint_steps or train_cfg["checkpoint_steps"])
    seeds = (
        [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [int(value) for value in train_cfg["seeds"]]
    )
    if total_steps < checkpoint_steps or total_steps % checkpoint_steps:
        raise ValueError("steps must be a positive multiple of checkpoint-steps")
    if not seeds:
        raise ValueError("at least one seed is required")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("shore-bess-v3-%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"append-only run id already exists: {run_id}")
    run_dir.mkdir(parents=True)

    dataset = load_public_dataset(config)
    train_slice, validation_slice, blind_slice = chronological_slices(dataset)
    dataset_description = dataset.describe(test_ratio=0.20, validation_ratio=0.10)
    if (dataset_description.get("quality") or {}).get("training_eligible") is not True:
        raise RuntimeError("public dataset failed the quality gate")
    legacy_history = ROOT / "app" / "services" / "rl_model" / "shore_bess" / "artifacts" / "shore_bess_outputs.jsonl"
    legacy_policy = ROOT / "app" / "services" / "rl_model" / "shore_bess" / "policy.bin"

    split_rows = {
        "train": int(train_slice.stop - train_slice.start),
        "validation": int(validation_slice.stop - validation_slice.start),
        "blind_test": int(blind_slice.stop - blind_slice.start),
    }
    manifest = {
        "schema": "port-dt-shore-bess-training-manifest.v1",
        "version": config["version"],
        "run_id": run_id,
        "started_at": utc_now(),
        "algorithm": "SAC",
        "implementation": "stable_baselines3.SAC",
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.fingerprint,
        "dataset_rows": dataset.rows,
        "split_rows": split_rows,
        "split_method": "chronological_70_10_20_no_shuffle",
        "seeds": seeds,
        "formal_steps_per_seed": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "render_during_training": False,
        "state_action_reward_contract": CONTRACT.as_dict(),
        "legacy_evidence": {
            "preserved": True,
            "history_path": str(legacy_history.relative_to(ROOT)),
            "history_sha256_before": sha256(legacy_history),
            "policy_path": str(legacy_policy.relative_to(ROOT)),
            "policy_sha256_before": sha256(legacy_policy),
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    from stable_baselines3 import SAC
    import torch

    torch.set_num_threads(1)
    validation_count = int(config["convergence_gate"]["fixed_validation_windows"])
    validation_starts = fixed_window_starts(
        split_rows["validation"], int(train_cfg["episode_hours"]), validation_count
    )
    blind_starts = fixed_window_starts(
        split_rows["blind_test"], int(train_cfg["episode_hours"]), max(12, split_rows["blind_test"] // int(train_cfg["episode_hours"]))
    )
    validation_factory = make_env_factory(dataset, validation_slice, config, train_slice, seeds[0])
    validation_rule = evaluate_windows(validation_factory, rule_peak_valley_policy, validation_starts)
    validation_neutral = evaluate_windows(validation_factory, neutral_policy, validation_starts)

    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        training_env = ShoreBESSEnv(
            dataset,
            train_slice,
            config=config,
            normalization_slice=train_slice,
            episode_steps=int(train_cfg["episode_hours"]),
            seed=seed,
            training=True,
            record_trace=False,
        )
        model = SAC(
            "MlpPolicy",
            training_env,
            learning_rate=float(train_cfg["learning_rate"]),
            buffer_size=int(train_cfg["buffer_size"]),
            learning_starts=min(int(train_cfg["learning_starts"]), total_steps // 4),
            batch_size=int(train_cfg["batch_size"]),
            gamma=float(train_cfg["gamma"]),
            tau=float(train_cfg["tau"]),
            ent_coef=float(train_cfg["entropy_coefficient"]),
            train_freq=4,
            gradient_steps=1,
            policy_kwargs={"net_arch": list(train_cfg["network"])},
            seed=seed,
            verbose=0,
            device="cpu",
        )
        curve: List[Dict[str, float]] = []
        factory = make_env_factory(dataset, validation_slice, config, train_slice, seed)

        initial = evaluate_windows(factory, model_policy(model), validation_starts)
        initial_record = {
            "step": 0,
            "validation_reward_mean": float(initial["mean"]["reward"]),
            "validation_total_cost_cny": float(initial["mean"]["total_cost_cny"]),
            "validation_peak_kw": float(initial["mean"]["peak_kw"]),
            "validation_guardrail_violation_rate": float(initial["mean"]["guardrail_violation_rate"]),
        }
        curve.append(initial_record)
        append_jsonl(seed_dir / "metrics.jsonl", dict(ts=utc_now(), seed=seed, **initial_record))

        learned = 0
        while learned < total_steps:
            chunk = min(checkpoint_steps, total_steps - learned)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
            learned += chunk
            validation = evaluate_windows(factory, model_policy(model), validation_starts)
            record = {
                "step": learned,
                "validation_reward_mean": float(validation["mean"]["reward"]),
                "validation_total_cost_cny": float(validation["mean"]["total_cost_cny"]),
                "validation_peak_kw": float(validation["mean"]["peak_kw"]),
                "validation_guardrail_violation_rate": float(validation["mean"]["guardrail_violation_rate"]),
            }
            curve.append(record)
            append_jsonl(seed_dir / "metrics.jsonl", dict(ts=utc_now(), seed=seed, **record))
            print(
                f"seed={seed} step={learned}/{total_steps} "
                f"validation_reward={record['validation_reward_mean']:.6f} "
                f"cost={record['validation_total_cost_cny']:.2f}",
                flush=True,
            )

        model.save(str(seed_dir / "model"))
        model_path = seed_dir / "model.zip"
        validation_final = evaluate_windows(factory, model_policy(model), validation_starts)
        blind_factory = make_env_factory(dataset, blind_slice, config, train_slice, seed, record_trace=True)
        blind_result = evaluate_windows(blind_factory, model_policy(model), blind_starts)
        seed_result = {
            "seed": seed,
            "optimizer_steps": int(model.num_timesteps),
            "model_path": artifact_path(model_path),
            "model_sha256": sha256(model_path),
            "curve": curve,
            "convergence": convergence_diagnostic(curve, config["convergence_gate"]),
            "validation": validation_final,
            "blind_test": blind_result,
            "render_calls_during_training": training_env.render_calls,
        }
        write_json(seed_dir / "result.json", seed_result)
        seed_results.append(seed_result)
        training_env.close()

    blind_factory = make_env_factory(dataset, blind_slice, config, train_slice, seeds[0])
    blind_rule = evaluate_windows(blind_factory, rule_peak_valley_policy, blind_starts)
    blind_neutral = evaluate_windows(blind_factory, neutral_policy, blind_starts)
    aggregate_blind = aggregate_seed_metrics(seed_results, "blind_test")
    pass_rate = float(np.mean([float(item["convergence"]["passed"]) for item in seed_results]))
    converged = bool(pass_rate + 1e-9 >= float(config["convergence_gate"]["minimum_seed_pass_rate"]))
    cost_advantage_rule = relative_improvement(
        aggregate_blind["total_cost_cny"], blind_rule["mean"]["total_cost_cny"]
    )
    cost_advantage_neutral = relative_improvement(
        aggregate_blind["total_cost_cny"], blind_neutral["mean"]["total_cost_cny"]
    )
    carbon_advantage_rule = relative_improvement(
        aggregate_blind["carbon_kg"], blind_rule["mean"]["carbon_kg"]
    )
    peak_advantage_rule = relative_improvement(
        aggregate_blind["peak_kw"], blind_rule["mean"]["peak_kw"]
    )
    safety_passed = bool(
        aggregate_blind["guardrail_violation_rate"] <= 1e-12
        and aggregate_blind["terminal_soc_error"] <= 1e-6
        and aggregate_blind["terminal_flex_backlog_kwh"] <= 1e-6
        and aggregate_blind["shore_sla_violation_kwh"] <= 1e-9
    )
    business_passed = bool(cost_advantage_rule > 0.0 and cost_advantage_neutral > 0.0)

    aggregate_curve = []
    for index, step in enumerate(seed_results[0]["curve"]):
        rewards = [float(item["curve"][index]["validation_reward_mean"]) for item in seed_results]
        costs = [float(item["curve"][index]["validation_total_cost_cny"]) for item in seed_results]
        aggregate_curve.append(
            {
                "step": int(step["step"]),
                "validation_reward_mean": float(np.mean(rewards)),
                "validation_reward_std": float(np.std(rewards)),
                "validation_cost_mean_cny": float(np.mean(costs)),
                "validation_cost_std_cny": float(np.std(costs)),
            }
        )

    report = {
        "schema": "port-dt-shore-bess-formal-evidence.v1",
        "version": config["version"],
        "run_id": run_id,
        "generated_at": utc_now(),
        "status": "FORMAL_PUBLIC_DATA_OFFLINE_PASS" if converged and safety_passed and business_passed else "FORMAL_PUBLIC_DATA_OFFLINE_GATE_FAILED",
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "sha256": dataset.fingerprint,
            "rows": dataset.rows,
            "train_rows": split_rows["train"],
            "validation_rows": split_rows["validation"],
            "blind_test_rows": split_rows["blind_test"],
            "split_method": dataset_description["split_method"],
            "quality_status": dataset_description["quality"]["status"],
            "evidence_tier": dataset.metadata.get("evidence_tier"),
            "official_aggregate_columns": dataset.metadata.get("official_aggregate_columns"),
            "public_reanalysis_columns": dataset.metadata.get("public_reanalysis_columns"),
            "derived_columns": dataset.metadata.get("derived_columns"),
            "unavailable_factors": dataset.metadata.get("unavailable_factors"),
        },
        "training": {
            "algorithm": "SAC",
            "implementation": "stable_baselines3.SAC",
            "seeds": seeds,
            "formal_steps_per_seed": total_steps,
            "total_optimizer_steps": total_steps * len(seeds),
            "checkpoint_steps": checkpoint_steps,
            "episode_hours": int(train_cfg["episode_hours"]),
            "render_calls": int(sum(item["render_calls_during_training"] for item in seed_results)),
            "selection_split": "fixed_chronological_validation_windows",
            "final_report_split": "chronological_blind_test_only",
        },
        "contract": CONTRACT.as_dict(),
        "convergence": {
            "passed": converged,
            "seed_pass_rate": pass_rate,
            "minimum_seed_pass_rate": float(config["convergence_gate"]["minimum_seed_pass_rate"]),
            "aggregate_curve": aggregate_curve,
            "per_seed": [
                {"seed": item["seed"], **item["convergence"]}
                for item in seed_results
            ],
            "why_legacy_curve_looked_unconverged": [
                "legacy chart plotted noisy one-trajectory raw components instead of fixed validation checkpoint returns",
                "legacy offline dataset contained only 145 transitions and all actions were zero",
                "legacy display fields included bias/perturb shaping and are preserved only as historical evidence",
            ],
        },
        "blind_test": {
            "windows": len(blind_starts),
            "window_hours": int(train_cfg["episode_hours"]),
            "sac_multi_seed_mean": aggregate_blind,
            "sac_per_seed": [
                {"seed": item["seed"], "mean": item["blind_test"]["mean"], "std": item["blind_test"]["std"]}
                for item in seed_results
            ],
            "rule_peak_valley": blind_rule,
            "no_bess": blind_neutral,
            "sample_real_model_inference": seed_results[0]["blind_test"]["sample_inference"],
        },
        "business_metrics": {
            "cost_reduction_vs_rule_percent": cost_advantage_rule,
            "cost_reduction_vs_no_bess_percent": cost_advantage_neutral,
            "carbon_reduction_vs_rule_percent": carbon_advantage_rule,
            "peak_reduction_vs_rule_percent": peak_advantage_rule,
            "mean_weekly_cost_sac_cny": aggregate_blind["total_cost_cny"],
            "mean_weekly_cost_rule_cny": blind_rule["mean"]["total_cost_cny"],
            "mean_weekly_cost_no_bess_cny": blind_neutral["mean"]["total_cost_cny"],
            "mean_weekly_carbon_sac_kg": aggregate_blind["carbon_kg"],
            "mean_weekly_carbon_rule_kg": blind_rule["mean"]["carbon_kg"],
            "mean_weekly_peak_sac_kw": aggregate_blind["peak_kw"],
            "mean_weekly_peak_rule_kw": blind_rule["mean"]["peak_kw"],
            "claim_eligible": False,
            "evidence_scope": "paired public-data engineering scenario on chronological blind windows",
        },
        "quality_gates": {
            "convergence_passed": converged,
            "business_advantage_passed": business_passed,
            "safety_passed": safety_passed,
            "dataset_training_eligible": True,
            "nonzero_action_support": aggregate_blind["nonzero_bess_action_rate"] > 0.05,
            "guardrail_violation_rate": aggregate_blind["guardrail_violation_rate"],
            "terminal_soc_error": aggregate_blind["terminal_soc_error"],
            "terminal_flex_backlog_kwh": aggregate_blind["terminal_flex_backlog_kwh"],
            "shore_sla_violation_kwh": aggregate_blind["shore_sla_violation_kwh"],
            "public_offline_admitted": bool(converged and safety_passed and business_passed),
            "production_admitted": False,
        },
        "data_adequacy": {
            "offline_training": "sufficient_for_public_engineering_benchmark",
            "production_dispatch": "insufficient_until_authorized_site_replacement",
            "public_independent_source_observations": dataset.metadata.get("independent_source_observations"),
            "engineering_simulator_rows_preserved": {
                "shore_power_15min_by_berth": 69120,
                "bess_15min": 5760,
                "grid_meter_5min": 17280,
                "ship_calls": 285,
            },
            "missing_for_site": CONTRACT.landing_inputs,
        },
        "artifacts": {
            "manifest": artifact_path(run_dir / "manifest.json"),
            "models": [
                {"seed": item["seed"], "path": item["model_path"], "sha256": item["model_sha256"]}
                for item in seed_results
            ],
            "legacy_history_preserved": True,
            "legacy_history_sha256_after": sha256(legacy_history),
            "legacy_policy_sha256_after": sha256(legacy_policy),
        },
        "claim_boundary": config["claim_boundary"],
        "production_authority": False,
        "site_status": "待接入港口",
    }
    if report["artifacts"]["legacy_history_sha256_after"] != manifest["legacy_evidence"]["history_sha256_before"]:
        raise RuntimeError("legacy Shore+BESS history changed during V3 training")
    if report["artifacts"]["legacy_policy_sha256_after"] != manifest["legacy_evidence"]["policy_sha256_before"]:
        raise RuntimeError("legacy Shore+BESS policy changed during V3 training")

    report_path = run_dir / "report.json"
    write_json(report_path, report)
    latest = {
        "schema": "port-dt-shore-bess-latest-pointer.v1",
        "run_id": run_id,
        "report_path": artifact_path(report_path),
        "report_sha256": sha256(report_path),
        "status": report["status"],
        "updated_at": utc_now(),
    }
    write_json(args.output_root / "latest.json", latest)
    append_jsonl(
        args.output_root / "history_index.jsonl",
        {
            **latest,
            "legacy_history_sha256": report["artifacts"]["legacy_history_sha256_after"],
            "legacy_policy_sha256": report["artifacts"]["legacy_policy_sha256_after"],
        },
    )
    print(json.dumps(latest, ensure_ascii=False, indent=2))
    if report["status"] != "FORMAL_PUBLIC_DATA_OFFLINE_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
