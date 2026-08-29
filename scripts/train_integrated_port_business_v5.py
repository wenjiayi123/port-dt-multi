"""Train, select and blind-test the V5 integrated port-business policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.services.rl_training.baselines import (
    EngineeringCurrentOpsRulePolicy,
    FCFSNeutralPolicy,
    IntegratedCurrentOpsRulePolicy,
)
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.integrated_environment import IntegratedPortOperationsEnv
from app.services.rl_training.statistics import bootstrap_summary, summarize_metric_rows
from app.services.rl_training.trainer import TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config/rl_business_profiles_v3.json"
EVIDENCE_ROOT = ROOT / "evidence/v5/integrated_business"

METRIC_WEIGHTS = {
    "gate_queue_truck_hours": 0.18,
    "intermodal_backlog_teu_hours": 0.15,
    "marine_service_wait_vessel_hours": 0.15,
    "reefer_risk_hours": 0.14,
    "maintenance_risk_hours": 0.14,
    "regulatory_delay_teu_hours": 0.08,
    "cost_per_teu": 0.06,
    "carbon_kg_per_teu": 0.04,
    "service_completion_ratio": 0.06,
}
METRIC_DIRECTIONS = {
    **{name: "lower" for name in METRIC_WEIGHTS},
    "service_completion_ratio": "higher",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved.resolve().relative_to(ROOT.resolve()))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def wait(job_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = TRAINING_MANAGER.status(job_id)
        if status.get("status") in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "INTERRUPTED",
        }:
            return status
        time.sleep(0.5)
    raise TimeoutError(f"training timed out: {job_id}")


def legacy_hashes() -> dict[str, str]:
    paths = [
        *ROOT.glob("data/rl/runs/*/model.zip"),
        *ROOT.glob("data/rl/runs/*/manifest.json"),
        *ROOT.glob("evidence/v3/**/*.json"),
        *ROOT.glob("evidence/v4/**/*.json"),
    ]
    return {
        relative(path): sha256(path)
        for path in sorted(paths)
        if path.is_file()
    }


def make_env(
    dataset: Any,
    data_slice: slice,
    train_slice: slice,
    config: dict[str, Any],
    *,
    normalization_dataset: Any | None = None,
    record_trace: bool = False,
) -> IntegratedPortOperationsEnv:
    return IntegratedPortOperationsEnv(
        dataset,
        data_slice,
        action_mode="continuous",
        episode_steps=config["episode_steps"],
        seed=config["seed"],
        demand_cap_kw=config["demand_cap_kw"],
        reward_weights=config["reward_weights"],
        projection_penalty_weight=float(
            config.get("projection_penalty_weight") or 0.0
        ),
        regulatory_delay_penalty_weight=float(
            config.get("regulatory_delay_penalty_weight") or 0.35
        ),
        integrated_reward_weights=dict(
            config.get("integrated_reward_weights") or {}
        ),
        port_profile=config["port_profile"],
        normalization_slice=train_slice,
        normalization_dataset=normalization_dataset,
        training=False,
        record_trace=record_trace,
    )


def evaluate_windows(
    policy: Any,
    env: IntegratedPortOperationsEnv,
    start_indices: list[int],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for episode, start_index in enumerate(start_indices):
        observation, _ = env.reset(
            seed=12000 + episode, options={"start_index": int(start_index)}
        )
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(action)
        row = env.totals
        row["delay_index_mean"] = row.pop("delay") / max(1, env.episode_steps)
        row["guardrail_violation_rate"] = row.pop("violations") / max(
            1, env.episode_steps
        )
        rows.append({key: float(value) for key, value in row.items()})
    return rows


def metric_means(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


def evaluate_forward_policy(
    job_id: str,
    training_dataset: Any,
    forward_dataset: Any,
    train_slice: slice,
    config: dict[str, Any],
    episodes: int,
) -> dict[str, Any]:
    env = make_env(
        forward_dataset,
        slice(0, forward_dataset.rows),
        train_slice,
        config,
        normalization_dataset=training_dataset,
        record_trace=True,
    )
    policy = TRAINING_MANAGER._load_policy(
        config, TRAINING_MANAGER.run_dir(job_id), env
    )
    max_start = max(0, len(env.segment) - config["episode_steps"] - 1)
    starts = np.linspace(
        0,
        max_start,
        num=min(max(1, episodes), max_start + 1),
        dtype=int,
    ).tolist()
    rows: list[dict[str, float]] = []
    first_trace: list[dict[str, Any]] = []
    for episode, start_index in enumerate(starts):
        observation, _ = env.reset(
            seed=config["seed"] + 3000 + episode,
            options={"start_index": int(start_index)},
        )
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(action)
        row = env.totals
        row["delay_index_mean"] = row.pop("delay") / max(1, env.episode_steps)
        row["guardrail_violation_rate"] = row.pop("violations") / max(
            1, env.episode_steps
        )
        rows.append({key: float(value) for key, value in row.items()})
        if episode == 0:
            first_trace = list(env.trace)
    env.close()
    return {
        "job_id": job_id,
        "dataset_id": forward_dataset.dataset_id,
        "dataset_sha256": forward_dataset.fingerprint,
        "split": "independent_2026_forward_public_challenge_never_used_for_selection",
        "episode_metrics": rows,
        "metrics": metric_means(rows),
        "uncertainty": summarize_metric_rows(rows, seed=config["seed"]),
        "evaluation_protocol": {
            "deterministic_policy": True,
            "window_start_indices": starts,
            "normalization_fit": "2024_2025_chronological_training_rows_only",
            "forward_rows_not_used_for_training_validation_or_selection": True,
        },
        "render": {
            "type": "trajectory",
            "frames": first_trace,
            "frame_count": len(first_trace),
        },
        "evaluated_at": now(),
    }


def compare(
    candidate: list[dict[str, float]], baseline: list[dict[str, float]]
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("paired comparison requires equal window counts")
    metric_values: dict[str, list[float]] = {
        name: [] for name in METRIC_WEIGHTS
    }
    composite_values = []
    for current, reference in zip(candidate, baseline):
        weighted = 0.0
        for metric, weight in METRIC_WEIGHTS.items():
            denominator = max(abs(reference[metric]), 1e-9)
            improvement = (
                (current[metric] - reference[metric]) / denominator
                if METRIC_DIRECTIONS[metric] == "higher"
                else (reference[metric] - current[metric]) / denominator
            )
            improvement = float(np.clip(improvement, -2.0, 2.0))
            metric_values[metric].append(improvement)
            weighted += weight * improvement
        composite_values.append(weighted)
    return {
        "paired_windows": len(candidate),
        "metric_relative_improvement": {
            metric: bootstrap_summary(values, seed=20260826)
            for metric, values in metric_values.items()
        },
        "integrated_business_score_relative_improvement": bootstrap_summary(
            composite_values, seed=20260826
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--seeds", default="86,186,286")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--reuse-jobs",
        default="",
        help="comma-separated completed job IDs aligned with --seeds; no retraining",
    )
    parser.add_argument(
        "--final-dataset-id",
        default="",
        help="independent forward dataset used once after validation selection",
    )
    args = parser.parse_args()
    seeds = sorted({int(value) for value in args.seeds.split(",") if value.strip()})
    reuse_jobs = [
        value.strip() for value in args.reuse_jobs.split(",") if value.strip()
    ]
    if len(seeds) < 3:
        parser.error("at least three distinct seeds are required")
    if args.steps < 10000:
        parser.error("at least 10,000 optimizer steps are required")
    if args.episodes < 10:
        parser.error("at least 10 paired validation and blind-test windows are required")
    if reuse_jobs and len(reuse_jobs) != len(seeds):
        parser.error("--reuse-jobs must contain one job ID for each sorted seed")

    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    business = profiles["profiles"]["integrated_business_value_v5"]
    dataset = load_port_dataset("public_cn_sha_integrated_scenario_v5")
    train_slice, validation_slice, blind_slice = dataset.split_three_way(0.2, 0.1)
    legacy_before = legacy_hashes()
    runs: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds):
        if reuse_jobs:
            started = {"job_id": reuse_jobs[seed_index]}
            status = TRAINING_MANAGER.status(started["job_id"])
        else:
            started = TRAINING_MANAGER.start(
                {
                "algorithm": "sac",
                "dataset_id": dataset.dataset_id,
                "environment_version": "port_ops_v5",
                "port_profile_id": "cn_sha_integrated_scenario_v5",
                "business_profile_id": "integrated_business_value_v5",
                "reward_weights": business["reward_weights"],
                "regulatory_delay_penalty_weight": business[
                    "regulatory_delay_penalty_weight"
                ],
                "integrated_reward_weights": business[
                    "integrated_reward_weights"
                ],
                "projection_penalty_weight": 0.15,
                "total_steps": args.steps,
                "episode_hours": 48,
                "episode_steps": 48,
                "test_ratio": 0.2,
                "validation_ratio": 0.1,
                "batch_size": 256,
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "replay_buffer": max(100000, args.steps * 3),
                "seed": seed,
                }
            )
            status = wait(started["job_id"], args.timeout)
        if status.get("status") not in {"COMPLETED", "EVALUATED"}:
            runs.append(
                {
                    "job_id": started["job_id"],
                    "seed": seed,
                    "status": status.get("status"),
                    "error": status.get("error"),
                }
            )
            continue
        validation = TRAINING_MANAGER.evaluate_split_evidence(
            started["job_id"],
            split_name="validation",
            episodes=args.episodes,
            persist=True,
        )
        config = json.loads(
            (TRAINING_MANAGER.run_dir(started["job_id"]) / "config.json").read_text(
                encoding="utf-8"
            )
        )
        starts = [
            int(value)
            for value in validation["evaluation_protocol"]["window_start_indices"]
        ]
        baseline_env = make_env(dataset, validation_slice, train_slice, config)
        fcfs_rows = evaluate_windows(
            FCFSNeutralPolicy(IntegratedPortOperationsEnv.ACTION_DIMENSIONS),
            baseline_env,
            starts,
        )
        baseline_env.close()
        validation_comparison = compare(
            [
                {key: float(value) for key, value in row.items()}
                for row in validation["episode_metrics"]
            ],
            fcfs_rows,
        )
        model_path = TRAINING_MANAGER.run_dir(started["job_id"]) / "model.zip"
        record = {
            "job_id": started["job_id"],
            "seed": seed,
            "status": "COMPLETED",
            "model_path": relative(model_path),
            "model_sha256": sha256(model_path),
            "validation_metrics": validation["metrics"],
            "validation_uncertainty": validation["uncertainty"],
            "validation_vs_fcfs": validation_comparison,
            "render_calls_during_training": status.get("rendering", {}).get(
                "render_calls", 0
            ),
        }
        runs.append(record)
        print(
            json.dumps(
                {
                    "job_id": record["job_id"],
                    "seed": seed,
                    "validation_reward": record["validation_metrics"]["reward"],
                    "validation_business_score": validation_comparison[
                        "integrated_business_score_relative_improvement"
                    ],
                    "guardrail_violation_rate": record["validation_metrics"][
                        "guardrail_violation_rate"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    completed = [row for row in runs if row.get("status") == "COMPLETED"]
    if not completed:
        raise RuntimeError("all V5 training runs failed; failure artifacts were preserved")
    completed.sort(
        key=lambda row: (
            float(row["validation_metrics"]["guardrail_violation_rate"]),
            -float(
                row["validation_vs_fcfs"][
                    "integrated_business_score_relative_improvement"
                ]["ci_low"]
            ),
            -float(row["validation_metrics"]["reward"]),
        )
    )
    selected = completed[0]
    selected_job_id = str(selected["job_id"])
    selected_config = json.loads(
        (TRAINING_MANAGER.run_dir(selected_job_id) / "config.json").read_text(
            encoding="utf-8"
        )
    )
    if args.final_dataset_id:
        final_dataset = load_port_dataset(args.final_dataset_id)
        blind = evaluate_forward_policy(
            selected_job_id,
            dataset,
            final_dataset,
            train_slice,
            selected_config,
            args.episodes,
        )
        evaluation_dataset = final_dataset
        evaluation_slice = slice(0, final_dataset.rows)
    else:
        blind = TRAINING_MANAGER.evaluate(selected_job_id, episodes=args.episodes)
        evaluation_dataset = dataset
        evaluation_slice = blind_slice
    candidate_rows = [
        {key: float(value) for key, value in row.items()}
        for row in blind["episode_metrics"]
    ]
    starts = [
        int(value)
        for value in blind["evaluation_protocol"]["window_start_indices"]
    ]
    baseline_specs = {
        "fcfs_neutral": FCFSNeutralPolicy(
            IntegratedPortOperationsEnv.ACTION_DIMENSIONS
        ),
        "integrated_current_operations_rule_proxy": IntegratedCurrentOpsRulePolicy(
            EngineeringCurrentOpsRulePolicy(
                action_dim=5,
                episode_steps=selected_config["episode_steps"],
                soc_min=selected_config["port_profile"]["control_limits"]["soc_min"],
                soc_max=selected_config["port_profile"]["control_limits"]["soc_max"],
                initial_soc=0.55,
                bess_capacity_kwh=selected_config["port_profile"]["assets"]["bess_capacity_kwh"],
                bess_power_kw=selected_config["port_profile"]["assets"]["bess_power_kw"],
                step_hours=selected_config["step_hours"],
            )
        ),
    }
    baseline_rows: dict[str, list[dict[str, float]]] = {}
    baseline_parameters: dict[str, Any] = {}
    for name, policy in baseline_specs.items():
        env = make_env(
            evaluation_dataset,
            evaluation_slice,
            train_slice,
            selected_config,
            normalization_dataset=(
                dataset if evaluation_dataset.dataset_id != dataset.dataset_id else None
            ),
        )
        baseline_rows[name] = evaluate_windows(policy, env, starts)
        baseline_parameters[name] = policy.parameters()
        env.close()
    comparisons = {
        name: compare(candidate_rows, rows)
        for name, rows in baseline_rows.items()
    }
    candidate_means = metric_means(candidate_rows)
    fcfs = comparisons["fcfs_neutral"]
    rule = comparisons["integrated_current_operations_rule_proxy"]
    fcfs_metrics = fcfs["metric_relative_improvement"]
    gate_checks = {
        "integrated_business_value_95ci_above_2_percent_vs_fcfs": bool(
            fcfs["integrated_business_score_relative_improvement"]["ci_low"]
            > 0.02
        ),
        "integrated_business_value_mean_nonnegative_vs_fixed_rule": bool(
            rule["integrated_business_score_relative_improvement"]["mean"]
            >= 0.0
        ),
        "gate_congestion_non_degradation_vs_fcfs": bool(
            fcfs_metrics["gate_queue_truck_hours"]["mean"] >= 0.0
        ),
        "intermodal_backlog_non_degradation_vs_fcfs": bool(
            fcfs_metrics["intermodal_backlog_teu_hours"]["mean"] >= 0.0
        ),
        "marine_wait_non_degradation_vs_fcfs": bool(
            fcfs_metrics["marine_service_wait_vessel_hours"]["mean"] >= 0.0
        ),
        "reefer_and_maintenance_risk_non_degradation_vs_fcfs": bool(
            fcfs_metrics["reefer_risk_hours"]["mean"] >= 0.0
            and fcfs_metrics["maintenance_risk_hours"]["mean"] >= 0.0
        ),
        "cost_regression_within_2_percent_95ci": bool(
            fcfs_metrics["cost_per_teu"]["ci_low"] >= -0.02
        ),
        "zero_guardrail_violations": bool(
            candidate_means["guardrail_violation_rate"] <= 1e-12
        ),
        "terminal_soc_recovered": bool(
            candidate_means["terminal_soc_error"] <= 1e-6
        ),
        "three_seed_training_completed": len(completed) >= 3,
        "training_rendering_disabled": all(
            int(row.get("render_calls_during_training") or 0) == 0
            for row in completed
        ),
        "blind_trace_generated_after_selection": bool(
            blind["render"]["frame_count"] > 0
        ),
    }
    admitted = all(gate_checks.values())

    legacy_after = {path: sha256(ROOT / path) for path in legacy_before}
    if legacy_after != legacy_before:
        raise RuntimeError("a pre-existing V3/V4 model or evidence artifact changed")
    run_id = "integrated-business-v5-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = EVIDENCE_ROOT / "runs" / run_id
    report = {
        "schema": "port-dt-integrated-business-evidence.v1",
        "run_id": run_id,
        "status": "ADMITTED_OFFLINE_CHAMPION" if admitted else "BLOCKED_RETAINED_CANDIDATE",
        "generated_at": now(),
        "evidence_label": "PUBLIC_AGGREGATE_REANALYSIS_PLUS_PREDECLARED_ENGINEERING_PORT_WIDE_SCENARIO",
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "artifact": relative(dataset.path),
            "sha256": dataset.fingerprint,
            "rows": dataset.rows,
            "independent_source_observations": int(
                dataset.metadata.get("independent_source_observations") or 0
            ),
            "train_rows": train_slice.stop - train_slice.start,
            "validation_rows": validation_slice.stop - validation_slice.start,
            "blind_test_rows": blind_slice.stop - blind_slice.start,
            "chronological_no_shuffle": True,
            "field_provenance": dataset.metadata.get("field_provenance"),
            "site_replacement_contract": dataset.metadata.get(
                "site_replacement_contract"
            ),
        },
        "final_evaluation_dataset": {
            "dataset_id": evaluation_dataset.dataset_id,
            "artifact": relative(evaluation_dataset.path),
            "sha256": evaluation_dataset.fingerprint,
            "rows": evaluation_dataset.rows,
            "independent_forward_challenge": bool(args.final_dataset_id),
            "never_used_for_training_validation_or_selection": bool(
                args.final_dataset_id
            ),
            "normalization_fit": "2024_2025_chronological_training_rows_only",
        },
        "contract": {
            "environment_version": "port_ops_v5",
            "observation_dimensions": IntegratedPortOperationsEnv.OBSERVATION_DIMENSIONS,
            "action_dimensions": IntegratedPortOperationsEnv.ACTION_DIMENSIONS,
            "authority": "recommendation_only_no_release_navigation_or_actuator_authority",
            "hard_constraints_outside_rl": [
                "maritime_and_customs_release",
                "channel_closure_and_weather_stop",
                "under_keel_clearance",
                "dangerous_goods_segregation",
                "reefer_minimum_service",
                "maintenance_minimum_reserve",
                "grid_and_equipment_interlocks",
                "human_approval_and_execution",
            ],
            "safety_revision": IntegratedPortOperationsEnv.SAFETY_REVISION,
        },
        "training": {
            "implementation": "stable_baselines3.SAC",
            "real_optimizer": True,
            "steps_per_seed": args.steps,
            "seeds_requested": seeds,
            "runs": runs,
            "selection_protocol": "validation guardrail rate, then lower confidence bound of integrated business score versus FCFS; blind test sealed until selection",
            "selected_job_id": selected_job_id,
            "selected_seed": selected["seed"],
        },
        "blind_test": {
            "split": blind["split"],
            "paired_window_start_indices": starts,
            "window_count": len(starts),
            "episode_steps": selected_config["episode_steps"],
            "selected_metrics": candidate_means,
            "selected_uncertainty": blind["uncertainty"],
            "comparators": {
                name: {
                    "parameters": baseline_parameters[name],
                    "metrics": metric_means(baseline_rows[name]),
                }
                for name in baseline_rows
            },
            "paired_comparisons": comparisons,
        },
        "admission": {
            "checks": gate_checks,
            "passed": admitted,
            "offline_champion_saved": admitted,
            "model_registry_champion_alias_changed": False,
            "production_authority": False,
        },
        "legacy_preservation": {
            "checked_artifact_count": len(legacy_before),
            "preserved": True,
            "sha256_before": legacy_before,
            "sha256_after": legacy_after,
        },
        "limitations": [
            "Only official aggregate throughput and public reanalysis are public observations; added port-wide fields are a deterministic replaceable engineering scenario.",
            "The rule comparator is a predeclared engineering proxy, not measured Shanghai operator dispatch.",
            "No policy can release a vessel, approve navigation, override a hard constraint or actuate equipment.",
            "Field value claims require authorized source replacement, calibration, shadow operation, independent interlocks and human acceptance.",
        ],
    }
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    write_json(
        run_dir / "blind_window_metrics.json",
        {
            "selected_job_id": selected_job_id,
            "window_start_indices": starts,
            "candidate": candidate_rows,
            "baselines": baseline_rows,
        },
    )
    report_hash = sha256(report_path)
    latest = {
        "schema": "port-dt-integrated-business-latest.v1",
        "run_id": run_id,
        "status": report["status"],
        "report_path": relative(report_path),
        "report_sha256": report_hash,
        "selected_job_id": selected_job_id,
        "selected_model_path": selected["model_path"],
        "selected_model_sha256": selected["model_sha256"],
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.fingerprint,
        "final_evaluation_dataset_id": evaluation_dataset.dataset_id,
        "final_evaluation_dataset_sha256": evaluation_dataset.fingerprint,
        "production_authority": False,
        "updated_at": now(),
    }
    write_json(EVIDENCE_ROOT / "latest.json", latest)
    history_path = EVIDENCE_ROOT / "history_index.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(latest, ensure_ascii=False) + "\n")
    if admitted:
        write_json(
            EVIDENCE_ROOT / "offline_champion.json",
            {
                **latest,
                "champion_metrics": candidate_means,
                "business_score": fcfs[
                    "integrated_business_score_relative_improvement"
                ],
                "blind_window_data": relative(
                    run_dir / "blind_window_metrics.json"
                ),
                "promotion_boundary": "offline evidence champion only; production authority false",
            },
        )
    markdown = [
        "# V5 integrated port-business evidence",
        "",
        f"- Status: `{report['status']}`",
        f"- Dataset: `{dataset.dataset_id}` / `{dataset.fingerprint}`",
        f"- Selected: SAC seed `{selected['seed']}` / `{selected_job_id}`",
        f"- Training: {len(completed)} seeds x {args.steps:,} real optimizer steps; rendering disabled",
        f"- Blind test: {len(starts)} paired 48-hour windows",
        f"- Integrated score versus FCFS: {100 * fcfs['integrated_business_score_relative_improvement']['mean']:.2f}%",
        "- Boundary: offline public-data and engineering-scenario evidence; production authority false",
        "",
    ]
    markdown_path = run_dir / "report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    (run_dir / "report.sha256").write_text(
        f"{report_hash}  report.json\n{sha256(markdown_path)}  report.md\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": report["status"],
                "selected_job_id": selected_job_id,
                "report_path": relative(report_path),
                "business_score_vs_fcfs": fcfs[
                    "integrated_business_score_relative_improvement"
                ],
                "business_score_vs_fixed_rule": rule[
                    "integrated_business_score_relative_improvement"
                ],
                "gate_checks": gate_checks,
                "legacy_artifacts_preserved": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
