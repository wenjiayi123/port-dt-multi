"""Train, select and independently challenge the V6 coordinated port policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.services.rl_training.baselines import (
    CoordinatedCurrentOpsRulePolicy,
    EngineeringCurrentOpsRulePolicy,
    FCFSNeutralPolicy,
    IntegratedCurrentOpsRulePolicy,
)
from app.services.rl_training.coordinated_environment import (
    CoordinatedPortOperationsEnv,
)
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.statistics import bootstrap_summary, summarize_metric_rows
from app.services.rl_training.trainer import TRAINING_MANAGER
from scripts.train_integrated_port_business_v5 import (
    legacy_hashes,
    now,
    relative,
    sha256,
    wait,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config/rl_business_profiles_v3.json"
EVIDENCE_ROOT = ROOT / "evidence/v6/coordinated_business"
DATASET_ID = "public_cn_sha_integrated_scenario_v5"

METRIC_WEIGHTS = {
    "gate_queue_truck_hours": 0.10,
    "rail_backlog_teu_hours": 0.09,
    "barge_backlog_teu_hours": 0.08,
    "pilotage_wait_vessel_hours": 0.08,
    "towage_wait_vessel_hours": 0.08,
    "quay_move_backlog_hours": 0.08,
    "horizontal_move_backlog_hours": 0.06,
    "yard_move_backlog_hours": 0.06,
    "reefer_risk_hours": 0.06,
    "maintenance_risk_hours": 0.06,
    "regulatory_delay_teu_hours": 0.08,
    "cost_per_teu": 0.05,
    "carbon_kg_per_teu": 0.04,
    "service_completion_ratio": 0.04,
    "yard_crane_completion_ratio": 0.04,
}
METRIC_DIRECTIONS = {
    **{name: "lower" for name in METRIC_WEIGHTS},
    "service_completion_ratio": "higher",
    "yard_crane_completion_ratio": "higher",
}


def make_env(
    dataset: Any,
    data_slice: slice,
    train_slice: slice,
    config: dict[str, Any],
    *,
    normalization_dataset: Any | None = None,
    record_trace: bool = False,
) -> CoordinatedPortOperationsEnv:
    return CoordinatedPortOperationsEnv(
        dataset,
        data_slice,
        action_mode="continuous",
        episode_steps=config["episode_steps"],
        seed=config["seed"],
        demand_cap_kw=config["demand_cap_kw"],
        reward_weights=config["reward_weights"],
        projection_penalty_weight=float(config.get("projection_penalty_weight") or 0.0),
        regulatory_delay_penalty_weight=float(
            config.get("regulatory_delay_penalty_weight") or 0.35
        ),
        integrated_reward_weights=dict(config.get("integrated_reward_weights") or {}),
        coordinated_reward_weights=dict(config.get("coordinated_reward_weights") or {}),
        port_profile=config["port_profile"],
        normalization_slice=train_slice,
        normalization_dataset=normalization_dataset,
        training=False,
        record_trace=record_trace,
    )


def metric_means(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


def finalize_totals(env: CoordinatedPortOperationsEnv) -> dict[str, float]:
    row = env.totals
    row["delay_index_mean"] = row.pop("delay") / max(1, env.episode_steps)
    row["guardrail_violation_rate"] = row.pop("violations") / max(
        1, env.episode_steps
    )
    return {key: float(value) for key, value in row.items()}


def evaluate_windows(
    policy: Any,
    env: CoordinatedPortOperationsEnv,
    start_indices: list[int],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for episode, start_index in enumerate(start_indices):
        observation, _ = env.reset(
            seed=16000 + episode, options={"start_index": int(start_index)}
        )
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(action)
        rows.append(finalize_totals(env))
    return rows


def compare(
    candidate: list[dict[str, float]], baseline: list[dict[str, float]]
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("paired comparison requires equal window counts")
    values = {name: [] for name in METRIC_WEIGHTS}
    composite: list[float] = []
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
            values[metric].append(improvement)
            weighted += weight * improvement
        composite.append(weighted)
    return {
        "paired_windows": len(candidate),
        "metric_relative_improvement": {
            metric: bootstrap_summary(metric_values, seed=20260830)
            for metric, metric_values in values.items()
        },
        "coordinated_business_score_relative_improvement": bootstrap_summary(
            composite, seed=20260830
        ),
    }


def rule_policy(config: dict[str, Any]) -> CoordinatedCurrentOpsRulePolicy:
    return CoordinatedCurrentOpsRulePolicy(
        IntegratedCurrentOpsRulePolicy(
            EngineeringCurrentOpsRulePolicy(
                action_dim=5,
                episode_steps=config["episode_steps"],
                soc_min=config["port_profile"]["control_limits"]["soc_min"],
                soc_max=config["port_profile"]["control_limits"]["soc_max"],
                initial_soc=0.55,
                bess_capacity_kwh=config["port_profile"]["assets"]["bess_capacity_kwh"],
                bess_power_kw=min(
                    config["port_profile"]["assets"]["bess_power_kw"],
                    config["demand_cap_kw"] * 0.5,
                ),
                step_hours=config["step_hours"],
            )
        )
    )


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
            seed=config["seed"] + 6000 + episode,
            options={"start_index": int(start_index)},
        )
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(action)
        rows.append(finalize_totals(env))
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


def incumbent_score() -> float | None:
    path = EVIDENCE_ROOT / "offline_champion.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(
            payload["business_score_vs_fixed_rule"]["mean"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seeds", default="606,706,806")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--reuse-jobs", default="")
    parser.add_argument(
        "--final-dataset-id",
        default="public_cn_sha_integrated_forward_2026m05_v5",
    )
    args = parser.parse_args()
    seeds = sorted({int(value) for value in args.seeds.split(",") if value.strip()})
    reuse_jobs = [value.strip() for value in args.reuse_jobs.split(",") if value.strip()]
    if len(seeds) < 3:
        parser.error("at least three distinct seeds are required")
    if args.steps < 10000:
        parser.error("at least 10,000 optimizer steps are required")
    if args.episodes < 10:
        parser.error("at least 10 paired evaluation windows are required")
    if reuse_jobs and len(reuse_jobs) != len(seeds):
        parser.error("--reuse-jobs must contain one job ID per seed")

    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    business = profiles["profiles"]["coordinated_business_value_v6"]
    dataset = load_port_dataset(DATASET_ID)
    train_slice, validation_slice, blind_slice = dataset.split_three_way(0.2, 0.1)
    legacy_before = legacy_hashes()
    runs: list[dict[str, Any]] = []

    for index, seed in enumerate(seeds):
        if reuse_jobs:
            job_id = reuse_jobs[index]
            status = TRAINING_MANAGER.status(job_id)
        else:
            started = TRAINING_MANAGER.start(
                {
                    "algorithm": "sac",
                    "dataset_id": dataset.dataset_id,
                    "environment_version": "port_ops_v6",
                    "port_profile_id": "cn_sha_coordinated_scenario_v6",
                    "business_profile_id": "coordinated_business_value_v6",
                    "reward_weights": business["reward_weights"],
                    "regulatory_delay_penalty_weight": business[
                        "regulatory_delay_penalty_weight"
                    ],
                    "projection_penalty_weight": business[
                        "projection_penalty_weight"
                    ],
                    "integrated_reward_weights": business[
                        "integrated_reward_weights"
                    ],
                    "coordinated_reward_weights": business[
                        "coordinated_reward_weights"
                    ],
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
            job_id = str(started["job_id"])
            status = wait(job_id, args.timeout)
        if status.get("status") not in {"COMPLETED", "EVALUATED"}:
            runs.append(
                {
                    "job_id": job_id,
                    "seed": seed,
                    "status": status.get("status"),
                    "error": status.get("error"),
                }
            )
            continue
        config = json.loads(
            (TRAINING_MANAGER.run_dir(job_id) / "config.json").read_text(
                encoding="utf-8"
            )
        )
        validation = TRAINING_MANAGER.evaluate_split_evidence(
            job_id, split_name="validation", episodes=args.episodes, persist=True
        )
        candidate_rows = [
            {key: float(value) for key, value in row.items()}
            for row in validation["episode_metrics"]
        ]
        starts = [
            int(value)
            for value in validation["evaluation_protocol"]["window_start_indices"]
        ]
        comparisons: dict[str, Any] = {}
        for name, policy in {
            "fcfs_neutral": FCFSNeutralPolicy(
                CoordinatedPortOperationsEnv.ACTION_DIMENSIONS
            ),
            "coordinated_current_operations_rule_proxy": rule_policy(config),
        }.items():
            env = make_env(dataset, validation_slice, train_slice, config)
            baseline_rows = evaluate_windows(policy, env, starts)
            env.close()
            comparisons[name] = compare(candidate_rows, baseline_rows)
        model_path = TRAINING_MANAGER.run_dir(job_id) / "model.zip"
        record = {
            "job_id": job_id,
            "seed": seed,
            "status": "COMPLETED",
            "model_path": relative(model_path),
            "model_sha256": sha256(model_path),
            "validation_metrics": validation["metrics"],
            "validation_uncertainty": validation["uncertainty"],
            "validation_comparisons": comparisons,
            "render_calls_during_training": status.get("rendering", {}).get(
                "render_calls", 0
            ),
        }
        runs.append(record)
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "seed": seed,
                    "validation_vs_fcfs": comparisons["fcfs_neutral"][
                        "coordinated_business_score_relative_improvement"
                    ],
                    "validation_vs_rule": comparisons[
                        "coordinated_current_operations_rule_proxy"
                    ]["coordinated_business_score_relative_improvement"],
                    "latent_action_correction_mean": validation["metrics"].get(
                        "latent_action_correction_mean"
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    completed = [row for row in runs if row.get("status") == "COMPLETED"]
    if not completed:
        raise RuntimeError("all V6 training runs failed; failure artifacts were preserved")

    def validation_constraint_failures(row: dict[str, Any]) -> int:
        metrics = row["validation_metrics"]
        terminal = min(
            float(metrics["quay_crane_completion_ratio"]),
            float(metrics["horizontal_transport_completion_ratio"]),
            float(metrics["yard_crane_completion_ratio"]),
        )
        checks = (
            float(metrics["gate_service_completion_ratio"]) >= 0.95,
            float(metrics["rail_service_completion_ratio"]) >= 0.80,
            float(metrics["barge_service_completion_ratio"]) >= 0.80,
            float(metrics["pilotage_service_completion_ratio"]) >= 0.80,
            float(metrics["towage_service_completion_ratio"]) >= 0.80,
            terminal >= 0.85,
            float(metrics["service_completion_ratio"]) >= 0.72,
            float(metrics["regulatory_clearance_completion_ratio"]) >= 0.65,
            float(metrics["shore_power_service_ratio"]) >= 0.45,
            float(metrics["guardrail_violation_rate"]) <= 1e-12,
            float(metrics["action_projection_severity_mean"]) <= 0.01,
            float(metrics["latent_action_correction_mean"]) <= 0.30,
        )
        return sum(not passed for passed in checks)

    completed.sort(
        key=lambda row: (
            validation_constraint_failures(row) > 0,
            validation_constraint_failures(row),
            -float(
                row["validation_comparisons"][
                    "coordinated_current_operations_rule_proxy"
                ]["coordinated_business_score_relative_improvement"]["ci_low"]
            ),
            -float(
                row["validation_comparisons"]["fcfs_neutral"][
                    "coordinated_business_score_relative_improvement"
                ]["ci_low"]
            ),
            float(row["validation_metrics"]["latent_action_correction_mean"]),
        )
    )
    selected = completed[0]
    selected_job_id = str(selected["job_id"])
    selected_config = json.loads(
        (TRAINING_MANAGER.run_dir(selected_job_id) / "config.json").read_text(
            encoding="utf-8"
        )
    )
    final_dataset = load_port_dataset(args.final_dataset_id)
    blind = evaluate_forward_policy(
        selected_job_id,
        dataset,
        final_dataset,
        train_slice,
        selected_config,
        args.episodes,
    )
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
            CoordinatedPortOperationsEnv.ACTION_DIMENSIONS
        ),
        "coordinated_current_operations_rule_proxy": rule_policy(selected_config),
    }
    baseline_rows: dict[str, list[dict[str, float]]] = {}
    baseline_parameters: dict[str, Any] = {}
    for name, policy in baseline_specs.items():
        env = make_env(
            final_dataset,
            slice(0, final_dataset.rows),
            train_slice,
            selected_config,
            normalization_dataset=dataset,
        )
        baseline_rows[name] = evaluate_windows(policy, env, starts)
        baseline_parameters[name] = policy.parameters()
        env.close()
    comparisons = {
        name: compare(candidate_rows, rows) for name, rows in baseline_rows.items()
    }
    candidate_means = metric_means(candidate_rows)
    fcfs = comparisons["fcfs_neutral"]
    rule = comparisons["coordinated_current_operations_rule_proxy"]
    fcfs_score = fcfs["coordinated_business_score_relative_improvement"]
    rule_score = rule["coordinated_business_score_relative_improvement"]
    fixed_metrics = rule["metric_relative_improvement"]
    terminal_completion = min(
        candidate_means["quay_crane_completion_ratio"],
        candidate_means["horizontal_transport_completion_ratio"],
        candidate_means["yard_crane_completion_ratio"],
    )
    gate_checks = {
        "coordinated_business_value_95ci_above_2_percent_vs_fcfs": bool(
            fcfs_score["ci_low"] > 0.02
        ),
        "coordinated_business_value_mean_nonnegative_vs_fixed_rule": bool(
            rule_score["mean"] >= 0.0
        ),
        "cost_regression_within_2_percent_vs_fixed_rule_95ci": bool(
            fixed_metrics["cost_per_teu"]["ci_low"] >= -0.02
        ),
        "gate_service_completion_at_least_95_percent": bool(
            candidate_means["gate_service_completion_ratio"] >= 0.95
        ),
        "rail_service_completion_at_least_80_percent": bool(
            candidate_means["rail_service_completion_ratio"] >= 0.80
        ),
        "barge_service_completion_at_least_80_percent": bool(
            candidate_means["barge_service_completion_ratio"] >= 0.80
        ),
        "pilotage_service_completion_at_least_80_percent": bool(
            candidate_means["pilotage_service_completion_ratio"] >= 0.80
        ),
        "towage_service_completion_at_least_80_percent": bool(
            candidate_means["towage_service_completion_ratio"] >= 0.80
        ),
        "terminal_move_chain_completion_at_least_85_percent": bool(
            terminal_completion >= 0.85
        ),
        "base_service_completion_at_least_72_percent": bool(
            candidate_means["service_completion_ratio"] >= 0.72
        ),
        "regulatory_clearance_completion_at_least_65_percent": bool(
            candidate_means["regulatory_clearance_completion_ratio"] >= 0.65
        ),
        "shore_power_service_at_least_45_percent": bool(
            candidate_means["shore_power_service_ratio"] >= 0.45
        ),
        "zero_guardrail_violations": bool(
            candidate_means["guardrail_violation_rate"] <= 1e-12
        ),
        "projection_severity_at_most_1_percent": bool(
            candidate_means["action_projection_severity_mean"] <= 0.01
        ),
        "latent_action_correction_mean_at_most_30_percent": bool(
            candidate_means["latent_action_correction_mean"] <= 0.30
        ),
        "terminal_soc_recovered": bool(
            candidate_means["terminal_soc_error"] <= 1e-6
        ),
        "three_seed_training_completed": len(completed) >= 3,
        "training_rendering_disabled": all(
            int(row.get("render_calls_during_training") or 0) == 0
            for row in completed
        ),
        "independent_forward_trace_generated_after_selection": bool(
            blind["render"]["frame_count"] > 0
        ),
    }
    admitted = all(gate_checks.values())
    previous_score = incumbent_score()
    promoted = bool(
        admitted
        and (previous_score is None or float(rule_score["mean"]) > previous_score)
    )

    legacy_after = {path: sha256(ROOT / path) for path in legacy_before}
    if legacy_after != legacy_before:
        raise RuntimeError("a pre-existing V3/V4/V5 model or evidence artifact changed")
    run_id = "coordinated-business-v6-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = EVIDENCE_ROOT / "runs" / run_id
    status = (
        "ADMITTED_OFFLINE_CHAMPION"
        if promoted
        else "ADMITTED_RETAINED_CANDIDATE"
        if admitted
        else "BLOCKED_RETAINED_CANDIDATE"
    )
    report = {
        "schema": "port-dt-coordinated-business-evidence.v1",
        "run_id": run_id,
        "status": status,
        "generated_at": now(),
        "evidence_label": "PUBLIC_AGGREGATE_REANALYSIS_PLUS_PREDECLARED_ENGINEERING_SPLIT_RESOURCE_CHAIN",
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
            "dataset_id": final_dataset.dataset_id,
            "artifact": relative(final_dataset.path),
            "sha256": final_dataset.fingerprint,
            "rows": final_dataset.rows,
            "independent_forward_challenge": True,
            "never_used_for_training_validation_or_selection": True,
            "normalization_fit": "2024_2025_chronological_training_rows_only",
        },
        "contract": {
            "environment_version": "port_ops_v6",
            "observation_dimensions": CoordinatedPortOperationsEnv.OBSERVATION_DIMENSIONS,
            "action_dimensions": CoordinatedPortOperationsEnv.ACTION_DIMENSIONS,
            "observation_design": {
                "measured_or_scenario_features": 53,
                "availability_masks": "all optional factors retain explicit masks",
                "stateful_business_pressures": [
                    "regulatory holds", "gate", "intermodal", "marine service",
                    "reefer", "maintenance", "shore power", "rail", "barge",
                    "pilotage", "towage", "quay moves", "horizontal moves",
                    "yard moves",
                ],
            },
            "action_design": list(CoordinatedPortOperationsEnv.ACTION_NAMES),
            "reward_design": {
                "business_objectives": business["reward_weights"],
                "integrated_penalties": business["integrated_reward_weights"],
                "coordinated_penalties": business["coordinated_reward_weights"],
            },
            "feasible_parameterization_outside_rl": [
                "bess_soc_and_terminal_reachability",
                "channel_closure_and_under_keel_clearance",
                "dangerous_goods_yard_inflow",
                "reefer_minimum_service",
                "maintenance_minimum_reserve",
                "gate_and_intermodal_minimum_service_commitments",
                "marine_minimum_service_when_navigation_window_is_open",
                "quay_horizontal_yard_chain_minimum_service_commitments",
            ],
            "hard_constraints_outside_rl": [
                "maritime_and_customs_release",
                "navigation_and_under_keel_clearance_acceptance",
                "dangerous_goods_rules",
                "equipment_interlocks",
                "schedule_commitment",
                "human_approval_and_execution",
            ],
            "authority": "recommendation_only_no_schedule_release_navigation_or_actuator_authority",
            "safety_revision": CoordinatedPortOperationsEnv.SAFETY_REVISION,
            "code_artifacts": {
                relative(ROOT / "app/services/rl_training/coordinated_environment.py"): sha256(
                    ROOT / "app/services/rl_training/coordinated_environment.py"
                ),
                relative(ROOT / "app/services/rl_training/integrated_environment.py"): sha256(
                    ROOT / "app/services/rl_training/integrated_environment.py"
                ),
                relative(ROOT / "config/ports/cn_sha_coordinated_scenario_v6.json"): sha256(
                    ROOT / "config/ports/cn_sha_coordinated_scenario_v6.json"
                ),
                relative(ROOT / "config/rl_business_profiles_v3.json"): sha256(
                    ROOT / "config/rl_business_profiles_v3.json"
                ),
                relative(ROOT / "scripts/train_coordinated_port_business_v6.py"): sha256(
                    ROOT / "scripts/train_coordinated_port_business_v6.py"
                ),
            },
        },
        "training": {
            "implementation": "stable_baselines3.SAC",
            "real_optimizer": True,
            "steps_per_seed": args.steps,
            "seeds_requested": seeds,
            "runs": runs,
            "selection_protocol": "all validation absolute service, safety, projection and correction gates first; then lower confidence bound versus fixed rule and FCFS; forward challenge sealed until selection",
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
            "promoted": promoted,
            "incumbent_business_score_vs_fixed_rule_mean": previous_score,
            "candidate_business_score_vs_fixed_rule_mean": rule_score["mean"],
            "offline_champion_saved": promoted,
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
            "Only official aggregate throughput and public reanalysis are public observations; split resource capacities remain declared engineering assumptions.",
            "The fixed-rule comparator is not measured Shanghai operator dispatch.",
            "The policy cannot commit a schedule, release a vessel, approve navigation or actuate equipment.",
            "Field value requires authorized source replacement, calibration, read-only shadow operation, independent interlocks and benefit attribution.",
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
    pointer = {
        "schema": "port-dt-coordinated-business-latest.v1",
        "run_id": run_id,
        "status": status,
        "report_path": relative(report_path),
        "report_sha256": report_hash,
        "selected_job_id": selected_job_id,
        "selected_model_path": selected["model_path"],
        "selected_model_sha256": selected["model_sha256"],
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.fingerprint,
        "final_evaluation_dataset_id": final_dataset.dataset_id,
        "final_evaluation_dataset_sha256": final_dataset.fingerprint,
        "production_authority": False,
        "updated_at": now(),
    }
    write_json(EVIDENCE_ROOT / "latest.json", pointer)
    history_path = EVIDENCE_ROOT / "history_index.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(pointer, ensure_ascii=False) + "\n")
    if promoted:
        write_json(
            EVIDENCE_ROOT / "offline_champion.json",
            {
                **pointer,
                "champion_metrics": candidate_means,
                "business_score_vs_fcfs": fcfs_score,
                "business_score_vs_fixed_rule": rule_score,
                "blind_window_data": relative(run_dir / "blind_window_metrics.json"),
                "promotion_boundary": "offline evidence champion only; production authority false",
            },
        )
    markdown = [
        "# V6 coordinated port-business evidence",
        "",
        f"- Status: `{status}`",
        f"- Dataset: `{dataset.dataset_id}` / `{dataset.fingerprint}`",
        f"- Selected: SAC seed `{selected['seed']}` / `{selected_job_id}`",
        f"- Training: {len(completed)} seeds x {args.steps:,} real optimizer steps; rendering disabled",
        f"- Forward challenge: {len(starts)} paired 48-hour windows",
        f"- Coordinated score versus FCFS: {100 * fcfs_score['mean']:.2f}%",
        f"- Coordinated score versus fixed rule: {100 * rule_score['mean']:.2f}%",
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
                "status": status,
                "selected_job_id": selected_job_id,
                "report_path": relative(report_path),
                "business_score_vs_fcfs": fcfs_score,
                "business_score_vs_fixed_rule": rule_score,
                "gate_checks": gate_checks,
                "legacy_artifacts_preserved": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
