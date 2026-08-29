"""Train, select and blind-test the additive V4 regulatory resilience policy."""

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
    LegacyV3PolicyAdapter,
)
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.mpc import MPCPolicy
from app.services.rl_training.regulatory_environment import (
    RegulatoryPortOperationsEnv,
)
from app.services.rl_training.statistics import bootstrap_summary
from app.services.rl_training.trainer import ALGORITHMS, TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config/rl_business_profiles_v3.json"
EVIDENCE_ROOT = ROOT / "evidence/v4/regulatory_delay"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


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


def legacy_artifact_hashes() -> dict[str, str]:
    paths = [
        *ROOT.glob("data/rl/runs/*/model.zip"),
        *ROOT.glob("data/rl/runs/*/manifest.json"),
        *ROOT.glob("evidence/v3/*.json"),
        *ROOT.glob("evidence/v3/*.sha256"),
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
) -> RegulatoryPortOperationsEnv:
    return RegulatoryPortOperationsEnv(
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
        port_profile=config["port_profile"],
        normalization_slice=train_slice,
        training=False,
        record_trace=False,
    )


def evaluate_windows(
    policy: Any,
    env: RegulatoryPortOperationsEnv,
    start_indices: list[int],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for episode, start_index in enumerate(start_indices):
        observation, _ = env.reset(
            seed=9000 + episode, options={"start_index": int(start_index)}
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


DIRECTIONS = {
    "regulatory_delay_teu_hours": "lower",
    "regulatory_delay_index_mean": "lower",
    "service_completion_ratio": "higher",
    "throughput_teu": "higher",
    "cost_per_teu": "lower",
    "carbon_kg_per_teu": "lower",
    "guardrail_violation_rate": "lower",
}


def compare(
    candidate: list[dict[str, float]], baseline: list[dict[str, float]]
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("paired comparison requires equal window counts")
    summaries: dict[str, Any] = {}
    for metric, direction in DIRECTIONS.items():
        values = []
        for current, reference in zip(candidate, baseline):
            denominator = max(abs(reference[metric]), 1e-12)
            improvement = (
                (current[metric] - reference[metric]) / denominator
                if direction == "higher"
                else (reference[metric] - current[metric]) / denominator
            )
            values.append(improvement)
        summaries[metric] = bootstrap_summary(values, seed=20260821)
    return {
        "paired_windows": len(candidate),
        "relative_improvement": summaries,
    }


def metric_means(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and blind-test V4 regulatory resilience"
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seeds", default="84,184,284")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    seeds = sorted({int(value) for value in args.seeds.split(",") if value.strip()})
    if len(seeds) < 3:
        parser.error("at least three distinct seeds are required")
    if args.steps < 10000:
        parser.error("at least 10,000 optimizer steps are required for business evidence")
    if args.episodes < 10:
        parser.error("at least 10 paired blind-test windows are required")

    profile_payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    business_profile = profile_payload["profiles"]["regulatory_resilience_v4"]
    legacy_before = legacy_artifact_hashes()
    dataset = load_port_dataset("public_cn_sha_regulatory_scenario_v4")
    train_slice, validation_slice, blind_slice = dataset.split_three_way(0.2, 0.1)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        started = TRAINING_MANAGER.start(
            {
                "algorithm": "sac",
                "dataset_id": dataset.dataset_id,
                "environment_version": "port_ops_v4",
                "port_profile_id": "cn_sha_regulatory_scenario_v4",
                "business_profile_id": "regulatory_resilience_v4",
                "reward_weights": business_profile["reward_weights"],
                "regulatory_delay_penalty_weight": business_profile[
                    "regulatory_delay_penalty_weight"
                ],
                "projection_penalty_weight": 0.1,
                "total_steps": args.steps,
                "episode_hours": 48,
                "episode_steps": 48,
                "test_ratio": 0.2,
                "validation_ratio": 0.1,
                "batch_size": 256,
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "replay_buffer": max(100000, args.steps * 2),
                "seed": seed,
            }
        )
        status = wait(started["job_id"], args.timeout)
        if status.get("status") != "COMPLETED":
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
        model_path = TRAINING_MANAGER.run_dir(started["job_id"]) / "model.zip"
        runs.append(
            {
                "job_id": started["job_id"],
                "seed": seed,
                "status": "COMPLETED",
                "model_path": relative(model_path),
                "model_sha256": sha256(model_path),
                "validation_metrics": validation["metrics"],
                "validation_uncertainty": validation["uncertainty"],
                "render_calls_during_training": status.get("rendering", {}).get(
                    "render_calls", 0
                ),
            }
        )
        print(
            json.dumps(
                {
                    "job_id": started["job_id"],
                    "seed": seed,
                    "validation_reward": validation["metrics"]["reward"],
                    "validation_regulatory_delay_teu_hours": validation["metrics"][
                        "regulatory_delay_teu_hours"
                    ],
                    "validation_guardrail_violation_rate": validation["metrics"][
                        "guardrail_violation_rate"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    completed = [row for row in runs if row.get("status") == "COMPLETED"]
    if not completed:
        raise RuntimeError("all V4 training runs failed; run artifacts were preserved")
    completed.sort(
        key=lambda row: (
            float(row["validation_metrics"]["guardrail_violation_rate"]),
            -float(row["validation_metrics"]["reward"]),
        )
    )
    selected = completed[0]
    selected_job_id = str(selected["job_id"])
    blind_evaluation = TRAINING_MANAGER.evaluate(
        selected_job_id, episodes=args.episodes
    )
    candidate_rows = [
        {key: float(value) for key, value in row.items()}
        for row in blind_evaluation["episode_metrics"]
    ]
    starts = [
        int(value)
        for value in blind_evaluation["evaluation_protocol"]["window_start_indices"]
    ]
    selected_config = json.loads(
        (TRAINING_MANAGER.run_dir(selected_job_id) / "config.json").read_text(
            encoding="utf-8"
        )
    )
    probe = make_env(dataset, blind_slice, train_slice, selected_config)
    controller_args = {
        "action_dim": 5,
        "episode_steps": probe.episode_steps,
        "soc_min": probe.soc_min,
        "soc_max": probe.soc_max,
        "initial_soc": 0.55,
        "bess_capacity_kwh": probe.bess_capacity_kwh,
        "bess_power_kw": probe.bess_power_kw,
        "step_hours": probe.step_hours,
    }
    probe.close()
    baseline_specs = {
        "legacy_v3_engineering_sop": LegacyV3PolicyAdapter(
            EngineeringCurrentOpsRulePolicy(**controller_args)
        ),
        "legacy_v3_mpc": LegacyV3PolicyAdapter(MPCPolicy(**controller_args)),
        "fcfs_neutral": FCFSNeutralPolicy(
            RegulatoryPortOperationsEnv.ACTION_DIMENSIONS
        ),
    }
    baseline_rows: dict[str, list[dict[str, float]]] = {}
    baseline_parameters: dict[str, Any] = {}
    for name, policy in baseline_specs.items():
        env = make_env(dataset, blind_slice, train_slice, selected_config)
        baseline_rows[name] = evaluate_windows(policy, env, starts)
        baseline_parameters[name] = (
            policy.parameters()
            if hasattr(policy, "parameters")
            else {"implementation": type(policy).__name__}
        )
        env.close()
    comparisons = {
        name: compare(candidate_rows, rows)
        for name, rows in baseline_rows.items()
    }
    incumbent = comparisons["legacy_v3_engineering_sop"]["relative_improvement"]
    candidate_means = metric_means(candidate_rows)
    gate_checks = {
        "regulatory_delay_reduction_95ci_above_2_percent": bool(
            incumbent["regulatory_delay_teu_hours"]["ci_low"] > 0.02
        ),
        "service_completion_non_degradation_95ci": bool(
            incumbent["service_completion_ratio"]["ci_low"] >= 0.0
        ),
        "cost_per_teu_regression_within_2_percent_95ci": bool(
            incumbent["cost_per_teu"]["ci_low"] >= -0.02
        ),
        "zero_guardrail_violations": bool(
            candidate_means["guardrail_violation_rate"] <= 1e-12
        ),
        "three_seed_training_completed": len(completed) >= 3,
        "training_rendering_disabled": all(
            int(row.get("render_calls_during_training") or 0) == 0
            for row in completed
        ),
        "blind_trace_generated_after_selection": bool(
            blind_evaluation["render"]["frame_count"] > 0
        ),
    }
    admitted = all(gate_checks.values())
    legacy_after = {
        path: sha256(ROOT / path) for path in legacy_before
    }
    legacy_preserved = legacy_after == legacy_before
    if not legacy_preserved:
        raise RuntimeError("a pre-existing V3 model/evidence artifact changed")
    run_id = (
        "regulatory-resilience-v4-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir = EVIDENCE_ROOT / "runs" / run_id
    report = {
        "schema": "port-dt-regulatory-resilience-evidence.v1",
        "run_id": run_id,
        "status": "ADMITTED_OFFLINE_SCENARIO_CANDIDATE" if admitted else "BLOCKED",
        "generated_at": now(),
        "evidence_label": "PREDECLARED_ENGINEERING_STRESS_SCENARIO_NOT_FIELD_KPI",
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "sha256": dataset.fingerprint,
            "rows": dataset.rows,
            "train_rows": train_slice.stop - train_slice.start,
            "validation_rows": validation_slice.stop - validation_slice.start,
            "blind_test_rows": blind_slice.stop - blind_slice.start,
            "chronological_no_shuffle": True,
        },
        "contract": {
            "environment_version": "port_ops_v4",
            "observation_dimensions": RegulatoryPortOperationsEnv.OBSERVATION_DIMENSIONS,
            "action_dimensions": RegulatoryPortOperationsEnv.ACTION_DIMENSIONS,
            "new_actions": ["inspection_buffer", "recovery_priority"],
            "authority": "recommendation_only_no_release_authority",
            "safety_revision": RegulatoryPortOperationsEnv.SAFETY_REVISION,
        },
        "training": {
            "algorithm": "sac",
            "steps_per_seed": args.steps,
            "seeds_requested": seeds,
            "runs": runs,
            "selection_protocol": "minimum validation guardrail violation rate, then maximum validation reward; blind test sealed until selection",
            "selected_job_id": selected_job_id,
            "selected_seed": selected["seed"],
        },
        "blind_test": {
            "split": blind_evaluation["split"],
            "paired_window_start_indices": starts,
            "window_count": len(starts),
            "episode_steps": selected_config["episode_steps"],
            "selected_metrics": candidate_means,
            "selected_uncertainty": blind_evaluation["uncertainty"],
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
            "model_promoted": False,
            "production_authority": False,
        },
        "legacy_preservation": {
            "checked_artifact_count": len(legacy_before),
            "preserved": legacy_preserved,
            "sha256_before": legacy_before,
            "sha256_after": legacy_after,
        },
        "limitations": [
            "Regulatory factors are a predeclared engineering stress scenario, not Shanghai inspection telemetry.",
            "The legacy V3 engineering SOP is a transparent proxy, not measured incumbent operator dispatch.",
            "No policy can alter inspection findings, detention, secondary checks or official release decisions.",
            "Site claims require authorized maritime, customs, vessel-call and TOS event replacement plus shadow acceptance.",
        ],
    }
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    report_hash = sha256(report_path)
    latest = {
        "schema": "port-dt-regulatory-resilience-latest.v1",
        "run_id": run_id,
        "status": report["status"],
        "report_path": relative(report_path),
        "report_sha256": report_hash,
        "selected_job_id": selected_job_id,
        "selected_model_path": selected["model_path"],
        "selected_model_sha256": selected["model_sha256"],
        "production_authority": False,
        "updated_at": now(),
    }
    write_json(EVIDENCE_ROOT / "latest.json", latest)
    history_path = EVIDENCE_ROOT / "history_index.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(latest, ensure_ascii=False) + "\n")
    incumbent_summary = incumbent["regulatory_delay_teu_hours"]
    markdown = [
        "# V4 regulatory resilience evidence",
        "",
        f"- Status: `{report['status']}`",
        f"- Dataset: `{dataset.dataset_id}` / `{dataset.fingerprint}`",
        f"- Selected: SAC seed `{selected['seed']}` / `{selected_job_id}`",
        f"- Training: {len(completed)} seeds x {args.steps:,} steps; rendering disabled",
        f"- Blind test: {len(starts)} paired 48-hour windows",
        "- Boundary: predeclared engineering regulatory stress scenario, not field KPI",
        "",
        "## Versus regulator-unaware V3 engineering SOP adapter",
        "",
        f"- Regulatory delay TEU-hours improvement: {100*incumbent_summary['mean']:.2f}% (95% CI {100*incumbent_summary['ci_low']:.2f}% to {100*incumbent_summary['ci_high']:.2f}%)",
        f"- Service completion improvement: {100*incumbent['service_completion_ratio']['mean']:.2f}%",
        f"- Cost/TEU improvement: {100*incumbent['cost_per_teu']['mean']:.2f}%",
        f"- Guardrail violation rate: {candidate_means['guardrail_violation_rate']:.6f}",
        "",
        "The policy reserves inspection readiness and prioritizes post-release recovery only. It has no authority to change a maritime/customs decision or execute production dispatch.",
        "",
    ]
    markdown_path = run_dir / "report.md"
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
                "regulatory_delay_improvement": incumbent_summary,
                "gate_checks": gate_checks,
                "legacy_artifacts_preserved": legacy_preserved,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
