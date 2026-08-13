"""Compare the selected V3 policy with transparent stronger control baselines."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.services.rl_training.baselines import (
    EngineeringCurrentOpsRulePolicy,
    FCFSNeutralPolicy,
)
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.environment import PortOperationsEnv
from app.services.rl_training.mpc import MPCPolicy
from app.services.rl_training.statistics import bootstrap_summary
from app.services.rl_training.trainer import ALGORITHMS, TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[1]
ADVANTAGE_PATH = ROOT / "evidence/v3/shanghai_public_advantage_v3.json"
OUTPUT_DIR = ROOT / "evidence/v3"
METRICS = ("throughput_teu", "delay_index_mean", "energy_cost", "carbon_kg", "peak_kw")
WEIGHTS = {
    "throughput_teu": 0.25,
    "delay_index_mean": 0.25,
    "energy_cost": 0.20,
    "carbon_kg": 0.15,
    "peak_kw": 0.15,
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_windows(policy: Any, env: PortOperationsEnv, start_indices: list[int]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for episode, start_index in enumerate(start_indices):
        obs, _ = env.reset(seed=9000 + episode, options={"start_index": start_index})
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
        row = env.totals
        row["delay_index_mean"] = row.pop("delay") / max(1, env.episode_steps)
        row["guardrail_violation_rate"] = row.pop("violations") / max(1, env.episode_steps)
        rows.append({key: float(value) for key, value in row.items()})
    return rows


def comparison(candidate: list[dict[str, float]], baseline: list[dict[str, float]]) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("paired comparison requires equal window counts")
    relative_rows: list[dict[str, float]] = []
    for current, reference in zip(candidate, baseline):
        row: dict[str, float] = {}
        for metric in METRICS:
            denominator = max(abs(reference[metric]), 1e-12)
            row[metric] = (
                (current[metric] - reference[metric]) / denominator
                if metric == "throughput_teu"
                else (reference[metric] - current[metric]) / denominator
            )
        relative_rows.append(row)
    metric_summaries = {
        metric: bootstrap_summary([row[metric] for row in relative_rows], seed=20260813)
        for metric in METRICS
    }
    composite_rows = [
        sum(WEIGHTS[name] * row[name] for name in METRICS)
        for row in relative_rows
    ]
    composite = bootstrap_summary(composite_rows, seed=20260813)
    return {
        "paired_windows": len(relative_rows),
        "metrics_relative_improvement": metric_summaries,
        "weighted_relative_improvement": composite,
        "strict_advantage_95ci": bool((composite.get("ci_low") or 0.0) > 0.0),
    }


def main() -> None:
    advantage = read_json(ADVANTAGE_PATH)
    selected = advantage["selected"]
    job_ids = [str(value) for value in selected["job_ids"]]
    first_config = read_json(TRAINING_MANAGER.run_dir(job_ids[0]) / "config.json")
    dataset = load_port_dataset(first_config["dataset_id"], TRAINING_MANAGER.data_root)
    train_slice, _validation_slice, test_slice = dataset.split_three_way(
        first_config["test_ratio"], first_config["validation_ratio"]
    )

    def make_env() -> PortOperationsEnv:
        return PortOperationsEnv(
            dataset,
            test_slice,
            action_mode="continuous",
            episode_steps=first_config["episode_steps"],
            seed=first_config["seed"],
            demand_cap_kw=first_config["demand_cap_kw"],
            reward_weights=first_config["reward_weights"],
            projection_penalty_weight=float(first_config.get("projection_penalty_weight") or 0.0),
            environment_version=first_config["environment_version"],
            port_profile=first_config["port_profile"],
            normalization_slice=train_slice,
            training=False,
            record_trace=False,
        )

    probe = make_env()
    max_start = max(0, len(probe.segment) - probe.episode_steps - 1)
    starts = np.linspace(0, max_start, num=min(10, max_start + 1), dtype=int).tolist()
    action_dim = int(np.prod(probe.action_space.shape))
    controller_args = dict(
        action_dim=action_dim,
        episode_steps=probe.episode_steps,
        soc_min=probe.soc_min,
        soc_max=probe.soc_max,
        initial_soc=0.55,
        bess_capacity_kwh=probe.bess_capacity_kwh,
        bess_power_kw=probe.bess_power_kw,
        step_hours=probe.step_hours,
    )
    probe.close()

    baseline_specs = {
        "fcfs_neutral": (FCFSNeutralPolicy(action_dim), {"measured_operator_policy": False, "kind": "neutral_comparator"}),
        "engineering_ops_rule": (EngineeringCurrentOpsRulePolicy(**controller_args), EngineeringCurrentOpsRulePolicy(**controller_args).parameters()),
        "mpc": (MPCPolicy(**controller_args), {**MPCPolicy(**controller_args).parameters(), "measured_operator_policy": False, "kind": "receding_horizon_control"}),
    }
    baseline_rows: dict[str, list[dict[str, float]]] = {}
    for name, (policy, _parameters) in baseline_specs.items():
        env = make_env()
        baseline_rows[name] = evaluate_windows(policy, env, starts)
        env.close()

    seed_rows: list[list[dict[str, float]]] = []
    model_hashes: list[str] = []
    for job_id in job_ids:
        config = read_json(TRAINING_MANAGER.run_dir(job_id) / "config.json")
        env = make_env()
        policy = TRAINING_MANAGER._load_policy(config, TRAINING_MANAGER.run_dir(job_id), env)
        seed_rows.append(evaluate_windows(policy, env, starts))
        model_hashes.append(sha256(TRAINING_MANAGER.run_dir(job_id) / "model.zip"))
        env.close()
    ensemble_rows = [
        {
            metric: float(np.mean([seed[index][metric] for seed in seed_rows]))
            for metric in seed_rows[0][index]
        }
        for index in range(len(starts))
    ]
    comparisons = {
        name: comparison(ensemble_rows, rows)
        for name, rows in baseline_rows.items()
    }
    payload = {
        "schema": "port-dt-v3-strong-baseline-evidence.v1",
        "version": advantage["version"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.fingerprint,
        "environment_version": first_config["environment_version"],
        "selected_policy": {
            "algorithm": selected["algorithm"],
            "job_ids": job_ids,
            "model_sha256": model_hashes,
            "seeds": selected["seeds"],
            "ensemble": "per-window arithmetic mean of three deterministic seed policies",
        },
        "protocol": {
            "split": "chronological_blind_test_only",
            "window_start_indices": starts,
            "window_count": len(starts),
            "episode_steps": first_config["episode_steps"],
            "render_during_policy_execution": False,
            "paired_comparison": True,
            "metric_weights": WEIGHTS,
            "baseline_thresholds_tuned_on_blind_test": False,
        },
        "baselines": {
            name: {
                "parameters": parameters,
                "window_metrics": baseline_rows[name],
            }
            for name, (_policy, parameters) in baseline_specs.items()
        },
        "selected_window_metrics": ensemble_rows,
        "comparisons": comparisons,
        "strong_baseline_gate": {
            "all_comparators_strictly_beaten": all(
                row["strict_advantage_95ci"] for row in comparisons.values()
            ),
            "fcfs_only_is_not_sufficient_for_production_claim": True,
            "measured_current_operations_baseline_available": False,
            "production_claim_admitted": False,
        },
        "site_replacement": {
            "required": True,
            "replace_engineering_ops_rule_with": "timestamped operator dispatch, SOP mode and realized action logs",
            "acceptance": "paired shadow replay against the approved incumbent policy",
        },
        "claim_boundary": "The engineering operations rule is a transparent stronger proxy, not measured Shanghai incumbent control. No production or group-savings claim is admitted until site policy/action logs replace it.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "strong_baseline_evidence_v3.json"
    md_path = OUTPUT_DIR / "strong_baseline_evidence_v3.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V3 strong-baseline evidence", "",
        f"- Selected policy: `{selected['algorithm']}` / seeds {selected['seeds']}",
        "- Split: chronological blind test only; 10 paired windows",
        "- Engineering rule baseline is a proxy, not measured incumbent operations.", "",
        "| Comparator | Weighted improvement | 95% CI | Strict advantage |",
        "|---|---:|---:|---|",
    ]
    for name, row in comparisons.items():
        summary = row["weighted_relative_improvement"]
        lines.append(
            f"| {name} | {100*summary['mean']:.2f}% | [{100*summary['ci_low']:.2f}%, {100*summary['ci_high']:.2f}%] | {row['strict_advantage_95ci']} |"
        )
    lines.extend(["", payload["claim_boundary"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    sidecar = OUTPUT_DIR / "strong_baseline_evidence_v3.sha256"
    sidecar.write_text(
        f"{sha256(json_path)}  {json_path.name}\n{sha256(md_path)}  {md_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"comparisons": comparisons, "gate": payload["strong_baseline_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
