"""Evaluate the locked V4 candidate on the independent 2026 forward challenge."""

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
    LegacyV3PolicyAdapter,
)
from app.services.rl_training.datasets import PortDataset, load_port_dataset
from app.services.rl_training.mpc import MPCPolicy
from app.services.rl_training.regulatory_environment import (
    RegulatoryPortOperationsEnv,
)
from app.services.rl_training.statistics import bootstrap_summary
from app.services.rl_training.trainer import TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "evidence/v4/regulatory_delay"
TRAIN_SCENARIO_CONFIG = ROOT / "config/regulatory_delay_scenario_v4.json"
FORWARD_SCENARIO_CONFIG = ROOT / "config/regulatory_delay_forward_challenge_v4.json"
DIRECTIONS = {
    "regulatory_delay_teu_hours": "lower",
    "regulatory_delay_index_mean": "lower",
    "service_completion_ratio": "higher",
    "throughput_teu": "higher",
    "cost_per_teu": "lower",
    "carbon_kg_per_teu": "lower",
    "guardrail_violation_rate": "lower",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def evaluate_windows(
    policy: Any,
    env: RegulatoryPortOperationsEnv,
    starts: list[int],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for episode, start in enumerate(starts):
        observation, _ = env.reset(
            seed=12000 + episode, options={"start_index": start}
        )
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(action)
        row = env.totals
        row["delay_index_mean"] = row.pop("delay") / env.episode_steps
        row["guardrail_violation_rate"] = row.pop("violations") / env.episode_steps
        rows.append({key: float(value) for key, value in row.items()})
    return rows


def means(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in sorted(rows[0])
    }


def compare(
    candidate: list[dict[str, float]], baseline: list[dict[str, float]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric, direction in DIRECTIONS.items():
        paired = []
        for current, reference in zip(candidate, baseline):
            denominator = max(abs(reference[metric]), 1e-12)
            paired.append(
                (current[metric] - reference[metric]) / denominator
                if direction == "higher"
                else (reference[metric] - current[metric]) / denominator
            )
        output[metric] = bootstrap_summary(paired, seed=20260821)
    return output


def composite_dataset(training: PortDataset, forward: PortDataset) -> PortDataset:
    return PortDataset(
        dataset_id="public_cn_sha_regulatory_v4_train_plus_forward_composite",
        path=forward.path,
        timestamps=[*training.timestamps, *forward.timestamps],
        values=np.concatenate([training.values, forward.values], axis=0),
        metadata={
            "sha256": hashlib.sha256(
                f"{training.fingerprint}:{forward.fingerprint}".encode("utf-8")
            ).hexdigest(),
            "provenance_type": "in_memory_train_normalization_plus_forward_evaluation",
        },
        factor_values=np.concatenate(
            [training.factor_values, forward.factor_values], axis=0
        ),
        factor_availability=np.concatenate(
            [training.factor_availability, forward.factor_availability], axis=0
        ),
        regulatory_values=np.concatenate(
            [training.regulatory_values, forward.regulatory_values], axis=0
        ),
        regulatory_availability=np.concatenate(
            [
                training.regulatory_availability,
                forward.regulatory_availability,
            ],
            axis=0,
        ),
    )


def main() -> None:
    pointer_path = EVIDENCE_ROOT / "latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    report_path = ROOT / pointer["report_path"]
    if sha256(report_path) != pointer["report_sha256"]:
        raise RuntimeError("selected V4 report hash gate failed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "ADMITTED_OFFLINE_SCENARIO_CANDIDATE":
        raise RuntimeError("the latest V4 candidate is not admitted for forward challenge")
    job_id = str(report["training"]["selected_job_id"])
    run_dir = TRAINING_MANAGER.run_dir(job_id)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    training = load_port_dataset("public_cn_sha_regulatory_scenario_v4")
    forward = load_port_dataset("public_cn_sha_regulatory_forward_2026m05_v4")
    training_scenario = json.loads(TRAIN_SCENARIO_CONFIG.read_text(encoding="utf-8"))
    forward_scenario = json.loads(FORWARD_SCENARIO_CONFIG.read_text(encoding="utf-8"))
    if training_scenario["parameters"] != forward_scenario["parameters"]:
        raise RuntimeError("forward regulatory parameters differ from training scenario")
    if forward_scenario["evidence_boundary"].get("candidate_selection_allowed") is not False:
        raise RuntimeError("forward challenge must prohibit candidate selection")
    train_slice, _validation_slice, _blind_slice = training.split_three_way(0.2, 0.1)
    combined = composite_dataset(training, forward)
    forward_slice = slice(training.rows, training.rows + forward.rows)

    def make_env() -> RegulatoryPortOperationsEnv:
        return RegulatoryPortOperationsEnv(
            combined,
            forward_slice,
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

    probe = make_env()
    max_start = len(probe.segment) - probe.episode_steps - 1
    starts = np.linspace(0, max_start, num=20, dtype=int).tolist()
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
    candidate_policy = TRAINING_MANAGER._load_policy(config, run_dir, probe)
    candidate_rows = evaluate_windows(candidate_policy, probe, starts)
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
    for name, policy in baseline_specs.items():
        env = make_env()
        baseline_rows[name] = evaluate_windows(policy, env, starts)
        env.close()
    comparisons = {
        name: compare(candidate_rows, rows)
        for name, rows in baseline_rows.items()
    }
    incumbent = comparisons["legacy_v3_engineering_sop"]
    candidate_means = means(candidate_rows)
    checks = {
        "candidate_locked_before_forward_read": True,
        "forward_candidate_selection_prohibited": True,
        "scenario_parameters_frozen": True,
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
    }
    payload = {
        "schema": "port-dt-regulatory-resilience-forward-challenge.v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "generated_at": now(),
        "evidence_label": "OUT_OF_PERIOD_FORWARD_ENGINEERING_STRESS_CHALLENGE_NOT_FIELD_KPI",
        "selected_before_forward": {
            "job_id": job_id,
            "seed": report["training"]["selected_seed"],
            "model_path": pointer["selected_model_path"],
            "model_sha256": pointer["selected_model_sha256"],
            "selection_report_path": pointer["report_path"],
            "selection_report_sha256": pointer["report_sha256"],
        },
        "forward_dataset": {
            "dataset_id": forward.dataset_id,
            "sha256": forward.fingerprint,
            "rows": forward.rows,
            "start_at": forward.timestamps[0],
            "end_at": forward.timestamps[-1],
            "independent_source_observations": forward.metadata.get(
                "independent_source_observations"
            ),
            "candidate_selection_allowed": False,
            "scenario_config_path": str(FORWARD_SCENARIO_CONFIG.relative_to(ROOT)),
            "scenario_config_sha256": sha256(FORWARD_SCENARIO_CONFIG),
        },
        "protocol": {
            "normalization_fit": "2024-2025 chronological training rows only",
            "evaluation": "2026-01-01 through 2026-05-31 forward rows only",
            "paired_window_start_indices": starts,
            "paired_windows": len(starts),
            "episode_steps": config["episode_steps"],
            "render_during_evaluation": False,
            "candidate_selection_or_tuning": False,
        },
        "candidate_metrics": candidate_means,
        "comparators": {
            name: means(rows) for name, rows in baseline_rows.items()
        },
        "paired_comparisons": comparisons,
        "admission": {
            "checks": checks,
            "passed": all(checks.values()),
            "model_promoted": False,
            "production_authority": False,
        },
        "limitations": [
            "The 2026 base package uses official aggregate anchors and public environmental model/reanalysis data, not terminal telemetry.",
            "Regulatory fields are generated by the frozen engineering scenario, not maritime/customs observations.",
            "The result is an out-of-period software/value challenge, not a field KPI, causal effect or production approval.",
        ],
    }
    output_path = report_path.parent / "forward_challenge.json"
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        previous_stamp = str(previous.get("generated_at") or "unknown").replace(
            ":", "-"
        )
        archive_path = (
            report_path.parent
            / "forward_history"
            / previous_stamp
            / "forward_challenge.json"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(archive_path, previous)
        history_path = report_path.parent / "forward_history_index.jsonl"
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "status": previous.get("status"),
                        "generated_at": previous.get("generated_at"),
                        "path": str(archive_path.relative_to(ROOT)),
                        "sha256": sha256(archive_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    write_json(output_path, payload)
    output_hash = sha256(output_path)
    pointer.update(
        forward_challenge_status=payload["status"],
        forward_challenge_path=str(output_path.relative_to(ROOT)),
        forward_challenge_sha256=output_hash,
        production_authority=False,
        updated_at=now(),
    )
    write_json(pointer_path, pointer)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "path": str(output_path.relative_to(ROOT)),
                "sha256": output_hash,
                "candidate_metrics": candidate_means,
                "vs_legacy_v3_engineering_sop": incumbent,
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
