"""Export the version-pinned V3 validation-selection and blind-test report."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.statistics import bootstrap_summary
from app.services.rl_training.trainer import ALGORITHMS, TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/v3_advantage_benchmark.json"
OUTPUT_DIR = ROOT / "evidence/v3"

BUSINESS_METRICS = (
    "throughput_teu",
    "delay_index_mean",
    "energy_cost",
    "carbon_kg",
    "peak_kw",
    "cost_per_teu",
    "carbon_kg_per_teu",
    "energy_kwh_per_teu",
    "grid_energy_kwh",
    "service_completion_ratio",
    "queue_peak_teu",
    "queue_end_teu",
    "operational_resource_factor_mean",
    "service_factor_mean",
    "guardrail_violation_rate",
    "action_projection_rate",
    "action_projection_correction_kw_mean",
    "action_projection_severity_mean",
    "action_projection_grid_cap_rate",
    "action_projection_soc_bound_rate",
    "action_projection_terminal_reachability_rate",
    "action_projection_power_bound_rate",
    "terminal_soc_error",
    "weather_block_rate",
    "bess_throughput_kwh",
    "bess_equivalent_full_cycles",
    "flex_shift_energy_kwh",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_runs(
    registry: dict[str, Any],
    config: dict[str, Any],
    dataset_sha256: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in registry.get("runs") or []:
        algorithm = str(run.get("algorithm") or "")
        if algorithm not in ALGORITHMS:
            continue
        if run.get("dataset_id") != config["dataset_id"]:
            continue
        if run.get("dataset_sha256") != dataset_sha256:
            continue
        if int(run.get("episodes") or 0) < int(config["minimum_holdout_episodes"]):
            continue
        spec = ALGORITHMS[algorithm]
        if spec.trainable:
            if int(run.get("total_steps") or 0) < int(config["minimum_optimizer_steps"]):
                continue
            if run.get("evidence_label") != "RL_HELD_OUT_EVALUATION":
                continue
        elif run.get("evidence_label") != "DETERMINISTIC_CONTROLLER_BASELINE":
            continue
        run_dir = TRAINING_MANAGER.run_dir(str(run["job_id"]))
        run_config = read_json(run_dir / "config.json")
        expected_profile = str(config.get("business_profile_id") or "default_port_profile")
        observed_profile = str(run_config.get("business_profile_id") or "default_port_profile")
        if observed_profile != expected_profile:
            continue
        expected_projection_penalty = float(
            config.get("projection_penalty_weight") or 0.0
        )
        observed_projection_penalty = float(
            run_config.get("projection_penalty_weight") or 0.0
        )
        if spec.trainable and abs(observed_projection_penalty - expected_projection_penalty) > 1e-12:
            continue
        if run_config.get("environment_version") != config.get("environment_version"):
            continue
        evaluation = read_json(run_dir / "evaluation.json")
        manifest = read_json(run_dir / "manifest.json")
        observed_steps = int(
            manifest.get("total_steps_observed")
            if manifest.get("total_steps_observed") is not None
            else run.get("total_steps") or 0
        )
        if spec.trainable and observed_steps < int(config["minimum_optimizer_steps"]):
            continue
        if manifest.get("dataset_sha256") != dataset_sha256:
            continue
        if evaluation.get("split") != "chronological_blind_test_only":
            continue
        output.append(
            {
                **run,
                "total_steps": observed_steps,
                "environment_version": run_config.get("environment_version"),
                "episode_steps": run_config.get("episode_steps"),
                "test_ratio": run_config.get("test_ratio"),
                "validation_ratio": run_config.get("validation_ratio"),
                "projection_penalty_weight": observed_projection_penalty,
                "model_sha256": manifest.get("model_sha256"),
            }
        )
    return output


def relative(candidate: float, baseline: float, *, maximize: bool) -> float:
    if abs(baseline) <= 1e-12:
        raise ValueError("baseline metric is zero; relative improvement is undefined")
    ratio = candidate / baseline
    return ratio - 1.0 if maximize else 1.0 - ratio


def summarize_relative_metrics(
    metric_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    weights: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative_rows = [
        {
            metric: relative(
                float(row[metric]),
                float(baseline_metrics[metric]),
                maximize=metric == "throughput_teu",
            )
            for metric in weights
        }
        for row in metric_rows
    ]
    per_metric = {
        metric: bootstrap_summary(
            [row[metric] for row in relative_rows],
            seed=20260812,
        )
        for metric in weights
    }
    composites = [
        sum(weights[name] * row[name] for name in weights)
        for row in relative_rows
    ]
    return per_metric, bootstrap_summary(composites, seed=20260812)


def safety_admission(
    metric_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    limits = config["eligibility"]
    observed = {
        "guardrail_violation_rate_max_observed": max(
            float(row["guardrail_violation_rate"]) for row in metric_rows
        ),
        "action_projection_rate_max_observed": max(
            float(row["action_projection_rate"]) for row in metric_rows
        ),
        "terminal_soc_error_max_observed": max(
            float(row["terminal_soc_error"]) for row in metric_rows
        ),
    }
    observed["passed"] = bool(
        observed["guardrail_violation_rate_max_observed"]
        <= float(limits["guardrail_violation_rate_max"])
        and observed["action_projection_rate_max_observed"]
        <= float(limits["action_projection_rate_max"])
        and observed["terminal_soc_error_max_observed"]
        <= float(limits["terminal_soc_error_max"])
    )
    return observed


def select_latest_seed_runs(
    runs: list[dict[str, Any]],
    algorithm: str,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for seed in expected_seeds:
        matches = [
            run
            for run in runs
            if run.get("algorithm") == algorithm and run.get("seed") == seed
        ]
        if not matches:
            return []
        selected.append(max(matches, key=lambda run: str(run.get("evaluated_at") or "")))
    return selected


def main() -> None:
    config = read_json(CONFIG_PATH)
    dataset = load_port_dataset(config["dataset_id"], TRAINING_MANAGER.data_root)
    registry = read_json(TRAINING_MANAGER.benchmark_path)
    runs = eligible_runs(registry, config, dataset.fingerprint)
    baseline_algorithm = str(config["baseline_algorithm"])
    baseline_runs = [run for run in runs if run["algorithm"] == baseline_algorithm]
    if not baseline_runs:
        raise RuntimeError("no eligible FCFS baseline on the current v3 dataset")
    baseline_run = max(baseline_runs, key=lambda run: str(run.get("evaluated_at") or ""))
    baseline_metrics = baseline_run["metrics"]
    validation_episodes = int(config["minimum_holdout_episodes"])
    baseline_validation = TRAINING_MANAGER.evaluate_split_evidence(
        str(baseline_run["job_id"]),
        split_name="validation",
        episodes=validation_episodes,
    )
    baseline_validation_metrics = baseline_validation["metrics"]
    weights = {name: float(value) for name, value in config["metric_weights"].items()}
    expected_seeds = [int(seed) for seed in config.get("seeds") or [42, 142, 242]]
    candidates: list[dict[str, Any]] = []
    candidate_runs: dict[str, list[dict[str, Any]]] = {}
    for algorithm, spec in ALGORITHMS.items():
        if spec.family != config["candidate_family"]:
            continue
        compatible = [
            run for run in runs
            if run.get("environment_version") == baseline_run.get("environment_version")
            and run.get("episode_steps") == baseline_run.get("episode_steps")
            and run.get("test_ratio") == baseline_run.get("test_ratio")
            and run.get("validation_ratio") == baseline_run.get("validation_ratio")
        ]
        selected = select_latest_seed_runs(compatible, algorithm, expected_seeds)
        seeds = sorted({int(run["seed"]) for run in selected if isinstance(run.get("seed"), int)})
        if len(seeds) < int(config["minimum_distinct_seeds"]):
            continue
        validation_evaluations = [
            TRAINING_MANAGER.evaluate_split_evidence(
                str(run["job_id"]),
                split_name="validation",
                episodes=validation_episodes,
            )
            for run in selected
        ]
        validation_rows = [row["metrics"] for row in validation_evaluations]
        admission = safety_admission(validation_rows, config)
        if not admission["passed"]:
            continue
        validation_relative, validation_composite = summarize_relative_metrics(
            validation_rows,
            baseline_validation_metrics,
            weights,
        )
        candidate_runs[algorithm] = selected
        candidates.append(
            {
                "algorithm": algorithm,
                "name": spec.name,
                "implementation": spec.implementation,
                "environment_version": selected[0].get("environment_version"),
                "job_ids": [run["job_id"] for run in selected],
                "model_sha256": [run.get("model_sha256") for run in selected],
                "seeds": seeds,
                "selection_split": "chronological_validation_only",
                "validation_metrics_relative_to_fcfs": validation_relative,
                "validation_weighted_relative_improvement": validation_composite,
                "validation_safety_admission": admission,
            }
        )
    if not candidates:
        raise RuntimeError("no RL candidate satisfies the version-pinned V3 validation gate")
    candidates.sort(
        key=lambda item: float(item["validation_weighted_relative_improvement"]["mean"]),
        reverse=True,
    )
    validation_winner = candidates[0]
    selected_runs = candidate_runs[validation_winner["algorithm"]]
    blind_rows = [run["metrics"] for run in selected_runs]
    metric_improvements, composite = summarize_relative_metrics(
        blind_rows,
        baseline_metrics,
        weights,
    )
    blind_test_metrics = {
        metric: bootstrap_summary(
            [float(row[metric]) for row in blind_rows],
            seed=20260812,
        )
        for metric in BUSINESS_METRICS
        if all(isinstance(row.get(metric), (int, float)) for row in blind_rows)
    }
    test_admission = safety_admission(blind_rows, config)
    strict_advantage = bool(
        test_admission["passed"]
        and composite.get("ci_low") is not None
        and composite["ci_low"] > 0
    )
    winner = {
        **validation_winner,
        "final_report_split": "chronological_blind_test_only",
        "metrics_relative_to_fcfs": metric_improvements,
        "blind_test_metrics": blind_test_metrics,
        "weighted_relative_improvement": composite,
        "strict_advantage": strict_advantage,
        "safety_admission": test_admission,
    }
    historical_projection: dict[str, Any] | None = None
    history_reports = sorted((OUTPUT_DIR / "history").glob("advantage-*/shanghai_public_advantage_v3.json"))
    for historical_path in reversed(history_reports):
        historical = read_json(historical_path)
        historical_row = (
            ((historical.get("selected") or {}).get("blind_test_metrics") or {})
            .get("action_projection_rate")
        )
        if historical.get("version") == config["version"] or not isinstance(historical_row, dict):
            continue
        old_mean = float(historical_row.get("mean") or 0.0)
        new_mean = float(blind_test_metrics["action_projection_rate"]["mean"])
        historical_projection = {
            "historical_version": historical.get("version"),
            "historical_report": str(historical_path.relative_to(ROOT)),
            "historical_mean": old_mean,
            "current_version": config["version"],
            "current_mean": new_mean,
            "absolute_reduction": old_mean - new_mean,
            "relative_reduction": (old_mean - new_mean) / old_mean if old_mean > 0 else None,
            "historical_preserved": True,
        }
        break
    claim_status = (
        "TEST_SAFETY_ADMISSION_FAILED"
        if not test_admission["passed"]
        else "STRICT_ADVANTAGE_95CI"
        if strict_advantage
        else "POINT_ESTIMATE_ADVANTAGE_NOT_95CI_CONFIRMED"
    )
    payload = {
        "schema": "port-dt-v3-advantage-evidence.v1",
        "version": config["version"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "benchmark_contract": {
            **config,
            "artifact_id": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": file_sha256(CONFIG_PATH),
        },
        "dataset": dataset.describe(test_ratio=0.2, validation_ratio=0.1),
        "baseline": {
            "algorithm": baseline_algorithm,
            "job_id": baseline_run["job_id"],
            "metrics": baseline_metrics,
            "episodes": baseline_run["episodes"],
            "dataset_sha256": baseline_run["dataset_sha256"],
            "environment_version": baseline_run.get("environment_version"),
            "validation_metrics": baseline_validation_metrics,
        },
        "selection_protocol": {
            "algorithm_selection": "chronological_validation_only",
            "final_advantage_report": "chronological_blind_test_only",
            "blind_test_used_for_selection": False,
            "expected_seeds": expected_seeds,
            "validation_episode_windows": validation_episodes,
        },
        "candidates": candidates,
        "selected": winner,
        "claim_status": claim_status,
        "claim_boundary": config["claim_boundary"],
        "production_authority": False,
        "historical_evidence_preserved": True,
        "projection_hardening": historical_projection,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "shanghai_public_advantage_v3.json"
    md_path = OUTPUT_DIR / "shanghai_public_advantage_v3.md"
    digest_path = OUTPUT_DIR / "shanghai_public_advantage_v3.sha256"
    if json_path.exists():
        previous = read_json(json_path)
        previous_version = str(previous.get("version") or "unknown").replace("/", "-")
        previous_generated = str(previous.get("generated_at") or "unknown").replace(":", "-")
        if previous_version != str(config["version"]):
            history_dir = OUTPUT_DIR / "history" / f"advantage-{previous_version}-{previous_generated}"
            history_dir.mkdir(parents=True, exist_ok=True)
            for source in (json_path, md_path, digest_path):
                if source.exists():
                    shutil.copy2(source, history_dir / source.name)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = payload["selected"]
    lines = [
        "# V3 Shanghai public-data advantage evidence",
        "",
        f"- Status: `{payload['claim_status']}`",
        f"- Selected policy family: **{selected['name']}**",
        f"- Dataset SHA-256: `{dataset.fingerprint}`",
        f"- Weighted relative improvement vs FCFS: **{100 * selected['weighted_relative_improvement']['mean']:.2f}%**",
        f"- 95% bootstrap CI: **[{100 * selected['weighted_relative_improvement']['ci_low']:.2f}%, {100 * selected['weighted_relative_improvement']['ci_high']:.2f}%]**",
        "",
        "| Metric | Relative improvement | 95% CI | Direction |",
        "|---|---:|---:|---|",
    ]
    for metric, summary in selected["metrics_relative_to_fcfs"].items():
        direction = "higher is better" if metric == "throughput_teu" else "lower is better"
        lines.append(
            f"| {metric} | {100 * summary['mean']:.2f}% | [{100 * summary['ci_low']:.2f}%, {100 * summary['ci_high']:.2f}%] | {direction} |"
        )
    lines.extend(
        [
            "",
            "The algorithm was selected on chronological validation rows. This final deterministic-policy comparison uses the untouched chronological blind test only; blind-test scores did not select the winner.",
            payload["claim_boundary"],
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    digest_path.write_text(
        f"{file_sha256(json_path)}  {json_path.name}\n{file_sha256(md_path)}  {md_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": winner, "claim_status": payload["claim_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
