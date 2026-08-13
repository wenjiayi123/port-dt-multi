"""Export a portable, integrity-checked RL benchmark evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.services.rl_training.datasets import file_sha256, load_port_dataset
from app.services.rl_training.trainer import ALGORITHMS, TRAINING_MANAGER


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def safe_run_record(run: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(run["job_id"])
    run_dir = TRAINING_MANAGER.run_dir(job_id)
    config = read_json(run_dir / "config.json")
    manifest = read_json(run_dir / "manifest.json")
    evaluation = read_json(run_dir / "evaluation.json")
    algorithm = str(config["algorithm"])
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm in run {job_id}: {algorithm}")
    dataset = load_port_dataset(config["dataset_id"], TRAINING_MANAGER.data_root)
    training_dataset_hash = str(manifest.get("dataset_sha256") or "")
    current_dataset_match = training_dataset_hash == dataset.fingerprint
    model_integrity: Dict[str, Any]
    if ALGORITHMS[algorithm].trainable:
        model_path = run_dir / "model.zip"
        observed_hash = file_sha256(model_path)
        expected_hash = str(manifest.get("model_sha256") or "")
        if observed_hash != expected_hash:
            raise ValueError(f"model hash mismatch in run {job_id}")
        model_integrity = {
            "artifact_id": "model.zip",
            "sha256": observed_hash,
            "bytes": model_path.stat().st_size,
            "verified": True,
        }
    else:
        model_integrity = {
            "artifact_id": None,
            "controller_parameters": manifest.get("controller_parameters"),
            "verified": True,
        }
    profile = config.get("port_profile") or {}
    split = manifest.get("split") or {}
    optimizer_history: list[dict[str, Any]] = []
    history_path = run_dir / "metrics.jsonl"
    if history_path.is_file():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                optimizer_history.append(
                    {
                        key: row.get(key)
                        for key in (
                            "progress", "step", "reward_mean", "actor_loss",
                            "critic_loss", "policy_gradient_loss", "value_loss",
                            "entropy_loss", "exploration_rate", "updates",
                        )
                        if row.get(key) is not None
                    }
                )
    validation_path = run_dir / "validation_evaluation.json"
    validation_evaluation = (
        read_json(validation_path) if validation_path.is_file() else None
    )
    return {
        "job_id": job_id,
        "algorithm": algorithm,
        "implementation": manifest.get("implementation"),
        "seed": config.get("seed"),
        "total_steps": config.get("total_steps"),
        "episode_steps": config.get("episode_steps"),
        "episode_hours": config.get("episode_hours"),
        "environment_version": config.get("environment_version", "port_ops_v1"),
        "business_profile_id": config.get("business_profile_id") or "default_port_profile",
        "port_profile_id": config.get("port_profile_id"),
        "port_profile": {
            "port_code": profile.get("port_code"),
            "calibration_status": profile.get("calibration_status"),
            "control_authority": profile.get("control_authority"),
            "objectives": profile.get("objectives"),
        },
        "observation_dimensions": config.get("observation_dimensions"),
        "action_dimensions": config.get("action_dimensions"),
        "algorithm_parameters": config.get("algorithm_parameters"),
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": training_dataset_hash,
        "dataset_integrity": {
            "current_artifact_id": dataset.path.name,
            "current_artifact_sha256": dataset.fingerprint,
            "current_artifact_matches_training": current_dataset_match,
            "historical_training_artifact_available": current_dataset_match,
        },
        "evidence_label": run.get("evidence_label"),
        "training": {
            "render_calls": manifest.get("render_calls_during_training"),
            "total_steps_observed": manifest.get("total_steps_observed"),
            "training_device": manifest.get("training_device") or "cpu",
            "runtime": manifest.get("runtime"),
            "split": {
                name: split.get(name)
                for name in (
                    "train_rows",
                    "validation_rows",
                    "test_rows",
                    "split_method",
                    "test_ratio",
                    "validation_ratio",
                )
            },
            "completed_at": manifest.get("completed_at"),
            "optimizer_history": optimizer_history,
        },
        "validation_evaluation": validation_evaluation,
        "evaluation": {
            "split": evaluation.get("split"),
            "episodes": evaluation.get("episodes"),
            "metrics": evaluation.get("metrics"),
            "uncertainty": evaluation.get("uncertainty"),
            "evaluation_protocol": evaluation.get("evaluation_protocol"),
            "evaluated_at": evaluation.get("evaluated_at"),
        },
        "model_integrity": model_integrity,
    }


def markdown_summary(bundle: Dict[str, Any]) -> str:
    dataset = bundle["dataset"]
    validation_text = (
        f" / {int(dataset['validation_rows']):,} validation"
        if int(dataset.get("validation_rows") or 0) > 0
        else ""
    )
    lines = [
        f"# RL benchmark evidence: `{dataset['dataset_id']}`",
        "",
        f"- Dataset SHA-256: `{dataset['sha256']}`",
        f"- Rows: {int(dataset['rows']):,}",
        f"- Split: {int(dataset['train_rows']):,} train{validation_text} / {int(dataset['test_rows']):,} chronological blind holdout",
        "- Training rendering: disabled and verified per run",
        "- Comparative gate: at least three seeds and 10,000 optimizer steps per RL method",
        "",
        "| Controller | Formal runs | Seeds | Gate | Reward | Energy cost | Carbon kg | Throughput TEU | Peak kW | Delay index | Violations |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bundle["benchmark_summary"]["algorithms"]:
        metrics = item.get("metrics") or {}
        current_runs = [
            run
            for run in bundle["runs"]
            if run.get("algorithm") == item["id"]
            and run.get("dataset_integrity", {}).get(
                "current_artifact_matches_training"
            )
            is True
            and run.get("evidence_label")
            in {
                "RL_HELD_OUT_EVALUATION",
                "DETERMINISTIC_CONTROLLER_BASELINE",
            }
            and (
                run.get("algorithm") != "mpc"
                or int(run.get("evaluation", {}).get("episodes") or 0) >= 10
            )
        ]
        current_seeds = sorted(
            {
                int(run["seed"])
                for run in current_runs
                if isinstance(run.get("seed"), int)
            }
        )
        current_ready = (
            len(current_seeds) >= 3
            if item.get("trainable")
            else len(current_runs) >= 1
        )

        def mean(name: str) -> str:
            if not current_ready:
                return "—"
            value = (metrics.get(name) or {}).get("mean")
            return "—" if value is None else f"{float(value):,.4f}"

        seeds = ", ".join(str(value) for value in current_seeds)
        lines.append(
            "| "
            + " | ".join(
                (
                    item["name"],
                    str(len(current_runs)),
                    seeds or "—",
                    (
                        "PASS"
                        if current_ready
                        else "STALE ARTIFACT"
                        if item.get("claim_eligible_runs")
                        else "PENDING"
                    ),
                    mean("reward"),
                    mean("energy_cost"),
                    mean("carbon_kg"),
                    mean("throughput_teu"),
                    mean("peak_kw"),
                    mean("delay_index_mean"),
                    mean("guardrail_violation_rate"),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "`RL_SMOKE_WIRING_ONLY` runs are retained in the JSON bundle but excluded from this performance table.",
            "The figures are deterministic-policy results on the chronological public-data holdout, not measured terminal KPIs or production savings.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export portable benchmark evidence")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="evidence/rl")
    args = parser.parse_args()
    dataset = load_port_dataset(args.dataset, TRAINING_MANAGER.data_root)
    validation_ratio = 0.1 if (dataset.metadata.get("split_policy") or {}).get("validation") else 0.0
    registry = read_json(TRAINING_MANAGER.benchmark_path)
    selected = [
        run
        for run in registry.get("runs") or []
        if run.get("dataset_id") == dataset.dataset_id
    ]
    if not selected:
        raise RuntimeError(f"no persisted evaluations found for {dataset.dataset_id}")
    declared_environment = str(dataset.metadata.get("environment_version") or "") or None
    runs = [safe_run_record(run) for run in selected]
    if declared_environment:
        runs = [
            run
            for run in runs
            if run.get("environment_version") == declared_environment
        ]
    runs = [
        run
        for run in runs
        if run.get("business_profile_id") == "default_port_profile"
    ]
    runs.sort(key=lambda item: (list(ALGORITHMS).index(item["algorithm"]), int(item.get("seed") or 0)))
    bundle = {
        "schema": "port-dt-rl-benchmark-evidence.v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": dataset.describe(validation_ratio=validation_ratio),
        "benchmark_summary": TRAINING_MANAGER.benchmark_summary(
            dataset.dataset_id,
            environment_version=declared_environment,
            business_profile_id="default_port_profile",
        ),
        "runs": runs,
        "evidence_boundary": {
            "training_status_source": "stable_baselines3_callback_or_deterministic_controller",
            "training_rendering": False,
            "evaluation_split": (
                "chronological_blind_test_only"
                if validation_ratio > 0
                else "chronological_test_holdout_only"
            ),
            "production_kpi_claim": False,
            "site_dispatch_claim": False,
            "smoke_runs_are_not_performance_evidence": True,
            "environment_version": declared_environment,
            "business_profile_id": "default_port_profile",
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset.dataset_id}_benchmark.json"
    encoded = json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8")
    output_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    output_path.with_suffix(".sha256").write_text(
        f"{digest}  {output_path.name}\n",
        encoding="utf-8",
    )
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(markdown_summary(bundle), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "markdown": str(markdown_path),
                "sha256": digest,
                "runs": len(runs),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
