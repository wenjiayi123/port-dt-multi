"""Append-only interactive evaluations, separate from formal training evidence.

The rollout mirrors the frozen TrainingManager evaluator, but never invokes
its benchmark/registry/status writers. Repeated user tests are not fresh blind
experiments and cannot promote a model.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .trainer import ALGORITHMS, _read_json, utc_now
from .datasets import load_port_dataset
from .identifiers import validate_identifier
from .statistics import summarize_metric_rows
from .runtime_policy import load_runtime_policy, resolve_runtime_artifact


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def evaluate_user_run(manager, job_id: str, episodes: int = 10) -> Dict[str, Any]:
    job_id = validate_identifier(job_id, field="job_id")
    if not manager.evaluation_slots.acquire(blocking=False):
        raise ValueError(f"evaluation capacity reached ({manager.max_concurrent_evaluation}); retry later")
    try:
        run_dir = manager.run_dir(job_id)
        tracked = [run_dir / name for name in ("config.json", "manifest.json", "model.zip", "evaluation.json", "evaluation_trajectory.json", "status.json", "model_card.json", "MODEL_CARD.md")]
        config = _read_json(run_dir / "config.json", {})
        manifest = _read_json(run_dir / "manifest.json", {})
        model_artifact = None
        if config.get("algorithm") not in {"mpc", "fcfs"}:
            resolved_model = resolve_runtime_artifact(run_dir, manifest.get("model_sha256"))
            if resolved_model not in tracked:
                tracked.append(resolved_model)
            model_artifact = {"training_sha256": manifest.get("model_sha256"), "runtime_sha256": _sha(resolved_model), "metadata_only_public_export": resolved_model.resolve() != (run_dir / "model.zip").resolve()}
        before = {str(path): _sha(path) for path in tracked if path.is_file()}
        result = _compute_user_evaluation(manager, job_id, episodes)
        after = {str(path): _sha(path) for path in tracked if path.is_file()}
        if before != after:
            raise ValueError("source training artifacts changed during user evaluation; result not admitted")
        evaluation_id = "ui-eval-" + uuid.uuid4().hex
        output = manager.run_root.parent / "user_evaluations" / job_id / evaluation_id
        output.mkdir(parents=True, exist_ok=False)
        result.update({
            "evaluation_id": evaluation_id,
            "evaluation_kind": "interactive_repeated_holdout_replay",
            "formal_evidence_updated": False,
            "benchmark_registry_updated": False,
            "model_registry_updated": False,
            "production_authority": False,
            "model_artifact": model_artifact,
            "claim_boundary": "User-requested repeated evaluation of an existing model; not a new blind experiment or promotion.",
            "evaluation_artifacts": {
                "result_url": f"/api/rl/train/{job_id}/evaluation-runs/{evaluation_id}",
                "history_url": f"/api/rl/train/{job_id}/evaluation-runs",
            },
        })
        _write_new(output / "evaluation.json", result)
        _write_new(output / "provenance.json", {
            "evaluation_id": evaluation_id,
            "source_job_id": job_id,
            "source_files_sha256": {Path(path).name: digest for path, digest in before.items()},
            "source_artifacts_unchanged": before == after,
            "model_artifact": model_artifact,
            "evaluator_source_sha256": _sha(Path(__file__)),
            "frozen_trainer_source_sha256": _sha(Path(__file__).with_name("trainer.py")),
            "evaluation_sha256": _sha(output / "evaluation.json"),
            "gradient_updates": 0,
            "formal_or_champion_registration": False,
        })
        return result
    finally:
        manager.evaluation_slots.release()


def _compute_user_evaluation(manager, job_id: str, episodes: int = 10) -> Dict[str, Any]:
    job_id = validate_identifier(job_id, field="job_id")
    run_dir = manager.run_dir(job_id)
    status = manager._resolve_status(job_id)
    if status.get("status") not in {"COMPLETED", "EVALUATED"}:
        raise ValueError("training must complete before evaluation/rendering")
    config = _read_json(run_dir / "config.json", {})
    dataset = load_port_dataset(config["dataset_id"], manager.data_root)
    env = manager._make_env(dataset, config, training=False, record_trace=True)
    try:
        policy = load_runtime_policy(manager, config, run_dir, env)
        episode_metrics: List[Dict[str, float]] = []
        first_trace: List[Dict[str, Any]] = []
        requested_episodes = max(1, min(50, int(episodes)))
        max_start = max(0, len(env.segment) - config["episode_steps"] - 1)
        start_indices = np.linspace(0, max_start, num=min(requested_episodes, max_start + 1), dtype=int).tolist()
        for episode, start_index in enumerate(start_indices):
            obs, _ = env.reset(seed=config["seed"] + episode, options={"start_index": start_index})
            terminated = truncated = False
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            while not (terminated or truncated):
                if config["algorithm"] == "recurrent_ppo":
                    action, recurrent_state = policy.predict(
                        obs,
                        state=recurrent_state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    episode_start[0] = False
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(action)
            row = env.totals
            row["delay_index_mean"] = row.pop("delay") / max(1, config["episode_steps"])
            row["guardrail_violation_rate"] = row.pop("violations") / max(1, config["episode_steps"])
            episode_metrics.append(row)
            if episode == 0:
                first_trace = list(env.trace)
        keys = episode_metrics[0].keys()
        metrics = {key: float(np.mean([item[key] for item in episode_metrics])) for key in keys}
        uncertainty = summarize_metric_rows(episode_metrics, seed=config["seed"])
        result = {
            "job_id": job_id,
            "algorithm": config["algorithm"],
            "implementation": ALGORITHMS[config["algorithm"]].implementation,
            "dataset_id": dataset.dataset_id,
            "dataset_sha256": dataset.fingerprint,
            "port_profile_id": config.get("port_profile_id"),
            "business_profile_id": config.get("business_profile_id") or "default_port_profile",
            "environment_version": config.get("environment_version", "port_ops_v1"),
            "observation_dimensions": config.get("observation_dimensions"),
            "action_dimensions": config.get("action_dimensions"),
            "split": (
                "chronological_blind_test_only"
                if float(config.get("validation_ratio") or 0.0) > 0
                else "chronological_test_holdout_only"
            ),
            "episodes": len(episode_metrics),
            "metrics": metrics,
            "uncertainty": uncertainty,
            "episode_metrics": episode_metrics,
            "evaluation_protocol": {
                "deterministic_policy": True,
                "render_during_policy_execution": False,
                "holdout": (
                    "chronological_blind_test_only"
                    if float(config.get("validation_ratio") or 0.0) > 0
                    else "chronological_test_only"
                ),
                "window_start_indices": start_indices,
                "confidence_interval": "95% percentile bootstrap of episode means",
            },
            "render": {"type": "trajectory", "frames": first_trace, "frame_count": len(first_trace)},
            "evaluated_at": utc_now(),
        }
        return result
    finally:
        env.close()
