from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROCESS_FIELDS = (
    "epoch",
    "optimizer_updates",
    "imitation_loss",
    "validation_reward_mean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_metric_paths(repo_root: Path, latest: dict[str, Any]) -> list[Path]:
    report_path = repo_root / str(latest.get("report_path") or "")
    if not report_path.is_file():
        return []
    return sorted(report_path.parent.glob("seed_*/metrics.jsonl"))


def checkpoint_reward_replay_path(repo_root: Path, latest: dict[str, Any]) -> Path:
    report_path = repo_root / str(latest.get("report_path") or "")
    return report_path.parent / "checkpoint_reward_replay.json"


def load_checkpoint_reward_replay(repo_root: Path, latest: dict[str, Any]) -> dict[str, Any]:
    """Load derived validation replay while preserving its non-training boundary."""
    path = checkpoint_reward_replay_path(repo_root, latest)
    if not path.is_file():
        return {
            "schema": "port-dt-checkpoint-reward-replay.v1",
            "available": False,
            "reason": "derived checkpoint reward replay has not been exported",
            "series": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if payload.get("run_id") != latest.get("run_id"):
        raise ValueError(f"checkpoint reward replay run mismatch: {path}")
    if payload.get("training_time_log") is not False or payload.get("blind_test_access") is not False:
        raise ValueError(f"checkpoint reward replay claim boundary is invalid: {path}")
    for seed in payload.get("series") or []:
        for point in seed.get("points") or []:
            for field in ("reward_block_mean", "reward_delta_from_epoch1"):
                if not math.isfinite(float(point[field])):
                    raise ValueError(f"non-finite {field} in {path}")
    payload["available"] = True
    payload["artifact_path"] = str(path.relative_to(repo_root))
    payload["artifact_sha256"] = _sha256(path)
    return payload


def load_seed_process_evidence(repo_root: Path, latest: dict[str, Any]) -> dict[str, Any]:
    """Expose persisted optimizer checkpoints without interpolation or fabrication."""
    report_path = repo_root / str(latest.get("report_path") or "")
    report = json.loads(report_path.read_text(encoding="utf-8", errors="strict"))
    training = report.get("training") or {}
    algorithm_registry = report.get("algorithm_registry") or []
    fine_tune_rows = [
        row
        for row in algorithm_registry
        if "fine-tune" in str(row.get("name") or "").lower()
    ]
    training_method = {
        "method_family": "constraint_projected_teacher_actor_distillation",
        "display_name_cn": "安全教师策略蒸馏",
        "display_name_en": "constraint-projected teacher actor distillation",
        "declared_algorithm": training.get("algorithm"),
        "actor_runtime": training.get("actor_runtime") or "exported deterministic MLP actor",
        "teacher": training.get("teacher") or "constraint-aware rule/MPC teacher",
        "optimizer_objective": "teacher_action_mean_squared_error",
        "environment_reward_optimized": False,
        "policy_gradient_updates": False,
        "q_function_updates": False,
        "validation_reward_is_training_reward": False,
        "checkpoint_reward_replay_is_training_log": False,
        "checkpoint_selection": training.get("selection")
        or training.get("safe_policy_improvement")
        or "fixed validation reward/business/safety gates",
        "rl_fine_tune": fine_tune_rows or [{
            "name": "RL fine-tune",
            "state": "not_admitted",
            "reason": "no independently admitted environment-reward fine-tune in this evidence bundle",
        }],
        "claim_boundary_cn": "当前晋级策略是教师动作监督蒸馏；PPO仅在部分模块承担策略网络/推理载体，不代表执行过PPO策略梯度更新。",
        "report_path": str(report_path.relative_to(repo_root)),
        "report_sha256": _sha256(report_path),
    }
    series: list[dict[str, Any]] = []
    total_records = 0
    for path in seed_metric_paths(repo_root, latest):
        points: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            point: dict[str, Any] = {"timestamp": row.get("ts")}
            for field in PROCESS_FIELDS:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {field} in {path}")
                point[field] = int(value) if field in {"epoch", "optimizer_updates"} else value
            points.append(point)
        points.sort(key=lambda item: item["epoch"])
        if not points:
            continue
        seed_value = path.parent.name.removeprefix("seed_")
        seed = int(seed_value) if seed_value.isdigit() else seed_value
        total_records += len(points)
        series.append(
            {
                "seed": seed,
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256(path),
                "records": len(points),
                "points": points,
            }
        )
    series.sort(key=lambda item: int(item["seed"]) if isinstance(item["seed"], int) else str(item["seed"]))
    return {
        "schema": "port-dt-persisted-training-process.v1",
        "source": "append_only_seed_metrics_jsonl",
        "run_id": latest.get("run_id"),
        "retrained_for_display": False,
        "interpolated_points": False,
        "frontend_random_noise": False,
        "training_method": training_method,
        "total_persisted_checkpoints": total_records,
        "metric_definitions": {
            "imitation_loss": "backend optimizer actor imitation loss; lower is better",
            "validation_reward_mean": "post-checkpoint fixed validation-window mean reward; it is not an on-policy training reward",
        },
        "series": series,
    }
