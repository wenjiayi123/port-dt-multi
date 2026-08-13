from __future__ import annotations

import json
import math
import os
import platform
import threading
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .datasets import (
    DEFAULT_DATA_ROOT,
    FACTOR_COLUMNS,
    NUMERIC_COLUMNS,
    PortDataset,
    file_sha256,
    list_datasets,
    load_port_dataset,
)
from .environment import PortOperationsEnv, dataset_quality_cadence
from .baselines import FCFSNeutralPolicy
from .mpc import MPCPolicy
from .model_registry import ModelRegistry
from .identifiers import resolve_child_dir, validate_identifier
from .safety import assess_recommendation
from .statistics import bootstrap_summary, summarize_metric_rows
from .profiles import DEFAULT_PROFILE_ID, list_profiles, load_profile


SB3_IMPORT_LOCK = threading.RLock()


RUN_ROOT = Path("data/rl/runs")
BENCHMARK_PATH = Path("data/rl/benchmarks.json")
MODEL_REGISTRY_PATH = Path("data/rl/model_registry.json")


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    family: str
    action_space: str
    trainable: bool
    implementation: str
    description: str


ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "sac": AlgorithmSpec("sac", "SAC", "RL", "continuous", True, "stable_baselines3.SAC", "Maximum-entropy off-policy actor-critic for continuous port setpoints."),
    "ppo": AlgorithmSpec("ppo", "PPO", "RL", "continuous", True, "stable_baselines3.PPO", "Clipped on-policy optimization for stable constrained control."),
    "td3": AlgorithmSpec("td3", "TD3", "RL", "continuous", True, "stable_baselines3.TD3", "Twin critics and delayed actor updates for continuous dispatch."),
    "dqn": AlgorithmSpec("dqn", "DQN", "RL", "discrete", True, "stable_baselines3.DQN", "Replay-buffer Q-learning on an explicit finite port-control lattice."),
    "a2c": AlgorithmSpec("a2c", "A2C", "RL", "continuous", True, "stable_baselines3.A2C", "Synchronous advantage actor-critic baseline for low-overhead on-policy optimization."),
    "tqc": AlgorithmSpec("tqc", "TQC", "RL", "continuous", True, "sb3_contrib.TQC", "Distributional off-policy actor-critic with truncated quantile critics."),
    "qrdqn": AlgorithmSpec("qrdqn", "QR-DQN", "RL", "discrete", True, "sb3_contrib.QRDQN", "Distributional value learning for a finite auditable port-control lattice."),
    "trpo": AlgorithmSpec("trpo", "TRPO", "RL", "continuous", True, "sb3_contrib.TRPO", "Trust-region policy optimization for conservative on-policy updates."),
    "recurrent_ppo": AlgorithmSpec("recurrent_ppo", "Recurrent PPO", "RL", "continuous", True, "sb3_contrib.RecurrentPPO", "LSTM policy for delayed congestion and equipment-availability effects."),
    "ars": AlgorithmSpec("ars", "ARS", "RL", "continuous", True, "sb3_contrib.ARS", "Derivative-free random-search policy baseline for robustness comparison."),
    "mpc": AlgorithmSpec("mpc", "MPC", "Control", "continuous", False, "scipy.optimize.minimize", "Receding-horizon model predictive control baseline."),
    "fcfs": AlgorithmSpec("fcfs", "FCFS neutral", "Rule", "continuous", False, "port_dt.FCFSNeutralPolicy", "Non-optimizing first-come-first-served comparator with neutral energy and allocation commands."),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        # Training artifacts are fixed filenames below resolve_child_dir roots.
        # codeql[py/path-injection]
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one durable, JSON-safe metrics observation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json.loads(json.dumps(payload, default=lambda value: value.item() if hasattr(value, "item") else str(value)))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")


class TrainingStopped(Exception):
    pass


class TrainingJob:
    def __init__(self, job_id: str, config: Dict[str, Any], manager: "TrainingManager") -> None:
        self.job_id = job_id
        self.config = config
        self.manager = manager
        self.run_dir = manager.run_dir(job_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.cancel_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.rewards: deque[float] = deque(maxlen=200)
        self.logs: deque[str] = deque(maxlen=80)
        self.status: Dict[str, Any] = {
            "job_id": job_id,
            "status": "QUEUED",
            "progress": 0.0,
            "stage": "queued",
            "step": 0,
            "total_steps": int(config["total_steps"]),
            "algorithm": config["algorithm"],
            "dataset_id": config["dataset_id"],
            "metrics": {},
            "logs": [],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "artifact_paths": {
                "run_id": job_id,
                "root_url": f"/api/rl/models/{job_id}",
                "model_artifact_id": "model.zip" if ALGORITHMS[config["algorithm"]].trainable else None,
                "manifest_artifact_id": "manifest.json",
            },
            "rendering": {"enabled": False, "render_calls": 0, "trace_rows": 0},
            "evaluation_available": False,
        }
        self.log("job queued; backend owns progress and metrics")
        self.persist()

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs.appendleft(f"[{stamp}] {message}")
        self.status["logs"] = list(self.logs)

    def update(self, **changes: Any) -> None:
        with self.lock:
            self.status.update(changes)
            self.status["updated_at"] = utc_now()
            self.status["logs"] = list(self.logs)
            self.status["summary"] = self.summary()
            self.persist()

    def summary(self) -> str:
        metrics = self.status.get("metrics") or {}
        reward = metrics.get("reward_mean")
        reward_text = "—" if reward is None else f"{float(reward):.5f}"
        return (
            f"{self.status.get('status')} · {self.config['algorithm'].upper()} · "
            f"step={int(self.status.get('step', 0)):,}/{int(self.status.get('total_steps', 0)):,} · "
            f"reward_mean={reward_text} · dataset={self.config['dataset_id']}"
        )

    def persist(self) -> None:
        _write_json(self.run_dir / "status.json", self.status)
        _write_json(self.run_dir / "config.json", self.config)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.status))


class TrainingManager:
    def __init__(
        self,
        data_root: Path = DEFAULT_DATA_ROOT,
        run_root: Path = RUN_ROOT,
        benchmark_path: Path = BENCHMARK_PATH,
    ) -> None:
        self.data_root = data_root
        self.run_root = run_root
        self.benchmark_path = benchmark_path
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, TrainingJob] = {}
        self.lock = threading.RLock()
        # Lazy imports in SB3/sb3-contrib are not safe when two first-time
        # inference requests initialize the package concurrently.
        self.policy_load_lock = SB3_IMPORT_LOCK
        self.max_concurrent_training = _bounded_env_int("PORT_DT_MAX_CONCURRENT_TRAINING", 1, 1, 32)
        self.max_concurrent_evaluation = _bounded_env_int("PORT_DT_MAX_CONCURRENT_EVALUATION", 2, 1, 32)
        self.max_training_steps = _bounded_env_int("PORT_DT_MAX_TRAINING_STEPS", 5_000_000, 64, 50_000_000)
        self.evaluation_slots = threading.BoundedSemaphore(self.max_concurrent_evaluation)
        self.latest_job_id: Optional[str] = None
        self._restore_statuses()

    def run_dir(self, job_id: str) -> Path:
        return resolve_child_dir(self.run_root, job_id, field="job_id")

    def _restore_statuses(self) -> None:
        for path in sorted(self.run_root.glob("*/status.json")):
            payload = _read_json(path, {})
            try:
                job_id = validate_identifier(path.parent.name, field="job_id")
                if validate_identifier(payload.get("job_id"), field="job_id") != job_id:
                    continue
            except ValueError:
                continue
            if payload.get("status") in {"QUEUED", "RUNNING", "PAUSED", "EVALUATING"}:
                payload.update(status="INTERRUPTED", stage="server_restarted", updated_at=utc_now())
                _write_json(path, payload)
            self.latest_job_id = job_id

    def capabilities(self) -> Dict[str, Any]:
        try:
            with self.policy_load_lock:
                import gymnasium
                import sb3_contrib
                import stable_baselines3
                import torch

            runtime = {
                "available": True,
                "stable_baselines3": stable_baselines3.__version__,
                "sb3_contrib": sb3_contrib.__version__,
                "gymnasium": gymnasium.__version__,
                "torch": torch.__version__,
                "training_device": "cpu",
                "available_accelerator": "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else None,
            }
        except Exception:
            runtime = {
                "available": False,
                "error": "RL runtime import failed; inspect the dependency installation",
            }
        return {
            "engine": "port-rl-engine-v1",
            "runtime": runtime,
            "algorithms": [asdict(spec) for spec in ALGORITHMS.values()],
            "datasets": list_datasets(self.data_root),
            "port_profiles": list_profiles(),
            "contracts": {
                "port_ops_v1": {
                    "observation_dimensions": 13,
                    "observation_fields": [
                        "hour_sin", "hour_cos", *NUMERIC_COLUMNS,
                        "soc", "queue_pressure", "last_bess_power", "episode_progress",
                    ],
                    "continuous_action_dimensions": 3,
                    "actions": ["bess_power", "service_factor", "flexible_load"],
                },
                "port_ops_v2": {
                    "observation_dimensions": 13 + 2 * len(FACTOR_COLUMNS),
                    "observation_fields": [
                        "hour_sin", "hour_cos", *NUMERIC_COLUMNS,
                        *FACTOR_COLUMNS,
                        *[f"{name}_available" for name in FACTOR_COLUMNS],
                        "soc", "queue_pressure", "last_bess_power", "episode_progress",
                    ],
                    "continuous_action_dimensions": 5,
                    "actions": [
                        "bess_power", "service_factor", "flexible_load",
                        "berth_priority", "yard_flow",
                    ],
                    "missing_factor_policy": "neutral_value_plus_explicit_availability_mask",
                },
                "port_ops_v3": {
                    "observation_dimensions": 13 + 2 * len(FACTOR_COLUMNS),
                    "observation_fields": [
                        "hour_sin", "hour_cos", *NUMERIC_COLUMNS,
                        *FACTOR_COLUMNS,
                        *[f"{name}_available" for name in FACTOR_COLUMNS],
                        "soc", "queue_pressure", "last_bess_power", "episode_progress",
                    ],
                    "continuous_action_dimensions": 5,
                    "actions": [
                        "bess_power", "service_factor", "flexible_load",
                        "berth_priority", "yard_flow",
                    ],
                    "missing_factor_policy": "neutral_value_plus_explicit_availability_mask",
                    "causal_coupling": "service and allocation actions change operational electric load",
                    "raw_action_projection_diagnostics": [
                        "event_rate", "mean_correction_kw", "mean_severity",
                        "grid_cap", "soc_bound", "terminal_reachability", "power_bound",
                    ],
                    "projection_penalty": "explicit opt-in run parameter; absent historical configs remain reproducible",
                },
            },
            "training_rendering": "disabled",
            "evaluation_rendering": "trajectory_json_only",
            "python": platform.python_version(),
            "resource_limits": {
                "max_concurrent_training": self.max_concurrent_training,
                "max_concurrent_evaluation": self.max_concurrent_evaluation,
                "max_training_steps": self.max_training_steps,
            },
        }

    def validate_config(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        algorithm = str(raw.get("algorithm") or "sac").lower()
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of: {', '.join(ALGORITHMS)}")
        dataset_id = str(raw.get("dataset_id") or "public_port_ops_v1")
        dataset = load_port_dataset(dataset_id, self.data_root)
        profile_id = str(
            raw.get("port_profile_id")
            or dataset.metadata.get("port_profile_id")
            or DEFAULT_PROFILE_ID
        )
        profile = load_profile(profile_id)
        quality = dataset.describe().get("quality") or {}
        if quality.get("training_eligible") is not True:
            raise ValueError("dataset failed the quality gate: " + "; ".join(quality.get("errors") or ["unknown error"]))
        trainable = ALGORITHMS[algorithm].trainable
        total_steps = int(raw.get("total_steps") or 10000) if trainable else 0
        if trainable and not 64 <= total_steps <= self.max_training_steps:
            raise ValueError(f"total_steps must be between 64 and {self.max_training_steps:,}")
        environment_version = str(
            raw.get("environment_version")
            or dataset.metadata.get("environment_version")
            or profile.get("environment_version")
            or "port_ops_v1"
        )
        if environment_version not in {"port_ops_v1", "port_ops_v2", "port_ops_v3"}:
            raise ValueError("environment_version must be port_ops_v1, port_ops_v2 or port_ops_v3")
        required_factors = list(profile["factor_requirements"].get("required_for_training") or [])
        factor_coverage = quality.get("factor_coverage") or {}
        unavailable_required = [
            name for name in required_factors
            if float(factor_coverage.get(name) or 0.0) <= 0.0
        ]
        if unavailable_required:
            raise ValueError(
                "dataset lacks factors required by port profile: "
                + ", ".join(unavailable_required)
            )
        cadence_hours = dataset_quality_cadence(dataset) / 3600.0
        episode_hours = max(1.0, float(raw.get("episode_hours") or 48.0))
        default_episode_steps = min(
            max(4, dataset.rows // 10),
            max(4, round(episode_hours / cadence_hours)),
        )
        episode_steps = int(raw.get("episode_steps") or default_episode_steps)
        bounded_episode_steps = max(4, episode_steps)
        rollout_steps = min(512, max(32, bounded_episode_steps))
        algorithm_parameters: Dict[str, Any] = {
            "policy": "MlpPolicy",
            "network_architecture": [64, 64],
        }
        if algorithm == "qrdqn":
            algorithm_parameters.update(
                n_quantiles=50,
                train_frequency=8,
                gradient_steps=1,
            )
        elif algorithm == "trpo":
            algorithm_parameters.update(n_steps=rollout_steps)
        elif algorithm == "recurrent_ppo":
            algorithm_parameters.update(
                policy="MlpLstmPolicy",
                n_steps=rollout_steps,
                lstm_hidden_size=64,
            )
        elif algorithm == "ars":
            algorithm_parameters.update(
                network_architecture=[32, 32],
                n_delta=4,
                n_top=2,
                delta_std=0.03,
                zero_policy=False,
            )
        test_ratio = float(raw.get("test_ratio") or 0.2)
        declared_validation = 0.1 if (dataset.metadata.get("split_policy") or {}).get("validation") else 0.0
        validation_ratio = float(
            raw["validation_ratio"]
            if raw.get("validation_ratio") is not None
            else declared_validation
        )
        if validation_ratio and not 0.05 <= validation_ratio <= 0.2:
            raise ValueError("validation_ratio must be 0 or between 0.05 and 0.2")
        if test_ratio + validation_ratio > 0.5:
            raise ValueError("test_ratio + validation_ratio must not exceed 0.5")
        reward_weights = dict(raw.get("reward_weights") or {})
        if "safety" not in reward_weights and raw.get("safety_weight") is not None:
            reward_weights["safety"] = float(raw["safety_weight"])
        projection_penalty_weight = min(
            10.0,
            max(0.0, float(raw.get("projection_penalty_weight") or 0.0)),
        )
        return {
            **raw,
            "algorithm": algorithm,
            "dataset_id": dataset.dataset_id,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_quality_status": quality.get("status"),
            "dataset_evidence": quality.get("evidence"),
            "port_profile_id": profile["profile_id"],
            "port_profile": profile,
            "business_profile_id": str(
                raw.get("business_profile_id") or "default_port_profile"
            ),
            "environment_version": environment_version,
            "observation_dimensions": 13 if environment_version == "port_ops_v1" else 13 + 2 * len(FACTOR_COLUMNS),
            "action_dimensions": 3 if environment_version == "port_ops_v1" else 5,
            "total_steps": total_steps,
            "episode_steps": bounded_episode_steps,
            "episode_hours": episode_hours,
            "step_hours": cadence_hours,
            "test_ratio": min(0.4, max(0.1, test_ratio)),
            "validation_ratio": validation_ratio,
            "batch_size": max(16, min(2048, int(raw.get("batch_size") or 256))),
            "learning_rate": min(0.01, max(1e-6, float(raw.get("learning_rate") or 3e-4))),
            "gamma": min(0.9999, max(0.80, float(raw.get("gamma") or 0.99))),
            "tau": min(1.0, max(1e-4, float(raw.get("tau") or 0.005))),
            "replay_buffer": max(1000, min(5_000_000, int(raw.get("replay_buffer") or 100000))),
            "seed": int(raw.get("seed") or 42),
            "demand_cap_kw": max(
                100.0,
                float(raw.get("demand_cap_kw") or profile["assets"]["demand_cap_kw"]),
            ),
            "reward_weights": reward_weights,
            "projection_penalty_weight": projection_penalty_weight,
            "training_split": "chronological_train_only",
            "render_during_training": False,
            "algorithm_parameters": algorithm_parameters,
        }

    def start(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        config = self.validate_config(raw_config)
        with self.lock:
            active = sum(1 for item in self.jobs.values() if item.status.get("status") in {"QUEUED", "RUNNING", "PAUSED"})
            if ALGORITHMS[config["algorithm"]].trainable and active >= self.max_concurrent_training:
                raise ValueError(f"training capacity reached ({active}/{self.max_concurrent_training}); wait or cancel an active job")
            job_id = "rl-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"
            job = TrainingJob(job_id, config, self)
            self.jobs[job_id] = job
            self.latest_job_id = job_id
        if not ALGORITHMS[config["algorithm"]].trainable:
            algorithm = config["algorithm"]
            job.log(f"{ALGORITHMS[algorithm].name} is a non-learning comparator; no training samples consumed")
            dataset = load_port_dataset(config["dataset_id"], self.data_root)
            if algorithm == "mpc":
                controller_parameters = MPCPolicy(
                    action_dim=config["action_dimensions"],
                    episode_steps=config["episode_steps"],
                ).parameters()
            else:
                controller_parameters = FCFSNeutralPolicy(
                    config["action_dimensions"]
                ).parameters()
            manifest = {
                "job_id": job.job_id,
                "algorithm": algorithm,
                "implementation": ALGORITHMS[algorithm].implementation,
                "controller_only": True,
                "controller_parameters": controller_parameters,
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.fingerprint,
                "split": dataset.describe(
                    config["test_ratio"], config.get("validation_ratio", 0.0)
                ),
                "seed": config["seed"],
                "port_profile_id": config["port_profile_id"],
                "business_profile_id": config["business_profile_id"],
                "environment_version": config["environment_version"],
                "observation_dimensions": config["observation_dimensions"],
                "action_dimensions": config["action_dimensions"],
                "algorithm_parameters": config.get("algorithm_parameters"),
                "render_calls_during_training": 0,
                "completed_at": utc_now(),
            }
            _write_json(job.run_dir / "manifest.json", manifest)
            job.update(status="COMPLETED", progress=100.0, stage="controller_ready", evaluation_available=True, manifest=manifest)
            self._sync_model_registry(job.job_id)
        else:
            job.thread = threading.Thread(target=self._run_training, args=(job,), name=job_id, daemon=True)
            job.thread.start()
        return job.snapshot()

    def _make_env(self, dataset: PortDataset, config: Dict[str, Any], *, training: bool, record_trace: bool = False) -> PortOperationsEnv:
        validation_ratio = float(config.get("validation_ratio") or 0.0)
        if validation_ratio > 0:
            train_slice, _validation_slice, test_slice = dataset.split_three_way(
                config["test_ratio"], validation_ratio
            )
        else:
            train_slice, test_slice = dataset.split(config["test_ratio"])
        return PortOperationsEnv(
            dataset,
            train_slice if training else test_slice,
            action_mode=ALGORITHMS[config["algorithm"]].action_space,
            episode_steps=config["episode_steps"],
            seed=config["seed"],
            demand_cap_kw=config["demand_cap_kw"],
            reward_weights=config["reward_weights"],
            projection_penalty_weight=float(config.get("projection_penalty_weight") or 0.0),
            environment_version=config.get("environment_version", "port_ops_v1"),
            port_profile=config.get("port_profile"),
            normalization_slice=train_slice,
            training=training,
            record_trace=record_trace,
        )

    def _run_training(self, job: TrainingJob) -> None:
        env: Optional[PortOperationsEnv] = None
        try:
            with self.policy_load_lock:
                from sb3_contrib import ARS, QRDQN, RecurrentPPO, TQC, TRPO
                from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
                from stable_baselines3.common.callbacks import BaseCallback
                from stable_baselines3.common.monitor import Monitor

            config = job.config
            dataset = load_port_dataset(config["dataset_id"], self.data_root)
            env = self._make_env(dataset, config, training=True)
            monitored = Monitor(env, filename=str(job.run_dir / "monitor.csv"))
            split = dataset.describe(
                config["test_ratio"], config.get("validation_ratio", 0.0)
            )
            job.log(f"chronological train split loaded: {split['train_rows']} rows")
            if split.get("validation_rows"):
                job.log(
                    f"validation={split['validation_rows']} rows; blind test={split['test_rows']} rows"
                )
            job.log("render disabled; evaluation holdout is not visible to trainer")
            job.update(status="RUNNING", stage="environment_ready")

            class RealMetricsCallback(BaseCallback):
                def __init__(self) -> None:
                    super().__init__(verbose=0)
                    self.last_flush = 0

                def _on_step(self) -> bool:
                    if job.cancel_event.is_set():
                        raise TrainingStopped("cancelled by operator")
                    while not job.pause_event.wait(timeout=0.25):
                        if job.cancel_event.is_set():
                            raise TrainingStopped("cancelled by operator")
                    rewards = self.locals.get("rewards")
                    if rewards is not None:
                        for value in np.asarray(rewards).reshape(-1):
                            job.rewards.append(float(value))
                    step = min(int(self.num_timesteps), config["total_steps"])
                    flush_every = max(16, config["total_steps"] // 100)
                    if step - self.last_flush >= flush_every or step >= config["total_steps"]:
                        logger_values = dict(getattr(self.model.logger, "name_to_value", {}) or {})
                        metrics = {
                            "step": step,
                            "reward_mean": float(np.mean(job.rewards)) if job.rewards else None,
                            "actor_loss": logger_values.get("train/actor_loss"),
                            "critic_loss": logger_values.get("train/critic_loss"),
                            "policy_gradient_loss": logger_values.get("train/policy_gradient_loss"),
                            "value_loss": logger_values.get("train/value_loss"),
                            "entropy_loss": logger_values.get("train/ent_coef_loss", logger_values.get("train/entropy_loss")),
                            "exploration_rate": logger_values.get("rollout/exploration_rate"),
                            "updates": logger_values.get("train/n_updates"),
                        }
                        history_record = {
                            "ts": utc_now(),
                            "job_id": job.job_id,
                            "algorithm": config["algorithm"],
                            "dataset_id": config["dataset_id"],
                            "progress": round(100.0 * step / config["total_steps"], 3),
                            **metrics,
                        }
                        _append_jsonl(job.run_dir / "metrics.jsonl", history_record)
                        job.log(f"real optimizer step={step:,}; reward_mean={metrics['reward_mean']}")
                        job.update(
                            status="RUNNING",
                            stage="optimizing_policy_network",
                            step=step,
                            progress=round(100.0 * step / config["total_steps"], 3),
                            metrics=metrics,
                        )
                        self.last_flush = step
                    return step < config["total_steps"]

            common = {
                "policy": "MlpPolicy",
                "env": monitored,
                "learning_rate": config["learning_rate"],
                "gamma": config["gamma"],
                "seed": config["seed"],
                "verbose": 0,
                "device": "cpu",
                "policy_kwargs": {"net_arch": [64, 64]},
            }
            algo = config["algorithm"]
            if algo == "ppo":
                n_steps = min(512, max(32, config["episode_steps"]))
                batch = min(config["batch_size"], n_steps)
                while n_steps % batch:
                    batch -= 1
                model = PPO(**common, n_steps=n_steps, batch_size=max(8, batch), ent_coef=float(config.get("entropy_coef") or 0.0))
            elif algo == "a2c":
                n_steps = min(512, max(32, config["episode_steps"]))
                model = A2C(**common, n_steps=n_steps, ent_coef=float(config.get("entropy_coef") or 0.0))
            elif algo == "sac":
                model = SAC(**common, buffer_size=config["replay_buffer"], batch_size=config["batch_size"], learning_starts=min(1000, max(32, config["total_steps"] // 10)), tau=config["tau"])
            elif algo == "tqc":
                model = TQC(**common, buffer_size=config["replay_buffer"], batch_size=config["batch_size"], learning_starts=min(1000, max(32, config["total_steps"] // 10)), tau=config["tau"])
            elif algo == "td3":
                model = TD3(**common, buffer_size=config["replay_buffer"], batch_size=config["batch_size"], learning_starts=min(1000, max(32, config["total_steps"] // 10)), tau=config["tau"])
            elif algo == "dqn":
                model = DQN(**common, buffer_size=config["replay_buffer"], batch_size=min(config["batch_size"], 256), learning_starts=min(1000, max(32, config["total_steps"] // 10)), tau=config["tau"], target_update_interval=max(50, config["episode_steps"] * 4))
            elif algo == "qrdqn":
                qrdqn_common = {
                    **common,
                    "policy_kwargs": {"net_arch": [64, 64], "n_quantiles": 50},
                }
                model = QRDQN(
                    **qrdqn_common,
                    buffer_size=config["replay_buffer"],
                    batch_size=min(config["batch_size"], 256),
                    learning_starts=min(1000, max(32, config["total_steps"] // 10)),
                    target_update_interval=max(50, config["episode_steps"] * 4),
                    train_freq=8,
                    gradient_steps=1,
                )
            elif algo == "trpo":
                n_steps = min(512, max(32, config["episode_steps"]))
                batch = min(config["batch_size"], n_steps)
                while n_steps % batch:
                    batch -= 1
                model = TRPO(
                    **common,
                    n_steps=n_steps,
                    batch_size=max(8, batch),
                )
            elif algo == "recurrent_ppo":
                n_steps = min(512, max(32, config["episode_steps"]))
                batch = min(config["batch_size"], n_steps)
                while n_steps % batch:
                    batch -= 1
                recurrent_common = {
                    **common,
                    "policy": "MlpLstmPolicy",
                    "policy_kwargs": {
                        "net_arch": [64, 64],
                        "lstm_hidden_size": 64,
                    },
                }
                model = RecurrentPPO(
                    **recurrent_common,
                    n_steps=n_steps,
                    batch_size=max(8, batch),
                    ent_coef=float(config.get("entropy_coef") or 0.0),
                )
            elif algo == "ars":
                model = ARS(
                    policy="MlpPolicy",
                    env=monitored,
                    learning_rate=max(0.001, config["learning_rate"] * 50),
                    n_delta=4,
                    n_top=2,
                    delta_std=0.03,
                    zero_policy=False,
                    seed=config["seed"],
                    verbose=0,
                    device="cpu",
                    policy_kwargs={"net_arch": [32, 32]},
                )
            else:
                raise ValueError(f"unknown algorithm: {algo}")
            callback = RealMetricsCallback()
            model.learn(total_timesteps=config["total_steps"], callback=callback, progress_bar=False)
            if job.cancel_event.is_set():
                raise TrainingStopped("cancelled by operator")
            model.save(str(job.run_dir / "model"))
            model_path = job.run_dir / "model.zip"
            manifest = {
                "job_id": job.job_id,
                "algorithm": algo,
                "implementation": ALGORITHMS[algo].implementation,
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.fingerprint,
                "split": split,
                "seed": config["seed"],
                "total_steps_requested": config["total_steps"],
                "total_steps_observed": int(model.num_timesteps),
                "training_device": str(model.device),
                "model_sha256": file_sha256(model_path),
                "runtime": self.capabilities().get("runtime"),
                "port_profile_id": config["port_profile_id"],
                "business_profile_id": config["business_profile_id"],
                "environment_version": config["environment_version"],
                "observation_dimensions": config["observation_dimensions"],
                "action_dimensions": config["action_dimensions"],
                "algorithm_parameters": config.get("algorithm_parameters"),
                "render_calls_during_training": env.render_calls,
                "completed_at": utc_now(),
            }
            _write_json(job.run_dir / "manifest.json", manifest)
            job.log("model artifact saved; chronological test split remains untouched")
            job.update(
                status="COMPLETED",
                progress=100.0,
                step=config["total_steps"],
                stage="training_complete_evaluation_pending",
                evaluation_available=True,
                rendering={"enabled": False, "render_calls": env.render_calls, "trace_rows": len(env.trace)},
                manifest=manifest,
            )
            self._sync_model_registry(job.job_id)
        except TrainingStopped:
            job.log("training stopped by an approved control request")
            job.update(status="CANCELLED", stage="cancelled", evaluation_available=False)
        except Exception:
            job.log("training failed; inspect the private error artifact")
            (job.run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            job.update(status="FAILED", stage="failed", error="training failed; inspect server-side diagnostics", evaluation_available=False)
        finally:
            if env is not None:
                env.close()

    def _resolve_status(self, job_id: Optional[str]) -> Dict[str, Any]:
        resolved = job_id or self.latest_job_id
        if not resolved:
            return {"job_id": None, "status": "IDLE", "progress": 0.0, "metrics": {}, "logs": [], "evaluation_available": False}
        resolved = validate_identifier(resolved, field="job_id")
        if resolved in self.jobs:
            return self.jobs[resolved].snapshot()
        path = self.run_dir(resolved) / "status.json"
        # run_dir applies strict identifier validation and root containment.
        # codeql[py/path-injection]
        if not path.exists():
            raise KeyError(resolved)
        return _read_json(path, {})

    def status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        status = json.loads(json.dumps(self._resolve_status(job_id)))
        resolved = status.get("job_id")
        if not resolved:
            return status
        algorithm = str(status.get("algorithm") or "")
        status["artifact_paths"] = {
            "run_id": str(resolved),
            "root_url": f"/api/rl/models/{resolved}",
            "model_artifact_id": (
                "model.zip"
                if algorithm in ALGORITHMS and ALGORITHMS[algorithm].trainable
                else None
            ),
            "manifest_artifact_id": "manifest.json",
        }
        manifest = status.get("manifest")
        if isinstance(manifest, dict):
            split = manifest.get("split") or {}
            quality = split.get("quality") or {}
            status["manifest"] = {
                name: manifest.get(name)
                for name in (
                    "job_id", "algorithm", "implementation", "controller_only", "dataset_id",
                    "dataset_sha256", "seed", "total_steps_requested", "total_steps_observed",
                    "port_profile_id", "environment_version", "observation_dimensions",
                    "action_dimensions", "algorithm_parameters",
                    "model_sha256", "render_calls_during_training", "completed_at",
                )
                if manifest.get(name) is not None
            }
            status["manifest"]["split"] = {
                name: split.get(name)
                for name in (
                    "dataset_id", "sha256", "rows", "train_rows",
                    "validation_rows", "test_rows", "split_method",
                    "test_ratio", "validation_ratio",
                )
                if split.get(name) is not None
            }
            status["manifest"]["split"]["quality"] = {
                name: quality.get(name)
                for name in ("status", "training_eligible", "errors", "warnings")
                if quality.get(name) is not None
            }
        evaluation = status.get("evaluation")
        if isinstance(evaluation, dict):
            render = evaluation.get("render") or {}
            status["evaluation"] = {
                name: evaluation.get(name)
                for name in (
                    "job_id", "algorithm", "implementation", "dataset_id", "dataset_sha256",
                    "port_profile_id", "environment_version", "observation_dimensions",
                    "action_dimensions",
                    "split", "episodes", "metrics", "uncertainty", "evaluation_protocol", "evaluated_at",
                )
                if evaluation.get(name) is not None
            }
            status["evaluation"]["render"] = {
                "type": render.get("type", "trajectory"),
                "frame_count": int(render.get("frame_count") or len(render.get("frames") or [])),
            }
        return status

    def history(self, job_id: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
        status = self._resolve_status(job_id)
        resolved = status.get("job_id")
        if not resolved:
            return {"job_id": None, "records": [], "count": 0, "source": "backend_optimizer_callback"}
        path = self.run_dir(str(resolved)) / "metrics.jsonl"
        records: List[Dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        bounded = records[-max(1, min(int(limit), 5000)) :]
        return {
            "job_id": resolved,
            "records": bounded,
            "count": len(bounded),
            "source": "backend_optimizer_callback",
            "rendering": False,
        }

    def control(self, job_id: str, action: str) -> Dict[str, Any]:
        job_id = validate_identifier(job_id, field="job_id")
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        action = action.lower()
        if action == "pause" and job.status["status"] == "RUNNING":
            job.pause_event.clear()
            job.log("paused by operator")
            job.update(status="PAUSED", stage="paused")
        elif action == "resume" and job.status["status"] == "PAUSED":
            job.pause_event.set()
            job.log("resumed by operator")
            job.update(status="RUNNING", stage="optimizing_policy_network")
        elif action == "cancel" and job.status["status"] in {"QUEUED", "RUNNING", "PAUSED"}:
            job.cancel_event.set()
            job.pause_event.set()
            job.log("cancellation requested")
            job.update(stage="cancelling")
        else:
            raise ValueError(f"action {action!r} is invalid for status {job.status['status']}")
        return job.snapshot()

    def _load_policy(self, config: Dict[str, Any], run_dir: Path, env: PortOperationsEnv):
        with self.policy_load_lock:
            algo = config["algorithm"]
            if algo == "mpc":
                action_dim = int(np.prod(env.action_space.shape)) if getattr(env.action_space, "shape", None) else int(config.get("action_dimensions") or 3)
                return MPCPolicy(
                    action_dim=action_dim,
                    episode_steps=env.episode_steps,
                    soc_min=env.soc_min,
                    soc_max=env.soc_max,
                    initial_soc=0.55,
                    bess_capacity_kwh=env.bess_capacity_kwh,
                    bess_power_kw=env.bess_power_kw,
                    step_hours=env.step_hours,
                )
            if algo == "fcfs":
                action_dim = int(np.prod(env.action_space.shape)) if getattr(env.action_space, "shape", None) else int(config.get("action_dimensions") or 3)
                return FCFSNeutralPolicy(action_dim=action_dim)
            from sb3_contrib import ARS, QRDQN, RecurrentPPO, TQC, TRPO
            from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

            classes = {
                "sac": SAC,
                "ppo": PPO,
                "td3": TD3,
                "dqn": DQN,
                "a2c": A2C,
                "tqc": TQC,
                "qrdqn": QRDQN,
                "trpo": TRPO,
                "recurrent_ppo": RecurrentPPO,
                "ars": ARS,
            }
            return classes[algo].load(str(run_dir / "model"), env=env, device="cpu")

    def evaluate_split_evidence(
        self,
        job_id: str,
        *,
        split_name: str = "validation",
        episodes: int = 10,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a saved policy on one explicit chronological split.

        V3 uses this method to select an algorithm on validation rows without
        reading blind-test results. It intentionally records no render trace and
        does not append validation scores to the held-out benchmark ledger.
        """
        job_id = validate_identifier(job_id, field="job_id")
        if split_name not in {"validation", "blind_test"}:
            raise ValueError("split_name must be validation or blind_test")
        run_dir = self.run_dir(job_id)
        status = self._resolve_status(job_id)
        if status.get("status") not in {"COMPLETED", "EVALUATED"}:
            raise ValueError("training must complete before split evaluation")
        config = _read_json(run_dir / "config.json", {})
        dataset = load_port_dataset(config["dataset_id"], self.data_root)
        validation_ratio = float(config.get("validation_ratio") or 0.0)
        if validation_ratio <= 0:
            raise ValueError("validation split is not configured for this run")
        train_slice, validation_slice, test_slice = dataset.split_three_way(
            config["test_ratio"], validation_ratio
        )
        selected_slice = validation_slice if split_name == "validation" else test_slice
        env = PortOperationsEnv(
            dataset,
            selected_slice,
            action_mode=ALGORITHMS[config["algorithm"]].action_space,
            episode_steps=config["episode_steps"],
            seed=config["seed"],
            demand_cap_kw=config["demand_cap_kw"],
            reward_weights=config["reward_weights"],
            projection_penalty_weight=float(config.get("projection_penalty_weight") or 0.0),
            environment_version=config.get("environment_version", "port_ops_v1"),
            port_profile=config.get("port_profile"),
            normalization_slice=train_slice,
            training=False,
            record_trace=False,
        )
        policy = self._load_policy(config, run_dir, env)
        episode_metrics: List[Dict[str, float]] = []
        requested_episodes = max(1, min(50, int(episodes)))
        max_start = max(0, len(env.segment) - config["episode_steps"] - 1)
        start_indices = np.linspace(
            0,
            max_start,
            num=min(requested_episodes, max_start + 1),
            dtype=int,
        ).tolist()
        for episode, start_index in enumerate(start_indices):
            obs, _ = env.reset(
                seed=config["seed"] + episode,
                options={"start_index": start_index},
            )
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
        metrics = {
            key: float(np.mean([item[key] for item in episode_metrics]))
            for key in episode_metrics[0]
        }
        result = {
            "job_id": job_id,
            "algorithm": config["algorithm"],
            "dataset_id": dataset.dataset_id,
            "dataset_sha256": dataset.fingerprint,
            "environment_version": config.get("environment_version", "port_ops_v1"),
            "business_profile_id": config.get("business_profile_id") or "default_port_profile",
            "split": f"chronological_{split_name}_only",
            "episodes": len(episode_metrics),
            "metrics": metrics,
            "uncertainty": summarize_metric_rows(episode_metrics, seed=config["seed"]),
            "episode_metrics": episode_metrics,
            "evaluation_protocol": {
                "deterministic_policy": True,
                "render_during_policy_execution": False,
                "window_start_indices": start_indices,
                "normalization_fit": "chronological_train_rows_only",
            },
            "evaluated_at": utc_now(),
        }
        if persist:
            _write_json(run_dir / f"{split_name}_evaluation.json", result)
        env.close()
        return result

    def evaluate(self, job_id: str, episodes: int = 10) -> Dict[str, Any]:
        if not self.evaluation_slots.acquire(blocking=False):
            raise ValueError(f"evaluation capacity reached ({self.max_concurrent_evaluation}); retry later")
        try:
            return self._evaluate_once(job_id, episodes)
        finally:
            self.evaluation_slots.release()

    def _evaluate_once(self, job_id: str, episodes: int = 10) -> Dict[str, Any]:
        job_id = validate_identifier(job_id, field="job_id")
        run_dir = self.run_dir(job_id)
        status = self._resolve_status(job_id)
        if status.get("status") not in {"COMPLETED", "EVALUATED"}:
            raise ValueError("training must complete before evaluation/rendering")
        config = _read_json(run_dir / "config.json", {})
        dataset = load_port_dataset(config["dataset_id"], self.data_root)
        env = self._make_env(dataset, config, training=False, record_trace=True)
        policy = self._load_policy(config, run_dir, env)
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
        _write_json(run_dir / "evaluation.json", {**result, "render": {"type": "trajectory", "frame_count": len(first_trace)}})
        _write_json(run_dir / "evaluation_trajectory.json", result["render"])
        self._record_benchmark(result)
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.log(f"evaluation completed on held-out rows; rendered frames={len(first_trace)}")
            job.update(
                status="EVALUATED",
                stage="evaluation_complete",
                evaluation={
                    name: result[name]
                    for name in (
                        "job_id", "algorithm", "implementation", "dataset_id", "dataset_sha256",
                        "split", "episodes", "metrics", "uncertainty", "evaluation_protocol", "evaluated_at",
                    )
                }
                | {"render": {"type": "trajectory", "frame_count": len(first_trace)}},
            )
        self._sync_model_registry(job_id)
        return result

    def predict(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run deterministic inference without advancing or rendering an environment."""
        job_id = validate_identifier(job_id, field="job_id")
        run_dir = self.run_dir(job_id)
        status = self._resolve_status(job_id)
        if status.get("status") not in {"COMPLETED", "EVALUATED"}:
            raise ValueError("training must complete before inference")
        config = _read_json(run_dir / "config.json", {})
        dataset = load_port_dataset(config["dataset_id"], self.data_root)
        env = self._make_env(dataset, config, training=False, record_trace=False)
        try:
            raw_observation = payload.get("observation")
            canonical_state: Optional[Dict[str, Any]] = None
            if raw_observation is not None:
                observation = np.asarray(raw_observation, dtype=np.float32)
                if observation.shape != env.observation_space.shape or not np.all(np.isfinite(observation)):
                    raise ValueError(f"observation must be finite with shape {env.observation_space.shape}")
                input_kind = "normalized_observation"
            else:
                canonical_state = dict(payload.get("state") or {})
                observation = env.observation_from_state(canonical_state)
                input_kind = "canonical_state"
            policy = self._load_policy(config, run_dir, env)
            raw_action, _ = policy.predict(observation, deterministic=True)
            requested = env.describe_action(raw_action)
            inference_soc = float((canonical_state or {}).get("soc", 0.55))
            inference_last_bess = float((canonical_state or {}).get("last_bess_kw", 0.0))
            decoded = env.project_control(raw_action, soc=inference_soc, last_bess_kw=inference_last_bess)
            return {
                "job_id": job_id,
                "algorithm": config["algorithm"],
                "implementation": ALGORITHMS[config["algorithm"]].implementation,
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.fingerprint,
                "port_profile_id": config.get("port_profile_id"),
                "environment_version": config.get("environment_version", "port_ops_v1"),
                "observation_dimensions": config.get("observation_dimensions"),
                "action_dimensions": config.get("action_dimensions"),
                "input_kind": input_kind,
                "deterministic": True,
                "action": np.asarray(raw_action).tolist(),
                "unconstrained_control": requested,
                "decoded_control": decoded,
                "safety_envelope": assess_recommendation(
                    state=canonical_state,
                    decoded_control=decoded,
                    dataset=dataset,
                    demand_cap_kw=config["demand_cap_kw"],
                    bess_power_kw=env.bess_power_kw,
                    port_profile=config.get("port_profile"),
                ),
                "rendered": False,
                "predicted_at": utc_now(),
            }
        finally:
            env.close()

    def _record_benchmark(self, result: Dict[str, Any]) -> None:
        registry = _read_json(self.benchmark_path, {"results": {}, "runs": []})
        config = _read_json(
            self.run_dir(result["job_id"]) / "config.json",
            {},
        )
        requested_steps = int(config.get("total_steps") or 0)
        manifest = _read_json(
            self.run_dir(result["job_id"]) / "manifest.json",
            {},
        )
        total_steps = int(
            manifest.get("total_steps_observed")
            if manifest.get("total_steps_observed") is not None
            else requested_steps
        )
        trainable = bool(ALGORITHMS[result["algorithm"]].trainable)
        run_record = {
            "algorithm": result["algorithm"],
            "dataset_id": result["dataset_id"],
            "dataset_sha256": result["dataset_sha256"],
            "metrics": result["metrics"],
            "uncertainty": result.get("uncertainty") or {},
            "episodes": result.get("episodes"),
            "job_id": result["job_id"],
            "seed": config.get("seed"),
            "total_steps": total_steps,
            "total_steps_requested": requested_steps,
            "environment_version": config.get("environment_version", "port_ops_v1"),
            "port_profile_id": config.get("port_profile_id"),
            "business_profile_id": config.get("business_profile_id") or "default_port_profile",
            "observation_dimensions": config.get("observation_dimensions"),
            "action_dimensions": config.get("action_dimensions"),
            "evidence_label": (
                "RL_SMOKE_WIRING_ONLY"
                if trainable and total_steps < 10_000
                else "RL_HELD_OUT_EVALUATION"
                if trainable
                else "DETERMINISTIC_CONTROLLER_BASELINE"
                if int(result.get("episodes") or 0) >= 10
                else "CONTROL_SMOKE_WIRING_ONLY"
            ),
            "evaluated_at": result["evaluated_at"],
        }
        runs = [item for item in registry.get("runs", []) if item.get("job_id") != result["job_id"]]
        runs.append(run_record)
        registry["runs"] = runs
        key = f"{result['dataset_id']}::{result['algorithm']}"
        registry.setdefault("results", {})[key] = run_record
        registry["updated_at"] = utc_now()
        _write_json(self.benchmark_path, registry)

    def baselines(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        registry = _read_json(self.benchmark_path, {"results": {}})
        items = []
        for spec in ALGORITHMS.values():
            matches = [value for value in registry.get("results", {}).values() if value.get("algorithm") == spec.id and (not dataset_id or value.get("dataset_id") == dataset_id)]
            latest = max(matches, key=lambda item: item.get("evaluated_at", "")) if matches else None
            items.append({**asdict(spec), "status": "EVALUATED" if latest else ("READY" if not spec.trainable else "UNTRAINED"), "latest_evaluation": latest})
        return {"baselines": items, "count": len(items), "dataset_id": dataset_id, "updated_at": registry.get("updated_at") or utc_now(), "source": "persisted_test_evaluations"}

    def benchmark_summary(
        self,
        dataset_id: Optional[str] = None,
        environment_version: Optional[str] = None,
        business_profile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        registry = _read_json(self.benchmark_path, {"results": {}, "runs": []})
        runs = list(registry.get("runs") or registry.get("results", {}).values())
        if dataset_id:
            runs = [run for run in runs if run.get("dataset_id") == dataset_id]
            if environment_version is None:
                try:
                    environment_version = str(
                        load_port_dataset(dataset_id, self.data_root).metadata.get(
                            "environment_version"
                        )
                        or ""
                    ) or None
                except Exception:
                    environment_version = None
        if environment_version:
            runs = [
                run
                for run in runs
                if run.get("environment_version") == environment_version
            ]
        if dataset_id and business_profile_id is None:
            business_profile_id = "default_port_profile"
        if business_profile_id:
            profile_filtered = []
            for run in runs:
                observed_profile = run.get("business_profile_id")
                if observed_profile is None and run.get("job_id"):
                    config = _read_json(
                        self.run_dir(str(run["job_id"])) / "config.json",
                        {},
                    )
                    observed_profile = config.get("business_profile_id")
                observed_profile = observed_profile or "default_port_profile"
                if observed_profile == business_profile_id:
                    profile_filtered.append(run)
            runs = profile_filtered
        groups: List[Dict[str, Any]] = []
        for spec in ALGORITHMS.values():
            selected = [run for run in runs if run.get("algorithm") == spec.id]
            claim_eligible = [
                run
                for run in selected
                if (
                    (
                        not spec.trainable
                        and run.get("evidence_label")
                        == "DETERMINISTIC_CONTROLLER_BASELINE"
                        and int(run.get("episodes") or 0) >= 10
                    )
                    or (
                        spec.trainable
                        and
                        int(run.get("total_steps") or 0) >= 10_000
                        and run.get("evidence_label")
                        == "RL_HELD_OUT_EVALUATION"
                    )
                )
            ]
            comparable_runs = claim_eligible if dataset_id else []
            energy_cost_values = sorted(
                float(run["metrics"]["energy_cost"])
                for run in comparable_runs
                if isinstance((run.get("metrics") or {}).get("energy_cost"), (int, float))
            )
            if energy_cost_values:
                tail_start = max(0, math.ceil(0.95 * len(energy_cost_values)) - 1)
                energy_cost_tail = energy_cost_values[tail_start:]
                energy_cost_cvar95 = sum(energy_cost_tail) / len(energy_cost_tail)
            else:
                energy_cost_cvar95 = None
            formal_metric_names = sorted(
                {
                    name
                    for run in comparable_runs
                    for name in (run.get("metrics") or {})
                }
            )
            formal_summaries = {
                name: bootstrap_summary(
                    [
                        float(run["metrics"][name])
                        for run in comparable_runs
                        if isinstance((run.get("metrics") or {}).get(name), (int, float))
                    ],
                    seed=20260720,
                )
                for name in formal_metric_names
            }
            diagnostic_runs = [
                run
                for run in selected
                if run.get("evidence_label") == "RL_SMOKE_WIRING_ONLY"
            ]
            diagnostic_metric_names = sorted(
                {
                    name
                    for run in diagnostic_runs
                    for name in (run.get("metrics") or {})
                }
            )
            diagnostic_summaries = {
                name: bootstrap_summary(
                    [
                        float(run["metrics"][name])
                        for run in diagnostic_runs
                        if isinstance((run.get("metrics") or {}).get(name), (int, float))
                    ],
                    seed=20260720,
                )
                for name in diagnostic_metric_names
            }
            seeds = sorted(
                {
                    int(run["seed"])
                    for run in comparable_runs
                    if isinstance(run.get("seed"), int)
                }
            )
            groups.append({
                **asdict(spec),
                "runs": len(selected),
                "claim_eligible_runs": len(comparable_runs),
                "distinct_seeds": seeds,
                "multi_seed_ready": (
                    len(seeds) >= 3
                    if spec.trainable
                    else len(comparable_runs) >= 1
                ),
                "smoke_runs": sum(
                    run.get("evidence_label")
                    in {"RL_SMOKE_WIRING_ONLY", "CONTROL_SMOKE_WIRING_ONLY"}
                    for run in selected
                ),
                "metrics": formal_summaries,
                "diagnostic_metrics": diagnostic_summaries,
                "metric_evidence_scope": (
                    "formal_claim_eligible_runs_only"
                    if dataset_id
                    else "dataset_filter_required_for_comparison"
                ),
                "tail_risk": {
                    "metric": "energy_cost",
                    "alpha": 0.95,
                    "cvar": energy_cost_cvar95,
                    "n": len(energy_cost_values),
                    "basis": "claim_eligible_run_means_on_fixed_chronological_holdout",
                },
                "job_ids": [run.get("job_id") for run in selected],
            })
        return {
            "dataset_id": dataset_id,
            "environment_version": environment_version,
            "business_profile_id": business_profile_id,
            "algorithms": groups,
            "minimum_distinct_seeds_for_claim": 3,
            "comparison_requires_dataset_filter": True,
            "comparison_requires_dataset_environment_and_business_profile_filter": True,
            "deterministic_controller_exception": "MPC requires one fixed-window run because it has no learned stochastic initialization",
            "source": "persisted_chronological_holdout_evaluations",
            "updated_at": registry.get("updated_at") or utc_now(),
        }

    def model_registry(self) -> ModelRegistry:
        registry_path = self.benchmark_path.with_name(MODEL_REGISTRY_PATH.name)
        return ModelRegistry(self.run_root, registry_path)

    def _sync_model_registry(self, job_id: str) -> None:
        try:
            self.model_registry().sync(job_id)
        except Exception:
            job = self.jobs.get(job_id)
            if job is not None:
                job.log("model registry sync failed without invalidating run; inspect server logs")
                job.persist()


TRAINING_MANAGER = TrainingManager()
