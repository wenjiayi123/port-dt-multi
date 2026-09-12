"""Train fresh Shore+BESS neural policies with frozen, month-separated evaluation.

Pilot runs evaluate May--July 2025 validation only. Formal runs freeze all
checkpoint/seed choices before opening the previously used Aug--Dec 2025 and
Jan--May 2026 benchmarks. No teacher actions or historical model weights enter
training. All run artifacts are append-only; V3/V7 pointers are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.services.rl_model.shore_bess.v3_environment import load_config
from app.services.rl_training.datasets import (
    FACTOR_COLUMNS, PortDataset, dataset_quality_report, file_sha256, load_port_dataset,
)
from app.services.rl_training.statistics import bootstrap_summary

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/shore_bess_v32_balanced.json"
OUTPUT_ROOT = ROOT / "evidence/v8/shore_bess"
EPISODE_HOURS = 168
GATE = {
    "cost_reduction_percent_min": 0.03,
    "carbon_reduction_percent_min": 0.001,
    "peak_reduction_percent_min": 0.0,
    "all_three_95ci_lower_bounds_strictly_positive": True,
    "formal_source_period_cluster_95ci_lower_bounds_strictly_positive": True,
    "formal_source_period_cluster_count_min": 3,
    "every_window_carbon_non_regression": True,
    "terminal_absolute_tolerance": 1e-6,
    "safety_tolerance": 1e-12,
    "tail_checkpoints": 3,
    "tail_relative_range_max": 0.25,
    "tail_range_floors_pp": {"total_cost_cny": 0.02, "carbon_kg": 0.001, "peak_kw": 0.05},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative(path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def hashes_match(expected: dict[str, str]) -> bool:
    return all((ROOT / path).is_file() and file_sha256(ROOT / path) == digest for path, digest in expected.items())


def input_hashes(dataset: PortDataset) -> dict[str, str]:
    paths = [dataset.path, dataset.path.with_suffix(".meta.json")]
    return {relative(path): file_sha256(path) for path in paths if path.is_file()}


def official_source_period(timestamp: str) -> str:
    """The pinned MOT source combines January and February into one anchor."""
    return timestamp[:4] + "-01/02" if timestamp[5:7] in {"01", "02"} else timestamp[:7]


def month_slice(dataset: PortDataset, first: str, stop: str) -> slice:
    """Use complete UTC calendar months, without splitting a source month."""
    timestamps = np.asarray(dataset.timestamps, dtype=str)
    matches = np.flatnonzero((timestamps >= first) & (timestamps < stop))
    if len(matches) < EPISODE_HOURS + 1 or np.any(np.diff(matches) != 1):
        raise ValueError(f"missing or non-contiguous monthly interval {first} .. {stop}")
    expected_hours = int((datetime.fromisoformat(stop) - datetime.fromisoformat(first)).total_seconds() / 3600)
    if len(matches) != expected_hours:
        raise ValueError(f"monthly interval {first} .. {stop} must contain {expected_hours} hourly rows")
    return slice(int(matches[0]), int(matches[-1]) + 1)


def split_description(dataset: PortDataset, split: slice) -> dict[str, Any]:
    return {"start_row": split.start, "stop_row_exclusive": split.stop,
            "rows": split.stop - split.start, "first_timestamp": dataset.timestamps[split.start],
            "last_timestamp": dataset.timestamps[split.stop - 1]}


def window_starts(length: int, count: int | None = None) -> list[int]:
    # The inherited simulator requires one next-observation context row beyond
    # each evaluated 168-hour window. No next-row reward enters the evaluation.
    starts = list(range(0, length - EPISODE_HOURS, EPISODE_HOURS))
    if not starts:
        raise ValueError("evaluation segment lacks a full episode and context row")
    if count is not None and len(starts) > count:
        indices = np.linspace(0, len(starts) - 1, count, dtype=int)
        starts = [starts[int(index)] for index in indices]
    return starts


def checked_quality(dataset: PortDataset) -> dict[str, Any]:
    if file_sha256(dataset.path) != dataset.fingerprint:
        raise ValueError("dataset metadata fingerprint differs from the actual CSV")
    report = dataset_quality_report(dataset)
    if not report["training_eligible"]:
        raise ValueError(f"dataset quality gate failed: {dataset.dataset_id}")
    if report["time"]["median_cadence_seconds"] != 3600 or report["time"]["irregular_gap_count"]:
        raise ValueError("Shore+BESS requires complete hourly chronology")
    coverage = {}
    for name in ("equipment_availability_ratio", "berth_occupancy_ratio"):
        index = FACTOR_COLUMNS.index(name)
        mask = dataset.factor_availability[:, index] > 0.5
        coverage[name] = float(np.mean(mask))
        if not np.all(mask) or not np.isfinite(dataset.factor_values[:, index]).all():
            raise ValueError(f"essential Shore+BESS factor missing: {name}")
    report["essential_factor_coverage"] = coverage
    return report


def joined_forward(history: PortDataset, forward: PortDataset) -> PortDataset:
    """Carry historical train rows for normalization; evaluate only forward rows."""
    if history.timestamps[-1] >= forward.timestamps[0]:
        raise ValueError("forward rows overlap the historical benchmark")
    return PortDataset(
        dataset_id="shore_bess_v8_history_plus_forward_evaluation_only",
        path=forward.path, timestamps=[*history.timestamps, *forward.timestamps],
        values=np.vstack((history.values, forward.values)),
        metadata={"sha256": "evaluation_composite", "historical_sha256": history.fingerprint,
                  "forward_sha256": forward.fingerprint},
        factor_values=np.vstack((history.factor_values, forward.factor_values)),
        factor_availability=np.vstack((history.factor_availability, forward.factor_availability)),
    )


def evaluate(factory: Callable, model: Any, starts: list[int], idle_action: Any) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    masked_policy = isinstance(model, MaskablePPO)
    rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    for start in starts:
        env = factory()
        try:
            obs, reset_info = env.reset(options={"start_index": start})
            if reset_info["start_index"] != start:
                raise ValueError("simulator silently changed a fixed evaluation start")
            accumulated_reward = 0.0
            while True:
                if model is None:
                    action = idle_action
                else:
                    options = {"action_masks": env.action_masks()} if masked_policy else {}
                    predicted = np.asarray(model.predict(obs, deterministic=True, **options)[0])
                    action = int(predicted.item()) if hasattr(env.action_space, "n") else predicted.astype(np.float32)
                action_key = str(action) if hasattr(env.action_space, "n") else "continuous_action"
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
                obs, reward, terminated, truncated, _info = env.step(action)
                accumulated_reward += float(reward)
                if terminated or truncated:
                    break
            totals = {key: float(value) for key, value in env.totals.items()
                      if isinstance(value, (float, int, np.number))}
            if not all(np.isfinite(value) for value in totals.values()):
                raise ValueError("non-finite evaluation metric")
            totals["learning_reward"] = accumulated_reward
            rows.append(totals)
            first_timestamp = str(env.timestamps[start])
            windows.append({"start_index": start, "first_timestamp": first_timestamp,
                            "last_timestamp": str(env.timestamps[start + EPISODE_HOURS - 1]),
                            "source_month": first_timestamp[:7],
                            "source_period": official_source_period(first_timestamp)})
        finally:
            env.close()
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {"mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
            "rows": rows, "starts": starts, "episode_hours": EPISODE_HOURS,
            "windows": windows, "window_overlap": False, "action_counts": action_counts}


def source_period_summary(values: list[float], groups: list[str]) -> dict[str, Any]:
    """Resample official source periods, retaining every within-period window.

    Weekly windows share engineered monthly throughput anchors. The ordinary
    window bootstrap is retained for comparison; this additional cluster CI
    combines January/February because their official anchor is shared. Windows
    crossing an official period are assigned to their start period; that
    approximation is stated explicitly rather than calling them independent.
    """
    array = np.asarray(values, dtype=np.float64)
    if len(array) != len(groups) or not len(array) or not np.isfinite(array).all():
        raise ValueError("finite paired values and official source-period labels are required")
    months = sorted(set(groups))
    group_array = np.asarray(groups)
    sums = np.asarray([np.sum(array[group_array == month]) for month in months])
    counts = np.asarray([np.sum(group_array == month) for month in months])
    indices = np.random.default_rng(20260912).integers(0, len(months), size=(5000, len(months)))
    means = np.sum(sums[indices], axis=1) / np.sum(counts[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"clustered_ci_low": float(low), "clustered_ci_high": float(high),
            "cluster_count": len(months), "cluster_confidence": 0.95,
            "cluster_method": "percentile_bootstrap_of_official_source_period_groups",
            "cluster_group_field": "source_period",
            "cluster_resamples": 5000, "cluster_window_counts": dict(zip(months, map(int, counts), strict=True)),
            "cluster_boundary": "Official MOT period of window start; January/February share YYYY-01/02. Cross-period windows belong to their starting period, an approximation. Few-cluster engineering uncertainty, not field confidence."}


def compare(candidate: dict, reference: dict) -> dict[str, Any]:
    if candidate["starts"] != reference["starts"]:
        raise ValueError("paired evaluation windows do not match")
    if candidate.get("windows") != reference.get("windows"):
        raise ValueError("paired evaluation timestamps do not match")
    groups = [window["source_period"] for window in candidate["windows"]]
    result = {}
    for metric in ("total_cost_cny", "carbon_kg", "peak_kw"):
        gains = [100.0 * (base[metric] - row[metric]) / max(abs(base[metric]), 1e-9)
                 for row, base in zip(candidate["rows"], reference["rows"], strict=True)]
        summary = bootstrap_summary(gains, seed=20260912, resamples=5000)
        summary.update({"per_window_percent": gains, "minimum_percent": float(min(gains))})
        summary.update(source_period_summary(gains, groups))
        result[metric] = summary
    return result


def business_gates(evaluation: dict, comparison: dict, *, require_month_ci: bool = False) -> dict[str, bool]:
    checks = {}
    for metric, threshold in (("total_cost_cny", "cost_reduction_percent_min"),
                              ("carbon_kg", "carbon_reduction_percent_min"),
                              ("peak_kw", "peak_reduction_percent_min")):
        checks[metric + "_minimum_mean_gain"] = comparison[metric]["mean"] >= GATE[threshold]
        checks[metric + "_positive_95ci"] = comparison[metric]["ci_low"] > 0.0
        if require_month_ci:
            checks[metric + "_positive_source_period_95ci"] = comparison[metric]["clustered_ci_low"] > 0.0
            checks[metric + "_source_period_count"] = comparison[metric]["cluster_count"] >= GATE["formal_source_period_cluster_count_min"]
    checks["every_window_carbon_non_regression"] = comparison["carbon_kg"]["minimum_percent"] >= 0.0
    required = ("guardrail_violation_rate", "terminal_soc_error", "terminal_flex_backlog_kwh",
                "shore_sla_violation_kwh", "reserve_shortfall_kwh", "flex_deadline_violation_kwh",
                "physical_power_violations")
    optional = ("terminal_soc_absolute_error", "terminal_energy_debt_kwh")
    for metric in required + optional:
        if metric in optional and metric not in evaluation["mean"]:
            continue
        tolerance = GATE["terminal_absolute_tolerance"] if metric.startswith("terminal") else GATE["safety_tolerance"]
        checks[metric] = all(metric in row and abs(row[metric]) <= tolerance for row in evaluation["rows"])
    checks["maximum_flex_age_within_12_hours"] = all(
        "max_flex_age_hours" in row and row["max_flex_age_hours"] <= 12 for row in evaluation["rows"])
    return checks


def convergence(curve: list[dict]) -> dict[str, Any]:
    tail = curve[-3:]
    stability = {}
    for metric, floor in GATE["tail_range_floors_pp"].items():
        values = [row["comparison"][metric]["mean"] for row in tail]
        spread = float(np.ptp(values)) if values else None
        tolerance = max(floor, GATE["tail_relative_range_max"] * abs(float(np.mean(values)))) if values else floor
        stability[metric] = {"range_pp": spread, "tolerance_pp": tolerance,
                             "passed": len(tail) == 3 and spread <= tolerance}
    business_pass = len(tail) == 3 and all(all(row["gates"].values()) for row in tail)
    return {"passed": business_pass and all(item["passed"] for item in stability.values()),
            "tail_checkpoints": len(tail), "all_tail_business_gates_passed": business_pass,
            "stability": stability}


def model_optimizers(model: Any) -> dict[str, Any]:
    if hasattr(model.policy, "optimizer"):
        return {"policy": model.policy.optimizer}
    result = {"actor": model.actor.optimizer, "critic": model.critic.optimizer}
    if getattr(model, "ent_coef_optimizer", None) is not None:
        result["entropy"] = model.ent_coef_optimizer
    return result


def model_parameters(model: Any) -> dict[str, Any]:
    fields = ("num_timesteps", "_n_updates", "gamma", "learning_starts", "buffer_size", "batch_size",
              "gradient_steps", "target_update_interval", "exploration_fraction", "exploration_final_eps",
              "exploration_rate", "n_steps", "n_epochs", "gae_lambda", "ent_coef", "target_kl", "max_grad_norm",
              "tau", "policy_delay", "target_policy_noise", "target_noise_clip")
    result = {key: getattr(model, key) for key in fields if hasattr(model, key)}
    result["learning_rate"] = float(model.lr_schedule(1.0))
    result["optimizer_learning_rates"] = {
        name: [float(group["lr"]) for group in optimizer.param_groups]
        for name, optimizer in model_optimizers(model).items()}
    if hasattr(model, "train_freq"):
        result["train_freq"] = {"frequency": model.train_freq.frequency, "unit": model.train_freq.unit.value}
    if hasattr(model, "clip_range"):
        result["clip_range"] = float(model.clip_range(1.0))
    result["policy_kwargs"] = {"net_arch": [128, 128]}
    result["policy_class"] = type(model.policy).__name__
    result["network_structure"] = str(model.policy)
    result["parameter_count"] = sum(parameter.numel() for parameter in model.policy.parameters())
    result["trainable_parameter_count"] = sum(parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad)
    result["gradient_optimizer_step_calls"] = int(getattr(model, "_v8_optimizer_step_calls", 0))
    result["optimizer_step_calls_by_component"] = dict(getattr(model, "_v8_optimizer_steps_by_component", {}))
    result["sb3_update_counter_semantics"] = (
        "SB3 gradient iterations; actor/critic/entropy optimizer calls are counted separately" if type(model).__name__ in {"DQN", "SAC", "TD3"}
        else "SB3 PPO epoch counter; actual optimizer calls are counted separately by a PyTorch step hook")
    return result


def weights_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def rank(row: dict) -> tuple:
    failed = sum(not passed for passed in row["gates"].values())
    return failed, -row["comparison"]["carbon_kg"]["ci_low"], -row["comparison"]["total_cost_cny"]["ci_low"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--block", type=int, default=20000)
    parser.add_argument("--seeds", default="1212,1312,1412")
    parser.add_argument("--carbon-price", type=float, default=12.0)
    parser.add_argument("--algorithm", choices=("dqn", "ppo", "maskppo", "sac", "td3"), default="dqn")
    parser.add_argument("--run-id")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",")]
    except ValueError:
        parser.error("seeds must be comma-separated integers")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if args.steps <= 0 or args.block <= 0 or (not args.pilot and args.steps < 3 * args.block):
        parser.error("positive budgets required; formal runs require at least three checkpoints")
    if not args.pilot and len(seeds) < 3:
        parser.error("formal admission requires at least three independent seeds")
    if not np.isfinite(args.carbon_price) or args.carbon_price < 0:
        parser.error("carbon-price must be finite and nonnegative")
    if args.run_id and (Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}):
        parser.error("run-id must be a single directory name")

    import torch
    import stable_baselines3 as sb3
    import sb3_contrib
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from app.services.rl_model.shore_bess import v8_environment as environment

    torch.set_num_threads(1)
    cfg = load_config(CONFIG_PATH)
    dataset = load_port_dataset(cfg["dataset_id"])
    historical_quality = checked_quality(dataset)
    protected_inputs = {relative(CONFIG_PATH): file_sha256(CONFIG_PATH), **input_hashes(dataset)}
    train = month_slice(dataset, "2024-01-01", "2025-05-01")
    validation = month_slice(dataset, "2025-05-01", "2025-08-01")
    starts = window_starts(validation.stop - validation.start, 6 if args.pilot else None)
    lattice = np.asarray(environment.LATTICE)
    continuous = args.algorithm in {"sac", "td3"}
    idle_indices = np.flatnonzero(np.all(lattice == 0.0, axis=1))
    if len(idle_indices) != 1:
        raise ValueError("action lattice requires exactly one exact idle command")
    idle_action = np.zeros(2, dtype=np.float32) if continuous else int(idle_indices[0])

    def factory(source, split, seed=0, training=False):
        env_cls = environment.ShoreBESSV8SACEnv if continuous else environment.ShoreBESSV8Env
        return lambda: env_cls(source, split, config=cfg, normalization_slice=train,
                   episode_steps=EPISODE_HOURS, carbon_price=args.carbon_price, discrete=not continuous,
                   seed=seed, training=training, record_trace=False)

    run_id = args.run_id or f"shore-bess-v8-{args.algorithm}-{'pilot' if args.pilot else 'formal'}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    protected_files = sorted({*list((ROOT / "evidence/v3/shore_bess").glob("*.json")),
                              *list((ROOT / "evidence/v7/shore_bess").glob("*.json"))})
    protected = {relative(path): file_sha256(path) for path in protected_files}
    source_paths = [Path(__file__), Path(environment.__file__),
                    ROOT / "app/services/rl_model/shore_bess/v3_environment.py", CONFIG_PATH,
                    ROOT / "app/services/rl_training/statistics.py", ROOT / "app/services/rl_training/datasets.py"]
    source_hashes = {}
    for path in source_paths:
        target = run_dir / "source" / relative(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[relative(path)] = file_sha256(path)
    probe = factory(dataset, train)()
    scaler_names = ("base_scale", "price_scale", "carbon_scale", "soft_cap_kw", "price_center",
                    "carbon_center", "carbon_span", "load_center", "load_std", "reward_scale")
    scaler = {key: float(getattr(probe, key)) for key in scaler_names if hasattr(probe, key)}
    observation_dimensions = int(np.prod(probe.observation_space.shape))
    action_count = None if continuous else int(probe.action_space.n)
    action_shape = list(probe.action_space.shape)
    mapping_metadata = getattr(probe, "action_mapping_metadata", {
        "kind": "state_dependent_physical_parameterization" if continuous else "fixed_discrete_dispatch_lattice",
        "class": type(probe).__name__, "source_sha256": source_hashes[relative(Path(environment.__file__))],
        "exact_idle_available": True,
    })
    if continuous:
        mapping_metadata.update(version=probe.action_mapping_version, deadband=probe.action_deadband,
                                inputs="current equipment, SOC, temperature, ramp, PCC, reserve, terminal reachability and FIFO deadlines only",
                                economic_teacher_used=False)
    reward_credit_assignment = getattr(probe, "reward_credit_assignment", "see hashed environment source")
    probe.close()
    algorithm_config = ({"learning_rate": 0.0001, "buffer_size": 200000, "batch_size": 256,
                         "train_freq": 4, "gradient_steps": 1, "learning_starts": 2000,
                         "target_update_interval": 1000, "exploration_fraction": 0.5,
                         "exploration_final_eps": 0.02} if args.algorithm == "dqn" else
                        {"learning_rate": 0.00015, "n_steps": 1344, "batch_size": 168,
                         "n_epochs": 10, "gae_lambda": 0.99, "ent_coef": 0.0001,
                         "clip_range": 0.15, "target_kl": 0.02, "max_grad_norm": 0.5})
    if continuous:
        algorithm_config = {"learning_rate": 0.0003, "buffer_size": 200000, "batch_size": 256,
                            "learning_starts": 2000, "train_freq": 1, "gradient_steps": 1, "tau": 0.005}
        if args.algorithm == "sac":
            algorithm_config["ent_coef"] = 0.001
        else:
            algorithm_config.update(policy_delay=2, target_policy_noise=0.1, target_noise_clip=0.3)
    algorithm_name = "sb3_contrib.MaskablePPO" if args.algorithm == "maskppo" else "stable_baselines3." + args.algorithm.upper()
    run_config = {"algorithm": algorithm_name, "seeds": seeds,
                  "requested_steps_per_seed": args.steps, "checkpoint_interval": args.block,
                  "episode_hours": EPISODE_HOURS, "gamma": 1.0, "network": [128, 128],
                  "carbon_price_cny_per_kg_constraint_multiplier": args.carbon_price,
                  "algorithm_parameters": algorithm_config, "discrete": not continuous,
                  "action_lattice": None if continuous else lattice.tolist(), "state_names": list(environment.STATE_NAMES),
                  "observation_dimensions": observation_dimensions, "action_count": action_count,
                  "action_shape": action_shape, "action_mapping": mapping_metadata,
                  "action_noise": {"class": "NormalActionNoise", "mean": [0.0, 0.0], "sigma": [0.1, 0.1]} if args.algorithm == "td3" else None,
                  "physical_config": cfg, "normalization": scaler, "admission_gate": GATE,
                  "reward_credit_assignment": reward_credit_assignment,
                  "physical_config_legacy_fields": {
                      "training_algorithm_and_optimizer_settings_used": False,
                      "legacy_reward_weights_used_for_v8_objective": False,
                      "episode_hours_overridden_by_runner": EPISODE_HOURS,
                      "actual_optimizer_configuration": "top-level algorithm_parameters and recorded per-checkpoint parameters",
                  },
                  "pilot": args.pilot, "physical_action_masking": args.algorithm == "maskppo",
                  "optimizer_update_count_basis": "actual PyTorch optimizer step post-hook calls"}
    write_json(run_dir / "config.json", run_config)
    manifest = {"schema": "port-shore-bess-v8-manifest.v1", "run_id": run_id, "started_at": utc_now(),
                "dataset_id": dataset.dataset_id, "dataset_sha256": dataset.fingerprint,
                "dataset_quality": historical_quality,
                "input_files_sha256": dict(protected_inputs),
                "train": split_description(dataset, train), "validation": split_description(dataset, validation),
                "validation_starts": starts, "normalization_fit_split": "train_only",
                "selection_protocol": "validation_only_then_frozen_checkpoint_and_seed",
                "test_status": "not_opened" if args.pilot else "sealed_until_selection_previously_used_benchmark",
                "forward_status": "not_loaded" if args.pilot else "sealed_until_selection_previously_used_benchmark",
                "teacher_actions_used": False, "warm_start_used": False,
                "source_sha256": source_hashes, "historical_pointer_sha256": protected,
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "stable_baselines3": sb3.__version__, "sb3_contrib": sb3_contrib.__version__},
                "simulation_mode": True, "production_authority": False, "dispatch_allowed": False,
                "live_data_verified": False}
    write_json(run_dir / "manifest.json", manifest)
    reference = evaluate(factory(dataset, validation), None, starts, idle_action)
    write_json(run_dir / "validation_reference.json", reference)
    algorithm_cls = {"dqn": sb3.DQN, "ppo": sb3.PPO, "maskppo": MaskablePPO, "sac": sb3.SAC, "td3": sb3.TD3}[args.algorithm]
    results = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        env = Monitor(factory(dataset, train, seed, True)(), str(seed_dir / "monitor.csv"))
        construction_kwargs = dict(algorithm_config)
        if args.algorithm == "td3":
            construction_kwargs["action_noise"] = NormalActionNoise(np.zeros(2), np.full(2, 0.1))
        model = algorithm_cls("MlpPolicy", env, gamma=1.0, policy_kwargs={"net_arch": [128, 128]},
                              seed=seed, device="cpu", verbose=0, **construction_kwargs)
        model._v8_optimizer_step_calls = 0
        model._v8_optimizer_steps_by_component = {name: 0 for name in model_optimizers(model)}
        optimizer_hooks = []
        for optimizer_name, optimizer in model_optimizers(model).items():
            def count_optimizer_step(_optimizer, _args, _kwargs, component=optimizer_name):
                model._v8_optimizer_step_calls += 1
                model._v8_optimizer_steps_by_component[component] += 1
            optimizer_hooks.append(optimizer.register_step_post_hook(count_optimizer_step))
        initial_weights = weights_sha256(model)
        initial_evaluation = evaluate(factory(dataset, validation), model, starts, idle_action)
        write_json(seed_dir / "initial_validation.json", {"evaluation": initial_evaluation,
                   "comparison": compare(initial_evaluation, reference), "weights_sha256": initial_weights,
                   "parameters": model_parameters(model)})
        curve: list[dict] = []

        def checkpoint() -> None:
            path = seed_dir / f"step_{int(model.num_timesteps)}.zip"
            if path.exists():
                raise FileExistsError("immutable checkpoint already exists")
            model.save(path)
            ev = evaluate(factory(dataset, validation), model, starts, idle_action)
            comparison = compare(ev, reference)
            checks = business_gates(ev, comparison, require_month_ci=not args.pilot)
            row = {"step": int(model.num_timesteps), "optimizer_updates": int(model._v8_optimizer_step_calls),
                   "sb3_update_counter": int(model._n_updates),
                   "model_path": relative(path), "model_sha256": file_sha256(path),
                   "weights_sha256": weights_sha256(model), "parameters": model_parameters(model),
                   "evaluation": ev, "comparison": comparison, "gates": checks,
                   "optimizer": {str(key): float(value) for key, value in model.logger.name_to_value.items()
                                 if key.startswith("train/") and np.isscalar(value) and np.isfinite(value)}}
            curve.append(row)
            write_json(seed_dir / "curve.json", curve)
            print(json.dumps({"run_id": run_id, "seed": seed, "step": row["step"], "updates": row["optimizer_updates"],
                              "validation_gain_percent": {key: round(value["mean"], 6) for key, value in comparison.items()},
                              "failed": [key for key, passed in checks.items() if not passed],
                              "stable": convergence(curve)["passed"]}, ensure_ascii=False, allow_nan=False), flush=True)

        class Checkpoints(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_step = args.block

            def _on_step(self):
                # Never abort the final rollout before the optimizer uses it.
                return True

            def _on_rollout_start(self):
                # Called after the preceding rollout's optimizer update. One
                # learn() invocation keeps the DQN exploration schedule global.
                if self.num_timesteps >= self.next_step and self.num_timesteps < args.steps:
                    checkpoint()
                    self.next_step = (self.num_timesteps // args.block + 1) * args.block

            def _on_training_end(self):
                checkpoint()

        try:
            model.learn(total_timesteps=args.steps, callback=Checkpoints(), progress_bar=False)
            chosen = min(curve, key=rank)
            selected_path = seed_dir / "selected_model.zip"
            shutil.copy2(ROOT / chosen["model_path"], selected_path)
            result = {"seed": seed, "steps": int(model.num_timesteps), "optimizer_updates": int(model._v8_optimizer_step_calls),
                      "sb3_update_counter": int(model._n_updates),
                      "initial_weights_sha256": initial_weights, "final_weights_sha256": weights_sha256(model),
                      "parameters": model_parameters(model), "selected": chosen,
                      "model_path": relative(selected_path), "model_sha256": file_sha256(selected_path),
                      "convergence": convergence(curve), "render_calls": env.unwrapped.render_calls}
            write_json(seed_dir / "result.json", result)
            results.append(result)
        finally:
            for hook in optimizer_hooks:
                hook.remove()
            env.close()

    selected = min(results, key=lambda row: (not row["convergence"]["passed"], *rank(row["selected"])))
    selection = {"selected_seed": selected["seed"], "model_path": selected["model_path"],
                 "model_sha256": selected["model_sha256"], "selection_access_to_test": False,
                 "selection_access_to_forward": False, "frozen_at": utc_now(),
                 "per_seed_model_sha256": {str(row["seed"]): row["model_sha256"] for row in results}}
    write_json(run_dir / "selection.json", selection)
    evaluations: dict[str, Any] = {}
    if not args.pilot:
        test = month_slice(dataset, "2025-08-01", "2026-01-01")
        forward = load_port_dataset("public_cn_sha_forward_2026m05_v1")
        forward_quality = checked_quality(forward)
        forward_input_hashes = input_hashes(forward)
        protected_inputs.update(forward_input_hashes)
        write_json(run_dir / "forward_input_manifest.json", {"dataset_id": forward.dataset_id,
                   "input_files_sha256": forward_input_hashes, "opened_after_selection_sha256": file_sha256(run_dir / "selection.json")})
        combined = joined_forward(dataset, forward)
        forward_split = slice(dataset.rows, dataset.rows + forward.rows)
        for label, source, split in (("test", dataset, test), ("forward", combined, forward_split)):
            test_starts = window_starts(split.stop - split.start)
            baseline = evaluate(factory(source, split), None, test_starts, idle_action)
            partition = {"status": "previously_used_chronological_benchmark_not_fresh_blind_data",
                         "dataset_id": dataset.dataset_id if label == "test" else forward.dataset_id,
                         "dataset_sha256": dataset.fingerprint if label == "test" else forward.fingerprint,
                         "dataset_quality": historical_quality if label == "test" else forward_quality,
                         "split": split_description(source, split), "reference": baseline, "per_seed": []}
            for result in results:
                if file_sha256(ROOT / result["model_path"]) != result["model_sha256"]:
                    raise ValueError("selected model changed after selection freeze")
                model = algorithm_cls.load(ROOT / result["model_path"], device="cpu")
                ev = evaluate(factory(source, split), model, test_starts, idle_action)
                comparison = compare(ev, baseline)
                partition["per_seed"].append({"seed": result["seed"], "evaluation": ev,
                                              "comparison": comparison, "gates": business_gates(ev, comparison, require_month_ci=True)})
            evaluations[label] = partition
            write_json(run_dir / f"{label}_evaluation.json", partition)

    checks = {"formal_run": not args.pilot, "three_independent_seeds": len(seeds) >= 3,
              "all_seeds_validation_passed": all(all(row["selected"]["gates"].values()) for row in results),
              "all_seeds_last_three_stable": all(row["convergence"]["passed"] for row in results),
              "real_optimizer_updates": all(row["optimizer_updates"] > 0 for row in results),
              "weights_changed": all(row["initial_weights_sha256"] != row["final_weights_sha256"] for row in results),
              "source_unchanged_during_training": hashes_match(source_hashes),
              "config_and_dataset_files_unchanged": hashes_match(protected_inputs),
              "historical_pointers_preserved": hashes_match(protected),
              "no_training_rendering": all(row["render_calls"] == 0 for row in results),
              "test_all_seed_gates": "test" in evaluations and all(all(row["gates"].values()) for row in evaluations["test"]["per_seed"]),
              "forward_all_seed_gates": "forward" in evaluations and all(all(row["gates"].values()) for row in evaluations["forward"]["per_seed"])}
    admitted = all(checks.values())
    status = "PILOT_VALIDATION_ONLY" if args.pilot else ("ADMITTED_OFFLINE_RL" if admitted else "CANDIDATE_NOT_ADMITTED")
    report = {"schema": "port-shore-bess-v8-report.v1", "run_id": run_id, "generated_at": utc_now(),
              "status": status, "config": run_config, "manifest": manifest, "checks": checks,
              "results": results, "selected_seed": selected["seed"], "selection": selection,
              "evaluations": evaluations, "total_environment_steps": sum(row["steps"] for row in results),
              "input_files_sha256": protected_inputs,
              "total_optimizer_updates": sum(row["optimizer_updates"] for row in results),
              "promoted": admitted, "simulation_mode": True, "production_authority": False,
              "live_data_verified": False, "dispatch_allowed": False,
              "claim_boundary": "Public/engineering offline scenario. Repeated historical and forward benchmarks; no fresh blind test and no measured port savings."}
    report["evidence_files_sha256"] = {
        relative(path): file_sha256(path) for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".csv", ".zip"} and "source" not in path.relative_to(run_dir).parts
    }
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    pointer = {"schema": "port-shore-bess-v8-pointer.v1", "run_id": run_id, "status": status,
               "report_path": relative(report_path), "report_sha256": file_sha256(report_path),
               "model_path": selected["model_path"], "model_sha256": selected["model_sha256"],
               "production_authority": False, "updated_at": utc_now()}
    if not args.pilot:
        write_json(OUTPUT_ROOT / "latest.json", pointer)
        if admitted:
            write_json(OUTPUT_ROOT / "offline_champion.json", pointer)
    print(json.dumps(pointer, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
