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
from app.services.rl_training.datasets import PortDataset, file_sha256, load_port_dataset
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


def month_slice(dataset: PortDataset, first: str, stop: str) -> slice:
    """Use complete UTC calendar months, without splitting a source month."""
    timestamps = np.asarray(dataset.timestamps, dtype=str)
    matches = np.flatnonzero((timestamps >= first) & (timestamps < stop))
    if len(matches) < EPISODE_HOURS + 1 or np.any(np.diff(matches) != 1):
        raise ValueError(f"missing or non-contiguous monthly interval {first} .. {stop}")
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


def evaluate(factory: Callable, model: Any, starts: list[int], idle_action: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    for start in starts:
        env = factory()
        try:
            obs, reset_info = env.reset(options={"start_index": start})
            if reset_info["start_index"] != start:
                raise ValueError("simulator silently changed a fixed evaluation start")
            accumulated_reward = 0.0
            while True:
                action = idle_action if model is None else int(np.asarray(model.predict(obs, deterministic=True)[0]).item())
                action_counts[str(action)] = action_counts.get(str(action), 0) + 1
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
        finally:
            env.close()
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {"mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
            "rows": rows, "starts": starts, "episode_hours": EPISODE_HOURS,
            "window_overlap": False, "action_counts": action_counts}


def compare(candidate: dict, reference: dict) -> dict[str, Any]:
    if candidate["starts"] != reference["starts"]:
        raise ValueError("paired evaluation windows do not match")
    result = {}
    for metric in ("total_cost_cny", "carbon_kg", "peak_kw"):
        gains = [100.0 * (base[metric] - row[metric]) / max(abs(base[metric]), 1e-9)
                 for row, base in zip(candidate["rows"], reference["rows"], strict=True)]
        summary = bootstrap_summary(gains, seed=20260912, resamples=5000)
        summary.update({"per_window_percent": gains, "minimum_percent": float(min(gains))})
        result[metric] = summary
    return result


def business_gates(evaluation: dict, comparison: dict) -> dict[str, bool]:
    checks = {}
    for metric, threshold in (("total_cost_cny", "cost_reduction_percent_min"),
                              ("carbon_kg", "carbon_reduction_percent_min"),
                              ("peak_kw", "peak_reduction_percent_min")):
        checks[metric + "_minimum_mean_gain"] = comparison[metric]["mean"] >= GATE[threshold]
        checks[metric + "_positive_95ci"] = comparison[metric]["ci_low"] > 0.0
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


def model_parameters(model: Any) -> dict[str, Any]:
    fields = ("num_timesteps", "_n_updates", "gamma", "learning_starts", "buffer_size", "batch_size",
              "gradient_steps", "target_update_interval", "exploration_fraction", "exploration_final_eps",
              "exploration_rate", "n_steps", "n_epochs", "gae_lambda", "ent_coef", "target_kl", "max_grad_norm")
    result = {key: getattr(model, key) for key in fields if hasattr(model, key)}
    result["learning_rate"] = float(model.lr_schedule(1.0))
    result["optimizer_learning_rates"] = [float(group["lr"]) for group in model.policy.optimizer.param_groups]
    if hasattr(model, "train_freq"):
        result["train_freq"] = {"frequency": model.train_freq.frequency, "unit": model.train_freq.unit.value}
    if hasattr(model, "clip_range"):
        result["clip_range"] = float(model.clip_range(1.0))
    result["policy_kwargs"] = {"net_arch": [128, 128]}
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
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default="dqn")
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
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from app.services.rl_model.shore_bess import v8_environment as environment

    torch.set_num_threads(1)
    cfg = load_config(CONFIG_PATH)
    dataset = load_port_dataset(cfg["dataset_id"])
    train = month_slice(dataset, "2024-01-01", "2025-05-01")
    validation = month_slice(dataset, "2025-05-01", "2025-08-01")
    starts = window_starts(validation.stop - validation.start, 6 if args.pilot else None)
    lattice = np.asarray(environment.LATTICE)
    idle_indices = np.flatnonzero(np.all(lattice == 0.0, axis=1))
    if len(idle_indices) != 1:
        raise ValueError("action lattice requires exactly one exact idle command")
    idle_action = int(idle_indices[0])

    def factory(source, split, seed=0, training=False):
        return lambda: environment.ShoreBESSV8Env(source, split, config=cfg, normalization_slice=train,
                   episode_steps=EPISODE_HOURS, carbon_price=args.carbon_price, discrete=True,
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
    action_count = int(probe.action_space.n)
    probe.close()
    algorithm_config = ({"learning_rate": 0.0001, "buffer_size": 200000, "batch_size": 256,
                         "train_freq": 4, "gradient_steps": 1, "learning_starts": 2000,
                         "target_update_interval": 1000, "exploration_fraction": 0.5,
                         "exploration_final_eps": 0.02} if args.algorithm == "dqn" else
                        {"learning_rate": 0.00015, "n_steps": 1344, "batch_size": 168,
                         "n_epochs": 10, "gae_lambda": 0.99, "ent_coef": 0.0001,
                         "clip_range": 0.15, "target_kl": 0.02, "max_grad_norm": 0.5})
    run_config = {"algorithm": "stable_baselines3." + args.algorithm.upper(), "seeds": seeds,
                  "requested_steps_per_seed": args.steps, "checkpoint_interval": args.block,
                  "episode_hours": EPISODE_HOURS, "gamma": 1.0, "network": [128, 128],
                  "carbon_price_cny_per_kg_constraint_multiplier": args.carbon_price,
                  "algorithm_parameters": algorithm_config, "discrete": True,
                  "action_lattice": lattice.tolist(), "state_names": list(environment.STATE_NAMES),
                  "observation_dimensions": observation_dimensions, "action_count": action_count,
                  "physical_config": cfg, "normalization": scaler, "admission_gate": GATE,
                  "pilot": args.pilot}
    write_json(run_dir / "config.json", run_config)
    manifest = {"schema": "port-shore-bess-v8-manifest.v1", "run_id": run_id, "started_at": utc_now(),
                "dataset_id": dataset.dataset_id, "dataset_sha256": dataset.fingerprint,
                "train": split_description(dataset, train), "validation": split_description(dataset, validation),
                "validation_starts": starts, "normalization_fit_split": "train_only",
                "selection_protocol": "validation_only_then_frozen_checkpoint_and_seed",
                "test_status": "not_opened" if args.pilot else "sealed_until_selection_previously_used_benchmark",
                "forward_status": "not_loaded" if args.pilot else "sealed_until_selection_previously_used_benchmark",
                "teacher_actions_used": False, "warm_start_used": False,
                "source_sha256": source_hashes, "historical_pointer_sha256": protected,
                "versions": {"python": platform.python_version(), "torch": torch.__version__, "stable_baselines3": sb3.__version__},
                "simulation_mode": True, "production_authority": False, "dispatch_allowed": False,
                "live_data_verified": False}
    write_json(run_dir / "manifest.json", manifest)
    reference = evaluate(factory(dataset, validation), None, starts, idle_action)
    write_json(run_dir / "validation_reference.json", reference)
    algorithm_cls = sb3.DQN if args.algorithm == "dqn" else sb3.PPO
    results = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        env = Monitor(factory(dataset, train, seed, True)(), str(seed_dir / "monitor.csv"))
        model = algorithm_cls("MlpPolicy", env, gamma=1.0, policy_kwargs={"net_arch": [128, 128]},
                              seed=seed, device="cpu", verbose=0, **algorithm_config)
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
            checks = business_gates(ev, comparison)
            row = {"step": int(model.num_timesteps), "optimizer_updates": int(model._n_updates),
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
            result = {"seed": seed, "steps": int(model.num_timesteps), "optimizer_updates": int(model._n_updates),
                      "initial_weights_sha256": initial_weights, "final_weights_sha256": weights_sha256(model),
                      "parameters": model_parameters(model), "selected": chosen,
                      "model_path": relative(selected_path), "model_sha256": file_sha256(selected_path),
                      "convergence": convergence(curve), "render_calls": env.unwrapped.render_calls}
            write_json(seed_dir / "result.json", result)
            results.append(result)
        finally:
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
        combined = joined_forward(dataset, forward)
        forward_split = slice(dataset.rows, dataset.rows + forward.rows)
        for label, source, split in (("test", dataset, test), ("forward", combined, forward_split)):
            test_starts = window_starts(split.stop - split.start)
            baseline = evaluate(factory(source, split), None, test_starts, idle_action)
            partition = {"status": "previously_used_chronological_benchmark_not_fresh_blind_data",
                         "dataset_id": dataset.dataset_id if label == "test" else forward.dataset_id,
                         "dataset_sha256": dataset.fingerprint if label == "test" else forward.fingerprint,
                         "split": split_description(source, split), "reference": baseline, "per_seed": []}
            for result in results:
                if file_sha256(ROOT / result["model_path"]) != result["model_sha256"]:
                    raise ValueError("selected model changed after selection freeze")
                model = algorithm_cls.load(ROOT / result["model_path"], device="cpu")
                ev = evaluate(factory(source, split), model, test_starts, idle_action)
                comparison = compare(ev, baseline)
                partition["per_seed"].append({"seed": result["seed"], "evaluation": ev,
                                              "comparison": comparison, "gates": business_gates(ev, comparison)})
            evaluations[label] = partition
            write_json(run_dir / f"{label}_evaluation.json", partition)

    checks = {"formal_run": not args.pilot, "three_independent_seeds": len(seeds) >= 3,
              "all_seeds_validation_passed": all(all(row["selected"]["gates"].values()) for row in results),
              "all_seeds_last_three_stable": all(row["convergence"]["passed"] for row in results),
              "real_optimizer_updates": all(row["optimizer_updates"] > 0 for row in results),
              "weights_changed": all(row["initial_weights_sha256"] != row["final_weights_sha256"] for row in results),
              "source_unchanged_during_training": all(file_sha256(ROOT / path) == value for path, value in source_hashes.items()),
              "historical_pointers_preserved": all(file_sha256(ROOT / path) == value for path, value in protected.items()),
              "no_training_rendering": all(row["render_calls"] == 0 for row in results),
              "test_all_seed_gates": "test" in evaluations and all(all(row["gates"].values()) for row in evaluations["test"]["per_seed"]),
              "forward_all_seed_gates": "forward" in evaluations and all(all(row["gates"].values()) for row in evaluations["forward"]["per_seed"])}
    admitted = all(checks.values())
    status = "PILOT_VALIDATION_ONLY" if args.pilot else ("ADMITTED_OFFLINE_RL" if admitted else "CANDIDATE_NOT_ADMITTED")
    report = {"schema": "port-shore-bess-v8-report.v1", "run_id": run_id, "generated_at": utc_now(),
              "status": status, "config": run_config, "manifest": manifest, "checks": checks,
              "results": results, "selected_seed": selected["seed"], "selection": selection,
              "evaluations": evaluations, "total_environment_steps": sum(row["steps"] for row in results),
              "total_optimizer_updates": sum(row["optimizer_updates"] for row in results),
              "promoted": admitted, "simulation_mode": True, "production_authority": False,
              "live_data_verified": False, "dispatch_allowed": False,
              "claim_boundary": "Public/engineering offline scenario. Repeated historical and forward benchmarks; no fresh blind test and no measured port savings."}
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
