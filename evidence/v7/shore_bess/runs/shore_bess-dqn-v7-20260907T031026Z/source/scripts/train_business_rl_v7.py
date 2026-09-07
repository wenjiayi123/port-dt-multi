"""Pure SAC specialist training, immutable checkpoints and validation-only selection.

    .venv312/bin/python -m scripts.train_business_rl_v7 --module yard_lighting

Runs are append-only. Historical actors are evaluated as comparators, never
used as teachers. Forward benchmarks have been seen by earlier experiments;
this script explicitly labels them repeated benchmarks, not fresh blind data.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services.rl_training.business_learning import BusinessDeltaReward, FiniteDispatchActions
from app.services.rl_training.datasets import file_sha256, load_port_dataset
from app.services.rl_training.statistics import bootstrap_summary

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "hvac": ("hvac_cooling", "HVACV3Env", "hvac_v3.json"),
    "yard_crane": ("yard_crane", "YardCraneV3Env", "yard_crane_v3.json"),
    "yard_lighting": ("yard_lighting", "YardLightingV3Env", "yard_lighting_v3.json"),
    "shore_bess": ("shore_bess", "ShoreBESSEnv", "shore_bess_v32_balanced.json"),
    "bess_energy": ("bess_energy", "BESSEnergyV3Env", "bess_energy_v32_grid_only.json"),
}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def window_starts(length, episode, count=8):
    # Paired non-overlapping windows; never bootstrap duplicated/overlapping
    # episodes as if they were independent evidence.
    available = list(range(0, length - episode, episode))
    if not available:
        raise ValueError("split too short for a complete evaluation episode")
    indices = np.linspace(0, len(available) - 1, min(count, len(available)), dtype=int)
    return [available[i] for i in indices]


def evaluate(module, factory, policy, starts):
    rows = []
    for start in starts:
        value = module.evaluate_windows(factory, policy, [start])
        rows.append(value["mean"])
    return {"mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, "rows": rows, "starts": starts}


def compare(candidate, reference):
    out = {}
    for name in ("total_cost_cny", "carbon_kg", "peak_kw", "energy_kwh"):
        if name not in candidate["mean"]:
            continue
        gains = [100.0 * (b[name] - a[name]) / max(abs(b[name]), 1e-9) for a, b in zip(candidate["rows"], reference["rows"], strict=True)]
        out[name] = bootstrap_summary(gains, seed=20260907)
    return out


def gates(name, evaluation, reference, config):
    m = evaluation["mean"]
    comparison = compare(evaluation, reference)
    requirements = config.get("business_gate", {})
    result = {}
    for metric, threshold_name in (("total_cost_cny", "cost_reduction_percent_min"), ("peak_kw", "peak_reduction_percent_min"), ("energy_kwh", "energy_reduction_percent_min")):
        if metric in comparison:
            threshold = float(requirements.get(threshold_name, 0.0))
            result[metric + "_business_gate"] = comparison[metric]["mean"] >= threshold
    result["carbon_non_regression"] = comparison["carbon_kg"]["mean"] >= 0.0
    for metric in ("guardrail_violation_rate", "terminal_soc_error", "terminal_flex_backlog_kwh", "shore_sla_violation_kwh"):
        if metric in m:
            result[metric] = max(row[metric] for row in evaluation["rows"]) <= (1e-6 if metric.startswith("terminal") else 1e-12)
    for metric in ("cooling_satisfaction_rate", "moves_retention_rate", "job_sla_non_degradation_rate", "minimum_lux_compliance_rate", "critical_lux_compliance_rate"):
        if metric in m:
            result[metric] = min(row[metric] for row in evaluation["rows"]) >= 1.0 - 1e-9
    result["positive_cost_value_95ci"] = comparison["total_cost_cny"]["ci_low"] > 0.0
    return result


def convergence(curve):
    tail = curve[-3:]
    gains = np.asarray([r["comparison"]["total_cost_cny"]["mean"] for r in tail])
    spread = float(np.ptp(gains)) if len(gains) else float("inf")
    # Stability is measured on controllable savings (percentage points), not
    # on a huge constant total-cost denominator that makes every actor flat.
    tolerance = max(0.10, 0.10 * abs(float(np.mean(gains)))) if len(gains) else 0.0
    passed = len(tail) == 3 and spread <= tolerance and all(all(r["gates"].values()) for r in tail)
    return {"passed": bool(passed), "tail_cost_gain_range_pp": spread, "tolerance_pp": tolerance, "tail_checkpoints": len(tail)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=RECIPES, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--block", type=int, default=5000)
    parser.add_argument("--seeds", default="907,1007,1107")
    parser.add_argument("--run-id")
    parser.add_argument("--algorithm", choices=("sac", "dqn"), default="sac")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--carbon-multiplier", type=float, default=1.0)
    parser.add_argument("--peak-multiplier", type=float, default=1.0)
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--strict-terminal", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or args.steps < 3 * args.block:
        parser.error("distinct seeds and at least three checkpoint blocks are required")
    import torch
    from stable_baselines3 import SAC, DQN
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    module_name, cls_name, config_name = RECIPES[args.module]
    module = importlib.import_module("app.services.rl_model." + module_name + ".v3_environment")
    cfg = module.load_config(ROOT / "config" / config_name)
    if args.strict_terminal:
        if args.module != "bess_energy":
            parser.error("strict-terminal is an explicit BESS profile")
        cfg["asset"]["terminal_soc_tolerance"] = 0.0
    dataset = module.load_dataset(cfg) if hasattr(module, "load_dataset") else load_port_dataset(cfg["dataset_id"])
    train, validation, test = module.chronological_slices(dataset)
    episode = args.episode_steps or int(cfg["training"].get("episode_steps", cfg["training"].get("episode_hours", 168)))
    # Preserve all normalization from training data and the original contract.
    env_cls = getattr(module, cls_name)
    def factory(split, seed=0, training=False):
        return lambda: env_cls(dataset, split, config=cfg, normalization_slice=train, episode_steps=episode, seed=seed, training=training, record_trace=False)
    val_starts = window_starts(validation.stop - validation.start, episode)
    test_starts = window_starts(test.stop - test.start, episode)
    run_id = args.run_id or args.module + "-" + args.algorithm + "-v7-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ROOT / "evidence/v7" / args.module
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    description = dataset.describe(validation_ratio=0.1, test_ratio=0.2) if args.module == "hvac" else dataset.describe()
    if not description.get("quality", {}).get("training_eligible"):
        raise ValueError("dataset quality gate failed")
    lattice = None
    if args.algorithm == "dqn":
        if args.module not in {"shore_bess", "bess_energy"}:
            parser.error("DQN lattice is registered for energy modules only")
        lattice = [[b, f] for b in (-0.5, -0.1, -0.03, 0.0, 0.03, 0.1, 0.5) for f in ((-1.0, 0.0, 1.0) if args.module == "shore_bess" else (-1.0,))]
    algorithm_cls = SAC if args.algorithm == "sac" else DQN
    config = {"module": args.module, "algorithm": "stable_baselines3." + args.algorithm.upper(), "seeds": seeds, "steps": args.steps, "block": args.block, "episode_steps": episode, "gamma": 0.995, "learning_rate": args.learning_rate, "entropy_coefficient": args.entropy, "carbon_multiplier": args.carbon_multiplier, "peak_multiplier": args.peak_multiplier, "train_freq": 4, "gradient_steps": 1, "network": [64, 64], "action_lattice": lattice, "config": cfg}
    def predict(model, obs):
        a = model.predict(obs, deterministic=True)[0]
        return np.asarray(lattice[int(a)], dtype=np.float32) if lattice is not None else a
    write(run_dir / "config.json", config)
    old_pointer = ROOT / "evidence/v3" / args.module / "latest.json"
    protected = {relative(p): file_sha256(p) for p in (old_pointer,) if p.exists()}
    source_files = [Path(__file__), ROOT / module.__file__, ROOT / "app/services/rl_training/business_learning.py", ROOT / "config" / config_name]
    for source in source_files:
        snapshot = run_dir / "source" / relative(source)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
    write(run_dir / "manifest.json", {"schema": "port-business-real-rl.v7", "run_id": run_id, "dataset": description, "protocol": {"selection": "fixed_chronological_validation_only", "validation_starts": val_starts, "test_starts": test_starts, "window_overlap": False, "test_status": "previously_used_chronological_benchmark_not_fresh_blind_data", "teacher_actions_used": False, "seeds": seeds}, "code_sha256": {relative(p): file_sha256(p) for p in source_files}, "historical_pointer_sha256": protected, "production_authority": False})
    neutral_val = evaluate(module, factory(validation), module.neutral_policy, val_starts)
    write(run_dir / "validation_reference.json", neutral_val)
    results = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        raw_env = factory(train, seed, True)()
        wrapped = BusinessDeltaReward(raw_env, args.module, config["gamma"], carbon_multiplier=args.carbon_multiplier, peak_multiplier=args.peak_multiplier)
        if lattice is not None:
            wrapped = FiniteDispatchActions(wrapped, lattice)
        env = Monitor(wrapped, str(seed_dir / "monitor.csv"))
        kwargs = {"ent_coef": config["entropy_coefficient"]} if lattice is None else {"exploration_fraction": 0.4, "exploration_final_eps": 0.03, "target_update_interval": 500}
        model = algorithm_cls("MlpPolicy", env, learning_rate=config["learning_rate"], gamma=config["gamma"], learning_starts=1000, buffer_size=100000, batch_size=256, train_freq=config["train_freq"], gradient_steps=config["gradient_steps"], policy_kwargs={"net_arch": config["network"]}, seed=seed, device="cpu", verbose=0, **kwargs)
        actor = lambda obs, _env: predict(model, obs)
        initial = evaluate(module, factory(validation), actor, val_starts)
        write(seed_dir / "initial_validation.json", initial)
        curve = []
        best = None
        while model.num_timesteps < args.steps:
            model.learn(total_timesteps=min(args.block, args.steps - model.num_timesteps), reset_num_timesteps=False, progress_bar=False)
            checkpoint = seed_dir / f"step_{model.num_timesteps}.zip"
            model.save(checkpoint)
            ev = evaluate(module, factory(validation), actor, val_starts)
            c = compare(ev, neutral_val)
            checks = gates(args.module, ev, neutral_val, cfg)
            row = {"step": model.num_timesteps, "updates": model._n_updates, "model_path": relative(checkpoint), "model_sha256": file_sha256(checkpoint), "comparison": c, "gates": checks, "metrics": ev["mean"], "optimizer": {str(k): float(v) for k, v in model.logger.name_to_value.items() if k.startswith("train/") and np.isscalar(v)}}
            curve.append(row)
            rank = (sum(not v for v in checks.values()), -c["total_cost_cny"]["ci_low"])
            if best is None or rank < best[0]:
                best = (rank, row)
            write(seed_dir / "curve.json", curve)
            print(json.dumps({"module": args.module, "seed": seed, "step": model.num_timesteps, "gain_percent": {k: round(v["mean"], 5) for k,v in c.items()}, "failed": [k for k,v in checks.items() if not v], "convergence": convergence(curve)}, ensure_ascii=False), flush=True)
        selected = seed_dir / "selected_model.zip"
        shutil.copy2(ROOT / best[1]["model_path"], selected)
        result = {"seed": seed, "steps": model.num_timesteps, "optimizer_updates": model._n_updates, "selected": best[1], "model_path": relative(selected), "model_sha256": file_sha256(selected), "convergence": convergence(curve), "initial_comparison": compare(initial, neutral_val), "render_calls": raw_env.render_calls}
        write(seed_dir / "result.json", result)
        results.append(result)
        env.close()
    # Freeze selection BEFORE evaluating test rows. Do not select a seed by test.
    selected = min(results, key=lambda r: (not r["convergence"]["passed"], sum(not v for v in r["selected"]["gates"].values()), -r["selected"]["comparison"]["total_cost_cny"]["ci_low"]))
    write(run_dir / "selection.json", {"selected_seed": selected["seed"], "model_path": selected["model_path"], "model_sha256": selected["model_sha256"], "selection_access_to_test": False})
    reference_test = evaluate(module, factory(test), module.neutral_policy, test_starts)
    for result in results:
        model = algorithm_cls.load(ROOT / result["model_path"], device="cpu")
        ev = evaluate(module, factory(test), lambda o,e: predict(model, o), test_starts)
        result["test"] = {"evaluation": ev, "comparison": compare(ev, reference_test), "gates": gates(args.module, ev, reference_test, cfg)}
    selected = next(r for r in results if r["seed"] == selected["seed"])
    checks = {"three_independent_seeds": len(seeds) >= 3, "all_seeds_converged": all(r["convergence"]["passed"] for r in results), "all_seeds_business_pass": all(all(r["test"]["gates"].values()) for r in results), "real_optimizer_updates": all(r["optimizer_updates"] > 0 for r in results), "historical_pointers_preserved": all(file_sha256(ROOT / p) == h for p,h in protected.items()), "no_training_rendering": all(r["render_calls"] == 0 for r in results), "multiple_nonoverlapping_test_windows": len(test_starts) >= 3}
    report = {"schema": "port-business-real-rl-report.v7", "run_id": run_id, "module": args.module, "status": "ADMITTED_OFFLINE_RL" if all(checks.values()) else "CANDIDATE_NOT_ADMITTED", "checks": checks, "config": config, "results": results, "selected_seed": selected["seed"], "test_reference": reference_test, "selected_model_path": selected["model_path"], "selected_model_sha256": selected["model_sha256"], "production_authority": False, "live_data_verified": False, "dispatch_allowed": False, "claim_boundary": "engineering/public offline benchmark, not measured port savings; historical test previously used, not a new blind evaluation"}
    write(run_dir / "report.json", report)
    pointer = {"run_id": run_id, "status": report["status"], "report_path": relative(run_dir / "report.json"), "report_sha256": file_sha256(run_dir / "report.json"), "model_path": selected["model_path"], "model_sha256": selected["model_sha256"], "production_authority": False}
    write(root / "latest.json", pointer)
    if all(checks.values()):
        write(root / "offline_champion.json", pointer)
    print(json.dumps(pointer), flush=True)


if __name__ == "__main__":
    main()
