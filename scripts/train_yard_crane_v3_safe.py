"""Formal multi-seed yard-crane V3.1 training and sealed chronological blind test."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.services.rl_model.yard_crane.v3_environment import (
    ACTION_NAMES, CONTRACT, NumpyMLPPolicy, YardCraneV3Env, artifact_policy,
    chronological_slices, evaluate_windows, fixed_window_starts, load_config,
    load_dataset, neutral_policy, safe_teacher_policy,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evidence" / "v3" / "yard_crane"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def export_actor(path: Path, model, *, seed: int, epoch: int, config: Dict[str, Any]) -> None:
    import torch.nn as nn

    layers = []
    for module in model:
        if isinstance(module, nn.Linear):
            layers.append({
                "weight": module.weight.detach().cpu().numpy().tolist(),
                "bias": module.bias.detach().cpu().numpy().tolist(),
            })
    write_json(path, {
        "schema": "port-dt-yard-crane-safe-actor.v1",
        "algorithm": "constrained_mpc_teacher_actor_distillation",
        "seed": seed, "selected_epoch": epoch,
        "state_dimensions": CONTRACT["state_dimensions"],
        "action_dimensions": CONTRACT["action_dimensions"],
        "action_names": ACTION_NAMES,
        "network": config["training"]["network"], "output_activation": "tanh", "layers": layers,
    })


def model_policy(model):
    import torch

    def policy(observation, _env):
        with torch.no_grad():
            return model(torch.as_tensor(observation, dtype=torch.float32)).cpu().numpy().astype(np.float32)
    return policy


def make_factory(dataset, split, config, train_slice, seed, *, trace=False):
    return lambda: YardCraneV3Env(
        dataset, split, config=config, normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_steps"]), seed=seed,
        training=False, record_trace=trace,
    )


def collect_teacher_data(dataset, train_slice, config, seed) -> Tuple[np.ndarray, np.ndarray]:
    env = YardCraneV3Env(
        dataset, train_slice, config=config, normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_steps"]), seed=seed, training=True,
    )
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    for episode in range(int(config["training"]["teacher_trajectories_per_seed"])):
        observation, _ = env.reset(seed=seed * 1000 + episode)
        done = False
        while not done:
            action = safe_teacher_policy(observation, env)
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
    if env.render_calls:
        raise RuntimeError("yard-crane training rendered unexpectedly")
    env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def build_actor(config: Dict[str, Any]):
    import torch.nn as nn

    layers: List[nn.Module] = []
    incoming = int(CONTRACT["state_dimensions"])
    for width in [int(value) for value in config["training"]["network"]]:
        layers.extend([nn.Linear(incoming, width), nn.ReLU()])
        incoming = width
    layers.extend([nn.Linear(incoming, int(CONTRACT["action_dimensions"])), nn.Tanh()])
    return nn.Sequential(*layers)


def train_one_seed(*, seed, seed_dir, dataset, train_slice, validation_slice, config, validation_starts):
    import torch

    torch.manual_seed(seed)
    observations, actions = collect_teacher_data(dataset, train_slice, config, seed)
    model = build_actor(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    x, y = torch.as_tensor(observations), torch.as_tensor(actions)
    rng = np.random.default_rng(seed)
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    every = int(config["training"]["validation_every_epochs"])
    factory = make_factory(dataset, validation_slice, config, train_slice, seed)
    curve: List[Dict[str, float]] = []
    best = None
    optimizer_updates = 0
    for epoch in range(1, epochs + 1):
        losses = []
        indices = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            index = torch.as_tensor(indices[start:start + batch_size])
            predicted = model(x[index])
            loss = torch.mean((predicted - y[index]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_updates += 1
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses))
        if epoch == 1 or epoch % every == 0 or epoch == epochs:
            validation = evaluate_windows(factory, model_policy(model), validation_starts)
            record = {
                "epoch": epoch, "optimizer_updates": optimizer_updates, "imitation_loss": epoch_loss,
                "validation_reward_mean": validation["mean"]["reward"],
                "validation_total_cost_cny": validation["mean"]["total_cost_cny"],
                "validation_peak_kw": validation["mean"]["peak_kw"],
                "validation_energy_kwh": validation["mean"]["energy_kwh"],
                "validation_carbon_kg": validation["mean"]["carbon_kg"],
                "validation_moves_retention_rate": validation["mean"]["moves_retention_rate"],
                "validation_job_sla_non_degradation_rate": validation["mean"]["job_sla_non_degradation_rate"],
                "validation_projection_rate": validation["mean"]["projection_rate"],
                "validation_guardrail_violation_rate": validation["mean"]["guardrail_violation_rate"],
            }
            curve.append(record)
            append_jsonl(seed_dir / "metrics.jsonl", {"ts": utc_now(), "seed": seed, **record})
            checkpoint = seed_dir / f"checkpoint_epoch_{epoch}.json"
            export_actor(checkpoint, model, seed=seed, epoch=epoch, config=config)
            admissible = (
                record["validation_guardrail_violation_rate"] <= 1e-12
                and record["validation_moves_retention_rate"] >= config["business_gate"]["moves_satisfaction_rate_min"]
                and record["validation_job_sla_non_degradation_rate"] >= config["business_gate"]["job_sla_satisfaction_rate_min"]
            )
            if admissible and (best is None or record["validation_total_cost_cny"] < best["record"]["validation_total_cost_cny"]):
                best = {"record": record, "checkpoint": checkpoint}
            print(
                f"seed={seed} epoch={epoch}/{epochs} loss={epoch_loss:.7f} "
                f"validation_cost={record['validation_total_cost_cny']:.2f} "
                f"moves={record['validation_moves_retention_rate']:.4f}", flush=True,
            )
    if best is None:
        raise RuntimeError(f"yard-crane seed {seed} has no safety-admissible checkpoint")
    selected_path = seed_dir / "selected_model.json"
    shutil.copy2(best["checkpoint"], selected_path)
    selected_validation = evaluate_windows(factory, artifact_policy(NumpyMLPPolicy.load(selected_path)), validation_starts)
    tail = curve[-min(3, len(curve)):]
    loss_reduction = 1.0 - curve[-1]["imitation_loss"] / max(curve[0]["imitation_loss"], 1e-12)
    tail_costs = np.asarray([row["validation_total_cost_cny"] for row in tail])
    tail_range = float((tail_costs.max() - tail_costs.min()) / max(abs(tail_costs.mean()), 1.0))
    convergence = {
        "passed": bool(
            loss_reduction >= config["convergence_gate"]["loss_reduction_min"]
            and tail_range <= config["convergence_gate"]["tail_relative_range_max"]
        ),
        "criterion": "actor_imitation_loss_reduction_plus_fixed_validation_business_plateau",
        "loss_reduction_ratio": loss_reduction,
        "loss_reduction_min": config["convergence_gate"]["loss_reduction_min"],
        "tail_validation_cost_relative_range": tail_range,
        "tail_validation_cost_relative_range_max": config["convergence_gate"]["tail_relative_range_max"],
        "selected_epoch": int(best["record"]["epoch"]),
        "selection_metric": "minimum_fixed_validation_cost_subject_to_moves_sla_and_guardrail_gates",
        "blind_test_used_for_selection": False,
    }
    result = {
        "seed": seed, "training_samples": len(observations), "optimizer_updates": optimizer_updates,
        "curve": curve, "convergence": convergence, "selected_validation": selected_validation,
        "selected_model_path": relative(selected_path), "selected_model_sha256": sha256(selected_path),
        "render_calls_during_training": 0,
    }
    write_json(seed_dir / "result.json", result)
    return result


def improvement(policy_value: float, baseline_value: float) -> float:
    return 100.0 * (baseline_value - policy_value) / max(abs(baseline_value), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal yard-crane V3.1 safe-policy training")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    config = load_config()
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else list(config["training"]["seeds"])
    run_id = args.run_id or datetime.now(timezone.utc).strftime("yard-crane-v3-safe-%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"append-only run already exists: {run_id}")
    run_dir.mkdir(parents=True)

    dataset = load_dataset(config)
    train_slice, validation_slice, blind_slice = chronological_slices(dataset)
    description = dataset.describe()
    if not description["quality"]["training_eligible"]:
        raise RuntimeError("yard-crane dataset quality gate failed")
    steps = int(config["training"]["episode_steps"])
    validation_starts = fixed_window_starts(validation_slice.stop - validation_slice.start, steps, config["convergence_gate"]["fixed_validation_windows"])
    blind_starts = fixed_window_starts(blind_slice.stop - blind_slice.start, steps, 8)
    validation_factory = make_factory(dataset, validation_slice, config, train_slice, seeds[0])
    blind_factory = make_factory(dataset, blind_slice, config, train_slice, seeds[0], trace=True)
    validation_neutral = evaluate_windows(validation_factory, neutral_policy, validation_starts)
    validation_teacher = evaluate_windows(validation_factory, safe_teacher_policy, validation_starts)
    manifest = {
        "schema": "port-dt-yard-crane-formal-run.v1", "run_id": run_id, "created_at": utc_now(),
        "dataset": description, "config_path": "config/yard_crane_v3.json",
        "config_sha256": sha256(ROOT / "config" / "yard_crane_v3.json"),
        "protocol": {"seeds": seeds, "validation_starts": validation_starts, "blind_test_starts_sealed_until_selection": blind_starts},
    }
    write_json(run_dir / "manifest.json", manifest)

    results = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        results.append(train_one_seed(
            seed=seed, seed_dir=seed_dir, dataset=dataset, train_slice=train_slice,
            validation_slice=validation_slice, config=config, validation_starts=validation_starts,
        ))
    converged = [result for result in results if result["convergence"]["passed"]]
    selected = min(converged or results, key=lambda result: result["selected_validation"]["mean"]["total_cost_cny"])
    selected_path = ROOT / selected["selected_model_path"]
    blind_neutral = evaluate_windows(blind_factory, neutral_policy, blind_starts)
    blind_policy = evaluate_windows(blind_factory, artifact_policy(NumpyMLPPolicy.load(selected_path)), blind_starts)
    baseline, policy = blind_neutral["mean"], blind_policy["mean"]
    cost_gain = improvement(policy["total_cost_cny"], baseline["total_cost_cny"])
    energy_gain = improvement(policy["energy_kwh"], baseline["energy_kwh"])
    peak_gain = improvement(policy["peak_kw"], baseline["peak_kw"])
    carbon_gain = improvement(policy["carbon_kg"], baseline["carbon_kg"])
    window_hours = steps * 0.25
    annual_scale = config["counterfactual"]["annual_operating_hours"] / window_hours
    business_metrics = {
        "cost_reduction_vs_historical_control_percent": cost_gain,
        "energy_reduction_vs_historical_control_percent": energy_gain,
        "peak_reduction_vs_historical_control_percent": peak_gain,
        "carbon_reduction_vs_historical_control_percent": carbon_gain,
        "historical_moves_retention_rate_percent": 100.0 * policy["moves_retention_rate"],
        "job_sla_non_degradation_rate_percent": 100.0 * policy["job_sla_non_degradation_rate"],
        "delay_delta_minutes": policy["delay_delta_minutes"],
        "annualized_scenario_savings_cny": (baseline["total_cost_cny"] - policy["total_cost_cny"]) * annual_scale,
        "annualized_scenario_carbon_reduction_t": (baseline["carbon_kg"] - policy["carbon_kg"]) * annual_scale / 1000.0,
        "window_hours": window_hours, "window_count": len(blind_starts), "claim_eligible": False,
        "scope": "checked_in_engineering_emulator_chronological_blind_test_not_site_measurement",
    }
    gates = config["business_gate"]
    quality_gates = {
        "dataset_quality_passed": bool(description["quality"]["training_eligible"]),
        "convergence_passed": len(converged) == len(seeds), "seed_pass_rate": len(converged) / len(seeds),
        "business_advantage_passed": bool(
            cost_gain >= gates["cost_reduction_percent_min"]
            and energy_gain >= gates["energy_reduction_percent_min"]
            and peak_gain >= gates["peak_reduction_percent_min"]
        ),
        "moves_service_passed": policy["moves_retention_rate"] >= gates["moves_satisfaction_rate_min"],
        "job_sla_passed": policy["job_sla_non_degradation_rate"] >= gates["job_sla_satisfaction_rate_min"],
        "guardrail_violation_rate": policy["guardrail_violation_rate"],
        "safety_passed": policy["guardrail_violation_rate"] <= gates["guardrail_violation_rate_max"],
    }
    quality_gates["public_offline_admitted"] = all(quality_gates[key] for key in (
        "dataset_quality_passed", "convergence_passed", "business_advantage_passed",
        "moves_service_passed", "job_sla_passed", "safety_passed",
    ))

    epochs = sorted(set(row["epoch"] for result in results for row in result["curve"]))
    aggregate_curve = []
    for epoch in epochs:
        rows = [next(row for row in result["curve"] if row["epoch"] == epoch) for result in results]
        aggregate_curve.append({
            "epoch": epoch, "optimizer_updates": int(np.mean([row["optimizer_updates"] for row in rows])),
            "imitation_loss_mean": float(np.mean([row["imitation_loss"] for row in rows])),
            "imitation_loss_std": float(np.std([row["imitation_loss"] for row in rows])),
            "validation_cost_mean_cny": float(np.mean([row["validation_total_cost_cny"] for row in rows])),
            "validation_cost_std_cny": float(np.std([row["validation_total_cost_cny"] for row in rows])),
            "validation_peak_mean_kw": float(np.mean([row["validation_peak_kw"] for row in rows])),
            "validation_moves_retention_mean": float(np.mean([row["validation_moves_retention_rate"] for row in rows])),
        })
    convergence = {
        "passed": quality_gates["convergence_passed"], "seed_pass_rate": quality_gates["seed_pass_rate"],
        "per_seed": [{"seed": result["seed"], **result["convergence"]} for result in results],
        "aggregate_curve": aggregate_curve, "blind_test_used_for_selection": False,
    }
    report = {
        "schema": "port-dt-yard-crane-formal-report.v1", "version": "V3.1", "run_id": run_id,
        "created_at": utc_now(), "status": "passed" if quality_gates["public_offline_admitted"] else "failed",
        "dataset": description,
        "training": {
            "algorithm": "constrained_mpc_teacher_actor_distillation",
            "legacy_algorithms_preserved": ["IQL", "TD3+BC", "Quantile Safe-SAC", "Crane MPC"],
            "seeds": seeds, "samples_per_seed": [r["training_samples"] for r in results],
            "optimizer_updates_per_seed": [r["optimizer_updates"] for r in results],
            "selection": "minimum fixed-validation cost after moves/SLA/safety gates",
            "render_calls_during_training": sum(r["render_calls_during_training"] for r in results),
        },
        "contract": CONTRACT,
        "counterfactual_model": {
            **config["counterfactual"], "measured": False,
            "thermal_state": "derived engineering proxy, not a sensor reading",
            "replacement_requirement": "replace power elasticity and thermal proxies with site-calibrated PLC curves before production admission",
        },
        "validation": {"neutral": validation_neutral, "teacher": validation_teacher},
        "convergence": convergence,
        "blind_test": {
            "windows": len(blind_starts), "window_hours": window_hours, "selection_access": False,
            "baseline": blind_neutral, "selected_policy": blind_policy,
            "sample_real_model_inference": blind_policy["windows"][0]["last_info"],
        },
        "business_metrics": business_metrics, "quality_gates": quality_gates,
        "artifacts": {"models": [{"seed": selected["seed"], "path": relative(selected_path), "sha256": sha256(selected_path)}]},
        "algorithm_registry": [
            {"name": "Legacy IQL", "state": "code_preserved", "reason": "old base actions were all zero; retraining evidence was not admitted"},
            {"name": "Legacy TD3+BC", "state": "code_preserved", "reason": "old augmentation had only 144 rows"},
            {"name": "Legacy Quantile Safe-SAC", "state": "history_preserved", "reason": "old policy artifact remains 0 bytes"},
            {"name": "Constrained Crane MPC teacher", "state": "active_comparator", "reason": "explicit service-aware teacher"},
            {"name": "Safe actor distillation", "state": "public_offline_candidate", "reason": "three seeds and sealed chronological blind test"},
            {"name": "CMDP safety projection", "state": "active", "reason": "16 fail-closed SLA, thermal, dwell, PCC and DR constraints"},
        ],
    }
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    latest = {
        "schema": "port-dt-yard-crane-latest-pointer.v1", "run_id": run_id, "status": report["status"],
        "report_path": relative(report_path), "report_sha256": sha256(report_path),
        "model_path": relative(selected_path), "model_sha256": sha256(selected_path), "updated_at": utc_now(),
    }
    write_json(OUTPUT_ROOT / "latest.json", latest)
    append_jsonl(OUTPUT_ROOT / "history_index.jsonl", latest)
    print(json.dumps({**latest, "business_metrics": business_metrics, "quality_gates": quality_gates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
