"""Formal multi-seed yard-lighting V3.1 training and chronological blind test."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.services.rl_model.yard_lighting.v3_environment import (
    ACTION_NAMES, CONTRACT, NumpyMLPPolicy, YardLightingV3Env, artifact_policy,
    chronological_slices, evaluate_windows, fixed_window_starts, load_config,
    load_dataset, neutral_policy, safe_teacher_policy,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evidence" / "v3" / "yard_lighting"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def export_actor(path: Path, model, seed: int, epoch: int, config: Dict[str, Any]) -> None:
    import torch.nn as nn

    layers = [{"weight": module.weight.detach().cpu().numpy().tolist(), "bias": module.bias.detach().cpu().numpy().tolist()}
              for module in model if isinstance(module, nn.Linear)]
    write(path, {
        "schema": "port-dt-yard-lighting-safe-actor.v1",
        "algorithm": "constrained_lux_teacher_actor_distillation", "seed": seed, "selected_epoch": epoch,
        "state_dimensions": CONTRACT["state_dimensions"], "action_dimensions": CONTRACT["action_dimensions"],
        "action_names": ACTION_NAMES, "network": config["training"]["network"], "output_activation": "tanh", "layers": layers,
    })


def build_actor(config: Dict[str, Any]):
    import torch.nn as nn

    modules: List[nn.Module] = []
    incoming = CONTRACT["state_dimensions"]
    for width in config["training"]["network"]:
        modules.extend([nn.Linear(incoming, int(width)), nn.ReLU()])
        incoming = int(width)
    modules.extend([nn.Linear(incoming, CONTRACT["action_dimensions"]), nn.Tanh()])
    return nn.Sequential(*modules)


def torch_policy(model):
    import torch

    def policy(observation, _env):
        with torch.no_grad():
            return model(torch.as_tensor(observation, dtype=torch.float32)).cpu().numpy().astype(np.float32)
    return policy


def factory(dataset, split, config, train_slice, seed, trace=False):
    return lambda: YardLightingV3Env(
        dataset, split, config=config, normalization_slice=train_slice,
        episode_steps=config["training"]["episode_steps"], seed=seed, training=False, record_trace=trace,
    )


def teacher_samples(dataset, train_slice, config, seed) -> Tuple[np.ndarray, np.ndarray]:
    env = YardLightingV3Env(
        dataset, train_slice, config=config, normalization_slice=train_slice,
        episode_steps=config["training"]["episode_steps"], seed=seed, training=True,
    )
    observations, actions = [], []
    for episode in range(config["training"]["teacher_trajectories_per_seed"]):
        observation, _ = env.reset(seed=seed * 1000 + episode)
        done = False
        while not done:
            action = safe_teacher_policy(observation, env)
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
    if env.render_calls:
        raise RuntimeError("yard-lighting training rendered unexpectedly")
    env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def train_seed(seed, seed_dir, dataset, train_slice, validation_slice, config, starts):
    import torch

    torch.manual_seed(seed)
    observations, actions = teacher_samples(dataset, train_slice, config, seed)
    model = build_actor(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    x, y = torch.as_tensor(observations), torch.as_tensor(actions)
    rng = np.random.default_rng(seed)
    curve, best, updates = [], None, 0
    validation_factory = factory(dataset, validation_slice, config, train_slice, seed)
    for epoch in range(1, config["training"]["epochs"] + 1):
        losses = []
        indices = rng.permutation(len(x))
        for start in range(0, len(x), config["training"]["batch_size"]):
            index = torch.as_tensor(indices[start:start + config["training"]["batch_size"]])
            loss = torch.mean((model(x[index]) - y[index]) ** 2)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.item())); updates += 1
        epoch_loss = float(np.mean(losses))
        if epoch == 1 or epoch % config["training"]["validation_every_epochs"] == 0 or epoch == config["training"]["epochs"]:
            validation = evaluate_windows(validation_factory, torch_policy(model), starts)
            record = {
                "epoch": epoch, "optimizer_updates": updates, "imitation_loss": epoch_loss,
                "validation_reward_mean": validation["mean"]["reward"],
                "validation_total_cost_cny": validation["mean"]["total_cost_cny"],
                "validation_peak_kw": validation["mean"]["peak_kw"],
                "validation_energy_kwh": validation["mean"]["energy_kwh"],
                "validation_carbon_kg": validation["mean"]["carbon_kg"],
                "validation_lux_compliance_rate": validation["mean"]["minimum_lux_compliance_rate"],
                "validation_critical_lux_compliance_rate": validation["mean"]["critical_lux_compliance_rate"],
                "validation_projection_rate": validation["mean"]["projection_rate"],
                "validation_guardrail_violation_rate": validation["mean"]["guardrail_violation_rate"],
            }
            curve.append(record); append(seed_dir / "metrics.jsonl", {"ts": now(), "seed": seed, **record})
            checkpoint = seed_dir / f"checkpoint_epoch_{epoch}.json"
            export_actor(checkpoint, model, seed, epoch, config)
            safe = (
                record["validation_lux_compliance_rate"] >= config["business_gate"]["minimum_lux_compliance_rate"]
                and record["validation_critical_lux_compliance_rate"] >= config["business_gate"]["critical_lux_compliance_rate"]
                and record["validation_guardrail_violation_rate"] <= config["business_gate"]["guardrail_violation_rate_max"]
            )
            if safe and (best is None or record["validation_total_cost_cny"] < best["record"]["validation_total_cost_cny"]):
                best = {"record": record, "checkpoint": checkpoint}
            print(f"seed={seed} epoch={epoch}/{config['training']['epochs']} loss={epoch_loss:.7f} cost={record['validation_total_cost_cny']:.2f} lux={record['validation_lux_compliance_rate']:.4f}", flush=True)
    if best is None:
        raise RuntimeError(f"lighting seed {seed} has no safety-admissible checkpoint")
    selected_path = seed_dir / "selected_model.json"
    shutil.copy2(best["checkpoint"], selected_path)
    selected_validation = evaluate_windows(validation_factory, artifact_policy(NumpyMLPPolicy.load(selected_path)), starts)
    tail = curve[-min(3, len(curve)):]
    loss_reduction = 1.0 - curve[-1]["imitation_loss"] / max(curve[0]["imitation_loss"], 1e-12)
    costs = np.asarray([row["validation_total_cost_cny"] for row in tail])
    tail_range = float((costs.max() - costs.min()) / max(abs(costs.mean()), 1.0))
    convergence = {
        "passed": bool(loss_reduction >= config["convergence_gate"]["loss_reduction_min"] and tail_range <= config["convergence_gate"]["tail_relative_range_max"]),
        "criterion": "actor_imitation_loss_reduction_plus_fixed_validation_business_plateau",
        "loss_reduction_ratio": loss_reduction, "loss_reduction_min": config["convergence_gate"]["loss_reduction_min"],
        "tail_validation_cost_relative_range": tail_range, "tail_validation_cost_relative_range_max": config["convergence_gate"]["tail_relative_range_max"],
        "selected_epoch": best["record"]["epoch"], "selection_metric": "minimum_fixed_validation_cost_subject_to_lux_and_safety_gates",
        "blind_test_used_for_selection": False,
    }
    result = {
        "seed": seed, "training_samples": len(observations), "optimizer_updates": updates, "curve": curve,
        "convergence": convergence, "selected_validation": selected_validation,
        "selected_model_path": rel(selected_path), "selected_model_sha256": sha(selected_path), "render_calls_during_training": 0,
    }
    write(seed_dir / "result.json", result)
    return result


def improvement(policy: float, baseline: float) -> float:
    return 100.0 * (baseline - policy) / max(abs(baseline), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None); parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    config = load_config()
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else list(config["training"]["seeds"])
    run_id = args.run_id or datetime.now(timezone.utc).strftime("yard-lighting-v3-safe-%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"append-only run exists: {run_id}")
    run_dir.mkdir(parents=True)
    dataset = load_dataset(config)
    train_slice, validation_slice, blind_slice = chronological_slices(dataset)
    description = dataset.describe()
    if not description["quality"]["training_eligible"]:
        raise RuntimeError("lighting dataset quality gate failed")
    steps = config["training"]["episode_steps"]
    validation_starts = fixed_window_starts(validation_slice.stop - validation_slice.start, steps, config["convergence_gate"]["fixed_validation_windows"])
    blind_starts = fixed_window_starts(blind_slice.stop - blind_slice.start, steps, 5)
    validation_factory = factory(dataset, validation_slice, config, train_slice, seeds[0])
    blind_factory = factory(dataset, blind_slice, config, train_slice, seeds[0], True)
    validation_neutral = evaluate_windows(validation_factory, neutral_policy, validation_starts)
    validation_teacher = evaluate_windows(validation_factory, safe_teacher_policy, validation_starts)
    write(run_dir / "manifest.json", {
        "schema": "port-dt-yard-lighting-formal-run.v1", "run_id": run_id, "created_at": now(), "dataset": description,
        "config_path": "config/yard_lighting_v3.json", "config_sha256": sha(ROOT / "config" / "yard_lighting_v3.json"),
        "protocol": {"seeds": seeds, "validation_starts": validation_starts, "blind_test_starts_sealed_until_selection": blind_starts},
    })
    results = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"; seed_dir.mkdir()
        results.append(train_seed(seed, seed_dir, dataset, train_slice, validation_slice, config, validation_starts))
    converged = [row for row in results if row["convergence"]["passed"]]
    selected = min(converged or results, key=lambda row: row["selected_validation"]["mean"]["total_cost_cny"])
    selected_path = ROOT / selected["selected_model_path"]
    blind_neutral = evaluate_windows(blind_factory, neutral_policy, blind_starts)
    blind_policy = evaluate_windows(blind_factory, artifact_policy(NumpyMLPPolicy.load(selected_path)), blind_starts)
    baseline, policy = blind_neutral["mean"], blind_policy["mean"]
    cost_gain, energy_gain = improvement(policy["total_cost_cny"], baseline["total_cost_cny"]), improvement(policy["energy_kwh"], baseline["energy_kwh"])
    peak_gain, carbon_gain = improvement(policy["peak_kw"], baseline["peak_kw"]), improvement(policy["carbon_kg"], baseline["carbon_kg"])
    window_hours = steps * 5.0 / 60.0
    annual_scale = config["counterfactual"]["annual_operating_hours"] / window_hours
    metrics = {
        "cost_reduction_vs_historical_control_percent": cost_gain, "energy_reduction_vs_historical_control_percent": energy_gain,
        "peak_reduction_vs_historical_control_percent": peak_gain, "carbon_reduction_vs_historical_control_percent": carbon_gain,
        "minimum_lux_compliance_rate_percent": 100.0 * policy["minimum_lux_compliance_rate"],
        "critical_lux_compliance_rate_percent": 100.0 * policy["critical_lux_compliance_rate"],
        "under_lux_zone_steps": policy["under_lux_zone_count"],
        "annualized_scenario_savings_cny": (baseline["total_cost_cny"] - policy["total_cost_cny"]) * annual_scale,
        "annualized_scenario_carbon_reduction_t": (baseline["carbon_kg"] - policy["carbon_kg"]) * annual_scale / 1000.0,
        "window_hours": window_hours, "window_count": len(blind_starts), "claim_eligible": False,
        "scope": "public_signal_enriched_engineering_emulator_chronological_blind_test_not_site_measurement",
    }
    gate = config["business_gate"]
    quality = {
        "dataset_quality_passed": description["quality"]["training_eligible"], "convergence_passed": len(converged) == len(seeds),
        "seed_pass_rate": len(converged) / len(seeds),
        "business_advantage_passed": cost_gain >= gate["cost_reduction_percent_min"] and energy_gain >= gate["energy_reduction_percent_min"] and peak_gain >= gate["peak_reduction_percent_min"],
        "minimum_lux_passed": policy["minimum_lux_compliance_rate"] >= gate["minimum_lux_compliance_rate"],
        "critical_lux_passed": policy["critical_lux_compliance_rate"] >= gate["critical_lux_compliance_rate"],
        "guardrail_violation_rate": policy["guardrail_violation_rate"],
        "safety_passed": policy["guardrail_violation_rate"] <= gate["guardrail_violation_rate_max"],
    }
    quality["public_offline_admitted"] = all(quality[key] for key in ("dataset_quality_passed", "convergence_passed", "business_advantage_passed", "minimum_lux_passed", "critical_lux_passed", "safety_passed"))
    epochs = sorted(set(row["epoch"] for result in results for row in result["curve"]))
    aggregate = []
    for epoch in epochs:
        rows = [next(row for row in result["curve"] if row["epoch"] == epoch) for result in results]
        aggregate.append({
            "epoch": epoch, "optimizer_updates": int(np.mean([row["optimizer_updates"] for row in rows])),
            "imitation_loss_mean": float(np.mean([row["imitation_loss"] for row in rows])),
            "imitation_loss_std": float(np.std([row["imitation_loss"] for row in rows])),
            "validation_cost_mean_cny": float(np.mean([row["validation_total_cost_cny"] for row in rows])),
            "validation_cost_std_cny": float(np.std([row["validation_total_cost_cny"] for row in rows])),
            "validation_peak_mean_kw": float(np.mean([row["validation_peak_kw"] for row in rows])),
            "validation_lux_compliance_mean": float(np.mean([row["validation_lux_compliance_rate"] for row in rows])),
        })
    convergence = {"passed": quality["convergence_passed"], "seed_pass_rate": quality["seed_pass_rate"], "per_seed": [{"seed": row["seed"], **row["convergence"]} for row in results], "aggregate_curve": aggregate, "blind_test_used_for_selection": False}
    report = {
        "schema": "port-dt-yard-lighting-formal-report.v1", "version": "V3.1", "run_id": run_id, "created_at": now(),
        "status": "passed" if quality["public_offline_admitted"] else "failed", "dataset": description,
        "training": {"algorithm": "constrained_lux_teacher_actor_distillation", "legacy_algorithm_preserved": "IQL", "seeds": seeds,
                     "samples_per_seed": [row["training_samples"] for row in results], "optimizer_updates_per_seed": [row["optimizer_updates"] for row in results],
                     "selection": "minimum fixed-validation cost after lux and safety gates", "render_calls_during_training": 0},
        "contract": CONTRACT,
        "counterfactual_model": {**config["counterfactual"], "measured": False, "replacement_requirement": "replace fixture power/lux curves and zone topology with site photometric commissioning data before production"},
        "validation": {"neutral": validation_neutral, "teacher": validation_teacher}, "convergence": convergence,
        "blind_test": {"windows": len(blind_starts), "window_hours": window_hours, "selection_access": False, "baseline": blind_neutral, "selected_policy": blind_policy, "sample_real_model_inference": blind_policy["windows"][0]["last_info"]},
        "business_metrics": metrics, "quality_gates": quality,
        "artifacts": {"models": [{"seed": selected["seed"], "path": rel(selected_path), "sha256": sha(selected_path)}]},
        "algorithm_registry": [
            {"name": "Legacy IQL", "state": "historical_preserved", "reason": "old policy remains OOD-blocked after correct boolean parsing"},
            {"name": "CQL", "state": "code_preserved", "reason": "must beat the identical lux-safe validation gate before promotion"},
            {"name": "Residual Safe-SAC", "state": "code_preserved", "reason": "requires site shadow replay before promotion"},
            {"name": "Lux-constrained teacher", "state": "active_comparator", "reason": "zone-level minimum lux and action envelopes"},
            {"name": "Safe actor distillation", "state": "public_offline_candidate", "reason": "three seeds and sealed chronological blind test"},
            {"name": "Lighting safety projection", "state": "active", "reason": "17 fail-closed lux, ramp, dwell, fault, override and rollback constraints"},
        ],
    }
    report_path = run_dir / "report.json"; write(report_path, report)
    latest = {"schema": "port-dt-yard-lighting-latest-pointer.v1", "run_id": run_id, "status": report["status"], "report_path": rel(report_path), "report_sha256": sha(report_path), "model_path": rel(selected_path), "model_sha256": sha(selected_path), "updated_at": now()}
    write(OUTPUT_ROOT / "latest.json", latest); append(OUTPUT_ROOT / "history_index.jsonl", latest)
    print(json.dumps({**latest, "business_metrics": metrics, "quality_gates": quality}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
