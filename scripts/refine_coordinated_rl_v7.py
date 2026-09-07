"""Refine three independent V6 SAC actors with fixed-validation checkpoints."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services.rl_training.datasets import load_port_dataset, file_sha256
from app.services.rl_training.model_artifacts import resolve_model_artifact
from app.services.rl_training.coordinated_environment import CoordinatedPortOperationsEnv
from scripts.train_business_rl_v7 import write, relative, window_starts
from scripts.train_coordinated_port_business_v6 import evaluate_windows, compare, make_env, metric_means

ROOT = Path(__file__).resolve().parents[1]


def checks(rows, reference):
    m = metric_means(rows)
    c = compare(rows, reference)
    gates = {"positive_business_value_vs_incumbent": c["coordinated_business_score_relative_improvement"]["ci_low"] > 0.0}
    for key, floor in {"service_completion_ratio": 0.72, "gate_service_completion_ratio": 0.95, "rail_service_completion_ratio": 0.8, "barge_service_completion_ratio": 0.8, "pilotage_service_completion_ratio": 0.8, "towage_service_completion_ratio": 0.8, "quay_crane_completion_ratio": 0.85, "horizontal_transport_completion_ratio": 0.85, "yard_crane_completion_ratio": 0.85, "regulatory_clearance_completion_ratio": 0.65, "shore_power_service_ratio": 0.45}.items():
        gates[key] = m[key] >= floor
    gates["zero_guardrail_violations"] = max(r["guardrail_violation_rate"] for r in rows) <= 1e-12
    gates["terminal_soc_recovered"] = max(r["terminal_soc_error"] for r in rows) <= 1e-6
    gates["latent_correction_bounded"] = m["latent_action_correction_mean"] <= 0.3
    # Incumbent paired non-inferiority prevents a composite gain hiding
    # materially worse unit economics, carbon or service coverage.
    for key in ("cost_per_teu", "carbon_kg_per_teu", "service_completion_ratio"):
        gates[key + "_noninferiority_2pct"] = c["metric_relative_improvement"][key]["ci_low"] >= -0.02
    return gates


def stable(curve):
    tail = curve[-3:]
    scores = np.array([r["comparison"]["coordinated_business_score_relative_improvement"]["mean"] for r in tail])
    spread = float(np.ptp(scores)) if len(scores) else 0.0
    return {"passed": len(tail) == 3 and spread <= 0.02 and all(all(r["checks"].values()) for r in tail), "tail_business_gain_range": spread, "max_range": 0.02, "tail_checkpoints": len(tail)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--block", type=int, default=5000)
    args = p.parse_args()
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    pointer_path = ROOT / "evidence/v6/coordinated_business/offline_champion.json"
    old_pointer_hash = file_sha256(pointer_path)
    pointer = json.loads(pointer_path.read_text())
    old = json.loads((ROOT / pointer["report_path"]).read_text())
    incumbent_path = ROOT / pointer["selected_model_path"]
    if file_sha256(incumbent_path) != pointer["selected_model_sha256"]:
        raise ValueError("incumbent model hash mismatch")
    cfg = json.loads((incumbent_path.parent / "config.json").read_text())
    dataset = load_port_dataset(cfg["dataset_id"])
    train, validation, test = dataset.split_three_way(0.2, 0.1)
    starts = window_starts(validation.stop - validation.start, 48, 16)
    run_id = "coordinated-sac-v7-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ROOT / "evidence/v7/coordinated_business"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {**cfg, "learning_rate": 0.0001, "additional_steps": args.steps, "train_freq": 4, "gradient_steps": 1, "block": args.block, "seed_sources": [{"seed": r["seed"], "model_path": r["model_path"], "model_sha256": r["model_sha256"]} for r in old["training"]["runs"]]}
    write(run_dir / "config.json", config)
    write(run_dir / "manifest.json", {"schema": "port-coordinated-rl-refinement.v7", "incumbent": pointer, "source_sha256": {relative(Path(__file__)): file_sha256(Path(__file__)), relative(ROOT / "app/services/rl_training/coordinated_environment.py"): file_sha256(ROOT / "app/services/rl_training/coordinated_environment.py")}, "dataset_sha256": dataset.fingerprint, "validation_starts": starts, "selection": "validation_only_against_current_RL_incumbent", "forward_status": "repeated_public_benchmark_not_new_blind_data", "observation_dimensions": 110, "action_dimensions": 18, "production_authority": False})
    incumbent = SAC.load(incumbent_path, device="cpu")
    env = make_env(dataset, validation, train, cfg)
    reference = evaluate_windows(incumbent, env, starts)
    env.close()
    write(run_dir / "validation_incumbent.json", reference)
    results = []
    for source in config["seed_sources"]:
        seed = source["seed"]
        source_path = resolve_model_artifact(ROOT, source["model_path"], source["model_sha256"])
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        env = CoordinatedPortOperationsEnv(dataset, train, action_mode="continuous", episode_steps=48, seed=seed, demand_cap_kw=cfg["demand_cap_kw"], reward_weights=cfg["reward_weights"], projection_penalty_weight=cfg["projection_penalty_weight"], regulatory_delay_penalty_weight=cfg["regulatory_delay_penalty_weight"], integrated_reward_weights=cfg["integrated_reward_weights"], coordinated_reward_weights=cfg["coordinated_reward_weights"], port_profile=cfg["port_profile"], normalization_slice=train, training=True, record_trace=False)
        monitored = Monitor(env, str(seed_dir / "monitor.csv"))
        # Resume learned actor/critics, explicitly override inherited settings.
        # Empty replay gets fresh training-only interactions before updates.
        model = SAC.load(source_path, env=monitored, device="cpu", learning_rate=config["learning_rate"], learning_starts=1000, train_freq=4, gradient_steps=1, buffer_size=100000)
        model.set_random_seed(seed)
        initial_updates = model._n_updates
        eval_env = make_env(dataset, validation, train, cfg)
        initial = evaluate_windows(model, eval_env, starts)
        write(seed_dir / "initial_validation.json", initial)
        curve = []
        best = None
        elapsed = 0
        while elapsed < args.steps:
            block = min(args.block, args.steps - elapsed)
            model.learn(total_timesteps=block, reset_num_timesteps=(elapsed == 0), progress_bar=False)
            elapsed = model.num_timesteps
            path = seed_dir / f"step_{elapsed}.zip"
            model.save(path)
            rows = evaluate_windows(model, eval_env, starts)
            c = compare(rows, reference)
            gates = checks(rows, reference)
            item = {"step": elapsed, "optimizer_updates": model._n_updates - initial_updates, "comparison": c, "checks": gates, "metrics": metric_means(rows), "model_path": relative(path), "model_sha256": file_sha256(path)}
            curve.append(item)
            rank = (sum(not x for x in gates.values()), -c["coordinated_business_score_relative_improvement"]["ci_low"])
            if best is None or rank < best[0]:
                best = (rank, item)
            write(seed_dir / "curve.json", curve)
            print(json.dumps({"module": "coordinated", "seed": seed, "step": elapsed, "gain_vs_incumbent": c["coordinated_business_score_relative_improvement"], "failed": [k for k,v in gates.items() if not v], "convergence": stable(curve)}), flush=True)
        selected_path = seed_dir / "selected_model.zip"
        shutil.copy2(ROOT / best[1]["model_path"], selected_path)
        result = {"seed": seed, "selected": best[1], "convergence": stable(curve), "model_path": relative(selected_path), "model_sha256": file_sha256(selected_path), "real_additional_environment_steps": elapsed, "real_additional_optimizer_updates": model._n_updates - initial_updates, "render_calls": env.render_calls}
        write(seed_dir / "result.json", result)
        results.append(result)
        eval_env.close()
        monitored.close()
    selected = min(results, key=lambda r: (not r["convergence"]["passed"], sum(not x for x in r["selected"]["checks"].values()), -r["selected"]["comparison"]["coordinated_business_score_relative_improvement"]["ci_low"]))
    write(run_dir / "selection.json", {"seed": selected["seed"], "model_sha256": selected["model_sha256"], "test_access": False})
    forward = load_port_dataset("public_cn_sha_integrated_forward_2026m05_v5")
    forward_starts = window_starts(forward.rows, 48, 20)
    env = make_env(forward, slice(0, forward.rows), train, cfg, normalization_dataset=dataset)
    forward_reference = evaluate_windows(incumbent, env, forward_starts)
    for result in results:
        model = SAC.load(ROOT / result["model_path"], device="cpu")
        rows = evaluate_windows(model, env, forward_starts)
        result["forward"] = {"metrics": metric_means(rows), "rows": rows, "comparison": compare(rows, forward_reference), "checks": checks(rows, forward_reference)}
    env.close()
    admitted = all(r["convergence"]["passed"] and all(r["forward"]["checks"].values()) for r in results) and len(results) >= 3
    report = {"schema": "port-coordinated-real-rl-report.v7", "run_id": run_id, "status": "ADMITTED_OFFLINE_RL" if admitted else "CANDIDATE_NOT_ADMITTED", "results": results, "selected_seed": selected["seed"], "selected_model_path": selected["model_path"], "selected_model_sha256": selected["model_sha256"], "forward_incumbent_metrics": metric_means(forward_reference), "forward_incumbent_rows": forward_reference, "forward_starts": forward_starts, "incumbent_preserved": file_sha256(pointer_path) == old_pointer_hash, "production_authority": False, "claim_boundary": "previously used public/engineering forward benchmark; not fresh blind or field-measured savings"}
    write(run_dir / "report.json", report)
    entry = {"run_id": run_id, "status": report["status"], "report_path": relative(run_dir / "report.json"), "report_sha256": file_sha256(run_dir / "report.json"), "model_path": selected["model_path"], "model_sha256": selected["model_sha256"], "production_authority": False}
    write(root / "latest.json", entry)
    if admitted:
        write(root / "offline_champion.json", entry)
    print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
