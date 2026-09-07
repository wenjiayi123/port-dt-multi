"""Retrain AGV IQL with chronological separation and honest offline diagnostics.

No policy-value or real savings claim is made from behavior-data row proxies.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from app.services.rl_model.agv_charge import train_iql as engine
from app.services.rl_training.datasets import file_sha256
from scripts.train_business_rl_v7 import ROOT, write, relative


def main():
    torch.set_num_threads(1)
    source = ROOT / "app/services/rl_model/agv_charge"
    before = {relative(p): file_sha256(p) for p in [source / "policy.bin", source / "policy_meta.json"]}
    S, A, R, S2, done, original_meta = engine.build_dataset(source, hours=72, time_col="timestamp")
    times = np.asarray(original_meta.pop("transition_timestamps"))
    unique = np.unique(times)
    train_end = unique[int(0.7 * len(unique))]
    validation_end = unique[int(0.8 * len(unique))]
    train = times < train_end
    validation = (times >= train_end) & (times < validation_end)
    test = times >= validation_end
    # Purge the final five-minute transition before either split boundary.
    train &= times < unique[int(0.7 * len(unique)) - 1]
    validation &= times < unique[int(0.8 * len(unique)) - 1]
    meta = copy.deepcopy(original_meta)
    mean = S[train].astype(np.float64).mean(0)
    std = S[train].astype(np.float64).std(0)
    std[std < 1e-3] = 1.0
    meta["standardize"] = {"mean": mean.tolist(), "std": std.tolist()}
    run_id = "agv-iql-v7-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ROOT / "evidence/v7/agv_charge"
    run = root / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    write(run / "manifest.json", {"run_id": run_id, "algorithm": "IQL", "seeds": [947,1047,1147], "optimizer_steps_per_seed": 10000, "train_rows": int(train.sum()), "validation_rows": int(validation.sum()), "test_rows": int(test.sum()), "normalization": "float64_training_rows_only_near_constant_features_use_unit_scale", "split": "chronological_70_10_20_with_one_transition_purge", "input_hashes": {relative(p): file_sha256(p) for p in sorted((source / "data").glob("*")) if p.is_file()}, "source_sha256": file_sha256(Path(engine.__file__)), "prior_model_hashes": before, "policy_value_estimator_available": False, "production_authority": False})
    results = []
    for seed in (947,1047,1147):
        path = run / f"seed_{seed}"
        path.mkdir()
        engine.ART_DIR = path
        engine.MIRROR_DIR = path / "mirror"
        engine.CUM_RL_COST = engine.CUM_BASE_COST = 0.0
        engine.BEST_Q_LOSS = engine.BEST_V_LOSS = engine.BEST_PI_LOSS = None
        curves = []
        def evaluate(agent, steps):
            with torch.no_grad():
                state = torch.as_tensor((S[validation] - mean) / std, dtype=torch.float32)
                next_state = torch.as_tensor((S2[validation] - mean) / std, dtype=torch.float32)
                action = torch.as_tensor(A[validation], dtype=torch.float32)
                target = torch.as_tensor(R[validation], dtype=torch.float32) + 0.995 * (1.0 - torch.as_tensor(done[validation], dtype=torch.float32)) * agent.v_targ(next_state)
                q1 = agent.q1(torch.cat([state, action], dim=1))
                q2 = agent.q2(torch.cat([state, action], dim=1))
                predictions = agent.pi(state)
                curves.append({"optimizer_steps": steps, "validation_bellman_mse": float(((q1-target)**2+(q2-target)**2).mean()), "validation_action_mae_vs_behavior": float((predictions-action).abs().mean()), "charge_ratio_mean": float(predictions.mean()), "behavior_charge_ratio_mean": float(action.mean()), "finite_outputs": bool(torch.isfinite(predictions).all()), "metric_scope": "fixed_validation_behavior_transition_diagnostics_not_counterfactual_business_value"})
            write(path / "validation_curve.json", curves)
        engine.train_iql_np(S[train], A[train], R[train], S2[train], done[train], meta, path, steps=10000, batch_size=256, lr=0.0001, seed=seed, log_every=1000, evaluation_callback=evaluate)
        policy = path / "policy.bin"
        result = {"seed": seed, "optimizer_steps": 10000, "curve": curves, "model_path": relative(policy), "model_sha256": file_sha256(policy), "business_admitted": False}
        results.append(result)
        print(json.dumps({"seed": seed, "last_validation": curves[-1]}), flush=True)
    report = {"schema": "port-agv-iql-audit.v7", "run_id": run_id, "status": "OFFLINE_IQL_TRAINED_NOT_BUSINESS_ADMITTED", "algorithm": "IQL", "results": results, "historical_models_preserved": all(file_sha256(ROOT / p) == h for p,h in before.items()), "reason_not_admitted": "Logged next-state transitions cannot establish counterfactual fleet SOC, charger congestion, deadline service or metered peak. Validation critic error and behavior-action fit are not measured business value.", "test_access": False, "production_authority": False, "claim_eligible": False}
    write(run / "report.json", report)
    write(root / "latest.json", {"run_id": run_id, "status": report["status"], "report_path": relative(run / "report.json"), "report_sha256": file_sha256(run / "report.json"), "production_authority": False})


if __name__ == "__main__":
    main()
