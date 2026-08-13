"""Evaluate a validation-selected grid-only BESS candidate once on 2026 forward data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services.rl_model.bess_energy.v3_environment import (
    BESSEnergyV3Env, chronological_slices, evaluate_windows, fixed_window_starts,
    load_config, neutral_policy,
)
from app.services.rl_training.datasets import PortDataset, load_port_dataset
from scripts.train_bess_energy_v3_safe import model_policy, no_bess_baseline, relative, sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evidence/v3/bess_energy"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def gain(policy: float, baseline: float) -> float:
    return 100.0 * (baseline - policy) / max(abs(baseline), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    from stable_baselines3 import PPO

    run_dir = OUTPUT_ROOT / "runs" / args.run_id
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = load_config(ROOT / "config/bess_energy_v32_grid_only.json")
    old = load_port_dataset("public_cn_sha_hourly_v3")
    forward = load_port_dataset("public_cn_sha_forward_2026m05_v1")
    train_slice, _, _ = chronological_slices(old)
    combined = PortDataset(
        "public_cn_sha_history_plus_forward_eval_v1", forward.path,
        [*old.timestamps, *forward.timestamps], np.vstack([old.values, forward.values]),
        {**forward.metadata, "sha256": "evaluation_composite_not_published"},
        np.vstack([old.factor_values, forward.factor_values]),
        np.vstack([old.factor_availability, forward.factor_availability]),
    )
    forward_slice = slice(old.rows, old.rows + forward.rows)
    episode = int(config["training"]["episode_hours"])
    starts = fixed_window_starts(forward.rows, episode, max(12, forward.rows // episode))

    def factory(seed: int):
        return lambda: BESSEnergyV3Env(
            combined, forward_slice, config=config, normalization_slice=train_slice,
            episode_steps=episode, seed=seed, training=False, record_trace=False,
        )

    baseline = no_bess_baseline(evaluate_windows(factory(71), neutral_policy, starts))
    rows = []
    for artifact in report.get("artifacts", {}).get("models", []):
        seed = int(artifact["seed"])
        seed_result = json.loads((run_dir / f"seed_{seed}/result.json").read_text(encoding="utf-8"))
        model = PPO.load(str(ROOT / artifact["path"]), device="cpu")
        evaluation = evaluate_windows(factory(seed), model_policy(model), starts)
        policy = evaluation["mean"]
        metrics = {
            "cost_reduction_vs_no_bess_percent": gain(policy["total_cost_cny"], baseline["mean"]["total_cost_cny"]),
            "carbon_reduction_vs_no_bess_percent": gain(policy["carbon_kg"], baseline["mean"]["carbon_kg"]),
            "peak_reduction_vs_no_bess_percent": gain(policy["peak_kw"], baseline["mean"]["peak_kw"]),
            "event_compliance_rate_percent": 100.0 * policy["event_compliance_rate"],
            "guardrail_violation_rate": policy["guardrail_violation_rate"],
            "terminal_soc_error": policy["terminal_soc_error"],
            "reserve_revenue_cny": policy["reserve_revenue_cny"],
            "dr_revenue_cny": policy["dr_revenue_cny"],
            "annualized_scenario_savings_cny": (
                baseline["mean"]["total_cost_cny"] - policy["total_cost_cny"]
            ) * 365.25 / 7.0,
            "annualized_scenario_carbon_reduction_t": (
                baseline["mean"]["carbon_kg"] - policy["carbon_kg"]
            ) * 365.25 / 7.0 / 1000.0,
            "claim_eligible": False,
        }
        forward_passed = bool(
            metrics["cost_reduction_vs_no_bess_percent"] >= 0.0
            and metrics["carbon_reduction_vs_no_bess_percent"] >= 0.0
            and metrics["peak_reduction_vs_no_bess_percent"] >= 0.0
            and metrics["guardrail_violation_rate"] <= 1e-12
            and metrics["terminal_soc_error"] <= 1e-6
            and metrics["reserve_revenue_cny"] == 0.0
            and metrics["dr_revenue_cny"] == 0.0
        )
        rows.append({
            "seed": seed, "validation_selection_gate_passed": bool(seed_result.get("selection_gate_passed")),
            "forward_gate_passed": forward_passed, "admitted": bool(seed_result.get("selection_gate_passed") and forward_passed),
            "metrics": metrics, "model_path": artifact["path"], "model_sha256": artifact["sha256"],
        })
    pass_rate = sum(bool(row["admitted"]) for row in rows) / max(1, len(rows))
    admitted = bool(pass_rate + 1e-9 >= float(config["convergence_gate"]["minimum_seed_pass_rate"]))
    payload = {
        "schema": "port-dt-bess-grid-only-forward-v32.v1", "run_id": args.run_id,
        "generated_at": now(), "status": "GRID_ONLY_PROFILE_FORWARD_PASS" if admitted else "GRID_ONLY_PROFILE_FORWARD_REJECTED",
        "dataset": {"dataset_id": forward.dataset_id, "sha256": forward.fingerprint, "rows": forward.rows, "start_at": forward.timestamps[0], "end_at": forward.timestamps[-1], "candidate_selection_allowed": False},
        "profile": {"market_revenue_enabled": False, "reserve_revenue_cny": 0.0, "dr_revenue_cny": 0.0, "production_authority": False},
        "windows": len(starts), "window_hours": episode, "per_seed": rows,
        "seed_pass_rate": pass_rate, "admitted_public_offline_profile": admitted,
        "claim_boundary": config["claim_boundary"],
    }
    evidence_path = run_dir / "forward_challenge_2026.json"
    write_json(evidence_path, payload)
    pointer = {"schema": "port-dt-bess-grid-only-latest.v1", "run_id": args.run_id, "status": payload["status"], "report_path": relative(report_path), "forward_evidence_path": relative(evidence_path), "forward_evidence_sha256": sha256(evidence_path), "updated_at": now()}
    if admitted:
        champion = max((row for row in rows if row["admitted"]), key=lambda row: row["metrics"]["cost_reduction_vs_no_bess_percent"])
        pointer.update({"model_path": champion["model_path"], "model_sha256": champion["model_sha256"]})
    write_json(OUTPUT_ROOT / "latest_grid_only.json", pointer)
    print(json.dumps(pointer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
