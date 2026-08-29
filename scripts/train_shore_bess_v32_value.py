"""Warm-start a real PPO balanced Shore+BESS candidate and gate promotion.

The V3.1 economic champion remains immutable. Candidate selection uses only
the historical validation split. The pinned 2026 public forward challenge is
opened after selection and is never used to tune a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.services.rl_model.shore_bess.v3_environment import (
    CONTRACT, ShoreBESSEnv, chronological_slices, evaluate_windows,
    fixed_window_starts, load_config, neutral_policy,
)
from app.services.rl_training.datasets import PortDataset, load_port_dataset
from scripts.train_shore_bess_v3_safe import append_jsonl, model_policy, relative, sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evidence/v3/shore_bess"
CONFIG_PATH = ROOT / "config/shore_bess_v32_balanced.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def improvement(policy: float, baseline: float) -> float:
    return 100.0 * (baseline - policy) / max(abs(baseline), 1e-12)


def combine(old: PortDataset, forward: PortDataset) -> PortDataset:
    return PortDataset(
        dataset_id="public_cn_sha_history_plus_forward_eval_v1",
        path=forward.path,
        timestamps=[*old.timestamps, *forward.timestamps],
        values=np.vstack([old.values, forward.values]),
        metadata={**forward.metadata, "sha256": "evaluation_composite_not_a_published_dataset"},
        factor_values=np.vstack([old.factor_values, forward.factor_values]),
        factor_availability=np.vstack([old.factor_availability, forward.factor_availability]),
    )


def factory(dataset, split, config, normalization_slice, seed, *, training=False):
    return lambda: ShoreBESSEnv(
        dataset, split, config=config, normalization_slice=normalization_slice,
        episode_steps=int(config["training"]["episode_hours"]), seed=seed,
        training=training, record_trace=False,
    )


def metrics(policy: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "cost_reduction_vs_no_bess_percent": improvement(policy["total_cost_cny"], baseline["total_cost_cny"]),
        "carbon_reduction_vs_no_bess_percent": improvement(policy["carbon_kg"], baseline["carbon_kg"]),
        "peak_reduction_vs_no_bess_percent": improvement(policy["peak_kw"], baseline["peak_kw"]),
        "guardrail_violation_rate": policy["guardrail_violation_rate"],
        "terminal_soc_error": policy["terminal_soc_error"],
        "terminal_flex_backlog_kwh": policy["terminal_flex_backlog_kwh"],
        "shore_sla_violation_kwh": policy["shore_sla_violation_kwh"],
        "projection_rate": policy["projection_rate"],
    }


def admitted(values: dict[str, float], gate: dict[str, float], prefix: str) -> bool:
    return bool(
        values["cost_reduction_vs_no_bess_percent"] >= gate[f"{prefix}_cost_reduction_vs_no_bess_percent_min"]
        and values["carbon_reduction_vs_no_bess_percent"] >= gate[f"{prefix}_carbon_reduction_vs_no_bess_percent_min"]
        and values["peak_reduction_vs_no_bess_percent"] >= gate[f"{prefix}_peak_reduction_vs_no_bess_percent_min"]
        and values["guardrail_violation_rate"] <= gate["guardrail_violation_rate_max"]
        and values["terminal_soc_error"] <= gate["terminal_soc_error_max"]
        and values["terminal_flex_backlog_kwh"] <= gate["terminal_flex_backlog_kwh_max"]
        and values["shore_sla_violation_kwh"] <= 1e-9
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    config = load_config(CONFIG_PATH)
    gate = config["promotion_gate"]
    steps = int(args.steps or config["training"]["formal_steps_per_seed"])
    block = int(config["training"]["checkpoint_steps"])
    run_id = args.run_id or datetime.now(timezone.utc).strftime("shore-bess-v32-value-%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"append-only run already exists: {run_id}")
    run_dir.mkdir(parents=True)

    old_latest = json.loads((OUTPUT_ROOT / "latest.json").read_text(encoding="utf-8"))
    old_report = json.loads((ROOT / old_latest["report_path"]).read_text(encoding="utf-8"))
    old_models = {int(item["seed"]): item for item in old_report["artifacts"]["models"]}
    old = load_port_dataset(str(config["dataset_id"]))
    forward = load_port_dataset("public_cn_sha_forward_2026m05_v1")
    train_slice, validation_slice, _ = chronological_slices(old)
    episode = int(config["training"]["episode_hours"])
    validation_starts = fixed_window_starts(
        validation_slice.stop - validation_slice.start, episode,
        int(config["convergence_gate"]["fixed_validation_windows"]),
    )
    neutral_validation = evaluate_windows(
        factory(old, validation_slice, config, train_slice, 43), neutral_policy, validation_starts
    )
    write_json(run_dir / "manifest.json", {
        "schema": "port-dt-shore-bess-v32-value-manifest.v1", "run_id": run_id,
        "started_at": utc_now(), "config_path": relative(CONFIG_PATH),
        "config_sha256": sha256(CONFIG_PATH), "warm_start_run_id": old_latest["run_id"],
        "candidate_selection_dataset": old.dataset_id,
        "candidate_selection_rows": validation_slice.stop - validation_slice.start,
        "forward_dataset_id_sealed_until_selection": forward.dataset_id,
        "forward_dataset_sha256": forward.fingerprint, "training_rendering": False,
        "seeds": list(config["training"]["seeds"]), "rl_steps_per_seed": steps,
        "contract": CONTRACT.as_dict(), "promotion_gate": gate,
    })

    selected: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    for seed in (int(value) for value in config["training"]["seeds"]):
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        warm = old_models[seed]
        shutil.copy2(ROOT / warm["path"], seed_dir / "warm_start_v31.zip")
        env = ShoreBESSEnv(
            old, train_slice, config=config, normalization_slice=train_slice,
            episode_steps=episode, seed=seed, training=True, record_trace=False,
        )
        model = PPO.load(str(seed_dir / "warm_start_v31.zip"), env=env, device="cpu")
        curve: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        elapsed = 0
        while elapsed < steps:
            current = min(block, steps - elapsed)
            model.learn(total_timesteps=current, reset_num_timesteps=False, progress_bar=False)
            elapsed += current
            validation = evaluate_windows(
                factory(old, validation_slice, config, train_slice, seed),
                model_policy(model), validation_starts,
            )
            values = metrics(validation["mean"], neutral_validation["mean"])
            checkpoint = seed_dir / f"ppo_step_{elapsed}.zip"
            model.save(str(checkpoint.with_suffix("")))
            record = {"timesteps": elapsed, "validation": values, "validation_reward_mean": validation["mean"]["reward"], "checkpoint": relative(checkpoint)}
            curve.append(record)
            append_jsonl(seed_dir / "metrics.jsonl", {"ts": utc_now(), "seed": seed, **record})
            if admitted(values, gate, "validation") and (
                best is None or values["cost_reduction_vs_no_bess_percent"] > best["validation"]["cost_reduction_vs_no_bess_percent"]
            ):
                best = record
            print(f"seed={seed} step={elapsed}/{steps} cost={values['cost_reduction_vs_no_bess_percent']:.4f}% carbon={values['carbon_reduction_vs_no_bess_percent']:.4f}% peak={values['peak_reduction_vs_no_bess_percent']:.4f}% admitted={admitted(values, gate, 'validation')}", flush=True)
        result = {"seed": seed, "curve": curve, "validation_candidate": best, "validation_admitted": best is not None, "render_calls_during_training": env.render_calls}
        if best:
            selected_path = seed_dir / "selected_model.zip"
            shutil.copy2(ROOT / best["checkpoint"], selected_path)
            result.update({"selected_model_path": relative(selected_path), "selected_model_sha256": sha256(selected_path)})
            selected.append(result)
        write_json(seed_dir / "result.json", result)
        seed_results.append(result)
        env.close()

    forward_results: list[dict[str, Any]] = []
    if selected:
        combined = combine(old, forward)
        forward_slice = slice(old.rows, old.rows + forward.rows)
        forward_starts = fixed_window_starts(forward.rows, episode, max(12, forward.rows // episode))
        forward_neutral = evaluate_windows(
            factory(combined, forward_slice, config, train_slice, 43), neutral_policy, forward_starts
        )
        for result in selected:
            model = PPO.load(str(ROOT / result["selected_model_path"]), device="cpu")
            evaluation = evaluate_windows(
                factory(combined, forward_slice, config, train_slice, result["seed"]),
                model_policy(model), forward_starts,
            )
            values = metrics(evaluation["mean"], forward_neutral["mean"])
            forward_results.append({"seed": result["seed"], "metrics": values, "admitted": admitted(values, gate, "forward")})
    pass_rate = sum(bool(item["admitted"]) for item in forward_results) / max(1, len(config["training"]["seeds"]))
    promoted = bool(pass_rate + 1e-9 >= float(config["convergence_gate"]["minimum_seed_pass_rate"]))
    report = {
        "schema": "port-dt-shore-bess-v32-value-report.v1", "version": config["version"],
        "run_id": run_id, "generated_at": utc_now(),
        "status": "PROMOTED_BALANCED_FORWARD_PASS" if promoted else "REJECTED_BALANCED_GATE",
        "warm_start_champion": old_latest, "historical_report_preserved": True,
        "training": {"algorithm": config["training"]["algorithm"], "real_ppo_environment_steps_per_seed": steps, "seeds": list(config["training"]["seeds"]), "render_calls": sum(item["render_calls_during_training"] for item in seed_results)},
        "validation": {"neutral": neutral_validation, "per_seed": seed_results, "selection_access_to_forward": False},
        "forward_challenge": {"dataset_id": forward.dataset_id, "sha256": forward.fingerprint, "candidate_selection_allowed": False, "per_seed": forward_results, "seed_pass_rate": pass_rate},
        "promotion_gate": gate, "promoted": promoted, "production_admitted": False,
        "claim_boundary": config["claim_boundary"],
        "diagnosis_if_rejected": "The public benchmark's hourly carbon factor is a declared engineering derivative and tariff/carbon timing may make a balanced policy infeasible. Rejection is retained instead of fabricating a saving.",
    }
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    entry = {"schema": "port-dt-shore-bess-v32-value-pointer.v1", "run_id": run_id, "report_path": relative(report_path), "report_sha256": sha256(report_path), "status": report["status"], "promotion_state": "promoted" if promoted else "candidate_not_promoted", "updated_at": utc_now()}
    append_jsonl(OUTPUT_ROOT / "history_index.jsonl", entry)
    if promoted:
        admitted_models = [item for item in forward_results if item["admitted"]]
        champion_seed = max(admitted_models, key=lambda item: item["metrics"]["cost_reduction_vs_no_bess_percent"])["seed"]
        champion = next(item for item in selected if item["seed"] == champion_seed)
        write_json(OUTPUT_ROOT / "latest.json", {**entry, "model_path": champion["selected_model_path"], "model_sha256": champion["selected_model_sha256"]})
    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
