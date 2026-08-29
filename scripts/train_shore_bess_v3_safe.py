"""Formal three-seed Shore+BESS safe-policy training and blind evaluation.

The accepted actor is first trained against a constraint-projected peak/valley
teacher on chronological training rows.  Every checkpoint is evaluated on fixed
validation windows.  Any optional RL fine-tune must beat that actor and pass the
same safety gate before it can replace it; failed SAC/PPO probes are never
silently promoted.  Blind-test rows are opened only after checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.services.rl_model.shore_bess.v3_environment import (
    CONTRACT,
    ShoreBESSEnv,
    chronological_slices,
    evaluate_windows,
    fixed_window_starts,
    load_config,
    load_public_dataset,
    neutral_policy,
    rule_peak_valley_policy,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evidence" / "v3" / "shore_bess"


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


def model_policy(model):
    def policy(observation, _env):
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return policy


def make_eval_factory(dataset, split, config, train_slice, seed, *, trace=False):
    return lambda: ShoreBESSEnv(
        dataset,
        split,
        config=config,
        normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_hours"]),
        seed=seed,
        training=False,
        record_trace=trace,
    )


def collect_teacher_data(dataset, train_slice, config, seed) -> Tuple[np.ndarray, np.ndarray]:
    env = ShoreBESSEnv(
        dataset,
        train_slice,
        config=config,
        normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_hours"]),
        seed=seed,
        training=True,
        record_trace=False,
    )
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    for episode in range(int(config["training"]["behavior_trajectories_per_seed"])):
        observation, _ = env.reset(seed=seed * 1000 + episode)
        done = False
        while not done:
            action = rule_peak_valley_policy(observation, env)
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
    if env.render_calls:
        raise RuntimeError("training environment rendered unexpectedly")
    env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def train_one_seed(
    *,
    seed: int,
    seed_dir: Path,
    dataset,
    train_slice: slice,
    validation_slice: slice,
    config: Dict[str, Any],
    validation_starts: List[int],
    teacher_validation: Dict[str, Any],
) -> Dict[str, Any]:
    import torch
    from stable_baselines3 import PPO

    torch.manual_seed(seed)
    training_env = ShoreBESSEnv(
        dataset,
        train_slice,
        config=config,
        normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_hours"]),
        seed=seed,
        training=True,
        record_trace=False,
    )
    model = PPO(
        "MlpPolicy",
        training_env,
        learning_rate=float(config["training"]["behavior_learning_rate"]),
        n_steps=int(config["training"]["episode_hours"]) * 2,
        batch_size=84,
        n_epochs=5,
        gamma=float(config["training"]["gamma"]),
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs={"net_arch": list(config["training"]["network"])},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    observations, actions = collect_teacher_data(dataset, train_slice, config, seed)
    x = torch.as_tensor(observations, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.policy.parameters(), lr=float(config["training"]["behavior_learning_rate"])
    )
    rng = np.random.default_rng(seed)
    validation_factory = make_eval_factory(
        dataset, validation_slice, config, train_slice, seed
    )
    batch_size = int(config["training"]["behavior_batch_size"])
    epochs = int(config["training"]["behavior_epochs"])
    evaluate_every = int(config["training"]["behavior_validation_every_epochs"])
    curve: List[Dict[str, float]] = []
    best: Dict[str, Any] | None = None
    last_epoch_loss = float("nan")

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(x))
        losses: List[float] = []
        for start in range(0, len(x), batch_size):
            index = torch.as_tensor(indices[start : start + batch_size])
            distribution = model.policy.get_distribution(x[index])
            predicted = distribution.distribution.mean
            loss = torch.mean((predicted - y[index]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        last_epoch_loss = float(np.mean(losses))
        if epoch == 1 or epoch % evaluate_every == 0 or epoch == epochs:
            validation = evaluate_windows(
                validation_factory, model_policy(model), validation_starts
            )
            record = {
                "epoch": epoch,
                "optimizer_updates": epoch * int(np.ceil(len(x) / batch_size)),
                "imitation_loss": last_epoch_loss,
                "validation_reward_mean": float(validation["mean"]["reward"]),
                "validation_total_cost_cny": float(validation["mean"]["total_cost_cny"]),
                "validation_peak_kw": float(validation["mean"]["peak_kw"]),
                "validation_carbon_kg": float(validation["mean"]["carbon_kg"]),
                "validation_projection_rate": float(validation["mean"]["projection_rate"]),
                "validation_guardrail_violation_rate": float(
                    validation["mean"]["guardrail_violation_rate"]
                ),
                "validation_terminal_soc_error": float(
                    validation["mean"]["terminal_soc_error"]
                ),
            }
            curve.append(record)
            append_jsonl(seed_dir / "metrics.jsonl", {"ts": utc_now(), "seed": seed, **record})
            checkpoint = seed_dir / f"checkpoint_epoch_{epoch}.zip"
            model.save(str(checkpoint.with_suffix("")))
            safety_ok = bool(
                record["validation_guardrail_violation_rate"] <= 1e-12
                and record["validation_terminal_soc_error"] <= 1e-6
            )
            if safety_ok and (
                best is None
                or record["validation_total_cost_cny"]
                < best["record"]["validation_total_cost_cny"]
            ):
                best = {"record": record, "checkpoint": checkpoint}
            print(
                f"seed={seed} epoch={epoch}/{epochs} loss={last_epoch_loss:.7f} "
                f"validation_cost={record['validation_total_cost_cny']:.2f}",
                flush=True,
            )

    if best is None:
        raise RuntimeError(f"seed {seed} has no safety-admissible checkpoint")
    selected_path = seed_dir / "selected_model.zip"
    shutil.copy2(best["checkpoint"], selected_path)
    selected = PPO.load(str(selected_path), device="cpu")
    selected_validation = evaluate_windows(
        validation_factory, model_policy(selected), validation_starts
    )
    tail = curve[-min(3, len(curve)) :]
    loss_reduction = 1.0 - curve[-1]["imitation_loss"] / max(
        curve[0]["imitation_loss"], 1e-12
    )
    tail_costs = np.asarray(
        [row["validation_total_cost_cny"] for row in tail], dtype=np.float64
    )
    tail_cost_relative_range = float(
        (np.max(tail_costs) - np.min(tail_costs)) / max(1.0, abs(np.mean(tail_costs)))
    )
    convergence = {
        "passed": bool(loss_reduction >= 0.75 and tail_cost_relative_range <= 0.01),
        "criterion": "imitation_loss_reduction_plus_fixed_validation_business_plateau",
        "loss_reduction_ratio": loss_reduction,
        "loss_reduction_min": 0.75,
        "tail_validation_cost_relative_range": tail_cost_relative_range,
        "tail_validation_cost_relative_range_max": 0.01,
        "selected_epoch": int(best["record"]["epoch"]),
        "selection_metric": "minimum_fixed_validation_total_cost_subject_to_zero_guardrail_violations",
        "blind_test_used_for_selection": False,
    }
    result = {
        "seed": seed,
        "training_samples": int(len(observations)),
        "optimizer_updates": epochs * int(np.ceil(len(x) / batch_size)),
        "curve": curve,
        "convergence": convergence,
        "selected_validation": selected_validation,
        "selected_model_path": relative(selected_path),
        "selected_model_sha256": sha256(selected_path),
        "teacher_cost_cny": float(teacher_validation["mean"]["total_cost_cny"]),
        "render_calls_during_training": training_env.render_calls,
    }
    write_json(seed_dir / "result.json", result)
    training_env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal Shore+BESS V3 safe-policy training")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    config = load_config()
    seeds = (
        [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [int(value) for value in config["training"]["seeds"]]
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "shore-bess-v3-safe-%Y%m%dT%H%M%SZ"
    )
    run_dir = OUTPUT_ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"append-only run already exists: {run_id}")
    run_dir.mkdir(parents=True)

    dataset = load_public_dataset(config)
    train_slice, validation_slice, blind_slice = chronological_slices(dataset)
    description = dataset.describe(test_ratio=0.20, validation_ratio=0.10)
    if description["quality"]["training_eligible"] is not True:
        raise RuntimeError("dataset quality gate failed")
    episode_steps = int(config["training"]["episode_hours"])
    validation_starts = fixed_window_starts(
        validation_slice.stop - validation_slice.start,
        episode_steps,
        int(config["convergence_gate"]["fixed_validation_windows"]),
    )
    blind_starts = fixed_window_starts(
        blind_slice.stop - blind_slice.start,
        episode_steps,
        max(12, (blind_slice.stop - blind_slice.start) // episode_steps),
    )
    validation_factory = make_eval_factory(
        dataset, validation_slice, config, train_slice, seeds[0]
    )
    blind_factory = make_eval_factory(dataset, blind_slice, config, train_slice, seeds[0])
    validation_teacher = evaluate_windows(
        validation_factory, rule_peak_valley_policy, validation_starts
    )
    validation_neutral = evaluate_windows(validation_factory, neutral_policy, validation_starts)
    blind_teacher = evaluate_windows(blind_factory, rule_peak_valley_policy, blind_starts)
    blind_neutral = evaluate_windows(blind_factory, neutral_policy, blind_starts)

    legacy_history = (
        ROOT
        / "app"
        / "services"
        / "rl_model"
        / "shore_bess"
        / "artifacts"
        / "shore_bess_outputs.jsonl"
    )
    legacy_policy = ROOT / "app" / "services" / "rl_model" / "shore_bess" / "policy.bin"
    before = {"history": sha256(legacy_history), "policy": sha256(legacy_policy)}
    manifest = {
        "schema": "port-dt-shore-bess-safe-policy-manifest.v1",
        "version": config["version"],
        "run_id": run_id,
        "started_at": utc_now(),
        "algorithm": config["training"]["algorithm"],
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.fingerprint,
        "split": {
            "train_rows": train_slice.stop - train_slice.start,
            "validation_rows": validation_slice.stop - validation_slice.start,
            "blind_test_rows": blind_slice.stop - blind_slice.start,
            "method": description["split_method"],
        },
        "seeds": seeds,
        "training_rendering": False,
        "contract": CONTRACT.as_dict(),
        "legacy_evidence_sha256_before": before,
    }
    write_json(run_dir / "manifest.json", manifest)

    seed_results = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        seed_results.append(
            train_one_seed(
                seed=seed,
                seed_dir=seed_dir,
                dataset=dataset,
                train_slice=train_slice,
                validation_slice=validation_slice,
                config=config,
                validation_starts=validation_starts,
                teacher_validation=validation_teacher,
            )
        )

    blind_results = []
    for seed_result in seed_results:
        model = PPO.load(
            str(ROOT / seed_result["selected_model_path"]), device="cpu"
        )
        trace_factory = make_eval_factory(
            dataset,
            blind_slice,
            config,
            train_slice,
            int(seed_result["seed"]),
            trace=True,
        )
        evaluation = evaluate_windows(
            trace_factory, model_policy(model), blind_starts
        )
        blind_results.append({"seed": seed_result["seed"], **evaluation})

    metric_names = sorted(
        set.intersection(*(set(row["mean"]) for row in blind_results))
    )
    aggregate = {
        name: float(np.mean([row["mean"][name] for row in blind_results]))
        for name in metric_names
    }
    aggregate_std = {
        name: float(np.std([row["mean"][name] for row in blind_results]))
        for name in metric_names
    }
    cost_vs_neutral = 100.0 * (
        blind_neutral["mean"]["total_cost_cny"] - aggregate["total_cost_cny"]
    ) / blind_neutral["mean"]["total_cost_cny"]
    cost_vs_teacher = 100.0 * (
        blind_teacher["mean"]["total_cost_cny"] - aggregate["total_cost_cny"]
    ) / blind_teacher["mean"]["total_cost_cny"]
    carbon_vs_neutral = 100.0 * (
        blind_neutral["mean"]["carbon_kg"] - aggregate["carbon_kg"]
    ) / blind_neutral["mean"]["carbon_kg"]
    peak_vs_neutral = 100.0 * (
        blind_neutral["mean"]["peak_kw"] - aggregate["peak_kw"]
    ) / blind_neutral["mean"]["peak_kw"]
    annual_weeks = 365.25 / 7.0
    annual_savings_neutral = (
        blind_neutral["mean"]["total_cost_cny"] - aggregate["total_cost_cny"]
    ) * annual_weeks
    annual_carbon_reduction_t = (
        blind_neutral["mean"]["carbon_kg"] - aggregate["carbon_kg"]
    ) * annual_weeks / 1000.0
    convergence_pass_rate = float(
        np.mean([float(row["convergence"]["passed"]) for row in seed_results])
    )
    convergence_passed = bool(
        convergence_pass_rate + 1e-9
        >= float(config["convergence_gate"]["minimum_seed_pass_rate"])
    )
    safety_passed = bool(
        aggregate["guardrail_violation_rate"] <= 1e-12
        and aggregate["terminal_soc_error"] <= 1e-6
        and aggregate["terminal_flex_backlog_kwh"] <= 1e-6
        and aggregate["shore_sla_violation_kwh"] <= 1e-9
    )
    business_passed = bool(cost_vs_neutral > 0.0 and peak_vs_neutral >= 0.0)
    carbon_guardrail_passed = bool(carbon_vs_neutral >= 0.0)
    economic_profile_passed = convergence_passed and safety_passed and business_passed

    max_curve = max(len(row["curve"]) for row in seed_results)
    aggregate_curve = []
    for index in range(max_curve):
        rows = [row["curve"][index] for row in seed_results if index < len(row["curve"])]
        aggregate_curve.append(
            {
                "epoch": int(rows[0]["epoch"]),
                "optimizer_updates": int(rows[0]["optimizer_updates"]),
                "imitation_loss_mean": float(np.mean([row["imitation_loss"] for row in rows])),
                "imitation_loss_std": float(np.std([row["imitation_loss"] for row in rows])),
                "validation_cost_mean_cny": float(
                    np.mean([row["validation_total_cost_cny"] for row in rows])
                ),
                "validation_cost_std_cny": float(
                    np.std([row["validation_total_cost_cny"] for row in rows])
                ),
                "validation_peak_mean_kw": float(
                    np.mean([row["validation_peak_kw"] for row in rows])
                ),
            }
        )

    report = {
        "schema": "port-dt-shore-bess-formal-evidence.v2",
        "version": config["version"],
        "run_id": run_id,
        "generated_at": utc_now(),
        "status": (
            "FORMAL_PUBLIC_DATA_ECONOMIC_PROFILE_PASS"
            if economic_profile_passed and carbon_guardrail_passed
            else "FORMAL_PUBLIC_DATA_ECONOMIC_PROFILE_PASS_CARBON_BLOCKED"
            if economic_profile_passed
            else "FORMAL_PUBLIC_DATA_OFFLINE_GATE_FAILED"
        ),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "sha256": dataset.fingerprint,
            "rows": dataset.rows,
            "train_rows": train_slice.stop - train_slice.start,
            "validation_rows": validation_slice.stop - validation_slice.start,
            "blind_test_rows": blind_slice.stop - blind_slice.start,
            "split_method": description["split_method"],
            "quality_status": description["quality"]["status"],
            "evidence_tier": dataset.metadata.get("evidence_tier"),
            "official_aggregate_columns": dataset.metadata.get(
                "official_aggregate_columns"
            ),
            "public_reanalysis_columns": dataset.metadata.get(
                "public_reanalysis_columns"
            ),
            "derived_columns": dataset.metadata.get("derived_columns"),
            "unavailable_factors": dataset.metadata.get("unavailable_factors"),
        },
        "training": {
            "algorithm": config["training"]["algorithm"],
            "actor_runtime": "stable_baselines3.PPO MLP actor used for deterministic inference",
            "teacher": "constraint-projected peak/valley controller",
            "safe_policy_improvement": "validation-selected checkpoint; RL fine-tune is rejected unless it beats the admitted actor under identical gates",
            "seeds": seeds,
            "training_samples_per_seed": seed_results[0]["training_samples"],
            "optimizer_updates_per_seed": seed_results[0]["optimizer_updates"],
            "total_optimizer_updates": sum(row["optimizer_updates"] for row in seed_results),
            "episode_hours": episode_steps,
            "training_render_calls": sum(
                row["render_calls_during_training"] for row in seed_results
            ),
        },
        "contract": CONTRACT.as_dict(),
        "convergence": {
            "passed": convergence_passed,
            "seed_pass_rate": convergence_pass_rate,
            "minimum_seed_pass_rate": float(
                config["convergence_gate"]["minimum_seed_pass_rate"]
            ),
            "aggregate_curve": aggregate_curve,
            "per_seed": [
                {"seed": row["seed"], **row["convergence"]}
                for row in seed_results
            ],
            "legacy_curve_diagnosis": [
                "legacy page plotted one-trajectory raw components rather than fixed validation checkpoints",
                "legacy offline set contained 145 transitions with 145 zero actions",
                "legacy bias/perturb display fields remain preserved but are excluded from V3.1 admission",
            ],
        },
        "blind_test": {
            "windows": len(blind_starts),
            "window_hours": episode_steps,
            "selected_actor_multi_seed_mean": aggregate,
            "selected_actor_multi_seed_std": aggregate_std,
            "per_seed": [
                {"seed": row["seed"], "mean": row["mean"], "std": row["std"]}
                for row in blind_results
            ],
            "rule_teacher": blind_teacher,
            "no_bess": blind_neutral,
            "sample_real_model_inference": blind_results[0]["sample_inference"],
        },
        "business_metrics": {
            "cost_reduction_vs_no_bess_percent": cost_vs_neutral,
            "cost_reduction_vs_rule_percent": cost_vs_teacher,
            "carbon_reduction_vs_no_bess_percent": carbon_vs_neutral,
            "peak_reduction_vs_no_bess_percent": peak_vs_neutral,
            "mean_weekly_cost_actor_cny": aggregate["total_cost_cny"],
            "mean_weekly_cost_no_bess_cny": blind_neutral["mean"]["total_cost_cny"],
            "mean_weekly_cost_rule_cny": blind_teacher["mean"]["total_cost_cny"],
            "annualized_scenario_savings_vs_no_bess_cny": annual_savings_neutral,
            "annualized_scenario_carbon_reduction_t": annual_carbon_reduction_t,
            "annualized_scenario_carbon_change_t": -annual_carbon_reduction_t,
            "mean_weekly_carbon_actor_kg": aggregate["carbon_kg"],
            "mean_weekly_carbon_no_bess_kg": blind_neutral["mean"]["carbon_kg"],
            "mean_weekly_peak_actor_kw": aggregate["peak_kw"],
            "mean_weekly_peak_no_bess_kw": blind_neutral["mean"]["peak_kw"],
            "claim_eligible": False,
            "evidence_scope": "paired multi-seed public-data engineering scenario; annualization is descriptive, not a measured Shanghai saving",
        },
        "quality_gates": {
            "convergence_passed": convergence_passed,
            "business_advantage_passed": business_passed,
            "carbon_guardrail_passed": carbon_guardrail_passed,
            "safety_passed": safety_passed,
            "dataset_training_eligible": True,
            "nonzero_action_support": aggregate["nonzero_bess_action_rate"] > 0.05,
            "guardrail_violation_rate": aggregate["guardrail_violation_rate"],
            "terminal_soc_error": aggregate["terminal_soc_error"],
            "terminal_flex_backlog_kwh": aggregate["terminal_flex_backlog_kwh"],
            "shore_sla_violation_kwh": aggregate["shore_sla_violation_kwh"],
            "public_offline_admitted": economic_profile_passed,
            "profile_admission": {
                "economic_cost_peak": economic_profile_passed,
                "carbon_reduction": carbon_guardrail_passed,
                "balanced_cost_carbon": bool(
                    economic_profile_passed and carbon_guardrail_passed
                ),
            },
            "production_admitted": False,
        },
        "algorithm_registry": [
            {
                "name": "Legacy offline SAC",
                "state": "historical_rejected",
                "reason": "145/145 zero-action support and berth/load schema drift",
            },
            {
                "name": "Constrained BC actor",
                "state": "public_offline_candidate",
                "reason": "three-seed fixed-validation selection and chronological blind test",
            },
            {
                "name": "PPO fine-tune",
                "state": "admission_gated",
                "reason": "must beat selected actor without increasing projection or safety violations",
            },
            {
                "name": "Peak/valley rule teacher",
                "state": "active_comparator",
                "reason": "constraint-projected operational fallback",
            },
            {
                "name": "Safety projection + SPI selector",
                "state": "active",
                "reason": "fail-closed action and checkpoint admission",
            },
        ],
        "data_adequacy": {
            "offline_training": "sufficient_for_public_engineering_benchmark",
            "production_dispatch": "insufficient_until_authorized_site_replacement",
            "public_independent_source_observations": dataset.metadata.get(
                "independent_source_observations"
            ),
            "engineering_simulator_rows_preserved": {
                "shore_power_15min_by_berth": 69120,
                "bess_15min": 5760,
                "grid_meter_5min": 17280,
                "ship_calls": 285,
            },
            "missing_for_site": CONTRACT.landing_inputs,
        },
        "artifacts": {
            "manifest": relative(run_dir / "manifest.json"),
            "models": [
                {
                    "seed": row["seed"],
                    "path": row["selected_model_path"],
                    "sha256": row["selected_model_sha256"],
                }
                for row in seed_results
            ],
            "legacy_history_preserved": True,
            "legacy_history_sha256_after": sha256(legacy_history),
            "legacy_policy_sha256_after": sha256(legacy_policy),
        },
        "claim_boundary": config["claim_boundary"],
        "production_authority": False,
        "site_status": "待接入港口",
    }
    if report["artifacts"]["legacy_history_sha256_after"] != before["history"]:
        raise RuntimeError("legacy history changed during formal training")
    if report["artifacts"]["legacy_policy_sha256_after"] != before["policy"]:
        raise RuntimeError("legacy policy changed during formal training")

    report_path = run_dir / "report.json"
    write_json(report_path, report)
    latest = {
        "schema": "port-dt-shore-bess-latest-pointer.v1",
        "run_id": run_id,
        "report_path": relative(report_path),
        "report_sha256": sha256(report_path),
        "status": report["status"],
        "updated_at": utc_now(),
    }
    write_json(OUTPUT_ROOT / "latest.json", latest)
    append_jsonl(
        OUTPUT_ROOT / "history_index.jsonl",
        {
            **latest,
            "legacy_history_sha256": before["history"],
            "legacy_policy_sha256": before["policy"],
        },
    )
    print(json.dumps(latest, ensure_ascii=False, indent=2))
    if not economic_profile_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
