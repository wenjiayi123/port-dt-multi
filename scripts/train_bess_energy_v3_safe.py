"""Formal three-seed event-aware site BESS training and blind evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.services.rl_model.bess_energy.v3_environment import (
    CONTRACT,
    BESSEnergyV3Env,
    balanced_event_policy,
    chronological_slices,
    evaluate_windows,
    fixed_window_starts,
    load_config,
    load_public_dataset,
    neutral_policy,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evidence" / "v3" / "bess_energy"


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


def make_factory(dataset, split, config, train_slice, seed, *, trace=False):
    return lambda: BESSEnergyV3Env(
        dataset, split, config=config, normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_hours"]), seed=seed,
        training=False, record_trace=trace,
    )


def collect_teacher_data(dataset, train_slice, config, seed) -> Tuple[np.ndarray, np.ndarray]:
    env = BESSEnergyV3Env(
        dataset, train_slice, config=config, normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_hours"]), seed=seed,
        training=True, record_trace=False,
    )
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    for episode in range(int(config["training"]["behavior_trajectories_per_seed"])):
        observation, _ = env.reset(seed=seed * 1000 + episode)
        done = False
        while not done:
            action = balanced_event_policy(observation, env)
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
    if env.render_calls:
        raise RuntimeError("training environment rendered unexpectedly")
    env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def train_one_seed(*, seed, seed_dir, dataset, train_slice, validation_slice, config, validation_starts, validation_baseline=None):
    import torch
    from stable_baselines3 import PPO

    torch.manual_seed(seed)
    training_env = BESSEnergyV3Env(
        dataset, train_slice, config=config, normalization_slice=train_slice,
        episode_steps=int(config["training"]["episode_hours"]), seed=seed,
        training=True, record_trace=False,
    )
    model = PPO(
        "MlpPolicy", training_env,
        learning_rate=float(config["training"]["behavior_learning_rate"]),
        n_steps=int(config["training"]["episode_hours"]) * 2,
        batch_size=84, n_epochs=5, gamma=float(config["training"]["gamma"]),
        gae_lambda=0.95, ent_coef=0.0,
        policy_kwargs={"net_arch": list(config["training"]["network"])},
        seed=seed, verbose=0, device="cpu",
    )
    observations, actions = collect_teacher_data(dataset, train_slice, config, seed)
    x = torch.as_tensor(observations, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=float(config["training"]["behavior_learning_rate"]))
    rng = np.random.default_rng(seed)
    validation_factory = make_factory(dataset, validation_slice, config, train_slice, seed)
    batch_size = int(config["training"]["behavior_batch_size"])
    epochs = int(config["training"]["behavior_epochs"])
    every = int(config["training"]["behavior_validation_every_epochs"])
    curve: List[Dict[str, float]] = []
    best = None
    best_safety = None

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(x))
        losses = []
        for start in range(0, len(x), batch_size):
            index = torch.as_tensor(indices[start : start + batch_size])
            predicted = model.policy.get_distribution(x[index]).distribution.mean
            loss = torch.mean((predicted - y[index]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses))
        if epoch == 1 or epoch % every == 0 or epoch == epochs:
            validation = evaluate_windows(validation_factory, model_policy(model), validation_starts)
            record = {
                "epoch": epoch, "optimizer_updates": epoch * int(np.ceil(len(x) / batch_size)),
                "imitation_loss": epoch_loss,
                "validation_reward_mean": validation["mean"]["reward"],
                "validation_total_cost_cny": validation["mean"]["total_cost_cny"],
                "validation_peak_kw": validation["mean"]["peak_kw"],
                "validation_carbon_kg": validation["mean"]["carbon_kg"],
                "validation_event_compliance_rate": validation["mean"]["event_compliance_rate"],
                "validation_projection_rate": validation["mean"]["projection_rate"],
                "validation_guardrail_violation_rate": validation["mean"]["guardrail_violation_rate"],
                "validation_terminal_soc_error": validation["mean"]["terminal_soc_error"],
            }
            curve.append(record)
            append_jsonl(seed_dir / "metrics.jsonl", {"ts": utc_now(), "seed": seed, **record})
            checkpoint = seed_dir / f"checkpoint_epoch_{epoch}.zip"
            model.save(str(checkpoint.with_suffix("")))
            safety_ok = (
                record["validation_guardrail_violation_rate"] <= 1e-12
                and record["validation_terminal_soc_error"] <= 1e-6
                and record["validation_event_compliance_rate"] >= 1.0 - 1e-9
            )
            balanced_selection = str(config.get("checkpoint_selection") or "cost_only") == "cost_carbon_peak_non_regression"
            business_ok = bool(
                not balanced_selection
                or (
                    validation_baseline is not None
                    and record["validation_total_cost_cny"] <= validation_baseline["total_cost_cny"]
                    and record["validation_peak_kw"] <= validation_baseline["peak_kw"]
                    and record["validation_carbon_kg"] <= validation_baseline["carbon_kg"]
                )
            )
            if safety_ok and (
                best_safety is None
                or record["validation_total_cost_cny"] < best_safety["record"]["validation_total_cost_cny"]
            ):
                best_safety = {"record": record, "checkpoint": checkpoint}
            if safety_ok and business_ok and (best is None or record["validation_total_cost_cny"] < best["record"]["validation_total_cost_cny"]):
                best = {"record": record, "checkpoint": checkpoint}
            print(f"seed={seed} epoch={epoch}/{epochs} loss={epoch_loss:.7f} validation_cost={record['validation_total_cost_cny']:.2f}", flush=True)

    selection_gate_passed = best is not None
    if best is None:
        best = best_safety
    if best is None:
        raise RuntimeError(f"seed {seed} has no safety-admissible checkpoint")
    selected_path = seed_dir / "selected_model.zip"
    shutil.copy2(best["checkpoint"], selected_path)
    selected = PPO.load(str(selected_path), device="cpu")
    selected_validation = evaluate_windows(validation_factory, model_policy(selected), validation_starts)
    tail = curve[-min(3, len(curve)):]
    loss_reduction = 1.0 - curve[-1]["imitation_loss"] / max(curve[0]["imitation_loss"], 1e-12)
    tail_costs = np.asarray([row["validation_total_cost_cny"] for row in tail], dtype=np.float64)
    tail_range = float((np.max(tail_costs) - np.min(tail_costs)) / max(1.0, abs(np.mean(tail_costs))))
    convergence = {
        "passed": bool(loss_reduction >= float(config["convergence_gate"]["loss_reduction_min"]) and tail_range <= float(config["convergence_gate"]["tail_relative_range_max"])),
        "criterion": "imitation_loss_reduction_plus_fixed_validation_business_plateau",
        "loss_reduction_ratio": loss_reduction,
        "loss_reduction_min": float(config["convergence_gate"]["loss_reduction_min"]),
        "tail_validation_cost_relative_range": tail_range,
        "tail_validation_cost_relative_range_max": float(config["convergence_gate"]["tail_relative_range_max"]),
        "selected_epoch": int(best["record"]["epoch"]),
        "selection_metric": (
            "minimum_fixed_validation_total_cost_subject_to_cost_carbon_peak_non_regression_and_safety"
            if str(config.get("checkpoint_selection") or "cost_only") == "cost_carbon_peak_non_regression"
            else "minimum_fixed_validation_total_cost_subject_to_zero_guardrail_and_full_event_compliance"
        ),
        "blind_test_used_for_selection": False,
    }
    result = {
        "seed": seed, "training_samples": int(len(observations)),
        "optimizer_updates": epochs * int(np.ceil(len(x) / batch_size)),
        "curve": curve, "convergence": convergence, "selected_validation": selected_validation,
        "selection_gate_passed": selection_gate_passed,
        "selected_model_path": relative(selected_path), "selected_model_sha256": sha256(selected_path),
        "render_calls_during_training": training_env.render_calls,
    }
    write_json(seed_dir / "result.json", result)
    training_env.close()
    return result


def no_bess_baseline(neutral: Dict[str, Any]) -> Dict[str, Any]:
    baseline = json.loads(json.dumps(neutral))
    mean = baseline["mean"]
    mean["reserve_revenue_cny"] = 0.0
    mean["dr_revenue_cny"] = 0.0
    mean["total_cost_cny"] = mean["energy_cost_cny"] + mean["degradation_cost_cny"] + mean["demand_charge_cny"]
    baseline["definition"] = "same public load without BESS dispatch, reserve or DR revenue"
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal site BESS V3.1 safe-policy training")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--config", default="config/bess_energy_v3.json")
    parser.add_argument("--candidate-only", action="store_true")
    args = parser.parse_args()
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    config_path = (ROOT / args.config).resolve()
    if ROOT not in config_path.parents:
        raise ValueError("config must stay inside the repository")
    config = load_config(config_path)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()] if args.seeds else [int(x) for x in config["training"]["seeds"]]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("bess-energy-v3-safe-%Y%m%dT%H%M%SZ")
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
    validation_starts = fixed_window_starts(validation_slice.stop - validation_slice.start, episode_steps, int(config["convergence_gate"]["fixed_validation_windows"]))
    blind_starts = fixed_window_starts(blind_slice.stop - blind_slice.start, episode_steps, max(12, (blind_slice.stop - blind_slice.start) // episode_steps))
    validation_factory = make_factory(dataset, validation_slice, config, train_slice, seeds[0])
    blind_factory = make_factory(dataset, blind_slice, config, train_slice, seeds[0])
    validation_teacher = evaluate_windows(validation_factory, balanced_event_policy, validation_starts)
    validation_no_bess = no_bess_baseline(evaluate_windows(validation_factory, neutral_policy, validation_starts))
    blind_teacher = evaluate_windows(blind_factory, balanced_event_policy, blind_starts)
    blind_no_bess = no_bess_baseline(evaluate_windows(blind_factory, neutral_policy, blind_starts))

    legacy_root = ROOT / "app" / "services" / "rl_model" / "bess_energy"
    legacy = {name: sha256(legacy_root / name) for name in ("policy_evaluate_history.jsonl", "offline_dataset.jsonl", "policy.bin", "policy_meta.json")}
    manifest = {
        "schema": "port-dt-bess-energy-safe-policy-manifest.v1", "version": config["version"],
        "run_id": run_id, "started_at": utc_now(), "algorithm": config["training"]["algorithm"],
        "dataset_id": dataset.dataset_id, "dataset_sha256": dataset.fingerprint,
        "event_provenance": config["services"]["event_provenance"],
        "split": {"train_rows": train_slice.stop - train_slice.start, "validation_rows": validation_slice.stop - validation_slice.start,
                  "blind_test_rows": blind_slice.stop - blind_slice.start, "method": description["split_method"]},
        "seeds": seeds, "training_rendering": False, "contract": CONTRACT.as_dict(), "legacy_sha256_before": legacy,
        "config_path": relative(config_path), "config_sha256": sha256(config_path),
        "candidate_only": bool(args.candidate_only),
    }
    write_json(run_dir / "manifest.json", manifest)
    seed_results = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        seed_results.append(train_one_seed(seed=seed, seed_dir=seed_dir, dataset=dataset, train_slice=train_slice,
                                           validation_slice=validation_slice, config=config, validation_starts=validation_starts,
                                           validation_baseline=validation_no_bess["mean"]))

    blind_results = []
    for seed_result in seed_results:
        model = PPO.load(str(ROOT / seed_result["selected_model_path"]), device="cpu")
        evaluation = evaluate_windows(make_factory(dataset, blind_slice, config, train_slice, int(seed_result["seed"]), trace=True), model_policy(model), blind_starts)
        blind_results.append({"seed": seed_result["seed"], **evaluation})
    metric_names = sorted(set.intersection(*(set(row["mean"]) for row in blind_results)))
    aggregate = {name: float(np.mean([row["mean"][name] for row in blind_results])) for name in metric_names}
    aggregate_std = {name: float(np.std([row["mean"][name] for row in blind_results])) for name in metric_names}
    no_bess = blind_no_bess["mean"]
    teacher = blind_teacher["mean"]
    cost_vs_no_bess = 100.0 * (no_bess["total_cost_cny"] - aggregate["total_cost_cny"]) / no_bess["total_cost_cny"]
    cost_vs_teacher = 100.0 * (teacher["total_cost_cny"] - aggregate["total_cost_cny"]) / teacher["total_cost_cny"]
    carbon_vs_no_bess = 100.0 * (no_bess["carbon_kg"] - aggregate["carbon_kg"]) / no_bess["carbon_kg"]
    peak_vs_no_bess = 100.0 * (no_bess["peak_kw"] - aggregate["peak_kw"]) / no_bess["peak_kw"]
    annual_weeks = 365.25 / 7.0
    annual_savings = (no_bess["total_cost_cny"] - aggregate["total_cost_cny"]) * annual_weeks
    annual_carbon_t = (no_bess["carbon_kg"] - aggregate["carbon_kg"]) * annual_weeks / 1000.0
    convergence_rate = float(np.mean([float(row["convergence"]["passed"]) for row in seed_results]))
    convergence_passed = convergence_rate + 1e-9 >= float(config["convergence_gate"]["minimum_seed_pass_rate"])
    safety_passed = aggregate["guardrail_violation_rate"] <= 1e-12 and aggregate["terminal_soc_error"] <= 1e-6 and aggregate["event_compliance_rate"] >= 1.0 - 1e-9
    business_passed = cost_vs_no_bess > 0.0 and peak_vs_no_bess >= 0.0
    carbon_passed = carbon_vs_no_bess >= 0.0
    public_offline_admitted = bool(convergence_passed and safety_passed and business_passed)
    max_curve = max(len(row["curve"]) for row in seed_results)
    aggregate_curve = []
    for index in range(max_curve):
        rows = [row["curve"][index] for row in seed_results if index < len(row["curve"])]
        aggregate_curve.append({
            "epoch": int(rows[0]["epoch"]), "optimizer_updates": int(rows[0]["optimizer_updates"]),
            "imitation_loss_mean": float(np.mean([row["imitation_loss"] for row in rows])),
            "imitation_loss_std": float(np.std([row["imitation_loss"] for row in rows])),
            "validation_cost_mean_cny": float(np.mean([row["validation_total_cost_cny"] for row in rows])),
            "validation_cost_std_cny": float(np.std([row["validation_total_cost_cny"] for row in rows])),
            "validation_peak_mean_kw": float(np.mean([row["validation_peak_kw"] for row in rows])),
            "validation_event_compliance_mean": float(np.mean([row["validation_event_compliance_rate"] for row in rows])),
        })
    report = {
        "schema": "port-dt-bess-energy-formal-evidence.v1", "version": config["version"], "run_id": run_id,
        "generated_at": utc_now(),
        "status": "FORMAL_PUBLIC_DATA_BALANCED_PROFILE_PASS" if public_offline_admitted and carbon_passed else (
            "FORMAL_PUBLIC_DATA_ECONOMIC_PROFILE_PASS_CARBON_BLOCKED" if public_offline_admitted else "FORMAL_PUBLIC_DATA_OFFLINE_GATE_FAILED"),
        "dataset": {"dataset_id": dataset.dataset_id, "sha256": dataset.fingerprint, "rows": dataset.rows,
                    "train_rows": train_slice.stop - train_slice.start, "validation_rows": validation_slice.stop - validation_slice.start,
                    "blind_test_rows": blind_slice.stop - blind_slice.start, "split_method": description["split_method"],
                    "quality_status": description["quality"]["status"], "evidence_tier": dataset.metadata.get("evidence_tier")},
        "scenario_supplement": {"reserve_and_dr_events": config["services"]["event_provenance"],
                                "observed_site_event_rows": 0, "claim_as_real_market_settlement": False,
                                "replacement_required": "authorized DR/reserve clearing and performance ledger"},
        "training": {"algorithm": config["training"]["algorithm"], "actor_runtime": "stable_baselines3.PPO MLP actor deterministic inference",
                     "teacher": "constraint-projected carbon/peak/event-aware controller", "seeds": seeds,
                     "training_samples_per_seed": seed_results[0]["training_samples"],
                     "optimizer_updates_per_seed": seed_results[0]["optimizer_updates"],
                     "total_optimizer_updates": sum(row["optimizer_updates"] for row in seed_results),
                     "episode_hours": episode_steps, "training_render_calls": sum(row["render_calls_during_training"] for row in seed_results)},
        "contract": CONTRACT.as_dict(),
        "convergence": {"passed": convergence_passed, "seed_pass_rate": convergence_rate,
                        "minimum_seed_pass_rate": float(config["convergence_gate"]["minimum_seed_pass_rate"]),
                        "aggregate_curve": aggregate_curve,
                        "per_seed": [{"seed": row["seed"], **row["convergence"]} for row in seed_results],
                        "legacy_curve_diagnosis": ["legacy training curve mixed display bias/anchor with raw components",
                                                   "legacy offline reserve action dR had zero support and event_active coverage was zero",
                                                   "legacy saved actor saturates on sampled states and had no chronological held-out evaluation"]},
        "blind_test": {"windows": len(blind_starts), "window_hours": episode_steps,
                       "selected_actor_multi_seed_mean": aggregate, "selected_actor_multi_seed_std": aggregate_std,
                       "per_seed": [{"seed": row["seed"], "mean": row["mean"], "std": row["std"]} for row in blind_results],
                       "rule_teacher": blind_teacher, "no_bess": blind_no_bess,
                       "sample_real_model_inference": blind_results[0]["sample_inference"]},
        "business_metrics": {"cost_reduction_vs_no_bess_percent": cost_vs_no_bess,
                             "cost_reduction_vs_rule_percent": cost_vs_teacher,
                             "carbon_reduction_vs_no_bess_percent": carbon_vs_no_bess,
                             "peak_reduction_vs_no_bess_percent": peak_vs_no_bess,
                             "event_compliance_rate_percent": aggregate["event_compliance_rate"] * 100.0,
                             "annualized_scenario_savings_vs_no_bess_cny": annual_savings,
                             "annualized_scenario_carbon_reduction_t": annual_carbon_t,
                             "mean_weekly_total_cost_actor_cny": aggregate["total_cost_cny"],
                             "mean_weekly_total_cost_no_bess_cny": no_bess["total_cost_cny"],
                             "mean_weekly_reserve_revenue_actor_cny": aggregate["reserve_revenue_cny"],
                             "mean_weekly_dr_revenue_actor_cny": aggregate["dr_revenue_cny"],
                             "claim_eligible": False,
                             "evidence_scope": "paired public-data engineering scenario; event and annualization figures are not Shanghai settlement records"},
        "quality_gates": {"convergence_passed": convergence_passed, "business_advantage_passed": business_passed,
                          "carbon_guardrail_passed": carbon_passed, "safety_passed": safety_passed,
                          "dataset_training_eligible": True, "dispatch_action_support": aggregate["nonzero_dispatch_action_rate"] > 0.05,
                          "reserve_action_support": aggregate["nonzero_reserve_action_rate"] > 0.05,
                          "event_coverage_hours": aggregate["event_hours"], "event_compliance_rate": aggregate["event_compliance_rate"],
                          "guardrail_violation_rate": aggregate["guardrail_violation_rate"],
                          "terminal_soc_error": aggregate["terminal_soc_error"],
                          "public_offline_admitted": public_offline_admitted, "production_admitted": False},
        "algorithm_registry": [
            {"name": "Legacy SAC + linear twin critics", "state": "historical_rejected", "reason": "zero reserve/event coverage and saturated actor"},
            {"name": "Constrained event-aware BC actor", "state": "public_offline_candidate", "reason": "three seeds, validation selection, blind test"},
            {"name": "PPO safe fine-tune", "state": "admission_gated", "reason": "must improve selected actor under identical gates"},
            {"name": "Economic/carbon/event rule teacher", "state": "active_comparator", "reason": "constraint-projected fallback"},
            {"name": "CMDP safety projection", "state": "active", "reason": "SOC/PCS/PCC/N-1/thermal/event/terminal constraints"},
            {"name": "PCS/BMS write gateway", "state": "contract_ready", "reason": "shadow only until nonce/TTL/ack/rollback site acceptance"}],
        "artifacts": {"manifest": relative(run_dir / "manifest.json"),
                      "models": [{"seed": row["seed"], "path": row["selected_model_path"], "sha256": row["selected_model_sha256"]} for row in seed_results],
                      "legacy_preserved": True, "legacy_sha256_after": {name: sha256(legacy_root / name) for name in legacy}},
        "claim_boundary": config["claim_boundary"], "production_authority": False, "site_status": "待接入港口",
    }
    if report["artifacts"]["legacy_sha256_after"] != legacy:
        raise RuntimeError("legacy BESS history or policy changed during formal training")
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    latest = {"schema": "port-dt-bess-energy-latest-pointer.v1", "run_id": run_id,
              "report_path": relative(report_path), "report_sha256": sha256(report_path),
              "status": report["status"], "updated_at": utc_now()}
    history_entry = {
        **latest, "legacy_sha256": legacy,
        "promotion_state": "candidate_not_promoted" if args.candidate_only else "promoted",
    }
    if not args.candidate_only:
        write_json(OUTPUT_ROOT / "latest.json", latest)
    append_jsonl(OUTPUT_ROOT / "history_index.jsonl", history_entry)
    print(json.dumps(latest, ensure_ascii=False, indent=2))
    if not public_offline_admitted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
