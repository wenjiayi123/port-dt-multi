"""Admit three immutable pilot training runs without training their models again.

All checkpoint choices are recomputed on the complete validation months. Only
after every seed clears full-validation and tail-stability gates is selection
frozen and the repeated test/forward benchmark opened. January/February use
their shared official source period. Existing pilot evidence remains intact.

Usage: python -m scripts.admit_shore_bess_v8 --run-ids RUN1,RUN2,RUN3
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts import train_shore_bess_v8 as training

ROOT = training.ROOT
OUTPUT_ROOT = training.OUTPUT_ROOT
INTEGRITY_CHECKS = (
    "real_optimizer_updates", "weights_changed", "source_unchanged_during_training",
    "config_and_dataset_files_unchanged", "historical_pointers_preserved", "no_training_rendering",
)


def checked_path(relative_path: str, *, base: Path = ROOT, scope: Path | None = None) -> Path:
    if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
        raise ValueError("evidence paths must be repository-relative")
    path = (base / relative_path).resolve()
    if not path.is_relative_to(base.resolve()) or (scope is not None and not path.is_relative_to(scope.resolve())):
        raise ValueError("evidence path escapes its permitted scope")
    if not path.is_file():
        raise ValueError(f"evidence file missing: {relative_path}")
    return path


def verify_references(references: dict[str, str], *, base: Path = ROOT, scope: Path | None = None) -> None:
    if not references:
        raise ValueError("empty evidence hash registry")
    for relative_path, expected in references.items():
        path = checked_path(relative_path, base=base, scope=scope)
        if training.file_sha256(path) != expected:
            raise ValueError(f"evidence SHA-256 mismatch: {relative_path}")


def protocol_key(report: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(report["config"])
    config.pop("seeds")
    return {"config": config, "source_sha256": report["manifest"]["source_sha256"],
            "input_files_sha256": report["input_files_sha256"],
            "versions": report["manifest"]["versions"],
            "train": report["manifest"]["train"], "validation": report["manifest"]["validation"]}


def assert_compatible(reports: list[dict[str, Any]]) -> None:
    if len(reports) != 3:
        raise ValueError("admission requires exactly three independent pilot runs")
    seeds = []
    for report in reports:
        if report.get("status") != "PILOT_VALIDATION_ONLY" or report["config"].get("pilot") is not True:
            raise ValueError("only completed validation-only pilot runs can be adopted")
        if report.get("evaluations") or report["manifest"].get("test_status") != "not_opened" or report["manifest"].get("forward_status") != "not_loaded":
            raise ValueError("source pilot accessed held-out evaluation")
        if report["manifest"].get("teacher_actions_used", True) or report["manifest"].get("warm_start_used", True):
            raise ValueError("source training must start from random initialization without teacher actions")
        if len(report["results"]) != 1 or len(report["config"]["seeds"]) != 1:
            raise ValueError("each input must contain exactly one seed")
        seed = int(report["results"][0]["seed"])
        if report["config"]["seeds"] != [seed]:
            raise ValueError("source seed does not match its training configuration")
        if not all(report["checks"].get(key) is True for key in INTEGRITY_CHECKS):
            raise ValueError("source training integrity gate failed")
        seeds.append(seed)
    if len(set(seeds)) != 3:
        raise ValueError("source runs must use three distinct training seeds")
    first = protocol_key(reports[0])
    if any(protocol_key(report) != first for report in reports[1:]):
        raise ValueError("source configuration, reward/action version, inputs, code or runtime differs")


def load_source(run_id: str) -> dict[str, Any]:
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("source run ID must be a single directory name")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "port-shore-bess-v8-report.v1" or report.get("run_id") != run_id:
        raise ValueError("source report identity mismatch")
    verify_references(report["evidence_files_sha256"], scope=run_dir)
    verify_references(report["manifest"]["source_sha256"])
    verify_references(report["input_files_sha256"])
    for filename, field in (("config.json", "config"), ("manifest.json", "manifest"), ("selection.json", "selection")):
        path = run_dir / filename
        if training.relative(path) not in report["evidence_files_sha256"]:
            raise ValueError(f"source report omitted its {filename} hash")
        if json.loads(path.read_text(encoding="utf-8")) != report[field]:
            raise ValueError(f"source {filename} differs from its report mirror")
    if len(report["results"]) != 1:
        raise ValueError("one seed per source run is required")
    result = report["results"][0]
    seed_dir = run_dir / f"seed_{result['seed']}"
    curve_path = seed_dir / "curve.json"
    initial_path = seed_dir / "initial_validation.json"
    for path in (curve_path, initial_path, seed_dir / "result.json", seed_dir / "monitor.csv"):
        if training.relative(path) not in report["evidence_files_sha256"]:
            raise ValueError(f"source report omitted required evidence: {path.name}")
    if json.loads((seed_dir / "result.json").read_text(encoding="utf-8")) != result:
        raise ValueError("source seed result differs from its report mirror")
    ledger = result.get("training_episode_ledger")
    if ledger is not None:
        ledger_path = checked_path(ledger["path"], scope=seed_dir)
        if report["evidence_files_sha256"].get(ledger["path"]) != ledger["sha256"] or training.file_sha256(ledger_path) != ledger["sha256"]:
            raise ValueError("source physical training episode ledger is not bound to its report")
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    if len(curve) < 3 or any(left["step"] >= right["step"] for left, right in zip(curve, curve[1:])):
        raise ValueError("source must preserve at least three ordered checkpoints")
    if curve[-1]["step"] != result["steps"] or curve[-1]["weights_sha256"] != result["final_weights_sha256"]:
        raise ValueError("source final checkpoint does not match actual training completion")
    for row in curve:
        path = checked_path(row["model_path"], scope=seed_dir)
        if report["evidence_files_sha256"].get(row["model_path"]) != row["model_sha256"] or training.file_sha256(path) != row["model_sha256"]:
            raise ValueError("source checkpoint hash is not bound to the pilot report")
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    if initial["weights_sha256"] != result["initial_weights_sha256"]:
        raise ValueError("source initial-network identity mismatch")
    return {"run_id": run_id, "run_dir": run_dir, "report_path": report_path,
            "report_sha256": training.file_sha256(report_path), "report": report,
            "result": result, "curve": curve, "initial": initial}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", required=True, help="exactly three comma-separated single-seed pilot run IDs")
    parser.add_argument("--run-id", help="new append-only formal admission run ID")
    args = parser.parse_args()
    run_ids = [value.strip() for value in args.run_ids.split(",")]
    if len(run_ids) != 3 or len(set(run_ids)) != 3:
        parser.error("exactly three distinct source run IDs are required")
    if args.run_id and (Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}):
        parser.error("run-id must be a single directory name")

    import torch
    import stable_baselines3 as sb3
    import sb3_contrib
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.noise import NormalActionNoise
    from app.services.rl_model.shore_bess import v8_environment as environment

    torch.set_num_threads(1)
    sources = [load_source(run_id) for run_id in run_ids]
    assert_compatible([source["report"] for source in sources])
    sources.sort(key=lambda source: source["result"]["seed"])
    source_config = sources[0]["report"]["config"]
    current_versions = {"python": platform.python_version(), "torch": torch.__version__,
                        "stable_baselines3": sb3.__version__, "sb3_contrib": sb3_contrib.__version__}
    if sources[0]["report"]["manifest"]["versions"] != current_versions:
        raise ValueError("admission runtime differs from the source training runtime")
    if source_config["admission_gate"] != training.GATE:
        raise ValueError("admission gates changed after source training")
    algorithm_cls = {"stable_baselines3.SAC": sb3.SAC, "stable_baselines3.TD3": sb3.TD3,
                     "stable_baselines3.DQN": sb3.DQN, "stable_baselines3.PPO": sb3.PPO,
                     "sb3_contrib.MaskablePPO": MaskablePPO}.get(source_config["algorithm"])
    if algorithm_cls is None:
        raise ValueError("unregistered source algorithm")
    continuous = not source_config["discrete"]
    dataset = training.load_port_dataset(source_config["physical_config"]["dataset_id"])
    quality = training.checked_quality(dataset)
    train = training.month_slice(dataset, "2024-01-01", "2025-05-01")
    validation = training.month_slice(dataset, "2025-05-01", "2025-08-01")
    for name, split in (("train", train), ("validation", validation)):
        if training.split_description(dataset, split) != sources[0]["report"]["manifest"][name]:
            raise ValueError("source chronological split differs from full-validation protocol")
    starts = training.window_starts(validation.stop - validation.start)
    idle = np.zeros(2, dtype=np.float32) if continuous else int(np.flatnonzero(np.all(environment.LATTICE == 0, axis=1))[0])

    def factory(data, split, seed=0, training_mode=False):
        env_cls = environment.ShoreBESSV8SACEnv if continuous else environment.ShoreBESSV8Env
        return lambda: env_cls(data, split, config=source_config["physical_config"], normalization_slice=train,
                               episode_steps=training.EPISODE_HOURS,
                               carbon_price=source_config["carbon_price_cny_per_kg_constraint_multiplier"],
                               discrete=not continuous, seed=seed, training=training_mode, record_trace=False)

    run_id = args.run_id or "shore-bess-v8-admission-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    original_inputs = copy.deepcopy(sources[0]["report"]["input_files_sha256"])
    protected_sources = {}
    for source in sources:
        protected_sources[training.relative(source["report_path"])] = source["report_sha256"]
        protected_sources.update(source["report"]["evidence_files_sha256"])
    source_hashes = dict(sources[0]["report"]["manifest"]["source_sha256"])
    source_hashes[training.relative(Path(__file__))] = training.file_sha256(Path(__file__))
    for path, digest in source_hashes.items():
        source_path = checked_path(path)
        target = run_dir / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        if training.file_sha256(target) != digest:
            raise ValueError("source snapshot copy mismatch")
    protected_pointers = {training.relative(path): training.file_sha256(path)
                          for version in ("v3", "v7")
                          for path in (ROOT / "evidence" / version / "shore_bess").glob("*.json")}
    config = copy.deepcopy(source_config)
    config.update(pilot=False, seeds=[source["result"]["seed"] for source in sources],
                  reused_training_run_ids=[source["run_id"] for source in sources])
    manifest = {"schema": "port-shore-bess-v8-manifest.v1", "run_id": run_id, "started_at": training.utc_now(),
                "training_reused": True, "training_protocol": "no learning; immutable existing checkpoints re-evaluated",
                "dataset_id": dataset.dataset_id, "dataset_sha256": dataset.fingerprint, "dataset_quality": quality,
                "train": training.split_description(dataset, train), "validation": training.split_description(dataset, validation),
                "validation_starts": starts, "normalization_fit_split": "train_only",
                "selection_protocol": "recompute every checkpoint on all validation windows, then freeze seed before test/forward",
                "test_status": "sealed_until_selection_previously_used_benchmark",
                "forward_status": "sealed_until_selection_previously_used_benchmark",
                "access_protocol_details": copy.deepcopy(sources[0]["report"]["manifest"].get("access_protocol_details", {})),
                "teacher_actions_used": False, "warm_start_used": False, "versions": current_versions,
                "training_episode_ledger_contract": sources[0]["report"]["manifest"].get("training_episode_ledger_contract"),
                "source_sha256": source_hashes, "input_files_sha256": dict(original_inputs),
                "historical_pointer_sha256": protected_pointers, "simulation_mode": True,
                "production_authority": False, "live_data_verified": False, "dispatch_allowed": False}
    training.write_json(run_dir / "config.json", config)
    training.write_json(run_dir / "manifest.json", manifest)
    reference = training.evaluate(factory(dataset, validation), None, starts, idle)
    training.write_json(run_dir / "validation_reference.json", reference)
    results = []
    provenance = []
    for source in sources:
        original_result = source["result"]
        seed = original_result["seed"]
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir()
        origin = {"run_id": source["run_id"], "report_path": training.relative(source["report_path"]),
                  "report_sha256": source["report_sha256"], "seed": seed,
                  "environment_steps": original_result["steps"], "optimizer_updates": original_result["optimizer_updates"],
                  "sb3_update_counter": original_result["sb3_update_counter"],
                  "config_path": training.relative(source["run_dir"] / "config.json"),
                  "config_sha256": training.file_sha256(source["run_dir"] / "config.json"),
                  "manifest_path": training.relative(source["run_dir"] / "manifest.json"),
                  "manifest_sha256": training.file_sha256(source["run_dir"] / "manifest.json"),
                  "initial_weights_sha256": original_result["initial_weights_sha256"], "checkpoints": []}
        kwargs = dict(source_config["algorithm_parameters"])
        kwargs.update(training.replay_construction_kwargs(source_config))
        if algorithm_cls is sb3.TD3:
            noise = source_config["action_noise"]
            kwargs["action_noise"] = NormalActionNoise(np.asarray(noise["mean"]), np.asarray(noise["sigma"]))
        initial_model = algorithm_cls("MlpPolicy", factory(dataset, train, seed, True)(), gamma=config["gamma"],
                                      policy_kwargs={"net_arch": config["network"]}, seed=seed,
                                      device="cpu", verbose=0, **kwargs)
        try:
            initial_model._v8_optimizer_step_calls = 0
            initial_model._v8_optimizer_steps_by_component = {name: 0 for name in training.model_optimizers(initial_model)}
            training.capture_replay_snapshot(initial_model, "initial_zero_updates")
            if training.weights_sha256(initial_model) != original_result["initial_weights_sha256"]:
                raise ValueError("reconstructed zero-update model differs from source initialization")
            initial = training.evaluate(factory(dataset, validation), initial_model, starts, idle)
            training.write_json(seed_dir / "initial_validation.json", {
                "evaluation": initial, "comparison": training.compare(initial, reference),
                "weights_sha256": training.weights_sha256(initial_model), "parameters": training.model_parameters(initial_model),
                "reconstructed_from_source_seed_without_learning": True,
            })
        finally:
            initial_model.env.close()
        curve = []
        for original in source["curve"]:
            source_model = checked_path(original["model_path"], scope=source["run_dir"])
            copied_model = seed_dir / f"step_{original['step']}.zip"
            shutil.copy2(source_model, copied_model)
            if training.file_sha256(copied_model) != original["model_sha256"]:
                raise ValueError("checkpoint bytes changed during adoption")
            model = algorithm_cls.load(copied_model, device="cpu")
            if training.weights_sha256(model) != original["weights_sha256"] or training.model_parameters(model) != original["parameters"]:
                raise ValueError("checkpoint weights or actual parameters differ from source evidence")
            ev = training.evaluate(factory(dataset, validation), model, starts, idle)
            comparison = training.compare(ev, reference)
            row = {"step": original["step"], "optimizer_updates": original["optimizer_updates"],
                   "checkpoint_phase": original.get("checkpoint_phase", "post_rollout_and_optimizer"),
                   "sb3_update_counter": original["sb3_update_counter"], "model_path": training.relative(copied_model),
                   "model_sha256": original["model_sha256"], "weights_sha256": original["weights_sha256"],
                   "parameters": training.model_parameters(model), "evaluation": ev, "comparison": comparison,
                   "gates": training.business_gates(ev, comparison, require_month_ci=True),
                   "optimizer": original.get("optimizer", {}), "source_training_run_id": source["run_id"]}
            curve.append(row)
            origin["checkpoints"].append({"step": original["step"], "source_model_path": original["model_path"],
                                           "source_model_sha256": original["model_sha256"],
                                           "source_weights_sha256": original["weights_sha256"],
                                           "formal_model_path": row["model_path"], "formal_model_sha256": row["model_sha256"]})
            training.write_json(seed_dir / "curve.json", curve)
            print(json.dumps({"run_id": run_id, "seed": seed, "reused_step": row["step"],
                              "validation_gain_percent": {key: value["mean"] for key, value in comparison.items()},
                              "failed": [key for key, value in row["gates"].items() if not value]}, allow_nan=False), flush=True)
        chosen = min(curve, key=training.rank)
        selected_path = seed_dir / "selected_model.zip"
        shutil.copy2(ROOT / chosen["model_path"], selected_path)
        shutil.copy2(source["run_dir"] / f"seed_{seed}" / "monitor.csv", seed_dir / "monitor.csv")
        result = {"seed": seed, "steps": original_result["steps"], "optimizer_updates": original_result["optimizer_updates"],
                  "sb3_update_counter": original_result["sb3_update_counter"],
                  "initial_weights_sha256": original_result["initial_weights_sha256"],
                  "final_weights_sha256": original_result["final_weights_sha256"],
                  "parameters": original_result["parameters"], "selected": chosen,
                  "model_path": training.relative(selected_path), "model_sha256": training.file_sha256(selected_path),
                  "convergence": training.convergence(curve), "render_calls": 0,
                  "source_training_run_id": source["run_id"], "training_reused": True}
        original_ledger = original_result.get("training_episode_ledger")
        if original_ledger is not None:
            source_ledger = checked_path(original_ledger["path"], scope=source["run_dir"])
            copied_ledger = seed_dir / "training_episodes.csv"
            shutil.copy2(source_ledger, copied_ledger)
            if training.file_sha256(copied_ledger) != original_ledger["sha256"]:
                raise ValueError("physical training episode ledger changed during adoption")
            result["training_episode_ledger"] = {**original_ledger, "path": training.relative(copied_ledger),
                "training_reused": True, "source_path": original_ledger["path"], "source_sha256": original_ledger["sha256"]}
        training.write_json(seed_dir / "result.json", result)
        results.append(result)
        provenance.append(origin)

    selected = min(results, key=lambda result: (not result["convergence"]["passed"], *training.rank(result["selected"])))
    selection = {"selected_seed": selected["seed"], "model_path": selected["model_path"],
                 "model_sha256": selected["model_sha256"], "selection_access_to_test": False,
                 "selection_access_to_forward": False, "frozen_at": training.utc_now(),
                 "per_seed_model_sha256": {str(result["seed"]): result["model_sha256"] for result in results}}
    selection_path = run_dir / "selection.json"
    training.write_json(selection_path, selection)
    frozen_selection_sha256 = training.file_sha256(selection_path)
    validation_passed = all(all(result["selected"]["gates"].values()) and result["convergence"]["passed"] for result in results)
    evaluations = {}
    holdout_access = {"test": "not_opened_validation_rejected", "forward": "not_loaded_validation_rejected"}
    if validation_passed:
        test = training.month_slice(dataset, "2025-08-01", "2026-01-01")
        forward = training.load_port_dataset("public_cn_sha_forward_2026m05_v1")
        forward_quality = training.checked_quality(forward)
        original_inputs.update(training.input_hashes(forward))
        combined = training.joined_forward(dataset, forward)
        forward_split = slice(dataset.rows, dataset.rows + forward.rows)
        training.write_json(run_dir / "forward_input_manifest.json", {"dataset_id": forward.dataset_id,
                            "input_files_sha256": training.input_hashes(forward),
                            "opened_after_selection_sha256": frozen_selection_sha256})
        for label, data, split in (("test", dataset, test), ("forward", combined, forward_split)):
            windows = training.window_starts(split.stop - split.start)
            baseline = training.evaluate(factory(data, split), None, windows, idle)
            partition = {"status": "previously_used_chronological_benchmark_not_fresh_blind_data",
                         "dataset_id": dataset.dataset_id if label == "test" else forward.dataset_id,
                         "dataset_sha256": dataset.fingerprint if label == "test" else forward.fingerprint,
                         "dataset_quality": quality if label == "test" else forward_quality,
                         "split": training.split_description(data, split), "reference": baseline, "per_seed": []}
            for result in results:
                if training.file_sha256(ROOT / result["model_path"]) != result["model_sha256"]:
                    raise ValueError("frozen selected model changed")
                model = algorithm_cls.load(ROOT / result["model_path"], device="cpu")
                ev = training.evaluate(factory(data, split), model, windows, idle)
                comparison = training.compare(ev, baseline)
                partition["per_seed"].append({"seed": result["seed"], "evaluation": ev,
                                              "comparison": comparison,
                                              "gates": training.business_gates(ev, comparison, require_month_ci=True)})
            evaluations[label] = partition
            holdout_access[label] = "evaluated_after_full_validation_selection_freeze"
            training.write_json(run_dir / f"{label}_evaluation.json", partition)

    checks = {"formal_run": True, "three_independent_seeds": True,
              "all_seeds_validation_passed": all(all(result["selected"]["gates"].values()) for result in results),
              "all_seeds_last_three_stable": all(result["convergence"]["passed"] for result in results),
              "real_optimizer_updates": all(result["optimizer_updates"] > 0 for result in results),
              "weights_changed": all(result["initial_weights_sha256"] != result["final_weights_sha256"] for result in results),
              "source_unchanged_during_training": training.hashes_match(source_hashes),
              "config_and_dataset_files_unchanged": training.hashes_match(original_inputs),
              "historical_pointers_preserved": training.hashes_match(protected_pointers),
              "no_training_rendering": True, "source_training_evidence_preserved": training.hashes_match(protected_sources),
              "frozen_selection_preserved": training.file_sha256(selection_path) == frozen_selection_sha256,
              "test_all_seed_gates": "test" in evaluations and all(all(result["gates"].values()) for result in evaluations["test"]["per_seed"]),
              "forward_all_seed_gates": "forward" in evaluations and all(all(result["gates"].values()) for result in evaluations["forward"]["per_seed"])}
    admitted = all(checks.values())
    champion_path = OUTPUT_ROOT / "offline_champion.json"
    promoted = admitted and not champion_path.exists()
    status = "VALIDATION_REJECTED" if not validation_passed else ("ADMITTED_OFFLINE_RL" if admitted else "CANDIDATE_NOT_ADMITTED")
    referenced_steps = sum(result["steps"] for result in results)
    referenced_updates = sum(result["optimizer_updates"] for result in results)
    report = {"schema": "port-shore-bess-v8-report.v1", "report_kind": "formal_admission_of_immutable_pilot_checkpoints",
              "run_id": run_id, "generated_at": training.utc_now(), "status": status, "training_reused": True,
              "config": config, "manifest": manifest, "checks": checks, "results": results,
              "selected_seed": selected["seed"], "selection": selection, "selection_sha256": frozen_selection_sha256,
              "evaluations": evaluations, "holdout_access": holdout_access, "source_training_runs": provenance,
              "source_training_files_sha256": protected_sources, "input_files_sha256": original_inputs,
              "total_environment_steps": referenced_steps, "total_optimizer_updates": referenced_updates,
              "new_training_environment_steps": 0, "new_optimizer_updates": 0,
              "reused_training_environment_steps": referenced_steps,
              "training_accounting": {"count_as_new_training_run": False, "environment_steps_executed_by_this_run": 0,
                                      "optimizer_updates_executed_by_this_run": 0,
                                      "referenced_environment_steps": referenced_steps, "referenced_optimizer_updates": referenced_updates},
              "promoted": promoted, "promotion": {"candidate_qualified": admitted, "promoted": promoted,
                  "reason": "qualified vacant champion slot" if promoted else "validation/benchmark gate failed" if not admitted else "existing champion retained pending paired incumbent comparison"},
              "simulation_mode": True, "production_authority": False, "live_data_verified": False, "dispatch_allowed": False,
              "claim_boundary": "Public/engineering offline scenario; reused immutable RL training. Repeated benchmarks, not fresh blind data or measured port savings."}
    report["evidence_files_sha256"] = {training.relative(path): training.file_sha256(path)
                                      for path in sorted(run_dir.rglob("*"))
                                      if path.is_file() and path.suffix in {".json", ".csv", ".zip"}
                                      and "source" not in path.relative_to(run_dir).parts}
    report_path = run_dir / "report.json"
    training.write_json(report_path, report)
    pointer = {"schema": "port-shore-bess-v8-pointer.v1", "run_id": run_id, "status": status,
               "report_path": training.relative(report_path), "report_sha256": training.file_sha256(report_path),
               "model_path": selected["model_path"], "model_sha256": selected["model_sha256"],
               "production_authority": False, "updated_at": training.utc_now()}
    training.write_json(OUTPUT_ROOT / "latest.json", pointer)
    if promoted:
        training.write_json(champion_path, pointer)
    print(json.dumps(pointer, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
