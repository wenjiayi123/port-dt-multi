"""Read-only V8 evidence and real selected-actor inference; never promotes a run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.services.rl_training.datasets import file_sha256, load_port_dataset
from app.services.rl_model.shore_bess.v8_public_artifacts import resolve_public_artifact
from app.services.rl_training.statistics import bootstrap_summary
from app.services.rl_training.trainer import SB3_IMPORT_LOCK


BOUNDARY = {"simulation_mode": True, "production_authority": False,
            "dispatch_allowed": False, "live_data_verified": False,
            "claim_eligible": False, "site_status": "待接入港口"}
REQUIRED_CHECKS = {
    "formal_run", "three_independent_seeds", "all_seeds_validation_passed",
    "all_seeds_last_three_stable", "real_optimizer_updates", "weights_changed",
    "source_unchanged_during_training", "historical_pointers_preserved",
    "no_training_rendering", "test_all_seed_gates", "forward_all_seed_gates",
}


def boolean_gates(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def verify_checkpoint_counters(actor: Any, selected: dict) -> dict:
    sb3_updates = int(actor._n_updates)
    optimizer_calls = int(getattr(actor, "_v8_optimizer_step_calls", actor._n_updates))
    if (int(actor.num_timesteps) != selected["step"]
            or sb3_updates != selected.get("sb3_update_counter", selected["optimizer_updates"])
            or optimizer_calls != selected["optimizer_updates"]):
        raise ValueError("V8 loaded model optimizer/checkpoint counters differ")
    return {"loaded_checkpoint_environment_steps": int(actor.num_timesteps),
            "loaded_checkpoint_optimizer_updates": optimizer_calls,
            "loaded_checkpoint_sb3_update_counter": sb3_updates}


class ShoreBESSV8EvidenceService:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.directory = self.root / "evidence/v8/shore_bess"

    def _path(self, value: str) -> Path:
        if Path(value).is_absolute():
            raise ValueError("V8 artifact path must be repository relative")
        path = (self.root / value).resolve()
        if self.root not in path.parents:
            raise ValueError("V8 artifact path escapes repository")
        return path

    def _json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _verify(self, relative: str, expected: str) -> Path:
        return resolve_public_artifact(self.root, relative, expected)

    def build(self) -> dict:
        diagnostics = self._diagnostics()
        champion = self.directory / "offline_champion.json"
        pointer_path = champion if champion.is_file() else self.directory / "latest.json"
        if not pointer_path.is_file():
            return {"version": "V8", "available": False, "status": "NO_FORMAL_V8_REPORT",
                    "boundary": dict(BOUNDARY), "admitted": False, "diagnostics": diagnostics}
        try:
            return {**self._build(pointer_path, champion.is_file()), "diagnostics": diagnostics}
        except (ValueError, OSError, KeyError, TypeError, StopIteration, RuntimeError) as exc:
            return {"version": "V8", "available": False, "status": "INVALID_EVIDENCE",
                    "admitted": False, "error": str(exc), "boundary": dict(BOUNDARY),
                    "pointer_path": str(pointer_path.relative_to(self.root)), "diagnostics": diagnostics}

    def _diagnostics(self) -> dict:
        """Independent last-checkpoint pilots never acquire formal/current status."""
        path = self.directory / "diagnostics_latest.json"
        empty = {"available": False, "runs": [], "admitted": False, "formal": False,
                 "online_inference_performed": False, "boundary": dict(BOUNDARY)}
        if not path.is_file():
            return {**empty, "status": "NO_DIAGNOSTIC_REPORTS"}
        try:
            pointer = self._json(path)
            entries = pointer["reports"]
            if (pointer.get("schema") != "port-shore-bess-v8-diagnostics-pointer.v1"
                    or not isinstance(entries, list) or not 1 <= len(entries) <= 4
                    or len({row["run_id"] for row in entries}) != len(entries)):
                raise ValueError("V8 diagnostic pointer schema or run identity is invalid")
            runs = [self._diagnostic_run(entry) for entry in entries]
            return {**empty, "available": True, "status": "INDEPENDENT_SINGLE_SEED_DIAGNOSTICS",
                    "runs": runs, "pointer_path": str(path.relative_to(self.root)),
                    "note": "独立单种子诊断，固定最后60k检查点；验证指标只作训练诊断，不是正式准入、留出表现或在线接管。"}
        except (ValueError, OSError, KeyError, TypeError, StopIteration, RuntimeError) as exc:
            return {**empty, "status": "INVALID_DIAGNOSTIC_EVIDENCE", "error": str(exc)}

    def _diagnostic_run(self, entry: dict) -> dict:
        from scripts.train_shore_bess_v8 import business_gates, weights_sha256

        report_path = self._verify(entry["report_path"], entry["report_sha256"])
        report = self._json(report_path)
        cfg, manifest = report["config"], report["manifest"]
        if (report.get("schema") != "port-shore-bess-v8-report.v1"
                or report["run_id"] != entry["run_id"] or manifest["run_id"] != report["run_id"]
                or report["status"] != "PILOT_VALIDATION_ONLY" or cfg.get("pilot") is not True
                or report["evaluations"] != {} or report.get("promoted") is not False
                or len(report["results"]) != 1 or len(cfg["seeds"]) != 1
                or manifest.get("test_status") != "not_opened" or manifest.get("forward_status") != "not_loaded"
                or manifest.get("teacher_actions_used") is not False or manifest.get("warm_start_used") is not False
                or manifest.get("normalization_fit_split") != "train_only"):
            raise ValueError("V8 diagnostic must remain a single-seed pilot without heldout access")
        result = report["results"][0]
        if (cfg["seeds"] != [result["seed"]] or result["render_calls"] != 0
                or report["total_environment_steps"] != result["steps"]
                or report["total_optimizer_updates"] != result["optimizer_updates"]
                or result["optimizer_updates"] <= 0 or entry["checkpoint_step"] != 60000
                or result["steps"] != entry["checkpoint_step"]
                or cfg["requested_steps_per_seed"] != entry["checkpoint_step"]):
            raise ValueError("V8 diagnostic final 60k-step training accounting differs")
        checks = report["checks"]
        integrity_checks = ("real_optimizer_updates", "weights_changed", "source_unchanged_during_training",
                            "config_and_dataset_files_unchanged", "historical_pointers_preserved", "no_training_rendering")
        if any(type(value) is not bool for value in checks.values()) or not all(checks.get(key) is True for key in integrity_checks):
            raise ValueError("V8 diagnostic training integrity checks failed")
        anchors = report["evidence_files_sha256"]
        if not anchors or not manifest["source_sha256"]:
            raise ValueError("V8 diagnostic evidence anchors are empty")
        for relative, expected in anchors.items():
            self._verify(relative, expected)
        for name, embedded in (("config.json", cfg), ("manifest.json", manifest)):
            artifact = report_path.with_name(name)
            if str(artifact.relative_to(self.root)) not in anchors or self._json(artifact) != embedded:
                raise ValueError("V8 diagnostic config/manifest mirror or hash is missing")
        # Old diagnostic snapshots remain inspectable after subsequent source
        # changes. No current-environment policy replay is performed here.
        for relative, expected in manifest["source_sha256"].items():
            self._path(relative)
            if file_sha256(report_path.parent / "source" / relative) != expected:
                raise ValueError("V8 diagnostic frozen source snapshot differs")
        seed_path = report_path.parent / f"seed_{result['seed']}"
        curve_path = seed_path / "curve.json"
        reference_path = report_path.with_name("validation_reference.json")
        for artifact in (curve_path, reference_path):
            if str(artifact.relative_to(self.root)) not in anchors:
                raise ValueError("V8 diagnostic validation evidence is not anchored")
        curve = self._json(curve_path)
        if not curve or any(left["step"] >= right["step"] for left, right in zip(curve, curve[1:])):
            raise ValueError("V8 diagnostic checkpoints are not strictly ordered")
        final = curve[-1]
        if (final["step"] != result["steps"] or final["optimizer_updates"] != result["optimizer_updates"]
                or final["sb3_update_counter"] != result["sb3_update_counter"]
                or final["weights_sha256"] != result["final_weights_sha256"]
                or anchors.get(final["model_path"]) != final["model_sha256"]):
            raise ValueError("V8 diagnostic must show the actual final checkpoint, not validation-selected weights")
        reference = self._json(reference_path)
        if reference["starts"] != manifest["validation_starts"]:
            raise ValueError("V8 diagnostic validation window identity differs")
        expected_gates = business_gates(final["evaluation"], final["comparison"])
        if final["gates"] != expected_gates:
            raise ValueError("V8 diagnostic window gates differ from final-checkpoint metrics")
        partition = {"status": "single_seed_fixed_validation_diagnostic_only", "dataset_id": manifest["dataset_id"],
                     "dataset_sha256": manifest["dataset_sha256"], "split": manifest["validation"], "reference": reference}
        summary = self._summary(partition, {"seed": result["seed"], **final})
        loaded_path = self._verify(final["model_path"], final["model_sha256"])
        actor = self._actor(str(loaded_path), file_sha256(loaded_path), cfg["algorithm"])
        counters = verify_checkpoint_counters(actor, final)
        if weights_sha256(actor) != final["weights_sha256"]:
            raise ValueError("V8 diagnostic deserialized network weights differ")
        return {"run_id": report["run_id"], "report_path": entry["report_path"], "report_sha256": entry["report_sha256"],
                "algorithm": cfg["algorithm"], "algorithm_variant": cfg.get("algorithm_variant", cfg["algorithm"]),
                "n_step": cfg.get("n_step", 1), "seed": result["seed"], "checkpoint_step": final["step"],
                "optimizer_updates": final["optimizer_updates"], "model_path": final["model_path"],
                "model_sha256": final["model_sha256"],
                "loaded_model_path": str(loaded_path.relative_to(self.root)),
                "loaded_model_sha256": file_sha256(loaded_path), "comparison": summary["comparison"],
                "windows": summary["windows"], "episode_hours": summary["episode_hours"],
                "gates": final["gates"], "diagnostic_window_gates_passed": boolean_gates(final["gates"]),
                "failed_gates": [key for key, passed in final["gates"].items() if not passed],
                "metrics": summary["metrics"], "integrity_verified": True,
                "checkpoint_deserialized_for_integrity_only": True, "online_inference_performed": False,
                "selection_basis": "predeclared_final_60000_environment_steps_without_validation_ranking",
                "admitted": False, "formal": False, "heldout_evaluated": False, **counters}

    def _build(self, pointer_path: Path, from_champion: bool) -> dict:
        pointer = self._json(pointer_path)
        report_path = self._verify(pointer["report_path"], pointer["report_sha256"])
        report = self._json(report_path)
        if report.get("schema") != "port-shore-bess-v8-report.v1":
            raise ValueError("unsupported V8 report schema")
        if (report["run_id"] != pointer["run_id"] or report["status"] != pointer["status"]
                or report["status"] not in {"ADMITTED_OFFLINE_RL", "CANDIDATE_NOT_ADMITTED", "VALIDATION_REJECTED"}):
            raise ValueError("V8 pointer/report identity mismatch or non-formal run")
        cfg, manifest, selection = report["config"], report["manifest"], report["selection"]
        if cfg.get("pilot") is not False or manifest["run_id"] != report["run_id"]:
            raise ValueError("V8 report does not identify a formal run")
        for name, embedded in (("config.json", cfg), ("manifest.json", manifest), ("selection.json", selection)):
            if self._json(report_path.with_name(name)) != embedded:
                raise ValueError("V8 embedded artifact mismatch: " + name)
        if (selection.get("selection_access_to_test") is not False
                or selection.get("selection_access_to_forward") is not False
                or manifest.get("teacher_actions_used") is not False
                or manifest.get("warm_start_used") is not False
                or manifest.get("normalization_fit_split") != "train_only"):
            raise ValueError("V8 frozen selection or learning provenance is invalid")
        if (selection["selected_seed"] != report["selected_seed"]
                or selection["model_path"] != pointer["model_path"]
                or selection["model_sha256"] != pointer["model_sha256"]):
            raise ValueError("V8 selected actor differs from frozen selection")
        if not manifest["source_sha256"]:
            raise ValueError("V8 frozen source manifest is empty")
        for relative, expected in manifest["source_sha256"].items():
            self._verify(relative, expected)
            snapshot = report_path.parent / "source" / relative
            if file_sha256(snapshot) != expected:
                raise ValueError("V8 frozen source hash mismatch: " + relative)
        for relative, expected in report.get("evidence_files_sha256", {}).items():
            self._verify(relative, expected)
        for relative, expected in manifest["historical_pointer_sha256"].items():
            self._verify(relative, expected)
        results = report["results"]
        seeds = [row["seed"] for row in results]
        if (len(seeds) < 3 or len(set(seeds)) != len(seeds) or seeds != cfg["seeds"]
                or report["total_environment_steps"] != sum(row["steps"] for row in results)
                or report["total_optimizer_updates"] != sum(row["optimizer_updates"] for row in results)):
            raise ValueError("V8 seed or optimizer accounting mismatch")
        for row in results:
            self._verify(row["model_path"], row["model_sha256"])
            if (selection["per_seed_model_sha256"][str(row["seed"])] != row["model_sha256"]
                    or row["selected"]["model_sha256"] != row["model_sha256"]):
                raise ValueError("V8 per-seed model changed after selection")
        selected = next(row for row in results if row["seed"] == report["selected_seed"])
        if selected["model_sha256"] != pointer["model_sha256"]:
            raise ValueError("V8 selected result/model mismatch")
        checks = report["checks"]
        if not REQUIRED_CHECKS.issubset(checks) or any(type(value) is not bool for value in checks.values()):
            raise ValueError("V8 admission checks are missing or not boolean")
        admitted = boolean_gates(checks)
        reused = report.get("report_kind") == "formal_admission_of_immutable_pilot_checkpoints"
        if admitted != (report["status"] == "ADMITTED_OFFLINE_RL"):
            raise ValueError("V8 status contradicts admission checks")
        if reused:
            self._verify_reused_training(report, report_path)
            promotion = report["promotion"]
            if (type(report.get("promoted")) is not bool or promotion["promoted"] is not report["promoted"]
                    or promotion["candidate_qualified"] is not admitted or (report["promoted"] and not admitted)):
                raise ValueError("V8 promotion contradicts candidate qualification")
        elif report.get("promoted") is not admitted or report.get("training_reused", False) is not False:
            raise ValueError("V8 status contradicts training or promotion provenance")
        if from_champion and not admitted:
            raise ValueError("V8 champion pointer refers to a rejected candidate")
        summaries = {}
        from scripts.train_shore_bess_v8 import business_gates
        validation_only = report["status"] == "VALIDATION_REJECTED"
        if validation_only:
            if (report["evaluations"] != {}
                    or report.get("holdout_access") != {"test": "not_opened_validation_rejected",
                                                        "forward": "not_loaded_validation_rejected"}
                    or checks["test_all_seed_gates"] is not False
                    or checks["forward_all_seed_gates"] is not False
                    or (checks["all_seeds_validation_passed"] and checks["all_seeds_last_three_stable"])):
                raise ValueError("V8 validation rejection contradicts holdout access or gates")
            reference_path = report_path.with_name("validation_reference.json")
            if str(reference_path.relative_to(self.root)) not in report.get("evidence_files_sha256", {}):
                raise ValueError("V8 validation reference lacks a SHA-256 anchor")
            partition = {"status": "validation_only_rejected_before_heldout",
                         "dataset_id": manifest["dataset_id"], "dataset_sha256": manifest["dataset_sha256"],
                         "split": manifest["validation"], "reference": self._json(reference_path)}
            if partition["reference"]["starts"] != manifest["validation_starts"]:
                raise ValueError("V8 validation reference differs from frozen validation windows")
            for row in results:
                candidate = {"seed": row["seed"], **row["selected"]}
                if candidate["gates"] != business_gates(candidate["evaluation"], candidate["comparison"], require_month_ci=True):
                    raise ValueError("V8 validation gates do not match evaluated metrics")
                summary = self._summary(partition, candidate)
                if row["seed"] == report["selected_seed"]:
                    summaries["validation"] = summary
            if (checks["all_seeds_validation_passed"] != all(boolean_gates(row["selected"]["gates"]) for row in results)
                    or checks["all_seeds_last_three_stable"] != all(row["convergence"].get("passed") is True for row in results)):
                raise ValueError("V8 validation rejection gate accounting differs")
        for label in (() if validation_only else ("test", "forward")):
            partition = report["evaluations"][label]
            rows = partition["per_seed"]
            if {row["seed"] for row in rows} != set(seeds):
                raise ValueError("V8 evaluation seed coverage mismatch")
            seed_summaries = {}
            for row in rows:
                if row["gates"] != business_gates(row["evaluation"], row["comparison"], require_month_ci=True):
                    raise ValueError("V8 business gate does not match evaluated metrics")
                seed_summaries[row["seed"]] = self._summary(partition, row)
            if admitted and not all(boolean_gates(row["gates"]) for row in rows):
                raise ValueError("V8 admitted report contains failed evaluation gates")
            summaries[label] = seed_summaries[report["selected_seed"]]
        if admitted and not all(boolean_gates(row["selected"]["gates"])
                                and row["convergence"].get("passed") is True
                                and row["optimizer_updates"] > 0 and row["render_calls"] == 0
                                for row in results):
            raise ValueError("V8 admitted report contains unqualified training results")
        primary = "validation" if validation_only else "forward"
        output = self._inference(report, pointer, summaries[primary])
        return {"version": "V8", "available": True, "status": report["status"],
                "admitted": admitted, "run_id": report["run_id"], "pointer": pointer,
                "pointer_role": "offline_champion" if from_champion else "latest_formal_candidate",
                "integrity_verified": True, "boundary": dict(BOUNDARY), "config": cfg,
                "dataset": {"dataset_id": manifest["dataset_id"], "sha256": manifest["dataset_sha256"],
                            "train": manifest["train"], "validation": manifest["validation"]},
                "checks": checks, "failed_gates": [key for key, value in checks.items() if not value],
                "selected_seed": report["selected_seed"], "benchmark_summaries": summaries,
                "primary_benchmark": primary, "holdout_access": report.get("holdout_access", {}),
                "report_kind": report.get("report_kind", "independent_neural_policy_training"),
                "training_reused": report.get("training_reused", False),
                "new_training_environment_steps": report.get("new_training_environment_steps", report["total_environment_steps"]),
                "new_optimizer_updates": report.get("new_optimizer_updates", report["total_optimizer_updates"]),
                "source_training_runs": report.get("source_training_runs", []),
                "total_environment_steps": report["total_environment_steps"],
                "total_optimizer_updates": report["total_optimizer_updates"],
                "convergence": [{"seed": row["seed"], **row["convergence"]} for row in results],
                "training_process": self._curves(report, report_path),
                "current_model_output": output, "selection": selection,
                "claim_boundary": report["claim_boundary"],
                "benchmark_note": ("完整固定验证未通过；测试区间未打开，前向数据未加载。验证结果不作为留出表现或年化收益。" if validation_only else
                                   "历史与前向区间均是已使用的公开工程基准；不称全新盲测，不代表港口实测。")}

    def _verify_reused_training(self, report: dict, report_path: Path) -> None:
        """Bind admission to three completed pilot runs without double counting."""
        from scripts.admit_shore_bess_v8 import INTEGRITY_CHECKS, assert_compatible
        from scripts.train_shore_bess_v8 import business_gates, convergence, rank, window_starts

        origins = report["source_training_runs"]
        cfg, manifest, results = report["config"], report["manifest"], report["results"]
        anchors = report["source_training_files_sha256"]
        run_ids = [row["run_id"] for row in origins]
        if (report.get("training_reused") is not True or manifest.get("training_reused") is not True
                or len(origins) != 3 or len(set(run_ids)) != 3
                or [row["seed"] for row in origins] != cfg["seeds"]
                or cfg["reused_training_run_ids"] != run_ids
                or any(type(report[key]) is not int or report[key] != 0
                       for key in ("new_training_environment_steps", "new_optimizer_updates"))
                or report["reused_training_environment_steps"] != report["total_environment_steps"]):
            raise ValueError("V8 reused training identity or new-work accounting mismatch")
        expected_accounting = {"count_as_new_training_run": False, "environment_steps_executed_by_this_run": 0,
                               "optimizer_updates_executed_by_this_run": 0,
                               "referenced_environment_steps": report["total_environment_steps"],
                               "referenced_optimizer_updates": report["total_optimizer_updates"]}
        if report["training_accounting"] != expected_accounting:
            raise ValueError("V8 reused training must not be counted as new learning")
        selection_path = report_path.with_name("selection.json")
        if file_sha256(selection_path) != report["selection_sha256"]:
            raise ValueError("V8 reused frozen selection hash mismatch")
        if not anchors:
            raise ValueError("V8 reused training evidence registry is empty")
        for relative, expected in anchors.items():
            self._verify(relative, expected)
        required = {"source_training_evidence_preserved", "frozen_selection_preserved", "config_and_dataset_files_unchanged"}
        if not required.issubset(report["checks"]) or any(report["checks"][key] is not True for key in required):
            raise ValueError("V8 reused source integrity gate failed")
        validation_path = report_path.with_name("validation_reference.json")
        formal_anchors = report["evidence_files_sha256"]
        if str(validation_path.relative_to(self.root)) not in formal_anchors:
            raise ValueError("V8 reused full-validation reference is not anchored")
        reference = self._json(validation_path)
        if (reference["starts"] != manifest["validation_starts"]
                or reference["starts"] != window_starts(manifest["validation"]["rows"])):
            raise ValueError("V8 reused full-validation window identity mismatch")
        partition = {"status": "full_fixed_validation", "dataset_id": manifest["dataset_id"],
                     "dataset_sha256": manifest["dataset_sha256"], "split": manifest["validation"], "reference": reference}
        pilot_reports = []
        for origin, result in zip(origins, results, strict=True):
            source_path = self._verify(origin["report_path"], origin["report_sha256"])
            source = self._json(source_path)
            if anchors.get(origin["report_path"]) != origin["report_sha256"] or source["run_id"] != origin["run_id"]:
                raise ValueError("V8 source pilot report identity is not anchored")
            pilot_reports.append(source)
            for name in ("config", "manifest"):
                path = self._verify(origin[name + "_path"], origin[name + "_sha256"])
                if path != source_path.with_name(name + ".json") or self._json(path) != source[name]:
                    raise ValueError("V8 source pilot config/manifest differs from original report")
            for relative, expected in source["evidence_files_sha256"].items():
                if anchors.get(relative) != expected:
                    raise ValueError("V8 source pilot evidence omitted from preserved registry")
            for relative, expected in source["manifest"]["source_sha256"].items():
                if manifest["source_sha256"].get(relative) != expected:
                    raise ValueError("V8 reused training source differs from admission environment")
            for relative, expected in source["input_files_sha256"].items():
                if report["input_files_sha256"].get(relative) != expected:
                    raise ValueError("V8 reused training inputs differ from admission inputs")
                self._verify(relative, expected)
            original = source["results"][0]
            expected_cfg = {**source["config"], "pilot": False, "seeds": cfg["seeds"], "reused_training_run_ids": run_ids}
            if cfg != expected_cfg or any(source["checks"].get(key) is not True for key in INTEGRITY_CHECKS):
                raise ValueError("V8 source pilot protocol or integrity differs")
            if (result.get("training_reused") is not True or result.get("source_training_run_id") != origin["run_id"]
                    or origin["seed"] != original["seed"] or result["seed"] != original["seed"]
                    or origin["initial_weights_sha256"] != original["initial_weights_sha256"]):
                raise ValueError("V8 reused seed origin differs")
            for origin_key, result_key in (("environment_steps", "steps"), ("optimizer_updates", "optimizer_updates"),
                                           ("sb3_update_counter", "sb3_update_counter")):
                if origin[origin_key] != original[result_key] or result[result_key] != original[result_key]:
                    raise ValueError("V8 reused optimizer/environment counters differ from actual training")
            for name in ("initial_weights_sha256", "final_weights_sha256", "parameters", "render_calls"):
                if result[name] != original[name]:
                    raise ValueError("V8 reused model provenance differs from original training")
            old_curve_path = source_path.parent / f"seed_{origin['seed']}" / "curve.json"
            curve_path = report_path.parent / f"seed_{origin['seed']}" / "curve.json"
            if str(old_curve_path.relative_to(self.root)) not in source["evidence_files_sha256"] or str(curve_path.relative_to(self.root)) not in formal_anchors:
                raise ValueError("V8 reused checkpoint curves are not anchored")
            old_curve, curve = self._json(old_curve_path), self._json(curve_path)
            if len(curve) < 3 or len(curve) != len(old_curve) or len(curve) != len(origin["checkpoints"]):
                raise ValueError("V8 reused complete checkpoint coverage differs")
            for old, current, link in zip(old_curve, curve, origin["checkpoints"], strict=True):
                expected_link = {"step": old["step"], "source_model_path": old["model_path"],
                                 "source_model_sha256": old["model_sha256"], "source_weights_sha256": old["weights_sha256"],
                                 "formal_model_path": current["model_path"], "formal_model_sha256": current["model_sha256"]}
                if link != expected_link or current["model_sha256"] != old["model_sha256"]:
                    raise ValueError("V8 reused checkpoint copy differs from immutable pilot")
                for key in ("step", "optimizer_updates", "sb3_update_counter", "weights_sha256", "parameters", "optimizer"):
                    if current[key] != old[key]:
                        raise ValueError("V8 reused checkpoint training provenance changed")
                if (anchors.get(old["model_path"]) != old["model_sha256"]
                        or formal_anchors.get(current["model_path"]) != current["model_sha256"]):
                    raise ValueError("V8 reused checkpoint weights lack report hash anchors")
                if current["gates"] != business_gates(current["evaluation"], current["comparison"], require_month_ci=True):
                    raise ValueError("V8 reused full-validation checkpoint gates differ")
                self._summary(partition, {"seed": result["seed"], **current})
            if result["selected"] != min(curve, key=rank) or result["convergence"] != convergence(curve):
                raise ValueError("V8 reused selection or tail stability differs from full validation")
        assert_compatible(pilot_reports)
        selected = min(results, key=lambda row: (not row["convergence"]["passed"], *rank(row["selected"])))
        if selected["seed"] != report["selected_seed"]:
            raise ValueError("V8 reused selected seed differs from full-validation ranking")

    @staticmethod
    def _summary(partition: dict, chosen: dict) -> dict:
        evaluation, baseline = chosen["evaluation"], partition["reference"]
        starts = evaluation["starts"]
        hours = int(evaluation["episode_hours"])
        if (starts != baseline["starts"] or len(starts) != len(evaluation["rows"])
                or not starts or hours <= 0
                or any(right - left < hours for left, right in zip(starts, starts[1:]))):
            raise ValueError("V8 paired windows overlap or are incomplete")
        for data in (evaluation, baseline):
            for key, claimed_mean in data["mean"].items():
                values = [row[key] for row in data["rows"]]
                if not np.isfinite(values).all() or not np.isclose(np.mean(values), claimed_mean, rtol=1e-10, atol=1e-10):
                    raise ValueError("V8 mean KPI differs from actual per-window accounting")
        from scripts.train_shore_bess_v8 import official_source_period
        windows = evaluation.get("windows")
        if not isinstance(windows, list) or len(windows) != len(starts):
            raise ValueError("V8 source-month window metadata is missing")
        for start, window in zip(starts, windows, strict=True):
            first = datetime.fromisoformat(window["first_timestamp"].replace("Z", "+00:00"))
            last = datetime.fromisoformat(window["last_timestamp"].replace("Z", "+00:00"))
            if (first.tzinfo is None or last.tzinfo is None or window["start_index"] != start
                    or window["source_month"] != first.astimezone(timezone.utc).strftime("%Y-%m")
                    or window["source_period"] != official_source_period(first.astimezone(timezone.utc).isoformat())
                    or (last - first).total_seconds() != (hours - 1) * 3600):
                raise ValueError("V8 source-month labels or window duration are inconsistent")
        annual = {}
        from scripts.train_shore_bess_v8 import compare, source_period_summary
        recomputed = compare(evaluation, baseline)
        for metric, summary in recomputed.items():
            claimed = chosen["comparison"][metric]
            for key in ("mean", "ci_low", "ci_high", "minimum_percent", "clustered_ci_low", "clustered_ci_high", "cluster_confidence"):
                if not np.isclose(summary[key], claimed[key], rtol=1e-10, atol=1e-10):
                    raise ValueError("V8 claimed improvement differs from paired evaluation")
            for key in ("cluster_count", "cluster_method", "cluster_group_field", "cluster_resamples", "cluster_window_counts", "cluster_boundary"):
                if summary[key] != claimed[key]:
                    raise ValueError("V8 source-month CI metadata differs from paired evaluation")
            if not np.allclose(summary["per_window_percent"], claimed["per_window_percent"], rtol=1e-10, atol=1e-10):
                raise ValueError("V8 per-window improvements differ from actual evaluation")
        for metric, name, divisor in (("total_cost_cny", "cost_savings_cny", 1.0), ("carbon_kg", "carbon_reduction_t", 1000.0)):
            values = [(base[metric] - candidate[metric]) * 8760 / hours / divisor
                      for base, candidate in zip(baseline["rows"], evaluation["rows"], strict=True)]
            if not np.isfinite(values).all():
                raise ValueError("V8 annualization contains non-finite values")
            annual[name] = bootstrap_summary(values, seed=20260912, resamples=5000)
            annual[name].update(source_period_summary(values, [window["source_period"] for window in windows]))
        return {"status": partition["status"], "dataset_id": partition["dataset_id"],
                "dataset_sha256": partition["dataset_sha256"], "split": partition["split"],
                "windows": len(starts), "episode_hours": hours, "evaluated_hours": len(starts) * hours,
                "starts": starts, "selected_seed": chosen["seed"], "comparison": chosen["comparison"],
                "window_metadata": windows,
                "confidence_interval_boundary": "保留逐周窗口bootstrap区间；准入另要求按官方来源期间整组重采样。1—2月共享官方锚点并合组，跨期间窗口归起始期。少簇工程不确定性不等于现场95%保障。",
                "gates": chosen["gates"], "metrics": evaluation["mean"],
                "annualized_scenario": annual, "annualization_hours": 8760,
                "annualization_boundary": "配对周窗口外推；电能费、需量费与退化成本；非已实现利润或现场账单。"}

    def _curves(self, report: dict, report_path: Path) -> dict:
        series = []
        anchors = report.get("evidence_files_sha256", {})
        for result in report["results"]:
            path = report_path.parent / f"seed_{result['seed']}" / "curve.json"
            relative = str(path.relative_to(self.root))
            rows = self._json(path) if relative in anchors else [result["selected"]]
            points = [{"step": row["step"], "optimizer_updates": row["optimizer_updates"],
                       "validation_reward": row["evaluation"]["mean"]["learning_reward"],
                       "cost_gain_percent": row["comparison"]["total_cost_cny"]["mean"],
                       "carbon_gain_percent": row["comparison"]["carbon_kg"]["mean"],
                       "peak_gain_percent": row["comparison"]["peak_kw"]["mean"],
                       "optimizer": row["optimizer"]} for row in rows]
            series.append({"seed": result["seed"], "source": relative if relative in anchors else "report_selected_checkpoint_only", "points": points})
        return {"series": series, "environment_reward_optimized": True, "teacher_actions_used": False,
                "training_render_calls": sum(row["render_calls"] for row in report["results"]),
                "total_persisted_checkpoints": sum(len(row["points"]) for row in series),
                "curve_kind": "saved_optimizer_checkpoints_and_fixed_validation_replay",
                "raw_curves_hash_verified": all(row["source"] != "report_selected_checkpoint_only" for row in series)}

    @staticmethod
    @lru_cache(maxsize=4)
    def _actor(path: str, expected: str, algorithm: str):
        if file_sha256(Path(path)) != expected:
            raise ValueError("V8 actor hash mismatch before load")
        with SB3_IMPORT_LOCK:
            from stable_baselines3 import DQN, PPO, SAC, TD3
            implementations = {"stable_baselines3.DQN": DQN, "stable_baselines3.PPO": PPO,
                               "stable_baselines3.SAC": SAC, "stable_baselines3.TD3": TD3}
            if algorithm == "sb3_contrib.MaskablePPO":
                from sb3_contrib import MaskablePPO
                implementations[algorithm] = MaskablePPO
            implementation = implementations.get(algorithm)
            if implementation is None:
                raise ValueError("unsupported V8 neural policy algorithm")
            return implementation.load(path, device="cpu")

    def _inference(self, report: dict, pointer: dict, summary: dict) -> dict:
        from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8Env, ShoreBESSV8SACEnv
        from scripts.train_shore_bess_v8 import joined_forward

        cfg, manifest = report["config"], report["manifest"]
        history = load_port_dataset(manifest["dataset_id"])
        if history.fingerprint != manifest["dataset_sha256"]:
            raise ValueError("V8 inference dataset hash mismatch")
        if summary["dataset_id"] == history.dataset_id:
            if summary["dataset_sha256"] != history.fingerprint:
                raise ValueError("V8 historical inference dataset hash mismatch")
            source = history
        else:
            if report["status"] == "VALIDATION_REJECTED":
                raise ValueError("V8 validation-only evidence cannot load a forward dataset")
            forward = load_port_dataset(summary["dataset_id"])
            if forward.fingerprint != summary["dataset_sha256"]:
                raise ValueError("V8 forward inference dataset hash mismatch")
            source = joined_forward(history, forward)
        split = summary["split"]
        train = manifest["train"]
        discrete = cfg["discrete"]
        env_cls = ShoreBESSV8Env if discrete else ShoreBESSV8SACEnv
        env = env_cls(source, slice(split["start_row"], split["stop_row_exclusive"]),
                            config=cfg["physical_config"], normalization_slice=slice(train["start_row"], train["stop_row_exclusive"]),
                            episode_steps=cfg["episode_hours"],
                            carbon_price=cfg["carbon_price_cny_per_kg_constraint_multiplier"],
                            discrete=discrete, training=False, record_trace=False)
        try:
            observation, reset = env.reset(options={"start_index": summary["starts"][0]})
            if len(observation) != cfg["observation_dimensions"]:
                raise ValueError("V8 observation contract has changed")
            for name, expected in cfg["normalization"].items():
                if not np.isclose(float(getattr(env, name)), expected, rtol=1e-12, atol=1e-12):
                    raise ValueError("V8 normalization differs from frozen training: " + name)
            loaded_path = self._verify(pointer["model_path"], pointer["model_sha256"])
            actor = self._actor(str(loaded_path), file_sha256(loaded_path), cfg["algorithm"])
            selected = next(row for row in report["results"] if row["seed"] == report["selected_seed"])["selected"]
            counters = verify_checkpoint_counters(actor, selected)
            predict_options = {"action_masks": env.action_masks()} if cfg["algorithm"] == "sb3_contrib.MaskablePPO" else {}
            action, _ = actor.predict(observation, deterministic=True, **predict_options)
            _, reward, _, _, info = env.step(action)
            return {"policy_loaded": True, "policy_admitted_for_public_offline": report["status"] == "ADMITTED_OFFLINE_RL",
                    "algorithm": cfg["algorithm"], "decision_source": "trained_rl_actor_plus_safety_projection",
                    "model_path": pointer["model_path"], "model_sha256": pointer["model_sha256"],
                    "selected_seed": report["selected_seed"], "reset": reset,
                    **counters,
                    "observation_vector": observation.tolist(),
                    "action_index": int(np.asarray(action).item()) if discrete else None,
                    "neural_action": np.asarray(action).tolist(),
                    "reward": float(reward), **info, **BOUNDARY}
        finally:
            env.close()
