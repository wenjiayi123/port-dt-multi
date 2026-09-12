from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.rl_training.datasets import file_sha256
from app.services.shore_bess_v8_evidence import REQUIRED_CHECKS, ShoreBESSV8EvidenceService, verify_checkpoint_counters
from scripts.train_shore_bess_v8 import business_gates, compare, convergence, official_source_period, source_period_summary


class ShoreBESSV8EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "evidence/v8/shore_bess"
        self.run = self.directory / "runs/test-formal"
        self.run.mkdir(parents=True)
        self.service = ShoreBESSV8EvidenceService(self.root)

    def write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def relative(self, path):
        return str(path.relative_to(self.root))

    def fixture(self, *, admitted=True, carbon_worse=False):
        # Deliberately synthetic fixtures test integrity decisions, not training
        # quality. Real model loading is covered separately by runtime replay.
        seeds = [12, 13, 14]
        cfg = {"pilot": False, "seeds": seeds, "algorithm": "stable_baselines3.DQN"}
        source = self.root / "app/test_source.py"
        source.parent.mkdir(parents=True)
        source.write_text("fixture only\n")
        snapshot = self.run / "source" / self.relative(source)
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(source.read_bytes())
        manifest = {"run_id": "test-formal", "teacher_actions_used": False,
                    "warm_start_used": False, "normalization_fit_split": "train_only",
                    "source_sha256": {self.relative(source): file_sha256(source)},
                    "historical_pointer_sha256": {}, "dataset_id": "synthetic",
                    "dataset_sha256": "1" * 64, "train": {}, "validation": {}}
        safety = {name: 0.0 for name in (
            "guardrail_violation_rate", "terminal_soc_error", "terminal_flex_backlog_kwh",
            "shore_sla_violation_kwh", "reserve_shortfall_kwh", "flex_deadline_violation_kwh",
            "physical_power_violations", "max_flex_age_hours")}
        metric = {**safety, "total_cost_cny": 100.0, "carbon_kg": 60.0 if carbon_worse else 50.0,
                  "peak_kw": 10.0, "learning_reward": 2.0}
        base = {**safety, "total_cost_cny": 110.0, "carbon_kg": 55.0, "peak_kw": 11.0, "learning_reward": 0.0}
        starts = [0, 744, 1464]
        origin = datetime(2025, 8, 1, tzinfo=timezone.utc)
        windows = [{"start_index": start, "first_timestamp": (origin + timedelta(hours=start)).isoformat(),
                    "last_timestamp": (origin + timedelta(hours=start + 167)).isoformat(),
                    "source_month": (origin + timedelta(hours=start)).strftime("%Y-%m"),
                    "source_period": official_source_period((origin + timedelta(hours=start)).isoformat())} for start in starts]
        evaluation = {"rows": [dict(metric) for _ in range(3)], "mean": dict(metric),
                      "starts": starts, "episode_hours": 168, "windows": windows}
        reference = {"rows": [dict(base) for _ in range(3)], "mean": dict(base),
                     "starts": starts, "episode_hours": 168, "windows": windows}
        comparison = compare(evaluation, reference)
        gates = business_gates(evaluation, comparison, require_month_ci=True)
        results, curves, frozen = [], {}, {}
        for seed in seeds:
            seed_dir = self.run / f"seed_{seed}"
            seed_dir.mkdir()
            model = seed_dir / "selected_model.zip"
            model.write_bytes(f"synthetic model {seed}".encode())
            model_hash = file_sha256(model)
            row = {"step": 10, "optimizer_updates": 3, "evaluation": evaluation,
                   "comparison": comparison, "gates": gates, "optimizer": {"train/loss": 0.2},
                   "model_sha256": model_hash}
            result = {"seed": seed, "steps": 10, "optimizer_updates": 3, "model_path": self.relative(model),
                      "model_sha256": model_hash, "selected": row, "convergence": {"passed": admitted},
                      "render_calls": 0, "initial_weights_sha256": "0" * 64, "final_weights_sha256": "1" * 64}
            results.append(result)
            curve_path = seed_dir / "curve.json"
            self.write(curve_path, [row])
            curves[self.relative(curve_path)] = file_sha256(curve_path)
            frozen[str(seed)] = model_hash
        selection = {"selected_seed": seeds[0], "model_path": results[0]["model_path"],
                     "model_sha256": results[0]["model_sha256"], "selection_access_to_test": False,
                     "selection_access_to_forward": False, "per_seed_model_sha256": frozen}
        partition = {"status": "previously_used_chronological_benchmark_not_fresh_blind_data",
                     "dataset_id": "synthetic", "dataset_sha256": "1" * 64,
                     "split": {"start_row": 0, "stop_row_exclusive": 2000}, "reference": reference,
                     "per_seed": [{"seed": seed, "evaluation": evaluation, "comparison": comparison, "gates": gates} for seed in seeds]}
        checks = {name: True for name in REQUIRED_CHECKS}
        if not admitted:
            checks["test_all_seed_gates"] = checks["forward_all_seed_gates"] = False
        report = {"schema": "port-shore-bess-v8-report.v1", "run_id": "test-formal",
                  "status": "ADMITTED_OFFLINE_RL" if admitted else "CANDIDATE_NOT_ADMITTED",
                  "config": cfg, "manifest": manifest, "selection": selection,
                  "selected_seed": seeds[0], "results": results, "checks": checks, "promoted": admitted,
                  "evaluations": {"test": copy.deepcopy(partition), "forward": copy.deepcopy(partition)},
                  "total_environment_steps": 30, "total_optimizer_updates": 9,
                  "claim_boundary": "synthetic test fixture", "evidence_files_sha256": curves}
        for name, item in (("config.json", cfg), ("manifest.json", manifest), ("selection.json", selection)):
            self.write(self.run / name, item)
        self.save_report(report)
        return report

    def save_report(self, report, *, champion=False):
        report_path = self.run / "report.json"
        self.write(report_path, report)
        selection = report["selection"]
        pointer = {"run_id": report["run_id"], "status": report["status"],
                   "report_path": self.relative(report_path), "report_sha256": file_sha256(report_path),
                   "model_path": selection["model_path"], "model_sha256": selection["model_sha256"]}
        self.write(self.directory / ("offline_champion.json" if champion else "latest.json"), pointer)

    def reused_fixture(self, *, rejected=True):
        report = self.fixture(admitted=not rejected, carbon_worse=rejected)
        seeds = report["config"]["seeds"]
        run_ids = [f"source-pilot-{seed}" for seed in seeds]
        report["config"]["reused_training_run_ids"] = run_ids
        manifest = report["manifest"]
        manifest.update(training_reused=True, versions={"fixture": "1"},
                        validation={"start_row": 100, "stop_row_exclusive": 2308, "rows": 2208},
                        validation_starts=list(range(0, 13 * 168, 168)))
        reference = copy.deepcopy(report["evaluations"]["test"]["reference"])
        evaluation = copy.deepcopy(report["results"][0]["selected"]["evaluation"])
        origin_time = datetime(2025, 5, 1, tzinfo=timezone.utc)
        windows = [{"start_index": start, "first_timestamp": (origin_time + timedelta(hours=start)).isoformat(),
                    "last_timestamp": (origin_time + timedelta(hours=start + 167)).isoformat(),
                    "source_month": (origin_time + timedelta(hours=start)).strftime("%Y-%m"),
                    "source_period": official_source_period((origin_time + timedelta(hours=start)).isoformat())}
                   for start in manifest["validation_starts"]]
        for data in (reference, evaluation):
            data.update(starts=manifest["validation_starts"], windows=windows,
                        rows=[copy.deepcopy(data["rows"][0]) for _ in windows])
        validation_path = self.run / "validation_reference.json"
        self.write(validation_path, reference)
        sources, preserved, formal_anchors = [], {}, {}
        for result, run_id in zip(report["results"], run_ids, strict=True):
            seed = result["seed"]
            source_dir = self.directory / "runs" / run_id
            source_seed = source_dir / f"seed_{seed}"
            source_seed.mkdir(parents=True)
            formal_seed = self.run / f"seed_{seed}"
            current_curve, old_curve, links = [], [], []
            for index in range(1, 4):
                step = index * 10
                old_model, formal_model = source_seed / f"step_{step}.zip", formal_seed / f"step_{step}.zip"
                old_model.write_bytes(f"fixture checkpoint {seed} {step}".encode())
                formal_model.write_bytes(old_model.read_bytes())
                row = {"step": step, "optimizer_updates": index * 3, "sb3_update_counter": index,
                       "model_path": self.relative(formal_model), "model_sha256": file_sha256(formal_model),
                       "weights_sha256": str(index) * 64, "parameters": {"gamma": 1.0},
                       "evaluation": copy.deepcopy(evaluation), "comparison": compare(evaluation, reference),
                       "optimizer": {"train/loss": 0.2}, "source_training_run_id": run_id}
                row["gates"] = business_gates(row["evaluation"], row["comparison"], require_month_ci=True)
                current_curve.append(row)
                old = {**copy.deepcopy(row), "model_path": self.relative(old_model)}
                old["gates"] = business_gates(old["evaluation"], old["comparison"])
                old_curve.append(old)
                links.append({"step": step, "source_model_path": old["model_path"],
                              "source_model_sha256": old["model_sha256"], "source_weights_sha256": old["weights_sha256"],
                              "formal_model_path": row["model_path"], "formal_model_sha256": row["model_sha256"]})
            selected_model = formal_seed / "selected_model.zip"
            selected_model.write_bytes((self.root / current_curve[0]["model_path"]).read_bytes())
            result.update(steps=30, optimizer_updates=9, sb3_update_counter=3, selected=current_curve[0],
                          model_sha256=file_sha256(selected_model), convergence=convergence(current_curve),
                          parameters={"gamma": 1.0}, final_weights_sha256="3" * 64,
                          source_training_run_id=run_id, training_reused=True)
            self.write(formal_seed / "curve.json", current_curve)
            source_cfg = {**report["config"], "seeds": [seed], "pilot": True}
            del source_cfg["reused_training_run_ids"]
            source_manifest = {**copy.deepcopy(manifest), "run_id": run_id,
                               "test_status": "not_opened", "forward_status": "not_loaded"}
            source_manifest.pop("training_reused")
            source_result = {**copy.deepcopy(result), "selected": old_curve[0],
                             "model_path": old_curve[0]["model_path"]}
            source_result.pop("training_reused")
            source_result.pop("source_training_run_id")
            source_selection = {"model_path": old_curve[0]["model_path"]}
            for name, value in (("config.json", source_cfg), ("manifest.json", source_manifest),
                                ("selection.json", source_selection)):
                self.write(source_dir / name, value)
            self.write(source_seed / "curve.json", old_curve)
            self.write(source_seed / "result.json", source_result)
            self.write(source_seed / "initial_validation.json", {"weights_sha256": result["initial_weights_sha256"]})
            (source_seed / "monitor.csv").write_text("r,l,t\n0,168,1\n")
            source_anchors = {self.relative(path): file_sha256(path) for path in source_dir.rglob("*") if path.is_file()}
            source_report = {"schema": report["schema"], "run_id": run_id, "status": "PILOT_VALIDATION_ONLY",
                             "config": source_cfg, "manifest": source_manifest, "results": [source_result],
                             "checks": {name: True for name in report["checks"]} | {"config_and_dataset_files_unchanged": True},
                             "selection": source_selection, "evaluations": {}, "input_files_sha256": {},
                             "evidence_files_sha256": source_anchors}
            source_report_path = source_dir / "report.json"
            self.write(source_report_path, source_report)
            preserved.update(source_anchors)
            preserved[self.relative(source_report_path)] = file_sha256(source_report_path)
            sources.append({"run_id": run_id, "report_path": self.relative(source_report_path),
                            "report_sha256": file_sha256(source_report_path), "seed": seed,
                            "environment_steps": 30, "optimizer_updates": 9, "sb3_update_counter": 3,
                            "config_path": self.relative(source_dir / "config.json"),
                            "config_sha256": file_sha256(source_dir / "config.json"),
                            "manifest_path": self.relative(source_dir / "manifest.json"),
                            "manifest_sha256": file_sha256(source_dir / "manifest.json"),
                            "initial_weights_sha256": result["initial_weights_sha256"], "checkpoints": links})
        selection = report["selection"]
        selection.update(model_sha256=report["results"][0]["model_sha256"],
                         per_seed_model_sha256={str(row["seed"]): row["model_sha256"] for row in report["results"]})
        for name, value in (("config.json", report["config"]), ("manifest.json", manifest), ("selection.json", selection)):
            self.write(self.run / name, value)
        report.update(report_kind="formal_admission_of_immutable_pilot_checkpoints", training_reused=True,
                      status="VALIDATION_REJECTED" if rejected else "ADMITTED_OFFLINE_RL",
                      source_training_runs=sources, source_training_files_sha256=preserved, input_files_sha256={},
                      new_training_environment_steps=0, new_optimizer_updates=0,
                      reused_training_environment_steps=90, total_environment_steps=90, total_optimizer_updates=27,
                      selection_sha256=file_sha256(self.run / "selection.json"),
                      training_accounting={"count_as_new_training_run": False, "environment_steps_executed_by_this_run": 0,
                                           "optimizer_updates_executed_by_this_run": 0,
                                           "referenced_environment_steps": 90, "referenced_optimizer_updates": 27},
                      promotion={"candidate_qualified": not rejected, "promoted": not rejected})
        report["checks"].update(all_seeds_validation_passed=not rejected, all_seeds_last_three_stable=not rejected,
                                source_training_evidence_preserved=True, frozen_selection_preserved=True,
                                config_and_dataset_files_unchanged=True)
        if rejected:
            report.update(evaluations={}, holdout_access={"test": "not_opened_validation_rejected",
                                                        "forward": "not_loaded_validation_rejected"})
        report["evidence_files_sha256"] = {self.relative(path): file_sha256(path) for path in self.run.rglob("*")
                                           if path.is_file() and path.name != "report.json" and "source" not in path.relative_to(self.run).parts}
        self.save_report(report)
        return report

    def diagnostic_fixture(self):
        report = self.fixture(admitted=False, carbon_worse=True)
        result = report["results"][0]
        report["results"] = [result]
        cfg, manifest = report["config"], report["manifest"]
        cfg.update(pilot=True, seeds=[result["seed"]], algorithm="stable_baselines3.SAC", requested_steps_per_seed=60000)
        manifest.update(test_status="not_opened", forward_status="not_loaded",
                        validation_starts=result["selected"]["evaluation"]["starts"])
        reference = report["evaluations"]["test"]["reference"]
        self.write(self.run / "validation_reference.json", reference)
        final_model = self.run / "seed_12/step_60000.zip"
        final_model.write_bytes(b"synthetic final diagnostic actor")
        final = copy.deepcopy(result["selected"])
        final.update(step=60000, optimizer_updates=80000, sb3_update_counter=40000,
                     model_path=self.relative(final_model), model_sha256=file_sha256(final_model),
                     weights_sha256="2" * 64)
        final["gates"] = business_gates(final["evaluation"], final["comparison"])
        self.write(self.run / "seed_12/curve.json", [result["selected"], final])
        result.update(steps=60000, optimizer_updates=80000, sb3_update_counter=40000, final_weights_sha256="2" * 64)
        report.update(status="PILOT_VALIDATION_ONLY", evaluations={}, total_environment_steps=60000, total_optimizer_updates=80000)
        report["checks"].update(formal_run=False, three_independent_seeds=False, config_and_dataset_files_unchanged=True)
        for name, value in (("config.json", cfg), ("manifest.json", manifest)):
            self.write(self.run / name, value)
        report["evidence_files_sha256"] = {self.relative(path): file_sha256(path) for path in self.run.rglob("*")
                                           if path.is_file() and path.name != "report.json" and "source" not in path.relative_to(self.run).parts}
        self.save_diagnostic(report)
        (self.directory / "latest.json").unlink()
        return report

    def save_diagnostic(self, report):
        report_path = self.run / "report.json"
        self.write(report_path, report)
        pointer = {"schema": "port-shore-bess-v8-diagnostics-pointer.v1", "reports": [
            {"run_id": report["run_id"], "report_path": self.relative(report_path),
             "report_sha256": file_sha256(report_path), "checkpoint_step": 60000}]}
        self.write(self.directory / "diagnostics_latest.json", pointer)

    def build_diagnostic(self, *, steps=60000):
        actor = SimpleNamespace(num_timesteps=steps, _n_updates=40000, _v8_optimizer_step_calls=80000)
        with patch.object(self.service, "_actor", return_value=actor), \
                patch("scripts.train_shore_bess_v8.weights_sha256", return_value="2" * 64), \
                patch("app.services.shore_bess_v8_evidence.load_port_dataset", side_effect=AssertionError("diagnostic loaded data")):
            return self.service.build()

    def build(self):
        with patch.object(self.service, "_inference", return_value={"policy_loaded": True}) as inference:
            output = self.service.build()
        return output, inference

    def test_absent_report_cannot_create_admission(self):
        output, inference = self.build()
        self.assertEqual(output["status"], "NO_FORMAL_V8_REPORT")
        self.assertFalse(output["admitted"])
        inference.assert_not_called()
        self.assertFalse((self.directory / "offline_champion.json").exists())

    def test_pilot_diagnostic_displays_final_checkpoint_without_formal_or_runtime_promotion(self):
        report = self.diagnostic_fixture()
        output = self.build_diagnostic()
        self.assertFalse(output["available"])
        self.assertFalse(output["admitted"])
        self.assertEqual(output["status"], "NO_FORMAL_V8_REPORT")
        diagnostic = output["diagnostics"]
        self.assertTrue(diagnostic["available"], diagnostic)
        row = diagnostic["runs"][0]
        self.assertEqual(row["checkpoint_step"], 60000)
        self.assertEqual(row["optimizer_updates"], 80000)
        self.assertNotEqual(row["model_path"], report["selection"]["model_path"])
        self.assertFalse(row["diagnostic_window_gates_passed"])
        self.assertLess(row["comparison"]["carbon_kg"]["mean"], 0)
        self.assertIn("ci_low", row["comparison"]["carbon_kg"])
        self.assertFalse(row["online_inference_performed"])
        self.assertFalse(row["heldout_evaluated"])
        self.assertFalse((self.directory / "offline_champion.json").exists())

    def test_diagnostic_rejects_heldout_access_wrong_counters_and_forged_ci(self):
        report = self.diagnostic_fixture()
        for mode in ("heldout", "model_steps", "report_steps", "ci", "model_bytes"):
            with self.subTest(mode=mode):
                changed = copy.deepcopy(report)
                curve_path = self.run / "seed_12/curve.json"
                original_curve = curve_path.read_bytes()
                model = self.run / "seed_12/step_60000.zip"
                original_model = model.read_bytes()
                if mode == "heldout":
                    changed["evaluations"] = {"test": {}}
                elif mode == "report_steps":
                    changed["total_environment_steps"] += 1
                elif mode == "ci":
                    curve = json.loads(curve_path.read_text())
                    curve[-1]["comparison"]["carbon_kg"]["ci_low"] += 1
                    self.write(curve_path, curve)
                    changed["evidence_files_sha256"][self.relative(curve_path)] = file_sha256(curve_path)
                elif mode == "model_bytes":
                    model.write_bytes(original_model + b"changed")
                self.save_diagnostic(changed)
                output = self.build_diagnostic(steps=59999 if mode == "model_steps" else 60000)
                self.assertEqual(output["diagnostics"]["status"], "INVALID_DIAGNOSTIC_EVIDENCE", output)
                self.assertFalse(output["available"])
                curve_path.write_bytes(original_curve)
                model.write_bytes(original_model)

    def test_diagnostic_source_snapshot_remains_readable_after_current_source_changes(self):
        self.diagnostic_fixture()
        (self.root / "app/test_source.py").write_text("next version of current environment\n")
        output = self.build_diagnostic()
        self.assertTrue(output["diagnostics"]["available"], output)
        self.assertFalse(output["available"])

    def test_verified_evidence_keeps_partitions_separate_and_only_annualizes_paired_hours(self):
        self.fixture()
        output, inference = self.build()
        self.assertTrue(output["available"])
        self.assertTrue(output["admitted"])
        inference.assert_called_once()
        self.assertEqual(set(output["benchmark_summaries"]), {"test", "forward"})
        summary = output["benchmark_summaries"]["forward"]
        self.assertEqual(summary["evaluated_hours"], 504)
        self.assertAlmostEqual(summary["annualized_scenario"]["cost_savings_cny"]["mean"], 10 * 8760 / 168)
        self.assertEqual(summary["annualization_hours"], 8760)
        self.assertEqual(summary["annualized_scenario"]["cost_savings_cny"]["cluster_count"], 3)
        self.assertIn("ci_low", summary["comparison"]["carbon_kg"])
        self.assertIn("clustered_ci_low", summary["comparison"]["carbon_kg"])
        self.assertTrue(output["training_process"]["raw_curves_hash_verified"])
        self.assertFalse(output["boundary"]["production_authority"])
        self.assertFalse((self.directory / "offline_champion.json").exists())

    def test_rejected_candidate_preserves_negative_carbon_without_promotion(self):
        self.fixture(admitted=False, carbon_worse=True)
        output, _ = self.build()
        self.assertTrue(output["available"])
        self.assertFalse(output["admitted"])
        self.assertLess(output["benchmark_summaries"]["forward"]["annualized_scenario"]["carbon_reduction_t"]["mean"], 0)
        self.assertFalse((self.directory / "offline_champion.json").exists())

    def test_changed_weight_source_curve_or_selection_fails_before_model_load(self):
        report = self.fixture()
        paths = [self.root / report["selection"]["model_path"], self.root / "app/test_source.py",
                 self.run / "source/app/test_source.py", self.run / "seed_12/curve.json",
                 self.run / "selection.json", self.run / "config.json"]
        for path in paths:
            with self.subTest(path=path):
                saved = path.read_bytes()
                path.write_bytes(saved + b"\nchanged")
                output, inference = self.build()
                self.assertEqual(output["status"], "INVALID_EVIDENCE")
                inference.assert_not_called()
                path.write_bytes(saved)

    def test_missing_or_truthy_nonboolean_admission_gates_fail_closed(self):
        report = self.fixture()
        for alteration in ("missing", "truthy"):
            changed = copy.deepcopy(report)
            if alteration == "missing":
                del changed["checks"]["real_optimizer_updates"]
            else:
                changed["checks"]["real_optimizer_updates"] = "true"
            self.save_report(changed)
            output, inference = self.build()
            self.assertEqual(output["status"], "INVALID_EVIDENCE")
            inference.assert_not_called()

    def test_claimed_gain_must_match_actual_paired_meter_accounting(self):
        report = self.fixture()
        report["evaluations"]["forward"]["per_seed"][0]["comparison"]["carbon_kg"]["mean"] += 2
        self.save_report(report)
        output, inference = self.build()
        self.assertEqual(output["status"], "INVALID_EVIDENCE")
        inference.assert_not_called()

    def test_champion_pointer_never_accepts_failed_candidate(self):
        report = self.fixture(admitted=False, carbon_worse=True)
        self.save_report(report, champion=True)
        output, inference = self.build()
        self.assertEqual(output["status"], "INVALID_EVIDENCE")
        inference.assert_not_called()

    def test_runtime_load_failure_cannot_present_verified_current_policy(self):
        self.fixture()
        with patch.object(self.service, "_inference", side_effect=ValueError("wrong observation shape")):
            output = self.service.build()
        self.assertEqual(output["status"], "INVALID_EVIDENCE")
        self.assertFalse(output["available"])
        self.assertFalse(output["admitted"])

    def test_reused_validation_rejection_remains_current_without_opening_heldout(self):
        self.reused_fixture()
        output, inference = self.build()
        self.assertTrue(output["available"], output)
        self.assertEqual(output["status"], "VALIDATION_REJECTED")
        self.assertFalse(output["admitted"])
        self.assertEqual(output["primary_benchmark"], "validation")
        self.assertEqual(set(output["benchmark_summaries"]), {"validation"})
        self.assertEqual(output["benchmark_summaries"]["validation"]["windows"], 13)
        self.assertEqual(output["holdout_access"]["forward"], "not_loaded_validation_rejected")
        self.assertTrue(output["training_reused"])
        self.assertEqual(output["total_environment_steps"], 90)
        self.assertEqual(output["new_training_environment_steps"], 0)
        self.assertEqual(output["new_optimizer_updates"], 0)
        self.assertEqual(inference.call_args.args[2]["status"], "validation_only_rejected_before_heldout")
        self.assertFalse((self.directory / "offline_champion.json").exists())

    def test_reused_training_cannot_change_origin_or_count_updates_twice(self):
        report = self.reused_fixture()
        for mode in ("counter", "model", "report", "new_steps", "accounting", "heldout"):
            with self.subTest(mode=mode):
                changed = copy.deepcopy(report)
                if mode == "counter":
                    changed["source_training_runs"][0]["optimizer_updates"] += 1
                elif mode == "model":
                    changed["source_training_runs"][0]["checkpoints"][0]["source_model_sha256"] = "a" * 64
                elif mode == "report":
                    changed["source_training_runs"][0]["report_sha256"] = "a" * 64
                elif mode == "new_steps":
                    changed["new_training_environment_steps"] = 90
                elif mode == "accounting":
                    changed["training_accounting"]["count_as_new_training_run"] = True
                else:
                    changed["holdout_access"]["forward"] = "evaluated_after_full_validation_selection_freeze"
                self.save_report(changed)
                output, inference = self.build()
                self.assertEqual(output["status"], "INVALID_EVIDENCE", output)
                inference.assert_not_called()

    def test_qualified_reused_candidate_may_preserve_existing_champion(self):
        report = self.reused_fixture(rejected=False)
        report["promoted"] = report["promotion"]["promoted"] = False
        self.save_report(report)
        output, _ = self.build()
        self.assertTrue(output["available"], output)
        self.assertTrue(output["admitted"])
        self.assertFalse((self.directory / "offline_champion.json").exists())

    def test_validation_only_runtime_loads_only_historical_dataset(self):
        import numpy as np
        from tests.test_shore_bess_v8 import fixture_dataset
        from app.services.rl_model.shore_bess.v8_environment import LATTICE, ShoreBESSV8Env
        dataset = fixture_dataset()
        train = slice(0, 168)
        env = ShoreBESSV8Env(dataset, slice(168, 400), normalization_slice=train, episode_steps=168)
        cfg = {"physical_config": env.config, "discrete": True, "episode_hours": 168,
               "carbon_price_cny_per_kg_constraint_multiplier": env.carbon_price,
               "observation_dimensions": env.observation_space.shape[0], "normalization": {},
               "algorithm": "stable_baselines3.DQN"}
        idle = int(np.flatnonzero(np.all(LATTICE == 0, axis=1))[0])
        env.close()
        actor = SimpleNamespace(num_timesteps=10, _n_updates=3, predict=lambda *_args, **_kwargs: (idle, None))
        report = {"config": cfg, "manifest": {"dataset_id": dataset.dataset_id, "dataset_sha256": dataset.fingerprint,
                  "train": {"start_row": 0, "stop_row_exclusive": 168}}, "status": "VALIDATION_REJECTED",
                  "selected_seed": 12, "results": [{"seed": 12, "selected": {"step": 10, "optimizer_updates": 3}}]}
        fixture = self.service.root / "fixture.zip"
        fixture.write_bytes(b"isolated mocked actor artifact")
        pointer = {"model_path": "fixture.zip", "model_sha256": file_sha256(fixture)}
        summary = {"dataset_id": dataset.dataset_id, "dataset_sha256": dataset.fingerprint,
                   "split": {"start_row": 168, "stop_row_exclusive": 400}, "starts": [0]}
        with patch("app.services.shore_bess_v8_evidence.load_port_dataset", return_value=dataset) as loader, \
                patch.object(self.service, "_actor", return_value=actor), \
                patch("scripts.train_shore_bess_v8.joined_forward", side_effect=AssertionError("heldout loaded")):
            output = self.service._inference(report, pointer, summary)
        self.assertTrue(output["policy_loaded"])
        loader.assert_called_once_with(dataset.dataset_id)

    def test_pointer_paths_cannot_read_outside_repository(self):
        self.write(self.directory / "latest.json", {"report_path": "../outside.json", "report_sha256": "0" * 64})
        output, inference = self.build()
        self.assertEqual(output["status"], "INVALID_EVIDENCE")
        inference.assert_not_called()

    def test_pytorch_optimizer_calls_are_distinguished_from_sb3_update_counter(self):
        actor = SimpleNamespace(num_timesteps=10000, _n_updates=8000, _v8_optimizer_step_calls=16000)
        selected = {"step": 10000, "sb3_update_counter": 8000, "optimizer_updates": 16000}
        receipt = verify_checkpoint_counters(actor, selected)
        self.assertEqual(receipt["loaded_checkpoint_optimizer_updates"], 16000)
        self.assertEqual(receipt["loaded_checkpoint_sb3_update_counter"], 8000)
        for key in selected:
            with self.subTest(key=key):
                bad = {**selected, key: selected[key] + 1}
                with self.assertRaisesRegex(ValueError, "counters differ"):
                    verify_checkpoint_counters(actor, bad)

    def test_source_month_ci_numbers_counts_and_labels_cannot_be_fabricated(self):
        report = self.fixture()
        for mode in ("interval", "count", "label", "period", "group_field"):
            with self.subTest(mode=mode):
                changed = copy.deepcopy(report)
                row = changed["evaluations"]["forward"]["per_seed"][0]
                if mode == "interval":
                    row["comparison"]["carbon_kg"]["clustered_ci_low"] += 1
                elif mode == "count":
                    row["comparison"]["carbon_kg"]["cluster_count"] += 1
                elif mode == "label":
                    row["evaluation"]["windows"][0]["source_month"] = "2024-01"
                elif mode == "period":
                    row["evaluation"]["windows"][0]["source_period"] = "2024-01/02"
                else:
                    row["comparison"]["carbon_kg"]["cluster_group_field"] = "source_month"
                self.save_report(changed)
                output, inference = self.build()
                self.assertEqual(output["status"], "INVALID_EVIDENCE")
                inference.assert_not_called()

    def test_omitting_formal_month_cluster_gate_fails_closed(self):
        report = self.fixture()
        del report["evaluations"]["forward"]["per_seed"][0]["gates"]["carbon_kg_positive_source_period_95ci"]
        self.save_report(report)
        output, inference = self.build()
        self.assertEqual(output["status"], "INVALID_EVIDENCE")
        inference.assert_not_called()

    def test_annualized_cluster_ci_uses_same_8760_scaling_and_official_periods(self):
        report = self.fixture()
        partition = copy.deepcopy(report["evaluations"]["forward"])
        chosen = partition["per_seed"][0]
        for i, row in enumerate(chosen["evaluation"]["rows"]):
            row["total_cost_cny"] -= i * 3
        chosen["evaluation"]["mean"]["total_cost_cny"] = sum(row["total_cost_cny"] for row in chosen["evaluation"]["rows"]) / len(chosen["evaluation"]["rows"])
        chosen["comparison"] = compare(chosen["evaluation"], partition["reference"])
        summary = self.service._summary(partition, chosen)
        annual = summary["annualized_scenario"]["cost_savings_cny"]
        gains = [(base["total_cost_cny"] - row["total_cost_cny"]) * 8760 / 168 for base, row in zip(partition["reference"]["rows"], chosen["evaluation"]["rows"], strict=True)]
        expected = source_period_summary(gains, [window["source_period"] for window in chosen["evaluation"]["windows"]])
        self.assertEqual(annual["clustered_ci_low"], expected["clustered_ci_low"])
        self.assertEqual(annual["clustered_ci_high"], expected["clustered_ci_high"])
        self.assertEqual(annual["cluster_group_field"], "source_period")
        self.assertIn("ci_low", annual)


if __name__ == "__main__":
    unittest.main()
