"""Tamper checks independent of training, validation selection and UI code."""
from __future__ import annotations

import hashlib
import copy
import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from scripts.verify_shore_bess_v8 import (
    VerificationError, anchored_path, archive_proof, comparisons, convergence,
    declared_business_gates, same, validate_evaluation, validate_split,
    verify_hash, verify_optimizer_proof, verify_zero_reference, source_period_clusters,
    official_source_period, contained_period_sensitivity, verify_window_provenance,
    verify_reused_accounting, verify_reused_training,
    verify_n_step_snapshot, verify_training_ledger, LEDGER_PHASE,
)

GATE = {
    "cost_reduction_percent_min": .03, "carbon_reduction_percent_min": .001,
    "peak_reduction_percent_min": 0., "all_three_95ci_lower_bounds_strictly_positive": True,
    "every_window_carbon_non_regression": True, "terminal_absolute_tolerance": 1e-6,
    "safety_tolerance": 1e-12, "tail_relative_range_max": .25,
    "tail_range_floors_pp": {"total_cost_cny": .02, "carbon_kg": .001, "peak_kw": .05},
}


def evaluation(costs=(100., 101., 99.), carbon=(100., 100., 100.), peaks=(10., 10., 10.)):
    rows = []
    for cost, co2, peak in zip(costs, carbon, peaks, strict=True):
        rows.append({"total_cost_cny": cost, "carbon_kg": co2, "peak_kw": peak,
                     "guardrail_violation_rate": 0., "terminal_soc_error": 0.,
                     "terminal_flex_backlog_kwh": 0., "shore_sla_violation_kwh": 0.,
                     "reserve_shortfall_kwh": 0., "flex_deadline_violation_kwh": 0.,
                     "physical_power_violations": 0., "max_flex_age_hours": 12.})
    return {"rows": rows, "starts": [0, 168, 336], "episode_hours": 168, "window_overlap": False,
            "mean": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}}


class IndependentVerifierTests(unittest.TestCase):
    @staticmethod
    def ledger_fixture(root):
        seed = root / "seed_1"
        seed.mkdir()
        path = seed / "training_episodes.csv"
        metrics = dict(evaluation()["rows"][0])
        metrics.update(energy_cost_cny=90., degradation_cost_cny=1., demand_charge_cny=9.,
                       financial_delta_cny=-3., carbon_delta_kg=-.1, training_reward=.021,
                       guardrail_violations=0., projection_count=0., projection_rate=0.,
                       nonzero_bess_actions=0., nonzero_bess_action_rate=0., nonzero_flex_actions=0.,
                       nonzero_flex_action_rate=0., soc_min_observed=.58, soc_max_observed=.58, temperature_max_c=28.)
        rows = []
        for i in (1, 2):
            updates = i * 168 - 1 - 100
            rows.append({"episode_index": i, "environment_steps": i * 168,
                "optimizer_step_calls_at_observation": updates + updates // 2,
                "sb3_update_counter_at_observation": updates, "record_phase": LEDGER_PHASE,
                **{"episode_metrics." + key: value for key, value in metrics.items()}})
        config = {"algorithm": "stable_baselines3.TD3", "episode_hours": 168,
                  "carbon_price_cny_per_kg_constraint_multiplier": 12., "normalization": {"reward_scale": 200.}}
        result = {"seed": 1, "steps": 350, "optimizer_updates": 375, "sb3_update_counter": 250,
                  "parameters": {"train_freq": {"frequency": 1, "unit": "step"}, "learning_starts": 100,
                                 "gradient_steps": 1, "policy_delay": 2},
                  "training_episode_ledger": {"path": "seed_1/training_episodes.csv", "completed_episodes": 2,
                    "record_phase": LEDGER_PHASE, "partial_final_episode_included": False,
                    "metrics_source": "unaltered numeric environment info.episode_metrics from physical training rollouts"}}
        def save():
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            result["training_episode_ledger"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (seed / "monitor.csv").write_text('#{}\nr,l,t\n' + ''.join(
                f'{row["episode_metrics.training_reward"]:.6f},168,{i}\n' for i, row in enumerate(rows)))
        save()
        return result, config, {"training_episode_ledger_contract": LEDGER_PHASE}, seed, rows, save

    def test_training_ledger_reconstructs_pre_optimizer_counts_and_complete_episode_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config, manifest, seed, rows, save = self.ledger_fixture(root)
            audit = verify_training_ledger(result, config, manifest, seed, root)
            self.assertEqual(audit["completed_episodes"], 2)
            self.assertEqual(audit["remaining_partial_episode_steps"], 14)
            self.assertEqual(audit["unsafe_completed_episode_indices"], [])
            self.assertTrue(audit["exact_pre_optimizer_counters_recomputed"])
            rows[1]["optimizer_step_calls_at_observation"] += 1; save()
            with self.assertRaisesRegex(VerificationError, "pre-optimizer phase"):
                verify_training_ledger(result, config, manifest, seed, root)
            rows[1]["optimizer_step_calls_at_observation"] -= 1
            rows[1]["environment_steps"] += 1; save()
            with self.assertRaisesRegex(VerificationError, "not contiguous"):
                verify_training_ledger(result, config, manifest, seed, root)

    def test_training_ledger_rejects_forged_objective_and_preserves_honest_unsafe_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, config, manifest, seed, rows, save = self.ledger_fixture(root)
            rows[0]["episode_metrics.training_reward"] += 1.; save()
            with self.assertRaisesRegex(VerificationError, "telescoping"):
                verify_training_ledger(result, config, manifest, seed, root)
            rows[0]["episode_metrics.training_reward"] = .021 - 100.
            rows[0]["episode_metrics.physical_power_violations"] = 1.
            rows[0]["episode_metrics.guardrail_violations"] = 1.
            rows[0]["episode_metrics.guardrail_violation_rate"] = 1. / 168; save()
            audit = verify_training_ledger(result, config, manifest, seed, root)
            self.assertEqual(audit["unsafe_completed_episode_indices"], [1])
            self.assertFalse(audit["admission_reinterpreted"])
            (seed / "monitor.csv").write_text('#{}\nr,l,t\n0,168,1\n0,168,2\n')
            with self.assertRaisesRegex(VerificationError, "Monitor return"):
                verify_training_ledger(result, config, manifest, seed, root)

    @staticmethod
    def reused_fixture(root):
        formal = root / "formal"
        formal.mkdir()
        report = {"config": {"pilot": False, "seeds": [1, 2, 3], "algorithm": "TD3",
                             "reused_training_run_ids": ["pilot1", "pilot2", "pilot3"]},
                  "manifest": {"training_reused": True, "started_at": "2026-09-12T00:00:00Z",
                               "versions": {}, "train": {}, "validation": {}, "dataset_id": "pinned",
                               "dataset_sha256": "a" * 64, "input_files_sha256": {},
                               "source_sha256": {"trainer.py": "b" * 64, "scripts/admit_shore_bess_v8.py": "c" * 64}},
                  "training_reused": True, "new_training_environment_steps": 0, "new_optimizer_updates": 0,
                  "total_environment_steps": 30, "total_optimizer_updates": 12, "reused_training_environment_steps": 30,
                  "training_accounting": {"count_as_new_training_run": False, "environment_steps_executed_by_this_run": 0,
                     "optimizer_updates_executed_by_this_run": 0, "referenced_environment_steps": 30, "referenced_optimizer_updates": 12},
                  "results": [], "source_training_runs": [], "source_training_files_sha256": {}}

        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return hashlib.sha256(path.read_bytes()).hexdigest()

        for seed in (1, 2, 3):
            source_dir, formal_seed = root / f"pilot{seed}", formal / f"seed_{seed}"
            source_seed = source_dir / f"seed_{seed}"
            source_seed.mkdir(parents=True); formal_seed.mkdir()
            old_path, new_path = source_seed / "step_10.zip", formal_seed / "step_10.zip"
            old_path.write_bytes(f"immutable original archive {seed}".encode())
            new_path.write_bytes(old_path.read_bytes())
            sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
            old = {"step": 10, "optimizer_updates": 4, "sb3_update_counter": 3, "weights_sha256": str(seed) * 64,
                   "parameters": {}, "model_path": str(old_path.relative_to(root)), "model_sha256": sha}
            new = {**old, "model_path": str(new_path.relative_to(root)), "source_training_run_id": f"pilot{seed}"}
            save(source_seed / "curve.json", [old]); save(formal_seed / "curve.json", [new])
            save(formal_seed / "initial_validation.json", {"reconstructed_from_source_seed_without_learning": True})
            for directory in (source_seed, formal_seed):
                (directory / "monitor.csv").write_text("same original training rows\n")
            original = {"seed": seed, "steps": 10, "optimizer_updates": 4, "sb3_update_counter": 3,
                        "initial_weights_sha256": str(seed + 3) * 64, "final_weights_sha256": str(seed) * 64,
                        "parameters": {}, "render_calls": 0}
            source_manifest = {key: report["manifest"][key] for key in
                               ("versions", "train", "validation", "dataset_id", "dataset_sha256", "input_files_sha256")}
            source_manifest["source_sha256"] = {"trainer.py": "b" * 64}
            source = {"run_id": f"pilot{seed}", "config": {"pilot": True, "seeds": [seed], "algorithm": "TD3"},
                      "manifest": source_manifest, "input_files_sha256": {}, "results": [original],
                      "generated_at": "2026-09-11T00:00:00Z", "status": "PILOT_VALIDATION_ONLY", "evaluations": {}}
            source["checks"] = {key: True for key in ("real_optimizer_updates", "weights_changed", "source_unchanged_during_training",
                "config_and_dataset_files_unchanged", "historical_pointers_preserved", "no_training_rendering")}
            config_sha = save(source_dir / "config.json", source["config"])
            manifest_sha = save(source_dir / "manifest.json", source_manifest)
            source["evidence_files_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in source_dir.rglob("*") if path.is_file()}
            report_sha = save(source_dir / "report.json", source)
            report["results"].append({**original, "source_training_run_id": f"pilot{seed}", "training_reused": True})
            origin = {"run_id": source["run_id"], "seed": seed, "report_path": f"pilot{seed}/report.json", "report_sha256": report_sha,
                      "config_path": f"pilot{seed}/config.json", "config_sha256": config_sha,
                      "manifest_path": f"pilot{seed}/manifest.json", "manifest_sha256": manifest_sha,
                      "environment_steps": 10, "optimizer_updates": 4, "sb3_update_counter": 3,
                      "initial_weights_sha256": original["initial_weights_sha256"],
                      "checkpoints": [{"step": 10, "source_model_path": old["model_path"], "source_model_sha256": sha,
                                       "source_weights_sha256": old["weights_sha256"], "formal_model_path": new["model_path"],
                                       "formal_model_sha256": sha}]}
            report["source_training_runs"].append(origin)
            report["source_training_files_sha256"][origin["report_path"]] = report_sha
            report["source_training_files_sha256"].update(source["evidence_files_sha256"])
        return report, formal

    def test_immutable_adoption_never_counts_reused_work_as_new_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self.reused_fixture(Path(tmp))
            verify_reused_accounting(report)
            report["new_optimizer_updates"] = 12
            with self.assertRaisesRegex(VerificationError, "new training work"):
                verify_reused_accounting(report)
            report["new_optimizer_updates"] = 0
            report["training_accounting"]["count_as_new_training_run"] = True
            with self.assertRaisesRegex(VerificationError, "count_as_new_training_run"):
                verify_reused_accounting(report)

    def test_adoption_binds_all_original_checkpoint_bytes_counts_and_source_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, formal = self.reused_fixture(root)
            def verified_source(path, **_):
                return {"integrity_status": "PASS", "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            # This unit isolates the provenance join. Real archive validation is
            # independently exercised by the SAC/TD3 optimizer tests below.
            with patch("scripts.verify_shore_bess_v8.verify_report", side_effect=verified_source) as verifier:
                verified = verify_reused_training(report, formal, root)
                self.assertEqual(len(verified), 3)
                self.assertEqual(verifier.call_count, 3)
                forged = copy.deepcopy(report)
                forged["source_training_runs"][0]["optimizer_updates"] += 1
                with self.assertRaisesRegex(VerificationError, "source training accounting"):
                    verify_reused_training(forged, formal, root)
                forged = copy.deepcopy(report)
                forged["source_training_files_sha256"].pop("pilot1/seed_1/monitor.csv")
                with self.assertRaisesRegex(VerificationError, "exhaustive original training file"):
                    verify_reused_training(forged, formal, root)
                (formal / "seed_1" / "step_10.zip").write_bytes(b"additional secretly trained weights")
                with self.assertRaisesRegex(VerificationError, "SHA-256 mismatch"):
                    verify_reused_training(report, formal, root)

    def test_january_february_use_one_official_reporting_period(self):
        self.assertEqual(official_source_period("2026-01-29T00:00:00Z"), "2026-01/02")
        self.assertEqual(official_source_period("2026-02-04T23:00:00Z"), "2026-01/02")
        self.assertEqual(official_source_period("2026-03-01T00:00:00Z"), "2026-03")
        with self.assertRaisesRegex(VerificationError, "not UTC"):
            official_source_period("2026-02-01T00:00:00+08:00")

    def test_cluster_bootstrap_keeps_all_rows_of_unequal_sized_periods(self):
        result = source_period_clusters([1., 1., 1., -3.], ["2026-01/02"] * 3 + ["2026-03"])
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["cluster_window_counts"], {"2026-01/02": 3, "2026-03": 1})
        self.assertEqual(result["clustered_ci_low"], -3.)
        self.assertEqual(result["clustered_ci_high"], 1.)
        self.assertEqual(result["cluster_group_field"], "source_period")

    def test_fake_period_labels_or_fake_cluster_ci_are_rejected(self):
        ev = evaluation()
        base = datetime(2026, 1, 22)
        ev["windows"] = []
        for start in ev["starts"]:
            first = (base + timedelta(hours=start)).isoformat() + "Z"
            last = (base + timedelta(hours=start + 167)).isoformat() + "Z"
            ev["windows"].append({"start_index": start, "first_timestamp": first, "last_timestamp": last,
                                   "source_month": first[:7], "source_period": official_source_period(first)})
        validate_evaluation(ev)
        timestamps = [(base + timedelta(hours=i)).isoformat() + "Z" for i in range(505)]
        verify_window_provenance(ev, timestamps, 0)
        forged_windows = json.loads(json.dumps(ev))
        forged_windows["windows"][0]["first_timestamp"] = "2026-01-22T01:00:00Z"
        with self.assertRaisesRegex(VerificationError, "pinned dataset"):
            verify_window_provenance(forged_windows, timestamps, 0)
        comp = comparisons(ev, ev)
        self.assertEqual(comp["carbon_kg"]["cluster_count"], 1)
        forged = json.loads(json.dumps(comp)); forged["carbon_kg"]["clustered_ci_low"] = 1.
        with self.assertRaisesRegex(VerificationError, "numeric mismatch"):
            same(forged, comparisons(ev, ev), "source period CI")
        ev["windows"][-1]["source_period"] = "2026-02"
        with self.assertRaisesRegex(VerificationError, "source-period label"):
            validate_evaluation(ev)

    def test_contained_period_sensitivity_keeps_jan_feb_and_excludes_only_cross_period_ci_rows(self):
        ev = evaluation()
        starts = ["2026-01-29T00:00:00Z", "2026-02-26T00:00:00Z", "2026-03-05T00:00:00Z"]
        ev["windows"] = [{"first_timestamp": first,
                           "last_timestamp": (datetime.fromisoformat(first[:-1]) + timedelta(hours=167)).isoformat() + "Z"}
                          for first in starts]
        result = contained_period_sensitivity(ev, ev)
        self.assertEqual(result["included_start_indices"], [0, 336])
        self.assertEqual(result["cross_period_start_indices_excluded_from_sensitivity_only"], [168])
        self.assertTrue(result["all_business_windows_retained"])
        self.assertFalse(result["admission_reinterpreted"])
        self.assertEqual(len(ev["rows"]), 3)

    def test_cluster_formal_gate_cannot_pass_with_too_few_sources(self):
        baseline = evaluation()
        candidate = evaluation(costs=(90., 91., 89.), carbon=(99., 98., 99.), peaks=(9., 9., 9.))
        comp = comparisons(candidate, baseline)
        for metric in comp:
            comp[metric].update(clustered_ci_low=.01, cluster_count=2)
        gate = {**GATE, "formal_source_period_cluster_95ci_lower_bounds_strictly_positive": True,
                "formal_source_period_cluster_count_min": 3}
        checks = declared_business_gates(candidate, comp, gate, require_month_ci=True)
        self.assertTrue(checks["carbon_kg_positive_source_period_95ci"])
        self.assertFalse(checks["carbon_kg_source_period_count"])

    def test_inflated_summary_is_rejected_even_when_rows_look_valid(self):
        ev = evaluation()
        validate_evaluation(ev, split_rows=505)
        ev["mean"]["total_cost_cny"] -= 1.
        with self.assertRaisesRegex(VerificationError, "numeric mismatch"):
            validate_evaluation(ev)

    def test_overlapping_or_truncated_windows_are_rejected(self):
        ev = evaluation()
        with self.assertRaisesRegex(VerificationError, "partition boundary"):
            validate_evaluation(ev, split_rows=504)
        ev["starts"] = [0, 167, 336]
        with self.assertRaisesRegex(VerificationError, "overlap"):
            validate_evaluation(ev)

    def test_rejected_carbon_policy_has_valid_recomputable_evidence(self):
        baseline = evaluation()
        candidate = evaluation(costs=(90., 91., 89.), carbon=(101., 102., 101.), peaks=(9., 9., 9.))
        validate_evaluation(candidate)
        comparison = comparisons(candidate, baseline)
        gates = declared_business_gates(candidate, comparison, GATE)
        self.assertTrue(gates["total_cost_cny_minimum_mean_gain"])
        self.assertFalse(gates["carbon_kg_minimum_mean_gain"])
        self.assertFalse(gates["every_window_carbon_non_regression"])
        self.assertLess(comparison["carbon_kg"]["ci_high"], 0.)
        # Integrity comparison is allowed for failed admission; no metric gets flipped.
        same(gates, declared_business_gates(candidate, comparisons(candidate, baseline), GATE), "failed gates")

    def test_forged_ci_or_hidden_deadline_violation_is_detected(self):
        baseline = evaluation()
        candidate = evaluation(costs=(90., 91., 89.), carbon=(99., 98., 99.), peaks=(9., 9., 9.))
        comparison = comparisons(candidate, baseline)
        forged = json.loads(json.dumps(comparison))
        forged["carbon_kg"]["ci_low"] += 1.
        with self.assertRaisesRegex(VerificationError, "numeric mismatch"):
            same(forged, comparisons(candidate, baseline), "bootstrap")
        candidate["rows"][1]["flex_deadline_violation_kwh"] = .01
        self.assertFalse(declared_business_gates(candidate, comparison, GATE)["flex_deadline_violation_kwh"])
        candidate["rows"][1]["max_flex_age_hours"] = 13.
        self.assertFalse(declared_business_gates(candidate, comparison, GATE)["maximum_flex_age_within_12_hours"])

    def test_stable_but_unprofitable_tail_does_not_become_converged_admission(self):
        ev = evaluation()
        comp = comparisons(ev, ev)
        row = {"comparison": comp, "gates": declared_business_gates(ev, comp, GATE)}
        result = convergence([row, row, row], GATE)
        self.assertTrue(all(v["passed"] for v in result["stability"].values()))
        self.assertFalse(result["passed"])

    def test_full_file_hash_and_path_boundary_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "model.zip"
            path.write_bytes(b"first complete archive")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            verify_hash(root, "model.zip", sha)
            path.write_bytes(b"changed optimizer member")
            with self.assertRaisesRegex(VerificationError, "SHA-256 mismatch"):
                verify_hash(root, "model.zip", sha)
            with self.assertRaisesRegex(VerificationError, "escapes"):
                anchored_path(root, "../report.json")

    def test_month_boundary_and_missing_hour_are_not_accepted(self):
        start = datetime(2025, 4, 1)
        stamps = [(start + timedelta(hours=i)).isoformat() + "Z" for i in range(30 * 24)]
        description = {"start_row": 0, "stop_row_exclusive": len(stamps), "rows": len(stamps),
                       "first_timestamp": stamps[0], "last_timestamp": stamps[-1]}
        validate_split(description, stamps, "2025-04-01T00:00:00", "2025-05-01T00:00:00")
        changed = stamps.copy(); changed[123] = stamps[122]
        with self.assertRaisesRegex(VerificationError, "complete isolated source months"):
            validate_split(description, changed, "2025-04-01T00:00:00", "2025-05-01T00:00:00")

    def test_actual_optimizer_count_is_not_sb3_epoch_counter(self):
        parameters = {"gradient_optimizer_step_calls": 120, "optimizer_step_calls_by_component": {"policy": 120},
                      "_n_updates": 10, "policy_class": "ActorCriticPolicy"}
        verify_optimizer_proof(parameters, 120, 10)
        parameters["optimizer_step_calls_by_component"]["policy"] = 10
        with self.assertRaisesRegex(VerificationError, "component sum"):
            verify_optimizer_proof(parameters, 120, 10)

    def test_forged_zero_baseline_is_rejected_against_dataset_meter_values(self):
        values = np.zeros((169, 7), dtype=np.float32)
        values[:, 0] = 100.; values[:, 4] = 2.; values[:, 5] = .5
        demand = 100. * 38. * 168 / (24 * 30.4375)
        row = {"energy_cost_cny": 33600., "carbon_kg": 8400., "peak_kw": 100.,
               "degradation_cost_cny": 0., "demand_charge_cny": demand,
               "total_cost_cny": 33600. + demand, "bess_throughput_kwh": 0., "aux_shift_kwh": 0.}
        reference = {"starts": [0], "rows": [row]}
        physical = {"grid": {"demand_charge_cny_per_kw_month": 38., "hard_pcc_limit_kw": 36000.}}
        verify_zero_reference(reference, values, 0, physical)
        row["carbon_kg"] += 10.
        with self.assertRaisesRegex(VerificationError, "zero baseline carbon_kg"):
            verify_zero_reference(reference, values, 0, physical)

    def test_saved_tensor_and_adam_step_corroborate_hook_count(self):
        policy = {"weight": torch.tensor([1., 2.], dtype=torch.float32)}
        array = policy["weight"].numpy()
        sha = hashlib.sha256(b"weight" + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()
        parameters = {"optimizer_step_calls_by_component": {"policy": 4}}
        row = {"step": 40, "sb3_update_counter": 4, "optimizer_updates": 4,
               "weights_sha256": sha, "parameters": parameters}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.zip"
            for adam_step in (4, 3):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("data", json.dumps({"num_timesteps": 40, "_n_updates": 4,
                        "_v8_optimizer_step_calls": 4, "_v8_optimizer_steps_by_component": {"policy": 4}}))
                    payload = io.BytesIO(); torch.save(policy, payload); archive.writestr("policy.pth", payload.getvalue())
                    payload = io.BytesIO(); torch.save({"state": {0: {"step": torch.tensor(float(adam_step))}}}, payload)
                    archive.writestr("policy.optimizer.pth", payload.getvalue())
                if adam_step == 4:
                    archive_proof(path, row)
                else:
                    with self.assertRaisesRegex(VerificationError, "Adam state step"):
                        archive_proof(path, row)

    def test_real_sac_and_td3_archives_keep_separate_actor_critic_counters(self):
        import gymnasium as gym
        from stable_baselines3 import SAC, TD3
        torch.set_num_threads(1)
        from app.services.rl_model.shore_bess.v8_replay import NStepReplayBuffer
        from scripts.train_shore_bess_v8 import replay_configuration
        for algorithm, n_step in ((SAC, 1), (TD3, 1), (TD3, 3)):
            with self.subTest(algorithm=algorithm.__name__, n_step=n_step), tempfile.TemporaryDirectory() as tmp:
                env = gym.make("Pendulum-v1")
                model = algorithm("MlpPolicy", env, learning_starts=2, buffer_size=32, batch_size=2,
                                  train_freq=1, gradient_steps=1, gamma=1., seed=31,
                                  policy_kwargs={"net_arch": [8, 8]}, device="cpu", verbose=0,
                                  **({"ent_coef": .001} if algorithm is SAC else {}),
                                  **({"replay_buffer_class": NStepReplayBuffer, "replay_buffer_kwargs": {"n_step": n_step, "gamma": 1.}}
                                     if n_step > 1 else {}))
                model._v8_optimizer_step_calls = 0
                model._v8_optimizer_steps_by_component = {"actor": 0, "critic": 0}
                handles = []
                for name, optimizer in (("actor", model.actor.optimizer), ("critic", model.critic.optimizer)):
                    def record_step(_optimizer, _args, _kwargs, component=name):
                        model._v8_optimizer_step_calls += 1
                        model._v8_optimizer_steps_by_component[component] += 1
                    handles.append(optimizer.register_step_post_hook(record_step))
                try:
                    model.learn(8)
                    sha = hashlib.sha256()
                    for key, tensor in sorted(model.policy.state_dict().items()):
                        array = tensor.detach().cpu().numpy()
                        sha.update(key.encode()); sha.update(str(array.dtype).encode())
                        sha.update(str(array.shape).encode()); sha.update(array.tobytes())
                    parameters = {"num_timesteps": model.num_timesteps,
                        "gradient_optimizer_step_calls": model._v8_optimizer_step_calls,
                        "optimizer_step_calls_by_component": model._v8_optimizer_steps_by_component,
                        "_n_updates": model._n_updates, "gamma": 1., "policy_kwargs": {"net_arch": [8, 8]},
                        "optimizer_learning_rates": {name: [float(group["lr"]) for group in optimizer.param_groups]
                            for name, optimizer in (("actor", model.actor.optimizer), ("critic", model.critic.optimizer))}}
                    row = {"step": model.num_timesteps, "sb3_update_counter": model._n_updates,
                           "optimizer_updates": model._v8_optimizer_step_calls,
                           "weights_sha256": sha.hexdigest(), "parameters": parameters}
                    if n_step > 1:
                        config = {"n_step": n_step, "algorithm": "stable_baselines3.TD3", "gamma": 1.,
                                  "algorithm_variant": f"td3_n_step_{n_step}_off_policy_uncorrected",
                                  "algorithm_parameters": {"batch_size": 2, "buffer_size": 32},
                                  "replay_buffer_config": replay_configuration("td3", n_step)}
                        parameters["replay_buffer_class"] = "app.services.rl_model.shore_bess.v8_replay.NStepReplayBuffer"
                        parameters["replay_buffer_kwargs"] = {"n_step": n_step, "gamma": 1.}
                        snapshot = model.replay_buffer.describe()
                        snapshot["checkpoint_phase"] = "post_rollout_and_optimizer"
                        parameters["replay_buffer_snapshot"] = snapshot
                        verify_n_step_snapshot(parameters, config, model.num_timesteps, "post_rollout_and_optimizer")
                        emitted = model.replay_buffer.flush_pending(reason="training_budget_end")
                        snapshot = model.replay_buffer.describe()
                        counters = {"optimizer_step_calls": model._v8_optimizer_step_calls, "sb3_update_counter": model._n_updates,
                                    "optimizer_step_calls_by_component": model._v8_optimizer_steps_by_component}
                        snapshot.update(checkpoint_phase="training_end_after_storage_only_flush", storage_only_flush_emitted=emitted,
                                        optimizer_steps_before_flush=copy.deepcopy(counters), optimizer_steps_after_flush=copy.deepcopy(counters))
                        parameters["replay_buffer_snapshot"] = snapshot
                        verify_n_step_snapshot(parameters, config, model.num_timesteps, snapshot["checkpoint_phase"])
                        forged = copy.deepcopy(parameters)
                        forged["replay_buffer_snapshot"]["sampled_batches"] += 1
                        with self.assertRaisesRegex(VerificationError, "sample batches"):
                            verify_n_step_snapshot(forged, config, model.num_timesteps, snapshot["checkpoint_phase"])
                        forged = copy.deepcopy(parameters)
                        forged["replay_buffer_snapshot"]["optimizer_steps_after_flush"]["optimizer_step_calls"] += 1
                        with self.assertRaisesRegex(VerificationError, "post-flush optimizer counts"):
                            verify_n_step_snapshot(forged, config, model.num_timesteps, snapshot["checkpoint_phase"])
                        model._v8_replay_snapshot = copy.deepcopy(snapshot)
                    path = Path(tmp) / "model.zip"
                    model.save(path)
                    verify_optimizer_proof(parameters, row["optimizer_updates"], row["sb3_update_counter"])
                    archive_proof(path, row)
                finally:
                    for handle in handles:
                        handle.remove()
                    env.close()


if __name__ == "__main__":
    unittest.main()
