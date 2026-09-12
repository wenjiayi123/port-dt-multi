"""Admission protocol tests using tiny local fixtures, never held-out datasets."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.admit_shore_bess_v8 import INTEGRITY_CHECKS, assert_compatible, checked_path, verify_references
from app.services.rl_training.datasets import file_sha256


def fixture(seed):
    return {
        "status": "PILOT_VALIDATION_ONLY", "evaluations": {},
        "config": {"pilot": True, "seeds": [seed], "algorithm": "stable_baselines3.SAC",
                   "requested_steps_per_seed": 60000, "reward_credit_assignment": "fixed-reward-v2",
                   "action_mapping": {"version": "fixed-physical-map-v1"}},
        "manifest": {"test_status": "not_opened", "forward_status": "not_loaded",
                     "teacher_actions_used": False, "warm_start_used": False,
                     "source_sha256": {"source.py": "source-digest"}, "versions": {"python": "fixture"},
                     "train": {"start_row": 0, "stop_row_exclusive": 100},
                     "validation": {"start_row": 100, "stop_row_exclusive": 200}},
        "input_files_sha256": {"data.csv": "data-digest"},
        "results": [{"seed": seed}], "checks": {key: True for key in INTEGRITY_CHECKS},
    }


class ShoreBESSV8AdmissionTests(unittest.TestCase):
    def test_three_independent_identical_protocols_are_compatible(self):
        assert_compatible([fixture(seed) for seed in (912, 1012, 1112)])

    def test_duplicate_seed_or_wrong_number_is_rejected(self):
        for seeds in ((912, 912, 1112), (912, 1012)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                assert_compatible([fixture(seed) for seed in seeds])

    def test_different_algorithm_reward_action_source_data_or_budget_is_rejected(self):
        mutations = [
            lambda report: report["config"].update(algorithm="stable_baselines3.TD3"),
            lambda report: report["config"].update(reward_credit_assignment="different"),
            lambda report: report["config"]["action_mapping"].update(version="different"),
            lambda report: report["config"].update(requested_steps_per_seed=120000),
            lambda report: report["manifest"]["source_sha256"].update({"source.py": "changed"}),
            lambda report: report["input_files_sha256"].update({"data.csv": "changed"}),
        ]
        for mutate in mutations:
            reports = [fixture(seed) for seed in (912, 1012, 1112)]
            mutate(reports[2])
            with self.assertRaises(ValueError):
                assert_compatible(reports)

    def test_heldout_access_teacher_warm_start_or_failed_integrity_is_rejected(self):
        mutations = [
            lambda report: report.update(evaluations={"test": {}}),
            lambda report: report["manifest"].update(forward_status="loaded"),
            lambda report: report["manifest"].update(teacher_actions_used=True),
            lambda report: report["manifest"].update(warm_start_used=True),
            lambda report: report["checks"].update(real_optimizer_updates=False),
        ]
        for mutate in mutations:
            reports = [fixture(seed) for seed in (912, 1012, 1112)]
            mutate(reports[0])
            with self.assertRaises(ValueError):
                assert_compatible(reports)

    def test_validation_business_failure_does_not_disqualify_honest_retraining_sources(self):
        # Full validation is re-evaluated later; pilot business pass is not an
        # integrity requirement and must not silently choose the source seed.
        reports = [fixture(seed) for seed in (912, 1012, 1112)]
        for report in reports:
            report["checks"]["all_seeds_validation_passed"] = False
        assert_compatible(reports)

    def test_hash_registry_rejects_corruption_missing_files_and_scope_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            path = run / "weights.zip"
            path.write_bytes(b"tiny immutable fixture, not a model")
            references = {"run/weights.zip": file_sha256(path)}
            verify_references(references, base=root, scope=run)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_references(references, base=root, scope=run)
            outside = root / "outside.txt"
            outside.write_text("fixture", encoding="utf-8")
            for relative in ("outside.txt", "../escape.txt", str(outside)):
                with self.subTest(path=relative), self.assertRaises(ValueError):
                    checked_path(relative, base=root, scope=run)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_references(references, base=root, scope=run)


if __name__ == "__main__":
    unittest.main()
