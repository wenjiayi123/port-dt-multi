from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.services.rl_model.shore_bess.v3_environment import chronological_slices, load_config
from app.services.rl_model.shore_bess.v8_environment import LATTICE, ShoreBESSV8SACEnv
from app.services.rl_training.datasets import file_sha256
from scripts.evaluate_shore_bess_v8_comparators import ablate_action, evaluate_arm, run_comparisons
from tests.test_shore_bess_v8 import fixture_dataset


class ConstantActor:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        return self.action.copy(), None


class ShoreBESSV8ComparatorTests(unittest.TestCase):
    def factory(self):
        data = fixture_dataset()
        train, _, test = chronological_slices(data)
        return lambda: ShoreBESSV8SACEnv(data, test, config=load_config(), normalization_slice=train,
                                        episode_steps=48, training=False, record_trace=False)

    def test_continuous_ablation_preserves_original_actor_and_other_coordinate(self):
        original = np.asarray([0.75, -0.25], dtype=np.float32)
        disabled_bess = ablate_action(original, "learned_no_bess_coordinate", False)
        disabled_flex = ablate_action(original, "learned_no_flex_coordinate", False)
        np.testing.assert_array_equal(original, [0.75, -0.25])
        np.testing.assert_array_equal(disabled_bess, [0, -0.25])
        np.testing.assert_array_equal(disabled_flex, [0.75, 0])

    def test_discrete_ablation_keeps_exact_other_action_and_lattice_membership(self):
        for index, action in enumerate(LATTICE):
            for arm, dimension in (("learned_no_bess_coordinate", 0), ("learned_no_flex_coordinate", 1)):
                result = LATTICE[ablate_action(index, arm, True)]
                self.assertEqual(result[dimension], 0)
                self.assertEqual(result[1 - dimension], action[1 - dimension])

    def test_idle_replay_writes_actual_full_window_and_zero_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            result = evaluate_arm(self.factory(), ConstantActor([0, 0]), "idle", [0, 48], discrete=False, masked=False, trace_path=path)
            self.assertEqual(result["starts"], [0, 48])
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(result["windows"][0]["source_period"], "2025-01/02")
            self.assertEqual(len(json.loads(path.read_text())["trace"]), 48)
            for row in result["rows"]:
                for key in ("charge_kwh", "discharge_kwh", "flex_deferred_kwh", "flex_repaid_kwh",
                            "energy_reconciliation_residual_kwh", "carbon_reconciliation_residual_kg",
                            "financial_reconciliation_residual_cny", "bess_carbon_delta_kg", "flex_carbon_delta_kg"):
                    self.assertAlmostEqual(row[key], 0, places=6)

    def test_requested_channel_ablation_measures_actual_executed_channel_energy(self):
        for arm, disabled in (("learned_no_bess_coordinate", ("charge_kwh", "discharge_kwh")),
                              ("learned_no_flex_coordinate", ("flex_deferred_kwh", "flex_repaid_kwh"))):
            with self.subTest(arm=arm):
                result = evaluate_arm(self.factory(), ConstantActor([0.6, -0.6]), arm, [0], discrete=False, masked=False)
                row = result["rows"][0]
                for key in disabled:
                    self.assertEqual(row[key], 0.0)
                self.assertAlmostEqual(row["terminal_soc_error"], 0, places=9)
                self.assertAlmostEqual(row["terminal_flex_backlog_kwh"], 0, places=6)
                for key in ("energy_reconciliation_residual_kwh", "carbon_reconciliation_residual_kg", "financial_reconciliation_residual_cny"):
                    self.assertAlmostEqual(row[key], 0, places=6)

    def test_validation_rejected_report_cannot_trigger_heldout_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "report.json"
            path.write_text(json.dumps({"schema": "port-shore-bess-v8-report.v1",
                                       "status": "VALIDATION_REJECTED", "config": {"pilot": False},
                                       "evaluations": {}}))
            with patch("scripts.evaluate_shore_bess_v8_comparators.ROOT", root), \
                    patch("scripts.evaluate_shore_bess_v8_comparators.load_port_dataset") as loader:
                with self.assertRaisesRegex(ValueError, "completed formal V8 report"):
                    run_comparisons(path, file_sha256(path))
            loader.assert_not_called()
            self.assertFalse((root / "evidence").exists())


if __name__ == "__main__":
    unittest.main()
