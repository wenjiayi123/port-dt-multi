"""Independent uncertainty and evidence-integrity checks for the V8 runner."""
from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path
from types import SimpleNamespace

from app.services.rl_training.datasets import file_sha256
from app.services.rl_training.statistics import bootstrap_summary
from scripts.train_shore_bess_v8 import (
    capture_replay_snapshot, hashes_match, model_optimizers, model_parameters, official_source_period,
    replay_configuration, replay_construction_kwargs, source_period_summary, weights_sha256, TrainingEpisodeLedger,
)


class ShoreBESSV8TrainingTests(unittest.TestCase):
    def test_training_episode_ledger_keeps_observed_physics_and_prior_optimizer_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_episodes.csv"
            ledger = TrainingEpisodeLedger(path)
            model = SimpleNamespace(num_timesteps=168, _v8_optimizer_step_calls=318, _n_updates=159)
            physical_metrics = {"physical_power_violations": 2, "guardrail_violations": 2,
                                "flex_deadline_violation_kwh": 0.0, "training_reward": -203.5,
                                "total_cost_cny": 1000.0, "carbon_kg": 600.0}
            try:
                ledger.record(model, [{"episode_metrics": physical_metrics}])
                ledger.record(model, [{"other_step_info": 1}])
                model._v8_optimizer_step_calls += 2
                model._n_updates += 1
                with path.open(encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["environment_steps"], "168")
                self.assertEqual(rows[0]["optimizer_step_calls_at_observation"], "318")
                self.assertEqual(rows[0]["sb3_update_counter_at_observation"], "159")
                self.assertEqual(rows[0]["record_phase"], TrainingEpisodeLedger.phase)
                for key, value in physical_metrics.items():
                    self.assertEqual(float(rows[0]["episode_metrics." + key]), value)
                self.assertEqual(ledger.episodes, 1)
                with self.assertRaises(ValueError):
                    ledger.record(model, [{"episode_metrics": {"training_reward": float("nan")}}])
            finally:
                ledger.close()

    def test_multistep_variant_rejects_sac_and_preserves_default_construction(self):
        self.assertEqual(replay_construction_kwargs({"n_step": 1}), {})
        for algorithm, n_step in (("sac", 24), ("dqn", 24), ("td3", 0), ("td3", 169)):
            with self.subTest(algorithm=algorithm, n_step=n_step), self.assertRaises(ValueError):
                replay_configuration(algorithm, n_step)
        config = {"n_step": 24, "algorithm": "stable_baselines3.TD3", "gamma": 0.99,
                  "replay_buffer_config": replay_configuration("td3", 24)}
        with self.assertRaises(ValueError):
            replay_construction_kwargs(config)

    def test_actual_td3_snapshot_survives_zip_and_budget_flush_does_not_train(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import TD3
        torch.set_num_threads(1)
        config = {"n_step": 24, "algorithm": "stable_baselines3.TD3", "gamma": 1.0,
                  "replay_buffer_config": replay_configuration("td3", 24)}
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, gamma=1.0, buffer_size=1000, batch_size=8,
                    learning_starts=24, train_freq=1, gradient_steps=1, seed=912,
                    policy_kwargs={"net_arch": [128, 128]}, **replay_construction_kwargs(config))
        model._v8_optimizer_step_calls = 0
        model._v8_optimizer_steps_by_component = {name: 0 for name in model_optimizers(model)}
        hooks = []
        try:
            initial = capture_replay_snapshot(model, "initial_zero_updates")
            self.assertEqual(initial["observed_transitions"], 0)
            for name, optimizer in model_optimizers(model).items():
                def count(_optimizer, _args, _kwargs, component=name):
                    model._v8_optimizer_step_calls += 1
                    model._v8_optimizer_steps_by_component[component] += 1
                hooks.append(optimizer.register_step_post_hook(count))
            model.learn(60)
            before = capture_replay_snapshot(model, "post_rollout_and_optimizer")
            self.assertEqual(before["pending_transitions"], 23)
            weights_before = weights_sha256(model)
            final = capture_replay_snapshot(model, "training_end_after_storage_only_flush", flush_at_training_end=True)
            self.assertEqual(final["storage_only_flush_emitted"], 23)
            self.assertEqual(final["pending_transitions"], 0)
            self.assertEqual(final["emitted_transitions"], 60)
            self.assertEqual(final["sampled_batches"], model._n_updates)
            self.assertEqual(final["sampled_batches"], model._v8_optimizer_steps_by_component["critic"])
            self.assertEqual(final["optimizer_steps_before_flush"], final["optimizer_steps_after_flush"])
            self.assertEqual(weights_before, weights_sha256(model))
            self.assertEqual(before["sampled_transition_draws"], final["sampled_transition_draws"])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.zip"
                model.save(path)
                restored = TD3.load(path, device="cpu")
                self.assertEqual(restored._v8_replay_snapshot, final)
                self.assertEqual(model_parameters(restored), model_parameters(model))
            self.assertEqual(initial["observed_transitions"], 0)
        finally:
            for hook in hooks:
                hook.remove()
            env.close()

    def test_month_clusters_expose_uncertainty_hidden_by_many_weekly_rows(self):
        gains = [0.1] * 5 + [-0.2] + [0.1] * 5
        groups = ["2025-05"] * 5 + ["2025-06"] + ["2025-07"] * 5
        weekly = bootstrap_summary(gains, seed=20260912, resamples=5000)
        monthly = source_period_summary(gains, groups)
        self.assertGreater(weekly["ci_low"], 0)
        self.assertLess(monthly["clustered_ci_low"], 0)
        self.assertEqual(monthly["cluster_count"], 3)
        self.assertEqual(monthly["cluster_window_counts"], {"2025-05": 5, "2025-06": 1, "2025-07": 5})

    def test_month_bootstrap_is_deterministic_and_preserves_constant_gain(self):
        first = source_period_summary([0.02] * 6, ["2025-05", "2025-05", "2025-06", "2025-07", "2025-07", "2025-07"])
        second = source_period_summary([0.02] * 6, ["2025-05", "2025-05", "2025-06", "2025-07", "2025-07", "2025-07"])
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["clustered_ci_low"], 0.02)
        self.assertAlmostEqual(first["clustered_ci_high"], 0.02)

    def test_month_bootstrap_rejects_missing_or_nonfinite_observations(self):
        for values, groups in (([], []), ([0.1], []), ([float("nan")], ["2025-05"])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                source_period_summary(values, groups)

    def test_january_february_are_one_official_anchor(self):
        periods = [official_source_period(timestamp) for timestamp in
                   ["2026-01-05T00:00:00Z", "2026-02-02T00:00:00Z", "2026-03-02T00:00:00Z",
                    "2026-04-06T00:00:00Z", "2026-05-04T00:00:00Z"]]
        summary = source_period_summary([0.02] * 5, periods)
        self.assertEqual(summary["cluster_count"], 4)
        self.assertEqual(summary["cluster_window_counts"]["2026-01/02"], 2)

    def test_input_gate_rejects_modified_or_deleted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            path.write_text("original\n", encoding="utf-8")
            expected = {str(path): file_sha256(path)}
            self.assertTrue(hashes_match(expected))
            path.write_text("changed\n", encoding="utf-8")
            self.assertFalse(hashes_match(expected))
            path.unlink()
            self.assertFalse(hashes_match(expected))


if __name__ == "__main__":
    unittest.main()
