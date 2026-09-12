"""Restricted-control diagnostic checks use only generated hourly fixtures."""
from __future__ import annotations

import unittest

import numpy as np

from app.services.rl_model.shore_bess.v3_environment import load_config
from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8SACEnv
from scripts.train_shore_bess_v8_flex_specialist import ShoreBESSV8FlexSpecialistEnv
from tests.test_shore_bess_v8 import fixture_dataset


class FlexSpecialistTests(unittest.TestCase):
    def make_env(self, env_class=ShoreBESSV8FlexSpecialistEnv):
        env = env_class(fixture_dataset(), slice(672, 960), config=load_config(),
                        normalization_slice=slice(0, 672), episode_steps=168,
                        carbon_price=12.0, training=False)
        self.addCleanup(env.close)
        env.reset(options={"start_index": 0})
        return env

    def test_one_action_matches_original_physics_and_bill_with_bess_idle(self):
        env = self.make_env()
        reference = self.make_env(ShoreBESSV8SACEnv)
        self.assertEqual(env.action_space.shape, (1,))
        self.assertEqual(env.observation_space.shape, (31,))
        rng = np.random.default_rng(913)
        for _ in range(168):
            action = rng.uniform(-1, 1, 1).astype(np.float32)
            obs, reward, done, truncated, info = env.step(action)
            baseline_obs, baseline_reward, baseline_done, _, baseline_info = reference.step(
                np.asarray([0, action[0]], dtype=np.float32))
            np.testing.assert_array_equal(obs, baseline_obs)
            self.assertEqual(reward, baseline_reward)
            self.assertEqual(done, baseline_done)
            self.assertFalse(truncated)
            self.assertEqual(info["final_action"], baseline_info["final_action"])
            self.assertEqual(info["final_action"]["bess_kw"], 0)
        self.assertEqual(env.totals, reference.totals)
        self.assertEqual(env.totals["bess_throughput_kwh"], 0)
        self.assertLessEqual(abs(env.totals["terminal_flex_backlog_kwh"]), 1e-6)
        self.assertEqual(env.totals["flex_deadline_violation_kwh"], 0)
        self.assertEqual(env.totals["physical_power_violations"], 0)

    def test_invalid_neural_action_is_rejected(self):
        env = self.make_env()
        for action in ([], [0, 1], [float("nan")]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)

    def test_actual_sac_learns_with_a_single_actor_output(self):
        import torch
        from stable_baselines3 import SAC
        from scripts.train_shore_bess_v8 import weights_sha256
        torch.set_num_threads(1)
        env = self.make_env()
        model = SAC("MlpPolicy", env, gamma=1.0, ent_coef=0.001, learning_starts=8,
                    buffer_size=1000, batch_size=8, train_freq=1, gradient_steps=1,
                    policy_kwargs={"net_arch": [128, 128]}, seed=912)
        initial = weights_sha256(model)
        model.learn(32)
        self.assertEqual(model.num_timesteps, 32)
        self.assertEqual(model._n_updates, 24)
        self.assertNotEqual(weights_sha256(model), initial)
        self.assertEqual(model.actor.mu.out_features, 1)
        self.assertEqual(env.totals["bess_throughput_kwh"], 0)


if __name__ == "__main__":
    unittest.main()
