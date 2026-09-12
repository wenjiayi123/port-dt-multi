"""Independent n-step return, terminal-boundary and SB3 integration checks."""
from __future__ import annotations

import pickle
import unittest

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer

from app.services.rl_model.shore_bess.v8_replay import NStepReplayBuffer


class NStepReplayTests(unittest.TestCase):
    def test_describe_distinguishes_observed_events_emitted_tails_and_sampled_draws(self):
        buffer = self.make_buffer()
        for index in range(5):
            self.add(buffer, index, index + 1, done=index == 4)
        self.add(buffer, 100, 2)
        self.add(buffer, 101, 3, done=True, timeout=True)
        buffer._get_samples(np.array([0, 2, 3, 4, 5, 6, 6]))
        snapshot = buffer.describe()
        self.assertEqual(snapshot["observed_transitions"], 7)
        self.assertEqual(snapshot["emitted_transitions"], 7)
        self.assertEqual(snapshot["emitted_horizon_counts"], {"1": 2, "2": 2, "3": 3})
        self.assertEqual(snapshot["sampled_horizon_counts"], {"1": 3, "2": 2, "3": 2})
        self.assertEqual(snapshot["sampled_transition_draws"], 7)
        self.assertEqual(snapshot["sampled_batches"], 1)
        self.assertEqual(snapshot["observed_terminal_events"], 1)
        self.assertEqual(snapshot["observed_timeout_events"], 1)
        self.assertEqual(snapshot["emitted_terminal_endpoints"], 3)
        self.assertEqual(snapshot["emitted_timeout_endpoints"], 2)
        self.assertEqual(snapshot["sampled_terminal_endpoints"], 3)
        self.assertEqual(snapshot["sampled_timeout_endpoints"], 3)
        self.assertEqual(snapshot["flush_counts"], {"episode_end": 2})
        self.assertTrue(snapshot["transition_accounting_balanced"])
        self.assertTrue(snapshot["emitted_horizon_accounting_balanced"])

    def test_discard_and_budget_flush_preserve_accounting_without_claiming_learning(self):
        buffer = self.make_buffer(n_step=24)
        self.add(buffer, 0, 1)
        self.add(buffer, 1, 2)
        snapshot = buffer.describe()
        self.assertEqual(snapshot["pending_transitions"], 2)
        self.assertTrue(snapshot["transition_accounting_balanced"])
        self.assertEqual(buffer.discard_pending(), 2)
        self.add(buffer, 100, 5)
        self.add(buffer, 101, 7)
        buffer.flush_pending(reason="training_budget_end")
        snapshot = buffer.describe()
        self.assertEqual(snapshot["observed_transitions"], 4)
        self.assertEqual(snapshot["emitted_transitions"], 2)
        self.assertEqual(snapshot["discarded_pending_transitions"], 2)
        self.assertEqual(snapshot["sampled_transition_draws"], 0)
        self.assertEqual(snapshot["last_flush"]["newly_stored_transitions"], 2)
        self.assertFalse(snapshot["last_flush"]["endpoint_done"])
        self.assertTrue(snapshot["last_flush"]["storage_only_no_optimizer_update_performed"])
        self.assertTrue(snapshot["transition_accounting_balanced"])
        with self.assertRaisesRegex(ValueError, "flush reason"):
            buffer.flush_pending(reason="invented_terminal")
        buffer.reset()
        snapshot = buffer.describe()
        self.assertEqual(snapshot["reset_count"], 1)
        self.assertEqual(snapshot["observed_transitions"], 0)
        self.assertEqual(snapshot["emitted_horizon_counts"], {})
        self.assertEqual(snapshot["sampled_horizon_counts"], {})
        self.assertEqual(snapshot["flush_counts"], {})
        self.assertIsNone(snapshot["last_flush"])

    def test_resident_horizons_are_separate_from_lifetime_emissions_after_ring_wrap(self):
        buffer = NStepReplayBuffer(4, spaces.Box(-1000, 1000, (1,), dtype=np.float32),
                                  spaces.Box(-1, 1, (1,), dtype=np.float32), device="cpu", n_step=3)
        for episode in range(5):
            self.add(buffer, episode * 100, 1)
            self.add(buffer, episode * 100 + 1, 2, done=True)
        snapshot = buffer.describe()
        self.assertEqual(snapshot["emitted_transitions"], 10)
        self.assertEqual(snapshot["emitted_horizon_counts"], {"1": 5, "2": 5})
        self.assertEqual(snapshot["resident_transitions"], 4)
        self.assertEqual(snapshot["resident_horizon_counts"], {"1": 2, "2": 2})
        self.assertEqual(snapshot["resident_terminal_endpoints"], 4)
        self.assertTrue(snapshot["resident_full"])

    def make_buffer(self, n_step=3, **kwargs):
        return NStepReplayBuffer(64, spaces.Box(-1000, 1000, (1,), dtype=np.float32),
                                spaces.Box(-1, 1, (1,), dtype=np.float32), device="cpu",
                                n_step=n_step, **kwargs)

    @staticmethod
    def add(buffer, state, reward, *, done=False, timeout=False):
        buffer.add(np.array([[state]], np.float32), np.array([[state + 1]], np.float32),
                   np.array([[state / 1000]], np.float32), np.array([reward], np.float32),
                   np.array([done]), [{"TimeLimit.truncated": timeout}])

    def test_three_step_sum_uses_real_endpoint_and_original_action(self):
        buffer = self.make_buffer()
        self.add(buffer, 0, 1)
        self.add(buffer, 1, 2)
        self.assertEqual(buffer.size(), 0)
        self.add(buffer, 2, 3)
        self.assertEqual(buffer.size(), 1)
        self.assertEqual(buffer.rewards[0, 0], 6)
        self.assertEqual(buffer.observations[0, 0, 0], 0)
        self.assertEqual(buffer.next_observations[0, 0, 0], 3)
        self.assertEqual(buffer.actions[0, 0, 0], 0)
        self.assertEqual(buffer.n_step_horizons[0, 0], 3)
        self.assertEqual(buffer.dones[0, 0], 0)

    def test_terminal_flush_all_short_tails_without_cross_episode_rewards(self):
        buffer = self.make_buffer()
        for index, reward in enumerate([1, 2, 3, 4, 5]):
            self.add(buffer, index, reward, done=index == 4)
        np.testing.assert_array_equal(buffer.rewards[:5, 0], [6, 9, 12, 9, 5])
        np.testing.assert_array_equal(buffer.next_observations[:5, 0, 0], [3, 4, 5, 5, 5])
        np.testing.assert_array_equal(buffer.dones[:5, 0], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(buffer.n_step_horizons[:5, 0], [3, 3, 3, 2, 1])
        self.assertEqual(buffer.pending_count, 0)
        self.add(buffer, 100, 100)
        self.add(buffer, 101, 200, done=True)
        np.testing.assert_array_equal(buffer.rewards[5:7, 0], [300, 200])
        self.assertEqual(buffer.observed_transition_count, buffer.emitted_transition_count)

    def test_timeout_flush_preserves_endpoint_bootstrap_semantics(self):
        buffer = self.make_buffer()
        self.add(buffer, 0, 2)
        self.add(buffer, 1, 3, done=True, timeout=True)
        samples = buffer._get_samples(np.array([0, 1]))
        np.testing.assert_array_equal(samples.rewards.cpu().numpy().ravel(), [5, 3])
        np.testing.assert_array_equal(samples.next_observations.cpu().numpy().ravel(), [2, 2])
        np.testing.assert_array_equal(samples.dones.cpu().numpy().ravel(), [0, 0])
        self.assertEqual(buffer.pending_count, 0)

    def test_budget_flush_bootstraps_without_inventing_done(self):
        buffer = self.make_buffer(n_step=24)
        self.add(buffer, 0, -2)
        self.add(buffer, 1, 7)
        self.assertEqual(buffer.flush_pending(), 2)
        np.testing.assert_array_equal(buffer.rewards[:2, 0], [5, 7])
        np.testing.assert_array_equal(buffer.dones[:2, 0], [0, 0])
        np.testing.assert_array_equal(buffer.next_observations[:2, 0, 0], [2, 2])
        self.assertEqual(buffer.flush_pending(), 0)

    def test_unannounced_reset_is_rejected_and_explicit_reset_clears_queue(self):
        buffer = self.make_buffer()
        self.add(buffer, 0, 1)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            self.add(buffer, 100, 100)
        buffer.reset()
        self.assertEqual(buffer.pending_count, 0)
        self.assertEqual(buffer.size(), 0)
        self.add(buffer, 100, 100, done=True)
        self.assertEqual(buffer.rewards[0, 0], 100)
        self.assertEqual(buffer.observed_transition_count, 1)

    def test_inputs_are_copied_and_pending_queue_survives_serialization(self):
        buffer = self.make_buffer()
        obs, next_obs, action = (np.array([[value]], np.float32) for value in (0, 1, 0.5))
        buffer.add(obs, next_obs, action, np.array([4.0]), np.array([False]), [{}])
        obs[:], next_obs[:], action[:] = 100, 100, 100
        restored = pickle.loads(pickle.dumps(buffer))
        self.add(restored, 1, 5, done=True)
        self.assertEqual(restored.rewards[0, 0], 9)
        self.assertEqual(restored.observations[0, 0, 0], 0)
        self.assertEqual(restored.actions[0, 0, 0], 0.5)

    def test_one_step_is_identical_to_sb3_replay(self):
        nstep = self.make_buffer(n_step=1)
        ordinary = ReplayBuffer(64, nstep.observation_space, nstep.action_space, device="cpu")
        for index, reward in enumerate([1, -4, 7, 3, -2]):
            for buffer in (nstep, ordinary):
                self.add(buffer, index, reward, done=index == 4)
        for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
            np.testing.assert_array_equal(getattr(nstep, name)[:5], getattr(ordinary, name)[:5])

    def test_invalid_discount_multienv_and_memory_aliasing_fail_closed(self):
        for kwargs in ({"gamma": 0.99}, {"n_envs": 2}, {"optimize_memory_usage": True},
                       {"n_step": 0}, {"n_step": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_buffer(**kwargs)

    def test_undiscounted_td3_trains_with_delayed_storage_and_terminal_tails(self):
        class ToyEnv(gym.Env):
            observation_space = spaces.Box(0, 10, (1,), dtype=np.float32)
            action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

            def reset(self, **kwargs):
                self.index = 0
                return np.array([0], np.float32), {}

            def step(self, action):
                self.index += 1
                return np.array([self.index], np.float32), float(self.index), self.index == 4, False, {}

        torch.set_num_threads(1)
        model = TD3("MlpPolicy", ToyEnv(), gamma=1, learning_starts=4, batch_size=2,
                    train_freq=1, gradient_steps=1, replay_buffer_class=NStepReplayBuffer,
                    replay_buffer_kwargs={"n_step": 3, "gamma": 1.0},
                    policy_kwargs={"net_arch": [16, 16]}, seed=912, device="cpu")
        try:
            model.learn(total_timesteps=12)
            self.assertEqual(model.num_timesteps, 12)
            self.assertEqual(model._n_updates, 8)
            self.assertEqual(model.replay_buffer.size(), 12)
            self.assertEqual(model.replay_buffer.pending_count, 0)
            np.testing.assert_array_equal(model.replay_buffer.rewards[:4, 0], [6, 9, 7, 4])
            np.testing.assert_array_equal(model.replay_buffer.next_observations[:4, 0, 0], [3, 4, 4, 4])
        finally:
            model.env.close()


if __name__ == "__main__":
    unittest.main()
