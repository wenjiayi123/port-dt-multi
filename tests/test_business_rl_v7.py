from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from app.services.rl_training.business_learning import BusinessDeltaReward, FiniteDispatchActions
from app.services.rl_training.business_runtime import BusinessPolicyRegistry
from app.services.rl_training.model_artifacts import resolve_model_artifact
from app.services.rl_training.trainer import TrainingManager, TrainingJob
from scripts.train_business_rl_v7 import window_starts, convergence, improves_incumbent


class BusinessRLTests(unittest.TestCase):
    def test_final_ppo_rollout_is_actually_optimized_and_seed_zero_retained(self):
        # Previously a 64-step job with n_steps=64 reported COMPLETED while
        # saving an entirely untrained PPO actor (_n_updates == 0).
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingManager(run_root=Path(tmp) / "runs", benchmark_path=Path(tmp) / "benchmarks.json")
            cfg = manager.validate_config({"algorithm": "ppo", "dataset_id": "public_port_ops_v1", "total_steps": 64, "episode_steps": 64, "seed": 0})
            self.assertEqual(cfg["seed"], 0)
            job = TrainingJob("rl-final-rollout", cfg, manager)
            manager._run_training(job)
            self.assertEqual(job.status["status"], "COMPLETED")
            manifest = json.loads((job.run_dir / "manifest.json").read_text())
            self.assertGreater(manifest["optimizer_updates_observed"], 0)
            self.assertEqual(manifest["total_steps_observed"], 64)

    def test_evaluation_windows_do_not_overlap(self):
        starts = window_starts(1755, 168, 8)
        self.assertGreater(len(starts), 3)
        self.assertTrue(all(b - a >= 168 for a,b in zip(starts, starts[1:])))
        self.assertTrue(all(x + 168 < 1755 for x in starts))

    def test_flat_but_loss_making_policy_is_not_converged(self):
        curve = [{"comparison": {"total_cost_cny": {"mean": -0.01}}, "gates": {"positive_value": False}}] * 3
        self.assertFalse(convergence(curve)["passed"])

    def test_uncertain_or_regressing_candidate_does_not_replace_incumbent(self):
        delta = {name: {"ci_low": 0.05} for name in ("total_cost_cny", "carbon_kg", "peak_kw")}
        self.assertTrue(improves_incumbent(delta))
        for name, value in (("total_cost_cny", 0.0), ("carbon_kg", -0.02), ("peak_kw", -0.02)):
            with self.subTest(metric=name):
                regressed = {k: dict(v) for k,v in delta.items()}
                regressed[name]["ci_low"] = value
                self.assertFalse(improves_incumbent(regressed))

    def test_candidate_cannot_be_used_as_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = BusinessPolicyRegistry(tmp)
            with self.assertRaisesRegex(RuntimeError, "no admitted"):
                registry.predict("shore_bess", [0.0] * 34)
            with self.assertRaises(ValueError):
                registry.evidence("../shore_bess")

    def test_real_admitted_actor_matches_direct_inference_and_rejects_bad_input(self):
        from app.services.rl_training.business_runtime import ROOT
        from app.services.rl_model.yard_lighting import v3_environment as lighting
        from stable_baselines3 import SAC
        registry = BusinessPolicyRegistry(ROOT)
        evidence = registry.evidence("yard_lighting", champion=True)
        if evidence["status"] != "ADMITTED_OFFLINE_RL":
            self.skipTest("no completed lighting artifact in this checkout")
        cfg = lighting.load_config()
        dataset = lighting.load_dataset(cfg)
        train, _, test = lighting.chronological_slices(dataset)
        env = lighting.YardLightingV3Env(dataset, test, config=cfg, normalization_slice=train, training=False)
        observation, _ = env.reset(options={"start_index": 0})
        receipt = registry.predict("yard_lighting", observation)
        expected, _ = SAC.load(resolve_model_artifact(ROOT, evidence["model_path"], evidence["model_sha256"]), device="cpu").predict(observation, deterministic=True)
        np.testing.assert_array_equal(receipt["action"], expected)
        env.step(receipt["action"])
        self.assertFalse(receipt["production_authority"])
        with self.assertRaisesRegex(ValueError, "observation"):
            registry.predict("yard_lighting", [float("nan")] * len(observation))
        with self.assertRaisesRegex(ValueError, "observation"):
            registry.predict("yard_lighting", [0.0])
        with self.assertRaisesRegex(ValueError, "version"):
            registry.predict("yard_lighting", observation, expected_model_sha256="0" * 64)
        env.close()

    def test_legacy_bess_display_cannot_create_training_progress_savings(self):
        from app.services.rl_model.bess_energy.module import BessSiteConfig
        from app.services.rl_model.bess_energy.rl_engine import DisplayMetricBuilder
        from types import SimpleNamespace
        values = []
        for step in (1, 500, 1000):
            builder = DisplayMetricBuilder(BessSiteConfig.from_json({"rated_power_kW": 1000, "rated_energy_kWh": 4000}, 5), 5, 10000, 1.0, 0.6)
            values.append(builder.build(step, 1000, -3.0,
                {"price": 0.8, "ef": 0.5, "pcc_base": 8000.0, "soc": 0.5},
                {"p_ref_kW": 0.0, "p_act_kW": 0.0, "pcc_kW": 8000.0, "econ_advantage_yuan": 0.0},
                SimpleNamespace(ts=[0], idx=0)))
        for record in values:
            self.assertEqual(record["step_reward"], -3.0)
            self.assertEqual(record["econ_save"], 0.0)
            self.assertEqual(record["carbon_save"], 0.0)
            self.assertEqual(record["reward_anchor"], 0.0)

    def test_inventory_shaping_preserves_discounted_episode_return(self):
        class Inventory(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(-1,1,(1,),dtype=np.float32)
                self.action_space = gym.spaces.Box(-1,1,(2,),dtype=np.float32)
                self._soc = self.soc_initial = 0.5
                self.energy_kwh, self.power_kw, self.step_number = 100, 10, 0
            def reset(self, **kwargs):
                self.step_number = 0
                self._soc = 0.5
                return np.zeros(1,dtype=np.float32), {}
            def step(self, action):
                self.step_number += 1
                self._soc = 0.6 if self.step_number == 1 else 0.5
                return np.zeros(1,dtype=np.float32), 0.0, self.step_number == 2, False, {}
        env = BusinessDeltaReward(Inventory(), "shore_bess", 0.995)
        env.reset()
        r1 = env.step([0,0])[1]
        r2 = env.step([0,0])[1]
        self.assertAlmostEqual(r1 + 0.995 * r2, 0.0)

    def test_soft_peak_cap_cannot_leave_unpaid_terminal_energy(self):
        from app.services.rl_model.bess_energy import v3_environment as energy
        from app.services.rl_training.datasets import load_port_dataset
        from app.services.rl_training.business_runtime import ROOT
        cfg = energy.load_config(ROOT / "config/bess_energy_v32_grid_only.json")
        cfg["asset"]["terminal_soc_tolerance"] = 0.0
        dataset = load_port_dataset(cfg["dataset_id"])
        train, _, test = energy.chronological_slices(dataset)
        env = energy.BESSEnergyV3Env(dataset, test, config=cfg, normalization_slice=train, training=False)
        env.reset(options={"start_index": 0})
        env._step = env.episode_steps - 1
        env._soc = env.soc_initial - 0.001
        context = env._context()
        context["base_load_kw"] = env.soft_cap_kw + 1000.0
        projected = env._project(np.array([0.0, -1.0]), context)
        self.assertAlmostEqual(projected["next_soc"], env.soc_initial)
        self.assertIn("terminal_recovery_before_soft_peak_target", projected["projection_reasons"])
        self.assertAlmostEqual(projected["net_kw"], context["base_load_kw"] - projected["power_kw"])
        self.assertLess(projected["net_kw"], env.hard_pcc_limit_kw - env.n_minus_1_margin_kw)
        env.close()

    def test_qc_kl_is_zero_for_unchanged_policy(self):
        from app.services.rl_model.port_G_qc_mvp.rl_engine_g import gaussian_policy_kl
        mu = np.zeros((8,2))
        logs = np.log([0.1, 1.0])
        self.assertAlmostEqual(gaussian_policy_kl(mu, logs, mu, logs), 0.0)
        self.assertGreater(gaussian_policy_kl(mu, logs, mu + 0.1, logs), 0.0)


if __name__ == "__main__":
    unittest.main()
