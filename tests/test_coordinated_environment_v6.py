from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient

from app import server
from app.services.rl_training.coordinated_environment import (
    CoordinatedPortOperationsEnv,
)
from app.services.rl_training.datasets import (
    FACTOR_COLUMNS,
    PORT_WIDE_COLUMNS,
    load_port_dataset,
)
from app.services.rl_training.profiles import load_profile


def _env(dataset=None, *, episode_steps: int = 4) -> CoordinatedPortOperationsEnv:
    data = dataset or load_port_dataset("public_cn_sha_integrated_scenario_v5")
    return CoordinatedPortOperationsEnv(
        data,
        slice(0, 96),
        episode_steps=episode_steps,
        seed=22,
        demand_cap_kw=36000,
        port_profile=load_profile("cn_sha_coordinated_scenario_v6"),
        normalization_slice=slice(0, 96),
        training=False,
        record_trace=True,
    )


class CoordinatedEnvironmentV6Tests(unittest.TestCase):
    def test_hash_gated_coordinated_champion_api_preserves_offline_authority_boundary(self):
        response = TestClient(server.app).get("/api/rl/coordinated-business/evidence")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "ADMITTED_OFFLINE_CHAMPION")
        self.assertEqual(payload["selected_job_id"], "rl-20260830T070107706Z")
        self.assertFalse(payload["production_authority"])
        self.assertEqual(len(payload["selected_model_sha256"]), 64)
        self.assertEqual(len(payload["report_sha256"]), 64)
        self.assertEqual(
            payload["training_dataset"]["dataset_id"],
            "public_cn_sha_integrated_scenario_v5",
        )
        self.assertEqual(
            payload["forward_dataset"]["dataset_id"],
            "public_cn_sha_integrated_forward_2026m05_v5",
        )

    def test_contract_exposes_110_observations_and_18_named_continuous_actions(self):
        env = _env()
        observation, info = env.reset(seed=3, options={"start_index": 0})
        self.assertEqual(observation.shape, (110,))
        self.assertEqual(env.action_space.shape, (18,))
        self.assertEqual(len(env.ACTION_NAMES), 18)
        self.assertIn("rail_allocation_ratio", env.ACTION_NAMES)
        self.assertIn("barge_allocation_ratio", env.ACTION_NAMES)
        self.assertIn("pilotage_allocation_ratio", env.ACTION_NAMES)
        self.assertIn("towage_allocation_ratio", env.ACTION_NAMES)
        self.assertEqual(info["environment_version"], "port_ops_v6")
        env.close()

    def test_split_resources_are_independently_decoded(self):
        env = _env()
        raw = np.zeros(18, dtype=np.float32)
        raw[8], raw[9] = 1.0, -1.0
        raw[13], raw[14] = -0.5, 0.75
        described = env.describe_action(raw)
        self.assertEqual(described["rail_allocation_ratio"], 1.0)
        self.assertEqual(described["barge_allocation_ratio"], 0.85)
        self.assertEqual(described["pilotage_allocation_ratio"], 0.25)
        self.assertEqual(described["towage_allocation_ratio"], 0.875)
        env.close()

    def test_navigation_closure_blocks_pilotage_and_towage_outside_rl(self):
        source = load_port_dataset("public_cn_sha_integrated_scenario_v5")
        factors = source.factor_values.copy()
        factor_masks = source.factor_availability.copy()
        closure = FACTOR_COLUMNS.index("closure_flag")
        factors[0, closure] = 1.0
        factor_masks[0, closure] = 1.0
        dataset = replace(
            source,
            factor_values=factors,
            factor_availability=factor_masks,
        )
        env = _env(dataset)
        env.reset(seed=4, options={"start_index": 0})
        raw = np.ones(18, dtype=np.float32)
        _obs, _reward, _terminated, _truncated, info = env.step(raw)
        self.assertFalse(info["marine_window_open"])
        self.assertEqual(info["coordinated_control"]["pilotage_allocation_ratio"], 0.0)
        self.assertEqual(info["coordinated_control"]["towage_allocation_ratio"], 0.0)
        self.assertEqual(info["pilotage_served_vessels"], 0.0)
        self.assertEqual(info["towage_served_vessels"], 0.0)
        self.assertIn(
            "marine_window_feasible_parameterization",
            info["latent_action_correction_reasons"],
        )
        self.assertFalse(info["guardrail_violation"])
        env.close()

    def test_dangerous_goods_reefer_and_maintenance_minima_are_not_learnable_away(self):
        source = load_port_dataset("public_cn_sha_integrated_scenario_v5")
        factors = source.factor_values.copy()
        factor_masks = source.factor_availability.copy()
        yard = FACTOR_COLUMNS.index("yard_occupancy_ratio")
        factors[0, yard] = 0.99
        factor_masks[0, yard] = 1.0
        port = source.port_wide_values.copy()
        port_masks = source.port_wide_availability.copy()
        dg = PORT_WIDE_COLUMNS.index("dangerous_goods_workload_ratio")
        port[0, dg] = 1.0
        port_masks[0, dg] = 1.0
        dataset = replace(
            source,
            factor_values=factors,
            factor_availability=factor_masks,
            port_wide_values=port,
            port_wide_availability=port_masks,
        )
        env = _env(dataset)
        env.reset(seed=5, options={"start_index": 0})
        raw = np.zeros(18, dtype=np.float32)
        raw[4], raw[10], raw[12] = 1.0, -1.0, -1.0
        _obs, reward, _terminated, _truncated, info = env.step(raw)
        self.assertEqual(info["coordinated_control"]["yard_flow"], 0.0)
        self.assertEqual(info["coordinated_control"]["reefer_service_ratio"], 0.5)
        self.assertEqual(info["coordinated_control"]["maintenance_reserve_ratio"], 0.5)
        self.assertTrue(np.isfinite(reward))
        reasons = set(info["latent_action_correction_reasons"])
        self.assertIn("dangerous_goods_inflow_feasible_parameterization", reasons)
        self.assertNotIn("reefer_minimum_reserve_parameterization", reasons)
        self.assertNotIn("maintenance_minimum_reserve_parameterization", reasons)
        env.close()

    def test_reward_ledger_and_resource_chain_have_auditable_business_metrics(self):
        env = _env(episode_steps=2)
        observation, _ = env.reset(seed=6, options={"start_index": 0})
        self.assertEqual(observation.shape, env.observation_space.shape)
        raw = np.ones(18, dtype=np.float32)
        _obs, reward, _terminated, _truncated, info = env.step(raw)
        self.assertLessEqual(info["coordinated_reward"], 0.0)
        self.assertAlmostEqual(
            reward,
            info["port_base_reward"] + info["integrated_reward"] + info["coordinated_reward"],
            places=6,
        )
        for key in (
            "rail_backlog",
            "barge_backlog",
            "pilotage_backlog",
            "towage_backlog",
            "terminal_move_chain",
            "resource_imbalance",
            "allocation_change",
            "latent_action_correction",
        ):
            self.assertIn(key, info["coordinated_reward_components"])
        totals = env.totals
        for key in (
            "rail_service_completion_ratio",
            "barge_service_completion_ratio",
            "pilotage_service_completion_ratio",
            "towage_service_completion_ratio",
            "quay_crane_completion_ratio",
            "horizontal_transport_completion_ratio",
            "yard_crane_completion_ratio",
            "latent_action_correction_mean",
        ):
            self.assertIn(key, totals)
            self.assertTrue(np.isfinite(totals[key]))
        projection = env.project_control(raw, soc=0.55, last_bess_kw=0.0)
        self.assertFalse("dispatch_allowed" in projection)
        self.assertIn("recommendation_only", projection["coordinated_authority"])
        env.close()

    def test_operational_minimum_commitments_are_projected_outside_rl(self):
        env = _env()
        env.reset(seed=7, options={"start_index": 0})
        raw = np.full(18, -1.0, dtype=np.float32)
        _obs, _reward, _terminated, _truncated, info = env.step(raw)
        control = info["coordinated_control"]
        self.assertGreaterEqual(control["gate_smoothing_ratio"], 0.80)
        self.assertGreaterEqual(control["rail_allocation_ratio"], 0.85)
        self.assertGreaterEqual(control["barge_allocation_ratio"], 0.85)
        self.assertGreaterEqual(control["quay_crane_allocation_ratio"], 0.90 - 1e-6)
        self.assertGreaterEqual(control["horizontal_transport_allocation_ratio"], 0.90 - 1e-6)
        self.assertGreaterEqual(control["yard_crane_allocation_ratio"], 0.90 - 1e-6)
        reasons = set(info["latent_action_correction_reasons"])
        self.assertNotIn("gate_minimum_service_commitment", reasons)
        self.assertNotIn("quay_minimum_service_commitment", reasons)
        self.assertFalse(info["guardrail_violation"])
        env.close()

    def test_terminal_soc_recovery_reduces_discretionary_shore_power_before_peak_breach(self):
        source = load_port_dataset("public_cn_sha_integrated_forward_2026m05_v5")
        env = CoordinatedPortOperationsEnv(
            source,
            slice(0, source.rows),
            episode_steps=48,
            seed=9,
            demand_cap_kw=36000,
            port_profile=load_profile("cn_sha_coordinated_scenario_v6"),
            normalization_slice=slice(0, 96),
            training=False,
        )
        env.reset(seed=9, options={"start_index": 2822})
        last = None
        for _ in range(48):
            _obs, _reward, _terminated, _truncated, last = env.step(
                np.ones(18, dtype=np.float32)
            )
        self.assertIsNotNone(last)
        self.assertLessEqual(last["net_load_kw"], env.demand_cap_kw + 1e-6)
        self.assertIn(
            "shore_power_grid_headroom_parameterization",
            last["latent_action_correction_reasons"],
        )
        self.assertFalse(last["guardrail_violation"])
        env.close()

    def test_gate_completion_denominator_includes_initial_backlog(self):
        env = _env(episode_steps=1)
        env.reset(seed=8, options={"start_index": 0})
        _obs, _reward, _terminated, _truncated, _info = env.step(
            np.ones(18, dtype=np.float32)
        )
        totals = env.totals
        self.assertLessEqual(totals["gate_service_completion_ratio"], 1.0)
        expected = totals["gate_served_trucks"] / max(
            totals["gate_demand_trucks"] + totals["gate_initial_backlog_trucks"],
            1e-9,
        )
        self.assertAlmostEqual(
            totals["gate_service_completion_ratio"], expected, places=9
        )
        env.close()


if __name__ == "__main__":
    unittest.main()
