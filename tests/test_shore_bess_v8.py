"""Independent accounting and causal-service checks for the V8 learning env."""
from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.services.rl_model.shore_bess.v3_environment import chronological_slices, load_config
from app.services.rl_model.shore_bess.v8_environment import LATTICE, STATE_NAMES, ShoreBESSV8Env, ShoreBESSV8SACEnv
from app.services.rl_training.datasets import FACTOR_COLUMNS, PortDataset


def fixture_dataset(rows=960):
    """Known hourly inputs make meter and SOC checks independent of reports."""
    hour = np.arange(rows, dtype=np.float64)
    values = np.column_stack([
        20_000 + 800 * np.sin(hour * 2 * np.pi / 24),
        np.full(rows, 200), np.full(rows, 5), np.full(rows, 2),
        0.8 + 0.25 * np.sin(hour * 2 * np.pi / 24),
        0.55 + 0.08 * np.cos(hour * 2 * np.pi / 24), np.full(rows, 24),
    ]).astype(np.float32)
    factors = np.ones((rows, len(FACTOR_COLUMNS)), dtype=np.float32)
    factors[:, FACTOR_COLUMNS.index("berth_occupancy_ratio")] = 0.5
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return PortDataset(
        dataset_id="v8-physical-test", path=Path(__file__),
        timestamps=[(start + timedelta(hours=int(h))).isoformat() for h in hour],
        values=values, metadata={"sha256": "0" * 64},
        factor_values=factors, factor_availability=np.ones_like(factors),
    )


class ShoreBESSV8Tests(unittest.TestCase):
    def make_env(self, *, dataset=None, episode=48, env_type=ShoreBESSV8Env, **kwargs):
        dataset = dataset or fixture_dataset()
        train, _, test = chronological_slices(dataset)
        env = env_type(
            dataset, test, config=load_config(), normalization_slice=train,
            episode_steps=episode, training=False, discrete=False, **kwargs,
        )
        self.addCleanup(env.close)
        env.reset(options={"start_index": 0})
        return env

    def test_executed_meter_power_soc_and_total_energy_are_conserved(self):
        env = self.make_env(episode=168)
        rng = np.random.default_rng(812)
        base_energy = meter_energy = charge = discharge = flex_total = 0.0
        cost = carbon = wear = peak = 0.0
        for _ in range(env.episode_steps):
            previous_soc = env._soc
            _, _, done, truncated, info = env.step(rng.uniform(-1, 1, 2))
            ctx = info["context"]
            power, flex = info["final_action"]["bess_kw"], info["final_action"]["flex_kw"]
            meter = ctx["base_load_kw"] - power + flex
            expected_soc = previous_soc - max(power, 0) / (env.energy_kwh * env.discharge_eff)
            expected_soc += max(-power, 0) * env.charge_eff / env.energy_kwh
            self.assertAlmostEqual(info["soc"], expected_soc, places=10)
            self.assertAlmostEqual(info["pcc_kw"], meter, places=8)
            self.assertAlmostEqual(info["business_step"]["energy_cost_cny"], meter * ctx["price_cny_per_kwh"], places=8)
            self.assertAlmostEqual(info["business_step"]["carbon_kg"], meter * ctx["carbon_kg_per_kwh"], places=8)
            self.assertLessEqual(abs(flex), ctx["auxiliary_shore_kw"] * env.flex_limit + 1e-6)
            self.assertAlmostEqual(sum(amount for _, amount in env.flex_queue), env._flex_backlog_kwh, places=6)
            base_energy += ctx["base_load_kw"]
            meter_energy += meter
            charge += max(-power, 0)
            discharge += max(power, 0)
            flex_total += flex
            cost += meter * ctx["price_cny_per_kwh"]
            carbon += meter * ctx["carbon_kg_per_kwh"]
            wear += abs(power) * env.cycle_cost
            peak = max(peak, meter)
            self.assertFalse(truncated)
        self.assertTrue(done)
        self.assertAlmostEqual(env._soc, env.soc_initial, places=10)
        self.assertAlmostEqual(discharge, charge * env.charge_eff * env.discharge_eff, places=6)
        self.assertAlmostEqual(flex_total, 0, places=6)
        self.assertAlmostEqual(meter_energy - base_energy, charge - discharge, places=6)
        self.assertAlmostEqual(env.totals["carbon_kg"], carbon, places=6)
        demand = peak * env.config["grid"]["demand_charge_cny_per_kw_month"] * env.episode_steps / (24 * 30.4375)
        self.assertAlmostEqual(env.totals["total_cost_cny"], cost + wear + demand, places=6)

    def test_gamma_one_reward_matches_actual_bill_carbon_and_safety_penalties(self):
        env = self.make_env(carbon_price=12.0)
        baseline_cost = baseline_carbon = baseline_peak = reward_sum = 0.0
        violations = 0
        for step in range(env.episode_steps):
            action = [-0.15, -1.0] if step % 12 < 6 else [0.15, 1.0]
            _, reward, done, _, info = env.step(action)
            ctx = info["context"]
            baseline_cost += ctx["base_load_kw"] * ctx["price_cny_per_kwh"]
            baseline_carbon += ctx["base_load_kw"] * ctx["carbon_kg_per_kwh"]
            baseline_peak = max(baseline_peak, ctx["base_load_kw"])
            reward_sum += reward
            violations += int(info["guardrail_violation"])
        self.assertTrue(done)
        baseline_cost += baseline_peak * env.config["grid"]["demand_charge_cny_per_kw_month"] * env.episode_steps / (24 * 30.4375)
        financial_delta = env.totals["total_cost_cny"] - baseline_cost
        carbon_delta = env.totals["carbon_kg"] - baseline_carbon
        expected = -(financial_delta + env.carbon_price * carbon_delta) / env.reward_scale - 100 * violations
        self.assertAlmostEqual(reward_sum, expected, places=7)
        self.assertAlmostEqual(env.totals["financial_delta_cny"], financial_delta, places=6)
        self.assertAlmostEqual(env.totals["carbon_delta_kg"], carbon_delta, places=6)
        self.assertAlmostEqual(env.totals["training_reward"], reward_sum, places=8)
        self.assertEqual(env.previous_potential, 0.0)

    def test_idle_cannot_create_reward_cost_saving_or_carbon_credit(self):
        env = self.make_env()
        for _ in range(env.episode_steps):
            _, reward, _, _, info = env.step([0, 0])
            self.assertEqual(reward, 0.0)
            self.assertEqual(info["final_action"], {"bess_kw": 0.0, "flex_kw": 0.0})
        self.assertEqual(env.totals["financial_delta_cny"], 0.0)
        self.assertEqual(env.totals["carbon_delta_kg"], 0.0)
        self.assertEqual(env.totals["training_reward"], 0.0)

    def test_demand_credit_cancels_provisional_peaks_but_keeps_true_bill(self):
        env = self.make_env()
        env.soft_cap_kw = 1e6  # Every provisional peak stays below the threshold.
        reward_sum = 0.0
        for step in range(env.episode_steps):
            previous_inventory = env._inventory_potential()
            _, reward, done, _, info = env.step([0.0, -0.3 if step % 12 < 6 else 0.3])
            reward_sum += reward
            if not done:
                ctx = info["context"]
                energy_wear_delta = (info["business_step"]["energy_cost_cny"]
                    - ctx["base_load_kw"] * ctx["price_cny_per_kwh"]
                    + info["business_step"]["degradation_cost_cny"])
                expected = -(energy_wear_delta + env.carbon_price * info["v8_reward"]["carbon_delta_kg"]) / env.reward_scale
                expected += env._inventory_potential() - previous_inventory
                expected -= 100 * int(info["guardrail_violation"])
                self.assertAlmostEqual(reward, expected, places=9)
        expected_total = -(env.cost_delta + env.carbon_price * env.carbon_delta) / env.reward_scale
        expected_total -= 100 * env.totals["guardrail_violations"]
        self.assertAlmostEqual(reward_sum, expected_total, places=7)
        self.assertEqual(info["v8_reward"]["total_potential"], 0.0)

    def test_fifo_repayment_consumes_oldest_work_first(self):
        env = self.make_env()
        env.step([0, -1])
        first_due, first_amount = env.flex_queue[0]
        env.step([0, -1])
        second_due, second_amount = env.flex_queue[1]
        capacity = env._row_context()["auxiliary_shore_kw"] * env.flex_limit
        repay = first_amount / 2
        _, _, _, _, info = env.step([0, repay / capacity])
        actual = info["final_action"]["flex_kw"]
        self.assertAlmostEqual(actual, repay, places=4)
        self.assertEqual(env.flex_queue[0][0], first_due)
        self.assertAlmostEqual(env.flex_queue[0][1], first_amount - actual, places=6)
        self.assertEqual(env.flex_queue[1][0], second_due)
        self.assertAlmostEqual(env.flex_queue[1][1], second_amount, places=6)

    def test_repeated_deferral_respects_twelve_hour_deadlines_and_terminal_recovery(self):
        env = self.make_env(episode=48)
        postponed = repaid = 0.0
        for step in range(env.episode_steps):
            _, _, done, _, info = env.step([0, -1])
            flex = info["final_action"]["flex_kw"]
            postponed += max(-flex, 0)
            repaid += max(flex, 0)
            self.assertAlmostEqual(info["flex_overdue_kwh"], 0.0, places=6)
            self.assertTrue(all(due > step for due, _ in env.flex_queue))
            if env.episode_steps - step - 1 < env.flex_deadline_hours:
                self.assertGreaterEqual(flex, 0.0)
        self.assertTrue(done)
        self.assertGreater(postponed, 0)
        self.assertAlmostEqual(repaid, postponed, places=6)
        self.assertEqual(env.flex_queue, [])
        self.assertEqual(env.totals["terminal_flex_backlog_kwh"], 0)
        self.assertLessEqual(env.totals["max_flex_age_hours"], 12)

    def test_unexpected_capacity_loss_is_metered_and_reported_as_service_violation(self):
        env = self.make_env()
        env.step([0, -1])
        debt = env._flex_backlog_kwh
        for _ in range(11):
            env.step([0, 0])
        env.segment[env._start + env._step, 0] = 0.0
        _, _, _, _, info = env.step([0, -1])
        self.assertEqual(info["final_action"]["flex_kw"], 0.0)
        self.assertAlmostEqual(info["flex_overdue_kwh"], debt, places=6)
        self.assertTrue(info["guardrail_violation"])
        self.assertGreater(env.totals["flex_deadline_violation_kwh"], 0.0)
        self.assertAlmostEqual(env._flex_backlog_kwh, debt, places=6)

    def test_impossible_terminal_recovery_is_not_marked_physically_safe(self):
        env = self.make_env()
        for _ in range(env.episode_steps - 2):
            env.step([0, 0])
        env.step([0.15, 0])
        self.assertLess(env._soc, env.soc_initial)
        idx = env._start + env._step
        env.segment_factor_values[idx, env.factor_index["equipment_availability_ratio"]] = 0.0
        before = env._soc
        _, _, done, _, info = env.step([0, 0])
        power = info["final_action"]["bess_kw"]
        expected = before - max(power, 0) / (env.energy_kwh * env.discharge_eff) + max(-power, 0) * env.charge_eff / env.energy_kwh
        self.assertTrue(done)
        self.assertAlmostEqual(info["soc"], expected, places=10)
        self.assertTrue(info["physical_power_violation"] or env.totals["terminal_soc_error"] > 1e-6)
        if info["physical_power_violation"]:
            self.assertTrue(info["guardrail_violation"])
            self.assertGreater(env.totals["guardrail_violation_rate"], 0.0)

    def test_training_normalizers_and_observation_ignore_unseen_load_and_carbon(self):
        original = fixture_dataset()
        changed_values = original.values.copy()
        train, _, test = chronological_slices(original)
        changed_values[test.start + 1 :, 0] *= 3
        changed_values[test.start + 1 :, 5] *= 5
        changed = replace(original, values=changed_values)
        first = self.make_env(dataset=original)
        second = self.make_env(dataset=changed)
        for name in ("price_center", "carbon_center", "carbon_span", "load_center", "load_std", "defer_limit_kw"):
            self.assertEqual(getattr(first, name), getattr(second, name))
        np.testing.assert_array_equal(first._observation(), second._observation())
        self.assertAlmostEqual(first.carbon_center, float(np.mean(original.values[train, 5])), places=7)
        self.assertEqual(len(STATE_NAMES), len(first._observation()))

    def test_omitted_normalizer_still_uses_chronological_training_only(self):
        dataset = fixture_dataset()
        _, _, test = chronological_slices(dataset)
        values = dataset.values.copy()
        values[test, 0] *= 2
        values[test, 5] *= 4
        dataset = replace(dataset, values=values)
        explicit = self.make_env(dataset=dataset)
        for options in ({}, {"normalization_slice": None}):
            with self.subTest(options=options):
                env = ShoreBESSV8Env(dataset, test, config=load_config(), training=False, **options)
                self.addCleanup(env.close)
                self.assertEqual(env.carbon_center, explicit.carbon_center)
                self.assertEqual(env.load_center, explicit.load_center)

    def test_queue_observation_labels_identify_current_and_future_due_work(self):
        env = self.make_env()
        env.step([0, -1])
        due, amount = env.flex_queue[0]
        obs = env._observation()
        horizon = due - env._step
        self.assertEqual(horizon, 11)
        self.assertAlmostEqual(float(obs[STATE_NAMES.index(f"flex_due_in_{horizon}_hours")]), amount / 200, places=6)
        self.assertEqual(float(obs[STATE_NAMES.index("flex_due_in_0_hours")]), 0.0)

    def test_reserve_clock_converts_utc_to_shanghai_local_service_hours(self):
        env = self.make_env()
        idx = env._start
        for stamp, local_hour, critical in (("2025-01-01T01:00:00+00:00", 9, True), ("2025-01-01T09:00:00+00:00", 17, False)):
            with self.subTest(stamp=stamp):
                env.timestamps[idx] = stamp
                context = env._row_context()
                service = env.config["shore_service"]
                expected = service["reserve_critical_kw" if critical else "reserve_min_kw"] * (0.75 + 0.5 * context["berth_occupancy_ratio"])
                self.assertEqual(context["local_service_hour"], local_hour)
                self.assertEqual(context["reserve_required_kw"], expected)

    def test_discrete_commands_reject_out_of_lattice_indices(self):
        dataset = fixture_dataset()
        train, _, test = chronological_slices(dataset)
        env = ShoreBESSV8Env(dataset, test, config=load_config(), normalization_slice=train, training=False)
        self.addCleanup(env.close)
        env.reset()
        for invalid in (-1, len(LATTICE), 0.5, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                env.step(invalid)

    def test_continuous_random_week_respects_physics_service_and_terminal_stocks(self):
        env = self.make_env(episode=168, env_type=ShoreBESSV8SACEnv)
        rng = np.random.default_rng(918)
        meter_energy = baseline_energy = 0.0
        charge = discharge = 0.0
        for _ in range(env.episode_steps):
            before = env._soc
            _, _, _, _, info = env.step(rng.uniform(-1, 1, 2))
            power = info["final_action"]["bess_kw"]
            expected = before - max(power, 0) / (env.energy_kwh * env.discharge_eff) + max(-power, 0) * env.charge_eff / env.energy_kwh
            self.assertAlmostEqual(info["soc"], expected, places=10)
            self.assertFalse(info["guardrail_violation"])
            self.assertFalse(info["physical_power_violation"])
            self.assertEqual(info["flex_overdue_kwh"], 0.0)
            meter_energy += info["pcc_kw"]
            baseline_energy += info["context"]["base_load_kw"]
            charge += max(-power, 0)
            discharge += max(power, 0)
        self.assertAlmostEqual(env.totals["terminal_soc_error"], 0.0, places=9)
        self.assertAlmostEqual(env.totals["terminal_flex_backlog_kwh"], 0.0, places=6)
        self.assertAlmostEqual(meter_energy - baseline_energy, charge - discharge, places=6)

    def test_continuous_idle_deadband_cannot_create_unrequested_energy(self):
        env = self.make_env(env_type=ShoreBESSV8SACEnv)
        commands = ([0, 0], [-0.1, 0.1], [0.1, -0.1], [0.05, -0.05])
        for step in range(env.episode_steps):
            _, reward, _, _, info = env.step(commands[step % len(commands)])
            self.assertEqual(info["final_action"], {"bess_kw": 0.0, "flex_kw": 0.0})
            self.assertEqual(reward, 0.0)
        self.assertEqual(env.totals["financial_delta_cny"], 0.0)
        self.assertEqual(env.totals["carbon_delta_kg"], 0.0)

    def test_exact_braking_boundary_can_cross_zero_without_false_violation(self):
        env = self.make_env(episode=168, env_type=ShoreBESSV8SACEnv)
        # Reachable boundary from the r3 failure, translated to this fixture's
        # initial SOC. A float32 physical command used to overcharge by ~0.1 W.
        env._step = 2
        env._soc = env.soc_initial + 0.1176119680413574
        env._last_bess_kw = -3725.6233354873657
        lower, upper = env._continuous_bounds(env._row_context())
        self.assertLessEqual(lower, upper)
        _, _, _, _, first = env.step([-1.0, 0.0])
        self.assertAlmostEqual(first["requested_action"]["bess_kw"], lower, places=9)
        self.assertAlmostEqual(first["final_action"]["bess_kw"], lower, places=9)
        lower, upper = env._continuous_bounds(env._row_context())
        self.assertLessEqual(lower, upper)
        _, _, _, _, second = env.step([-1.0, 0.0])
        _, _, _, _, third = env.step([0.0, 0.0])
        self.assertLess(second["final_action"]["bess_kw"], 0.0)
        self.assertGreater(third["final_action"]["bess_kw"], 0.0)
        for info in (first, second, third):
            self.assertFalse(info["physical_power_violation"])
            self.assertFalse(info["guardrail_violation"])
        self.assertIsNone(env._exact_projection_action)

    def test_saturated_continuous_commands_preserve_braking_and_terminal_stock(self):
        for direction in (-1.0, 1.0):
            with self.subTest(direction=direction):
                env = self.make_env(episode=168, env_type=ShoreBESSV8SACEnv)
                for _ in range(env.episode_steps):
                    lower, upper = env._continuous_bounds(env._row_context())
                    self.assertLessEqual(lower, upper)
                    _, _, _, _, info = env.step([direction, -1.0])
                    self.assertFalse(info["physical_power_violation"])
                    self.assertFalse(info["guardrail_violation"])
                self.assertAlmostEqual(env.totals["terminal_soc_error"], 0.0, places=9)
                self.assertAlmostEqual(env.totals["terminal_flex_backlog_kwh"], 0.0, places=6)

    def test_continuous_physical_mapping_does_not_supply_a_price_or_carbon_teacher(self):
        first = self.make_env(env_type=ShoreBESSV8SACEnv)
        second = self.make_env(env_type=ShoreBESSV8SACEnv)
        second.segment[:, 4] *= 3
        second.segment[:, 5] *= 5
        for command in ([-0.5, -0.5], [0.5, 0.5], [0.8, 0], [-1, -1]):
            a, b = first.step(command)[4], second.step(command)[4]
            self.assertEqual(a["action_mapping"], b["action_mapping"])
            self.assertEqual(a["final_action"], b["final_action"])
            self.assertNotEqual(a["business_step"]["carbon_kg"], b["business_step"]["carbon_kg"])


if __name__ == "__main__":
    unittest.main()
