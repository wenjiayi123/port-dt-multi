"""V8 learning contract. V3 physical accounting and historical actors stay intact.

Rewards are exact differences in the reported electricity, degradation and
demand bill, plus an explicitly non-financial carbon constraint multiplier.
Inventory shaping telescopes to zero with gamma=1 and equal terminal stocks.
No teacher actions or future carbon/load rows enter the policy observation.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.rl_model.shore_bess.v3_environment import ShoreBESSEnv

STATE_NAMES = [
    "hour_sin", "hour_cos", "load_centered", "auxiliary_load_ratio",
    "price_centered", "scheduled_tariff_6h_centered", "carbon_centered",
    "soc_inventory", "soh", "temperature", "last_bess_power",
    "flex_backlog", "equipment_availability", "reserve_requirement",
    "episode_progress", "running_policy_peak", "running_baseline_peak",
    "cumulative_carbon_delta", "cumulative_financial_delta",
] + [f"flex_due_in_{hour}_hours" for hour in range(12)]
LATTICE = np.asarray([(b, f) for b in (-0.5, -0.15, 0.0, 0.15, 0.5)
                      for f in (-1.0, 0.0, 1.0)], dtype=np.float32)


class ShoreBESSV8Env(ShoreBESSEnv):
    def __init__(self, *args, carbon_price=12.0, discrete=True, **kwargs):
        train_slice = kwargs.get("normalization_slice")
        if train_slice is None:
            dataset = args[0] if args else kwargs["dataset"]
            cutoff = next((i for i, stamp in enumerate(dataset.timestamps)
                           if stamp >= "2025-05-01T00:00:00Z"), len(dataset.timestamps))
            if cutoff < 169:
                raise ValueError("historical training normalization is required")
            train_slice = slice(0, cutoff)
        kwargs["normalization_slice"] = train_slice
        super().__init__(*args, **kwargs)
        train = self.dataset.values[train_slice]
        self.price_center = float(np.mean(train[:, 4]))
        self.carbon_center = float(np.mean(train[:, 5]))
        self.carbon_span = max(float(np.ptp(train[:, 5])) / 2.0, 1e-6)
        self.load_center = float(np.mean(train[:, 0]))
        self.load_std = max(float(np.std(train[:, 0])), 1.0)
        self.carbon_price = float(carbon_price)
        self.reward_scale = 200.0
        self.discrete = bool(discrete)
        self.observation_space = spaces.Box(-10.0, 10.0, (len(STATE_NAMES),), dtype=np.float32)
        if self.discrete:
            self.action_space = spaces.Discrete(len(LATTICE))
        self.baseline_peak = 0.0
        self.carbon_delta = 0.0
        self.cost_delta = 0.0
        self.previous_potential = 0.0
        self.flex_queue = []
        self.flex_deadline_hours = 12
        # A train-only conservative admission limit. Unexpected lower future
        # capacity is reported as a service violation, never silently shed.
        factors = self.dataset.factor_values[train_slice]
        berth = factors[:, self.factor_index["berth_occupancy_ratio"]]
        service = self.config["shore_service"]
        train_aux = train[:, 0] * (service["mandatory_load_fraction_min"]
                    + service["mandatory_load_fraction_occupancy_gain"] * berth) * self.flex_fraction
        self.defer_limit_kw = float(np.min(train_aux) * self.flex_limit)

    def _row_context(self):
        ctx = super()._row_context()
        stamp = datetime.fromisoformat(self.timestamps[self._start + self._step].replace("Z", "+00:00"))
        local_hour = stamp.astimezone(ZoneInfo("Asia/Shanghai")).hour
        service = self.config["shore_service"]
        reserve = service["reserve_critical_kw"] if local_hour in service["critical_hours_local"] else service["reserve_min_kw"]
        ctx["reserve_required_kw"] = float(reserve) * (0.75 + .5 * ctx["berth_occupancy_ratio"])
        ctx["local_service_hour"] = float(local_hour)
        # Price/carbon remain the pinned engineering scenario, indexed by UTC.
        return ctx

    def _potential(self):
        inventory = (self._soc - self.soc_initial) * self.energy_kwh - self._flex_backlog_kwh
        return inventory * (self.price_center + self.carbon_price * self.carbon_center) / self.reward_scale

    def _observation(self):
        ctx = self._row_context()
        angle = ctx["timestamp_hour"] * 2 * np.pi / 24
        queue = [sum(kwh for due, kwh in self.flex_queue if due - self._step == h) / 200.0
                 for h in range(12)]
        return np.clip(np.asarray([
            np.sin(angle), np.cos(angle),
            (ctx["base_load_kw"] - self.load_center) / self.load_std,
            ctx["auxiliary_shore_kw"] / 400.0,
            (ctx["price_cny_per_kwh"] - self.price_center) / .35,
            (ctx["known_tariff_6h_mean"] - self.price_center) / .35,
            (ctx["carbon_kg_per_kwh"] - self.carbon_center) / self.carbon_span,
            (self._soc - self.soc_initial) / .18,
            self._soh, (self._temperature_c - 25) / 20,
            self._last_bess_kw / self.power_kw,
            self._flex_backlog_kwh / 2000.0,
            ctx["equipment_availability_ratio"], ctx["reserve_required_kw"] / self.power_kw,
            self._step / max(1, self.episode_steps - 1),
            (self._totals.get("peak_kw", 0.0) - self.load_center) / self.load_std,
            (self.baseline_peak - self.load_center) / self.load_std,
            self.carbon_delta / 1000.0, self.cost_delta / 10000.0,
        ] + queue, dtype=np.float32), -10, 10)

    def reset(self, **kwargs):
        self.baseline_peak = self.carbon_delta = self.cost_delta = 0.0
        self.previous_potential = 0.0
        self.flex_queue = []
        obs, info = super().reset(**kwargs)
        self._totals.update(flex_deadline_violation_kwh=0.0, physical_power_violations=0.0,
                            training_reward=0.0, max_flex_age_hours=0.0)
        return obs, info

    def _project(self, action, ctx):
        bounded = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        safe_action = bounded.copy()
        flex_capacity = ctx["auxiliary_shore_kw"] * self.flex_limit
        requested = float(bounded[1]) * flex_capacity
        remaining = self.episode_steps - self._step - 1
        due = sum(kwh for deadline, kwh in self.flex_queue if deadline <= self._step)
        flex = max(requested, due) if due > 1e-9 else requested
        flex = max(flex, -min(self.defer_limit_kw, flex_capacity))
        if remaining < self.flex_deadline_hours:
            flex = max(flex, 0.0)
        safe_action[1] = np.clip(flex / max(flex_capacity, 1e-9), -1, 1)
        projected = super()._project(safe_action, ctx)
        executed = float(np.clip(projected["flex_kw"], -flex_capacity, flex_capacity))
        projected["net_kw"] += executed - projected["flex_kw"]
        projected["flex_kw"] = executed
        projected["backlog_after_kwh"] = max(0.0, self._flex_backlog_kwh - executed)
        projected["requested_flex_kw"] = requested
        if abs(executed - requested) > 1e-6:
            projected["projection_reasons"].append("flex_deadline_or_capacity")
            projected["projection_applied"] = True
        return projected

    def step(self, action):
        if self.discrete:
            value = np.asarray(action).item()
            if not np.isfinite(value) or int(value) != value or not 0 <= value < len(LATTICE):
                raise ValueError("dispatch action outside lattice")
            raw = LATTICE[int(value)]
        else:
            raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (2,) or not np.isfinite(raw).all():
            raise ValueError("finite two-dimensional dispatch required")
        ctx = self._row_context()
        last_peak = self._totals["peak_kw"]
        last_baseline_peak = self.baseline_peak
        self.baseline_peak = max(self.baseline_peak, ctx["base_load_kw"])
        previous_power = self._last_bess_kw
        previous_soh = self._soh
        previous_temperature = self._temperature_c
        timestep = self._step
        obs, _, terminated, truncated, info = super().step(raw)
        flex = info["final_action"]["flex_kw"]
        if flex < 0:
            self.flex_queue.append((timestep + self.flex_deadline_hours, -flex))
        else:
            repayment = flex
            while self.flex_queue and repayment > 1e-9:
                due, amount = self.flex_queue[0]
                repaid = min(amount, repayment)
                self._totals["max_flex_age_hours"] = max(self._totals["max_flex_age_hours"],
                    timestep - (due - self.flex_deadline_hours))
                repayment -= repaid
                if amount <= repaid + 1e-9:
                    self.flex_queue.pop(0)
                else:
                    self.flex_queue[0] = (due, amount - repaid)
        overdue = sum(amount for due, amount in self.flex_queue if due <= timestep)
        self._totals["flex_deadline_violation_kwh"] += overdue
        available = self.power_kw * ctx["equipment_availability_ratio"] * max(.5, previous_soh)
        if ctx["equipment_availability_ratio"] < .5 or previous_temperature >= self.temperature_trip_c:
            available = 0.0
        elif previous_temperature > self.temperature_derate_c:
            available *= (self.temperature_trip_c - previous_temperature) / (self.temperature_trip_c - self.temperature_derate_c)
        power = info["final_action"]["bess_kw"]
        physical_bad = abs(power) > available + 1e-6 or abs(power - previous_power) > self.ramp_kw + 1e-6
        self._totals["physical_power_violations"] += float(physical_bad)
        if (physical_bad or overdue > 1e-6) and not info["guardrail_violation"]:
            self._totals["guardrail_violations"] += 1.0
            info["guardrail_violation"] = True
        demand_rate = self.config["grid"]["demand_charge_cny_per_kw_month"] * self.episode_steps / (24 * 30.4375)
        bill_delta = (info["business_step"]["energy_cost_cny"]
                      - ctx["base_load_kw"] * ctx["price_cny_per_kwh"]
                      + info["business_step"]["degradation_cost_cny"]
                      + demand_rate * ((self._totals["peak_kw"] - last_peak)
                                       - (self.baseline_peak - last_baseline_peak)))
        carbon_delta = info["business_step"]["carbon_kg"] - ctx["base_load_kw"] * ctx["carbon_kg_per_kwh"]
        self.cost_delta += bill_delta
        self.carbon_delta += carbon_delta
        potential = 0.0 if terminated else self._potential()
        reward = -(bill_delta + self.carbon_price * carbon_delta) / self.reward_scale
        reward += potential - self.previous_potential
        self.previous_potential = potential
        reward -= 100.0 * float(info["guardrail_violation"])
        info["v8_reward"] = {"bill_delta_cny": bill_delta, "carbon_delta_kg": carbon_delta,
                             "carbon_shadow_price_cny_per_kg": self.carbon_price,
                             "inventory_potential": potential}
        self._totals["training_reward"] += reward
        info["observation_contract"] = "shore_bess_v8_31d"
        info["flex_queue"] = [[int(due), float(amount)] for due, amount in self.flex_queue]
        info["flex_overdue_kwh"] = overdue
        info["physical_power_violation"] = bool(physical_bad)
        if terminated:
            self._totals["guardrail_violation_rate"] = self._totals["guardrail_violations"] / self.episode_steps
            self._totals["financial_delta_cny"] = self.cost_delta
            self._totals["carbon_delta_kg"] = self.carbon_delta
            info["episode_metrics"] = self.totals
        else:
            obs = self._observation()
        return obs, float(reward), terminated, truncated, info
