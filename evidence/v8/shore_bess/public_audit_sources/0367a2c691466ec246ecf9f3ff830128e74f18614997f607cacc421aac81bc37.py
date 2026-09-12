"""V8 learning contract. V3 physical accounting and historical actors stay intact.

Rewards are exact differences in the reported electricity, degradation and
demand bill, plus an explicitly non-financial carbon constraint multiplier.
Inventory and running-demand shaping telescope to zero with gamma=1.
No teacher actions or future carbon/load rows enter the policy observation.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.rl_model.shore_bess.v3_environment import ShoreBESSEnv, chronological_slices

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
    reward_credit_assignment = "inventory_and_train_threshold_demand_potential_v2"
    def __init__(self, *args, carbon_price=12.0, discrete=True, **kwargs):
        train_slice = kwargs.get("normalization_slice")
        if train_slice is None:
            dataset = args[0] if args else kwargs["dataset"]
            # Generic default matches the historical train-only contract.
            # The V8 runner always supplies its stricter complete-month split.
            train_slice = chronological_slices(dataset)[0]
        kwargs["normalization_slice"] = train_slice
        super().__init__(*args, **kwargs)
        train = self.dataset.values[train_slice].astype(np.float64)
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
        factors = self.dataset.factor_values[train_slice].astype(np.float64)
        berth = factors[:, self.factor_index["berth_occupancy_ratio"]]
        service = self.config["shore_service"]
        train_aux = train[:, 0] * (service["mandatory_load_fraction_min"]
                    + service["mandatory_load_fraction_occupancy_gain"] * berth) * self.flex_fraction
        self.defer_limit_kw = float(np.min(train_aux) * self.flex_limit) * (1.0 - 1e-9)

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

    def _inventory_potential(self):
        inventory = (self._soc - self.soc_initial) * self.energy_kwh - self._flex_backlog_kwh
        return inventory * (self.price_center + self.carbon_price * self.carbon_center) / self.reward_scale

    def _demand_potential(self):
        # A provisional early peak below the TRAIN-only threshold is commonly
        # superseded later in the week. Cancel its temporary credit, without
        # changing the actual billed peak or consulting future episode rows.
        policy_peak = self._totals.get("peak_kw", 0.0)
        raw_delta = policy_peak - self.baseline_peak
        threshold_delta = max(policy_peak, self.soft_cap_kw) - max(self.baseline_peak, self.soft_cap_kw)
        rate = self.config["grid"]["demand_charge_cny_per_kw_month"] * self.episode_steps / (24 * 30.4375)
        return (raw_delta - threshold_delta) * rate / self.reward_scale

    def _potential(self):
        return self._inventory_potential() + self._demand_potential()

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
        action = getattr(self, "_v8_exact_dispatch_action", None) if getattr(self, "_v8_exact_dispatch_action", None) is not None else action
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
        executed = float(np.clip(max(projected["flex_kw"], due) if due > 1e-9
                                else projected["flex_kw"], -flex_capacity, flex_capacity))
        projected["net_kw"] += executed - projected["flex_kw"]
        projected["flex_kw"] = executed
        projected["backlog_after_kwh"] = max(0.0, self._flex_backlog_kwh - executed)
        projected["requested_flex_kw"] = requested
        if abs(executed - requested) > 1e-6:
            projected["projection_reasons"].append("flex_deadline_or_capacity")
            projected["projection_applied"] = True
        return projected

    def action_masks(self):
        """Only mechanical/service feasibility, never price or carbon ranking.

        Remove aliases such as asking an empty battery to discharge, or
        repaying nonexistent work. When the discrete grid cannot exactly
        restore terminal SOC, idle is the explicit safety-repair command.
        """
        ctx = self._row_context()
        equipment = ctx["equipment_availability_ratio"]
        available = self.power_kw * equipment * max(.5, self._soh)
        if equipment < .5 or self._temperature_c >= self.temperature_trip_c:
            available = 0.0
        elif self._temperature_c > self.temperature_derate_c:
            available *= (self.temperature_trip_c - self._temperature_c) / (self.temperature_trip_c - self.temperature_derate_c)
        discharge = max(0.0, min(available, (self._soc - self.soc_min) * self.energy_kwh * self.discharge_eff)
                        - ctx["reserve_required_kw"])
        charge = min(available, max(0.0, (self.soc_max - self._soc) * self.energy_kwh / self.charge_eff),
                     max(0.0, self.hard_pcc_limit_kw - ctx["base_load_kw"]))
        remaining = self.episode_steps - self._step - 1
        band = min(remaining * self.power_kw * min(self.charge_eff, self.discharge_eff) / self.energy_kwh,
                   .18 * remaining / max(1, self.episode_steps - 1))
        due = sum(amount for deadline, amount in self.flex_queue if deadline <= self._step)
        feasible_bess = {}
        for ratio in np.unique(LATTICE[:, 0]):
            power = float(ratio) * self.power_kw
            next_soc = self._soc - max(power, 0) / (self.energy_kwh * self.discharge_eff)
            next_soc += max(-power, 0) * self.charge_eff / self.energy_kwh
            feasible_bess[float(ratio)] = bool(-charge - 1e-6 <= power <= discharge + 1e-6
                and abs(power - self._last_bess_kw) <= self.ramp_kw + 1e-6
                and max(self.soc_min, self.soc_initial - band) - 1e-9 <= next_soc
                <= min(self.soc_max, self.soc_initial + band) + 1e-9)
        if not any(feasible_bess.values()):
            feasible_bess[0.0] = True
        mask = []
        for bess, flex in LATTICE:
            feasible_flex = (flex > 0) if due > 1e-6 else (
                (flex == 0) or (flex > 0 and self._flex_backlog_kwh > 1e-6)
                or (flex < 0 and remaining >= self.flex_deadline_hours
                    and self._flex_backlog_kwh < self.max_backlog_kwh - 1e-6))
            mask.append(feasible_bess[float(bess)] and feasible_flex)
        return np.asarray(mask, dtype=bool)

    def step(self, action):
        if self.discrete:
            value = np.asarray(action).item()
            if not np.isfinite(value) or int(value) != value or not 0 <= value < len(LATTICE):
                raise ValueError("dispatch action outside lattice")
            raw = LATTICE[int(value)]
        else:
            raw = np.asarray(action, dtype=np.float64)
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
        self._v8_exact_dispatch_action = np.asarray(raw, dtype=np.float64).copy()
        try:
            obs, _, terminated, truncated, info = super().step(raw)
        finally:
            self._v8_exact_dispatch_action = None
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
                             "inventory_potential": 0.0 if terminated else self._inventory_potential(),
                             "demand_potential": 0.0 if terminated else self._demand_potential(),
                             "total_potential": potential,
                             "credit_assignment": self.reward_credit_assignment}
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


class ShoreBESSV8SACEnv(ShoreBESSV8Env):
    """Continuous feasible-action coordinates for SAC and TD3.

    Coordinates parameterize the same equipment and service constraints. No
    price, carbon threshold, forecast or rule-policy action enters this map.
    A zero-preserving deadband gives lossy storage an exact idle command.
    """
    action_deadband = 0.1
    action_mapping_version = "physical_feasible_interval_zero_preserving_v1"

    def __init__(self, *args, **kwargs):
        kwargs["discrete"] = False
        super().__init__(*args, **kwargs)

    def _continuous_bounds(self, ctx):
        available = self.power_kw * ctx["equipment_availability_ratio"] * max(.5, self._soh)
        if ctx["equipment_availability_ratio"] < .5 or self._temperature_c >= self.temperature_trip_c:
            available = 0.0
        elif self._temperature_c > self.temperature_derate_c:
            available *= (self.temperature_trip_c - self._temperature_c) / (self.temperature_trip_c - self.temperature_derate_c)
        discharge = max(0.0, min(available, (self._soc - self.soc_min) * self.energy_kwh * self.discharge_eff)
                        - ctx["reserve_required_kw"])
        charge = min(available, max(0.0, (self.soc_max - self._soc) * self.energy_kwh / self.charge_eff),
                     max(0.0, self.hard_pcc_limit_kw - ctx["base_load_kw"]))
        remaining = self.episode_steps - self._step - 1
        band = min(remaining * self.power_kw * min(self.charge_eff, self.discharge_eff) / self.energy_kwh,
                   .18 * remaining / max(1, self.episode_steps - 1))
        def power_for_soc(target):
            return ((self._soc - target) * self.energy_kwh * self.discharge_eff if target <= self._soc
                    else -(target - self._soc) * self.energy_kwh / self.charge_eff)
        lower = max(-charge, self._last_bess_kw - self.ramp_kw,
                    power_for_soc(min(self.soc_max, self.soc_initial + band)))
        upper = min(discharge, self._last_bess_kw + self.ramp_kw,
                    power_for_soc(max(self.soc_min, self.soc_initial - band)))
        if remaining > 0:
            # Leave room to brake the current command next hour. A one-step
            # SOC limit alone can admit 3.6 MW charging right at the envelope,
            # then require a >2.4 MW ramp the following hour. Rated power is
            # below two ramp steps, so this exact braking tail has one term.
            next_band = min((remaining - 1) * self.power_kw * min(self.charge_eff, self.discharge_eff) / self.energy_kwh,
                            .18 * (remaining - 1) / max(1, self.episode_steps - 1))
            charge_room = (min(self.soc_max, self.soc_initial + next_band) - self._soc) * self.energy_kwh / self.charge_eff
            discharge_room = (self._soc - max(self.soc_min, self.soc_initial - next_band)) * self.energy_kwh * self.discharge_eff
            efficiency = self.charge_eff * self.discharge_eff
            def charge_with_braking(room):
                return (room + self.ramp_kw) / 2.0 if room > self.ramp_kw else (efficiency * room + self.ramp_kw) / (1.0 + efficiency)
            def discharge_with_braking(room):
                return (room + self.ramp_kw) / 2.0 if room > self.ramp_kw else (room + efficiency * self.ramp_kw) / (1.0 + efficiency)
            lower = max(lower, -charge_with_braking(charge_room))
            upper = min(upper, discharge_with_braking(discharge_room))
        scale = max(1.0, abs(lower), abs(upper))
        if 0.0 < lower - upper <= 32 * np.finfo(np.float64).eps * scale:
            lower = upper = (lower + upper) / 2.0
        return lower, upper

    def _coordinate(self, value, lower, upper):
        if lower > upper:
            # Expose a physical infeasibility through the shared safety ledger.
            return 0.0
        value = float(np.clip(value, -1, 1))
        magnitude = max(0.0, (abs(value) - self.action_deadband) / (1 - self.action_deadband))
        neutral = float(np.clip(0.0, lower, upper))
        return neutral + magnitude * ((upper - neutral) if value >= 0 else (lower - neutral))

    def step(self, action):
        latent = np.asarray(action, dtype=np.float64)
        if latent.shape != (2,) or not np.isfinite(latent).all():
            raise ValueError("finite two-dimensional continuous dispatch required")
        ctx = self._row_context()
        lower, upper = self._continuous_bounds(ctx)
        capacity = ctx["auxiliary_shore_kw"] * self.flex_limit
        due = sum(amount for deadline, amount in self.flex_queue if deadline <= self._step)
        # Reserve PCC capacity for service due now before choosing storage.
        lower = max(lower, ctx["base_load_kw"] + due - self.hard_pcc_limit_kw)
        bess = self._coordinate(latent[0], lower, upper)
        flex_low = -min(self.defer_limit_kw, capacity, self.max_backlog_kwh - self._flex_backlog_kwh)
        if self.episode_steps - self._step - 1 < self.flex_deadline_hours:
            flex_low = 0.0
        if due > 1e-9:
            flex_low = due
        flex_high = min(capacity, self._flex_backlog_kwh,
                        max(0.0, self.hard_pcc_limit_kw - ctx["base_load_kw"] + bess))
        flex = self._coordinate(latent[1], flex_low, flex_high)
        mapped = np.asarray([bess / self.power_kw, flex / max(capacity, 1e-9)], dtype=np.float64)
        obs, reward, terminated, truncated, info = super().step(mapped)
        info["latent_neural_action"] = latent.tolist()
        info["action_mapping"] = {"version": self.action_mapping_version,
            "deadband": self.action_deadband, "bess_interval_kw": [lower, upper],
            "flex_interval_kw": [flex_low, flex_high], "mapped_action": mapped.tolist()}
        return obs, reward, terminated, truncated, info
