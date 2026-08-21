from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from .datasets import FACTOR_COLUMNS, NUMERIC_COLUMNS, REGULATORY_COLUMNS, PortDataset
from .environment import PortOperationsEnv


class RegulatoryPortOperationsEnv(PortOperationsEnv):
    """V4 port environment with stateful maritime/customs delay propagation.

    Regulatory decisions are exogenous. The agent can only recommend inspection
    readiness and post-release recovery priority; it cannot change an inspection
    result, release a vessel, or exercise production authority.
    """

    OBSERVATION_DIMENSIONS = 13 + 2 * len(FACTOR_COLUMNS) + 2 * len(REGULATORY_COLUMNS) + 4
    ACTION_DIMENSIONS = 7
    SAFETY_REVISION = "v4_terminal_soc_then_ramp_recheck_v1"

    def __init__(
        self,
        dataset: PortDataset,
        data_slice: slice,
        *,
        action_mode: str = "continuous",
        episode_steps: int = 48,
        seed: int = 42,
        demand_cap_kw: float = 3500.0,
        reward_weights: Optional[Dict[str, float]] = None,
        environment_version: str = "port_ops_v4",
        port_profile: Optional[Dict[str, Any]] = None,
        projection_penalty_weight: float = 0.0,
        regulatory_delay_penalty_weight: float = 0.35,
        normalization_slice: Optional[slice] = None,
        training: bool = True,
        record_trace: bool = False,
    ) -> None:
        if environment_version != "port_ops_v4":
            raise ValueError("RegulatoryPortOperationsEnv requires port_ops_v4")
        super().__init__(
            dataset,
            data_slice,
            action_mode="continuous",
            episode_steps=episode_steps,
            seed=seed,
            demand_cap_kw=demand_cap_kw,
            reward_weights=reward_weights,
            environment_version="port_ops_v3",
            port_profile=port_profile,
            projection_penalty_weight=projection_penalty_weight,
            normalization_slice=normalization_slice,
            training=training,
            record_trace=record_trace,
        )
        self.environment_version = "port_ops_v4"
        self.action_mode = action_mode
        self.regulatory_delay_penalty_weight = max(
            0.0, float(regulatory_delay_penalty_weight)
        )
        self.segment_regulatory_values = dataset.regulatory_values[data_slice].astype(
            np.float32, copy=True
        )
        self.segment_regulatory_availability = dataset.regulatory_availability[
            data_slice
        ].astype(np.float32, copy=True)
        normalization_train = normalization_slice or dataset.split()[0]
        regulatory_reference = dataset.regulatory_values[normalization_train].astype(
            np.float32, copy=False
        )
        regulatory_mask_reference = dataset.regulatory_availability[
            normalization_train
        ].astype(np.float32, copy=False)
        self._regulatory_mins = np.zeros(len(REGULATORY_COLUMNS), dtype=np.float32)
        self._regulatory_spans = np.ones(len(REGULATORY_COLUMNS), dtype=np.float32)
        for index in range(len(REGULATORY_COLUMNS)):
            available = regulatory_mask_reference[:, index] > 0.5
            if np.any(available):
                observed = regulatory_reference[available, index]
                self._regulatory_mins[index] = float(np.min(observed))
                self._regulatory_spans[index] = max(
                    float(np.max(observed) - np.min(observed)), 1e-6
                )
        train_values = dataset.values[normalization_train].astype(np.float32, copy=False)
        self._reference_arrivals = max(0.05, float(np.mean(train_values[:, 2])))
        self._reference_throughput = max(1.0, float(np.mean(train_values[:, 1])))
        limits = self.port_profile["control_limits"]
        self.inspection_buffer_limit = float(limits["inspection_buffer_limit"])
        self.recovery_priority_limit = float(limits["recovery_priority_limit"])
        regulatory = self.port_profile["regulatory_operations"]
        self.maritime_inspection_capacity = float(
            regulatory["maritime_inspection_capacity_vessels_per_hour"]
        )
        self.customs_inspection_capacity = float(
            regulatory["customs_inspection_capacity_vessels_per_hour"]
        )
        self.inspection_buffer_capacity_gain = float(
            regulatory["inspection_buffer_capacity_gain"]
        )
        self.inspection_buffer_service_reserve_fraction = float(
            regulatory["inspection_buffer_service_reserve_fraction"]
        )
        self.recovery_service_capacity_gain = float(
            regulatory["recovery_service_capacity_gain"]
        )
        self.inspection_readiness_load_fraction = float(
            regulatory["inspection_readiness_load_fraction"]
        )
        self.recovery_load_fraction = float(regulatory["recovery_load_fraction"])
        self.observation_space = spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(self.OBSERVATION_DIMENSIONS,),
            dtype=np.float32,
        )
        if action_mode == "continuous":
            self._discrete_actions = None
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.ACTION_DIMENSIONS,),
                dtype=np.float32,
            )
        elif action_mode == "discrete":
            allocation_pairs = (
                (-self.berth_priority_limit, -self.yard_flow_limit),
                (0.0, 0.0),
                (self.berth_priority_limit, self.yard_flow_limit),
                (self.berth_priority_limit, 0.0),
                (0.0, self.yard_flow_limit),
            )
            regulatory_pairs = (
                (0.0, 0.0),
                (self.inspection_buffer_limit, 0.0),
                (0.0, self.recovery_priority_limit),
                (self.inspection_buffer_limit, self.recovery_priority_limit),
                (-self.inspection_buffer_limit, -self.recovery_priority_limit),
            )
            lattice = [
                (bess, service, flexible, berth, yard, buffer, recovery)
                for bess in (-1.0, -0.5, 0.0, 0.5, 1.0)
                for service in (self.service_min, 1.0, self.service_max)
                for flexible in (-self.flexible_limit, 0.0, self.flexible_limit)
                for berth, yard in allocation_pairs
                for buffer, recovery in regulatory_pairs
            ]
            self._discrete_actions = np.asarray(lattice, dtype=np.float32)
            self.action_space = spaces.Discrete(len(self._discrete_actions))
        else:
            raise ValueError(f"unsupported action_mode: {action_mode}")
        self._maritime_hold_vessels = 0.0
        self._customs_hold_vessels = 0.0
        self._maritime_hold_teu = 0.0
        self._customs_hold_teu = 0.0
        self._recovery_queue = 0.0

    def _decode_action(self, action: Any) -> np.ndarray:
        if self.action_mode == "discrete":
            return self._discrete_actions[int(action)].copy()
        continuous = np.asarray(action, dtype=np.float32).reshape(
            self.ACTION_DIMENSIONS
        )
        continuous = np.clip(continuous, -1.0, 1.0)
        continuous[1] = (
            1.0 + (self.service_max - 1.0) * continuous[1]
            if continuous[1] >= 0
            else 1.0 + (1.0 - self.service_min) * continuous[1]
        )
        continuous[2] *= self.flexible_limit
        continuous[3] *= self.berth_priority_limit
        continuous[4] *= self.yard_flow_limit
        continuous[5] *= self.inspection_buffer_limit
        continuous[6] *= self.recovery_priority_limit
        return continuous

    def _regulatory_observation(
        self, values: np.ndarray, mask: np.ndarray
    ) -> list[float]:
        normalized = 2.0 * (
            values - self._regulatory_mins
        ) / self._regulatory_spans - 1.0
        normalized = np.where(mask > 0.5, normalized, 0.0)
        return [
            *normalized.astype(np.float32).tolist(),
            *mask.astype(np.float32).tolist(),
        ]

    def _internal_regulatory_observation(self) -> list[float]:
        vessel_scale = max(1.0, self._reference_arrivals * 4.0)
        work_scale = max(1.0, self._reference_throughput * 4.0)
        return [
            float(np.clip(self._maritime_hold_vessels / vessel_scale, 0.0, 1.5)),
            float(np.clip(self._customs_hold_vessels / vessel_scale, 0.0, 1.5)),
            float(
                np.clip(
                    (self._maritime_hold_teu + self._customs_hold_teu)
                    / work_scale,
                    0.0,
                    1.5,
                )
            ),
            float(np.clip(self._recovery_queue / work_scale, 0.0, 1.5)),
        ]

    def _observation(self) -> np.ndarray:
        row = self._row()
        normalized = 2.0 * (row - self._mins) / self._spans - 1.0
        index = self._start + self._local_step
        timestamp = datetime.fromisoformat(
            self.segment_timestamps[index].replace("Z", "+00:00")
        )
        hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
        return np.asarray(
            [
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                *normalized.tolist(),
                *self._factor_observation(
                    self.segment_factor_values[index],
                    self.segment_factor_availability[index],
                ),
                *self._regulatory_observation(
                    self.segment_regulatory_values[index],
                    self.segment_regulatory_availability[index],
                ),
                2.0 * self._soc - 1.0,
                np.clip(self._queue / max(1.0, row[1] * 4.0), 0.0, 1.5),
                self._last_bess_kw / self.bess_power_kw,
                self._local_step / max(1, self.episode_steps - 1),
                *self._internal_regulatory_observation(),
            ],
            dtype=np.float32,
        )

    def observation_from_state(self, state: Dict[str, Any]) -> np.ndarray:
        missing = [column for column in NUMERIC_COLUMNS if state.get(column) is None]
        if missing:
            raise ValueError(
                f"state is missing canonical fields: {', '.join(missing)}"
            )
        row = np.asarray(
            [float(state[column]) for column in NUMERIC_COLUMNS], dtype=np.float32
        )
        if not np.all(np.isfinite(row)):
            raise ValueError("state contains non-finite canonical values")
        normalized = 2.0 * (row - self._mins) / self._spans - 1.0
        hour = int(state.get("hour", 0)) % 24
        factor_values = np.zeros(len(FACTOR_COLUMNS), dtype=np.float32)
        factor_mask = np.zeros(len(FACTOR_COLUMNS), dtype=np.float32)
        for index, column in enumerate(FACTOR_COLUMNS):
            if state.get(column) is not None:
                factor_values[index] = float(state[column])
                factor_mask[index] = 1.0
        regulatory_values = np.zeros(len(REGULATORY_COLUMNS), dtype=np.float32)
        regulatory_mask = np.zeros(len(REGULATORY_COLUMNS), dtype=np.float32)
        for index, column in enumerate(REGULATORY_COLUMNS):
            if state.get(column) is not None:
                regulatory_values[index] = float(state[column])
                regulatory_mask[index] = 1.0
        vessel_scale = max(1.0, self._reference_arrivals * 4.0)
        work_scale = max(1.0, self._reference_throughput * 4.0)
        return np.asarray(
            [
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                *normalized.tolist(),
                *self._factor_observation(factor_values, factor_mask),
                *self._regulatory_observation(
                    regulatory_values, regulatory_mask
                ),
                2.0 * float(state.get("soc", 0.55)) - 1.0,
                np.clip(
                    float(state.get("queue", 0.0))
                    / max(1.0, float(state["throughput_teu"]) * 4.0),
                    0.0,
                    1.5,
                ),
                float(state.get("last_bess_kw", 0.0)) / self.bess_power_kw,
                np.clip(float(state.get("episode_progress", 0.0)), 0.0, 1.0),
                np.clip(
                    float(state.get("maritime_hold_vessels", 0.0)) / vessel_scale,
                    0.0,
                    1.5,
                ),
                np.clip(
                    float(state.get("customs_hold_vessels", 0.0)) / vessel_scale,
                    0.0,
                    1.5,
                ),
                np.clip(
                    float(state.get("regulatory_hold_teu", 0.0)) / work_scale,
                    0.0,
                    1.5,
                ),
                np.clip(
                    float(state.get("recovery_queue_teu", 0.0)) / work_scale,
                    0.0,
                    1.5,
                ),
            ],
            dtype=np.float32,
        )

    def describe_action(self, action: Any) -> Dict[str, float]:
        control = self._decode_action(action)
        return {
            "bess_kw": round(float(control[0]) * self.bess_power_kw, 6),
            "service_factor": round(float(control[1]), 6),
            "flexible_load_command": round(float(control[2]), 6),
            "berth_priority": round(float(control[3]), 6),
            "yard_flow_command": round(float(control[4]), 6),
            "inspection_buffer": round(float(control[5]), 6),
            "recovery_priority": round(float(control[6]), 6),
        }

    def project_control(self, action: Any, **kwargs: Any) -> Dict[str, Any]:
        projected = super().project_control(action, **kwargs)
        control = self._decode_action(action)
        # The inherited terminal-SOC reachability projection can change BESS
        # power after the first ramp clamp. V4 rechecks the final command so a
        # terminal correction cannot reintroduce a ramp violation.
        previous_bess_kw = float(kwargs.get("last_bess_kw", 0.0))
        soc = float(kwargs.get("soc", 0.55))
        ramp_low = previous_bess_kw - self.bess_power_kw
        ramp_high = previous_bess_kw + self.bess_power_kw
        before_ramp_recheck = float(projected["bess_kw"])
        safe_bess_kw = float(np.clip(before_ramp_recheck, ramp_low, ramp_high))
        if abs(safe_bess_kw - before_ramp_recheck) > 1e-6:
            reasons = list(projected["projection_reasons"])
            reasons.append("v4_post_terminal_ramp_recheck")
            projected["projection_reasons"] = list(dict.fromkeys(reasons))
            projected["bess_kw"] = round(safe_bess_kw, 6)
            if safe_bess_kw >= 0:
                projected_soc = (
                    soc
                    + safe_bess_kw
                    * self.step_hours
                    * 0.96
                    / self.bess_capacity_kwh
                )
            else:
                projected_soc = (
                    soc
                    - abs(safe_bess_kw)
                    * self.step_hours
                    / (0.96 * self.bess_capacity_kwh)
                )
            projected["projected_soc"] = round(
                float(np.clip(projected_soc, self.soc_min, self.soc_max)), 8
            )
            correction = abs(
                safe_bess_kw - float(projected["requested_bess_kw"])
            )
            projected["projection_applied"] = True
            projected["projection_correction_kw"] = round(correction, 6)
            projected["projection_severity"] = round(
                correction / max(1.0, self.bess_power_kw), 8
            )
        projected.update(
            berth_priority=round(float(control[3]), 6),
            yard_flow_command=round(float(control[4]), 6),
            inspection_buffer=round(float(control[5]), 6),
            recovery_priority=round(float(control[6]), 6),
            regulatory_authority="recommendation_only_no_release_authority",
            safety_revision=self.SAFETY_REVISION,
        )
        return projected

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._maritime_hold_vessels = 0.0
        self._customs_hold_vessels = 0.0
        self._maritime_hold_teu = 0.0
        self._customs_hold_teu = 0.0
        self._recovery_queue = 0.0
        _observation, info = super().reset(seed=seed, options=options)
        self._totals.update(
            regulatory_delay=0.0,
            regulatory_delay_teu_hours=0.0,
            regulatory_vessel_hold_hours=0.0,
            regulatory_inspected_work_teu=0.0,
            regulatory_released_work_teu=0.0,
            regulatory_recovered_work_teu=0.0,
            maritime_inspection_arrivals=0.0,
            customs_inspection_arrivals=0.0,
            maritime_release_vessels=0.0,
            customs_release_vessels=0.0,
            maritime_hold_vessels_peak=0.0,
            customs_hold_vessels_peak=0.0,
            regulatory_hold_teu_peak=0.0,
            recovery_queue_teu_peak=0.0,
            inspection_buffer_action_sum=0.0,
            recovery_priority_action_sum=0.0,
            inspection_buffer_active_steps=0.0,
            recovery_priority_active_steps=0.0,
            regulatory_readiness_energy_kwh=0.0,
            regulatory_recovery_energy_kwh=0.0,
        )
        return self._observation(), info

    def step(self, action: Any):
        row = self._row()
        control = self._decode_action(action)
        flex_command = float(control[2])
        berth_priority = float(control[3])
        yard_flow = float(control[4])
        inspection_buffer = float(control[5])
        recovery_priority = float(control[6])
        flex_kw = flex_command * min(250.0, 0.08 * max(row[0], 1.0))
        service_delta = float(control[1]) - 1.0
        service_load_kw = (
            float(row[0]) * self.operational_load_fraction * service_delta
        )
        berth_ratio = (
            berth_priority / self.berth_priority_limit
            if self.berth_priority_limit
            else 0.0
        )
        yard_ratio = yard_flow / self.yard_flow_limit if self.yard_flow_limit else 0.0
        allocation_load_kw = (
            float(row[0])
            * self.allocation_load_fraction
            * 0.5
            * (berth_ratio + yard_ratio)
        )
        positive_buffer = max(0.0, inspection_buffer)
        positive_recovery = max(0.0, recovery_priority)
        readiness_load_kw = (
            float(row[0])
            * self.inspection_readiness_load_fraction
            * positive_buffer
        )
        recovery_load_kw = (
            float(row[0]) * self.recovery_load_fraction * positive_recovery
        )
        projected = self.project_control(
            action,
            soc=self._soc,
            last_bess_kw=self._last_bess_kw,
            initial_soc=self._initial_soc,
            remaining_steps=self.episode_steps - self._local_step - 1,
            max_grid_charge_kw=(
                self.demand_cap_kw
                - float(row[0])
                - flex_kw
                - service_load_kw
                - allocation_load_kw
                - readiness_load_kw
                - recovery_load_kw
            ),
        )
        bess_kw = float(projected["bess_kw"])
        service_factor = float(projected["service_factor"])
        self._soc = float(projected["projected_soc"])
        factor_index = self._start + self._local_step
        factor_values = self.segment_factor_values[factor_index]
        factor_mask = self.segment_factor_availability[factor_index]
        factors = {
            name: float(factor_values[index]) if factor_mask[index] > 0.5 else None
            for index, name in enumerate(FACTOR_COLUMNS)
        }
        regulatory_values = self.segment_regulatory_values[factor_index]
        regulatory_mask = self.segment_regulatory_availability[factor_index]
        regulatory = {
            name: (
                float(regulatory_values[index])
                if regulatory_mask[index] > 0.5
                else None
            )
            for index, name in enumerate(REGULATORY_COLUMNS)
        }
        availability = [
            factors[name]
            for name in (
                "crane_availability_ratio",
                "equipment_availability_ratio",
                "pilot_tug_availability_ratio",
            )
            if factors[name] is not None
        ]
        resource_factor = float(np.prod(availability)) if availability else 1.0
        congestion = factors["channel_congestion_ratio"]
        if congestion is not None:
            resource_factor *= max(0.2, 1.0 - 0.35 * congestion)
        if factors["closure_flag"] is not None and factors["closure_flag"] >= 0.5:
            resource_factor = 0.0
        weather_limits = self.port_profile["weather_limits"]
        weather_blocked = False
        for factor_name, limit_name, comparison in (
            ("wind_speed_mps", "wind_stop_mps", "high"),
            ("visibility_km", "visibility_stop_km", "low"),
            ("wave_height_m", "wave_stop_m", "high"),
        ):
            value = factors[factor_name]
            limit = weather_limits.get(limit_name)
            if value is None or limit is None:
                continue
            weather_blocked = weather_blocked or (
                value >= float(limit)
                if comparison == "high"
                else value <= float(limit)
            )
        if weather_blocked:
            resource_factor = 0.0

        arrivals = max(0.0, float(row[2]))
        incoming_work = max(0.0, float(row[1]) + 2.0 * arrivals)
        maritime_ratio = float(regulatory["maritime_inspection_ratio"] or 0.0)
        customs_ratio = float(regulatory["customs_inspection_ratio"] or 0.0)
        combined_ratio = min(
            0.95,
            max(0.0, 1.0 - (1.0 - maritime_ratio) * (1.0 - customs_ratio)),
        )
        inspected_work = incoming_work * combined_ratio
        normal_work = incoming_work - inspected_work
        ratio_total = maritime_ratio + customs_ratio
        maritime_share = maritime_ratio / ratio_total if ratio_total > 0 else 0.0
        customs_share = customs_ratio / ratio_total if ratio_total > 0 else 0.0
        maritime_arrivals = arrivals * maritime_ratio
        customs_arrivals = arrivals * customs_ratio
        self._maritime_hold_vessels += maritime_arrivals
        self._customs_hold_vessels += customs_arrivals
        self._maritime_hold_teu += inspected_work * maritime_share
        self._customs_hold_teu += inspected_work * customs_share

        inspection_availability = float(
            regulatory["inspection_resource_availability_ratio"] or 0.0
        )
        release_ratio = float(regulatory["regulatory_release_ratio"] or 0.0)
        buffer_multiplier = max(
            0.35,
            1.0 + self.inspection_buffer_capacity_gain * inspection_buffer,
        )
        maritime_capacity = (
            self.maritime_inspection_capacity
            * self.step_hours
            * inspection_availability
            * buffer_multiplier
        )
        customs_capacity = (
            self.customs_inspection_capacity
            * self.step_hours
            * inspection_availability
            * buffer_multiplier
        )
        maritime_before = self._maritime_hold_vessels
        customs_before = self._customs_hold_vessels
        maritime_release_effect = release_ratio * max(
            0.02, 1.0 - float(regulatory["maritime_detention_ratio"] or 0.0)
        )
        customs_release_effect = release_ratio * max(
            0.02,
            1.0 - float(regulatory["customs_secondary_check_ratio"] or 0.0),
        )
        maritime_released_vessels = (
            min(maritime_before, maritime_capacity) * maritime_release_effect
        )
        customs_released_vessels = (
            min(customs_before, customs_capacity) * customs_release_effect
        )
        maritime_release_fraction = (
            min(1.0, maritime_released_vessels / maritime_before)
            if maritime_before > 0
            else 0.0
        )
        customs_release_fraction = (
            min(1.0, customs_released_vessels / customs_before)
            if customs_before > 0
            else 0.0
        )
        maritime_released_work = self._maritime_hold_teu * maritime_release_fraction
        customs_released_work = self._customs_hold_teu * customs_release_fraction
        self._maritime_hold_vessels = max(
            0.0, self._maritime_hold_vessels - maritime_released_vessels
        )
        self._customs_hold_vessels = max(
            0.0, self._customs_hold_vessels - customs_released_vessels
        )
        self._maritime_hold_teu = max(
            0.0, self._maritime_hold_teu - maritime_released_work
        )
        self._customs_hold_teu = max(
            0.0, self._customs_hold_teu - customs_released_work
        )
        released_work = maritime_released_work + customs_released_work
        self._recovery_queue += released_work
        self._queue += normal_work

        allocation_factor = max(
            0.55,
            1.0
            + 0.08 * berth_priority
            + 0.08 * yard_flow
            + self.recovery_service_capacity_gain * positive_recovery,
        )
        service_reserve = (
            self.inspection_buffer_service_reserve_fraction * positive_buffer
        )
        service_capacity = (
            max(1.0, float(row[1]))
            * service_factor
            * resource_factor
            * allocation_factor
            * max(0.75, 1.0 - service_reserve)
        )
        total_eligible = self._queue + self._recovery_queue
        natural_recovery_share = (
            self._recovery_queue / total_eligible if total_eligible > 0 else 0.0
        )
        recovery_share = float(
            np.clip(natural_recovery_share + 0.25 * recovery_priority, 0.0, 1.0)
        )
        recovery_served = min(
            self._recovery_queue, service_capacity * recovery_share
        )
        remaining_capacity = max(0.0, service_capacity - recovery_served)
        normal_served = min(self._queue, remaining_capacity)
        remaining_capacity -= normal_served
        if remaining_capacity > 0:
            extra_recovery = min(
                self._recovery_queue - recovery_served, remaining_capacity
            )
            recovery_served += extra_recovery
        self._recovery_queue = max(0.0, self._recovery_queue - recovery_served)
        self._queue = max(0.0, self._queue - normal_served)
        served = normal_served + recovery_served
        regulatory_hold_teu = self._maritime_hold_teu + self._customs_hold_teu

        net_kw = max(
            0.0,
            float(row[0])
            + service_load_kw
            + allocation_load_kw
            + readiness_load_kw
            + recovery_load_kw
            + bess_kw
            + flex_kw,
        )
        price = max(0.0, float(row[4]))
        carbon_factor = max(0.0, float(row[5]))
        energy_cost = net_kw * self.step_hours * price
        carbon_kg = net_kw * self.step_hours * carbon_factor
        exceed_kw = max(0.0, net_kw - self.demand_cap_kw - 0.001)
        unsafe_soc = max(0.0, self.soc_min - self._soc) + max(
            0.0, self._soc - self.soc_max
        )
        ramp_violation = max(
            0.0,
            abs(bess_kw - self._last_bess_kw) - self.bess_power_kw - 0.001,
        )
        terminal_soc_error = (
            abs(self._soc - self._initial_soc)
            if self._local_step + 1 >= self.episode_steps
            else 0.0
        )
        safety_penalty = 50.0 * unsafe_soc + ramp_violation / max(
            1.0, self.bess_power_kw
        )
        safety_penalty += 50.0 * terminal_soc_error
        if weather_blocked and service_factor > self.service_min + 1e-6:
            safety_penalty += 1.0
        yard_occupancy = factors["yard_occupancy_ratio"]
        # CSV 0.920000 is represented as 0.9200000167 in float32. Apply an
        # explicit ratio tolerance so the boundary itself is not classified as
        # an exceedance; materially higher occupancy remains penalized.
        yard_excess = (
            max(0.0, float(yard_occupancy) - 0.92 - 1e-6)
            if yard_occupancy is not None
            else 0.0
        )
        if (
            yard_excess > 0.0
            and recovery_priority > 0.5 * self.recovery_priority_limit
        ):
            safety_penalty += yard_excess * 10.0
        total_delay_work = (
            self._queue + self._recovery_queue + regulatory_hold_teu
        )
        delay_index = total_delay_work / max(1.0, incoming_work)
        regulatory_delay_index = (
            regulatory_hold_teu + self._recovery_queue
        ) / max(1.0, incoming_work)
        degradation = abs(bess_kw) * self.step_hours * 0.02
        components = {
            "cost": energy_cost / max(1.0, self.demand_cap_kw),
            "carbon": carbon_kg / max(1.0, self.demand_cap_kw),
            "peak": exceed_kw / self.demand_cap_kw,
            "safety": safety_penalty,
            "delay": delay_index,
            "regulatory_delay": regulatory_delay_index,
            "projection": float(projected["projection_severity"]),
        }
        weighted_business = sum(
            self.weights[name] * components[name] for name in self.weights
        )
        reward = -weighted_business - degradation / 1000.0
        reward -= self.projection_penalty_weight * components["projection"]
        reward -= (
            self.regulatory_delay_penalty_weight * components["regulatory_delay"]
        )
        violation = bool(exceed_kw > 0 or safety_penalty > 0)

        self._totals["reward"] += reward
        self._totals["energy_cost"] += energy_cost
        self._totals["carbon_kg"] += carbon_kg
        self._totals["throughput_teu"] += served
        self._totals["delay"] += delay_index
        self._totals["violations"] += float(violation)
        self._totals["peak_kw"] = max(self._totals["peak_kw"], net_kw)
        self._totals["grid_energy_kwh"] += net_kw * self.step_hours
        self._totals["bess_throughput_kwh"] += abs(bess_kw) * self.step_hours
        self._totals["flex_shift_energy_kwh"] += abs(flex_kw) * self.step_hours
        self._totals["work_demand_teu"] += incoming_work
        self._totals["queue_peak_teu"] = max(
            self._totals["queue_peak_teu"], total_delay_work
        )
        self._totals["queue_end_teu"] = total_delay_work
        self._totals["weather_blocked_steps"] += float(weather_blocked)
        self._totals["projection_count"] += float(
            projected["projection_applied"]
        )
        self._totals["projection_correction_kw"] += float(
            projected["projection_correction_kw"]
        )
        self._totals["projection_severity"] += float(
            projected["projection_severity"]
        )
        reasons = set(projected["projection_reasons"])
        self._totals["projection_grid_cap_count"] += float(
            "grid_charge_cap" in reasons
        )
        self._totals["projection_soc_bound_count"] += float(
            bool({"soc_upper_bound", "soc_lower_bound"} & reasons)
        )
        self._totals["projection_terminal_reachability_count"] += float(
            "terminal_soc_reachability" in reasons
        )
        self._totals["projection_power_bound_count"] += float(
            "power_or_ramp_bound" in reasons
        )
        self._totals["resource_factor_sum"] += resource_factor
        self._totals["service_factor_sum"] += service_factor
        self._totals["terminal_soc_error"] += terminal_soc_error
        self._totals["regulatory_delay"] += regulatory_delay_index
        self._totals["regulatory_delay_teu_hours"] += (
            regulatory_hold_teu + self._recovery_queue
        ) * self.step_hours
        self._totals["regulatory_vessel_hold_hours"] += (
            self._maritime_hold_vessels + self._customs_hold_vessels
        ) * self.step_hours
        self._totals["regulatory_inspected_work_teu"] += inspected_work
        self._totals["regulatory_released_work_teu"] += released_work
        self._totals["regulatory_recovered_work_teu"] += recovery_served
        self._totals["maritime_inspection_arrivals"] += maritime_arrivals
        self._totals["customs_inspection_arrivals"] += customs_arrivals
        self._totals["maritime_release_vessels"] += maritime_released_vessels
        self._totals["customs_release_vessels"] += customs_released_vessels
        self._totals["maritime_hold_vessels_peak"] = max(
            self._totals["maritime_hold_vessels_peak"],
            self._maritime_hold_vessels,
        )
        self._totals["customs_hold_vessels_peak"] = max(
            self._totals["customs_hold_vessels_peak"],
            self._customs_hold_vessels,
        )
        self._totals["regulatory_hold_teu_peak"] = max(
            self._totals["regulatory_hold_teu_peak"], regulatory_hold_teu
        )
        self._totals["recovery_queue_teu_peak"] = max(
            self._totals["recovery_queue_teu_peak"], self._recovery_queue
        )
        self._totals["inspection_buffer_action_sum"] += inspection_buffer
        self._totals["recovery_priority_action_sum"] += recovery_priority
        self._totals["inspection_buffer_active_steps"] += float(
            inspection_buffer > 1e-6
        )
        self._totals["recovery_priority_active_steps"] += float(
            recovery_priority > 1e-6
        )
        self._totals["regulatory_readiness_energy_kwh"] += (
            readiness_load_kw * self.step_hours
        )
        self._totals["regulatory_recovery_energy_kwh"] += (
            recovery_load_kw * self.step_hours
        )

        timestamp = self.segment_timestamps[factor_index]
        info = {
            "timestamp": timestamp,
            "baseline_kw": float(row[0]),
            "net_load_kw": net_kw,
            "service_load_delta_kw": service_load_kw,
            "allocation_load_delta_kw": allocation_load_kw,
            "regulatory_readiness_load_kw": readiness_load_kw,
            "regulatory_recovery_load_kw": recovery_load_kw,
            "bess_kw": bess_kw,
            "soc": self._soc,
            "queue": self._queue,
            "recovery_queue_teu": self._recovery_queue,
            "regulatory_hold_teu": regulatory_hold_teu,
            "maritime_hold_vessels": self._maritime_hold_vessels,
            "customs_hold_vessels": self._customs_hold_vessels,
            "released_work_teu": released_work,
            "recovered_work_teu": recovery_served,
            "served_teu": served,
            "energy_cost": energy_cost,
            "carbon_kg": carbon_kg,
            "delay_index": delay_index,
            "regulatory_delay_index": regulatory_delay_index,
            "guardrail_violation": violation,
            "terminal_soc_error": terminal_soc_error,
            "action_projection_applied": bool(
                projected["projection_applied"]
            ),
            "action_projection_correction_kw": float(
                projected["projection_correction_kw"]
            ),
            "action_projection_severity": float(
                projected["projection_severity"]
            ),
            "action_projection_reasons": list(
                projected["projection_reasons"]
            ),
            "environment_version": self.environment_version,
            "berth_priority": berth_priority,
            "yard_flow_command": yard_flow,
            "inspection_buffer": inspection_buffer,
            "recovery_priority": recovery_priority,
            "regulatory_authority": "recommendation_only_no_release_authority",
            "safety_revision": self.SAFETY_REVISION,
            "operational_resource_factor": resource_factor,
            "weather_blocked": weather_blocked,
            "factor_availability": {
                name: bool(factor_mask[index] > 0.5)
                for index, name in enumerate(FACTOR_COLUMNS)
            },
            "regulatory_factor_availability": {
                name: bool(regulatory_mask[index] > 0.5)
                for index, name in enumerate(REGULATORY_COLUMNS)
            },
            "reward_components": components,
            "safety_penalty": safety_penalty,
        }
        if self.record_trace:
            self.trace.append(
                dict(info, reward=reward, action=[float(value) for value in control])
            )
        self._last_bess_kw = bess_kw
        self._local_step += 1
        terminated = self._local_step >= self.episode_steps
        truncated = False
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info["episode_metrics"] = dict(self.totals)
        else:
            observation = self._observation()
        return observation, float(reward), terminated, truncated, info

    @property
    def totals(self) -> Dict[str, float]:
        row = super().totals
        steps = max(1, self._local_step)
        row.update(
            regulatory_delay_index_mean=row["regulatory_delay"] / steps,
            regulatory_clearance_completion_ratio=(
                row["regulatory_released_work_teu"]
                / max(1.0, row["regulatory_inspected_work_teu"])
            ),
            regulatory_recovery_completion_ratio=(
                row["regulatory_recovered_work_teu"]
                / max(1.0, row["regulatory_released_work_teu"])
            ),
            inspection_buffer_action_mean=(
                row["inspection_buffer_action_sum"] / steps
            ),
            recovery_priority_action_mean=(
                row["recovery_priority_action_sum"] / steps
            ),
            inspection_buffer_active_rate=(
                row["inspection_buffer_active_steps"] / steps
            ),
            recovery_priority_active_rate=(
                row["recovery_priority_active_steps"] / steps
            ),
            regulatory_hold_work_end_teu=(
                self._maritime_hold_teu + self._customs_hold_teu
            ),
            recovery_queue_end_teu=self._recovery_queue,
        )
        return row
