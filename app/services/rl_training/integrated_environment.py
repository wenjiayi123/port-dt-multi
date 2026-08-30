from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from .datasets import FACTOR_COLUMNS, PORT_WIDE_COLUMNS, PortDataset
from .regulatory_environment import RegulatoryPortOperationsEnv


DEFAULT_INTEGRATED_REWARD_WEIGHTS = {
    "gate_congestion": 0.24,
    "intermodal_backlog": 0.18,
    "reefer_risk": 0.26,
    "shore_power_unserved": 0.14,
    "maintenance_risk": 0.22,
    "marine_services_delay": 0.22,
    "dangerous_goods_safety": 0.40,
    "under_keel_clearance": 0.55,
    "forecast_uncertainty": 0.08,
    "integrated_peak": 0.18,
}


class IntegratedPortOperationsEnv(RegulatoryPortOperationsEnv):
    """V5 integrated port-business environment.

    The learner recommends bounded resource allocations. Regulatory release,
    channel closure, under-keel-clearance acceptance, dangerous-goods rules and
    production actuation remain deterministic external constraints.
    """

    INTERNAL_STATE_DIMENSIONS = 6
    OBSERVATION_DIMENSIONS = (
        RegulatoryPortOperationsEnv.OBSERVATION_DIMENSIONS
        + 2 * len(PORT_WIDE_COLUMNS)
        + INTERNAL_STATE_DIMENSIONS
    )
    ACTION_DIMENSIONS = 13
    SAFETY_REVISION = "v5_integrated_advisory_plus_deterministic_hard_constraints_v2"

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
        environment_version: str = "port_ops_v5",
        port_profile: Optional[Dict[str, Any]] = None,
        projection_penalty_weight: float = 0.0,
        regulatory_delay_penalty_weight: float = 0.35,
        integrated_reward_weights: Optional[Dict[str, float]] = None,
        normalization_slice: Optional[slice] = None,
        normalization_dataset: Optional[PortDataset] = None,
        training: bool = True,
        record_trace: bool = False,
    ) -> None:
        if environment_version != "port_ops_v5":
            raise ValueError("IntegratedPortOperationsEnv requires port_ops_v5")
        if action_mode != "continuous":
            raise ValueError(
                "port_ops_v5 is continuous-only; a discrete 13-control lattice is not operationally auditable"
            )
        super().__init__(
            dataset,
            data_slice,
            action_mode="continuous",
            episode_steps=episode_steps,
            seed=seed,
            demand_cap_kw=demand_cap_kw,
            reward_weights=reward_weights,
            environment_version="port_ops_v4",
            port_profile=port_profile,
            projection_penalty_weight=projection_penalty_weight,
            regulatory_delay_penalty_weight=regulatory_delay_penalty_weight,
            normalization_slice=normalization_slice,
            normalization_dataset=normalization_dataset,
            training=training,
            record_trace=record_trace,
        )
        self.environment_version = "port_ops_v5"
        self.segment_port_wide_values = dataset.port_wide_values[data_slice].astype(
            np.float32, copy=True
        )
        self.segment_port_wide_availability = dataset.port_wide_availability[
            data_slice
        ].astype(np.float32, copy=True)
        normalization_source = normalization_dataset or dataset
        normalization_train = normalization_slice or normalization_source.split()[0]
        reference = normalization_source.port_wide_values[normalization_train].astype(
            np.float32, copy=False
        )
        reference_mask = normalization_source.port_wide_availability[normalization_train].astype(
            np.float32, copy=False
        )
        self._port_wide_mins = np.zeros(len(PORT_WIDE_COLUMNS), dtype=np.float32)
        self._port_wide_spans = np.ones(len(PORT_WIDE_COLUMNS), dtype=np.float32)
        for index in range(len(PORT_WIDE_COLUMNS)):
            available = reference_mask[:, index] > 0.5
            if np.any(available):
                observed = reference[available, index]
                self._port_wide_mins[index] = float(np.min(observed))
                self._port_wide_spans[index] = max(
                    float(np.max(observed) - np.min(observed)), 1e-6
                )
        limits = self.port_profile["control_limits"]
        self._integrated_action_limits = np.asarray(
            [
                limits["gate_smoothing_limit"],
                limits["intermodal_allocation_limit"],
                limits["reefer_service_limit"],
                limits["shore_power_allocation_limit"],
                limits["maintenance_reserve_limit"],
                limits["marine_service_allocation_limit"],
            ],
            dtype=np.float32,
        )
        operations = self.port_profile["integrated_operations"]
        self.gate_service_capacity = float(
            operations["gate_service_trucks_per_hour"]
        )
        self.intermodal_service_capacity = float(
            operations["rail_barge_service_teu_per_hour"]
        )
        self.marine_service_capacity = float(
            operations["marine_service_vessels_per_hour"]
        )
        self.reefer_risk_recovery_rate = float(
            operations["reefer_risk_recovery_rate"]
        )
        self.maintenance_recovery_rate = float(
            operations["maintenance_recovery_rate"]
        )
        self.shore_power_efficiency = float(
            operations["shore_power_service_efficiency"]
        )
        self.shore_power_avoided_carbon_factor = float(
            operations["shore_power_avoided_carbon_kg_per_kwh"]
        )
        self.required_under_keel_clearance_m = float(
            operations["required_under_keel_clearance_m"]
        )
        self.dangerous_goods_yard_limit = float(
            operations["dangerous_goods_yard_occupancy_limit"]
        )
        self.high_reefer_risk = float(operations["high_reefer_risk_ratio"])
        self.high_failure_risk = float(
            operations["high_equipment_failure_risk_ratio"]
        )
        self.uncertainty_review_ratio = float(
            operations["forecast_uncertainty_review_ratio"]
        )
        weights = dict(DEFAULT_INTEGRATED_REWARD_WEIGHTS)
        for name, value in (integrated_reward_weights or {}).items():
            if name in weights:
                weights[name] = max(0.0, float(value))
        self.integrated_reward_weights = weights
        self.observation_space = spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(self.OBSERVATION_DIMENSIONS,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.ACTION_DIMENSIONS,),
            dtype=np.float32,
        )
        self._gate_backlog = 0.0
        self._intermodal_backlog = 0.0
        self._marine_service_backlog = 0.0
        self._reefer_risk_stock = 0.0
        self._maintenance_debt = 0.0
        self._unmet_shore_power_kw = 0.0

    def _port_wide_observation(
        self, values: np.ndarray, mask: np.ndarray
    ) -> list[float]:
        normalized = 2.0 * (
            values - self._port_wide_mins
        ) / self._port_wide_spans - 1.0
        normalized = np.where(mask > 0.5, normalized, 0.0)
        return [
            *normalized.astype(np.float32).tolist(),
            *mask.astype(np.float32).tolist(),
        ]

    def _integrated_state_observation(self) -> list[float]:
        return [
            float(np.clip(self._gate_backlog / max(1.0, self.gate_service_capacity * 4.0), 0.0, 1.5)),
            float(np.clip(self._intermodal_backlog / max(1.0, self.intermodal_service_capacity * 4.0), 0.0, 1.5)),
            float(np.clip(self._marine_service_backlog / max(1.0, self.marine_service_capacity * 4.0), 0.0, 1.5)),
            float(np.clip(self._reefer_risk_stock, 0.0, 1.5)),
            float(np.clip(self._maintenance_debt, 0.0, 1.5)),
            float(np.clip(self._unmet_shore_power_kw / max(1.0, self.demand_cap_kw), 0.0, 1.5)),
        ]

    def _observation(self) -> np.ndarray:
        base = super()._observation()
        index = self._start + self._local_step
        return np.concatenate(
            [
                base,
                np.asarray(
                    self._port_wide_observation(
                        self.segment_port_wide_values[index],
                        self.segment_port_wide_availability[index],
                    ),
                    dtype=np.float32,
                ),
                np.asarray(self._integrated_state_observation(), dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)

    def observation_from_state(self, state: Dict[str, Any]) -> np.ndarray:
        base = super().observation_from_state(state)
        values = np.zeros(len(PORT_WIDE_COLUMNS), dtype=np.float32)
        mask = np.zeros(len(PORT_WIDE_COLUMNS), dtype=np.float32)
        for index, column in enumerate(PORT_WIDE_COLUMNS):
            if state.get(column) is not None:
                values[index] = float(state[column])
                mask[index] = 1.0
        internal = np.asarray(
            [
                np.clip(float(state.get("gate_backlog_trucks", 0.0)) / max(1.0, self.gate_service_capacity * 4.0), 0.0, 1.5),
                np.clip(float(state.get("intermodal_backlog_teu", 0.0)) / max(1.0, self.intermodal_service_capacity * 4.0), 0.0, 1.5),
                np.clip(float(state.get("marine_service_backlog_vessels", 0.0)) / max(1.0, self.marine_service_capacity * 4.0), 0.0, 1.5),
                np.clip(float(state.get("reefer_risk_stock", 0.0)), 0.0, 1.5),
                np.clip(float(state.get("maintenance_debt", 0.0)), 0.0, 1.5),
                np.clip(float(state.get("unmet_shore_power_kw", 0.0)) / max(1.0, self.demand_cap_kw), 0.0, 1.5),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                base,
                np.asarray(self._port_wide_observation(values, mask), dtype=np.float32),
                internal,
            ]
        ).astype(np.float32, copy=False)

    def _decode_action(self, action: Any) -> np.ndarray:
        """Decode the seven-action V4 prefix for inherited dynamics."""
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.size != RegulatoryPortOperationsEnv.ACTION_DIMENSIONS:
            raise ValueError("the inherited port_ops_v4 action prefix must contain 7 values")
        continuous = np.clip(raw, -1.0, 1.0).astype(np.float32, copy=True)
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

    def _decode_integrated_action(self, action: Any) -> np.ndarray:
        # Use the V5 contract width explicitly so additive subclasses can map
        # their larger action vector into this compatibility prefix.
        raw = np.asarray(action, dtype=np.float32).reshape(
            IntegratedPortOperationsEnv.ACTION_DIMENSIONS
        )
        raw = np.clip(raw, -1.0, 1.0)
        base = self._decode_action(raw[:7])
        # Zero is the declared current-operations midpoint. Negative values
        # reserve less capacity and positive values reserve more.
        allocations = 0.5 * (raw[7:] + 1.0) * self._integrated_action_limits
        return np.concatenate([base, allocations]).astype(np.float32, copy=False)

    def describe_action(self, action: Any) -> Dict[str, float]:
        control = self._decode_integrated_action(action)
        return {
            "bess_kw": round(float(control[0]) * self.bess_power_kw, 6),
            "service_factor": round(float(control[1]), 6),
            "flexible_load_command": round(float(control[2]), 6),
            "berth_priority": round(float(control[3]), 6),
            "yard_flow_command": round(float(control[4]), 6),
            "inspection_buffer": round(float(control[5]), 6),
            "recovery_priority": round(float(control[6]), 6),
            "gate_smoothing_ratio": round(float(control[7]), 6),
            "intermodal_allocation_ratio": round(float(control[8]), 6),
            "reefer_service_ratio": round(float(control[9]), 6),
            "shore_power_allocation_ratio": round(float(control[10]), 6),
            "maintenance_reserve_ratio": round(float(control[11]), 6),
            "marine_service_allocation_ratio": round(float(control[12]), 6),
        }

    def project_control(self, action: Any, **kwargs: Any) -> Dict[str, Any]:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.size not in {7, IntegratedPortOperationsEnv.ACTION_DIMENSIONS}:
            raise ValueError("port_ops_v5 control must contain 7 base or 13 integrated actions")
        base_action = raw[:7]
        projected = super().project_control(base_action, **kwargs)
        if raw.size == IntegratedPortOperationsEnv.ACTION_DIMENSIONS:
            integrated = self._decode_integrated_action(raw)
            projected.update(
                gate_smoothing_ratio=round(float(integrated[7]), 6),
                intermodal_allocation_ratio=round(float(integrated[8]), 6),
                reefer_service_ratio=round(float(integrated[9]), 6),
                shore_power_allocation_ratio=round(float(integrated[10]), 6),
                maintenance_reserve_ratio=round(float(integrated[11]), 6),
                marine_service_allocation_ratio=round(float(integrated[12]), 6),
            )
        projected.update(
            integrated_authority="recommendation_only_no_release_navigation_or_actuator_authority",
            safety_revision=self.SAFETY_REVISION,
        )
        return projected

    def _port_values(self, index: int) -> tuple[Dict[str, Optional[float]], Dict[str, bool]]:
        values = self.segment_port_wide_values[index]
        mask = self.segment_port_wide_availability[index]
        available = {
            name: bool(mask[position] > 0.5)
            for position, name in enumerate(PORT_WIDE_COLUMNS)
        }
        return (
            {
                name: float(values[position]) if available[name] else None
                for position, name in enumerate(PORT_WIDE_COLUMNS)
            },
            available,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._gate_backlog = 0.0
        self._intermodal_backlog = 0.0
        self._marine_service_backlog = 0.0
        self._reefer_risk_stock = 0.0
        self._maintenance_debt = 0.0
        self._unmet_shore_power_kw = 0.0
        _observation, info = super().reset(seed=seed, options=options)
        port, _available = self._port_values(self._start)
        self._gate_backlog = max(0.0, float(port["gate_queue_trucks"] or 0.0))
        self._reefer_risk_stock = max(
            0.0, float(port["reefer_temperature_risk_ratio"] or 0.0)
        )
        self._maintenance_debt = max(
            0.0, float(port["maintenance_backlog_ratio"] or 0.0)
        )
        self._totals.update(
            gate_demand_trucks=0.0,
            gate_served_trucks=0.0,
            gate_queue_truck_hours=0.0,
            gate_queue_peak_trucks=self._gate_backlog,
            intermodal_demand_teu=0.0,
            intermodal_served_teu=0.0,
            intermodal_backlog_teu_hours=0.0,
            marine_service_demand_vessels=0.0,
            marine_service_served_vessels=0.0,
            marine_service_wait_vessel_hours=0.0,
            reefer_risk_hours=0.0,
            maintenance_risk_hours=0.0,
            shore_power_demand_kwh=0.0,
            shore_power_served_kwh=0.0,
            shore_power_unserved_kwh=0.0,
            shore_power_avoided_carbon_kg=0.0,
            dangerous_goods_risk_hours=0.0,
            under_keel_clearance_blocked_steps=0.0,
            hard_constraint_interventions=0.0,
            forecast_review_steps=0.0,
            integrated_reward=0.0,
        )
        return self._observation(), info

    @staticmethod
    def _value(port: Dict[str, Optional[float]], name: str, default: float = 0.0) -> float:
        value = port.get(name)
        return default if value is None else float(value)

    def step(self, action: Any):
        raw_action = np.asarray(action, dtype=np.float32).reshape(
            IntegratedPortOperationsEnv.ACTION_DIMENSIONS
        )
        data_index = self._start + self._local_step
        row = self._row().copy()
        port, availability = self._port_values(data_index)
        factor_values = self.segment_factor_values[data_index]
        factor_mask = self.segment_factor_availability[data_index]
        factors = {
            name: float(factor_values[index]) if factor_mask[index] > 0.5 else None
            for index, name in enumerate(FACTOR_COLUMNS)
        }
        requested_control = self._decode_integrated_action(raw_action)
        available_depth = (
            self._value(port, "channel_chart_depth_m")
            + float(row[3])
            - self._value(port, "planning_vessel_draft_m")
            - self._value(port, "squat_allowance_m")
        )
        under_keel_open = available_depth >= self.required_under_keel_clearance_m
        closure_open = not (
            factors.get("closure_flag") is not None
            and float(factors["closure_flag"]) >= 0.5
        )
        marine_window_open = under_keel_open and closure_open
        yard_occupancy_before = 0.0 if factors.get("yard_occupancy_ratio") is None else float(factors["yard_occupancy_ratio"])
        dangerous_goods_before = self._value(
            port, "dangerous_goods_workload_ratio"
        )
        safe_raw_action = np.clip(raw_action, -1.0, 1.0).copy()
        intervention_reasons: list[str] = []
        if (
            dangerous_goods_before > 0.0
            and yard_occupancy_before > self.dangerous_goods_yard_limit
            and safe_raw_action[4] > 0.0
        ):
            safe_raw_action[4] = 0.0
            intervention_reasons.append("dangerous_goods_positive_yard_inflow_blocked")
        if safe_raw_action[9] < 0.0:
            safe_raw_action[9] = 0.0
            intervention_reasons.append("reefer_minimum_baseline_service_reserve")
        if safe_raw_action[11] < 0.0:
            safe_raw_action[11] = 0.0
            intervention_reasons.append("maintenance_minimum_baseline_reserve")
        if not marine_window_open and safe_raw_action[12] > -1.0:
            safe_raw_action[12] = -1.0
            intervention_reasons.append("marine_window_closed")
        control = self._decode_integrated_action(safe_raw_action)

        _base_observation, base_reward, terminated, truncated, info = super().step(
            safe_raw_action[:7]
        )

        labor = self._value(port, "labor_availability_ratio", 1.0)
        gate_capacity_ratio = self._value(port, "gate_capacity_ratio", 1.0)
        truck_arrivals = self._value(port, "truck_arrivals_per_hour") * self.step_hours
        self._gate_backlog += truck_arrivals
        gate_capacity = (
            self.gate_service_capacity
            * self.step_hours
            * gate_capacity_ratio
            * labor
            * (0.5 + float(control[7]))
        )
        gate_served = min(self._gate_backlog, max(0.0, gate_capacity))
        self._gate_backlog = max(0.0, self._gate_backlog - gate_served)

        intermodal_demand = self._value(port, "rail_transfer_demand_teu") + self._value(
            port, "barge_transfer_demand_teu"
        )
        self._intermodal_backlog += intermodal_demand
        intermodal_capacity = (
            self.intermodal_service_capacity
            * self.step_hours
            * self._value(port, "intermodal_capacity_ratio", 1.0)
            * labor
            * (0.5 + float(control[8]))
        )
        intermodal_served = min(
            self._intermodal_backlog, max(0.0, intermodal_capacity)
        )
        self._intermodal_backlog = max(
            0.0, self._intermodal_backlog - intermodal_served
        )

        marine_demand = max(
            self._value(port, "pilotage_demand_vessels"),
            self._value(port, "tug_demand_vessels"),
        )
        self._marine_service_backlog += marine_demand
        pilot_tug = 1.0 if factors.get("pilot_tug_availability_ratio") is None else float(factors["pilot_tug_availability_ratio"])
        marine_capacity = (
            self.marine_service_capacity
            * self.step_hours
            * self._value(port, "channel_capacity_ratio", 1.0)
            * pilot_tug
            * labor
            * (0.5 + float(control[12]))
            if marine_window_open
            else 0.0
        )
        marine_served = min(
            self._marine_service_backlog, max(0.0, marine_capacity)
        )
        self._marine_service_backlog = max(
            0.0, self._marine_service_backlog - marine_served
        )

        reefer_occupancy = self._value(port, "reefer_occupancy_ratio")
        observed_reefer_risk = self._value(
            port, "reefer_temperature_risk_ratio"
        )
        self._reefer_risk_stock = float(
            np.clip(
                0.75 * self._reefer_risk_stock
                + observed_reefer_risk * (0.4 + 0.6 * reefer_occupancy)
                - self.reefer_risk_recovery_rate * float(control[9]),
                0.0,
                1.5,
            )
        )

        failure_risk = self._value(port, "equipment_failure_risk_ratio")
        observed_maintenance = self._value(port, "maintenance_backlog_ratio")
        self._maintenance_debt = float(
            np.clip(
                0.72 * self._maintenance_debt
                + 0.45 * failure_risk
                + 0.25 * observed_maintenance
                - self.maintenance_recovery_rate * float(control[11]) * labor,
                0.0,
                1.5,
            )
        )

        connected_shore_kw = (
            self._value(port, "shore_power_demand_kw")
            * self._value(port, "shore_power_connection_ratio")
        )
        shore_power_served_kw = (
            connected_shore_kw
            * self.shore_power_efficiency
            * (0.35 + 0.65 * float(control[10]))
        )
        self._unmet_shore_power_kw = max(
            0.0, connected_shore_kw - shore_power_served_kw
        )
        additional_energy_kwh = shore_power_served_kw * self.step_hours
        additional_cost = additional_energy_kwh * max(0.0, float(row[4]))
        additional_carbon = additional_energy_kwh * max(0.0, float(row[5]))
        avoided_carbon = (
            additional_energy_kwh * self.shore_power_avoided_carbon_factor
        )
        combined_net_kw = float(info["net_load_kw"]) + shore_power_served_kw
        integrated_exceed_kw = max(0.0, combined_net_kw - self.demand_cap_kw)

        yard_occupancy = 0.0 if factors.get("yard_occupancy_ratio") is None else float(factors["yard_occupancy_ratio"])
        dangerous_goods = self._value(port, "dangerous_goods_workload_ratio")
        dangerous_goods_risk = dangerous_goods * max(
            0.0, yard_occupancy - self.dangerous_goods_yard_limit
        ) * (1.0 + max(0.0, float(control[4])))
        uncertainty = self._value(port, "forecast_uncertainty_ratio")
        allocation_deviation = float(np.mean(np.abs(control[7:] - 0.5)))

        components = {
            "gate_congestion": self._gate_backlog / max(1.0, self.gate_service_capacity),
            "intermodal_backlog": self._intermodal_backlog / max(1.0, self.intermodal_service_capacity),
            "reefer_risk": self._reefer_risk_stock
            + 0.25 * max(0.0, float(control[9]) - float(requested_control[9])),
            "shore_power_unserved": self._unmet_shore_power_kw / max(1.0, connected_shore_kw),
            "maintenance_risk": self._maintenance_debt
            + 0.25 * max(0.0, float(control[11]) - float(requested_control[11])),
            "marine_services_delay": self._marine_service_backlog / max(1.0, self.marine_service_capacity),
            "dangerous_goods_safety": dangerous_goods_risk,
            "under_keel_clearance": (
                max(0.0, self.required_under_keel_clearance_m - available_depth)
                + (
                    max(0.0, float(requested_control[12]) - float(control[12]))
                    if not marine_window_open
                    else 0.0
                )
            ),
            "forecast_uncertainty": uncertainty * allocation_deviation,
            "integrated_peak": integrated_exceed_kw / max(1.0, self.demand_cap_kw),
        }
        integrated_penalty = sum(
            self.integrated_reward_weights[name] * components[name]
            for name in self.integrated_reward_weights
        )
        integrated_reward = -integrated_penalty
        reward = float(base_reward + integrated_reward)

        hard_intervention = float(bool(intervention_reasons))
        # One part per million of the configured demand cap is numerical
        # tolerance, not an operational breach. This suppresses sub-watt
        # float noise while retaining every material capacity exceedance.
        extra_violation = bool(
            integrated_exceed_kw > max(1e-6, self.demand_cap_kw * 1e-6)
        )
        if extra_violation and not bool(info["guardrail_violation"]):
            self._totals["violations"] += 1.0
        self._totals["reward"] += integrated_reward
        self._totals["integrated_reward"] += integrated_reward
        self._totals["energy_cost"] += additional_cost
        self._totals["carbon_kg"] += additional_carbon
        self._totals["grid_energy_kwh"] += additional_energy_kwh
        self._totals["peak_kw"] = max(self._totals["peak_kw"], combined_net_kw)
        self._totals["gate_demand_trucks"] += truck_arrivals
        self._totals["gate_served_trucks"] += gate_served
        self._totals["gate_queue_truck_hours"] += self._gate_backlog * self.step_hours
        self._totals["gate_queue_peak_trucks"] = max(
            self._totals["gate_queue_peak_trucks"], self._gate_backlog
        )
        self._totals["intermodal_demand_teu"] += intermodal_demand
        self._totals["intermodal_served_teu"] += intermodal_served
        self._totals["intermodal_backlog_teu_hours"] += self._intermodal_backlog * self.step_hours
        self._totals["marine_service_demand_vessels"] += marine_demand
        self._totals["marine_service_served_vessels"] += marine_served
        self._totals["marine_service_wait_vessel_hours"] += self._marine_service_backlog * self.step_hours
        self._totals["reefer_risk_hours"] += self._reefer_risk_stock * self.step_hours
        self._totals["maintenance_risk_hours"] += self._maintenance_debt * self.step_hours
        self._totals["shore_power_demand_kwh"] += connected_shore_kw * self.step_hours
        self._totals["shore_power_served_kwh"] += additional_energy_kwh
        self._totals["shore_power_unserved_kwh"] += self._unmet_shore_power_kw * self.step_hours
        self._totals["shore_power_avoided_carbon_kg"] += avoided_carbon
        self._totals["dangerous_goods_risk_hours"] += dangerous_goods_risk * self.step_hours
        self._totals["under_keel_clearance_blocked_steps"] += float(not under_keel_open)
        self._totals["hard_constraint_interventions"] += hard_intervention
        self._totals["forecast_review_steps"] += float(
            uncertainty >= self.uncertainty_review_ratio
        )

        info.update(
            net_load_kw=combined_net_kw,
            gate_queue_trucks=self._gate_backlog,
            gate_served_trucks=gate_served,
            intermodal_backlog_teu=self._intermodal_backlog,
            intermodal_served_teu=intermodal_served,
            marine_service_backlog_vessels=self._marine_service_backlog,
            marine_service_served_vessels=marine_served,
            reefer_risk_stock=self._reefer_risk_stock,
            maintenance_debt=self._maintenance_debt,
            shore_power_served_kw=shore_power_served_kw,
            shore_power_unserved_kw=self._unmet_shore_power_kw,
            shore_power_avoided_carbon_kg=avoided_carbon,
            under_keel_clearance_m=available_depth,
            marine_window_open=marine_window_open,
            dangerous_goods_risk=dangerous_goods_risk,
            forecast_uncertainty_ratio=uncertainty,
            hard_constraint_intervention=bool(hard_intervention),
            hard_constraint_intervention_reasons=intervention_reasons,
            requested_integrated_control={
                name: float(value)
                for name, value in zip(
                    (
                        "bess_action", "service_factor", "flexible_load",
                        "berth_priority", "yard_flow", "inspection_buffer",
                        "recovery_priority", "gate_smoothing_ratio",
                        "intermodal_allocation_ratio", "reefer_service_ratio",
                        "shore_power_allocation_ratio", "maintenance_reserve_ratio",
                        "marine_service_allocation_ratio",
                    ),
                    requested_control,
                )
            },
            guardrail_violation=bool(info["guardrail_violation"] or extra_violation),
            port_wide_factor_availability=availability,
            integrated_reward_components=components,
            integrated_reward=integrated_reward,
            integrated_authority="recommendation_only_no_release_navigation_or_actuator_authority",
        )
        if self.record_trace and self.trace:
            self.trace[-1].update(info)
            self.trace[-1]["reward"] = reward
            self.trace[-1]["action"] = [float(value) for value in control]
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info["episode_metrics"] = dict(self.totals)
        else:
            observation = self._observation()
        return observation, reward, terminated, truncated, info

    @property
    def totals(self) -> Dict[str, float]:
        row = super().totals
        steps = max(1, self._local_step)
        row.update(
            gate_service_completion_ratio=row["gate_served_trucks"] / max(1.0, row["gate_demand_trucks"]),
            intermodal_service_completion_ratio=row["intermodal_served_teu"] / max(1.0, row["intermodal_demand_teu"]),
            marine_service_completion_ratio=row["marine_service_served_vessels"] / max(1.0, row["marine_service_demand_vessels"]),
            shore_power_service_ratio=row["shore_power_served_kwh"] / max(1.0, row["shore_power_demand_kwh"]),
            reefer_risk_index_mean=row["reefer_risk_hours"] / max(self.step_hours, steps * self.step_hours),
            maintenance_risk_index_mean=row["maintenance_risk_hours"] / max(self.step_hours, steps * self.step_hours),
            dangerous_goods_risk_index_mean=row["dangerous_goods_risk_hours"] / max(self.step_hours, steps * self.step_hours),
            gate_queue_end_trucks=self._gate_backlog,
            intermodal_backlog_end_teu=self._intermodal_backlog,
            marine_service_backlog_end_vessels=self._marine_service_backlog,
            hard_constraint_intervention_rate=row["hard_constraint_interventions"] / steps,
            forecast_review_rate=row["forecast_review_steps"] / steps,
        )
        return row
