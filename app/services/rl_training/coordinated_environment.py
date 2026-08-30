from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from .datasets import FACTOR_COLUMNS, PortDataset
from .integrated_environment import IntegratedPortOperationsEnv


DEFAULT_COORDINATED_REWARD_WEIGHTS = {
    "rail_backlog": 0.16,
    "barge_backlog": 0.14,
    "pilotage_backlog": 0.20,
    "towage_backlog": 0.20,
    "terminal_move_chain": 0.34,
    "resource_imbalance": 0.08,
    "allocation_change": 0.04,
    "latent_action_correction": 0.42,
}


class CoordinatedPortOperationsEnv(IntegratedPortOperationsEnv):
    """V6 coordinated port-business environment.

    V5 proved the integrated business loop but deliberately grouped rail with
    barge and pilotage with towage. V6 separates those resource owners and adds
    a quay-crane -> horizontal-transport -> yard-crane move chain. The policy
    still recommends bounded resource ratios only; schedules, regulatory
    releases, navigation decisions and equipment actuation stay outside RL.

    Safety-critical action dimensions are parameterized inside their feasible
    set before the inherited deterministic safety projection runs. The latent
    learner request and every correction remain visible in the trace and KPI
    ledger, so feasibility-by-construction cannot hide a poorly learned policy.
    """

    SPLIT_INTERNAL_STATE_DIMENSIONS = 7
    OBSERVATION_DIMENSIONS = (
        IntegratedPortOperationsEnv.OBSERVATION_DIMENSIONS
        + SPLIT_INTERNAL_STATE_DIMENSIONS
    )
    ACTION_DIMENSIONS = 18
    SAFETY_REVISION = "v6_feasible_parameterization_split_resource_chain_v3"

    ACTION_NAMES = (
        "bess_action",
        "service_factor",
        "flexible_load",
        "berth_priority",
        "yard_flow",
        "inspection_buffer",
        "recovery_priority",
        "gate_smoothing_ratio",
        "rail_allocation_ratio",
        "barge_allocation_ratio",
        "reefer_service_ratio",
        "shore_power_allocation_ratio",
        "maintenance_reserve_ratio",
        "pilotage_allocation_ratio",
        "towage_allocation_ratio",
        "quay_crane_allocation_ratio",
        "horizontal_transport_allocation_ratio",
        "yard_crane_allocation_ratio",
    )

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
        environment_version: str = "port_ops_v6",
        port_profile: Optional[Dict[str, Any]] = None,
        projection_penalty_weight: float = 0.0,
        regulatory_delay_penalty_weight: float = 0.35,
        integrated_reward_weights: Optional[Dict[str, float]] = None,
        coordinated_reward_weights: Optional[Dict[str, float]] = None,
        normalization_slice: Optional[slice] = None,
        normalization_dataset: Optional[PortDataset] = None,
        training: bool = True,
        record_trace: bool = False,
    ) -> None:
        if environment_version != "port_ops_v6":
            raise ValueError("CoordinatedPortOperationsEnv requires port_ops_v6")
        if action_mode != "continuous":
            raise ValueError(
                "port_ops_v6 is continuous-only; an 18-control discrete lattice is not operationally auditable"
            )
        super().__init__(
            dataset,
            data_slice,
            action_mode="continuous",
            episode_steps=episode_steps,
            seed=seed,
            demand_cap_kw=demand_cap_kw,
            reward_weights=reward_weights,
            environment_version="port_ops_v5",
            port_profile=port_profile,
            projection_penalty_weight=projection_penalty_weight,
            regulatory_delay_penalty_weight=regulatory_delay_penalty_weight,
            integrated_reward_weights=integrated_reward_weights,
            normalization_slice=normalization_slice,
            normalization_dataset=normalization_dataset,
            training=training,
            record_trace=record_trace,
        )
        self.environment_version = "port_ops_v6"
        limits = self.port_profile["control_limits"]
        self._coordinated_action_limits = np.asarray(
            [
                limits["gate_smoothing_limit"],
                limits["rail_allocation_limit"],
                limits["barge_allocation_limit"],
                limits["reefer_service_limit"],
                limits["shore_power_allocation_limit"],
                limits["maintenance_reserve_limit"],
                limits["pilotage_allocation_limit"],
                limits["towage_allocation_limit"],
                limits["quay_crane_allocation_limit"],
                limits["horizontal_transport_allocation_limit"],
                limits["yard_crane_allocation_limit"],
            ],
            dtype=np.float32,
        )
        operations = self.port_profile["coordinated_operations"]
        self.rail_service_capacity = float(operations["rail_service_teu_per_hour"])
        self.barge_service_capacity = float(operations["barge_service_teu_per_hour"])
        self.pilotage_service_capacity = float(
            operations["pilotage_service_vessels_per_hour"]
        )
        self.towage_service_capacity = float(
            operations["towage_service_vessels_per_hour"]
        )
        self.quay_crane_capacity = float(operations["quay_crane_moves_per_hour"])
        self.horizontal_transport_capacity = float(
            operations["horizontal_transport_moves_per_hour"]
        )
        self.yard_crane_capacity = float(operations["yard_crane_moves_per_hour"])
        self.container_moves_per_teu = float(operations["container_moves_per_teu"])
        self.minimum_capacity_factors = {
            "gate": float(operations["gate_minimum_capacity_factor"]),
            "intermodal": float(operations["intermodal_minimum_capacity_factor"]),
            "marine": float(operations["marine_minimum_capacity_factor"]),
            "terminal_chain": float(
                operations["terminal_chain_minimum_capacity_factor"]
            ),
        }
        # Static service commitments are part of the action parameterization,
        # not post-hoc clipping. Each learner coordinate therefore retains a
        # smooth one-to-one mapping across the discretionary feasible range.
        # Pilotage/towage remain state-dependent because a closed channel must
        # map them to zero and an open channel applies the marine commitment.
        self._coordinated_action_minima = np.asarray(
            [
                self.minimum_capacity_factors["gate"],
                self.minimum_capacity_factors["intermodal"],
                self.minimum_capacity_factors["intermodal"],
                0.50,
                0.0,
                0.50,
                0.0,
                0.0,
                self.minimum_capacity_factors["terminal_chain"],
                self.minimum_capacity_factors["terminal_chain"],
                self.minimum_capacity_factors["terminal_chain"],
            ],
            dtype=np.float32,
        )
        weights = dict(DEFAULT_COORDINATED_REWARD_WEIGHTS)
        for name, value in (coordinated_reward_weights or {}).items():
            if name in weights:
                weights[name] = max(0.0, float(value))
        self.coordinated_reward_weights = weights
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
        self._reset_coordinated_state()

    def _reset_coordinated_state(self) -> None:
        self._rail_backlog = 0.0
        self._barge_backlog = 0.0
        self._pilotage_backlog = 0.0
        self._towage_backlog = 0.0
        self._quay_move_backlog = 0.0
        self._horizontal_move_backlog = 0.0
        self._yard_move_backlog = 0.0
        self._previous_allocations = np.full(11, 0.5, dtype=np.float32)

    def _coordinated_state_observation(self) -> list[float]:
        return [
            float(np.clip(self._rail_backlog / max(1.0, 4.0 * self.rail_service_capacity), 0.0, 1.5)),
            float(np.clip(self._barge_backlog / max(1.0, 4.0 * self.barge_service_capacity), 0.0, 1.5)),
            float(np.clip(self._pilotage_backlog / max(1.0, 4.0 * self.pilotage_service_capacity), 0.0, 1.5)),
            float(np.clip(self._towage_backlog / max(1.0, 4.0 * self.towage_service_capacity), 0.0, 1.5)),
            float(np.clip(self._quay_move_backlog / max(1.0, 4.0 * self.quay_crane_capacity), 0.0, 1.5)),
            float(np.clip(self._horizontal_move_backlog / max(1.0, 4.0 * self.horizontal_transport_capacity), 0.0, 1.5)),
            float(np.clip(self._yard_move_backlog / max(1.0, 4.0 * self.yard_crane_capacity), 0.0, 1.5)),
        ]

    def _observation(self) -> np.ndarray:
        base = super()._observation()
        return np.concatenate(
            [
                base,
                np.asarray(self._coordinated_state_observation(), dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)

    def observation_from_state(self, state: Dict[str, Any]) -> np.ndarray:
        base = super().observation_from_state(state)
        coordinated = np.asarray(
            [
                np.clip(float(state.get("rail_backlog_teu", 0.0)) / max(1.0, 4.0 * self.rail_service_capacity), 0.0, 1.5),
                np.clip(float(state.get("barge_backlog_teu", 0.0)) / max(1.0, 4.0 * self.barge_service_capacity), 0.0, 1.5),
                np.clip(float(state.get("pilotage_backlog_vessels", 0.0)) / max(1.0, 4.0 * self.pilotage_service_capacity), 0.0, 1.5),
                np.clip(float(state.get("towage_backlog_vessels", 0.0)) / max(1.0, 4.0 * self.towage_service_capacity), 0.0, 1.5),
                np.clip(float(state.get("quay_move_backlog", 0.0)) / max(1.0, 4.0 * self.quay_crane_capacity), 0.0, 1.5),
                np.clip(float(state.get("horizontal_move_backlog", 0.0)) / max(1.0, 4.0 * self.horizontal_transport_capacity), 0.0, 1.5),
                np.clip(float(state.get("yard_move_backlog", 0.0)) / max(1.0, 4.0 * self.yard_crane_capacity), 0.0, 1.5),
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, coordinated]).astype(np.float32, copy=False)

    def _decode_coordinated_action(self, action: Any) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float32).reshape(self.ACTION_DIMENSIONS)
        raw = np.clip(raw, -1.0, 1.0)
        base = self._decode_action(raw[:7])
        allocations = self._coordinated_action_minima + 0.5 * (
            raw[7:] + 1.0
        ) * (
            self._coordinated_action_limits - self._coordinated_action_minima
        )
        return np.concatenate([base, allocations]).astype(np.float32, copy=False)

    def describe_action(self, action: Any) -> Dict[str, float]:
        control = self._decode_coordinated_action(action)
        return {
            "bess_kw": round(float(control[0]) * self.bess_power_kw, 6),
            "service_factor": round(float(control[1]), 6),
            "flexible_load_command": round(float(control[2]), 6),
            "berth_priority": round(float(control[3]), 6),
            "yard_flow_command": round(float(control[4]), 6),
            "inspection_buffer": round(float(control[5]), 6),
            "recovery_priority": round(float(control[6]), 6),
            **{
                name: round(float(control[index]), 6)
                for index, name in enumerate(self.ACTION_NAMES[7:], start=7)
            },
        }

    def _v5_compatibility_action(self, raw: np.ndarray, port: Dict[str, Optional[float]]) -> np.ndarray:
        coordinated = self._decode_coordinated_action(raw)

        def parent_raw(allocation: float, limit_index: int) -> float:
            limit = float(self._integrated_action_limits[limit_index])
            return float(np.clip(2.0 * allocation / max(limit, 1e-9) - 1.0, -1.0, 1.0))

        rail_demand = self._value(port, "rail_transfer_demand_teu")
        barge_demand = self._value(port, "barge_transfer_demand_teu")
        intermodal_total = rail_demand + barge_demand
        intermodal_allocation = (
            (rail_demand * float(coordinated[8]) + barge_demand * float(coordinated[9]))
            / intermodal_total
            if intermodal_total > 1e-9
            else 0.5 * (float(coordinated[8]) + float(coordinated[9]))
        )
        return np.asarray(
            [
                *raw[:7].tolist(),
                parent_raw(float(coordinated[7]), 0),
                parent_raw(intermodal_allocation, 1),
                parent_raw(float(coordinated[10]), 2),
                parent_raw(float(coordinated[11]), 3),
                parent_raw(float(coordinated[12]), 4),
                parent_raw(min(float(coordinated[13]), float(coordinated[14])), 5),
            ],
            dtype=np.float32,
        )

    def project_control(self, action: Any, **kwargs: Any) -> Dict[str, Any]:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.size not in {
            7,
            IntegratedPortOperationsEnv.ACTION_DIMENSIONS,
            self.ACTION_DIMENSIONS,
        }:
            raise ValueError(
                "port_ops_v6 control must contain 7 base, 13 compatibility or 18 coordinated actions"
            )
        if raw.size in {7, IntegratedPortOperationsEnv.ACTION_DIMENSIONS}:
            projected = IntegratedPortOperationsEnv.project_control(self, raw, **kwargs)
        else:
            # Canonical state is checked by the independent business guardrail.
            # This projection keeps the exact inherited power/SOC constraints.
            v5_action = np.asarray(
                [
                    *raw[:8].tolist(),
                    0.5 * (float(raw[8]) + float(raw[9])),
                    float(raw[10]),
                    float(raw[11]),
                    float(raw[12]),
                    min(float(raw[13]), float(raw[14])),
                ],
                dtype=np.float32,
            )
            projected = IntegratedPortOperationsEnv.project_control(
                self, v5_action, **kwargs
            )
            decoded = self._decode_coordinated_action(raw)
            projected.update(
                {
                    name: round(float(decoded[index]), 6)
                    for index, name in enumerate(self.ACTION_NAMES[7:], start=7)
                }
            )
        projected.update(
            coordinated_authority="recommendation_only_no_schedule_release_navigation_or_actuator_authority",
            safety_revision=self.SAFETY_REVISION,
        )
        return projected

    def _feasible_parameterization(
        self,
        raw: np.ndarray,
        *,
        row: np.ndarray,
        port: Dict[str, Optional[float]],
        factors: Dict[str, Optional[float]],
    ) -> tuple[np.ndarray, list[str], float]:
        safe = np.clip(raw, -1.0, 1.0).astype(np.float32, copy=True)
        reasons: list[str] = []

        # Terminal-SOC reachability and battery bounds become the action
        # parameterization rather than a post-hoc learner violation.
        v5_before = self._v5_compatibility_action(safe, port)
        base_control = self._decode_action(v5_before[:7])
        flex_kw = float(base_control[2]) * min(250.0, 0.08 * max(float(row[0]), 1.0))
        service_load_kw = (
            float(row[0])
            * self.operational_load_fraction
            * (float(base_control[1]) - 1.0)
        )
        berth_ratio = (
            float(base_control[3]) / self.berth_priority_limit
            if self.berth_priority_limit
            else 0.0
        )
        yard_ratio = (
            float(base_control[4]) / self.yard_flow_limit
            if self.yard_flow_limit
            else 0.0
        )
        allocation_load_kw = (
            float(row[0])
            * self.allocation_load_fraction
            * 0.5
            * (berth_ratio + yard_ratio)
        )
        readiness_load_kw = (
            float(row[0])
            * self.inspection_readiness_load_fraction
            * max(0.0, float(base_control[5]))
        )
        recovery_load_kw = (
            float(row[0])
            * self.recovery_load_fraction
            * max(0.0, float(base_control[6]))
        )
        coordinated_control = self._decode_coordinated_action(safe)
        connected_shore_kw = (
            self._value(port, "shore_power_demand_kw")
            * self._value(port, "shore_power_connection_ratio")
        )
        shore_power_served_kw = (
            connected_shore_kw
            * self.shore_power_efficiency
            * (0.35 + 0.65 * float(coordinated_control[11]))
        )
        terminal_projected = IntegratedPortOperationsEnv.project_control(
            self,
            v5_before,
            soc=self._soc,
            last_bess_kw=self._last_bess_kw,
            initial_soc=self._initial_soc,
            remaining_steps=self.episode_steps - self._local_step - 1,
        )
        projected_bess_kw = float(terminal_projected["bess_kw"])
        estimated_net_kw = (
            float(row[0])
            + flex_kw
            + service_load_kw
            + allocation_load_kw
            + readiness_load_kw
            + recovery_load_kw
            + shore_power_served_kw
            + projected_bess_kw
        )
        grid_tolerance_kw = max(1e-6, self.demand_cap_kw * 1e-6)
        grid_excess_kw = max(0.0, estimated_net_kw - self.demand_cap_kw)
        shore_adjustment_kw = (
            connected_shore_kw * self.shore_power_efficiency * 0.65
        )
        if grid_excess_kw > grid_tolerance_kw and shore_adjustment_kw > 1e-9:
            shore_limit = float(self._coordinated_action_limits[11 - 7])
            current_shore_allocation = float(coordinated_control[11])
            new_shore_allocation = max(
                0.0,
                current_shore_allocation - grid_excess_kw / shore_adjustment_kw,
            )
            if new_shore_allocation < current_shore_allocation - 1e-9:
                safe[11] = float(
                    np.clip(
                        2.0 * new_shore_allocation / max(shore_limit, 1e-9)
                        - 1.0,
                        -1.0,
                        1.0,
                    )
                )
                reasons.append("shore_power_grid_headroom_parameterization")
                coordinated_control = self._decode_coordinated_action(safe)
                shore_power_served_kw = (
                    connected_shore_kw
                    * self.shore_power_efficiency
                    * (0.35 + 0.65 * float(coordinated_control[11]))
                )
                v5_before = self._v5_compatibility_action(safe, port)
        projected = IntegratedPortOperationsEnv.project_control(
            self,
            v5_before,
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
                - shore_power_served_kw
            ),
        )
        feasible_bess = float(projected["bess_kw"]) / self.bess_power_kw
        if abs(float(safe[0]) - feasible_bess) > 1e-6:
            safe[0] = feasible_bess
            reasons.append("bess_soc_terminal_feasible_parameterization")

        yard_occupancy = 0.0 if factors.get("yard_occupancy_ratio") is None else float(factors["yard_occupancy_ratio"])
        dangerous_goods = self._value(port, "dangerous_goods_workload_ratio")
        if dangerous_goods > 0.0 and yard_occupancy > self.dangerous_goods_yard_limit and safe[4] > 0.0:
            safe[4] = 0.0
            reasons.append("dangerous_goods_inflow_feasible_parameterization")

        # Static gate/intermodal/reefer/maintenance/terminal commitments are
        # already encoded by _decode_coordinated_action. Only state-dependent
        # safety constraints below can correct a latent learner coordinate.

        available_depth = (
            self._value(port, "channel_chart_depth_m")
            + float(row[3])
            - self._value(port, "planning_vessel_draft_m")
            - self._value(port, "squat_allowance_m")
        )
        closure_open = not (
            factors.get("closure_flag") is not None
            and float(factors["closure_flag"]) >= 0.5
        )
        marine_open = (
            available_depth >= self.required_under_keel_clearance_m
            and closure_open
        )
        if not marine_open:
            if safe[13] > -1.0 or safe[14] > -1.0:
                reasons.append("marine_window_feasible_parameterization")
            safe[13] = -1.0
            safe[14] = -1.0
        else:
            pilot_limit = float(self._coordinated_action_limits[13 - 7])
            tow_limit = float(self._coordinated_action_limits[14 - 7])
            pilot_minimum_raw = (
                2.0 * self.minimum_capacity_factors["marine"]
                / max(pilot_limit, 1e-9)
                - 1.0
            )
            tow_minimum_raw = (
                2.0 * self.minimum_capacity_factors["marine"]
                / max(tow_limit, 1e-9)
                - 1.0
            )
            if safe[13] < pilot_minimum_raw:
                safe[13] = pilot_minimum_raw
                reasons.append("pilotage_minimum_service_commitment")
            if safe[14] < tow_minimum_raw:
                safe[14] = tow_minimum_raw
                reasons.append("towage_minimum_service_commitment")

        correction = float(np.mean(np.abs(safe - raw)))
        return safe, reasons, correction

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._reset_coordinated_state()
        _observation, info = super().reset(seed=seed, options=options)
        self._totals.update(
            gate_initial_backlog_trucks=self._gate_backlog,
            rail_demand_teu=0.0,
            rail_served_teu=0.0,
            rail_backlog_teu_hours=0.0,
            barge_demand_teu=0.0,
            barge_served_teu=0.0,
            barge_backlog_teu_hours=0.0,
            pilotage_demand_vessels=0.0,
            pilotage_served_vessels=0.0,
            pilotage_wait_vessel_hours=0.0,
            towage_demand_vessels=0.0,
            towage_served_vessels=0.0,
            towage_wait_vessel_hours=0.0,
            terminal_move_demand=0.0,
            quay_crane_moves=0.0,
            horizontal_transport_moves=0.0,
            yard_crane_moves=0.0,
            quay_move_backlog_hours=0.0,
            horizontal_move_backlog_hours=0.0,
            yard_move_backlog_hours=0.0,
            coordinated_reward=0.0,
            latent_action_correction_sum=0.0,
            latent_action_correction_steps=0.0,
        )
        info.update(
            environment_version=self.environment_version,
            observation_dimensions=self.OBSERVATION_DIMENSIONS,
            action_dimensions=self.ACTION_DIMENSIONS,
            coordinated_authority="recommendation_only_no_schedule_release_navigation_or_actuator_authority",
        )
        return self._observation(), info

    def step(self, action: Any):
        raw = np.asarray(action, dtype=np.float32).reshape(self.ACTION_DIMENSIONS)
        data_index = self._start + self._local_step
        row = self._row().copy()
        port, availability = self._port_values(data_index)
        factor_values = self.segment_factor_values[data_index]
        factor_mask = self.segment_factor_availability[data_index]
        factors = {
            name: float(factor_values[index]) if factor_mask[index] > 0.5 else None
            for index, name in enumerate(FACTOR_COLUMNS)
        }
        safe_raw, correction_reasons, correction = self._feasible_parameterization(
            raw, row=row, port=port, factors=factors
        )
        control = self._decode_coordinated_action(safe_raw)
        v5_action = self._v5_compatibility_action(safe_raw, port)
        _observation, base_reward, terminated, truncated, info = super().step(v5_action)

        labor = self._value(port, "labor_availability_ratio", 1.0)
        intermodal_ratio = self._value(port, "intermodal_capacity_ratio", 1.0)
        rail_demand = self._value(port, "rail_transfer_demand_teu")
        barge_demand = self._value(port, "barge_transfer_demand_teu")
        self._rail_backlog += rail_demand
        self._barge_backlog += barge_demand
        rail_capacity = self.rail_service_capacity * self.step_hours * intermodal_ratio * labor * (0.5 + float(control[8]))
        barge_capacity = self.barge_service_capacity * self.step_hours * intermodal_ratio * labor * (0.5 + float(control[9]))
        rail_served = min(self._rail_backlog, max(0.0, rail_capacity))
        barge_served = min(self._barge_backlog, max(0.0, barge_capacity))
        self._rail_backlog = max(0.0, self._rail_backlog - rail_served)
        self._barge_backlog = max(0.0, self._barge_backlog - barge_served)

        pilotage_demand = self._value(port, "pilotage_demand_vessels")
        towage_demand = self._value(port, "tug_demand_vessels")
        self._pilotage_backlog += pilotage_demand
        self._towage_backlog += towage_demand
        marine_open = bool(info.get("marine_window_open"))
        pilot_tug = 1.0 if factors.get("pilot_tug_availability_ratio") is None else float(factors["pilot_tug_availability_ratio"])
        channel = self._value(port, "channel_capacity_ratio", 1.0)
        pilotage_capacity = self.pilotage_service_capacity * self.step_hours * channel * pilot_tug * labor * (0.5 + float(control[13])) if marine_open else 0.0
        towage_capacity = self.towage_service_capacity * self.step_hours * channel * pilot_tug * labor * (0.5 + float(control[14])) if marine_open else 0.0
        pilotage_served = min(self._pilotage_backlog, max(0.0, pilotage_capacity))
        towage_served = min(self._towage_backlog, max(0.0, towage_capacity))
        self._pilotage_backlog = max(0.0, self._pilotage_backlog - pilotage_served)
        self._towage_backlog = max(0.0, self._towage_backlog - towage_served)

        move_demand = max(0.0, float(row[1]) * self.container_moves_per_teu)
        self._quay_move_backlog += move_demand
        crane_availability = 1.0 if factors.get("crane_availability_ratio") is None else float(factors["crane_availability_ratio"])
        equipment_availability = 1.0 if factors.get("equipment_availability_ratio") is None else float(factors["equipment_availability_ratio"])
        quay_capacity = self.quay_crane_capacity * self.step_hours * crane_availability * labor * (0.5 + float(control[15]))
        quay_served = min(self._quay_move_backlog, max(0.0, quay_capacity))
        self._quay_move_backlog = max(0.0, self._quay_move_backlog - quay_served)
        self._horizontal_move_backlog += quay_served
        horizontal_capacity = self.horizontal_transport_capacity * self.step_hours * equipment_availability * labor * (0.5 + float(control[16]))
        horizontal_served = min(self._horizontal_move_backlog, max(0.0, horizontal_capacity))
        self._horizontal_move_backlog = max(0.0, self._horizontal_move_backlog - horizontal_served)
        self._yard_move_backlog += horizontal_served
        yard_capacity = self.yard_crane_capacity * self.step_hours * min(crane_availability, equipment_availability) * labor * (0.5 + float(control[17]))
        yard_served = min(self._yard_move_backlog, max(0.0, yard_capacity))
        self._yard_move_backlog = max(0.0, self._yard_move_backlog - yard_served)

        allocations = control[7:]
        components = {
            "rail_backlog": self._rail_backlog / max(1.0, self.rail_service_capacity),
            "barge_backlog": self._barge_backlog / max(1.0, self.barge_service_capacity),
            "pilotage_backlog": self._pilotage_backlog / max(1.0, self.pilotage_service_capacity),
            "towage_backlog": self._towage_backlog / max(1.0, self.towage_service_capacity),
            "terminal_move_chain": (
                self._quay_move_backlog / max(1.0, self.quay_crane_capacity)
                + self._horizontal_move_backlog / max(1.0, self.horizontal_transport_capacity)
                + self._yard_move_backlog / max(1.0, self.yard_crane_capacity)
            ) / 3.0,
            "resource_imbalance": float(np.std(allocations[8:11])),
            "allocation_change": float(np.mean(np.abs(allocations - self._previous_allocations))),
            "latent_action_correction": correction,
        }
        coordinated_penalty = sum(
            self.coordinated_reward_weights[name] * components[name]
            for name in self.coordinated_reward_weights
        )
        coordinated_reward = -float(coordinated_penalty)
        reward = float(base_reward + coordinated_reward)
        self._previous_allocations = allocations.astype(np.float32, copy=True)

        self._totals["reward"] += coordinated_reward
        self._totals["coordinated_reward"] += coordinated_reward
        self._totals["rail_demand_teu"] += rail_demand
        self._totals["rail_served_teu"] += rail_served
        self._totals["rail_backlog_teu_hours"] += self._rail_backlog * self.step_hours
        self._totals["barge_demand_teu"] += barge_demand
        self._totals["barge_served_teu"] += barge_served
        self._totals["barge_backlog_teu_hours"] += self._barge_backlog * self.step_hours
        self._totals["pilotage_demand_vessels"] += pilotage_demand
        self._totals["pilotage_served_vessels"] += pilotage_served
        self._totals["pilotage_wait_vessel_hours"] += self._pilotage_backlog * self.step_hours
        self._totals["towage_demand_vessels"] += towage_demand
        self._totals["towage_served_vessels"] += towage_served
        self._totals["towage_wait_vessel_hours"] += self._towage_backlog * self.step_hours
        self._totals["terminal_move_demand"] += move_demand
        self._totals["quay_crane_moves"] += quay_served
        self._totals["horizontal_transport_moves"] += horizontal_served
        self._totals["yard_crane_moves"] += yard_served
        self._totals["quay_move_backlog_hours"] += self._quay_move_backlog * self.step_hours
        self._totals["horizontal_move_backlog_hours"] += self._horizontal_move_backlog * self.step_hours
        self._totals["yard_move_backlog_hours"] += self._yard_move_backlog * self.step_hours
        self._totals["latent_action_correction_sum"] += correction
        self._totals["latent_action_correction_steps"] += float(bool(correction_reasons))

        info.update(
            environment_version=self.environment_version,
            port_base_reward=float(
                base_reward - float(info.get("integrated_reward") or 0.0)
            ),
            rail_backlog_teu=self._rail_backlog,
            rail_served_teu=rail_served,
            barge_backlog_teu=self._barge_backlog,
            barge_served_teu=barge_served,
            pilotage_backlog_vessels=self._pilotage_backlog,
            pilotage_served_vessels=pilotage_served,
            towage_backlog_vessels=self._towage_backlog,
            towage_served_vessels=towage_served,
            quay_move_backlog=self._quay_move_backlog,
            quay_crane_moves=quay_served,
            horizontal_move_backlog=self._horizontal_move_backlog,
            horizontal_transport_moves=horizontal_served,
            yard_move_backlog=self._yard_move_backlog,
            yard_crane_moves=yard_served,
            coordinated_control={name: float(value) for name, value in zip(self.ACTION_NAMES, control)},
            latent_action_correction=correction,
            latent_action_correction_reasons=correction_reasons,
            coordinated_reward_components=components,
            coordinated_reward=coordinated_reward,
            coordinated_authority="recommendation_only_no_schedule_release_navigation_or_actuator_authority",
            port_wide_factor_availability=availability,
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
            gate_service_completion_ratio=min(
                1.0,
                row["gate_served_trucks"]
                / max(
                    1.0,
                    row["gate_demand_trucks"]
                    + row["gate_initial_backlog_trucks"],
                ),
            ),
            rail_service_completion_ratio=row["rail_served_teu"] / max(1.0, row["rail_demand_teu"]),
            barge_service_completion_ratio=row["barge_served_teu"] / max(1.0, row["barge_demand_teu"]),
            pilotage_service_completion_ratio=row["pilotage_served_vessels"] / max(1.0, row["pilotage_demand_vessels"]),
            towage_service_completion_ratio=row["towage_served_vessels"] / max(1.0, row["towage_demand_vessels"]),
            quay_crane_completion_ratio=row["quay_crane_moves"] / max(1.0, row["terminal_move_demand"]),
            horizontal_transport_completion_ratio=row["horizontal_transport_moves"] / max(1.0, row["terminal_move_demand"]),
            yard_crane_completion_ratio=row["yard_crane_moves"] / max(1.0, row["terminal_move_demand"]),
            latent_action_correction_rate=row["latent_action_correction_steps"] / steps,
            latent_action_correction_mean=row["latent_action_correction_sum"] / steps,
            rail_backlog_end_teu=self._rail_backlog,
            barge_backlog_end_teu=self._barge_backlog,
            pilotage_backlog_end_vessels=self._pilotage_backlog,
            towage_backlog_end_vessels=self._towage_backlog,
            quay_move_backlog_end=self._quay_move_backlog,
            horizontal_move_backlog_end=self._horizontal_move_backlog,
            yard_move_backlog_end=self._yard_move_backlog,
        )
        return row
