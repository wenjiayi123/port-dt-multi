from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .datasets import FACTOR_COLUMNS, NUMERIC_COLUMNS, PortDataset
from .profiles import DEFAULT_PROFILE, validate_profile


class PortOperationsEnv(gym.Env):
    """Numerical port energy/throughput environment with no training renderer.

    The environment consumes chronological rows only. Continuous policies control
    BESS power, service intensity and flexible load. DQN receives a finite action
    lattice over the same controls. Trace collection is opt-in and used only by
    the separate evaluation endpoint.
    """

    metadata = {"render_modes": []}

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
        environment_version: str = "port_ops_v1",
        port_profile: Optional[Dict[str, Any]] = None,
        normalization_slice: Optional[slice] = None,
        training: bool = True,
        record_trace: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.segment = dataset.values[data_slice].astype(np.float32, copy=True)
        self.segment_timestamps = list(dataset.timestamps[data_slice])
        self.segment_factor_values = dataset.factor_values[data_slice].astype(np.float32, copy=True)
        self.segment_factor_availability = dataset.factor_availability[data_slice].astype(np.float32, copy=True)
        if len(self.segment) < 4:
            raise ValueError("dataset slice is too short")
        self.action_mode = action_mode
        self.environment_version = str(environment_version)
        if self.environment_version not in {"port_ops_v1", "port_ops_v2"}:
            raise ValueError(f"unsupported environment_version: {self.environment_version}")
        self.port_profile = validate_profile(port_profile or DEFAULT_PROFILE)
        self.episode_steps = max(4, min(int(episode_steps), len(self.segment) - 1))
        self.training = bool(training)
        self.record_trace = bool(record_trace)
        if self.training and self.record_trace:
            raise ValueError("training environments must not collect render traces")
        self.demand_cap_kw = max(float(demand_cap_kw), 1.0)
        base_weights = dict(self.port_profile["objectives"])
        base_weights.update({k: float(v) for k, v in (reward_weights or {}).items() if k in base_weights})
        total = sum(max(0.0, v) for v in base_weights.values()) or 1.0
        self.weights = {k: max(0.0, v) / total for k, v in base_weights.items()}
        assets = self.port_profile["assets"]
        limits = self.port_profile["control_limits"]
        self.bess_capacity_kwh = float(assets["bess_capacity_kwh"])
        self.bess_power_kw = min(float(assets["bess_power_kw"]), self.demand_cap_kw * 0.50)
        self.soc_min = float(limits["soc_min"])
        self.soc_max = float(limits["soc_max"])
        self.service_min = float(limits["service_factor_min"])
        self.service_max = float(limits["service_factor_max"])
        self.flexible_limit = float(limits["flexible_load_fraction"])
        self.berth_priority_limit = float(limits["berth_priority_limit"])
        self.yard_flow_limit = float(limits["yard_flow_limit"])
        cadence_seconds = float(dataset_quality_cadence(dataset))
        self.step_hours = cadence_seconds / 3600.0
        self._rng = np.random.default_rng(seed)
        self._seed = int(seed)
        # Observation scaling is fitted on the chronological training slice
        # only. Evaluation must not use held-out extrema to normalize itself.
        normalization_train = normalization_slice or dataset.split()[0]
        normalization_reference = dataset.values[normalization_train].astype(
            np.float32, copy=False
        )
        self._mins = np.nanmin(normalization_reference, axis=0)
        self._maxs = np.nanmax(normalization_reference, axis=0)
        self._spans = np.maximum(self._maxs - self._mins, 1e-6)
        factor_reference = dataset.factor_values[normalization_train].astype(np.float32, copy=False)
        factor_mask_reference = dataset.factor_availability[normalization_train].astype(np.float32, copy=False)
        self._factor_mins = np.zeros(len(FACTOR_COLUMNS), dtype=np.float32)
        self._factor_spans = np.ones(len(FACTOR_COLUMNS), dtype=np.float32)
        for index in range(len(FACTOR_COLUMNS)):
            available = factor_mask_reference[:, index] > 0.5
            if np.any(available):
                observed = factor_reference[available, index]
                self._factor_mins[index] = float(np.min(observed))
                self._factor_spans[index] = max(float(np.max(observed) - np.min(observed)), 1e-6)
        observation_size = 13 if self.environment_version == "port_ops_v1" else 13 + 2 * len(FACTOR_COLUMNS)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(observation_size,), dtype=np.float32)
        if action_mode == "discrete":
            if self.environment_version == "port_ops_v1":
                lattice = [
                    (b, s, f)
                    for b in (-1.0, -0.5, 0.0, 0.5, 1.0)
                    for s in (self.service_min, 1.0, self.service_max)
                    for f in (-self.flexible_limit, 0.0, self.flexible_limit)
                ]
            else:
                lattice = [
                    (b, s, f, berth, yard)
                    for b in (-1.0, -0.5, 0.0, 0.5, 1.0)
                    for s in (self.service_min, 1.0, self.service_max)
                    for f in (-self.flexible_limit, 0.0, self.flexible_limit)
                    for berth in (-self.berth_priority_limit, 0.0, self.berth_priority_limit)
                    for yard in (-self.yard_flow_limit, 0.0, self.yard_flow_limit)
                ]
            self._discrete_actions = np.asarray(lattice, dtype=np.float32)
            self.action_space = spaces.Discrete(len(self._discrete_actions))
        elif action_mode == "continuous":
            self._discrete_actions = None
            action_size = 3 if self.environment_version == "port_ops_v1" else 5
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)
        else:
            raise ValueError(f"unsupported action_mode: {action_mode}")
        self.trace: list[Dict[str, Any]] = []
        self.render_calls = 0
        self._start = 0
        self._local_step = 0
        self._soc = 0.55
        self._initial_soc = 0.55
        self._queue = 0.0
        self._last_bess_kw = 0.0
        self._totals: Dict[str, float] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = max(0, len(self.segment) - self.episode_steps - 1)
        requested = (options or {}).get("start_index")
        if requested is not None:
            self._start = max(0, min(int(requested), max_start))
        elif self.training and max_start:
            self._start = int(self._rng.integers(0, max_start + 1))
        else:
            self._start = 0
        self._local_step = 0
        self._soc = 0.45 + float(self._rng.random()) * 0.20 if self.training else 0.55
        self._initial_soc = self._soc
        self._queue = max(0.0, float(self.segment[self._start, 2]) * 0.8)
        self._last_bess_kw = 0.0
        self.trace = []
        self._totals = {"reward": 0.0, "energy_cost": 0.0, "carbon_kg": 0.0, "throughput_teu": 0.0, "delay": 0.0, "violations": 0.0, "peak_kw": 0.0}
        return self._observation(), {"dataset_index": self._start, "training": self.training}

    def _row(self) -> np.ndarray:
        return self.segment[self._start + self._local_step]

    def _decode_action(self, action: Any) -> np.ndarray:
        if self.action_mode == "discrete":
            return self._discrete_actions[int(action)].copy()
        action_size = 3 if self.environment_version == "port_ops_v1" else 5
        continuous = np.asarray(action, dtype=np.float32).reshape(action_size)
        continuous = np.clip(continuous, -1.0, 1.0)
        continuous[1] = (
            1.0 + (self.service_max - 1.0) * continuous[1]
            if continuous[1] >= 0
            else 1.0 + (1.0 - self.service_min) * continuous[1]
        )
        continuous[2] = self.flexible_limit * continuous[2]
        if action_size == 5:
            continuous[3] *= self.berth_priority_limit
            continuous[4] *= self.yard_flow_limit
        return continuous

    def _factor_observation(self, values: np.ndarray, mask: np.ndarray) -> list[float]:
        normalized = 2.0 * (values - self._factor_mins) / self._factor_spans - 1.0
        normalized = np.where(mask > 0.5, normalized, 0.0)
        return [*normalized.astype(np.float32).tolist(), *mask.astype(np.float32).tolist()]

    def observation_from_state(self, state: Dict[str, Any]) -> np.ndarray:
        """Encode one canonical port state using this dataset's train/test scale."""
        missing = [column for column in NUMERIC_COLUMNS if state.get(column) is None]
        if missing:
            raise ValueError(f"state is missing canonical fields: {', '.join(missing)}")
        row = np.asarray([float(state[column]) for column in NUMERIC_COLUMNS], dtype=np.float32)
        if not np.all(np.isfinite(row)):
            raise ValueError("state contains non-finite canonical values")
        normalized = 2.0 * (row - self._mins) / self._spans - 1.0
        try:
            hour = int(state.get("hour", 0)) % 24
        except (TypeError, ValueError):
            hour = 0
        soc = float(state.get("soc", 0.55))
        queue = max(0.0, float(state.get("queue", 0.0)))
        last_bess_kw = float(state.get("last_bess_kw", 0.0))
        progress = min(1.0, max(0.0, float(state.get("episode_progress", 0.0))))
        factor_values = np.zeros(len(FACTOR_COLUMNS), dtype=np.float32)
        factor_mask = np.zeros(len(FACTOR_COLUMNS), dtype=np.float32)
        for index, column in enumerate(FACTOR_COLUMNS):
            if state.get(column) is None:
                continue
            factor_values[index] = float(state[column])
            factor_mask[index] = 1.0
        return np.asarray(
            [
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                *normalized.tolist(),
                *(
                    self._factor_observation(factor_values, factor_mask)
                    if self.environment_version == "port_ops_v2"
                    else []
                ),
                2.0 * soc - 1.0,
                np.clip(queue / max(1.0, row[1] * 4.0), 0.0, 1.5),
                last_bess_kw / self.bess_power_kw,
                progress,
            ],
            dtype=np.float32,
        )

    def describe_action(self, action: Any) -> Dict[str, float]:
        control = self._decode_action(action)
        result = {
            "bess_kw": round(float(control[0]) * self.bess_power_kw, 6),
            "service_factor": round(float(control[1]), 6),
            "flexible_load_command": round(float(control[2]), 6),
        }
        if self.environment_version == "port_ops_v2":
            result.update(
                berth_priority=round(float(control[3]), 6),
                yard_flow_command=round(float(control[4]), 6),
            )
        return result

    def project_control(
        self,
        action: Any,
        *,
        soc: float,
        last_bess_kw: float,
        initial_soc: float | None = None,
        remaining_steps: int | None = None,
        max_grid_charge_kw: float | None = None,
    ) -> Dict[str, Any]:
        """Apply the same software constraints used by ``step`` without mutation."""
        control = self._decode_action(action)
        requested_bess_kw = float(control[0]) * self.bess_power_kw
        # BESS is dispatched at hourly resolution and may traverse its rated
        # power range within a step. The former 35%/hour limit was a synthetic
        # bottleneck rather than a sourced equipment constraint.
        ramp = self.bess_power_kw
        upper_power = self.bess_power_kw
        if max_grid_charge_kw is not None:
            upper_power = min(upper_power, float(max_grid_charge_kw))
        bess_kw = float(
            np.clip(
                requested_bess_kw,
                max(-self.bess_power_kw, last_bess_kw - ramp),
                min(upper_power, last_bess_kw + ramp),
            )
        )
        projected_soc = float(soc)
        if bess_kw >= 0:
            max_charge = (self.soc_max - projected_soc) * self.bess_capacity_kwh / (0.96 * self.step_hours)
            bess_kw = min(bess_kw, max(0.0, max_charge))
            projected_soc += bess_kw * self.step_hours * 0.96 / self.bess_capacity_kwh
        else:
            max_discharge = (projected_soc - self.soc_min) * self.bess_capacity_kwh * 0.96 / self.step_hours
            bess_kw = -min(abs(bess_kw), max(0.0, max_discharge))
            projected_soc -= abs(bess_kw) * self.step_hours / (0.96 * self.bess_capacity_kwh)
        if initial_soc is not None and remaining_steps is not None:
            remaining_fraction = max(0.0, min(1.0, remaining_steps / self.episode_steps))
            reachable_low = initial_soc - (initial_soc - self.soc_min) * remaining_fraction
            reachable_high = initial_soc + (self.soc_max - initial_soc) * remaining_fraction
            target_soc = float(
                np.clip(projected_soc, reachable_low, reachable_high)
            )
            if target_soc >= soc:
                bess_kw = (
                    (target_soc - soc)
                    * self.bess_capacity_kwh
                    / (0.96 * self.step_hours)
                )
            else:
                bess_kw = -(
                    (soc - target_soc)
                    * self.bess_capacity_kwh
                    * 0.96
                    / self.step_hours
                )
            bess_kw = float(
                np.clip(bess_kw, -self.bess_power_kw, upper_power)
            )
            projected_soc = float(soc)
            if bess_kw >= 0:
                projected_soc += (
                    bess_kw * self.step_hours * 0.96 / self.bess_capacity_kwh
                )
            else:
                projected_soc -= (
                    abs(bess_kw)
                    * self.step_hours
                    / (0.96 * self.bess_capacity_kwh)
                )
        result = {
            "bess_kw": round(float(bess_kw), 6),
            "service_factor": round(float(control[1]), 6),
            "flexible_load_command": round(float(control[2]), 6),
            "projected_soc": round(float(projected_soc), 8),
            "projection_applied": abs(bess_kw - requested_bess_kw) > 1e-6,
            "requested_bess_kw": round(float(requested_bess_kw), 6),
        }
        if self.environment_version == "port_ops_v2":
            result.update(
                berth_priority=round(float(control[3]), 6),
                yard_flow_command=round(float(control[4]), 6),
            )
        return result

    def _observation(self) -> np.ndarray:
        row = self._row()
        normalized = 2.0 * (row - self._mins) / self._spans - 1.0
        idx = self._start + self._local_step
        timestamp = datetime.fromisoformat(
            self.segment_timestamps[idx].replace("Z", "+00:00")
        )
        hour = (
            timestamp.hour
            + timestamp.minute / 60.0
            + timestamp.second / 3600.0
        )
        return np.asarray(
            [
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                *normalized.tolist(),
                *(
                    self._factor_observation(
                        self.segment_factor_values[idx],
                        self.segment_factor_availability[idx],
                    )
                    if self.environment_version == "port_ops_v2"
                    else []
                ),
                2.0 * self._soc - 1.0,
                np.clip(self._queue / max(1.0, row[1] * 4.0), 0.0, 1.5),
                self._last_bess_kw / self.bess_power_kw,
                self._local_step / max(1, self.episode_steps - 1),
            ],
            dtype=np.float32,
        )

    def step(self, action: Any):
        row = self._row()
        control = self._decode_action(action)
        flex_command = float(control[2])
        berth_priority = float(control[3]) if self.environment_version == "port_ops_v2" else 0.0
        yard_flow = float(control[4]) if self.environment_version == "port_ops_v2" else 0.0
        flex_kw = flex_command * min(250.0, 0.08 * max(row[0], 1.0))
        projected = self.project_control(
            action,
            soc=self._soc,
            last_bess_kw=self._last_bess_kw,
            initial_soc=self._initial_soc,
            remaining_steps=self.episode_steps - self._local_step - 1,
            max_grid_charge_kw=self.demand_cap_kw - float(row[0]) - flex_kw,
        )
        bess_kw = float(projected["bess_kw"])
        service_factor = float(projected["service_factor"])
        self._soc = float(projected["projected_soc"])
        # Positive BESS power charges; negative discharges. Projection enforces
        # the same SoC and conservative ramp limits used by inference.
        ramp = self.bess_power_kw
        factor_index = self._start + self._local_step
        factor_values = self.segment_factor_values[factor_index]
        factor_mask = self.segment_factor_availability[factor_index]
        factors = {
            name: (
                float(factor_values[index])
                if factor_mask[index] > 0.5
                else None
            )
            for index, name in enumerate(FACTOR_COLUMNS)
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
                value >= float(limit) if comparison == "high" else value <= float(limit)
            )
        if weather_blocked:
            resource_factor = 0.0
        allocation_factor = max(0.6, 1.0 + 0.08 * berth_priority + 0.08 * yard_flow)
        service_capacity = max(1.0, row[1]) * service_factor * resource_factor * allocation_factor
        incoming_work = max(0.0, row[1] + 2.0 * row[2])
        served = min(self._queue + incoming_work, service_capacity)
        self._queue = max(0.0, self._queue + incoming_work - served)
        net_kw = max(0.0, float(row[0]) + bess_kw + flex_kw)
        price = max(0.0, float(row[4]))
        carbon_factor = max(0.0, float(row[5]))
        energy_cost = net_kw * self.step_hours * price
        carbon_kg = net_kw * self.step_hours * carbon_factor
        # Ignore sub-mill watt floating-point residue created by the action
        # projection; engineering violations are evaluated at 1 W resolution.
        exceed_kw = max(0.0, net_kw - self.demand_cap_kw - 0.001)
        unsafe_soc = max(0.0, self.soc_min - self._soc) + max(0.0, self._soc - self.soc_max)
        ramp_violation = max(0.0, abs(bess_kw - self._last_bess_kw) - ramp)
        terminal_soc_error = (
            abs(self._soc - self._initial_soc)
            if self._local_step + 1 >= self.episode_steps
            else 0.0
        )
        safety_penalty = 50.0 * unsafe_soc + ramp_violation / max(1.0, ramp)
        safety_penalty += 50.0 * terminal_soc_error
        if weather_blocked and service_factor > self.service_min + 1e-6:
            safety_penalty += 1.0
        delay_index = self._queue / max(1.0, incoming_work)
        degradation = abs(bess_kw) * self.step_hours * 0.02
        # Scales are explicit engineering normalizers, not random score shaping.
        components = {
            "cost": energy_cost / max(1.0, self.demand_cap_kw),
            "carbon": carbon_kg / max(1.0, self.demand_cap_kw),
            "peak": exceed_kw / self.demand_cap_kw,
            "safety": safety_penalty,
            "delay": delay_index,
        }
        reward = -sum(self.weights[name] * value for name, value in components.items()) - degradation / 1000.0
        violation = bool(exceed_kw > 0 or safety_penalty > 0)
        self._totals["reward"] += reward
        self._totals["energy_cost"] += energy_cost
        self._totals["carbon_kg"] += carbon_kg
        self._totals["throughput_teu"] += served
        self._totals["delay"] += delay_index
        self._totals["violations"] += float(violation)
        self._totals["peak_kw"] = max(self._totals["peak_kw"], net_kw)
        timestamp = self.segment_timestamps[self._start + self._local_step]
        info = {
            "timestamp": timestamp,
            "baseline_kw": float(row[0]),
            "net_load_kw": net_kw,
            "bess_kw": bess_kw,
            "soc": self._soc,
            "queue": self._queue,
            "served_teu": served,
            "energy_cost": energy_cost,
            "carbon_kg": carbon_kg,
            "delay_index": delay_index,
            "guardrail_violation": violation,
            "terminal_soc_error": terminal_soc_error,
            "action_projection_applied": bool(projected["projection_applied"]),
            "environment_version": self.environment_version,
            "berth_priority": berth_priority,
            "yard_flow_command": yard_flow,
            "operational_resource_factor": resource_factor,
            "weather_blocked": weather_blocked,
            "factor_availability": {
                name: bool(factor_mask[index] > 0.5)
                for index, name in enumerate(FACTOR_COLUMNS)
            },
            "reward_components": components,
        }
        if self.record_trace:
            self.trace.append(dict(info, reward=reward, action=[float(x) for x in control]))
        self._last_bess_kw = bess_kw
        self._local_step += 1
        terminated = self._local_step >= self.episode_steps
        truncated = False
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info["episode_metrics"] = dict(self._totals)
        else:
            observation = self._observation()
        return observation, float(reward), terminated, truncated, info

    def render(self):
        self.render_calls += 1
        raise RuntimeError("Rendering is disabled inside the training environment; use the evaluation API")

    @property
    def totals(self) -> Dict[str, float]:
        return dict(self._totals)


def dataset_quality_cadence(dataset: PortDataset) -> float:
    if len(dataset.timestamps) < 2:
        return 3600.0
    start = datetime.fromisoformat(dataset.timestamps[0].replace("Z", "+00:00"))
    end = datetime.fromisoformat(dataset.timestamps[1].replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    seconds = float((end - start).total_seconds())
    return max(1.0, seconds)
