"""Landing-oriented Shore+BESS environment for the V3.1 evidence track.

The checked-in legacy Shore+BESS trainer is intentionally left untouched.  It
used a 145-row all-zero-action dataset, so it is useful history but not an
admissible policy.  This environment consumes the canonical 17,544-row public
Shanghai time series, keeps chronological isolation, and models only advisory
controls that can later be mapped to an EMS/BMS gateway.

Sign convention: positive BESS power discharges into the PCC; negative power
charges from the PCC.  Mandatory vessel hotel load is never shed by an action.
Only explicitly flexible shore-side auxiliary load can be shifted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.services.rl_training.datasets import FACTOR_COLUMNS, PortDataset, load_port_dataset


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = ROOT / "config" / "shore_bess_v3.json"

STATE_NAMES = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "base_load_ratio",
    "shore_mandatory_ratio",
    "shore_auxiliary_ratio",
    "price_ratio",
    "known_tariff_6h_mean_ratio",
    "carbon_factor_ratio",
    "throughput_ratio",
    "vessel_arrivals_ratio",
    "tide_ratio",
    "ambient_temperature_ratio",
    "wind_speed_ratio",
    "wave_height_ratio",
    "current_speed_ratio",
    "berth_occupancy_ratio",
    "yard_occupancy_ratio",
    "equipment_availability_ratio",
    "channel_congestion_ratio",
    "reefer_load_ratio",
    "pilot_tug_availability_ratio",
    "closure_flag",
    "visibility_available",
    "soc_scaled",
    "soh_scaled",
    "battery_temperature_scaled",
    "last_bess_power_ratio",
    "flex_backlog_ratio",
    "trailing_pcc_ratio",
    "demand_headroom_ratio",
    "reserve_requirement_ratio",
    "episode_progress",
]

ACTION_NAMES = [
    "bess_dispatch_ratio",
    "shore_auxiliary_shift_ratio",
]

DISCRETE_ACTION_LATTICE = np.asarray(
    [
        (bess_ratio, flex_ratio)
        for bess_ratio in (-1.0, -0.5, 0.0, 0.5, 1.0)
        for flex_ratio in (-0.30, 0.0, 0.30)
    ],
    dtype=np.float32,
)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("control_authority") != "recommendation_only":
        raise ValueError("open-source Shore+BESS control authority must remain recommendation_only")
    weights = payload.get("reward_weights") or {}
    required = {
        "energy_cost",
        "carbon",
        "demand_peak",
        "degradation",
        "reserve",
        "shore_sla",
        "safety_projection",
        "terminal_state",
    }
    if set(weights) != required or sum(float(value) for value in weights.values()) <= 0:
        raise ValueError("shore_bess_v3 reward weights are incomplete")
    return payload


def load_public_dataset(config: Optional[Mapping[str, Any]] = None) -> PortDataset:
    resolved = dict(config or load_config())
    return load_port_dataset(str(resolved["dataset_id"]))


def chronological_slices(dataset: PortDataset) -> tuple[slice, slice, slice]:
    return dataset.split_three_way(test_ratio=0.20, validation_ratio=0.10)


@dataclass(frozen=True)
class ShoreBESSContract:
    state_names: List[str]
    action_names: List[str]
    reward_components: List[str]
    hard_constraints: List[str]
    landing_inputs: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state_dimensions": len(self.state_names),
            "state_names": list(self.state_names),
            "action_dimensions": len(self.action_names),
            "action_names": list(self.action_names),
            "reward_components": list(self.reward_components),
            "hard_constraints": list(self.hard_constraints),
            "landing_inputs": list(self.landing_inputs),
        }


CONTRACT = ShoreBESSContract(
    state_names=STATE_NAMES,
    action_names=ACTION_NAMES,
    reward_components=[
        "energy_cost",
        "carbon",
        "demand_peak",
        "degradation",
        "reserve",
        "shore_sla",
        "safety_projection",
        "terminal_state",
    ],
    hard_constraints=[
        "mandatory_shore_power_not_shed",
        "soc_min_max",
        "charge_discharge_power",
        "hourly_ramp",
        "temperature_derate_and_trip",
        "soh_derate",
        "pcc_hard_limit",
        "no_reverse_power",
        "n_minus_1_reserve_headroom",
        "terminal_soc_reachability",
        "flexible_auxiliary_backlog_recovery",
        "equipment_availability_interlock",
    ],
    landing_inputs=[
        "berth_shore_meter_kw",
        "ship_connection_and_compatibility",
        "mandatory_and_flexible_load_tags",
        "bms_soc_soh_temperature_alarm",
        "pcc_active_power_and_15min_demand",
        "settlement_tariff_and_demand_charge",
        "marginal_grid_carbon_factor",
        "bess_efficiency_and_degradation_curve",
        "n_minus_1_reserve_requirement",
        "equipment_availability_and_interlocks",
        "forecast_quality_and_clock_quality",
        "gateway_ack_nonce_ttl_and_operator_authority",
    ],
)


class ShoreBESSEnv(gym.Env):
    """Two-action continuous Shore+BESS simulator with fail-closed projection."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: PortDataset,
        data_slice: slice,
        *,
        config: Optional[Mapping[str, Any]] = None,
        normalization_slice: Optional[slice] = None,
        episode_steps: Optional[int] = None,
        seed: int = 43,
        training: bool = True,
        record_trace: bool = False,
    ) -> None:
        super().__init__()
        self.config = dict(config or load_config())
        self.dataset = dataset
        self.data_slice = data_slice
        self.segment = dataset.values[data_slice].astype(np.float32, copy=True)
        self.segment_factor_values = dataset.factor_values[data_slice].astype(np.float32, copy=True)
        self.segment_factor_masks = dataset.factor_availability[data_slice].astype(np.float32, copy=True)
        self.timestamps = list(dataset.timestamps[data_slice])
        if len(self.segment) < 169:
            raise ValueError("Shore+BESS split requires at least 169 chronological rows")
        self.training = bool(training)
        self.record_trace = bool(record_trace)
        if self.training and self.record_trace:
            raise ValueError("training must not render or record replay traces")
        configured_steps = int(self.config["training"]["episode_hours"])
        self.episode_steps = max(24, min(int(episode_steps or configured_steps), len(self.segment) - 1))
        self._rng = np.random.default_rng(seed)

        asset = self.config["asset"]
        service = self.config["shore_service"]
        grid = self.config["grid"]
        self.energy_kwh = float(asset["rated_energy_kwh"])
        self.power_kw = float(asset["rated_power_kw"])
        self.charge_eff = float(asset["charge_efficiency"])
        self.discharge_eff = float(asset["discharge_efficiency"])
        self.soc_min = float(asset["soc_min"])
        self.soc_max = float(asset["soc_max"])
        self.soc_initial = float(asset["soc_initial"])
        self.soh_initial = float(asset["soh_initial"])
        self.ramp_kw = float(asset["ramp_kw_per_hour"])
        self.temperature_derate_c = float(asset["temperature_derate_c"])
        self.temperature_trip_c = float(asset["temperature_trip_c"])
        self.cycle_cost = float(asset["cycle_cost_cny_per_kwh"])
        self.flex_fraction = float(service["flexible_aux_fraction"])
        self.flex_limit = float(service["flexible_aux_limit"])
        self.hard_pcc_limit_kw = float(grid["hard_pcc_limit_kw"])
        self.export_allowed = bool(grid["export_allowed"])
        self.weights = {name: float(value) for name, value in self.config["reward_weights"].items()}
        weight_total = sum(self.weights.values())
        self.weights = {name: value / weight_total for name, value in self.weights.items()}

        train_slice = normalization_slice or chronological_slices(dataset)[0]
        train_values = dataset.values[train_slice].astype(np.float64, copy=False)
        self.base_scale = float(max(1.0, np.quantile(train_values[:, 0], 0.99)))
        self.throughput_scale = float(max(1.0, np.quantile(train_values[:, 1], 0.99)))
        self.arrival_scale = float(max(1.0, np.quantile(train_values[:, 2], 0.99)))
        self.price_scale = float(max(1e-6, np.max(train_values[:, 4])))
        self.carbon_scale = float(max(1e-6, np.max(train_values[:, 5])))
        self.ambient_scale = 45.0
        self.soft_cap_kw = float(
            np.quantile(train_values[:, 0], float(grid["soft_cap_train_quantile"]))
        )
        self.max_backlog_kwh = self.base_scale * self.flex_fraction * 12.0
        self.factor_index = {name: index for index, name in enumerate(FACTOR_COLUMNS)}

        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(len(STATE_NAMES),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(ACTION_NAMES),), dtype=np.float32)
        self.render_calls = 0
        self.trace: List[Dict[str, Any]] = []
        self._start = 0
        self._step = 0
        self._soc = self.soc_initial
        self._soh = self.soh_initial
        self._temperature_c = 28.0
        self._last_bess_kw = 0.0
        self._flex_backlog_kwh = 0.0
        self._pcc_history: List[float] = []
        self._totals: Dict[str, float] = {}

    def _factor(self, local_index: int, name: str, default: float = 0.0) -> float:
        index = self.factor_index[name]
        if self.segment_factor_masks[local_index, index] <= 0.5:
            return float(default)
        return float(self.segment_factor_values[local_index, index])

    def _row_context(self) -> Dict[str, float]:
        index = self._start + self._step
        row = self.segment[index]
        berth = self._factor(index, "berth_occupancy_ratio", 0.0)
        service = self.config["shore_service"]
        shore_fraction = float(service["mandatory_load_fraction_min"]) + float(
            service["mandatory_load_fraction_occupancy_gain"]
        ) * berth
        mandatory_kw = float(row[0]) * shore_fraction
        auxiliary_kw = mandatory_kw * self.flex_fraction
        timestamp = datetime.fromisoformat(self.timestamps[index].replace("Z", "+00:00"))
        critical = timestamp.hour in set(int(value) for value in service["critical_hours_local"])
        reserve_kw = float(service["reserve_critical_kw"] if critical else service["reserve_min_kw"])
        reserve_kw *= 0.75 + 0.5 * berth
        tariff_window = self.segment[index : min(len(self.segment), index + 6), 4]
        known_tariff_6h = float(np.mean(tariff_window)) if len(tariff_window) else float(row[4])
        return {
            "index": float(index),
            "base_load_kw": float(row[0]),
            "throughput_teu": float(row[1]),
            "vessel_arrivals": float(row[2]),
            "tide_m": float(row[3]),
            "price_cny_per_kwh": float(row[4]),
            "carbon_kg_per_kwh": float(row[5]),
            "ambient_c": float(row[6]),
            "mandatory_shore_kw": mandatory_kw,
            "auxiliary_shore_kw": auxiliary_kw,
            "reserve_required_kw": reserve_kw,
            "known_tariff_6h_mean": known_tariff_6h,
            "berth_occupancy_ratio": berth,
            "yard_occupancy_ratio": self._factor(index, "yard_occupancy_ratio", 0.0),
            "equipment_availability_ratio": self._factor(index, "equipment_availability_ratio", 0.0),
            "channel_congestion_ratio": self._factor(index, "channel_congestion_ratio", 0.0),
            "reefer_load_kw": self._factor(index, "reefer_load_kw", 0.0),
            "pilot_tug_availability_ratio": self._factor(index, "pilot_tug_availability_ratio", 0.0),
            "closure_flag": self._factor(index, "closure_flag", 0.0),
            "wind_speed_mps": self._factor(index, "wind_speed_mps", 0.0),
            "wave_height_m": self._factor(index, "wave_height_m", 0.0),
            "current_speed_mps": self._factor(index, "current_speed_mps", 0.0),
            "visibility_available": float(
                self.segment_factor_masks[index, self.factor_index["visibility_km"]] > 0.5
            ),
            "timestamp_hour": float(timestamp.hour + timestamp.minute / 60.0),
            "weekday": float(timestamp.weekday()),
        }

    def _observation(self) -> np.ndarray:
        ctx = self._row_context()
        hour_angle = 2.0 * math.pi * ctx["timestamp_hour"] / 24.0
        weekday_angle = 2.0 * math.pi * ctx["weekday"] / 7.0
        trailing_pcc = float(np.mean(self._pcc_history[-3:])) if self._pcc_history else ctx["base_load_kw"]
        headroom = (self.hard_pcc_limit_kw - ctx["base_load_kw"]) / self.hard_pcc_limit_kw
        values = [
            math.sin(hour_angle),
            math.cos(hour_angle),
            math.sin(weekday_angle),
            math.cos(weekday_angle),
            ctx["base_load_kw"] / self.base_scale,
            ctx["mandatory_shore_kw"] / self.base_scale,
            ctx["auxiliary_shore_kw"] / self.base_scale,
            ctx["price_cny_per_kwh"] / self.price_scale,
            ctx["known_tariff_6h_mean"] / self.price_scale,
            ctx["carbon_kg_per_kwh"] / self.carbon_scale,
            ctx["throughput_teu"] / self.throughput_scale,
            ctx["vessel_arrivals"] / self.arrival_scale,
            ctx["tide_m"] / 5.0,
            ctx["ambient_c"] / self.ambient_scale,
            ctx["wind_speed_mps"] / 25.0,
            ctx["wave_height_m"] / 4.0,
            ctx["current_speed_mps"] / 4.0,
            ctx["berth_occupancy_ratio"],
            ctx["yard_occupancy_ratio"],
            ctx["equipment_availability_ratio"],
            ctx["channel_congestion_ratio"],
            ctx["reefer_load_kw"] / self.base_scale,
            ctx["pilot_tug_availability_ratio"],
            ctx["closure_flag"],
            ctx["visibility_available"],
            2.0 * (self._soc - self.soc_min) / (self.soc_max - self.soc_min) - 1.0,
            2.0 * self._soh - 1.0,
            (self._temperature_c - 25.0) / 25.0,
            self._last_bess_kw / self.power_kw,
            self._flex_backlog_kwh / self.max_backlog_kwh,
            trailing_pcc / self.hard_pcc_limit_kw,
            headroom,
            ctx["reserve_required_kw"] / self.power_kw,
            self._step / max(1, self.episode_steps - 1),
        ]
        return np.clip(np.asarray(values, dtype=np.float32), -2.0, 2.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = len(self.segment) - self.episode_steps - 1
        requested = (options or {}).get("start_index")
        if requested is not None:
            self._start = max(0, min(int(requested), max_start))
        elif self.training:
            self._start = int(self._rng.integers(0, max_start + 1))
        else:
            self._start = 0
        self._step = 0
        self._soc = self.soc_initial
        self._soh = self.soh_initial
        first_ambient = float(self.segment[self._start, 6])
        self._temperature_c = max(24.0, min(32.0, first_ambient + 8.0))
        self._last_bess_kw = 0.0
        self._flex_backlog_kwh = 0.0
        self._pcc_history = []
        self.trace = []
        self._totals = {
            "reward": 0.0,
            "energy_cost_cny": 0.0,
            "carbon_kg": 0.0,
            "degradation_cost_cny": 0.0,
            "bess_throughput_kwh": 0.0,
            "aux_shift_kwh": 0.0,
            "reserve_shortfall_kwh": 0.0,
            "shore_sla_violation_kwh": 0.0,
            "guardrail_violations": 0.0,
            "projection_count": 0.0,
            "nonzero_bess_actions": 0.0,
            "nonzero_flex_actions": 0.0,
            "peak_kw": 0.0,
            "soc_min_observed": self._soc,
            "soc_max_observed": self._soc,
            "temperature_max_c": self._temperature_c,
        }
        return self._observation(), {"start_index": self._start, "split_timestamp": self.timestamps[self._start]}

    def _project(self, action: np.ndarray, ctx: Mapping[str, float]) -> Dict[str, float | bool | List[str]]:
        bounded = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if len(bounded) != 2:
            raise ValueError("Shore+BESS action must contain two continuous values")
        requested_bess = float(bounded[0]) * self.power_kw
        requested_flex = float(bounded[1]) * float(ctx["auxiliary_shore_kw"]) * self.flex_limit
        reasons: List[str] = []

        equipment = max(0.0, min(1.0, float(ctx["equipment_availability_ratio"])))
        available_power = self.power_kw * equipment * max(0.50, self._soh)
        if equipment < 0.5:
            available_power = 0.0
            reasons.append("equipment_interlock")
        if self._temperature_c >= self.temperature_trip_c:
            available_power = 0.0
            reasons.append("temperature_trip")
        elif self._temperature_c > self.temperature_derate_c:
            derate = (self.temperature_trip_c - self._temperature_c) / (
                self.temperature_trip_c - self.temperature_derate_c
            )
            available_power *= max(0.0, min(1.0, derate))
            reasons.append("temperature_derate")

        bess_kw = float(np.clip(requested_bess, -available_power, available_power))
        ramped = float(np.clip(bess_kw, self._last_bess_kw - self.ramp_kw, self._last_bess_kw + self.ramp_kw))
        if abs(ramped - bess_kw) > 1e-6:
            reasons.append("ramp")
        bess_kw = ramped

        available_discharge = max(0.0, (self._soc - self.soc_min) * self.energy_kwh * self.discharge_eff)
        available_charge = max(0.0, (self.soc_max - self._soc) * self.energy_kwh / self.charge_eff)
        reserve = float(ctx["reserve_required_kw"])
        discharge_limit = max(0.0, min(available_power, available_discharge) - reserve)
        charge_limit = min(available_power, available_charge)
        charge_limit = min(
            charge_limit,
            max(0.0, self.hard_pcc_limit_kw - float(ctx["base_load_kw"])),
        )
        limited = float(np.clip(bess_kw, -charge_limit, discharge_limit))
        if abs(limited - bess_kw) > 1e-6:
            reasons.append("soc_pcc_or_reserve")
        bess_kw = limited

        next_soc = self._soc - max(bess_kw, 0.0) / (self.energy_kwh * self.discharge_eff)
        next_soc += max(-bess_kw, 0.0) * self.charge_eff / self.energy_kwh
        remaining = self.episode_steps - self._step - 1
        physical_reachable_delta = (
            remaining * self.power_kw * min(self.charge_eff, self.discharge_eff) / self.energy_kwh
        )
        # A shrinking operational envelope prevents a policy from draining the
        # asset and then creating an artificial demand spike in the final hour.
        # It is deliberately tighter than the purely physical reachability set.
        operational_terminal_band = 0.18 * remaining / max(1, self.episode_steps - 1)
        reachable_delta = min(physical_reachable_delta, operational_terminal_band)
        reachable_low = max(self.soc_min, self.soc_initial - reachable_delta)
        reachable_high = min(self.soc_max, self.soc_initial + reachable_delta)
        reachable_soc = float(np.clip(next_soc, reachable_low, reachable_high))
        if abs(reachable_soc - next_soc) > 1e-9:
            reasons.append("terminal_soc_reachability")
            if reachable_soc <= self._soc:
                bess_kw = (self._soc - reachable_soc) * self.energy_kwh * self.discharge_eff
            else:
                bess_kw = -(reachable_soc - self._soc) * self.energy_kwh / self.charge_eff
            next_soc = reachable_soc

        flex_kw = requested_flex
        if flex_kw < 0:
            max_defer = max(0.0, self.max_backlog_kwh - self._flex_backlog_kwh)
            flex_kw = -min(-flex_kw, max_defer)
        else:
            flex_kw = min(flex_kw, self._flex_backlog_kwh)
        remaining = self.episode_steps - self._step - 1
        max_future_recovery = remaining * float(ctx["auxiliary_shore_kw"]) * self.flex_limit
        backlog_after = self._flex_backlog_kwh - flex_kw
        if backlog_after > max_future_recovery:
            flex_kw += backlog_after - max_future_recovery
            backlog_after = max_future_recovery
            reasons.append("flex_backlog_reachability")
        if remaining == 0 and abs(backlog_after) > 1e-6:
            flex_kw += backlog_after
            backlog_after = 0.0
            reasons.append("terminal_flex_recovery")

        net_kw = float(ctx["base_load_kw"]) - bess_kw + flex_kw
        if not self.export_allowed and net_kw < 0.0:
            bess_kw += net_kw
            net_kw = 0.0
            reasons.append("no_reverse_power")
        if net_kw > self.hard_pcc_limit_kw:
            required_discharge = net_kw - self.hard_pcc_limit_kw
            extra = min(required_discharge, max(0.0, discharge_limit - bess_kw))
            bess_kw += extra
            net_kw -= extra
            if net_kw > self.hard_pcc_limit_kw + 1e-6:
                reasons.append("upstream_load_exceeds_pcc_limit")
            else:
                reasons.append("pcc_hard_limit")

        # The no-export/PCC projections above may change the command after the
        # first SOC calculation. Reconcile the state with the *executed* power,
        # then re-apply the terminal reachability envelope. This keeps the BMS
        # state transition, audit trace and physical command identical even in
        # extreme site conditions that are absent from the public replay.
        executed_soc = self._soc - max(bess_kw, 0.0) / (self.energy_kwh * self.discharge_eff)
        executed_soc += max(-bess_kw, 0.0) * self.charge_eff / self.energy_kwh
        reconciled_soc = float(np.clip(executed_soc, reachable_low, reachable_high))
        if abs(reconciled_soc - executed_soc) > 1e-9:
            if "terminal_soc_reachability" not in reasons:
                reasons.append("terminal_soc_reachability")
            if reconciled_soc <= self._soc:
                bess_kw = (self._soc - reconciled_soc) * self.energy_kwh * self.discharge_eff
            else:
                bess_kw = -(reconciled_soc - self._soc) * self.energy_kwh / self.charge_eff
            net_kw = float(ctx["base_load_kw"]) - bess_kw + flex_kw
        next_soc = reconciled_soc

        if net_kw > self.hard_pcc_limit_kw + 1e-6 and "upstream_load_exceeds_pcc_limit" not in reasons:
            reasons.append("upstream_load_exceeds_pcc_limit")

        return {
            "requested_bess_kw": requested_bess,
            "bess_kw": bess_kw,
            "requested_flex_kw": requested_flex,
            "flex_kw": flex_kw,
            "next_soc": next_soc,
            "backlog_after_kwh": backlog_after,
            "net_kw": net_kw,
            "projection_applied": bool(reasons),
            "projection_reasons": reasons,
        }

    def step(self, action: Any):
        ctx = self._row_context()
        projected = self._project(np.asarray(action, dtype=np.float32), ctx)
        bess_kw = float(projected["bess_kw"])
        flex_kw = float(projected["flex_kw"])
        net_kw = float(projected["net_kw"])
        self._soc = float(projected["next_soc"])
        self._flex_backlog_kwh = float(projected["backlog_after_kwh"])
        throughput_kwh = abs(bess_kw)
        self._soh = max(0.75, self._soh - throughput_kwh / (2.0 * self.energy_kwh * 6000.0))
        thermal_load = 1.05 * abs(bess_kw) / self.power_kw
        passive_cooling = 0.10 * max(0.0, self._temperature_c - max(22.0, float(ctx["ambient_c"])))
        self._temperature_c = max(
            float(ctx["ambient_c"]),
            self._temperature_c + thermal_load - passive_cooling,
        )
        self._last_bess_kw = bess_kw
        self._pcc_history.append(net_kw)

        price = float(ctx["price_cny_per_kwh"])
        carbon_factor = float(ctx["carbon_kg_per_kwh"])
        energy_cost = net_kw * price
        carbon_kg = net_kw * carbon_factor
        degradation_cost = throughput_kwh * self.cycle_cost
        peak_excess = max(0.0, net_kw - self.soft_cap_kw)
        available_reserve = max(0.0, min(self.power_kw, (self._soc - self.soc_min) * self.energy_kwh * self.discharge_eff) - max(0.0, bess_kw))
        reserve_shortfall = max(0.0, float(ctx["reserve_required_kw"]) - available_reserve)
        shore_sla_shortfall = 0.0  # mandatory shore power is not an action dimension
        guardrail_violation = bool(
            self._soc < self.soc_min - 1e-7
            or self._soc > self.soc_max + 1e-7
            or net_kw < -1e-6
            or net_kw > self.hard_pcc_limit_kw + 1e-6
            or shore_sla_shortfall > 0
        )
        projection_distance = abs(float(projected["requested_bess_kw"]) - bess_kw) / self.power_kw
        projection_distance += abs(float(projected["requested_flex_kw"]) - flex_kw) / max(1.0, float(ctx["auxiliary_shore_kw"]))
        terminal = self._step + 1 >= self.episode_steps
        terminal_state = (
            abs(self._soc - self.soc_initial) / max(1e-6, self.soc_max - self.soc_min)
            + self._flex_backlog_kwh / self.max_backlog_kwh
            if terminal
            else 0.0
        )
        baseline_energy_cost = float(ctx["base_load_kw"]) * price
        baseline_carbon_kg = float(ctx["base_load_kw"]) * carbon_factor
        baseline_peak_excess = max(0.0, float(ctx["base_load_kw"]) - self.soft_cap_kw)
        # Optimize the controllable delta, not the fixed 20+ MW port base load.
        # This keeps the learning signal tied to what the two actions can change.
        normalizers = {
            "energy_cost": (energy_cost - baseline_energy_cost) / (self.power_kw * self.price_scale),
            "carbon": (carbon_kg - baseline_carbon_kg) / (self.power_kw * self.carbon_scale),
            "demand_peak": (peak_excess - baseline_peak_excess) / self.power_kw,
            "degradation": degradation_cost / max(1.0, self.power_kw * self.cycle_cost),
            "reserve": reserve_shortfall / max(1.0, float(ctx["reserve_required_kw"])),
            "shore_sla": shore_sla_shortfall / max(1.0, float(ctx["mandatory_shore_kw"])),
            "safety_projection": projection_distance + float(guardrail_violation),
            "terminal_state": terminal_state,
        }
        reward = -sum(self.weights[name] * normalizers[name] for name in self.weights)

        self._totals["reward"] += reward
        self._totals["energy_cost_cny"] += energy_cost
        self._totals["carbon_kg"] += carbon_kg
        self._totals["degradation_cost_cny"] += degradation_cost
        self._totals["bess_throughput_kwh"] += throughput_kwh
        self._totals["aux_shift_kwh"] += abs(flex_kw)
        self._totals["reserve_shortfall_kwh"] += reserve_shortfall
        self._totals["shore_sla_violation_kwh"] += shore_sla_shortfall
        self._totals["guardrail_violations"] += float(guardrail_violation)
        self._totals["projection_count"] += float(projected["projection_applied"])
        self._totals["nonzero_bess_actions"] += float(abs(bess_kw) > 1.0)
        self._totals["nonzero_flex_actions"] += float(abs(flex_kw) > 1.0)
        self._totals["peak_kw"] = max(self._totals["peak_kw"], net_kw)
        self._totals["soc_min_observed"] = min(self._totals["soc_min_observed"], self._soc)
        self._totals["soc_max_observed"] = max(self._totals["soc_max_observed"], self._soc)
        self._totals["temperature_max_c"] = max(self._totals["temperature_max_c"], self._temperature_c)

        info = {
            "timestamp": self.timestamps[self._start + self._step],
            "observation_contract": "shore_bess_v3_34x2",
            "context": ctx,
            "requested_action": {
                "bess_kw": float(projected["requested_bess_kw"]),
                "flex_kw": float(projected["requested_flex_kw"]),
            },
            "final_action": {"bess_kw": bess_kw, "flex_kw": flex_kw},
            "projection": {
                "applied": bool(projected["projection_applied"]),
                "reasons": list(projected["projection_reasons"]),
            },
            "soc": self._soc,
            "soh": self._soh,
            "temperature_c": self._temperature_c,
            "pcc_kw": net_kw,
            "reserve_shortfall_kw": reserve_shortfall,
            "shore_sla_shortfall_kw": shore_sla_shortfall,
            "guardrail_violation": guardrail_violation,
            "reward_components": normalizers,
            "business_step": {
                "energy_cost_cny": energy_cost,
                "carbon_kg": carbon_kg,
                "degradation_cost_cny": degradation_cost,
                "peak_excess_kw": peak_excess,
            },
        }
        if self.record_trace:
            self.trace.append(dict(info, reward=reward))

        self._step += 1
        terminated = self._step >= self.episode_steps
        if terminated:
            demand_rate = float(self.config["grid"]["demand_charge_cny_per_kw_month"])
            weekly_demand_charge = self._totals["peak_kw"] * demand_rate * self.episode_steps / (24.0 * 30.4375)
            self._totals["demand_charge_cny"] = weekly_demand_charge
            self._totals["total_cost_cny"] = (
                self._totals["energy_cost_cny"]
                + self._totals["degradation_cost_cny"]
                + weekly_demand_charge
            )
            self._totals["guardrail_violation_rate"] = self._totals["guardrail_violations"] / self.episode_steps
            self._totals["projection_rate"] = self._totals["projection_count"] / self.episode_steps
            self._totals["nonzero_bess_action_rate"] = self._totals["nonzero_bess_actions"] / self.episode_steps
            self._totals["nonzero_flex_action_rate"] = self._totals["nonzero_flex_actions"] / self.episode_steps
            self._totals["terminal_soc_error"] = abs(self._soc - self.soc_initial)
            self._totals["terminal_flex_backlog_kwh"] = self._flex_backlog_kwh
            info["episode_metrics"] = dict(self._totals)
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._observation()
        return observation, float(reward), terminated, False, info

    @property
    def totals(self) -> Dict[str, float]:
        return dict(self._totals)

    def render(self):
        self.render_calls += 1
        raise RuntimeError("rendering is disabled during Shore+BESS training; use trace-enabled blind evaluation")


class ShoreBESSDiscreteEnv(ShoreBESSEnv):
    """DQN-compatible finite lattice over the same projected EMS controls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTION_LATTICE))

    def step(self, action: Any):
        index = int(np.asarray(action).item())
        if not 0 <= index < len(DISCRETE_ACTION_LATTICE):
            raise ValueError("discrete Shore+BESS action index is outside the registered lattice")
        return super().step(DISCRETE_ACTION_LATTICE[index])


def fixed_window_starts(length: int, episode_steps: int, count: int) -> List[int]:
    max_start = max(0, length - episode_steps - 1)
    if count <= 1 or max_start == 0:
        return [0]
    return sorted(set(int(round(value)) for value in np.linspace(0, max_start, count)))


def neutral_policy(_observation: np.ndarray, _env: ShoreBESSEnv) -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def rule_peak_valley_policy(_observation: np.ndarray, env: ShoreBESSEnv) -> np.ndarray:
    ctx = env._row_context()
    price = float(ctx["price_cny_per_kwh"])
    base = float(ctx["base_load_kw"])
    if base > env.soft_cap_kw + 500.0 and env._soc > env.soc_initial - 0.12:
        bess = min(0.55, max(0.0, (base - env.soft_cap_kw) / env.power_kw))
    elif price >= 0.95 and env._soc > env.soc_initial - 0.10:
        bess = 0.55
    elif (
        price <= 0.50
        and env._soc < env.soc_initial + 0.10
        and base + 500.0 < env.soft_cap_kw
    ):
        bess = -0.70
    else:
        bess = 0.0
    # The public benchmark has no measured flexible shore auxiliary tags, so
    # this baseline leaves the second action neutral instead of inventing load.
    flex = 0.0
    return np.asarray([bess, flex], dtype=np.float32)


def evaluate_windows(
    env_factory: Callable[[], ShoreBESSEnv],
    policy: Callable[[np.ndarray, ShoreBESSEnv], np.ndarray],
    starts: Iterable[int],
) -> Dict[str, Any]:
    episodes: List[Dict[str, float]] = []
    sample: Optional[Dict[str, Any]] = None
    for start in starts:
        env = env_factory()
        observation, _ = env.reset(options={"start_index": int(start)})
        done = False
        while not done:
            action = np.asarray(policy(observation, env), dtype=np.float32)
            observation, _reward, terminated, truncated, info = env.step(action)
            if sample is None:
                sample = {
                    "timestamp": info["timestamp"],
                    "state": info["context"],
                    "requested_action": info["requested_action"],
                    "final_action": info["final_action"],
                    "projection": info["projection"],
                    "model_output_derived_business": info["business_step"],
                }
            done = bool(terminated or truncated)
        episodes.append(env.totals)
        env.close()
    numeric_keys = sorted(
        set.intersection(
            *(set(row) for row in episodes)
        )
    ) if episodes else []
    mean = {
        key: float(np.mean([float(row[key]) for row in episodes]))
        for key in numeric_keys
        if all(isinstance(row[key], (int, float)) for row in episodes)
    }
    std = {
        key: float(np.std([float(row[key]) for row in episodes]))
        for key in mean
    }
    return {"episodes": episodes, "mean": mean, "std": std, "sample_inference": sample}


__all__ = [
    "ACTION_NAMES",
    "CONTRACT",
    "DEFAULT_CONFIG_PATH",
    "DISCRETE_ACTION_LATTICE",
    "STATE_NAMES",
    "ShoreBESSEnv",
    "ShoreBESSDiscreteEnv",
    "chronological_slices",
    "evaluate_windows",
    "fixed_window_starts",
    "load_config",
    "load_public_dataset",
    "neutral_policy",
    "rule_peak_valley_policy",
]
