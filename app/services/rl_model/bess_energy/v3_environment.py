"""Landing-oriented site BESS environment for the append-only V3.1 track.

The historical 2,000-step SAC run and its 8,927 transitions remain untouched.
This environment uses the canonical public Shanghai hourly time axis and adds
an explicitly labelled engineering calendar for reserve/DR coverage where no
public site event ledger exists. Positive BESS power discharges into the PCC;
negative power charges. Training never renders or writes control commands.
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
DEFAULT_CONFIG_PATH = ROOT / "config" / "bess_energy_v3.json"

STATE_NAMES = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "base_load_ratio", "load_forecast_1h_ratio", "load_forecast_6h_ratio", "load_ramp_ratio",
    "price_ratio", "known_tariff_6h_ratio", "carbon_factor_ratio", "carbon_forecast_6h_ratio",
    "throughput_ratio", "vessel_arrivals_ratio", "ambient_temperature_ratio", "wind_speed_ratio",
    "wave_height_ratio", "current_speed_ratio", "berth_occupancy_ratio", "yard_occupancy_ratio",
    "crane_availability_ratio", "equipment_availability_ratio", "channel_congestion_ratio",
    "reefer_load_ratio", "closure_flag", "visibility_available", "solar_proxy_ratio",
    "dr_event_active", "dr_target_ratio", "reserve_requirement_ratio", "reserve_price_ratio",
    "soc_scaled", "soh_scaled", "battery_temperature_scaled", "last_bess_power_ratio",
    "last_reserve_commitment_ratio", "trailing_pcc_ratio", "soft_headroom_ratio",
    "hard_headroom_ratio", "episode_progress",
]

ACTION_NAMES = ["bess_dispatch_ratio", "upward_reserve_commitment_ratio"]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("control_authority") != "recommendation_only":
        raise ValueError("open-source BESS control authority must remain recommendation_only")
    required = {
        "energy_cost", "demand_peak", "carbon", "degradation", "reserve_service",
        "dr_performance", "safety_projection", "thermal_health", "terminal_soc",
    }
    weights = payload.get("reward_weights") or {}
    if set(weights) != required or sum(float(value) for value in weights.values()) <= 0:
        raise ValueError("bess_energy_v3 reward weights are incomplete")
    return payload


def load_public_dataset(config: Optional[Mapping[str, Any]] = None) -> PortDataset:
    resolved = dict(config or load_config())
    return load_port_dataset(str(resolved["dataset_id"]))


def chronological_slices(dataset: PortDataset) -> tuple[slice, slice, slice]:
    return dataset.split_three_way(test_ratio=0.20, validation_ratio=0.10)


@dataclass(frozen=True)
class BESSEnergyContract:
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


CONTRACT = BESSEnergyContract(
    state_names=STATE_NAMES,
    action_names=ACTION_NAMES,
    reward_components=[
        "energy_cost", "demand_peak", "carbon", "degradation", "reserve_service",
        "dr_performance", "safety_projection", "thermal_health", "terminal_soc",
    ],
    hard_constraints=[
        "soc_min_max", "pcs_power_and_c_rate", "hourly_ramp", "no_simultaneous_charge_discharge",
        "no_reverse_power", "pcc_hard_limit", "demand_charge_no_new_peak", "n_minus_1_grid_margin", "reserve_dispatch_cooptimization",
        "event_deliverability", "temperature_derate_and_trip", "soh_power_derate",
        "equipment_and_bms_interlock", "terminal_soc_reachability", "fault_or_missing_ack_fail_closed",
    ],
    landing_inputs=[
        "pcs_active_power_and_available_capacity", "bms_soc_soh_temperature_alarm", "pcc_active_power_and_15min_demand",
        "settlement_energy_and_demand_tariff", "reserve_and_dr_clearing_event", "reserve_performance_baseline",
        "load_and_renewable_forecast", "marginal_grid_carbon_factor", "battery_efficiency_and_degradation_curve",
        "transformer_n_minus_1_limit", "equipment_availability_and_interlocks", "clock_quality_and_data_freshness",
        "gateway_ack_nonce_ttl_operator_and_rollback_authority",
    ],
)


class BESSEnergyV3Env(gym.Env):
    """40-state, two-action CMDP simulator with one shared safety projection."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: PortDataset,
        data_slice: slice,
        *,
        config: Optional[Mapping[str, Any]] = None,
        normalization_slice: Optional[slice] = None,
        episode_steps: Optional[int] = None,
        seed: int = 47,
        training: bool = True,
        record_trace: bool = False,
    ) -> None:
        super().__init__()
        self.config = dict(config or load_config())
        self.dataset = dataset
        self.segment = dataset.values[data_slice].astype(np.float32, copy=True)
        self.factor_values = dataset.factor_values[data_slice].astype(np.float32, copy=True)
        self.factor_masks = dataset.factor_availability[data_slice].astype(np.float32, copy=True)
        self.timestamps = list(dataset.timestamps[data_slice])
        if len(self.segment) < 169:
            raise ValueError("BESS split requires at least 169 chronological rows")
        self.training = bool(training)
        self.record_trace = bool(record_trace)
        if self.training and self.record_trace:
            raise ValueError("training must not render or record replay traces")
        self.episode_steps = max(24, min(int(episode_steps or self.config["training"]["episode_hours"]), len(self.segment) - 1))
        self._rng = np.random.default_rng(seed)

        asset = self.config["asset"]
        grid = self.config["grid"]
        self.energy_kwh = float(asset["rated_energy_kwh"])
        self.power_kw = min(float(asset["rated_power_kw"]), float(asset["c_rate_max"]) * self.energy_kwh)
        self.charge_eff = float(asset["charge_efficiency"])
        self.discharge_eff = float(asset["discharge_efficiency"])
        self.soc_min = float(asset["soc_min"])
        self.soc_max = float(asset["soc_max"])
        self.soc_initial = float(asset["soc_initial"])
        self.terminal_soc_tolerance = float(asset["terminal_soc_tolerance"])
        self.soh_initial = float(asset["soh_initial"])
        self.ramp_kw = float(asset["ramp_kw_per_hour"])
        self.cycle_cost = float(asset["cycle_cost_cny_per_kwh"])
        self.temperature_derate_c = float(asset["temperature_derate_c"])
        self.temperature_trip_c = float(asset["temperature_trip_c"])
        self.hard_pcc_limit_kw = float(grid["hard_pcc_limit_kw"])
        self.n_minus_1_margin_kw = float(grid["n_minus_1_margin_kw"])
        self.export_allowed = bool(grid["export_allowed"])
        self.demand_rate = float(grid["demand_charge_cny_per_kw_month"]) * 12.0 / (365.25 / 7.0)
        self.weights = {name: float(value) for name, value in self.config["reward_weights"].items()}
        total_weight = sum(self.weights.values())
        self.weights = {name: value / total_weight for name, value in self.weights.items()}

        train_slice = normalization_slice or chronological_slices(dataset)[0]
        train = dataset.values[train_slice].astype(np.float64, copy=False)
        self.base_scale = float(max(1.0, np.quantile(train[:, 0], 0.99)))
        self.throughput_scale = float(max(1.0, np.quantile(train[:, 1], 0.99)))
        self.arrival_scale = float(max(1.0, np.quantile(train[:, 2], 0.99)))
        self.price_scale = float(max(1e-6, np.max(train[:, 4])))
        self.carbon_scale = float(max(1e-6, np.max(train[:, 5])))
        self.soft_cap_kw = float(np.quantile(train[:, 0], float(grid["soft_cap_train_quantile"])))
        self.factor_index = {name: index for index, name in enumerate(FACTOR_COLUMNS)}
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(len(STATE_NAMES),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(ACTION_NAMES),), dtype=np.float32)
        self.render_calls = 0
        self.trace: List[Dict[str, Any]] = []
        self._start = self._step = 0
        self._soc = self.soc_initial
        self._soh = self.soh_initial
        self._temperature_c = 28.0
        self._last_bess_kw = self._last_reserve_kw = 0.0
        self._pcc_history: List[float] = []
        self._totals: Dict[str, float] = {}

    def _factor(self, index: int, name: str, default: float = 0.0) -> float:
        column = self.factor_index[name]
        return float(self.factor_values[index, column]) if self.factor_masks[index, column] > 0.5 else float(default)

    def _mean_future(self, index: int, column: int, hours: int) -> float:
        values = self.segment[index : min(len(self.segment), index + hours), column]
        return float(np.mean(values)) if len(values) else float(self.segment[index, column])

    def _context(self) -> Dict[str, float]:
        index = self._start + self._step
        row = self.segment[index]
        ts = datetime.fromisoformat(self.timestamps[index].replace("Z", "+00:00"))
        services = self.config["services"]
        critical = ts.hour in {int(value) for value in services["critical_hours_local"]}
        event = ts.weekday() in {int(value) for value in services["dr_event_weekdays"]} and ts.hour in {
            int(value) for value in services["dr_event_hours_local"]
        }
        congestion = self._factor(index, "channel_congestion_ratio", 0.0)
        event_target = (
            float(services["dr_target_min_kw"])
            + congestion * (float(services["dr_target_max_kw"]) - float(services["dr_target_min_kw"]))
            if event else 0.0
        )
        reserve_required = float(services["reserve_critical_kw"] if critical else services["reserve_min_kw"])
        if event:
            reserve_required = max(reserve_required, event_target)
        daylight = max(0.0, math.sin(math.pi * (ts.hour - 6.0) / 12.0))
        weather_derate = max(0.1, 1.0 - min(0.8, self._factor(index, "wave_height_m", 0.0) / 5.0))
        return {
            "index": float(index), "hour": float(ts.hour), "weekday": float(ts.weekday()),
            "base_load_kw": float(row[0]), "load_forecast_1h_kw": self._mean_future(index, 0, 2),
            "load_forecast_6h_kw": self._mean_future(index, 0, 6),
            "load_ramp_kw": float(self.segment[min(len(self.segment) - 1, index + 1), 0] - row[0]),
            "price": float(row[4]), "price_6h": self._mean_future(index, 4, 6),
            "carbon": float(row[5]), "carbon_6h": self._mean_future(index, 5, 6),
            "throughput": float(row[1]), "arrivals": float(row[2]), "ambient_c": float(row[6]),
            "wind": self._factor(index, "wind_speed_mps"), "wave": self._factor(index, "wave_height_m"),
            "current": self._factor(index, "current_speed_mps"), "berth": self._factor(index, "berth_occupancy_ratio"),
            "yard": self._factor(index, "yard_occupancy_ratio"), "crane": self._factor(index, "crane_availability_ratio"),
            "equipment": self._factor(index, "equipment_availability_ratio"), "congestion": congestion,
            "reefer_kw": self._factor(index, "reefer_load_kw"), "closure": self._factor(index, "closure_flag"),
            "visibility_available": float(self.factor_masks[index, self.factor_index["visibility_km"]] > 0.5),
            "solar_proxy": daylight * weather_derate, "dr_event_active": float(event),
            "dr_target_kw": event_target, "reserve_required_kw": reserve_required,
            "reserve_price": float(services["reserve_capacity_price_cny_per_kw_h"]) * (1.5 if critical else 1.0),
        }

    def _observation(self) -> np.ndarray:
        ctx = self._context()
        hour_angle = 2.0 * math.pi * ctx["hour"] / 24.0
        weekday_angle = 2.0 * math.pi * ctx["weekday"] / 7.0
        trailing = float(np.mean(self._pcc_history[-3:])) if self._pcc_history else ctx["base_load_kw"]
        values = [
            math.sin(hour_angle), math.cos(hour_angle), math.sin(weekday_angle), math.cos(weekday_angle),
            ctx["base_load_kw"] / self.base_scale, ctx["load_forecast_1h_kw"] / self.base_scale,
            ctx["load_forecast_6h_kw"] / self.base_scale, ctx["load_ramp_kw"] / self.base_scale,
            ctx["price"] / self.price_scale, ctx["price_6h"] / self.price_scale,
            ctx["carbon"] / self.carbon_scale, ctx["carbon_6h"] / self.carbon_scale,
            ctx["throughput"] / self.throughput_scale, ctx["arrivals"] / self.arrival_scale,
            ctx["ambient_c"] / 45.0, ctx["wind"] / 25.0, ctx["wave"] / 4.0, ctx["current"] / 4.0,
            ctx["berth"], ctx["yard"], ctx["crane"], ctx["equipment"], ctx["congestion"],
            ctx["reefer_kw"] / self.base_scale, ctx["closure"], ctx["visibility_available"], ctx["solar_proxy"],
            ctx["dr_event_active"], ctx["dr_target_kw"] / self.power_kw, ctx["reserve_required_kw"] / self.power_kw,
            ctx["reserve_price"] / 0.1, 2.0 * (self._soc - self.soc_min) / (self.soc_max - self.soc_min) - 1.0,
            2.0 * self._soh - 1.0, (self._temperature_c - 25.0) / 25.0,
            self._last_bess_kw / self.power_kw, self._last_reserve_kw / self.power_kw,
            trailing / self.hard_pcc_limit_kw, (self.soft_cap_kw - ctx["base_load_kw"]) / self.hard_pcc_limit_kw,
            (self.hard_pcc_limit_kw - ctx["base_load_kw"]) / self.hard_pcc_limit_kw,
            self._step / max(1, self.episode_steps - 1),
        ]
        return np.clip(np.asarray(values, dtype=np.float32), -2.0, 2.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = len(self.segment) - self.episode_steps - 1
        requested = (options or {}).get("start_index")
        self._start = max(0, min(int(requested), max_start)) if requested is not None else (
            int(self._rng.integers(0, max_start + 1)) if self.training else 0
        )
        self._step = 0
        self._soc, self._soh = self.soc_initial, self.soh_initial
        self._temperature_c = max(24.0, min(32.0, float(self.segment[self._start, 6]) + 8.0))
        self._last_bess_kw = self._last_reserve_kw = 0.0
        self._pcc_history, self.trace = [], []
        self._totals = {
            "reward": 0.0, "energy_cost_cny": 0.0, "carbon_kg": 0.0, "degradation_cost_cny": 0.0,
            "reserve_revenue_cny": 0.0, "dr_revenue_cny": 0.0, "bess_throughput_kwh": 0.0,
            "event_hours": 0.0, "event_shortfall_kwh": 0.0, "reserve_commitment_kwh": 0.0,
            "guardrail_violations": 0.0, "projection_count": 0.0, "nonzero_dispatch_actions": 0.0,
            "nonzero_reserve_actions": 0.0, "peak_kw": 0.0, "soc_min_observed": self._soc,
            "soc_max_observed": self._soc, "temperature_max_c": self._temperature_c,
        }
        return self._observation(), {"start_index": self._start, "split_timestamp": self.timestamps[self._start]}

    def _project(self, action: np.ndarray, ctx: Mapping[str, float]) -> Dict[str, Any]:
        bounded = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if len(bounded) != 2:
            raise ValueError("BESS action must contain dispatch and reserve commitment")
        requested_power = float(bounded[0]) * self.power_kw
        requested_reserve = max(0.0, float(bounded[1])) * self.power_kw
        reasons: List[str] = []
        equipment = max(0.0, min(1.0, float(ctx["equipment"])))
        available_power = self.power_kw * equipment * max(0.50, self._soh)
        if equipment < 0.5:
            available_power = 0.0
            reasons.append("equipment_bms_interlock")
        if self._temperature_c >= self.temperature_trip_c:
            available_power = 0.0
            reasons.append("temperature_trip")
        elif self._temperature_c > self.temperature_derate_c:
            available_power *= max(0.0, (self.temperature_trip_c - self._temperature_c) / (self.temperature_trip_c - self.temperature_derate_c))
            reasons.append("temperature_derate")

        power = float(np.clip(requested_power, -available_power, available_power))
        ramped = float(np.clip(power, self._last_bess_kw - self.ramp_kw, self._last_bess_kw + self.ramp_kw))
        if abs(ramped - power) > 1e-6:
            reasons.append("ramp")
        power = ramped
        max_discharge = min(available_power, max(0.0, (self._soc - self.soc_min) * self.energy_kwh * self.discharge_eff))
        max_charge = min(available_power, max(0.0, (self.soc_max - self._soc) * self.energy_kwh / self.charge_eff))
        max_charge = min(max_charge, max(0.0, self.hard_pcc_limit_kw - float(ctx["base_load_kw"])))
        # Charging is an economic action, never a safety necessity. Do not let
        # actor approximation error create a new demand peak above the
        # validation-derived operational cap.
        max_charge = min(max_charge, max(0.0, self.soft_cap_kw - float(ctx["base_load_kw"])))
        clipped = float(np.clip(power, -max_charge, max_discharge))
        if abs(clipped - power) > 1e-6:
            reasons.append("soc_power_or_pcc")
        power = clipped

        required_reserve = float(ctx["reserve_required_kw"])
        reserve = max(requested_reserve, required_reserve)
        if reserve > requested_reserve + 1e-6:
            reasons.append("reserve_requirement")
        reserve_cap = max(0.0, min(available_power - max(0.0, power), max_discharge - max(0.0, power)))
        reserve = min(reserve, reserve_cap)
        if power > max(0.0, available_power - reserve):
            power = max(0.0, available_power - reserve)
            reasons.append("dispatch_reserve_cooptimization")

        net_kw = float(ctx["base_load_kw"]) - power
        if not self.export_allowed and net_kw < 0.0:
            power += net_kw
            net_kw = 0.0
            reasons.append("no_reverse_power")
        hard_operating_limit = self.hard_pcc_limit_kw - self.n_minus_1_margin_kw
        if net_kw > hard_operating_limit:
            extra = min(net_kw - hard_operating_limit, max(0.0, max_discharge - reserve - power))
            power += extra
            net_kw -= extra
            reasons.append("n_minus_1_pcc_limit")

        remaining = self.episode_steps - self._step - 1
        physical_delta = remaining * self.power_kw * min(self.charge_eff, self.discharge_eff) / self.energy_kwh
        operational_delta = 0.20 * remaining / max(1, self.episode_steps - 1)
        reachable_delta = max(self.terminal_soc_tolerance, min(physical_delta, operational_delta))
        reachable_low = max(self.soc_min, self.soc_initial - reachable_delta)
        reachable_high = min(self.soc_max, self.soc_initial + reachable_delta)
        next_soc = self._soc - max(power, 0.0) / (self.energy_kwh * self.discharge_eff)
        next_soc += max(-power, 0.0) * self.charge_eff / self.energy_kwh
        reconciled = float(np.clip(next_soc, reachable_low, reachable_high))
        if abs(reconciled - next_soc) > 1e-9:
            reasons.append("terminal_soc_reachability")
            power = ((self._soc - reconciled) * self.energy_kwh * self.discharge_eff) if reconciled <= self._soc else (
                -(reconciled - self._soc) * self.energy_kwh / self.charge_eff
            )
            net_kw = float(ctx["base_load_kw"]) - power
        next_soc = reconciled
        if power < 0.0 and net_kw > self.soft_cap_kw:
            power = -max(0.0, self.soft_cap_kw - float(ctx["base_load_kw"]))
            next_soc = self._soc + max(-power, 0.0) * self.charge_eff / self.energy_kwh
            net_kw = float(ctx["base_load_kw"]) - power
            reasons.append("demand_charge_no_new_peak")
        reserve_cap = max(0.0, min(available_power - max(0.0, power), max_discharge - max(0.0, power)))
        reserve = min(reserve, reserve_cap)
        event_shortfall = max(0.0, float(ctx["dr_target_kw"]) - reserve) if ctx["dr_event_active"] else 0.0
        if event_shortfall > 1e-6:
            reasons.append("event_capacity_unavailable")
        return {
            "requested_power_kw": requested_power, "requested_reserve_kw": requested_reserve,
            "power_kw": power, "reserve_kw": reserve, "next_soc": next_soc, "net_kw": net_kw,
            "event_shortfall_kw": event_shortfall, "projection_applied": bool(reasons), "projection_reasons": reasons,
        }

    def step(self, action: Any):
        ctx = self._context()
        projected = self._project(np.asarray(action, dtype=np.float32), ctx)
        power, reserve, net_kw = float(projected["power_kw"]), float(projected["reserve_kw"]), float(projected["net_kw"])
        self._soc = float(projected["next_soc"])
        throughput = abs(power)
        self._soh = max(0.75, self._soh - throughput / (2.0 * self.energy_kwh * 6000.0))
        thermal_load = 1.10 * throughput / self.power_kw
        cooling = 0.10 * max(0.0, self._temperature_c - max(22.0, float(ctx["ambient_c"])))
        self._temperature_c = max(float(ctx["ambient_c"]), self._temperature_c + thermal_load - cooling)
        self._last_bess_kw, self._last_reserve_kw = power, reserve
        self._pcc_history.append(net_kw)

        energy_cost = net_kw * float(ctx["price"])
        carbon_kg = net_kw * float(ctx["carbon"])
        degradation = throughput * self.cycle_cost
        reserve_revenue = reserve * float(ctx["reserve_price"])
        delivered_event = max(0.0, float(ctx["dr_target_kw"]) - float(projected["event_shortfall_kw"]))
        dr_revenue = delivered_event * float(self.config["services"]["dr_performance_price_cny_per_kwh"]) if ctx["dr_event_active"] else 0.0
        peak_excess = max(0.0, net_kw - self.soft_cap_kw)
        baseline_energy = float(ctx["base_load_kw"]) * float(ctx["price"])
        baseline_carbon = float(ctx["base_load_kw"]) * float(ctx["carbon"])
        baseline_peak = max(0.0, float(ctx["base_load_kw"]) - self.soft_cap_kw)
        guardrail = bool(
            self._soc < self.soc_min - 1e-7 or self._soc > self.soc_max + 1e-7 or net_kw < -1e-6
            or net_kw > self.hard_pcc_limit_kw - self.n_minus_1_margin_kw + 1e-6
            or float(projected["event_shortfall_kw"]) > 1e-6
        )
        projection_distance = abs(float(projected["requested_power_kw"]) - power) / self.power_kw
        projection_distance += abs(float(projected["requested_reserve_kw"]) - reserve) / self.power_kw
        terminal = self._step + 1 >= self.episode_steps
        terminal_error = max(0.0, abs(self._soc - self.soc_initial) - self.terminal_soc_tolerance) / max(1e-6, self.soc_max - self.soc_min) if terminal else 0.0
        components = {
            "energy_cost": (energy_cost - baseline_energy) / (self.power_kw * self.price_scale),
            "demand_peak": (peak_excess - baseline_peak) / self.power_kw,
            "carbon": (carbon_kg - baseline_carbon) / (self.power_kw * self.carbon_scale),
            "degradation": degradation / max(1.0, self.power_kw * self.cycle_cost),
            "reserve_service": -reserve_revenue / max(1.0, self.power_kw * 0.1),
            "dr_performance": (float(projected["event_shortfall_kw"]) - delivered_event) / self.power_kw,
            "safety_projection": projection_distance + float(guardrail),
            "thermal_health": max(0.0, self._temperature_c - self.temperature_derate_c) / 10.0 + max(0.0, 0.90 - self._soh),
            "terminal_soc": terminal_error,
        }
        reward = -sum(self.weights[name] * components[name] for name in self.weights)
        self._totals["reward"] += reward
        self._totals["energy_cost_cny"] += energy_cost
        self._totals["carbon_kg"] += carbon_kg
        self._totals["degradation_cost_cny"] += degradation
        self._totals["reserve_revenue_cny"] += reserve_revenue
        self._totals["dr_revenue_cny"] += dr_revenue
        self._totals["bess_throughput_kwh"] += throughput
        self._totals["event_hours"] += float(ctx["dr_event_active"])
        self._totals["event_shortfall_kwh"] += float(projected["event_shortfall_kw"])
        self._totals["reserve_commitment_kwh"] += reserve
        self._totals["guardrail_violations"] += float(guardrail)
        self._totals["projection_count"] += float(projected["projection_applied"])
        self._totals["nonzero_dispatch_actions"] += float(abs(power) > 1.0)
        self._totals["nonzero_reserve_actions"] += float(reserve > 1.0)
        self._totals["peak_kw"] = max(self._totals["peak_kw"], net_kw)
        self._totals["soc_min_observed"] = min(self._totals["soc_min_observed"], self._soc)
        self._totals["soc_max_observed"] = max(self._totals["soc_max_observed"], self._soc)
        self._totals["temperature_max_c"] = max(self._totals["temperature_max_c"], self._temperature_c)
        info = {
            "timestamp": self.timestamps[self._start + self._step], "context": ctx,
            "requested_action": {"bess_kw": projected["requested_power_kw"], "reserve_kw": projected["requested_reserve_kw"]},
            "final_action": {"bess_kw": power, "reserve_kw": reserve},
            "projection": {"applied": projected["projection_applied"], "reasons": projected["projection_reasons"]},
            "soc": self._soc, "soh": self._soh, "temperature_c": self._temperature_c, "pcc_kw": net_kw,
            "event_shortfall_kw": projected["event_shortfall_kw"], "guardrail_violation": guardrail,
            "reward_components": components,
            "business_step": {"energy_cost_cny": energy_cost, "carbon_kg": carbon_kg, "degradation_cost_cny": degradation,
                              "reserve_revenue_cny": reserve_revenue, "dr_revenue_cny": dr_revenue, "peak_excess_kw": peak_excess},
        }
        if self.record_trace:
            self.trace.append(dict(info, reward=reward))
        self._step += 1
        terminated = self._step >= self.episode_steps
        if terminated:
            demand_charge = self._totals["peak_kw"] * self.demand_rate
            self._totals.update({
                "demand_charge_cny": demand_charge,
                "total_cost_cny": self._totals["energy_cost_cny"] + self._totals["degradation_cost_cny"] + demand_charge
                - self._totals["reserve_revenue_cny"] - self._totals["dr_revenue_cny"],
                "guardrail_violation_rate": self._totals["guardrail_violations"] / self.episode_steps,
                "projection_rate": self._totals["projection_count"] / self.episode_steps,
                "nonzero_dispatch_action_rate": self._totals["nonzero_dispatch_actions"] / self.episode_steps,
                "nonzero_reserve_action_rate": self._totals["nonzero_reserve_actions"] / self.episode_steps,
                "event_compliance_rate": 1.0 - self._totals["event_shortfall_kwh"] / max(1.0, sum(
                    self._context_at(i)["dr_target_kw"] for i in range(self.episode_steps)
                )),
                "terminal_soc_error": max(0.0, abs(self._soc - self.soc_initial) - self.terminal_soc_tolerance),
            })
        next_observation = self._observation() if not terminated else np.zeros(len(STATE_NAMES), dtype=np.float32)
        return next_observation, float(reward), terminated, False, info

    def _context_at(self, step: int) -> Dict[str, float]:
        current = self._step
        self._step = step
        try:
            return self._context()
        finally:
            self._step = current

    @property
    def totals(self) -> Dict[str, float]:
        return dict(self._totals)

    def render(self):
        self.render_calls += 1
        if self.training:
            raise RuntimeError("BESS training rendering is prohibited")
        return list(self.trace)


def fixed_window_starts(length: int, episode_steps: int, count: int) -> List[int]:
    max_start = max(0, length - episode_steps - 1)
    if count <= 1 or max_start == 0:
        return [0]
    return sorted(set(int(round(value)) for value in np.linspace(0, max_start, count)))


def neutral_policy(_observation: np.ndarray, _env: BESSEnergyV3Env) -> np.ndarray:
    return np.asarray([0.0, -1.0], dtype=np.float32)


def balanced_event_policy(_observation: np.ndarray, env: BESSEnergyV3Env) -> np.ndarray:
    ctx = env._context()
    base, price, carbon = float(ctx["base_load_kw"]), float(ctx["price"]), float(ctx["carbon"])
    remaining = env.episode_steps - env._step
    soc_gap = env._soc - env.soc_initial
    recovery_threshold = max(0.0001, 0.002 * remaining / env.episode_steps)
    # Priority: deliver committed events, reduce physical peaks, then move
    # energy from lower-carbon to higher-carbon hours without exceeding PCC.
    # A rolling terminal correction starts early enough that the safety layer
    # never needs a last-hour recharge spike.
    if remaining <= 24:
        terminal_floor_gap = -env.terminal_soc_tolerance
        terminal_ceiling_gap = env.terminal_soc_tolerance
        if base > env.soft_cap_kw and soc_gap > terminal_floor_gap + 1e-4:
            energy_ratio = (soc_gap - terminal_floor_gap) * env.energy_kwh * env.discharge_eff / env.power_kw
            dispatch = min(0.15, max(0.0, (base - env.soft_cap_kw) / env.power_kw), energy_ratio)
        elif soc_gap < terminal_floor_gap:
            recovery_hours = max(1.0, remaining - 8.0)
            needed = -(soc_gap - terminal_floor_gap) * env.energy_kwh / recovery_hours / env.power_kw / env.charge_eff
            headroom = max(0.0, (env.soft_cap_kw - base) / env.power_kw)
            dispatch = -min(0.20, headroom, needed)
        elif soc_gap > terminal_ceiling_gap:
            recovery_hours = max(1.0, remaining - 8.0)
            dispatch = min(0.20, (soc_gap - terminal_ceiling_gap) * env.energy_kwh * env.discharge_eff / recovery_hours / env.power_kw)
        else:
            dispatch = 0.0
    elif soc_gap < -recovery_threshold and base < env.soft_cap_kw:
        needed = -soc_gap * env.energy_kwh / max(1.0, remaining) / env.power_kw / env.charge_eff
        headroom = max(0.0, (env.soft_cap_kw - base) / env.power_kw)
        dispatch = -min(0.20, headroom, needed)
    elif soc_gap > recovery_threshold and base > env.soft_cap_kw - 0.08 * env.power_kw:
        needed = soc_gap * env.energy_kwh * env.discharge_eff / max(1.0, remaining) / env.power_kw
        dispatch = min(0.20, needed)
    elif ctx["dr_event_active"] and env._soc > env.soc_initial + 0.001:
        dispatch = max(0.20, min(0.42, float(ctx["dr_target_kw"]) / env.power_kw))
    elif base > env.soft_cap_kw and env._soc > env.soc_initial + 0.001:
        dispatch = min(0.35, max(0.08, (base - env.soft_cap_kw) / env.power_kw))
    elif carbon >= 0.575 and env._soc > env.soc_initial + 0.001:
        dispatch = 0.18
    elif carbon <= 0.555 and env._soc < env.soc_initial + 0.08 and base + 0.06 * env.power_kw < env.soft_cap_kw:
        dispatch = -min(0.12, max(0.02, (env.soft_cap_kw - base) / env.power_kw))
    elif price >= 0.95 and env._soc > env.soc_initial + 0.001:
        dispatch = 0.12
    else:
        dispatch = 0.0
    required_reserve = float(ctx["reserve_required_kw"])
    reserve = 0.0 if required_reserve <= 1e-9 else min(0.70, required_reserve / env.power_kw + 0.03)
    return np.asarray([dispatch, reserve], dtype=np.float32)


def evaluate_windows(
    env_factory: Callable[[], BESSEnergyV3Env],
    policy: Callable[[np.ndarray, BESSEnergyV3Env], np.ndarray],
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
                sample = {"timestamp": info["timestamp"], "state": info["context"],
                          "requested_action": info["requested_action"], "final_action": info["final_action"],
                          "projection": info["projection"], "model_output_derived_business": info["business_step"]}
            done = bool(terminated or truncated)
        episodes.append(env.totals)
        env.close()
    numeric = sorted(set.intersection(*(set(row) for row in episodes))) if episodes else []
    mean = {key: float(np.mean([float(row[key]) for row in episodes])) for key in numeric}
    std = {key: float(np.std([float(row[key]) for row in episodes])) for key in numeric}
    return {"episodes": len(episodes), "mean": mean, "std": std, "sample_inference": sample}
