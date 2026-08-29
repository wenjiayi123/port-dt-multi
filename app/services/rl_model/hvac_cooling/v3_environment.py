"""Chronological HVAC V3.1 engineering-replay environment.

The checked-in source is an engineering emulator, not site telemetry.  The
environment therefore keeps every counterfactual coefficient explicit and
reports scenario KPIs only.  Training never renders; rendering is reserved for
post-training replay consumers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


REPO_ROOT = Path(__file__).resolve().parents[4]

STATE_NAMES = [
    "ambient_temp_c", "ambient_rh_pct", "wetbulb_c", "occupancy_index",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "cooling_load_kw", "cooling_load_forecast_p50_kw", "cooling_load_forecast_p90_kw",
    "ambient_temp_forecast_c", "ambient_rh_forecast_pct", "n_chillers_on", "plr",
    "baseline_chws_c", "baseline_sat_c", "baseline_static_pressure_pa",
    "chiller_power_kw", "chw_pumps_kw", "cw_pumps_kw", "tower_fans_kw",
    "baseline_plant_power_kw", "price_yuan_per_kwh", "ef_kg_per_kwh",
    "previous_chws_c", "previous_sat_c", "previous_static_pressure_pa",
    "load_to_available_capacity_ratio",
]

ACTION_NAMES = ["chws_residual_c", "sat_residual_c", "static_pressure_residual_pa"]

CONTRACT: Dict[str, Any] = {
    "state_dimensions": len(STATE_NAMES),
    "state": STATE_NAMES,
    "action_dimensions": len(ACTION_NAMES),
    "actions": ACTION_NAMES,
    "reward_terms": [
        "energy_cost", "demand_peak", "carbon_shadow_cost", "cooling_shortfall",
        "humidity_risk", "setpoint_ramp", "safety_projection", "action_chatter",
    ],
    "hard_constraints": [
        "chws_absolute_bounds", "sat_absolute_bounds", "static_pressure_bounds",
        "chws_15min_ramp", "sat_15min_ramp", "static_pressure_15min_ramp",
        "high_plr_chws_envelope", "high_occupancy_sat_envelope",
        "high_humidity_sat_envelope", "minimum_static_pressure_under_load",
        "n_minus_1_capacity_envelope", "finite_action_fail_closed",
    ],
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path | None = None) -> Dict[str, Any]:
    target = path or REPO_ROOT / "config" / "hvac_v3.json"
    return json.loads(target.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty HVAC source: {path}")
    return rows


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class HVACDataset:
    timestamps: List[datetime]
    rows: List[Dict[str, float]]
    source_files: List[Dict[str, Any]]
    quality: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.rows)

    def describe(self, *, validation_ratio: float, test_ratio: float) -> Dict[str, Any]:
        train_ratio = 1.0 - validation_ratio - test_ratio
        train_rows = int(round(len(self) * train_ratio))
        validation_rows = int(round(len(self) * validation_ratio))
        return {
            "dataset_id": "hvac_engineering_replay_v3",
            "rows": len(self),
            "time_range": {
                "start": self.timestamps[0].isoformat(sep=" "),
                "end": self.timestamps[-1].isoformat(sep=" "),
            },
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "blind_test_rows": len(self) - train_rows - validation_rows,
            "split": "chronological_70_10_20",
            "measured": False,
            "evidence_tier": "checked_in_engineering_emulator_replay",
            "quality": self.quality,
            "files": self.source_files,
        }


def load_dataset(config: Dict[str, Any] | None = None) -> HVACDataset:
    cfg = config or load_config()
    ds_cfg = cfg["dataset"]
    telemetry_path = REPO_ROOT / ds_cfg["telemetry_path"]
    load_path = REPO_ROOT / ds_cfg["load_forecast_path"]
    weather_path = REPO_ROOT / ds_cfg["weather_forecast_path"]
    plant_map_path = REPO_ROOT / ds_cfg["plant_map_path"]
    plant_master_path = REPO_ROOT / ds_cfg["plant_master_path"]
    telemetry = _csv_rows(telemetry_path)
    load_rows = _csv_rows(load_path)
    weather_rows = _csv_rows(weather_path)
    if not (len(telemetry) == len(load_rows) == len(weather_rows)):
        raise RuntimeError("HVAC telemetry/forecast row counts do not align")

    rows: List[Dict[str, float]] = []
    timestamps: List[datetime] = []
    required = [
        "ambient_temp_C", "ambient_rh_pct", "wetbulb_C", "occ_index", "hourofday",
        "dayofweek", "is_weekend", "cooling_load_kw", "n_chillers_on", "plr",
        "chws_sp_C", "avg_sat_C", "chiller_power_kw", "chw_pumps_kw", "cw_pumps_kw",
        "tower_fans_kw", "plant_power_kw", "price_yuan_per_kwh", "ef_kg_per_kwh",
    ]
    missing = sorted(set(required) - set(telemetry[0]))
    if missing:
        raise RuntimeError(f"HVAC telemetry missing required fields: {missing}")
    for telemetry_row, load_row, weather_row in zip(telemetry, load_rows, weather_rows):
        if telemetry_row["timestamp"] != load_row["timestamp"] or telemetry_row["timestamp"] != weather_row["timestamp"]:
            raise RuntimeError("HVAC source timestamps do not align")
        timestamp = datetime.strptime(telemetry_row["timestamp"], "%Y-%m-%d %H:%M:%S")
        timestamps.append(timestamp)
        row = {name: _num(telemetry_row.get(name)) for name in required}
        row.update({
            "q_cooling_forecast_kw_p50": _num(load_row.get("q_cooling_forecast_kw_p50")),
            "q_cooling_forecast_kw_p90": _num(load_row.get("q_cooling_forecast_kw_p90")),
            "ambient_temp_forecast_C": _num(load_row.get("ambient_temp_forecast_C")),
            "ambient_rh_forecast_pct": _num(load_row.get("ambient_rh_forecast_pct")),
        })
        rows.append(row)

    gaps = np.asarray([(timestamps[i + 1] - timestamps[i]).total_seconds() / 60 for i in range(len(timestamps) - 1)])
    numeric = np.asarray([[row[name] for name in required] for row in rows], dtype=np.float64)
    expected_interval = float(ds_cfg["expected_interval_minutes"])
    quality = {
        "training_eligible": bool(
            len(rows) >= int(ds_cfg["minimum_rows"])
            and np.isfinite(numeric).all()
            and np.all(gaps > 0)
            and np.max(np.abs(gaps - expected_interval)) < 1e-9
        ),
        "finite_numeric_rate": float(np.mean(np.isfinite(numeric))),
        "duplicate_timestamp_count": len(timestamps) - len(set(timestamps)),
        "interval_minutes": float(np.median(gaps)),
        "maximum_interval_minutes": float(np.max(gaps)),
    }
    source_files = []
    for path, source_rows in (
        (telemetry_path, telemetry), (load_path, load_rows), (weather_path, weather_rows),
        (plant_map_path, _csv_rows(plant_map_path)),
    ):
        source_files.append({"path": str(path.relative_to(REPO_ROOT)), "rows": len(source_rows), "sha256": _sha(path)})
    source_files.append({"path": str(plant_master_path.relative_to(REPO_ROOT)), "rows": None, "sha256": _sha(plant_master_path)})
    return HVACDataset(timestamps=timestamps, rows=rows, source_files=source_files, quality=quality)


def chronological_slices(dataset: HVACDataset) -> Tuple[slice, slice, slice]:
    n = len(dataset)
    train_end = int(round(n * 0.70))
    validation_end = train_end + int(round(n * 0.10))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, n)


def fixed_window_starts(length: int, window: int, count: int) -> List[int]:
    if length <= window:
        return [0]
    return sorted(set(int(value) for value in np.linspace(0, length - window, count)))


class HVACV3Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset: HVACDataset,
        data_slice: slice,
        *,
        config: Dict[str, Any] | None = None,
        normalization_slice: slice | None = None,
        episode_steps: int = 384,
        seed: int = 0,
        training: bool = False,
        record_trace: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_slice = data_slice
        self.config = config or load_config()
        self.normalization_slice = normalization_slice or data_slice
        self.episode_steps = min(episode_steps, data_slice.stop - data_slice.start)
        self.training = training
        self.record_trace = record_trace
        self.render_calls = 0
        self.trace: List[Dict[str, Any]] = []
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-8.0, 8.0, shape=(len(STATE_NAMES),), dtype=np.float32)
        self._med, self._scale = self._normalization_stats()
        self._start = data_slice.start
        self._position = self._start
        self._steps = 0
        self._previous = np.asarray([7.5, 14.0, self.config["action"]["reference_static_pressure_pa"]], dtype=np.float64)
        self._previous_action = np.zeros(3, dtype=np.float64)

    def _raw_state(self, index: int) -> np.ndarray:
        row = self.dataset.rows[index]
        hour = row["hourofday"]
        dow = row["dayofweek"]
        capacity = max(row["n_chillers_on"] * 1200.0, 1.0)
        return np.asarray([
            row["ambient_temp_C"], row["ambient_rh_pct"], row["wetbulb_C"], row["occ_index"],
            math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * dow / 7.0), math.cos(2 * math.pi * dow / 7.0), row["is_weekend"],
            row["cooling_load_kw"], row["q_cooling_forecast_kw_p50"], row["q_cooling_forecast_kw_p90"],
            row["ambient_temp_forecast_C"], row["ambient_rh_forecast_pct"], row["n_chillers_on"], row["plr"],
            row["chws_sp_C"], row["avg_sat_C"], self.config["action"]["reference_static_pressure_pa"],
            row["chiller_power_kw"], row["chw_pumps_kw"], row["cw_pumps_kw"], row["tower_fans_kw"],
            row["plant_power_kw"], row["price_yuan_per_kwh"], row["ef_kg_per_kwh"],
            self._previous[0], self._previous[1], self._previous[2], row["cooling_load_kw"] / capacity,
        ], dtype=np.float64)

    def _normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        saved_previous = self._previous.copy() if hasattr(self, "_previous") else np.asarray([7.5, 14.0, 800.0])
        self._previous = saved_previous
        matrix = np.asarray([self._raw_state(i) for i in range(self.normalization_slice.start, self.normalization_slice.stop)])
        med = np.median(matrix, axis=0)
        q25 = np.percentile(matrix, 25, axis=0)
        q75 = np.percentile(matrix, 75, axis=0)
        scale = np.maximum(q75 - q25, np.maximum(np.abs(med) * 0.01, 1e-3))
        # Previous setpoints are dynamic controller state, not constant source
        # columns.  Reuse the corresponding baseline centers but guarantee a
        # full physical-action-range scale so normal operation is not clipped.
        med[26:29] = med[16:19]
        scale[26:29] = np.maximum(scale[16:19], np.asarray([0.75, 0.6, 100.0]))
        return med, scale

    def _observation(self) -> np.ndarray:
        return np.clip((self._raw_state(self._position) - self._med) / self._scale, -8.0, 8.0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        max_offset = max(0, (self.data_slice.stop - self.data_slice.start) - self.episode_steps)
        if options and "start_index" in options:
            offset = int(options["start_index"])
        elif self.training and max_offset:
            offset = int(self.rng.integers(0, max_offset + 1))
        else:
            offset = 0
        offset = max(0, min(offset, max_offset))
        self._start = self.data_slice.start + offset
        self._position = self._start
        self._steps = 0
        row = self.dataset.rows[self._position]
        self._previous = np.asarray([row["chws_sp_C"], row["avg_sat_C"], self.config["action"]["reference_static_pressure_pa"]], dtype=np.float64)
        self._previous_action = np.zeros(3, dtype=np.float64)
        self.trace = []
        return self._observation(), {"timestamp": self.dataset.timestamps[self._position].isoformat(sep=" "), "start_index": offset}

    def _scale_action(self, action: Sequence[float]) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float64).reshape(3)
        if not np.isfinite(raw).all():
            raw = np.zeros(3, dtype=np.float64)
        raw = np.clip(raw, -1.0, 1.0)
        action_cfg = self.config["action"]
        scales = []
        for value, key in zip(raw, ("chws_residual_c", "sat_residual_c", "static_pressure_residual_pa")):
            low, high = action_cfg[key]
            scaled = low + (value + 1.0) * 0.5 * (high - low)
            scales.append(0.0 if abs(scaled) < 1e-6 else scaled)
        return np.asarray(scales, dtype=np.float64)

    def _project(self, requested: np.ndarray, row: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        baseline = np.asarray([row["chws_sp_C"], row["avg_sat_C"], self.config["action"]["reference_static_pressure_pa"]])
        target = baseline + requested
        reasons: List[str] = []
        limits = np.asarray([[6.0, 9.0], [12.0, 15.0], [500.0, 1200.0]])
        ramp = np.asarray([0.5, 0.6, 50.0])
        clipped = np.clip(target, limits[:, 0], limits[:, 1])
        if not np.allclose(clipped, target):
            reasons.append("absolute_setpoint_bounds")
        target = clipped
        ramped = np.clip(target, self._previous - ramp, self._previous + ramp)
        if not np.allclose(ramped, target):
            reasons.append("fifteen_minute_ramp")
        target = ramped
        high_load = row["plr"] >= 0.80 or row["cooling_load_kw"] / max(row["n_chillers_on"] * 1200.0, 1.0) >= 0.82
        if high_load:
            ceiling = baseline[0] + 0.25
            if target[0] > ceiling:
                target[0] = ceiling
                reasons.append("high_load_chws_envelope")
        if (high_load or row["occ_index"] >= 0.75) and target[1] > baseline[1] + 0.2:
            target[1] = baseline[1] + 0.2
            reasons.append("load_or_occupied_sat_envelope")
        if row["ambient_rh_pct"] >= 85 and target[1] > baseline[1]:
            target[1] = baseline[1]
            reasons.append("humidity_sat_envelope")
        minimum_sp = 760.0 if row["occ_index"] >= 0.75 or row["plr"] >= 0.8 else 650.0
        if target[2] < minimum_sp:
            target[2] = minimum_sp
            reasons.append("minimum_static_pressure")
        return target, reasons

    def _business_step(self, row: Dict[str, float], final: np.ndarray, projection: Sequence[str]) -> Dict[str, float]:
        counter = self.config["counterfactual"]
        baseline_sp = float(self.config["action"]["reference_static_pressure_pa"])
        d_chws = final[0] - row["chws_sp_C"]
        d_sat = final[1] - row["avg_sat_C"]
        chiller = row["chiller_power_kw"] * max(0.82, 1.0 - counter["chiller_saving_per_chws_c"] * d_chws - counter["chiller_saving_per_sat_c"] * d_sat)
        chw = row["chw_pumps_kw"] * max(0.90, 1.0 - counter["chw_pump_saving_per_chws_c"] * d_chws)
        other = row["cw_pumps_kw"] + row["tower_fans_kw"]
        fan_baseline = row["plant_power_kw"] * counter["ahu_fan_fraction_of_plant_power"]
        fan = fan_baseline * max(0.45, (final[2] / baseline_sp) ** counter["fan_pressure_exponent"])
        baseline_power = row["plant_power_kw"] + fan_baseline
        policy_power = max(0.0, chiller + chw + other + fan)
        load_risk = max(0.0, row["plr"] - 0.80) + max(
            0.0,
            row["cooling_load_kw"] / max(row["n_chillers_on"] * 1200.0, 1.0) - 0.82,
        )
        occupancy_risk = max(0.0, row["occ_index"] - 0.75)
        humidity_risk = max(0.0, row["ambient_rh_pct"] - 85.0) / 15.0 * max(0.0, d_sat - 1e-6)
        cooling_shortfall = (
            max(0.0, d_chws - 0.250001) * load_risk
            + max(0.0, d_sat - 0.200001) * (load_risk + occupancy_risk)
        )
        cooling_satisfied = 1.0 if cooling_shortfall <= 1e-9 and humidity_risk <= 1e-9 else 0.0
        hours = 0.25
        baseline_energy = baseline_power * hours
        energy = policy_power * hours
        baseline_cost = baseline_energy * row["price_yuan_per_kwh"]
        cost = energy * row["price_yuan_per_kwh"]
        baseline_carbon = baseline_energy * row["ef_kg_per_kwh"]
        carbon = energy * row["ef_kg_per_kwh"]
        return {
            "baseline_power_kw": baseline_power,
            "power_kw": policy_power,
            "baseline_energy_kwh": baseline_energy,
            "energy_kwh": energy,
            "baseline_energy_cost_cny": baseline_cost,
            "energy_cost_cny": cost,
            "baseline_carbon_kg": baseline_carbon,
            "carbon_kg": carbon,
            "cooling_satisfaction": cooling_satisfied,
            "cooling_shortfall": cooling_shortfall,
            "humidity_risk": humidity_risk,
            "projection_count": float(len(projection)),
        }

    def preview_action(self, action: Sequence[float]) -> Dict[str, Any]:
        row = self.dataset.rows[self._position]
        requested = self._scale_action(action)
        final, reasons = self._project(requested, row)
        business = self._business_step(row, final, reasons)
        return {"requested": requested, "final": final, "projection": reasons, "business": business}

    def step(self, action: Sequence[float]):
        row = self.dataset.rows[self._position]
        preview = self.preview_action(action)
        requested, final, reasons, business = preview["requested"], preview["final"], preview["projection"], preview["business"]
        chatter = float(np.mean(np.abs(np.asarray(action, dtype=np.float64) - self._previous_action)))
        reward = -(
            business["energy_cost_cny"] / max(business["baseline_energy_cost_cny"], 1.0)
            + self.config["counterfactual"]["carbon_shadow_price_cny_per_kg"] * business["carbon_kg"] / 100.0
            + 8.0 * business["cooling_shortfall"]
            + 8.0 * business["humidity_risk"]
            + 0.004 * len(reasons)
            + 0.002 * chatter
        )
        info = {
            "timestamp": self.dataset.timestamps[self._position].isoformat(sep=" "),
            "context": {name: row[name] for name in row},
            "requested_action": dict(zip(ACTION_NAMES, [float(v) for v in requested])),
            "final_action": {"chws_c": float(final[0]), "sat_c": float(final[1]), "static_pressure_pa": float(final[2])},
            "projection": list(reasons),
            "business_step": business,
            "guardrail_violation": False,
        }
        if self.record_trace:
            self.trace.append(info)
        self._previous = final
        self._previous_action = np.asarray(action, dtype=np.float64)
        self._steps += 1
        self._position += 1
        terminated = self._steps >= self.episode_steps or self._position >= self.data_slice.stop
        truncated = False
        if terminated:
            observation = np.zeros(len(STATE_NAMES), dtype=np.float32)
        else:
            observation = self._observation()
        return observation, float(reward), terminated, truncated, info

    def render(self):
        self.render_calls += 1
        if self.training:
            raise RuntimeError("HVAC training must not render")
        return self.trace[-1] if self.trace else None


def neutral_policy(_observation: np.ndarray, _env: HVACV3Env) -> np.ndarray:
    # Normalized values that map to exact zero residual for asymmetric ranges.
    cfg = _env.config["action"]
    values = []
    for key in ("chws_residual_c", "sat_residual_c", "static_pressure_residual_pa"):
        low, high = cfg[key]
        values.append(2.0 * (0.0 - low) / (high - low) - 1.0)
    return np.asarray(values, dtype=np.float32)


def safe_teacher_policy(_observation: np.ndarray, env: HVACV3Env) -> np.ndarray:
    best_action = neutral_policy(_observation, env)
    best_score = float("inf")
    action_cfg = env.config["action"]

    def normalized(value: float, key: str) -> float:
        low, high = action_cfg[key]
        return float(np.clip(2.0 * (value - low) / (high - low) - 1.0, -1.0, 1.0))

    pressure_low = float(action_cfg["static_pressure_residual_pa"][0])
    pressure_grid = tuple(sorted({pressure_low, -100.0, -50.0, 0.0, 50.0}))
    for d_chws in (0.0, 0.25, 0.5, 0.75):
        for d_sat in (-0.2, 0.0, 0.2, 0.4):
            for d_sp in pressure_grid:
                action = np.asarray([
                    normalized(d_chws, "chws_residual_c"),
                    normalized(d_sat, "sat_residual_c"),
                    normalized(d_sp, "static_pressure_residual_pa"),
                ], dtype=np.float32)
                preview = env.preview_action(action)
                business = preview["business"]
                score = (
                    business["energy_cost_cny"]
                    + env.config["counterfactual"]["carbon_shadow_price_cny_per_kg"] * business["carbon_kg"]
                    + 5000.0 * business["cooling_shortfall"]
                    + 5000.0 * business["humidity_risk"]
                    + 0.05 * len(preview["projection"])
                )
                if score < best_score:
                    best_score, best_action = score, action
    return best_action


def evaluate_windows(
    factory: Callable[[], HVACV3Env],
    policy: Callable[[np.ndarray, HVACV3Env], np.ndarray],
    starts: Iterable[int],
) -> Dict[str, Any]:
    windows: List[Dict[str, Any]] = []
    for start in starts:
        env = factory()
        observation, reset_info = env.reset(options={"start_index": int(start)})
        totals = {
            "reward": 0.0, "baseline_energy_kwh": 0.0, "energy_kwh": 0.0,
            "baseline_energy_cost_cny": 0.0, "energy_cost_cny": 0.0,
            "baseline_carbon_kg": 0.0, "carbon_kg": 0.0,
            "cooling_satisfaction": 0.0, "projection_count": 0.0,
        }
        baseline_peak = 0.0
        policy_peak = 0.0
        guardrail_violations = 0
        steps = 0
        last_info: Dict[str, Any] = {}
        done = False
        while not done:
            action = policy(observation, env)
            observation, reward, terminated, truncated, info = env.step(action)
            business = info["business_step"]
            totals["reward"] += reward
            for key in totals:
                if key != "reward":
                    totals[key] += float(business.get(key, 0.0))
            baseline_peak = max(baseline_peak, business["baseline_power_kw"])
            policy_peak = max(policy_peak, business["power_kw"])
            guardrail_violations += int(bool(info["guardrail_violation"]))
            steps += 1
            last_info = info
            done = bool(terminated or truncated)
        demand_rate = float(env.config["counterfactual"]["demand_charge_cny_per_kw_window"])
        baseline_total_cost = totals["baseline_energy_cost_cny"] + baseline_peak * demand_rate
        total_cost = totals["energy_cost_cny"] + policy_peak * demand_rate
        windows.append({
            **totals,
            "start_index": int(start), "start_timestamp": reset_info["timestamp"], "steps": steps,
            "baseline_peak_kw": baseline_peak, "peak_kw": policy_peak,
            "baseline_total_cost_cny": baseline_total_cost, "total_cost_cny": total_cost,
            "cooling_satisfaction_rate": totals["cooling_satisfaction"] / max(steps, 1),
            "projection_rate": totals["projection_count"] / max(steps, 1),
            "guardrail_violation_rate": guardrail_violations / max(steps, 1),
            "last_info": last_info,
        })
        env.close()
    metric_names = [
        "reward", "baseline_energy_kwh", "energy_kwh", "baseline_energy_cost_cny", "energy_cost_cny",
        "baseline_carbon_kg", "carbon_kg", "baseline_peak_kw", "peak_kw",
        "baseline_total_cost_cny", "total_cost_cny", "cooling_satisfaction_rate",
        "projection_rate", "guardrail_violation_rate",
    ]
    return {
        "windows": windows,
        "mean": {name: float(np.mean([row[name] for row in windows])) for name in metric_names},
        "std": {name: float(np.std([row[name] for row in windows])) for name in metric_names},
    }


class NumpyMLPPolicy:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self.weights = [np.asarray(layer["weight"], dtype=np.float64) for layer in payload["layers"]]
        self.biases = [np.asarray(layer["bias"], dtype=np.float64) for layer in payload["layers"]]

    @classmethod
    def load(cls, path: Path) -> "NumpyMLPPolicy":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float64)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = weight @ value + bias
            value = np.tanh(value) if index == len(self.weights) - 1 else np.maximum(value, 0.0)
        return np.asarray(value, dtype=np.float32)


def artifact_policy(policy: NumpyMLPPolicy):
    return lambda observation, _env: policy.predict(observation)
