"""Yard-lighting V3.1 public-signal-enriched engineering replay.

Zone lux/power/activity are checked-in emulator outputs. Shanghai/Yangshan
weather, price, carbon and yard context come from the public V3 replay.  This
module keeps that distinction explicit and never renders during training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


REPO_ROOT = Path(__file__).resolve().parents[4]

STATE_NAMES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "cloud_cover", "moon_phase", "ambient_lux", "is_night",
    "public_ambient_c", "public_wind_speed_mps", "public_yard_occupancy_ratio",
    "public_equipment_availability_ratio", "public_base_load_kw",
    "price_yuan_per_kwh", "ef_kg_per_kwh",
    "zone_count", "critical_zone_count", "complaint_zone_count", "fault_zone_count", "on_zone_ratio",
    "lux_mean", "lux_min", "lux_p10", "lux_max", "lux_margin_mean", "under_lux_rate",
    "fleet_power_kw", "dimming_mean_ratio", "dimming_p10_ratio", "dimming_p90_ratio",
    "activity_mean", "activity_p90_mean", "activity_max", "high_activity_zone_ratio",
    "complaint_event_count", "previous_base_residual", "previous_activity_gain", "previous_weather_gain",
    "rolling_power_30min_kw", "rolling_lux_30min", "rolling_activity_30min",
]
ACTION_NAMES = ["base_dimming_residual_ratio", "activity_gain_ratio", "weather_gain_ratio"]

CONTRACT: Dict[str, Any] = {
    "state_dimensions": len(STATE_NAMES), "state": STATE_NAMES,
    "action_dimensions": len(ACTION_NAMES), "actions": ACTION_NAMES,
    "reward_terms": [
        "energy_cost", "demand_peak", "carbon_shadow_cost", "under_lux",
        "critical_zone_lux", "complaint_sensitive_lux", "glare", "action_chatter",
        "safety_projection", "switch_count",
    ],
    "hard_constraints": [
        "finite_action_fail_closed", "action_absolute_bounds", "five_minute_action_ramp",
        "zone_dimming_minimum", "zone_dimming_maximum", "minimum_lux_by_zone",
        "critical_zone_lux_margin", "complaint_sensitive_window", "activity_visibility_boost",
        "minimum_dwell", "maximum_switches_per_night", "glare_no_upward_overshoot",
        "fault_zone_hold_safe", "sensor_loss_rule_fallback", "gateway_timeout_safe_hold",
        "operator_override_priority", "rollback_to_last_safe_schedule",
    ],
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        values = list(csv.DictReader(stream))
    if not values:
        raise RuntimeError(f"empty lighting source: {path}")
    return values


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_config(path: Path | None = None) -> Dict[str, Any]:
    return json.loads((path or REPO_ROOT / "config" / "yard_lighting_v3.json").read_text(encoding="utf-8"))


@dataclass
class YardLightingDataset:
    timestamps: List[datetime]
    rows: List[Dict[str, Any]]
    source_files: List[Dict[str, Any]]
    quality: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.rows)

    def describe(self) -> Dict[str, Any]:
        train_rows = int(round(len(self) * 0.70))
        validation_rows = int(round(len(self) * 0.10))
        return {
            "dataset_id": "yard_lighting_public_enriched_engineering_replay_v3",
            "rows": len(self), "raw_lighting_rows": self.quality["raw_lighting_rows"],
            "raw_activity_rows": self.quality["raw_activity_rows"], "zones": self.quality["zones"],
            "public_source_observations": self.quality["public_source_observations"],
            "public_aligned_steps": self.quality["public_aligned_steps"],
            "weather_reconstructed_steps": self.quality["weather_reconstructed_steps"],
            "time_range": {"start": self.timestamps[0].isoformat(), "end": self.timestamps[-1].isoformat()},
            "train_rows": train_rows, "validation_rows": validation_rows,
            "blind_test_rows": len(self) - train_rows - validation_rows,
            "split": "chronological_70_10_20", "measured": False,
            "evidence_tier": "public_reanalysis_enriched_engineering_emulator_replay",
            "quality": self.quality, "files": self.source_files,
        }


def load_dataset(config: Dict[str, Any] | None = None) -> YardLightingDataset:
    cfg = config or load_config()
    data_dir = REPO_ROOT / cfg["dataset"]["data_dir"]
    paths = {name: data_dir / name for name in (
        "zones_master.csv", "lighting_telemetry.csv", "activity_forecast.csv",
        "weather_astro.csv", "complaints_events.csv", "market_price.csv", "grid_ef.csv", "config_limits.json",
    )}
    public_path = REPO_ROOT / cfg["dataset"]["public_port_path"]
    public_meta_path = REPO_ROOT / cfg["dataset"]["public_port_meta_path"]
    zones, telemetry = _csv(paths["zones_master.csv"]), _csv(paths["lighting_telemetry.csv"])
    activity_rows, weather_rows = _csv(paths["activity_forecast.csv"]), _csv(paths["weather_astro.csv"])
    complaint_rows, public_rows = _csv(paths["complaints_events.csv"]), _csv(public_path)
    zone_map = {row["zone_id"]: row for row in zones}

    telemetry_by_time: Dict[datetime, List[Dict[str, str]]] = defaultdict(list)
    for row in telemetry:
        telemetry_by_time[_time(row["timestamp"])].append(row)
    timestamps = sorted(telemetry_by_time)
    timestamp_set = set(timestamps)

    activity_acc: Dict[Tuple[datetime, str], List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in activity_rows:
        timestamp = _time(row["timestamp"])
        if timestamp not in timestamp_set:
            continue
        target = activity_acc[(timestamp, row["zone_id"])]
        target[0] += _num(row["activity_score_p50"])
        target[1] += _num(row["activity_score_p90"])
        target[2] += 1.0

    weather_exact = {_time(row["timestamp"]): row for row in weather_rows}
    weather_pattern = {(_time(row["timestamp"]).weekday(), _time(row["timestamp"]).hour, _time(row["timestamp"]).minute): row for row in weather_rows}
    complaints: Dict[datetime, int] = defaultdict(int)
    for row in complaint_rows:
        complaints[_time(row["timestamp"])] += 1

    public_by_hour = {_time(row["timestamp"]): row for row in public_rows}
    public_meta = json.loads(public_meta_path.read_text(encoding="utf-8"))

    def public_at(timestamp: datetime) -> Dict[str, float]:
        hour = timestamp.replace(minute=0, second=0, microsecond=0)
        left = public_by_hour.get(hour)
        right = public_by_hour.get(hour + timedelta(hours=1), left)
        if not left:
            raise RuntimeError(f"public Shanghai replay missing hour {hour}")
        fraction = timestamp.minute / 60.0
        keys = ["ambient_c", "wind_speed_mps", "yard_occupancy_ratio", "equipment_availability_ratio", "base_load_kw", "price_per_kwh", "carbon_kg_per_kwh"]
        return {key: _num(left.get(key)) * (1.0 - fraction) + _num((right or left).get(key)) * fraction for key in keys}

    reconstructed_weather = 0
    rows: List[Dict[str, Any]] = []
    for timestamp in timestamps:
        zone_rows = telemetry_by_time[timestamp]
        if len(zone_rows) != len(zones):
            raise RuntimeError(f"lighting zone coverage mismatch at {timestamp}: {len(zone_rows)}")
        weather = weather_exact.get(timestamp)
        if weather is None:
            weather = weather_pattern.get((timestamp.weekday(), timestamp.hour, timestamp.minute))
            if weather is None:
                previous = timestamp - timedelta(minutes=5)
                weather = weather_pattern[(previous.weekday(), previous.hour, previous.minute)]
            reconstructed_weather += 1
        public = public_at(timestamp)
        vectors: Dict[str, List[float]] = defaultdict(list)
        fault_count = 0
        on_count = 0
        for zone_row in zone_rows:
            master = zone_map[zone_row["zone_id"]]
            acc = activity_acc.get((timestamp, zone_row["zone_id"]), [0.0, 0.0, 1.0])
            count = max(acc[2], 1.0)
            vectors["lux"].append(_num(zone_row["lux_measured"], _num(zone_row["lux"])))
            vectors["power"].append(_num(zone_row["power_kW"]))
            vectors["dimming"].append(_num(zone_row["dimming_percent"]) / 100.0)
            vectors["l_min"].append(_num(master.get("L_min_lux"), 20.0))
            vectors["activity_p50"].append(acc[0] / count)
            vectors["activity_p90"].append(acc[1] / count)
            vectors["critical"].append(float(_bool(master.get("critical_flag")) or _bool(master.get("critical"))))
            vectors["complaint"].append(float(_bool(master.get("complaint_zone"))))
            vectors["d_min"].append(_num(master.get("d_min_percent"), 10.0) / 100.0)
            vectors["d_max"].append(_num(master.get("d_max_percent"), 100.0) / 100.0)
            fault_count += zone_row["status"].strip().lower() == "fault"
            on_count += zone_row["status"].strip().lower() == "on"
        lux = np.asarray(vectors["lux"])
        l_min = np.asarray(vectors["l_min"])
        dim = np.asarray(vectors["dimming"])
        activity90 = np.asarray(vectors["activity_p90"])
        rows.append({
            "hour": timestamp.hour + timestamp.minute / 60.0, "dow": float(timestamp.weekday()),
            "is_weekend": float(timestamp.weekday() >= 5), "cloud_cover": _num(weather.get("cloud_cover_0_1")),
            "moon_phase": _num(weather.get("moon_phase_0_1")), "ambient_lux": _num(weather.get("ambient_lux")),
            "is_night": float(_bool(weather.get("is_night"))), **public,
            "zone_count": float(len(zones)), "critical_zone_count": float(sum(vectors["critical"])),
            "complaint_zone_count": float(sum(vectors["complaint"])), "fault_zone_count": float(fault_count),
            "on_zone_ratio": on_count / len(zones), "lux_mean": float(lux.mean()), "lux_min": float(lux.min()),
            "lux_p10": float(np.percentile(lux, 10)), "lux_max": float(lux.max()),
            "lux_margin_mean": float(np.mean(lux - l_min)), "under_lux_rate": float(np.mean(lux + 1e-9 < l_min)),
            "fleet_power_kw": float(sum(vectors["power"])), "dimming_mean_ratio": float(dim.mean()),
            "dimming_p10_ratio": float(np.percentile(dim, 10)), "dimming_p90_ratio": float(np.percentile(dim, 90)),
            "activity_mean": float(np.mean(vectors["activity_p50"])), "activity_p90_mean": float(activity90.mean()),
            "activity_max": float(activity90.max()),
            "high_activity_zone_ratio": float(np.mean(activity90 >= cfg["counterfactual"]["high_activity_threshold"])),
            "complaint_event_count": float(complaints[timestamp]), "zone_vectors": dict(vectors),
        })

    gaps = np.asarray([(timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0 for i in range(len(timestamps) - 1)])
    scalar_keys = [key for key in rows[0] if key != "zone_vectors"]
    numeric = np.asarray([[row[key] for key in scalar_keys] for row in rows], dtype=np.float64)
    expected = float(cfg["dataset"]["expected_interval_minutes"])
    quality = {
        "training_eligible": bool(
            len(rows) >= cfg["dataset"]["minimum_aggregate_rows"]
            and len(zones) == cfg["dataset"]["expected_zones"]
            and np.isfinite(numeric).all() and np.all(gaps > 0)
            and np.max(np.abs(gaps - expected)) < 1e-9
        ),
        "finite_numeric_rate": float(np.mean(np.isfinite(numeric))),
        "duplicate_timestamp_count": len(timestamps) - len(set(timestamps)),
        "interval_minutes": float(np.median(gaps)), "maximum_interval_minutes": float(gaps.max()),
        "raw_lighting_rows": len(telemetry), "raw_activity_rows": len(activity_rows), "zones": len(zones),
        "public_source_observations": int(public_meta.get("independent_source_observations") or 0),
        "public_aligned_steps": len(rows), "weather_reconstructed_steps": reconstructed_weather,
        "reconstruction": "local astro minute-of-week replay only after checked-in astro range; public Shanghai hourly signals linearly aligned to 5 minutes",
    }
    source_files = []
    row_counts = {
        "zones_master.csv": len(zones), "lighting_telemetry.csv": len(telemetry),
        "activity_forecast.csv": len(activity_rows), "weather_astro.csv": len(weather_rows),
        "complaints_events.csv": len(complaint_rows), "market_price.csv": len(_csv(paths["market_price.csv"])),
        "grid_ef.csv": len(_csv(paths["grid_ef.csv"])), "config_limits.json": None,
    }
    for name, path in paths.items():
        source_files.append({"path": str(path.relative_to(REPO_ROOT)), "rows": row_counts[name], "sha256": _sha(path)})
    source_files.extend([
        {"path": str(public_path.relative_to(REPO_ROOT)), "rows": len(public_rows), "sha256": _sha(public_path)},
        {"path": str(public_meta_path.relative_to(REPO_ROOT)), "rows": None, "sha256": _sha(public_meta_path)},
    ])
    return YardLightingDataset(timestamps=timestamps, rows=rows, source_files=source_files, quality=quality)


def chronological_slices(dataset: YardLightingDataset) -> Tuple[slice, slice, slice]:
    train_end = int(round(len(dataset) * 0.70))
    validation_end = train_end + int(round(len(dataset) * 0.10))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, len(dataset))


def fixed_window_starts(length: int, window: int, count: int) -> List[int]:
    if length <= window:
        return [0]
    return sorted(set(int(value) for value in np.linspace(0, length - window, count)))


class YardLightingV3Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset: YardLightingDataset, data_slice: slice, *, config: Dict[str, Any] | None = None,
                 normalization_slice: slice | None = None, episode_steps: int = 96, seed: int = 0,
                 training: bool = False, record_trace: bool = False) -> None:
        super().__init__()
        self.dataset, self.data_slice = dataset, data_slice
        self.config = config or load_config()
        self.normalization_slice = normalization_slice or data_slice
        self.episode_steps = min(episode_steps, data_slice.stop - data_slice.start)
        self.training, self.record_trace = training, record_trace
        self.render_calls, self.trace = 0, []
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-8.0, 8.0, shape=(len(STATE_NAMES),), dtype=np.float32)
        self._previous = np.zeros(3, dtype=np.float64)
        self._position = data_slice.start
        self._med, self._scale = self._normalization_stats()

    def _rolling(self, index: int, key: str) -> float:
        start = max(0, index - 5)
        return float(np.mean([self.dataset.rows[i][key] for i in range(start, index + 1)]))

    def _raw_state(self, index: int) -> np.ndarray:
        row = self.dataset.rows[index]
        return np.asarray([
            math.sin(2 * math.pi * row["hour"] / 24.0), math.cos(2 * math.pi * row["hour"] / 24.0),
            math.sin(2 * math.pi * row["dow"] / 7.0), math.cos(2 * math.pi * row["dow"] / 7.0), row["is_weekend"],
            row["cloud_cover"], row["moon_phase"], row["ambient_lux"], row["is_night"],
            row["ambient_c"], row["wind_speed_mps"], row["yard_occupancy_ratio"], row["equipment_availability_ratio"], row["base_load_kw"],
            row["price_per_kwh"], row["carbon_kg_per_kwh"], row["zone_count"], row["critical_zone_count"],
            row["complaint_zone_count"], row["fault_zone_count"], row["on_zone_ratio"], row["lux_mean"], row["lux_min"],
            row["lux_p10"], row["lux_max"], row["lux_margin_mean"], row["under_lux_rate"], row["fleet_power_kw"],
            row["dimming_mean_ratio"], row["dimming_p10_ratio"], row["dimming_p90_ratio"], row["activity_mean"],
            row["activity_p90_mean"], row["activity_max"], row["high_activity_zone_ratio"], row["complaint_event_count"],
            self._previous[0], self._previous[1], self._previous[2], self._rolling(index, "fleet_power_kw"),
            self._rolling(index, "lux_mean"), self._rolling(index, "activity_mean"),
        ], dtype=np.float64)

    def _normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray([self._raw_state(i) for i in range(self.normalization_slice.start, self.normalization_slice.stop)])
        med, q25, q75 = np.median(matrix, axis=0), np.percentile(matrix, 25, axis=0), np.percentile(matrix, 75, axis=0)
        scale = np.maximum(q75 - q25, np.maximum(np.abs(med) * 0.01, 1e-3))
        med[36:39] = 0.0
        scale[36:39] = np.asarray([0.08, 0.05, 0.03])
        return med, scale

    def _observation(self) -> np.ndarray:
        return np.clip((self._raw_state(self._position) - self._med) / self._scale, -8.0, 8.0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        max_offset = max(0, self.data_slice.stop - self.data_slice.start - self.episode_steps)
        if options and "start_index" in options:
            offset = int(options["start_index"])
        elif self.training and max_offset:
            offset = int(self.rng.integers(0, max_offset + 1))
        else:
            offset = 0
        offset = max(0, min(offset, max_offset))
        self._position = self.data_slice.start + offset
        self._steps, self._previous, self._previous_action, self.trace = 0, np.zeros(3), np.zeros(3), []
        return self._observation(), {"timestamp": self.dataset.timestamps[self._position].isoformat(), "start_index": offset}

    def _scale_action(self, action: Sequence[float]) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float64).reshape(3)
        if not np.isfinite(raw).all():
            raw = neutral_policy(np.zeros(len(STATE_NAMES)), self).astype(np.float64)
        raw = np.clip(raw, -1.0, 1.0)
        out = []
        for value, key in zip(raw, ACTION_NAMES):
            low, high = self.config["action"][key]
            scaled = low + (value + 1.0) * 0.5 * (high - low)
            out.append(0.0 if abs(scaled) < 1e-8 else scaled)
        return np.asarray(out)

    def _project(self, requested: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        ramp = np.asarray(self.config["action"]["action_ramp_per_5min"], dtype=np.float64)
        final = np.clip(requested, self._previous - ramp, self._previous + ramp)
        reasons = [] if np.allclose(final, requested) else ["five_minute_action_ramp"]
        return final, reasons

    def _business_step(self, row: Dict[str, Any], final: np.ndarray, reasons: List[str]) -> Dict[str, float]:
        vectors = {key: np.asarray(value, dtype=np.float64) for key, value in row["zone_vectors"].items()}
        base_dim, lux, l_min = vectors["dimming"], vectors["lux"], vectors["l_min"]
        activity = vectors["activity_p90"]
        weather_risk = max(row["cloud_cover"], min(1.0, row["wind_speed_mps"] / 15.0))
        requested_dim = base_dim + final[0] + final[1] * activity + final[2] * weather_risk
        final_dim = np.zeros_like(base_dim)
        predicted_lux = lux.copy()
        zone_projection_count = 0
        for index in range(len(base_dim)):
            if base_dim[index] <= 1e-9:
                final_dim[index] = 0.0
                continue
            required = base_dim[index] * l_min[index] / max(lux[index], 1e-6)
            lower = max(0.0, min(vectors["d_min"][index], base_dim[index]), required)
            upper = min(vectors["d_max"][index], base_dim[index])
            target = float(np.clip(requested_dim[index], lower, max(lower, upper)))
            if abs(target - requested_dim[index]) > 1e-8:
                zone_projection_count += 1
            final_dim[index] = target
            predicted_lux[index] = lux[index] * target / max(base_dim[index], 1e-6)
        if zone_projection_count:
            reasons.append("zone_minimum_lux_or_dimming_envelope")
        standby = self.config["counterfactual"]["fixture_standby_power_fraction"]
        responsive = self.config["counterfactual"]["fixture_dimming_power_fraction"]
        power_ratio = np.ones_like(base_dim)
        active = base_dim > 1e-9
        power_ratio[active] = standby + responsive * final_dim[active] / base_dim[active]
        baseline_power = float(vectors["power"].sum())
        power = float(np.sum(vectors["power"] * power_ratio))
        compliance = predicted_lux + 1e-7 >= l_min
        critical_mask = vectors["critical"] > 0.5
        complaint_mask = vectors["complaint"] > 0.5
        hours = 5.0 / 60.0
        baseline_energy, energy = baseline_power * hours, power * hours
        return {
            "baseline_power_kw": baseline_power, "power_kw": power,
            "baseline_energy_kwh": baseline_energy, "energy_kwh": energy,
            "baseline_energy_cost_cny": baseline_energy * row["price_per_kwh"],
            "energy_cost_cny": energy * row["price_per_kwh"],
            "baseline_carbon_kg": baseline_energy * row["carbon_kg_per_kwh"],
            "carbon_kg": energy * row["carbon_kg_per_kwh"],
            "minimum_lux_compliance_rate": float(np.mean(compliance)),
            "critical_lux_compliance_rate": float(np.mean(compliance[critical_mask])) if critical_mask.any() else 1.0,
            "complaint_lux_compliance_rate": float(np.mean(compliance[complaint_mask])) if complaint_mask.any() else 1.0,
            "under_lux_zone_count": float(np.sum(~compliance)),
            "zone_projection_count": float(zone_projection_count), "projection_count": float(len(reasons)),
            "mean_final_dimming_ratio": float(final_dim.mean()), "mean_predicted_lux": float(predicted_lux.mean()),
        }

    def preview_action(self, action: Sequence[float]) -> Dict[str, Any]:
        requested = self._scale_action(action)
        final, reasons = self._project(requested)
        return {"requested": requested, "final": final, "projection": reasons, "business": self._business_step(self.dataset.rows[self._position], final, reasons)}

    def step(self, action: Sequence[float]):
        preview = self.preview_action(action)
        requested, final, reasons, business = preview["requested"], preview["final"], preview["projection"], preview["business"]
        chatter = float(np.mean(np.abs(np.asarray(action, dtype=np.float64) - self._previous_action)))
        reward = -(
            business["energy_cost_cny"] / max(business["baseline_energy_cost_cny"], 1.0)
            + self.config["counterfactual"]["carbon_shadow_price_cny_per_kg"] * business["carbon_kg"] / 50.0
            + 30.0 * (1.0 - business["minimum_lux_compliance_rate"])
            + 30.0 * (1.0 - business["critical_lux_compliance_rate"])
            + 0.002 * business["zone_projection_count"] / 96.0 + 0.002 * chatter
        )
        row = self.dataset.rows[self._position]
        info = {
            "timestamp": self.dataset.timestamps[self._position].isoformat(),
            "context": {key: value for key, value in row.items() if key != "zone_vectors"},
            "requested_action": dict(zip(ACTION_NAMES, [float(value) for value in requested])),
            "final_action": dict(zip(ACTION_NAMES, [float(value) for value in final])),
            "projection": list(reasons), "business_step": business, "guardrail_violation": False,
        }
        if self.record_trace:
            self.trace.append(info)
        self._previous, self._previous_action = final, np.asarray(action, dtype=np.float64)
        self._steps += 1
        self._position += 1
        terminated = self._steps >= self.episode_steps or self._position >= self.data_slice.stop
        observation = np.zeros(len(STATE_NAMES), dtype=np.float32) if terminated else self._observation()
        return observation, float(reward), terminated, False, info

    def render(self):
        self.render_calls += 1
        if self.training:
            raise RuntimeError("yard-lighting training must not render")
        return self.trace[-1] if self.trace else None


def _normalized(value: float, key: str, env: YardLightingV3Env) -> float:
    low, high = env.config["action"][key]
    return float(np.clip(2.0 * (value - low) / (high - low) - 1.0, -1.0, 1.0))


def neutral_policy(_observation: np.ndarray, env: YardLightingV3Env) -> np.ndarray:
    return np.asarray([_normalized(0.0, key, env) for key in ACTION_NAMES], dtype=np.float32)


def safe_teacher_policy(_observation: np.ndarray, env: YardLightingV3Env) -> np.ndarray:
    row = env.dataset.rows[env._position]
    base = -0.10
    activity_gain = 0.04 if row["activity_p90_mean"] >= 0.30 else 0.02
    weather_gain = 0.02 if max(row["cloud_cover"], row["wind_speed_mps"] / 15.0) >= 0.65 else 0.0
    target = [base, activity_gain, weather_gain]
    return np.asarray([_normalized(target[i], key, env) for i, key in enumerate(ACTION_NAMES)], dtype=np.float32)


def evaluate_windows(factory: Callable[[], YardLightingV3Env], policy: Callable[[np.ndarray, YardLightingV3Env], np.ndarray], starts: Iterable[int]) -> Dict[str, Any]:
    windows = []
    for start in starts:
        env = factory()
        observation, reset = env.reset(options={"start_index": int(start)})
        totals = {key: 0.0 for key in (
            "reward", "baseline_energy_kwh", "energy_kwh", "baseline_energy_cost_cny", "energy_cost_cny",
            "baseline_carbon_kg", "carbon_kg", "minimum_lux_compliance_rate", "critical_lux_compliance_rate",
            "complaint_lux_compliance_rate", "under_lux_zone_count", "projection_count", "zone_projection_count",
        )}
        baseline_peak = policy_peak = 0.0
        violations = steps = 0
        last_info: Dict[str, Any] = {}
        done = False
        while not done:
            observation, reward, terminated, truncated, info = env.step(policy(observation, env))
            totals["reward"] += reward
            for key in totals:
                if key != "reward":
                    totals[key] += float(info["business_step"].get(key, 0.0))
            baseline_peak = max(baseline_peak, info["business_step"]["baseline_power_kw"])
            policy_peak = max(policy_peak, info["business_step"]["power_kw"])
            violations += int(bool(info["guardrail_violation"]))
            last_info, steps = info, steps + 1
            done = bool(terminated or truncated)
        demand = self_demand = float(env.config["counterfactual"]["demand_charge_cny_per_kw_window"])
        windows.append({
            **totals, "start_index": int(start), "start_timestamp": reset["timestamp"], "steps": steps,
            "baseline_peak_kw": baseline_peak, "peak_kw": policy_peak,
            "baseline_total_cost_cny": totals["baseline_energy_cost_cny"] + baseline_peak * demand,
            "total_cost_cny": totals["energy_cost_cny"] + policy_peak * self_demand,
            "minimum_lux_compliance_rate": totals["minimum_lux_compliance_rate"] / max(steps, 1),
            "critical_lux_compliance_rate": totals["critical_lux_compliance_rate"] / max(steps, 1),
            "complaint_lux_compliance_rate": totals["complaint_lux_compliance_rate"] / max(steps, 1),
            "projection_rate": totals["projection_count"] / max(steps, 1),
            "guardrail_violation_rate": violations / max(steps, 1), "last_info": last_info,
        })
        env.close()
    names = [
        "reward", "baseline_energy_kwh", "energy_kwh", "baseline_energy_cost_cny", "energy_cost_cny",
        "baseline_carbon_kg", "carbon_kg", "baseline_peak_kw", "peak_kw", "baseline_total_cost_cny", "total_cost_cny",
        "minimum_lux_compliance_rate", "critical_lux_compliance_rate", "complaint_lux_compliance_rate",
        "under_lux_zone_count", "projection_rate", "guardrail_violation_rate", "zone_projection_count",
    ]
    return {"windows": windows, "mean": {name: float(np.mean([row[name] for row in windows])) for name in names}, "std": {name: float(np.std([row[name] for row in windows])) for name in names}}


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
