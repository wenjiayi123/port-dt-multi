"""Yard-crane V3.1 chronological safe-control engineering replay.

The checked-in source contains a rich 16-crane/TOS/queue emulator, not port
PLC telemetry.  Every thermal and control counterfactual is therefore explicit,
hashable and replaceable.  Training never renders; only post-training replay may
render the selected policy.
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
    "moves_per_step", "moves_forecast_p50", "moves_forecast_p90",
    "scheduled_min_moves", "backlog", "active_cranes", "idle_cranes",
    "fleet_move_capacity", "fleet_utilization", "rmg_share", "rtg_share",
    "active_power_kw", "idle_power_kw", "baseline_power_kw", "fleet_rated_power_kw",
    "pcc_kw", "pcc_loading_ratio", "price_yuan_per_kwh", "ef_kg_per_kwh",
    "dr_active", "dr_required_reduction_kw", "critical_blocks", "quiet_blocks",
    "transformer_loading_proxy", "motor_temp_proxy_c", "inverter_temp_proxy_c",
    "previous_power_cap_residual_pct", "previous_idle_timeout_residual_min",
    "rolling_moves_1h", "rolling_backlog_1h", "rolling_power_1h_kw",
]
ACTION_NAMES = ["power_cap_residual_pct", "idle_timeout_residual_min"]

CONTRACT: Dict[str, Any] = {
    "state_dimensions": len(STATE_NAMES),
    "state": STATE_NAMES,
    "action_dimensions": len(ACTION_NAMES),
    "actions": ACTION_NAMES,
    "reward_terms": [
        "energy_cost", "demand_peak", "carbon_shadow_cost", "moves_retention",
        "job_sla_non_degradation", "thermal_risk", "action_chatter",
        "safety_projection", "mode_switch_cost",
    ],
    "hard_constraints": [
        "finite_action_fail_closed", "power_cap_absolute_bounds", "idle_timeout_bounds",
        "power_cap_15min_ramp", "idle_timeout_15min_ramp", "moves_capacity_reserve",
        "scheduled_job_minimum_protection", "backlog_recovery_envelope",
        "motor_temperature_envelope", "inverter_temperature_envelope",
        "fleet_rated_power", "pcc_loading_envelope", "dr_no_upward_action",
        "critical_block_service", "minimum_dwell_and_switch_interval", "fault_rollback",
    ],
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        values = list(csv.DictReader(stream))
    if not values:
        raise RuntimeError(f"empty yard-crane source: {path}")
    return values


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_config(path: Path | None = None) -> Dict[str, Any]:
    target = path or REPO_ROOT / "config" / "yard_crane_v3.json"
    return json.loads(target.read_text(encoding="utf-8"))


@dataclass
class YardCraneDataset:
    timestamps: List[datetime]
    rows: List[Dict[str, float]]
    source_files: List[Dict[str, Any]]
    quality: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.rows)

    def describe(self) -> Dict[str, Any]:
        train_rows = int(round(len(self) * 0.70))
        validation_rows = int(round(len(self) * 0.10))
        return {
            "dataset_id": "yard_crane_fleet_engineering_replay_v3",
            "rows": len(self),
            "raw_crane_telemetry_rows": int(self.quality["raw_crane_telemetry_rows"]),
            "tos_job_rows": int(self.quality["tos_job_rows"]),
            "queue_forecast_rows": int(self.quality["queue_forecast_rows"]),
            "cranes": int(self.quality["cranes"]),
            "yard_blocks": int(self.quality["yard_blocks"]),
            "time_range": {"start": self.timestamps[0].isoformat(), "end": self.timestamps[-1].isoformat()},
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "blind_test_rows": len(self) - train_rows - validation_rows,
            "split": "chronological_70_10_20",
            "measured": False,
            "evidence_tier": "checked_in_engineering_emulator_replay",
            "quality": self.quality,
            "files": self.source_files,
        }


def load_dataset(config: Dict[str, Any] | None = None) -> YardCraneDataset:
    cfg = config or load_config()
    data_dir = REPO_ROOT / cfg["dataset"]["data_dir"]
    paths = {name: data_dir / name for name in (
        "crane_telemetry.csv", "cranes_master.csv", "yard_blocks.csv", "job_events.csv",
        "queue_forecast.csv", "grid_meter.csv", "market_price.csv", "grid_ef.csv", "dr_events.json",
    )}
    telemetry = _rows(paths["crane_telemetry.csv"])
    cranes = _rows(paths["cranes_master.csv"])
    blocks = _rows(paths["yard_blocks.csv"])
    jobs = _rows(paths["job_events.csv"])
    queues = _rows(paths["queue_forecast.csv"])
    meter = _rows(paths["grid_meter.csv"])
    prices = _rows(paths["market_price.csv"])
    factors = _rows(paths["grid_ef.csv"])
    dr_events = json.loads(paths["dr_events.json"].read_text(encoding="utf-8"))

    crane_by_id = {row["crane_id"]: row for row in cranes}
    block_by_id = {row["yard_block"]: row for row in blocks}
    fleet_capacity = sum(_num(row["move_capacity_per_step"]) for row in cranes)
    fleet_rated = sum(_num(row["rated_active_kw"]) for row in cranes)
    rmg_share = sum(row["crane_type"].upper() == "RMG" for row in cranes) / len(cranes)

    aggregate: Dict[datetime, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    seen_cranes: Dict[datetime, set[str]] = defaultdict(set)
    for row in telemetry:
        timestamp = _time(row["timestamp"])
        target = aggregate[timestamp]
        moves = _num(row["moves"])
        power = _num(row["power_kw"])
        target["moves"] += moves
        target["baseline_power_kw"] += power
        target["active_power_kw"] += power if moves > 0 else 0.0
        target["idle_power_kw"] += power if moves <= 0 else 0.0
        target["backlog"] += _num(row["backlog_after_step"])
        target["baseline_cost_cny"] += _num(row["cost_yuan_per_step"])
        target["baseline_carbon_kg"] += _num(row["carbon_kg_per_step"])
        target["price"] += _num(row["price_yuan_per_kwh"])
        target["ef"] += _num(row["ef_kg_per_kwh"])
        target["active_cranes"] += float(moves > 0)
        seen_cranes[timestamp].add(row["crane_id"])

    queue_agg: Dict[datetime, List[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in queues:
        target = queue_agg[_time(row["timestamp"])]
        target[0] += _num(row["arrivals_p50_per_step"])
        target[1] += _num(row["arrivals_p90_per_step"])

    job_min: Dict[datetime, float] = defaultdict(float)
    critical_jobs: Dict[datetime, float] = defaultdict(float)
    for row in jobs:
        start, end = _time(row["start_time_utc"]), _time(row["end_time_utc"])
        steps = max(1, int(round((end - start).total_seconds() / 900.0)))
        per_step = _num(row["moves_min_accept"]) / steps
        cursor = start
        while cursor < end:
            job_min[cursor] += per_step
            block = block_by_id.get(row["yard_block"], {})
            critical_jobs[cursor] += float(block.get("is_critical") == "1")
            cursor += timedelta(minutes=15)

    meter_15m: Dict[datetime, List[float]] = defaultdict(list)
    for row in meter:
        timestamp = _time(row["timestamp"])
        bucket = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
        meter_15m[bucket].append(_num(row["pcc_kw"]))
    pcc = {key: float(np.mean(value)) for key, value in meter_15m.items()}

    dr_map: Dict[datetime, float] = defaultdict(float)
    for event in dr_events:
        cursor, end = _time(event["start_utc"]), _time(event["end_utc"])
        while cursor < end:
            dr_map[cursor] = max(dr_map[cursor], _num(event["required_reduction_kw"]))
            cursor += timedelta(minutes=15)

    timestamps = sorted(aggregate)
    rows: List[Dict[str, float]] = []
    for timestamp in timestamps:
        source = aggregate[timestamp]
        count = len(seen_cranes[timestamp])
        if count != len(cranes):
            raise RuntimeError(f"yard-crane fleet coverage mismatch at {timestamp}: {count}")
        utilization = source["moves"] / max(fleet_capacity, 1.0)
        power_ratio = source["baseline_power_kw"] / max(fleet_rated, 1.0)
        quiet_blocks = sum(
            row["noise_restriction_flag"] == "1"
            and (_num(row["quiet_hours_start_local"]) <= timestamp.hour < _num(row["quiet_hours_end_local"]))
            for row in blocks
        )
        transformer_loading = min(1.5, 0.30 + 0.55 * power_ratio + 0.15 * utilization)
        motor_proxy = 36.0 + 42.0 * utilization + 10.0 * power_ratio
        inverter_proxy = 34.0 + 38.0 * utilization + 9.0 * power_ratio
        rows.append({
            "hour": float(timestamp.hour + timestamp.minute / 60), "dow": float(timestamp.weekday()),
            "is_weekend": float(timestamp.weekday() >= 5), "moves": source["moves"],
            "moves_forecast_p50": queue_agg[timestamp][0], "moves_forecast_p90": queue_agg[timestamp][1],
            "scheduled_min_moves": job_min[timestamp], "backlog": source["backlog"],
            "active_cranes": source["active_cranes"], "idle_cranes": len(cranes) - source["active_cranes"],
            "fleet_capacity": fleet_capacity, "utilization": utilization, "rmg_share": rmg_share,
            "rtg_share": 1.0 - rmg_share, "active_power_kw": source["active_power_kw"],
            "idle_power_kw": source["idle_power_kw"], "baseline_power_kw": source["baseline_power_kw"],
            "fleet_rated_power_kw": fleet_rated, "pcc_kw": pcc.get(timestamp, source["baseline_power_kw"]),
            "pcc_loading_ratio": pcc.get(timestamp, source["baseline_power_kw"]) / max(fleet_rated * 1.35, 1.0),
            "price_yuan_per_kwh": source["price"] / count, "ef_kg_per_kwh": source["ef"] / count,
            "baseline_cost_cny": source["baseline_cost_cny"], "baseline_carbon_kg": source["baseline_carbon_kg"],
            "dr_active": float(timestamp in dr_map), "dr_required_reduction_kw": dr_map[timestamp],
            "critical_blocks": critical_jobs[timestamp], "quiet_blocks": float(quiet_blocks),
            "transformer_loading_proxy": transformer_loading, "motor_temp_proxy_c": motor_proxy,
            "inverter_temp_proxy_c": inverter_proxy,
        })

    gaps = np.asarray([(timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0 for i in range(len(timestamps) - 1)])
    numeric = np.asarray([[value for value in row.values()] for row in rows], dtype=np.float64)
    expected = float(cfg["dataset"]["expected_interval_minutes"])
    quality = {
        "training_eligible": bool(
            len(rows) >= int(cfg["dataset"]["minimum_aggregate_rows"])
            and len(cranes) == int(cfg["dataset"]["expected_cranes"])
            and len(blocks) == int(cfg["dataset"]["expected_yard_blocks"])
            and np.isfinite(numeric).all() and np.all(gaps > 0)
            and np.max(np.abs(gaps - expected)) < 1e-9
        ),
        "finite_numeric_rate": float(np.mean(np.isfinite(numeric))),
        "duplicate_timestamp_count": len(timestamps) - len(set(timestamps)),
        "interval_minutes": float(np.median(gaps)), "maximum_interval_minutes": float(np.max(gaps)),
        "raw_crane_telemetry_rows": len(telemetry), "tos_job_rows": len(jobs),
        "queue_forecast_rows": len(queues), "cranes": len(cranes), "yard_blocks": len(blocks),
        "thermal_state_source": "engineering_proxy_replacement_required",
    }
    source_files = []
    for name, path in paths.items():
        row_count = None if path.suffix == ".json" else len(_rows(path))
        source_files.append({"path": str(path.relative_to(REPO_ROOT)), "rows": row_count, "sha256": _sha(path)})
    return YardCraneDataset(timestamps=timestamps, rows=rows, source_files=source_files, quality=quality)


def chronological_slices(dataset: YardCraneDataset) -> Tuple[slice, slice, slice]:
    train_end = int(round(len(dataset) * 0.70))
    validation_end = train_end + int(round(len(dataset) * 0.10))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, len(dataset))


def fixed_window_starts(length: int, window: int, count: int) -> List[int]:
    if length <= window:
        return [0]
    return sorted(set(int(value) for value in np.linspace(0, length - window, count)))


class YardCraneV3Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset: YardCraneDataset, data_slice: slice, *, config: Dict[str, Any] | None = None,
                 normalization_slice: slice | None = None, episode_steps: int = 384, seed: int = 0,
                 training: bool = False, record_trace: bool = False) -> None:
        super().__init__()
        self.dataset, self.data_slice = dataset, data_slice
        self.config = config or load_config()
        self.normalization_slice = normalization_slice or data_slice
        self.episode_steps = min(episode_steps, data_slice.stop - data_slice.start)
        self.training, self.record_trace = training, record_trace
        self.render_calls, self.trace = 0, []
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-8.0, 8.0, shape=(len(STATE_NAMES),), dtype=np.float32)
        self._previous = np.zeros(2, dtype=np.float64)
        self._position = data_slice.start
        self._med, self._scale = self._normalization_stats()

    def _rolling(self, index: int, key: str) -> float:
        start = max(0, index - 3)
        values = [self.dataset.rows[i][key] for i in range(start, index + 1)]
        return float(sum(values)) if key != "baseline_power_kw" else float(np.mean(values))

    def _raw_state(self, index: int) -> np.ndarray:
        row = self.dataset.rows[index]
        return np.asarray([
            math.sin(2 * math.pi * row["hour"] / 24.0), math.cos(2 * math.pi * row["hour"] / 24.0),
            math.sin(2 * math.pi * row["dow"] / 7.0), math.cos(2 * math.pi * row["dow"] / 7.0), row["is_weekend"],
            row["moves"], row["moves_forecast_p50"], row["moves_forecast_p90"], row["scheduled_min_moves"],
            row["backlog"], row["active_cranes"], row["idle_cranes"], row["fleet_capacity"], row["utilization"],
            row["rmg_share"], row["rtg_share"], row["active_power_kw"], row["idle_power_kw"],
            row["baseline_power_kw"], row["fleet_rated_power_kw"], row["pcc_kw"], row["pcc_loading_ratio"],
            row["price_yuan_per_kwh"], row["ef_kg_per_kwh"], row["dr_active"], row["dr_required_reduction_kw"],
            row["critical_blocks"], row["quiet_blocks"], row["transformer_loading_proxy"], row["motor_temp_proxy_c"],
            row["inverter_temp_proxy_c"], self._previous[0], self._previous[1], self._rolling(index, "moves"),
            self._rolling(index, "backlog"), self._rolling(index, "baseline_power_kw"),
        ], dtype=np.float64)

    def _normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray([self._raw_state(i) for i in range(self.normalization_slice.start, self.normalization_slice.stop)])
        med = np.median(matrix, axis=0)
        q25, q75 = np.percentile(matrix, 25, axis=0), np.percentile(matrix, 75, axis=0)
        scale = np.maximum(q75 - q25, np.maximum(np.abs(med) * 0.01, 1e-3))
        med[31:33] = 0.0
        scale[31:33] = np.asarray([0.12, 3.0])
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
        self._steps, self._previous, self._previous_action, self.trace = 0, np.zeros(2), np.zeros(2), []
        return self._observation(), {"timestamp": self.dataset.timestamps[self._position].isoformat(), "start_index": offset}

    def _scale_action(self, action: Sequence[float]) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float64).reshape(2)
        if not np.isfinite(raw).all():
            raw = neutral_policy(np.zeros(len(STATE_NAMES)), self).astype(np.float64)
        raw = np.clip(raw, -1.0, 1.0)
        cfg = self.config["action"]
        out = []
        for value, key in zip(raw, ACTION_NAMES):
            low, high = cfg[key]
            scaled = low + (value + 1.0) * 0.5 * (high - low)
            out.append(0.0 if abs(scaled) < 1e-7 else scaled)
        return np.asarray(out)

    def _project(self, requested: np.ndarray, row: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        cfg = self.config["action"]
        final = requested.copy()
        reasons: List[str] = []
        ramp = np.asarray([cfg["power_cap_ramp_per_15min"], cfg["idle_timeout_ramp_min_per_15min"]])
        ramped = np.clip(final, self._previous - ramp, self._previous + ramp)
        if not np.allclose(ramped, final):
            reasons.append("fifteen_minute_action_ramp")
        final = ramped
        protected = row["scheduled_min_moves"] > 0 or row["critical_blocks"] > 0 or row["moves_forecast_p90"] > 70
        if protected and final[0] < -0.04:
            final[0] = -0.04
            reasons.append("job_sla_capacity_reserve")
        if protected and final[1] < -2.0:
            final[1] = -2.0
            reasons.append("job_sla_idle_timeout")
        if row["utilization"] >= 0.90 or row["backlog"] >= 140:
            if final[0] < 0.0:
                final[0] = 0.0
                reasons.append("high_utilization_or_backlog_recovery")
            if final[1] < 0.0:
                final[1] = 0.0
                reasons.append("high_utilization_or_backlog_idle_lock")
        if row["motor_temp_proxy_c"] >= 82 or row["inverter_temp_proxy_c"] >= 78:
            if final[0] > 0.0:
                final[0] = 0.0
                reasons.append("thermal_no_upward_action")
        if row["pcc_loading_ratio"] >= 0.95 or row["dr_active"] > 0:
            if final[0] > 0.0:
                final[0] = 0.0
                reasons.append("pcc_or_dr_no_upward_action")
        return final, reasons

    def _business_step(self, row: Dict[str, float], final: np.ndarray, reasons: Sequence[str]) -> Dict[str, float]:
        counter = self.config["counterfactual"]
        active_factor = max(0.97, 1.0 + counter["active_power_elasticity_to_cap"] * min(0.0, final[0]))
        idle_factor = max(
            counter["minimum_idle_power_ratio"],
            1.0 + counter["idle_power_elasticity_to_cap"] * min(0.0, final[0])
            + counter["idle_power_elasticity_per_timeout_min"] * min(0.0, final[1]),
        )
        power = row["active_power_kw"] * active_factor + row["idle_power_kw"] * idle_factor
        power = min(power, row["fleet_rated_power_kw"])
        service_floor = 0.0 if row["utilization"] >= 0.90 or row["backlog"] >= 140 else (-0.04 if row["scheduled_min_moves"] > 0 or row["critical_blocks"] > 0 else -0.18)
        moves_retained = float(final[0] + 1e-9 >= service_floor)
        job_protected = float(row["scheduled_min_moves"] <= 0 or final[0] >= -0.0400001)
        hours = 0.25
        energy = power * hours
        baseline_energy = row["baseline_power_kw"] * hours
        return {
            "baseline_power_kw": row["baseline_power_kw"], "power_kw": power,
            "baseline_energy_kwh": baseline_energy, "energy_kwh": energy,
            "baseline_energy_cost_cny": row["baseline_cost_cny"],
            "energy_cost_cny": energy * row["price_yuan_per_kwh"],
            "baseline_carbon_kg": row["baseline_carbon_kg"], "carbon_kg": energy * row["ef_kg_per_kwh"],
            "historical_moves": row["moves"], "policy_moves": row["moves"] * moves_retained,
            "moves_retention": moves_retained, "job_sla_non_degradation": job_protected,
            "delay_delta_minutes": 0.0 if moves_retained else 15.0,
            "projection_count": float(len(reasons)),
        }

    def preview_action(self, action: Sequence[float]) -> Dict[str, Any]:
        row = self.dataset.rows[self._position]
        requested = self._scale_action(action)
        final, reasons = self._project(requested, row)
        return {"requested": requested, "final": final, "projection": reasons, "business": self._business_step(row, final, reasons)}

    def step(self, action: Sequence[float]):
        row = self.dataset.rows[self._position]
        preview = self.preview_action(action)
        requested, final, reasons, business = preview["requested"], preview["final"], preview["projection"], preview["business"]
        chatter = float(np.mean(np.abs(np.asarray(action, dtype=np.float64) - self._previous_action)))
        reward = -(
            business["energy_cost_cny"] / max(business["baseline_energy_cost_cny"], 1.0)
            + self.config["counterfactual"]["carbon_shadow_price_cny_per_kg"] * business["carbon_kg"] / 100.0
            + 20.0 * (1.0 - business["moves_retention"])
            + 20.0 * (1.0 - business["job_sla_non_degradation"])
            + 0.004 * len(reasons) + 0.002 * chatter
        )
        info = {
            "timestamp": self.dataset.timestamps[self._position].isoformat(),
            "context": {key: float(value) for key, value in row.items()},
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
            raise RuntimeError("yard-crane training must not render")
        return self.trace[-1] if self.trace else None


def _normalized(value: float, key: str, env: YardCraneV3Env) -> float:
    low, high = env.config["action"][key]
    return float(np.clip(2.0 * (value - low) / (high - low) - 1.0, -1.0, 1.0))


def neutral_policy(_observation: np.ndarray, env: YardCraneV3Env) -> np.ndarray:
    return np.asarray([_normalized(0.0, key, env) for key in ACTION_NAMES], dtype=np.float32)


def safe_teacher_policy(_observation: np.ndarray, env: YardCraneV3Env) -> np.ndarray:
    row = env.dataset.rows[env._position]
    if row["utilization"] >= 0.90 or row["backlog"] >= 140:
        target = (0.0, 0.0)
    elif row["scheduled_min_moves"] > 0 or row["critical_blocks"] > 0 or row["moves_forecast_p90"] > 70:
        target = (-0.04, -2.0)
    elif row["moves"] > 0 or row["backlog"] >= 90:
        target = (-0.10, -3.0)
    else:
        target = (-0.18, -5.0)
    return np.asarray([_normalized(target[i], key, env) for i, key in enumerate(ACTION_NAMES)], dtype=np.float32)


def evaluate_windows(factory: Callable[[], YardCraneV3Env], policy: Callable[[np.ndarray, YardCraneV3Env], np.ndarray],
                     starts: Iterable[int]) -> Dict[str, Any]:
    windows: List[Dict[str, Any]] = []
    for start in starts:
        env = factory()
        observation, reset_info = env.reset(options={"start_index": int(start)})
        totals = {key: 0.0 for key in (
            "reward", "baseline_energy_kwh", "energy_kwh", "baseline_energy_cost_cny", "energy_cost_cny",
            "baseline_carbon_kg", "carbon_kg", "historical_moves", "policy_moves", "moves_retention",
            "job_sla_non_degradation", "delay_delta_minutes", "projection_count",
        )}
        baseline_peak = policy_peak = 0.0
        steps = violations = 0
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
        demand_rate = float(env.config["counterfactual"]["demand_charge_cny_per_kw_window"])
        windows.append({
            **totals, "start_index": int(start), "start_timestamp": reset_info["timestamp"], "steps": steps,
            "baseline_peak_kw": baseline_peak, "peak_kw": policy_peak,
            "baseline_total_cost_cny": totals["baseline_energy_cost_cny"] + baseline_peak * demand_rate,
            "total_cost_cny": totals["energy_cost_cny"] + policy_peak * demand_rate,
            "moves_retention_rate": totals["moves_retention"] / max(steps, 1),
            "job_sla_non_degradation_rate": totals["job_sla_non_degradation"] / max(steps, 1),
            "projection_rate": totals["projection_count"] / max(steps, 1),
            "guardrail_violation_rate": violations / max(steps, 1), "last_info": last_info,
        })
        env.close()
    metric_names = [
        "reward", "baseline_energy_kwh", "energy_kwh", "baseline_energy_cost_cny", "energy_cost_cny",
        "baseline_carbon_kg", "carbon_kg", "historical_moves", "policy_moves", "baseline_peak_kw", "peak_kw",
        "baseline_total_cost_cny", "total_cost_cny", "moves_retention_rate", "job_sla_non_degradation_rate",
        "projection_rate", "guardrail_violation_rate", "delay_delta_minutes",
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
