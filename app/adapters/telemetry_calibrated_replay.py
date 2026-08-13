"""Continuous, provenance-labelled telemetry for the offline V3 experience.

The adapter time-warps the checked-in Shanghai public benchmark so a cloned
repository has a continuously moving data source.  It is deliberately labelled
as a calibrated public-data replay, never as measured terminal telemetry.
Asset values are a deterministic, mass-conserving decomposition of the public
aggregate load using the row's operational factors.  Replacing this adapter
with an authorized TOS/EMS/PLC implementation does not change the public API.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


ASSET_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "qc-01", "label": "岸桥 QC-01", "category": "岸桥", "rated_kw": 4200, "share": 0.15},
    {"id": "qc-02", "label": "岸桥 QC-02", "category": "岸桥", "rated_kw": 4200, "share": 0.15},
    {"id": "yard-01", "label": "场桥 YC-01", "category": "场桥", "rated_kw": 3200, "share": 0.14},
    {"id": "agv-01", "label": "AGV / 集卡充换电", "category": "AGV", "rated_kw": 2600, "share": 0.10},
    {"id": "bess-01", "label": "岸电储能 BESS-01", "category": "BESS", "rated_kw": 4000, "share": 0.03},
    {"id": "hvac-01", "label": "港区冷站 HVAC-01", "category": "冷站", "rated_kw": 3000, "share": 0.08},
    {"id": "reefer-01", "label": "冷藏箱堆场", "category": "冷藏箱", "rated_kw": 2800, "share": 0.08},
    {"id": "shore-01", "label": "船舶岸电区", "category": "岸电", "rated_kw": 5000, "share": 0.12},
    {"id": "gate-01", "label": "闸口 / 铁水联运", "category": "闸口", "rated_kw": 1900, "share": 0.05},
    {"id": "lighting-01", "label": "堆场高杆照明", "category": "照明", "rated_kw": 1800, "share": 0.07},
    {"id": "aux-01", "label": "辅助系统", "category": "其他", "rated_kw": 1600, "share": 0.03},
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _ratio(value: Any, default: float = 0.5) -> float:
    return max(0.0, min(1.0, _finite(value, default)))


class CalibratedReplayTelemetry:
    """Accelerated Shanghai public replay with a production-compatible API."""

    def __init__(self, path: str | None = None, *, history_points: int = 240) -> None:
        root = Path(__file__).resolve().parents[2]
        configured = path or os.getenv("PORT_DT_CALIBRATED_REPLAY_DATASET")
        self.path = (
            Path(configured).expanduser().resolve()
            if configured
            else root / "data/rl/datasets/public_cn_sha_hourly_v3.csv"
        )
        self.history_points = max(60, min(720, int(history_points)))
        self._rows = self._load()
        self._sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._started_monotonic = time.monotonic()
        # One wall-clock second advances one minute in the public replay.
        self._source_minutes_per_wall_second = 1.0
        self._base_position = int(datetime.now(timezone.utc).timestamp() // 3600) % len(self._rows)
        self._assets = {str(item["id"]): dict(item) for item in ASSET_SPECS}

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            raise FileNotFoundError(f"calibrated replay dataset not found: {self.path}")
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                try:
                    timestamp = datetime.fromisoformat(str(raw["timestamp"]).replace("Z", "+00:00"))
                    base_load = float(raw["base_load_kw"])
                except (KeyError, TypeError, ValueError):
                    continue
                row: dict[str, Any] = {"timestamp": timestamp.astimezone(timezone.utc).isoformat(), "base_load_kw": base_load}
                for key, value in raw.items():
                    if key in {"timestamp", "base_load_kw"}:
                        continue
                    row[key] = None if value in {None, ""} else _finite(value)
                rows.append(row)
        if len(rows) < 720:
            raise ValueError("calibrated replay requires at least 720 valid hourly rows")
        return rows

    def _position(self) -> float:
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        return self._base_position + elapsed * self._source_minutes_per_wall_second / 60.0

    def _interpolated_row(self, position: float) -> dict[str, Any]:
        left_index = math.floor(position) % len(self._rows)
        right_index = (left_index + 1) % len(self._rows)
        fraction = position - math.floor(position)
        left, right = self._rows[left_index], self._rows[right_index]
        row: dict[str, Any] = {}
        for key in left:
            if key == "timestamp":
                continue
            a, b = left.get(key), right.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                row[key] = float(a) + (float(b) - float(a)) * fraction
            else:
                row[key] = a if a is not None else b
        source_start = datetime.fromisoformat(str(left["timestamp"]).replace("Z", "+00:00"))
        row["source_timestamp"] = (source_start + timedelta(hours=fraction)).isoformat()
        return row

    @staticmethod
    def _asset_raw_weights(row: dict[str, Any]) -> dict[str, float]:
        berth = _ratio(row.get("berth_occupancy_ratio"), 0.65)
        yard = _ratio(row.get("yard_occupancy_ratio"), 0.65)
        crane = _ratio(row.get("crane_availability_ratio"), 0.9)
        equipment = _ratio(row.get("equipment_availability_ratio"), 0.9)
        congestion = _ratio(row.get("channel_congestion_ratio"), 0.5)
        arrivals = max(0.0, _finite(row.get("vessel_arrivals"), 1.0))
        throughput = max(0.0, _finite(row.get("throughput_teu"), 1.0))
        ambient = _finite(row.get("ambient_c"), 20.0)
        reefer = max(0.0, _finite(row.get("reefer_load_kw"), 500.0))
        try:
            hour = datetime.fromisoformat(str(row["source_timestamp"]).replace("Z", "+00:00")).hour
        except Exception:
            hour = 12
        night = 1.0 if hour >= 18 or hour < 6 else 0.35
        throughput_factor = max(0.72, min(1.30, throughput / 6100.0))
        return {
            "qc-01": 0.15 * (0.65 + 0.35 * crane) * (0.72 + 0.28 * berth) * throughput_factor,
            "qc-02": 0.15 * (0.67 + 0.33 * crane) * (0.76 + 0.24 * berth) * throughput_factor,
            "yard-01": 0.14 * (0.62 + 0.38 * yard) * (0.72 + 0.28 * equipment),
            "agv-01": 0.10 * (0.66 + 0.34 * equipment) * (0.80 + 0.20 * congestion),
            "bess-01": 0.03,
            "hvac-01": 0.08 * max(0.72, min(1.38, 1.0 + (ambient - 22.0) * 0.025)),
            "reefer-01": 0.08 * max(0.70, min(1.35, reefer / 600.0)),
            "shore-01": 0.12 * max(0.68, min(1.35, 0.55 + 0.24 * arrivals + 0.30 * berth)),
            "gate-01": 0.05 * (0.76 + 0.24 * congestion) * throughput_factor,
            "lighting-01": 0.07 * (0.55 + 0.45 * night),
            "aux-01": 0.03,
        }

    def asset_breakdown(self, row: dict[str, Any]) -> dict[str, float]:
        raw = self._asset_raw_weights(row)
        total = sum(raw.values()) or 1.0
        aggregate_kw = max(0.0, _finite(row.get("base_load_kw")))
        return {asset_id: aggregate_kw * weight / total for asset_id, weight in raw.items()}

    def list_assets(self) -> List[Dict[str, Any]]:
        return [
            {
                **item,
                "port": "CNSHA public benchmark",
                "supports": ["now", "forecast", "sim"],
                "source": "calibrated_public_replay_simulator",
                "measured": False,
                "include_in_aggregate": True,
            }
            for item in ASSET_SPECS
        ]

    def current_port_state(self) -> Dict[str, Any]:
        row = self._interpolated_row(self._position())
        source = datetime.fromisoformat(str(row["source_timestamp"]).replace("Z", "+00:00"))
        return {
            **row,
            "hour": source.hour + source.minute / 60.0,
            "soc": 0.55,
            "queue": max(0.0, _finite(row.get("throughput_teu")) * _ratio(row.get("channel_congestion_ratio"))),
            "last_bess_kw": 0.0,
            "episode_progress": 0.0,
        }

    def port_state_series(self, horizon_min: int, step_min: int) -> List[Dict[str, Any]]:
        steps = max(1, int(horizon_min) // max(1, int(step_min)))
        start = self._position()
        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        for index in range(steps):
            # Dataset cadence is hourly; interpolate source states at requested minutes.
            state = self._interpolated_row(start + (index + 1) * max(1, int(step_min)) / 60.0)
            source = datetime.fromisoformat(str(state["source_timestamp"]).replace("Z", "+00:00"))
            state.update(
                timestamp=(now + timedelta(minutes=(index + 1) * max(1, int(step_min)))).isoformat(),
                hour=source.hour + source.minute / 60.0,
                soc=0.55,
                queue=max(0.0, _finite(state.get("throughput_teu")) * _ratio(state.get("channel_congestion_ratio"))),
                last_bess_kw=0.0,
                episode_progress=min(1.0, (index + 1) / max(1, steps)),
            )
            rows.append(state)
        return rows

    def get_recent_power(self, asset_id: str) -> List[Dict[str, Any]]:
        if asset_id not in self._assets and asset_id != "port-grid-aggregate":
            return []
        end = self._position()
        now = datetime.now(timezone.utc)
        points: list[dict[str, Any]] = []
        for offset in range(self.history_points - 1, -1, -1):
            # One emitted second maps to one source minute; values remain smooth
            # because the hourly public rows are linearly interpolated.
            row = self._interpolated_row(end - offset / 60.0)
            breakdown = self.asset_breakdown(row)
            kw = _finite(row.get("base_load_kw")) if asset_id == "port-grid-aggregate" else breakdown[asset_id]
            points.append(
                {
                    "ts": (now - timedelta(seconds=offset)).isoformat(),
                    "kW": round(max(0.0, kw), 6),
                    "source_ts": row.get("source_timestamp"),
                    "source_mode": "calibrated_public_replay_simulator",
                    "measured": False,
                }
            )
        return points

    def get_series(
        self,
        asset_id: str,
        point: str,
        start_ts: float,
        end_ts: float,
        step_sec: int = 60,
    ) -> List[Dict[str, Any]]:
        """Production-compatible polling contract used by the ingest pipeline."""
        if (
            point != "active_power_kw"
            or end_ts <= start_ts
            or (asset_id not in self._assets and asset_id != "port-grid-aggregate")
        ):
            return []
        cadence = max(1, int(step_sec))
        count = max(1, int((float(end_ts) - float(start_ts)) // cadence) + 1)
        # Keep up to one week of minute-resolution replay so monitoring windows
        # are observed rather than silently extended by forward fill.
        count = min(count, 10_080)
        current_position = self._position()
        current_wall_epoch = time.time()
        output: list[dict[str, Any]] = []
        for index in range(count):
            epoch = float(start_ts) + index * cadence
            if epoch > float(end_ts) + 1e-9:
                break
            # Anchor every requested wall-clock timestamp to one unique source
            # position.  Using request ``end_ts`` here made two equal-length,
            # non-overlapping windows replay the same source segment and forced
            # PSI toward zero.
            seconds_from_now = current_wall_epoch - epoch
            # One wall second maps to one source minute; dataset positions are hours.
            row = self._interpolated_row(current_position - seconds_from_now / 60.0)
            breakdown = self.asset_breakdown(row)
            value = (
                _finite(row.get("base_load_kw"))
                if asset_id == "port-grid-aggregate"
                else breakdown[asset_id]
            )
            output.append(
                {
                    "ts": datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(),
                    "v": round(max(0.0, value), 6),
                    "source_ts": row.get("source_timestamp"),
                    "measured": False,
                }
            )
        return output

    def source_status(self) -> Dict[str, Any]:
        return {
            "mode": "calibrated_public_replay_simulator",
            "artifact_id": self.path.name,
            "sha256": self._sha256,
            "rows": len(self._rows),
            "assets": len(ASSET_SPECS),
            "continuous": True,
            "time_warp": "1 wall-clock second = 1 public-dataset minute",
            "asset_decomposition": "deterministic_factor_conditioned_mass_conserving",
            "measured": False,
            "production": False,
            "replacement_contract": "list_assets/get_recent_power/current_port_state; replace with authorized TOS/EMS/PLC adapter",
            "note": "Shanghai public aggregate plus public reanalysis and declared engineering derivatives; not terminal telemetry.",
        }
