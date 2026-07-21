"""Read-only telemetry replay backed by the canonical RL dataset contract."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class DatasetTelemetry:
    def __init__(self, path: str | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        configured = path or os.getenv("PORT_DT_TELEMETRY_DATASET")
        self.path = Path(configured).expanduser().resolve() if configured else root / "data/rl/datasets/public_port_ops_v1.csv"
        self._rows = self._load()

    @staticmethod
    def _epoch(value: str) -> float:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.is_file():
            raise FileNotFoundError(f"telemetry dataset not found: {self.path}")
        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                try:
                    timestamp = str(raw["timestamp"])
                    power = float(raw["base_load_kw"])
                    epoch = self._epoch(timestamp)
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append({"ts": timestamp, "epoch": epoch, "kW": power})
        rows.sort(key=lambda item: item["epoch"])
        if len(rows) < 18:
            raise ValueError("telemetry dataset requires at least 18 valid power rows")
        return rows

    def list_assets(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "port-grid-aggregate",
                "label": "公开数据集港区聚合负荷",
                "source": "canonical_dataset_replay",
                "measured": False,
            }
        ]

    def get_recent_power(self, asset_id: str) -> List[Dict[str, Any]]:
        if asset_id != "port-grid-aggregate":
            return []
        return [{"ts": row["ts"], "kW": row["kW"]} for row in self._rows[-720:]]

    def get_series(
        self,
        asset_id: str,
        point: str,
        start_ts: float,
        end_ts: float,
        step_sec: int = 60,
    ) -> List[Dict[str, Any]]:
        if asset_id != "port-grid-aggregate" or point != "active_power_kw" or end_ts <= start_ts:
            return []
        minimum_gap = max(1, int(step_sec))
        selected: List[Dict[str, Any]] = []
        last_epoch: float | None = None
        for row in self._rows:
            epoch = float(row["epoch"])
            if epoch < start_ts or epoch > end_ts:
                continue
            if last_epoch is not None and epoch - last_epoch < minimum_gap:
                continue
            selected.append({"ts": row["ts"], "v": row["kW"]})
            last_epoch = epoch
        return selected

    def source_status(self) -> Dict[str, Any]:
        return {
            "mode": "canonical_dataset_replay",
            "artifact_id": self.path.name,
            "rows": len(self._rows),
            "measured": False,
            "production": False,
            "note": "Public integration dataset; replace PORT_DT_TELEMETRY_DATASET for port data",
        }
