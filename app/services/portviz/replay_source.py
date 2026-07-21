from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetReplaySource:
    """Deterministic PortViz frames driven by the canonical public/port dataset.

    Entity coordinates are a documented visual projection of aggregate rows; they
    are not represented as measured vehicle tracks. For measured tracks use
    ``JsonLinesPortSource`` with ``PORTVIZ_MODE=real``.
    """

    def __init__(self, overrides: Dict[str, Any], dataset_path: str) -> None:
        self.overrides = overrides
        candidate = Path(dataset_path)
        self.path = candidate if candidate.is_absolute() else Path(__file__).resolve().parents[3] / candidate
        if not self.path.exists():
            raise FileNotFoundError(f"PortViz replay dataset not found: {self.path}")
        with self.path.open("r", encoding="utf-8-sig", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows:
            raise ValueError(f"PortViz replay dataset is empty: {self.path}")
        loads = [float(row["base_load_kw"]) for row in self.rows]
        throughputs = [float(row["throughput_teu"]) for row in self.rows]
        self.load_min, self.load_max = min(loads), max(loads)
        self.teu_min, self.teu_max = min(throughputs), max(throughputs)
        self.index = 0
        self.world = overrides.get("world") or {"W": 1600, "H": 900}
        self.lanes = overrides.get("lanes") or [
            [{"x": 140, "y": 720}, {"x": 1460, "y": 720}],
            [{"x": 140, "y": 640}, {"x": 1460, "y": 640}],
            [{"x": 140, "y": 560}, {"x": 1460, "y": 560}],
        ]
        self.yards = overrides.get("yards") or [
            {"x": 160 + c * 160, "y": 280 + r * 80, "w": 140, "h": 56}
            for r in range(4) for c in range(8)
        ]
        self.berth = overrides.get("berth") or {"x": 120, "y": 60, "w": 1360, "h": 24}
        self.qcs = overrides.get("qcs") or [{"x": 200 + i * 220, "y": 110} for i in range(6)]
        self.ycs = overrides.get("ycs") or [{"x": 220 + (i % 5) * 260, "y": 320 + (i // 5) * 160} for i in range(10)]
        self.agv_n = int(overrides.get("agv_n", 26))
        self.truck_n = int(overrides.get("truck_n", 12))

    @staticmethod
    def _scale(value: float, low: float, high: float) -> float:
        return max(0.0, min(1.0, (value - low) / max(1e-9, high - low)))

    def get_bootstrap(self) -> Dict[str, Any]:
        original_meta = dict(self.overrides.get("meta") or {})
        return {
            "world": self.world,
            "lanes": self.lanes,
            "yards": self.yards,
            "berth": self.berth,
            "qcs": self.qcs,
            "ycs": self.ycs,
            "meta": {
                **original_meta,
                "source_mode": "dataset_replay",
                "dataset_artifact": self.path.name,
                "provenance": "aggregate dataset projected deterministically; positions are visual derivatives",
                "measured_entity_tracks": False,
            },
        }

    def next_frame(self, since: Optional[int] = None) -> Dict[str, Any]:
        row = self.rows[self.index % len(self.rows)]
        idx = self.index
        self.index += 1
        load = float(row["base_load_kw"])
        teu = float(row["throughput_teu"])
        arrivals = float(row["vessel_arrivals"])
        load_factor = self._scale(load, self.load_min, self.load_max)
        teu_factor = self._scale(teu, self.teu_min, self.teu_max)
        agv = [
            {"lane": i % max(1, len(self.lanes)), "s": float((i * 17.0 + idx * (0.8 + teu_factor)) % 100), "alarm": False}
            for i in range(self.agv_n)
        ]
        qc = [
            {"busy": i < max(1, min(len(self.qcs), round(arrivals))), "trolley": float((0.11 * i + 0.013 * idx) % 1.0)}
            for i in range(len(self.qcs))
        ]
        yc = [{"busy": i / max(1, len(self.ycs) - 1) <= teu_factor} for i in range(len(self.ycs))]
        trucks = [
            {"x": float(200 + ((i * 103 + idx * 7) % 1160)), "y": float(600 + ((i * 31 + idx * 3) % 120))}
            for i in range(self.truck_n)
        ]
        hotspots = [] if load_factor < 0.75 else [{"x": 900, "y": 650, "r": 28 + 36 * load_factor}]
        return {
            "ts": int(time.time() * 1000),
            "source_timestamp": row["timestamp"],
            "agv": agv,
            "qc": qc,
            "yc": yc,
            "tr": trucks,
            "hotspots": hotspots,
            "vessels": [{"berth": 0, "progress": float((idx * 0.002) % 1.0), "len": 900.0}],
            "metrics": {"base_load_kw": load, "throughput_teu": teu, "vessel_arrivals": arrivals},
            "meta": {"source_mode": "dataset_replay", "measured_entity_tracks": False},
        }


class JsonLinesPortSource:
    """Measured/replayed port frames from a validated JSONL adapter contract."""

    REQUIRED_FRAME_ARRAYS = ("agv", "qc", "yc", "tr")

    def __init__(self, overrides: Dict[str, Any], frames_path: str) -> None:
        self.overrides = overrides
        candidate = Path(frames_path)
        self.path = candidate if candidate.is_absolute() else Path(__file__).resolve().parents[3] / candidate
        if not self.path.exists():
            raise FileNotFoundError(f"PORTVIZ_FRAME_PATH not found: {self.path}")
        self.frames: List[Dict[str, Any]] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            frame = json.loads(line)
            missing = [key for key in self.REQUIRED_FRAME_ARRAYS if not isinstance(frame.get(key), list)]
            if missing:
                raise ValueError(f"{self.path}:{line_no} missing frame arrays: {', '.join(missing)}")
            self.frames.append(frame)
        if not self.frames:
            raise ValueError(f"no valid frames in {self.path}")
        self.index = 0

    def get_bootstrap(self) -> Dict[str, Any]:
        required = ("world", "lanes", "yards", "berth", "qcs", "ycs")
        missing = [key for key in required if key not in self.overrides]
        if missing:
            raise ValueError(f"PORTVIZ_CONFIG missing real-source geometry: {', '.join(missing)}")
        return {**{key: self.overrides[key] for key in required}, "meta": {**dict(self.overrides.get("meta") or {}), "source_mode": "real_jsonl_adapter", "frames_artifact": self.path.name, "measured_entity_tracks": True}}

    def next_frame(self, since: Optional[int] = None) -> Dict[str, Any]:
        frame = dict(self.frames[self.index % len(self.frames)])
        self.index += 1
        frame.setdefault("ts", int(time.time() * 1000))
        frame.setdefault("hotspots", [])
        frame.setdefault("vessels", [])
        frame["meta"] = {**dict(frame.get("meta") or {}), "source_mode": "real_jsonl_adapter", "measured_entity_tracks": True}
        return frame
