"""Run all four RL algorithms and the MPC baseline on a disposable dataset."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.rl_training.datasets import write_canonical_rows
from app.services.rl_training.trainer import ALGORITHMS, TrainingManager


def build_rows(count: int = 144):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        hour = index % 24
        yield {
            "timestamp": (start + timedelta(hours=index)).isoformat().replace("+00:00", "Z"),
            "base_load_kw": 2200 + 260 * math.sin(2 * math.pi * hour / 24),
            "throughput_teu": 180 + 35 * math.sin(2 * math.pi * (hour - 4) / 24),
            "vessel_arrivals": 2 + (index % 3),
            "tide_m": 1.1 * math.sin(2 * math.pi * index / 12.42),
            "price_per_kwh": 1.3 if 17 <= hour < 22 else 0.8,
            "carbon_kg_per_kwh": 0.48,
            "ambient_c": 29 + 3 * math.sin(2 * math.pi * (hour - 8) / 24),
        }


def wait_for_job(manager: TrainingManager, job_id: str, timeout: float = 90.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status(job_id)
        if status["status"] in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
            return status
        time.sleep(0.05)
    raise TimeoutError(job_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    if args.steps < 64:
        parser.error("--steps must be at least 64")

    with tempfile.TemporaryDirectory(prefix="port-rl-smoke-") as tmp:
        root = Path(tmp)
        data_root = root / "datasets"
        write_canonical_rows(
            "smoke_port",
            build_rows(),
            {
                "provenance_type": "deterministic_test_fixture",
                "license": "test-only",
                "owner": "ci",
                "timezone": "UTC",
                "intended_use": "automated smoke testing",
            },
            data_root,
        )
        manager = TrainingManager(data_root, root / "runs", root / "benchmarks.json")
        results = []
        for algorithm in ALGORITHMS:
            started = manager.start(
                {
                    "algorithm": algorithm,
                    "dataset_id": "smoke_port",
                    "total_steps": args.steps,
                    "episode_steps": 12,
                    "batch_size": 16,
                    "seed": 7,
                    "test_ratio": 0.2,
                    "demand_cap_kw": 2600,
                }
            )
            status = wait_for_job(manager, started["job_id"])
            if status["status"] != "COMPLETED":
                raise RuntimeError(f"{algorithm} failed: {status}")
            if status.get("rendering", {}).get("render_calls") != 0:
                raise AssertionError(f"{algorithm} rendered during training")
            if algorithm != "mpc" and manager.history(started["job_id"])["count"] < 1:
                raise AssertionError(f"{algorithm} did not emit optimizer history")
            evaluation = manager.evaluate(started["job_id"], episodes=1)
            if evaluation["split"] != "chronological_test_holdout_only":
                raise AssertionError("evaluation did not use holdout")
            if evaluation["render"]["frame_count"] < 1:
                raise AssertionError("evaluation did not emit render frames")
            inference = manager.predict(
                started["job_id"],
                {
                    "state": {
                        "base_load_kw": 2300.0,
                        "throughput_teu": 190.0,
                        "vessel_arrivals": 3.0,
                        "tide_m": 0.4,
                        "price_per_kwh": 1.1,
                        "carbon_kg_per_kwh": 0.48,
                        "ambient_c": 31.0,
                        "hour": 18,
                        "soc": 0.58,
                        "queue": 10.0,
                        "last_bess_kw": 0.0,
                    }
                },
            )
            if inference.get("rendered") is not False or not inference.get("decoded_control"):
                raise AssertionError(f"{algorithm} inference contract failed")
            results.append(
                {
                    "algorithm": algorithm,
                    "implementation": evaluation["implementation"],
                    "training_render_calls": status.get("rendering", {}).get("render_calls"),
                    "history_records": manager.history(started["job_id"])["count"],
                    "evaluation_frames": evaluation["render"]["frame_count"],
                    "inference_control": inference["decoded_control"],
                    "reward": evaluation["metrics"]["reward"],
                }
            )
        print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
