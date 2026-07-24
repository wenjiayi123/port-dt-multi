from __future__ import annotations

import csv
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.services.rl_training.datasets import CANONICAL_COLUMNS, import_dataset, load_port_dataset, write_canonical_rows
from app.services.rl_training.environment import PortOperationsEnv


def rows(count: int = 96):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        yield {
            "timestamp": (start + timedelta(hours=index)).isoformat().replace("+00:00", "Z"),
            "base_load_kw": 2000 + 100 * math.sin(index / 4),
            "throughput_teu": 150 + index % 12,
            "vessel_arrivals": 2 + index % 2,
            "tide_m": math.sin(index / 2),
            "price_per_kwh": 0.8 + (index % 24 >= 17) * 0.4,
            "carbon_kg_per_kwh": 0.47,
            "ambient_c": 29.0,
        }


class DatasetTests(unittest.TestCase):
    def test_chronological_split_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("port_a", rows(), {"license": "test"}, root)
            dataset = load_port_dataset("port_a", root)
            train, test = dataset.split(0.2)
            self.assertEqual(train.stop, test.start)
            self.assertLess(datetime.fromisoformat(dataset.timestamps[train.stop - 1].replace("Z", "+00:00")), datetime.fromisoformat(dataset.timestamps[test.start].replace("Z", "+00:00")))
            self.assertEqual(len(dataset.fingerprint), 64)

    def test_import_rejects_missing_canonical_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bad.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS[:-1])
                writer.writeheader()
            with self.assertRaisesRegex(ValueError, "missing mapped columns"):
                import_dataset(
                    source,
                    "bad",
                    metadata={"license": "test", "owner": "test", "timezone": "UTC", "intended_use": "test"},
                    data_root=root / "out",
                )


class EnvironmentTests(unittest.TestCase):
    def test_training_cannot_render_or_collect_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("port_a", rows(), {"license": "test"}, root)
            dataset = load_port_dataset("port_a", root)
            train, _ = dataset.split()
            with self.assertRaisesRegex(ValueError, "must not collect render traces"):
                PortOperationsEnv(dataset, train, training=True, record_trace=True)
            env = PortOperationsEnv(dataset, train, training=True, episode_steps=12)
            env.reset(seed=3)
            env.step(np.zeros(3, dtype=np.float32))
            self.assertEqual(env.trace, [])
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                env.render()
            self.assertEqual(env.render_calls, 1)

    def test_evaluation_trace_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("port_a", rows(), {"license": "test"}, root)
            dataset = load_port_dataset("port_a", root)
            _, test = dataset.split()
            env = PortOperationsEnv(dataset, test, training=False, record_trace=True, episode_steps=8)
            env.reset(seed=3)
            env.step(np.zeros(3, dtype=np.float32))
            self.assertEqual(len(env.trace), 1)

    def test_inference_projection_matches_step_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("port_a", rows(), {"license": "test"}, root)
            dataset = load_port_dataset("port_a", root)
            _, test = dataset.split()
            env = PortOperationsEnv(dataset, test, training=False, episode_steps=8)
            projected = env.project_control(np.ones(3, dtype=np.float32), soc=0.55, last_bess_kw=0.0)
            self.assertLessEqual(projected["bess_kw"], env.bess_power_kw)
            self.assertEqual(projected["flexible_load_command"], 0.6)
            self.assertTrue(projected["projection_applied"])


if __name__ == "__main__":
    unittest.main()
