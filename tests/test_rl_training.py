from __future__ import annotations

import csv
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.services.rl_training.datasets import (
    CANONICAL_COLUMNS,
    FACTOR_COLUMNS,
    import_dataset,
    load_port_dataset,
    write_canonical_rows,
    write_extended_rows,
)
from app.services.rl_training.environment import PortOperationsEnv
from app.services.rl_training.profiles import load_profile
from app.services.rl_training.trainer import ALGORITHMS


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
    def test_seven_controller_contract(self):
        self.assertEqual(list(ALGORITHMS), ["sac", "ppo", "td3", "dqn", "a2c", "tqc", "mpc"])
        self.assertEqual(sum(spec.trainable for spec in ALGORITHMS.values()), 6)
        self.assertEqual(ALGORITHMS["mpc"].family, "Control")

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

    def test_v2_factor_masks_and_five_action_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enriched_rows = []
            for row in rows():
                enriched_rows.append(
                    {
                        **row,
                        "wind_speed_mps": 4.2,
                        "berth_occupancy_ratio": 0.72,
                        "yard_occupancy_ratio": 0.68,
                        "channel_congestion_ratio": 0.44,
                    }
                )
            write_extended_rows(
                "port_v2",
                enriched_rows,
                {
                    "provenance_type": "verified_test",
                    "license": "test",
                    "owner": "test",
                    "timezone": "UTC",
                    "intended_use": "test",
                    "environment_version": "port_ops_v2",
                },
                root,
            )
            dataset = load_port_dataset("port_v2", root)
            train, _ = dataset.split()
            env = PortOperationsEnv(
                dataset,
                train,
                training=True,
                episode_steps=12,
                environment_version="port_ops_v2",
                port_profile=load_profile("sgsin_public_replay_v2"),
            )
            observation, _ = env.reset(seed=3)
            self.assertEqual(observation.shape, (13 + 2 * len(FACTOR_COLUMNS),))
            self.assertEqual(env.action_space.shape, (5,))
            _, _, _, _, info = env.step(np.zeros(5, dtype=np.float32))
            self.assertTrue(info["factor_availability"]["wind_speed_mps"])
            self.assertFalse(info["factor_availability"]["visibility_km"])
            self.assertIn("berth_priority", info)

    def test_time_features_follow_timestamp_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            frequent_rows = []
            for index, row in enumerate(rows(400)):
                frequent_rows.append(
                    {
                        **row,
                        "timestamp": (
                            start + timedelta(minutes=6 * index)
                        ).isoformat().replace("+00:00", "Z"),
                    }
                )
            write_canonical_rows("six_minute", frequent_rows, {"license": "test"}, root)
            dataset = load_port_dataset("six_minute", root)
            train, _ = dataset.split()
            env = PortOperationsEnv(dataset, train, training=True, episode_steps=4)
            first, _ = env.reset(options={"start_index": 0})
            next_day, _ = env.reset(options={"start_index": 240})
            np.testing.assert_allclose(first[:2], next_day[:2], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
