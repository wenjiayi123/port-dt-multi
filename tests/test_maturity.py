from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.rl_training.datasets import PortDataset, dataset_quality_report, file_sha256, load_port_dataset, write_canonical_rows
from app.services.rl_training.model_registry import ModelRegistry
from app.services.rl_training.safety import assess_recommendation
from app.services.rl_training.statistics import bootstrap_summary
from app.services.rl_training.trainer import TrainingManager
from app.services.twin_schema.service import TwinSchemaService
from app.operations import configure_operations, cors_origins
from app.adapters import actuators as actuator_module
from app.adapters.actuators import Command, IdempotencyStore, PortSouthboundGateway


def canonical_rows(count: int = 96):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        yield {
            "timestamp": (start + timedelta(hours=index)).isoformat().replace("+00:00", "Z"),
            "base_load_kw": 1800 + index * 2,
            "throughput_teu": 130 + index % 20,
            "vessel_arrivals": 1 + index % 4,
            "tide_m": -1.0 + (index % 24) / 12,
            "price_per_kwh": 0.7 + (index % 24) / 100,
            "carbon_kg_per_kwh": 0.42 + (index % 5) / 100,
            "ambient_c": 25 + index % 7,
        }


GOVERNANCE = {
    "provenance_type": "test_fixture",
    "license": "test-only",
    "owner": "test",
    "timezone": "UTC",
    "intended_use": "unit testing",
}


class DataAndStatisticsMaturityTests(unittest.TestCase):
    def test_quality_gate_records_units_and_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("quality", canonical_rows(), GOVERNANCE, root)
            report = dataset_quality_report(load_port_dataset("quality", root))
            self.assertTrue(report["training_eligible"])
            self.assertEqual(report["columns"]["base_load_kw"]["unit"], "kW")
            self.assertEqual(report["missing_governance_metadata"], [])

    def test_quality_gate_blocks_physical_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("bad_physics", canonical_rows(), GOVERNANCE, root)
            dataset = load_port_dataset("bad_physics", root)
            values = dataset.values.copy()
            values[0, 0] = -1
            bad = PortDataset(dataset.dataset_id, dataset.path, dataset.timestamps, values, dataset.metadata)
            self.assertFalse(dataset_quality_report(bad)["training_eligible"])

    def test_quality_gate_blocks_missing_governance_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("ungoverned", canonical_rows(), {"license": "test"}, root)
            report = dataset_quality_report(load_port_dataset("ungoverned", root))
            self.assertFalse(report["training_eligible"])
            self.assertIn("owner", report["missing_governance_metadata"])

    def test_bootstrap_is_deterministic_and_reports_interval(self):
        left = bootstrap_summary([1, 2, 3, 4, 5], seed=7)
        right = bootstrap_summary([1, 2, 3, 4, 5], seed=7)
        self.assertEqual(left, right)
        self.assertLessEqual(left["ci_low"], left["mean"])
        self.assertGreaterEqual(left["ci_high"], left["mean"])


class SafetyEnvelopeTests(unittest.TestCase):
    def test_out_of_distribution_state_is_blocked_and_never_dispatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("safety", canonical_rows(), GOVERNANCE, root)
            dataset = load_port_dataset("safety", root)
            state = dict(next(canonical_rows()))
            state["base_load_kw"] = 999999
            state.update(soc=0.55, last_bess_kw=0)
            result = assess_recommendation(
                state=state,
                decoded_control={"bess_kw": 0, "service_factor": 1, "flexible_load_command": 0},
                dataset=dataset,
                demand_cap_kw=3000,
                bess_power_kw=900,
            )
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["dispatch_allowed"])
            self.assertIn("OUT_OF_DISTRIBUTION", {item["code"] for item in result["violations"]})

    def test_normalized_observation_cannot_claim_engineering_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_canonical_rows("safety", canonical_rows(), GOVERNANCE, root)
            result = assess_recommendation(
                state=None,
                decoded_control={},
                dataset=load_port_dataset("safety", root),
                demand_cap_kw=3000,
                bess_power_kw=900,
            )
            self.assertIsNone(result["within_software_envelope"])
            self.assertFalse(result["dispatch_allowed"])


class ModelRegistryTests(unittest.TestCase):
    def _make_run(self, root: Path, job_id: str = "run-1") -> tuple[ModelRegistry, dict]:
        run_dir = root / "runs" / job_id
        run_dir.mkdir(parents=True)
        model_path = run_dir / "model.zip"
        model_path.write_bytes(b"real-test-artifact")
        config = {"algorithm": "sac", "dataset_id": "port", "dataset_fingerprint": "a" * 64, "seed": 1}
        status = {"job_id": job_id, "status": "EVALUATED", "created_at": "2026-01-01T00:00:00Z"}
        manifest = {
            "implementation": "stable_baselines3.SAC",
            "model_sha256": file_sha256(model_path),
            "split": {"quality": {"training_eligible": True, "status": "pass"}},
        }
        evaluation = {
            "episodes": 10,
            "metrics": {"guardrail_violation_rate": 0.0},
            "uncertainty": {"reward": {"ci95_low": 0, "ci95_high": 1}},
            "evaluated_at": "2026-01-02T00:00:00Z",
        }
        for name, payload in (("config.json", config), ("status.json", status), ("manifest.json", manifest), ("evaluation.json", evaluation)):
            (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        registry = ModelRegistry(root / "runs", root / "model_registry.json")
        benchmark = {"algorithms": [{"id": "sac", "multi_seed_ready": True}]}
        return registry, benchmark

    def test_registry_verifies_artifact_and_writes_model_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry, _ = self._make_run(Path(tmp))
            record = registry.sync("run-1")
            self.assertTrue(record["artifact"]["verified"])
            self.assertTrue((Path(tmp) / "runs" / "run-1" / "MODEL_CARD.md").exists())

    def test_champion_alias_requires_opt_in_and_human_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry, benchmark = self._make_run(Path(tmp))
            registry.sync("run-1")
            blocked = registry.readiness("run-1", benchmark)
            self.assertFalse(blocked["ready_for_champion_alias"])
            with patch.dict(os.environ, {"PORT_DT_ALLOW_MODEL_PROMOTION": "1"}):
                result = registry.set_alias("run-1", "champion", approved_by="reviewer", reason="validated test", benchmark=benchmark)
            self.assertEqual(result["job_id"], "run-1")

    def test_registry_rejects_path_escape_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(Path(tmp) / "runs", Path(tmp) / "model_registry.json")
            for malicious in ("..", "../outside", "/tmp/outside", "run/child"):
                with self.subTest(job_id=malicious), self.assertRaises(ValueError):
                    registry.sync(malicious)


class TwinSchemaTests(unittest.TestCase):
    def test_example_graph_is_valid_and_has_provenance(self):
        service = TwinSchemaService()
        graph = json.loads(Path("config/twin_graph.example.json").read_text(encoding="utf-8"))
        result = service.validate_graph(graph)
        self.assertTrue(result["valid"], result["errors"])

    def test_calibration_rejects_failed_threshold(self):
        payload = json.loads(Path("config/twin_calibration.example.json").read_text(encoding="utf-8"))
        payload["metrics"]["soc_mae"] = 1.0
        result = TwinSchemaService.validate_calibration(payload)
        self.assertFalse(result["valid"])


class RuntimeHardeningTests(unittest.TestCase):
    def test_default_ui_has_no_static_readiness_or_synthetic_demand_claims(self):
        html = Path("app/ui/index.html").read_text(encoding="utf-8")
        self.assertNotIn("mock-ready", html)
        self.assertNotIn("future://", html)
        self.assertNotIn("需量预测（演示用合成", html)
        self.assertIn("/api/system/provenance", html)
        self.assertIn("window.__markOptionalModuleUnavailable", html)
        self.assertIn("等待接入港口 · 旧版实验制品未启用", html)
        self.assertIn("现场曲线须等待港口适配器接入", html)

    def test_default_public_apis_hide_local_paths_and_legacy_artifacts(self):
        from app import server as server_module

        production_app = server_module.app

        client = TestClient(production_app)
        for endpoint in (
            "/api/twin-models", "/api/rl/datasets", "/api/rl/train/status", "/api/rl/models",
            "/api/rl/port-profiles", "/api/rl/engine/capabilities",
            "/api/rl/integration/config", "/api/rl/integration/health", "/api/system/provenance",
            "/api/portviz/bootstrap", "/api/rl/business-benchmark",
        ):
            response = client.get(endpoint)
            self.assertEqual(response.status_code, 200, endpoint)
            self.assertNotIn(str(Path.cwd()), response.text)
            self.assertNotIn('"path":', response.text)
        self.assertEqual(client.get("/api/rl/model/agv_charge/kpi_cards.json").status_code, 404)
        self.assertEqual(client.get("/api/rl/artifacts/policy_evaluate_history.jsonl").status_code, 404)
        panel = client.get("/rl-panel")
        self.assertEqual(panel.status_code, 200)
        self.assertIn("/ui/adapters/rl_evidence_console.js", panel.text)
        self.assertIn('episode_hours: Math.max(1, horizon / 60)', panel.text)
        self.assertIn('select.value = "public_us_la_6min_v1"', panel.text)
        self.assertIn("syncSelectedDatasetContract", panel.text)
        self.assertEqual(client.get("/ui/adapters/rl_evidence_console.js").status_code, 200)
        strategies = client.get("/api/rl/strategies").json()
        self.assertEqual(strategies["source"], "verified_model_registry")
        self.assertFalse(strategies["generated_values"])
        heldout = {
            "job_id": "registered-test-job", "algorithm": "sac", "dataset_id": "dataset-a", "dataset_sha256": "a" * 64,
            "metrics": {"guardrail_violation_rate": 0.0}, "evaluation_protocol": {"holdout": "chronological_test_only"},
            "render": {"frame_count": 2, "frames": [
                {"timestamp": "2026-01-01T00:00:00Z", "baseline_kw": 100.0, "net_load_kw": 90.0},
                {"timestamp": "2026-01-01T01:00:00Z", "baseline_kw": 120.0, "net_load_kw": 105.0},
            ]},
        }
        with patch.object(server_module.TRAINING_MANAGER, "evaluate", return_value=heldout):
            evaluation = client.post("/api/rl/simulate", json={"strategy_id": "registered-test-job", "episodes": 5})
        self.assertEqual(evaluation.status_code, 200)
        self.assertEqual(evaluation.json()["mode"], "chronological_holdout_evaluation")
        self.assertFalse(evaluation.json()["production_dispatched"])
        self.assertFalse(evaluation.json()["summary"]["dispatch_ready"])
        capabilities = client.get("/api/actuators/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        self.assertTrue(capabilities.json()["two_person_confirmation_required"])
        flags = client.get("/api/system/provenance").json()["feature_flags"]
        self.assertFalse(flags["market_adapter_live"])
        self.assertFalse(flags["ais_tide_adapter_live"])
        self.assertTrue(capabilities.json()["requester_confirmer_must_differ"])
        self.assertEqual(capabilities.json()["audit_evidence"], "atomic_json_mode_0600")
        for endpoint in ("/api/rl/future/history", "/api/mas/simulate", "/api/xiaoyi/status", "/api/sailing/status"):
            self.assertEqual(client.get(endpoint).status_code, 404, endpoint)

    def test_production_has_no_wildcard_or_implicit_cors(self):
        with patch.dict(os.environ, {"PORT_DT_ENV": "production", "PORT_DT_CORS_ORIGINS": ""}):
            self.assertEqual(cors_origins(), [])

    def test_configured_cors_is_explicit(self):
        with patch.dict(os.environ, {"PORT_DT_ENV": "production", "PORT_DT_CORS_ORIGINS": "https://ops.example, https://audit.example"}):
            self.assertEqual(cors_origins(), ["https://ops.example", "https://audit.example"])

    def test_production_api_requires_a_long_configured_key(self):
        application = FastAPI()
        configure_operations(application)

        @application.get("/api/check")
        async def check():
            return {"ok": True}

        key = "a-valid-test-key-with-at-least-32-characters"
        with patch.dict(os.environ, {"PORT_DT_ENV": "production", "PORT_DT_API_KEYS": key}):
            client = TestClient(application)
            self.assertEqual(client.get("/api/check").status_code, 401)
            self.assertEqual(client.get("/api/check", headers={"X-API-Key": key}).status_code, 200)

    def test_privileged_mutation_requires_distinct_admin_key(self):
        application = FastAPI()
        configure_operations(application)

        @application.post("/api/rl/models/sync")
        async def privileged():
            return {"ok": True}

        operator = "operator-test-key-with-at-least-32-characters"
        admin = "admin-test-key-with-at-least-32-characters"
        with patch.dict(os.environ, {"PORT_DT_ENV": "production", "PORT_DT_API_KEYS": operator, "PORT_DT_ADMIN_API_KEYS": admin}):
            client = TestClient(application)
            self.assertEqual(client.post("/api/rl/models/sync", headers={"X-API-Key": operator}).status_code, 403)
            self.assertEqual(client.post("/api/rl/models/sync", headers={"X-API-Key": admin}).status_code, 200)

    def test_training_and_evaluation_capacity_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PORT_DT_MAX_CONCURRENT_TRAINING": "1", "PORT_DT_MAX_CONCURRENT_EVALUATION": "1"}):
            root = Path(tmp)
            data_root = root / "datasets"
            write_canonical_rows("capacity", canonical_rows(), GOVERNANCE, data_root)
            manager = TrainingManager(data_root, root / "runs", root / "benchmarks.json")
            manager.jobs["active"] = SimpleNamespace(status={"status": "RUNNING"})
            with self.assertRaisesRegex(ValueError, "training capacity reached"):
                manager.start({"algorithm": "sac", "dataset_id": "capacity", "total_steps": 64})
            manager.evaluation_slots.acquire()
            try:
                with self.assertRaisesRegex(ValueError, "evaluation capacity reached"):
                    manager.evaluate("missing", 5)
            finally:
                manager.evaluation_slots.release()

    def test_training_manager_rejects_path_escape_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TrainingManager(root / "datasets", root / "runs", root / "benchmarks.json")
            for malicious in ("..", "../outside", "/tmp/outside", "run/child"):
                with self.subTest(job_id=malicious), self.assertRaises(ValueError):
                    manager.run_dir(malicious)

    def test_training_config_derives_v2_contract_from_dataset_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "datasets"
            write_canonical_rows(
                "profiled",
                canonical_rows(),
                {
                    **GOVERNANCE,
                    "port_profile_id": "sgsin_public_replay_v2",
                    "environment_version": "port_ops_v2",
                },
                data_root,
            )
            manager = TrainingManager(
                data_root, root / "runs", root / "benchmarks.json"
            )
            config = manager.validate_config(
                {
                    "algorithm": "sac",
                    "dataset_id": "profiled",
                    "total_steps": 64,
                    "episode_hours": 12,
                }
            )
            self.assertEqual(config["port_profile_id"], "sgsin_public_replay_v2")
            self.assertEqual(config["environment_version"], "port_ops_v2")
            self.assertEqual(config["observation_dimensions"], 37)
            self.assertEqual(config["action_dimensions"], 5)
            self.assertEqual(config["episode_steps"], 9)

    def test_benchmark_comparison_requires_one_dataset_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_path = root / "benchmarks.json"
            runs = []
            for dataset_id in ("left", "right"):
                for seed in (42, 142, 242):
                    runs.append(
                        {
                            "algorithm": "sac",
                            "dataset_id": dataset_id,
                            "seed": seed,
                            "total_steps": 10_000,
                            "evidence_label": "RL_HELD_OUT_EVALUATION",
                            "metrics": {"reward": float(seed)},
                        }
                    )
            benchmark_path.write_text(
                json.dumps({"runs": runs}), encoding="utf-8"
            )
            manager = TrainingManager(
                root / "datasets", root / "runs", benchmark_path
            )
            unscoped = manager.benchmark_summary()
            sac_unscoped = next(
                item for item in unscoped["algorithms"] if item["id"] == "sac"
            )
            self.assertFalse(sac_unscoped["multi_seed_ready"])
            self.assertEqual(sac_unscoped["metrics"], {})
            scoped = manager.benchmark_summary("left")
            sac_scoped = next(
                item for item in scoped["algorithms"] if item["id"] == "sac"
            )
            self.assertTrue(sac_scoped["multi_seed_ready"])
            self.assertEqual(sac_scoped["claim_eligible_runs"], 3)


class ActuatorGatewayTests(unittest.TestCase):
    def test_staging_is_constrained_idempotent_and_requires_distinct_confirmer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit"
            audit.mkdir()
            config = root / "actuators.json"
            config.write_text(json.dumps({
                "enabled": True,
                "whitelist": {"BESS-1": ["set"]},
                "routing": {"asset": {"BESS-1": {"channel": "dry_run"}}, "type": {}},
                "security": {"confirmation_token_env": "TEST_SECOND_CHANNEL", "require_two_channel": True, "require_constraints": True},
                "constraints": {"asset": {"BESS-1": {"set": {"power_kw": {"required": True, "min": -100, "max": 100}}}}, "type": {}},
            }), encoding="utf-8")
            token = "a-separate-test-confirmation-token-123456"
            with patch.object(actuator_module, "AUDIT_DIR", str(audit)), patch.dict(os.environ, {"PORT_DT_ENABLE_ACTUATOR_DRY_RUN": "1", "TEST_SECOND_CHANNEL": token}):
                gateway = PortSouthboundGateway(str(config))
                gateway.idem = IdempotencyStore(str(audit))
                out_of_bounds = gateway.dispatch(Command("BESS-1", "bess", "set", {"power_kw": 101}, requested_by="alice", two_channel_required=True))
                self.assertEqual(out_of_bounds.message, "site_constraints_failed")
                command = Command("BESS-1", "bess", "set", {"power_kw": 80}, requested_by="alice", idempotency_key="same-command", two_channel_required=True)
                staged = gateway.dispatch(command)
                self.assertEqual(staged.status, "PENDING")
                repeated = gateway.dispatch(command)
                self.assertEqual(repeated.status, "PENDING")
                self.assertEqual(repeated.command_id, staged.command_id)
                self.assertTrue(Path(repeated.evidence_path or "").name.startswith("guard-"))
                self.assertEqual(gateway.confirm(staged.command_id, "alice", token).message, "confirmer_must_differ_from_requester")
                self.assertEqual(gateway.confirm(staged.command_id, "bob", "wrong").message, "second_channel_token_invalid")
                self.assertEqual(gateway.confirm(staged.command_id, "bob", token).status, "EXECUTED")
                self.assertEqual(gateway.confirm(staged.command_id, "bob", token).message, "pending_evidence_not_found")
                os.environ.pop("PORT_DT_ENABLE_ACTUATOR_DRY_RUN")
                failed_rollback = gateway.rollback(staged.command_id, "test retry", "carol", token)
                self.assertEqual(failed_rollback.status, "FAILED")
                os.environ["PORT_DT_ENABLE_ACTUATOR_DRY_RUN"] = "1"
                successful_rollback = gateway.rollback(staged.command_id, "approved retry", "carol", token)
                self.assertEqual(successful_rollback.status, "ROLLEDBACK")
                evidence_text = "\n".join(path.read_text(encoding="utf-8") for path in audit.glob("*.json"))
                self.assertNotIn(token, evidence_text)


if __name__ == "__main__":
    unittest.main()
