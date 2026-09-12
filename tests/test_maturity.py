from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.rl_training.datasets import PortDataset, dataset_quality_report, file_sha256, load_port_dataset, safe_dataset_id, write_canonical_rows
from app.services.rl_training.model_registry import ModelRegistry
from app.services.rl_training.safety import assess_recommendation
from app.services.rl_training.statistics import bootstrap_summary
from app.services.rl_training.trainer import TrainingManager
from app.services.twin_schema.service import TwinSchemaService
from app.services.site_twin_calibration import SiteTwinCalibrationService
from app.services.site_shadow_acceptance import SiteShadowAcceptanceService
from app.services.site_execution_acceptance import SiteExecutionAcceptanceService
from app.services.port_call_collaboration import PortCallCollaborationService
from tests.test_port_call_collaboration import authorized_bundle as canonical_collaboration_bundle
from app.services.maritime_interoperability import MaritimeInteroperabilityService
from tests.test_maritime_interoperability import authorized_bundle as canonical_interoperability_bundle
from app.services.forecast_uncertainty import ForecastUncertaintyService
from tests.test_forecast_uncertainty import authorized_bundle as canonical_forecast_bundle
from app.services.business_benefit_attribution import BusinessBenefitAttributionService
from tests.test_business_benefit_attribution import authorized_bundle as canonical_benefit_bundle
from app.services.end_to_end_coordination import EndToEndCoordinationService
from tests.test_end_to_end_coordination import authorized_bundle as canonical_coordination_bundle
from app.services.production_continuity import ProductionContinuityService
from tests.test_production_continuity import authorized_bundle as canonical_continuity_bundle
from app.services.operating_model_governance import OperatingModelGovernanceService
from tests.test_operating_model_governance import authorized_bundle as canonical_operating_model_bundle
from app.operations import RATE_LIMITER, configure_operations, cors_origins, readiness_report
from app.adapters import actuators as actuator_module
from app.adapters.actuators import Command, IdempotencyStore, PortSouthboundGateway
from app.services.rl_suite import rl_admin
from app.services.sailing_simulator import api as sailing_api


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


def canonical_shadow_bundle(site_id: str = "test-terminal-01"):
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    cycles = []
    for day in range(7):
        for slot in range(5):
            index = day * 5 + slot
            started = start + timedelta(days=day, hours=slot * 3)
            energy = 1800.0 + slot * 40.0
            throughput = 120.0 + slot * 5.0
            delay = 30.0 + slot
            cycles.append({
                "cycle_id": f"SHADOW.CYCLE.{index:04d}",
                "started_at": started.isoformat(),
                "ended_at": (started + timedelta(minutes=45)).isoformat(),
                "asset_group": "terminal.power.aggregate",
                "scenario": "normal_operations" if slot < 4 else "peak_berthing",
                "data_quality_passed": True,
                "incumbent": {
                    "actual_energy_kwh": energy,
                    "actual_throughput_teu": throughput,
                    "actual_delay_minutes": delay,
                    "source_reference": f"TEST.TOS.METER.{index:04d}",
                },
                "candidate": {
                    "projected_energy_kwh": energy * 0.98,
                    "projected_throughput_teu": throughput,
                    "projected_delay_minutes": delay * 0.98,
                    "recommendation_receipt_id": f"TEST.DECISION.{index:04d}",
                    "recommendation_available": True,
                    "action_feasible": True,
                    "recommendation_latency_ms": 500.0,
                },
                "guardrail": {"violation_count": 0, "codes": []},
                "review": {"disposition": "accepted"},
                "side_effect": False,
            })
    return {
        "schema_version": "site_shadow_observation.v1",
        "site_id": site_id,
        "run_id": "test-shadow-run-2026-02",
        "source": {
            "source_system": "authorized-test-shadow-export",
            "owner": "test-terminal-operator",
            "license": "authorized-test-shadow",
            "timezone": "UTC",
            "extracted_at": (start + timedelta(days=8)).isoformat(),
            "evidence_class": "authorized_site_shadow_export",
        },
        "policy": {
            "candidate_policy_version": "test-candidate-policy-v1",
            "incumbent_policy_version": "test-incumbent-policy-v1",
            "calibration_evidence_digest": "a" * 64,
            "twin_graph_digest": "b" * 64,
            "recommendation_only": True,
        },
        "cycles": cycles,
    }


def canonical_rollback_drill():
    return {
        "command_id": "ROLLBACK-TEST-001",
        "results": [
            {"at": 1000.0, "ok": True, "detail": {"acknowledged": True}},
            {"at": 1010.0, "ok": True, "detail": {"restored": True}, "rollback": True},
        ],
        "timestamps": {"executed_at": 1000.0, "rolledback_at": 1010.0},
        "approvals": [
            {"type": "rollback", "by": "site-duty-manager", "reason": "readiness contract test"},
        ],
    }


def canonical_execution_config(site_id: str = "test-terminal-01", *, authorized: bool = True):
    return {
        "schema_version": "site_actuator_config.v2",
        "site_id": site_id,
        "mode": "authorized_site" if authorized else "contract_test",
        "enabled": True,
        "whitelist": {"BESS-1": ["set"]},
        "routing": {
            "asset": {
                "BESS-1": {
                    "channel": "http" if authorized else "dry_run",
                    "endpoint": "https://control.test.example/api" if authorized else "contract-only",
                    "interlock_id": "INTERLOCK.TEST.01",
                    "readback_mode": "signed_gateway_receipt" if authorized else "contract_receipt",
                    "rollback_supported": True,
                }
            },
            "type": {},
        },
        "security": {
            "confirmation_token_env": "TEST_SECOND_CHANNEL",
            "require_two_channel": True,
            "require_constraints": True,
            "require_verified_readback": True,
            "require_independent_interlock": True,
            "require_rollback": True,
            "command_ttl_seconds": 60,
        },
        "constraints": {
            "asset": {
                "BESS-1": {
                    "set": {"power_kw": {"required": True, "min": -100, "max": 100}}
                }
            },
            "type": {},
        },
    }


def canonical_execution_bundle(site_id: str = "test-terminal-01", *, authorized: bool = True):
    scenarios = (
        "safe_command_readback",
        "out_of_bounds_block",
        "expired_command_block",
        "duplicate_command_block",
        "lost_acknowledgement_block",
        "independent_interlock_trip",
        "emergency_stop",
        "rollback_restore",
    )
    start = datetime(2026, 2, 20, tzinfo=timezone.utc)
    tests = []
    for index, scenario in enumerate(scenarios):
        began = start + timedelta(minutes=index * 10)
        tests.append({
            "test_id": f"EXEC.TEST.{index:02d}",
            "scenario": scenario,
            "asset_id": "BESS-1",
            "action": "set",
            "started_at": began.isoformat(),
            "ended_at": (began + timedelta(minutes=2)).isoformat(),
            "passed": True,
            "equipment_command_sent": bool(
                authorized and scenario in {"safe_command_readback", "rollback_restore"}
            ),
            "readback_verified": scenario in {"safe_command_readback", "rollback_restore"},
            "rollback_verified": scenario == "rollback_restore",
            "interlock_verified": scenario in {"independent_interlock_trip", "emergency_stop"},
            "requester": "test-command-requester",
            "confirmer": "test-command-confirmer",
            "source_reference": f"TEST.EXECUTION.{index:02d}",
        })
    return {
        "schema_version": "site_execution_commissioning_dataset.v1",
        "site_id": site_id,
        "run_id": "test-execution-commissioning-2026-02",
        "source": {
            "source_system": "authorized-test-control-gateway" if authorized else "browser-contract-fixture",
            "owner": "test-terminal-control-owner",
            "license": "authorized-test-execution" if authorized else "contract-test-only",
            "timezone": "UTC",
            "extracted_at": "2026-02-21T00:00:00Z",
            "evidence_class": "authorized_site_commissioning_export" if authorized else "contract_test_only",
        },
        "shadow_acceptance_digest": "c" * 64,
        "actuator_config": canonical_execution_config(site_id, authorized=authorized),
        "tests": tests,
    }


GOVERNANCE = {
    "provenance_type": "test_fixture",
    "license": "test-only",
    "owner": "test",
    "timezone": "UTC",
    "intended_use": "unit testing",
}


class DataAndStatisticsMaturityTests(unittest.TestCase):
    def test_dataset_identifier_rejects_traversal_and_normalization_collisions(self):
        self.assertEqual(safe_dataset_id("public_port_ops_v1"), "public_port_ops_v1")
        for malicious in ("../outside", "/tmp/outside", "name.csv", "name space", "中文"):
            with self.subTest(dataset_id=malicious), self.assertRaises(ValueError):
                safe_dataset_id(malicious)

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
    def test_sailing_launch_ignores_request_scene_and_rejects_unknown_preset(self):
        safe_status = {"launchable": True}
        safe_cfg = {"godot_executable": "/opt/godot", "project_path": "/srv/sailing"}
        with patch.object(sailing_api, "sailing_status", return_value=safe_status), patch.object(sailing_api, "_sailing_cfg", return_value=safe_cfg), patch.dict(os.environ, {"PORT_DT_ENABLE_DESKTOP_INTEGRATIONS": "1"}):
            preview = sailing_api.launch_sailing_simulator(
                {"preset": "main_scene", "scene": "--editor", "source": "../../unsafe"},
                dry_run=True,
            )
            self.assertEqual(preview["status"], "ready_to_launch")
            self.assertEqual(preview["scene"], sailing_api.MAIN_SCENE)
            self.assertNotIn("--editor", preview["command_artifacts"])
            blocked = sailing_api.launch_sailing_simulator({"preset": "--path"}, dry_run=True)
            self.assertEqual(blocked["status"], "failed")

    def test_default_ui_has_no_static_readiness_or_synthetic_demand_claims(self):
        html = Path("app/ui/index.html").read_text(encoding="utf-8")
        self.assertNotIn("mock-ready", html)
        self.assertNotIn("future://", html)
        self.assertNotIn("需量预测（演示用合成", html)
        self.assertIn("/api/system/provenance", html)
        self.assertIn("window.__markOptionalModuleUnavailable", html)
        self.assertIn("等待接入港口 · 旧版实验制品未启用", html)
        self.assertIn("现场曲线须等待港口适配器接入", html)
        self.assertIn("max-height:min(520px, calc(100vh - 300px))", html)
        self.assertIn("overflow-y:auto", html)
        self.assertIn("TWIN_POWER_DEVICE_CACHE", html)
        self.assertIn("TWIN_REFRESH_IN_FLIGHT", html)
        for route in ("/ops-copilot", "/rl-panel", "/integration-hub", "/rl_future/rl_future_panel.html", "/docs"):
            self.assertIn(f'href="{route}" data-route="{route}"', html)
        self.assertIn('href="/v3?from=home" data-route="/v3?from=home"', html)
        self.assertNotIn('<button class="nav-trigger is-route"', html)
        sprite = Path("app/ui/adapters/xiaoyi_sprite.js").read_text(encoding="utf-8")
        self.assertIn("overlapsProtectedNavigation", sprite)
        self.assertIn("restoreSafeDock(root)", sprite)
        self.assertIn("ensureCharacterReady(root)", sprite)
        self.assertIn('fetchpriority="high"', sprite)
        self.assertIn('localStorage.removeItem(STORAGE_KEY)', sprite)
        recovery = Path("app/ui/adapters/runtime_recovery.js").read_text(encoding="utf-8")
        self.assertIn("/health/live", recovery)
        self.assertIn("本地数据服务连接中断", recovery)
        self.assertIn("window.location.reload()", recovery)
        self.assertIn("window.__portDtTrackEvidenceLoad", recovery)
        self.assertIn("window.__portDtDeferHeavyRender", recovery)
        self.assertIn("restoreActiveModuleAnchor", recovery)
        self.assertIn('role="progressbar"', recovery)
        self.assertIn("本地证据加载进度", recovery)
        self.assertIn("evidenceLoadsPending()", recovery)
        self.assertIn("if(!immediate && evidenceLoadsPending()) return", recovery)
        self.assertIn("读取后端证据", recovery)
        self.assertIn("loadYardCraneV3", html)
        self.assertIn("syncLocalTwinAnimation", html)
        self.assertIn("syncPortvizAnimation", html)
        self.assertIn('href="/v3?from=home"', html)
        v3_html = Path("app/ui/v3/index.html").read_text(encoding="utf-8")
        v3_css = Path("app/ui/v3/v3.css").read_text(encoding="utf-8")
        self.assertIn("v3.css?v=3.2.2-controls", v3_html)
        self.assertIn('id="returnHome"', v3_html)
        self.assertIn("v3.js?v=3.2.4-null-evidence", v3_html)
        v3_js = Path("app/ui/v3/v3.js").read_text(encoding="utf-8")
        self.assertIn("function returnToHome(event)", v3_js)
        self.assertIn("window.location.assign('/')", v3_js)
        self.assertNotIn("window.history.back()", v3_js)
        self.assertIn(".gate-action{display:block;min-height:32px", v3_css)
        self.assertIn(".lineage-action{display:block;min-height:34px", v3_css)
        ops_copilot = Path("app/ui/ops_copilot.html").read_text(encoding="utf-8")
        self.assertIn("grid-template-columns:minmax(0,1fr);gap:8px", ops_copilot)
        self.assertIn(".context-item>div{min-width:0}", ops_copilot)
        self.assertNotIn('data-app-path="${path}"', html)
        self.assertIn("riskList.replaceChildren()", html)
        self.assertIn("tb.replaceChildren()", html)

    def test_default_public_apis_hide_local_paths_and_legacy_artifacts(self):
        from app import server as server_module

        production_app = server_module.app

        client = TestClient(production_app)
        home = client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn('id="xiaoyi-character-preload"', home.text)
        self.assertIn('/ui/adapters/runtime_recovery.js?v=20260912-page-loading-v6', home.text)
        self.assertIn('/static/module_loading.js?v=20260912-1', home.text)
        loading_adapter = client.get('/static/module_loading.js')
        self.assertEqual(loading_adapter.status_code, 200)
        self.assertIn('javascript', loading_adapter.headers.get('content-type', ''))
        self.assertIn('window.PortModuleLoading', loading_adapter.text)
        recovery_adapter = client.get("/ui/adapters/runtime_recovery.js")
        self.assertEqual(recovery_adapter.status_code, 200)
        self.assertIn("window.__portDtRuntimeRecoveryInstalled", recovery_adapter.text)
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
        console = client.get("/ui/adapters/rl_evidence_console.js")
        self.assertEqual(console.status_code, 200)
        self.assertIn("port_ops_v4", console.text)
        self.assertIn("/api/rl/regulatory-resilience/evidence", console.text)
        self.assertIn("data-reg-delay", console.text)
        self.assertIn("无执法结论或放行权", console.text)
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
        with patch.object(server_module, "evaluate_user_run", return_value=heldout):
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

    def test_asset_curve_endpoint_coalesces_dashboard_polling(self):
        from app import server as server_module

        server_module._ASSET_CURVE_CACHE.clear()
        client = TestClient(server_module.app)
        endpoint = "/api/curves/asset/qc-01?mode=now&horizon_min=360&step_min=1"
        first = client.get(endpoint)
        second = client.get(endpoint)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.headers.get("X-Port-DT-Cache"), "miss")
        self.assertEqual(second.headers.get("X-Port-DT-Cache"), "hit")

    def test_production_has_no_wildcard_or_implicit_cors(self):
        with patch.dict(os.environ, {"PORT_DT_ENV": "production", "PORT_DT_CORS_ORIGINS": ""}):
            self.assertEqual(cors_origins(), [])

    def test_development_cors_allows_flutter_web_acceptance_origin(self):
        with patch.dict(
            os.environ,
            {"PORT_DT_ENV": "development", "PORT_DT_CORS_ORIGINS": ""},
        ):
            self.assertIn("http://127.0.0.1:8765", cors_origins())

    def test_configured_cors_is_explicit(self):
        with patch.dict(os.environ, {"PORT_DT_ENV": "production", "PORT_DT_CORS_ORIGINS": "https://ops.example, https://audit.example"}):
            self.assertEqual(cors_origins(), ["https://ops.example", "https://audit.example"])

    def test_production_cors_rejects_non_tls_origin(self):
        with patch.dict(os.environ, {"PORT_DT_ENV": "production", "PORT_DT_CORS_ORIGINS": "http://ops.example"}):
            self.assertEqual(cors_origins(), [])

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

    def test_artifact_mutations_require_admin_and_zip_slip_is_blocked(self):
        operator = "operator-artifact-key-with-at-least-32-characters"
        admin = "admin-artifact-key-with-at-least-32-characters"

        guarded = FastAPI()
        configure_operations(guarded)

        @guarded.post("/api/rl/artifacts/upload")
        async def guarded_upload():
            return {"ok": True}

        with patch.dict(os.environ, {
            "PORT_DT_ENV": "production",
            "PORT_DT_API_KEYS": operator,
            "PORT_DT_ADMIN_API_KEYS": admin,
        }):
            client = TestClient(guarded)
            self.assertEqual(client.post("/api/rl/artifacts/upload", headers={"X-API-Key": operator}).status_code, 403)
            self.assertEqual(client.post("/api/rl/artifacts/upload", headers={"X-API-Key": admin}).status_code, 200)

        with tempfile.TemporaryDirectory() as tmp:
            model_root = Path(tmp) / "rl_model"
            artifact_dir = model_root / "registered_model" / "artifacts"
            artifact_dir.mkdir(parents=True)
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w") as archive:
                archive.writestr("../escape.json", "{}")
            application = FastAPI()
            application.include_router(rl_admin.router)
            with patch.object(rl_admin, "MODEL_ROOT", model_root):
                response = TestClient(application).post(
                    "/api/rl/artifacts/upload?model=registered_model",
                    files={"file": ("payload.zip", payload.getvalue(), "application/zip")},
                )
                self.assertEqual(response.status_code, 400)
                self.assertFalse((model_root / "registered_model" / "escape.json").exists())
                self.assertEqual(
                    TestClient(application).get("/api/rl/metrics/latest?model=../escape").status_code,
                    404,
                )

    def test_production_api_enforces_body_limit_rate_limit_and_headers(self):
        application = FastAPI()
        configure_operations(application)

        @application.api_route("/api/check", methods=["GET", "POST"])
        async def check():
            return {"ok": True}

        key = "rate-limit-test-key-with-at-least-32-characters"
        environment = {
            "PORT_DT_ENV": "production",
            "PORT_DT_API_KEYS": key,
            "PORT_DT_RATE_LIMIT_RPM": "2",
            "PORT_DT_MAX_REQUEST_BYTES": "1024",
        }
        with RATE_LIMITER.lock:
            RATE_LIMITER.events.clear()
        with patch.dict(os.environ, environment):
            client = TestClient(application)
            headers = {"X-API-Key": key}
            first = client.get("/api/check", headers=headers)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.headers["cache-control"], "no-store")
            self.assertIn("max-age=31536000", first.headers["strict-transport-security"])
            self.assertIn("frame-ancestors 'none'", first.headers["content-security-policy"])
            self.assertEqual(first.headers["x-ratelimit-limit"], "2")
            self.assertEqual(client.get("/api/check", headers=headers).status_code, 200)
            limited = client.get("/api/check", headers=headers)
            self.assertEqual(limited.status_code, 429)
            self.assertEqual(limited.headers["retry-after"], "60")

        body_key = "body-limit-test-key-with-at-least-32-characters"
        with patch.dict(os.environ, {**environment, "PORT_DT_API_KEYS": body_key}):
            oversized = TestClient(application).post(
                "/api/check",
                headers={"X-API-Key": body_key},
                content=b"x" * 1025,
            )
            self.assertEqual(oversized.status_code, 413)

    def test_readiness_rejects_unverified_site_file_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in ("graph", "calibration", "shadow", "actuator", "execution", "collaboration", "interoperability", "forecast", "benefit", "coordination", "continuity", "operating_model"):
                path = root / f"{name}.json"
                path.write_text("{}", encoding="utf-8")
                paths[name] = str(path)
            environment = {
                "PORT_DT_ENV": "production",
                "PORT_DT_CORS_ORIGINS": "https://ops.example",
                "PORT_DT_API_KEYS": "operator-readiness-key-with-32-characters",
                "PORT_DT_ADMIN_API_KEYS": "admin-readiness-key-with-at-least-32-characters",
                "PORT_DT_TLS_TERMINATION_ATTESTED": "true",
                "PORT_DT_SECRET_MANAGER_ATTESTED": "true",
                "PORT_DT_TWIN_GRAPH_PATH": paths["graph"],
                "PORT_DT_TWIN_CALIBRATION_PATH": paths["calibration"],
                "PORT_DT_SHADOW_ACCEPTANCE_PATH": paths["shadow"],
                "PORT_DT_ACTUATOR_CONFIG": paths["actuator"],
                "PORT_DT_EXECUTION_ACCEPTANCE_PATH": paths["execution"],
                "PORT_DT_PORT_CALL_COLLABORATION_PATH": paths["collaboration"],
                "PORT_DT_MARITIME_INTEROPERABILITY_PATH": paths["interoperability"],
                "PORT_DT_FORECAST_UNCERTAINTY_PATH": paths["forecast"],
                "PORT_DT_BUSINESS_BENEFIT_ATTRIBUTION_PATH": paths["benefit"],
                "PORT_DT_END_TO_END_COORDINATION_PATH": paths["coordination"],
                "PORT_DT_PRODUCTION_CONTINUITY_PATH": paths["continuity"],
                "PORT_DT_OPERATING_MODEL_GOVERNANCE_PATH": paths["operating_model"],
            }
            with patch.dict(os.environ, environment):
                report = readiness_report()
            self.assertFalse(report["production_site_ready"])
            for name in (
                "twin_graph", "site_calibration", "shadow_acceptance",
                "site_execution_acceptance", "port_call_collaboration", "maritime_interoperability", "forecast_uncertainty",
                "business_benefit_attribution",
                "end_to_end_coordination",
                "production_continuity", "operating_model_governance",
            ):
                self.assertFalse(report["checks"][name]["ok"])
                self.assertEqual(report["checks"][name]["status"], "evidence_incomplete")
            self.assertFalse(report["checks"]["site_evidence_consistency"]["ok"])

    def test_readiness_verifies_hashes_and_same_site_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_rows = list(canonical_rows(72))
            calibration_input = {
                "schema_version": "site_twin_calibration_dataset.v1",
                "site_id": "test-terminal-01",
                "dataset_id": "test-terminal-calibration-v2",
                "source": {
                    "source_system": "authorized-test-meter-export",
                    "owner": "test-terminal-operator",
                    "license": "authorized-test-calibration",
                    "timezone": "UTC",
                    "extracted_at": "2026-01-05T00:00:00Z",
                    "evidence_class": "authorized_site_export",
                },
                "split": {
                    "training_window": {
                        "start_at": source_rows[0]["timestamp"],
                        "end_at": source_rows[47]["timestamp"],
                    },
                    "validation_window": {
                        "start_at": source_rows[48]["timestamp"],
                        "end_at": source_rows[71]["timestamp"],
                    },
                },
                "model": {
                    "target": "observed_power_kw",
                    "features": ["throughput_teu", "vessel_arrivals", "ambient_c", "tide_m"],
                    "ridge_alpha": 0.1,
                },
                "rows": [
                    {
                        "timestamp": row["timestamp"],
                        "asset_group": "terminal.power.aggregate",
                        "observed_power_kw": (
                            400.0
                            + 2.0 * row["throughput_teu"]
                            + 30.0 * row["vessel_arrivals"]
                            + 15.0 * row["ambient_c"]
                            + 5.0 * row["tide_m"]
                        ),
                        "throughput_teu": row["throughput_teu"],
                        "vessel_arrivals": row["vessel_arrivals"],
                        "ambient_c": row["ambient_c"],
                        "tide_m": row["tide_m"],
                        "source_reference": f"TEST.METER.{index:04d}",
                    }
                    for index, row in enumerate(source_rows)
                ],
            }
            calibration_result = SiteTwinCalibrationService().run(
                calibration_input,
                source_verified=True,
                approved_by="site-model-risk",
                change_ticket="CHG-TEST-001",
            )
            self.assertTrue(calibration_result["valid"], calibration_result["errors"])
            shadow_result = SiteShadowAcceptanceService().run(
                canonical_shadow_bundle("test-terminal-01"),
                source_verified=True,
                operations_approved_by="site-operations-reviewer",
                safety_approved_by="site-safety-reviewer",
                change_ticket="CHG-TEST-SHADOW-001",
                rollback_drill_reference="ROLLBACK-TEST-001",
                rollback_drill_evidence=canonical_rollback_drill(),
            )
            self.assertTrue(shadow_result["valid"], shadow_result["errors"])
            execution_config = canonical_execution_config("test-terminal-01")
            execution_result = SiteExecutionAcceptanceService().run(
                canonical_execution_bundle("test-terminal-01"),
                source_verified=True,
                operations_approved_by="site-operations-execution-reviewer",
                maritime_safety_approved_by="site-maritime-safety-reviewer",
                controls_engineering_approved_by="site-controls-engineer",
                change_ticket="CHG-TEST-EXECUTION-001",
            )
            self.assertTrue(execution_result["valid"], execution_result["errors"])
            self.assertTrue(execution_result["evidence"]["approved"])
            collaboration_input = canonical_collaboration_bundle()
            collaboration_input["site_id"] = "test-terminal-01"
            collaboration_responses = collaboration_input["responses"]
            collaboration_input["responses"] = []
            collaboration_draft = PortCallCollaborationService().run(collaboration_input)
            self.assertTrue(collaboration_draft["valid"], collaboration_draft["errors"])
            for response in collaboration_responses:
                response["proposal_digest"] = collaboration_draft["evidence"]["proposal_digest"]
            collaboration_input["responses"] = collaboration_responses
            collaboration_result = PortCallCollaborationService().run(
                collaboration_input,
                source_verified=True,
                terminal_operations_approved_by="site-terminal-operations-reviewer",
                port_authority_approved_by="site-harbour-master-reviewer",
                change_ticket="CHG-TEST-PORT-CALL-001",
            )
            self.assertTrue(collaboration_result["valid"], collaboration_result["errors"])
            self.assertTrue(collaboration_result["evidence"]["approved"])
            interoperability_input = canonical_interoperability_bundle("test-terminal-01")
            interoperability_result = MaritimeInteroperabilityService().run(
                interoperability_input,
                source_verified=True,
                data_governance_approved_by="site-data-governance-reviewer",
                maritime_authority_approved_by="site-maritime-authority-reviewer",
                hydrographic_authority_approved_by="site-hydrographic-authority-reviewer",
                change_ticket="CHG-TEST-INTEROPERABILITY-001",
            )
            self.assertTrue(interoperability_result["valid"], interoperability_result["errors"])
            self.assertTrue(interoperability_result["evidence"]["approved"])
            forecast_result = ForecastUncertaintyService().run(
                canonical_forecast_bundle("test-terminal-01"),
                source_verified=True,
                operations_planning_approved_by="site-operations-planning-reviewer",
                model_risk_approved_by="site-model-risk-reviewer",
                maritime_safety_approved_by="site-maritime-safety-reviewer",
                change_ticket="CHG-TEST-FORECAST-001",
            )
            self.assertTrue(forecast_result["valid"], forecast_result["errors"])
            self.assertTrue(forecast_result["evidence"]["approved"])
            benefit_result = BusinessBenefitAttributionService().run(
                canonical_benefit_bundle("test-terminal-01"),
                source_verified=True,
                business_owner_approved_by="site-business-owner-reviewer",
                operations_assurance_approved_by="site-operations-assurance-reviewer",
                causal_methods_approved_by="independent-causal-methods-reviewer",
                change_ticket="CHG-TEST-BENEFIT-001",
            )
            self.assertTrue(benefit_result["valid"], benefit_result["errors"])
            self.assertTrue(benefit_result["evidence"]["approved"])
            coordination_result = EndToEndCoordinationService().run(
                canonical_coordination_bundle(),
                source_verified=True,
                integrated_planning_approved_by="site-integrated-planning-reviewer",
                marine_services_approved_by="site-marine-services-reviewer",
                terminal_operations_approved_by="site-terminal-operations-coordination-reviewer",
                equipment_energy_approved_by="site-equipment-energy-reviewer",
                change_ticket="CHG-TEST-COORDINATION-001",
            )
            self.assertTrue(coordination_result["valid"], coordination_result["errors"])
            self.assertTrue(coordination_result["evidence"]["approved"])
            continuity_result = ProductionContinuityService().run(
                canonical_continuity_bundle(),
                source_verified=True,
                service_owner_approved_by="site-service-owner-reviewer",
                site_reliability_approved_by="site-reliability-reviewer",
                continuity_cybersecurity_approved_by="site-continuity-cybersecurity-reviewer",
                change_ticket="CHG-TEST-CONTINUITY-001",
            )
            self.assertTrue(continuity_result["valid"], continuity_result["errors"])
            self.assertTrue(continuity_result["evidence"]["approved"])
            operating_model_result = OperatingModelGovernanceService().run(
                canonical_operating_model_bundle(),
                source_verified=True,
                executive_accountability_approved_by="site-executive-accountability-reviewer",
                governance_assurance_approved_by="site-governance-assurance-reviewer",
                maritime_safety_approved_by="site-maritime-safety-governance-reviewer",
                change_ticket="CHG-TEST-OPERATING-MODEL-001",
            )
            self.assertTrue(operating_model_result["valid"], operating_model_result["errors"])
            self.assertTrue(operating_model_result["evidence"]["approved"])
            payloads = {
                "graph": {
                    "site_id": "test-terminal-01", "approved": True,
                    "approved_by": "site-twin-owner", "source_mode": "authorized_site",
                    "entities": [{"id": "qc-01", "type": "quay_crane"}],
                },
                "calibration": calibration_result["evidence"],
                "shadow": shadow_result["evidence"],
                "actuator": execution_config,
                "execution": execution_result["evidence"],
                "collaboration": collaboration_result["evidence"],
                "interoperability": interoperability_result["evidence"],
                "forecast": forecast_result["evidence"],
                "benefit": benefit_result["evidence"],
                "coordination": coordination_result["evidence"],
                "continuity": continuity_result["evidence"],
                "operating_model": operating_model_result["evidence"],
            }
            paths = {}
            for name, payload in payloads.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = str(path)
            environment = {
                "PORT_DT_ENV": "production",
                "PORT_DT_CORS_ORIGINS": "https://ops.example",
                "PORT_DT_API_KEYS": "operator-readiness-key-with-32-characters",
                "PORT_DT_ADMIN_API_KEYS": "admin-readiness-key-with-at-least-32-characters",
                "PORT_DT_TLS_TERMINATION_ATTESTED": "true",
                "PORT_DT_SECRET_MANAGER_ATTESTED": "true",
                "PORT_DT_TWIN_GRAPH_PATH": paths["graph"],
                "PORT_DT_TWIN_CALIBRATION_PATH": paths["calibration"],
                "PORT_DT_SHADOW_ACCEPTANCE_PATH": paths["shadow"],
                "PORT_DT_ACTUATOR_CONFIG": paths["actuator"],
                "PORT_DT_EXECUTION_ACCEPTANCE_PATH": paths["execution"],
                "PORT_DT_PORT_CALL_COLLABORATION_PATH": paths["collaboration"],
                "PORT_DT_MARITIME_INTEROPERABILITY_PATH": paths["interoperability"],
                "PORT_DT_FORECAST_UNCERTAINTY_PATH": paths["forecast"],
                "PORT_DT_BUSINESS_BENEFIT_ATTRIBUTION_PATH": paths["benefit"],
                "PORT_DT_END_TO_END_COORDINATION_PATH": paths["coordination"],
                "PORT_DT_PRODUCTION_CONTINUITY_PATH": paths["continuity"],
                "PORT_DT_OPERATING_MODEL_GOVERNANCE_PATH": paths["operating_model"],
            }
            with patch.dict(os.environ, environment):
                report = readiness_report()
            for name in (
                "twin_graph", "site_calibration", "shadow_acceptance",
                "site_execution_acceptance", "port_call_collaboration", "maritime_interoperability", "forecast_uncertainty",
                "business_benefit_attribution",
                "end_to_end_coordination",
                "production_continuity", "operating_model_governance",
            ):
                self.assertTrue(report["checks"][name]["ok"])
                self.assertEqual(len(report["checks"][name]["sha256"]), 64)
            self.assertTrue(report["checks"]["site_evidence_consistency"]["ok"])
            self.assertTrue(report["production_site_ready"])

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
            config.write_text(json.dumps(canonical_execution_config(authorized=False)), encoding="utf-8")
            token = "a-separate-test-confirmation-token-123456"
            with patch.object(actuator_module, "AUDIT_DIR", str(audit)), patch.dict(os.environ, {"PORT_DT_ENV": "development", "PORT_DT_ENABLE_ACTUATOR_DRY_RUN": "1", "TEST_SECOND_CHANNEL": token}):
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

    def test_expired_or_tampered_pending_command_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit"
            audit.mkdir()
            config = root / "actuators.json"
            config.write_text(json.dumps(canonical_execution_config(authorized=False)), encoding="utf-8")
            token = "a-separate-test-confirmation-token-123456"
            environment = {
                "PORT_DT_ENV": "development",
                "PORT_DT_ENABLE_ACTUATOR_DRY_RUN": "1",
                "TEST_SECOND_CHANNEL": token,
            }
            with patch.object(actuator_module, "AUDIT_DIR", str(audit)), patch.dict(os.environ, environment):
                gateway = PortSouthboundGateway(str(config))
                gateway.idem = IdempotencyStore(str(audit))
                first = gateway.dispatch(Command(
                    "BESS-1", "bess", "set", {"power_kw": 20}, requested_by="alice",
                    idempotency_key="expires", two_channel_required=True,
                ))
                first_path = Path(first.evidence_path or "")
                expired = json.loads(first_path.read_text(encoding="utf-8"))
                with patch("app.adapters.actuators.time.time", return_value=expired["timestamps"]["expires_at"] + 1.0):
                    blocked = gateway.confirm(first.command_id, "bob", token)
                self.assertEqual(blocked.message, "command_confirmation_expired")
                persisted = json.loads(first_path.read_text(encoding="utf-8"))
                self.assertIn("expired_at", persisted["timestamps"])
                self.assertTrue(persisted["results"][-1]["blocked"])

                second = gateway.dispatch(Command(
                    "BESS-1", "bess", "set", {"power_kw": 30}, requested_by="alice",
                    idempotency_key="tampered", two_channel_required=True,
                ))
                second_path = Path(second.evidence_path or "")
                tampered = json.loads(second_path.read_text(encoding="utf-8"))
                tampered["command"]["parameters"]["power_kw"] = 99
                second_path.write_text(json.dumps(tampered), encoding="utf-8")
                blocked = gateway.confirm(second.command_id, "bob", token)
                self.assertEqual(blocked.message, "pending_evidence_or_config_integrity_failed")

                third = gateway.dispatch(Command(
                    "BESS-1", "bess", "set", {"power_kw": 40}, requested_by="alice",
                    idempotency_key="expiry-tampered", two_channel_required=True,
                ))
                third_path = Path(third.evidence_path or "")
                tampered = json.loads(third_path.read_text(encoding="utf-8"))
                tampered["timestamps"]["expires_at"] += 600
                third_path.write_text(json.dumps(tampered), encoding="utf-8")
                blocked = gateway.confirm(third.command_id, "bob", token)
                self.assertEqual(blocked.message, "pending_evidence_or_config_integrity_failed")

    def test_production_dispatch_requires_bound_execution_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit"
            audit.mkdir()
            config = root / "actuators.json"
            config.write_text(json.dumps(canonical_execution_config()), encoding="utf-8")
            environment = {
                "PORT_DT_ENV": "production",
                "PORT_DT_ACTUATOR_CONFIG": str(config),
                "PORT_DT_EXECUTION_ACCEPTANCE_PATH": "",
                "TEST_SECOND_CHANNEL": "a-separate-test-confirmation-token-123456",
            }
            with patch.object(actuator_module, "AUDIT_DIR", str(audit)), patch.dict(os.environ, environment):
                gateway = PortSouthboundGateway(str(config))
                gateway.idem = IdempotencyStore(str(audit))
                result = gateway.dispatch(Command(
                    "BESS-1", "bess", "set", {"power_kw": 20}, requested_by="alice",
                    idempotency_key="production-block", two_channel_required=True,
                ))
                self.assertEqual(result.status, "FAILED")
                self.assertEqual(result.message, "site_execution_acceptance_required")
                self.assertEqual(list(audit.glob("guard-*.json")), [])


if __name__ == "__main__":
    unittest.main()
