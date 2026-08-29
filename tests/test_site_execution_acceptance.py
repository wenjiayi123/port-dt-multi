from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.adapters.actuators import Command, PortSouthboundGateway
from app.services.site_execution_acceptance import SiteExecutionAcceptanceService
from scripts.verify_site_execution import main as verify_site_execution


SCENARIOS = (
    "safe_command_readback",
    "out_of_bounds_block",
    "expired_command_block",
    "duplicate_command_block",
    "lost_acknowledgement_block",
    "independent_interlock_trip",
    "emergency_stop",
    "rollback_restore",
)


def actuator_config(site_id: str = "TEST.TERMINAL.01", *, authorized: bool = False) -> dict:
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
                    "set": {"power_kw": {"required": True, "min": -100.0, "max": 100.0}}
                }
            },
            "type": {},
        },
    }


def commissioning_bundle(site_id: str = "TEST.TERMINAL.01", *, authorized: bool = False) -> dict:
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    records = []
    for index, scenario in enumerate(SCENARIOS):
        began = start + timedelta(minutes=index * 10)
        records.append({
            "test_id": f"EXECUTION.TEST.{index:02d}",
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
            "requester": "fixture-requester",
            "confirmer": "fixture-confirmer",
            "source_reference": f"FIXTURE.EXECUTION.{index:02d}",
        })
    return {
        "schema_version": "site_execution_commissioning_dataset.v1",
        "site_id": site_id,
        "run_id": "EXECUTION.COMMISSIONING.202603",
        "source": {
            "source_system": "authorized-control-gateway" if authorized else "browser-contract-fixture",
            "owner": "terminal-control-owner",
            "license": "authorized-site-evidence" if authorized else "contract-test-only",
            "timezone": "UTC",
            "extracted_at": "2026-03-02T00:00:00Z",
            "evidence_class": "authorized_site_commissioning_export" if authorized else "contract_test_only",
        },
        "shadow_acceptance_digest": "a" * 64,
        "actuator_config": actuator_config(site_id, authorized=authorized),
        "tests": records,
    }


def run_authorized(bundle: dict) -> dict:
    return SiteExecutionAcceptanceService().run(
        bundle,
        source_verified=True,
        operations_approved_by="operations-reviewer",
        maritime_safety_approved_by="maritime-safety-reviewer",
        controls_engineering_approved_by="controls-engineering-reviewer",
        change_ticket="CHANGE.EXECUTION.001",
    )


class SiteExecutionAcceptanceTests(unittest.TestCase):
    def test_contract_passes_but_cannot_claim_site_commissioning_or_authority(self):
        result = SiteExecutionAcceptanceService().run(commissioning_bundle())
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["accepted_tests"], 8)
        self.assertEqual(result["evidence"]["acceptance_status"], "pass")
        self.assertEqual(result["evidence"]["metrics"]["unsafe_command_execution_count"], 0)
        self.assertFalse(result["evidence"]["site_commissioning_verified"])
        self.assertFalse(result["evidence"]["approved"])
        self.assertFalse(result["boundary"]["site_execution_accepted"])
        self.assertFalse(result["boundary"]["dispatch_allowed"])
        self.assertFalse(result["boundary"]["production_authority"])

    def test_fixed_matrix_rejects_missing_duplicate_and_unsafe_block_execution(self):
        service = SiteExecutionAcceptanceService()
        missing = commissioning_bundle()
        missing["tests"].pop()
        self.assertFalse(service.run(missing)["valid"])

        duplicate = commissioning_bundle()
        duplicate["tests"].append(copy.deepcopy(duplicate["tests"][0]))
        duplicate["tests"][-1]["test_id"] = "EXECUTION.TEST.DUPLICATE"
        self.assertFalse(service.run(duplicate)["valid"])

        unsafe = commissioning_bundle()
        unsafe["tests"][1]["equipment_command_sent"] = True
        result = service.run(unsafe)
        self.assertFalse(result["valid"])
        self.assertTrue(any(error["code"] == "unsafe_execution" for error in result["errors"]))

    def test_config_requires_ttl_readback_interlock_rollback_and_constraints(self):
        config = actuator_config()
        del config["security"]["command_ttl_seconds"]
        config["security"]["require_verified_readback"] = False
        del config["routing"]["asset"]["BESS-1"]["interlock_id"]
        config["routing"]["asset"]["BESS-1"]["rollback_supported"] = False
        config["constraints"]["asset"]["BESS-1"]["set"] = {}
        validation = SiteExecutionAcceptanceService.validate_actuator_config(config)
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["production_gate_eligible"])
        joined = " ".join(validation["errors"])
        for marker in ("command_ttl_seconds", "require_verified_readback", "interlock_id", "rollback_supported", "parameter constraints"):
            self.assertIn(marker, joined)

    def test_example_placeholders_cannot_become_an_eligible_site_config(self):
        example_path = Path(__file__).resolve().parents[1] / "config" / "actuators.example.json"
        example = json.loads(example_path.read_text(encoding="utf-8"))
        example["enabled"] = True
        validation = SiteExecutionAcceptanceService.validate_actuator_config(example)
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["production_gate_eligible"])

    def test_runtime_rejects_parameters_not_declared_by_the_action_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "actuator.json"
            config_path.write_text(json.dumps(actuator_config()), encoding="utf-8")
            environment = {
                "PORT_DT_ENV": "development",
                "TEST_SECOND_CHANNEL": "contract-second-channel-token-at-least-32-characters",
            }
            with patch.dict(os.environ, environment):
                gateway = PortSouthboundGateway(str(config_path))
                result = gateway.dispatch(Command(
                    "BESS-1",
                    "bess",
                    "set",
                    {"power_kw": 20, "force_override": True},
                    requested_by="operator-a",
                    two_channel_required=True,
                ))
            self.assertEqual(result.status, "FAILED")
            self.assertEqual(result.message, "site_constraints_failed")
            self.assertTrue(any(
                row["parameter"] == "force_override" and row["reason"] == "undeclared_parameter"
                for row in result.details["violations"]
            ))

    def test_authorized_evidence_needs_three_independent_reviewers_and_attestation(self):
        bundle = commissioning_bundle(authorized=True)
        unattested = SiteExecutionAcceptanceService().run(bundle)
        self.assertTrue(unattested["valid"])
        self.assertFalse(unattested["evidence"]["approved"])

        owner_review = SiteExecutionAcceptanceService().run(
            bundle,
            source_verified=True,
            operations_approved_by="terminal-control-owner",
            maritime_safety_approved_by="maritime-safety-reviewer",
            controls_engineering_approved_by="controls-engineering-reviewer",
            change_ticket="CHANGE.EXECUTION.001",
        )
        self.assertTrue(owner_review["valid"])
        self.assertFalse(owner_review["evidence"]["approved"])

        approved = run_authorized(bundle)
        self.assertTrue(approved["valid"], approved["errors"])
        self.assertTrue(approved["evidence"]["site_commissioning_verified"])
        self.assertTrue(approved["evidence"]["approved"])
        self.assertTrue(approved["evidence"]["boundary"]["site_execution_accepted"])
        self.assertFalse(approved["evidence"]["boundary"]["direct_model_dispatch"])
        self.assertTrue(approved["evidence"]["boundary"]["per_command_human_confirmation_required"])
        self.assertFalse(approved["evidence"]["boundary"]["production_authority"])

    def test_evidence_tampering_and_config_digest_mismatch_fail_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = commissioning_bundle(authorized=True)
            result = run_authorized(bundle)
            self.assertTrue(result["valid"], result["errors"])
            config_path = root / "actuator.json"
            evidence_path = root / "execution.json"
            config_path.write_text(json.dumps(bundle["actuator_config"]), encoding="utf-8")
            evidence_path.write_text(json.dumps(result["evidence"]), encoding="utf-8")
            environment = {
                "PORT_DT_ACTUATOR_CONFIG": str(config_path),
                "PORT_DT_EXECUTION_ACCEPTANCE_PATH": str(evidence_path),
            }
            with patch.dict(os.environ, environment):
                ready = SiteExecutionAcceptanceService().readiness()
                self.assertTrue(ready["boundary"]["site_execution_accepted"])

                changed_config = json.loads(config_path.read_text(encoding="utf-8"))
                changed_config["constraints"]["asset"]["BESS-1"]["set"]["power_kw"]["max"] = 90
                config_path.write_text(json.dumps(changed_config), encoding="utf-8")
                mismatch = SiteExecutionAcceptanceService().readiness()
                self.assertFalse(mismatch["boundary"]["site_execution_accepted"])
                self.assertIn("actuator_config_digest_mismatch", mismatch["configured_artifacts"]["blockers"])

                tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
                tampered["metrics"]["unsafe_command_execution_count"] = 1
                evidence_path.write_text(json.dumps(tampered), encoding="utf-8")
                invalid = SiteExecutionAcceptanceService().readiness()
                self.assertFalse(invalid["configured_artifacts"]["verified"])
                self.assertTrue(any("metric unsafe_command_execution_count" in item for item in invalid["configured_artifacts"]["blockers"]))

    def test_browser_api_is_contract_only_and_creates_no_control_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp)
            with patch("app.adapters.actuators.AUDIT_DIR", str(audit)):
                response = TestClient(server.app).post(
                    "/api/v3/execution-acceptance/run",
                    json=commissioning_bundle(),
                )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["valid"])
            self.assertFalse(payload["evidence"]["approved"])
            self.assertFalse(payload["boundary"]["dispatch_allowed"])
            self.assertFalse(payload["boundary"]["production_authority"])
            self.assertEqual(list(audit.iterdir()), [])

    def test_cli_writes_new_version_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "commissioning.json"
            output = root / "execution-evidence-v1.json"
            source.write_text(json.dumps(commissioning_bundle()), encoding="utf-8")
            argv = ["--input", str(source), "--output", str(output)]
            self.assertEqual(verify_site_execution(argv), 0)
            self.assertTrue(output.exists())
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["approved"])
            with self.assertRaises(FileExistsError):
                verify_site_execution(argv)


if __name__ == "__main__":
    unittest.main()
