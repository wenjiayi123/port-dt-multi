from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.services.site_shadow_acceptance import SiteShadowAcceptanceService
from scripts.verify_site_shadow import main as shadow_cli_main


def shadow_bundle(*, evidence_class: str = "contract_test_only") -> dict:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    cycles = []
    for day in range(7):
        for slot in range(5):
            index = day * 5 + slot
            started = start + timedelta(days=day, hours=slot * 3)
            energy = 1800.0 + slot * 45.0 + day * 8.0
            throughput = 120.0 + slot * 6.0
            delay = 34.0 + slot
            cycles.append({
                "cycle_id": f"SHADOW.CYCLE.{index:04d}",
                "started_at": started.isoformat(),
                "ended_at": (started + timedelta(minutes=45)).isoformat(),
                "asset_group": "TERMINAL.POWER.A" if slot % 2 == 0 else "TERMINAL.POWER.B",
                "scenario": ("normal_operations", "peak_berthing", "high_temperature")[index % 3],
                "data_quality_passed": True,
                "incumbent": {
                    "actual_energy_kwh": energy,
                    "actual_throughput_teu": throughput,
                    "actual_delay_minutes": delay,
                    "source_reference": f"TOS.METER.CYCLE.{index:04d}",
                },
                "candidate": {
                    "projected_energy_kwh": energy * 0.98,
                    "projected_throughput_teu": throughput,
                    "projected_delay_minutes": delay * 0.97,
                    "recommendation_receipt_id": f"DECISION.RECEIPT.{index:04d}",
                    "recommendation_available": True,
                    "action_feasible": True,
                    "recommendation_latency_ms": 420.0 + index,
                },
                "guardrail": {"violation_count": 0, "codes": []},
                "review": {"disposition": "not_reviewed" if index % 6 == 0 else "accepted"},
                "side_effect": False,
            })
    return {
        "schema_version": "site_shadow_observation.v1",
        "site_id": "SITE.CNSHA.CONTRACT",
        "run_id": "SHADOW.CONTRACT.202602",
        "source": {
            "source_system": "PORT_DT_CONTRACT_TEST",
            "owner": "PORT_DT_CONTRACT_TEST",
            "license": "contract_test_only_not_authorized",
            "timezone": "Asia/Shanghai",
            "extracted_at": (start + timedelta(days=8)).isoformat(),
            "evidence_class": evidence_class,
        },
        "policy": {
            "candidate_policy_version": "candidate.policy.contract.v1",
            "incumbent_policy_version": "incumbent.sop.contract.v1",
            "calibration_evidence_digest": "a" * 64,
            "twin_graph_digest": "b" * 64,
            "recommendation_only": True,
        },
        "cycles": cycles,
    }


def authorized_bundle() -> dict:
    payload = shadow_bundle(evidence_class="authorized_site_shadow_export")
    payload["source"].update(
        source_system="SITE_TOS_METER_DECISION_EXPORT",
        owner="AUTHORIZED_PORT_OPERATOR",
        license="AUTHORIZED_INTERNAL_SHADOW_REVIEW",
    )
    return payload


def rollback_drill_evidence() -> dict:
    return {
        "command_id": "rollback-drill-2026-02",
        "command": {"asset_id": "site-test-asset", "action": "setpoint"},
        "results": [
            {"at": 1000.0, "ok": True, "detail": {"acknowledged": True}},
            {"at": 1012.0, "ok": True, "detail": {"readback_restored": True}, "rollback": True},
        ],
        "timestamps": {"executed_at": 1000.0, "rolledback_at": 1012.0},
        "approvals": [
            {"type": "rollback", "by": "rollback-duty-manager", "reason": "site shadow acceptance drill", "at": 1008.0},
        ],
    }


def approved_evidence() -> dict:
    result = SiteShadowAcceptanceService().run(
        authorized_bundle(),
        source_verified=True,
        operations_approved_by="terminal-operations-reviewer",
        safety_approved_by="maritime-safety-reviewer",
        change_ticket="deployment-change-2026-02",
        rollback_drill_reference="rollback-drill-2026-02",
        rollback_drill_evidence=rollback_drill_evidence(),
    )
    if not result["valid"]:
        raise AssertionError(result["errors"])
    return result["evidence"]


class SiteShadowAcceptanceTests(unittest.TestCase):
    def test_contract_sample_passes_calculation_without_site_claim(self):
        first = SiteShadowAcceptanceService().run(shadow_bundle())
        second = SiteShadowAcceptanceService().run(shadow_bundle())
        self.assertTrue(first["valid"], first["errors"])
        self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
        evidence = first["evidence"]
        self.assertEqual(evidence["shadow_cycles"], 35)
        self.assertEqual(evidence["operational_days"], 7)
        self.assertEqual(evidence["acceptance_status"], "pass")
        self.assertTrue(all(evidence["threshold_checks"].values()))
        self.assertEqual(evidence["guardrail_violation_rate"], 0.0)
        self.assertFalse(evidence["measured_incumbent_baseline"])
        self.assertFalse(evidence["candidate_business_impact_measured"])
        self.assertFalse(evidence["approved"])
        self.assertFalse(first["boundary"]["site_shadow_accepted"])
        self.assertFalse(first["boundary"]["dispatch_allowed"])
        self.assertEqual(first["warnings"][0]["code"], "contract_only")

    def test_short_or_nonconsecutive_shadow_run_is_rejected(self):
        short = shadow_bundle()
        short["cycles"] = short["cycles"][:30]
        result = SiteShadowAcceptanceService().run(short)
        self.assertFalse(result["valid"])
        codes = {item["code"] for item in result["errors"]}
        self.assertIn("minimum_cycles", codes)

        nonconsecutive = shadow_bundle()
        for row in nonconsecutive["cycles"][-5:]:
            for field in ("started_at", "ended_at"):
                row[field] = (datetime.fromisoformat(row[field]) + timedelta(days=1)).isoformat()
        result = SiteShadowAcceptanceService().run(nonconsecutive)
        self.assertFalse(result["valid"])
        self.assertIn("operational_continuity", {item["code"] for item in result["errors"]})

    def test_guardrail_violation_fails_fixed_gate(self):
        payload = shadow_bundle()
        payload["cycles"][3]["guardrail"]["violation_count"] = 1
        result = SiteShadowAcceptanceService().run(payload)
        self.assertTrue(result["valid"], result["errors"])
        evidence = result["evidence"]
        self.assertEqual(evidence["acceptance_status"], "fail")
        self.assertFalse(evidence["threshold_checks"]["guardrail_violation_rate"])
        self.assertFalse(evidence["approved"])

    def test_authorized_run_needs_two_independent_reviewers_and_rollback(self):
        service = SiteShadowAcceptanceService()
        unattested = service.run(authorized_bundle())
        self.assertTrue(unattested["valid"], unattested["errors"])
        self.assertFalse(unattested["evidence"]["measured_incumbent_baseline"])
        self.assertEqual(unattested["warnings"][0]["code"], "source_attestation_missing")

        evidence = approved_evidence()
        self.assertTrue(evidence["measured_incumbent_baseline"])
        self.assertTrue(evidence["approved"])
        self.assertTrue(evidence["boundary"]["site_shadow_accepted"])
        self.assertFalse(evidence["boundary"]["production_authority"])

        self_approved = service.run(
            authorized_bundle(),
            source_verified=True,
            operations_approved_by="AUTHORIZED_PORT_OPERATOR",
            safety_approved_by="maritime-safety-reviewer",
            change_ticket="deployment-change-2026-02",
            rollback_drill_reference="rollback-drill-2026-02",
            rollback_drill_evidence=rollback_drill_evidence(),
        )
        self.assertFalse(self_approved["evidence"]["approved"])
        self.assertIn("dual_approval_invalid", {item["code"] for item in self_approved["warnings"]})

    def test_tampering_cycle_or_threshold_breaks_digest_and_gate(self):
        evidence = approved_evidence()
        evidence["cycles"][0]["guardrail_violation_count"] = 1
        validation = SiteShadowAcceptanceService.validate_evidence(evidence)
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["production_gate_eligible"])
        self.assertIn("evidence_digest does not match shadow evidence content", validation["errors"])
        self.assertIn("metric guardrail_violation_rate does not match cycle records", validation["errors"])

        evidence = approved_evidence()
        evidence["thresholds"]["guardrail_violation_rate"] = 0.2
        validation = SiteShadowAcceptanceService.validate_evidence(evidence)
        self.assertFalse(validation["valid"])
        self.assertIn("shadow thresholds do not match the fixed acceptance contract", validation["errors"])

    def test_readiness_only_accepts_approved_versioned_artifact(self):
        evidence = approved_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site_shadow_acceptance_v2.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_SHADOW_ACCEPTANCE_PATH": str(path)}, clear=True):
                readiness = SiteShadowAcceptanceService().readiness()
        self.assertTrue(readiness["configured_artifact"]["verified"])
        self.assertTrue(readiness["configured_artifact"]["production_gate_eligible"])
        self.assertTrue(readiness["boundary"]["site_shadow_accepted"])
        self.assertNotIn(str(path), json.dumps(readiness))

    def test_api_runs_contract_but_never_approves_browser_input(self):
        client = TestClient(server.app)
        readiness = client.get("/api/v3/shadow-acceptance/readiness")
        self.assertEqual(readiness.status_code, 200)
        self.assertFalse(readiness.json()["boundary"]["site_shadow_accepted"])
        result = client.post("/api/v3/shadow-acceptance/run", json=shadow_bundle())
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertTrue(body["valid"])
        self.assertEqual(body["accepted_cycles"], 35)
        self.assertFalse(body["evidence"]["approved"])
        self.assertFalse(body["boundary"]["production_authority"])

    def test_cli_writes_new_version_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "shadow_input.json"
            rollback_path = Path(tmp) / "rollback_drill.json"
            output_path = Path(tmp) / "shadow_evidence_v1.json"
            input_path.write_text(json.dumps(authorized_bundle()), encoding="utf-8")
            rollback_path.write_text(json.dumps(rollback_drill_evidence()), encoding="utf-8")
            arguments = [
                "--input", str(input_path), "--output", str(output_path),
                "--authorized-source-attested",
                "--operations-approved-by", "terminal-operations-reviewer",
                "--safety-approved-by", "maritime-safety-reviewer",
                "--change-ticket", "deployment-change-2026-02",
                "--rollback-drill-reference", "rollback-drill-2026-02",
                "--rollback-drill-evidence", str(rollback_path),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(shadow_cli_main(arguments), 0)
            self.assertTrue(json.loads(output_path.read_text(encoding="utf-8"))["approved"])
            with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                shadow_cli_main(arguments)


if __name__ == "__main__":
    unittest.main()
