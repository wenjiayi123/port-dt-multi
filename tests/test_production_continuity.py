from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.server import app

from app.services.production_continuity import (
    BINDING_FIELDS,
    COMPONENTS,
    DATASET_SCHEMA,
    DRILL_LIMITS,
    ProductionContinuityService,
)
from scripts.verify_production_continuity import main as continuity_cli_main


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def continuity_bundle(*, authorized: bool = False, hours: int | None = None) -> dict:
    hours = hours or (720 if authorized else 168)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=hours)
    extracted = end + timedelta(hours=2)
    records = []
    for index in range(hours):
        records.append({
            "hour_start": _iso(start + timedelta(hours=index)),
            "collection_receipt_id": f"CONTINUITY.COLLECTION.{index:04d}",
            "components": [{
                "component_id": component,
                "uptime_minutes": 60.0,
                "request_count": 1000 + index,
                "error_count": 0,
                "latency_p95_ms": 90.0 + (index % 5),
                "data_freshness_p95_seconds": 12.0 + (index % 3),
                "audit_events_expected": 5,
                "audit_events_delivered": 5,
                "source_reference": f"OBSERVABILITY.{component}.{index:04d}",
            } for component in COMPONENTS],
        })
    drills = []
    for index, (drill_type, limits) in enumerate(DRILL_LIMITS.items()):
        started = start + timedelta(hours=24 + index)
        drills.append({
            "drill_id": f"DRILL.{drill_type}.001",
            "drill_type": drill_type,
            "started_at": _iso(started),
            "completed_at": _iso(started + timedelta(minutes=limits["rto_minutes_max"])),
            "rto_minutes": float(limits["rto_minutes_max"]),
            "rpo_minutes": float(limits["rpo_minutes_max"]),
            "result": "pass",
            "execution_receipt_id": f"DRILL.EXECUTION.{drill_type}.001",
            "restore_receipt_id": f"DRILL.RESTORE.{drill_type}.001",
            "approved_by": f"DRILL.REVIEWER.{drill_type}",
        })
    backups = []
    for index in range((hours + 23) // 24):
        created = start + timedelta(days=index, hours=1)
        backups.append({
            "backup_id": f"BACKUP.CONTINUITY.{index:03d}",
            "created_at": _iso(created),
            "restore_tested_at": _iso(end + timedelta(hours=1)),
            "sha256": hashlib.sha256(f"backup-{index}".encode()).hexdigest(),
            "immutable": True,
            "encrypted": True,
            "restore_receipt_id": f"BACKUP.RESTORE.{index:03d}",
        })
    incident_start = start + timedelta(hours=10)
    return {
        "schema_version": DATASET_SCHEMA,
        "site_id": "test-terminal-01",
        "run_id": f"CONTINUITY.RUN.{hours}",
        "source": {
            "source_system": "site-observability-and-service-management-export",
            "owner": "site-continuity-data-owner",
            "license": "authorized-operational-export" if authorized else "contract-test-fixture",
            "timezone": "Asia/Kuala_Lumpur",
            "extracted_at": _iso(extracted),
            "evidence_class": "authorized_site_continuity_export" if authorized else "contract_test_only",
        },
        "bindings": {field: hashlib.sha256(field.encode()).hexdigest() for field in BINDING_FIELDS},
        "window": {"start_at": _iso(start), "end_at": _iso(end), "step_minutes": 60},
        "hourly_records": records,
        "drills": drills,
        "backups": backups,
        "incidents": [{
            "incident_id": "INCIDENT.P2.20260701.001",
            "severity": "P2",
            "detected_at": _iso(incident_start),
            "acknowledged_at": _iso(incident_start + timedelta(minutes=5)),
            "mitigated_at": _iso(incident_start + timedelta(minutes=10)),
            "recovered_at": _iso(incident_start + timedelta(minutes=15)),
            "closed_at": _iso(incident_start + timedelta(minutes=30)),
            "commander_id": "incident-commander.01",
            "work_order_id": "work-order.20260701.001",
            "root_cause_reference": "root-cause.20260701.001",
            "postmortem_reference": "postmortem.20260701.001",
        }],
        "changes": [{
            "change_id": "CHANGE.CONTINUITY.202607.001",
            "deployed_at": _iso(start + timedelta(hours=48)),
            "approval_receipt_id": "CHANGE.APPROVAL.202607.001",
            "canary_receipt_id": "CHANGE.CANARY.202607.001",
            "health_receipt_id": "CHANGE.HEALTH.202607.001",
            "rollback_receipt_id": "CHANGE.ROLLBACK.202607.001",
            "rollback_ready": True,
        }],
    }


def authorized_bundle() -> dict:
    return continuity_bundle(authorized=True)


def run_authorized(service: ProductionContinuityService | None = None) -> dict:
    service = service or ProductionContinuityService()
    return service.run(
        authorized_bundle(),
        source_verified=True,
        service_owner_approved_by="reviewer.service-owner",
        site_reliability_approved_by="reviewer.site-reliability",
        continuity_cybersecurity_approved_by="reviewer.continuity-cybersecurity",
        change_ticket="change.continuity.202607.001",
    )


class ProductionContinuityTests(unittest.TestCase):
    def test_seven_day_contract_proves_continuity_without_field_slo_claim(self):
        result = ProductionContinuityService().run(continuity_bundle())
        self.assertTrue(result["valid"], result["errors"])
        evidence = result["evidence"]
        self.assertEqual(evidence["continuity_status"], "pass")
        self.assertEqual(evidence["metrics"]["continuous_hours"], 168)
        self.assertEqual(evidence["metrics"]["component_slo_pass_count"], 8)
        self.assertEqual(evidence["metrics"]["drill_pass_count"], 6)
        self.assertEqual(evidence["metrics"]["restore_tested_backup_count"], 7)
        self.assertFalse(evidence["approved"])
        self.assertFalse(evidence["boundary"]["field_slo_claim_eligible"])
        self.assertFalse(evidence["boundary"]["automatic_failover_authority"])
        self.assertFalse(evidence["boundary"]["production_authority"])

    def test_gap_or_missing_component_is_rejected(self):
        payload = continuity_bundle()
        payload["hourly_records"].pop(4)
        payload["hourly_records"][0]["components"].pop()
        result = ProductionContinuityService().run(payload)
        self.assertFalse(result["valid"])
        codes = {row["code"] for row in result["errors"]}
        self.assertIn("continuous_coverage", codes)
        self.assertIn("component_coverage", codes)

    def test_failed_slo_and_drill_remain_visible_and_block_acceptance(self):
        payload = continuity_bundle()
        payload["hourly_records"][0]["components"][0]["latency_p95_ms"] = 2000.0
        payload["drills"][0]["result"] = "fail"
        result = ProductionContinuityService().run(payload)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["evidence"]["continuity_status"], "blocked")
        self.assertFalse(result["evidence"]["threshold_checks"]["all_component_slos"])
        self.assertFalse(result["evidence"]["threshold_checks"]["all_resilience_drills"])

    def test_authorized_thirty_day_export_requires_three_independent_reviewers(self):
        service = ProductionContinuityService()
        result = run_authorized(service)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["evidence"]["approved"])
        self.assertEqual(result["evidence"]["metrics"]["continuous_hours"], 720)
        self.assertEqual(service.validate_evidence(result["evidence"]), {"valid": True, "errors": [], "production_gate_eligible": True})
        unapproved = service.run(authorized_bundle())
        self.assertTrue(unapproved["valid"])
        self.assertFalse(unapproved["evidence"]["approved"])

    def test_semantic_tampering_is_rejected_after_digest_recalculation(self):
        service = ProductionContinuityService()
        evidence = copy.deepcopy(run_authorized(service)["evidence"])
        evidence["metrics"]["continuous_hours"] = 999
        body = copy.deepcopy(evidence)
        body.pop("evidence_digest")
        evidence["evidence_digest"] = _digest(body)
        validation = service.validate_evidence(evidence)
        self.assertFalse(validation["valid"])
        self.assertIn("metrics does not reproduce from source_input", validation["errors"])

    def test_readiness_accepts_only_approved_artifact(self):
        service = ProductionContinuityService()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continuity.json"
            path.write_text(json.dumps(run_authorized(service)["evidence"]), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_PRODUCTION_CONTINUITY_PATH": str(path)}, clear=False):
                readiness = service.readiness()
        self.assertTrue(readiness["configured_artifact"]["verified"])
        self.assertTrue(readiness["boundary"]["site_continuity_accepted"])
        self.assertFalse(readiness["boundary"]["automatic_failover_authority"])

    def test_http_routes_remain_contract_only(self):
        client = TestClient(app)
        readiness = client.get("/api/v3/production-continuity/readiness")
        result = client.post("/api/v3/production-continuity/run", json=continuity_bundle())
        self.assertEqual(readiness.status_code, 200)
        self.assertFalse(readiness.json()["boundary"]["site_continuity_accepted"])
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["valid"])
        self.assertFalse(result.json()["evidence"]["source"]["source_verified"])
        self.assertFalse(result.json()["evidence"]["approved"])
        self.assertFalse(result.json()["boundary"]["automatic_failover_authority"])

    def test_cli_writes_new_evidence_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path, output_path = Path(directory) / "input.json", Path(directory) / "continuity.v1.json"
            input_path.write_text(json.dumps(continuity_bundle()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = continuity_cli_main(["--input", str(input_path), "--output", str(output_path)])
            summary = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(summary["continuous_hours"], 168)
            self.assertFalse(summary["approved"])
            with self.assertRaises(FileExistsError):
                continuity_cli_main(["--input", str(input_path), "--output", str(output_path)])


if __name__ == "__main__":
    unittest.main()
