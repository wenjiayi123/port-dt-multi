from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.server import app

from app.services.operating_model_governance import (
    BINDING_FIELDS,
    DATASET_SCHEMA,
    DOMAINS,
    ESCALATION_TARGETS,
    SHIFTS,
    SHIFT_ROLES,
    SHIFT_ROLE_COMPETENCY,
    WORKFLOWS,
    OperatingModelGovernanceService,
)
from scripts.verify_operating_model_governance import main as governance_cli_main


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def governance_bundle(*, authorized: bool = False) -> dict:
    roster_days = 28 if authorized else 7
    start = date(2026, 7, 1)
    end = start + timedelta(days=roster_days)
    person_ids = [f"person.role-holder.{index:02d}" for index in range(15)]
    people = [{
        "person_id": person_id,
        "department": f"department-{index % 8}",
        "directory_reference": f"directory.person.{index:02d}",
        "active": True,
        "role_codes": [*SHIFT_ROLES, "requester", "approver", "executor", "verifier", "reviewer"],
    } for index, person_id in enumerate(person_ids)]
    assignments = []
    for index, domain in enumerate(DOMAINS):
        assignments.append({
            "domain": domain,
            "accountable_id": person_ids[index % len(person_ids)],
            "responsible_ids": [person_ids[(index + 1) % len(person_ids)]],
            "consulted_ids": [person_ids[(index + 2) % len(person_ids)]],
            "informed_ids": [person_ids[(index + 3) % len(person_ids)]],
        })
    workflows = []
    for index, workflow in enumerate(WORKFLOWS):
        workflows.append({
            "workflow_id": workflow,
            "requester_id": person_ids[index % len(person_ids)],
            "approver_id": person_ids[(index + 1) % len(person_ids)],
            "executor_id": person_ids[(index + 2) % len(person_ids)],
            "verifier_id": person_ids[(index + 3) % len(person_ids)],
            "reviewer_ids": [person_ids[(index + 4) % len(person_ids)], person_ids[(index + 5) % len(person_ids)]],
            "authority_scope": f"bounded.{workflow}",
            "change_ticket_required": True,
            "emergency_override_allowed": workflow == "emergency_stop",
            "production_control_granted": False,
        })
    competency_records = []
    for index, person_id in enumerate(person_ids):
        for competency in sorted(set(SHIFT_ROLE_COMPETENCY.values())):
            competency_records.append({
                "person_id": person_id,
                "competency": competency,
                "valid_until": (end + timedelta(days=365)).isoformat(),
                "evidence_reference": f"competency.{index:02d}.{competency}",
            })
    shift_roster = []
    for day_index in range(roster_days):
        current = start + timedelta(days=day_index)
        for shift_index, shift in enumerate(SHIFTS):
            for role_index, role in enumerate(SHIFT_ROLES):
                primary_index = (shift_index * len(SHIFT_ROLES) + role_index) % len(person_ids)
                backup_index = (primary_index + 1) % len(person_ids)
                shift_roster.append({
                    "date": current.isoformat(),
                    "shift": shift,
                    "role": role,
                    "primary_id": person_ids[primary_index],
                    "backup_id": person_ids[backup_index],
                    "handover_receipt_id": f"handover.{day_index:02d}.{shift}.{role}",
                })
    escalation_matrix = [{
        "severity": severity,
        **targets,
        "incident_commander_id": person_ids[0],
        "first_escalation_id": person_ids[1],
        "second_escalation_id": person_ids[2],
        "communication_plan_reference": f"communication-plan.{severity}",
    } for severity, targets in ESCALATION_TARGETS.items()]
    return {
        "schema_version": DATASET_SCHEMA,
        "site_id": "test-terminal-01",
        "run_id": f"OPERATING.MODEL.{roster_days}.DAYS",
        "source": {
            "source_system": "authorized-identity-roster-and-governance-export",
            "owner": "site-governance-data-owner",
            "license": "authorized-operational-export" if authorized else "contract-test-fixture",
            "timezone": "Asia/Kuala_Lumpur",
            "extracted_at": datetime(end.year, end.month, end.day, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            "evidence_class": "authorized_site_operating_model_export" if authorized else "contract_test_only",
        },
        "bindings": {field: hashlib.sha256(field.encode()).hexdigest() for field in BINDING_FIELDS},
        "roster_period": {"start_date": start.isoformat(), "end_date": end.isoformat(), "shifts_per_day": 3},
        "people": people,
        "responsibility_assignments": assignments,
        "approval_workflows": workflows,
        "competency_records": competency_records,
        "shift_roster": shift_roster,
        "escalation_matrix": escalation_matrix,
    }


def authorized_bundle() -> dict:
    return governance_bundle(authorized=True)


def run_authorized(service: OperatingModelGovernanceService | None = None) -> dict:
    service = service or OperatingModelGovernanceService()
    return service.run(
        authorized_bundle(),
        source_verified=True,
        executive_accountability_approved_by="reviewer.executive-accountability",
        governance_assurance_approved_by="reviewer.governance-assurance",
        maritime_safety_approved_by="reviewer.maritime-safety",
        change_ticket="change.operating-model.202607.001",
    )


class OperatingModelGovernanceTests(unittest.TestCase):
    def test_contract_covers_responsibility_workflows_and_seven_day_roster_without_authority(self):
        result = OperatingModelGovernanceService().run(governance_bundle())
        self.assertTrue(result["valid"], result["errors"])
        evidence = result["evidence"]
        self.assertEqual(evidence["governance_status"], "pass")
        self.assertEqual(evidence["metrics"]["responsibility_domain_count"], 12)
        self.assertEqual(evidence["metrics"]["approval_workflow_count"], 10)
        self.assertEqual(evidence["metrics"]["roster_days"], 7)
        self.assertEqual(evidence["metrics"]["roster_assignment_count"], 105)
        self.assertEqual(evidence["metrics"]["segregation_conflict_count"], 0)
        self.assertFalse(evidence["approved"])
        self.assertFalse(evidence["boundary"]["system_can_assign_roles"])
        self.assertFalse(evidence["boundary"]["self_approval_allowed"])
        self.assertFalse(evidence["boundary"]["production_authority"])

    def test_missing_domain_and_self_approval_are_rejected(self):
        payload = governance_bundle()
        payload["responsibility_assignments"].pop()
        payload["approval_workflows"][0]["approver_id"] = payload["approval_workflows"][0]["requester_id"]
        result = OperatingModelGovernanceService().run(payload)
        self.assertFalse(result["valid"])
        codes = {row["code"] for row in result["errors"]}
        self.assertIn("domain_coverage", codes)
        self.assertIn("segregation_of_duties", codes)

    def test_roster_gap_and_expired_competency_are_rejected(self):
        payload = governance_bundle()
        payload["shift_roster"].pop()
        payload["competency_records"][0]["valid_until"] = "2025-01-01"
        result = OperatingModelGovernanceService().run(payload)
        self.assertFalse(result["valid"])
        codes = {row["code"] for row in result["errors"]}
        self.assertIn("roster_coverage", codes)
        self.assertIn("competency", codes)

    def test_authorized_four_week_model_needs_three_independent_reviewers(self):
        service = OperatingModelGovernanceService()
        result = run_authorized(service)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["evidence"]["approved"])
        self.assertEqual(result["evidence"]["metrics"]["roster_days"], 28)
        self.assertEqual(service.validate_evidence(result["evidence"]), {"valid": True, "errors": [], "production_gate_eligible": True})
        unapproved = service.run(authorized_bundle())
        self.assertTrue(unapproved["valid"])
        self.assertFalse(unapproved["evidence"]["approved"])

    def test_semantic_tampering_is_rejected_after_digest_recalculation(self):
        service = OperatingModelGovernanceService()
        evidence = copy.deepcopy(run_authorized(service)["evidence"])
        evidence["metrics"]["segregation_conflict_count"] = 0
        evidence["metrics"]["active_person_count"] = 999
        body = copy.deepcopy(evidence)
        body.pop("evidence_digest")
        evidence["evidence_digest"] = _digest(body)
        validation = service.validate_evidence(evidence)
        self.assertFalse(validation["valid"])
        self.assertIn("metrics does not reproduce from source_input", validation["errors"])

    def test_readiness_accepts_only_approved_artifact(self):
        service = OperatingModelGovernanceService()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operating-model.json"
            path.write_text(json.dumps(run_authorized(service)["evidence"]), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_OPERATING_MODEL_GOVERNANCE_PATH": str(path)}, clear=False):
                readiness = service.readiness()
        self.assertTrue(readiness["configured_artifact"]["verified"])
        self.assertTrue(readiness["boundary"]["organization_authority_verified"])
        self.assertFalse(readiness["boundary"]["system_can_assign_roles"])

    def test_http_routes_remain_contract_only(self):
        client = TestClient(app)
        readiness = client.get("/api/v3/operating-model-governance/readiness")
        result = client.post("/api/v3/operating-model-governance/run", json=governance_bundle())
        self.assertEqual(readiness.status_code, 200)
        self.assertFalse(readiness.json()["boundary"]["site_operating_model_accepted"])
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["valid"])
        self.assertFalse(result.json()["evidence"]["source"]["source_verified"])
        self.assertFalse(result.json()["evidence"]["approved"])
        self.assertFalse(result.json()["boundary"]["system_can_assign_roles"])

    def test_cli_writes_new_evidence_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path, output_path = Path(directory) / "input.json", Path(directory) / "operating-model.v1.json"
            input_path.write_text(json.dumps(governance_bundle()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = governance_cli_main(["--input", str(input_path), "--output", str(output_path)])
            summary = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(summary["responsibility_domain_count"], 12)
            self.assertFalse(summary["approved"])
            with self.assertRaises(FileExistsError):
                governance_cli_main(["--input", str(input_path), "--output", str(output_path)])


if __name__ == "__main__":
    unittest.main()
