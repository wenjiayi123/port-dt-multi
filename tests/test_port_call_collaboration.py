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
from app.services.port_call_collaboration import PortCallCollaborationService
from scripts.verify_port_call_collaboration import main as collaboration_cli_main


ROLES = (
    "shipping_line",
    "vessel_agent",
    "terminal",
    "pilotage",
    "towage",
    "port_authority",
)
EVENTS = (
    "port_arrival",
    "pilotage_start",
    "berth_start",
    "cargo_start",
    "cargo_complete",
    "berth_complete",
    "port_departure",
)


def collaboration_bundle(*, evidence_class: str = "contract_test_only", bind_responses: bool = True) -> dict:
    base = datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc)
    participants = [
        {
            "participant_id": f"PARTY.{role.upper()}",
            "role": role,
            "organization": f"CONTRACT {role}",
            "source_reference": f"IDENTITY.{role.upper()}.V1",
            "authorized": evidence_class == "authorized_site_collaboration_export",
        }
        for role in ROLES
    ]
    resources = [
        {"resource_id": "BERTH.03", "resource_type": "berth", "capacity": 1, "owner_participant_id": "PARTY.TERMINAL", "source_reference": "TOS.RESOURCE.BERTH.03"},
        {"resource_id": "PILOT.01", "resource_type": "pilot", "capacity": 1, "owner_participant_id": "PARTY.PILOTAGE", "source_reference": "PILOT.RESOURCE.01"},
        {"resource_id": "TUG.01", "resource_type": "tug", "capacity": 1, "owner_participant_id": "PARTY.TOWAGE", "source_reference": "TUG.RESOURCE.01"},
    ]

    def port_call(call_id: str, imo: str, priority: int, hours: tuple[float, ...]) -> dict:
        milestones = []
        previous = None
        for index, (event_type, hour) in enumerate(zip(EVENTS, hours)):
            milestone_id = f"{call_id}.M{index + 1:02d}"
            if event_type == "pilotage_start":
                resource_ids = ["PILOT.01", "TUG.01"]
                duration = 30
            elif event_type == "berth_start":
                resource_ids = ["BERTH.03"]
                duration = 480 if call_id.endswith("001") else 360
            else:
                resource_ids = []
                duration = 0
            milestones.append({
                "milestone_id": milestone_id,
                "event_type": event_type,
                "planned_at": (base + timedelta(hours=hour)).isoformat(),
                "duration_minutes": duration,
                "depends_on": [] if previous is None else [previous],
                "resource_ids": resource_ids,
                "source_reference": f"PCS.{milestone_id}.R1",
            })
            previous = milestone_id
        return {
            "port_call_id": call_id,
            "vessel_name": f"CONTRACT VESSEL {call_id[-3:]}",
            "vessel_imo": imo,
            "priority": priority,
            "milestones": milestones,
        }

    calls = [
        port_call("PC.CNSHA.001", "9176187", 1, (6, 7, 8, 8.5, 15.5, 16, 17)),
        port_call("PC.CNSHA.002", "9308742", 2, (14, 15, 17, 17.5, 22.5, 23, 24)),
    ]
    payload = {
        "schema_version": "port_call_collaboration_dataset.v1",
        "site_id": "SITE.CNSHA.CONTRACT",
        "run_id": "COLLABORATION.CONTRACT.202604",
        "collaboration_revision": 2,
        "port_call_event_digest": "d" * 64,
        "source": {
            "source_system": "PORT_DT_CONTRACT_TEST",
            "owner": "PORT_DT_CONTRACT_TEST",
            "license": "contract_test_only_not_authorized",
            "timezone": "Asia/Shanghai",
            "extracted_at": (base + timedelta(hours=31)).isoformat(),
            "evidence_class": evidence_class,
        },
        "participants": participants,
        "resources": resources,
        "port_calls": calls,
        "disruptions": [{
            "disruption_id": "DELAY.PC001.ARRIVAL.01",
            "port_call_id": "PC.CNSHA.001",
            "milestone_id": "PC.CNSHA.001.M01",
            "delay_minutes": 120,
            "occurred_at": (base + timedelta(hours=5)).isoformat(),
            "reason": "restricted visibility at anchorage",
            "source_reference": "VTS.DELAY.PC001.01",
        }],
        "responses": [],
    }
    if bind_responses:
        draft = PortCallCollaborationService().run(deepcopy(payload))
        if not draft["valid"]:
            raise AssertionError(draft["errors"])
        proposal_digest = draft["evidence"]["proposal_digest"]
        payload["responses"] = [
            {
                "participant_id": row["participant_id"],
                "revision": 2,
                "proposal_digest": proposal_digest,
                "disposition": "acknowledged",
                "reason": "reviewed recommendation-only revision",
                "responded_at": (base + timedelta(hours=30)).isoformat(),
                "source_reference": f"ACK.{row['role'].upper()}.R2",
            }
            for row in participants
        ]
    return payload


def authorized_bundle() -> dict:
    payload = collaboration_bundle(evidence_class="authorized_site_collaboration_export")
    payload["source"].update(
        source_system="AUTHORIZED_PORT_CALL_COLLABORATION_EXPORT",
        owner="AUTHORIZED_PORT_OPERATOR",
        license="AUTHORIZED_INTERNAL_COLLABORATION_REVIEW",
    )
    return payload


def approved_evidence() -> dict:
    result = PortCallCollaborationService().run(
        authorized_bundle(),
        source_verified=True,
        terminal_operations_approved_by="terminal-operations-reviewer",
        port_authority_approved_by="harbour-master-reviewer",
        change_ticket="PORT-CALL-CHANGE-2026-04",
    )
    if not result["valid"]:
        raise AssertionError(result["errors"])
    return result["evidence"]


class PortCallCollaborationTests(unittest.TestCase):
    def test_draft_requires_receipts_bound_to_the_exact_proposal_digest(self):
        draft = PortCallCollaborationService().run(collaboration_bundle(bind_responses=False))
        self.assertTrue(draft["valid"], draft["errors"])
        self.assertEqual(draft["evidence"]["collaboration_status"], "needs_revision")
        self.assertEqual(draft["evidence"]["metrics"]["acknowledgement_coverage_rate"], 0.0)
        self.assertRegex(draft["evidence"]["proposal_digest"], "^[a-f0-9]{64}$")

        stale = collaboration_bundle()
        stale["responses"][0]["proposal_digest"] = "f" * 64
        result = PortCallCollaborationService().run(stale)
        self.assertFalse(result["valid"])
        self.assertIn("stale_proposal_response", {row["code"] for row in result["errors"]})

    def test_contract_propagates_delay_resolves_conflict_without_site_claim(self):
        first = PortCallCollaborationService().run(collaboration_bundle())
        second = PortCallCollaborationService().run(collaboration_bundle())
        self.assertTrue(first["valid"], first["errors"])
        self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
        evidence = first["evidence"]
        metrics = evidence["metrics"]
        self.assertEqual(metrics["participant_count"], 6)
        self.assertEqual(metrics["participant_coverage_rate"], 1.0)
        self.assertEqual(metrics["port_call_count"], 2)
        self.assertEqual(metrics["conflicts_before_replan"], 1)
        self.assertEqual(metrics["conflicts_after_replan"], 0)
        self.assertEqual(metrics["replan_action_count"], 1)
        self.assertEqual(metrics["max_propagated_delay_minutes"], 120)
        self.assertEqual(evidence["collaboration_status"], "accepted")
        self.assertFalse(evidence["approved"])
        self.assertFalse(first["boundary"]["site_collaboration_accepted"])
        self.assertFalse(first["boundary"]["shared_plan_mutated"])
        self.assertFalse(first["boundary"]["authority_to_change_eta"])
        self.assertEqual(first["warnings"][0]["code"], "contract_only")

    def test_missing_party_broken_dependency_or_unknown_resource_is_rejected(self):
        missing_party = collaboration_bundle()
        missing_party["participants"].pop()
        missing_party["responses"].pop()
        result = PortCallCollaborationService().run(missing_party)
        self.assertFalse(result["valid"])
        self.assertIn("participant_coverage", {row["code"] for row in result["errors"]})

        broken_dependency = collaboration_bundle()
        broken_dependency["port_calls"][0]["milestones"][2]["depends_on"] = []
        result = PortCallCollaborationService().run(broken_dependency)
        self.assertFalse(result["valid"])
        self.assertIn("dependency_chain", {row["code"] for row in result["errors"]})

        unknown_resource = collaboration_bundle()
        unknown_resource["port_calls"][1]["milestones"][2]["resource_ids"] = ["BERTH.UNKNOWN"]
        result = PortCallCollaborationService().run(unknown_resource)
        self.assertFalse(result["valid"])
        self.assertIn("resource_reference", {row["code"] for row in result["errors"]})

    def test_objection_keeps_revision_open_and_blocks_approval(self):
        payload = authorized_bundle()
        payload["responses"][3].update(disposition="objected", reason="pilot window unavailable")
        result = PortCallCollaborationService().run(
            payload,
            source_verified=True,
            terminal_operations_approved_by="terminal-operations-reviewer",
            port_authority_approved_by="harbour-master-reviewer",
            change_ticket="PORT-CALL-CHANGE-2026-04",
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["evidence"]["collaboration_status"], "needs_revision")
        self.assertEqual(result["evidence"]["metrics"]["unresolved_objection_count"], 1)
        self.assertFalse(result["evidence"]["approved"])
        self.assertIn("unresolved_objection", {row["code"] for row in result["warnings"]})

    def test_authorized_export_requires_independent_approvals(self):
        unattested = PortCallCollaborationService().run(authorized_bundle())
        self.assertTrue(unattested["valid"], unattested["errors"])
        self.assertFalse(unattested["evidence"]["approved"])

        evidence = approved_evidence()
        self.assertTrue(evidence["approved"])
        self.assertTrue(evidence["boundary"]["site_collaboration_accepted"])
        self.assertTrue(evidence["boundary"]["dual_approval_verified"])
        self.assertFalse(evidence["boundary"]["production_authority"])
        validation = PortCallCollaborationService.validate_evidence(evidence)
        self.assertTrue(validation["production_gate_eligible"], validation["errors"])

        self_approved = PortCallCollaborationService().run(
            authorized_bundle(),
            source_verified=True,
            terminal_operations_approved_by="AUTHORIZED_PORT_OPERATOR",
            port_authority_approved_by="harbour-master-reviewer",
            change_ticket="PORT-CALL-CHANGE-2026-04",
        )
        self.assertFalse(self_approved["evidence"]["approved"])
        self.assertIn("dual_approval_invalid", {row["code"] for row in self_approved["warnings"]})

    def test_tampering_timeline_or_metrics_breaks_evidence(self):
        evidence = approved_evidence()
        evidence["proposed_timeline"][2]["planned_at"] = evidence["baseline_timeline"][2]["planned_at"]
        validation = PortCallCollaborationService.validate_evidence(evidence)
        self.assertFalse(validation["valid"])
        self.assertIn("evidence_digest does not match collaboration evidence content", validation["errors"])
        self.assertIn("metrics do not match timeline, conflicts and participant responses", validation["errors"])

    def test_readiness_accepts_only_approved_versioned_artifact(self):
        evidence = approved_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "port_call_collaboration_v1.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_PORT_CALL_COLLABORATION_PATH": str(path)}, clear=True):
                readiness = PortCallCollaborationService().readiness()
        self.assertTrue(readiness["configured_artifact"]["verified"])
        self.assertTrue(readiness["configured_artifact"]["production_gate_eligible"])
        self.assertTrue(readiness["boundary"]["site_collaboration_accepted"])
        self.assertNotIn(str(path), json.dumps(readiness))

    def test_api_runs_contract_but_cannot_approve_or_mutate_plan(self):
        client = TestClient(server.app)
        readiness = client.get("/api/v3/port-call-collaboration/readiness")
        self.assertEqual(readiness.status_code, 200)
        response = client.post("/api/v3/port-call-collaboration/run", json=collaboration_bundle())
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["evidence"]["approved"])
        self.assertFalse(payload["boundary"]["shared_plan_mutated"])
        self.assertFalse(payload["boundary"]["dispatch_allowed"])

    def test_cli_writes_once_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "collaboration.json"
            output = root / "evidence_v1.json"
            source.write_text(json.dumps(collaboration_bundle()), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = collaboration_cli_main(["--input", str(source), "--output", str(output)])
            self.assertEqual(code, 0, stdout.getvalue())
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                collaboration_cli_main(["--input", str(source), "--output", str(output)])

    def test_run_does_not_mutate_input(self):
        payload = collaboration_bundle()
        original = deepcopy(payload)
        PortCallCollaborationService().run(payload)
        self.assertEqual(payload, original)


if __name__ == "__main__":
    unittest.main()
