from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app.server import app
from app.services.end_to_end_coordination import (
    BINDING_FIELDS,
    CAPACITY_UNITS,
    DATASET_SCHEMA,
    EVIDENCE_SCHEMA,
    FIXED_PLANNING,
    OBJECTIVE_WEIGHTS,
    RESOURCE_TYPES,
    EndToEndCoordinationService,
)
from scripts.verify_end_to_end_coordination import main as coordination_cli_main


BASELINES = {
    "port_arrival": ("2026-08-01T06:15:00Z", 15),
    "channel_transit": ("2026-08-01T06:45:00Z", 30),
    "berthing": ("2026-08-01T07:30:00Z", 30),
    "cargo_operation": ("2026-08-01T08:15:00Z", 120),
    "yard_transfer": ("2026-08-01T10:30:00Z", 30),
    "yard_operation": ("2026-08-01T11:15:00Z", 60),
    "gate_release": ("2026-08-01T12:30:00Z", 45),
    "rail_release": ("2026-08-01T12:30:00Z", 45),
    "maintenance": ("2026-08-01T12:45:00Z", 60),
}
REQUIREMENTS = {
    "port_arrival": {"channel_slot": 1.0},
    "channel_transit": {"channel_slot": 1.0, "pilot": 1.0, "tug": 1.0},
    "berthing": {"berth": 1.0},
    "cargo_operation": {
        "berth": 1.0,
        "quay_crane": 2.0,
        "horizontal_transport": 4.0,
        "yard_block": 20.0,
        "shore_power_bess": 2500.0,
    },
    "yard_transfer": {"horizontal_transport": 2.0, "yard_block": 20.0},
    "yard_operation": {"yard_block": 20.0},
    "gate_release": {"gate": 1.0},
    "rail_release": {"rail": 1.0},
    "maintenance": {"quay_crane": 1.0, "maintenance_team": 1.0},
}
PREDECESSORS = {
    "port_arrival": [],
    "channel_transit": ["port_arrival"],
    "berthing": ["channel_transit"],
    "cargo_operation": ["berthing"],
    "yard_transfer": ["cargo_operation"],
    "yard_operation": ["yard_transfer"],
    "gate_release": ["yard_operation"],
    "rail_release": ["yard_operation"],
    "maintenance": ["cargo_operation"],
}
CAPACITIES = {
    "channel_slot": 1.0,
    "berth": 1.0,
    "pilot": 1.0,
    "tug": 1.0,
    "quay_crane": 3.0,
    "horizontal_transport": 4.0,
    "yard_block": 40.0,
    "gate": 2.0,
    "rail": 1.0,
    "shore_power_bess": 6000.0,
    "maintenance_team": 1.0,
}


def _end(start: str, duration_minutes: int) -> str:
    from datetime import datetime, timedelta, timezone

    value = datetime.fromisoformat(start.replace("Z", "+00:00")) + timedelta(minutes=duration_minutes)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_bundle(*, chain_count: int = 2, authorized: bool = False, shared_resources: bool = True) -> dict:
    evidence_class = "authorized_site_coordination_export" if authorized else "contract_test_only"
    resources = []
    resource_ids: dict[tuple[int, str], str] = {}
    resource_groups = 1 if shared_resources else chain_count
    for group in range(resource_groups):
        for resource_type in RESOURCE_TYPES:
            resource_id = f"resource.{group}.{resource_type}"
            resource_ids[(group, resource_type)] = resource_id
            resources.append(
                {
                    "resource_id": resource_id,
                    "resource_type": resource_type,
                    "capacity": CAPACITIES[resource_type],
                    "capacity_unit": CAPACITY_UNITS[resource_type],
                    "available_from": "2026-08-01T06:00:00Z",
                    "available_to": "2026-08-02T06:00:00Z",
                    "owner": f"owner.{group}.{resource_type}",
                    "source_reference": f"source.{group}.{resource_type}",
                    "safety_interlock_reference": f"interlock.{group}.{resource_type}",
                }
            )

    tasks = []
    for chain_number in range(chain_count):
        chain_id = f"chain.{chain_number + 1:02d}"
        group = 0 if shared_resources else chain_number
        task_ids = {stage: f"task.{chain_number + 1:02d}.{stage}" for stage in BASELINES}
        for stage, (default_start, duration) in BASELINES.items():
            # Only the first contract chain deliberately enters the frozen window.
            start = default_start
            if stage == "port_arrival" and shared_resources and chain_number > 0:
                start = "2026-08-01T06:30:00Z"
            requirements = REQUIREMENTS[stage]
            tasks.append(
                {
                    "task_id": task_ids[stage],
                    "chain_id": chain_id,
                    "stage": stage,
                    "priority": 5 if chain_number == 0 else 3,
                    "earliest_start": start,
                    "baseline_start": start,
                    "baseline_end": _end(start, duration),
                    "deadline": "2026-08-01T18:00:00Z",
                    "duration_minutes": duration,
                    "frozen": start < "2026-08-01T06:30:00Z",
                    "predecessor_ids": [task_ids[item] for item in PREDECESSORS[stage]],
                    "resource_requirements": requirements,
                    "baseline_allocations": [
                        {"resource_id": resource_ids[(group, resource_type)], "quantity": quantity}
                        for resource_type, quantity in requirements.items()
                    ],
                    "source_reference": f"task-source.{chain_number + 1:02d}.{stage}",
                    "forecast_receipt_id": f"forecast-receipt.{chain_number + 1:02d}.{stage}",
                    "baseline_plan_receipt_id": f"baseline-receipt.{chain_number + 1:02d}.{stage}",
                }
            )
    return {
        "schema_version": DATASET_SCHEMA,
        "site_id": "test-terminal-01",
        "run_id": f"coordination-run-{chain_count:02d}",
        "planning_revision": 1,
        "source": {
            "source_system": "terminal-integrated-planning-system",
            "owner": "site-planning-data-owner",
            "license": "authorized-operational-export" if authorized else "contract-test-fixture",
            "timezone": "Asia/Kuala_Lumpur",
            "extracted_at": "2026-08-01T05:45:00Z",
            "evidence_class": evidence_class,
        },
        "bindings": {field: hashlib.sha256(field.encode("utf-8")).hexdigest() for field in BINDING_FIELDS},
        "planning": {
            "plan_id": "shared-plan.20260801.01",
            "plan_system_reference": "integrated-planning-system.revision.1",
            "decision_at": "2026-08-01T06:00:00Z",
            "horizon_start": "2026-08-01T06:00:00Z",
            "horizon_end": "2026-08-02T06:00:00Z",
            **FIXED_PLANNING,
            "objective_weights": dict(OBJECTIVE_WEIGHTS),
        },
        "resources": resources,
        "tasks": tasks,
        "safety_incident_count": 0,
        "unresolved_exception_ids": [],
    }


def authorized_bundle() -> dict:
    return contract_bundle(chain_count=10, authorized=True, shared_resources=False)


def run_authorized(service: EndToEndCoordinationService | None = None) -> dict:
    service = service or EndToEndCoordinationService()
    return service.run(
        authorized_bundle(),
        source_verified=True,
        integrated_planning_approved_by="reviewer.integrated-planning",
        marine_services_approved_by="reviewer.marine-services",
        terminal_operations_approved_by="reviewer.terminal-operations",
        equipment_energy_approved_by="reviewer.equipment-energy",
        change_ticket="change.coordination.20260801.001",
    )


class TestEndToEndCoordination(unittest.TestCase):
    def test_contract_run_resolves_shared_capacity_conflicts_without_operational_authority(self):
        result = EndToEndCoordinationService().run(contract_bundle())

        assert result["valid"] is True
        evidence = result["evidence"]
        assert evidence["schema_version"] == EVIDENCE_SCHEMA
        assert evidence["coordination_status"] == "pass"
        assert evidence["metrics"]["chain_count"] == 2
        assert evidence["metrics"]["task_count"] == 18
        assert evidence["metrics"]["stage_coverage_count"] == 9
        assert evidence["metrics"]["resource_type_coverage_count"] == 11
        assert evidence["metrics"]["baseline_conflict_slot_count"] > 0
        assert evidence["metrics"]["candidate_conflict_slot_count"] == 0
        assert evidence["metrics"]["maximum_reschedule_minutes"] <= 180
        assert evidence["approved"] is False
        assert result["boundary"] == {
            "site_end_to_end_coordination_accepted": False,
            "recommendation_only": True,
            "shared_plan_mutated": False,
            "automatic_resource_commitment_allowed": False,
            "dispatch_allowed": False,
            "production_authority": False,
            "claim": "contract_or_unapproved_coordination_recommendation",
            "reason": "合同求解、局部策略输出或未复核计划不能形成现场统一资源承诺。",
        }

    def test_contract_rejects_unknown_predecessor_and_incomplete_resource_coverage(self):
        payload = contract_bundle()
        payload["resources"] = payload["resources"][:-1]
        payload["tasks"][1]["predecessor_ids"] = ["task.not-declared"]

        result = EndToEndCoordinationService().run(payload)

        assert result["valid"] is False
        codes = {row["code"] for row in result["errors"]}
        assert "resource_type_coverage" in codes
        assert "unknown_predecessor" in codes

    def test_clean_contract_preserves_blocked_safety_and_exception_evidence(self):
        payload = contract_bundle()
        payload["safety_incident_count"] = 1
        payload["unresolved_exception_ids"] = ["exception.capacity.001"]

        result = EndToEndCoordinationService().run(payload)

        assert result["valid"] is True
        assert result["evidence"]["coordination_status"] == "blocked"
        assert result["evidence"]["threshold_checks"]["safety"] is False
        assert result["evidence"]["threshold_checks"]["no_unresolved_exception"] is False
        assert result["evidence"]["approved"] is False

    def test_authorized_export_needs_scale_attestation_and_four_independent_reviewers(self):
        service = EndToEndCoordinationService()
        result = run_authorized(service)

        assert result["valid"] is True
        assert result["evidence"]["approved"] is True
        assert result["evidence"]["metrics"]["chain_count"] == 10
        assert result["evidence"]["metrics"]["task_count"] == 90
        assert service.validate_evidence(result["evidence"]) == {
            "valid": True,
            "errors": [],
            "production_gate_eligible": True,
        }

        not_attested = service.run(authorized_bundle())
        assert not_attested["valid"] is True
        assert not_attested["evidence"]["approved"] is False

    def test_semantic_candidate_plan_tampering_is_rejected_even_with_recomputed_digest(self):
        service = EndToEndCoordinationService()
        evidence = copy.deepcopy(run_authorized(service)["evidence"])
        evidence["candidate_plan"][0]["candidate_start"] = "2026-08-01T06:30:00Z"
        evidence["candidate_plan_digest"] = _digest(evidence["candidate_plan"])
        body = copy.deepcopy(evidence)
        body.pop("evidence_digest")
        evidence["evidence_digest"] = _digest(body)

        validation = service.validate_evidence(evidence)

        assert validation["valid"] is False
        assert "candidate plan does not reproduce from the fixed solver" in validation["errors"]
        assert validation["production_gate_eligible"] is False

    def test_readiness_accepts_only_a_verified_site_artifact(self):
        service = EndToEndCoordinationService()
        evidence = run_authorized(service)["evidence"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coordination.json"
            path.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
            with mock.patch.dict(os.environ, {"PORT_DT_END_TO_END_COORDINATION_PATH": str(path)}, clear=False):
                readiness = service.readiness()

        assert readiness["configured_artifact"]["mode"] == "verified_site_artifact"
        assert readiness["configured_artifact"]["verified"] is True
        assert readiness["boundary"]["site_end_to_end_coordination_accepted"] is True
        assert readiness["boundary"]["dispatch_allowed"] is False
        assert readiness["boundary"]["production_authority"] is False

    def test_http_routes_remain_contract_only(self):
        client = TestClient(app)

        readiness = client.get("/api/v3/end-to-end-coordination/readiness")
        result = client.post("/api/v3/end-to-end-coordination/run", json=contract_bundle())

        assert readiness.status_code == 200
        assert readiness.json()["boundary"]["site_end_to_end_coordination_accepted"] is False
        assert result.status_code == 200
        assert result.json()["valid"] is True
        assert result.json()["evidence"]["source"]["source_verified"] is False
        assert result.json()["evidence"]["approved"] is False
        assert result.json()["boundary"]["automatic_resource_commitment_allowed"] is False

    def test_cli_writes_versioned_contract_evidence_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "evidence.v1.json"
            input_path.write_text(json.dumps(contract_bundle(), ensure_ascii=False), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = coordination_cli_main(["--input", str(input_path), "--output", str(output_path)])
            summary = json.loads(stdout.getvalue())

            assert status == 0
            assert output_path.exists()
            assert summary["chain_count"] == 2
            assert summary["candidate_conflict_slot_count"] == 0
            assert summary["approved"] is False
            with self.assertRaises(FileExistsError):
                coordination_cli_main(["--input", str(input_path), "--output", str(output_path)])
