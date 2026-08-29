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
from app.services.maritime_interoperability import (
    DCSA_SCOPES,
    IHO_PRODUCTS,
    MaritimeInteroperabilityService,
)
from scripts.verify_maritime_interoperability import main as interoperability_cli_main


def interoperability_bundle(*, evidence_class: str = "contract_test_only", site_id: str = "SITE.CNSHA.CONTRACT") -> dict:
    base = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    port_call_id = "3910eb91-8791-4699-8029-8bba8cedb6f5"
    terminal_call_id = "b9f0e5ef-03f7-41bc-884a-87be971cb76a"
    services = [
        {
            "event_id": "3d58910e-36d8-4bc1-84f8-5d285b0cf475",
            "service_id": "34b65730-7c01-49ce-a065-46f945eeadcc",
            "event_type": "pilotage",
            "event_phase": "estimated",
            "event_side": "start",
            "event_time": (base + timedelta(hours=6)).isoformat(),
            "updated_at": (base + timedelta(hours=1)).isoformat(),
            "port_call_phase": "inbound",
            "facility_type": "pilot_boarding_place",
            "facility_code": "CN01",
            "facility_code_list_provider": "SMDG",
            "location_name": "Contract pilot boarding place",
            "source_reference": "PCS.EVENT.PILOTAGE.EST.R1",
        },
        {
            "event_id": "fb32ff28-5dc0-4b1b-b10e-8d7538bc3de4",
            "service_id": "f495f439-862f-4767-8d4b-4a4aa1b67bcd",
            "event_type": "berth",
            "event_phase": "planned",
            "event_side": "start",
            "event_time": (base + timedelta(hours=8)).isoformat(),
            "updated_at": (base + timedelta(hours=2)).isoformat(),
            "port_call_phase": "inbound",
            "facility_type": "berth",
            "facility_code": "CNSHA1",
            "facility_code_list_provider": "SMDG",
            "location_name": "Contract berth one",
            "source_reference": "TOS.EVENT.BERTH.PLN.R2",
        },
        {
            "event_id": "1a49a916-0029-47cc-8a95-fc46793857cf",
            "service_id": "41660216-d2a8-45c0-adc9-025f6e9ff39b",
            "event_type": "cargo_operations",
            "event_phase": "actual",
            "event_side": "complete",
            "event_time": (base + timedelta(hours=18)).isoformat(),
            "updated_at": (base + timedelta(hours=18, minutes=5)).isoformat(),
            "port_call_phase": "alongside",
            "facility_type": "berth",
            "facility_code": "CNSHA1",
            "facility_code_list_provider": "SMDG",
            "location_name": "Contract berth one",
            "source_reference": "TOS.EVENT.CARGO.ACT.R1",
        },
        {
            "event_id": "21c11dce-2334-43f8-b609-f3cf699c27d8",
            "service_id": "36f14df3-d53b-4ee1-b601-e8c0bb4fed3f",
            "event_type": "moves",
            "event_phase": "",
            "event_side": "instant",
            "event_time": "",
            "updated_at": (base + timedelta(hours=2)).isoformat(),
            "port_call_phase": "",
            "facility_type": "berth",
            "facility_code": "CNSHA1",
            "facility_code_list_provider": "SMDG",
            "location_name": "Contract berth one",
            "source_reference": "TOS.EVENT.MOVES.R1",
            "move_forecast": {
                "carrier_code": "TEST",
                "carrier_code_list_provider": "SMDG",
                "restow_units": {"totalUnits": 12},
                "load_units": {"totalUnits": {"size20Units": 80, "size40Units": 120}},
                "discharge_units": {"ladenUnits": {"totalUnits": 150}},
            },
        },
    ]
    products = []
    for index, (product, edition) in enumerate(IHO_PRODUCTS.items(), start=1):
        products.append({
            "product_specification": product,
            "edition": edition,
            "dataset_id": f"CONTRACT.CNSHA.{product}.001",
            "producer_code": "TEST",
            "issue_date": (base - timedelta(days=1)).isoformat(),
            "valid_from": base.isoformat(),
            "valid_to": (base + timedelta(days=30)).isoformat(),
            "coverage_bbox": {"west": 121.0, "south": 30.5, "east": 122.0, "north": 31.5},
            "dataset_sha256": format(index, "064x"),
            "official_source": evidence_class == "authorized_site_interoperability_export",
            "source_reference": f"S100.CATALOG.{product}.ED{edition}",
        })
    return {
        "schema_version": "maritime_interoperability_dataset.v1",
        "site_id": site_id,
        "run_id": "INTEROPERABILITY.CONTRACT.202605",
        "source": {
            "source_system": "PORT_DT_CONTRACT_TEST",
            "owner": "PORT_DT_CONTRACT_TEST",
            "license": "contract_test_only_not_authorized",
            "timezone": "Asia/Shanghai",
            "extracted_at": (base + timedelta(hours=20)).isoformat(),
            "evidence_class": evidence_class,
        },
        "bindings": {
            "port_call_event_digest": "a" * 64,
            "collaboration_evidence_digest": "b" * 64,
        },
        "port_call": {
            "port_call_id": port_call_id,
            "terminal_call_id": terminal_call_id,
            "port_visit_reference": "CNSHA-PORT-VISIT-2026-001",
            "terminal_call_reference": "CNSHA-TERMINAL-CALL-2026-001",
            "sequence_number": 1,
            "port_unlocode": "CNSHA",
            "latitude": 31.23,
            "longitude": 121.48,
            "vessel": {
                "imo_number": "9176187",
                "mmsi": "413123456",
                "name": "CONTRACT VESSEL",
                "type_code": "CONT",
            },
        },
        "operational_events": services,
        "maritime_single_window": {
            "declaration_type": "general_declaration",
            "arrival_departure_code": "A",
            "submission_reference": "MSW.CONTRACT.ARRIVAL.001",
            "submission_at": (base + timedelta(hours=3)).isoformat(),
            "values": {
                "IMO0140": "9176187",
                "IMO0142": "CONTRACT VESSEL",
                "IMO0108": "CNSHA",
                "IMO0064": (base + timedelta(hours=6)).isoformat(),
            },
        },
        "s100_catalog": products,
        "external_conformance": [],
    }


def authorized_bundle(site_id: str = "SITE.CNSHA.CONTRACT") -> dict:
    payload = interoperability_bundle(evidence_class="authorized_site_interoperability_export", site_id=site_id)
    payload["source"].update(
        source_system="AUTHORIZED_SITE_INTEROPERABILITY_EXPORT",
        owner="AUTHORIZED_PORT_OPERATOR",
        license="AUTHORIZED_INTERNAL_INTEROPERABILITY_REVIEW",
    )
    draft = MaritimeInteroperabilityService().run(payload)
    if not draft["valid"]:
        raise AssertionError(draft["errors"])
    mapping_digest = draft["mapping_digest"]
    payload["external_conformance"] = [
        {
            "profile": "dcsa_port_call",
            "report_id": "DCSA-CONFORMANCE-PORT-CALL-2026-001",
            "report_sha256": "c" * 64,
            "tested_mapping_digest": mapping_digest,
            "executed_at": "2026-05-03T00:00:00Z",
            "executor": "independent-dcsa-conformance-lab",
            "scope": list(DCSA_SCOPES),
            "passed": True,
            "source_reference": "DCSA.CONFORMANCE.REPORT.001",
        },
        {
            "profile": "imo_msw",
            "report_id": "MSW-ACCEPTANCE-2026-001",
            "report_sha256": "d" * 64,
            "tested_mapping_digest": mapping_digest,
            "executed_at": "2026-05-03T01:00:00Z",
            "executor": "authorized-maritime-single-window-test-environment",
            "scope": ["General Declaration arrival minimum semantic subset"],
            "passed": True,
            "source_reference": "MSW.ACCEPTANCE.REPORT.001",
        },
        {
            "profile": "iho_s100",
            "report_id": "S100-CATALOG-VALIDATION-2026-001",
            "report_sha256": "e" * 64,
            "tested_mapping_digest": mapping_digest,
            "executed_at": "2026-05-03T02:00:00Z",
            "executor": "authorized-hydrographic-data-validation-environment",
            "scope": list(IHO_PRODUCTS),
            "passed": True,
            "source_reference": "S100.VALIDATION.REPORT.001",
        },
    ]
    return payload


def approved_evidence() -> dict:
    result = MaritimeInteroperabilityService().run(
        authorized_bundle(),
        source_verified=True,
        data_governance_approved_by="site-data-governance-reviewer",
        maritime_authority_approved_by="site-maritime-authority-reviewer",
        hydrographic_authority_approved_by="site-hydrographic-authority-reviewer",
        change_ticket="INTEROPERABILITY-CHANGE-2026-05",
    )
    if not result["valid"]:
        raise AssertionError(result["errors"])
    return result["evidence"]


class MaritimeInteroperabilityTests(unittest.TestCase):
    def test_contract_maps_three_profiles_without_authority_claim(self):
        first = MaritimeInteroperabilityService().run(interoperability_bundle())
        second = MaritimeInteroperabilityService().run(interoperability_bundle())
        self.assertTrue(first["valid"], first["errors"])
        self.assertEqual(first["mapping_digest"], second["mapping_digest"])
        evidence = first["evidence"]
        self.assertEqual(evidence["dcsa_events"][0]["timestamp"]["classifierCode"], "EST")
        self.assertEqual(evidence["dcsa_events"][1]["portCallService"]["portCallServiceTypeCode"], "BERTH")
        self.assertEqual(evidence["dcsa_events"][3]["movesForecasts"][0]["restowUnits"]["totalUnits"], 12)
        self.assertEqual({row["data_number"] for row in evidence["imo_msw_envelope"]["data_elements"]}, {"IMO0140", "IMO0142", "IMO0108", "IMO0064"})
        self.assertEqual({row["product_specification"] for row in evidence["s100_catalog_handoff"]}, set(IHO_PRODUCTS))
        self.assertEqual(evidence["metrics"]["semantic_gap_count"], 0)
        self.assertFalse(evidence["approved"])
        self.assertFalse(first["boundary"]["authority_submission_allowed"])
        self.assertFalse(first["boundary"]["navigational_use_allowed"])
        self.assertFalse(first["boundary"]["official_certification_claim_allowed"])

    def test_identity_version_and_coverage_mismatches_are_rejected(self):
        identity = interoperability_bundle()
        identity["maritime_single_window"]["values"]["IMO0140"] = "9308742"
        result = MaritimeInteroperabilityService().run(identity)
        self.assertFalse(result["valid"])
        self.assertIn("msw_identity", {row["code"] for row in result["errors"]})

        edition = interoperability_bundle()
        edition["s100_catalog"][2]["edition"] = "1.0.0"
        result = MaritimeInteroperabilityService().run(edition)
        self.assertFalse(result["valid"])
        self.assertIn("s100_edition", {row["code"] for row in result["errors"]})

        coverage = interoperability_bundle()
        coverage["s100_catalog"][0]["coverage_bbox"] = {"west": 0, "south": 0, "east": 1, "north": 1}
        result = MaritimeInteroperabilityService().run(coverage)
        self.assertFalse(result["valid"])
        self.assertIn("s100_coverage", {row["code"] for row in result["errors"]})

    def test_external_reports_must_bind_current_mapping_digest(self):
        payload = authorized_bundle()
        payload["operational_events"][0]["event_time"] = "2026-05-01T07:00:00Z"
        result = MaritimeInteroperabilityService().run(payload)
        self.assertFalse(result["valid"])
        self.assertIn("stale_external_report", {row["code"] for row in result["errors"]})

    def test_authorized_site_requires_three_independent_reviewers(self):
        evidence = approved_evidence()
        self.assertTrue(evidence["approved"])
        self.assertTrue(evidence["boundary"]["external_conformance_verified"])
        self.assertTrue(evidence["boundary"]["site_interoperability_accepted"])
        self.assertFalse(evidence["boundary"]["authority_submission_allowed"])
        validation = MaritimeInteroperabilityService.validate_evidence(evidence)
        self.assertTrue(validation["production_gate_eligible"], validation["errors"])

        self_approved = MaritimeInteroperabilityService().run(
            authorized_bundle(),
            source_verified=True,
            data_governance_approved_by="AUTHORIZED_PORT_OPERATOR",
            maritime_authority_approved_by="site-maritime-authority-reviewer",
            hydrographic_authority_approved_by="site-hydrographic-authority-reviewer",
            change_ticket="INTEROPERABILITY-CHANGE-2026-05",
        )
        self.assertTrue(self_approved["valid"], self_approved["errors"])
        self.assertFalse(self_approved["evidence"]["approved"])
        self.assertIn("independent_review_invalid", {row["code"] for row in self_approved["warnings"]})

    def test_tampering_mapped_message_breaks_evidence_digest_and_metrics(self):
        evidence = approved_evidence()
        evidence["dcsa_events"][0]["timestamp"]["classifierCode"] = "ACT"
        validation = MaritimeInteroperabilityService.validate_evidence(evidence)
        self.assertFalse(validation["valid"])
        self.assertIn("mapping_digest does not bind all mapped standards messages", validation["errors"])
        self.assertIn("evidence_digest does not match interoperability evidence content", validation["errors"])

    def test_readiness_accepts_only_approved_versioned_artifact(self):
        evidence = approved_evidence()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "maritime_interoperability_v1.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_MARITIME_INTEROPERABILITY_PATH": str(path)}, clear=True):
                readiness = MaritimeInteroperabilityService().readiness()
        self.assertTrue(readiness["configured_artifact"]["verified"])
        self.assertTrue(readiness["configured_artifact"]["production_gate_eligible"])
        self.assertTrue(readiness["boundary"]["site_interoperability_accepted"])
        self.assertNotIn(str(path), json.dumps(readiness))

    def test_api_runs_contract_but_cannot_submit_or_navigate(self):
        client = TestClient(server.app)
        readiness = client.get("/api/v3/maritime-interoperability/readiness")
        self.assertEqual(readiness.status_code, 200)
        response = client.post("/api/v3/maritime-interoperability/run", json=interoperability_bundle())
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["boundary"]["authority_submission_allowed"])
        self.assertFalse(payload["boundary"]["navigational_use_allowed"])
        self.assertFalse(payload["boundary"]["production_authority"])

    def test_cli_writes_once_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "interoperability.json"
            output = root / "evidence_v1.json"
            source.write_text(json.dumps(interoperability_bundle()), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = interoperability_cli_main(["--input", str(source), "--output", str(output)])
            self.assertEqual(code, 0, stdout.getvalue())
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                interoperability_cli_main(["--input", str(source), "--output", str(output)])

    def test_run_does_not_mutate_input(self):
        payload = interoperability_bundle()
        original = deepcopy(payload)
        MaritimeInteroperabilityService().run(payload)
        self.assertEqual(payload, original)


if __name__ == "__main__":
    unittest.main()
