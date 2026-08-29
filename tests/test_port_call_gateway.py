from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app import server
from app.services.port_call_gateway import PortCallGateway


def _bundle(*, evidence_class: str = "contract_test_only") -> dict:
    retrieved_at = datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc)
    source_updated_at = retrieved_at - timedelta(minutes=1)
    schedule = [
        ("ARR", "port_arrival", "instant", 1),
        ("BERTH-START", "berth", "start", 2),
        ("CARGO-START", "cargo_operations", "start", 3),
        ("CARGO-COMPLETE", "cargo_operations", "complete", 9),
        ("BERTH-COMPLETE", "berth", "complete", 10),
        ("DEP", "port_departure", "instant", 11),
    ]
    events = []
    for suffix, event_type, event_side, hour in schedule:
        events.append(
            {
                "event_id": f"EV.PC001.{suffix}.1",
                "port_call_id": "PC.CNSHA.001",
                "vessel_name": "PORT DT CONTRACT TEST",
                "vessel_imo": "9176187",
                "vessel_mmsi": "563123456",
                "port_unlocode": "CNSHA",
                "terminal_id": "TERM.YS.1",
                "berth_id": "BERTH.03",
                "event_type": event_type,
                "event_phase": "planned",
                "event_side": event_side,
                "event_time": (retrieved_at + timedelta(hours=hour)).isoformat(),
                "source_updated_at": source_updated_at.isoformat(),
                "revision": 1,
                "source_reference": f"TOS.PC001.{suffix}.1",
            }
        )
    return {
        "schema_version": "port_call_event.v1",
        "site_id": "SITE.CNSHA.TEST",
        "source": {
            "source_system": "PORT_DT_CONTRACT_TEST",
            "owner": "PORT_DT_CONTRACT_TEST",
            "license": "contract_test_only_not_authorized",
            "timezone": "Asia/Shanghai",
            "retrieved_at": retrieved_at.isoformat(),
            "evidence_class": evidence_class,
        },
        "events": events,
    }


class PortCallGatewayTests(unittest.TestCase):
    def test_unconfigured_gateway_is_explicitly_unavailable_and_secret_safe(self):
        with patch.dict(os.environ, {}, clear=True):
            status = PortCallGateway().source_status()
        self.assertEqual(status["mode"], "unavailable")
        self.assertFalse(status["live_data_verified"])
        self.assertFalse(status["fallback_simulator"])
        self.assertIn("base_url_missing", status["blockers"])
        self.assertNotIn("base_url", status)
        self.assertNotIn("token", status)

    def test_invalid_numeric_configuration_fails_closed_without_crashing(self):
        env = {
            "PORT_DT_PORT_CALL_BASE_URL": "https://port-gateway.example",
            "PORT_DT_PORT_CALL_AUTH_MODE": "gateway",
            "PORT_DT_PORT_CALL_SITE_ID": "SITE.CNSHA.01",
            "PORT_DT_PORT_CALL_OWNER": "AUTHORIZED_PORT_OPERATOR",
            "PORT_DT_PORT_CALL_LICENSE": "AUTHORIZED_INTERNAL_OPERATIONS",
            "PORT_DT_PORT_CALL_TIMEOUT_SEC": "not-a-number",
            "PORT_DT_PORT_CALL_MAX_AGE_SEC": "10",
        }
        with patch.dict(os.environ, env, clear=True):
            status = PortCallGateway().source_status()
        self.assertFalse(status["query_ready"])
        self.assertIn("timeout_invalid", status["blockers"])
        self.assertIn("max_age_invalid", status["blockers"])

    def test_contract_bundle_normalizes_and_stays_non_live(self):
        now = datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc)
        with patch.dict(os.environ, {}, clear=True):
            result = PortCallGateway().validate_bundle(_bundle(), now=now)
        self.assertTrue(result["valid"])
        self.assertEqual(result["accepted_event_count"], 6)
        self.assertEqual(len(result["evidence_digest"]), 64)
        self.assertFalse(result["boundary"]["live_data_verified"])
        self.assertFalse(result["boundary"]["dispatch_allowed"])
        self.assertEqual(result["boundary"]["site_status"], "待接入港口")
        self.assertEqual(result["warnings"][0]["code"], "contract_only")
        normalized = result["normalized_bundle"]["events"]
        self.assertTrue(all(row["event_time"].endswith("Z") for row in normalized))

    def test_invalid_identity_duplicate_and_naive_time_are_rejected(self):
        payload = _bundle()
        payload["events"][0]["vessel_imo"] = "9176188"
        payload["events"][0]["vessel_mmsi"] = ""
        payload["events"][1]["event_id"] = payload["events"][0]["event_id"]
        payload["events"][2]["event_time"] = "2026-08-26T05:00:00"
        with patch.dict(os.environ, {}, clear=True):
            result = PortCallGateway().validate_bundle(
                payload,
                now=datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc),
            )
        self.assertFalse(result["valid"])
        codes = {item["code"] for item in result["errors"]}
        self.assertIn("imo_checksum", codes)
        self.assertIn("duplicate_event_id", codes)
        self.assertIn("event_time", codes)
        self.assertIsNone(result["normalized_bundle"])

    def test_sequence_validation_uses_the_latest_event_revision(self):
        payload = _bundle()
        first_berth = payload["events"][1]
        first_berth["event_time"] = "2026-08-26T10:00:00+00:00"
        revised_berth = deepcopy(first_berth)
        revised_berth.update(
            {
                "event_id": "EV.PC001.BERTH-START.2",
                "event_time": "2026-08-26T04:00:00+00:00",
                "source_updated_at": "2026-08-26T01:59:30+00:00",
                "revision": 2,
                "source_reference": "TOS.PC001.BERTH-START.2",
            }
        )
        payload["events"].append(revised_berth)

        with patch.dict(os.environ, {}, clear=True):
            result = PortCallGateway().validate_bundle(
                payload,
                now=datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(result["valid"])
        self.assertEqual(result["accepted_event_count"], 7)
        self.assertNotIn("port_call_sequence", {item["code"] for item in result["errors"]})

    def test_live_claim_requires_config_attestation_and_matching_governance(self):
        payload = _bundle(evidence_class="authorized_live_api")
        payload["site_id"] = "SITE.CNSHA.01"
        payload["source"]["owner"] = "AUTHORIZED_PORT_OPERATOR"
        payload["source"]["license"] = "AUTHORIZED_INTERNAL_OPERATIONS"
        env = {
            "PORT_DT_PORT_CALL_BASE_URL": "https://port-gateway.example",
            "PORT_DT_PORT_CALL_TOKEN": "test-token-not-a-real-secret",
            "PORT_DT_PORT_CALL_SITE_ID": "SITE.CNSHA.01",
            "PORT_DT_PORT_CALL_OWNER": "AUTHORIZED_PORT_OPERATOR",
            "PORT_DT_PORT_CALL_LICENSE": "AUTHORIZED_INTERNAL_OPERATIONS",
            "PORT_DT_PORT_CALL_LIVE_ATTESTED": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            gateway = PortCallGateway()
            result = gateway.validate_bundle(
                payload,
                connection_verified=True,
                now=datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc),
            )
        self.assertEqual(gateway.source_status()["mode"], "configured_unverified")
        self.assertIn("live_response_not_yet_verified", gateway.source_status()["blockers"])
        self.assertTrue(result["valid"])
        self.assertTrue(result["boundary"]["live_data_verified"])
        self.assertFalse(result["boundary"]["production_authority"])

    def test_successful_live_fetch_unlocks_verified_source_status(self):
        payload = _bundle(evidence_class="authorized_live_api")
        now = datetime.now(timezone.utc)
        payload["site_id"] = "SITE.CNSHA.01"
        payload["source"].update(
            {
                "owner": "AUTHORIZED_PORT_OPERATOR",
                "license": "AUTHORIZED_INTERNAL_OPERATIONS",
                "retrieved_at": now.isoformat(),
            }
        )
        for event in payload["events"]:
            event["source_updated_at"] = (now - timedelta(minutes=1)).isoformat()
        env = {
            "PORT_DT_PORT_CALL_BASE_URL": "https://port-gateway.example",
            "PORT_DT_PORT_CALL_TOKEN": "test-token-not-a-real-secret",
            "PORT_DT_PORT_CALL_SITE_ID": "SITE.CNSHA.01",
            "PORT_DT_PORT_CALL_OWNER": "AUTHORIZED_PORT_OPERATOR",
            "PORT_DT_PORT_CALL_LICENSE": "AUTHORIZED_INTERNAL_OPERATIONS",
            "PORT_DT_PORT_CALL_LIVE_ATTESTED": "true",
        }
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode("utf-8")
        response.__enter__.return_value = response
        with patch.dict(os.environ, env, clear=True), patch(
            "app.services.port_call_gateway.urlopen", return_value=response
        ):
            gateway = PortCallGateway()
            self.assertEqual(gateway.source_status()["mode"], "configured_unverified")
            result = gateway.fetch_events(
                start="2026-08-26T00:00:00+00:00",
                end="2026-08-27T00:00:00+00:00",
                port_unlocode="CNSHA",
            )

        self.assertTrue(result["boundary"]["live_data_verified"])
        self.assertEqual(gateway.source_status()["mode"], "live_rest")
        self.assertIsNotNone(gateway.source_status()["last_verified_at"])

    def test_port_call_api_is_fail_closed_without_site_configuration(self):
        client = TestClient(server.app)
        readiness = client.get("/api/v3/port-call/readiness")
        self.assertEqual(readiness.status_code, 200)
        ready_payload = readiness.json()
        self.assertFalse(ready_payload["boundary"]["live_data_verified"])
        self.assertEqual(ready_payload["boundary"]["site_status"], "待接入港口")
        self.assertFalse(ready_payload["adapter_status"]["fallback_simulator"])

        validated = client.post("/api/v3/port-call/validate", json=_bundle())
        self.assertEqual(validated.status_code, 200)
        self.assertTrue(validated.json()["valid"])
        self.assertFalse(validated.json()["boundary"]["live_data_verified"])

        unavailable = client.get(
            "/api/v3/port-call/events",
            params={
                "start": "2026-08-26T00:00:00Z",
                "end": "2026-08-27T00:00:00Z",
                "port_unlocode": "CNSHA",
            },
        )
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["detail"], "port call gateway is not configured")


if __name__ == "__main__":
    unittest.main()
