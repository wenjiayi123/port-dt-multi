from __future__ import annotations

import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app import server
from app.services.site_integration_gateway import (
    ADAPTER_CONTRACTS,
    SCHEMA_VERSION,
    SiteIntegrationGateway,
    _sha256,
    sign_snapshot,
)


NOW = datetime(2026, 8, 30, 5, 30, tzinfo=timezone.utc)
SECRET = "contract-test-secret-not-for-production"


def _gateway(*, live_attested: bool = False) -> SiteIntegrationGateway:
    return SiteIntegrationGateway(
        site_id="SITE.CNSHA.CONTRACT",
        adapters={
            "tos": {
                "secret": SECRET,
                "owner": "CONTRACT_TEST_OWNER",
                "source_system": "CONTRACT_TEST_TOS",
            }
        },
        live_attested=live_attested,
        max_age_seconds=300,
    )


def _snapshot(*, sequence: int = 1, observed_at: datetime = NOW) -> dict:
    payload = {
        field: {
            "value": 0.6 if unit == "ratio" else 12.0,
            "unit": unit,
            "quality": "measured",
            "observed_at": observed_at.isoformat(),
        }
        for field, unit in ADAPTER_CONTRACTS["tos"].items()
    }
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": f"SNAP.CNSHA.TOS.{sequence:06d}",
        "site_id": "SITE.CNSHA.CONTRACT",
        "adapter_id": "tos",
        "owner": "CONTRACT_TEST_OWNER",
        "source_system": "CONTRACT_TEST_TOS",
        "sequence": sequence,
        "observed_at": observed_at.isoformat(),
        "payload": payload,
        "payload_sha256": _sha256(payload),
    }
    envelope["signature"] = sign_snapshot(envelope, SECRET)
    return envelope


class SiteIntegrationGatewayTests(unittest.TestCase):
    def test_default_readiness_is_fail_closed_for_all_required_adapters(self):
        status = SiteIntegrationGateway().readiness()
        self.assertEqual(status["adapter_count"], 8)
        self.assertEqual(status["configured_adapter_count"], 0)
        self.assertEqual(status["live_verified_adapter_count"], 0)
        self.assertFalse(status["replacement_coverage"]["training_dataset_replacement_ready"])
        self.assertFalse(status["boundary"]["dispatch_allowed"])
        self.assertFalse(status["boundary"]["production_authority"])

    def test_valid_snapshot_is_accepted_without_storing_raw_telemetry(self):
        gateway = _gateway()
        result = gateway.validate_envelope(_snapshot(), now=NOW)
        self.assertTrue(result["valid"])
        self.assertFalse(result["boundary"]["live_data_verified"])
        self.assertNotIn("payload", result)
        self.assertNotIn("payload", gateway._accepted["tos"])
        self.assertEqual(result["lineage"]["field_count"], 4)
        self.assertEqual(len(result["lineage"]["payload_sha256"]), 64)

    def test_live_attestation_marks_source_live_but_never_grants_authority(self):
        gateway = _gateway(live_attested=True)
        result = gateway.validate_envelope(_snapshot(), now=NOW)
        status = gateway.readiness()
        self.assertTrue(result["boundary"]["live_data_verified"])
        self.assertEqual(status["live_verified_adapter_count"], 1)
        self.assertFalse(result["boundary"]["dispatch_allowed"])
        self.assertFalse(result["boundary"]["production_authority"])
        self.assertFalse(status["replacement_coverage"]["training_dataset_replacement_ready"])

    def test_payload_tampering_stale_data_bad_unit_and_bad_range_are_rejected(self):
        envelope = _snapshot(observed_at=NOW - timedelta(minutes=10))
        envelope["payload"]["berth_occupancy_ratio"]["value"] = 1.4
        envelope["payload"]["yard_occupancy_ratio"]["unit"] = "percent"
        result = _gateway().validate_envelope(envelope, now=NOW)
        codes = {item["code"] for item in result["errors"]}
        self.assertFalse(result["valid"])
        self.assertIn("stale_snapshot", codes)
        self.assertIn("payload_sha256", codes)
        self.assertIn("signature", codes)
        self.assertIn("unit", codes)
        self.assertIn("range", codes)

    def test_signature_and_replay_protection_fail_closed(self):
        gateway = _gateway()
        first = _snapshot(sequence=1)
        self.assertTrue(gateway.validate_envelope(first, now=NOW)["valid"])
        replay = gateway.validate_envelope(first, now=NOW)
        self.assertFalse(replay["valid"])
        replay_codes = {item["code"] for item in replay["errors"]}
        self.assertIn("sequence_replay", replay_codes)
        self.assertIn("snapshot_replay", replay_codes)

        second = _snapshot(sequence=2)
        second["signature"] = "0" * 64
        rejected = gateway.validate_envelope(second, now=NOW)
        self.assertFalse(rejected["valid"])
        self.assertIn("signature", {item["code"] for item in rejected["errors"]})

    def test_missing_required_reading_is_rejected(self):
        envelope = _snapshot()
        del envelope["payload"]["throughput_teu"]
        envelope["payload_sha256"] = _sha256(envelope["payload"])
        unsigned = {key: value for key, value in envelope.items() if key != "signature"}
        envelope["signature"] = sign_snapshot(unsigned, SECRET)
        result = _gateway().validate_envelope(envelope, now=NOW)
        self.assertFalse(result["valid"])
        self.assertIn("missing_field", {item["code"] for item in result["errors"]})

    def test_api_defaults_to_unconfigured_and_rejects_unsigned_data(self):
        client = TestClient(server.app)
        readiness = client.get("/api/v3/site-integration/readiness")
        self.assertEqual(readiness.status_code, 200)
        self.assertEqual(readiness.json()["configured_adapter_count"], 0)
        response = client.post("/api/v3/site-integration/ingest", json=deepcopy(_snapshot()))
        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.json()["valid"])
        self.assertFalse(response.json()["boundary"]["production_authority"])


if __name__ == "__main__":
    unittest.main()
