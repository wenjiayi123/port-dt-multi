from __future__ import annotations

import json
import io
import math
import os
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.services.site_twin_calibration import SiteTwinCalibrationService
from app.services.twin_schema.service import TwinSchemaService
from scripts.calibrate_site_twin import main as calibration_cli_main


def calibration_bundle(*, evidence_class: str = "contract_test_only") -> dict:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(72):
        timestamp = start + timedelta(hours=index)
        hour = index % 24
        throughput = 100.0 + 24.0 * math.sin(2.0 * math.pi * hour / 24.0)
        arrivals = float(hour % 4)
        ambient = 28.0 + 5.0 * math.sin(2.0 * math.pi * (hour - 4) / 24.0)
        tide = 1.5 * math.sin(2.0 * math.pi * (hour + 2) / 12.0)
        noise = float((index % 5) - 2) * 0.6
        observed = 800.0 + 2.5 * throughput + 30.0 * arrivals + 15.0 * ambient + 5.0 * tide + noise
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "asset_group": "QC.GROUP.A" if index % 2 == 0 else "QC.GROUP.B",
                "observed_power_kw": observed,
                "throughput_teu": throughput,
                "vessel_arrivals": arrivals,
                "ambient_c": ambient,
                "tide_m": tide,
                "source_reference": f"METER.QC.{index:04d}",
            }
        )
    return {
        "schema_version": "site_twin_calibration_dataset.v1",
        "site_id": "SITE.CNSHA.CONTRACT",
        "dataset_id": "CNSHA.QC.CALIBRATION.CONTRACT.202601",
        "source": {
            "source_system": "PORT_DT_CONTRACT_TEST",
            "owner": "PORT_DT_CONTRACT_TEST",
            "license": "contract_test_only_not_authorized",
            "timezone": "Asia/Shanghai",
            "extracted_at": (start + timedelta(hours=96)).isoformat(),
            "evidence_class": evidence_class,
        },
        "split": {
            "training_window": {
                "start_at": start.isoformat(),
                "end_at": (start + timedelta(hours=47)).isoformat(),
            },
            "validation_window": {
                "start_at": (start + timedelta(hours=48)).isoformat(),
                "end_at": (start + timedelta(hours=71)).isoformat(),
            },
        },
        "model": {
            "target": "observed_power_kw",
            "features": ["throughput_teu", "vessel_arrivals", "ambient_c", "tide_m"],
            "ridge_alpha": 0.1,
        },
        "rows": rows,
    }


class SiteTwinCalibrationTests(unittest.TestCase):
    def test_contract_sample_fits_and_validates_without_site_claim(self):
        service = SiteTwinCalibrationService()
        first = service.run(calibration_bundle())
        second = service.run(calibration_bundle())
        self.assertTrue(first["valid"], first["errors"])
        self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
        evidence = first["evidence"]
        self.assertEqual(evidence["training_rows"], 48)
        self.assertEqual(evidence["validation_rows"], 24)
        self.assertEqual(evidence["validation_status"], "pass")
        self.assertTrue(all(evidence["threshold_checks"].values()))
        self.assertEqual(evidence["uncertainty"]["samples"], 500)
        self.assertEqual(len(evidence["error_decomposition"]["by_asset_group"]), 2)
        self.assertFalse(evidence["measured_outcomes"])
        self.assertFalse(evidence["approved"])
        self.assertFalse(first["boundary"]["site_calibrated"])
        self.assertFalse(first["boundary"]["production_authority"])
        self.assertEqual(first["warnings"][0]["code"], "contract_only")

    def test_overlapping_windows_and_duplicate_observations_are_rejected(self):
        payload = calibration_bundle()
        payload["split"]["validation_window"]["start_at"] = payload["split"]["training_window"]["end_at"]
        duplicate = deepcopy(payload["rows"][0])
        duplicate["source_reference"] = "METER.QC.DUPLICATE"
        payload["rows"].append(duplicate)
        result = SiteTwinCalibrationService().run(payload)
        self.assertFalse(result["valid"])
        codes = {item["code"] for item in result["errors"]}
        self.assertIn("window_overlap", codes)
        self.assertIn("duplicate_observation", codes)
        self.assertIsNone(result["evidence"])

    def test_authorized_export_needs_source_attestation_and_independent_approval(self):
        payload = calibration_bundle(evidence_class="authorized_site_export")
        payload["source"].update(
            owner="AUTHORIZED_PORT_OPERATOR",
            license="AUTHORIZED_INTERNAL_CALIBRATION",
            source_system="SITE_METER_AND_TOS_EXPORT",
        )
        service = SiteTwinCalibrationService()
        unattested = service.run(payload)
        self.assertTrue(unattested["valid"])
        self.assertFalse(unattested["evidence"]["measured_outcomes"])
        self.assertEqual(unattested["warnings"][0]["code"], "source_attestation_missing")

        attested = service.run(
            payload,
            source_verified=True,
            approved_by="site-model-risk-reviewer",
            change_ticket="CHG-2026-001",
        )
        self.assertTrue(attested["evidence"]["measured_outcomes"])
        self.assertTrue(attested["evidence"]["approved"])
        self.assertTrue(attested["boundary"]["site_calibrated"])
        self.assertFalse(attested["boundary"]["dispatch_allowed"])

        self_approved = service.run(
            payload,
            source_verified=True,
            approved_by="AUTHORIZED_PORT_OPERATOR",
            change_ticket="CHG-2026-002",
        )
        self.assertFalse(self_approved["evidence"]["approved"])
        self.assertIn(
            "independent_approval_invalid",
            {item["code"] for item in self_approved["warnings"]},
        )

    def test_readiness_only_accepts_a_valid_approved_artifact(self):
        payload = calibration_bundle(evidence_class="authorized_site_export")
        payload["source"].update(
            owner="AUTHORIZED_PORT_OPERATOR",
            license="AUTHORIZED_INTERNAL_CALIBRATION",
            source_system="SITE_METER_AND_TOS_EXPORT",
        )
        evidence = SiteTwinCalibrationService().run(
            payload,
            source_verified=True,
            approved_by="site-model-risk-reviewer",
            change_ticket="CHG-2026-001",
        )["evidence"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site_calibration_v2.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_TWIN_CALIBRATION_PATH": str(path)}, clear=True):
                readiness = SiteTwinCalibrationService().readiness()
        self.assertTrue(readiness["configured_artifact"]["verified"])
        self.assertTrue(readiness["boundary"]["site_calibrated"])
        self.assertEqual(len(readiness["configured_artifact"]["sha256"]), 64)
        self.assertNotIn(str(path), json.dumps(readiness))

    def test_tampering_breaks_evidence_digest_and_production_eligibility(self):
        payload = calibration_bundle(evidence_class="authorized_site_export")
        payload["source"].update(
            owner="AUTHORIZED_PORT_OPERATOR",
            license="AUTHORIZED_INTERNAL_CALIBRATION",
            source_system="SITE_METER_AND_TOS_EXPORT",
        )
        evidence = SiteTwinCalibrationService().run(
            payload,
            source_verified=True,
            approved_by="site-model-risk-reviewer",
            change_ticket="CHG-2026-001",
        )["evidence"]
        evidence["metrics"]["normalized_mae"] = 0.0
        validation = TwinSchemaService.validate_calibration(evidence)
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["production_gate_eligible"])
        self.assertIn("evidence_digest does not match calibration content", validation["errors"])

    def test_approved_artifact_populates_power_fidelity_but_not_interval_claim(self):
        payload = calibration_bundle(evidence_class="authorized_site_export")
        payload["source"].update(
            owner="AUTHORIZED_PORT_OPERATOR",
            license="AUTHORIZED_INTERNAL_CALIBRATION",
            source_system="SITE_METER_AND_TOS_EXPORT",
        )
        evidence = SiteTwinCalibrationService().run(
            payload,
            source_verified=True,
            approved_by="site-model-risk-reviewer",
            change_ticket="CHG-2026-001",
        )["evidence"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site_calibration_v2.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_TWIN_CALIBRATION_PATH": str(path)}, clear=True):
                reliability = server.twin_reliability.build(refresh=True)
        self.assertTrue(reliability["site_fidelity"]["available"])
        self.assertEqual(reliability["site_fidelity"]["status"], "approved_site_power_calibration")
        self.assertEqual(len(reliability["site_error_decomposition"]["groups"]), 2)
        self.assertFalse(reliability["forecast_interval_calibration"]["available"])
        self.assertEqual(
            reliability["forecast_interval_calibration"]["status"],
            "separate_interval_calibration_required",
        )

    def test_api_runs_contract_sample_but_never_approves_browser_input(self):
        client = TestClient(server.app)
        readiness = client.get("/api/v3/twin-calibration/readiness")
        self.assertEqual(readiness.status_code, 200)
        self.assertFalse(readiness.json()["boundary"]["site_calibrated"])

        result = client.post("/api/v3/twin-calibration/run", json=calibration_bundle())
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertTrue(body["valid"])
        self.assertEqual(body["accepted_rows"], 72)
        self.assertFalse(body["evidence"]["approved"])
        self.assertFalse(body["boundary"]["production_authority"])

    def test_cli_writes_a_new_version_and_refuses_to_overwrite_it(self):
        payload = calibration_bundle(evidence_class="authorized_site_export")
        payload["source"].update(
            owner="AUTHORIZED_PORT_OPERATOR",
            license="AUTHORIZED_INTERNAL_CALIBRATION",
            source_system="SITE_METER_AND_TOS_EXPORT",
        )
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "calibration_input.json"
            output_path = Path(tmp) / "calibration_evidence_v1.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            arguments = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--authorized-source-attested",
                "--approved-by",
                "site-model-risk-reviewer",
                "--change-ticket",
                "CHG-2026-001",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(calibration_cli_main(arguments), 0)
            evidence = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(evidence["approved"])
            with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                calibration_cli_main(arguments)


if __name__ == "__main__":
    unittest.main()
