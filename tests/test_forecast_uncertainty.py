from __future__ import annotations

import io
import hashlib
import json
import math
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
from app.services.forecast_uncertainty import ForecastUncertaintyService, TARGETS
from scripts.verify_forecast_uncertainty import main as forecast_cli_main


def _feature_values(target_id: str, index: int) -> dict:
    phase = index % 24
    sine = math.sin(2.0 * math.pi * phase / 24.0)
    cosine = math.cos(2.0 * math.pi * phase / 24.0)
    modular = phase % 7
    values = {
        "vessel_eta_minutes": {
            "distance_nm": 45.0 + 8.0 * sine,
            "speed_knots": 13.0 + 1.5 * cosine,
            "channel_wait_minutes": 18.0 + 3.0 * modular,
            "wind_mps": 7.0 + 0.35 * phase,
            "current_knots": -0.8 + 0.08 * (phase % 11),
        },
        "berth_duration_minutes": {
            "container_moves": 720.0 + 70.0 * sine + 5.0 * phase,
            "crane_count": 3.0 + phase % 4,
            "crane_productivity": 27.0 + 2.0 * cosine,
            "labor_availability": 0.78 + 0.02 * modular,
            "yard_congestion_ratio": 0.45 + 0.015 * phase,
        },
        "quay_crane_productivity_mph": {
            "crane_age_years": 4.0 + phase % 9,
            "wind_mps": 6.0 + 0.4 * phase,
            "labor_availability": 0.80 + 0.015 * modular,
            "yard_congestion_ratio": 0.40 + 0.018 * phase,
            "equipment_availability": 0.90 + 0.008 * (phase % 8),
        },
        "yard_congestion_ratio": {
            "yard_occupancy_ratio": 0.48 + 0.012 * phase,
            "gate_queue_trucks": 12.0 + 2.0 * (phase % 10),
            "planned_discharge_units": 350.0 + 18.0 * phase,
            "rail_backlog_units": 20.0 + 5.0 * modular,
            "horizontal_transport_availability": 0.76 + 0.02 * cosine,
        },
        "equipment_failure_probability": {
            "runtime_hours": 1200.0 + 25.0 * phase,
            "vibration_mm_s": 2.1 + 0.1 * (phase % 6),
            "temperature_c": 42.0 + 0.5 * phase,
            "fault_count_24h": 1.0 if phase % 4 == 0 else 0.0,
            "maintenance_overdue_hours": 6.0 * (phase % 5),
        },
        "weather_stoppage_probability": {
            "wind_mps": 23.0 + 0.2 * phase if phase % 6 == 0 else 7.0 + 0.25 * phase,
            "wave_height_m": 3.2 if phase % 6 == 0 else 0.8 + 0.03 * phase,
            "visibility_m": 650.0 if phase % 6 == 0 else 5000.0 - 40.0 * phase,
            "precipitation_mm_h": 12.0 if phase % 6 == 0 else 0.2 * (phase % 5),
            "lightning_distance_km": 4.0 if phase % 6 == 0 else 28.0 - 0.2 * phase,
        },
        "regulatory_delay_minutes": {
            "inspection_load": 4.0 + phase % 8,
            "document_completeness_ratio": 0.82 + 0.007 * phase,
            "dangerous_goods_indicator": 1.0 if phase % 5 == 0 else 0.0,
            "authority_queue": 2.0 + phase % 6,
            "exception_count": float(phase % 4),
        },
        "energy_load_kw": {
            "vessel_calls": 2.0 + phase % 5,
            "crane_moves": 250.0 + 15.0 * phase,
            "reefer_count": 180.0 + 7.0 * (phase % 9),
            "shore_power_kw": 1800.0 if phase % 4 < 2 else 0.0,
            "yard_occupancy_ratio": 0.50 + 0.01 * phase,
            "ambient_c": 25.0 + 3.0 * sine,
        },
    }
    return values[target_id]


def _observed(target_id: str, features: dict, index: int) -> float:
    noise = ((index % 5) - 2) * 0.15
    if target_id == "vessel_eta_minutes":
        return 55 + 2.2 * features["distance_nm"] - 2.5 * features["speed_knots"] + 0.45 * features["channel_wait_minutes"] + 0.8 * features["wind_mps"] - 1.5 * features["current_knots"] + noise
    if target_id == "berth_duration_minutes":
        return 210 + 0.55 * features["container_moves"] - 24 * features["crane_count"] - 4 * features["crane_productivity"] - 65 * features["labor_availability"] + 75 * features["yard_congestion_ratio"] + noise
    if target_id == "quay_crane_productivity_mph":
        return 36 - 0.25 * features["crane_age_years"] - 0.18 * features["wind_mps"] + 8 * features["labor_availability"] - 5 * features["yard_congestion_ratio"] + 4 * features["equipment_availability"] + noise * 0.2
    if target_id == "yard_congestion_ratio":
        return 0.08 + 0.65 * features["yard_occupancy_ratio"] + 0.001 * features["gate_queue_trucks"] + 0.00008 * features["planned_discharge_units"] + 0.0002 * features["rail_backlog_units"] - 0.08 * features["horizontal_transport_availability"] + noise * 0.002
    if target_id == "equipment_failure_probability":
        return features["fault_count_24h"]
    if target_id == "weather_stoppage_probability":
        return 1.0 if features["wind_mps"] >= 20.0 else 0.0
    if target_id == "regulatory_delay_minutes":
        return 8 + 7 * features["inspection_load"] - 25 * features["document_completeness_ratio"] + 18 * features["dangerous_goods_indicator"] + 5 * features["authority_queue"] + 9 * features["exception_count"] + noise
    if target_id == "energy_load_kw":
        return 850 + 95 * features["vessel_calls"] + 3.2 * features["crane_moves"] + 1.8 * features["reefer_count"] + 0.96 * features["shore_power_kw"] + 520 * features["yard_occupancy_ratio"] + 11 * features["ambient_c"] + noise * 5
    raise AssertionError(target_id)


def forecast_bundle(*, evidence_class: str = "contract_test_only", site_id: str = "SITE.CNSHA.CONTRACT") -> dict:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    row_count = 120 if evidence_class == "authorized_site_forecast_export" else 72
    training_end = 71 if row_count == 120 else 23
    calibration_start = training_end + 1
    calibration_end = 95 if row_count == 120 else 47
    test_start = calibration_end + 1
    test_end = row_count - 1
    horizons = {
        "vessel_eta_minutes": 120,
        "berth_duration_minutes": 240,
        "quay_crane_productivity_mph": 60,
        "yard_congestion_ratio": 180,
        "equipment_failure_probability": 360,
        "weather_stoppage_probability": 180,
        "regulatory_delay_minutes": 240,
        "energy_load_kw": 60,
    }
    series = []
    for target_id, definition in TARGETS.items():
        horizon = horizons[target_id]
        rows = []
        for index in range(row_count):
            target_at = base + timedelta(hours=index)
            issued_at = target_at - timedelta(minutes=horizon)
            features = _feature_values(target_id, index)
            rows.append({
                "forecast_id": f"FORECAST.{target_id.upper()}.{index:04d}",
                "feature_snapshot_at": (issued_at - timedelta(minutes=5)).isoformat(),
                "issued_at": issued_at.isoformat(),
                "target_at": target_at.isoformat(),
                "observed_at": (target_at + timedelta(minutes=15)).isoformat(),
                "features": features,
                "observed_value": _observed(target_id, features, index),
                "source_reference": f"SITE.OUTCOME.{target_id.upper()}.{index:04d}",
            })
        series.append({
            "target_id": target_id,
            "forecast_horizon_minutes": horizon,
            "ridge_alpha": 0.01,
            "features": list(definition["features"]),
            "rows": rows,
        })
    return {
        "schema_version": "forecast_uncertainty_dataset.v1",
        "site_id": site_id,
        "run_id": "FORECAST.CALIBRATION.202606",
        "source": {
            "source_system": "AUTHORIZED_SITE_FORECAST_EXPORT" if evidence_class == "authorized_site_forecast_export" else "PORT_DT_CONTRACT_TEST",
            "owner": "AUTHORIZED_PORT_OPERATOR" if evidence_class == "authorized_site_forecast_export" else "PORT_DT_CONTRACT_TEST",
            "license": "AUTHORIZED_INTERNAL_FORECAST_EVALUATION" if evidence_class == "authorized_site_forecast_export" else "contract_test_only_not_authorized",
            "timezone": "Asia/Shanghai",
            "extracted_at": (base + timedelta(hours=row_count, minutes=30)).isoformat(),
            "evidence_class": evidence_class,
        },
        "split": {
            "training_window": {"start_at": base.isoformat(), "end_at": (base + timedelta(hours=training_end)).isoformat()},
            "calibration_window": {"start_at": (base + timedelta(hours=calibration_start)).isoformat(), "end_at": (base + timedelta(hours=calibration_end)).isoformat()},
            "test_window": {"start_at": (base + timedelta(hours=test_start)).isoformat(), "end_at": (base + timedelta(hours=test_end)).isoformat()},
        },
        "series": series,
    }


def authorized_bundle(site_id: str = "SITE.CNSHA.CONTRACT") -> dict:
    return forecast_bundle(evidence_class="authorized_site_forecast_export", site_id=site_id)


def approved_evidence(site_id: str = "SITE.CNSHA.CONTRACT") -> dict:
    result = ForecastUncertaintyService().run(
        authorized_bundle(site_id),
        source_verified=True,
        operations_planning_approved_by="site-operations-planning-reviewer",
        model_risk_approved_by="site-model-risk-reviewer",
        maritime_safety_approved_by="site-maritime-safety-reviewer",
        change_ticket="FORECAST-CHANGE-2026-06",
    )
    if not result["valid"]:
        raise AssertionError(result["errors"])
    return result["evidence"]


class ForecastUncertaintyTests(unittest.TestCase):
    def test_contract_evaluates_eight_targets_without_site_claim(self):
        result = ForecastUncertaintyService().run(forecast_bundle())
        self.assertTrue(result["valid"], result["errors"])
        evidence = result["evidence"]
        self.assertEqual(len(evidence["metrics_by_target"]), 8)
        self.assertEqual(evidence["calibration_status"], "pass")
        self.assertEqual(len(evidence["test_receipts"]), 8 * 24)
        self.assertTrue(all(row["metrics"]["coverage_95"] >= 0.85 for row in evidence["metrics_by_target"]))
        self.assertFalse(evidence["measured_outcomes"])
        self.assertFalse(evidence["approved"])
        self.assertTrue(evidence["boundary"]["forecast_advisory_only"])
        self.assertFalse(evidence["boundary"]["dispatch_allowed"])

    def test_timing_leakage_and_missing_target_are_rejected(self):
        leakage = forecast_bundle()
        leakage["series"][0]["rows"][0]["feature_snapshot_at"] = leakage["series"][0]["rows"][0]["target_at"]
        result = ForecastUncertaintyService().run(leakage)
        self.assertFalse(result["valid"])
        self.assertIn("forecast_timing", {row["code"] for row in result["errors"]})

        incomplete = forecast_bundle()
        incomplete["series"].pop()
        result = ForecastUncertaintyService().run(incomplete)
        self.assertFalse(result["valid"])
        self.assertIn("target_coverage", {row["code"] for row in result["errors"]})

    def test_binary_risk_requires_realized_outcome(self):
        payload = forecast_bundle()
        failure = next(row for row in payload["series"] if row["target_id"] == "equipment_failure_probability")
        failure["rows"][0]["observed_value"] = 0.4
        result = ForecastUncertaintyService().run(payload)
        self.assertFalse(result["valid"])
        self.assertIn("binary_outcome", {row["code"] for row in result["errors"]})

    def test_authorized_source_requires_three_independent_reviewers(self):
        evidence = approved_evidence()
        self.assertTrue(evidence["measured_outcomes"])
        self.assertTrue(evidence["live_service_level_verified"])
        self.assertTrue(evidence["approved"])
        self.assertTrue(evidence["boundary"]["site_forecast_service_accepted"])
        validation = ForecastUncertaintyService.validate_evidence(evidence)
        self.assertTrue(validation["production_gate_eligible"], validation["errors"])

        result = ForecastUncertaintyService().run(
            authorized_bundle(),
            source_verified=True,
            operations_planning_approved_by="same-reviewer",
            model_risk_approved_by="same-reviewer",
            maritime_safety_approved_by="same-reviewer",
            change_ticket="FORECAST-CHANGE-2026-06",
        )
        self.assertTrue(result["valid"])
        self.assertFalse(result["evidence"]["approved"])

    def test_tampering_invalidates_configured_artifact(self):
        evidence = approved_evidence()
        evidence["metrics_by_target"][0]["metrics"]["coverage_95"] = 0.0
        evidence.pop("evidence_digest")
        canonical = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        evidence["evidence_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecast.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_FORECAST_UNCERTAINTY_PATH": str(path)}):
                readiness = ForecastUncertaintyService().readiness()
        self.assertFalse(readiness["configured_artifact"]["verified"])
        self.assertFalse(readiness["boundary"]["site_forecast_service_accepted"])

    def test_readiness_accepts_versioned_approved_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecast.json"
            path.write_text(json.dumps(approved_evidence()), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_FORECAST_UNCERTAINTY_PATH": str(path)}):
                readiness = ForecastUncertaintyService().readiness()
        self.assertTrue(readiness["configured_artifact"]["verified"])
        self.assertEqual(readiness["configured_artifact"]["target_count"], 8)
        self.assertTrue(readiness["boundary"]["site_forecast_service_accepted"])

    def test_http_run_is_contract_only(self):
        client = TestClient(server.app)
        response = client.post("/api/v3/forecast-uncertainty/run", json=forecast_bundle())
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["evidence"]["approved"])
        self.assertFalse(payload["boundary"]["production_authority"])
        readiness = client.get("/api/v3/forecast-uncertainty/readiness")
        self.assertEqual(readiness.status_code, 200)
        self.assertEqual(readiness.json()["dataset_schema"], "forecast_uncertainty_dataset.v1")

    def test_cli_writes_new_artifact_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            output = root / "evidence.json"
            source.write_text(json.dumps(forecast_bundle()), encoding="utf-8")
            capture = io.StringIO()
            with redirect_stdout(capture):
                code = forecast_cli_main(["--input", str(source), "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                forecast_cli_main(["--input", str(source), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
