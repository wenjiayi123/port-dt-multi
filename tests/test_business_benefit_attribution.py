from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.services.business_benefit_attribution import BusinessBenefitAttributionService, METRICS
from scripts.verify_business_benefit_attribution import main as attribution_cli_main


def _outcomes(pair_index: int, group: str, period: str) -> dict:
    energy = 1000.0 + 10.0 * pair_index
    waiting = 180.0 + 2.0 * pair_index
    utilization = 0.75 + 0.002 * pair_index
    throughput = 1000.0 + 5.0 * pair_index
    downtime = 40.0 + 0.2 * pair_index
    maintenance = 5000.0 + 20.0 * pair_index
    regulatory = 60.0 + pair_index
    if group == "incumbent" and period == "post":
        energy += 20.0
        waiting += 8.0
        utilization += 0.01
        throughput += 20.0
        downtime += 2.0
        maintenance += 100.0
        regulatory += 5.0
    elif group == "candidate" and period == "pre":
        energy += 5.0
        waiting -= 3.0
        utilization += 0.002
        throughput += 5.0
        downtime -= 1.0
        maintenance -= 50.0
        regulatory -= 2.0
    elif group == "candidate" and period == "post":
        energy += 5.0 + 20.0 - 100.0
        waiting += -3.0 + 8.0 - 25.0
        utilization += 0.002 + 0.01 + 0.025
        throughput += 5.0 + 20.0
        downtime += -1.0 + 2.0 - 8.0
        maintenance += -50.0 + 100.0 - 400.0
        regulatory += -2.0 + 5.0 - 12.0
    return {
        "energy_consumption_kwh": energy,
        "tariff_cny_per_kwh": 0.8,
        "carbon_factor_kg_per_kwh": 0.5,
        "waiting_time_minutes": waiting,
        "berth_productive_minutes": utilization * 720.0,
        "berth_available_minutes": 720.0,
        "throughput_teu": throughput,
        "unplanned_downtime_minutes": downtime,
        "maintenance_cost_cny": maintenance,
        "regulatory_delay_minutes": regulatory,
    }


def attribution_bundle(*, evidence_class: str = "contract_test_only", site_id: str = "SITE.CNSHA.CONTRACT") -> dict:
    pair_count = 30 if evidence_class == "authorized_site_benefit_export" else 12
    pre_start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    post_start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assignment_reference = "ASSIGNMENT.MATCHED.PAIRS.202607"
    units = []
    for pair_index in range(pair_count):
        factors = {
            "cargo_teu": 800.0 + 10.0 * pair_index,
            "vessel_size_teu": 10000.0 + 100.0 * pair_index,
            "weather_severity_ratio": 0.20 + 0.01 * (pair_index % 10),
            "tide_m": -1.0 + 0.1 * pair_index,
            "shift_index": float(pair_index % 3),
            "equipment_availability_ratio": 0.90 + 0.002 * pair_index,
        }
        for group in ("candidate", "incumbent"):
            for period in ("pre", "post"):
                started = (pre_start if period == "pre" else post_start) + timedelta(days=pair_index)
                ended = started + timedelta(hours=12)
                candidate_post = group == "candidate" and period == "post"
                suffix = f"{pair_index:03d}.{group.upper()}.{period.upper()}"
                units.append({
                    "pair_id": f"PAIR.BENEFIT.{pair_index:03d}",
                    "unit_id": f"UNIT.BENEFIT.{suffix}",
                    "cluster_id": f"CLUSTER.BENEFIT.{pair_index:03d}.{group.upper()}",
                    "group": group,
                    "period": period,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "assignment_reference": assignment_reference,
                    "matching_factors": dict(factors),
                    "outcomes": _outcomes(pair_index, group, period),
                    "source_references": {
                        "energy_meter_reference": f"METER.{suffix}",
                        "tos_operations_reference": f"TOS.{suffix}",
                        "emissions_factor_reference": f"EMISSIONS.{suffix}",
                        "maintenance_reference": f"CMMS.{suffix}",
                        "regulatory_reference": f"REGULATORY.{suffix}",
                        "safety_reference": f"SAFETY.{suffix}",
                    },
                    "decision": {
                        "actual_plan_reference": f"PLAN.{suffix}",
                        "execution_receipt_id": f"EXECUTION.{suffix}",
                        "executed_at": (started + timedelta(hours=1)).isoformat(),
                        "candidate_executed": candidate_post,
                        "recommendation_receipt_id": f"RECOMMENDATION.{suffix}" if candidate_post else "",
                        "human_approval_receipt_id": f"APPROVAL.{suffix}" if candidate_post else "",
                        "recommendation_digest": format(pair_index + 1, "064x") if candidate_post else "",
                    },
                    "other_intervention_ids": [],
                    "safety_incident_count": 0,
                })
    return {
        "schema_version": "business_benefit_attribution_dataset.v1",
        "site_id": site_id,
        "run_id": "BENEFIT.ATTRIBUTION.202607",
        "source": {
            "source_system": "AUTHORIZED_SITE_BENEFIT_EXPORT" if evidence_class == "authorized_site_benefit_export" else "PORT_DT_CONTRACT_TEST",
            "owner": "AUTHORIZED_PORT_OPERATOR" if evidence_class == "authorized_site_benefit_export" else "PORT_DT_CONTRACT_TEST",
            "license": "AUTHORIZED_INTERNAL_BENEFIT_REVIEW" if evidence_class == "authorized_site_benefit_export" else "contract_test_only_not_authorized",
            "timezone": "Asia/Shanghai",
            "extracted_at": "2026-08-02T00:00:00Z",
            "evidence_class": evidence_class,
        },
        "bindings": {
            "forecast_uncertainty_evidence_digest": "a" * 64,
            "shadow_acceptance_evidence_digest": "b" * 64,
            "execution_acceptance_evidence_digest": "c" * 64,
            "collaboration_evidence_digest": "d" * 64,
        },
        "design": {
            "method": "matched_difference_in_differences",
            "primary_metric": "energy_cost_cny",
            "protocol_id": "PROTOCOL.BENEFIT.2026.01",
            "protocol_sha256": "e" * 64,
            "analysis_plan_sha256": "f" * 64,
            "protocol_registered_at": "2026-05-01T00:00:00Z",
            "assignment_reference": assignment_reference,
            "assignment_at": "2026-06-30T23:00:00Z",
            "interference_scope": "INDEPENDENT.PORT.CALL.AND.ASSET.CLUSTERS",
            "pre_window": {"start_at": "2026-06-01T00:00:00Z", "end_at": "2026-06-30T23:59:59Z"},
            "post_window": {"start_at": "2026-07-01T00:00:00Z", "end_at": "2026-07-31T23:59:59Z"},
        },
        "units": units,
    }


def authorized_bundle(site_id: str = "SITE.CNSHA.CONTRACT") -> dict:
    return attribution_bundle(evidence_class="authorized_site_benefit_export", site_id=site_id)


def approved_evidence(site_id: str = "SITE.CNSHA.CONTRACT") -> dict:
    result = BusinessBenefitAttributionService().run(
        authorized_bundle(site_id),
        source_verified=True,
        business_owner_approved_by="site-business-owner-reviewer",
        operations_assurance_approved_by="site-operations-assurance-reviewer",
        causal_methods_approved_by="independent-causal-methods-reviewer",
        change_ticket="BENEFIT-CHANGE-2026-07",
    )
    if not result["valid"]:
        raise AssertionError(result["errors"])
    return result["evidence"]


class BusinessBenefitAttributionTests(unittest.TestCase):
    def test_contract_attributes_eight_metrics_without_field_claim(self):
        result = BusinessBenefitAttributionService().run(attribution_bundle())
        self.assertTrue(result["valid"], result["errors"])
        evidence = result["evidence"]
        self.assertEqual(evidence["data_quality"]["complete_pair_count"], 12)
        self.assertEqual(len(evidence["unit_receipts"]), 48)
        self.assertEqual({row["metric_id"] for row in evidence["metric_summaries"]}, set(METRICS))
        self.assertEqual(evidence["attribution_status"], "pass")
        self.assertGreater(next(row for row in evidence["metric_summaries"] if row["metric_id"] == "energy_cost_cny")["ci95_relative_percent"]["low"], 0.0)
        self.assertFalse(evidence["measured_outcomes"])
        self.assertFalse(evidence["approved"])
        self.assertFalse(evidence["boundary"]["field_kpi_claim_eligible"])
        self.assertFalse(evidence["boundary"]["dispatch_allowed"])

    def test_missing_execution_receipt_and_incomplete_pair_are_rejected(self):
        missing = attribution_bundle()
        candidate_post = next(row for row in missing["units"] if row["group"] == "candidate" and row["period"] == "post")
        candidate_post["decision"]["human_approval_receipt_id"] = ""
        result = BusinessBenefitAttributionService().run(missing)
        self.assertFalse(result["valid"])
        self.assertIn("candidate_receipt", {row["code"] for row in result["errors"]})

        incomplete = attribution_bundle()
        incomplete["units"].pop()
        result = BusinessBenefitAttributionService().run(incomplete)
        self.assertFalse(result["valid"])
        self.assertIn("incomplete_pair", {row["code"] for row in result["errors"]})

    def test_concurrent_intervention_fails_gate_without_hiding_results(self):
        payload = attribution_bundle()
        payload["units"][0]["other_intervention_ids"] = ["OTHER.CHANGE.001"]
        result = BusinessBenefitAttributionService().run(payload)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["evidence"]["attribution_status"], "fail")
        self.assertFalse(result["evidence"]["threshold_checks"]["no_concurrent_intervention"])
        self.assertFalse(result["evidence"]["approved"])

    def test_authorized_site_requires_three_independent_reviewers(self):
        evidence = approved_evidence()
        self.assertTrue(evidence["measured_outcomes"])
        self.assertTrue(evidence["approved"])
        self.assertTrue(evidence["boundary"]["realized_business_benefit_verified"])
        validation = BusinessBenefitAttributionService.validate_evidence(evidence)
        self.assertTrue(validation["production_gate_eligible"], validation["errors"])

        result = BusinessBenefitAttributionService().run(
            authorized_bundle(),
            source_verified=True,
            business_owner_approved_by="same-reviewer",
            operations_assurance_approved_by="same-reviewer",
            causal_methods_approved_by="same-reviewer",
            change_ticket="BENEFIT-CHANGE-2026-07",
        )
        self.assertTrue(result["valid"])
        self.assertFalse(result["evidence"]["approved"])

    def test_semantic_tampering_is_rejected_even_with_recomputed_evidence_digest(self):
        evidence = approved_evidence()
        evidence["threshold_checks"]["primary_effect"] = False
        evidence.pop("evidence_digest")
        canonical = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        evidence["evidence_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        validation = BusinessBenefitAttributionService.validate_evidence(evidence)
        self.assertFalse(validation["production_gate_eligible"])
        self.assertIn("threshold checks failed or were altered", validation["errors"])

    def test_readiness_accepts_only_approved_versioned_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benefit.json"
            path.write_text(json.dumps(approved_evidence()), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_BUSINESS_BENEFIT_ATTRIBUTION_PATH": str(path)}):
                readiness = BusinessBenefitAttributionService().readiness()
        self.assertTrue(readiness["configured_artifact"]["verified"])
        self.assertEqual(readiness["configured_artifact"]["complete_pair_count"], 30)
        self.assertTrue(readiness["boundary"]["field_kpi_claim_eligible"])

    def test_http_run_is_contract_only(self):
        client = TestClient(server.app)
        response = client.post("/api/v3/business-benefit-attribution/run", json=attribution_bundle())
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["evidence"]["approved"])
        self.assertFalse(payload["boundary"]["production_authority"])
        readiness = client.get("/api/v3/business-benefit-attribution/readiness")
        self.assertEqual(readiness.status_code, 200)
        self.assertEqual(readiness.json()["dataset_schema"], "business_benefit_attribution_dataset.v1")

    def test_cli_writes_new_artifact_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            output = root / "evidence.json"
            source.write_text(json.dumps(attribution_bundle()), encoding="utf-8")
            capture = io.StringIO()
            with redirect_stdout(capture):
                code = attribution_cli_main(["--input", str(source), "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                attribution_cli_main(["--input", str(source), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
