from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient

from app import server
from app.services.rl_training.datasets import (
    FACTOR_COLUMNS,
    NUMERIC_COLUMNS,
    PORT_WIDE_COLUMNS,
    REGULATORY_COLUMNS,
    load_port_dataset,
    site_replacement_readiness_report,
)


class SiteDatasetReadinessTests(unittest.TestCase):
    def test_public_scenario_is_never_mislabeled_as_site_replacement(self):
        dataset = load_port_dataset("public_cn_sha_integrated_scenario_v5")
        report = site_replacement_readiness_report(dataset)
        self.assertFalse(report["site_replacement_ready"])
        codes = {item["code"] for item in report["blockers"]}
        self.assertIn("site_authority", codes)
        self.assertIn("provenance_type", codes)
        self.assertIn("field_coverage", codes)
        self.assertIn("field_lineage_missing", codes)
        self.assertFalse(report["boundary"]["production_authority"])

    def test_structurally_complete_authorized_export_passes_dataset_gate_only(self):
        source = load_port_dataset("public_cn_sha_integrated_scenario_v5")
        required = (
            *NUMERIC_COLUMNS,
            *FACTOR_COLUMNS,
            *REGULATORY_COLUMNS,
            *PORT_WIDE_COLUMNS,
        )
        metadata = {
            **source.metadata,
            "authorized_site_data": True,
            "site_id": "SITE.CONTRACT.TEST",
            "source_manifest_sha256": "a" * 64,
            "provenance_type": "verified_site_export",
            "field_provenance": {
                field: {
                    "source_system": "CONTRACT_TEST_SOURCE",
                    "source_field": field,
                    "owner": "CONTRACT_TEST_OWNER",
                    "evidence_class": "measured",
                }
                for field in required
            },
        }
        dataset = replace(
            source,
            metadata=metadata,
            factor_availability=np.ones_like(source.factor_availability),
        )
        report = site_replacement_readiness_report(dataset)
        self.assertTrue(report["site_replacement_ready"], report["blockers"])
        self.assertTrue(report["training_dataset_replacement_ready"])
        self.assertGreaterEqual(report["coverage_hours"], 720)
        self.assertFalse(report["boundary"]["live_data_verified"])
        self.assertFalse(report["boundary"]["dispatch_allowed"])
        self.assertFalse(report["boundary"]["production_authority"])

    def test_site_readiness_api_exposes_blockers_for_public_dataset(self):
        response = TestClient(server.app).get(
            "/api/rl/datasets/public_cn_sha_integrated_scenario_v5/site-readiness"
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["training_dataset_replacement_ready"])


if __name__ == "__main__":
    unittest.main()
