from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.services.public_port_demo import service
from app.services.public_port_demo.workspace import workspace

AT = "2025-12-15T12:00:00+08:00"


class DemoWorkspaceTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"PORT_DT_PRESENTATION_SNAPSHOT": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_day_shift_and_snapshot_totals_conserve_energy_flow_and_carbon(self):
        d = workspace(AT)
        w, s = d["window"], d["snapshot"]
        self.assertTrue(w["available"])
        self.assertEqual([r["hours"] for r in w["shifts"]], [8, 8, 8])
        for report_key, snapshot_key in (("energy_kwh", "daily_energy_kwh"),
                                         ("throughput_teu", "daily_throughput_teu"),
                                         ("electricity_carbon_kg", "daily_carbon_kg")):
            self.assertAlmostEqual(sum(r[report_key] for r in w["shifts"]), w[report_key], places=6)
            self.assertAlmostEqual(w[report_key], s["operating"][snapshot_key], places=2)
        self.assertAlmostEqual(w["carbon_kg_per_teu"], w["electricity_carbon_kg"] / w["throughput_teu"])
        self.assertTrue(d["groups_complete"])

    def test_monthly_year_is_complete_and_normalizes_only_terminal_scope(self):
        a = workspace(AT)["annual"]
        self.assertEqual(len(a["months"]), 12)
        self.assertEqual(a["totals"]["hours"], 8760)
        self.assertEqual(a["months"][1]["hours"], 28 * 24)
        self.assertAlmostEqual(a["totals"]["throughput_teu"], 8_000_000, places=5)
        for key in ("energy_kwh", "throughput_teu", "electricity_carbon_kg"):
            self.assertAlmostEqual(a["totals"][key], sum(r[key] for r in a["months"]))
        self.assertIsNone(a["scope1_kg"])
        self.assertFalse(a["audit_ready"])

    def test_fixed_time_metrics_and_plans_are_reproducible_and_explainable(self):
        a, b = workspace(AT), workspace(AT)
        self.assertEqual(a["metrics"], b["metrics"])
        self.assertEqual(a["coverage"], {"available": 24, "total": 24,
                                       "pending_site_categories": 10, "label": "展示字段齐备度，非生产接入率"})
        metrics = {m["id"]: m for m in a["metrics"]}
        self.assertEqual(metrics["qc_moves_per_hour"]["value"], 30)
        self.assertTrue(all(m["replacement"] and m["detail"] and m["source_ref"] for m in a["metrics"]))
        self.assertIn("safety", [p["id"] for p in a["site_pending"]])
        self.assertIn("approval", [p["id"] for p in a["site_pending"]])
        self.assertFalse(a["production_authority"])
        self.assertFalse(a["dispatch_allowed"])

    def external(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_PRESENTATION_SNAPSHOT": str(path)}):
                return workspace(AT)

    def test_external_snapshot_never_inherits_annual_or_qc_parameters(self):
        s = service.snapshot(AT)
        s["series"] = s["series"][:2]
        for row in s["series"]:
            row.update(power_kw=100, throughput_teu_per_hour=20, carbon_kg_per_kwh=0.4, price_cny_per_kwh=0.7)
        d = self.external(s)
        self.assertEqual(d["window"]["energy_kwh"], 200)
        self.assertEqual(d["window"]["electricity_carbon_kg"], 80)
        self.assertEqual(d["window"]["hours"], 2)
        self.assertFalse(d["annual"]["available"])
        self.assertIsNone(next(m["value"] for m in d["metrics"] if m["id"] == "qc_moves_per_hour"))

    def test_missing_factors_do_not_become_zero_carbon_or_money(self):
        s = service.snapshot(AT)
        s["series"][0]["carbon_kg_per_kwh"] = None
        s["series"][0].pop("price_cny_per_kwh")
        d = self.external(s)
        self.assertTrue(d["window"]["available"])
        self.assertIsNone(d["window"]["electricity_carbon_kg"])
        self.assertIsNone(d["window"]["carbon_kg_per_teu"])
        self.assertIsNone(d["window"]["energy_cost_cny"])
        self.assertTrue(all(r["electricity_carbon_kg"] is None for r in d["window"]["shifts"]))
        self.assertLess(d["coverage"]["available"], d["coverage"]["total"])

    def test_incomplete_duplicate_or_reordered_samples_do_not_claim_complete_window(self):
        template = service.snapshot(AT)
        variants = []
        s = copy.deepcopy(template);s["series"].pop(2);variants.append(s)
        s = copy.deepcopy(template);s["series"][2] = s["series"][1];variants.append(s)
        s = copy.deepcopy(template);s["series"].reverse();variants.append(s)
        for s in variants:
            with self.subTest(timestamps=[r["timestamp"] for r in s["series"][:3]]):
                d = self.external(s)
                self.assertFalse(d["window"]["available"])
                self.assertIsNone(next(m["value"] for m in d["metrics"] if m["id"] == "energy_kwh"))

    def test_external_group_mismatch_is_visible_and_invalid_yard_ratio_rejected(self):
        s = service.snapshot(AT)
        s["asset_groups"].pop()
        self.assertFalse(self.external(s)["groups_complete"])
        s["operating"]["yard_occupancy_ratio"] = 1.2
        with self.assertRaisesRegex(ValueError, "yard_occupancy_ratio"):
            self.external(s)

    def test_broken_adapter_has_no_business_demo_fallback(self):
        with patch.dict(os.environ, {"PORT_DT_PRESENTATION_SNAPSHOT": "/missing/snapshot.json"}):
            with self.assertRaises(OSError):
                workspace(AT)


if __name__ == "__main__":
    unittest.main()
