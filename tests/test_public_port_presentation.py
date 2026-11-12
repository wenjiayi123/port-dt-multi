from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.services.public_port_demo import service
from app.services.portviz.public_presentation_source import PublicPresentationSource


AT = "2025-12-15T12:00:00+08:00"


class PublicPresentationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"PORT_DT_PRESENTATION_SNAPSHOT": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_public_facts_are_cited_and_record_is_not_an_average(self):
        p = service.profile()
        sources = {s["id"] for s in p["sources"]}
        self.assertTrue(all(f["source_id"] in sources for f in p["facts"].values()))
        self.assertEqual(p["facts"]["qc_count"]["value"], 29)
        self.assertEqual(p["facts"]["rmg_count"]["value"], 121)
        self.assertEqual(p["facts"]["agv_count"]["value"], 155)
        self.assertEqual(p["facts"]["qc_record_moves_per_hour_lower_bound"]["class"], "record_not_average")
        rows, scale, _, _ = service.inputs()
        self.assertAlmostEqual(sum(float(r["throughput_teu"]) * scale for r in rows), 8_000_000, places=5)

    def test_fixed_time_reproduces_calls_weather_and_mass_conserving_power(self):
        a, b = service.snapshot(AT), service.snapshot(AT)
        for key in ("operating", "calls", "series", "asset_groups", "environment", "provenance"):
            self.assertEqual(a[key], b[key])
        self.assertAlmostEqual(sum(g["kW"] for g in a["asset_groups"]), a["operating"]["power_kw"], places=5)
        for g in a["asset_groups"]:
            recent = service.asset_series(g["id"], AT)
            self.assertEqual(len(recent), 24)
            self.assertAlmostEqual(recent[-1]["kW"], g["kW"], places=6)
            self.assertFalse(recent[-1]["measured"])

    def test_schedule_never_overlaps_berths_or_exceeds_resource_limits(self):
        for month in (1, 4, 7, 12):
            for hour in range(24):
                s = service.snapshot(f"2025-{month:02d}-15T{hour:02d}:00:00+08:00")
                o = s["operating"]
                for resource in ("berths", "qc", "rmg", "agv"):
                    self.assertLessEqual(o[f"{resource}_busy"], o[f"{resource}_total"])
                for berth in range(7):
                    calls = [c for c in s["calls"] if c["berth_index"] == berth]
                    for previous, current in zip(calls, calls[1:]):
                        self.assertLessEqual(service._timestamp(previous["planned_completion"]), service._timestamp(current["planned_berth_time"]))
                for call in s["calls"]:
                    self.assertTrue(call["call_id"].startswith("DEMO-"))
                    self.assertIsNone(call["imo"])
                    self.assertIsNone(call["mmsi"])
                    self.assertEqual(call["identity_kind"], "synthetic_scenario_id")

    def test_broken_external_adapter_fails_without_demo_fallback(self):
        with patch.dict(os.environ, {"PORT_DT_PRESENTATION_SNAPSHOT": "/missing/presentation.json"}):
            with self.assertRaises(FileNotFoundError):
                service.snapshot()
        template = service.snapshot(AT)
        for mutate in (
            lambda s: s.update(source_timestamp="2025-12-15T12:00:00"),
            lambda s: s["operating"].update(power_kw=float("nan")),
            lambda s: s["operating"].update(agv_total=100000),
            lambda s: s["operating"].update(qc_busy=30),
            lambda s: s["provenance"].update(source_ref=""),
            lambda s: s["calls"][0].update(progress=1.5),
        ):
            invalid = copy.deepcopy(template)
            mutate(invalid)
            with self.assertRaises(ValueError):
                service._validate_snapshot(invalid)

    def test_display_replacement_cannot_grant_control_or_silently_make_submeters(self):
        s = service.snapshot(AT)
        s.update(production_authority=True, live_data_verified=True)
        s.pop("asset_groups")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text(json.dumps(s), encoding="utf-8")
            with patch.dict(os.environ, {"PORT_DT_PRESENTATION_SNAPSHOT": str(path)}):
                result = service.snapshot()
                self.assertFalse(result["production_authority"])
                self.assertFalse(result["live_data_verified"])
                self.assertEqual(result["mode"], "external_presentation_snapshot")
                with self.assertRaisesRegex(ValueError, "asset_groups"):
                    service.assets()
                with self.assertRaisesRegex(ValueError, "asset_series"):
                    service.asset_series("qc-01")

    def test_spatial_projection_has_matching_resources_and_no_claimed_real_tracks(self):
        source = PublicPresentationSource()
        with patch("app.services.portviz.public_presentation_source.snapshot", return_value=service.snapshot(AT)):
            frame = source.next_frame()
        self.assertEqual(len(source.get_bootstrap()["qcs"]), len(frame["qc"]))
        self.assertEqual(len(source.get_bootstrap()["ycs"]), len(frame["yc"]))
        self.assertEqual(len(frame["agv"]), 155)
        self.assertFalse(frame["meta"]["measured_entity_tracks"])
        self.assertFalse(frame["meta"]["production_authority"])
        self.assertFalse(frame["tr"])
        incompatible = service.snapshot(AT)
        incompatible["operating"]["qc_total"] = 10
        with patch("app.services.portviz.public_presentation_source.snapshot", return_value=incompatible):
            with self.assertRaisesRegex(ValueError, "site layout"):
                source.next_frame()


if __name__ == "__main__":
    unittest.main()
