from __future__ import annotations

import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.adapters.telemetry_dataset import DatasetTelemetry
from app.adapters.actuators import Command, PortSouthboundGateway
from app.adapters.schedule_sources import ScheduleSources
from app.services.curves.bess_capability import CurvesBessCapability
from app.services.curves.economic_benefit import CurvesEconomicBenefit
from app.services.curves.peak_risk import CurvesPeakRisk
from app.services.forecast import ForecastService
from app.services.forecast_twin.sim_aggregate import aggregate_sim
from app.services.forecast_twin.twin import TwinService
from app.services.exec_cockpit.service import get_summary as get_exec_summary
from app.services.platform_map.service import get_graph as get_platform_graph
from app.services.energy_reporting.energy import EnergyService
from app.services.energy_reporting.reporting import ReportingService


class _Telemetry:
    def __init__(self, points):
        self.points = points

    def get_recent_power(self, _asset_id):
        return self.points


class DataDrivenModuleTests(unittest.TestCase):
    def test_default_telemetry_is_labeled_dataset_replay(self):
        telemetry = DatasetTelemetry()
        status = telemetry.source_status()
        self.assertEqual(status["mode"], "canonical_dataset_replay")
        self.assertFalse(status["measured"])
        self.assertGreaterEqual(status["rows"], 18)
        self.assertEqual(status["artifact_id"], "public_port_ops_v1.csv")
        self.assertNotIn("dataset_path", status)

    def test_forecast_fails_empty_and_is_deterministic_with_history(self):
        empty = ForecastService(_Telemetry([])).forecast_load(["asset"])
        self.assertEqual(empty["asset"], [])

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            {"ts": (start + timedelta(minutes=i)).isoformat(), "kW": 100 + 8 * math.sin(i / 4)}
            for i in range(72)
        ]
        service = ForecastService(_Telemetry(points))
        first = service.forecast_load(["asset"], horizon_min=12, step_min=1)
        second = service.forecast_load(["asset"], horizon_min=12, step_min=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first["asset"]), 12)
        self.assertTrue(all(row["model"] == "ridge_autoregression" for row in first["asset"]))

    def test_recursive_forecast_is_bounded_by_registered_asset_rating(self):
        class RatedTelemetry(_Telemetry):
            def list_assets(self):
                return [{"id": "asset", "rated_kw": 4200.0}]

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        points = [
            {
                "ts": (start + timedelta(minutes=i)).isoformat(),
                "kW": 1500.0 + (i ** 2) * 0.10,
            }
            for i in range(120)
        ]
        rows = ForecastService(RatedTelemetry(points)).forecast_load(
            ["asset"], horizon_min=360, step_min=1
        )["asset"]
        self.assertTrue(rows)
        self.assertTrue(all(0.0 <= row["p50"] <= row["stability_cap_kw"] for row in rows))
        self.assertTrue(all(row["stability_guard"] == "registered_asset_rating" for row in rows))
        self.assertTrue(any(row["guard_applied"] for row in rows))

    def test_twin_does_not_create_uncertainty_bands(self):
        class Forecast:
            def forecast_load(self, asset_ids, **_kwargs):
                return {asset_ids[0]: [{"ts": "2026-01-01T00:00:00Z", "kW": 50.0, "p50": 50.0}]}

        result = TwinService(fcst=Forecast()).run("asset", scenario="baseline")
        self.assertTrue(result["available"])
        self.assertFalse(result["uncertainty_available"])
        self.assertNotIn("p10", result["plan"][0])
        self.assertNotIn("p90", result["plan"][0])

    def test_aggregate_sim_does_not_invent_assets(self):
        di = SimpleNamespace(telemetry=SimpleNamespace(list_assets=lambda: []), twin=SimpleNamespace())
        result = aggregate_sim(di)
        self.assertFalse(result["available"])
        self.assertEqual(result["assets"], [])

    def test_peak_risk_uses_exact_requested_threshold(self):
        service = CurvesPeakRisk(SimpleNamespace())
        points = [{"ts": f"t{i}", "kW": value} for i, value in enumerate([90, 100, 110, 120])]
        service.curves.aggregate = lambda **_kwargs: {
            "_source": "test_quantile_model",
            "series": {
                "p50": points,
                "p10": [{**point, "kW": point["kW"] - 10} for point in points],
                "p90": [{**point, "kW": point["kW"] + 10} for point in points],
            },
        }
        result = service.peak_risk(cap_kw=105, avg_window_min=1)
        self.assertEqual(result["cap_kw"], 105.0)
        self.assertEqual(result["cap_kw_input"], 105.0)

    def test_bess_capability_is_unavailable_without_adapter(self):
        result = CurvesBessCapability(SimpleNamespace()).capability()
        self.assertFalse(result["available"])
        self.assertEqual(result["series"]["charge_cap_kw"], [])

    def test_economic_benefit_requires_distinct_policy_output(self):
        service = CurvesEconomicBenefit(SimpleNamespace())
        same_curve = [{"ts": "2026-01-01T00:00:00Z", "kW": 100.0}]
        service.curves.aggregate = lambda **_kwargs: {
            "available": True,
            "series": {"p50": same_curve},
            "_step_min": 60,
        }
        result = service.benefit()
        self.assertFalse(result["available"])
        self.assertEqual(result["series"]["total"], [])

    def test_exec_cockpit_replaces_unverified_snapshot_with_v3_public_evidence(self):
        result = get_exec_summary(SimpleNamespace())
        self.assertTrue(result["available"])
        self.assertEqual(result["evidence_mode"], "public_data_offline_verified")
        self.assertGreater(result["yearly_saving_cny"], 0)
        self.assertIsNone(result["peak_risk_30d"])
        self.assertTrue(result["evidence"]["site_kpis_pending"])

    def test_platform_map_separates_repository_map_from_live_topology(self):
        result = get_platform_graph(SimpleNamespace())
        self.assertTrue(result["available"])
        self.assertEqual(
            result["data_class"],
            "repository_architecture_configuration_not_runtime_topology",
        )
        self.assertFalse(result["deployment"]["runtime_topology_connected"])
        self.assertIsNone(result["deployment"]["production_instances"])
        self.assertGreaterEqual(len(result["deployment"]["required_adapters"]), 4)

    def test_today_energy_rejects_stale_dataset_replay(self):
        telemetry = DatasetTelemetry()
        result = EnergyService(telemetry, ReportingService(telemetry), ForecastService(telemetry)).build_today_summary()
        self.assertFalse(result["available"])
        self.assertEqual(result["_source"], "telemetry_stale")

    def test_unconfigured_actuator_gateway_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_config = f"{temp_dir}/actuators.json"
            result = PortSouthboundGateway(missing_config).dispatch(
                Command(asset_id="BESS-SITE-01", asset_type="bess", action="set", parameters={"power_kw": 100})
            )
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.message, "actuator_gateway_disabled")
        self.assertNotIn("simulated", result.details)

    def test_schedule_sources_do_not_generate_data_by_default(self):
        with patch.dict(
            os.environ,
            {"PORTDT_REAL": "0", "PORT_DT_ENABLE_ENGINEERING_SIMULATORS": ""},
            clear=False,
        ):
            source = ScheduleSources()
            weather = source.weather(
                "2026-01-01T00:00:00Z",
                "2026-01-01T02:00:00Z",
                1.3,
                103.8,
            )
        self.assertEqual(source.source_status()["mode"], "unavailable")
        self.assertEqual(weather, [])

    def test_live_schedule_status_does_not_expose_private_endpoint(self):
        with patch.dict(
            os.environ,
            {"PORTDT_REAL": "1", "PORTDT_BASE_URL": "https://private-gateway.example/internal"},
            clear=False,
        ):
            status = ScheduleSources().source_status()
        self.assertEqual(status["mode"], "live_rest")
        self.assertTrue(status["configured"])
        self.assertNotIn("base_url", status)


if __name__ == "__main__":
    unittest.main()
