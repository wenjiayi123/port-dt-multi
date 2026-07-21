from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.adapters.ais_tide_client import AISTideClient
from app.adapters.market_client import MarketClient
from app.adapters.tos_client import TOSClient
from app.services.esg.service import get_compliance_timeseries, get_summary
from app.services.ai_trust.service import get_badge
from app.services.multiport.service import MultiportService
from app.services.opsx.api import rollout_status
from app.services.rl_ops_center.service import RLOpsService
from app.services.twinlab.service import TwinLabService
from app.services.exec_closedloop.closed_loop import ClosedLoopService


class AdapterProvenanceTests(unittest.TestCase):
    def test_unconfigured_adapters_are_explicit_simulators(self):
        tos = TOSClient().source_status()
        market = MarketClient().source_status()
        ais = AISTideClient().source_status()
        self.assertEqual(tos["mode"], "engineering_simulator")
        self.assertEqual(market["mode"], "engineering_simulator")
        self.assertEqual(ais["ais_mode"], "engineering_simulator")
        self.assertEqual(ais["tide_mode"], "engineering_simulator")
        self.assertFalse(tos["fallback_on_live_error"])
        self.assertFalse(market["fallback_on_live_error"])
        self.assertFalse(ais["fallback_on_live_error"])

    @patch.dict("os.environ", {}, clear=False)
    def test_demo_esg_and_unverified_compliance_are_blocked(self):
        summary = get_summary(None)
        compliance = get_compliance_timeseries("CNSHA", 2024)
        self.assertFalse(summary["available"])
        self.assertEqual(summary["_source"], "esg.unavailable")
        self.assertFalse(compliance["available"])
        self.assertEqual(compliance["items"], [])

    @patch.dict("os.environ", {}, clear=False)
    def test_opsx_simulator_requires_explicit_opt_in(self):
        with self.assertRaises(HTTPException) as ctx:
            rollout_status()
        self.assertEqual(ctx.exception.status_code, 503)

    def test_twinlab_does_not_load_unverified_sample_files(self):
        service = TwinLabService()
        self.assertFalse(service.scenarios()["available"])
        self.assertFalse(service.drills()["available"])
        self.assertFalse(service.contracts()["available"])

    def test_sample_trust_and_multiport_snapshots_are_blocked(self):
        self.assertFalse(get_badge()["available"])
        self.assertFalse(MultiportService().get_summary()["available"])

    def test_rlops_uses_persisted_evaluations_and_labels_non_ope(self):
        service = RLOpsService()
        overview = service.overview()
        self.assertEqual(overview["kind"], "heldout_policy_evaluation_not_ope")
        self.assertFalse(service.ope_eval({})["available"])

    def test_closed_loop_never_synthesizes_measured_ab_results(self):
        class Panel:
            def simulate(self, **_kwargs):
                return {
                    "baseline": {"agg_kW": [10.0, 12.0], "total_kWh": 0.3667},
                    "simulated": {"agg_kW": [9.0, 10.0], "total_kWh": 0.3167},
                    "summary": {"delta_kWh": -0.05},
                }

        class Dispatch:
            def validate_strategy(self, _strategy):
                return {"ok": True}

            def dispatch(self, **_kwargs):
                return {"status": "DRY_RUN_RECORDED"}

        service = ClosedLoopService(Panel(), Dispatch(), telemetry=object(), persist_path="/tmp/unused-port-dt-model.json")
        created = service.submit({"id": "sac"}, dry_run=True)
        job_id = created["job"]["job_id"]
        comparison = service.ab_compare(job_id)
        self.assertFalse(comparison["available"])
        self.assertIsNone(comparison["actual"])
        self.assertFalse(service.learn(job_id)["ok"])


if __name__ == "__main__":
    unittest.main()
