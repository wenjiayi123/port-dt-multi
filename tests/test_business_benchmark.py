from __future__ import annotations

import unittest

from app.services.business_benchmark import build_report, load_verified_report


class BusinessKpiBenchmarkTests(unittest.TestCase):
    def test_fixed_benchmark_preserves_claims_and_boundaries(self) -> None:
        report = build_report()
        self.assertEqual(
            report["dataset"]["split_sizes"],
            {"train": 35064, "validation": 8784, "test": 8760},
        )
        self.assertEqual(
            report["resume_claims_rounded_percent"],
            {
                "berth_utilization_relative_improvement_percent": 9.0,
                "average_waiting_time_reduction_percent": 17.0,
                "scenario_energy_cost_reduction_percent": 12.0,
            },
        )
        self.assertFalse(report["evidence_boundary"]["measured_port_kpi"])
        self.assertFalse(report["release_gate"]["production_claim_allowed"])
        self.assertFalse(report["test"]["energy_balance"]["throughput_changed"])
        self.assertTrue(
            report["test"]["energy_balance"]["terminal_settlement_included"]
        )
        self.assertEqual(
            report["test"]["energy_balance"]["terminal_settlement_frequency"],
            "daily",
        )
        self.assertFalse(
            report["evidence_boundary"][
                "geographically_coherent_single_port_series"
            ]
        )
        self.assertTrue(
            report["evidence_boundary"]["official_input_geography_coherent"]
        )
        self.assertFalse(
            report["evidence_boundary"][
                "policy_parameters_calibrated_from_measured_outcomes"
            ]
        )
        self.assertFalse(
            report["evidence_boundary"]["berth_wait_causal_identification"]
        )
        self.assertIn("4 h versus 2 h", report["attribution"]["berth_utilization_and_waiting"])
        self.assertIn("not evidence of a trained RL policy", report["attribution"]["energy_cost"])
        for summary in report["test"]["uncertainty"].values():
            self.assertEqual(summary["n"], 365)
            self.assertGreater(summary["ci_low"], 0.0)

    def test_checked_in_report_hashes_are_current(self) -> None:
        report = load_verified_report()
        self.assertTrue(report["release_gate"]["passed"])
        self.assertEqual(report["dataset"]["rows"], 52608)
        self.assertEqual(report["test"]["rows"], 8760)


if __name__ == "__main__":
    unittest.main()
