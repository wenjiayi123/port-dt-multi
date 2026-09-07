"""Describe V6 equivalent-throughput value from frozen paired windows.

This is a post-evaluation, linear annualization, not an observed annual saving
or a new evaluation. It never changes training, selection or champion pointers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services.rl_training.datasets import file_sha256
from app.services.rl_training.statistics import bootstrap_summary

ROOT = Path(__file__).resolve().parents[1]


def main():
    pointer_path = ROOT / "evidence/v6/coordinated_business/offline_champion.json"
    pointer = json.loads(pointer_path.read_text())
    report_path = ROOT / pointer["report_path"]
    assert file_sha256(report_path) == pointer["report_sha256"]
    assert file_sha256(ROOT / pointer["selected_model_path"]) == pointer["selected_model_sha256"]
    report = json.loads(report_path.read_text())
    window_path = report_path.with_name("blind_window_metrics.json")
    windows = json.loads(window_path.read_text())
    assert windows["selected_job_id"] == pointer["selected_job_id"]
    assert windows["window_start_indices"] == report["blind_test"]["paired_window_start_indices"]
    config_path = (ROOT / pointer["selected_model_path"]).with_name("config.json")
    config = json.loads(config_path.read_text())
    episode_steps = report["blind_test"]["episode_steps"]
    hours = episode_steps * float(config["step_hours"])
    factor = 8760.0 / hours
    starts = windows["window_start_indices"]
    assert all(b - a >= episode_steps for a, b in zip(starts, starts[1:]))
    assert len(windows["candidate"]) == report["blind_test"]["window_count"]
    metric_keys = ("energy_cost", "throughput_teu", "cost_per_teu", "carbon_kg", "carbon_kg_per_teu")
    for name, rows in {"candidate": windows["candidate"], **windows["baselines"]}.items():
        expected = (report["blind_test"]["selected_metrics"] if name == "candidate"
                    else report["blind_test"]["comparators"][name]["metrics"])
        for key in metric_keys:
            np.testing.assert_allclose(np.mean([row[key] for row in rows]), expected[key], rtol=1e-10)
        for row in rows:
            assert row["throughput_teu"] > 0
            np.testing.assert_allclose(row["energy_cost"] / row["throughput_teu"], row["cost_per_teu"], rtol=1e-10)
            np.testing.assert_allclose(row["carbon_kg"] / row["throughput_teu"], row["carbon_kg_per_teu"], rtol=1e-10)
    comparisons = {}
    for name, rows in windows["baselines"].items():
        paired = []
        for start, candidate, baseline in zip(starts, windows["candidate"], rows, strict=True):
            paired.append({
                "window_start_index": start,
                "rl_throughput_teu": candidate["throughput_teu"],
                "baseline_throughput_teu": baseline["throughput_teu"],
                "unit_cost_reduction_percent": 100 * (1 - candidate["cost_per_teu"] / baseline["cost_per_teu"]),
                "equivalent_throughput_annual_cost_avoidance_cny": (baseline["cost_per_teu"] - candidate["cost_per_teu"]) * candidate["throughput_teu"] * factor,
                "equivalent_throughput_annual_carbon_avoidance_tonnes": (baseline["carbon_kg_per_teu"] - candidate["carbon_kg_per_teu"]) * candidate["throughput_teu"] * factor / 1000,
                "absolute_annual_cost_difference_baseline_minus_rl_cny": (baseline["energy_cost"] - candidate["energy_cost"]) * factor,
            })
        comparisons[name] = {"paired_windows": paired, "summary": {
            key: bootstrap_summary([r[key] for r in paired], seed=20260907)
            for key in paired[0] if key != "window_start_index"
        }}
    now = datetime.now(timezone.utc)
    run_id = "v6-annual-value-" + now.strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "evidence/v6/coordinated_business/value_estimates" / run_id
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "port-v6-descriptive-annual-value.v1", "run_id": run_id,
        "calculated_at": now.isoformat(), "source_model_version": "V6",
        "selected_job_id": pointer["selected_job_id"], "model_sha256": pointer["selected_model_sha256"],
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in (report_path, window_path, config_path, Path(__file__))},
        "method": {"episode_hours": hours, "assumed_annual_hours": 8760, "annualization_factor": factor,
                   "formula": "mean((baseline_cost_per_teu_i - rl_cost_per_teu_i) * rl_throughput_teu_i) * 8760 / episode_hours",
                   "comparison_basis": "linear equivalent-throughput unit-intensity accounting, not a baseline rerun at increased throughput",
                   "confidence_interval": "2000 paired-window bootstrap resamples; conditional on fixed engineering assumptions; not annual forecast uncertainty",
                   "post_evaluation_calculation": True, "new_training_or_model_selection": False},
        "comparisons": comparisons,
        "claim_boundary": [
            "Throughput, tariff, load, carbon and resource assumptions retain the V6 public/engineering source boundaries.",
            "Assumes window-average conditions persist through a 8760-hour year; excludes seasonal, downtime and implementation-cost uncertainty.",
            "Cost avoidance is a unit-intensity counterfactual; it is not an observed bill reduction, revenue, net profit or field ROI.",
            "Actual paired total energy cost increases because the RL policy processes more work; signed absolute differences are reported alongside normalized value.",
            "Do not sum this value with V3, V4 or V7 estimates or compare version differences as measured incremental annual savings.",
        ],
        "simulation_mode": True, "live_data_verified": False, "dispatch_allowed": False, "production_authority": False,
    }
    target = output / "annualized_value.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(target.relative_to(ROOT)), "comparisons": {k: v["summary"] for k,v in comparisons.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
