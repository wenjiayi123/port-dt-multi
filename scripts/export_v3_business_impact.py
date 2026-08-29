"""Export a transparent paired-window V3 business-impact scenario.

This report is deliberately descriptive.  It compares the deterministic MPC
controller with the neutral FCFS controller on the same chronological blind-test
windows.  The annual values are mechanical extrapolations of a 48-hour public-
data scenario, not operator savings or audited financial/carbon results.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.rl_training.statistics import bootstrap_summary
from app.services.rl_training.trainer import TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "evidence/v3"
ADVANTAGE_PATH = OUTPUT_DIR / "shanghai_public_advantage_v3.json"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compatible_controller_run(
    runs: list[dict[str, Any]], baseline: dict[str, Any], dataset_id: str
) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    baseline_dir = TRAINING_MANAGER.run_dir(str(baseline["job_id"]))
    baseline_config = read_json(baseline_dir / "config.json")
    for run in runs:
        if run.get("algorithm") != "mpc":
            continue
        if run.get("dataset_id") != dataset_id:
            continue
        if run.get("dataset_sha256") != baseline.get("dataset_sha256"):
            continue
        if run.get("evidence_label") != "DETERMINISTIC_CONTROLLER_BASELINE":
            continue
        run_dir = TRAINING_MANAGER.run_dir(str(run["job_id"]))
        config = read_json(run_dir / "config.json")
        evaluation = read_json(run_dir / "evaluation.json")
        same_protocol = all(
            config.get(field) == baseline_config.get(field)
            for field in ("environment_version", "episode_steps", "test_ratio", "validation_ratio")
        )
        if not same_protocol:
            continue
        if evaluation.get("split") != "chronological_blind_test_only":
            continue
        if float((evaluation.get("metrics") or {}).get("guardrail_violation_rate") or 0.0) > 0:
            continue
        if float((evaluation.get("metrics") or {}).get("action_projection_rate") or 1.0) > 0.9:
            continue
        eligible.append({**run, "evaluation": evaluation, "config": config})
    if not eligible:
        raise RuntimeError("no compatible zero-violation MPC baseline found")
    return max(eligible, key=lambda run: str(run.get("evaluated_at") or ""))


def main() -> None:
    advantage = read_json(ADVANTAGE_PATH)
    registry = read_json(TRAINING_MANAGER.benchmark_path)
    baseline_ref = advantage["baseline"]
    baseline_dir = TRAINING_MANAGER.run_dir(str(baseline_ref["job_id"]))
    baseline_evaluation = read_json(baseline_dir / "evaluation.json")
    baseline_config = read_json(baseline_dir / "config.json")
    candidate = compatible_controller_run(
        list(registry.get("runs") or []),
        baseline_ref,
        str(advantage["dataset"]["dataset_id"]),
    )
    candidate_evaluation = candidate["evaluation"]

    baseline_rows = list(baseline_evaluation.get("episode_metrics") or [])
    candidate_rows = list(candidate_evaluation.get("episode_metrics") or [])
    baseline_windows = (baseline_evaluation.get("evaluation_protocol") or {}).get("window_start_indices")
    candidate_windows = (candidate_evaluation.get("evaluation_protocol") or {}).get("window_start_indices")
    if baseline_windows != candidate_windows or not baseline_rows or len(baseline_rows) != len(candidate_rows):
        raise RuntimeError("MPC and FCFS evaluations are not paired on identical blind-test windows")

    directions = {
        "throughput_teu": "candidate_minus_baseline",
        "delay_index_mean": "baseline_minus_candidate",
        "energy_cost": "baseline_minus_candidate",
        "carbon_kg": "baseline_minus_candidate",
        "peak_kw": "baseline_minus_candidate",
    }
    paired: dict[str, Any] = {}
    for metric, direction in directions.items():
        values = [
            float(candidate_row[metric]) - float(baseline_row[metric])
            if direction == "candidate_minus_baseline"
            else float(baseline_row[metric]) - float(candidate_row[metric])
            for baseline_row, candidate_row in zip(baseline_rows, candidate_rows)
        ]
        baseline_mean = float(baseline_evaluation["metrics"][metric])
        candidate_mean = float(candidate_evaluation["metrics"][metric])
        absolute = candidate_mean - baseline_mean if direction == "candidate_minus_baseline" else baseline_mean - candidate_mean
        paired[metric] = {
            "direction": direction,
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "absolute_improvement": absolute,
            "relative_improvement": absolute / baseline_mean if baseline_mean else None,
            "paired_window_improvement": bootstrap_summary(values, seed=20260812),
        }

    episode_hours = float(baseline_config["episode_steps"]) * float(baseline_config["step_hours"])
    annualization_factor = 365.0 * 24.0 / episode_hours
    cost_delta = paired["energy_cost"]["absolute_improvement"]
    carbon_delta = paired["carbon_kg"]["absolute_improvement"]
    mpc_throughput = float(candidate_evaluation["metrics"]["throughput_teu"])
    mpc_unit_cost = float(candidate_evaluation["metrics"]["cost_per_teu"])
    mpc_unit_carbon = float(candidate_evaluation["metrics"]["carbon_kg_per_teu"])
    fcfs_unit_cost = float(baseline_evaluation["metrics"]["cost_per_teu"])
    fcfs_unit_carbon = float(baseline_evaluation["metrics"]["carbon_kg_per_teu"])
    mpc_equivalent_cost = (fcfs_unit_cost - mpc_unit_cost) * mpc_throughput
    mpc_equivalent_carbon = (fcfs_unit_carbon - mpc_unit_carbon) * mpc_throughput
    selected = advantage["selected"]
    selected_evaluations = [
        read_json(TRAINING_MANAGER.run_dir(str(job_id)) / "evaluation.json")
        for job_id in selected.get("job_ids") or []
    ]
    if len(selected_evaluations) < 3:
        raise RuntimeError("selected learned policy lacks three evaluation artifacts")
    baseline_unit_cost = float(baseline_evaluation["metrics"]["cost_per_teu"])
    baseline_unit_carbon = float(baseline_evaluation["metrics"]["carbon_kg_per_teu"])
    unit_cost_improvements: list[float] = []
    unit_carbon_improvements: list[float] = []
    avoided_cost_values: list[float] = []
    avoided_carbon_values: list[float] = []
    for evaluation in selected_evaluations:
        metrics = evaluation["metrics"]
        candidate_throughput = float(metrics["throughput_teu"])
        candidate_unit_cost = float(metrics["cost_per_teu"])
        candidate_unit_carbon = float(metrics["carbon_kg_per_teu"])
        unit_cost_improvements.append(1.0 - candidate_unit_cost / baseline_unit_cost)
        unit_carbon_improvements.append(1.0 - candidate_unit_carbon / baseline_unit_carbon)
        avoided_cost_values.append(
            (baseline_unit_cost - candidate_unit_cost) * candidate_throughput
        )
        avoided_carbon_values.append(
            (baseline_unit_carbon - candidate_unit_carbon) * candidate_throughput
        )
    learned_value = {
        "algorithm": selected["algorithm"],
        "name": selected["name"],
        "job_ids": selected["job_ids"],
        "seeds": selected["seeds"],
        "comparison_basis": "equivalent candidate throughput at FCFS unit intensity",
        "cost_per_teu_relative_improvement": bootstrap_summary(
            unit_cost_improvements, seed=20260812
        ),
        "carbon_per_teu_relative_improvement": bootstrap_summary(
            unit_carbon_improvements, seed=20260812
        ),
        "avoided_cost_per_episode": bootstrap_summary(
            avoided_cost_values, seed=20260812
        ),
        "avoided_carbon_kg_per_episode": bootstrap_summary(
            avoided_carbon_values, seed=20260812
        ),
        "annualized_avoided_cost": sum(avoided_cost_values)
        / len(avoided_cost_values)
        * annualization_factor,
        "annualized_avoided_carbon_kg": sum(avoided_carbon_values)
        / len(avoided_carbon_values)
        * annualization_factor,
        "claim_status": "EQUIVALENT_THROUGHPUT_VALUE_95CI"
        if min(
            bootstrap_summary(unit_cost_improvements, seed=20260812)["ci_low"],
            bootstrap_summary(unit_carbon_improvements, seed=20260812)["ci_low"],
        )
        > 0
        else "EQUIVALENT_THROUGHPUT_POINT_ESTIMATE_NOT_95CI_CONFIRMED",
        "not_absolute_bill_saving": True,
    }
    payload = {
        "schema": "port-dt-v3-business-impact-scenario.v1",
        "version": advantage["version"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "DESCRIPTIVE_PAIRED_PUBLIC_DATA_SCENARIO",
        "dataset": advantage["dataset"],
        "comparison": {
            "candidate": "mpc",
            "candidate_job_id": candidate["job_id"],
            "baseline": "fcfs",
            "baseline_job_id": baseline_ref["job_id"],
            "same_chronological_blind_test_windows": True,
            "window_count": len(baseline_rows),
            "episode_hours": episode_hours,
            "environment_version": baseline_config.get("environment_version"),
            "guardrail_violation_rate": candidate_evaluation["metrics"]["guardrail_violation_rate"],
            "action_projection_rate": candidate_evaluation["metrics"]["action_projection_rate"],
        },
        "metrics": paired,
        "mpc_efficiency_value": {
            "comparison_basis": "equivalent MPC throughput at FCFS unit intensity",
            "cost_per_teu_relative_improvement": 1.0 - mpc_unit_cost / fcfs_unit_cost,
            "carbon_per_teu_relative_improvement": 1.0 - mpc_unit_carbon / fcfs_unit_carbon,
            "avoided_cost_per_episode": mpc_equivalent_cost,
            "avoided_carbon_kg_per_episode": mpc_equivalent_carbon,
            "annualized_avoided_cost": mpc_equivalent_cost * annualization_factor,
            "annualized_avoided_carbon_kg": mpc_equivalent_carbon * annualization_factor,
            "not_absolute_bill_saving": True,
        },
        "learned_efficiency_value": learned_value,
        "scenario_value": {
            "currency": "CNY",
            "cost_saving_per_episode": cost_delta,
            "carbon_saving_kg_per_episode": carbon_delta,
            "annualization_factor": annualization_factor,
            "annualized_cost_saving": cost_delta * annualization_factor,
            "annualized_carbon_saving_kg": carbon_delta * annualization_factor,
            "throughput_gain_per_episode": paired["throughput_teu"]["absolute_improvement"],
            "absolute_total_cost_and_carbon_can_increase_with_throughput": True,
            "annualized_values_are_mechanical_extrapolations": True,
        },
        "selection_disclosure": "MPC is reported as the named deterministic safety controller; this descriptive business-impact export was generated after the controller benchmark and is not a preregistered superiority test.",
        "claim_boundary": "Public Shanghai aggregate throughput and public Yangshan reanalysis are used. Electricity tariff, carbon factor, terminal load, equipment and operating fields are engineering assumptions. Values are not Shanghai International Port Group savings, audited carbon reductions or production KPIs; replace them with authorized EMS/TOS/finance/carbon-ledger data and pass shadow-operation acceptance before any site claim.",
        "production_authority": False,
        "historical_evidence_preserved": True,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "shanghai_public_business_impact_v3.json"
    md_path = OUTPUT_DIR / "shanghai_public_business_impact_v3.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V3 Shanghai public-data business-impact scenario",
        "",
        f"- Status: `{payload['status']}`",
        f"- Comparison: **MPC vs FCFS**, {len(baseline_rows)} identical blind-test windows",
        f"- Window length: **{episode_hours:.0f} hours**",
        f"- Safety guardrail violation rate: **{100 * float(payload['comparison']['guardrail_violation_rate']):.2f}%**",
        "",
        "| Metric | Improvement | Relative improvement | Paired-window 95% CI |",
        "|---|---:|---:|---:|",
    ]
    units = {"throughput_teu": "TEU", "delay_index_mean": "index", "energy_cost": "CNY", "carbon_kg": "kg", "peak_kw": "kW"}
    for metric, row in paired.items():
        summary = row["paired_window_improvement"]
        lines.append(
            f"| {metric} | {row['absolute_improvement']:.2f} {units[metric]} | {100 * row['relative_improvement']:.2f}% | [{summary['ci_low']:.2f}, {summary['ci_high']:.2f}] {units[metric]} |"
        )
    value = payload["scenario_value"]
    lines.extend(
        [
            "",
            f"- Absolute total-cost difference, FCFS minus MPC: **CNY {value['annualized_cost_saving']:,.0f}/year** (may be negative when MPC handles more work)",
            f"- Absolute total-carbon difference, FCFS minus MPC: **{value['annualized_carbon_saving_kg'] / 1000:,.2f} tCO2/year** (may be negative when MPC handles more work)",
            f"- MPC equivalent-throughput avoided cost: **CNY {payload['mpc_efficiency_value']['annualized_avoided_cost']:,.0f}/year**",
            f"- MPC equivalent-throughput avoided carbon: **{payload['mpc_efficiency_value']['annualized_avoided_carbon_kg'] / 1000:,.2f} tCO2/year**",
            "",
            "## Learned-policy equivalent-throughput value",
            "",
            f"- Policy: **{learned_value['name']}**",
            f"- Status: `{learned_value['claim_status']}`",
            f"- Unit cost improvement: **{100 * learned_value['cost_per_teu_relative_improvement']['mean']:.2f}%**",
            f"- Unit carbon improvement: **{100 * learned_value['carbon_per_teu_relative_improvement']['mean']:.2f}%**",
            f"- Mechanical annualized avoided cost at equivalent throughput: **CNY {learned_value['annualized_avoided_cost']:,.0f}/year**",
            f"- Mechanical annualized avoided carbon at equivalent throughput: **{learned_value['annualized_avoided_carbon_kg'] / 1000:,.2f} tCO2/year**",
            "",
            "Equivalent-throughput avoided value is not an absolute electricity-bill reduction: it prices the learned policy's throughput at the FCFS unit intensity.",
            "",
            payload["selection_disclosure"],
            payload["claim_boundary"],
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    digest_path = OUTPUT_DIR / "shanghai_public_business_impact_v3.sha256"
    digest_path.write_text(
        f"{file_sha256(json_path)}  {json_path.name}\n{file_sha256(md_path)}  {md_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["scenario_value"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
