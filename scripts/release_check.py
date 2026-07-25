"""Fail closed when checked-in business KPI or RL release evidence is stale."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.services.business_benchmark import ROOT, load_verified_report
from app.services.mobile_api.benchmark import (
    load_verified_report as load_mobile_workflow_report,
)
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.profiles import load_profile


REQUIRED = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "MODEL_GOVERNANCE.md",
    "docs/BUSINESS_KPI_BENCHMARK.md",
    "docs/RESUME_CLAIMS_WEB.md",
    "docs/SHARED_WEB_MOBILE_ARCHITECTURE.md",
    "docs/MOBILE_WORKFLOW_BENCHMARK.md",
    "docs/RESUME_CLAIMS_DUAL_FRONTEND.md",
    "config/business_kpi_benchmark_v1.json",
    "config/mobile_workflow_benchmark_v1.json",
    "data/rl/business_kpi_benchmark_v1.json",
    "data/mobile/mobile_workflow_benchmark_v1.json",
    "data/rl/business_kpi_benchmark_v1_daily.csv",
    "data/rl/datasets/public_port_ops_v1.meta.json",
    "data/rl/datasets/public_us_la_6min_v1.csv",
    "data/rl/datasets/public_us_la_6min_v1.meta.json",
    "config/ports/port-profile.schema.json",
    "config/ports/sgsin_public_replay_v2.json",
    "config/ports/us_la_public_benchmark_v2.json",
    "docs/DATASET_CARD_public_us_la_6min_v1.md",
    "docs/LANDING_ROADMAP_2026-07-25.md",
    "evidence/rl/public_port_ops_v1_benchmark.json",
    "evidence/rl/public_port_ops_v1_benchmark.md",
    "evidence/rl/public_port_ops_v1_benchmark.sha256",
    "evidence/rl/public_us_la_6min_v1_benchmark.json",
    "evidence/rl/public_us_la_6min_v1_benchmark.md",
    "evidence/rl/public_us_la_6min_v1_benchmark.sha256",
    "scripts/fetch_public_la_benchmark.py",
    "scripts/export_rl_evidence.py",
    "app/ui/adapters/rl_evidence_console.js",
    "docs/assets/training-center-algorithm-matrix-xiaoyi.png",
    "docs/assets/xiaoyi-system-assistant-button-linkage.png",
    "docs/assets/rl-training-console-real-backend.png",
    "docs/assets/seven-controller-backend-results.png",
    ".github/workflows/ci.yml",
)


def verify_portable_evidence(dataset_id: str, errors: list[str]) -> None:
    bundle_path = ROOT / "evidence" / "rl" / f"{dataset_id}_benchmark.json"
    sidecar_path = bundle_path.with_suffix(".sha256")
    encoded = bundle_path.read_bytes()
    observed = hashlib.sha256(encoded).hexdigest()
    expected = sidecar_path.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        errors.append(f"portable RL evidence hash mismatch: {dataset_id}")
        return
    bundle = json.loads(encoded)
    if bundle.get("schema") != "port-dt-rl-benchmark-evidence.v1":
        errors.append(f"portable RL evidence schema changed: {dataset_id}")
    if bundle.get("evidence_boundary", {}).get("production_kpi_claim") is not False:
        errors.append(f"portable evidence allows a production KPI claim: {dataset_id}")
    runs = bundle.get("runs") or []
    if dataset_id == "public_port_ops_v1":
        if not any(
            run.get("dataset_integrity", {}).get(
                "current_artifact_matches_training"
            )
            is False
            for run in runs
        ):
            errors.append("historical public_port_ops_v1 artifact boundary is missing")
        return
    for algorithm in ("sac", "ppo", "td3", "dqn", "a2c", "tqc"):
        formal = [
            run
            for run in runs
            if run.get("algorithm") == algorithm
            and run.get("evidence_label") == "RL_HELD_OUT_EVALUATION"
            and int(run.get("total_steps") or 0) >= 10_000
        ]
        seeds = {run.get("seed") for run in formal}
        if len(seeds) < 3:
            errors.append(f"{algorithm} lacks three formal seeds in portable evidence")
        for run in formal:
            if run.get("training", {}).get("render_calls") != 0:
                errors.append(f"{algorithm} formal training rendered frames")
            if (
                run.get("dataset_integrity", {}).get(
                    "current_artifact_matches_training"
                )
                is not True
            ):
                errors.append(f"{algorithm} formal run dataset hash is stale")
            if run.get("model_integrity", {}).get("verified") is not True:
                errors.append(f"{algorithm} formal model hash was not verified")
    if not any(
        run.get("algorithm") == "mpc"
        and run.get("evidence_label") == "DETERMINISTIC_CONTROLLER_BASELINE"
        and int(run.get("evaluation", {}).get("episodes") or 0) >= 10
        for run in runs
    ):
        errors.append("portable evidence lacks deterministic MPC baseline")


def main() -> int:
    errors: list[str] = []
    report: dict = {}
    claims: dict = {}
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing release evidence: {relative}")
    try:
        report = load_verified_report()
        claims = report.get("resume_claims_rounded_percent") or {}
        expected = {
            "berth_utilization_relative_improvement_percent": 9.0,
            "average_waiting_time_reduction_percent": 17.0,
            "scenario_energy_cost_reduction_percent": 12.0,
        }
        if claims != expected:
            errors.append(f"resume KPI claims changed: {claims}")
        if report.get("dataset", {}).get("split_sizes") != {
            "train": 35064,
            "validation": 8784,
            "test": 8760,
        }:
            errors.append("business benchmark split sizes changed")
        if report.get("evidence_boundary", {}).get("measured_port_kpi") is not False:
            errors.append("benchmark incorrectly claims measured port KPI")
        if (
            report.get("evidence_boundary", {}).get(
                "geographically_coherent_single_port_series"
            )
            is not False
        ):
            errors.append("benchmark incorrectly labels derived hourly data as a measured single-port series")
        if report.get("sensitivity", {}).get("predeclared_scenarios") != 27:
            errors.append("business benchmark sensitivity grid changed")
        if any(
            summary.get("n") != 365
            for summary in report.get("test", {})
            .get("uncertainty", {})
            .values()
        ):
            errors.append("business benchmark no longer covers 365 complete test days")
        if (
            report.get("evidence_boundary", {}).get(
                "scenario_parameters_measured_or_calibrated_at_port"
            )
            is not False
        ):
            errors.append("benchmark incorrectly claims port-calibrated parameters")
        if report.get("release_gate", {}).get("production_claim_allowed") is not False:
            errors.append("benchmark incorrectly allows production claim")
    except Exception as exc:
        errors.append(f"business benchmark verification failed: {exc}")
    try:
        dataset = load_port_dataset("public_port_ops_v1")
        if dataset.fingerprint != report.get("dataset", {}).get("sha256"):
            errors.append("RL dataset fingerprint differs from business report")
        if dataset.describe().get("quality", {}).get("training_eligible") is not True:
            errors.append("public dataset no longer passes the RL quality gate")
        if (
            dataset.metadata.get("geographically_coherent_single_port_series")
            is not False
        ):
            errors.append("public fixture geographic boundary is missing")
        if dataset.metadata.get("official_input_geography_coherent") is not True:
            errors.append("official MPA anchors no longer share one port geography")
        if any(
            "noaa" in str(source.get("publisher") or "").lower()
            for source in dataset.metadata.get("sources", [])
        ):
            errors.append("cross-geography NOAA source reintroduced")
    except Exception as exc:
        errors.append(f"RL dataset verification failed: {exc}")
    try:
        dataset = load_port_dataset("public_us_la_6min_v1")
        quality = dataset.describe().get("quality") or {}
        if dataset.fingerprint != "9455a32251c521f567887f0205b0b5db4556801924d443ae1856db9ab4262897":
            errors.append("Los Angeles public benchmark fingerprint changed")
        if dataset.rows != 87_459:
            errors.append("Los Angeles public benchmark row count changed")
        if quality.get("training_eligible") is not True:
            errors.append("Los Angeles public benchmark failed its quality gate")
        evidence = quality.get("evidence") or {}
        if evidence.get("tier") != "public_measured_enriched":
            errors.append("Los Angeles public benchmark evidence tier changed")
        if evidence.get("independent_source_observations") != 262_347:
            errors.append("Los Angeles independent-observation count changed")
        if quality.get("available_factor_count") != 5:
            errors.append("Los Angeles optional-factor coverage changed")
        if dataset.metadata.get("environment_version") != "port_ops_v2":
            errors.append("Los Angeles dataset is not bound to port_ops_v2")
        if dataset.metadata.get("port_profile_id") != "us_la_public_benchmark_v2":
            errors.append("Los Angeles port-profile binding changed")
    except Exception as exc:
        errors.append(f"Los Angeles dataset verification failed: {exc}")
    try:
        for profile_id in (
            "sgsin_public_replay_v2",
            "us_la_public_benchmark_v2",
        ):
            profile = load_profile(profile_id)
            if profile.get("control_authority") != "recommendation_only":
                errors.append(f"port profile grants execution authority: {profile_id}")
            if profile.get("environment_version") != "port_ops_v2":
                errors.append(f"port profile environment changed: {profile_id}")
    except Exception as exc:
        errors.append(f"port-profile verification failed: {exc}")
    try:
        verify_portable_evidence("public_port_ops_v1", errors)
        verify_portable_evidence("public_us_la_6min_v1", errors)
    except Exception as exc:
        errors.append(f"portable RL evidence verification failed: {exc}")
    try:
        workflow = load_mobile_workflow_report()
        results = workflow.get("results") or {}
        if workflow.get("operations", {}).get("total") != 500:
            errors.append("mobile workflow operation count changed")
        for metric in (
            "duplicate_suppression_percent",
            "conflicting_reuse_block_percent",
            "unsafe_dispatch_block_percent",
            "client_audit_accept_percent",
        ):
            if results.get(metric) != 100.0:
                errors.append(f"mobile workflow metric changed: {metric}")
        if results.get("audit_chain_valid") is not True:
            errors.append("mobile workflow audit chain failed")
        if results.get("production_execution_receipt_count") != 0:
            errors.append("mobile benchmark fabricated production receipts")
    except Exception as exc:
        errors.append(f"mobile workflow verification failed: {exc}")
    server = (ROOT / "app/server.py").read_text(encoding="utf-8")
    for marker in (
        "/api/rl/business-benchmark",
        "businessEvidenceChip",
        "businessBerthKpi",
        "businessWaitKpi",
        "businessCostKpi",
        "/ui/adapters/rl_evidence_console.js",
    ):
        if marker not in server:
            errors.append(f"Web evidence surface missing: {marker}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("release check: FAIL")
        return 1
    mobile_api = (ROOT / "app/services/mobile_api/api.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "/status",
        "/situation",
        "/strategy/candidates",
        "/strategy/decisions",
        "/audit/events",
        "Idempotency-Key",
    ):
        if marker not in mobile_api:
            errors.append(f"shared mobile API surface missing: {marker}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("release check: FAIL")
        return 1
    print(
        "release check: PASS "
        f"(business KPI {json.dumps(claims, ensure_ascii=False)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
