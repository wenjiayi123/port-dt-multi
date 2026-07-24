"""Fail closed when checked-in business KPI or RL release evidence is stale."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.business_benchmark import ROOT, load_verified_report
from app.services.mobile_api.benchmark import (
    load_verified_report as load_mobile_workflow_report,
)
from app.services.rl_training.datasets import load_port_dataset


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
    ".github/workflows/ci.yml",
)


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
