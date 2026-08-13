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
    ".gitattributes",
    ".env.example",
    "LICENSE",
    "Dockerfile",
    "SECURITY.md",
    "requirements.txt",
    "requirements-linux.in",
    "requirements-linux.lock",
    "requirements-ci.in",
    "requirements-ci.lock",
    ".dockerignore",
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
    "data/rl/benchmarks.json",
    "data/rl/model_registry.json",
    "data/rl/datasets/public_port_ops_v1.meta.json",
    "data/rl/datasets/public_us_la_6min_v1.csv",
    "data/rl/datasets/public_us_la_6min_v1.meta.json",
    "config/ports/port-profile.schema.json",
    "config/ports/sgsin_public_replay_v2.json",
    "config/ports/us_la_public_benchmark_v2.json",
    "docs/DATASET_CARD_public_us_la_6min_v1.md",
    "docs/DATASET_CARD_public_cn_sha_hourly_v3.md",
    "docs/DATASET_CARD_public_cn_sha_forward_2026m05_v1.md",
    "docs/SITE_DATA_REPLACEMENT_CONTRACT_V3.md",
    "docs/PRODUCTION_READINESS.md",
    "docs/V3_TECHNICAL_EVIDENCE.md",
    "docs/V3_HR_TECHNICAL_AUDIT.md",
    "data/public_sources/shanghai_port_mot_2024_2025.json",
    "data/public_sources/shanghai_yangshan_reanalysis_2024_2025.csv",
    "data/public_sources/shanghai_port_mot_2026_forward.json",
    "data/public_sources/shanghai_yangshan_reanalysis_2026_01_05.csv",
    "data/rl/datasets/public_cn_sha_hourly_v3.csv",
    "data/rl/datasets/public_cn_sha_hourly_v3.meta.json",
    "data/rl/datasets/public_cn_sha_forward_2026m05_v1.csv",
    "data/rl/datasets/public_cn_sha_forward_2026m05_v1.meta.json",
    "config/ports/cn_sha_public_benchmark_v3.json",
    "config/v3_advantage_benchmark.json",
    "config/rl_business_profiles_v3.json",
    "evidence/v3/shanghai_public_advantage_v3.json",
    "evidence/v3/shanghai_public_advantage_v3.md",
    "evidence/v3/shanghai_public_advantage_v3.sha256",
    "evidence/v3/shanghai_public_business_impact_v3.json",
    "evidence/v3/shanghai_public_business_impact_v3.md",
    "evidence/v3/shanghai_public_business_impact_v3.sha256",
    "evidence/v3/strong_baseline_evidence_v3.json",
    "evidence/v3/strong_baseline_evidence_v3.md",
    "evidence/v3/strong_baseline_evidence_v3.sha256",
    "evidence/v3/public_cn_sha_hourly_v3_benchmark.json",
    "evidence/v3/public_cn_sha_hourly_v3_benchmark.md",
    "evidence/v3/public_cn_sha_hourly_v3_benchmark.sha256",
    "scripts/fetch_shanghai_public_dataset.py",
    "scripts/fetch_shanghai_forward_2026.py",
    "scripts/export_v3_advantage.py",
    "scripts/export_v3_strong_baselines.py",
    "scripts/export_v3_business_impact.py",
    "scripts/export_v3_runtime_model.py",
    "app/adapters/telemetry_calibrated_replay.py",
    "app/services/v3_runtime.py",
    "app/services/copilot/mission_control.py",
    "app/services/copilot/api.py",
    "config/shore_bess_v3.json",
    "config/bess_energy_v3.json",
    "app/services/rl_model/shore_bess/v3_environment.py",
    "app/services/rl_model/bess_energy/v3_environment.py",
    "scripts/train_shore_bess_v3_safe.py",
    "scripts/train_bess_energy_v3_safe.py",
    "scripts/train_shore_bess_v32_value.py",
    "scripts/evaluate_bess_grid_only_forward_v32.py",
    "evidence/v3/shore_bess/latest.json",
    "evidence/v3/bess_energy/latest.json",
    "evidence/v3/bess_energy/latest_grid_only.json",
    "evidence/v3/value_improvement_v32.json",
    "evidence/v3/runtime/selected_sac_v3.zip",
    "evidence/v3/runtime/selected_sac_v3.config.json",
    "evidence/v3/runtime/runtime_model.json",
    "evidence/v3/runtime/runtime_model.sha256",
    "data/rl/runs/rl-20260813T063524662Z/config.json",
    "data/rl/runs/rl-20260813T063524662Z/status.json",
    "data/rl/runs/rl-20260813T063524662Z/manifest.json",
    "data/rl/runs/rl-20260813T063524662Z/metrics.jsonl",
    "data/rl/runs/rl-20260813T063524662Z/evaluation.json",
    "data/rl/runs/rl-20260813T063524662Z/evaluation_trajectory.json",
    "data/rl/runs/rl-20260813T063524662Z/model.zip",
    "data/rl/runs/rl-20260813T064228701Z/MODEL_CARD.md",
    "data/rl/runs/rl-20260813T064228701Z/config.json",
    "data/rl/runs/rl-20260813T064228701Z/evaluation.json",
    "data/rl/runs/rl-20260813T064228701Z/evaluation_trajectory.json",
    "data/rl/runs/rl-20260813T064228701Z/manifest.json",
    "data/rl/runs/rl-20260813T064228701Z/metrics.jsonl",
    "data/rl/runs/rl-20260813T064228701Z/model.zip",
    "data/rl/runs/rl-20260813T064228701Z/model_card.json",
    "data/rl/runs/rl-20260813T064228701Z/monitor.csv",
    "data/rl/runs/rl-20260813T064228701Z/status.json",
    "data/rl/runs/rl-20260813T064228701Z/validation_evaluation.json",
    "data/rl/runs/rl-20260812T065606173Z/status.json",
    "data/rl/runs/rl-20260812T065606173Z/manifest.json",
    "data/rl/runs/rl-20260812T065606173Z/metrics.jsonl",
    "data/rl/runs/rl-20260812T065606173Z/model.zip",
    "data/rl/runs/rl-20260812T070320120Z/status.json",
    "data/rl/runs/rl-20260812T070320120Z/manifest.json",
    "data/rl/runs/rl-20260812T070320120Z/metrics.jsonl",
    "data/rl/runs/rl-20260812T070320120Z/model.zip",
    "data/rl/runs/rl-20260812T070452272Z/status.json",
    "data/rl/runs/rl-20260812T070452272Z/manifest.json",
    "data/rl/runs/rl-20260812T070452272Z/metrics.jsonl",
    "data/rl/runs/rl-20260812T070452272Z/model.zip",
    "data/rl/runs/rl-20260812T074531393Z/config.json",
    "data/rl/runs/rl-20260812T074531393Z/status.json",
    "data/rl/runs/rl-20260812T074531393Z/manifest.json",
    "data/rl/runs/rl-20260812T074531393Z/evaluation.json",
    "data/rl/runs/rl-20260812T074531393Z/evaluation_trajectory.json",
    "app/services/rl_model/hvac_cooling/artifacts/hvac_cooling_state.json",
    "app/services/rl_model/hvac_cooling/artifacts/policy_evaluate_history.jsonl",
    "app/services/rl_model/shore_bess/artifacts/shore_bess_outputs.jsonl",
    "app/services/rl_model/bess_energy/policy_evaluate_history.jsonl",
    "app/services/rl_model/bess_energy/offline_dataset.jsonl",
    "app/services/rl_model/yard_crane/policy_evaluate_history.jsonl",
    "app/services/rl_model/yard_crane/artifacts/offline_dataset_crane.jsonl",
    "app/services/rl_model/yard_crane/artifacts/offline_dataset_crane_aug.jsonl",
    "app/services/rl_model/yard_lighting/artifacts/offline_train.jsonl",
    "app/services/rl_model/hvac_cooling/data/ahu_zones_master.csv",
    "app/services/rl_model/hvac_cooling/data/demand_window_config.json",
    "app/services/rl_model/hvac_cooling/data/grid_ef.csv",
    "app/services/rl_model/hvac_cooling/data/hvac_telemetry.csv",
    "app/services/rl_model/hvac_cooling/data/load_forecast.csv",
    "app/services/rl_model/hvac_cooling/data/market_price.csv",
    "app/services/rl_model/hvac_cooling/data/plant_efficiency_map.csv",
    "app/services/rl_model/hvac_cooling/data/plant_master.json",
    "app/services/rl_model/hvac_cooling/data/weather_forecast.csv",
    "app/services/rl_model/yard_crane/data/crane_telemetry.csv",
    "app/services/rl_model/yard_crane/data/cranes_master.csv",
    "app/services/rl_model/yard_crane/data/dr_events.json",
    "app/services/rl_model/yard_crane/data/grid_ef.csv",
    "app/services/rl_model/yard_crane/data/grid_meter.csv",
    "app/services/rl_model/yard_crane/data/job_events.csv",
    "app/services/rl_model/yard_crane/data/market_price.csv",
    "app/services/rl_model/yard_crane/data/queue_forecast.csv",
    "app/services/rl_model/yard_crane/data/yard_blocks.csv",
    "app/services/rl_model/yard_lighting/data/activity_forecast.csv",
    "app/services/rl_model/yard_lighting/data/complaints_events.csv",
    "app/services/rl_model/yard_lighting/data/config_limits.json",
    "app/services/rl_model/yard_lighting/data/grid_ef.csv",
    "app/services/rl_model/yard_lighting/data/lighting_telemetry.csv",
    "app/services/rl_model/yard_lighting/data/market_price.csv",
    "app/services/rl_model/yard_lighting/data/weather_astro.csv",
    "app/services/rl_model/yard_lighting/data/zones_master.csv",
    "evidence/v3/shore_bess/runs/shore-bess-v3-safe-20260813T015000Z/seed_43/selected_model.zip",
    "evidence/v3/shore_bess/runs/shore-bess-v3-safe-20260813T015000Z/seed_143/selected_model.zip",
    "evidence/v3/shore_bess/runs/shore-bess-v3-safe-20260813T015000Z/seed_243/selected_model.zip",
    "evidence/v3/bess_energy/runs/bess-energy-v3-safe-20260813T043000Z/seed_47/selected_model.zip",
    "evidence/v3/bess_energy/runs/bess-energy-v3-safe-20260813T043000Z/seed_147/selected_model.zip",
    "evidence/v3/bess_energy/runs/bess-energy-v3-safe-20260813T043000Z/seed_247/selected_model.zip",
    "evidence/v3/bess_energy/runs/bess-energy-v32-grid-only-balanced-20260813T090000Z/seed_71/selected_model.zip",
    "evidence/v3/bess_energy/runs/bess-energy-v32-grid-only-balanced-20260813T090000Z/seed_171/selected_model.zip",
    "evidence/v3/bess_energy/runs/bess-energy-v32-grid-only-balanced-20260813T090000Z/seed_271/selected_model.zip",
    "docs/V3_RUNTIME_DATA_CONTRACT.md",
    "app/ui/v3/index.html",
    "app/ui/v3/v3.css",
    "app/ui/v3/v3.js",
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
    "app/ui/adapters/xiaoyi_sprite.js",
    "app/ui/ops_copilot.html",
    "app/static/xiaoyi_maritime_officer.png",
    "app/static/vendor/echarts/echarts.min.js",
    "app/static/vendor/echarts/LICENSE",
    "docs/assets/system-overview-provenance-governance.png",
    "docs/assets/training-center-algorithm-matrix-xiaoyi.png",
    "docs/assets/xiaoyi-system-assistant-button-linkage.png",
    "docs/assets/rl-training-console-real-backend.png",
    "docs/assets/seven-controller-backend-results.png",
    ".github/workflows/ci.yml",
    ".github/workflows/codeql.yml",
)


def verify_container_contract(errors: list[str]) -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    required_copies = (
        "app ./app",
        "data ./data",
        "config ./config",
        "evidence ./evidence",
        "docs ./docs",
        "scripts ./scripts",
    )
    for marker in required_copies:
        if marker not in dockerfile:
            errors.append(f"Docker image omits required release content: {marker}")
    if "USER portdt" not in dockerfile:
        errors.append("Docker image does not declare its non-root runtime user")
    if not dockerfile.startswith("FROM python:3.12-slim@sha256:"):
        errors.append("Docker base image is not pinned to an immutable digest")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for vulnerable_pin in (
        "torch==2.2.2",
        "stable-baselines3==2.3.2",
        "sb3-contrib==2.3.0",
    ):
        if vulnerable_pin in requirements:
            errors.append(
                "Release pins a known-vulnerable Intel macOS RL dependency: "
                f"{vulnerable_pin}"
            )
    ci_workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    linux_lock = (ROOT / "requirements-linux.lock").read_text(encoding="utf-8")
    ci_lock = (ROOT / "requirements-ci.lock").read_text(encoding="utf-8")
    for name, lock in (("Linux", linux_lock), ("CI", ci_lock)):
        if "--hash=sha256:" not in lock:
            errors.append(f"{name} dependency lock does not contain distribution hashes")
    for package in ("torch==2.13.0", "stable-baselines3==2.9.0", "sb3-contrib==2.9.0"):
        if package not in linux_lock:
            errors.append(f"Linux dependency lock omits the supported RL runtime: {package}")
    if "pip-audit==2.10.1" not in ci_lock:
        errors.append("CI dependency lock omits the vulnerability auditor")
    if "--require-hashes -r requirements-linux.lock" not in dockerfile:
        errors.append("Docker dependency installation does not require locked hashes")
    if "--require-hashes -r requirements-ci.lock" not in ci_workflow:
        errors.append("CI dependency installation does not require locked hashes")
    if "| python" in ci_workflow:
        errors.append("CI contains a download-then-execute style curl pipeline")
    codeql_workflow = (ROOT / ".github/workflows/codeql.yml").read_text(encoding="utf-8")
    top_level = codeql_workflow.split("jobs:", 1)[0]
    if "security-events: write" in top_level:
        errors.append("CodeQL grants security-events write at workflow scope")
    security_policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    if "https://github.com/wenjiayi123/port-dt-multi/security/advisories/new" not in security_policy:
        errors.append("Security policy lacks a private vulnerability reporting URL")
    ignored = {
        line.strip().rstrip("/")
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if "evidence" in ignored or "evidence/v3" in ignored:
        errors.append("Docker context excludes portable V3 evidence")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    archive_allowlists = (
        "!evidence/v3/runtime/selected_sac_v3.zip",
        "!evidence/v3/shore_bess/runs/shore-bess-v3-safe-20260813T015000Z/seed_*/selected_model.zip",
        "!evidence/v3/bess_energy/runs/bess-energy-v3-safe-20260813T043000Z/seed_*/selected_model.zip",
        "!evidence/v3/bess_energy/runs/bess-energy-v32-grid-only-balanced-20260813T090000Z/seed_*/selected_model.zip",
    )
    gitignore_rules = {line.strip() for line in gitignore}
    for archive_allowlist in archive_allowlists:
        if archive_allowlist not in gitignore_rules:
            errors.append(
                "Git release excludes a selected V3 policy archive: "
                f"missing {archive_allowlist}"
            )

    attributes = {
        line.strip()
        for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for dataset in (
        "data/public_sources/shanghai_yangshan_reanalysis_2024_2025.csv",
        "data/public_sources/shanghai_yangshan_reanalysis_2026_01_05.csv",
        "data/rl/datasets/public_cn_sha_forward_2026m05_v1.csv",
        "data/rl/datasets/public_cn_sha_hourly_v3.csv",
    ):
        marker = f"{dataset} binary"
        if marker not in attributes:
            errors.append(
                "Git may normalize byte-hash evidence in a clean clone: "
                f"missing {marker}"
            )
    hvac_marker = "app/services/rl_model/hvac_cooling/data/*.csv binary"
    if hvac_marker not in attributes:
        errors.append(
            "Git may normalize HVAC byte-hash evidence in a clean clone: "
            f"missing {hvac_marker}"
        )


def verify_portable_evidence(
    dataset_id: str,
    errors: list[str],
    *,
    evidence_folder: str = "evidence/rl",
) -> None:
    bundle_path = ROOT / evidence_folder / f"{dataset_id}_benchmark.json"
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
    expected_algorithms = (
        ("sac", "ppo", "td3", "dqn", "a2c", "tqc", "qrdqn", "trpo", "recurrent_ppo", "ars")
        if dataset_id == "public_cn_sha_hourly_v3"
        else ("sac", "ppo", "td3", "dqn", "a2c", "tqc")
    )
    for algorithm in expected_algorithms:
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
            if dataset_id == "public_cn_sha_hourly_v3":
                if not (run.get("training", {}).get("optimizer_history") or []):
                    errors.append(f"{algorithm} portable evidence lacks optimizer history")
                if (run.get("validation_evaluation") or {}).get("split") != "chronological_validation_only":
                    errors.append(f"{algorithm} portable evidence lacks validation-only selection metrics")
    if not any(
        run.get("algorithm") == "mpc"
        and run.get("evidence_label") == "DETERMINISTIC_CONTROLLER_BASELINE"
        and int(run.get("evaluation", {}).get("episodes") or 0) >= 10
        for run in runs
    ):
        errors.append("portable evidence lacks deterministic MPC baseline")
    if dataset_id == "public_cn_sha_hourly_v3" and not any(
        run.get("algorithm") == "fcfs"
        and run.get("evidence_label") == "DETERMINISTIC_CONTROLLER_BASELINE"
        for run in runs
    ):
        errors.append("Shanghai portable evidence lacks neutral FCFS baseline")


def main() -> int:
    errors: list[str] = []
    report: dict = {}
    claims: dict = {}
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing release evidence: {relative}")
    mission_api = (ROOT / "app/services/copilot/api.py").read_text(encoding="utf-8")
    mission_ui = (ROOT / "app/ui/ops_copilot.html").read_text(encoding="utf-8")
    for marker in (
        '"/mission"',
        '"/handoff"',
        "true_xiaoyi_called",
        "context_sha256",
        "production_authority",
    ):
        if marker not in mission_api:
            errors.append(f"Xiaoyi mission API marker missing: {marker}")
    for marker in (
        "/api/copilot/mission",
        "missionRail",
        "engineProof",
        "confirmHandoff",
        "xiaoyi_maritime_officer.png",
    ):
        if marker not in mission_ui:
            errors.append(f"Xiaoyi mission UI marker missing: {marker}")
    verify_container_contract(errors)
    try:
        runtime_root = ROOT / "evidence/v3/runtime"
        runtime = json.loads((runtime_root / "runtime_model.json").read_text(encoding="utf-8"))
        if runtime.get("schema") != "port-dt-v3-runtime-policy.v1":
            errors.append("V3 runtime policy schema changed")
        if runtime.get("production_authority") is not False:
            errors.append("V3 runtime policy incorrectly grants production authority")
        for line in (runtime_root / "runtime_model.sha256").read_text(encoding="utf-8").splitlines():
            expected, name = line.split(maxsplit=1)
            observed = hashlib.sha256((runtime_root / name.strip()).read_bytes()).hexdigest()
            if observed != expected:
                errors.append(f"V3 runtime artifact hash mismatch: {name.strip()}")
        shanghai = load_port_dataset("public_cn_sha_hourly_v3")
        if runtime.get("dataset_sha256") != shanghai.fingerprint:
            errors.append("V3 runtime policy dataset hash is stale")
    except Exception as exc:
        errors.append(f"V3 runtime policy verification failed: {exc}")
    xiaoyi_asset = ROOT / "app/static/xiaoyi_maritime_officer.png"
    if xiaoyi_asset.is_file():
        observed_xiaoyi_sha256 = hashlib.sha256(xiaoyi_asset.read_bytes()).hexdigest()
        if observed_xiaoyi_sha256 != "8f56a569dcb098ef08cd2bed92ffc098474c7d6ef911141760982e9323ffe714":
            errors.append("Xiaoyi Q-style character asset fingerprint changed")
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
            "cn_sha_public_benchmark_v3",
        ):
            profile = load_profile(profile_id)
            if profile.get("control_authority") != "recommendation_only":
                errors.append(f"port profile grants execution authority: {profile_id}")
            expected_environment = (
                "port_ops_v3" if profile_id == "cn_sha_public_benchmark_v3" else "port_ops_v2"
            )
            if profile.get("environment_version") != expected_environment:
                errors.append(f"port profile environment changed: {profile_id}")
    except Exception as exc:
        errors.append(f"port-profile verification failed: {exc}")
    try:
        shanghai = load_port_dataset("public_cn_sha_hourly_v3")
        shanghai_quality = shanghai.describe(0.2, 0.1).get("quality") or {}
        if shanghai.fingerprint != "803214ea0202abde241f75a28d7bf46b9c7ad801d40605a0916ec14ef7906a01":
            errors.append("Shanghai V3 public benchmark fingerprint changed")
        if shanghai.rows != 17_544:
            errors.append("Shanghai V3 row count changed")
        if shanghai_quality.get("training_eligible") is not True:
            errors.append("Shanghai V3 dataset failed its quality gate")
        if shanghai.metadata.get("measured_columns"):
            errors.append("Shanghai V3 reanalysis was incorrectly labelled measured")
        if shanghai.metadata.get("independent_source_observations") != 17_566:
            errors.append("Shanghai V3 independent-observation count changed")
        if shanghai.metadata.get("environment_version") != "port_ops_v3":
            errors.append("Shanghai V3 dataset is not bound to port_ops_v3")
        split = shanghai.describe(0.2, 0.1)
        if (split.get("train_rows"), split.get("validation_rows"), split.get("test_rows")) != (12_280, 1_755, 3_509):
            errors.append("Shanghai V3 three-way chronological split changed")
    except Exception as exc:
        errors.append(f"Shanghai V3 dataset verification failed: {exc}")
    try:
        forward = load_port_dataset("public_cn_sha_forward_2026m05_v1")
        if forward.fingerprint != "616fe7cde24695f0d19118c64d1e5c534f9adee47a886b33b6003e7e372bb06a":
            errors.append("Shanghai 2026 forward-challenge fingerprint changed")
        if forward.rows != 3_624:
            errors.append("Shanghai 2026 forward-challenge row count changed")
        if forward.metadata.get("measured_columns"):
            errors.append("Shanghai 2026 forward reanalysis was incorrectly labelled measured")
        if forward.metadata.get("independent_source_observations") != 3_628:
            errors.append("Shanghai 2026 forward independent-observation count changed")
        split_policy = forward.metadata.get("split_policy") or {}
        if split_policy.get("role") != "forward_challenge_only":
            errors.append("Shanghai 2026 forward data is not isolated as challenge-only")
        if split_policy.get("candidate_selection_allowed") is not False:
            errors.append("Shanghai 2026 forward data may influence candidate selection")
        anchors = json.loads(
            (ROOT / "data/public_sources/shanghai_port_mot_2026_forward.json").read_text(
                encoding="utf-8"
            )
        )
        if [row.get("cumulative_teu_10000") for row in anchors.get("observations") or []] != [
            941,
            1_411,
            1_896,
            2_375,
        ]:
            errors.append("Shanghai 2026 official cumulative throughput anchors changed")
        if (anchors.get("derivation_boundary") or {}).get("official_observations") != 4:
            errors.append("Shanghai 2026 official-observation boundary changed")
    except Exception as exc:
        errors.append(f"Shanghai 2026 forward-challenge verification failed: {exc}")
    try:
        value = json.loads(
            (ROOT / "evidence/v3/value_improvement_v32.json").read_text(encoding="utf-8")
        )
        if value.get("schema") != "port-dt-v32-value-improvement.v1":
            errors.append("V3.2 value-improvement registry schema changed")
        policy = value.get("policy") or {}
        if policy.get("historical_metrics_preserved") is not True:
            errors.append("V3.2 value work did not preserve historical evidence")
        if policy.get("forward_challenge_used_for_tuning") is not False:
            errors.append("V3.2 forward challenge leaked into tuning")
        if policy.get("production_authority") is not False:
            errors.append("V3.2 value registry grants production authority")
        modules = value.get("modules") or {}
        expected_statuses = {
            "yard_lighting": "retained_constraint_ceiling",
            "hvac": "candidate_rejected_strict_peak_gate",
            "shore_bess": "balanced_candidate_rejected",
            "bess_energy": "grid_only_forward_pass",
        }
        if {key: (modules.get(key) or {}).get("status") for key in expected_statuses} != expected_statuses:
            errors.append("V3.2 module admission decisions changed")
        grid_latest = json.loads(
            (ROOT / "evidence/v3/bess_energy/latest_grid_only.json").read_text(
                encoding="utf-8"
            )
        )
        forward_path = ROOT / str(grid_latest.get("forward_evidence_path") or "")
        if hashlib.sha256(forward_path.read_bytes()).hexdigest() != grid_latest.get(
            "forward_evidence_sha256"
        ):
            errors.append("BESS grid-only forward evidence hash mismatch")
        forward_evidence = json.loads(forward_path.read_text(encoding="utf-8"))
        if forward_evidence.get("status") != "GRID_ONLY_PROFILE_FORWARD_PASS":
            errors.append("BESS grid-only forward profile is not admitted")
        if forward_evidence.get("admitted_public_offline_profile") is not True:
            errors.append("BESS grid-only public-offline admission changed")
        seeds = forward_evidence.get("per_seed") or []
        if len(seeds) != 3 or not all(row.get("admitted") is True for row in seeds):
            errors.append("BESS grid-only forward profile lacks three admitted seeds")
        for row in seeds:
            metrics = row.get("metrics") or {}
            if metrics.get("reserve_revenue_cny") != 0.0 or metrics.get("dr_revenue_cny") != 0.0:
                errors.append("BESS grid-only profile contains unsupported market revenue")
            if metrics.get("claim_eligible") is not False:
                errors.append("BESS grid-only profile fabricates a site claim")
            model_path = ROOT / str(row.get("model_path") or "")
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != row.get("model_sha256"):
                errors.append(f"BESS grid-only selected model hash mismatch: {row.get('seed')}")
    except Exception as exc:
        errors.append(f"V3.2 value-improvement verification failed: {exc}")
    try:
        for module_id, expected_schema, expected_states, expected_rewards, minimum_constraints in (
            ("shore_bess", "port-dt-shore-bess-formal-evidence.v2", 34, 8, 12),
            ("bess_energy", "port-dt-bess-energy-formal-evidence.v1", 40, 9, 15),
        ):
            evidence_root = ROOT / "evidence" / "v3" / module_id
            latest = json.loads((evidence_root / "latest.json").read_text(encoding="utf-8"))
            report_path = ROOT / str(latest.get("report_path") or "")
            observed_report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
            if observed_report_hash != latest.get("report_sha256"):
                errors.append(f"{module_id} latest report hash mismatch")
                continue
            specialized = json.loads(report_path.read_text(encoding="utf-8"))
            if specialized.get("schema") != expected_schema:
                errors.append(f"{module_id} formal evidence schema changed")
            if specialized.get("production_authority") is not False:
                errors.append(f"{module_id} evidence grants production authority")
            if (specialized.get("training") or {}).get("training_render_calls") != 0:
                errors.append(f"{module_id} formal training rendered frames")
            contract = specialized.get("contract") or {}
            if contract.get("state_dimensions") != expected_states:
                errors.append(f"{module_id} state contract changed")
            if len(contract.get("reward_components") or []) != expected_rewards:
                errors.append(f"{module_id} reward contract changed")
            if len(contract.get("hard_constraints") or []) < minimum_constraints:
                errors.append(f"{module_id} hard-constraint coverage regressed")
            gates = specialized.get("quality_gates") or {}
            if gates.get("convergence_passed") is not True or gates.get("safety_passed") is not True:
                errors.append(f"{module_id} latest formal run failed convergence or safety")
            if gates.get("public_offline_admitted") is not True:
                errors.append(f"{module_id} latest formal run is not public-offline admitted")
            for model in (specialized.get("artifacts") or {}).get("models") or []:
                model_path = ROOT / str(model.get("path") or "")
                if hashlib.sha256(model_path.read_bytes()).hexdigest() != model.get("sha256"):
                    errors.append(f"{module_id} selected model hash mismatch: {model.get('seed')}")
        bess_report_path = ROOT / json.loads(
            (ROOT / "evidence/v3/bess_energy/latest.json").read_text(encoding="utf-8")
        )["report_path"]
        bess_report = json.loads(bess_report_path.read_text(encoding="utf-8"))
        supplement = bess_report.get("scenario_supplement") or {}
        if supplement.get("observed_site_event_rows") != 0 or supplement.get("claim_as_real_market_settlement") is not False:
            errors.append("BESS engineering events can be misread as observed settlement evidence")
        if not (ROOT / "evidence/v3/bess_energy/history_index.jsonl").read_text(encoding="utf-8").count("\n") >= 2:
            errors.append("BESS append-only failed/pass history is incomplete")
    except Exception as exc:
        errors.append(f"specialized BESS evidence verification failed: {exc}")
    try:
        v3_json = ROOT / "evidence/v3/shanghai_public_advantage_v3.json"
        v3_md = ROOT / "evidence/v3/shanghai_public_advantage_v3.md"
        sidecar_lines = (ROOT / "evidence/v3/shanghai_public_advantage_v3.sha256").read_text(encoding="utf-8").splitlines()
        expected_hashes = {line.split()[1]: line.split()[0] for line in sidecar_lines if len(line.split()) == 2}
        for evidence_path in (v3_json, v3_md):
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != expected_hashes.get(evidence_path.name):
                errors.append(f"V3 advantage evidence hash mismatch: {evidence_path.name}")
        advantage = json.loads(v3_json.read_text(encoding="utf-8"))
        selected = advantage.get("selected") or {}
        if advantage.get("schema") != "port-dt-v3-advantage-evidence.v1":
            errors.append("V3 advantage evidence schema changed")
        if advantage.get("production_authority") is not False:
            errors.append("V3 advantage evidence grants production authority")
        if advantage.get("historical_evidence_preserved") is not True:
            errors.append("V3 evidence does not assert append-only history")
        if (advantage.get("baseline") or {}).get("environment_version") != "port_ops_v3":
            errors.append("V3 advantage is not bound to the causal port_ops_v3 environment")
        selection_protocol = advantage.get("selection_protocol") or {}
        if selection_protocol.get("algorithm_selection") != "chronological_validation_only":
            errors.append("V3 algorithm selection is not isolated to validation rows")
        if selection_protocol.get("final_advantage_report") != "chronological_blind_test_only":
            errors.append("V3 final advantage is not reported on the blind test")
        if selection_protocol.get("blind_test_used_for_selection") is not False:
            errors.append("V3 blind test may have influenced algorithm selection")
        if len(set(selected.get("seeds") or [])) < 3:
            errors.append("V3 selected policy lacks three formal seeds")
        portable_v3_bundle = json.loads(
            (ROOT / "evidence/v3/public_cn_sha_hourly_v3_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        portable_v3_runs = {
            str(run.get("job_id")): run
            for run in portable_v3_bundle.get("runs") or []
            if run.get("job_id")
        }
        for job_id in selected.get("job_ids") or []:
            validation_path = ROOT / "data/rl/runs" / str(job_id) / "validation_evaluation.json"
            if validation_path.is_file():
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
            else:
                validation = (
                    portable_v3_runs.get(str(job_id), {}).get("validation_evaluation")
                    or {}
                )
            if not validation:
                errors.append(f"V3 selected run lacks validation evidence: {job_id}")
                continue
            if validation.get("split") != "chronological_validation_only":
                errors.append(f"V3 validation artifact has the wrong split: {job_id}")
            if (validation.get("evaluation_protocol") or {}).get("render_during_policy_execution") is not False:
                errors.append(f"V3 validation rendered policy execution: {job_id}")
        required_business_metrics = {
            "throughput_teu", "delay_index_mean", "energy_cost", "carbon_kg",
            "peak_kw", "cost_per_teu", "carbon_kg_per_teu",
            "service_completion_ratio", "queue_end_teu", "action_projection_rate",
            "action_projection_correction_kw_mean", "action_projection_severity_mean",
            "action_projection_terminal_reachability_rate", "terminal_soc_error",
        }
        if not required_business_metrics.issubset((selected.get("blind_test_metrics") or {}).keys()):
            errors.append("V3 selected policy lacks the expanded blind-test business metrics")
        safety_admission = selected.get("safety_admission") or {}
        if float(safety_admission.get("guardrail_violation_rate_max_observed") or 0.0) > 0:
            errors.append("V3 selected policy crossed a hard guardrail")
        configured_projection_max = float(
            (((advantage.get("benchmark_contract") or {}).get("eligibility") or {})
             .get("action_projection_rate_max") or 0.0)
        )
        if configured_projection_max <= 0 or configured_projection_max > 0.6:
            errors.append("V3 projection admission threshold is not hardened to 60% or lower")
        if float(safety_admission.get("action_projection_rate_max_observed") or 1.0) > configured_projection_max:
            errors.append("V3 selected policy relies excessively on safety projection")
        hardening = advantage.get("projection_hardening") or {}
        if hardening.get("historical_preserved") is not True:
            errors.append("V3.1 projection evidence was not preserved")
        if float(hardening.get("current_mean") or 1.0) >= float(hardening.get("historical_mean") or 0.0):
            errors.append("V3.2 projection dependence did not improve over V3.1")
        historical_report = ROOT / str(hardening.get("historical_report") or "")
        if not historical_report.is_file():
            errors.append("V3.1 archived advantage report is missing")
        if float(safety_admission.get("terminal_soc_error_max_observed") or 0.0) > 0.000001:
            errors.append("V3 selected policy does not restore terminal SOC")
        if float((selected.get("weighted_relative_improvement") or {}).get("mean") or 0.0) <= 0:
            errors.append("V3 selected policy has no positive version-pinned advantage")
    except Exception as exc:
        errors.append(f"V3 advantage verification failed: {exc}")
    try:
        impact_json = ROOT / "evidence/v3/shanghai_public_business_impact_v3.json"
        impact_md = ROOT / "evidence/v3/shanghai_public_business_impact_v3.md"
        impact_sidecar = (ROOT / "evidence/v3/shanghai_public_business_impact_v3.sha256").read_text(encoding="utf-8").splitlines()
        impact_hashes = {line.split()[1]: line.split()[0] for line in impact_sidecar if len(line.split()) == 2}
        for evidence_path in (impact_json, impact_md):
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != impact_hashes.get(evidence_path.name):
                errors.append(f"V3 business-impact evidence hash mismatch: {evidence_path.name}")
        impact = json.loads(impact_json.read_text(encoding="utf-8"))
        if impact.get("schema") != "port-dt-v3-business-impact-scenario.v1":
            errors.append("V3 business-impact schema changed")
        if (impact.get("comparison") or {}).get("environment_version") != "port_ops_v3":
            errors.append("V3 business impact is not bound to port_ops_v3")
        if impact.get("production_authority") is not False:
            errors.append("V3 business-impact evidence grants production authority")
        if (impact.get("scenario_value") or {}).get("annualized_values_are_mechanical_extrapolations") is not True:
            errors.append("V3 annualized value is missing the mechanical-extrapolation boundary")
        if (impact.get("mpc_efficiency_value") or {}).get("not_absolute_bill_saving") is not True:
            errors.append("V3 MPC equivalent-throughput value can be misread as absolute bill saving")
        if float((impact.get("comparison") or {}).get("action_projection_rate") or 1.0) > 0.9:
            errors.append("V3 MPC business comparison relies excessively on action projection")
        if "not Shanghai International Port Group savings" not in str(impact.get("claim_boundary") or ""):
            errors.append("V3 business-impact group-savings disclaimer is missing")
    except Exception as exc:
        errors.append(f"V3 business-impact verification failed: {exc}")
    try:
        strong_json = ROOT / "evidence/v3/strong_baseline_evidence_v3.json"
        strong_md = ROOT / "evidence/v3/strong_baseline_evidence_v3.md"
        strong_lines = (ROOT / "evidence/v3/strong_baseline_evidence_v3.sha256").read_text(encoding="utf-8").splitlines()
        strong_hashes = {line.split()[1]: line.split()[0] for line in strong_lines if len(line.split()) == 2}
        for evidence_path in (strong_json, strong_md):
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != strong_hashes.get(evidence_path.name):
                errors.append(f"V3 strong-baseline evidence hash mismatch: {evidence_path.name}")
        strong = json.loads(strong_json.read_text(encoding="utf-8"))
        if strong.get("schema") != "port-dt-v3-strong-baseline-evidence.v1":
            errors.append("V3 strong-baseline schema changed")
        if (strong.get("protocol") or {}).get("split") != "chronological_blind_test_only":
            errors.append("V3 strong-baseline comparison is not blind-test-only")
        if (strong.get("protocol") or {}).get("paired_comparison") is not True:
            errors.append("V3 strong-baseline windows are not paired")
        comparisons = strong.get("comparisons") or {}
        if set(comparisons) != {"fcfs_neutral", "engineering_ops_rule", "mpc"}:
            errors.append("V3 strong-baseline comparator coverage changed")
        gate = strong.get("strong_baseline_gate") or {}
        if gate.get("fcfs_only_is_not_sufficient_for_production_claim") is not True:
            errors.append("V3 allows FCFS-only production claims")
        if gate.get("measured_current_operations_baseline_available") is not False:
            errors.append("V3 fabricates a measured incumbent baseline")
        if gate.get("production_claim_admitted") is not False:
            errors.append("V3 strong-baseline gate grants production authority")
        if (strong.get("site_replacement") or {}).get("required") is not True:
            errors.append("V3 strong-baseline site replacement is not required")
    except Exception as exc:
        errors.append(f"V3 strong-baseline verification failed: {exc}")
    try:
        from app.services.v3_port_ai import BUSINESS_CAPABILITIES, _business_depth

        depths = [_business_depth(row["id"]) for row in BUSINESS_CAPABILITIES]
        if len(depths) != 12:
            errors.append("V3 business-domain coverage changed")
        if sum(bool(row.get("model_output_available")) for row in depths) != 9:
            errors.append("V3 business execution-depth classification changed")
        if any(row.get("production_ready") is not False for row in depths):
            errors.append("V3 business domain grants production readiness")
        for capability, depth in zip(BUSINESS_CAPABILITIES, depths):
            if not depth.get("runtime_endpoints") or not depth.get("site_blockers"):
                errors.append(f"V3 business domain lacks runtime/site contract: {capability['id']}")
            for artifact in depth.get("code_artifacts") or []:
                if artifact.get("exists") is not True or len(str(artifact.get("sha256") or "")) != 64:
                    errors.append(f"V3 business code evidence failed: {capability['id']}")
    except Exception as exc:
        errors.append(f"V3 business execution-depth verification failed: {exc}")
    try:
        verify_portable_evidence("public_port_ops_v1", errors)
        verify_portable_evidence("public_us_la_6min_v1", errors)
        verify_portable_evidence(
            "public_cn_sha_hourly_v3",
            errors,
            evidence_folder="evidence/v3",
        )
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
        "v3_port_ai_router",
    ):
        if marker not in server:
            errors.append(f"Web evidence surface missing: {marker}")
    v3_ui = (ROOT / "app/ui/v3/index.html").read_text(encoding="utf-8")
    for marker in (
        "公开数据离线验证",
        "无生产控制权",
        "SITE DATA REPLACEMENT CONTRACT",
        "/api/v3/overview",
        "查看训练指标",
        "查看技术链路",
        "绝对业务结果",
        "安全稳健性",
        "孪生可靠性",
        "部署自检",
        "/api/v3/twin/reliability?refresh=1",
        "/health/ready",
        "restoreHashTarget",
        "查看数据血缘",
        "验收规则",
    ):
        if marker not in v3_ui and marker not in (ROOT / "app/ui/v3/v3.js").read_text(encoding="utf-8"):
            errors.append(f"V3 evidence UI marker missing: {marker}")
    operations = (ROOT / "app/operations.py").read_text(encoding="utf-8")
    for marker in (
        "PORT_DT_RATE_LIMIT_RPM",
        "PORT_DT_MAX_REQUEST_BYTES",
        "PORT_DT_SHADOW_ACCEPTANCE_PATH",
        "site_evidence_consistency",
        "Strict-Transport-Security",
        "Content-Security-Policy",
    ):
        if marker not in operations:
            errors.append(f"Production readiness hardening marker missing: {marker}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    screenshot_paths = (
        "docs/assets/system-overview-provenance-governance.png",
        "docs/assets/training-center-algorithm-matrix-xiaoyi.png",
        "docs/assets/rl-training-console-real-backend.png",
        "docs/assets/seven-controller-backend-results.png",
        "docs/assets/xiaoyi-system-assistant-button-linkage.png",
    )
    if any(readme.count(path) != 1 for path in screenshot_paths):
        errors.append("README must embed each of the five evidence screenshots exactly once")
    for script_path in (
        ROOT / "app/ui/adapters/xiaoyi_sprite.js",
        ROOT / "app/ui/adapters/rl_evidence_console.js",
    ):
        source = script_path.read_text(encoding="utf-8")
        if "/static/xiaoyi_maritime_officer.png" not in source:
            errors.append(f"Xiaoyi Q-style asset is not wired: {script_path.name}")
        if "xiaoyi_maritime_officer.svg" in source:
            errors.append(f"legacy Xiaoyi SVG is still referenced: {script_path.name}")
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
