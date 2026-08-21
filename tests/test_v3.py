from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import server
from app.adapters.telemetry_calibrated_replay import CalibratedReplayTelemetry
from app.services.rl_training.baselines import FCFSNeutralPolicy
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.environment import PortOperationsEnv
from app.services.rl_training.mpc import MPCPolicy
from app.services.rl_training.profiles import load_profile
from app.services.rl_training.trainer import TRAINING_MANAGER
from app.services.v3_port_ai import (
    BUSINESS_CAPABILITIES,
    _algorithm_rows,
    v3_data_readiness,
    v3_overview,
)
from app.services.rl_model.yard_lighting.api import _as_bool as lighting_api_bool
from app.services.rl_model.yard_lighting.rl_engine import _as_bool as lighting_engine_bool
from app.services.rl_model.shore_bess.v3_environment import (
    ACTION_NAMES as SHORE_BESS_ACTIONS,
    CONTRACT as SHORE_BESS_CONTRACT,
    STATE_NAMES as SHORE_BESS_STATES,
    ShoreBESSEnv,
    chronological_slices as shore_bess_slices,
    load_config as load_shore_bess_config,
)
from app.services.rl_actions.api import resolve_action


ROOT = Path(__file__).resolve().parents[1]


class V3DatasetTests(unittest.TestCase):
    def test_calibrated_replay_is_continuous_mass_conserving_and_not_measured(self):
        telemetry = CalibratedReplayTelemetry(history_points=60)
        state = telemetry.current_port_state()
        assets = telemetry.asset_breakdown(state)
        status = telemetry.source_status()
        self.assertEqual(status["mode"], "calibrated_public_replay_simulator")
        self.assertTrue(status["continuous"])
        self.assertFalse(status["measured"])
        self.assertFalse(status["production"])
        self.assertEqual(status["assets"], 11)
        self.assertEqual(len(status["sha256"]), 64)
        self.assertAlmostEqual(sum(assets.values()), state["base_load_kw"], places=6)
        now = datetime.now(timezone.utc).timestamp()
        polled = telemetry.get_series("qc-01", "active_power_kw", now - 180, now, 60)
        self.assertEqual(len(polled), 4)
        self.assertTrue(all(point["v"] > 0 and point["measured"] is False for point in polled))

    def test_shanghai_official_total_is_conserved(self):
        anchor = json.loads(
            (ROOT / "data/public_sources/shanghai_port_mot_2024_2025.json").read_text(
                encoding="utf-8"
            )
        )
        annual_cumulative = {}
        for row in anchor["observations"]:
            year = str(row["period"])[:4]
            annual_cumulative[year] = max(
                float(row["cumulative_teu_10000"]),
                annual_cumulative.get(year, 0.0),
            )
        official_total = sum(annual_cumulative.values()) * 10_000
        with (ROOT / "data/rl/datasets/public_cn_sha_hourly_v3.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            allocated_total = sum(float(row["throughput_teu"]) for row in csv.DictReader(stream))
        self.assertAlmostEqual(official_total, 106_570_000.0, places=3)
        self.assertAlmostEqual(allocated_total, official_total, places=2)

    def test_metadata_never_labels_reanalysis_as_measured(self):
        dataset = load_port_dataset("public_cn_sha_hourly_v3")
        self.assertEqual(dataset.metadata["measured_columns"], [])
        self.assertIn("wind_speed_mps", dataset.metadata["public_reanalysis_columns"])
        self.assertIn("berth_occupancy_ratio", dataset.metadata["derived_columns"])
        self.assertIn("visibility_km", dataset.metadata["unavailable_factors"])

    def test_2026_forward_challenge_is_pinned_separate_and_not_terminal_telemetry(self):
        dataset = load_port_dataset("public_cn_sha_forward_2026m05_v1")
        self.assertEqual(dataset.rows, 3624)
        self.assertEqual(dataset.timestamps[0], "2026-01-01T00:00:00Z")
        self.assertEqual(dataset.timestamps[-1], "2026-05-31T23:00:00Z")
        self.assertEqual(dataset.metadata["measured_columns"], [])
        self.assertEqual(dataset.metadata["split_policy"]["role"], "forward_challenge_only")
        self.assertFalse(dataset.metadata["split_policy"]["candidate_selection_allowed"])
        self.assertEqual(dataset.metadata["independent_source_observations"], 3628)
        self.assertIn("terminal_metering", dataset.metadata["unavailable_factors"])
        anchor = json.loads(
            (ROOT / "data/public_sources/shanghai_port_mot_2026_forward.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [row["cumulative_teu_10000"] for row in anchor["observations"]],
            [941, 1411, 1896, 2375],
        )
        self.assertTrue(all(len(row["source_sha256"]) == 64 for row in anchor["observations"]))
        with dataset.path.open(encoding="utf-8", newline="") as stream:
            allocated = sum(float(row["throughput_teu"]) for row in csv.DictReader(stream))
        self.assertAlmostEqual(allocated, 23_750_000.0, places=2)

    def test_fcfs_comparator_is_neutral_and_deterministic(self):
        policy = FCFSNeutralPolicy(5)
        first, _ = policy.predict([1, 2, 3])
        second, _ = policy.predict([9, 8, 7])
        self.assertEqual(first.tolist(), [0.0] * 5)
        self.assertEqual(second.tolist(), [0.0] * 5)
        self.assertFalse(policy.parameters()["holdout_tuning"])

    def test_mpc_command_respects_terminal_soc_reachability(self):
        policy = MPCPolicy(
            action_dim=5,
            episode_steps=48,
            soc_min=0.15,
            soc_max=0.90,
            initial_soc=0.55,
            bess_capacity_kwh=100_000,
            bess_power_kw=10_000,
        )
        last_step_charge = policy._terminal_feasible_bess_action(
            1.0,
            soc=0.55,
            progress=1.0,
        )
        self.assertAlmostEqual(last_step_charge, 0.0, places=8)
        self.assertTrue(policy.parameters()["terminal_soc_aware"])

    def test_v3_service_and_allocation_gain_consumes_more_power(self):
        dataset = load_port_dataset("public_cn_sha_hourly_v3")
        train_slice, _validation_slice, test_slice = dataset.split_three_way()
        kwargs = {
            "dataset": dataset,
            "data_slice": test_slice,
            "normalization_slice": train_slice,
            "environment_version": "port_ops_v3",
            "port_profile": load_profile("cn_sha_public_benchmark_v3"),
            "demand_cap_kw": 36000,
            "episode_steps": 48,
            "training": False,
        }
        neutral = PortOperationsEnv(**kwargs)
        high_service = PortOperationsEnv(**kwargs)
        neutral.reset(seed=42, options={"start_index": 0})
        high_service.reset(seed=42, options={"start_index": 0})
        _obs, _reward, _terminated, _truncated, neutral_info = neutral.step(
            np.zeros(5, dtype=np.float32)
        )
        _obs, _reward, _terminated, _truncated, high_info = high_service.step(
            np.asarray([0, 1, 0, 1, 1], dtype=np.float32)
        )
        self.assertGreater(high_info["served_teu"], neutral_info["served_teu"])
        self.assertGreater(high_info["net_load_kw"], neutral_info["net_load_kw"])
        self.assertGreater(high_info["service_load_delta_kw"], 0)
        self.assertGreater(high_info["allocation_load_delta_kw"], 0)


class V3FactsApiTests(unittest.TestCase):
    def test_xiaoyi_mission_uses_runtime_context_and_has_no_production_authority(self):
        client = TestClient(server.app)
        payload = client.post(
            "/api/copilot/mission",
            json={
                "mission": "strategy",
                "query": "解释当前策略与准入门",
                "engine": "local_rag",
                "asset_id": "qc-01",
            },
        ).json()
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["context_sha256"]), 64)
        self.assertFalse(payload["production_authority"])
        self.assertFalse(payload["llm"]["true_xiaoyi_called"])
        self.assertEqual(payload["llm"]["engine_execution"], "local_evidence_fallback")
        context = payload["context"]
        self.assertEqual(context["source"]["mode"], "calibrated_public_replay_simulator")
        self.assertFalse(context["source"]["measured"])
        self.assertEqual(context["policy"]["dataset_id"], "public_cn_sha_hourly_v3")
        self.assertFalse(context["policy"]["production_authority"])
        self.assertIn("TOS/VTS作业与船期", context["missing_site_factors"])
        self.assertIn("production_authority", json.dumps(payload["audit_packet"]))

    def test_xiaoyi_context_uses_registered_asset_ids_and_fails_closed_for_unknown_asset(self):
        client = TestClient(server.app)
        for asset_id in ("qc-01", "yard-01", "hvac-01", "shore-01", "bess-01", "agv-01", "lighting-01"):
            with self.subTest(asset_id=asset_id):
                payload = client.get(
                    "/api/copilot/context",
                    params={"mission": "situation", "asset_id": asset_id},
                ).json()
                self.assertIn(payload["status"], {"ready", "review"})
                self.assertTrue(payload["context"]["source"]["sample_count"] > 0)
        missing = client.get(
            "/api/copilot/context",
            params={"mission": "situation", "asset_id": "unknown-site-asset"},
        ).json()
        self.assertEqual(missing["status"], "unavailable")
        self.assertEqual(missing["overall_state"], "data_unavailable")
        self.assertFalse(missing["production_authority"])

    def test_xiaoyi_frontline_intents_resolve_to_allowlisted_missions(self):
        expected = {
            "研判当前态势": "summarize_current_situation",
            "未来六小时有风险吗": "review_twin_forecast",
            "为什么这样调度": "explain_current_strategy",
            "帮我做告警分诊": "triage_monitoring",
            "准备交接班": "prepare_shift_handoff",
            "做一次策略预演": "prepare_strategy_dry_run",
        }
        for instruction, action_id in expected.items():
            with self.subTest(instruction=instruction):
                action = resolve_action(instruction)["action"]
                self.assertEqual(action["id"], action_id)
                self.assertEqual(action["category"], "xiaoyi_mission")
                self.assertFalse(action["requires_human_confirm"])

    def test_xiaoyi_handoff_preview_does_not_persist_without_confirmation(self):
        client = TestClient(server.app)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "handoffs.jsonl"
            with patch("app.services.copilot.mission_control.HANDOFF_LOG", log_path):
                payload = client.post(
                    "/api/copilot/handoff",
                    json={"asset_id": "qc-01", "operator": "tester", "confirm": False},
                ).json()
            self.assertFalse(payload["persisted"])
            self.assertEqual(payload["status"], "confirmation_required")
            self.assertFalse(payload["production_action_executed"])
            self.assertFalse(log_path.exists())
            self.assertEqual(len(payload["packet"]["handoff_sha256"]), 64)

    def test_v32_value_improvement_is_clickable_and_preserves_rejected_champions(self):
        client = TestClient(server.app)
        expected = {
            "yard-lighting": "retained_constraint_ceiling",
            "hvac": "candidate_rejected_strict_peak_gate",
            "shore-bess": "balanced_candidate_rejected",
            "bess-energy": "grid_only_forward_pass",
        }
        for endpoint, status in expected.items():
            with self.subTest(endpoint=endpoint):
                payload = client.get(f"/api/v3/modules/{endpoint}/evidence").json()
                self.assertEqual(payload["value_improvement"]["status"], status)
                self.assertFalse(payload["boundary"]["production_authority"])
        html = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for button in ("btn-b-v32", "btn-c-v32", "btn-d-v32", "btn-be-v32"):
            self.assertIn(f'id="{button}"', html)
            self.assertIn(f"getElementById('{button}')", html)
        self.assertEqual(
            json.loads((ROOT / "evidence/v3/hvac/latest.json").read_text(encoding="utf-8"))["run_id"],
            "hvac-v3-safe-20260813T100000Z",
        )
        self.assertEqual(
            json.loads((ROOT / "evidence/v3/shore_bess/latest.json").read_text(encoding="utf-8"))["run_id"],
            "shore-bess-v3-safe-20260813T015000Z",
        )

    def test_grid_only_bess_forward_profile_has_no_unobserved_market_revenue(self):
        pointer_path = ROOT / "evidence/v3/bess_energy/latest_grid_only.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        forward_path = ROOT / pointer["forward_evidence_path"]
        self.assertEqual(pointer["forward_evidence_sha256"], hashlib.sha256(forward_path.read_bytes()).hexdigest())
        evidence = json.loads(forward_path.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "GRID_ONLY_PROFILE_FORWARD_PASS")
        self.assertEqual(evidence["seed_pass_rate"], 1.0)
        self.assertFalse(evidence["profile"]["market_revenue_enabled"])
        self.assertEqual(len(evidence["per_seed"]), 3)
        for row in evidence["per_seed"]:
            metrics = row["metrics"]
            self.assertTrue(row["admitted"])
            self.assertGreaterEqual(metrics["cost_reduction_vs_no_bess_percent"], 0.0)
            self.assertGreaterEqual(metrics["carbon_reduction_vs_no_bess_percent"], 0.0)
            self.assertGreaterEqual(metrics["peak_reduction_vs_no_bess_percent"], 0.0)
            self.assertEqual(metrics["reserve_revenue_cny"], 0.0)
            self.assertEqual(metrics["dr_revenue_cny"], 0.0)
            self.assertFalse(metrics["claim_eligible"])

    def test_asset_modules_expose_two_real_per_seed_training_process_metrics(self):
        client = TestClient(server.app)
        expected_checkpoints = {
            "shore-bess": 27,
            "bess-energy": 24,
            "hvac": 24,
            "yard-crane": 21,
            "yard-lighting": 21,
        }
        for module, checkpoint_count in expected_checkpoints.items():
            with self.subTest(module=module):
                process = client.get(f"/api/v3/modules/{module}/evidence").json()["training_process"]
                self.assertEqual(process["source"], "append_only_seed_metrics_jsonl")
                self.assertFalse(process["retrained_for_display"])
                self.assertFalse(process["interpolated_points"])
                self.assertFalse(process["frontend_random_noise"])
                self.assertEqual(process["total_persisted_checkpoints"], checkpoint_count)
                self.assertEqual(len(process["series"]), 3)
                self.assertTrue(all(len(seed["sha256"]) == 64 for seed in process["series"]))
                self.assertTrue(
                    all(
                        {"imitation_loss", "validation_reward_mean", "optimizer_updates"}
                        <= point.keys()
                        for seed in process["series"]
                        for point in seed["points"]
                    )
                )

    def test_asset_modules_expose_dense_checkpoint_reward_replay_without_retraining(self):
        client = TestClient(server.app)
        expected_samples = {
            "shore-bess": 459,
            "bess-energy": 408,
            "hvac": 936,
            "yard-crane": 819,
            "yard-lighting": 210,
        }
        for module, sample_count in expected_samples.items():
            with self.subTest(module=module):
                replay = client.get(f"/api/v3/modules/{module}/evidence").json()["checkpoint_reward_replay"]
                self.assertTrue(replay["available"])
                self.assertEqual(replay["source"], "deterministic_post_training_checkpoint_replay")
                self.assertEqual(replay["split"], "fixed_validation_only")
                self.assertEqual(replay["sample_every_environment_steps"], 10)
                self.assertFalse(replay["retrained_model"])
                self.assertFalse(replay["training_time_log"])
                self.assertFalse(replay["blind_test_access"])
                self.assertFalse(replay["frontend_interpolation"])
                self.assertFalse(replay["frontend_random_noise"])
                self.assertEqual(replay["total_reward_samples"], sample_count)
                self.assertEqual(len(replay["artifact_sha256"]), 64)
                self.assertEqual(len(replay["series"]), 3)
                self.assertTrue(all(seed["records"] > 0 for seed in replay["series"]))
                self.assertTrue(
                    all(
                        {"reward_block_mean", "reward_delta_from_epoch1", "environment_step_end"}
                        <= point.keys()
                        for seed in replay["series"]
                        for point in seed["points"]
                    )
                )
                self.assertTrue(
                    all(
                        len(checkpoint["checkpoint_sha256"]) == 64
                        for seed in replay["series"]
                        for checkpoint in seed["checkpoint_summaries"]
                    )
                )

    def test_formal_business_advantage_is_positive_and_history_is_append_only(self):
        advantage = json.loads(
            (ROOT / "evidence/v3/shanghai_public_advantage_v3.json").read_text(encoding="utf-8")
        )
        impact = json.loads(
            (ROOT / "evidence/v3/shanghai_public_business_impact_v3.json").read_text(encoding="utf-8")
        )
        self.assertTrue(advantage["selected"]["strict_advantage"])
        self.assertGreater(advantage["selected"]["weighted_relative_improvement"]["ci_low"], 0)
        self.assertGreater(impact["learned_efficiency_value"]["cost_per_teu_relative_improvement"]["ci_low"], 0)
        self.assertGreater(impact["learned_efficiency_value"]["carbon_per_teu_relative_improvement"]["ci_low"], 0)
        self.assertTrue(impact["historical_evidence_preserved"])

    def test_v32_projection_dependency_is_lower_and_strictly_admitted(self):
        advantage = json.loads(
            (ROOT / "evidence/v3/shanghai_public_advantage_v3.json").read_text(encoding="utf-8")
        )
        selected = advantage["selected"]
        hardening = advantage["projection_hardening"]
        self.assertEqual(advantage["version"], "3.2.0")
        self.assertTrue(hardening["historical_preserved"])
        self.assertGreater(hardening["historical_mean"], hardening["current_mean"])
        self.assertGreater(hardening["relative_reduction"], 0.30)
        self.assertLessEqual(
            selected["safety_admission"]["action_projection_rate_max_observed"],
            advantage["benchmark_contract"]["eligibility"]["action_projection_rate_max"],
        )
        self.assertEqual(selected["safety_admission"]["guardrail_violation_rate_max_observed"], 0)
        self.assertEqual(selected["safety_admission"]["terminal_soc_error_max_observed"], 0)
        for metric in (
            "action_projection_correction_kw_mean",
            "action_projection_severity_mean",
            "action_projection_terminal_reachability_rate",
        ):
            self.assertIn(metric, selected["blind_test_metrics"])
        archived = ROOT / hardening["historical_report"]
        self.assertTrue(archived.is_file())

    def test_v32_ui_exposes_projection_hardening_and_dynamic_gate(self):
        script = (ROOT / "app/ui/v3/v3.js").read_text(encoding="utf-8")
        self.assertIn("V3.1 → V3.2 投影依赖", script)
        self.assertIn("action_projection_terminal_reachability_rate", script)
        self.assertIn("benchmark_contract?.eligibility?.action_projection_rate_max", script)

    def test_clone_without_ignored_run_registry_uses_portable_v3_evidence(self):
        missing = ROOT / "does-not-exist"
        with (
            patch.object(TRAINING_MANAGER, "benchmark_path", missing / "benchmarks.json"),
            patch.object(TRAINING_MANAGER, "run_dir", return_value=missing),
        ):
            rows = {row["id"]: row for row in _algorithm_rows()}
        for algorithm in (
            "sac", "ppo", "td3", "dqn", "a2c", "tqc", "qrdqn", "trpo",
            "recurrent_ppo", "ars",
        ):
            self.assertGreaterEqual(rows[algorithm]["formal_runs"], 3)
            self.assertEqual(rows[algorithm]["v3_profiles"][0]["seeds"], [42, 142, 242])
            self.assertIn(
                "evidence/v3/public_cn_sha_hourly_v3_benchmark.json",
                rows[algorithm]["v3_profiles"][0]["evidence_sources"],
            )

    def test_overview_is_evidence_backed_and_advisory(self):
        payload = asyncio.run(v3_overview())
        self.assertEqual(payload["version"], "3.2.0")
        self.assertFalse(payload["production_authority"])
        self.assertEqual(len(payload["algorithms"]), 12)
        self.assertEqual(len(BUSINESS_CAPABILITIES), 12)
        self.assertTrue(all(item["depth"]["code_evidence"] for item in payload["capabilities"]))
        coverage = payload["business_domain_coverage"]
        self.assertEqual(coverage["domain_count"], 12)
        self.assertEqual(coverage["runtime_output_available_count"], 9)
        self.assertEqual(coverage["no_independent_optimizer_count"], 3)
        self.assertTrue(coverage["all_code_artifacts_hash_verified"])
        self.assertEqual(coverage["production_ready_count"], 0)
        self.assertEqual(len(payload["deployment_gates"]), 5)
        self.assertEqual(payload["dataset"]["validation_rows"], 1755)
        self.assertIn("business_impact", payload)
        self.assertIn("strong_baselines", payload)
        self.assertFalse(payload["strong_baselines"]["strong_baseline_gate"]["production_claim_admitted"])
        if payload["business_impact"] is not None:
            self.assertEqual(payload["business_impact"]["comparison"]["baseline"], "fcfs")
            self.assertEqual(payload["business_impact"]["comparison"]["environment_version"], "port_ops_v3")
            self.assertTrue(payload["business_impact"]["scenario_value"]["annualized_values_are_mechanical_extrapolations"])

    def test_data_readiness_requires_site_replacement(self):
        payload = asyncio.run(v3_data_readiness())
        self.assertTrue(payload["fail_closed"])
        self.assertEqual(len(payload["ports"]), 3)
        self.assertTrue(all(len(port["dataset_sha256"]) == 64 for port in payload["ports"]))
        self.assertGreaterEqual(len(payload["mandatory_site_replacements"]), 7)

    def test_v3_routes_serve_without_frontend_fabrication(self):
        client = TestClient(server.app)
        page = client.get("/v3")
        self.assertEqual(page.status_code, 200)
        self.assertIn("无生产控制权", page.text)
        overview = client.get("/api/v3/overview")
        self.assertEqual(overview.status_code, 200)
        self.assertFalse(overview.json()["production_authority"])
        algorithm = client.get("/api/v3/algorithms/sac/evidence")
        self.assertEqual(algorithm.status_code, 200)
        self.assertEqual(algorithm.json()["protocol"]["holdout_episodes"], 10)
        self.assertGreaterEqual(len(algorithm.json()["v3_profiles"]), 1)
        self.assertIn("training_traces", algorithm.json())
        self.assertTrue(algorithm.json()["historical_evidence"]["append_only"])
        self.assertGreaterEqual(
            len(algorithm.json()["historical_evidence"]["runs"]),
            algorithm.json()["formal_runs"],
        )
        capability = client.get("/api/v3/capabilities/energy")
        self.assertEqual(capability.status_code, 200)
        self.assertIn("SOC 15%–90%", capability.json()["depth"]["hard_constraints"])
        self.assertTrue(capability.json()["fail_closed"])
        script = client.get("/v3/assets/v3.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("查看训练指标", script.text)
        self.assertIn("绝对业务结果", page.text)
        self.assertIn("安全稳健性", page.text)
        self.assertIn("强基线对照", page.text)
        self.assertIn("孪生可靠性", page.text)
        self.assertIn("部署自检", page.text)
        self.assertIn("/api/v3/twin/reliability?refresh=1", script.text)
        self.assertIn("/health/ready", script.text)
        self.assertIn("restoreHashTarget", script.text)
        impact_source = script.text[script.text.index("function renderImpact"):]
        self.assertLess(
            impact_source.index("classList.toggle"),
            impact_source.index("if(!overviewData) return"),
        )

    def test_readiness_exposes_fail_closed_site_and_security_gates(self):
        payload = TestClient(server.app).get("/health/ready").json()
        self.assertTrue(payload["open_source_runtime_ready"])
        self.assertFalse(payload["production_site_ready"])
        for name in (
            "api_rate_limit", "request_body_limit", "security_headers",
            "twin_graph", "site_calibration", "shadow_acceptance",
            "site_evidence_consistency",
        ):
            self.assertIn(name, payload["checks"])
        self.assertFalse(payload["checks"]["site_evidence_consistency"]["ok"])

    def test_business_domains_disclose_execution_depth_and_site_blockers(self):
        payload = asyncio.run(v3_overview())
        rows = {row["id"]: row["depth"] for row in payload["capabilities"]}
        self.assertEqual(set(rows), {row["id"] for row in BUSINESS_CAPABILITIES})
        for domain, depth in rows.items():
            self.assertTrue(depth["implementation_level"], domain)
            self.assertTrue(depth["implementation_label"], domain)
            self.assertTrue(depth["decision_source"], domain)
            self.assertTrue(depth["runtime_endpoints"], domain)
            self.assertTrue(depth["site_blockers"], domain)
            self.assertTrue(depth["fail_closed_fallback"], domain)
            self.assertFalse(depth["production_ready"], domain)
            self.assertTrue(all(row["exists"] for row in depth["code_artifacts"]), domain)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in depth["code_artifacts"]), domain)
        for domain in ("gate", "reefer", "maintenance"):
            self.assertFalse(rows[domain]["model_output_available"])
        ui = (ROOT / "app/ui/v3/v3.js").read_text(encoding="utf-8")
        self.assertIn("执行状态与真实输出来源", ui)
        self.assertIn("无独立模型输出", ui)

    def test_v32_strong_baselines_are_paired_and_fail_closed(self):
        payload = json.loads(
            (ROOT / "evidence/v3/strong_baseline_evidence_v3.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["protocol"]["split"], "chronological_blind_test_only")
        self.assertTrue(payload["protocol"]["paired_comparison"])
        self.assertEqual(payload["protocol"]["window_count"], 10)
        self.assertEqual(set(payload["comparisons"]), {"fcfs_neutral", "engineering_ops_rule", "mpc"})
        self.assertTrue(payload["comparisons"]["fcfs_neutral"]["strict_advantage_95ci"])
        self.assertFalse(payload["comparisons"]["engineering_ops_rule"]["strict_advantage_95ci"])
        self.assertFalse(payload["comparisons"]["mpc"]["strict_advantage_95ci"])
        gate = payload["strong_baseline_gate"]
        self.assertFalse(gate["all_comparators_strictly_beaten"])
        self.assertFalse(gate["measured_current_operations_baseline_available"])
        self.assertFalse(gate["production_claim_admitted"])
        self.assertTrue(payload["site_replacement"]["required"])

    def test_runtime_routes_use_hash_verified_policy_and_expose_site_boundaries(self):
        client = TestClient(server.app)
        status = client.get("/api/v3/runtime/status").json()
        self.assertTrue(status["available"])
        self.assertEqual(status["inference"], "deterministic_saved_policy")
        self.assertEqual(status["model"]["algorithm"], "sac")
        self.assertEqual(len(status["model"]["model_sha256"]), 64)
        self.assertFalse(status["telemetry"]["measured"])
        self.assertFalse(status["production_authority"])

        strategy = client.get(
            "/api/v3/runtime/series?scenario=strategy&horizon_min=120&step_min=10"
        ).json()
        self.assertTrue(strategy["available"])
        self.assertEqual(len(strategy["series"]["p50"]), 12)
        self.assertGreaterEqual(len(strategy["actions"]), 2)
        self.assertEqual(strategy["policy"]["model_sha256"], status["model"]["model_sha256"])
        projection = strategy["summary"]["business_projection"]
        self.assertEqual(projection["schema"], "port-dt-v3-online-open-loop-business-projection.v1")
        self.assertGreater(projection["improvement_percent"]["throughput_teu"], 0)
        self.assertGreater(projection["improvement_percent"]["delay_index_mean"], 0)
        equivalent_value = projection["equivalent_throughput_value"]
        self.assertEqual(
            equivalent_value["comparison_basis"],
            "baseline_scaled_to_policy_throughput",
        )
        self.assertGreater(equivalent_value["avoided_energy_cost"], 0)
        self.assertGreater(equivalent_value["avoided_carbon_kg"], 0)
        self.assertFalse(equivalent_value["financial_audit_ready"])
        self.assertEqual(
            equivalent_value["site_tariff_contract"],
            "pending_port_connection",
        )

        frame = client.get("/api/v3/runtime/frame").json()
        self.assertTrue(frame["available"])
        self.assertFalse(frame["telemetry"]["measured"])
        self.assertIn("source_timestamp", frame["public_conditions"])
        self.assertIn("ambient_c", frame["public_conditions"])
        self.assertIn("wave_height_m", frame["public_conditions"])

        coverage = client.get("/api/v3/runtime/coverage").json()
        self.assertEqual(coverage["total"], 10)
        self.assertEqual(coverage["runtime_covered"], 9)
        self.assertTrue(any(row["state"] == "contract_only" for row in coverage["scenarios"]))
        self.assertIn("not proof", coverage["claim_boundary"])

    def test_realtime_insights_use_active_stream_and_model_outputs(self):
        client = TestClient(server.app)
        current = client.get(
            "/api/v3/realtime/insights?asset_id=qc-01&mode=now"
            "&cap_kw=36000&horizon_min=60&step_min=5"
        ).json()
        self.assertTrue(current["available"])
        self.assertEqual(current["schema"], "port-dt-v3-realtime-insights.v1")
        self.assertFalse(current["telemetry"]["measured"])
        self.assertEqual(
            current["telemetry"]["mode"],
            "calibrated_public_replay_simulator",
        )
        self.assertGreaterEqual(current["quality"]["sample_count"], 2)
        self.assertIsNotNone(current["quality"]["missing_rate"])
        self.assertEqual(
            current["quality"]["site_sensor_quality"],
            "pending_port_connection",
        )
        self.assertTrue(current["forecast"]["available"])
        self.assertEqual(current["forecast"]["model"], "ridge_autoregression")
        self.assertEqual(current["forecast"]["point_count"], 12)
        self.assertFalse(current["forecast"]["calibration"]["available"])
        self.assertIn("no in-sample", current["forecast"]["calibration"]["reason"])
        self.assertTrue(current["peak_risk"]["available"])
        self.assertIn("normal", current["peak_risk"]["logic"])
        self.assertFalse(current["business_value"]["available"])
        self.assertFalse(current["approvals"]["available"])
        self.assertFalse(current["production_authority"])

        simulation = client.get(
            "/api/v3/realtime/insights?asset_id=qc-01&mode=sim"
            "&cap_kw=36000&horizon_min=60&step_min=5"
        ).json()
        self.assertTrue(simulation["business_value"]["available"])
        self.assertGreater(
            simulation["business_value"]["avoided_energy_cost_cny"], 0
        )
        self.assertGreater(simulation["business_value"]["avoided_carbon_kg"], 0)
        self.assertFalse(simulation["business_value"]["financial_audit_ready"])

        missing = client.get(
            "/api/v3/realtime/insights?asset_id=unknown&mode=now"
        ).json()
        self.assertFalse(missing["available"])
        self.assertFalse(missing["production_authority"])

    def test_twin_reliability_separates_software_stress_from_site_fidelity(self):
        client = TestClient(server.app)
        payload = client.get(
            "/api/v3/twin/reliability?refresh=1&scenario=typhoon_closure"
        ).json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["schema"], "port-dt-v3-twin-reliability.v1")
        self.assertFalse(payload["site_fidelity"]["available"])
        self.assertEqual(
            payload["site_fidelity"]["status"], "pending_port_connection"
        )
        self.assertFalse(payload["forecast_interval_calibration"]["available"])
        self.assertFalse(payload["site_error_decomposition"]["available"])
        self.assertEqual(payload["software_coverage"]["covered"], 9)
        self.assertEqual(payload["software_coverage"]["total"], 10)
        self.assertEqual(payload["software_stress"]["total"], 7)
        self.assertEqual(payload["software_stress"]["passed"], 7)
        self.assertGreaterEqual(
            sum(
                row["fail_closed_triggered"]
                for row in payload["software_stress"]["runs"]
            ),
            1,
        )
        self.assertTrue(
            all(
                row["safe_action_count"] == row["decision_count"]
                for row in payload["software_stress"]["runs"]
            )
        )
        self.assertTrue(all(row["recommendation_only"] for row in payload["software_stress"]["runs"]))
        self.assertTrue(payload["selected_replay"]["passed"])
        self.assertEqual(payload["selected_replay"]["violation_count"], 0)
        self.assertEqual(payload["selected_replay"]["side_effect"], "none")
        self.assertTrue(payload["runtime_envelope"]["available"])
        self.assertFalse(payload["production_authority"])

        pending = client.get(
            "/api/v3/twin/reliability?scenario=cyber_or_actuator_fault"
        ).json()
        self.assertEqual(
            pending["selected_replay"]["status"], "pending_port_connection"
        )
        self.assertIsNone(pending["selected_replay"]["passed"])
        self.assertEqual(pending["selected_replay"]["side_effect"], "none")

    def test_strategy_list_surfaces_controller_coverage_without_duplicate_runs(self):
        client = TestClient(server.app)
        payload = client.get("/api/rl/strategies?max_items=12").json()
        algorithms = payload["algorithm_coverage"]
        self.assertEqual(len(algorithms), len(set(algorithms)))
        self.assertEqual(algorithms[0], "sac")
        self.assertEqual(algorithms[-1], "fcfs")
        self.assertEqual(payload["selection_basis"], "newest_evaluated_record_per_algorithm_in_controller_order")
        self.assertFalse(payload["generated_values"])
        self.assertIn("cost_per_teu", payload["strategies"][0]["impact"])

    def test_benchmark_summary_keeps_multi_seed_tail_risk_evidence(self):
        payload = TestClient(server.app).get(
            "/api/rl/benchmarks/summary?dataset_id=public_cn_sha_hourly_v3"
        ).json()
        sac = next(row for row in payload["algorithms"] if row["id"] == "sac")
        self.assertTrue(sac["multi_seed_ready"])
        self.assertGreaterEqual(len(sac["distinct_seeds"]), 3)
        self.assertEqual(sac["tail_risk"]["alpha"], 0.95)
        self.assertEqual(sac["tail_risk"]["metric"], "energy_cost")
        self.assertGreater(sac["tail_risk"]["cvar"], 0)

    def test_rlops_signals_select_persisted_trainable_optimizer_history(self):
        payload = TestClient(server.app).get("/api/rlops/signals").json()
        self.assertTrue(payload["available"])
        self.assertTrue(payload["optimizer_history_available"])
        self.assertNotIn(payload["job"]["algorithm"], {"mpc", "fcfs"})
        self.assertGreater(len(payload["history"]), 0)
        self.assertEqual(
            payload["selection_basis"],
            "latest_registered_trainable_run_with_persisted_optimizer_history",
        )
        ppo = TestClient(server.app).get("/api/rlops/signals?algorithm=ppo").json()
        self.assertEqual(ppo["requested_algorithm"], "ppo")
        self.assertEqual(ppo["job"]["algorithm"], "ppo")
        self.assertTrue(ppo["optimizer_history_available"])
        self.assertIn("ppo", ppo["available_algorithms"])

    def test_multi_agent_view_uses_blind_test_state_and_verified_sac_inference(self):
        client = TestClient(server.app)
        payload = client.get("/api/v3/mas/evidence?scenario=dense").json()
        self.assertTrue(payload["available"])
        self.assertEqual(
            payload["mode"],
            "public_data_calibrated_replay_plus_model_inference",
        )
        self.assertEqual(payload["scenario"]["id"], "dense")
        self.assertEqual(payload["decision"]["algorithm"], "sac")
        self.assertEqual(payload["decision"]["implementation"], "stable_baselines3.SAC")
        self.assertEqual(payload["decision"]["safety_envelope"]["status"], "pass")
        self.assertFalse(payload["decision"]["rendered"])
        self.assertFalse(payload["production_authority"])
        self.assertGreaterEqual(payload["evidence"]["formal_seed_count"], 3)
        self.assertEqual(len(payload["evidence"]["dataset_sha256"]), 64)
        self.assertEqual(payload["site_replacement"]["status"], "pending_port_connection")
        self.assertGreater(len(payload["graph"]["nodes"]), 5)
        self.assertGreater(len(payload["timeline"]["items"]), 3)

    def test_twinlab_exposes_bounded_stress_drills_and_real_factor_contracts(self):
        payload = TestClient(server.app).get("/api/v3/twinlab/evidence").json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["schema"], "port-dt-v3-twinlab-evidence.v1")
        self.assertEqual(payload["scenarios"]["passed"], 7)
        self.assertEqual(payload["scenarios"]["total"], 7)
        self.assertEqual(len(payload["scenarios"]["items"]), 7)
        self.assertGreater(payload["scenarios"]["distribution"]["safe_actions"], 0)
        self.assertGreater(payload["drills"]["fail_closed_covered"], 0)
        self.assertEqual(payload["drills"]["site_rto_rpo"], "pending_port_connection")
        self.assertEqual(payload["contracts"]["dataset_id"], "public_cn_sha_hourly_v3")
        self.assertEqual(len(payload["contracts"]["dataset_sha256"]), 64)
        visibility = next(
            row for row in payload["contracts"]["items"]
            if row["feature"] == "visibility_km"
        )
        self.assertEqual(visibility["coverage"], 0.0)
        self.assertEqual(visibility["status"], "PENDING_PORT")
        self.assertFalse(payload["production_authority"])

    def test_ars_progress_is_shown_without_fabricated_reward_curve(self):
        payload = TestClient(server.app).get("/api/v3/algorithms/ars/evidence").json()
        traces = list((payload.get("training_traces") or {}).values())
        self.assertTrue(traces)
        self.assertTrue(any(trace["reward_available"] is False for trace in traces))
        self.assertTrue(all("progress" in point for trace in traces for point in trace["points"]))

    def test_story_replay_pairs_selected_sac_with_aligned_fcfs_evidence(self):
        client = TestClient(server.app)
        payload = client.get(
            "/api/story/summary?hour=0&port=shanghai&scenario=sac_vs_fcfs"
        ).json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["schema"], "port-dt-v3-story-evidence.v1")
        self.assertEqual(payload["evidence"]["policy_algorithm"], "sac")
        self.assertEqual(payload["evidence"]["baseline_algorithm"], "fcfs")
        self.assertEqual(payload["evidence"]["aligned_frames"], 48)
        self.assertEqual(payload["evidence"]["split"], "chronological_blind_test_only")
        self.assertEqual(len(payload["events"]), 3)
        self.assertGreater(
            payload["blind_test_summary"]["improvement_percent"]["throughput_teu"],
            0,
        )
        self.assertGreater(
            payload["blind_test_summary"]["improvement_percent"]["delay_index_mean"],
            0,
        )
        self.assertFalse(payload["production_authority"])

        pending = client.get(
            "/api/story/summary?hour=0&port=ningbo&scenario=sac_vs_fcfs"
        ).json()
        self.assertFalse(pending["available"])
        self.assertEqual(pending["status"], "pending_port_connection")
        self.assertIn("待接入港口", pending["reason"])

        ack = client.post("/api/story/play", json={"mode": "demo"}).json()
        self.assertTrue(ack["ok"])
        self.assertEqual(ack["mode"], "heldout_evidence_replay")
        self.assertEqual(ack["side_effect"], "none")

    def test_kpi_dashboard_exposes_simulator_basis_and_real_carbon_factor(self):
        client = TestClient(server.app)
        energy = client.get("/api/energy/today?teu=24000").json()
        self.assertTrue(energy["available"])
        self.assertFalse(energy["data_status"]["measured"])
        self.assertEqual(
            energy["data_status"]["mode"],
            "calibrated_public_replay_simulator",
        )
        self.assertEqual(
            energy["electricity"]["carbon_factor_source"],
            "calibrated_port_state",
        )
        self.assertGreater(energy["electricity"]["avg_carbon_intensity_g_per_kwh"], 0)
        self.assertGreater(energy["intensity"]["kgCO2e_per_TEU"], 0)
        self.assertEqual(
            energy["assumptions"]["pue_status"],
            "pending_port_cooling_and_it_meter_connection",
        )

        alerts = client.get(
            "/api/alerts/scan?teu=24000&demand_limit_kw=36000&quota_kgco2e=5000"
        ).json()
        self.assertTrue(alerts["summary"]["carbon_assessment_available"])
        self.assertGreater(alerts["summary"]["total_kgco2e_est"], 0)
        self.assertFalse(alerts["summary"]["production_alarm_authority"])

    def test_yard_lighting_v31_public_linkage_formal_training_and_legacy_preservation(self):
        self.assertFalse(lighting_engine_bool("False"))
        self.assertFalse(lighting_engine_bool("0"))
        self.assertFalse(lighting_api_bool("false"))
        self.assertTrue(lighting_api_bool("1"))

        payload = TestClient(server.app).get(
            "/api/v3/modules/yard-lighting/evidence"
        ).json()
        self.assertEqual(payload["version"], "V3.1")
        self.assertTrue(payload["historical_evidence"]["preserved"])
        self.assertGreater(payload["historical_evidence"]["records"], 0)
        self.assertEqual(len(payload["historical_evidence"]["history_sha256"]), 64)
        self.assertEqual(len(payload["historical_evidence"]["policy_sha256"]), 64)
        self.assertFalse(payload["boundary"]["claim_eligible"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        probe = payload["model_probe"]
        self.assertTrue(probe["policy_loaded"])
        self.assertFalse(probe["policy_admitted"])
        self.assertEqual(probe["decision_source"], "rule_fallback")
        self.assertGreater(probe["normalized_abs_max"], probe["ood_threshold"])
        formal = payload["formal_training"]
        self.assertEqual(formal["status"], "passed")
        self.assertEqual(formal["dataset"]["rows"], 2783)
        self.assertEqual(formal["dataset"]["raw_lighting_rows"], 267168)
        self.assertEqual(formal["dataset"]["raw_activity_rows"], 953856)
        self.assertEqual(formal["dataset"]["zones"], 96)
        self.assertEqual(formal["dataset"]["public_source_observations"], 17566)
        self.assertEqual(formal["dataset"]["train_rows"], 1948)
        self.assertEqual(formal["dataset"]["validation_rows"], 278)
        self.assertEqual(formal["dataset"]["blind_test_rows"], 557)
        self.assertEqual(formal["contract"]["state_dimensions"], 42)
        self.assertEqual(formal["contract"]["action_dimensions"], 3)
        self.assertTrue(formal["convergence"]["passed"])
        self.assertFalse(formal["convergence"]["blind_test_used_for_selection"])
        self.assertFalse(formal["blind_test_protocol"]["selection_access"])
        current = payload["current_model_output"]["model_inference"]
        self.assertTrue(current["policy_loaded"])
        self.assertTrue(current["policy_admitted_for_engineering_replay"])
        self.assertFalse(current["production_admitted"])
        self.assertEqual(len(current["observation_vector"]), 42)
        self.assertEqual(set(current["final_action"]), {
            "base_dimming_residual_ratio", "activity_gain_ratio", "weather_gain_ratio",
        })
        gates = payload["quality_gates"]
        self.assertTrue(gates["convergence_passed"])
        self.assertTrue(gates["business_advantage_passed"])
        self.assertTrue(gates["minimum_lux_passed"])
        self.assertTrue(gates["critical_lux_passed"])
        self.assertTrue(gates["safety_passed"])
        self.assertTrue(gates["legacy_policy_ood_blocked"])
        self.assertFalse(gates["production_admitted"])
        business = payload["business_metrics"]
        self.assertGreater(business["cost_reduction_vs_historical_control_percent"], 1.0)
        self.assertGreater(business["energy_reduction_vs_historical_control_percent"], 2.0)
        self.assertGreater(business["peak_reduction_vs_historical_control_percent"], 1.0)
        self.assertGreater(business["carbon_reduction_vs_historical_control_percent"], 2.0)
        self.assertEqual(business["minimum_lux_compliance_rate_percent"], 100.0)
        self.assertEqual(business["under_lux_zone_steps"], 0.0)
        self.assertFalse(business["claim_eligible"])
        self.assertEqual(payload["data_manifest"]["public_linkage"]["dataset_id"], "public_cn_sha_hourly_v3")
        self.assertGreater(len(payload["algorithm_registry"]), 3)
        self.assertGreater(len(payload["history_series"]), 5)
        self.assertGreater(len(payload["legacy_history_series"]), 5)

    def test_hvac_v31_preserves_legacy_and_loads_formal_selected_actor(self):
        payload = TestClient(server.app).get("/api/v3/modules/hvac/evidence").json()
        self.assertEqual(payload["version"], "V3.1")
        self.assertFalse(payload["boundary"]["claim_eligible"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        self.assertTrue(payload["historical_evidence"]["preserved"])
        self.assertEqual(payload["historical_evidence"]["records"], 4003)
        self.assertEqual(len(payload["historical_evidence"]["history_sha256"]), 64)
        self.assertEqual(len(payload["historical_evidence"]["policy_sha256"]), 64)
        formal = payload["formal_training"]
        self.assertEqual(formal["dataset"]["train_rows"], 4032)
        self.assertEqual(formal["dataset"]["validation_rows"], 576)
        self.assertEqual(formal["dataset"]["blind_test_rows"], 1152)
        self.assertEqual(formal["contract"]["state_dimensions"], 30)
        self.assertEqual(formal["contract"]["action_dimensions"], 3)
        self.assertTrue(formal["convergence"]["passed"])
        self.assertFalse(formal["convergence"]["blind_test_used_for_selection"])
        inference = payload["current_model_output"]["model_inference"]
        self.assertTrue(inference["policy_loaded"])
        self.assertTrue(inference["policy_admitted_for_engineering_replay"])
        self.assertIn("chws_c", inference["final_action"])
        self.assertFalse(inference["production_admitted"])
        gates = payload["quality_gates"]
        self.assertTrue(gates["convergence_passed"])
        self.assertTrue(gates["business_advantage_passed"])
        self.assertTrue(gates["cooling_service_passed"])
        self.assertTrue(gates["safety_passed"])
        self.assertFalse(gates["production_admitted"])
        business = payload["business_metrics"]
        self.assertGreater(business["cost_reduction_vs_historical_control_percent"], 1.0)
        self.assertGreater(business["energy_reduction_vs_historical_control_percent"], 1.0)
        self.assertGreater(business["peak_reduction_vs_historical_control_percent"], 0.5)
        self.assertEqual(business["cooling_satisfaction_rate_percent"], 100.0)
        self.assertFalse(business["claim_eligible"])
        self.assertGreater(len(payload["history_series"]), 5)
        self.assertGreater(len(payload["legacy_history_series"]), 10)

    def test_shore_bess_v31_preserves_legacy_and_loads_formal_selected_actor(self):
        payload = TestClient(server.app).get(
            "/api/v3/modules/shore-bess/evidence"
        ).json()
        self.assertEqual(payload["version"], "V3.1")
        self.assertFalse(payload["boundary"]["claim_eligible"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        self.assertTrue(payload["historical_evidence"]["preserved"])
        self.assertEqual(payload["historical_evidence"]["records"], 2293)
        self.assertEqual(len(payload["historical_evidence"]["history_sha256"]), 64)
        self.assertEqual(len(payload["historical_evidence"]["policy_sha256"]), 64)
        gates = payload["quality_gates"]
        self.assertTrue(gates["policy_artifact_loads"])
        self.assertTrue(gates["convergence_passed"])
        self.assertTrue(gates["business_advantage_passed"])
        self.assertTrue(gates["safety_passed"])
        self.assertFalse(gates["carbon_guardrail_passed"])
        self.assertTrue(gates["admitted"])
        self.assertFalse(gates["production_admitted"])
        legacy = gates["legacy_audit"]
        self.assertEqual(legacy["offline_rows"], 145)
        self.assertEqual(legacy["nonzero_action_rows"], 0)
        self.assertEqual(legacy["shore_positive_rows"], 0)
        self.assertFalse(legacy["legacy_policy_admitted"])
        model = payload["current_model_output"]["model_inference"]
        self.assertTrue(model["policy_loaded"])
        self.assertTrue(model["policy_admitted_for_public_offline"])
        self.assertFalse(model["production_admitted"])
        self.assertEqual(
            model["decision_source"],
            "selected_constrained_actor_plus_safety_projection",
        )
        self.assertIn("bess_kw", model["final_action"])
        formal = payload["formal_training"]
        self.assertEqual(formal["dataset"]["rows"], 17544)
        self.assertEqual(formal["dataset"]["train_rows"], 12280)
        self.assertEqual(formal["dataset"]["validation_rows"], 1755)
        self.assertEqual(formal["dataset"]["blind_test_rows"], 3509)
        self.assertEqual(formal["contract"]["state_dimensions"], 34)
        self.assertEqual(formal["contract"]["action_dimensions"], 2)
        self.assertEqual(len(formal["contract"]["reward_components"]), 8)
        self.assertGreaterEqual(len(formal["contract"]["hard_constraints"]), 10)
        business = payload["business_metrics"]
        self.assertGreater(business["cost_reduction_vs_no_bess_percent"], 0)
        self.assertGreater(business["peak_reduction_vs_no_bess_percent"], 0)
        self.assertLess(business["carbon_reduction_vs_no_bess_percent"], 0)
        self.assertLess(
            payload["historical_business_replay"]["strategy_advantage_yuan"],
            0,
        )
        self.assertGreater(len(payload["algorithm_registry"]), 3)
        self.assertGreater(len(payload["history_series"]), 5)
        self.assertGreater(len(payload["legacy_history_series"]), 10)

    def test_shore_bess_v31_environment_contract_and_hard_projection(self):
        dataset = load_port_dataset("public_cn_sha_hourly_v3")
        train_slice, _validation_slice, _blind_slice = shore_bess_slices(dataset)
        env = ShoreBESSEnv(
            dataset,
            train_slice,
            config=load_shore_bess_config(),
            normalization_slice=train_slice,
            episode_steps=168,
            seed=7,
            training=True,
        )
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, (len(SHORE_BESS_STATES),))
        self.assertEqual(len(SHORE_BESS_STATES), 34)
        self.assertEqual(len(SHORE_BESS_ACTIONS), 2)
        self.assertEqual(SHORE_BESS_CONTRACT.as_dict()["reward_components"], [
            "energy_cost", "carbon", "demand_peak", "degradation",
            "reserve", "shore_sla", "safety_projection", "terminal_state",
        ])
        done = False
        while not done:
            observation, _reward, terminated, truncated, _info = env.step(
                np.asarray([1.0, -1.0], dtype=np.float32)
            )
            done = terminated or truncated
        totals = env.totals
        self.assertEqual(totals["guardrail_violation_rate"], 0.0)
        self.assertAlmostEqual(totals["terminal_soc_error"], 0.0, places=8)
        self.assertAlmostEqual(totals["terminal_flex_backlog_kwh"], 0.0, places=8)
        self.assertEqual(totals["shore_sla_violation_kwh"], 0.0)

    def test_bess_energy_v31_accepts_formal_policy_and_preserves_rejected_legacy(self):
        payload = TestClient(server.app).get(
            "/api/v3/modules/bess-energy/evidence"
        ).json()
        self.assertEqual(payload["version"], "V3.1")
        self.assertFalse(payload["boundary"]["claim_eligible"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        history = payload["historical_evidence"]
        self.assertTrue(history["preserved"])
        self.assertEqual(history["records"], 2000)
        self.assertEqual(history["offline_rows"], 8927)
        self.assertEqual(len(history["history_sha256"]), 64)
        self.assertEqual(len(history["offline_sha256"]), 64)
        gates = payload["quality_gates"]
        self.assertTrue(gates["policy_artifact_loads"])
        self.assertTrue(gates["convergence_passed"])
        self.assertTrue(gates["business_advantage_passed"])
        self.assertTrue(gates["carbon_guardrail_passed"])
        self.assertTrue(gates["safety_passed"])
        self.assertTrue(gates["admitted"])
        self.assertFalse(gates["production_admitted"])
        self.assertGreater(gates["event_coverage_hours"], 0)
        self.assertEqual(gates["event_compliance_rate"], 1.0)
        legacy = gates["legacy_audit"]
        self.assertGreater(legacy["nonzero_dP_rows"], 0)
        self.assertEqual(legacy["nonzero_dR_rows"], 0)
        self.assertEqual(legacy["event_active_rows"], 0)
        self.assertEqual(legacy["heldout_evaluation_rows"], 0)
        self.assertGreater(legacy["sampled_policy_saturation_rate"], 0.99)
        model = payload["current_model_output"]["model_inference"]
        self.assertTrue(model["policy_loaded"])
        self.assertTrue(model["policy_admitted_for_public_offline"])
        self.assertFalse(model["production_admitted"])
        self.assertEqual(model["decision_source"], "selected_event_aware_actor_plus_cmdp_safety_projection")
        self.assertEqual(len(model["observation_vector"]), 40)
        formal = payload["formal_training"]
        self.assertEqual(formal["dataset"]["rows"], 17544)
        self.assertEqual(formal["dataset"]["train_rows"], 12280)
        self.assertEqual(formal["dataset"]["validation_rows"], 1755)
        self.assertEqual(formal["dataset"]["blind_test_rows"], 3509)
        self.assertEqual(formal["training"]["training_render_calls"], 0)
        self.assertEqual(formal["contract"]["state_dimensions"], 40)
        self.assertEqual(formal["contract"]["action_dimensions"], 2)
        self.assertEqual(len(formal["contract"]["reward_components"]), 9)
        self.assertGreaterEqual(len(formal["contract"]["hard_constraints"]), 14)
        self.assertEqual(formal["scenario_supplement"]["observed_site_event_rows"], 0)
        self.assertFalse(formal["scenario_supplement"]["claim_as_real_market_settlement"])
        business = payload["business_metrics"]
        self.assertGreater(business["cost_reduction_vs_no_bess_percent"], 0)
        self.assertGreaterEqual(business["peak_reduction_vs_no_bess_percent"], 0)
        self.assertGreaterEqual(business["carbon_reduction_vs_no_bess_percent"], 0)
        self.assertEqual(business["event_compliance_rate_percent"], 100.0)
        self.assertEqual(payload["excluded_static_card"]["status"], "excluded_from_v3_evidence")
        self.assertGreater(len(payload["algorithm_registry"]), 3)
        self.assertGreater(len(payload["history_series"]), 5)
        self.assertGreater(len(payload["legacy_history_series"]), 10)

    def test_bess_energy_v31_projection_survives_extreme_actions(self):
        from app.services.rl_model.bess_energy.v3_environment import (
            BESSEnergyV3Env,
            chronological_slices,
            load_config,
            load_public_dataset,
        )

        config = load_config()
        dataset = load_public_dataset(config)
        train_slice, validation_slice, _blind_slice = chronological_slices(dataset)
        env = BESSEnergyV3Env(
            dataset,
            validation_slice,
            config=config,
            normalization_slice=train_slice,
            episode_steps=168,
            training=False,
        )
        observation, _ = env.reset(options={"start_index": 0})
        self.assertEqual(observation.shape, (40,))
        done = False
        step = 0
        while not done:
            action = np.asarray([1.0, 1.0] if step % 2 == 0 else [-1.0, 1.0], dtype=np.float32)
            observation, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            step += 1
        totals = env.totals
        self.assertEqual(totals["guardrail_violation_rate"], 0.0)
        self.assertAlmostEqual(totals["terminal_soc_error"], 0.0, places=8)
        self.assertEqual(totals["event_compliance_rate"], 1.0)

    def test_yard_crane_v31_trains_blind_tests_and_preserves_failed_legacy_history(self):
        payload = TestClient(server.app).get(
            "/api/v3/modules/yard-crane/evidence"
        ).json()
        self.assertEqual(payload["version"], "V3.1")
        self.assertFalse(payload["boundary"]["claim_eligible"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        history = payload["historical_evidence"]
        self.assertTrue(history["preserved"])
        self.assertEqual(history["records"], 1001)
        self.assertEqual(history["step_records"], 1000)
        self.assertEqual(len(history["history_sha256"]), 64)
        self.assertEqual(len(history["offline_sha256"]), 64)
        gates = payload["quality_gates"]
        self.assertTrue(gates["dataset_quality_passed"])
        self.assertTrue(gates["convergence_passed"])
        self.assertEqual(gates["seed_pass_rate"], 1.0)
        self.assertTrue(gates["business_advantage_passed"])
        self.assertTrue(gates["moves_service_passed"])
        self.assertTrue(gates["job_sla_passed"])
        self.assertEqual(gates["guardrail_violation_rate"], 0.0)
        self.assertTrue(gates["policy_artifact_loads"])
        self.assertTrue(gates["admitted"])
        self.assertFalse(gates["production_admitted"])
        legacy = gates["legacy_audit"]
        self.assertEqual(legacy["policy_artifact_bytes"], 0)
        self.assertEqual(legacy["base_offline_rows"], 17278)
        self.assertEqual(legacy["base_nonzero_action_rows"], 0)
        self.assertEqual(legacy["augmented_rows"], 144)
        self.assertEqual(legacy["augmented_nonzero_action_rows"], 144)
        self.assertEqual(legacy["historical_job_positive_rows"], 0)
        self.assertEqual(legacy["historical_thermal_available_rows"], 0)
        self.assertEqual(legacy["heldout_evaluation_rows"], 0)
        self.assertGreater(legacy["historical_mask_rate"], 0.8)
        current = payload["current_model_output"]["model_inference"]
        self.assertTrue(current["policy_loaded"])
        self.assertEqual(current["decision_source"], "hash_verified_selected_safe_actor_plus_cmdp_projection")
        self.assertEqual(len(current["observation_vector"]), 36)
        self.assertEqual(set(current["final_action"]), {"power_cap_residual_pct", "idle_timeout_residual_min"})
        formal = payload["formal_training"]
        self.assertEqual(formal["status"], "passed")
        self.assertEqual(formal["dataset"]["rows"], 5760)
        self.assertEqual(formal["dataset"]["raw_crane_telemetry_rows"], 92160)
        self.assertEqual(formal["dataset"]["tos_job_rows"], 8559)
        self.assertEqual(formal["dataset"]["queue_forecast_rows"], 69120)
        self.assertEqual(formal["dataset"]["cranes"], 16)
        self.assertEqual(formal["dataset"]["yard_blocks"], 12)
        self.assertEqual(formal["dataset"]["train_rows"], 4032)
        self.assertEqual(formal["dataset"]["validation_rows"], 576)
        self.assertEqual(formal["dataset"]["blind_test_rows"], 1152)
        self.assertFalse(formal["blind_test_protocol"]["selection_access"])
        self.assertEqual(formal["blind_test_protocol"]["windows"], 8)
        self.assertEqual([run["status"] for run in formal["run_history"]], ["failed", "passed"])
        metrics = payload["business_metrics"]
        self.assertGreater(metrics["cost_reduction_vs_historical_control_percent"], 3.0)
        self.assertGreater(metrics["energy_reduction_vs_historical_control_percent"], 3.0)
        self.assertGreater(metrics["peak_reduction_vs_historical_control_percent"], 0.5)
        self.assertGreater(metrics["carbon_reduction_vs_historical_control_percent"], 3.0)
        self.assertEqual(metrics["historical_moves_retention_rate_percent"], 100.0)
        self.assertEqual(metrics["job_sla_non_degradation_rate_percent"], 100.0)
        self.assertEqual(metrics["delay_delta_minutes"], 0.0)
        self.assertFalse(metrics["claim_eligible"])
        diagnostics = payload["historical_training_diagnostics"]
        self.assertGreater(
            diagnostics["sla_penalty_yuan"],
            diagnostics["economic_advantage_yuan"],
        )
        self.assertFalse(diagnostics["claim_eligible"])
        self.assertTrue(payload["schema_repairs"]["implemented"])
        self.assertGreater(len(payload["algorithm_registry"]), 4)
        self.assertGreater(len(payload["history_series"]), 5)
        self.assertGreater(len(payload["legacy_history_series"]), 10)

    def test_ai_trust_v3_separates_offline_advantage_from_site_authority(self):
        payload = TestClient(server.app).get("/api/v3/ai-trust/evidence").json()
        self.assertEqual(payload["version"], "V3")
        self.assertEqual(payload["trust_grade"], "B+")
        self.assertTrue(payload["boundary"]["offline_claim_eligible"])
        self.assertFalse(payload["boundary"]["causal_claim_eligible"])
        self.assertFalse(payload["boundary"]["production_authority"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        benchmark = payload["benchmark"]
        self.assertTrue(benchmark["sidecar_sha256_match"])
        self.assertEqual(benchmark["dataset_rows"], 17544)
        self.assertEqual(benchmark["blind_test_rows"], 3509)
        self.assertEqual(len(benchmark["seeds"]), 3)
        self.assertGreater(benchmark["weighted_improvement_percent"], 0)
        self.assertGreater(benchmark["weighted_ci_percent"][0], 0)
        self.assertEqual(len(payload["advantage_metrics"]), 5)
        self.assertEqual(len(payload["controls"]), 6)
        self.assertEqual(len(payload["scenes"]), 6)
        global_scene = payload["scenes"][0]
        self.assertEqual(global_scene["gate_state"], "pass")
        self.assertEqual(global_scene["history_records"], 3)
        module_scenes = payload["scenes"][1:]
        self.assertTrue(all(scene["gate_state"] != "pass" for scene in module_scenes))
        self.assertTrue(all(scene["site_status"] == "待接入港口" for scene in module_scenes))
        self.assertTrue(payload["historical_evidence"]["preserved"])
        self.assertIn("上海港现场已提效", payload["claim_registry"]["prohibited"])

    def test_monitoring_v3_computes_from_labelled_replay_and_blocks_site_claims(self):
        payload = TestClient(server.app).get("/api/v3/monitoring/evidence").json()
        self.assertEqual(payload["version"], "V3")
        self.assertTrue(payload["boundary"]["analysis_available"])
        self.assertFalse(payload["boundary"]["live_data_verified"])
        self.assertFalse(payload["boundary"]["incident_claim_eligible"])
        self.assertFalse(payload["boundary"]["production_authority"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        source = payload["source"]
        self.assertEqual(source["mode"], "calibrated_public_replay_simulator")
        self.assertFalse(source["measured"])
        self.assertEqual(source["rows"], 17544)
        self.assertEqual(len(source["sha256"]), 64)
        analysis = payload["current_analysis"]
        self.assertEqual(analysis["anomaly"]["asset_count"], 3)
        self.assertGreater(analysis["anomaly"]["sample_count"], 300)
        self.assertEqual(analysis["drift"]["baseline"]["n"], 1441)
        self.assertEqual(analysis["drift"]["recent"]["n"], 121)
        self.assertEqual(len(analysis["drift"]["bins"]), 12)
        self.assertFalse(analysis["admission_decision"]["site_command_allowed"])
        self.assertGreater(len(payload["method_registry"]), 5)
        self.assertEqual(payload["alert_policy"]["site_state"], "待接入港口告警中心/CMMS/工单回执")

    def test_calibrated_replay_preserves_non_overlapping_time_windows(self):
        import time

        from app.adapters.telemetry_calibrated_replay import CalibratedReplayTelemetry

        telemetry = CalibratedReplayTelemetry()
        end = time.time()
        baseline = telemetry.get_series("qc-01", "active_power_kw", end - 3600, end - 1800, 60)
        recent = telemetry.get_series("qc-01", "active_power_kw", end - 1800, end, 60)
        self.assertEqual(len(baseline), 31)
        self.assertEqual(len(recent), 31)
        self.assertNotEqual(baseline[0]["source_ts"], recent[0]["source_ts"])
        self.assertNotEqual(
            [round(row["v"], 6) for row in baseline],
            [round(row["v"], 6) for row in recent],
        )

    def test_monitoring_psi_does_not_flatten_rated_quay_crane_power(self):
        import time

        from app.ops.data_quality import clean_and_impute

        end = time.time()
        raw = server.di.telemetry.get_series(
            "qc-01", "active_power_kw", end - 1800, end, 60
        )
        series = [(datetime.fromisoformat(row["ts"]).timestamp(), row["v"]) for row in raw]
        cleaned, quality, _ = clean_and_impute(
            series,
            start=end - 1800,
            end=end,
            step_sec=60,
            asset_type="quay_crane",
            point="active_power_kw",
        )
        values = [value for _, value in cleaned]
        self.assertGreater(max(values), 2000)
        self.assertGreater(max(values) - min(values), 1)
        self.assertGreater(quality["validity"], 0.9)

    def test_opsx_v3_keeps_rollout_at_zero_without_site_control_plane(self):
        payload = TestClient(server.app).get("/api/v3/opsx/evidence").json()
        self.assertEqual(payload["version"], "V3")
        self.assertFalse(payload["boundary"]["live_rollout_verified"])
        self.assertFalse(payload["boundary"]["production_authority"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        rollout = payload["rollout"]
        self.assertEqual(rollout["phase"], "blocked_pre_shadow")
        self.assertEqual(rollout["traffic_percent"], 0)
        self.assertFalse(rollout["mutations_enabled"])
        self.assertEqual(rollout["decision"], "BLOCK")
        self.assertIn("actuator_config", rollout["blockers"])
        self.assertIn("two_person", rollout["blockers"])
        self.assertEqual(len(payload["gates"]), 8)
        self.assertEqual(len(payload["stage_ladder"]), 6)
        self.assertEqual(len(payload["eight_capabilities"]), 8)
        self.assertTrue(payload["audit_manifest"]["all_owner_only"])
        self.assertIn("independent_confirm", payload["security_contract"]["command_flow"])

    def test_object_storage_writes_owner_only_atomic_artifacts(self):
        from app.infra.storage import ObjectStorage, StorageConfig

        with tempfile.TemporaryDirectory() as directory:
            storage = ObjectStorage(StorageConfig(backend_url=f"file://{directory}"))
            uri = storage.save_json("audit/evidence.json", {"ok": True})
            path = Path(uri.removeprefix("file://"))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(storage.load_json(uri), {"ok": True})
            self.assertFalse(any(item.name.startswith(".evidence.json.tmp") for item in path.parent.iterdir()))

    def test_external_signals_v3_uses_public_replay_and_never_mock_entities(self):
        payload = TestClient(server.app).get("/api/v3/external-signals/evidence").json()
        self.assertEqual(payload["version"], "V3")
        self.assertTrue(payload["boundary"]["public_replay_available"])
        self.assertFalse(payload["boundary"]["live_external_data_verified"])
        self.assertFalse(payload["boundary"]["production_authority"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        self.assertEqual(payload["port"]["unlocode"], "CNSHA")
        dataset = payload["dataset"]
        self.assertEqual(dataset["dataset_id"], "public_cn_sha_hourly_v3")
        self.assertEqual(dataset["rows"], 17544)
        self.assertEqual(len(dataset["sha256"]), 64)
        self.assertEqual(dataset["official_reporting_periods"], 22)
        self.assertEqual(dataset["reanalysis_hours"], 17544)
        self.assertEqual(payload["live_adapter_count"], 0)
        self.assertEqual(len(payload["timeline"]), 24)
        self.assertTrue(all(row["source_timestamp"] for row in payload["timeline"]))
        self.assertGreater(
            max(row["demand_kw"] for row in payload["timeline"])
            - min(row["demand_kw"] for row in payload["timeline"]),
            1,
        )
        self.assertEqual(payload["tables"]["tos_schedule"]["rows"], [])
        self.assertEqual(payload["tables"]["ais_arrivals"]["rows"], [])
        registry = {row["id"]: row for row in payload["signal_registry"]}
        self.assertTrue(registry["tide_m"]["model_input"])
        self.assertFalse(registry["tos_schedule"]["model_input"])
        self.assertEqual(registry["ais_tracks"]["availability"], "待接入港口")
        self.assertGreaterEqual(len(payload["public_sources"]), 3)

    def test_mlops_v3_separates_formal_smoke_and_site_promotion(self):
        payload = TestClient(server.app).get("/api/v3/mlops/evidence").json()
        self.assertEqual(payload["version"], "V3")
        self.assertTrue(payload["boundary"]["offline_lifecycle_verified"])
        self.assertFalse(payload["boundary"]["live_model_monitoring_verified"])
        self.assertFalse(payload["boundary"]["automatic_promotion_enabled"])
        self.assertFalse(payload["boundary"]["production_authority"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        summary = payload["summary"]
        self.assertEqual(summary["algorithm_count"], 12)
        self.assertEqual(summary["trainable_rl_count"], 10)
        self.assertGreaterEqual(summary["registry_history_records"], 153)
        self.assertGreaterEqual(summary["formal_runs"], 37)
        self.assertGreaterEqual(summary["smoke_runs"], 33)
        self.assertEqual(summary["selected_algorithm"], "sac")
        self.assertTrue(summary["selected_integrity"])
        evaluation = payload["evaluation"]
        self.assertGreaterEqual(evaluation["learner_formal_runs"], 33)
        self.assertFalse(evaluation["smoke_is_claim_evidence"])
        self.assertEqual(evaluation["selected_seeds"], [42, 142, 242])
        self.assertTrue(evaluation["strict_advantage"])
        self.assertEqual(len(payload["pipeline"]), 8)
        self.assertEqual(payload["pipeline"][-1]["status"], "pending")
        self.assertEqual(len(payload["algorithms"]), 12)
        self.assertEqual(len(payload["selected_models"]), 3)
        self.assertTrue(all(row["verified"] for row in payload["artifact_manifest"]))
        rollback = payload["replacement_and_rollback"]
        self.assertFalse(rollback["automatic_promotion_enabled"])
        self.assertEqual(rollback["current_decision"], "BLOCK")
        self.assertTrue(payload["historical_evidence"]["preserved"])

    def test_governance_v3_is_comprehensive_and_fail_closed(self):
        payload = TestClient(server.app).get("/api/v3/governance/evidence").json()
        self.assertEqual(payload["version"], "V3")
        self.assertTrue(payload["boundary"]["offline_governance_verified"])
        self.assertFalse(payload["boundary"]["site_identity_verified"])
        self.assertFalse(payload["boundary"]["audited_carbon_ledger_verified"])
        self.assertFalse(payload["boundary"]["production_authority"])
        self.assertEqual(payload["boundary"]["site_status"], "待接入港口")
        summary = payload["summary"]
        self.assertEqual(summary["control_count"], 12)
        self.assertEqual(summary["fail"], 0)
        self.assertGreaterEqual(summary["pending"], 4)
        self.assertTrue(summary["audit_owner_only"])
        self.assertEqual(summary["live_adapter_count"], 0)
        self.assertEqual(summary["release_decision"], "BLOCK")
        self.assertEqual(summary["rollout_traffic_percent"], 0)
        self.assertEqual(len(payload["role_matrix"]), 5)
        self.assertFalse(payload["separation_of_duties"]["one_person_execution_allowed"])
        self.assertGreaterEqual(len(payload["claim_registry"]["allowed"]), 4)
        self.assertGreaterEqual(len(payload["claim_registry"]["prohibited"]), 4)
        self.assertEqual(len(payload["risk_register"]), 6)
        security = payload["open_source_security"]
        self.assertTrue(all(row["present"] for row in security["required_files"]))
        self.assertTrue(all(row["configured"] for row in security["workflows"]))
        self.assertTrue(all(row["actions_pinned_to_commit"] for row in security["workflows"]))
        self.assertEqual(payload["data_policy"]["personal_or_vessel_identity_rows"], 0)
        self.assertFalse(payload["ab_test_policy"]["measured_experiment_available"])
        self.assertEqual(payload["release_gate"]["decision"], "BLOCK")
        self.assertFalse(payload["release_gate"]["production_release_allowed"])
        self.assertGreaterEqual(len(payload["release_gate"]["blockers"]), 4)

    def test_asset_modules_disclose_teacher_distillation_and_reward_curve_boundary(self):
        client = TestClient(server.app)
        for endpoint in (
            "/api/v3/modules/yard-lighting/evidence",
            "/api/v3/modules/hvac/evidence",
            "/api/v3/modules/shore-bess/evidence",
            "/api/v3/modules/bess-energy/evidence",
            "/api/v3/modules/yard-crane/evidence",
        ):
            payload = client.get(endpoint).json()
            process = payload["training_process"]
            method = process["training_method"]
            self.assertEqual(
                method["method_family"],
                "constraint_projected_teacher_actor_distillation",
            )
            self.assertEqual(
                method["optimizer_objective"],
                "teacher_action_mean_squared_error",
            )
            self.assertFalse(method["environment_reward_optimized"])
            self.assertFalse(method["policy_gradient_updates"])
            self.assertFalse(method["q_function_updates"])
            self.assertFalse(method["validation_reward_is_training_reward"])
            self.assertFalse(method["checkpoint_reward_replay_is_training_log"])
            self.assertEqual(len(method["report_sha256"]), 64)

        ui = Path("app/ui/index.html").read_text(encoding="utf-8")
        self.assertIn("检查点验证奖励回放 ΔR（非训练时RL奖励）", ui)
        self.assertIn("教师动作模仿损失", ui)
        self.assertIn("策略梯度更新", ui)

    def test_regulatory_resilience_v4_evidence_is_hash_gated_and_fail_closed(self):
        response = TestClient(server.app).get(
            "/api/rl/regulatory-resilience/evidence"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["status"], "ADMITTED_OFFLINE_SCENARIO_CANDIDATE"
        )
        self.assertEqual(len(payload["report_sha256"]), 64)
        self.assertEqual(payload["forward_challenge_status"], "PASS")
        self.assertEqual(len(payload["forward_challenge_sha256"]), 64)
        self.assertFalse(payload["production_authority"])
        report = payload["report"]
        self.assertEqual(report["contract"]["observation_dimensions"], 53)
        self.assertEqual(report["contract"]["action_dimensions"], 7)
        self.assertEqual(report["training"]["steps_per_seed"], 20000)
        self.assertEqual(len(report["training"]["runs"]), 3)
        self.assertTrue(report["admission"]["passed"])
        self.assertFalse(report["admission"]["model_promoted"])
        self.assertFalse(report["admission"]["production_authority"])
        self.assertTrue(report["legacy_preservation"]["preserved"])
        self.assertNotIn("sha256_before", report["legacy_preservation"])
        self.assertEqual(
            report["evidence_label"],
            "PREDECLARED_ENGINEERING_STRESS_SCENARIO_NOT_FIELD_KPI",
        )
        forward = payload["forward_challenge"]
        self.assertEqual(
            forward["evidence_label"],
            "OUT_OF_PERIOD_FORWARD_ENGINEERING_STRESS_CHALLENGE_NOT_FIELD_KPI",
        )
        self.assertEqual(forward["forward_dataset"]["rows"], 3624)
        self.assertFalse(
            forward["forward_dataset"]["candidate_selection_allowed"]
        )
        self.assertFalse(forward["protocol"]["candidate_selection_or_tuning"])
        self.assertEqual(forward["protocol"]["paired_windows"], 20)
        self.assertEqual(forward["protocol"]["episode_steps"], 48)
        self.assertTrue(forward["admission"]["passed"])
        self.assertFalse(forward["admission"]["model_promoted"])
        self.assertFalse(forward["admission"]["production_authority"])
        self.assertEqual(
            forward["candidate_metrics"]["guardrail_violation_rate"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
