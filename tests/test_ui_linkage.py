from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import server


ROOT = Path(__file__).resolve().parents[1]


class UiLinkageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(server.app)

    def test_future_decision_uses_v3_verified_evidence_and_fails_closed(self):
        response = self.client.post(
            "/api/v3/future-decision/run",
            json={
                "horizon_min": 90,
                "step_min": 5,
                "max_candidates": 3,
                "source": "ui-linkage-test",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schema"], "port-dt-v3-future-decision.v1")
        self.assertEqual(len(payload["candidates"]), 3)
        self.assertEqual(
            {item["id"] for item in payload["candidates"]},
            {"sac-selected-policy", "mpc-formal-reference", "fcfs-neutral-reference"},
        )
        self.assertIsNone(payload["recommended_strategy_id"])
        self.assertFalse(payload["decision"]["ready_for_human_dry_run"])
        self.assertFalse(payload["production_authority"])
        self.assertFalse(payload["audit"]["production_action_executed"])
        self.assertEqual(len(payload["audit"]["evidence_digest"]), 64)
        self.assertTrue(
            any(
                item["id"] == "model_drift" and item["passed"] is False
                for item in payload["guardrails"]
            )
        )

    def test_future_decision_rejects_invalid_bounds(self):
        response = self.client.post(
            "/api/v3/future-decision/run",
            json={"horizon_min": 5, "step_min": 10, "max_candidates": 9},
        )
        self.assertEqual(response.status_code, 422)

    def test_desktop_buttons_receive_explicit_route_capabilities(self):
        payload = self.client.get("/api/rl/integration/health").json()
        self.assertFalse(payload["desktop_integrations_enabled"])
        self.assertFalse(payload["systems"]["xiaoyi_ai"]["desktop_control_available"])
        sailing = payload["systems"]["sailing_simulator"]
        self.assertFalse(sailing["desktop_control_available"])
        self.assertIn("/api/sailing/logs", sailing["routes"])
        self.assertFalse(sailing["routes"]["/api/sailing/logs"])
        self.assertIn("未启用", payload["summary"]["sailing"])

    def test_frontends_use_capability_guards_and_current_v3_route(self):
        future_js = (ROOT / "app/ui/rl_future/rl_future.js").read_text(encoding="utf-8")
        self.assertIn("/api/v3/future-decision/run", future_js)
        self.assertNotIn("fetch('/api/rl/future/run'", future_js)
        hub = (ROOT / "app/ui/integration_hub.html").read_text(encoding="utf-8")
        for marker in (
            "updateIntegrationCapabilities",
            "integrationCapabilities.sailingLogs",
            "confirmSailingOperation",
            "compactRlStatus",
        ):
            self.assertIn(marker, hub)

    def test_v3_detail_requests_cannot_overwrite_a_newer_drawer(self):
        source = (ROOT / "app/ui/v3/v3.js").read_text(encoding="utf-8")
        self.assertIn("let detailRequestToken=0;", source)
        self.assertIn("const requestToken=++detailRequestToken;", source)
        self.assertIn("if(requestToken!==detailRequestToken) return;", source)
        self.assertIn("function closeDrawer(){ detailRequestToken+=1;", source)

    def test_integration_hub_adapts_v5_policy_test_metrics_and_compacts_receipt(self):
        source = (ROOT / "app/ui/integration_hub.html").read_text(encoding="utf-8")
        for marker in (
            "result?.test_metrics?.metrics",
            "hard_constraint_intervention_rate",
            "function compactPolicyTestReceipt(data)",
            'setText("#rlPacket", pretty(compactPolicyTestReceipt(data)))',
            "完整轨迹保留在后端评测接口",
        ):
            self.assertIn(marker, source)

    def test_rl_panel_keeps_read_only_southbound_capability_accessible_and_null_metrics_unavailable(self):
        source = (ROOT / "app/server.py").read_text(encoding="utf-8")
        self.assertIn('<button id="btnHistory" class="btn ghost">查看南向能力</button>', source)
        self.assertIn('value !== null && value !== undefined && value !== ""', source)
        self.assertIn('const rowSimulationButtons = $$("#tbl button[data-simid]")', source)
        self.assertIn('rowSimulationButtons.forEach(button=>{ button.disabled = true; })', source)
        self.assertIn('rowSimulationButtons.forEach(button=>{ button.disabled = false; })', source)
        self.assertIn('$("#btnDispatch").disabled = true;', source)

    def test_main_simulation_export_button_reaches_canvas_download_handler(self):
        source = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        self.assertIn('<canvas id="simCanvas" aria-label="强化学习策略与基线聚合功率对比图"></canvas>', source)
        self.assertIn("globalThis.saveCanvasPNG = saveCanvasPNG;", source)
        self.assertIn("saveCanvasPNG('simCanvas', `rl-sim-${Date.now()}.png`)", source)
        self.assertIn("function downloadSimulationCanvas()", source)
        self.assertIn("el.addEventListener('click', downloadSimulationCanvas)", source)

    def test_qc_aggregate_simulation_reuses_the_registered_twin_mode_control(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        self.assertIn(
            "document.querySelector('#viewSeg button[data-mode=\"sim\"]')?.click();",
            home,
        )
        self.assertNotIn("clickMode('sim')", home)

    def test_primary_navigation_grid_covers_all_thirteen_entries(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        self.assertEqual(home.count("grid-template-columns:repeat(13,"), 3)
        self.assertNotIn("grid-template-columns:repeat(12,", home)

    def test_direct_training_button_requires_the_same_human_review_gate(self):
        source = (ROOT / "app/server.py").read_text(encoding="utf-8")
        self.assertIn(
            '$("#btnStartTrain")?.addEventListener("click", showAssistantRunConfirm);',
            source,
        )
        self.assertNotIn(
            '$("#btnStartTrain")?.addEventListener("click", startTraining);',
            source,
        )
        self.assertIn("已取消训练启动；未调用 /api/rl/train/start。", source)

    def test_policy_test_failure_exits_testing_state(self):
        hub = (ROOT / "app/ui/integration_hub.html").read_text(encoding="utf-8")
        self.assertIn('setText("#impactRisk", "BLOCKED");', hub)
        self.assertIn(
            'setText("#impactTargetState", `策略测试未通过 · ${status} · 未进入上线护栏`);',
            hub,
        )
        self.assertIn(
            'setText("#impactTargetState", "策略测试请求失败 · 未进入上线护栏");',
            hub,
        )

    def test_port_call_gateway_has_visible_fail_closed_preflight_flow(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'id="port-call-gateway-status"',
            'id="btn-port-call-readiness"',
            'id="btn-port-call-contract"',
            'id="btn-port-call-preflight"',
            "/api/v3/port-call/readiness",
            "/api/v3/port-call/validate",
            "schema_version:payload.schema_version",
            "合同样例通过 · 真实数据仍未接入",
        ):
            self.assertIn(marker, home)
        self.assertIn("fallback_simulator", (ROOT / "app/services/port_call_gateway.py").read_text(encoding="utf-8"))

    def test_site_twin_calibration_has_visible_holdout_and_fail_closed_flow(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="calibration"',
            'id="calibration-readiness"',
            'id="calibration-contract"',
            'id="calibration-preflight"',
            "/api/v3/twin-calibration/readiness",
            "/api/v3/twin-calibration/run",
            "训练窗并核验独立验证窗",
            "合同通过≠现场标定",
            "无生产控制权",
        ):
            self.assertIn(marker, home)

    def test_site_shadow_acceptance_has_visible_read_only_evidence_flow(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="shadow-acceptance"',
            'id="shadow-readiness"',
            'id="shadow-contract"',
            'id="shadow-preflight"',
            "/api/v3/shadow-acceptance/readiness",
            "/api/v3/shadow-acceptance/run",
            "候选业务效果仍是预测值，不冒充实测收益",
            "合同通过≠现场验收",
            "不执行设备指令",
        ):
            self.assertIn(marker, home)

    def test_site_execution_acceptance_has_visible_fail_closed_control_flow(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="execution-acceptance"',
            'id="execution-readiness"',
            'id="execution-contract"',
            'id="execution-preflight"',
            "/api/v3/execution-acceptance/readiness",
            "/api/v3/execution-acceptance/run",
            "八项场景齐全",
            "合同通过≠现场放行",
            "未发送设备指令",
        ):
            self.assertIn(marker, home)

    def test_port_call_collaboration_has_visible_delay_and_receipt_flow(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="port-call-collaboration"',
            'id="collaboration-readiness"',
            'id="collaboration-contract"',
            'id="collaboration-preflight"',
            "/api/v3/port-call-collaboration/readiness",
            "/api/v3/port-call-collaboration/run",
            "正在传播延误并核验泊位、引航员、拖轮冲突与六方回执",
            "合同通过≠现场协同",
            "未修改真实计划 · 无生产控制权",
        ):
            self.assertIn(marker, home)
        service = (ROOT / "app/services/port_call_collaboration.py").read_text(encoding="utf-8")
        self.assertIn('"shared_plan_mutated": False', service)
        self.assertIn('"authority_to_change_eta": False', service)

    def test_maritime_interoperability_has_visible_mapping_and_authority_boundary(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="maritime-interoperability"',
            'id="interoperability-readiness"',
            'id="interoperability-contract"',
            'id="interoperability-preflight"',
            "/api/v3/maritime-interoperability/readiness",
            "/api/v3/maritime-interoperability/run",
            "正在核对靠泊事件、单一窗口数据元、水文产品版本与统一映射摘要",
            "合同通过≠外部符合性",
            "不报送、不导航、无生产控制权",
        ):
            self.assertIn(marker, home)
        service = (ROOT / "app/services/maritime_interoperability.py").read_text(encoding="utf-8")
        self.assertIn('"authority_submission_allowed": False', service)
        self.assertIn('"navigational_use_allowed": False', service)
        self.assertIn('"official_certification_claim_allowed": False', service)

    def test_forecast_uncertainty_has_visible_holdout_and_advisory_boundary(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="forecast-uncertainty"',
            'id="forecast-readiness"',
            'id="forecast-contract"',
            'id="forecast-preflight"',
            "/api/v3/forecast-uncertainty/readiness",
            "/api/v3/forecast-uncertainty/run",
            "正在按签发时点核对特征快照",
            "合同通过≠现场服务水平",
            "非现场服务水平 · 只建议、不执行",
        ):
            self.assertIn(marker, home)
        service = (ROOT / "app/services/forecast_uncertainty.py").read_text(encoding="utf-8")
        self.assertIn('"forecast_advisory_only": True', service)
        self.assertIn('"automatic_resource_commitment_allowed": False', service)
        self.assertIn('"dispatch_allowed": False', service)

    def test_business_benefit_has_execution_bound_attribution_and_claim_boundary(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="business-benefit-attribution"',
            'id="benefit-readiness"',
            'id="benefit-contract"',
            'id="benefit-preflight"',
            "/api/v3/business-benefit-attribution/readiness",
            "/api/v3/business-benefit-attribution/run",
            "正在绑定建议、人工批准、实际执行、后验计量与同期对照",
            "合同样例≠现场已实现收益",
            "合同样例≠现场已实现收益 · 不自动承诺资源 · 无生产控制权",
        ):
            self.assertIn(marker, home)
        service = (ROOT / "app/services/business_benefit_attribution.py").read_text(encoding="utf-8")
        self.assertIn('"offline_counterfactual_relabelled_as_field_kpi": False', service)
        self.assertIn('"automatic_resource_commitment_allowed": False', service)
        self.assertIn('"dispatch_allowed": False', service)
        benchmark = (ROOT / "docs/BUSINESS_KPI_BENCHMARK.md").read_text(encoding="utf-8")
        self.assertIn("不是港口实测运营 KPI", benchmark)

    def test_end_to_end_coordination_has_cross_resource_rolling_plan_and_no_commitment_boundary(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="end-to-end-coordination"',
            'id="coordination-readiness"',
            'id="coordination-contract"',
            'id="coordination-preflight"',
            "/api/v3/end-to-end-coordination/readiness",
            "/api/v3/end-to-end-coordination/run",
            "正在核对冻结计划、前后工序、十一类容量、原计划回执和预测回执",
            "合同样例≠现场统一计划",
            "合同求解≠现场统一计划 · 不承诺资源 · 无生产控制权",
        ):
            self.assertIn(marker, home)
        service = (ROOT / "app/services/end_to_end_coordination.py").read_text(encoding="utf-8")
        self.assertIn('"shared_plan_mutated": False', service)
        self.assertIn('"automatic_resource_commitment_allowed": False', service)
        self.assertIn('"dispatch_allowed": False', service)
        self.assertIn('"production_authority": False', service)

    def test_production_continuity_has_hourly_slo_incident_restore_and_drill_boundary(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="production-continuity"',
            'id="continuity-readiness"',
            'id="continuity-contract"',
            'id="continuity-preflight"',
            "/api/v3/production-continuity/readiness",
            "/api/v3/production-continuity/run",
            "正在逐小时核对八个组件",
            "合同样例≠现场连续运行",
            "合同通过≠现场连续运行验收 · 不自动切换 · 无生产控制权",
            "k==='production-continuity' && !continuityReadiness",
            "$('#continuity-incidents').textContent=a.closed_incident_count??'—'",
        ):
            self.assertIn(marker, home)
        service = (ROOT / "app/services/production_continuity.py").read_text(encoding="utf-8")
        self.assertIn('"automatic_failover_authority": False', service)
        self.assertIn('"dispatch_allowed": False', service)
        self.assertIn('"production_authority": False', service)

    def test_operating_model_has_named_raci_four_eyes_roster_and_no_role_assignment_boundary(self):
        home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-tab="operating-model-governance"',
            'id="governance-readiness"',
            'id="governance-contract"',
            'id="governance-preflight"',
            "/api/v3/operating-model-governance/readiness",
            "/api/v3/operating-model-governance/run",
            "正在核对十二类责任、十项异人审批",
            "合同样例≠现场正式任命",
            "职责表≠现场正式任命 · 禁止自批自执 · 无生产控制权",
            "k==='operating-model-governance' && !governanceReadiness",
            "$('#governance-assignments').textContent=a.roster_assignment_count??'—'",
        ):
            self.assertIn(marker, home)
        service = (ROOT / "app/services/operating_model_governance.py").read_text(encoding="utf-8")
        self.assertIn('"system_can_assign_roles": False', service)
        self.assertIn('"self_approval_allowed": False', service)
        self.assertIn('"dispatch_allowed": False', service)
        self.assertIn('"production_authority": False', service)

    def test_local_xiaoyi_fallback_is_operator_facing_chinese(self):
        response = self.client.post(
            "/api/copilot/mission",
            json={
                "mission": "strategy",
                "query": "为什么当前策略不能进入生产？",
                "engine": "local_rag",
                "asset_id": "qc-01",
            },
        )
        self.assertEqual(response.status_code, 200)
        answer = response.json()["summary"]["operator_note"]
        self.assertNotIn("保持保持", answer)
        self.assertNotIn("calibrated_public_replay_simulator", answer)
        self.assertNotIn("block_to_safe_baseline", answer)
        self.assertNotIn("production control authority is false", answer.lower())
        self.assertIn("公开数据校准连续回放", answer)
        self.assertIn("无生产控制权", answer)


if __name__ == "__main__":
    unittest.main()
