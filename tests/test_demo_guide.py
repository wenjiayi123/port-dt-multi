from __future__ import annotations

import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from app import server


ROOT = Path(__file__).resolve().parents[1]


class DemoGuideContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(server.app)
        cls.guide = (ROOT / "app/ui/adapters/demo_guide.js").read_text(encoding="utf-8")
        cls.home = (ROOT / "app/ui/index.html").read_text(encoding="utf-8")
        cls.rl_console = (ROOT / "app/ui/adapters/rl_evidence_console.js").read_text(
            encoding="utf-8"
        )

    def test_guide_is_loaded_on_core_and_quick_tour_surfaces(self):
        marker = "/ui/adapters/demo_guide.js?v=20261018-native-v3"
        for path in (
            "/",
            "/rl-panel",
            "/ops-copilot",
            "/integration-hub",
            "/v3",
            "/rl_future/rl_future_panel.html",
            "/docs",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(marker, response.text)
        asset = self.client.get("/ui/adapters/demo_guide.js")
        self.assertEqual(asset.status_code, 200)
        self.assertIn("port-dt:demo-receipt", asset.text)

    def test_steps_use_digital_twin_rl_and_xiaoyi_native_controls(self):
        for marker in (
            'selector:\'a.home-action[data-panel="cockpit"]\'',
            'selector:"#story-play"',
            'selector:"#btn-ext-provenance"',
            'selector:"[data-proof-datasets-focus]"',
            'selector:"[data-proof-contract-focus]"',
            'selector:\'.xiaoyi-sprite-btn[type="submit"]\'',
            'selector:"#btnStartTrain"',
            'selector:"#btnConfirmAssistantRun"',
            'selector:"#btnPollTrainStatus"',
            'selector:"#btn-d-convergence"',
            'selector:"#btn-d-business"',
            'selector:"#btn-trust-boundary"',
            'selector:"#btn-opsx-audit"',
            'selector:\'a[href="#twin3d-section"][data-module-page]\'',
            'selector:\'a[href="#multiport-section"][data-module-page]\'',
            'selector:\'a[href="#esg-section"][data-module-page]\'',
            'selector:\'a[href="#strategy-exec-module"][data-module-page]\'',
            'sourceSelector:\'a[data-route="/ops-copilot"]\'',
            'sourceSelector:\'a[data-route="/v3?from=home"]\'',
            'sourceSelector:\'a[data-route="/integration-hub"]\'',
            'sourceSelector:\'a[data-route="/rl_future/rl_future_panel.html"]\'',
            'sourceSelector:\'a[data-route="/docs"]\'',
        ):
            self.assertIn(marker, self.guide)
        self.assertNotIn("#askBtn", self.guide)

    def test_one_primary_button_switches_from_action_to_next_after_receipt(self):
        self.assertIn('data-guide-action="primary"', self.guide)
        self.assertNotIn('data-guide-action="next"', self.guide)
        self.assertIn('state.phase === "done"', self.guide)
        self.assertIn('"✓ 下一步并演示"', self.guide)
        self.assertIn('`▶ 点击：${step.label}`', self.guide)
        self.assertIn('if(state.phase === "done"){ advanceCurrent(false, true); return; }', self.guide)

    def test_sync_is_event_driven_and_has_no_page_observer_or_polling(self):
        for marker in (
            'const RECEIPT_EVENT = "port-dt:demo-receipt"',
            "window.portDtDemoReceipt = function",
            'window.addEventListener(RECEIPT_EVENT',
            'document.addEventListener("click",onPageClick,true)',
            "STEPS.findIndex",
            "jumpToStep",
            'state.pendingExecute',
            'window.addEventListener("port-dt:module-menu-ready"',
            "scheduleLandingReceipt",
        ):
            self.assertIn(marker, self.guide)
        self.assertNotIn("MutationObserver", self.guide)
        self.assertNotIn("setInterval", self.guide)

    def test_quick_tour_module_links_emit_explicit_receipts(self):
        for marker in (
            "'twin3d-showcase'",
            "'multiport-showcase'",
            "'esg-showcase'",
            "'strategy-showcase'",
            "demoStepByTarget",
        ):
            self.assertIn(marker, self.home)
        standalone = (ROOT / "app/static/standalone_navigation.js").read_text(encoding="utf-8")
        self.assertIn("port-dt:module-menu-ready", standalone)

    def test_original_handlers_emit_explicit_visible_result_receipts(self):
        for marker in (
            "portDtDemoReceipt?.('opening'",
            "portDtDemoReceipt?.('situation'",
            "portDtDemoReceipt?.('dataset'",
            "portDtDemoReceipt?.('convergence'",
            "portDtDemoReceipt?.('business'",
            "portDtDemoReceipt?.('safety'",
            "portDtDemoReceipt?.('audit'",
        ):
            self.assertIn(marker, self.home)
        for marker in (
            '"rl-dataset"',
            '"rl-contract"',
            "data-proof-datasets-focus",
            "data-proof-contract-focus",
        ):
            self.assertIn(marker, self.rl_console)
        sprite = (ROOT / "app/ui/adapters/xiaoyi_sprite.js").read_text(encoding="utf-8")
        self.assertIn('window.portDtDemoReceipt?.("xiaoyi-command"', sprite)
        server_source = (ROOT / "app/server.py").read_text(encoding="utf-8")
        for marker in ('"training-review"', '"training-start"', '"training-status"'):
            self.assertIn(marker, server_source)

    def test_quick_demo_training_is_real_but_evidence_bounded(self):
        source = (ROOT / "app/services/assistant_actions/api.py").read_text(encoding="utf-8")
        self.assertIn('"id": "demo_quick"', source)
        self.assertIn('"total_steps": 320', source)
        self.assertIn("只验证链路", source)
        response = self.client.post(
            "/api/assistant/actions/execute",
            json={"instruction": "小鸥，请启动演示级快速训练，平衡成本与碳排，保留严格安全边界和人工确认"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["will_execute"]["action_id"], "start_rl_training")
        self.assertEqual(body["recommendation"]["objective_id"], "demo_quick")
        self.assertEqual(body["recommendation"]["config"]["total_steps"], 320)
        self.assertTrue(body["human_confirmation"]["required"])

    def test_mobile_handoff_opens_and_focuses_the_exact_desktop_request(self):
        response = self.client.post(
            "/api/rl/desktop/launch",
            json={
                "same_window": True,
                "return_to": "http://127.0.0.1:8765/",
                "trace_id": "PORTAI-DEMO-TRACE",
                "action": "review_mobile_request",
                "request_id": "MOBILE-DEMO-REQUEST",
            },
        )
        self.assertEqual(response.status_code, 202)
        body = response.json()
        launch = urlparse(body["launch_url"])
        params = parse_qs(launch.query)
        self.assertEqual(params["from"], ["portai"])
        self.assertEqual(params["action"], ["review_mobile_request"])
        self.assertEqual(params["request_id"], ["MOBILE-DEMO-REQUEST"])
        self.assertEqual(params["trace_id"], ["PORTAI-DEMO-TRACE"])
        self.assertIn("requestFocusFrom", self.client.get("/rl-panel").text)
        self.assertIn("focusMobileTrainingRequest", self.client.get("/rl-panel").text)
        self.assertIn("← 返回 PortAI 移动端", self.client.get("/rl-panel").text)
        self.assertIn("body.portai-handoff #btnBackToPlatform", self.client.get("/rl-panel").text)

    def test_mobile_handoff_uses_the_current_server_origin(self):
        with TestClient(server.app, base_url="http://127.0.0.1:18800") as client:
            status = client.get("/api/rl/desktop/status").json()
            self.assertEqual(status["url"], "http://127.0.0.1:18800/rl-panel")
            result = client.post("/api/rl/desktop/launch", json={
                "same_window": True, "return_to": "http://127.0.0.1:7357/",
                "request_id": "current-origin-test", "trace_id": "origin-test",
            }).json()
            launch = urlparse(result["launch_url"])
            self.assertEqual(launch.netloc, "127.0.0.1:18800")
            self.assertEqual(launch.path, "/rl-panel")
            self.assertEqual(parse_qs(launch.query)["return_to"], ["http://127.0.0.1:7357/"])
            self.assertTrue(result["human_confirmation_required"])
            self.assertFalse(result["can_mobile_bypass"])

    def test_compact_layout_and_recovery_controls_are_present(self):
        for marker in (
            "width:286px",
            "重试本步",
            "重置本步",
            "重置全部",
            "全部停止",
            "保底下一步",
            "重新开始",
            "结束演示",
            "舷鸥下发训练任务",
            "RL 训练数据集",
            "等待结果超时，流程已自动解锁",
            "公开数据 · 离线训练 · 校准模拟",
            "生产权限仍保持关闭",
            "以下为能力速览 · 完成情况以回执为准",
            "API 接口文档",
        ):
            self.assertIn(marker, self.guide)


if __name__ == "__main__":
    unittest.main()
