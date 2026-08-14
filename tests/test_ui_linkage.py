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
