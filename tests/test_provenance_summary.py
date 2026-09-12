"""Homepage provenance stays truthful without eagerly loading every learner."""
import asyncio
from threading import Event
import unittest
from unittest.mock import patch

import httpx

from app import server


class ProvenanceSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_summary_defers_only_expensive_model_details(self):
        with patch.object(server.di.strategy_runtime, "status", return_value={"available": True}) as runtime, \
             patch.object(server.TRAINING_MANAGER, "capabilities", return_value={"runtime": {"available": True}}) as training:
            summary = (await self.client.get("/api/system/provenance?detail=summary")).json()
            runtime.assert_not_called()
            training.assert_not_called()
            self.assertIsNone(summary["runtime_policy"]["available"])
            self.assertIsNone(summary["rl"]["runtime"]["available"])
            self.assertIsNone(summary["rl"]["datasets"])
            self.assertEqual(summary["rl"]["verification_state"], "deferred")
            self.assertFalse(summary["production_claim_allowed"])
            full = (await self.client.get("/api/system/provenance")).json()
            runtime.assert_called_once()
            training.assert_called_once()
        for key in ("external_adapters", "telemetry", "portviz", "feature_flags", "production_blockers"):
            self.assertEqual(summary[key], full[key], key)
        self.assertTrue(full["runtime_policy"]["available"])
        self.assertTrue(full["rl"]["runtime"]["available"])

    async def test_each_summary_rechecks_site_approval_without_blocking_health(self):
        entered, release = Event(), Event()
        original = server._site_twin_calibration.readiness
        calls = []

        def slow_read():
            calls.append(True)
            entered.set()
            release.wait(timeout=2)
            return original()

        with patch.object(server._site_twin_calibration, "readiness", slow_read):
            request = asyncio.create_task(self.client.get("/api/system/provenance?detail=summary"))
            try:
                self.assertTrue(await asyncio.wait_for(asyncio.to_thread(entered.wait, 1), timeout=1.5))
                self.assertFalse(request.done())
                live = await asyncio.wait_for(self.client.get("/health/live"), timeout=.5)
                self.assertEqual(live.status_code, 200)
            finally:
                release.set()
                response = await request
            self.assertEqual(response.status_code, 200)
            await self.client.get("/api/system/provenance?detail=summary")
        self.assertEqual(len(calls), 2)

    async def test_invalid_detail_is_rejected(self):
        response = await self.client.get("/api/system/provenance?detail=pretend-ready")
        self.assertEqual(response.status_code, 422)
