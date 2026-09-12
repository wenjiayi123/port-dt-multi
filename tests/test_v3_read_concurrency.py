"""CPU/disk evidence reads must not stall unrelated HTTP requests."""
import asyncio
from threading import Event
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException

from app.services import v3_port_ai as api


class V3ReadConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_readiness_and_provenance_leave_health_live_available(self):
        from app import operations, server
        cases = (
            ('/health/ready', operations, 'readiness_report'),
            ('/api/system/provenance', server.TRAINING_MANAGER, 'capabilities'),
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
            for route, owner, dependency in cases:
                with self.subTest(route=route):
                    entered, release = Event(), Event()
                    def slow_read(*args, **kwargs):
                        entered.set()
                        release.wait(timeout=2.)
                        raise HTTPException(status_code=503, detail='isolated delayed readiness fixture')
                    with patch.object(owner, dependency, slow_read), patch.object(server.di.strategy_runtime, 'status', return_value={}):
                        pending = asyncio.create_task(client.get(route))
                        try:
                            await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1.)
                            self.assertFalse(pending.done(), 'readiness or provenance blocked the HTTP event loop')
                            response = await asyncio.wait_for(client.get('/health/live'), timeout=.5)
                            self.assertEqual(response.status_code, 200)
                        finally:
                            release.set()
                            result = await pending
                        self.assertEqual(result.status_code, 503)

    async def test_blocked_evidence_reads_leave_event_loop_and_liveness_available(self):
        application = FastAPI()
        application.include_router(api.router)

        @application.get('/test/live')
        async def live():
            return {'status': 'alive'}

        cases = (
            ('/api/v3/overview', 'load_port_dataset'),
            ('/api/v3/data-readiness', 'load_port_dataset'),
            ('/api/v3/algorithms/sac/evidence', '_algorithm_rows'),
            ('/api/v3/capabilities/' + api.BUSINESS_CAPABILITIES[0]['id'], '_business_depth'),
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            for route, dependency in cases:
                with self.subTest(route=route):
                    entered, release = Event(), Event()
                    def slow_read(*args, **kwargs):
                        entered.set()
                        release.wait(timeout=2.)
                        raise HTTPException(status_code=503, detail='isolated delayed evidence fixture')
                    with patch.dict(api._OVERVIEW_CACHE, {}, clear=True), patch.object(api, dependency, slow_read):
                        pending = asyncio.create_task(client.get(route))
                        try:
                            await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1.)
                            self.assertFalse(pending.done(), 'evidence ran synchronously on the HTTP event loop')
                            response = await asyncio.wait_for(client.get('/test/live'), timeout=.5)
                            self.assertEqual(response.status_code, 200)
                            self.assertEqual(response.json(), {'status': 'alive'})
                        finally:
                            release.set()
                            result = await pending
                        self.assertEqual(result.status_code, 503)
                        self.assertEqual(result.json()['detail'], 'isolated delayed evidence fixture')
