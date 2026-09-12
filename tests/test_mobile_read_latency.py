from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
import asyncio
from threading import Event
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.mobile_api import api


class MobileReadLatencyTests(unittest.TestCase):
    def setUp(self):
        api._workflow_cache = None

    def tearDown(self):
        api._workflow_cache = None

    def test_concurrent_reads_validate_once_and_return_independent_reports(self):
        with patch.object(api, '_workflow_fingerprint', return_value=b'first'), patch.object(
            api, 'load_workflow_benchmark', return_value={'release_gate': {'passed': True}}
        ) as validate:
            with ThreadPoolExecutor(max_workers=4) as pool:
                reports = list(pool.map(lambda _: api._verified_workflow(), range(8)))
            self.assertEqual(validate.call_count, 1)
            reports[0]['release_gate']['passed'] = False
            self.assertTrue(reports[1]['release_gate']['passed'])
            with patch.object(api, '_workflow_fingerprint', return_value=b'changed'):
                api._verified_workflow()
            self.assertEqual(validate.call_count, 2)

    def test_changed_or_missing_evidence_never_returns_cached_success(self):
        with patch.object(api, '_workflow_fingerprint', return_value=b'first'), patch.object(
            api, 'load_workflow_benchmark', return_value={'ok': True}
        ):
            api._verified_workflow()
        with patch.object(api, '_workflow_fingerprint', return_value=b'changed'), patch.object(
            api, 'load_workflow_benchmark', side_effect=ValueError('stale')
        ):
            with self.assertRaises(ValueError):
                api._verified_workflow()
        with patch.object(api, '_workflow_fingerprint', side_effect=FileNotFoundError()):
            with self.assertRaises(FileNotFoundError):
                api._verified_workflow()

    def test_content_change_invalidates_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            path.write_text('aaaa')
            with patch.object(api.workflow_benchmark, 'DEFAULT_REPORT', path):
                before = api._workflow_fingerprint()
                path.write_text('bbbb')
                self.assertNotEqual(before, api._workflow_fingerprint())

    def test_situation_does_not_run_unrelated_workflow_benchmark(self):
        app = FastAPI()
        app.include_router(api.router)
        with patch.object(api, '_verified_workflow', side_effect=AssertionError('expensive rebuild')):
            response = TestClient(app).get('/api/mobile/situation')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['live_data_verified'])


class RuntimeLoadingLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_import_does_not_block_other_coroutines(self):
        from app.services.rl_training import api as rl_api
        entered, release = Event(), Event()

        def slow_capabilities():
            entered.set()
            release.wait(timeout=3)
            return {'runtime': {'available': True}}

        with patch.object(rl_api.TRAINING_MANAGER, 'capabilities', slow_capabilities):
            task = asyncio.create_task(rl_api.capabilities())
            try:
                await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1)
                self.assertFalse(task.done(), 'runtime import blocked the event loop')
            finally:
                release.set()
                await task
