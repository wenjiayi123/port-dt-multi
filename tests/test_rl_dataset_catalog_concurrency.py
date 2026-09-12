"""The catalog scans CSVs; a cold scan must not block unrelated HTTP reads."""
import asyncio
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from app.services.rl_training import api


class DatasetCatalogConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_catalog_does_not_block_event_loop(self):
        release = threading.Event()
        watchdog = threading.Timer(0.4, release.set)
        watchdog.start()
        app = FastAPI()
        app.include_router(api.router)

        @app.get("/responsive")
        async def responsive():
            return {"ok": True}

        def scan(_root):
            release.wait(1)
            return [{"dataset_id": "fixture-no-csv"}]

        try:
            with patch.object(api, "list_datasets", side_effect=scan):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                    start = time.monotonic()
                    pending = asyncio.create_task(client.get("/api/rl/datasets"))
                    await asyncio.sleep(0.02)
                    response = await client.get("/responsive")
                    elapsed = time.monotonic() - start
                    release.set()
                    catalog = await pending
            self.assertLess(elapsed, 0.2, "catalog scan blocked unrelated HTTP work")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(catalog.json()["count"], 1)
        finally:
            release.set()
            watchdog.cancel()
