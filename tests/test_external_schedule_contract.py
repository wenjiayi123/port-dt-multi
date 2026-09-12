import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import server


class ExternalScheduleContractTests(unittest.TestCase):
    def test_missing_adapter_is_unavailable_not_upstream_failure_or_sample_success(self):
        client = TestClient(server.app)
        for adapter in (None, object(), SimpleNamespace(source_status=lambda: {'mode': 'unavailable'}, vessels=lambda **kw: [])):
            with self.subTest(adapter=type(adapter).__name__), patch.object(server.di, 'schedule', adapter):
                response = client.get('/external/vessels_schedule', params={
                    'start': '2026-09-12T00:00:00Z', 'end': '2026-09-12T01:00:00Z'})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()['detail'], 'vessel schedule adapter is not configured')

    def test_configured_schedule_preserves_source_fields_and_real_failure(self):
        adapter = SimpleNamespace(source_status=lambda: {'mode': 'live_rest'}, vessels=lambda **kw: [
            {'vesselName': 'fixture-vessel', 'berthCode': 'B1', 'draft': '8.2', 'planned_moves': 25}])
        client = TestClient(server.app)
        params = {'start': '2026-09-12T00:00:00Z', 'end': '2026-09-12T01:00:00Z', 'port': 'CNSHA'}
        with patch.object(server.di, 'schedule', adapter):
            response = client.get('/external/vessels_schedule', params=params)
            self.assertEqual(response.status_code, 200)
            row = response.json()[0]
            self.assertEqual((row['vessel_id'], row['berth_id'], row['draft_m'], row['moves'], row['_source']),
                             ('fixture-vessel', 'B1', 8.2, 25, 'live_rest'))
            with patch.object(adapter, 'vessels', side_effect=RuntimeError('isolated upstream failure')):
                response = client.get('/external/vessels_schedule', params=params)
                self.assertEqual(response.status_code, 502)
