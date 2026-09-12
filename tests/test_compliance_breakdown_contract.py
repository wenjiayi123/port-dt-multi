"""Absent audited monthly ESG data is unavailable, never measured zero or HTTP 500."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.services.esg import service


class ComplianceBreakdownContractTests(unittest.TestCase):
    def test_absent_audited_month_returns_explicit_nulls_via_real_route(self):
        from app.server import app
        with tempfile.TemporaryDirectory() as temp, patch.object(service, 'DATA_DIR', Path(temp)):
            with TestClient(app) as client:
                response = client.get('/api/compliance/breakdown', params={'port': 'CNSHA', 'year': 2026, 'month': 9})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body['available'])
        self.assertEqual(body['_source'], 'compliance.unavailable')
        self.assertIsNone(body['scope2_ton'])
        self.assertTrue(all(value is None for value in body['electric_mwh'].values()))

    def test_available_row_keeps_actual_values_and_missing_month_stays_unavailable(self):
        row = {'month': 1, 'scope1_ton': 3., 'scope2_grid_ton': 2., 'scope2_shore_ton': 1.,
               'scope2_ton': 3., 'grid_mwh': 4., 'shore_power_mwh': 2., 'onsite_renewables_mwh': .5,
               'electric_mwh_total': 6.5, 'intensity_kg_per_teu': .12}
        with patch.object(service, 'get_compliance_timeseries', return_value={'items': [row], '_source': 'compliance.file_verified'}):
            actual = service.get_compliance_breakdown('CNSHA', 2026, 1)
            missing = service.get_compliance_breakdown('CNSHA', 2026, 2)
        self.assertTrue(actual['available']); self.assertEqual(actual['electric_mwh']['total'], 6.5)
        self.assertFalse(missing['available']); self.assertIsNone(missing['scope2_ton'])
