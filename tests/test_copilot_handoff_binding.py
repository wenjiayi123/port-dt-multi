from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.copilot.api import router
from app.services.copilot.mission_control import XiaoyiMissionControl


class FixtureMissionControl(XiaoyiMissionControl):
    def __init__(self):
        super().__init__(None, None, None)
        self.builds = 0
        self.blocked = False

    def build_context(self, *, asset_id='qc-01', mission_id='handoff', **_kwargs):
        self.builds += 1
        return {'context_sha256': f'{self.builds:064x}', 'asset_id': asset_id, 'mission': mission_id,
                'overall_state': 'review', 'source': {'mode': 'test_fixture'},
                'forecast': {'peak_p50_kw': self.builds * 100},
                'monitoring': {'new_policy_suggestions_allowed': not self.blocked},
                'signals': [{'id': 'source', 'name': 'fixture source', 'value': 'replay', 'source': 'fixture'}],
                'policy': {}, 'missing_site_factors': ['site meter'], 'claim_boundary': 'fixture only'}


class CopilotHandoffBindingTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        self.service = FixtureMissionControl()
        app.state.xiaoyi_mission_control = self.service
        app.include_router(router, prefix='/api/copilot')
        self.client = TestClient(app)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.log = Path(temp.name) / 'handoff.jsonl'
        override = patch('app.services.copilot.mission_control.HANDOFF_LOG', self.log)
        override.start()
        self.addCleanup(override.stop)

    def preview(self):
        response = self.client.post('/api/copilot/handoff', json={
            'port': 'CNSHA', 'asset_id': 'qc-01', 'operator': 'local-fixture',
            'shift': 'QA', 'confirm': False, 'answer': 'stale browser answer'})
        self.assertEqual(response.status_code, 200)
        return response.json()['packet']

    def confirmation(self, packet):
        return {'port': 'CNSHA', 'asset_id': 'qc-01', 'operator': 'local-fixture',
                'shift': 'QA', 'confirm': True, 'handoff_sha256': packet['handoff_sha256'],
                'context_sha256': packet['context_sha256'], 'answer': packet['xiaoyi_summary']}

    def test_confirmation_persists_exact_preview_without_rebuilding_context_and_is_idempotent(self):
        packet = self.preview()
        self.assertFalse(self.log.exists())
        self.assertNotIn('stale browser answer', packet['xiaoyi_summary'])
        request = self.confirmation(packet)
        first = self.client.post('/api/copilot/handoff', json=request)
        second = self.client.post('/api/copilot/handoff', json=request)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.json()['status'], 'already_recorded')
        self.assertEqual(self.service.builds, 1)
        stored = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(stored, [packet])
        self.assertFalse(first.json()['production_action_executed'])

    def test_changed_hash_asset_operator_or_answer_cannot_be_confirmed(self):
        packet = self.preview()
        for field, value in [('context_sha256', '0'*64), ('handoff_sha256', '0'*64),
                             ('asset_id', 'yard-01'), ('operator', 'changed'),
                             ('shift', 'changed'), ('answer', 'changed')]:
            with self.subTest(field=field):
                response = self.client.post('/api/copilot/handoff', json={**self.confirmation(packet), field: value})
                self.assertEqual(response.status_code, 409)
                self.assertFalse(self.log.exists())

    def test_expired_or_missing_preview_requires_a_new_preview(self):
        packet = self.preview()
        self.service._handoff_previews[packet['handoff_sha256']]['expires_at'] = 0
        self.assertEqual(self.client.post('/api/copilot/handoff', json=self.confirmation(packet)).status_code, 409)
        self.assertEqual(self.client.post('/api/copilot/handoff', json={'confirm': True}).status_code, 409)
        self.assertFalse(self.log.exists())

    def test_unmapped_port_is_rejected_before_context_or_llm_work(self):
        with patch('app.services.copilot.api._probe_xiaoyi_status', side_effect=AssertionError('unexpected probe')):
            for method, route in [('get', '/context'), ('post', '/mission'), ('post', '/handoff')]:
                with self.subTest(route=route):
                    response = (self.client.get('/api/copilot'+route, params={'port': 'SGSIN'}) if method == 'get'
                                else self.client.post('/api/copilot'+route, json={'port': 'SGSIN'}))
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()['detail']['actual_source_port'], 'CNSHA')
        self.assertEqual(self.service.builds, 0)
        self.assertFalse(self.log.exists())

    def test_visible_retrieval_controls_and_output_mode_are_effective_and_audited(self):
        knowledge = [
            {'id': 'sop-1', 'type': 'sop', 'title': 'SOP first', 'snippet': 'fixture'},
            {'id': 'sop-2', 'type': 'sop', 'title': 'SOP second', 'snippet': 'fixture'},
            {'id': 'device-1', 'type': 'device', 'title': 'device', 'snippet': 'fixture'},
        ]
        with patch('app.services.copilot.api._load_knowledge_items', return_value=knowledge):
            response = self.client.post('/api/copilot/mission', json={
                'engine': 'local_rag', 'scope': 'sop', 'top_k': 1, 'severity': 'critical',
                'mode': 'audit_note', 'query': 'fixture', 'mission': 'situation'})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual([row['id'] for row in data['evidence']], ['source', 'sop-1'])
        self.assertEqual(data['summary']['risk_level'], '高')
        self.assertIn('审计上下文：' + data['context_sha256'], data['summary']['operator_note'])
        audit = data['audit_packet']
        self.assertEqual(audit['effective_parameters'], {'scope': 'sop', 'top_k': 1, 'severity': 'critical', 'mode': 'audit_note'})
        self.assertEqual(audit['knowledge_evidence_count'], 1)
        self.assertEqual(audit['runtime_evidence_count'], 1)
        self.assertEqual(audit['severity_source'], 'operator_report_not_measured_alarm')
        self.assertFalse(self.log.exists())

    def test_output_mode_resolves_mission_and_operator_severity_cannot_lower_guardrail(self):
        self.service.blocked = True
        for mode, mission in [('handoff', 'handoff'), ('alert_triage', 'triage')]:
            with self.subTest(mode=mode):
                data = self.client.post('/api/copilot/mission', json={
                    'engine': 'local_rag', 'severity': 'medium', 'mode': mode,
                    'mission': 'situation'}).json()
                self.assertEqual(data['audit_packet']['mission'], mission)
                self.assertEqual(data['summary']['risk_level'], '高')
                self.assertEqual(data['audit_packet']['risk_basis'], 'runtime_guardrail')
                self.assertFalse(data['production_authority'])

    def test_invalid_controls_fail_before_build_and_empty_scope_does_not_fall_back_to_other_knowledge(self):
        for override in [{'top_k': 0}, {'top_k': 11}, {'top_k': 2.5}, {'top_k': True},
                         {'top_k': 'bad'}, {'scope': 'unknown'}, {'mode': 'unknown'}, {'severity': 'unknown'}]:
            with self.subTest(override=override):
                self.assertEqual(self.client.post('/api/copilot/mission', json=override).status_code, 422)
        self.assertEqual(self.service.builds, 0)
        with patch('app.services.copilot.api._load_knowledge_items', return_value=[{'id': 'device-1', 'type': 'device'}]):
            data = self.client.post('/api/copilot/mission', json={'engine': 'local_rag', 'scope': 'protocol'}).json()
        self.assertEqual(data['audit_packet']['knowledge_evidence_count'], 0)
        self.assertEqual([row['id'] for row in data['evidence']], ['source'])


if __name__ == '__main__':
    unittest.main()
