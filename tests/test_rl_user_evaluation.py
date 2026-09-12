from __future__ import annotations
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from fastapi.testclient import TestClient
from app import server
from app.services.rl_training import api
from app.services.rl_training.user_evaluation import evaluate_user_run

class ToyEnv:
    segment = list(range(12))
    def __init__(self):
        self.trace=[]; self.closed=False
    def reset(self,seed,options):
        self.step_count=0;self.start=options['start_index'];self.trace=[]
        return np.zeros(2),{}
    def step(self,action):
        self.step_count+=1
        self.trace.append({'timestamp':f'2025-01-01T0{self.step_count}:00:00Z','baseline_kw':10,'net_load_kw':9})
        return np.zeros(2),0,self.step_count==2,False,{}
    @property
    def totals(self):
        return {'energy_cost':10+self.start,'carbon_kg':2,'peak_kw':9,'delay':4,'violations':0}
    def close(self):self.closed=True

class UserEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.run=self.root/'runs'/'fixture-job';self.run.mkdir(parents=True)
        config={'algorithm':'sac','dataset_id':'fixture-data','seed':912,'episode_steps':2,'validation_ratio':.1}
        for name in ('evaluation.json','evaluation_trajectory.json','manifest.json','status.json','model_card.json','MODEL_CARD.md','model.zip'):
            (self.run/name).write_text('original-'+name)
        (self.run/'config.json').write_text(json.dumps(config))
        (self.run/'manifest.json').write_text(json.dumps({'model_sha256':hashlib.sha256((self.run/'model.zip').read_bytes()).hexdigest()}))
        self.benchmark=self.root/'benchmarks.json';self.benchmark.write_text('original-benchmark')
        (self.root/'model_registry.json').write_text('original-registry')
        self.envs=[]
        def make_env(*args,**kwargs):
            env=ToyEnv();self.envs.append(env);return env
        self.manager=SimpleNamespace(run_root=self.root/'runs',data_root=self.root/'datasets',benchmark_path=self.benchmark,run_dir=lambda job:self.root/'runs'/job,_resolve_status=lambda job:{'status':'COMPLETED'},_make_env=make_env,_load_policy=lambda *args:SimpleNamespace(predict=lambda obs,**kwargs:(np.zeros(2),None)),evaluation_slots=threading.BoundedSemaphore(1),max_concurrent_evaluation=1,_record_benchmark=Mock(side_effect=AssertionError('must not write benchmark')),_sync_model_registry=Mock(side_effect=AssertionError('must not write registry')))
        self.data=SimpleNamespace(dataset_id='fixture-data',fingerprint='a'*64)
    def snapshot(self):
        return {str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file() and 'user_evaluations' not in p.parts}
    def test_repeat_evaluations_are_new_runs_and_never_touch_training_artifacts(self):
        before=self.snapshot()
        with patch('app.services.rl_training.user_evaluation.load_port_dataset',return_value=self.data):
            first=evaluate_user_run(self.manager,'fixture-job',5)
            second=evaluate_user_run(self.manager,'fixture-job',7)
        self.assertNotEqual(first['evaluation_id'],second['evaluation_id'])
        self.assertEqual(before,self.snapshot())
        self.assertEqual(first['episodes'],5);self.assertEqual(second['episodes'],7)
        self.assertEqual(first['metrics']['guardrail_violation_rate'],0)
        self.assertEqual(first['metrics']['delay_index_mean'],2)
        self.assertEqual(first['render']['frame_count'],2)
        self.assertFalse(first['formal_evidence_updated'])
        self.assertTrue(all(env.closed for env in self.envs))
        for result in [first,second]:
            folder=self.root/'user_evaluations'/'fixture-job'/result['evaluation_id']
            provenance=json.loads((folder/'provenance.json').read_text())
            self.assertEqual(provenance['evaluation_sha256'],hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest())
            self.assertEqual(provenance['gradient_updates'],0)
        self.manager._record_benchmark.assert_not_called();self.manager._sync_model_registry.assert_not_called()
    def test_api_simulate_uses_interactive_service_not_formal_evaluator(self):
        with patch.object(server,'evaluate_user_run',return_value={'metrics':{},'render':{'frames':[]},'evaluation_id':'fixture-id'}) as evaluate,patch.object(server.TRAINING_MANAGER,'evaluate',side_effect=AssertionError('formal writer reached')):
            response=TestClient(server.app).post('/api/rl/simulate',json={'strategy_id':'fixture-job','episodes':7})
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['evaluation']['evaluation_id'],'fixture-id');self.assertEqual(evaluate.call_args.args[2],7)
    def test_train_evaluate_endpoint_uses_interactive_service(self):
        with patch.object(api,'evaluate_user_run',return_value={'evaluation_id':'new-fixture','formal_evidence_updated':False}) as evaluate,patch.object(api.TRAINING_MANAGER,'evaluate',side_effect=AssertionError('formal writer reached')):
            response=TestClient(server.app).post('/api/rl/train/fixture-job/evaluate',json={'episodes':5})
        self.assertEqual(response.status_code,200);evaluate.assert_called_once()
    def test_history_and_detail_read_new_run_without_registering_it(self):
        with patch('app.services.rl_training.user_evaluation.load_port_dataset',return_value=self.data):result=evaluate_user_run(self.manager,'fixture-job',5)
        with patch.object(api,'TRAINING_MANAGER',self.manager):
            client=TestClient(server.app)
            history=client.get('/api/rl/train/fixture-job/evaluation-runs')
            detail=client.get(result['evaluation_artifacts']['result_url'])
        self.assertEqual(history.json()['count'],1);self.assertFalse(history.json()['formal_evidence_updated']);self.assertEqual(detail.json()['evaluation_id'],result['evaluation_id'])
