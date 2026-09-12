"""Consumers can load a verified public archive with the original archive absent."""
import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.services.rl_training import runtime_policy
from app.services.rl_training.model_registry import ModelRegistry

class RuntimePublicModelTests(unittest.TestCase):
    def test_clean_tree_public_sac_load_preserves_actions_and_registry_training_identity(self):
        import gymnasium as gym
        from stable_baselines3 import SAC
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();run=root/'data/rl/runs/fixture-job';run.mkdir(parents=True)
            public=root/'evidence/public_models/model.zip';public.parent.mkdir(parents=True)
            env=gym.make('Pendulum-v1')
            self.addCleanup(env.close)
            model=SAC('MlpPolicy',env,seed=912,policy_kwargs={'net_arch':[8,8]},device='cpu',verbose=0)
            model.save(run/'model.zip')
            original_sha=hashlib.sha256((run/'model.zip').read_bytes()).hexdigest()
            shutil.copy2(run/'model.zip',public);(run/'model.zip').unlink()
            config={'algorithm':'sac','dataset_id':'fixture','seed':912}
            manifest={'model_sha256':original_sha,'implementation':'stable_baselines3.SAC'}
            for name,data in [('config.json',config),('manifest.json',manifest),('status.json',{'job_id':'fixture-job','status':'COMPLETED'}),('evaluation.json',{'episodes':10,'metrics':{},'uncertainty':{}})]:
                (run/name).write_text(json.dumps(data))
            manager=SimpleNamespace(policy_load_lock=threading.RLock(),_load_policy=Mock(side_effect=AssertionError('missing original loader called')))
            with patch.object(runtime_policy,'ROOT',root),patch.object(runtime_policy,'resolve_model_artifact',return_value=public) as resolver:
                loaded=runtime_policy.load_runtime_policy(manager,config,run,env)
                observation,_=env.reset(seed=12)
                np.testing.assert_array_equal(model.predict(observation,deterministic=True)[0],loaded.predict(observation,deterministic=True)[0])
                registry=ModelRegistry(root/'data/rl/runs',root/'data/rl/model_registry.json')
                record=registry.sync('fixture-job')
            self.assertFalse((run/'model.zip').exists())
            self.assertTrue(record['artifact']['verified']);self.assertTrue(record['artifact']['metadata_only_public_export'])
            self.assertEqual(record['artifact']['training_sha256'],original_sha)
            resolver.assert_called_with(root,'data/rl/runs/fixture-job/model.zip',original_sha)
            self.assertEqual(json.loads((run/'manifest.json').read_text()),manifest)
    def test_predict_wrapper_only_changes_policy_loader(self):
        class Manager:
            def predict(self,job_id,payload):return {'job_id':job_id,'policy':self._load_policy({'algorithm':'sac'},Path('fixture'),None),'payload':payload}
        manager=Manager()
        with patch.object(runtime_policy,'load_runtime_policy',return_value='public-policy') as loader:
            result=runtime_policy.predict_runtime(manager,'fixture',{'state':1})
        self.assertEqual(result,{'job_id':'fixture','policy':'public-policy','payload':{'state':1}})
        self.assertIs(loader.call_args.args[0],manager)
    def test_unmapped_external_artifact_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);(run/'model.zip').write_bytes(b'wrong bytes')
            with self.assertRaisesRegex(ValueError,'checksum'):
                runtime_policy.resolve_runtime_artifact(run,'0'*64)
