"""Verified public model loading at runtime, without editing frozen trainers."""
from __future__ import annotations

import json
from pathlib import Path

from .datasets import file_sha256
from .model_artifacts import resolve_model_artifact

ROOT = Path(__file__).resolve().parents[3]


def resolve_runtime_artifact(run_dir: Path, expected_sha256: str) -> Path:
    original = (Path(run_dir) / 'model.zip').resolve()
    if not expected_sha256:
        raise ValueError('model manifest has no original model SHA-256')
    if original.is_relative_to(ROOT):
        return resolve_model_artifact(ROOT, str(original.relative_to(ROOT)), expected_sha256)
    # Isolated local/test training roots have no repository export manifest.
    if not original.is_file() or file_sha256(original) != expected_sha256:
        raise ValueError('runtime model artifact checksum failed')
    return original


def load_runtime_policy(manager, config, run_dir, env):
    if config['algorithm'] in {'mpc', 'fcfs'}:
        return manager._load_policy(config, run_dir, env)
    manifest = json.loads((Path(run_dir) / 'manifest.json').read_text(encoding='utf-8'))
    model_path = resolve_runtime_artifact(run_dir, manifest.get('model_sha256'))
    if model_path == (Path(run_dir) / 'model.zip').resolve():
        return manager._load_policy(config, run_dir, env)
    with manager.policy_load_lock:
        from sb3_contrib import ARS, QRDQN, RecurrentPPO, TQC, TRPO
        from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
        classes = {'sac': SAC, 'ppo': PPO, 'td3': TD3, 'dqn': DQN,
                   'a2c': A2C, 'tqc': TQC, 'qrdqn': QRDQN, 'trpo': TRPO,
                   'recurrent_ppo': RecurrentPPO, 'ars': ARS}
        return classes[config['algorithm']].load(str(model_path), env=env, device='cpu')


class _RuntimePolicyView:
    """Delegate unchanged physics/inference to the manager with one loader override."""
    def __init__(self, manager):
        self.manager = manager

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def _load_policy(self, config, run_dir, env):
        return load_runtime_policy(self.manager, config, run_dir, env)


def predict_runtime(manager, job_id, payload):
    return type(manager).predict(_RuntimePolicyView(manager), job_id, payload)
