"""Export V8 metadata-only model copies; --verify works without raw originals."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.services.rl_model.shore_bess.v8_public_artifacts import (
    MODEL_MANIFEST, SOURCE_MANIFEST, inside, metadata_digest, model_exports, resolve_public_artifact, sha256, verify_model_export,
)
LOCAL_PATH = re.compile(rb'(?:/(?:Users|home)/[^/\x00\s]+|[A-Z]:\\Users\\[^\\\x00\s]+)')
SECRET = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b')


def scan_bytes(raw: bytes, label: str) -> None:
    if LOCAL_PATH.search(raw) or SECRET.search(raw):
        raise ValueError('private metadata found in ' + label)


def scan_serialized(value, label='data'):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == ':serialized:':
                scan_bytes(base64.b64decode(child), label + '/serialized')
            else:
                scan_serialized(child, label + '/' + key)
    elif isinstance(value, list):
        for child in value:
            scan_serialized(child, label)


def dump_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('refusing to overwrite a different export: ' + str(path.relative_to(ROOT)))
    else:
        with path.open('xb') as handle:
            handle.write(raw)


def verify_bundle(root: Path = ROOT) -> dict:
    models = model_exports(root)
    for row in models:
        path = verify_model_export(root, row)
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError('corrupt public archive')
            for member in archive.namelist():
                scan_bytes(archive.read(member), path.name + '/' + member)
            scan_serialized(json.loads(archive.read('data')))
    source_manifest = json.loads((root / SOURCE_MANIFEST).read_text())
    sources = source_manifest['entries']
    for row in sources:
        path = resolve_public_artifact(root, row['original_path'], row['original_sha256'])
        scan_bytes(path.read_bytes(), row['public_path'])
    # Scan every stored path->SHA anchor. Explicit unavailable historical
    # versions remain failures of historical completeness, not hidden matches.
    unavailable = {(row['original_path'], row['original_sha256']): row
                   for row in source_manifest.get('unavailable_historical_source_snapshots', [])}
    verified_anchors, observed_gaps = set(), set()
    def walk(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if (isinstance(name, str) and '/' in name and isinstance(child, str)
                        and re.fullmatch(r'[a-f0-9]{64}', child)):
                    identity = (name, child)
                    if identity in unavailable:
                        observed_gaps.add(identity)
                    else:
                        resolve_public_artifact(root, name, child)
                        verified_anchors.add(identity)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for audit in (root / 'evidence/v8/shore_bess/audits').glob('*.json'):
        walk(json.loads(audit.read_text()))
    if observed_gaps != set(unavailable):
        raise ValueError('historical audit gap declaration differs from actual anchors')
    return {'status': 'PASS', 'public_model_count': len(models),
            'original_archive_path_count': sum(len(row['source_paths']) for row in models),
            'audit_source_mapping_count': len(sources), 'verified_distinct_audit_anchors': len(verified_anchors),
            'historical_audit_source_gaps': list(unavailable.values()), 'all_historical_audits_reproducible': not bool(unavailable),
            'raw_archives_required_to_verify_public_bundle': False,
            'production_authority': False}


def export(source_algorithms=None, destination="evidence/v8/shore_bess/public_models",
           manifest_relative=MODEL_MANIFEST, schema="shore-bess-v8-public-model-export.v1"):

    import io
    import numpy as np
    import torch
    from stable_baselines3 import SAC, TD3, DQN, PPO, A2C
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    classes = {'stable_baselines3.SAC': SAC, 'stable_baselines3.TD3': TD3,
               'stable_baselines3.DQN': DQN, 'stable_baselines3.PPO': PPO, 'stable_baselines3.A2C': A2C, 'sb3_contrib.MaskablePPO': MaskablePPO}
    grouped = {}
    paths = sorted((ROOT / name for name in source_algorithms)) if source_algorithms is not None else sorted((ROOT / 'evidence/v8/shore_bess/runs').rglob('*.zip'))
    for path in paths:
        if source_algorithms is None:
            config_path = next((parent / 'config.json' for parent in path.parents
                                if (parent / 'config.json').is_file()), None)
            if config_path is None or not config_path.is_relative_to(ROOT / 'evidence/v8/shore_bess/runs'):
                raise ValueError('missing run config for ' + str(path.relative_to(ROOT)))
            algorithm = json.loads(config_path.read_text())['algorithm']
        else:
            algorithm = source_algorithms[str(path.relative_to(ROOT))]
        digest = sha256(path)
        entry = grouped.setdefault(digest, {'source_paths': [], 'algorithm': algorithm})
        if entry['algorithm'] != algorithm:
            raise ValueError('identical original bytes claim different algorithms')
        entry['source_paths'].append(str(path.relative_to(ROOT)))
    exports = []
    for digest, item in grouped.items():
        original = ROOT / item['source_paths'][0]
        implementation = classes[item['algorithm']]
        before = implementation.load(original, device='cpu')
        with zipfile.ZipFile(original) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        data = json.loads(members['data'])
        if not isinstance(data['learning_rate'], (int, float)):
            raise ValueError('only unchanged scalar learning rates can be reconstructed')
        protected = metadata_digest(data)
        changed = [key for key in ('lr_schedule', 'exploration_schedule') if key in data]
        for key in changed:
            del data[key]
        clip = None
        if 'clip_range' in data:
            grid = np.linspace(0., 1., 101)
            values = [float(before.clip_range(float(progress))) for progress in grid]
            if not all(value == values[0] for value in values):
                raise ValueError('PPO clip schedule is not constant')
            clip = values[0]
            data['clip_range'] = clip
            changed.append('clip_range')
        if metadata_digest(data) != protected:
            raise ValueError('non-schedule model metadata changed')
        scan_serialized(data)
        public_data = json.dumps(data, ensure_ascii=True, indent=2).encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                payload = public_data if name == 'data' else raw
                scan_bytes(payload, name)
                archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 9, 12, 0, 0, 0)), payload, compress_type=zipfile.ZIP_DEFLATED)
        target = ROOT / destination / (digest + '.zip')
        dump_new(target, buffer.getvalue())
        unchanged = {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items() if name != 'data'}
        with zipfile.ZipFile(target) as archive:
            assert all(archive.read(name) == raw for name, raw in members.items() if name != 'data')
        after = implementation.load(target, device='cpu')
        rng = np.random.default_rng(20260912)
        probes = np.clip(rng.normal(0., .5, size=(64,) + before.observation_space.shape),
                         before.observation_space.low, before.observation_space.high).astype(np.float32)
        np.testing.assert_array_equal(before.predict(probes, deterministic=True)[0], after.predict(probes, deterministic=True)[0])
        if isinstance(before, MaskablePPO):
            masks = rng.random((64, before.action_space.n)) > .5
            masks[:, 0] = True
            np.testing.assert_array_equal(before.predict(probes, deterministic=True, action_masks=masks)[0],
                                          after.predict(probes, deterministic=True, action_masks=masks)[0])
        for progress in np.linspace(0., 1., 101):
            for field in changed:
                assert getattr(before, field)(float(progress)) == getattr(after, field)(float(progress))
        for field in ('num_timesteps', '_n_updates', '_v8_optimizer_step_calls', '_v8_optimizer_steps_by_component', '_v8_replay_snapshot'):
            assert getattr(before, field, None) == getattr(after, field, None)
        assert sha256(original) == digest
        row = {**item, 'training_model_sha256': digest, 'public_model_path': str(target.relative_to(ROOT)),
               'public_model_sha256': sha256(target), 'original_data_member_sha256': hashlib.sha256(members['data']).hexdigest(),
               'protected_metadata_sha256': protected, 'regenerated_schedule_fields': changed, 'constant_clip_range': clip,
               'unchanged_zip_member_sha256': unchanged, 'weights_and_optimizer_members_identical': True,
               'inference_identical': True, 'inference_probe_count': 64, 'masked_inference_identical': isinstance(before, MaskablePPO),
               'schedules_identical': True, 'schedule_progress_probe_count': 101, 'training_counters_identical': True}
        exports.append(row)
        print(json.dumps({'exported': len(exports), 'source_paths': len(item['source_paths']), 'algorithm': item['algorithm'],
                          'training_model_sha256': digest}), flush=True)
    manifest = {'schema': schema, 'models': exports,
                'method': 'Regenerate scalar learning-rate and DQN exploration schedules from unchanged parameters; replace verified constant PPO clipping schedule with its identical scalar. All non-data archive members are byte-identical.',
                'source_model_archives_modified': False, 'historical_reports_modified': False, 'production_authority': False}
    dump_new(ROOT / manifest_relative, (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    return verify_bundle() if manifest_relative == MODEL_MANIFEST else {"status": "PASS", "models": len(exports), "source_paths": len(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_bundle() if args.verify else export(), ensure_ascii=False))

if __name__ == '__main__':
    main()
