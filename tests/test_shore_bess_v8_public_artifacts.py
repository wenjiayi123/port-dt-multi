"""Publication mapping must preserve immutable training identities and fail closed."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from app.services.rl_model.shore_bess.v8_public_artifacts import (
    MODEL_MANIFEST, SOURCE_MANIFEST, metadata_digest, resolve_public_artifact, sha256,
)


class V8PublicArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.original = 'evidence/v8/shore_bess/runs/fixture/seed_1/step_10.zip'
        self.public = 'evidence/v8/shore_bess/public_models/fixture.zip'
        self.data = {'num_timesteps': 10, '_n_updates': 5, 'learning_rate': .0003, 'clip_range': .2}
        self.write_zip()
        self.row = {'training_model_sha256': 'a' * 64, 'source_paths': [self.original],
                    'public_model_path': self.public, 'public_model_sha256': sha256(self.root / self.public),
                    'protected_metadata_sha256': metadata_digest(self.data),
                    'regenerated_schedule_fields': ['lr_schedule', 'clip_range'], 'constant_clip_range': .2,
                    'unchanged_zip_member_sha256': {'policy.pth': hashlib.sha256(b'unchanged tensor fixture').hexdigest()},
                    **{key: True for key in ('weights_and_optimizer_members_identical', 'inference_identical',
                                             'schedules_identical', 'training_counters_identical')}}
        self.manifest = {'schema': 'shore-bess-v8-public-model-export.v1', 'models': [self.row],
                         'source_model_archives_modified': False, 'historical_reports_modified': False}
        self.save()

    def write_zip(self, weights=b'unchanged tensor fixture'):
        p = self.root / self.public
        p.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, 'w') as z:
            z.writestr('data', json.dumps(self.data))
            z.writestr('policy.pth', weights)

    def save(self):
        (self.root / MODEL_MANIFEST).write_text(json.dumps(self.manifest))

    def resolve(self):
        return resolve_public_artifact(self.root, self.original, 'a' * 64)

    def test_missing_raw_archive_resolves_to_hash_bound_public_copy(self):
        self.assertFalse((self.root / self.original).exists())
        self.assertEqual(self.resolve(), self.root / self.public)
        with self.assertRaises(ValueError):
            resolve_public_artifact(self.root, self.original, 'b' * 64)
        with self.assertRaises(ValueError):
            resolve_public_artifact(self.root, '../escape.zip', 'a' * 64)

    def test_present_modified_raw_archive_cannot_be_hidden_by_public_copy(self):
        p = self.root / self.original; p.parent.mkdir(parents=True)
        p.write_bytes(b'changed raw archive')
        with self.assertRaisesRegex(ValueError, 'original V8 archive was modified'):
            self.resolve()

    def test_tampered_public_bytes_rejected_before_deserialization(self):
        (self.root / self.public).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.resolve()

    def test_rehashed_public_network_or_counter_mutation_rejected(self):
        for kind in ('weights', 'counter', 'schedule'):
            with self.subTest(kind=kind):
                self.data['num_timesteps'] = 11 if kind == 'counter' else 10
                self.data['clip_range'] = .3 if kind == 'schedule' else .2
                self.write_zip(b'changed' if kind == 'weights' else b'unchanged tensor fixture')
                self.row['public_model_sha256'] = sha256(self.root / self.public); self.save()
                with self.assertRaises(ValueError):
                    self.resolve()

    def test_missing_equivalence_and_ambiguous_identity_rejected(self):
        self.row['inference_identical'] = False; self.save()
        with self.assertRaisesRegex(ValueError, 'equivalence'):
            self.resolve()
        self.row['inference_identical'] = True
        self.manifest['models'].append(copy.deepcopy(self.row)); self.save()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.resolve()

    def test_source_snapshot_retains_old_hash_after_live_source_changes(self):
        original = 'scripts/old_audit.py'; public = 'evidence/v8/shore_bess/public_audit_sources/old.py'
        path = self.root / public; path.parent.mkdir(parents=True); path.write_text('old source\n')
        original_path = self.root / original; original_path.parent.mkdir(); original_path.write_text('new source\n')
        expected = sha256(path)
        entry = {'original_path': original, 'original_sha256': expected, 'public_path': public, 'public_sha256': expected}
        (self.root / SOURCE_MANIFEST).write_text(json.dumps({'schema': 'shore-bess-v8-public-audit-sources.v1', 'entries': [entry]}))
        self.assertEqual(resolve_public_artifact(self.root, original, expected), path)
        self.assertEqual(original_path.read_text(), 'new source\n')
        path.write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            resolve_public_artifact(self.root, original, expected)
