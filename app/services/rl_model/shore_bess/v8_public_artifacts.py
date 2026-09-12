"""Resolve immutable V8 archive identities to equivalent publishable copies."""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path

MODEL_MANIFEST = 'evidence/v8/shore_bess/public_models/manifest.json'
SOURCE_MANIFEST = 'evidence/v8/shore_bess/public_audit_sources/manifest.json'
SCHEDULE_FIELDS = {'lr_schedule', 'exploration_schedule', 'clip_range'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('public artifact requires a repository-relative path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('public artifact path escapes repository')
    return path


def metadata_digest(data: dict) -> str:
    protected = {key: value for key, value in data.items() if key not in SCHEDULE_FIELDS}
    return hashlib.sha256(json.dumps(protected, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def model_exports(root: Path) -> list[dict]:
    path = root / MODEL_MANIFEST
    if not path.is_file():
        return []
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'shore-bess-v8-public-model-export.v1'
            or manifest.get('source_model_archives_modified') is not False
            or manifest.get('historical_reports_modified') is not False):
        raise ValueError('invalid V8 public model manifest')
    rows = manifest.get('models')
    if not isinstance(rows, list) or len({row['training_model_sha256'] for row in rows}) != len(rows):
        raise ValueError('ambiguous V8 public model identities')
    paths = [path for row in rows for path in row['source_paths']]
    if len(paths) != len(set(paths)):
        raise ValueError('ambiguous V8 public original paths')
    return rows


def verify_model_export(root: Path, row: dict, *, export_directory="evidence/v8/shore_bess/public_models") -> Path:
    if not all(row.get(key) is True for key in ('weights_and_optimizer_members_identical', 'inference_identical',
                                               'schedules_identical', 'training_counters_identical')):
        raise ValueError('V8 public model lacks equivalence evidence')
    path = inside(root, row['public_model_path'])
    if not path.is_relative_to((root / export_directory).resolve()):
        raise ValueError('V8 public model is outside its versioned export directory')
    if sha256(path) != row['public_model_sha256']:
        raise ValueError('V8 public model SHA-256 mismatch')
    with zipfile.ZipFile(path) as archive:
        members = row['unchanged_zip_member_sha256']
        if len(set(archive.namelist())) != len(archive.namelist()) or set(archive.namelist()) != set(members) | {'data'}:
            raise ValueError('V8 public archive member inventory differs')
        for name, expected in members.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('V8 public network/optimizer member differs: ' + name)
        data = json.loads(archive.read('data'))
        if 'lr_schedule' in data or 'exploration_schedule' in data:
            raise ValueError('V8 public archive retains serialized schedule metadata')
        if metadata_digest(data) != row['protected_metadata_sha256']:
            raise ValueError('V8 protected model metadata differs')
        if not set(row['regenerated_schedule_fields']) <= SCHEDULE_FIELDS:
            raise ValueError('undeclared V8 model metadata transformation')
        if 'clip_range' in row['regenerated_schedule_fields'] and data.get('clip_range') != row['constant_clip_range']:
            raise ValueError('V8 public PPO clip constant differs')
    return path


def resolve_public_artifact(root, relative: str, expected: str) -> Path:
    root = Path(root).resolve()
    original = inside(root, relative)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('invalid artifact SHA-256')
    if relative.endswith('.zip'):
        for row in model_exports(root):
            if row['training_model_sha256'] == expected and relative in row['source_paths']:
                if original.is_file() and sha256(original) != expected:
                    raise ValueError('original V8 archive was modified after export')
                return verify_model_export(root, row)
    if original.is_file() and sha256(original) == expected:
        return original
    source_manifest = root / SOURCE_MANIFEST
    if source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text())
        if manifest.get('schema') != 'shore-bess-v8-public-audit-sources.v1':
            raise ValueError('invalid V8 public audit-source manifest')
        matches = [row for row in manifest['entries'] if row['original_path'] == relative and row['original_sha256'] == expected]
        if len(matches) > 1:
            raise ValueError('ambiguous V8 public audit source identity')
        if matches:
            row = matches[0]
            path = inside(root, row['public_path'])
            if not path.is_relative_to((root / 'evidence/v8/shore_bess/public_audit_sources').resolve()):
                raise ValueError('V8 public audit source is outside its versioned directory')
            if row['public_sha256'] != expected or sha256(path) != expected:
                raise ValueError('V8 public audit source SHA-256 mismatch')
            return path
    if not original.is_file() or sha256(original) != expected:
        raise ValueError('artifact missing or SHA-256 mismatch: ' + relative)
    return original
