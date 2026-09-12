"""Resolve original training archives to verified public metadata-only exports."""
from __future__ import annotations

import json
from pathlib import Path

from .datasets import file_sha256


def resolve_model_artifact(root, relative, training_sha256):
    root = Path(root).resolve()
    def within(value):
        path = (root / value).resolve()
        if root not in path.parents:
            raise ValueError("model artifact must stay within repository")
        return path
    original = within(relative)
    legacy_directory = "evidence/public_models/legacy_v3_v6_20260912"
    legacy_manifest = root / legacy_directory / "manifest.json"
    if legacy_manifest.is_file():
        from app.services.rl_model.shore_bess.v8_public_artifacts import verify_model_export
        payload = json.loads(legacy_manifest.read_text())
        if (payload.get("schema") != "port-sb3-legacy-public-metadata-export.v1"
                or payload.get("source_model_archives_modified") is not False
                or payload.get("historical_reports_modified") is not False):
            raise ValueError("invalid legacy public-model manifest")
        rows = [row for row in payload["models"] if row["training_model_sha256"] == training_sha256
                and relative in row["source_paths"]]
        if len(rows) > 1:
            raise ValueError("ambiguous legacy public-model identity")
        if rows:
            if original.is_file() and file_sha256(original) != training_sha256:
                raise ValueError("original legacy model bytes changed")
            return verify_model_export(root, rows[0], export_directory=legacy_directory)
    manifest = root / "evidence/v7/public_models/manifest.json"
    if manifest.is_file():
        exports = json.loads(manifest.read_text())["models"]
        exported = next((r for r in exports if r["training_model_sha256"] == training_sha256 and relative in r["source_paths"]), None)
        if exported is not None:
            path = within(exported["public_model_path"])
            if (not exported.get("weights_and_optimizer_members_identical")
                    or not exported.get("inference_identical")
                    or not exported.get("schedules_identical")
                    or file_sha256(path) != exported["public_model_sha256"]):
                raise ValueError("public model export integrity or equivalence check failed")
            return path
    if file_sha256(original) != training_sha256:
        raise ValueError("business RL model hash mismatch")
    return original
