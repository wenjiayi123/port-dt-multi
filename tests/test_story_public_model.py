"""Story replay keeps its identity gates when raw training ZIPs are absent."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from app.services.story_evidence import StoryEvidenceService


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MANIFEST = Path("evidence/public_models/legacy_v3_v6_20260912/manifest.json")
RUNTIME = Path("evidence/v3/runtime")


class StoryPublicModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((ROOT / RUNTIME / "runtime_model.json").read_text())
        cls.original = RUNTIME / cls.metadata["model_artifact"]
        cls.manifest = json.loads((ROOT / PUBLIC_MANIFEST).read_text())
        cls.export = next(
            row for row in cls.manifest["models"]
            if cls.original.as_posix() in row["source_paths"]
            and row["training_model_sha256"] == cls.metadata["model_sha256"]
        )
        source = StoryEvidenceService(ROOT / "data/rl/runs", None)
        selected = source._run_bundle(cls.metadata["job_id"])
        cls.baseline_job = source._compatible_fcfs(selected)["job_id"]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        paths = [
            RUNTIME / "runtime_model.json",
            RUNTIME / self.metadata["config_artifact"],
            PUBLIC_MANIFEST,
            Path(self.export["public_model_path"]),
        ]
        for job in (self.metadata["job_id"], self.baseline_job):
            paths.extend(
                Path("data/rl/runs") / job / name
                for name in ("config.json", "manifest.json", "evaluation.json", "evaluation_trajectory.json")
            )
        for relative in paths:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.service = StoryEvidenceService(self.root / "data/rl/runs", None)
        self.assertFalse((self.root / self.original).exists())

    def summary(self):
        return self.service.summary(hour=0, port="shanghai", replay="sac_vs_fcfs")

    def test_public_only_tree_replays_aligned_frames_and_preserves_original_identity(self):
        result = self.summary()
        self.assertTrue(result["available"])
        self.assertEqual(result["evidence"]["aligned_frames"], 48)
        self.assertEqual(result["evidence"]["baseline_job_id"], self.baseline_job)
        self.assertEqual(result["evidence"]["model_sha256"], self.metadata["model_sha256"])
        self.assertEqual(result["evidence"]["loaded_model_path"], self.export["public_model_path"])
        self.assertEqual(result["evidence"]["loaded_model_sha256"], self.export["public_model_sha256"])
        self.assertNotEqual(result["evidence"]["model_sha256"], result["evidence"]["loaded_model_sha256"])
        self.assertFalse(result["production_authority"])
        self.assertFalse((self.root / self.original).exists())

    def test_corrupted_public_archive_fails_closed(self):
        public = self.root / self.export["public_model_path"]
        public.write_bytes(public.read_bytes() + b"corrupted archive")
        self.assertFalse(self.summary()["available"])

    def test_missing_equivalence_proof_fails_closed(self):
        path = self.root / PUBLIC_MANIFEST
        manifest = json.loads(path.read_text())
        row = next(item for item in manifest["models"] if item["training_model_sha256"] == self.metadata["model_sha256"])
        row["schedules_identical"] = False
        path.write_text(json.dumps(manifest))
        self.assertFalse(self.summary()["available"])

    def test_changed_config_is_not_admitted_by_valid_public_model(self):
        config = self.root / RUNTIME / self.metadata["config_artifact"]
        config.write_bytes(config.read_bytes() + b"\n")
        self.assertNotEqual(hashlib.sha256(config.read_bytes()).hexdigest(), self.metadata["config_sha256"])
        self.assertFalse(self.summary()["available"])

    def test_changed_raw_archive_is_not_hidden_by_valid_public_copy(self):
        (self.root / self.original).write_bytes(b"changed original")
        self.assertFalse(self.summary()["available"])


if __name__ == "__main__":
    unittest.main()
