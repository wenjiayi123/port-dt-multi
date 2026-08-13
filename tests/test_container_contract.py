from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ContainerContractTests(unittest.TestCase):
    def test_release_does_not_pin_known_vulnerable_intel_torch(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("torch==2.2.2", requirements)
        self.assertNotIn("stable-baselines3==2.3.2", requirements)
        self.assertNotIn("sb3-contrib==2.3.0", requirements)
        self.assertIn("torch==2.13.0", requirements)

    def test_v3_runtime_and_evidence_are_packaged(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        for required_copy in (
            "COPY --chown=portdt:portdt app ./app",
            "COPY --chown=portdt:portdt data ./data",
            "COPY --chown=portdt:portdt config ./config",
            "COPY --chown=portdt:portdt evidence ./evidence",
            "COPY --chown=portdt:portdt docs ./docs",
            "COPY --chown=portdt:portdt scripts ./scripts",
        ):
            self.assertIn(required_copy, dockerfile)
        self.assertIn("USER portdt", dockerfile)
        self.assertIn("/health/live", dockerfile)
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("PORT_DT_VERSION=3.2.0", workflow)
        self.assertIn("python:3.12-slim@sha256:", dockerfile)
        self.assertNotIn("| python", workflow)
        codeql = (ROOT / ".github/workflows/codeql.yml").read_text(encoding="utf-8")
        top_level = codeql.split("jobs:", 1)[0]
        self.assertNotIn("security-events: write", top_level)
        self.assertIn("security-events: write", codeql.split("jobs:", 1)[1])

    def test_linux_and_ci_dependency_installs_require_hash_locks(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        linux_lock = (ROOT / "requirements-linux.lock").read_text(encoding="utf-8")
        ci_lock = (ROOT / "requirements-ci.lock").read_text(encoding="utf-8")
        self.assertIn("--require-hashes -r requirements-linux.lock", dockerfile)
        self.assertIn("--require-hashes -r requirements-ci.lock", workflow)
        for package in ("torch==2.13.0", "stable-baselines3==2.9.0", "sb3-contrib==2.9.0"):
            self.assertIn(package, linux_lock)
        self.assertIn("pip-audit==2.10.1", ci_lock)
        self.assertGreater(linux_lock.count("--hash=sha256:"), 50)
        self.assertGreater(ci_lock.count("--hash=sha256:"), 50)

    def test_docker_context_keeps_portable_evidence(self) -> None:
        ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        active = {
            line.strip().rstrip("/")
            for line in ignored
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertNotIn("evidence", active)
        self.assertNotIn("evidence/v3", active)
        for required in (
            ROOT / "evidence/v3/runtime/runtime_model.json",
            ROOT / "evidence/v3/shanghai_public_advantage_v3.json",
            ROOT / "evidence/v3/hvac/latest.json",
            ROOT / "app/ui/v3/index.html",
        ):
            self.assertTrue(required.is_file(), required)

    def test_clone_keeps_runtime_dependencies_and_dataset_bytes(self) -> None:
        engine = (
            ROOT / "app/services/rl_model/shore_bess/rl_engine.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("rl_engine_副本", engine)
        self.assertIn("class GaussianPolicy", engine)
        self.assertIn("class MLP", engine)

        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        for dataset in (
            "data/public_sources/shanghai_yangshan_reanalysis_2024_2025.csv",
            "data/public_sources/shanghai_yangshan_reanalysis_2026_01_05.csv",
            "data/rl/datasets/public_cn_sha_forward_2026m05_v1.csv",
            "data/rl/datasets/public_cn_sha_hourly_v3.csv",
        ):
            self.assertIn(f"{dataset} binary", attributes)

        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for archive_allowlist in (
            "!evidence/v3/runtime/selected_sac_v3.zip",
            "!evidence/v3/shore_bess/runs/shore-bess-v3-safe-20260813T015000Z/seed_*/selected_model.zip",
            "!evidence/v3/bess_energy/runs/bess-energy-v3-safe-20260813T043000Z/seed_*/selected_model.zip",
            "!evidence/v3/bess_energy/runs/bess-energy-v32-grid-only-balanced-20260813T090000Z/seed_*/selected_model.zip",
        ):
            self.assertIn(archive_allowlist, gitignore)


if __name__ == "__main__":
    unittest.main()
