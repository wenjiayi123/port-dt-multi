from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ContainerContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
