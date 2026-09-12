from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scripts import profile_startup_loading as profiler


class StartupLoadingProfileTests(unittest.TestCase):
    def test_fixed_homepage_subset_is_unique_and_read_only(self):
        self.assertEqual(len(profiler.HOME), 16)
        self.assertEqual(len(set(profiler.HOME)), 16)
        self.assertEqual(profiler.HOME[-5:], profiler.ENERGY)
        self.assertIn("/api/system/provenance", profiler.HOME)
        self.assertFalse(any("evaluate" in path or "train" in path for path in profiler.HOME))

    def test_sandbox_is_optional_on_non_macos(self):
        args = ["python", "-B", "-m", "uvicorn"]
        with patch.object(profiler.sys, "platform", "linux"):
            self.assertEqual(profiler.sandbox_command(args, Path("/repo"), False), args)
            with self.assertRaisesRegex(RuntimeError, "macOS"):
                profiler.sandbox_command(args, Path("/repo"), True)

    def test_http_error_keeps_status_bytes_and_semantic_failure(self):
        body = '{"status":"证据待补","available":false}'.encode()
        error = HTTPError("http://127.0.0.1/test", 422, "Unprocessable", {"Content-Type": "application/json"}, io.BytesIO(body))
        with patch.object(profiler, "urlopen", side_effect=error):
            row = profiler.fetch("http://127.0.0.1", "/test", time.perf_counter())
        self.assertEqual(row["status"], 422)
        self.assertEqual(row["response_bytes"], len(body))
        self.assertFalse(row["semantic_flags"]["available"])

    def test_network_timeout_does_not_become_a_success(self):
        with patch.object(profiler, "urlopen", side_effect=TimeoutError("bounded request")):
            row = profiler.fetch("http://127.0.0.1", "/test", time.perf_counter())
        self.assertIsNone(row["status"])
        self.assertIn("TimeoutError", row["error"])
        self.assertEqual(row["response_bytes"], 0)

    def test_source_inventory_binds_untracked_code_and_excludes_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            files = {"app/server.py": "old", "app/static/new_loading.js": "new",
                     "data/runtime.json": "runtime", "evidence/run/config.json": "artifact",
                     "requirements.txt": "httpx", "scripts/profile.py": "script"}
            for rel, content in files.items():
                path = repo / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            with patch.object(profiler, "command", return_value="\0".join(files)) as git:
                before = profiler.source_inventory(repo)
                (repo / "app/static/new_loading.js").write_text("changed untracked source")
                after = profiler.source_inventory(repo)
            git.assert_called_with(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], repo)
            self.assertEqual(set(before["files_sha256"]), {
                "app/server.py", "app/static/new_loading.js", "requirements.txt", "scripts/profile.py"})
            self.assertNotEqual(before["inventory_sha256"], after["inventory_sha256"])

    def test_profile_branch_cannot_return_success_after_checkout_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo, output = root / "repo", root / "output"
            repo.mkdir()
            output.mkdir()
            (output / "provenance_components.json").write_text('{}')
            args = ["profiler", "--repo", str(repo), "--output", str(output), "--phase", "provenance-profile"]
            with patch.object(profiler.sys, "argv", args), \
                 patch.object(profiler, "metadata", return_value={}), \
                 patch.object(profiler, "checkout_state", side_effect=[{"head": "before"}, {"head": "after"}]), \
                 patch.object(profiler.subprocess, "run"):
                with self.assertRaisesRegex(RuntimeError, "checkout state changed"):
                    profiler.main()
            report = json.loads((output / "provenance-profile.json").read_text())
            self.assertFalse(report["checkout_unchanged"])
            self.assertIn("profile", report)

    def test_font_cache_state_distinguishes_first_import_and_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            before = profiler.font_cache_state(output)
            self.assertFalse(before["directory_exists"])
            self.assertEqual(before["fontlist_files"], {})
            directory = output / "matplotlib"
            directory.mkdir()
            cache = directory / "fontlist-v-test.json"
            cache.write_text('{"font":"existing system font"}')
            created = profiler.font_cache_state(output)
            reused = profiler.font_cache_state(output)
            self.assertEqual(created, reused)
            self.assertEqual(set(created["fontlist_files"]), {cache.name})
            (output / "energy_server.log").write_text("Matplotlib is building the font cache; this may take a moment.")
            report = {}
            with patch.object(profiler, "checkout_state", return_value={"head": "fixed"}):
                profiler.finalize_report(output, output / "energy.json", report, {"head": "fixed"})
            self.assertTrue(report["font_cache_rebuild_logged"])
            self.assertEqual(report["font_cache_after_phase"], created)


if __name__ == "__main__":
    unittest.main()
