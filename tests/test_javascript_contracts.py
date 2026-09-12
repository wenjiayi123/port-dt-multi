"""Run every JavaScript VM contract under the normal unittest/CI entry point.

These are deterministic DOM/VM contracts, not browser acceptance evidence.
Node is a required test dependency: absence must never silently skip coverage.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
# This separately documented optional browser test needs Playwright + a browser.
# Its deterministic navigation contracts are also covered by the VM suite.
OPTIONAL_BROWSER_SCRIPTS = {"standalone_navigation.cjs"}


class JavascriptContractTests(unittest.TestCase):
    def test_all_javascript_ui_contracts(self):
        node = os.environ.get("NODE_BINARY") or shutil.which("node")
        self.assertTrue(node, "Node.js is required for UI contracts; install Node or set NODE_BINARY to its executable.")
        scripts = sorted(
            script for script in {*ROOT.glob("tests/js/*.cjs"), *ROOT.glob("tests/js/*.test.js")}
            if script.name not in OPTIONAL_BROWSER_SCRIPTS
        )
        self.assertTrue(scripts, "No JavaScript UI contracts were found.")
        for script in scripts:
            with self.subTest(script=script.relative_to(ROOT).as_posix()):
                try:
                    result = subprocess.run(
                        [str(node), str(script)], cwd=ROOT, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        timeout=90, check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.fail(f"Cannot run {script.name}: {exc}")
                self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
