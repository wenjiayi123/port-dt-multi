from __future__ import annotations

import unittest

from scripts.audit_ui_controls_v6 import audit


class UiControlInventoryV6Tests(unittest.TestCase):
    def test_every_static_button_definition_has_an_interaction_contract(self):
        result = audit()
        self.assertEqual(result["status"], "PASS", result["unresolved"])
        self.assertGreaterEqual(result["static_button_definition_count"], 275)
        self.assertEqual(result["unresolved_button_definition_count"], 0)
        self.assertTrue(result["boundary"]["runtime_browser_acceptance_required"])


if __name__ == "__main__":
    unittest.main()
