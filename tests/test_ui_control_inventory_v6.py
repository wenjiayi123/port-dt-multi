from __future__ import annotations

import unittest

from scripts.audit_ui_controls_v6 import audit, _collection_listener_reference


class UiControlInventoryV6Tests(unittest.TestCase):
    def test_bounded_collection_requires_membership_and_click_binding(self):
        source = "['rl','twin'].forEach(kind=>$('#btn-'+kind).onclick=()=>read(kind));"
        self.assertTrue(_collection_listener_reference('btn-rl', source))
        self.assertFalse(_collection_listener_reference('btn-other', source))
        self.assertFalse(_collection_listener_reference('btn-rl', "['rl'].forEach(kind=>log(kind));"))
        self.assertFalse(_collection_listener_reference('btn-rl', source.replace('.onclick=', '.textContent=')))

    def test_object_key_binding_resolves_only_declared_view_names(self):
        source = """const views={
          controls:j=>({data:j.controls}),
          claims:j=>({data:j.claims})
        };
        Object.keys(views).forEach(kind=>byId('btn-'+kind)?.addEventListener('click',()=>render(kind)));
        """
        self.assertTrue(_collection_listener_reference('btn-controls', source))
        self.assertFalse(_collection_listener_reference('btn-site', source))
        self.assertFalse(_collection_listener_reference('btn-controls', source.replace('Object.keys(views)', 'Object.keys(unknown)')))

    def test_every_static_button_definition_has_an_interaction_contract(self):
        result = audit()
        self.assertEqual(result["status"], "PASS", result["unresolved"])
        self.assertGreaterEqual(result["static_button_definition_count"], 275)
        self.assertEqual(result["unresolved_button_definition_count"], 0)
        self.assertTrue(result["boundary"]["runtime_browser_acceptance_required"])


if __name__ == "__main__":
    unittest.main()
