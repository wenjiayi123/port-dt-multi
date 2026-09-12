import base64
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts.public_privacy_scan import archive_findings


class PublicArchivePrivacyTests(unittest.TestCase):
    def test_hidden_base64_code_path_is_detected_without_unpickling(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.zip'
            private='/'.join(['','Users','private-fixture','module.py']).encode()
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('data',json.dumps({'lr_schedule':{':serialized:':base64.b64encode(private).decode()}}))
                z.writestr('policy.pth',b'unchanged weights')
            found=archive_findings(p)
            self.assertEqual(len(found),1)
            self.assertIn('base64: local account path',found[0])

    def test_clean_model_and_corrupt_archive_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('data',json.dumps({'learning_rate':.0003,'clip_range':.2}))
                z.writestr('policy.pth',b'unchanged weights')
            self.assertEqual(archive_findings(p),[])
            p.write_bytes(b'not zip')
            self.assertTrue(archive_findings(p))
