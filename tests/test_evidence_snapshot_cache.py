from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from app.services.evidence_snapshot_cache import (
    FileBoundEvidenceCache, csv_record_count, energy_evidence_dependencies,
    file_snapshot, verify_report_dataset,
)


class EvidenceSnapshotCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "weights.json"
        self.source.write_text("good")
        self.cache = FileBoundEvidenceCache()
        self.calls = 0

    def build(self):
        self.calls += 1
        return {"admitted": self.source.is_file() and self.source.read_text() == "good", "nested": []}

    def get(self):
        return self.cache.get(lambda: [self.root], self.build)

    def test_same_size_restored_mtime_corruption_invalidates_admission(self):
        self.assertTrue(self.get()["admitted"])
        before = self.source.stat()
        self.source.write_text("evil")
        os.utime(self.source, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertFalse(self.get()["admitted"])
        self.assertEqual(self.calls, 2)

    def test_concurrent_cold_requests_build_once(self):
        entered, release = threading.Event(), threading.Event()
        def slow():
            entered.set()
            self.assertTrue(release.wait(3))
            return self.build()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.cache.get, lambda: [self.root], slow) for _ in range(8)]
            self.assertTrue(entered.wait(3))
            release.set()
            results = [f.result(timeout=3) for f in futures]
        self.assertEqual(self.calls, 1)
        self.assertTrue(all(item["admitted"] for item in results))

    def test_replaced_deleted_and_added_files_invalidate(self):
        self.get()
        replacement = self.root / "replacement"
        replacement.write_text("good")
        replacement.replace(self.source)
        self.get()
        self.source.unlink()
        self.assertFalse(self.get()["admitted"])
        self.source.write_text("good")
        self.assertTrue(self.get()["admitted"])
        (self.root / "new_pointer.json").write_text("{}")
        self.get()
        self.assertEqual(self.calls, 5)

    def test_returned_payload_cannot_poison_next_reader(self):
        first = self.get()
        first["admitted"] = False
        first["nested"].append("polluted")
        self.assertEqual(self.get(), {"admitted": True, "nested": []})
        self.assertEqual(self.calls, 1)

    def test_changed_during_build_is_rejected_and_not_cached(self):
        def changing():
            result = self.build()
            self.source.write_text("evil")
            return result
        with self.assertRaisesRegex(RuntimeError, "changed during verification"):
            self.cache.get(lambda: [self.root], changing)
        self.assertFalse(self.get()["admitted"])
        self.assertEqual(self.calls, 2)

    def test_failed_rebuild_does_not_keep_prior_payload(self):
        self.get()
        self.source.write_text("evil")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.cache.get(lambda: [self.root], Mock(side_effect=ValueError("hash mismatch")))
        self.assertIsNone(self.cache._signature)
        self.assertFalse(self.get()["admitted"])

    def test_ignored_bytecode_does_not_invalidate(self):
        first = file_snapshot([self.root])
        bytecode = self.root / "__pycache__"
        bytecode.mkdir()
        (bytecode / "module.pyc").write_bytes(b"cached")
        self.assertEqual(first, file_snapshot([self.root]))

    def test_dependency_symlink_retargeting_is_observed(self):
        alternative = self.root / "alternate"
        alternative.write_text("evil")
        link = self.root / "pointer"
        link.symlink_to(self.source)
        first = file_snapshot([link])
        link.unlink()
        link.symlink_to(alternative)
        self.assertNotEqual(first, file_snapshot([link]))

    def test_dependency_directory_cannot_escape_repository_by_symlink(self):
        link = self.root / "outside"
        link.symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "escapes repository"):
            file_snapshot([self.root], root=self.root)

    def test_dependency_directory_cycles_are_rejected(self):
        (self.root / "cycle").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "directory cycle"):
            file_snapshot([self.root], root=self.root)

    def test_explicit_absolute_escape_is_rejected_before_scanning(self):
        with self.assertRaisesRegex(RuntimeError, "escapes repository"):
            file_snapshot([Path("/")], root=self.root)

    def test_csv_counts_records_not_newlines_or_dict_allocations(self):
        for content in ["", "a,b\n", 'a,b\n\n1,"two\nlines"\n,\n3,4\n', '\na,b\n\n1,2\n']:
            with self.subTest(content=content):
                self.source.write_text(content)
                self.assertEqual(csv_record_count(self.source), len(list(csv.DictReader(io.StringIO(content)))))
        self.assertEqual(csv_record_count(self.root / "missing"), 0)

    def test_report_dataset_hash_detects_valid_csv_content_change(self):
        report = {"dataset": {"files": [{"path": self.source.name, "sha256": hashlib.sha256(b"good").hexdigest()}]}}
        verify_report_dataset(self.root, report)
        self.source.write_text("evil")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_report_dataset(self.root, report)


class EnergyServiceCacheIntegrationTests(unittest.TestCase):
    def test_discovery_rejects_pointer_and_config_symlinks_before_read(self):
        from app.services.yard_lighting_evidence import YardLightingEvidenceService
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            repo = parent / "repo"
            root = repo / "app/services/rl_model/yard_lighting"
            root.mkdir(parents=True)
            outside = parent / "outside.json"
            outside.write_text('{"report_path":"/"}')
            for relative in ["config/yard_lighting_v3.json", "evidence/v3/yard_lighting/latest.json"]:
                with self.subTest(relative=relative):
                    service = YardLightingEvidenceService(root)
                    link = repo / relative
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(outside)
                    original_read = Path.read_text
                    def checked_read(path, *args, **kwargs):
                        self.assertNotEqual(path, link, "out-of-repository symlink must not be opened")
                        return original_read(path, *args, **kwargs)
                    with patch.object(Path, "read_text", checked_read):
                        with self.assertRaisesRegex(RuntimeError, "escapes repository"):
                            service.build()
                    link.unlink()

    def test_v8_historical_pointer_sha_map_invalidates_external_run_cache(self):
        from app.services.shore_bess_evidence import ShoreBESSEvidenceService
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo / "app/services/rl_model/shore_bess"
            root.mkdir(parents=True)
            service = ShoreBESSEvidenceService(root)
            historical = repo / "evidence/v7/shore_bess/latest.json"
            historical.parent.mkdir(parents=True)
            historical.write_text("good")
            expected = hashlib.sha256(b"good").hexdigest()
            report = repo / "evidence/independent_run/custom_report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"manifest": {"historical_pointer_sha256": {str(historical.relative_to(repo)): expected}}}))
            pointer = repo / "evidence/v8/shore_bess/latest.json"
            pointer.parent.mkdir(parents=True)
            pointer.write_text(json.dumps({"report_path": str(report.relative_to(repo))}))
            count = 0
            def build():
                nonlocal count
                count += 1
                return {"admitted": hashlib.sha256(historical.read_bytes()).hexdigest() == expected}
            service._build_evidence = build
            self.assertTrue(service.build()["admitted"])
            self.assertTrue(service.build()["admitted"])
            historical.write_text("evil")
            self.assertFalse(service.build()["admitted"])
            self.assertEqual(count, 2)

    def test_bad_pointer_cannot_expand_scan_to_filesystem_root(self):
        from app.services.yard_lighting_evidence import YardLightingEvidenceService
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "app/services/rl_model/yard_lighting"
            root.mkdir(parents=True)
            service = YardLightingEvidenceService(root)
            service.v3_evidence.mkdir(parents=True)
            (service.v3_evidence / "latest.json").write_text('{"report_path":"/"}')
            service._build_evidence = Mock()
            with self.assertRaisesRegex(RuntimeError, "escapes repository"):
                service.build()
            service._build_evidence.assert_not_called()

    def test_lighting_real_hash_gates_rerun_on_corruption_after_success(self):
        from app.services import yard_lighting_evidence as module
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo / "app/services/rl_model/yard_lighting"
            root.mkdir(parents=True)
            service = module.YardLightingEvidenceService(root)
            model = root / "selected_model.json"
            model.write_text("good")
            report = {"artifacts": {"models": [{"path": str(model.relative_to(repo)), "sha256": hashlib.sha256(b"good").hexdigest()}]},
                      "quality_gates": {"public_offline_admitted": True},
                      "blind_test": {"sample_real_model_inference": {"policy_loaded": True}}}
            service.v3_evidence.mkdir(parents=True)
            report_path = service.v3_evidence / "report.json"
            report_path.write_text(json.dumps(report))
            pointer = service.v3_evidence / "latest.json"
            pointer.write_text(json.dumps({"report_path": str(report_path.relative_to(repo)), "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}))
            env = Mock()
            env.reset.return_value = ([0.0], {})
            env.step.return_value = (None, 0, False, False, {})
            def verified_build():
                _, current = service._formal()
                result = service._inference(current)
                return {"quality_gates": {"admitted": result["policy_loaded"]}, "inference": result}
            service._build_evidence = verified_build
            with patch.object(module, "load_v3_config", return_value={"training": {"episode_steps": 1}}), \
                 patch.object(module, "load_v3_dataset"), \
                 patch.object(module, "chronological_slices", return_value=(slice(0, 1), slice(1, 2), slice(2, 3))), \
                 patch.object(module, "YardLightingV3Env", return_value=env), \
                 patch.object(module.NumpyMLPPolicy, "load") as loader:
                loader.return_value.predict.return_value = [0.0]
                self.assertTrue(service.build()["quality_gates"]["admitted"])
                self.assertTrue(service.build()["quality_gates"]["admitted"])
                before = model.stat()
                model.write_text("evil")
                os.utime(model, ns=(before.st_atime_ns, before.st_mtime_ns))
                corrupt = service.build()
                self.assertFalse(corrupt["quality_gates"]["admitted"])
                self.assertIn("hash gate", corrupt["inference"]["error"])
                self.assertEqual(loader.call_count, 1)
                pointer.write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "hash gate"):
                    service.build()

    def test_saved_success_cannot_override_missing_model_failure(self):
        from app.services.hvac_evidence import HVACEvidenceService
        from app.services.shore_bess_evidence import ShoreBESSEvidenceService
        from app.services.bess_energy_evidence import BESSEnergyEvidenceService
        from app.services.yard_crane_evidence import YardCraneEvidenceService
        from app.services.yard_lighting_evidence import YardLightingEvidenceService
        report = {"blind_test": {"sample_real_model_inference": {"policy_loaded": True}}}
        for klass, method in [(HVACEvidenceService, "_current_v3_inference"), (ShoreBESSEvidenceService, "_current_v3_inference"),
                              (BESSEnergyEvidenceService, "_current_v3_inference"), (YardCraneEvidenceService, "_current_inference"),
                              (YardLightingEvidenceService, "_inference")]:
            with self.subTest(service=klass.__name__):
                self.assertFalse(getattr(klass(), method)(report)["policy_loaded"])

    def test_complete_dependencies_cover_previously_omitted_inputs_and_v8(self):
        from app.services.shore_bess_evidence import ShoreBESSEvidenceService
        from app.services.bess_energy_evidence import BESSEnergyEvidenceService
        from app.services.yard_lighting_evidence import YardLightingEvidenceService
        for service, relative in [(YardLightingEvidenceService(), "app/services/rl_model/yard_lighting/data/weather_astro.csv"),
                                  (ShoreBESSEvidenceService(), "evidence/v8/shore_bess/latest.json"),
                                  (BESSEnergyEvidenceService(), "app/services/rl_model/shore_bess/v8_public_artifacts.py")]:
            snapshot = dict(file_snapshot(energy_evidence_dependencies(service)))
            self.assertIn(str(service.repo_root / relative), snapshot)
            self.assertIn(str(service.repo_root / "config" / f"{service.v3_evidence.name}_v3.json"), snapshot)


if __name__ == "__main__":
    unittest.main()
