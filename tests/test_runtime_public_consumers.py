"""Real public SAC ZIP inference through MAS/execution with no raw archive.

Only the run selection and southbound transport are fixtures. Network weights,
export manifest/SHA verification, canonical state, policy inference and software
safety envelope are the actual implementations.
"""
import hashlib
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.adapters.actuators import CommandResult, PortSouthboundGateway
from app.services import mas_evidence
from app.services.execution import api as execution_api
from app.services.rl_training import runtime_policy
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.trainer import TRAINING_MANAGER, TrainingManager

ROOT = Path(__file__).resolve().parents[1]
JOB = "rl-20260813T064228701Z"
RUN = f"data/rl/runs/{JOB}"
EXPORT_MANIFEST = "evidence/public_models/legacy_v3_v6_20260912/manifest.json"


@contextmanager
def public_only_run():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        run = root / RUN
        run.mkdir(parents=True)
        for name in ("config.json", "status.json", "manifest.json"):
            shutil.copyfile(ROOT / RUN / name, run / name)
        manifest = json.loads((ROOT / EXPORT_MANIFEST).read_text())
        row = next(row for row in manifest["models"] if RUN + "/model.zip" in row["source_paths"])
        public = root / row["public_model_path"]
        public.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / row["public_model_path"], public)
        (root / EXPORT_MANIFEST).write_text(json.dumps({**manifest, "models": [row]}))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
        manager = TrainingManager(data_root=TRAINING_MANAGER.data_root, run_root=run.parent,
                                  benchmark_path=root / "data/rl/benchmarks.json")
        service = mas_evidence.MASEvidenceService()
        dataset = load_port_dataset(service.dataset_id, manager.data_root)
        state = service._canonical_state(dataset, service._row_index(dataset, "dense"))
        with patch.object(runtime_policy, "ROOT", root), \
             patch.object(manager, "_load_policy", side_effect=AssertionError("raw model loader invoked")):
            yield manager, service, state, public
        assert not (run / "model.zip").exists()
        assert not (root / "data/rl/model_registry.json").exists()
        assert not manager.benchmark_path.exists()
        for p, digest in before.items():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, str(p)


class RuntimePublicConsumerTests(unittest.TestCase):
    def test_mas_real_public_sac_inference_with_raw_absent(self):
        with public_only_run() as (manager, service, state, _public):
            # Fixed selection isolates the consumer from the multi-run selector;
            # the existing V3 endpoint test exercises that selector in the clone.
            with patch.object(mas_evidence, "TRAINING_MANAGER", manager), \
                 patch.object(service, "_sac_evidence", return_value=({"metrics": {}}, JOB)):
                result = service.build(scenario="dense")
            direct = runtime_policy.predict_runtime(manager, JOB, {"state": state})
            self.assertTrue(result["available"])
            self.assertEqual(result["decision"]["decoded_control"], direct["decoded_control"])
            self.assertEqual(result["decision"]["safety_envelope"]["status"], "pass")
            self.assertFalse(result["production_authority"])
            self.assertFalse(result["decision"]["rendered"])

    def test_rl_stage_public_inference_keeps_gateway_and_two_person_gate(self):
        with public_only_run() as (manager, _service, state, public):
            app = FastAPI()
            app.include_router(execution_api.router)
            client = TestClient(app)
            payload = {"job_id": JOB, "state": state, "control_field": "bess_kw",
                       "setpoint_parameter": "power_kw", "asset_id": "fixture-bess",
                       "asset_type": "bess", "action": "set", "requested_by": "fixture-requester"}
            disabled = public.parent / "disabled-actuator.json"
            disabled.write_text(json.dumps({"enabled": False, "reason": "fixture fail closed"}))
            gateway = PortSouthboundGateway(str(disabled))
            with patch.object(execution_api, "TRAINING_MANAGER", manager), \
                 patch.object(execution_api, "gateway", gateway):
                response = client.post("/api/actuators/rl-stage", json=payload)
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["message"], "actuator_gateway_disabled")
                self.assertEqual(response.json()["recommendation"]["algorithm"], "sac")
                pending = CommandResult("PENDING", "fixture-command", "fixture-bess", "guard", "human confirmation required")
                with patch.object(gateway, "dispatch", return_value=pending) as dispatch:
                    response = client.post("/api/actuators/rl-stage", json=payload)
                self.assertEqual(response.status_code, 202)
                command = dispatch.call_args.args[0]
                self.assertTrue(command.two_channel_required)
                self.assertEqual(command.model_version, JOB)
                self.assertEqual(command.requested_by, "fixture-requester")
                self.assertEqual(command.constraints_check["source"], "verified_rl_recommendation")
                self.assertEqual(command.parameters["power_kw"], response.json()["recommendation"]["decoded_control"]["bess_kw"])
