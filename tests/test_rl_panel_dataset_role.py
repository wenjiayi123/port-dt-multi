"""HTTP admission fixtures: no optimizer, real dataset or model is opened."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server


class RLPanelDatasetRoleTests(TestCase):
    def test_desktop_rejects_forward_role_even_when_selection_flag_is_absent(self):
        dataset = SimpleNamespace(metadata={"split_policy": {"role": "forward_challenge_only"}})
        with patch.object(server, "load_port_dataset", return_value=dataset), patch.object(server.TRAINING_MANAGER, "start") as start:
            result = TestClient(server.app).post("/api/rl/train/start", json={"config": {"dataset_id": "forward-fixture"}})
        self.assertEqual(result.status_code, 422)
        self.assertIn("禁止用于训练", result.json()["detail"])
        start.assert_not_called()

    def test_mobile_approval_cannot_bypass_candidate_selection_prohibition(self):
        dataset = SimpleNamespace(metadata={"split_policy": {"candidate_selection_allowed": False}})
        request = {"request_id": "role-fixture", "status": "pending", "requested_by": "fixture-mobile", "config": {"dataset_id": "forward-fixture"}}
        with patch.dict(server._RL_MOBILE_TRAIN_REQUESTS, {"role-fixture": request}, clear=True), patch.object(server, "load_port_dataset", return_value=dataset), patch.object(server.TRAINING_MANAGER, "start") as start:
            result = TestClient(server.app).post("/api/rl/train/requests/role-fixture/approve", json={"operator": "fixture-desktop"})
            self.assertEqual(request["status"], "pending")
            self.assertNotIn("job_id", request)
        self.assertEqual(result.status_code, 422)
        start.assert_not_called()

    def test_missing_dataset_fails_before_job_creation(self):
        with patch.object(server, "load_port_dataset", side_effect=FileNotFoundError("fixture absent")), patch.object(server.TRAINING_MANAGER, "start") as start:
            result = TestClient(server.app).post("/api/rl/train/start", json={"config": {"dataset_id": "missing-fixture"}})
        self.assertEqual(result.status_code, 422)
        start.assert_not_called()

    def test_trainable_dataset_reaches_manager_with_reviewed_config(self):
        dataset = SimpleNamespace(metadata={"split_policy": {"candidate_selection_allowed": True}})
        cfg = {"dataset_id": "training-fixture", "algorithm": "sac", "total_steps": 64, "seed": 912}
        with patch.object(server, "load_port_dataset", return_value=dataset), patch.object(server.TRAINING_MANAGER, "start", return_value={"job_id": "fixture-no-training", "status": "QUEUED"}) as start, patch.dict(server._RL_TRAIN_JOBS, {}, clear=True), patch.dict(server._RL_TRAIN_STATUS, {}, clear=True):
            result = TestClient(server.app).post("/api/rl/train/start", json={"config": cfg})
        self.assertEqual(result.status_code, 202)
        self.assertEqual(start.call_count, 1)
        self.assertEqual(start.call_args.args[0], {**cfg, "source": "rl-panel", "approval": None})
