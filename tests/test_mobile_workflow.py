from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.mobile_api import api as mobile_api
from app.services.mobile_api.benchmark import build_report, load_verified_report
from app.services.mobile_api.workflow import IdempotencyConflict, MobileWorkflowStore


class MobileWorkflowTests(unittest.TestCase):
    def test_decision_is_idempotent_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MobileWorkflowStore(Path(tmp))
            payload = {
                "target_policy_id": "policy-1",
                "humanChoiceType": "guidance",
                "requested_by": "alice",
            }
            first, first_replayed = store.record_decision(
                payload, "test-idempotency-key-0001"
            )
            second, second_replayed = store.record_decision(
                payload, "test-idempotency-key-0001"
            )
            self.assertFalse(first_replayed)
            self.assertTrue(second_replayed)
            self.assertEqual(first, second)
            self.assertEqual(first["execution_status"], "dry_run_recorded")
            self.assertTrue(store.verify()["valid"])

    def test_conflict_and_unsafe_dispatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MobileWorkflowStore(Path(tmp))
            store.record_decision(
                {"target_policy_id": "a"},
                "test-idempotency-key-0002",
            )
            with self.assertRaises(IdempotencyConflict):
                store.record_decision(
                    {"target_policy_id": "b"},
                    "test-idempotency-key-0002",
                )
            receipt, _ = store.record_decision(
                {
                    "target_policy_id": "a",
                    "production_dispatch": True,
                },
                "test-idempotency-key-0003",
            )
            self.assertFalse(receipt["accepted"])
            self.assertEqual(receipt["execution_status"], "blocked")
            self.assertFalse(receipt["production_dispatch"])

    def test_checked_in_500_operation_report_is_current(self) -> None:
        report = build_report()
        self.assertEqual(report["operations"]["total"], 500)
        self.assertEqual(
            report["results"]["duplicate_suppression_percent"], 100.0
        )
        self.assertEqual(
            report["results"]["unsafe_dispatch_block_percent"], 100.0
        )
        self.assertTrue(report["results"]["audit_chain_valid"])
        self.assertTrue(load_verified_report()["release_gate"]["passed"])

    def test_shared_mobile_api_exposes_evidence_and_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            mobile_api,
            "STORE",
            MobileWorkflowStore(Path(tmp)),
        ):
            application = FastAPI()
            application.include_router(mobile_api.router)
            client = TestClient(application)
            status = client.get("/api/mobile/status")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["backend_id"], "port-dt-multi")
            self.assertEqual(
                status.json()["business_benchmark"]["test_rows"],
                8760,
            )
            self.assertEqual(
                status.json()["mobile_workflow_benchmark"]["operations"],
                500,
            )
            candidates = client.get("/api/mobile/strategy/candidates")
            self.assertEqual(candidates.status_code, 200)
            self.assertGreaterEqual(candidates.json()["count"], 1)
            payload = {
                "target_policy_id": candidates.json()["items"][0]["id"],
                "requested_by": "mobile_operator",
                "production_dispatch": False,
            }
            headers = {"Idempotency-Key": "mobile-api-test-key-0001"}
            first = client.post(
                "/api/mobile/strategy/decisions",
                json=payload,
                headers=headers,
            )
            repeated = client.post(
                "/api/mobile/strategy/decisions",
                json=payload,
                headers=headers,
            )
            self.assertEqual(first.status_code, 202)
            self.assertEqual(repeated.status_code, 202)
            self.assertEqual(
                first.json()["request_id"],
                repeated.json()["request_id"],
            )
            self.assertTrue(repeated.json()["idempotent_replay"])
            receipt = client.get(
                "/api/mobile/strategy/decisions/"
                + first.json()["request_id"]
            )
            self.assertEqual(
                receipt.json()["execution_status"],
                "dry_run_recorded",
            )
            self.assertTrue(client.get("/api/mobile/audit/verify").json()["valid"])

    def test_flutter_training_contract_and_distinct_desktop_approver(self) -> None:
        from app import server

        client = TestClient(server.app)
        baselines = client.get("/api/rl/train/baselines")
        self.assertEqual(baselines.status_code, 200)
        self.assertEqual(
            [item["id"] for item in baselines.json()["items"]],
            ["sac", "ppo", "td3", "dqn", "a2c", "tqc", "mpc"],
        )
        self.assertEqual(
            baselines.json()["dataset"]["dataset_id"],
            "public_port_ops_v1",
        )
        self.assertEqual(
            len(baselines.json()["dataset"]["dataset_sha256"]),
            64,
        )
        created = client.post(
            "/api/rl/train/requests",
            json={
                "source": "dt_mobile_app",
                "requested_by": "same-operator",
                "config": {
                    "algorithm": "ppo",
                    "dataset_id": "public_port_ops_v1",
                    "total_steps": 128,
                },
            },
        )
        request_id = created.json()["request_id"]
        try:
            rejected = client.post(
                f"/api/rl/train/requests/{request_id}/approve",
                json={"approved_by": "same-operator"},
            )
            self.assertEqual(rejected.status_code, 409)
            self.assertIsNone(
                server._RL_MOBILE_TRAIN_REQUESTS[request_id]["job_id"]
            )
        finally:
            server._RL_MOBILE_TRAIN_REQUESTS.pop(request_id, None)


if __name__ == "__main__":
    unittest.main()
