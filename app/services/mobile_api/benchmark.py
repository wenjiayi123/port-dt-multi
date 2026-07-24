from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .workflow import IdempotencyConflict, MobileWorkflowStore


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "config/mobile_workflow_benchmark_v1.json"
DEFAULT_REPORT = ROOT / "data/mobile/mobile_workflow_benchmark_v1.json"


class DeterministicClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 4, 1, tzinfo=timezone.utc)

    def __call__(self) -> str:
        value = self.current
        self.current += timedelta(milliseconds=1)
        return value.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    per_case = int(config["cases_per_category"])
    total_operations = per_case * 5
    duplicate_passed = 0
    conflict_blocked = 0
    dispatch_blocked = 0
    client_audits = 0
    with tempfile.TemporaryDirectory(prefix="mobile-workflow-benchmark-") as tmp:
        store = MobileWorkflowStore(Path(tmp), clock=DeterministicClock())
        payloads: list[dict[str, Any]] = []
        keys: list[str] = []
        receipts: list[dict[str, Any]] = []
        for index in range(per_case):
            payload = {
                "target_policy_id": f"heldout-policy-{index % 5}",
                "humanChoiceType": "guidance",
                "remark": f"benchmark-case-{index}",
                "requested_by": f"mobile-operator-{index % 7}",
                "production_dispatch": False,
            }
            key = f"mobile-benchmark-idempotency-{index:04d}"
            receipt, replayed = store.record_decision(payload, key)
            if replayed:
                raise AssertionError("first decision cannot be a replay")
            payloads.append(payload)
            keys.append(key)
            receipts.append(receipt)

        for payload, key, expected in zip(payloads, keys, receipts):
            actual, replayed = store.record_decision(payload, key)
            if replayed and actual == expected:
                duplicate_passed += 1

        for index, (payload, key) in enumerate(zip(payloads, keys)):
            conflict = {**payload, "remark": f"changed-payload-{index}"}
            try:
                store.record_decision(conflict, key)
            except IdempotencyConflict:
                conflict_blocked += 1

        for index in range(per_case):
            blocked, replayed = store.record_decision(
                {
                    "target_policy_id": f"heldout-policy-{index % 5}",
                    "humanChoiceType": "override",
                    "requested_by": f"mobile-operator-{index % 7}",
                    "production_dispatch": True,
                },
                f"mobile-benchmark-production-{index:04d}",
            )
            if (
                not replayed
                and blocked["accepted"] is False
                and blocked["execution_status"] == "blocked"
                and blocked["production_dispatch"] is False
            ):
                dispatch_blocked += 1

        for index in range(per_case):
            event = store.append_client_audit(
                {
                    "eventId": f"client-event-{index:04d}",
                    "source": "benchmark_mobile_client",
                    "requestId": receipts[index]["request_id"],
                }
            )
            if event["production_dispatch"] is False:
                client_audits += 1

        verification = store.verify()

    passed = all(
        (
            duplicate_passed == per_case,
            conflict_blocked == per_case,
            dispatch_blocked == per_case,
            client_audits == per_case,
            verification["valid"] is True,
            verification["decision_count"] == per_case * 2,
            verification["event_count"] == per_case * 3,
        )
    )
    return {
        "schema_version": "mobile_workflow_benchmark_v1",
        "benchmark_id": config["benchmark_id"],
        "generated_at": "2026-04-01T00:00:01Z",
        "evidence_level": "deterministic_local_integration_benchmark",
        "config": {
            **config,
            "sha256": _sha256(config_path),
        },
        "operations": {
            "total": total_operations,
            "new_dry_run_decisions": per_case,
            "exact_idempotent_retries": per_case,
            "conflicting_key_reuse_attempts": per_case,
            "unsafe_production_dispatch_attempts": per_case,
            "client_audit_uploads": per_case,
        },
        "results": {
            "duplicate_suppression_percent": 100.0
            * duplicate_passed
            / per_case,
            "conflicting_reuse_block_percent": 100.0
            * conflict_blocked
            / per_case,
            "unsafe_dispatch_block_percent": 100.0
            * dispatch_blocked
            / per_case,
            "client_audit_accept_percent": 100.0
            * client_audits
            / per_case,
            "audit_chain_valid": verification["valid"],
            "audit_event_count": verification["event_count"],
            "unique_decision_receipt_count": verification["decision_count"],
            "production_execution_receipt_count": 0,
        },
        "release_gate": {"passed": passed},
        "boundary": {
            "measures": "shared-backend API workflow semantics and durable local evidence",
            "does_not_measure": [
                "mobile network latency",
                "port field availability",
                "production actuator success",
                "business KPI uplift",
            ],
            "business_kpis_are_system_level": True,
        },
    }


def write_report(
    report_path: Path = DEFAULT_REPORT,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    report = build_report(config_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def load_verified_report(
    report_path: Path = DEFAULT_REPORT,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    checked_in = json.loads(report_path.read_text(encoding="utf-8"))
    rebuilt = build_report(config_path)
    if checked_in != rebuilt:
        raise ValueError("checked-in mobile workflow report is stale")
    if checked_in.get("release_gate", {}).get("passed") is not True:
        raise ValueError("mobile workflow benchmark release gate failed")
    return checked_in

