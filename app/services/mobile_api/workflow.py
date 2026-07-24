from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


class IdempotencyConflict(ValueError):
    """The same key was reused for a different decision payload."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class MobileWorkflowStore:
    """Durable, fail-closed decision receipts and a SHA-256 audit chain."""

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.root = root
        self.path = root / "mobile_workflow_state_v1.json"
        self._clock = clock
        self._lock = threading.RLock()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "schema_version": "mobile_workflow_state_v1",
            "decisions": {},
            "idempotency": {},
            "audit": [],
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "mobile_workflow_state_v1":
            raise ValueError("unsupported mobile workflow state schema")
        if not all(
            isinstance(raw.get(name), expected)
            for name, expected in (
                ("decisions", dict),
                ("idempotency", dict),
                ("audit", list),
            )
        ):
            raise ValueError("invalid mobile workflow state")
        return raw

    def _write(self, state: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".mobile-workflow-",
            suffix=".json",
            dir=self.root,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    state,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.path)
            os.chmod(self.path, 0o600)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _append_audit(
        state: dict[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        audit = state["audit"]
        previous_hash = (
            str(audit[-1]["event_hash"]) if audit else "0" * 64
        )
        payload = dict(event)
        event_hash = hashlib.sha256(
            f"{previous_hash}:{_canonical(payload)}".encode("utf-8")
        ).hexdigest()
        chained = {
            **payload,
            "sequence": len(audit) + 1,
            "previous_hash": previous_hash,
            "event_hash": event_hash,
        }
        audit.append(chained)
        return chained

    def record_decision(
        self,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        key = idempotency_key.strip()
        if len(key) < 12 or len(key) > 128:
            raise ValueError("Idempotency-Key must contain 12 to 128 characters")
        bounded_payload = dict(payload)
        encoded = _canonical(bounded_payload).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise ValueError("decision payload exceeds 64 KiB")
        fingerprint = hashlib.sha256(encoded).hexdigest()
        with self._lock:
            state = self._read()
            existing = state["idempotency"].get(key)
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise IdempotencyConflict(
                        "Idempotency-Key was already used for another payload"
                    )
                return dict(state["decisions"][existing["request_id"]]), True

            request_id = "decision-" + hashlib.sha256(
                f"{key}:{fingerprint}".encode("utf-8")
            ).hexdigest()[:20]
            production_requested = (
                bounded_payload.get("production_dispatch") is True
            )
            status = "blocked" if production_requested else "dry_run_recorded"
            message = (
                "production dispatch is blocked; use the separately configured "
                "two-person /api/actuators gate"
                if production_requested
                else "human decision recorded as dry-run; no production command was sent"
            )
            receipt = {
                "request_id": request_id,
                "accepted": not production_requested,
                "execution_status": status,
                "production_dispatch": False,
                "message": message,
                "policy_id": str(
                    bounded_payload.get("target_policy_id")
                    or bounded_payload.get("targetPolicyId")
                    or ""
                ),
                "requested_by": str(
                    bounded_payload.get("requested_by")
                    or bounded_payload.get("actor")
                    or "mobile_operator"
                ),
                "updated_at": self._clock(),
            }
            state["decisions"][request_id] = receipt
            state["idempotency"][key] = {
                "fingerprint": fingerprint,
                "request_id": request_id,
            }
            self._append_audit(
                state,
                {
                    "event_id": request_id,
                    "event_type": "production_dispatch_blocked"
                    if production_requested
                    else "human_strategy_decision",
                    "at": receipt["updated_at"],
                    "request_id": request_id,
                    "policy_id": receipt["policy_id"],
                    "requested_by": receipt["requested_by"],
                    "payload_sha256": fingerprint,
                    "execution_status": status,
                    "production_dispatch": False,
                },
            )
            self._write(state)
            return dict(receipt), False

    def get_receipt(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            receipt = self._read()["decisions"].get(request_id)
            if receipt is None:
                raise KeyError(request_id)
            return dict(receipt)

    def append_client_audit(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        encoded = _canonical(payload).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise ValueError("audit payload exceeds 64 KiB")
        with self._lock:
            state = self._read()
            event = self._append_audit(
                state,
                {
                    "event_id": "mobile-audit-"
                    + hashlib.sha256(
                        f"{len(state['audit'])}:{encoded.hex()}".encode("utf-8")
                    ).hexdigest()[:20],
                    "event_type": "mobile_client_audit",
                    "at": self._clock(),
                    "payload_sha256": hashlib.sha256(encoded).hexdigest(),
                    "production_dispatch": False,
                },
            )
            self._write(state)
            return dict(event)

    def verify(self) -> dict[str, Any]:
        with self._lock:
            state = self._read()
        previous_hash = "0" * 64
        first_invalid_sequence: int | None = None
        for index, item in enumerate(state["audit"], 1):
            payload = {
                key: value
                for key, value in item.items()
                if key not in {"sequence", "previous_hash", "event_hash"}
            }
            expected = hashlib.sha256(
                f"{previous_hash}:{_canonical(payload)}".encode("utf-8")
            ).hexdigest()
            if (
                item.get("sequence") != index
                or item.get("previous_hash") != previous_hash
                or item.get("event_hash") != expected
            ):
                first_invalid_sequence = index
                break
            previous_hash = expected
        return {
            "valid": first_invalid_sequence is None,
            "algorithm": "sha256_forward_chain",
            "event_count": len(state["audit"]),
            "decision_count": len(state["decisions"]),
            "idempotency_key_count": len(state["idempotency"]),
            "first_invalid_sequence": first_invalid_sequence,
            "last_event_hash": previous_hash
            if first_invalid_sequence is None
            else None,
        }

