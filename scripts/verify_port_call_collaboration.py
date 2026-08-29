"""Build a new, versioned port-call collaboration evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.port_call_collaboration import PortCallCollaborationService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing collaboration evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Propagate port-call delays, resolve shared-resource conflicts and verify six-party receipts.",
    )
    parser.add_argument("--input", required=True, help="Port-call collaboration dataset JSON")
    parser.add_argument("--output", required=True, help="New versioned evidence JSON; existing files are never overwritten")
    parser.add_argument(
        "--authorized-source-attested",
        action="store_true",
        help="Attest that the timeline, resources, identities, disruptions and receipts came from an authorized site export",
    )
    parser.add_argument("--terminal-operations-approved-by", default="", help="Independent terminal-operations reviewer")
    parser.add_argument("--port-authority-approved-by", default="", help="Independent port-authority reviewer")
    parser.add_argument("--change-ticket", default="", help="Approved port-call collaboration change ticket")
    args = parser.parse_args(argv)

    approval_values = (
        args.terminal_operations_approved_by,
        args.port_authority_approved_by,
        args.change_ticket,
    )
    if any(approval_values) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(approval_values) and not all(approval_values):
        parser.error("site approval requires both independent reviewers and a change ticket")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    result = PortCallCollaborationService().run(
        payload,
        source_verified=args.authorized_source_attested,
        terminal_operations_approved_by=args.terminal_operations_approved_by,
        port_authority_approved_by=args.port_authority_approved_by,
        change_ticket=args.change_ticket,
    )
    if not result["valid"]:
        print(json.dumps({"valid": False, "errors": result["errors"]}, ensure_ascii=False, indent=2))
        return 2
    _write_new(Path(args.output), result["evidence"])
    evidence = result["evidence"]
    print(json.dumps({
        "valid": True,
        "run_id": evidence["run_id"],
        "dataset_sha256": evidence["dataset_sha256"],
        "evidence_digest": evidence["evidence_digest"],
        "collaboration_status": evidence["collaboration_status"],
        "participant_count": evidence["metrics"]["participant_count"],
        "conflicts_before_replan": evidence["metrics"]["conflicts_before_replan"],
        "conflicts_after_replan": evidence["metrics"]["conflicts_after_replan"],
        "unresolved_objection_count": evidence["metrics"]["unresolved_objection_count"],
        "approved": evidence["approved"],
        "production_authority": evidence["boundary"]["production_authority"],
        "output": Path(args.output).name,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
