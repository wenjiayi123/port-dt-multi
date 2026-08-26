"""Build a new, versioned site-shadow acceptance artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.site_shadow_acceptance import SiteShadowAcceptanceService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing shadow evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify consecutive read-only site-shadow cycles against fixed acceptance gates.",
    )
    parser.add_argument("--input", required=True, help="Shadow observation bundle JSON")
    parser.add_argument("--output", required=True, help="New versioned evidence JSON; existing files are never overwritten")
    parser.add_argument(
        "--authorized-source-attested",
        action="store_true",
        help="Attest that incumbent outcomes and recommendation receipts came from an authorized site export",
    )
    parser.add_argument("--operations-approved-by", default="", help="Independent port-operations reviewer")
    parser.add_argument("--safety-approved-by", default="", help="Independent maritime-safety reviewer")
    parser.add_argument("--change-ticket", default="", help="Approved deployment or shadow-acceptance ticket")
    parser.add_argument("--rollback-drill-reference", default="", help="Completed rollback drill evidence reference")
    parser.add_argument("--rollback-drill-evidence", default="", help="Actuator rollback evidence JSON containing successful execution and rollback receipts")
    args = parser.parse_args(argv)

    approval_values = (
        args.operations_approved_by,
        args.safety_approved_by,
        args.change_ticket,
        args.rollback_drill_reference,
        args.rollback_drill_evidence,
    )
    if any(approval_values) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(approval_values) and not all(approval_values):
        parser.error("site approval requires both reviewers, a change ticket, a rollback reference, and rollback evidence")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    rollback_drill_evidence = (
        json.loads(Path(args.rollback_drill_evidence).expanduser().resolve().read_text(encoding="utf-8"))
        if args.rollback_drill_evidence
        else None
    )
    result = SiteShadowAcceptanceService().run(
        payload,
        source_verified=args.authorized_source_attested,
        operations_approved_by=args.operations_approved_by,
        safety_approved_by=args.safety_approved_by,
        change_ticket=args.change_ticket,
        rollback_drill_reference=args.rollback_drill_reference,
        rollback_drill_evidence=rollback_drill_evidence,
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
        "acceptance_status": evidence["acceptance_status"],
        "shadow_cycles": evidence["shadow_cycles"],
        "operational_days": evidence["operational_days"],
        "measured_incumbent_baseline": evidence["measured_incumbent_baseline"],
        "approved": evidence["approved"],
        "production_authority": evidence["boundary"]["production_authority"],
        "output": Path(args.output).name,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
