"""Build a new, versioned site execution-acceptance artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.site_execution_acceptance import SiteExecutionAcceptanceService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing execution evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an actuator config and a fixed site commissioning matrix without dispatching commands.",
    )
    parser.add_argument("--input", required=True, help="Execution commissioning bundle JSON")
    parser.add_argument("--output", required=True, help="New versioned evidence JSON; existing files are never overwritten")
    parser.add_argument("--authorized-source-attested", action="store_true", help="Attest that the input is an authorized site commissioning export")
    parser.add_argument("--operations-approved-by", default="", help="Independent terminal-operations reviewer")
    parser.add_argument("--maritime-safety-approved-by", default="", help="Independent maritime-safety reviewer")
    parser.add_argument("--controls-engineering-approved-by", default="", help="Independent controls-engineering reviewer")
    parser.add_argument("--change-ticket", default="", help="Approved execution-release change ticket")
    args = parser.parse_args(argv)

    approvals = (
        args.operations_approved_by,
        args.maritime_safety_approved_by,
        args.controls_engineering_approved_by,
        args.change_ticket,
    )
    if any(approvals) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(approvals) and not all(approvals):
        parser.error("site execution approval requires all three reviewers and a change ticket")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    result = SiteExecutionAcceptanceService().run(
        payload,
        source_verified=args.authorized_source_attested,
        operations_approved_by=args.operations_approved_by,
        maritime_safety_approved_by=args.maritime_safety_approved_by,
        controls_engineering_approved_by=args.controls_engineering_approved_by,
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
        "actuator_config_sha256": evidence["actuator_config_sha256"],
        "evidence_digest": evidence["evidence_digest"],
        "acceptance_status": evidence["acceptance_status"],
        "site_commissioning_verified": evidence["site_commissioning_verified"],
        "approved": evidence["approved"],
        "production_authority": evidence["boundary"]["production_authority"],
        "output": Path(args.output).name,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
