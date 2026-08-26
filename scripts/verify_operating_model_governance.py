"""Create one new versioned operating-model governance evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.operating_model_governance import OperatingModelGovernanceService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing operating-model evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify named responsibility, duty separation, shift coverage and escalation evidence.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorized-source-attested", action="store_true")
    parser.add_argument("--executive-accountability-approved-by", default="")
    parser.add_argument("--governance-assurance-approved-by", default="")
    parser.add_argument("--maritime-safety-approved-by", default="")
    parser.add_argument("--change-ticket", default="")
    args = parser.parse_args(argv)
    values = (args.executive_accountability_approved_by, args.governance_assurance_approved_by, args.maritime_safety_approved_by, args.change_ticket)
    if any(values) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(values) and not all(values):
        parser.error("site approval requires all three reviewers and a change ticket")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    result = OperatingModelGovernanceService().run(
        payload,
        source_verified=args.authorized_source_attested,
        executive_accountability_approved_by=args.executive_accountability_approved_by,
        governance_assurance_approved_by=args.governance_assurance_approved_by,
        maritime_safety_approved_by=args.maritime_safety_approved_by,
        change_ticket=args.change_ticket,
    )
    if not result["valid"]:
        print(json.dumps({"valid": False, "errors": result["errors"]}, ensure_ascii=False, indent=2))
        return 2
    _write_new(Path(args.output), result["evidence"])
    evidence, metrics = result["evidence"], result["evidence"]["metrics"]
    print(json.dumps({"valid": True, "run_id": evidence["run_id"], "dataset_sha256": evidence["dataset_sha256"], "evidence_digest": evidence["evidence_digest"], **metrics, "governance_status": evidence["governance_status"], "approved": evidence["approved"], "organization_authority_verified": evidence["boundary"]["organization_authority_verified"], "system_can_assign_roles": evidence["boundary"]["system_can_assign_roles"], "production_authority": evidence["boundary"]["production_authority"], "output": Path(args.output).name}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
