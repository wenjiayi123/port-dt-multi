"""Build a new, versioned maritime interoperability evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.maritime_interoperability import MaritimeInteroperabilityService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing interoperability evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map port-call, maritime single-window and S-100 catalog data and verify bound conformance reports.",
    )
    parser.add_argument("--input", required=True, help="Maritime interoperability dataset JSON")
    parser.add_argument("--output", required=True, help="New versioned evidence JSON; existing files are never overwritten")
    parser.add_argument(
        "--authorized-source-attested",
        action="store_true",
        help="Attest that the source identities, events, declarations, hydrographic catalogs and reports are authorized site evidence",
    )
    parser.add_argument("--data-governance-approved-by", default="", help="Independent data-governance reviewer")
    parser.add_argument("--maritime-authority-approved-by", default="", help="Independent maritime-authority reviewer")
    parser.add_argument("--hydrographic-authority-approved-by", default="", help="Independent hydrographic-data reviewer")
    parser.add_argument("--change-ticket", default="", help="Approved interoperability change ticket")
    args = parser.parse_args(argv)

    approval_values = (
        args.data_governance_approved_by,
        args.maritime_authority_approved_by,
        args.hydrographic_authority_approved_by,
        args.change_ticket,
    )
    if any(approval_values) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(approval_values) and not all(approval_values):
        parser.error("site approval requires all three independent reviewers and a change ticket")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    result = MaritimeInteroperabilityService().run(
        payload,
        source_verified=args.authorized_source_attested,
        data_governance_approved_by=args.data_governance_approved_by,
        maritime_authority_approved_by=args.maritime_authority_approved_by,
        hydrographic_authority_approved_by=args.hydrographic_authority_approved_by,
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
        "mapping_digest": evidence["mapping_digest"],
        "evidence_digest": evidence["evidence_digest"],
        "dcsa_event_count": evidence["metrics"]["dcsa_event_count"],
        "imo_data_element_count": evidence["metrics"]["imo_data_element_count"],
        "s100_product_count": evidence["metrics"]["s100_product_count"],
        "semantic_gap_count": evidence["metrics"]["semantic_gap_count"],
        "approved": evidence["approved"],
        "authority_submission_allowed": evidence["boundary"]["authority_submission_allowed"],
        "navigational_use_allowed": evidence["boundary"]["navigational_use_allowed"],
        "production_authority": evidence["boundary"]["production_authority"],
        "output": Path(args.output).name,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
