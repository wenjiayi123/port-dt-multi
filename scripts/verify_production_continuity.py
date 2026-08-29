"""Create one new versioned production-continuity evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.production_continuity import ProductionContinuityService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing continuity evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify continuous service, incident closure, restore tests and resilience drills.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorized-source-attested", action="store_true")
    parser.add_argument("--service-owner-approved-by", default="")
    parser.add_argument("--site-reliability-approved-by", default="")
    parser.add_argument("--continuity-cybersecurity-approved-by", default="")
    parser.add_argument("--change-ticket", default="")
    args = parser.parse_args(argv)
    values = (args.service_owner_approved_by, args.site_reliability_approved_by, args.continuity_cybersecurity_approved_by, args.change_ticket)
    if any(values) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(values) and not all(values):
        parser.error("site approval requires all three reviewers and a change ticket")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    result = ProductionContinuityService().run(
        payload,
        source_verified=args.authorized_source_attested,
        service_owner_approved_by=args.service_owner_approved_by,
        site_reliability_approved_by=args.site_reliability_approved_by,
        continuity_cybersecurity_approved_by=args.continuity_cybersecurity_approved_by,
        change_ticket=args.change_ticket,
    )
    if not result["valid"]:
        print(json.dumps({"valid": False, "errors": result["errors"]}, ensure_ascii=False, indent=2))
        return 2
    _write_new(Path(args.output), result["evidence"])
    evidence, metrics = result["evidence"], result["evidence"]["metrics"]
    print(json.dumps({"valid": True, "run_id": evidence["run_id"], "dataset_sha256": evidence["dataset_sha256"], "evidence_digest": evidence["evidence_digest"], **metrics, "continuity_status": evidence["continuity_status"], "approved": evidence["approved"], "field_slo_claim_eligible": evidence["boundary"]["field_slo_claim_eligible"], "automatic_failover_authority": evidence["boundary"]["automatic_failover_authority"], "production_authority": evidence["boundary"]["production_authority"], "output": Path(args.output).name}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
