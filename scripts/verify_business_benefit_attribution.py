"""Build a new versioned business-benefit attribution evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.business_benefit_attribution import BusinessBenefitAttributionService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing benefit attribution evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify pre-registered paired business-benefit attribution with actual execution and measured outcomes.",
    )
    parser.add_argument("--input", required=True, help="Business-benefit attribution dataset JSON")
    parser.add_argument("--output", required=True, help="New versioned evidence JSON; existing files are never overwritten")
    parser.add_argument("--authorized-source-attested", action="store_true", help="Attest that execution, metering and comparator records are authorized site evidence")
    parser.add_argument("--business-owner-approved-by", default="", help="Independent business-owner reviewer")
    parser.add_argument("--operations-assurance-approved-by", default="", help="Independent operations-assurance reviewer")
    parser.add_argument("--causal-methods-approved-by", default="", help="Independent causal-methods reviewer")
    parser.add_argument("--change-ticket", default="", help="Approved field-benefit review ticket")
    args = parser.parse_args(argv)
    approval_values = (
        args.business_owner_approved_by,
        args.operations_assurance_approved_by,
        args.causal_methods_approved_by,
        args.change_ticket,
    )
    if any(approval_values) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(approval_values) and not all(approval_values):
        parser.error("site approval requires all three independent reviewers and a change ticket")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    result = BusinessBenefitAttributionService().run(
        payload,
        source_verified=args.authorized_source_attested,
        business_owner_approved_by=args.business_owner_approved_by,
        operations_assurance_approved_by=args.operations_assurance_approved_by,
        causal_methods_approved_by=args.causal_methods_approved_by,
        change_ticket=args.change_ticket,
    )
    if not result["valid"]:
        print(json.dumps({"valid": False, "errors": result["errors"]}, ensure_ascii=False, indent=2))
        return 2
    _write_new(Path(args.output), result["evidence"])
    evidence = result["evidence"]
    primary_id = evidence["design"]["primary_metric"]
    primary = next(row for row in evidence["metric_summaries"] if row["metric_id"] == primary_id)
    print(json.dumps({
        "valid": True,
        "run_id": evidence["run_id"],
        "dataset_sha256": evidence["dataset_sha256"],
        "pair_effects_digest": evidence["pair_effects_digest"],
        "evidence_digest": evidence["evidence_digest"],
        "complete_pair_count": evidence["data_quality"]["complete_pair_count"],
        "primary_metric": primary_id,
        "primary_relative_effect_percent": primary["mean_benefit_relative_percent"],
        "primary_ci95_low_percent": primary["ci95_relative_percent"]["low"],
        "attribution_status": evidence["attribution_status"],
        "approved": evidence["approved"],
        "field_kpi_claim_eligible": evidence["boundary"]["field_kpi_claim_eligible"],
        "production_authority": evidence["boundary"]["production_authority"],
        "output": Path(args.output).name,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
