"""Build a new versioned forecast uncertainty evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.forecast_uncertainty import ForecastUncertaintyService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing forecast evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit eight port forecast targets, calibrate intervals and evaluate a chronological holdout window.",
    )
    parser.add_argument("--input", required=True, help="Forecast uncertainty dataset JSON")
    parser.add_argument("--output", required=True, help="New versioned evidence JSON; existing files are never overwritten")
    parser.add_argument("--authorized-source-attested", action="store_true", help="Attest that features, issued forecasts and measured outcomes are authorized site evidence")
    parser.add_argument("--operations-planning-approved-by", default="", help="Independent operations-planning reviewer")
    parser.add_argument("--model-risk-approved-by", default="", help="Independent model-risk reviewer")
    parser.add_argument("--maritime-safety-approved-by", default="", help="Independent maritime-safety reviewer")
    parser.add_argument("--change-ticket", default="", help="Approved forecast service change ticket")
    args = parser.parse_args(argv)
    approval_values = (
        args.operations_planning_approved_by,
        args.model_risk_approved_by,
        args.maritime_safety_approved_by,
        args.change_ticket,
    )
    if any(approval_values) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(approval_values) and not all(approval_values):
        parser.error("site approval requires all three independent reviewers and a change ticket")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    result = ForecastUncertaintyService().run(
        payload,
        source_verified=args.authorized_source_attested,
        operations_planning_approved_by=args.operations_planning_approved_by,
        model_risk_approved_by=args.model_risk_approved_by,
        maritime_safety_approved_by=args.maritime_safety_approved_by,
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
        "model_bundle_digest": evidence["model_bundle_digest"],
        "test_receipts_digest": evidence["test_receipts_digest"],
        "evidence_digest": evidence["evidence_digest"],
        "target_count": len(evidence["metrics_by_target"]),
        "calibration_status": evidence["calibration_status"],
        "approved": evidence["approved"],
        "dispatch_allowed": evidence["boundary"]["dispatch_allowed"],
        "production_authority": evidence["boundary"]["production_authority"],
        "output": Path(args.output).name,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
