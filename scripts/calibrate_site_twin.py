"""Build a versioned site-twin calibration evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.site_twin_calibration import SiteTwinCalibrationService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing calibration evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit and independently validate a site power-twin calibration artifact.",
    )
    parser.add_argument("--input", required=True, help="Calibration dataset bundle JSON")
    parser.add_argument("--output", required=True, help="New versioned evidence JSON; existing files are never overwritten")
    parser.add_argument(
        "--authorized-source-attested",
        action="store_true",
        help="Attest that the input is an authorized site export with verified source governance",
    )
    parser.add_argument("--approved-by", default="", help="Independent site model-risk reviewer")
    parser.add_argument("--change-ticket", default="", help="Approved site change or calibration ticket")
    args = parser.parse_args(argv)

    if (args.approved_by or args.change_ticket) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    input_path = Path(args.input).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = SiteTwinCalibrationService().run(
        payload,
        source_verified=args.authorized_source_attested,
        approved_by=args.approved_by,
        change_ticket=args.change_ticket,
    )
    if not result["valid"]:
        print(json.dumps({"valid": False, "errors": result["errors"]}, ensure_ascii=False, indent=2))
        return 2
    _write_new(Path(args.output), result["evidence"])
    evidence = result["evidence"]
    print(
        json.dumps(
            {
                "valid": True,
                "model_version": evidence["model_version"],
                "dataset_sha256": evidence["dataset_sha256"],
                "evidence_digest": evidence["evidence_digest"],
                "validation_status": evidence["validation_status"],
                "measured_outcomes": evidence["measured_outcomes"],
                "approved": evidence["approved"],
                "production_authority": evidence["boundary"]["production_authority"],
                "output": Path(args.output).name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
