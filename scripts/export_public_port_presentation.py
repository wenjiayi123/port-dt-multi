"""Export the read-only presentation contract for review or adapter development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.public_port_demo.service import asset_series, replacement_contract, snapshot
from app.services.public_port_demo.workspace import workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSON output file; existing files are preserved")
    parser.add_argument("--at", default="2025-12-15T12:00:00+08:00", help="fixed historical scenario timestamp, including timezone")
    parser.add_argument("--review-output", type=Path, help="optional separate business report and completeness review JSON")
    args = parser.parse_args()
    for output in (args.output, args.review_output):
        if output is not None and output.exists():
            raise FileExistsError(f"Existing output preserved: {output}")
    if args.review_output and args.review_output.resolve() == args.output.resolve():
        raise ValueError("snapshot and review outputs must be different files")
    data = snapshot(args.at)
    data["asset_series"] = {g["id"]: asset_series(g["id"], args.at) for g in data.get("asset_groups", [])}
    data["adapter_contract"] = replacement_contract()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Presentation snapshot exported: {args.output.resolve()}")
    if args.review_output:
        with args.review_output.open("x", encoding="utf-8") as stream:
            json.dump(workspace(args.at), stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"Business data review exported: {args.review_output.resolve()}")


if __name__ == "__main__":
    main()
