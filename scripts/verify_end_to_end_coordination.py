"""Build one new, versioned full-chain coordination evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.services.end_to_end_coordination import EndToEndCoordinationService


def _write_new(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing end-to-end coordination evidence: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify one freeze-aware, capacity-feasible port-wide rolling coordination recommendation.",
    )
    parser.add_argument("--input", required=True, help="Full-chain planning dataset JSON")
    parser.add_argument("--output", required=True, help="New versioned evidence JSON; existing files are never overwritten")
    parser.add_argument("--authorized-source-attested", action="store_true", help="Attest that the unified plan, capacities and receipts are authorized site evidence")
    parser.add_argument("--integrated-planning-approved-by", default="", help="Independent integrated-planning reviewer")
    parser.add_argument("--marine-services-approved-by", default="", help="Independent marine-services reviewer")
    parser.add_argument("--terminal-operations-approved-by", default="", help="Independent terminal-operations reviewer")
    parser.add_argument("--equipment-energy-approved-by", default="", help="Independent equipment-and-energy reviewer")
    parser.add_argument("--change-ticket", default="", help="Approved coordination review ticket")
    args = parser.parse_args(argv)
    approval_values = (
        args.integrated_planning_approved_by,
        args.marine_services_approved_by,
        args.terminal_operations_approved_by,
        args.equipment_energy_approved_by,
        args.change_ticket,
    )
    if any(approval_values) and not args.authorized_source_attested:
        parser.error("approval metadata requires --authorized-source-attested")
    if any(approval_values) and not all(approval_values):
        parser.error("site approval requires all four independent reviewers and a change ticket")
    payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
    result = EndToEndCoordinationService().run(
        payload,
        source_verified=args.authorized_source_attested,
        integrated_planning_approved_by=args.integrated_planning_approved_by,
        marine_services_approved_by=args.marine_services_approved_by,
        terminal_operations_approved_by=args.terminal_operations_approved_by,
        equipment_energy_approved_by=args.equipment_energy_approved_by,
        change_ticket=args.change_ticket,
    )
    if not result["valid"]:
        print(json.dumps({"valid": False, "errors": result["errors"]}, ensure_ascii=False, indent=2))
        return 2
    _write_new(Path(args.output), result["evidence"])
    evidence = result["evidence"]
    metrics = evidence["metrics"]
    print(json.dumps({
        "valid": True,
        "run_id": evidence["run_id"],
        "dataset_sha256": evidence["dataset_sha256"],
        "baseline_plan_digest": evidence["baseline_plan_digest"],
        "candidate_plan_digest": evidence["candidate_plan_digest"],
        "evidence_digest": evidence["evidence_digest"],
        "chain_count": metrics["chain_count"],
        "task_count": metrics["task_count"],
        "baseline_conflict_slot_count": metrics["baseline_conflict_slot_count"],
        "candidate_conflict_slot_count": metrics["candidate_conflict_slot_count"],
        "maximum_reschedule_minutes": metrics["maximum_reschedule_minutes"],
        "coordination_status": evidence["coordination_status"],
        "approved": evidence["approved"],
        "shared_plan_mutated": evidence["boundary"]["shared_plan_mutated"],
        "automatic_resource_commitment_allowed": evidence["boundary"]["automatic_resource_commitment_allowed"],
        "production_authority": evidence["boundary"]["production_authority"],
        "output": Path(args.output).name,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
