from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATASET_SCHEMA = "port_call_collaboration_dataset.v1"
EVIDENCE_SCHEMA = "port_call_collaboration_evidence.v1"
EVIDENCE_CLASSES = ("authorized_site_collaboration_export", "contract_test_only")
REQUIRED_ROLES = (
    "shipping_line",
    "vessel_agent",
    "terminal",
    "pilotage",
    "towage",
    "port_authority",
)
ROLE_LABELS = {
    "shipping_line": "船公司",
    "vessel_agent": "船舶代理",
    "terminal": "码头",
    "pilotage": "引航机构",
    "towage": "拖轮机构",
    "port_authority": "港口管理机构",
}
RESOURCE_TYPES = ("berth", "pilot", "tug", "mooring", "channel")
MILESTONE_SEQUENCE = (
    "port_arrival",
    "pilotage_start",
    "berth_start",
    "cargo_start",
    "cargo_complete",
    "berth_complete",
    "port_departure",
)
REPLAN_BUFFER_MINUTES = 15
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IMO = re.compile(r"^[0-9]{7}$")
_PLACEHOLDERS = {"unknown", "unset", "todo", "replace", "replace_me", "n/a", "none"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _parse_timestamp(value: Any) -> datetime:
    text = _clean(value)
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp requires an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_digest(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _is_named(value: Any) -> bool:
    text = _clean(value)
    return bool(text and text.lower() not in _PLACEHOLDERS)


def _timeline_conflicts(
    rows: List[Dict[str, Any]], resources: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    resource_ids = {row["resource_id"] for row in resources if row.get("capacity") == 1}
    usages: Dict[str, List[Dict[str, Any]]] = {resource_id: [] for resource_id in resource_ids}
    for row in rows:
        duration = int(row.get("duration_minutes") or 0)
        if duration <= 0:
            continue
        start = _parse_timestamp(row["planned_at"])
        for resource_id in row.get("resource_ids") or []:
            if resource_id in usages:
                usages[resource_id].append({
                    **row,
                    "_start": start,
                    "_end": start + timedelta(minutes=duration),
                })
    conflicts: List[Dict[str, Any]] = []
    for resource_id, items in sorted(usages.items()):
        items.sort(key=lambda row: (row["_start"], row["port_call_id"], row["milestone_id"]))
        for left_index, left in enumerate(items):
            for right in items[left_index + 1:]:
                if right["_start"] >= left["_end"]:
                    break
                if left["port_call_id"] == right["port_call_id"]:
                    continue
                overlap_start = max(left["_start"], right["_start"])
                overlap_end = min(left["_end"], right["_end"])
                if overlap_end <= overlap_start:
                    continue
                conflicts.append({
                    "conflict_id": "CONFLICT." + _canonical_digest([
                        resource_id,
                        left["milestone_id"],
                        right["milestone_id"],
                        _iso(overlap_start),
                    ])[:16].upper(),
                    "resource_id": resource_id,
                    "left_port_call_id": left["port_call_id"],
                    "left_milestone_id": left["milestone_id"],
                    "right_port_call_id": right["port_call_id"],
                    "right_milestone_id": right["milestone_id"],
                    "overlap_start": _iso(overlap_start),
                    "overlap_end": _iso(overlap_end),
                    "overlap_minutes": int((overlap_end - overlap_start).total_seconds() // 60),
                })
    return conflicts


def _metrics(evidence: Dict[str, Any]) -> Dict[str, Any]:
    participants = evidence.get("participants") if isinstance(evidence.get("participants"), list) else []
    baseline = evidence.get("baseline_timeline") if isinstance(evidence.get("baseline_timeline"), list) else []
    proposed = evidence.get("proposed_timeline") if isinstance(evidence.get("proposed_timeline"), list) else []
    baseline_by_id = {row.get("milestone_id"): row for row in baseline if isinstance(row, dict)}
    responses = evidence.get("responses") if isinstance(evidence.get("responses"), list) else []
    acknowledged = {
        row.get("participant_id")
        for row in responses
        if isinstance(row, dict) and row.get("disposition") == "acknowledged"
    }
    objections = [
        row for row in responses
        if isinstance(row, dict) and row.get("disposition") == "objected"
    ]
    delays: List[int] = []
    affected = 0
    for row in proposed:
        base = baseline_by_id.get(row.get("milestone_id"))
        if not base:
            continue
        delay = int((_parse_timestamp(row["planned_at"]) - _parse_timestamp(base["planned_at"])).total_seconds() // 60)
        delays.append(delay)
        if delay > 0:
            affected += 1
    return {
        "required_participant_count": len(REQUIRED_ROLES),
        "participant_count": len(participants),
        "participant_coverage_rate": len({row.get("role") for row in participants}) / len(REQUIRED_ROLES),
        "port_call_count": len({row.get("port_call_id") for row in baseline}),
        "milestone_count": len(baseline),
        "affected_milestone_count": affected,
        "max_propagated_delay_minutes": max(delays, default=0),
        "conflicts_before_replan": len(evidence.get("conflicts_before_replan") or []),
        "conflicts_after_replan": len(evidence.get("conflicts_after_replan") or []),
        "replan_action_count": len(evidence.get("replan_actions") or []),
        "acknowledgement_coverage_rate": len(acknowledged) / len(REQUIRED_ROLES),
        "unresolved_objection_count": len(objections),
    }


class PortCallCollaborationService:
    """Validate a shared port-call timeline and produce recommendation-only replan evidence."""

    @staticmethod
    def _error(
        errors: List[Dict[str, Any]], code: str, field: str, message: str, *, row_index: int | None = None
    ) -> None:
        item: Dict[str, Any] = {"code": code, "field": field, "message": message}
        if row_index is not None:
            item["row_index"] = row_index
        errors.append(item)

    @staticmethod
    def validate_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {"valid": False, "errors": ["collaboration evidence must be an object"], "production_gate_eligible": False}
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        for field in (
            "site_id", "run_id", "dataset_sha256", "port_call_event_digest", "proposal_digest", "source",
            "participants", "resources", "baseline_timeline", "disrupted_timeline",
            "proposed_timeline", "disruptions", "conflicts_before_replan",
            "replan_actions", "responses", "metrics", "provenance", "boundary", "evidence_digest",
        ):
            if payload.get(field) in (None, "", [], {}):
                if field not in {"conflicts_before_replan", "replan_actions", "responses"}:
                    errors.append("missing field: " + field)
        for field in ("dataset_sha256", "port_call_event_digest", "proposal_digest"):
            if not _SHA256.fullmatch(_clean(payload.get(field))):
                errors.append(f"{field} must be a lowercase SHA-256 digest")
        participants = payload.get("participants") if isinstance(payload.get("participants"), list) else []
        roles = [row.get("role") for row in participants if isinstance(row, dict)]
        if Counter(roles) != Counter(REQUIRED_ROLES):
            errors.append("participants must contain exactly one record for every required role")
        resources = payload.get("resources") if isinstance(payload.get("resources"), list) else []
        disrupted = payload.get("disrupted_timeline") if isinstance(payload.get("disrupted_timeline"), list) else []
        proposed = payload.get("proposed_timeline") if isinstance(payload.get("proposed_timeline"), list) else []
        expected_proposal_digest = _canonical_digest({
            "site_id": payload.get("site_id"),
            "run_id": payload.get("run_id"),
            "collaboration_revision": payload.get("collaboration_revision"),
            "port_call_event_digest": payload.get("port_call_event_digest"),
            "disruptions": payload.get("disruptions"),
            "proposed_timeline": proposed,
            "replan_actions": payload.get("replan_actions"),
        })
        if payload.get("proposal_digest") != expected_proposal_digest:
            errors.append("proposal_digest does not bind the proposed timeline and replan actions")
        responses = payload.get("responses") if isinstance(payload.get("responses"), list) else []
        participant_ids = {row.get("participant_id") for row in participants if isinstance(row, dict)}
        response_ids: set[str] = set()
        for index, row in enumerate(responses):
            if not isinstance(row, dict):
                errors.append(f"responses[{index}] must be an object")
                continue
            participant_id = row.get("participant_id")
            if participant_id not in participant_ids or participant_id in response_ids:
                errors.append(f"responses[{index}] participant must be declared and unique")
            response_ids.add(participant_id)
            if row.get("revision") != payload.get("collaboration_revision"):
                errors.append(f"responses[{index}] revision does not match the collaboration revision")
            if row.get("proposal_digest") != expected_proposal_digest:
                errors.append(f"responses[{index}] does not bind the proposed timeline digest")
        if payload.get("conflicts_before_replan") != _timeline_conflicts(disrupted, resources):
            errors.append("conflicts_before_replan does not match the disrupted resource timeline")
        if payload.get("conflicts_after_replan") != _timeline_conflicts(proposed, resources):
            errors.append("conflicts_after_replan does not match the proposed resource timeline")
        derived_metrics = _metrics(payload)
        if payload.get("metrics") != derived_metrics:
            errors.append("metrics do not match timeline, conflicts and participant responses")
        expected_status = (
            "accepted"
            if derived_metrics["conflicts_after_replan"] == 0
            and derived_metrics["acknowledgement_coverage_rate"] == 1.0
            and derived_metrics["unresolved_objection_count"] == 0
            else "needs_revision"
        )
        if payload.get("collaboration_status") != expected_status:
            errors.append("collaboration_status does not match conflicts and responses")
        digest_payload = dict(payload)
        provided_digest = _clean(digest_payload.pop("evidence_digest", ""))
        if _canonical_digest(digest_payload) != provided_digest:
            errors.append("evidence_digest does not match collaboration evidence content")
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        if (
            boundary.get("recommendation_only") is not True
            or boundary.get("shared_plan_mutated") is not False
            or boundary.get("dispatch_allowed") is not False
            or boundary.get("authority_to_change_eta") is not False
            or boundary.get("production_authority") is not False
        ):
            errors.append("collaboration evidence must remain recommendation-only without plan mutation or production authority")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        approvers = payload.get("approved_by") if isinstance(payload.get("approved_by"), dict) else {}
        operations = _clean(approvers.get("terminal_operations"))
        authority = _clean(approvers.get("port_authority"))
        owner = _clean(source.get("owner")).lower()
        all_authorized = bool(participants) and all(row.get("authorized") is True for row in participants)
        approval_contract = bool(
            payload.get("approved") is True
            and source.get("evidence_class") == "authorized_site_collaboration_export"
            and provenance.get("source_attestation") is True
            and provenance.get("change_ticket")
            and all_authorized
            and operations
            and authority
            and operations.lower() != authority.lower()
            and operations.lower() != owner
            and authority.lower() != owner
            and expected_status == "accepted"
            and boundary.get("site_collaboration_accepted") is True
            and boundary.get("dual_approval_verified") is True
        )
        if payload.get("approved") is True and not approval_contract:
            errors.append("approved collaboration evidence requires an attested site export, six authorized parties, clean replan, complete acknowledgements and two independent reviewers")
        return {
            "valid": not errors,
            "errors": errors,
            "production_gate_eligible": bool(not errors and approval_contract),
            "evidence_type": "port_call_collaboration_evidence_v1",
        }

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_PORT_CALL_COLLABORATION_PATH"))
        artifact: Dict[str, Any] = {
            "mode": "unconfigured",
            "configured": False,
            "verified": False,
            "production_gate_eligible": False,
            "artifact_id": None,
            "sha256": None,
            "blockers": ["port_call_collaboration_artifact_not_configured"],
        }
        if raw_path:
            path = Path(raw_path).expanduser()
            artifact.update(
                mode="configured_invalid",
                configured=True,
                artifact_id=path.name,
                blockers=["port_call_collaboration_artifact_invalid"],
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validation = self.validate_evidence(payload)
                artifact.update(
                    mode="verified_site_artifact" if validation["production_gate_eligible"] else "configured_invalid",
                    verified=bool(validation["valid"]),
                    production_gate_eligible=bool(validation["production_gate_eligible"]),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    site_id=payload.get("site_id"),
                    run_id=payload.get("run_id"),
                    collaboration_status=payload.get("collaboration_status"),
                    participant_count=(payload.get("metrics") or {}).get("participant_count"),
                    conflicts_after_replan=(payload.get("metrics") or {}).get("conflicts_after_replan"),
                    unresolved_objection_count=(payload.get("metrics") or {}).get("unresolved_objection_count"),
                    approved=payload.get("approved") is True,
                    blockers=[] if validation["production_gate_eligible"] else list(validation["errors"]),
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        accepted = bool(artifact.get("production_gate_eligible"))
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "configured_artifact": artifact,
            "contract": {
                "required_participants": [
                    {"role": role, "label": ROLE_LABELS[role]} for role in REQUIRED_ROLES
                ],
                "timeline_phases": ["预计", "请求", "计划", "实际"],
                "required_milestones": list(MILESTONE_SEQUENCE),
                "resource_types": list(RESOURCE_TYPES),
                "delay_propagation": "dependency_order_with_original_time_gaps",
                "conflict_resolution": f"lower_priority_shift_with_{REPLAN_BUFFER_MINUTES}_minute_buffer",
                "required_evidence": [
                    "标准靠泊事件证据摘要",
                    "六方授权身份与来源引用",
                    "统一计划版本、依赖关系与资源占用窗",
                    "延误事实、影响传播和重排前后冲突",
                    "绑定候选重排摘要的六方确认或异议回执",
                    "码头运营与港口管理机构独立审批",
                ],
            },
            "boundary": {
                "site_collaboration_accepted": accepted,
                "recommendation_only": True,
                "shared_plan_mutated": False,
                "dispatch_allowed": False,
                "authority_to_change_eta": False,
                "production_authority": False,
                "site_status": "现场靠泊协同证据已验证" if accepted else "待接入现场六方协同证据",
                "reason": (
                    "六方授权身份、事件绑定、延误传播、无冲突重排、完整回执和独立审批均已验证；每次真实计划变更仍由业务系统和责任人确认。"
                    if accepted
                    else "合同样例不能替代现场协同；必须接入授权统一时间线、资源占用、延误事实、六方回执和独立审批。"
                ),
            },
        }

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        terminal_operations_approved_by: str | None = None,
        port_authority_approved_by: str | None = None,
        change_ticket: str | None = None,
    ) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        if not isinstance(payload, dict):
            payload = {}
            self._error(errors, "dataset_type", "$", "payload must be a JSON object")
        if _clean(payload.get("schema_version")) != DATASET_SCHEMA:
            self._error(errors, "schema_version", "schema_version", f"must equal {DATASET_SCHEMA}")
        site_id = _clean(payload.get("site_id"))
        run_id = _clean(payload.get("run_id"))
        revision = payload.get("collaboration_revision")
        for field, value in (("site_id", site_id), ("run_id", run_id)):
            if not _IDENTIFIER.fullmatch(value):
                self._error(errors, "identifier", field, f"{field} must be a stable identifier")
        try:
            revision = int(revision)
            if revision < 1:
                raise ValueError
        except (TypeError, ValueError):
            self._error(errors, "revision", "collaboration_revision", "collaboration_revision must be a positive integer")
            revision = 0
        port_call_event_digest = _clean(payload.get("port_call_event_digest"))
        if not _SHA256.fullmatch(port_call_event_digest):
            self._error(errors, "event_digest", "port_call_event_digest", "must bind a lowercase SHA-256 port-call event digest")

        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        for field in ("source_system", "owner", "license"):
            if not _is_named(source.get(field)):
                self._error(errors, "source_metadata", f"source.{field}", f"source.{field} is required and cannot be a placeholder")
        evidence_class = _clean(source.get("evidence_class"))
        if evidence_class not in EVIDENCE_CLASSES:
            self._error(errors, "evidence_class", "source.evidence_class", f"must be one of {EVIDENCE_CLASSES}")
        try:
            ZoneInfo(_clean(source.get("timezone")))
        except ZoneInfoNotFoundError:
            self._error(errors, "timezone", "source.timezone", "source.timezone must be an IANA timezone")
        try:
            extracted_at = _parse_timestamp(source.get("extracted_at"))
        except (TypeError, ValueError):
            extracted_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
            self._error(errors, "timestamp", "source.extracted_at", "source.extracted_at requires an explicit timezone")

        participants = payload.get("participants") if isinstance(payload.get("participants"), list) else []
        normalized_participants: List[Dict[str, Any]] = []
        participant_ids: set[str] = set()
        roles: List[str] = []
        for index, row in enumerate(participants):
            if not isinstance(row, dict):
                self._error(errors, "participant_type", "participants", "participant must be an object", row_index=index)
                continue
            participant_id = _clean(row.get("participant_id"))
            role = _clean(row.get("role"))
            if not _IDENTIFIER.fullmatch(participant_id) or participant_id in participant_ids:
                self._error(errors, "participant_id", "participants.participant_id", "participant_id must be stable and unique", row_index=index)
            if role not in REQUIRED_ROLES:
                self._error(errors, "participant_role", "participants.role", "participant role is not in the fixed six-party contract", row_index=index)
            if not _is_named(row.get("organization")) or not _is_named(row.get("source_reference")):
                self._error(errors, "participant_source", "participants", "organization and source_reference are required", row_index=index)
            participant_ids.add(participant_id)
            roles.append(role)
            normalized_participants.append({
                "participant_id": participant_id,
                "role": role,
                "role_label": ROLE_LABELS.get(role, role),
                "organization": _clean(row.get("organization")),
                "source_reference": _clean(row.get("source_reference")),
                "authorized": row.get("authorized") is True,
            })
        if Counter(roles) != Counter(REQUIRED_ROLES):
            self._error(errors, "participant_coverage", "participants", "exactly one participant is required for each of the six roles")

        resources = payload.get("resources") if isinstance(payload.get("resources"), list) else []
        normalized_resources: List[Dict[str, Any]] = []
        resource_ids: set[str] = set()
        resource_types: set[str] = set()
        for index, row in enumerate(resources):
            if not isinstance(row, dict):
                self._error(errors, "resource_type", "resources", "resource must be an object", row_index=index)
                continue
            resource_id = _clean(row.get("resource_id"))
            resource_type = _clean(row.get("resource_type"))
            if not _IDENTIFIER.fullmatch(resource_id) or resource_id in resource_ids:
                self._error(errors, "resource_id", "resources.resource_id", "resource_id must be stable and unique", row_index=index)
            if resource_type not in RESOURCE_TYPES:
                self._error(errors, "resource_kind", "resources.resource_type", "resource_type is outside the fixed contract", row_index=index)
            if row.get("capacity") != 1:
                self._error(errors, "resource_capacity", "resources.capacity", "each operational resource must represent one capacity unit", row_index=index)
            if not _is_named(row.get("owner_participant_id")) or row.get("owner_participant_id") not in participant_ids:
                self._error(errors, "resource_owner", "resources.owner_participant_id", "resource owner must be a declared participant", row_index=index)
            resource_ids.add(resource_id)
            resource_types.add(resource_type)
            normalized_resources.append({
                "resource_id": resource_id,
                "resource_type": resource_type,
                "capacity": 1,
                "owner_participant_id": _clean(row.get("owner_participant_id")),
                "source_reference": _clean(row.get("source_reference")),
            })
        for required_type in ("berth", "pilot", "tug"):
            if required_type not in resource_types:
                self._error(errors, "resource_coverage", "resources", f"at least one {required_type} resource is required")

        port_calls = payload.get("port_calls") if isinstance(payload.get("port_calls"), list) else []
        if len(port_calls) < 2:
            self._error(errors, "port_call_count", "port_calls", "at least two port calls are required to test shared-resource conflicts")
        baseline: List[Dict[str, Any]] = []
        call_meta: Dict[str, Dict[str, Any]] = {}
        milestone_to_call: Dict[str, str] = {}
        call_milestones: Dict[str, List[str]] = {}
        for call_index, call in enumerate(port_calls):
            if not isinstance(call, dict):
                self._error(errors, "port_call_type", "port_calls", "port call must be an object", row_index=call_index)
                continue
            call_id = _clean(call.get("port_call_id"))
            if not _IDENTIFIER.fullmatch(call_id) or call_id in call_meta:
                self._error(errors, "port_call_id", "port_calls.port_call_id", "port_call_id must be stable and unique", row_index=call_index)
            imo = _clean(call.get("vessel_imo"))
            if not _IMO.fullmatch(imo):
                self._error(errors, "vessel_imo", "port_calls.vessel_imo", "vessel_imo must contain seven digits", row_index=call_index)
            try:
                priority = int(call.get("priority"))
                if not 1 <= priority <= 9:
                    raise ValueError
            except (TypeError, ValueError):
                priority = 9
                self._error(errors, "priority", "port_calls.priority", "priority must be between one and nine", row_index=call_index)
            call_meta[call_id] = {"priority": priority, "vessel_imo": imo, "vessel_name": _clean(call.get("vessel_name"))}
            milestones = call.get("milestones") if isinstance(call.get("milestones"), list) else []
            event_types = [row.get("event_type") for row in milestones if isinstance(row, dict)]
            if event_types != list(MILESTONE_SEQUENCE):
                self._error(errors, "milestone_sequence", "port_calls.milestones", "milestones must follow the fixed arrival-to-departure sequence", row_index=call_index)
            previous_id: str | None = None
            call_milestones[call_id] = []
            previous_at: datetime | None = None
            for milestone_index, row in enumerate(milestones):
                if not isinstance(row, dict):
                    continue
                milestone_id = _clean(row.get("milestone_id"))
                if not _IDENTIFIER.fullmatch(milestone_id) or milestone_id in milestone_to_call:
                    self._error(errors, "milestone_id", "port_calls.milestones.milestone_id", "milestone_id must be stable and globally unique", row_index=milestone_index)
                dependencies = row.get("depends_on") if isinstance(row.get("depends_on"), list) else []
                expected_dependencies = [] if previous_id is None else [previous_id]
                if dependencies != expected_dependencies:
                    self._error(errors, "dependency_chain", "port_calls.milestones.depends_on", "each milestone must depend on the preceding milestone", row_index=milestone_index)
                try:
                    planned_at = _parse_timestamp(row.get("planned_at"))
                    if previous_at and planned_at < previous_at:
                        raise ValueError("non-monotonic")
                except (TypeError, ValueError):
                    planned_at = extracted_at
                    self._error(errors, "milestone_time", "port_calls.milestones.planned_at", "planned_at must be timezone-aware and non-decreasing", row_index=milestone_index)
                try:
                    duration = int(row.get("duration_minutes"))
                    if not 0 <= duration <= 2880:
                        raise ValueError
                except (TypeError, ValueError):
                    duration = 0
                    self._error(errors, "duration", "port_calls.milestones.duration_minutes", "duration_minutes must be between zero and 2880", row_index=milestone_index)
                assigned_resources = row.get("resource_ids") if isinstance(row.get("resource_ids"), list) else []
                if any(resource_id not in resource_ids for resource_id in assigned_resources):
                    self._error(errors, "resource_reference", "port_calls.milestones.resource_ids", "milestone references an undeclared resource", row_index=milestone_index)
                if not _is_named(row.get("source_reference")):
                    self._error(errors, "milestone_source", "port_calls.milestones.source_reference", "source_reference is required", row_index=milestone_index)
                baseline.append({
                    "milestone_id": milestone_id,
                    "port_call_id": call_id,
                    "vessel_imo": imo,
                    "vessel_name": _clean(call.get("vessel_name")),
                    "priority": priority,
                    "event_type": _clean(row.get("event_type")),
                    "planned_at": _iso(planned_at),
                    "duration_minutes": duration,
                    "depends_on": dependencies,
                    "resource_ids": list(assigned_resources),
                    "source_reference": _clean(row.get("source_reference")),
                })
                milestone_to_call[milestone_id] = call_id
                call_milestones[call_id].append(milestone_id)
                previous_id = milestone_id
                previous_at = planned_at

        disruptions = payload.get("disruptions") if isinstance(payload.get("disruptions"), list) else []
        if not disruptions:
            self._error(errors, "disruption_count", "disruptions", "at least one measured delay disruption is required")
        normalized_disruptions: List[Dict[str, Any]] = []
        disruption_ids: set[str] = set()
        for index, row in enumerate(disruptions):
            if not isinstance(row, dict):
                self._error(errors, "disruption_type", "disruptions", "disruption must be an object", row_index=index)
                continue
            disruption_id = _clean(row.get("disruption_id"))
            call_id = _clean(row.get("port_call_id"))
            milestone_id = _clean(row.get("milestone_id"))
            if not _IDENTIFIER.fullmatch(disruption_id) or disruption_id in disruption_ids:
                self._error(errors, "disruption_id", "disruptions.disruption_id", "disruption_id must be stable and unique", row_index=index)
            if milestone_to_call.get(milestone_id) != call_id:
                self._error(errors, "disruption_target", "disruptions.milestone_id", "disruption must target a declared milestone in the same port call", row_index=index)
            try:
                delay = int(row.get("delay_minutes"))
                if not 1 <= delay <= 1440:
                    raise ValueError
            except (TypeError, ValueError):
                delay = 0
                self._error(errors, "delay", "disruptions.delay_minutes", "delay_minutes must be between one and 1440", row_index=index)
            try:
                occurred_at = _parse_timestamp(row.get("occurred_at"))
            except (TypeError, ValueError):
                occurred_at = extracted_at
                self._error(errors, "disruption_time", "disruptions.occurred_at", "occurred_at requires an explicit timezone", row_index=index)
            if not _is_named(row.get("source_reference")):
                self._error(errors, "disruption_source", "disruptions.source_reference", "source_reference is required", row_index=index)
            disruption_ids.add(disruption_id)
            normalized_disruptions.append({
                "disruption_id": disruption_id,
                "port_call_id": call_id,
                "milestone_id": milestone_id,
                "delay_minutes": delay,
                "occurred_at": _iso(occurred_at),
                "reason": _clean(row.get("reason")),
                "source_reference": _clean(row.get("source_reference")),
            })

        responses = payload.get("responses") if isinstance(payload.get("responses"), list) else []
        normalized_responses: List[Dict[str, Any]] = []
        response_participants: set[str] = set()
        for index, row in enumerate(responses):
            if not isinstance(row, dict):
                self._error(errors, "response_type", "responses", "response must be an object", row_index=index)
                continue
            participant_id = _clean(row.get("participant_id"))
            if participant_id not in participant_ids or participant_id in response_participants:
                self._error(errors, "response_participant", "responses.participant_id", "each declared participant must have exactly one response", row_index=index)
            if row.get("revision") != revision:
                self._error(errors, "response_revision", "responses.revision", "response must bind the current collaboration revision", row_index=index)
            disposition = _clean(row.get("disposition"))
            if disposition not in {"acknowledged", "objected"}:
                self._error(errors, "response_disposition", "responses.disposition", "disposition must be acknowledged or objected", row_index=index)
            if disposition == "objected" and not _is_named(row.get("reason")):
                self._error(errors, "objection_reason", "responses.reason", "an objection requires a reason", row_index=index)
            try:
                responded_at = _parse_timestamp(row.get("responded_at"))
            except (TypeError, ValueError):
                responded_at = extracted_at
                self._error(errors, "response_time", "responses.responded_at", "responded_at requires an explicit timezone", row_index=index)
            if not _is_named(row.get("source_reference")):
                self._error(errors, "response_source", "responses.source_reference", "source_reference is required", row_index=index)
            proposal_digest = _clean(row.get("proposal_digest"))
            if not _SHA256.fullmatch(proposal_digest):
                self._error(errors, "response_proposal_digest", "responses.proposal_digest", "response must bind a lowercase SHA-256 proposal digest", row_index=index)
            response_participants.add(participant_id)
            normalized_responses.append({
                "participant_id": participant_id,
                "revision": revision,
                "proposal_digest": proposal_digest,
                "disposition": disposition,
                "reason": _clean(row.get("reason")),
                "responded_at": _iso(responded_at),
                "source_reference": _clean(row.get("source_reference")),
            })
        if errors:
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "evidence": None,
                "boundary": {
                    "site_collaboration_accepted": False,
                    "recommendation_only": True,
                    "shared_plan_mutated": False,
                    "dispatch_allowed": False,
                    "authority_to_change_eta": False,
                    "production_authority": False,
                },
            }

        disrupted = deepcopy(baseline)
        disrupted_by_id = {row["milestone_id"]: row for row in disrupted}
        for disruption in normalized_disruptions:
            call_ids = call_milestones[disruption["port_call_id"]]
            start_index = call_ids.index(disruption["milestone_id"])
            for milestone_id in call_ids[start_index:]:
                row = disrupted_by_id[milestone_id]
                row["planned_at"] = _iso(
                    _parse_timestamp(row["planned_at"]) + timedelta(minutes=disruption["delay_minutes"])
                )
                row.setdefault("propagated_disruption_ids", []).append(disruption["disruption_id"])

        conflicts_before = _timeline_conflicts(disrupted, normalized_resources)
        proposed = deepcopy(disrupted)
        proposed_by_id = {row["milestone_id"]: row for row in proposed}
        replan_actions: List[Dict[str, Any]] = []
        for iteration in range(32):
            conflicts = _timeline_conflicts(proposed, normalized_resources)
            if not conflicts:
                break
            conflict = conflicts[0]
            left = proposed_by_id[conflict["left_milestone_id"]]
            right = proposed_by_id[conflict["right_milestone_id"]]
            if left["priority"] > right["priority"]:
                target, other = left, right
            elif right["priority"] > left["priority"]:
                target, other = right, left
            else:
                target, other = right, left
            target_start = _parse_timestamp(target["planned_at"])
            other_end = _parse_timestamp(other["planned_at"]) + timedelta(minutes=int(other["duration_minutes"]))
            shift_minutes = max(
                REPLAN_BUFFER_MINUTES,
                int((other_end - target_start).total_seconds() // 60) + REPLAN_BUFFER_MINUTES,
            )
            affected_ids = call_milestones[target["port_call_id"]]
            start_index = affected_ids.index(target["milestone_id"])
            for milestone_id in affected_ids[start_index:]:
                row = proposed_by_id[milestone_id]
                row["planned_at"] = _iso(_parse_timestamp(row["planned_at"]) + timedelta(minutes=shift_minutes))
                row.setdefault("replan_action_ids", []).append(f"REPLAN.{iteration + 1:03d}")
            replan_actions.append({
                "action_id": f"REPLAN.{iteration + 1:03d}",
                "action_type": "shift_resource_window",
                "port_call_id": target["port_call_id"],
                "from_milestone_id": target["milestone_id"],
                "resource_id": conflict["resource_id"],
                "shift_minutes": shift_minutes,
                "buffer_minutes": REPLAN_BUFFER_MINUTES,
                "reason": f"resolve {conflict['conflict_id']}",
                "recommendation_only": True,
            })
        conflicts_after = _timeline_conflicts(proposed, normalized_resources)
        proposal_digest = _canonical_digest({
            "site_id": site_id,
            "run_id": run_id,
            "collaboration_revision": revision,
            "port_call_event_digest": port_call_event_digest,
            "disruptions": normalized_disruptions,
            "proposed_timeline": proposed,
            "replan_actions": replan_actions,
        })
        stale_responses = [
            row for row in normalized_responses if row["proposal_digest"] != proposal_digest
        ]
        if stale_responses:
            return {
                "valid": False,
                "errors": [{
                    "code": "stale_proposal_response",
                    "field": "responses.proposal_digest",
                    "message": "every response must bind the exact proposed timeline and replan-action digest",
                }],
                "warnings": warnings,
                "evidence": None,
                "boundary": {
                    "site_collaboration_accepted": False,
                    "recommendation_only": True,
                    "shared_plan_mutated": False,
                    "dispatch_allowed": False,
                    "authority_to_change_eta": False,
                    "production_authority": False,
                },
            }

        owner = _clean(source.get("owner"))
        operations_approver = _clean(terminal_operations_approved_by)
        authority_approver = _clean(port_authority_approved_by)
        approvals_valid = bool(
            operations_approver
            and authority_approver
            and operations_approver.lower() != authority_approver.lower()
            and operations_approver.lower() != owner.lower()
            and authority_approver.lower() != owner.lower()
        )
        all_authorized = all(row["authorized"] for row in normalized_participants)
        all_acknowledged = bool(
            response_participants == participant_ids
            and all(row["disposition"] == "acknowledged" for row in normalized_responses)
        )
        source_attested = bool(source_verified and evidence_class == "authorized_site_collaboration_export")
        approved = bool(
            source_attested
            and all_authorized
            and all_acknowledged
            and not conflicts_after
            and approvals_valid
            and _is_named(change_ticket)
        )
        if evidence_class == "contract_test_only":
            warnings.append({
                "code": "contract_only",
                "message": "合同数据仅验证六方时间线、延误传播与重排算法，不能证明现场协同或修改真实计划。",
            })
        elif not source_verified:
            warnings.append({"code": "source_attestation_missing", "message": "授权现场导出尚未由离线验证流程确认。"})
        if normalized_responses and not all_acknowledged:
            warnings.append({"code": "unresolved_objection", "message": "当前协同版本仍有异议，必须重新协商后生成新版本。"})
        if source_attested and not approvals_valid:
            warnings.append({"code": "dual_approval_invalid", "message": "现场协同准入需要独立的码头运营与港口管理机构审批。"})

        evidence: Dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "collaboration_revision": revision,
            "dataset_sha256": _canonical_digest(payload),
            "port_call_event_digest": port_call_event_digest,
            "proposal_digest": proposal_digest,
            "source": {
                "source_system": _clean(source.get("source_system")),
                "owner": owner,
                "license": _clean(source.get("license")),
                "timezone": _clean(source.get("timezone")),
                "extracted_at": _iso(extracted_at),
                "evidence_class": evidence_class,
            },
            "participants": normalized_participants,
            "resources": normalized_resources,
            "baseline_timeline": baseline,
            "disrupted_timeline": disrupted,
            "proposed_timeline": proposed,
            "disruptions": normalized_disruptions,
            "conflicts_before_replan": conflicts_before,
            "conflicts_after_replan": conflicts_after,
            "replan_actions": replan_actions,
            "responses": normalized_responses,
            "metrics": {},
            "collaboration_status": "needs_revision",
            "approved": approved,
            "approved_by": {
                "terminal_operations": operations_approver or None,
                "port_authority": authority_approver or None,
            },
            "provenance": {
                "source_attestation": source_attested,
                "change_ticket": _clean(change_ticket) or None,
                "algorithm": "dependency-delay-propagation-and-capacity-one-resource-replan-v1",
                "generated_at": _iso(extracted_at),
            },
            "boundary": {
                "site_collaboration_accepted": approved,
                "recommendation_only": True,
                "shared_plan_mutated": False,
                "dispatch_allowed": False,
                "authority_to_change_eta": False,
                "production_authority": False,
                "dual_approval_verified": approvals_valid,
            },
        }
        evidence["metrics"] = _metrics(evidence)
        evidence["collaboration_status"] = (
            "accepted"
            if evidence["metrics"]["conflicts_after_replan"] == 0
            and evidence["metrics"]["acknowledgement_coverage_rate"] == 1.0
            and evidence["metrics"]["unresolved_objection_count"] == 0
            else "needs_revision"
        )
        evidence["evidence_digest"] = _canonical_digest(evidence)
        validation = self.validate_evidence(evidence)
        if not validation["valid"]:
            return {
                "valid": False,
                "errors": [{"code": "evidence_validation", "field": "evidence", "message": message} for message in validation["errors"]],
                "warnings": warnings,
                "evidence": evidence,
                "boundary": evidence["boundary"],
            }
        return {
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "dataset_sha256": evidence["dataset_sha256"],
            "evidence": evidence,
            "boundary": evidence["boundary"],
        }
