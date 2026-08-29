from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATASET_SCHEMA = "operating_model_governance_dataset.v1"
EVIDENCE_SCHEMA = "operating_model_governance_evidence.v1"
EVIDENCE_CLASSES = ("authorized_site_operating_model_export", "contract_test_only")
DOMAINS = (
    "data_source_ownership",
    "model_risk",
    "maritime_safety",
    "terminal_operations",
    "marine_services",
    "equipment_controls",
    "energy_management",
    "cybersecurity",
    "incident_command",
    "business_benefit",
    "change_management",
    "executive_accountability",
)
DOMAIN_LABELS = {
    "data_source_ownership": "数据来源责任",
    "model_risk": "模型风险管理",
    "maritime_safety": "海事安全",
    "terminal_operations": "码头生产运营",
    "marine_services": "引航拖轮与航道服务",
    "equipment_controls": "设备与控制工程",
    "energy_management": "能源管理",
    "cybersecurity": "网络与数据安全",
    "incident_command": "事件指挥",
    "business_benefit": "业务收益确认",
    "change_management": "变更管理",
    "executive_accountability": "管理层最终问责",
}
WORKFLOWS = (
    "data_contract_change",
    "model_promotion",
    "integrated_plan_release",
    "actuator_configuration_change",
    "emergency_stop",
    "incident_response",
    "rollback_release",
    "evidence_acceptance",
    "business_claim_release",
    "privileged_access_change",
)
WORKFLOW_LABELS = {
    "data_contract_change": "数据合同变更",
    "model_promotion": "模型晋级",
    "integrated_plan_release": "全链计划发布",
    "actuator_configuration_change": "执行配置变更",
    "emergency_stop": "紧急停止",
    "incident_response": "生产事件响应",
    "rollback_release": "回退放行",
    "evidence_acceptance": "场站证据验收",
    "business_claim_release": "业务收益声明",
    "privileged_access_change": "高权限访问变更",
}
SHIFT_ROLES = (
    "duty_manager",
    "terminal_operations",
    "maritime_safety",
    "site_reliability",
    "cybersecurity",
)
SHIFT_ROLE_COMPETENCY = {
    "duty_manager": "incident_command",
    "terminal_operations": "terminal_operations",
    "maritime_safety": "maritime_safety",
    "site_reliability": "site_reliability",
    "cybersecurity": "cyber_incident_response",
}
SHIFTS = ("day", "evening", "night")
ESCALATION_TARGETS = {
    "P0": {"ack_minutes_max": 5, "executive_notification_minutes_max": 10},
    "P1": {"ack_minutes_max": 15, "executive_notification_minutes_max": 30},
    "P2": {"ack_minutes_max": 30, "executive_notification_minutes_max": 60},
    "P3": {"ack_minutes_max": 60, "executive_notification_minutes_max": 240},
}
MINIMUM_ROSTER_DAYS = {"contract_test_only": 7, "authorized_site_operating_model_export": 28}
BINDING_FIELDS = (
    "production_continuity_evidence_digest",
    "end_to_end_coordination_evidence_digest",
    "execution_acceptance_evidence_digest",
    "business_benefit_attribution_evidence_digest",
)
REVIEWER_ROLES = ("executive_accountability", "governance_assurance", "maritime_safety")

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PLACEHOLDERS = {"unknown", "unset", "todo", "replace", "replace_me", "n/a", "none"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _named(value: Any) -> bool:
    text = _clean(value)
    return bool(text and text.lower() not in _PLACEHOLDERS)


def _parse_timestamp(value: Any) -> datetime:
    text = _clean(value)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone required")
    return parsed.astimezone(timezone.utc)


def _parse_date(value: Any) -> date:
    return date.fromisoformat(_clean(value))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class OperatingModelGovernanceService:
    """Validate responsibility, segregation-of-duty, roster and escalation evidence."""

    @staticmethod
    def _error(errors: List[Dict[str, Any]], code: str, field: str, message: str, row_index: int | None = None) -> None:
        item: Dict[str, Any] = {"code": code, "field": field, "message": message}
        if row_index is not None:
            item["row_index"] = row_index
        errors.append(item)

    @classmethod
    def _normalize(cls, raw_payload: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        payload = deepcopy(raw_payload) if isinstance(raw_payload, dict) else {}
        errors: List[Dict[str, Any]] = []
        if _clean(payload.get("schema_version")) != DATASET_SCHEMA:
            cls._error(errors, "schema_version", "schema_version", f"must equal {DATASET_SCHEMA}")
        site_id, run_id = _clean(payload.get("site_id")), _clean(payload.get("run_id"))
        for field, value in (("site_id", site_id), ("run_id", run_id)):
            if not _IDENTIFIER.fullmatch(value):
                cls._error(errors, "identifier", field, "stable identifier is required")

        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        for field in ("source_system", "owner", "license"):
            if not _named(source.get(field)):
                cls._error(errors, "source_metadata", f"source.{field}", "is required and cannot be a placeholder")
        evidence_class = _clean(source.get("evidence_class"))
        if evidence_class not in EVIDENCE_CLASSES:
            cls._error(errors, "evidence_class", "source.evidence_class", f"must be one of {EVIDENCE_CLASSES}")
        timezone_name = _clean(source.get("timezone"))
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            cls._error(errors, "timezone", "source.timezone", "IANA timezone is required")
        try:
            extracted_at = _parse_timestamp(source.get("extracted_at"))
        except (TypeError, ValueError):
            extracted_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
            cls._error(errors, "extracted_at", "source.extracted_at", "timezone-aware timestamp is required")

        bindings = payload.get("bindings") if isinstance(payload.get("bindings"), dict) else {}
        normalized_bindings: Dict[str, str] = {}
        if set(bindings) != set(BINDING_FIELDS):
            cls._error(errors, "binding_fields", "bindings", "keys must exactly match the four operational evidence bindings")
        for field in BINDING_FIELDS:
            value = _clean(bindings.get(field))
            if not _SHA256.fullmatch(value):
                cls._error(errors, "binding_digest", f"bindings.{field}", "lowercase SHA-256 digest is required")
            normalized_bindings[field] = value

        roster_period = payload.get("roster_period") if isinstance(payload.get("roster_period"), dict) else {}
        try:
            start_date, end_date = _parse_date(roster_period.get("start_date")), _parse_date(roster_period.get("end_date"))
            roster_days = (end_date - start_date).days
            if roster_days < MINIMUM_ROSTER_DAYS.get(evidence_class, 7):
                raise ValueError
        except (TypeError, ValueError):
            start_date = end_date = date(1970, 1, 1)
            roster_days = 0
            cls._error(errors, "roster_period", "roster_period", f"at least {MINIMUM_ROSTER_DAYS.get(evidence_class, 7)} roster days are required")
        if roster_period.get("shifts_per_day") != 3:
            cls._error(errors, "shift_contract", "roster_period.shifts_per_day", "must equal three")
        if extracted_at.date() < end_date or extracted_at.date() > end_date + timedelta(days=1):
            cls._error(errors, "source_cutoff", "source.extracted_at", "directory and roster export must follow the roster and be no more than one day late")

        people_payload = payload.get("people") if isinstance(payload.get("people"), list) else []
        people: List[Dict[str, Any]] = []
        person_ids: set[str] = set()
        people_by_id: Dict[str, Dict[str, Any]] = {}
        for index, raw in enumerate(people_payload):
            raw = raw if isinstance(raw, dict) else {}
            person_id = _clean(raw.get("person_id"))
            role_codes = raw.get("role_codes") if isinstance(raw.get("role_codes"), list) else []
            if not _IDENTIFIER.fullmatch(person_id) or person_id in person_ids:
                cls._error(errors, "person_id", "people", "unique stable person identifier is required", index)
                continue
            person_ids.add(person_id)
            if not role_codes or any(not _IDENTIFIER.fullmatch(_clean(role)) for role in role_codes):
                cls._error(errors, "role_codes", "people.role_codes", "one or more stable role codes are required", index)
            department, directory_reference = _clean(raw.get("department")), _clean(raw.get("directory_reference"))
            if not _named(department) or not _IDENTIFIER.fullmatch(directory_reference) or raw.get("active") is not True:
                cls._error(errors, "person_contract", "people", "active directory-backed person and department are required", index)
            row = {"person_id": person_id, "department": department, "directory_reference": directory_reference, "active": raw.get("active") is True, "role_codes": sorted({_clean(role) for role in role_codes})}
            people.append(row)
            people_by_id[person_id] = row
        if len(people) < 12:
            cls._error(errors, "people_count", "people", "at least twelve active named role holders are required")

        assignments_payload = payload.get("responsibility_assignments") if isinstance(payload.get("responsibility_assignments"), list) else []
        assignments: List[Dict[str, Any]] = []
        assigned_domains: set[str] = set()
        for index, raw in enumerate(assignments_payload):
            raw = raw if isinstance(raw, dict) else {}
            domain = _clean(raw.get("domain"))
            accountable = _clean(raw.get("accountable_id"))
            categories = {}
            for field in ("responsible_ids", "consulted_ids", "informed_ids"):
                values = raw.get(field) if isinstance(raw.get(field), list) else []
                categories[field] = [_clean(item) for item in values]
            if domain not in DOMAINS or domain in assigned_domains:
                cls._error(errors, "domain_assignment", "responsibility_assignments", "each fixed domain must appear exactly once", index)
                continue
            assigned_domains.add(domain)
            all_ids = [accountable, *categories["responsible_ids"], *categories["consulted_ids"], *categories["informed_ids"]]
            if accountable not in people_by_id or any(item not in people_by_id for item in all_ids) or any(not values or len(values) != len(set(values)) for values in categories.values()):
                cls._error(errors, "raci_people", "responsibility_assignments", "accountable, responsible, consulted and informed people must be declared and nonempty", index)
            if accountable in categories["responsible_ids"]:
                cls._error(errors, "raci_separation", "responsibility_assignments", "accountable and responsible must be different people", index)
            assignments.append({"domain": domain, "accountable_id": accountable, **{field: sorted(values) for field, values in categories.items()}})
        if assigned_domains != set(DOMAINS):
            cls._error(errors, "domain_coverage", "responsibility_assignments", "all twelve decision domains are required")

        workflows_payload = payload.get("approval_workflows") if isinstance(payload.get("approval_workflows"), list) else []
        workflows: List[Dict[str, Any]] = []
        workflow_ids: set[str] = set()
        for index, raw in enumerate(workflows_payload):
            raw = raw if isinstance(raw, dict) else {}
            workflow_id = _clean(raw.get("workflow_id"))
            actor_fields = {field: _clean(raw.get(field)) for field in ("requester_id", "approver_id", "executor_id", "verifier_id")}
            reviewers = [_clean(item) for item in raw.get("reviewer_ids", [])] if isinstance(raw.get("reviewer_ids"), list) else []
            if workflow_id not in WORKFLOWS or workflow_id in workflow_ids:
                cls._error(errors, "workflow_coverage", "approval_workflows", "each fixed workflow must appear exactly once", index)
                continue
            workflow_ids.add(workflow_id)
            actors = [*actor_fields.values(), *reviewers]
            if any(item not in people_by_id for item in actors) or len(set(actor_fields.values())) != 4 or len(reviewers) < 2 or len(reviewers) != len(set(reviewers)):
                cls._error(errors, "segregation_of_duties", "approval_workflows", "requester, approver, executor and verifier must differ and two declared reviewers are required", index)
            emergency_override = raw.get("emergency_override_allowed") is True
            if emergency_override != (workflow_id == "emergency_stop") or raw.get("change_ticket_required") is not True or raw.get("production_control_granted") is not False:
                cls._error(errors, "workflow_authority", "approval_workflows", "fixed emergency, ticket and no-self-granted-production-authority rules are required", index)
            scope = _clean(raw.get("authority_scope"))
            if not _IDENTIFIER.fullmatch(scope):
                cls._error(errors, "authority_scope", "approval_workflows", "stable bounded authority scope is required", index)
            workflows.append({"workflow_id": workflow_id, **actor_fields, "reviewer_ids": sorted(reviewers), "authority_scope": scope, "change_ticket_required": True, "emergency_override_allowed": emergency_override, "production_control_granted": False})
        if workflow_ids != set(WORKFLOWS):
            cls._error(errors, "workflow_coverage", "approval_workflows", "all ten fixed workflows are required")

        competency_payload = payload.get("competency_records") if isinstance(payload.get("competency_records"), list) else []
        competencies: List[Dict[str, Any]] = []
        valid_competencies: set[tuple[str, str]] = set()
        for index, raw in enumerate(competency_payload):
            raw = raw if isinstance(raw, dict) else {}
            person_id, competency = _clean(raw.get("person_id")), _clean(raw.get("competency"))
            try:
                valid_until = _parse_date(raw.get("valid_until"))
            except (TypeError, ValueError):
                valid_until = date(1970, 1, 1)
            evidence_reference = _clean(raw.get("evidence_reference"))
            if person_id not in people_by_id or competency not in set(SHIFT_ROLE_COMPETENCY.values()) or valid_until < end_date or not _IDENTIFIER.fullmatch(evidence_reference):
                cls._error(errors, "competency", "competency_records", "current evidence-backed shift competency is required", index)
            valid_competencies.add((person_id, competency))
            competencies.append({"person_id": person_id, "competency": competency, "valid_until": valid_until.isoformat(), "evidence_reference": evidence_reference})

        roster_payload = payload.get("shift_roster") if isinstance(payload.get("shift_roster"), list) else []
        roster: List[Dict[str, Any]] = []
        roster_keys: set[tuple[str, str, str]] = set()
        for index, raw in enumerate(roster_payload):
            raw = raw if isinstance(raw, dict) else {}
            try:
                roster_date = _parse_date(raw.get("date"))
            except (TypeError, ValueError):
                roster_date = date(1970, 1, 1)
            shift, role = _clean(raw.get("shift")), _clean(raw.get("role"))
            primary, backup = _clean(raw.get("primary_id")), _clean(raw.get("backup_id"))
            key = (roster_date.isoformat(), shift, role)
            if not start_date <= roster_date < end_date or shift not in SHIFTS or role not in SHIFT_ROLES or key in roster_keys:
                cls._error(errors, "roster_key", "shift_roster", "unique in-period date, shift and role are required", index)
                continue
            roster_keys.add(key)
            competency = SHIFT_ROLE_COMPETENCY[role]
            if primary == backup or primary not in people_by_id or backup not in people_by_id or (primary, competency) not in valid_competencies or (backup, competency) not in valid_competencies:
                cls._error(errors, "roster_people", "shift_roster", "distinct primary and backup with current competency are required", index)
            handover = _clean(raw.get("handover_receipt_id"))
            if not _IDENTIFIER.fullmatch(handover):
                cls._error(errors, "handover_receipt", "shift_roster", "stable handover receipt is required", index)
            roster.append({"date": roster_date.isoformat(), "shift": shift, "role": role, "primary_id": primary, "backup_id": backup, "handover_receipt_id": handover})
        expected_roster_count = roster_days * len(SHIFTS) * len(SHIFT_ROLES)
        if len(roster_keys) != expected_roster_count or len(roster) != expected_roster_count:
            cls._error(errors, "roster_coverage", "shift_roster", "every day, shift and critical role needs primary, backup and handover")

        escalation_payload = payload.get("escalation_matrix") if isinstance(payload.get("escalation_matrix"), list) else []
        escalations: List[Dict[str, Any]] = []
        severity_set: set[str] = set()
        for index, raw in enumerate(escalation_payload):
            raw = raw if isinstance(raw, dict) else {}
            severity = _clean(raw.get("severity"))
            refs = {field: _clean(raw.get(field)) for field in ("incident_commander_id", "first_escalation_id", "second_escalation_id", "communication_plan_reference")}
            if severity not in ESCALATION_TARGETS or severity in severity_set:
                cls._error(errors, "escalation_coverage", "escalation_matrix", "each fixed severity must appear once", index)
                continue
            severity_set.add(severity)
            target = ESCALATION_TARGETS[severity]
            if raw.get("ack_minutes_max") != target["ack_minutes_max"] or raw.get("executive_notification_minutes_max") != target["executive_notification_minutes_max"] or any(value not in people_by_id for field, value in refs.items() if field != "communication_plan_reference") or len({refs["incident_commander_id"], refs["first_escalation_id"], refs["second_escalation_id"]}) != 3 or not _IDENTIFIER.fullmatch(refs["communication_plan_reference"]):
                cls._error(errors, "escalation_contract", "escalation_matrix", "fixed response targets and three distinct directory-backed escalation people are required", index)
            escalations.append({"severity": severity, **target, **refs})
        if severity_set != set(ESCALATION_TARGETS):
            cls._error(errors, "escalation_coverage", "escalation_matrix", "all four severity levels are required")

        normalized = {
            "schema_version": DATASET_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "source": {"source_system": _clean(source.get("source_system")), "owner": _clean(source.get("owner")), "license": _clean(source.get("license")), "timezone": timezone_name, "extracted_at": _iso(extracted_at), "evidence_class": evidence_class},
            "bindings": normalized_bindings,
            "roster_period": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "shifts_per_day": 3},
            "people": sorted(people, key=lambda row: row["person_id"]),
            "responsibility_assignments": sorted(assignments, key=lambda row: row["domain"]),
            "approval_workflows": sorted(workflows, key=lambda row: row["workflow_id"]),
            "competency_records": sorted(competencies, key=lambda row: (row["person_id"], row["competency"])),
            "shift_roster": sorted(roster, key=lambda row: (row["date"], row["shift"], row["role"])),
            "escalation_matrix": sorted(escalations, key=lambda row: row["severity"]),
        }
        return normalized, errors

    @staticmethod
    def _evaluate(normalized: Dict[str, Any]) -> Dict[str, Any]:
        roster_days = (_parse_date(normalized["roster_period"]["end_date"]) - _parse_date(normalized["roster_period"]["start_date"])).days
        actor_conflicts = []
        for row in normalized["approval_workflows"]:
            primary = [row["requester_id"], row["approver_id"], row["executor_id"], row["verifier_id"]]
            if len(set(primary)) != 4:
                actor_conflicts.append(row["workflow_id"])
        metrics = {
            "active_person_count": len(normalized["people"]),
            "responsibility_domain_count": len(normalized["responsibility_assignments"]),
            "approval_workflow_count": len(normalized["approval_workflows"]),
            "segregation_conflict_count": len(actor_conflicts),
            "roster_days": roster_days,
            "roster_assignment_count": len(normalized["shift_roster"]),
            "critical_shift_role_count": len(SHIFT_ROLES),
            "handover_receipt_coverage": sum(bool(row["handover_receipt_id"]) for row in normalized["shift_roster"]) / max(len(normalized["shift_roster"]), 1),
            "competency_record_count": len(normalized["competency_records"]),
            "escalation_level_count": len(normalized["escalation_matrix"]),
        }
        threshold_checks = {
            "named_people": metrics["active_person_count"] >= 12,
            "domain_coverage": metrics["responsibility_domain_count"] == len(DOMAINS),
            "workflow_coverage": metrics["approval_workflow_count"] == len(WORKFLOWS),
            "segregation_of_duties": metrics["segregation_conflict_count"] == 0,
            "roster_duration": metrics["roster_days"] >= MINIMUM_ROSTER_DAYS[normalized["source"]["evidence_class"]],
            "full_shift_coverage": metrics["roster_assignment_count"] == roster_days * len(SHIFTS) * len(SHIFT_ROLES),
            "handover_receipts": metrics["handover_receipt_coverage"] == 1.0,
            "escalation_coverage": metrics["escalation_level_count"] == len(ESCALATION_TARGETS),
            "no_self_granted_control": all(row["production_control_granted"] is False for row in normalized["approval_workflows"]),
        }
        return {"metrics": metrics, "threshold_checks": threshold_checks, "governance_status": "pass" if all(threshold_checks.values()) else "blocked", "algorithm": "named-raci-four-eyes-roster-escalation-gate-v1"}

    @classmethod
    def validate_evidence(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {"valid": False, "errors": ["operating-model evidence must be an object"], "production_gate_eligible": False}
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        body = deepcopy(payload)
        digest = _clean(body.pop("evidence_digest", ""))
        if not _SHA256.fullmatch(digest) or _digest(body) != digest:
            errors.append("evidence_digest mismatch")
        source_input = payload.get("source_input") if isinstance(payload.get("source_input"), dict) else {}
        normalized, normalization_errors = cls._normalize(source_input)
        if normalization_errors or normalized != source_input:
            errors.append("source_input is not a valid normalized operating-model contract")
        else:
            expected = cls._evaluate(normalized)
            if payload.get("dataset_sha256") != _digest(normalized):
                errors.append("dataset_sha256 mismatch")
            for field in ("metrics", "threshold_checks", "governance_status", "algorithm"):
                if payload.get(field) != expected[field]:
                    errors.append(f"{field} does not reproduce from source_input")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        reviewers = payload.get("approved_by") if isinstance(payload.get("approved_by"), list) else []
        reviewer_ids = [_clean(row.get("reviewer_id")) for row in reviewers if isinstance(row, dict)]
        roles = Counter(row.get("role") for row in reviewers if isinstance(row, dict))
        owner = _clean(source.get("owner")).lower()
        approval_contract = bool(
            not errors
            and source.get("evidence_class") == "authorized_site_operating_model_export"
            and source.get("source_verified") is True
            and payload.get("governance_status") == "pass"
            and payload.get("approved") is True
            and roles == Counter(REVIEWER_ROLES)
            and len(reviewer_ids) == 3
            and len(set(reviewer_ids)) == 3
            and all(_IDENTIFIER.fullmatch(item) and item.lower() != owner for item in reviewer_ids)
            and _IDENTIFIER.fullmatch(_clean(payload.get("change_ticket")))
            and boundary.get("site_operating_model_accepted") is True
            and boundary.get("organization_authority_verified") is True
            and boundary.get("system_can_assign_roles") is False
            and boundary.get("self_approval_allowed") is False
            and boundary.get("dispatch_allowed") is False
            and boundary.get("production_authority") is False
        )
        if payload.get("approved") is True and not approval_contract:
            errors.append("approved operating model requires authorized directory evidence, full duty separation and three independent reviewers")
        return {"valid": not errors, "errors": errors, "production_gate_eligible": bool(not errors and approval_contract)}

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        executive_accountability_approved_by: str | None = None,
        governance_assurance_approved_by: str | None = None,
        maritime_safety_approved_by: str | None = None,
        change_ticket: str | None = None,
    ) -> Dict[str, Any]:
        normalized, errors = self._normalize(payload)
        rejected = {"site_operating_model_accepted": False, "organization_authority_verified": False, "system_can_assign_roles": False, "self_approval_allowed": False, "dispatch_allowed": False, "production_authority": False}
        if errors:
            return {"schema_version": EVIDENCE_SCHEMA, "valid": False, "errors": errors, "warnings": [], "dataset_sha256": None, "evidence": None, "boundary": rejected}
        evaluated = self._evaluate(normalized)
        source = normalized["source"]
        source_attested = bool(source_verified and source["evidence_class"] == "authorized_site_operating_model_export")
        reviewers = [
            {"role": "executive_accountability", "reviewer_id": _clean(executive_accountability_approved_by)},
            {"role": "governance_assurance", "reviewer_id": _clean(governance_assurance_approved_by)},
            {"role": "maritime_safety", "reviewer_id": _clean(maritime_safety_approved_by)},
        ]
        reviewer_ids = [row["reviewer_id"] for row in reviewers]
        owner = source["owner"].lower()
        ticket = _clean(change_ticket)
        reviewers_valid = bool(all(_IDENTIFIER.fullmatch(item) and item.lower() != owner for item in reviewer_ids) and len(set(reviewer_ids)) == 3 and _IDENTIFIER.fullmatch(ticket))
        approved = bool(source_attested and evaluated["governance_status"] == "pass" and reviewers_valid)
        boundary = {
            "site_operating_model_accepted": approved,
            "organization_authority_verified": approved,
            "system_can_assign_roles": False,
            "self_approval_allowed": False,
            "dispatch_allowed": False,
            "production_authority": False,
            "claim": "approved_site_operating_model_evidence" if approved else "contract_or_unapproved_governance_evidence",
            "reason": "现场目录身份、十二类责任、十项审批流程、四周值守、交接和升级链均已验证；具体生产权限仍由外部身份与访问管理系统授予。" if approved else "合同人员和职责样例不能替代现场任命、身份目录、值班表、培训资质和正式授权。",
        }
        warnings = [] if approved else [{"code": "contract_only", "message": "职责合同只检查覆盖和分权，不任命人员、不授予访问权或生产控制权。"}]
        evidence: Dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": normalized["site_id"],
            "run_id": normalized["run_id"],
            "dataset_sha256": _digest(normalized),
            "source": {**source, "source_verified": source_attested},
            "bindings": normalized["bindings"],
            "source_input": normalized,
            **evaluated,
            "approved": approved,
            "approved_by": reviewers if approved else [],
            "change_ticket": ticket if approved else "",
            "boundary": boundary,
        }
        evidence["evidence_digest"] = _digest(evidence)
        validation = self.validate_evidence(evidence)
        if not validation["valid"]:
            return {"schema_version": EVIDENCE_SCHEMA, "valid": False, "errors": [{"code": "evidence_validation", "field": "evidence", "message": item} for item in validation["errors"]], "warnings": warnings, "dataset_sha256": evidence["dataset_sha256"], "evidence": evidence, "boundary": boundary}
        return {"schema_version": EVIDENCE_SCHEMA, "valid": True, "errors": [], "warnings": warnings, "dataset_sha256": evidence["dataset_sha256"], "evidence": evidence, "boundary": boundary}

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_OPERATING_MODEL_GOVERNANCE_PATH"))
        artifact: Dict[str, Any] = {"mode": "unconfigured", "configured": False, "verified": False, "artifact_id": None, "sha256": None, "blockers": ["operating_model_governance_artifact_not_configured"]}
        if raw_path:
            path = Path(raw_path).expanduser()
            artifact.update(mode="configured_invalid", configured=True, artifact_id=path.name, blockers=["operating_model_governance_artifact_invalid"])
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validation = self.validate_evidence(payload)
                eligible = validation["production_gate_eligible"]
                metrics = payload.get("metrics") or {}
                artifact.update(mode="verified_site_artifact" if eligible else "configured_invalid", verified=eligible, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), site_id=payload.get("site_id"), run_id=payload.get("run_id"), active_person_count=metrics.get("active_person_count"), responsibility_domain_count=metrics.get("responsibility_domain_count"), approval_workflow_count=metrics.get("approval_workflow_count"), roster_days=metrics.get("roster_days"), segregation_conflict_count=metrics.get("segregation_conflict_count"), approved=payload.get("approved") is True, blockers=[] if eligible else list(validation["errors"]) or ["site_operating_model_not_approved"])
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        accepted = artifact.get("verified") is True
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "configured_artifact": artifact,
            "contract": {"responsibility_domains": [{"domain": item, "label": DOMAIN_LABELS[item]} for item in DOMAINS], "approval_workflows": [{"workflow_id": item, "label": WORKFLOW_LABELS[item]} for item in WORKFLOWS], "shift_roles": list(SHIFT_ROLES), "shifts": list(SHIFTS), "escalation_targets": deepcopy(ESCALATION_TARGETS), "minimum_roster_days": dict(MINIMUM_ROSTER_DAYS), "required_bindings": list(BINDING_FIELDS), "required_reviewer_roles": list(REVIEWER_ROLES)},
            "boundary": {"site_operating_model_accepted": accepted, "organization_authority_verified": accepted, "system_can_assign_roles": False, "self_approval_allowed": False, "dispatch_allowed": False, "production_authority": False, "site_status": "现场组织职责与值守证据已验证" if accepted else "待接入现场任命、值守与授权证据", "reason": "现场身份、职责、异人审批、值守交接和升级链已验证；权限授予仍由现场身份系统执行。" if accepted else "静态职责表和合同样例不能替代真实任命、身份目录、培训资质、轮值表和访问授权。"},
        }
