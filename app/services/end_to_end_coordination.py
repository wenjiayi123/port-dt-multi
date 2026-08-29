from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATASET_SCHEMA = "end_to_end_coordination_dataset.v1"
EVIDENCE_SCHEMA = "end_to_end_coordination_evidence.v1"
EVIDENCE_CLASSES = ("authorized_site_coordination_export", "contract_test_only")
RESOURCE_TYPES = (
    "channel_slot",
    "berth",
    "pilot",
    "tug",
    "quay_crane",
    "horizontal_transport",
    "yard_block",
    "gate",
    "rail",
    "shore_power_bess",
    "maintenance_team",
)
RESOURCE_LABELS = {
    "channel_slot": "航道窗口",
    "berth": "泊位",
    "pilot": "引航员",
    "tug": "拖轮",
    "quay_crane": "岸桥",
    "horizontal_transport": "水平运输车辆",
    "yard_block": "堆场箱位",
    "gate": "闸口通道",
    "rail": "铁路作业窗口",
    "shore_power_bess": "岸电与储能容量",
    "maintenance_team": "维修班组",
}
CAPACITY_UNITS = {
    "channel_slot": "slot",
    "berth": "berth",
    "pilot": "person",
    "tug": "tug",
    "quay_crane": "crane",
    "horizontal_transport": "vehicle",
    "yard_block": "teu_slot",
    "gate": "lane",
    "rail": "train_slot",
    "shore_power_bess": "kilowatt",
    "maintenance_team": "crew",
}
STAGES = (
    "port_arrival",
    "channel_transit",
    "berthing",
    "cargo_operation",
    "yard_transfer",
    "yard_operation",
    "gate_release",
    "rail_release",
    "maintenance",
)
STAGE_LABELS = {
    "port_arrival": "船舶到港",
    "channel_transit": "航道与引航拖轮",
    "berthing": "靠泊与系泊准备",
    "cargo_operation": "岸桥装卸与岸电",
    "yard_transfer": "水平运输交接",
    "yard_operation": "堆场作业",
    "gate_release": "闸口疏运",
    "rail_release": "铁路疏运",
    "maintenance": "设备维修窗口",
}
BINDING_FIELDS = (
    "site_twin_calibration_evidence_digest",
    "port_call_collaboration_evidence_digest",
    "forecast_uncertainty_evidence_digest",
    "execution_acceptance_evidence_digest",
    "business_benefit_attribution_evidence_digest",
)
OBJECTIVE_WEIGHTS = {
    "vessel_and_cargo_delay": 0.35,
    "plan_stability": 0.20,
    "energy_peak": 0.15,
    "resource_balance": 0.15,
    "maintenance_risk": 0.15,
}
FIXED_PLANNING = {
    "time_step_minutes": 15,
    "freeze_horizon_minutes": 30,
    "handoff_buffer_minutes": 15,
    "maximum_reschedule_minutes": 180,
}
MINIMUM_CHAINS = {
    "contract_test_only": 2,
    "authorized_site_coordination_export": 10,
}
REVIEWER_ROLES = (
    "integrated_planning",
    "marine_services",
    "terminal_operations",
    "equipment_energy",
)

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
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp requires an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


def _digest(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _overlap(left_start: datetime, left_end: datetime, right_start: datetime, right_end: datetime) -> bool:
    return left_start < right_end and right_start < left_end


def _ceil_step(value: datetime, origin: datetime, step_minutes: int) -> datetime:
    seconds = (value - origin).total_seconds()
    step_seconds = step_minutes * 60
    ticks = math.ceil(max(0.0, seconds) / step_seconds - 1e-12)
    return origin + timedelta(seconds=ticks * step_seconds)


class EndToEndCoordinationService:
    """Build a recommendation-only, capacity-feasible cross-resource rolling plan."""

    @staticmethod
    def _error(
        errors: List[Dict[str, Any]],
        code: str,
        field: str,
        message: str,
        *,
        row_index: int | None = None,
        task_id: str | None = None,
    ) -> None:
        item: Dict[str, Any] = {"code": code, "field": field, "message": message}
        if row_index is not None:
            item["row_index"] = row_index
        if task_id:
            item["task_id"] = task_id
        errors.append(item)

    @classmethod
    def _normalize(cls, raw_payload: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        payload = deepcopy(raw_payload) if isinstance(raw_payload, dict) else {}
        errors: List[Dict[str, Any]] = []
        if _clean(payload.get("schema_version")) != DATASET_SCHEMA:
            cls._error(errors, "schema_version", "schema_version", f"must equal {DATASET_SCHEMA}")
        site_id = _clean(payload.get("site_id"))
        run_id = _clean(payload.get("run_id"))
        for field, value in (("site_id", site_id), ("run_id", run_id)):
            if not _IDENTIFIER.fullmatch(value):
                cls._error(errors, "identifier", field, "stable 3-128 character identifier is required")
        try:
            revision = int(payload.get("planning_revision"))
            if isinstance(payload.get("planning_revision"), bool) or revision < 1:
                raise ValueError
        except (TypeError, ValueError):
            revision = 0
            cls._error(errors, "planning_revision", "planning_revision", "positive integer is required")

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
            cls._error(errors, "timezone", "source.timezone", "must be an IANA timezone")
        try:
            extracted_at = _parse_timestamp(source.get("extracted_at"))
        except (TypeError, ValueError):
            extracted_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
            cls._error(errors, "extracted_at", "source.extracted_at", "timezone-aware timestamp is required")

        bindings = payload.get("bindings") if isinstance(payload.get("bindings"), dict) else {}
        normalized_bindings: Dict[str, str] = {}
        if set(bindings) != set(BINDING_FIELDS):
            cls._error(errors, "binding_fields", "bindings", "keys must exactly match the five upstream evidence bindings")
        for field in BINDING_FIELDS:
            value = _clean(bindings.get(field))
            if not _SHA256.fullmatch(value):
                cls._error(errors, "binding_digest", f"bindings.{field}", "lowercase SHA-256 digest is required")
            normalized_bindings[field] = value

        planning = payload.get("planning") if isinstance(payload.get("planning"), dict) else {}
        plan_id = _clean(planning.get("plan_id"))
        plan_system_reference = _clean(planning.get("plan_system_reference"))
        for field, value in (("plan_id", plan_id), ("plan_system_reference", plan_system_reference)):
            if not _IDENTIFIER.fullmatch(value):
                cls._error(errors, "planning_identifier", f"planning.{field}", "stable identifier is required")
        timestamps: Dict[str, datetime] = {}
        for field in ("decision_at", "horizon_start", "horizon_end"):
            try:
                timestamps[field] = _parse_timestamp(planning.get(field))
            except (TypeError, ValueError):
                timestamps[field] = datetime(1970, 1, 1, tzinfo=timezone.utc)
                cls._error(errors, "planning_time", f"planning.{field}", "timezone-aware timestamp is required")
        decision_at = timestamps["decision_at"]
        horizon_start = timestamps["horizon_start"]
        horizon_end = timestamps["horizon_end"]
        if not decision_at <= horizon_start < horizon_end or horizon_end - horizon_start > timedelta(hours=72):
            cls._error(errors, "planning_horizon", "planning", "decision_at must not follow a positive horizon of at most seventy-two hours")
        if extracted_at > decision_at or decision_at - extracted_at > timedelta(hours=24):
            cls._error(errors, "source_cutoff", "source.extracted_at", "source export must be at or before the decision and no more than twenty-four hours old")
        for field, expected in FIXED_PLANNING.items():
            try:
                actual = int(planning.get(field))
            except (TypeError, ValueError):
                actual = -1
            if actual != expected:
                cls._error(errors, "fixed_planning", f"planning.{field}", f"must equal fixed value {expected}")
        weights = planning.get("objective_weights") if isinstance(planning.get("objective_weights"), dict) else {}
        if weights != OBJECTIVE_WEIGHTS:
            cls._error(errors, "objective_weights", "planning.objective_weights", "must equal the fixed multi-objective contract")
        normalized_planning = {
            "plan_id": plan_id,
            "plan_system_reference": plan_system_reference,
            "decision_at": _iso(decision_at),
            "horizon_start": _iso(horizon_start),
            "horizon_end": _iso(horizon_end),
            **FIXED_PLANNING,
            "objective_weights": dict(OBJECTIVE_WEIGHTS),
        }

        resources = payload.get("resources")
        if not isinstance(resources, list):
            resources = []
            cls._error(errors, "resources", "resources", "array is required")
        normalized_resources: List[Dict[str, Any]] = []
        resource_ids: set[str] = set()
        resource_by_id: Dict[str, Dict[str, Any]] = {}
        for index, raw in enumerate(resources):
            if not isinstance(raw, dict):
                cls._error(errors, "resource_type", "resources", "resource must be an object", row_index=index)
                continue
            before = len(errors)
            resource_id = _clean(raw.get("resource_id"))
            resource_type = _clean(raw.get("resource_type"))
            if not _IDENTIFIER.fullmatch(resource_id) or resource_id in resource_ids:
                cls._error(errors, "resource_id", "resource_id", "unique stable identifier is required", row_index=index)
            resource_ids.add(resource_id)
            if resource_type not in RESOURCE_TYPES:
                cls._error(errors, "resource_contract", "resource_type", f"must be one of {RESOURCE_TYPES}", row_index=index)
            if _clean(raw.get("capacity_unit")) != CAPACITY_UNITS.get(resource_type):
                cls._error(errors, "capacity_unit", "capacity_unit", "must match the fixed unit for the resource type", row_index=index)
            try:
                capacity = _finite(raw.get("capacity"))
                if capacity <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                capacity = 0.0
                cls._error(errors, "capacity", "capacity", "positive finite capacity is required", row_index=index)
            try:
                available_from = _parse_timestamp(raw.get("available_from"))
                available_to = _parse_timestamp(raw.get("available_to"))
                if available_from > horizon_start or available_to < horizon_end:
                    raise ValueError
            except (TypeError, ValueError):
                available_from = available_to = datetime(1970, 1, 1, tzinfo=timezone.utc)
                cls._error(errors, "availability", "available_from/available_to", "resource availability must cover the full planning horizon", row_index=index)
            owner = _clean(raw.get("owner"))
            source_reference = _clean(raw.get("source_reference"))
            interlock_reference = _clean(raw.get("safety_interlock_reference"))
            for field, value in (("owner", owner), ("source_reference", source_reference), ("safety_interlock_reference", interlock_reference)):
                if not _IDENTIFIER.fullmatch(value):
                    cls._error(errors, "resource_reference", field, "stable reference is required", row_index=index)
            if len(errors) == before:
                normalized = {
                    "resource_id": resource_id,
                    "resource_type": resource_type,
                    "capacity": capacity,
                    "capacity_unit": CAPACITY_UNITS[resource_type],
                    "available_from": _iso(available_from),
                    "available_to": _iso(available_to),
                    "owner": owner,
                    "source_reference": source_reference,
                    "safety_interlock_reference": interlock_reference,
                }
                normalized_resources.append(normalized)
                resource_by_id[resource_id] = normalized
        if {row["resource_type"] for row in normalized_resources} != set(RESOURCE_TYPES):
            cls._error(errors, "resource_type_coverage", "resources", "all eleven cross-chain resource types are required")

        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            tasks = []
            cls._error(errors, "tasks", "tasks", "array is required")
        normalized_tasks: List[Dict[str, Any]] = []
        task_ids: set[str] = set()
        raw_predecessors: Dict[str, List[str]] = {}
        freeze_cutoff = decision_at + timedelta(minutes=FIXED_PLANNING["freeze_horizon_minutes"])
        step = FIXED_PLANNING["time_step_minutes"]
        for index, raw in enumerate(tasks):
            if not isinstance(raw, dict):
                cls._error(errors, "task_type", "tasks", "task must be an object", row_index=index)
                continue
            before = len(errors)
            task_id = _clean(raw.get("task_id"))
            chain_id = _clean(raw.get("chain_id"))
            stage = _clean(raw.get("stage"))
            for field, value in (("task_id", task_id), ("chain_id", chain_id)):
                if not _IDENTIFIER.fullmatch(value):
                    cls._error(errors, "task_identifier", field, "stable identifier is required", row_index=index, task_id=task_id or None)
            if task_id in task_ids:
                cls._error(errors, "duplicate_task", "task_id", "task identifier must be unique", row_index=index, task_id=task_id)
            task_ids.add(task_id)
            if stage not in STAGES:
                cls._error(errors, "stage", "stage", f"must be one of {STAGES}", row_index=index, task_id=task_id)
            try:
                priority = int(raw.get("priority"))
                if isinstance(raw.get("priority"), bool) or not 1 <= priority <= 5:
                    raise ValueError
            except (TypeError, ValueError):
                priority = 0
                cls._error(errors, "priority", "priority", "integer from one to five is required", row_index=index, task_id=task_id)
            try:
                earliest_start = _parse_timestamp(raw.get("earliest_start"))
                baseline_start = _parse_timestamp(raw.get("baseline_start"))
                baseline_end = _parse_timestamp(raw.get("baseline_end"))
                deadline = _parse_timestamp(raw.get("deadline"))
                duration = int(raw.get("duration_minutes"))
                if isinstance(raw.get("duration_minutes"), bool) or duration <= 0 or duration % step:
                    raise ValueError
                if baseline_end - baseline_start != timedelta(minutes=duration):
                    raise ValueError
                if not horizon_start <= earliest_start <= baseline_start < baseline_end <= deadline <= horizon_end:
                    raise ValueError
                if any(int((stamp - horizon_start).total_seconds() // 60) % step for stamp in (earliest_start, baseline_start, baseline_end, deadline)):
                    raise ValueError
            except (TypeError, ValueError):
                earliest_start = baseline_start = baseline_end = deadline = horizon_start
                duration = 0
                cls._error(errors, "task_time", "task timing", "aligned earliest, baseline, duration and deadline must stay inside the horizon", row_index=index, task_id=task_id)
            frozen = raw.get("frozen") is True
            if frozen != (baseline_start < freeze_cutoff):
                cls._error(errors, "freeze_horizon", "frozen", "tasks inside the fixed freeze horizon must be frozen and later tasks must remain mutable", row_index=index, task_id=task_id)
            predecessors = raw.get("predecessor_ids")
            if not isinstance(predecessors, list) or any(not _IDENTIFIER.fullmatch(_clean(item)) for item in predecessors) or len({_clean(item) for item in predecessors}) != len(predecessors):
                cls._error(errors, "predecessors", "predecessor_ids", "must be a unique array of stable identifiers", row_index=index, task_id=task_id)
                predecessors = []
            predecessors = [_clean(item) for item in predecessors]
            raw_predecessors[task_id] = predecessors
            requirements_payload = raw.get("resource_requirements") if isinstance(raw.get("resource_requirements"), dict) else {}
            requirements: Dict[str, float] = {}
            if not requirements_payload or any(key not in RESOURCE_TYPES for key in requirements_payload):
                cls._error(errors, "resource_requirements", "resource_requirements", "one or more fixed resource types are required", row_index=index, task_id=task_id)
            for resource_type, raw_quantity in requirements_payload.items():
                try:
                    quantity = _finite(raw_quantity)
                    if quantity <= 0:
                        raise ValueError
                    requirements[resource_type] = quantity
                except (TypeError, ValueError):
                    cls._error(errors, "resource_quantity", f"resource_requirements.{resource_type}", "positive finite quantity is required", row_index=index, task_id=task_id)
            allocations_payload = raw.get("baseline_allocations")
            if not isinstance(allocations_payload, list) or not allocations_payload:
                allocations_payload = []
                cls._error(errors, "baseline_allocations", "baseline_allocations", "current-plan resource assignments are required", row_index=index, task_id=task_id)
            allocations: List[Dict[str, Any]] = []
            allocated_by_type: Dict[str, float] = defaultdict(float)
            seen_allocations: set[str] = set()
            for allocation in allocations_payload:
                resource_id = _clean(allocation.get("resource_id")) if isinstance(allocation, dict) else ""
                try:
                    quantity = _finite(allocation.get("quantity")) if isinstance(allocation, dict) else 0.0
                    if quantity <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    quantity = 0.0
                    cls._error(errors, "baseline_allocation_quantity", "baseline_allocations", "positive finite quantity is required", row_index=index, task_id=task_id)
                resource = resource_by_id.get(resource_id)
                if not resource or resource_id in seen_allocations:
                    cls._error(errors, "baseline_allocation_resource", "baseline_allocations", "declared unique resource identifier is required", row_index=index, task_id=task_id)
                else:
                    seen_allocations.add(resource_id)
                    allocated_by_type[resource["resource_type"]] += quantity
                    allocations.append({"resource_id": resource_id, "quantity": quantity})
            if dict(allocated_by_type) != requirements:
                cls._error(errors, "baseline_allocation_contract", "baseline_allocations", "assigned resource quantities must exactly satisfy task requirements", row_index=index, task_id=task_id)
            references = {
                "source_reference": _clean(raw.get("source_reference")),
                "forecast_receipt_id": _clean(raw.get("forecast_receipt_id")),
                "baseline_plan_receipt_id": _clean(raw.get("baseline_plan_receipt_id")),
            }
            for field, value in references.items():
                if not _IDENTIFIER.fullmatch(value):
                    cls._error(errors, "task_reference", field, "stable source or receipt reference is required", row_index=index, task_id=task_id)
            if len(errors) == before:
                normalized_tasks.append({
                    "task_id": task_id,
                    "chain_id": chain_id,
                    "stage": stage,
                    "priority": priority,
                    "earliest_start": _iso(earliest_start),
                    "baseline_start": _iso(baseline_start),
                    "baseline_end": _iso(baseline_end),
                    "deadline": _iso(deadline),
                    "duration_minutes": duration,
                    "frozen": frozen,
                    "predecessor_ids": predecessors,
                    "resource_requirements": requirements,
                    "baseline_allocations": sorted(allocations, key=lambda row: row["resource_id"]),
                    **references,
                })

        normalized_task_by_id = {row["task_id"]: row for row in normalized_tasks}
        stage_index = {stage: index for index, stage in enumerate(STAGES)}
        for task_id, predecessors in raw_predecessors.items():
            task = normalized_task_by_id.get(task_id)
            if not task:
                continue
            for predecessor_id in predecessors:
                predecessor = normalized_task_by_id.get(predecessor_id)
                if not predecessor:
                    cls._error(errors, "unknown_predecessor", "predecessor_ids", "predecessor must be a declared task", task_id=task_id)
                elif predecessor["chain_id"] != task["chain_id"] or stage_index[predecessor["stage"]] >= stage_index[task["stage"]]:
                    cls._error(errors, "predecessor_order", "predecessor_ids", "predecessor must belong to the same chain and an earlier stage", task_id=task_id)
        indegree = {task_id: 0 for task_id in normalized_task_by_id}
        children: Dict[str, List[str]] = defaultdict(list)
        for task in normalized_tasks:
            for predecessor_id in task["predecessor_ids"]:
                if predecessor_id in indegree:
                    indegree[task["task_id"]] += 1
                    children[predecessor_id].append(task["task_id"])
        ready = sorted(task_id for task_id, count in indegree.items() if count == 0)
        visited: List[str] = []
        while ready:
            task_id = ready.pop(0)
            visited.append(task_id)
            for child in sorted(children[task_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(visited) != len(normalized_tasks):
            cls._error(errors, "dependency_cycle", "tasks", "task dependency graph must be acyclic")
        chains: Dict[str, set[str]] = defaultdict(set)
        for task in normalized_tasks:
            chains[task["chain_id"]].add(task["stage"])
        minimum_chains = MINIMUM_CHAINS.get(evidence_class, MINIMUM_CHAINS["contract_test_only"])
        if len(chains) < minimum_chains:
            cls._error(errors, "chain_count", "tasks", f"at least {minimum_chains} independent operation chains are required")
        for chain_id, stages in chains.items():
            if stages != set(STAGES):
                cls._error(errors, "chain_stage_coverage", "tasks", "every chain must cover all nine fixed operation stages", task_id=chain_id)
        used_types = {resource_type for task in normalized_tasks for resource_type in task["resource_requirements"]}
        if used_types != set(RESOURCE_TYPES):
            cls._error(errors, "task_resource_coverage", "tasks", "task requirements must exercise all eleven resource types")
        try:
            safety_incident_count = int(payload.get("safety_incident_count"))
            if isinstance(payload.get("safety_incident_count"), bool) or safety_incident_count < 0:
                raise ValueError
        except (TypeError, ValueError):
            safety_incident_count = 0
            cls._error(errors, "safety_incident_count", "safety_incident_count", "nonnegative integer is required")
        exceptions = payload.get("unresolved_exception_ids")
        if not isinstance(exceptions, list) or any(not _IDENTIFIER.fullmatch(_clean(item)) for item in exceptions):
            exceptions = []
            cls._error(errors, "unresolved_exceptions", "unresolved_exception_ids", "array of stable identifiers is required")
        normalized = {
            "schema_version": DATASET_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "planning_revision": revision,
            "source": {
                "source_system": _clean(source.get("source_system")),
                "owner": _clean(source.get("owner")),
                "license": _clean(source.get("license")),
                "timezone": timezone_name,
                "extracted_at": _iso(extracted_at),
                "evidence_class": evidence_class,
            },
            "bindings": normalized_bindings,
            "planning": normalized_planning,
            "resources": sorted(normalized_resources, key=lambda row: row["resource_id"]),
            "tasks": sorted(normalized_tasks, key=lambda row: row["task_id"]),
            "safety_incident_count": safety_incident_count,
            "unresolved_exception_ids": sorted(_clean(item) for item in exceptions),
        }
        return normalized, errors

    @staticmethod
    def _capacity_conflicts(
        schedule: List[Dict[str, Any]], resources: List[Dict[str, Any]], *, start_field: str, end_field: str, allocation_field: str
    ) -> List[Dict[str, Any]]:
        resource_by_id = {row["resource_id"]: row for row in resources}
        slots: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        step = timedelta(minutes=FIXED_PLANNING["time_step_minutes"])
        for task in schedule:
            start = _parse_timestamp(task[start_field])
            end = _parse_timestamp(task[end_field])
            cursor = start
            while cursor < end:
                for allocation in task.get(allocation_field) or []:
                    slots[(allocation["resource_id"], _iso(cursor))].append({
                        "task_id": task["task_id"],
                        "quantity": float(allocation["quantity"]),
                    })
                cursor += step
        conflicts: List[Dict[str, Any]] = []
        for (resource_id, slot_at), usages in sorted(slots.items()):
            capacity = float(resource_by_id[resource_id]["capacity"])
            demand = sum(row["quantity"] for row in usages)
            if demand > capacity + 1e-9:
                conflicts.append({
                    "resource_id": resource_id,
                    "resource_type": resource_by_id[resource_id]["resource_type"],
                    "slot_at": slot_at,
                    "capacity": capacity,
                    "demand": demand,
                    "task_ids": sorted(row["task_id"] for row in usages),
                })
        return conflicts

    @staticmethod
    def _available_quantity(
        resource: Dict[str, Any], start: datetime, end: datetime, scheduled_allocations: List[Dict[str, Any]]
    ) -> float:
        step = timedelta(minutes=FIXED_PLANNING["time_step_minutes"])
        cursor = start
        minimum = float(resource["capacity"])
        while cursor < end:
            slot_end = cursor + step
            used = sum(
                row["quantity"]
                for row in scheduled_allocations
                if row["resource_id"] == resource["resource_id"]
                and _overlap(cursor, slot_end, row["start"], row["end"])
            )
            minimum = min(minimum, float(resource["capacity"]) - used)
            cursor = slot_end
        return max(0.0, minimum)

    @classmethod
    def _allocate(
        cls,
        task: Dict[str, Any],
        start: datetime,
        resources_by_type: Dict[str, List[Dict[str, Any]]],
        scheduled_allocations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]] | None:
        end = start + timedelta(minutes=int(task["duration_minutes"]))
        selected: List[Dict[str, Any]] = []
        provisional = list(scheduled_allocations)
        for resource_type, required in sorted(task["resource_requirements"].items()):
            remaining = float(required)
            for resource in resources_by_type[resource_type]:
                available = cls._available_quantity(resource, start, end, provisional)
                quantity = min(remaining, available)
                if quantity > 1e-9:
                    selected.append({"resource_id": resource["resource_id"], "quantity": quantity})
                    provisional.append({
                        "resource_id": resource["resource_id"],
                        "quantity": quantity,
                        "start": start,
                        "end": end,
                    })
                    remaining -= quantity
                if remaining <= 1e-9:
                    break
            if remaining > 1e-9:
                return None
        return sorted(selected, key=lambda row: row["resource_id"])

    @classmethod
    def _solve(cls, normalized: Dict[str, Any]) -> Dict[str, Any]:
        planning = normalized["planning"]
        horizon_start = _parse_timestamp(planning["horizon_start"])
        horizon_end = _parse_timestamp(planning["horizon_end"])
        step_minutes = int(planning["time_step_minutes"])
        buffer_minutes = int(planning["handoff_buffer_minutes"])
        max_shift = int(planning["maximum_reschedule_minutes"])
        tasks = normalized["tasks"]
        resources = normalized["resources"]
        task_by_id = {row["task_id"]: row for row in tasks}
        resources_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for resource in resources:
            resources_by_type[resource["resource_type"]].append(resource)
        for rows in resources_by_type.values():
            rows.sort(key=lambda row: row["resource_id"])

        children: Dict[str, List[str]] = defaultdict(list)
        indegree = {row["task_id"]: len(row["predecessor_ids"]) for row in tasks}
        for task in tasks:
            for predecessor in task["predecessor_ids"]:
                children[predecessor].append(task["task_id"])
        ready = [task_id for task_id, count in indegree.items() if count == 0]
        order: List[str] = []
        while ready:
            ready.sort(key=lambda task_id: (
                0 if task_by_id[task_id]["frozen"] else 1,
                -int(task_by_id[task_id]["priority"]),
                task_by_id[task_id]["baseline_start"],
                task_id,
            ))
            task_id = ready.pop(0)
            order.append(task_id)
            for child in children[task_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

        scheduled_allocations: List[Dict[str, Any]] = []
        candidate_by_id: Dict[str, Dict[str, Any]] = {}
        unscheduled: List[Dict[str, Any]] = []
        for task_id in order:
            task = task_by_id[task_id]
            predecessor_rows = [candidate_by_id.get(item) for item in task["predecessor_ids"]]
            if any(row is None for row in predecessor_rows):
                unscheduled.append({"task_id": task_id, "reason": "predecessor_unscheduled"})
                continue
            predecessor_ready = max(
                (_parse_timestamp(row["candidate_end"]) + timedelta(minutes=buffer_minutes) for row in predecessor_rows),
                default=horizon_start,
            )
            baseline_start = _parse_timestamp(task["baseline_start"])
            deadline = _parse_timestamp(task["deadline"])
            duration = timedelta(minutes=int(task["duration_minutes"]))
            if task["frozen"]:
                candidate_start = baseline_start
                selected = deepcopy(task["baseline_allocations"])
            else:
                candidate_start = _ceil_step(
                    max(baseline_start, _parse_timestamp(task["earliest_start"]), predecessor_ready),
                    horizon_start,
                    step_minutes,
                )
                latest_start = min(deadline - duration, horizon_end - duration, baseline_start + timedelta(minutes=max_shift))
                selected = None
                while candidate_start <= latest_start:
                    selected = cls._allocate(task, candidate_start, resources_by_type, scheduled_allocations)
                    if selected is not None:
                        break
                    candidate_start += timedelta(minutes=step_minutes)
            if selected is None:
                unscheduled.append({"task_id": task_id, "reason": "no_capacity_before_deadline_or_maximum_shift"})
                continue
            candidate_end = candidate_start + duration
            shift_minutes = int((candidate_start - baseline_start).total_seconds() // 60)
            recommendation_receipt_id = "COORD.RECOMMENDATION." + _digest({
                "run_id": normalized["run_id"],
                "revision": normalized["planning_revision"],
                "task_id": task_id,
                "candidate_start": _iso(candidate_start),
                "allocations": selected,
            })[:20].upper()
            row = {
                "task_id": task_id,
                "chain_id": task["chain_id"],
                "stage": task["stage"],
                "priority": task["priority"],
                "candidate_start": _iso(candidate_start),
                "candidate_end": _iso(candidate_end),
                "deadline": task["deadline"],
                "shift_minutes": shift_minutes,
                "frozen": task["frozen"],
                "allocations": selected,
                "predecessor_ids": task["predecessor_ids"],
                "baseline_plan_receipt_id": task["baseline_plan_receipt_id"],
                "forecast_receipt_id": task["forecast_receipt_id"],
                "recommendation_receipt_id": recommendation_receipt_id,
            }
            candidate_by_id[task_id] = row
            for allocation in selected:
                scheduled_allocations.append({
                    "resource_id": allocation["resource_id"],
                    "quantity": float(allocation["quantity"]),
                    "start": candidate_start,
                    "end": candidate_end,
                })

        candidate_plan = [candidate_by_id[task_id] for task_id in order if task_id in candidate_by_id]
        baseline_conflicts = cls._capacity_conflicts(
            tasks,
            resources,
            start_field="baseline_start",
            end_field="baseline_end",
            allocation_field="baseline_allocations",
        )
        candidate_conflicts = cls._capacity_conflicts(
            candidate_plan,
            resources,
            start_field="candidate_start",
            end_field="candidate_end",
            allocation_field="allocations",
        )
        deadline_misses = sum(_parse_timestamp(row["candidate_end"]) > _parse_timestamp(row["deadline"]) for row in candidate_plan)
        frozen_changes = sum(
            row["frozen"] and (
                row["candidate_start"] != task_by_id[row["task_id"]]["baseline_start"]
                or row["allocations"] != task_by_id[row["task_id"]]["baseline_allocations"]
            )
            for row in candidate_plan
        )
        predecessor_violations = 0
        for row in candidate_plan:
            for predecessor_id in row["predecessor_ids"]:
                predecessor = candidate_by_id.get(predecessor_id)
                if not predecessor or _parse_timestamp(row["candidate_start"]) < _parse_timestamp(predecessor["candidate_end"]) + timedelta(minutes=buffer_minutes):
                    predecessor_violations += 1
        shifts = [row["shift_minutes"] for row in candidate_plan]
        chain_count = len({row["chain_id"] for row in tasks})
        task_count = len(tasks)
        metrics = {
            "chain_count": chain_count,
            "task_count": task_count,
            "scheduled_task_count": len(candidate_plan),
            "scheduled_task_coverage": len(candidate_plan) / max(task_count, 1),
            "stage_coverage_count": len({row["stage"] for row in tasks}),
            "resource_type_coverage_count": len({key for row in tasks for key in row["resource_requirements"]}),
            "baseline_conflict_slot_count": len(baseline_conflicts),
            "candidate_conflict_slot_count": len(candidate_conflicts),
            "conflict_slot_reduction_count": len(baseline_conflicts) - len(candidate_conflicts),
            "rescheduled_task_count": sum(value != 0 for value in shifts),
            "maximum_reschedule_minutes": max(shifts, default=0),
            "mean_reschedule_minutes": sum(shifts) / max(len(shifts), 1),
            "deadline_miss_count": deadline_misses,
            "frozen_task_change_count": frozen_changes,
            "predecessor_violation_count": predecessor_violations,
            "handoff_count": sum(len(row["predecessor_ids"]) for row in tasks),
            "resource_assignment_count": sum(len(row["allocations"]) for row in candidate_plan),
            "task_source_coverage": sum(bool(row["source_reference"]) for row in tasks) / max(task_count, 1),
            "forecast_receipt_coverage": sum(bool(row["forecast_receipt_id"]) for row in tasks) / max(task_count, 1),
            "baseline_plan_receipt_coverage": sum(bool(row["baseline_plan_receipt_id"]) for row in tasks) / max(task_count, 1),
            "safety_incident_count": normalized["safety_incident_count"],
            "unresolved_exception_count": len(normalized["unresolved_exception_ids"]),
        }
        minimum_chains = MINIMUM_CHAINS[normalized["source"]["evidence_class"]]
        threshold_checks = {
            "minimum_chain_count": metrics["chain_count"] >= minimum_chains,
            "stage_coverage": metrics["stage_coverage_count"] == len(STAGES),
            "resource_type_coverage": metrics["resource_type_coverage_count"] == len(RESOURCE_TYPES),
            "all_tasks_scheduled": metrics["scheduled_task_coverage"] == 1.0,
            "candidate_capacity": metrics["candidate_conflict_slot_count"] == 0,
            "deadlines": metrics["deadline_miss_count"] == 0,
            "frozen_commitments": metrics["frozen_task_change_count"] == 0,
            "predecessor_handoffs": metrics["predecessor_violation_count"] == 0,
            "maximum_reschedule": metrics["maximum_reschedule_minutes"] <= FIXED_PLANNING["maximum_reschedule_minutes"],
            "source_and_receipts": (
                metrics["task_source_coverage"] == 1.0
                and metrics["forecast_receipt_coverage"] == 1.0
                and metrics["baseline_plan_receipt_coverage"] == 1.0
            ),
            "safety": metrics["safety_incident_count"] == 0,
            "no_unresolved_exception": metrics["unresolved_exception_count"] == 0,
        }
        return {
            "candidate_plan": candidate_plan,
            "unscheduled_tasks": unscheduled,
            "baseline_conflicts": baseline_conflicts,
            "candidate_conflicts": candidate_conflicts,
            "metrics": metrics,
            "threshold_checks": threshold_checks,
            "coordination_status": "pass" if all(threshold_checks.values()) else "blocked",
            "algorithm": "priority-topological-earliest-feasible-capacity-freeze-v1",
        }

    @classmethod
    def validate_evidence(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return {"valid": False, "errors": ["coordination evidence must be an object"], "production_gate_eligible": False}
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        required = (
            "site_id", "run_id", "planning_revision", "dataset_sha256", "source", "bindings",
            "planning_input", "baseline_plan_digest", "candidate_plan", "candidate_plan_digest",
            "unscheduled_tasks", "baseline_conflicts", "candidate_conflicts", "metrics",
            "threshold_checks", "coordination_status", "algorithm", "approved", "approved_by",
            "change_ticket", "boundary", "evidence_digest",
        )
        for field in required:
            if field not in payload:
                errors.append(f"missing {field}")
        body = deepcopy(payload)
        evidence_digest = _clean(body.pop("evidence_digest", ""))
        if not _SHA256.fullmatch(evidence_digest) or _digest(body) != evidence_digest:
            errors.append("evidence_digest mismatch")
        planning_input = payload.get("planning_input") if isinstance(payload.get("planning_input"), dict) else {}
        normalized, normalization_errors = cls._normalize(planning_input)
        if normalization_errors or normalized != planning_input:
            errors.append("planning_input is not a valid normalized full-chain contract")
        else:
            expected = cls._solve(normalized)
            if payload.get("dataset_sha256") != _digest(normalized):
                errors.append("dataset_sha256 mismatch")
            baseline_contract = {
                "plan_id": normalized["planning"]["plan_id"],
                "plan_system_reference": normalized["planning"]["plan_system_reference"],
                "tasks": normalized["tasks"],
            }
            if payload.get("baseline_plan_digest") != _digest(baseline_contract):
                errors.append("baseline_plan_digest mismatch")
            if payload.get("candidate_plan") != expected["candidate_plan"] or payload.get("candidate_plan_digest") != _digest(expected["candidate_plan"]):
                errors.append("candidate plan does not reproduce from the fixed solver")
            for field in ("unscheduled_tasks", "baseline_conflicts", "candidate_conflicts", "metrics", "threshold_checks", "coordination_status", "algorithm"):
                if payload.get(field) != expected[field]:
                    errors.append(f"{field} does not reproduce from planning_input")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        reviewer_rows = payload.get("approved_by") if isinstance(payload.get("approved_by"), list) else []
        reviewer_ids = [_clean(row.get("reviewer_id")) for row in reviewer_rows if isinstance(row, dict)]
        roles = Counter(row.get("role") for row in reviewer_rows if isinstance(row, dict))
        owner = _clean(source.get("owner")).lower()
        approval_contract = bool(
            not errors
            and source.get("evidence_class") == "authorized_site_coordination_export"
            and source.get("source_verified") is True
            and payload.get("coordination_status") == "pass"
            and payload.get("approved") is True
            and roles == Counter(REVIEWER_ROLES)
            and len(reviewer_ids) == len(REVIEWER_ROLES)
            and len(set(reviewer_ids)) == len(REVIEWER_ROLES)
            and all(_IDENTIFIER.fullmatch(item) and item.lower() != owner for item in reviewer_ids)
            and _IDENTIFIER.fullmatch(_clean(payload.get("change_ticket")))
            and boundary.get("site_end_to_end_coordination_accepted") is True
            and boundary.get("recommendation_only") is True
            and boundary.get("shared_plan_mutated") is False
            and boundary.get("automatic_resource_commitment_allowed") is False
            and boundary.get("dispatch_allowed") is False
            and boundary.get("production_authority") is False
        )
        if payload.get("approved") is True and not approval_contract:
            errors.append("approved site coordination requires an attested authorized export, fixed clean plan and four independent reviewers")
        return {
            "valid": not errors,
            "errors": errors,
            "production_gate_eligible": bool(not errors and approval_contract),
        }

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_END_TO_END_COORDINATION_PATH"))
        artifact: Dict[str, Any] = {
            "mode": "unconfigured",
            "configured": False,
            "verified": False,
            "artifact_id": None,
            "sha256": None,
            "blockers": ["end_to_end_coordination_artifact_not_configured"],
        }
        if raw_path:
            path = Path(raw_path).expanduser()
            artifact.update(mode="configured_invalid", configured=True, artifact_id=path.name, blockers=["end_to_end_coordination_artifact_invalid"])
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validation = self.validate_evidence(payload)
                eligible = validation["production_gate_eligible"]
                metrics = payload.get("metrics") or {}
                artifact.update(
                    mode="verified_site_artifact" if eligible else "configured_invalid",
                    verified=eligible,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    site_id=payload.get("site_id"),
                    run_id=payload.get("run_id"),
                    chain_count=metrics.get("chain_count"),
                    task_count=metrics.get("task_count"),
                    candidate_conflict_slot_count=metrics.get("candidate_conflict_slot_count"),
                    coordination_status=payload.get("coordination_status"),
                    approved=payload.get("approved") is True,
                    blockers=[] if eligible else list(validation["errors"]) or ["site_coordination_not_approved"],
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        accepted = artifact.get("verified") is True
        return {
            "dataset_schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "configured_artifact": artifact,
            "contract": {
                "resource_types": [
                    {"resource_type": item, "label": RESOURCE_LABELS[item], "capacity_unit": CAPACITY_UNITS[item]}
                    for item in RESOURCE_TYPES
                ],
                "stages": [{"stage": item, "label": STAGE_LABELS[item]} for item in STAGES],
                "required_bindings": list(BINDING_FIELDS),
                "objective_weights": dict(OBJECTIVE_WEIGHTS),
                "fixed_planning": dict(FIXED_PLANNING),
                "minimum_chains": dict(MINIMUM_CHAINS),
                "required_reviewer_roles": list(REVIEWER_ROLES),
                "solver": "priority topological earliest-feasible allocation with freeze horizon, capacity, handoff and deadline gates",
            },
            "boundary": {
                "site_end_to_end_coordination_accepted": accepted,
                "recommendation_only": True,
                "shared_plan_mutated": False,
                "automatic_resource_commitment_allowed": False,
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "现场全链条协同证据已验证" if accepted else "待接入现场全链条计划与资源证据",
                "reason": (
                    "同站点真实计划、十一类资源、九阶段任务、上游证据、冻结窗口和四方独立复核均已验证；真实资源承诺仍由各责任系统确认。"
                    if accepted
                    else "局部策略、公开回放或合同求解不能替代现场统一计划、实际资源容量、冻结承诺和跨部门批准。"
                ),
            },
        }

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        integrated_planning_approved_by: str | None = None,
        marine_services_approved_by: str | None = None,
        terminal_operations_approved_by: str | None = None,
        equipment_energy_approved_by: str | None = None,
        change_ticket: str | None = None,
    ) -> Dict[str, Any]:
        normalized, errors = self._normalize(payload)
        boundary_rejected = {
            "site_end_to_end_coordination_accepted": False,
            "recommendation_only": True,
            "shared_plan_mutated": False,
            "automatic_resource_commitment_allowed": False,
            "dispatch_allowed": False,
            "production_authority": False,
        }
        if errors:
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": errors,
                "warnings": [],
                "dataset_sha256": None,
                "evidence": None,
                "boundary": boundary_rejected,
            }
        solved = self._solve(normalized)
        source = normalized["source"]
        evidence_class = source["evidence_class"]
        source_attested = bool(source_verified and evidence_class == "authorized_site_coordination_export")
        reviewer_rows = [
            {"role": "integrated_planning", "reviewer_id": _clean(integrated_planning_approved_by)},
            {"role": "marine_services", "reviewer_id": _clean(marine_services_approved_by)},
            {"role": "terminal_operations", "reviewer_id": _clean(terminal_operations_approved_by)},
            {"role": "equipment_energy", "reviewer_id": _clean(equipment_energy_approved_by)},
        ]
        reviewer_ids = [row["reviewer_id"] for row in reviewer_rows]
        owner = source["owner"].lower()
        ticket = _clean(change_ticket)
        reviewers_valid = bool(
            len(set(reviewer_ids)) == len(REVIEWER_ROLES)
            and all(_IDENTIFIER.fullmatch(item) and item.lower() != owner for item in reviewer_ids)
            and _IDENTIFIER.fullmatch(ticket)
        )
        approved = bool(source_attested and solved["coordination_status"] == "pass" and reviewers_valid)
        warnings: List[Dict[str, Any]] = []
        if evidence_class == "contract_test_only":
            warnings.append({"code": "contract_only", "message": "合同样例只验证跨资源滚动排程，不构成现场统一计划或资源承诺。"})
        elif not source_attested:
            warnings.append({"code": "source_attestation_missing", "message": "现场计划、资源容量和业务回执尚未完成授权来源证明。"})
        if solved["coordination_status"] != "pass":
            warnings.append({"code": "coordination_gate_blocked", "message": "任务未排完、容量冲突、冻结变更、交接、时限、安全或例外门禁未通过。"})
        if source_attested and any(reviewer_ids + [ticket]) and not reviewers_valid:
            warnings.append({"code": "independent_review_invalid", "message": "综合计划、海事服务、码头运营、设备能源四名复核人必须互异、独立于数据责任主体并绑定变更单。"})
        dataset_sha256 = _digest(normalized)
        baseline_contract = {
            "plan_id": normalized["planning"]["plan_id"],
            "plan_system_reference": normalized["planning"]["plan_system_reference"],
            "tasks": normalized["tasks"],
        }
        boundary = {
            "site_end_to_end_coordination_accepted": approved,
            "recommendation_only": True,
            "shared_plan_mutated": False,
            "automatic_resource_commitment_allowed": False,
            "dispatch_allowed": False,
            "production_authority": False,
            "claim": "approved_site_coordination_evidence" if approved else "contract_or_unapproved_coordination_recommendation",
            "reason": (
                "授权统一计划、十一类资源、冻结承诺、跨域交接和四方独立复核均已通过；候选仍须由责任系统分别确认后才能承诺资源。"
                if approved
                else "合同求解、局部策略输出或未复核计划不能形成现场统一资源承诺。"
            ),
        }
        evidence: Dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": normalized["site_id"],
            "run_id": normalized["run_id"],
            "planning_revision": normalized["planning_revision"],
            "dataset_sha256": dataset_sha256,
            "source": {**source, "source_verified": source_attested},
            "bindings": normalized["bindings"],
            "planning_input": normalized,
            "baseline_plan_digest": _digest(baseline_contract),
            "candidate_plan": solved["candidate_plan"],
            "candidate_plan_digest": _digest(solved["candidate_plan"]),
            "unscheduled_tasks": solved["unscheduled_tasks"],
            "baseline_conflicts": solved["baseline_conflicts"],
            "candidate_conflicts": solved["candidate_conflicts"],
            "metrics": solved["metrics"],
            "threshold_checks": solved["threshold_checks"],
            "coordination_status": solved["coordination_status"],
            "algorithm": solved["algorithm"],
            "approved": approved,
            "approved_by": reviewer_rows if approved else [],
            "change_ticket": ticket if approved else "",
            "boundary": boundary,
        }
        evidence["evidence_digest"] = _digest(evidence)
        validation = self.validate_evidence(evidence)
        if not validation["valid"]:
            return {
                "schema_version": EVIDENCE_SCHEMA,
                "valid": False,
                "errors": [{"code": "evidence_validation", "field": "evidence", "message": item} for item in validation["errors"]],
                "warnings": warnings,
                "dataset_sha256": dataset_sha256,
                "evidence": evidence,
                "boundary": boundary,
            }
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "dataset_sha256": dataset_sha256,
            "evidence": evidence,
            "boundary": boundary,
        }
