from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DATASET_SCHEMA = "maritime_interoperability_dataset.v1"
EVIDENCE_SCHEMA = "maritime_interoperability_evidence.v1"
EVIDENCE_CLASSES = ("authorized_site_interoperability_export", "contract_test_only")
DCSA_PROFILE = "DCSA Port Call 2.0.0"
IMO_PROFILE = "IMO Compendium FAL.5/Circ.56"
IHO_FRAMEWORK = "S-100 5.2.1"
IHO_PRODUCTS = {
    "S-101": "2.0.0",
    "S-102": "3.0.0",
    "S-104": "2.0.0",
    "S-111": "2.0.0",
    "S-124": "2.0.0",
    "S-129": "2.0.0",
}
EXTERNAL_PROFILES = ("dcsa_port_call", "imo_msw", "iho_s100")
DCSA_SCOPES = (
    "GET_timestamp",
    "POST_timestamp",
    "GET_moves_forecast",
    "POST_moves_forecast",
)
IMO_ARRIVAL_ELEMENTS = {
    "IMO0140": "Ship IMO number",
    "IMO0142": "Ship name",
    "IMO0108": "Port of arrival, coded",
    "IMO0064": "Date and time of arrival - estimated",
}
IMO_DEPARTURE_ELEMENTS = {
    "IMO0140": "Ship IMO number",
    "IMO0142": "Ship name",
    "IMO0111": "Port of departure, coded",
    "IMO0066": "Date and time of departure - estimated",
}
PHASE_CODES = {
    "estimated": "EST",
    "requested": "REQ",
    "planned": "PLN",
    "actual": "ACT",
}
SERVICE_CODES = {
    "berth": "BERTH",
    "cargo_operations": "CARGO_OPERATIONS",
    "pilotage": "PILOTAGE",
    "towage": "TOWAGE",
    "mooring": "MOORING",
    "bunkering": "BUNKERING",
    "shore_power": "SHORE_POWER",
    "moves": "MOVES",
}
PHASE_TYPE_CODES = {
    "inbound": "INBD",
    "alongside": "ALGS",
    "shifting": "SHIF",
    "outbound": "OUTB",
}
FACILITY_CODES = {
    "pilot_boarding_place": "PBPL",
    "berth": "BRTH",
    "anchorage": "ANCH",
}
VESSEL_TYPES = {"GCGO", "CONT", "RORO", "CARC", "PASS", "FERY", "BULK", "TANK", "LGTK", "ASSI", "PILO"}
STANDARD_SOURCES = {
    "dcsa": "https://reference.dcsa.org/content/standards/releases/port-call/v2-0-0/port-call-v2-0-0-implementation-guide",
    "imo": "https://www.imo.org/en/ourwork/facilitation/pages/maritimesinglewindow-default.aspx",
    "iho": "https://iho.int/en/enc-ecdis",
}

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_UNLOCODE = re.compile(r"^[A-Z]{2}[A-Z2-9]{3}$")
_IMO = re.compile(r"^[0-9]{7}$")
_MMSI = re.compile(r"^[0-9]{9}$")
_FACILITY = re.compile(r"^[A-Z0-9]{1,6}$")
_PRODUCER = re.compile(r"^[A-Z0-9]{2,8}$")
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


def _digest(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _valid_uuid(value: Any) -> bool:
    try:
        return str(uuid.UUID(_clean(value))) == _clean(value).lower()
    except (ValueError, AttributeError):
        return False


def _valid_imo(value: Any) -> bool:
    text = _clean(value)
    if not _IMO.fullmatch(text):
        return False
    digits = [int(item) for item in text]
    return sum(digit * weight for digit, weight in zip(digits[:6], range(7, 1, -1))) % 10 == digits[6]


def _event_type_code(service: str, side: str) -> str | None:
    if service == "berth":
        return {"start": "ARRI", "complete": "DEPA"}.get(side)
    if service == "moves":
        return "ARRI" if side == "instant" else None
    return {"start": "STRT", "complete": "CMPL"}.get(side)


def _container_count(value: Any) -> Dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    allowed = ("totalUnits", "size20Units", "size40Units", "size45Units")
    output: Dict[str, int] = {}
    for field in allowed:
        if field not in value:
            continue
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            return None
        output[field] = number
    return output or None


def _moves_forecast(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    output: Dict[str, Any] = {}
    carrier = _clean(value.get("carrier_code"))
    if carrier:
        if not re.fullmatch(r"[A-Z0-9]{2,4}", carrier):
            return None
        output["carrierCode"] = carrier
        output["carrierCodeListProvider"] = _clean(value.get("carrier_code_list_provider"))
        if output["carrierCodeListProvider"] not in {"SMDG", "NMFTA"}:
            return None
    restow = _container_count(value.get("restow_units"))
    if restow:
        output["restowUnits"] = restow
    for source_field, target_field in (("load_units", "loadUnits"), ("discharge_units", "dischargeUnits")):
        source = value.get(source_field)
        if not isinstance(source, dict):
            continue
        typed: Dict[str, Any] = {}
        for category in ("totalUnits", "ladenUnits", "emptyUnits", "pluggedReeferUnits", "outOfGaugeUnits"):
            count = _container_count(source.get(category))
            if count:
                typed[category] = count
        if typed:
            output[target_field] = typed
    return output or None


def _internal_cases(dcsa_events: List[Dict[str, Any]], imo_values: List[Dict[str, Any]], s100: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timestamp_events = [row for row in dcsa_events if isinstance(row.get("timestamp"), dict)]
    moves_events = [row for row in dcsa_events if row.get("movesForecasts")]
    imo_ids = {row.get("data_number") for row in imo_values}
    return [
        {
            "case_id": "DCSA.EVENT.IDENTITY",
            "profile": DCSA_PROFILE,
            "passed": bool(dcsa_events) and all(_valid_uuid(row.get("eventID")) for row in dcsa_events),
        },
        {
            "case_id": "DCSA.TIMESTAMP.CLASSIFIER_AND_TIME",
            "profile": DCSA_PROFILE,
            "passed": bool(timestamp_events) and all(
                row["timestamp"].get("classifierCode") in set(PHASE_CODES.values())
                and bool(row["timestamp"].get("serviceDateTime"))
                for row in timestamp_events
            ),
        },
        {
            "case_id": "DCSA.MOVES.FORECAST_UNITS",
            "profile": DCSA_PROFILE,
            "passed": bool(moves_events),
        },
        {
            "case_id": "IMO.GENERAL_DECLARATION.MINIMUM_IDENTITY_AND_PORT",
            "profile": IMO_PROFILE,
            "passed": {"IMO0140", "IMO0142"}.issubset(imo_ids)
            and bool({"IMO0108", "IMO0111"} & imo_ids),
        },
        {
            "case_id": "IMO.GENERAL_DECLARATION.ESTIMATED_TIME",
            "profile": IMO_PROFILE,
            "passed": bool({"IMO0064", "IMO0066"} & imo_ids),
        },
        {
            "case_id": "IHO.S100.CATALOG.PRODUCT_COVERAGE",
            "profile": IHO_FRAMEWORK,
            "passed": {row.get("product_specification") for row in s100} == set(IHO_PRODUCTS),
        },
    ]


class MaritimeInteroperabilityService:
    """Map internal port-call data into reviewable standards profiles.

    This service produces deterministic mapping evidence. It is not an external
    conformance sandbox, an authority reporting gateway, an ECDIS, or a source
    of navigationally authoritative hydrographic products.
    """

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
            return {"valid": False, "errors": ["interoperability evidence must be an object"], "production_gate_eligible": False}
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            errors.append(f"schema_version must equal {EVIDENCE_SCHEMA}")
        required = (
            "site_id", "run_id", "dataset_sha256", "source", "bindings", "profiles",
            "dcsa_events", "imo_msw_envelope", "s100_catalog_handoff", "internal_conformance_cases",
            "external_conformance", "metrics", "mapping_digest", "provenance", "approved_by",
            "approved", "boundary", "evidence_digest",
        )
        for field in required:
            if payload.get(field) in (None, "", {}):
                errors.append("missing field: " + field)
        for field in ("dataset_sha256", "mapping_digest", "evidence_digest"):
            if not _SHA256.fullmatch(_clean(payload.get(field))):
                errors.append(f"{field} must be a lowercase SHA-256 digest")
        profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else {}
        if profiles != {
            "dcsa_port_call": DCSA_PROFILE,
            "imo_msw": IMO_PROFILE,
            "iho_framework": IHO_FRAMEWORK,
            "iho_products": IHO_PRODUCTS,
        }:
            errors.append("profiles do not match the fixed interoperability contract")
        dcsa = payload.get("dcsa_events") if isinstance(payload.get("dcsa_events"), list) else []
        imo_envelope = payload.get("imo_msw_envelope") if isinstance(payload.get("imo_msw_envelope"), dict) else {}
        imo_values = imo_envelope.get("data_elements") if isinstance(imo_envelope.get("data_elements"), list) else []
        s100 = payload.get("s100_catalog_handoff") if isinstance(payload.get("s100_catalog_handoff"), list) else []
        cases = _internal_cases(dcsa, imo_values, s100)
        if payload.get("internal_conformance_cases") != cases:
            errors.append("internal_conformance_cases do not match mapped messages")
        expected_mapping_digest = _digest({
            "site_id": payload.get("site_id"),
            "run_id": payload.get("run_id"),
            "bindings": payload.get("bindings"),
            "profiles": profiles,
            "dcsa_events": dcsa,
            "imo_msw_envelope": imo_envelope,
            "s100_catalog_handoff": s100,
            "internal_conformance_cases": cases,
        })
        if payload.get("mapping_digest") != expected_mapping_digest:
            errors.append("mapping_digest does not bind all mapped standards messages")
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        expected_metrics = {
            "profile_count": 3,
            "dcsa_event_count": len(dcsa),
            "dcsa_timestamp_event_count": sum(isinstance(row.get("timestamp"), dict) for row in dcsa),
            "dcsa_moves_event_count": sum(bool(row.get("movesForecasts")) for row in dcsa),
            "imo_data_element_count": len(imo_values),
            "s100_product_count": len(s100),
            "internal_case_count": len(cases),
            "internal_case_pass_count": sum(row["passed"] is True for row in cases),
            "semantic_gap_count": sum(row["passed"] is not True for row in cases),
            "mapping_coverage_rate": sum(row["passed"] is True for row in cases) / len(cases),
        }
        if metrics != expected_metrics:
            errors.append("metrics do not match mapped messages and internal cases")
        external = payload.get("external_conformance") if isinstance(payload.get("external_conformance"), list) else []
        external_by_profile = {row.get("profile"): row for row in external if isinstance(row, dict)}
        reports_valid = len(external) == len(EXTERNAL_PROFILES) and set(external_by_profile) == set(EXTERNAL_PROFILES)
        for profile in EXTERNAL_PROFILES:
            row = external_by_profile.get(profile) or {}
            reports_valid = bool(
                reports_valid
                and row.get("passed") is True
                and row.get("tested_mapping_digest") == expected_mapping_digest
                and _SHA256.fullmatch(_clean(row.get("report_sha256")))
                and _named(row.get("report_id"))
                and _named(row.get("executor"))
                and _named(row.get("source_reference"))
            )
        if external_by_profile.get("dcsa_port_call", {}).get("scope") != list(DCSA_SCOPES):
            reports_valid = False
        if external_by_profile.get("iho_s100", {}).get("scope") != list(IHO_PRODUCTS):
            reports_valid = False
        digest_payload = dict(payload)
        provided_digest = _clean(digest_payload.pop("evidence_digest", ""))
        if _digest(digest_payload) != provided_digest:
            errors.append("evidence_digest does not match interoperability evidence content")
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        approved_by = payload.get("approved_by") if isinstance(payload.get("approved_by"), dict) else {}
        reviewers = [
            _clean(approved_by.get("data_governance")),
            _clean(approved_by.get("maritime_authority")),
            _clean(approved_by.get("hydrographic_authority")),
        ]
        owner = _clean(source.get("owner")).lower()
        independent_review = bool(
            all(reviewers)
            and len({item.lower() for item in reviewers}) == 3
            and owner not in {item.lower() for item in reviewers}
        )
        boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
        fixed_boundary = bool(
            boundary.get("mapping_contract_passed") is True
            and boundary.get("authority_submission_allowed") is False
            and boundary.get("navigational_use_allowed") is False
            and boundary.get("official_certification_claim_allowed") is False
            and boundary.get("dispatch_allowed") is False
            and boundary.get("production_authority") is False
        )
        approval_contract = bool(
            payload.get("approved") is True
            and source.get("evidence_class") == "authorized_site_interoperability_export"
            and provenance.get("source_attestation") is True
            and provenance.get("change_ticket")
            and reports_valid
            and independent_review
            and expected_metrics["semantic_gap_count"] == 0
            and boundary.get("external_conformance_verified") is True
            and boundary.get("site_interoperability_accepted") is True
            and fixed_boundary
        )
        if payload.get("approved") is True and not approval_contract:
            errors.append("approved evidence requires three bound external reports, an attested site export, zero semantic gaps and three independent reviewers")
        if not fixed_boundary:
            errors.append("interoperability evidence must prohibit authority submission, navigational use, certification claims, dispatch and production authority")
        return {
            "valid": not errors,
            "errors": errors,
            "production_gate_eligible": bool(not errors and approval_contract),
            "evidence_type": "maritime_interoperability_evidence_v1",
        }

    def readiness(self) -> Dict[str, Any]:
        raw_path = _clean(os.getenv("PORT_DT_MARITIME_INTEROPERABILITY_PATH"))
        artifact: Dict[str, Any] = {
            "mode": "unconfigured",
            "configured": False,
            "verified": False,
            "production_gate_eligible": False,
            "artifact_id": None,
            "sha256": None,
            "blockers": ["maritime_interoperability_artifact_not_configured"],
        }
        if raw_path:
            path = Path(raw_path).expanduser()
            artifact.update(
                mode="configured_invalid",
                configured=True,
                artifact_id=path.name,
                blockers=["maritime_interoperability_artifact_invalid"],
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
                    mapping_digest=payload.get("mapping_digest"),
                    profile_count=(payload.get("metrics") or {}).get("profile_count"),
                    semantic_gap_count=(payload.get("metrics") or {}).get("semantic_gap_count"),
                    external_conformance_verified=(payload.get("boundary") or {}).get("external_conformance_verified") is True,
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
                "profiles": {
                    "dcsa_port_call": DCSA_PROFILE,
                    "imo_msw": IMO_PROFILE,
                    "iho_framework": IHO_FRAMEWORK,
                    "iho_products": IHO_PRODUCTS,
                },
                "dcsa_conformance_scopes": list(DCSA_SCOPES),
                "imo_minimum_arrival_elements": IMO_ARRIVAL_ELEMENTS,
                "iho_catalog_products": IHO_PRODUCTS,
                "external_reports": list(EXTERNAL_PROFILES),
                "required_evidence": [
                    "标准靠泊事件与六方协同证据摘要",
                    "字段级数字化集装箱航运协会第二版映射",
                    "国际海事组织海事单一窗口最小申报语义映射",
                    "通用水文数据模型产品版本、覆盖范围、时效、生产者与摘要目录",
                    "绑定同一映射摘要的三类外部符合性测试报告",
                    "数据治理、海事主管与水文资料责任方三方独立复核",
                ],
                "standard_sources": STANDARD_SOURCES,
            },
            "boundary": {
                "site_interoperability_accepted": accepted,
                "mapping_contract_passed": accepted,
                "external_conformance_verified": accepted,
                "authority_submission_allowed": False,
                "navigational_use_allowed": False,
                "official_certification_claim_allowed": False,
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "现场标准互操作证据已验证" if accepted else "待接入外部符合性与现场复核证据",
                "reason": (
                    "现场授权来源、三套映射、三类外部报告和三方独立复核均已验证；主管机关报送与适航使用仍由外部权威系统负责。"
                    if accepted
                    else "内部映射不能替代外部符合性测试、主管机关受理或电子海图设备型式认可。"
                ),
            },
        }

    def run(
        self,
        payload: Dict[str, Any],
        *,
        source_verified: bool = False,
        data_governance_approved_by: str | None = None,
        maritime_authority_approved_by: str | None = None,
        hydrographic_authority_approved_by: str | None = None,
        change_ticket: str | None = None,
    ) -> Dict[str, Any]:
        payload = deepcopy(payload) if isinstance(payload, dict) else {}
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        if _clean(payload.get("schema_version")) != DATASET_SCHEMA:
            self._error(errors, "schema_version", "schema_version", f"must equal {DATASET_SCHEMA}")
        site_id = _clean(payload.get("site_id"))
        run_id = _clean(payload.get("run_id"))
        for field, value in (("site_id", site_id), ("run_id", run_id)):
            if not _IDENTIFIER.fullmatch(value):
                self._error(errors, "identifier", field, f"{field} must be a stable identifier")

        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        for field in ("source_system", "owner", "license"):
            if not _named(source.get(field)):
                self._error(errors, "source_metadata", f"source.{field}", f"source.{field} is required and cannot be a placeholder")
        evidence_class = _clean(source.get("evidence_class"))
        if evidence_class not in EVIDENCE_CLASSES:
            self._error(errors, "evidence_class", "source.evidence_class", f"must be one of {EVIDENCE_CLASSES}")
        try:
            ZoneInfo(_clean(source.get("timezone")))
        except ZoneInfoNotFoundError:
            self._error(errors, "timezone", "source.timezone", "must be an IANA timezone")
        try:
            extracted_at = _iso(_parse_timestamp(source.get("extracted_at")))
        except (TypeError, ValueError):
            extracted_at = "1970-01-01T00:00:00Z"
            self._error(errors, "timestamp", "source.extracted_at", "requires an explicit timezone")

        bindings = payload.get("bindings") if isinstance(payload.get("bindings"), dict) else {}
        normalized_bindings: Dict[str, str] = {}
        for field in ("port_call_event_digest", "collaboration_evidence_digest"):
            value = _clean(bindings.get(field))
            if not _SHA256.fullmatch(value):
                self._error(errors, "binding_digest", f"bindings.{field}", "must be a lowercase SHA-256 digest")
            normalized_bindings[field] = value

        port_call = payload.get("port_call") if isinstance(payload.get("port_call"), dict) else {}
        port_call_id = _clean(port_call.get("port_call_id"))
        terminal_call_id = _clean(port_call.get("terminal_call_id"))
        if not _valid_uuid(port_call_id):
            self._error(errors, "uuid", "port_call.port_call_id", "must be a canonical UUID")
        if not _valid_uuid(terminal_call_id):
            self._error(errors, "uuid", "port_call.terminal_call_id", "must be a canonical UUID")
        port_visit_reference = _clean(port_call.get("port_visit_reference"))
        terminal_call_reference = _clean(port_call.get("terminal_call_reference"))
        if not _named(port_visit_reference) or len(port_visit_reference) > 50:
            self._error(errors, "port_reference", "port_call.port_visit_reference", "is required and limited to 50 characters")
        if not _named(terminal_call_reference) or len(terminal_call_reference) > 100:
            self._error(errors, "terminal_reference", "port_call.terminal_call_reference", "is required and limited to 100 characters")
        try:
            sequence_number = int(port_call.get("sequence_number"))
            if sequence_number < 1:
                raise ValueError
        except (TypeError, ValueError):
            sequence_number = 0
            self._error(errors, "sequence_number", "port_call.sequence_number", "must be a positive integer")
        unlocode = _clean(port_call.get("port_unlocode"))
        if not _UNLOCODE.fullmatch(unlocode):
            self._error(errors, "unlocode", "port_call.port_unlocode", "must be a five-character UN/LOCODE")
        try:
            latitude = float(port_call.get("latitude"))
            longitude = float(port_call.get("longitude"))
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError
        except (TypeError, ValueError):
            latitude = longitude = 0.0
            self._error(errors, "coordinate", "port_call", "latitude and longitude must be valid WGS 84 coordinates")
        vessel = port_call.get("vessel") if isinstance(port_call.get("vessel"), dict) else {}
        imo_number = _clean(vessel.get("imo_number"))
        vessel_name = _clean(vessel.get("name"))
        mmsi = _clean(vessel.get("mmsi"))
        vessel_type = _clean(vessel.get("type_code"))
        if not _valid_imo(imo_number):
            self._error(errors, "imo_number", "port_call.vessel.imo_number", "must be a checksum-valid seven-digit IMO number")
        if not _named(vessel_name) or len(vessel_name) > 50:
            self._error(errors, "vessel_name", "port_call.vessel.name", "is required and limited to 50 characters")
        if mmsi and not _MMSI.fullmatch(mmsi):
            self._error(errors, "mmsi", "port_call.vessel.mmsi", "must contain nine digits")
        if vessel_type not in VESSEL_TYPES:
            self._error(errors, "vessel_type", "port_call.vessel.type_code", "is outside the DCSA vessel type code list")

        dcsa_events: List[Dict[str, Any]] = []
        event_ids: set[str] = set()
        source_events = payload.get("operational_events") if isinstance(payload.get("operational_events"), list) else []
        for index, row in enumerate(source_events):
            if not isinstance(row, dict):
                self._error(errors, "event_type", "operational_events", "event must be an object", row_index=index)
                continue
            event_id = _clean(row.get("event_id"))
            if not _valid_uuid(event_id) or event_id in event_ids:
                self._error(errors, "event_id", "operational_events.event_id", "must be a unique canonical UUID", row_index=index)
            event_ids.add(event_id)
            service = _clean(row.get("event_type"))
            side = _clean(row.get("event_side"))
            phase = _clean(row.get("event_phase"))
            port_phase = _clean(row.get("port_call_phase"))
            facility_type = _clean(row.get("facility_type"))
            service_code = SERVICE_CODES.get(service)
            event_code = _event_type_code(service, side)
            if not service_code or not event_code:
                self._error(errors, "service_event", "operational_events", "event type and side are not a supported DCSA mapping", row_index=index)
            if port_phase not in PHASE_TYPE_CODES and service != "moves":
                self._error(errors, "port_call_phase", "operational_events.port_call_phase", "is not a supported phase", row_index=index)
            if facility_type not in FACILITY_CODES:
                self._error(errors, "facility_type", "operational_events.facility_type", "is not a supported facility", row_index=index)
            facility_code = _clean(row.get("facility_code"))
            facility_provider = _clean(row.get("facility_code_list_provider"))
            if not _FACILITY.fullmatch(facility_code) or facility_provider not in {"SMDG", "BIC"}:
                self._error(errors, "facility_identity", "operational_events", "facility code and SMDG/BIC provider are required", row_index=index)
            if not _named(row.get("location_name")) or not _named(row.get("source_reference")):
                self._error(errors, "event_source", "operational_events", "location_name and source_reference are required", row_index=index)
            try:
                updated_at = _iso(_parse_timestamp(row.get("updated_at")))
            except (TypeError, ValueError):
                updated_at = extracted_at
                self._error(errors, "timestamp", "operational_events.updated_at", "requires an explicit timezone", row_index=index)
            timestamp: Dict[str, Any] | None = None
            moves: Dict[str, Any] | None = None
            if service == "moves":
                if phase or _clean(row.get("event_time")):
                    self._error(errors, "moves_timestamp", "operational_events", "MOVES must not carry an ERP/A timestamp", row_index=index)
                moves = _moves_forecast(row.get("move_forecast"))
                if not moves:
                    self._error(errors, "moves_forecast", "operational_events.move_forecast", "requires at least one valid container count", row_index=index)
            else:
                classifier = PHASE_CODES.get(phase)
                if not classifier:
                    self._error(errors, "event_phase", "operational_events.event_phase", "must be estimated, requested, planned or actual", row_index=index)
                try:
                    event_time = _iso(_parse_timestamp(row.get("event_time")))
                except (TypeError, ValueError):
                    event_time = extracted_at
                    self._error(errors, "timestamp", "operational_events.event_time", "requires an explicit timezone", row_index=index)
                timestamp = {"classifierCode": classifier, "serviceDateTime": event_time}
            event: Dict[str, Any] = {
                "eventID": event_id,
                "eventUpdatedDateTime": updated_at,
                "isFYI": row.get("is_fyi") is True,
                "portCall": {
                    "portCallID": port_call_id,
                    "portVisitReference": port_visit_reference,
                    "UNLocationCode": unlocode,
                    "isOmitted": False,
                },
                "terminalCall": {
                    "terminalCallID": terminal_call_id,
                    "terminalCallReference": terminal_call_reference,
                    "sequenceNumber": sequence_number,
                    "isOmitted": False,
                },
                "portCallService": {
                    "portCallServiceID": _clean(row.get("service_id")),
                    "portCallServiceTypeCode": service_code,
                    "portCallServiceEventTypeCode": event_code,
                    **({"portCallPhaseTypeCode": PHASE_TYPE_CODES[port_phase]} if port_phase in PHASE_TYPE_CODES else {}),
                    "facilityTypeCode": FACILITY_CODES.get(facility_type),
                    "serviceLocation": {
                        "locationName": _clean(row.get("location_name")),
                        "facilityTypeCode": "POTE",
                        "UNLocationCode": unlocode,
                        "facility": {
                            "facilityName": _clean(row.get("location_name")),
                            "facilityCode": facility_code,
                            "facilityCodeListProvider": facility_provider,
                        },
                    },
                    "isCanceled": False,
                    "isDeclined": False,
                },
                "vessel": {
                    "vesselIMONumber": imo_number,
                    **({"vesselMMSINumber": mmsi} if mmsi else {}),
                    "vesselName": vessel_name,
                    "vesselTypeCode": vessel_type,
                },
                "sourceReference": _clean(row.get("source_reference")),
            }
            service_id = event["portCallService"]["portCallServiceID"]
            if not _valid_uuid(service_id):
                self._error(errors, "service_id", "operational_events.service_id", "must be a canonical UUID", row_index=index)
            reply_to = _clean(row.get("reply_to_event_id"))
            if reply_to:
                if not _valid_uuid(reply_to):
                    self._error(errors, "reply_event_id", "operational_events.reply_to_event_id", "must be a canonical UUID", row_index=index)
                event["replyToEventID"] = reply_to
            if timestamp:
                event["timestamp"] = timestamp
            if moves:
                event["movesForecasts"] = [moves]
            dcsa_events.append(event)
        if not any("timestamp" in row for row in dcsa_events):
            self._error(errors, "dcsa_timestamp_coverage", "operational_events", "at least one timestamp event is required")
        if not any(row.get("movesForecasts") for row in dcsa_events):
            self._error(errors, "dcsa_moves_coverage", "operational_events", "at least one moves forecast event is required")

        msw = payload.get("maritime_single_window") if isinstance(payload.get("maritime_single_window"), dict) else {}
        declaration_type = _clean(msw.get("declaration_type"))
        arrival_departure_code = _clean(msw.get("arrival_departure_code"))
        if declaration_type != "general_declaration":
            self._error(errors, "msw_declaration", "maritime_single_window.declaration_type", "must be general_declaration")
        if arrival_departure_code not in {"A", "D"}:
            self._error(errors, "arrival_departure", "maritime_single_window.arrival_departure_code", "must be A or D")
        if not _named(msw.get("submission_reference")):
            self._error(errors, "submission_reference", "maritime_single_window.submission_reference", "is required")
        try:
            submission_at = _iso(_parse_timestamp(msw.get("submission_at")))
        except (TypeError, ValueError):
            submission_at = extracted_at
            self._error(errors, "timestamp", "maritime_single_window.submission_at", "requires an explicit timezone")
        values = msw.get("values") if isinstance(msw.get("values"), dict) else {}
        element_map = IMO_ARRIVAL_ELEMENTS if arrival_departure_code == "A" else IMO_DEPARTURE_ELEMENTS
        expected_values = {
            "IMO0140": imo_number,
            "IMO0142": vessel_name,
            ("IMO0108" if arrival_departure_code == "A" else "IMO0111"): unlocode,
        }
        for element, expected in expected_values.items():
            if _clean(values.get(element)) != expected:
                self._error(errors, "msw_identity", f"maritime_single_window.values.{element}", "must match the port-call identity")
        time_element = "IMO0064" if arrival_departure_code == "A" else "IMO0066"
        try:
            values[time_element] = _iso(_parse_timestamp(values.get(time_element)))
        except (TypeError, ValueError):
            self._error(errors, "msw_time", f"maritime_single_window.values.{time_element}", "requires an explicit timezone")
        imo_elements = [
            {"data_number": key, "data_element": label, "value": _clean(values.get(key))}
            for key, label in element_map.items()
        ]
        imo_envelope = {
            "profile": IMO_PROFILE,
            "declarationType": "General Declaration",
            "arrivalDepartureCode": arrival_departure_code,
            "submissionReference": _clean(msw.get("submission_reference")),
            "submissionDateTime": submission_at,
            "data_elements": imo_elements,
        }

        s100_rows = payload.get("s100_catalog") if isinstance(payload.get("s100_catalog"), list) else []
        s100_handoff: List[Dict[str, Any]] = []
        seen_products: set[str] = set()
        for index, row in enumerate(s100_rows):
            if not isinstance(row, dict):
                self._error(errors, "s100_record", "s100_catalog", "record must be an object", row_index=index)
                continue
            product = _clean(row.get("product_specification"))
            version = _clean(row.get("edition"))
            if product not in IHO_PRODUCTS or product in seen_products:
                self._error(errors, "s100_product", "s100_catalog.product_specification", "product must be required and unique", row_index=index)
            elif version != IHO_PRODUCTS[product]:
                self._error(errors, "s100_edition", "s100_catalog.edition", f"{product} must use edition {IHO_PRODUCTS[product]}", row_index=index)
            seen_products.add(product)
            dataset_id = _clean(row.get("dataset_id"))
            producer_code = _clean(row.get("producer_code"))
            if not _IDENTIFIER.fullmatch(dataset_id) or not _PRODUCER.fullmatch(producer_code):
                self._error(errors, "s100_identity", "s100_catalog", "dataset_id and producer_code are required", row_index=index)
            dataset_sha = _clean(row.get("dataset_sha256"))
            if not _SHA256.fullmatch(dataset_sha):
                self._error(errors, "s100_digest", "s100_catalog.dataset_sha256", "must be a lowercase SHA-256 digest", row_index=index)
            try:
                issue_date = _parse_timestamp(row.get("issue_date"))
                valid_from = _parse_timestamp(row.get("valid_from"))
                valid_to = _parse_timestamp(row.get("valid_to"))
                if valid_to <= valid_from or issue_date > valid_to:
                    raise ValueError
            except (TypeError, ValueError):
                issue_date = valid_from = datetime(1970, 1, 1, tzinfo=timezone.utc)
                valid_to = datetime(1970, 1, 2, tzinfo=timezone.utc)
                self._error(errors, "s100_time", "s100_catalog", "issue and validity timestamps must be ordered and timezone-aware", row_index=index)
            coverage = row.get("coverage_bbox") if isinstance(row.get("coverage_bbox"), dict) else {}
            try:
                west = float(coverage.get("west"))
                south = float(coverage.get("south"))
                east = float(coverage.get("east"))
                north = float(coverage.get("north"))
                if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                    raise ValueError
                if not (west <= longitude <= east and south <= latitude <= north):
                    self._error(errors, "s100_coverage", "s100_catalog.coverage_bbox", "coverage must contain the declared port coordinate", row_index=index)
            except (TypeError, ValueError):
                west = south = east = north = 0.0
                self._error(errors, "s100_bbox", "s100_catalog.coverage_bbox", "must be an ordered WGS 84 bounding box", row_index=index)
            if not _named(row.get("source_reference")):
                self._error(errors, "s100_source", "s100_catalog.source_reference", "is required", row_index=index)
            s100_handoff.append({
                "framework": IHO_FRAMEWORK,
                "product_specification": product,
                "edition": version,
                "dataset_id": dataset_id,
                "producer_code": producer_code,
                "issue_date": _iso(issue_date),
                "valid_from": _iso(valid_from),
                "valid_to": _iso(valid_to),
                "coverage_bbox": {"crs": "WGS84", "west": west, "south": south, "east": east, "north": north},
                "dataset_sha256": dataset_sha,
                "official_source": row.get("official_source") is True,
                "source_reference": _clean(row.get("source_reference")),
                "handoff_level": "catalog_metadata_only",
            })
        if seen_products != set(IHO_PRODUCTS):
            self._error(errors, "s100_coverage", "s100_catalog", "all six fixed port-relevant product specifications are required")

        profiles = {
            "dcsa_port_call": DCSA_PROFILE,
            "imo_msw": IMO_PROFILE,
            "iho_framework": IHO_FRAMEWORK,
            "iho_products": IHO_PRODUCTS,
        }
        cases = _internal_cases(dcsa_events, imo_elements, s100_handoff)
        if any(row["passed"] is not True for row in cases):
            self._error(errors, "internal_conformance", "mapped_messages", "all fixed internal mapping cases must pass")
        mapping_digest = _digest({
            "site_id": site_id,
            "run_id": run_id,
            "bindings": normalized_bindings,
            "profiles": profiles,
            "dcsa_events": dcsa_events,
            "imo_msw_envelope": imo_envelope,
            "s100_catalog_handoff": s100_handoff,
            "internal_conformance_cases": cases,
        })

        external_source = payload.get("external_conformance") if isinstance(payload.get("external_conformance"), list) else []
        normalized_external: List[Dict[str, Any]] = []
        profiles_seen: set[str] = set()
        for index, row in enumerate(external_source):
            if not isinstance(row, dict):
                self._error(errors, "external_report", "external_conformance", "report must be an object", row_index=index)
                continue
            profile = _clean(row.get("profile"))
            if profile not in EXTERNAL_PROFILES or profile in profiles_seen:
                self._error(errors, "external_profile", "external_conformance.profile", "profile must be fixed and unique", row_index=index)
            profiles_seen.add(profile)
            try:
                executed_at = _iso(_parse_timestamp(row.get("executed_at")))
            except (TypeError, ValueError):
                executed_at = extracted_at
                self._error(errors, "external_time", "external_conformance.executed_at", "requires an explicit timezone", row_index=index)
            scope = row.get("scope") if isinstance(row.get("scope"), list) else []
            if profile == "dcsa_port_call" and scope != list(DCSA_SCOPES):
                self._error(errors, "dcsa_scope", "external_conformance.scope", "must cover all fixed GET/POST timestamp and moves scenarios", row_index=index)
            if profile == "iho_s100" and scope != list(IHO_PRODUCTS):
                self._error(errors, "iho_scope", "external_conformance.scope", "must cover all fixed S-100 product specifications", row_index=index)
            for field in ("report_id", "executor", "source_reference"):
                if not _named(row.get(field)):
                    self._error(errors, "external_metadata", f"external_conformance.{field}", f"{field} is required", row_index=index)
            if not _SHA256.fullmatch(_clean(row.get("report_sha256"))):
                self._error(errors, "external_digest", "external_conformance.report_sha256", "must be a lowercase SHA-256 digest", row_index=index)
            if _clean(row.get("tested_mapping_digest")) != mapping_digest:
                self._error(errors, "stale_external_report", "external_conformance.tested_mapping_digest", "report must bind the current mapping digest", row_index=index)
            if row.get("passed") is not True:
                self._error(errors, "external_failed", "external_conformance.passed", "external report must pass", row_index=index)
            normalized_external.append({
                "profile": profile,
                "report_id": _clean(row.get("report_id")),
                "report_sha256": _clean(row.get("report_sha256")),
                "tested_mapping_digest": _clean(row.get("tested_mapping_digest")),
                "executed_at": executed_at,
                "executor": _clean(row.get("executor")),
                "scope": list(scope),
                "passed": row.get("passed") is True,
                "source_reference": _clean(row.get("source_reference")),
            })
        normalized_external.sort(key=lambda row: row["profile"])

        dataset_sha = _digest(payload)
        metrics = {
            "profile_count": 3,
            "dcsa_event_count": len(dcsa_events),
            "dcsa_timestamp_event_count": sum(isinstance(row.get("timestamp"), dict) for row in dcsa_events),
            "dcsa_moves_event_count": sum(bool(row.get("movesForecasts")) for row in dcsa_events),
            "imo_data_element_count": len(imo_elements),
            "s100_product_count": len(s100_handoff),
            "internal_case_count": len(cases),
            "internal_case_pass_count": sum(row["passed"] is True for row in cases),
            "semantic_gap_count": sum(row["passed"] is not True for row in cases),
            "mapping_coverage_rate": sum(row["passed"] is True for row in cases) / len(cases),
        }
        reviewers = {
            "data_governance": _clean(data_governance_approved_by),
            "maritime_authority": _clean(maritime_authority_approved_by),
            "hydrographic_authority": _clean(hydrographic_authority_approved_by),
        }
        reviewer_values = [value for value in reviewers.values() if value]
        owner = _clean(source.get("owner")).lower()
        independent_review = bool(
            len(reviewer_values) == 3
            and len({value.lower() for value in reviewer_values}) == 3
            and owner not in {value.lower() for value in reviewer_values}
        )
        reports_complete = bool(
            len(normalized_external) == 3
            and {row["profile"] for row in normalized_external} == set(EXTERNAL_PROFILES)
            and all(row["passed"] and row["tested_mapping_digest"] == mapping_digest for row in normalized_external)
        )
        all_products_official = bool(s100_handoff) and all(row["official_source"] for row in s100_handoff)
        approval = bool(
            not errors
            and evidence_class == "authorized_site_interoperability_export"
            and source_verified
            and reports_complete
            and all_products_official
            and independent_review
            and _named(change_ticket)
        )
        if evidence_class == "contract_test_only":
            warnings.append({
                "code": "contract_only",
                "message": "内部映射通过不等于外部符合性验证、主管机关受理、官方认证或适航许可。",
            })
        elif not source_verified:
            warnings.append({"code": "source_unattested", "message": "现场来源尚未独立核验。"})
        if normalized_external and not reports_complete:
            warnings.append({"code": "external_conformance_incomplete", "message": "外部报告未完整覆盖并绑定当前映射摘要。"})
        if source_verified and not independent_review:
            warnings.append({"code": "independent_review_invalid", "message": "需要三名互异且独立于数据所有者的责任人复核。"})
        if source_verified and not all_products_official:
            warnings.append({"code": "hydrographic_source_unverified", "message": "水文产品目录必须绑定官方或授权生产者来源。"})

        boundary = {
            "mapping_contract_passed": not errors,
            "external_conformance_verified": approval,
            "site_interoperability_accepted": approval,
            "authority_submission_allowed": False,
            "navigational_use_allowed": False,
            "official_certification_claim_allowed": False,
            "dispatch_allowed": False,
            "production_authority": False,
        }
        if errors:
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "dataset_sha256": dataset_sha,
                "mapping_digest": mapping_digest,
                "boundary": boundary,
            }
        evidence: Dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "site_id": site_id,
            "run_id": run_id,
            "dataset_sha256": dataset_sha,
            "source": {
                "source_system": _clean(source.get("source_system")),
                "owner": _clean(source.get("owner")),
                "license": _clean(source.get("license")),
                "timezone": _clean(source.get("timezone")),
                "extracted_at": extracted_at,
                "evidence_class": evidence_class,
            },
            "bindings": normalized_bindings,
            "profiles": profiles,
            "dcsa_events": dcsa_events,
            "imo_msw_envelope": imo_envelope,
            "s100_catalog_handoff": s100_handoff,
            "internal_conformance_cases": cases,
            "external_conformance": normalized_external,
            "metrics": metrics,
            "mapping_digest": mapping_digest,
            "provenance": {
                "source_attestation": bool(source_verified),
                "change_ticket": _clean(change_ticket) or None,
                "standard_sources": STANDARD_SOURCES,
                "generated_at": extracted_at,
            },
            "approved_by": reviewers,
            "approved": approval,
            "boundary": boundary,
        }
        evidence["evidence_digest"] = _digest(evidence)
        validation = self.validate_evidence(evidence)
        if not validation["valid"]:
            return {
                "valid": False,
                "errors": [{"code": "evidence_validation", "field": "evidence", "message": message} for message in validation["errors"]],
                "warnings": warnings,
                "dataset_sha256": dataset_sha,
                "mapping_digest": mapping_digest,
                "boundary": boundary,
            }
        return {
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "dataset_sha256": dataset_sha,
            "mapping_digest": mapping_digest,
            "evidence": evidence,
            "boundary": boundary,
        }
