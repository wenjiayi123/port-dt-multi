from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SCHEMA_VERSION = "port_call_event.v1"
EVENT_PHASES = ("estimated", "requested", "planned", "actual")
EVENT_TYPES = (
    "port_arrival",
    "pilotage",
    "towage",
    "mooring",
    "berth",
    "cargo_operations",
    "bunkering",
    "port_departure",
)
EVENT_SIDES = ("start", "complete", "instant")
EVIDENCE_CLASSES = (
    "authorized_live_api",
    "authorized_site_export",
    "contract_test_only",
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_UNLOCODE = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}$")
_IMO = re.compile(r"^[0-9]{7}$")
_MMSI = re.compile(r"^[0-9]{9}$")
_PLACEHOLDERS = {"unknown", "unset", "todo", "replace", "replace_me", "n/a", "none"}


class PortCallGatewayUnavailable(RuntimeError):
    pass


class PortCallPayloadRejected(RuntimeError):
    def __init__(self, result: Dict[str, Any]):
        super().__init__("port call payload failed validation")
        self.result = result


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(value: str | None, *, default: int, minimum: int, maximum: int) -> tuple[int, bool]:
    try:
        parsed = int(str(value if value is not None else default).strip())
    except (TypeError, ValueError):
        return default, False
    if parsed < minimum or parsed > maximum:
        return default, False
    return parsed, True


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _parse_timestamp(value: Any) -> datetime:
    text = _clean(value)
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _valid_imo(value: str) -> bool:
    if not _IMO.fullmatch(value):
        return False
    digits = [int(item) for item in value]
    return sum(digit * weight for digit, weight in zip(digits[:6], range(7, 1, -1))) % 10 == digits[6]


def _canonical_digest(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class PortCallGateway:
    """Validate and query a fail-closed port-call event feed.

    The schema follows the shared port-call concepts of estimated, requested,
    planned and actual events. It is an internal interoperability profile, not
    a claim of third-party standards certification.
    """

    def __init__(self) -> None:
        self.base_url = _clean(os.getenv("PORT_DT_PORT_CALL_BASE_URL"))
        self.path = _clean(os.getenv("PORT_DT_PORT_CALL_PATH")) or "/api/port-calls/events"
        self.auth_mode = (_clean(os.getenv("PORT_DT_PORT_CALL_AUTH_MODE")) or "bearer").lower()
        self.token = _clean(os.getenv("PORT_DT_PORT_CALL_TOKEN"))
        self.site_id = _clean(os.getenv("PORT_DT_PORT_CALL_SITE_ID"))
        self.owner = _clean(os.getenv("PORT_DT_PORT_CALL_OWNER"))
        self.license = _clean(os.getenv("PORT_DT_PORT_CALL_LICENSE"))
        self.live_attested = _truthy(os.getenv("PORT_DT_PORT_CALL_LIVE_ATTESTED"))
        self.timeout_sec, timeout_valid = _bounded_int(
            os.getenv("PORT_DT_PORT_CALL_TIMEOUT_SEC"), default=8, minimum=1, maximum=30
        )
        self.max_age_sec, max_age_valid = _bounded_int(
            os.getenv("PORT_DT_PORT_CALL_MAX_AGE_SEC"), default=300, minimum=30, maximum=86400
        )
        self._config_errors = []
        if not timeout_valid:
            self._config_errors.append("timeout_invalid")
        if not max_age_valid:
            self._config_errors.append("max_age_invalid")
        self._last_verified_at: datetime | None = None

    def source_status(self) -> Dict[str, Any]:
        blockers: List[str] = list(self._config_errors)
        if not self.base_url:
            blockers.append("base_url_missing")
        elif not self.base_url.lower().startswith("https://"):
            blockers.append("https_required")
        if self.auth_mode not in {"bearer", "gateway"}:
            blockers.append("unsupported_auth_mode")
        if self.auth_mode == "bearer" and not self.token:
            blockers.append("bearer_token_missing")
        for value, code in (
            (self.site_id, "site_id_missing"),
            (self.owner, "owner_missing"),
            (self.license, "license_missing"),
        ):
            if not value:
                blockers.append(code)
        query_ready = not blockers
        live_verified = bool(query_ready and self.live_attested and self._last_verified_at)
        if query_ready and not self.live_attested:
            blockers.append("live_attestation_missing")
        elif query_ready and not self._last_verified_at:
            blockers.append("live_response_not_yet_verified")
        return {
            "adapter": "port_call_event_gateway",
            "schema_version": SCHEMA_VERSION,
            "standard_alignment": "port_call_estimated_requested_planned_actual_profile",
            "conformance_certified": False,
            "mode": "live_rest" if live_verified else ("configured_unverified" if query_ready else "unavailable"),
            "configured": query_ready,
            "query_ready": query_ready,
            "live_data_verified": live_verified,
            "measured": live_verified,
            "site_id": self.site_id or None,
            "auth_mode": self.auth_mode,
            "last_verified_at": _iso_utc(self._last_verified_at) if self._last_verified_at else None,
            "fallback_simulator": False,
            "blockers": blockers,
        }

    def readiness(self) -> Dict[str, Any]:
        status = self.source_status()
        return {
            "schema_version": SCHEMA_VERSION,
            "adapter_status": status,
            "event_contract": {
                "phases": list(EVENT_PHASES),
                "event_types": list(EVENT_TYPES),
                "event_sides": list(EVENT_SIDES),
                "required_bundle_fields": ["schema_version", "site_id", "source", "events"],
                "required_source_fields": [
                    "source_system",
                    "owner",
                    "license",
                    "timezone",
                    "retrieved_at",
                    "evidence_class",
                ],
                "required_event_fields": [
                    "event_id",
                    "port_call_id",
                    "vessel_name",
                    "port_unlocode",
                    "event_type",
                    "event_phase",
                    "event_side",
                    "event_time",
                    "source_updated_at",
                    "revision",
                    "source_reference",
                ],
                "identity_rule": "at least one checksum-valid IMO number or nine-digit MMSI per event",
                "terminal_location_rule": "berth/mooring/cargo/bunkering events require terminal_id or berth_id",
            },
            "quality_gates": [
                "schema and explicit source governance",
                "timezone-aware UTC normalization",
                "vessel identity validation",
                "event and semantic-key uniqueness",
                "revision and source-update ordering",
                "arrival/berth/cargo/departure temporal sequence",
                "configured site/owner/license reconciliation",
                "freshness and live-attestation gate",
            ],
            "boundary": {
                "live_data_verified": status["live_data_verified"],
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "已验证实时靠泊源" if status["live_data_verified"] else "待接入港口",
                "reason": (
                    "接口、来源治理元数据和实时证明均已配置；事件仍须逐批通过质量闸门。"
                    if status["live_data_verified"]
                    else "未配置并证明经授权的实时靠泊事件源；合同样例通过也不代表现场已接入。"
                ),
            },
        }

    @staticmethod
    def _error(
        errors: List[Dict[str, Any]],
        code: str,
        field: str,
        message: str,
        *,
        event_index: int | None = None,
    ) -> None:
        item: Dict[str, Any] = {"code": code, "field": field, "message": message}
        if event_index is not None:
            item["event_index"] = event_index
        errors.append(item)

    def validate_bundle(
        self,
        payload: Dict[str, Any],
        *,
        connection_verified: bool = False,
        now: datetime | None = None,
    ) -> Dict[str, Any]:
        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        if not isinstance(payload, dict):
            payload = {}
            self._error(errors, "bundle_type", "$", "payload must be a JSON object")

        schema_version = _clean(payload.get("schema_version"))
        if schema_version != SCHEMA_VERSION:
            self._error(errors, "schema_version", "schema_version", f"must equal {SCHEMA_VERSION}")

        site_id = _clean(payload.get("site_id"))
        if not _IDENTIFIER.fullmatch(site_id):
            self._error(errors, "site_id", "site_id", "site_id must be a stable 3-128 character identifier")

        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        normalized_source: Dict[str, Any] = {}
        for field in ("source_system", "owner", "license", "timezone", "retrieved_at", "evidence_class"):
            if not _clean(source.get(field)):
                self._error(errors, "source_metadata", f"source.{field}", "field is required")

        source_system = _clean(source.get("source_system"))
        owner = _clean(source.get("owner"))
        license_name = _clean(source.get("license"))
        evidence_class = _clean(source.get("evidence_class"))
        if owner.lower() in _PLACEHOLDERS or license_name.lower() in _PLACEHOLDERS:
            self._error(errors, "source_placeholder", "source.owner/license", "placeholder governance metadata is rejected")
        if evidence_class and evidence_class not in EVIDENCE_CLASSES:
            self._error(errors, "evidence_class", "source.evidence_class", f"must be one of {', '.join(EVIDENCE_CLASSES)}")
        timezone_name = _clean(source.get("timezone"))
        if timezone_name:
            try:
                ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError:
                self._error(errors, "timezone", "source.timezone", "must be a valid IANA timezone name")

        retrieved_at: datetime | None = None
        try:
            retrieved_at = _parse_timestamp(source.get("retrieved_at"))
            if retrieved_at > now_utc.replace(microsecond=0) and (retrieved_at - now_utc).total_seconds() > 300:
                self._error(errors, "future_retrieval", "source.retrieved_at", "retrieved_at is more than five minutes in the future")
        except (TypeError, ValueError):
            self._error(errors, "retrieved_at", "source.retrieved_at", "must be a timezone-aware ISO 8601 timestamp")

        normalized_source.update(
            {
                "source_system": source_system,
                "owner": owner,
                "license": license_name,
                "timezone": timezone_name,
                "retrieved_at": _iso_utc(retrieved_at) if retrieved_at else None,
                "evidence_class": evidence_class,
            }
        )

        status = self.source_status()
        if connection_verified:
            for actual, expected, field in (
                (site_id, self.site_id, "site_id"),
                (owner, self.owner, "source.owner"),
                (license_name, self.license, "source.license"),
            ):
                if expected and actual != expected:
                    self._error(errors, "configured_source_mismatch", field, "payload does not match configured source governance")

        events = payload.get("events")
        if not isinstance(events, list) or not events:
            self._error(errors, "events", "events", "events must be a non-empty array")
            events = []
        elif len(events) > 5000:
            self._error(errors, "event_limit", "events", "a bundle may contain at most 5000 events")

        normalized_events: List[Dict[str, Any]] = []
        event_ids: set[str] = set()
        semantic_keys: set[tuple[Any, ...]] = set()
        revisions: Dict[tuple[str, str, str, str], List[tuple[int, datetime, int]]] = defaultdict(list)
        sequence_candidates: Dict[
            tuple[str, str, str, str], tuple[int, int, Dict[str, Any]]
        ] = {}

        for index, raw in enumerate(events):
            if not isinstance(raw, dict):
                self._error(errors, "event_type", "events", "event must be a JSON object", event_index=index)
                continue
            before = len(errors)
            event_id = _clean(raw.get("event_id"))
            port_call_id = _clean(raw.get("port_call_id"))
            vessel_name = _clean(raw.get("vessel_name"))
            vessel_imo = _clean(raw.get("vessel_imo"))
            vessel_mmsi = _clean(raw.get("vessel_mmsi"))
            port_unlocode = _clean(raw.get("port_unlocode")).upper()
            event_type = _clean(raw.get("event_type"))
            event_phase = _clean(raw.get("event_phase"))
            event_side = _clean(raw.get("event_side"))
            terminal_id = _clean(raw.get("terminal_id"))
            berth_id = _clean(raw.get("berth_id"))
            source_reference = _clean(raw.get("source_reference"))

            for value, field in (
                (event_id, "event_id"),
                (port_call_id, "port_call_id"),
                (vessel_name, "vessel_name"),
                (source_reference, "source_reference"),
            ):
                if not value or (field != "vessel_name" and not _IDENTIFIER.fullmatch(value)):
                    self._error(errors, "event_identifier", field, "required stable identifier is invalid", event_index=index)
            if event_id in event_ids:
                self._error(errors, "duplicate_event_id", "event_id", "event_id must be unique in the bundle", event_index=index)
            event_ids.add(event_id)

            if not _UNLOCODE.fullmatch(port_unlocode):
                self._error(errors, "port_unlocode", "port_unlocode", "must be a five-character UN/LOCODE", event_index=index)
            if not ((vessel_imo and _valid_imo(vessel_imo)) or (vessel_mmsi and _MMSI.fullmatch(vessel_mmsi))):
                self._error(errors, "vessel_identity", "vessel_imo/vessel_mmsi", "provide a checksum-valid IMO number or nine-digit MMSI", event_index=index)
            if vessel_imo and not _valid_imo(vessel_imo):
                self._error(errors, "imo_checksum", "vessel_imo", "IMO number checksum is invalid", event_index=index)
            if vessel_mmsi and not _MMSI.fullmatch(vessel_mmsi):
                self._error(errors, "mmsi", "vessel_mmsi", "MMSI must contain nine digits", event_index=index)
            if event_type not in EVENT_TYPES:
                self._error(errors, "event_type", "event_type", f"must be one of {', '.join(EVENT_TYPES)}", event_index=index)
            if event_phase not in EVENT_PHASES:
                self._error(errors, "event_phase", "event_phase", f"must be one of {', '.join(EVENT_PHASES)}", event_index=index)
            if event_side not in EVENT_SIDES:
                self._error(errors, "event_side", "event_side", f"must be one of {', '.join(EVENT_SIDES)}", event_index=index)
            if event_type in {"mooring", "berth", "cargo_operations", "bunkering"} and not (terminal_id or berth_id):
                self._error(errors, "terminal_location", "terminal_id/berth_id", "terminal event requires terminal_id or berth_id", event_index=index)

            event_time: datetime | None = None
            source_updated_at: datetime | None = None
            try:
                event_time = _parse_timestamp(raw.get("event_time"))
            except (TypeError, ValueError):
                self._error(errors, "event_time", "event_time", "must be a timezone-aware ISO 8601 timestamp", event_index=index)
            try:
                source_updated_at = _parse_timestamp(raw.get("source_updated_at"))
                if retrieved_at and source_updated_at > retrieved_at:
                    self._error(errors, "source_update_order", "source_updated_at", "source_updated_at cannot be after bundle retrieved_at", event_index=index)
            except (TypeError, ValueError):
                self._error(errors, "source_updated_at", "source_updated_at", "must be a timezone-aware ISO 8601 timestamp", event_index=index)

            revision = raw.get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                self._error(errors, "revision", "revision", "revision must be an integer greater than or equal to one", event_index=index)
                revision = 0

            semantic_key = (port_call_id, event_type, event_phase, event_side, revision)
            if semantic_key in semantic_keys:
                self._error(errors, "duplicate_semantic_event", "events", "duplicate port-call semantic event and revision", event_index=index)
            semantic_keys.add(semantic_key)

            if len(errors) == before and event_time and source_updated_at:
                normalized = {
                    "event_id": event_id,
                    "port_call_id": port_call_id,
                    "vessel_name": vessel_name,
                    "vessel_imo": vessel_imo or None,
                    "vessel_mmsi": vessel_mmsi or None,
                    "port_unlocode": port_unlocode,
                    "terminal_id": terminal_id or None,
                    "berth_id": berth_id or None,
                    "event_type": event_type,
                    "event_phase": event_phase,
                    "event_side": event_side,
                    "event_time": _iso_utc(event_time),
                    "source_updated_at": _iso_utc(source_updated_at),
                    "revision": revision,
                    "cancelled": raw.get("cancelled") is True,
                    "omitted": raw.get("omitted") is True,
                    "source_reference": source_reference,
                }
                normalized_events.append(normalized)
                revision_key = (port_call_id, event_type, event_phase, event_side)
                revisions[revision_key].append((revision, source_updated_at, index))
                previous_candidate = sequence_candidates.get(revision_key)
                if previous_candidate is None or revision > previous_candidate[0]:
                    sequence_candidates[revision_key] = (revision, index, normalized)

        for rows in revisions.values():
            ordered = sorted(rows, key=lambda item: item[0])
            for previous, current in zip(ordered, ordered[1:]):
                if current[1] < previous[1]:
                    self._error(errors, "revision_order", "source_updated_at", "higher revision has an older source_updated_at", event_index=current[2])

        sequence_rank = {
            ("port_arrival", "instant"): 10,
            ("berth", "start"): 20,
            ("cargo_operations", "start"): 30,
            ("cargo_operations", "complete"): 40,
            ("berth", "complete"): 50,
            ("port_departure", "instant"): 60,
        }
        sequence_groups: Dict[tuple[str, str], List[tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for _, index, row in sequence_candidates.values():
            sequence_groups[(row["port_call_id"], row["event_phase"])].append((index, row))
        for rows in sequence_groups.values():
            checkpoints = [
                (index, sequence_rank[(row["event_type"], row["event_side"])], _parse_timestamp(row["event_time"]))
                for index, row in rows
                if (row["event_type"], row["event_side"]) in sequence_rank
                and not row["cancelled"]
                and not row["omitted"]
            ]
            checkpoints.sort(key=lambda item: item[1])
            for previous, current in zip(checkpoints, checkpoints[1:]):
                if current[2] < previous[2]:
                    self._error(errors, "port_call_sequence", "event_time", "arrival/berth/cargo/departure sequence is inconsistent", event_index=current[0])

        freshness_sec: float | None = None
        if retrieved_at:
            freshness_sec = max(0.0, (now_utc - retrieved_at).total_seconds())
            if evidence_class == "authorized_live_api" and freshness_sec > self.max_age_sec:
                self._error(errors, "stale_live_bundle", "source.retrieved_at", f"live bundle exceeds {self.max_age_sec} second freshness limit")

        valid = not errors and len(normalized_events) == len(events)
        live_data_verified = bool(
            valid
            and connection_verified
            and status.get("query_ready")
            and self.live_attested
            and evidence_class == "authorized_live_api"
        )
        if valid and evidence_class == "contract_test_only":
            warnings.append(
                {
                    "code": "contract_only",
                    "message": "合同样例通过；它不构成现场数据、实时连接或生产许可。",
                }
            )
        phase_counts = Counter(row["event_phase"] for row in normalized_events)
        type_counts = Counter(row["event_type"] for row in normalized_events)
        normalized_bundle = {
            "schema_version": SCHEMA_VERSION,
            "site_id": site_id,
            "source": normalized_source,
            "events": sorted(normalized_events, key=lambda row: (row["port_call_id"], row["event_time"], row["event_id"])),
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": valid,
            "received_event_count": len(events),
            "accepted_event_count": len(normalized_events) if valid else 0,
            "rejected_event_count": 0 if valid else max(1, len(events) - len(normalized_events)),
            "errors": errors,
            "warnings": warnings,
            "quality": {
                "unique_event_ids": len(event_ids) == len(events),
                "timezone_normalized": valid,
                "phase_counts": dict(sorted(phase_counts.items())),
                "event_type_counts": dict(sorted(type_counts.items())),
                "freshness_seconds": round(freshness_sec, 3) if freshness_sec is not None else None,
                "max_live_age_seconds": self.max_age_sec,
            },
            "evidence_digest": _canonical_digest(normalized_bundle) if valid else None,
            "normalized_bundle": normalized_bundle if valid else None,
            "boundary": {
                "live_data_verified": live_data_verified,
                "contract_validated": valid,
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "已验证实时靠泊源" if live_data_verified else "待接入港口",
                "claim": (
                    "validated_authorized_live_port_call_events"
                    if live_data_verified
                    else "contract_validation_only_not_live_port_evidence"
                ),
            },
        }

    def fetch_events(self, *, start: str, end: str, port_unlocode: str) -> Dict[str, Any]:
        status = self.source_status()
        if not status.get("query_ready"):
            raise PortCallGatewayUnavailable("port call gateway is not configured")
        params = urlencode({"start": start, "end": end, "port_unlocode": port_unlocode})
        url = urljoin(self.base_url.rstrip("/") + "/", self.path.lstrip("/"))
        url = f"{url}{'&' if '?' in url else '?'}{params}"
        headers = {"Accept": "application/json"}
        if self.auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(url, headers=headers, method="GET")
        with urlopen(request, timeout=self.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = self.validate_bundle(
            payload,
            connection_verified=bool(status.get("query_ready") and self.live_attested),
        )
        if not result["valid"]:
            raise PortCallPayloadRejected(result)
        self._last_verified_at = datetime.now(timezone.utc)
        return result
