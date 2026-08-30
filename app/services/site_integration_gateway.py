from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


SCHEMA_VERSION = "port-snapshot.v1"
QUALITY_VALUES = {"measured", "estimated", "derived", "missing", "stale"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")


ADAPTER_CONTRACTS: Dict[str, Dict[str, str]] = {
    "tos": {
        "throughput_teu": "TEU/hour",
        "vessel_arrivals": "vessel/hour",
        "berth_occupancy_ratio": "ratio",
        "yard_occupancy_ratio": "ratio",
    },
    "ais_vts": {
        "channel_congestion_ratio": "ratio",
        "pilot_tug_availability_ratio": "ratio",
        "closure_flag": "binary",
        "planning_vessel_draft_m": "m",
    },
    "weather_hydro": {
        "tide_m": "m",
        "wind_speed_mps": "m/s",
        "visibility_km": "km",
        "wave_height_m": "m",
        "current_speed_mps": "m/s",
        "channel_chart_depth_m": "m",
        "squat_allowance_m": "m",
    },
    "energy_scada": {
        "base_load_kw": "kW",
        "price_per_kwh": "currency/kWh",
        "carbon_kg_per_kwh": "kgCO2e/kWh",
        "shore_power_demand_kw": "kW",
        "shore_power_connection_ratio": "ratio",
    },
    "equipment_plc": {
        "crane_availability_ratio": "ratio",
        "equipment_availability_ratio": "ratio",
        "quay_crane_moves_per_hour": "move/hour",
        "horizontal_transport_moves_per_hour": "move/hour",
        "yard_crane_moves_per_hour": "move/hour",
    },
    "gate_rail_barge": {
        "truck_arrivals_per_hour": "truck/hour",
        "gate_queue_trucks": "truck",
        "gate_capacity_ratio": "ratio",
        "rail_transfer_demand_teu": "TEU/hour",
        "barge_transfer_demand_teu": "TEU/hour",
        "intermodal_capacity_ratio": "ratio",
    },
    "reefer": {
        "reefer_load_kw": "kW",
        "reefer_occupancy_ratio": "ratio",
        "reefer_temperature_risk_ratio": "ratio",
    },
    "cmms_workforce": {
        "equipment_failure_risk_ratio": "ratio",
        "maintenance_backlog_ratio": "ratio",
        "labor_availability_ratio": "ratio",
    },
}

_SIGNED_FIELDS = {"tide_m", "price_per_kwh"}


def _value_error(field_name: str, unit: str, value: float) -> str | None:
    if unit == "ratio" and not 0.0 <= value <= 1.0:
        return "ratio must be between 0 and 1"
    if unit == "binary" and value not in {0.0, 1.0}:
        return "binary reading must be 0 or 1"
    if field_name not in _SIGNED_FIELDS and value < 0.0:
        return "reading must be non-negative"
    return None


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _parse_time(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sign_snapshot(envelope_without_signature: Mapping[str, Any], secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        _canonical_bytes(dict(envelope_without_signature)),
        hashlib.sha256,
    ).hexdigest()


class SiteIntegrationGateway:
    """Validate read-only multi-adapter site snapshots.

    Accepted envelopes are never control commands. The in-memory state stores
    only replay-protection identifiers and lineage summaries, not raw telemetry.
    """

    def __init__(
        self,
        *,
        site_id: str | None = None,
        adapters: Mapping[str, Mapping[str, Any]] | None = None,
        live_attested: bool | None = None,
        max_age_seconds: int = 300,
    ) -> None:
        self.site_id = str(site_id or os.getenv("PORT_DT_SNAPSHOT_SITE_ID") or "").strip()
        self.live_attested = (
            bool(live_attested)
            if live_attested is not None
            else str(os.getenv("PORT_DT_SNAPSHOT_LIVE_ATTESTED") or "").lower()
            in {"1", "true", "yes", "on"}
        )
        self.max_age_seconds = max(30, min(86400, int(max_age_seconds)))
        self.adapters = self._load_adapters(adapters)
        self._last_sequence: Dict[str, int] = {}
        self._snapshot_ids: set[str] = set()
        self._accepted: Dict[str, Dict[str, Any]] = {}

    def _load_adapters(
        self, adapters: Mapping[str, Mapping[str, Any]] | None
    ) -> Dict[str, Dict[str, Any]]:
        raw: Mapping[str, Mapping[str, Any]] = adapters or {}
        if adapters is None:
            config_path = str(os.getenv("PORT_DT_SNAPSHOT_ADAPTER_CONFIG") or "").strip()
            if config_path:
                try:
                    candidate = Path(config_path).expanduser().resolve()
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    raw = dict(payload.get("adapters") or {})
                except Exception:
                    raw = {}
        configured: Dict[str, Dict[str, Any]] = {}
        for adapter_id, item in raw.items():
            if adapter_id not in ADAPTER_CONTRACTS or not isinstance(item, Mapping):
                continue
            secret = str(item.get("secret") or "").strip()
            secret_env = str(item.get("secret_env") or "").strip()
            if not secret and secret_env:
                secret = str(os.getenv(secret_env) or "").strip()
            configured[adapter_id] = {
                "secret": secret,
                "owner": str(item.get("owner") or "").strip(),
                "source_system": str(item.get("source_system") or "").strip(),
            }
        return configured

    def readiness(self) -> Dict[str, Any]:
        statuses = []
        verified_count = 0
        for adapter_id, fields in ADAPTER_CONTRACTS.items():
            config = self.adapters.get(adapter_id) or {}
            blockers = []
            if not self.site_id:
                blockers.append("site_id_missing")
            if not config:
                blockers.append("adapter_config_missing")
            else:
                if not config.get("secret"):
                    blockers.append("hmac_secret_missing")
                if not config.get("owner"):
                    blockers.append("owner_missing")
                if not config.get("source_system"):
                    blockers.append("source_system_missing")
            accepted = self._accepted.get(adapter_id)
            live_verified = bool(
                accepted
                and not blockers
                and self.live_attested
                and accepted.get("live_data_verified")
            )
            verified_count += int(live_verified)
            statuses.append(
                {
                    "adapter_id": adapter_id,
                    "required_fields": fields,
                    "configured": bool(config) and not blockers,
                    "live_data_verified": live_verified,
                    "last_sequence": self._last_sequence.get(adapter_id),
                    "last_accepted_at": (accepted or {}).get("accepted_at"),
                    "blockers": blockers
                    + ([] if self.live_attested else ["live_attestation_missing"]),
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "site_id": self.site_id or None,
            "adapter_count": len(ADAPTER_CONTRACTS),
            "configured_adapter_count": sum(
                int(row["configured"]) for row in statuses
            ),
            "live_verified_adapter_count": verified_count,
            "adapters": statuses,
            "quality_gates": [
                "per-adapter HMAC-SHA256",
                "canonical payload SHA-256",
                "site and adapter identity",
                "timezone-aware freshness",
                "monotonic sequence and snapshot replay protection",
                "required fields, exact units, finite values and quality labels",
            ],
            "replacement_coverage": {
                "required_adapter_count": len(ADAPTER_CONTRACTS),
                "live_verified_adapter_count": verified_count,
                "all_current_fields_covered": verified_count == len(ADAPTER_CONTRACTS),
                "720_hour_history_verified": False,
                "training_dataset_replacement_ready": False,
            },
            "boundary": {
                "read_only": True,
                "dispatch_allowed": False,
                "production_authority": False,
                "site_status": "待接入港口" if verified_count < len(ADAPTER_CONTRACTS) else "实时源已验证但历史窗口与现场准入未完成",
            },
        }

    def validate_envelope(
        self,
        envelope: Mapping[str, Any],
        *,
        now: datetime | None = None,
        consume: bool = True,
    ) -> Dict[str, Any]:
        observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        errors: list[Dict[str, Any]] = []

        def error(code: str, field: str, message: str) -> None:
            errors.append({"code": code, "field": field, "message": message})

        if envelope.get("schema_version") != SCHEMA_VERSION:
            error("schema_version", "schema_version", f"must be {SCHEMA_VERSION}")
        snapshot_id = str(envelope.get("snapshot_id") or "").strip()
        if not _IDENTIFIER.fullmatch(snapshot_id):
            error("snapshot_id", "snapshot_id", "invalid identifier")
        site_id = str(envelope.get("site_id") or "").strip()
        if not self.site_id or site_id != self.site_id:
            error("site_id", "site_id", "must match the configured site")
        adapter_id = str(envelope.get("adapter_id") or "").strip()
        contract = ADAPTER_CONTRACTS.get(adapter_id)
        config = self.adapters.get(adapter_id)
        if contract is None:
            error("adapter_id", "adapter_id", "unknown adapter")
            contract = {}
        if not config or not config.get("secret"):
            error("adapter_not_configured", "adapter_id", "configured HMAC adapter is required")
            config = {}
        if config and str(envelope.get("owner") or "").strip() != config.get("owner"):
            error("owner_mismatch", "owner", "must match configured data owner")
        if config and str(envelope.get("source_system") or "").strip() != config.get("source_system"):
            error("source_system_mismatch", "source_system", "must match configured source system")
        try:
            sequence = int(envelope.get("sequence"))
            if sequence < 1:
                raise ValueError
        except (TypeError, ValueError):
            sequence = 0
            error("sequence", "sequence", "must be a positive integer")
        previous = self._last_sequence.get(adapter_id, 0)
        if sequence <= previous:
            error("sequence_replay", "sequence", "must be strictly monotonic")
        if snapshot_id in self._snapshot_ids:
            error("snapshot_replay", "snapshot_id", "snapshot was already accepted")
        try:
            observed_at = _parse_time(envelope.get("observed_at"))
            age = (observed_now - observed_at).total_seconds()
            if age > self.max_age_seconds:
                error("stale_snapshot", "observed_at", "snapshot exceeds the freshness limit")
            if age < -30:
                error("future_snapshot", "observed_at", "snapshot is too far in the future")
        except ValueError as exc:
            observed_at = observed_now
            error("observed_at", "observed_at", str(exc))
        payload = envelope.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}
            error("payload", "payload", "must be an object")
        expected_payload_hash = _sha256(payload)
        if not hmac.compare_digest(
            str(envelope.get("payload_sha256") or ""), expected_payload_hash
        ):
            error("payload_sha256", "payload_sha256", "canonical payload digest mismatch")
        for field_name, unit in contract.items():
            reading = payload.get(field_name)
            if not isinstance(reading, Mapping):
                error("missing_field", f"payload.{field_name}", "required reading is missing")
                continue
            if str(reading.get("unit") or "") != unit:
                error("unit", f"payload.{field_name}.unit", f"must be {unit}")
            quality = str(reading.get("quality") or "").lower()
            if quality not in QUALITY_VALUES:
                error("quality", f"payload.{field_name}.quality", "invalid quality label")
            if quality in {"missing", "stale"}:
                error("unavailable_reading", f"payload.{field_name}.quality", "required reading must be current")
            try:
                value = float(reading.get("value"))
                if not math.isfinite(value):
                    raise ValueError
                semantic_error = _value_error(field_name, unit, value)
                if semantic_error:
                    error("range", f"payload.{field_name}.value", semantic_error)
            except (TypeError, ValueError):
                error("value", f"payload.{field_name}.value", "must be finite numeric")
            try:
                reading_time = _parse_time(reading.get("observed_at"))
                if abs((reading_time - observed_at).total_seconds()) > self.max_age_seconds:
                    error("reading_time", f"payload.{field_name}.observed_at", "not aligned to snapshot")
            except ValueError as exc:
                error("reading_time", f"payload.{field_name}.observed_at", str(exc))
        unsigned = {key: value for key, value in envelope.items() if key != "signature"}
        expected_signature = sign_snapshot(unsigned, str(config.get("secret") or ""))
        if not hmac.compare_digest(
            str(envelope.get("signature") or ""), expected_signature
        ):
            error("signature", "signature", "HMAC-SHA256 verification failed")

        valid = not errors
        live_verified = bool(valid and self.live_attested)
        if valid and consume:
            self._last_sequence[adapter_id] = sequence
            self._snapshot_ids.add(snapshot_id)
            self._accepted[adapter_id] = {
                "snapshot_id": snapshot_id,
                "sequence": sequence,
                "observed_at": _iso(observed_at),
                "accepted_at": _iso(observed_now),
                "payload_sha256": expected_payload_hash,
                "field_count": len(payload),
                "live_data_verified": live_verified,
            }
        return {
            "valid": valid,
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id or None,
            "adapter_id": adapter_id or None,
            "sequence": sequence or None,
            "errors": errors,
            "lineage": {
                "payload_sha256": expected_payload_hash,
                "field_count": len(payload),
                "observed_at": _iso(observed_at),
                "accepted_at": _iso(observed_now) if valid else None,
            },
            "boundary": {
                "live_data_verified": live_verified,
                "read_only": True,
                "dispatch_allowed": False,
                "production_authority": False,
            },
        }
