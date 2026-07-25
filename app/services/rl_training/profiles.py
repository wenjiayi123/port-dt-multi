from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping
from zoneinfo import ZoneInfo

from .datasets import FACTOR_COLUMNS
from .identifiers import validate_identifier


DEFAULT_PROFILE_ROOT = Path("config/ports")
DEFAULT_PROFILE_ID = "reference_port_v1"
DEFAULT_PROFILE: Dict[str, Any] = {
    "profile_id": DEFAULT_PROFILE_ID,
    "name": "Reference port integration profile",
    "port_code": "REFERENCE",
    "timezone": "UTC",
    "currency": "USD",
    "calibration_status": "engineering_reference_not_site_calibrated",
    "environment_version": "port_ops_v1",
    "control_authority": "recommendation_only",
    "assets": {
        "bess_capacity_kwh": 2500.0,
        "bess_power_kw": 900.0,
        "demand_cap_kw": 3500.0,
    },
    "control_limits": {
        "soc_min": 0.12,
        "soc_max": 0.88,
        "service_factor_min": 0.75,
        "service_factor_max": 1.25,
        "flexible_load_fraction": 0.60,
        "berth_priority_limit": 1.0,
        "yard_flow_limit": 1.0,
    },
    "objectives": {
        "cost": 0.25,
        "carbon": 0.20,
        "peak": 0.20,
        "safety": 0.20,
        "delay": 0.15,
    },
    "factor_requirements": {
        "required_for_training": [],
        "required_for_site_claim": [
            "berth_occupancy_ratio",
            "yard_occupancy_ratio",
            "crane_availability_ratio",
            "equipment_availability_ratio",
            "channel_congestion_ratio",
            "wind_speed_mps",
            "visibility_km",
            "pilot_tug_availability_ratio",
        ],
    },
    "weather_limits": {
        "wind_stop_mps": None,
        "visibility_stop_km": None,
        "wave_stop_m": None,
    },
}


def _merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def validate_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
    merged = _merge(DEFAULT_PROFILE, profile)
    profile_id = validate_identifier(merged.get("profile_id"), field="profile_id")
    merged["profile_id"] = profile_id
    if merged.get("environment_version") not in {"port_ops_v1", "port_ops_v2"}:
        raise ValueError("environment_version must be port_ops_v1 or port_ops_v2")
    if merged.get("control_authority") != "recommendation_only":
        raise ValueError("open-source port profiles must keep control_authority=recommendation_only")
    port_code = str(merged.get("port_code") or "").strip().upper()
    if not port_code or len(port_code) > 16:
        raise ValueError("profile port_code must contain 1-16 characters")
    merged["port_code"] = port_code
    timezone_name = str(merged.get("timezone") or "").strip()
    try:
        ZoneInfo(timezone_name)
    except Exception as exc:
        raise ValueError(f"profile timezone is not an IANA zone: {timezone_name}") from exc
    currency = str(merged.get("currency") or "").strip().upper()
    if len(currency) != 3 or not currency.isalpha():
        raise ValueError("profile currency must be a three-letter ISO-style code")
    merged["currency"] = currency
    if not str(merged.get("calibration_status") or "").strip():
        raise ValueError("profile calibration_status is required")
    assets = merged["assets"]
    for name in ("bess_capacity_kwh", "bess_power_kw", "demand_cap_kw"):
        value = float(assets[name])
        if value <= 0:
            raise ValueError(f"profile assets.{name} must be positive")
        assets[name] = value
    limits = merged["control_limits"]
    numeric_limits = (
        "soc_min",
        "soc_max",
        "service_factor_min",
        "service_factor_max",
        "flexible_load_fraction",
        "berth_priority_limit",
        "yard_flow_limit",
    )
    for name in numeric_limits:
        limits[name] = float(limits[name])
    if not 0 <= limits["soc_min"] < limits["soc_max"] <= 1:
        raise ValueError("profile SOC limits must satisfy 0 <= min < max <= 1")
    if not 0 < limits["service_factor_min"] <= 1 <= limits["service_factor_max"]:
        raise ValueError("profile service factor limits must contain 1.0")
    for name in (
        "flexible_load_fraction",
        "berth_priority_limit",
        "yard_flow_limit",
    ):
        if not 0 <= limits[name] <= 1:
            raise ValueError(f"profile control_limits.{name} must be in [0, 1]")
    objectives = {name: max(0.0, float(value)) for name, value in merged["objectives"].items()}
    required_objectives = {"cost", "carbon", "peak", "safety", "delay"}
    missing_objectives = sorted(required_objectives - set(objectives))
    if missing_objectives:
        raise ValueError(
            "profile objectives missing: " + ", ".join(missing_objectives)
        )
    objective_sum = sum(objectives.values())
    if objective_sum <= 0:
        raise ValueError("profile objectives must contain at least one positive weight")
    merged["objectives"] = {name: value / objective_sum for name, value in objectives.items()}
    requirements = merged["factor_requirements"]
    for scope in ("required_for_training", "required_for_site_claim"):
        factors = list(requirements.get(scope) or [])
        unknown = sorted(set(factors) - set(FACTOR_COLUMNS))
        if unknown:
            raise ValueError(
                f"profile factor_requirements.{scope} contains unknown factors: "
                + ", ".join(unknown)
            )
        requirements[scope] = list(dict.fromkeys(factors))
    weather_limits = merged["weather_limits"]
    for name in ("wind_stop_mps", "visibility_stop_km", "wave_stop_m"):
        value = weather_limits.get(name)
        if value is not None and float(value) <= 0:
            raise ValueError(f"profile weather_limits.{name} must be positive or null")
        weather_limits[name] = None if value is None else float(value)
    return merged


def load_profile(profile_id: str, profile_root: Path = DEFAULT_PROFILE_ROOT) -> Dict[str, Any]:
    resolved = validate_identifier(profile_id, field="profile_id")
    if resolved == DEFAULT_PROFILE_ID:
        return validate_profile(DEFAULT_PROFILE)
    path = profile_root / f"{resolved}.json"
    if not path.exists():
        raise FileNotFoundError(f"port profile not found: {resolved}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"port profile must be a JSON object: {resolved}")
    return validate_profile(payload)


def list_profiles(profile_root: Path = DEFAULT_PROFILE_ROOT) -> List[Dict[str, Any]]:
    items = [load_profile(DEFAULT_PROFILE_ID, profile_root)]
    if not profile_root.exists():
        return items
    for path in sorted(profile_root.glob("*.json")):
        if path.name.endswith(".schema.json"):
            continue
        try:
            profile = load_profile(path.stem, profile_root)
        except Exception as exc:
            items.append({"profile_id": path.stem, "valid": False, "error": str(exc)})
            continue
        items.append(profile)
    unique: Dict[str, Dict[str, Any]] = {}
    for item in items:
        unique[str(item["profile_id"])] = item
    return list(unique.values())
