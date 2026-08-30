from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping
from zoneinfo import ZoneInfo

from .datasets import FACTOR_COLUMNS, PORT_WIDE_COLUMNS, REGULATORY_COLUMNS
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
        "operational_load_fraction": 0.35,
        "allocation_load_fraction": 0.08,
    },
    "control_limits": {
        "soc_min": 0.12,
        "soc_max": 0.88,
        "service_factor_min": 0.75,
        "service_factor_max": 1.25,
        "flexible_load_fraction": 0.60,
        "berth_priority_limit": 1.0,
        "yard_flow_limit": 1.0,
        "inspection_buffer_limit": 1.0,
        "recovery_priority_limit": 1.0,
        "gate_smoothing_limit": 1.0,
        "intermodal_allocation_limit": 1.0,
        "reefer_service_limit": 1.0,
        "shore_power_allocation_limit": 1.0,
        "maintenance_reserve_limit": 1.0,
        "marine_service_allocation_limit": 1.0,
        "rail_allocation_limit": 1.0,
        "barge_allocation_limit": 1.0,
        "pilotage_allocation_limit": 1.0,
        "towage_allocation_limit": 1.0,
        "quay_crane_allocation_limit": 1.0,
        "horizontal_transport_allocation_limit": 1.0,
        "yard_crane_allocation_limit": 1.0,
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
    "regulatory_operations": {
        "maritime_inspection_capacity_vessels_per_hour": 0.22,
        "customs_inspection_capacity_vessels_per_hour": 0.28,
        "inspection_buffer_capacity_gain": 0.55,
        "inspection_buffer_service_reserve_fraction": 0.05,
        "recovery_service_capacity_gain": 0.12,
        "inspection_readiness_load_fraction": 0.015,
        "recovery_load_fraction": 0.025,
    },
    "integrated_operations": {
        "gate_service_trucks_per_hour": 160.0,
        "rail_barge_service_teu_per_hour": 110.0,
        "marine_service_vessels_per_hour": 1.2,
        "reefer_risk_recovery_rate": 0.18,
        "maintenance_recovery_rate": 0.12,
        "shore_power_service_efficiency": 0.96,
        "shore_power_avoided_carbon_kg_per_kwh": 0.62,
        "required_under_keel_clearance_m": 1.0,
        "dangerous_goods_yard_occupancy_limit": 0.85,
        "high_reefer_risk_ratio": 0.65,
        "high_equipment_failure_risk_ratio": 0.60,
        "forecast_uncertainty_review_ratio": 0.55,
    },
    "coordinated_operations": {
        "rail_service_teu_per_hour": 45.0,
        "barge_service_teu_per_hour": 80.0,
        "pilotage_service_vessels_per_hour": 0.8,
        "towage_service_vessels_per_hour": 1.0,
        "quay_crane_moves_per_hour": 180.0,
        "horizontal_transport_moves_per_hour": 200.0,
        "yard_crane_moves_per_hour": 190.0,
        "container_moves_per_teu": 1.35,
        "gate_minimum_capacity_factor": 0.5,
        "intermodal_minimum_capacity_factor": 0.5,
        "marine_minimum_capacity_factor": 0.5,
        "terminal_chain_minimum_capacity_factor": 0.5,
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
    if merged.get("environment_version") not in {"port_ops_v1", "port_ops_v2", "port_ops_v3", "port_ops_v4", "port_ops_v5", "port_ops_v6"}:
        raise ValueError("environment_version must be port_ops_v1, port_ops_v2, port_ops_v3, port_ops_v4, port_ops_v5 or port_ops_v6")
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
    for name in ("operational_load_fraction", "allocation_load_fraction"):
        value = float(assets[name])
        if not 0 <= value <= 1:
            raise ValueError(f"profile assets.{name} must be in [0, 1]")
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
        "inspection_buffer_limit",
        "recovery_priority_limit",
        "gate_smoothing_limit",
        "intermodal_allocation_limit",
        "reefer_service_limit",
        "shore_power_allocation_limit",
        "maintenance_reserve_limit",
        "marine_service_allocation_limit",
        "rail_allocation_limit",
        "barge_allocation_limit",
        "pilotage_allocation_limit",
        "towage_allocation_limit",
        "quay_crane_allocation_limit",
        "horizontal_transport_allocation_limit",
        "yard_crane_allocation_limit",
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
        "inspection_buffer_limit",
        "recovery_priority_limit",
        "gate_smoothing_limit",
        "intermodal_allocation_limit",
        "reefer_service_limit",
        "shore_power_allocation_limit",
        "maintenance_reserve_limit",
        "marine_service_allocation_limit",
        "rail_allocation_limit",
        "barge_allocation_limit",
        "pilotage_allocation_limit",
        "towage_allocation_limit",
        "quay_crane_allocation_limit",
        "horizontal_transport_allocation_limit",
        "yard_crane_allocation_limit",
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
        unknown = sorted(set(factors) - set((*FACTOR_COLUMNS, *REGULATORY_COLUMNS, *PORT_WIDE_COLUMNS)))
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
    regulatory = merged["regulatory_operations"]
    for name in (
        "maritime_inspection_capacity_vessels_per_hour",
        "customs_inspection_capacity_vessels_per_hour",
    ):
        regulatory[name] = float(regulatory[name])
        if regulatory[name] <= 0:
            raise ValueError(f"profile regulatory_operations.{name} must be positive")
    for name in (
        "inspection_buffer_capacity_gain",
        "inspection_buffer_service_reserve_fraction",
        "recovery_service_capacity_gain",
        "inspection_readiness_load_fraction",
        "recovery_load_fraction",
    ):
        regulatory[name] = float(regulatory[name])
        if not 0 <= regulatory[name] <= 1:
            raise ValueError(f"profile regulatory_operations.{name} must be in [0, 1]")
    integrated = merged["integrated_operations"]
    for name in (
        "gate_service_trucks_per_hour",
        "rail_barge_service_teu_per_hour",
        "marine_service_vessels_per_hour",
        "required_under_keel_clearance_m",
    ):
        integrated[name] = float(integrated[name])
        if integrated[name] <= 0:
            raise ValueError(f"profile integrated_operations.{name} must be positive")
    for name in (
        "reefer_risk_recovery_rate",
        "maintenance_recovery_rate",
        "shore_power_service_efficiency",
        "shore_power_avoided_carbon_kg_per_kwh",
        "dangerous_goods_yard_occupancy_limit",
        "high_reefer_risk_ratio",
        "high_equipment_failure_risk_ratio",
        "forecast_uncertainty_review_ratio",
    ):
        integrated[name] = float(integrated[name])
        if not 0 <= integrated[name] <= 1:
            raise ValueError(f"profile integrated_operations.{name} must be in [0, 1]")
    coordinated = merged["coordinated_operations"]
    for name in (
        "rail_service_teu_per_hour",
        "barge_service_teu_per_hour",
        "pilotage_service_vessels_per_hour",
        "towage_service_vessels_per_hour",
        "quay_crane_moves_per_hour",
        "horizontal_transport_moves_per_hour",
        "yard_crane_moves_per_hour",
        "container_moves_per_teu",
    ):
        coordinated[name] = float(coordinated[name])
        if coordinated[name] <= 0:
            raise ValueError(f"profile coordinated_operations.{name} must be positive")
    for name in (
        "gate_minimum_capacity_factor",
        "intermodal_minimum_capacity_factor",
        "marine_minimum_capacity_factor",
        "terminal_chain_minimum_capacity_factor",
    ):
        coordinated[name] = float(coordinated[name])
        if not 0.5 <= coordinated[name] <= 1.0:
            raise ValueError(
                f"profile coordinated_operations.{name} must be in [0.5, 1.0]"
            )
    return merged


def load_profile(profile_id: str, profile_root: Path = DEFAULT_PROFILE_ROOT) -> Dict[str, Any]:
    resolved = validate_identifier(profile_id, field="profile_id")
    if resolved == DEFAULT_PROFILE_ID:
        return validate_profile(DEFAULT_PROFILE)
    path = profile_root / f"{resolved}.json"
    # `resolved` is one validated path component and profile_root is configured.
    # codeql[py/path-injection]
    if not path.exists():
        raise FileNotFoundError(f"port profile not found: {resolved}")
    # codeql[py/path-injection]
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
