from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from .profiles import DEFAULT_PROFILE, validate_profile


def assess_integrated_business_constraints(
    *,
    state: Optional[Mapping[str, Any]],
    decoded_control: Mapping[str, Any],
    demand_cap_kw: float,
    port_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fail closed around decisions that must not be delegated to a learner."""

    if state is None:
        return {
            "status": "unavailable",
            "recommendation_feasible": None,
            "dispatch_allowed": False,
            "violations": [],
            "warnings": [
                {
                    "code": "CANONICAL_STATE_REQUIRED",
                    "message": "normalized observations cannot prove maritime or terminal feasibility",
                }
            ],
            "authority": "deterministic_review_gate_no_execution_authority",
        }

    profile = validate_profile(port_profile or DEFAULT_PROFILE)
    operations = profile["integrated_operations"]
    weather_limits = profile["weather_limits"]
    violations: list[Dict[str, Any]] = []
    warnings: list[Dict[str, Any]] = []

    def value(name: str, *, required: bool = False, default: float = 0.0) -> float:
        raw = state.get(name)
        if raw is None:
            if required:
                violations.append(
                    {
                        "code": "MISSING_HARD_CONSTRAINT_INPUT",
                        "field": name,
                        "message": "deterministic feasibility cannot be established",
                    }
                )
            return default
        try:
            number = float(raw)
        except (TypeError, ValueError):
            violations.append(
                {
                    "code": "INVALID_HARD_CONSTRAINT_INPUT",
                    "field": name,
                    "message": "value must be numeric",
                }
            )
            return default
        if not math.isfinite(number):
            violations.append(
                {
                    "code": "NON_FINITE_HARD_CONSTRAINT_INPUT",
                    "field": name,
                    "message": "value must be finite",
                }
            )
            return default
        return number

    tide = value("tide_m", required=True)
    chart_depth = value("channel_chart_depth_m", required=True)
    draft = value("planning_vessel_draft_m", required=True)
    squat = value("squat_allowance_m", required=True)
    under_keel_clearance = chart_depth + tide - draft - squat
    required_clearance = float(operations["required_under_keel_clearance_m"])
    marine_allocation = float(
        decoded_control.get("marine_service_allocation_ratio", 0.0)
    )
    if under_keel_clearance < required_clearance:
        violations.append(
            {
                "code": "UNDER_KEEL_CLEARANCE",
                "field": "under_keel_clearance_m",
                "value": under_keel_clearance,
                "required_min": required_clearance,
                "message": "channel transit must remain closed regardless of policy recommendation",
            }
        )

    closure = value("closure_flag", required=True)
    if closure >= 0.5 and marine_allocation > 1e-9:
        violations.append(
            {
                "code": "CHANNEL_CLOSED",
                "field": "closure_flag",
                "value": closure,
                "message": "a learner cannot override a declared channel closure",
            }
        )
    pilot_tug = value("pilot_tug_availability_ratio", required=True, default=0.0)
    channel_capacity = value("channel_capacity_ratio", required=True, default=0.0)
    if marine_allocation > 0.0 and (pilot_tug < 0.5 or channel_capacity < 0.3):
        violations.append(
            {
                "code": "MARINE_RESOURCE_INFEASIBLE",
                "field": "pilot_tug_availability_ratio/channel_capacity_ratio",
                "message": "pilot, tug and channel capacity must be confirmed by their authoritative systems",
            }
        )

    for factor_name, limit_name, direction in (
        ("wind_speed_mps", "wind_stop_mps", "high"),
        ("visibility_km", "visibility_stop_km", "low"),
        ("wave_height_m", "wave_stop_m", "high"),
    ):
        limit = weather_limits.get(limit_name)
        raw = state.get(factor_name)
        if limit is None:
            continue
        if raw is None:
            violations.append(
                {
                    "code": "MISSING_WEATHER_LIMIT_INPUT",
                    "field": factor_name,
                    "message": "weather-dependent marine feasibility cannot be established",
                }
            )
            continue
        observed = value(factor_name)
        exceeded = observed >= float(limit) if direction == "high" else observed <= float(limit)
        if exceeded:
            violations.append(
                {
                    "code": "WEATHER_STOP_LIMIT",
                    "field": factor_name,
                    "value": observed,
                    "limit": float(limit),
                }
            )

    yard_occupancy = value("yard_occupancy_ratio", required=True)
    dangerous_goods = value("dangerous_goods_workload_ratio", required=True)
    yard_flow = float(decoded_control.get("yard_flow_command", 0.0))
    if (
        dangerous_goods > 0.0
        and yard_occupancy > float(operations["dangerous_goods_yard_occupancy_limit"])
        and yard_flow > 0.0
    ):
        violations.append(
            {
                "code": "DANGEROUS_GOODS_YARD_LIMIT",
                "field": "yard_flow_command",
                "message": "positive yard inflow is blocked until dangerous-goods segregation capacity is restored",
            }
        )

    reefer_risk = value("reefer_temperature_risk_ratio", required=True)
    reefer_service = float(decoded_control.get("reefer_service_ratio", 0.0))
    if reefer_service < 0.5:
        violations.append(
            {
                "code": "REEFER_SAFETY_RESERVE",
                "field": "reefer_service_ratio",
                "message": "cold-chain integrity requires the declared minimum baseline service reserve",
            }
        )

    failure_risk = value("equipment_failure_risk_ratio", required=True)
    maintenance_reserve = float(
        decoded_control.get("maintenance_reserve_ratio", 0.0)
    )
    if maintenance_reserve < 0.5:
        violations.append(
            {
                "code": "MAINTENANCE_SAFETY_RESERVE",
                "field": "maintenance_reserve_ratio",
                "message": "the minimum preventive-maintenance reserve cannot be traded away for short-term throughput",
            }
        )

    base_load_kw = value("base_load_kw", required=True)
    bess_kw = float(decoded_control.get("bess_kw", 0.0))
    flexible = float(decoded_control.get("flexible_load_command", 0.0))
    shore_demand = value("shore_power_demand_kw", required=True)
    shore_connection = value("shore_power_connection_ratio", required=True)
    shore_allocation = float(
        decoded_control.get("shore_power_allocation_ratio", 0.0)
    )
    flexible_kw = flexible * min(250.0, 0.08 * max(base_load_kw, 1.0))
    estimated_shore_kw = (
        shore_demand
        * shore_connection
        * float(operations["shore_power_service_efficiency"])
        * (0.35 + 0.65 * shore_allocation)
    )
    estimated_net_kw = max(
        0.0, base_load_kw + bess_kw + flexible_kw + estimated_shore_kw
    )
    if estimated_net_kw > float(demand_cap_kw) + 1e-6:
        violations.append(
            {
                "code": "INTEGRATED_DEMAND_CAP",
                "field": "estimated_net_load_kw",
                "value": estimated_net_kw,
                "allowed_max": float(demand_cap_kw),
            }
        )

    uncertainty = value("forecast_uncertainty_ratio", required=True)
    if uncertainty >= float(operations["forecast_uncertainty_review_ratio"]):
        warnings.append(
            {
                "code": "HIGH_FORECAST_UNCERTAINTY",
                "field": "forecast_uncertainty_ratio",
                "value": uncertainty,
                "message": "freeze the near-term plan and obtain operator review before release",
            }
        )

    feasible = not violations
    return {
        "status": "pass" if feasible and not warnings else "review_required" if feasible else "blocked",
        "recommendation_feasible": feasible,
        "dispatch_allowed": False,
        "human_review_required": True,
        "violations": violations,
        "warnings": warnings,
        "estimates": {
            "under_keel_clearance_m": under_keel_clearance,
            "required_under_keel_clearance_m": required_clearance,
            "integrated_net_load_kw": estimated_net_kw,
        },
        "fallback": {
            "channel_transit_allowed": feasible and under_keel_clearance >= required_clearance and closure < 0.5,
            "automatic_execution": False,
            "instruction": "hold affected task, retain incumbent feasible plan, and escalate to the named human approval workflow",
        },
        "authority": "deterministic_review_gate_no_release_navigation_or_actuator_authority",
    }
