from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .datasets import NUMERIC_COLUMNS, PortDataset


def assess_recommendation(
    *,
    state: Optional[Mapping[str, Any]],
    decoded_control: Mapping[str, Any],
    dataset: PortDataset,
    demand_cap_kw: float,
    bess_power_kw: float,
) -> Dict[str, Any]:
    """Assess a policy recommendation without mutating or dispatching it.

    This is a conservative software envelope. It is intentionally not a claim
    that site PLC/BMS interlocks, electrical studies, or operating permits have
    been validated.
    """
    if state is None:
        return {
            "status": "unavailable",
            "within_software_envelope": None,
            "dispatch_allowed": False,
            "violations": [],
            "warnings": ["normalized observations cannot be checked against canonical engineering units"],
            "required_next_step": "submit a canonical state and obtain site safety approval",
            "authority": "recommendation_only_no_actuator_authority",
        }

    violations: list[Dict[str, Any]] = []
    warnings: list[Dict[str, Any]] = []
    values: Dict[str, float] = {}
    for index, column in enumerate(NUMERIC_COLUMNS):
        try:
            value = float(state[column])
        except (KeyError, TypeError, ValueError):
            violations.append({"code": "INVALID_STATE", "field": column, "message": "missing or non-numeric canonical value"})
            continue
        if not math.isfinite(value):
            violations.append({"code": "NON_FINITE_STATE", "field": column, "message": "value must be finite"})
            continue
        values[column] = value
        observed_min = float(np.min(dataset.values[:, index]))
        observed_max = float(np.max(dataset.values[:, index]))
        span = max(observed_max - observed_min, 1e-6)
        if value < observed_min - 0.10 * span or value > observed_max + 0.10 * span:
            violations.append({
                "code": "OUT_OF_DISTRIBUTION",
                "field": column,
                "value": value,
                "observed_range": [observed_min, observed_max],
                "message": "value is outside the dataset range plus 10% tolerance",
            })
        elif value < observed_min or value > observed_max:
            warnings.append({
                "code": "EDGE_OF_DISTRIBUTION",
                "field": column,
                "value": value,
                "observed_range": [observed_min, observed_max],
            })

    def number(name: str, default: float) -> float:
        try:
            value = float(state.get(name, default))
        except (TypeError, ValueError):
            violations.append({"code": "INVALID_STATE", "field": name, "message": "value must be numeric"})
            return default
        if not math.isfinite(value):
            violations.append({"code": "NON_FINITE_STATE", "field": name, "message": "value must be finite"})
            return default
        return value

    soc = number("soc", 0.55)
    last_bess_kw = number("last_bess_kw", 0.0)
    bess_kw = float(decoded_control.get("bess_kw", 0.0))
    service_factor = float(decoded_control.get("service_factor", 1.0))
    flexible = float(decoded_control.get("flexible_load_command", 0.0))

    if not 0.10 <= soc <= 0.90:
        violations.append({"code": "SOC_LIMIT", "field": "soc", "value": soc, "allowed": [0.10, 0.90]})
    if abs(bess_kw) > bess_power_kw + 1e-6:
        violations.append({"code": "BESS_POWER_LIMIT", "field": "bess_kw", "value": bess_kw, "allowed": [-bess_power_kw, bess_power_kw]})
    # The environment advances in one-hour intervals and permits a full
    # nameplate power transition within a step. Keep inference safety checks
    # aligned with the same declared, conservative hourly envelope.
    ramp_limit = bess_power_kw
    if abs(bess_kw - last_bess_kw) > ramp_limit + 1e-6:
        violations.append({"code": "BESS_RAMP_LIMIT", "field": "bess_kw", "value": bess_kw, "last": last_bess_kw, "max_delta": ramp_limit})
    if not 0.75 <= service_factor <= 1.25:
        violations.append({"code": "SERVICE_FACTOR_LIMIT", "field": "service_factor", "value": service_factor, "allowed": [0.75, 1.25]})
    if not -0.60 <= flexible <= 0.60:
        violations.append({"code": "FLEXIBLE_LOAD_LIMIT", "field": "flexible_load_command", "value": flexible, "allowed": [-0.60, 0.60]})

    estimated_net_kw = None
    if "base_load_kw" in values:
        flexible_kw = flexible * min(250.0, 0.08 * max(values["base_load_kw"], 1.0))
        estimated_net_kw = max(0.0, values["base_load_kw"] + bess_kw + flexible_kw)
        if estimated_net_kw > float(demand_cap_kw):
            violations.append({
                "code": "DEMAND_CAP",
                "field": "estimated_net_load_kw",
                "value": estimated_net_kw,
                "allowed_max": float(demand_cap_kw),
            })

    within = not violations
    return {
        "status": "pass" if within else "blocked",
        "within_software_envelope": within,
        "dispatch_allowed": False,
        "human_review_eligible": within,
        "violations": violations,
        "warnings": warnings,
        "estimates": {"net_load_kw": estimated_net_kw, "bess_ramp_limit_kw": ramp_limit},
        "authority": "recommendation_only_no_actuator_authority",
        "required_next_step": "site adapter, calibrated limits, operator approval, and independent hardware interlocks",
    }
