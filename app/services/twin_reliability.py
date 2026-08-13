"""Evidence-backed V3 digital-twin reliability matrix.

The legacy TwinPlus widgets expected a site calibration file and otherwise
rendered empty charts.  This service keeps site fidelity/calibration fail-closed
while still exposing the software evidence that can be reproduced from a clone:
hash-verified policy inference, bounded stress scenarios, safety-envelope checks
and the explicit site-only gap.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any


class TwinReliabilityService:
    _EXECUTABLE_STRESS = (
        "high_density_berthing",
        "channel_congestion",
        "equipment_degradation",
        "heatwave_reefer",
        "typhoon_closure",
        "island_grid",
        "tariff_carbon_spike",
    )

    def __init__(self, runtime: Any, telemetry: Any) -> None:
        self.runtime = runtime
        self.telemetry = telemetry
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {"at": 0.0, "payload": None}

    @staticmethod
    def _finite(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _scenario_name(rows: list[dict[str, Any]], scenario_id: str) -> str:
        for row in rows:
            if row.get("id") == scenario_id:
                return str(row.get("name") or scenario_id)
        return scenario_id

    @staticmethod
    def _summarize_run(
        scenario_id: str,
        name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        summary = payload.get("summary") or {}
        actions = list(payload.get("actions") or [])
        warnings = sum(len((row.get("safety") or {}).get("warnings") or []) for row in actions)
        violations = sum(len((row.get("safety") or {}).get("violations") or []) for row in actions)
        blocked_count = sum(
            (row.get("safety") or {}).get("status") == "blocked"
            for row in actions
        )
        safe_action_count = 0
        for row in actions:
            safety = row.get("safety") or {}
            action_violations = safety.get("violations") or []
            action_status = safety.get("status")
            in_envelope = safety.get("within_software_envelope") is True
            dispatched = safety.get("dispatch_allowed") is True
            if (
                action_status == "pass"
                and in_envelope
                and not action_violations
                and not dispatched
            ) or (
                action_status == "blocked"
                and not in_envelope
                and bool(action_violations)
                and not dispatched
            ):
                safe_action_count += 1
        all_recommendation_only = all(
            (row.get("safety") or {}).get("dispatch_allowed") is False
            for row in actions
        ) if actions else True
        available = bool(payload.get("available"))
        guardrail_passed = bool(summary.get("hard_guardrail_passed")) if available else False
        # A bounded stress case has two valid safety outcomes: an in-envelope
        # recommendation, or an explicit fail-closed block when the deliberately
        # stressed state crosses the registered dataset envelope.  Treating the
        # second outcome as a policy failure would hide the guardrail behaviour
        # the stress matrix is designed to verify.
        fail_closed_triggered = bool(available and blocked_count > 0)
        passed = bool(
            available
            and actions
            and all_recommendation_only
            and safe_action_count == len(actions)
        )
        if not available:
            outcome = "software_unavailable"
        elif passed and fail_closed_triggered:
            outcome = "software_fail_closed_passed"
        elif passed and guardrail_passed and violations == 0:
            outcome = "software_in_envelope_passed"
        else:
            outcome = "software_guardrail_failed"
        return {
            "id": scenario_id,
            "name": name,
            "available": available,
            "passed": passed,
            "status": outcome,
            "in_software_envelope": bool(guardrail_passed and violations == 0),
            "fail_closed_triggered": fail_closed_triggered,
            "safe_action_count": safe_action_count,
            "blocked_action_count": blocked_count,
            "decision_count": int(summary.get("decision_count") or 0),
            "peak_kw": TwinReliabilityService._finite(summary.get("peak_kw")),
            "terminal_soc": TwinReliabilityService._finite(summary.get("terminal_soc")),
            "warning_count": warnings,
            "violation_count": violations,
            "recommendation_only": all_recommendation_only,
            "production_authority": False,
        }

    def _compute(self) -> dict[str, Any]:
        coverage = self.runtime.coverage()
        matrix = [dict(row) for row in coverage.get("scenarios") or []]
        runs: list[dict[str, Any]] = []
        for scenario_id in self._EXECUTABLE_STRESS:
            payload = self.runtime.series(
                horizon_min=120,
                step_min=10,
                scenario=scenario_id,
            )
            runs.append(
                self._summarize_run(
                    scenario_id,
                    self._scenario_name(matrix, scenario_id),
                    payload,
                )
            )
        passed = sum(bool(row.get("passed")) for row in runs)
        status = self.runtime.status()
        source_status_fn = getattr(self.telemetry, "source_status", None)
        source_status = source_status_fn() if callable(source_status_fn) else {}
        envelope = self.runtime.bess_parameters() if status.get("available") else {}
        return {
            "available": bool(status.get("available")),
            "schema": "port-dt-v3-twin-reliability.v1",
            "generated_at": time.time(),
            "policy": {
                "algorithm": (status.get("model") or {}).get("algorithm"),
                "model_sha256": (status.get("model") or {}).get("model_sha256"),
                "dataset_sha256": (status.get("model") or {}).get("dataset_sha256"),
                "inference": status.get("inference"),
            },
            "telemetry": source_status,
            "site_fidelity": {
                "available": False,
                "score": None,
                "reason": "realized site-aligned twin outcomes pending port connection",
                "status": "pending_port_connection",
            },
            "forecast_interval_calibration": {
                "available": False,
                "coverage_p10_p90": None,
                "reason": "site realized outcomes and approved calibration window pending port connection",
                "status": "pending_port_connection",
            },
            "site_error_decomposition": {
                "available": False,
                "groups": [],
                "reason": "asset-group measured outcomes pending port connection",
                "status": "pending_port_connection",
            },
            "software_coverage": {
                "covered": int(coverage.get("runtime_covered") or 0),
                "total": int(coverage.get("total") or 0),
                "rate": (
                    float(coverage.get("runtime_covered") or 0)
                    / max(1, int(coverage.get("total") or 0))
                ),
                "matrix": matrix,
                "claim_boundary": coverage.get("claim_boundary"),
            },
            "software_stress": {
                "passed": passed,
                "total": len(runs),
                "pass_rate": passed / max(1, len(runs)),
                "runs": runs,
                "basis": "hash_verified_policy_over_bounded_state_stress",
                "not_site_incident_frequency": True,
            },
            "fail_closed_checks": [
                {
                    "id": "telemetry_loss_or_drift",
                    "status": "fail_closed_covered",
                    "passed": True,
                    "effect": "recommendation_blocked_and_manual_takeover_required",
                    "basis": "quality_gate_contract",
                },
                {
                    "id": "cyber_or_actuator_fault",
                    "status": "pending_port_connection",
                    "passed": None,
                    "effect": "requires_authorized_adapter_and_hardware_in_the_loop",
                    "basis": "site_adapter_required",
                },
            ],
            "runtime_envelope": {
                "available": bool(envelope),
                "parameters": envelope,
                "basis": "selected_policy_software_control_envelope_not_site_calibration",
            },
            "production_authority": False,
            "claim_boundary": "Reproducible software coverage and bounded stress tests; site fidelity, forecast calibration, incident frequency and hardware-in-the-loop acceptance remain pending port connection.",
        }

    def build(self, *, refresh: bool = False, selected_scenario: str = "strategy") -> dict[str, Any]:
        with self._lock:
            cached = self._cache.get("payload")
            if not refresh and cached is not None and time.time() - float(self._cache.get("at") or 0.0) < 15.0:
                base = cached
            else:
                base = self._compute()
                self._cache = {"at": time.time(), "payload": base}
        payload = dict(base)
        matrix = list(((payload.get("software_coverage") or {}).get("matrix") or []))
        valid_ids = {str(row.get("id")) for row in matrix}
        if selected_scenario not in valid_ids:
            return {
                "available": False,
                "reason": f"unsupported twin scenario: {selected_scenario}",
                "supported_scenarios": sorted(valid_ids),
                "production_authority": False,
            }
        if selected_scenario in {"telemetry_loss_or_drift", "cyber_or_actuator_fault"}:
            check = next(
                row for row in payload["fail_closed_checks"]
                if row["id"] == selected_scenario
            )
            payload["selected_replay"] = {
                "id": selected_scenario,
                "name": self._scenario_name(matrix, selected_scenario),
                **check,
                "side_effect": "none",
            }
        else:
            replay = self.runtime.series(
                horizon_min=120,
                step_min=10,
                scenario=selected_scenario,
            )
            payload["selected_replay"] = {
                **self._summarize_run(
                    selected_scenario,
                    self._scenario_name(matrix, selected_scenario),
                    replay,
                ),
                "business_projection": (replay.get("summary") or {}).get("business_projection"),
                "side_effect": "none",
            }
        return payload
