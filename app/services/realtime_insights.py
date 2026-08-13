"""Evidence-backed realtime dashboard insights.

This replaces the legacy random dashlet simulators with measurements computed
from the active telemetry adapter, Ridge quantiles and the selected SAC runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import statistics
from typing import Any


class RealtimeInsightsService:
    def __init__(self, container: Any, peak_risk_service: Any) -> None:
        self.di = container
        self.peak_risk_service = peak_risk_service

    @staticmethod
    def _parse_ts(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _asset(self, asset_id: str) -> dict[str, Any]:
        for row in self.di.telemetry.list_assets() or []:
            if str(row.get("id")) == asset_id:
                return dict(row)
        return {}

    def _quality(self, asset: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        samples: list[tuple[datetime, float]] = []
        for row in rows:
            ts = self._parse_ts(row.get("ts"))
            value = self._number(row.get("kW"))
            if ts is not None and value is not None:
                samples.append((ts, value))
        samples.sort(key=lambda item: item[0])
        gaps = [
            (samples[index][0] - samples[index - 1][0]).total_seconds()
            for index in range(1, len(samples))
            if samples[index][0] > samples[index - 1][0]
        ]
        cadence = statistics.median(gaps) if gaps else None
        missing_steps = 0
        if cadence and cadence > 0:
            missing_steps = sum(max(0, round(gap / cadence) - 1) for gap in gaps)
        expected = len(samples) + missing_steps
        missing_rate = missing_steps / expected if expected else None
        latest = samples[-1][0] if samples else None
        stale_seconds = (
            max(0.0, (datetime.now(timezone.utc) - latest.astimezone(timezone.utc)).total_seconds())
            if latest
            else None
        )
        stale_rate = (
            1.0 if stale_seconds is not None and cadence and stale_seconds > max(5.0, 3.0 * cadence) else 0.0
        ) if stale_seconds is not None else None
        rated_kw = self._number(asset.get("rated_kw"))
        out_of_range = None
        if samples and rated_kw and rated_kw > 0:
            out_of_range = sum(1 for _, value in samples if abs(value) > rated_kw) / len(samples)
        return {
            "available": bool(samples),
            "sample_count": len(samples),
            "cadence_seconds": round(cadence, 3) if cadence is not None else None,
            "missing_rate": round(missing_rate, 6) if missing_rate is not None else None,
            "stale_rate": stale_rate,
            "stale_seconds": round(stale_seconds, 3) if stale_seconds is not None else None,
            "out_of_engineering_range_rate": round(out_of_range, 6) if out_of_range is not None else None,
            "rated_kw": rated_kw,
            "latest_at": latest.isoformat() if latest else None,
            "basis": "active_adapter_sequence_continuity_and_engineering_asset_rating",
            "site_sensor_quality": "pending_port_connection",
        }

    @staticmethod
    def _risk_at(risk_rows: list[dict[str, Any]], minutes: int, step_min: int) -> float | None:
        if not risk_rows:
            return None
        count = max(1, min(len(risk_rows), math.ceil(minutes / max(1, step_min))))
        values = [float(row.get("p") or 0.0) for row in risk_rows[:count]]
        return max(values) if values else None

    def build(
        self,
        *,
        asset_id: str,
        mode: str,
        cap_kw: float,
        horizon_min: int = 60,
        step_min: int = 5,
    ) -> dict[str, Any]:
        asset = self._asset(asset_id)
        if not asset:
            return {
                "available": False,
                "reason": "asset is not registered in the active telemetry adapter",
                "production_authority": False,
            }
        rows = list(self.di.telemetry.get_recent_power(asset_id) or [])
        source_status_fn = getattr(self.di.telemetry, "source_status", None)
        source_status = source_status_fn() if callable(source_status_fn) else {}
        quality = self._quality(asset, rows)

        forecast_map = self.di.fcst.forecast_load(
            [asset_id], horizon_min=horizon_min, step_min=step_min
        ) or {}
        forecast = list(forecast_map.get(asset_id) or [])
        forecast_peak = max(forecast, key=lambda row: float(row.get("p50", row.get("kW", 0.0)) or 0.0)) if forecast else {}
        risk_payload = self.peak_risk_service.peak_risk(
            mode="forecast",
            horizon_min=horizon_min,
            step_min=step_min,
            cap_kw=cap_kw,
            avg_window_min=15,
        )
        risk_rows = ((risk_payload.get("series") or {}).get("risk") or []) if risk_payload.get("available") else []
        risk = {
            "available": bool(risk_rows),
            "m15": self._risk_at(risk_rows, 15, step_min),
            "m30": self._risk_at(risk_rows, 30, step_min),
            "m60": self._risk_at(risk_rows, 60, step_min),
            "peak_probability": risk_payload.get("peak_risk"),
            "peak_p50_kw": risk_payload.get("peak_p50_max"),
            "cap_kw": cap_kw,
            "logic": risk_payload.get("risk_logic"),
            "quantile_source": risk_payload.get("quantile_source"),
            "site_contract_demand": "pending_port_connection",
        }

        state_fn = getattr(self.di.telemetry, "current_port_state", None)
        state = state_fn() if callable(state_fn) else {}
        events: list[dict[str, Any]] = []
        if forecast_peak:
            events.append(
                {
                    "kind": "RIDGE PEAK",
                    "ts": forecast_peak.get("ts"),
                    "desc": f"{asset.get('label') or asset_id} P50 峰值 {float(forecast_peak.get('p50', forecast_peak.get('kW', 0.0)) or 0.0):.0f} kW",
                    "source": "ridge_autoregression",
                }
            )
        if risk.get("peak_probability") is not None:
            events.append(
                {
                    "kind": "DEMAND",
                    "ts": forecast_peak.get("ts"),
                    "desc": f"全港 15min 需量越限概率峰值 {float(risk['peak_probability']) * 100:.1f}%（工程阈值 {cap_kw:.0f} kW）",
                    "source": "ridge_p10_p50_p90_quantile_probability",
                }
            )
        if state.get("visibility_km") is None:
            events.append(
                {
                    "kind": "DATA GAP",
                    "ts": None,
                    "desc": "能见度待接入港口；SAC 安全门保持无现场执行权",
                    "source": "canonical_state_contract",
                }
            )

        business = {
            "available": False,
            "reason": "switch_to_strategy_simulation_for_model_derived_value",
            "avoided_energy_cost_cny": None,
            "avoided_carbon_kg": None,
        }
        if mode == "sim":
            strategy = self.di.strategy_runtime.series(
                horizon_min=360, step_min=5, scenario="strategy"
            )
            value = (((strategy.get("summary") or {}).get("business_projection") or {}).get("equivalent_throughput_value") or {})
            cost = self._number(value.get("avoided_energy_cost"))
            carbon = self._number(value.get("avoided_carbon_kg"))
            business = {
                "available": cost is not None and carbon is not None,
                "reason": None if cost is not None and carbon is not None else "selected_policy_value_unavailable",
                "avoided_energy_cost_cny": cost,
                "avoided_carbon_kg": carbon,
                "comparison_basis": value.get("comparison_basis"),
                "financial_audit_ready": False,
                "site_tariff_contract": "pending_port_connection",
            }

        return {
            "available": True,
            "schema": "port-dt-v3-realtime-insights.v1",
            "asset": asset,
            "mode": mode,
            "telemetry": {
                **source_status,
                "sample_count": quality.get("sample_count"),
                "latest_at": quality.get("latest_at"),
            },
            "quality": quality,
            "forecast": {
                "available": bool(forecast),
                "model": forecast[0].get("model") if forecast else None,
                "stability_guard": forecast[0].get("stability_guard") if forecast else None,
                "stability_cap_kw": forecast[0].get("stability_cap_kw") if forecast else None,
                "guard_applied_count": sum(bool(row.get("guard_applied")) for row in forecast),
                "step_min": step_min,
                "horizon_min": horizon_min,
                "point_count": len(forecast),
                "peak": forecast_peak or None,
                "calibration": {
                    "available": False,
                    "mape": None,
                    "coverage_p10_p90": None,
                    "bias_kw": None,
                    "residual_sigma_kw": None,
                    "reason": "realized site-aligned outcomes pending port connection; no in-sample residual is shown as forecast evidence",
                },
            },
            "peak_risk": risk,
            "business_value": business,
            "events": events,
            "approvals": {
                "available": False,
                "pending": None,
                "last_job": None,
                "reason": "southbound approval ledger pending port connection",
            },
            "claim_boundary": "Public-data calibrated continuous simulator and model outputs; not terminal telemetry, site forecast calibration, contract-demand evidence or production alarms.",
            "production_authority": False,
        }
