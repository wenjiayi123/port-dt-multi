"""Deterministic, data-driven load-twin adapter.

The twin consumes the configured forecasting model and applies transparent
scenario stress parameters. It does not invent device histories, random outage
events, or percentage uncertainty bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Scenario:
    name: str
    load_multiplier: float
    note: str


SCENARIOS: Dict[str, Scenario] = {
    "baseline": Scenario("baseline", 1.00, "数据驱动基线"),
    "heatwave": Scenario("heatwave", 1.08, "高温压力参数：负荷乘数 1.08"),
    "typhoon": Scenario("typhoon", 0.70, "台风停工压力参数：负荷乘数 0.70"),
    "dense_berthing": Scenario("dense_berthing", 1.25, "密集靠泊压力参数：负荷乘数 1.25"),
    "islanded": Scenario("islanded", 0.85, "孤网限载压力参数：负荷乘数 0.85"),
}


class TwinService:
    def __init__(
        self,
        fcst: Any = None,
        telemetry: Any = None,
        schedule: Any = None,
        forecast: Any = None,
    ) -> None:
        self.forecast = fcst or forecast
        self.telemetry = telemetry
        self.schedule = schedule

    def run(
        self,
        asset_id: str,
        horizon_min: int = 360,
        step_min: int = 1,
        scenario: str = "baseline",
        use_drivers: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:
        scenario_key = str(scenario or "baseline").lower()
        if scenario_key not in SCENARIOS:
            raise ValueError(f"unsupported twin scenario: {scenario_key}")
        setting = SCENARIOS[scenario_key]
        if self.forecast is None:
            return self._unavailable(asset_id, setting, "forecast adapter is unavailable")

        explicit_drivers: Optional[Dict[str, Any]] = None
        if use_drivers:
            explicit_drivers = {"load_multiplier": setting.load_multiplier}
        try:
            forecast_map = self.forecast.forecast_load(
                [asset_id],
                horizon_min=horizon_min,
                step_min=step_min,
                drivers=explicit_drivers,
                scenario=scenario_key,
                return_quantiles=True,
            ) or {}
        except Exception as exc:
            return self._unavailable(asset_id, setting, f"forecast adapter failed: {exc}")

        points = [point for point in (forecast_map.get(asset_id) or []) if isinstance(point, dict)]
        if not points:
            return self._unavailable(asset_id, setting, "insufficient telemetry history for forecasting")

        plan = []
        for point in points:
            p50 = float(point.get("p50", point.get("kW", 0.0)) or 0.0)
            row = {
                "ts": point.get("ts"),
                "kW": p50,
                "p50": p50,
                "model_step_min": int(point.get("model_step_min") or step_min),
            }
            if point.get("p10") is not None:
                row["p10"] = float(point["p10"])
            if point.get("p90") is not None:
                row["p90"] = float(point["p90"])
            plan.append(row)

        values = [point["p50"] for point in plan]
        effective_step_min = int(plan[0].get("model_step_min") or step_min)
        return {
            "available": True,
            "asset": asset_id,
            "scenario": setting.name,
            "scenario_parameters": {"load_multiplier": setting.load_multiplier},
            "scenario_note": setting.note,
            "window": {
                "horizon_min_requested": horizon_min,
                "step_min_requested": step_min,
                "step_min_effective": effective_step_min,
                "start": plan[0].get("ts"),
                "end": plan[-1].get("ts"),
            },
            "summary": {
                "avgKW": round(sum(values) / len(values), 6),
                "peak_kW_p50": round(max(values), 6),
                "total_kWh_p50": round(sum(values) * effective_step_min / 60.0, 6),
            },
            "plan": plan,
            "uncertainty_available": all("p10" in point and "p90" in point for point in plan),
            "_source": "ridge_autoregression_plus_explicit_scenario_parameter",
            "production": False,
        }

    @staticmethod
    def _unavailable(asset_id: str, scenario: Scenario, reason: str) -> Dict[str, Any]:
        return {
            "available": False,
            "asset": asset_id,
            "scenario": scenario.name,
            "scenario_parameters": {"load_multiplier": scenario.load_multiplier},
            "reason": reason,
            "summary": {},
            "plan": [],
            "_source": "twin_unavailable",
            "production": False,
        }
