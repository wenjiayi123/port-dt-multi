"""Clone-safe runtime bridge from calibrated telemetry to the selected V3 policy."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.safety import assess_recommendation
from app.services.rl_training.trainer import SB3_IMPORT_LOCK, TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "evidence/v3/runtime"


class V3RuntimeService:
    """Runs deterministic SAC inference against the same port_ops_v3 contract."""

    def __init__(self, container: Any) -> None:
        self.di = container
        self._lock = threading.Lock()
        self._policy: Any = None
        self._env: Any = None
        self._metadata: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._load_error: str | None = None
        self._cache: dict[str, Any] = {"key": None, "at": 0.0, "value": None}

    @staticmethod
    def _json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object: {path}")
        return payload

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _ensure_loaded(self) -> None:
        if self._policy is not None or self._load_error is not None:
            return
        with self._lock:
            if self._policy is not None or self._load_error is not None:
                return
            try:
                metadata_path = RUNTIME_ROOT / "runtime_model.json"
                config_path = RUNTIME_ROOT / "selected_sac_v3.config.json"
                model_path = RUNTIME_ROOT / "selected_sac_v3.zip"
                metadata = self._json(metadata_path)
                config = self._json(config_path)
                if metadata.get("schema") != "port-dt-v3-runtime-policy.v1":
                    raise ValueError("runtime policy schema mismatch")
                if self._sha(model_path) != metadata.get("model_sha256"):
                    raise ValueError("runtime policy model hash mismatch")
                if self._sha(config_path) != metadata.get("config_sha256"):
                    raise ValueError("runtime policy config hash mismatch")
                dataset = load_port_dataset(str(config["dataset_id"]), ROOT / "data/rl/datasets")
                if dataset.fingerprint != metadata.get("dataset_sha256"):
                    raise ValueError("runtime policy dataset hash mismatch")
                env = TRAINING_MANAGER._make_env(
                    dataset,
                    config,
                    training=False,
                    record_trace=False,
                )
                with SB3_IMPORT_LOCK:
                    from stable_baselines3 import SAC

                    policy = SAC.load(model_path, env=env, device="cpu")
                self._metadata, self._config = metadata, config
                self._env, self._policy = env, policy
            except Exception:
                self._load_error = "runtime policy could not be loaded; inspect server logs"

    def status(self) -> dict[str, Any]:
        self._ensure_loaded()
        telemetry_status_fn = getattr(self.di.telemetry, "source_status", None)
        telemetry = telemetry_status_fn() if callable(telemetry_status_fn) else {"mode": "unavailable"}
        return {
            "available": self._policy is not None,
            "model": self._metadata,
            "model_error": self._load_error,
            "telemetry": telemetry,
            "inference": "deterministic_saved_policy" if self._policy is not None else "unavailable",
            "production_authority": False,
        }

    @staticmethod
    def coverage() -> dict[str, Any]:
        """Declare what the offline twin covers and what still needs site data."""
        scenarios = [
            {
                "id": "strategy",
                "name": "常态作业 / 公开数据连续回放",
                "state": "runtime_covered",
                "basis": "public_dataset_range",
                "checks": ["泊位与堆场占用", "岸桥/设备可用率", "船舶到港", "通道拥堵", "能源与碳因子"],
            },
            {
                "id": "high_density_berthing",
                "name": "高密靠泊 / 吞吐激增",
                "state": "stress_test_covered",
                "basis": "bounded_state_stress",
                "checks": ["到港密度", "泊位占用", "队列与吞吐服务分配"],
            },
            {
                "id": "channel_congestion",
                "name": "航道 / 闸口拥堵",
                "state": "stress_test_covered",
                "basis": "bounded_state_stress",
                "checks": ["通道拥堵率", "队列累积", "场桥与 AGV 联动"],
            },
            {
                "id": "equipment_degradation",
                "name": "岸桥 / 设备降级",
                "state": "stress_test_covered",
                "basis": "bounded_state_stress",
                "checks": ["岸桥可用率", "设备可用率", "安全投影与服务降级"],
            },
            {
                "id": "heatwave_reefer",
                "name": "高温 / 冷藏箱与冷站压力",
                "state": "stress_test_covered",
                "basis": "bounded_state_stress",
                "checks": ["环境温度", "冷藏箱负荷", "可柔性负荷指令"],
            },
            {
                "id": "typhoon_closure",
                "name": "台风 / 封航与低能见度",
                "state": "stress_test_covered",
                "basis": "bounded_state_stress",
                "checks": ["风速", "波高", "能见度", "封航标志", "失效安全"],
            },
            {
                "id": "island_grid",
                "name": "孤网 / 需量受限",
                "state": "stress_test_covered",
                "basis": "bounded_state_stress",
                "checks": ["BESS SOC", "储能功率", "需量上限", "负荷削减"],
            },
            {
                "id": "tariff_carbon_spike",
                "name": "电价 / 碳因子峰值",
                "state": "stress_test_covered",
                "basis": "bounded_state_stress",
                "checks": ["分时电价", "电网碳因子", "单位吞吐成本与碳强度"],
            },
            {
                "id": "telemetry_loss_or_drift",
                "name": "遥测丢失 / 分布漂移",
                "state": "fail_closed_covered",
                "basis": "quality_gate",
                "checks": ["缺失字段", "超分布输入", "禁止推荐", "人工接管"],
            },
            {
                "id": "cyber_or_actuator_fault",
                "name": "网络 / 执行器异常",
                "state": "contract_only",
                "basis": "site_adapter_required",
                "checks": ["鉴权", "双人确认", "幂等", "回滚", "PLC/BMS 硬联锁"],
            },
        ]
        return {
            "schema": "port-dt-v3-scenario-coverage.v1",
            "scenarios": scenarios,
            "runtime_covered": sum(row["state"] in {"runtime_covered", "stress_test_covered", "fail_closed_covered"} for row in scenarios),
            "total": len(scenarios),
            "claim_boundary": "Offline coverage matrix, not proof that every site event has been observed. Cyber/actuator faults require authorized site adapters and hardware-in-the-loop acceptance.",
        }

    @staticmethod
    def _apply_scenario(state: dict[str, Any], scenario: str) -> dict[str, Any]:
        row = dict(state)
        if scenario in {"strategy", "normal", "baseline"}:
            return row
        if scenario == "high_density_berthing":
            row["vessel_arrivals"] = float(row.get("vessel_arrivals", 0.0)) * 1.35
            row["berth_occupancy_ratio"] = min(1.0, float(row.get("berth_occupancy_ratio", 0.0)) + 0.18)
            row["throughput_teu"] = float(row.get("throughput_teu", 0.0)) * 1.12
            row["queue"] = float(row.get("queue", 0.0)) * 1.30
        elif scenario == "channel_congestion":
            row["channel_congestion_ratio"] = min(1.0, float(row.get("channel_congestion_ratio", 0.0)) + 0.30)
            row["queue"] = float(row.get("queue", 0.0)) * 1.55
        elif scenario == "equipment_degradation":
            row["crane_availability_ratio"] = max(0.2, float(row.get("crane_availability_ratio", 1.0)) * 0.70)
            row["equipment_availability_ratio"] = max(0.2, float(row.get("equipment_availability_ratio", 1.0)) * 0.65)
            row["queue"] = float(row.get("queue", 0.0)) * 1.25
        elif scenario == "heatwave_reefer":
            row["ambient_c"] = float(row.get("ambient_c", 20.0)) + 8.0
            row["reefer_load_kw"] = float(row.get("reefer_load_kw", 0.0)) * 1.30
            row["base_load_kw"] = float(row.get("base_load_kw", 0.0)) * 1.06
        elif scenario == "typhoon_closure":
            row.update(wind_speed_mps=28.0, visibility_km=0.8, wave_height_m=4.5, closure_flag=1.0)
            row["pilot_tug_availability_ratio"] = min(0.25, float(row.get("pilot_tug_availability_ratio", 1.0)))
        elif scenario == "island_grid":
            row["price_per_kwh"] = float(row.get("price_per_kwh", 0.8)) * 1.50
            row["base_load_kw"] = float(row.get("base_load_kw", 0.0)) * 1.08
        elif scenario == "tariff_carbon_spike":
            row["price_per_kwh"] = float(row.get("price_per_kwh", 0.8)) * 1.80
            row["carbon_kg_per_kwh"] = float(row.get("carbon_kg_per_kwh", 0.5)) * 1.35
        return row

    def _baseline_forecasts(self, horizon_min: int, step_min: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        assets = [
            row for row in (self.di.telemetry.list_assets() or [])
            if row.get("include_in_aggregate", True)
        ]
        ids = [str(row["id"]) for row in assets]
        forecast_map = self.di.fcst.forecast_load(
            ids,
            horizon_min=horizon_min,
            step_min=step_min,
            return_quantiles=True,
        ) or {}
        lengths = [len(forecast_map.get(asset_id) or []) for asset_id in ids]
        count = min(lengths) if lengths and all(lengths) else 0
        aggregate: list[dict[str, Any]] = []
        for index in range(count):
            rows = [forecast_map[asset_id][index] for asset_id in ids]
            aggregate.append(
                {
                    "ts": rows[0].get("ts"),
                    "p50": sum(float(row.get("p50", row.get("kW", 0.0)) or 0.0) for row in rows),
                    "p10": sum(float(row.get("p10", row.get("p50", row.get("kW", 0.0))) or 0.0) for row in rows),
                    "p90": sum(float(row.get("p90", row.get("p50", row.get("kW", 0.0))) or 0.0) for row in rows),
                }
            )
        return aggregate, {asset_id: list(forecast_map.get(asset_id) or [])[:count] for asset_id in ids}

    def _policy_control(self, state: dict[str, Any], *, soc: float, last_bess_kw: float) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
        observation = self._env.observation_from_state(
            {**state, "soc": soc, "last_bess_kw": last_bess_kw}
        )
        raw_action, _ = self._policy.predict(observation, deterministic=True)
        control = self._env.project_control(
            raw_action,
            soc=soc,
            last_bess_kw=last_bess_kw,
        )
        safety = assess_recommendation(
            state={**state, "soc": soc, "last_bess_kw": last_bess_kw},
            decoded_control=control,
            dataset=self._env.dataset,
            demand_cap_kw=float(self._config["demand_cap_kw"]),
            bess_power_kw=float(self._env.bess_power_kw),
            port_profile=self._config.get("port_profile"),
        )
        return control, safety, np.asarray(raw_action, dtype=float).reshape(-1).tolist()

    def _business_projection(
        self,
        decision_states: list[tuple[dict[str, Any], dict[str, Any]]],
        baseline: list[dict[str, Any]],
        policy_series: list[dict[str, Any]],
        states: list[dict[str, Any]],
        step_min: int,
    ) -> dict[str, Any]:
        """Apply the environment's declared service/queue equations open-loop."""
        base_queue = policy_queue = 0.0
        base_served = policy_served = 0.0
        base_delay: list[float] = []
        policy_delay: list[float] = []
        service_factors: list[float] = []
        limits = self._config["port_profile"]["weather_limits"]
        for state, control in decision_states:
            availability = [
                float(state[name])
                for name in (
                    "crane_availability_ratio",
                    "equipment_availability_ratio",
                    "pilot_tug_availability_ratio",
                )
                if state.get(name) is not None
            ]
            resource = float(np.prod(availability)) if availability else 1.0
            congestion = state.get("channel_congestion_ratio")
            if congestion is not None:
                resource *= max(0.2, 1.0 - 0.35 * float(congestion))
            weather_blocked = float(state.get("closure_flag") or 0.0) >= 0.5
            for factor, limit_name, comparison in (
                ("wind_speed_mps", "wind_stop_mps", "high"),
                ("visibility_km", "visibility_stop_km", "low"),
                ("wave_height_m", "wave_stop_m", "high"),
            ):
                value, limit = state.get(factor), limits.get(limit_name)
                if value is None or limit is None:
                    continue
                weather_blocked = weather_blocked or (
                    float(value) >= float(limit)
                    if comparison == "high"
                    else float(value) <= float(limit)
                )
            if weather_blocked:
                resource = 0.0
            demand = max(0.0, float(state.get("throughput_teu") or 0.0) + 2.0 * float(state.get("vessel_arrivals") or 0.0))
            base_capacity = max(1.0, float(state.get("throughput_teu") or 0.0)) * resource
            allocation = max(
                0.6,
                1.0
                + 0.08 * float(control.get("berth_priority") or 0.0)
                + 0.08 * float(control.get("yard_flow_command") or 0.0),
            )
            service = float(control.get("service_factor") or 1.0)
            policy_capacity = base_capacity * service * allocation
            base_step = min(base_queue + demand, base_capacity)
            policy_step = min(policy_queue + demand, policy_capacity)
            base_queue = max(0.0, base_queue + demand - base_step)
            policy_queue = max(0.0, policy_queue + demand - policy_step)
            base_served += base_step
            policy_served += policy_step
            base_delay.append(base_queue / max(1.0, demand))
            policy_delay.append(policy_queue / max(1.0, demand))
            service_factors.append(service)

        hours = max(1, int(step_min)) / 60.0
        count = min(len(baseline), len(policy_series), len(states))
        base_cost = policy_cost = base_carbon = policy_carbon = 0.0
        for index in range(count):
            price = max(0.0, float(states[index].get("price_per_kwh") or 0.0))
            carbon_factor = max(0.0, float(states[index].get("carbon_kg_per_kwh") or 0.0))
            base_kw = float(baseline[index]["p50"])
            policy_kw = float(policy_series[index]["kW"])
            base_cost += base_kw * hours * price
            policy_cost += policy_kw * hours * price
            base_carbon += base_kw * hours * carbon_factor
            policy_carbon += policy_kw * hours * carbon_factor

        def improve(new: float, old: float, *, lower_better: bool = False) -> float | None:
            if abs(old) < 1e-9:
                return None
            return ((old - new) if lower_better else (new - old)) / abs(old) * 100.0

        base_delay_mean = sum(base_delay) / len(base_delay) if base_delay else 0.0
        policy_delay_mean = sum(policy_delay) / len(policy_delay) if policy_delay else 0.0
        equivalent_cost = None
        equivalent_carbon = None
        avoided_cost = None
        avoided_carbon = None
        if base_served > 1e-9 and policy_served > 1e-9:
            throughput_scale = policy_served / base_served
            equivalent_cost = base_cost * throughput_scale
            equivalent_carbon = base_carbon * throughput_scale
            avoided_cost = equivalent_cost - policy_cost
            avoided_carbon = equivalent_carbon - policy_carbon
        return {
            "schema": "port-dt-v3-online-open-loop-business-projection.v1",
            "decision_interval_min": 60,
            "baseline": {
                "throughput_teu": base_served,
                "delay_index_mean": base_delay_mean,
                "energy_cost": base_cost,
                "carbon_kg": base_carbon,
            },
            "policy": {
                "throughput_teu": policy_served,
                "delay_index_mean": policy_delay_mean,
                "energy_cost": policy_cost,
                "carbon_kg": policy_carbon,
            },
            "improvement_percent": {
                "throughput_teu": improve(policy_served, base_served),
                "delay_index_mean": improve(policy_delay_mean, base_delay_mean, lower_better=True),
                "cost_per_teu": improve(policy_cost / max(policy_served, 1e-9), base_cost / max(base_served, 1e-9), lower_better=True),
                "carbon_per_teu": improve(policy_carbon / max(policy_served, 1e-9), base_carbon / max(base_served, 1e-9), lower_better=True),
            },
            "equivalent_throughput_value": {
                "comparison_basis": "baseline_scaled_to_policy_throughput",
                "counterfactual_energy_cost": equivalent_cost,
                "policy_energy_cost": policy_cost,
                "avoided_energy_cost": avoided_cost,
                "counterfactual_carbon_kg": equivalent_carbon,
                "policy_carbon_kg": policy_carbon,
                "avoided_carbon_kg": avoided_carbon,
                "financial_audit_ready": False,
                "site_tariff_contract": "pending_port_connection",
            },
            "mean_service_factor": sum(service_factors) / len(service_factors) if service_factors else 1.0,
            "claim_boundary": "Open-loop projection over the current calibrated replay window using the environment service equations; formal multi-seed blind-test evidence remains the release KPI evidence.",
        }

    def series(self, *, horizon_min: int = 360, step_min: int = 1, scenario: str = "strategy") -> dict[str, Any]:
        self._ensure_loaded()
        if self._policy is None:
            return {"available": False, "reason": self._load_error or "runtime policy unavailable", "series": {"p50": [], "p10": [], "p90": []}, "assets": {}}
        state_series_fn = getattr(self.di.telemetry, "port_state_series", None)
        if not callable(state_series_fn):
            return {"available": False, "reason": "telemetry adapter does not expose canonical port states", "series": {"p50": [], "p10": [], "p90": []}, "assets": {}}
        cache_key = f"{horizon_min}:{step_min}:{scenario}:{int(time.time() // 5)}"
        if self._cache["key"] == cache_key and self._cache["value"] is not None:
            return self._cache["value"]
        baseline, asset_forecasts = self._baseline_forecasts(horizon_min, step_min)
        if not baseline:
            return {"available": False, "reason": "forecast model returned no baseline", "series": {"p50": [], "p10": [], "p90": []}, "assets": {}}
        states = state_series_fn(horizon_min, step_min)
        count = min(len(states), len(baseline))
        if scenario in {"baseline", "forecast", "forecast_baseline"}:
            output = {
                "available": True,
                "mode": "forecast_baseline",
                "series": {
                    key: [{"ts": row["ts"], "kW": round(float(row[key]), 6)} for row in baseline[:count]]
                    for key in ("p50", "p10", "p90")
                },
                "assets": {
                    asset_id: [{"ts": row.get("ts"), "kW": float(row.get("p50", row.get("kW", 0.0)) or 0.0)} for row in rows[:count]]
                    for asset_id, rows in asset_forecasts.items()
                },
                "_source": "ridge_autoregression_baseline",
                "production_authority": False,
            }
            return output

        asset_defs = {str(row["id"]): row for row in self.di.telemetry.list_assets() or []}
        assets: dict[str, list[dict[str, Any]]] = {asset_id: [] for asset_id in asset_forecasts}
        p50: list[dict[str, Any]] = []
        p10: list[dict[str, Any]] = []
        p90: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        decision_states: list[tuple[dict[str, Any], dict[str, Any]]] = []
        soc, last_bess_kw = 0.55, 0.0
        control: dict[str, Any] | None = None
        safety: dict[str, Any] = {}
        raw_action: list[float] = []
        action_interval = max(1, int(round(60 / max(1, step_min))))
        for index in range(count):
            state = self._apply_scenario(dict(states[index]), scenario)
            base = baseline[index]
            state["base_load_kw"] = float(base["p50"])
            state["soc"], state["last_bess_kw"] = soc, last_bess_kw
            if control is None or index % action_interval == 0:
                control, safety, raw_action = self._policy_control(
                    state, soc=soc, last_bess_kw=last_bess_kw
                )
                soc = float(control["projected_soc"])
                last_bess_kw = float(control["bess_kw"])
                actions.append(
                    {
                        "ts": base["ts"],
                        "source_ts": state.get("source_timestamp"),
                        "raw_action": raw_action,
                        "decoded_control": dict(control),
                        "safety": safety,
                    }
                )
                decision_states.append((dict(state), dict(control)))
            assert control is not None
            service_factor = float(control["service_factor"])
            berth = float(control.get("berth_priority", 0.0))
            yard = float(control.get("yard_flow_command", 0.0))
            flex_kw = float(control["flexible_load_command"]) * min(250.0, 0.08 * max(float(base["p50"]), 1.0))
            service_load = float(base["p50"]) * float(self._config["port_profile"]["assets"]["operational_load_fraction"]) * (service_factor - 1.0)
            limits = self._config["port_profile"]["control_limits"]
            berth_ratio = berth / float(limits["berth_priority_limit"])
            yard_ratio = yard / float(limits["yard_flow_limit"])
            allocation_load = float(base["p50"]) * float(self._config["port_profile"]["assets"]["allocation_load_fraction"]) * 0.5 * (berth_ratio + yard_ratio)
            target = max(0.0, float(base["p50"]) + service_load + allocation_load + float(control["bess_kw"]) + flex_kw)
            baseline_assets = {
                asset_id: float(rows[index].get("p50", rows[index].get("kW", 0.0)) or 0.0)
                for asset_id, rows in asset_forecasts.items()
            }
            provisional: dict[str, float] = {}
            for asset_id, value in baseline_assets.items():
                category = str((asset_defs.get(asset_id) or {}).get("category") or "")
                if asset_id == "bess-01":
                    continue
                multiplier = 1.0
                if category in {"岸桥", "场桥", "岸电"}:
                    multiplier *= max(0.5, service_factor)
                if category == "岸桥":
                    multiplier *= max(0.7, 1.0 + 0.10 * berth_ratio)
                if category == "场桥":
                    multiplier *= max(0.7, 1.0 + 0.10 * yard_ratio)
                provisional[asset_id] = max(0.0, value * multiplier)
            flexible_ids = [asset_id for asset_id in provisional if str((asset_defs.get(asset_id) or {}).get("category")) in {"AGV", "冷站", "冷藏箱"}]
            if flexible_ids:
                each = flex_kw / len(flexible_ids)
                for asset_id in flexible_ids:
                    provisional[asset_id] = max(0.0, provisional[asset_id] + each)
            bess_value = baseline_assets.get("bess-01", 0.0) + float(control["bess_kw"])
            non_bess_target = max(0.0, target - bess_value)
            scale = non_bess_target / max(1e-9, sum(provisional.values()))
            for asset_id in assets:
                value = bess_value if asset_id == "bess-01" else provisional.get(asset_id, 0.0) * scale
                assets[asset_id].append({"ts": base["ts"], "kW": round(value, 6)})
            delta = target - float(base["p50"])
            p50.append({"ts": base["ts"], "kW": round(target, 6)})
            p10.append({"ts": base["ts"], "kW": round(max(0.0, float(base["p10"]) + delta), 6)})
            p90.append({"ts": base["ts"], "kW": round(max(0.0, float(base["p90"]) + delta), 6)})
        business_projection = self._business_projection(
            decision_states,
            baseline[:count],
            p50,
            states[:count],
            step_min,
        )
        output = {
            "available": True,
            "mode": "selected_policy_strategy",
            "scenario": scenario,
            "series": {"p50": p50, "p10": p10, "p90": p90},
            "assets": assets,
            "actions": actions,
            "summary": {
                "decision_count": len(actions),
                "peak_kw": max((row["kW"] for row in p50), default=0.0),
                "energy_kwh": sum(row["kW"] * max(1, step_min) / 60.0 for row in p50),
                "terminal_soc": soc,
                "business_projection": business_projection,
                "hard_guardrail_passed": all(
                    bool((row.get("safety") or {}).get("within_software_envelope"))
                    for row in actions
                ),
            },
            "policy": self._metadata,
            "telemetry": self.status()["telemetry"],
            "_source": "hash_verified_sac_policy_over_ridge_forecast_and_calibrated_public_replay",
            "production_authority": False,
        }
        self._cache = {"key": cache_key, "at": time.time(), "value": output}
        return output

    def current_frame(self) -> dict[str, Any]:
        payload = self.series(horizon_min=60, step_min=60, scenario="strategy")
        if not payload.get("available"):
            return payload
        current_state_fn = getattr(self.di.telemetry, "current_port_state", None)
        current_state = current_state_fn() if callable(current_state_fn) else {}
        public_condition_keys = (
            "timestamp",
            "source_timestamp",
            "ambient_c",
            "wind_speed_mps",
            "wave_height_m",
            "current_speed_mps",
            "tide_m",
            "visibility_km",
            "closure_flag",
        )
        return {
            "available": True,
            "generated_at": payload["series"]["p50"][0]["ts"],
            "aggregate_kw": payload["series"]["p50"][0]["kW"],
            "assets": {asset_id: rows[0] for asset_id, rows in payload["assets"].items() if rows},
            "public_conditions": {
                key: current_state.get(key)
                for key in public_condition_keys
                if key in current_state
            },
            "decision": payload["actions"][0] if payload.get("actions") else None,
            "policy": payload["policy"],
            "telemetry": payload["telemetry"],
            "production_authority": False,
        }

    def bess_parameters(self) -> dict[str, float]:
        self._ensure_loaded()
        if not self._config:
            return {}
        assets = self._config["port_profile"]["assets"]
        limits = self._config["port_profile"]["control_limits"]
        return {
            "rating_kw": float(assets["bess_power_kw"]),
            "energy_mwh": float(assets["bess_capacity_kwh"]) / 1000.0,
            "soc_init_pct": 55.0,
            "soc_min_pct": float(limits["soc_min"]) * 100.0,
            "soc_max_pct": float(limits["soc_max"]) * 100.0,
        }

    def bess_capability(self, *, asset_id: str, horizon_min: int, step_min: int) -> List[Dict[str, Any]]:
        payload = self.series(horizon_min=horizon_min, step_min=step_min, scenario="strategy")
        if not payload.get("available"):
            return []
        actions = payload.get("actions") or []
        rows: list[dict[str, Any]] = []
        capacity = float(self._config["port_profile"]["assets"]["bess_capacity_kwh"])
        power = float(self._config["port_profile"]["assets"]["bess_power_kw"])
        soc_min = float(self._config["port_profile"]["control_limits"]["soc_min"])
        soc_max = float(self._config["port_profile"]["control_limits"]["soc_max"])
        for action in actions:
            control = action.get("decoded_control") or {}
            soc = float(control.get("projected_soc") or 0.55)
            rows.append(
                {
                    "ts": action.get("ts"),
                    "soc_pct": soc * 100.0,
                    "charge_cap_kw": min(power, max(0.0, (soc_max - soc) * capacity)),
                    "discharge_cap_kw": min(power, max(0.0, (soc - soc_min) * capacity)),
                }
            )
        return rows
