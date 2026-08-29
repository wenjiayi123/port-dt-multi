from __future__ import annotations

import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.services.rl_model.hvac_cooling.api import (
    DEFAULT_RESIDUAL_DELTA,
    ResidualPolicy,
    SafetyShield,
    dew_point_C,
)
from app.services.rl_model.hvac_cooling.v3_environment import (
    HVACV3Env,
    NumpyMLPPolicy,
    chronological_slices,
    load_config as load_v3_config,
    load_dataset as load_v3_dataset,
)
from app.services.training_process_evidence import (
    checkpoint_reward_replay_path,
    load_checkpoint_reward_replay,
    load_seed_process_evidence,
    seed_metric_paths,
)
from app.services.value_improvement import evidence_path, load_module_value_improvement


class HVACEvidenceService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent / "rl_model" / "hvac_cooling"
        self.data = self.root / "data"
        self.artifacts = self.root / "artifacts"
        self.repo_root = self.root.parents[3]
        self.v3_evidence = self.repo_root / "evidence" / "v3" / "hvac"

    @staticmethod
    def _sha(path: Path) -> str | None:
        if not path.exists():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _rows(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def _jsonl(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _probe(self, telemetry: Dict[str, str], policy_sha: str | None) -> Dict[str, Any]:
        state = json.loads((self.artifacts / "hvac_cooling_state.json").read_text(encoding="utf-8"))
        plant = json.loads((self.data / "plant_master.json").read_text(encoding="utf-8"))
        demand = json.loads((self.data / "demand_window_config.json").read_text(encoding="utf-8"))
        ref = {
            "CHWS_set": self._num(telemetry.get("chws_sp_C"), 7.5),
            "SAT_set": self._num(telemetry.get("avg_sat_C"), 14.0),
            "SP_set": self._num((state.get("last_targets") or {}).get("SP_set"), 800.0),
        }
        policy = ResidualPolicy(DEFAULT_RESIDUAL_DELTA)
        context = {
            "ref": ref,
            "price": self._num(telemetry.get("price_yuan_per_kwh"), 0.8),
            "ef": self._num(telemetry.get("ef_kg_per_kwh"), 0.7),
            "db_C": self._num(telemetry.get("ambient_temp_C"), 30.0),
            "rh_pct": self._num(telemetry.get("ambient_rh_pct"), 70.0),
            "dr_mode": False,
            "demand_tight": False,
            "state_features": telemetry,
        }
        residual = policy.decide(context)
        proposed = {
            "CHWS_set": ref["CHWS_set"] + residual["dCHWS"],
            "SAT_set": ref["SAT_set"] + residual["dSAT"],
            "SP_set": ref["SP_set"] + residual["dSP"],
        }
        dew_point = dew_point_C(context["db_C"], context["rh_pct"])
        final, reasons = SafetyShield(plant, demand).apply(
            ref,
            proposed,
            {"dew_point_C": dew_point, "CHW_flow": 0.0, "G_min": 0.3, "demand_tight": False},
        )
        raw_action = self._num(policy.last_audit.get("raw_action"))
        plant_power = self._num(telemetry.get("plant_power_kw"))
        projected_saving_kw = max(0.0, raw_action) * plant_power * 0.12
        return {
            "input": {name: self._num(value) if name != "timestamp" else value for name, value in telemetry.items()},
            "input_source": "checked_in_hvac_engineering_replay",
            "measured": False,
            "policy": {
                "algorithm": policy.backend,
                "policy_sha256": policy_sha,
                **policy.last_audit,
            },
            "reference": ref,
            "residual": {key: round(value, 6) for key, value in residual.items()},
            "proposed": {key: round(value, 4) for key, value in proposed.items()},
            "safety": {
                "dew_point_C": round(dew_point, 3),
                "masks": reasons,
                "final_action": final,
                "hard_limits": plant.get("setpoints") or {},
            },
            "derived_step_projection": {
                "power_saving_kw": round(projected_saving_kw, 4),
                "cost_saving_yuan_per_15min": round(projected_saving_kw * context["price"] * 0.25, 4),
                "carbon_saving_kg_per_15min": round(projected_saving_kw * context["ef"] * 0.25, 4),
                "claim_eligible": False,
                "reason": "Engineering sensitivity proxy from trainer; not a calibrated HVAC thermodynamic counterfactual.",
            },
            "production_authority": False,
        }

    @staticmethod
    def _series(rows: List[Dict[str, Any]], limit: int = 180) -> List[Dict[str, Any]]:
        train_rows = [row for row in rows if row.get("kind") == "train_step"]
        stride = max(1, math.ceil(len(train_rows) / limit))
        chosen = train_rows[::stride]
        if train_rows and chosen[-1] is not train_rows[-1]:
            chosen.append(train_rows[-1])
        return [
            {
                "step": int(row.get("step") or 0),
                "reward": HVACEvidenceService._num(row.get("raw_step_reward")),
                "economic_proxy": HVACEvidenceService._num(row.get("raw_metric_econ_save")),
                "carbon_proxy": HVACEvidenceService._num(row.get("raw_metric_carbon_save")),
                "comfort_proxy": HVACEvidenceService._num(row.get("raw_metric_comfort_score")),
                "actor_loss": HVACEvidenceService._num(row.get("actor")),
                "q1_loss": HVACEvidenceService._num(row.get("q1")),
                "q2_loss": HVACEvidenceService._num(row.get("q2")),
            }
            for row in chosen
        ]

    def build(self) -> Dict[str, Any]:
        formal_paths = [self.v3_evidence / "latest.json"]
        try:
            latest, formal = self._load_formal()
            if formal:
                formal_paths.append(self.repo_root / str(latest.get("report_path") or ""))
                for item in (formal.get("artifacts") or {}).get("models") or []:
                    formal_paths.append(self.repo_root / str(item.get("path") or ""))
                formal_paths.extend(seed_metric_paths(self.repo_root, latest))
                formal_paths.append(checkpoint_reward_replay_path(self.repo_root, latest))
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            pass
        tracked = [
            self.root / "policy.bin",
            self.artifacts / "policy_evaluate_history.jsonl",
            self.data / "hvac_telemetry.csv",
            self.data / "load_forecast.csv",
            self.data / "plant_master.json",
            *formal_paths,
            evidence_path(self.repo_root),
        ]
        key = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in tracked if path.exists())
        return self._build_cached(key)

    def _load_formal(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        latest_path = self.v3_evidence / "latest.json"
        if not latest_path.exists():
            return {}, {}
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        report_path = self.repo_root / str(latest.get("report_path") or "")
        if not report_path.exists() or self._sha(report_path) != latest.get("report_sha256"):
            raise RuntimeError("HVAC formal evidence pointer or report hash is invalid")
        return latest, json.loads(report_path.read_text(encoding="utf-8"))

    def _current_v3_inference(self, report: Dict[str, Any]) -> Dict[str, Any]:
        models = (report.get("artifacts") or {}).get("models") or []
        fallback = (report.get("blind_test") or {}).get("sample_real_model_inference") or {}
        if not models:
            return {"policy_loaded": False, "error": "formal HVAC model artifact is missing", **fallback}
        selected = models[0]
        model_path = self.repo_root / str(selected.get("path") or "")
        if not model_path.exists() or self._sha(model_path) != selected.get("sha256"):
            return {"policy_loaded": False, "error": "formal HVAC model hash gate failed", **fallback}
        try:
            config = load_v3_config()
            dataset = load_v3_dataset(config)
            train_slice, _validation_slice, blind_slice = chronological_slices(dataset)
            env = HVACV3Env(
                dataset, blind_slice, config=config, normalization_slice=train_slice,
                episode_steps=int(config["training"]["episode_steps"]),
                seed=int(selected.get("seed") or 53), training=False, record_trace=False,
            )
            observation, reset_info = env.reset(options={"start_index": 0})
            action = NumpyMLPPolicy.load(model_path).predict(observation)
            _next, _reward, _terminated, _truncated, info = env.step(action)
            env.close()
            return {
                "policy_loaded": True,
                "policy_admitted_for_engineering_replay": bool((report.get("quality_gates") or {}).get("public_offline_admitted")),
                "production_admitted": False,
                "decision_source": "hash_verified_selected_safe_actor_plus_hvac_projection",
                "algorithm": (report.get("training") or {}).get("algorithm"),
                "seed": selected.get("seed"),
                "model_path": selected.get("path"),
                "model_sha256": selected.get("sha256"),
                "timestamp": info.get("timestamp"),
                "reset": reset_info,
                "observation_vector": [round(float(value), 6) for value in observation],
                "state": info.get("context"),
                "requested_action": info.get("requested_action"),
                "final_action": info.get("final_action"),
                "projection": info.get("projection"),
                "derived_business_step": info.get("business_step"),
                "guardrail_violation": info.get("guardrail_violation"),
            }
        except Exception as exc:
            return {"policy_loaded": False, "error": str(exc), "saved_blind_inference": fallback}

    @lru_cache(maxsize=4)
    def _build_cached(self, _key: Tuple[Tuple[str, int, int], ...]) -> Dict[str, Any]:
        latest, formal = self._load_formal()
        history_path = self.artifacts / "policy_evaluate_history.jsonl"
        history = self._jsonl(history_path)
        telemetry = self._rows(self.data / "hvac_telemetry.csv")
        forecast = self._rows(self.data / "load_forecast.csv")
        eval_rows = [row for row in history if row.get("kind") == "train_eval"]
        train_starts = [row for row in history if row.get("kind") == "train_start"]
        last_eval = eval_rows[-1] if eval_rows else {}
        start = train_starts[-1] if train_starts else {}
        forecast_day = forecast[-96:] if len(forecast) >= 96 else forecast
        loads = [self._num(row.get("q_cooling_forecast_kw_p50")) for row in forecast_day]
        policy_sha = self._sha(self.root / "policy.bin")
        probe = self._probe(telemetry[-1] if telemetry else {}, policy_sha)
        manifests = []
        for name in (
            "hvac_telemetry.csv", "load_forecast.csv", "weather_forecast.csv", "market_price.csv",
            "grid_ef.csv", "plant_efficiency_map.csv", "plant_master.json", "demand_window_config.json",
        ):
            path = self.data / name
            rows = self._rows(path) if path.suffix == ".csv" else []
            manifests.append({"file": name, "rows": len(rows) if rows else None, "sha256": self._sha(path)})
        quality = formal.get("quality_gates") or {}
        business = formal.get("business_metrics") or {}
        convergence = formal.get("convergence") or {}
        inference = self._current_v3_inference(formal) if formal else {
            "policy_loaded": False,
            "error": "formal HVAC V3.1 evidence is unavailable",
        }
        final_action = inference.get("final_action") or {}
        return {
            "version": "V3.1",
            "value_improvement": load_module_value_improvement(self.repo_root, "hvac"),
            "module": {"id": "hvac_cooling", "name": "HVAC制冷", "state": "formal_engineering_offline_site_pending"},
            "boundary": {
                "evidence_tier": (formal.get("dataset") or {}).get("evidence_tier") or "checked_in_engineering_emulator_replay",
                "claim_eligible": False,
                "live_data_verified": False,
                "production_authority": False,
                "site_status": "待接入港口",
                "reason": "V3.1已完成工程时序多种子训练、固定验证、独立盲测和真实权重推理；BAS点位、冷机性能图、流量/阀位、舒适度与结算计量仍未获得现场授权，收益不是上海港实测。",
            },
            "current_model_output": {
                "model_inference": inference,
                "source": "runtime_reload_of_hash_verified_selected_model",
                "not_static_card": True,
            },
            "historical_evidence": {
                "preserved": True,
                "records": len(history),
                "history_sha256": self._sha(history_path),
                "algorithm": (start.get("config") or {}).get("algo") or "sac",
                "seed": (start.get("config") or {}).get("seed"),
                "steps": (start.get("config") or {}).get("steps"),
                "action_mode": (start.get("config") or {}).get("action_mode"),
                "eval_chws_mse": self._num(last_eval.get("chws_mse")),
                "eval_policy_l2_mean": self._num(last_eval.get("policy_l2_mean")),
                "policy_sha256": policy_sha,
                "note": "4003条旧日志与原SAC权重原样保留；旧训练中的人工bias/noise展示字段不用于V3.1业务证据。",
            },
            "replay_kpis": {
                "model_chws_C": final_action.get("chws_c"),
                "model_sat_C": final_action.get("sat_c"),
                "model_sp_Pa": final_action.get("static_pressure_pa"),
                "forecast_24h_peak_kw": max(loads) if loads else None,
                "forecast_24h_energy_kwh": sum(loads) * 0.25 if loads else None,
                "eval_chws_mse": self._num(last_eval.get("chws_mse")),
                "claim_eligible": False,
            },
            "formal_training": {
                "pointer": latest,
                "status": formal.get("status"),
                "dataset": formal.get("dataset"),
                "training": formal.get("training"),
                "contract": formal.get("contract"),
                "counterfactual_model": formal.get("counterfactual_model"),
                "convergence": convergence,
                "blind_test_protocol": {
                    "windows": (formal.get("blind_test") or {}).get("windows"),
                    "window_hours": (formal.get("blind_test") or {}).get("window_hours"),
                    "selection_access": (formal.get("blind_test") or {}).get("selection_access"),
                },
            },
            "business_metrics": business,
            "quality_gates": {
                **quality,
                "policy_artifact_loads": bool(inference.get("policy_loaded")),
                "admitted": bool(quality.get("public_offline_admitted")),
                "production_admitted": False,
                "reasons": [
                    "authorized_bas_meter_flow_valve_comfort_and_gateway_not_connected",
                    "site_performance_map_calibration_shadow_ab_and_operator_acceptance_pending",
                ],
            },
            "legacy_model_output": probe,
            "algorithm_registry": [{
                "name": row.get("name"),
                "state": row.get("state"),
                "artifact": "formal V3.1 evidence" if row.get("state") != "historical_preserved" else "policy.bin",
                "admission": row.get("reason"),
            } for row in (formal.get("algorithm_registry") or [])],
            "data_manifest": {
                "mode": "checked_in_engineering_emulator_chronological_replay",
                "telemetry_rows": len(telemetry),
                "time_range": {
                    "start": telemetry[0].get("timestamp") if telemetry else None,
                    "end": telemetry[-1].get("timestamp") if telemetry else None,
                },
                "files": manifests,
                "measured": False,
                "formal_dataset": formal.get("dataset"),
            },
            "site_contract": {
                "required_inputs": ["BAS点表与回执", "冷冻水流量/供回水温", "冷机/水泵/冷却塔功率", "区域舒适度/阀位", "室外干湿球", "冷负荷预测", "电价/排放因子", "需量窗口"],
                "outputs": ["CHWS设定", "SAT设定", "风机静压", "遮罩原因", "TTL写点任务"],
                "hard_constraints": (formal.get("contract") or {}).get("hard_constraints"),
                "acceptance": ["时间盲测冷量满足率", "舒适度/湿度不劣化", "kWh/RT与COP", "最大需量", "电费/碳", "多种子收敛", "遮罩/回退/回读成功率"],
                "replacement": "保留30维状态、3维动作、策略格式与安全投影；替换BAS适配器、现场性能图和灵敏度标定后重训。",
            },
            "history_series": convergence.get("aggregate_curve") or [],
            "training_process": load_seed_process_evidence(self.repo_root, latest),
            "checkpoint_reward_replay": load_checkpoint_reward_replay(self.repo_root, latest),
            "legacy_history_series": self._series(history),
        }
