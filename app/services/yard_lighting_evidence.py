from __future__ import annotations

import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.services.rl_model.yard_lighting.v3_environment import (
    NumpyMLPPolicy, YardLightingV3Env, chronological_slices,
    load_config as load_v3_config, load_dataset as load_v3_dataset,
)
from app.services.training_process_evidence import (
    checkpoint_reward_replay_path,
    load_checkpoint_reward_replay,
    load_seed_process_evidence,
    seed_metric_paths,
)
from app.services.value_improvement import evidence_path, load_module_value_improvement


class YardLightingEvidenceService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent / "rl_model" / "yard_lighting"
        self.artifacts, self.data = self.root / "artifacts", self.root / "data"
        self.repo_root = self.root.parents[3]
        self.v3_evidence = self.repo_root / "evidence" / "v3" / "yard_lighting"

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
    def _jsonl(path: Path) -> List[Dict[str, Any]]:
        rows = []
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    @staticmethod
    def _count(path: Path) -> int:
        with path.open("rb") as stream:
            return max(0, sum(1 for _ in stream) - 1)

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _legacy_series(rows: Iterable[Dict[str, Any]], limit: int = 160) -> List[Dict[str, Any]]:
        source = list(rows)
        stride = max(1, math.ceil(len(source) / limit))
        selected = source[::stride]
        if source and selected[-1] is not source[-1]:
            selected.append(source[-1])
        output = []
        for row in selected:
            savings = ((row.get("economics") or {}).get("savings") or {})
            metrics, rewards = row.get("metrics") or {}, row.get("rewards") or {}
            output.append({
                "step": int(row.get("step") or len(output)),
                "reward_gain_yuan": YardLightingEvidenceService._float(rewards.get("gain")),
                "saving_kwh": YardLightingEvidenceService._float(savings.get("kWh"), YardLightingEvidenceService._float(metrics.get("delta_kWh"))),
                "peak_reduction_kw": YardLightingEvidenceService._float(savings.get("peak_kW"), YardLightingEvidenceService._float(metrics.get("peak_reduction_kW"))),
                "under_lux_score": YardLightingEvidenceService._float(metrics.get("under_lux")),
                "v_loss": YardLightingEvidenceService._float(row.get("v_loss")),
                "q_loss": YardLightingEvidenceService._float(row.get("q_loss")),
                "pi_loss": YardLightingEvidenceService._float(row.get("pi_loss")),
            })
        return output

    def _legacy_probe(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        probe = {
            "timestamp": "2025-10-11T11:50:00Z", "zone_id": "ZN_001",
            "price_yuan_per_kwh": 1.08, "grid_ef_kg_per_kwh": 0.74, "activity_score": 0.20,
            "lux": 24.2, "lux_min": 20.0, "critical": False, "complaint_zone": False,
            "previous_dimming_ratio": 0.20, "dwell_steps": 1, "power_kw": 3.6,
        }
        result = {"input": probe, "measured": False, "production_authority": False, "policy_loaded": False}
        try:
            import numpy as np
            import torch
            from app.services.rl_model.yard_lighting.rl_engine import Policy, SafetyLayer, load_limits

            hour = 11 + 50 / 60
            raw = np.asarray([
                probe["price_yuan_per_kwh"], probe["grid_ef_kg_per_kwh"], probe["activity_score"],
                probe["lux"], probe["lux_min"], 0.0, 0.0, probe["previous_dimming_ratio"],
                math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24), probe["dwell_steps"] / 24,
            ], dtype=np.float32)
            mean, std = np.asarray(meta.get("obs_mean") or []), np.asarray(meta.get("obs_std") or [])
            normalized = (raw - mean) / np.maximum(std, 1e-6)
            policy = Policy(len(raw)); weights = torch.load(self.root / "policy.bin", map_location="cpu", weights_only=True)
            policy.load_state_dict(weights["pi"]); policy.eval()
            with torch.inference_mode():
                raw_action = float(policy(torch.tensor(normalized, dtype=torch.float32)[None, :]).item())
            projected, applied = SafetyLayer(load_limits()).project(probe["previous_dimming_ratio"], raw_action, 0)
            absolute = float(np.max(np.abs(normalized))); threshold = 8.0
            admitted = absolute <= threshold
            result.update({
                "policy_loaded": True, "algorithm": meta.get("algo") or "IQL",
                "raw_action_dimming_ratio": round(raw_action, 6), "safety_projected_action_ratio": round(float(projected), 6),
                "safety_projection_applied": bool(applied), "normalized_abs_max": round(absolute, 3),
                "ood_threshold": threshold, "policy_admitted": admitted,
                "decision_source": "trained_policy" if admitted else "rule_fallback",
                "fail_closed_reason": None if admitted else "V2 boolean encoding drift; V3.1 keeps the old policy OOD-blocked",
            })
        except Exception as exc:
            result.update({"policy_admitted": False, "decision_source": "rule_fallback", "fail_closed_reason": f"legacy probe unavailable: {exc}"})
        return result

    def _formal(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pointer = self.v3_evidence / "latest.json"
        if not pointer.exists():
            return {}, {}
        latest = json.loads(pointer.read_text(encoding="utf-8"))
        report_path = self.repo_root / str(latest.get("report_path") or "")
        if not report_path.exists() or self._sha(report_path) != latest.get("report_sha256"):
            raise RuntimeError("yard-lighting formal report hash gate failed")
        return latest, json.loads(report_path.read_text(encoding="utf-8"))

    def _inference(self, report: Dict[str, Any]) -> Dict[str, Any]:
        models = (report.get("artifacts") or {}).get("models") or []
        fallback = (report.get("blind_test") or {}).get("sample_real_model_inference") or {}
        if not models:
            return {"policy_loaded": False, "error": "formal lighting model missing", "saved_blind_inference": fallback}
        selected = models[0]
        path = self.repo_root / str(selected.get("path") or "")
        if not path.exists() or self._sha(path) != selected.get("sha256"):
            return {"policy_loaded": False, "error": "formal lighting model hash gate failed", "saved_blind_inference": fallback}
        try:
            config = load_v3_config(); dataset = load_v3_dataset(config)
            train_slice, _validation_slice, blind_slice = chronological_slices(dataset)
            env = YardLightingV3Env(dataset, blind_slice, config=config, normalization_slice=train_slice,
                                    episode_steps=config["training"]["episode_steps"], seed=int(selected.get("seed") or 67))
            observation, reset = env.reset(options={"start_index": 0})
            action = NumpyMLPPolicy.load(path).predict(observation)
            _next, _reward, _terminated, _truncated, info = env.step(action); env.close()
            return {
                "policy_loaded": True,
                "policy_admitted_for_engineering_replay": bool((report.get("quality_gates") or {}).get("public_offline_admitted")),
                "production_admitted": False,
                "decision_source": "hash_verified_selected_lux_safe_actor_plus_zone_projection",
                "algorithm": (report.get("training") or {}).get("algorithm"), "seed": selected.get("seed"),
                "model_path": selected.get("path"), "model_sha256": selected.get("sha256"),
                "timestamp": info.get("timestamp"), "reset": reset,
                "observation_vector": [round(float(value), 6) for value in observation],
                "state": info.get("context"), "requested_action": info.get("requested_action"),
                "final_action": info.get("final_action"), "projection": info.get("projection"),
                "derived_business_step": info.get("business_step"), "guardrail_violation": info.get("guardrail_violation"),
            }
        except Exception as exc:
            return {"policy_loaded": False, "error": str(exc), "saved_blind_inference": fallback}

    def build(self) -> Dict[str, Any]:
        formal_paths = [self.v3_evidence / "latest.json", self.v3_evidence / "history_index.jsonl"]
        try:
            latest, report = self._formal()
            if report:
                formal_paths.append(self.repo_root / str(latest.get("report_path") or ""))
                formal_paths.extend(self.repo_root / str(row.get("path") or "") for row in (report.get("artifacts") or {}).get("models") or [])
                formal_paths.extend(seed_metric_paths(self.repo_root, latest))
                formal_paths.append(checkpoint_reward_replay_path(self.repo_root, latest))
        except Exception:
            pass
        tracked = [
            self.artifacts / "offline_train.jsonl", self.root / "policy.bin", self.root / "policy_meta.json",
            self.data / "zones_master.csv", self.data / "lighting_telemetry.csv", self.data / "activity_forecast.csv",
            self.repo_root / "data/rl/datasets/public_cn_sha_hourly_v3.csv", *formal_paths,
            evidence_path(self.repo_root),
        ]
        key = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in tracked if path.exists())
        return self._build_cached(key)

    @lru_cache(maxsize=4)
    def _build_cached(self, _key: Tuple[Tuple[str, int, int], ...]) -> Dict[str, Any]:
        latest, report = self._formal()
        history_path = self.artifacts / "offline_train.jsonl"
        history = self._jsonl(history_path); last = history[-1] if history else {}
        economics = last.get("economics") or {}; baseline = economics.get("baseline") or {}; policy = economics.get("policy") or {}
        savings, metrics = economics.get("savings") or {}, last.get("metrics") or {}
        meta = json.loads((self.root / "policy_meta.json").read_text(encoding="utf-8"))
        legacy_probe = self._legacy_probe(meta)
        inference = self._inference(report)
        convergence, quality, business = report.get("convergence") or {}, report.get("quality_gates") or {}, report.get("business_metrics") or {}
        return {
            "version": "V3.1",
            "value_improvement": load_module_value_improvement(self.repo_root, "yard_lighting"),
            "module": {"id": "yard_lighting", "name": "堆场照明", "state": "formal_public_enriched_offline_site_pending"},
            "boundary": {
                "evidence_tier": (report.get("dataset") or {}).get("evidence_tier") or "public_reanalysis_enriched_engineering_emulator_replay",
                "claim_eligible": False, "live_data_verified": False, "production_authority": False, "site_status": "待接入港口",
                "reason": "V3.1联动上海/洋山公开再分析信号并完成三种子、封存盲测和真实权重推理；96分区照度/功率/灯杆/热区仍是工程模拟器，现场光度标定、故障回读、投诉与网关未接入。",
            },
            "current_model_output": {"model_inference": inference, "source": "runtime_reload_of_hash_verified_selected_model", "not_static_card": True},
            "model_probe": legacy_probe,
            "historical_evidence": {
                "preserved": True, "records": len(history), "first_step": int(history[0].get("step") or 0) if history else None,
                "last_step": int(last.get("step") or 0) if history else None, "history_sha256": self._sha(history_path),
                "policy_step": meta.get("step"), "policy_score": meta.get("score"), "policy_sha256": self._sha(self.root / "policy.bin"),
                "algorithm": meta.get("algo") or last.get("algo"),
                "note": "498条旧训练和原IQL权重完整保留；正确布尔解析后旧权重继续被OOD门禁拦截，不参与V3.1业务指标。",
            },
            "historical_business_metrics": {
                "reward_baseline_yuan": self._float((baseline.get("rewards") or {}).get("reward_total")),
                "reward_policy_yuan": self._float((policy.get("rewards") or {}).get("reward_total")),
                "reward_gain_yuan": self._float((last.get("rewards") or {}).get("gain")),
                "saving_kwh": self._float(savings.get("kWh"), self._float(metrics.get("delta_kWh"))),
                "saving_percent": self._float(savings.get("percent")), "peak_reduction_kw": self._float(savings.get("peak_kW")),
                "carbon_reduction_kg": self._float(savings.get("carbon_kg")), "under_lux_score": self._float(metrics.get("under_lux")),
                "claim_eligible": False,
            },
            "formal_training": {
                "pointer": latest, "status": report.get("status"), "dataset": report.get("dataset"), "training": report.get("training"),
                "contract": report.get("contract"), "counterfactual_model": report.get("counterfactual_model"), "convergence": convergence,
                "blind_test_protocol": {"windows": (report.get("blind_test") or {}).get("windows"), "window_hours": (report.get("blind_test") or {}).get("window_hours"), "selection_access": (report.get("blind_test") or {}).get("selection_access")},
                "run_history": self._jsonl(self.v3_evidence / "history_index.jsonl"),
            },
            "business_metrics": business,
            "quality_gates": {
                **quality, "policy_artifact_loads": bool(inference.get("policy_loaded")),
                "admitted": bool(quality.get("public_offline_admitted")), "production_admitted": False,
                "legacy_policy_ood_blocked": not bool(legacy_probe.get("policy_admitted")),
                "reasons": ["authorized_lux_power_fixture_fault_and_gateway_feedback_not_connected", "site_photometric_calibration_shadow_ab_complaint_acceptance_and_rollback_pending"],
            },
            "algorithm_registry": [{
                "name": row.get("name"), "state": row.get("state"),
                "artifact": "formal V3.1 evidence" if row.get("state") not in {"historical_preserved", "code_preserved"} else "legacy source/history",
                "admission": row.get("reason"),
            } for row in report.get("algorithm_registry") or []],
            "data_manifest": {
                "mode": "public_signal_enriched_engineering_emulator_chronological_replay", "measured": False,
                "formal_dataset": report.get("dataset"), "known_v2_issue": "CSV False/0 was previously parsed with bool(string); fixed code preserves but rejects the old policy.",
                "public_linkage": {"dataset_id": "public_cn_sha_hourly_v3", "signals": ["ambient_c", "wind_speed_mps", "yard_occupancy_ratio", "equipment_availability_ratio", "base_load_kw", "price_per_kwh", "carbon_kg_per_kwh"]},
            },
            "site_contract": {
                "required_inputs": ["区域/灯杆资产ID与拓扑", "照度与有功功率表计", "调光反馈/故障/开关状态", "作业热区与关键通道", "日落日出/能见度/天气", "分时电价与边际排放因子", "投诉敏感区/最低照度/SOP"],
                "outputs": ["基础调光残差", "作业热度增益", "天气增益", "分区安全投影", "拒绝原因", "审计回执"],
                "hard_constraints": (report.get("contract") or {}).get("hard_constraints"),
                "acceptance": ["零关键区低照度违规", "现场A/B节电", "峰值降低", "灯具故障回退", "投诉不劣化", "人工覆盖与回滚"],
                "replacement": "保留42维状态、3维动作、策略JSON与分区安全投影；替换96区工程模拟器为现场灯杆拓扑、光度曲线、照度/功率/故障回读后重训。",
            },
            "history_series": convergence.get("aggregate_curve") or [],
            "training_process": load_seed_process_evidence(self.repo_root, latest),
            "checkpoint_reward_replay": load_checkpoint_reward_replay(self.repo_root, latest),
            "legacy_history_series": self._legacy_series(history),
        }
