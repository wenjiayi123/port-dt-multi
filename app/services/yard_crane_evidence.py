from __future__ import annotations

import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.services.rl_model.yard_crane.v3_environment import (
    NumpyMLPPolicy, YardCraneV3Env, chronological_slices,
    load_config as load_v3_config, load_dataset as load_v3_dataset,
)
from app.services.training_process_evidence import (
    checkpoint_reward_replay_path,
    load_checkpoint_reward_replay,
    load_seed_process_evidence,
    seed_metric_paths,
)


class YardCraneEvidenceService:
    """Hash-gated V3.1 evidence while preserving every legacy artifact."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent / "rl_model" / "yard_crane"
        self.data, self.artifacts = self.root / "data", self.root / "artifacts"
        self.repo_root = self.root.parents[3]
        self.v3_evidence = self.repo_root / "evidence" / "v3" / "yard_crane"

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
        out: List[Dict[str, Any]] = []
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    @staticmethod
    def _csv(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _legacy_series(steps: List[Dict[str, Any]], limit: int = 180) -> List[Dict[str, Any]]:
        stride = max(1, math.ceil(len(steps) / limit))
        selected = steps[::stride]
        if steps and selected[-1] is not steps[-1]:
            selected.append(steps[-1])
        return [{
            "step": index + 1,
            "reward": YardCraneEvidenceService._num(row.get("reward")),
            "economic_advantage": YardCraneEvidenceService._num(row.get("econ_advantage_yuan")),
            "sla_penalty": YardCraneEvidenceService._num((row.get("reward_breakdown") or {}).get("sla_penalty")),
            "power_kw": YardCraneEvidenceService._num(row.get("p_act_kW")),
            "mask_applied": int(row.get("mask_applied") or 0),
        } for index, row in enumerate(selected)]

    def _load_formal(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        pointer = self.v3_evidence / "latest.json"
        if not pointer.exists():
            return {}, {}
        latest = json.loads(pointer.read_text(encoding="utf-8"))
        report_path = self.repo_root / str(latest.get("report_path") or "")
        if not report_path.exists() or self._sha(report_path) != latest.get("report_sha256"):
            raise RuntimeError("yard-crane formal report hash gate failed")
        return latest, json.loads(report_path.read_text(encoding="utf-8"))

    def _current_inference(self, report: Dict[str, Any]) -> Dict[str, Any]:
        models = (report.get("artifacts") or {}).get("models") or []
        fallback = (report.get("blind_test") or {}).get("sample_real_model_inference") or {}
        if not models:
            return {"policy_loaded": False, "error": "formal yard-crane model is missing", "saved_blind_inference": fallback}
        selected = models[0]
        model_path = self.repo_root / str(selected.get("path") or "")
        if not model_path.exists() or self._sha(model_path) != selected.get("sha256"):
            return {"policy_loaded": False, "error": "formal yard-crane model hash gate failed", "saved_blind_inference": fallback}
        try:
            config = load_v3_config()
            dataset = load_v3_dataset(config)
            train_slice, _validation_slice, blind_slice = chronological_slices(dataset)
            env = YardCraneV3Env(
                dataset, blind_slice, config=config, normalization_slice=train_slice,
                episode_steps=int(config["training"]["episode_steps"]), seed=int(selected.get("seed") or 61),
                training=False, record_trace=False,
            )
            observation, reset = env.reset(options={"start_index": 0})
            action = NumpyMLPPolicy.load(model_path).predict(observation)
            _next, _reward, _terminated, _truncated, info = env.step(action)
            env.close()
            return {
                "policy_loaded": True,
                "policy_admitted_for_engineering_replay": bool((report.get("quality_gates") or {}).get("public_offline_admitted")),
                "production_admitted": False,
                "decision_source": "hash_verified_selected_safe_actor_plus_cmdp_projection",
                "algorithm": (report.get("training") or {}).get("algorithm"),
                "seed": selected.get("seed"), "model_path": selected.get("path"), "model_sha256": selected.get("sha256"),
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
            latest, formal = self._load_formal()
            if formal:
                formal_paths.append(self.repo_root / str(latest.get("report_path") or ""))
                formal_paths.extend(self.repo_root / str(row.get("path") or "") for row in (formal.get("artifacts") or {}).get("models") or [])
                formal_paths.extend(seed_metric_paths(self.repo_root, latest))
                formal_paths.append(checkpoint_reward_replay_path(self.repo_root, latest))
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            pass
        tracked = [
            self.root / "policy.bin", self.root / "policy_evaluate_history.jsonl",
            self.artifacts / "offline_dataset_crane.jsonl", self.artifacts / "offline_dataset_crane_aug.jsonl",
            self.data / "crane_telemetry.csv", self.data / "job_events.csv", self.data / "queue_forecast.csv",
            *formal_paths,
        ]
        key = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in tracked if path.exists())
        return self._build_cached(key)

    @lru_cache(maxsize=4)
    def _build_cached(self, _key: Tuple[Tuple[str, int, int], ...]) -> Dict[str, Any]:
        latest, formal = self._load_formal()
        history_path = self.root / "policy_evaluate_history.jsonl"
        base_path = self.artifacts / "offline_dataset_crane.jsonl"
        aug_path = self.artifacts / "offline_dataset_crane_aug.jsonl"
        legacy_policy = self.root / "policy.bin"
        history, base, aug = self._jsonl(history_path), self._jsonl(base_path), self._jsonl(aug_path)
        steps = [row for row in history if row.get("key") == "crane_step"]
        updates = [row for row in history if row.get("key") == "policy_update"]
        base_nonzero = sum(any(abs(self._num((row.get("action") or {}).get(key))) > 1e-9 for key in ("d_power_pct", "d_idle_min")) for row in base)
        aug_nonzero = sum(any(abs(self._num((row.get("action") or {}).get(key))) > 1e-9 for key in ("d_power_pct", "d_idle_min")) for row in aug)
        mask_rate = sum(int(row.get("mask_applied") or 0) for row in steps) / len(steps) if steps else 0.0
        econ = sum(self._num(row.get("econ_advantage_yuan")) for row in steps)
        sla = sum(self._num((row.get("reward_breakdown") or {}).get("sla_penalty")) for row in steps)
        legacy_job = sum(self._num((row.get("obs") or {}).get("boxes_15m")) > 0 for row in steps)
        legacy_thermal = sum((row.get("obs") or {}).get("tmotor") is not None and (row.get("obs") or {}).get("tinv") is not None for row in steps)
        inference = self._current_inference(formal)
        quality = formal.get("quality_gates") or {}
        convergence = formal.get("convergence") or {}
        business = formal.get("business_metrics") or {}
        run_history = self._jsonl(self.v3_evidence / "history_index.jsonl")
        manifests = []
        for name in ("cranes_master.csv", "yard_blocks.csv", "crane_telemetry.csv", "job_events.csv", "queue_forecast.csv", "grid_meter.csv", "market_price.csv", "grid_ef.csv", "dr_events.json"):
            path = self.data / name
            rows = self._csv(path) if path.suffix == ".csv" else []
            manifests.append({"file": name, "rows": len(rows) if rows else None, "sha256": self._sha(path)})
        return {
            "version": "V3.1",
            "module": {"id": "yard_crane", "name": "场桥/轨道吊节能调度", "state": "formal_engineering_offline_site_pending"},
            "boundary": {
                "evidence_tier": (formal.get("dataset") or {}).get("evidence_tier") or "checked_in_engineering_emulator_replay",
                "claim_eligible": False, "live_data_verified": False, "production_authority": False,
                "site_status": "待接入港口",
                "reason": "V3.1已完成16台吊机工程时序的三种子训练、固定验证、封存盲测和真实权重推理；PLC热传感、TOS回执、设备性能曲线、故障/维护与影子运行仍未接入，收益不是港口实测。",
            },
            "current_model_output": {"model_inference": inference, "source": "runtime_reload_of_hash_verified_selected_model", "not_static_card": True},
            "formal_training": {
                "pointer": latest, "status": formal.get("status"), "dataset": formal.get("dataset"),
                "training": formal.get("training"), "contract": formal.get("contract"),
                "counterfactual_model": formal.get("counterfactual_model"), "convergence": convergence,
                "blind_test_protocol": {
                    "windows": (formal.get("blind_test") or {}).get("windows"),
                    "window_hours": (formal.get("blind_test") or {}).get("window_hours"),
                    "selection_access": (formal.get("blind_test") or {}).get("selection_access"),
                },
                "run_history": run_history,
            },
            "business_metrics": business,
            "quality_gates": {
                **quality, "policy_artifact_loads": bool(inference.get("policy_loaded")),
                "admitted": bool(quality.get("public_offline_admitted")), "production_admitted": False,
                "legacy_audit": {
                    "policy_artifact_bytes": legacy_policy.stat().st_size if legacy_policy.exists() else 0,
                    "base_offline_rows": len(base), "base_nonzero_action_rows": base_nonzero,
                    "augmented_rows": len(aug), "augmented_nonzero_action_rows": aug_nonzero,
                    "historical_job_positive_rows": legacy_job, "historical_thermal_available_rows": legacy_thermal,
                    "historical_mask_rate": round(mask_rate, 6), "heldout_evaluation_rows": 0,
                },
                "reasons": [
                    "authorized_tos_plc_thermal_fault_and_gateway_feedback_not_connected",
                    "site_power_curve_calibration_shadow_ab_operator_acceptance_and_rollback_pending",
                ],
            },
            "historical_evidence": {
                "preserved": True, "records": len(history), "step_records": len(steps),
                "history_sha256": self._sha(history_path), "offline_rows": len(base), "offline_sha256": self._sha(base_path),
                "augmented_rows": len(aug), "augmented_sha256": self._sha(aug_path), "policy_sha256": self._sha(legacy_policy),
                "note": "旧1000步、1条策略更新、17278条基线、144条增强数据和0字节旧策略均原样保留；失败的首次V3.1盲测也保留在run_history。",
            },
            "historical_training_diagnostics": {
                "economic_advantage_yuan": round(econ, 6), "sla_penalty_yuan": round(sla, 6),
                "mask_rate": round(mask_rate, 6), "last_policy_update": updates[-1] if updates else {},
                "claim_eligible": False,
                "conclusion": "旧训练的电费优势被高额SLA罚金、空作业映射、缺失温度和0字节策略否决，只作为历史审计。",
            },
            "algorithm_registry": [{
                "name": row.get("name"), "state": row.get("state"),
                "artifact": "formal V3.1 evidence" if row.get("state") not in {"code_preserved", "history_preserved"} else "legacy source/history",
                "admission": row.get("reason"),
            } for row in formal.get("algorithm_registry") or []],
            "data_manifest": {"mode": "checked_in_engineering_emulator_chronological_replay", "measured": False, "files": manifests, "formal_dataset": formal.get("dataset")},
            "schema_repairs": {"implemented": True, "fields": ["yard_block", "crane_type", "moves_planned", "moves_min_accept", "end_time_utc", "arrivals_p50_per_step"], "effect": "V3.1聚合16台吊机、8559条TOS作业和12个箱区预测；温度明确标为工程代理。"},
            "site_contract": {
                "required_inputs": ["TOS作业指令/完成事件", "箱区队列与箱量预测", "吊机模式/功率/再生电量", "电机/变频器温度", "驻留/启停状态", "PCC需量与电价/碳因子", "DR事件", "故障与维护窗口"],
                "outputs": ["功率上限残差", "待机超时残差", "eco档位", "安全投影原因", "TTL写点任务"],
                "hard_constraints": (formal.get("contract") or {}).get("hard_constraints"),
                "acceptance": ["moves/h与作业时延不劣化", "TOS任务SLA", "kWh/move", "峰值需量", "电费/碳", "多种子时间盲测", "PLC/TOS影子运行", "故障回退与回读成功率"],
                "replacement": "保留36维状态、2维动作、策略JSON与CMDP投影；用现场PLC/TOS适配器、温度传感和标定性能曲线替换工程源后重训。",
            },
            "history_series": convergence.get("aggregate_curve") or [],
            "training_process": load_seed_process_evidence(self.repo_root, latest),
            "checkpoint_reward_replay": load_checkpoint_reward_replay(self.repo_root, latest),
            "legacy_history_series": self._legacy_series(steps),
        }
