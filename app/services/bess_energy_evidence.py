from __future__ import annotations

import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.services.rl_model.bess_energy.module import BessSiteConfig
from app.services.rl_model.bess_energy.rl_engine import FeatureMaker, GaussianPolicy
from app.services.rl_model.bess_energy.v3_environment import (
    BESSEnergyV3Env,
    chronological_slices,
    load_config as load_v3_config,
    load_public_dataset,
)
from app.services.training_process_evidence import (
    checkpoint_reward_replay_path,
    load_checkpoint_reward_replay,
    load_seed_process_evidence,
    seed_metric_paths,
)
from app.services.value_improvement import evidence_path, load_module_value_improvement


class BESSEnergyEvidenceService:
    """Append-only V3.1 evidence for the site BESS policy and admission gates."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent / "rl_model" / "bess_energy"
        self.data = self.root / "data"
        self.repo_root = self.root.parents[3]
        self.v3_evidence = self.repo_root / "evidence" / "v3" / "bess_energy"

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
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    @staticmethod
    def _csv_rows(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _legacy_series(history: List[Dict[str, Any]], limit: int = 180) -> List[Dict[str, Any]]:
        source = [row for row in history if row.get("key") == "rl_train_step"]
        stride = max(1, math.ceil(len(source) / limit))
        selected = source[::stride]
        if source and selected[-1] is not source[-1]:
            selected.append(source[-1])
        return [{
            "step": int(row.get("step") or 0),
            "raw_economic": BESSEnergyEvidenceService._num(row.get("raw_econ_component")),
            "raw_carbon": BESSEnergyEvidenceService._num(row.get("raw_carbon_component")),
            "raw_peak": BESSEnergyEvidenceService._num(row.get("raw_peak_component")),
            "critic_loss": BESSEnergyEvidenceService._num(row.get("critic_loss_1")),
            "actor_loss": BESSEnergyEvidenceService._num(row.get("actor_loss")),
            "mask_applied": int(row.get("mask_applied") or 0),
        } for row in selected]

    def _load_formal(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        latest_path = self.v3_evidence / "latest.json"
        if not latest_path.exists():
            return {}, {}
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        report_path = self.repo_root / str(latest.get("report_path") or "")
        if not report_path.exists() or self._sha(report_path) != latest.get("report_sha256"):
            raise RuntimeError("BESS formal evidence pointer or report hash is invalid")
        return latest, json.loads(report_path.read_text(encoding="utf-8"))

    def build(self) -> Dict[str, Any]:
        formal_paths = [self.v3_evidence / "latest.json"]
        try:
            _latest, formal = self._load_formal()
            if formal:
                formal_paths.append(self.repo_root / str((_latest or {}).get("report_path") or ""))
                for item in (formal.get("artifacts") or {}).get("models") or []:
                    formal_paths.append(self.repo_root / str(item.get("path") or ""))
                formal_paths.extend(seed_metric_paths(self.repo_root, _latest))
                formal_paths.append(checkpoint_reward_replay_path(self.repo_root, _latest))
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            pass
        tracked = [
            self.root / "policy_evaluate_history.jsonl", self.root / "offline_dataset.jsonl",
            self.root / "policy.bin", self.root / "policy_meta.json", self.root / "kpi_cards.json",
            self.data / "bess_telemetry.csv", self.data / "grid_meter.csv", *formal_paths,
            evidence_path(self.repo_root),
        ]
        key = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in tracked if path.exists())
        return self._build_cached(key)

    def _legacy_audit(self, history: List[Dict[str, Any]], dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        meta = json.loads((self.root / "policy_meta.json").read_text(encoding="utf-8"))
        artifact = json.loads((self.root / "policy.bin").read_text(encoding="utf-8"))
        cfg = BessSiteConfig(**meta["cfg_bess"])
        obs_rows = [row.get("obs") or {} for row in dataset if row.get("key") == "transition"]
        maker = FeatureMaker(cfg)
        maker.calibrate([self._num(row.get("price")) for row in obs_rows], [self._num(row.get("ef")) for row in obs_rows])
        policy = GaussianPolicy(16, self._num(artifact.get("residual_band_kW"), 3600.0))
        policy.W = np.asarray(artifact["policy_W"], dtype=np.float64)
        policy.b = np.asarray(artifact["policy_b"], dtype=np.float64)
        policy.log_std = np.asarray(artifact["policy_log_std"], dtype=np.float64)
        sample = obs_rows[:: max(1, len(obs_rows) // 256)] if obs_rows else []
        sample_actions = [policy.deterministic(maker.obs_to_phi(obs)) for obs in sample]
        band = self._num(artifact.get("residual_band_kW"), 3600.0)
        saturation = sum(np.max(np.abs(action)) >= 0.99 * band for action in sample_actions) / len(sample_actions) if sample_actions else 0.0
        transitions = [row for row in dataset if row.get("key") == "transition"]
        nonzero_dp = sum(abs(self._num((row.get("action") or {}).get("dP"))) > 1e-9 for row in transitions)
        nonzero_dr = sum(abs(self._num((row.get("action") or {}).get("dR"))) > 1e-9 for row in transitions)
        event_rows = sum(int(self._num(obs.get("event_active"))) for obs in obs_rows)
        eval_rows = sum(row.get("stage") == "sac_eval" for row in history)
        return {
            "offline_rows": len(transitions), "nonzero_dP_rows": nonzero_dp, "nonzero_dR_rows": nonzero_dr,
            "event_active_rows": event_rows, "heldout_evaluation_rows": eval_rows,
            "sampled_policy_saturation_rate": round(float(saturation), 6),
            "legacy_policy_artifact_loads": True, "legacy_policy_admitted": False,
            "legacy_reasons": ["reserve_action_support_is_zero", "event_active_coverage_is_zero",
                               "no_chronological_heldout_evaluation", "saved_policy_is_saturated_on_sampled_states"],
        }

    def _current_v3_inference(self, report: Dict[str, Any]) -> Dict[str, Any]:
        models = (report.get("artifacts") or {}).get("models") or []
        fallback = (report.get("blind_test") or {}).get("sample_real_model_inference") or {}
        if not models:
            return {"policy_loaded": False, "error": "formal model artifact is missing", **fallback}
        selected = models[0]
        model_path = self.repo_root / str(selected.get("path") or "")
        if not model_path.exists() or self._sha(model_path) != selected.get("sha256"):
            return {"policy_loaded": False, "error": "formal model hash gate failed", **fallback}
        try:
            from stable_baselines3 import PPO
            config = load_v3_config()
            dataset = load_public_dataset(config)
            train_slice, _validation_slice, blind_slice = chronological_slices(dataset)
            env = BESSEnergyV3Env(dataset, blind_slice, config=config, normalization_slice=train_slice,
                                  episode_steps=int(config["training"]["episode_hours"]),
                                  seed=int(selected.get("seed") or 47), training=False, record_trace=False)
            observation, reset_info = env.reset(options={"start_index": 0})
            action, _ = PPO.load(str(model_path), device="cpu").predict(observation, deterministic=True)
            _next, _reward, _terminated, _truncated, info = env.step(action)
            env.close()
            return {
                "policy_loaded": True,
                "policy_admitted_for_public_offline": bool((report.get("quality_gates") or {}).get("public_offline_admitted")),
                "production_admitted": False, "decision_source": "selected_event_aware_actor_plus_cmdp_safety_projection",
                "algorithm": (report.get("training") or {}).get("algorithm"), "seed": selected.get("seed"),
                "model_path": selected.get("path"), "model_sha256": selected.get("sha256"),
                "timestamp": info.get("timestamp"), "reset": reset_info,
                "observation_vector": [round(float(value), 6) for value in observation],
                "state": info.get("context"), "requested_action": info.get("requested_action"),
                "final_action": info.get("final_action"), "projection": info.get("projection"),
                "derived_business_step": info.get("business_step"),
                "safety": {"soc": info.get("soc"), "soh": info.get("soh"), "temperature_c": info.get("temperature_c"),
                           "pcc_kw": info.get("pcc_kw"), "event_shortfall_kw": info.get("event_shortfall_kw"),
                           "guardrail_violation": info.get("guardrail_violation")},
            }
        except Exception as exc:
            return {"policy_loaded": False, "error": str(exc), "saved_blind_inference": fallback}

    @lru_cache(maxsize=4)
    def _build_cached(self, _key: Tuple[Tuple[str, int, int], ...]) -> Dict[str, Any]:
        latest, formal = self._load_formal()
        history_path = self.root / "policy_evaluate_history.jsonl"
        dataset_path = self.root / "offline_dataset.jsonl"
        history, dataset = self._jsonl(history_path), self._jsonl(dataset_path)
        legacy = self._legacy_audit(history, dataset)
        meta = json.loads((self.root / "policy_meta.json").read_text(encoding="utf-8"))
        last = history[-1] if history else {}
        manifests = []
        for name in ("bess_telemetry.csv", "grid_meter.csv", "market_price.csv", "grid_ef.csv", "dr_events.csv",
                     "reserve_events.csv", "bess_master.json", "demand_window_config.json"):
            path = self.data / name
            rows = self._csv_rows(path) if path.suffix == ".csv" else []
            manifests.append({"file": name, "rows": len(rows) if path.suffix == ".csv" else None, "sha256": self._sha(path)})
        quality = formal.get("quality_gates") or {}
        business = formal.get("business_metrics") or {}
        inference = self._current_v3_inference(formal) if formal else {"policy_loaded": False, "error": "formal evidence unavailable"}
        legacy_diag = {
            "raw_economic_component_sum": round(sum(self._num(row.get("raw_econ_component")) for row in history), 6),
            "raw_carbon_component_sum": round(sum(self._num(row.get("raw_carbon_component")) for row in history), 6),
            "mask_rate": round(sum(int(row.get("mask_applied") or 0) for row in history) / len(history), 6) if history else 0.0,
            "final_critic_loss": self._num(last.get("critic_loss_1")), "final_actor_loss": self._num(last.get("actor_loss")),
            "valid_business_improvement": None, "claim_eligible": False,
            "reason": "legacy in-training diagnostics are not chronological held-out KPIs",
        }
        return {
            "version": "V3.1", "module": {"id": "bess_energy", "name": "场内储能", "state": "formal_public_offline_site_pending"},
            "value_improvement": load_module_value_improvement(self.repo_root, "bess_energy"),
            "boundary": {"evidence_tier": (formal.get("dataset") or {}).get("evidence_tier") or "public_engineering_offline",
                         "claim_eligible": False, "live_data_verified": False, "production_authority": False,
                         "site_status": "待接入港口",
                         "reason": "公开数据多种子训练、真实权重推断和工程事件盲测已完成；DR/备用事件不是上海市场结算记录，PCS/BMS/PCC仍待现场接入。"},
            "historical_evidence": {"preserved": True, "records": len(history), "history_sha256": self._sha(history_path),
                                    "offline_rows": legacy["offline_rows"], "offline_sha256": self._sha(dataset_path),
                                    "policy_sha256": self._sha(self.root / "policy.bin"), "policy_meta_sha256": self._sha(self.root / "policy_meta.json"),
                                    "algorithm": meta.get("algo") or "sac", "steps": int(last.get("step") or len(history)),
                                    "note": "2000步与8927条旧transition原样保留；bias/anchor和静态kpi_cards不用于V3.1业务主张。"},
            "quality_gates": {**quality, "policy_artifact_loads": bool(inference.get("policy_loaded")),
                              "admitted": bool(quality.get("public_offline_admitted")), "production_admitted": False,
                              "reasons": ["authorized_pcs_bms_pcc_settlement_and_gateway_not_connected",
                                          "site_shadow_ab_rollback_and_operator_acceptance_pending"],
                              "legacy_audit": legacy},
            "current_model_output": {"model_inference": inference, "source": "runtime_reload_of_hash_verified_selected_model", "not_static_card": True},
            "formal_training": {"pointer": latest, "status": formal.get("status"), "dataset": formal.get("dataset"),
                                "scenario_supplement": formal.get("scenario_supplement"), "training": formal.get("training"),
                                "contract": formal.get("contract"), "convergence": formal.get("convergence"),
                                "blind_test_protocol": {"windows": (formal.get("blind_test") or {}).get("windows"),
                                                        "window_hours": (formal.get("blind_test") or {}).get("window_hours")}},
            "business_metrics": business, "training_diagnostics": legacy_diag,
            "excluded_static_card": {"file": "kpi_cards.json", "sha256": self._sha(self.root / "kpi_cards.json"),
                                     "status": "excluded_from_v3_evidence", "reason": "static values are not linked to model/window provenance"},
            "algorithm_registry": [{"name": row.get("name"), "state": row.get("state"),
                                    "artifact": "formal V3.1 evidence" if row.get("state") != "historical_rejected" else "policy.bin",
                                    "admission": row.get("reason")} for row in (formal.get("algorithm_registry") or [])],
            "data_manifest": {"mode": "public_shanghai_chronological_benchmark_plus_explicit_engineering_event_calendar",
                              "measured": False, "public_dataset": formal.get("dataset"),
                              "event_provenance": formal.get("scenario_supplement"), "legacy_engineering_files": manifests},
            "site_contract": {"required_inputs": ["PCS/BMS功率、SOC、SOH、温度、故障与可用容量", "PCC 15分钟需量与变压器N-1",
                                                        "结算电价/需量电价/辅助服务价格", "真实DR与备用出清、基线及履约记录", "负荷/光伏预测",
                                                        "边际排放因子", "电池退化与效率曲线", "设备联锁、时钟、点表回读、权限与回滚"],
                              "outputs": ["充放电功率", "上调备用容量", "SOC目标", "安全投影原因", "TTL/nonce写点任务"],
                              "hard_constraints": (formal.get("contract") or {}).get("hard_constraints"),
                              "acceptance": ["时间盲测成本/需量/碳", "DR/备用履约率", "SOC/SOH/热安全", "多种子收敛",
                                             "影子运行与A/B", "回读、回滚与运营签字"],
                              "replacement": "保留40维状态、CMDP投影、策略格式和PCS网关；替换数据适配器、合同与工程事件日历后重训。"},
            "history_series": ((formal.get("convergence") or {}).get("aggregate_curve") or []),
            "training_process": load_seed_process_evidence(self.repo_root, latest),
            "checkpoint_reward_replay": load_checkpoint_reward_replay(self.repo_root, latest),
            "legacy_history_series": self._legacy_series(history),
        }
