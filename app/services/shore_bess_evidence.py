from __future__ import annotations

import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.services.rl_model.shore_bess.api import PolicyRunner
from app.services.rl_model.shore_bess.v3_environment import (
    ShoreBESSEnv,
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


class ShoreBESSEvidenceService:
    """Expose the legacy Shore+BESS run with V3 admission and provenance gates."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent / "rl_model" / "shore_bess"
        self.data = self.root / "data"
        self.artifacts = self.root / "artifacts"
        self.repo_root = self.root.parents[3]
        self.v3_evidence = self.repo_root / "evidence" / "v3" / "shore_bess"

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
        rows: List[Dict[str, Any]] = []
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

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
    def _sample_series(rows: List[Dict[str, Any]], limit: int = 180) -> List[Dict[str, Any]]:
        source = [row for row in rows if row.get("key") == "core_metrics"]
        stride = max(1, math.ceil(len(source) / limit))
        selected = source[::stride]
        if source and selected[-1] is not source[-1]:
            selected.append(source[-1])
        return [
            {
                "step": int(row.get("train_step") or 0),
                "raw_reward": ShoreBESSEvidenceService._num(row.get("raw_reward_component")),
                "raw_economic": ShoreBESSEvidenceService._num(row.get("raw_econ_component")),
                "raw_carbon": ShoreBESSEvidenceService._num(row.get("raw_carbon_component")),
                "raw_service": ShoreBESSEvidenceService._num(row.get("raw_service_component")),
                "display_reward": ShoreBESSEvidenceService._num(row.get("cumulative_reward")),
            }
            for row in selected
        ]

    def build(self) -> Dict[str, Any]:
        latest_path = self.v3_evidence / "latest.json"
        formal_paths: List[Path] = [latest_path]
        if latest_path.exists():
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                report_path = self.repo_root / str(latest.get("report_path") or "")
                formal_paths.append(report_path)
                if report_path.exists():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    for item in (report.get("artifacts") or {}).get("models") or []:
                        formal_paths.append(self.repo_root / str(item.get("path") or ""))
                    formal_paths.extend(seed_metric_paths(self.repo_root, latest))
                    formal_paths.append(checkpoint_reward_replay_path(self.repo_root, latest))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        tracked = [
            self.artifacts / "shore_bess_outputs.jsonl",
            self.root / "policy.bin",
            self.root / "policy_meta.json",
            self.data / "berths_master.csv",
            self.data / "ship_calls.csv",
            self.data / "shore_power_telemetry.csv",
            self.data / "bess_telemetry.csv",
            self.data / "grid_meter.csv",
            *formal_paths,
            evidence_path(self.repo_root),
        ]
        key = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in tracked
            if path.exists()
        )
        return self._build_cached(key)

    def _load_formal(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        latest_path = self.v3_evidence / "latest.json"
        if not latest_path.exists():
            return {}, {}
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        report_path = self.repo_root / str(latest.get("report_path") or "")
        if not report_path.exists() or self._sha(report_path) != latest.get("report_sha256"):
            raise RuntimeError("Shore+BESS formal evidence pointer or report hash is invalid")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return latest, report

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
            env = ShoreBESSEnv(
                dataset,
                blind_slice,
                config=config,
                normalization_slice=train_slice,
                episode_steps=int(config["training"]["episode_hours"]),
                seed=int(selected.get("seed") or 43),
                training=False,
                record_trace=False,
            )
            observation, reset_info = env.reset(options={"start_index": 0})
            action, _ = PPO.load(str(model_path), device="cpu").predict(
                observation, deterministic=True
            )
            _next, _reward, _terminated, _truncated, info = env.step(action)
            env.close()
            return {
                "policy_loaded": True,
                "policy_admitted_for_public_offline": bool(
                    (report.get("quality_gates") or {}).get("public_offline_admitted")
                ),
                "production_admitted": False,
                "decision_source": "selected_constrained_actor_plus_safety_projection",
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
                "safety": {
                    "soc": info.get("soc"),
                    "soh": info.get("soh"),
                    "temperature_c": info.get("temperature_c"),
                    "pcc_kw": info.get("pcc_kw"),
                    "reserve_shortfall_kw": info.get("reserve_shortfall_kw"),
                    "shore_sla_shortfall_kw": info.get("shore_sla_shortfall_kw"),
                    "guardrail_violation": info.get("guardrail_violation"),
                },
            }
        except Exception as exc:
            return {
                "policy_loaded": False,
                "error": str(exc),
                "saved_blind_inference": fallback,
            }

    @lru_cache(maxsize=4)
    def _build_cached(self, _key: Tuple[Tuple[str, int, int], ...]) -> Dict[str, Any]:
        latest, formal = self._load_formal()
        history_path = self.artifacts / "shore_bess_outputs.jsonl"
        rows = self._jsonl(history_path)
        baseline = [row for row in rows if row.get("key") == "baseline_dispatch"]
        offline = [row for row in rows if row.get("key") == "offline_dataset"]
        metric_rows = [row for row in rows if row.get("key") == "metrics"]
        metrics = (metric_rows[-1].get("kpis") or {}) if metric_rows else {}
        meta = json.loads((self.root / "policy_meta.json").read_text(encoding="utf-8"))

        nonzero_actions = sum(
            1
            for row in offline
            if any(abs(self._num(value)) > 1e-9 for value in (row.get("act") or []))
        )
        shore_positive = sum(
            1
            for row in baseline
            if sum(self._num(value) for value in (row.get("P_shore") or {}).values()) > 0
        )
        replay_berths = sorted((baseline[0].get("P_shore") or {}).keys()) if baseline else []
        source_berths = sorted(
            row.get("berth_id", "")
            for row in self._csv_rows(self.data / "berths_master.csv")
            if row.get("berth_id")
        )

        runner = PolicyRunner(dt_min=10)
        probe_row = max(runner.baseline, key=lambda row: self._num(row.get("P_pcc_kW")))
        probe = runner.recommend_at(probe_row.get("ts"), record_audit=False)
        raw_values = list((probe.get("model_inference") or {}).get("raw_action", {}).values())
        raw_abs_max = max((abs(self._num(value)) for value in raw_values), default=0.0)
        admission_reasons: List[str] = []
        if nonzero_actions == 0:
            admission_reasons.append("offline_action_support_is_all_zero")
        if shore_positive == 0:
            admission_reasons.append("historical_replay_has_no_shore_load")
        if replay_berths != source_berths:
            admission_reasons.append("historical_berth_schema_drift")
        if raw_abs_max < 1.0:
            admission_reasons.append("policy_output_below_1kW_materiality_gate")
        admitted = bool((probe.get("model_inference") or {}).get("policy_loaded")) and not admission_reasons
        probe["model_inference"].update(
            {
                "policy_admitted": admitted,
                "decision_source": "trained_policy" if admitted else "rule_baseline_fail_closed",
                "raw_abs_max_kW": round(raw_abs_max, 6),
                "admission_reasons": admission_reasons,
            }
        )
        if not admitted:
            probe["recommended"] = dict(probe["baseline"])
            probe["pcc_new_kW"] = probe["baseline"]["P_pcc_kW"]
            probe["save_yuan_step"] = 0.0
            probe["final_action"] = "zero residual; keep rule baseline in shadow mode"

        strategy_cost = self._num(metrics.get("cost_ref_yuan"))
        comparison_cost = self._num(metrics.get("cost_rule_yuan"))
        advantage = self._num(metrics.get("advantage_yuan"), comparison_cost - strategy_cost)
        baseline_charge = sum(max(-self._num(row.get("P_bess_kW")), 0.0) for row in baseline) / 6.0
        baseline_discharge = sum(max(self._num(row.get("P_bess_kW")), 0.0) for row in baseline) / 6.0

        manifests = []
        for name in (
            "berths_master.csv", "ship_calls.csv", "shore_power_telemetry.csv",
            "bess_telemetry.csv", "grid_meter.csv", "market_price.csv", "grid_ef.csv",
            "bess_master.json", "demand_window_config.json",
        ):
            path = self.data / name
            csv_rows = self._csv_rows(path) if path.suffix == ".csv" else []
            manifests.append(
                {"file": name, "rows": len(csv_rows) if path.suffix == ".csv" else None, "sha256": self._sha(path)}
            )

        formal_quality = formal.get("quality_gates") or {}
        formal_business = formal.get("business_metrics") or {}
        formal_convergence = formal.get("convergence") or {}
        v3_inference = self._current_v3_inference(formal) if formal else {
            "policy_loaded": False,
            "error": "formal Shore+BESS V3 evidence is not available",
        }
        production_reasons = [
            "authorized_shore_meter_bms_pcc_and_gateway_not_connected",
            "site_calibration_shadow_operation_and_operator_acceptance_pending",
        ]
        if self._num(formal_business.get("carbon_reduction_vs_no_bess_percent")) < 0:
            production_reasons.append("economic_profile_increases_scenario_carbon")

        return {
            "version": "V3.1",
            "value_improvement": load_module_value_improvement(self.repo_root, "shore_bess"),
            "module": {"id": "shore_bess", "name": "岸电储能", "state": "formal_public_offline_site_pending"},
            "boundary": {
                "evidence_tier": (formal.get("dataset") or {}).get("evidence_tier") or "public_engineering_offline",
                "claim_eligible": False,
                "live_data_verified": False,
                "production_authority": False,
                "site_status": "待接入港口",
                "reason": "V3.1已完成公开数据多种子离线训练和真实权重推断；现场岸电表计、BMS、PCC与网关仍未接入，指标不是上海港实测节省。",
            },
            "historical_evidence": {
                "preserved": True,
                "records": len(rows),
                "history_sha256": self._sha(history_path),
                "policy_sha256": self._sha(self.root / "policy.bin"),
                "policy_meta_sha256": self._sha(self.root / "policy_meta.json"),
                "algorithm": meta.get("algo") or "SAC",
                "steps": 2000,
                "feature_count": len(meta.get("feat_names") or []),
                "action_count": len(meta.get("act_names") or []),
                "note": "2293条原历史与2000步曲线原样保留；含bias/perturb的展示累计值不再作为V3业务证据。",
            },
            "quality_gates": {
                **formal_quality,
                "policy_artifact_loads": bool(v3_inference.get("policy_loaded")),
                "admitted": bool(formal_quality.get("public_offline_admitted")),
                "production_admitted": False,
                "reasons": production_reasons,
                "legacy_audit": {
                    "offline_rows": len(offline),
                    "nonzero_action_rows": nonzero_actions,
                    "shore_positive_rows": shore_positive,
                    "replay_berths": replay_berths,
                    "source_berths": source_berths,
                    "legacy_policy_loaded": bool(probe["model_inference"].get("policy_loaded")),
                    "legacy_policy_admitted": admitted,
                    "legacy_reasons": admission_reasons,
                    "schema_alias_fix": "adapter.py maps ship_call_id/eta_utc/etd_utc/nominal_shore_kw/shore_max_kw",
                },
            },
            "current_model_output": {
                "model_inference": v3_inference,
                "source": "runtime_reload_of_hash_verified_selected_model",
                "not_static_card": True,
            },
            "formal_training": {
                "pointer": latest,
                "status": formal.get("status"),
                "dataset": formal.get("dataset"),
                "training": formal.get("training"),
                "contract": formal.get("contract"),
                "convergence": formal_convergence,
                "blind_test_protocol": {
                    "windows": (formal.get("blind_test") or {}).get("windows"),
                    "window_hours": (formal.get("blind_test") or {}).get("window_hours"),
                },
            },
            "business_metrics": formal_business,
            "legacy_model_output": probe,
            "historical_business_replay": {
                "window": metric_rows[-1].get("window") if metric_rows else None,
                "configured_bess_strategy_cost_yuan": strategy_cost,
                "comparison_rule_cost_yuan": comparison_cost,
                "strategy_advantage_yuan": advantage,
                "strategy_advantage_percent": round(100.0 * advantage / comparison_cost, 4) if comparison_cost else None,
                "energy_kwh": self._num(metrics.get("energy_kWh")),
                "co2_kg": self._num(metrics.get("co2_kg")),
                "peak_roll15_kw": self._num(metrics.get("peak_ref_roll15_kW")),
                "charge_kwh": round(baseline_charge, 3),
                "discharge_kwh": round(baseline_discharge, 3),
                "claim_eligible": False,
                "conclusion": "旧策略未产生可证明提升：工程回放advantage为负，V3不把负结果改写成节省。",
            },
            "algorithm_registry": [
                {
                    "name": row.get("name"),
                    "state": row.get("state"),
                    "artifact": "formal V3.1 evidence" if row.get("state") != "historical_rejected" else "policy.bin",
                    "admission": row.get("reason"),
                }
                for row in (formal.get("algorithm_registry") or [])
            ] + [
                {"name": "Nonce + TTL write gateway", "state": "implemented", "artifact": "api.py", "admission": "shadow by default"},
                {"name": "OPC/Modbus adapter", "state": "contract_ready", "artifact": "api.py", "admission": "待现场白名单与回执"},
            ],
            "data_manifest": {
                "mode": "public_shanghai_chronological_benchmark_plus_preserved_engineering_simulator",
                "measured": False,
                "public_dataset": formal.get("dataset"),
                "legacy_engineering_files": manifests,
            },
            "site_contract": {
                "required_inputs": ["岸电表计/船舶连接状态", "船期与最低保供功率", "BMS SOC/SOH/温度/告警", "PCC 15分钟需量", "实时电价/排放因子", "储能寿命曲线", "N-1备用", "点表读回与时钟质量"],
                "outputs": ["各泊位岸电设定", "BESS充放电功率", "备用容量", "安全遮罩原因", "TTL/nonce写点任务"],
                "hard_constraints": (formal.get("contract") or {}).get("hard_constraints") or ["岸电SLA", "SOC/功率/斜坡", "PCC需量", "禁止反送", "备用不降级", "过温/故障闭锁", "断链回退"],
                "acceptance": ["岸电满足率", "电费与最大需量", "kgCO2e", "等效循环与SOH", "动作可执行率", "遮罩/回退成功率", "多种子留出集提升"],
                "replacement": "保留状态、动作、安全投影、影子网关与审计结构；重建现场字段映射并用非零动作数据重新训练。",
            },
            "history_series": formal_convergence.get("aggregate_curve") or [],
            "training_process": load_seed_process_evidence(self.repo_root, latest),
            "checkpoint_reward_replay": load_checkpoint_reward_replay(self.repo_root, latest),
            "legacy_history_series": self._sample_series(rows),
        }
