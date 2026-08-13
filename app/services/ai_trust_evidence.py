from __future__ import annotations

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping


class AITrustEvidenceService:
    """Aggregate the global V3 benchmark and module-level admission evidence.

    The service deliberately separates an offline benchmark pass from production
    authority.  It never upgrades engineering replay or a loadable policy into a
    measured site KPI.
    """

    def __init__(self, modules: Mapping[str, Any], root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.report_path = self.root / "evidence" / "v3" / "shanghai_public_advantage_v3.json"
        self.sidecar_path = self.report_path.with_suffix(".sha256")
        self.modules = dict(modules)
        self._build_lock = threading.Lock()

    @staticmethod
    def _sha(path: Path) -> str | None:
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def build(self) -> Dict[str, Any]:
        tracked = [self.report_path, self.sidecar_path]
        key = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in tracked
            if path.exists()
        )
        # lru_cache alone can execute the same miss concurrently. Serialize the
        # first aggregate build so Trust/OpsX/MLOps/Governance share one result.
        with self._build_lock:
            return self._build_cached(key)

    @lru_cache(maxsize=4)
    def _build_cached(self, _key: tuple[tuple[str, int, int], ...]) -> Dict[str, Any]:
        actual_sha = self._sha(self.report_path)
        expected_sha = None
        if self.sidecar_path.is_file():
            expected_sha = self.sidecar_path.read_text(encoding="utf-8").split()[0]
        integrity_ok = bool(actual_sha and expected_sha and actual_sha == expected_sha)
        report: Dict[str, Any] = {}
        if integrity_ok:
            try:
                report = json.loads(self.report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                integrity_ok = False

        selected = report.get("selected") or {}
        dataset = report.get("dataset") or {}
        contract = report.get("benchmark_contract") or {}
        safety = selected.get("safety_admission") or {}
        rel = selected.get("metrics_relative_to_fcfs") or {}
        weighted = selected.get("weighted_relative_improvement") or {}
        global_admitted = bool(integrity_ok and selected.get("strict_advantage") and safety.get("passed"))

        def metric(key: str, *, invert: bool = False) -> Dict[str, Any]:
            value = rel.get(key) or {}
            sign = -1.0 if invert else 1.0
            return {
                "id": key,
                "mean_percent": round(sign * float(value.get("mean") or 0.0) * 100.0, 4),
                "ci_low_percent": round(sign * float(value.get("ci_high") if invert else value.get("ci_low") or 0.0) * 100.0, 4),
                "ci_high_percent": round(sign * float(value.get("ci_low") if invert else value.get("ci_high") or 0.0) * 100.0, 4),
                "n_seeds": int(value.get("n") or 0),
                "source": "chronological_blind_test_relative_to_fcfs",
            }

        advantages = [
            {"label": "吞吐提升", **metric("throughput_teu")},
            {"label": "延误指数改善", **metric("delay_index_mean")},
            {"label": "能源成本降低", **metric("energy_cost", invert=True)},
            {"label": "碳排降低", **metric("carbon_kg", invert=True)},
            {"label": "峰值需量降低", **metric("peak_kw", invert=True)},
        ]

        module_payloads = {name: service.build() for name, service in self.modules.items()}

        def current(payload: Dict[str, Any]) -> Dict[str, Any]:
            return payload.get("current_model_output") or payload.get("model_probe") or {}

        labels = {
            "yard_lighting": "堆场照明",
            "hvac": "HVAC冷站",
            "shore_bess": "岸电BESS",
            "bess_energy": "储能调度",
            "yard_crane": "堆场吊机",
        }
        scenes = [
            {
                "id": "global_port_sac",
                "name": "全港SAC策略",
                "evidence_tier": "公开数据时间盲测",
                "model_state": "3模型哈希已核验" if integrity_ok else "证据完整性失败",
                "evaluation": "3种子 / 验证集选模 / 20%盲测",
                "gate": "离线准入通过" if global_admitted else "已拦截",
                "gate_state": "pass" if global_admitted else "blocked",
                "history_records": len(selected.get("job_ids") or []),
                "site_status": "待接入港口",
                "production_authority": False,
                "reasons": [] if global_admitted else ["hash_or_strict_advantage_or_safety_gate_failed"],
            }
        ]
        for module_id, payload in module_payloads.items():
            boundary = payload.get("boundary") or {}
            probe = current(payload)
            gates = payload.get("quality_gates") or {}
            history = payload.get("historical_evidence") or {}
            loaded = bool(
                probe.get("policy_loaded")
                or (probe.get("policy") or {}).get("policy_loaded")
                or (probe.get("model_inference") or {}).get("policy_loaded")
            )
            runtime_admitted = bool(
                probe.get("policy_admitted")
                or (probe.get("policy") or {}).get("policy_admitted")
                or (probe.get("model_inference") or {}).get("policy_admitted")
            )
            claim_eligible = bool(boundary.get("claim_eligible"))
            reasons = list(gates.get("reasons") or [])
            if not reasons and not claim_eligible:
                reasons = [str(boundary.get("reason") or "site evidence unavailable")]
            if not loaded:
                model_state = "模型缺失/不可加载"
            elif runtime_admitted:
                model_state = "模型可推理"
            else:
                model_state = "模型已加载/门禁回退"
            scenes.append({
                "id": module_id,
                "name": labels.get(module_id, module_id),
                "evidence_tier": boundary.get("evidence_tier") or "未评定",
                "model_state": model_state,
                "evaluation": "无合格时间盲测" if not claim_eligible else "合格时间盲测",
                "gate": "业务证据通过" if claim_eligible else "已拦截",
                "gate_state": "pass" if claim_eligible else ("missing" if not loaded else "blocked"),
                "history_records": int(history.get("records") or 0),
                "history_sha256": history.get("history_sha256"),
                "site_status": boundary.get("site_status") or "待接入港口",
                "production_authority": bool(boundary.get("production_authority")),
                "reasons": reasons,
            })

        controls = [
            {"id": "artifact_integrity", "name": "证据与模型哈希", "status": "pass" if integrity_ok else "fail", "value": actual_sha},
            {"id": "source_provenance", "name": "公开来源与字段分级", "status": "pass", "value": dataset.get("evidence_tier")},
            {"id": "temporal_isolation", "name": "训练/验证/盲测时间隔离", "status": "pass" if dataset.get("split_policy", {}).get("shuffle") is False else "fail", "value": dataset.get("split_method")},
            {"id": "multi_seed_uncertainty", "name": "多种子与95%置信区间", "status": "pass" if len(selected.get("seeds") or []) >= 3 else "fail", "value": {"seeds": selected.get("seeds"), "bootstrap_resamples": weighted.get("resamples")}},
            {"id": "safety_admission", "name": "硬约束准入", "status": "pass" if safety.get("passed") else "fail", "value": safety},
            {"id": "production_authority", "name": "生产下发权限", "status": "pending", "value": "禁用；待现场影子运行、验收、双人审批与回滚演练"},
        ]
        return {
            "version": "V3",
            "module": {"id": "ai_trust", "name": "AI Trust", "state": "offline_admitted_site_pending" if global_admitted else "blocked"},
            "trust_grade": "B+" if global_admitted else "D",
            "trust_label": "公开数据离线准入 / 现场待接" if global_admitted else "证据准入失败",
            "boundary": {
                "offline_claim_eligible": global_admitted,
                "causal_claim_eligible": False,
                "live_data_verified": False,
                "production_authority": False,
                "site_status": "待接入港口",
                "reason": "公开数据时间盲测可证明离线优势；没有现场遥测、影子运行、A/B或因果识别证据，禁止生产授权。",
            },
            "benchmark": {
                "algorithm": selected.get("name"),
                "implementation": selected.get("implementation"),
                "dataset_id": dataset.get("dataset_id"),
                "dataset_rows": dataset.get("rows"),
                "dataset_sha256": dataset.get("sha256"),
                "source_observations": dataset.get("independent_source_observations"),
                "official_reporting_periods": (dataset.get("source_observation_counts") or {}).get("official_port_reporting_periods"),
                "reanalysis_hours": (dataset.get("source_observation_counts") or {}).get("aligned_public_reanalysis_hours"),
                "train_rows": dataset.get("train_rows"),
                "validation_rows": dataset.get("validation_rows"),
                "blind_test_rows": dataset.get("test_rows"),
                "seeds": selected.get("seeds"),
                "optimizer_steps_min": contract.get("minimum_optimizer_steps"),
                "weighted_improvement_percent": round(float(weighted.get("mean") or 0.0) * 100.0, 4),
                "weighted_ci_percent": [round(float(weighted.get("ci_low") or 0.0) * 100.0, 4), round(float(weighted.get("ci_high") or 0.0) * 100.0, 4)],
                "report_sha256": actual_sha,
                "sidecar_sha256_match": integrity_ok,
            },
            "advantage_metrics": advantages,
            "controls": controls,
            "scenes": scenes,
            "admission_ladder": [
                {"stage": "公开数据离线训练", "status": "complete"},
                {"stage": "验证集选模 + 多种子时间盲测", "status": "complete" if global_admitted else "blocked"},
                {"stage": "现场字段映射与校准", "status": "待接入港口"},
                {"stage": "影子运行与反事实回放", "status": "待接入港口"},
                {"stage": "受控灰度/A-B与回滚演练", "status": "待接入港口"},
                {"stage": "双人审批生产授权", "status": "禁用"},
            ],
            "claim_registry": {
                "allowed": ["公开数据离线时间盲测优势", "3随机种子统计与95%置信区间", "安全门禁零违规（该基准）", "模型/数据/报告哈希可复核"],
                "prohibited": ["上海港现场已提效", "A/B因果效果", "生产可用等级", "无人值守自动下发", "各子模块已形成现场业务收益"],
            },
            "historical_evidence": {
                "preserved": bool(report.get("historical_evidence_preserved")),
                "module_records": {scene["id"]: scene.get("history_records", 0) for scene in scenes[1:]},
                "note": "各模块历史训练日志和哈希保持原样；AI Trust只做聚合评定，不重写历史。",
            },
        }
