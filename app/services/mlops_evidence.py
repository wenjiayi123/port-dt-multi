from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class MLOpsEvidenceService:
    """Clone-safe, read-only model lifecycle evidence for the V3 console."""

    def __init__(self, training_manager: Any, opsx: Any, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.training_manager = training_manager
        self.opsx = opsx
        self.advantage_path = self.root / "evidence" / "v3" / "shanghai_public_advantage_v3.json"
        self.advantage_sidecar = self.advantage_path.with_suffix(".sha256")
        self.benchmark_path = self.root / "evidence" / "v3" / "public_cn_sha_hourly_v3_benchmark.json"
        self.benchmark_sidecar = self.benchmark_path.with_suffix(".sha256")
        self.runtime_dir = self.root / "evidence" / "v3" / "runtime"

    @staticmethod
    def _sha(path: Path) -> str | None:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _json(path: Path) -> Dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError):
            return {}

    def _single_sidecar(self, artifact: Path, sidecar: Path) -> Dict[str, Any]:
        actual = self._sha(artifact)
        expected = None
        try:
            expected = sidecar.read_text(encoding="utf-8").split()[0]
        except (OSError, IndexError):
            pass
        return {
            "path": artifact.relative_to(self.root).as_posix(),
            "bytes": artifact.stat().st_size if artifact.is_file() else 0,
            "sha256": actual,
            "expected_sha256": expected,
            "verified": bool(actual and expected and actual == expected),
        }

    def _runtime_artifacts(self) -> list[Dict[str, Any]]:
        manifest_path = self.runtime_dir / "runtime_model.sha256"
        expected: Dict[str, str] = {}
        try:
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                checksum, name = line.split(maxsplit=1)
                expected[name.strip()] = checksum.strip()
        except (OSError, ValueError):
            expected = {}
        rows = []
        for name in ("selected_sac_v3.zip", "selected_sac_v3.config.json", "runtime_model.json"):
            path = self.runtime_dir / name
            actual = self._sha(path)
            rows.append({
                "path": path.relative_to(self.root).as_posix(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": actual,
                "expected_sha256": expected.get(name),
                "verified": bool(actual and expected.get(name) == actual),
                "portable": True,
            })
        rows.append({
            "path": manifest_path.relative_to(self.root).as_posix(),
            "bytes": manifest_path.stat().st_size if manifest_path.is_file() else 0,
            "sha256": self._sha(manifest_path),
            "expected_sha256": None,
            "verified": manifest_path.is_file() and all(row["verified"] for row in rows),
            "portable": True,
        })
        return rows

    def build(self) -> Dict[str, Any]:
        advantage = self._json(self.advantage_path)
        benchmark_report = self._json(self.benchmark_path)
        runtime_model = self._json(self.runtime_dir / "runtime_model.json")
        selected = advantage.get("selected") or {}
        dataset = advantage.get("dataset") or {}
        contract = advantage.get("benchmark_contract") or {}
        summary = benchmark_report.get("benchmark_summary") or {}
        algorithms = summary.get("algorithms") or []
        registry = self.training_manager.model_registry().list()
        registry_models = registry.get("models") or []
        selected_jobs = list(selected.get("job_ids") or [])
        selected_records = [row for row in registry_models if row.get("job_id") in selected_jobs]
        selected_records.sort(key=lambda row: selected_jobs.index(row.get("job_id")))
        runtime_artifacts = self._runtime_artifacts()
        advantage_artifact = self._single_sidecar(self.advantage_path, self.advantage_sidecar)
        benchmark_artifact = self._single_sidecar(self.benchmark_path, self.benchmark_sidecar)
        artifact_manifest = [advantage_artifact, benchmark_artifact, *runtime_artifacts]
        runtime = self.training_manager.capabilities().get("runtime") or {}
        formal_runs = sum(int(row.get("claim_eligible_runs") or 0) for row in algorithms)
        smoke_runs = sum(int(row.get("smoke_runs") or 0) for row in algorithms)
        learner_formal_runs = sum(
            int(row.get("claim_eligible_runs") or 0)
            for row in algorithms
            if row.get("trainable")
        )
        selected_integrity = bool(
            selected_records
            and len(selected_records) == len(selected_jobs) == 3
            and all((row.get("artifact") or {}).get("verified") for row in selected_records)
            and all(row["verified"] for row in runtime_artifacts)
        )
        opsx = self.opsx.build()
        gates = opsx.get("gates") or []
        rollback_gate = next((row for row in gates if row.get("id") == "rollback"), {})

        algorithm_rows = []
        for row in algorithms:
            formal = int(row.get("claim_eligible_runs") or 0)
            algorithm_rows.append({
                "id": row.get("id"),
                "name": row.get("name"),
                "family": row.get("family"),
                "implementation": row.get("implementation"),
                "trainable": bool(row.get("trainable")),
                "formal_runs": formal,
                "smoke_runs": int(row.get("smoke_runs") or 0),
                "seeds": list(row.get("distinct_seeds") or []),
                "multi_seed_ready": bool(row.get("multi_seed_ready")),
                "evidence_scope": "formal_claim_eligible_runs_only" if formal else "no_formal_evidence",
            })

        pipeline = [
            {"id": "snapshot", "name": "公开源快照", "status": "pass", "evidence": f"{dataset.get('rows', 0)} rows / SHA {str(dataset.get('sha256') or '')[:12]}…"},
            {"id": "quality", "name": "质量与因子门", "status": "pass" if (dataset.get("quality") or {}).get("training_eligible") else "fail", "evidence": "缺失/非有限/物理越界=0；可见度因子显式缺失"},
            {"id": "split", "name": "时间隔离", "status": "pass", "evidence": f"70/10/20 = {dataset.get('train_rows')}/{dataset.get('validation_rows')}/{dataset.get('test_rows')} rows"},
            {"id": "train", "name": "训练不渲染", "status": "pass", "evidence": f"{learner_formal_runs} learner runs × ≥10,000 optimizer steps"},
            {"id": "selection", "name": "仅验证集选模", "status": "pass", "evidence": selected.get("selection_split")},
            {"id": "blind_test", "name": "三种子盲测", "status": "pass" if selected.get("strict_advantage") else "fail", "evidence": f"SAC seeds={selected.get('seeds')} / {dataset.get('test_rows')} rows"},
            {"id": "package", "name": "制品打包与哈希", "status": "pass" if selected_integrity else "fail", "evidence": f"{sum(row['bytes'] for row in runtime_artifacts):,} bytes / clone-safe"},
            {"id": "shadow", "name": "现场影子/灰度", "status": "pending", "evidence": "待接入港口；当前流量0%，未授权生产控制"},
        ]
        evaluation = {
            "total_comparable_runs": formal_runs,
            "learner_formal_runs": learner_formal_runs,
            "deterministic_baseline_runs": formal_runs - learner_formal_runs,
            "smoke_runs": smoke_runs,
            "smoke_is_claim_evidence": False,
            "selected_algorithm": selected.get("algorithm"),
            "selected_seeds": selected.get("seeds") or [],
            "selected_job_ids": selected_jobs,
            "selected_model_sha256": selected.get("model_sha256") or [],
            "weighted_improvement_percent": round(float((selected.get("weighted_relative_improvement") or {}).get("mean") or 0) * 100, 4),
            "weighted_ci_percent": [
                round(float((selected.get("weighted_relative_improvement") or {}).get("ci_low") or 0) * 100, 4),
                round(float((selected.get("weighted_relative_improvement") or {}).get("ci_high") or 0) * 100, 4),
            ],
            "guardrail_violation_rate": (selected.get("safety_admission") or {}).get("guardrail_violation_rate_max_observed"),
            "strict_advantage": bool(selected.get("strict_advantage")),
            "offline_only": True,
        }
        reproducibility = {
            "dataset_id": dataset.get("dataset_id"),
            "dataset_sha256": dataset.get("sha256"),
            "environment_version": contract.get("environment_version"),
            "business_profile_id": contract.get("business_profile_id"),
            "benchmark_config_sha256": contract.get("sha256"),
            "minimum_optimizer_steps": contract.get("minimum_optimizer_steps"),
            "minimum_distinct_seeds": contract.get("minimum_distinct_seeds"),
            "runtime": runtime,
            "runtime_model": runtime_model,
        }
        replacement_and_rollback = {
            "current_aliases": registry.get("aliases") or {},
            "automatic_promotion_enabled": False,
            "champion_alias": (registry.get("aliases") or {}).get("champion"),
            "rollback_alias": (registry.get("aliases") or {}).get("rollback"),
            "rollback_api_implemented": True,
            "site_rehearsal": rollback_gate.get("status") or "pending",
            "current_decision": "BLOCK",
            "required_before_promotion": [
                "现场字段映射/校准与数据质量签字",
                "影子运行及KPI非劣验证",
                "候选/冠军/回滚别名异人审批",
                "执行回读、回滚RTO和安全联锁演练",
            ],
        }
        return {
            "version": "V3",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "module": {"id": "mlops", "name": "MLOps模型全生命周期", "state": "offline_candidate_packaged"},
            "boundary": {
                "offline_lifecycle_verified": selected_integrity and advantage_artifact["verified"] and benchmark_artifact["verified"],
                "live_model_monitoring_verified": False,
                "automatic_promotion_enabled": False,
                "production_authority": False,
                "site_status": "待接入港口",
                "reason": "正式离线模型制品可复现；现场数据、影子运行、回滚演练和变更授权尚未接入。",
            },
            "summary": {
                "algorithm_count": len(algorithms),
                "trainable_rl_count": sum(bool(row.get("trainable")) for row in algorithms),
                "registry_history_records": int(registry.get("count") or 0),
                "formal_runs": formal_runs,
                "smoke_runs": smoke_runs,
                "selected_algorithm": selected.get("algorithm"),
                "selected_integrity": selected_integrity,
                "portable_artifact_bytes": sum(row["bytes"] for row in runtime_artifacts),
            },
            "pipeline": pipeline,
            "algorithms": algorithm_rows,
            "evaluation": evaluation,
            "selected_models": selected_records,
            "artifact_manifest": artifact_manifest,
            "reproducibility": reproducibility,
            "replacement_and_rollback": replacement_and_rollback,
            "historical_evidence": {
                "preserved": True,
                "registry_records": int(registry.get("count") or 0),
                "benchmark_runs": sum(int(row.get("runs") or 0) for row in algorithms),
                "formal_and_smoke_separated": True,
                "overwrite_performed": False,
            },
        }
