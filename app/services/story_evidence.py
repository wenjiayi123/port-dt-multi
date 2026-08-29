"""Evidence-backed Story Mode replay for the V3 dashboard.

The service pairs the hash-selected SAC policy with a configuration-compatible
FCFS blind-test trajectory.  It never treats the most recently run job as the
deployed policy and it fails closed for ports without a calibrated trace.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class StoryEvidenceService:
    def __init__(self, run_root: Path, strategy_runtime: Any) -> None:
        self.run_root = Path(run_root)
        self.strategy_runtime = strategy_runtime
        self.runtime_root = self.run_root.parents[2] / "evidence/v3/runtime"
        self._baseline_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _run_bundle(self, job_id: str) -> dict[str, Any]:
        root = self.run_root / job_id
        return {
            "job_id": job_id,
            "config": self._read_json(root / "config.json"),
            "evaluation": self._read_json(root / "evaluation.json"),
            "trajectory": self._read_json(root / "evaluation_trajectory.json"),
            "manifest": self._read_json(root / "manifest.json"),
        }

    def _runtime_evidence(self) -> dict[str, Any]:
        metadata = self._read_json(self.runtime_root / "runtime_model.json")
        model_path = self.runtime_root / str(metadata.get("model_artifact") or "")
        config_path = self.runtime_root / str(metadata.get("config_artifact") or "")
        try:
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        except OSError:
            return {}
        if model_sha != metadata.get("model_sha256") or config_sha != metadata.get("config_sha256"):
            return {}
        return metadata

    @staticmethod
    def _trajectory_signature(bundle: dict[str, Any]) -> tuple[Any, ...]:
        frames = (bundle.get("trajectory") or {}).get("frames") or []
        return (
            len(frames),
            frames[0].get("timestamp") if frames else None,
            frames[-1].get("timestamp") if frames else None,
        )

    def _compatible_fcfs(self, selected: dict[str, Any]) -> dict[str, Any]:
        selected_job = str(selected.get("job_id") or "")
        if selected_job in self._baseline_cache:
            return self._baseline_cache[selected_job]
        cfg = selected.get("config") or {}
        signature = self._trajectory_signature(selected)
        candidates: list[dict[str, Any]] = []
        for root in self.run_root.glob("rl-*"):
            candidate = self._run_bundle(root.name)
            other = candidate.get("config") or {}
            if other.get("algorithm") != "fcfs":
                continue
            if any(
                other.get(key) != cfg.get(key)
                for key in (
                    "dataset_id",
                    "dataset_fingerprint",
                    "environment_version",
                    "episode_steps",
                    "seed",
                    "test_ratio",
                    "validation_ratio",
                )
            ):
                continue
            if self._trajectory_signature(candidate) != signature:
                continue
            if not ((candidate.get("evaluation") or {}).get("metrics") or {}):
                continue
            candidates.append(candidate)
        if not candidates:
            return {}
        result = sorted(candidates, key=lambda row: row["job_id"])[-1]
        self._baseline_cache[selected_job] = result
        return result

    @staticmethod
    def _improve(new: float | None, old: float | None, *, lower_better: bool = False) -> float | None:
        if new is None or old is None or abs(float(old)) < 1e-9:
            return None
        delta = float(old) - float(new) if lower_better else float(new) - float(old)
        return delta / abs(float(old)) * 100.0

    @staticmethod
    def _metric(frame: dict[str, Any], name: str) -> float | None:
        value = frame.get(name)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _frame_card(self, frame: dict[str, Any], *, label: str) -> dict[str, Any]:
        served = self._metric(frame, "served_teu")
        cost = self._metric(frame, "energy_cost")
        carbon = self._metric(frame, "carbon_kg")
        return {
            "label": label,
            "load_kw": self._metric(frame, "net_load_kw"),
            "energy_cost_cny": cost,
            "served_teu": served,
            "cost_per_teu": cost / served if cost is not None and served and served > 0 else None,
            "carbon_kg_per_teu": carbon / served if carbon is not None and served and served > 0 else None,
            "delay_index": self._metric(frame, "delay_index"),
            "queue_teu": self._metric(frame, "queue"),
            "guardrail_violation": bool(frame.get("guardrail_violation")),
        }

    @staticmethod
    def _hour_for_index(index: int, count: int) -> int:
        if count <= 1:
            return 0
        return round(-24 + index / (count - 1) * 48)

    def _events(self, baseline: list[dict[str, Any]], policy: list[dict[str, Any]]) -> list[dict[str, Any]]:
        count = min(len(baseline), len(policy))
        if not count:
            return []
        peak_index = max(range(count), key=lambda i: float(baseline[i].get("net_load_kw") or 0.0))
        relief_index = max(
            range(count),
            key=lambda i: float(baseline[i].get("queue") or 0.0) - float(policy[i].get("queue") or 0.0),
        )
        service_index = max(
            range(count),
            key=lambda i: float(policy[i].get("served_teu") or 0.0) - float(baseline[i].get("served_teu") or 0.0),
        )
        return [
            {
                "t": self._hour_for_index(peak_index, count),
                "label": f"FCFS 负荷峰值 {float(baseline[peak_index].get('net_load_kw') or 0.0):.0f} kW",
            },
            {
                "t": self._hour_for_index(relief_index, count),
                "label": f"积压差最大 {max(0.0, float(baseline[relief_index].get('queue') or 0.0) - float(policy[relief_index].get('queue') or 0.0)):.0f} TEU",
            },
            {
                "t": self._hour_for_index(service_index, count),
                "label": f"单步服务增量 {max(0.0, float(policy[service_index].get('served_teu') or 0.0) - float(baseline[service_index].get('served_teu') or 0.0)):.0f} TEU",
            },
        ]

    def summary(self, *, hour: int, port: str, replay: str) -> dict[str, Any]:
        if port != "shanghai":
            return {
                "available": False,
                "status": "pending_port_connection",
                "port": port,
                "reason": "该港口的授权 TOS/EMS/作业轨迹待接入港口；不复用上海结果。",
                "production_authority": False,
            }
        if replay != "sac_vs_fcfs":
            return {
                "available": False,
                "status": "pending_port_connection",
                "port": port,
                "reason": "该专题场景尚无同口径盲测轨迹，待接入港口后生成。",
                "production_authority": False,
            }

        runtime_metadata = self._runtime_evidence()
        selected_job = str(runtime_metadata.get("job_id") or "")
        selected = self._run_bundle(selected_job) if selected_job else {}
        baseline = self._compatible_fcfs(selected) if selected else {}
        policy_frames = (selected.get("trajectory") or {}).get("frames") or []
        baseline_frames = (baseline.get("trajectory") or {}).get("frames") or []
        count = min(len(policy_frames), len(baseline_frames))
        if not selected_job or count < 2:
            return {
                "available": False,
                "status": "evidence_unavailable",
                "reason": "未找到已选 SAC 与同配置 FCFS 的时间对齐盲测轨迹；未生成替代故事。",
                "production_authority": False,
            }

        index = round((hour + 24) / 48 * (count - 1))
        index = max(0, min(count - 1, index))
        policy_frame = policy_frames[index]
        baseline_frame = baseline_frames[index]
        policy_card = self._frame_card(policy_frame, label="已选 SAC 策略")
        baseline_card = self._frame_card(baseline_frame, label="FCFS 基线")
        policy_metrics = (selected.get("evaluation") or {}).get("metrics") or {}
        baseline_metrics = (baseline.get("evaluation") or {}).get("metrics") or {}
        window_improvement = {
            "throughput_teu": self._improve(policy_metrics.get("throughput_teu"), baseline_metrics.get("throughput_teu")),
            "delay_index_mean": self._improve(policy_metrics.get("delay_index_mean"), baseline_metrics.get("delay_index_mean"), lower_better=True),
            "cost_per_teu": self._improve(policy_metrics.get("cost_per_teu"), baseline_metrics.get("cost_per_teu"), lower_better=True),
            "carbon_kg_per_teu": self._improve(policy_metrics.get("carbon_kg_per_teu"), baseline_metrics.get("carbon_kg_per_teu"), lower_better=True),
        }
        return {
            "available": True,
            "schema": "port-dt-v3-story-evidence.v1",
            "hour": hour,
            "source_timestamp": policy_frame.get("timestamp"),
            "events": self._events(baseline_frames[:count], policy_frames[:count]),
            "baseline": baseline_card,
            "policy": policy_card,
            "frame_improvement_percent": {
                "cost_per_teu": self._improve(policy_card.get("cost_per_teu"), baseline_card.get("cost_per_teu"), lower_better=True),
                "carbon_kg_per_teu": self._improve(policy_card.get("carbon_kg_per_teu"), baseline_card.get("carbon_kg_per_teu"), lower_better=True),
                "delay_index": self._improve(policy_card.get("delay_index"), baseline_card.get("delay_index"), lower_better=True),
            },
            "blind_test_summary": {
                "episodes": (selected.get("evaluation") or {}).get("episodes"),
                "improvement_percent": window_improvement,
                "guardrail_violation_rate": policy_metrics.get("guardrail_violation_rate"),
            },
            "evidence": {
                "policy_job_id": selected_job,
                "baseline_job_id": baseline.get("job_id"),
                "policy_algorithm": (selected.get("config") or {}).get("algorithm"),
                "baseline_algorithm": (baseline.get("config") or {}).get("algorithm"),
                "dataset_id": (selected.get("config") or {}).get("dataset_id"),
                "dataset_sha256": (selected.get("config") or {}).get("dataset_fingerprint"),
                "model_sha256": runtime_metadata.get("model_sha256"),
                "split": "chronological_blind_test_only",
                "aligned_frames": count,
            },
            "claim_boundary": "上海公开聚合与公开再分析数据上的离线留出集回放；不是码头现场 KPI、财务审计或自动执行授权。",
            "production_authority": False,
        }

    def play_ack(self) -> dict[str, Any]:
        runtime_metadata = self._runtime_evidence()
        return {
            "ok": True,
            "mode": "heldout_evidence_replay",
            "policy_job_id": runtime_metadata.get("job_id"),
            "side_effect": "none",
            "production_authority": False,
        }
