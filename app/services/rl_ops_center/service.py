# app/services/rl_ops_center/service.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.rl_training.trainer import TRAINING_MANAGER

class RLOpsService:
    """
    RL Ops Center backed by persisted training and held-out evaluations.
    对应前端 tabs：OPE / 守护栏 / 可观测性 / 实验 / 因果
    """

    # ---------- OPE（占位：当前前端OPE页签沿用原Platform脚本，保留概览接口以供后续使用） ----------
    def overview(self) -> Dict[str, Any]:
        benchmark = TRAINING_MANAGER.baselines()
        rows = []
        for item in benchmark.get("baselines", []):
            evaluation = item.get("latest_evaluation") or {}
            metrics = evaluation.get("metrics") or {}
            if not metrics:
                continue
            rows.append({
                "policy": item["id"],
                "family": item["family"],
                "implementation": item["implementation"],
                "dataset_id": evaluation.get("dataset_id"),
                "dataset_sha256": evaluation.get("dataset_sha256"),
                "job_id": evaluation.get("job_id"),
                "energy_cost": metrics.get("energy_cost"),
                "peak_kw": metrics.get("peak_kw"),
                "carbon_kg": metrics.get("carbon_kg"),
                "guardrail_violation_rate": metrics.get("guardrail_violation_rate"),
                "reward": metrics.get("reward"),
            })
        best = min(rows, key=lambda row: float(row["energy_cost"]))["policy"] if rows else None
        return {
            "available": bool(rows),
            "kind": "heldout_policy_evaluation_not_ope",
            "ts": benchmark.get("updated_at"),
            "summary": {
                "best_policy_by_energy_cost": best,
                "evaluated_policies": len(rows),
                "ope_available": False,
            },
            "leaderboard": rows,
            "_source": benchmark.get("source"),
        }

    def ope_eval(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "available": False,
            "metric": str(payload.get("metric", "delta_kWh")),
            "reason": "True OPE requires logged behavior-policy probabilities and action trajectories; held-out evaluation is exposed by /overview instead.",
        }

    # ---------- 守护栏 ----------
    def list_policies(self) -> Dict[str, Any]:
        return {"items": [
            {"level": "hard", "rule": "peak_kw <= contract_kw", "reason": "防越峰"},
            {"level": "hard", "rule": "soc in [0.2, 0.9]", "reason": "保护BESS"},
            {"level": "soft", "rule": "shore_power_kw + bess_charge_kw <= feeder_limit", "reason": "馈线限额"},
        ], "_source": "rl_training.environment_constraints"}

    def verify_policy(self, strategy_id: str) -> Dict[str, Any]:
        item = next(
            (row for row in TRAINING_MANAGER.baselines().get("baselines", []) if row.get("id") == strategy_id),
            None,
        )
        evaluation = (item or {}).get("latest_evaluation") or {}
        metrics = evaluation.get("metrics") or {}
        rate = metrics.get("guardrail_violation_rate")
        if rate is None:
            return {"ok": False, "available": False, "strategy_id": strategy_id, "reason": "No held-out evaluation found"}
        return {
            "ok": float(rate) <= 0.05,
            "available": True,
            "strategy_id": strategy_id,
            "guardrail_violation_rate": rate,
            "threshold": 0.05,
            "job_id": evaluation.get("job_id"),
            "dataset_sha256": evaluation.get("dataset_sha256"),
        }

    # ---------- 可观测性 ----------
    def signals(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        selected_status: Dict[str, Any] = {}
        selected_history: List[Dict[str, Any]] = []
        registry_rows = TRAINING_MANAGER.model_registry().list().get("models", [])
        available_algorithms = [
            row.get("id")
            for row in TRAINING_MANAGER.benchmark_summary(
                dataset_id="public_cn_sha_hourly_v3"
            ).get("algorithms", [])
            if row.get("trainable") and int(row.get("claim_eligible_runs") or 0) > 0
        ]
        for record in registry_rows:
            if record.get("algorithm") in {"mpc", "fcfs"}:
                continue
            if algorithm and record.get("algorithm") != algorithm:
                continue
            job_id = str(record.get("job_id") or "")
            history = TRAINING_MANAGER.history(job_id, limit=200).get("records", []) if job_id else []
            if history:
                selected_status = TRAINING_MANAGER.status(job_id)
                selected_history = history
                break
        if not selected_status and not algorithm:
            selected_status = TRAINING_MANAGER.status()
        return {
            "available": bool(selected_status.get("job_id")),
            "optimizer_history_available": bool(selected_history),
            "job": selected_status,
            "history": selected_history,
            "requested_algorithm": algorithm,
            "available_algorithms": available_algorithms,
            "selection_basis": (
                "latest_registered_run_with_persisted_optimizer_history_for_requested_algorithm"
                if algorithm
                else "latest_registered_trainable_run_with_persisted_optimizer_history"
            ),
            "_source": "persisted_training_callback_metrics",
        }

    # ---------- 实验 ----------
    def experiments(self) -> Dict[str, Any]:
        benchmark = TRAINING_MANAGER.baselines()
        return {"items": benchmark.get("baselines", []), "updated_at": benchmark.get("updated_at"), "_source": benchmark.get("source")}

    def rollback(self, id_: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "executed": False,
            "id": id_,
            "status": "pending_port_connection",
            "reason": "待接入港口：未配置生产模型注册表或部署适配器；本次未执行回滚。",
        }

    # ---------- 因果 ----------
    def causal_estimate(self, metric: str, segment: Optional[str]) -> Dict[str, Any]:
        return {
            "available": False,
            "metric": metric,
            "segment": segment,
            "status": "pending_port_connection",
            "reason": "待接入港口：因果估计需要处理组/对照组、结果回流、倾向重叠与干扰诊断；当前不会生成替代 ATE/CATE。",
        }
