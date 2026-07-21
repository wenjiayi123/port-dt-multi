# ============================================
# app/services/closed_loop.py
# --------------------------------------------
# 执行与闭环（Execution & Closed-Loop）服务
#
# 能力：
#  1) 一键下发 / 半自动审批
#     - submit(strategy, operator, mode='auto'|'manual', dry_run=False)
#     - approve(job_id, operator)
#     - get(job_id), list(limit)
#  2) A/B 对照：预测与现场观测严格分离。
#     未注入 collect_ab_observations 适配器时，只返回预测并标记实测不可用。
#  3) 在线学习器（轻量）
#     - learn(job_id)：用 A/B 结果对“策略效果”做 EMA 更新
#     - get_model(strategy_id)：查询累计效果、偏差、可靠度
#
# 依赖（通过 DI 注入）：
#  - rlpanel：提供 simulate(strategy) 得到 baseline/simulated 聚合曲线（提交时快照保存）
#  - dispatch：守护栏校验、演示下发
#  - telemetry / forecast / reporting：可选用于将来真实 A/B（此处只做演示回落）
#
# 持久化：
#  - 轻量把在线学习器状态落在仓库运行态目录（若不可写则仅在内存）
#
# ============================================

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return d
    return v


@dataclass
class ExecJob:
    """执行工单结构（内存态 + 可序列化）"""
    job_id: str
    created_at: str
    updated_at: str
    status: str                      # PENDING_APPROVAL | APPROVED | SENT | FAILED | CANCELLED
    operator: str
    mode: str                        # 'auto' | 'manual'
    dry_run: bool
    strategy_id: str
    strategy: Dict[str, Any]

    # 提交时保存的预测快照（用于 A/B）
    forecast_snapshot: Dict[str, Any]     # {"baseline":{"agg_kW":[...],"total_kWh":...}, "simulated":{...}, "summary":{...}}

    # 发送结果/备注
    send_result: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ClosedLoopService:
    """
    执行与闭环服务：
      - 对接 dispatch（守护栏/演示下发）并抽象成“一键下发/半自动审批”的流程；
      - 在提交时把“预测快照”保存下来（baseline/simulated 聚合曲线和汇总 kWh）；
      - 提供 A/B 对照（将来接 EMS/SCADA 时替换 _collect_actual 实现）；
      - 提供一个极简在线学习器（EMA）。
    """

    def __init__(self, rlpanel, dispatch, telemetry=None, forecast=None, reporting=None, persist_path: Optional[str] = None):
        self.rlpanel = rlpanel
        self.dispatch = dispatch
        self.telemetry = telemetry
        self.forecast = forecast
        self.reporting = reporting

        self._jobs: Dict[str, ExecJob] = {}
        default_model_path = Path(__file__).resolve().parents[3] / "data" / "objects" / "exec" / "rl_online_model.json"
        self._model_path = persist_path or str(default_model_path)
        self._model: Dict[str, Dict[str, float]] = self._load_model()  # {strategy_id: {"n":count,"ema_delta":...,"ema_bias":...,"ema_abs_err":...}}

    # -------------------------
    # 提交 / 审批 / 查询
    # -------------------------
    def submit(self, strategy: Dict[str, Any], operator: str = "system", mode: str = "auto", dry_run: bool = False, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        创建执行工单：
          - 先用 rlpanel.simulate(strategy) 拿“预测快照”（baseline/simulated/summary）
          - 走 dispatch.validate / dispatch.dispatch（演示）
          - mode='auto'：自动设为已批准并直接发送
          - mode='manual'：进入待审批态
        """
        if not isinstance(strategy, dict) or not strategy.get("id"):
            return {"ok": False, "error": "strategy 缺失或缺少 id"}

        # 1) 预测快照
        snap = self.rlpanel.simulate(strategy=strategy, horizon_min=360, step_min=1) or {}
        # 2) 守护栏校验（使用 dispatch 的校验与估算）
        guard = self.dispatch.validate_strategy(strategy)
        if not guard.get("ok", False):
            return {"ok": False, "error": f"策略校验未通过：{guard.get('errors')}"}

        job = ExecJob(
            job_id=str(uuid.uuid4()),
            created_at=_now_iso(),
            updated_at=_now_iso(),
            status="PENDING_APPROVAL" if mode == "manual" else "APPROVED",
            operator=str(operator or "system"),
            mode=mode,
            dry_run=bool(dry_run),
            strategy_id=str(strategy["id"]),
            strategy=strategy,
            forecast_snapshot={
                "baseline": {"agg_kW": snap.get("baseline", {}).get("agg_kW", []), "total_kWh": _safe_float(snap.get("baseline", {}).get("total_kWh"), 0.0)},
                "simulated": {"agg_kW": snap.get("simulated", {}).get("agg_kW", []), "total_kWh": _safe_float(snap.get("simulated", {}).get("total_kWh"), 0.0)},
                "summary": snap.get("summary", {}),
            },
            notes=notes
        )
        self._jobs[job.job_id] = job

        # 自动模式：直接发送
        if mode != "manual":
            self._send_job(job)

        return {"ok": True, "job": asdict(job)}

    def approve(self, job_id: str, operator: str = "system") -> Dict[str, Any]:
        """
        审批通过并发送（仅适用于 PENDING_APPROVAL）。
        """
        job = self._jobs.get(job_id)
        if not job:
            return {"ok": False, "error": "job 不存在"}
        if job.status != "PENDING_APPROVAL":
            return {"ok": False, "error": f"状态不允许审批：{job.status}"}
        job.status = "APPROVED"
        job.updated_at = _now_iso()
        job.operator = operator or job.operator
        self._send_job(job)
        return {"ok": True, "job": asdict(job)}

    def get(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"ok": False, "error": "job 不存在"}
        return {"ok": True, "job": asdict(job)}

    def list(self, limit: int = 50) -> Dict[str, Any]:
        items = list(self._jobs.values())
        items.sort(key=lambda j: j.created_at, reverse=True)
        return {"ok": True, "items": [asdict(x) for x in items[:max(1, int(limit))]]}

    # -------------------------
    # 发送（演示对接）
    # -------------------------
    def _send_job(self, job: ExecJob) -> None:
        """
        对接 dispatch.dispatch（演示/干跑）：
          - 这里不直连 EMS/SCADA，只做“编排 + 守护栏 + 记录”。
          - 将来接入真实系统时，可在此处分发到不同 connector。
        """
        try:
            res = self.dispatch.dispatch(
                strategy=job.strategy,
                operator=job.operator,
                dry_run=job.dry_run,
                enforce_guardrails=True,
                guardrail_min_peak_kw=1.0,
                notes="closed_loop.send"
            )
            job.send_result = res
            result_status = str((res or {}).get("status") or "")
            job.status = "DRY_RUN_RECORDED" if "DRY_RUN" in result_status else (
                "REJECTED" if result_status == "REJECTED" else "NO_PRODUCTION_ACTUATOR"
            )
            job.updated_at = _now_iso()
        except Exception as e:
            job.send_result = {"error": str(e)}
            job.status = "FAILED"
            job.updated_at = _now_iso()

    # -------------------------
    # A/B 对照：实际 vs 预测
    # -------------------------
    def ab_compare(self, job_id: str) -> Dict[str, Any]:
        """
        返回该 job 的 A/B 对照结果：
          - predicted（来自 submit 时保存的 forecast_snapshot）
          - actual（仅来自显式注入的现场观测适配器）
          - 差异指标与误差
        """
        job = self._jobs.get(job_id)
        if not job:
            return {"ok": False, "error": "job 不存在"}

        pred_base = job.forecast_snapshot.get("baseline", {})
        pred_sim  = job.forecast_snapshot.get("simulated", {})
        base_curve = list(map(_safe_float, pred_base.get("agg_kW", []) or []))
        sim_curve  = list(map(_safe_float, pred_sim.get("agg_kW", []) or []))

        # 预测汇总
        pred = {
            "baseline_kWh": _safe_float(pred_base.get("total_kWh"), 0.0),
            "simulated_kWh": _safe_float(pred_sim.get("total_kWh"), 0.0),
            "delta_kWh_pred": _safe_float(job.forecast_snapshot.get("summary", {}).get("delta_kWh"), 0.0),
        }

        actual_curves = self._collect_actual(job, base_curve, sim_curve)
        if actual_curves is None:
            return {
                "ok": True,
                "available": False,
                "job_id": job_id,
                "pred": pred,
                "actual": None,
                "error": None,
                "forecast_snapshot": job.forecast_snapshot,
                "reason": "No measured A/B observation adapter is configured",
                "_source": "prediction_only_no_measured_observation",
            }
        actual = {
            "baseline_kWh_obs": actual_curves["baseline_kWh"],
            "simulated_kWh_obs": actual_curves["simulated_kWh"],
            "delta_kWh_obs": actual_curves["delta_kWh_obs"],
        }

        # 误差/相对误差
        err = {
            "delta_kWh_error": actual["delta_kWh_obs"] - pred["delta_kWh_pred"],
            "abs_error": abs(actual["delta_kWh_obs"] - pred["delta_kWh_pred"]),
        }
        rel = pred["delta_kWh_pred"]
        err["rel_error"] = (err["delta_kWh_error"] / rel) if abs(rel) > 1e-6 else None

        actual["baseline_curve"] = actual_curves.get("baseline_curve", [])
        actual["simulated_curve"] = actual_curves.get("simulated_curve", [])
        return {
            "ok": True,
            "available": True,
            "job_id": job_id,
            "pred": pred,
            "actual": actual,
            "error": err,
            "forecast_snapshot": job.forecast_snapshot,
            "_source": actual_curves.get("_source", "measured_observation_adapter"),
        }

    def _collect_actual(self, job: ExecJob, base_curve: List[float], sim_curve: List[float]) -> Optional[Dict[str, Any]]:
        """
        现场 telemetry 适配器可实现
        ``collect_ab_observations(job_id, strategy, predicted)``。
        结果必须包含 baseline_kWh、simulated_kWh 与可选实测曲线。
        """
        collector = getattr(self.telemetry, "collect_ab_observations", None)
        if not callable(collector):
            return None
        result = collector(
            job_id=job.job_id,
            strategy=job.strategy,
            predicted={"baseline_curve": base_curve, "simulated_curve": sim_curve},
        )
        if not isinstance(result, dict):
            raise ValueError("collect_ab_observations must return a mapping")
        base_kwh = _safe_float(result.get("baseline_kWh"), float("nan"))
        sim_kwh = _safe_float(result.get("simulated_kWh"), float("nan"))
        if not math.isfinite(base_kwh) or not math.isfinite(sim_kwh):
            raise ValueError("measured A/B observations must contain finite baseline_kWh and simulated_kWh")
        return {
            "baseline_kWh": base_kwh,
            "simulated_kWh": sim_kwh,
            "delta_kWh_obs": sim_kwh - base_kwh,
            "baseline_curve": list(result.get("baseline_curve") or []),
            "simulated_curve": list(result.get("simulated_curve") or []),
            "_source": result.get("_source") or "measured_observation_adapter",
        }

    # -------------------------
    # 在线学习器（极简 EMA）
    # -------------------------
    def learn(self, job_id: str, alpha: float = 0.3) -> Dict[str, Any]:
        """
        基于 ab_compare 的 delta_kWh_obs 和 delta_kWh_pred 更新在线指标：
          - ema_delta：观测到的“策略带来的 ΔkWh”（负值代表节电）
          - ema_bias：观测-预测 的偏差（负值=>实际比预测更省电）
          - ema_abs_err：|观测-预测| 的 EMA（衡量稳定性）
          - n：样本数
        """
        job = self._jobs.get(job_id)
        if not job:
            return {"ok": False, "error": "job 不存在"}

        ab = self.ab_compare(job_id)
        if not ab.get("ok"):
            return {"ok": False, "error": "A/B 对照失败"}
        if not ab.get("available") or not ab.get("actual"):
            return {"ok": False, "error": "未配置实测 A/B 观测适配器，禁止用预测合成值更新在线模型"}

        delta_obs = float(ab["actual"]["delta_kWh_obs"])
        delta_pred = float(ab["pred"]["delta_kWh_pred"])
        bias = delta_obs - delta_pred
        abs_err = abs(bias)

        sid = job.strategy_id
        rec = self._model.get(sid, {"n": 0.0, "ema_delta": 0.0, "ema_bias": 0.0, "ema_abs_err": 0.0})
        n0 = float(rec.get("n", 0.0))

        def ema(old: float, x: float) -> float:
            return (1.0 - alpha) * float(old) + alpha * float(x)

        rec["n"] = n0 + 1.0
        rec["ema_delta"] = ema(rec.get("ema_delta", 0.0), delta_obs)
        rec["ema_bias"] = ema(rec.get("ema_bias", 0.0), bias)
        rec["ema_abs_err"] = ema(rec.get("ema_abs_err", 0.0), abs_err)

        self._model[sid] = rec
        self._save_model()

        return {"ok": True, "strategy_id": sid, "model": rec, "ab": ab}

    def get_model(self, strategy_id: str) -> Dict[str, Any]:
        rec = self._model.get(strategy_id)
        if not rec:
            return {"ok": False, "error": "暂无该策略的在线学习记录"}
        return {"ok": True, "strategy_id": strategy_id, "model": rec}

    # -------------------------
    # 轻量持久化
    # -------------------------
    def _load_model(self) -> Dict[str, Dict[str, float]]:
        path = self._model_path
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _save_model(self) -> None:
        path = self._model_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._model, f, ensure_ascii=False, indent=2)
        except Exception:
            # 不可写则忽略，维持内存态
            pass
