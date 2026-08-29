"""Evidence-bound Xiaoyi mission context and frontline handoff packets.

The language model is never the source of telemetry, policy metrics or control
authority.  This service first builds one compact, hash-addressed operational
context from the active twin/runtime services.  Xiaoyi may explain that packet;
all executable navigation stays in a deterministic allow-list and production
control remains disabled.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
VALUE_REGISTRY = ROOT / "evidence/v3/value_improvement_v32.json"
HANDOFF_LOG = ROOT / "data/runtime/xiaoyi_handoffs.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


class XiaoyiMissionControl:
    """Build the only operational context Xiaoyi is allowed to describe."""

    MISSION_MODES: tuple[dict[str, str], ...] = (
        {
            "id": "situation",
            "label": "当前态势",
            "description": "读取当前回放、预测、数据质量、模型和安全门。",
            "route": "/?xiaoyi_focus=situation#twin3d-section",
        },
        {
            "id": "forecast",
            "label": "未来风险",
            "description": "解释P10/P50/P90、峰值概率与缺失的现场校准。",
            "route": "/?xiaoyi_focus=forecast#twin3d-section",
        },
        {
            "id": "strategy",
            "label": "策略解释",
            "description": "说明真实模型输出、相对基线变化及安全投影。",
            "route": "/?xiaoyi_focus=strategy#strategy-exec-module",
        },
        {
            "id": "triage",
            "label": "告警分诊",
            "description": "联动异常、漂移和准入门，给出检查与回退顺序。",
            "route": "/?xiaoyi_focus=triage#monitoring-center",
        },
        {
            "id": "handoff",
            "label": "交接班",
            "description": "生成带上下文哈希、未决风险和版本号的交接预览。",
            "route": "/ops-copilot?mission=handoff",
        },
        {
            "id": "dry_run",
            "label": "执行预演",
            "description": "只准备dry-run和人工审批入口，不下发生产指令。",
            "route": "/?xiaoyi_focus=dry_run#strategy-exec-module",
        },
    )

    def __init__(self, realtime: Any, monitoring: Any, runtime: Any) -> None:
        self.realtime = realtime
        self.monitoring = monitoring
        self.runtime = runtime

    @classmethod
    def mission_modes(cls) -> list[dict[str, str]]:
        return [dict(row) for row in cls.MISSION_MODES]

    @classmethod
    def _mission(cls, mission_id: str) -> dict[str, str]:
        return next(
            (dict(row) for row in cls.MISSION_MODES if row["id"] == mission_id),
            dict(cls.MISSION_MODES[0]),
        )

    def build_context(
        self,
        *,
        asset_id: str = "qc-01",
        mission_id: str = "situation",
        cap_kw: float = 36_000.0,
        horizon_min: int = 60,
        step_min: int = 5,
    ) -> dict[str, Any]:
        mission = self._mission(mission_id)
        realtime = self.realtime.build(
            asset_id=asset_id,
            mode="sim",
            cap_kw=cap_kw,
            horizon_min=horizon_min,
            step_min=step_min,
        )
        monitoring = self.monitoring.build()
        runtime_status = self.runtime.status()
        strategy = self.runtime.series(
            horizon_min=horizon_min,
            step_min=step_min,
            scenario="strategy",
        )
        value_registry = _load_json(VALUE_REGISTRY)

        telemetry = realtime.get("telemetry") or {}
        quality = realtime.get("quality") or {}
        forecast = realtime.get("forecast") or {}
        peak_risk = realtime.get("peak_risk") or {}
        business = realtime.get("business_value") or {}
        current_analysis = monitoring.get("current_analysis") or {}
        anomaly = current_analysis.get("anomaly") or {}
        drift = current_analysis.get("drift") or {}
        admission = current_analysis.get("admission_decision") or {}
        model = runtime_status.get("model") or {}
        strategy_summary = strategy.get("summary") or {}
        projection = strategy_summary.get("business_projection") or {}

        forecast_peak = forecast.get("peak") or {}
        peak_probability = _finite(peak_risk.get("peak_probability"))
        drift_psi = _finite(drift.get("psi"))
        source_mode = str(telemetry.get("mode") or "unavailable")
        measured = bool(telemetry.get("measured"))
        production = bool(telemetry.get("production"))
        policy_allowed = bool(admission.get("new_policy_suggestions_allowed"))

        if not realtime.get("available"):
            overall = "data_unavailable"
        elif str(admission.get("decision")) == "block_to_safe_baseline":
            overall = "review_required_safe_baseline"
        elif peak_probability is not None and peak_probability >= 0.5:
            overall = "forecast_risk_review"
        else:
            overall = "offline_replay_stable"

        signals = [
            {
                "id": "telemetry_source",
                "name": "当前数据源",
                "value": "现场遥测" if measured and production else "公开校准连续回放",
                "level": "ok" if measured and production else "watch",
                "source": "/api/v3/realtime/insights.telemetry",
                "available": bool(realtime.get("available")),
            },
            {
                "id": "forecast_peak",
                "name": f"{horizon_min}分钟P50峰值",
                "value": (
                    f"{float(forecast_peak.get('p50') or forecast_peak.get('kW')):,.0f} kW"
                    if forecast_peak.get("p50") is not None or forecast_peak.get("kW") is not None
                    else "待接入港口"
                ),
                "level": "watch" if forecast.get("available") else "pending",
                "source": str(forecast.get("model") or "/api/v3/realtime/insights.forecast"),
                "available": bool(forecast.get("available")),
            },
            {
                "id": "peak_probability",
                "name": "工程阈值越限概率",
                "value": f"{peak_probability * 100:.1f}%" if peak_probability is not None else "待接入港口",
                "level": "major" if peak_probability is not None and peak_probability >= 0.5 else "ok",
                "source": str(peak_risk.get("logic") or "/api/v3/realtime/insights.peak_risk"),
                "available": peak_probability is not None,
            },
            {
                "id": "monitoring_gate",
                "name": "策略准入门",
                "value": str(admission.get("decision") or "unavailable"),
                "level": "major" if not policy_allowed else "ok",
                "source": "/api/v3/monitoring/evidence.current_analysis.admission_decision",
                "available": bool(admission),
            },
            {
                "id": "anomaly_count",
                "name": "回放异常点",
                "value": str(anomaly.get("anomaly_count")) if anomaly.get("anomaly_count") is not None else "待接入港口",
                "level": "major" if int(anomaly.get("anomaly_count") or 0) > 0 else "ok",
                "source": "/api/v3/monitoring/evidence.current_analysis.anomaly",
                "available": anomaly.get("anomaly_count") is not None,
            },
            {
                "id": "drift",
                "name": "回放分布漂移PSI",
                "value": f"{drift_psi:.3f}" if drift_psi is not None else "待接入港口",
                "level": "major" if str(drift.get("level")) == "drift" else ("watch" if drift.get("level") == "warn" else "ok"),
                "source": "/api/v3/monitoring/evidence.current_analysis.drift",
                "available": drift_psi is not None,
            },
            {
                "id": "policy",
                "name": "当前策略模型",
                "value": str(model.get("algorithm") or "unavailable").upper(),
                "level": "ok" if runtime_status.get("available") else "pending",
                "source": "/api/v3/runtime/status.model",
                "available": bool(runtime_status.get("available")),
            },
            {
                "id": "execution_authority",
                "name": "生产控制权限",
                "value": "已授权" if runtime_status.get("production_authority") else "无生产控制权",
                "level": "major" if runtime_status.get("production_authority") else "ok",
                "source": "/api/v3/runtime/status.production_authority",
                "available": True,
            },
        ]

        missing_site_factors = [
            "合同需量与结算电价",
            "TOS/VTS作业与船期",
            "设备PLC/BMS/EMS回读",
            "现场预测实绩与校准窗口",
            "告警确认、工单和执行回执",
        ]
        if measured and production:
            missing_site_factors = []

        context_core: dict[str, Any] = {
            "schema": "port-dt-xiaoyi-mission-context.v1",
            "generated_at": _utc_now(),
            "mission": mission,
            "asset_id": asset_id,
            "overall_state": overall,
            "source": {
                "mode": source_mode,
                "artifact_id": telemetry.get("artifact_id"),
                "sha256": telemetry.get("sha256"),
                "sample_count": telemetry.get("sample_count"),
                "latest_at": telemetry.get("latest_at"),
                "measured": measured,
                "production": production,
            },
            "data_quality": {
                "available": quality.get("available"),
                "missing_rate": quality.get("missing_rate"),
                "stale_rate": quality.get("stale_rate"),
                "out_of_engineering_range_rate": quality.get("out_of_engineering_range_rate"),
                "site_sensor_quality": quality.get("site_sensor_quality"),
            },
            "forecast": {
                "available": forecast.get("available"),
                "model": forecast.get("model"),
                "horizon_min": horizon_min,
                "peak_at": forecast_peak.get("ts"),
                "peak_p50_kw": _finite(forecast_peak.get("p50") or forecast_peak.get("kW")),
                "peak_p10_kw": _finite(forecast_peak.get("p10")),
                "peak_p90_kw": _finite(forecast_peak.get("p90")),
                "peak_probability": peak_probability,
                "engineering_cap_kw": _finite(peak_risk.get("cap_kw")),
                "site_calibration_available": bool((forecast.get("calibration") or {}).get("available")),
            },
            "monitoring": {
                "anomaly_count": anomaly.get("anomaly_count"),
                "drift_psi": drift_psi,
                "drift_level": drift.get("level"),
                "admission_decision": admission.get("decision"),
                "new_policy_suggestions_allowed": policy_allowed,
                "site_command_allowed": bool(admission.get("site_command_allowed")),
                "fallback": admission.get("fallback"),
                "requires_human_review": bool(admission.get("requires_human_review")),
            },
            "policy": {
                "available": bool(runtime_status.get("available")),
                "algorithm": model.get("algorithm"),
                "implementation": model.get("implementation"),
                "job_id": model.get("job_id"),
                "model_sha256": model.get("model_sha256"),
                "dataset_id": model.get("dataset_id"),
                "dataset_sha256": model.get("dataset_sha256"),
                "inference": runtime_status.get("inference"),
                "hard_guardrail_passed": strategy_summary.get("hard_guardrail_passed"),
                "production_authority": bool(runtime_status.get("production_authority")),
            },
            "strategy_projection": {
                "available": bool(strategy.get("available") and projection),
                "improvement_percent": projection.get("improvement_percent") or {},
                "equivalent_throughput_value": projection.get("equivalent_throughput_value") or {},
                "claim_boundary": projection.get("claim_boundary"),
            },
            "business_value": {
                "available": business.get("available"),
                "avoided_energy_cost_cny": _finite(business.get("avoided_energy_cost_cny")),
                "avoided_carbon_kg": _finite(business.get("avoided_carbon_kg")),
                "financial_audit_ready": bool(business.get("financial_audit_ready")),
            },
            "module_value_decisions": {
                key: {
                    "status": row.get("status"),
                    "status_cn": row.get("status_cn"),
                }
                for key, row in (value_registry.get("modules") or {}).items()
                if isinstance(row, dict)
            },
            "missing_site_factors": missing_site_factors,
            "signals": signals,
            "events": list(realtime.get("events") or []),
            "claim_boundary": (
                "小懿只解释当前后端返回的公开数据回放、预测和模型输出；"
                "不得将其表述为现场告警、集团收益或生产指令。"
            ),
            "production_authority": False,
        }
        context_core["context_sha256"] = _sha256_json(context_core)
        return context_core

    @staticmethod
    def local_fallback_answer(context: dict[str, Any], query: str) -> str:
        forecast = context.get("forecast") or {}
        monitoring = context.get("monitoring") or {}
        policy = context.get("policy") or {}
        peak = forecast.get("peak_p50_kw")
        probability = forecast.get("peak_probability")
        peak_text = f"{float(peak):,.0f} kW" if peak is not None else "暂无可用预测"
        risk_text = f"{float(probability) * 100:.1f}%" if probability is not None else "未计算"
        gate = str(monitoring.get("admission_decision") or "unavailable")
        gate_text = {
            "block_to_safe_baseline": "阻断并回退安全基线",
            "review": "需要人工复核",
            "pass_offline_analysis_only": "仅通过离线分析门",
            "unavailable": "当前不可用",
        }.get(gate, gate)
        source_mode = str((context.get("source") or {}).get("mode") or "unavailable")
        source_text = {
            "calibrated_public_replay_simulator": "公开数据校准连续回放",
            "public_data_calibrated_replay": "公开数据校准回放",
            "unavailable": "当前不可用",
        }.get(source_mode, source_mode)
        fallback = str(monitoring.get("fallback") or "保持上一稳定策略或 FCFS/MPC 安全基线")
        if not fallback.startswith("保持"):
            fallback = "保持" + fallback
        return (
            f"当前结论\n针对“{query}”，当前策略不应直接进入生产。数据是"
            f"{source_text}，不是现场实测；准入门为“{gate_text}”。\n\n"
            f"后端依据\n未来窗口P50峰值 {peak_text}，越限概率 {risk_text}；"
            f"回放漂移 PSI={monitoring.get('drift_psi')}；模型 {str(policy.get('algorithm') or 'unavailable').upper()}，"
            f"硬约束={'通过' if policy.get('hard_guardrail_passed') else '未通过或不可用'}，当前无生产控制权。\n\n"
            f"建议检查\n1. {fallback}。\n2. 核对合同需量与现场预测实绩。\n"
            "3. 接入TOS/VTS、PLC/BMS/EMS回读及告警工单回执。\n\n"
            "可预演动作\n仅允许固定输入的dry-run、留出集复核和安全基线对比。\n\n"
            "必须人工确认\n现场负责人确认数据新鲜度、约束参数、回退方案和执行权限后，才能进入下一准入阶段。"
        )

    def recommended_actions(self, context: dict[str, Any], mission_id: str) -> list[dict[str, Any]]:
        monitoring = context.get("monitoring") or {}
        risk_blocked = monitoring.get("new_policy_suggestions_allowed") is False
        actions = [
            {
                "id": "review_twin_now",
                "priority": "P0",
                "label": "查看当前孪生态势",
                "reason": "核对当前输入、预测时窗和数据来源标签。",
                "href": "/?xiaoyi_focus=situation#twin3d-section",
                "execution": "navigation_only",
                "human_confirmation": False,
            },
            {
                "id": "review_forecast",
                "priority": "P1",
                "label": "查看未来六小时预测",
                "reason": "复核P10/P50/P90与工程阈值，现场合同需量仍待替换。",
                "href": "/?xiaoyi_focus=forecast#twin3d-section",
                "execution": "navigation_only",
                "human_confirmation": False,
            },
            {
                "id": "review_policy",
                "priority": "P1",
                "label": "解释当前策略",
                "reason": "查看模型哈希、相对基线变化、硬约束和声明边界。",
                "href": "/?xiaoyi_focus=strategy#strategy-exec-module",
                "execution": "navigation_only",
                "human_confirmation": False,
            },
            {
                "id": "review_monitoring",
                "priority": "P0" if risk_blocked else "P2",
                "label": "进入监控分诊",
                "reason": (
                    "当前漂移门已阻止新策略建议，应先复核再回到安全基线。"
                    if risk_blocked
                    else "检查异常、漂移和质量门，避免把季节性误判为事故。"
                ),
                "href": "/?xiaoyi_focus=triage#monitoring-center",
                "execution": "navigation_only",
                "human_confirmation": False,
            },
            {
                "id": "prepare_handoff",
                "priority": "P2",
                "label": "生成交接班预览",
                "reason": "携带上下文哈希、模型/数据版本、未决风险和缺失现场字段。",
                "href": "/ops-copilot?mission=handoff",
                "execution": "preview_only",
                "human_confirmation": False,
            },
            {
                "id": "prepare_dry_run",
                "priority": "BLOCKED" if risk_blocked else "P2",
                "label": "准备策略dry-run",
                "reason": (
                    "当前准入门为阻断，必须先处理漂移或保持FCFS/MPC安全基线。"
                    if risk_blocked
                    else "只进入仿真和人工审批，不执行现场设备指令。"
                ),
                "href": "/?xiaoyi_focus=dry_run#strategy-exec-module",
                "execution": "blocked_by_monitoring_gate" if risk_blocked else "dry_run_only",
                "human_confirmation": True,
            },
        ]
        if mission_id == "handoff":
            actions.sort(key=lambda row: row["id"] != "prepare_handoff")
        elif mission_id == "triage":
            actions.sort(key=lambda row: row["id"] != "review_monitoring")
        elif mission_id == "strategy":
            actions.sort(key=lambda row: row["id"] != "review_policy")
        return actions

    @staticmethod
    def handoff_preview(
        context: dict[str, Any],
        *,
        answer: str,
        operator: str,
        shift: str,
    ) -> dict[str, Any]:
        packet: dict[str, Any] = {
            "schema": "port-dt-xiaoyi-shift-handoff.v1",
            "generated_at": _utc_now(),
            "operator": operator or "未填写",
            "shift": shift or "当前班次",
            "context_sha256": context.get("context_sha256"),
            "overall_state": context.get("overall_state"),
            "source": context.get("source"),
            "forecast": context.get("forecast"),
            "monitoring": context.get("monitoring"),
            "policy": context.get("policy"),
            "missing_site_factors": context.get("missing_site_factors"),
            "xiaoyi_summary": answer,
            "production_authority": False,
            "requires_human_confirmation": True,
            "claim_boundary": context.get("claim_boundary"),
        }
        packet["handoff_sha256"] = _sha256_json(packet)
        return packet

    @staticmethod
    def persist_handoff(packet: dict[str, Any], *, confirm: bool) -> dict[str, Any]:
        if not confirm:
            return {
                "persisted": False,
                "status": "confirmation_required",
                "packet": packet,
            }
        HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
        with HANDOFF_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
        return {
            "persisted": True,
            "status": "recorded",
            "handoff_sha256": packet.get("handoff_sha256"),
            "audit_artifact": "data/runtime/xiaoyi_handoffs.jsonl",
            "production_action_executed": False,
        }
