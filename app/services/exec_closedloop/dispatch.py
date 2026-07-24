# ============================================
# app/services/exec_closedloop/dispatch.py
# --------------------------------------------
# “策略下发（演示）”服务 · 更接近真实落地版
#
# 目标：
# 1) 在原有 dry-run 基础上，补全更真实的守护栏口径：
#    - 策略字段校验
#    - 仿真结果校验
#    - dispatch_ready 校验
#    - 风险标记归并
# 2) 返回更完整的下发记录：
#    - impact_summary
#    - readiness
#    - top_contributors
#    - evidence
# 3) 历史记录更像“上线前控制层”的审计记录，而不是只有 job_id/status。
# 4) 新增“数据源适配口径”：
#    - 不急着新建很多散落 mock 文件
#    - 先让每个执行工单都携带 source_profile / data_contract_ready
#    - 未来真实落地时，只需要把 source_mode 从 mock 切到 real，
#      或把 adapter_target / contract_name 对上真实接口，不必重改上层 UI 与闭环逻辑。
# ============================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import math
import uuid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return d


def _safe_int(x: Any, d: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return d


@dataclass
class DispatchRecord:
    job_id: str
    created_at: str
    operator: str
    dry_run: bool
    status: str
    strategy_id: str
    strategy_title: str
    strategy: Dict[str, Any]
    validation: Dict[str, Any]
    guardrails: Dict[str, Any]
    estimate: Dict[str, Any]
    readiness: Dict[str, Any]
    contributors: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    notes: Optional[str] = None


class DispatchService:
    """
    用法（由 DI 注入）：
        dispatch = DispatchService(telemetry=..., rlpanel=..., twin=...)
        rec = dispatch.dispatch(strategy, operator="admin", dry_run=True)
        hist = dispatch.list_history()
    """

    def __init__(self, telemetry, rlpanel, twin=None):
        self.telemetry = telemetry
        self.rlpanel = rlpanel
        self.twin = twin
        self._history: List[DispatchRecord] = []

        try:
            self._assets = {a["id"]: a for a in (self.telemetry.list_assets() or [])}
        except Exception:
            self._assets = {}

    # -----------------------------
    # 0) 数据源适配口径（新增）
    # -----------------------------
    def _infer_source_profile(self, strategy: Dict[str, Any], estimate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        给每个策略/工单补一份“数据源适配画像”，目的不是现在就接真实库，
        而是把未来真实落地时会替换的接口位、契约名、主键字段先稳定下来。

        兼容两类来源：
        1) 策略里主动声明：source_profile / data_contract / source_mode / adapter_target
        2) 没声明时：按策略动作和资产类型给出 mock-ready 默认画像
        """
        strategy = strategy or {}
        sp = dict(strategy.get("source_profile") or {})
        dc = dict(strategy.get("data_contract") or {})
        actions = strategy.get("actions") or []
        scope = strategy.get("scope") or {}
        assets = list(scope.get("asset_ids") or [])

        declared_mode = str(
            sp.get("source_mode")
            or strategy.get("source_mode")
            or dc.get("source_mode")
            or "mock"
        ).strip().lower() or "mock"

        contract_name = str(
            sp.get("contract_name")
            or dc.get("name")
            or strategy.get("contract_name")
            or "exec_closedloop.v1"
        ).strip() or "exec_closedloop.v1"

        contract_version = str(
            sp.get("contract_version")
            or dc.get("version")
            or strategy.get("contract_version")
            or "v1"
        ).strip() or "v1"

        primary_key = str(
            sp.get("primary_key")
            or dc.get("primary_key")
            or "asset_id"
        ).strip() or "asset_id"

        timestamp_key = str(
            sp.get("timestamp_key")
            or dc.get("timestamp_key")
            or "ts"
        ).strip() or "ts"

        adapter_target = str(
            sp.get("adapter_target")
            or strategy.get("adapter_target")
            or dc.get("adapter_target")
            or "mock://exec_closedloop"
        ).strip() or "mock://exec_closedloop"

        observed_fields = list(sp.get("observed_fields") or dc.get("observed_fields") or [])
        control_fields = list(sp.get("control_fields") or dc.get("control_fields") or [])
        if not observed_fields:
            observed_fields = ["asset_id", "ts", "kW", "status"]
        if not control_fields:
            control_fields = ["asset_id", "action", "setpoint", "operator", "job_id"]

        inferred_asset_family = "generic"
        inferred_scene = "general_dispatch"
        inferred_target = "dispatch"
        if actions:
            cmd0 = str((actions[0] or {}).get("cmd") or "").strip().lower()
            if cmd0 in ("charge", "discharge"):
                inferred_asset_family = "energy_asset"
                inferred_scene = "energy_dispatch"
                inferred_target = "ems_or_bess"
            elif cmd0 in ("lighting_dim", "open", "close"):
                inferred_asset_family = "lighting_asset"
                inferred_scene = "yard_lighting"
                inferred_target = "lighting_plc"
            elif cmd0 in ("shore_power", "setpoint"):
                inferred_asset_family = "shore_power_asset"
                inferred_scene = "shore_power"
                inferred_target = "shore_power_gateway"
            elif cmd0 == "reduce":
                inferred_asset_family = "equipment_asset"
                inferred_scene = "peak_shaving"
                inferred_target = "equipment_dispatch"

        asset_count = len(assets)
        if asset_count <= 0:
            asset_count = _safe_int(((estimate or {}).get("summary") or {}).get("scope_size"), 0)

        has_declared_profile = bool(sp or dc)
        mock_ready = declared_mode in ("mock", "hybrid", "demo")
        real_ready = declared_mode == "real" and not adapter_target.startswith("mock://")
        contract_ready = bool(contract_name and contract_version and primary_key and timestamp_key)

        replace_hint = (
            "后续真实落地时，优先替换 adapter_target / source_mode；"
            "只要字段契约不变，上层 UI、审计和执行闭环无需重写。"
        )

        readiness_label = "real-ready" if real_ready else ("mock-ready" if mock_ready else "adapter-pending")
        summary_text = (
            f"{readiness_label} · {contract_name}:{contract_version} · "
            f"target={adapter_target} · assets={asset_count}"
        )

        return {
            "source_mode": declared_mode,
            "readiness_label": readiness_label,
            "contract_ready": contract_ready,
            "has_declared_profile": has_declared_profile,
            "contract_name": contract_name,
            "contract_version": contract_version,
            "primary_key": primary_key,
            "timestamp_key": timestamp_key,
            "observed_fields": observed_fields,
            "control_fields": control_fields,
            "adapter_target": adapter_target,
            "asset_family": inferred_asset_family,
            "scene": inferred_scene,
            "downstream_target": inferred_target,
            "asset_count": asset_count,
            "mock_ready": mock_ready,
            "real_ready": real_ready,
            "replace_hint": replace_hint,
            "summary_text": summary_text,
        }

    def _compose_visible_note(
        self,
        base_notes: Optional[str],
        source_profile: Dict[str, Any],
        readiness: Dict[str, Any],
    ) -> str:
        parts: List[str] = []
        if base_notes:
            parts.append(str(base_notes).strip())
        parts.append(source_profile.get("summary_text") or "source-profile unavailable")
        if readiness.get("dispatch_ready") is False:
            parts.append("dispatch_ready=false")
        if readiness.get("ok") is False:
            parts.append("guardrails-blocked")
        return " | ".join([p for p in parts if p])

    # -----------------------------
    # 1) 策略校验
    # -----------------------------
    def validate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        errs: List[str] = []
        warn: List[str] = []

        if not isinstance(strategy, dict):
            return {"ok": False, "errors": ["策略必须是对象。"], "warnings": []}

        sid = str(strategy.get("id") or "").strip()
        title = str(strategy.get("title") or "").strip()
        window = strategy.get("window") or {}
        actions = strategy.get("actions") or []
        scope = strategy.get("scope") or {}
        source_mode = str(strategy.get("source_mode") or "mock").strip().lower() or "mock"
        source_profile = strategy.get("source_profile") or {}
        data_contract = strategy.get("data_contract") or {}

        if not sid:
            errs.append("缺少字段：id")
        if not title:
            warn.append("缺少标题 title（建议补充，便于审计）")

        st = window.get("start")
        ed = window.get("end")
        if not st or not ed:
            errs.append("缺少 window.start / window.end")
        else:
            if not ("T" in str(st) and "T" in str(ed)):
                errs.append("window.start / window.end 应为 ISO 时间格式（含 T）")

        if not isinstance(actions, list) or len(actions) == 0:
            errs.append("缺少 actions 列表")
        else:
            for i, act in enumerate(actions):
                if not isinstance(act, dict):
                    errs.append(f"第 {i+1} 个动作不是对象")
                    continue

                cmd = str(act.get("cmd") or "")
                asset = act.get("asset")
                pct = act.get("percent")
                kW_delta = act.get("kW_delta")

                if cmd not in ("idle", "reduce", "charge", "discharge", "lighting_dim", "setpoint", "shore_power"):
                    warn.append(f"第 {i+1} 个动作 cmd 不在常见列表：{cmd}")

                if not asset:
                    warn.append(f"第 {i+1} 个动作缺少 asset")

                if pct is not None and not (0.0 <= _safe_float(pct, -1.0) <= 1.0):
                    warn.append(f"第 {i+1} 个动作 percent 应在 [0,1]")

                if kW_delta is not None and not math.isfinite(_safe_float(kW_delta, float("nan"))):
                    warn.append(f"第 {i+1} 个动作 kW_delta 不是有效数值")

        ids = scope.get("asset_ids")
        if isinstance(ids, list) and ids:
            for rid in ids:
                if self._assets and rid not in self._assets:
                    warn.append(f"scope.asset_ids 包含未知资产：{rid}")

        if source_mode not in ("mock", "real", "hybrid", "demo"):
            warn.append(f"source_mode 建议为 mock/real/hybrid/demo，当前={source_mode}")

        if source_mode == "real" and not (source_profile or data_contract):
            warn.append("source_mode=real，但未声明 source_profile/data_contract；建议补契约后再用于真实落地。")

        if source_profile and not (source_profile.get("adapter_target") or source_profile.get("contract_name")):
            warn.append("source_profile 已声明，但缺少 adapter_target 或 contract_name。")

        if data_contract and not data_contract.get("version"):
            warn.append("data_contract 已声明，但缺少 version。")

        return {"ok": len(errs) == 0, "errors": errs, "warnings": warn}

    # -----------------------------
    # 2) 下发前影响评估
    # -----------------------------
    def estimate_effect(self, strategy: Dict[str, Any], horizon_min: int = 360, step_min: int = 1) -> Dict[str, Any]:
        try:
            sim = self.rlpanel.simulate(strategy=strategy, horizon_min=horizon_min, step_min=step_min) or {}
            summary = sim.get("summary") or {}
            feasibility = sim.get("feasibility") or {}
            contributors = sim.get("contributors") or []
            baseline = sim.get("baseline") or {}
            simulated = sim.get("simulated") or {}

            return {
                "ok": True,
                "summary": {
                    "delta_kWh": float(summary.get("delta_kWh", 0.0)),
                    "delta_carbon_kg": float(summary.get("delta_carbon_kg", 0.0)),
                    "peak_reduction_kW": float(summary.get("peak_reduction_kW", 0.0)),
                    "window": summary.get("window") or {},
                    "scope_size": int(summary.get("scope_size", 0)),
                    "adjusted_asset_count": int(summary.get("adjusted_asset_count", 0)),
                    "dispatch_ready": bool(summary.get("dispatch_ready", False)),
                },
                "feasibility": feasibility,
                "contributors": contributors[:5],
                "baseline": {
                    "total_kWh": float(baseline.get("total_kWh", 0.0)),
                    "peak_kW": float(baseline.get("peak_kW", 0.0)),
                },
                "simulated": {
                    "total_kWh": float(simulated.get("total_kWh", 0.0)),
                    "peak_kW": float(simulated.get("peak_kW", 0.0)),
                },
                "raw": sim,
            }
        except Exception as e:
            return {"ok": False, "error": f"simulate 失败：{e}"}

    # -----------------------------
    # 3) 守护栏
    # -----------------------------
    def _check_guardrails(
        self,
        estimate: Dict[str, Any],
        enforce: bool = True,
        guardrail_min_peak_kw: float = 1.0
    ) -> Dict[str, Any]:
        rules: List[Dict[str, Any]] = []

        if not estimate.get("ok"):
            rules.append({"rule": "have_estimate", "pass": not enforce, "detail": "缺少有效影响评估"})
            return {
                "ok": (not enforce),
                "rules": rules,
                "detail": "未能获取影响评估",
                "risk_flags": ["simulate_failed"],
            }

        s = estimate.get("summary") or {}
        f = estimate.get("feasibility") or {}

        dkwh = _safe_float(s.get("delta_kWh"), 0.0)
        dkg = _safe_float(s.get("delta_carbon_kg"), 0.0)
        pk = _safe_float(s.get("peak_reduction_kW"), 0.0)
        ready = bool(s.get("dispatch_ready", False))
        adjusted_asset_count = int(s.get("adjusted_asset_count", 0))

        r1 = dkwh <= 0.0
        rules.append({"rule": "delta_kWh<=0", "pass": r1, "value": dkwh})

        r2 = pk >= guardrail_min_peak_kw
        rules.append({"rule": f"peak_reduction_kW>={guardrail_min_peak_kw}", "pass": r2, "value": pk})

        r3 = dkg <= 0.0
        rules.append({"rule": "delta_carbon_kg<=0", "pass": r3, "value": dkg})

        r4 = adjusted_asset_count > 0
        rules.append({"rule": "adjusted_asset_count>0", "pass": r4, "value": adjusted_asset_count})

        r5 = ready
        rules.append({"rule": "dispatch_ready", "pass": r5, "value": ready})

        risk_flags = list((f.get("risk_flags") or []))
        if not r4:
            risk_flags.append("未命中有效资产")
        if not r5:
            risk_flags.append("dispatch_ready=false")

        ok = (r1 or r2) and r4 and (r5 or not enforce)

        return {
            "ok": ok,
            "rules": rules,
            "detail": "至少满足节电/削峰之一，且必须命中有效资产；强制模式下还需 dispatch_ready",
            "risk_flags": risk_flags,
        }

    def _build_evidence(
        self,
        strategy: Dict[str, Any],
        estimate: Dict[str, Any],
        validation: Dict[str, Any],
        guardrails: Dict[str, Any],
        source_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = estimate.get("summary") or {}
        return {
            "strategy_id": strategy.get("id", ""),
            "strategy_title": strategy.get("title", ""),
            "window": summary.get("window") or strategy.get("window") or {},
            "validation_errors": validation.get("errors", []),
            "validation_warnings": validation.get("warnings", []),
            "guardrail_flags": guardrails.get("risk_flags", []),
            "impact_snapshot": {
                "delta_kWh": summary.get("delta_kWh", 0.0),
                "delta_carbon_kg": summary.get("delta_carbon_kg", 0.0),
                "peak_reduction_kW": summary.get("peak_reduction_kW", 0.0),
                "scope_size": summary.get("scope_size", 0),
            },
            "data_source": {
                "source_mode": source_profile.get("source_mode"),
                "readiness_label": source_profile.get("readiness_label"),
                "contract_name": source_profile.get("contract_name"),
                "contract_version": source_profile.get("contract_version"),
                "adapter_target": source_profile.get("adapter_target"),
                "primary_key": source_profile.get("primary_key"),
                "timestamp_key": source_profile.get("timestamp_key"),
                "observed_fields": source_profile.get("observed_fields") or [],
                "control_fields": source_profile.get("control_fields") or [],
                "replace_hint": source_profile.get("replace_hint"),
            },
            "generated_at": _now_iso(),
        }

    # -----------------------------
    # 4) 下发（演示）
    # -----------------------------
    def dispatch(
        self,
        strategy: Dict[str, Any],
        operator: str = "system",
        dry_run: bool = True,
        enforce_guardrails: bool = True,
        guardrail_min_peak_kw: float = 1.0,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        validation = self.validate_strategy(strategy)
        source_profile = self._infer_source_profile(strategy=strategy, estimate=None)

        if not validation.get("ok"):
            readiness = {
                "ok": False,
                "reason": "字段校验失败",
                "dispatch_ready": False,
                "data_contract_ready": bool(source_profile.get("contract_ready")),
                "source_mode": source_profile.get("source_mode"),
                "source_readiness_label": source_profile.get("readiness_label"),
                "adapter_target": source_profile.get("adapter_target"),
            }
            visible_notes = self._compose_visible_note(notes or "字段校验失败", source_profile, readiness)
            rec = DispatchRecord(
                job_id=str(uuid.uuid4()),
                created_at=_now_iso(),
                operator=operator,
                dry_run=dry_run,
                status="REJECTED",
                strategy_id=str(strategy.get("id", "")),
                strategy_title=str(strategy.get("title", "")),
                strategy=strategy,
                validation=validation,
                guardrails={"ok": False, "rules": [], "detail": "字段校验失败", "risk_flags": ["validation_failed"]},
                estimate={},
                readiness=readiness,
                contributors=[],
                evidence={
                    "strategy_id": str(strategy.get("id", "")),
                    "validation_errors": validation.get("errors", []),
                    "validation_warnings": validation.get("warnings", []),
                    "data_source": source_profile,
                    "generated_at": _now_iso(),
                },
                notes=visible_notes,
            )
            self._history.append(rec)
            return asdict(rec)

        est = self.estimate_effect(strategy)
        source_profile = self._infer_source_profile(strategy=strategy, estimate=est)
        gr = self._check_guardrails(
            est,
            enforce=enforce_guardrails,
            guardrail_min_peak_kw=guardrail_min_peak_kw,
        )

        estimate_summary = est.get("summary") if est.get("ok") else {}
        contributors = est.get("contributors", []) if est.get("ok") else []

        readiness = {
            "ok": bool(gr.get("ok", False)),
            "dispatch_ready": bool((estimate_summary or {}).get("dispatch_ready", False)),
            "reason": "守护栏通过，可记录为 dry-run" if gr.get("ok") else "守护栏未通过，拒绝记录",
            "data_contract_ready": bool(source_profile.get("contract_ready")),
            "source_mode": source_profile.get("source_mode"),
            "source_readiness_label": source_profile.get("readiness_label"),
            "adapter_target": source_profile.get("adapter_target"),
            "replace_hint": source_profile.get("replace_hint"),
        }

        status = "DRY_RUN_RECORDED" if (dry_run and readiness["ok"]) else (
            "REJECTED" if enforce_guardrails and not readiness["ok"] else "DRY_RUN_RECORDED"
        )

        evidence = self._build_evidence(strategy, est, validation, gr, source_profile)
        visible_notes = self._compose_visible_note(notes, source_profile, readiness)

        rec = DispatchRecord(
            job_id=str(uuid.uuid4()),
            created_at=_now_iso(),
            operator=operator,
            dry_run=dry_run,
            status=status,
            strategy_id=str(strategy.get("id", "")),
            strategy_title=str(strategy.get("title", "")),
            strategy=strategy,
            validation=validation,
            guardrails=gr,
            estimate=estimate_summary,
            readiness=readiness,
            contributors=contributors,
            evidence=evidence,
            notes=visible_notes,
        )
        self._history.append(rec)
        return asdict(rec)

    # -----------------------------
    # 5) 历史/取消
    # -----------------------------
    def list_history(self, limit: int = 50) -> Dict[str, Any]:
        arr = [asdict(x) for x in self._history[-max(1, int(limit)):]][::-1]
        return {
            "total": len(self._history),
            "items": arr,
        }

    def cancel(self, job_id: str, operator: str = "system") -> Dict[str, Any]:
        for i in range(len(self._history) - 1, -1, -1):
            if self._history[i].job_id == job_id:
                r = self._history[i]
                if r.status == "CANCELLED":
                    return {"ok": True, "job_id": job_id, "status": r.status}
                r.status = "CANCELLED"
                r.notes = ((r.notes or "") + f" | cancelled by {operator} at {_now_iso()}").strip()
                return {"ok": True, "job_id": job_id, "status": r.status}
        return {"ok": False, "error": "job_id 不存在"}
