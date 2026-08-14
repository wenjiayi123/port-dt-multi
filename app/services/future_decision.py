"""Evidence-backed future decision deck for the default V3 runtime.

The deck compares the selected SAC evidence with formal FCFS/MPC blind-test
references and overlays the current monitoring admission gate.  It never
creates an execution receipt or grants production authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STRONG_BASELINE_PATH = ROOT / "evidence/v3/strong_baseline_evidence_v3.json"
STRONG_BASELINE_DIGEST_PATH = ROOT / "evidence/v3/strong_baseline_evidence_v3.sha256"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


class FutureDecisionService:
    """Compose a no-dispatch V3 decision packet from verified evidence."""

    def __init__(self, runtime: Any, monitoring: Any) -> None:
        self.runtime = runtime
        self.monitoring = monitoring

    @staticmethod
    def _formal_evidence() -> tuple[dict[str, Any], str]:
        raw = STRONG_BASELINE_PATH.read_bytes()
        actual_digest = hashlib.sha256(raw).hexdigest()
        expected_digest = STRONG_BASELINE_DIGEST_PATH.read_text(encoding="utf-8").split()[0]
        if actual_digest != expected_digest:
            raise RuntimeError("strong-baseline evidence hash verification failed")
        payload = json.loads(raw)
        if payload.get("schema") != "port-dt-v3-strong-baseline-evidence.v1":
            raise RuntimeError("strong-baseline evidence schema mismatch")
        return payload, actual_digest

    @staticmethod
    def _candidate(
        *,
        candidate_id: str,
        title: str,
        mode: str,
        tag: str,
        rows: list[dict[str, Any]],
        fcfs_energy_kwh: float,
        fcfs_peak_kw: float,
        risk_level: str,
        basis: str,
    ) -> dict[str, Any]:
        energy_kwh = _mean(rows, "grid_energy_kwh")
        peak_kw = _mean(rows, "peak_kw")
        return {
            "id": candidate_id,
            "title": title,
            "mode": mode,
            "tag": tag,
            "baseline_energy_kwh": round(fcfs_energy_kwh, 6),
            "energy_saving_kwh": round(fcfs_energy_kwh - energy_kwh, 6),
            "peak_reduction_kw": round(fcfs_peak_kw - peak_kw, 6),
            "confidence": 0.95,
            "confidence_basis": "paired blind-test report uses 95% bootstrap intervals",
            "risk_level": risk_level,
            "dispatch_ready": False,
            "evidence_basis": basis,
            "production_authority": False,
        }

    def run(
        self,
        *,
        horizon_min: int = 90,
        step_min: int = 5,
        max_candidates: int = 3,
        source: str = "rl-future-deck",
    ) -> dict[str, Any]:
        runtime_status = self.runtime.status()
        runtime_series = self.runtime.series(
            horizon_min=horizon_min,
            step_min=step_min,
            scenario="strategy",
        )
        monitoring = self.monitoring.build()
        formal, formal_digest = self._formal_evidence()

        model = runtime_status.get("model") or {}
        if not runtime_status.get("available") or not runtime_series.get("available"):
            raise RuntimeError("V3 hash-verified runtime is unavailable")
        if formal.get("dataset_sha256") != model.get("dataset_sha256"):
            raise RuntimeError("runtime and formal evidence dataset hashes do not match")
        if formal.get("environment_version") != model.get("environment_version"):
            raise RuntimeError("runtime and formal evidence environments do not match")

        baselines = formal.get("baselines") or {}
        fcfs_rows = list((baselines.get("fcfs_neutral") or {}).get("window_metrics") or [])
        mpc_rows = list((baselines.get("mpc") or {}).get("window_metrics") or [])
        sac_rows = list(formal.get("selected_window_metrics") or [])
        if not fcfs_rows or not mpc_rows or not sac_rows:
            raise RuntimeError("formal candidate windows are incomplete")

        fcfs_energy = _mean(fcfs_rows, "grid_energy_kwh")
        fcfs_peak = _mean(fcfs_rows, "peak_kw")
        candidates = [
            self._candidate(
                candidate_id="mpc-formal-reference",
                title="MPC 正式盲测参照",
                mode="10窗口配对盲测",
                tag="SAFE REFERENCE",
                rows=mpc_rows,
                fcfs_energy_kwh=fcfs_energy,
                fcfs_peak_kw=fcfs_peak,
                risk_level="人工复核",
                basis="strong_baseline_evidence_v3.baselines.mpc",
            ),
            self._candidate(
                candidate_id="sac-selected-policy",
                title="SAC 已选模型证据",
                mode="三种子集成盲测",
                tag="SELECTED MODEL",
                rows=sac_rows,
                fcfs_energy_kwh=fcfs_energy,
                fcfs_peak_kw=fcfs_peak,
                risk_level="准入门复核",
                basis="strong_baseline_evidence_v3.selected_window_metrics",
            ),
            self._candidate(
                candidate_id="fcfs-neutral-reference",
                title="FCFS 中性安全参照",
                mode="10窗口配对盲测",
                tag="FALLBACK REFERENCE",
                rows=fcfs_rows,
                fcfs_energy_kwh=fcfs_energy,
                fcfs_peak_kw=fcfs_peak,
                risk_level="保守参照",
                basis="strong_baseline_evidence_v3.baselines.fcfs_neutral",
            ),
        ][:max(1, min(3, int(max_candidates)))]

        analysis = monitoring.get("current_analysis") or {}
        drift = analysis.get("drift") or {}
        admission = analysis.get("admission_decision") or {}
        summary = runtime_series.get("summary") or {}
        telemetry = runtime_status.get("telemetry") or {}
        strong_gate = formal.get("strong_baseline_gate") or {}
        policy_suggestions_allowed = bool(admission.get("new_policy_suggestions_allowed"))

        data_hash_passed = bool(
            model.get("dataset_sha256")
            and model.get("dataset_sha256") == formal.get("dataset_sha256")
            and model.get("dataset_sha256") == telemetry.get("sha256")
        )
        guardrails = [
            {
                "id": "data_lineage",
                "name": "数据血缘与哈希一致",
                "level": "hard",
                "passed": data_hash_passed,
                "actual": "一致" if data_hash_passed else "不一致",
                "unit": "",
                "threshold": "runtime = replay = formal evidence",
                "source": "runtime/status + strong_baseline_evidence_v3",
            },
            {
                "id": "model_artifact",
                "name": "已保存模型可复现加载",
                "level": "hard",
                "passed": bool(runtime_status.get("available") and model.get("model_sha256")),
                "actual": str(model.get("model_sha256") or "unavailable")[:12],
                "unit": "",
                "threshold": "SHA-256 verified",
                "source": "evidence/v3/runtime/runtime_model.json",
            },
            {
                "id": "software_envelope",
                "name": "当前窗口软件硬约束",
                "level": "hard",
                "passed": bool(summary.get("hard_guardrail_passed")),
                "actual": "通过" if summary.get("hard_guardrail_passed") else "阻断",
                "unit": "",
                "threshold": "all inferred actions within software envelope",
                "source": "/api/v3/runtime/series.summary",
            },
            {
                "id": "model_drift",
                "name": "监控漂移准入门",
                "level": "hard",
                "passed": policy_suggestions_allowed,
                "actual": f"PSI {float(drift.get('psi')):.3f}" if drift.get("psi") is not None else "不可用",
                "unit": "",
                "threshold": "PSI < 0.25 且允许新策略建议",
                "source": "/api/v3/monitoring/evidence.current_analysis",
            },
            {
                "id": "strong_baseline",
                "name": "强基线严格优势门",
                "level": "hard",
                "passed": bool(strong_gate.get("all_comparators_strictly_beaten")),
                "actual": "通过" if strong_gate.get("all_comparators_strictly_beaten") else "未通过",
                "unit": "",
                "threshold": "95%CI 下严格击败全部强基线",
                "source": "strong_baseline_evidence_v3.strong_baseline_gate",
            },
            {
                "id": "production_authority",
                "name": "生产权限准入（现场门）",
                "level": "soft",
                "passed": False,
                "actual": "无生产控制权",
                "unit": "",
                "threshold": "现场授权、双人确认与执行回执",
                "source": "/api/v3/runtime/status.production_authority",
            },
        ]
        hard_passed = all(item["passed"] for item in guardrails if item["level"] == "hard")

        p50 = list((runtime_series.get("series") or {}).get("p50") or [])
        actions = list(runtime_series.get("actions") or [])
        first_control = (actions[0].get("decoded_control") or {}) if actions else {}
        snapshot = {
            "bess_soc_pct": round(float(first_control.get("projected_soc") or summary.get("terminal_soc") or 0.0) * 100.0, 6),
            "shore_power_kw": round(float(p50[0].get("kW") if p50 else 0.0), 6),
            "reward_drift": drift.get("psi"),
            "candidate_pool_size": len(candidates),
            "horizon_min": horizon_min,
            "step_min": step_min,
            "source_mode": telemetry.get("mode"),
        }

        # The monitoring fallback allows FCFS or MPC, but the repository has no
        # site-approved incumbent policy that would justify choosing between
        # them automatically.  Keep both visible and leave selection to review.
        recommended_id = None
        recommended_title = "FCFS / MPC 安全参照（待人工选择）" if candidates else None
        decision = {
            "ready_for_human_dry_run": hard_passed,
            "label": "可进入人工 dry-run" if hard_passed else "准入门已阻断，保持安全参照",
            "next_action": (
                "仅可由操作员确认后进入无设备控制的 dry-run"
                if hard_passed
                else str(admission.get("fallback") or "保持上一稳定策略或 FCFS/MPC 安全基线")
            ),
            "recommended_strategy_id": recommended_id,
            "recommended_strategy_title": recommended_title,
            "recommendation_boundary": "MPC/FCFS 仅为正式盲测参照，不是现场当前策略或生产推荐。",
            "production_boundary": "本页只做公开数据校准回放、正式盲测参照、护栏和审计；无生产控制权，不下发设备指令。",
            "production_authority": False,
        }

        generated_at = _utc_now()
        evidence_core = {
            "generated_at": generated_at,
            "source": source,
            "snapshot": snapshot,
            "candidates": candidates,
            "guardrails": guardrails,
            "decision": decision,
            "runtime_model_sha256": model.get("model_sha256"),
            "dataset_sha256": model.get("dataset_sha256"),
            "formal_evidence_sha256": formal_digest,
        }
        evidence_digest = _sha256_json(evidence_core)
        run_id = f"v3-future-{generated_at.replace(':', '').replace('-', '')[:15]}-{evidence_digest[:8]}"

        stages = [
            {"id": "situation", "status": "done"},
            {"id": "candidates", "status": "done"},
            {"id": "counterfactual", "status": "done"},
            {"id": "guardrails", "status": "done" if hard_passed else "blocked"},
            {"id": "receipt", "status": "done"},
        ]
        logs = [
            f"[Snapshot] V3 saved-policy window · {horizon_min} min / {step_min} min step",
            f"[Evidence] runtime model {str(model.get('model_sha256') or '')[:12]} · dataset {str(model.get('dataset_sha256') or '')[:12]}",
            f"[Candidates] SAC / MPC / FCFS formal paired blind-test references · {len(candidates)} cards",
            f"[Guardrail] monitoring={admission.get('decision') or 'unavailable'} · strong-baseline={'PASS' if strong_gate.get('all_comparators_strictly_beaten') else 'BLOCK'}",
            "[Boundary] public-data calibrated replay; production authority=false; no device command was sent",
            f"[Audit] evidence digest {evidence_digest}",
        ]
        return {
            "schema": "port-dt-v3-future-decision.v1",
            "run_id": run_id,
            "generated_at": generated_at,
            "snapshot": snapshot,
            "candidates": candidates,
            "recommended_strategy_id": recommended_id,
            "guardrails": guardrails,
            "decision": decision,
            "stages": stages,
            "logs": logs,
            "audit": {
                "evidence_digest": evidence_digest,
                "runtime_model_sha256": model.get("model_sha256"),
                "dataset_sha256": model.get("dataset_sha256"),
                "formal_evidence_sha256": formal_digest,
                "source": source,
                "production_action_executed": False,
            },
            "production_authority": False,
        }
