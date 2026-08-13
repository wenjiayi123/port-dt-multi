from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class OpsXEvidenceService:
    """Read-only release/operations control-plane evidence for V3."""

    def __init__(self, ai_trust: Any, monitoring: Any, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.ai_trust = ai_trust
        self.monitoring = monitoring
        self.audit_dir = self.root / "data" / "objects" / "audit"

    @staticmethod
    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _audit_manifest(self) -> Dict[str, Any]:
        rows = []
        for path in sorted(self.audit_dir.glob("*.json"), key=lambda item: item.stat().st_mtime_ns):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                payload = {}
            kind = "southbound_command" if payload.get("command_id") else ("monitoring_drift" if "psi" in payload else "other")
            mode = stat.S_IMODE(path.stat().st_mode)
            rows.append({
                "evidence_id": path.name,
                "kind": kind,
                "sha256": self._sha(path),
                "bytes": path.stat().st_size,
                "mode": oct(mode),
                "owner_only": mode & 0o077 == 0,
                "recorded_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
        return {
            "records": len(rows),
            "command_records": sum(row["kind"] == "southbound_command" for row in rows),
            "monitoring_records": sum(row["kind"] == "monitoring_drift" for row in rows),
            "all_owner_only": all(row["owner_only"] for row in rows),
            "items": rows,
        }

    def build(self) -> Dict[str, Any]:
        trust = self.ai_trust.build()
        monitoring = self.monitoring.build()
        trust_benchmark = trust.get("benchmark") or {}
        trust_boundary = trust.get("boundary") or {}
        monitor_analysis = monitoring.get("current_analysis") or {}
        drift = monitor_analysis.get("drift") or {}
        monitor_decision = monitor_analysis.get("admission_decision") or {}
        audit = self._audit_manifest()
        config_path = os.getenv("PORT_DT_ACTUATOR_CONFIG", "").strip()
        actuator_configured = bool(config_path and Path(config_path).is_file())
        second_channel = len(os.getenv("PORT_DT_SECOND_CHANNEL_TOKEN", "")) >= 32
        rollout_backend_configured = actuator_configured and second_channel

        gates = [
            {"id": "artifact_integrity", "name": "模型/报告哈希", "status": "pass" if trust_benchmark.get("sidecar_sha256_match") else "fail", "evidence": trust_benchmark.get("report_sha256")},
            {"id": "offline_advantage", "name": "多种子时间盲测", "status": "pass" if trust_boundary.get("offline_claim_eligible") else "fail", "evidence": {"seeds": trust_benchmark.get("seeds"), "weighted_improvement_percent": trust_benchmark.get("weighted_improvement_percent")}},
            {"id": "monitoring", "name": "运行漂移门", "status": "fail" if drift.get("level") == "drift" else ("warn" if drift.get("level") == "warn" else "pass"), "evidence": {"psi": drift.get("psi"), "level": drift.get("level")}},
            {"id": "site_telemetry", "name": "现场遥测", "status": "pass" if monitoring.get("boundary", {}).get("live_data_verified") else "pending", "evidence": monitoring.get("source", {}).get("mode")},
            {"id": "actuator_config", "name": "白名单/南向路由", "status": "pass" if actuator_configured else "pending", "evidence": "PORT_DT_ACTUATOR_CONFIG" if actuator_configured else "unconfigured"},
            {"id": "two_person", "name": "双人异人确认", "status": "pass" if second_channel else "pending", "evidence": "secret configured" if second_channel else "second channel secret absent"},
            {"id": "rollback", "name": "回滚能力", "status": "pending", "evidence": "API implemented; site rehearsal absent"},
            {"id": "audit", "name": "审计证据权限/哈希", "status": "pass" if audit["all_owner_only"] else "fail", "evidence": {"records": audit["records"], "all_owner_only": audit["all_owner_only"]}},
        ]
        blockers = [row["id"] for row in gates if row["status"] != "pass"]
        rollout = {
            "phase": "blocked_pre_shadow",
            "traffic_percent": 0,
            "candidate": "SAC V3 public-data offline candidate",
            "stable_baseline": "FCFS / MPC safety baseline",
            "backend_configured": rollout_backend_configured,
            "mutations_enabled": False,
            "decision": "BLOCK",
            "blockers": blockers,
            "reason": "No site control plane is configured and the current calibrated replay drift gate is not clean.",
        }
        stages = [
            {"stage": "offline_candidate", "label": "离线候选", "status": "complete", "traffic_percent": 0},
            {"stage": "site_mapping", "label": "现场映射", "status": "待接入港口", "traffic_percent": 0},
            {"stage": "shadow", "label": "影子运行", "status": "blocked", "traffic_percent": 0},
            {"stage": "canary_5", "label": "5%灰度", "status": "not_started", "traffic_percent": 0},
            {"stage": "canary_25", "label": "25%灰度", "status": "not_started", "traffic_percent": 0},
            {"stage": "production", "label": "生产全量", "status": "not_authorized", "traffic_percent": 0},
        ]
        return {
            "version": "V3",
            "module": {"id": "opsx", "name": "OpsX", "state": rollout["phase"]},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "boundary": {
                "read_only_evidence_available": True,
                "live_rollout_verified": False,
                "production_authority": False,
                "site_status": "待接入港口",
                "engineering_simulator_enabled": False,
                "reason": "默认不开启生成式OpsX模拟器；页面展示真实仓库证据与失效安全状态，控制写操作保持禁用。",
            },
            "rollout": rollout,
            "gates": gates,
            "stage_ladder": stages,
            "current_incidents": [
                {
                    "id": "monitoring_input_drift",
                    "severity": "P1" if drift.get("level") == "drift" else "P2",
                    "state": "open" if drift.get("level") in {"drift", "warn"} else "cleared",
                    "source": "V3 monitoring evidence",
                    "detail": f"PSI={drift.get('psi')} / {drift.get('level')}; calibrated public replay, not a site incident.",
                    "action": monitor_decision.get("fallback"),
                },
                {
                    "id": "site_control_plane_missing",
                    "severity": "P1",
                    "state": "open",
                    "source": "actuator fail-closed config",
                    "detail": "Whitelist, southbound route and second-channel secret are not configured.",
                    "action": "Keep traffic at 0%; site owner must configure and rehearse rollback.",
                },
            ],
            "eight_capabilities": [
                {"id": "rollout", "name": "灰度流量控制", "implementation": "implemented", "runtime": "disabled_fail_closed"},
                {"id": "quality_gate", "name": "质量门", "implementation": "evidence_backed", "runtime": "active_read_only"},
                {"id": "timeline", "name": "事件时间线", "implementation": "audit_backed", "runtime": "site events pending"},
                {"id": "profile", "name": "策略画像", "implementation": "offline metrics available", "runtime": "online profiling pending"},
                {"id": "objective", "name": "多目标权重", "implementation": "training contract available", "runtime": "mutations disabled"},
                {"id": "guardrail", "name": "守护栏预演", "implementation": "software envelope available", "runtime": "site dry-run pending"},
                {"id": "health", "name": "运维健康", "implementation": "monitoring gate available", "runtime": "site SLO pending"},
                {"id": "audit", "name": "审计导出", "implementation": "hash manifest available", "runtime": "signed site report pending"},
            ],
            "audit_manifest": audit,
            "security_contract": {
                "command_flow": ["stage", "independent_confirm", "execute", "readback", "verify", "close_or_rollback"],
                "mandatory": ["asset/action whitelist", "idempotency key", "different requester/confirmer", "second-channel token", "hard constraint snapshot", "atomic owner-only evidence", "readback TTL", "rollback evidence"],
                "southbound_channels": ["OPC-UA", "Modbus-TCP", "MQTT", "HTTP/EMS/TOS"],
                "site_acceptance": ["0 hard constraint violations", "rollback RTO signed off", "readback success SLA", "alert/work-order closure", "canary KPI non-inferiority", "cybersecurity and change approval"],
            },
        }
