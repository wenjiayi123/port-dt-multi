from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class GovernanceEvidenceService:
    """Aggregate V3 data, model, claim, access, safety and release governance."""

    def __init__(self, ai_trust: Any, opsx: Any, external_signals: Any, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.ai_trust = ai_trust
        self.opsx = opsx
        self.external_signals = external_signals

    def _exists(self, relative: str) -> bool:
        return (self.root / relative).is_file()

    def _workflow(self, relative: str) -> Dict[str, Any]:
        path = self.root / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        uses = re.findall(r"\buses:\s*([^\s#]+)", text)
        pinned = bool(uses) and all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", item) for item in uses)
        return {
            "path": relative,
            "configured": path.is_file(),
            "third_party_actions": len(uses),
            "actions_pinned_to_commit": pinned,
            "remote_run_state": "待GitHub上传后验证",
        }

    def build(self) -> Dict[str, Any]:
        trust = self.ai_trust.build()
        opsx = self.opsx.build()
        external = self.external_signals.build()
        trust_boundary = trust.get("boundary") or {}
        benchmark = trust.get("benchmark") or {}
        audit = opsx.get("audit_manifest") or {}
        ops_gates = {row.get("id"): row for row in (opsx.get("gates") or [])}
        live_signal_count = int(external.get("live_adapter_count") or 0)
        production_env = os.getenv("PORT_DT_ENV", "development").strip().lower() == "production"
        api_auth = bool(os.getenv("PORT_DT_API_KEYS", "").strip())
        admin_auth = bool(os.getenv("PORT_DT_ADMIN_API_KEYS", "").strip())
        actuator_config = bool(os.getenv("PORT_DT_ACTUATOR_CONFIG", "").strip())
        second_channel = len(os.getenv("PORT_DT_SECOND_CHANNEL_TOKEN", "")) >= 32

        workflows = [
            self._workflow(".github/workflows/ci.yml"),
            self._workflow(".github/workflows/codeql.yml"),
            self._workflow(".github/workflows/dependency-review.yml"),
            self._workflow(".github/workflows/scorecard.yml"),
            self._workflow(".github/workflows/release-attestation.yml"),
        ]
        supply_chain_configured = all(row["configured"] for row in workflows)
        actions_pinned = all(row["actions_pinned_to_commit"] for row in workflows)
        required_files = [
            "LICENSE", "SECURITY.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md",
            ".github/CODEOWNERS", "THIRD_PARTY_NOTICES.md", "CITATION.cff",
        ]
        open_source_files = [{"path": name, "present": self._exists(name)} for name in required_files]

        controls = [
            {"id": "data_provenance", "domain": "数据", "name": "来源/许可/快照/哈希", "status": "pass", "evidence": f"{benchmark.get('source_observations', 0)}个独立源观测；数据SHA={str(benchmark.get('dataset_sha256') or '')[:12]}…"},
            {"id": "data_classification", "domain": "数据", "name": "实测/官方汇总/再分析/衍生/缺失分级", "status": "pass", "evidence": "measured=0；TOS/AIS/实时市场明确待接入"},
            {"id": "temporal_isolation", "domain": "评测", "name": "训练/验证/盲测时间隔离", "status": "pass", "evidence": f"{benchmark.get('train_rows')}/{benchmark.get('validation_rows')}/{benchmark.get('blind_test_rows')}，不打乱"},
            {"id": "model_integrity", "domain": "模型", "name": "模型/数据/报告哈希", "status": "pass" if benchmark.get("sidecar_sha256_match") else "fail", "evidence": str(benchmark.get("report_sha256") or "")},
            {"id": "claim_governance", "domain": "声明", "name": "允许与禁止声明白名单", "status": "pass", "evidence": f"allowed={len((trust.get('claim_registry') or {}).get('allowed') or [])}; prohibited={len((trust.get('claim_registry') or {}).get('prohibited') or [])}"},
            {"id": "drift_incident", "domain": "运维", "name": "异常/漂移/工单闭环", "status": "warn", "evidence": "公开回放可算法验证；现场告警、CMMS、事故结论待接入"},
            {"id": "least_privilege", "domain": "权限", "name": "身份/RBAC/最小权限", "status": "pending", "evidence": "生产API密钥与管理员分层已实现；现场IAM/SSO未配置"},
            {"id": "two_person", "domain": "执行", "name": "请求人/确认人异人双审", "status": "pending", "evidence": "网关强制异人与第二通道；现场密钥未配置"},
            {"id": "rollback", "domain": "执行", "name": "回读/核验/回滚/RTO", "status": "pending", "evidence": "回滚API已实现；现场绑定和演练签字未完成"},
            {"id": "audit_chain", "domain": "审计", "name": "原子写/权限0600/哈希链", "status": "pass" if audit.get("all_owner_only") else "fail", "evidence": f"records={audit.get('records', 0)}; owner_only={audit.get('all_owner_only')}"},
            {"id": "carbon_ledger", "domain": "合规", "name": "Scope 1/2边界/排放因子/分摊台账", "status": "pending", "evidence": "公开工程碳口径可演算；经审计港口台账与签发人待接入"},
            {"id": "supply_chain", "domain": "开源安全", "name": "CodeQL/依赖审查/SBOM/签名证明", "status": "pass" if supply_chain_configured and actions_pinned and all(row["present"] for row in open_source_files) else "fail", "evidence": "5条GitHub工作流已配置；远程结果待上传后复核"},
        ]
        counts = {
            status: sum(row["status"] == status for row in controls)
            for status in ("pass", "warn", "pending", "fail")
        }
        role_matrix = [
            {"role": "viewer", "read": ["驾驶舱", "公开证据"], "write": [], "site_enforcement": "待接入港口IAM"},
            {"role": "auditor", "read": ["全部证据", "审计导出"], "write": ["审计备注"], "site_enforcement": "待接入港口IAM"},
            {"role": "operator", "read": ["策略", "设备状态"], "write": ["仅可stage白名单命令"], "site_enforcement": "生产API普通密钥+资产范围"},
            {"role": "approver", "read": ["命令与约束快照"], "write": ["异人确认/回滚"], "site_enforcement": "第二通道+请求人不同"},
            {"role": "admin", "read": ["配置/注册表"], "write": ["数据上传/模型推广配置"], "site_enforcement": "独立管理员密钥+变更工单"},
        ]
        risks = [
            {"id": "R-01", "severity": "P0", "risk": "公开回放被误当现场实测", "control": "每个API/UI显式标注evidence_class和production_authority=false", "owner": "Data/ML"},
            {"id": "R-02", "severity": "P0", "risk": "模型绕过人审直接下发", "control": "南向网关默认关闭；stage→异人确认→回读→核验", "owner": "Operations/Safety"},
            {"id": "R-03", "severity": "P1", "risk": "数据漂移后继续采用新策略", "control": "PSI/质量门触发BLOCK并回退FCFS/MPC", "owner": "MLOps"},
            {"id": "R-04", "severity": "P1", "risk": "离线相关性被宣传为现场因果收益", "control": "声明白/黑名单；A/B与经营签字未完成前禁止", "owner": "Product/Audit"},
            {"id": "R-05", "severity": "P1", "risk": "碳因子和边界未审计即签发合规报表", "control": "Scope 1/2台账无签发人时只显示待接入", "owner": "ESG/Audit"},
            {"id": "R-06", "severity": "P2", "risk": "开源依赖或发布制品被篡改", "control": "锁版+依赖审计+CodeQL+SBOM+发布证明；Actions绑定commit", "owner": "Security/Release"},
        ]
        release_blockers = [
            "现场IAM/SSO、资产范围与审批人目录未接入",
            "现场TOS/AIS/遥测/告警/工单/执行回执未接入",
            "影子运行、小流量灰度、回滚RTO与联锁演练未签字",
            "经审计Scope 1/2台账、碳因子版本和报表签发人未接入",
        ]
        return {
            "version": "V3",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "module": {"id": "governance", "name": "治理中心", "state": "offline_governed_site_blocked"},
            "boundary": {
                "offline_governance_verified": counts["fail"] == 0,
                "site_identity_verified": False,
                "audited_carbon_ledger_verified": False,
                "remote_ci_verified_for_current_changes": False,
                "production_authority": False,
                "site_status": "待接入港口",
                "reason": "开源离线证据、声明红线、执行失效安全与供应链配置可复核；现场身份、控制平面、合规台账与验收签字不在开源数据中。",
            },
            "summary": {
                "control_count": len(controls),
                "pass": counts["pass"], "warn": counts["warn"], "pending": counts["pending"], "fail": counts["fail"],
                "audit_records": int(audit.get("records") or 0),
                "audit_owner_only": bool(audit.get("all_owner_only")),
                "live_adapter_count": live_signal_count,
                "release_decision": "BLOCK",
                "rollout_traffic_percent": 0,
            },
            "controls": controls,
            "role_matrix": role_matrix,
            "separation_of_duties": {
                "policy_defined": True,
                "live_identity_backend": "待接入港口IAM/SSO",
                "production_env": production_env,
                "operator_api_key_configured": api_auth,
                "admin_api_key_configured": admin_auth,
                "actuator_configured": actuator_config,
                "second_channel_secret_configured": second_channel,
                "one_person_execution_allowed": False,
            },
            "claim_registry": trust.get("claim_registry") or {},
            "risk_register": risks,
            "audit_chain": {
                "records": audit.get("items") or [],
                "owner_only": bool(audit.get("all_owner_only")),
                "flow": ["事件/命令", "约束+模型+数据快照", "原子JSON写入", "chmod 0600", "SHA-256清单", "现场签名/时戳待接入"],
            },
            "open_source_security": {
                "required_files": open_source_files,
                "workflows": workflows,
                "dependency_versions_pinned": True,
                "release_check_script": "scripts/release_check.py",
                "secret_or_private_port_data_included": False,
                "current_remote_ci_state": "待视觉验收后上传GitHub再复核",
            },
            "data_policy": {
                "public_model_inputs": [row["id"] for row in (external.get("signal_registry") or []) if row.get("model_input")],
                "site_inputs_pending": [row["id"] for row in (external.get("signal_registry") or []) if row.get("availability") == "待接入港口"],
                "personal_or_vessel_identity_rows": 0,
                "replacement_contract": external.get("replacement_contract") or {},
            },
            "ab_test_policy": {
                "calculator_purpose": "前瞻样本量与监控设计，不是已实施A/B结果",
                "required": ["业务KPI和安全护栏预注册", "作业班次/码头/船型分层", "CUPED协变量只用实验前数据", "台风/维修/封航排除规则", "序贯监控与早停偏差控制", "回滚和事故开关"],
                "measured_experiment_available": False,
                "site_status": "待接入港口",
            },
            "release_gate": {
                "decision": "BLOCK",
                "traffic_percent": 0,
                "offline_candidate_may_be_reviewed": bool(trust_boundary.get("offline_claim_eligible")),
                "production_release_allowed": False,
                "blockers": release_blockers,
                "next_review": "现场接入、影子运行、小流量验收与回滚演练完成后",
            },
        }
