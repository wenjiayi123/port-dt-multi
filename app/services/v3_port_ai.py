from __future__ import annotations

import asyncio
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.rl_training.datasets import FACTOR_COLUMNS, load_port_dataset
from app.services.rl_training.statistics import summarize_metric_rows
from app.services.rl_training.trainer import ALGORITHMS, TRAINING_MANAGER


router = APIRouter(tags=["v3-port-ai"])
ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = ROOT / "app/ui/v3"
ADVANTAGE_PATH = ROOT / "evidence/v3/shanghai_public_advantage_v3.json"
STRONG_BASELINE_PATH = ROOT / "evidence/v3/strong_baseline_evidence_v3.json"
BUSINESS_IMPACT_PATH = ROOT / "evidence/v3/shanghai_public_business_impact_v3.json"
_LIVE_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_OVERVIEW_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_OVERVIEW_CACHE_TTL_SECONDS = 30.0


BUSINESS_CAPABILITIES = [
    {"id": "vessel", "name": "船舶到港与泊位", "state": "implemented", "engine": "到港压力 + 泊位优先控制", "site_replacement": "AIS、ETA、泊位计划、引航/拖轮"},
    {"id": "quay", "name": "岸桥与装卸节拍", "state": "contract_ready", "engine": "可用率掩码 + 服务能力约束", "site_replacement": "QC 状态、作业路、Moves/h、故障码"},
    {"id": "yard", "name": "堆场与箱区流动", "state": "implemented", "engine": "堆场占用 + Yard Flow 动作", "site_replacement": "箱位、翻箱、堆存期、场桥任务"},
    {"id": "horizontal", "name": "集疏运 / AGV / 集卡", "state": "adapter_required", "engine": "设备可用率与拥堵契约", "site_replacement": "车辆位置、任务队列、充电、路权"},
    {"id": "gate", "name": "闸口 / 铁路 / 驳船", "state": "adapter_required", "engine": "通道拥堵状态契约", "site_replacement": "预约、过闸、班列、驳船计划"},
    {"id": "energy", "name": "微电网与储能", "state": "implemented", "engine": "BESS、柔性负荷、峰值约束", "site_replacement": "电表、需量、电价、BMS、光储充"},
    {"id": "reefer", "name": "冷藏箱与岸电", "state": "contract_ready", "engine": "冷藏负荷因子 + 负荷联动", "site_replacement": "插拔箱、温控、岸电连接与费率"},
    {"id": "maintenance", "name": "设备健康与维护", "state": "contract_ready", "engine": "设备可用率 + 降级边界", "site_replacement": "振动、温度、工单、备件、RUL 标签"},
    {"id": "weather", "name": "气象海况与封航", "state": "implemented", "engine": "风浪能见度停机硬约束", "site_replacement": "港区站点、VTS、封航/复航事件"},
    {"id": "safety", "name": "安全审批与回滚", "state": "implemented", "engine": "建议态、限幅、双人审批、回滚证据", "site_replacement": "控制权限、SOP、PLC/TOS 回执"},
    {"id": "carbon", "name": "成本 / 碳 / 峰值", "state": "implemented", "engine": "多目标奖励 + 盲测指标", "site_replacement": "结算电价、排放因子、碳核算边界"},
    {"id": "multiport", "name": "跨港迁移与漂移", "state": "implemented", "engine": "端口画像、数据哈希、可用性掩码", "site_replacement": "字段映射、校准、影子运行与验收"},
]


BUSINESS_DEPTH = {
    "vessel": {
        "state_inputs": ["AIS/ETA", "泊位占用", "航道拥堵", "引航拖轮可用率", "风浪与封航"],
        "decision_outputs": ["泊位优先级", "服务强度", "到港节奏建议"],
        "hard_constraints": ["封航时服务归零", "引航拖轮资源门", "建议态不直接写入 TOS"],
        "acceptance_metrics": ["锚地等待", "靠泊准点率", "船舶周转时间", "泊位冲突率"],
    },
    "quay": {
        "state_inputs": ["岸桥可用率", "Moves/h", "故障与维修窗口", "船舶作业路"],
        "decision_outputs": ["服务能力因子", "岸桥资源建议", "降级运行建议"],
        "hard_constraints": ["设备可用率掩码", "天气停机边界", "作业安全间距"],
        "acceptance_metrics": ["桥吊效率", "净作业率", "故障停机时长", "计划偏差"],
    },
    "yard": {
        "state_inputs": ["箱区占用率", "箱位/堆存期", "翻箱任务", "场桥与集卡可用率"],
        "decision_outputs": ["Yard Flow 动作", "箱区流量建议", "拥堵降载"],
        "hard_constraints": ["箱区容量", "场桥冲突", "危险品/冷藏箱隔离"],
        "acceptance_metrics": ["翻箱率", "平均堆存期", "箱区拥堵", "场桥空驶"],
    },
    "horizontal": {
        "state_inputs": ["AGV/集卡位置", "任务队列", "电量与充电位", "路网拥堵"],
        "decision_outputs": ["任务分配契约", "充电窗口建议", "路权优先级"],
        "hard_constraints": ["最小安全距离", "SOC 下限", "道路与设备互锁"],
        "acceptance_metrics": ["空驶率", "任务等待", "车辆利用率", "充电冲突"],
    },
    "gate": {
        "state_inputs": ["预约时窗", "闸口队列", "铁路/驳船计划", "实际到离场"],
        "decision_outputs": ["预约削峰建议", "通道分流契约", "多式联运节拍"],
        "hard_constraints": ["闸口容量", "海关/安保放行", "班列与驳船固定窗"],
        "acceptance_metrics": ["卡车周转", "闸口排队", "预约命中", "集疏运峰谷差"],
    },
    "energy": {
        "state_inputs": ["分项电表", "BMS/SOC", "需量", "分时电价", "岸电/光储充"],
        "decision_outputs": ["BESS 充放电", "柔性负荷", "需量控制"],
        "hard_constraints": ["SOC 15%–90%", "功率/爬坡限幅", "终端 SOC 回归", "需量上限"],
        "acceptance_metrics": ["结算电费", "最大需量", "储能退化", "削峰填谷率"],
    },
    "reefer": {
        "state_inputs": ["插拔箱状态", "温控告警", "冷藏负荷", "岸电连接"],
        "decision_outputs": ["非关键负荷时移契约", "岸电接入建议"],
        "hard_constraints": ["温控不可违约", "生命/危险货物优先", "断电时长上限"],
        "acceptance_metrics": ["温控合规", "岸电使用率", "峰值负荷", "告警闭环"],
    },
    "maintenance": {
        "state_inputs": ["振动/温度", "故障码", "工单", "RUL", "备件库存"],
        "decision_outputs": ["健康降级因子", "维护窗口建议", "资源备用策略"],
        "hard_constraints": ["强制检修状态", "安全联锁", "无健康数据时保守降级"],
        "acceptance_metrics": ["非计划停机", "MTBF/MTTR", "工单及时率", "备件等待"],
    },
    "weather": {
        "state_inputs": ["港区风速", "能见度", "浪高", "流速", "封航事件"],
        "decision_outputs": ["停复工门", "作业能力折减", "气象风险提示"],
        "hard_constraints": ["风浪硬阈值", "缺字段不自动放行", "现场事件优先于模型"],
        "acceptance_metrics": ["误放行率", "预警提前量", "停复工一致性", "源可用率"],
    },
    "safety": {
        "state_inputs": ["策略版本", "数据质量", "约束投影", "审批与回执", "漂移告警"],
        "decision_outputs": ["建议/拒绝", "限幅动作", "回滚到上一冠军"],
        "hard_constraints": ["默认无生产控制权", "质量失败即关闭", "双人审批", "全链路审计"],
        "acceptance_metrics": ["违规率", "拒绝正确率", "回滚时延", "审计完整率"],
    },
    "carbon": {
        "state_inputs": ["结算电价", "边际排放因子", "分项负荷", "作业吞吐", "碳核算边界"],
        "decision_outputs": ["成本/碳/峰值 Pareto 建议", "策略画像选择"],
        "hard_constraints": ["吞吐不降级门", "安全违规为零", "金额与碳仅在数据边界内核算"],
        "acceptance_metrics": ["CNY/TEU", "kgCO₂e/TEU", "最大需量", "业务价值置信区间"],
    },
    "multiport": {
        "state_inputs": ["端口画像", "字段可用性掩码", "数据哈希", "时区/币种", "漂移统计"],
        "decision_outputs": ["迁移资格", "再校准建议", "策略拒绝/降级"],
        "hard_constraints": ["未知数据集拒绝", "训练/验证/盲测隔离", "现场字段缺失即禁止站点声明"],
        "acceptance_metrics": ["跨港退化", "漂移检出率", "校准误差", "影子运行通过率"],
    },
}


BUSINESS_CODE_EVIDENCE = {
    "vessel": ["app/services/forecast_twin/schedule.py", "app/services/forecast_twin/simulation.py", "app/services/rl_training/environment.py"],
    "quay": ["app/services/rl_model/port_G_qc_mvp/rl_engine_g.py", "app/services/forecast_twin/simulation.py", "tests/test_rl_training.py"],
    "yard": ["app/services/rl_model/yard_crane/rl_engine_f.py", "app/services/rl_training/environment.py", "tests/test_v3.py"],
    "horizontal": ["app/services/rl_model/agv_charge/train_iql.py", "app/services/rl_model/agv_charge/adapter.py", "app/data_contracts/energyx_contract.json"],
    "gate": ["app/services/forecast_twin/schedule.py", "app/services/mobile_api/workflow.py", "docs/SITE_DATA_REPLACEMENT_CONTRACT_V3.md"],
    "energy": ["app/services/rl_training/environment.py", "app/services/energyx/api.py", "app/services/rl_model/bess_energy/rl_engine.py"],
    "reefer": ["app/data_contracts/energyx_contract.json", "app/services/rl_training/datasets.py", "app/services/rl_training/environment.py"],
    "maintenance": ["app/services/monitoring_quality/data_quality.py", "app/services/monitoring_quality/alerts.py", "app/services/rl_training/environment.py"],
    "weather": ["scripts/fetch_shanghai_public_dataset.py", "app/services/rl_training/environment.py", "app/services/v3_port_ai.py"],
    "safety": ["app/services/exec_closedloop/rl_safety.py", "app/services/platform/rl/safety.py", "scripts/release_check.py"],
    "carbon": ["app/services/energy_reporting/compliance.py", "app/services/esg/service.py", "scripts/export_v3_business_impact.py"],
    "multiport": ["app/services/rl_training/profiles.py", "app/services/rl_training/datasets.py", "app/services/multiport/service.py"],
}


BUSINESS_EXECUTION_EVIDENCE = {
    "vessel": {
        "implementation_level": "model_backed_recommendation",
        "implementation_label": "模型建议已执行",
        "decision_source": "three-seed selected SAC on port_ops_v3 plus hard safety projection",
        "runtime_endpoints": ["/api/v3/runtime/frame", "/api/v3/runtime/series?scenario=strategy"],
        "current_data_mode": "public-data calibrated replay; AIS/ETA and berth actuals pending",
        "model_output_available": True,
        "site_blockers": ["AIS/ETA identity reconciliation", "berth plan and actual timestamps", "pilot/tug dispatch state"],
        "fail_closed_fallback": "FCFS/MPC recommendation only; no TOS write authority",
    },
    "quay": {
        "implementation_level": "executable_sandbox",
        "implementation_label": "沙箱引擎已执行",
        "decision_source": "quay-crane RL sandbox and forecast-twin service-capacity simulation",
        "runtime_endpoints": ["/api/curves/asset/qc-01?mode=forecast", "/api/forecast/qc-01"],
        "current_data_mode": "engineering replay and calibrated simulator; no QC PLC feed",
        "model_output_available": True,
        "site_blockers": ["QC PLC state", "work instruction and move events", "fault and maintenance windows"],
        "fail_closed_fallback": "availability mask reduces capacity; site KPI claim disabled",
    },
    "yard": {
        "implementation_level": "model_backed_specialized",
        "implementation_label": "专项模型已执行",
        "decision_source": "hash-verified yard-crane actor plus safety projection and port_ops_v3 yard-flow action",
        "runtime_endpoints": ["/api/v3/modules/yard-crane/evidence", "/api/v3/runtime/series?scenario=strategy"],
        "current_data_mode": "checked-in engineering replay; container genealogy pending",
        "model_output_available": True,
        "site_blockers": ["container position genealogy", "yard task events", "dangerous/reefer isolation rules"],
        "fail_closed_fallback": "block-level flow recommendation only; no equipment dispatch",
    },
    "horizontal": {
        "implementation_level": "legacy_model_adapter_pending",
        "implementation_label": "旧模型可运行·现场适配待接",
        "decision_source": "legacy AGV IQL artifact and adapter sandbox; not in the selected V3 policy",
        "runtime_endpoints": ["/api/rl/module_a/summary", "/api/rl/rollout/status"],
        "current_data_mode": "repository sandbox records; no fleet telemetry",
        "model_output_available": True,
        "site_blockers": ["fleet positions and missions", "road graph/version", "battery and charger receipts"],
        "fail_closed_fallback": "no fleet command; output remains offline recommendation",
    },
    "gate": {
        "implementation_level": "simulation_contract_only",
        "implementation_label": "仿真契约·无独立优化器",
        "decision_source": "forecast-twin schedule simulator; no learned gate/rail/barge controller",
        "runtime_endpoints": ["/api/v3/runtime/frame"],
        "current_data_mode": "derived congestion factor; appointment and actual flows unavailable",
        "model_output_available": False,
        "site_blockers": ["truck appointment/arrival/service/departure", "rail consist plan", "barge plan and customs release"],
        "fail_closed_fallback": "show 待接入港口; do not fabricate lane allocation",
    },
    "energy": {
        "implementation_level": "model_backed_specialized",
        "implementation_label": "专项模型已执行",
        "decision_source": "selected SAC plus shore/site BESS actors, MPC and hard SOC/power constraints",
        "runtime_endpoints": ["/api/v3/modules/shore-bess/evidence", "/api/v3/modules/bess-energy/evidence", "/api/energyx/breakdown"],
        "current_data_mode": "public replay and engineering asset ratings; settlement meters pending",
        "model_output_available": True,
        "site_blockers": ["meter hierarchy", "tariff and demand contract", "BMS/PCS limits and receipts"],
        "fail_closed_fallback": "recommendation disabled on missing BMS/limit fields",
    },
    "reefer": {
        "implementation_level": "coupled_factor_contract_only",
        "implementation_label": "负荷联动·无独立优化器",
        "decision_source": "reefer load enters port energy state; no temperature-aware reefer controller",
        "runtime_endpoints": ["/api/curves/asset/reefer-01?mode=forecast"],
        "current_data_mode": "derived reefer load; plug and temperature telemetry unavailable",
        "model_output_available": False,
        "site_blockers": ["container temperature/setpoint", "plug state", "cargo criticality and alarm acknowledgement"],
        "fail_closed_fallback": "no reefer curtailment action; load remains non-controllable",
    },
    "maintenance": {
        "implementation_level": "monitoring_only",
        "implementation_label": "监测告警·无维护优化器",
        "decision_source": "data-quality, residual anomaly and drift services; no trained RUL/work-order optimizer",
        "runtime_endpoints": ["/api/v3/monitoring/evidence"],
        "current_data_mode": "simulated/calibrated monitoring feed; CMMS and condition telemetry pending",
        "model_output_available": False,
        "site_blockers": ["condition sensors", "fault taxonomy", "CMMS work orders", "RUL labels and spare stock"],
        "fail_closed_fallback": "conservative availability degradation; no maintenance command",
    },
    "weather": {
        "implementation_level": "executable_safety_guard",
        "implementation_label": "安全门已执行",
        "decision_source": "public met-ocean feed/replay plus deterministic stop-work constraints",
        "runtime_endpoints": ["/api/v3/public-data/shanghai/live", "/api/v3/runtime/frame"],
        "current_data_mode": "public Yangshan reanalysis/live model; not port station or VTS event feed",
        "model_output_available": True,
        "site_blockers": ["port weather stations", "VTS closure/reopening events", "approved local thresholds"],
        "fail_closed_fallback": "missing mandatory site factor cannot automatically authorize work",
    },
    "safety": {
        "implementation_level": "executable_governance_workflow",
        "implementation_label": "审批回滚链已执行",
        "decision_source": "constraint projection, four-eyes workflow, audit hashes and rollback gate",
        "runtime_endpoints": ["/api/v3/opsx/evidence", "/api/v3/runtime/status"],
        "current_data_mode": "software workflow verified; real IAM/PLC/TOS receipts pending",
        "model_output_available": True,
        "site_blockers": ["site IAM identities", "control allowlist", "execution receipts", "rollback SLA drill"],
        "fail_closed_fallback": "production authority remains false",
    },
    "carbon": {
        "implementation_level": "executable_evidence_calculator",
        "implementation_label": "核算与门禁已执行",
        "decision_source": "multi-objective reward, blind-test comparison and unit-throughput scenario calculator",
        "runtime_endpoints": ["/api/v3/overview"],
        "current_data_mode": "public factors and scenario tariff; not audited settlement/carbon inventory",
        "model_output_available": True,
        "site_blockers": ["settlement tariff", "audited meter boundary", "approved emission factor and carbon scope"],
        "fail_closed_fallback": "show scenario value only; financial_audit_ready=false",
    },
    "multiport": {
        "implementation_level": "executable_transfer_guard",
        "implementation_label": "迁移与漂移门已执行",
        "decision_source": "port profile schema, dataset hash, availability masks and drift checks",
        "runtime_endpoints": ["/api/v3/data-readiness", "/api/multiport/summary"],
        "current_data_mode": "three public reference domains; no authorized site adapter",
        "model_output_available": True,
        "site_blockers": ["site field mapping", "calibration sample", "shadow outcomes", "owner-approved drift thresholds"],
        "fail_closed_fallback": "unknown dataset/profile is rejected instead of silently falling back",
    },
}


def _business_depth(capability_id: str) -> dict[str, Any]:
    code_paths = BUSINESS_CODE_EVIDENCE.get(capability_id, [])
    return {
        **BUSINESS_DEPTH.get(capability_id, {}),
        **BUSINESS_EXECUTION_EVIDENCE.get(capability_id, {}),
        "code_evidence": code_paths,
        "code_artifacts": [
            {
                "path": path,
                "exists": (ROOT / path).is_file(),
                "sha256": (
                    hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                    if (ROOT / path).is_file()
                    else None
                ),
            }
            for path in code_paths
        ],
        "production_ready": False,
    }


DEPLOYMENT_GATES = [
    {"id": "mapping", "name": "授权现场字段映射", "state": "required", "required_evidence": ["TOS/VTS/EMS/PLC 数据字典", "字段单位/时区/枚举映射", "数据责任人与授权范围"], "pass_criteria": ["必填字段覆盖 100%", "单位与时区转换测试通过", "来源等级不混淆"], "failure_action": "拒绝加载数据集，不回退到工程派生字段。"},
    {"id": "quality", "name": "时间同步与数据质量门", "state": "required", "required_evidence": ["端到端时钟偏差", "缺失/重复/迟到统计", "物理边界与设备状态一致性"], "pass_criteria": ["时间偏差小于站点 SLA", "质量阈值经运营方签字", "连续异常触发熔断"], "failure_action": "停止产生新建议，保持上一安全策略并告警。"},
    {"id": "calibration", "name": "端口校准与回放验收", "state": "required", "required_evidence": ["设备铭牌与控制限值", "历史事故/封航/故障样本", "TOS/EMS 回放对账"], "pass_criteria": ["状态估计误差达标", "基线可复现", "动作可行域与 SOP 一致"], "failure_action": "策略保持研究态，禁止设置 champion 或站点 KPI。"},
    {"id": "shadow", "name": "影子运行与漂移监控", "state": "required", "required_evidence": ["不少于完整业务周期的影子记录", "策略建议与人工决策差异", "输入/性能漂移报告"], "pass_criteria": ["零硬约束违规", "关键 KPI 区间不劣于基线", "漂移告警与降级演练通过"], "failure_action": "自动撤销候选资格，回到 FCFS/MPC 安全基线。"},
    {"id": "authority", "name": "人工审批 / 回滚 / 审计", "state": "software_ready", "required_evidence": ["双人审批身份", "策略/数据/配置哈希", "执行回执与回滚演练"], "pass_criteria": ["权限最小化", "审计链完整", "回滚时延符合 SLA"], "failure_action": "控制权保持关闭；开源版本始终只输出建议。"},
]


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _portable_evidence_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for folder in (ROOT / "evidence/rl", ROOT / "evidence/v3"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*_benchmark.json")):
            bundle = _read_json(path, {})
            if bundle.get("schema") != "port-dt-rl-benchmark-evidence.v1":
                continue
            for run in bundle.get("runs") or []:
                evaluation = run.get("evaluation") or {}
                rows.append(
                    {
                        "algorithm": run.get("algorithm"),
                        "dataset_id": run.get("dataset_id"),
                        "dataset_sha256": run.get("dataset_sha256"),
                        "metrics": evaluation.get("metrics") or {},
                        "uncertainty": evaluation.get("uncertainty") or {},
                        "episodes": evaluation.get("episodes"),
                        "job_id": run.get("job_id"),
                        "seed": run.get("seed"),
                        "total_steps": run.get("total_steps"),
                        "environment_version": run.get("environment_version"),
                        "business_profile_id": run.get("business_profile_id") or "default_port_profile",
                        "evidence_label": run.get("evidence_label"),
                        "evaluated_at": evaluation.get("evaluated_at"),
                        "_portable_record": run,
                        "_portable_bundle": str(path.relative_to(ROOT)),
                    }
                )
    return rows


def _all_evidence_runs() -> list[dict[str, Any]]:
    registry = _read_json(TRAINING_MANAGER.benchmark_path, {"runs": []})
    combined: dict[str, dict[str, Any]] = {
        str(run.get("job_id")): run
        for run in registry.get("runs") or []
        if run.get("job_id")
    }
    for run in _portable_evidence_runs():
        job_id = str(run.get("job_id") or "")
        if job_id and job_id not in combined:
            combined[job_id] = run
    return list(combined.values())


def _portable_run(job_id: str) -> dict[str, Any] | None:
    return next(
        (run.get("_portable_record") for run in _portable_evidence_runs() if run.get("job_id") == job_id),
        None,
    )


def _evidence_inventory() -> dict[str, Any]:
    runs = _all_evidence_runs()
    evidence_files = sorted(
        str(path.relative_to(ROOT))
        for folder in (ROOT / "evidence/rl", ROOT / "evidence/v3")
        if folder.exists()
        for path in folder.iterdir()
        if path.is_file()
    )
    return {
        "historical_run_count": len(runs),
        "formal_run_count": sum(
            run.get("evidence_label")
            in {"RL_HELD_OUT_EVALUATION", "DETERMINISTIC_CONTROLLER_BASELINE"}
            for run in runs
        ),
        "smoke_run_count": sum("SMOKE" in str(run.get("evidence_label")) for run in runs),
        "artifact_count": len(evidence_files),
        "artifacts": evidence_files,
        "append_only": True,
    }


def _algorithm_rows() -> list[dict[str, Any]]:
    runs = _all_evidence_runs()
    rows = []
    for spec in ALGORITHMS.values():
        selected = [run for run in runs if run.get("algorithm") == spec.id]
        v3_runs = [
            run
            for run in selected
            if run.get("dataset_id") == "public_cn_sha_hourly_v3"
            and run.get("environment_version") == "port_ops_v3"
            and run.get("evidence_label")
            in {"RL_HELD_OUT_EVALUATION", "DETERMINISTIC_CONTROLLER_BASELINE"}
        ]
        profile_groups: dict[str, list[dict[str, Any]]] = {}
        enriched_runs: list[dict[str, Any]] = []
        for run in v3_runs:
            portable = run.get("_portable_record") or {}
            config = _read_json(
                TRAINING_MANAGER.run_dir(str(run.get("job_id"))) / "config.json",
                {},
            )
            manifest = _read_json(
                TRAINING_MANAGER.run_dir(str(run.get("job_id"))) / "manifest.json",
                {},
            )
            portable_training = portable.get("training") or {}
            portable_model = portable.get("model_integrity") or {}
            profile_id = str(config.get("business_profile_id") or "default_port_profile")
            if portable:
                profile_id = str(portable.get("business_profile_id") or profile_id)
            enriched = {
                **run,
                "business_profile_id": profile_id,
                "reward_weights": config.get("reward_weights") or (portable.get("port_profile") or {}).get("objectives") or {},
                "episode_steps": config.get("episode_steps") or portable.get("episode_steps"),
                "test_ratio": config.get("test_ratio") or ((portable_training.get("split") or {}).get("test_ratio")),
                "validation_ratio": config.get("validation_ratio") or ((portable_training.get("split") or {}).get("validation_ratio")),
                "model_sha256": manifest.get("model_sha256") or portable_model.get("sha256"),
                "render_calls_during_training": manifest.get("render_calls_during_training") if manifest else portable_training.get("render_calls"),
                "evidence_source": run.get("_portable_bundle") or "local_run_registry",
            }
            profile_groups.setdefault(profile_id, []).append(enriched)
            enriched_runs.append(enriched)
        profiles = []
        for profile_id, group in sorted(profile_groups.items()):
            metrics = summarize_metric_rows([dict(run.get("metrics") or {}) for run in group])
            profiles.append(
                {
                    "id": profile_id,
                    "formal_runs": len(group),
                    "seeds": sorted(
                        {int(run["seed"]) for run in group if isinstance(run.get("seed"), int)}
                    ),
                    "reward_weights": group[0].get("reward_weights") or {},
                    "metrics": metrics,
                    "job_ids": [run.get("job_id") for run in group],
                    "model_sha256": [run.get("model_sha256") for run in group],
                    "render_calls_during_training": max(
                        (int(run.get("render_calls_during_training") or 0) for run in group),
                        default=0,
                    ),
                    "evidence_sources": sorted({str(run.get("evidence_source")) for run in group}),
                    "minimum_optimizer_steps": min(
                        (int(run.get("total_steps") or 0) for run in group),
                        default=0,
                    ),
                }
            )
        rows.append(
            {
                "id": spec.id,
                "name": spec.name,
                "family": spec.family,
                "action_space": spec.action_space,
                "trainable": spec.trainable,
                "implementation": spec.implementation,
                "description": spec.description,
                "formal_runs": len(v3_runs),
                "historical_formal_runs": sum(
                    run.get("evidence_label")
                    in {"RL_HELD_OUT_EVALUATION", "DETERMINISTIC_CONTROLLER_BASELINE"}
                    for run in selected
                ),
                "smoke_runs": sum("SMOKE" in str(run.get("evidence_label")) for run in selected),
                "v3_profiles": profiles,
                "v3_job_ids": [run.get("job_id") for run in enriched_runs],
            }
        )
    return rows


def _training_trace(job_id: str, *, max_points: int = 28) -> dict[str, Any] | None:
    path = TRAINING_MANAGER.run_dir(job_id) / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("progress"), (int, float)):
                rows.append(row)
    else:
        portable = _portable_run(job_id) or {}
        rows = [
            row
            for row in ((portable.get("training") or {}).get("optimizer_history") or [])
            if isinstance(row.get("progress"), (int, float))
        ]
    if not rows:
        return None
    stride = max(1, len(rows) // max_points)
    sampled = rows[::stride]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    reward_rows = [
        row for row in rows if isinstance(row.get("reward_mean"), (int, float))
    ]
    plotted_rows = reward_rows if len(reward_rows) >= 2 else rows
    stride = max(1, len(plotted_rows) // max_points)
    sampled = plotted_rows[::stride]
    if sampled[-1] is not plotted_rows[-1]:
        sampled.append(plotted_rows[-1])
    reward_available = len(reward_rows) >= 2
    return {
        "job_id": job_id,
        "observed_points": len(rows),
        "reward_available": reward_available,
        "points": [
            {
                "step": int(row.get("step") or 0),
                "progress": float(row.get("progress") or 0.0),
                **(
                    {"reward_mean": float(row["reward_mean"])}
                    if isinstance(row.get("reward_mean"), (int, float))
                    else {}
                ),
            }
            for row in sampled
        ],
        "final_optimizer_snapshot": {
            key: rows[-1].get(key)
            for key in (
                "actor_loss",
                "critic_loss",
                "policy_gradient_loss",
                "value_loss",
                "entropy_loss",
                "exploration_rate",
                "updates",
            )
            if rows[-1].get(key) is not None
        },
    }


def _historical_algorithm_runs(algorithm_id: str) -> list[dict[str, Any]]:
    rows = [
        run
        for run in _all_evidence_runs()
        if run.get("algorithm") == algorithm_id
        and run.get("evidence_label")
        in {"RL_HELD_OUT_EVALUATION", "DETERMINISTIC_CONTROLLER_BASELINE"}
    ]
    rows.sort(key=lambda run: str(run.get("evaluated_at") or ""), reverse=True)
    return [
        {
            "job_id": run.get("job_id"),
            "dataset_id": run.get("dataset_id"),
            "dataset_sha256": run.get("dataset_sha256"),
            "environment_version": run.get("environment_version"),
            "business_profile_id": run.get("business_profile_id") or "default_port_profile",
            "seed": run.get("seed"),
            "total_steps": run.get("total_steps"),
            "episodes": run.get("episodes"),
            "evidence_label": run.get("evidence_label"),
            "evaluated_at": run.get("evaluated_at"),
            "metrics": run.get("metrics") or {},
        }
        for run in rows
    ]


@router.get("/v3", include_in_schema=False)
async def v3_page() -> FileResponse:
    return FileResponse(UI_ROOT / "index.html", media_type="text/html")


@router.get("/v3/assets/{asset_name}", include_in_schema=False)
async def v3_asset(asset_name: str) -> FileResponse:
    allowed = {"v3.css": "text/css", "v3.js": "text/javascript"}
    if asset_name not in allowed:
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(UI_ROOT / asset_name, media_type=allowed[asset_name])


@router.get("/api/v3/overview")
async def v3_overview() -> dict[str, Any]:
    now = time.monotonic()
    cached = _OVERVIEW_CACHE.get("payload")
    if cached is not None and now - float(_OVERVIEW_CACHE.get("at") or 0.0) < _OVERVIEW_CACHE_TTL_SECONDS:
        return cached
    dataset = load_port_dataset("public_cn_sha_hourly_v3", TRAINING_MANAGER.data_root)
    advantage = _read_json(ADVANTAGE_PATH, None)
    strong_baselines = _read_json(STRONG_BASELINE_PATH, None)
    business_impact = _read_json(BUSINESS_IMPACT_PATH, None)
    if (advantage or {}).get("baseline", {}).get("environment_version") != "port_ops_v3":
        advantage = None
    if (strong_baselines or {}).get("environment_version") != "port_ops_v3":
        strong_baselines = None
    if (business_impact or {}).get("comparison", {}).get("environment_version") != "port_ops_v3":
        business_impact = None
    description = dataset.describe(test_ratio=0.2, validation_ratio=0.1)
    domain_depth = {
        capability["id"]: _business_depth(capability["id"])
        for capability in BUSINESS_CAPABILITIES
    }
    payload = {
        "version": "3.2.0",
        "product": "Port Twin AI Decision Platform",
        "mode": "public-data-offline-recommendation",
        "production_authority": False,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "id": dataset.dataset_id,
            "sha256": dataset.fingerprint,
            "rows": dataset.rows,
            "train_rows": description["train_rows"],
            "validation_rows": description["validation_rows"],
            "test_rows": description["test_rows"],
            "independent_source_observations": dataset.metadata.get("independent_source_observations"),
            "official_anchor_count": (dataset.metadata.get("source_observation_counts") or {}).get("official_port_reporting_periods"),
            "source_boundary": dataset.metadata.get("warning"),
        },
        "algorithms": _algorithm_rows(),
        "capabilities": [
            {**capability, "depth": domain_depth[capability["id"]]}
            for capability in BUSINESS_CAPABILITIES
        ],
        "business_domain_coverage": {
            "domain_count": len(BUSINESS_CAPABILITIES),
            "runtime_output_available_count": sum(
                bool(row.get("model_output_available")) for row in domain_depth.values()
            ),
            "no_independent_optimizer_count": sum(
                not bool(row.get("model_output_available")) for row in domain_depth.values()
            ),
            "all_code_artifacts_hash_verified": all(
                artifact.get("exists") and len(str(artifact.get("sha256") or "")) == 64
                for row in domain_depth.values()
                for artifact in row.get("code_artifacts") or []
            ),
            "production_ready_count": sum(
                bool(row.get("production_ready")) for row in domain_depth.values()
            ),
            "site_replacement_required_count": len(BUSINESS_CAPABILITIES),
        },
        "evidence": _evidence_inventory(),
        "advantage": advantage,
        "strong_baselines": strong_baselines,
        "business_impact": business_impact,
        "deployment_gates": DEPLOYMENT_GATES,
    }
    _OVERVIEW_CACHE.update({"at": time.monotonic(), "payload": payload})
    return payload


@router.get("/api/v3/algorithms/{algorithm_id}/evidence")
async def v3_algorithm_evidence(algorithm_id: str) -> dict[str, Any]:
    row = next((item for item in _algorithm_rows() if item["id"] == algorithm_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="algorithm not found")
    trace_job_ids = [
        str(profile.get("job_ids", [])[-1])
        for profile in row.get("v3_profiles", [])
        if profile.get("job_ids")
    ]
    return {
        **row,
        "training_traces": {
            job_id: trace
            for job_id in trace_job_ids
            if (trace := _training_trace(job_id)) is not None
        },
        "historical_evidence": {
            "append_only": True,
            "runs": _historical_algorithm_runs(algorithm_id),
        },
        "dataset_id": "public_cn_sha_hourly_v3",
        "dataset_sha256": load_port_dataset(
            "public_cn_sha_hourly_v3", TRAINING_MANAGER.data_root
        ).fingerprint,
        "protocol": {
            "split": "70% chronological train / 10% validation / 20% blind test",
            "render_during_training": False,
            "evaluation_policy": "deterministic",
            "holdout_episodes": 10,
            "minimum_distinct_seeds_for_rl_claim": 3,
        },
        "claim_boundary": "Public Shanghai aggregate plus public Yangshan reanalysis offline evidence; not measured terminal KPI or production authority.",
    }


@router.get("/api/v3/capabilities/{capability_id}")
async def v3_capability_detail(capability_id: str) -> dict[str, Any]:
    capability = next(
        (item for item in BUSINESS_CAPABILITIES if item["id"] == capability_id),
        None,
    )
    if capability is None:
        raise HTTPException(status_code=404, detail="capability not found")
    return {
        **capability,
        "depth": _business_depth(capability_id),
        "production_authority": False,
        "fail_closed": True,
    }


@router.get("/api/v3/data-readiness")
async def v3_data_readiness() -> dict[str, Any]:
    dataset_ids = ["public_us_la_6min_v1", "public_port_ops_v1", "public_cn_sha_hourly_v3"]
    ports = []
    for dataset_id in dataset_ids:
        dataset = load_port_dataset(dataset_id, TRAINING_MANAGER.data_root)
        quality = dataset.describe().get("quality") or {}
        ports.append(
            {
                "dataset_id": dataset_id,
                "dataset_sha256": dataset.fingerprint,
                "rows": dataset.rows,
                "evidence_tier": dataset.metadata.get("evidence_tier"),
                "independent_source_observations": dataset.metadata.get("independent_source_observations"),
                "factor_coverage": quality.get("factor_coverage"),
                "measured_columns": dataset.metadata.get("measured_columns") or [],
                "derived_columns": dataset.metadata.get("derived_columns") or [],
                "unavailable_factors": dataset.metadata.get("unavailable_factors") or [],
                "sources": [
                    {
                        "publisher": source.get("publisher"),
                        "url": source.get("url") or (source.get("source_urls") or [None])[0],
                    }
                    for source in dataset.metadata.get("sources") or []
                ],
                "recommended_role": (
                    "high-frequency public reference training and neutral cross-port comparison"
                    if dataset_id == "public_us_la_6min_v1"
                    else "long-horizon official aggregate scenario"
                    if dataset_id == "public_port_ops_v1"
                    else "Shanghai target adaptation and chronological blind test"
                ),
            }
        )
    return {
        "strategy": "multi-port public reference training + Shanghai target training + site-data replacement",
        "ports": ports,
        "canonical_factors": list(FACTOR_COLUMNS),
        "mandatory_site_replacements": [
            "TOS vessel/voyage/berth plan and actual timestamps",
            "quay crane, yard crane and horizontal transport telemetry",
            "container position, dwell, rehandle and yard-block state",
            "gate/rail/barge appointment and actual flow",
            "VTS/AIS/pilot/tug and local met-ocean observations",
            "meters, tariffs, BMS, shore-power and reefer state",
            "equipment alarms, maintenance work orders and safety events",
        ],
        "fail_closed": True,
    }


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "port-dt-multi-v3/3.1"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError(str(payload.get("reason") or "invalid public source response"))
    return payload


@router.get("/api/v3/public-data/shanghai/live")
async def shanghai_public_live() -> dict[str, Any]:
    now = time.monotonic()
    if _LIVE_CACHE.get("payload") and now - float(_LIVE_CACHE.get("at") or 0) < 300:
        return {**_LIVE_CACHE["payload"], "cache": "hit"}
    weather_query = urllib.parse.urlencode(
        {
            "latitude": 30.62,
            "longitude": 122.05,
            "current": "temperature_2m,wind_speed_10m,visibility",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
    )
    marine_query = urllib.parse.urlencode(
        {
            "latitude": 30.62,
            "longitude": 122.05,
            "current": "wave_height,sea_level_height_msl,ocean_current_velocity",
            "timezone": "UTC",
        }
    )
    try:
        weather, marine = await asyncio.gather(
            asyncio.to_thread(_fetch_json, "https://api.open-meteo.com/v1/forecast?" + weather_query),
            asyncio.to_thread(_fetch_json, "https://marine-api.open-meteo.com/v1/marine?" + marine_query),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "state": "public_source_unavailable",
                "fail_closed": True,
                "message": str(exc),
            },
        ) from exc
    payload = {
        "state": "live_public_model_feed",
        "site_telemetry": False,
        "decision_authority": False,
        "location": {"name": "Yangshan public grid", "latitude": 30.62, "longitude": 122.05},
        "weather": weather.get("current") or {},
        "weather_units": weather.get("current_units") or {},
        "marine": marine.get("current") or {},
        "marine_units": marine.get("current_units") or {},
        "sources": [
            {"name": "Open-Meteo weather", "url": "https://open-meteo.com/en/docs"},
            {"name": "Open-Meteo marine", "url": "https://open-meteo.com/en/docs/marine-weather-api"},
        ],
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cache": "miss",
    }
    _LIVE_CACHE.update(at=now, payload=payload)
    return payload
