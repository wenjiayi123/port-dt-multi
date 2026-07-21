"""Xiaoyi command execution gateway.

This gateway accepts either a natural-language instruction or a registered
action id, resolves it through the RL panel action registry, and returns a
single command packet for Xiaoyi/port-dt-multi integration:

- which button will be executed
- which parameters are required or optional
- whether human confirmation is required
- what execution result was produced, or why execution is only staged
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from app.services.rl_actions.api import (
    DEFAULT_TRAIN_CONFIG,
    action_url,
    execute_registered_action,
    get_action_by_id,
    public_action,
    resolve_action,
)


router = APIRouter(prefix="/api/assistant/actions", tags=["assistant-actions"])


TRAINING_GOAL_ALIASES: List[Dict[str, Any]] = [
    {"id": "multi_objective", "label": "综合最优", "keywords": ["综合", "综合最优", "多目标", "整体最优"]},
    {"id": "energy_min", "label": "能耗最低", "keywords": ["能耗最低", "节能", "能耗", "省电", "最低能耗"]},
    {"id": "carbon_min", "label": "碳排最低", "keywords": ["碳排最低", "低碳", "降碳", "碳排", "碳排放最低"]},
    {"id": "cost_min", "label": "电费最低", "keywords": ["电费最低", "成本最低", "降成本", "省钱", "电价"]},
    {"id": "peak_shaving", "label": "需量峰值削减", "keywords": ["峰值削减", "削峰", "需量", "峰值", "越峰"]},
    {"id": "throughput_max", "label": "吞吐最大", "keywords": ["吞吐最大", "吞吐", "效率最高", "作业效率"]},
    {"id": "delay_min", "label": "等待最短", "keywords": ["等待最短", "船舶等待", "延误最低", "减少延误"]},
    {"id": "safety_guard", "label": "安全约束优先", "keywords": ["安全", "安全优先", "护栏", "保守"]},
    {"id": "battery_life", "label": "BESS 寿命友好", "keywords": ["电池寿命", "bess寿命", "储能寿命", "寿命友好"]},
    {"id": "shore_power_priority", "label": "岸电优先", "keywords": ["岸电优先", "岸电", "船舶岸电"]},
    {"id": "emission_quota", "label": "碳配额达标", "keywords": ["碳配额", "配额达标", "合规碳"]},
    {"id": "resilience", "label": "扰动韧性", "keywords": ["扰动韧性", "韧性", "台风", "暴雨", "极端天气"]},
    {"id": "agv_turnaround", "label": "AGV 周转效率", "keywords": ["agv周转", "agv", "充换电", "车辆周转"]},
    {"id": "berth_reliability", "label": "泊位窗口稳定", "keywords": ["泊位", "靠泊", "离泊", "船期稳定", "泊位窗口"]},
    {"id": "grid_stability", "label": "电网稳定", "keywords": ["电网稳定", "馈线", "电压", "供电稳定"]},
    {"id": "carbon_cost_balance", "label": "碳成本平衡", "keywords": ["碳成本", "碳成本平衡", "碳和电费", "成本和碳"]},
    {"id": "low_risk_canary", "label": "低风险试运行", "keywords": ["低风险", "灰度", "试运行", "canary", "小流量"]},
    {"id": "storm_resilience", "label": "台风扰动鲁棒", "keywords": ["台风鲁棒", "台风扰动", "风暴鲁棒", "应急鲁棒"]},
]


TRAINING_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "multi_objective": {
        "title": "综合稳态推荐",
        "reason": "适合首次演示：SAC 对连续功率调度稳定，四类 reward 权重保持均衡。",
        "operator_note": "我会先用综合最优作为稳态基线，便于后续和单目标训练做对比。",
        "config": {"algorithm": "sac", "scenario": "mapped_dataset", "asset_group": "all_port", "horizon_min": 720, "step_min": 5, "total_steps": 240000, "batch_size": 256, "learning_rate": 0.0003, "gamma": 0.995, "tau": 0.005, "entropy_coef": 0.02, "replay_buffer": 120000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.24, "carbon": 0.22, "peak": 0.18, "safety": 0.20}},
    },
    "energy_min": {
        "title": "节能优先推荐",
        "reason": "能耗优化需要兼顾 BESS、岸电和冷站连续动作，SAC 的探索更平滑。",
        "operator_note": "我会把奖励权重偏向能耗与安全，避免为了省电过度压缩设备响应。",
        "config": {"algorithm": "sac", "scenario": "mapped_dataset", "asset_group": "all_port", "horizon_min": 720, "step_min": 5, "total_steps": 260000, "batch_size": 256, "learning_rate": 0.00028, "gamma": 0.996, "tau": 0.005, "entropy_coef": 0.018, "replay_buffer": 140000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.22, "carbon": 0.20, "peak": 0.16, "safety": 0.26}},
    },
    "carbon_min": {
        "title": "低碳窗口推荐",
        "reason": "碳排最低通常依赖低碳时段和岸电/BESS 联动，夜间低碳场景更容易看出效果。",
        "operator_note": "我会优先选择低碳窗口训练，并提高碳权重；训练结果仍需策略测试和 dry-run。",
        "config": {"algorithm": "sac", "scenario": "night_low_carbon", "asset_group": "qc_bess_shore", "horizon_min": 720, "step_min": 5, "total_steps": 280000, "batch_size": 256, "learning_rate": 0.00025, "gamma": 0.996, "tau": 0.004, "entropy_coef": 0.018, "replay_buffer": 150000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.18, "carbon": 0.42, "peak": 0.16, "safety": 0.24}},
    },
    "cost_min": {
        "title": "电价套利推荐",
        "reason": "电费最低需要连续控制充放电和削峰，TD3 对连续动作的控制更直接。",
        "operator_note": "我会偏向电费权重，同时保留峰值和安全限制，防止低价时段形成新峰值。",
        "config": {"algorithm": "td3", "scenario": "mapped_dataset", "asset_group": "qc_bess_shore", "horizon_min": 720, "step_min": 5, "total_steps": 300000, "batch_size": 256, "learning_rate": 0.00024, "gamma": 0.995, "tau": 0.004, "entropy_coef": 0.012, "replay_buffer": 160000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.44, "carbon": 0.14, "peak": 0.20, "safety": 0.22}},
    },
    "peak_shaving": {
        "title": "削峰保守推荐",
        "reason": "削峰目标要控制瞬时功率爬坡，TD3 配合严格护栏更适合连续功率边界。",
        "operator_note": "我会拉高峰值权重，并采用严格护栏，先保证不越限再谈收益。",
        "config": {"algorithm": "td3", "scenario": "noon_peak", "asset_group": "all_port", "horizon_min": 480, "step_min": 5, "total_steps": 320000, "batch_size": 256, "learning_rate": 0.00022, "gamma": 0.994, "tau": 0.003, "entropy_coef": 0.010, "replay_buffer": 180000, "guardrail_mode": "strict", "demand_cap_kw": 480, "reward_weights": {"cost": 0.18, "carbon": 0.14, "peak": 0.44, "safety": 0.24}},
    },
    "throughput_max": {
        "title": "吞吐效率推荐",
        "reason": "吞吐目标更容易触碰服务水平和安全边界，PPO 的稳定更新适合给操作员审核。",
        "operator_note": "我会优先保障作业效率，但把安全权重保留在高位，防止策略过激。",
        "config": {"algorithm": "ppo", "scenario": "noon_peak", "asset_group": "berth_ops", "horizon_min": 360, "step_min": 5, "total_steps": 220000, "batch_size": 512, "learning_rate": 0.0002, "gamma": 0.993, "tau": 0.006, "entropy_coef": 0.025, "replay_buffer": 100000, "guardrail_mode": "balanced", "reward_weights": {"cost": 0.12, "carbon": 0.12, "peak": 0.16, "safety": 0.28}},
    },
    "delay_min": {
        "title": "船期延误压降推荐",
        "reason": "等待时间优化需要平衡泊位、岸电和排队，PPO 更新稳定，适合短窗口演示。",
        "operator_note": "我会让训练优先压低等待和延误，但不绕过泊位冲突与岸电容量护栏。",
        "config": {"algorithm": "ppo", "scenario": "shore_power_peak", "asset_group": "berth_ops", "horizon_min": 360, "step_min": 5, "total_steps": 220000, "batch_size": 512, "learning_rate": 0.0002, "gamma": 0.993, "tau": 0.006, "entropy_coef": 0.024, "replay_buffer": 110000, "guardrail_mode": "balanced", "reward_weights": {"cost": 0.12, "carbon": 0.14, "peak": 0.14, "safety": 0.30}},
    },
    "safety_guard": {
        "title": "安全优先推荐",
        "reason": "安全优先目标要减少探索幅度，PPO + 严格护栏更适合人机确认边界。",
        "operator_note": "我会降低学习率和探索强度，先把越限、冲突和人工确认边界压住。",
        "config": {"algorithm": "ppo", "scenario": "storm_disruption", "asset_group": "all_port", "horizon_min": 480, "step_min": 5, "total_steps": 200000, "batch_size": 512, "learning_rate": 0.00016, "gamma": 0.992, "tau": 0.004, "entropy_coef": 0.012, "replay_buffer": 90000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.12, "carbon": 0.12, "peak": 0.16, "safety": 0.52}},
    },
    "battery_life": {
        "title": "BESS 寿命友好推荐",
        "reason": "电池寿命目标需要限制充放电深度和爬坡，TD3 更适合连续动作边界。",
        "operator_note": "我会降低探索强度并保留 SOC/爬坡护栏，避免为了削峰伤害电池寿命。",
        "config": {"algorithm": "td3", "scenario": "mapped_dataset", "asset_group": "qc_bess_shore", "horizon_min": 720, "step_min": 5, "total_steps": 280000, "batch_size": 256, "learning_rate": 0.0002, "gamma": 0.996, "tau": 0.003, "entropy_coef": 0.008, "replay_buffer": 150000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.22, "carbon": 0.18, "peak": 0.24, "safety": 0.34}},
    },
    "shore_power_priority": {
        "title": "岸电优先推荐",
        "reason": "岸电优先会受接入窗口和馈线容量约束，SAC 适合做岸电/BESS 联动。",
        "operator_note": "我会选择岸电高峰场景，并保持严格馈线容量护栏。",
        "config": {"algorithm": "sac", "scenario": "shore_power_peak", "asset_group": "qc_bess_shore", "horizon_min": 720, "step_min": 5, "total_steps": 260000, "batch_size": 256, "learning_rate": 0.00026, "gamma": 0.995, "tau": 0.004, "entropy_coef": 0.017, "replay_buffer": 140000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.18, "carbon": 0.30, "peak": 0.20, "safety": 0.28}},
    },
    "emission_quota": {
        "title": "碳配额合规推荐",
        "reason": "配额达标要保留审计证据和安全边界，PPO 的稳定性更适合合规演示。",
        "operator_note": "我会把碳权重拉高，但保留安全权重；结果只作为策略建议，不作为合规结论。",
        "config": {"algorithm": "ppo", "scenario": "night_low_carbon", "asset_group": "all_port", "horizon_min": 720, "step_min": 5, "total_steps": 240000, "batch_size": 512, "learning_rate": 0.00018, "gamma": 0.995, "tau": 0.005, "entropy_coef": 0.016, "replay_buffer": 120000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.16, "carbon": 0.40, "peak": 0.14, "safety": 0.30}},
    },
    "resilience": {
        "title": "扰动韧性推荐",
        "reason": "扰动韧性强调保守冗余和异常恢复，PPO + 台风扰动场景更好解释。",
        "operator_note": "我会优先训练台风/异常场景下的稳健策略，短期收益可能不如激进优化。",
        "config": {"algorithm": "ppo", "scenario": "storm_disruption", "asset_group": "all_port", "horizon_min": 480, "step_min": 5, "total_steps": 260000, "batch_size": 512, "learning_rate": 0.00016, "gamma": 0.992, "tau": 0.004, "entropy_coef": 0.014, "replay_buffer": 130000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.14, "carbon": 0.14, "peak": 0.20, "safety": 0.44}},
    },
    "agv_turnaround": {
        "title": "AGV 周转推荐",
        "reason": "AGV 周转目标涉及充换电排队和车辆利用率，PPO 在离散作业窗口上更稳。",
        "operator_note": "我会选择 AGV 充换电设备组，先提升周转效率，再用安全间隔约束防止拥堵。",
        "config": {"algorithm": "ppo", "scenario": "noon_peak", "asset_group": "agv_charge", "horizon_min": 360, "step_min": 5, "total_steps": 240000, "batch_size": 512, "learning_rate": 0.00022, "gamma": 0.993, "tau": 0.006, "entropy_coef": 0.026, "replay_buffer": 120000, "guardrail_mode": "balanced", "reward_weights": {"cost": 0.14, "carbon": 0.14, "peak": 0.18, "safety": 0.30}},
    },
    "berth_reliability": {
        "title": "泊位可靠性推荐",
        "reason": "泊位窗口稳定需要船期可靠性和岸电窗口协同，PPO 便于稳健审核。",
        "operator_note": "我会以泊位作业链为主，优先降低船期偏差和资源冲突。",
        "config": {"algorithm": "ppo", "scenario": "shore_power_peak", "asset_group": "berth_ops", "horizon_min": 480, "step_min": 5, "total_steps": 240000, "batch_size": 512, "learning_rate": 0.00018, "gamma": 0.994, "tau": 0.005, "entropy_coef": 0.018, "replay_buffer": 120000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.12, "carbon": 0.16, "peak": 0.16, "safety": 0.42}},
    },
    "grid_stability": {
        "title": "电网稳定推荐",
        "reason": "电网稳定要限制馈线、电压和同时动作风险，TD3 + 严格护栏更保守。",
        "operator_note": "我会优先压住馈线和电压扰动，收益会比激进节能策略低一些。",
        "config": {"algorithm": "td3", "scenario": "shore_power_peak", "asset_group": "qc_bess_shore", "horizon_min": 480, "step_min": 5, "total_steps": 300000, "batch_size": 256, "learning_rate": 0.00018, "gamma": 0.994, "tau": 0.003, "entropy_coef": 0.008, "replay_buffer": 170000, "guardrail_mode": "strict", "demand_cap_kw": 460, "reward_weights": {"cost": 0.16, "carbon": 0.16, "peak": 0.34, "safety": 0.40}},
    },
    "carbon_cost_balance": {
        "title": "碳成本平衡推荐",
        "reason": "该目标需要在电价和碳因子之间折中，SAC 适合连续多目标权衡。",
        "operator_note": "我会让成本和碳权重接近，适合展示多目标取舍。",
        "config": {"algorithm": "sac", "scenario": "mapped_dataset", "asset_group": "all_port", "horizon_min": 720, "step_min": 5, "total_steps": 260000, "batch_size": 256, "learning_rate": 0.00026, "gamma": 0.995, "tau": 0.004, "entropy_coef": 0.018, "replay_buffer": 140000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.32, "carbon": 0.32, "peak": 0.16, "safety": 0.24}},
    },
    "low_risk_canary": {
        "title": "低风险灰度推荐",
        "reason": "低风险试运行更适合小步数、低学习率和严格人工确认边界。",
        "operator_note": "我会使用保守参数，只做灰度训练演示，不直接生产执行。",
        "config": {"algorithm": "ppo", "scenario": "mapped_dataset", "asset_group": "all_port", "horizon_min": 240, "step_min": 5, "total_steps": 120000, "batch_size": 256, "learning_rate": 0.00012, "gamma": 0.990, "tau": 0.004, "entropy_coef": 0.010, "replay_buffer": 60000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.18, "carbon": 0.18, "peak": 0.18, "safety": 0.48}},
    },
    "storm_resilience": {
        "title": "台风鲁棒推荐",
        "reason": "台风扰动下应急预案优先，PPO + 严格护栏更便于审计解释。",
        "operator_note": "我会将场景切到台风扰动，优先保持关键设备冗余和人工接管边界。",
        "config": {"algorithm": "ppo", "scenario": "storm_disruption", "asset_group": "all_port", "horizon_min": 480, "step_min": 5, "total_steps": 220000, "batch_size": 512, "learning_rate": 0.00014, "gamma": 0.991, "tau": 0.004, "entropy_coef": 0.012, "replay_buffer": 110000, "guardrail_mode": "strict", "reward_weights": {"cost": 0.12, "carbon": 0.12, "peak": 0.18, "safety": 0.54}},
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _input_text(payload: Dict[str, Any]) -> str:
    xiaoyi = payload.get("xiaoyi") if isinstance(payload.get("xiaoyi"), dict) else {}
    return str(
        payload.get("instruction")
        or payload.get("text")
        or payload.get("question")
        or xiaoyi.get("question")
        or xiaoyi.get("answer")
        or ""
    )


def _input_intent(payload: Dict[str, Any]) -> str:
    xiaoyi = payload.get("xiaoyi") if isinstance(payload.get("xiaoyi"), dict) else {}
    return str(payload.get("intent") or payload.get("xiaoyi_intent") or xiaoyi.get("intent") or "")


def _normalise_text(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _training_goal_from_text(text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    explicit = str(payload.get("objective") or payload.get("training_goal") or "").strip()
    if explicit:
        for item in TRAINING_GOAL_ALIASES:
            if explicit == item["id"] or _normalise_text(explicit) == _normalise_text(item["label"]):
                return item
        return {"id": explicit, "label": explicit, "keywords": []}
    norm = _normalise_text(text)
    for item in TRAINING_GOAL_ALIASES:
        for keyword in item["keywords"]:
            if _normalise_text(keyword) in norm:
                return item
    return TRAINING_GOAL_ALIASES[0]


def _start_training_route_with_params(route: str, payload: Dict[str, Any]) -> str:
    instruction = _input_text(payload)
    goal, recommendation, cfg = _recommended_training_packet(instruction, payload)
    weights = cfg.get("reward_weights") if isinstance(cfg.get("reward_weights"), dict) else {}
    params = {
        "objective": goal["id"],
        "objective_label": goal["label"],
        "command": instruction,
        "confirm": "prompt",
        "algorithm": cfg.get("algorithm"),
        "scenario": cfg.get("scenario"),
        "asset_group": cfg.get("asset_group"),
        "horizon_min": cfg.get("horizon_min"),
        "step_min": cfg.get("step_min"),
        "total_steps": cfg.get("total_steps"),
        "batch_size": cfg.get("batch_size"),
        "learning_rate": cfg.get("learning_rate"),
        "gamma": cfg.get("gamma"),
        "tau": cfg.get("tau"),
        "entropy_coef": cfg.get("entropy_coef"),
        "replay_buffer": cfg.get("replay_buffer"),
        "demand_cap_kw": cfg.get("demand_cap_kw"),
        "guardrail_mode": cfg.get("guardrail_mode"),
        "cost_w": weights.get("cost"),
        "carbon_w": weights.get("carbon"),
        "peak_w": weights.get("peak"),
        "safety_w": weights.get("safety"),
        "recommendation_title": recommendation.get("title"),
        "recommendation_reason": recommendation.get("reason"),
        "operator_note": recommendation.get("operator_note"),
        "module_target": cfg.get("module_target"),
        "module_label": cfg.get("module_label"),
        "advanced_config": json.dumps(cfg, ensure_ascii=False, separators=(",", ":")),
    }
    separator = "&" if "?" in route else "?"
    return route + separator + urlencode(params)


def _recommended_training_packet(text: str, payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    goal = _training_goal_from_text(text, payload)
    base = TRAINING_RECOMMENDATIONS.get(goal["id"], TRAINING_RECOMMENDATIONS["multi_objective"])
    recommendation = deepcopy(base)
    cfg = deepcopy(DEFAULT_TRAIN_CONFIG)
    cfg.update(deepcopy(recommendation.get("config") or {}))
    if isinstance(payload.get("config"), dict):
        cfg.update(payload["config"])
    cfg["objective"] = goal["id"]
    cfg["objective_label"] = goal["label"]
    cfg["source_command"] = text
    recommendation["objective_id"] = goal["id"]
    recommendation["objective_label"] = goal["label"]
    recommendation["config"] = cfg
    return goal, recommendation, cfg


def _param_spec(action_id: str) -> List[Dict[str, Any]]:
    if action_id == "start_rl_training":
        return [
            {
                "name": "config",
                "required": False,
                "source": "payload.config",
                "default": DEFAULT_TRAIN_CONFIG,
                "description": "训练配置；未提供时使用 RL 面板默认训练参数。",
            },
            {
                "name": "confirm",
                "required": True,
                "source": "payload.confirm",
                "default": False,
                "description": "真正执行训练启动前需要人工确认；dry_run=true 时只预演。",
            },
        ]
    if action_id == "run_policy_test":
        return [
            {
                "name": "strategy_id",
                "required": False,
                "source": "payload.strategy_id",
                "default": "auto:first",
                "description": "策略 ID；未提供时先拉取策略列表并使用第一条策略。",
            },
            {
                "name": "horizon_min",
                "required": False,
                "source": "payload.horizon_min",
                "default": 360,
                "description": "策略仿真预测窗口。",
            },
            {
                "name": "step_min",
                "required": False,
                "source": "payload.step_min",
                "default": 5,
                "description": "策略仿真步长。",
            },
        ]
    if action_id == "verify_policy_for_online":
        return [
            {
                "name": "strategy_id",
                "required": False,
                "source": "payload.strategy_id",
                "default": "auto:first",
                "description": "要验证的策略 ID；当前训练未完成时，必须显式选择已完成归档策略。",
            },
            {
                "name": "use_completed_strategy",
                "required": False,
                "source": "payload.use_completed_strategy",
                "default": False,
                "description": "当当前训练处于运行、暂停或重置状态时，经人工确认后改用已完成归档策略继续演练。",
            },
            {
                "name": "confirm_production",
                "required": False,
                "source": "not_supported",
                "default": False,
                "description": "本动作只做上线前校验和 dry-run，不支持生产执行；生产上线必须走人工确认边界。",
            },
        ]
    if action_id == "view_rl_training_status":
        return [
            {
                "name": "job_id",
                "required": False,
                "source": "payload.job_id",
                "default": "latest",
                "description": "训练任务 ID；未提供时返回最近一次 RL 训练任务状态。",
            }
        ]
    if action_id == "stop_rl_training":
        return [
            {
                "name": "confirm",
                "required": True,
                "source": "payload.confirm",
                "default": False,
                "description": "暂停/停止训练前需要人工确认。",
            }
        ]
    if action_id == "start_xiaoyi_ai":
        return [
            {
                "name": "confirm",
                "required": True,
                "source": "payload.confirm",
                "default": False,
                "description": "启动小懿AI本地服务前需要人工确认。",
            }
        ]
    if action_id in {"open_sailing_simulator", "start_navigation_demo", "switch_ship_view", "run_sailing_rl_smoke_test"}:
        return [
            {
                "name": "confirm",
                "required": True,
                "source": "payload.confirm",
                "default": False,
                "description": "启动桌面航行模拟器或运行 Godot smoke test 前需要人工确认。",
            }
        ]
    return []


def _missing_parameters(action_id: str, payload: Dict[str, Any], dry_run: bool, requires_human_confirm: bool) -> List[str]:
    missing: List[str] = []
    if not dry_run and requires_human_confirm and not bool(payload.get("confirm")):
        missing.append("confirm")
    if action_id == "start_rl_training" and "config" in payload and not isinstance(payload.get("config"), dict):
        missing.append("config")
    return missing


def _button_packet(action: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selector": action.get("button_selector"),
        "label": action.get("button_label"),
        "sequence": action.get("button_sequence", []),
    }


def _backend_packet(action: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    backend = action.get("backend_request")
    if not isinstance(backend, dict):
        return None
    packet = dict(backend)
    if action.get("id") == "start_rl_training":
        body = dict(packet.get("body") or {})
        _goal, recommendation, cfg = _recommended_training_packet(_input_text(payload), payload)
        cfg["recommendation"] = {
            "title": recommendation.get("title"),
            "reason": recommendation.get("reason"),
            "operator_note": recommendation.get("operator_note"),
        }
        body["config"] = cfg
        packet["body"] = body
    if action.get("id") == "run_policy_test":
        body = dict(packet.get("body") or {})
        if payload.get("strategy_id"):
            body["strategy_id"] = str(payload["strategy_id"])
        if payload.get("horizon_min"):
            body["horizon_min"] = int(payload["horizon_min"])
        if payload.get("step_min"):
            body["step_min"] = int(payload["step_min"])
        packet["body"] = body
    if action.get("id") == "verify_policy_for_online":
        body = dict(packet.get("body") or {})
        if payload.get("strategy_id"):
            body["strategy_id"] = str(payload["strategy_id"])
        body["dry_run"] = True
        packet["body"] = body
    if action.get("id") == "view_rl_training_status" and payload.get("job_id"):
        packet["path"] = str(packet.get("path") or "/api/rl/train/status") + f"?job_id={payload['job_id']}"
    return packet


def _resolve_for_gateway(payload: Dict[str, Any]) -> Dict[str, Any]:
    action_id = str(payload.get("action_id") or "").strip()
    if action_id:
        action = get_action_by_id(action_id)
        if not action:
            raise HTTPException(status_code=404, detail=f"未知动作：{action_id}")
        public = public_action(action)
        public["score"] = 999
        public["match_reasons"] = ["explicit_action_id"]
        return {"matched": True, "action": public, "candidates": [public]}
    return resolve_action(instruction=_input_text(payload), intent=_input_intent(payload))


@router.post("/execute", summary="小懿指令执行网关")
def assistant_action_execute(request: Request, payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    resolved = _resolve_for_gateway(payload)
    action_public = resolved["action"]
    action = get_action_by_id(str(action_public["id"]))
    if not action:
        raise HTTPException(status_code=404, detail=f"未知动作：{action_public.get('id')}")

    execution_policy = action.get("execution") if isinstance(action.get("execution"), dict) else {}
    dry_run = bool(payload.get("dry_run", execution_policy.get("dry_run_default", True)))
    requires_human_confirm = bool(action.get("requires_human_confirm"))
    confirm_provided = bool(payload.get("confirm"))
    missing = _missing_parameters(action["id"], payload, dry_run=dry_run, requires_human_confirm=requires_human_confirm)
    route = str(action_public.get("route") or "/rl-panel")
    recommendation: Optional[Dict[str, Any]] = None
    if action["id"] == "start_rl_training":
        _goal, recommendation, _cfg = _recommended_training_packet(_input_text(payload), payload)
        route = _start_training_route_with_params(route, payload)
    backend_request = _backend_packet(action_public, payload)
    action_for_execution = dict(action)
    action_for_execution["route"] = route
    if backend_request is not None:
        action_for_execution["backend_request"] = backend_request

    can_execute = not missing
    if dry_run:
        execution = execute_registered_action(request, action_for_execution, payload, dry_run=True)
        execution_result = {
            "status": execution.get("status", "ready_to_execute"),
            "mode": "dry_run",
            "executed": False,
            "result": execution,
        }
    elif can_execute:
        execution = execute_registered_action(request, action_for_execution, payload, dry_run=False)
        non_executed_statuses = {
            "failed",
            "ready_to_execute",
            "ready_to_launch",
            "training_incomplete",
            "completed_strategy_selection_required",
        }
        execution_result = {
            "status": execution.get("status", "executed"),
            "mode": "executed",
            "executed": execution.get("status") not in non_executed_statuses,
            "result": execution,
        }
    else:
        execution_result = {
            "status": "confirmation_required" if "confirm" in missing else "missing_parameters",
            "mode": "blocked",
            "executed": False,
            "result": None,
        }

    will_execute = {
        "action_id": action_public["id"],
        "action_label": action_public["label"],
        "open_url": action_url(request, route),
        "route": route,
        "button": _button_packet(action_public),
        "backend_request": backend_request,
        "execution_type": execution_policy.get("type"),
    }

    return JSONResponse(
        {
            "ok": execution_result["status"] not in {"failed", "missing_parameters"},
            "updated_at": _utc_now(),
            "gateway": "xiaoyi_assistant_action_gateway",
            "input": {
                "instruction": _input_text(payload),
                "intent": _input_intent(payload),
                "action_id": payload.get("action_id"),
            },
            "matched": resolved["matched"],
            "action": action_public,
            "candidates": resolved.get("candidates", []),
            "will_execute": will_execute,
            "recommendation": recommendation,
            "required_parameters": _param_spec(action["id"]),
            "missing_parameters": missing,
            "human_confirmation": {
                "required": requires_human_confirm,
                "provided": confirm_provided,
                "needed_before_execution": bool(requires_human_confirm and not confirm_provided),
                "reason": "该动作会启动训练、测试或桌面程序，需要人工确认。" if requires_human_confirm else "该动作只打开面板或查询状态。",
            },
            "execution_result": execution_result,
        }
    )
