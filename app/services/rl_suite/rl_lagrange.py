# ============================================
# app/services/rl_lagrange.py
# --------------------------------------------
# 约束 RL（CMDP）与拉格朗日乘子管理 + 目标函数打分器
#
# 大白话：
#   - Combines economic and carbon objectives with peak, SLA, intensity, and thermal constraints.
#     在一个地方统一管理。给每条策略一个“总目标分数”，用于排序/选优。
#   - 用拉格朗日乘子（lambda）把“约束违规”转化为惩罚项，训练/在线微调时滚动更新 lambda。
#   - 结果可持久化到 data/objects/rl/lagrange.json，支持影子/灰度/全量切换时回滚。
#
# 接口最常用：
#   mgr = LagrangeManager(storage=..., caps=..., init_config=ObjectiveConfig(...))
#   score = mgr.evaluate_strategy(strategy, context=..., price=..., carbon_factor=...)
#   mgr.update_lambdas_from_feedback(measured_constraints)
#
# 策略结构（与 rl_panel/dispatch 一致）示例：
#   strategy = {
#     "id": "agv_charge_shift",
#     "window": {"start":"2025-10-07T02:00:00Z","end":"2025-10-07T03:00:00Z"},
#     "actions": [
#       {"asset":"agv-01", "cmd":"charge", "kW": 80.0},
#       {"asset":"yard-01","cmd":"reduce","kW": 10.0}
#     ],
#     "meta": {"expected_delay_min": 0.0, "expected_throughput_teu": 120}
#   }
#
# 说明：
#   - 本文件不依赖重型 RL 框架；训练/环境在后续文件里提供（rl_train / rl_env）。
#   - evaluate_strategy 内部有“可解释输出”：各目标/各约束的贡献、被罚的原因。
# ============================================

from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- 可配置目标（权重） ----------
@dataclass
class ObjectiveConfig:
    """目标函数权重配置：
    - total_cost = elec_cost + carbon_cost + demand_charge（按需）
    - carbon_intensity = kgCO2e / TEU
    - energy = kWh 总量
    你可以只开其中一个（另两个权重设成 0），实现“最小成本”或“最小碳强度”等。
    """
    w_cost: float = 1.0
    w_carbon_intensity: float = 0.0
    w_energy: float = 0.0
    # 额外：对延迟（SLA）也可以作为目标的一部分（而非约束）
    w_delay_min: float = 0.0

# ---------- 约束定义 ----------
@dataclass
class ConstraintSpec:
    """单个约束项：g(x) <= limit 即通过；超出部分视为违约度（violation = g - limit >= 0）"""
    name: str
    limit: float
    # 描述：解释用途（前端显示）
    desc: str = ""
    # 违规软阈值（deadband），小于该幅度不触发更新，抑制抖动
    deadband: float = 0.0
    # 乘子更新步长（可独立于全局 alpha）
    step: Optional[float] = None

@dataclass
class ConstraintState:
    """约束当前值、违约度、乘子等"""
    value: float
    limit: float
    violation: float
    lambda_val: float

# ---------- 价格&因子 ----------
@dataclass
class PriceSignal:
    """电价/碳价/需量价简版"""
    tou_price_per_kwh: float = 0.9     # 元/kWh 或本币/kWh
    carbon_price_per_kg: float = 0.0   # 元/kgCO2e
    demand_charge_per_kw: float = 0.0  # 元/kW（按峰值收取，粗略估计）

@dataclass
class CarbonFactor:
    """电网碳排因子（kgCO2e/kWh），可按时段变化；这里先给简单常量"""
    kg_per_kwh: float = 0.55

# ---------- 主类：拉格朗日乘子管理器 ----------
class LagrangeManager:
    """
    管目标 & 约束 & 乘子；提供“策略打分”和“乘子滚动更新”。
    注意：这里是**无模型**打分器（用于排序/rule-based RL/或优化器协同）；
    真正的策略网络训练/环境交互在后续文件（rl_train / rl_env）。
    """

    def __init__(
        self,
        storage=None,
        init_config: Optional[ObjectiveConfig] = None,
        constraint_specs: Optional[List[ConstraintSpec]] = None,
        caps: Optional[Dict[str, Any]] = None,
        state_file: str = "data/objects/rl/lagrange.json",
        alpha: float = 0.02,  # 全局乘子更新步长上限（会根据各约束 step 微调）
    ):
        self.storage = storage
        self.cfg = init_config or ObjectiveConfig()
        self.state_file = state_file
        self.alpha = alpha
        self.caps = caps or {}  # 可传入设备/并网能力口径，便于约束解释
        self.lambda_map: Dict[str, float] = {}
        self.specs: Dict[str, ConstraintSpec] = {}

        # Default constraints may be overridden by the deployment profile.
        default_specs = [
            ConstraintSpec(name="feeder_peak_kw", limit= self._grid_limit_kw(), desc="馈线峰值不得超过并网限额（含N-1裕度）", deadband=5.0),
            ConstraintSpec(name="sla_delay_min", limit= 10.0, desc="作业延迟不超过10分钟（示例）", deadband=1.0),
            ConstraintSpec(name="carbon_intensity_kg_per_teu", limit= 6.0, desc="碳强度上限（示例：6 kgCO2e/TEU）", deadband=0.1),
            ConstraintSpec(name="device_temp_max_c", limit= 85.0, desc="设备温度不超过85℃（软约束，硬约束由 rl_safety 拦截）", deadband=0.5),
        ]
        for s in (constraint_specs or default_specs):
            self.specs[s.name] = s

        self._load_state()

    # ----- 公共：策略打分 -----
    def evaluate_strategy(
        self,
        strategy: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        price: Optional[PriceSignal] = None,
        carbon_factor: Optional[CarbonFactor] = None,
    ) -> Dict[str, Any]:
        """
        给策略打分（越小越好）： total = w_cost*cost + w_carbon*CI + w_energy*energy + w_delay*delay + Σ lambda_i * violation_i
        返回包含可解释组成，用于“策略贡献榜/为何被罚/可行域”显示。
        """
        price = price or PriceSignal()
        cf = carbon_factor or CarbonFactor()
        ctx = context or {}

        # 1) 估计策略的能量/峰值/延迟/吞吐等（不用仿真引擎也能出一个合理估算）
        est = self._estimate_impacts(strategy, ctx, cf)

        # 2) 目标函数组成
        elec_cost = est["energy_kwh"] * price.tou_price_per_kwh
        carbon_cost = est["carbon_kg"] * price.carbon_price_per_kg
        demand_cost = max(0.0, est["peak_kw_delta"]) * price.demand_charge_per_kw  # 简化：只对新增峰值计费
        total_cost = elec_cost + carbon_cost + demand_cost

        carbon_intensity = (est["carbon_kg"] / max(1e-6, est["throughput_teu"])) if est["throughput_teu"] else 0.0
        energy_kwh = est["energy_kwh"]
        delay_min = est["delay_min"]

        obj = (
            self.cfg.w_cost * total_cost
            + self.cfg.w_carbon_intensity * carbon_intensity
            + self.cfg.w_energy * energy_kwh
            + self.cfg.w_delay_min * delay_min
        )

        # 3) 约束计算（违约度）+ 拉格朗日惩罚
        constr: Dict[str, ConstraintState] = {}
        lag_pen = 0.0

        # 3.1 馈线峰值约束（基线+策略后的峰值，不超过 limit）
        s_peak = self.specs.get("feeder_peak_kw")
        if s_peak:
            peak_val = max(0.0, est["baseline_peak_kw"] + est["peak_kw_delta"])
            viol = max(0.0, peak_val - s_peak.limit - s_peak.deadband)
            lam = self.lambda_map.get(s_peak.name, 0.0)
            lag_pen += lam * viol
            constr[s_peak.name] = ConstraintState(value=peak_val, limit=s_peak.limit, violation=viol, lambda_val=lam)

        # 3.2 SLA 延迟
        s_sla = self.specs.get("sla_delay_min")
        if s_sla:
            val = max(0.0, delay_min)  # 延迟为正表示不利
            viol = max(0.0, val - s_sla.limit - s_sla.deadband)
            lam = self.lambda_map.get(s_sla.name, 0.0)
            lag_pen += lam * viol
            constr[s_sla.name] = ConstraintState(value=val, limit=s_sla.limit, violation=viol, lambda_val=lam)

        # 3.3 碳强度
        s_ci = self.specs.get("carbon_intensity_kg_per_teu")
        if s_ci:
            val = carbon_intensity
            viol = max(0.0, val - s_ci.limit - s_ci.deadband)
            lam = self.lambda_map.get(s_ci.name, 0.0)
            lag_pen += lam * viol
            constr[s_ci.name] = ConstraintState(value=val, limit=s_ci.limit, violation=viol, lambda_val=lam)

        # 3.4 设备温度（软约束，硬约束由 rl_safety 先挡）——估算
        s_temp = self.specs.get("device_temp_max_c")
        if s_temp:
            val = est["max_device_temp_c"]
            viol = max(0.0, val - s_temp.limit - s_temp.deadband)
            lam = self.lambda_map.get(s_temp.name, 0.0)
            lag_pen += lam * viol
            constr[s_temp.name] = ConstraintState(value=val, limit=s_temp.limit, violation=viol, lambda_val=lam)

        total = obj + lag_pen

        return {
            "strategy_id": strategy.get("id"),
            "objective": {
                "config": asdict(self.cfg),
                "elec_cost": round(elec_cost, 4),
                "carbon_cost": round(carbon_cost, 4),
                "demand_cost": round(demand_cost, 4),
                "total_cost": round(total_cost, 4),
                "carbon_intensity": round(carbon_intensity, 6),
                "energy_kwh": round(energy_kwh, 4),
                "delay_min": round(delay_min, 4),
                "obj_value": round(obj, 4),
            },
            "constraints": {k: asdict(v) for k, v in constr.items()},
            "lagrangian_penalty": round(lag_pen, 4),
            "total_score": round(total, 4),  # 越小越好
            "explain": est["explain"],       # 可解释明细：每个动作贡献了什么
        }

    # ----- 公共：从反馈（仿真/实绩）更新乘子 -----
    def update_lambdas_from_feedback(
        self,
        measured: Dict[str, float],
        specs_override: Optional[Dict[str, ConstraintSpec]] = None,
        step_scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        典型用法：一次仿真/一段实际运行后，把测得的约束指标传进来（例如峰值/碳强度/延迟），
        这里按拉格朗日法更新 lambda：lambda_i <- max(0, lambda_i + step * (val - limit))
        """
        specs = specs_override or self.specs
        for name, val in measured.items():
            if name not in specs:
                continue
            spec = specs[name]
            lam = self.lambda_map.get(name, 0.0)
            step = min(self.alpha, spec.step or self.alpha) * float(step_scale)
            violation = max(0.0, val - spec.limit - spec.deadband)
            lam_new = max(0.0, lam + step * violation)
            self.lambda_map[name] = lam_new
        self._save_state()
        return dict(self.lambda_map)

    # ----- 公共：读取/设置/清空 乘子 -----
    def get_lambdas(self) -> Dict[str, float]:
        return dict(self.lambda_map)

    def set_lambdas(self, lambdas: Dict[str, float]) -> None:
        for k, v in (lambdas or {}).items():
            if k in self.specs:
                self.lambda_map[k] = max(0.0, float(v))
        self._save_state()

    def reset_lambdas(self) -> None:
        self.lambda_map = {k: 0.0 for k in self.specs.keys()}
        self._save_state()

    # ----- 内部：估计策略影响（无需仿真引擎的粗估） -----
    def _estimate_impacts(
        self, strategy: Dict[str, Any], ctx: Dict[str, Any], cf: CarbonFactor
    ) -> Dict[str, Any]:
        """
        估算策略对 kWh/峰值/延迟/吞吐/温度 的影响（简化版）：
        - energy_kwh: sum(kW * dt)
        - peak_kw_delta: 如果动作净效应增加负荷，则认为峰值增加；reduce/放电可降低峰值
        - delay_min/throughput_teu: 从 strategy.meta 或 ctx 中读取（没有则给默认）
        - baseline_peak_kw: 从 ctx/forecast 读，没有就给一个合理的常量（2000kW）
        - max_device_temp_c: 由电流近似估算（与 rl_safety 同口径但更保守）
        """
        window = strategy.get("window") or {}
        start = window.get("start")
        end = window.get("end")
        # 窗口长度（分钟）
        dt_min = max(1.0, self._minutes_between(start, end) or float(ctx.get("step_min", 60)))
        dt_h = dt_min / 60.0

        actions = list(strategy.get("actions") or [])
        net_kw = 0.0
        pos_kw = 0.0
        neg_kw = 0.0

        explain: List[Dict[str, Any]] = []
        max_temp = 30.0  # 环境温度基础

        for a in actions:
            kw = float(a.get("kW", 0.0))
            cmd = str(a.get("cmd", ""))
            # reduce 定义为降低负荷（负数）
            signed_kw = -abs(kw) if cmd == "reduce" else kw
            net_kw += signed_kw
            if signed_kw > 0:
                pos_kw += signed_kw
            else:
                neg_kw += abs(signed_kw)

            # 粗估温度：T = 30 + 0.1 * I，其中 I ≈ kW/(V*pf)*1000；pf=0.9, V=400
            I = abs(signed_kw) / (400.0 * 0.9) * 1000.0
            T = 30.0 + 0.1 * I
            max_temp = max(max_temp, T)

            explain.append({
                "asset": a.get("asset"),
                "cmd": cmd,
                "kW": kw,
                "signed_kW": round(signed_kw, 3),
                "est_temp_C": round(T, 2),
                "energy_kwh_contrib": round(signed_kw * dt_h, 4),
            })

        energy_kwh = net_kw * dt_h
        # 峰值：如果净效应是增加负荷，认为峰值可能上升（保守）；净减少则负的 delta
        peak_kw_delta = max(0.0, pos_kw - neg_kw)

        baseline_peak_kw = float(ctx.get("baseline_peak_kw", 2000.0))
        delay_min = float((strategy.get("meta") or {}).get("expected_delay_min", ctx.get("expected_delay_min", 0.0)))
        throughput_teu = float((strategy.get("meta") or {}).get("expected_throughput_teu", ctx.get("expected_throughput_teu", 0.0)))

        carbon_kg = max(0.0, energy_kwh) * cf.kg_per_kwh

        return {
            "energy_kwh": round(energy_kwh, 6),
            "peak_kw_delta": round(peak_kw_delta, 6),
            "baseline_peak_kw": round(baseline_peak_kw, 3),
            "delay_min": round(delay_min, 3),
            "throughput_teu": round(throughput_teu, 3),
            "carbon_kg": round(carbon_kg, 6),
            "max_device_temp_c": round(max_temp, 3),
            "explain": explain,
        }

    # ----- 内部：状态持久化（lambda） -----
    def _save_state(self) -> None:
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "lambdas": self.lambda_map,
            "specs": {k: asdict(v) for k, v in self.specs.items()},
            "objective": asdict(self.cfg),
        }
        path = Path(self.state_file)
        try:
            if self.storage:
                # 约定：storage 使用相对路径即可（内部会映射到 data/objects/...）
                parent = str(Path(self.state_file).parent)
                self.storage.ensure_dir(parent)
                self.storage.write_json(self.state_file, data)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # 忽略持久化错误，避免阻塞线上
            pass

    def _load_state(self) -> None:
        path = Path(self.state_file)
        data: Optional[Dict[str, Any]] = None
        try:
            if self.storage:
                data = self.storage.read_json(self.state_file)
            elif path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = None

        # 初始化 lambda（没有则默认 0）
        if data and isinstance(data.get("lambdas"), dict):
            self.lambda_map = {k: max(0.0, float(v)) for k, v in data["lambdas"].items()}
        else:
            self.lambda_map = {k: 0.0 for k in self.specs.keys()}

        # 恢复目标与 specs（允许现场覆盖）
        try:
            if data and "objective" in data:
                o = data["objective"]
                self.cfg = ObjectiveConfig(**o)
            if data and "specs" in data and isinstance(data["specs"], dict):
                self.specs = {k: ConstraintSpec(**v) for k, v in data["specs"].items()}
        except Exception:
            pass

    # ----- 工具 -----
    @staticmethod
    def _minutes_between(start_iso: Optional[str], end_iso: Optional[str]) -> Optional[float]:
        from datetime import datetime as dt
        try:
            if start_iso and end_iso:
                s = dt.fromisoformat(start_iso.replace("Z","+00:00"))
                e = dt.fromisoformat(end_iso.replace("Z","+00:00"))
                return (e - s).total_seconds() / 60.0
        except Exception:
            return None
        return None

    def _grid_limit_kw(self) -> float:
        """从 caps/grid 中读馈线限额（含 N-1 预留后），没有就返回默认 2800kW。"""
        grid = (self.caps or {}).get("grid") or {}
        feeder = float(grid.get("feeder_limit_kw", 3000.0))
        reserve = float(grid.get("nminus1_reserve_kw", 200.0))
        return max(0.0, feeder - reserve)
