"""
多目标权衡（Objective Try）· 港口场景友好模拟器
- 端点：由 app/services/opsx/api.py 调用
    POST /api/opsx/objective/try  { "cost":0.34, "peak":0.33, "carbon":0.33 }
    -> try_objective(weights)

【大白话】
- 前端的三条滑块只是“偏好”。这里把偏好正规化（归一和裁剪），
  然后给出一个“估计效果”的预览，包括：
  1) 经济成本下降百分比（delta_cost_%，负数=下降更省钱）
  2) 削峰百分比（delta_peak_%，负数=峰更低）
  3) 碳强度/碳排下降百分比（delta_co2_%，负数=更低）
  4) 门槛预测：预计 Guard 拦截率、SLA 违约率，并对照质量门槛判断是否安全（gates_ok）

- 真接入时，把“TODO 真接入(读/写/算)”位置替换成：
  - 从策略引擎/仿真器拿快速评估（如 RL propose/sandbox sim）
  - 或从历史 A/B 拟合的代理模型（surrogate）计算预估效果
"""

from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass
import math

# ========== 默认门槛（若 quality_gate 模块可用会优先取那个） ==========
DEFAULT_THRESHOLDS = {
    "mape_energy_max": 0.05,
    "guard_block_rate_max": 0.05,
    "sla_violation_rate_max": 0.02
}

def _load_thresholds() -> Dict[str, float]:
    """优先从 quality_gate 模块拿阈值，拿不到就用默认值"""
    try:
        from app.services.opsx.quality_gate.simulator import THRESHOLDS  # type: ignore
        return dict(THRESHOLDS)
    except Exception:
        return dict(DEFAULT_THRESHOLDS)

@dataclass
class Weights:
    cost: float
    peak: float
    carbon: float

def _clamp01(x: float) -> float:
    return 0.0 if math.isnan(x) else max(0.0, min(1.0, float(x)))

def _normalize(w: Dict[str, float]) -> Weights:
    # 把任意输入归一（负数裁剪为0，sum为0时回到均分）
    c = _clamp01(float(w.get("cost", 0.34)))
    p = _clamp01(float(w.get("peak", 0.33)))
    r = _clamp01(float(w.get("carbon", 0.33)))
    s = c + p + r
    if s <= 1e-9:
        c, p, r = 1/3, 1/3, 1/3
    else:
        c, p, r = c/s, p/s, r/s
    return Weights(cost=c, peak=p, carbon=r)

def _estimate_impacts(w: Weights) -> Dict[str, float]:
    """
    经验型代理模型（便于前端预览）：
    - cost 侧边际：~2%/100%权重
    - peak 侧边际：~6%/100%权重
    - carbon 侧边际：~4%/100%权重
    - 有一点“冲突惩罚”：当三者都很高时，整体收益会打折
    """
    IMP = {"cost": 2.0, "peak": 6.0, "carbon": 4.0}  # 百分比点
    # 冲突惩罚：总权重越“尖锐”，折扣越小；越“平均”时折扣也小。这里用简单二次项做下限折扣。
    harmony = (w.cost**2 + w.peak**2 + w.carbon**2)  # 0.33/0.33/0.33 -> ~0.333
    discount = 0.85 + 0.15 * harmony                  # 0.85~1.0 之间
    return {
        "delta_cost_%":  - round(IMP["cost"]   * w.cost   * discount, 3),
        "delta_peak_%":  - round(IMP["peak"]   * w.peak   * discount, 3),
        "delta_co2_%":   - round(IMP["carbon"] * w.carbon * discount, 3)
    }

def _predict_gates(w: Weights, th: Dict[str, float]) -> Dict[str, Any]:
    """
    预测“守护拦截率/违约率”随偏好的趋势（极简启发式）：
    - peak 欲望越强，越可能“顶格”操作 -> guard、sla 风险上升
    - cost 侧侧重过高时，也可能触发保守策略的边界（guard 略升）
    - carbon 一般相对温和
    """
    base_guard = 0.010   # 1.0%
    base_sla   = 0.004   # 0.4%
    guard = base_guard + 0.05 * max(0.0, w.peak - 0.45) + 0.02 * max(0.0, w.cost - 0.60)
    sla   = base_sla   + 0.03 * max(0.0, w.peak - 0.55) + 0.01 * max(0.0, w.cost - 0.65)
    guard = round(guard, 5); sla = round(sla, 5)

    gates_ok = (guard <= th.get("guard_block_rate_max", 0.05)
                and sla <= th.get("sla_violation_rate_max", 0.02))
    notes = []
    if not gates_ok:
        if guard > th.get("guard_block_rate_max", 0.05):
            notes.append("Guard 预计偏高，建议降低削峰或成本偏好")
        if sla > th.get("sla_violation_rate_max", 0.02):
            notes.append("SLA 违约风险偏高，建议降低削峰偏好或提高稳定性约束")

    return {
        "predict": {"guard_block_rate": guard, "sla_violation_rate": sla},
        "thresholds": th,
        "gates_ok": gates_ok,
        "notes": notes
    }

def try_objective(weights: Dict[str, float]) -> Dict[str, Any]:
    """
    主函数：把偏好 -> 预估影响 + 门槛校验
    - 真接入(算)：在这里调用你的“快速仿真/评估”接口替代 _estimate_impacts/_predict_gates
    """
    w = _normalize(weights)
    th = _load_thresholds()
    est = _estimate_impacts(w)
    gates = _predict_gates(w, th)

    preview = {
        "weights": {"cost": round(w.cost, 4), "peak": round(w.peak, 4), "carbon": round(w.carbon, 4)},
        "estimate": est,
        "gates": gates
    }

    return {"ok": True, "preview": preview}
