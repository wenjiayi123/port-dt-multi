"""
守护栏预演 · 港口场景友好模拟器
- 端点：由 app/services/opsx/api.py 调用
    POST /api/opsx/guard/dryrun { "strategy_id": "S-341" } -> dryrun(strategy_id)

【大白话】
- 输入一个策略ID（或你未来传更多参数），我们估算它上线后
  可能触发的“守护规则”列表：rule（规则名）、level（info/warn/critical）、reason（原因）、ref（参考值）。
- 真实落地时，只需要把“TODO 真接入”替换为：
  1) 读取候选策略的关键配置（动作上限、阈值、可用设备状态等）
  2) 读取最近窗口的现场特征（峰/谷、CO₂ 强度、DR 窗口、设备占空比等）
  3) 用你们的 Guard 引擎实际跑一遍，返回触发清单
"""

from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
import hashlib, random, math
from datetime import datetime

# ============== 一些“港口现场规则”模板（可自行增删） ==============
@dataclass
class Context:
    # 现场上下文（可从 TSDB/配置中心来；这里先用策略ID派生出来的稳定随机数模拟）
    peak_kw: float                # 预测尖峰功率（kW）
    grid_limit_kw: float          # 并网限额（kW）
    bess_discharge_limit_kw: float# BESS 最大放电功率（kW）
    shore_power_plan_kw: float    # 预计岸电负荷（kW）
    duty_cycle_est: float         # 关键设备占空比（0~1）
    co2_intensity: float          # 当前电网碳强度（kg/kWh）
    next_dr: str                  # 下一次 DR 窗口
    max_delta_p_kw: float         # 单步最大动作幅度（kW）
    comfort_min_temp_c: float     # 最低舒适温度（HVAC约束）

def _seed(sid: str) -> random.Random:
    # 用 strategy_id 生成稳定随机种子 -> 让同一个ID每次结果一致
    h = hashlib.sha256(sid.encode("utf-8")).hexdigest()
    return random.Random(int(h[:12], 16))

def _fake_context(sid: str) -> Context:
    r = _seed(sid)
    return Context(
        peak_kw = r.uniform(10500, 14500),
        grid_limit_kw = 15000.0,
        bess_discharge_limit_kw = r.uniform(1800, 2600),
        shore_power_plan_kw = r.uniform(3000, 7000),
        duty_cycle_est = max(0.6, min(0.99, r.gauss(0.88, 0.05))),
        co2_intensity = max(0.25, min(0.80, r.gauss(0.46, 0.08))),  # kg/kWh
        next_dr = "17:00-18:00 reserve up",
        max_delta_p_kw = r.uniform(1200, 1800),
        comfort_min_temp_c = 18.0
    )

def _rule(name: str, ok: bool, level_if_fail: str, reason: str, ref: Dict[str, Any]) -> Dict[str, Any]:
    return {"rule": name, "level": ("info" if ok else level_if_fail), "reason": reason, "ref": ref}

def dryrun(strategy_id: str) -> Dict[str, Any]:
    """
    预演守护规则触发情况
    - 真实落地：
      TODO 真接入(读)：从“候选策略仓库/配置中心”读取该 strategy_id 的动作上限、偏好、护栏设定；
                       从 TSDB/OLAP 读取窗口内现场统计（峰、DR、设备温度、占空比、岸电功率等）。
      TODO 真接入(算)：调用你们现有 Guard 引擎对“下一小时/下一窗口”进行校验，收集触发的规则。
    """
    sid = (strategy_id or "demo").strip()
    ctx = _fake_context(sid)

    # ====== 估算策略特征（这里用策略ID派生一个“动作意图”）======
    r = _seed(sid)
    intent_peak_bias   = r.uniform(0.2, 0.8)   # 削峰的“偏好” 0~1
    intent_cost_bias   = r.uniform(0.2, 0.8)   # 降本的“偏好”
    intent_carbon_bias = r.uniform(0.2, 0.8)   # 降碳的“偏好”
    # 预估本策略单步动作幅度（kW），越偏向 peak/cost，越可能动作大
    action_delta_kw = (800 + 1600*intent_peak_bias + 600*intent_cost_bias) * r.uniform(0.85, 1.15)

    # ====== 逐条规则检查（可以按需扩展/调整阈值）======
    rules: List[Dict[str, Any]] = []

    # R1 ΔkWh >= 0 或 ΔPeak >= 阈（二选一），否则没有实质收益
    # 这里简单用“存在动作且峰值改善超过 3% 或能耗非负”做一个预估
    delta_peak_ratio = max(0.0, min(0.10, 0.02 + 0.08*intent_peak_bias))  # 预估削峰比例 2%~10%
    ok_r1 = (delta_peak_ratio >= 0.03) or True  # 允许 ΔkWh≥0 的宽松条件（演示）
    rules.append(_rule(
        "ΔkWh≥0 或 ΔPeak≥阈",
        ok=ok_r1,
        level_if_fail="warn",
        reason="保证至少一个目标达成（节能或削峰）",
        ref={"delta_peak_ratio_est": round(delta_peak_ratio,3)}
    ))

    # R2 设备占空比 ≤ 95%
    ok_r2 = (ctx.duty_cycle_est <= 0.95)
    rules.append(_rule(
        "设备占空比 ≤ 95%",
        ok=ok_r2,
        level_if_fail="warn",
        reason="避免长时间顶格运行影响寿命/温度",
        ref={"duty_cycle_est": round(ctx.duty_cycle_est,3), "limit":0.95}
    ))

    # R3 岸电功率 ≤ BESS + 并网限额（容量约束）
    ok_r3 = (ctx.shore_power_plan_kw <= (ctx.bess_discharge_limit_kw + ctx.grid_limit_kw))
    rules.append(_rule(
        "岸电功率 ≤ BESS + Grid 限额",
        ok=ok_r3,
        level_if_fail="critical",
        reason="超出容量上限存在保护性跳闸风险",
        ref={"shore_power_plan_kw": int(ctx.shore_power_plan_kw),
             "cap_sum_kw": int(ctx.bess_discharge_limit_kw + ctx.grid_limit_kw)}
    ))

    # R4 单步动作幅度 ≤ Max ΔP（避免动作过猛）
    ok_r4 = (action_delta_kw <= ctx.max_delta_p_kw)
    rules.append(_rule(
        "单步动作幅度 ≤ Max ΔP",
        ok=ok_r4,
        level_if_fail="warn",
        reason="过猛的功率跃迁可能引发保护或舒适性问题",
        ref={"action_delta_kw_est": int(action_delta_kw), "max_delta_p_kw": int(ctx.max_delta_p_kw)}
    ))

    # R5 安全/舒适温度不越界（示例：HVAC）
    # 这里仅示意，以 17°C 为更硬的保底
    min_allowed = min(17.0, ctx.comfort_min_temp_c)
    ok_r5 = (ctx.comfort_min_temp_c >= min_allowed)
    rules.append(_rule(
        "舒适/安全温度不越界",
        ok=ok_r5,
        level_if_fail="critical",
        reason="确保人/设备舒适与安全边界",
        ref={"comfort_min_temp_c": ctx.comfort_min_temp_c, "min_allowed_c": min_allowed}
    ))

    # R6 CO₂ 强度窗口偏高时避免“过度动作”（示例：若电网碳强度高且动作很大 -> 警告）
    ok_r6 = not (ctx.co2_intensity > 0.60 and action_delta_kw > 1500)
    rules.append(_rule(
        "高碳强度窗口下限制大动作",
        ok=ok_r6,
        level_if_fail="warn",
        reason="电网碳强度偏高时避免不必要的功率调度",
        ref={"co2_intensity_kg_per_kwh": round(ctx.co2_intensity,3), "delta_kw_est": int(action_delta_kw)}
    ))

    # 汇总
    result = {
        "strategy_id": sid,
        "window": {"next_dr": ctx.next_dr, "ts": datetime.utcnow().isoformat()},
        "rules": [r for r in rules if r["level"] != "info"],  # 只返回触发/需关注项；如想看全部可不筛
        "context": {
            "peak_kw": int(ctx.peak_kw),
            "grid_limit_kw": int(ctx.grid_limit_kw),
            "bess_discharge_limit_kw": int(ctx.bess_discharge_limit_kw),
            "shore_power_plan_kw": int(ctx.shore_power_plan_kw),
            "duty_cycle_est": round(ctx.duty_cycle_est,3),
            "co2_intensity": round(ctx.co2_intensity,3),
            "max_delta_p_kw": int(ctx.max_delta_p_kw)
        }
    }
    return result
