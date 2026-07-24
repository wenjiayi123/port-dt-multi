"""
策略画像（Radar）· 港口场景友好模拟器
- 端点：由 app/services/opsx/api.py 调用
    GET /api/opsx/profile -> get_profile()

【大白话】
- 返回 0~1 的五维画像：动作强度、守护命中、稳定性、收益、风险(反向)。
- 前端当前只用 "values" 数组(长度=5)。我额外带了 "labels" 和 "raw"（原始KPI），方便你后续把图表做得更“聪明”。
- 真接入时，只要把“TODO 真接入(读)”的位置换成你们的真实KPI（来自 A/B结果、TSDB 指标、RL引擎日志等），
  然后走 normalize() 归一化即可。
"""
from __future__ import annotations
from typing import Dict, Any
from datetime import datetime
import random
import math

# ============== 内部可变状态（用于让画像“平滑抖动”，避免跳变） ==============
_STATE = {
    # 原始KPI（示例），真实接入时这些应来自你们的指标库 / 日志聚合
    "raw": {
        "action_amplitude": 0.65,   # 动作强度（0~1，越大越激进）
        "guard_block_rate": 0.010,  # 守护拦截率（0~1，越小越好）
        "stability_score" : 0.72,   # 稳定性（0~1，越大越稳定）
        "economics_gain"  : 0.66,   # 经济收益（0~1，越大越好）
        "risk_rate"       : 0.18,   # 风险率（0~1，越小越好）
    },
    "last_values": [0.62, 0.55, 0.70, 0.66, 0.40],  # 与前端顺序对应的初值
    "updated_at": datetime.utcnow().isoformat()
}

# ============== 小工具：截断/平滑/归一 ==============
def _clamp01(x: float) -> float:
    return 0.0 if math.isnan(x) else max(0.0, min(1.0, x))

def _ema(prev: float, new: float, alpha: float = 0.2) -> float:
    """指数平滑，避免每次刷新时画像大起大落"""
    return (1 - alpha) * prev + alpha * new

def _jitter(x: float, amp: float = 0.03) -> float:
    """轻微随机抖动"""
    return _clamp01(x + random.uniform(-amp, amp))

def _inv(x: float) -> float:
    """反向指标映射（例如风险率越小越好 -> 分数越高）"""
    return _clamp01(1.0 - x)

# ============== 真接入时：在这里读你的“原始 KPI”并覆盖 _STATE["raw"] ==============
def _pull_real_kpi() -> Dict[str, float]:
    """
    TODO 真接入(读)：
    - 示例：从 TSDB/OLAP 读取过去15~30分钟窗口内的KPI：
      action_amplitude: RL 动作强度（可取动作范数/比例）
      guard_block_rate: 守护规则拦截率
      stability_score : 策略稳定性（动作方差/切换频率的反向映射）
      economics_gain  : 相对基线的节能/经济收益归一化值
      risk_rate       : 报警/越限/违约等风险事件率
    - 返回值范围建议都归一在[0,1]，若不是，也可以在下方 normalize 里做映射
    """
    raw = _STATE["raw"]

    # 为了更有“生命力”，这里给原始 KPI 叠加一点轻微随机波动
    raw["action_amplitude"] = _clamp01(raw["action_amplitude"] + random.uniform(-0.02, 0.02))
    raw["guard_block_rate"] = _clamp01(raw["guard_block_rate"] + random.uniform(-0.002, 0.002))
    raw["stability_score"]  = _clamp01(raw["stability_score"]  + random.uniform(-0.02, 0.02))
    raw["economics_gain"]   = _clamp01(raw["economics_gain"]   + random.uniform(-0.02, 0.02))
    raw["risk_rate"]        = _clamp01(raw["risk_rate"]        + random.uniform(-0.02, 0.02))

    return dict(raw)

# ============== 画像主函数：把原始KPI -> 0~1的五维画像 ==============
def get_profile() -> Dict[str, Any]:
    raw = _pull_real_kpi()

    # 1) 动作强度：直接使用（或映射到 [0.2, 1.0] 以区分度更强）
    action_strength = _clamp01(raw["action_amplitude"])

    # 2) 守护命中：这里理解为“低拦截率 => 高命中分”
    #    例：0%拦截 -> 1.0；5%拦截 -> ~0.0（按经验可调 5%~8%为红线）
    guard_rate = raw["guard_block_rate"]
    guard_hit  = _clamp01(1.0 - guard_rate / 0.08)  # 8% 作为最差参考

    # 3) 稳定性：直接使用（或结合动作切换频率/方差做映射）
    stability = _clamp01(raw["stability_score"])

    # 4) 收益：直接使用（你也可以用 ΔkWh/ΔUSD 的归一化）
    economics = _clamp01(raw["economics_gain"])

    # 5) 风险(反向)：根据风险率做反向映射（0风险=1分；≥40%风险≈0分）
    risk_inverse = _clamp01(1.0 - (raw["risk_rate"] / 0.40))

    # 轻微抖动 + 指数平滑，避免图形跳动太猛
    target = [
        _jitter(action_strength, 0.02),
        _jitter(guard_hit, 0.02),
        _jitter(stability, 0.02),
        _jitter(economics, 0.02),
        _jitter(risk_inverse, 0.02),
    ]
    smooth = []
    for prev, new in zip(_STATE["last_values"], target):
        smooth.append(_ema(prev, new, alpha=0.25))

    _STATE["last_values"] = smooth
    _STATE["updated_at"] = datetime.utcnow().isoformat()

    return {
        # 前端雷达用的主字段（顺序固定）
        "values": [round(v, 4) for v in smooth],

        # 额外：方便你后续做提示/解释
        "labels": ["动作强度", "守护命中", "稳定性", "收益", "风险(反向)"],
        "raw": raw,  # 原始KPI，便于溯源或做二次映射
        "updated_at": _STATE["updated_at"],

        # 建议的阈值/解释（可在 UI 提示）
        "explain": {
            "action_strength": "策略动作幅度（0~1），越高越激进，需与风控联动",
            "guard_hit":       "守护规则低拦截 -> 高命中分；>5%通常需要关注",
            "stability":       "动作平滑/低抖动更稳；可由切换频率/方差计算",
            "economics":       "节能/经济收益归一化值",
            "risk_inverse":    "风险率越低越好（报警/越限/违约等）"
        }
    }
