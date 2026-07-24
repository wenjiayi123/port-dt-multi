"""
质量门槛 Gate · 港口场景友好模拟器
- 端点（由 app/services/opsx/api.py 调用）：
    GET /api/opsx/gates -> get_gates()

【大白话】
- 返回三类门槛：MAPE（预测误差）、Guard 拦截率、SLA 违约率。
- 除了前端当前用到的三个数值，我再带上了“指标窗口”、“样本数”、“分模块详细”等字段，
  方便你后面直接把真源接进来（TSDB、Flink、OLAP 或你们的离线统计）。
- 真接入时，把“TODO 真接入”处替换为实际读取逻辑即可。
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

# ======== 配置：窗口/阈值（你后续可以改为从 DB/配置中心读取） ========
WINDOW_MIN = 15   # 统计窗口（分钟）
THRESHOLDS = {
    "mape_energy_max": 0.05,            # 预测能耗 MAPE 最多 5%
    "guard_block_rate_max": 0.05,       # 守护规则拦截率最多 5%
    "sla_violation_rate_max": 0.02      # SLA 违约率最多 2%
}

# ======== 模拟的“分模块”评估数据（真实落地时改为读库/读流） ========
_MODULES = ["agv_charge", "yard_crane", "shore_bess"]

def _fake_series(n: int = 60, base: float = 100.0, noise: float = 3.0) -> List[float]:
    """
    生成 n 个点的时间序列：base 附近随机波动（不依赖 numpy，方便部署）
    """
    x = base
    out = []
    for _ in range(n):
        x += random.uniform(-noise, noise)
        out.append(max(0.0, x))
    return out

def _mape(actual: List[float], pred: List[float]) -> float:
    """
    计算简化版 MAPE（避免除零，实际可用更健壮实现）
    """
    s = 0.0
    c = 0
    for a, p in zip(actual, pred):
        if a is None or a == 0:  # 跳过 0
            continue
        s += abs(a - p) / a
        c += 1
    return s / max(1, c)

def _guard_block_rate(total_checks: int, blocked: int) -> float:
    total = max(1, int(total_checks))
    blocked = max(0, min(blocked, total))
    return blocked / total

def _sla_violation_rate(total_jobs: int, violations: int) -> float:
    total = max(1, int(total_jobs))
    violations = max(0, min(violations, total))
    return violations / total

def _module_detail() -> Dict[str, Any]:
    """
    构造每个模块的指标（便于你前端后面做展开/heatmap/火焰图）
    - 真接入：把这里替换为各模块的真实统计（例如 ClickHouse/Timescale/Flink 输出）
    """
    detail = {}
    now = datetime.utcnow()
    for m in _MODULES:
        # 模拟能耗实际与预测
        actual = _fake_series(60, base=100 + random.uniform(-10, 10), noise=2.5)
        pred   = [v * (1.0 + random.uniform(-0.02, 0.02)) for v in actual]

        # 守护规则/违约统计（每分钟1~4个检查/作业）
        checks = sum(random.randint(1, 4) for _ in range(WINDOW_MIN))
        blocked = int(checks * max(0.0, random.gauss(0.012, 0.006)))  # ~1.2%±
        jobs = sum(random.randint(1, 3) for _ in range(WINDOW_MIN))
        violations = int(jobs * max(0.0, random.gauss(0.004, 0.003)))  # ~0.4%±

        detail[m] = {
            "mape": round(_mape(actual, pred), 5),
            "guard_block_rate": round(_guard_block_rate(checks, blocked), 5),
            "sla_violation_rate": round(_sla_violation_rate(jobs, violations), 5),
            # 附带缩略趋势，便于后续画小火焰/迷你图
            "spark": {
                "mape": [round(v, 4) for v in _fake_series(12, base=3.0, noise=0.7)],   # 伪装成百分数的 *100 之前值
                "guard": [round(abs(random.gauss(1.2, 0.4)), 3) for _ in range(12)],    # 千分比近似
                "sla":   [round(abs(random.gauss(0.4, 0.2)), 3) for _ in range(12)],
            },
        }
    return detail

def get_gates() -> Dict[str, Any]:
    """
    返回质量门槛汇总（给 GET /api/opsx/gates）
    - 前端当前只用 metrics 和 thresholds 三个数字，其它字段为扩展（未来可用）
    """
    # TODO 真接入(读)：从 TSDB/OLAP 里计算或读取窗口内汇总值
    detail = _module_detail()
    # 汇总为“全局”指标
    mape = sum(d["mape"] for d in detail.values()) / len(detail)
    guard = sum(d["guard_block_rate"] for d in detail.values()) / len(detail)
    sla = sum(d["sla_violation_rate"] for d in detail.values()) / len(detail)

    # 轻微抖动，避免静止
    mape = max(0.0, mape + random.uniform(-0.002, 0.002))
    guard = max(0.0, guard + random.uniform(-0.002, 0.002))
    sla   = max(0.0, sla + random.uniform(-0.001, 0.001))

    return {
        "window": {"minutes": WINDOW_MIN, "end_utc": datetime.utcnow().isoformat()},
        "sample_count": {"checks": sum(random.randint(60, 90) for _ in _MODULES),
                         "jobs":   sum(random.randint(50, 80) for _ in _MODULES)},
        "metrics": {
            # 前端正在用的三项（保持字段名一致）
            "mape": round(float(mape), 5),
            "guard": round(float(guard), 5),
            "sla": round(float(sla), 5),
        },
        "thresholds": dict(THRESHOLDS),
        # 额外信息（目前前端未用，但保留便于你后续图表增强）
        "by_module": detail,
        "explain": {
            "mape": "能耗/负荷预测相对误差的平均值（绝对值），越低越好",
            "guard": "守护规则拦截的占比，越低越好（高说明策略经常被挡下）",
            "sla": "违反服务等级协议的作业占比，越低越好"
        }
    }
