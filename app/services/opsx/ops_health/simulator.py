"""
运维健康度（Health）· 港口场景友好模拟器
- 端点：由 app/services/opsx/api.py 调用
    GET /api/opsx/health  -> get_health()

【大白话】
- 返回整体健康分（0~100）+ 漂移 TopN（PSI 指标）。分越高代表越“健康”（漂移越小）。
- PSI（Population Stability Index）简化版：用于衡量“当前分布 vs. 基线分布”的偏移，
  常见经验：0.1~0.25 为轻/中度漂移，>0.25 需关注。
- 真接入时，把“TODO 真接入(读)”处替换为从 TSDB/OLAP 拉过去 X 分钟窗口数据：
  - 基线分布：可选“近30天同班同时段”、或“策略上线前一周”；
  - 当前分布：近15分钟或近1小时。
- 该文件是自给自足的：第一次调用会生成“基线”；后续每次返回“当前窗口”+ 轻微抖动。
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import math, random

# ======== 配置：窗口 & 监控特征 ========
BASELINE_MIN = 60 * 24 * 7   # 基线：过去7天（示意；模拟器内部只做一次性初始化）
CURRENT_MIN  = 60            # 当前窗口：过去1小时（示意）
FEATURES = [
    "market_price",   # 电价（USD/MWh）
    "active_power",   # 有功功率（kW）
    "queue_len",      # 堆场排队长度
    "co2_intensity",  # 电网碳强度（kg/kWh）
    "temp_ahu"        # 空调送风温度（°C）
]

# ======== 内部状态：基线统计 & 最近输出 ========
_STATE: Dict[str, Any] = {
    "baseline": None,     # {"feature": [values...]} 仅一次初始化
    "last_score": 86,
    "updated_at": None
}

# ======== 工具：生成分布（模拟器用；真接入时替换为 TSDB 查询） ========
def _gen_baseline() -> Dict[str, List[float]]:
    r = random.Random(20251024)  # 固定种子：基线稳定
    return {
        "market_price": [r.gauss(78, 12)  for _ in range(2400)],  # ~ 78±12
        "active_power": [max(0, r.gauss(5200, 900)) for _ in range(2400)],
        "queue_len":    [max(0, r.gauss(8, 3))      for _ in range(2400)],
        "co2_intensity":[max(0.2, r.gauss(0.46,0.08)) for _ in range(2400)],
        "temp_ahu":     [r.gauss(19.5, 1.2)         for _ in range(2400)],
    }

def _gen_current_from_baseline(base: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    模拟“当前一小时”分布：在基线基础上叠加轻度漂移（不同特征幅度不同）
    真接入：改为查询近 CURRENT_MIN 的值即可
    """
    cur: Dict[str, List[float]] = {}
    for k, arr in base.items():
        mu = sum(arr)/len(arr)
        sd = (sum((x-mu)**2 for x in arr)/len(arr))**0.5
        # 设一点“现场扰动”：不同特征偏移不同方向/幅度
        if k == "market_price": shift = 0.15*sd
        elif k == "active_power": shift = -0.10*sd
        elif k == "queue_len": shift = 0.05*sd
        elif k == "co2_intensity": shift = 0.08*sd
        else: shift = 0.02*sd
        cur[k] = [max(0, random.gauss(mu+shift, sd*1.05)) for _ in range(240)]
    return cur

# ======== PSI 计算（简化稳定实现） ========
def _hist(values: List[float], bins: List[float]) -> List[int]:
    """基于分位点的分箱"""
    cnt = [0]* (len(bins)+1)
    for v in values:
        i = 0
        while i < len(bins) and v > bins[i]: i += 1
        cnt[i] += 1
    return cnt

def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / (b + eps)

def _psi(ref: List[float], cur: List[float], nbins: int = 10) -> Tuple[float, Dict[str, float]]:
    """
    计算 Population Stability Index
    1) 用“参考分布”的分位点作为分箱边界
    2) 统计两边频率 p_i, q_i
    3) PSI = sum( (q_i - p_i) * ln(q_i/p_i) )
    返回 (psi, {"p50":..., "ref_p50":...}) 供前端提示方向
    """
    if not ref or not cur: return 0.0, {"p50":0.0, "ref_p50":0.0}
    ref_sorted = sorted(ref)
    bins = [ref_sorted[int(len(ref_sorted)*(i/(nbins)))] for i in range(1, nbins)]
    h_ref = _hist(ref, bins); h_cur = _hist(cur, bins)
    nref, ncur = float(sum(h_ref)), float(sum(h_cur))
    psi = 0.0
    for i in range(nbins):
        p = _safe_div(h_ref[i], nref)
        q = _safe_div(h_cur[i], ncur)
        p = max(p, 1e-6); q = max(q, 1e-6)
        psi += (q - p) * math.log(q/p)
    # 中位数（粗略，用于“方向”提示）
    mid = len(ref_sorted)//2
    ref_p50 = ref_sorted[mid]
    cur_sorted = sorted(cur)
    p50 = cur_sorted[len(cur_sorted)//2]
    return float(psi), {"p50": float(p50), "ref_p50": float(ref_p50)}

# ======== 健康分映射 ========
def _score_from_psis(psis: List[float]) -> int:
    """
    把多特征的 psi 合成一个 0~100 分：psi 越大 => 分越低
    简单经验映射：score = 100 - scale * sum(psi)，取 0~100 之间
    """
    spsi = sum(psis)
    scale = 35.0  # 可按经验调；漂移 0.4 左右会压到 ~86 分
    raw = 100.0 - scale * spsi
    return int(max(0, min(100, round(raw))))

# ======== 主函数 ========
def get_health() -> Dict[str, Any]:
    # 初始化基线
    if _STATE["baseline"] is None:
        _STATE["baseline"] = _gen_baseline()

    base = _STATE["baseline"]
    cur  = _gen_current_from_baseline(base)  # TODO 真接入(读)：改为查询 CURRENT_MIN 窗口

    # 逐特征 PSI
    items = []
    for feat in FEATURES:
        psi, extra = _psi(base[feat], cur[feat], nbins=10)
        direction = "up" if extra["p50"] > extra["ref_p50"] else "down"
        items.append({
            "feature": feat,
            "psi": round(psi, 3),
            "direction": direction,
            "p50": round(extra["p50"], 3),
            "ref_p50": round(extra["ref_p50"], 3)
        })

    # Top3 按 PSI 降序
    items.sort(key=lambda x: x["psi"], reverse=True)
    top3 = items[:3]

    # 健康分 & 备注
    score = _score_from_psis([x["psi"] for x in items])
    notes = []
    for x in items:
        if x["psi"] >= 0.25:
            notes.append(f"特征 {x['feature']} 漂移较大(PSI={x['psi']})，建议复核数据或回归测试")

    _STATE["last_score"] = score
    _STATE["updated_at"] = datetime.utcnow().isoformat()

    return {
        "score": score,                  # 前端当前使用
        "top_drifts": top3,              # 前端当前使用
        "all": items,                    # Complete collection for extended charts and dialogs
        "baseline_window_min": BASELINE_MIN,
        "current_window_min": CURRENT_MIN,
        "updated_at": _STATE["updated_at"],
        "notes": notes
    }
