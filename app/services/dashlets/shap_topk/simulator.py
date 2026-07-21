from __future__ import annotations
from typing import List, Dict, Optional
import random, math

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 5150)

def _pick_topk(cands: List[Dict[str, float]], k: int) -> List[Dict[str, float]]:
    # 按绝对值排序，取前K
    cands = sorted(cands, key=lambda x: abs(x["contribution"]), reverse=True)
    # 微小扰动避免完全并列
    for i, it in enumerate(cands): it["contribution"] += (1e-3*(len(cands)-i))
    return cands[:k]

def simulate_shap_topk(asset: str, k: int = 5, seed: Optional[int]=None) -> List[Dict[str, float]]:
    """
    资产特定的“解释特征”：
    - QC：电价、DR、作业强度、换班窗口、潮汐、阵风、队列长度
    - YC：电价、DR、堆场活跃、转运拥堵、夜班效率、风阵
    - BESS/shore：电价差、SOC、DR、备用合约、效率/退化惩罚、充电限流
    - HVAC/plant：室外温度、相对湿度、占用率、设定点偏移、冷机COP、日照辐射
    贡献值单位近似 kWh（+增耗、-节电），仅作演示。
    """
    rng = _rng(seed)
    a = asset.lower()

    # 定义候选特征与“期望符号/幅度范围”（均值±方差）
    cands: List[Dict[str, float]] = []

    def add(name: str, mean: float, sd: float):
        val = rng.gauss(mean, sd)
        cands.append({"name": name, "contribution": float(val)})

    if a.startswith(("qc","g_","port_g")):
        # QC：作业强度、价格、DR 易带来增耗；策略/避峰可能带来节电（负）
        add("作业强度指数",  +6.0, 2.0)
        add("电价(18-21)",  +3.5, 1.2)
        add("DR 响应",      -2.8, 1.0)   # 节电
        add("换班窗口(7/12/17)", +1.8, 0.8)
        add("潮汐极值邻近", +1.2, 0.7)
        add("阵风风险",     +1.0, 0.6)
        add("队列长度",     +2.5, 1.1)

    elif a.startswith(("yc","f_","port_f")):
        add("堆场活跃度",   +4.5, 1.6)
        add("电价(18-21)",  +2.8, 1.0)
        add("DR 响应",      -2.0, 0.9)   # 节电
        add("转运拥堵",     +2.2, 1.0)
        add("夜班效率",     -1.2, 0.7)   # 效率提升→节电
        add("阵风风险",     +0.9, 0.6)

    elif a.startswith(("bess","shore")):
        add("电价差(峰-谷)", -6.5, 2.0)  # 利用价差→节电/收益
        add("SOC 状态",      +2.0, 1.2)  # 高SOC可能限制/增耗
        add("DR 需求",       -2.8, 1.1)
        add("备用合约",      -1.6, 0.8)
        add("效率/退化惩罚", +1.3, 0.9)
        add("充电限流",      +0.9, 0.6)

    elif a.startswith(("hvac","plant","cool")):
        add("室外温度(OAT)", +5.0, 1.7)
        add("相对湿度(RH)",  +1.4, 0.8)
        add("占用率",        +2.2, 1.1)
        add("设定点下调",    -2.6, 1.1)  # 降设定点→节电
        add("冷机COP",       -1.8, 0.9)  # COP 高→用电少
        add("日照辐射",      +0.9, 0.6)

    else:
        # 其它资产：给一组通用特征
        add("电价",         +2.5, 1.0)
        add("DR 响应",     -2.0, 0.9)
        add("负荷指数",     +3.0, 1.5)
        add("效率提升",     -1.2, 0.7)
        add("外部扰动",     +1.0, 0.6)

    # 轻微零均值扰动，避免全正或全负
    for it in cands:
        it["contribution"] += rng.uniform(-0.4, 0.4)

    return _pick_topk(cands, k)
