from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Optional
import math, random

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 3107)

def _gauss(x, mu, sigma):
    return math.exp(-((x - mu) ** 2) / (2 * sigma * sigma))

def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def simulate_residual_heatmap(asset: str, start: datetime, end: datetime,
                              seed: Optional[int]=None) -> List[List[float]]:
    """
    逼真港口假设：
    - QC（岸桥）：白班 8-12、13-17 负荷上升 → 预测难度↑；换班窗口(7-8,12-13,17-18)扰动↑；
      潮汐（~12h周期）在极值附近（涨/落潮顶）对靠泊/作业有影响 → 误差↑；
      DR/限电 14-16 → 策略扰动 ↑。
    - YC（场桥）：傍晚/夜间 17-22 活跃度↑。
    - BESS/shore：受电价/DR & 叠栈策略影响，14-22 偏高。
    - HVAC/plant：午后 12-18 因温控/负荷耦合，误差偏高。
    返回：7 行（周一到周日）、24 列（0..23 点），值∈[0,1]，数值越大越“红”。
    """
    rng = _rng(seed)
    asset_l = asset.lower()
    is_qc   = asset_l.startswith(("qc", "g_", "port_g"))
    is_yc   = asset_l.startswith(("yc", "f_", "port_f"))
    is_bess = asset_l.startswith(("bess", "shore"))
    is_hvac = asset_l.startswith(("hvac", "plant", "cool"))

    # 基线（越小越绿）
    base_level = 0.25
    if is_qc:   base_level = 0.30
    if is_yc:   base_level = 0.28
    if is_bess: base_level = 0.26
    if is_hvac: base_level = 0.27

    # 计算“潮汐相位”: 以 start 为参考，每 12h 一个周期
    # 用 sin^2 表示“接近极值越大”
    def tide_bump(hour_index: int) -> float:
        # hour_index 从 start 起算的小时数
        phase = (hour_index % 12) / 12.0  # 0..1
        s = math.sin(2 * math.pi * phase)
        return abs(s) ** 2  # [0,1]

    grid: List[List[float]] = [[0.0]*24 for _ in range(7)]

    # 以 start 当周为参照（行号 0=周一）
    week_start = start - timedelta(days=(start.weekday()))
    for d in range(7):
        day = week_start + timedelta(days=d)
        weekday_factor = 1.0 if d < 5 else 0.8  # 周末整体更“绿”
        for h in range(24):
            v = base_level * weekday_factor
            hour = h

            # —— 不同设备的小时型谱 —— #
            if is_qc:
                # 白班两个高斯峰（10:30 与 15:00）
                v += 0.22 * _gauss(hour, 10.5, 2.0)
                v += 0.18 * _gauss(hour, 15.0, 2.0)
                # 换班窗口波动（7-8, 12-13, 17-18）
                if hour in (7,8,12,13,17,18): v += 0.08
            elif is_yc:
                # 傍晚/夜间作业更活跃
                v += 0.20 * _gauss(hour, 19.0, 2.5)
                v += 0.12 * _gauss(hour, 22.0, 2.0)
            elif is_bess:
                # DR/峰价段 14-22 误差略高（叠栈策略切换）
                if 14 <= hour <= 22: v += 0.12
            elif is_hvac:
                # 午后温控耦合
                v += 0.18 * _gauss(hour, 15.0, 2.5)

            # DR 窗口（14-16）：所有资产都可能被扰动
            if 14 <= hour <= 16: v += 0.06

            # 潮汐影响（QC 明显，其它次之）
            hours_from_start = int((day - start).total_seconds()//3600) + h
            v += (0.10 if is_qc else 0.04) * tide_bump(hours_from_start)

            # 随机扰动（天/小时级）
            v += rng.uniform(-0.02, 0.02)
            v = _clamp01(v)

            grid[d][h] = float(v)

    # 轻微平滑（卷积 3×3），避免棋盘格
    def smooth(g):
        out = [[0.0]*24 for _ in range(7)]
        for i in range(7):
            for j in range(24):
                acc = 0.0; cnt = 0
                for di in (-1,0,1):
                    for dj in (-1,0,1):
                        ii = (i+di) % 7
                        jj = (j+dj) % 24
                        acc += g[ii][jj]; cnt += 1
                out[i][j] = _clamp01(acc / cnt)
        return out
    grid = smooth(grid)

    # 保底/封顶，避免纯 0 或 1
    for i in range(7):
        for j in range(24):
            grid[i][j] = float(min(0.95, max(0.05, grid[i][j])))

    return grid
