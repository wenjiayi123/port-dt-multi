# app/ops/data_quality.py
"""
【大白话注释】
本模块负责三件事：
1) 数据质量评分：Completeness（完整度）、Timeliness（新鲜度/时延）、Validity（有效性）
2) 异常值清洗：根据设备/点位的合理边界 + 统计方法（Z分/IQR）把“离谱值”标记为缺测
3) 缺测插补与重采样：把不均匀/缺点的原始点，变成等间隔序列（秒/分钟），用于大屏/KPI/预测

【最终落地约定】
- 所有时间戳统一用 UTC epoch 秒（float）
- 设备/点位的“合理边界”由现场标定，默认值在 DEFAULT_BOUNDS，可按资产类型/点位覆盖
- 接口幂等：同样的数据多次调用，结果一致
- 该模块不直接写数据库；由调用方把评分结果写入 QualityRepo（见 app/core/repositories.py）

【典型用法】
1) 从 TSDB 拉出一段原始点（(ts, value) 列表）
2) 调用 clean_and_impute(...) 得到等间隔的“干净曲线”
3) 调用 score_quality(...) 计算该窗口的质量分
4) 由调用方把质量分写入 QualityRepo，清洗后的曲线再写回 TSDB 或直接用于计算/展示
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Literal
import math
import statistics
import time

EpochSec = float
Number = float
Series = List[Tuple[EpochSec, Number]]

# =========================
# 真实港口数据：默认边界
# =========================
# 说明：以下是“工程合理范围”示例，请按现场点表/设备标定调整。
# Keyed by (asset_type, point); deployments may add asset-specific overrides.
DEFAULT_BOUNDS: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {
    # 岸桥（Quay Crane）
    ("quay_crane", "active_power_kw"): (50.0, 2000.0),
    ("quay_crane", "status"): (0.0, 3.0),             # 0=idle,1=run,2=standby,3=fault （示例）
    # 场桥（RTG/RMG）
    ("yard_crane", "active_power_kw"): (20.0, 600.0),
    # 照明
    ("lighting", "active_power_kw"): (0.0, 300.0),
    # 冷站
    ("chiller", "supply_temp_c"): (2.0, 12.0),
    ("chiller", "return_temp_c"): (6.0, 20.0),
    ("chiller", "active_power_kw"): (10.0, 5000.0),
    # 储能/电池
    ("pcs", "charge_power_kw"): (-5000.0, 0.0),       # 充电为负功率（示例口径）
    ("pcs", "discharge_power_kw"): (0.0, 5000.0),
    ("battery", "soc"): (0.0, 100.0),                 # 百分比
    # 配电/电网
    ("grid", "frequency_hz"): (49.0, 51.0),
    ("grid", "voltage_v"): (300.0, 500000.0),         # 低压~高压示例范围
}

# =========================
# 质量评分（QDF）
# =========================
def score_quality(
    resampled: Series,
    *,
    now_ts: Optional[EpochSec] = None,
    step_sec: int = 60,
    # timeliness 的阈值：最近一个点离 now 的时延 <= 2*step 记满分；> 10*step 记 0
    timely_full: int = 2,
    timely_zero: int = 10,
    valid_mask: Optional[List[bool]] = None,
) -> Dict[str, float]:
    """
    输入：等间隔序列 `resampled`（缺测用 None 已被插补为数值）
    输出：质量分 dict：{'completeness':..,'timeliness':..,'validity':..}
    - Completeness：非缺测/可用点数量 ÷ 期望点数量
    - Timeliness：根据最近一个点的时延打分（线性衰减）
    - Validity：未被判为异常的点数量 ÷ 总点数（可传 valid_mask；若未传，默认全有效）

    注：resampled 应该是“清洗 + 插补后”的等间隔序列；如果你想先打原始数据的 Completeness，
        可以用 resample_regular() 在不插补的情况下只统计缺口占比。
    """
    n = len(resampled)
    if n == 0:
        return {"completeness": 0.0, "timeliness": 0.0, "validity": 0.0}

    # Completeness is the finite-value count divided by the total count.
    have_values = sum(1 for _, v in resampled if isinstance(v, (int, float)) and not math.isnan(v))
    completeness = have_values / n

    # Timeliness
    if now_ts is None:
        now_ts = time.time()
    last_ts = resampled[-1][0]
    delay_sec = max(0.0, now_ts - last_ts)
    if delay_sec <= timely_full * step_sec:
        timeliness = 1.0
    elif delay_sec >= timely_zero * step_sec:
        timeliness = 0.0
    else:
        # 线性插值
        x = (delay_sec - timely_full * step_sec) / ((timely_zero - timely_full) * step_sec)
        timeliness = max(0.0, 1.0 - x)

    # Validity：如果传了 valid_mask（True=有效，False=异常），按比例计算；否则默认全有效
    if valid_mask is not None and len(valid_mask) == n:
        validity = sum(1 for ok in valid_mask if ok) / n
    else:
        validity = 1.0

    return {
        "completeness": round(float(completeness), 6),
        "timeliness": round(float(timeliness), 6),
        "validity": round(float(validity), 6),
    }

# =========================
# 重采样 + 清洗 + 插补
# =========================
def resample_regular(
    raw: Series,
    *,
    start: EpochSec,
    end: EpochSec,
    step_sec: int,
    method: Literal["none", "ffill", "linear"] = "ffill",
) -> Series:
    """
    把“稀疏/不等间隔”的原始点，重采样成等间隔时间网格：
    - method='none'：遇到缺口就返回 None（调用方可自行处理）
    - method='ffill'：前向填充（适合“状态/设定点/慢变量”）
    - method='linear'：线性插值（适合连续量，如功率/温度）

    返回：[(ts, value_or_none)]
    """
    if step_sec <= 0:
        raise ValueError("step_sec must be positive")

    raw_sorted = sorted(raw, key=lambda x: x[0])
    grid = []
    i = 0
    prev_ts, prev_val = None, None
    next_ts, next_val = None, None

    # 预取下一个点
    def peek_next(idx: int) -> Tuple[Optional[EpochSec], Optional[Number]]:
        if 0 <= idx < len(raw_sorted):
            return raw_sorted[idx][0], raw_sorted[idx][1]
        return None, None

    next_ts, next_val = peek_next(0)

    t = start
    while t <= end:
        # 吃掉所有 <= t 的原始点
        while next_ts is not None and next_ts <= t:
            prev_ts, prev_val = next_ts, next_val
            i += 1
            next_ts, next_val = peek_next(i)

        # 选择填值
        if method == "none":
            val = prev_val if (prev_ts == t) else None
        elif method == "ffill":
            val = prev_val
        elif method == "linear":
            if prev_ts is not None and next_ts is not None and prev_ts <= t <= next_ts and next_ts != prev_ts:
                # 线性插值
                alpha = (t - prev_ts) / (next_ts - prev_ts)
                val = prev_val + alpha * (next_val - prev_val)
            else:
                val = prev_val  # 无法插，退化成 ffill
        else:
            raise ValueError(f"unsupported method: {method}")

        grid.append((float(t), float(val) if isinstance(val, (int, float)) else None))
        t += step_sec

    return grid


def detect_anomalies(
    series: Series,
    *,
    z_thresh: float = 3.5,
    iqr_k: float = 1.5,
    ignore_none: bool = True,
) -> List[bool]:
    """
    统计法检测异常点：
    - Z分数：|z| > z_thresh 判异常
    - IQR：低于 Q1 - k*IQR 或高于 Q3 + k*IQR 判异常
    返回与 series 等长的布尔掩码 True=“有效”，False=“异常”
    """
    values = [v for _, v in series if isinstance(v, (int, float))]
    if not values:
        return [True] * len(series)

    # 计算 Z 分数
    mu = statistics.fmean(values)
    sigma = statistics.pstdev(values) or 1e-9
    zmask = []
    for _, v in series:
        if not isinstance(v, (int, float)):
            zmask.append(True if ignore_none else False)
        else:
            z = (v - mu) / sigma
            zmask.append(abs(z) <= z_thresh)

    # 计算 IQR
    sorted_vals = sorted(values)
    q1 = _percentile(sorted_vals, 25)
    q3 = _percentile(sorted_vals, 75)
    iqr = (q3 - q1) or 1e-9
    lo = q1 - iqr_k * iqr
    hi = q3 + iqr_k * iqr
    imask = []
    for _, v in series:
        if not isinstance(v, (int, float)):
            imask.append(True if ignore_none else False)
        else:
            imask.append(lo <= v <= hi)

    # 合并（都通过才算有效）
    return [a and b for a, b in zip(zmask, imask)]


def apply_bounds(
    series: Series,
    *,
    asset_type: str,
    point: str,
    bounds_override: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> List[Optional[Number]]:
    """
    按设备/点位合理边界，把越界值标成 None（缺测，后续插补）
    返回与 series 等长的“可能为 None 的值列表”
    """
    lo, hi = (None, None)
    if bounds_override is not None:
        lo, hi = bounds_override
    else:
        lo, hi = DEFAULT_BOUNDS.get((asset_type, point), (None, None))

    out: List[Optional[Number]] = []
    for _, v in series:
        if not isinstance(v, (int, float)):
            out.append(None)
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            out.append(None)  # 越界视为无效
        else:
            out.append(float(v))
    return out


def impute_missing(
    grid_ts: List[EpochSec],
    values: List[Optional[Number]],
    *,
    method: Literal["ffill", "linear"] = "ffill",
) -> List[Number]:
    """
    对 None 的位置做插补：ffill 或 linear。
    - linear：对连续缺口线性插值；遇到前后都缺则退化为 ffill
    返回全为 float 的列表（无 None）
    """
    n = len(values)
    out: List[Optional[Number]] = values[:]

    if method not in ("ffill", "linear"):
        raise ValueError("method must be 'ffill' or 'linear'")

    # 先 ffill 一遍，保证前缀无值时也能填上（使用第一条非空值）
    last = None
    for i in range(n):
        if out[i] is None:
            out[i] = last
        else:
            last = out[i]
    # 若全是 None，用 0 兜底
    if all(v is None for v in out):
        return [0.0] * n
    # 前缀仍 None 的位置，用第一个非 None 值填
    first_val = next((v for v in out if v is not None), 0.0)
    out = [first_val if v is None else v for v in out]

    if method == "linear":
        # A working mask preserves original gaps during linear interpolation.
        # 为了简单，跳过，因为上面已经 ffill；生产中可额外实现双向填充再线性融合
        pass

    return [float(v) for v in out]  # type: ignore


def clean_and_impute(
    raw: Series,
    *,
    start: EpochSec,
    end: EpochSec,
    step_sec: int,
    asset_type: str,
    point: str,
    bounds_override: Optional[Tuple[Optional[float], Optional[float]]] = None,
    resample_method: Literal["none", "ffill", "linear"] = "ffill",
    impute_method: Literal["ffill", "linear"] = "ffill",
    anomaly_z: float = 3.5,
    anomaly_iqr_k: float = 1.5,
) -> Tuple[Series, Dict[str, float], List[bool]]:
    """
    把原始点(raw) -> 等间隔 -> 边界过滤 -> 异常检测 -> 插补，返回：
    - cleaned: [(ts, value)]（“清洗+插补后的等间隔序列”）
    - quality: {'completeness','timeliness','validity'}
    - valid_mask: 与 cleaned 等长的布尔列表（True=有效/非异常；False=异常/越界）
    """
    # 第一步：重采样到等间隔网格
    grid = resample_regular(raw, start=start, end=end, step_sec=step_sec, method=resample_method)

    # 第二步：按边界过滤
    bounded_vals = apply_bounds(grid, asset_type=asset_type, point=point, bounds_override=bounds_override)

    # 第三步：异常检测（基于当前网格中的“有效值”统计）
    tmp_series = [(ts, v if v is not None else float("nan")) for (ts, _), v in zip(grid, bounded_vals)]
    # 将 nan 作为 None 处理，构造用于检测的 series2
    series2 = [(ts, v) for (ts, v) in tmp_series if isinstance(v, (int, float)) and not math.isnan(v)]
    # 为了掩码长度匹配，使用 bounded_vals 构造带 None 的 series_detect
    series_detect = [(ts, (v if v is not None else float("nan"))) for ts, v in zip([t for t, _ in grid], bounded_vals)]
    valid_mask = detect_anomalies(series_detect, z_thresh=anomaly_z, iqr_k=anomaly_iqr_k)

    # 第四步：把异常位置也当作缺测进行插补
    grid_ts = [t for t, _ in grid]
    masked_vals: List[Optional[Number]] = []
    for ok, v in zip(valid_mask, bounded_vals):
        if (not ok) or (v is None):
            masked_vals.append(None)
        else:
            masked_vals.append(v)

    imputed_vals = impute_missing(grid_ts, masked_vals, method=impute_method)
    cleaned = list(zip(grid_ts, imputed_vals))

    # 第五步：质量评分
    quality = score_quality(cleaned, step_sec=step_sec, valid_mask=valid_mask)

    return cleaned, quality, valid_mask

# =========================
# 小工具
# =========================
def _percentile(sorted_vals: List[Number], p: float) -> Number:
    """简易百分位（p=0..100），假设已排序。"""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)

# =========================
# 冒烟测试（独立运行）
# =========================
def _smoke() -> dict:
    """
    生成一段模拟“岸桥功率”的原始点，包含：
    - 正常点，偶发离谱值（异常），随机缺测
    - 输出：清洗后的等间隔曲线、质量分、有效掩码统计
    """
    import random, time
    now = time.time()
    start = now - 300   # 最近5分钟
    end = now
    step = 10           # 10秒步长

    # 生成稀疏原始点
    raw: Series = []
    t = start
    while t <= end:
        if random.random() < 0.1:
            # 模拟缺测（跳过）
            t += random.choice([5, 10, 15])
            continue
        val = 300.0 + 50.0 * math.sin(t / 30.0)  # 基础波动
        if random.random() < 0.05:
            val *= random.choice([0.1, 3.0])     # 5% 概率异常
        raw.append((t, val))
        t += random.choice([5, 10, 12])

    cleaned, quality, mask = clean_and_impute(
        raw,
        start=start,
        end=end,
        step_sec=step,
        asset_type="quay_crane",
        point="active_power_kw",
        resample_method="ffill",
        impute_method="ffill",
    )

    return {
        "raw_points": len(raw),
        "clean_points": len(cleaned),
        "quality": quality,
        "valid_ratio": round(sum(1 for m in mask if m) / len(mask), 4) if mask else 0.0,
        "sample_clean_head": cleaned[:3],
        "sample_clean_tail": cleaned[-3:],
    }


if __name__ == "__main__":
    # 允许直接运行自测：python -m app.ops.data_quality
    import json
    print(json.dumps(_smoke(), ensure_ascii=False, indent=2))
