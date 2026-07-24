# ============================================
# app/services/reporting.py
# --------------------------------------------
# 小报表 / 设备侧诊断统计（后端聚合口径）
#
# 目标：
#  - 为前端“报表”按钮与预警/指挥盘提供一致的统计口径；
#  - 计算近窗均值、分位数(P50/P95/P99)、窗口能量积分(kWh)、
#    覆盖时长(coverage)、利用率估计、简单异常分数等；
#  - 保持对旧前端字段的向后兼容：avg_kW_last5min / p95_kW / carbonIntensity。
#
# 依赖：
#   - 需要 telemetry 提供
#       list_assets() -> [{"id":"qc-01","label":"..."}, ...]
#       get_recent_power(asset_id) -> [{"ts":"...Z","kW":12.3}, ...]  时间升序或乱序均可
#
# 说明：
#   - 本服务本身不持久化，仅对“最近窗口”进行瞬时统计；
#   - 当窗口不足 5 分钟时，自动退化为“使用可得到的全部点”并标注 coverage；
#   - 额定功率和碳强度必须来自资产注册表；缺失时返回 None。
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math
import statistics


# ---- 一些常量（避免跨模块强耦合） ----
DEFAULT_WINDOW_MIN = 5.0  # 近窗统计的目标分钟数


# ---- 小工具 ----
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _parse_iso(ts: str) -> Optional[datetime]:
    """尽量健壮地解析 ISO8601；失败返回 None。"""
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _classify(asset_id: str, label: str = "") -> str:
    s = (asset_id or "").lower()
    if s.startswith("qc") or "岸桥" in label: return "qc"
    if s.startswith("yc") or "场桥" in label: return "yc"
    if s.startswith("agv") or "agv" in s: return "agv"
    if s.startswith("truck") or "拖车" in label: return "truck"
    if s.startswith("wh") or "仓库" in label: return "wh"
    if s.startswith("cs") or "充电" in label: return "cs"
    if s.startswith("ps") or "配电" in label: return "ps"
    if s.startswith("yard") or "堆场" in label: return "yard"
    return "misc"


def _percentile(sorted_values: List[float], q: float) -> float:
    """
    简单分位数计算：q ∈ (0,1)，使用线性插值。
    输入必须已排序。
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = (n - 1) * q
    i = int(math.floor(pos))
    j = min(i + 1, n - 1)
    frac = pos - i
    return sorted_values[i] * (1 - frac) + sorted_values[j] * frac


def _trapezoid_kwh(points: List[Tuple[datetime, float]]) -> Tuple[float, float]:
    """
    梯形积分：返回 (kWh, 覆盖秒数)
    points: [(ts, kW), ...] 需要按时间升序
    """
    if len(points) < 2:
        return 0.0, 0.0
    kwh = 0.0
    span_sec = 0.0
    last_t, last_kw = points[0]
    for i in range(1, len(points)):
        t, kw = points[i]
        dt_h = max(0.0, (t - last_t).total_seconds() / 3600.0)
        kwh += max(0.0, (last_kw + kw) * 0.5 * dt_h)
        span_sec += dt_h * 3600.0
        last_t, last_kw = t, kw
    return kwh, span_sec


@dataclass
class WindowStats:
    count: int
    coverage_sec: float
    avg_kw: float
    min_kw: float
    max_kw: float
    p50_kw: float
    p95_kw: float
    p99_kw: float
    energy_kwh: float


class ReportingService:
    """
    小报表服务：
      - generate_mini_report(asset_id) -> Dict[str, Any]
    """

    def __init__(self, telemetry, window_min: float = DEFAULT_WINDOW_MIN):
        self.telemetry = telemetry
        self.window_min = float(window_min)

    # -------- 内部：提取窗口数据并计算统计量 --------
    def _collect_window_stats(self, asset_id: str, window_min: float) -> WindowStats:
        # 1) 拿数据
        raw = self.telemetry.get_recent_power(asset_id) or []
        # 2) 解析并排序
        pts: List[Tuple[datetime, float]] = []
        for p in raw:
            if not isinstance(p, dict):
                continue
            ts = _parse_iso(p.get("ts"))
            kw = _safe_float(p.get("kW"))
            if ts is not None:
                pts.append((ts, kw))
        if not pts:
            return WindowStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pts.sort(key=lambda x: x[0])

        # 3) 根据 window_min 过滤“近窗”
        now = datetime.now(timezone.utc)
        win_start = now - timedelta(minutes=window_min)
        win_pts = [p for p in pts if p[0] >= win_start]
        # 若窗口内点太少（例如仅 60 秒），退化为“全量可用点”
        if len(win_pts) < 3:
            win_pts = pts

        # 4) 基本统计
        values = [v for _, v in win_pts]
        values_sorted = sorted(values)
        avg_kw = statistics.fmean(values) if values else 0.0
        min_kw = values_sorted[0] if values_sorted else 0.0
        max_kw = values_sorted[-1] if values_sorted else 0.0
        p50_kw = _percentile(values_sorted, 0.50)
        p95_kw = _percentile(values_sorted, 0.95)
        p99_kw = _percentile(values_sorted, 0.99)

        # 5) 梯形积分（kWh）与覆盖秒数
        energy_kwh, span_sec = _trapezoid_kwh(win_pts)

        return WindowStats(
            count=len(win_pts),
            coverage_sec=span_sec,
            avg_kw=avg_kw,
            min_kw=min_kw,
            max_kw=max_kw,
            p50_kw=p50_kw,
            p95_kw=p95_kw,
            p99_kw=p99_kw,
            energy_kwh=energy_kwh,
        )

    # -------- 对外：生成“迷你报表” --------
    def generate_mini_report(self, asset_id: str) -> Dict[str, Any]:
        # 设备 label（用于类型推断；如果 telemetry 没提供 label，也不影响）
        label = ""
        rated = None
        carbon_intensity = None
        try:
            for a in (self.telemetry.list_assets() or []):
                if a.get("id") == asset_id:
                    label = a.get("label", "")
                    raw_rated = a.get("rated_kW", a.get("rated_kw", a.get("power_kw")))
                    rated = float(raw_rated) if raw_rated is not None else None
                    raw_ci = a.get("carbon_g_per_kwh", a.get("carbonIntensity"))
                    carbon_intensity = float(raw_ci) if raw_ci is not None else None
                    break
        except Exception:
            pass

        typ = _classify(asset_id, label)

        # 窗口统计
        stats = self._collect_window_stats(asset_id, self.window_min)

        # 利用率估计（用 avg_kw / rated）
        util = None
        if rated is not None and rated > 0:
            util = max(0.0, min(1.0, stats.avg_kw / rated))

        # 简单异常分数：avg > p95 的“超出程度”
        anomaly_score = 0.0
        is_above_p95 = False
        if stats.p95_kw > 1e-6 and stats.avg_kw > stats.p95_kw:
            ratio = min(2.0, stats.avg_kw / stats.p95_kw)
            anomaly_score = round((ratio - 1.0) / 1.0, 3)  # [0,1] 之间，越大越异常
            is_above_p95 = True

        # 友好的一些派生值
        coverage_min = stats.coverage_sec / 60.0
        # 若覆盖时长不足 60s，视为“窗口不充分”
        window_ok = coverage_min >= 1.0

        # ---- 构造返回（向后兼容 + 增强字段）----
        return {
            # 兼容旧前端的关键字段：
            "avg_kW_last5min": round(stats.avg_kw, 3),
            "p95_kW": round(stats.p95_kw, 3),
            "carbonIntensity": round(carbon_intensity, 1) if carbon_intensity is not None else None,
            "carbon_intensity_source": "asset_registry" if carbon_intensity is not None else "unavailable",

            # 增强可视化与诊断字段：
            "asset": asset_id,
            "type": typ,
            "rated_kW": rated,
            "n_points": stats.count,
            "coverage_min": round(coverage_min, 3),
            "window_ok": window_ok,  # 是否认为近窗充分
            "min_kW": round(stats.min_kw, 3),
            "p50_kW": round(stats.p50_kw, 3),
            "p99_kW": round(stats.p99_kw, 3),
            "max_kW": round(stats.max_kw, 3),
            "energy_kWh_window": round(stats.energy_kwh, 6),  # 仅对“已覆盖窗口”的积分，非全天
            "utilization_est_percent": round(util * 100.0, 2) if util is not None else None,
            "anomaly": {
                "is_above_p95": is_above_p95,
                "score_0to1": anomaly_score,
            },
            "notes": "avg/pXX 基于近窗数据；energy_kWh_window 为窗口积分；缺失额定功率或碳因子时对应字段为空。"
        }
