from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
import re

from .service import CurvesService

GROUPS = ["QC", "YC", "AGV", "BESS", "LIGHT", "HVAC", "SHORE", "OTHER"]

# 典型港区负荷构成（仅作组级兜底）
_BASE_WEIGHTS: Dict[str, float] = {
    "QC": 0.30,
    "YC": 0.23,
    "AGV": 0.14,
    "BESS": 0.03,
    "LIGHT": 0.07,
    "HVAC": 0.09,
    "SHORE": 0.10,
    "OTHER": 0.04,
}


class CurvesStacked:
    """将聚合总负荷拆为设备组曲线。

    这版重点解决两个问题：
    1) 不能只是“同一条总曲线换不同权重”——那样各组看起来会过于一致；
    2) 各组之和仍需与 aggregate 总负荷同口径对齐。

    处理思路：
    - 先为每个设备组生成具有明显不同业务节奏的“活跃度曲线”；
    - 再按每个时刻的活跃度比例拆分 aggregate 总负荷；
    - BESS 单独做成与系统峰值有逆向关系的调节型曲线，而不是生产设备型曲线。
    """

    def __init__(self, di) -> None:
        self.di = di
        self.curves = CurvesService(di)

    def _group_of(self, a: Dict[str, Any]) -> str:
        t = (a.get("type") or a.get("asset_type") or "").lower()
        n = (a.get("name") or a.get("id") or a.get("asset_id") or "").lower()
        s = f"{t} {n}"
        if re.search(r"\b(qc|quay crane|岸桥|桥吊|sts)\b", s):
            return "QC"
        if re.search(r"\b(yc|rtg|rmg|yard crane|场桥)\b", s):
            return "YC"
        if re.search(r"\b(agv|truck|terminal tractor|集卡|拖车)\b", s):
            return "AGV"
        if re.search(r"\b(shore\s*power|shore-power|岸电|cold ironing|shore)\b", s):
            return "SHORE"
        if re.search(r"\b(bess|battery|储能|ess)\b", s):
            return "BESS"
        if re.search(r"\b(light|lighting|照明|高杆灯)\b", s):
            return "LIGHT"
        if re.search(r"\b(hvac|chiller|冷站|空调|制冷)\b", s):
            return "HVAC"
        return "OTHER"

    def _list_assets(self, limit: int) -> List[Dict[str, Any]]:
        try:
            raw = self.di.telemetry.list_assets() or []
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for a in raw:
            if not isinstance(a, dict):
                continue
            aid = a.get("id") or a.get("asset_id")
            if not aid:
                continue
            out.append({"id": str(aid), "group": self._group_of(a)})
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _smoothstep(x: float, left: float, right: float) -> float:
        if right <= left:
            return 1.0 if x >= right else 0.0
        t = (x - left) / (right - left)
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _base_activity(self, g: str, i: int, n: int, agg_total: List[float]) -> float:
        x = i / max(1, n - 1)

        # 归一化总负荷，用于让部分组对系统状态有响应，但不直接复制总曲线形状。
        tmin = min(agg_total)
        tmax = max(agg_total)
        norm_total = 0.5 if tmax - tmin < 1e-9 else (agg_total[i] - tmin) / (tmax - tmin)

        if g == "QC":
            # 岸桥：更像作业窗口驱动，中段/后段会出现几个高作业平台，而不是全程跟总负荷同步。
            window_1 = 0.70 * self._smoothstep(x, 0.16, 0.23) * (1.0 - self._smoothstep(x, 0.33, 0.40))
            window_2 = 0.85 * self._smoothstep(x, 0.56, 0.63) * (1.0 - self._smoothstep(x, 0.74, 0.82))
            window_3 = 0.35 * self._smoothstep(x, 0.86, 0.91) * (1.0 - self._smoothstep(x, 0.96, 0.995))
            ripple = 0.05 * math.sin(10.0 * math.pi * x + 0.3)
            return max(0.05, 0.62 + window_1 + window_2 + window_3 + ripple)

        if g == "YC":
            # 场桥：跟随箱区作业，滞后于岸桥，更宽、更平，低频块状变化更明显。
            broad_1 = 0.42 * self._smoothstep(x, 0.22, 0.30) * (1.0 - self._smoothstep(x, 0.44, 0.52))
            broad_2 = 0.56 * self._smoothstep(x, 0.60, 0.68) * (1.0 - self._smoothstep(x, 0.88, 0.96))
            drift = 0.10 * x
            ripple = 0.03 * math.sin(5.0 * math.pi * x + 1.2)
            return max(0.05, 0.56 + broad_1 + broad_2 + drift + ripple)

        if g == "AGV":
            # AGV：高频碎波动，不应和 QC/YC 同节奏。加入轻微脉冲与更高频谐波。
            hi = 0.14 * math.sin(18.0 * math.pi * x + 0.4)
            hi2 = 0.08 * math.sin(36.0 * math.pi * x + 1.1)
            packet = 0.18 * self._smoothstep(x, 0.28, 0.33) * (1.0 - self._smoothstep(x, 0.40, 0.46))
            packet += 0.22 * self._smoothstep(x, 0.66, 0.72) * (1.0 - self._smoothstep(x, 0.82, 0.88))
            return max(0.05, 0.50 + packet + hi + hi2)

        if g == "BESS":
            # BESS：调节型，系统越接近峰值越容易动作，但会呈现脉冲式削峰，而不是生产负荷式连续上升。
            trigger = max(0.0, norm_total - 0.55)
            pulse = 0.16 * math.sin(12.0 * math.pi * x + 2.0)
            anti = 0.24 * trigger
            return max(0.02, 0.12 + anti + pulse)

        if g == "LIGHT":
            # 照明：应更平稳，只在边缘时段略抬升。
            edge = 0.18 * (1.0 - self._smoothstep(x, 0.08, 0.22)) + 0.20 * self._smoothstep(x, 0.82, 0.95)
            return max(0.05, 0.74 + edge + 0.015 * math.sin(4.0 * math.pi * x + 0.7))

        if g == "HVAC":
            # HVAC：更平滑的中长周期变化，带轻微午后爬升感。
            hump = 0.22 * math.sin(math.pi * max(0.0, min(1.0, (x - 0.10) / 0.80)))
            return max(0.05, 0.64 + 0.16 * x + hump + 0.02 * math.sin(6.0 * math.pi * x + 1.8))

        if g == "SHORE":
            # 岸电：更像船舶接靠后的分段接入，块状台阶感更明显。
            level = 0.46
            if x >= 0.18:
                level += 0.22
            if x >= 0.48:
                level += 0.16
            if x >= 0.78:
                level -= 0.10
            return max(0.05, level + 0.01 * math.sin(4.0 * math.pi * x))

        # OTHER：小幅随机感近似，但仍保持平稳。
        return max(0.05, 0.34 + 0.04 * math.sin(7.0 * math.pi * x + 2.3) + 0.02 * math.sin(15.0 * math.pi * x + 0.6))

    def _aggregate_total(self, mode: str, horizon_min: int, step_min: int, limit: int) -> Tuple[List[Any], List[float]]:
        agg = self.curves.aggregate(mode=mode, horizon_min=horizon_min, step_min=step_min, limit=limit) or {}
        p50 = ((agg.get("series") or {}).get("p50") or [])
        x = [p.get("ts") for p in p50]
        total = [float(p.get("kW", p.get("p50", 0.0)) or 0.0) for p in p50]
        return x, total

    def stacked_power(
        self,
        mode: str = "forecast",
        horizon_min: int = 360,
        step_min: int = 1,
        limit: int = 200,
    ) -> Dict[str, Any]:
        x, agg_total = self._aggregate_total(mode, horizon_min, step_min, limit)
        n = len(agg_total)
        if n == 0:
            return {"mode": mode, "unit": "kW", "groups": GROUPS, "x": [], "series": {g: [] for g in GROUPS}, "total": []}

        assets = self._list_assets(limit=limit)
        group_counts = {g: 0 for g in GROUPS}
        for a in assets:
            group_counts[a["group"]] = group_counts.get(a["group"], 0) + 1

        weights: Dict[str, float] = {}
        total_assets = sum(group_counts.values())
        for g in GROUPS:
            base = _BASE_WEIGHTS[g]
            if total_assets > 0:
                obs = group_counts[g] / total_assets
                # 给资产数量更高一点参与度，但仍保留港口经验权重。
                w = 0.60 * base + 0.40 * obs
            else:
                w = base
            weights[g] = max(0.001, w)

        ssum = sum(weights.values()) or 1.0
        for g in GROUPS:
            weights[g] /= ssum

        raw: Dict[str, List[float]] = {g: [] for g in GROUPS}
        for i, tot in enumerate(agg_total):
            activity: Dict[str, float] = {}
            for g in GROUPS:
                activity[g] = max(0.001, weights[g] * self._base_activity(g, i, n, agg_total))

            denom = sum(activity.values()) or 1.0
            for g in GROUPS:
                raw[g].append(round(tot * activity[g] / denom, 3))

        total = [round(sum(raw[g][i] for g in GROUPS), 3) for i in range(n)]
        return {
            "mode": mode,
            "unit": "kW",
            "groups": GROUPS,
            "x": x,
            "series": raw,
            "total": total,
        }
