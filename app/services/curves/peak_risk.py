from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math

from .service import CurvesService

_Z = 1.28155  # 正态分布 10%/90% 分位对应的 ±z


class CurvesPeakRisk:
    """
    需量峰值风险（15min滚动）估计：
      - 对全港聚合功率分位曲线（p10/p50/p90）做 W=avg_window_min 的滚动平均；
      - 用 p50 作为滚动需量均值 μ，用 (p90-p10)/(2*Z) 估计波动 σ；
      - 由 p50 和模型真实输出的 p10/p90 估算正态分布参数；
      - 风险严格定义为 P(滚动需量 > 用户输入合同阈值)。

    不自动调整合同阈值，也不把启发式“接近阈值”分数冒充概率。
    """

    def __init__(self, di) -> None:
        self.di = di
        self.curves = CurvesService(di)

    # ----------------- helpers -----------------
    @staticmethod
    def _rolling_avg(vals: List[float], w: int) -> List[float]:
        if w <= 1:
            return vals[:]
        out: List[float] = []
        s = 0.0
        for i, x in enumerate(vals):
            s += x
            if i >= w:
                s -= vals[i - w]
            if i >= w - 1:
                out.append(s / w)
        return out

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # ----------------- main API -----------------
    def peak_risk(
        self,
        mode: str = "forecast",
        horizon_min: int = 360,
        step_min: int = 1,
        limit: int = 200,
        cap_kw: float = 500.0,
        avg_window_min: int = 15,
        scenario: str = "baseline",
        use_drivers: bool = True,
    ) -> Dict[str, Any]:
        agg = self.curves.aggregate(
            mode=mode,
            horizon_min=horizon_min,
            step_min=step_min,
            limit=limit,
            scenario=scenario,
            use_drivers=use_drivers,
        ) or {}
        series = agg.get("series") or {}
        p50 = series.get("p50") or []
        p10 = series.get("p10")
        p90 = series.get("p90")

        if not p50:
            if mode == "sim":
                agg = self.curves.aggregate(
                    mode="forecast",
                    horizon_min=horizon_min,
                    step_min=step_min,
                    limit=limit,
                    scenario=scenario,
                    use_drivers=use_drivers,
                ) or {}
                series = agg.get("series") or {}
                p50 = series.get("p50") or []
                p10 = series.get("p10")
                p90 = series.get("p90")

        if not p50 or not p10 or not p90:
            return {
                "mode": mode,
                "available": False,
                "reason": "peak probability requires model-produced p10 and p90 quantiles",
                "series": {"risk": [], "p50_avg": [], "p10_avg": [], "p90_avg": []},
            }

        v50 = [float(p.get("kW", p.get("p50", 0.0)) or 0.0) for p in p50]
        v10 = [float(p.get("kW", p.get("p10", 0.0)) or 0.0) for p in p10]
        v90 = [float(p.get("kW", p.get("p90", 0.0)) or 0.0) for p in p90]
        ts = [p.get("ts") for p in p50]

        effective_step_min = int(agg.get("_step_min") or step_min)
        w = max(1, int(round(avg_window_min / max(1, effective_step_min))))
        mu = self._rolling_avg(v50, w)
        lo = self._rolling_avg(v10, w)
        hi = self._rolling_avg(v90, w)
        ts_w = ts[w - 1 :] if len(ts) >= w else []

        cap_eff_kw = max(1.0, float(cap_kw))

        risk_series: List[Dict[str, Any]] = []
        p50_avg_series: List[Dict[str, Any]] = []
        p10_avg_series: List[Dict[str, Any]] = []
        p90_avg_series: List[Dict[str, Any]] = []
        util_series: List[Dict[str, Any]] = []
        exceed_prob_series: List[Dict[str, Any]] = []

        first_cross = None
        peak_p50 = 0.0
        peak_risk = 0.0

        for i in range(len(mu)):
            m = mu[i]
            l = lo[i] if i < len(lo) else m
            h = hi[i] if i < len(hi) else m
            sigma = max(1e-6, (h - l) / (2.0 * _Z))
            z = (float(cap_eff_kw) - m) / sigma
            p_exceed = max(0.0, min(1.0, 1.0 - self._norm_cdf(z)))
            util = (m / float(cap_eff_kw)) if float(cap_eff_kw) > 1e-6 else 0.0
            p_risk = p_exceed
            t = ts_w[i] if i < len(ts_w) else None

            risk_series.append({"ts": t, "p": round(p_risk, 6)})
            exceed_prob_series.append({"ts": t, "p": round(p_exceed, 6)})
            util_series.append({"ts": t, "ratio": round(util, 6)})
            p50_avg_series.append({"ts": t, "kW": round(m, 3)})
            p10_avg_series.append({"ts": t, "kW": round(l, 3)})
            p90_avg_series.append({"ts": t, "kW": round(h, 3)})

            if first_cross is None and p_risk >= 0.5:
                first_cross = {"index": i, "ts": t, "p": round(p_risk, 6)}
            peak_p50 = max(peak_p50, m)
            peak_risk = max(peak_risk, p_risk)

        peak_util = (peak_p50 / float(cap_eff_kw)) if float(cap_eff_kw) > 1e-6 else 0.0

        return {
            "mode": mode,
            "available": True,
            "unit": "probability",
            "cap_kw": float(cap_eff_kw),
            "cap_kw_input": float(cap_kw),
            "window_min": int(avg_window_min),
            "step_min": int(step_min),
            "step_min_effective": effective_step_min,
            "series": {
                "risk": risk_series,
                "exceed_prob": exceed_prob_series,
                "utilization": util_series,
                "p50_avg": p50_avg_series,
                "p10_avg": p10_avg_series,
                "p90_avg": p90_avg_series,
            },
            "first_cross": first_cross,
            "peak_p50_max": round(peak_p50, 3),
            "cap_gap_p50": round(peak_p50 - float(cap_eff_kw), 3),
            "peak_utilization": round(peak_util, 6),
            "peak_risk": round(peak_risk, 6),
            "risk_logic": "normal_quantile_exceedance_probability",
            "quantile_source": agg.get("_source", "unknown"),
        }
