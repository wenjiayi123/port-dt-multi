from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

from .service import CurvesService

class CurvesCarbonIntensity:
    """
    单位 TEU 碳排（累计）kgCO2e/TEU 分位曲线。

    TEU 仅接受请求参数或作业适配器；排放因子优先使用上游时序，
    否则使用明确标注的请求参数常数。不会合成峰谷因子或不确定性区间。
    """

    def __init__(self, di) -> None:
        self.di = di
        self.curves = CurvesService(di)

    # ---------- TEU 推断 ----------
    @staticmethod
    def _scalar_from_unknown(obj: Any, keys: List[str]) -> Optional[float]:
        if obj is None:
            return None
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            for k in keys:
                if k in obj and obj[k] is not None:
                    try:
                        return float(obj[k])
                    except Exception:
                        continue
            return None
        if isinstance(obj, list) and obj:
            total = 0.0
            found = False
            for x in obj:
                v = CurvesCarbonIntensity._scalar_from_unknown(x, keys)
                if v is not None:
                    total += v
                    found = True
            if found:
                return total
        return None

    def _teu_from_obj(self, obj: Any, horizon_min: int, step_min: int) -> float:
        if obj is None:
            return 0.0

        for name in ("teu_total", "get_teu_total", "throughput_total", "throughput"):
            f = getattr(obj, name, None)
            if callable(f):
                for with_window in (True, False):
                    try:
                        v = f(horizon_min=horizon_min, step_min=step_min) if with_window else f()
                    except TypeError:
                        continue
                    except Exception:
                        break
                    val = self._scalar_from_unknown(v, ["teu", "TEU", "y", "value", "total"])
                    if val and val > 0:
                        return float(val)

        for name in ("teu_series", "throughput_series", "series", "get_series"):
            f = getattr(obj, name, None)
            if callable(f):
                try:
                    s = f(horizon_min=horizon_min, step_min=step_min)
                except TypeError:
                    try:
                        s = f()
                    except Exception:
                        continue
                except Exception:
                    continue
                val = self._scalar_from_unknown(s, ["teu", "TEU", "y", "value"])
                if val and val > 0:
                    return float(val)

        return 0.0

    def _resolve_teu(self, teu: float, horizon_min: int, step_min: int) -> Tuple[float, str]:
        if teu and teu > 0:
            return float(teu), "request_parameter"
        for attr in ("ops", "operations", "terminal", "traffic", "business"):
            obj = getattr(self.di, attr, None)
            val = self._teu_from_obj(obj, horizon_min, step_min)
            if val > 0:
                return float(val), f"{attr}_adapter"
        return 0.0, "none"

    # ---------- EF 获取 ----------
    def _ef_series(
        self,
        mode: str,
        horizon_min: int,
        step_min: int,
        ef_const_kg_per_kwh: float,
        power_arr: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], float, str]:
        candidates: List[Any] = []
        adapters = getattr(self.di, "adapters", None)
        if adapters is not None:
            for name in ("carbon_factors", "carbon", "esg"):
                obj = getattr(adapters, name, None)
                if obj is not None:
                    candidates.append(obj)
        for name in ("carbon_factors", "carbon", "esg"):
            obj = getattr(self.di, name, None)
            if obj is not None:
                candidates.append(obj)

        for obj in candidates:
            for fn_name in ("ef_series", "get_ef_series", "grid_ef_series", "series", "get_series"):
                fn = getattr(obj, fn_name, None)
                if callable(fn):
                    try:
                        try:
                            s = fn(mode=mode, horizon_min=horizon_min, step_min=step_min)
                        except TypeError:
                            s = fn(horizon_min=horizon_min, step_min=step_min)
                    except Exception:
                        continue
                    if isinstance(s, list) and s:
                        vals = []
                        out = []
                        for p in s:
                            if not isinstance(p, dict):
                                continue
                            ts = p.get("ts")
                            v = p.get("kg_per_kwh")
                            if v is None:
                                v = p.get("value", p.get("ef"))
                            try:
                                fv = float(v)
                            except Exception:
                                continue
                            out.append({"ts": ts, "kg_per_kwh": fv})
                            vals.append(fv)
                        if out:
                            avg = sum(vals) / len(vals)
                            return out, float(avg), "timeseries"

        constant = max(0.0, float(ef_const_kg_per_kwh))
        return ([{"ts": point.get("ts"), "kg_per_kwh": constant} for point in power_arr], constant, "parameter_constant")

    # ---------- 数学/积分 ----------
    @staticmethod
    def _acc_intensity(
        power_arr: List[Dict[str, Any]],
        ef_series: List[Dict[str, Any]],
        step_min: int,
        teu: float,
    ) -> Tuple[List[Dict[str, Any]], float]:
        if not power_arr:
            return [], 0.0

        ef_vals: List[float] = []
        for p in ef_series:
            try:
                ef_vals.append(float(p.get("kg_per_kwh", p.get("value", 0.0)) or 0.0))
            except Exception:
                ef_vals.append(0.0)
        if not ef_vals:
            ef_vals = [0.0]

        dt_h = float(step_min) / 60.0
        denom = max(1.0, float(teu))
        kg_total = 0.0
        out: List[Dict[str, Any]] = []
        for i, p in enumerate(power_arr):
            vkw = float(p.get("kW", p.get("p50", 0.0)) or 0.0)
            ef = ef_vals[i] if i < len(ef_vals) else ef_vals[-1]
            kg_total += vkw * dt_h * ef
            out.append({"ts": p.get("ts"), "kW": round(kg_total / denom, 6)})
        return out, round(kg_total, 3)

    # ---------- 主入口 ----------
    def intensity(
        self,
        mode: str = "forecast",
        horizon_min: int = 360,
        step_min: int = 1,
        limit: int = 200,
        teu: float = 12000.0,
        ef_const_kg_per_kwh: float = 0.55,
        scenario: str = "baseline",
        use_drivers: bool = True,
    ) -> Dict[str, Any]:
        teu_eff, teu_source = self._resolve_teu(teu, horizon_min, step_min)
        if teu_eff <= 0:
            return {
                "mode": mode,
                "available": False,
                "reason": "TEU denominator is required from the request or operations adapter",
                "unit": "kgCO2e/TEU",
                "series": {"p50": [], "p10": [], "p90": []},
                "teu": 0.0,
                "teu_source": "none",
            }

        agg = self.curves.aggregate(
            mode=mode,
            horizon_min=horizon_min,
            step_min=step_min,
            limit=limit,
            scenario=scenario,
            use_drivers=use_drivers,
        ) or {}

        S = agg.get("series") or {}
        p50 = S.get("p50") or []
        if not p50:
            return {
                "mode": mode,
                "available": False,
                "reason": "power curve is unavailable for the requested mode",
                "unit": "kgCO2e/TEU",
                "series": {"p50": [], "p10": [], "p90": []},
                "window_min": horizon_min,
                "step_min": step_min,
                "teu": float(teu_eff),
                "teu_source": teu_source,
                "ef_source": "none",
                "ef_avg_kg_per_kwh": 0.0,
                "cum_kg_total": 0.0,
                "intensity_total": 0.0,
            }

        p10 = S.get("p10") or []
        p90 = S.get("p90") or []
        effective_step_min = int(agg.get("_step_min") or step_min)

        ef_series, ef_avg, ef_src = self._ef_series(
            mode=mode,
            horizon_min=horizon_min,
            step_min=effective_step_min,
            ef_const_kg_per_kwh=ef_const_kg_per_kwh,
            power_arr=p50,
        )

        s50, kg_total = self._acc_intensity(p50, ef_series, effective_step_min, teu_eff)
        s10, _ = self._acc_intensity(p10, ef_series, effective_step_min, teu_eff)
        s90, _ = self._acc_intensity(p90, ef_series, effective_step_min, teu_eff)

        return {
            "mode": mode,
            "available": True,
            "unit": "kgCO2e/TEU",
            "series": {"p50": s50, "p10": s10, "p90": s90},
            "window_min": horizon_min,
            "step_min": step_min,
            "step_min_effective": effective_step_min,
            "teu": float(teu_eff),
            "teu_source": teu_source,
            "power_source": agg.get("_source", "unknown"),
            "ef_source": ef_src,
            "ef_avg_kg_per_kwh": ef_avg,
            "cum_kg_total": kg_total,
            "uncertainty_available": bool(s10 and s90),
            "intensity_total": (s50[-1]["kW"] if s50 else 0.0),
        }
