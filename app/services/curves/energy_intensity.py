# app/services/curves/energy_intensity.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

from .service import CurvesService

class CurvesEnergyIntensity:
    """
    单位 TEU 能耗（累计）kWh/TEU 分位曲线。

    TEU 分母只能来自请求参数或港口作业适配器。缺少二者时返回不可用，
    不再用所谓“典型港口吞吐”生成展示性分母。

    返回格式：
      {
        "mode": "forecast",
        "unit": "kWh/TEU",
        "series": {"p50": [...], "p10": [...], "p90": [...]},
        "window_min": 360,
        "step_min": 1,
        "teu": 12000.0,
        "cum_kwh_total": 1588.3,
        "intensity_total": 0.132358,
      }
    """

    def __init__(self, di) -> None:
        self.di = di
        self.curves = CurvesService(di)

    # ------------------------------------------------------------------
    # helpers: TEU 解析
    # ------------------------------------------------------------------

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
                v = CurvesEnergyIntensity._scalar_from_unknown(x, keys)
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

    # ------------------------------------------------------------------
    # helpers: 数学/积分
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate_to_intensity(
        arr: List[Dict[str, Any]], step_min: int, teu: float
    ) -> Tuple[List[Dict[str, Any]], float]:
        kwh = 0.0
        out: List[Dict[str, Any]] = []
        if not arr:
            return out, 0.0
        denom = max(1.0, float(teu))
        dt_h = float(step_min) / 60.0
        for p in arr:
            v = float(p.get("kW", p.get("p50", 0.0)) or 0.0)
            kwh += v * dt_h
            out.append({"ts": p.get("ts"), "kW": round(kwh / denom, 6)})
        return out, round(kwh, 3)

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------

    def intensity(
        self,
        mode: str = "forecast",
        horizon_min: int = 360,
        step_min: int = 1,
        limit: int = 200,
        teu: float = 12000.0,
        scenario: str = "baseline",
        use_drivers: bool = True,
    ) -> Dict[str, Any]:
        teu_eff, teu_source = self._resolve_teu(teu, horizon_min, step_min)
        if teu_eff <= 0:
            return {
                "mode": mode,
                "available": False,
                "reason": "TEU denominator is required from the request or operations adapter",
                "unit": "kWh/TEU",
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
                "unit": "kWh/TEU",
                "series": {"p50": [], "p10": [], "p90": []},
                "window_min": horizon_min,
                "step_min": step_min,
                "teu": float(teu_eff),
                "teu_source": teu_source,
                "cum_kwh_total": 0.0,
                "intensity_total": 0.0,
            }

        p10 = S.get("p10") or []
        p90 = S.get("p90") or []
        effective_step_min = int(agg.get("_step_min") or step_min)

        s50, kwh_total = self._accumulate_to_intensity(p50, effective_step_min, teu_eff)
        s10, _ = self._accumulate_to_intensity(p10, effective_step_min, teu_eff)
        s90, _ = self._accumulate_to_intensity(p90, effective_step_min, teu_eff)

        return {
            "mode": mode,
            "available": True,
            "unit": "kWh/TEU",
            "series": {"p50": s50, "p10": s10, "p90": s90},
            "window_min": horizon_min,
            "step_min": step_min,
            "step_min_effective": effective_step_min,
            "teu": float(teu_eff),
            "teu_source": teu_source,
            "power_source": agg.get("_source", "unknown"),
            "cum_kwh_total": kwh_total,
            "intensity_total": (s50[-1]["kW"] if s50 else 0.0),
            "uncertainty_available": bool(s10 and s90),
        }
