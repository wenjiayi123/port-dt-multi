from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

from .service import CurvesService

_DEFAULT_TOU = [
    ((0, 0), (6, 0), 0.42, "valley"),
    ((6, 0), (10, 0), 0.78, "flat"),
    ((10, 0), (15, 0), 1.18, "peak"),
    ((15, 0), (18, 0), 0.82, "flat"),
    ((18, 0), (21, 0), 1.12, "peak"),
    ((21, 0), (24, 0), 0.72, "flat"),
]


class CurvesEconomicBenefit:
    """
    累计经济收益（¥）分解：
      1) 电费节省 energy
      2) 碳成本节省 carbon
      3) 需量费节省 demand

    真实口径：
      - 允许时变电价 / 时变碳因子 / 时变碳价
      - 上游未提供价格序列时仅使用明确标注的 API 参数/TOU 参数表
      - 收益默认按“净收益”累计，而不是把负收益静默截断为 0
      - 需量费按 15min 滚动均值峰值差结算，可在窗口末端一次性入账

    返回格式保持兼容现有前端：
      series.total / energy / carbon / demand
      components.energy_total / carbon_total / demand_total
      params.price_src / ef_src / carbon_price_src ...
    """

    def __init__(self, di) -> None:
        self.di = di
        self.curves = CurvesService(di)

    # ------------------------------------------------------------------
    # helpers: generic extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_numeric_series(s: Any, n: int, keys: List[str]) -> List[float]:
        vals: List[float] = []
        if isinstance(s, list) and s:
            for x in s:
                v: Optional[float] = None
                if isinstance(x, dict):
                    for k in keys:
                        if k in x and x[k] is not None:
                            try:
                                v = float(x[k])
                                break
                            except Exception:
                                continue
                elif isinstance(x, (int, float)):
                    v = float(x)
                if v is not None:
                    vals.append(v)
        if not vals:
            return []
        if len(vals) < n:
            vals = (vals + [vals[-1]])[:n]
        else:
            vals = vals[:n]
        return vals

    @staticmethod
    def _rolling_avg(vals: List[float], w: int) -> List[float]:
        if w <= 1:
            return vals[:]
        out: List[float] = []
        acc = 0.0
        for i, x in enumerate(vals):
            acc += x
            if i >= w:
                acc -= vals[i - w]
            if i >= w - 1:
                out.append(acc / w)
        return out

    @staticmethod
    def _parse_iso(ts: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return None

    def _fallback_tou_price_series(self, ts_list: List[str], n: int, price_const: float) -> List[float]:
        vals: List[float] = []
        for i in range(n):
            dt = self._parse_iso(ts_list[i]) if i < len(ts_list) else None
            if dt is None:
                vals.append(float(price_const))
                continue
            minute = dt.hour * 60 + dt.minute
            picked = None
            for (sh, sm), (eh, em), price, _tier in _DEFAULT_TOU:
                start = sh * 60 + sm
                end = eh * 60 + em
                if start <= minute < end:
                    picked = price
                    break
            vals.append(float(picked if picked is not None else price_const))
        return vals

    # ------------------------------------------------------------------
    # helpers: price / factor / carbon price sources
    # ------------------------------------------------------------------
    def _get_price_series(
        self,
        ts_list: List[str],
        n: int,
        horizon_min: int,
        step_min: int,
        price_const: float,
    ) -> Tuple[List[float], str]:
        try:
            e = getattr(self.di, "energy", None)
            for name in ("price_series", "get_price_series", "tou_price", "grid_price"):
                f = getattr(e, name, None)
                if callable(f):
                    s = f(horizon_min=horizon_min, step_min=step_min)
                    vals = self._extract_numeric_series(s, n, keys=["price", "y", "p", "kwh_price"])
                    if vals:
                        return vals, "series"
        except Exception:
            pass

        try:
            sch = getattr(self.di, "schedule", None)
            tou_fn = getattr(sch, "tou_tariff", None)
            if callable(tou_fn):
                raw = tou_fn(horizon_min=horizon_min, step_min=step_min)
                vals = self._extract_numeric_series(raw, n, keys=["price", "y", "p", "kwh_price"])
                if vals:
                    return vals, "series"
        except Exception:
            pass

        try:
            adapters = getattr(self.di, "adapters", None)
            mc = getattr(adapters, "market_client", None) if adapters else None
            for name in ("series", "get_series", "price_series"):
                f = getattr(mc, name, None) if mc else None
                if callable(f):
                    s = f(horizon_min=horizon_min, step_min=step_min)
                    vals = self._extract_numeric_series(s, n, keys=["price", "y", "p"])
                    if vals:
                        return vals, "series"
        except Exception:
            pass

        return self._fallback_tou_price_series(ts_list, n, price_const), "parameter_tou_schedule"

    def _get_ef_series(
        self,
        n: int,
        horizon_min: int,
        step_min: int,
        ef_const: float,
    ) -> Tuple[List[float], str]:
        try:
            e = getattr(self.di, "energy", None)
            for name in ("grid_emission_factor", "get_grid_ef", "grid_ef"):
                f = getattr(e, name, None)
                if callable(f):
                    s = f(horizon_min=horizon_min, step_min=step_min)
                    vals = self._extract_numeric_series(s, n, keys=["kg_per_kwh", "ef", "factor"])
                    if vals:
                        return vals, "series"
        except Exception:
            pass
        try:
            adapters = getattr(self.di, "adapters", None)
            cf = getattr(adapters, "carbon_factors", None) if adapters else None
            for name in ("series", "get_series"):
                f = getattr(cf, name, None) if cf else None
                if callable(f):
                    s = f(horizon_min=horizon_min, step_min=step_min)
                    vals = self._extract_numeric_series(s, n, keys=["kg_per_kwh", "ef", "factor"])
                    if vals:
                        return vals, "series"
        except Exception:
            pass
        return [float(ef_const)] * max(0, n), "parameter_constant"

    def _get_carbon_price_series(
        self,
        n: int,
        horizon_min: int,
        step_min: int,
        carbon_price_const_y_per_ton: float,
    ) -> Tuple[List[float], str]:
        try:
            e = getattr(self.di, "energy", None)
            for name in ("carbon_price_series", "get_carbon_price_series"):
                f = getattr(e, name, None)
                if callable(f):
                    s = f(horizon_min=horizon_min, step_min=step_min)
                    vals = self._extract_numeric_series(s, n, keys=["y_per_ton", "price", "y"])
                    if vals:
                        return vals, "series"
        except Exception:
            pass
        return [float(carbon_price_const_y_per_ton)] * max(0, n), "parameter_constant"


    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------
    def benefit(
        self,
        mode: str = "sim",
        horizon_min: int = 360,
        step_min: int = 1,
        scenario_base: str = "baseline",
        scenario_opt: str = "strategy",
        limit: int = 200,
        price_const_y_per_kwh: float = 0.85,
        ef_const_kg_per_kwh: float = 0.55,
        carbon_price_const_y_per_ton: float = 50.0,
        demand_rate_y_per_kw: float = 22.0,
        demand_avg_window_min: int = 15,
        use_drivers: bool = True,
    ) -> Dict[str, Any]:
        # Strategy value must compare a fitted forecast baseline with the
        # hash-verified learned-policy output, never two locally-scaled curves.
        base_mode = "forecast" if mode == "sim" else mode
        base = self.curves.aggregate(
            mode=base_mode,
            horizon_min=horizon_min,
            step_min=step_min,
            limit=limit,
            scenario=scenario_base,
            use_drivers=use_drivers,
        ) or {}
        opt = self.curves.aggregate(
            mode=mode,
            horizon_min=horizon_min,
            step_min=step_min,
            limit=limit,
            scenario=scenario_opt,
            use_drivers=use_drivers,
        ) or {}

        s_b = (base.get("series") or {}).get("p50") or []
        s_o = (opt.get("series") or {}).get("p50") or []
        n = min(len(s_b), len(s_o))

        if n == 0:
            return {
                "mode": mode,
                "available": False,
                "reason": "baseline and strategy curves are required from the requested mode",
                "unit": "¥",
                "series": {"total": [], "energy": [], "carbon": [], "demand": []},
                "components": {"energy_total": 0.0, "carbon_total": 0.0, "demand_total": 0.0},
                "params": {
                    "price_src": "none",
                    "price_avg": 0.0,
                    "ef_src": "none",
                    "ef_avg_kg_per_kwh": 0.0,
                    "carbon_price_src": "none",
                    "carbon_price_avg_y_per_ton": 0.0,
                    "demand_rate_y_per_kw": float(demand_rate_y_per_kw),
                    "window_min": horizon_min,
                    "step_min": step_min,
                    "scenario_base": scenario_base,
                    "scenario_opt": scenario_opt,
                    "settlement_basis": "empty",
                },
            }

        s_b = s_b[:n]
        s_o = s_o[:n]
        ts = [str(p.get("ts") or "") for p in s_b]
        effective_step_min = int(base.get("_step_min") or opt.get("_step_min") or step_min)
        dt_h = float(effective_step_min) / 60.0

        vb = [float(x.get("kW", x.get("p50", 0.0)) or 0.0) for x in s_b]
        vo = [float(x.get("kW", x.get("p50", 0.0)) or 0.0) for x in s_o]
        if all(abs(baseline - strategy) <= 1e-9 for baseline, strategy in zip(vb, vo)):
            return {
                "mode": mode,
                "available": False,
                "reason": "no distinct strategy curve or evaluated policy output is available",
                "unit": "¥",
                "series": {"total": [], "energy": [], "carbon": [], "demand": []},
                "components": {"energy_total": None, "carbon_total": None, "demand_total": None},
                "params": {
                    "price_src": "not_evaluated",
                    "ef_src": "not_evaluated",
                    "carbon_price_src": "not_evaluated",
                    "window_min": horizon_min,
                    "step_min": step_min,
                    "step_min_effective": effective_step_min,
                    "scenario_base": scenario_base,
                    "scenario_opt": scenario_opt,
                    "settlement_basis": "unavailable_without_distinct_policy_output",
                },
            }
        price, price_src = self._get_price_series(ts, n, horizon_min, effective_step_min, price_const_y_per_kwh)
        ef, ef_src = self._get_ef_series(n, horizon_min, effective_step_min, ef_const_kg_per_kwh)
        co2p, cp_src = self._get_carbon_price_series(n, horizon_min, effective_step_min, carbon_price_const_y_per_ton)
        price_avg = sum(price) / n if n else price_const_y_per_kwh
        ef_avg = sum(ef) / n if n else ef_const_kg_per_kwh
        co2p_avg = sum(co2p) / n if n else carbon_price_const_y_per_ton

        energy_acc = 0.0
        carbon_acc = 0.0
        energy_series: List[Dict[str, Any]] = []
        carbon_series: List[Dict[str, Any]] = []
        total_series: List[Dict[str, Any]] = []
        delta_kw_series: List[float] = []

        for i in range(n):
            pb = float(s_b[i].get("kW", s_b[i].get("p50", 0.0)) or 0.0)
            po = float(s_o[i].get("kW", s_o[i].get("p50", 0.0)) or 0.0)
            delta_kw = pb - po
            delta_kw_series.append(delta_kw)

            energy_acc += delta_kw * price[i] * dt_h
            energy_series.append({"ts": ts[i], "y": round(energy_acc, 3)})

            kg_delta = delta_kw * dt_h * ef[i]
            carbon_acc += (kg_delta / 1000.0) * co2p[i]
            carbon_series.append({"ts": ts[i], "y": round(carbon_acc, 3)})

            total_series.append({"ts": ts[i], "y": round(energy_acc + carbon_acc, 3)})

        demand_total = 0.0
        demand_series: List[Dict[str, Any]] = [{"ts": ts[i], "y": 0.0} for i in range(n)]
        if demand_rate_y_per_kw > 0.0:
            w = max(1, int(round(demand_avg_window_min / max(1, effective_step_min))))
            mb = self._rolling_avg(vb, w)
            mo = self._rolling_avg(vo, w)
            peak_b = max(mb) if mb else 0.0
            peak_o = max(mo) if mo else 0.0
            demand_total = round((peak_b - peak_o) * float(demand_rate_y_per_kw), 3)
            if demand_series:
                demand_series[-1]["y"] = demand_total
                total_series[-1]["y"] = round(total_series[-1]["y"] + demand_total, 3)

        return {
            "mode": mode,
            "available": True,
            "unit": "¥",
            "series": {
                "total": total_series,
                "energy": energy_series,
                "carbon": carbon_series,
                "demand": demand_series,
            },
            "components": {
                "energy_total": energy_series[-1]["y"] if energy_series else 0.0,
                "carbon_total": carbon_series[-1]["y"] if carbon_series else 0.0,
                "demand_total": demand_total,
            },
            "params": {
                "price_src": price_src,
                "price_avg": round(price_avg, 6),
                "ef_src": ef_src,
                "ef_avg_kg_per_kwh": round(ef_avg, 6),
                "carbon_price_src": cp_src,
                "carbon_price_avg_y_per_ton": round(co2p_avg, 6),
                "demand_rate_y_per_kw": float(demand_rate_y_per_kw),
                "window_min": horizon_min,
                "step_min": step_min,
                "step_min_effective": effective_step_min,
                "scenario_base": scenario_base,
                "scenario_opt": scenario_opt,
                "settlement_basis": "net_benefit",
                "delta_kw_min": round(min(delta_kw_series), 6) if delta_kw_series else 0.0,
                "delta_kw_max": round(max(delta_kw_series), 6) if delta_kw_series else 0.0,
            },
        }
