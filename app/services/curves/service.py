# app/services/curves/service.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta


class CurvesService:
    """
    统一提供设备/聚合曲线：
      - mode="now"      -> 最近实时点（按索引对齐后求和）
      - mode="forecast" -> di.fcst.forecast_load(…, return_quantiles=True)
      - mode="sim"      -> di.twin.run / services.sim_aggregate.aggregate_sim

    所有模式的返回格式统一为:
      {
        "mode": "...",
        "series": {
          "p50": [{"ts": ..., "kW": ...}],
          "p10": [{"ts": ..., "kW": ...}],
          "p90": [{"ts": ..., "kW": ...}],
        },
      }

    数据契约：没有遥测、预测或仿真结果时返回 available=false；不得为展示目的
    合成功率、时间戳或分位区间。p10/p90 仅在上游模型真实返回分位预测时提供。
    """

    def __init__(self, di) -> None:
        self.di = di

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------

    def _list_asset_ids(self, limit: int) -> List[str]:
        """从 telemetry 列表中提取资产 id，做一层容错。"""
        try:
            assets = self.di.telemetry.list_assets() or []
        except Exception:
            return []

        ids: List[str] = []
        for a in assets:
            if not isinstance(a, dict):
                continue
            aid = a.get("id") or a.get("asset_id")
            if not aid:
                continue
            ids.append(str(aid))
            if len(ids) >= limit:
                break
        return ids

    def _call_forecast_load(
        self,
        ids: List[str],
        horizon_min: int,
        step_min: int,
        scenario: str,
        use_drivers: bool,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """统一封装 di.fcst.forecast_load 调用，尽量把场景信息透传进去。

        做了几件事：
        1) 优先尝试带 drivers/scenario 的新接口形态；
        2) 若底层不支持这些参数，则自动回退到旧接口；
        3) 异常返回 {}，由上层明确标记数据不可用。
        """
        fcst = getattr(self.di, "fcst", None)
        if fcst is None:
            return {}
        fn = getattr(fcst, "forecast_load", None)
        if not callable(fn):
            return {}

        # 构造 drivers（如果上层希望使用 drivers）
        drivers: Optional[Any] = None
        if use_drivers:
            for name in ("get_drivers", "build_drivers"):
                get_drivers = getattr(fcst, name, None)
                if callable(get_drivers):
                    try:
                        # 兼容两种常见签名：
                        #   get_drivers(scenario=..., horizon_min=..., step_min=...)
                        #   get_drivers(scenario=...)
                        try:
                            drivers = get_drivers(
                                scenario=scenario,
                                horizon_min=horizon_min,
                                step_min=step_min,
                            )
                        except TypeError:
                            drivers = get_drivers(scenario=scenario)
                    except Exception:
                        drivers = None
                    break
            if drivers is None:
                # 最保底：把 scenario 塞到一个 dict 里传下去
                drivers = {"scenario": scenario}

        # 依次尝试多种参数组合，避免接口签名不一致导致报错
        combos = [
            dict(with_scenario=True, with_drivers=True),
            dict(with_scenario=True, with_drivers=False),
            dict(with_scenario=False, with_drivers=True),
            dict(with_scenario=False, with_drivers=False),
        ]

        for combo in combos:
            kwargs: Dict[str, Any] = {
                "horizon_min": horizon_min,
                "step_min": step_min,
                "return_quantiles": True,
            }
            if combo["with_scenario"]:
                kwargs["scenario"] = scenario
            if combo["with_drivers"]:
                kwargs["drivers"] = drivers if use_drivers else None
            try:
                fmap = fn(ids, **kwargs) or {}
                if isinstance(fmap, dict):
                    return fmap
            except TypeError:
                # 参数不兼容，继续尝试下一种组合
                continue
            except Exception:
                return {}

        # 最旧的兜底形态（不带任何额外参数）
        try:
            return fn(ids, horizon_min=horizon_min, step_min=step_min) or {}
        except Exception:
            return {}

    @staticmethod
    def _iso_series_from_window(data: Dict[str, Any], n: int, step_min: int) -> List[str]:
        """Build a timestamp axis for aggregate_sim() payloads that only expose window metadata."""
        if n <= 0:
            return []
        window = data.get("window") or {}
        raw_start = window.get("start") or data.get("updated")
        try:
            start = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00")) if raw_start else datetime.now(timezone.utc)
        except Exception:
            start = datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        return [(start + timedelta(minutes=i * max(1, int(step_min) or 1))).isoformat() for i in range(n)]

    # ------------------------------------------------------------------
    # 单资产曲线
    # ------------------------------------------------------------------

    def asset(
        self,
        asset_id: str,
        mode: str = "forecast",
        horizon_min: int = 360,
        step_min: int = 1,
        scenario: str = "baseline",
        use_drivers: bool = True,
    ) -> Dict[str, Any]:
        if mode == "now":
            try:
                arr = self.di.telemetry.get_recent_power(asset_id) or []
            except Exception:
                arr = []
            out = [
                {
                    "ts": p.get("ts"),
                    "kW": float(p.get("kW", p.get("value", 0.0))),
                }
                for p in arr
                if isinstance(p, dict)
            ]
            return {
                "asset": asset_id,
                "available": bool(out),
                "series": {"p50": out, "p10": [], "p90": []},
                "_source": "telemetry_adapter",
            }

        if mode == "sim":
            # Twin 兼容口径：只传 asset_id，其他窗口参数由 Twin 内部决定
            try:
                data = self.di.twin.run(asset_id=asset_id) or {}
            except Exception:
                data = {}
            seq = data.get("plan", []) or []
            p50 = [
                {"ts": p.get("ts"), "kW": float(p.get("kW", p.get("p50", 0.0)))}
                for p in seq
            ]
            p10 = [{"ts": p.get("ts"), "kW": float(p["p10"])} for p in seq if "p10" in p]
            p90 = [{"ts": p.get("ts"), "kW": float(p["p90"])} for p in seq if "p90" in p]
            return {"asset": asset_id, "available": bool(p50), "series": {"p50": p50, "p10": p10, "p90": p90}, "_source": "twin_adapter"}

        # forecast
        fmap = self._call_forecast_load(
            [asset_id],
            horizon_min=horizon_min,
            step_min=step_min,
            scenario=scenario,
            use_drivers=use_drivers,
        )
        seq = fmap.get(asset_id, []) or []
        p50 = [
            {"ts": p.get("ts"), "kW": float(p.get("kW", p.get("p50", 0.0)))}
            for p in seq
        ]
        p10 = [{"ts": p.get("ts"), "kW": float(p["p10"])} for p in seq if "p10" in p]
        p90 = [{"ts": p.get("ts"), "kW": float(p["p90"])} for p in seq if "p90" in p]
        effective_step = int(seq[0].get("model_step_min") or step_min) if seq else step_min
        return {"asset": asset_id, "available": bool(p50), "series": {"p50": p50, "p10": p10, "p90": p90}, "_source": "forecast_adapter", "_step_min": effective_step}

    # ------------------------------------------------------------------
    # 聚合曲线（全港）
    # ------------------------------------------------------------------

    def aggregate(
        self,
        mode: str = "forecast",
        horizon_min: int = 360,
        step_min: int = 1,
        limit: int = 50,
        scenario: str = "baseline",
        use_drivers: bool = True,
    ) -> Dict[str, Any]:
        ids = self._list_asset_ids(limit)
        empty = {
            "mode": mode,
            "available": False,
            "series": {"p50": [], "p10": [], "p90": []},
            "_source": "curves.unavailable",
        }
        if not ids:
            return empty
        

        # -------------------------- 仿真模式 --------------------------
        if mode == "sim":
            # 优先用已有的聚合仿真实现（如果上层已经实现了更复杂的港口模型，就直接复用）
            try:
                from app.services.forecast_twin.sim_aggregate import aggregate_sim  # type: ignore

                data = (
                    aggregate_sim(
                        self.di,
                        scenario=scenario,
                        horizon_min=horizon_min,
                        step_min=step_min,
                        limit=limit,
                    )
                    or {}
                )
                agg = data.get("agg") if isinstance(data.get("agg"), dict) else {}
                raw_p50 = data.get("p50") or agg.get("p50") or []
                raw_p10 = data.get("p10") or agg.get("p10") or []
                raw_p90 = data.get("p90") or agg.get("p90") or []
                max_len = max(len(raw_p50), len(raw_p10), len(raw_p90), 0)
                ts = data.get("ts") or self._iso_series_from_window(data, max_len, step_min)

                def pack(arr: List[float]) -> List[Dict[str, Any]]:
                    L = min(len(ts), len(arr))
                    return [
                        {"ts": ts[i], "kW": float(arr[i])}
                        for i in range(L)
                    ]

                p50 = pack(raw_p50)
                p10 = pack(raw_p10)
                p90 = pack(raw_p90)

                # 仿真空时明确不可用，不跨模式或生成替代曲线。
                if len(p50) == 0:
                    return empty

                return {
                    "mode": mode,
                    "available": True,
                    "series": {"p50": p50, "p10": p10, "p90": p90},
                    "_source": "twin_adapter",
                    "_step_min": int((data.get("window") or {}).get("step_min_effective") or step_min),
                }

            except Exception:
                return empty

        # -------------------------- 实时模式 --------------------------
        if mode == "now":
            seqs: List[List[float]] = []
            for aid in ids:
                try:
                    arr = self.di.telemetry.get_recent_power(aid) or []
                except Exception:
                    continue
                seq = [
                    float(p.get("kW", p.get("value", 0.0)))
                    for p in arr
                    if isinstance(p, dict)
                ]
                if seq:
                    seqs.append(seq[-360:])  # 最近 360 点
            L = min((len(s) for s in seqs), default=0)
            if L == 0:
                return empty

            s = [sum(s[-L:][i] for s in seqs) for i in range(L)]
            p50 = [{"ts": None, "kW": round(v, 3)} for v in s]
            return {"mode": mode, "available": True, "series": {"p50": p50, "p10": [], "p90": []}, "_source": "telemetry_aggregate"}

        # -------------------------- 预测模式 --------------------------
        fmap = self._call_forecast_load(
            ids,
            horizon_min=horizon_min,
            step_min=step_min,
            scenario=scenario,
            use_drivers=use_drivers,
        )
        if not fmap:
            return empty

        L = min((len(fmap.get(a, [])) for a in ids), default=0)
        if L == 0:
            return empty

        p50_arr: List[float] = [0.0] * L
        p10_arr: List[float] = [0.0] * L
        p90_arr: List[float] = [0.0] * L
        ts_list: List[Any] = [None] * L
        has_p10 = all(all("p10" in point for point in (fmap.get(a, [])[:L])) for a in ids)
        has_p90 = all(all("p90" in point for point in (fmap.get(a, [])[:L])) for a in ids)

        for i in range(L):
            s50 = s10 = s90 = 0.0
            t: Optional[Any] = None
            for aid in ids:
                arr = fmap.get(aid, [])
                if i < len(arr):
                    p = arr[i]
                    t = t or p.get("ts")
                    s50 += float(p.get("kW", p.get("p50", 0.0)))
                    if has_p10:
                        s10 += float(p["p10"])
                    if has_p90:
                        s90 += float(p["p90"])
            ts_list[i] = t
            p50_arr[i] = round(s50, 3)
            p10_arr[i] = round(s10, 3)
            p90_arr[i] = round(s90, 3)

        def pack_agg(arr: List[float]) -> List[Dict[str, Any]]:
            return [{"ts": ts_list[i], "kW": arr[i]} for i in range(L)]

        return {
            "mode": mode,
            "available": True,
            "series": {
                "p50": pack_agg(p50_arr),
                "p10": pack_agg(p10_arr) if has_p10 else [],
                "p90": pack_agg(p90_arr) if has_p90 else [],
            },
            "_source": "forecast_adapter",
            "_step_min": int(next(iter(fmap.values()))[0].get("model_step_min") or step_min),
        }
