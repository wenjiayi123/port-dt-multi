from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class CurvesBessCapability:
    """
    BESS 调度能力曲线。只接受储能适配器或已训练策略预演的真实返回；
    无适配器时明确返回 unavailable，不从额定参数合成展示曲线。
    """

    def __init__(self, di) -> None:
        self.di = di

    def capability(
        self,
        asset_id: str = "bess-01",
        mode: str = "forecast",
        horizon_min: int = 120,
        step_min: int = 1,
        rating_kw: float = 1000.0,
        energy_mwh: float = 2.0,
        soc_init_pct: float = 60.0,
        soc_min_pct: float = 20.0,
        soc_max_pct: float = 90.0,
    ) -> Dict[str, Any]:
        params = self._merge_asset_params(
            asset_id=asset_id,
            rating_kw=rating_kw,
            energy_mwh=energy_mwh,
            soc_init_pct=soc_init_pct,
            soc_min_pct=soc_min_pct,
            soc_max_pct=soc_max_pct,
        )

        data = self._try_energy_api(asset_id, horizon_min, step_min)
        if not data:
            data = self._try_rl_engine(asset_id, horizon_min, step_min)
        if not data:
            return {
                "mode": mode,
                "asset_id": asset_id,
                "available": False,
                "reason": "No BESS capability or policy-preview adapter returned data",
                "series": {"soc_pct": [], "soc_min": [], "soc_max": [], "charge_cap_kw": [], "discharge_cap_kw": []},
                "totals": {},
                "params": params,
            }

        ts = data.get("ts", [])
        n = len(ts)
        soc = data.get("soc_pct", [params["soc_init_pct"]] * n)
        soc_min = [float(params["soc_min_pct"])] * len(ts)
        soc_max = [float(params["soc_max_pct"])] * len(ts)
        terminal_soc = float(soc[-1]) if soc else float(params["soc_init_pct"])
        # Available energy is SOC/headroom constrained; integrating an
        # instantaneous power-capability line would overstate MWh flexibility.
        up_mwh = max(0.0, terminal_soc - float(params["soc_min_pct"])) / 100.0 * float(params["energy_mwh"])
        down_mwh = max(0.0, float(params["soc_max_pct"]) - terminal_soc) / 100.0 * float(params["energy_mwh"])

        return {
            "mode": mode,
            "available": True,
            "source": data.get("source", "capability_adapter"),
            "asset_id": asset_id,
            "unit_caps": "kW",
            "unit_soc": "%",
            "series": {
                "soc_pct": [{"ts": ts[i], "pct": float(soc[i])} for i in range(len(ts))],
                "soc_min": [{"ts": ts[i], "pct": float(soc_min[i])} for i in range(len(ts))],
                "soc_max": [{"ts": ts[i], "pct": float(soc_max[i])} for i in range(len(ts))],
                "charge_cap_kw": [
                    {"ts": ts[i], "kW": float(data["charge_cap_kw"][i])} for i in range(len(ts))
                ],
                "discharge_cap_kw": [
                    {"ts": ts[i], "kW": float(data["discharge_cap_kw"][i])} for i in range(len(ts))
                ],
            },
            "totals": {
                "energy_up_mwh": round(up_mwh, 4),
                "energy_down_mwh": round(down_mwh, 4),
            },
            "params": {
                "rating_kw": float(params["rating_kw"]),
                "energy_mwh": float(params["energy_mwh"]),
                "soc_init_pct": float(params["soc_init_pct"]),
                "soc_min_pct": float(params["soc_min_pct"]),
                "soc_max_pct": float(params["soc_max_pct"]),
                "step_min": int(step_min),
                "horizon_min": int(horizon_min),
            },
        }

    def _merge_asset_params(
        self,
        asset_id: str,
        rating_kw: float,
        energy_mwh: float,
        soc_init_pct: float,
        soc_min_pct: float,
        soc_max_pct: float,
    ) -> Dict[str, float]:
        out = {
            "rating_kw": float(rating_kw),
            "energy_mwh": float(energy_mwh),
            "soc_init_pct": float(soc_init_pct),
            "soc_min_pct": float(soc_min_pct),
            "soc_max_pct": float(soc_max_pct),
        }
        runtime = getattr(self.di, "strategy_runtime", None)
        runtime_params = getattr(runtime, "bess_parameters", None) if runtime is not None else None
        if callable(runtime_params):
            try:
                configured = runtime_params() or {}
                for key in out:
                    if configured.get(key) is not None:
                        out[key] = float(configured[key])
            except Exception:
                pass
        try:
            assets = None
            energy = getattr(self.di, "energy", None)
            if energy is not None:
                for name in ("assets", "list_assets", "asset_registry", "get_assets"):
                    fn = getattr(energy, name, None)
                    if callable(fn):
                        assets = fn() or []
                        if assets:
                            break
            if isinstance(assets, list):
                hit = None
                for x in assets:
                    if not isinstance(x, dict):
                        continue
                    xid = str(x.get("asset_id", x.get("id", ""))).lower()
                    if xid == str(asset_id).lower():
                        hit = x
                        break
                if isinstance(hit, dict):
                    out["rating_kw"] = float(hit.get("rating_kw", hit.get("power_kw", out["rating_kw"])) or out["rating_kw"])
                    out["energy_mwh"] = float(hit.get("energy_mwh", hit.get("capacity_mwh", out["energy_mwh"])) or out["energy_mwh"])
                    out["soc_init_pct"] = float(hit.get("soc_pct", hit.get("soc_init_pct", out["soc_init_pct"])) or out["soc_init_pct"])
                    out["soc_min_pct"] = float(hit.get("soc_min_pct", out["soc_min_pct"]) or out["soc_min_pct"])
                    out["soc_max_pct"] = float(hit.get("soc_max_pct", out["soc_max_pct"]) or out["soc_max_pct"])
        except Exception:
            pass
        out["soc_min_pct"] = max(0.0, min(100.0, out["soc_min_pct"]))
        out["soc_max_pct"] = max(out["soc_min_pct"] + 1.0, min(100.0, out["soc_max_pct"]))
        out["soc_init_pct"] = max(out["soc_min_pct"], min(out["soc_max_pct"], out["soc_init_pct"]))
        return out

    def _try_energy_api(self, asset_id: str, horizon_min: int, step_min: int) -> Optional[Dict[str, Any]]:
        src = getattr(self.di, "energy", None)
        for name in ("bess_capability", "get_bess_capability"):
            fn = getattr(src, name, None) if src else None
            if callable(fn):
                try:
                    arr = fn(asset_id=asset_id, horizon_min=horizon_min, step_min=step_min) or []
                    normalized = self._normalize_capability(arr)
                    if normalized:
                        normalized["source"] = f"energy.{name}"
                    return normalized
                except Exception:
                    pass
        return None

    def _try_rl_engine(self, asset_id: str, horizon_min: int, step_min: int) -> Optional[Dict[str, Any]]:
        runtime = getattr(self.di, "strategy_runtime", None)
        fn = getattr(runtime, "bess_capability", None) if runtime is not None else None
        if callable(fn):
            try:
                rows = fn(asset_id=asset_id, horizon_min=horizon_min, step_min=step_min) or []
                normalized = self._normalize_capability(rows)
                if normalized:
                    normalized["source"] = "selected_hash_verified_policy_runtime"
                    return normalized
            except Exception:
                pass
        paths = [
            ("rl_model", "bess_energy", "rl_engine"),
            ("services", "rl_model", "bess_energy", "rl_engine"),
        ]
        for p in paths:
            node = self.di
            ok = True
            for seg in p:
                node = getattr(node, seg, None)
                if node is None:
                    ok = False
                    break
            if not ok:
                continue
            for name in ("preview", "simulate", "simulate_open_loop"):
                fn = getattr(node, name, None)
                if callable(fn):
                    try:
                        arr = fn(asset_id=asset_id, horizon_min=horizon_min, step_min=step_min) or []
                        norm = self._normalize_capability(arr)
                        if norm:
                            norm["source"] = ".".join((*p, name))
                            return norm
                    except Exception:
                        pass
        return None

    @staticmethod
    def _normalize_capability(arr: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(arr, list) or not arr:
            return None
        ts, soc, pc, pd = [], [], [], []
        for x in arr:
            if not isinstance(x, dict):
                continue
            ts.append(x.get("ts"))
            s = x.get("soc_pct", x.get("soc", 50.0))
            soc.append(float(s))
            c = x.get("charge_cap_kw", x.get("p_charge", 0.0))
            d = x.get("discharge_cap_kw", x.get("p_discharge", 0.0))
            pc.append(max(0.0, float(c or 0.0)))
            pd.append(max(0.0, float(d or 0.0)))
        if not ts:
            return None
        return {"ts": ts, "soc_pct": soc, "charge_cap_kw": pc, "discharge_cap_kw": pd}

    @staticmethod
    def _accumulate_energy(charge_cap_kw: List[float], discharge_cap_kw: List[float], step_min: int) -> Tuple[float, float]:
        dt_h = float(step_min) / 60.0
        up_mwh = sum(max(0.0, float(p)) * dt_h for p in discharge_cap_kw) / 1000.0
        down_mwh = sum(max(0.0, float(p)) * dt_h for p in charge_cap_kw) / 1000.0
        return up_mwh, down_mwh
