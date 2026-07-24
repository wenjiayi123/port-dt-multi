# ============================================
# app/services/energy.py
# --------------------------------------------
# 能耗与碳排“口径中心”（EnergyService）
#
# 目标：
#   - 统一 /api/energy/today 的聚合口径；
#   - 优先采用“积分累计口径（integral）”：对当天窗口内的 kW·h 做梯形积分；
#   - 覆盖不足时自动回退“均值外推（avg）”：用近窗均值 × 当天已过小时数；
#   - 计算峰/平/谷（TOU）占比、单位 TEU 强度指标、设备利用率；
#   - 保持字段与前端兼容（electricity.kWh、kWh_est、tou_share、avg_carbon_intensity_g_per_kwh 等）。
#
# 依赖（通过 DI 注入）：
#   - telemetry: list_assets(), get_recent_power(asset_id)->[{"ts","kW"},...]（升序或乱序皆可）
#   - reporting: generate_mini_report(asset_id)->含 avg_kW_last5min / carbonIntensity ...
#   - forecast:  forecast_load([asset_id], horizon_min, step_min)->{"id":[{"ts","kW"},...]}
#
# 说明：
#   - 积分口径只使用“当天范围内可用的点”；覆盖阈值不足则整体回退至均值外推；
#   - TOU 占比基于未来 horizon 的预测功率能量分配（仅用于占比显示，不影响今日 kWh）；
#   - 油/气位置留白（返回 0），便于后续对接计量台账。
# ============================================

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math


# ---------- 工具函数 ----------
def _safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return d


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _tou_bucket(hour: float) -> str:
    """
    峰/平/谷示例：
      峰：10:00-15:00, 19:00-21:00
      谷：23:00-07:00
      其余：平
    """
    if (10 <= hour < 15) or (19 <= hour < 21):
        return "peak"
    if hour >= 23 or hour < 7:
        return "valley"
    return "flat"


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


class EnergyService:
    """
    build_today_summary(
        teu=12000,
        limit_assets=50,
        min_integral_coverage_min=30.0,
        horizon_min=360,
        step_min=1
    ) -> Dict
    关键返回字段：
      {
        "range": {"start","end","hours"},
        "electricity": {
            "kWh": <float>,               # 今日电耗（口径见 method）
            "kWh_est": <float>,           # 兼容字段 = kWh
            "by_asset": [{"id","avg_kW","kWh" 或 "kWh_est"}],
            "tou_share": {"peak":..,"flat":..,"valley":..},
            "avg_carbon_intensity_g_per_kwh": <float>,
            "method": "integral" | "avg"
        },
        "oil": {"liters":0.0,"kgCO2e":0.0},
        "gas": {"nm3":0.0,"kWh":0.0,"kgCO2e":0.0},
        "intensity": {"kWh_per_TEU":..,"kgCO2e_per_TEU":..},
        "utilization_percent": <float>,  # 平均设备利用率（%）
        "assumptions": {...}
      }
    """

    def __init__(self, telemetry, reporting, forecast):
        self.telemetry = telemetry
        self.reporting = reporting
        self.forecast = forecast

        # 缓存 label（用于类型识别与利用率估计）
        self._labels: Dict[str, str] = {}
        self._asset_meta: Dict[str, Dict[str, Any]] = {}
        try:
            for a in (self.telemetry.list_assets() or []):
                aid = a.get("id")
                lab = a.get("label", "")
                if aid:
                    self._labels[aid] = lab
                    self._asset_meta[aid] = dict(a)
        except Exception:
            pass

    # ---------- 内部：积分口径 ----------
    def _integral_today(self, asset_ids: List[str], midnight: datetime, now: datetime) -> Tuple[float, List[Dict[str, Any]], float]:
        """
        对每个资产：取“当天范围”的功率点做梯形积分（kWh），并统计覆盖时长（分钟）。
        返回：(total_kWh, by_asset, avg_coverage_min)
        """
        total_kwh = 0.0
        by_asset: List[Dict[str, Any]] = []
        cover_mins = []

        for aid in asset_ids:
            raw = self.telemetry.get_recent_power(aid) or []
            pts = []
            for p in raw:
                if not isinstance(p, dict):
                    continue
                t = _parse_iso(p.get("ts"))
                if not t:
                    continue
                if t < midnight or t > now:
                    continue
                kw = _safe_float(p.get("kW"), 0.0)
                pts.append((t, kw))
            pts.sort(key=lambda x: x[0])

            if len(pts) < 2:
                # 无法积分
                by_asset.append({"id": aid, "avg_kW": 0.0, "kWh": 0.0, "coverage_min": 0.0})
                cover_mins.append(0.0)
                continue

            # 梯形积分
            kwh = 0.0
            last_t, last_kw = pts[0]
            for i in range(1, len(pts)):
                t, kw = pts[i]
                dt_h = max(0.0, (t - last_t).total_seconds() / 3600.0)
                kwh += max(0.0, (last_kw + kw) * 0.5 * dt_h)
                last_t, last_kw = t, kw

            span_min = max(0.0, (pts[-1][0] - pts[0][0]).total_seconds() / 60.0)
            avg_kw = (kwh / (span_min / 60.0)) if span_min > 0 else 0.0

            total_kwh += kwh
            cover_mins.append(span_min)
            by_asset.append({"id": aid, "avg_kW": round(avg_kw, 3), "kWh": round(kwh, 6), "coverage_min": round(span_min, 3)})

        avg_cov = (sum(cover_mins) / len(cover_mins)) if cover_mins else 0.0
        return total_kwh, by_asset, avg_cov

    # ---------- 内部：均值外推 ----------
    def _average_today(self, asset_ids: List[str], midnight: datetime, now: datetime) -> Tuple[float, List[Dict[str, Any]]]:
        """
        用 Reporting 的近窗均值作为“今日代表值”，乘以当日已过小时数得到 kWh 估算。
        """
        hours = max(0.0, (now - midnight).total_seconds() / 3600.0)
        total_kwh = 0.0
        by_asset: List[Dict[str, Any]] = []

        for aid in asset_ids:
            try:
                rpt = self.reporting.generate_mini_report(aid) or {}
                avg_kw = _safe_float(rpt.get("avg_kW_last5min"), 0.0)
            except Exception:
                avg_kw = 0.0
            kwh = max(0.0, avg_kw * hours)
            total_kwh += kwh
            by_asset.append({"id": aid, "avg_kW": round(avg_kw, 3), "kWh_est": round(kwh, 6)})

        return total_kwh, by_asset

    # ---------- 内部：TOU 占比（基于未来预测曲线） ----------
    def _calc_tou_share(self, asset_ids: List[str], horizon_min: int = 360, step_min: int = 1) -> Dict[str, Any]:
        """
        以预测功率曲线估算“未来窗口中的峰/平/谷能量占比”（仅用于展示）。
        """
        try:
            # 汇总到聚合序列
            L = 0
            buckets = {"peak": 0.0, "flat": 0.0, "valley": 0.0}
            for aid in asset_ids:
                seq = (self.forecast.forecast_load([aid], horizon_min=horizon_min, step_min=step_min) or {}).get(aid, [])
                L = max(L, len(seq))
                for i, p in enumerate(seq):
                    kw = _safe_float(p.get("kW"), 0.0)
                    ts = _parse_iso(p.get("ts"))
                    if not ts:
                        continue
                    h = ts.astimezone().hour + ts.astimezone().minute / 60.0
                    b = _tou_bucket(h)
                    effective_step = int(p.get("model_step_min") or step_min)
                    buckets[b] += kw * (effective_step / 60.0)
            tot = sum(buckets.values())
            if tot <= 0:
                return {"peak": None, "flat": None, "valley": None}
            return {k: max(0.0, v / tot) for k, v in buckets.items()}
        except Exception:
            return {"peak": None, "flat": None, "valley": None}

    # ---------- 内部：平均碳强度 ----------
    def _avg_carbon_intensity(self, asset_ids: List[str]) -> Optional[float]:
        vals = []
        for aid in asset_ids:
            try:
                rpt = self.reporting.generate_mini_report(aid) or {}
                raw = rpt.get("carbonIntensity")
                if raw is None:
                    continue
                ci = float(raw)
            except Exception:
                continue
            vals.append(ci)
        if not vals:
            return None
        return sum(vals) / len(vals)

    # ---------- 内部：设备利用率（%） ----------
    def _avg_utilization_percent(self, asset_ids: List[str]) -> Optional[float]:
        """
        优先使用报表里的 utilization_est_percent；没有则用 avg_kw / 额定 推估。
        """
        utils = []
        for aid in asset_ids:
            try:
                rpt = self.reporting.generate_mini_report(aid) or {}
            except Exception:
                rpt = {}

            if rpt.get("utilization_est_percent") is not None:
                u = float(rpt["utilization_est_percent"])
                utils.append(max(0.0, min(100.0, u)))
        return (sum(utils) / len(utils)) if utils else None

    # ---------- 对外：今日汇总 ----------
    def build_today_summary(
        self,
        teu: int = 12000,
        limit_assets: int = 50,
        min_integral_coverage_min: float = 30.0,
        horizon_min: int = 360,
        step_min: int = 1,
    ) -> Dict[str, Any]:
        """
        返回“今日能耗与碳排”汇总，字段与前端完全对齐。
        """
        # 基本时间范围（当天 0 点到当前）
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        hours = max(0.0, (now - midnight).total_seconds() / 3600.0)

        # 资产清单
        try:
            assets = self.telemetry.list_assets() or []
        except Exception:
            assets = []
        assets = assets[: max(1, int(limit_assets))]
        asset_ids = [a["id"] for a in assets]
        if not asset_ids:
            return {"available": False, "reason": "telemetry asset registry is empty", "electricity": {}, "intensity": {}}

        latest_timestamps = []
        for asset_id in asset_ids:
            try:
                rows = self.telemetry.get_recent_power(asset_id) or []
                parsed = [_parse_iso(row.get("ts")) for row in rows if isinstance(row, dict)]
                latest_timestamps.extend(value for value in parsed if value is not None)
            except Exception:
                continue
        if not latest_timestamps:
            return {"available": False, "reason": "telemetry contains no timestamped power rows", "electricity": {}, "intensity": {}}
        latest = max(latest_timestamps)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        if (now - latest.astimezone(timezone.utc)).total_seconds() > 48 * 3600:
            return {
                "available": False,
                "reason": "latest telemetry is older than 48 hours; today summary is not computed from replay history",
                "latest_telemetry_at": latest.isoformat(),
                "electricity": {},
                "intensity": {},
                "_source": "telemetry_stale",
            }

        # ------- 口径选择：积分优先 -------
        total_kwh = 0.0
        elec_payload: Dict[str, Any] = {}
        method = "avg"

        # 尝试积分
        total_int_kwh, by_asset_int, avg_cov_min = self._integral_today(asset_ids, midnight, now)

        # 判断是否满足启用“积分口径”的条件：覆盖分钟数阈值 + 覆盖资产比例
        coverage_flags = [ (x.get("coverage_min", 0.0) >= float(min_integral_coverage_min)) for x in by_asset_int ]
        coverage_ratio = (sum(1 for f in coverage_flags if f) / max(1, len(coverage_flags))) if coverage_flags else 0.0

        if coverage_ratio >= 0.6 and total_int_kwh > 0.0:
            # 采用积分
            method = "integral"
            total_kwh = total_int_kwh
            elec_payload["by_asset"] = by_asset_int
        else:
            # 退回均值外推
            total_avg_kwh, by_asset_avg = self._average_today(asset_ids, midnight, now)
            total_kwh = total_avg_kwh
            elec_payload["by_asset"] = by_asset_avg

        # ------- TOU 占比（与口径无关，仅用于展示） -------
        tou_share = self._calc_tou_share(asset_ids, horizon_min=horizon_min, step_min=step_min)

        # ------- 平均碳强度（g/kWh） -------
        avg_ci = self._avg_carbon_intensity(asset_ids)

        # ------- 设备利用率（%） -------
        util_pct = self._avg_utilization_percent(asset_ids)

        # ------- 碳排（仅电力；油/气留白） -------
        elec_kg = total_kwh * (avg_ci / 1000.0) if avg_ci is not None else None
        teu = max(1, int(teu))
        total_kg = elec_kg

        # ------- 构造返回 -------
        electricity = {
            "kWh": round(total_kwh, 6),
            "kWh_est": round(total_kwh, 6),  # 兼容字段
            "by_asset": elec_payload.get("by_asset", []),
            "tou_share": {k: (float(round(v, 6)) if v is not None else None) for k, v in tou_share.items()},
            "avg_carbon_intensity_g_per_kwh": round(avg_ci, 1) if avg_ci is not None else None,
            "method": method,
        }

        payload = {
            "available": True,
            "range": {"start": midnight.isoformat(), "end": now.isoformat(), "hours": round(hours, 3)},
            "electricity": electricity,
            "oil": {"available": False, "liters": None, "kgCO2e": None},
            "gas": {"available": False, "nm3": None, "kWh": None, "kgCO2e": None},
            "intensity": {
                "kWh_per_TEU": round(total_kwh / teu, 6),
                "kgCO2e_per_TEU": round(total_kg / teu, 6) if total_kg is not None else None,
            },
            "utilization_percent": round(util_pct, 2) if util_pct is not None else None,
            "assumptions": {
                "integral_threshold_min": float(min_integral_coverage_min),
                "integral_coverage_ratio": round(coverage_ratio, 3),
                "integral_avg_coverage_min": round(avg_cov_min, 3),
                "tou_horizon_min": int(horizon_min),
                "tou_step_min": int(step_min),
                "note": "oil/gas 当前留白；TOU 占比基于未来预测，仅用于展示。",
            },
        }
        return payload
