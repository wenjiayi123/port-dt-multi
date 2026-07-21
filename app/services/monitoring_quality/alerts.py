# ============================================
# app/services/alerts.py
# --------------------------------------------
# 统一的“能耗与碳排”预警服务（AlertsService）
#
# 目标：
#   1) 将三类预警的口径集中、参数化，便于统一维护与调参：
#        - 异常能耗设备（abnormal_energy）
#        - 即将越峰时段（peak_risk）
#        - 碳配额超标风险（carbon_quota）
#   2) 对外暴露一个 scan(...) 方法，返回结构与现有 /api/alerts/scan 保持一致，
#      以便 server.py 替换为“服务调用”，同时前端无需改动。
#   3) 对“异常能耗”使用 ReportingService 的 P95 与近窗均值；
#      对“越峰”采用 ForecastService 的未来聚合；对“碳配额”复用 EnergyService 口径。
#
# 依赖（通过 DI 注入）：
#   - telemetry: list_assets()
#   - reporting: generate_mini_report(asset_id)
#   - forecast:  forecast_load([asset_id], horizon_min, step_min)
#   - energy:    build_today_summary(teu, limit_assets, ...)
#
# 说明：
#   - 本服务只做“即时扫描”，不读库不持久化；
#   - 若某些依赖不可用，会尽量兜底，保证接口不致失败（但会降低准确性）。
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------
# 数据类：更清晰的内部表达
# ------------------------------
@dataclass
class AlertItem:
    type: str          # 'abnormal_energy' | 'peak_risk' | 'carbon_quota' | 'custom'
    title: str
    detail: str
    score: float = 0.0 # [0,1] 可视化强度（排序/着色）
    meta: Optional[Dict[str, Any]] = None


# ------------------------------
# 服务实现
# ------------------------------
class AlertsService:
    """
    统一的预警服务。
    """

    def __init__(self, telemetry, reporting, forecast, energy):
        self.telemetry = telemetry
        self.reporting = reporting
        self.forecast = forecast
        self.energy = energy

    # -------- 工具：取资产清单（带上限） --------
    def _list_assets_limited(self, limit: int) -> List[Dict[str, str]]:
        try:
            assets = self.telemetry.list_assets() or []
        except Exception:
            assets = [{"id": "agv-01", "label": "AGV-01"}]
        if limit > 0:
            assets = assets[:limit]
        return assets

    # -------- 1) 异常能耗（近窗均值 > P95） --------
    def _scan_abnormal_energy(self, assets: List[Dict[str, str]]) -> Tuple[List[AlertItem], int]:
        alerts: List[AlertItem] = []
        abnormal_count = 0

        for a in assets:
            aid = a.get("id")
            try:
                rpt = self.reporting.generate_mini_report(aid) or {}
                avg5 = float(rpt.get("avg_kW_last5min", 0.0))
                p95 = float(rpt.get("p95_kW", float("inf")))
            except Exception:
                avg5, p95 = 0.0, float("inf")

            if avg5 > p95:
                abnormal_count += 1
                ratio = max(1.0, avg5 / max(1e-6, p95))
                score = round(min(1.0, (ratio - 1.0) / 1.0), 2)  # [0,1]
                alerts.append(AlertItem(
                    type="abnormal_energy",
                    title=f"异常能耗：{aid}",
                    detail=f"近窗平均 {avg5:.3f} kW 高于 P95 {p95:.3f} kW，请核查工况/空转/故障。",
                    score=score,
                    meta={"asset": aid, "avg": avg5, "p95": p95, "ratio": ratio}
                ))

        return alerts, abnormal_count

    # -------- 2) 即将越峰：聚合预测超过阈值的 ETA（分钟） --------
    def _scan_peak_risk(self, assets: List[Dict[str, str]], demand_limit_kw: float, horizon_min: int, step_min: int = 1) -> List[AlertItem]:
        # 聚合未来曲线（逐分钟累加）
        L = 0
        seq_map: Dict[str, List[float]] = {}
        for a in assets:
            aid = a.get("id")
            try:
                seq = (self.forecast.forecast_load([aid], horizon_min=horizon_min, step_min=step_min) or {}).get(aid, [])
                vals = [float(p.get("kW", 0.0)) for p in seq if isinstance(p, dict)]
            except Exception:
                vals = []
            seq_map[aid] = vals
            L = max(L, len(vals))

        agg: List[float] = []
        for i in range(L):
            s = 0.0
            for vals in seq_map.values():
                if i < len(vals):
                    s += vals[i]
            agg.append(s)

        # 找到首次超过阈值的位置
        alerts: List[AlertItem] = []
        for i, kw in enumerate(agg):
            if kw >= demand_limit_kw:
                # 越早越高的分数
                score = round((horizon_min - (i * step_min)) / max(1, horizon_min), 2)
                alerts.append(AlertItem(
                    type="peak_risk",
                    title=f"即将越峰（~{i * step_min} 分钟后）",
                    detail=f"聚合预测将在 ~{i * step_min} 分钟后超过阈值 {demand_limit_kw:.0f} kW，建议提前移峰/错峰。",
                    score=score,
                    meta={"eta_min": i * step_min, "limit_kW": demand_limit_kw}
                ))
                break  # 只报第一处（最临近的风险）
        return alerts

    # -------- 3) 碳配额：今日估算排放 > 配额 --------
    def _scan_carbon_quota(self, teu: int, quota_kgco2e: float, limit_assets: int) -> List[AlertItem]:
        try:
            energy = self.energy.build_today_summary(teu=teu, limit_assets=limit_assets)
            kg_per_teu = float(energy.get("intensity", {}).get("kgCO2e_per_TEU", 0.0))
            total_kgco2e = kg_per_teu * max(1, int(teu))
        except Exception:
            total_kgco2e = 0.0

        if total_kgco2e > quota_kgco2e:
            ratio = max(1.0, total_kgco2e / max(1.0, quota_kgco2e))
            score = round(min(1.0, (ratio - 1.0) / 1.0), 2)
            return [AlertItem(
                type="carbon_quota",
                title="碳配额超标风险",
                detail=f"当日估算排放 {total_kgco2e:.1f} kgCO₂e 超过配额 {quota_kgco2e:.1f} kgCO₂e。",
                score=score,
                meta={"total_kgco2e": total_kgco2e, "quota_kgco2e": quota_kgco2e}
            )]
        return []

    # -------- 对外：一次性扫描三类预警 --------
    def scan(
        self,
        teu: int = 12000,
        demand_limit_kw: float = 500.0,
        quota_kgco2e: float = 5000.0,
        limit: int = 50,
        horizon_min: int = 360,
        step_min: int = 1,
    ) -> Dict[str, Any]:
        """
        返回结构示例（与 /api/alerts/scan 对齐）：
        {
          "params": {...},
          "summary": {
              "abnormal_count": 1,
              "will_peak": true,
              "peak_eta_min": 37,
              "quota_over": false,
              "total_kgco2e_est": 4321.5
          },
          "alerts": [ {type,title,detail,score,meta}, ... ]
        }
        """
        assets = self._list_assets_limited(limit)

        # 1) 异常能耗
        abnormal_alerts, abnormal_count = self._scan_abnormal_energy(assets)

        # 2) 越峰
        peak_alerts = self._scan_peak_risk(assets, demand_limit_kw=demand_limit_kw, horizon_min=horizon_min, step_min=step_min)
        will_peak = len(peak_alerts) > 0
        peak_eta_min = peak_alerts[0].meta.get("eta_min") if will_peak else None

        # 3) 碳配额
        carbon_alerts = self._scan_carbon_quota(teu=teu, quota_kgco2e=quota_kgco2e, limit_assets=limit)
        quota_over = len(carbon_alerts) > 0
        total_kgco2e_est = carbon_alerts[0].meta.get("total_kgco2e") if quota_over else 0.0

        # 汇总
        all_alerts = abnormal_alerts + peak_alerts + carbon_alerts
        payload = {
            "params": {
                "teu": teu,
                "demand_limit_kw": demand_limit_kw,
                "quota_kgco2e": quota_kgco2e,
                "limit": limit,
                "horizon_min": horizon_min,
                "step_min": step_min,
            },
            "summary": {
                "abnormal_count": abnormal_count,
                "will_peak": will_peak,
                "peak_eta_min": peak_eta_min,
                "quota_over": quota_over,
                "total_kgco2e_est": round(float(total_kgco2e_est or 0.0), 3),
            },
            "alerts": [
                {
                    "type": a.type,
                    "score": a.score,
                    "title": a.title,
                    "detail": a.detail,
                    **({"meta": a.meta} if a.meta else {})
                } for a in all_alerts
            ]
        }
        return payload
