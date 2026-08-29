from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


class MonitoringEvidenceService:
    """Produce a provenance-labelled, non-authoritative monitoring snapshot."""

    def __init__(self, telemetry: Any, monitoring: Any) -> None:
        self.telemetry = telemetry
        self.monitoring = monitoring
        self._cache_lock = threading.Lock()
        self._cache_at = 0.0
        self._cache: Dict[str, Any] | None = None
        # A dashboard load fans one snapshot into Monitoring, OpsX, MLOps and
        # Governance. One wall minute is short enough for the replay clock and
        # prevents the same 24-hour PSI window from being recomputed four times.
        self._cache_ttl_seconds = 60.0

    @staticmethod
    def _psi(base_vals: List[float], recent_vals: List[float], bins: int = 12) -> Dict[str, Any]:
        if len(base_vals) < 10 or len(recent_vals) < 10:
            return {"available": False, "psi": None, "level": "insufficient", "bins": []}
        lo, hi = min(base_vals + recent_vals), max(base_vals + recent_vals)
        if hi <= lo:
            hi = lo + 1e-6
        width = (hi - lo) / bins
        edges = [lo + index * width for index in range(bins + 1)]

        def distribution(values: List[float]) -> List[float]:
            counts = [0] * bins
            for value in values:
                index = min(bins - 1, max(0, int((value - lo) / width)))
                counts[index] += 1
            total = sum(counts) or 1
            return [count / total for count in counts]

        ref, current = distribution(base_vals), distribution(recent_vals)
        details = []
        total_psi = 0.0
        for index, (p_raw, q_raw) in enumerate(zip(ref, current)):
            p, q = max(1e-9, p_raw), max(1e-9, q_raw)
            contribution = (q - p) * math.log(q / p)
            total_psi += contribution
            details.append({
                "lo": round(edges[index], 6),
                "hi": round(edges[index + 1], 6),
                "p_ref": round(p, 9),
                "p_cur": round(q, 9),
                "psi": round(contribution, 9),
            })
        level = "ok" if total_psi < 0.1 else ("warn" if total_psi < 0.25 else "drift")
        return {"available": True, "psi": round(total_psi, 9), "level": level, "bins": details}

    def build(self) -> Dict[str, Any]:
        """Share one expensive PSI/anomaly snapshot across dependent V3 panels."""
        with self._cache_lock:
            now = time.monotonic()
            if self._cache is not None and now - self._cache_at < self._cache_ttl_seconds:
                return self._cache
            payload = self._build_uncached()
            self._cache = payload
            self._cache_at = time.monotonic()
            return payload

    def _build_uncached(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        recent_start = now - timedelta(hours=2)
        baseline_start = recent_start - timedelta(hours=24)
        assets = [row for row in (self.telemetry.list_assets() or []) if isinstance(row, dict)]
        asset_ids = [str(row.get("id") or row.get("asset_id")) for row in assets if row.get("id") or row.get("asset_id")]
        monitored_assets = asset_ids[:3] or ["qc-01"]
        anomaly = self.monitoring.scan_anomalies(
            asset_ids=monitored_assets,
            point="active_power_kw",
            asset_type="generic",
            start_ts=recent_start.timestamp(),
            end_ts=now.timestamp(),
            step_sec=60,
            method="iqr",
            sensitivity=1.5,
            residual=False,
        )
        anomaly_items = anomaly.get("items") or []
        anomaly_total = sum(len(item.get("anomalies") or []) for item in anomaly_items)
        quality = {
            str(item.get("asset_id") or item.get("asset") or "unknown"): item.get("quality") or {}
            for item in anomaly_items
        }

        baseline, baseline_quality, baseline_source = self.monitoring._load_series(
            monitored_assets[0], "active_power_kw", baseline_start.timestamp(), recent_start.timestamp(), 60, "generic"
        )
        recent, recent_quality, recent_source = self.monitoring._load_series(
            monitored_assets[0], "active_power_kw", recent_start.timestamp(), now.timestamp(), 60, "generic"
        )
        drift = self._psi([value for _, value in baseline], [value for _, value in recent])
        drift.update({
            "asset_id": monitored_assets[0],
            "point": "active_power_kw",
            "baseline": {"start": baseline_start.isoformat(), "end": recent_start.isoformat(), "n": len(baseline), "quality": baseline_quality},
            "recent": {"start": recent_start.isoformat(), "end": now.isoformat(), "n": len(recent), "quality": recent_quality},
            "source": recent_source or baseline_source,
            "warning": "PSI is seasonality-sensitive; it is an admission diagnostic, not a site incident or root-cause conclusion.",
        })
        source_status_fn = getattr(self.telemetry, "source_status", None)
        source = source_status_fn() if callable(source_status_fn) else {"mode": "unknown", "measured": False, "production": False}
        analysis_available = bool(anomaly_items and drift.get("available"))
        current_gate = "block_to_safe_baseline" if drift.get("level") == "drift" else ("review" if drift.get("level") == "warn" else "pass_offline_analysis_only")
        return {
            "version": "V3",
            "module": {"id": "monitoring", "name": "监控中心", "state": current_gate},
            "generated_at": now.isoformat(),
            "boundary": {
                "analysis_available": analysis_available,
                "live_data_verified": bool(source.get("measured") and source.get("production")),
                "incident_claim_eligible": False,
                "production_authority": False,
                "site_status": "待接入港口",
                "reason": "当前为公开数据校准回放，可验证算法和接口；告警、工单、传感器质量SLA与控制回执须由现场系统确认。",
            },
            "source": source,
            "current_analysis": {
                "anomaly": {
                    "method": "IQR",
                    "sensitivity": 1.5,
                    "window_minutes": 120,
                    "assets": monitored_assets,
                    "asset_count": len(anomaly_items),
                    "sample_count": sum(int(round((item.get("quality") or {}).get("completeness", 0) * 121)) for item in anomaly_items),
                    "anomaly_count": anomaly_total,
                    "quality": quality,
                    "items": anomaly_items,
                },
                "drift": drift,
                "admission_decision": {
                    "decision": current_gate,
                    "new_policy_suggestions_allowed": current_gate != "block_to_safe_baseline",
                    "site_command_allowed": False,
                    "fallback": "保持上一稳定策略或FCFS/MPC安全基线",
                    "requires_human_review": current_gate in {"block_to_safe_baseline", "review"},
                },
            },
            "method_registry": [
                {"name": "IQR", "purpose": "稳健点异常", "implemented": True, "caveat": "对多模态工况需分组基线"},
                {"name": "Z-Score", "purpose": "标准化点异常", "implemented": True, "caveat": "假设近似稳定分布"},
                {"name": "EWMA", "purpose": "缓慢偏移", "implemented": True, "caveat": "需按设备调alpha/阈值"},
                {"name": "PSI", "purpose": "窗口分布漂移", "implemented": True, "caveat": "需季节/班次分层"},
                {"name": "预测残差监控", "purpose": "实际值-模型预测", "implemented": True, "caveat": "需现场预测时间对齐"},
                {"name": "数据质量门", "purpose": "完整性/及时性/有效性", "implemented": True, "caveat": "现场阈值待运营方签字"},
            ],
            "alert_policy": {
                "levels": [
                    {"level": "P0", "condition": "硬约束/安全联锁/持续断链", "action": "立即熔断并回退，双人确认"},
                    {"level": "P1", "condition": "PSI>=0.25或关键质量门失败", "action": "停止新策略建议，创建工单"},
                    {"level": "P2", "condition": "0.10<=PSI<0.25或点异常聚集", "action": "人工复核与分层诊断"},
                    {"level": "P3", "condition": "单点低置信异常", "action": "观察并抑制重复告警"},
                ],
                "deduplication": "asset + point + rule + rolling window",
                "lifecycle": ["detected", "acknowledged", "assigned", "mitigated", "verified", "closed"],
                "site_state": "待接入港口告警中心/CMMS/工单回执",
            },
            "site_contract": {
                "inputs": ["TSDB/OPC-UA/MQTT时序", "资产点表与单位", "设备模式/作业上下文", "预测版本与生成时间", "告警确认/工单/维护窗口", "PLC/TOS执行回执"],
                "quality_sla": ["端到端时钟偏差", "缺失/重复/迟到率", "物理边界", "采样周期", "跨测点一致性", "数据血缘与校准版本"],
                "outputs": ["异常事件", "漂移报告", "策略准入门", "降级原因", "工单关联", "恢复准入审计"],
                "replacement": str(source.get("replacement_contract") or "implement list_assets/get_series/source_status"),
            },
        }
