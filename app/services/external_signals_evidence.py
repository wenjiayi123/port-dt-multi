from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class ExternalSignalsEvidenceService:
    """Expose public replay signals without invoking unconfigured mock clients."""

    def __init__(
        self,
        telemetry: Any,
        *,
        tos: Any = None,
        market: Any = None,
        ais_tide: Any = None,
        schedule: Any = None,
        root: Path | None = None,
    ) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.telemetry = telemetry
        self.adapters = {"tos": tos, "market": market, "ais_tide": ais_tide, "schedule": schedule}
        self.report_path = self.root / "evidence" / "v3" / "shanghai_public_advantage_v3.json"

    @staticmethod
    def _status(adapter: Any) -> Dict[str, Any]:
        fn = getattr(adapter, "source_status", None)
        if not callable(fn):
            return {"mode": "unavailable", "configured": False}
        try:
            result = fn()
            return result if isinstance(result, dict) else {"mode": "unavailable", "configured": False}
        except Exception as exc:
            return {"mode": "unavailable", "configured": False, "error": type(exc).__name__}

    def build(self) -> Dict[str, Any]:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        dataset = report.get("dataset") or {}
        source_status = {name: self._status(adapter) for name, adapter in self.adapters.items()}
        telemetry_status = self._status(self.telemetry)
        rows = self.telemetry.port_state_series(24 * 60, 60) if hasattr(self.telemetry, "port_state_series") else []
        timeline = []
        for row in rows[:24]:
            timeline.append({
                "timestamp": row.get("timestamp"),
                "source_timestamp": row.get("source_timestamp"),
                "demand_kw": round(float(row.get("base_load_kw") or 0.0), 3),
                "price_yuan_per_kwh": round(float(row.get("price_per_kwh") or 0.0), 6),
                "carbon_kg_per_kwh": round(float(row.get("carbon_kg_per_kwh") or 0.0), 6),
                "tide_m": round(float(row.get("tide_m") or 0.0), 4),
                "ambient_c": round(float(row.get("ambient_c") or 0.0), 3),
                "wind_speed_mps": round(float(row.get("wind_speed_mps") or 0.0), 3),
                "wave_height_m": round(float(row.get("wave_height_m") or 0.0), 3),
                "current_speed_mps": round(float(row.get("current_speed_mps") or 0.0), 4),
                "throughput_teu": round(float(row.get("throughput_teu") or 0.0), 3),
                "vessel_arrivals": round(float(row.get("vessel_arrivals") or 0.0), 4),
            })
        signal_registry = [
            {"id": "demand_kw", "name": "港区需量", "availability": "public_replay", "evidence_class": "declared_engineering_derivative", "model_input": True, "unit": "kW"},
            {"id": "price_yuan_per_kwh", "name": "电价", "availability": "public_replay", "evidence_class": "declared_engineering_derivative", "model_input": True, "unit": "CNY/kWh"},
            {"id": "carbon_kg_per_kwh", "name": "电网碳因子", "availability": "public_replay", "evidence_class": "declared_engineering_derivative", "model_input": True, "unit": "kgCO2e/kWh"},
            {"id": "tide_m", "name": "潮位", "availability": "public_replay", "evidence_class": "public_marine_reanalysis", "model_input": True, "unit": "m"},
            {"id": "weather", "name": "气象/海况", "availability": "public_replay", "evidence_class": "public_weather_marine_reanalysis", "model_input": True, "unit": "mixed"},
            {"id": "throughput_teu", "name": "吞吐", "availability": "public_replay", "evidence_class": "official_aggregate_anchor_distributed", "model_input": True, "unit": "TEU/h"},
            {"id": "tos_schedule", "name": "TOS/WMS船期与作业指令", "availability": "待接入港口", "evidence_class": "unavailable", "model_input": False, "unit": None},
            {"id": "ais_tracks", "name": "AIS船位/航迹/ETA", "availability": "待接入港口", "evidence_class": "unavailable", "model_input": False, "unit": None},
            {"id": "live_market", "name": "实时/日前市场与DR", "availability": "待接入港口", "evidence_class": "unavailable", "model_input": False, "unit": None},
        ]
        public_sources = []
        for source in dataset.get("sources") or []:
            public_sources.append({
                "publisher": source.get("publisher"),
                "role": source.get("role"),
                "url": source.get("url") or ((source.get("source_urls") or [None])[0]),
                "source_url_count": len(source.get("source_urls") or ([source.get("url")] if source.get("url") else [])),
                "accessed_at": source.get("accessed_at"),
            })
        live_adapters = sum(
            status.get("mode") == "live_rest"
            or (status.get("ais_mode") == "live_rest" and status.get("tide_mode") == "live_rest")
            for status in source_status.values()
        )
        return {
            "version": "V3",
            "module": {"id": "external_signals", "name": "外部信号", "state": "public_replay_site_adapters_pending"},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "port": {"unlocode": "CNSHA", "name": "上海港公开目标域", "site_terminal": "待接入港口"},
            "boundary": {
                "public_replay_available": bool(timeline),
                "live_external_data_verified": False,
                "production_authority": False,
                "site_status": "待接入港口",
                "reason": "当前可用信号来自上海官方汇总、公开气象/海洋再分析及明示衍生字段；TOS、AIS与实时市场没有现场授权连接。",
            },
            "dataset": {
                "dataset_id": dataset.get("dataset_id"),
                "rows": dataset.get("rows"),
                "sha256": dataset.get("sha256"),
                "start_at": dataset.get("start_at"),
                "end_at": dataset.get("end_at"),
                "independent_source_observations": dataset.get("independent_source_observations"),
                "official_reporting_periods": (dataset.get("source_observation_counts") or {}).get("official_port_reporting_periods"),
                "reanalysis_hours": (dataset.get("source_observation_counts") or {}).get("aligned_public_reanalysis_hours"),
                "measured_columns": dataset.get("measured_columns") or [],
                "official_aggregate_columns": dataset.get("official_aggregate_columns") or [],
                "public_reanalysis_columns": dataset.get("public_reanalysis_columns") or [],
                "derived_columns": dataset.get("derived_columns") or [],
                "unavailable_factors": dataset.get("unavailable_factors") or [],
                "report_sha256": hashlib.sha256(self.report_path.read_bytes()).hexdigest(),
            },
            "telemetry_source": telemetry_status,
            "adapter_status": source_status,
            "live_adapter_count": live_adapters,
            "signal_registry": signal_registry,
            "timeline": timeline,
            "tables": {
                "tos_schedule": {"rows": [], "state": "待接入港口", "reason": "未配置经授权的TOS/WMS船期适配器；不生成虚构航次。"},
                "ais_arrivals": {"rows": [], "state": "待接入港口", "reason": "未配置经授权的AIS提供商；不生成虚构船名、IMO或ETA。"},
                "demand": [{"timestamp": row["timestamp"], "value": row["demand_kw"], "source_timestamp": row["source_timestamp"]} for row in timeline],
                "price": [{"timestamp": row["timestamp"], "value": row["price_yuan_per_kwh"], "source_timestamp": row["source_timestamp"]} for row in timeline],
                "tide": [{"timestamp": row["timestamp"], "value": row["tide_m"], "source_timestamp": row["source_timestamp"]} for row in timeline],
            },
            "public_sources": public_sources,
            "replacement_contract": {
                "stable_interface": ["source_status", "UTC timestamp", "source_timestamp", "value", "unit", "quality", "provenance"],
                "tos_required": ["voyage/call_id", "vessel/IMO/MMSI", "ETA/ETB/ETD", "berth", "move plan", "priority", "revision", "event time"],
                "ais_required": ["MMSI/IMO", "position", "SOG/COG", "nav status", "message time", "provider latency/coverage"],
                "market_required": ["day-ahead/real-time price", "contract demand", "demand charge", "DR event", "carbon/REC/grid factor", "publication time/revision"],
                "gates": ["schema/version", "UTC monotonicity", "freshness SLA", "missing/duplicate/late rate", "source license", "hash/snapshot", "cross-source reconciliation", "fail-closed on stale data"],
            },
        }
