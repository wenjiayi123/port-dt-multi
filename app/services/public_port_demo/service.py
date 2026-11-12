"""A coherent presentation feed calibrated to cited Yangshan IV public parameters.

The old model datasets/artifacts are immutable inputs. This feed is an explicitly
separate operational scenario, replaceable via one validated JSON snapshot.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.adapters.telemetry_calibrated_replay import ASSET_SPECS, CalibratedReplayTelemetry

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "data/public_sources/yangshan_phase4_presentation.json"
DATASET = ROOT / "data/rl/datasets/public_cn_sha_hourly_v3.csv"
STARTED = time.monotonic()
GROUP_LABELS = {
    "qc-01": "岸桥机群 A", "qc-02": "岸桥机群 B", "yard-01": "轨道吊机群",
    "agv-01": "AGV 充换电机群", "bess-01": "储能辅助负荷", "hvac-01": "建筑冷站工程负荷",
    "reefer-01": "冷藏箱工程负荷", "shore-01": "船舶岸电工程负荷", "gate-01": "闸口与联运工程负荷",
    "lighting-01": "堆场照明工程负荷", "aux-01": "其他辅助工程负荷",
}


@lru_cache(maxsize=1)
def profile() -> dict[str, Any]:
    data = json.loads(PROFILE.read_text(encoding="utf-8"))
    return {**data, "sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest()}


@lru_cache(maxsize=1)
def inputs() -> tuple[list[dict[str, str]], float, float, str]:
    with DATASET.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["timestamp"].startswith("2025-")]
    if len(rows) != 8760:
        raise ValueError("presentation calibration requires the complete 2025 hourly year")
    target = profile()["scenario_parameters"]["annual_teu_calibration_target"]["value"]
    scale = target / sum(float(r["throughput_teu"]) for r in rows)
    arrivals = sum(float(r["vessel_arrivals"]) * scale for r in rows)
    return rows, scale, arrivals, hashlib.sha256(DATASET.read_bytes()).hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def clock(at: str | None = None) -> datetime:
    if at:
        return _timestamp(at)
    start = _timestamp(profile()["scenario_parameters"]["replay_start"]["value"])
    # One real second is one scenario minute. The date visibly remains historical.
    return start + timedelta(minutes=(time.monotonic() - STARTED) % (24 * 60))


def _row(at: datetime) -> dict[str, str]:
    rows = inputs()[0]
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    index = int((at - start).total_seconds() // 3600) % len(rows)
    return rows[index]


def _rates(at: datetime) -> dict[str, float]:
    row, scale = _row(at), inputs()[1]
    return {
        "throughput_teu_per_hour": float(row["throughput_teu"]) * scale,
        "power_kw": float(row["base_load_kw"]) * scale,
        "price_cny_per_kwh": float(row["price_per_kwh"]),
        "carbon_kg_per_kwh": float(row["carbon_kg_per_kwh"]),
        "ambient_c": float(row["ambient_c"]),
        "wind_speed_mps": float(row["wind_speed_mps"]),
        "wave_height_m": float(row["wave_height_m"]),
        "tide_m": float(row["tide_m"]),
        "current_speed_mps": float(row["current_speed_mps"]),
        "equipment_availability_ratio": float(row["equipment_availability_ratio"]),
        "yard_occupancy_ratio": float(row["yard_occupancy_ratio"]),
    }


def _groups(at: datetime) -> list[dict[str, Any]]:
    row = _row(at)
    # Reuse the existing engineering allocation, preserving aggregate power.
    # These are groups, never individual crane meters or verified installations.
    weights = CalibratedReplayTelemetry._asset_raw_weights({**row, "source_timestamp": row["timestamp"]})
    total = sum(weights.values())
    return [{"id": item["id"], "label": GROUP_LABELS[item["id"]], "category": item["category"],
             "kW": _rates(at)["power_kw"] * weights[item["id"]] / total,
             "measured": False, "include_in_aggregate": True,
             "source": "engineering_mass_conserving_allocation_not_metered"} for item in ASSET_SPECS]


def assets() -> list[dict[str, Any]]:
    if os.getenv("PORT_DT_PRESENTATION_SNAPSHOT", "").strip():
        data = snapshot()
        if not data.get("asset_groups"):
            raise ValueError("external presentation requires asset_groups for group load display")
        return data["asset_groups"]
    return _groups(clock())


def asset_series(asset_id: str, at: str | None = None) -> list[dict[str, Any]]:
    if asset_id not in GROUP_LABELS:
        raise ValueError("unknown presentation asset group")
    if os.getenv("PORT_DT_PRESENTATION_SNAPSHOT", "").strip():
        data = snapshot()
        rows = data.get("asset_series", {}).get(asset_id)
        if not isinstance(rows, list) or not rows:
            raise ValueError("external presentation has no asset_series for this group")
        for row in rows:
            _timestamp(row["source_ts"])
            if _number(row["kW"], "asset power") < 0:
                raise ValueError("asset power cannot be negative")
        return rows
    end = clock(at).replace(minute=0, second=0, microsecond=0)
    rows = []
    for offset in range(-23, 1):
        instant = end + timedelta(hours=offset)
        group = next(g for g in _groups(instant) if g["id"] == asset_id)
        rows.append({"ts": instant.isoformat(), "source_ts": _row(instant)["timestamp"], **group})
    return rows


def _schedule(at: datetime) -> list[dict[str, Any]]:
    p = profile(); facts = p["facts"]; params = p["scenario_parameters"]
    target = params["annual_teu_calibration_target"]["value"]
    arrivals_per_year = inputs()[2]
    mean_cargo = target / arrivals_per_year
    headway_hours = 8760 / arrivals_per_year
    qc_per_call = params["qcs_per_call"]["value"]
    net_moves = facts["qc_record_moves_per_hour_lower_bound"]["value"] * params["net_qc_rate_fraction_of_public_record"]["value"]
    service_rate = qc_per_call * net_moves * params["teu_per_natural_box"]["value"]
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    call_index = math.floor((at - base).total_seconds() / 3600 / headway_hours)
    berth_free = [datetime.min.replace(tzinfo=timezone.utc)] * int(facts["berths"]["value"])
    rows = []
    # A seven-call warm-up spans every berth; no randomized queue or identity.
    for index in range(call_index - 14, call_index + 13):
        eta = base + timedelta(hours=index * headway_hours)
        reference = _row(eta)
        mean_hourly = target / 8760
        shape = float(reference["throughput_teu"]) * inputs()[1] / mean_hourly
        cargo = round(mean_cargo * shape)
        berth = index % len(berth_free)
        etb = max(eta, berth_free[berth])
        complete = etb + timedelta(hours=cargo / service_rate)
        berth_free[berth] = complete
        progress = max(0.0, min(1.0, (at - etb).total_seconds() / max(1.0, (complete - etb).total_seconds())))
        phase = "completed" if at >= complete else "working" if at >= etb else "waiting" if at >= eta else "planned"
        rows.append({
            "call_id": f"DEMO-YS4-{index:05d}", "label": f"推演航次 {index:05d}",
            "berth_id": f"YS4-B{berth + 1:02d}", "berth_index": berth,
            "imo": None, "mmsi": None, "identity_kind": "synthetic_scenario_id",
            "eta": eta.isoformat(), "planned_berth_time": etb.isoformat(), "planned_completion": complete.isoformat(),
            "phase": phase, "cargo_teu": cargo, "completed_teu": round(cargo * progress),
            "progress": round(progress, 6), "assigned_qcs": qc_per_call,
            "waiting_minutes": round(max(0.0, (etb - eta).total_seconds() / 60), 2),
            "service_hours": round(cargo / service_rate, 3), "evidence_class": "resource_constrained_engineering_schedule",
        })
    return rows


def _validate_snapshot(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema") != "port-dt-presentation-snapshot.v1":
        raise ValueError("presentation snapshot schema mismatch")
    _timestamp(data["source_timestamp"])
    for name in ("profile", "operating", "environment", "series", "calls", "provenance"):
        expected = list if name in ("series", "calls") else dict
        if not isinstance(data.get(name), expected):
            raise ValueError(f"{name} has invalid shape")
    if not data["provenance"].get("source_ref"):
        raise ValueError("source_ref required; no silent calibrated fallback for a broken adapter")
    p = data["profile"]
    if not p.get("profile_id") or not isinstance(p.get("facts"), dict) or not isinstance(p.get("sources"), list) or not isinstance(p.get("scenario_parameters"), dict):
        raise ValueError("profile requires profile_id, facts, sources and scenario_parameters")
    ZoneInfo(p["timezone"])
    sources = {s["id"] for s in p["sources"]}
    for source in p["sources"]:
        if not str(source.get("url", "")).startswith(("https://", "http://")):
            raise ValueError("source URLs must be HTTP(S)")
    for fact in p["facts"].values():
        _number(fact["value"], "profile fact")
        if fact.get("source_id") not in sources:
            raise ValueError("profile fact has no matching source reference")
    for key in ("throughput_teu_per_hour", "power_kw"):
        if _number(data["operating"].get(key), key) < 0:
            raise ValueError(f"{key} cannot be negative")
    for stem, limit in (("berths", 64), ("qc", 256), ("rmg", 512), ("agv", 512)):
        total = _number(data["operating"].get(f"{stem}_total"), f"{stem}_total")
        busy = _number(data["operating"].get(f"{stem}_busy"), f"{stem}_busy")
        if total != int(total) or busy != int(busy) or not 0 <= busy <= total <= limit:
            raise ValueError(f"{stem} resource counts exceed the presentation envelope")
    if len(data["series"]) > 8760 or len(data["calls"]) > 1000:
        raise ValueError("presentation snapshot exceeds the bounded UI data envelope")
    for point in data["series"]:
        _timestamp(point["timestamp"])
        for key in ("power_kw", "throughput_teu_per_hour"):
            if _number(point.get(key), key) < 0:
                raise ValueError(f"{key} cannot be negative")
    for call in data["calls"]:
        _timestamp(call["eta"])
        start = _timestamp(call["planned_berth_time"])
        if _timestamp(call["planned_completion"]) < start:
            raise ValueError("call completion precedes berth start")
        if not 0 <= _number(call["progress"], "call progress") <= 1:
            raise ValueError("call progress must be in [0, 1]")
        berth = _number(call["berth_index"], "berth_index")
        if berth != int(berth) or not 0 <= berth < data["operating"]["berths_total"]:
            raise ValueError("call berth is outside the resource envelope")
        if call.get("phase") not in ("working", "waiting", "planned", "completed"):
            raise ValueError("unsupported call phase")
        if not isinstance(call.get("call_id"), str) or not call["call_id"] or _number(call["cargo_teu"], "cargo_teu") < 0:
            raise ValueError("call identity or cargo invalid")
    groups = data.get("asset_groups", [])
    if not isinstance(groups, list) or len(groups) > 64:
        raise ValueError("asset_groups exceeds the UI data envelope")
    for group in groups:
        if group.get("id") not in GROUP_LABELS or _number(group["kW"], "group power") < 0:
            raise ValueError("invalid mapped asset group or power")
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("duplicate asset group IDs")
    # A presentation file cannot grant production authority or claim verified live access.
    return {**data, "mode": "external_presentation_snapshot", "simulation_mode": data.get("simulation_mode", True),
            "production_authority": False, "live_data_verified": False,
            "boundary": "外部展示数据快照；来源核验与生产系统验收仍由既有独立网关负责。"}


def snapshot(at: str | None = None) -> dict[str, Any]:
    override = os.getenv("PORT_DT_PRESENTATION_SNAPSHOT", "").strip()
    if override:
        path = Path(override).expanduser()
        # Deliberately propagate config/validation errors; never silently substitute demo data.
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("presentation JSON exceeds 16 MiB")
        return _validate_snapshot(json.loads(path.read_text(encoding="utf-8")))
    at_time = clock(at); p = profile(); facts = p["facts"]; rate = _rates(at_time)
    calls = _schedule(at_time); working = [c for c in calls if c["phase"] == "working"]
    active_qcs = sum(c["assigned_qcs"] for c in working)
    qc_total = int(facts["qc_count"]["value"])
    if active_qcs > qc_total:
        raise ValueError("generated schedule exceeds public QC resource envelope")
    use = min(1.0, active_qcs / qc_total)
    active_agvs = math.ceil(facts["agv_count"]["value"] * use)
    active_rmgs = math.ceil(facts["rmg_count"]["value"] * use)
    series = []
    start = at_time.astimezone(ZoneInfo(p["timezone"])).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    for hour in range(24):
        t = start + timedelta(hours=hour); values = _rates(t)
        series.append({"timestamp": t.isoformat(), "source_timestamp": _row(t)["timestamp"], **{k: round(v, 6) for k, v in values.items()}})
    daily_teu = sum(v["throughput_teu_per_hour"] for v in series)
    daily_kwh = sum(v["power_kw"] for v in series)
    daily_co2 = sum(v["power_kw"] * v["carbon_kg_per_kwh"] for v in series)
    operating = {
        **{k: round(v, 6) for k, v in rate.items()},
        "berths_total": facts["berths"]["value"], "berths_busy": len(working),
        "berth_occupancy_pct": round(len(working) / facts["berths"]["value"] * 100, 3),
        "qc_total": qc_total, "qc_busy": active_qcs,
        "rmg_total": facts["rmg_count"]["value"], "rmg_busy": active_rmgs,
        "agv_total": facts["agv_count"]["value"], "agv_busy": active_agvs,
        "daily_throughput_teu": round(daily_teu, 3), "daily_energy_kwh": round(daily_kwh, 3),
        "daily_carbon_kg": round(daily_co2, 3), "energy_kwh_per_teu": round(daily_kwh / daily_teu, 6),
        "carbon_kg_per_teu": round(daily_co2 / daily_teu, 6),
        "daily_energy_cost_cny": round(sum(v["power_kw"] * v["price_cny_per_kwh"] for v in series), 3),
        "working_call_count": len(working),
        "waiting_call_count": sum(c["phase"] == "waiting" for c in calls),
        "planned_call_count": sum(c["phase"] == "planned" for c in calls),
    }
    return {
        "schema": "port-dt-presentation-snapshot.v1", "mode": "public_parameter_calibrated_demo",
        "generated_at": datetime.now(timezone.utc).isoformat(), "source_timestamp": at_time.isoformat(),
        "simulation_mode": True, "live_data_verified": False, "production_authority": False,
        "profile": p, "operating": operating,
        "environment": {k: rate[k] for k in ("ambient_c", "wind_speed_mps", "wave_height_m", "tide_m", "current_speed_mps")},
        "calls": calls, "series": series, "series_interval_seconds": 3600, "asset_groups": _groups(at_time),
        "provenance": {
            "source_ref": p["profile_id"], "profile_sha256": p["sha256"], "dataset_sha256": inputs()[3],
            "shape_source": str(DATASET.relative_to(ROOT)),
            "scale_to_terminal": inputs()[1], "annual_calibration_target_teu": p["scenario_parameters"]["annual_teu_calibration_target"]["value"],
            "fact_class": "cited_public_terminal_parameters", "operating_class": "engineering_simulation_not_terminal_telemetry",
            "weather_class": "public_historical_reanalysis", "power_class": "aggregate_engineering_energy_scaled_by_throughput_not_metered",
            "schedule_class": "derived_synthetic_calls_no_real_identity", "clock": "historical scenario; 1 real second = 1 scenario minute",
        },
        "boundary": p["boundary"],
        "formulas": {
            "terminal_scale": "8,000,000 / sum(2025 canonical engineering throughput)",
            "hourly_flow_and_power": "canonical Shanghai hourly series × terminal_scale; full-year throughput normalizes to 8,000,000 TEU",
            "call_cargo": "annual TEU / scaled annual arrivals × hourly flow shape",
            "call_duration": "cargo TEU / (4 cranes × 30 natural boxes/hour × 1.5 TEU/natural box)",
            "berth_assignment": "round-robin 7 berths; planned start=max(ETA, previous completion at this berth)",
            "agv_and_rmg_activity": "ceil(public resource count × assigned QCs / 29); workload mapping, not measured dispatch",
            "carbon": "sum(hourly engineering kW × hourly engineering grid factor), one-hour steps",
            "group_load": "existing calibrated replay operating-factor weights normalized to sum to the terminal engineering load; no submeter readings",
        },
    }


def replacement_contract() -> dict[str, Any]:
    return {
        "schema": "port-dt-presentation-adapter-contract.v1", "read_only": True,
        "environment_variable": "PORT_DT_PRESENTATION_SNAPSHOT=/absolute/path/to/snapshot.json",
        "endpoint": "/api/public-port-demo/snapshot", "snapshot_schema": "port-dt-presentation-snapshot.v1",
        "business_view": "/api/public-port-demo/workspace?at=timezone-aware-ISO",
        "required": ["schema", "source_timestamp", "profile", "operating", "environment", "series", "calls", "provenance.source_ref"],
        "mapping": [
            {"target": "profile.facts", "source": "港口/码头主数据、CMDB", "unit": "个/台/m/m²"},
            {"target": "calls", "source": "TOS 航次、泊位计划与作业事件", "unit": "UTC timezone-aware timestamps / TEU"},
            {"target": "operating.qc_busy/rmg_busy/agv_busy", "source": "ECS 设备状态", "unit": "台"},
            {"target": "operating.power_kw / series[].power_kw", "source": "EMS/分项电表", "unit": "kW"},
            {"target": "environment", "source": "授权气象/海事/潮位数据", "unit": "°C, m/s, m"},
            {"target": "operating.carbon_kg_per_teu", "source": "核算边界内计量 × 经核验排放因子 / TOS吞吐量", "unit": "kgCO2e/TEU"},
            {"target": "operating.yard_occupancy_ratio", "source": "TOS同窗箱位快照；可选，缺失保留为空", "unit": "ratio [0,1]"},
            {"target": "series[].price_cny_per_kwh / carbon_kg_per_kwh", "source": "电价合同 / 经核验因子版本；缺失不生成金额或碳排", "unit": "CNY/kWh / kgCO2e/kWh"},
        ],
        "report_contract": {"series_interval_seconds": 3600, "ordering": "连续、有时区的整点小时序列；乱序/缺点不生成窗口合计",
                            "external_scope": "外部源只按提供的小时序列求和；不回填内置2025年或分项假设。",
                            "null_semantics": "未提供/未核算用null，不能用0代表未知；Scope 1、审计、执行权不由此源授予。"},
        "algorithm_adapter": {"existing_list": "/api/rl/strategies", "existing_evaluate": "/api/rl/simulate",
                              "presentation_inputs": "算法输出若用于作业展示，由适配器映射为calls计划时窗、配桥与series小时序列；保留source_ref与算法版本。",
                              "boundary": "这里只展示建议；算法注册/评测沿用原有接口，模型权重、留出集与现场验收不随展示源替换。"},
        "validation": ["schema", "timezone", "finite units", "source reference", "no silent fallback on bad configuration"],
        "scope": "仅替换展示数据；训练/预测模型仍须按原有契约校准和验证，执行权限继续由南向网关独立管理。",
        "group_load": "分项展示可提供 asset_groups 与 asset_series[id]；缺少时显示不可用，不擅自拆分现场总电表。",
        "spatial_adapter": "实际布局与实体轨迹另用 PORTVIZ_MODE=real + PORTVIZ_CONFIG + PORTVIZ_FRAME_PATH；公开参数示意布局不会冒充测绘或 AIS。",
    }
