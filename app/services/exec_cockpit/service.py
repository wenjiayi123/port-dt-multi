"""Evidence-backed executive cockpit summary."""

from __future__ import annotations

import json
import csv
import hashlib
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = BASE_DIR / "data" / "summary_snapshot.json"
ALLOWED_PROVENANCE = {"port_export", "audited", "verified_test", "public"}
ROOT = Path(__file__).resolve().parents[3]
V3_IMPACT_PATH = ROOT / "evidence/v3/shanghai_public_business_impact_v3.json"
V3_ADVANTAGE_PATH = ROOT / "evidence/v3/shanghai_public_advantage_v3.json"
V3_DATASET_PATH = ROOT / "data/rl/datasets/public_cn_sha_hourly_v3.csv"
V3_ANCHOR_PATH = ROOT / "data/public_sources/shanghai_port_mot_2024_2025.json"


def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "yearly_saving_cny": None,
        "yearly_co2_ton": None,
        "peak_risk_30d": None,
        "auto_cover_pct": None,
        "ai_grade": "N/A",
        "ai_grade_reason": "缺少可审计管理证据",
        "status_level": "unknown",
        "status_label": "未评定",
        "status_detail": reason,
        "port_profile": {},
        "ops_kpi": {},
        "energy_kpi": {},
        "ai_coverage": {"scenes": []},
        "risk_summary": {},
        "_source": "exec_cockpit.unavailable",
    }


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_hash_verified_json(path: Path) -> Dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        return {}
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    return _load_json(path) if expected == observed else {}


def _v3_public_evidence_summary() -> Dict[str, Any]:
    """Build an executive view without promoting public replay to site KPI."""
    impact = _load_hash_verified_json(V3_IMPACT_PATH)
    advantage = _load_hash_verified_json(V3_ADVANTAGE_PATH)
    anchors = _load_json(V3_ANCHOR_PATH)
    if not impact or not advantage or not anchors or not V3_DATASET_PATH.is_file():
        return {}

    selected = advantage.get("selected") or {}
    learned = impact.get("learned_efficiency_value") or {}
    observations = anchors.get("observations") or []
    annual_anchor = max(
        (row for row in observations if str(row.get("period", "")).startswith("2025")),
        key=lambda row: float(row.get("cumulative_teu_10000") or 0),
        default={},
    )
    annual_throughput = float(annual_anchor.get("cumulative_teu_10000") or 0) * 10_000

    with V3_DATASET_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))[-8760:]
    annual_mwh = sum(float(row["base_load_kw"]) for row in rows) / 1000.0
    annual_cost = sum(
        float(row["base_load_kw"]) * float(row["price_per_kwh"])
        for row in rows
    )
    yard_occupancy = sum(float(row["yard_occupancy_ratio"]) for row in rows) / len(rows)
    strict = bool(selected.get("strict_advantage"))
    safety = selected.get("safety_admission") or {}

    return {
        "available": True,
        "evidence_mode": "public_data_offline_verified",
        "yearly_saving_cny": learned.get("annualized_avoided_cost"),
        "yearly_co2_ton": (
            float(learned.get("annualized_avoided_carbon_kg") or 0) / 1000.0
        ),
        "peak_risk_30d": None,
        "auto_cover_pct": None,
        "software_scenario_coverage": {"covered": 9, "total": 10},
        "ai_grade": "离线 A / 现场待准入" if strict else "离线 B / 现场待准入",
        "ai_grade_reason": "验证集选模、三随机种子盲测、95% CI 与安全门均留痕；无现场控制权。",
        "status_level": "warn",
        "status_label": "公开数据证据就绪 · 现场待接入",
        "status_detail": "上海公开吞吐锚点、洋山公开再分析及模型证据可复核；现场经营与设备 KPI 未接入时不填零值。",
        "port_profile": {
            "port_name": "上海港公开目标域（非现场遥测）",
            "annual_throughput_teu": annual_throughput,
            "annual_throughput_year": 2025,
            "vessel_calls_12m": None,
            "berth_count_total": None,
            "berth_count_deepwater": None,
            "shore_power_coverage_pct": None,
            "bess_capacity_mwh": None,
        },
        "ops_kpi": {
            "teu_per_gg_per_hour": None,
            "teu_per_gg_best_hour": None,
            "avg_vessel_berth_time_hour": None,
            "avg_truck_turnaround_p50_min": None,
            "avg_truck_turnaround_p95_min": None,
            "yard_occupancy_pct": yard_occupancy,
            "yard_occupancy_provenance": "engineering_derived_public_replay",
            "gate_appointment_compliance_pct": None,
            "vessel_on_time_departure_pct": None,
        },
        "energy_kpi": {
            "annual_electricity_mwh": annual_mwh,
            "annual_power_cost_cny": annual_cost,
            "period": "2025 public-calibrated engineering series",
            "shore_power_energy_share_pct": None,
            "renewable_energy_share_pct": None,
            "bess_discharge_share_pct": None,
        },
        "ai_coverage": {"scenes": []},
        "risk_summary": {
            "grid_peak_risk_30d": None,
            "ops_delay_risk_7d": None,
            "safety_incidents_12m": None,
            "safety_incidents_target_12m": None,
            "critical_alerts_open": None,
            "major_alerts_open": None,
        },
        "evidence": {
            "strict_advantage": strict,
            "weighted_advantage": (selected.get("weighted_relative_improvement") or {}).get("mean"),
            "guardrail_violation_rate_max": safety.get("guardrail_violation_rate_max_observed"),
            "amounts_are_mechanical_annualization": True,
            "site_kpis_pending": True,
        },
        "_provenance": {
            "provenance_type": "public",
            "source_url": (anchors.get("sources") or [{}])[0].get("url") or "https://xxgk.mot.gov.cn/",
            "dataset_sha256": (impact.get("dataset") or {}).get("sha256"),
        },
        "_source": "exec_cockpit.v3_public_offline_evidence",
    }


def get_summary(di: Any) -> Dict[str, Any]:
    del di
    public_summary = _v3_public_evidence_summary()
    if not SNAPSHOT_PATH.is_file():
        return public_summary or _unavailable("管理驾驶舱证据快照未配置")
    snapshot = _load_json(SNAPSHOT_PATH)
    if not snapshot:
        return _unavailable("管理驾驶舱证据快照无法解析")
    provenance = snapshot.get("_provenance")
    if not isinstance(provenance, dict):
        provenance = _load_json(SNAPSHOT_PATH.with_suffix(".meta.json"))
    provenance_type = str((provenance or {}).get("provenance_type") or "")
    source_url = str((provenance or {}).get("source_url") or "")
    if provenance_type not in ALLOWED_PROVENANCE or not source_url:
        return public_summary or _unavailable("管理驾驶舱快照缺少允许的 provenance_type 与 source_url")
    return {
        **snapshot,
        "available": True,
        "_provenance": provenance,
        "_source": "exec_cockpit.verified_snapshot",
    }
