"""Evidence-backed executive cockpit summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = BASE_DIR / "data" / "summary_snapshot.json"
ALLOWED_PROVENANCE = {"port_export", "audited", "verified_test", "public"}


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


def get_summary(di: Any) -> Dict[str, Any]:
    del di
    if not SNAPSHOT_PATH.is_file():
        return _unavailable("管理驾驶舱证据快照未配置")
    snapshot = _load_json(SNAPSHOT_PATH)
    if not snapshot:
        return _unavailable("管理驾驶舱证据快照无法解析")
    provenance = snapshot.get("_provenance")
    if not isinstance(provenance, dict):
        provenance = _load_json(SNAPSHOT_PATH.with_suffix(".meta.json"))
    provenance_type = str((provenance or {}).get("provenance_type") or "")
    source_url = str((provenance or {}).get("source_url") or "")
    if provenance_type not in ALLOWED_PROVENANCE or not source_url:
        return _unavailable("管理驾驶舱快照缺少允许的 provenance_type 与 source_url")
    return {
        **snapshot,
        "available": True,
        "_provenance": provenance,
        "_source": "exec_cockpit.verified_snapshot",
    }
