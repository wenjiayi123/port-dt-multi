from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_PATH = DATA_DIR / "badge_snapshot.json"
ROOT = Path(__file__).resolve().parents[3]
V3_ADVANTAGE_PATH = ROOT / "evidence/v3/shanghai_public_advantage_v3.json"


class SnapshotError(RuntimeError):
    """Raised when the local trust badge snapshot is missing or invalid."""


def _load_snapshot() -> Dict[str, Any]:
    """Load the raw snapshot JSON from disk.

    The snapshot is intended to be generated offline from
    evaluation pipelines (OPE、守护栏校验、实验等)，这里仅负责读取。
    """
    if not SNAPSHOT_PATH.exists():
        raise SnapshotError(f"snapshot not found: {SNAPSHOT_PATH}")
    try:
        with SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"invalid snapshot json: {exc}") from exc


def _extract_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize snapshot structure and compute a compact summary.

    返回值会直接暴露在 /api/ai/trust_badge 给前端使用，
    请保持字段名向后兼容。
    """
    # ---- 基本字段（向后兼容：旧文件可能只有这些键） ----
    grade = raw.get("grade") or "N/A"
    grade_label = raw.get("grade_label") or ""
    ope_pass = bool(raw.get("ope_pass", False))

    # ---- 守护栏聚合 ----
    guardrail = raw.get("guardrail") or {}
    try:
        pending = int(guardrail.get("pending", 0) or 0)
    except (TypeError, ValueError):
        pending = 0
    try:
        total = int(guardrail.get("total", 0) or 0)
    except (TypeError, ValueError):
        total = 0

    # ---- 实验聚合 ----
    experiments = raw.get("experiments") or {}
    try:
        running = int(experiments.get("running", 0) or 0)
    except (TypeError, ValueError):
        running = 0
    try:
        completed = int(experiments.get("completed", 0) or 0)
    except (TypeError, ValueError):
        completed = 0

    # ---- 因果效应（比例）----
    causal_effect = raw.get("causal_effect", 0.0)
    try:
        causal_effect = float(causal_effect)
    except (TypeError, ValueError):
        causal_effect = 0.0

    # ---- 整体状态灯：给前端简单三色灯使用 ----
    if grade in ("A+", "A") and ope_pass and pending == 0:
        overall_status = "ok"
    elif pending <= 2:
        overall_status = "warn"
    else:
        overall_status = "err"

    # ---- 端到端上下文信息：港口 + 场景（前端暂未使用，预留扩展） ----
    port = raw.get("port") or {}
    meta = raw.get("meta") or {}
    scenes = raw.get("scenes") or []

    return {
        "grade": grade,
        "grade_label": grade_label,
        "overall_status": overall_status,           # ok | warn | err
        "ope_pass": ope_pass,
        "guardrail": {"total": total, "pending": pending},
        "experiments": {"running": running, "completed": completed},
        "causal_effect": causal_effect,
        "port": port,
        "meta": meta,
        "scenes": scenes,
        "_source": "ai_trust.snapshot",
    }


def _v3_evidence_badge() -> Dict[str, Any]:
    sidecar = V3_ADVANTAGE_PATH.with_suffix(".sha256")
    if not V3_ADVANTAGE_PATH.is_file() or not sidecar.is_file():
        return {}
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if hashlib.sha256(V3_ADVANTAGE_PATH.read_bytes()).hexdigest() != expected:
        return {}
    payload = json.loads(V3_ADVANTAGE_PATH.read_text(encoding="utf-8"))
    selected = payload.get("selected") or {}
    admission = selected.get("safety_admission") or {}
    strict = bool(selected.get("strict_advantage"))
    return {
        "available": True,
        "grade": "B+" if strict else "D",
        "grade_label": "公开数据离线准入 / 现场待接",
        "overall_status": "warn",
        "ope_pass": strict,
        "evaluation_kind": "chronological_blind_test_3_seeds",
        "guardrail": {
            "total": 3,
            "pending": 0 if admission.get("passed") else 1,
            "violation_rate_max": admission.get("guardrail_violation_rate_max_observed"),
        },
        "experiments": {"running": 0, "completed": 30},
        "causal_effect": None,
        "port": {"name": "上海港公开目标域（非现场遥测）"},
        "meta": {
            "is_sample": False,
            "note": "验证集选模，3 随机种子与盲测时间窗；因果/A-B 与现场闭环仍待接入港口。",
            "production_authority": False,
        },
        "scenes": [],
        "_source": "ai_trust.v3_hash_verified_offline_evidence",
    }


def get_badge(di: Any | None = None) -> Dict[str, Any]:
    """主入口：供 FastAPI 路由调用。

    参数 di 预留给依赖注入容器，目前未使用，保留以与其它模块接口一致。
    """
    raw = _load_snapshot()
    meta = raw.get("meta") or {}
    provenance = raw.get("_provenance") or {}
    verified = (
        isinstance(provenance, dict)
        and provenance.get("provenance_type") in {"port_export", "audited", "verified_test"}
        and bool(provenance.get("source_url"))
    )
    if meta.get("is_sample") or not verified:
        return _v3_evidence_badge() or {
            "available": False,
            "grade": "N/A",
            "grade_label": "未评定",
            "overall_status": "unavailable",
            "ope_pass": None,
            "guardrail": {"total": None, "pending": None},
            "experiments": {"running": None, "completed": None},
            "causal_effect": None,
            "port": {},
            "meta": {"is_sample": False, "note": "仓库内样例可信度快照已屏蔽；请接入带来源的评测证据。"},
            "scenes": [],
            "_source": "ai_trust.unavailable",
        }
    return _extract_summary(raw)
