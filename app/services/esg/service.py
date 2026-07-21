from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# app/services/esg/service.py
# ESG / 合规模块 · 数据装载 + 来源校验 + 计算出入口
#
# 设计目标（只改本文件，不侵入其他模块）：
# 1) 保持 /api/esg/summary 的既有协定：导出 get_summary(di)。
# 2) 预置“合规报表”所需的数据计算入口（供下步在 app/server.py 暴露成 /api/compliance/*）：
#       - get_ports_catalog()
#       - get_compliance_timeseries(port_code, year, granularity="month")
#       - get_compliance_breakdown(port_code, year, month)
# 3) 默认拒绝演示/未验证数据。仅开发者手动设置 PORT_DT_ALLOW_DEMO_ESG=1 时可查看旧样例。
#
# 文件组织（均位于本模块 data/ 目录下）：
#   ├── summary_snapshot.json
#   ├── ports_catalog.json
#   ├── compliance_monthly_<code>_<year>.json
#   └── factors.json
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_PATH = DATA_DIR / "summary_snapshot.json"
CATALOG_PATH = DATA_DIR / "ports_catalog.json"
FACTOR_PATH = DATA_DIR / "factors.json"


# -------------------------
# 基础 I/O
# -------------------------
def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# 1) ESG 概览（/api/esg/summary 使用）
# -------------------------
def _load_snapshot() -> Dict[str, Any]:
    data = _read_json(SNAPSHOT_PATH)
    return data or {}


def _demo_esg_allowed() -> bool:
    return os.getenv("PORT_DT_ALLOW_DEMO_ESG", "").strip().lower() in {"1", "true", "yes", "on"}


def _unavailable_summary(reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "period_label": "待接入审计报告期",
        "port_name": "ESG 正式数据未接入",
        "comment": reason,
        "_source": "esg.unavailable",
    }


def get_summary(di: Any) -> Dict[str, Any]:
    """
    ESG / 合规驾驶舱汇总入口（被 app/server.py 的 /api/esg/summary 调用）。
    """
    snap = _load_snapshot()
    if not snap:
        return _unavailable_summary("未找到 ESG 快照；系统未生成替代指标。")

    source = str(snap.get("_source", "")).lower()
    looks_like_demo = "demo" in source or "演示" in str(snap.get("period_label", ""))
    if looks_like_demo and not _demo_esg_allowed():
        return _unavailable_summary(
            "检测到仓库内旧演示快照，已默认屏蔽。"
            "请接入经审计的能源、吞吐量、排放因子与事件数据。"
        )

    def _f(val: Any) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            return None

    def _i(val: Any) -> Optional[int]:
        try:
            return int(val)
        except Exception:
            return None

    res: Dict[str, Any] = {
        "available": True,
        "period_label": str(snap.get("period_label") or "未标注报告期"),
        "port_name": str(snap.get("port_name") or "未标注港口"),
        "score_esg": _f(snap.get("score_esg")),
        "score_e": _f(snap.get("score_e")),
        "score_s": _f(snap.get("score_s")),
        "score_g": _f(snap.get("score_g")),
        "co2_12m_ton": _f(snap.get("co2_12m_ton")),
        "co2_vs_target_pct": _f(snap.get("co2_vs_target_pct")),
        "renewable_share_pct": _f(snap.get("renewable_share_pct")),
        "auto_compliance_pct": _f(snap.get("auto_compliance_pct")),
        "incidents_12m": _i(snap.get("incidents_12m")),
        "audits_on_time_pct": _f(snap.get("audits_on_time_pct")),
        "comment": str(snap.get("comment") or "ESG 快照未提供备注。"),
        "_source": str(snap.get("_source") or "esg.snapshot_unlabelled"),
    }

    if "ci_kg_per_teu" in snap:
        try:
            res["ci_kg_per_teu"] = round(float(snap["ci_kg_per_teu"]), 2)
        except Exception:
            pass

    return res


# -------------------------
# 2) 合规数据模型（月度明细）
# -------------------------
@dataclass
class MonthlyItem:
    month: int
    teu: float
    grid_mwh: float
    shore_power_mwh: float
    onsite_renewables_mwh: float
    scope1_ton: float
    scope2_grid_ton: float
    scope2_shore_ton: float
    notes: str = ""

    @property
    def electric_mwh_total(self) -> float:
        return self.grid_mwh + self.shore_power_mwh + self.onsite_renewables_mwh

    @property
    def scope2_ton(self) -> float:
        return self.scope2_grid_ton + self.scope2_shore_ton

    @property
    def scope12_ton(self) -> float:
        return self.scope1_ton + self.scope2_ton

    @property
    def intensity_kg_per_teu(self) -> float:
        if self.teu <= 0:
            return 0.0
        return (self.scope12_ton * 1000.0) / self.teu

    def to_dict(self) -> Dict[str, Any]:
        return {
            "month": self.month,
            "teu": round(self.teu, 0),
            "grid_mwh": round(self.grid_mwh, 2),
            "shore_power_mwh": round(self.shore_power_mwh, 2),
            "onsite_renewables_mwh": round(self.onsite_renewables_mwh, 2),
            "electric_mwh_total": round(self.electric_mwh_total, 2),
            "scope1_ton": round(self.scope1_ton, 3),
            "scope2_grid_ton": round(self.scope2_grid_ton, 3),
            "scope2_shore_ton": round(self.scope2_shore_ton, 3),
            "scope2_ton": round(self.scope2_ton, 3),
            "scope12_ton": round(self.scope12_ton, 3),
            "intensity_kg_per_teu": round(self.intensity_kg_per_teu, 2),
            "notes": self.notes,
        }


# -------------------------
# 3) 合规入口（供 server.py 映射到 /api/compliance/*）
# -------------------------
def get_ports_catalog() -> Dict[str, Any]:
    data = _read_json(CATALOG_PATH)
    if data:
        return {"ports": data, "_source": "catalog.file"}

    return {"ports": [], "available": False, "_source": "catalog.unavailable"}


def get_compliance_timeseries(port_code: str, year: int, granularity: str = "month") -> Dict[str, Any]:
    if granularity != "month":
        raise ValueError("仅支持 granularity='month'")

    path = DATA_DIR / f"compliance_monthly_{port_code}_{year}.json"
    data = _read_json(path)
    metadata = _read_json(path.with_suffix(".meta.json")) or {}
    verified = bool(metadata.get("source_url") and metadata.get("provenance_type") in {"public", "port_export", "audited"})
    if not data or (not verified and not _demo_esg_allowed()):
        return {
            "port_code": port_code,
            "year": year,
            "granularity": granularity,
            "items": [],
            "totals": {},
            "available": False,
            "reason": "数据不存在或缺少可审计的 .meta.json 来源说明。",
            "_source": "compliance.unavailable",
        }

    totals = {
        "teu": 0.0,
        "electric_mwh_total": 0.0,
        "scope1_ton": 0.0,
        "scope2_ton": 0.0,
        "scope12_ton": 0.0,
    }
    for it in data:
        totals["teu"] += float(it.get("teu", 0))
        totals["electric_mwh_total"] += float(it.get("electric_mwh_total", 0))
        totals["scope1_ton"] += float(it.get("scope1_ton", 0))
        totals["scope2_ton"] += float(it.get("scope2_ton", 0))
        totals["scope12_ton"] += float(it.get("scope12_ton", 0))

    return {
        "port_code": port_code,
        "year": year,
        "granularity": granularity,
        "items": data,
        "totals": {k: round(v, 3) for k, v in totals.items()},
        "available": True,
        "provenance": metadata,
        "_source": "compliance.file_verified" if verified else "compliance.demo_opt_in",
    }


def get_compliance_breakdown(port_code: str, year: int, month: int) -> Dict[str, Any]:
    ts = get_compliance_timeseries(port_code, year, "month")
    rows = ts["items"]
    row = next((r for r in rows if int(r["month"]) == int(month)), None)
    if not row:
        raise ValueError(f"未找到 {port_code} {year}-{month:02d} 的数据")

    return {
        "port_code": port_code,
        "year": year,
        "month": month,
        "scope1_ton": row["scope1_ton"],
        "scope2_grid_ton": row["scope2_grid_ton"],
        "scope2_shore_ton": row["scope2_shore_ton"],
        "scope2_ton": row["scope2_ton"],
        "electric_mwh": {
            "grid": row["grid_mwh"],
            "shore_power": row["shore_power_mwh"],
            "onsite_renewables": row["onsite_renewables_mwh"],
            "total": row["electric_mwh_total"],
        },
        "intensity_kg_per_teu": row["intensity_kg_per_teu"],
        "_source": ts["_source"],
    }


# -------------------------
# 4) 演示数据生成（国际大港口）
# -------------------------
def ensure_demo_files(year: int = 2024) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    default_factors = {
        "CN/East": 0.62,
        "SG": 0.45,
        "NL": 0.37,
        "US/CA": 0.20,
        "DEFAULT": 0.50,
    }
    if not FACTOR_PATH.exists():
        _write_json(FACTOR_PATH, default_factors)

    catalog = get_ports_catalog()["ports"]

    for p in catalog:
        code = p["code"]
        region = p.get("region", "DEFAULT")
        factors = _read_json(FACTOR_PATH) or default_factors
        ef = factors.get(region, default_factors["DEFAULT"])
        _gen_monthly_if_absent(code, region, ef, year)


def _gen_monthly_if_absent(code: str, region: str, ef_ton_per_mwh: float, year: int) -> None:
    path = DATA_DIR / f"compliance_monthly_{code}_{year}.json"
    if path.exists():
        return

    base_teu = {
        "CNSHA": 3800000,
        "SGSIN": 3100000,
        "NLRTM": 1200000,
        "USLAXLGB": 1500000,
    }.get(code, 1000000)

    items: List[Dict[str, Any]] = []
    for m in range(1, 13):
        season_factor = 1.0 + (0.05 if m in (3, 4, 9, 10) else 0.0) + (-0.04 if m in (2, 7) else 0.0)
        teu = base_teu * season_factor

        grid_mwh = 0.95 * 0.18 * teu / 10_000
        shore_power_mwh = 0.05 * 0.18 * teu / 10_000
        shore_power_mwh *= (1.0 + 0.02 * (m - 1))
        onsite_ren_mwh = 0.02 * 0.18 * teu / 10_000

        scope1_ton = 0.0009 * teu / 10.0
        scope2_grid_ton = grid_mwh * ef_ton_per_mwh
        scope2_shore_ton = shore_power_mwh * ef_ton_per_mwh

        item = MonthlyItem(
            month=m,
            teu=teu,
            grid_mwh=grid_mwh,
            shore_power_mwh=shore_power_mwh,
            onsite_renewables_mwh=onsite_ren_mwh,
            scope1_ton=scope1_ton,
            scope2_grid_ton=scope2_grid_ton,
            scope2_shore_ton=scope2_shore_ton,
            notes=f"{region}·EF={ef_ton_per_mwh:.2f} tCO₂/MWh",
        )
        items.append(item.to_dict())

    _write_json(path, items)


# -------------------------
# 调试入口
# -------------------------
if __name__ == "__main__":
    ensure_demo_files()
    print(json.dumps(get_summary(di=None), ensure_ascii=False, indent=2))
    cat = get_ports_catalog()
    print("catalog:", cat)
    ex = get_compliance_timeseries(cat["ports"][0]["code"], 2024)
    print("timeseries example:", ex["items"][0])
