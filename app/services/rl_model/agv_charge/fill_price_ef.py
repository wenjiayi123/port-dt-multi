# app/services/rl_model/agv_charge/fill_price_ef.py
# -*- coding: utf-8 -*-
"""
把电价/碳因子文件重写成“标准 CSV”（对齐孪生 5 分钟时基）
----------------------------------------------------------------
作用：
- 修复 market_price.csv / grid_ef.csv 与训练时间轴不对齐、整列 NaN、重复时间戳等问题
- 给模块 A 提供稳定的 price / ef 基础数据，保证“错峰 / 低碳”逻辑有真实口径
- 额外输出一个补数审计文件，方便首页/API后续解释“当前采用的是哪一套价格/碳因子口径”

输出：
- data/market_price.csv : timestamp,price_yuan_per_kwh
- data/grid_ef.csv      : timestamp,ef_kg_per_kwh
- artifacts/price_ef_fill_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from .adapter import AGVChargeAdapter


def _tou_cn_v1_price(hour: int) -> float:
    """中国常见分时电价示意（单位：元/kWh）。"""
    if 0 <= hour < 7:
        return 0.60  # 谷
    if 17 <= hour < 21:
        return 1.20  # 峰
    return 0.85      # 平


def _tou_cn_v1_ef(hour: int) -> float:
    """电网边际排放因子示意（单位：kgCO2e/kWh）。"""
    if 0 <= hour < 7:
        return 0.45
    if 17 <= hour < 21:
        return 0.70
    return 0.55


def _weekend_bias_price(hour: int, weekday: int) -> float:
    """周末整体较低、工作日晚高峰更明显的示意价。"""
    base = _tou_cn_v1_price(hour)
    if weekday >= 5:
        return round(base * 0.92, 6)
    return round(base, 6)


def _weekend_bias_ef(hour: int, weekday: int) -> float:
    """周末整体负荷偏低、峰谷差略收敛的示意排放因子。"""
    base = _tou_cn_v1_ef(hour)
    if weekday >= 5:
        return round(max(0.38, base * 0.95), 6)
    return round(base, 6)


def _flat_price(_: int, v: float) -> float:
    return float(v)


def _flat_ef(_: int, v: float) -> float:
    return float(v)


def _safe_index(adapter: AGVChargeAdapter) -> List[datetime]:
    if hasattr(adapter, "_index") and getattr(adapter, "_index") is not None:
        return list(getattr(adapter, "_index"))
    if hasattr(adapter, "_time_index") and getattr(adapter, "_time_index") is not None:
        return list(getattr(adapter, "_time_index"))
    raise RuntimeError("adapter time index missing; check adapter.load_all() and source data")


def build_series(
    tindex: Sequence[datetime],
    pattern: str,
    price_flat: float,
    ef_flat: float,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    price_rows: List[Tuple[str, float]] = []
    ef_rows: List[Tuple[str, float]] = []
    for t in tindex:
        h = t.hour
        wd = t.weekday()
        if pattern == "cn_tou_v1":
            p = _tou_cn_v1_price(h)
            e = _tou_cn_v1_ef(h)
        elif pattern == "cn_weekend_bias_v1":
            p = _weekend_bias_price(h, wd)
            e = _weekend_bias_ef(h, wd)
        elif pattern == "flat":
            p = _flat_price(h, price_flat)
            e = _flat_ef(h, ef_flat)
        else:
            raise ValueError(f"unknown pattern={pattern}")
        ts = t.isoformat()
        price_rows.append((ts, round(float(p), 6)))
        ef_rows.append((ts, round(float(e), 6)))
    return price_rows, ef_rows


def write_csv(path: Path, header: Tuple[str, str], rows: Iterable[Tuple[str, float]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for ts, value in rows:
            writer.writerow([ts, f"{float(value):.6f}"])
            count += 1
    return count


def summarize_rows(rows: Sequence[Tuple[str, float]]) -> Dict[str, object]:
    values = [float(v) for _, v in rows]
    preview = [{"timestamp": ts, "value": float(v)} for ts, v in rows[:6]]
    return {
        "rows": len(rows),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": round(mean(values), 6) if values else None,
        "preview": preview,
    }


def build_audit(
    tindex: Sequence[datetime],
    pattern: str,
    price_rows: Sequence[Tuple[str, float]],
    ef_rows: Sequence[Tuple[str, float]],
    applied: bool,
    base_dir: Path,
) -> Dict[str, object]:
    return {
        "module": "agv_charge",
        "task": "fill_price_ef",
        "applied": bool(applied),
        "pattern": pattern,
        "timebase_minutes": 5,
        "time_range": {
            "start": tindex[0].isoformat() if tindex else None,
            "end": tindex[-1].isoformat() if tindex else None,
        },
        "base_dir": str(base_dir),
        "outputs": {
            "market_price": summarize_rows(price_rows),
            "grid_ef": summarize_rows(ef_rows),
        },
        "notes": [
            "price/ef generated to match adapter time index",
            "intended for module A training fallback and explainability",
            "replace with real tariff/carbon feeds later when available",
        ],
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fill/repair market_price.csv & grid_ef.csv to standard CSV aligned by 5-min timebase"
    )
    ap.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="directory that contains config.yaml and data/",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="cn_tou_v1",
        choices=["cn_tou_v1", "cn_weekend_bias_v1", "flat"],
        help="price/ef daily pattern",
    )
    ap.add_argument("--price-flat", type=float, default=0.80, help="flat price when pattern=flat (¥/kWh)")
    ap.add_argument("--ef-flat", type=float, default=0.55, help="flat grid emission factor when pattern=flat (kgCO2e/kWh)")
    ap.add_argument("--apply", action="store_true", help="write files to data/ (otherwise only print preview)")
    ap.add_argument("--audit-only", action="store_true", help="only refresh audit summary based on generated rows")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    adapter = AGVChargeAdapter(base_dir=base_dir)
    adapter.load_all()
    tindex = _safe_index(adapter)
    if not tindex:
        raise RuntimeError("empty time index; check your data package")

    price_rows, ef_rows = build_series(tindex, args.pattern, args.price_flat, args.ef_flat)

    price_path = base_dir / "data" / "market_price.csv"
    ef_path = base_dir / "data" / "grid_ef.csv"
    audit_path = base_dir / "artifacts" / "price_ef_fill_summary.json"

    print(f"[Preview] rows={len(tindex)}, range: {tindex[0].isoformat()} ~ {tindex[-1].isoformat()}")
    print(f"  market_price.csv -> {price_path}")
    print(f"  grid_ef.csv      -> {ef_path}")
    print(f"  pattern={args.pattern} (flat price={args.price_flat}, flat ef={args.ef_flat})")
    print(f"  audit            -> {audit_path}")

    wrote_price = 0
    wrote_ef = 0
    if args.apply and not args.audit_only:
        wrote_price = write_csv(price_path, ("timestamp", "price_yuan_per_kwh"), price_rows)
        wrote_ef = write_csv(ef_path, ("timestamp", "ef_kg_per_kwh"), ef_rows)
        print(f"[OK] Files written. market_price={wrote_price} rows, grid_ef={wrote_ef} rows")
    elif args.audit_only:
        print("[INFO] audit-only mode; CSV files will not be overwritten")
    else:
        print("(dry-run) Use --apply to write files.")

    audit = build_audit(
        tindex=tindex,
        pattern=args.pattern,
        price_rows=price_rows,
        ef_rows=ef_rows,
        applied=bool(args.apply and not args.audit_only),
        base_dir=base_dir,
    )
    if wrote_price:
        audit["written_rows"] = {"market_price": wrote_price, "grid_ef": wrote_ef}
    write_json(audit_path, audit)
    print("[OK] Audit summary written.")


if __name__ == "__main__":
    main()
