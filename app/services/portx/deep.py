# coding: utf-8
from __future__ import annotations
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

router = APIRouter(tags=["port-deep"])

# ========= 工具 =========
def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _seed(port: str) -> random.Random:
    # 固定口岸得到稳定的“真口径假数”
    return random.Random(abs(hash(port)) & 0xFFFFFFFF)

def _days(n=7) -> List[str]:
    today = datetime.utcnow().date()
    return [(today - timedelta(days=(n - 1 - i))).strftime("%m-%d") for i in range(n)]


# ========= 14. 端到端 KPI 盘 =========
@router.get("/kpi/dashboard")
def kpi_dashboard(
    port: str = Query("CNYTN"),
    period: str = Query("last_7d")
) -> JSONResponse:
    """
    船时效率、岸桥作业率、单箱能耗、堆场周转、卡车在港时长、泊位利用率
    与能耗/碳/成本同屏；并返回“上线后 KPI 改善贡献拆分”（waterfall）。
    """
    rnd = _seed(port)
    days = _days(7)

    quay_crane_rate = [round(30 + rnd.random() * 8, 1) for _ in days]      # moves/h
    berth_util       = [round(0.60 + rnd.random() * 0.2, 2) for _ in days] # 0~1
    energy_kwh       = [int(150_000 + rnd.random() * 50_000) for _ in days]
    carbon_kg        = [int(e * (0.58 + rnd.random() * 0.12)) for e in energy_kwh]
    cost_yuan        = [int(e * (0.78 + rnd.random() * 0.25)) for e in energy_kwh]

    vessel_turn_hours      = round(26.0 + rnd.random()*6.0, 1)
    qc_rate_now            = round(sum(quay_crane_rate)/len(quay_crane_rate), 1)
    energy_per_teu_kwh     = round(6.5 + rnd.random()*1.2, 2)
    yard_turnover_days     = round(4.0 + rnd.random()*1.5, 2)
    truck_time_in_port_min = round(38 + rnd.random()*12)
    berth_util_now         = round(sum(berth_util)/len(berth_util), 2)

    # “上线前后”贡献拆分（例：单箱能耗，越低越好 → 负值=改善）
    baseline = round(energy_per_teu_kwh * 1.12, 2)  # 上线前高 12%
    items = [
        {"name": "泊位/桥机指派",    "delta": round(-0.18 - rnd.random()*0.15, 2)},
        {"name": "堆场配载/堆策略",  "delta": round(-0.12 - rnd.random()*0.12, 2)},
        {"name": "AGV 充换电编排",  "delta": round(-0.22 - rnd.random()*0.15, 2)},
        {"name": "岸电+BESS+电价",  "delta": round(-0.28 - rnd.random()*0.20, 2)},
    ]
    current = round(baseline + sum(d["delta"] for d in items), 2)

    payload: Dict[str, Any] = {
        "policy_version": "RL-v4.2.1",
        "period": period,
        "kpis": {
            "vessel_turn_hours": vessel_turn_hours,
            "quay_crane_rate": qc_rate_now,
            "energy_per_teu_kwh": energy_per_teu_kwh,
            "yard_turnover_days": yard_turnover_days,
            "truck_time_in_port_min": truck_time_in_port_min,
            "berth_utilization": berth_util_now
        },
        "ecc": {
            "energy_kwh_total": sum(energy_kwh),
            "carbon_kg_total": sum(carbon_kg),
            "cost_yuan_total": sum(cost_yuan)
        },
        "timeseries": {
            "days": days,
            "energy_kwh": energy_kwh,
            "carbon_kg": carbon_kg,
            "cost_yuan": cost_yuan,
            "qc_rate": quay_crane_rate,
            "berth_util": berth_util
        },
        "contrib": {
            "metric_name": "单箱能耗 (kWh/TEU) ↓",
            "baseline": baseline,
            "items": items,
            "current": current
        },
        "ts": _now(), "port": port
    }
    return JSONResponse(payload)


# ========= 15. 更全面的策略对象（Pareto + 可达集 + 指派） =========
@router.get("/strategy/policyspace")
def strategy_policyspace(
    port: str = Query("CNYTN")
) -> JSONResponse:
    rnd = _seed(port)

    # 多目标样本：收益(¥, max)、能耗(kWh, min)、峰值(kW, min)、准点率(max)
    all_pts: List[Dict[str, Any]] = []
    for i in range(60):
        energy = 900_000 + rnd.random() * 400_000
        peak   = 10_000  + rnd.random() * 6_000
        ontime = 0.82 + rnd.random() * 0.15
        profit = (
            6_000_000
            + (1.15 - energy / 1_300_000) * 2_500_000
            + (ontime - 0.90) * 2_000_000
            - (peak - 10_000) * 120
        )
        all_pts.append({
            "name": f"S{i+1}",
            "energy_kwh": int(energy),
            "peak_kw": int(peak),
            "on_time_rate": round(max(0.0, min(1.0, ontime)), 3),
            "profit_yuan": int(profit)
        })

    # Pareto：按能耗升序取“收益最优”前沿
    all_pts.sort(key=lambda d: (d["energy_kwh"], -d["profit_yuan"]))
    frontier: List[Dict[str, Any]] = []
    best_profit = -10**18
    for p in all_pts:
        if p["profit_yuan"] > best_profit:
            best_profit = p["profit_yuan"]
            frontier.append(p)

    # 可达集热区（能耗×峰值二维网格的计数密度）
    xs = [p["energy_kwh"] for p in all_pts]
    ys = [p["peak_kw"] for p in all_pts]
    x_min, x_max = int(min(xs)*0.97), int(max(xs)*1.03)
    y_min, y_max = int(min(ys)*0.97), int(max(ys)*1.03)
    x_bins, y_bins = 20, 12
    x_edges = [x_min + i*(x_max - x_min)/x_bins for i in range(x_bins)]
    y_edges = [y_min + i*(y_max - y_min)/y_bins for i in range(y_bins)]
    grid = [[0 for _ in range(x_bins)] for _ in range(y_bins)]
    for p in all_pts:
        xi = min(x_bins-1, int((p["energy_kwh"] - x_min) / max(1, (x_max - x_min)) * x_bins))
        yi = min(y_bins-1, int((p["peak_kw"]    - y_min) / max(1, (y_max - y_min)) * y_bins))
        grid[yi][xi] += 1
    vmax = max(max(row) for row in grid) if grid else 1

    # 样例指派（修复 IndexError：使用安全循环+取模）
    vessels = ["MSC A", "CMA CGM B", "MAERSK C", "EMC D"]
    berths  = ["B01", "B02", "B03", "B04"]
    qcs     = ["QC-01", "QC-02", "QC-03", "QC-04", "QC-05"]

    assignments: List[Dict[str, str]] = []
    n = min(len(vessels), len(berths))
    for i in range(n):
        assignments.append({
            "resource": "泊位",
            "target": f"{berths[i]} ← {vessels[i]} (ETA+{i*2}h)",
            "note": "潮窗 & 桥距校核通过"
        })
        assignments.append({
            "resource": "桥机",
            "target": f"{qcs[i % len(qcs)]} → {vessels[i]} ({round(28 + rnd.random()*6, 1)} moves/h)",
            "note": "平衡边/高箱"
        })
        assignments.append({
            "resource": "堆场",
            "target": f"Y{i%3+1}-Block{chr(65+i)}（冷藏占比 {round(0.12 + rnd.random()*0.08, 2)}）",
            "note": "对齐重心与疏港流"
        })
        assignments.append({
            "resource": "AGV充换电",
            "target": f"可用 {12+i} 台；峰前 {3 + (i%2)} 批预充",
            "note": "与 BESS 协同削峰"
        })

    payload = {
        "pareto": {"all": all_pts, "frontier": frontier},
        "feasible": {"x": x_edges, "y": y_edges, "z": grid, "max": vmax},
        "assignments": assignments,
        "ts": _now(), "port": port
    }
    return JSONResponse(payload)
