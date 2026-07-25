# ============================================
# app/services/compliance.py
# --------------------------------------------
# 合规与报表服务（GHG 范畴 1/2）
#
# 提供：
#   - class EmissionFactors: 排放因子口径对象
#   - class ComplianceService: 月度 / 季度 / 通用报表
#
# 设计思路（演示可跑，口径清晰）：
#   1) 电力口径（Scope 2）：
#       - 以“今日口径”作为样本日（调用 EnergyService.build_today_summary）
#       - 取其中 electricity.by_asset[kWh_est] 作为“样本日每设备的电量”
#       - 月度/季度：按“样本日 × 天数”外推（简化做法，便于无历史表计也能跑通演示）
#       - 岸电（购电）与自发电（柴油机）按 selfgen_share（0~1）拆分
#       - Scope 2 = 购电电量 × (grid_g_per_kwh / 1000)
#       - 自发电部分计入 Scope 1：自发电电量 × selfgen_kg_per_kwh
#
#   2) 柴油口径（Scope 1）：
#       - 默认 diesel_model = "rule_of_thumb"：演示中保守置 0（无真实油表数据）
#         Site-specific AGV and tractor models may add liters × diesel_kg_per_liter.
#       - 若 diesel_model = "none"：明确禁用油耗估算（与默认效果相同）
#
#   3) 分摊（allocations）：
#       - by_asset：输出每设备的 kWh 与 kgCO2e（按全站比例分摊 Scope1/2）
#       - by_process：按设备前缀归类（qc/yc/agv/wh/cs/ps/yard/misc）
#       - by_group / by_berth：保留空数组占位（如前端需要可后续填充）
#
# 说明：
#   - 本文件为“轻口径可跑版”，真实项目应接入：历史表计/油表/工艺数据、时段/TOU、分电房/回路汇总等。
#   - 统一返回 assumptions 提示“当前为样本日外推”与自发电份额的假设，避免误用。
# ============================================

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# 排放因子口径对象
# ---------------------------
@dataclass
class EmissionFactors:
    """
    排放因子口径：
      - grid_g_per_kwh：电网/岸电因子（gCO2e/kWh）
      - diesel_kg_per_liter：柴油因子（kgCO2e/L）
      - selfgen_kg_per_kwh：自发电因子（kgCO2e/kWh）
      - selfgen_share：自发电占总电量的比例（0~1）
    """
    grid_g_per_kwh: float = 120.0
    diesel_kg_per_liter: float = 2.68
    selfgen_kg_per_kwh: float = 0.70
    selfgen_share: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "grid_g_per_kwh": float(self.grid_g_per_kwh),
            "diesel_kg_per_liter": float(self.diesel_kg_per_liter),
            "selfgen_kg_per_kwh": float(self.selfgen_kg_per_kwh),
            "selfgen_share": float(self.selfgen_share),
        }


# ---------------------------
# 工具：时间边界与分类
# ---------------------------
def _month_bounds(yyyy_mm: Optional[str]) -> Tuple[str, str, int]:
    """
    给定 "YYYY-MM" 返回（含起始、不含结束）的 ISO8601 UTC 边界，以及当月天数。
    若 yyyy_mm 为空，则取当前月。
    """
    now = datetime.now(timezone.utc)
    if not yyyy_mm:
        y, m = now.year, now.month
    else:
        try:
            y, m = map(int, yyyy_mm.split("-"))
        except Exception:
            # 兜底：解析失败改用当前月
            y, m = now.year, now.month

    days = calendar.monthrange(y, m)[1]
    start = datetime(y, m, 1, 0, 0, 0, tzinfo=timezone.utc)
    if m == 12:
        end = datetime(y + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        end = datetime(y, m + 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return start.isoformat(), end.isoformat(), days


def _add_months(iso_start: str, offset: int) -> str:
    """Return ``YYYY-MM`` after applying a calendar-month offset."""
    year, month = map(int, iso_start[:7].split("-"))
    absolute = year * 12 + (month - 1) + int(offset)
    shifted_year, shifted_month_zero = divmod(absolute, 12)
    return f"{shifted_year:04d}-{shifted_month_zero + 1:02d}"


def _classify(asset_id: str) -> str:
    """与 server.py 中一致的简易分类，便于 by_process 分摊。"""
    s = (asset_id or "").lower()
    if s.startswith("qc"): return "qc"
    if s.startswith("yc"): return "yc"
    if s.startswith("agv"): return "agv"
    if s.startswith("wh"): return "wh"
    if s.startswith("cs"): return "cs"
    if s.startswith("ps"): return "ps"
    if s.startswith("yard"): return "yard"
    return "misc"


# ---------------------------
# 核心服务
# ---------------------------
class ComplianceService:
    """
    合规报表服务：
      - monthly_report / quarterly_report / make_report
    依赖：
      - telemetry：资产清单（list_assets）
      - energy：今日聚合（build_today_summary）
      - forecast / reporting：可作为扩展数据源（本实现未强依赖）
    """
    def __init__(self, telemetry: Any, energy: Any, forecast: Any = None, reporting: Any = None):
        self.telemetry = telemetry
        self.energy = energy
        self.forecast = forecast
        self.reporting = reporting

    # ------- 对外 API -------

    def monthly_report(
        self,
        month_yyyy_mm: Optional[str] = None,
        teu: int = 12000,
        granularity: str = "all",
        factors: Optional[EmissionFactors] = None,
        diesel_model: str = "rule_of_thumb",
    ) -> Dict[str, Any]:
        """
        生成月度合规报告（范畴 1/2）：
          - 用今日样本日 × 当月天数 的方式外推月度电量；
          - Scope 2：购电部分；
          - Scope 1：自发电部分（与 diesel 模型估算）。
        """
        ef = self._ensure_factors(factors)
        rng_start, rng_end, days_in_month = _month_bounds(month_yyyy_mm)

        # 1) 获取“样本日”电量（每资产）
        sample = self._sample_day_energy_by_asset(teu=teu)
        day_asset = sample["by_asset"]       # [{id, kWh_est, avg_kW}]
        day_total_kwh = sample["kWh"]
        # 2) 外推到月
        month_asset = [
            {
                "id": x["id"],
                "kWh": float(x["kWh_est"]) * days_in_month
            }
            for x in day_asset
        ]
        month_total_kwh = float(day_total_kwh) * days_in_month

        # 3) Scope 2（购电）与 Scope 1（自发电 + 柴油）
        scope_split = self._split_and_emissions(month_total_kwh, ef, diesel_model=diesel_model)

        # 4) 分摊（资产 / 工艺）
        alloc = self._allocations(month_asset, scope_split)

        # 5) 汇总
        totals = {
            "electricity_kWh": round(month_total_kwh, 3),
            "oil_liters": round(scope_split["diesel_liters"], 3),
            "gas_nm3": 0.0,
            "scope1_kg": round(scope_split["scope1_kg"], 3),
            "scope2_kg": round(scope_split["scope2_kg"], 3),
            "total_kg": round(scope_split["scope1_kg"] + scope_split["scope2_kg"], 3),
            "intensity": {
                "kWh_per_TEU": round(month_total_kwh / max(1, int(teu)), 6),
                "kgCO2e_per_TEU": round((scope_split["scope1_kg"] + scope_split["scope2_kg"]) / max(1, int(teu)), 6),
            }
        }

        return {
            "range": {"start": rng_start, "end": rng_end, "days": days_in_month},
            "factors": ef.as_dict(),
            "totals": totals,
            "allocations": {
                "by_asset": alloc["by_asset"],
                "by_process": alloc["by_process"],
                "by_group": [],     # 预留
                "by_berth": [],     # 预留
            },
            "assumptions": {
                "method": "sample_day_extrapolation",
                "sample_day_hours": sample["hours"],
                "notes": [
                    "以“样本日”外推当月，无历史表计下的演示口径。",
                    "自发电份额由 selfgen_share 指定；柴油估算默认禁用（liters=0）。"
                ]
            }
        }

    def quarterly_report(
        self,
        start_month_yyyy_mm: Optional[str] = None,
        teu: int = 36000,
        granularity: str = "all",
        factors: Optional[EmissionFactors] = None,
        diesel_model: str = "rule_of_thumb",
    ) -> Dict[str, Any]:
        """
        生成季度合规报告：从 start_month 起，连续 3 个月累加。
        依旧采用“样本日 × 各月天数”的外推方法。
        """
        ef = self._ensure_factors(factors)

        # 三个月的边界与天数
        s1, e1, d1 = _month_bounds(start_month_yyyy_mm)
        # 下两个月
        s2, e2, d2 = _month_bounds(_add_months(s1, 1))
        s3, e3, d3 = _month_bounds(_add_months(s1, 2))

        # 样本日（统一一份样本用来外推三个月）
        sample = self._sample_day_energy_by_asset(teu=teu)
        day_asset = sample["by_asset"]
        day_total_kwh = sample["kWh"]

        # 外推到三个月
        def _month_pack(days: int) -> Dict[str, Any]:
            month_asset = [{"id": x["id"], "kWh": float(x["kWh_est"]) * days} for x in day_asset]
            month_total_kwh = float(day_total_kwh) * days
            split = self._split_and_emissions(month_total_kwh, ef, diesel_model=diesel_model)
            alloc = self._allocations(month_asset, split)
            return {
                "electricity_kWh": month_total_kwh,
                "scope1_kg": split["scope1_kg"],
                "scope2_kg": split["scope2_kg"],
                "diesel_liters": split["diesel_liters"],
                "alloc": alloc,
            }

        m1 = _month_pack(d1)
        m2 = _month_pack(d2)
        m3 = _month_pack(d3)

        q_kwh = m1["electricity_kWh"] + m2["electricity_kWh"] + m3["electricity_kWh"]
        q_s1 = m1["scope1_kg"] + m2["scope1_kg"] + m3["scope1_kg"]
        q_s2 = m1["scope2_kg"] + m2["scope2_kg"] + m3["scope2_kg"]
        q_oil = m1["diesel_liters"] + m2["diesel_liters"] + m3["diesel_liters"]

        totals = {
            "electricity_kWh": round(q_kwh, 3),
            "oil_liters": round(q_oil, 3),
            "gas_nm3": 0.0,
            "scope1_kg": round(q_s1, 3),
            "scope2_kg": round(q_s2, 3),
            "total_kg": round(q_s1 + q_s2, 3),
            "intensity": {
                "kWh_per_TEU": round(q_kwh / max(1, int(teu)), 6),
                "kgCO2e_per_TEU": round((q_s1 + q_s2) / max(1, int(teu)), 6),
            }
        }

        return {
            "range": {"start": s1, "end": e3, "months": 3, "days_each": [d1, d2, d3]},
            "factors": ef.as_dict(),
            "totals": totals,
            "allocations": {
                # 简化：给出加总后的分摊（按总量比例分配，减少响应体体积）
                "by_asset": self._sum_allocations([m1["alloc"]["by_asset"], m2["alloc"]["by_asset"], m3["alloc"]["by_asset"]]),
                "by_process": self._sum_allocations([m1["alloc"]["by_process"], m2["alloc"]["by_process"], m3["alloc"]["by_process"]], key="process"),
                "by_group": [],
                "by_berth": [],
            },
            "assumptions": {
                "method": "sample_day_extrapolation",
                "sample_day_hours": sample["hours"],
                "notes": [
                    "季度=三个月合计；每月均基于样本日外推。",
                    "自发电份额由 selfgen_share 指定；柴油估算默认禁用（liters=0）。"
                ]
            }
        }

    def make_report(
        self,
        config: Dict[str, Any],
        factors: Optional[EmissionFactors] = None,
        diesel_model: str = "rule_of_thumb",
    ) -> Dict[str, Any]:
        """
        通用制作：由 config 指定 period('month'|'quarter') / start_month / granularity / teu。
        """
        period = str(config.get("period", "month")).lower()
        start = config.get("start_month")
        granularity = str(config.get("granularity", "all"))
        teu = int(config.get("teu", 12000))
        if period == "quarter":
            return self.quarterly_report(
                start_month_yyyy_mm=start,
                teu=teu,
                granularity=granularity,
                factors=factors,
                diesel_model=diesel_model,
            )
        else:
            # 默认 month
            return self.monthly_report(
                month_yyyy_mm=start,
                teu=teu,
                granularity=granularity,
                factors=factors,
                diesel_model=diesel_model,
            )

    # ------- 内部实现 -------

    def _ensure_factors(self, f: Optional[EmissionFactors]) -> EmissionFactors:
        """容错地把 dict/None 转成 EmissionFactors。"""
        if isinstance(f, EmissionFactors):
            return f
        if isinstance(f, dict):
            return EmissionFactors(
                grid_g_per_kwh=float(f.get("grid_g_per_kwh", 120.0)),
                diesel_kg_per_liter=float(f.get("diesel_kg_per_liter", 2.68)),
                selfgen_kg_per_kwh=float(f.get("selfgen_kg_per_kwh", 0.70)),
                selfgen_share=float(f.get("selfgen_share", 0.0)),
            )
        return EmissionFactors()

    def _sample_day_energy_by_asset(self, teu: int = 12000, limit_assets: int = 200) -> Dict[str, Any]:
        """
        使用 EnergyService 的“今日汇总”作为样本日数据来源。
        返回：
          {
            "hours": 13.5,                     # 样本日范围小时数
            "kWh": 1234.56,                    # 样本日全站电量估计
            "by_asset": [{"id":"...","kWh_est":...,"avg_kW":...}, ...]
          }
        """
        try:
            s = self.energy.build_today_summary(teu=teu, limit_assets=limit_assets)
        except Exception:
            # 万一 energy 服务不可用，返回极简占位，避免报错
            return {"hours": 12.0, "kWh": 0.0, "by_asset": []}

        hours = float(s.get("range", {}).get("hours", 0.0))
        elec = s.get("electricity", {}) or {}
        site_kwh = float(elec.get("kWh", elec.get("kWh_est", 0.0)))
        by_asset = []
        for a in elec.get("by_asset", []):
            # The DI energy summary guarantees kWh_est for each asset.
            by_asset.append({
                "id": a.get("id"),
                "kWh_est": float(a.get("kWh_est", 0.0)),
                "avg_kW": float(a.get("avg_kW", 0.0)),
            })
        return {"hours": hours, "kWh": site_kwh, "by_asset": by_asset}

    def _split_and_emissions(self, total_kWh: float, ef: EmissionFactors, diesel_model: str = "rule_of_thumb") -> Dict[str, float]:
        """
        按自发电占比拆分电量，并计算范畴排放。
        - grid_kWh 走 Scope 2（grid_g_per_kwh）
        - selfgen_kWh 走 Scope 1（selfgen_kg_per_kwh）
        - 柴油 liters：演示中默认 0（可按现场接油表/经验模型补充）
        """
        total_kWh = max(0.0, float(total_kWh))
        self_share = min(max(float(ef.selfgen_share), 0.0), 1.0)
        selfgen_kWh = total_kWh * self_share
        grid_kWh = total_kWh - selfgen_kWh

        scope2_kg = grid_kWh * (float(ef.grid_g_per_kwh) / 1000.0)
        scope1_kg = selfgen_kWh * float(ef.selfgen_kg_per_kwh)

        diesel_liters = 0.0
        if diesel_model == "rule_of_thumb":
            # 这里可按现场扩展：例如对 'agv'/'tractor' 等设备按“作业小时 × 平均油耗率”估算
            # 演示保持 0，避免误判
            diesel_liters = 0.0
        scope1_kg += diesel_liters * float(ef.diesel_kg_per_liter)

        return {
            "grid_kWh": grid_kWh,
            "selfgen_kWh": selfgen_kWh,
            "scope1_kg": scope1_kg,
            "scope2_kg": scope2_kg,
            "diesel_liters": diesel_liters,
        }

    def _allocations(self, month_asset: List[Dict[str, Any]], split: Dict[str, float]) -> Dict[str, List[Dict[str, Any]]]:
        """
        按资产用电占比分摊 Scope1/2 -> kgCO2e。
        返回：
          {
            "by_asset": [{"id","kWh","kgCO2e","share"}...],
            "by_process": [{"process","kWh","kgCO2e","share"}...]
          }
        """
        total_kWh = sum(float(x.get("kWh", 0.0)) for x in month_asset)
        total_kWh = max(1e-9, total_kWh)  # 防止除零
        total_kg = float(split.get("scope1_kg", 0.0) + split.get("scope2_kg", 0.0))

        # by_asset
        by_asset: List[Dict[str, Any]] = []
        for x in month_asset:
            k = float(x.get("kWh", 0.0))
            share = k / total_kWh
            by_asset.append({
                "id": x.get("id"),
                "kWh": round(k, 3),
                "kgCO2e": round(total_kg * share, 3),
                "share": round(share, 6),
            })

        # by_process
        proc_map: Dict[str, float] = {}
        for x in month_asset:
            pid = _classify(str(x.get("id") or ""))
            proc_map[pid] = proc_map.get(pid, 0.0) + float(x.get("kWh", 0.0))
        by_process: List[Dict[str, Any]] = []
        for k, v in proc_map.items():
            share = v / total_kWh
            by_process.append({
                "process": k,
                "kWh": round(v, 3),
                "kgCO2e": round(total_kg * share, 3),
                "share": round(share, 6),
            })
        by_process.sort(key=lambda d: d["kWh"], reverse=True)

        return {"by_asset": by_asset, "by_process": by_process}

    @staticmethod
    def _sum_allocations(groups: List[List[Dict[str, Any]]], key: str = "id") -> List[Dict[str, Any]]:
        """
        将多个月份的 allocations 合并（同 id/process 聚合）。
        """
        agg: Dict[str, Dict[str, float]] = {}
        total_kWh = 0.0
        for arr in groups:
            for item in arr:
                name = str(item.get(key))
                kWh = float(item.get("kWh", 0.0))
                kg = float(item.get("kgCO2e", 0.0))
                total_kWh += kWh
                if name not in agg:
                    agg[name] = {"kWh": 0.0, "kgCO2e": 0.0}
                agg[name]["kWh"] += kWh
                agg[name]["kgCO2e"] += kg

        total_kWh = max(1e-9, total_kWh)
        out: List[Dict[str, Any]] = []
        for name, v in agg.items():
            share = v["kWh"] / total_kWh
            out.append({
                key: name,
                "kWh": round(v["kWh"], 3),
                "kgCO2e": round(v["kgCO2e"], 3),
                "share": round(share, 6),
            })
        out.sort(key=lambda d: d["kWh"], reverse=True)
        return out
