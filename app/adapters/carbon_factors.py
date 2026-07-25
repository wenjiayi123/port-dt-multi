# app/adapters/carbon_factors.py
"""
【大白话注释】
这个文件是“碳因子服务”，给整个平台提供“把用电/用油/用气换算成 CO₂e”的标准接口。
- 支持：电网因子（边际/平均/残余Mix等）、燃料因子（柴油/天然气/LNG/氢等）
- 版本化：不同版本（如 2025Q1）与生效时间（valid_from/valid_to），便于审计与回放
- 地区化：region（如 CN-DEFAULT/MY-DEFAULT/DE）区分地区电网结构
- 时间统一：所有时间参数都用 UTC（这里支持 ISO8601 或 epoch 秒，内部转 UTC）

【谁会调用它】
- services/energy.py（KPI 与碳核算分摊）
- services/reporting.py（合规报表）
- services/optimize.py / services/rl.py（把碳成本纳入目标函数）
- server API（对外提供 “查询某时段用电对应的 kgCO2e/TEU” 等）

【它会调用谁】
- data/factors/*.csv（默认文件数据源）
  未来切换到“电网/监管官方接口”时，只需要把 load_* 换成HTTP/DB读取，外部调用不变。

【真实落地怎么接】
- 先用 CSV 跑通，验收后把 CSV 同步到官方口径（例如电网公司/监管机构发布的地区因子表）
- 如果你们有接口：只要在 CarbonFactors.__init__ 里改成调用“官方接口”，其余不变
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal, Tuple
import csv
import pathlib
import time
import datetime as dt


# ========= 工具：时间解析（统一 UTC） =========

def _to_epoch_utc(when: Optional[str | float | int]) -> Optional[float]:
    """
    支持三种输入：
    - None：返回 None
    - epoch（float/int）
    - ISO8601（'2025-10-05T12:00:00Z' 或 '2025-10-05 12:00:00+00:00'）
    返回：UTC epoch 秒(float)
    """
    if when is None:
        return None
    if isinstance(when, (int, float)):
        return float(when)
    s = str(when).strip()
    # 兼容末尾Z/带时区/不带时区（不带时区时按 UTC 解释）
    try:
        if s.endswith("Z"):
            dtobj = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dtobj = dt.datetime.fromisoformat(s)
            if dtobj.tzinfo is None:
                dtobj = dtobj.replace(tzinfo=dt.timezone.utc)
        return dtobj.timestamp()
    except Exception:
        raise ValueError(f"Invalid time format: {when}")


# ========= 数据结构 =========

GridKind = Literal["marginal", "average", "residual_mix"]

@dataclass
class GridFactorRow:
    region: str                 # 如 CN-DEFAULT / MY-DEFAULT / DE
    kind: GridKind              # 'marginal'/'average'/'residual_mix'
    kgco2_per_kwh: float        # kgCO2e / kWh
    version: str                # 如 '2025Q1'
    valid_from: Optional[float] # epoch UTC
    valid_to: Optional[float]   # epoch UTC
    source: Optional[str] = None

@dataclass
class FuelFactorRow:
    fuel: str                   # 'diesel'/'natural_gas'/'lng'/'h2_green'/'h2_grey' 等
    scope: str                  # 'wtw'（井到轮）/'ttw'（油箱到轮）等
    unit: str                   # 因子单位（'kg_per_l' / 'kg_per_kg' / 'kg_per_kwh'）
    factor: float               # 因子数值（例如 2.68 kgCO2e/L 柴油 ttw）
    version: str                # '2025Q1'
    source: Optional[str] = None
    notes: Optional[str] = None


# ========= 核心服务 =========

class CarbonFactors:
    """
    CarbonFactors 提供稳定的“最终落地接口”：
    - grid(region, when, kind, version) -> kgCO2e/kWh
    - fuel(fuel, scope, unit, version)  -> kgCO2e per <unit>
    - 刷新/热加载：reload() 可重新加载 CSV（便于数据更新）
    """
    def __init__(
        self,
        grid_csv: str = "data/factors/grid_factors.csv",
        fuels_csv: str = "data/factors/fuel_factors.csv",
    ):
        self.grid_csv = grid_csv
        self.fuels_csv = fuels_csv
        self._grid_rows: List[GridFactorRow] = []
        self._fuel_rows: List[FuelFactorRow] = []
        self.reload()

    # ----- 加载 -----
    def reload(self) -> None:
        self._grid_rows = self._load_grid_csv(self.grid_csv)
        self._fuel_rows = self._load_fuels_csv(self.fuels_csv)

    def _load_grid_csv(self, path: str) -> List[GridFactorRow]:
        rows: List[GridFactorRow] = []
        p = pathlib.Path(path)
        if not p.exists():
            return rows
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append(GridFactorRow(
                        region=r.get("region", "").strip(),
                        kind=r.get("kind", "marginal").strip() or "marginal",
                        kgco2_per_kwh=float(r.get("kgco2_per_kwh", "0") or 0),
                        version=r.get("version", "").strip(),
                        valid_from=_to_epoch_utc(r.get("valid_from") or None),
                        valid_to=_to_epoch_utc(r.get("valid_to") or None),
                        source=(r.get("source") or "").strip() or None,
                    ))
                except Exception:
                    # 略过坏行，真实落地可写入质量日志
                    continue
        # 可按有效期、版本排序，方便“未指定版本”时挑最近的
        rows.sort(key=lambda x: (x.region, x.kind, x.valid_from or 0, x.version))
        return rows

    def _load_fuels_csv(self, path: str) -> List[FuelFactorRow]:
        rows: List[FuelFactorRow] = []
        p = pathlib.Path(path)
        if not p.exists():
            return rows
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append(FuelFactorRow(
                        fuel=r.get("fuel", "").strip().lower(),
                        scope=r.get("scope", "ttw").strip().lower(),
                        unit=r.get("unit", "kg_per_l").strip().lower(),
                        factor=float(r.get("factor", "0") or 0),
                        version=r.get("version", "").strip(),
                        source=(r.get("source") or "").strip() or None,
                        notes=(r.get("notes") or "").strip() or None,
                    ))
                except Exception:
                    continue
        rows.sort(key=lambda x: (x.fuel, x.scope, x.unit, x.version))
        return rows

    # ----- 查询：电网因子 -----
    def grid(
        self,
        *,
        region: str,
        when: Optional[str | float | int] = None,
        kind: GridKind = "marginal",
        version: Optional[str] = None,
    ) -> Optional[float]:
        """
        查电网因子（kgCO2e/kWh）。
        优先级：
          1) region + kind + version 完全匹配
          2) region + kind + 当时落在 [valid_from, valid_to)
          3) region + kind + 最近版本（按 valid_from/版本排序）
        """
        rows = [r for r in self._grid_rows if r.region == region and r.kind == kind]
        if not rows:
            return None

        # 1) 指定版本
        if version:
            for r in reversed(rows):
                if r.version == version:
                    return r.kgco2_per_kwh

        # 2) 指定时间
        t = _to_epoch_utc(when) if when is not None else None
        if t is not None:
            # 找生效窗口覆盖 t 的条目
            for r in reversed(rows):
                if (r.valid_from is None or t >= r.valid_from) and (r.valid_to is None or t < r.valid_to):
                    return r.kgco2_per_kwh

        # 3) 兜底：取最近（按排序后的最后一条）
        return rows[-1].kgco2_per_kwh if rows else None

    # ----- 查询：燃料因子 -----
    def fuel(
        self,
        *,
        fuel: str,
        scope: str = "ttw",
        unit: str = "kg_per_l",
        version: Optional[str] = None,
    ) -> Optional[float]:
        """
        查燃料因子，返回单位对应的 kgCO2e/<unit>。
        例：
          fuel('diesel', scope='ttw', unit='kg_per_l') -> 2.68（示例值）
        """
        f = fuel.lower()
        sc = scope.lower()
        un = unit.lower()
        rows = [r for r in self._fuel_rows if r.fuel == f and r.scope == sc and r.unit == un]
        if not rows:
            return None

        if version:
            for r in reversed(rows):
                if r.version == version:
                    return r.factor

        return rows[-1].factor

    # ----- 帮助：把当前加载的因子写入 FactorRepo（可选） -----
    def populate_factor_repo(self, factor_repo) -> None:
        """
        把当前内存中的因子写入 core.repositories.FactorRepo
        （便于其它模块统一从 FactorRepo 取值；非必须）
        """
        for r in self._grid_rows:
            factor_repo.upsert_grid_factor(r.region, r.kind, r.version, r.kgco2_per_kwh)


# ========= 直接运行的自测 =========

def _smoke() -> dict:
    cf = CarbonFactors()  # 默认读取 data/factors/*.csv
    g_cn = cf.grid(region="CN-DEFAULT", when="2025-10-05T12:00:00Z", kind="marginal")
    g_my = cf.grid(region="MY-DEFAULT", when="2025-10-05T12:00:00Z", kind="average")
    d_ttw = cf.fuel(fuel="diesel", scope="ttw", unit="kg_per_l")
    h2_green = cf.fuel(fuel="h2_green", scope="wtw", unit="kg_per_kg")
    return {
        "grid_CN_marginal": g_cn,
        "grid_MY_average": g_my,
        "diesel_ttw_kg_per_l": d_ttw,
        "h2_green_wtw_kg_per_kg": h2_green,
    }


if __name__ == "__main__":
    # Supports direct execution with: python -m app.adapters.carbon_factors
    import json
    print(json.dumps(_smoke(), ensure_ascii=False, indent=2))
