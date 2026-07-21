# -*- coding: utf-8 -*-
"""
simulation.py
--------------
“预测与仿真”模块的第一块：混合仿真引擎（离散事件 + 连续能流）。
大白话：
- 你把“船什么时候来、多少箱子要干、设备有多少、车什么时候充电、储能怎么配、电价碳价怎么变”等丢进来，
  我按 15 分钟（或 1 分钟）一步一步往前推，把“功率曲线、能耗、碳排、峰值、SOC、设备利用率”等结果按时间吐出来。
- 结果会保存到 data/objects/simulations 目录，包含“证据包”（输入、模型版本、策略、结果统计），方便回放/追责。
- 真实落地时，只需要把下面的 TOS/EMS/SCADA 接口换成实际实现就行（这里先用 demo 生成器撑起来）。

和谁有关：
- 现在：可以被命令行直接运行（__main__），也可以被未来的 RL/Forecast 服务调用。
- 之后：server.py 会挂 API，把这个服务作为后端沙盒；reporting.py/explain.py 会读它的输出做报表/解释。
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Protocol, Tuple, Iterable

import numpy as np
import pandas as pd

# 统一时区（马来西亚 / 新加坡 / 中国都常用这个东八区时基；前端可以再切换）
TZ = "Asia/Kuala_Lumpur"

# ========== 一些对接真实系统时要替换的接口（Protocol） ==========
class TOSClient(Protocol):
    """对接码头作业系统（Terminal Operating System）：
    - 给我一个时间窗，我要拿到“泊位/船舶靠离时间 + 预计箱量（装/卸/冷/危险等可细分）”
    - 到真实上线时，把这个协议用你们的 TOS SDK/HTTP API 实现即可。
    """
    def get_vessel_schedule(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        返回列至少包含：
        ['vessel_id','berth_id','eta','etd','teu_plan','reefer_ratio']
        时间列 tz-aware（东八区）
        """
        ...


class EMSClient(Protocol):
    """对接能量管理系统（Energy Management System）/微网控制：
    - 获取/下发储能功率上下限、效率、SOC 上下限等
    - 真实上线换成实际 EMS/PCS API 即可
    """
    def get_bess_specs(self) -> dict:
        """
        返回: {'p_charge_max_kW':..., 'p_discharge_max_kW':..., 'e_capacity_kWh':..., 'soc_min':0.1, 'soc_max':0.9, 'eta_charge':0.95, 'eta_discharge':0.95}
        """
        ...


class SCADAClient(Protocol):
    """对接 SCADA/PLC/DDC：
    - 拉到当前/历史设备功率、状态；这里只要接口签名对了，真实上线替换实现即可
    """
    def get_equipment_catalog(self) -> pd.DataFrame:
        """返回设备清单：['equip_id','equip_type','rated_kW','area','is_critical']"""
        ...
    def get_baseline_power(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """返回历史或基线功率：index=时间，列=设备ID，值=kW"""
        ...


# ========== 配置/数据结构 ==========
@dataclass
class Tariff:
    """分时电价/需量等价信息；真实上线从电力公司/合同接口拉"""
    tou: List[Tuple[str, float]]  # e.g. [('00:00-07:00', 0.5), ('07:00-23:00', 0.9), ('23:00-24:00', 0.6)]  单位：RM/kWh
    demand_charge_rm_per_kW: float = 0.0

    def price_at(self, ts: pd.Timestamp) -> float:
        t = ts.tz_convert(TZ).time()
        for span, price in self.tou:
            s, e = span.split("-")
            sh, sm = map(int, s.split(":"))
            eh, em = map(int, e.split(":"))
            sinterval = (sh, sm)
            einterval = (eh, em)
            if (t.hour, t.minute) >= sinterval and (t.hour, t.minute) < einterval:
                return price
        # fallback
        return self.tou[-1][1] if self.tou else 0.0


@dataclass
class CarbonFactorSchedule:
    """碳排因子（kgCO2e/kWh），可随时间变（边际/平均），真实上线从 data/factors/grid_factors.csv 读取并按地区/时段匹配"""
    time_series: pd.Series  # index=ts, value=factor

    def factor_at(self, ts: pd.Timestamp) -> float:
        if ts in self.time_series.index:
            return float(self.time_series.loc[ts])
        # 最近邻近似
        idx = self.time_series.index.get_indexer([ts], method="nearest")[0]
        return float(self.time_series.iloc[idx])


@dataclass
class Strategy:
    """策略输入（简版）：
    - 充放电：在每个时间步指定储能功率（+放电/-充电）；也可以传一个“目标峰值”，引擎内自算
    - 照明/冷站：区域开关或设定点偏移（这里只给简版参数位，后续可按设备粒度覆盖）
    """
    bess_dispatch_kW: Optional[pd.Series] = None
    peak_shaving_target_kW: Optional[float] = None
    yard_lighting_on_ratio: float = 1.0      # [0,1]
    chiller_sp_offset_degC: float = 0.0      # +2 表示升2度，省电

    def to_dict(self):
        d = {
            "peak_shaving_target_kW": self.peak_shaving_target_kW,
            "yard_lighting_on_ratio": self.yard_lighting_on_ratio,
            "chiller_sp_offset_degC": self.chiller_sp_offset_degC,
        }
        if self.bess_dispatch_kW is not None:
            d["bess_dispatch_kW"] = self.bess_dispatch_kW.round(3).to_dict()
        return d


@dataclass
class Scenario:
    """场景定义（可视化大屏“一键切换”的三选项之一：现在/预测6小时/策略仿真）
    - horizon: 仿真时长
    - resolution: 时间步长（'15min' 或 '1min'）
    - demand_scale: 作业强度缩放（用来做 P50/P90 的扰动）
    """
    start: pd.Timestamp
    horizon: timedelta
    resolution: str = "15min"
    demand_scale: float = 1.0
    name: str = "default"


@dataclass
class SimResult:
    """仿真结果打包：时序 + KPI + 证据包路径"""
    timeseries: pd.DataFrame
    kpis: Dict[str, float]
    evidence_path: str


# ========== Demo 适配器（没有真实系统也能跑） ==========
class DemoTOS(TOSClient):
    def get_vessel_schedule(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # 大白话：随机构造 1~3 条船，靠 4~8 小时，箱量 2000~6000 TEU，冷箱比例 5~20%
        random.seed(42)
        rows = []
        cursor = start + pd.Timedelta(hours=1)
        for v in range(random.randint(1, 3)):
            eta = cursor + pd.Timedelta(hours=random.randint(0, 4))
            stay = random.randint(4, 8)
            etd = eta + pd.Timedelta(hours=stay)
            rows.append({
                "vessel_id": f"V{100+v}",
                "berth_id": f"B{1+v}",
                "eta": eta.tz_convert(TZ),
                "etd": etd.tz_convert(TZ),
                "teu_plan": random.randint(2000, 6000),
                "reefer_ratio": round(random.uniform(0.05, 0.2), 3),
            })
            cursor = etd + pd.Timedelta(hours=2)
        return pd.DataFrame(rows)


class DemoEMS(EMSClient):
    def get_bess_specs(self) -> dict:
        # 一套 2MW/4MWh 的储能，SOC 在 10%~90%，充放电效率 95%
        return {
            "p_charge_max_kW": 2000.0,
            "p_discharge_max_kW": 2000.0,
            "e_capacity_kWh": 4000.0,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "eta_charge": 0.95,
            "eta_discharge": 0.95,
        }


class DemoSCADA(SCADAClient):
    def get_equipment_catalog(self) -> pd.DataFrame:
        # 设备清单（简化）：岸桥 6 台、场桥 12 台、AGV 40 台、照明、冷站、杂项
        data = [
            *[("QC{:02d}".format(i), "quay_crane", 500, "berth", True) for i in range(1, 7)],
            *[("YC{:02d}".format(i), "yard_crane", 120, "yard", False) for i in range(1, 13)],
            *[("AGV{:02d}".format(i), "agv", 40, "yard", False) for i in range(1, 41)],
            ("LIGHT", "lighting", 600, "yard", False),
            ("CHILLER", "chiller", 1200, "utility", True),
            ("MISC", "misc", 300, "utility", False),
        ]
        return pd.DataFrame(data, columns=["equip_id","equip_type","rated_kW","area","is_critical"])

    def get_baseline_power(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # 生成一个基础负荷曲线（不含作业负荷），lighting 与 chiller 按昼夜&温度/作业热区略微变化（这里简化成周期）
        idx = pd.date_range(start, end, freq="15min", inclusive="left", tz=TZ)
        base = pd.DataFrame(index=idx)
        base["LIGHT"] = 400 + 200*(idx.hour >= 19)  # 夜间更亮
        base["CHILLER"] = 900 + 300*np.sin(2*np.pi*(idx.hour/24.0))  # 日变化
        base["MISC"] = 300.0
        # 其他设备默认 0，作业时才上
        for col in [*["QC{:02d}".format(i) for i in range(1,7)],
                    *["YC{:02d}".format(i) for i in range(1,13)],
                    *["AGV{:02d}".format(i) for i in range(1,41)]]:
            base[col] = 0.0
        return base


# ========== 引擎主体 ==========
class HybridSimEngine:
    """
    混合仿真引擎：把“离散事件（靠离泊/队列/节拍）”和“连续能流（功率/SOC/冷站/照明）”统一在时间轴上推进。
    大白话：
    - 每一个时间步（比如 15min），我先看看有没有“事件”（船来/走、作业节奏变化），再算这一小段里设备功率/储能/SOC。
    - 最终拼成整段时序结果，再算 KPI 和费用/碳排。
    """

    MODEL_VERSION = "sim-v0.1.0"

    def __init__(
        self,
        tos: TOSClient,
        ems: EMSClient,
        scada: SCADAClient,
        tariff: Tariff,
        carbon: CarbonFactorSchedule,
    ):
        self.tos = tos
        self.ems = ems
        self.scada = scada
        self.tariff = tariff
        self.carbon = carbon

    # ---- 核心：运行一次仿真 ----
    def run(self, scenario: Scenario, strategy: Strategy) -> SimResult:
        start = scenario.start.tz_convert(TZ)
        end = (scenario.start + scenario.horizon).tz_convert(TZ)

        # 设备清单 + 基线功率
        catalog = self.scada.get_equipment_catalog()
        base = self.scada.get_baseline_power(start, end)  # index=ts, 列=设备ID

        # 船期作业 -> 生成“生产驱动负荷”（岸桥、场桥、AGV）
        schedule = self.tos.get_vessel_schedule(start, end)
        ops_load = self._build_ops_load(schedule, base.index, catalog, scenario.demand_scale)

        # 策略作用（照明开关比例、冷站设定点偏移 -> 功率缩放）
        base_adj = base.copy()
        base_adj["LIGHT"] *= float(strategy.yard_lighting_on_ratio)
        # 冷站：+1°C 约省 4~7%（取 5%/°C 简化）
        chiller_factor = max(0.0, 1.0 - 0.05*float(strategy.chiller_sp_offset_degC))
        base_adj["CHILLER"] *= chiller_factor

        # 总功率（不含储能）：基线 + 作业
        total_no_bess = base_adj.add(ops_load, fill_value=0.0)
        total_no_bess["TOTAL_kW_no_bess"] = total_no_bess.sum(axis=1)

        # 储能功率轨迹（如果没给，就按“削峰目标”自算一个最简单的轨迹）
        bess_specs = self.ems.get_bess_specs()
        dispatch = self._plan_bess_dispatch(
            total_no_bess["TOTAL_kW_no_bess"],
            strategy,
            bess_specs,
            resolution=scenario.resolution
        )

        # 计算 SOC & 总功率 & 能量 & 碳排
        ts = total_no_bess.index
        df = pd.DataFrame(index=ts)
        df["p_no_bess_kW"] = total_no_bess["TOTAL_kW_no_bess"]
        df["p_bess_kW"] = dispatch
        # 电池放电（正数）降低购电功率；充电（负数）提高购电功率
        df["p_grid_kW"] = df["p_no_bess_kW"] - df["p_bess_kW"]

        # 积分得到 kWh（按分辨率换算）
        step_h = pd.Timedelta(scenario.resolution).total_seconds()/3600.0
        df["e_grid_kWh"] = df["p_grid_kW"] * step_h

        # 简化：SOC **在外面**用能量守恒 + 充放电效率计算
        df["soc"] = self._integrate_soc(dispatch, bess_specs, step_h)

        # 费用 & 碳排
        prices = df.index.map(self.tariff.price_at)
        df["price_RM_per_kWh"] = prices
        df["cost_energy_RM"] = df["e_grid_kWh"] * df["price_RM_per_kWh"]

        carbon_factors = df.index.map(self.carbon.factor_at)
        df["carbon_factor_kg_per_kWh"] = carbon_factors
        df["emission_kgCO2e"] = df["e_grid_kWh"] * df["carbon_factor_kg_per_kWh"]

        # 需量/峰值（简单取窗口内最大功率）
        peak_kW = float(df["p_grid_kW"].max())
        demand_cost = peak_kW * float(self.tariff.demand_charge_rm_per_kW)

        kpis = {
            "energy_kWh": float(df["e_grid_kWh"].sum()),
            "peak_kW": peak_kW,
            "cost_energy_RM": float(df["cost_energy_RM"].sum()),
            "cost_demand_RM": demand_cost,
            "cost_total_RM": float(df["cost_energy_RM"].sum()) + demand_cost,
            "emission_kgCO2e": float(df["emission_kgCO2e"].sum()),
            "p95_percentile_kW": float(np.percentile(df["p_grid_kW"], 95)),
        }

        evidence_path = self._persist_evidence(scenario, strategy, catalog, schedule, df, kpis)

        return SimResult(timeseries=df, kpis=kpis, evidence_path=evidence_path)

    # ---- 生产驱动负荷（离散事件 -> 平摊到时间步上的设备功率）----
    def _build_ops_load(
        self,
        schedule: pd.DataFrame,
        timeline: pd.DatetimeIndex,
        catalog: pd.DataFrame,
        demand_scale: float
    ) -> pd.DataFrame:
        """
        大白话：
        - 船来了 -> 岸桥开工（按 teu_plan / 生产率 算出需要多少小时，分配到每个时间步）
        - 岸桥有活 -> 场桥与 AGV 也跟着跑（按比例系数给功率）
        - 结果：每个时间步上，QC/YC/AGV 的“作业负荷”有值（其它时候为 0）
        """
        out = pd.DataFrame(0.0, index=timeline, columns=catalog["equip_id"].tolist())

        # 假设：岸桥平均生产率 30 TEU/小时/台；场桥与 AGV 用功率比跟行车比例
        qc_ids = [e for e,t in zip(catalog["equip_id"],catalog["equip_type"]) if t=="quay_crane"]
        yc_ids = [e for e,t in zip(catalog["equip_id"],catalog["equip_type"]) if t=="yard_crane"]
        agv_ids = [e for e,t in zip(catalog["equip_id"],catalog["equip_type"]) if t=="agv"]
        rated = dict(zip(catalog["equip_id"], catalog["rated_kW"]))

        qc_rate_teu_per_hr = 30.0
        # 基于 Reefer 比例调整“单位箱能耗”（冷箱更慢一点）
        for _, row in schedule.iterrows():
            eta: pd.Timestamp = pd.Timestamp(row["eta"]).tz_convert(TZ)
            etd: pd.Timestamp = pd.Timestamp(row["etd"]).tz_convert(TZ)
            teu = float(row["teu_plan"]) * demand_scale
            reefer_ratio = float(row.get("reefer_ratio", 0.1))
            work_hours = teu / (qc_rate_teu_per_hr * max(1, len(qc_ids))) * (1.0 + 0.2*reefer_ratio)
            # 简单：从 ETA 开始干，如果超过 ETD 就截断
            work_end = min(eta + pd.Timedelta(hours=work_hours), etd)

            # 在 [eta, work_end) 区间里，让 QC/YC/AGV “上电”
            mask = (timeline >= eta) & (timeline < work_end)
            active_steps = int(mask.sum())

            if active_steps <= 0:
                continue

            # 简化功率模型：活跃时段按“额定功率的负载因子”取值
            # QC 70%、YC 60%、AGV 40% 作为平均负载因子
            for q in qc_ids:
                out.loc[mask, q] += 0.7 * rated[q]
            for y in yc_ids:
                out.loc[mask, y] += 0.6 * rated[y] / (len(yc_ids)/max(1,len(qc_ids))*1.0)  # 粗略摊派
            for a in agv_ids:
                out.loc[mask, a] += 0.4 * rated[a] / (len(agv_ids)/max(1,len(qc_ids))*1.0)

        return out

    # ---- 储能调度（若没给轨迹，则按“削峰目标”生成一个最简单的轨迹） ----
    def _plan_bess_dispatch(
        self,
        p_no_bess: pd.Series,
        strategy: Strategy,
        specs: dict,
        resolution: str = "15min"
    ) -> pd.Series:
        idx = p_no_bess.index
        step_h = pd.Timedelta(resolution).total_seconds()/3600.0

        if strategy.bess_dispatch_kW is not None:
            # 来自上游策略（RL/MILP），做一下功率限幅
            disp = strategy.bess_dispatch_kW.reindex(idx).fillna(0.0).astype(float)
            return disp.clip(-specs["p_charge_max_kW"], specs["p_discharge_max_kW"])

        target = strategy.peak_shaving_target_kW
        if target is None:
            # 没给就默认“把 95 分位功率拉平到 P90”作为个超简目标
            p95 = np.percentile(p_no_bess, 95)
            p90 = np.percentile(p_no_bess, 90)
            target = float((p95 + p90)/2.0)

        disp = pd.Series(0.0, index=idx)
        soc = 0.5 * specs["e_capacity_kWh"]  # 50% 初始（kWh）
        e_min = specs["soc_min"] * specs["e_capacity_kWh"]
        e_max = specs["soc_max"] * specs["e_capacity_kWh"]

        for t in idx:
            p = float(p_no_bess.loc[t])
            # 需要放电（削峰）
            if p > target:
                p_need = min(specs["p_discharge_max_kW"], p - target)
                e_need = p_need * step_h / specs["eta_discharge"]
                if soc - e_need >= e_min:
                    disp.loc[t] = +p_need
                    soc -= e_need
                else:
                    # 能量不够，尽力放
                    e_avail = max(0.0, soc - e_min)
                    p_can = e_avail * specs["eta_discharge"] / step_h
                    disp.loc[t] = +min(p_need, p_can)
                    soc -= disp.loc[t]*step_h/specs["eta_discharge"]
            # 需要充电（抬谷）
            elif p < target*0.8:
                p_need = min(specs["p_charge_max_kW"], target*0.9 - p)
                e_need = p_need * step_h * specs["eta_charge"]
                if soc + e_need <= e_max:
                    disp.loc[t] = -p_need
                    soc += e_need
                else:
                    e_avail = max(0.0, e_max - soc)
                    p_can = e_avail / (step_h*specs["eta_charge"])
                    disp.loc[t] = -min(p_need, p_can)
                    soc += (-disp.loc[t])*step_h*specs["eta_charge"]
            else:
                disp.loc[t] = 0.0

        return disp.clip(-specs["p_charge_max_kW"], specs["p_discharge_max_kW"])

    # ---- SOC 积分（考虑效率） ----
    def _integrate_soc(self, bess_dispatch: pd.Series, specs: dict, step_h: float) -> pd.Series:
        e = 0.5 * specs["e_capacity_kWh"]  # 初始 50% 容量
        e_min = specs["soc_min"] * specs["e_capacity_kWh"]
        e_max = specs["soc_max"] * specs["e_capacity_kWh"]
        soc = []
        for p in bess_dispatch:
            if p > 0:  # 放电
                e_delta = p * step_h / specs["eta_discharge"]
                e = max(e_min, e - e_delta)
            elif p < 0:  # 充电
                e_delta = (-p) * step_h * specs["eta_charge"]
                e = min(e_max, e + e_delta)
            soc.append(e / specs["e_capacity_kWh"])
        return pd.Series(soc, index=bess_dispatch.index)

    # ---- 证据包落盘 ----
    def _persist_evidence(
        self,
        scenario: Scenario,
        strategy: Strategy,
        catalog: pd.DataFrame,
        schedule: pd.DataFrame,
        df: pd.DataFrame,
        kpis: Dict[str, float],
    ) -> str:
        base_dir = os.path.join("data", "objects", "simulations")
        os.makedirs(base_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(base_dir, f"run-{ts_str}")
        os.makedirs(run_dir, exist_ok=True)

        # 时序结果
        df.to_parquet(os.path.join(run_dir, "timeseries.parquet"))

        # 证据包（JSON）
        evidence = {
            "model_version": self.MODEL_VERSION,
            "scenario": {
                "start": str(scenario.start),
                "horizon_hours": scenario.horizon.total_seconds()/3600.0,
                "resolution": scenario.resolution,
                "demand_scale": scenario.demand_scale,
                "name": scenario.name,
            },
            "strategy": strategy.to_dict(),
            "kpis": kpis,
            "catalog_head": catalog.head(5).to_dict(orient="records"),
            "schedule": schedule.to_dict(orient="records"),
        }
        with open(os.path.join(run_dir, "evidence.json"), "w", encoding="utf-8") as f:
            json.dump(evidence, f, ensure_ascii=False, indent=2)

        return run_dir


# ========== Monte Carlo（P50/P90）辅助 ==========
def monte_carlo(
    engine: HybridSimEngine,
    scenario: Scenario,
    strategy: Strategy,
    n: int = 50,
    demand_jitter: float = 0.1,
) -> Dict[str, float]:
    """
    大白话：
    - 把作业强度（demand_scale）做点扰动，重复仿真 n 次，汇总关键指标的 P50/P90。
    - 这里先输出功率峰值和总能耗/碳排的 P50/P90，后续可以把更多 KPI 加进来。
    """
    peaks = []
    energies = []
    emissions = []
    for i in range(n):
        scn = Scenario(
            start=scenario.start,
            horizon=scenario.horizon,
            resolution=scenario.resolution,
            demand_scale=max(0.5, np.random.normal(scenario.demand_scale, demand_jitter)),
            name=f"{scenario.name}-mc{i:02d}",
        )
        res = engine.run(scn, strategy)
        peaks.append(res.kpis["peak_kW"])
        energies.append(res.kpis["energy_kWh"])
        emissions.append(res.kpis["emission_kgCO2e"])
    return {
        "peak_P50_kW": float(np.percentile(peaks, 50)),
        "peak_P90_kW": float(np.percentile(peaks, 90)),
        "energy_P50_kWh": float(np.percentile(energies, 50)),
        "energy_P90_kWh": float(np.percentile(energies, 90)),
        "emission_P50_kg": float(np.percentile(emissions, 50)),
        "emission_P90_kg": float(np.percentile(emissions, 90)),
    }


# ========== 命令行入口（可直接跑起来验货） ==========
def _demo_tariff_and_carbon(index: pd.DatetimeIndex) -> Tuple[Tariff, CarbonFactorSchedule]:
    # TOU：07:00-23:00 高峰 0.85 RM/kWh，其它 0.55 RM/kWh；需量费 35 RM/kW（示意）
    tariff = Tariff(
        tou=[("00:00-07:00", 0.55), ("07:00-23:00", 0.85), ("23:00-24:00", 0.55)],
        demand_charge_rm_per_kW=35.0
    )
    # 碳因子（简化：全天 0.6 kgCO2e/kWh），真实落地可从 data/factors/grid_factors.csv 根据国家/电网取
    cf = pd.Series(0.6, index=index)
    carbon = CarbonFactorSchedule(cf)
    return tariff, carbon


def run_demo(start_iso: str, horizon_hours: int = 12, resolution: str = "15min") -> SimResult:
    # 更稳健的时区处理：若字符串无时区，则本地化到 KL；若有时区，则统一转到 KL
    start_ts = pd.Timestamp(start_iso)
    if start_ts.tzinfo is None:
        start = start_ts.tz_localize(TZ)
    else:
        start = start_ts.tz_convert(TZ)

    end = start + timedelta(hours=int(horizon_hours))

    idx = pd.date_range(
        start,
        end,
        freq=resolution,
        inclusive="left",
        tz=TZ
    )
    tariff, carbon = _demo_tariff_and_carbon(idx)

    engine = HybridSimEngine(
        tos=DemoTOS(),
        ems=DemoEMS(),
        scada=DemoSCADA(),
        tariff=tariff,
        carbon=carbon,
    )
    # 统一使用标准库 timedelta，避免 pandas Timedelta 的版本差异
    scenario = Scenario(start=start, horizon=timedelta(hours=int(horizon_hours)), resolution=resolution, name="demo")
    strategy = Strategy(peak_shaving_target_kW=None, yard_lighting_on_ratio=0.8, chiller_sp_offset_degC=1.0)
    return engine.run(scenario, strategy)


if __name__ == "__main__":
    import argparse
    from datetime import timedelta

    parser = argparse.ArgumentParser(description="Hybrid Simulation Engine")
    parser.add_argument("--demo", action="store_true", help="Run demo scenario")
    parser.add_argument("--start", type=str, default=None, help="ISO start time (e.g., 2025-10-06T00:00:00+08:00)")
    parser.add_argument("--hours", type=int, default=12, help="horizon hours")
    parser.add_argument("--resolution", type=str, default="15min", choices=["1min","5min","15min"])
    parser.add_argument("--mc", type=int, default=0, help="Monte Carlo runs (0 to skip)")
    args = parser.parse_args()

    if args.demo:
        # —— demo 主流程 ——
        start_iso = args.start or datetime.now().astimezone().isoformat()
        res = run_demo(start_iso, horizon_hours=args.hours, resolution=args.resolution)
        print("[OK] Demo simulation done")
        print("KPIs:", json.dumps(res.kpis, ensure_ascii=False, indent=2))
        print("Evidence:", res.evidence_path)

        # —— Monte Carlo（在 demo 分支内，能拿到 start_iso）——
        if args.mc and args.mc > 0:
            start_ts = pd.Timestamp(start_iso)
            if start_ts.tzinfo is None:
                start = start_ts.tz_localize(TZ)
            else:
                start = start_ts.tz_convert(TZ)

            end = start + timedelta(hours=int(args.hours))
            idx = pd.date_range(start, end, freq=args.resolution, inclusive="left", tz=TZ)
            tariff, carbon = _demo_tariff_and_carbon(idx)
            engine = HybridSimEngine(DemoTOS(), DemoEMS(), DemoSCADA(), tariff, carbon)
            mc_scn = Scenario(start=start, horizon=timedelta(hours=int(args.hours)), resolution=args.resolution, name="demo-mc")
            mc_strategy = Strategy()
            mc = monte_carlo(engine, mc_scn, mc_strategy, n=args.mc)
            print("MC P50/P90:", json.dumps(mc, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
