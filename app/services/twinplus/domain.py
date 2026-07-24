"""TwinPlus Domain Models
--------------------------
该文件定义 TwinPlus 相关的数据模型，包括：
- PortProfile / BootstrapResponse：TwinPlus 主卡片（港口画像）。
- Fidelity*：Twin Fidelity 可信度视图（雷达图、场景分解、参数表等）。
- ScenarioRunResponse / CalibrateResponse / ReplayResponse：TwinPlus 交互 API 返回体。

设计原则：
- 所有模型统一继承 _SafeModel，允许额外字段(extra='allow')，避免轻微字段漂移导致报错。
- 字段尽量标为 Optional，并给默认值，这样 repo 里即使缺部分字段也不会抛异常。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pydantic v1/v2 兼容
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - 理论上不会走到
    from pydantic import BaseModel  # type: ignore
    Field = lambda default=None, **_: default  # type: ignore


class _SafeModel(BaseModel):
    """所有 TwinPlus 模型的基类：允许额外字段。"""

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
#  港口画像：PortProfile / BootstrapResponse
# ---------------------------------------------------------------------------


class OpsKPIs(_SafeModel):
    """运营 KPI 快照。字段保持宽松，实际内容由 repo.bootstrap_shanghai() 决定。"""

    throughput_teu: Optional[float] = None
    quay_crane_eff: Optional[float] = None
    yard_crane_eff: Optional[float] = None
    berth_occupancy: Optional[float] = None
    vessel_punctuality: Optional[float] = None


class EnergySnapshot(_SafeModel):
    """能源 & 碳排放画像。"""

    annual_demand_mwh: Optional[float] = None
    annual_bill_m_cny: Optional[float] = None
    annual_co2_t: Optional[float] = None
    bess_capacity_mwh: Optional[float] = None
    pv_capacity_mw: Optional[float] = None


class PortProfile(_SafeModel):
    """TwinPlus 主概要：港口 Phase、Twin Fidelity、节能收益等。"""

    id: Optional[str] = None
    name: Optional[str] = None
    country: Optional[str] = None
    code: Optional[str] = None  # e.g. CNSHA
    phase: Optional[str] = None  # 规划 / POC / 量产
    twin_fidelity: Optional[float] = None

    ops: OpsKPIs = Field(default_factory=OpsKPIs)
    energy: EnergySnapshot = Field(default_factory=EnergySnapshot)

    # 其它字段允许通过 extra='allow' 扩展


class BootstrapResponse(_SafeModel):
    """bootstrap_shanghai() 结果：写入哪些文件等。"""

    data_dir: Optional[str] = None
    written: List[str] = Field(default_factory=list)


def coerce_port_profile(data: Dict[str, Any]) -> PortProfile:
    """容错构造 PortProfile：缺少 ops / energy 时自动补空壳。"""
    d = dict(data or {})
    d.setdefault("ops", {})
    d.setdefault("energy", {})
    return PortProfile(**d)


# ---------------------------------------------------------------------------
#  Twin Fidelity 可信度视图
# ---------------------------------------------------------------------------


class FidelityGroup(_SafeModel):
    """Twin Fidelity - 分场景/分子系统的得分条形图。"""

    key: Optional[str] = None  # 场景/子系统标识
    name: Optional[str] = None
    score: Optional[float] = None
    mape: Optional[float] = None
    sample_size: Optional[int] = None


class FidelityRadar(_SafeModel):
    """
    Twin Fidelity - 雷达图数据。

    对应 repo._radar_from_params() 返回的结构：
        return FidelityRadar(labels=labels, old=old_v, new=new_v)
    """
    labels: List[str] = Field(default_factory=list)   # ["eff","loss","ramp","cap","delay"]
    old: List[float] = Field(default_factory=list)    # 基线参数对应的 0~1 值
    new: List[float] = Field(default_factory=list)    # 最新参数对应的 0~1 值


class ParamChange(_SafeModel):
    """Twin Fidelity - 参数对比 / 调参敏感性的一行。"""

    code: Optional[str] = None
    name: Optional[str] = None
    unit: Optional[str] = None

    baseline: Optional[float] = None
    value: Optional[float] = None
    delta: Optional[float] = None

    impact: Optional[str] = None  # 对 KPI 影响的文字描述（可选）


class FidelityPayload(_SafeModel):
    """/api/twin/fidelity 的主体数据结构，对应 repo.compute_fidelity() 里构造的字段。"""

    # 基本信息
    port_code: Optional[str] = None
    scene_code: Optional[str] = None
    horizon_days: Optional[int] = None

    # 总体分数
    score: Optional[float] = None
    coverage: Optional[float] = None
    stress: Optional[float] = None

    # 设备组误差条形图：repo 里 metrics: List[FidelityGroup]
    groups: List[FidelityGroup] = Field(default_factory=list)

    # 场景通过率：scen_rates: Dict[str, float]
    scenarios: Dict[str, float] = Field(default_factory=dict)

    # 参数变化表：changes: List[ParamChange]
    params: List[ParamChange] = Field(default_factory=list)

    # 参数雷达图（注意：这里是“单个对象”，不是列表）
    radar: FidelityRadar = Field(default_factory=FidelityRadar)

    # 更新时间
    updated_at: Optional[str] = None


class FidelityResponse(_SafeModel):
    """/api/twin/fidelity 顶层响应。"""

    status: str = "ok"
    payload: FidelityPayload = Field(default_factory=FidelityPayload)


# ---------------------------------------------------------------------------
#  交互类 API：场景仿真 / 标定 / 回放
# ---------------------------------------------------------------------------


class ScenarioRunResponse(_SafeModel):
    """/api/twin/scenario.run 的返回：各场景通过率。"""

    pass_rate: Optional[float] = None
    rates: Dict[str, float] = Field(default_factory=dict)


class CalibrateResponse(_SafeModel):
    """/api/twin/calibrate 的返回：前后参数对比。"""

    base: Dict[str, Any] = Field(default_factory=dict)
    last: Dict[str, Any] = Field(default_factory=dict)
    new: Dict[str, Any] = Field(default_factory=dict)


class ReplayResponse(_SafeModel):
    """/api/twin/replay 的返回：一次决策回放。"""

    decision_id: Optional[str] = None
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    # Port profile
    "OpsKPIs",
    "EnergySnapshot",
    "PortProfile",
    "BootstrapResponse",
    "coerce_port_profile",
    # Fidelity
    "FidelityGroup",
    "FidelityRadar",
    "ParamChange",
    "FidelityPayload",
    "FidelityResponse",
    # Interactions
    "ScenarioRunResponse",
    "CalibrateResponse",
    "ReplayResponse",
]

