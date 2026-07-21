# ============================================
# app/services/sim_scenarios.py
# --------------------------------------------
# 场景库（模板） + 公共参数解析
#
# 目标：
# - 为 TwinService 提供标准化的场景口径（高温/台风/密集靠泊/孤网）
# - 每个场景包含：各设备系数(multiplier) / TOU 响应倾向 / 不确定度带宽 / 园区级限电（可选）
# - 该文件与 TwinService 解耦；TwinService 可选择性 import 使用（若不存在也能跑）
# ============================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

AssetType = str  # "qc" | "yc" | "agv" | "cs" | "wh" | "yard" | "ps" | "misc"

@dataclass(frozen=True)
class Scenario:
    name: str
    # 各设备类型的功率倍数（对 p50 作用）
    multipliers: Dict[AssetType, float]
    # TOU 响应偏置（对峰/谷段响应强度的额外加权；1=不变，<1=更强抑制，>1=更弱）
    tou_bias: Dict[AssetType, Dict[str, float]]  # {asset: {"peak":0.9,"flat":1.0,"valley":1.05}}
    # 置信区间带宽（p90/p10）
    band_width: Dict[AssetType, float]           # 0.1 = ±10%
    # 园区级（聚合层）上限，单位 kW（可选；TwinService 单点不强制）
    grid_limit_kw: Optional[float] = None

    def coeff(self, typ: AssetType) -> float:
        return self.multipliers.get(typ, 1.0)

    def width(self, typ: AssetType) -> float:
        return self.band_width.get(typ, 0.1)

    def tou_weight(self, typ: AssetType, tier: str) -> float:
        return self.tou_bias.get(typ, {}).get(tier, 1.0)

# ---------- 场景模板 ----------

BASELINE = Scenario(
    name="baseline",
    multipliers={},
    tou_bias={},
    band_width={"qc":0.12,"yc":0.12,"agv":0.10,"cs":0.15,"wh":0.10,"yard":0.10,"ps":0.08,"misc":0.10},
    grid_limit_kw=None
)

HEATWAVE = Scenario(
    name="heatwave",
    multipliers={"wh":1.12,"yard":1.08,"ps":1.10,"qc":1.03,"yc":1.03,"agv":1.02,"cs":1.02},
    tou_bias={"wh":{"peak":0.95,"flat":1.0,"valley":1.02},"yard":{"peak":0.95,"flat":1.0,"valley":1.02}},
    band_width={"qc":0.13,"yc":0.13,"agv":0.11,"cs":0.16,"wh":0.12,"yard":0.11,"ps":0.09,"misc":0.11},
    grid_limit_kw=None
)

TYPHOON = Scenario(
    name="typhoon",
    multipliers={"qc":0.70,"yc":0.75,"agv":0.80,"cs":0.85,"wh":1.05,"ps":1.06,"yard":0.90},
    tou_bias={"qc":{"peak":0.92,"flat":1.0,"valley":1.02},"yc":{"peak":0.92,"flat":1.0,"valley":1.02}},
    band_width={"qc":0.15,"yc":0.15,"agv":0.12,"cs":0.15,"wh":0.12,"yard":0.12,"ps":0.10,"misc":0.12},
    grid_limit_kw=None
)

DENSE_BERTHING = Scenario(
    name="dense_berthing",
    multipliers={"qc":1.15,"yc":1.12,"agv":1.10,"cs":1.05},
    tou_bias={"agv":{"peak":0.90,"flat":1.0,"valley":1.05},"cs":{"peak":0.85,"flat":0.95,"valley":1.10}},
    band_width={"qc":0.13,"yc":0.12,"agv":0.11,"cs":0.15,"wh":0.10,"yard":0.10,"ps":0.08,"misc":0.10},
    grid_limit_kw=None
)

ISLANDED = Scenario(
    name="islanded",
    multipliers={"qc":0.85,"yc":0.85,"agv":0.88,"cs":0.9,"wh":0.9,"yard":0.88,"ps":0.9},
    tou_bias={"wh":{"peak":0.9,"flat":0.95,"valley":1.05},"yard":{"peak":0.9,"flat":0.95,"valley":1.05}},
    band_width={"qc":0.12,"yc":0.12,"agv":0.11,"cs":0.14,"wh":0.11,"yard":0.11,"ps":0.09,"misc":0.10},
    grid_limit_kw=1500.0  # 演示口径；真实应改为园区 N-1/并网容量
)

_SCENES = {
    "baseline": BASELINE,
    "heatwave": HEATWAVE,
    "typhoon": TYPHOON,
    "dense_berthing": DENSE_BERTHING,
    "islanded": ISLANDED,
}

def load_scenario(name: str) -> Scenario:
    return _SCENES.get((name or "baseline").lower(), BASELINE)

def list_scenarios():
    return list(_SCENES.keys())
