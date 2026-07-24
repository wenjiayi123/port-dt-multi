# ============================================
# app/services/scenario_lib.py
# --------------------------------------------
# 场景库 + P50/P90 生成器（对标落地）
#
# 设计目标：
# 1) 提供“台风/高温/密集靠泊/孤网/N-1/应急限电”等标准剧本；
# 2) 将剧本转换为 ForecastService 可消费的 drivers（workload_boost 规则）；
# 3) 生成 P50/P90（在不改变接口的情况下，给前端区间带）；
# 4) 与真实港口落地：只需把剧本触发器对接 TOS/天气/电网事件即可，
#    本文件对外统一输出 drivers，ForecastService 不用改。
# ============================================

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------
# 小工具
# -----------------------
def _to_dt(s: str) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def _seed(*xs: Any) -> int:
    s = "|".join(str(x) for x in xs)
    return abs(hash(s)) % (2**31)


# -----------------------
# 场景定义
# -----------------------
@dataclass
class Scenario:
    key: str                       # 场景键：typhoon / heatwave / dense_berthing / islanded / n_minus_1 / emergency_curtail
    name: str                      # 中文名
    desc: str                      # 备注
    windows: List[Tuple[str,str]]  # 生效时间窗口列表 [(start_iso,end_iso), ...]
    # 强度参数（0~1，具体意义见各场景策略）
    intensity: float = 0.5
    # 额外策略（可选）
    demand_cap_adjust_pct: float = 0.0   # 调整需量上限（%），负数=更严格
    crane_avail_drop_pct: float = 0.0    # 岸桥可用率下降（%）
    agv_charge_throttle_pct: float = 0.0 # AGV充电限流（%）
    chiller_setpoint_offset_c: float = 0.0 # 冷站设定点偏移（°C）
    lighting_curtail_pct: float = 0.0    # 照明压降（%）
    islanded: bool = False               # 孤网/微电网模式
    n_minus_1: bool = False              # N-1 约束
    emergency: bool = False              # 紧急限电

    meta: Dict[str, Any] = field(default_factory=dict)


class ScenarioLib:
    """
    场景库与计算器：
    - presets(): 返回场景模板（给 UI 列表/默认参数）
    - build(key, start, end, intensity): 实例化一个场景（填充窗口）
    - to_drivers(scn): -> {"workload_boost":[{start,end,ratio},...], "meta":{...}}
    - apply_on_series(base, scn): 返回 {"p50":[...], "p90":[...], "notes":...}
    """

    # -----------------------
    # 场景模板
    # -----------------------
    def presets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "typhoon": {
                "name": "台风应急",
                "desc": "岸桥/场桥作业受阻、部分停机；AGV 降速；照明/冷站进入保底；可启用孤网策略",
                "params": {"crane_avail_drop_pct": 0.4, "agv_charge_throttle_pct": 0.5, "lighting_curtail_pct": 0.2, "chiller_setpoint_offset_c": +1.0, "demand_cap_adjust_pct": -0.15, "islanded": False}
            },
            "heatwave": {
                "name": "极端高温",
                "desc": "冷站负荷上升；非关键照明压降；充电错峰；需量更严格",
                "params": {"chiller_setpoint_offset_c": -0.5, "lighting_curtail_pct": 0.1, "agv_charge_throttle_pct": 0.3, "demand_cap_adjust_pct": -0.1}
            },
            "dense_berthing": {
                "name": "密集靠泊",
                "desc": "多个泊位同时作业，岸桥数提升，AGV/场桥任务集中",
                "params": {"crane_avail_drop_pct": -0.2}  # 负数=提升可用率
            },
            "islanded": {
                "name": "孤网/N-1",
                "desc": "按微电网/孤网运行：需量严格、储能参与、非关键负荷抑制",
                "params": {"islanded": True, "demand_cap_adjust_pct": -0.25, "lighting_curtail_pct": 0.15, "agv_charge_throttle_pct": 0.4}
            },
            "n_minus_1": {
                "name": "N-1 约束",
                "desc": "主变或母线 N-1 退化能力：需量上限临时下调",
                "params": {"n_minus_1": True, "demand_cap_adjust_pct": -0.2}
            },
            "emergency": {
                "name": "应急限电",
                "desc": "根据电网通知短时限电：快速压降，策略保底",
                "params": {"emergency": True, "demand_cap_adjust_pct": -0.3, "lighting_curtail_pct": 0.3, "agv_charge_throttle_pct": 0.6}
            },
        }

    # -----------------------
    # 实例化场景（填充窗口）
    # -----------------------
    def build(self, key: str, start: str, end: str, intensity: float = 0.5) -> Scenario:
        pre = self.presets().get(key)
        if not pre:
            raise ValueError(f"unknown scenario key: {key}")
        sdt, edt = _to_dt(start), _to_dt(end)
        params = pre["params"].copy()
        return Scenario(
            key=key, name=pre["name"], desc=pre["desc"],
            windows=[(_iso(sdt), _iso(edt))],
            intensity=max(0.0, min(1.0, float(intensity))),
            **params
        )

    # -----------------------
    # 转换为 Forecast 驱动（workload_boost）
    # -----------------------
    def to_drivers(self, scn: Scenario, base_drivers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        boosts: List[Dict[str, Any]] = []
        base = (base_drivers or {}).get("workload_boost") or []

        # 先把已有 boost 放进来
        boosts.extend(base)

        # 基于场景生成额外 boost（ratio>1 增负荷，<1 降负荷）
        for s, e in scn.windows:
            sdt, edt = _to_dt(s), _to_dt(e)
            dur_h = max(0.25, (edt - sdt).total_seconds() / 3600.0)
            if scn.key == "dense_berthing":
                # 负荷抬升：1.05 ~ 1.25（随强度/时长）
                ratio = 1.05 + 0.2 * scn.intensity * min(1.0, dur_h / 12.0)
                boosts.append({"start": s, "end": e, "ratio": round(ratio, 3)})
            elif scn.key in ("typhoon", "emergency", "islanded", "n_minus_1"):
                # 负荷压降：0.75 ~ 0.95（强度高 -> 更低）
                ratio = 0.95 - 0.2 * scn.intensity
                # 应急可更狠
                if scn.key == "emergency":
                    ratio -= 0.05
                boosts.append({"start": s, "end": e, "ratio": round(max(0.6, ratio), 3)})
            elif scn.key == "heatwave":
                # 高温：冷站抬升 1.05 ~ 1.15
                ratio = 1.05 + 0.1 * scn.intensity
                boosts.append({"start": s, "end": e, "ratio": round(ratio, 3)})

        # 相邻合并（参考 schedule_sources 的逻辑）
        boosts = self._merge_boosts(boosts)

        out = {"workload_boost": boosts, "meta": {"scenario": scn.key, "name": scn.name}}
        # 额外元数据：传给 Twin 可选用（未来扩展）
        out["meta"].update({
            "demand_cap_adjust_pct": scn.demand_cap_adjust_pct,
            "islanded": scn.islanded,
            "n_minus_1": scn.n_minus_1,
            "emergency": scn.emergency,
            "intensity": scn.intensity,
        })
        return out

    # -----------------------
    # 将场景作用到一条基线序列：返回 p50/p90
    # base_points: [{"ts": "...Z", "kW": 12.3}, ...]
    # -----------------------
    def apply_on_series(
        self,
        base_points: List[Dict[str, Any]],
        scn: Scenario,
        residual_sigma_kw: float = 1.0,
        p90_side: str = "upper",  # "upper" or "two-sided"
    ) -> Dict[str, Any]:
        if not base_points:
            return {"p50": [], "p90": [], "notes": "empty series"}

        rnd = random.Random(_seed("scn", scn.key, scn.intensity, base_points[0].get("ts")))
        pts = list(base_points)

        # 1) 先按 drivers（ratio）调整，得到 p50
        drivers = self.to_drivers(scn)
        p50 = self._apply_boosts(pts, drivers.get("workload_boost") or [])

        # 2) 注入场景噪声（不同场景噪声幅度不同）
        noise_scale = {
            "typhoon": 1.8,
            "emergency": 1.4,
            "islanded": 1.2,
            "n_minus_1": 1.1,
            "dense_berthing": 1.0,
            "heatwave": 0.9,
        }.get(scn.key, 1.0)
        p50 = [max(0.0, v + rnd.gauss(0.0, residual_sigma_kw * 0.15 * noise_scale)) for v in p50]

        # 3) 生成 p90（右侧分位或双侧）
        if p90_side == "upper":
            p90 = [max(0.0, v + 1.28 * residual_sigma_kw * noise_scale) for v in p50]
        else:
            # 双侧：上 +1.28σ, 下 -1.28σ（前端如需可传回上下界）
            p90 = [
                {
                    "upper": max(0.0, v + 1.28 * residual_sigma_kw * noise_scale),
                    "lower": max(0.0, v - 1.28 * residual_sigma_kw * noise_scale),
                }
                for v in p50
            ]

        return {"p50": self._zip_series(pts, p50), "p90": self._zip_series(pts, p90), "notes": drivers.get("meta", {})}

    # -----------------------
    # 内部：应用 boosts 到基线
    # -----------------------
    def _apply_boosts(self, base: List[Dict[str, Any]], boosts: List[Dict[str, Any]]) -> List[float]:
        if not base:
            return []
        out = []
        for p in base:
            ts = _to_dt(p.get("ts", ""))
            v = float(p.get("kW", 0.0))
            r = 1.0
            for b in boosts:
                try:
                    s, e = _to_dt(b["start"]), _to_dt(b["end"])
                    if s <= ts <= e:
                        r *= float(b.get("ratio", 1.0))
                except Exception:
                    continue
            out.append(max(0.0, v * r))
        return out

    # -----------------------
    # 内部：相邻合并
    # -----------------------
    def _merge_boosts(self, boosts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not boosts:
            return []
        arr = sorted(boosts, key=lambda b: b["start"])
        out: List[Dict[str, Any]] = []
        cur = arr[0].copy()
        for b in arr[1:]:
            try:
                cs, ce, cr = _to_dt(cur["start"]), _to_dt(cur["end"]), float(cur["ratio"])
                bs, be, br = _to_dt(b["start"]), _to_dt(b["end"]), float(b["ratio"])
                if bs <= ce + timedelta(minutes=5) and abs(br - cr) <= 0.04:
                    cur["end"] = _iso(max(ce, be))
                    cur["ratio"] = round((cr + br) / 2.0, 3)
                else:
                    out.append(cur)
                    cur = b.copy()
            except Exception:
                out.append(cur)
                cur = b.copy()
        out.append(cur)
        return out

    # -----------------------
    # 内部：把数值列表与时间戳重新绑定
    # -----------------------
    def _zip_series(self, base: List[Dict[str, Any]], vals: Any) -> List[Dict[str, Any]]:
        seq: List[Dict[str, Any]] = []
        if isinstance(vals, list) and vals and isinstance(vals[0], dict):
            # 双侧区间的场景
            for i, p in enumerate(base):
                seq.append({"ts": p.get("ts"), **vals[i]})
            return seq
        # 单值场景
        for i, p in enumerate(base):
            seq.append({"ts": p.get("ts"), "kW": float(vals[i])})
        return seq
