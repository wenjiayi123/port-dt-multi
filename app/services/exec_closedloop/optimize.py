# ============================================
# app/services/optimize.py
# --------------------------------------------
# MILP/CP 与 RL 协同（轻量版，无外部依赖）：
# - 给出“可行域”（各设备/动作的 kW 上下界）
# - 给出“启发式初值”（供 RL 滚动细化；失败回退）
#
# 大白话：
#   - Provides a bounded fallback when a production LP/MILP solver is unavailable:
#     1) 把“馈线限额 + 设备额定 + SOC + 充放电效率 + N-1 裕度”等硬条件，收敛成每个动作的上下界。
#     2) 用“水位法”把“尖峰”削平：先用储能放电顶峰，再把 AGV 充电移到低谷，再对照明/非关键负载做减载。
#   - 输出“可解释说明”，方便前端“策略贡献榜/为什么这么排”的展示。
# ============================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math


# ---------- 数据模型 ----------
@dataclass
class FeasibleBound:
    """某设备/某动作在当前窗口内的 kW 上下界（常量界，便于前端绘制/供RL采样）"""
    asset: str
    cmd: str                 # "charge"|"discharge"|"reduce"|"idle"|"setpoint" 等
    min_kw: float
    max_kw: float
    notes: str = ""

@dataclass
class InitialAction:
    """优化器给出的初值（窗口内常量动作）；RL 可在其上微调"""
    asset: str
    cmd: str
    kW: float
    reason: str = ""         # 为何给出这个数（可解释输出）


# ---------- 工具函数 ----------
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _to_f(x: Any, d=0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return d


# ============================================================
# 主类：HybridOptimizer（轻量启发式优化协同器）
# ============================================================
class HybridOptimizer:
    """
    主入口：
      - compute_feasible_region(context, horizon_min, step_min) -> List[FeasibleBound]
      - propose_initial_plan(context, horizon_min, step_min) -> List[InitialAction]

    需要的 context（尽量宽松，可缺省，有则用，没有就给稳健默认）：
      {
        "agg_forecast_kw": [ ... ],       # 聚合负荷预测（kW，按 step_min 采样）
        "feeder_limit_kw": 2800.0,        # 馈线限额（已扣除 N-1 裕度，或者另外给 nminus1_reserve_kw）
        "nminus1_reserve_kw": 200.0,      # 可选，若给了再扣一次
        "storage": {                      # 可选：站内储能（BESS）
          "id": "bess-01",
          "capacity_kwh": 2500.0,
          "soc": 0.55,
          "p_charge_max_kw": 800.0,
          "p_discharge_max_kw": 800.0,
          "efficiency": 0.95,
          "soc_min": 0.10,
          "soc_max": 0.90
        },
        "agv_list": [                     # 可选：AGV/无人集卡（充电可移峰）
          {"id":"agv-01","soc":0.5,"need_kwh":30.0,"p_charge_max_kw":120.0,"p_discharge_max_kw":0.0},
          ...
        ],
        "lighting_zones": [               # 可选：照明分区（可减载）
          {"id":"yard-z1","max_reduce_kw":25.0,"min_duty":0.6},
          ...
        ],
        "device_caps": {                  # 可选：设备能力（等价于 asset_caps.json 中的 assets 子表）
          "qc-01": {"device":{"rated_kw":80.0,"amp_max":220.0,"temp_max_c":85.0}},
          "agv-01": {"device":{"rated_kw":35.0},"battery":{"charge_kw_max":120.0,"discharge_kw_max":60.0,"soc_min":0.1,"soc_max":0.9}},
          ...
        }
      }
    """

    def __init__(self, telemetry=None, storage=None, caps_path: Optional[str] = None):
        self.telemetry = telemetry           # 可用来推断资产列表/最近功率
        self.storage = storage               # 可选：写优化证据包
        self.caps_path = caps_path or "data/objects/config/asset_caps.json"
        self.caps = self._load_caps()        # 若没有该文件，会返回空 dict；算法会用 context 内传入的 device_caps 兜底

    # ------------------ 对外：可行域 ------------------
    def compute_feasible_region(
        self,
        context: Dict[str, Any],
        horizon_min: int = 60,
        step_min: int = 5
    ) -> List[FeasibleBound]:
        """
        整合“馈线限额 + 设备额定 + SOC + 充放电效率 + 照明最小占空”等，给出每类动作的 kW 上下界。
        这里给“常量上下界”（窗口内不随时间变），让 RL 先在这个盒子里采样/细化。
        """
        bounds: List[FeasibleBound] = []
        dev_caps = (context.get("device_caps") or self.caps.get("assets") or {})

        # 1) 储能
        bess = context.get("storage") or {}
        if bess:
            soc = _to_f(bess.get("soc"), 0.5)
            soc_min = _to_f(bess.get("soc_min"), 0.1)
            soc_max = _to_f(bess.get("soc_max"), 0.9)
            cap = _to_f(bess.get("capacity_kwh"), 0.0)
            p_ch = _to_f(bess.get("p_charge_max_kw"), 0.0)
            p_dis = _to_f(bess.get("p_discharge_max_kw"), 0.0)
            # SOC 限界换算为功率边界（粗略，按一个 step_min，避免一次动作把 SOC 撞边）
            step_h = max(1, step_min) / 60.0
            headroom_ch_kwh = max(0.0, (soc_max - soc) * cap)
            headroom_dis_kwh = max(0.0, (soc - soc_min) * cap)
            max_kW_charge_by_soc = headroom_ch_kwh / step_h if step_h > 0 else p_ch
            max_kW_dis_by_soc = headroom_dis_kwh / step_h if step_h > 0 else p_dis
            bounds.append(FeasibleBound(asset=bess.get("id","bess-01"), cmd="charge",
                                        min_kw=0.0, max_kw=min(p_ch, max_kW_charge_by_soc),
                                        notes="由SOC与p_charge_max共同限定"))
            bounds.append(FeasibleBound(asset=bess.get("id","bess-01"), cmd="discharge",
                                        min_kw=0.0, max_kw=min(p_dis, max_kW_dis_by_soc),
                                        notes="由SOC与p_discharge_max共同限定"))

        # 2) AGV 充电
        for agv in (context.get("agv_list") or []):
            pid = agv.get("id","agv")
            pmax = _to_f(agv.get("p_charge_max_kw"), 0.0)
            bounds.append(FeasibleBound(asset=pid, cmd="charge", min_kw=0.0, max_kw=pmax,
                                        notes="AGV 充电上界（额定）"))

        # 3) 照明分区减载
        for z in (context.get("lighting_zones") or []):
            zid = z.get("id","zone")
            rmax = _to_f(z.get("max_reduce_kw"), 0.0)
            bounds.append(FeasibleBound(asset=zid, cmd="reduce", min_kw=0.0, max_kw=rmax,
                                        notes=f"照明可减载上限（min_duty={z.get('min_duty',0.6)}）"))

        # 4) 其它设备（若 context 没给，尝试按资产能力给一个保守上界）
        for aid, cap in dev_caps.items():
            dev = (cap or {}).get("device") or {}
            rated = _to_f(dev.get("rated_kw"), 0.0)
            if rated > 0:
                bounds.append(FeasibleBound(asset=aid, cmd="reduce", min_kw=0.0, max_kw=max(0.0, 0.3*rated),
                                            notes="非关键负载保守可降 30% 作为上界（按现场调整）"))

        return bounds

    # ------------------ 对外：初始方案 ------------------
    def propose_initial_plan(
        self,
        context: Dict[str, Any],
        horizon_min: int = 60,
        step_min: int = 5
    ) -> Dict[str, Any]:
        """
        基于“水位法”的初始方案：
          1) 若有储能：优先在尖峰放电、在低谷充电。
          2) 若仍超：在峰段推进“照明减载”（不影响作业SLA的非关键负载）。
          3) 若仍超：把 AGV 充电移到低价/低负荷时段。
        返回：
          {
            "initial_actions": [InitialAction...（窗口常量动作）],
            "explain": [ ... 步骤解释 ... ],
            "residual_peak_kw": <削峰后峰值>,
            "feasible_region": [FeasibleBound...（便于RL采样）]
          }
        """
        explain: List[str] = []
        bounds = self.compute_feasible_region(context, horizon_min, step_min)

        series = list(context.get("agg_forecast_kw") or [])
        if not series:
            # 没有聚合预测时，给一个平滑基线
            L = max(1, int(horizon_min / max(1, step_min)))
            base = 1800.0
            series = [base + 80.0*math.sin(i/12.0) for i in range(L)]

        limit = _to_f(context.get("feeder_limit_kw"), 2800.0)
        # 若给了 N-1，则再扣一次保障裕度
        limit -= _to_f(context.get("nminus1_reserve_kw"), 0.0)
        limit = max(0.0, limit)

        step_h = max(1, step_min) / 60.0

        L = len(series)
        load = series[:]  # 拷贝，不破坏外部
        peak0 = max(load) if load else 0.0

        # ---- 1) 储能削峰 ----
        bess = context.get("storage") or {}
        if bess:
            plan_bess = self._waterfill_bess(load, limit, bess, step_h)
            if plan_bess:
                load = plan_bess["net_load"]
                explain.append(f"[BESS] 尖峰放电 {round(plan_bess['discharge_kwh'],2)} kWh，低谷充电 {round(plan_bess['charge_kwh'],2)} kWh")
        peak1 = max(load) if load else 0.0

        # ---- 2) 照明减载 ----
        light_actions: List[InitialAction] = []
        if peak1 > limit and context.get("lighting_zones"):
            # 计算整体还需削减的峰值余量
            need_cut_kw = peak1 - limit
            # 每个分区取上界的 60% 作为初值
            for z in context["lighting_zones"]:
                if need_cut_kw <= 0:
                    break
                rmax = _to_f(z.get("max_reduce_kw"), 0.0)
                ruse = min(need_cut_kw, 0.6 * rmax)
                if ruse > 0:
                    light_actions.append(InitialAction(asset=z["id"], cmd="reduce", kW=round(ruse,3),
                                                       reason="峰段减载初值（60% 上界）"))
                    need_cut_kw -= ruse
            # 更新净负荷
            if light_actions:
                for i in range(L):
                    load[i] = max(0.0, load[i] - sum(a.kW for a in light_actions))
        peak2 = max(load) if load else 0.0

        # ---- 3) AGV 充电移峰 ----
        agv_actions: List[InitialAction] = []
        if peak2 > limit and context.get("agv_list"):
            # 将 AGV 充电尽量移到 <limit-安全裕度> 的低谷时段；窗口内常量近似（初值）
            margin = max(0.0, limit - min(load))
            if margin > 0:
                # 取每台 AGV 上界的 50% 作为初值（避免一下把低谷填满）
                for agv in context["agv_list"]:
                    pmax = _to_f(agv.get("p_charge_max_kw"), 0.0)
                    puse = 0.5 * pmax
                    if puse > 0:
                        agv_actions.append(InitialAction(asset=agv["id"], cmd="charge", kW=round(puse,3),
                                                         reason="低谷充电初值（50% 上界）"))
                # 更新低谷抬升：简化为常量叠加
                for i in range(L):
                    load[i] = load[i] + sum(a.kW for a in agv_actions)
        peak3 = max(load) if load else 0.0

        # 汇总初值（窗口常量）
        actions: List[InitialAction] = []
        # BESS：窗口常量（若实际需要时序曲线，可在下一版扩展返回 time-series）
        if bess:
            # 常量近似：取峰段放电能力的 60% 作为初值
            pdis = _to_f(bess.get("p_discharge_max_kw"), 0.0)
            if pdis > 0:
                actions.append(InitialAction(asset=bess.get("id","bess-01"), cmd="discharge", kW=round(0.6*pdis,3),
                                             reason="按峰段可用放电功率的 60% 作为窗口常量初值"))
        actions.extend(light_actions)
        actions.extend(agv_actions)

        return {
            "initial_actions": [asdict(a) for a in actions],
            "feasible_region": [asdict(b) for b in bounds],
            "residual_peak_kw": round(peak3, 3),
            "baseline_peak_kw": round(peak0, 3),
            "limit_kw": round(limit, 3),
            "explain": explain
        }

    # ------------------ 内部：BESS 水位法削峰/填谷 ------------------
    def _waterfill_bess(self, load: List[float], limit: float, bess: Dict[str, Any], step_h: float) -> Optional[Dict[str, Any]]:
        """
        经典“水位法”：尽量把 >limit 的尖峰用放电压下来，把 <(limit-裕度) 的低谷用充电抬起来。
        简化：不做复杂时序优化，按当前窗口一次扫描。
        返回：净负荷曲线与充/放电 kWh 统计（用于 explain）
        """
        if not load:
            return None
        net = load[:]
        cap = _to_f(bess.get("capacity_kwh"), 0.0)
        soc = _to_f(bess.get("soc"), 0.5)
        soc_min = _to_f(bess.get("soc_min"), 0.1)
        soc_max = _to_f(bess.get("soc_max"), 0.9)
        pch = _to_f(bess.get("p_charge_max_kw"), 0.0)
        pdis = _to_f(bess.get("p_discharge_max_kw"), 0.0)
        eff = max(0.5, min(1.0, _to_f(bess.get("efficiency"), 0.95)))

        energy = soc * cap
        ch_kwh = 0.0
        dis_kwh = 0.0

        # 1) 放电降峰
        for i in range(len(net)):
            if net[i] > limit and pdis > 0 and energy > soc_min * cap:
                need = net[i] - limit
                use = min(need, pdis)
                # 放电能量（kWh）
                e = use * step_h
                # 受 SOC 下限限制
                e = min(e, max(0.0, energy - soc_min*cap))
                use = e / step_h if step_h > 0 else 0.0
                net[i] -= use
                energy -= e
                dis_kwh += e

        # 2) 充电填谷（尽量在低谷把能量补回，按效率折算）
        target = max(limit - 0.15*limit, 0.0)  # 低谷目标：limit 的 85%
        for i in range(len(net)):
            if net[i] < target and pch > 0 and energy < soc_max * cap:
                gap = target - net[i]
                use = min(gap, pch)
                e = use * step_h * eff  # 充电进入电池的能量按效率计
                e = min(e, max(0.0, soc_max*cap - energy))
                use = e / (step_h * eff) if step_h > 0 else 0.0
                net[i] += use
                energy += e
                ch_kwh += e

        # 更新 SOC（不回写到 context，这里只是估计）
        new_soc = energy / cap if cap > 0 else soc
        return {"net_load": net, "soc": new_soc, "charge_kwh": ch_kwh, "discharge_kwh": dis_kwh}

    # ------------------ 配置加载 ------------------
    def _load_caps(self) -> Dict[str, Any]:
        """
        读取 data/objects/config/asset_caps.json（若不存在返回空 dict）。
        与 rl_safety 的口径一致：现场替换该文件即可。
        """
        p = Path(self.caps_path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
