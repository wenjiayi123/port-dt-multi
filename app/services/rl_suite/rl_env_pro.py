# ============================================
# app/services/rl_env_pro.py
# --------------------------------------------
# Pro 级端到端 RL 环境（多资产 + 时序约束 + 多目标/多约束）
# - 观测含：聚合负荷/岸电/电价/碳因子/环境温度/SoC/SLA 目标等随时间变化的信号
# - 动作含：BESS 充放（效率/爬坡/SOC边界/寿命成本）、AGV 车队充电分配、照明分区减载、冷站设定点偏置
# - 约束：馈线限额 + N-1、设备功率上限、AGV 电量需求、冷站舒适边界、温度软约束、SLA 负反馈
# - 奖励：来自 LagrangeManager（reward = - total_score），天然支持成本/碳/能耗 + 约束惩罚；若缺失则使用近似成本函数
# - 可行域：HybridOptimizer 提供每步 bounds（若不可用自动降级）
# - 安全守护：RLSafetyGuard 校验/裁剪（若不可用自动降级）
# - 数据集：rollouts_pro(...) 生成更贴近真实港口的离线轨迹
# ============================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json, math, random

# ---------- 小工具 ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _to_f(x: Any, d=0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v): return v
    except Exception:
        pass
    return d

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ---------- Pro 环境 ----------
class PortEnergyEnvPro:
    """
    核心思想：
      - 把港口能流关键环节（BESS / AGV / 照明 / 冷站 / 岸电 / 基线）拼装成时序环境；
      - 每步长度 step_min，窗口 horizon_min = L * step_min；我们按“逐步推进”而不是整窗常量近似；
      - 动作先过“可行域 + 安全守护”，再评估奖励 & 更新状态。
    观测 observation（dict，随时间滚动）：
      {
        "t": 当前步序号（从 0 开始）,
        "step_min": 步长,
        "price": 本步电价,
        "grid_carbon": 本步电网碳因子（kg/kWh）,
        "ambient_c": 环境温度,
        "soc": BESS SoC,
        "agv_unmet_kwh": 车队剩余需电（总和）,
        "feeder_limit_kw": 馈线限额,
        "nminus1_reserve_kw": N-1 预留,
        "net_load_kw": 上步执行后的净负荷,
      }
    动作 action（list[dict]）：
      - {"asset":"bess-01","cmd":"charge|discharge","kW":...}
      - {"asset":"agv-fleet","cmd":"charge","kW":...}   # 聚合充电，内部按需分配到单车
      - {"asset":"yard-zX","cmd":"reduce","kW":...}     # 照明分区减载
      - {"asset":"chiller-01","cmd":"set_sp_delta","delta_c":...}  # 设定点偏置（±2℃范围）
    """

    def __init__(self, telemetry=None, optimizer=None, lagrange=None, safety=None):
        self.telemetry = telemetry
        # 依赖：允许缺失（降级）
        try:
            self.optimizer = optimizer or __import__("app.services.optimize", fromlist=["HybridOptimizer"]).HybridOptimizer(telemetry=telemetry)
        except Exception:
            self.optimizer = None
        try:
            self.lagrange = lagrange or __import__("app.services.rl_lagrange", fromlist=["LagrangeManager"]).LagrangeManager()
        except Exception:
            self.lagrange = None
        try:
            self.safety = safety or __import__("app.services.rl_safety", fromlist=["RLSafetyGuard"]).RLSafetyGuard(telemetry=telemetry)
        except Exception:
            self.safety = None

        # 运行状态
        self._t = 0
        self._L = 0
        self._step_min = 5
        self._price: List[float] = []
        self._grid_carbon: List[float] = []
        self._ambient_c: List[float] = []
        self._base_kw: List[float] = []   # 不可控基线（含岸电/岸桥/场桥等聚合）
        self._net_kw_prev: float = 0.0

        # 馈线/N-1
        self._limit_kw = 2800.0
        self._reserve_kw = 0.0

        # BESS
        self._bess_id = "bess-01"
        self._cap_kwh = 2500.0
        self._soc = 0.55
        self._pch_max = 800.0
        self._pdis_max = 800.0
        self._eta_ch = 0.96
        self._eta_dis = 0.96
        self._ramp_kw_per_step = 400.0   # 爬坡限制（每步最大变化）
        self._deg_cost_per_kwh = 0.02    # 寿命成本（本币/kWh-throughput）
        self._soc_min = 0.10
        self._soc_max = 0.90
        self._bess_last_p = 0.0

        # AGV 车队
        self._agv_fleet: List[Dict[str, Any]] = []  # 每辆 {"id","soc","need_kwh","p_charge_max_kw"}
        self._agv_total_need_kwh = 0.0

        # 照明分区
        self._zones: List[Dict[str, Any]] = []  # {"id","max_reduce_kw","min_duty"}

        # 冷站
        self._chiller_id = "chiller-01"
        self._chiller_base_kw = 300.0
        self._sp_delta_acc = 0.0  # 累积设定点偏置
        self._sp_delta_limit = 2.0  # ±2℃
        # COP 简模：COP = a - b*(Ta-25)，Ta∈[10,45] 截断；偏置 +1℃ 可降冷量需求约 3%
        self._cop_a = 5.5
        self._cop_b = 0.06

    # ------------------------ reset ------------------------
    def reset(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        self._step_min = int(ctx.get("step_min", 5))
        horizon_min = int(ctx.get("horizon_min", 120))
        self._L = max(1, horizon_min // max(1, self._step_min))

        # 时间序列信号
        def mk_series(key, default, fgen=None):
            s = ctx.get(key)
            if s and isinstance(s, list) and len(s) >= self._L:
                return list(s[:self._L])
            if fgen:
                return [fgen(i) for i in range(self._L)]
            return [default for _ in range(self._L)]

        avg = _to_f(ctx.get("baseline_kw", 2000.0), 2000.0)
        self._base_kw = mk_series("base_load_kw", avg, fgen=lambda i: avg + 0.12*avg*math.sin(i/12.0))
        price0 = _to_f(ctx.get("price_per_kwh", 1.1), 1.1)
        self._price = mk_series("price_curve", price0)
        carbon0 = _to_f(ctx.get("grid_carbon_kg_per_kwh", 0.55), 0.55)
        self._grid_carbon = mk_series("grid_carbon_curve", carbon0)
        amb0 = _to_f(ctx.get("ambient_c", 30.0), 30.0)
        self._ambient_c = mk_series("ambient_temp_curve", amb0)

        # 馈线/N-1
        self._limit_kw = _to_f(ctx.get("feeder_limit_kw"), 2800.0)
        self._reserve_kw = _to_f(ctx.get("nminus1_reserve_kw"), 0.0)

        # BESS
        b = ctx.get("storage") or {}
        self._bess_id = b.get("id", "bess-01")
        self._cap_kwh = _to_f(b.get("capacity_kwh"), 2500.0)
        self._soc = _to_f(b.get("soc"), 0.55)
        self._pch_max = _to_f(b.get("p_charge_max_kw"), 800.0)
        self._pdis_max = _to_f(b.get("p_discharge_max_kw"), 800.0)
        self._eta_ch = _to_f(b.get("eta_charge"), 0.96)
        self._eta_dis = _to_f(b.get("eta_discharge"), 0.96)
        self._ramp_kw_per_step = _to_f(b.get("ramp_kw_per_step"), 400.0)
        self._deg_cost_per_kwh = _to_f(b.get("degradation_cost_per_kwh"), 0.02)
        self._soc_min = _to_f(b.get("soc_min"), 0.10)
        self._soc_max = _to_f(b.get("soc_max"), 0.90)
        self._bess_last_p = 0.0

        # AGV 车队
        self._agv_fleet = []
        for agv in (ctx.get("agv_list") or [{"id":"agv-01","soc":0.5,"need_kwh":30.0,"p_charge_max_kw":120.0}]):
            self._agv_fleet.append({
                "id": agv.get("id","agv"),
                "soc": _to_f(agv.get("soc"), 0.5),
                "need_kwh": _to_f(agv.get("need_kwh"), 30.0),
                "p_charge_max_kw": _to_f(agv.get("p_charge_max_kw"), 120.0)
            })
        self._agv_total_need_kwh = sum(a["need_kwh"] for a in self._agv_fleet)

        # 照明
        self._zones = []
        for z in (ctx.get("lighting_zones") or [{"id":"yard-z1","max_reduce_kw":25.0,"min_duty":0.6}]):
            self._zones.append({"id": z.get("id","zone"), "max_reduce_kw": _to_f(z.get("max_reduce_kw"), 25.0), "min_duty": _to_f(z.get("min_duty"), 0.6)})

        # 冷站
        ch = ctx.get("chiller") or {}
        self._chiller_id = ch.get("id","chiller-01")
        self._chiller_base_kw = _to_f(ch.get("base_kw"), 300.0)
        self._sp_delta_acc = 0.0
        self._sp_delta_limit = _to_f(ch.get("sp_delta_limit"), 2.0)
        self._cop_a = _to_f(ch.get("cop_a"), 5.5)
        self._cop_b = _to_f(ch.get("cop_b"), 0.06)

        self._t = 0
        self._net_kw_prev = max(0.0, self._base_kw[0])
        return self._obs()

    # ------------------------ step ------------------------
    def step(self, actions: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        actions = list(actions or [])
        step_h = self._step_min / 60.0
        t = self._t

        # 1) 可行域裁剪（若可用）
        if self.optimizer:
            fr = self.optimizer.compute_feasible_region(self._ctx_for_opt(t), horizon_min=self._step_min, step_min=self._step_min)
            bounds_map = {(b["asset"] if isinstance(b, dict) else b.asset,
                           (b["cmd"] if isinstance(b, dict) else b.cmd)): (float((b["min_kw"] if isinstance(b, dict) else b.min_kw)),
                                                                          float((b["max_kw"] if isinstance(b, dict) else b.max_kw)))
                          for b in (fr if isinstance(fr, list) else [])}
            for a in actions:
                if "kW" in a:
                    lo, hi = bounds_map.get((a.get("asset"), a.get("cmd")), (None, None))
                    if lo is not None and hi is not None:
                        a["kW"] = _clip(_to_f(a.get("kW"), 0.0), lo, hi)

        # 2) 安全守护（若可用）
        guard_info = {"ok": True, "rules": [], "actions_after_shield": actions}
        if self.safety:
            try:
                guard = self.safety.validate_and_shield(
                    {"id": f"pro-step-{t}",
                     "window": self._window_now(),
                     "actions": actions},
                    enforce_guardrails=True, horizon_min=self._step_min, step_min=self._step_min,
                    baseline_agg_kw=[self._base_kw[t]]
                )
                guard_info = guard
                actions = guard.get("actions_after_shield", actions)
            except Exception as e:
                guard_info = {"ok": False, "error": str(e), "rules": [], "actions_after_shield": actions}

        # 3) 计算各资产功率/能量 & 状态更新
        p_bess, e_bess_cost = self._apply_bess(actions, step_h)          # p>0 表示充电（吸收）
        p_agv = self._apply_agv(actions, step_h)                          # 充电功率（吸收）
        p_light = - self._apply_lighting(actions)                          # 减载为负功率
        p_ch = self._apply_chiller(actions, step_h)                        # 冷站功率（吸收）
        base = max(0.0, self._base_kw[t])                                  # 不可控基线

        net = max(0.0, base + p_bess + p_agv + p_light + p_ch)             # 简化：不考虑回馈上网
        self._net_kw_prev = net

        # 4) 奖励（来自 LagrangeManager；若缺失则近似成本）
        if self.lagrange:
            try:
                from app.services.rl_suite.rl_lagrange import PriceSignal, CarbonFactor, ObjectiveConfig
                price = self._price[t] if t < len(self._price) else self._price[-1]
                cf = self._grid_carbon[t] if t < len(self._grid_carbon) else self._grid_carbon[-1]
                ocfg = ObjectiveConfig(
                    w_cost=float(self._ctx_get("w_cost", 1.0)),
                    w_carbon_intensity=float(self._ctx_get("w_carbon_intensity", 0.0)),
                    w_energy=float(self._ctx_get("w_energy", 0.0)),
                    w_delay_min=float(self._ctx_get("w_delay_min", 0.0)),
                )
                self.lagrange.cfg = ocfg
                # 这里构造“窗口=一步”的策略，便于分步训练；吞吐/延迟由 ctx 提供或设 0
                score = self.lagrange.evaluate_strategy(
                    {"id": f"pro-step-{t}", "window": self._window_now(),
                     "actions": [{"asset": self._bess_id, "cmd": "charge" if p_bess>0 else "discharge", "kW": abs(p_bess)}] +
                                ([{"asset":"agv-fleet","cmd":"charge","kW": p_agv}] if p_agv>0 else []) +
                                ([{"asset": z["id"], "cmd":"reduce","kW": -p_light}] if p_light<0 else []) +
                                ([{"asset": self._chiller_id, "cmd":"elec","kW": p_ch}] if p_ch>0 else []),
                     "meta": {"expected_delay_min": float(self._ctx_get("expected_delay_min", 0.0)),
                              "expected_throughput_teu": float(self._ctx_get("expected_throughput_teu", 100))}},
                    context={"baseline_peak_kw": float(self._ctx_get("baseline_peak_kw", base)),
                             "expected_delay_min": float(self._ctx_get("expected_delay_min", 0.0)),
                             "expected_throughput_teu": float(self._ctx_get("expected_throughput_teu", 100))},
                    price=PriceSignal(tou_price_per_kwh=price,
                                      carbon_price_per_kg=float(self._ctx_get("carbon_price_per_kg", 0.0)),
                                      demand_charge_per_kw=float(self._ctx_get("demand_charge_per_kw", 0.0))),
                    carbon_factor=CarbonFactor(kg_per_kwh=cf),
                )
                reward = - float(score["total_score"]) - e_bess_cost  # 加上寿命成本
                constraints = score.get("constraints", {})
                score_obj = score.get("objective", {})
            except Exception:
                reward, constraints, score_obj = self._fallback_cost(net, p_bess, p_agv, p_light, p_ch, t), {}, {}
        else:
            reward, constraints, score_obj = self._fallback_cost(net, p_bess, p_agv, p_light, p_ch, t), {}, {}

        # 5) 推进时间
        self._t = min(self._t + 1, self._L - 1)
        done = (self._t == self._L - 1)

        obs = self._obs()
        info = {
            "p_breakdown_kw": {
                "base": base, "bess": p_bess, "agv": p_agv, "lighting": p_light, "chiller": p_ch, "net": net
            },
            "guard": guard_info,
            "constraints": constraints,
            "objective": score_obj
        }
        return obs, float(reward), bool(done), info

    # ------------------------ 离线轨迹（更真实分布） ------------------------
    def rollouts_pro(self, episodes: int = 5, out_dir: str = "data/objects/rl/datasets") -> List[str]:
        """
        使用 Pro 环境合成离线数据：随机化船期/价差/气温/负荷起伏/AGV 需求，生成 episode 轨迹。
        策略：用优化器的初值作为行为策略（若不可用则用简单启发式）。
        """
        paths: List[str] = []
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for ep in range(episodes):
            ctx = self._random_ctx()
            self.reset(ctx)
            traj = {"created_at": _now_iso(), "episodes": ep, "steps": [], "ctx": ctx}
            done = False
            while not done:
                actions = self._default_policy_step()
                obs, rew, done, info = self.step(actions)
                traj["steps"].append({"t": obs["t"], "obs": obs, "acts": actions, "rew": rew, "info": info})
            fp = f"{out_dir}/episode-pro-{int(datetime.now(timezone.utc).timestamp())}-{ep}.json"
            Path(fp).write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
            paths.append(fp)
        return paths

    # ====================== 内部：资产模型 ======================
    def _apply_bess(self, actions: List[Dict[str, Any]], step_h: float) -> Tuple[float, float]:
        """返回 (p_bess_kw [>0充电], degradation_cost)"""
        # 目标功率（kW，>0 充电，<0 放电）
        p_target = 0.0
        for a in actions:
            if a.get("asset") == self._bess_id:
                if a.get("cmd") == "charge":
                    p_target += max(0.0, _to_f(a.get("kW"), 0.0))
                elif a.get("cmd") == "discharge":
                    p_target -= max(0.0, _to_f(a.get("kW"), 0.0))
        # 爬坡限制
        p_limit_pos = self._pch_max
        p_limit_neg = -self._pdis_max
        p_target = _clip(p_target, p_limit_neg, p_limit_pos)
        p = _clip(p_target, self._bess_last_p - self._ramp_kw_per_step, self._bess_last_p + self._ramp_kw_per_step)

        # SOC 限制（考虑效率）
        if p >= 0:  # 充电
            e_in = p * step_h * self._eta_ch
            soc_new = _clip(self._soc + e_in / max(1e-6, self._cap_kwh), self._soc_min, self._soc_max)
            # 如果撞上限，回推出允许的 p
            delta = (soc_new - self._soc) * self._cap_kwh / max(1e-6, self._eta_ch) / step_h
            p = max(0.0, min(p, delta))
            self._soc = soc_new
            throughput_kwh = e_in
        else:       # 放电
            e_out = abs(p) * step_h
            soc_new = _clip(self._soc - e_out / max(1e-6, self._cap_kwh), self._soc_min, self._soc_max)
            delta = (self._soc - soc_new) * self._cap_kwh / step_h
            p = -max(0.0, min(abs(p), delta))
            self._soc = soc_new
            throughput_kwh = e_out

        self._bess_last_p = p
        # 寿命成本按能量通量计费（可替换为更复杂的 DoD/温度模型）
        degr_cost = self._deg_cost_per_kwh * throughput_kwh
        return (p if p >= 0 else p), degr_cost

    def _apply_agv(self, actions: List[Dict[str, Any]], step_h: float) -> float:
        """AGV 聚合充电功率（kW，>0 吸收），内部按“剩余需电优先”分配到单车，并更新 need_kwh。"""
        p_req = 0.0
        for a in actions:
            if a.get("asset") == "agv-fleet" and a.get("cmd") == "charge":
                p_req += max(0.0, _to_f(a.get("kW"), 0.0))
        if p_req <= 0 or not self._agv_fleet:
            return 0.0
        # 单车上限
        p_caps = [x["p_charge_max_kw"] for x in self._agv_fleet]
        p_use = min(p_req, sum(p_caps))
        # 按需电排序分配
        fleet = sorted(self._agv_fleet, key=lambda x: x["need_kwh"], reverse=True)
        remain = p_use
        for agv in fleet:
            if remain <= 0: break
            cap = agv["p_charge_max_kw"]
            alloc = min(remain, cap)
            e = alloc * step_h
            # 更新需电与 soc
            agv["need_kwh"] = max(0.0, agv["need_kwh"] - e)
            remain -= alloc
        self._agv_total_need_kwh = sum(x["need_kwh"] for x in self._agv_fleet)
        return p_use

    def _apply_lighting(self, actions: List[Dict[str, Any]]) -> float:
        """照明减载功率（kW，>0 表示减少负荷）。"""
        reduce_kw = 0.0
        for a in actions:
            if a.get("cmd") == "reduce":
                zid = a.get("asset")
                z = next((z for z in self._zones if z["id"] == zid), None)
                if not z:
                    continue
                r = _clip(_to_f(a.get("kW"), 0.0), 0.0, z["max_reduce_kw"])
                reduce_kw += r
        return reduce_kw

    def _apply_chiller(self, actions: List[Dict[str, Any]], step_h: float) -> float:
        """冷站功率（kW，>0 吸收），COP 随环境温度 & 设定点偏置变化；偏置越大，舒适风险越高。"""
        for a in actions:
            if a.get("asset") == self._chiller_id and a.get("cmd") == "set_sp_delta":
                self._sp_delta_acc = _clip(self._sp_delta_acc + _to_f(a.get("delta_c"), 0.0), -self._sp_delta_limit, self._sp_delta_limit)
        t = self._t
        amb = self._ambient_c[t] if t < len(self._ambient_c) else self._ambient_c[-1]
        cop = max(1.5, self._cop_a - self._cop_b * max(0.0, amb - 25.0))
        # 设定点 +1℃ 可降冷量需求 ~3%；-1℃ 增加 ~3%
        demand_factor = 1.0 - 0.03 * self._sp_delta_acc
        cooling_kw = max(0.0, self._chiller_base_kw * demand_factor)
        elec_kw = cooling_kw / max(1e-6, cop)
        return elec_kw

    # ------------------------ 降级成本（无 Lagrange 时） ------------------------
    def _fallback_cost(self, net_kw: float, p_bess: float, p_agv: float, p_light: float, p_ch: float, t: int) -> float:
        price = self._price[t] if t < len(self._price) else self._price[-1]
        carbon = self._grid_carbon[t] if t < len(self._grid_carbon) else self._grid_carbon[-1]
        step_h = self._step_min / 60.0
        energy_kwh = net_kw * step_h
        demand_pen = max(0.0, (net_kw - max(0.0, self._limit_kw - self._reserve_kw))) * 0.3  # 超限惩罚
        comfort_pen = max(0.0, abs(self._sp_delta_acc) - 1.5) * 2.0                            # 舒适惩罚（>1.5℃）
        return price * energy_kwh + carbon * 0.05 * energy_kwh + demand_pen + comfort_pen

    # ------------------------ 观测/上下文 ------------------------
    def _obs(self) -> Dict[str, Any]:
        t = self._t
        return {
            "t": t,
            "step_min": self._step_min,
            "price": self._price[t] if t < len(self._price) else self._price[-1],
            "grid_carbon": self._grid_carbon[t] if t < len(self._grid_carbon) else self._grid_carbon[-1],
            "ambient_c": self._ambient_c[t] if t < len(self._ambient_c) else self._ambient_c[-1],
            "soc": round(self._soc, 4),
            "agv_unmet_kwh": round(self._agv_total_need_kwh, 3),
            "feeder_limit_kw": self._limit_kw,
            "nminus1_reserve_kw": self._reserve_kw,
            "net_load_kw": self._net_kw_prev
        }

    def _window_now(self) -> Dict[str, str]:
        now = datetime.now(timezone.utc)
        return {"start": now.isoformat(), "end": (now + timedelta(minutes=self._step_min)).isoformat()}

    def _ctx_for_opt(self, t: int) -> Dict[str, Any]:
        return {
            "agg_forecast_kw": [self._base_kw[t]],
            "feeder_limit_kw": self._limit_kw,
            "nminus1_reserve_kw": self._reserve_kw,
            "storage": {"id": self._bess_id, "capacity_kwh": self._cap_kwh, "soc": self._soc,
                        "p_charge_max_kw": self._pch_max, "p_discharge_max_kw": self._pdis_max,
                        "efficiency": (self._eta_ch + self._eta_dis) / 2.0,
                        "soc_min": self._soc_min, "soc_max": self._soc_max},
            "agv_list": [{"id": a["id"], "soc": a["soc"], "need_kwh": a["need_kwh"], "p_charge_max_kw": a["p_charge_max_kw"]} for a in self._agv_fleet],
            "lighting_zones": self._zones
        }

    def _ctx_get(self, k: str, default=None):
        # 占位：可接 DI 注入的全局策略上下文；当前直接用 default
        return default

    # ------------------------ 默认策略（行为策略） ------------------------
    def _default_policy_step(self) -> List[Dict[str, Any]]:
        """用于 rollouts_pro 的行为策略：峰段放电、低谷充电、少量照明减载、适度抬高冷站设定点"""
        t = self._t
        price = self._price[t]
        base = self._base_kw[t]
        lim = max(0.0, self._limit_kw - self._reserve_kw)
        acts: List[Dict[str, Any]] = []
        # 峰段：放电 + 照明减载 + 冷站 +0.5℃
        if base > 0.9 * lim:
            if self._soc > self._soc_min + 0.05:
                acts.append({"asset": self._bess_id, "cmd": "discharge", "kW": min(self._pdis_max, 0.5*self._pdis_max)})
            if self._zones:
                acts.append({"asset": self._zones[0]["id"], "cmd": "reduce", "kW": 0.5 * self._zones[0]["max_reduce_kw"]})
            acts.append({"asset": self._chiller_id, "cmd": "set_sp_delta", "delta_c": 0.5})
        # 低谷：AGV 充电 + BESS 充电
        else:
            if self._agv_total_need_kwh > 0.1:
                acts.append({"asset": "agv-fleet", "cmd": "charge", "kW": min(150.0, max(50.0, 0.1*lim))})
            if self._soc < self._soc_max - 0.05:
                acts.append({"asset": self._bess_id, "cmd": "charge", "kW": min(self._pch_max, 0.4*self._pch_max)})
        return acts

    # ------------------------ 随机上下文生成器（更真实） ------------------------
    def _random_ctx(self) -> Dict[str, Any]:
        """随机化价差/温度/负荷起伏/AGV 需求，用于合成离线数据"""
        step_min = random.choice([5, 10, 15])
        horizon_min = random.choice([60, 90, 120])
        L = horizon_min // step_min
        base = random.uniform(1700, 2300)
        price_base = random.uniform(0.9, 1.3)
        # 峰谷价/温度/负荷
        price_curve = [round(price_base + 0.3*math.sin(i/8.0), 3) for i in range(L)]
        carbon_curve = [round(random.uniform(0.45, 0.65), 4) for _ in range(L)]
        ambient_curve = [round(28 + 6*math.sin(i/10.0) + random.uniform(-0.5,0.5), 2) for i in range(L)]
        base_curve = [round(base + 0.15*base*math.sin(i/12.0) + random.uniform(-30,30), 2) for i in range(L)]
        # 车队
        n_agv = random.choice([8, 12, 16])
        agv_list = [{"id": f"agv-{i:02d}", "soc": random.uniform(0.3, 0.7), "need_kwh": random.uniform(10, 40), "p_charge_max_kw": random.choice([90, 120, 150])} for i in range(n_agv)]
        # 照明
        zones = [{"id":"yard-z1","max_reduce_kw":random.uniform(15,30),"min_duty":0.6}]
        # 储能
        storage = {"id":"bess-01","capacity_kwh":random.uniform(2000,3000),"soc":random.uniform(0.4,0.7),
                   "p_charge_max_kw":random.choice([600,800,1000]),"p_discharge_max_kw":random.choice([600,800,1000]),
                   "eta_charge":0.96,"eta_discharge":0.96,"soc_min":0.1,"soc_max":0.9,"ramp_kw_per_step":random.choice([300,400,500]),
                   "degradation_cost_per_kwh": random.choice([0.015,0.02,0.03])}

        return {
            "step_min": step_min, "horizon_min": horizon_min,
            "base_load_kw": base_curve,
            "price_curve": price_curve,
            "grid_carbon_curve": carbon_curve,
            "ambient_temp_curve": ambient_curve,
            "feeder_limit_kw": random.choice([2600, 2800, 3000]),
            "nminus1_reserve_kw": random.choice([0, 100, 200]),
            "storage": storage,
            "agv_list": agv_list,
            "lighting_zones": zones,
            "chiller": {"id":"chiller-01","base_kw":random.uniform(250,350),"sp_delta_limit":2.0}
        }
