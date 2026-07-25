# ============================================
# app/services/rl.py
# --------------------------------------------
# 可解释的 RL 策略服务（轻量启发式/占位，接口与后续真模型对齐）
#
# 目标：
#   - 提供 propose_actions(state, objective="cost") -> dict
#   - 根据当前状态（平均预测负荷、储能 SoC、电价等）给出“可执行建议”
#   - 输出包含：动作列表、预估收益/成本、简单解释说明，便于在前端右侧输出框查看
#
# 说明：
#   - 这是一个“可解释启发式/混合优化”的实现，便于前端联调与演示；
#   - 支持加载离线训练产物（policy-*.json）的“参数化策略”，对动作做比例/限幅/偏置；
#   - simulate_with_envpro() 仅依赖 Pro 环境（rl_env_pro），无旧 env 依赖。
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
import time
import math

# ----------- 简单的价格/需量分档（示例） -----------
def _price_bucket(price: float) -> str:
    if price >= 1.30:
        return "peak"
    if price <= 1.00:
        return "valley"
    return "flat"


@dataclass
class BatteryCaps:
    """储能/充电桩的简化能力约束（示例）。"""
    charge_kw_max: float = 120.0   # 最大充电功率（吸收电力）
    discharge_kw_max: float = 80.0 # 最大放电功率（反馈电力）
    soc_min: float = 0.10
    soc_max: float = 0.90


class RLService:
    """
    对外主接口：
      propose_actions(state: dict, objective="cost") -> dict

    入参 state（可选键）：
      - avgForecastKW: float  未来短期平均负荷（聚合）
      - soc: float            储能 SoC (0..1)
      - price: float          当前电价（元/kWh）或价差
      - demand_limit_kw: float 需量上限（用于需量风险规避）
      - demand_eta_min: float   距离越峰 ETA（分钟）

    出参结构示例：
      {
        "objective": "cost",
        "score": {"savings_est": 128.4, "risk_reduction": 0.62},
        "actions": [
          {"asset":"cs-01","cmd":"charge","kW":60,"duration_min":30,"note":"谷价充电"},
          {"asset":"ps-01","cmd":"discharge","kW":25,"duration_min":20,"note":"高价放电"}
        ],
        "explain": "基于 SoC / 电价 / 预测负荷的启发式/混合策略。",
        "policy_version": "policy-cql-1699999999",          # Active policy parameter version
        "policy_params_applied": true                        # Whether parameterized policy values are active
      }
    """

    # --------------------- 构造 & 依赖 ---------------------
    def __init__(self, caps: Optional[BatteryCaps] = None, optimizer=None, lagrange=None):
        # 保持向后兼容。可从 DI 注入 optimizer/lagrange；否则尝试本地构造；失败则为 None。
        self.caps = caps or BatteryCaps()
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            try:
                from app.services.exec_closedloop.optimize import HybridOptimizer
                self.optimizer = HybridOptimizer()
            except Exception:
                self.optimizer = None

        if lagrange is not None:
            self.lagrange = lagrange
        else:
            try:
                from app.services.rl_suite.rl_lagrange import LagrangeManager
                self.lagrange = LagrangeManager()
            except Exception:
                self.lagrange = None

        # 策略参数缓存（来自 data/objects/rl/policies/policy-*.json 或 RL_POLICY_PATH）
        self._policy_cache: Tuple[Optional[str], Optional[dict]] = (None, None)  # (version, params)

    # ===================== 策略参数（训练产物） =====================
    def _find_latest_policy_file(self) -> Optional[str]:
        """优先使用环境变量 RL_POLICY_PATH；否则在默认目录里找 mtime 最新的 policy-*.json。"""
        env_path = os.getenv("RL_POLICY_PATH")
        if env_path and Path(env_path).exists():
            return env_path
        root = Path("data/objects/rl/policies")
        if not root.exists():
            return None
        candidates = sorted(root.glob("policy-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0]) if candidates else None

    def _load_policy_params(self) -> Tuple[Optional[str], Optional[dict]]:
        """
        加载训练器导出的 best_params：
        返回 (version, params)；params 示例：
          {
            "bess_charge_scale": 1.1,
            "bess_discharge_scale": 0.95,
            "agv_charge_scale": 1.0,
            "lighting_reduce_scale": 1.0,
            "chiller_delta_scale": 1.0,
            "max_kw_cap": 1000.0
          }
        """
        # 命中缓存直接返回
        ver, params = self._policy_cache
        if ver and params:
            return ver, params

        f = self._find_latest_policy_file()
        if not f:
            return None, None
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            params = data.get("best_params") or {}
            ver = Path(f).stem  # e.g. policy-cql-1699999999
            # 基本校验
            if not isinstance(params, dict) or not params:
                return None, None
            self._policy_cache = (ver, params)
            return ver, params
        except Exception:
            return None, None

    def _apply_policy_params_to_actions(self, actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
        """
        把训练产物参数应用到动作上（比例/限幅/偏置），与 rl_train.PolicyParams 保持一致。
        返回 (new_actions, version, applied_flag)。
        """
        ver, params = self._load_policy_params()
        if not params:
            return actions, None, False

        # 读取参数（带默认值防御）
        bc = float(params.get("bess_charge_scale", 1.0))
        bd = float(params.get("bess_discharge_scale", 1.0))
        ag = float(params.get("agv_charge_scale", 1.0))
        lg = float(params.get("lighting_reduce_scale", 1.0))
        cd = float(params.get("chiller_delta_scale", 1.0))
        cap = float(params.get("max_kw_cap", 1000.0))

        new: List[Dict[str, Any]] = []
        for a in actions:
            b = dict(a)
            cmd = str(b.get("cmd", ""))
            asset = str(b.get("asset", ""))

            # kW 类动作
            if "kW" in b:
                kw = float(b.get("kW", 0.0))
                if cmd == "charge" and "bess" in asset:
                    kw *= bc
                elif cmd == "discharge" and "bess" in asset:
                    kw *= bd
                elif cmd == "charge" and asset == "agv-fleet":
                    kw *= ag
                elif cmd == "reduce":
                    kw *= lg
                # 限幅
                kw = max(0.0, min(abs(kw), cap))
                b["kW"] = kw

            # 冷站设定点偏置
            if cmd == "set_sp_delta":
                delta = float(b.get("delta_c", 0.0)) * cd
                b["delta_c"] = max(-2.0, min(2.0, delta))

            new.append(b)

        return new, ver, True

    # ===================== 启发式策略（回退） =====================
    def _policy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        soc = float(state.get("soc", 0.5))
        price = float(state.get("price", 1.10))
        avg_forecast_kw = float(state.get("avgForecastKW", 80.0))
        demand_limit_kw = float(state.get("demand_limit_kw", 500.0))
        demand_eta_min = float(state.get("demand_eta_min", 60.0))

        bucket = _price_bucket(price)
        actions: List[Dict[str, Any]] = []
        note_parts: List[str] = []

        # 1) 需量风险优先
        risk = max(0.0, (avg_forecast_kw - 0.9 * demand_limit_kw) / max(1.0, demand_limit_kw))
        risk = min(1.0, risk)
        if risk > 0.0:
            if soc > max(self.caps.soc_min, 0.25):
                kw = min(self.caps.discharge_kw_max, max(10.0, 0.2 * demand_limit_kw))
                dur = 15 if demand_eta_min < 20 else 30
                actions.append({"asset": "ps-01", "cmd": "discharge", "kW": round(kw, 1), "duration_min": dur, "note": "需量风险削峰"})
                soc -= 0.02 * (dur / 15)  # 粗略 SoC 预估
                note_parts.append("优先削峰")
            else:
                note_parts.append("SoC 偏低，无法削峰")

        # 2) 价差策略
        if bucket == "valley" and soc < self.caps.soc_max:
            kw = min(self.caps.charge_kw_max, 60.0)
            actions.append({"asset": "cs-01", "cmd": "charge", "kW": round(kw, 1), "duration_min": 30, "note": "谷价充电"})
            note_parts.append("谷价充电")
        elif bucket == "peak" and soc > self.caps.soc_min:
            kw = min(self.caps.discharge_kw_max, 40.0)
            actions.append({"asset": "ps-01", "cmd": "discharge", "kW": round(kw, 1), "duration_min": 20, "note": "高价放电"})
            note_parts.append("高价放电")
        else:
            actions.append({"asset": "cs-01", "cmd": "idle", "kW": 0.0, "duration_min": 15, "note": "保持待机"})
            note_parts.append("平段待机")

        # 3) 应用训练参数（若有）
        actions_adj, ver, applied = self._apply_policy_params_to_actions(actions)

        # 4) 粗略收益估计
        savings = 0.0
        for act in actions_adj:
            kwh = float(act.get("kW", 0.0)) * (float(act.get("duration_min", 0.0)) / 60.0)
            if act.get("cmd") == "discharge":
                savings += kwh * price
            elif act.get("cmd") == "charge":
                savings -= kwh * price * 0.8  # 假设充放效率 80%

        out = {
            "actions": actions_adj,
            "score": {
                "savings_est": round(savings, 2),
                "risk_reduction": round(risk, 2),
            },
            "explain": "启发式策略（已自动应用训练参数）" if applied else "启发式策略",
        }
        if ver:
            out["policy_version"] = ver
            out["policy_params_applied"] = applied
        return out

    # ===================== 对外：策略建议 =====================
    def propose_actions(self, state: Dict[str, Any], objective: str = "cost") -> Dict[str, Any]:
        # 优先走“优化器+拉格朗日”的混合策略；失败则回退到启发式
        st = dict(state or {})
        st["objective"] = objective
        try:
            res = self._policy_hybrid(st)
        except Exception:
            res = self._policy(st)
        res["objective"] = objective
        res["ts"] = int(time.time())
        return res

    # ===================== 混合策略（优化器+拉格朗日） =====================
    def _policy_hybrid(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """结合 HybridOptimizer 给可行域/初值，再用 LagrangeManager 打分排序。"""
        from datetime import datetime, timezone, timedelta
        if not (getattr(self, "optimizer", None) and getattr(self, "lagrange", None)):
            raise RuntimeError("optimizer or lagrange is unavailable")

        horizon_min = int(state.get("horizon_min", 60))
        step_min = int(state.get("step_min", 5))

        # 聚合预测：若 state 提供数组则用之，否则从 avgForecastKW 生成
        series = state.get("aggForecastSeriesKW")
        if not series:
            avg = float(state.get("avgForecastKW", 1800.0))
            L = max(1, int(horizon_min / max(1, step_min)))
            series = [avg + 0.05 * avg * math.sin(i / 12.0) for i in range(L)]

        # 上下文（真实项目请从 telemetry/asset_caps 注入）
        ctx = {
            "agg_forecast_kw": series,
            "feeder_limit_kw": float(state.get("demand_limit_kw", 2800.0)),
            "nminus1_reserve_kw": float(state.get("nminus1_reserve_kw", 0.0)),
            "storage": {
                "id": state.get("bess_id", "bess-01"),
                "capacity_kwh": float(state.get("bess_capacity_kwh", 2500.0)),
                "soc": float(state.get("soc", 0.55)),
                "p_charge_max_kw": float(state.get("bess_p_charge_max_kw", 800.0)),
                "p_discharge_max_kw": float(state.get("bess_p_discharge_max_kw", 800.0)),
                "efficiency": float(state.get("bess_efficiency", 0.95)),
                "soc_min": float(state.get("bess_soc_min", 0.10)),
                "soc_max": float(state.get("bess_soc_max", 0.90)),
            },
            "agv_list": state.get("agv_list", [{"id": "agv-01", "soc": 0.5, "need_kwh": 30.0, "p_charge_max_kw": 120.0}]),
            "lighting_zones": state.get("lighting_zones", [{"id": "yard-z1", "max_reduce_kw": 25.0, "min_duty": 0.6}]),
            "device_caps": state.get("device_caps") or {},
        }

        # 1) 优化器给可行域 & 初值
        plan = self.optimizer.propose_initial_plan(ctx, horizon_min=horizon_min, step_min=step_min)

        # 2) 组装候选策略（本轮用一步常量近似）
        now = datetime.now(timezone.utc)
        strategy = {
            "id": f"hybrid-{int(time.time())}",
            "window": {"start": now.isoformat(), "end": (now + timedelta(minutes=horizon_min)).isoformat()},
            "actions": [{"asset": a["asset"], "cmd": a["cmd"], "kW": float(a["kW"])} for a in plan.get("initial_actions", [])],
            "meta": {
                "expected_delay_min": float(state.get("expected_delay_min", 0.0)),
                "expected_throughput_teu": float(state.get("expected_throughput_teu", 100)),
            },
        }

        # 3) 目标权重配置
        from app.services.rl_suite.rl_lagrange import ObjectiveConfig, PriceSignal, CarbonFactor
        obj = (state.get("objective") or "cost").lower()
        if obj == "carbon":
            ocfg = ObjectiveConfig(w_cost=0.0, w_carbon_intensity=1.0, w_energy=0.0, w_delay_min=0.0)
        elif obj == "energy":
            ocfg = ObjectiveConfig(w_cost=0.0, w_carbon_intensity=0.0, w_energy=1.0, w_delay_min=0.0)
        elif obj == "delay":
            ocfg = ObjectiveConfig(w_cost=0.0, w_carbon_intensity=0.0, w_energy=0.0, w_delay_min=1.0)
        else:
            ocfg = ObjectiveConfig(w_cost=1.0, w_carbon_intensity=0.2, w_energy=0.0, w_delay_min=0.1)
        self.lagrange.cfg = ocfg

        price = PriceSignal(
            tou_price_per_kwh=float(state.get("price", 1.1)),
            carbon_price_per_kg=float(state.get("carbon_price_per_kg", 0.0)),
            demand_charge_per_kw=float(state.get("demand_charge_per_kw", 0.0)),
        )
        cf = CarbonFactor(kg_per_kwh=float(state.get("grid_carbon_kg_per_kwh", 0.55)))

        # 4) 打分（含可解释组成）
        score = self.lagrange.evaluate_strategy(
            strategy,
            context={
                "baseline_peak_kw": float(state.get("baseline_peak_kw", plan.get("baseline_peak_kw", 2000.0))),
                "expected_delay_min": float(state.get("expected_delay_min", 0.0)),
                "expected_throughput_teu": float(state.get("expected_throughput_teu", 100)),
            },
            price=price,
            carbon_factor=cf,
        )

        # 5) 组织返回：先把优化器动作转成输出格式，再应用训练参数
        actions_out: List[Dict[str, Any]] = []
        for a in strategy["actions"]:
            actions_out.append({
                "asset": a["asset"],
                "cmd": a["cmd"],
                "kW": float(a["kW"]),
                "duration_min": step_min,  # 常量近似
                "note": "优化器初值，已按目标打分",
            })

        actions_adj, ver, applied = self._apply_policy_params_to_actions(actions_out)

        # 6) 经济性/约束摘要
        savings = -score["total_score"]  # 用 -total_score 作为“越大越好”的代理
        risk_red = 1.0 if score["constraints"].get("feeder_peak_kw", {}).get("violation", 0.0) == 0 else 0.0

        out = {
            "actions": actions_adj,
            "score": {"savings_est": round(float(savings), 2), "risk_reduction": round(float(risk_red), 2)},
            "explain": "混合策略：优化器初值 + 约束RL（拉格朗日）打分" + (" + 应用训练参数" if applied else ""),
            "feasible_region": plan.get("feasible_region", []),
            "residual_peak_kw": plan.get("residual_peak_kw"),
            "baseline_peak_kw": plan.get("baseline_peak_kw"),
            "limit_kw": plan.get("limit_kw"),
            "objective_breakdown": score["objective"],
            "constraints": score["constraints"],
            "lagrangian_penalty": score["lagrangian_penalty"],
            "total_score": score["total_score"],
        }
        if ver:
            out["policy_version"] = ver
            out["policy_params_applied"] = applied
        return out

    # ===================== Pro 环境（一步仿真预览） =====================
    def simulate_with_envpro(
        self,
        state: Dict[str, Any],
        actions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        用 Pro 环境做“一步”策略仿真（不改 UI 路由，服务侧可直接调用）。
        - 若不传 actions：先调用本服务算建议（优先混合策略，失败回退启发式），再把动作送入仿真；
        - 会对动作应用训练参数（若存在），保证“建议”和“仿真”一致。
        """
        # 延迟导入 Pro 环境，避免部署阶段强依赖
        try:
            from app.services.rl_suite.rl_env_pro import PortEnergyEnvPro
        except Exception as e:
            raise RuntimeError(f"缺少 Pro 环境模块 rl_env_pro：{e}")

        horizon_min = int(state.get("horizon_min", 60))
        step_min = int(state.get("step_min", 5))

        # 组装环境上下文：如果给了曲线就用曲线，否则 Pro 环境会自合成
        ctx = {
            "horizon_min": horizon_min,
            "step_min": step_min,
            "feeder_limit_kw": float(state.get("feeder_limit_kw", state.get("demand_limit_kw", 2800.0))),
            "nminus1_reserve_kw": float(state.get("nminus1_reserve_kw", 0.0)),
            "storage": {
                "id": state.get("bess_id", "bess-01"),
                "capacity_kwh": float(state.get("bess_capacity_kwh", 2500.0)),
                "soc": float(state.get("soc", 0.55)),
                "p_charge_max_kw": float(state.get("bess_p_charge_max_kw", 800.0)),
                "p_discharge_max_kw": float(state.get("bess_p_discharge_max_kw", 800.0)),
                "eta_charge": float(state.get("bess_eta_charge", state.get("bess_efficiency", 0.96))),
                "eta_discharge": float(state.get("bess_eta_discharge", state.get("bess_efficiency", 0.96))),
                "soc_min": float(state.get("bess_soc_min", 0.10)),
                "soc_max": float(state.get("bess_soc_max", 0.90)),
                "ramp_kw_per_step": float(state.get("bess_ramp_kw_per_step", 400.0)),
                "degradation_cost_per_kwh": float(state.get("bess_degradation_cost_per_kwh", 0.02)),
            },
            "base_load_kw": state.get("base_load_kw"),
            "price_curve": state.get("price_curve") or ([float(state.get("price_per_kwh", state.get("price", 1.1)))] * max(1, horizon_min // max(1, step_min))),
            "grid_carbon_curve": state.get("grid_carbon_curve"),
            "ambient_temp_curve": state.get("ambient_temp_curve"),
            "agv_list": state.get("agv_list"),
            "lighting_zones": state.get("lighting_zones"),
            "chiller": state.get("chiller"),
        }

        # 动作：没传则先走建议；随后统一应用训练参数
        acts = list(actions or [])
        if not acts:
            try:
                cand = self._policy_hybrid(dict(state or {}))
                src_acts = cand.get("actions", [])
            except Exception:
                cand = self._policy(dict(state or {}))
                src_acts = cand.get("actions", [])
            acts = [{"asset": a.get("asset"), "cmd": a.get("cmd"), "kW": float(a.get("kW", 0.0)), **({"delta_c": a.get("delta_c")} if "delta_c" in a else {})} for a in src_acts]

        # 应用训练参数（与建议一致）
        acts, ver, applied = self._apply_policy_params_to_actions(acts)

        env = PortEnergyEnvPro()
        obs0 = env.reset(ctx)
        obs1, reward, done, info = env.step(acts)

        out = {
            "obs0": obs0,
            "obs1": obs1,
            "actions": info.get("guard", {}).get("actions_after_shield", acts),
            "reward": float(reward),
            "info": info,
        }
        if ver:
            out["policy_version"] = ver
            out["policy_params_applied"] = applied
        return out
