# ============================================
# app/services/rl_safety.py
# --------------------------------------------
# RL 策略安全守护（Shielding）与硬性红线校验
#
# 大白话：
#   - 这个文件是“刹车+护栏”。任何要下发到设备/系统的策略，先过这里的硬性规则检查。
#   - 该守护不仅能“判定是否安全”，还能“自动限幅/裁剪”超标动作（比如把kW剪到额定上限内）。
#   - 同时输出“证据包”（谁提了啥策略、用的模型版本/配置、怎么判定、怎么裁剪、最后结果）。
#
# 真实对接：
#   - 它读取（或生成）资产能力上限表：data/objects/config/asset_caps.json
#   - 没有真实数据时，会根据你 Telemetry 的功率序列推导“电流/温度/SOC”的合理近似（可替换为真实点位）。
#   - 将证据包写入：data/objects/audit/guard-*.json
# ============================================

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------- 数据类：各类约束上限（设备/电池/电网） ---------
@dataclass
class DeviceCaps:
    """设备级硬约束（例：岸桥/场桥/充电桩等）"""
    rated_kw: float = 60.0         # 设备名义额定功率
    amp_max: float = 180.0         # 允许最大电流（A）
    temp_max_c: float = 85.0       # 允许最高温度（℃）
    temp_min_c: float = -10.0      # 允许最低温度（℃）
    ramp_kw_per_min: float = 30.0  # 斜坡限制（kW/分钟），防止突变冲击
    phase_voltage_v: float = 400.0 # 估算电压（V），用于由kW->A的近似（缺电流点位时）

@dataclass
class BatteryCaps:
    """电池/充电系统约束（适用于 AGV/无人集卡/储能/充换电等）"""
    charge_kw_max: float = 120.0
    discharge_kw_max: float = 80.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_step_max_per_min: float = 0.02  # 单位时间SOC变化限幅（防异常估算）

@dataclass
class GridCaps:
    """电网/馈线侧约束"""
    feeder_limit_kw: float = 3000.0    # 馈线需量上限（kW）
    nminus1_reserve_kw: float = 200.0  # N-1 预留（kW），预留后不许超过（安全裕度）

@dataclass
class GuardrailsConfig:
    """守护配置"""
    device: DeviceCaps = DeviceCaps()
    battery: BatteryCaps = BatteryCaps()
    grid: GridCaps = GridCaps()


# ---------- 工具函数 ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return d

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ============================================================
# 主类：RLSafetyGuard
# ============================================================
class RLSafetyGuard:
    """
    大白话用途：
      - 提供 validate_and_shield(strategy, ...) ，输入一条策略，返回是否安全、触发了哪些红线、
        是否对动作做了自动裁剪/限幅，以及一份“证据包”文件路径。
      - 该类依赖 telemetry（拿实时功率）和 storage（存证据包与配置）。
      - 若没有真实端口点位（电流/温度/SOC），会基于功率用简单物理近似“推断”出合理的A/℃/SOC，
        以便你先把流程打通；落地时只需把推断换成真实点即可。

    关键输入（strategy 结构与你现有RL面板一致）：
      strategy = {
        "id": "xxx",
        "actions": [
          {"asset": "qc-01", "cmd": "reduce"|"idle"|"charge"|"discharge"|..., "kW": 20, "percent": 0.1}
        ],
        "window": {"start": "ISO", "end": "ISO"},
        "scope": {"asset_ids": ["..."]} | {"type":"qc"} | ...
      }
    """

    def __init__(self, telemetry, storage=None, caps_path: Optional[str] = None):
        self.telemetry = telemetry     # 需要：list_assets(), get_recent_power(asset_id) 或 get_series(...)
        self.storage = storage         # 需要：.write_json(path, data) / .ensure_dir(path) （若无则用本地文件写入）
        self.caps_path = caps_path or "data/objects/config/asset_caps.json"
        self.cfg = self._load_or_init_caps()

    # ---------------- 配置加载/初始化 ----------------
    def _load_or_init_caps(self) -> Dict[str, Any]:
        """从对象存储/本地加载资产约束；没有则按资产类型生成一份默认模板"""
        # 1) 尝试从对象存储读取
        if self.storage:
            try:
                # ObjectStorage 的约定路径风格：file://./data/objects/... 已在 di 里配置
                p = self.caps_path
                obj = self.storage.read_json(p)  # 你现有 storage 若不支持 read_json，就走 except 写本地
                if isinstance(obj, dict) and obj.get("assets"):
                    return obj
            except Exception:
                pass

        # 2) 尝试从本地文件系统读取
        p_local = Path(self.caps_path)
        if p_local.exists():
            try:
                return json.loads(p_local.read_text(encoding="utf-8"))
            except Exception:
                pass

        # 3) 自动生成（按资产类型）——真实落地时请替换为真实台账
        assets = []
        try:
            assets = self.telemetry.list_assets() or []
        except Exception:
            assets = [{"id": "qc-01", "label": "QC-01"}, {"id": "agv-01", "label": "AGV-01"}]

        def _type_of(aid: str, label: str) -> str:
            s = (aid or "").lower() + (label or "")
            if s.startswith("qc") or "岸桥" in label: return "qc"
            if s.startswith("yc") or "场桥" in label: return "yc"
            if s.startswith("agv") or "truck" in s:  return "agv"
            if s.startswith("cs") or "充电" in label: return "cs"
            if s.startswith("ps") or "配电" in label: return "ps"
            if "冷" in label or "warehouse" in s or "wh" in s: return "wh"
            return "misc"

        caps_map = {}
        for a in assets:
            typ = _type_of(a.get("id",""), a.get("label",""))
            if typ == "qc":
                caps_map[a["id"]] = {"type": "qc", "device": asdict(DeviceCaps(rated_kw=80.0, amp_max=220.0, temp_max_c=85.0)),
                                     "battery": asdict(BatteryCaps(charge_kw_max=40.0, discharge_kw_max=30.0))}
            elif typ == "yc":
                caps_map[a["id"]] = {"type": "yc", "device": asdict(DeviceCaps(rated_kw=60.0, amp_max=200.0, temp_max_c=85.0))}
            elif typ == "agv":
                caps_map[a["id"]] = {"type": "agv", "device": asdict(DeviceCaps(rated_kw=35.0, amp_max=160.0, temp_max_c=70.0)),
                                     "battery": asdict(BatteryCaps(charge_kw_max=120.0, discharge_kw_max=60.0))}
            elif typ == "cs":
                caps_map[a["id"]] = {"type": "cs", "device": asdict(DeviceCaps(rated_kw=150.0, amp_max=300.0, temp_max_c=75.0))}
            else:
                caps_map[a["id"]] = {"type": "misc", "device": asdict(DeviceCaps())}

        cfg = {
            "generated_at": _now_iso(),
            "grid": asdict(GridCaps()),
            "assets": caps_map
        }

        # 落盘（存对象存储优先，否则写本地）
        try:
            if self.storage:
                self.storage.ensure_dir(os.path.dirname(self.caps_path))
                self.storage.write_json(self.caps_path, cfg)
            else:
                p_local.parent.mkdir(parents=True, exist_ok=True)
                p_local.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        return cfg

    # ---------------- 对外主接口：校验 + 屏蔽/裁剪 ----------------
    def validate_and_shield(
        self,
        strategy: Dict[str, Any],
        enforce_guardrails: bool = True,
        horizon_min: int = 60,
        step_min: int = 1,
        baseline_agg_kw: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        返回：
          {
            "ok": True/False,
            "rules": [ {"rule":"...", "passed":True/False, "detail":"...", "affected_actions":[...]} ],
            "actions_after_shield": [...],
            "peak_check": {"baseline_max_kw":..., "simulated_max_kw":..., "feeder_limit_kw":..., "passed":...},
            "evidence_path": "data/objects/audit/guard-*.json"
          }
        """
        actions = list(strategy.get("actions") or [])
        window = strategy.get("window") or {}
        start_iso = window.get("start")
        end_iso = window.get("end")

        # 1) 估算基线/窗口长度
        L = max(1, int(horizon_min / max(1, step_min)))
        if baseline_agg_kw is None:
            baseline_agg_kw = self._estimate_baseline_agg_kw(L)

        # 2) 对每个动作进行设备级剪裁/安全检查
        rules = []
        new_actions = []
        for act in actions:
            a2, rlist = self._guard_one_action(act)
            new_actions.append(a2)
            rules.extend(rlist)

        # 3) 估算策略后的聚合负荷（仅用于需量/馈线守护粗估）
        sim_agg_kw = self._apply_actions_on_agg(baseline_agg_kw, new_actions, step_min)

        # 4) 馈线/N-1 守护
        gcfg = self.cfg.get("grid", {}) or {}
        feeder_limit = float(gcfg.get("feeder_limit_kw", 3000.0)) - float(gcfg.get("nminus1_reserve_kw", 200.0))
        peak_check = {
            "baseline_max_kw": round(max(baseline_agg_kw or [0.0]), 3),
            "simulated_max_kw": round(max(sim_agg_kw or [0.0]), 3),
            "feeder_limit_kw": round(feeder_limit, 3),
            "passed": (max(sim_agg_kw or [0.0]) <= feeder_limit)
        }
        if not peak_check["passed"]:
            rules.append({"rule":"grid_feeder_limit", "passed":False, "detail": f"模拟后峰值 {peak_check['simulated_max_kw']} 超过馈线限额 {feeder_limit}kW", "affected_actions":"ALL"})

        # 5) 通过/不通过 & 证据包
        ok = all(r.get("passed", True) for r in rules) and peak_check["passed"]

        # 若不通过且“强制守护”，尝试进一步限幅（第二次裁剪：把超出的动作按比例压小）
        if enforce_guardrails and not ok:
            new_actions2 = self._rebalance_to_fit(new_actions, baseline_agg_kw, feeder_limit, step_min)
            sim_agg_kw2 = self._apply_actions_on_agg(baseline_agg_kw, new_actions2, step_min)
            peak_check2 = dict(peak_check)
            peak_check2["simulated_max_kw"] = round(max(sim_agg_kw2 or [0.0]), 3)
            peak_check2["passed"] = (max(sim_agg_kw2 or [0.0]) <= feeder_limit)

            # 如果第二次压缩后通过，就更新动作与检查结果
            if peak_check2["passed"]:
                new_actions = new_actions2
                peak_check = peak_check2
                rules.append({"rule":"auto_rebalance", "passed":True, "detail":"已自动等比例压缩功率，满足馈线限额", "affected_actions":"ALL"})
                ok = True
            else:
                rules.append({"rule":"auto_rebalance", "passed":False, "detail":"自动压缩后仍无法满足馈线限额，需调整窗口/范围", "affected_actions":"ALL"})

        evidence = {
            "generated_at": _now_iso(),
            "strategy_id": strategy.get("id"),
            "window": {"start": start_iso, "end": end_iso, "horizon_min": horizon_min, "step_min": step_min},
            "baseline": {"agg_kW": [round(x,3) for x in baseline_agg_kw]},
            "simulated": {"agg_kW": [round(x,3) for x in sim_agg_kw]},
            "rules": rules,
            "peak_check": peak_check,
            "actions_before": actions,
            "actions_after_shield": new_actions
        }
        evidence_path = self._save_evidence(evidence)

        return {
            "ok": bool(ok),
            "rules": rules,
            "actions_after_shield": new_actions,
            "peak_check": peak_check,
            "evidence_path": evidence_path
        }

    # ---------------- 内部：对单个动作做校验/限幅 ----------------
    def _guard_one_action(self, act: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        校验&限幅逻辑（设备/电池/温度/电流/SOC）：
          - 所有设备：|ΔkW| 不得超过额定/斜坡；由kW估算电流，不得超过 amp_max；估算温度不得超 temp_max_c。
          - AGV/电池类：charge/discharge 时检查 SOC 边界与 kW 上限。
        返回：裁剪后的动作 + 触发的规则列表
        """
        aid = str(act.get("asset", ""))
        cmd = str(act.get("cmd", ""))
        kw = float(act.get("kW", 0.0))
        pct = float(act.get("percent", 0.0))

        caps = self.cfg.get("assets", {}).get(aid) or {}
        dev = DeviceCaps(**(caps.get("device") or {}))
        bat = BatteryCaps(**(caps.get("battery") or {})) if ("battery" in caps or cmd in ("charge","discharge")) else None

        rules: List[Dict[str, Any]] = []

        # 额定&斜坡限幅
        kw_allowed = dev.rated_kw
        kw_clipped = _clip(abs(kw), 0.0, kw_allowed)
        if abs(kw) > kw_allowed:
            rules.append({"rule":"device_rated_kw", "passed":False, "detail":f"请求 |kW|={abs(kw)} 超过额定 {kw_allowed}，已限幅到 {kw_clipped}"})
        # 斜坡：这里简单认为动作只在1分钟内生效，斜坡按分钟检查（可扩展为基于最近功率点）
        kw_ramp = dev.ramp_kw_per_min
        if abs(kw_clipped) > kw_ramp:
            rules.append({"rule":"device_ramp_limit", "passed":False, "detail":f"请求 |kW|={kw_clipped} 超过斜坡 {kw_ramp}，已限幅到 {kw_ramp}"})
            kw_clipped = _clip(kw_clipped, -kw_ramp, kw_ramp)

        # 电流估算（I ≈ kW / (V*pf) * 1000；pf 取 0.9）
        pf = 0.9
        I = (kw_clipped / max(1e-6, dev.phase_voltage_v * pf)) * 1000.0
        if I > dev.amp_max:
            # 进一步按电流限幅
            kw_i_max = dev.amp_max * dev.phase_voltage_v * pf / 1000.0
            rules.append({"rule":"device_amp_limit", "passed":False, "detail":f"估算电流 {I:.1f}A 超过上限 {dev.amp_max}A，kW 限幅到 {kw_i_max:.1f}"})
            kw_clipped = min(kw_clipped, kw_i_max)

        # 温度估算：简化线性近似 T = T_amb + a*I（a取0.12 近似），T_amb=30℃
        T_amb = 30.0
        T_est = T_amb + 0.12 * I
        if T_est > dev.temp_max_c:
            # 超温则再限幅，使 T_est <= temp_max（反解I）
            I_allowed = max(0.0, (dev.temp_max_c - T_amb) / 0.12)
            kw_t_max = I_allowed * dev.phase_voltage_v * pf / 1000.0
            rules.append({"rule":"device_temp_limit", "passed":False, "detail":f"估算温度 {T_est:.1f}℃ 超过上限 {dev.temp_max_c}℃，kW 限幅到 {kw_t_max:.1f}"})
            kw_clipped = min(kw_clipped, kw_t_max)

        # 电池类：充放电功率 & SOC 边界
        if bat and cmd in ("charge", "discharge"):
            if cmd == "charge":
                if kw_clipped > bat.charge_kw_max:
                    rules.append({"rule":"battery_charge_kw", "passed":False, "detail":f"充电功率 {kw_clipped} 超过上限 {bat.charge_kw_max}，已限幅"})
                    kw_clipped = bat.charge_kw_max
            if cmd == "discharge":
                if kw_clipped > bat.discharge_kw_max:
                    rules.append({"rule":"battery_discharge_kw", "passed":False, "detail":f"放电功率 {kw_clipped} 超过上限 {bat.discharge_kw_max}，已限幅"})
                    kw_clipped = bat.discharge_kw_max

            # SOC 近似：若没有真实SOC点位，估 0.5；落地时替换为 telemetry.get_series(..., point="soc")
            soc = self._estimate_soc(aid)
            if cmd == "charge" and soc >= bat.soc_max:
                rules.append({"rule":"battery_soc_max", "passed":False, "detail":f"SOC≈{soc:.2f} 已达上限 {bat.soc_max}，不允许继续充电"})
                kw_clipped = 0.0
            if cmd == "discharge" and soc <= bat.soc_min:
                rules.append({"rule":"battery_soc_min", "passed":False, "detail":f"SOC≈{soc:.2f} 已达下限 {bat.soc_min}，不允许继续放电"})
                kw_clipped = 0.0

        # 构造被裁剪后的动作（保留符号方向）
        act2 = dict(act)
        act2["kW"] = round(math.copysign(kw_clipped, kw), 3)
        if pct:
            # 百分比类动作（如照明分区调光），这里默认只做记录，不做裁剪；你可按需扩展
            pass

        # 没有触发的规则也记一条通过项，便于前端展示“通过”率
        if not rules:
            rules.append({"rule":"basic_safety", "passed":True, "detail":"设备/电池约束均满足"})

        return act2, rules

    # ---------------- 内部：基线估计/负荷叠加 ----------------
    def _estimate_baseline_agg_kw(self, L: int) -> List[float]:
        """
        粗估聚合基线曲线：取所有资产最近功率均值，做一个平滑基线（仅用于守护校验）
        若你已经有 /api/forecast 的基线，后续可改为从 forecast 取。
        """
        try:
            assets = self.telemetry.list_assets() or []
        except Exception:
            assets = [{"id":"agv-01"}]

        vals = []
        for a in assets:
            pts = self._recent_power(a["id"])
            if pts:
                vals.append(sum(pts)/len(pts))
        base = (sum(vals)/len(vals)) if vals else 20.0

        return [round(max(0.0, base + 4.0*math.sin(i/12.0)), 3) for i in range(L)]

    def _apply_actions_on_agg(self, baseline: List[float], actions: List[Dict[str, Any]], step_min: int) -> List[float]:
        """
        将动作影响叠加到聚合基线上：这里简单处理，只对窗口内“所有步长”统一加和 kW_delta
        真实孪生里应按资产/时间对齐逐点叠加。
        """
        if not baseline:
            return []
        delta = 0.0
        for a in actions:
            kw = _safe_float(a.get("kW"), 0.0)
            # reduce/idle 等动作：约定 reduce 的 kW 为负（降负荷），charge 为正（升负荷），视你的策略结构而定
            # 如果前端传来的 reduce 是正数，亦可在此统一转负数（按你实际定义来）
            if str(a.get("cmd","")) == "reduce":
                delta += -abs(kw)
            else:
                delta += kw
        return [round(max(0.0, x + delta), 3) for x in baseline]

    # ---------------- 内部：估算 SOC / 最近功率 ----------------
    def _estimate_soc(self, asset_id: str) -> float:
        # 没有真实SOC点位时：根据 asset_id 简单给一个稳定值；落地时替换为真实点位
        if "agv" in asset_id.lower():
            return 0.55
        if "cs" in asset_id.lower():
            return 0.60
        return 0.50

    def _recent_power(self, asset_id: str, minutes: int = 10) -> List[float]:
        """
        读取最近几分钟功率（kW）。优先使用 telemetry.get_recent_power；
        没有的话，用 telemetry.get_series(..., point='active_power_kw') 组装。
        """
        try:
            if hasattr(self.telemetry, "get_recent_power"):
                pts = self.telemetry.get_recent_power(asset_id) or []
                return [float(p.get("kW", 0.0)) for p in pts if isinstance(p, dict)]
        except Exception:
            pass

        # 回退：用 get_series
        try:
            now = datetime.now(timezone.utc).timestamp()
            seq = self.telemetry.get_series(asset_id, "active_power_kw", now - minutes*60, now, 60) or []
            return [float(p.get("v", 0.0)) for p in seq if isinstance(p, dict)]
        except Exception:
            return []

    # ---------------- 内部：自动再平衡（使峰值不超馈线） ----------------
    def _rebalance_to_fit(self, actions: List[Dict[str, Any]], baseline: List[float], feeder_limit: float, step_min: int) -> List[Dict[str, Any]]:
        """
        将所有提高负荷的动作按相同比例压缩，直到“基线+动作”的峰值不超过馈线限额
        """
        sim = self._apply_actions_on_agg(baseline, actions, step_min)
        peak = max(sim or [0.0])
        if peak <= feeder_limit:
            return actions

        # 只对“增加负载”的动作压缩（kW>0 或 charge）
        inc_idx = [i for i,a in enumerate(actions) if (a.get("kW", 0.0) > 0.0 and str(a.get("cmd","")) != "reduce")]
        if not inc_idx:
            return actions

        # 二分比例
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2.0
            test = actions.copy()
            for i in inc_idx:
                k = float(actions[i].get("kW", 0.0))
                test[i] = dict(actions[i]); test[i]["kW"] = round(k * mid, 3)
            sim2 = self._apply_actions_on_agg(baseline, test, step_min)
            if max(sim2 or [0.0]) <= feeder_limit:
                hi = mid
            else:
                lo = mid
        # 应用最终比例
        out = actions.copy()
        for i in inc_idx:
            k = float(actions[i].get("kW", 0.0))
            out[i] = dict(actions[i]); out[i]["kW"] = round(k * hi, 3)
        return out

    # ---------------- 内部：证据包落盘 ----------------
    def _save_evidence(self, evidence: Dict[str, Any]) -> str:
        # 优先用对象存储；否则写本地
        rel_path = f"data/objects/audit/guard-{int(datetime.now(timezone.utc).timestamp())}.json"
        try:
            if self.storage:
                self.storage.ensure_dir("data/objects/audit")
                self.storage.write_json(rel_path, evidence)
                return rel_path
        except Exception:
            pass
        p = Path(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        return rel_path
