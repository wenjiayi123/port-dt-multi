# ============================================
# app/services/rl_rollout.py
# --------------------------------------------
# 策略/模型发布管控：影子（Shadow）→ 灰度（Canary）→ 全量（Full） + 一键回滚（Rollback）
#
# 大白话：
#   - 影子：只做对比，不下发；收集指标（MAPE、SLA、Guard拦截率、削峰、降碳）。
#   - 灰度：小流量上线（如 10%），监控指标，达标则逐步增量（20%、50%…），不达标就回滚。
#   - 全量：100% 上线；仍持续监控，退化则回滚到“上一个稳定版本”。
#   - 所有决策/阈值/证据落盘，便于审计与追责。
# ============================================

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math


# --------- 阈值配置（现场可根据 KPI 调整） ---------
@dataclass
class RolloutThresholds:
    """决策阈值（命中则允许升级；不满足则停止或回滚）"""
    # 预测命中率相关（越小越好）
    mape_energy_max: float = 0.10           # 能耗/峰值 MAPE ≤ 10%
    # 安全守护相关（越小越好）
    guard_block_rate_max: float = 0.05      # 守护拦截率 ≤ 5%
    # 作业服务水平（越小越好）
    sla_violation_rate_max: float = 0.02    # SLA 违约率 ≤ 2%
    # 成本/效果（越大越好，若为 None 则不强制）
    peak_reduction_kw_min: Optional[float] = None
    carbon_reduction_kg_min: Optional[float] = None
    # Minimum decision window in minutes or configured batch units
    min_batches_for_decision: int = 10


# --------- 发布状态/快照 ---------
@dataclass
class RolloutState:
    phase: str = "idle"                # idle|shadow|canary|full|rollback
    stable_version: Optional[str] = None
    candidate_version: Optional[str] = None
    traffic_pct: float = 0.0           # 灰度流量 [0,1]，full=1.0
    created_at: str = ""
    updated_at: str = ""
    thresholds: RolloutThresholds = field(default_factory=RolloutThresholds)
    # 统计累计（滑窗/全窗均可；这里做简单累计 + 最近N条缓存）
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "batches": 0,
        "sum_mape_energy": 0.0,
        "sum_guard_block": 0,
        "sum_sla_violate": 0,
        "sum_peak_before_kw": 0.0,
        "sum_peak_after_kw": 0.0,
        "sum_carbon_before_kg": 0.0,
        "sum_carbon_after_kg": 0.0,
        "recent": []  # 最近若干条样本（便于证据包查看）
    })


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RLPolicyRollout:
    """
    发布状态机（无外部依赖）：
      - register_candidate()：登记候选策略/模型版本
      - start_shadow()：进入影子模式
      - promote_to_canary(pct)：进入灰度并设置流量
      - step_canary()：灰度增量（满足阈值才会推进）
      - promote_to_full()：转全量
      - rollback()：回滚到 stable_version
      - ingest_batch_metrics(batch)：吃一批指标（来自仿真/线上 A/B）
      - get_status()：读当前状态与聚合 KPI
    """

    def __init__(self, storage=None,
                 state_path: str = "data/objects/rl/rollout.json",
                 audit_dir: str = "data/objects/audit"):
        self.storage = storage
        self.state_path = state_path
        self.audit_dir = audit_dir
        self.state = self._load_state()

    # ---------------- 公共：候选版本登记 ----------------
    def register_candidate(self, candidate_version: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        st = self.state
        st.candidate_version = str(candidate_version)
        st.created_at = st.created_at or _now_iso()
        st.updated_at = _now_iso()
        # 清空历史指标（换候选版本时重新评估）
        st.stats = RolloutState().stats
        self._save_state(st, audit_note=f"register_candidate {candidate_version}", extra={"meta": meta or {}})
        return self.get_status()

    # ---------------- 公共：启动影子 ----------------
    def start_shadow(self) -> Dict[str, Any]:
        st = self.state
        if not st.candidate_version:
            raise ValueError("请先 register_candidate() 再 start_shadow()")
        st.phase = "shadow"
        st.traffic_pct = 0.0
        st.updated_at = _now_iso()
        self._save_state(st, audit_note="start_shadow")
        return self.get_status()

    # ---------------- 公共：转灰度（给定小流量） ----------------
    def promote_to_canary(self, traffic_pct: float = 0.1) -> Dict[str, Any]:
        st = self.state
        if st.phase not in ("shadow", "idle"):
            raise ValueError(f"当前阶段 {st.phase} 不允许直接转灰度")
        if not (0.0 < traffic_pct <= 0.5):
            raise ValueError("灰度初始流量建议 (0, 0.5]")
        st.phase = "canary"
        st.traffic_pct = float(traffic_pct)
        st.updated_at = _now_iso()
        self._save_state(st, audit_note=f"promote_to_canary pct={traffic_pct}")
        return self.get_status()

    # ---------------- 公共：灰度增量（满足阈值才推进） ----------------
    def step_canary(self, increment: float = 0.2) -> Dict[str, Any]:
        st = self.state
        if st.phase != "canary":
            raise ValueError("只有在 canary 阶段才允许 step_canary")
        # 决策：是否可以增量
        ok, reason = self._check_thresholds_ready()
        if not ok:
            self._save_state(st, audit_note="canary_hold", extra={"reason": reason})
            return {"ok": False, "reason": reason, "status": self.get_status()}
        st.traffic_pct = min(1.0, st.traffic_pct + float(increment))
        st.updated_at = _now_iso()
        self._save_state(st, audit_note=f"canary_step +{increment}")
        return {"ok": True, "status": self.get_status()}

    # ---------------- 公共：转全量 ----------------
    def promote_to_full(self) -> Dict[str, Any]:
        st = self.state
        if st.phase not in ("canary", "shadow"):
            raise ValueError(f"当前阶段 {st.phase} 不允许 promote_to_full")
        ok, reason = self._check_thresholds_ready(strict=True)
        if not ok:
            self._save_state(st, audit_note="full_blocked", extra={"reason": reason})
            return {"ok": False, "reason": reason, "status": self.get_status()}

        # 全量生效：candidate 成为 stable
        st.phase = "full"
        st.traffic_pct = 1.0
        st.stable_version = st.candidate_version
        st.updated_at = _now_iso()
        self._save_state(st, audit_note="promote_to_full")
        return {"ok": True, "status": self.get_status()}

    # ---------------- 公共：一键回滚 ----------------
    def rollback(self) -> Dict[str, Any]:
        st = self.state
        if not st.stable_version:
            raise ValueError("没有可回滚的 stable_version")
        st.phase = "rollback"
        st.candidate_version = None
        st.traffic_pct = 1.0
        st.updated_at = _now_iso()
        self._save_state(st, audit_note="rollback_to_stable")
        # 回滚完成视为 full 到 stable
        st.phase = "full"
        self._save_state(st, audit_note="rollback_done")
        return self.get_status()

    # ---------------- 公共：吃一批指标（仿真/线上 A/B） ----------------
    def ingest_batch_metrics(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量指标 schema（真实线上请按此对接；未提供的字段会自动忽略）：
          每个样本可包含：
            - mape_energy: float            # 能耗/峰值预测 MAPE
            - guard_blocked: bool           # 是否被 RLSafety 拦截
            - sla_violated: bool            # 本次作业是否 SLA 违约
            - peak_before_kw: float         # 策略前聚合峰值
            - peak_after_kw: float          # 策略后聚合峰值
            - carbon_before_kg: float       # 策略前碳排
            - carbon_after_kg: float        # 策略后碳排
            - meta: {...}                   # 任意附加信息（泊位/班组/设备等）
        """
        st = self.state
        stats = st.stats
        for row in (batch or []):
            stats["batches"] += 1
            if "mape_energy" in row and isinstance(row["mape_energy"], (int, float)):
                stats["sum_mape_energy"] += float(row["mape_energy"])
            if row.get("guard_blocked") is True:
                stats["sum_guard_block"] += 1
            if row.get("sla_violated") is True:
                stats["sum_sla_violate"] += 1
            if "peak_before_kw" in row:
                stats["sum_peak_before_kw"] += float(row["peak_before_kw"])
            if "peak_after_kw" in row:
                stats["sum_peak_after_kw"] += float(row["peak_after_kw"])
            if "carbon_before_kg" in row:
                stats["sum_carbon_before_kg"] += float(row["carbon_before_kg"])
            if "carbon_after_kg" in row:
                stats["sum_carbon_after_kg"] += float(row["carbon_after_kg"])

            # 仅缓存最近 50 条用于证据包
            if len(stats["recent"]) >= 50:
                stats["recent"].pop(0)
            stats["recent"].append(row)

        st.updated_at = _now_iso()
        self._save_state(st, audit_note="ingest_batch", extra={"batch_size": len(batch or [])})
        return self.get_status()

    # ---------------- 公共：读当前状态与 KPI 聚合 ----------------
    def get_status(self) -> Dict[str, Any]:
        st = self.state
        s = st.stats
        n = max(1, int(s.get("batches", 0) or 1))
        mean_mape = (s["sum_mape_energy"] / n) if n > 0 else 0.0
        guard_rate = (s["sum_guard_block"] / n) if n > 0 else 0.0
        sla_rate = (s["sum_sla_violate"] / n) if n > 0 else 0.0
        peak_before = (s["sum_peak_before_kw"] / n) if n > 0 else 0.0
        peak_after = (s["sum_peak_after_kw"] / n) if n > 0 else 0.0
        carbon_before = (s["sum_carbon_before_kg"] / n) if n > 0 else 0.0
        carbon_after = (s["sum_carbon_after_kg"] / n) if n > 0 else 0.0

        peak_reduction = max(0.0, peak_before - peak_after)
        carbon_reduction = max(0.0, carbon_before - carbon_after)

        return {
            "phase": st.phase,
            "stable_version": st.stable_version,
            "candidate_version": st.candidate_version,
            "traffic_pct": round(st.traffic_pct, 3),
            "updated_at": st.updated_at,
            "thresholds": asdict(st.thresholds),
            "metrics": {
                "batches": s["batches"],
                "mape_energy_mean": round(mean_mape, 6),
                "guard_block_rate": round(guard_rate, 6),
                "sla_violation_rate": round(sla_rate, 6),
                "peak_reduction_kw_mean": round(peak_reduction, 3),
                "carbon_reduction_kg_mean": round(carbon_reduction, 3),
            }
        }

    # ---------------- 私有：阈值判断（可选严格模式） ----------------
    def _check_thresholds_ready(self, strict: bool = False) -> (bool, str):
        st = self.state
        th = st.thresholds
        k = self.get_status()
        m = k["metrics"]
        if m["batches"] < th.min_batches_for_decision:
            return False, f"样本数不足：{m['batches']} < {th.min_batches_for_decision}"

        conds = []
        conds.append((m["mape_energy_mean"] <= th.mape_energy_max,
                      f"MAPE {m['mape_energy_mean']} <= {th.mape_energy_max}"))
        conds.append((m["guard_block_rate"] <= th.guard_block_rate_max,
                      f"Guard拦截率 {m['guard_block_rate']} <= {th.guard_block_rate_max}"))
        conds.append((m["sla_violation_rate"] <= th.sla_violation_rate_max,
                      f"SLA违约率 {m['sla_violation_rate']} <= {th.sla_violation_rate_max}"))

        if th.peak_reduction_kw_min is not None:
            conds.append((m["peak_reduction_kw_mean"] >= th.peak_reduction_kw_min,
                          f"削峰 {m['peak_reduction_kw_mean']} >= {th.peak_reduction_kw_min}"))
        if th.carbon_reduction_kg_min is not None:
            conds.append((m["carbon_reduction_kg_mean"] >= th.carbon_reduction_kg_min,
                          f"降碳 {m['carbon_reduction_kg_mean']} >= {th.carbon_reduction_kg_min}"))

        # 严格模式下必须全部满足；非严格模式可放宽（例如至少满足 80% 条件）
        if strict:
            ok = all(c[0] for c in conds)
        else:
            ok = (sum(1 for c in conds if c[0]) >= math.ceil(0.8 * len(conds)))

        reason = "; ".join([f"{'OK' if c[0] else 'NO'}:{c[1]}" for c in conds])
        return ok, reason

    # ---------------- 状态落盘/读取/证据包 ----------------
    def _save_state(self, st: RolloutState, audit_note: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
        st.updated_at = _now_iso()
        data = asdict(st)
        path = Path(self.state_path)
        try:
            if self.storage:
                parent = str(path.parent)
                self.storage.ensure_dir(parent)
                self.storage.write_json(self.state_path, data)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        # 证据包
        try:
            audit_dir = Path(self.audit_dir)
            audit_dir.mkdir(parents=True, exist_ok=True)
            evidence = {
                "ts": _now_iso(),
                "note": audit_note,
                "state": data,
                "extra": extra or {},
            }
            ev_path = audit_dir / f"rollout-{int(datetime.now(timezone.utc).timestamp())}.json"
            ev_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_state(self) -> RolloutState:
        path = Path(self.state_path)
        try:
            if self.storage:
                d = self.storage.read_json(self.state_path)
            elif path.exists():
                d = json.loads(path.read_text(encoding="utf-8"))
            else:
                d = None
        except Exception:
            d = None

        if isinstance(d, dict):
            try:
                # 兼容旧字段/缺省
                th = d.get("thresholds") or {}
                st = RolloutState(
                    phase=d.get("phase", "idle"),
                    stable_version=d.get("stable_version"),
                    candidate_version=d.get("candidate_version"),
                    traffic_pct=float(d.get("traffic_pct", 0.0)),
                    created_at=d.get("created_at") or _now_iso(),
                    updated_at=d.get("updated_at") or _now_iso(),
                    thresholds=RolloutThresholds(**th),
                    stats=d.get("stats") or RolloutState().stats,
                )
                return st
            except Exception:
                pass

        # 初始化
        st = RolloutState()
        st.created_at = _now_iso()
        st.updated_at = _now_iso()
        return st
