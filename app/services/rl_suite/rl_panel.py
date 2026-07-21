# ============================================
# app/services/rl_suite/rl_panel.py
# --------------------------------------------
# RL 策略面板服务（更接近真实落地版）
# ============================================

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math


def _safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return d


def _parse_iso(ts: str):
    try:
        if isinstance(ts, str) and ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _round_list(values: List[float], nd: int = 3) -> List[float]:
    return [round(_safe_float(v, 0.0), nd) for v in values]


def _sum_kwh(values: List[float], step_min: int) -> float:
    return sum(_safe_float(v, 0.0) for v in values) * (step_min / 60.0)


def _agg_max(values: List[float]) -> float:
    return max(values) if values else 0.0


def _mean(values: List[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _classify(asset_id: str, label: str = "") -> str:
    s = (asset_id or "").lower()
    label = label or ""
    if s.startswith("qc") or "岸桥" in label:
        return "qc"
    if s.startswith("yc") or "场桥" in label:
        return "yc"
    if s.startswith("agv") or "agv" in s:
        return "agv"
    if s.startswith("truck") or "拖车" in label:
        return "truck"
    if s.startswith("wh") or "仓库" in label or "冷" in label:
        return "wh"
    if s.startswith("cs") or "充电" in label:
        return "cs"
    if s.startswith("ps") or "配电" in label:
        return "ps"
    if s.startswith("yard") or "堆场" in label:
        return "yard"
    if "bess" in s or "储能" in label:
        return "bess"
    return "misc"


RATED_KW = {
    "qc": 80.0,
    "yc": 60.0,
    "agv": 30.0,
    "truck": 30.0,
    "wh": 40.0,
    "cs": 120.0,
    "ps": 15.0,
    "yard": 20.0,
    "bess": 250.0,
    "misc": 20.0,
}


@dataclass
class Strategy:
    id: str
    title: str
    category: str
    scope: Dict[str, Any]
    window: Dict[str, str]
    actions: List[Dict[str, Any]]
    impact: Dict[str, Any]
    explain: Dict[str, Any]
    meta: Dict[str, Any]


class RLPanelService:
    def __init__(self, telemetry, forecast, reporting, energy, rl, twin=None):
        self.telemetry = telemetry
        self.forecast = forecast
        self.reporting = reporting
        self.energy = energy
        self.rl = rl
        self.twin = twin

        try:
            self._assets = self.telemetry.list_assets() or []
        except Exception:
            self._assets = [
                {"id": "qc-01", "label": "岸桥 QC-01"},
                {"id": "agv-01", "label": "AGV-01"},
                {"id": "yard-01", "label": "堆场照明 01"},
                {"id": "wh-01", "label": "冷库 01"},
            ]
        self._labels = {a.get("id", ""): a.get("label", "") for a in self._assets}

    def _assets_by_type(self, typ: str) -> List[str]:
        return [
            a["id"]
            for a in self._assets
            if _classify(a.get("id", ""), a.get("label", "")) == typ
        ]

    def _avg_ci(self, asset_ids: Optional[List[str]] = None) -> float:
        ids = asset_ids or [a["id"] for a in self._assets]
        vals = []
        for aid in ids:
            try:
                rpt = self.reporting.generate_mini_report(aid) or {}
                vals.append(_safe_float(rpt.get("carbonIntensity"), 120.0))
            except Exception:
                vals.append(120.0)
        return _mean(vals, 120.0)

    def _energy_price_hint(self) -> Dict[str, float]:
        try:
            summary = self.energy.build_today_summary({}) or {}
        except Exception:
            summary = {}
        return {
            "grid_ci_g_per_kwh": _safe_float(summary.get("avgCarbonIntensity"), 120.0),
            "tou_price": _safe_float(summary.get("touPrice"), 0.72),
            "demand_limit_kw": _safe_float(summary.get("demandLimitKw"), 2800.0),
        }

    def _baseline(self, ids: List[str], horizon_min: int, step_min: int) -> Tuple[Dict[str, List[Dict[str, Any]]], List[float]]:
        per: Dict[str, List[Dict[str, Any]]] = {}
        L = 0
        for aid in ids:
            try:
                seq = (
                    self.forecast.forecast_load(
                        [aid], horizon_min=horizon_min, step_min=step_min
                    )
                    or {}
                ).get(aid, [])
            except Exception:
                seq = []
            if not seq:
                now = _now_utc()
                n = max(1, int(horizon_min / max(step_min, 1)))
                typ = _classify(aid, self._labels.get(aid, ""))
                base = RATED_KW.get(typ, 20.0) * 0.55
                seq = []
                for i in range(n):
                    t = now + timedelta(minutes=i * step_min)
                    shape = 0.92 + 0.15 * math.sin(i / 4.0) + 0.08 * math.sin(i / 11.0)
                    seq.append({"ts": t.isoformat(), "kW": round(max(0.0, base * shape), 3)})
            per[aid] = seq
            L = max(L, len(seq))

        agg: List[float] = []
        for i in range(L):
            s = 0.0
            for aid in ids:
                seq = per.get(aid, [])
                if i < len(seq):
                    s += _safe_float(seq[i].get("kW"), 0.0)
            agg.append(round(s, 3))
        return per, agg

    def _window_index(self, seq: List[Dict[str, Any]], start_iso: str, end_iso: str) -> Tuple[int, int]:
        st = _parse_iso(start_iso)
        ed = _parse_iso(end_iso)
        if not seq or st is None or ed is None:
            return (0, max(0, len(seq) - 1))
        i0, i1 = 0, max(0, len(seq) - 1)
        for i, p in enumerate(seq):
            ts = _parse_iso(p.get("ts"))
            if ts and ts >= st:
                i0 = i
                break
        for i, p in enumerate(seq):
            ts = _parse_iso(p.get("ts"))
            if ts and ts <= ed:
                i1 = i
        if i1 < i0:
            i1 = i0
        return i0, i1

    def _pick_top_assets_by_peak(self, per_asset: Dict[str, List[Dict[str, Any]]], asset_ids: List[str], top_k: int = 3) -> List[str]:
        scored = []
        for aid in asset_ids:
            seq = per_asset.get(aid, [])
            peak = max((_safe_float(p.get("kW"), 0.0) for p in seq), default=0.0)
            scored.append((peak, aid))
        scored.sort(reverse=True)
        return [aid for _, aid in scored[:top_k]]

    def _estimate_delay_min(self, category: str, asset_count: int, intensity: float) -> float:
        base = {
            "quay_crane": 4.0,
            "agv": 0.0,
            "yard_lighting": 0.0,
            "warehouse_cooling": 0.5,
            "shore_power": 1.0,
        }.get(category, 0.5)
        return round(base * max(1, asset_count) * max(0.4, intensity), 2)

    def _confidence(self, category: str, asset_count: int, ratio: float) -> float:
        base = {
            "quay_crane": 0.64,
            "agv": 0.58,
            "yard_lighting": 0.74,
            "warehouse_cooling": 0.68,
            "shore_power": 0.80,
        }.get(category, 0.6)
        penalty = 0.0
        if asset_count == 0:
            penalty += 0.22
        if ratio > 0.28:
            penalty += 0.08
        return round(_clamp(base - penalty, 0.35, 0.92), 2)


    def _business_basis(self, category: str, cmd: str) -> List[str]:
        basis_map = {
            "quay_crane": ["船边作业低谷窗口", "泊位周转不能被明显拉长", "岸桥短时待机优先在非关键作业面"],
            "agv": ["车队补能尽量后移到低价低碳窗口", "避免与车队任务波峰重叠", "不以牺牲可用率换取表面节能"],
            "yard_lighting": ["低作业密度区优先分区调光", "主通道与关键作业面照度优先保留", "安全摄像识别质量不能明显下降"],
            "warehouse_cooling": ["库温裕度允许时再调设定点", "敏感货类不能直接复用", "持续时间不宜过长以免累积温控风险"],
            "shore_power": ["仅在可接岸电窗口内推荐", "需确认接口能力与靠泊状态", "优先把柴油替代收益转化为可解释降碳"],
            "hybrid_rl": ["多资产协同只建议先仿真与影子验证", "需要同时看峰值、能耗、碳与作业约束", "不能把 RL 输出直接视作生产指令"],
        }
        return basis_map.get(category, [f"cmd={cmd} 动作需先经过仿真与守护栏复核"])

    def _dispatch_supports(self, peak_reduction: float, delta_kwh: float, adjusted_asset_count: int) -> List[str]:
        supports: List[str] = []
        if peak_reduction > 0:
            supports.append(f"窗口峰值下降 {round(peak_reduction, 3)} kW")
        if delta_kwh <= 0:
            supports.append(f"总电量变化 {round(delta_kwh, 6)} kWh（未增耗）")
        if adjusted_asset_count > 0:
            supports.append(f"动作命中 {adjusted_asset_count} 个有效资产")
        return supports

    def _dispatch_blockers(self, risk_flags: List[str], adjusted_asset_count: int) -> List[str]:
        blockers = list(risk_flags)
        if adjusted_asset_count <= 0 and "策略未命中有效资产" not in blockers:
            blockers.append("策略未命中有效资产")
        return blockers

    def _readiness_label(self, dispatch_ready: bool, risk_flags: List[str], peak_reduction: float, delta_kwh: float) -> str:
        if dispatch_ready and peak_reduction > 0 and delta_kwh <= 0:
            return "可进入 dry-run"
        if dispatch_ready:
            return "可进入人工复核后 dry-run"
        if any("未命中" in x for x in risk_flags):
            return "需先修正作用范围"
        if any("峰值上升" in x for x in risk_flags):
            return "不适合进入下发"
        return "需补充仿真复核"

    def _readiness_reason(self, dispatch_ready: bool, risk_flags: List[str], peak_reduction: float, delta_kwh: float, adjusted_asset_count: int) -> str:
        if dispatch_ready and peak_reduction > 0 and delta_kwh <= 0 and adjusted_asset_count > 0:
            return "削峰、节能、命中范围三项同时满足，可先进入 dry-run。"
        if dispatch_ready and adjusted_asset_count > 0:
            return "虽然存在轻微保留项，但未触发硬性阻断，可先人工复核后进入 dry-run。"
        if not adjusted_asset_count:
            return "策略未实际命中资产，当前不应进入下发。"
        if any("峰值上升" in x for x in risk_flags):
            return "模拟后峰值反而上升，当前不适合进入下发。"
        if any("总电量未下降" in x for x in risk_flags):
            return "总电量没有明显下降，需结合补能/作业窗口再判断是否保留。"
        return "存在未通过项，建议回看策略原因与仿真支撑字段。"

    def _simulate_basis_lines(self, strategy: Dict[str, Any], ids: List[str], adjusted_assets: set, peak_base: float, peak_sim: float, delta_kwh: float) -> List[str]:
        explain = strategy.get("explain") or {}
        features = explain.get("features") or {}
        selected_assets = ', '.join((features.get('selected_assets') or [])[:3]) or '—'
        dispatch_hint = explain.get('dispatch_hint') or '建议先仿真，再做 dry-run。'
        risk_hint = explain.get('risk_hint') or '需结合守护栏与人工复核判断。'
        return [
            f"资产范围 {len(ids)} 个，实际命中 {len(adjusted_assets)} 个。",
            f"窗口基线峰值 {round(peak_base, 3)} kW，策略后峰值 {round(peak_sim, 3)} kW。",
            f"总电量变化 {round(delta_kwh, 6)} kWh；负值表示节能，正值表示增耗。",
            f"策略出现原因：{explain.get('appearance_reason') or explain.get('reason') or '未提供'}",
            f"候选资产：{selected_assets}",
            f"进入下发前提示：{dispatch_hint}",
            f"业务保留项：{risk_hint}",
        ]

    def _build_strategy(
        self,
        *,
        sid: str,
        title: str,
        category: str,
        asset_ids: List[str],
        start: datetime,
        duration_min: int,
        cmd: str,
        percent: Optional[float] = None,
        kw_delta: Optional[float] = None,
        reason: str,
        risk_hint: str,
        source_hint: str,
        horizon_min: int,
        step_min: int,
    ) -> Strategy:
        end = start + timedelta(minutes=duration_min)
        ids = list(asset_ids)
        if not ids:
            ids = [a["id"] for a in self._assets]

        per_base, agg_base = self._baseline(ids, horizon_min=horizon_min, step_min=step_min)
        selected_ids = self._pick_top_assets_by_peak(per_base, ids, top_k=min(3, len(ids) or 1))
        target_asset = f"*type:{_classify(ids[0], self._labels.get(ids[0], ''))}" if len(ids) > 1 else (ids[0] if ids else "*")

        acts = [{
            "asset": target_asset,
            "cmd": cmd,
            "percent": percent,
            "kW_delta": kw_delta,
        }]

        one_seq = next(iter(per_base.values()), [])
        i0, i1 = self._window_index(one_seq, start.isoformat(), end.isoformat())
        win_base = agg_base[i0:i1 + 1] if agg_base else []
        avg_base_kw = _mean(win_base, _mean(agg_base, 0.0))
        peak_base_kw = max(win_base) if win_base else _agg_max(agg_base)

        ratio = percent if percent is not None else (
            abs(_safe_float(kw_delta, 0.0)) / max(avg_base_kw, 1.0)
        )
        ratio = _clamp(ratio, 0.0, 0.45)

        cmd_factor = {
            "idle": 0.95,
            "reduce": 0.85,
            "lighting_dim": 0.98,
            "setpoint": 0.42,
            "charge": -0.55,
            "discharge": 1.00,
            "shore_power": 0.12,
        }.get(cmd, 0.7)

        est_delta_kw = avg_base_kw * ratio * cmd_factor
        est_peak_reduce = peak_base_kw * ratio * max(cmd_factor, 0.0) * 0.82
        est_delta_kwh = -est_delta_kw * (duration_min / 60.0)
        ci_g = self._avg_ci(ids)
        if cmd == "shore_power":
            est_delta_kwh = -max(3.0, avg_base_kw * 0.03)
            est_carbon = -max(8.0, abs(est_delta_kwh) * (ci_g / 1000.0) * 1.8)
        else:
            est_carbon = est_delta_kwh * (ci_g / 1000.0)

        confidence = self._confidence(category, len(ids), ratio)
        impact = {
            "energy_kWh_saving_est": round(-est_delta_kwh, 2),
            "carbon_kg_saving_est": round(-est_carbon, 2),
            "peak_reduction_kW_est": round(est_peak_reduce, 2),
            "throughput_delay_min_est": self._estimate_delay_min(category, max(1, len(ids)), ratio),
            "confidence_0to1": confidence,
            "risk_level": "low" if confidence >= 0.72 else ("medium" if confidence >= 0.58 else "watch"),
        }

        explain = {
            "reason": reason,
            "appearance_reason": reason,
            "business_basis": self._business_basis(category, cmd),
            "features": {
                "scope_asset_count": len(ids),
                "avg_base_kw_window": round(avg_base_kw, 2),
                "peak_base_kw_window": round(peak_base_kw, 2),
                "carbon_intensity_g_per_kwh": round(ci_g, 2),
                "selected_assets": selected_ids,
                "estimated_action_ratio": round(ratio, 4),
                "window_minutes": duration_min,
                "command": cmd,
                "source_hint": source_hint,
            },
            "trigger_snapshot": {
                "avg_base_kw_window": round(avg_base_kw, 2),
                "peak_base_kw_window": round(peak_base_kw, 2),
                "confidence_0to1": confidence,
                "risk_level": impact["risk_level"],
                "expected_peak_reduction_kW": round(est_peak_reduce, 2),
                "expected_energy_kWh_saving": round(-est_delta_kwh, 2),
            },
            "risk_hint": risk_hint,
            "dispatch_hint": (
                "建议先仿真，再通过守护栏审查后 dry-run 下发"
                if confidence < 0.75
                else "可先仿真，若削峰与约束均通过，可进入 dry-run 下发"
            ),
            "control_boundary": "当前输出属于策略建议，不直接等同生产控制指令。",
            "route_targets": {
                "next_step": "simulate_then_dry_run",
                "opsx": "/#opsx-module",
                "audit": "/#audit-module",
                "twin": "/#twin-module",
            },
        }

        meta = {
            "source": source_hint,
            "generated_at": _now_utc().isoformat(),
            "window_minutes": duration_min,
            "horizon_min": horizon_min,
            "step_min": step_min,
            "selected_assets": selected_ids,
        }

        return Strategy(
            id=sid,
            title=title,
            category=category,
            scope={"asset_ids": ids},
            window={"start": start.isoformat(), "end": end.isoformat()},
            actions=acts,
            impact=impact,
            explain=explain,
            meta=meta,
        )

    def list_strategies(self, horizon_min: int = 360, step_min: int = 5, max_items: int = 8) -> Dict[str, Any]:
        now = _now_utc()
        price_hint = self._energy_price_hint()

        qc_ids = self._assets_by_type("qc")
        agv_ids = self._assets_by_type("agv")
        yard_ids = self._assets_by_type("yard")
        wh_ids = self._assets_by_type("wh")
        ps_ids = self._assets_by_type("ps")

        suggestions: List[Strategy] = [
            self._build_strategy(
                sid="qc_idle_midday",
                title="岸桥午间待机 30 分钟",
                category="quay_crane",
                asset_ids=qc_ids,
                start=now + timedelta(minutes=15),
                duration_min=30,
                cmd="idle",
                percent=0.18,
                reason="班次交接与船边作业低谷窗口出现，岸桥可做短时待机降载。",
                risk_hint="若临时插单或船期压缩，需缩短窗口，避免影响泊位周转。",
                source_hint="forecast_load + reporting.generate_mini_report + asset typing",
                horizon_min=horizon_min,
                step_min=step_min,
            ),
            self._build_strategy(
                sid="agv_valley_charge",
                title="AGV 谷段集中充电 45 分钟",
                category="agv",
                asset_ids=agv_ids,
                start=now + timedelta(minutes=10),
                duration_min=45,
                cmd="charge",
                kw_delta=18.0,
                reason="把补能尽量推向低价/低碳窗口，降低峰值段需量压力。",
                risk_hint="若当前 AGV 任务波峰临近，应减少集中补能规模，避免影响车队可用率。",
                source_hint="forecast + RL objective(cost/carbon) + AGV fleet heuristic",
                horizon_min=horizon_min,
                step_min=step_min,
            ),
            self._build_strategy(
                sid="yard_lighting_dim",
                title="堆场照明分区降 15%（1 小时）",
                category="yard_lighting",
                asset_ids=yard_ids,
                start=now + timedelta(minutes=5),
                duration_min=60,
                cmd="lighting_dim",
                percent=0.15,
                reason="夜间低作业密度区域可分区调光，先保主通道与关键作业面。",
                risk_hint="需保留主通道照度与安全摄像识别质量，不宜整区同时降档。",
                source_hint="yard zoning heuristic + safety first dimming policy",
                horizon_min=horizon_min,
                step_min=step_min,
            ),
            self._build_strategy(
                sid="wh_chiller_setpoint",
                title="冷机设定点上调 0.5℃（2 小时）",
                category="warehouse_cooling",
                asset_ids=wh_ids,
                start=now + timedelta(minutes=20),
                duration_min=120,
                cmd="setpoint",
                percent=0.07,
                reason="在库温余量可接受时，适度抬高设定点可平滑削减冷机负荷。",
                risk_hint="冷链敏感货种不可直接套用，需结合货类与库温裕度。",
                source_hint="warehouse cooling heuristic + comfort / process margin",
                horizon_min=horizon_min,
                step_min=step_min,
            ),
            self._build_strategy(
                sid="shore_power_switch",
                title="岸电替代柴油（等效 50 kWh）",
                category="shore_power",
                asset_ids=ps_ids or qc_ids[:1],
                start=now + timedelta(minutes=30),
                duration_min=60,
                cmd="shore_power",
                kw_delta=0.0,
                reason="在可接岸电窗口优先切换，减少船侧柴油机运行与碳排放。",
                risk_hint="需确认靠泊窗口、接入能力与岸电接口状态。",
                source_hint="shore power eligibility + carbon-first rule",
                horizon_min=horizon_min,
                step_min=step_min,
            ),
        ]

        try:
            state = {
                "avg_load_kw": price_hint["demand_limit_kw"] * 0.72,
                "tou_price": price_hint["tou_price"],
                "grid_ci": price_hint["grid_ci_g_per_kwh"],
                "asset_count": len(self._assets),
            }
            rl_prop = self.rl.propose_actions(state, objective="cost") or {}
            acts = rl_prop.get("actions") or []
            if acts:
                suggestions.append(
                    Strategy(
                        id="rl_hybrid_orchestration",
                        title="RL 综合协同建议（混合目标）",
                        category="hybrid_rl",
                        scope={"asset_ids": [a["id"] for a in self._assets]},
                        window={
                            "start": (now + timedelta(minutes=5)).isoformat(),
                            "end": (now + timedelta(minutes=65)).isoformat(),
                        },
                        actions=[
                            {
                                "asset": a.get("asset") or "*",
                                "cmd": a.get("cmd") or "reduce",
                                "kW_delta": _safe_float(a.get("kW"), 0.0),
                            }
                            for a in acts[:5]
                        ],
                        impact={
                            "energy_kWh_saving_est": round(abs(_safe_float(rl_prop.get("score", {}).get("energy_kWh"), 18.0)), 2),
                            "carbon_kg_saving_est": round(abs(_safe_float(rl_prop.get("score", {}).get("carbon_kg"), 3.5)), 2),
                            "peak_reduction_kW_est": round(abs(_safe_float(rl_prop.get("score", {}).get("peak_reduction_kW"), 12.0)), 2),
                            "throughput_delay_min_est": round(abs(_safe_float(rl_prop.get("score", {}).get("delay_min"), 2.5)), 2),
                            "confidence_0to1": 0.62,
                            "risk_level": "watch",
                        },
                        explain={
                            "reason": rl_prop.get("explain") or "RL 综合考虑成本 / 碳 / 峰值 / SLA 后给出多资产协同动作。",
                            "appearance_reason": rl_prop.get("explain") or "RL 在多目标约束下识别到更值得优先验证的协同动作。",
                            "business_basis": self._business_basis("hybrid_rl", "reduce"),
                            "features": {
                                **(rl_prop.get("score") or {}),
                                "selected_assets": [a.get("asset") for a in acts[:5]],
                                "source_hint": "rl.propose_actions + heuristic score adaptation",
                            },
                            "trigger_snapshot": {
                                "avg_load_kw": round(_safe_float(state.get("avg_load_kw"), 0.0), 2),
                                "tou_price": round(_safe_float(state.get("tou_price"), 0.0), 4),
                                "grid_ci": round(_safe_float(state.get("grid_ci"), 0.0), 2),
                                "asset_count": int(_safe_float(state.get("asset_count"), 0.0)),
                            },
                            "risk_hint": "属于多资产协同建议，必须先仿真再走 dry-run。",
                            "dispatch_hint": "仅建议进入仿真与影子发布，不建议直接人工下发。",
                            "control_boundary": "RL 输出属于候选动作集合，不直接等同生产调度指令。",
                        },
                        meta={
                            "source": "rl.propose_actions + heuristic score adaptation",
                            "generated_at": now.isoformat(),
                            "window_minutes": 60,
                            "horizon_min": horizon_min,
                            "step_min": step_min,
                            "selected_assets": [a.get("asset") for a in acts[:5]],
                        },
                    )
                )
        except Exception:
            pass

        suggestions.sort(
            key=lambda s: (
                _safe_float(s.impact.get("peak_reduction_kW_est"), 0.0),
                _safe_float(s.impact.get("energy_kWh_saving_est"), 0.0),
                _safe_float(s.impact.get("confidence_0to1"), 0.0),
            ),
            reverse=True,
        )
        suggestions = suggestions[:max_items]

        return {
            "generated_at": now.isoformat(),
            "objective": "cost + carbon + peak + operational safety",
            "assumptions": {
                "tou_price": price_hint["tou_price"],
                "grid_ci_g_per_kwh": price_hint["grid_ci_g_per_kwh"],
                "demand_limit_kw": price_hint["demand_limit_kw"],
                "note": "未接入真实 TOS / 船期 / 电价 / 气象时，部分口径采用 forecast + heuristic 估算。",
            },
            "linkage": {
                "main_to_rl_panel": "/rl-panel",
                "rl_panel_back_to_main": "#strategy-exec-module",
                "chain": ["strategy", "simulate", "dispatch", "audit"]
            },
            "strategies": [asdict(s) for s in suggestions],
        }

    def _apply_action_to_kw(self, kw: float, act: Dict[str, Any], typ: str) -> float:
        cmd = str(act.get("cmd") or "")
        pct = _safe_float(act.get("percent"), 0.0)
        delta = _safe_float(act.get("kW_delta"), 0.0)

        if cmd in ("idle", "reduce"):
            if pct > 0:
                kw = kw * (1.0 - pct)
            elif delta != 0:
                kw = max(0.0, kw + min(delta, 0.0))
            else:
                kw = kw * 0.82
        elif cmd == "lighting_dim":
            kw = kw * (1.0 - (pct if pct > 0 else 0.12))
        elif cmd == "setpoint":
            kw = kw * (1.0 - (pct if pct > 0 else 0.06))
        elif cmd == "charge":
            kw = kw + max(0.0, delta if delta else RATED_KW.get(typ, 20.0) * 0.25)
        elif cmd == "discharge":
            kw = max(0.0, kw - max(0.0, delta))
        elif cmd == "shore_power":
            kw = kw * 0.97
        return round(max(0.0, kw), 3)

    def simulate(self, strategy: Dict[str, Any], horizon_min: int = 360, step_min: int = 1) -> Dict[str, Any]:
        now = _now_utc()
        scope = strategy.get("scope") or {}
        ids = scope.get("asset_ids") or []

        if not ids:
            acts = strategy.get("actions") or []
            if acts and str(acts[0].get("asset", "")).startswith("*type:"):
                typ = str(acts[0]["asset"]).split(":", 1)[1]
                ids = self._assets_by_type(typ)

        if not ids:
            ids = [a["id"] for a in self._assets]

        per_base, agg_base = self._baseline(ids, horizon_min=horizon_min, step_min=step_min)

        w0 = _parse_iso((strategy.get("window") or {}).get("start", "")) or now
        w1 = _parse_iso((strategy.get("window") or {}).get("end", "")) or (now + timedelta(minutes=30))
        actions = strategy.get("actions") or []

        per_sim: Dict[str, List[Dict[str, Any]]] = {}
        asset_delta_kwh: Dict[str, float] = {}
        adjusted_assets = set()

        for aid in ids:
            typ = _classify(aid, self._labels.get(aid, ""))
            seq = per_base.get(aid, [])
            out_seq: List[Dict[str, Any]] = []
            before = []
            after = []
            for p in seq:
                t = _parse_iso(p.get("ts"))
                kw0 = _safe_float(p.get("kW"), 0.0)
                kw = kw0
                if t and (w0 <= t <= w1):
                    for act in actions:
                        asset_ref = str(act.get("asset") or "")
                        apply = (
                            asset_ref == aid
                            or asset_ref == "*"
                            or (asset_ref.startswith("*type:") and typ == asset_ref.split(":", 1)[1])
                        )
                        if apply:
                            kw = self._apply_action_to_kw(kw, act, typ)
                            adjusted_assets.add(aid)
                out_seq.append({"ts": p.get("ts"), "kW": round(kw, 3)})
                before.append(kw0)
                after.append(kw)

            per_sim[aid] = out_seq
            asset_delta_kwh[aid] = round(_sum_kwh(after, step_min) - _sum_kwh(before, step_min), 6)

        L = max((len(v) for v in per_sim.values()), default=0)
        agg_sim: List[float] = []
        for i in range(L):
            s = 0.0
            for aid in ids:
                seq = per_sim.get(aid, [])
                if i < len(seq):
                    s += _safe_float(seq[i].get("kW"), 0.0)
            agg_sim.append(round(s, 3))

        total_base_kwh = _sum_kwh(agg_base, step_min)
        total_sim_kwh = _sum_kwh(agg_sim, step_min)
        delta_kwh = total_sim_kwh - total_base_kwh

        ci_g = self._avg_ci(ids)
        delta_carbon_kg = delta_kwh * (ci_g / 1000.0)
        peak_base = _agg_max(agg_base)
        peak_sim = _agg_max(agg_sim)
        peak_reduction = peak_base - peak_sim

        contributors = sorted(
            [
                {
                    "asset_id": aid,
                    "label": self._labels.get(aid, aid),
                    "delta_kWh": round(asset_delta_kwh.get(aid, 0.0), 4),
                    "type": _classify(aid, self._labels.get(aid, "")),
                }
                for aid in ids
            ],
            key=lambda x: x["delta_kWh"]
        )[:5]

        risk_flags: List[str] = []
        if peak_reduction < 0:
            risk_flags.append("策略后峰值上升")
        if delta_kwh > 0 and not any(str(a.get("cmd")) == "charge" for a in actions):
            risk_flags.append("总电量未下降")
        if not adjusted_assets:
            risk_flags.append("策略未命中有效资产")
        if len(ids) == 0:
            risk_flags.append("无有效资产范围")

        dispatch_ready = len(risk_flags) == 0 or (
            len(risk_flags) == 1 and "总电量未下降" in risk_flags
        )
        readiness_label = self._readiness_label(dispatch_ready, risk_flags, peak_reduction, delta_kwh)
        readiness_reason = self._readiness_reason(
            dispatch_ready=dispatch_ready,
            risk_flags=risk_flags,
            peak_reduction=peak_reduction,
            delta_kwh=delta_kwh,
            adjusted_asset_count=len(adjusted_assets),
        )
        explain = strategy.get("explain") or {}
        simulate_basis = self._simulate_basis_lines(
            strategy=strategy,
            ids=ids,
            adjusted_assets=adjusted_assets,
            peak_base=peak_base,
            peak_sim=peak_sim,
            delta_kwh=delta_kwh,
        )

        supports = self._dispatch_supports(peak_reduction, delta_kwh, len(adjusted_assets))
        blockers = self._dispatch_blockers(risk_flags, len(adjusted_assets))

        return {
            "strategy_id": strategy.get("id", ""),
            "strategy_title": strategy.get("title", ""),
            "summary": {
                "delta_kWh": round(delta_kwh, 6),
                "delta_carbon_kg": round(delta_carbon_kg, 6),
                "peak_reduction_kW": round(peak_reduction, 3),
                "window": strategy.get("window", {}),
                "scope_size": len(ids),
                "adjusted_asset_count": len(adjusted_assets),
                "dispatch_ready": dispatch_ready,
                "dispatch_ready_label": readiness_label,
                "dispatch_ready_reason": readiness_reason,
                "supports": supports,
                "blockers": blockers,
            },
            "baseline": {
                "agg_kW": _round_list(agg_base, 3),
                "total_kWh": round(total_base_kwh, 6),
                "peak_kW": round(peak_base, 3),
            },
            "simulated": {
                "agg_kW": _round_list(agg_sim, 3),
                "total_kWh": round(total_sim_kwh, 6),
                "peak_kW": round(peak_sim, 3),
            },
            "contributors": contributors,
            "strategy_reasoning": {
                "appearance_reason": explain.get("appearance_reason") or explain.get("reason") or "",
                "business_basis": explain.get("business_basis") or [],
                "trigger_snapshot": explain.get("trigger_snapshot") or {},
                "dispatch_hint": explain.get("dispatch_hint") or "",
                "risk_hint": explain.get("risk_hint") or "",
            },
            "feasibility": {
                "ok": dispatch_ready,
                "risk_flags": risk_flags,
                "window_hit": {"start": w0.isoformat(), "end": w1.isoformat()},
                "simulate_basis": simulate_basis,
                "operator_guidance": {
                    "next_action": "dry_run" if dispatch_ready else "revise_or_resimulate",
                    "message": readiness_reason,
                },
                "decision": {
                    "dispatch_ready": dispatch_ready,
                    "label": readiness_label,
                    "reason": readiness_reason,
                    "supports": supports,
                    "blockers": blockers,
                    "supported_by": {
                        "peak_reduction_kW": round(peak_reduction, 3),
                        "delta_kWh": round(delta_kwh, 6),
                        "adjusted_asset_count": len(adjusted_assets),
                        "risk_flags": risk_flags,
                    },
                },
            },
            "audit_trace": {
                "source": strategy.get("meta", {}).get("source", "rl_panel_service"),
                "generated_at": _now_utc().isoformat(),
                "assumptions": {
                    "carbon_intensity_g_per_kwh": round(ci_g, 2),
                    "mode": "forecast + heuristic simulate",
                },
            },
        }


try:
    from app.services.rl_suite.rl import RLService
except Exception:
    RLService = None

try:
    from app.services.rl_suite.rl_rollout import RLPolicyRollout
except Exception:
    RLPolicyRollout = None


class RLPanelProAdapter:
    def __init__(self, rl_service: Optional[RLService] = None, rollout: Optional[RLPolicyRollout] = None):
        if rl_service is not None:
            self.rl = rl_service
        else:
            if RLService is None:
                raise RuntimeError("RLService 未可用，请检查 app/services/rl_suite/rl.py")
            self.rl = RLService()
        self.rollout = rollout if rollout is not None else (RLPolicyRollout() if RLPolicyRollout is not None else None)

    def propose(self, state: Dict[str, Any], objective: str = "cost") -> Dict[str, Any]:
        return self.rl.propose_actions(state, objective=objective)

    def simulate(self, state: Dict[str, Any], actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return self.rl.simulate_with_envpro(state, actions)

    def suggest_and_simulate(self, state: Dict[str, Any], objective: str = "cost") -> Dict[str, Any]:
        suggest_res = self.propose(state, objective=objective)
        actions = [
            {"asset": a.get("asset"), "cmd": a.get("cmd"), "kW": float(a.get("kW", 0.0))}
            for a in (suggest_res.get("actions") or [])
        ]
        sim_res = self.simulate(state, actions)
        return {"suggest": suggest_res, "simulate": sim_res}

    def rollout_status(self) -> Dict[str, Any]:
        if not self.rollout:
            return {"error": "rollout module not available"}
        return self.rollout.get_status()

    def rollout_register(self, version: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.rollout:
            return {"error": "rollout module not available"}
        return self.rollout.register_candidate(version, meta=meta or {})

    def rollout_start_shadow(self) -> Dict[str, Any]:
        if not self.rollout:
            return {"error": "rollout module not available"}
        return self.rollout.start_shadow()

    def rollout_start_canary(self, pct: float = 0.1) -> Dict[str, Any]:
        if not self.rollout:
            return {"error": "rollout module not available"}
        return self.rollout.promote_to_canary(traffic_pct=pct)

    def rollout_step_canary(self, increment: float = 0.2) -> Dict[str, Any]:
        if not self.rollout:
            return {"error": "rollout module not available"}
        return self.rollout.step_canary(increment=increment)

    def rollout_promote_full(self) -> Dict[str, Any]:
        if not self.rollout:
            return {"error": "rollout module not available"}
        return self.rollout.promote_to_full()

    def rollout_rollback(self) -> Dict[str, Any]:
        if not self.rollout:
            return {"error": "rollout module not available"}
        return self.rollout.rollback()

    def rollout_ingest_metrics(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.rollout:
            return {"error": "rollout module not available"}
        return self.rollout.ingest_batch_metrics(batch)


try:
    _PANEL_PRO = RLPanelProAdapter()
except Exception:
    _PANEL_PRO = None


def propose_v2(state: Dict[str, Any], objective: str = "cost") -> Dict[str, Any]:
    if _PANEL_PRO is None:
        raise RuntimeError("RLPanelProAdapter 未就绪（检查 rl/env/rollout 依赖）")
    return _PANEL_PRO.propose(state, objective)


def simulate_pro(state: Dict[str, Any], actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if _PANEL_PRO is None:
        raise RuntimeError("RLPanelProAdapter 未就绪（检查 rl/env/rollout 依赖）")
    return _PANEL_PRO.simulate(state, actions)


def suggest_and_simulate_v2(state: Dict[str, Any], objective: str = "cost") -> Dict[str, Any]:
    if _PANEL_PRO is None:
        raise RuntimeError("RLPanelProAdapter 未就绪（检查 rl/env/rollout 依赖）")
    return _PANEL_PRO.suggest_and_simulate(state, objective)


def rollout_status() -> Dict[str, Any]:
    if _PANEL_PRO is None:
        raise RuntimeError("RLPanelProAdapter 未就绪（检查 rl/env/rollout 依赖）")
    return _PANEL_PRO.rollout_status()


def rollout_command(cmd: str, **kwargs) -> Dict[str, Any]:
    if _PANEL_PRO is None:
        raise RuntimeError("RLPanelProAdapter 未就绪（检查 rl/env/rollout 依赖）")
    cmd = (cmd or "").lower()
    if cmd == "register":
        return _PANEL_PRO.rollout_register(kwargs.get("version", "policy-vX"), meta=kwargs.get("meta", {}))
    if cmd == "shadow":
        return _PANEL_PRO.rollout_start_shadow()
    if cmd == "canary":
        return _PANEL_PRO.rollout_start_canary(float(kwargs.get("pct", 0.1)))
    if cmd == "step":
        return _PANEL_PRO.rollout_step_canary(float(kwargs.get("increment", 0.2)))
    if cmd == "full":
        return _PANEL_PRO.rollout_promote_full()
    if cmd == "rollback":
        return _PANEL_PRO.rollout_rollback()
    if cmd == "status":
        return _PANEL_PRO.rollout_status()
    if cmd == "ingest":
        return _PANEL_PRO.rollout_ingest_metrics(kwargs.get("batch") or [])
    return {"error": f"unknown cmd: {cmd}"}
