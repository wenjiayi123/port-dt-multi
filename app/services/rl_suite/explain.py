# ============================================
# app/services/explain.py
# --------------------------------------------
# 策略可解释性服务（特征重要性、SHAP 简版）
#
# 设计目标：
#   - 不依赖训练好的黑箱模型，走“启发式 + 有限差分”的可解释路径；
#   - 对每条策略抽取一组可理解的上下文特征（时段/峰平谷/范围/资产构成/平均负荷/碳因子等）；
#   - 通过显式打分函数 score(feats) 估计“节电潜力（ΔkWh，负值越好）”；
#   - 做 SHAP-like：对每个特征做±扰动，观察 score 变化，得到“贡献（kWh）与重要度”；
#   - 输出结构化结果：features[], rankings[], reasons[]，便于前端画**特征重要性条形图**与展示说明。
#
# 注意：
#   - 这里的“SHAP 简版”并非严格 Shapley 值，而是“局部有限差分近似”，用于产品化解释展示。
#   - 结果用于**解释**与**对比**，非精确计量；真实口径仍以 /api/rl/simulate 的基线/策略后结果为准。
#
# 依赖（通过 DI 注入）：
#   - telemetry（获取近窗功率，兜底资产列表）
#   - forecast（获取未来负荷预测）
#   - reporting（获取 P95、碳因子等）
#   - energy（获取整体口径，如平均碳因子）
#   - rl / twin：可选，用于扩展丰富解释（当前版本未强耦合）
#
# 对外 API（给 server.py 使用）：
#   - explain(strategy: dict, horizon_min=360, step_min=1) -> dict
#     返回：{
#       "strategy_id": "...",
#       "features": [{"name": "...", "value": any, "contribution_kWh": float, "importance": float, "direction": "saving|worsen"} ...],
#       "rankings": [{"name":"...", "importance":0.23}, ...],  # 重要度归一化
#       "reasons": ["...", "..."],                               # 人类可读解释
#       "meta": {"window": {...}, "scope_size": int, "avg_load_kw": float, ...}
#     }
#   - explain_many(strategies: list[dict], ...) -> list[dict]
#
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import math
import copy
import statistics


# ------------------------
# 小工具
# ------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return d
    return v


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _tou_bucket(hour: float) -> str:
    """峰/平/谷划分（与 server.py 一致）：峰[10-15,19-21)、谷[23-7)、其余平"""
    if (10 <= hour < 15) or (19 <= hour < 21):
        return "peak"
    if hour >= 23 or hour < 7:
        return "valley"
    return "flat"


# ------------------------
# 数据抽取器
# ------------------------
@dataclass
class FeatureVector:
    """用于打分与解释的特征集合"""
    # 时间窗
    window_duration_min: float
    start_hour_local: float
    end_hour_local: float
    is_peak_window: float  # {0,1} 或 [0,1]：窗口内“峰时段”占比

    # 作用范围
    scope_size: int
    cnt_qc: int
    cnt_yc: int
    cnt_agv: int
    cnt_wh: int
    cnt_cs: int
    cnt_ps: int
    cnt_yard: int

    # 负荷与碳因子
    avg_load_kw: float              # 窗口内/或近窗平均功率（聚合）
    avg_carbon_intensity_gpkwh: float  # 区域电网碳强度估算

    # 电价代理（无电价数据时，用峰平谷映射）
    price_proxy: float  # valley≈0.7, flat≈1.0, peak≈1.3

    # 策略动作强度（启发式）
    action_intensity: float  # 0~1，按动作类型估个强度系数
    lighting_dim_ratio: float  # 照明调光比例（若有）
    setpoint_shift_degC: float  # 设定点上调（正值代表省电）

    # 其他
    explain_basis: Dict[str, Any]


class ExplainService:
    """
    可解释性服务主体。
    """

    def __init__(self, telemetry, forecast, reporting, energy, rl=None, twin=None):
        self.telemetry = telemetry
        self.forecast = forecast
        self.reporting = reporting
        self.energy = energy
        self.rl = rl
        self.twin = twin

    # =======================
    # 1) 对外：主解释入口
    # =======================
    def explain(self, strategy: Dict[str, Any], horizon_min: int = 360, step_min: int = 1) -> Dict[str, Any]:
        """
        生成某条策略的“特征重要度 + SHAP-like 贡献 + 可读解释”。
        """
        fv = self._build_features(strategy, horizon_min=horizon_min, step_min=step_min)
        base = self._score(fv)  # ΔkWh 估计（负值越好）
        contrib = self._shap_like(fv, base)  # {"name":..., "contribution_kWh":..., "importance":..., "direction":...}

        # 排序 & 归一化重要度
        feats = sorted(contrib, key=lambda x: abs(x["contribution_kWh"]), reverse=True)
        total_imp = sum(abs(x["contribution_kWh"]) for x in feats) or 1.0
        for x in feats:
            x["importance"] = abs(x["contribution_kWh"]) / total_imp

        # 生成 reasons
        reasons = self._craft_reasons(fv, feats, base)

        return {
            "generated_at": _now_iso(),
            "strategy_id": str(strategy.get("id") or ""),
            "features": feats,
            "rankings": [{"name": x["name"], "importance": x["importance"]} for x in feats],
            "reasons": reasons,
            "meta": {
                "window": strategy.get("window") or {},
                "scope_size": fv.scope_size,
                "avg_load_kw": round(fv.avg_load_kw, 3),
                "avg_carbon_intensity_gpkwh": round(fv.avg_carbon_intensity_gpkwh, 1),
                "price_proxy": round(fv.price_proxy, 2),
                "action_intensity": round(fv.action_intensity, 3),
                "baseline_delta_kWh_est": round(base, 3),  # 负值代表预计节电
            },
        }

    def explain_many(self, strategies: List[Dict[str, Any]], horizon_min: int = 360, step_min: int = 1) -> List[Dict[str, Any]]:
        out = []
        for s in strategies or []:
            try:
                out.append(self.explain(s, horizon_min=horizon_min, step_min=step_min))
            except Exception:
                continue
        return out

    # =======================
    # 2) 特征构建
    # =======================
    def _build_features(self, strategy: Dict[str, Any], horizon_min: int, step_min: int) -> FeatureVector:
        window = strategy.get("window") or {}
        w_start = window.get("start")
        w_end = window.get("end")

        # 窗口时长（分钟）
        try:
            # 这里不做时区转换，保持 ISO 字符串差值近似（演示）
            dt1 = datetime.fromisoformat(w_start.replace("Z", "+00:00")) if isinstance(w_start, str) else datetime.now(timezone.utc)
            dt2 = datetime.fromisoformat(w_end.replace("Z", "+00:00")) if isinstance(w_end, str) else (dt1 + timedelta(hours=1))
            dur_min = max(1.0, (dt2 - dt1).total_seconds() / 60.0)
        except Exception:
            dt1 = datetime.now(timezone.utc)
            dt2 = dt1 + timedelta(hours=1)
            dur_min = 60.0

        # 起止小时（本地时间近似：直接取 UTC 小时）
        h1 = dt1.hour + dt1.minute / 60.0
        h2 = dt2.hour + dt2.minute / 60.0

        # 峰时段占比（粗略：若窗口中心位于峰，视为 1；谷为 0；平为 0.5）
        mid_h = (h1 + h2) / 2.0
        bkt = _tou_bucket(mid_h)
        is_peak = 1.0 if bkt == "peak" else (0.0 if bkt == "valley" else 0.5)

        # 作用范围（资产列表）
        scope = strategy.get("scope") or {}
        asset_ids = scope.get("asset_ids") or []
        scope_size = len(asset_ids)

        # 资产类别计数
        def _cls(aid: str) -> str:
            s = (aid or "").lower()
            if s.startswith("qc"): return "qc"
            if s.startswith("yc"): return "yc"
            if s.startswith("agv"): return "agv"
            if s.startswith("wh"): return "wh"
            if s.startswith("cs"): return "cs"
            if s.startswith("ps"): return "ps"
            if s.startswith("yard"): return "yard"
            return "misc"

        cnt = {"qc":0,"yc":0,"agv":0,"wh":0,"cs":0,"ps":0,"yard":0}
        for a in asset_ids:
            t = _cls(a)
            if t in cnt: cnt[t] += 1

        # 预测/近窗负荷（聚合）
        avg_kw = self._estimate_avg_load(asset_ids, horizon_min=horizon_min, step_min=step_min)
        # 区域碳强度（g/kWh）
        avg_ci = self._estimate_avg_carbon_intensity(asset_ids)
        # 电价代理（峰1.3、平1.0、谷0.7）
        pp = 1.3 if bkt == "peak" else (0.7 if bkt == "valley" else 1.0)

        # 动作强度（启发式）：根据 actions 判断
        actions = strategy.get("actions") or []
        act_intensity = 0.0
        lighting_dim = 0.0
        setpoint_shift = 0.0
        for a in actions:
            cmd = str(a.get("cmd") or "")
            pct = _safe_float(a.get("percent"), 0.0)  # e.g., 调光/负荷比例
            if cmd in ("idle", "reduce"):
                act_intensity += max(0.1, pct)  # reduce/idle 视为较强动作
            elif cmd in ("charge", "discharge", "shore_power"):
                act_intensity += 0.15
            elif cmd == "lighting_dim":
                lighting_dim = max(lighting_dim, pct)
                act_intensity += 0.05 + pct
            elif cmd == "setpoint":
                # 设定点上调（制冷侧）：越高越省电
                setpoint_shift = max(setpoint_shift, _safe_float(a.get("delta_degC"), 0.0))
                act_intensity += 0.05 + 0.05 * (setpoint_shift > 0)

        act_intensity = min(1.0, act_intensity)

        basis = {
            "bucket": bkt,
            "asset_ids": asset_ids,
            "avg_load_kw_method": "forecast>reporting>fallback",
        }

        return FeatureVector(
            window_duration_min = float(dur_min),
            start_hour_local = float(h1),
            end_hour_local = float(h2),
            is_peak_window = float(is_peak),

            scope_size = int(scope_size),
            cnt_qc = cnt["qc"],
            cnt_yc = cnt["yc"],
            cnt_agv = cnt["agv"],
            cnt_wh = cnt["wh"],
            cnt_cs = cnt["cs"],
            cnt_ps = cnt["ps"],
            cnt_yard = cnt["yard"],

            avg_load_kw = float(avg_kw),
            avg_carbon_intensity_gpkwh = float(avg_ci),

            price_proxy = float(pp),

            action_intensity = float(act_intensity),
            lighting_dim_ratio = float(lighting_dim),
            setpoint_shift_degC = float(setpoint_shift),

            explain_basis = basis
        )

    def _estimate_avg_load(self, asset_ids: List[str], horizon_min: int, step_min: int) -> float:
        """估算窗口/近窗聚合平均功率（kW），forecast 优先，回退到 telemetry/reporting。"""
        if asset_ids:
            # 尝试 forecast（取各资产 6h 平均）：
            try:
                total = 0.0
                count = 0
                for aid in asset_ids:
                    fmap = self.forecast.forecast_load([aid], horizon_min=horizon_min, step_min=step_min) or {}
                    arr = fmap.get(aid, [])
                    vals = [_safe_float(p.get("kW"), 0.0) for p in arr if isinstance(p, dict)]
                    if vals:
                        total += sum(vals) / len(vals)
                        count += 1
                if count > 0:
                    return total
            except Exception:
                pass

            # 回退：recent power 均值
            try:
                total = 0.0
                count = 0
                for aid in asset_ids:
                    pts = self.telemetry.get_recent_power(aid) or []
                    vals = [_safe_float(p.get("kW"), 0.0) for p in pts if isinstance(p, dict)]
                    if vals:
                        total += sum(vals) / len(vals)
                        count += 1
                if count > 0:
                    return total
            except Exception:
                pass

        # 最兜底：常数
        return 50.0

    def _estimate_avg_carbon_intensity(self, asset_ids: List[str]) -> float:
        """估计区域平均碳强度（g/kWh）。优先 reporting/energy。"""
        # reporting 平均
        vals = []
        try:
            for aid in (asset_ids or [])[:12]:
                r = self.reporting.generate_mini_report(aid) or {}
                ci = _safe_float(r.get("carbonIntensity"), float("nan"))
                if math.isfinite(ci):
                    vals.append(ci)
        except Exception:
            pass
        if vals:
            return sum(vals) / len(vals)

        # energy 兜底
        try:
            summary = self.energy.build_today_summary(teu=12000, limit_assets=50)
            elec = summary.get("electricity", {})
            v = _safe_float(elec.get("avg_carbon_intensity_g_per_kwh"), float("nan"))
            if math.isfinite(v):
                return v
        except Exception:
            pass

        # 默认 120 g/kWh
        return 120.0

    # =======================
    # 3) 打分函数（显式）
    # =======================
    def _score(self, fv: FeatureVector) -> float:
        """
        估计“ΔkWh（负值代表节电）”。这是一个**可解释**的启发式模型：
          - 基础节电潜力 ~ 平均负荷 * 窗口小时 * 动作强度
          - 峰时段（价格代理高/峰占比高）乘以削峰收益系数
          - 照明调光与设定点上移单独贡献
          - 按资产类别（岸桥/场桥/AGV/仓库/充电/配电/堆场）给不同的削减潜力权重
        """
        hours = fv.window_duration_min / 60.0
        base_potential = fv.avg_load_kw * hours * fv.action_intensity  # kWh

        # 峰平谷影响（峰时收益更高）
        tou_mult = 0.8 + 0.6 * fv.is_peak_window + 0.3 * (fv.price_proxy - 1.0)  # ≈ [0.5, ~1.5]
        tou_mult = max(0.4, min(1.8, tou_mult))

        # 类别权重（基于设备对总负荷影响力的经验数）
        w_qc, w_yc, w_agv, w_wh, w_cs, w_ps, w_yard = 1.2, 1.0, 0.9, 0.8, 1.1, 0.4, 0.5
        class_mult = (
            w_qc * fv.cnt_qc + w_yc * fv.cnt_yc + w_agv * fv.cnt_agv +
            w_wh * fv.cnt_wh + w_cs * fv.cnt_cs + w_ps * fv.cnt_ps + w_yard * fv.cnt_yard
        ) / max(1, fv.scope_size or 1)

        # 照明调光 & 设定点（仅正向贡献）
        lighting_saving = fv.avg_load_kw * hours * (0.15 * fv.lighting_dim_ratio)
        setpoint_saving = fv.avg_load_kw * hours * (0.05 * max(0.0, fv.setpoint_shift_degC))

        # 综合“节电潜力”估计（负号代表节电）
        delta_kwh = - (base_potential * tou_mult * class_mult + lighting_saving + setpoint_saving)

        return float(delta_kwh)

    # =======================
    # 4) SHAP-like 有限差分
    # =======================
    def _shap_like(self, fv: FeatureVector, base_delta_kwh: float) -> List[Dict[str, Any]]:
        """
        对每个特征做小幅扰动，观察 score 变化：
          contribution = score(perturbed) - score(base)
        约定：负贡献（<0）代表**更节电**，方向记为 "saving"；正值代表“更糟”，方向记为 "worsen"。
        """
        def pack(name: str, value: Any, delta: float) -> Dict[str, Any]:
            return {
                "name": name,
                "value": value,
                "contribution_kWh": round(float(delta), 4),
                "direction": "saving" if delta < 0 else ("worsen" if delta > 0 else "neutral"),
                "importance": 0.0,  # 稍后归一化
            }

        res: List[Dict[str, Any]] = []

        # 选择一组“可解释、稳定”的特征进行扰动
        feats: List[Tuple[str, Any, Any]] = [
            ("is_peak_window", fv.is_peak_window, min(1.0, fv.is_peak_window + 0.25)),
            ("price_proxy", fv.price_proxy, fv.price_proxy + 0.2),
            ("scope_size", fv.scope_size, max(1, fv.scope_size + max(1, int(0.2 * (fv.scope_size or 1))))),
            ("avg_load_kw", fv.avg_load_kw, max(1e-6, fv.avg_load_kw * 1.1)),
            ("lighting_dim_ratio", fv.lighting_dim_ratio, min(1.0, fv.lighting_dim_ratio + 0.2)),
            ("setpoint_shift_degC", fv.setpoint_shift_degC, fv.setpoint_shift_degC + 1.0),
            ("action_intensity", fv.action_intensity, min(1.0, fv.action_intensity + 0.2)),
        ]
        # 类别计数：整体+1 的等效扰动（模拟“更多该类设备受控”）
        feats += [
            ("cnt_qc", fv.cnt_qc, fv.cnt_qc + 1),
            ("cnt_yc", fv.cnt_yc, fv.cnt_yc + 1),
            ("cnt_agv", fv.cnt_agv, fv.cnt_agv + 1),
            ("cnt_wh", fv.cnt_wh, fv.cnt_wh + 1),
            ("cnt_cs", fv.cnt_cs, fv.cnt_cs + 1),
            ("cnt_ps", fv.cnt_ps, fv.cnt_ps + 1),
            ("cnt_yard", fv.cnt_yard, fv.cnt_yard + 1),
        ]

        # 对每个特征进行一次单变量扰动
        for name, orig, newv in feats:
            fv2 = copy.deepcopy(fv)
            setattr(fv2, name, newv)
            # 维护 scope_size 与各类计数的联动关系（简单近似）
            if name.startswith("cnt_"):
                fv2.scope_size = max(fv2.scope_size, int(getattr(fv2, name)))
            # 重新打分
            s2 = self._score(fv2)
            delta = s2 - base_delta_kwh  # <0 => 更节电
            res.append(pack(name, orig, delta))

        return res

    # =======================
    # 5) 生成可读解释
    # =======================
    def _craft_reasons(self, fv: FeatureVector, feats: List[Dict[str, Any]], base_delta_kwh: float) -> List[str]:
        reasons: List[str] = []

        # 基本口径
        hours = fv.window_duration_min / 60.0
        reasons.append(
            f"窗口时长约 {hours:.1f} h，平均聚合负荷 ~{fv.avg_load_kw:.1f} kW，策略动作强度 ~{fv.action_intensity:.2f}。"
        )

        # 峰/平/谷
        bucket = fv.explain_basis.get("bucket", "flat")
        if bucket == "peak":
            reasons.append("窗口位于“峰时段”，削峰收益较高，解释权重更偏向峰价/峰段覆盖。")
        elif bucket == "valley":
            reasons.append("窗口位于“谷时段”，节电的边际收益较低（但合理的移峰/设定点仍有价值）。")
        else:
            reasons.append("窗口位于“平时段”，节电收益中等，可考虑优化作业与充放电时序。")

        # 类别构成
        comp = []
        if fv.cnt_qc: comp.append(f"岸桥×{fv.cnt_qc}")
        if fv.cnt_yc: comp.append(f"场桥×{fv.cnt_yc}")
        if fv.cnt_agv: comp.append(f"AGV×{fv.cnt_agv}")
        if fv.cnt_wh: comp.append(f"仓库×{fv.cnt_wh}")
        if fv.cnt_cs: comp.append(f"充电×{fv.cnt_cs}")
        if fv.cnt_ps: comp.append(f"配电×{fv.cnt_ps}")
        if fv.cnt_yard: comp.append(f"堆场×{fv.cnt_yard}")
        if comp:
            reasons.append("作用范围包含：" + "、".join(comp) + "。")

        # 照明/设定点
        if fv.lighting_dim_ratio > 0.0:
            reasons.append(f"包含照明调光，预计带来 {fv.lighting_dim_ratio*100:.0f}%×负荷×时长 的额外节电项。")
        if fv.setpoint_shift_degC > 0.0:
            reasons.append(f"包含设定点上调（{fv.setpoint_shift_degC:.1f}℃），对冷站/空调类负荷有直接节电贡献。")

        # Top-3 特征
        top3 = feats[:3]
        if top3:
            x = []
            for f in top3:
                name = f["name"]
                if name == "avg_load_kw": x.append("平均负荷")
                elif name == "is_peak_window": x.append("峰段覆盖")
                elif name == "price_proxy": x.append("电价（代理）")
                elif name == "action_intensity": x.append("动作强度")
                elif name.startswith("cnt_"): x.append(name.replace("cnt_", "").upper()+" 数量")
                elif name == "lighting_dim_ratio": x.append("照明调光比例")
                elif name == "setpoint_shift_degC": x.append("设定点上移")
                else: x.append(name)
            reasons.append("该策略的关键影响因子（Top-3）："+ "、".join(x) + "。")

        # 基线估计
        reasons.append(f"基于启发式估计，策略带来的 ΔkWh ≈ {base_delta_kwh:.1f}（负值代表节电；实际以仿真结果为准）。")

        return reasons
