# ============================================
# app/di.py
# --------------------------------------------
# 轻量依赖注入容器（Container）
#
# 本版要点（v2.1）：
#   1) 保持：telemetry / forecast / reporting / rl / twin
#   2) 保持：energy（指挥盘口径中心）/ alerts（统一预警）
#   3) 保持：rlpanel（策略列表+仿真）/ dispatch（策略下发·演示）
#   4) 保持：explain（策略可解释特征，SHAP-like）/ closedloop（执行与闭环）
#   5) ⭐ 新增：compliance（合规报表：月度/季度/通用） + factors(...)（排放因子构造器）
#
# 说明：
#  - 适配器缺失时保持接口可诊断，但默认失败关闭，不生成展示性业务结果。
#  - 你仍然可以保留旧版 twin.py，不影响本文件与其它服务的工作。
# ============================================

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
# （新增导入：消息总线 / 对象存储 / 碳因子服务）
from app.infra.message_bus import MessageBus
from app.infra.storage import ObjectStorage, StorageConfig
from app.adapters.carbon_factors import CarbonFactors


# -------------------------------------------------
# 1) Telemetry 适配（默认公开数据集回放；工程模拟需显式开启）
# -------------------------------------------------
def _init_telemetry():
    """
    需要暴露：
      - list_assets() -> [{"id":"qc-01","label":"QC-01"}, ...]
      - get_recent_power(asset_id) -> [{"ts":"...Z","kW":12.3}, ...]（时间升序）
    """
    engineering_sim = os.getenv("PORT_DT_ENABLE_ENGINEERING_SIMULATORS", "").strip().lower() in {"1", "true", "yes", "on"}
    if engineering_sim:
        try:
            from app.adapters.telemetry_sim import TelemetrySim  # type: ignore
            return TelemetrySim()
        except Exception:
            pass
    try:
        from app.adapters.telemetry_dataset import DatasetTelemetry
        return DatasetTelemetry()
    except Exception:
        class _TelemetryUnavailable:
            def list_assets(self):
                return []

            def get_recent_power(self, asset_id: str):
                return []

            def get_series(self, asset_id: str, point: str, start_ts: float, end_ts: float, step_sec: int = 60):
                return []

            def source_status(self):
                return {"mode": "unavailable", "production": False, "measured": False}

        return _TelemetryUnavailable()


# -------------------------------------------------
# 2) Forecast 适配
# -------------------------------------------------

def _init_forecast(telemetry) -> Any:
    """
    需要暴露：
      forecast_load(asset_ids, horizon_min=360, step_min=1, **kwargs)
      -> {asset_id: [{"ts":"...Z","kW":12.3}, ...], ...}

    口径说明：
      1) 优先使用 app.services.forecast（当前会直接替换的本地文件）
      2) 兼容路径 app.services.forecast_twin.forecast 仅重导出同一个实现
      3) 主实现不可用时返回空结果，不合成展示曲线

    这么做的目的：
      - 避免你已经替换了 app/services/forecast.py，但运行时仍被 forecast_twin 覆盖
      - 便于单文件迭代预测逻辑，减少“改了没生效”的困惑
    """
    # A. 优先走你当前直接替换的本地 forecast.py
    try:
        import app.services.forecast as mod  # type: ignore

        # 方案 A1：模块级函数 forecast_load(...)
        if hasattr(mod, "forecast_load"):
            class _ForecastModuleCompat:
                def __init__(self, m):
                    self._m = m

                def forecast_load(self, asset_ids, horizon_min=360, step_min=1, **kwargs):
                    return self._m.forecast_load(
                        asset_ids,
                        horizon_min=horizon_min,
                        step_min=step_min,
                        **kwargs,
                    )

            return _ForecastModuleCompat(mod)

        # 方案 A2：类 ForecastService(...)
        if hasattr(mod, "ForecastService"):
            svc = mod.ForecastService(telemetry=telemetry, schedule=_init_schedule_sources())

            class _ForecastServiceCompat:
                def __init__(self, s):
                    self._s = s

                def forecast_load(self, asset_ids, horizon_min=360, step_min=1, **kwargs):
                    return self._s.forecast_load(
                        asset_ids,
                        horizon_min=horizon_min,
                        step_min=step_min,
                        **kwargs,
                    )

            return _ForecastServiceCompat(svc)
    except Exception:
        pass

    class _ForecastUnavailable:
        def forecast_load(self, asset_ids, horizon_min=360, step_min=1, **kwargs):
            return {str(asset_id): [] for asset_id in asset_ids}

    return _ForecastUnavailable()


# -------------------------------------------------
# 3) Reporting 适配（mini 报表）
# -------------------------------------------------
def _init_reporting(telemetry) -> Any:
    """
    需要暴露：
      generate_mini_report(asset_id) -> {
        "avg_kW_last5min": 12.3,
        "p95_kW": 30.1,
        "carbonIntensity": 118.0
      }
    """
    try:
        from app.services.energy_reporting.reporting import ReportingService  # type: ignore
        return ReportingService(telemetry=telemetry)
    except Exception:
        try:
            import app.services.reporting as mod  # type: ignore

            class _ReportingCompat:
                def __init__(self, m, telem):
                    self._m, self._t = m, telem

                def generate_mini_report(self, asset_id: str) -> Dict[str, float]:
                    if hasattr(self._m, "generate_mini_report"):
                        return self._m.generate_mini_report(asset_id)
                    pts = self._t.get_recent_power(asset_id) or []
                    vals = [float(p.get("kW", 0.0)) for p in pts if isinstance(p, dict)]
                    if not vals:
                        return {"available": False, "avg_kW_last5min": None, "p95_kW": None, "carbonIntensity": None}
                    avg = sum(vals) / len(vals)
                    p95 = sorted(vals)[int(0.95 * (len(vals) - 1))]
                    return {"available": True, "avg_kW_last5min": round(avg, 3), "p95_kW": round(p95, 3), "carbonIntensity": None}

            return _ReportingCompat(mod, telemetry)
        except Exception:
            class _ReportingFallback:
                def __init__(self, telem):
                    self._t = telem

                def generate_mini_report(self, asset_id: str) -> Dict[str, float]:
                    pts = self._t.get_recent_power(asset_id) or []
                    vals = [float(p.get("kW", 0.0)) for p in pts if isinstance(p, dict)]
                    if not vals:
                        return {"available": False, "avg_kW_last5min": None, "p95_kW": None, "carbonIntensity": None}
                    avg = sum(vals) / len(vals)
                    p95 = sorted(vals)[int(0.95 * (len(vals) - 1))]
                    return {"available": True, "avg_kW_last5min": round(avg, 3), "p95_kW": round(p95, 3), "carbonIntensity": None}

            return _ReportingFallback(telemetry)


# -------------------------------------------------
# 4) RL 适配（动作建议）
# -------------------------------------------------
def _init_rl() -> Any:
    """
    需要暴露：
      propose_actions(state: dict, objective="cost") -> dict
    """
    class _RLTrainingRequired:
        def propose_actions(self, state: Dict[str, Any], objective: str = "cost") -> Dict[str, Any]:
            return {
                "available": False,
                "objective": objective,
                "actions": [],
                "reason": "Use /api/rl/train/{job_id}/predict with a completed real training job",
            }

    return _RLTrainingRequired()
# -------------------------------------------------
# 4.5) RLSafety 适配（策略守护/硬性红线）
# -------------------------------------------------
def _init_rlsafety(telemetry, storage=None) -> Any:
    try:
        from app.services.exec_closedloop.rl_safety import RLSafetyGuard  # type: ignore
        return RLSafetyGuard(telemetry=telemetry, storage=storage)
    except Exception:
        class _SafetyFallback:
            def validate_and_shield(self, strategy, **kwargs):
                return {"ok": False, "rules": [{"rule":"safety_unavailable","passed":False,"detail":"rl_safety 未启用，默认拒绝执行"}],
                        "actions_after_shield": [], "peak_check": {}, "evidence_path": None}
        return _SafetyFallback()


# -------------------------------------------------
# 5) Twin 适配（策略仿真）
# -------------------------------------------------
def _init_twin(forecast, telemetry=None) -> Any:
    """
    需要暴露：
      run(asset_id) -> {"asset":"...", "summary": {...}, "plan": [...]}
    """
    try:
        from app.services.forecast_twin.twin import TwinService  # type: ignore
        return TwinService(fcst=forecast, telemetry=telemetry)
    except Exception:
        class _TwinUnavailable:
            def run(self, asset_id: str, **kwargs):
                return {
                    "available": False,
                    "asset": asset_id,
                    "reason": "twin service failed to initialize",
                    "plan": [],
                    "summary": {},
                    "_source": "twin_unavailable",
                }

        return _TwinUnavailable()
# -------------------------------------------------
# 2.5) Schedule / External Sources 适配（TOS/AIS/天气/电价/碳价）
# -------------------------------------------------
def _init_schedule_sources() -> Any:
    """
    需要暴露（全部为只读）：
      - load_drivers(start: ISO8601, end: ISO8601, port_code: str, assets: List[str]) -> dict
      - weather(start: ISO8601, end: ISO8601, lat: float, lon: float) -> List[dict]
      - tide(start: ISO8601, end: ISO8601, port_code: str) -> List[dict]
      - vessels(start: ISO8601, end: ISO8601, port_code: str) -> List[dict]
      - tou_tariff(date: str, port_code: str) -> List[dict]

    真实落地：若存在 app.adapters.schedule_sources.ScheduleSources 则优先用；
    否则回退到 _ScheduleFallback（返回空列表/空字典，不阻塞其它功能）。
    """
    try:
        # 如果下一步你把 schedule_sources.py 放到了 app/adapters/ 下，这里会自动加载
        from app.adapters.schedule_sources import ScheduleSources  # type: ignore
        return ScheduleSources()
    except Exception:
        class _ScheduleFallback:
            def load_drivers(self, start: str, end: str, port_code: str = "CN_DEMO", assets=None) -> dict:
                # 兜底：不提供驱动（预测仍可跑，只是没有作业/天气增益）
                return {}
            def weather(self, start: str, end: str, lat: float, lon: float) -> list:
                return []
            def tide(self, start: str, end: str, port_code: str) -> list:
                return []
            def vessels(self, start: str, end: str, port_code: str) -> list:
                return []
            def tou_tariff(self, date: str, port_code: str) -> list:
                return []
        return _ScheduleFallback()


# -------------------------------------------------
# 6) Energy 适配（能耗与碳排“口径中心”）
# -------------------------------------------------
def _init_energy(telemetry, reporting, forecast) -> Any:
    """
    需要暴露：
      build_today_summary(teu=12000, limit_assets=50, ...) -> dict
    """
    try:
        from app.services.energy_reporting.energy import EnergyService  # type: ignore
        return EnergyService(telemetry=telemetry, reporting=reporting, forecast=forecast)
    except Exception:
        class _EnergyFallback:
            def __init__(self, telem, rpt, fcst):
                self._t, self._r, self._f = telem, rpt, fcst

            def build_today_summary(self, teu: int = 12000, limit_assets: int = 50, **kwargs) -> Dict[str, Any]:
                return {
                    "available": False,
                    "reason": "EnergyService failed to initialize",
                    "electricity": {},
                    "intensity": {},
                }

        return _EnergyFallback(telemetry, reporting, forecast)


# -------------------------------------------------
# 7) Alerts 适配（统一预警）
# -------------------------------------------------
def _init_alerts(telemetry, reporting, forecast, energy) -> Any:
    """
    需要暴露：
      scan(teu=12000, demand_limit_kw=500, quota_kgco2e=5000, limit=50, horizon_min=360, step_min=1) -> dict
    """
    try:
        from app.services.monitoring_quality.alerts import AlertsService  # type: ignore
        return AlertsService(telemetry=telemetry, reporting=reporting, forecast=forecast, energy=energy)
    except Exception:
        class _AlertsFallback:
            def __init__(self, telem, rpt, fcst, energy):
                self._t, self._r, self._f, self._e = telem, rpt, fcst, energy

            def scan(self, teu=12000, demand_limit_kw=500.0, quota_kgco2e=5000.0, limit=50, horizon_min=360, step_min=1):
                return {
                    "available": False,
                    "reason": "AlertsService failed to initialize",
                    "params": {
                        "teu": teu, "demand_limit_kw": demand_limit_kw, "quota_kgco2e": quota_kgco2e,
                        "limit": limit, "horizon_min": horizon_min, "step_min": step_min
                    },
                    "summary": {},
                    "alerts": []
                }
        return _AlertsFallback(telemetry, reporting, forecast, energy)


# -------------------------------------------------
# 8) RLPanel 适配（策略面板：列表 + 模拟）
# -------------------------------------------------
def _init_rlpanel(telemetry, forecast, reporting, energy, rl, twin=None) -> Any:
    """
    需要暴露：
      - list_strategies(horizon_min=360, step_min=5, max_items=8) -> dict
      - simulate(strategy: dict, horizon_min=360, step_min=1) -> dict
    """
    legacy_enabled = False  # unaudited legacy RL panel is not part of this distribution
    try:
        if not legacy_enabled:
            raise RuntimeError("legacy RL panel disabled")
        from app.services.rl_suite.rl_panel import RLPanelService  # type: ignore
        return RLPanelService(
            telemetry=telemetry,
            forecast=forecast,
            reporting=reporting,
            energy=energy,
            rl=rl,
            twin=twin
        )
    except Exception:
        class _RLPanelFallback:
            def __init__(self, telem, fcst, rpt, energy, rl, twin):
                self._t, self._f, self._r, self._e, self._rl, self._tw = telem, fcst, rpt, energy, rl, twin

            def list_strategies(self, horizon_min: int = 360, step_min: int = 5, max_items: int = 8) -> Dict[str, Any]:
                return {"available": False, "generated_at": datetime.now(timezone.utc).isoformat(), "avg_grid_ci_g_per_kwh": None, "strategies": []}

            def simulate(self, strategy: Dict[str, Any], horizon_min: int = 360, step_min: int = 1) -> Dict[str, Any]:
                return {
                    "available": False,
                    "reason": "RLPanelService failed to initialize",
                    "strategy_id": strategy.get("id", ""),
                    "summary": {},
                    "baseline": {"agg_kW": [], "total_kWh": 0.0},
                    "simulated": {"agg_kW": [], "total_kWh": 0.0},
                }
        return _RLPanelFallback(telemetry, forecast, reporting, energy, rl, twin)


# -------------------------------------------------
# 9) Dispatch 适配（策略下发 · 演示）
# -------------------------------------------------
def _init_dispatch(telemetry, rlpanel, twin=None, rlsafety=None) -> Any:

    """
    需要暴露：
      - validate_strategy(strategy) -> dict {ok, errors, warnings}
      - estimate_effect(strategy, ...) -> dict {ok, summary{...}}
      - dispatch(strategy, operator="system", dry_run=True, ...) -> dict  # 仅演示/干跑
      - list_history(limit=50) -> dict
      - cancel(job_id) -> dict
    """
    try:
        from app.services.exec_closedloop.dispatch import DispatchService  # type: ignore
        return DispatchService(telemetry=telemetry, rlpanel=rlpanel, twin=twin)
    except Exception:
        # 兜底：仅回显/记录最小信息，确保接口不报错
        class _DispatchFallback:
            def __init__(self, rlsafety=None):  # 保存来自容器的安全守护器
                self._hist: List[Dict[str, Any]] = []
                self._rlsafety = rlsafety

                self._hist: List[Dict[str, Any]] = []

            def validate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
                return {"ok": isinstance(strategy, dict) and bool(strategy.get("id")), "errors": [] if strategy.get("id") else ["缺少 id"], "warnings": []}

            def estimate_effect(self, strategy: Dict[str, Any], **kwargs) -> Dict[str, Any]:
                return {"ok": False, "available": False, "reason": "dispatch estimator unavailable", "summary": {}}

            def dispatch(self, strategy: Dict[str, Any], operator="system", dry_run=True, **kwargs) -> Dict[str, Any]:
                # —— 新增守护：若传入 enforce_guardrails=True，就调用 rlsafety
                enforce = bool(kwargs.get("enforce_guardrails", True))
                if enforce:
                    try:
                        # 容器层把 rlsafety 作为 _init_dispatch 的参数传进来了
                        guard = self._rlsafety  # 直接用在 __init__ 里保存的 rlsafety
                    except Exception:
                        guard = None
                    if guard:
                        g = guard.validate_and_shield(strategy, enforce_guardrails=True,
                                                      horizon_min=kwargs.get("horizon_min", 60),
                                                      step_min=kwargs.get("step_min", 1))
                        if not g.get("ok", True):
                            return {"job_id": None, "status": "BLOCKED_BY_GUARD", "guardrails": g, "strategy": strategy}
                        # 将守护后动作用于后续记录
                        strategy = dict(strategy)
                        strategy["actions"] = g.get("actions_after_shield", strategy.get("actions", []))
                        guardrails_info = g
                    else:
                        return {"job_id": None, "status": "BLOCKED_BY_GUARD", "reason": "rlsafety not injected"}
                else:
                    return {"job_id": None, "status": "BLOCKED_BY_GUARD", "reason": "guardrails cannot be skipped in fallback dispatch"}

                rec = {
                    "job_id": f"fallback-{len(self._hist) + 1}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "operator": operator,
                    "dry_run": dry_run,
                    "status": "DRY_RUN_RECORDED",
                    "strategy_id": strategy.get("id", ""),
                    "strategy": strategy,
                    "guardrails": guardrails_info,  # 保留这一处即可（删除原先那个占位的 guardrails）
                    "estimate": {"delta_kWh": 0.0, "delta_carbon_kg": 0.0, "peak_reduction_kW": 0.0},
                    "notes": "dispatch fallback",
                }

                self._hist.append(rec)
                return rec

            def list_history(self, limit: int = 50) -> Dict[str, Any]:
                return {"total": len(self._hist), "items": self._hist[::-1][:max(1, int(limit))]}

            def cancel(self, job_id: str, operator: str = "system") -> Dict[str, Any]:
                return {"ok": True, "job_id": job_id, "status": "CANCELLED"}
        return _DispatchFallback(rlsafety=rlsafety)



# -------------------------------------------------
# 10) Explain 适配（策略可解释性：特征重要度 & SHAP-like）
# -------------------------------------------------
def _init_explain(telemetry, forecast, reporting, energy, rl=None, twin=None) -> Any:
    """
    需要暴露：
      - explain(strategy, horizon_min=360, step_min=1) -> dict
      - explain_many(strategies, ...) -> list[dict]
    """
    try:
        from app.services.rl_suite.explain import ExplainService  # type: ignore
        return ExplainService(telemetry=telemetry, forecast=forecast, reporting=reporting, energy=energy, rl=rl, twin=twin)
    except Exception:
        class _ExplainFallback:
            def __init__(self, telem, fcst, rpt, energy):
                self._t, self._f, self._r, self._e = telem, fcst, rpt, energy

            def explain(self, strategy: Dict[str, Any], horizon_min: int = 360, step_min: int = 1) -> Dict[str, Any]:
                return {
                    "available": False,
                    "reason": "ExplainService failed to initialize",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "strategy_id": strategy.get("id", ""),
                    "features": [],
                    "rankings": [],
                    "reasons": [],
                    "meta": {"window": strategy.get("window", {}), "scope_size": len((strategy.get("scope") or {}).get("asset_ids", []))},
                }

            def explain_many(self, strategies: List[Dict[str, Any]], horizon_min: int = 360, step_min: int = 1) -> List[Dict[str, Any]]:
                return [self.explain(s, horizon_min=horizon_min, step_min=step_min) for s in (strategies or [])]

        return _ExplainFallback(telemetry, forecast, reporting, energy)


# -------------------------------------------------
# 11) ClosedLoop 适配（执行与闭环）
# -------------------------------------------------
def _init_closed_loop(rlpanel, dispatch, telemetry=None, forecast=None, reporting=None) -> Any:
    """
    需要暴露：
      - submit(strategy, operator="system", mode="auto"|"manual", dry_run=False, notes=None) -> {ok, job}
      - approve(job_id, operator="system") -> {ok, job}
      - get(job_id) / list(limit)
      - ab_compare(job_id) -> {ok, pred, actual, error}
      - learn(job_id, alpha=0.3) -> {ok, model, ab}
      - get_model(strategy_id) -> {ok, model}
    """
    try:
        from app.services.exec_closedloop.closed_loop import ClosedLoopService  # type: ignore
        return ClosedLoopService(
            rlpanel=rlpanel,
            dispatch=dispatch,
            telemetry=telemetry,
            forecast=forecast,
            reporting=reporting,
        )
    except Exception:
        # 兜底占位：接口存在但不可用，便于后续逐步替换
        class _ClosedLoopFallback:
            def __init__(self):
                self.msg = "ClosedLoopService 不可用（未找到 app/services/closed_loop.py 或导入失败）"

            def submit(self, *a, **kw):   return {"ok": False, "error": self.msg}
            def approve(self, *a, **kw):  return {"ok": False, "error": self.msg}
            def get(self, *a, **kw):      return {"ok": False, "error": self.msg}
            def list(self, *a, **kw):     return {"ok": True, "items": []}
            def ab_compare(self, *a, **kw): return {"ok": False, "error": self.msg}
            def learn(self, *a, **kw):    return {"ok": False, "error": self.msg}
            def get_model(self, *a, **kw):return {"ok": False, "error": self.msg}

        return _ClosedLoopFallback()


# -------------------------------------------------
# 12) ⭐ Compliance 适配（合规报表：月度/季度/通用）
# -------------------------------------------------
def _init_compliance(telemetry, energy, forecast=None, reporting=None) -> Any:


    """
    需要暴露：
      - monthly_report(month_yyyy_mm, teu, granularity, factors, diesel_model)
      - quarterly_report(start_month_yyyy_mm, teu, granularity, factors, diesel_model)
      - make_report(config, factors, diesel_model)
    """
    try:
        from app.services.energy_reporting.compliance import ComplianceService  # type: ignore
        return ComplianceService(telemetry=telemetry, energy=energy, forecast=forecast, reporting=reporting)
    except Exception as exc:
        # 保留调用面，但明确失败，禁止将全 0 占位值当成合规报表。
        init_error = f"ComplianceService 初始化失败: {exc}"
        class _ComplianceFallback:
            def monthly_report(self, *a, **kw):
                raise RuntimeError(init_error)
            def quarterly_report(self, *a, **kw):
                raise RuntimeError(init_error)
            def make_report(self, *a, **kw):
                raise RuntimeError(init_error)
        return _ComplianceFallback()

# -------------------------------------------------
# 12) Monitoring 适配（异常/漂移）
# -------------------------------------------------
def _init_monitoring(telemetry, forecast, storage=None, bus=None) -> Any:
    """
    需要暴露（与落地口径一致）：
      - scan_anomalies(asset_ids, point, window_min/start_end, step_sec, method, sensitivity, residual)
      - scan_drift_psi(asset_id, point, baseline_min, recent_min, step_sec, bins)
    真实港口落地：只需确保 telemetry.get_series(...) 接好（TSDB/OPC/MQTT），本服务即可运行。
    """
    try:
        from app.services.monitoring_quality.monitoring import MonitoringService  # type: ignore
        return MonitoringService(telemetry=telemetry, forecast=forecast, storage=storage, bus=bus)
    except Exception:
        # 兜底：不阻断其它模块
        class _MonitoringFallback:
            def scan_anomalies(self, **kw):
                from datetime import datetime, timezone
                return {"available": False, "generated_at": datetime.now(timezone.utc).isoformat(), "items": [], "reason": "MonitoringService failed to initialize"}
            def scan_drift_psi(self, **kw):
                from datetime import datetime, timezone
                return {"available": False, "generated_at": datetime.now(timezone.utc).isoformat(), "psi": None, "bins": [], "reason": "MonitoringService failed to initialize"}
        return _MonitoringFallback()

# -------------------------------------------------
# 13) 容器对象
# -------------------------------------------------
class Container:
    """
    统一持有各服务实例，供 app/server.py 使用。
    同时提供 factors(...) 方法构造排放因子（供 /api/compliance/* 使用）。
    """
    def __init__(self):
        # 基础数据源
        self.telemetry = _init_telemetry()
        # 统一消息总线 / 对象存储 / 碳因子服务
        # 大白话：这是平台“脊梁骨”三件套，服务/接口都从 DI 取它们，后续换后端只改这一处
        # 外部数据源（TOS/AIS/天气/电价/碳价），无文件时走兜底空实现
        self.schedule = _init_schedule_sources()

        try:
            self.bus = MessageBus()  # memory:// 版本；将来可改为 kafka:// 或 mqtt://
        except Exception:
            self.bus = None  # 不阻塞其它模块

        try:
            self.storage = ObjectStorage(StorageConfig(backend_url="file://./data/objects"))
        except Exception:
            self.storage = None

        try:
            self.factors_svc = CarbonFactors()  # 读 data/factors/*.csv，可热加载
        except Exception:
            self.factors_svc = None


        # 上层服务
        self.fcst = _init_forecast(self.telemetry)
        self.rpt = _init_reporting(self.telemetry)
        self.rl = _init_rl()
        self.rlsafety = _init_rlsafety(self.telemetry, storage=self.storage)
        self.twin = _init_twin(self.fcst, telemetry=self.telemetry)

        # 能耗与碳排口径中心
        self.energy = _init_energy(self.telemetry, self.rpt, self.fcst)

        # 统一预警
        self.alerts = _init_alerts(self.telemetry, self.rpt, self.fcst, self.energy)

        # 策略面板（策略清单 + 模拟执行）
        self.rlpanel = _init_rlpanel(
            telemetry=self.telemetry,
            forecast=self.fcst,
            reporting=self.rpt,
            energy=self.energy,
            rl=self.rl,
            twin=self.twin,
        )

        # 策略下发（演示）
        self.dispatch = _init_dispatch(self.telemetry, self.rlpanel, twin=self.twin, rlsafety=self.rlsafety)


        # 策略可解释性（特征重要度 & SHAP-like）
        self.explain = _init_explain(
            telemetry=self.telemetry,
            forecast=self.fcst,
            reporting=self.rpt,
            energy=self.energy,
            rl=self.rl,
            twin=self.twin,
        )
        # 监测与运维（异常/漂移）
        self.monitoring = _init_monitoring(self.telemetry, self.fcst, storage=self.storage, bus=self.bus)


        # 执行与闭环（审批/一键下发 + A/B 对照 + 在线学习）
        self.closedloop = _init_closed_loop(
            rlpanel=self.rlpanel,
            dispatch=self.dispatch,
            telemetry=self.telemetry,
            forecast=self.fcst,
            reporting=self.rpt,
        )

        # ⭐ 合规报表（GHG 范畴 1/2）
        self.compliance = _init_compliance(
            telemetry=self.telemetry,
            energy=self.energy,
            forecast=self.fcst,
            reporting=self.rpt,
        )



    # --------- 小工具：排放因子构造器 ---------
    def factors(
        self,
        grid_g_per_kwh: Optional[float] = None,
        diesel_kg_per_liter: Optional[float] = None,
        selfgen_kg_per_kwh: Optional[float] = None,
        selfgen_share: Optional[float] = None,
    ) -> Any:
        """
        统一在 DI 层构造 EmissionFactors，便于 server.py 直接调用 di.factors(...)
        如果 compliance 模块不可用，兼容返回一个简单的 dataclass（字段名相同）。
        """
        try:
            from app.services.energy_reporting.compliance import EmissionFactors  # type: ignore
            return EmissionFactors(
                grid_g_per_kwh=float(grid_g_per_kwh) if grid_g_per_kwh is not None else 120.0,
                diesel_kg_per_liter=float(diesel_kg_per_liter) if diesel_kg_per_liter is not None else 2.68,
                selfgen_kg_per_kwh=float(selfgen_kg_per_kwh) if selfgen_kg_per_kwh is not None else 0.70,
                selfgen_share=float(selfgen_share) if selfgen_share is not None else 0.0,
            )
        except Exception:
            # 兜底 dataclass：字段结构与 EmissionFactors 一致
            @dataclass
            class _EF:
                grid_g_per_kwh: float = 120.0
                diesel_kg_per_liter: float = 2.68
                selfgen_kg_per_kwh: float = 0.70
                selfgen_share: float = 0.0

                def as_dict(self) -> Dict[str, float]:
                    return {
                        "grid_g_per_kwh": float(self.grid_g_per_kwh),
                        "diesel_kg_per_liter": float(self.diesel_kg_per_liter),
                        "selfgen_kg_per_kwh": float(self.selfgen_kg_per_kwh),
                        "selfgen_share": float(self.selfgen_share),
                    }
            return _EF()


__all__ = ["Container"]
