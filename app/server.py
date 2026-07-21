# ============================================
# app/server.py
# --------------------------------------------
# FastAPI 后端服务（港区数据投影 + 数字孪生证据 + RL 训练评测 + 南向安全边界 + 合规报表）
#
# 当前安全边界：
# - 真实训练/留出集评测使用 /api/rl/train/*；训练期间不渲染。
# - 旧 RL、工程模拟器和桌面联动默认不挂载，且不能作为算法证据。
# - 南向执行只通过 /api/actuators/*，默认失效安全关闭并强制异人确认。
# - 数据、孪生校准和外部适配器均公开来源等级；未配置时不生成替代结果。
#
# 运行：
#   python -m app.server
# 文档（交互式）：
#   http://127.0.0.1:8000/docs
# RL 面板（独立页）：
#   http://127.0.0.1:8000/rl-panel
# ============================================

from __future__ import annotations
import os
import math  # 监测统计用（z-score/PSI 等）
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime, timezone, timedelta  # <- 加上 timedelta

from fastapi import Body, FastAPI, HTTPException, Query, APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.services.pipeline.ingest import register_ingest_startup
from app.services.forecast_twin.sim_aggregate import aggregate_sim
from app.services.exec_closedloop.dispatch_api import router as exec_router  # 执行网关REST路由（命令下发/确认/回滚/证据）
from app.services.curves import CurvesService
from app.services.curves.stacked_power import CurvesStacked
from app.services.curves.energy_intensity import CurvesEnergyIntensity
from app.services.curves.peak_risk import CurvesPeakRisk
from app.services.curves.carbon_intensity import CurvesCarbonIntensity
from app.services.curves.economic_benefit import CurvesEconomicBenefit
from app.services.curves.bess_capability import CurvesBessCapability
from app.services.dashlets import router as dashlets_router  # ← 新增：dashlets 路由
from app.services.opsx.api import router as opsx_router  # OpsX：上线与运维控制（8个子模块）
from app.services.platform.api import router as platform_router
from app.services.twinplus.api import router as twin_router
from app.services.portx.api import router as portx_router
from app.services.ux.api import router as ux_router
from app.services.portx.deep import router as port_deep_router
from app.services.portviz.api import router as portviz_router  # ← 新增：港区渲染/数据流路由
from app.services.mas_orchestrator.api import router as mas_router
from app.services.rl_ops_center.api import router as rlops_router
from app.services.rl_integration.api import router as rl_integration_router
from app.services.rl_actions.api import router as rl_actions_router
from app.services.assistant_actions.api import router as assistant_actions_router
from app.services.xiaoyi_ai.api import router as xiaoyi_ai_router
from app.services.sailing_simulator.api import router as sailing_simulator_router
from app.services.twinlab.api import router as twinlab_router
from app.services.copilot.api import router as copilot_router
from app.services.energyx.api import router as energyx_router
from app.services.app_center import service as app_center_service
from app.services.story.service import router as story_router
from app.services.rl_training.api import router as real_rl_training_router
from app.services.rl_training.trainer import ALGORITHMS as REAL_RL_ALGORITHMS
from app.services.rl_training.trainer import TRAINING_MANAGER
from app.services.twin_schema.api import router as twin_schema_router
from app.services.execution.api import router as site_execution_router
from app.operations import configure_operations, cors_origins, is_production

from fastapi.staticfiles import StaticFiles
# 路由（放在 /api/curves/asset 后面即可）
router = APIRouter()
@router.get("/api/curves/stacked_power", tags=["curves"])
async def curves_stacked_power(
    mode: str = Query("forecast", pattern="^(now|forecast|sim)$"),
    horizon_min: int = Query(360, ge=1, le=24*60),
    step_min: int = Query(1, ge=1, le=60),
    limit: int = Query(200, ge=1, le=500),
) -> JSONResponse:
    data = stacked.stacked_power(mode=mode, horizon_min=horizon_min, step_min=step_min, limit=limit)
    return JSONResponse(data)
@router.get("/api/curves/energy_intensity_legacy", tags=["curves"])
async def curves_energy_intensity_legacy(
    mode: str = Query("forecast", pattern="^(now|forecast|sim)$"),
    horizon_min: int = Query(360, ge=1, le=24*60),
    step_min: int = Query(1, ge=1, le=60),
    limit: int = Query(200, ge=1, le=500),
    teu: float = Query(12000.0, ge=1.0, description="分母：窗口内TEU总量"),
) -> JSONResponse:
    """
    【legacy】旧版单位TEU能耗接口（直接在路由里手写积分），
    仅保留用于对比和排查；前端已不再使用。
    """
    agg = curves.aggregate(mode=mode, horizon_min=horizon_min, step_min=step_min, limit=limit)
    S = agg.get("series", {})

    def _acc(arr):
        kwh = 0.0
        out = []
        for p in arr or []:
            v = float(p.get("kW", p.get("p50", 0.0)))    # 兼容有/无分位
            kwh += v * (step_min / 60.0)                 # kWh 累加
            out.append({"ts": p.get("ts"), "kW": round(kwh / max(teu, 1.0), 6)})
        return out, kwh

    s50, cum_kwh = _acc(S.get("p50", []))
    s10, _ = _acc(S.get("p10", [])) if S.get("p10") else ([], 0.0)
    s90, _ = _acc(S.get("p90", [])) if S.get("p90") else ([], 0.0)

    return JSONResponse({
        "mode": mode,
        "unit": "kWh/TEU",
        "series": {"p50": s50, "p10": s10, "p90": s90},
        "cum_kwh": round(cum_kwh, 3),
        "teu": teu,
        "intensity_total": (s50[-1]["kW"] if s50 else 0.0),
    })


@router.get("/api/curves/energy_intensity", tags=["curves"])
async def curves_energy_intensity(
        mode: str = Query("forecast", pattern="^(now|forecast|sim)$"),
        horizon_min: int = Query(360, ge=1, le=24 * 60),
        step_min: int = Query(1, ge=1, le=60),
        limit: int = Query(200, ge=1, le=500),
        teu: float = Query(12000.0, ge=1.0, description="窗口内 TEU 总量（分母）"),
) -> JSONResponse:
    """
    单位TEU能耗（累计）：从 CurvesService.aggregate() 取聚合功率的 p10/p50/p90，
    按 step_min 折算成累计 kWh 再除以 TEU，得到 kWh/TEU 的分位曲线（贴近真实港口统计口径）。
    """
    data = energy_intensity.intensity(
        mode=mode,
        horizon_min=horizon_min,
        step_min=step_min,
        limit=limit,
        teu=teu,
    )
    return JSONResponse(data)

# ===== 多港口 / 多场景一体化管理（跨港区 & 模型复用） =====
@router.get("/api/multiport/summary", tags=["multiport"])
async def multiport_summary() -> JSONResponse:
    """
    前端 /ui/index.html 的“跨港区 & 模型复用”模块使用的汇总接口。
    期望返回结构：
    {
      "updated_at": "2025-12-04T08:00:00Z",  # ISO8601，可为 None
      "ports": [                             # 列表项字段见 services/multiport/service.py
        {
          "id": "port-a",
          "name": "Port A · 集装箱港",
          "phase": "PoC|Pilot|全量",
          "twin_fidelity": 0.0,              # 0~1
          "annual_saving_mwy": 0,            # 单位: 万元
          "annual_co2_t": 0,                 # 单位: 吨
          "scenes": ["AGV 充电", "岸电 BESS", ...]
        },
        ...
      ]
    }
    """
    # 延迟导入，避免 services 目录未创建时启动失败
    try:
        from app.services.multiport.service import MultiportService  # 下一步我们会创建
    except Exception:
        # services 还没就绪 → 返回 503，让前端显示“接口错误或暂无数据”
        return JSONResponse(
            {"updated_at": None, "ports": []},
            status_code=503
        )

    svc = MultiportService()

    # 兼容 sync/async 实现
    try:
        getsum = getattr(svc, "get_summary")
        data = await getsum() if asyncio.iscoroutinefunction(getsum) else getsum()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"multiport service error: {e}")

    # 基础校验 & 轻度规范化
    if not isinstance(data, dict) or "ports" not in data:
        raise HTTPException(status_code=500, detail="Bad schema from MultiportService.get_summary")

    ua = data.get("updated_at")
    if isinstance(ua, datetime):
        # 规范成 ISO8601 Z 时区
        data["updated_at"] = ua.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    return JSONResponse(data)

# 数据清洗/插补/质量评分（A.感知与建模）
from app.ops.data_quality import clean_and_impute  # 返回 cleaned 曲线 + 质量分

# 依赖注入容器（确保 app/di.py 为含 explain / dispatch / closedloop / compliance 的版本）
try:
    from app.di import Container
except Exception as e:
    raise RuntimeError("未找到 app/di.py 或导入失败，请先使用我提供的 di.py。") from e
# 外部适配器（TOS/Market/AIS+Tide）—— 未配置时可复现模拟；真实接口失败不静默降级
try:
    from app.adapters.tos_client import TOSClient
except Exception:
    TOSClient = None

try:
    from app.adapters.market_client import MarketClient
except Exception:
    MarketClient = None

try:
    from app.adapters.ais_tide_client import AISTideClient
except Exception:
    AISTideClient = None


# -------------------------------------------------
# 初始化应用与依赖
# -------------------------------------------------
app = FastAPI(
    title="Smart Port Twin API",
    version="3.0.1",
    docs_url=None if is_production() else "/docs",
    redoc_url=None if is_production() else "/redoc",
    openapi_url=None if is_production() else "/openapi.json",
)
app.include_router(router)
_ENABLE_ENGINEERING_SIMULATORS = os.getenv(
    "PORT_DT_ENABLE_ENGINEERING_SIMULATORS", ""
).strip().lower() in {"1", "true", "yes", "on"}
_ENABLE_LEGACY_CLOSEDLOOP = os.getenv(
    "PORT_DT_ENABLE_LEGACY_CLOSEDLOOP", ""
).strip().lower() in {"1", "true", "yes", "on"}
_ENABLE_DESKTOP_INTEGRATIONS = os.getenv(
    "PORT_DT_ENABLE_DESKTOP_INTEGRATIONS", ""
).strip().lower() in {"1", "true", "yes", "on"}


def _engineering_route(method: str, path: str, **kwargs):
    if _ENABLE_ENGINEERING_SIMULATORS:
        return getattr(app, method)(path, **kwargs)

    def passthrough(endpoint):
        return endpoint

    return passthrough

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Accept", "Content-Type", "X-API-Key", "X-Request-ID"],
)
configure_operations(app)
app.include_router(opsx_router, prefix="/api/opsx", tags=["opsx"])
if _ENABLE_ENGINEERING_SIMULATORS:
    # 以下旧路由依赖生成数据，只能用于界面/契约联调。
    app.include_router(dashlets_router)
    app.include_router(platform_router, prefix="/api/platform", tags=["platform-engineering-simulator"])
    app.include_router(portx_router, prefix="/api/portx", tags=["portx-engineering-simulator"])
    app.include_router(port_deep_router, prefix="/api/port", tags=["port-deep-engineering-simulator"])
    app.include_router(energyx_router, prefix="/api/energyx", tags=["energyx-engineering-simulator"])
    app.include_router(exec_router)
    app.include_router(mas_router, prefix="/api/mas", tags=["mas-engineering-simulator"])
if os.getenv("PORT_DT_ENABLE_TWINPLUS_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}:
    app.include_router(twin_router, prefix="/api/twin", tags=["twinplus-demo"])
app.include_router(ux_router,       prefix="/api/ux",       tags=["ux"])
app.include_router(portviz_router, prefix="/api/portviz", tags=["portviz"])  # ← 新增：/api/portviz/*
app.include_router(rlops_router, prefix="/api/rlops", tags=["rlops"])
app.include_router(rl_integration_router)
app.include_router(rl_actions_router)
app.include_router(assistant_actions_router)
if _ENABLE_DESKTOP_INTEGRATIONS:
    app.include_router(xiaoyi_ai_router)
    app.include_router(sailing_simulator_router)
app.include_router(twinlab_router, prefix="/api/twinlab", tags=["twinlab"])
app.include_router(copilot_router, prefix="/api/copilot", tags=["copilot"])
# app.services.story.service 是旧多港口样例路由，默认不挂载。
if os.getenv("PORT_DT_ENABLE_LEGACY_STORY", "").strip().lower() in {"1", "true", "yes", "on"}:
    app.include_router(story_router)
app.include_router(real_rl_training_router)
app.include_router(twin_schema_router)
app.include_router(site_execution_router)

di = Container()
stacked = CurvesStacked(di)
curves = CurvesService(di)
energy_intensity = CurvesEnergyIntensity(di)
peak_risk = CurvesPeakRisk(di)
carbon_intensity = CurvesCarbonIntensity(di)
economic_benefit = CurvesEconomicBenefit(di)
bess_capability = CurvesBessCapability(di)


register_ingest_startup(app, di, interval_sec=30, step_sec=60)
# 适配器单例（来源状态由 /api/system/provenance 明示）
_tos = TOSClient() if TOSClient else None
_market = MarketClient() if MarketClient else None
_ais = AISTideClient() if AISTideClient else None


@app.get("/api/system/provenance", tags=["system"])
async def system_provenance() -> JSONResponse:
    """Single source-of-truth for data/algorithm provenance shown to operators."""
    from app.services.portviz.source import SourceConfig

    portviz_cfg = SourceConfig()
    adapters = {
        "tos": _tos.source_status() if _tos and hasattr(_tos, "source_status") else {"mode": "unavailable"},
        "market": _market.source_status() if _market and hasattr(_market, "source_status") else {"mode": "unavailable"},
        "ais_tide": _ais.source_status() if _ais and hasattr(_ais, "source_status") else {"mode": "unavailable"},
        "schedule": di.schedule.source_status() if hasattr(getattr(di, "schedule", None), "source_status") else {"mode": "unavailable"},
    }
    live_adapters = all(
        item.get("mode") == "live_rest"
        or (item.get("ais_mode") == "live_rest" and item.get("tide_mode") == "live_rest")
        for item in adapters.values()
    )
    portviz_measured = portviz_cfg.mode in {"real", "adapter", "prod"}
    telemetry_status_fn = getattr(di.telemetry, "source_status", None)
    telemetry_status = telemetry_status_fn() if callable(telemetry_status_fn) else {
        "mode": "engineering_simulator",
        "implementation": f"{type(di.telemetry).__module__}.{type(di.telemetry).__name__}",
        "production": False,
    }
    telemetry_live = telemetry_status.get("mode") in {"live", "live_rest", "opcua", "mqtt", "tsdb"}
    engineering_simulators_enabled = os.getenv("PORT_DT_ENABLE_ENGINEERING_SIMULATORS", "").strip().lower() in {"1", "true", "yes", "on"}
    legacy_rl_enabled = False
    return JSONResponse(
        {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "research_integration_ready": True,
            "production_claim_allowed": False,
            "production_blockers": [
                *( [] if live_adapters else ["external_adapters_not_all_live"] ),
                *( [] if telemetry_live else ["measured_telemetry_not_configured"] ),
                *( [] if portviz_measured else ["measured_entity_tracks_not_configured"] ),
                "production_actuator_and_site_acceptance_not_configured",
            ],
            "rl": TRAINING_MANAGER.capabilities(),
            "portviz": {
                "mode": portviz_cfg.mode,
                "dataset_artifact": Path(portviz_cfg.dataset_path).name if portviz_cfg.mode in {"dataset", "replay", "public"} else None,
                "frame_adapter_configured": bool(portviz_cfg.frames_path),
                "measured_entity_tracks": portviz_measured,
            },
            "external_adapters": adapters,
            "telemetry": telemetry_status,
            "feature_flags": {
                "engineering_simulators_enabled": engineering_simulators_enabled,
                "legacy_rl_enabled": legacy_rl_enabled,
                "schedule_adapter_configured": adapters.get("schedule", {}).get("mode") != "unavailable",
                "tos_adapter_live": adapters.get("tos", {}).get("mode") == "live_rest",
                "market_adapter_live": adapters.get("market", {}).get("mode") == "live_rest",
                "ais_tide_adapter_live": (
                    adapters.get("ais_tide", {}).get("ais_mode") == "live_rest"
                    and adapters.get("ais_tide", {}).get("tide_mode") == "live_rest"
                ),
                "twin_calibration_configured": bool(os.getenv("PORT_DT_TWIN_CALIBRATION_PATH", "").strip()),
            },
            "module_assessment": {
                "rl_training": "real_algorithm_and_dataset",
                "rl_evaluation": "chronological_holdout_only",
                "port_visualisation": "dataset_projection_or_strict_jsonl_adapter",
                "forecast_twin": "telemetry_fitted_ridge_autoregression_with_explicit_scenario_parameters",
                "rlops": "persisted_training_and_holdout_evaluation_not_ope",
                "opsx": "engineering_simulator_opt_in" if engineering_simulators_enabled else "unavailable_until_production_backend_is_configured",
                "legacy_dashboard_generators": "opt_in_engineering_simulators" if engineering_simulators_enabled else "disabled_by_default",
                "twinlab": "provenance_verified_evidence_files_required",
                "esg_compliance": "demo_blocked_provenance_required_not_legal_certification",
                "exec_cockpit": "provenance_verified_snapshot_required",
                "platform_map": "repository_architecture_config_not_runtime_topology",
                "execution": "dry_run_and_human_gate_no_default_production_actuator",
                "legacy_rl_routes": "not_distributed",
            },
        }
    )

def _parse_iso(s: str) -> datetime:
    # 容错 ISO8601（支持带/不带Z）
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

_UI_INDEX = Path(__file__).resolve().parent / "ui" / "index.html"
_OPS_COPILOT_UI = Path(__file__).resolve().parent / "ui" / "ops_copilot.html"
_INTEGRATION_HUB_UI = Path(__file__).resolve().parent / "ui" / "integration_hub.html"
_XIAOYI_SPRITE_JS = Path(__file__).resolve().parent / "ui" / "adapters" / "xiaoyi_sprite.js"
_BILINGUAL_UI_JS = Path(__file__).resolve().parent / "ui" / "adapters" / "bilingual_ui.js"


def _inject_xiaoyi_sprite(html: str) -> str:
    marker = "/ui/adapters/xiaoyi_sprite.js"
    if marker in html:
        return html
    tag = '  <script src="/ui/adapters/xiaoyi_sprite.js?v=20260712-speech-v2"></script>\n'
    if "</body>" in html:
        return html.replace("</body>", f"{tag}</body>")
    return html + tag


def _inject_bilingual_ui(html: str) -> str:
    marker = "/ui/adapters/bilingual_ui.js"
    if marker in html:
        return html
    tag = '  <script src="/ui/adapters/bilingual_ui.js?v=20260715-zh-headings-v5"></script>\n'
    if "</body>" in html:
        return html.replace("</body>", f"{tag}</body>")
    return html + tag

# ===== UI 子页面挂载（/rl_future -> app/ui/rl_future） =====
_RL_FUTURE_UI_DIR = Path(__file__).resolve().parent / "ui" / "rl_future"
try:
    if _RL_FUTURE_UI_DIR.exists():
        app.mount(
            "/rl_future",
            StaticFiles(directory=str(_RL_FUTURE_UI_DIR), html=True),
            name="rl_future_ui",
        )
except Exception as _e:
    print("[warn] unable to mount /rl_future UI:", _e)


# -------------------------------------------------
# 内嵌页面：RL 策略面板（保持）
# -------------------------------------------------
_RL_PANEL_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>强化学习面板 / RL Panel</title>
  <style>
    :root{--bg:#0f172a;--card:#111827;--muted:#9ca3af;--ok:#34d399;--warn:#f59e0b;--bad:#ef4444;--btn:#1d4ed8}
    *{box-sizing:border-box;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif}
    body{margin:0;background:linear-gradient(180deg,#0b1220,#0f172a);color:#e5e7eb}
    header{padding:16px 20px;border-bottom:1px solid #1f2937;display:flex;justify-content:space-between;align-items:center}
    header h1{margin:0;font-size:18px;letter-spacing:.5px}
    header .hint{color:var(--muted);font-size:12px}
    main{padding:18px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}
    .card{background:rgba(17,24,39,.85);border:1px solid #1f2937;border-radius:14px;padding:14px;box-shadow:0 18px 40px rgba(0,0,0,.18)}
    .toolbar{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
    .toolbar input,.toolbar select{background:#0b1020;color:#e5e7eb;border:1px solid #1f2937;border-radius:8px;padding:7px 9px}
    .btn{background:var(--btn);border:none;color:#fff;border-radius:10px;padding:8px 12px;cursor:pointer}
    .btn.secondary{background:#243b6b}
    .btn.ghost{background:#162036}
    .btn:disabled{opacity:.5;cursor:not-allowed}
    table{width:100%;border-collapse:separate;border-spacing:0 8px}
    th{color:#cbd5e1;font-weight:600;text-align:left;font-size:12px;padding:6px 8px}
    td{background:#0c1426;border:1px solid #1f2937;border-left:none;border-right:none;padding:10px 8px;font-size:13px;vertical-align:top}
    tr td:first-child{border-left:1px solid #1f2937;border-top-left-radius:10px;border-bottom-left-radius:10px}
    tr td:last-child{border-right:1px solid #1f2937;border-top-right-radius:10px;border-bottom-right-radius:10px}
    .badge{font-size:11px;padding:2px 6px;border-radius:999px;border:1px solid #2b3344;color:#cbd5e1;background:#0b1220}
    .metric{display:flex;gap:12px;flex-wrap:wrap}
    .metric .k{color:#94a3b8}
    .metric .v{font-weight:600}
    .muted{color:#9ca3af}
    .row-actions{display:flex;gap:8px;flex-wrap:wrap}
    .small{font-size:12px}
    .tiny{font-size:11px}
    .panel-title{font-size:13px;color:#cbd5e1;margin:0 0 8px}
    canvas{width:100%;height:220px;background:#0a1120;border:1px solid #1f2937;border-radius:8px}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace}
    .status-ok{color:#34d399}
    .status-bad{color:#ef4444}
    .status-warn{color:#f59e0b}
    .subgrid{display:grid;grid-template-columns:1fr;gap:12px;margin-top:12px}
    pre{white-space:pre-wrap;word-break:break-word}
    .history-item{padding:10px 12px;border:1px solid #1f2937;border-radius:12px;background:#0c1426;margin-top:8px}
    .history-head{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
    .history-body{margin-top:6px;color:#cbd5e1}
    .risk{display:inline-block;margin:4px 6px 0 0;padding:2px 6px;border-radius:999px;border:1px solid #60422a;background:#2b1f14;color:#fbbf24;font-size:11px}
    .train-card{grid-column:1/-1}
    .train-head{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap}
    .train-title{font-size:17px;font-weight:800;color:#f8fafc;letter-spacing:.4px}
    .train-title .en{display:block;margin-top:3px;font-size:11px;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;color:#93c5fd}
    .section-label{margin:0 0 8px;color:#dbeafe;font-size:12px;font-weight:800;letter-spacing:.7px;text-transform:uppercase}
    .train-layout{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(360px,.8fr);gap:14px;margin-top:12px}
    .param-grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}
    .field{min-width:0}
    .field label{display:block;margin:0 0 5px;color:#9fb3d9;font-size:11px;font-weight:700}
    .field input,.field select{width:100%;background:#0a1325;color:#e5e7eb;border:1px solid #23324e;border-radius:10px;padding:8px 9px;outline:none}
    .field input:focus,.field select:focus{border-color:#60a5fa;box-shadow:0 0 0 2px rgba(96,165,250,.16)}
    .baseline-grid{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px;margin-top:12px}
    .algo-card{appearance:none;text-align:left;color:#dbeafe;background:linear-gradient(180deg,#111c32,#0b1324);border:1px solid #23324e;border-radius:12px;padding:11px;min-height:128px;cursor:pointer;transition:.18s ease}
    .algo-card:hover{border-color:#4f7fc7;transform:translateY(-1px)}
    .algo-card.active{border-color:#60a5fa;box-shadow:0 0 0 2px rgba(96,165,250,.18),0 14px 32px rgba(37,99,235,.16)}
    .algo-top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}
    .algo-name{font-size:14px;font-weight:900;color:#f8fafc}
    .algo-desc{font-size:11px;line-height:1.5;color:#aab8d3}
    .algo-desc b{color:#dbeafe}
    .algo-metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:9px;color:#93a4c4;font-size:11px}
    .algo-metrics b{color:#bfdbfe}
    .pill{display:inline-flex;align-items:center;border-radius:999px;border:1px solid #2a3b5d;background:#0a1325;color:#93c5fd;padding:2px 7px;font-size:10px;font-weight:800}
    .pill.rule{color:#fbbf24;border-color:#60422a;background:#22170b}
    .connector-grid{display:grid;grid-template-columns:minmax(0,.95fr) minmax(0,1.05fr);gap:12px;margin-top:12px}
    .connector-box,.payload-preview{background:#0a1120;border:1px solid #23324e;border-radius:12px;padding:12px;min-height:146px}
    .connector-line{display:flex;justify-content:space-between;gap:10px;margin-top:8px;color:#cbd5e1}
    .connector-line span:first-child{color:#8ea3c5}
    .payload-preview{margin:0;max-height:230px;overflow:auto;color:#bfdbfe}
    .progress-card{height:100%;display:flex;flex-direction:column;background:linear-gradient(180deg,#0c1730,#091222);border:1px solid #23324e;border-radius:14px;padding:14px}
    .progress-meta{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
    .progress-meta strong{font-size:24px;color:#dbeafe}
    .progress-stage{margin-top:2px;color:#93c5fd;font-size:12px;font-weight:700}
    .progress-shell{height:18px;background:#07101f;border:1px solid #22314f;border-radius:999px;overflow:hidden;margin:13px 0 9px}
    .progress-fill{height:100%;width:0%;background:linear-gradient(90deg,#38bdf8,#22c55e,#fbbf24);box-shadow:0 0 18px rgba(34,197,94,.38);transition:width 1.2s ease}
    .train-log{height:168px;overflow:auto;background:#07101f;border:1px solid #1f2937;border-radius:12px;padding:10px;margin-top:12px;color:#cbd5e1;line-height:1.55}
    .train-summary{margin-top:12px;color:#bfdbfe;line-height:1.6;border-top:1px solid #1f2937;padding-top:10px}
    .evaluation-panel{display:none;margin-top:12px;border:1px solid #264b3e;border-radius:12px;background:#071a17;padding:10px}
    .evaluation-panel canvas{width:100%;height:180px;background:#06111d;border:1px solid #1f3b34;border-radius:10px;margin-top:8px}
    .train-stat-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:12px}
    .stat-card{background:#0a1120;border:1px solid #1f2937;border-radius:10px;padding:8px}
    .stat-card .k{display:block;color:#8ea3c5;font-size:10px;margin-bottom:4px}
    .stat-card .v{display:block;color:#e5e7eb;font-size:12px;font-weight:800;word-break:break-word}
    .progress-chip{border:1px solid #2a3b5d;background:#0a1325;color:#93c5fd;border-radius:999px;padding:6px 10px;font-size:11px;font-weight:800}
    .link-health-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:10px 0}
    .link-health-item{min-width:0;border:1px solid #22314f;border-radius:10px;background:#07101f;padding:8px}
    .link-health-item span{display:block;color:#8ea3c5;font-size:10px;margin-bottom:4px}
    .link-health-item b{display:block;color:#fbbf24;font-size:12px;line-height:1.35}
    .link-health-item.ok b{color:#34d399}
    .link-health-item.bad b{color:#ef4444}
    .link-health-detail{margin-top:6px;color:#93a4c4;line-height:1.55}
    .confirm-backdrop{position:fixed;inset:0;z-index:60;display:none;align-items:center;justify-content:center;background:rgba(2,6,23,.72);padding:18px}
    .confirm-dialog{width:min(760px,100%);border:1px solid #334155;border-radius:14px;background:#081222;box-shadow:0 28px 80px rgba(0,0,0,.5);padding:16px}
    .confirm-dialog h2{margin:0 0 8px;font-size:18px;color:#f8fafc}
    .confirm-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
    .confirm-item{border:1px solid #1f2937;border-radius:10px;background:#0a1120;padding:10px;min-width:0}
    .confirm-item span{display:block;color:#8ea3c5;font-size:11px;margin-bottom:5px}
    .confirm-item b{display:block;color:#e5e7eb;font-size:13px;word-break:break-word}
    .confirm-item.recommend{grid-column:1 / -1;border-color:#264b3e;background:#071a17}
    .confirm-item.recommend b{color:#bbf7d0;line-height:1.55}
    .risk-list{margin:12px 0 0;padding:10px 12px;border:1px solid #60422a;border-radius:10px;background:#22170b;color:#fde68a;line-height:1.65}
    @media(max-width:1180px){
      main{grid-template-columns:1fr}
      .train-layout,.connector-grid{grid-template-columns:1fr}
      .param-grid{grid-template-columns:repeat(2,minmax(140px,1fr))}
      .baseline-grid{grid-template-columns:repeat(2,minmax(140px,1fr))}
    }
    @media(max-width:680px){
      header{align-items:flex-start;gap:12px;flex-direction:column}
      .param-grid,.baseline-grid,.train-stat-grid,.confirm-grid{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<header>
  <div>
    <h1>强化学习面板 / RL Panel</h1>
    <div class="hint">接口 / API：/api/rl/train/start · /api/rl/train/status · /api/rl/train/:job/evaluate · /api/rl/models/:job/readiness</div>
  </div>
  <div class="row-actions">
    <button id="btnBackToPlatform" class="btn ghost">回主平台策略区</button>
    <button id="btnBackToHome" class="btn ghost">回主平台首页</button>
  </div>
</header>
<div id="returnBanner" class="card" style="margin:16px 18px 0 18px;padding:12px 14px;display:none;">
  <div class="panel-title" style="margin-bottom:6px;">主平台 → RL 面板联动</div>
  <div id="returnBannerText" class="small muted">当前从主平台进入，可在完成模型登记查看、留出集评测和上线门禁检查后回到主平台。</div>
</div>

<main>
  <section class="card" style="grid-column:1 / -1;border-color:rgba(77,228,255,.42);">
    <div class="train-head">
      <div>
        <div class="train-title">移动端训练申请 <span class="en">Mobile Training Requests</span></div>
        <div class="muted small">手机端只能提交参数和训练意图；电脑端人工批准后才会创建训练任务。手机端没有绕过批准的入口。</div>
      </div>
      <div id="mobileRequestChip" class="progress-chip">待确认 0</div>
    </div>
    <div class="connector-grid" style="margin-top:12px;">
      <div>
        <div class="connector-line"><span>申请入口</span><b class="mono">POST /api/rl/train/requests</b></div>
        <div class="connector-line"><span>批准入口</span><b class="mono">POST /api/rl/train/requests/:id/approve</b></div>
        <div class="connector-line"><span>当前电脑端确认人</span><b>港口调度员-01</b></div>
      </div>
      <div class="row-actions" style="align-items:flex-start;justify-content:flex-end;">
        <button id="btnRefreshMobileRequests" class="btn ghost">刷新移动端申请</button>
      </div>
    </div>
    <div id="mobileTrainingRequestList" style="display:grid;gap:10px;margin-top:12px;">
      <div class="small muted">正在读取移动端申请…</div>
    </div>
  </section>

  <section class="card train-card">
    <div class="train-head">
      <div>
        <div class="train-title">强化学习训练控制台 <span class="en">RL Training Console</span></div>
        <div class="muted small">参数设置、优化目标、算法接入口、baseline 对比与训练过程细节统一在这里承接。</div>
      </div>
      <div id="trainJobChip" class="progress-chip">未启动 · IDLE</div>
    </div>

    <div class="train-layout">
      <div>
        <div class="section-label">训练参数 / Training Parameters</div>
        <div class="param-grid">
          <div class="field">
            <label for="selAlgo">算法 / Algorithm</label>
            <select id="selAlgo">
              <option value="sac" selected>SAC · Soft Actor-Critic</option>
              <option value="ppo">PPO · Proximal Policy Optimization</option>
              <option value="td3">TD3 · Twin Delayed DDPG</option>
              <option value="dqn">DQN · Deep Q-Network</option>
              <option value="mpc">MPC · 模型预测控制基线</option>
            </select>
          </div>
          <div class="field">
            <label for="selDataset">训练数据集 / Dataset</label>
            <select id="selDataset"><option value="public_port_ops_v1">正在校验数据集…</option></select>
          </div>
          <div class="field">
            <label for="selObjective">优化目标 / Objective</label>
            <select id="selObjective">
              <option value="multi_objective" selected>综合最优 · Energy + Carbon + Cost + Safety</option>
              <option value="energy_min">能耗最低 · Min Energy</option>
              <option value="carbon_min">碳排最低 · Min Carbon</option>
              <option value="cost_min">电费最低 · Min Cost</option>
              <option value="peak_shaving">需量峰值削减 · Peak Shaving</option>
              <option value="throughput_max">吞吐最大 · Max Throughput</option>
              <option value="delay_min">船舶等待最短 · Min Vessel Delay</option>
              <option value="safety_guard">安全约束优先 · Safety First</option>
              <option value="battery_life">BESS 寿命友好 · Battery Life</option>
              <option value="shore_power_priority">岸电优先 · Shore Power Priority</option>
              <option value="emission_quota">碳配额达标 · Emission Quota</option>
              <option value="resilience">扰动韧性 · Disruption Resilience</option>
              <option value="agv_turnaround">AGV 周转效率 · AGV Turnaround</option>
              <option value="berth_reliability">泊位窗口稳定 · Berth Reliability</option>
              <option value="grid_stability">电网稳定 · Grid Stability</option>
              <option value="carbon_cost_balance">碳成本平衡 · Carbon-Cost Balance</option>
              <option value="low_risk_canary">低风险试运行 · Low-risk Canary</option>
              <option value="storm_resilience">台风扰动鲁棒 · Storm Resilience</option>
            </select>
          </div>
          <div class="field">
            <label for="selScenario">训练场景 / Scenario</label>
            <select id="selScenario">
              <option value="mapped_dataset" selected>当前映射数据集 · Mapped Dataset</option>
              <option value="noon_peak">午间作业高峰 · Noon Peak</option>
              <option value="night_low_carbon">夜间低碳窗口 · Low-carbon Night</option>
              <option value="storm_disruption">台风扰动 · Storm Disruption</option>
              <option value="shore_power_peak">岸电接入高峰 · Shore-power Peak</option>
            </select>
          </div>
          <div class="field">
            <label for="selAsset">设备组 / Asset Group</label>
            <select id="selAsset">
              <option value="all_port" selected>全港设备 · Whole Port</option>
              <option value="qc_bess_shore">岸桥 + BESS + 岸电</option>
              <option value="agv_charge">AGV 充换电</option>
              <option value="hvac_cooling">冷站 HVAC</option>
              <option value="yard_lighting">堆场照明</option>
              <option value="shore_power">岸电储能</option>
              <option value="bess_energy">储能能量调度</option>
              <option value="yard_crane">场桥作业</option>
              <option value="berth_ops">泊位作业链</option>
            </select>
          </div>

          <div class="field"><label for="inpTrainHorizon">预测窗口 min / Horizon</label><input id="inpTrainHorizon" type="number" value="720" min="60" max="2880" step="30"></div>
          <div class="field"><label for="inpTrainStep">步长 min / Step</label><input id="inpTrainStep" type="number" value="5" min="1" max="60" step="1"></div>
          <div class="field"><label for="inpTotalSteps">训练步数 / Total Steps</label><input id="inpTotalSteps" type="number" value="20000" min="64" max="5000000" step="1000"></div>
          <div class="field"><label for="inpBatch">批大小 / Batch Size</label><input id="inpBatch" type="number" value="256" min="32" max="2048" step="32"></div>

          <div class="field"><label for="inpLR">学习率 / Learning Rate</label><input id="inpLR" type="number" value="0.0003" min="0.00001" max="0.01" step="0.00001"></div>
          <div class="field"><label for="inpGamma">折扣因子 / Gamma</label><input id="inpGamma" type="number" value="0.995" min="0.8" max="0.999" step="0.001"></div>
          <div class="field"><label for="inpTau">目标网络 τ / Target Tau</label><input id="inpTau" type="number" value="0.005" min="0.001" max="0.1" step="0.001"></div>
          <div class="field"><label for="inpEntropy">熵系数 / Entropy Coef</label><input id="inpEntropy" type="number" value="0.02" min="0" max="1" step="0.001"></div>

          <div class="field"><label for="inpReplay">回放池 / Replay Buffer</label><input id="inpReplay" type="number" value="120000" min="10000" max="2000000" step="10000"></div>
          <div class="field"><label for="inpSeed">随机种子 / Seed</label><input id="inpSeed" type="number" value="42" min="1" max="9999" step="1"></div>
          <div class="field"><label for="inpDemandCap">需量上限 kW / Demand Cap</label><input id="inpDemandCap" type="number" value="3000" min="100" max="20000" step="10"></div>
          <div class="field">
            <label for="selGuardrail">安全护栏 / Guardrails</label>
            <select id="selGuardrail">
              <option value="strict" selected>严格 · Strict</option>
              <option value="balanced">均衡 · Balanced</option>
              <option value="explore">探索 · Explore</option>
            </select>
          </div>

          <div class="field"><label for="inpCostW">电费权重 / Cost W</label><input id="inpCostW" type="number" value="0.24" min="0" max="1" step="0.01"></div>
          <div class="field"><label for="inpCarbonW">碳权重 / Carbon W</label><input id="inpCarbonW" type="number" value="0.22" min="0" max="1" step="0.01"></div>
          <div class="field"><label for="inpPeakW">峰值权重 / Peak W</label><input id="inpPeakW" type="number" value="0.18" min="0" max="1" step="0.01"></div>
          <div class="field"><label for="inpSafetyW">安全权重 / Safety W</label><input id="inpSafetyW" type="number" value="0.20" min="0" max="1" step="0.01"></div>
        </div>

        <div class="section-label" style="margin-top:14px;">五算法可复现实验 / 4 RL + 1 Control Baseline</div>
        <div id="baselineGrid" class="baseline-grid"></div>

        <div class="connector-grid">
          <div class="connector-box small">
            <div class="section-label">算法接入口 / Algorithm Adapter</div>
            <div id="connectorStatus" class="status-warn">接入口状态：待启动训练</div>
            <div class="link-health-grid" id="linkHealthGrid" aria-live="polite">
              <div class="link-health-item" id="linkHealthXiaoyi"><span>小懿AI</span><b>检测中</b></div>
              <div class="link-health-item" id="linkHealthRl"><span>RL接口</span><b>检测中</b></div>
              <div class="link-health-item" id="linkHealthSailing"><span>航行模拟器</span><b>检测中</b></div>
            </div>
            <div id="linkHealthDetail" class="link-health-detail tiny">联动健康检查：等待 /api/rl/integration/health。</div>
            <div class="connector-line"><span>训练启动</span><b class="mono">POST /api/rl/train/start</b></div>
            <div class="connector-line"><span>状态轮询</span><b class="mono">GET /api/rl/train/status</b></div>
          <div class="connector-line"><span>指标读取</span><b class="mono">GET /api/rl/train/baselines</b></div>
          <div class="connector-line"><span>数据契约</span><b class="mono">GET /api/rl/datasets</b></div>
            <div class="connector-line"><span>联动健康</span><b class="mono">GET /api/rl/integration/health</b></div>
            <div class="connector-line"><span>模型产物</span><b class="mono">/api/rl/model/*/artifacts</b></div>
            <div class="connector-line"><span>当前产物</span><b id="artifactPathText" class="mono">等待训练启动</b></div>
            <div class="row-actions" style="margin-top:12px;">
              <button id="btnPingConnector" class="btn ghost small">刷新接入口 / 健康检查</button>
            </div>
          </div>
          <pre id="payloadPreview" class="payload-preview small mono"></pre>
        </div>
      </div>

      <div class="progress-card">
        <div class="progress-meta">
          <div>
            <div id="trainStatus" class="status-warn">WAITING</div>
            <div id="trainStage" class="progress-stage">等待启动训练 / Waiting for start</div>
          </div>
          <strong id="trainPercent">0.0%</strong>
        </div>
        <div class="progress-shell"><div id="trainProgressFill" class="progress-fill"></div></div>
        <div id="trainDetail" class="small muted">启动后会缓慢推进，并显示采样、回放池、actor/critic 更新、安全护栏评估、baseline 对比等细节。</div>
        <div class="row-actions" style="margin-top:12px;">
          <button id="btnStartTrain" class="btn">启动训练</button>
          <button id="btnPauseTrain" class="btn secondary" disabled>暂停</button>
          <button id="btnResetTrain" class="btn ghost">重置</button>
          <button id="btnPollTrainStatus" class="btn ghost">查看状态</button>
          <button id="btnEvaluateTrain" class="btn secondary" disabled>测试并渲染</button>
        </div>
        <div class="train-stat-grid">
          <div class="stat-card"><span class="k">Epoch</span><span id="statEpoch" class="v">—</span></div>
          <div class="stat-card"><span class="k">Replay</span><span id="statReplay" class="v">—</span></div>
          <div class="stat-card"><span class="k">Reward</span><span id="statReward" class="v">—</span></div>
          <div class="stat-card"><span class="k">Guardrail</span><span id="statGuardrail" class="v">—</span></div>
        </div>
        <div class="train-stat-grid">
          <div class="stat-card"><span class="k">Actor Loss</span><span id="statActor" class="v">—</span></div>
          <div class="stat-card"><span class="k">Critic Loss</span><span id="statCritic" class="v">—</span></div>
          <div class="stat-card"><span class="k">Entropy</span><span id="statEntropy" class="v">—</span></div>
          <div class="stat-card"><span class="k">Baseline Gap</span><span id="statGap" class="v">—</span></div>
        </div>
        <div class="train-stat-grid">
          <div class="stat-card"><span class="k">Step</span><span id="statStep" class="v">—</span></div>
          <div class="stat-card"><span class="k">Policy Version</span><span id="statPolicy" class="v">—</span></div>
          <div class="stat-card"><span class="k">Status Sync</span><span id="statStatusSync" class="v">—</span></div>
          <div class="stat-card"><span class="k">Last Poll</span><span id="statLastPoll" class="v">—</span></div>
        </div>
        <div id="trainRunSummary" class="train-summary small">等待训练状态轮询。</div>
        <div id="trainLog" class="train-log small mono">等待启动训练...</div>
        <div id="evaluationPanel" class="evaluation-panel">
          <div class="section-label">独立测试集回放 / Held-out Evaluation Render</div>
          <div id="evaluationMetrics" class="small muted">训练完成后才能读取测试集并渲染。</div>
          <canvas id="evaluationCanvas"></canvas>
        </div>
      </div>
    </div>
  </section>

  <section class="card">
    <div class="toolbar">
      <button id="btnLoad" class="btn">拉取策略列表</button>
      <label class="small muted">窗口(min)</label>
      <input id="inpHor" type="number" value="360" min="30" max="1440" step="30">
      <label class="small muted">步长(min)</label>
      <input id="inpStep" type="number" value="5" min="1" max="60" step="1">
    </div>
    <table>
      <thead>
        <tr>
          <th style="width:36px;">选</th>
          <th>策略</th>
          <th>影响评估</th>
          <th style="width:160px;">操作</th>
        </tr>
      </thead>
      <tbody id="tbl"></tbody>
    </table>
  </section>

  <section class="card">
    <div class="toolbar">
      <button id="btnSim" class="btn" disabled>留出集测试</button>
      <button id="btnVerifyDryRun" class="btn secondary" disabled>检查上线门禁</button>
      <button id="btnDispatch" class="btn secondary" disabled>设备执行仅走南向审批</button>
      <button id="btnHistory" class="btn ghost" disabled>查看南向能力</button>
      <span id="simHint" class="muted small">请选择左侧一条策略后再点击</span>
    </div>

    <div class="metric small">
      <div><span class="k">节电(ΔkWh)：</span><span id="m_dkwh" class="v">—</span></div>
      <div><span class="k">降碳(ΔkgCO₂e)：</span><span id="m_dco2" class="v">—</span></div>
      <div><span class="k">峰值降低(kW)：</span><span id="m_peak" class="v">—</span></div>
      <div><span class="k">窗口：</span><span id="m_win" class="v mono">—</span></div>
    </div>

    <div style="margin-top:10px">
      <canvas id="cv"></canvas>
    </div>

    <div class="subgrid">
      <div>
        <div class="panel-title">仿真结果摘要</div>
        <pre id="simRaw" class="small muted mono" style="max-height:200px;overflow:auto;"></pre>
      </div>

      <div>
        <div class="panel-title">设备执行权边界</div>
        <pre id="dispatchRaw" class="small muted mono" style="max-height:220px;overflow:auto;">未授予设备执行权</pre>
      </div>

      <div>
        <div class="panel-title">南向网关能力</div>
        <div id="historyList" class="small muted">等待读取能力声明</div>
      </div>
    </div>
  </section>
</main>
<div id="assistantConfirmBackdrop" class="confirm-backdrop" role="dialog" aria-modal="true" aria-labelledby="assistantConfirmTitle">
  <div class="confirm-dialog">
    <h2 id="assistantConfirmTitle">小懿请求执行 RL 训练</h2>
    <div id="assistantConfirmIntro" class="small muted">请确认训练目标、接口调用和风险边界。</div>
    <div class="confirm-grid">
      <div class="confirm-item"><span>自然语言指令</span><b id="confirmCommandText">—</b></div>
      <div class="confirm-item"><span>训练目标</span><b id="confirmObjectiveText">—</b></div>
      <div class="confirm-item"><span>将执行按钮</span><b>#btnStartTrain · 启动训练</b></div>
      <div class="confirm-item"><span>将调用接口</span><b>POST /api/rl/train/start</b></div>
      <div class="confirm-item"><span>算法 / 场景</span><b id="confirmAlgoScenarioText">—</b></div>
      <div class="confirm-item"><span>设备组 / 时窗</span><b id="confirmAssetHorizonText">—</b></div>
      <div class="confirm-item recommend"><span>小懿推荐参数</span><b id="confirmRecommendText">—</b></div>
    </div>
    <div class="risk-list" id="confirmRiskText"></div>
    <div class="row-actions" style="margin-top:14px;justify-content:flex-end;">
      <button id="btnCancelAssistantRun" class="btn ghost">取消</button>
      <button id="btnConfirmAssistantRun" class="btn">开始执行</button>
    </div>
  </div>
</div>

<script>
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
const query = new URLSearchParams(window.location.search);
const returnTo = query.get("return_to") || "/#strategy-exec-module";
const sourceFrom = query.get("from") || "";
const actionFrom = query.get("action") || "";
const commandFrom = query.get("command") || "";
const objectiveFrom = query.get("objective") || "";
const objectiveLabelFrom = query.get("objective_label") || "";
const advancedConfigFrom = (()=>{
  try{return JSON.parse(query.get("advanced_config") || "{}");}
  catch(_err){return {};}
})();
const recommendationTitleFrom = query.get("recommendation_title") || "";
const recommendationReasonFrom = query.get("recommendation_reason") || "";
const operatorNoteFrom = query.get("operator_note") || "";
let selectedId = null;
let currentList = [];
let lastSimulation = null;
const BASELINE_ALGOS = [
  {id:"sac", label:"SAC", type:"RL", cn:"Soft Actor-Critic", desc:"Stable-Baselines3 连续动作最大熵 actor-critic。"},
  {id:"ppo", label:"PPO", type:"RL", cn:"Proximal Policy Optimization", desc:"Stable-Baselines3 裁剪式 on-policy 策略优化。"},
  {id:"td3", label:"TD3", type:"RL", cn:"Twin Delayed DDPG", desc:"Stable-Baselines3 双 critic 连续控制。"},
  {id:"dqn", label:"DQN", type:"RL", cn:"Deep Q-Network", desc:"Stable-Baselines3 离散动作回放池 Q 学习。"},
  {id:"mpc", label:"MPC", type:"Control", cn:"模型预测控制", desc:"SciPy 约束优化的滚动时域控制基线。"}
];
const TRAIN_STAGES = [
  {cn:"任务排队", en:"Queued", detail:"后端校验算法、数据集哈希和训练配置。"},
  {cn:"训练环境就绪", en:"Environment Ready", detail:"只装载时间顺序训练段；测试留出段不可见。"},
  {cn:"真实优化器运行", en:"Optimizer Running", detail:"进度来自 Stable-Baselines3 实际 timestep 回调。"},
  {cn:"模型归档", en:"Artifact Archive", detail:"保存模型、监控日志、配置和可复现清单。"},
  {cn:"待独立测试", en:"Evaluation Pending", detail:"训练结束后才允许读取测试段并生成回放轨迹。"}
];
const OBJECTIVE_RISK = {
  multi_objective: ["多目标权重会影响能耗、碳排、成本与安全之间的取舍。", "训练过程中只进入演示任务，不直接生产下发。"],
  energy_min: ["能耗最低可能牺牲部分吞吐和设备舒适边界。", "需人工复核关键作业窗口，避免对高峰作业造成延迟。"],
  carbon_min: ["碳排最低会偏向低碳时段，可能推迟部分可延后负荷。", "需关注碳因子数据是否实时可信。"],
  cost_min: ["电费最低可能把负荷迁移到低价时段，需防止形成新的峰值。", "需人工确认电价策略和合同需量边界。"],
  peak_shaving: ["削峰目标会压低瞬时功率，可能影响岸桥/冷站响应速度。", "需关注服务水平和排队延迟。"],
  throughput_max: ["吞吐最大会更积极调用设备，可能增加能耗和峰值。", "需确认安全护栏和设备容量。"],
  delay_min: ["等待最短会优先船期，可能牺牲电费和碳排目标。", "需确认泊位和岸电资源冲突。"],
  safety_guard: ["安全优先会更保守，优化收益可能下降。", "适合高风险、台风或人工接管场景。"],
  battery_life: ["BESS 寿命友好会限制充放电深度和爬坡。", "可能降低削峰收益。"],
  shore_power_priority: ["岸电优先会增加岸电侧负荷，需要检查馈线容量。", "需确认船舶接入窗口和谐波风险。"],
  emission_quota: ["碳配额达标依赖碳因子和配额口径。", "需保留审计证据，不直接作为合规结论。"],
  resilience: ["扰动韧性会保留冗余，短期收益可能下降。", "适合天气、设备异常和船期扰动。"],
  agv_turnaround: ["AGV 周转效率会提高车辆利用率，需关注充电排队和电池寿命。", "需确认道路拥堵与安全间隔。"],
  berth_reliability: ["泊位窗口稳定会优先船期可靠性，可能牺牲部分能耗优化。", "需确认 TOS/船期数据一致。"],
  grid_stability: ["电网稳定会限制馈线和电压扰动，可能降低可调空间。", "需关注岸电、BESS、冷站同时动作风险。"],
  carbon_cost_balance: ["碳成本平衡会在电价和碳因子之间折中。", "需确认权重符合当前运营策略。"],
  low_risk_canary: ["低风险试运行仅适合灰度训练和小范围验证。", "收益较小，但便于展示人工确认边界。"],
  storm_resilience: ["台风扰动鲁棒会保守调度关键设备。", "需确认应急预案优先级高于优化收益。"]
};
let trainTimer = null;
let trainProgress = 0;
let trainTick = 0;
let trainPaused = false;
let trainJobId = null;
let trainArtifactPaths = null;
let trainStatusPollTimer = null;
let lastTrainStatus = null;
let connectorOnline = false;
let linkHealth = null;
let mobileRequestPollTimer = null;
let desktopHeartbeatTimer = null;

async function reportDesktopPanelHeartbeat(){
  try{
    await fetch("/api/rl/desktop/heartbeat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({panel:"rl-panel", operator:"港口调度员-01"})
    });
  }catch(_err){}
}
void reportDesktopPanelHeartbeat();

function mobileRequestStatusLabel(status){
  return ({
    pending_desktop_confirmation:"等待电脑端人工确认",
    approved:"已批准 · 训练已创建",
    rejected:"电脑端已拒绝"
  })[status] || status || "未知状态";
}

function applyMobileTrainingConfig(cfg={}){
  const mapping = {
    selAlgo: cfg.algorithm,
    selObjective: cfg.objective,
    selScenario: cfg.scenario,
    selAsset: cfg.asset_group,
    inpTrainHorizon: cfg.horizon_min,
    inpTrainStep: cfg.step_min,
    inpTotalSteps: cfg.total_steps,
    inpBatch: cfg.batch_size,
    inpLR: cfg.learning_rate,
    inpGamma: cfg.gamma,
    inpTau: cfg.tau,
    inpEntropy: cfg.entropy_coef,
    inpReplay: cfg.replay_buffer,
    inpSeed: cfg.seed,
    inpDemandCap: cfg.demand_cap_kw,
    selGuardrail: cfg.guardrail,
    inpCostW: cfg.cost_weight,
    inpCarbonW: cfg.carbon_weight,
    inpPeakW: cfg.peak_weight,
    inpSafetyW: cfg.safety_weight
  };
  Object.entries(mapping).forEach(([id,value])=>{
    if(value === undefined || value === null) return;
    const el = document.getElementById(id);
    if(!el) return;
    if(el.tagName === "SELECT" && !Array.from(el.options).some(opt=>opt.value===String(value))) return;
    el.value = String(value);
  });
  renderBaselineCards();
  updateConnectorPreview();
}

function renderMobileTrainingRequests(data={}){
  const list = $("#mobileTrainingRequestList");
  const items = Array.isArray(data.items) ? data.items : [];
  const pendingCount = Number(data.pending_count || 0);
  if($("#mobileRequestChip")) $("#mobileRequestChip").textContent = `待确认 ${pendingCount}`;
  if(!list) return;
  if(!items.length){
    list.innerHTML = `<div class="small muted" style="padding:12px;border:1px dashed rgba(148,163,184,.35);border-radius:12px;">暂无移动端训练申请。请先在手机“策略确认”页点击“向电脑端提交训练申请”。</div>`;
    return;
  }
  list.innerHTML = items.slice(0,6).map(item=>{
    const cfg = item.config || {};
    const pending = item.status === "pending_desktop_confirmation";
    const approved = item.status === "approved";
    const time = item.created_at ? new Date(item.created_at).toLocaleTimeString("zh-CN",{hour12:false}) : "—";
    const statusClassName = approved ? "status-ok" : (item.status === "rejected" ? "status-warn" : "status-warn");
    const actions = pending ? `
      <button class="btn" data-mobile-action="approve" data-request-id="${item.request_id}">电脑端批准并启动训练</button>
      <button class="btn ghost" data-mobile-action="reject" data-request-id="${item.request_id}">电脑端拒绝申请</button>` : "";
    return `<div style="padding:13px;border:1px solid rgba(77,228,255,.25);border-radius:14px;background:rgba(5,15,34,.52);">
      <div class="train-head" style="gap:8px;">
        <div><b class="mono">${item.request_id || "—"}</b><div class="tiny muted">手机提交 ${time} · ${item.requested_by || "mobile_operator"}</div></div>
        <div class="${statusClassName}">${mobileRequestStatusLabel(item.status)}</div>
      </div>
      <div class="train-stat-grid" style="margin-top:10px;">
        <div class="stat-card"><span class="k">算法</span><span class="v" style="font-size:14px;">${String(cfg.algorithm || "ppo").toUpperCase()}</span></div>
        <div class="stat-card"><span class="k">目标</span><span class="v" style="font-size:14px;">${cfg.objective || "berth_reliability"}</span></div>
        <div class="stat-card"><span class="k">训练步数</span><span class="v" style="font-size:14px;">${Number(cfg.total_steps || 0).toLocaleString("zh-CN")}</span></div>
        <div class="stat-card"><span class="k">安全护栏</span><span class="v" style="font-size:14px;">${cfg.guardrail || "strict"}</span></div>
      </div>
      <div class="small muted" style="margin-top:9px;">${item.policy_context?.summary || "B03 泊位拥堵候选策略训练"}${item.job_id ? ` · Job ${item.job_id}` : " · 尚未创建训练任务"}</div>
      ${actions ? `<div class="row-actions" style="margin-top:11px;">${actions}</div>` : ""}
    </div>`;
  }).join("");
}

async function loadMobileTrainingRequests(){
  try{
    const res = await fetch("/api/rl/train/requests?limit=10", {cache:"no-store"});
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderMobileTrainingRequests(data);
    return data;
  }catch(err){
    const list = $("#mobileTrainingRequestList");
    if(list) list.innerHTML = `<div class="small" style="color:#fca5a5;">移动端申请读取失败：${String(err).slice(0,160)}</div>`;
    return null;
  }
}

async function reviewMobileTrainingRequest(requestId, action){
  const operator = "港口调度员-01";
  try{
    const res = await fetch(`/api/rl/train/requests/${encodeURIComponent(requestId)}/${action}`, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({operator, reason: action === "reject" ? "电脑端人工复核后拒绝" : undefined})
    });
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if(action === "approve"){
      applyMobileTrainingConfig(data.config || {});
      trainJobId = data.job_id || data.job?.job_id || null;
      if(data.training_status) renderTrainingStatus(data.training_status, "desktop approved");
      appendTrainLog(`[${new Date().toLocaleTimeString("zh-CN",{hour12:false})}] desktop human approved · request=${requestId} · operator=${operator}`);
      await startTraining();
    }
    await loadMobileTrainingRequests();
  }catch(err){
    window.alert(`人工复核操作失败：${String(err).slice(0,180)}`);
  }
}

function numberValue(sel, fallback){
  const el = $(sel);
  if(!el) return fallback;
  const n = Number(el.value);
  return Number.isFinite(n) ? n : fallback;
}

function selectLabel(sel){
  const el = $(sel);
  return el?.selectedOptions?.[0]?.textContent || el?.value || "";
}

function trainConfig(){
  const horizon = numberValue("#inpTrainHorizon", 720);
  const stepMinutes = numberValue("#inpTrainStep", 5);
  return {
    ...advancedConfigFrom,
    algorithm: $("#selAlgo")?.value || "sac",
    algorithm_label: selectLabel("#selAlgo"),
    dataset_id: $("#selDataset")?.value || "public_port_ops_v1",
    dataset_label: selectLabel("#selDataset"),
    objective: $("#selObjective")?.value || "multi_objective",
    objective_label: selectLabel("#selObjective"),
    scenario: $("#selScenario")?.value || "mapped_dataset",
    scenario_label: selectLabel("#selScenario"),
    asset_group: $("#selAsset")?.value || "all_port",
    asset_label: selectLabel("#selAsset"),
    horizon_min: horizon,
    step_min: stepMinutes,
    episode_steps: Math.max(12, Math.min(168, Math.round(horizon / 60))),
    test_ratio: 0.20,
    total_steps: numberValue("#inpTotalSteps", 20000),
    batch_size: numberValue("#inpBatch", 256),
    learning_rate: numberValue("#inpLR", 0.0003),
    gamma: numberValue("#inpGamma", 0.995),
    tau: numberValue("#inpTau", 0.005),
    entropy_coef: numberValue("#inpEntropy", 0.02),
    replay_buffer: numberValue("#inpReplay", 120000),
    seed: numberValue("#inpSeed", 42),
    demand_cap_kw: numberValue("#inpDemandCap", 3000),
    guardrail_mode: $("#selGuardrail")?.value || "strict",
    reward_weights: {
      cost: numberValue("#inpCostW", 0.24),
      carbon: numberValue("#inpCarbonW", 0.22),
      peak: numberValue("#inpPeakW", 0.18),
      safety: numberValue("#inpSafetyW", 0.20)
    }
  };
}

function setSelectIfExists(selector, value){
  if(!value) return false;
  const el = $(selector);
  if(!el) return false;
  const found = Array.from(el.options).some(opt => opt.value === value);
  if(found){
    el.value = value;
    return true;
  }
  return false;
}

function setInputFromQuery(selector, key){
  const value = query.get(key);
  if(value === null || value === "") return false;
  const el = $(selector);
  if(!el) return false;
  el.value = value;
  return true;
}

function applyAssistantTrainingParams(){
  setSelectIfExists("#selObjective", objectiveFrom);
  setSelectIfExists("#selAlgo", query.get("algorithm"));
  setSelectIfExists("#selScenario", query.get("scenario"));
  setSelectIfExists("#selAsset", query.get("asset_group"));
  setSelectIfExists("#selGuardrail", query.get("guardrail_mode"));
  setInputFromQuery("#inpTrainHorizon", "horizon_min");
  setInputFromQuery("#inpTrainStep", "step_min");
  setInputFromQuery("#inpTotalSteps", "total_steps");
  setInputFromQuery("#inpBatch", "batch_size");
  setInputFromQuery("#inpLR", "learning_rate");
  setInputFromQuery("#inpGamma", "gamma");
  setInputFromQuery("#inpTau", "tau");
  setInputFromQuery("#inpEntropy", "entropy_coef");
  setInputFromQuery("#inpReplay", "replay_buffer");
  setInputFromQuery("#inpDemandCap", "demand_cap_kw");
  setInputFromQuery("#inpCostW", "cost_w");
  setInputFromQuery("#inpCarbonW", "carbon_w");
  setInputFromQuery("#inpPeakW", "peak_w");
  setInputFromQuery("#inpSafetyW", "safety_w");
  if(commandFrom && $("#trainDetail")){
    $("#trainDetail").textContent = `${operatorNoteFrom || "小懿已载入推荐训练参数"} · 指令：${commandFrom}`;
  }
  updateConnectorPreview();
  renderBaselineCards();
}

function recommendationSummaryText(cfg){
  const title = recommendationTitleFrom || "推荐训练参数";
  const reason = recommendationReasonFrom || "根据训练目标自动选择算法、窗口、护栏和 reward 权重。";
  const weights = cfg.reward_weights || {};
  return `${title}：${reason} 参数：算法=${cfg.algorithm_label}，场景=${cfg.scenario_label}，设备=${cfg.asset_label}，horizon=${cfg.horizon_min}min，step=${cfg.step_min}min，total_steps=${cfg.total_steps.toLocaleString("zh-CN")}，batch=${cfg.batch_size}，lr=${cfg.learning_rate}，gamma=${cfg.gamma}，entropy=${cfg.entropy_coef}，guardrail=${cfg.guardrail_mode}，权重 cost=${weights.cost} / carbon=${weights.carbon} / peak=${weights.peak} / safety=${weights.safety}`;
}

function confirmationSummary(){
  const cfg = trainConfig();
  const risks = OBJECTIVE_RISK[cfg.objective] || OBJECTIVE_RISK.multi_objective;
  return {
    command: commandFrom || "小懿，开始 RL 训练",
    objective: objectiveLabelFrom || cfg.objective_label,
    algoScenario: `${cfg.algorithm_label} · ${cfg.scenario_label}`,
    assetHorizon: `${cfg.asset_label} · horizon=${cfg.horizon_min}min · step=${cfg.step_min}min`,
    recommendation: recommendationSummaryText(cfg),
    risks
  };
}

function showAssistantRunConfirm(){
  const box = $("#assistantConfirmBackdrop");
  if(!box) return;
  const s = confirmationSummary();
  $("#confirmCommandText").textContent = s.command;
  $("#confirmObjectiveText").textContent = s.objective;
  $("#confirmAlgoScenarioText").textContent = s.algoScenario;
  $("#confirmAssetHorizonText").textContent = s.assetHorizon;
  $("#confirmRecommendText").textContent = s.recommendation;
  $("#confirmRiskText").innerHTML = `<b>执行风险与边界</b><br>${s.risks.map(x=>`• ${x}`).join("<br>")}<br>• 点击“开始执行”后才会调用 /api/rl/train/start；训练结果仍需策略测试、安全校验和 dry-run，不能直接生产执行。`;
  box.style.display = "flex";
  appendTrainLog(`[${new Date().toLocaleTimeString("zh-CN",{hour12:false})}] assistant confirmation pending · ${s.objective}`);
}

function hideAssistantRunConfirm(){
  const box = $("#assistantConfirmBackdrop");
  if(box) box.style.display = "none";
}

function renderBaselineCards(){
  const grid = $("#baselineGrid");
  if(!grid) return;
  const active = trainConfig().algorithm;
  grid.innerHTML = BASELINE_ALGOS.map(algo => `
    <button class="algo-card ${algo.id===active?'active':''}" data-algo="${algo.id}" type="button">
      <div class="algo-top">
        <span class="algo-name">${algo.label}</span>
        <span class="pill ${algo.type==='Control'?'rule':''}">${algo.type}</span>
      </div>
      <div class="algo-desc"><b>${algo.cn}</b><br>${algo.desc}</div>
      <div class="algo-metrics">
        <span>Reward <b id="algo_${algo.id}_reward">—</b></span>
        <span>Carbon <b id="algo_${algo.id}_carbon">—</b></span>
        <span>Peak <b id="algo_${algo.id}_peak">—</b></span>
        <span>Status <b id="algo_${algo.id}_status">LOADING</b></span>
      </div>
    </button>
  `).join("");
  $$("#baselineGrid .algo-card").forEach(card=>{
    card.addEventListener("click", ()=>{
      $("#selAlgo").value = card.dataset.algo;
      renderBaselineCards();
      updateConnectorPreview();
    });
  });
  void updateBaselineMetrics();
}

function setProgress(value){
  trainProgress = Math.max(0, Math.min(100, value));
  const fill = $("#trainProgressFill");
  if(fill) fill.style.width = trainProgress.toFixed(2) + "%";
  if($("#trainPercent")) $("#trainPercent").textContent = trainProgress.toFixed(1) + "%";
}

function appendTrainLog(text){
  const el = $("#trainLog");
  if(!el) return;
  const old = el.textContent === "等待启动训练..." ? [] : el.textContent.split("\n");
  const lines = [text, ...old].slice(0, 42);
  el.textContent = lines.join("\n");
}

function currentStage(progress){
  const idx = Math.min(TRAIN_STAGES.length - 1, Math.floor((progress / 100) * TRAIN_STAGES.length));
  return TRAIN_STAGES[idx];
}

function artifactLabel(paths){
  if(!paths) return "/api/rl/model/{model}/artifacts";
  if(typeof paths === "string") return paths;
  return paths.model_artifacts_url || paths.artifact_url || paths.root_url || paths.root || paths.local_path || "/api/rl/model/{model}/artifacts";
}

function setArtifactPath(paths){
  trainArtifactPaths = paths || null;
  const el = $("#artifactPathText");
  if(el) el.textContent = trainArtifactPaths ? artifactLabel(trainArtifactPaths) : "等待训练启动";
}

function finiteNumber(value, fallback){
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function trainLogLines(limit=14){
  const text = $("#trainLog")?.textContent || "";
  if(!text || text === "等待启动训练...") return [];
  return text.split("\n").map(line=>line.trim()).filter(Boolean).slice(0, limit);
}

function statusSummaryText(status){
  const metrics = status?.metrics || {};
  const step = finiteNumber(status?.step ?? metrics.step, 0);
  const total = finiteNumber(status?.total_steps ?? metrics.total_steps, trainConfig().total_steps);
  const reward = metrics.reward_mean;
  const state = status?.status || "IDLE";
  const logs = Array.isArray(status?.logs) ? status.logs.length : 0;
  const rewardText = Number.isFinite(Number(reward)) ? Number(reward).toFixed(5) : "—";
  return `运行摘要：${state} · step=${step.toLocaleString("zh-CN")} / ${total.toLocaleString("zh-CN")} · real reward mean=${rewardText} · logs=${logs}`;
}

function renderTrainingStatus(status, sourceLabel="poll"){
  if(!status) return;
  lastTrainStatus = status;
  const metrics = status.metrics || {};
  const step = finiteNumber(status.step ?? metrics.step, 0);
  const reward = metrics.reward_mean;
  const entropy = metrics.entropy_loss;
  const policyVersion = status.job_id ? `${String(status.algorithm || trainConfig().algorithm).toUpperCase()} · ${status.job_id}` : "—";
  if(status.job_id) trainJobId = status.job_id;
  if(status.artifact_paths) setArtifactPath(status.artifact_paths);
  if(Number.isFinite(Number(status.progress))) setProgress(Number(status.progress));
  if($("#trainStatus") && status.status){
    $("#trainStatus").textContent = status.status;
    $("#trainStatus").className = ["RUNNING","COMPLETED","EVALUATED"].includes(status.status) ? "status-ok" : "status-warn";
  }
  if($("#trainStage") && status.stage) $("#trainStage").textContent = status.stage;
  if($("#statStep")) $("#statStep").textContent = step.toLocaleString("zh-CN");
  if($("#statPolicy")) $("#statPolicy").textContent = policyVersion;
  if($("#statStatusSync")) $("#statStatusSync").textContent = sourceLabel;
  if($("#statLastPoll")) $("#statLastPoll").textContent = new Date().toLocaleTimeString("zh-CN",{hour12:false});
  const showMetric = (selector, value, digits=5)=>{ const el=$(selector); if(el) el.textContent=Number.isFinite(Number(value))?Number(value).toFixed(digits):"N/A"; };
  showMetric("#statReward", reward);
  showMetric("#statEntropy", entropy);
  showMetric("#statActor", metrics.actor_loss ?? metrics.policy_gradient_loss);
  showMetric("#statCritic", metrics.critic_loss ?? metrics.value_loss);
  if($("#statReplay")) $("#statReplay").textContent = metrics.updates == null ? "N/A" : Number(metrics.updates).toLocaleString("zh-CN");
  if($("#statEpoch")) $("#statEpoch").textContent = metrics.updates == null ? "N/A" : Number(metrics.updates).toLocaleString("zh-CN");
  if($("#statGuardrail")) $("#statGuardrail").textContent = status.rendering?.render_calls === 0 ? "TRAIN RENDER=0" : "N/A";
  if($("#statGap")) $("#statGap").textContent = status.evaluation?.metrics ? "TESTED" : "待独立测试";
  if($("#trainLog") && Array.isArray(status.logs) && status.logs.length) $("#trainLog").textContent = status.logs.join("\n");
  const terminal = ["COMPLETED","EVALUATED","FAILED","CANCELLED","INTERRUPTED"].includes(status.status);
  if($("#btnStartTrain")) $("#btnStartTrain").disabled = !terminal && status.status !== "IDLE";
  if($("#btnPauseTrain")) $("#btnPauseTrain").disabled = !["RUNNING","PAUSED"].includes(status.status);
  if($("#btnEvaluateTrain")) $("#btnEvaluateTrain").disabled = !status.evaluation_available;
  if(terminal) stopStatusPolling();
  if($("#trainRunSummary")) $("#trainRunSummary").textContent = status.summary || statusSummaryText(status);
  updateConnectorPreview();
}

async function pollTrainingStatus(options={}){
  const qs = trainJobId ? `?job_id=${encodeURIComponent(trainJobId)}` : "";
  try{
    const res = await fetch(`/api/rl/train/status${qs}`, {cache:"no-store"});
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const status = data.status || data;
    renderTrainingStatus(status, options.sourceLabel || "polled");
    if(options.log){
      const metrics = status.metrics || {};
      appendTrainLog(`[${new Date().toLocaleTimeString("zh-CN",{hour12:false})}] backend status · step=${finiteNumber(status.step ?? metrics.step, 0).toLocaleString("zh-CN")} · reward_mean=${metrics.reward_mean ?? "N/A"}`);
    }
    return status;
  }catch(err){
    if($("#trainRunSummary")) $("#trainRunSummary").textContent = `运行摘要读取失败：${String(err).slice(0,120)}`;
    if($("#statStatusSync")) $("#statStatusSync").textContent = "poll failed";
    return null;
  }
}

function startStatusPolling(){
  if(trainStatusPollTimer) return;
  trainStatusPollTimer = window.setInterval(()=>{ void pollTrainingStatus(); }, 1500);
}

function stopStatusPolling(){
  if(trainStatusPollTimer) clearInterval(trainStatusPollTimer);
  trainStatusPollTimer = null;
}

function updateConnectorPreview(){
  const cfg = trainConfig();
  const payload = {
    config: cfg,
    baselines: BASELINE_ALGOS.map(a=>({id:a.id, name:a.label, type:a.type, enabled:true})),
    connector: {
      train_start: "/api/rl/train/start",
      training_status: "/api/rl/train/status",
      training_metrics: "/api/rl/train/metrics",
      baseline_metrics: "/api/rl/train/baselines",
      integration_health: "/api/rl/integration/health",
      artifact_root: artifactLabel(trainArtifactPaths),
      artifact_paths: trainArtifactPaths
    },
    linked_systems: linkHealth ? linkHealth.summary : {
      xiaoyi: "等待健康检查",
      rl: "等待健康检查",
      sailing: "等待健康检查"
    },
    progress_source: "Stable-Baselines3 callback",
    rendering: "disabled during training; enabled only by held-out evaluation endpoint",
    notes: "The UI only polls backend-owned metrics. No local progress or score generation."
  };
  if($("#payloadPreview")) $("#payloadPreview").textContent = JSON.stringify(payload, null, 2);
  if($("#trainJobChip")){
    $("#trainJobChip").textContent = trainJobId
      ? `Job ${trainJobId} · ${lastTrainStatus?.status || (trainPaused ? 'PAUSED' : 'QUEUED')}`
      : "未启动 · IDLE";
  }
  if($("#connectorStatus") && !trainJobId){
    $("#connectorStatus").textContent = connectorOnline ? "接入口状态：已连接 · 待启动训练" : "接入口状态：待启动训练";
    $("#connectorStatus").className = connectorOnline ? "status-ok" : "status-warn";
  }
}

function setLinkHealthItem(id, label, ok){
  const item = document.getElementById(id);
  if(!item) return;
  item.className = `link-health-item ${ok ? "ok" : "bad"}`;
  const b = item.querySelector("b");
  if(b) b.textContent = label || (ok ? "在线" : "不可用");
}

function renderLinkHealth(data){
  linkHealth = data || null;
  const systems = data?.systems || {};
  const xiaoyi = systems.xiaoyi_ai || {};
  const rl = systems.rl_interface || {};
  const sailing = systems.sailing_simulator || {};
  setLinkHealthItem("linkHealthXiaoyi", xiaoyi.label || "小懿未启动", Boolean(xiaoyi.online));
  setLinkHealthItem("linkHealthRl", rl.label || "RL接口缺失", Boolean(rl.online));
  setLinkHealthItem("linkHealthSailing", sailing.label || "航行模拟器不可启动", Boolean(sailing.launchable));
  const detail = $("#linkHealthDetail");
  if(detail){
    const time = data?.updated_at ? new Date(data.updated_at).toLocaleTimeString("zh-CN",{hour12:false}) : new Date().toLocaleTimeString("zh-CN",{hour12:false});
    const xiaoyiUrl = xiaoyi.base_url || "未配置";
    const project = sailing.project_file?.path || sailing.project_root?.path || "未配置";
    detail.textContent = `最近检查 ${time} · 小懿 ${xiaoyiUrl} · 航行模拟器 ${project}`;
  }
  updateConnectorPreview();
}

async function refreshLinkHealth(){
  try{
    const res = await fetch("/api/rl/integration/health", {cache:"no-store"});
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderLinkHealth(data);
    return data;
  }catch(err){
    renderLinkHealth({
      systems: {
        xiaoyi_ai: {label:"小懿未启动", online:false},
        rl_interface: {label:"RL接口待检查", online:false},
        sailing_simulator: {label:"航行模拟器待检查", launchable:false}
      },
      summary: {
        xiaoyi:"小懿未启动",
        rl:"RL接口待检查",
        sailing:"航行模拟器待检查"
      }
    });
    const detail = $("#linkHealthDetail");
    if(detail) detail.textContent = `联动健康检查失败：${String(err).slice(0,140)}`;
    return null;
  }
}

async function refreshConnector(){
  const el = $("#connectorStatus");
  await refreshLinkHealth();
  try{
    const res = await fetch(`/api/rl/train/baselines?dataset_id=${encodeURIComponent(trainConfig().dataset_id)}`, {cache:"no-store"});
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    connectorOnline = true;
    if(el){
      el.textContent = `接入口状态：已连接 · baseline=${(data.baselines||[]).length}`;
      el.className = "status-ok";
    }
    await updateBaselineMetrics();
  }catch(err){
    connectorOnline = false;
    if(el){
      el.textContent = "接入口状态：本地 UI 可演示，后端接入口暂不可用";
      el.className = "status-warn";
    }
  }
  updateConnectorPreview();
}

async function requestTrainStart(cfg){
  const res = await fetch("/api/rl/train/start", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({config: cfg, baselines: BASELINE_ALGOS, source: "rl-panel"})
  });
  if(!res.ok) throw new Error(await res.text());
  const data = await res.json();
  connectorOnline = true;
  trainJobId = data.job_id;
  setArtifactPath(data.artifact_paths || null);
  if($("#connectorStatus")){
    $("#connectorStatus").textContent = `接入口状态：真实训练任务已接收 · ${trainJobId}`;
    $("#connectorStatus").className = "status-ok";
  }
  appendTrainLog(`[${new Date().toLocaleTimeString("zh-CN",{hour12:false})}] backend accepted · job=${trainJobId}`);
  updateConnectorPreview();
  return data;
}

async function updateBaselineMetrics(){
  const set = (id, value)=>{ const el = document.getElementById(id); if(el) el.textContent = value; };
  try{
    const dataset = encodeURIComponent(trainConfig().dataset_id);
    const response = await fetch(`/api/rl/train/baselines?dataset_id=${dataset}`, {cache:"no-store"});
    if(!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    (payload.baselines || []).forEach(item=>{
      const metrics = item.latest_evaluation?.metrics || {};
      set(`algo_${item.id}_reward`, Number.isFinite(Number(metrics.reward)) ? Number(metrics.reward).toFixed(4) : "—");
      set(`algo_${item.id}_carbon`, Number.isFinite(Number(metrics.carbon_kg)) ? Number(metrics.carbon_kg).toFixed(1) : "—");
      set(`algo_${item.id}_peak`, Number.isFinite(Number(metrics.peak_kw)) ? Number(metrics.peak_kw).toFixed(1) : "—");
      set(`algo_${item.id}_status`, item.status || "UNTRAINED");
    });
  }catch(_error){
    BASELINE_ALGOS.forEach(item=>set(`algo_${item.id}_status`, "API ERROR"));
  }
}

async function startTraining(){
  const activeStatus = lastTrainStatus?.status;
  if(["QUEUED","RUNNING","PAUSED"].includes(activeStatus)) return;
  const cfg = trainConfig();
  trainPaused = false;
  trainJobId = null;
  lastTrainStatus = null;
  $("#trainLog").textContent = "";
  $("#evaluationPanel").style.display = "none";
  $("#btnEvaluateTrain").disabled = true;
  setProgress(0);
  $("#btnStartTrain").disabled = true;
  try{
    const status = await requestTrainStart(cfg);
    renderTrainingStatus(status, "accepted");
    appendTrainLog(`[${new Date().toLocaleTimeString("zh-CN",{hour12:false})}] backend training started · ${cfg.algorithm_label} · dataset=${cfg.dataset_id}`);
    startStatusPolling();
  }catch(err){
    $("#btnStartTrain").disabled = false;
    $("#trainStatus").textContent = "FAILED";
    $("#trainRunSummary").textContent = `训练任务未创建：${String(err).slice(0,220)}`;
  }
}

async function pauseTraining(){
  if(!trainJobId || !["RUNNING","PAUSED"].includes(lastTrainStatus?.status)) return;
  const action = lastTrainStatus.status === "PAUSED" ? "resume" : "pause";
  try{
    const response = await fetch(`/api/rl/train/${encodeURIComponent(trainJobId)}/control`, {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({action})
    });
    if(!response.ok) throw new Error(await response.text());
    const status = await response.json();
    trainPaused = status.status === "PAUSED";
    $("#btnPauseTrain").textContent = trainPaused ? "继续" : "暂停";
    renderTrainingStatus(status, "control");
  }catch(err){ appendTrainLog(`control failed · ${String(err).slice(0,160)}`); }
}

async function resetTraining(){
  if(trainJobId && ["QUEUED","RUNNING","PAUSED"].includes(lastTrainStatus?.status)){
    try{
      await fetch(`/api/rl/train/${encodeURIComponent(trainJobId)}/control`, {
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({action:"cancel"})
      });
    }catch(_err){}
  }
  hideAssistantRunConfirm();
  stopStatusPolling();
  trainTimer = null;
  trainPaused = false;
  trainProgress = 0;
  trainTick = 0;
  trainJobId = null;
  lastTrainStatus = null;
  setArtifactPath(null);
  setProgress(0);
  if($("#trainStatus")){
    $("#trainStatus").textContent = "WAITING";
    $("#trainStatus").className = "status-warn";
  }
  if($("#trainStage")) $("#trainStage").textContent = "等待启动训练 / Waiting for start";
  if($("#trainDetail")) $("#trainDetail").textContent = "训练仅消费时间顺序训练集且不渲染；完成后点击“测试并渲染”才读取独立测试集。";
  ["#statEpoch","#statReplay","#statReward","#statGuardrail","#statActor","#statCritic","#statEntropy","#statGap","#statStep","#statPolicy","#statStatusSync","#statLastPoll"].forEach(sel=>{
    const el = $(sel); if(el) el.textContent = "—";
  });
  if($("#btnStartTrain")) $("#btnStartTrain").disabled = false;
  if($("#btnPauseTrain")){
    $("#btnPauseTrain").disabled = true;
    $("#btnPauseTrain").textContent = "暂停";
  }
  if($("#trainRunSummary")) $("#trainRunSummary").textContent = "等待训练状态轮询。";
  if($("#trainLog")) $("#trainLog").textContent = "等待启动训练...";
  if($("#evaluationPanel")) $("#evaluationPanel").style.display = "none";
  if($("#btnEvaluateTrain")) $("#btnEvaluateTrain").disabled = true;
  updateConnectorPreview();
  renderBaselineCards();
}

async function handleAssistantAction(){
  const action = String(actionFrom || "").toLowerCase();
  if(action === "start_rl_training"){
    applyAssistantTrainingParams();
    if($("#trainDetail")) $("#trainDetail").textContent = "小懿指令已映射到 RL 训练面板；请在确认框中核对详情和风险。";
    showAssistantRunConfirm();
  }else if(action === "view_rl_training_status"){
    if($("#trainDetail")) $("#trainDetail").textContent = "小懿指令已映射到 /api/rl/train/status，正在读取训练状态和指标。";
    await pollTrainingStatus({log:true, sourceLabel:"xiaoyi query"});
  }else if(action === "run_policy_test"){
    if($("#trainDetail")) $("#trainDetail").textContent = "小懿指令已映射到真实留出集评测，评测完成后才会返回轨迹。";
    await runPolicyTestFromArtifact();
  }else if(action === "verify_policy_for_online"){
    if($("#trainDetail")) $("#trainDetail").textContent = "小懿指令已映射到模型登记与上线证据门禁；本步不执行设备指令。";
    await verifyPolicyForOnline({source:"xiaoyi_verify_online"});
  }
}

function applyReturnBanner(){
  const banner = $("#returnBanner");
  const text = $("#returnBannerText");
  if(!banner || !text) return;
  if(sourceFrom === 'strategy'){
    banner.style.display = 'block';
    text.textContent = '当前从主平台进入。建议顺序：拉取模型登记 → 留出集评测 → 上线证据门禁 → 回主平台查看审计与孪生支撑。';
  }else{
    banner.style.display = 'block';
    text.textContent = '当前为独立 RL 面板。训练不渲染；完成后可在留出集上评测并渲染轨迹，再检查模型上线门禁。';
  }
}

function goBackTo(target){
  const url = target || returnTo || '/#strategy-exec-module';
  try{
    window.location.href = url;
  }catch(err){
    window.location.href = '/#strategy-exec-module';
  }
}

async function loadTrainingDatasets(){
  const select = $("#selDataset");
  if(!select) return;
  const response = await fetch("/api/rl/datasets", {cache:"no-store"});
  if(!response.ok) throw new Error(await response.text());
  const payload = await response.json();
  const valid = (payload.datasets || []).filter(item=>item.valid !== false);
  if(!valid.length) throw new Error("没有通过字段契约校验的训练数据集");
  select.innerHTML = valid.map(item=>{
    const source = item.provenance_type || "mapped_dataset";
    return `<option value="${item.dataset_id}">${item.dataset_id} · ${Number(item.rows||0).toLocaleString("zh-CN")} rows · ${source}</option>`;
  }).join("");
  if(valid.some(item=>item.dataset_id === "public_port_ops_v1")) select.value = "public_port_ops_v1";
  updateConnectorPreview();
}

function drawEvaluationTrajectory(frames){
  const canvas = $("#evaluationCanvas");
  if(!canvas || !frames?.length) return;
  const width = canvas.clientWidth || 640;
  const height = canvas.clientHeight || 180;
  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0,0,width,height);
  const values = frames.flatMap(item=>[Number(item.baseline_kw||0),Number(item.net_load_kw||0)]);
  const min = Math.min(...values), max = Math.max(...values);
  const pad = 24, span = Math.max(1,max-min);
  const line = (key,color)=>{
    ctx.beginPath();
    frames.forEach((item,index)=>{
      const x = pad + index / Math.max(1,frames.length-1) * (width-2*pad);
      const y = height-pad - (Number(item[key]||0)-min)/span*(height-2*pad);
      if(index===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke();
  };
  ctx.strokeStyle="#1f3b34"; ctx.strokeRect(pad,pad,width-2*pad,height-2*pad);
  line("baseline_kw","#60a5fa");
  line("net_load_kw","#34d399");
  ctx.fillStyle="#bfdbfe"; ctx.fillText("公开/映射数据基线",pad+6,pad+14);
  ctx.fillStyle="#bbf7d0"; ctx.fillText("测试策略净负荷",pad+132,pad+14);
}

async function evaluateTraining(){
  if(!trainJobId || !lastTrainStatus?.evaluation_available) return;
  const button = $("#btnEvaluateTrain");
  button.disabled = true;
  $("#trainStage").textContent = "读取独立测试集并生成回放 / Held-out evaluation";
  try{
    const response = await fetch(`/api/rl/train/${encodeURIComponent(trainJobId)}/evaluate`, {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({episodes:10})
    });
    if(!response.ok) throw new Error(await response.text());
    const result = await response.json();
    const metrics = result.metrics || {};
    $("#evaluationPanel").style.display = "block";
    $("#evaluationMetrics").textContent = `真实测试集 · ${result.algorithm.toUpperCase()} · reward=${Number(metrics.reward||0).toFixed(5)} · cost=${Number(metrics.energy_cost||0).toFixed(2)} · carbon=${Number(metrics.carbon_kg||0).toFixed(2)} kg · peak=${Number(metrics.peak_kw||0).toFixed(2)} kW · delay=${Number(metrics.delay_index_mean||0).toFixed(4)} · violations=${(100*Number(metrics.guardrail_violation_rate||0)).toFixed(2)}% · frames=${result.render?.frame_count||0}`;
    drawEvaluationTrajectory(result.render?.frames || []);
    await pollTrainingStatus({sourceLabel:"evaluation"});
    await updateBaselineMetrics();
  }catch(err){
    $("#evaluationPanel").style.display = "block";
    $("#evaluationMetrics").textContent = `测试失败：${String(err).slice(0,220)}`;
    button.disabled = false;
  }
}

function fmtImpact(imp){
  if(!imp) return "-";
  const value = (raw,digits=3)=>Number.isFinite(Number(raw))?Number(raw).toFixed(digits):"N/A";
  return `reward:${value(imp.reward,5)} · peak:${value(imp.peak_kw,2)}kW · violations:${Number.isFinite(Number(imp.guardrail_violation_rate))?(100*Number(imp.guardrail_violation_rate)).toFixed(2)+"%":"N/A"}`;
}

function rowHTML(item, idx){
  const cat = item.category || "-";
  const win = item.window ? `${item.window.start.split('T')[1]?.slice(0,5)}~${item.window.end.split('T')[1]?.slice(0,5)}` : "-";
  const ex = item.explain?.reason || "";
  return `
    <tr>
      <td><input type="radio" name="pick" value="${item.id}" ${idx===0?'checked':''}></td>
      <td>
        <div><span class="badge">${cat}</span> ${item.title}</div>
        <div class="muted small">${ex} <span class="mono">[${win}]</span></div>
      </td>
      <td class="small">${fmtImpact(item.impact)}</td>
      <td class="row-actions">
        <button class="btn small" data-simid="${item.id}">留出集测试</button>
      </td>
    </tr>`;
}

function drawChart(a,b){
  const cv = $("#cv"); const g = cv.getContext("2d");
  const W = cv.clientWidth, H = cv.clientHeight; cv.width=W; cv.height=H;
  g.clearRect(0,0,W,H);
  const pad=24; const w=W-2*pad, h=H-2*pad;
  const all = [...a,...b];
  const n = Math.max(a.length, b.length, 1);
  const maxv = Math.max(1, ...all, 1);
  g.strokeStyle="#1f2b45"; g.strokeRect(pad,pad,w,h);
  g.strokeStyle="#132035"; g.lineWidth=1;
  for(let i=0;i<=4;i++){ const y=pad+i*h/4; g.beginPath(); g.moveTo(pad,y); g.lineTo(W-pad,y); g.stroke(); }
  function pl(arr,color){
    if(!arr.length) return;
    g.beginPath();
    arr.forEach((v,i)=>{
      const x = pad + (i/(n-1||1))*w;
      const y = pad + h - (v/maxv)*h;
      if(i===0) g.moveTo(x,y); else g.lineTo(x,y);
    });
    g.strokeStyle=color; g.lineWidth=2; g.stroke();
  }
  pl(a,"#60a5fa");
  pl(b,"#34d399");
  g.fillStyle="#60a5fa"; g.fillRect(pad+8,pad+8,10,10);
  g.fillStyle="#e5e7eb"; g.fillText("基线", pad+24, pad+18);
  g.fillStyle="#34d399"; g.fillRect(pad+70,pad+8,10,10);
  g.fillStyle="#e5e7eb"; g.fillText("策略后", pad+86, pad+18);
}

function pickStrategy(strategyId){
  selectedId = strategyId;
  $$("input[name='pick']").forEach(r=>{ r.checked = (r.value===selectedId); });
  $("#btnSim").disabled = !selectedId;
  if($("#btnVerifyDryRun")) $("#btnVerifyDryRun").disabled = !selectedId;
  $("#btnDispatch").disabled = true;
  $("#simHint").textContent = selectedId ? ("已选中登记模型：" + selectedId + "，可执行留出集测试。") : "请先选择一个已登记模型";
  $("#dispatchRaw").textContent = "未授予设备执行权";
}

async function ensureStrategyList(){
  if(!currentList.length){
    await loadList();
  }
  if(!selectedId && currentList.length){
    pickStrategy(currentList[0].id);
  }
  return currentList.find(item=>item.id === selectedId) || currentList[0] || null;
}

async function latestPolicyArtifactInfo(){
  const status = await pollTrainingStatus({sourceLabel:"artifact read"});
  const metrics = status?.metrics || {};
  return {
    job_id: status?.job_id || trainJobId,
    training_status: status?.status || "UNKNOWN",
    policy_version: status?.policy_version || metrics.policy_version || $("#statPolicy")?.textContent || "—",
    artifact_paths: status?.artifact_paths || trainArtifactPaths,
    artifact_root: artifactLabel(status?.artifact_paths || trainArtifactPaths),
    summary: status?.summary || $("#trainRunSummary")?.textContent || ""
  };
}

async function loadList(){
  $("#btnLoad").disabled = true;
  $("#tbl").innerHTML = `<tr><td colspan="4" class="muted small">加载中...</td></tr>`;
  try{
    const h = Number($("#inpHor").value||360);
    const s = Number($("#inpStep").value||5);
    const res = await fetch(`/api/rl/strategies?horizon_min=${h}&step_min=${s}&max_items=12`);
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    currentList = data.strategies || [];
    if(currentList.length===0){
      $("#tbl").innerHTML = `<tr><td colspan="4" class="muted small">暂无策略。</td></tr>`;
      $("#btnSim").disabled = true; $("#btnDispatch").disabled = true; if($("#btnVerifyDryRun")) $("#btnVerifyDryRun").disabled = true; selectedId=null; return;
    }
    $("#tbl").innerHTML = currentList.map((it,i)=>rowHTML(it,i)).join("");
    pickStrategy(currentList[0].id);
    $$("input[name='pick']").forEach(r=>{
      r.addEventListener("change", e=> pickStrategy(e.target.value));
    });
    $$("#tbl button[data-simid]").forEach(b=>{
      b.addEventListener("click", async ()=>{
        pickStrategy(b.dataset.simid);
        await simulate();
      });
    });
  }catch(err){
    $("#tbl").innerHTML = `<tr><td colspan="4" class="small" style="color:#ef4444;">加载失败：${String(err).slice(0,200)}</td></tr>`;
  }finally{
    $("#btnLoad").disabled = false;
  }
}

async function simulate(options={}){
  if(!selectedId) return;
  $("#btnSim").disabled = true;
  $("#btnDispatch").disabled = true;
  if($("#btnVerifyDryRun")) $("#btnVerifyDryRun").disabled = true;
  $("#simHint").textContent = "正在独立留出集上测试...";
  try{
    const payload = {strategy_id: selectedId, horizon_min: 360, step_min: 1};
    const res = await fetch("/api/rl/simulate", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    lastSimulation = data;
    const metricText = (value,digits)=>Number.isFinite(Number(value))?Number(value).toFixed(digits):"N/A";
    $("#m_dkwh").textContent = metricText(data.summary?.delta_kWh,3);
    $("#m_dco2").textContent = metricText(data.summary?.delta_carbon_kg,3);
    $("#m_peak").textContent = metricText(data.summary?.peak_reduction_kW,2);
    const w = data.summary?.window || {};
    $("#m_win").textContent = (w.start||"--")+" ~ "+(w.end||"--");
    drawChart(data.baseline?.agg_kW||[], data.simulated?.agg_kW||[]);

    const pretty = {
      triggered_by: options.source || "manual",
      latest_policy_artifact: options.artifact || null,
      summary: data.summary || {},
      feasibility: data.feasibility || {},
      contributors: data.contributors || [],
      audit_trace: data.audit_trace || {}
    };
    $("#simRaw").textContent = JSON.stringify(pretty, null, 2);

    $("#btnDispatch").disabled = true;
    if($("#btnVerifyDryRun")) $("#btnVerifyDryRun").disabled = false;
    $("#simHint").textContent = `留出集测试完成：${selectedId}；请继续检查模型上线门禁，本结果不授予设备执行权。`;
    return data;
  }catch(err){
    lastSimulation = null;
    $("#simRaw").textContent = "模拟失败："+String(err).slice(0,400);
    $("#simHint").textContent = "模拟失败，请检查接口日志";
    return null;
  }finally{
    $("#btnSim").disabled = false;
    if($("#btnVerifyDryRun")) $("#btnVerifyDryRun").disabled = !selectedId;
  }
}

function statusClass(status){
  if((status||"").includes("REJECT")) return "status-bad";
  if((status||"").includes("CANCEL")) return "status-warn";
  return "status-ok";
}

async function runPolicyTestFromArtifact(){
  $("#simHint").textContent = "小懿正在读取最新 policy artifact...";
  const artifact = await latestPolicyArtifactInfo();
  await ensureStrategyList();
  if(!selectedId){
    $("#simHint").textContent = "没有可测试策略。";
    return null;
  }
  $("#simHint").textContent = `小懿策略测试：artifact=${artifact.policy_version || "latest"}，策略=${selectedId}`;
  const data = await simulate({source:"xiaoyi_policy_test", artifact});
  if(data){
    const reward = data.evaluation?.metrics?.reward;
    $("#simHint").textContent = `留出集测试完成：${selectedId} · reward=${Number.isFinite(Number(reward))?Number(reward).toFixed(5):"N/A"} · 设备执行权=未授予`;
  }
  return data;
}

async function dispatchSelected(options={}){
  const response = await fetch("/api/actuators/capabilities", {cache:"no-store"});
  const capability = response.ok ? await response.json() : {enabled:false, mode:"fail_closed"};
  const result = {
    status:"blocked_by_design",
    strategy_id:selectedId,
    production_boundary:"此页只做评测和门禁检查；设备指令必须经 /api/actuators/rl-stage 另行生成并由异人二通道确认。",
    actuator_capability:capability,
  };
  $("#dispatchRaw").textContent = JSON.stringify(result, null, 2);
  $("#simHint").textContent = "未执行设备指令；请由现场流程单独提交南向审批。";
  $("#btnDispatch").disabled = true;
  return result;
}

async function verifyPolicyForOnline(options={}){
  $("#simHint").textContent = "正在读取模型登记和上线证据门禁...";
  await ensureStrategyList();
  if(!selectedId){
    $("#dispatchRaw").textContent = "没有可验证策略。";
    return null;
  }
  try{
    const res = await fetch(`/api/rl/models/${encodeURIComponent(selectedId)}/readiness`, {cache:"no-store"});
    if(!res.ok) throw new Error(await res.text());
    const readiness = await res.json();
    const modelResponse = await fetch(`/api/rl/models/${encodeURIComponent(selectedId)}`, {cache:"no-store"});
    const model = modelResponse.ok ? await modelResponse.json() : null;
    const result = {
      triggered_by: options.source || "model_readiness",
      strategy_id: selectedId,
      model,
      readiness,
      production_deployment_approved:false,
      production_boundary:"本步只读证据门禁，不晋级模型，不执行设备命令。",
    };
    $("#dispatchRaw").textContent = JSON.stringify(result, null, 2);
    $("#simHint").textContent = readiness.ready_for_champion_alias
      ? "软件晋级门禁已通过；仍需人工审批，且不代表现场生产验收。"
      : `上线门禁阻断：${(readiness.blockers||[]).join("；") || "证据不足"}`;
    return result;
  }catch(err){
    $("#dispatchRaw").textContent = "上线门禁读取失败："+String(err).slice(0,500);
    $("#simHint").textContent = "上线门禁读取失败，未执行任何设备指令。";
    return null;
  }
}

function renderHistory(items){
  if(!items || items.length===0){
    $("#historyList").innerHTML = `<div class="muted small">暂无历史</div>`;
    return;
  }
  $("#historyList").innerHTML = items.map(it=>{
    const risks = (it.guardrails?.risk_flags || []).map(x=>`<span class="risk">${x}</span>`).join("");
    return `
      <div class="history-item">
        <div class="history-head">
          <div><span class="badge">${it.strategy_id || "-"}</span> ${it.strategy_title || "-"}</div>
          <div class="${statusClass(it.status)}">${it.status || "-"}</div>
        </div>
        <div class="history-body tiny">
          <div>job_id: <span class="mono">${it.job_id || "-"}</span></div>
          <div>operator: ${it.operator || "-"} · dry_run: ${String(it.dry_run)}</div>
          <div>dispatch_ready: ${String(it.readiness?.dispatch_ready)} · guardrails_ok: ${String(it.guardrails?.ok)}</div>
          <div>delta_kWh: ${(it.estimate?.delta_kWh ?? 0).toFixed ? it.estimate.delta_kWh.toFixed(3) : it.estimate?.delta_kWh ?? 0} · peak_reduction_kW: ${(it.estimate?.peak_reduction_kW ?? 0).toFixed ? it.estimate.peak_reduction_kW.toFixed(2) : it.estimate?.peak_reduction_kW ?? 0}</div>
          <div>${risks || '<span class="muted">无额外风险标记</span>'}</div>
        </div>
      </div>`;
  }).join("");
}

async function loadHistory(){
  try{
    const res = await fetch("/api/actuators/capabilities", {cache:"no-store"});
    if(!res.ok) throw new Error(await res.text());
    const capability = await res.json();
    const mode = capability.mode || "fail_closed";
    const enabled = capability.enabled === true;
    const reason = capability.reason || "评测页不产生设备指令";
    $("#historyList").innerHTML = `
      <div class="history-item">
        <div class="history-head">
          <div><span class="badge">南向网关</span></div>
          <div class="${enabled ? "status-warn" : "status-ok"}">${enabled ? "已配置·仍需异人确认" : "失效安全关闭"}</div>
        </div>
        <div class="history-body tiny">
          <div>mode: <span class="mono">${mode}</span></div>
          <div>enabled: ${String(enabled)}</div>
          <div>${reason}</div>
          <div>评测与设备执行分离；现场流程为 <span class="mono">rl-stage → confirm</span>。</div>
        </div>
      </div>`;
  }catch(err){
    $("#historyList").innerHTML = `<div class="small" style="color:#ef4444;">能力声明读取失败：${String(err).slice(0,300)}</div>`;
  }
}

$("#btnLoad").addEventListener("click", loadList);
$("#btnSim").addEventListener("click", simulate);
$("#btnDispatch").addEventListener("click", dispatchSelected);
$("#btnHistory").addEventListener("click", loadHistory);
$("#btnBackToPlatform")?.addEventListener("click", ()=> goBackTo(returnTo));
$("#btnBackToHome")?.addEventListener("click", ()=> goBackTo('/'));
$("#btnStartTrain")?.addEventListener("click", startTraining);
$("#btnPauseTrain")?.addEventListener("click", pauseTraining);
$("#btnResetTrain")?.addEventListener("click", resetTraining);
$("#btnPollTrainStatus")?.addEventListener("click", ()=> pollTrainingStatus({log:true, sourceLabel:"manual poll"}));
$("#btnEvaluateTrain")?.addEventListener("click", evaluateTraining);
$("#btnVerifyDryRun")?.addEventListener("click", ()=> verifyPolicyForOnline({source:"manual_verify_online"}));
$("#btnCancelAssistantRun")?.addEventListener("click", ()=>{
  hideAssistantRunConfirm();
  appendTrainLog(`[${new Date().toLocaleTimeString("zh-CN",{hour12:false})}] assistant command cancelled · human gate`);
  if($("#trainDetail")) $("#trainDetail").textContent = "已取消小懿训练指令；未调用 /api/rl/train/start。";
});
$("#btnConfirmAssistantRun")?.addEventListener("click", async ()=>{
  hideAssistantRunConfirm();
  const s = confirmationSummary();
  if($("#trainDetail")) $("#trainDetail").textContent = `人工确认通过：${s.objective}，正在启动训练。`;
  await startTraining();
  appendTrainLog(`[${new Date().toLocaleTimeString("zh-CN",{hour12:false})}] assistant confirmed · ${s.objective} -> #btnStartTrain -> /api/rl/train/start`);
});
$("#btnPingConnector")?.addEventListener("click", refreshConnector);
$("#btnRefreshMobileRequests")?.addEventListener("click", loadMobileTrainingRequests);
$("#mobileTrainingRequestList")?.addEventListener("click", event=>{
  const button = event.target.closest("button[data-mobile-action]");
  if(!button) return;
  void reviewMobileTrainingRequest(button.dataset.requestId, button.dataset.mobileAction);
});
function handleTrainConfigChange(){
  renderBaselineCards();
  updateConnectorPreview();
}
$$(".train-card input,.train-card select").forEach(el=>{
  el.addEventListener("change", handleTrainConfigChange);
  el.addEventListener("input", updateConnectorPreview);
});
window.addEventListener("load", async ()=>{
  await reportDesktopPanelHeartbeat();
  desktopHeartbeatTimer = window.setInterval(reportDesktopPanelHeartbeat, 3000);
  applyReturnBanner();
  resetTraining();
  try{ await loadTrainingDatasets(); }catch(err){ $("#trainRunSummary").textContent = `数据集加载失败：${String(err).slice(0,180)}`; }
  await refreshConnector();
  await loadMobileTrainingRequests();
  mobileRequestPollTimer = window.setInterval(loadMobileTrainingRequests, 3000);
  await loadList();
  await loadHistory();
  await handleAssistantAction();
});
</script>
</body>
</html>
"""


# -------------------------------------------------
# 主页（返回 index.html）
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def home() -> HTMLResponse:
    if not _UI_INDEX.exists():
        html = "<h1>UI 文件缺失</h1><p>请把前端放到 app/ui/index.html。</p>"
        return HTMLResponse(html, status_code=200)
    html = _UI_INDEX.read_text(encoding="utf-8")
    # 自动注入监测适配器脚本（不改动源文件；若文件不存在则忽略）
    try:
        _adapter_path = Path(__file__).resolve().parent / "ui" / "adapters" / "monitoring.js"
        if _adapter_path.exists() and ("ui/adapters/monitoring.js" not in html):
            html = html.replace("</body>", '  <script src="/ui/adapters/monitoring.js"></script>\n</body>')
    except Exception:
        pass
    html = _inject_bilingual_ui(html)
    html = _inject_xiaoyi_sprite(html)
    return HTMLResponse(html, status_code=200)

@app.get("/ui/adapters/monitoring.js", response_class=HTMLResponse, tags=["ui"])
async def monitoring_adapter_js() -> HTMLResponse:
    """
    前端适配器脚本（生产部署可由 Nginx/静态服务器托管；此处提供直连以便开发联调）
    """
    try:
        path = Path(__file__).resolve().parent / "ui" / "adapters" / "monitoring.js"
        if path.exists():
            return HTMLResponse(path.read_text(encoding="utf-8"),
                                media_type="application/javascript",
                                status_code=200)
        return HTMLResponse("// monitoring adapter not found: app/ui/adapters/monitoring.js",
                            media_type="application/javascript", status_code=404)
    except Exception as e:
        return HTMLResponse(f"// error: {e}", media_type="application/javascript", status_code=500)


@app.get("/ui/adapters/xiaoyi_sprite.js", response_class=HTMLResponse, tags=["ui"])
async def xiaoyi_sprite_adapter_js() -> HTMLResponse:
    """
    小懿AI 全局悬浮入口脚本：动画图标、拖拽、命令网关联动。
    """
    try:
        if _XIAOYI_SPRITE_JS.exists():
            return HTMLResponse(
                _XIAOYI_SPRITE_JS.read_text(encoding="utf-8"),
                media_type="application/javascript",
                status_code=200,
            )
        return HTMLResponse(
            "// xiaoyi sprite adapter not found: app/ui/adapters/xiaoyi_sprite.js",
            media_type="application/javascript",
            status_code=404,
        )
    except Exception as e:
        return HTMLResponse(f"// error: {e}", media_type="application/javascript", status_code=500)


@app.get("/ui/adapters/bilingual_ui.js", response_class=HTMLResponse, tags=["ui"])
async def bilingual_ui_adapter_js() -> HTMLResponse:
    """全局中英双语辅助层：保留中文主文案，并以英文小字补充。"""
    try:
        if _BILINGUAL_UI_JS.exists():
            return HTMLResponse(
                _BILINGUAL_UI_JS.read_text(encoding="utf-8"),
                media_type="application/javascript",
                status_code=200,
            )
        return HTMLResponse(
            "// bilingual UI adapter not found: app/ui/adapters/bilingual_ui.js",
            media_type="application/javascript",
            status_code=404,
        )
    except Exception as e:
        return HTMLResponse(f"// error: {e}", media_type="application/javascript", status_code=500)


def _read_exec_pending_snapshot(limit: int = 20) -> Dict[str, Any]:
    """聚合执行闭环的待审批与最近工单摘要，给首页规则口径使用。"""
    try:
        from app.services.exec_closedloop.dispatch_api import AUDIT_DIR as _AUDIT_DIR  # type: ignore
    except Exception:
        _AUDIT_DIR = Path(__file__).resolve().parent.parent / "data" / "objects" / "audit"

    items: List[Dict[str, Any]] = []
    audit_dir = Path(_AUDIT_DIR)
    if not audit_dir.exists():
        return {"pending_count": 0, "last_job": "—", "last_status": "unknown", "last_asset": "—", "last_action": "—"}

    for p in sorted(audit_dir.glob("guard-*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ts = data.get("timestamps") or {}
            if not isinstance(ts, dict):
                ts = {}
            if "pending_at" in ts and "executed_at" not in ts:
                status = "pending"
            elif "rolled_back_at" in ts:
                status = "rolled_back"
            elif "executed_at" in ts:
                status = "executed"
            else:
                status = "submitted"
            cmd = data.get("command") or {}
            items.append({
                "command_id": data.get("command_id") or p.stem,
                "asset_id": cmd.get("asset_id") or "—",
                "action": cmd.get("action") or "—",
                "status": status,
                "updated_at": ts.get("rolled_back_at") or ts.get("executed_at") or ts.get("pending_at") or ts.get("requested_at") or datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            })
        except Exception:
            continue
        if len(items) >= limit:
            break

    pending_count = sum(1 for item in items if item.get("status") == "pending")
    last = items[0] if items else {}
    return {
        "pending_count": pending_count,
        "last_job": last.get("command_id", "—"),
        "last_status": last.get("status", "unknown"),
        "last_asset": last.get("asset_id", "—"),
        "last_action": last.get("action", "—"),
        "last_updated_at": last.get("updated_at"),
        "items": items,
    }


@app.get("/api/platform/home_brief", tags=["platform"])
async def platform_home_brief() -> JSONResponse:
    """给首页首屏提供统一简报口径：状态、风险、待审批、最近闭环、OpsX 摘要。"""
    exec_snapshot = _read_exec_pending_snapshot(limit=12)
    pending_count = int(exec_snapshot.get("pending_count") or 0)
    last_status = str(exec_snapshot.get("last_status") or "unknown")
    last_job = str(exec_snapshot.get("last_job") or "—")
    last_asset = str(exec_snapshot.get("last_asset") or "—")
    last_action = str(exec_snapshot.get("last_action") or "—")

    apps_total = 0
    try:
        overview = app_center_service.get_overview(di) if hasattr(app_center_service, "get_overview") else None
        if isinstance(overview, dict):
            apps_total = len(overview.get("apps") or [])
    except Exception:
        apps_total = 0

    status = "稳定"
    risk_label = "中低"
    focus = "先巡检主链路"
    status_sub = "当前无明显待审批拥塞，适合先看 Twin → Strategy → Execution → Audit 主链路一致性。"
    risk_sub = "规则判定：待审批=0，最近闭环未出现异常回滚。"
    focus_sub = "首页先给结论，细节继续在孪生、策略、执行和审计模块承接。"

    if pending_count >= 5 or last_status == "rolled_back":
        status = "临界"
        risk_label = "高"
        focus = f"先清待审批（{pending_count}）"
        status_sub = "首页判定当前闭环动作积压较高，需先处理高风险动作与回执证据。"
        risk_sub = "规则判定：待审批≥5 或最近工单出现回滚。"
        focus_sub = "建议先进入策略编排与执行闭环，确认是否需要人工接管。"
    elif pending_count >= 1 or last_status == "pending":
        status = "需盯"
        risk_label = "中高"
        focus = f"先看执行闭环（待审批 {pending_count}）"
        status_sub = "首页判定当前存在待确认动作，需先核对策略下发、审批与执行证据。"
        risk_sub = "规则判定：存在待审批工单。"
        focus_sub = "建议先到 Execution / Audit 补齐确认动作，再回到 Twin 和策略模块做扩展分析。"

    opsx_meta = f"OpsX 已挂载；最近闭环={last_status}；App Center 已发现 {apps_total} 个应用入口。"
    payload = {
        "status": status,
        "status_sub": status_sub,
        "risk_label": risk_label,
        "risk_sub": risk_sub,
        "focus": focus,
        "focus_sub": focus_sub,
        "pending_count": pending_count,
        "pending_text": str(pending_count),
        "last_job": last_job,
        "last_status": last_status,
        "last_asset": last_asset,
        "last_action": last_action,
        "opsx_meta": opsx_meta,
        "rule_tag": "规则来源：后端首页简报 + 执行闭环审计摘要",
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return JSONResponse(payload)


@app.get("/ui/adapters/home_brief.js", response_class=HTMLResponse, tags=["ui"])
async def home_brief_adapter_js() -> HTMLResponse:
    script = """(function(){
  function $(id){ return document.getElementById(id); }
  function setText(id, value){ var el = $(id); if (el && value !== undefined && value !== null && value !== '') el.textContent = String(value); }
  function apply(data){
    if (!data || typeof data !== 'object') return;
    setText('home-status', data.status);
    setText('home-status-sub', (data.status_sub || '') + (data.rule_tag ? ' · ' + data.rule_tag : ''));
    setText('home-risk', data.risk_label);
    setText('home-risk-sub', data.risk_sub);
    setText('home-focus', data.focus);
    setText('home-focus-sub', data.focus_sub);
    setText('home-loop-dispatch', '待审批=' + (data.pending_text || '0') + '；最近工单=' + (data.last_job || '—') + '。首页已改为优先读取后端闭环摘要。');
    setText('home-loop-audit', data.opsx_meta || 'OpsX 摘要暂未返回。');
    var chain = [data.last_asset && data.last_asset !== '—' ? ('最近资产：' + data.last_asset) : '', data.last_action && data.last_action !== '—' ? ('动作：' + data.last_action) : '', data.last_status ? ('闭环状态：' + data.last_status) : ''].filter(Boolean).join(' · ');
    setText('home-chain-note', chain || '首页已接入后端首页简报接口。');
    setText('home-desc', '首页主结论现优先使用 /api/platform/home_brief 的后端口径，前端规则继续作为兜底。');
    setText('exec-status', data.status);
    setText('exec-status-sub', data.status_sub);
    setText('ap-pending', data.pending_text || '0');
    setText('ap-last', data.last_job || '—');
    setText('opsx-meta', data.opsx_meta || '—');
    if (typeof window.__syncHomeHero === 'function') {
      try { window.__syncHomeHero(); } catch (e) {}
    }
  }
  async function load(){
    try {
      var res = await fetch('/api/platform/home_brief', {cache: 'no-store'});
      if (!res.ok) return;
      var data = await res.json();
      apply(data);
    } catch (e) {}
  }
  window.addEventListener('load', function(){ setTimeout(load, 120); setTimeout(load, 1200); });
})();"""
    return HTMLResponse(script, media_type="application/javascript", status_code=200)


# 独立页：RL 策略面板
@app.get("/rl-panel", response_class=HTMLResponse, tags=["ui"])
async def rl_panel_page(request: Request) -> HTMLResponse:
    return HTMLResponse(_inject_xiaoyi_sprite(_inject_bilingual_ui(_RL_PANEL_HTML)), status_code=200)


@app.get("/ops-copilot", response_class=HTMLResponse, tags=["ui"])
async def ops_copilot_page(request: Request) -> HTMLResponse:
    if not _OPS_COPILOT_UI.exists():
        return HTMLResponse("<h1>Ops Copilot UI 文件缺失</h1>", status_code=404)
    return HTMLResponse(_inject_xiaoyi_sprite(_inject_bilingual_ui(_OPS_COPILOT_UI.read_text(encoding="utf-8"))), status_code=200)


@app.get("/integration-hub", response_class=HTMLResponse, tags=["ui"])
async def integration_hub_page(request: Request) -> HTMLResponse:
    if not _INTEGRATION_HUB_UI.exists():
        return HTMLResponse("<h1>项目联动中枢 UI 文件缺失</h1>", status_code=404)
    return HTMLResponse(_inject_xiaoyi_sprite(_inject_bilingual_ui(_INTEGRATION_HUB_UI.read_text(encoding="utf-8"))), status_code=200)


# -------------------------------------------------
# Telemetry SSE（实时流）
# -------------------------------------------------
@app.get("/api/assets", tags=["telemetry"])
async def list_assets() -> JSONResponse:
    try:
        assets = di.telemetry.list_assets()
        return JSONResponse(assets)
    except Exception:
        return JSONResponse([{"id": "agv-01", "label": "AGV-01"}], status_code=200)


async def _sse_generator(asset_id: str) -> AsyncGenerator[bytes, None]:
    """每秒发 1 点的 SSE 实时流，前端画滚动曲线。"""
    last_ts: Optional[str] = None
    while True:
        try:
            arr = di.telemetry.get_recent_power(asset_id) or []
            if arr:
                p = arr[-1]
                ts = p.get("ts") if isinstance(p, dict) else None
                if ts and ts != last_ts:
                    last_ts = ts
                    yield f"data: {json.dumps(p, ensure_ascii=False)}\n\n".encode("utf-8")
        except Exception:
            hb = json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "kW": 0.0})
            yield f"data: {hb}\n\n".encode("utf-8")
        await asyncio.sleep(1.0)


@app.get("/stream/telemetry/{asset_id}", tags=["telemetry"])
async def stream_telemetry(asset_id: str) -> StreamingResponse:
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(_sse_generator(asset_id), media_type="text/event-stream", headers=headers)


# -------------------------------------------------
# Telemetry 清洗 + 质量评分（统一口径）
# -------------------------------------------------
def _parse_epoch_or_iso(s: str) -> float:
    """把 epoch 秒 或 ISO8601（可带Z/时区/不带时区）统一转成 UTC epoch 秒。"""
    try:
        return float(s)
    except Exception:
        _s = s.strip()
        if _s.endswith("Z"):
            dtobj = datetime.fromisoformat(_s.replace("Z", "+00:00"))
        else:
            dtobj = datetime.fromisoformat(_s)
            if dtobj.tzinfo is None:
                dtobj = dtobj.replace(tzinfo=timezone.utc)
        return dtobj.timestamp()

def _norm_series(arr: list) -> list[tuple[float, float]]:
    """
    把各种格式统一成 [(ts_epoch,float_value), ...] 且按时间排序。
    支持：
      - [{'ts': ..., 'v': ...}, ...]
      - [{'time':..., 'value':...}, ...]
      - [(ts, v), ...] / [ts, v]
    """
    out = []
    for p in (arr or []):
        if isinstance(p, dict):
            ts = p.get("ts") or p.get("time") or p.get("t")
            v = p.get("v") or p.get("value") or p.get("val") or p.get("power_kw")
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            ts, v = p[0], p[1]
        else:
            continue
        try:
            t = _parse_epoch_or_iso(ts) if isinstance(ts, str) else float(ts)
            v = float(v)
            out.append((t, v))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out

def _to_iso(ts_epoch: float) -> str:
    """把 epoch 秒转成 ISO8601（UTC）。"""
    return datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()

@app.get("/api/telemetry/clean", tags=["telemetry"])
async def telemetry_clean(
    asset_id: str = Query(..., description="设备ID，例如 qch-01"),
    point: str = Query("active_power_kw", description="测点名，例如 active_power_kw"),
    asset_type: str = Query("quay_crane", description="设备类型，用于合理边界与清洗策略"),
    start: str = Query(..., description="开始时间（epoch 或 ISO8601，UTC口径）"),
    end: str = Query(..., description="结束时间（epoch 或 ISO8601，UTC口径）"),
    step_sec: int = Query(60, ge=1, le=3600, description="等间隔步长（秒）"),
    resample_method: str = Query("ffill", pattern="^(none|ffill|linear)$", description="重采样策略"),
    impute_method: str = Query("ffill", pattern="^(ffill|linear)$", description="插补策略"),
) -> JSONResponse:
    """
    返回：
      - cleaned：清洗+插补后的等间隔曲线（ISO时间+数值）
      - quality：{completeness, timeliness, validity}
      - source：数据来源（DI系列函数名）
    真实落地时只需让 di.telemetry.get_series(...) 返回 (ts,value) 列表或带 ts/v 的字典列表。
    """
    try:
        start_ts = _parse_epoch_or_iso(start)
        end_ts = _parse_epoch_or_iso(end)
        if end_ts <= start_ts:
            raise HTTPException(status_code=400, detail="end 必须大于 start")

        # 1) 优先用 DI 的标准接口（如果你们已实现）
        raw = None
        source = None
        if hasattr(di, "telemetry") and hasattr(di.telemetry, "get_series"):
            try:
                raw = di.telemetry.get_series(asset_id=asset_id, point=point, start_ts=start_ts, end_ts=end_ts) or []
                source = "di.telemetry.get_series"
            except Exception:
                raw = None

        # 2) 回退：用最近功率接口做演示（不精确，但便于先跑起来）
        if raw is None and hasattr(di, "telemetry") and hasattr(di.telemetry, "get_recent_power") and point == "active_power_kw":
            raw = di.telemetry.get_recent_power(asset_id) or []
            source = "di.telemetry.get_recent_power(fallback)"

        # 3) 仍然取不到就报错（提示你实现 DI 接口或打通 TSDB）
        if raw is None:
            raise HTTPException(status_code=501, detail="缺少数据源：请实现 di.telemetry.get_series 或保证 get_recent_power 可用")

        series = _norm_series(raw)
        cleaned, quality, _mask = clean_and_impute(
            series,
            start=start_ts,
            end=end_ts,
            step_sec=step_sec,
            asset_type=asset_type,
            point=point,
            resample_method=resample_method,  # 'none'/'ffill'/'linear'
            impute_method=impute_method,      # 'ffill'/'linear'
        )

        body = {
            "asset_id": asset_id,
            "point": point,
            "asset_type": asset_type,
            "step_sec": step_sec,
            "cleaned": [{"ts": _to_iso(ts), "v": v} for ts, v in cleaned],
            "quality": quality,
            "source": source or "unknown",
        }
        return JSONResponse(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"telemetry.clean 失败: {e}")

@app.get("/api/telemetry/quality", tags=["telemetry"])
async def telemetry_quality(
    asset_id: str = Query(..., description="设备ID"),
    point: str = Query("active_power_kw", description="测点名"),
    asset_type: str = Query("quay_crane", description="设备类型"),
    start: str = Query(..., description="开始时间（epoch 或 ISO8601）"),
    end: str = Query(..., description="结束时间（epoch 或 ISO8601）"),
    step_sec: int = Query(60, ge=1, le=3600, description="步长（秒）"),
) -> JSONResponse:
    """仅返回质量分；内部与 /api/telemetry/clean 相同清洗逻辑。"""
    try:
        res = await telemetry_clean(asset_id, point, asset_type, start, end, step_sec, "ffill", "ffill")
        data = res.body
        payload = json.loads(data.decode("utf-8")) if isinstance(data, (bytes, bytearray)) else data
        return JSONResponse({"asset_id": asset_id, "point": point, "quality": payload.get("quality", {}), "source": payload.get("source")})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"telemetry.quality 失败: {e}")

@app.get("/api/telemetry/recent/{asset_id}", tags=["telemetry"])
async def telemetry_recent(asset_id: str):
    """
    返回指定设备最近一段时间的功率点（约 60 秒 * 1Hz）。
    与文档注释保持一致，供前端初始化折线使用。
    """
    try:
        arr = di.telemetry.get_recent_power(asset_id) or []
        # 兜底：确保返回列表
        if not isinstance(arr, list):
            arr = [arr]
        return JSONResponse(arr)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"telemetry.recent 失败: {e}")

# [新增开始 @ 约 L514 之后 —— 监测与运维（Monitoring）接口组]
# -------------------------------------------------
# 监测与运维（Monitoring）：异常检测 + 漂移检测（PSI）
# -------------------------------------------------

def _percentile(vals: List[float], q: float) -> float:
    """简易百分位函数（无 numpy），q ∈ [0,1]。"""
    if not vals:
        return 0.0
    s = sorted(vals)
    pos = q * (len(s) - 1)
    i = int(pos)
    if i >= len(s) - 1:
        return float(s[-1])
    frac = pos - i
    return float(s[i] * (1 - frac) + s[i + 1] * frac)

def _iqr_anomaly(series: List[tuple], k: float = 1.5) -> List[Dict[str, Any]]:
    """
    IQR 异常：取 v 的 Q1/Q3，阈值 = [Q1-k*IQR, Q3+k*IQR]，越界即异常。
    series: [(ts_epoch, value), ...]
    返回: [{"ts": ISO, "v": float, "score": float, "reason": "iqr"}, ...]
    """
    if not series:
        return []
    vals = [float(v) for _, v in series]
    q1 = _percentile(vals, 0.25)
    q3 = _percentile(vals, 0.75)
    iqr = max(1e-9, q3 - q1)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    out = []
    for ts, v in series:
        if v < lo or v > hi:
            # 异常程度：超界距离 / IQR
            dist = (lo - v) if v < lo else (v - hi)
            score = abs(dist) / iqr
            out.append({"ts": _to_iso(ts), "v": float(v), "score": round(score, 3), "reason": "iqr"})
    return out

def _zscore_anomaly(series: List[tuple], z: float = 3.0) -> List[Dict[str, Any]]:
    """
    Z-Score 异常：全局均值/方差，|v-μ|/σ > z 即异常。
    注意：生产建议用“滚动窗口 z-score”，此处先提供全局版，方便联调。
    """
    if not series:
        return []
    vals = [float(v) for _, v in series]
    mu = sum(vals) / max(1, len(vals))
    var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
    sigma = max(1e-9, math.sqrt(var))
    out = []
    for ts, v in series:
        s = abs((v - mu) / sigma)
        if s > z:
            out.append({"ts": _to_iso(ts), "v": float(v), "score": round(float(s), 3), "reason": "zscore"})
    return out

def _ewma_anomaly(series: List[tuple], alpha: float = 0.2, k: float = 3.0) -> List[Dict[str, Any]]:
    """
    EWMA 异常：指数滑动均值/方差，|v-μ_t|/σ_t > k 即异常。对缓慢漂移较敏感。
    """
    if not series:
        return []
    mu = None
    var = None
    out = []
    for ts, v in series:
        v = float(v)
        if mu is None:
            mu = v
            var = 0.0
            continue
        mu = alpha * v + (1 - alpha) * mu
        var = alpha * (v - mu) ** 2 + (1 - alpha) * var
        sigma = max(1e-9, math.sqrt(var))
        s = abs((v - mu) / sigma)
        if s > k:
            out.append({"ts": _to_iso(ts), "v": v, "score": round(float(s), 3), "reason": "ewma"})
    return out

def _get_clean_series(asset_id: str, point: str, start_ts: float, end_ts: float, step_sec: int, asset_type: str):
    """
    统一入口：从 DI 拉原始 → 清洗插补 → 等间隔序列
    落地对接：只要将 di.telemetry.get_series(...) 接到 TSDB/OPC/MQTT 即可无缝替换。
    """
    raw = None
    source = None
    if hasattr(di, "telemetry") and hasattr(di.telemetry, "get_series"):
        try:
            raw = di.telemetry.get_series(asset_id=asset_id, point=point, start_ts=start_ts, end_ts=end_ts) or []
            source = "di.telemetry.get_series"
        except Exception:
            raw = None
    if raw is None and hasattr(di, "telemetry") and hasattr(di.telemetry, "get_recent_power") and point == "active_power_kw":
        raw = di.telemetry.get_recent_power(asset_id) or []
        source = "di.telemetry.get_recent_power(fallback)"
    if raw is None:
        raise HTTPException(status_code=501, detail="缺少数据源：请实现 di.telemetry.get_series 或保证 get_recent_power 可用")
    series = _norm_series(raw)
    cleaned, quality, _mask = clean_and_impute(
        series,
        start=start_ts,
        end=end_ts,
        step_sec=step_sec,
        asset_type=asset_type,
        point=point,
        resample_method="ffill",
        impute_method="ffill",
    )
    return cleaned, quality, source or "unknown"

@app.get("/api/monitoring/anomaly/scan", tags=["monitoring"])
async def monitoring_anomaly_scan(
    assets: Optional[str] = Query(None, description="逗号分隔资产ID（为空则自动截取前 10 个资产）"),
    point: str = Query("active_power_kw", description="测点名（默认功率）"),
    asset_type: str = Query("quay_crane", description="资产类型（影响清洗与合理边界）"),
    start: str = Query(..., description="开始时间（epoch 或 ISO8601, UTC 口径）"),
    end: str = Query(..., description="结束时间（epoch 或 ISO8601, UTC 口径）"),
    step_sec: int = Query(60, ge=1, le=3600, description="等间隔步长（秒）"),
    method: str = Query("iqr", pattern="^(iqr|zscore|ewma)$", description="检测方法"),
    sensitivity: float = Query(1.5, ge=0.1, le=10.0, description="灵敏度：iqr=k；zscore=Z；ewma=k"),
) -> JSONResponse:
    """
    大白话：拉一段时间的“干净等间隔曲线”，跑指定方法做异常检测，按资产返回异常点。
    - 真实港口接入：实现 di.telemetry.get_series(...) 就能直接用。
    - 常见用法：
        /api/monitoring/anomaly/scan?assets=qc-01,agv-03&start=2025-10-10T00:00:00Z&end=2025-10-10T06:00:00Z&method=iqr
    返回：
      {
        "params": {...},
        "items": [
          {"asset":"qc-01","quality":{...},"anomalies":[{"ts":"...","v":..,"score":..,"reason":"iqr"}, ...]},
          ...
        ]
      }
    """
    try:
        start_ts = _parse_epoch_or_iso(start)
        end_ts = _parse_epoch_or_iso(end)
        if end_ts <= start_ts:
            raise HTTPException(status_code=400, detail="end 必须大于 start")

        # 资产列表：未显式传 assets 时，自动取前 10 个做演示
        if assets:
            asset_list: List[str] = [a.strip() for a in assets.split(",") if a.strip()]
        else:
            try:
                asset_list = [a.get("asset_id") or a.get("id") for a in (di.telemetry.list_assets() or [])]
                asset_list = [x for x in asset_list if x][:10]
            except Exception:
                asset_list = ["qc-01"]

        # 优先走 DI 服务（生产口径）；若 DI 不可用，退回本地实现（保证兼容）
        if not hasattr(di, "monitoring") or not hasattr(di.monitoring, "scan_anomalies"):
            items = []
            for aid in asset_list:
                cleaned, quality, source = _get_clean_series(
                    aid, point, start_ts, end_ts, step_sec, asset_type
                )
                if method == "iqr":
                    anomalies = _iqr_anomaly(cleaned, k=sensitivity)
                elif method == "zscore":
                    anomalies = _zscore_anomaly(cleaned, z=sensitivity)
                else:
                    anomalies = _ewma_anomaly(cleaned, k=sensitivity)
                items.append({"asset": aid, "quality": quality, "source": source, "anomalies": anomalies})
            return JSONResponse({
                "params": {
                    "assets": asset_list, "point": point, "asset_type": asset_type,
                    "start": _to_iso(start_ts), "end": _to_iso(end_ts),
                    "step_sec": step_sec, "method": method, "sensitivity": sensitivity
                },
                "items": items
            })

        # ✅ 调用 DI 的 MonitoringService（已在 app/di.py 中挂载）
        res = di.monitoring.scan_anomalies(
            asset_ids=asset_list,
            point=point,
            asset_type=asset_type,
            start_ts=start_ts,
            end_ts=end_ts,
            step_sec=step_sec,
            method=method,
            sensitivity=sensitivity,
            residual=False  # 如果你要用“残差异常”，我下一步给你把参数开放到接口
        )
        return JSONResponse(res)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"monitoring.anomaly.scan 失败: {e}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"monitoring.anomaly.scan 失败: {e}")

def _psi(buckets_ref: List[float], ref_vals: List[float], cur_vals: List[float]) -> Dict[str, Any]:
    """
    计算人口稳定性指数（PSI），用于监测分布漂移。
    - buckets_ref: 分箱的边界（升序，长度>=2）
    - ref_vals / cur_vals: 两个窗口的数值样本
    """
    eps = 1e-6
    def count_bins(vals: List[float]) -> List[int]:
        counts = [0]*(len(buckets_ref)-1)
        for v in vals:
            # 找所在区间
            for i in range(len(buckets_ref)-1):
                if (v >= buckets_ref[i]) and (v <= buckets_ref[i+1] if i == len(buckets_ref)-2 else v < buckets_ref[i+1]):
                    counts[i]+=1; break
        return counts

    ref_cnt = count_bins(ref_vals)
    cur_cnt = count_bins(cur_vals)
    n_ref = max(1, sum(ref_cnt))
    n_cur = max(1, sum(cur_cnt))

    details = []
    psi_total = 0.0
    for i in range(len(ref_cnt)):
        p = max(eps, ref_cnt[i] / n_ref)
        q = max(eps, cur_cnt[i] / n_cur)
        iv = (p - q) * math.log(p / q)
        psi_total += iv
        details.append({"bin": i, "p_ref": p, "p_cur": q, "psi": iv})
    return {"psi": psi_total, "bins": [{"lo": buckets_ref[i], "hi": buckets_ref[i+1]} for i in range(len(buckets_ref)-1)], "details": details}

@app.get("/api/monitoring/drift/psi", tags=["monitoring"])
async def monitoring_drift_psi(
    asset_id: str = Query(..., description="资产ID"),
    point: str = Query("active_power_kw", description="测点名"),
    asset_type: str = Query("quay_crane", description="资产类型"),
    baseline_start: str = Query(..., description="基线开始（ISO 或 epoch）"),
    baseline_end: str = Query(..., description="基线结束（ISO 或 epoch）"),
    recent_start: str = Query(..., description="对比开始（ISO 或 epoch）"),
    recent_end: str = Query(..., description="对比结束（ISO 或 epoch）"),
    bins: int = Query(10, ge=3, le=50, description="分箱个数（等宽）"),
    step_sec: int = Query(60, ge=1, le=3600, description="等间隔步长（秒）"),
) -> JSONResponse:
    """
    大白话：对比“基线窗口”与“最近窗口”的分布，返回 PSI 值，>0.2 通常提示显著漂移。
    真实接入：同样只依赖 di.telemetry.get_series(...)。
    """
    try:
        b0 = _parse_epoch_or_iso(baseline_start)
        b1 = _parse_epoch_or_iso(baseline_end)
        r0 = _parse_epoch_or_iso(recent_start)
        r1 = _parse_epoch_or_iso(recent_end)
        if not (b1 > b0 and r1 > r0):
            raise HTTPException(status_code=400, detail="时间窗口非法")

        base_series, _, _ = _get_clean_series(asset_id, point, b0, b1, step_sec, asset_type)
        cur_series,  _, _ = _get_clean_series(asset_id, point, r0, r1, step_sec, asset_type)
        base_vals = [v for _, v in base_series]
        cur_vals  = [v for _, v in cur_series]
        if len(base_vals) < 10 or len(cur_vals) < 10:
            raise HTTPException(status_code=400, detail="样本过少，建议窗口更长或步长更短")

        lo, hi = min(base_vals), max(base_vals)
        if hi <= lo:
            hi = lo + 1.0
        step = (hi - lo) / float(bins)
        edges = [lo + i*step for i in range(bins)] + [hi]  # 长度 bins+1

        res = _psi(edges, base_vals, cur_vals)
        return JSONResponse({
            "asset": asset_id, "point": point, "asset_type": asset_type,
            "baseline": {"start": _to_iso(b0), "end": _to_iso(b1), "n": len(base_vals)},
            "recent":   {"start": _to_iso(r0), "end": _to_iso(r1), "n": len(cur_vals)},
            "psi": res["psi"], "bins": res["bins"], "details": res["details"]
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"monitoring.drift.psi 失败: {e}")

# [新增结束 —— Monitoring]

# -------------------------------------------------
# 预测（Forecast）—— 支持 use_drivers=1 叠加外部驱动
# -------------------------------------------------
@app.get("/api/forecast/{asset_id}", tags=["forecast"])
async def forecast_asset(
    asset_id: str,
    horizon_min: int = Query(360, ge=1, le=48 * 60),
    step_min: int = Query(1, ge=1, le=60),
    use_drivers: int = Query(0, ge=0, le=1, description="是否叠加外部驱动（TOS/AIS/天气/电价等）"),
    port: str = Query("CN_DEMO", description="港口代码（用于外部驱动）"),
) -> JSONResponse:
    # 当 use_drivers=1 时，从 di.schedule 拉驱动，并尽量传入 ForecastService
    drivers = None
    driver_provenance = "not_requested"
    if use_drivers == 1:
        now = datetime.now(timezone.utc)
        end = now + timedelta(minutes=horizon_min)
        try:
            sch = getattr(di, "schedule", None)
            if sch is None:
                raise HTTPException(status_code=503, detail="schedule driver adapter unavailable")
            status = sch.source_status() if hasattr(sch, "source_status") else {"mode": "unavailable"}
            if status.get("mode") == "unavailable":
                raise HTTPException(status_code=503, detail="schedule driver adapter is not configured")
            drivers = sch.load_drivers(
                start=now.isoformat(),
                end=end.isoformat(),
                port_code=port,
                assets=[asset_id],
            )
            driver_provenance = status
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"forecast driver load failed: {exc}") from exc

    # 向预测服务传入 drivers（若实现支持）；不支持时回退到原签名
    try:
        fmap = di.fcst.forecast_load(
            [asset_id], horizon_min=horizon_min, step_min=step_min, drivers=drivers
        ) or {}
    except TypeError:
        fmap = di.fcst.forecast_load(
            [asset_id], horizon_min=horizon_min, step_min=step_min
        ) or {}

    return JSONResponse(
        {
            "asset": asset_id,
            "points": fmap.get(asset_id, []),
            "_provenance": {
                "forecast_engine": type(di.fcst).__name__,
                "external_drivers": driver_provenance,
                "use_drivers": bool(use_drivers),
            },
        }
    )


# -------------------------------------------------
# 外部数据只读代理（真实/模拟数据接入）
# -------------------------------------------------
@app.get("/external/weather", tags=["external"])
async def external_weather(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
    lat: float = Query(31.2304, description="纬度"),
    lon: float = Query(121.4737, description="经度"),
) -> JSONResponse:
    try:
        sch = getattr(di, "schedule", None)
        if sch is None:
            raise HTTPException(status_code=503, detail="weather adapter unavailable")
        status = sch.source_status() if hasattr(sch, "source_status") else {"mode": "unavailable"}
        if status.get("mode") == "unavailable":
            raise HTTPException(status_code=503, detail="weather adapter is not configured")
        return JSONResponse(sch.weather(start=start, end=end, lat=lat, lon=lon))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"weather adapter failed: {exc}") from exc

@app.get("/external/vessels_schedule", tags=["external"])
async def external_vessels(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
    port: str = Query("CN_DEMO", description="港口代码"),
) -> JSONResponse:
    """
    船舶计划 / 靠离泊窗口查询。
    优先使用 DI 中的 schedule 适配器；若无，则回退到模块内 CSV（app/services/app_center/data/）。
    同时对返回结构做轻量“字段归一化”，保证前端可读到 vessel_id / berth_id 等。
    """
    try:
        sch = getattr(di, "schedule", None)
        if sch is not None and hasattr(sch, "vessels"):
            status = sch.source_status() if hasattr(sch, "source_status") else {"mode": "unavailable"}
            if status.get("mode") == "unavailable":
                raise HTTPException(status_code=503, detail="vessel schedule adapter is not configured")
            raw = sch.vessels(start=start, end=end, port_code=port)
            rows = []
            # 工具函数：从若干别名里挑第一项非空
            def pick(d, keys, default=""):
                for k in keys:
                    v = d.get(k)
                    if v is not None and str(v).strip() != "":
                        return v
                return default

            def to_float(x, dv=0.0):
                try:
                    return float(x)
                except Exception:
                    return dv

            def to_int(x, dv=0):
                try:
                    return int(float(x))
                except Exception:
                    return dv

            for r in (raw or []):
                # 归一化关键字段（适配市面常见命名）
                name = pick(r, ["vessel_id","name","vessel","ship_name","vesselName","VesselName","vsl_name","cn_name","en_name"], "—")
                berth = pick(r, ["berth_id","berth","berthCode","berth_name","berthNo","quay","quay_id"], "")
                eta = pick(r, ["eta","ETA","arrival","arrive_time","ata_eta"], "")
                etd = pick(r, ["etd","ETD","departure","depart_time"], "")
                draft = pick(r, ["draft_m","draft","draftMeter"], 0)
                loa = pick(r, ["loa_m","loa","length","LOA"], 0)
                moves = pick(r, ["moves","planned_moves","total_moves"], 0)

                rows.append({
                    "vessel_id": name,
                    "vessel_name": name,
                    "imo": pick(r, ["imo","IMO"], ""),
                    "mmsi": pick(r, ["mmsi","MMSI"], ""),
                    "carrier": pick(r, ["carrier","line","shipping_line"], ""),
                    "service": pick(r, ["service","loop"], ""),
                    "eta": eta,
                    "etd": etd,
                    "berth_id": berth,
                    "quay": pick(r, ["quay","quayName","berth_quay"], ""),
                    "draft_m": to_float(draft, 0.0),
                    "loa_m": to_float(loa, 0.0),
                    "moves": to_int(moves, 0),
                    "tide_window": pick(r, ["tide_window","tideWindow"], ""),
                    "tug_class": pick(r, ["tug_class","tugClass"], ""),
                    "priority": pick(r, ["priority"], "normal"),
                    "remarks": pick(r, ["remarks","note","notes"], ""),
                    "port_code": port,
                    "_source": r.get("_source") or status.get("mode"),
                })
            return JSONResponse(rows)

        # ==== Fallback: 读取 CSV（模块内） ====
        from pathlib import Path
        import csv
        from datetime import datetime

        def _parse_iso(ts: str) -> datetime:
            try:
                if ts.endswith("Z"):
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return datetime.fromisoformat(ts)
            except Exception:
                return datetime.fromisoformat(ts + "T00:00:00+00:00")

        t0 = _parse_iso(start)
        t1 = _parse_iso(end)

        base_dir = Path(__file__).resolve().parents[1]  # app/
        candidates = [
            base_dir / "services" / "app_center" / "data" / "vessel_plan_sipg.csv",
            base_dir / "services" / "app_center" / "data" / "vessel_plan.csv",
            base_dir / "rl_model" / "port_G_qc_mvp" / "data" / "vessel_plan_sipg.csv",
            base_dir / "rl_model" / "port_G_qc_mvp" / "data" / "vessel_plan.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return JSONResponse([])

        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                eta_str = (r.get("eta") or "").strip()
                if not eta_str:
                    continue
                try:
                    eta = _parse_iso(eta_str)
                except Exception:
                    continue
                if not (t0 <= eta <= t1):
                    continue

                def _get(name, default=""):
                    v = r.get(name, default)
                    return v.strip() if isinstance(v, str) else v

                def _pick(keys, default=""):
                    for k in keys:
                        v = r.get(k)
                        if v is not None and (not isinstance(v, str) or v.strip() != ""):
                            return v.strip() if isinstance(v, str) else v
                    return default

                def _to_float(x, dv=0.0):
                    try:
                        return float(x)
                    except Exception:
                        return dv

                def _to_int(x, dv=0):
                    try:
                        return int(float(x))
                    except Exception:
                        return dv

                rows.append({
                    # 关键：船名的多别名兜底
                    "vessel_id": _pick(
                        ["vessel_id", "name", "vessel", "ship_name", "vesselName", "VesselName", "vsl_name", "cn_name",
                         "en_name"],
                        "—"
                    ),
                    "vessel_name": _pick(  # <— 新增这一行
                        ["vessel_name", "name", "vessel", "ship_name", "vesselName", "VesselName", "vsl_name",
                         "cn_name", "en_name"],
                        "—"
                    ),
                    "imo": _pick(["imo", "IMO"], ""),
                    "mmsi": _pick(["mmsi", "MMSI"], ""),
                    "carrier": _pick(["carrier", "line", "shipping_line"], ""),
                    "service": _pick(["service", "loop"], ""),
                    "eta": _pick(["eta", "ETA", "arrival", "arrive_time", "ata_eta"], _get("eta")),
                    "etd": _pick(["etd", "ETD", "departure", "depart_time"], _get("etd")),
                    "berth_id": _pick(["berth_id", "berth", "berthNo", "berth_code", "berthCode"], _get("berth_id")),
                    "quay": _pick(["quay", "quayName", "berth_quay"], _get("quay")),
                    "draft_m": _to_float(_pick(["draft_m", "draft", "draftMeter"], 0.0)),
                    "loa_m": _to_float(_pick(["loa_m", "loa", "length", "LOA"], 0.0)),
                    "moves": _to_int(_pick(["moves", "planned_moves", "total_moves"], 0)),
                    "tide_window": _get("tide_window"),
                    "tug_class": _get("tug_class"),
                    "priority": _get("priority", "normal"),
                    "remarks": _get("remarks"),
                    "port_code": port,
                    "_source": f"csv:{path.name}",
                })

        return JSONResponse(rows)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"vessel schedule adapter failed: {exc}") from exc


@app.get("/external/power/tou_tariff", tags=["external"])
async def external_tou(
    date: str = Query(..., description="日期 YYYY-MM-DD"),
    port: str = Query("CN_DEMO", description="港口代码"),
) -> JSONResponse:
    try:
        sch = getattr(di, "schedule", None)
        if sch is None:
            raise HTTPException(status_code=503, detail="tariff adapter unavailable")
        status = sch.source_status() if hasattr(sch, "source_status") else {"mode": "unavailable"}
        if status.get("mode") == "unavailable":
            raise HTTPException(status_code=503, detail="tariff adapter is not configured")
        return JSONResponse(sch.tou_tariff(date=date, port_code=port))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"tariff adapter failed: {exc}") from exc

# —— TOS / WMS：船期/泊位/工单/堆场/预约 —— #
@app.get("/external/tos/vessels", tags=["external"])
async def ext_tos_vessels(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
) -> JSONResponse:
    if not _tos:
        raise HTTPException(status_code=503, detail="TOS adapter unavailable")
    try:
        return JSONResponse(_tos.vessel_calls(_parse_iso(start), _parse_iso(end)))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TOS vessel query failed: {exc}") from exc

@app.get("/external/tos/berths", tags=["external"])
async def ext_tos_berths(
    date: str = Query(..., description="日期（任意时分秒会被忽略）")
) -> JSONResponse:
    if not _tos:
        raise HTTPException(status_code=503, detail="TOS adapter unavailable")
    try:
        d0 = _parse_iso(date).replace(hour=0, minute=0, second=0, microsecond=0)
        return JSONResponse(_tos.berth_plan(d0))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TOS berth query failed: {exc}") from exc

@app.get("/external/tos/move_orders", tags=["external"])
async def ext_tos_moves(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
    status: Optional[str] = Query(None, description="PLANNED|INPROGRESS|DONE|CANCELLED")
) -> JSONResponse:
    if not _tos:
        raise HTTPException(status_code=503, detail="TOS adapter unavailable")
    try:
        return JSONResponse(_tos.move_orders(_parse_iso(start), _parse_iso(end), status=status))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TOS move-order query failed: {exc}") from exc

@app.get("/external/tos/yard", tags=["external"])
async def ext_tos_yard() -> JSONResponse:
    if not _tos:
        raise HTTPException(status_code=503, detail="TOS adapter unavailable")
    try:
        return JSONResponse(_tos.yard_inventory())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TOS yard query failed: {exc}") from exc

@app.get("/external/tos/truck_appts", tags=["external"])
async def ext_tos_truck(
    date: str = Query(..., description="日期")
) -> JSONResponse:
    if not _tos:
        raise HTTPException(status_code=503, detail="TOS adapter unavailable")
    try:
        d0 = _parse_iso(date).replace(hour=0, minute=0, second=0, microsecond=0)
        return JSONResponse(_tos.truck_appointments(d0))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TOS truck query failed: {exc}") from exc


# —— 电力市场：电价/需量/DR/碳价/绿证/边际因子 —— #
@app.get("/external/market/day_ahead", tags=["external"])
async def ext_mkt_da(
    date: str = Query(..., description="日期 YYYY-MM-DD")
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        d0 = _parse_iso(date).replace(hour=0, minute=0, second=0, microsecond=0)
        return JSONResponse(_market.day_ahead_price(d0))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"day-ahead price query failed: {exc}") from exc

@app.get("/external/market/real_time", tags=["external"])
async def ext_mkt_rt(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601")
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        return JSONResponse(_market.real_time_price(_parse_iso(start), _parse_iso(end)))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"real-time price query failed: {exc}") from exc

@app.get("/external/market/demand_limit", tags=["external"])
async def ext_mkt_dlimit(
    month: str = Query(..., description="YYYY-MM")
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        return JSONResponse(_market.demand_limit(month))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"demand-limit query failed: {exc}") from exc

@app.get("/external/market/demand_charge", tags=["external"])
async def ext_mkt_dcharge(
    month: str = Query(..., description="YYYY-MM")
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        return JSONResponse(_market.demand_charge(month))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"demand-charge query failed: {exc}") from exc

@app.get("/external/market/dr_events", tags=["external"])
async def ext_mkt_dr(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        return JSONResponse(_market.dr_events(_parse_iso(start), _parse_iso(end)))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"demand-response query failed: {exc}") from exc

@app.get("/external/market/carbon_price", tags=["external"])
async def ext_mkt_carbon(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        return JSONResponse(_market.carbon_price(_parse_iso(start), _parse_iso(end)))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"carbon-price query failed: {exc}") from exc

@app.get("/external/market/grid_factor", tags=["external"])
async def ext_mkt_gf(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        return JSONResponse(_market.grid_factor(_parse_iso(start), _parse_iso(end)))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"grid-factor query failed: {exc}") from exc

@app.get("/external/market/rec_price", tags=["external"])
async def ext_mkt_rec(
    date: str = Query(..., description="日期 YYYY-MM-DD")
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        d0 = _parse_iso(date).replace(hour=0, minute=0, second=0, microsecond=0)
        return JSONResponse(_market.rec_price(d0))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"REC-price query failed: {exc}") from exc

@app.get("/external/market/signals", tags=["external"])
async def ext_mkt_signals(
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
    prefer: str = Query("RT", description="RT|DA 优先级")
) -> JSONResponse:
    if not _market:
        raise HTTPException(status_code=503, detail="market adapter unavailable")
    try:
        return JSONResponse(_market.compose_signals(_parse_iso(start), _parse_iso(end), prefer=prefer))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"market-signal composition failed: {exc}") from exc


# —— AIS + 潮汐 —— #
@app.get("/external/ais/live", tags=["external"])
async def ext_ais_live(
    lat: float = Query(None, description="中心纬度（默认港口中心）"),
    lon: float = Query(None, description="中心经度（默认港口中心）"),
    radius_km: float = Query(25.0, description="搜索半径 km")
) -> JSONResponse:
    if not _ais:
        raise HTTPException(status_code=503, detail="AIS/tide adapter unavailable")
    try:
        return JSONResponse(_ais.live_ships(lat, lon, radius_km))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AIS live query failed: {exc}") from exc

@app.get("/external/ais/track", tags=["external"])
async def ext_ais_track(
    mmsi: str = Query(..., description="MMSI"),
    hours: int = Query(6, ge=1, le=48, description="回看小时数")
) -> JSONResponse:
    if not _ais:
        raise HTTPException(status_code=503, detail="AIS/tide adapter unavailable")
    try:
        return JSONResponse(_ais.track(mmsi, hours=hours))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AIS track query failed: {exc}") from exc

@app.get("/external/ais/tide", tags=["external"])
async def ext_tide_series(
    lat: float = Query(None, description="中心纬度（默认港口中心）"),
    lon: float = Query(None, description="中心经度（默认港口中心）"),
    start: str = Query(..., description="开始时间 ISO8601"),
    end: str = Query(..., description="结束时间 ISO8601"),
) -> JSONResponse:
    if not _ais:
        raise HTTPException(status_code=503, detail="AIS/tide adapter unavailable")
    try:
        return JSONResponse(_ais.tide_series(lat, lon, _parse_iso(start), _parse_iso(end)))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"tide query failed: {exc}") from exc

@app.get("/external/ais/context", tags=["external"])
async def ext_ais_ctx(
    hours_ahead: int = Query(24, ge=1, le=72, description="生成未来多少小时上下文")
) -> JSONResponse:
    if not _ais:
        raise HTTPException(status_code=503, detail="AIS/tide adapter unavailable")
    try:
        return JSONResponse(_ais.compose_context(hours_ahead=hours_ahead))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AIS/tide context failed: {exc}") from exc


# -------------------------------------------------
# RL 基础、Twin 仿真、报表
# -------------------------------------------------
@app.post("/api/rl/propose", tags=["rl"])
async def rl_propose(
    payload: Dict[str, Any] = Body(...)
) -> JSONResponse:
    job_id = str(payload.get("job_id") or TRAINING_MANAGER.latest_job_id or "")
    if not job_id:
        raise HTTPException(status_code=409, detail="no trained policy is available; provide job_id")
    try:
        return JSONResponse(await asyncio.to_thread(TRAINING_MANAGER.predict, job_id, payload))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown training job: {job_id}") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/twin/run/{asset_id}", tags=["twin"])
async def twin_run(
    asset_id: str,
    horizon_min: int = Query(360, ge=1, le=24*60),
    step_min:   int = Query(1, ge=1, le=60),
    scenario:   str = Query("baseline", description="场景：baseline/heatwave/typhoon/dense_berthing/islanded"),
    use_drivers:int = Query(1, ge=0, le=1, description="是否使用外部驱动（船期/TOU/天气）")
) -> JSONResponse:
    """
    数字孪生仿真（单资产）：
      - 默认场景 baseline；支持 heatwave/typhoon/dense_berthing/islanded
      - 返回 kW=p50，并包含 p10/p90
    """
    try:
        data = di.twin.run(
            asset_id=asset_id,
            horizon_min=horizon_min,
            step_min=step_min,
            scenario=scenario,
            use_drivers=bool(use_drivers),
        )
        return JSONResponse(data)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"twin adapter failed: {exc}") from exc



@app.get("/api/reporting/mini/{asset_id}", tags=["reporting"])
async def reporting_mini(asset_id: str) -> JSONResponse:
    return JSONResponse(di.rpt.generate_mini_report(asset_id))


# -------------------------------------------------
# 场景聚合（给大屏左上角“总负荷小曲线”）
# -------------------------------------------------
@app.get("/api/scene/snapshot", tags=["scene"])
async def scene_snapshot(
    mode: str = Query("now", pattern="^(now|forecast|sim)$"),
    limit: int = Query(50, ge=1, le=200),
) -> JSONResponse:
    try:
        assets = di.telemetry.list_assets() or []
    except Exception:
        assets = []
    assets = assets[:limit]
    out_items: List[Dict[str, Any]] = []
    if not assets:
        return JSONResponse({"mode": mode, "available": False, "reason": "telemetry asset registry is empty", "assets": []})

    if mode == "now":
        sum_kw = 0.0
        for a in assets:
            aid = a["id"]
            try:
                pts = di.telemetry.get_recent_power(aid) or []
                if not pts:
                    continue
                kw = float(pts[-1]["kW"])
            except Exception:
                continue
            out_items.append({"id": aid, "kw": round(kw, 3)})
            sum_kw += kw
        return JSONResponse({"mode": mode, "available": bool(out_items), "updated": datetime.now(timezone.utc).isoformat(), "assets": out_items, "sum_kw": round(sum_kw, 3), "_source": "telemetry_adapter"})

    elif mode == "forecast":
        sum_kw = 0.0
        for a in assets:
            aid = a["id"]
            try:
                seq = (di.fcst.forecast_load([aid], horizon_min=360, step_min=1) or {}).get(aid, [])
                vals = [float(p.get("kW", 0.0)) for p in seq if isinstance(p, dict)]
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
            except Exception:
                continue
            out_items.append({"id": aid, "kw": round(avg, 3)})
            sum_kw += avg
        return JSONResponse({"mode": mode, "available": bool(out_items), "updated": datetime.now(timezone.utc).isoformat(), "assets": out_items, "sum_kw": round(sum_kw, 3), "_source": "ridge_autoregression"})

    else:  # sim
        sum_kw = 0.0
        for a in assets:
            aid = a["id"]
            try:
                sim = di.twin.run(aid) or {}
                plan = sim.get("plan") or []
                vals = [float(p.get("kW", 0.0)) for p in plan if isinstance(p, dict)]
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
            except Exception:
                continue
            out_items.append({"id": aid, "kw": round(avg, 3)})
            sum_kw += avg
        return JSONResponse({"mode": mode, "available": bool(out_items), "updated": datetime.now(timezone.utc).isoformat(), "assets": out_items, "sum_kw": round(sum_kw, 3), "_source": "twin_adapter"})

@app.get("/api/scene/aggregate_sim", tags=["scene"])
async def scene_aggregate_sim(
    scenario: str = Query("baseline", description="场景：baseline/heatwave/typhoon/dense_berthing/islanded"),
    horizon_min: int = Query(360, ge=30, le=24*60, description="仿真时窗（分钟）"),
    step_min: int = Query(1, ge=1, le=60, description="步长（分钟）"),
    limit: int = Query(50, ge=1, le=200, description="纳入聚合的设备上限")
) -> JSONResponse:
    """
    聚合仿真（多设备）：
      - 返回 agg.p10/agg.p50/agg.p90 三条曲线（单位 kW）与总量指标
      - 仅聚合 Twin 适配器返回；不回退预测或合成分位区间
    """
    try:
        data = aggregate_sim(di, scenario=scenario, horizon_min=horizon_min, step_min=step_min, limit=limit)
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scene.aggregate_sim 失败: {e}")
# -------------------------------------------------
# 需量峰值风险（15min滚动）：曲线服务 · CurvesPeakRisk
# -------------------------------------------------
@app.get("/api/curves/peak_risk", tags=["curves"])
async def curves_peak_risk(
    mode: str = Query("forecast", pattern="^(now|forecast|sim)$"),
    horizon_min: int = Query(360, ge=1, le=24*60),
    step_min: int = Query(1, ge=1, le=60),
    limit: int = Query(200, ge=1, le=500),
    cap_kw: float = Query(500.0, ge=0.0, description="需量阈值（kW）"),
    avg_window_min: int = Query(15, ge=1, le=120, description="滚动窗口（分钟），常用 15min"),
) -> JSONResponse:
    """
    根据聚合功率分位（p10/p50/p90）估计月度需量/结算需量的超阈概率：
      - 先做 avg_window_min 的滚动均值
      - 用 (p90-p10)/(2*z) 估计标准差，z≈1.28155（10%/90%）
      - 返回 P(rolling_avg > cap_kw) 的风险曲线
    """
    data = peak_risk.peak_risk(
        mode=mode,
        horizon_min=horizon_min,
        step_min=step_min,
        limit=limit,
        cap_kw=cap_kw,
        avg_window_min=avg_window_min,
    )
    return JSONResponse(data)
@app.get("/api/curves/carbon_intensity", tags=["curves"])
async def curves_carbon_intensity(
    mode: str = Query("forecast", pattern="^(now|forecast|sim)$"),
    horizon_min: int = Query(360, ge=1, le=24*60),
    step_min: int = Query(1, ge=1, le=60),
    limit: int = Query(200, ge=1, le=500),
    teu: float = Query(12000.0, ge=1.0, description="窗口内 TEU 总量（分母）"),
    ef_const_kg_per_kwh: float = Query(0.55, ge=0.0, le=5.0, description="无时变因子时使用的显式计算参数"),
) -> JSONResponse:
    data = carbon_intensity.intensity(
        mode=mode, horizon_min=horizon_min, step_min=step_min, limit=limit,
        teu=teu, ef_const_kg_per_kwh=ef_const_kg_per_kwh
    )
    return JSONResponse(data)

@app.get("/api/curves/economic_benefit", tags=["curves"])
async def curves_economic_benefit(
    mode: str = Query("sim", pattern="^(now|forecast|sim)$"),
    horizon_min: int = Query(360, ge=1, le=24*60),
    step_min: int = Query(1, ge=1, le=60),
    limit: int = Query(200, ge=1, le=500),
    scenario_base: str = Query("baseline"),
    scenario_opt:  str = Query("strategy"),
    price_const_y_per_kwh: float = Query(0.85, ge=0.0),
    ef_const_kg_per_kwh: float = Query(0.55, ge=0.0),
    carbon_price_const_y_per_ton: float = Query(50.0, ge=0.0),
    demand_rate_y_per_kw: float = Query(0.0, ge=0.0),
    demand_avg_window_min: int = Query(15, ge=1, le=120),
) -> JSONResponse:
    data = economic_benefit.benefit(
        mode=mode,
        horizon_min=horizon_min,
        step_min=step_min,
        scenario_base=scenario_base,
        scenario_opt=scenario_opt,
        limit=limit,
        price_const_y_per_kwh=price_const_y_per_kwh,
        ef_const_kg_per_kwh=ef_const_kg_per_kwh,
        carbon_price_const_y_per_ton=carbon_price_const_y_per_ton,
        demand_rate_y_per_kw=demand_rate_y_per_kw,
        demand_avg_window_min=demand_avg_window_min,
    )
    return JSONResponse(data)

@app.get("/api/curves/bess_capability", tags=["curves"])
async def curves_bess_capability(
    asset_id: str = Query("bess-01"),
    mode: str = Query("forecast", pattern="^(now|forecast|sim)$"),
    horizon_min: int = Query(120, ge=1, le=24*60),
    step_min: int = Query(1, ge=1, le=60),
    rating_kw: float = Query(1000.0, ge=0.0),
    energy_mwh: float = Query(2.0, ge=0.0),
    soc_init_pct: float = Query(60.0, ge=0.0, le=100.0),
    soc_min_pct: float = Query(20.0, ge=0.0, le=100.0),
    soc_max_pct: float = Query(90.0, ge=0.0, le=100.0),
) -> JSONResponse:
    """
    BESS 调度能力曲线：SoC% / 可充功率 / 可放功率 及累计上/下调能量（MWh）。
    数据源优先级：di.energy.* → 已训练 RL 策略预演；无来源时返回 unavailable。
    """
    data = bess_capability.capability(
        asset_id=asset_id, mode=mode,
        horizon_min=horizon_min, step_min=step_min,
        rating_kw=rating_kw, energy_mwh=energy_mwh,
        soc_init_pct=soc_init_pct, soc_min_pct=soc_min_pct, soc_max_pct=soc_max_pct,
    )
    return JSONResponse(data)



# -------------------------------------------------
# -------------------------------
# 曲线服务（统一供前端小图使用）
# -------------------------------
@app.get("/api/curves/aggregate", tags=["curves"])
async def curves_aggregate(
    mode: str = Query("forecast", pattern="^(now|forecast|sim)$"),
    horizon_min: int = Query(360, ge=1, le=24*60),
    step_min: int = Query(1, ge=1, le=60),
    limit: int = Query(50, ge=1, le=200),
    scenario: str = Query("baseline"),
    use_drivers: int = Query(1, ge=0, le=1),
) -> JSONResponse:
    data = curves.aggregate(mode=mode, horizon_min=horizon_min, step_min=step_min,
                            limit=limit, scenario=scenario, use_drivers=bool(use_drivers))
    return JSONResponse(data)

@app.get("/api/curves/asset/{asset_id}", tags=["curves"])
async def curves_asset(
    asset_id: str,
    mode: str = Query("forecast", pattern="^(now|forecast|sim)$"),
    horizon_min: int = Query(360, ge=1, le=24*60),
    step_min: int = Query(1, ge=1, le=60),
    scenario: str = Query("baseline"),
    use_drivers: int = Query(1, ge=0, le=1),
) -> JSONResponse:
    data = curves.asset(asset_id=asset_id, mode=mode, horizon_min=horizon_min,
                        step_min=step_min, scenario=scenario, use_drivers=bool(use_drivers))
    return JSONResponse(data)

# 指挥盘 KPI 聚合（保持你的原实现）
# -------------------------------------------------
@app.get("/api/energy/today", tags=["dashboard"])
async def energy_today(
    teu: int = Query(12000, ge=1, description="今日吞吐量（TEU），用于强度指标"),
    limit: int = Query(50, ge=1, le=200, description="纳入计算的资产上限"),
    min_integral_coverage_min: float = Query(30.0, ge=0.0, le=600.0, description="采用积分口径所需的最小覆盖分钟数阈值（资产≥60%达标才切积分）"),
    horizon_min: int = Query(360, ge=30, le=24*60, description="峰/平/谷分摊的未来预测时窗（分钟）"),
    step_min: int = Query(1, ge=1, le=60, description="预测步长（分钟）"),
) -> JSONResponse:
    try:
        summary = di.energy.build_today_summary(
            teu=teu,
            limit_assets=limit,
            min_integral_coverage_min=min_integral_coverage_min,
            horizon_min=horizon_min,
            step_min=step_min,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"energy.today 聚合失败: {e}")

    elec = summary.get("electricity", {})
    if elec.get("kWh") is not None:
        elec["kWh_est"] = elec["kWh"]

    payload = {
        "available": summary.get("available", True),
        "reason": summary.get("reason"),
        "latest_telemetry_at": summary.get("latest_telemetry_at"),
        "range": summary.get("range", {}),
        "electricity": elec,
        "oil": summary.get("oil", {}),
        "gas": summary.get("gas", {}),
        "intensity": summary.get("intensity", {}),
        "utilization_percent": summary.get("utilization_percent"),
        "assumptions": summary.get("assumptions", {}),
        "_source": summary.get("_source", "energy_service"),
    }
    return JSONResponse(payload)


# -------------------------------------------------
# 预警接口（保持）
# -------------------------------------------------
@app.get("/api/alerts/scan", tags=["alerts"])
async def alerts_scan(
    teu: int = Query(12000, ge=1, description="今日 TEU，用于强度与碳配额评估"),
    demand_limit_kw: float = Query(500.0, ge=1, description="需量越峰阈值（kW）"),
    quota_kgco2e: float = Query(5000.0, ge=1, description="当日碳配额（kgCO₂e）"),
    limit: int = Query(50, ge=1, le=200, description="纳入扫描的资产上限"),
    horizon_min: int = Query(360, ge=30, le=24*60, description="预测/仿真扫描时窗（分钟）"),
    step_min: int = Query(1, ge=1, le=60, description="预测步长（分钟）"),
) -> JSONResponse:
    try:
        result = di.alerts.scan(
            teu=teu,
            demand_limit_kw=demand_limit_kw,
            quota_kgco2e=quota_kgco2e,
            limit=limit,
            horizon_min=horizon_min,
            step_min=step_min,
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"alerts.scan 失败: {e}")


# =================================================


# RL 策略面板接口（列表 + 模拟执行）
# =================================================
@app.get("/api/rl/strategies", tags=["rl-panel-real-models"])
async def rl_list_strategies(
    horizon_min: int = Query(360, ge=30, le=24*60, description="策略评估的预测时窗（分钟）"),
    step_min: int = Query(5, ge=1, le=60, description="评估步长（分钟）"),
    max_items: int = Query(8, ge=1, le=50, description="返回条目上限"),
) -> JSONResponse:
    registry = TRAINING_MANAGER.model_registry().list()
    strategies = []
    for record in registry.get("models", [])[:max_items]:
        evaluation = record.get("evaluation") or {}
        metrics = evaluation.get("metrics") or {}
        violation_rate = metrics.get("guardrail_violation_rate")
        strategies.append({
            "id": record.get("job_id"),
            "title": f"{str(record.get('algorithm') or '').upper()} · {record.get('job_id')}",
            "category": "registered_model",
            "impact": {
                "reward": metrics.get("reward"),
                "peak_kw": metrics.get("peak_kw"),
                "guardrail_violation_rate": violation_rate,
                "risk_level": "unassessed" if violation_rate is None else ("high" if float(violation_rate) > 0.05 else "reviewable"),
            },
            "explain": {
                "reason": "读取本地模型登记与真实留出集评测；未生成预期收益。",
            },
            "meta": {
                "source": "verified_model_registry",
                "algorithm": record.get("algorithm"),
                "dataset_id": record.get("dataset_id"),
                "dataset_sha256": record.get("dataset_sha256"),
                "artifact_verified": (record.get("artifact") or {}).get("verified"),
                "evaluation_available": evaluation.get("available"),
                "aliases": record.get("aliases") or [],
            },
        })
    return JSONResponse({
        "strategies": strategies,
        "count": len(strategies),
        "source": "verified_model_registry",
        "generated_values": False,
        "requested_display_horizon_min": horizon_min,
        "requested_display_step_min": step_min,
    })


@app.post("/api/rl/simulate", tags=["rl-panel-heldout-evaluation"])
async def rl_simulate(
    payload: Dict[str, Any] = Body(
        default={
            "strategy_id": "<registered-job-id>",
            "episodes": 10,
        }
    )
) -> JSONResponse:
    try:
        job_id = str(payload.get("job_id") or payload.get("strategy_id") or "").strip()
        if not job_id:
            raise HTTPException(status_code=422, detail="job_id or registered strategy_id is required")
        episodes = max(5, min(50, int(payload.get("episodes") or 10)))
        result = await asyncio.to_thread(TRAINING_MANAGER.evaluate, job_id, episodes)
        frames = (result.get("render") or {}).get("frames") or []
        baseline = [float(item.get("baseline_kw") or 0.0) for item in frames]
        policy = [float(item.get("net_load_kw") or 0.0) for item in frames]
        peak_reduction = (max(baseline) - max(policy)) if baseline and policy else None
        window = {
            "start": frames[0].get("timestamp") if frames else None,
            "end": frames[-1].get("timestamp") if frames else None,
        }
        return JSONResponse({
            "mode": "chronological_holdout_evaluation",
            "production_dispatched": False,
            "strategy_id": job_id,
            "summary": {
                "delta_kWh": None,
                "delta_carbon_kg": None,
                "peak_reduction_kW": peak_reduction,
                "window": window,
                "dispatch_ready": False,
                "reason": "留出集评测只产生决策证据，不授予设备执行权。",
            },
            "baseline": {"agg_kW": baseline},
            "simulated": {"agg_kW": policy},
            "feasibility": {
                "guardrail_violation_rate": (result.get("metrics") or {}).get("guardrail_violation_rate"),
                "software_evaluation_only": True,
            },
            "evaluation": result,
            "audit_trace": {
                "job_id": job_id,
                "dataset_id": result.get("dataset_id"),
                "dataset_sha256": result.get("dataset_sha256"),
                "evaluation_protocol": result.get("evaluation_protocol"),
            },
        })
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown registered model: {job_id}") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rl.simulate 失败: {e}")


_RL_FUTURE_RUNS: List[Dict[str, Any]] = []


def _rl_future_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@_engineering_route("post", "/api/rl/future/run", tags=["rl-future-engineering-simulator"])
async def rl_future_run(
    payload: Dict[str, Any] = Body(
        default={"horizon_min": 90, "step_min": 5, "max_candidates": 3, "source": "rl-future-deck"}
    )
) -> JSONResponse:
    """运行候选生成、反事实模拟、护栏校验和审计编排；不触发生产下发。"""
    try:
        horizon_min = max(30, min(24 * 60, int(payload.get("horizon_min", 90))))
        step_min = max(1, min(60, int(payload.get("step_min", 5))))
        max_candidates = max(1, min(3, int(payload.get("max_candidates", 3))))
        now = datetime.now(timezone.utc)
        run_id = f"FUT-{now.strftime('%Y%m%d-%H%M%S')}-{now.microsecond // 1000:03d}"

        strategy_payload = di.rlpanel.list_strategies(
            horizon_min=horizon_min,
            step_min=step_min,
            max_items=12,
        ) or {}
        available = strategy_payload.get("strategies", [])
        if not available:
            raise HTTPException(status_code=503, detail="当前没有可用于推演的策略候选。")

        # 固定选出低/中/观察三档，避免只按收益排序掩盖风险差异。
        selected: List[Dict[str, Any]] = []
        used_ids = set()
        for risk_level in ("low", "medium", "watch", "high"):
            match = next(
                (
                    item for item in available
                    if str((item.get("impact") or {}).get("risk_level", "")).lower() == risk_level
                    and item.get("id") not in used_ids
                ),
                None,
            )
            if match:
                selected.append(match)
                used_ids.add(match.get("id"))
            if len(selected) >= max_candidates:
                break
        for item in available:
            if len(selected) >= max_candidates:
                break
            if item.get("id") not in used_ids:
                selected.append(item)
                used_ids.add(item.get("id"))

        mode_names = ["保守稳态", "平衡收益", "进取削峰"]
        mode_tags = ["SAFE", "BALANCED", "REVIEW"]
        candidates: List[Dict[str, Any]] = []
        for index, strategy in enumerate(selected):
            impact = strategy.get("impact") or {}
            try:
                simulated = di.rlpanel.simulate(
                    strategy=strategy,
                    horizon_min=horizon_min,
                    step_min=step_min,
                ) or {}
                summary = simulated.get("summary") or {}
                feasibility = simulated.get("feasibility") or {}
                decision = feasibility.get("decision") or {}
                delta_kwh = _rl_future_number(summary.get("delta_kWh"))
                delta_carbon = _rl_future_number(summary.get("delta_carbon_kg"))
                baseline = simulated.get("baseline") or {}
                after = simulated.get("simulated") or {}
                candidate = {
                    "id": strategy.get("id"),
                    "title": strategy.get("title") or strategy.get("id"),
                    "mode": mode_names[min(index, len(mode_names) - 1)],
                    "tag": mode_tags[min(index, len(mode_tags) - 1)],
                    "risk_level": str(impact.get("risk_level", "unknown")).upper(),
                    "confidence": round(_rl_future_number(impact.get("confidence_0to1")), 3),
                    "dispatch_ready": bool(summary.get("dispatch_ready", decision.get("dispatch_ready", False))),
                    "energy_saving_kwh": round(max(0.0, -delta_kwh), 3),
                    "carbon_saving_kg": round(max(0.0, -delta_carbon), 3),
                    "peak_reduction_kw": round(max(0.0, _rl_future_number(summary.get("peak_reduction_kW"))), 3),
                    "baseline_energy_kwh": round(_rl_future_number(baseline.get("total_kWh")), 3),
                    "simulated_energy_kwh": round(_rl_future_number(after.get("total_kWh")), 3),
                    "baseline_peak_kw": round(_rl_future_number(baseline.get("peak_kW")), 3),
                    "simulated_peak_kw": round(_rl_future_number(after.get("peak_kW")), 3),
                    "delay_min": round(_rl_future_number(impact.get("throughput_delay_min_est")), 2),
                    "supports": decision.get("supports") or summary.get("supports") or [],
                    "blockers": decision.get("blockers") or summary.get("blockers") or [],
                    "risk_flags": feasibility.get("risk_flags") or [],
                    "reason": (strategy.get("explain") or {}).get("reason", ""),
                    "operator_guidance": (feasibility.get("operator_guidance") or {}).get("message", ""),
                }
            except Exception as simulation_error:
                candidate = {
                    "id": strategy.get("id"),
                    "title": strategy.get("title") or strategy.get("id"),
                    "mode": mode_names[min(index, len(mode_names) - 1)],
                    "tag": mode_tags[min(index, len(mode_tags) - 1)],
                    "risk_level": str(impact.get("risk_level", "unknown")).upper(),
                    "confidence": round(_rl_future_number(impact.get("confidence_0to1")), 3),
                    "dispatch_ready": False,
                    "energy_saving_kwh": 0.0,
                    "carbon_saving_kg": 0.0,
                    "peak_reduction_kw": 0.0,
                    "baseline_energy_kwh": 0.0,
                    "simulated_energy_kwh": 0.0,
                    "baseline_peak_kw": 0.0,
                    "simulated_peak_kw": 0.0,
                    "delay_min": round(_rl_future_number(impact.get("throughput_delay_min_est")), 2),
                    "supports": [],
                    "blockers": [f"反事实模拟失败：{simulation_error}"],
                    "risk_flags": ["simulation_error"],
                    "reason": (strategy.get("explain") or {}).get("reason", ""),
                    "operator_guidance": "保持阻断，不进入后续执行。",
                }
            candidates.append(candidate)

        ready_candidates = [item for item in candidates if item.get("dispatch_ready")]
        balanced = next((item for item in ready_candidates if item.get("tag") == "BALANCED"), None)
        recommended = balanced or (ready_candidates[0] if ready_candidates else candidates[0])

        from app.services.mas_orchestrator.service import OrchestratorService
        from app.services.rl_ops_center.service import RLOpsService

        situation = OrchestratorService().get_overview()
        agents = situation.get("agents") or {}
        bess_items = agents.get("bess") or []
        shore_items = agents.get("shore") or []
        bess_soc = _rl_future_number((bess_items[0] if bess_items else {}).get("soc"), 0.0)
        shore_power_kw = sum(_rl_future_number(item.get("power_kw")) for item in shore_items)

        rlops = RLOpsService()
        policy_verification = rlops.verify_policy(str(recommended.get("id") or "demo"))
        drift = rlops.signals()
        drift_value = _rl_future_number((drift.get("metrics") or {}).get("reward_drift"))
        drift_limit = _rl_future_number((drift.get("thresholds") or {}).get("reward_drift_max"), 0.08)
        projected_shore_kw = max(0.0, shore_power_kw - _rl_future_number(recommended.get("peak_reduction_kw")))

        guardrails = [
            {
                "id": "bess_soc",
                "name": "BESS 荷电状态安全带",
                "level": "hard",
                "actual": round(bess_soc * 100, 1),
                "unit": "%",
                "threshold": "35%–85%",
                "passed": 0.35 <= bess_soc <= 0.85,
                "source": "MAS 态势快照",
            },
            {
                "id": "shore_contract",
                "name": "岸电合同功率窗口",
                "level": "hard",
                "actual": round(projected_shore_kw, 1),
                "unit": "kW",
                "threshold": "≤ 5000 kW",
                "passed": projected_shore_kw <= 5000.0,
                "source": "MAS 岸电节点 + 反事实削峰量",
            },
            {
                "id": "service_delay",
                "name": "生产服务水平延迟",
                "level": "hard",
                "actual": round(_rl_future_number(recommended.get("delay_min")), 2),
                "unit": "min",
                "threshold": "≤ 5 min",
                "passed": _rl_future_number(recommended.get("delay_min")) <= 5.0,
                "source": "候选策略影响评估",
            },
            {
                "id": "model_drift",
                "name": "策略模型奖励漂移",
                "level": "soft",
                "actual": round(drift_value, 3),
                "unit": "",
                "threshold": f"≤ {drift_limit:.3f}",
                "passed": drift_value <= drift_limit,
                "source": "RL Ops 可观测性信号",
            },
            {
                "id": "policy_verify",
                "name": "策略动作爬坡审查",
                "level": "soft",
                "actual": len(policy_verification.get("violations") or []),
                "unit": " 条提示",
                "threshold": "无硬约束违规",
                "passed": bool(policy_verification.get("ok")),
                "source": "RL Ops 策略校验",
                "detail": "；".join(
                    str(item.get("detail", "")) for item in (policy_verification.get("violations") or [])
                ),
            },
        ]
        hard_passed = all(item.get("passed") for item in guardrails if item.get("level") == "hard")
        decision_ready = bool(recommended.get("dispatch_ready")) and hard_passed
        decision_status = "READY_FOR_HUMAN_DRY_RUN" if decision_ready else "BLOCKED_BY_GUARDRAIL"
        decision_label = "可进入人工确认演练" if decision_ready else "护栏阻断，禁止进入演练"

        audit_seed = json.dumps(
            {
                "run_id": run_id,
                "strategy": recommended.get("id"),
                "guardrails": [(item.get("id"), item.get("passed")) for item in guardrails],
                "decision": decision_status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        evidence_digest = hashlib.sha256(audit_seed.encode("utf-8")).hexdigest()[:16].upper()
        snapshot = {
            "timestamp": situation.get("ts") or now.isoformat(),
            "horizon_min": horizon_min,
            "bess_soc_pct": round(bess_soc * 100, 1),
            "shore_power_kw": round(shore_power_kw, 1),
            "reward_drift": round(drift_value, 3),
            "candidate_pool_size": len(available),
            "source_note": (strategy_payload.get("assumptions") or {}).get("note", ""),
        }
        stages = [
            {"id": "situation", "label": "态势锁定", "status": "passed"},
            {"id": "candidates", "label": "候选生成", "status": "passed"},
            {"id": "counterfactual", "label": "反事实模拟", "status": "passed" if ready_candidates else "blocked"},
            {"id": "guardrails", "label": "守护栏校验", "status": "passed" if hard_passed else "blocked"},
            {"id": "receipt", "label": "审计回执", "status": "passed"},
        ]
        run_logs = [
            f"[Situation] 已锁定态势快照：BESS SoC {snapshot['bess_soc_pct']}%，岸电功率 {snapshot['shore_power_kw']} kW",
            f"[Candidates] 从 {len(available)} 条策略中选出 {len(candidates)} 条风险分层候选",
        ]
        run_logs.extend(
            f"[Counterfactual] {item['mode']} / {item['title']}：节能 {item['energy_saving_kwh']:.2f} kWh，削峰 {item['peak_reduction_kw']:.2f} kW，{'可用' if item['dispatch_ready'] else '阻断'}"
            for item in candidates
        )
        run_logs.extend(
            f"[Guardrail] {item['name']}：{item['actual']}{item['unit']} / {item['threshold']} / {'PASS' if item['passed'] else 'BLOCK'}"
            for item in guardrails
        )
        run_logs.extend(
            [
                f"[Decision] 推荐候选：{recommended.get('mode')} / {recommended.get('title')}",
                f"[Boundary] {decision_label}；当前流程未调用生产下发接口",
                f"[Audit] 回执 {run_id}，证据摘要 {evidence_digest}",
            ]
        )

        response = {
            "run_id": run_id,
            "generated_at": now.isoformat(),
            "mode": "counterfactual_simulation_only",
            "production_dispatched": False,
            "snapshot": snapshot,
            "stages": stages,
            "candidates": candidates,
            "recommended_strategy_id": recommended.get("id"),
            "guardrails": guardrails,
            "decision": {
                "status": decision_status,
                "label": decision_label,
                "ready_for_human_dry_run": decision_ready,
                "recommended_strategy_id": recommended.get("id"),
                "recommended_strategy_title": recommended.get("title"),
                "next_action": "人工复核后进入 dry-run" if decision_ready else "修复阻断项后重新推演",
                "production_boundary": "本次只完成候选生成、反事实模拟、守护栏校验和审计记录，不下发生产控制指令。",
            },
            "audit": {
                "receipt_id": run_id,
                "evidence_digest": evidence_digest,
                "source": str(payload.get("source") or "rl-future-deck"),
                "recorded": True,
            },
            "logs": run_logs,
        }
        _RL_FUTURE_RUNS.insert(0, response)
        del _RL_FUTURE_RUNS[20:]
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rl.future.run 失败: {e}")


@_engineering_route("get", "/api/rl/future/history", tags=["rl-future-engineering-simulator"])
async def rl_future_history(limit: int = Query(10, ge=1, le=20)) -> JSONResponse:
    return JSONResponse({"items": _RL_FUTURE_RUNS[:limit], "count": min(limit, len(_RL_FUTURE_RUNS))})


_RL_TRAIN_BASELINES: List[Dict[str, Any]] = [
    {
        "id": "sac",
        "name": "SAC",
        "type": "RL",
        "description": "Soft Actor-Critic baseline for continuous dispatch control.",
        "port_scope": "BESS / shore-power / crane power allocation",
    },
    {
        "id": "ppo",
        "name": "PPO",
        "type": "RL",
        "description": "Proximal Policy Optimization baseline for stable constrained training.",
        "port_scope": "multi-objective operation policy",
    },
    {
        "id": "td3",
        "name": "TD3",
        "type": "RL",
        "description": "Twin Delayed DDPG baseline for continuous action optimization.",
        "port_scope": "peak shaving and dispatch setpoint control",
    },
    {
        "id": "dqn",
        "name": "DQN",
        "type": "RL",
        "description": "Deep Q-Network baseline over the documented discrete port-control lattice.",
        "port_scope": "discrete dispatch and equipment mode selection",
    },
    {
        "id": "mpc",
        "name": "MPC",
        "type": "Control",
        "description": "Receding-horizon model predictive control solved with constrained numerical optimization.",
        "port_scope": "BESS, peak and service setpoint control",
    },
]
_RL_TRAIN_JOBS: Dict[str, Dict[str, Any]] = {}
_RL_TRAIN_STATUS: Dict[str, Dict[str, Any]] = {}
_RL_MOBILE_TRAIN_REQUESTS: Dict[str, Dict[str, Any]] = {}
_RL_DESKTOP_PANEL_STATE: Dict[str, Any] = {
    "url": "http://127.0.0.1:8000/rl-panel",
    "launch_url": None,
    "launch_requested_at": None,
    "last_seen_at": None,
    "operator": None,
    "launch_error": None,
}


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except Exception:
        return fallback


def _latest_train_job_id() -> Optional[str]:
    if TRAINING_MANAGER.latest_job_id:
        return TRAINING_MANAGER.latest_job_id
    if not _RL_TRAIN_JOBS:
        return None
    return max(_RL_TRAIN_JOBS.values(), key=lambda item: str(item.get("created_at") or "")).get("job_id")


def _policy_version_for(algorithm: str, step: int, total_steps: int) -> str:
    total = max(1, int(total_steps or 1))
    version = min(99, max(1, int((max(0, step) / total) * 24) + 1))
    return f"{str(algorithm or 'sac').upper()}-policy-v{version:02d}"


def _train_status_summary(status: Dict[str, Any]) -> str:
    metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
    step = int(_number(status.get("step", metrics.get("step")), 0))
    total_steps = int(_number(status.get("total_steps", metrics.get("total_steps")), 0))
    reward = _number(status.get("reward", metrics.get("reward")), 0.0)
    entropy = _number(status.get("entropy", metrics.get("entropy")), 0.0)
    policy_version = str(status.get("policy_version") or metrics.get("policy_version") or "—")
    logs = status.get("logs") if isinstance(status.get("logs"), list) else []
    return (
        f"运行摘要：{status.get('status', 'IDLE')} · "
        f"step={step:,} / {total_steps:,} · reward={reward:.2f} · "
        f"entropy={entropy:.4f} · policy={policy_version} · logs={len(logs)}"
    )


def _normalise_train_status(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    job = _RL_TRAIN_JOBS.get(job_id, {})
    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else job.get("config") or {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    total_steps = int(_number(payload.get("total_steps", metrics.get("total_steps", cfg.get("total_steps", 240000))), 240000))
    step = int(_number(payload.get("step", metrics.get("step")), 0))
    reward = _number(payload.get("reward", metrics.get("reward")), 0.0)
    entropy = _number(payload.get("entropy", metrics.get("entropy", cfg.get("entropy_coef", 0.02))), 0.02)
    algorithm = str(cfg.get("algorithm") or job.get("algorithm") or "sac")
    policy_version = str(payload.get("policy_version") or metrics.get("policy_version") or _policy_version_for(algorithm, step, total_steps))
    logs_raw = payload.get("logs") if isinstance(payload.get("logs"), list) else []
    logs = [str(line) for line in logs_raw if str(line).strip()][:18]
    status = {
        "job_id": job_id,
        "status": str(payload.get("status") or job.get("status") or "accepted").upper(),
        "progress": round(_number(payload.get("progress"), 0.0), 2),
        "stage": str(payload.get("stage") or "等待启动训练 / Waiting for start"),
        "step": step,
        "total_steps": total_steps,
        "reward": reward,
        "entropy": entropy,
        "policy_version": policy_version,
        "logs": logs,
        "metrics": {
            **metrics,
            "step": step,
            "total_steps": total_steps,
            "reward": reward,
            "entropy": entropy,
            "policy_version": policy_version,
        },
        "artifact_paths": payload.get("artifact_paths") or job.get("artifact_paths"),
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    status["summary"] = _train_status_summary(status)
    return status


def _initial_train_status(job: Dict[str, Any]) -> Dict[str, Any]:
    cfg = job.get("config") if isinstance(job.get("config"), dict) else {}
    return _normalise_train_status(
        str(job.get("job_id") or ""),
        {
            "status": job.get("status", "accepted"),
            "progress": 0,
            "stage": "训练任务已接收 / Job Accepted",
            "step": 0,
            "total_steps": cfg.get("total_steps", 240000),
            "reward": 0,
            "entropy": cfg.get("entropy_coef", 0.02),
            "config": cfg,
            "logs": [f"job accepted · {job.get('job_id')}"],
            "artifact_paths": job.get("artifact_paths"),
        },
    )


def _idle_train_status() -> Dict[str, Any]:
    status = {
        "job_id": None,
        "status": "IDLE",
        "progress": 0,
        "stage": "等待启动训练 / Waiting for start",
        "step": 0,
        "total_steps": 0,
        "reward": 0,
        "entropy": 0,
        "policy_version": "—",
        "logs": [],
        "metrics": {"step": 0, "total_steps": 0, "reward": 0, "entropy": 0, "policy_version": "—"},
        "artifact_paths": None,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    status["summary"] = _train_status_summary(status)
    return status


def _status_response(status: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "metrics": status.get("metrics", {}),
        "logs": status.get("logs", []),
        "summary": status.get("summary"),
    }


@app.get("/api/rl/train/baselines", tags=["rl-panel"])
async def rl_train_baselines(dataset_id: Optional[str] = Query(None)) -> JSONResponse:
    benchmark = TRAINING_MANAGER.baselines(dataset_id)
    return JSONResponse(
        {
            **benchmark,
            "adapter": {
                "train_start": "/api/rl/train/start",
                "mode": "stable_baselines3_backend",
                "progress_owned_by": "training_callback",
                "render_during_training": False,
                "evaluate_after_training": "/api/rl/train/{job_id}/evaluate",
            },
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )


def _rl_desktop_panel_payload() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    last_seen_raw = _RL_DESKTOP_PANEL_STATE.get("last_seen_at")
    panel_active = False
    if last_seen_raw:
        try:
            last_seen = datetime.fromisoformat(str(last_seen_raw).replace("Z", "+00:00"))
            panel_active = (now - last_seen).total_seconds() <= 10
        except ValueError:
            panel_active = False
    if panel_active:
        message = "电脑端强化学习审批页已打开，等待电脑端人工操作"
    elif _RL_DESKTOP_PANEL_STATE.get("launch_requested_at"):
        message = "桌面启动命令已发送，等待审批页心跳"
    else:
        message = "电脑端服务在线，审批页尚未打开"
    return {
        **_RL_DESKTOP_PANEL_STATE,
        "panel_active": panel_active,
        "message": message,
        "human_confirmation_required": True,
        "can_mobile_bypass": False,
        "updated_at": now.isoformat().replace("+00:00", "Z"),
    }


@app.get("/api/rl/desktop/status", tags=["rl-panel"])
async def rl_desktop_panel_status() -> JSONResponse:
    return JSONResponse(_rl_desktop_panel_payload())


@app.post("/api/rl/desktop/launch", tags=["rl-panel"])
async def rl_desktop_panel_launch() -> JSONResponse:
    url = str(_RL_DESKTOP_PANEL_STATE["url"])
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    cache_buster = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    launch_url = f"{url}?mobile_launch={cache_buster}"
    launched = False
    launch_error: Optional[str] = None
    try:
        if sys.platform == "darwin":
            command = ["open", launch_url]
        elif sys.platform.startswith("win"):
            command = ["cmd", "/c", "start", "", launch_url]
        else:
            command = ["xdg-open", launch_url]
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        launched = True
    except (OSError, ValueError) as exc:
        launch_error = str(exc)
    _RL_DESKTOP_PANEL_STATE.update(
        {
            "launch_requested_at": now,
            "launch_url": launch_url,
            "launch_error": launch_error,
        }
    )
    payload = _rl_desktop_panel_payload()
    payload.update(
        {
            "launched": launched,
            "message": "已请求电脑打开强化学习审批页" if launched else f"桌面启动失败：{launch_error}",
        }
    )
    return JSONResponse(payload, status_code=202 if launched else 503)


@app.post("/api/rl/desktop/heartbeat", tags=["rl-panel"])
async def rl_desktop_panel_heartbeat(
    payload: Dict[str, Any] = Body(default={"panel": "rl-panel"}),
) -> JSONResponse:
    _RL_DESKTOP_PANEL_STATE.update(
        {
            "last_seen_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "operator": str(payload.get("operator") or "港口调度员-01"),
            "launch_error": None,
        }
    )
    return JSONResponse(_rl_desktop_panel_payload())


def _rl_artifact_model_for_config(cfg: Dict[str, Any]) -> str:
    module_target = str(cfg.get("module_target") or "").strip()
    asset_group = str(cfg.get("asset_group") or "").strip()
    scenario = str(cfg.get("scenario") or "").strip()
    if module_target in {"yard_lighting", "hvac_cooling", "shore_bess", "bess_energy", "yard_crane"}:
        return module_target
    mapping = {
        "all_port": "agv_charge",
        "agv_charge": "agv_charge",
        "hvac_cooling": "hvac_cooling",
        "yard_lighting": "yard_lighting",
        "qc_bess_shore": "shore_bess",
        "shore_power": "shore_bess",
        "bess_energy": "bess_energy",
        "yard_crane": "yard_crane",
        "berth_ops": "yard_crane",
    }
    if asset_group in mapping:
        return mapping[asset_group]
    if "shore" in asset_group or "shore" in scenario or "bess" in asset_group:
        return "shore_bess"
    if "hvac" in asset_group:
        return "hvac_cooling"
    if "lighting" in asset_group:
        return "yard_lighting"
    return "agv_charge"


def _rl_training_artifact_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model = _rl_artifact_model_for_config(cfg)
    root = Path(__file__).resolve().parent / "services" / "rl_model" / model
    artifacts_dir = root / "artifacts"
    if artifacts_dir.exists():
        base_url = f"/api/rl/model/{model}/artifacts"
        local_root = artifacts_dir
    else:
        base_url = f"/api/rl/model/{model}"
        local_root = root
    files = sorted(p.name for p in local_root.iterdir() if p.is_file())[:12] if local_root.exists() else []
    return {
        "model": model,
        "root_url": base_url,
        "model_artifacts_url": base_url,
        "train_history_url": f"{base_url}/policy_train_history.jsonl",
        "policy_history_url": f"{base_url}/policy_evaluate_history.jsonl",
        "kpi_summary_url": f"{base_url}/kpi_summary.json",
        "training_summary_url": f"{base_url}/training_summary.json",
        "existing_files": files,
    }


def _create_rl_train_job(
    cfg: Dict[str, Any],
    baselines: Optional[List[Dict[str, Any]]] = None,
    source: str = "rl-panel",
    approval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    real_cfg = {**cfg, "source": source, "approval": approval}
    job = TRAINING_MANAGER.start(real_cfg)
    job.update(
        {
            "config": real_cfg,
            "objective": str(cfg.get("objective") or "multi_objective"),
            "baselines": baselines or _RL_TRAIN_BASELINES,
            "source": source,
            "approval": approval,
            "notes": "Backend training consumes the chronological train split without rendering; evaluation is separate.",
        }
    )
    job_id = str(job["job_id"])
    _RL_TRAIN_JOBS[job_id] = job
    _RL_TRAIN_STATUS[job_id] = job
    return job


def _mobile_train_request_payload(request: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(request.get("job_id") or "").strip()
    job = _RL_TRAIN_JOBS.get(job_id) if job_id else None
    try:
        training_status = TRAINING_MANAGER.status(job_id) if job_id else None
    except KeyError:
        training_status = None
    return {
        **request,
        "job": job,
        "training_status": training_status,
        "human_gate": {
            "required": True,
            "approved": request.get("status") == "approved",
            "can_mobile_bypass": False,
        },
    }


@app.post("/api/rl/train/requests", tags=["rl-panel"])
async def rl_mobile_train_request_create(
    payload: Dict[str, Any] = Body(default={"config": {}})
) -> JSONResponse:
    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    request_id = "rl-request-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    request = {
        "request_id": request_id,
        "status": "pending_desktop_confirmation",
        "source": str(payload.get("source") or "dt_mobile_app"),
        "requested_by": str(payload.get("requested_by") or "mobile_operator"),
        "config": cfg,
        "scenario_snapshot": payload.get("scenario_snapshot") or {},
        "policy_context": payload.get("policy_context") or {},
        "created_at": now,
        "updated_at": now,
        "approved_at": None,
        "approved_by": None,
        "rejected_at": None,
        "rejected_by": None,
        "rejection_reason": None,
        "job_id": None,
    }
    _RL_MOBILE_TRAIN_REQUESTS[request_id] = request
    return JSONResponse(_mobile_train_request_payload(request), status_code=202)


@app.get("/api/rl/train/requests", tags=["rl-panel"])
async def rl_mobile_train_request_list(
    limit: int = Query(20, ge=1, le=100),
) -> JSONResponse:
    items = sorted(
        _RL_MOBILE_TRAIN_REQUESTS.values(),
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )[:limit]
    return JSONResponse(
        {
            "items": [_mobile_train_request_payload(item) for item in items],
            "count": len(items),
            "pending_count": sum(
                1 for item in _RL_MOBILE_TRAIN_REQUESTS.values()
                if item.get("status") == "pending_desktop_confirmation"
            ),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )


@app.get("/api/rl/train/requests/{request_id}", tags=["rl-panel"])
async def rl_mobile_train_request_detail(request_id: str) -> JSONResponse:
    request = _RL_MOBILE_TRAIN_REQUESTS.get(request_id)
    if not request:
        raise HTTPException(status_code=404, detail=f"未知移动端训练申请：{request_id}")
    return JSONResponse(_mobile_train_request_payload(request))


@app.post("/api/rl/train/requests/{request_id}/approve", tags=["rl-panel"])
async def rl_mobile_train_request_approve(
    request_id: str,
    payload: Dict[str, Any] = Body(default={"operator": "desktop_operator"}),
) -> JSONResponse:
    request = _RL_MOBILE_TRAIN_REQUESTS.get(request_id)
    if not request:
        raise HTTPException(status_code=404, detail=f"未知移动端训练申请：{request_id}")
    if request.get("status") == "rejected":
        raise HTTPException(status_code=409, detail="该训练申请已被拒绝，不能再次批准")
    if request.get("job_id"):
        return JSONResponse(_mobile_train_request_payload(request))

    operator = str(payload.get("operator") or "desktop_operator")
    approved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    approval = {
        "channel": "desktop_rl_panel",
        "operator": operator,
        "approved_at": approved_at,
        "request_id": request_id,
        "human_confirmed": True,
    }
    job = _create_rl_train_job(
        request.get("config") or {},
        source="mobile_request_after_desktop_approval",
        approval=approval,
    )
    request.update(
        {
            "status": "approved",
            "approved_at": approved_at,
            "approved_by": operator,
            "updated_at": approved_at,
            "job_id": job["job_id"],
        }
    )
    return JSONResponse(_mobile_train_request_payload(request))


@app.post("/api/rl/train/requests/{request_id}/reject", tags=["rl-panel"])
async def rl_mobile_train_request_reject(
    request_id: str,
    payload: Dict[str, Any] = Body(default={"operator": "desktop_operator"}),
) -> JSONResponse:
    request = _RL_MOBILE_TRAIN_REQUESTS.get(request_id)
    if not request:
        raise HTTPException(status_code=404, detail=f"未知移动端训练申请：{request_id}")
    if request.get("job_id"):
        raise HTTPException(status_code=409, detail="训练已获批启动，不能再拒绝")
    rejected_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    request.update(
        {
            "status": "rejected",
            "rejected_at": rejected_at,
            "rejected_by": str(payload.get("operator") or "desktop_operator"),
            "rejection_reason": str(payload.get("reason") or "电脑端人工拒绝"),
            "updated_at": rejected_at,
        }
    )
    return JSONResponse(_mobile_train_request_payload(request))


@app.post("/api/rl/train/start", tags=["rl-panel"])
async def rl_train_start(
    payload: Dict[str, Any] = Body(default={"config": {}, "baselines": []})
) -> JSONResponse:
    cfg = payload.get("config") or {}
    try:
        job = _create_rl_train_job(
            cfg,
            baselines=payload.get("baselines") or _RL_TRAIN_BASELINES,
            source=str(payload.get("source") or "rl-panel"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=429 if "capacity reached" in str(exc) else 422, detail=str(exc)) from exc
    return JSONResponse(
        {
            **job,
            "connector": {
                "mode": "real_backend_training_job",
                "baseline_count": len(_RL_TRAIN_BASELINES),
                "progress_owner": "stable_baselines3_callback",
                "render_during_training": False,
            },
        }, status_code=202
    )


@app.post("/api/rl/train/status", tags=["rl-panel"])
async def rl_train_status_update(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    return JSONResponse(
        {
            "ok": False,
            "detail": "training status is backend-owned and read-only; use GET /api/rl/train/status",
        },
        status_code=409,
    )


@app.get("/api/rl/train/status", tags=["rl-panel"])
async def rl_train_status(job_id: Optional[str] = Query(None)) -> JSONResponse:
    try:
        status = TRAINING_MANAGER.status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未知训练任务：{job_id}") from exc
    return JSONResponse(_status_response(status))


@app.get("/api/rl/train/metrics", tags=["rl-panel"])
async def rl_train_metrics(job_id: Optional[str] = Query(None)) -> JSONResponse:
    try:
        status = TRAINING_MANAGER.status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未知训练任务：{job_id}") from exc
    return JSONResponse(
        {
            "ok": True,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "job_id": status.get("job_id"),
            "status": status.get("status"),
            "metrics": status.get("metrics", {}),
            "summary": status.get("summary"),
        }
    )


# =================================================
# 仅在 PORT_DT_ENABLE_LEGACY_CLOSEDLOOP=1 时挂载的旧闭环模拟器适配器
# =================================================
async def rl_dispatch(
    payload: Dict[str, Any] = Body(
        default={
            "strategy_id": "qc_idle_midday",
            "operator": "system",
            "dry_run": True,
            "enforce_guardrails": True,
            "guardrail_min_peak_kw": 1.0,
            "notes": "旧闭环模拟器的兼容记录；不属于默认南向执行链路",
        }
    )
) -> JSONResponse:
    try:
        strategy = payload.get("strategy")
        strategy_id = payload.get("strategy_id")
        operator = str(payload.get("operator", "system"))
        dry_run = bool(payload.get("dry_run", True))
        enforce_guardrails = bool(payload.get("enforce_guardrails", True))
        guardrail_min_peak_kw = float(payload.get("guardrail_min_peak_kw", 1.0))
        notes = payload.get("notes")

        if not strategy and strategy_id:
            lst = di.rlpanel.list_strategies(horizon_min=360, step_min=5, max_items=50) or {}
            for item in lst.get("strategies", []):
                if item.get("id") == strategy_id:
                    strategy = item
                    break
            if not strategy:
                raise HTTPException(status_code=400, detail=f"未找到策略 ID: {strategy_id}")

        if not strategy:
            raise HTTPException(status_code=400, detail="请提供 strategy_id 或 strategy。")

        result = di.dispatch.dispatch(
            strategy=strategy,
            operator=operator,
            dry_run=dry_run,
            enforce_guardrails=enforce_guardrails,
            guardrail_min_peak_kw=guardrail_min_peak_kw,
            notes=notes,
        )
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"dispatch 失败: {e}")


async def rl_dispatch_history(
    limit: int = Query(50, ge=1, le=200, description="返回最近的记录条数")
) -> JSONResponse:
    try:
        data = di.dispatch.list_history(limit=limit)
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"dispatch.history 失败: {e}")


async def rl_dispatch_cancel(
    payload: Dict[str, Any] = Body(
        default={"job_id": "xxxx-uuid", "operator": "system"}
    )
) -> JSONResponse:
    try:
        job_id = str(payload.get("job_id") or "")
        operator = str(payload.get("operator", "system"))
        if not job_id:
            raise HTTPException(status_code=400, detail="缺少 job_id。")
        data = di.dispatch.cancel(job_id=job_id, operator=operator)
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"dispatch.cancel 失败: {e}")


# =================================================
# 可解释特征（SHAP-like）接口 —— 保持
# =================================================
@app.post("/api/rl/explain", tags=["explain"])
async def rl_explain(
    payload: Dict[str, Any] = Body(
        default={
            "strategy_id": "<registered-job-id>"
        }
    )
) -> JSONResponse:
    try:
        job_id = str(payload.get("job_id") or payload.get("strategy_id") or "").strip()
        if not job_id:
            raise HTTPException(status_code=422, detail="job_id or registered strategy_id is required")
        registry = TRAINING_MANAGER.model_registry()
        record = registry.get(job_id)
        benchmark = TRAINING_MANAGER.benchmark_summary(record.get("dataset_id"))
        readiness = registry.readiness(job_id, benchmark)
        metrics = (record.get("evaluation") or {}).get("metrics") or {}
        reasons = [
            f"algorithm={record.get('algorithm')}",
            f"dataset={record.get('dataset_id')}",
            f"dataset_sha256={record.get('dataset_sha256')}",
            f"artifact_verified={(record.get('artifact') or {}).get('verified')}",
            f"heldout_episodes={(record.get('evaluation') or {}).get('episodes') or 0}",
        ]
        return JSONResponse({
            "strategy_id": job_id,
            "features": [],
            "reasons": reasons,
            "metrics": metrics,
            "readiness": readiness,
            "meta": {
                "source": "model_card_registry_and_heldout_evaluation",
                "feature_attribution_available": False,
                "reason": "未训练独立特征归因器，因此不生成 SHAP-like 数值。",
            },
        })
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown registered model: {job_id}") from exc


@app.post("/api/rl/explain_many", tags=["explain"])
async def rl_explain_many(
    payload: Dict[str, Any] = Body(
        default={
            "strategies": [
                {"strategy_id": "<registered-job-id>"}
            ]
        }
    )
) -> JSONResponse:
    items = payload.get("strategies") or []
    results = []
    for item in items[:50]:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id") or item.get("strategy_id") or "").strip()
        if not job_id:
            continue
        try:
            record = TRAINING_MANAGER.model_registry().get(job_id)
        except KeyError:
            continue
        results.append({
            "strategy_id": job_id,
            "algorithm": record.get("algorithm"),
            "dataset_id": record.get("dataset_id"),
            "artifact_verified": (record.get("artifact") or {}).get("verified"),
            "evaluation": record.get("evaluation"),
            "feature_attribution_available": False,
        })
    if not results:
        raise HTTPException(status_code=422, detail="no registered model ids were resolved")
    return JSONResponse({"count": len(results), "items": results, "source": "model_registry", "generated_values": False})

# 放在 RL 相关 API 段落之前/之后均可

@app.get("/api/rl/rollout/status", tags=["rl"])
async def rl_rollout_status():
    """
    回兼容老前端/脚本的“滚动发布状态”查询。
    当前项目没有真实 rollout 管理器，先返回静态/兜底状态。
    如需接入真实服务，可在此处对接你的状态源。
    """
    registry = TRAINING_MANAGER.model_registry().list()
    return {
        "rollout": "site_deployment_not_configured",
        "aliases": registry.get("aliases") or {},
        "registered_model_count": registry.get("count", 0),
        "production_deployment_approved": False,
        "source": "model_registry_aliases",
        "message": "候选/冠军/回滚别名不等于现场流量切换；仓库默认不执行生产 rollout。",
        "updated_at": registry.get("updated_at"),
    }


# =================================================
# 执行与闭环（Closed Loop）接口组 —— 保持
# =================================================

def _resolve_strategy_by_id(strategy_id: str) -> Dict[str, Any]:
    """从 rlpanel 列表中解析策略对象；若未找到则抛 400。"""
    cat = di.rlpanel.list_strategies(horizon_min=360, step_min=5, max_items=100) or {}
    for item in cat.get("strategies", []):
        if item.get("id") == strategy_id:
            return item
    raise HTTPException(status_code=400, detail=f"未找到策略 ID: {strategy_id}")


async def exec_submit(
    payload: Dict[str, Any] = Body(
        default={
            # 二选一：提供 strategy_id 或完整 strategy 对象
            "strategy_id": "qc_idle_midday",
            # "strategy": {...},
            "operator": "system",
            "mode": "auto",         # auto | manual
            "dry_run": False,
            "notes": "首页一键下发/或提交审批"
        }
    )
) -> JSONResponse:
    """
    创建执行工单：
      - 保存“预测快照”（baseline/simulated 聚合曲线 + 汇总）；
      - mode='auto'：自动批准并下发；mode='manual'：待审批；
      - 守护栏校验由 di.dispatch.validate_strategy 负责。
    """
    try:
        strategy = payload.get("strategy")
        strategy_id = payload.get("strategy_id")
        operator = str(payload.get("operator", "system"))
        mode = str(payload.get("mode", "auto")).lower()
        dry_run = bool(payload.get("dry_run", False))
        notes = payload.get("notes")

        if not strategy and strategy_id:
            strategy = _resolve_strategy_by_id(strategy_id)

        if not strategy:
            raise HTTPException(status_code=400, detail="请提供 strategy_id 或 strategy。")

        data = di.closedloop.submit(strategy=strategy, operator=operator, mode=mode, dry_run=dry_run, notes=notes)
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"exec.submit 失败: {e}")


async def exec_approve(
    payload: Dict[str, Any] = Body(default={"job_id": "uuid", "operator": "auditor"})
) -> JSONResponse:
    """审批通过并下发。"""
    try:
        job_id = str(payload.get("job_id") or "")
        operator = str(payload.get("operator", "system"))
        if not job_id:
            raise HTTPException(status_code=400, detail="缺少 job_id。")
        data = di.closedloop.approve(job_id=job_id, operator=operator)
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"exec.approve 失败: {e}")


async def exec_get(job_id: str) -> JSONResponse:
    try:
        return JSONResponse(di.closedloop.get(job_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"exec.get 失败: {e}")


async def exec_list(
    limit: int = Query(50, ge=1, le=200, description="返回最近的工单条数")
) -> JSONResponse:
    try:
        return JSONResponse(di.closedloop.list(limit=limit))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"exec.list 失败: {e}")


async def exec_abtest(job_id: str) -> JSONResponse:
    """
    A/B 对照：
      - predicted：提交时保存的 baseline/simulated 与 ΔkWh 预测
      - actual：仅来自 telemetry.collect_ab_observations 现场适配器
      - 缺少实测时 available=false，禁止在线学习
    """
    try:
        return JSONResponse(di.closedloop.ab_compare(job_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"exec.abtest 失败: {e}")


async def exec_learn(
    payload: Dict[str, Any] = Body(default={"job_id": "uuid", "alpha": 0.3})
) -> JSONResponse:
    """
    在线学习（EMA）：
      - 基于 A/B 对照，更新策略的 ema_delta / ema_bias / ema_abs_err / n。
    """
    try:
        job_id = str(payload.get("job_id") or "")
        alpha = float(payload.get("alpha", 0.3))
        if not job_id:
            raise HTTPException(status_code=400, detail="缺少 job_id。")
        return JSONResponse(di.closedloop.learn(job_id=job_id, alpha=alpha))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"exec.learn 失败: {e}")


async def exec_model(strategy_id: str) -> JSONResponse:
    """查询某策略的在线学习画像。"""
    try:
        return JSONResponse(di.closedloop.get_model(strategy_id=strategy_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"exec.model 失败: {e}")


if _ENABLE_LEGACY_CLOSEDLOOP:
    app.add_api_route("/api/legacy/rl/dispatch", rl_dispatch, methods=["POST"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/rl/dispatch/history", rl_dispatch_history, methods=["GET"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/rl/dispatch/cancel", rl_dispatch_cancel, methods=["POST"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/exec/submit", exec_submit, methods=["POST"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/exec/approve", exec_approve, methods=["POST"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/exec/get/{job_id}", exec_get, methods=["GET"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/exec/list", exec_list, methods=["GET"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/exec/abtest/{job_id}", exec_abtest, methods=["GET"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/exec/learn", exec_learn, methods=["POST"], tags=["legacy-closedloop-simulator"])
    app.add_api_route("/api/legacy/exec/model/{strategy_id}", exec_model, methods=["GET"], tags=["legacy-closedloop-simulator"])

# =================================================
# ⭐⭐ 管理驾驶舱（Exec Cockpit）汇总接口 —— 新增
# =================================================
@app.get("/api/exec_cockpit/summary", tags=["exec_cockpit"])
async def exec_cockpit_summary() -> JSONResponse:
    """
    董事会 / CEO 驾驶舱：一屏聚合关键 KPI（节省、电碳、风险、自动化、可信度）。

    返回字段将由前端 index.html 的“管理驾驶舱”模块消费，字段含义：
      - yearly_saving_cny: 过去 12 个月预估节省电费（元）
      - yearly_co2_ton: 过去 12 个月减排 CO₂（吨 CO2e）
      - peak_risk_30d: 未来 30 天超合同需量的概率（0~1）
      - auto_cover_pct: 自动闭环覆盖的关键负荷比例（0~1）
      - ai_grade: AI 策略可信度等级（A/B/C 等）
      - ai_grade_reason: 等级说明（OPE / 守护栏 / 实验等来源）
      - status_level: 状态等级（ok / warn / bad）
      - status_label: 状态标签（短句，显示在 badge 上）
      - status_detail: 状态详情（小字说明）
    """
    try:
        # 延迟导入：后面你会在 app/services/exec_cockpit/service.py 里实现 get_summary(di)
        from app.services.exec_cockpit.service import get_summary  # type: ignore

        data = get_summary(di)
    except Exception as e:
        data = {
            "available": False,
            "reason": f"exec cockpit service unavailable: {type(e).__name__}",
            "yearly_saving_cny": None,
            "yearly_co2_ton": None,
            "peak_risk_30d": None,
            "auto_cover_pct": None,
            "ai_grade": "N/A",
            "status_level": "unknown",
            "_source": "exec_cockpit.unavailable",
        }
    return JSONResponse(data)


# =================================================
# ⭐⭐ 平台地图（Platform Map）接口组 —— 新增
# =================================================
@app.get("/api/platform_map/graph", tags=["platform_map"])
async def platform_map_graph() -> JSONResponse:
    """
    平台地图：返回分层结构（设备 / 数据与孪生 / RL 与协同 / 应用与运营）。

    主要用于首页“平台地图”板块，也可以给后续可视化或文档工具复用。
    """
    try:
        # 延迟导入：后面你可以在 app.services.platform_map.service 里实现 get_graph(di)
        from app.services.platform_map.service import get_graph  # type: ignore

        data = get_graph(di)
    except Exception as e:
        data = {
            "available": False,
            "reason": f"platform map service unavailable: {type(e).__name__}",
            "layers": [],
            "_source": "platform_map.unavailable",
        }
    return JSONResponse(data)
# =================================================
# ⭐⭐ Dev / 生态视角：开放 API & App Center 接口组 —— 新增
# =================================================
@app.get("/api/app_center/overview", tags=["app_center"])
async def app_center_overview() -> JSONResponse:
    """Expose the routes actually registered by the running FastAPI app."""
    rest_apis = []
    ui_apps = []
    for route in app.routes:
        path = str(getattr(route, "path", ""))
        methods = sorted(method for method in (getattr(route, "methods", set()) or set()) if method not in {"HEAD", "OPTIONS"})
        if path.startswith("/api/") and methods:
            rest_apis.append({
                "path": path,
                "methods": methods,
                "label": str(getattr(route, "summary", None) or getattr(route, "name", path)),
            })
        elif path in {"/rl-panel", "/integration-hub", "/ops-copilot"}:
            ui_apps.append({"id": path.strip("/").replace("-", "_"), "name": path, "path": path, "category": "ui"})
    rest_apis.sort(key=lambda item: (item["path"], item["methods"]))
    return JSONResponse({
        "platform_support": {
            "rest_apis": rest_apis,
            "webhooks": [],
            "sdk": {"notebooks": [], "languages": ["python", "javascript"]},
        },
        "apps": ui_apps,
        "counts": {"rest_routes": len(rest_apis), "ui_apps": len(ui_apps)},
        "_source": "fastapi_runtime_route_registry",
    })

# =================================================
# ⭐⭐ ESG / 合规模块：电碳 & 合规驾驶舱 —— 新增（严格后端版）
# =================================================
@app.get("/api/esg/summary", tags=["esg"])
async def esg_summary() -> JSONResponse:
    """
    ESG / 合规驾驶舱后端接口：
    - 只从 service 取数据（app.services.esg.service.get_summary(di)）
    - 不返回任何前端示例或服务器兜底数据
    - 如果 service 未实现或发生异常，直接返回 500，便于线上暴露真实问题
    """
    try:
        # 你在 app/services/esg/service.py 里实现的真实/模拟数据汇总逻辑
        from app.services.esg.service import get_summary  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"esg.service 导入失败: {e}")

    try:
        data = get_summary(di)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"esg.summary 执行失败: {e}")

    return JSONResponse(data)

# =================================================
# ⭐⭐ 合规报表（Compliance）—— 端到端数据接口（对接 ESG 服务）
# =================================================
@app.get("/api/compliance/catalog", tags=["compliance"])
async def compliance_catalog() -> JSONResponse:
    """
    港口清单（代码/名称/地区）。
    数据来自 app/services/esg/data/ports_catalog.json；若不存在则显式返回不可用，不生成演示清单。
    """
    try:
        from app.services.esg.service import get_ports_catalog  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"esg.service 导入失败: {e}")
    try:
        return JSONResponse(get_ports_catalog())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compliance.catalog 失败: {e}")


@app.get("/api/compliance/timeseries", tags=["compliance"])
async def compliance_timeseries(
    port: str = Query(..., min_length=3, description="港口代码，如 CNSHA/SGSIN/NLRTM/USLAXLGB"),
    year: Optional[int] = Query(None, ge=2000, le=2100, description="年份，默认当前年"),
    granularity: str = Query("month", pattern="^(month)$", description="时间粒度：仅支持 month"),
) -> JSONResponse:
    try:
        from app.services.esg.service import get_compliance_timeseries  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"esg.service 导入失败: {e}")
    try:
        y = year or datetime.now().year
        data = get_compliance_timeseries(port_code=port, year=int(y), granularity=granularity)
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compliance.timeseries 失败: {e}")


@app.get("/api/compliance/breakdown", tags=["compliance"])
async def compliance_breakdown(
    port: str = Query(..., min_length=3, description="港口代码"),
    year: Optional[int] = Query(None, ge=2000, le=2100, description="年份，默认当前年"),
    month: int = Query(..., ge=1, le=12, description="月份 1-12"),
) -> JSONResponse:
    try:
        from app.services.esg.service import get_compliance_breakdown  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"esg.service 导入失败: {e}")
    try:
        y = year or datetime.now().year
        data = get_compliance_breakdown(port_code=port, year=int(y), month=int(month))
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compliance.breakdown 失败: {e}")

# =================================================
# ⭐⭐ 可信 AI 等级（Trust Badge）接口 —— 新增
# =================================================
@app.get("/api/ai/trust_badge", tags=["ai_trust"])
async def ai_trust_badge() -> JSONResponse:
    """
    可信 AI 等级徽章：
    - 只从 service 取数据（app.services.ai_trust.service.get_badge(di)）
    - 不返回任何示例/兜底
    - service 未实现或异常 => 直接 500（便于暴露真实问题）
    期望返回结构：
    {
      "grade": "A",                       # 可信等级（A/A+/B...）
      "ope_pass": true,                   # OPE 是否通过
      "guardrail": {"total": 18, "pending": 0},     # 守护栏总数 / 待调优
      "experiments": {"running": 3, "completed": 5},# 实验运行中 / 已完成
      "causal_effect": -0.08              # 对能耗 ΔkWh 的平均因果效应（-0.08 = -8%）
    }
    """
    try:
        # 你下一步会实现：app/services/ai_trust/service.py -> get_badge(di)
        from app.services.ai_trust.service import get_badge  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ai_trust.service 导入失败: {e}")

    try:
        data = get_badge(di)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ai_trust.badge 执行失败: {e}")

    return JSONResponse(data)
# =================================================
# ⭐⭐ 案例回放（Story Mode）接口 —— 新增（严格后端）
# =================================================
from fastapi import Query

@app.get("/api/story/summary", tags=["story"])
async def story_summary(hour: int = Query(0, ge=-24, le=24)) -> JSONResponse:
    """
    从最新真实留出集评测轨迹取一帧；无评测时明确不可用。
    """
    status = TRAINING_MANAGER.status()
    evaluation = status.get("evaluation") or {}
    job_id = str(status.get("job_id") or "")
    trace_path = TRAINING_MANAGER.run_root / job_id / "evaluation_trajectory.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if job_id and trace_path.exists() else {}
    frames = trace.get("frames") or []
    if not frames:
        raise HTTPException(status_code=503, detail="尚无训练完成后的留出集评测轨迹")
    index = round((hour + 24) / 48 * (len(frames) - 1))
    frame = frames[max(0, min(len(frames) - 1, index))]
    return JSONResponse({
        "available": True,
        "hour": hour,
        "events": [],
        "baseline": {"peak_kw": frame.get("baseline_kw"), "bill_cny": None},
        "with_rl": {
            "peak_kw": frame.get("net_load_kw"),
            "bill_cny": None,
            "co2_ton": (float(frame["carbon_kg"]) / 1000.0) if frame.get("carbon_kg") is not None else None,
        },
        "frame": frame,
        "job_id": job_id,
        "algorithm": evaluation.get("algorithm"),
        "dataset_id": evaluation.get("dataset_id"),
        "dataset_sha256": evaluation.get("dataset_sha256"),
        "_source": "rl_heldout_evaluation_trajectory",
    })


@app.post("/api/story/play", tags=["story"])
async def story_play(payload: dict | None = None) -> JSONResponse:
    """
    触发“剧情式 Demo 播放”：
    仅从 service 取执行结果（app.services.story.service.play(di, mode)）
    无兜底。建议 service 内部去同步 PortViz/MAS/聚合等，前端只接收 ack。

    请求体：{"mode": "demo"}（可扩展）
    返回结构（由 service 决定，例如）：{"ok": true, "mode": "demo", "ts": "..."}
    """
    raise HTTPException(
        status_code=501,
        detail="服务端回放编排器未配置；可直接读取 /api/story/summary 的留出集轨迹帧。",
    )

# =================================================
# ⭐⭐ 合规报表（Compliance）接口组 —— 保持
# =================================================
@app.get("/api/compliance/monthly", tags=["compliance"])
async def compliance_monthly(
    month: Optional[str] = Query(None, description='月度起点（格式 "YYYY-MM"，为空取当前月）'),
    teu: int = Query(12000, ge=1),
    granularity: str = Query("all", pattern="^(all|by_asset|by_process|by_group|by_berth)$"),
    # 因子覆盖（可选）
    grid_g_per_kwh: Optional[float] = Query(None, description="电网/岸电排放因子 gCO2e/kWh"),
    diesel_kg_per_liter: Optional[float] = Query(None, description="柴油排放因子 kgCO2e/L"),
    selfgen_kg_per_kwh: Optional[float] = Query(None, description="自发电排放因子 kgCO2e/kWh"),
    selfgen_share: Optional[float] = Query(None, ge=0.0, le=1.0, description="自发电占比 0~1"),
    diesel_model: str = Query("rule_of_thumb", pattern="^(rule_of_thumb|none)$"),
) -> JSONResponse:
    try:
        ef = di.factors(
            grid_g_per_kwh=grid_g_per_kwh,
            diesel_kg_per_liter=diesel_kg_per_liter,
            selfgen_kg_per_kwh=selfgen_kg_per_kwh,
            selfgen_share=selfgen_share,
        )
        data = di.compliance.monthly_report(
            month_yyyy_mm=month,
            teu=teu,
            granularity=granularity,
            factors=ef,
            diesel_model=diesel_model,
        )
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compliance.monthly 失败: {e}")


@app.get("/api/compliance/quarterly", tags=["compliance"])
async def compliance_quarterly(
    start: Optional[str] = Query(None, description='季度起点（格式 "YYYY-MM"，为空取当前月为起点）'),
    teu: int = Query(36000, ge=1),
    granularity: str = Query("all", pattern="^(all|by_asset|by_process|by_group|by_berth)$"),
    grid_g_per_kwh: Optional[float] = Query(None),
    diesel_kg_per_liter: Optional[float] = Query(None),
    selfgen_kg_per_kwh: Optional[float] = Query(None),
    selfgen_share: Optional[float] = Query(None, ge=0.0, le=1.0),
    diesel_model: str = Query("rule_of_thumb", pattern="^(rule_of_thumb|none)$"),
) -> JSONResponse:
    try:
        ef = di.factors(
            grid_g_per_kwh=grid_g_per_kwh,
            diesel_kg_per_liter=diesel_kg_per_liter,
            selfgen_kg_per_kwh=selfgen_kg_per_kwh,
            selfgen_share=selfgen_share,
        )
        data = di.compliance.quarterly_report(
            start_month_yyyy_mm=start,
            teu=teu,
            granularity=granularity,
            factors=ef,
            diesel_model=diesel_model,
        )
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compliance.quarterly 失败: {e}")


@app.post("/api/compliance/make", tags=["compliance"])
async def compliance_make(
    payload: Dict[str, Any] = Body(
        ...,
        examples=[{
            "config": {
                "period": "month",
                "start_month": "2025-09",
                "granularity": "all",
                "teu": 12000
            },
            "factors": {
                "grid_g_per_kwh": 120.0,
                "diesel_kg_per_liter": 2.68,
                "selfgen_kg_per_kwh": 0.70,
                "selfgen_share": 0.0
            },
            "diesel_model": "rule_of_thumb"
        }],
        description="自定义配置：period/start_month/granularity/TEU/因子/自发电占比等",
    ),
) -> JSONResponse:
    try:
        cfg = payload.get("config") or {}
        f = payload.get("factors") or {}
        ef = di.factors(
            grid_g_per_kwh=f.get("grid_g_per_kwh"),
            diesel_kg_per_liter=f.get("diesel_kg_per_liter"),
            selfgen_kg_per_kwh=f.get("selfgen_kg_per_kwh"),
            selfgen_share=f.get("selfgen_share"),
        )
        diesel_model = payload.get("diesel_model", "rule_of_thumb")

        config = {
            "period": cfg.get("period", "month"),
            "start_month": cfg.get("start_month"),
            "granularity": cfg.get("granularity", "all"),
            "teu": int(cfg.get("teu", 12000)),
        }
        return JSONResponse(di.compliance.make_report(config=config, factors=ef, diesel_model=diesel_model))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compliance.make 失败: {e}")


# -------------------------------------------------
# 模块主入口
# -------------------------------------------------
def _detect_port() -> int:
    try:
        port = int(os.getenv("PORT_DT_SERVER_PORT", "8000"))
    except ValueError:
        return 8000
    return port if 1024 <= port <= 65535 else 8000


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=_detect_port(), reload=False, log_level="info")
# === [AGV Charge · RL Router Mount | Module A] ==============================
# 作用：把 AGV 充/换电 RL 接口（/api/rl/* 与 /api/exec/*）挂载到现有后端。
# 依赖：app/services/rl_model/agv_charge/api.py 中的 get_router()。
# 前端：index.html 的“RL 策略面板 · 集成”按钮会直接调用这些接口。
# 安全：若非 FastAPI 或挂载失败，不会中断主进程，只打印警告。
_ENABLE_LEGACY_RL = False  # unaudited legacy artifact routes are not distributed
get_router = None
if _ENABLE_LEGACY_RL:
    try:
        from app.services.rl_model.agv_charge.api import get_router  # 提供 FastAPI APIRouter
    except Exception as _e:
        print("[agv_charge] get_router import failed:", _e)

try:
    # 仅当全局存在 app 且具备 include_router（FastAPI）时执行
    if get_router is not None and "app" in globals() and hasattr(globals()["app"], "include_router"):
        _router = get_router()
        if _router is not None:
            globals()["app"].include_router(_router, prefix="")
            print("[agv_charge] RL router mounted at /api/*")
        else:
            print("[agv_charge] get_router() returned None; skip mount.")
    elif _ENABLE_LEGACY_RL:
        # 非 FastAPI 框架或未找到 app；保持静默，不中断主进程
        if "app" not in globals():
            print("[agv_charge] no global 'app' found in server.py; skip mount.")
        elif not hasattr(globals()["app"], "include_router"):
            print("[agv_charge] current 'app' has no include_router (not FastAPI?); skip mount.")
except Exception as e:
    try:
        print("[agv_charge] RL router mount failed:", e)
    except Exception:
        pass
# ============================================================================

# --- [auto-injected] RL artifacts & metrics router ---
if _ENABLE_LEGACY_RL:
    try:
        from app.services.rl_suite.rl_artifacts import router as rl_artifacts_router  # noqa
        app.include_router(rl_artifacts_router)
    except Exception as _e:
        print("[warn] unable to include rl_artifacts router:", _e)
# --- [end inject] ---

# --- [legacy] RL admin/router (models, csv, upload, short-train) ---
if _ENABLE_LEGACY_RL:
    try:
        from app.services.rl_suite.rl_admin import router as rl_admin_router  # noqa
        app.include_router(rl_admin_router)
    except Exception as _e:
        print("[warn] unable to include rl_admin router:", _e)
# --- [end inject] ---

# ==== [Module A · RL artifacts & metrics endpoints · BEGIN] ====================
from pathlib import Path

from fastapi.staticfiles import StaticFiles
import json, csv, os
# ===== 静态资源挂载（/static -> app/static） =====
_static_dir = Path(__file__).resolve().parent / "static"
try:
    _static_dir.mkdir(parents=True, exist_ok=True)  # 若不存在则创建，避免启动时报错
except Exception:
    pass
app.mount(
    "/static",
    StaticFiles(directory=str(_static_dir), html=False),
    name="static",
)

# 基础路径（定位到 app/services/rl_model/<model>/artifacts 和 data）
_BASE = Path(__file__).resolve().parent
_RL_MODEL = os.getenv("RL_MODEL", "agv_charge")
_ART_DIR = _BASE / "services" / "rl_model" / _RL_MODEL / "artifacts"
_DATA_DIR = _BASE / "services" / "rl_model" / _RL_MODEL / "data"
# Yard Lighting（B 模块）固定目录
_YL_ART_DIR  = _BASE / "services" / "rl_model" / "yard_lighting" / "artifacts"
_YL_DATA_DIR = _BASE / "services" / "rl_model" / "yard_lighting" / "data"

rl_router = APIRouter(prefix="/api/rl", tags=["rl"])

def _read_jsonl(p: Path):
    recs=[]
    if p.exists():
        # 以 utf-8 读文本，逐行 json 解析
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s=line.strip()
            if not s:
                continue
            try:
                o=json.loads(s)
            except Exception:
                continue
            # 统一字段名，匹配前端 index.html 的兼容读取逻辑（reward/peak/len/entropy）
            if "returns" in o and "reward" not in o:
                o["reward"] = o["returns"]
            if "episode_reward" in o and "reward" not in o:
                o["reward"] = o["episode_reward"]
            if "peak_kW_delta" in o and "peak_reduction_kW" not in o:
                o["peak_reduction_kW"] = o["peak_kW_delta"]
            if "len" in o and "episode_len" not in o:
                o["episode_len"] = o["len"]
            if "steps" in o and "episode_len" not in o:
                o["episode_len"] = o["steps"]
            if "latency_min" in o and "episode_len" not in o:
                o["episode_len"] = o["latency_min"]
            if "policy_entropy" in o and "entropy" not in o:
                o["entropy"] = o["policy_entropy"]
            recs.append(o)
    return recs

@rl_router.get("/metrics/history", response_class=JSONResponse)
def rl_metrics_history():
    """
    返回训练/评估历史（数组 JSON）。前端已写好兜底会优先尝试 JSONL，
    这个接口用于 JSONL 缺失或字段名不一致时的统一出口。
    """
    # 1) 最优先：artifacts/policy_evaluate_history.jsonl
    # 0) 允许使用 artifacts/offline_train.jsonl（如果存在）
    j = _read_jsonl(_ART_DIR / "offline_train.jsonl")
    if j:
        return j[-2000:]

    j = _read_jsonl(_ART_DIR / "policy_evaluate_history.jsonl")
    if j:
        return j[-2000:]   # 限制上限，避免前端加载过大

    # 2) 兜底：dispatch_history.jsonl 映射到相近字段
    dh = _read_jsonl(_ART_DIR / "dispatch_history.jsonl")
    if dh:
        out=[]
        for r in dh:
            o={}
            if "peak_reduction_kW" in r:
                o["peak_reduction_kW"] = r["peak_reduction_kW"]
            if "latency_min" in r:
                o["episode_len"] = r["latency_min"]
            # 如果有 delta_kWh 和 price，用 “-kWh*price” 近似为 reward 代理（节省=正向）
            try:
                dk = float(r.get("delta_kWh"))
                pr = float(r.get("price_yuan_per_kwh"))
                o["reward"] = - dk * pr
            except Exception:
                pass
            if o:
                out.append(o)
        if out:
            return out[-2000:]
    raise HTTPException(status_code=404, detail="no metrics available")
@rl_router.get("/yard_lighting/snapshot", response_class=JSONResponse)
def yl_snapshot():
    """
    B 模块（堆场照明）快照：
    - 优先读取 artifacts/offline_train.json（奖励分解完整）
    - 兜底读取 artifacts/offline_train.jsonl 的最后一行
    - 输出结构与前端 index.html 预期一致
    """
    # 1) 读取最新记录
    j = None
    p_json  = _YL_ART_DIR / "offline_train.json"
    p_jsonl = _YL_ART_DIR / "offline_train.jsonl"
    if p_json.exists():
        try:
            j = json.loads(p_json.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            j = None
    if j is None and p_jsonl.exists():
        recs = _read_jsonl(p_jsonl)  # 复用你已有的解析
        if recs:
            j = recs[-1]

    if not isinstance(j, dict):
        # 给出明确 404，前端会把 KPI 置为 "—"
        raise HTTPException(status_code=404, detail="yl snapshot not available")

    # 2) 抽字段（与前端 index.html 的 renderSnap() 对齐）
    rb = (j.get("rewards", {}).get("baseline", {})) or {}
    rp = (j.get("rewards", {}).get("policy",   {})) or {}
    met = (j.get("metrics", {}) or {})
    mw  = (j.get("metrics_window", {}) or {})

    # 3) 碳价：优先 data/config_limits.json；否则由 baseline 反推
    carbon_price = None
    try:
        cfg = json.loads((_YL_DATA_DIR / "config_limits.json").read_text(encoding="utf-8", errors="ignore"))
        carbon_price = cfg.get("price_config", {}).get("carbon_price_yuan_per_kg", None)
    except Exception:
        pass
    if carbon_price is None:
        try:
            fee = float(rb.get("carbon_fee_yuan", 0.0))
            kg  = float(rb.get("carbon_cost_kg", 0.0))
            carbon_price = fee / kg if kg > 0 else 0.0
        except Exception:
            carbon_price = 0.0

    out = {
        "economics": {
            "baseline": { "rewards": rb },
            "policy":   { "rewards": rp },
            "delta":    {
                "kWh_delta":          (met.get("delta_kWh")             or mw.get("delta_kWh_window")),
                "peak_reduction_kW":  (met.get("peak_reduction_kW")     or mw.get("peak_reduction_kW_window")),
            }
        },
        "metrics": { "under_lux_ratio_avg": met.get("under_lux") },
        "metrics_window": mw,
        "config": {
            "DT_MIN": j.get("DT_MIN", None),
            "price_config": { "carbon_price_yuan_per_kg": float(carbon_price or 0.0) }
        }
    }
    return JSONResponse(out)

@rl_router.get("/series/price_ef", response_class=JSONResponse)
def series_price_ef():
    """
    输出用于“电价 vs 电网排放因子”的时序数据（非图片）。
    读取 data/market_price.csv 与 data/grid_ef.csv，并按 timestamp 对齐。
    列名按 config.yaml: time_col=timestamp, price_col=price_yuan_per_kwh, ef_col=ef_kg_per_kwh。
    """
    time_col = "timestamp"
    price_col= "price_yuan_per_kwh"
    ef_col   = "ef_kg_per_kwh"

    price_map, ef_map = {}, {}

    # 读电价
    p_csv = _DATA_DIR / "market_price.csv"
    if p_csv.exists():
        with p_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ts = row.get(time_col) or row.get("ts") or row.get("time")
                if ts is None:
                    continue
                try:
                    price_map[ts] = float(row.get(price_col) or 0.0)
                except Exception:
                    continue

    # 读排放因子
    e_csv = _DATA_DIR / "grid_ef.csv"
    if e_csv.exists():
        with e_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ts = row.get(time_col) or row.get("ts") or row.get("time")
                if ts is None:
                    continue
                try:
                    ef_map[ts] = float(row.get(ef_col) or 0.0)
                except Exception:
                    continue

    # 对齐（取两者并集，按时间字符串排序）
    keys = sorted(set(price_map.keys()) | set(ef_map.keys()))
    ts   = keys
    price= [price_map.get(k) for k in keys]
    ef   = [ef_map.get(k)    for k in keys]

    if not ts:
        raise HTTPException(status_code=404, detail="no series data")

    return {"ts": ts, "price": price, "ef": ef, "units":{"price":"¥/kWh","ef":"kg/kWh"}}

@rl_router.get("/series/reward_costs", response_class=JSONResponse)
def series_reward_costs():
    """
    输出用于“训练奖励 / 成本分解”的时序数据（非图片）。
    主要来源 policy_evaluate_history.jsonl，字段名尽力兼容：
    reward / (cost_yuan | elec_cost_yuan | power_cost_yuan) / (kWh | energy_kWh) / (kgCO2e | carbon_kg)。
    """
    recs = _read_jsonl(_ART_DIR / "policy_evaluate_history.jsonl")
    if not recs:
        raise HTTPException(status_code=404, detail="no evaluate history")

    ts, reward, cost, kwh, kg = [], [], [], [], []
    for i, r in enumerate(recs):
        # 时间轴：如果没有时间戳，就用序号占位
        t = r.get("ts") or r.get("time") or r.get("timestamp") or i
        ts.append(t)
        # 奖励
        try:
            reward.append(float(r.get("reward")))
        except Exception:
            reward.append(None)
        # 成本（元）
        for key in ("cost_yuan", "elec_cost_yuan", "power_cost_yuan", "cost"):
            if key in r:
                try:
                    cost.append(float(r.get(key)))
                except Exception:
                    cost.append(None)
                break
        else:
            cost.append(None)
        # 电量 kWh
        for key in ("kWh","energy_kWh","elec_kWh","delta_kWh"):
            if key in r:
                try:
                    kwh.append(float(r.get(key)))
                except Exception:
                    kwh.append(None)
                break
        else:
            kwh.append(None)
        # 碳排 kgCO2e
        for key in ("kgCO2e","carbon_kg","emission_kg"):
            if key in r:
                try:
                    kg.append(float(r.get(key)))
                except Exception:
                    kg.append(None)
                break
        else:
            kg.append(None)

    return {"ts": ts, "reward": reward, "cost_yuan": cost, "kWh": kwh, "kgCO2e": kg}


def _legacy_rl_get(path: str):
    """Register legacy artifact routes only after an explicit operator opt-in."""
    if _ENABLE_LEGACY_RL:
        return app.get(path)

    def passthrough(endpoint):
        return endpoint

    return passthrough

# 静态挂载 artifacts（图片/JSONL）
try:
    if _ENABLE_LEGACY_RL and _ART_DIR.exists():
        app.mount("/api/rl/artifacts", StaticFiles(directory=str(_ART_DIR), html=False), name="rl_artifacts")
        # B 模块（yard_lighting）静态 artifacts
        if _YL_ART_DIR.exists():
            app.mount(
                "/api/rl/model/yard_lighting/artifacts",
                StaticFiles(directory=str(_YL_ART_DIR), html=False),
                name="yl_artifacts"
            )
        # 兼容旧路径
        app.mount("/api/rl/model/agv_charge/artifacts", StaticFiles(directory=str(_ART_DIR), html=False), name="rl_artifacts_legacy")
    # --- [E & D 模块静态 artifacts 挂载 · 新增] ---
    # E · bess_energy：注意该模块的 JSONL 位于模块根目录（无 artifacts 子目录）
    _BE_DIR = _BASE / "services" / "rl_model" / "bess_energy"
    if _ENABLE_LEGACY_RL and _BE_DIR.exists():
        @app.get("/api/rl/model/bess_energy/kpi_cards.json")
        async def bess_energy_kpi_cards_model():
            p = _BE_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="bess_energy kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

        @app.get("/api/rl/bess_energy/kpi_cards.json")
        async def bess_energy_kpi_cards_compat():
            p = _BE_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="bess_energy kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

        @app.get("/api/rl/model/bess_energy/artifacts/kpi_cards.json")
        async def bess_energy_kpi_cards_model_artifacts():
            p = _BE_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="bess_energy kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

        @app.get("/api/rl/bess_energy/artifacts/kpi_cards.json")
        async def bess_energy_kpi_cards_compat_artifacts():
            p = _BE_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="bess_energy kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

        # 兼容前端请求的 /api/rl/model/bess_energy/artifacts/*
        app.mount(
            "/api/rl/model/bess_energy/artifacts",
            StaticFiles(directory=str(_BE_DIR), html=False),
            name="be_artifacts",
        )
        # 兼容降级路径 /api/rl/bess_energy/artifacts/*
        app.mount(
            "/api/rl/bess_energy/artifacts",
            StaticFiles(directory=str(_BE_DIR), html=False),
            name="be_artifacts_compat",
        )

    # B · yard_lighting：KPI 卡片走模块根目录，历史曲线 JSONL 继续走 artifacts
    _YL_ROOT_DIR = _BASE / "services" / "rl_model" / "yard_lighting"
    if _ENABLE_LEGACY_RL and _YL_ROOT_DIR.exists():
        @app.get("/api/rl/model/yard_lighting/kpi_cards.json")
        async def yard_lighting_kpi_cards_model():
            p = _YL_ROOT_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="yard_lighting kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

        @app.get("/api/rl/yard_lighting/kpi_cards.json")
        async def yard_lighting_kpi_cards_compat():
            p = _YL_ROOT_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="yard_lighting kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    # A · agv_charge：KPI 卡片走模块根目录，历史曲线 JSONL 继续走 artifacts
    _AGV_ROOT_DIR = _BASE / "services" / "rl_model" / "agv_charge"
    if _ENABLE_LEGACY_RL and _AGV_ROOT_DIR.exists():
        @app.get("/api/rl/model/agv_charge/kpi_cards.json")
        async def agv_charge_kpi_cards_model():
            p = _AGV_ROOT_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="agv_charge kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

        @app.get("/api/rl/agv_charge/kpi_cards.json")
        async def agv_charge_kpi_cards_compat():
            p = _AGV_ROOT_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="agv_charge kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    # D · shore_bess：KPI 卡片走模块根目录，历史曲线 JSONL 走 artifacts 子目录
    _SB_ROOT_DIR = _BASE / "services" / "rl_model" / "shore_bess"
    _SB_DIR = _SB_ROOT_DIR / "artifacts"
    if _ENABLE_LEGACY_RL and _SB_ROOT_DIR.exists():
        @app.get("/api/rl/model/shore_bess/kpi_cards.json")
        async def shore_bess_kpi_cards_model():
            p = _SB_ROOT_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="shore_bess kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

        @app.get("/api/rl/shore_bess/kpi_cards.json")
        async def shore_bess_kpi_cards_compat():
            p = _SB_ROOT_DIR / "kpi_cards.json"
            if not p.exists():
                raise HTTPException(status_code=404, detail="shore_bess kpi_cards.json not found")
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    if _ENABLE_LEGACY_RL and _SB_DIR.exists():
        app.mount(
            "/api/rl/model/shore_bess/artifacts",
            StaticFiles(directory=str(_SB_DIR), html=False),
            name="sb_artifacts",
        )
        app.mount(
            "/api/rl/shore_bess/artifacts",
            StaticFiles(directory=str(_SB_DIR), html=False),
            name="sb_artifacts_compat",
        )

    # C · hvac_cooling：精确 kpi_cards.json 路由（优先于静态挂载，避免路径兼容问题）
    @_legacy_rl_get("/api/rl/model/hvac_cooling/kpi_cards.json")
    async def hvac_kpi_cards_model_root() -> JSONResponse:
        p = _BASE / "services" / "rl_model" / "hvac_cooling" / "kpi_cards.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="hvac_cooling kpi_cards.json not found")
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    @_legacy_rl_get("/api/rl/hvac_cooling/kpi_cards.json")
    async def hvac_kpi_cards_short_root() -> JSONResponse:
        p = _BASE / "services" / "rl_model" / "hvac_cooling" / "kpi_cards.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="hvac_cooling kpi_cards.json not found")
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    @_legacy_rl_get("/api/rl/model/hvac_cooling/artifacts/kpi_cards.json")
    async def hvac_kpi_cards_model_artifacts() -> JSONResponse:
        root_p = _BASE / "services" / "rl_model" / "hvac_cooling" / "kpi_cards.json"
        art_p = _BASE / "services" / "rl_model" / "hvac_cooling" / "artifacts" / "kpi_cards.json"
        p = art_p if art_p.exists() else root_p
        if not p.exists():
            raise HTTPException(status_code=404, detail="hvac_cooling kpi_cards.json not found")
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    @_legacy_rl_get("/api/rl/hvac_cooling/artifacts/kpi_cards.json")
    async def hvac_kpi_cards_short_artifacts() -> JSONResponse:
        root_p = _BASE / "services" / "rl_model" / "hvac_cooling" / "kpi_cards.json"
        art_p = _BASE / "services" / "rl_model" / "hvac_cooling" / "artifacts" / "kpi_cards.json"
        p = art_p if art_p.exists() else root_p
        if not p.exists():
            raise HTTPException(status_code=404, detail="hvac_cooling kpi_cards.json not found")
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    # C · hvac_cooling：根目录提供 kpi_cards.json，artifacts 子目录提供训练/评估 JSONL
    _HC_ROOT = _BASE / "services" / "rl_model" / "hvac_cooling"
    _HC_DIR = _HC_ROOT / "artifacts"
    if _ENABLE_LEGACY_RL and _HC_ROOT.exists():
        # ① 根目录：兼容前端直接读取 /api/rl/model/hvac_cooling/kpi_cards.json
        app.mount(
            "/api/rl/model/hvac_cooling",
            StaticFiles(directory=str(_HC_ROOT), html=False),
            name="hc_root",
        )
        # ② 兼容短路径 /api/rl/hvac_cooling/kpi_cards.json
        app.mount(
            "/api/rl/hvac_cooling",
            StaticFiles(directory=str(_HC_ROOT), html=False),
            name="hc_root_compat",
        )
    if _ENABLE_LEGACY_RL and _HC_DIR.exists():
        # ③ artifacts：训练/评估输出
        app.mount(
            "/api/rl/model/hvac_cooling/artifacts",
            StaticFiles(directory=str(_HC_DIR), html=False),
            name="hc_artifacts",
        )
        # ④ 兼容短路径 /api/rl/hvac_cooling/artifacts/*
        app.mount(
            "/api/rl/hvac_cooling/artifacts",
            StaticFiles(directory=str(_HC_DIR), html=False),
            name="hc_artifacts_compat",
        )

    # F · yard_crane：该模块的 JSONL 位于 artifacts 子目录（修正原注释）
    _YC_DIR = _BASE / "services" / "rl_model" / "yard_crane" / "artifacts"

    if _ENABLE_LEGACY_RL and _YC_DIR.exists():
        # ① 映射“无 /artifacts”的路径，兼容前端
        app.mount(
            "/api/rl/model/yard_crane",
            StaticFiles(directory=str(_YC_DIR), html=False),
            name="yc_artifacts_flat",
        )
        # ② 同时保留“带 /artifacts”的路径
        app.mount(
            "/api/rl/model/yard_crane/artifacts",
            StaticFiles(directory=str(_YC_DIR), html=False),
            name="yc_artifacts",
        )
        # ③ 兼容旧拼写（/api/rl/yard_crane ...）
        app.mount("/api/rl/yard_crane", StaticFiles(directory=str(_YC_DIR), html=False),
                  name="yc_artifacts_flat_compat")
        app.mount("/api/rl/yard_crane/artifacts", StaticFiles(directory=str(_YC_DIR), html=False),
                  name="yc_artifacts_compat")
    # G · port_G_qc_mvp：QC 码头（模块根包含 offline_dataset_qc.jsonl；artifacts 下有 policy_evaluate_history.jsonl）
    _G_DIR = _BASE / "services" / "rl_model" / "port_G_qc_mvp"
    _GA_DIR = _G_DIR / "artifacts"
    if _ENABLE_LEGACY_RL and _G_DIR.exists():
        # ① 映射“无 /artifacts”的路径，便于直接访问 offline_dataset_qc.jsonl
        app.mount(
            "/api/rl/model/port_G_qc_mvp",
            StaticFiles(directory=str(_G_DIR), html=False),
            name="g_artifacts_flat",
        )
        # 兼容降级路径
        app.mount(
            "/api/rl/port_G_qc_mvp",
            StaticFiles(directory=str(_G_DIR), html=False),
            name="g_artifacts_flat_compat",
        )
    if _ENABLE_LEGACY_RL and _GA_DIR.exists():
        # ② 标准 artifacts 路径（policy_evaluate_history.jsonl）
        app.mount(
            "/api/rl/model/port_G_qc_mvp/artifacts",
            StaticFiles(directory=str(_GA_DIR), html=False),
            name="g_artifacts",
        )
        # 兼容降级路径
        app.mount(
            "/api/rl/port_G_qc_mvp/artifacts",
            StaticFiles(directory=str(_GA_DIR), html=False),
            name="g_artifacts_compat",
        )


    # --- [end新增] ---

except NameError:
    # 如果当前文件里还没有 app（极少数自定义结构），创建最小 FastAPI 实例
    from fastapi import FastAPI
    app = FastAPI(title="RL API")
    if _ENABLE_LEGACY_RL and _ART_DIR.exists():
        app.mount("/api/rl/artifacts", StaticFiles(directory=str(_ART_DIR), html=False), name="rl_artifacts")
        app.mount("/api/rl/model/agv_charge/artifacts", StaticFiles(directory=str(_ART_DIR), html=False), name="rl_artifacts_legacy")

# 挂载路由
try:
    if _ENABLE_LEGACY_RL:
        app.include_router(rl_router)
except NameError:
    from fastapi import FastAPI
    app = FastAPI(title="RL API")
    if _ENABLE_LEGACY_RL:
        app.include_router(rl_router)
# ==== [Module A · RL artifacts & metrics endpoints · END] ======================
# ==== [Ops · MLOps/GRC/集成 · Router Mount · BEGIN] ==========================
try:
    from app.services.ops.api import get_router as get_ops_router
except Exception as _e:
    get_ops_router = None
    try: print("[ops] get_router import failed:", _e)
    except Exception: pass

try:
    if get_ops_router and _ENABLE_ENGINEERING_SIMULATORS:
        app.include_router(get_ops_router())
        print("[ops] engineering-simulator router mounted")
except Exception as _e:
    try: print("[ops] include_router failed:", _e)
    except Exception: pass
# ==== [Ops · Router Mount · END] =============================================
