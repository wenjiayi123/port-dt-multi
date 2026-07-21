# app/services/dashlets/__init__.py
"""
Dashlets 路由聚合器
- 统一前缀: /api/dashlets
- 子模块(插件)按“能导入就挂载、没有就跳过”的策略，不影响服务启动
"""
from __future__ import annotations
from fastapi import APIRouter

router = APIRouter(prefix="/api/dashlets", tags=["dashlets"])

def _try_include(modpath: str, attr: str = "router") -> None:
    """
    大白话：尝试导入某个插件的 router，如果这个插件目录还没建好，就忽略。
    这样你可以一个个地加插件，服务不会因为没做完的插件而起不来。
    """
    try:
        module = __import__(modpath, fromlist=[attr])
        sub_router = getattr(module, attr, None)
        if sub_router is not None:
            router.include_router(sub_router)
    except Exception:
        # 这里不打印日志，保持安静；需要调试时可换成 logger.warning(...)
        pass

# —— 在这里按名称“尝试挂载”各个插件 —— #
_try_include("app.services.dashlets.event_bands")        # /event_bands

# 预留的（以后你建好对应目录再自动挂载，不用改这里）
_try_include("app.services.dashlets.action_markers")     # /action_markers
_try_include("app.services.dashlets.calibration")        # /calibration
_try_include("app.services.dashlets.residual_heatmap")   # /residual_heatmap
_try_include("app.services.dashlets.peak_risk")          # /peak_risk
_try_include("app.services.dashlets.savings")            # /savings
_try_include("app.services.dashlets.dq_lights")          # /dq_lights
_try_include("app.services.dashlets.next_events")        # /next_events
_try_include("app.services.dashlets.approvals_summary")  # /approvals_summary
_try_include("app.services.dashlets.rollout_mini")       # /rollout_mini
_try_include("app.services.dashlets.shap_topk")          # /shap_topk
