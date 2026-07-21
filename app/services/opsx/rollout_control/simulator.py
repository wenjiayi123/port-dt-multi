"""
Rollout 控制台 · 港口真实场景友好的模拟器
- 作用：提供候选策略版本、稳定版本、灰度流量等状态，并能设置流量/一键回滚
- 端点：由 app/services/opsx/api.py 调用
    GET  /api/opsx/rollout/status      -> get_status()
    POST /api/opsx/rollout/traffic     -> set_traffic(pct)
    POST /api/opsx/rollout/rollback    -> do_rollback()

【大白话】
1) 这就是“上线流量拨盘”的假数据后台，先用内存变量模拟港口现场。
2) 以后你要接真数据，只要在“TODO 真接入”这些位置，把读取/写入改为你们的配置中心或数据库即可。
   - 例如：读 TOS/WMS 当前班次、读策略仓库当前候选版本、写入灰度百分比到配置中心等。
3) 我顺手加了 port 上下文与几个港口指标，前端目前没用到，但你以后能直接给到可视化。
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Any
import random

# ============== 内存里的“现场状态”（重启会重置） ==============
_STATE: Dict[str, Any] = {
    # 发布阶段：canary（灰度中）/ stable（全量稳定）/ rollback（刚回滚）/ freeze（冻结）
    "phase": "canary",

    # 版本：按模块粒度举例（你可换成统一版本号）
    "candidate": {
        "agv_charge": "v2.1",
        "yard_crane": "v1.8",
        "shore_bess": "v1.4"
    },
    "stable": {
        "agv_charge": "v2.0",
        "yard_crane": "v1.7",
        "shore_bess": "v1.3"
    },

    # 灰度流量（0~1），前端会显示成百分比
    "traffic_pct": 0.25,

    # 最近一次状态更新时间
    "updated_at": datetime.utcnow().isoformat(),

    # ---- 港口上下文（示例：方便你后面直接接入真实数据）----
    "port_ctx": {
        "port_id": "PORT_G",
        "terminal": "QC-YARD-A",
        "shift": "D",  # D=日班 / N=夜班
        "grid_co2_intensity_kg_per_kwh": 0.462,  # 电网碳强度
        "market_price_usd_per_mwh": 78.0,
        "meteo": {"wind": 6.2, "tide_cm": 45},
        "berth_utilization_1h": 0.72,   # 过去1小时泊位利用率
        "qc_throughput_teu_1h": 128,    # 过去1小时岸桥吞吐
        "dr_event_next": "17:00-18:00 reserve up",  # 下一次需求响应窗口
    },

    # ---- 指标摘要（供你前端/作战报告使用；此组件前端暂不展示）----
    "metrics": {
        "uptime_24h": 0.998,
        "abtest_delta_kwh": -1320.5,    # 相对基线的节电（正负均可能）
        "economics_usd_day": 423.4,     # 当天经济收益估算
    }
}

# ============== 工具函数 ==============
def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _promote_candidate_to_stable():
    """
    把 candidate 提升为 stable（模拟“全量上线”）
    - 真实环境：这里改成你们的版本管理流程（例如打标签/写配置中心/触发CD流水线）
    """
    _STATE["stable"] = dict(_STATE["candidate"])
    _STATE["phase"] = "stable"
    _STATE["updated_at"] = _now_iso()

def _port_drift_tick():
    """
    轻微随机抖动港口上下文（仅为了让前端看到“活”的数据）
    - 真接入时：这段可以删掉；从 TSDB/消息总线实时取数即可。
    """
    ctx = _STATE["port_ctx"]
    ctx["berth_utilization_1h"] = round(_clamp01(ctx["berth_utilization_1h"] + random.uniform(-0.01, 0.01)), 3)
    ctx["qc_throughput_teu_1h"] = max(0, int(ctx["qc_throughput_teu_1h"] + random.uniform(-3, 3)))
    ctx["market_price_usd_per_mwh"] = round(max(0, ctx["market_price_usd_per_mwh"] + random.uniform(-2, 2)), 2)
    ctx["meteo"]["wind"] = round(max(0, ctx["meteo"]["wind"] + random.uniform(-0.4, 0.4)), 1)
    ctx["touched_at"] = _now_iso()

# ============== 对外函数（被路由调用） ==============

def get_status() -> Dict[str, Any]:
    """
    返回当前上线状态（供 GET /rollout/status 使用）

    【真实接入提示】
    - TODO 真接入(读)：把 _STATE 的来源改成你的配置中心/DB（比如：Redis/Etcd/Postgres）
    - 例如：读 key（port_id/terminal）对应的 candidate/stable/traffic_pct/phase
    """
    _port_drift_tick()
    # 为了兼容前端，提供简单字符串版本
    return {
        "phase": _STATE["phase"],
        # 组合一个短字符串版本号，易读：agv_charge@v2.1|yard_crane@v1.8|shore_bess@v1.4
        "candidate": "|".join([f"{k}@{v}" for k, v in _STATE["candidate"].items()]),
        "stable":    "|".join([f"{k}@{v}" for k, v in _STATE["stable"].items()]),
        "traffic_pct": _STATE["traffic_pct"],
        "updated_at": _STATE["updated_at"],
        # 附带现场上下文（前端目前不展示，但你后续可以用）
        "port": _STATE["port_ctx"],
        "metrics": _STATE["metrics"]
    }

def set_traffic(pct: float) -> None:
    """
    设置灰度流量（供 POST /rollout/traffic 使用）
    - pct: 0~1 浮点数；前端传 0~100% 会被 api.py 解析为 0~1

    逻辑：
    - pct == 0    -> phase = "freeze"（冻结）
    - 0 < pct < 1 -> phase = "canary"（灰度）
    - pct >= 1    -> 视为“全量上线成功”，自动把 candidate 提升为 stable，phase="stable"
    """
    pct = _clamp01(pct)

    # TODO 真接入(写)：把 pct 写入配置中心/DB
    _STATE["traffic_pct"] = pct
    _STATE["updated_at"] = _now_iso()

    if pct == 0.0:
        _STATE["phase"] = "freeze"
    elif pct >= 1.0:
        _promote_candidate_to_stable()
        _STATE["traffic_pct"] = 1.0  # 全量
    else:
        _STATE["phase"] = "canary"

def do_rollback() -> None:
    """
    一键回滚（供 POST /rollout/rollback 使用）
    - 把 candidate 直接等于 stable，并清零流量；phase=rollback
    - 真实环境里，你可以在这里触发 CI/CD 回滚动作或写入事件总线
    """
    # TODO 真接入(写)：触发你们的回滚管道/写事件
    _STATE["candidate"] = dict(_STATE["stable"])
    _STATE["traffic_pct"] = 0.0
    _STATE["phase"] = "rollback"
    _STATE["updated_at"] = _now_iso()
