from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime
import csv

__all__ = [
    "get_overview",
    "get_vessels_schedule_local",
]


def get_overview(di: Any) -> Dict[str, Any]:
    """Dev / 生态视角：开放平台 & App Center 概览服务。

    这个模块是整个“港口 AI 运营中枢”的【开放平台 / 生态层】：
    - 面向第三方 / 甲方 IT / 内部应用，展示有哪些开放 API、事件总线、SDK 能力；
    - 同时列出已经基于这些能力构建好的 App（泊位助手、碳账本、运力沙盘等）；
    - 首页 index.html 的「开放平台 & App Center」卡片会直接消费这里返回的结构。
    """  # noqa: D401

    # === 1) 当前默认港口（仅用于文案展示） ===
    port_name = "上海国际港"
    try:
        cfg = getattr(di, "config", None)
        if isinstance(cfg, dict):
            port_name = cfg.get("port", {}).get("display_name", port_name) or port_name
    except Exception:
        pass

    # === 2) 平台支持能力：RESTful API / Webhook / SDK ===
    platform_support: Dict[str, Any] = {
        "rest_apis": [
            # ——— 港口 / 船舶作业 ———
            {
                "path": "/api/vessels/schedule",
                "label": f"{port_name} · 船舶计划 / 靠离泊窗口（含泊位号、ETA/ETD）",
                "domain": "vessel",
                "ports": ["SIPG-Yangshan", "SIPG-Waigaoqiao"],
            },
            {
                "path": "/api/berths/plan",
                "label": "泊位计划 & 码头作业图（Berth Gantt + 航道约束）",
                "domain": "vessel",
                "ports": ["SIPG"],
            },
            # ——— 堆场 / 设备 / 作业队列 ———
            {
                "path": "/api/yard/blocks",
                "label": "堆场区块视图（堆存量、冷热度、堆高、箱型结构）",
                "domain": "yard",
                "ports": ["SIPG-Yard"],
            },
            {
                "path": "/api/equipment/telemetry",
                "label": "关键设备遥测（QC / YC / AGV / 岸电 / BESS / HVAC 等）",
                "domain": "equipment",
                "ports": ["SIPG"],
            },
            # ——— 能源 / 电价 / 碳 ———
            {
                "path": "/api/energy/profile",
                "label": "港区综合负荷 & 电价档位（含峰谷电价 / 需量）",
                "domain": "energy",
                "ports": ["SIPG"],
            },
            {
                "path": "/api/esg/emissions",
                "label": "按设备 / 场景 / 船公司统计的碳排放与边际减排",
                "domain": "esg",
                "ports": ["SIPG"],
            },
            # ——— 数字孪生 / 场景库 / RL ———
            {
                "path": "/api/twin/run",
                "label": "数字孪生仿真接口（支持场景库 + 强化学习策略回放）",
                "domain": "twin",
                "ports": ["SIPG"],
            },
            {
                "path": "/api/rl/policies",
                "label": "RL 策略列表 / OPE 评估结果 / 守护栏状态",
                "domain": "rl",
                "ports": ["SIPG"],
            },
            # ——— 运营 & 告警 / 审计 ———
            {
                "path": "/api/ops/alerts",
                "label": "运营告警中心（策略异常、设备健康、需量风险等）",
                "domain": "ops",
                "ports": ["SIPG"],
            },
            {
                "path": "/api/compliance/reports",
                "label": "合规报表（月 / 季 · 范畴 1/2 · 电网/岸电/自发电分摊）",
                "domain": "compliance",
                "ports": ["SIPG"],
            },
        ],
        "webhooks": [
            {
                "event": "job_completed",
                "label": "作业完成事件（可用于自动结算 / 绩效）",
            },
            {
                "event": "vessel_eta_updated",
                "label": "船舶 ETA / 泊位调整事件",
            },
            {
                "event": "alert_triggered",
                "label": "能耗 / 需量 / 设备告警触发",
            },
            {
                "event": "policy_switched",
                "label": "RL 策略切换 / Rollout 状态变更",
            },
        ],
        "sdk": {
            "notebooks": [
                {
                    "name": "sipg_energy_quickstart.ipynb",
                    "label": f"{port_name} 能耗 & 需量 quickstart",
                },
                {
                    "name": "sipg_berth_rl_sandbox.ipynb",
                    "label": f"{port_name} 泊位调度 RL 策略沙盒",
                },
            ],
            "languages": ["python", "js"],
        },
    }

    # === 3) 已有 App 列表 ===
    apps: List[Dict[str, Any]] = [
        {
            "id": "berth_planner",
            "name": f"{port_name} 泊位调度助手",
            "description": "结合 ETA / 潮汐 / 吃水 / 设备计划，智能推荐靠离泊窗口并输出 Gantt 图。",
            "owner": "internal",
            "category": "operations",
        },
        {
            "id": "carbon_ledger",
            "name": "港口碳账本工具",
            "description": "按场景 / 船舶 / 码头核算碳排放，衔接 ESG 合规与报表模块。",
            "owner": "internal",
            "category": "esg",
        },
        {
            "id": "capacity_sandbox",
            "name": "港口运力沙盘",
            "description": "数字孪生 + 场景库推演，多方案对比不同策略下的运力与能耗表现。",
            "owner": "partner",
            "category": "twin",
        },
        {
            "id": "multiport_overview",
            "name": "多港区部署总览",
            "description": "一图看清不同港口 / 园区 RL / Twin 部署阶段与收益情况。",
            "owner": "internal",
            "category": "platform",
        },
        {
            "id": "ops_copilot",
            "name": "Ops Copilot（运营副驾）",
            "description": "自然语言查询 + 策略推荐，联动 EnergyX / Dispatch / TwinLab 做决策说明。",
            "owner": "internal",
            "category": "assistant",
        },
    ]

    return {
        "platform_support": platform_support,
        "apps": apps,
        "_source": "service:app_center+sipg",
    }


# =========================
# 下方为“模块内数据读取”能力
# =========================

def _resolve_data_path(filename: str) -> Path:
    """定位 app_center 模块内 data 目录下的文件路径。"""
    here = Path(__file__).resolve()  # .../app/services/app_center/service.py
    data_dir = here.parent / "data"
    return (data_dir / filename).resolve()


def _parse_iso(ts: str) -> datetime:
    """容错 ISO8601 解析：支持 '...Z'、无时区、仅日期。"""
    ts = (ts or "").strip()
    if not ts:
        raise ValueError("empty timestamp")
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        # 只有日期时补时分秒
        return datetime.fromisoformat(ts + "T00:00:00+00:00")


def get_vessels_schedule_local(
    start: str,
    end: str,
    port_code: str = "SIPG",
) -> List[Dict[str, Any]]:
    """读取模块内 CSV（优先 vessel_plan_sipg.csv）并按 ETA 过滤。

    参数:
      - start/end: ISO8601 字符串（支持 'Z' / 无时区 / 仅日期）
      - port_code: 仅作为输出字段回传，不参与筛选

    返回:
      - List[Dict]: 规范化的船舶计划记录
    """
    t0 = _parse_iso(start)
    t1 = _parse_iso(end)

    candidates = [
        _resolve_data_path("vessel_plan_sipg.csv"),
        _resolve_data_path("vessel_plan.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []

    rows: List[Dict[str, Any]] = []
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

            def _get(name: str, default=""):
                v = r.get(name, default)
                return v.strip() if isinstance(v, str) else v

            try:
                draft_m = float(_get("draft_m", 0) or 0)
            except Exception:
                draft_m = 0.0
            try:
                loa_m = float(_get("loa_m", 0) or 0)
            except Exception:
                loa_m = 0.0
            try:
                moves = int(float(_get("moves", 0) or 0))
            except Exception:
                moves = 0

            rows.append(
                {
                    "vessel_id": _get("vessel_id"),
                    "imo": _get("imo"),
                    "mmsi": _get("mmsi"),
                    "carrier": _get("carrier"),
                    "service": _get("service"),
                    "eta": _get("eta"),
                    "etd": _get("etd"),
                    "berth_id": _get("berth_id"),
                    "quay": _get("quay"),
                    "draft_m": draft_m,
                    "loa_m": loa_m,
                    "moves": moves,
                    "tide_window": _get("tide_window"),
                    "tug_class": _get("tug_class"),
                    "priority": _get("priority", "normal"),
                    "remarks": _get("remarks"),
                    "port_code": port_code,
                    "_source": f"csv:{path.name}",
                }
            )

    return rows
