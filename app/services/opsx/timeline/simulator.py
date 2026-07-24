"""
作战时间线 · 港口场景友好模拟器
- 端点（由 app/services/opsx/api.py 调用）：
    GET /api/opsx/timeline?horizon_min=60 -> get_timeline(horizon_min)

【大白话】
- 在“未来 horizon_min 分钟内”合成一条事件流：审批/下发、A/B、在线学习、回滚/发布、DR 窗口、
  靠泊计划、设备告警等。字段对齐你的前端列表：
    ts(ISO8601), kind, severity(info|warn|critical), text, meta(可选)
- 真接入时：把此文件替换为“从你们的调度/审批系统、消息总线、TSDB/OLAP”拉数据即可。
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import random

# 可按需扩展的事件种类标签（用于前端彩色标签展示）
KIND_LABELS = [
    "submit", "approve", "deploy", "rollback", "abtest", "learn",
    "dr_event", "vessel", "job", "alarm"
]

def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()

def _within(now: datetime, minutes: int, offset_min: int) -> datetime:
    return now + timedelta(minutes=offset_min)

def _maybe(prob: float) -> bool:
    return random.random() < prob

def _pick(seq: List[Any]) -> Any:
    return random.choice(seq)

def _vessel_code() -> str:
    return f"MV-{random.randint(100,999)}"

def _qc_id() -> str:
    return f"QC-{random.randint(5,12)}"

def _agv_id() -> str:
    return f"AGV-{random.randint(101,199)}"

def _strategy_id() -> str:
    return f"S-{random.randint(320,380)}"

def _version() -> str:
    return f"v{random.randint(1,2)}.{random.randint(0,9)}"

def _make_base_events(now: datetime) -> List[Dict[str, Any]]:
    """
    一些“主线”事件：审批->A/B->学习->发布/回滚；穿插 DR 窗口与靠泊
    """
    sid = _strategy_id()
    cand = _version()
    stable = _version()

    evs = [
        # 提交审批
        {"ts": _iso(_within(now, 60, 2)),  "kind": "submit",  "severity": "info",
         "text": f"提交审批 · 策略 {sid}", "meta": {"strategy_id": sid, "candidate": cand}},
        # 触发 A/B
        {"ts": _iso(_within(now, 60, 10)), "kind": "abtest",  "severity": "warn",
         "text": "A/B 触发 · ΔkWh 偏差较大", "meta": {"strategy_id": sid, "armA":"stable", "armB":"candidate"}},
        # 在线学习
        {"ts": _iso(_within(now, 60, 24)), "kind": "learn",   "severity": "info",
         "text": "在线学习 · EMA(0.2)", "meta": {"strategy_id": sid, "method":"EMA", "alpha":0.2}},
    ]

    # 发布 or 回滚 二选一（带点随机）
    if _maybe(0.35):
        evs.append({"ts": _iso(_within(now, 60, 40)), "kind": "rollback", "severity": "critical",
                    "text": f"回滚至 {stable}", "meta": {"from": cand, "to": stable}})
    else:
        evs.append({"ts": _iso(_within(now, 60, 40)), "kind": "deploy", "severity": "info",
                    "text": f"发布 {cand} -> 全量", "meta": {"version": cand}})

    # 需求响应（DR）窗口提示
    evs.append({"ts": _iso(_within(now, 60, 15)), "kind": "dr_event", "severity": "info",
                "text": "DR 窗口即将开始 · 17:00-18:00 reserve up",
                "meta": {"window":"17:00-18:00", "type":"reserve_up"}})

    # 靠泊计划/到港
    vessel_eta_min = random.randint(12, 55)
    evs.append({"ts": _iso(_within(now, 60, vessel_eta_min)), "kind":"vessel", "severity":"info",
                "text": f"靠泊计划 · {_vessel_code()} 预计到港", "meta": {"eta_min": vessel_eta_min}})

    return evs

def _make_noise_events(now: datetime) -> List[Dict[str, Any]]:
    """
    一些“补充/噪声”事件：设备告警、作业、AGV 充电切换等，让时间线更贴近现场
    """
    evs: List[Dict[str, Any]] = []

    # 设备告警（概率低一点）
    if _maybe(0.25):
        level = _pick(["warn","critical","warn"])
        evs.append({"ts": _iso(_within(now, 60, random.randint(6, 36))), "kind":"alarm", "severity": level,
                    "text": f"{_qc_id()} 扭矩峰值异常", "meta":{"threshold":"Nm_peak","value": _pick([1.3,1.6,2.1])}})

    # 作业完成/下发
    for off in [8, 18, 30, 50]:
        if _maybe(0.7):
            evs.append({"ts": _iso(_within(now, 60, off)), "kind": "job", "severity":"info",
                        "text": "作业下发 · YARD_BLOCK 12A", "meta":{"job_id": f"J{random.randint(1000,9999)}"}})

    # AGV 充电切换（轻量事件）
    if _maybe(0.55):
        evs.append({"ts": _iso(_within(now, 60, random.randint(12, 42))), "kind":"job", "severity":"info",
                    "text": f"{_agv_id()} 切换充电策略", "meta":{"scheme": _pick(["peak_shave","co2_min","cost_min"])}})

    return evs

def get_timeline(horizon_min: int = 60, port_id: Optional[str] = None) -> Dict[str, Any]:
    """
    生成“未来 horizon_min 分钟”的事件清单。
    - port_id 预留：真实落地时按港口/场区筛选（这里暂不使用）
    """
    horizon_min = max(1, min(int(horizon_min), 240))
    now = datetime.utcnow()

    events = _make_base_events(now) + _make_noise_events(now)

    # 截断到 horizon
    end = now + timedelta(minutes=horizon_min)
    events = [e for e in events if datetime.fromisoformat(e["ts"]) <= end]

    # 按时间升序
    events.sort(key=lambda e: e["ts"])
    return {"items": events}
