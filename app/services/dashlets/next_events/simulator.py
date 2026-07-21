from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import random, math

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else 86420)

def _local_str(ts: datetime) -> str:
    # 用传入 ts 的时区作为本地；若无 tz 则视为 UTC
    lt = ts.astimezone(ts.tzinfo or timezone.utc)
    return lt.strftime("%H:%M")

def _add(evts: List[Dict[str,Any]], ts: datetime, kind: str, desc: str,
         severity: str="info", **meta):
    evts.append({"ts": ts, "ts_local": _local_str(ts), "kind": kind, "desc": desc,
                 "severity": severity, "meta": meta or {}})

def _within(ts: datetime, now: datetime, horizon: timedelta) -> bool:
    return now <= ts <= now + horizon

def simulate_next_events(asset: str, now: datetime, horizon_min: int=60,
                         seed: Optional[int]=None) -> List[Dict[str,Any]]:
    """
    生成“未来 horizon_min 分钟内”的关键事件：
    - 价格峰值 (18-21)、DR(14-16) 的开始/结束
    - QC：作业高强度开始/换班，YC：傍晚活跃开始
    - 潮汐极值（~6h 半周期）→ 靠泊/作业影响
    - 策略动作：BESS 放电、HVAC 设定点调整、QC/YC 调度
    - Guard 风险预警
    - 天气突发（阵风/高温）
    - 船期靠泊/离泊（概率事件）
    """
    rng = _rng(seed)
    evts: List[Dict[str,Any]] = []
    horizon = timedelta(minutes=horizon_min)
    tz = now.tzinfo or timezone.utc

    asset_l = asset.lower()
    is_qc   = asset_l.startswith(("qc","g_","port_g"))
    is_yc   = asset_l.startswith(("yc","f_","port_f"))
    is_bess = asset_l.startswith(("bess","shore"))
    is_hvac = asset_l.startswith(("hvac","plant","cool"))

    # --- 价格峰/DR 窗口 --- #
    today = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    # DR 14-16
    for hh0, hh1 in ((14,16),):
        s = today.replace(hour=hh0) + timedelta(minutes=rng.randint(-10,10))
        e = today.replace(hour=hh1) + timedelta(minutes=rng.randint(-10,10))
        if _within(s, now, horizon): _add(evts, s, "dr_start", "DR 窗口开始", "warn")
        if _within(e, now, horizon): _add(evts, e, "dr_end",   "DR 窗口结束", "info")
    # 价格峰 18-21
    for hh0, hh1 in ((18,21),):
        s = today.replace(hour=hh0) + timedelta(minutes=rng.randint(-8,8))
        e = today.replace(hour=hh1) + timedelta(minutes=rng.randint(-8,8))
        if _within(s, now, horizon): _add(evts, s, "price_peak_start", "电价高峰开始", "warn")
        if _within(e, now, horizon): _add(evts, e, "price_peak_end",   "电价高峰结束", "info")

    # --- 作业强度（按资产） --- #
    if is_qc:
        for hh0 in (8,13):  # 白班两个开始时刻
            s = today.replace(hour=hh0) + timedelta(minutes=rng.randint(-5,5))
            if _within(s, now, horizon): _add(evts, s, "ops_high_start", "岸桥作业强度上升", "info")
        for sw in (7,12,17):  # 换班窗口提醒
            t = today.replace(hour=sw) + timedelta(minutes=rng.randint(0,20))
            if _within(t, now, horizon): _add(evts, t, "shift_window", "换班窗口，注意波动", "info")
    elif is_yc:
        s = today.replace(hour=17, minute=30) + timedelta(minutes=rng.randint(-5,5))
        if _within(s, now, horizon): _add(evts, s, "ops_high_start", "场桥傍晚作业上升", "info")

    # --- 潮汐极值（~6h 半周期，给出最近的一个） --- #
    # 简化：以 00:00 为参考每 6h 一个极值，交替 high/low
    base = today
    k = math.floor(((now - base).total_seconds()/3600.0) / 6.0) + 1
    next_tide = base + timedelta(hours=6*k)
    if _within(next_tide, now, horizon):
        kind = "tide_high" if (k % 2 == 0) else "tide_low"
        _add(evts, next_tide, kind, "潮汐极值", "info")

    # --- 策略动作建议 --- #
    h = now.hour + now.minute/60.0
    if is_bess:
        # 下一个高价段前 10 分钟提示放电
        hint = today.replace(hour=18) - timedelta(minutes=10)
        if hint < now: hint = now + timedelta(minutes=15)
        if _within(hint, now, horizon):
            _add(evts, hint, "dispatch_discharge", "建议 BESS 准备放电", "warn", power_kw=60)
    elif is_hvac:
        t = now + timedelta(minutes=15)
        if _within(t, now, horizon):
            _add(evts, t, "setpoint_adjust", "建议下调供水温度 0.3℃", "info", delta=-0.3)
    elif is_qc or is_yc:
        t = now + timedelta(minutes=20)
        if _within(t, now, horizon):
            _add(evts, t, "schedule_adjust", "建议重排 1 条作业队列以避峰", "info")

    # --- Guard 风险预警（概率） --- #
    if rng.random() < (0.25 if (is_qc or is_bess) else 0.15):
        t = now + timedelta(minutes=rng.randint(8, 28))
        sev = "critical" if rng.random() < 0.4 else "warn"
        reason = rng.choice(["功率上限","SOC 下限","温度上限","速率限制","SLA"])
        _add(evts, t, "guard_risk", f"可能触发 Guard：{reason}", sev)

    # --- 天气小扰动（概率） --- #
    if rng.random() < 0.12:
        t = now + timedelta(minutes=rng.randint(10, 40))
        kind = rng.choice(["gust","heat"])
        txt  = "阵风增强，注意吊具摆动" if kind=="gust" else "气温升高，冷站负荷上扬"
        _add(evts, t, kind, txt, "info")

    # --- 船期靠/离泊（小概率，但更贴近港口） --- #
    if is_qc and rng.random() < 0.20:
        t = now + timedelta(minutes=rng.randint(5, 45))
        _add(evts, t, "vessel_arrival", "船舶预计靠泊", "info", imo=rng.randint(9000000,9999999))

    # 排序、去重、限量
    evts.sort(key=lambda x: x["ts"])
    # 去重（相同 kind 且相差<2min 只留一条）
    dedup: List[Dict[str,Any]] = []
    for e in evts:
        if dedup and e["kind"]==dedup[-1]["kind"] and abs((e["ts"]-dedup[-1]["ts"]).total_seconds())<120:
            continue
        dedup.append(e)
    return dedup[:20]
