"""Read-only demo business views; never used as production readiness evidence.

All numbers come from the presentation snapshot. A changed adapter cannot borrow
the built-in terminal's annual report, group allocation or engineering constants.
"""
from __future__ import annotations

import math
from datetime import timedelta
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

from .service import _number, _timestamp, inputs, snapshot


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _ratio(a: float, b: float) -> float | None:
    return a / b if b > 0 else None


@lru_cache(maxsize=2)
def annual_report(dataset_sha: str, scale: float) -> dict[str, Any]:
    """One O(8760) pass on first report access; no startup job or polling."""
    rows, _, _, actual_sha = inputs()
    if actual_sha != dataset_sha:
        raise ValueError("annual report dataset hash mismatch")
    months = [{"month": f"2025-{m:02d}", "hours": 0, "throughput_teu": 0.0,
               "energy_kwh": 0.0, "electricity_carbon_kg": 0.0, "energy_cost_cny": 0.0,
               "peak_kw": 0.0, "scope1_kg": None} for m in range(1, 13)]
    for row in rows:
        m = months[int(row["timestamp"][5:7]) - 1]
        power = float(row["base_load_kw"]) * scale
        m["hours"] += 1
        m["throughput_teu"] += float(row["throughput_teu"]) * scale
        m["energy_kwh"] += power
        m["electricity_carbon_kg"] += power * float(row["carbon_kg_per_kwh"])
        m["energy_cost_cny"] += power * float(row["price_per_kwh"])
        m["peak_kw"] = max(m["peak_kw"], power)
    for m in months:
        m["energy_kwh_per_teu"] = _ratio(m["energy_kwh"], m["throughput_teu"])
        m["carbon_kg_per_teu"] = _ratio(m["electricity_carbon_kg"], m["throughput_teu"])
    totals = {key: sum(m[key] for m in months) for key in
              ("hours", "throughput_teu", "energy_kwh", "electricity_carbon_kg", "energy_cost_cny")}
    return {"available": True, "year": 2025, "timezone": "UTC", "months": months, "totals": totals,
            "source_ref": dataset_sha, "evidence_class": "scaled_engineering_series_not_audited_ledger",
            "audit_ready": False, "scope1_kg": None,
            "boundary": "2025 UTC完整年标定工程序列；包含电力间接排放估算，不含燃油/燃气与范围3，不构成Scope 1+2盘查或结算账单。"}


def _window(data: dict[str, Any]) -> dict[str, Any]:
    rows = data["series"]
    # Missing factors remain missing. Never silently coerce null to zero.
    if not rows:
        return {"available": False, "reason": "展示适配器未提供时间序列"}
    times = [_timestamp(r["timestamp"]) for r in rows]
    step = data.get("series_interval_seconds")
    if step is None and len(times) > 1:
        step = (times[1] - times[0]).total_seconds()
    if not _finite(step) or step <= 0 or step > 86400:
        return {"available": False, "reason": "需提供有效的series_interval_seconds，才能积分能耗"}
    if any(abs((b - a).total_seconds() - step) > 0.001 for a, b in zip(times, times[1:])):
        return {"available": False, "reason": "序列存在缺点/重复/乱序；本次不计算完整窗口或班次合计"}
    if any(t.microsecond or t.second or t.minute for t in times) or step != 3600:
        return {"available": False, "reason": "当前报表契约需要连续整点小时序列；其他采样周期请先由适配器聚合"}
    tz = ZoneInfo(data["profile"]["timezone"])
    shifts = [{"label": name, "start_hour": h, "hours": 0, "energy_kwh": 0.0,
               "throughput_teu": 0.0, "electricity_carbon_kg": 0.0, "energy_cost_cny": 0.0}
              for h, name in ((0, "00–08 时"), (8, "08–16 时"), (16, "16–24 时"))]
    carbon_ok = all(_finite(r.get("carbon_kg_per_kwh")) and r["carbon_kg_per_kwh"] >= 0 for r in rows)
    cost_ok = all(_finite(r.get("price_cny_per_kwh")) and r["price_cny_per_kwh"] >= 0 for r in rows)
    for row, instant in zip(rows, times):
        power = _number(row["power_kw"], "power_kw")
        flow = _number(row["throughput_teu_per_hour"], "throughput_teu_per_hour")
        if power < 0 or flow < 0:
            raise ValueError("negative power/flow in report inputs")
        shift = shifts[instant.astimezone(tz).hour // 8]
        shift["hours"] += 1
        shift["energy_kwh"] += power
        shift["throughput_teu"] += flow
        if carbon_ok:
            shift["electricity_carbon_kg"] += power * row["carbon_kg_per_kwh"]
        if cost_ok:
            shift["energy_cost_cny"] += power * row["price_cny_per_kwh"]
    for shift in shifts:
        if not carbon_ok:
            shift["electricity_carbon_kg"] = None
        if not cost_ok:
            shift["energy_cost_cny"] = None
    energy = sum(s["energy_kwh"] for s in shifts)
    teu = sum(s["throughput_teu"] for s in shifts)
    carbon = sum(s["electricity_carbon_kg"] for s in shifts) if carbon_ok else None
    return {"available": True, "start": times[0].isoformat(),
            "end": (times[-1] + timedelta(seconds=step)).isoformat(), "hours": len(rows),
            "timezone": str(tz), "shifts": shifts, "energy_kwh": energy, "throughput_teu": teu,
            "electricity_carbon_kg": carbon,
            "energy_cost_cny": sum(s["energy_cost_cny"] for s in shifts) if cost_ok else None,
            "energy_kwh_per_teu": _ratio(energy, teu),
            "carbon_kg_per_teu": _ratio(carbon, teu) if carbon is not None else None,
            "peak_kw": max(r["power_kw"] for r in rows), "scope1_kg": None,
            "audit_ready": False, "boundary": "完整序列窗口（可能含场景未来小时），不是墙钟今日实绩；分时按场景时区归组。"}


def workspace(at: str | None = None) -> dict[str, Any]:
    data = snapshot(at)
    internal = data["mode"] == "public_parameter_calibrated_demo"
    o = data["operating"]
    now = _timestamp(data["source_timestamp"])
    calls = [c for c in data["calls"] if _timestamp(c["planned_completion"]) > now - timedelta(hours=12)
             and _timestamp(c["planned_berth_time"]) < now + timedelta(hours=12)]
    service_hours = [(_timestamp(c["planned_completion"]) - _timestamp(c["planned_berth_time"])).total_seconds() / 3600 for c in calls]
    wait = [max(0, (_timestamp(c["planned_berth_time"]) - _timestamp(c["eta"])).total_seconds() / 60) for c in calls]
    params = data["profile"]["scenario_parameters"]
    qc_rate = data["profile"]["facts"]["qc_record_moves_per_hour_lower_bound"]["value"] * params["net_qc_rate_fraction_of_public_record"]["value"] if internal else None
    window = _window(data)
    metrics = []

    def metric(key, name, value, unit, evidence, adapter, detail):
        if not _finite(value):
            value = None
        metrics.append({"id": key, "name": name, "value": value, "unit": unit,
                        "available": value is not None, "evidence_class": evidence,
                        "replacement": adapter, "detail": detail, "source_ref": data["provenance"]["source_ref"]})

    for stem, label, unit in (("berths", "泊位", "个"), ("qc", "岸桥", "台"), ("rmg", "轨道吊", "台"), ("agv", "AGV", "台")):
        metric(stem + "_total", label + "规模", o[stem + "_total"], unit, "public_fact" if internal else "adapter_snapshot", "CMDB / 码头主数据", "公开规模有资料日期，不代表同日现场状态")
        metric(stem + "_busy", label + "作业分配", o[stem + "_busy"], unit, "resource_simulation" if internal else "adapter_snapshot", "TOS / ECS", "推演资源分配；AGV和轨道吊按岸桥工作量映射，不代表实际设备状态")
    metric("berth_occupancy_pct", "泊位占用率", 100 * o["berths_busy"] / o["berths_total"] if o["berths_total"] else None, "%", "resource_simulation", "TOS 泊位时窗", "当前时刻作业泊位 / 总泊位")
    metric("qc_moves_per_hour", "净单桥效率设定", qc_rate, "自然箱/台时", "engineering_parameter", "QC 作业事件 / 算法配置", "公开最快60自然箱/台时的50%=30；工程设定，不是平均实测或优化收益")
    metric("mean_service_hours", "计划装卸平均时长", sum(service_hours) / len(service_hours) if service_hours else None, "h", "schedule_derived", "TOS 计划靠泊/完成时刻", "与当前前后12小时相交的计划作业；不含引航/拖轮，不等于平均在港时间")
    metric("mean_wait_minutes", "计划待泊平均时长", sum(wait) / len(wait) if wait else None, "min", "schedule_derived", "TOS ETA / 计划靠泊", "同一计划窗口，零表示此资源方案没有泊位排队，不代表现场无延误")
    yard = o.get("yard_occupancy_ratio")
    if yard is not None and (not _finite(yard) or not 0 <= yard <= 1):
        raise ValueError("yard_occupancy_ratio must be in [0,1]")
    metric("yard_occupancy_pct", "堆场占用工程输入", yard * 100 if yard is not None else None, "%", "engineering_derived", "TOS 箱位快照", "仓库原有工程工况比例；未据此伪造总箱位、翻箱或堆存期")
    for key, label, unit in (("energy_kwh", "窗口用电", "kWh"), ("throughput_teu", "窗口标定箱量", "TEU"),
                             ("electricity_carbon_kg", "窗口电力碳排", "kgCO₂e"), ("peak_kw", "窗口负荷峰值", "kW"),
                             ("energy_kwh_per_teu", "单位吞吐能耗", "kWh/TEU"), ("carbon_kg_per_teu", "单位吞吐电力碳排", "kgCO₂e/TEU")):
        metric(key, label, window.get(key), unit, "engineering_derived" if internal else "snapshot_derived", "EMS + TOS + 因子版本", window.get("boundary", window.get("reason")))
    for key, label, unit in (("wind_speed_mps", "风速", "m/s"), ("tide_m", "潮位", "m"), ("wave_height_m", "浪高", "m"), ("ambient_c", "气温", "°C"), ("current_speed_mps", "流速", "m/s")):
        metric(key, label, data["environment"].get(key), unit, "historical_reanalysis" if internal else "adapter_snapshot", "授权气象 / 水文接口", "保留来源时间；潮位不是可直接使用的航道通航水深")
    pending = [
        ("truck_turnaround", "集卡周转 / 预约履约", "闸口进出和预约事件；没有场内外集卡流量依据，保留缺项"),
        ("yard_dwell", "堆存期 / 翻箱率", "逐箱箱位与事件履历；占用比例不足以推导停留时长"),
        ("shore_coverage", "岸电覆盖 / 储能装机", "岸电连接记录、BMS资产台账；工程负荷不证明已安装"),
        ("real_vessels", "真实船名 / AIS / 实际ETA", "授权TOS/AIS；演示只使用DEMO编号和计划时间"),
        ("scope1", "Scope 1 / 全范围碳盘查", "燃油/燃气、组织边界、核验因子与审计台账；缺失不填0"),
        ("financial", "财务节省 / 结算 / ESG评分", "发票、电费合同、独立归因和评级方法；工程测算不充当核证收益"),
        ("safety", "事故 / 告警 / 工单闭环", "EHS、CMMS和真实回执；未接入不等于零事故"),
        ("forecast", "预测现场准确率 / 孪生保真度", "授权实际观测和独立验证窗；训练误差不充当现场准确率"),
        ("approval", "审批 / 执行权 / 现场准入", "原有IAM、异人确认、联锁/回读/回滚验收；不通过演示源授权"),
        ("other_ports", "其他港口生产部署", "对应港口自己的授权数据；不将洋山演示复制成其他港口实绩"),
    ]
    groups = data.get("asset_groups", [])
    group_total = sum(_number(g["kW"], "group power") for g in groups)
    complete_groups = bool(groups) and math.isclose(group_total, o["power_kw"], rel_tol=1e-5, abs_tol=0.01)
    return {"schema": "port-dt-demo-workspace.v1", "snapshot": data, "window": window,
            "annual": annual_report(inputs()[3], inputs()[1]) if internal else
                      {"available": False, "reason": "外部展示源仅汇总所提供的序列；不混入内置洋山2025年报。可提供完整年度小时序列生成窗口合计。"},
            "operations": {"plan_window_start": (now - timedelta(hours=12)).isoformat(),
                           "plan_window_end": (now + timedelta(hours=12)).isoformat(), "calls_in_window": len(calls)},
            "metrics": metrics, "groups_complete": complete_groups,
            "coverage": {"available": sum(m["available"] for m in metrics), "total": len(metrics),
                         "pending_site_categories": len(pending), "label": "展示字段齐备度，非生产接入率"},
            "site_pending": [{"id": key, "name": name, "reason": reason} for key, name, reason in pending],
            "not_applicable": [{"name": "全港PUE", "reason": "PUE用于数据中心能效；没有设施边界与IT用电时，不用全港电量编造PUE。"}],
            "production_authority": False, "dispatch_allowed": False, "audit_ready": False,
            "source_mode": "公开参数标定演示" if internal else "外部展示快照",
            "boundary": "本工作台只补展示层；不向现场网关写入模拟证据，不改模型权重、封存盲测、现场告警或审批状态。"}
