# -*- coding: utf-8 -*-
"""
app/adapters/tos_client.py

【文件用途（大白话）】
- 统一对接 TOS/WMS：船期、泊位计划、桥机计划、作业工单、堆场库存、车队预约等。
- 未配置真实 TOS 时使用可复现模拟器；已配置真实接口但请求失败时默认直接报错，不静默伪装。
- 所有对外方法均返回“归一化”的 dict/list，字段稳定，便于上层直接使用。
- 自带审计落盘：自测时把抓取/模拟的快照落到 data/objects/audit/evt-tos-*.json。

【落地接入方式】
- 真实 TOS：编辑 data/objects/config/tos_client.json（启动时自动生成模板）。
- 只需填 base_url、token 以及各资源的 path（船期/vessel_calls、工单/move_orders 等），
  就能替换为现场数据；其它代码无须改动。

【与哪些文件关联】
- 可被 app/adapters/schedule_sources.py 或 app/services/schedule.py 调用，作为「作业驱动预测」输入；
- 也可被 RL/仿真读取，作为到港船期与装卸节拍的驱动量；
- 审计输出到 data/objects/audit/（与现有黑匣子一致）。

"""

from __future__ import annotations
import os
import json
import time
import uuid
import random
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

# 可选依赖：若没有 requests 也能跑（会走模拟数据）
try:
    import requests  # type: ignore
except Exception:
    requests = None

# ---------------------------------------------------------------------
# 路径与配置
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIT_DIR = os.path.join(ROOT_DIR, "data", "objects", "audit")
CONFIG_DIR = os.path.join(ROOT_DIR, "data", "objects", "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "tos_client.json")
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts_iso(dt: datetime) -> str:
    # 输出 ISO8601（UTC）
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class _Cfg:
    """配置加载与默认模板。支持环境变量覆盖。"""
    def __init__(self, path: str = CONFIG_FILE):
        self.path = path
        if not os.path.exists(self.path):
            self._write_default()
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # 环境变量覆盖（可选）
        self.data["base_url"] = os.getenv("PORT_TOS_BASE", self.data.get("base_url", ""))
        tok_env = os.getenv("PORT_TOS_TOKEN")
        if tok_env:
            self.data.setdefault("auth", {})["token"] = tok_env

    def _write_default(self):
        default = {
            "site_code": "INTL-MEGA-PORT",
            "base_url": "",  # 真实 TOS REST 根地址（空则走模拟）
            "auth": {"token": "REPLACE_WITH_TOS_TOKEN"},
            "paths": {
                "vessel_calls": "/api/tos/v1/vessel_calls",
                "berth_plan": "/api/tos/v1/berths",
                "crane_plan": "/api/tos/v1/crane_plan",
                "move_orders": "/api/tos/v1/move_orders",
                "yard_inventory": "/api/tos/v1/yard_inventory",
                "truck_appointments": "/api/tos/v1/truck_appointments"
            },
            "timeout_sec": 3,
            "fallback_mock": False,
            "mock_seed": 1729,
            "tz_offset_min": 0  # 若现场返回本地时区，可设偏移（分钟）
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 归一化数据结构（dataclass）
# ---------------------------------------------------------------------
@dataclass
class VesselCall:
    call_id: str
    vessel_name: str
    imo: Optional[str]
    mmsi: Optional[str]
    service: Optional[str]
    eta_utc: str
    etd_utc: str
    berth_id: Optional[str]
    quay_cranes_planned: int
    remark: Optional[str] = None

@dataclass
class BerthWindow:
    berth_id: str
    start_utc: str
    end_utc: str
    vessel_call_id: Optional[str] = None
    quay: Optional[str] = None

@dataclass
class CranePlan:
    call_id: str
    qc_id: str
    shift_start_utc: str
    shift_end_utc: str
    target_moves: int

@dataclass
class MoveOrder:
    order_id: str
    call_id: Optional[str]
    type: str              # load|discharge|relocate|gate_in|gate_out
    container: str
    size: str              # 20|40|45...
    iso: Optional[str]
    from_loc: Optional[str]
    to_loc: Optional[str]
    eqp: List[str]         # 设备类型，如 ["QC","AGV","YC"]
    est_kwh: float         # 该作业估计能耗（便于能耗预测/归集）
    ts_plan_utc: str
    ts_done_utc: Optional[str] = None
    status: str = "PLANNED" # PLANNED|INPROGRESS|DONE|CANCELLED

@dataclass
class YardBlock:
    block_id: str
    teus: int
    reefer: int
    full: int
    empty: int
    heat: float            # 热度（用于堆场热区可视化）

@dataclass
class TruckAppt:
    appt_id: str
    license: str
    op: str                # in|out
    container: Optional[str]
    time_utc: str


# ---------------------------------------------------------------------
# TOS 客户端（真实请求 + 模拟数据）
# ---------------------------------------------------------------------
class TOSClient:
    """
    统一对接 TOS/WMS：
    - 若配置了 base_url 且可通，走真实 REST；
    - 否则走内置模拟器，生成符合口径的数据（可控 seed）。
    """
    def __init__(self, config_path: str = CONFIG_FILE):
        self.cfg = _Cfg(config_path)
        random.seed(self.cfg.data.get("mock_seed", 1729))
        self._cache: Dict[str, Any] = {}

    def source_status(self) -> Dict[str, Any]:
        configured = bool(str(self.cfg.data.get("base_url") or "").strip())
        return {
            "adapter": "tos_rest",
            "mode": "live_rest" if self._use_real() else "engineering_simulator",
            "configured": configured,
            "http_runtime_available": requests is not None,
            "fallback_on_live_error": bool(self.cfg.data.get("fallback_mock", False)),
            "site_code": self.cfg.data.get("site_code"),
        }

    # ------------------ 公共方法（归一化输出） ------------------
    def vessel_calls(self, date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("vessel_calls", params={
                    "from": _ts_iso(date_from), "to": _ts_iso(date_to)
                })
                return [self._norm_vessel(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(v) for v in self._mock_vessel_calls(date_from, date_to)]

    def berth_plan(self, date: datetime) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("berth_plan", params={"date": _ts_iso(date)})
                return [self._norm_berth(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(b) for b in self._mock_berth_plan(date)]

    def crane_plan(self, call_id: str) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("crane_plan", params={"call_id": call_id})
                return [self._norm_crane(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(c) for c in self._mock_crane_plan(call_id)]

    def move_orders(self, date_from: datetime, date_to: datetime, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                params = {"from": _ts_iso(date_from), "to": _ts_iso(date_to)}
                if status: params["status"] = status
                items = self._http_get("move_orders", params=params)
                return [self._norm_move(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(m) for m in self._mock_move_orders(date_from, date_to, status=status)]

    def yard_inventory(self) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("yard_inventory", params={})
                return [self._norm_yard(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(y) for y in self._mock_yard_inventory()]

    def truck_appointments(self, date: datetime) -> List[Dict[str, Any]]:
        if self._use_real():
            try:
                items = self._http_get("truck_appointments", params={"date": _ts_iso(date)})
                return [self._norm_truck(x) for x in items]
            except Exception:
                if not self.cfg.data.get("fallback_mock", True):
                    raise
        return [asdict(t) for t in self._mock_truck_appointments(date)]

    # ------------------ 帮助方法：HTTP / 归一化 ------------------
    def _use_real(self) -> bool:
        base = self.cfg.data.get("base_url", "").strip()
        return bool(base) and requests is not None

    def _http_get(self, key: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = self.cfg.data.get("base_url", "").rstrip("/")
        path = self.cfg.data.get("paths", {}).get(key, "")
        if not base or not path:
            raise RuntimeError("TOS base_url 或 path 未配置")
        url = f"{base}{path}"
        headers = {"Accept": "application/json"}
        token = self.cfg.data.get("auth", {}).get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        timeout = int(self.cfg.data.get("timeout_sec", 3))
        r = requests.get(url, params=params, headers=headers, timeout=timeout)  # type: ignore
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "items" in data:
            return data["items"]  # 常见分页封装
        if isinstance(data, list):
            return data
        return [data]

    def _apply_tz_offset(self, iso_str: str) -> str:
        """
        若现场返回为本地时间，配置 tz_offset_min 后可统一转为 UTC ISO。
        这里假设传入已是 UTC；若配置了偏移，则做修正。
        """
        try:
            off = int(self.cfg.data.get("tz_offset_min", 0))
            if off == 0:
                return iso_str
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return _ts_iso(dt - timedelta(minutes=off))
        except Exception:
            return iso_str

    # ------------------ 归一化器（针对真实 TOS 字段差异做映射） ------------------
    def _norm_vessel(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(VesselCall(
            call_id=str(x.get("call_id") or x.get("visitId") or x.get("id") or uuid.uuid4()),
            vessel_name=str(x.get("vessel_name") or x.get("name") or "UNKNOWN"),
            imo=(x.get("imo") or x.get("imoNo")),
            mmsi=(x.get("mmsi") or None),
            service=(x.get("service") or x.get("line") or None),
            eta_utc=self._apply_tz_offset(str(x.get("eta_utc") or x.get("eta") or _ts_iso(_utc_now()))),
            etd_utc=self._apply_tz_offset(str(x.get("etd_utc") or x.get("etd") or _ts_iso(_utc_now() + timedelta(hours=12)))),
            berth_id=(x.get("berth_id") or x.get("berth") or None),
            quay_cranes_planned=int(x.get("qc_plan") or x.get("quayCranes") or 4),
            remark=(x.get("remark") or None),
        ))

    def _norm_berth(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(BerthWindow(
            berth_id=str(x.get("berth_id") or x.get("id") or "B0"),
            start_utc=self._apply_tz_offset(str(x.get("start_utc") or x.get("start") or _ts_iso(_utc_now()))),
            end_utc=self._apply_tz_offset(str(x.get("end_utc") or x.get("end") or _ts_iso(_utc_now() + timedelta(hours=8)))),
            vessel_call_id=(x.get("call_id") or x.get("visitId") or None),
            quay=(x.get("quay") or None),
        ))

    def _norm_crane(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(CranePlan(
            call_id=str(x.get("call_id") or x.get("visitId") or x.get("id") or uuid.uuid4()),
            qc_id=str(x.get("qc_id") or x.get("qc") or "QC-1"),
            shift_start_utc=self._apply_tz_offset(str(x.get("shift_start_utc") or x.get("start") or _ts_iso(_utc_now()))),
            shift_end_utc=self._apply_tz_offset(str(x.get("shift_end_utc") or x.get("end") or _ts_iso(_utc_now() + timedelta(hours=4)))),
            target_moves=int(x.get("target_moves") or x.get("moves") or random.randint(300, 800)),
        ))

    def _norm_move(self, x: Dict[str, Any]) -> Dict[str, Any]:
        eqp = x.get("eqp") or x.get("equipment") or []
        if isinstance(eqp, str):
            eqp = [eqp]
        est = x.get("est_kwh")
        if est is None:
            # 简化估算：装/卸 ~1.2kWh/TEU，堆存重定位 ~0.4，闸门 ~0.2（具体值落地再校准）
            base = {"load":1.2, "discharge":1.2, "relocate":0.4, "gate_in":0.2, "gate_out":0.2}.get(str(x.get("type","")).lower(), 0.5)
            size = str(x.get("size") or "20")
            mult = 2 if size == "40" else 1
            est = round(base * mult, 3)
        ts_done = x.get("ts_done_utc") or x.get("doneAt")
        return asdict(MoveOrder(
            order_id=str(x.get("order_id") or x.get("id") or uuid.uuid4()),
            call_id=(x.get("call_id") or x.get("visitId") or None),
            type=str(x.get("type") or x.get("op") or "relocate").lower(),
            container=str(x.get("container") or x.get("cntr") or f"{random.randint(100000,999999)}CNTR"),
            size=str(x.get("size") or "20"),
            iso=(x.get("iso") or None),
            from_loc=(x.get("from") or x.get("from_loc") or None),
            to_loc=(x.get("to") or x.get("to_loc") or None),
            eqp=list(eqp),
            est_kwh=float(est),
            ts_plan_utc=self._apply_tz_offset(str(x.get("ts_plan_utc") or x.get("planAt") or _ts_iso(_utc_now()))),
            ts_done_utc=self._apply_tz_offset(str(ts_done)) if ts_done else None,
            status=str(x.get("status") or "PLANNED").upper()
        ))

    def _norm_yard(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(YardBlock(
            block_id=str(x.get("block_id") or x.get("block") or "Y-A1"),
            teus=int(x.get("teus") or 0),
            reefer=int(x.get("reefer") or 0),
            full=int(x.get("full") or 0),
            empty=int(x.get("empty") or 0),
            heat=float(x.get("heat") or 0.0)
        ))

    def _norm_truck(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(TruckAppt(
            appt_id=str(x.get("appt_id") or x.get("id") or uuid.uuid4()),
            license=str(x.get("license") or x.get("plate") or "UNKNOWN"),
            op=str(x.get("op") or x.get("type") or "in"),
            container=(x.get("container") or x.get("cntr") or None),
            time_utc=self._apply_tz_offset(str(x.get("time_utc") or x.get("time") or _ts_iso(_utc_now()))),
        ))

    # -----------------------------------------------------------------
    # 模拟数据（逼真口径，供无 TOS 时联调/演示/CI）
    # -----------------------------------------------------------------
    def _mock_vessel_calls(self, date_from: datetime, date_to: datetime) -> List[VesselCall]:
        out: List[VesselCall] = []
        span = (date_to - date_from).days + 1
        for d in range(span):
            day = (date_from + timedelta(days=d)).replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            n = random.randint(1, 3)  # 每天 1-3 艘
            for k in range(n):
                eta = day + timedelta(hours=random.choice([0, 6, 12, 18]))
                duration_h = random.choice([8, 12, 16, 20])
                etd = eta + timedelta(hours=duration_h)
                out.append(VesselCall(
                    call_id=f"CALL-{eta.strftime('%m%d')}-{k}",
                    vessel_name=random.choice(["COSCO STAR","MSC AURORA","MAERSK HORIZON","CMA CGM JADE"]),
                    imo=str(9000000 + random.randint(0, 9999)),
                    mmsi=str(412000000 + random.randint(0, 99999)),
                    service=random.choice(["AE1","FAL3","TPX","AS1"]),
                    eta_utc=_ts_iso(eta),
                    etd_utc=_ts_iso(etd),
                    berth_id=random.choice(["B01","B02","B03","B04"]),
                    quay_cranes_planned=random.randint(3, 6),
                    remark=None
                ))
        return out

    def _mock_berth_plan(self, date: datetime) -> List[BerthWindow]:
        calls = self._mock_vessel_calls(date - timedelta(days=1), date + timedelta(days=1))
        out: List[BerthWindow] = []
        for v in calls:
            start = datetime.fromisoformat(v.eta_utc.replace("Z", "+00:00"))
            end = datetime.fromisoformat(v.etd_utc.replace("Z", "+00:00"))
            out.append(BerthWindow(
                berth_id=v.berth_id or "B01",
                start_utc=_ts_iso(start),
                end_utc=_ts_iso(end),
                vessel_call_id=v.call_id,
                quay=random.choice(["Q1","Q2","Q3"])
            ))
        return out

    def _mock_crane_plan(self, call_id: str) -> List[CranePlan]:
        start = _utc_now().replace(minute=0, second=0, microsecond=0)
        out: List[CranePlan] = []
        for i in range(random.randint(3, 5)):
            st = start + timedelta(hours=i*4)
            out.append(CranePlan(
                call_id=call_id,
                qc_id=f"QC-{i+1}",
                shift_start_utc=_ts_iso(st),
                shift_end_utc=_ts_iso(st + timedelta(hours=4)),
                target_moves=random.randint(300, 800)
            ))
        return out

    def _mock_move_orders(self, date_from: datetime, date_to: datetime, status: Optional[str]=None) -> List[MoveOrder]:
        out: List[MoveOrder] = []
        n = random.randint(80, 140)  # 一天几十到上百个作业
        for i in range(n):
            op = random.choices(["load","discharge","relocate","gate_in","gate_out"], weights=[25,25,35,7,8])[0]
            size = random.choice(["20","40"])
            eqp = ["QC","AGV","YC"] if op in ("load","discharge") else (["YC","TT"] if op=="relocate" else ["GATE"])
            plan = date_from + (date_to - date_from) * random.random()
            done = plan + timedelta(minutes=random.randint(3, 40))
            st = "PLANNED" if random.random()<0.2 else ("INPROGRESS" if random.random()<0.2 else "DONE")
            if status and st != status.upper():
                continue
            iso = "22G1" if size == "20" else "42G1"
            out.append(MoveOrder(
                order_id=f"M-{int(time.time())}-{i}",
                call_id=None,
                type=op,
                container=f"TCNU{random.randint(100000,999999)}",
                size=size,
                iso=iso,
                from_loc=random.choice(["Y-A1","Y-B3","Q1","G-IN","G-OUT","STACK"]),
                to_loc=random.choice(["Y-A2","Y-C4","Q2","G-IN","G-OUT","STACK"]),
                eqp=eqp,
                est_kwh=round((1.2 if op in ("load","discharge") else (0.4 if op=="relocate" else 0.2)) * (2 if size=="40" else 1), 3),
                ts_plan_utc=_ts_iso(plan),
                ts_done_utc=_ts_iso(done) if st=="DONE" else None,
                status=st
            ))
        return out

    def _mock_yard_inventory(self) -> List[YardBlock]:
        out: List[YardBlock] = []
        for b in ["Y-A1","Y-A2","Y-B3","Y-C4","Y-D1","Y-E2","Y-F3"]:
            teus = random.randint(120, 480)
            full = int(teus * random.uniform(0.5, 0.8))
            empty = teus - full
            reefer = int(full * random.uniform(0.05, 0.15))
            heat = round(random.uniform(0.2, 0.95), 3)
            out.append(YardBlock(block_id=b, teus=teus, reefer=reefer, full=full, empty=empty, heat=heat))
        return out

    def _mock_truck_appointments(self, date: datetime) -> List[TruckAppt]:
        out: List[TruckAppt] = []
        base = date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        for i in range(random.randint(10, 40)):
            tt = base + timedelta(minutes=random.randint(0, 24*60-1))
            out.append(TruckAppt(
                appt_id=f"APT-{base.strftime('%m%d')}-{i}",
                license=f"粤B{random.randint(10000,99999)}",
                op=random.choice(["in","out"]),
                container=(f"TRLU{random.randint(100000,999999)}" if random.random()<0.7 else None),
                time_utc=_ts_iso(tt)
            ))
        return out

    # -----------------------------------------------------------------
    # 自测：抓取/模拟并落证据（可直接运行，不依赖 server）
    # -----------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        today = _utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        vs = self.vessel_calls(today, today + timedelta(days=1))
        bp = self.berth_plan(today)
        mo = self.move_orders(today, today + timedelta(days=1))
        yi = self.yard_inventory()
        ta = self.truck_appointments(today)
        out = {
            "site": self.cfg.data.get("site_code"),
            "ts_utc": _ts_iso(_utc_now()),
            "vessel_calls": vs,
            "berth_plan": bp,
            "move_orders": mo[:50],  # 防止太大
            "yard_inventory": yi,
            "truck_appointments": ta[:50]
        }
        return out

    def save_audit(self, payload: Dict[str, Any]) -> str:
        ts = int(time.time())
        fn = os.path.join(AUDIT_DIR, f"evt-tos-{ts}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return fn


# ---------------------------------------------------------------------
# 便捷自测命令：python -c "from app.adapters.tos_client import demo_self_test; demo_self_test()"
# ---------------------------------------------------------------------
def demo_self_test() -> None:
    """
    不改 server 的前提下，快速验证兼容性与审计落盘：
    1) 自动生成/读取配置 data/objects/config/tos_client.json；
    2) 抓取/模拟 TOS 数据，归一化结构；
    3) 落证据到 data/objects/audit/evt-tos-*.json 并在控制台打印关键统计。
    """
    cli = TOSClient()
    snap = cli.snapshot()
    path = cli.save_audit(snap)

    # 控制台输出（方便你目测）
    vc = len(snap.get("vessel_calls", []))
    mo = len(snap.get("move_orders", []))
    yi = len(snap.get("yard_inventory", []))
    ta = len(snap.get("truck_appointments", []))
    print("[TOS] snapshot@{}  vessels={}  moves={}  yard_blocks={}  appts={}".format(
        snap.get("ts_utc"), vc, mo, yi, ta
    ))
    print("证据文件：", path)
