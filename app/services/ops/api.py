# app/services/ops/api.py
from __future__ import annotations
from fastapi import APIRouter, Body, Query, HTTPException
from typing import Any, Dict, List
from datetime import datetime, timedelta
import math
import random

router = APIRouter(tags=["ops"])

# ===== in-memory 状态（demo 可替换为持久层） ====================================
_ADAPTER_MODE = {"openadr":"read", "ocpp":"sim", "opcua":"read", "edifact":"read"}
_APPROVALS: Dict[str, set] = {}   # job_id -> set(approvers)

# ===== 公共小工具 =============================================================
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")+"Z"

def _z(p: float) -> float:
    """标准正态分位函数 Φ^{-1}(p)，Acklam 近似（误差 < 4.5e-4）。"""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    # Acklam constants
    a=[-3.969683028665376e+01,  2.209460984245205e+02,
       -2.759285104469687e+02,  1.383577518672690e+02,
       -3.066479806614716e+01,  2.506628277459239e+00]
    b=[-5.447609879822406e+01,  1.615858368580409e+02,
       -1.556989798598866e+02,  6.680131188771972e+01,
       -1.328068155288572e+01]
    c=[-7.784894002430293e-03, -3.223964580411365e-01,
       -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00,  2.938163982698783e+00]
    d=[ 7.784695709041462e-03,  3.224671290700398e-01,
        2.445134137142996e+00,  3.754408661907416e+00]
    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if phigh < p:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

# ===== 9. O11y & 根因 =========================================================
@router.get("/api/o11y/trace")
def o11y_trace(
    job_id: str = Query("demo"),
    asset: str = Query("agg"),
    port: str = Query("CNYTN")
) -> Dict[str, Any]:
    """
    决策路径（OpenTelemetry 风格简版）：
    感知 -> 特征 -> Q/值函数 -> 动作 -> Guard -> 执行/下发
    """
    base_ts = datetime.utcnow()
    spans = [
        {"id":"sense",   "name":"sense",   "ts":(base_ts).isoformat()+"Z", "dur_ms":42,  "attrs":{"asset":asset,"port":port}},
        {"id":"feature", "name":"feature", "ts":(base_ts+timedelta(ms=50)).isoformat()+"Z", "dur_ms":18,  "attrs":{"select":["price","queue_len","hour_onehot"]}},
        {"id":"qvalue",  "name":"q_value", "ts":(base_ts+timedelta(ms=70)).isoformat()+"Z", "dur_ms":21,  "attrs":{"algo":"dueling-dqn","q_max":1.83}},
        {"id":"action",  "name":"action",  "ts":(base_ts+timedelta(ms=92)).isoformat()+"Z", "dur_ms":9,   "attrs":{"intensity":0.42,"unit":"kW/ratio"}},
        {"id":"guard",   "name":"guard",   "ts":(base_ts+timedelta(ms=102)).isoformat()+"Z","dur_ms":7,   "attrs":{"peak_limit_kW":500,"hit":False}},
        {"id":"exec",    "name":"dispatch","ts":(base_ts+timedelta(ms=112)).isoformat()+"Z","dur_ms":35,  "attrs":{"job_id":job_id,"mode":"dry-run"}}
    ]
    edges = [
        {"from":"sense","to":"feature"},
        {"from":"feature","to":"qvalue"},
        {"from":"qvalue","to":"action"},
        {"from":"action","to":"guard"},
        {"from":"guard","to":"exec"},
    ]
    return {"job_id": job_id, "asset": asset, "port": port, "spans": spans, "edges": edges}

@router.get("/api/rca/topology")
def rca_topology(
    asset: str = Query("agg"),
    cuped: int = Query(1)
) -> Dict[str, Any]:
    """
    异常→指标→分桶拓扑；演示口径：价格上浮与靠泊密度变化导致越峰概率升高。
    """
    rnd = random.Random(hash(asset) & 0xffffffff)
    def row(level, node, eff, msg):
        return {"level":level, "node":node, "effect":round(eff, 3), "evidence":msg}
    rows = [
        row("anomaly","peak_risk", +0.23, "越峰概率↑（15min MA 超阈）"),
        row("metric","market.price", +0.11, "近 6h 电价均值+P90 上移"),
        row("metric","ops.berth_density", +0.07, "近 3h 靠泊密度↑"),
        row("bucket","price.bin[0.70~0.85]", +0.05, "高价分桶覆盖↑"),
    ]
    if cuped:
        # CUPED 当作方差缩减（给出 R^2）
        r2 = 0.30 + rnd.random()*0.1
    else:
        r2 = 0.0
    return {"asset":asset, "cuped_r2":round(r2,3), "items":rows}

# ===== 10. 数据治理与质量契约 ================================================
@router.post("/api/dq/contract/check")
def dq_contract_check(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    输入：
      { dataset_id, schema: { step_sec, unit, completeness_threshold } }
    输出：ok / details[] / 建议 action（失败时）
    """
    ds  = str(payload.get("dataset_id") or "telemetry.active_power_kw")
    sch = payload.get("schema") or {}
    step = int(sch.get("step_sec") or 60)
    unit = str(sch.get("unit") or "kW")
    th   = float(sch.get("completeness_threshold") or 0.96)

    # 演示：随机出一个完整率、单位/步长回显
    completeness = 0.93 + (hash(ds) % 7) * 0.01   # 0.93 ~ 0.99
    ok = completeness >= th
    details = [
        {"field":"step_sec", "expected":step, "observed":step, "ok": True},
        {"field":"unit",     "expected":unit, "observed":unit, "ok": True},
        {"field":"completeness", "expected": th, "observed": round(completeness,3), "ok": ok},
    ]
    action = None
    if not ok:
        # 失败：建议自动降级或切换基线
        action = "degrade: switch_baseline or freeze_rollout"

    return {"dataset": ds, "ok": ok, "details": details, "action": action, "ts": _now_iso()}

# ===== 11. 标准化对接层（OpenADR / OCPP / OPC-UA / EDIFACT）==================
@router.get("/api/adapters/capabilities")
def adapters_cap(port: str = Query("CNYTN")) -> Dict[str, Any]:
    items = [
        {"id":"openadr", "protocol":"OpenADR", "title":"需量/削峰事件", "assets":["PCC","BESS"], "cap":["read","sim","live"], "mode": _ADAPTER_MODE["openadr"]},
        {"id":"ocpp",    "protocol":"OCPP",    "title":"充电桩会话/功率", "assets":["EVSE"],    "cap":["read","sim","live"], "mode": _ADAPTER_MODE["ocpp"]},
        {"id":"opcua",   "protocol":"OPC-UA",  "title":"变电/测点（OPC-UA/IEC 61850）", "assets":["Substation"], "cap":["read","sim"], "mode": _ADAPTER_MODE["opcua"]},
        {"id":"edifact", "protocol":"EDIFACT", "title":"TOS（DCSA/EDIFACT）", "assets":["Vessel","Berth"], "cap":["read"], "mode": _ADAPTER_MODE["edifact"]},
    ]
    return {"port":port, "items":items, "ts":_now_iso()}

@router.post("/api/adapters/mode")
def adapters_mode(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    aid = str(payload.get("id") or "")
    mode = str(payload.get("mode") or "read")
    if aid not in _ADAPTER_MODE or mode not in ("read","sim","live"):
        raise HTTPException(status_code=400, detail="bad adapter/mode")
    _ADAPTER_MODE[aid] = mode
    return {"ok":True, "id":aid, "mode":mode}

# ===== 12. 权限/合规（细粒度 RBAC + 双人审核）===============================
@router.get("/api/rbac/whoami")
def rbac_whoami(user: str = Query("auditor")) -> Dict[str, Any]:
    roles = ["auditor"]
    if user.startswith("ops"): roles.append("ops")
    if user.startswith("approver"): roles.append("approver")
    perms = {
        "asset_scope": "CNYTN:*",
        "approve_limit_cny": 50000,
        "can_dual_approve": True,
        "policies": ["view", "abtest", "approve", "export"]
    }
    return {"user": user, "roles": roles, "permissions": perms, "ts": _now_iso()}

@router.post("/api/rbac/approve")
def rbac_approve(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    双人审核：同一 job_id 收到两个不同 approver 才算通过。
    """
    job = str(payload.get("job_id") or "")
    apr = str(payload.get("approver") or "")
    if not job or not apr:
        raise HTTPException(status_code=400, detail="need job_id & approver")
    s = _APPROVALS.setdefault(job, set())
    s.add(apr)
    return {"job_id": job, "approvers": sorted(list(s)), "approved": len(s) >= 2, "ts": _now_iso()}

# ===== 13. A/B 设计强化（样本量 + 监控阈值 + CUPED/分层）======================
@router.post("/api/ab/design")
def ab_design(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    输入：
      alpha, power, mdes_sigma(Δ/σ), sigma, r2_cuped, allocation(流量到 B), segments[], exclusions[]
    输出：
      n_per_arm, total_n, design_effect, thresholds 等
    """
    alpha = float(payload.get("alpha", 0.05))
    power = float(payload.get("power", 0.80))
    mdes  = float(payload.get("mdes_sigma", 0.05))   # 以 σ 为单位
    sigma = float(payload.get("sigma", 1.0))
    r2    = max(0.0, min(0.99, float(payload.get("r2_cuped", 0.3))))
    alloc = float(payload.get("allocation", 0.5))     # B 组比例
    segs  = payload.get("segments") or []
    excls = payload.get("exclusions") or []

    # 方差缩减：CUPED 相当于乘以 (1 - R^2)
    var_reduction = (1.0 - r2)
    # 分层（粗略）设计效应：权重不均会轻微上升，这里用 sum(w^2) 近似
    if segs:
        sw2 = sum((float(s.get("weight", 0.0)) or 0.0)**2 for s in segs) or 1.0
    else:
        sw2 = 1.0
    design_effect = var_reduction * (0.9 + 0.2*sw2)  # 经验近似：均衡分层 ~0.9，极端不均衡→~1.1

    z_alpha = _z(1 - alpha/2.0)
    z_beta  = _z(power)
    delta   = mdes * sigma
    # 不等分流量样本量（两独立样本）：n_A = k * (1-alloc)， n_B = k * alloc
    # 令 k = ((zα+zβ)^2 * σ^2 * (1/alloc + 1/(1-alloc))) / Δ^2
    k = ((z_alpha + z_beta)**2 * (sigma**2) * (1/alloc + 1/(1-alloc))) / (delta**2)
    k = k * design_effect
    nA = math.ceil(k * (1-alloc))
    nB = math.ceil(k * alloc)

    # 监控阈值：简化给 P50 ± 1.96 * σ/√n，各臂独立
    thrA = 1.96 * sigma / math.sqrt(max(1,nA))
    thrB = 1.96 * sigma / math.sqrt(max(1,nB))

    return {
        "assumptions": {
            "alpha": alpha, "power": power, "sigma": sigma,
            "mdes_sigma": mdes, "cuped_r2": r2, "allocation_B": alloc
        },
        "design_effect": round(design_effect, 3),
        "sample_size": {"A": nA, "B": nB, "total": nA + nB},
        "monitoring_thresholds": {"A_delta_sigma": round(thrA/sigma, 3),
                                  "B_delta_sigma": round(thrB/sigma, 3)},
        "segments": segs, "exclusions": excls,
        "ts": _now_iso()
    }

# ===== 对外：供主服务挂载 ======================================================
def get_router() -> APIRouter:
    return router
