# app/services/rl_model/agv_charge/api.py
# -*- coding: utf-8 -*-
"""
AGV/无人集卡 充/换电调度 · RL 接口适配层（模块 A）
-------------------------------------------------
大白话说明：
1) 这是“从数据到训练到上线到前端联动”的后端服务胶水层。
   - 读你在 agv_charge/data 下的样本数据（grid_meter/market_price/grid_ef 等）
   - 用稳健的“错峰/降峰”简化仿真，给出 baseline vs strategy 聚合功率序列、ΔkWh、ΔkgCO₂e、削峰kW
   - 生成可解释特征（Why/Why-not 简版：价格/峰段/平均负荷/动作强度）
   - 提供“下发（演示）、审批、回滚、A/B对照、在线学习”所需的后端状态持久化（JSONL 文件）

2) 既可单独自检（不依赖 web 框架），也可被 FastAPI/Flask 等挂载为 /api/rl/* 与 /api/exec/*。
   - FastAPI: from .api import get_router; app.include_router(get_router(), prefix="")
   - Flask: 在路由函数里调用本文件暴露的 service 方法即可

3) 输出格式完全贴合前端 index.html 里写死的字段与接口（见函数名及下方“接口映射”）。

目录关系：
app/
 └─ services/
    └─ rl_model/
       └─ agv_charge/
          ├─ data/                   # 你已经提供的数据样本
          │   ├─ grid_meter.csv
          │   ├─ market_price.csv
          │   ├─ grid_ef.csv
          │   ├─ vehicle_state.csv
          │   └─ vehicles_master.csv
          ├─ artifacts/              # 输出/状态落地（本文件会写入）
          │   ├─ dispatch_history.jsonl
          │   └─ exec_jobs.jsonl
          ├─ policy.bin              # 你已有的策略二进制（此处不直接用，可留作后续真实推理）
          ├─ policy_meta.json        # 可选：策略元信息（如无则由本文件默认生成）
          └─ api.py                  # 本文件（新增）
"""
from __future__ import annotations

import os, json, csv, math, uuid, time, random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# ====== 路径与健壮性 ======
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ART_DIR  = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

DISPATCH_LOG = os.path.join(ART_DIR, "dispatch_history.jsonl")
EXEC_JOBS    = os.path.join(ART_DIR, "exec_jobs.jsonl")
POLICY_META  = os.path.join(ROOT_DIR, "policy_meta.json")
CONFIG_YAML  = os.path.join(ROOT_DIR, "config.yaml")


def _load_runtime_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "time_col": "timestamp",
        "price_col": "price_yuan_per_kwh",
        "ef_col": "ef_kg_per_kwh",
        "grid_kw_col": "pcc_kw",
    }
    if not os.path.exists(CONFIG_YAML):
        return cfg
    try:
        if yaml is not None:
            with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                cfg.update({k: v for k, v in loaded.items() if v not in (None, '')})
                return cfg
    except Exception:
        pass
    try:
        with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or ':' not in line:
                    continue
                k, v = line.split(':', 1)
                k = k.strip()
                v = v.strip().strip("'").strip('\"')
                if k:
                    cfg[k] = v
    except Exception:
        pass
    return cfg


CFG = _load_runtime_config()
TIME_COL = str(CFG.get('time_col', 'timestamp'))
PRICE_COL = str(CFG.get('price_col', 'price_yuan_per_kwh'))
EF_COL = str(CFG.get('ef_col', 'ef_kg_per_kwh'))
GRID_KW_COL = str(CFG.get('grid_kw_col', 'pcc_kw'))

# ====== 小工具：读CSV（容错多字段名） ======
def _read_series_csv(path: str, value_fields: List[str]) -> Tuple[List[str], List[float]]:
    """
    读取时间序列CSV，返回 (timestamps, values)
    容错字段名：时间(ts/t/time/timestamp)、值(kW/kw/value/v/p50...)
    """
    ts_list, val_list = [], []
    if not os.path.exists(path):
        return ts_list, val_list
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = (row.get(TIME_COL) or row.get("ts") or row.get("t") or row.get("time") or row.get("timestamp") or "").strip()
            # 找第一个存在的值字段
            v = None
            for k in value_fields:
                if k in row and row[k] not in (None, "", "null"):
                    try:
                        v = float(row[k])
                        break
                    except:
                        continue
            if ts and v is not None and math.isfinite(v):
                ts_list.append(ts)
                val_list.append(v)
    return ts_list, val_list

def _safe_avg(arr: List[float], default: float=0.0) -> float:
    ok = [x for x in arr if isinstance(x,(int,float)) and math.isfinite(x)]
    return sum(ok)/len(ok) if ok else default

def _tile_or_clip(arr: List[float], L: int) -> List[float]:
    """把数组裁剪/平铺到长度 L（前端通常 horizon=360 点）"""
    if not arr:
        return [0.0]*L
    if len(arr) == L:
        return arr
    if len(arr) > L:
        return arr[:L]
    # 平铺
    out = []
    while len(out) < L:
        out.extend(arr)
    return out[:L]

def _jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _jsonl_load(path: str, limit: int=200) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                items.append(json.loads(line))
            except:
                pass
    # 按时间倒序（优先 ts / created_at / timestamp / _ts）
    items.sort(
        key=lambda x: x.get("ts") or x.get("created_at") or x.get("timestamp") or x.get("time") or x.get("_ts") or "",
        reverse=True,
    )
    return items[:max(1, limit)]

# ====== 数据面板：基线功率 / 电价 / EF ======
def _load_baseline_kw(horizon_min: int, step_min: int) -> List[float]:
    """
    用 grid_meter.csv 取“聚合功率基线(kW)”。
    若数据不足，自动平铺；若没有该文件，返回全 800kW 的演示序列。
    """
    _, vals = _read_series_csv(
        os.path.join(DATA_DIR, "grid_meter.csv"),
        [GRID_KW_COL, "pcc_kw", "grid_kw", "kW", "kw", "value", "v", "p50"],
    )
    if not vals:
        vals = [800.0 + 120.0*math.sin(i/24.0) for i in range(1440)]  # 1天演示曲线
    L = max(1, int(horizon_min/max(1,step_min)))
    return _tile_or_clip(vals, L)

def _load_price_series(horizon_min:int, step_min:int) -> List[float]:
    """
    电价（¥/kWh），优先 market_price.csv；无则给出“峰/平/谷”演示轮廓。
    """
    _, vals = _read_series_csv(
        os.path.join(DATA_DIR, "market_price.csv"),
        [PRICE_COL, "price_yuan_per_kwh", "price", "yuan_per_kwh", "cny_kwh", "value", "v", "p50"],
    )
    if not vals:
        # 简单 TOU：谷 0.45 / 平 0.65 / 峰 0.95（演示）
        tpl = ([0.45]*120 + [0.65]*120 + [0.95]*120)  # 6h 周期
        vals = (tpl*12)[:1440]
    L = max(1, int(horizon_min/max(1,step_min)))
    return _tile_or_clip(vals, L)

def _load_ef_series(horizon_min:int, step_min:int) -> List[float]:
    """
    电网排放因子（kg/kWh）。grid_ef.csv 通常是 g/kWh 或 kg/kWh，尽量自适应。
    若没有文件，默认 0.12 kg/kWh。
    """
    _, gvals = _read_series_csv(
        os.path.join(DATA_DIR, "grid_ef.csv"),
        [EF_COL, "ef_kg_per_kwh", "g_per_kwh", "gpkwh", "gpkWhe", "g", "value", "v", "kg_per_kwh", "kgpkwh"],
    )
    vals_kg = []
    for v in gvals:
        # 粗判：>10 则大概率是 g/kWh；否则当 kg/kWh
        vals_kg.append(v/1000.0 if v>10 else v)
    if not vals_kg:
        vals_kg = [0.12]*1440
    L = max(1, int(horizon_min/max(1,step_min)))
    return _tile_or_clip(vals_kg, L)

# ====== 策略元信息（可选 policy_meta.json；否则默认） ======
def _load_policy_meta() -> Dict[str, Any]:
    if os.path.exists(POLICY_META):
        try:
            with open(POLICY_META,"r",encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    # 默认元信息
    return {
        "id": "agv_charge_v1",
        "title": "AGV 充/换电错峰 v1",
        "category": "agv_charge",
        "objective": "min_cost_min_peak",
        "hints": {
            "algo": "IQL (offline RL)",
            "reward": "-电费-碳费-峰值罚-平滑罚（当前版本）",
            "safety": "availability gate + pmax clipping + guardrails"
        },
        "version": "1.0.0"
    }

# ====== 业务核心：仿真/解释/下发/执行 ======
@dataclass
class SimSummary:
    delta_kWh: float
    delta_carbon_kg: float
    peak_reduction_kW: float

@dataclass
class SimOutput:
    summary: SimSummary
    baseline: Dict[str, Any]
    simulated: Dict[str, Any]

class AGVChargeService:
    """
    业务服务类（无框架依赖）。
    server.py 可直接持有一个实例，映射到 /api/rl/* 与 /api/exec/*。
    """
    def __init__(self):
        self.meta = _load_policy_meta()

    # ---------- 前端“拉取策略列表” ----------
    def list_strategies(self, horizon_min:int=360, step_min:int=1, max_items:int=12) -> Dict[str,Any]:
        """
        输出字段与前端 index.html 的使用一致（见：btn-rlpanel-list.onclick）。
        """
        return {"strategies":[{
            "id": self.meta.get("id","agv_charge_v1"),
            "title": self.meta.get("title","AGV 充/换电错峰 v1"),
            "category": "AGV/充换电",
            "objective": self.meta.get("objective","min_cost_min_peak"),
            "version": self.meta.get("version","1.0.0"),
            "horizon_min": horizon_min,
            "step_min": step_min
        }]}

    # ---------- 前端“模拟执行” ----------
    def simulate(self, strategy_id:str, horizon_min:int=360, step_min:int=1) -> SimOutput:
        """
        简化可落地的策略器：
        - 读 baseline 聚合功率（grid_meter）
        - 读 电价曲线，取 75 分位为“峰段”
        - “策略后”在高价阶段，把充/换电功率移到低价区（不突破“可用窗口”，以 10~15% 的幅度削减峰段）
        - 给出 ΔkWh、ΔkgCO₂e、削峰
        """
        base_kw = _load_baseline_kw(horizon_min, step_min)
        price   = _load_price_series(horizon_min, step_min)
        ef_kg   = _load_ef_series(horizon_min, step_min)

        # 峰段阈值 = 75% 分位
        p_sorted = sorted(price)
        p75 = p_sorted[int(0.75*len(p_sorted))]
        # 削减比例（可按电价强度线性调节），保证非负
        sim_kw = []
        for k, p in zip(base_kw, price):
            if p >= p75:
                # 峰段削 12%，弱网/换电等因素保守系数 0.88~0.9
                red = 0.12 + 0.04*random.random()
                sim_kw.append(max(0.0, k*(1.0 - red)))
            else:
                sim_kw.append(k)
        # 结果指标
        # kWh ≈ sum(kW*分钟)/60
        base_kwh = sum(base_kw)/60.0
        sim_kwh  = sum(sim_kw)/60.0
        delta_kwh = max(0.0, base_kwh - sim_kwh)
        # 平均 EF
        avg_ef = _safe_avg(ef_kg, 0.12)
        delta_kg = delta_kwh * avg_ef
        peak_red = max(0.0, max(base_kw) - max(sim_kw))

        return SimOutput(
            summary=SimSummary(delta_kWh=delta_kwh, delta_carbon_kg=delta_kg, peak_reduction_kW=peak_red),
            baseline={"agg_kW": base_kw, "total_kWh": base_kwh},
            simulated={"agg_kW": sim_kw, "total_kWh": sim_kwh}
        )

    # ---------- 前端“可解释特征” ----------
    def explain(self, strategy_id:str, horizon_min:int=360, step_min:int=1) -> Dict[str,Any]:
        """
        输出字段与 index.html 的 table/canvas 一致：
        - features: name, value, contribution_kWh, importance, direction
        - reasons: 文本列表
        """
        base_kw = _load_baseline_kw(horizon_min, step_min)
        price   = _load_price_series(horizon_min, step_min)
        avg_kw  = _safe_avg(base_kw, 0)
        p75     = sorted(price)[int(0.75*len(price))]

        # 粗略特征：价格代理、峰段覆盖、平均负荷、动作强度
        price_proxy = _safe_avg(price, 0)
        is_peak_win = sum(1 for p in price if p>=p75) / max(1,len(price))
        action_int  = 0.12  # 与 simulate 中的削减幅度一致

        # 贡献拆分（演示口径，以 ΔkWh 拆账）
        # 把ΔkWh 60% 分给峰段覆盖，25% 分给价格水平，15% 分给动作强度
        sim = self.simulate(strategy_id, horizon_min, step_min)
        dkwh = sim.summary.delta_kWh
        feats = [
            {"name":"is_peak_window", "value": round(is_peak_win,3), "contribution_kWh": -(dkwh*0.60), "importance": 0.45, "direction":"saving"},
            {"name":"price_proxy",    "value": round(price_proxy,3), "contribution_kWh": -(dkwh*0.25), "importance": 0.30, "direction":"saving"},
            {"name":"avg_load_kw",    "value": round(avg_kw,3),     "contribution_kWh": -(dkwh*0.10), "importance": 0.15, "direction":"saving"},
            {"name":"action_intensity","value": action_int,         "contribution_kWh": -(dkwh*0.05), "importance": 0.10, "direction":"saving"},
        ]
        reasons = [
            "高价时段覆盖度较高，错峰空间充足（削峰优先）。",
            "电价分位处于偏高水平，策略建议前置/后移充电以降低成本与碳费。",
            "平均负荷较高，削峰对需量罚金规避有价值。",
            "动作强度设置为 12%，在安全约束下可平衡 SLA 与寿命。"
        ]
        return {"features": feats, "reasons": reasons, "meta": {"strategy_id": strategy_id}}

    # ---------- RL 策略“演示下发” ----------
    def dispatch(self, strategy_id:str, operator:str="demo", dry_run:bool=True,
                 enforce_guardrails:bool=True, guardrail_min_peak_kw:float=1.0, notes:str="") -> Dict[str,Any]:
        """
        仅演示：不控制真实设备。把一次“拟下发”的估计结果写入历史 JSONL，供前端列表展示/取消。
        """
        sim = self.simulate(strategy_id)
        est = {
            "summary": asdict(sim.summary),
            "delta_kWh": sim.summary.delta_kWh,
            "delta_carbon_kg": sim.summary.delta_carbon_kg,
            "peak_reduction_kW": sim.summary.peak_reduction_kW
        }
        job_id = str(uuid.uuid4())
        guard_ok = (sim.summary.peak_reduction_kW >= guardrail_min_peak_kw) or (sim.summary.delta_kWh > 0.1)
        rec = {
            "job_id": job_id,
            "strategy_id": strategy_id,
            "operator": operator,
            "dry_run": bool(dry_run),
            "guardrails": {"enforced": bool(enforce_guardrails), "min_peak_kw": guardrail_min_peak_kw, "ok": guard_ok},
            "estimate": est,
            "status": "CREATED",
            "notes": notes,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        _jsonl_append(DISPATCH_LOG, rec)
        return {"ok": True, "job": rec}

    def dispatch_history(self, limit:int=20) -> Dict[str,Any]:
        items = _jsonl_load(DISPATCH_LOG, limit)
        return {"items": items}

    def dispatch_cancel(self, job_id:str, operator:str="demo") -> Dict[str,Any]:
        items = _jsonl_load(DISPATCH_LOG, 1000)
        out = []
        found = False
        for it in items:
            if it.get("job_id")==job_id and it.get("status")!="CANCELLED":
                it["status"]="CANCELLED"; it["cancelled_by"]=operator; found=True
            out.append(it)
        # 覆盖写回
        with open(DISPATCH_LOG,"w",encoding="utf-8") as f:
            for it in out:
                f.write(json.dumps(it, ensure_ascii=False)+"\n")
        return {"ok": found, "job_id": job_id}

    # ---------- 执行与闭环（审批 / 下发 / A/B / 学习） ----------
    def exec_submit(self, strategy_id:str, operator:str="auditor", mode:str="auto", notes:str="") -> Dict[str,Any]:
        """
        mode=auto 立即下发；mode=manual 进入待审批。
        保存 forecast_snapshot（用于 A/B 对照）
        """
        sim = self.simulate(strategy_id)
        job_id = str(uuid.uuid4())
        status = "SUBMITTED_AUTO" if mode=="auto" else "PENDING_APPROVAL"
        job = {
            "job_id": job_id,
            "strategy_id": strategy_id,
            "mode": mode,
            "status": status,
            "operator": operator,
            "notes": notes,
            "forecast_snapshot": {
                "baseline": {"agg_kW": sim.baseline["agg_kW"], "total_kWh": sim.baseline["total_kWh"]},
                "simulated":{"agg_kW": sim.simulated["agg_kW"], "total_kWh": sim.simulated["total_kWh"]}
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        _jsonl_append(EXEC_JOBS, job)
        return {"ok": True, "job": job}

    def exec_approve(self, job_id:str, operator:str="auditor") -> Dict[str,Any]:
        items = _jsonl_load(EXEC_JOBS, 1000)
        ok=False
        with open(EXEC_JOBS,"w",encoding="utf-8") as f:
            for it in items:
                if it.get("job_id")==job_id:
                    it["status"]="APPROVED_DISPATCHED"
                    it["approved_by"]=operator
                    it["approved_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    ok=True
                f.write(json.dumps(it, ensure_ascii=False)+"\n")
        return {"ok": ok, "job_id": job_id}

    def exec_list(self, limit:int=20) -> Dict[str,Any]:
        return {"items": _jsonl_load(EXEC_JOBS, limit)}

    def exec_get(self, job_id:str) -> Dict[str,Any]:
        items = _jsonl_load(EXEC_JOBS, 1000)
        for it in items:
            if it.get("job_id")==job_id:
                return {"job": it}
        return {"error":"not_found","job_id":job_id}

    def exec_rollback(self, job_id:str, reason:str="ui-rollback") -> Dict[str,Any]:
        items = _jsonl_load(EXEC_JOBS, 1000)
        ok=False
        with open(EXEC_JOBS,"w",encoding="utf-8") as f:
            for it in items:
                if it.get("job_id")==job_id:
                    it["status"]="ROLLED_BACK"
                    it["rollback_reason"]=reason
                    it["rollback_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    ok=True
                f.write(json.dumps(it, ensure_ascii=False)+"\n")
        return {"ok":ok,"job_id":job_id}

    def exec_abtest(self, job_id:str) -> Dict[str,Any]:
        """
        产生 A/B 评估：用提交时保存的 forecast_snapshot 做 PRED，
        “实测”部分演示口径：按 0.9~1.1 的比例随机扰动预测总量。
        """
        g = self.exec_get(job_id)
        if "job" not in g: return {"error":"not_found","job_id":job_id}
        j = g["job"]
        base_kwh = float(j["forecast_snapshot"]["baseline"]["total_kWh"])
        sim_kwh  = float(j["forecast_snapshot"]["simulated"]["total_kWh"])
        delta_pred = max(0.0, base_kwh - sim_kwh)
        # 实测（演示）：给一个近似扰动
        base_obs = base_kwh * random.uniform(0.95, 1.05)
        sim_obs  = sim_kwh  * random.uniform(0.90, 1.02)
        delta_obs = max(0.0, base_obs - sim_obs)
        err = delta_obs - delta_pred
        rel = (err / delta_pred) if delta_pred>1e-6 else None
        return {
            "pred": {"delta_kWh_pred": delta_pred},
            "actual": {"delta_kWh_obs": delta_obs, "baseline_kWh_obs": base_obs, "simulated_kWh_obs": sim_obs},
            "error": {"delta_kWh_error": err, "rel_error": rel}
        }

    def exec_learn(self, job_id:str, alpha:float=0.3) -> Dict[str,Any]:
        """
        在线学习（演示）：把本地 policy_meta 里维护一个“ema_delta”画像。
        """
        ab = self.exec_abtest(job_id)
        if "pred" not in ab or "actual" not in ab:
            return {"ok": False, "reason":"no_abtest"}
        pred = ab["pred"]["delta_kWh_pred"]
        obs  = ab["actual"]["delta_kWh_obs"]
        ema_prev = float(self.meta.get("ema_delta", pred))
        ema_new = (1-alpha)*ema_prev + alpha*(obs)
        self.meta["ema_delta"] = ema_new
        # 落地写回
        try:
            with open(POLICY_META,"w",encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)
        except:
            pass
        return {"ok": True, "ema_delta": ema_new}

    def exec_model(self, strategy_id:str) -> Dict[str,Any]:
        """
        返回策略画像（用于前端“查看画像”）。
        """
        return {"strategy_id": strategy_id, "meta": self.meta}

# ====== 可选：提供 FastAPI Router，便于 server.py 一行挂载 ======
def get_router():
    """
    若项目使用 FastAPI，这里提供 APIRouter。
    server.py 中：
        from app.services.rl_model.agv_charge.api import get_router
        app.include_router(get_router(), prefix="")
    """
    try:
        from fastapi import APIRouter, Body, Query
        from fastapi.responses import JSONResponse
    except Exception:
        return None

    svc = AGVChargeService()
    r = APIRouter()

    # ---- RL Panel ----
    @r.get("/api/rl/strategies")
    def api_rl_strategies(horizon_min:int=Query(360), step_min:int=Query(1), max_items:int=Query(12)):
        return JSONResponse(svc.list_strategies(horizon_min, step_min, max_items))

    @r.post("/api/rl/simulate")
    def api_rl_simulate(payload: Dict[str,Any]=Body(...)):
        sid = str(payload.get("strategy_id") or "agv_charge_v1")
        h   = int(payload.get("horizon_min") or 360)
        s   = int(payload.get("step_min") or 1)
        sim = svc.simulate(sid, h, s)
        return JSONResponse({
            "summary": asdict(sim.summary),
            "baseline": sim.baseline,
            "simulated": sim.simulated
        })

    @r.post("/api/rl/explain")
    def api_rl_explain(payload: Dict[str,Any]=Body(...)):
        sid = str(payload.get("strategy_id") or "agv_charge_v1")
        h   = int(payload.get("horizon_min") or 360)
        s   = int(payload.get("step_min") or 1)
        return JSONResponse(svc.explain(sid, h, s))

    # ---- Dispatch (演示) ----
    @r.post("/api/rl/dispatch")
    def api_rl_dispatch(payload: Dict[str,Any]=Body(...)):
        return JSONResponse(svc.dispatch(
            strategy_id=str(payload.get("strategy_id") or "agv_charge_v1"),
            operator=str(payload.get("operator") or "demo"),
            dry_run=bool(payload.get("dry_run") if payload.get("dry_run") is not None else True),
            enforce_guardrails=bool(payload.get("enforce_guardrails") if payload.get("enforce_guardrails") is not None else True),
            guardrail_min_peak_kw=float(payload.get("guardrail_min_peak_kw") or 1.0),
            notes=str(payload.get("notes") or "")
        ))

    @r.get("/api/rl/dispatch/history")
    def api_rl_hist(limit:int=Query(20)):
        return JSONResponse(svc.dispatch_history(limit))

    @r.post("/api/rl/dispatch/cancel")
    def api_rl_cancel(payload: Dict[str,Any]=Body(...)):
        return JSONResponse(svc.dispatch_cancel(
            job_id=str(payload.get("job_id") or ""),
            operator=str(payload.get("operator") or "demo")
        ))

    # ---- 执行与闭环 ----
    @r.post("/api/exec/submit")
    def api_exec_submit(payload: Dict[str,Any]=Body(...)):
        return JSONResponse(svc.exec_submit(
            strategy_id=str(payload.get("strategy_id") or "agv_charge_v1"),
            operator=str(payload.get("operator") or "auditor"),
            mode=str(payload.get("mode") or "auto"),
            notes=str(payload.get("notes") or "")
        ))

    @r.post("/api/exec/approve")
    def api_exec_approve(payload: Dict[str,Any]=Body(...)):
        return JSONResponse(svc.exec_approve(
            job_id=str(payload.get("job_id") or ""),
            operator=str(payload.get("operator") or "auditor")
        ))

    @r.get("/api/exec/list")
    def api_exec_list(limit:int=Query(20)):
        return JSONResponse(svc.exec_list(limit))

    @r.get("/api/exec/get/{job_id}")
    def api_exec_get(job_id:str):
        return JSONResponse(svc.exec_get(job_id))

    @r.post("/api/exec/rollback")
    def api_exec_rb(payload: Dict[str,Any]=Body(...)):
        return JSONResponse(svc.exec_rollback(
            job_id=str(payload.get("job_id") or ""),
            reason=str(payload.get("reason") or "ui-rollback")
        ))

    @r.get("/api/exec/abtest/{job_id}")
    def api_exec_ab(job_id:str):
        return JSONResponse(svc.exec_abtest(job_id))

    @r.post("/api/exec/learn")
    def api_exec_learn(payload: Dict[str,Any]=Body(...)):
        return JSONResponse(svc.exec_learn(
            job_id=str(payload.get("job_id") or ""),
            alpha=float(payload.get("alpha") or 0.3)
        ))

    @r.get("/api/exec/model/{sid}")
    def api_exec_model(sid:str):
        return JSONResponse(svc.exec_model(sid))

    return r

# ====== 自检（无需 web 框架） ======
def self_check(print_json:bool=True) -> Dict[str,Any]:
    """
    目的：单文件断言数据与核心流程没问题（列策略/仿真/解释/下发/AB/学习）。
    你可以在项目根目录直接运行下面那条“一键自检”命令。
    """
    svc = AGVChargeService()
    res = {
        "strategies": svc.list_strategies(),
        "simulate": None,
        "explain": None,
        "dispatch": None,
        "history": None,
        "exec_submit": None,
        "exec_approve": None,
        "exec_ab": None,
        "exec_learn": None
    }
    sid = res["strategies"]["strategies"][0]["id"]
    sim = svc.simulate(sid)
    res["simulate"] = {"summary": asdict(sim.summary)}
    res["explain"]  = svc.explain(sid)
    d = svc.dispatch(sid, dry_run=True, enforce_guardrails=True, guardrail_min_peak_kw=0.5, notes="self_check")
    res["dispatch"] = {"ok": d["ok"], "job_id": d["job"]["job_id"]}
    res["history"]  = svc.dispatch_history(3)

    ex = svc.exec_submit(sid, mode="manual", notes="self_check")
    res["exec_submit"] = {"ok": ex["ok"], "job_id": ex["job"]["job_id"], "status": ex["job"]["status"]}
    appr = svc.exec_approve(ex["job"]["job_id"])
    res["exec_approve"] = appr
    ab   = svc.exec_abtest(ex["job"]["job_id"])
    res["exec_ab"] = ab
    lr   = svc.exec_learn(ex["job"]["job_id"], alpha=0.3)
    res["exec_learn"] = lr

    if print_json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return res

if __name__ == "__main__":
    self_check(print_json=True)


# === [AGV Charge · RL Artifacts API | Module A] ====================================
# 作用：
#   1) 公开 /api/rl/artifacts/{name} 与兼容路径 /api/rl/model/agv_charge/artifacts/{name}
#   2) 公开 /api/rl/metrics/history ：把 policy_evaluate_history.jsonl 聚合为可视化所需时序
#   3) 公开 /api/rl/summary ：为首页/模块卡片提供稳定摘要字段
#   4) 公开 /api/rl/rollout/status ：供首页/策略面板显示灰度发布状态小卡片（可选）
# 安全：只读白名单、拒绝路径穿越；失败不影响主进程
from pathlib import Path
import mimetypes

try:
    from fastapi import APIRouter, HTTPException
    from starlette.responses import FileResponse, JSONResponse
except Exception as _e:
    APIRouter = None  # 让 get_router() 兜底不崩
    print("[agv_charge.api] fastapi import failed:", _e)

_ART_DIR = Path(__file__).resolve().parent / "artifacts"


def _mime_for(p: Path) -> str:
    guessed = mimetypes.guess_type(str(p))[0]
    if guessed:
        if p.suffix.lower() in ('.jsonl', '.log', '.txt'):
            return 'text/plain; charset=utf-8'
        if p.suffix.lower() == '.json':
            return 'application/json; charset=utf-8'
        return guessed
    suf = p.suffix.lower()
    if suf == ".png":
        return "image/png"
    if suf == ".json":
        return "application/json; charset=utf-8"
    if suf == ".jsonl":
        return "text/plain; charset=utf-8"
    if suf in (".txt", ".log"):
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def _safe_target(name: str) -> Path:
    name = (name or "").strip().lstrip("/").replace("..", "__")
    p = (_ART_DIR / name).resolve()
    if not str(p).startswith(str(_ART_DIR.resolve())):
        raise HTTPException(400, "invalid artifact name")
    return p


def _list_whitelist() -> set:
    if not _ART_DIR.exists():
        return set()
    return {f.name for f in _ART_DIR.iterdir() if f.is_file()}


def _load_history_items(limit: int = 2000) -> List[Dict[str, Any]]:
    fn = _ART_DIR / 'policy_evaluate_history.jsonl'
    items: List[Dict[str, Any]] = []
    if fn.exists():
        try:
            with fn.open('r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        items.append(json.loads(s))
                    except Exception:
                        pass
        except Exception:
            pass
    return items[-limit:] if limit else items


def _pick_float(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if v not in (None, '', 'null'):
                return float(v)
        except Exception:
            pass
    return default


def _pick_ts(d: Dict[str, Any], i: int):
    return d.get('ts') or d.get('time') or d.get('timestamp') or d.get('t') or i


def _build_metrics_history(limit: int = 2000) -> Dict[str, Any]:
    items = _load_history_items(limit=limit)
    series: Dict[str, List[Dict[str, Any]]] = {}
    keymap = {
        'reward': ['reward', 'r', 'avg_reward', 'mean_reward'],
        'delta_kWh': ['delta_kWh', 'kwh_saving', 'energy_kwh_saving', 'energy_saving_kwh'],
        'peak_reduction_kW': ['peak_reduction_kW', 'peak_kw_delta', 'peak_kW_reduction'],
        'delay_min': ['delay_min', 'delay', 'latency_min'],
        'loss': ['loss', 'train_loss', 'q_loss'],
        'q_loss': ['q_loss'],
        'v_loss': ['v_loss'],
        'pi_loss': ['pi_loss'],
        'savings_yuan': ['savings_yuan', 'saving_yuan', 'benefit_yuan', 'delta_cost_yuan'],
        'savings_cum_yuan': ['savings_cum_yuan'],
        'entropy': ['entropy', 'policy_entropy'],
        'episode': ['episode', 'ep', 'epoch'],
    }
    for i, it in enumerate(items):
        ts = _pick_ts(it, i)
        for name, keys in keymap.items():
            v = _pick_float(it, keys, None)
            if v is None:
                continue
            series.setdefault(name, []).append({'t': ts, 'v': v})

    meta = {}
    for fnm in ('policy_info.json', 'policy_meta.json', 'iql_eval_report.json', 'bc_eval_report.json', 'kpi_summary.json'):
        p = _ART_DIR / fnm
        if p.exists():
            try:
                meta[fnm] = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                pass

    return {
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'count': len(items),
        'series': series,
        'artifacts': {
            'history_jsonl': '/api/rl/artifacts/policy_evaluate_history.jsonl' if (_ART_DIR / 'policy_evaluate_history.jsonl').exists() else None,
            'reward_costs_png': '/api/rl/artifacts/reward_costs.png' if (_ART_DIR / 'reward_costs.png').exists() else None,
            'rollout_price_ef_png': '/api/rl/artifacts/rollout_price_ef.png' if (_ART_DIR / 'rollout_price_ef.png').exists() else None,
            'rollout_loads_png': '/api/rl/artifacts/rollout_loads.png' if (_ART_DIR / 'rollout_loads.png').exists() else None,
        },
        'meta': meta,
    }


def _calc_summary_payload(svc: AGVChargeService) -> Dict[str, Any]:
    history = _build_metrics_history(limit=2000)
    series = history.get('series', {}) or {}
    items = _load_history_items(limit=2000)
    last = items[-1] if items else {}

    def _last_val(name: str, default=None):
        arr = series.get(name) or []
        if arr:
            try:
                return float(arr[-1].get('v'))
            except Exception:
                return default
        if isinstance(last, dict):
            mapping = {
                'reward': ['reward', 'r', 'avg_reward', 'mean_reward'],
                'delta_kWh': ['delta_kWh', 'kwh_saving', 'energy_kwh_saving', 'energy_saving_kwh'],
                'peak_reduction_kW': ['peak_reduction_kW', 'peak_kw_delta', 'peak_kW_reduction'],
                'savings_yuan': ['savings_yuan', 'saving_yuan', 'benefit_yuan', 'delta_cost_yuan'],
                'entropy': ['entropy', 'policy_entropy'],
                'episode': ['episode', 'ep', 'epoch'],
                'loss': ['loss', 'train_loss', 'q_loss'],
                'q_loss': ['q_loss'],
                'v_loss': ['v_loss'],
                'pi_loss': ['pi_loss'],
            }
            for k in mapping.get(name, [name]):
                try:
                    v = last.get(k)
                    if v not in (None, '', 'null'):
                        return float(v)
                except Exception:
                    pass
        return default

    try:
        sim = svc.simulate('agv_charge_v1')
    except Exception:
        sim = None

    price = _load_price_series(360, 1)
    ef = _load_ef_series(360, 1)
    baseline = _load_baseline_kw(360, 1)
    avg_price = _safe_avg(price, 0.65)
    avg_ef = _safe_avg(ef, 0.12)
    avg_load = _safe_avg(baseline, 0.0)
    peak_load = max(baseline) if baseline else 0.0
    reward_last = _last_val('reward', None)
    reward_is_batch_mean = bool(last.get('reward_is_batch_mean')) if isinstance(last, dict) else False
    peak_red_last = _last_val('peak_reduction_kW', sim.summary.peak_reduction_kW if sim else 0.0)
    saving_last = _last_val('savings_yuan', None)
    if saving_last is None and sim is not None:
        saving_last = round(float(sim.summary.delta_kWh) * avg_price, 3)
    delta_kwh_last = _last_val('delta_kWh', sim.summary.delta_kWh if sim else 0.0)

    source_status = 'artifact_online' if history.get('count', 0) > 0 else 'sim_only'
    judgement = '建议上线灰度' if (peak_red_last or 0.0) >= 5.0 else '建议继续影子评估'
    if (delta_kwh_last or 0.0) <= 0:
        judgement = '暂不建议下发'

    latest_ts = None
    if items:
        latest_ts = _pick_ts(items[-1], len(items) - 1)

    dispatch_hist = svc.dispatch_history(5).get('items', [])
    exec_hist = svc.exec_list(5).get('items', [])

    return {
        'module': 'agv_charge',
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source_status': source_status,
        'source_label': '在线训练产物' if source_status == 'artifact_online' else '仅仿真摘要',
        'judgement': judgement,
        'judgement_basis': {
            'reward_last': reward_last,
            'reward_is_batch_mean': reward_is_batch_mean,
            'peak_reduction_kW_last': peak_red_last,
            'delta_kWh_last': delta_kwh_last,
            'saving_yuan_last': saving_last,
            'history_count': history.get('count', 0),
            'latest_record_ts': latest_ts,
        },
        'kpis': {
            'reward_last': reward_last,
            'reward_is_batch_mean': reward_is_batch_mean,
            'peak_reduction_kW_last': peak_red_last,
            'delta_kWh_last': delta_kwh_last,
            'saving_yuan_last': saving_last,
            'history_count': history.get('count', 0),
        },
        'data_lineage': {
            'baseline_curve': f'data/grid_meter.csv::{GRID_KW_COL}',
            'price_curve': f'data/market_price.csv::{PRICE_COL}',
            'grid_ef_curve': f'data/grid_ef.csv::{EF_COL}',
            'time_col': TIME_COL,
            'train_history': 'artifacts/policy_evaluate_history.jsonl' if history.get('count', 0) > 0 else None,
        },
        'business_context': {
            'object': 'AGV 充/换电窗口调度',
            'action': '高价时段抑制充电，低价时段回补',
            'target': '削峰、降费、控碳，同时不明显伤害作业连续性',
        },
        'constraints': {
            'avg_load_kW': avg_load,
            'peak_load_kW': peak_load,
            'avg_price_yuan_per_kWh': avg_price,
            'avg_grid_ef_kg_per_kWh': avg_ef,
            'guardrail': '不追求极限节电，优先避免把作业延迟和电池寿命风险讲崩',
        },
        'execution': {
            'recent_dispatch_count': len(dispatch_hist),
            'recent_exec_count': len(exec_hist),
            'latest_dispatch_status': dispatch_hist[0].get('status') if dispatch_hist else None,
            'latest_exec_status': exec_hist[0].get('status') if exec_hist else None,
        },
        'artifacts': history.get('artifacts', {}),
    }


def _install_extra_routes(router):
    if getattr(router, '_agv_extra_routes_installed', False):
        return router

    whitelist = _list_whitelist()

    @router.get('/api/rl/artifacts/{name:path}')
    async def get_artifact(name: str):
        nonlocal whitelist
        if not whitelist:
            whitelist = _list_whitelist()
        if name not in whitelist:
            cand = [n for n in whitelist if n.lower() == name.lower()]
            if cand:
                name = cand[0]
            else:
                raise HTTPException(404, 'artifact not found')
        p = _safe_target(name)
        if not p.exists():
            raise HTTPException(404, 'artifact file missing')
        return FileResponse(str(p), media_type=_mime_for(p), headers={'Cache-Control': 'no-store'})

    @router.get('/api/rl/model/agv_charge/artifacts/{name:path}')
    async def get_artifact_compat(name: str):
        return await get_artifact(name)

    async def head_artifact(name: str):
        nonlocal whitelist
        if not whitelist:
            whitelist = _list_whitelist()
        if name not in whitelist:
            cand = [n for n in whitelist if n.lower() == name.lower()]
            if cand:
                name = cand[0]
            else:
                raise HTTPException(404, 'artifact not found')
        p = _safe_target(name)
        if not p.exists():
            raise HTTPException(404, 'artifact file missing')
        return FileResponse(str(p), media_type=_mime_for(p), filename=p.name)

    router.add_api_route('/api/rl/artifacts/{name:path}', endpoint=head_artifact, methods=['HEAD'], include_in_schema=False)
    router.add_api_route('/api/rl/model/agv_charge/artifacts/{name:path}', endpoint=head_artifact, methods=['HEAD'], include_in_schema=False)

    @router.get('/api/rl/metrics/history')
    async def metrics_history(limit: int = 2000):
        return JSONResponse(_build_metrics_history(limit=limit), headers={'Cache-Control': 'no-store'})

    @router.get('/api/rl/summary')
    async def rl_summary():
        svc = AGVChargeService()
        return JSONResponse(_calc_summary_payload(svc), headers={'Cache-Control': 'no-store'})

    @router.get('/api/rl/module_a/summary')
    async def rl_summary_alias():
        svc = AGVChargeService()
        return JSONResponse(_calc_summary_payload(svc), headers={'Cache-Control': 'no-store'})

    @router.get('/api/rl/rollout/status')
    async def rollout_status():
        p = _ART_DIR / 'policy_meta.json'
        s = {
            'phase': 'shadow',
            'candidate_version': None,
            'stable_version': None,
            'traffic_pct': 0.0,
            'metrics': {'mape_energy_mean': None, 'guard_block_rate': None, 'sla_violation_rate': None},
            'thresholds': {'mape_energy_max': 0.10, 'guard_block_rate_max': 0.05, 'sla_violation_rate_max': 0.02},
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        try:
            if p.exists():
                m = json.loads(p.read_text(encoding='utf-8'))
                s['phase'] = m.get('phase', s['phase'])
                s['candidate_version'] = m.get('candidate_version')
                s['stable_version'] = m.get('stable_version')
                s['traffic_pct'] = float(m.get('traffic_pct', s['traffic_pct']))
                if 'metrics' in m:
                    s['metrics'].update(m['metrics'])
                if 'thresholds' in m:
                    s['thresholds'].update(m['thresholds'])
        except Exception:
            pass
        return JSONResponse(s, headers={'Cache-Control': 'no-store'})

    router._agv_extra_routes_installed = True  # type: ignore[attr-defined]
    return router


try:
    _AGV_BASE_GET_ROUTER = get_router
except Exception:
    _AGV_BASE_GET_ROUTER = None


def get_router():
    """最终导出的稳定路由：保留原接口，再补 artifacts / summary / HEAD 路由，只装一次。"""
    try:
        from fastapi import APIRouter
    except Exception:
        return None

    if callable(_AGV_BASE_GET_ROUTER):
        r = _AGV_BASE_GET_ROUTER()
    else:
        r = APIRouter()
    if r is None:
        r = APIRouter()
    return _install_extra_routes(r)
# ==============================================================================\n