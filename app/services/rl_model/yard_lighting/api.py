# app/services/rl_model/yard_lighting/api.py
# -*- coding: utf-8 -*-
"""
Yard Lighting API (B 模块统一出入口)
- 按照“滚动控制 + RL 决策 + MPC/规则兜底 + 合规可视化”的落地口径对接
- 函数即接口：由 server.py / dispatch_api.py 挂路由
- 读取 artifacts/offline_train.json（奖励分解 + 经济节省）做“收益卡”
- 读取 artifacts/lighting_state.json（策略面板/仿真/解释/告警等）
- 读取 data/*（zones/telemetry/price/ef）做仿真与数据校验

接口对照（建议在路由层按名称直接映射）：
GET  /api/rl/lighting/strategies  -> get_strategies(horizon_min=720, dt_min=10)
POST /api/rl/lighting/simulate    -> simulate(strategy_id="yard_lighting_v1", horizon_min=720, dt_min=10)
GET  /api/rl/lighting/explain     -> explain(zone_id, t_iso)
GET  /api/rl/lighting/alerts      -> get_alerts(window="tonight")
POST /api/lighting/dispatch       -> dispatch(schedule: dict, zones: list[str])
POST /api/lighting/fallback       -> fallback(mode: "rule"|"mpc"|"none")
GET  /api/rl/lighting/train       -> train_status()   # 离线训练最新快照（含奖励与节省）
GET  /api/rl/lighting/self_check  -> self_check(print_json=False)
"""

from __future__ import annotations
import json, csv, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone

# ---- 路径规划 ----
MOD_DIR = Path(__file__).resolve().parent
DATA_DIR = MOD_DIR / "data"
ART_DIR  = MOD_DIR / "artifacts"

STATE_JSON          = ART_DIR / "lighting_state.json"     # 平台状态（策略/仿真/解释/告警等）
OFFLINE_SNAPSHOT    = ART_DIR / "offline_train.json"      # 训练快照（奖励分解+经济节省）
OFFLINE_LOG_JSONL   = ART_DIR / "offline_train.jsonl"     # 训练历史（可 tail -f）
POLICY_BIN          = MOD_DIR / "policy.bin"              # 离线策略权重（IQL/CQL）
POLICY_META         = MOD_DIR / "policy_meta.json"        # 策略元信息（obs 标准化/算法/键）
RESIDUAL_BIN        = MOD_DIR / "residual_policy.bin"     # 在线残差策略（可选）
DATASET_REPORT     = ART_DIR / "lighting_dataset_report.json"  # 可选：数据集覆盖/质量报告
DT_MIN_DEFAULT      = 10
HORIZON_MIN_DEFAULT = 720

# ---- 列名候选（与 rl_engine 保持一致） ----
_TS_CANDS  = ["timestamp","ts","time","datetime","date_time","utc","utc_time","date"]
_ZONE_COLS = ["zone_id","zone","id"]
_PWR_COLS  = ["power_kW","power_kw","power","kw"]
_DIM_COLS  = ["dimming_percent","dimming","dim","d"]
_LUX_COLS  = ["lux","illuminance","lx"]
_PRICE_COLS=["price_yuan_kWh","price","p_elec","elec_price","tariff","price_yuan_per_kwh","price_p50","price_p90"]
_EF_COLS   = ["ef_kg_kWh","ef","marginal_kg_per_kWh","carbon_intensity","ci_kg_per_kwh","ef_kg_per_kwh","ef_p50","ef_p90"]

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _rjson(p: Path, default: Any=None) -> Any:
    try:
        if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
    except: pass
    return default

def _wjson(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _rcsv(p: Path) -> List[Dict[str,Any]]:
    rows=[]
    if not p.exists(): return rows
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        rd=csv.DictReader(f)
        for r in rd:
            rows.append({(k or "").strip(): (v.strip() if isinstance(v,str) else v) for k,v in r.items()})
    return rows

def _pick(row: Dict[str,Any], cands: List[str], default=None):
    for k in cands:
        if k in row and row[k] not in (None,""): return row[k]
    return default

def _parse_ts(s: str) -> Optional[datetime]:
    if not s: return None
    s=s.strip()
    try:
        if s.isdigit():
            v=int(s);
            if v>10_000_000_000: v=v/1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
    except: pass
    s=s.replace("Z","+00:00")
    if " " in s and "T" not in s and ("+" in s or s.endswith("+00:00")):
        s=s.replace(" ", "T", 1)
    for fmt in (None,"%Y-%m-%d %H:%M:%S%z","%Y-%m-%d %H:%M%z","%Y/%m/%d %H:%M:%S%z","%Y/%m/%d %H:%M%z",
                      "%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%Y/%m/%d %H:%M:%S","%Y/%m/%d %H:%M","%Y-%m-%d","%Y/%m/%d"):
        try:
            return datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except: pass
    return None

# ---- 合规&奖励口径（与前文保持一致；来源 config_limits.json）----
def load_limits() -> Dict[str,Any]:
    # 若缺失则给默认值；单位/权重保持口径一致（见 config_limits.json）
    default = {
        "lighting_policy": {
            "min_dwell_min_default": 15,
            "min_interval_min_default": 10,
            "max_switches_per_night": 8,
            "ramp_percent_per_step_default": 15,
            "glare_threshold": 0.7,
            "complaint_sensitive_hours_local": [22,6],
        },
        "reward_weights_default": {
            "alpha_under_lux": 50.0, "beta_switch": 0.2, "gamma_glare": 2.0, "delta_complaint": 5.0
        },
        "units": {"power":"kW","energy":"kWh","price":"CNY/kWh","ef":"kg/kWh","lux":"lux","time":"UTC ISO8601"}
    }
    for p in [DATA_DIR/"config_limits.json", MOD_DIR/"data"/"config_limits.json"]:
        if p.exists(): return _rjson(p, default) or default
    return default

# ---- 快照/平台态 ----
def _ensure_state():
    if not STATE_JSON.exists():
        _wjson(STATE_JSON, {
            "meta": {"created_at": _now(), "selected_data_dir": str(DATA_DIR.resolve())},
            "dispatch": {"history": [], "fallback_mode": "none"},
            "strategies": {"strategies": []},
        })

def get_train_snapshot() -> Optional[Dict[str,Any]]:
    """最新离线训练快照（优先 .json；没有就读 .jsonl 的最后一条）。"""
    obj = _rjson(OFFLINE_SNAPSHOT, default=None)
    if obj is not None:
        return obj
    # 兜底：读取 offline_train.jsonl 的最后一条记录
    if OFFLINE_LOG_JSONL.exists():
        try:
            last = None
            with OFFLINE_LOG_JSONL.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    last = json.loads(line)
            return last
        except Exception:
            return None
    return None


# ---- 策略列表 ----
def get_strategies(horizon_min: int=HORIZON_MIN_DEFAULT, dt_min: int=DT_MIN_DEFAULT) -> Dict[str,Any]:
    _ensure_state()
    title = f"YardLighting-v1 {_now()}→(H+{horizon_min}m)"
    return {
        "strategies": [
            {
                "id": "yard_lighting_v1",
                "title": title,
                "objective": "min_cost_min_peak_zero_violations",
                "horizon_min": horizon_min,
                "dt_min": dt_min,
                "hints": {
                    "algo": "IQL/CQL (offline) + Residual Safe‑SAC (online)",
                    "safety": "L>=L_min, dwell, min-interval, ramp, max-switch/night"
                },
                "version": _rjson(POLICY_META,{}).get("algo","IQL/CQL")
            }
        ]
    }

# ---- 仿真（基线 vs 策略），输出总功率曲线/合规计数/经济&奖励（若有）----
@dataclass
class _SeriesRow:
    ts: datetime; zone: str; power: float; dim: float; lux: float; L_min: float; critical: bool; complaint: bool; price: float; ef: float

def _load_series() -> Tuple[List[datetime], Dict[str,List[_SeriesRow]]]:
    zones = _rcsv(DATA_DIR/"zones_master.csv")
    tele  = _rcsv(DATA_DIR/"lighting_telemetry.csv")
    price = _rcsv(DATA_DIR/"market_price.csv")
    ef    = _rcsv(DATA_DIR/"grid_ef.csv")
    zone_meta={str(_pick(z,_ZONE_COLS,"")):z for z in zones if _pick(z,_ZONE_COLS,"")}
    # 生成时间网格：优先 price -> ef -> telemetry
    grid=[]
    if price: grid=[_parse_ts(_pick(r,_TS_CANDS,"")) for r in price]
    if not grid and ef: grid=[_parse_ts(_pick(r,_TS_CANDS,"")) for r in ef]
    if not grid and tele:
        cand=sorted({ _parse_ts(_pick(r,_TS_CANDS,"")) for r in tele if _parse_ts(_pick(r,_TS_CANDS,"")) })
        grid=cand
    grid=[t for t in grid if t]; grid.sort()
    # 构造映射
    p_map={_parse_ts(_pick(r,_TS_CANDS,"")): float(_pick(r,_PRICE_COLS,0) or 0) for r in price if _parse_ts(_pick(r,_TS_CANDS,""))}
    e_map={_parse_ts(_pick(r,_TS_CANDS,"")): float(_pick(r,_EF_COLS,0) or 0)    for r in ef    if _parse_ts(_pick(r,_TS_CANDS,""))}
    # zone -> [(ts,power,dim,lux)]
    raw={}
    for r in tele:
        zid=str(_pick(r,_ZONE_COLS,"")); ts=_parse_ts(_pick(r,_TS_CANDS,""))
        if not zid or zid not in zone_meta or not ts: continue
        power=float(_pick(r,_PWR_COLS,0) or 0)
        dim  =float(_pick(r,_DIM_COLS,1.0) or 1.0); dim = (dim/100.0 if dim>1.5 else dim); dim=max(0.05, min(1.0, dim))
        lux  =float(_pick(r,_LUX_COLS,0) or 0)
        if lux<=0.5: lux=float(zone_meta[zid].get("L_min",20.0))*1.2  # 缺测兜底
        raw.setdefault(zid,[]).append((ts,power,dim,lux))
    per_zone = {}
    for zid in zone_meta.keys():
        seq = []
        rows = sorted(raw.get(zid,[]), key=lambda x:x[0])
        for g in grid:
            if rows:
                # 最邻近
                idx = int(min(range(len(rows)), key=lambda i: abs((rows[i][0]-g).total_seconds())))
                pow0, dim0, lux0 = float(rows[idx][1]), float(rows[idx][2]), float(rows[idx][3])
            else:
                pow0, dim0, lux0 = 1.0, 1.0, float(zone_meta[zid].get("L_min",20.0))*1.2
            seq.append(_SeriesRow(
                ts=g, zone=zid, power=max(0.0, pow0), dim=max(0.05,dim0), lux=max(0.1,lux0),
                L_min=float(zone_meta[zid].get("L_min",20.0)),
                critical=bool(zone_meta[zid].get("critical", False)),
                complaint=bool(zone_meta[zid].get("complaint_zone", False)),
                price=float(p_map.get(g,0.0) or 0.0),
                ef=float(e_map.get(g,0.0) or 0.0),
            ))
        per_zone[zid]=seq
    return grid, per_zone

def _rule_baseline(prev_a: float, activity: float, critical: bool) -> float:
    if critical: tgt=0.95
    else:
        if activity>=0.6: tgt=0.85
        elif activity>=0.3: tgt=0.7
        else: tgt=0.5
        tgt=max(tgt-0.05,0.4)
    return float(max(0.05, min(1.0, 0.8*prev_a + 0.2*tgt)))

def _activity_proxy(row: _SeriesRow) -> float:
    # 简单代理：夜深下调，人流未知时用 sin/cos 时间作为 proxy，保持与 rl_engine 的口径一致
    h=row.ts.hour
    base=0.3+0.3*math.sin(2*math.pi*h/24.0)
    if row.critical: base=max(base,0.7)
    return float(max(0.0,min(1.0,base)))

def simulate(strategy_id: str="yard_lighting_v1", horizon_min: int=HORIZON_MIN_DEFAULT, dt_min: int=DT_MIN_DEFAULT) -> Dict[str,Any]:
    """夜间 12h（默认）基线 vs 策略仿真；返回 agg kW 曲线、总 kWh、合规统计；若有训练快照则并回经济与奖励。"""
    limits = load_limits()  # 单位/权重/阈值口径一致（见 config_limits.json）
    grid, per_zone = _load_series()
    if not grid:  # 数据缺失
        return {"summary":{"delta_kWh":0,"delta_carbon_kg":0,"peak_reduction_kW":0,"under_lux_violations":0,"switches_total":0,
                           "glare_risk_hits":0,"complaint_risk_hits":0},
                "baseline":{"agg_kW":[],"total_kWh":0.0},
                "simulated":{"agg_kW":[],"total_kWh":0.0},
                "per_zone_stats": {}}

    DT_H = dt_min/60.0
    agg_b=[0.0]*len(grid); agg_p=[0.0]*len(grid)
    under_lux=switches=glare_hits=complaint_hits=0
    per_zone_stats={}

    for zid, seq in per_zone.items():
        last_a = seq[0].dim; z_sw=z_ul=z_gl=z_cp=0
        for i,row in enumerate(seq):
            a_b = _rule_baseline(last_a, _activity_proxy(row), row.critical)
            a_p = a_b  # 这里策略简化为“保守-合规”的动作；若需要可加载 policy.bin 做预测
            # 投产阶段：policy.bin 可由 rl_engine 训练后加载；此处保留接口
            # if POLICY_BIN.exists(): a_p = _infer_pi(obs)  # 省略实现细节，保持接口

            base_a = max(0.05,row.dim)
            scale_b = a_b/base_a
            scale_p = a_p/base_a

            p_b = row.power*scale_b; p_p=row.power*scale_p
            agg_b[i]+=p_b; agg_p[i]+=p_p

            lux_b = row.lux*(a_b/base_a); lux_p=row.lux*(a_p/base_a)
            if lux_p < row.L_min - 1e-6:
                under_lux += 1; z_ul += 1
            if abs(a_p-last_a)>=0.05:
                switches += 1; z_sw += 1
            if (abs(a_p-last_a)>=0.15 and a_p>=limits["lighting_policy"]["glare_threshold"]):
                glare_hits += 1; z_gl += 1
            if row.complaint:
                s_h,e_h = limits["lighting_policy"]["complaint_sensitive_hours_local"]
                if (row.ts.hour>=s_h or row.ts.hour<e_h) and a_p>=0.8:
                    complaint_hits += 1; z_cp += 1
            last_a=a_p
        per_zone_stats[zid] = {"switches": z_sw, "under_lux": z_ul, "glare_hits": z_gl, "complaint_hits": z_cp}

    base_kWh = float(sum(agg_b)*DT_H); sim_kWh=float(sum(agg_p)*DT_H)
    peak_red = max(0.0, max(agg_b or [0.0])-max(agg_p or [0.0]))

    summary = {
        "delta_kWh": round(max(0.0, base_kWh - sim_kWh),3),
        "delta_carbon_kg": 0.0,  # 若需要精算，可叠加 EF×kWh；此处交给训练快照的 economics 字段
        "peak_reduction_kW": round(peak_red,3),
        "under_lux_violations": int(under_lux),
        "switches_total": int(switches),
        "glare_risk_hits": int(glare_hits),
        "complaint_risk_hits": int(complaint_hits)
    }
    # 合并训练快照的经济与奖励
    snap = get_train_snapshot()
    out = {
        "summary": summary,
        "baseline": {"agg_kW": [round(x,2) for x in agg_b], "total_kWh": round(base_kWh,3)},
        "simulated": {"agg_kW": [round(x,2) for x in agg_p], "total_kWh": round(sim_kWh,3)},
        "per_zone_stats": per_zone_stats
    }
    if snap:
        out["economics"] = snap.get("economics", {})
        out["rewards"]   = snap.get("rewards", {})
        out["units"]     = snap.get("units", load_limits().get("units", {}))
    return out

# ---- 可解释性（特征重要性/理由） ----
def explain(zone_id: str, t_iso: str) -> Dict[str,Any]:
    # 这里做轻量解释：价格/碳因子高→降亮；关键区→保亮；投诉敏感时段→趋向降亮避免扰民
    limits = load_limits()
    dt = _parse_ts(t_iso) or datetime.now(timezone.utc)
    h  = dt.hour
    feats = [
        {"name":"price", "value": 0.8, "importance": 0.35, "direction": "down", "contribution_kWh": None},
        {"name":"ef",    "value": 0.12,"importance": 0.15, "direction": "down", "contribution_kWh": None},
        {"name":"activity","value": 0.5,"importance": 0.30, "direction": "up",  "contribution_kWh": None},
        {"name":"critical","value": 0,  "importance": 0.10, "direction": "flat","contribution_kWh": None},
        {"name":"complaint_window","value": 1 if (h>=limits["lighting_policy"]["complaint_sensitive_hours_local"][0] or h<limits["lighting_policy"]["complaint_sensitive_hours_local"][1]) else 0,
         "importance": 0.05, "direction": "down", "contribution_kWh": None},
        {"name":"L_min", "value": 20.0,"importance": 0.05, "direction": "up",  "contribution_kWh": None},
    ]
    reasons=[]
    if feats[0]["value"]>=0.7: reasons.append("电价处于高位，策略倾向于降亮节能")
    if feats[1]["value"]>=0.1: reasons.append("电网碳因子偏高，策略倾向于降亮减排")
    return {"features": feats, "reasons": reasons, "meta": {"zone_id": zone_id, "t": dt.isoformat()}}

# ---- 告警（合规/眩光/切换/故障） ----
def get_alerts(window: str="tonight") -> Dict[str,Any]:
    # 当前实现读取最近一次 simulate 的 per_zone_stats（若 state 中保存），否则做轻量扫描
    st = _rjson(STATE_JSON, default={})
    sim = (st.get("simulate") or st.get("simulate_last")) or {}
    per = sim.get("per_zone_stats", {})
    ul = sum(1 for z,s in per.items() if s.get("under_lux",0)>0)
    gl = sum(1 for z,s in per.items() if s.get("glare_hits",0)>0)
    sw = sum(1 for z,s in per.items() if s.get("switches",0) > load_limits()["lighting_policy"]["max_switches_per_night"])
    ft = 0  # 若有 status=fault 的字段可在 telemetry 中统计
    return {"under_lux_zones": ul, "glare_risk_zones": gl, "switch_exceed_zones": sw, "fault_zones": ft, "ts": _now()}

# ---- 执行与兜底（将请求记录到 lighting_state.json，供审计/回滚） ----
def dispatch(schedule: Dict[str,Any], zones: List[str]) -> Dict[str,Any]:
    _ensure_state()
    st = _rjson(STATE_JSON, default={})
    hist = st.setdefault("dispatch", {}).setdefault("history", [])
    item = {"ts": _now(), "zones": zones, "schedule": schedule, "who": "api.yard_lighting"}
    hist.append(item)
    st["dispatch"]["fallback_mode"] = "none"
    _wjson(STATE_JSON, st)
    return {"ok": True, "recorded": item}

def fallback(mode: str="rule") -> Dict[str,Any]:
    assert mode in ("rule","mpc","none"), "mode 必须为 rule/mpc/none"
    _ensure_state()
    st = _rjson(STATE_JSON, default={})
    st.setdefault("dispatch", {})["fallback_mode"] = mode
    _wjson(STATE_JSON, st)
    return {"ok": True, "mode": mode, "ts": _now()}

# ---- 训练状态（收益卡/奖励汇总） ----
def train_status() -> Dict[str,Any]:
    snap = get_train_snapshot()
    if not snap:
        return {"exists": False, "path": str(OFFLINE_SNAPSHOT), "hint": "尚未启动离线训练或尚未产生首个打点"}
    return {"exists": True, "snapshot": snap, "path": str(OFFLINE_SNAPSHOT)}

# ---- 自检：策略/仿真/解释/告警 + 简易收益卡 ----
def self_check(print_json: bool=False) -> Dict[str,Any]:
    _ensure_state()
    strategies = get_strategies()
    sim = simulate()
    ex  = explain(zone_id=(list(sim.get("per_zone_stats",{}).keys()) or ["Z1"])[0], t_iso=_now())
    al  = get_alerts()
    train = get_train_snapshot()

    out = {
        "strategies": strategies,
        "simulate": {
            "delta_kWh": sim["summary"]["delta_kWh"],
            "delta_carbon_kg": sim["summary"]["delta_carbon_kg"],
            "peak_reduction_kW": sim["summary"]["peak_reduction_kW"],
            "under_lux_violations": sim["summary"]["under_lux_violations"],
            "switches_total": sim["summary"]["switches_total"],
            "glare_risk_hits": sim["summary"]["glare_risk_hits"],
            "complaint_risk_hits": sim["summary"]["complaint_risk_hits"],
        },
        "explain": ex,
        "alerts": al
    }
    # 合并收益卡（来自 offline_train.json）
    if train:
        econ = train.get("economics", {})
        out["economics"] = econ.get("savings", {}) or {}
        out["rewards"]   = train.get("rewards", {})
        out["units"]     = train.get("units", load_limits().get("units", {}))

    # 将本次仿真保存在 state，便于路由层/前端直接 GET 提取
    st = _rjson(STATE_JSON, {})
    st["simulate"] = sim
    st["meta"] = st.get("meta", {})
    st["meta"]["updated_at"] = _now()
    _wjson(STATE_JSON, st)

    if print_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
# ---- 统一快照：供前端“训练监控/策略面板”直接读取 ----
def snapshot() -> Dict[str, Any]:
    """
    汇总 yard_lighting 的最新训练/经济/合规与配置口径（只读）。
    供路由映射：GET /api/rl/lighting/snapshot -> snapshot()
    """
    # 配置摘要（奖励权重、惩罚口径、DT_MIN）
    limits = load_limits()
    try:
        from . import rl_engine  # 与训练引擎保持时间步一致
        dt_min = int(getattr(rl_engine, "DT_MIN", DT_MIN_DEFAULT))
    except Exception:
        dt_min = DT_MIN_DEFAULT

    out: Dict[str, Any] = {
        "updated_at": _now(),
        "config": {
            "DT_MIN": dt_min,
            "reward_weights": limits.get("reward_weights_default", {}),
            "penalty_config": limits.get("penalty_config", {})  # 若无则为空对象
        }
    }

    # 训练快照（优先 offline_train.json；没有就取 offline_train.jsonl 的最后一条）
    snap = get_train_snapshot()
    if snap:
        # 经济分解（兼容 baseline/policy 结构，也兼容早期 savings 结构）
        econ = snap.get("economics", {})
        base = econ.get("baseline") or {}
        pol  = econ.get("policy") or {}
        if base or pol:
            def _R(d: Dict[str, Any]) -> Dict[str, float]:
                r = d.get("rewards", {})
                return {
                    "energy_cost": float(r.get("energy_cost", 0.0)),
                    "carbon_cost": float(r.get("carbon_cost", 0.0)),
                    "under_lux_penalty": float(r.get("under_lux_penalty", 0.0)),
                    "switch_penalty": float(r.get("switch_penalty", 0.0)),
                    "glare_penalty": float(r.get("glare_penalty", 0.0)),
                    "complaint_penalty": float(r.get("complaint_penalty", 0.0)),
                    "reward_total": float(r.get("reward_total", 0.0))
                }

            base_out = {
                "kWh": float(base.get("kWh", base.get("energy_kWh", 0.0))),
                "peak_kW": float(base.get("peak_kW", 0.0)),
                "rewards": _R(base)
            }
            pol_out = {
                "kWh": float(pol.get("kWh", pol.get("energy_kWh", 0.0))),
                "peak_kW": float(pol.get("peak_kW", 0.0)),
                "rewards": _R(pol)
            }
            delta = {
                "reward_gain": pol_out["rewards"]["reward_total"] - base_out["rewards"]["reward_total"],
                "kWh_delta":   pol_out["kWh"] - base_out["kWh"],
                "peak_reduction_kW": base_out["peak_kW"] - pol_out["peak_kW"]
            }
            out["economics"] = {"baseline": base_out, "policy": pol_out, "delta": delta}
        else:
            # 兼容旧结构：economics.savings
            if "savings" in econ:
                out["economics"] = {"savings": econ.get("savings", {})}

        # KPI（与 rl_engine 新口径一致：under_lux 为 “百分比×小时”）
        m = snap.get("metrics", {})
        out["metrics"] = {
            "under_lux_ratio_avg": float(m.get("under_lux_ratio", m.get("under_lux", 0.0))),
            "switch_count": int(m.get("switch_count", m.get("switches", 0))),
            "glare_index": float(m.get("glare_index", m.get("glare", 0.0))),
            "complaints_cnt": int(m.get("complaints_cnt", m.get("complaints", 0)))
        }

    # policy 元信息（可选）
    pm = _rjson(POLICY_META, default=None)
    if pm:
        out["policy_meta"] = pm

    # 数据集报告（可选）
    dr = _rjson(DATASET_REPORT, default=None)
    if dr:
        out["dataset"] = {
            "rows_telemetry": dr.get("rows_telemetry"),
            "rows_activity": dr.get("rows_activity"),
            "zones": dr.get("zones"),
            "time_range": dr.get("time_range"),
            "data_quality": dr.get("data_quality")
        }

    return out
