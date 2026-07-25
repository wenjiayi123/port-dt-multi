# app/services/rl_model/yard_lighting/rl_engine.py
# -*- coding: utf-8 -*-
"""
Yard Lighting RL Engine (Adapter + Offline IQL/CQL + Online Residual Safe-SAC)
- 统一数据适配、离线/在线训练、日志打点
- 新增：训练时每 log 步把 “奖励分解 + 经济指标(相对规则基线的省钱)” 同步写入:
    1) artifacts/offline_train.jsonl （历史 JSONLines）
    2) artifacts/offline_train.json  （快照 JSON，覆盖写入，便于前端直接读取最新）
- 新增：--sleep-every / --sleep-sec 周期性休息
"""

from __future__ import annotations
import os, csv, json, math, argparse, random, shutil, time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- 路径与常量 --------------------
MOD_DIR = Path(__file__).resolve().parent
DATA_DIR = MOD_DIR / "data"
ART_DIR  = MOD_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

POLICY_BIN       = MOD_DIR / "policy.bin"             # 离线策略权重
POLICY_META      = MOD_DIR / "policy_meta.json"       # 标准化统计/算法/键
RESIDUAL_BIN     = MOD_DIR / "residual_policy.bin"    # 在线 residual 策略

OFFLINE_LOG_JSONL = ART_DIR / "offline_train.jsonl"   # 历史日志（逐行追加）
OFFLINE_SNAPSHOT  = ART_DIR / "offline_train.json"    # Latest snapshot, overwritten atomically

ONLINE_LOG_JSONL  = ART_DIR / "online_train.jsonl"
DEVICE      = "cpu"
DT_MIN      = 5
GAMMA       = 0.995

# -------------------- 列名/时间解析（与前一版一致，略去赘述） --------------------
_TS_CANDIDATES = ["timestamp","ts","time","datetime","date_time","utc","utc_time","date"]
_PRICE_CANDS   = ["price_yuan_kWh","price","p_elec","elec_price","tariff","price_yuan_per_kwh","price_p50","price_p90"]
_EF_CANDS      = ["ef_kg_kWh","ef","marginal_kg_per_kWh","carbon_intensity","ci_kg_per_kwh","ef_kg_per_kwh","ef_p50","ef_p90"]
_ACT_CANDS     = ["activity_score_p50","activity_score","activity_score_p90","activity","heat","load_index","people_count","vehicle_count"]
_PWR_CANDS     = ["power_kW","power_kw","power","kw"]
_DIM_CANDS     = ["dimming_percent","dimming","dim","d"]
_LUX_CANDS     = ["lux","illuminance","lx"]
_ZONE_CANDS    = ["zone_id","zone","id"]

def _now_utc_iso() -> str: return datetime.now(timezone.utc).isoformat()

def _rjson(p: Path, default: Dict[str,Any]|None=None) -> Dict[str,Any]:
    if not p.exists(): return default or {}
    return json.loads(p.read_text(encoding="utf-8"))

def _rcsv(p: Path) -> List[Dict[str,Any]]:
    out=[]
    if not p.exists(): return out
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        rd=csv.DictReader(f)
        for r in rd:
            row={};
            for k,v in r.items():
                k=(k or "").strip()
                if not k: continue
                row[k]=v.strip() if isinstance(v,str) else v
            out.append(row)
    return out

def _pick(row: Dict[str,Any], cands: List[str], default=None):
    for k in cands:
        if k in row and row[k] not in (None,""):
            return row[k]
    return default

def _parse_ts_any(s: str) -> Optional[datetime]:
    if not s: return None
    s=s.strip()
    if s.isdigit():
        try:
            v=int(s);
            if v>10_000_000_000: v=v/1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except: return None
    s_norm=s.replace("Z","+00:00")
    if " " in s_norm and "T" not in s_norm and ("+" in s_norm or s_norm.endswith("+00:00")):
        s_norm=s_norm.replace(" ", "T", 1)
    try: return datetime.fromisoformat(s_norm)
    except: pass
    for fmt in ("%Y-%m-%d %H:%M:%S%z","%Y-%m-%d %H:%M%z","%Y/%m/%d %H:%M:%S%z","%Y/%m/%d %H:%M%z"):
        try: return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except: pass
    for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%Y/%m/%d %H:%M:%S","%Y/%m/%d %H:%M","%Y-%m-%d","%Y/%m/%d"):
        try: return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except: pass
    try: return datetime.fromisoformat(s.replace(" ", "T", 1).replace("Z","+00:00"))
    except: return None

def _ts_from_row(row: Dict[str,Any]) -> Optional[datetime]:
    val=_pick(row,_TS_CANDIDATES);
    return _parse_ts_any(val) if val else None

def _num_from_row(row: Dict[str,Any], cands: List[str], default: float=0.0) -> float:
    v=_pick(row,cands)
    try:
        if v is None or v=="": return default
        return float(v)
    except: return default

# -------------------- ingest：从 /mnt/data 复制到 data/ --------------------
REQUIRED_FILES = [
    "zones_master.csv","lighting_telemetry.csv","activity_forecast.csv",
    "weather_astro.csv","market_price.csv","grid_ef.csv","complaints_events.csv",
]
OPTIONAL_FILES = ["config_limits.json"]

def ingest_from(src_dir: str="/mnt/data") -> Dict[str,Any]:
    src = Path(src_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    copied=[]
    for fn in REQUIRED_FILES + OPTIONAL_FILES:
        sp = src / fn
        if sp.exists():
            shutil.copyfile(sp, DATA_DIR / fn); copied.append(fn)
    ok = all((DATA_DIR/f).exists() for f in REQUIRED_FILES)
    report = {"ts": _now_utc_iso(),"src_dir": str(src.resolve()) if src.exists() else str(src),
              "out_dir": str(DATA_DIR.resolve()),"copied": copied,"ok": ok}
    print("[INGEST]", report)
    return report

# -------------------- 合规/奖励权重（缺失则用默认） --------------------
def load_limits() -> Dict[str,Any]:
    defaults = {
        "lighting_policy": {
            "min_dwell_min_default": 15,
            "min_interval_min_default": 10,
            "max_switches_per_night": 8,
            "ramp_percent_per_step_default": 15,
            "glare_threshold": 0.7,
            "complaint_sensitive_hours_local": [22, 6]
        },
        "reward_weights_default": {
            "alpha_under_lux": 5.0, "beta_switch": 0.2, "gamma_glare": 2.0, "delta_complaint": 5.0

        },
        "penalty_config": {
            "under_lux_mode": "percent_time",  # 照度惩罚口径：相对缺口(%)×时间(小时)
            "under_lux_clip_per_step": 0.5  # 每步裁剪上限，避免异常值炸梯度
        },
        "price_config": {
            "carbon_price_yuan_per_kg": 0.10
        },

        "units": { "power": "kW", "energy": "kWh", "price": "CNY/kWh", "ef": "kg/kWh", "lux":"lux", "time":"UTC ISO8601" }
    }
    for p in [DATA_DIR/"config_limits.json", MOD_DIR/"data"/"config_limits.json"]:
        if p.exists():
            return _rjson(p, default=defaults) or defaults
    return defaults

# -------------------- 数据集构造 --------------------
@dataclass
class Transition:
    s: np.ndarray; a: float; r: float; s_next: np.ndarray; done: bool

class LightingDataset(torch.utils.data.Dataset):
    """
    s: [price, ef, activity, lux, L_min, critical, complaint_zone, prev_a, sin_h, cos_h, dwell]
    a: dimming_percent ∈ [0,1]
    r: -电费 - 碳费 - αUnderLux - βSwitch - γGlare - δComplaint
    """
    def __init__(self, dt_min: int=DT_MIN, gamma: float=GAMMA):
        super().__init__()
        self.dt_h = dt_min/60.0; self.gamma = gamma; self.limits = load_limits()
        self.zones = _rcsv(DATA_DIR/"zones_master.csv")
        self.tele  = _rcsv(DATA_DIR/"lighting_telemetry.csv")
        self.act   = _rcsv(DATA_DIR/"activity_forecast.csv")
        self.price = _rcsv(DATA_DIR/"market_price.csv")
        self.ef    = _rcsv(DATA_DIR/"grid_ef.csv")

        self.zone_meta = {str(_pick(z,_ZONE_CANDS,"")): z for z in self.zones if _pick(z,_ZONE_CANDS,"")}
        assert len(self.zone_meta)>0, "zones_master.csv 为空或缺少 zone_id/zone/id 列"

        # 时间网格：优先 price → ef → telemetry
        grid=[]
        if self.price: grid=[ _ts_from_row(r) for r in self.price ]; grid=[t for t in grid if t is not None]
        if not grid and self.ef: grid=[ _ts_from_row(r) for r in self.ef ]; grid=[t for t in grid if t is not None]
        if not grid and self.tele:
            tt=[ _ts_from_row(r) for r in self.tele ]; grid=sorted({t for t in tt if t is not None})
        self.grid = sorted(grid); self.steps=len(self.grid)
        assert self.steps>0, "无法从 price/ef/telemetry 推断时间网格；请检查时间列"

        self.per_zone = self._build_zone_series()
        self.trans: List[Transition]=[]
        for zid, rows in self.per_zone.items():
            last_a = rows[0]["dimming"]; dwell  = 1
            for i in range(self.steps-1):
                cur, nxt = rows[i], rows[i+1]
                hour = cur["ts"].hour
                s = np.array([
                    cur["price"], cur["ef"], cur["activity"],
                    cur["lux"], cur["L_min"],
                    1.0 if cur["critical"] else 0.0,
                    1.0 if cur["complaint_zone"] else 0.0,
                    last_a,
                    math.sin(2*math.pi*hour/24.0), math.cos(2*math.pi*hour/24.0),
                    dwell/24.0
                ], dtype=np.float32)
                a = float(cur["dimming"])
                E = max(0.0, cur["power_kW"]) * self.dt_h
                cost   = E * max(0.0, cur["price"])
                carbon = E * max(0.0, cur["ef"])
                under_lux  = max(0.0, cur["L_min"] - cur["lux"])
                switched   = 1.0 if abs(a - last_a) >= 0.05 else 0.0
                glare_hit  = 1.0 if (abs(a - last_a) >= 0.15 and a >= float(self.limits["lighting_policy"]["glare_threshold"])) else 0.0
                complaint_hit = 0.0
                if cur["complaint_zone"]:
                    h=hour; s_h,e_h = self.limits["lighting_policy"]["complaint_sensitive_hours_local"]
                    if (h>=s_h or h<e_h) and a>=0.8: complaint_hit=1.0
                rw = self.limits["reward_weights_default"]
                r = -(cost + carbon) - rw["alpha_under_lux"]*under_lux - rw["beta_switch"]*switched \
                    - rw["gamma_glare"]*glare_hit - rw["delta_complaint"]*complaint_hit

                hour2 = nxt["ts"].hour
                s2 = np.array([
                    nxt["price"], nxt["ef"], nxt["activity"],
                    nxt["lux"], nxt["L_min"],
                    1.0 if nxt["critical"] else 0.0,
                    1.0 if nxt["complaint_zone"] else 0.0,
                    a,
                    math.sin(2*math.pi*hour2/24.0), math.cos(2*math.pi*hour2/24.0),
                    0.0 if switched==1.0 else min(24.0, dwell+1)/24.0
                ], dtype=np.float32)
                done = (i==self.steps-1-1)
                self.trans.append(Transition(s,a,r,s2,done))
                dwell = 1 if switched==1.0 else min(24, dwell+1)
                last_a=a

        arr = np.stack([t.s for t in self.trans], axis=0)
        self.mean = arr.mean(axis=0).astype(np.float32)
        self.std  = (arr.std(axis=0)+1e-6).astype(np.float32)

    def __len__(self): return len(self.trans)
    def __getitem__(self, idx: int):
        t=self.trans[idx]
        s  = ((t.s  - self.mean)/self.std).astype(np.float32)
        s2 = ((t.s_next - self.mean)/self.std).astype(np.float32)
        return torch.from_numpy(s), torch.tensor([t.a],dtype=torch.float32), torch.tensor([t.r],dtype=torch.float32), torch.from_numpy(s2), torch.tensor([float(t.done)],dtype=torch.float32)

    def _build_zone_series(self) -> Dict[str,List[Dict[str,Any]]]:
        raw={}
        for r in self.tele:
            zid = str(_pick(r,_ZONE_CANDS,""))
            t   = _ts_from_row(r)
            if not zid or not t or zid not in self.zone_meta: continue
            lux = _num_from_row(r,_LUX_CANDS,0.0)
            pkw = _num_from_row(r,_PWR_CANDS,0.0)
            d   = _num_from_row(r,_DIM_CANDS,1.0)
            d = d/100.0 if d>1.5 else d
            d = 1.0 if d<=0.05 else max(0.05,min(1.0,d))
            raw.setdefault(zid,[]).append((t,lux,pkw,d))

        price_map={}
        for r in self.price:
            t=_ts_from_row(r)
            if not t: continue
            price_map[t]=_num_from_row(r,_PRICE_CANDS,0.0)

        ef_map={}
        for r in self.ef:
            t=_ts_from_row(r)
            if not t: continue
            ef_map[t]=_num_from_row(r,_EF_CANDS,0.0)

        act_map={}
        tmp=[]
        for r in self.act:
            zid=str(_pick(r,_ZONE_CANDS,""))
            t=_ts_from_row(r)
            if not zid or not t: continue
            val=_num_from_row(r,_ACT_CANDS,0.0)
            tmp.append(val)
            act_map[(zid,t)]=val
        if tmp:
            mn,mx=min(tmp),max(tmp)
            if mx>1.5 and mx>mn:
                for k,v in list(act_map.items()):
                    act_map[k]=(v-mn)/(mx-mn+1e-9)

        levels = { z: float(self.zone_meta[z].get("L_min",20.0)) for z in self.zone_meta.keys() }
        wsum = sum(levels.values()) or 1.0
        all_p=[_num_from_row(r,_PWR_CANDS,0.0) for r in self.tele]
        all_p=[p for p in all_p if p>0]
        total_p=float(np.mean(all_p)) if all_p else 1.0

        per_zone={}
        for zid in self.zone_meta.keys():
            zs=raw.get(zid,[]); zs.sort(key=lambda x:x[0]); seq=[]
            for g in self.grid:
                if zs:
                    idx=int(np.argmin([abs((t-g).total_seconds()) for (t,_,__,___) in zs]))
                    lux0,p0,d0 = float(zs[idx][1]), float(zs[idx][2]), float(zs[idx][3])
                else:
                    lux0,p0,d0 = levels[zid]*1.2, total_p*(levels[zid]/wsum), 1.0
                if lux0<=0.5: lux0 = levels[zid]*1.2
                if p0<=0.0:  p0   = total_p*(levels[zid]/wsum)
                d0 = 1.0 if d0<=0.05 else max(0.05,min(1.0,d0))
                pr  = price_map.get(g, 0.0); eff = ef_map.get(g, 0.0)
                act = float(act_map.get((zid,g), 0.0) or 0.0)
                meta=self.zone_meta[zid]
                seq.append({
                    "ts": g, "zone_id": zid,
                    "price": pr, "ef": eff, "activity": act,
                    "lux": lux0, "power_kW": p0, "dimming": d0,
                    "L_min": float(meta.get("L_min",20.0)),
                    "critical": bool(meta.get("critical", False)),
                    "complaint_zone": bool(meta.get("complaint_zone", False))
                })
            per_zone[zid]=seq
        return per_zone

# -------------------- 模型 --------------------
class MLP(nn.Module):
    def __init__(self, inp:int, out:int, hidden:int=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(inp,hidden), nn.ReLU(),
                               nn.Linear(hidden,hidden), nn.ReLU(),
                               nn.Linear(hidden,out))
    def forward(self,x): return self.net(x)

class Policy(nn.Module):
    """ 输出 dimming∈[0,1] 的均值（Sigmoid） """
    def __init__(self, obs_dim:int, hidden:int=256):
        super().__init__()
        self.mu=MLP(obs_dim,1,hidden)
    def forward(self,s): return torch.sigmoid(self.mu(s))

# -------------------- IQL/CQL 基础工具 --------------------
def expectile_loss(err: torch.Tensor, tau: float) -> torch.Tensor:
    w=torch.where(err<0,tau,1-tau); return (w*err.pow(2)).mean()

@torch.no_grad()
def evaluate_offline(ds: LightingDataset, policy: Policy) -> Dict[str,float]:
    """
    轻量评估：不涉及价格/EF，只粗略估计 ΔkWh/峰值/切换/低照度次
    """
    policy.eval()
    S=torch.from_numpy(((np.stack([t.s for t in ds.trans])-ds.mean)/ds.std).astype(np.float32))
    A_hat=policy(S).cpu().numpy().reshape(-1)
    a_beh=np.array([t.a for t in ds.trans],dtype=np.float32)
    k=float(np.clip(A_hat.mean()/max(a_beh.mean(),1e-6),0.5,1.1))  # 粗缩放
    agg_base,agg_sim=[],[]
    under_lux,switches=0,0
    for _,seq in ds.per_zone.items():
        last_a=seq[0]["dimming"]
        for i in range(len(seq)):
            bpow=seq[i]["power_kW"]; bdim=max(0.05,seq[i]["dimming"])
            sdim=float(np.clip(bdim*k,0.05,1.0))
            sim_p=bpow*(sdim/bdim); lux_sim=seq[i]["lux"]*(sdim/bdim)
            if lux_sim<seq[i]["L_min"]-1e-6: under_lux+=1
            if abs(sdim-last_a)>=0.05: switches+=1
            last_a=sdim
            if len(agg_base)<=i: agg_base.append(0.0); agg_sim.append(0.0)
            agg_base[i]+=bpow; agg_sim[i]+=sim_p
    dt_h=DT_MIN/60.0
    base_kWh=float(np.sum(agg_base)*dt_h)
    sim_kWh=float(np.sum(agg_sim)*dt_h)
    return {
        "delta_kWh": max(0.0,base_kWh-sim_kWh),
        "peak_reduction_kW": max(0.0, max(agg_base or [0.0])-max(agg_sim or [0.0])),
        "under_lux": int(under_lux),
        "switches": int(switches)
    }

# -------------------- Offline-policy economics and reward decomposition --------------------
@torch.no_grad()
def economics_and_rewards_offline(ds: LightingDataset, policy: Policy) -> Dict[str,Any]:
    """
    计算 “规则/MPC 基线 vs 离线策略” 的经济指标与奖励分解（与 config_limits.json 口径一致）.
    返回结构将直接写入 offline_train.json / jsonl.
    """
    policy.eval()
    units = load_limits().get("units", {})
    rw    = load_limits().get("reward_weights_default", {})
    glare_thr = load_limits()["lighting_policy"]["glare_threshold"]
    carbon_price = float(load_limits().get("price_config", {}).get("carbon_price_yuan_per_kg", 0.0))
    obs_mean, obs_std = ds.mean, ds.std
    def pi_act(s_raw: np.ndarray) -> float:
        s = ((s_raw - obs_mean)/obs_std).astype(np.float32)
        a = float(policy(torch.tensor(s)[None,:]).cpu().numpy().reshape(-1)[0])
        return float(np.clip(a,0.05,1.0))

    # 聚合容器
    agg_b, agg_p = [], []
    cny_b = cny_p = 0.0   # 电费（人民币）
    kg_b  = kg_p  = 0.0   # 碳（kg）
    dt_h  = DT_MIN/60.0

    # 奖励分解（baseline/policy）
    rb = {"energy_cost": 0.0, "carbon_cost_kg": 0.0, "carbon_fee_yuan": 0.0,
          "under_lux_penalty": 0.0, "switch_penalty": 0.0, "glare_penalty": 0.0,
          "complaint_penalty": 0.0, "reward_total": 0.0}
    rp = {"energy_cost": 0.0, "carbon_cost_kg": 0.0, "carbon_fee_yuan": 0.0,
          "under_lux_penalty": 0.0, "switch_penalty": 0.0, "glare_penalty": 0.0,
          "complaint_penalty": 0.0, "reward_total": 0.0}
    # 奖励权重（alpha/beta/gamma/delta），从配置加载，并兜底默认
    limits = load_limits()
    rw = dict(limits.get("reward_weights_default", {}))
    rw.setdefault("alpha_under_lux", 5.0)
    rw.setdefault("beta_switch", 0.2)
    rw.setdefault("gamma_glare", 2.0)
    rw.setdefault("delta_complaint", 5.0)

    def add_reward(bucket, E, price, ef, under_lux_score, switched, glare_hit, complaint_hit, rw):
        energy_cost = E * price
        carbon_cost_kg = E * ef
        carbon_fee_yuan = carbon_cost_kg * carbon_price

        bucket["energy_cost"] += energy_cost
        bucket["carbon_cost_kg"] += carbon_cost_kg
        bucket["carbon_fee_yuan"] += carbon_fee_yuan
        bucket["under_lux_penalty"] += rw["alpha_under_lux"] * under_lux_score
        bucket["switch_penalty"] += rw["beta_switch"] * (1.0 if switched else 0.0)
        bucket["glare_penalty"] += rw["gamma_glare"] * (1.0 if glare_hit else 0.0)
        bucket["complaint_penalty"] += rw["delta_complaint"] * (1.0 if complaint_hit else 0.0)

        # 统一使用“钱”的口径：reward_total 为 负的总成本（越接近 0 越好）
        bucket["reward_total"] += -(energy_cost + carbon_fee_yuan) \
                                  - rw["alpha_under_lux"] * under_lux_score \
                                  - rw["beta_switch"] * (1.0 if switched else 0.0) \
                                  - rw["gamma_glare"] * (1.0 if glare_hit else 0.0) \
                                  - rw["delta_complaint"] * (1.0 if complaint_hit else 0.0)

    # 逐区逐步
    for _, seq in ds.per_zone.items():
        last_b = seq[0]["dimming"]; last_p = seq[0]["dimming"]
        for i,row in enumerate(seq):
            hour=row["ts"].hour
            s_raw=np.array([
                row["price"], row["ef"], row["activity"],
                row["lux"], row["L_min"],
                1.0 if row["critical"] else 0.0,
                1.0 if row["complaint_zone"] else 0.0,
                last_p,
                math.sin(2*math.pi*hour/24.0), math.cos(2*math.pi*hour/24.0),
                0.0
            ],dtype=np.float32)

            # 规则基线动作 & 策略动作
            a_base = _rule_baseline_action(s_raw)
            a_pol  = pi_act(s_raw)

            base_a = max(0.05, float(row["dimming"]))
            scale_b = a_base/base_a
            scale_p = a_pol/base_a

            p_b = max(0.0, float(row["power_kW"])) * scale_b
            p_p = max(0.0, float(row["power_kW"])) * scale_p

            if len(agg_b)<=i: agg_b.append(0.0); agg_p.append(0.0)
            agg_b[i]+=p_b; agg_p[i]+=p_p

            E_b = p_b * dt_h; E_p = p_p * dt_h
            price = max(0.0, float(row["price"])); ef = max(0.0, float(row["ef"]))
            cny_b += E_b * price; cny_p += E_p * price
            kg_b  += E_b * ef;    kg_p  += E_p * ef

            lux_b = row["lux"] * (a_base / base_a);
            lux_p = row["lux"] * (a_pol / base_a)
            # 照度惩罚采用“相对缺口(%) × 时间(小时)”，并对每步做上限裁剪，避免惩罚爆表
            Lmin = max(1e-6, float(row["L_min"]))
            ulx_ratio_b = max(0.0, (Lmin - lux_b) / Lmin)
            ulx_ratio_p = max(0.0, (Lmin - lux_p) / Lmin)
            ulx_clip = load_limits().get("penalty_config", {}).get("under_lux_clip_per_step", 0.5)
            under_b = min(ulx_clip, ulx_ratio_b * dt_h)  # 无量纲（百分比×小时）
            under_p = min(ulx_clip, ulx_ratio_p * dt_h)

            sw_b = abs(a_base - last_b) >= 0.05
            sw_p = abs(a_pol  - last_p) >= 0.05
            glare_b = (abs(a_base - last_b) >= 0.15 and a_base >= glare_thr)
            glare_p = (abs(a_pol  - last_p) >= 0.15 and a_pol  >= glare_thr)
            comp_b = comp_p = False
            if row["complaint_zone"]:
                s_h,e_h = load_limits()["lighting_policy"]["complaint_sensitive_hours_local"]
                if (hour>=s_h or hour<e_h):
                    comp_b = (a_base>=0.8); comp_p = (a_pol>=0.8)

            add_reward(rb, E_b, price, ef, under_b, sw_b, glare_b, comp_b, rw)
            add_reward(rp, E_p, price, ef, under_p, sw_p, glare_p, comp_p, rw)

            last_b=a_base; last_p=a_pol

    base_kWh=float(np.sum(agg_b)*(DT_MIN/60.0))
    pol_kWh =float(np.sum(agg_p)*(DT_MIN/60.0))

    out = {
        "ts": _now_utc_iso(),
        "algorithm": _rjson(POLICY_META,{}).get("algo","IQL/CQL"),
        "units": units,
        "baseline": {
            "energy_cost_cny": round(cny_b,3), "kWh": round(base_kWh,3),
            "peak_kW": round(max(agg_b or [0.0]),3), "carbon_kg": round(kg_b,3),
            "rewards": { k: round(v,6) for k,v in rb.items() }
        },
        "policy":   {
            "energy_cost_cny": round(cny_p,3), "kWh": round(pol_kWh,3),
            "peak_kW": round(max(agg_p or [0.0]),3), "carbon_kg": round(kg_p,3),
            "rewards": { k: round(v,6) for k,v in rp.items() }
        },
        "savings":  {
            "cny": round(cny_b-cny_p,3),
            "percent": round((0.0 if cny_b<=1e-9 else (cny_b-cny_p)/cny_b*100.0),3),
            "kWh": round(base_kWh-pol_kWh,3),
            "peak_kW": round(max(0.0, max(agg_b or [0.0])-max(agg_p or [0.0])),3),
            "carbon_kg": round(kg_b-kg_p,3)
        },
        "reward_weights": load_limits().get("reward_weights_default", {})
    }
    return out

# -------------------- 基线动作（规则） --------------------
def _rule_baseline_action(obs: np.ndarray) -> float:
    activity=obs[2]; critical=obs[5]; prev_a=obs[7]
    if critical>0.5: tgt=0.95
    else:
        if activity>=0.6: tgt=0.85
        elif activity>=0.3: tgt=0.7
        else: tgt=0.5
        tgt=max(tgt-0.05,0.4)
    return float(np.clip(0.8*prev_a+0.2*tgt,0.05,1.0))

# -------------------- 安全层/残差（在线） --------------------
class SafetyLayer:
    def __init__(self, limits: Dict[str,Any]):
        lp=limits["lighting_policy"]
        self.min_dwell_min    = int(lp["min_dwell_min_default"])
        self.min_interval_min = int(lp["min_interval_min_default"])
        self.max_switches     = int(lp["max_switches_per_night"])
        self.ramp_pct_step    = float(lp["ramp_percent_per_step_default"])
        self.glare_thr        = float(lp["glare_threshold"])
    def project(self, prev_a: float, cand_a: float, switches_used:int) -> Tuple[float,bool]:
        ramp=self.ramp_pct_step/100.0
        a=float(np.clip(cand_a, prev_a-ramp, prev_a+ramp))
        if switches_used>=self.max_switches and abs(a-prev_a)>=0.15:
            a=prev_a+np.sign(a-prev_a)*0.05
        a=float(np.clip(a,0.05,1.0))
        return a,(abs(a-cand_a)>1e-6)

class ResidualActor(nn.Module):
    def __init__(self, obs_dim:int, hidden:int=256, bound:float=0.2):
        super().__init__()
        self.net=MLP(obs_dim,1,hidden); self.bound=bound
    def forward(self,s): return torch.tanh(self.net(s))*self.bound

class TwinQ(nn.Module):
    def __init__(self, obs_dim:int):
        super().__init__()
        self.q1=MLP(obs_dim+1,1,256); self.q2=MLP(obs_dim+1,1,256)
    def forward(self,s,a):
        sa=torch.cat([s,a],1); return self.q1(sa), self.q2(sa)

@dataclass
class ReplayItem:
    s: np.ndarray; a: float; r: float; s2: np.ndarray; done: float

class Replay:
    def __init__(self, cap:int=200000):
        self.cap=cap; self.buf: List[ReplayItem]=[]; self.ptr=0
    def push(self,it:ReplayItem):
        if len(self.buf)<self.cap: self.buf.append(it)
        else: self.buf[self.ptr]=it; self.ptr=(self.ptr+1)%self.cap
    def sample(self,n:int):
        idx=np.random.choice(len(self.buf),size=min(n,len(self.buf)),replace=False)
        b=[self.buf[i] for i in idx]
        s=np.stack([x.s for x in b]); a=np.array([x.a for x in b])[:,None]
        r=np.array([x.r for x in b])[:,None]; s2=np.stack([x.s2 for x in b]); d=np.array([x.done for x in b])[:,None]
        return (torch.tensor(s,dtype=torch.float32), torch.tensor(a,dtype=torch.float32),
                torch.tensor(r,dtype=torch.float32), torch.tensor(s2,dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))
    def __len__(self): return len(self.buf)

# -------------------- Offline training and snapshot persistence --------------------
def _save_offline_policy(weights: Dict[str,Any], ds: LightingDataset, algo:str, step:int, score:float):
    torch.save(weights, POLICY_BIN)
    meta={
        "algo": algo.upper(),
        "updated_at": _now_utc_iso(),
        "step": step, "score": score,
        "obs_mean": ds.mean.tolist(), "obs_std": ds.std.tolist(),
        "obs_keys": ["price","ef","activity","lux","L_min","critical","complaint_zone","prev_a","sin_h","cos_h","dwell"],
        "action": "dimming_percent[0..1]",
        "reward_weights": load_limits().get("reward_weights_default", {}),
        "note": "policy.bin=pi/v/q1/q2；api.py 会加载本策略并可叠加 residual_policy.bin"
    }
    POLICY_META.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")

def _write_offline_logs_line(obj: Dict[str,Any]):
    # 逐行历史
    with OFFLINE_LOG_JSONL.open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,ensure_ascii=False)+"\n")
    # 最新快照（覆盖）
    OFFLINE_SNAPSHOT.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def train_offline(algo:str="iql", steps:int=30000, batch_size:int=512, seed:int=42,
                  tau:float=0.7, beta:float=3.0, lr:float=3e-4, log_every:int=500,
                  sleep_every:int=0, sleep_sec:int=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds=LightingDataset()
    dl=torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True,drop_last=True)
    obs_dim=len(ds.mean)
    q1,q2=MLP(obs_dim+1,1).to(DEVICE),MLP(obs_dim+1,1).to(DEVICE)
    v=MLP(obs_dim,1).to(DEVICE)
    pi=Policy(obs_dim).to(DEVICE)
    opt_q=torch.optim.Adam(list(q1.parameters())+list(q2.parameters()),lr=lr)
    opt_v=torch.optim.Adam(v.parameters(),lr=lr)
    opt_pi=torch.optim.Adam(pi.parameters(),lr=lr)

    # 清空旧日志（可选）
    if OFFLINE_LOG_JSONL.exists(): OFFLINE_LOG_JSONL.unlink()

    def pack(b): s,a,r,s2,d=b; return s.to(DEVICE),a.to(DEVICE),r.to(DEVICE),s2.to(DEVICE),d.to(DEVICE)
    best_score=-1e18; best=None; it=0
    for _ in range(10**9):
        for batch in dl:
            it+=1
            s,a,r,s2,d=pack(batch)

            # ----- IQL/CQL 训练主循环 -----
            with torch.no_grad():
                qa1=q1(torch.cat([s,a],1)); qa2=q2(torch.cat([s,a],1))
                q_min=torch.min(qa1,qa2)
            v_pred=v(s)
            v_loss=expectile_loss(q_min-v_pred,tau)
            opt_v.zero_grad(); v_loss.backward(); opt_v.step()

            with torch.no_grad():
                vs2=v(s2); target=r+(1.0-d)*GAMMA*vs2
            q1p=q1(torch.cat([s,a],1)); q2p=q2(torch.cat([s,a],1))
            q_loss=(F.mse_loss(q1p,target)+F.mse_loss(q2p,target))*0.5
            if algo.lower()=="cql":
                u=torch.rand_like(a); a_pi=pi(s).detach(); a_rand=0.5*u+0.5*a_pi
                lse=torch.logsumexp(torch.cat([q1(torch.cat([s,a_rand],1)),q2(torch.cat([s,a_rand],1))],1),1).mean()
                q_mean=(q1p+q2p).mean(); q_loss=q_loss+1.0*(lse-q_mean)
            opt_q.zero_grad(); q_loss.backward(); opt_q.step()

            with torch.no_grad():
                adv=torch.min(q1(torch.cat([s,a],1)),q2(torch.cat([s,a],1))) - v(s)
                w=torch.clamp(torch.exp(adv/beta),max=100.0)
            a_pi=pi(s); pi_loss=(w*(a_pi-a).pow(2)).mean()
            opt_pi.zero_grad(); pi_loss.backward(); opt_pi.step()
            # ----- IQL/CQL 训练主循环 END -----

            # 日志与评估（每 log_every 步）
            if it%log_every==0 or it==1:
                pi.eval()
                ev=evaluate_offline(ds,pi)  # 轻量 ΔkWh/峰值/违规/切换
                econ = economics_and_rewards_offline(ds, pi)  # Economics and reward decomposition
                # 便于快速比较的综合分数（仅用于保存最优）
                # 以“奖励增益”为主，不再对 under_lux 做二次放大；更贴近最终目标
                score = (econ["policy"]["rewards"]["reward_total"] - econ["baseline"]["rewards"]["reward_total"])

                if score>best_score:
                    best_score=score; best={"pi":pi.state_dict(),"v":v.state_dict(),"q1":q1.state_dict(),"q2":q2.state_dict()}
                    _save_offline_policy(best,ds,algo,it,score)

                # 终端打印
                print(f"[OFFLINE-{algo.upper()}] step={it} v={v_loss.item():.4f} q={q_loss.item():.4f} pi={pi_loss.item():.4f} "
                      f"ΔkWh={ev['delta_kWh']:.1f} peak↓={ev['peak_reduction_kW']:.1f} "
                      f"savings¥={econ['savings']['cny']:.2f} reward_gain={(econ['policy']['rewards']['reward_total']-econ['baseline']['rewards']['reward_total']):.2f}")

                # —— 步级别名 + 窗口级指标（前端卡片用窗口级，曲线用步级） ——
                ev.setdefault("delta_kWh_step", ev.get("delta_kWh"))
                ev.setdefault("peak_reduction_kW_step", ev.get("peak_reduction_kW"))
                metrics_window = {
                    "delta_kWh_window": float(econ["baseline"].get("kWh", econ["baseline"].get("energy_kWh", 0.0))) -
                                        float(econ["policy"].get("kWh", econ["policy"].get("energy_kWh", 0.0))),
                    "peak_reduction_kW_window": float(econ["baseline"].get("peak_kW", 0.0)) -
                                                float(econ["policy"].get("peak_kW", 0.0))
                }

                # 历史行 + 快照（把“奖励值 + 经济节省”一并写入）
                log_obj = {
                    "ts": _now_utc_iso(),
                    "algo": algo,
                    "step": it,
                    "v_loss": float(v_loss.item()), "q_loss": float(q_loss.item()), "pi_loss": float(pi_loss.item()),
                    "metrics": ev,                      # ΔkWh / peak↓ / underLux / switches（轻量）
                    "economics": econ,                  # 人民币/碳/峰值/kWh + savings
                    "rewards": {                        # 直出奖励增益（钱的口径）
                        "baseline": econ["baseline"]["rewards"],
                        "policy":   econ["policy"]["rewards"],
                        "gain":     round(econ["policy"]["rewards"]["reward_total"] - econ["baseline"]["rewards"]["reward_total"], 6)
                    },
                    "metrics_window": metrics_window,

                    "units": load_limits().get("units", {}),
                    "reward_weights": load_limits().get("reward_weights_default", {})
                }
                _write_offline_logs_line(log_obj)

            # 周期性休息（防过热）
            if sleep_every>0 and it%sleep_every==0:
                print(f"[PAUSE] sleeping {sleep_sec}s to cool down CPU...", flush=True)
                time.sleep(max(0,int(sleep_sec)))

            if it>=steps: break
        if it>=steps: break

    if best is not None: _save_offline_policy(best,ds,algo,it,best_score)
    print("[OFFLINE] training done.")

# -------------------- 在线 Residual Safe‑SAC（保留，未改动日志口径） --------------------
def train_online_residual(steps:int=10000, batch:int=256, log_every:int=500, seed:int=123,
                          residual_bound:float=0.2, lr:float=3e-4, alpha:float=0.2,
                          sleep_every:int=0, sleep_sec:int=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds=LightingDataset(); obs_dim=len(ds.mean)
    actor=ResidualActor(obs_dim,256,residual_bound).to(DEVICE)
    twin=TwinQ(obs_dim).to(DEVICE)
    opt_a=torch.optim.Adam(actor.parameters(),lr=lr)
    opt_q=torch.optim.Adam(twin.parameters(),lr=lr)
    safety=SafetyLayer(load_limits())
    if ONLINE_LOG_JSONL.exists(): ONLINE_LOG_JSONL.unlink()
    rb=Replay(200000); step=0
    for zid, seq in ds.per_zone.items():
        switches=0; last_a=seq[0]["dimming"]; dwell_min=0
        for i in range(len(seq)-1):
            cur,nxt=seq[i],seq[i+1]
            s_raw=np.array([
                cur["price"], cur["ef"], cur["activity"],
                cur["lux"], cur["L_min"],
                1.0 if cur["critical"] else 0.0,
                1.0 if cur["complaint_zone"] else 0.0,
                last_a,
                math.sin(2*math.pi*(cur["ts"].hour)/24.0), math.cos(2*math.pi*(cur["ts"].hour)/24.0),
                min(24.0,dwell_min)/24.0
            ],dtype=np.float32)
            s=((s_raw-ds.mean)/ds.std).astype(np.float32); s=torch.tensor(s)[None,:]
            # 基线 + 残差 + 安全层
            a_base=_rule_baseline_action(s_raw)
            delta=actor(s).detach().cpu().numpy().item()
            a_cand=float(np.clip(a_base+delta,0.05,1.0))
            a_proj,_=safety.project(last_a,a_cand,switches)

            # 回报
            E=max(0.0,cur["power_kW"])*(DT_MIN/60.0)
            cost=E*max(0.0,cur["price"]); carbon=E*max(0.0,cur["ef"])
            Lmin = max(1e-6, float(cur["L_min"]))
            ulx_ratio = max(0.0, (Lmin - cur["lux"] * (a_proj / max(0.05, last_a))) / Lmin)
            ulx_clip = load_limits().get("penalty_config", {}).get("under_lux_clip_per_step", 0.5)
            under_lux = min(ulx_clip, ulx_ratio * (DT_MIN / 60.0))  # 百分比×小时

            switched=1.0 if abs(a_proj-last_a)>=0.05 else 0.0
            glare_hit=1.0 if (abs(a_proj-last_a)>=0.15 and a_proj>=safety.glare_thr) else 0.0
            complaint_hit=0.0
            if cur["complaint_zone"]:
                h=cur["ts"].hour; s_h,e_h=load_limits()["lighting_policy"]["complaint_sensitive_hours_local"]
                if (h>=s_h or h<e_h) and a_proj>=0.8: complaint_hit=1.0
            rw=load_limits()["reward_weights_default"]
            r=-(cost+carbon) - rw["alpha_under_lux"]*under_lux - rw["beta_switch"]*switched - rw["gamma_glare"]*glare_hit - rw["delta_complaint"]*complaint_hit

            s2_raw=np.array([
                nxt["price"], nxt["ef"], nxt["activity"],
                nxt["lux"], nxt["L_min"],
                1.0 if nxt["critical"] else 0.0,
                1.0 if nxt["complaint_zone"] else 0.0,
                a_proj,
                math.sin(2*math.pi*(nxt["ts"].hour)/24.0), math.cos(2*math.pi*(nxt["ts"].hour)/24.0),
                0.0 if switched==1.0 else min(24.0,dwell_min+DT_MIN)/24.0
            ],dtype=np.float32)
            done=1.0 if i==len(seq)-2 else 0.0
            rb.push(ReplayItem(s_raw,a_proj,r,s2_raw,done))
            last_a=a_proj; switches+=int(switched); dwell_min=0 if switched==1.0 else dwell_min+DT_MIN

            if len(rb)>=batch:
                bs,ba,br,bs2,bd=rb.sample(batch)
                bs=((bs-torch.tensor(ds.mean))/torch.tensor(ds.std)); bs2=((bs2-torch.tensor(ds.mean))/torch.tensor(ds.std))
                with torch.no_grad():
                    a2=actor(bs2)
                    q1_t,q2_t=twin(bs2,a2)
                    q_t=torch.min(q1_t,q2_t) - alpha*torch.log(torch.clamp(1.0-(a2.abs()/residual_bound),min=1e-6))
                    y=br + (1.0-bd)*GAMMA*q_t
                q1,q2=twin(bs,ba)
                q_loss=F.mse_loss(q1,y)+F.mse_loss(q2,y); opt_q.zero_grad(); q_loss.backward(); opt_q.step()
                a=actor(bs); q1a,q2a=twin(bs,a); q_min=torch.min(q1a,q2a)
                pi_loss=(-q_min + alpha*torch.log(torch.clamp(1.0-(a.abs()/residual_bound),min=1e-6))).mean()
                opt_a.zero_grad(); pi_loss.backward(); opt_a.step()

            step+=1
            if step%log_every==0:
                print(f"[ONLINE-RES] step={step} rb={len(rb)}")
                with ONLINE_LOG_JSONL.open("a",encoding="utf-8") as f:
                    f.write(json.dumps({"ts":_now_utc_iso(),"step":step,"rb":len(rb)},ensure_ascii=False)+"\n")
            if sleep_every>0 and step%sleep_every==0:
                print(f"[PAUSE] sleeping {sleep_sec}s to cool down CPU...", flush=True)
                time.sleep(max(0,int(sleep_sec)))
            if step>=steps:
                torch.save(actor.state_dict(),RESIDUAL_BIN)
                print("[ONLINE] residual saved:", RESIDUAL_BIN)
                return
    torch.save(actor.state_dict(),RESIDUAL_BIN)
    print("[ONLINE] residual saved:", RESIDUAL_BIN)

# -------------------- SANITY --------------------
def sanity():
    tables={
        "zones_master": _rcsv(DATA_DIR/"zones_master.csv"),
        "lighting_telemetry": _rcsv(DATA_DIR/"lighting_telemetry.csv"),
        "activity_forecast": _rcsv(DATA_DIR/"activity_forecast.csv"),
        "weather_astro": _rcsv(DATA_DIR/"weather_astro.csv"),
        "market_price": _rcsv(DATA_DIR/"market_price.csv"),
        "grid_ef": _rcsv(DATA_DIR/"grid_ef.csv"),
        "complaints_events": _rcsv(DATA_DIR/"complaints_events.csv"),
    }
    def colset(rows):
        return sorted(list({k for r in rows for k in r.keys()}))[:50]
    print("[SANITY] files in data/:")
    for k,v in tables.items():
        print(f"  - {k}: rows={len(v)} cols={len(colset(v))} sample_cols={colset(v)}")
    cand=[]
    if tables["market_price"]:
        cand=[_ts_from_row(r) for r in tables["market_price"] if _ts_from_row(r)]
    if not cand and tables["grid_ef"]:
        cand=[_ts_from_row(r) for r in tables["grid_ef"] if _ts_from_row(r)]
    if not cand and tables["lighting_telemetry"]:
        cand=list({ _ts_from_row(r) for r in tables["lighting_telemetry"] if _ts_from_row(r) })
        cand=sorted(cand)
    print(f"[SANITY] inferred grid steps: {len(cand)} (print first 5):", [c.isoformat() for c in cand[:5]])
    return {"steps": len(cand), "head_ts":[c.isoformat() for c in cand[:5]]}

# -------------------- CLI --------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--src", type=str, default="/mnt/data")
    ap.add_argument("--sanity", action="store_true", help="打印列名映射与时间网格样例")

    # 离线
    ap.add_argument("--train-offline", action="store_true")
    ap.add_argument("--algo", type=str, default="iql", choices=["iql","cql"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep-every", type=int, default=0, help="每多少步休息一次（0=不休息）")
    ap.add_argument("--sleep-sec", type=int, default=0, help="每次休息秒数")

    # 在线
    ap.add_argument("--train-online", action="store_true")
    ap.add_argument("--online-steps", type=int, default=10000)
    ap.add_argument("--online-batch", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--residual-bound", type=float, default=0.2)

    args=ap.parse_args()

    if args.ingest: ingest_from(args.src)
    if args.sanity: sanity()
    if args.train_offline:
        train_offline(algo=args.algo, steps=args.steps, batch_size=args.batch, seed=args.seed,
                      tau=args.tau, beta=args.beta, lr=args.lr, log_every=args.log_every,
                      sleep_every=args.sleep_every, sleep_sec=args.sleep_sec)
    if args.train_online:
        train_online_residual(steps=args.online_steps, batch=args.online_batch, log_every=args.log_every,
                              seed=123, residual_bound=args.residual_bound, lr=3e-4, alpha=args.alpha,
                              sleep_every=args.sleep_every, sleep_sec=args.sleep_sec)

if __name__=="__main__":
    main()
