from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json, math, random, time
from statistics import mean

from .domain import (
    FidelityGroup, FidelityRadar, FidelityPayload, FidelityResponse,
    ParamChange, ScenarioRunResponse, CalibrateResponse, ReplayResponse
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_RNG = random.Random(20251105)  # 可改为随时间或配置

def _read(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def get_port_profile() -> dict:
    return _read(DATA_DIR / "port_profile.json", {})

def get_params() -> Tuple[dict, dict]:
    base = _read(DATA_DIR / "params_base.json", {})
    last = _read(DATA_DIR / "last_calib.json", base)
    return base, last

def _save_last_params(params: dict):
    (DATA_DIR / "last_calib.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

# ---------- 误差&覆盖率生成 ----------
def _gen_group_metrics(group: str, N: int, noise_kw: float, param_boost: float) -> FidelityGroup:
    """
    生成每个设备组的 MAE/RMSE/MAPE（越小越好）。
    param_boost: 由参数质量、规模等带来的总体“更贴合”系数，>1 更好。
    """
    # 设备组规模差异：QC/YC 波动略大，BESS/Shore 稳定
    group_scale = {"QC": 1.15, "YC": 1.05, "AGV": 0.95, "BESS": 0.75, "Shore": 0.85}.get(group, 1.0)
    s = max(0.5, noise_kw * group_scale / param_boost)
    # 构造“真实”与“孪生”误差向量
    err = [abs(_RNG.gauss(0, s)) for _ in range(N)]
    mae = mean(err)
    rmse = math.sqrt(mean([e*e for e in err]))
    # 用一个名义基准（功率等级）来换算 MAPE
    denom = 200.0 if group in ("QC", "YC") else (80.0 if group=="AGV" else 150.0)
    mape = mae / max(1e-6, denom)
    return FidelityGroup(group=group, mae_kw=mae, rmse_kw=rmse, mape_pct=mape)

def _coverage_from_params(target: float, boost: float) -> float:
    # 覆盖率受参数质量（boost）正向影响，但上限 0.98
    c = min(0.98, target * (0.86 + 0.12 * (boost - 0.8)))
    return max(0.5, c)

def _score(groups: List[FidelityGroup], coverage: float, stress: float) -> float:
    # 综合得分：误差三项的倒数 + 覆盖率 + 压测
    # 归一化到 0~1，偏保守
    mae_n = sum(g.mae_kw for g in groups) / (len(groups) * 10.0)   # 10kW 视为良好水平
    rmse_n = sum(g.rmse_kw for g in groups) / (len(groups) * 14.0) # 14kW 良好
    mape_n = sum(g.mape_pct for g in groups) / len(groups) / 0.05  # 5% 良好
    raw = 0.35*(1/(1+mae_n)) + 0.25*(1/(1+rmse_n)) + 0.15*(1/(1+mape_n)) + 0.15*coverage + 0.10*stress
    return max(0.0, min(1.0, raw))

# ---------- 场景通过率 ----------
def _scenario_rates(base: float) -> Dict[str, float]:
    # 不同场景基于 base 做扰动
    clamp = lambda x: max(0.35, min(0.98, x))
    return {
        "typhoon":        clamp(base - 0.06 + _RNG.uniform(-0.01, 0.01)),
        "dense_berthing": clamp(base - 0.04 + _RNG.uniform(-0.01, 0.01)),
        "islanded":       clamp(base - 0.08 + _RNG.uniform(-0.01, 0.01)),
        "heatwave":       clamp(base - 0.05 + _RNG.uniform(-0.01, 0.01)),
        "derate":         clamp(base - 0.03 + _RNG.uniform(-0.01, 0.01))
    }

# ---------- 雷达 ----------
def _radar_from_params(old: dict, new: dict) -> FidelityRadar:
    labels = ["eff","loss","ramp","cap","delay"]
    # 把物理量换到 0~1：越大越好（loss 取反）
    def norm(p):
        return {
            "eff":  (p.get("bess_eff", 0.9) - 0.85) / 0.15,            # 0.85~1.0
            "loss": 1.0 - (1.0 - p.get("shore_pf", 0.9)) / 0.2,       # PF 越高越好
            "ramp": min(1.0, p.get("ramp_kw_s", 400) / 600.0),
            "cap":  p.get("cap_norm", 0.75),
            "delay":1.0 - p.get("delay_norm", 0.5)
        }
    o = norm(old); n = norm(new)
    old_v = [max(0.0, min(1.0, o[k])) for k in ["eff","loss","ramp","cap","delay"]]
    new_v = [max(0.0, min(1.0, n[k])) for k in ["eff","loss","ramp","cap","delay"]]
    return FidelityRadar(labels=labels, old=old_v, new=new_v)

# ---------- 对外函数 ----------
def compute_fidelity() -> FidelityResponse:
    prof = get_port_profile()
    base, last = get_params()
    groups = prof.get("groups", ["QC","YC","AGV","BESS","Shore"])
    N = int(prof.get("obs_per_group", 400))
    noise = float(prof.get("noise_kw", 6.0))
    # 参数“增益”估计：越优的参数 => 误差更小、覆盖率更高
    boost = 0.9 + 0.05 * ((base.get("bess_eff",0.94) + last.get("bess_eff",0.94)) - 1.8) \
                  + 0.03 * (last.get("shore_pf",0.92) - 0.9)
    metrics: List[FidelityGroup] = [_gen_group_metrics(g, N, noise, boost) for g in groups]

    # 场景通过率和 stress
    base_rate = 0.76 + 0.04*(boost-0.9)
    scen_rates = _scenario_rates(base=base_rate)
    stress = mean(scen_rates.values())

    coverage = _coverage_from_params(prof.get("coverage_target", 0.9), boost)
    score = _score(metrics, coverage, stress)

    # 参数变化（和上一次校准比差异）
    changes: List[ParamChange] = []
    keys = ["bess_eff","ramp_kw_s","thermal_max_C","shore_pf","cap_norm","delay_norm"]
    for k in keys:
        o = float(base.get(k, 0.0))
        n = float(last.get(k, o))
        changes.append(ParamChange(name=k, old=o, new=n, delta=(n-o)))

    radar = _radar_from_params(base, last)
    payload = FidelityPayload(
        score=score, coverage=coverage, stress=stress, groups=metrics,
        scenarios=scen_rates, params=changes, radar=radar,
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    return FidelityResponse(fidelity=payload)

def run_scenario(scenario: str) -> ScenarioRunResponse:
    # 根据场景类型对通过率做不同幅度扰动
    base = 0.75 + _RNG.uniform(-0.02, 0.03)
    rates = _scenario_rates(base)
    bump = {"baseline":0.00, "dense_berthing":-0.02, "heatwave":-0.03, "typhoon":-0.05, "islanded":-0.06}
    if scenario in bump:  # 强化该场景的影响
        rates[scenario if scenario!="baseline" else "dense_berthing"] = max(0.35, min(0.98, base + bump.get(scenario,0) + _RNG.uniform(-0.01,0.01)))
    return ScenarioRunResponse(pass_rate=mean(rates.values()), rates=rates)

def calibrate() -> CalibrateResponse:
    base, last = get_params()
    new_params = dict(last)
    # 做一些“合理”的微调：提升效率、提升 PF、放宽少许爬坡与热上限
    new_params["bess_eff"] = min(0.985, last.get("bess_eff",0.94) + _RNG.uniform(0.005, 0.012))
    new_params["shore_pf"] = min(0.98,  last.get("shore_pf",0.92) + _RNG.uniform(0.005, 0.015))
    new_params["ramp_kw_s"] = min(600,  last.get("ramp_kw_s",430) + _RNG.uniform(10, 40))
    new_params["thermal_max_C"] = min(90, last.get("thermal_max_C",86) + _RNG.uniform(0.5, 1.5))
    # cap/delay 轻微优化
    new_params["cap_norm"] = min(0.9,  last.get("cap_norm",0.78) + _RNG.uniform(0.01, 0.03))
    new_params["delay_norm"] = max(0.35, last.get("delay_norm",0.52) - _RNG.uniform(0.01, 0.03))

    # 记录变化
    chgs: List[ParamChange] = []
    for k in new_params.keys():
        o = float(last.get(k, base.get(k, 0.0))); n = float(new_params[k])
        chgs.append(ParamChange(name=k, old=o, new=n, delta=(n-o)))

    _save_last_params(new_params)

    # 校准后刷新一次 fidelity
    fid = compute_fidelity().fidelity
    return CalibrateResponse(changed_params=chgs, fidelity=fid)

def replay() -> ReplayResponse:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tid = f"tw_{_RNG.randrange(10**8, 10**9-1)}"
    return ReplayResponse(replayed_at=ts, trace_id=tid)
# ---------- 上海大港口数据引导（Bootstrap） ----------

import json
from pathlib import Path
import random

# 供 _shanghai_last_calib 使用
_RNG = random.Random()

DATA_DIR = Path(__file__).resolve().parent / "data"


def _shanghai_profile() -> dict:
    """
    TwinPlus 用“上海港·洋山深水港”示例画像。
    """
    return {
        "port_name": "上海港·洋山深水港",
        "teu_annual": 49_000_000,              # 年度吞吐量
        "vessel_calls_year": 52_000,           # 年靠泊艘次（示例）
        "berth_count": 58,
        "deepwater_berths": 42,
        "shore_power_coverage": 0.82,          # 岸电覆盖比例
        "bess_mwh": 60,                        # 储能规模
        "shore_power_mva": 160,                # 岸电容量

        # TwinPlus 所需
        "groups": ["QC", "YC", "AGV", "BESS", "Shore"],
        "obs_per_group": 800,
        "noise_kw": 5.0,
        "coverage_target": 0.92,

        # 运营 KPI（示例）
        "ops": {
            "teu_per_qc_hour": 35.0,
            "best_hour_teu": 50.0,
            "avg_berth_time_hour": 28.0,
            "truck_turnaround_p50_min": 26,
            "truck_turnaround_p95_min": 45,
            "yard_occupancy": 0.68,
            "gate_appointment_rate": 0.85,
            "on_time_departure_rate": 0.92,
        },

        # 能源 & 碳（示例）
        "energy": {
            "annual_power_mwh": 110_000.0,
            "annual_power_cost_cny_million": 3.4,
            "renewables_share": 0.22,
            "shore_power_share": 0.18,
            "bess_share": 0.06,
        }
    }


def _shanghai_params_base() -> dict:
    """
    TwinPlus 底层模型的参数基线。
    """
    return {
        "bess_eff": 0.94,
        "ramp_kw_s": 460,
        "thermal_max_C": 85,
        "shore_pf": 0.93,
        "cap_norm": 0.80,
        "delay_norm": 0.50,
    }


def _shanghai_last_calib(base: dict) -> dict:
    """
    模拟“最近一次标定结果”，在 base 基础上加入轻微扰动（±2~6%）。
    """

    def jitter(v, r=0.03):
        return float(max(0.0, v * (1.0 + _RNG.uniform(-r, r))))

    return {
        "bess_eff": jitter(base["bess_eff"], 0.02),
        "ramp_kw_s": jitter(base["ramp_kw_s"], 0.06),
        "thermal_max_C": jitter(base["thermal_max_C"], 0.02),
        "shore_pf": jitter(base["shore_pf"], 0.02),
        "cap_norm": jitter(base["cap_norm"], 0.04),
        "delay_norm": jitter(base["delay_norm"], 0.04),
    }


def bootstrap_shanghai() -> dict:
    """
    在 TwinPlus 数据目录下生成三份示例文件：

      - port_profile.json
      - params_base.json
      - last_calib.json

    已存在的不会覆盖。

    返回结构：
    {
        "data_dir": "...",
        "written": ["port_profile.json", ...]
    }
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    written = []

    p_profile = DATA_DIR / "port_profile.json"
    p_base = DATA_DIR / "params_base.json"
    p_last = DATA_DIR / "last_calib.json"

    # 1) port_profile.json
    if not p_profile.exists():
        p_profile.write_text(
            json.dumps(_shanghai_profile(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        written.append(p_profile.name)

    # 2) params_base.json
    if not p_base.exists():
        base = _shanghai_params_base()
        p_base.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_base.name)
    else:
        base = json.loads(p_base.read_text(encoding="utf-8"))

    # 3) last_calib.json
    if not p_last.exists():
        last = _shanghai_last_calib(base)
        p_last.write_text(json.dumps(last, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_last.name)

    return {"data_dir": str(DATA_DIR), "written": written}


def ensure_data(port: str = "shanghai") -> None:
    """
    若缺少关键文件，则自动生成示例数据。
    """
    if port.lower() != "shanghai":
        return

    need = []
    for name in ("port_profile.json", "params_base.json", "last_calib.json"):
        if not (DATA_DIR / name).exists():
            need.append(name)

    if need:
        bootstrap_shanghai()
