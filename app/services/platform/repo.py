from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json, random, time

from .domain import PolicyItem, Leaderboard, SafetyRule, SafetySummary, ActionsHist
from .rl.eval_ope import cvar95, mape_from_deltas

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 为了可复现，固定随机种子；你可改为随时间滚动
_RNG = random.Random(20251105)

def _read_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def get_port_profile() -> dict:
    path = DATA_DIR / "port_profile.json"
    prof = _read_json(path, {})
    if not prof:
        prof = {
            "port_name":"MegaPort-X","berth_count":6,"qc_count":12,"yc_count":80,
            "agv_count":220,"bess_mwh":20,"shore_power_mva":25,"job_kwh":250.0,
            "guard_threshold_kwh":18.0
        }
        path.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    return prof

def get_policies() -> List[str]:
    path = DATA_DIR / "policies.json"
    arr = _read_json(path, ["S-baseline","S-ruleV1","S-ruleV2","S-rlA","S-rlB"])
    return list(dict.fromkeys(arr))  # 去重保序

def _gen_samples_for_policy(policy: str, n: int, prof: dict) -> List[float]:
    """
    生成 delta kWh（正=更差/更耗电，负=节能）。带厚尾噪声。
    """
    base_map = {
        "S-baseline": +10.0,
        "S-ruleV1":   +5.0,
        "S-ruleV2":   +2.0,
        "S-rlA":      -2.0,
        "S-rlB":      -5.0
    }
    mean = base_map.get(policy, 0.0)
    scale = 1.0 + (prof.get("qc_count", 8) - 8) / 20.0 + (prof.get("bess_mwh", 10) - 10) / 50.0
    mu = mean * scale
    sigma = 6.0 * scale

    out: List[float] = []
    for _ in range(n):
        x = _RNG.gauss(mu, sigma)
        # 厚尾：突发事件（阵风/潮汐/峰价/设备降额）
        if _RNG.random() < 0.06:
            x += _RNG.choice([+20.0, -15.0]) * (1.0 + _RNG.random())
        out.append(float(x))
    return out

def compute_distributions() -> Dict[str, List[float]]:
    prof = get_port_profile()
    policies = get_policies()
    per = max(240, int(20 * prof.get("qc_count", 10)))  # 与岸桥规模相关
    return {pid: _gen_samples_for_policy(pid, per, prof) for pid in policies}

def compute_leaderboard() -> Leaderboard:
    prof = get_port_profile()
    dist = compute_distributions()
    items: List[PolicyItem] = []
    ref = float(prof.get("job_kwh", 250.0))
    guard = float(prof.get("guard_threshold_kwh", 18.0))

    for pid, vals in dist.items():
        mape = mape_from_deltas(vals, ref_kwh=ref)
        c = cvar95(vals)
        viol = sum(1 for v in vals if v > guard)
        ppm = int(viol / max(1, len(vals)) * 1_000_000)
        items.append(PolicyItem(id=pid, mape=mape, cvar95_kwh=c, violations_ppm=ppm, n=len(vals)))

    items.sort(key=lambda x: (x.cvar95_kwh, x.mape))
    return Leaderboard(sample_total=sum(i.n for i in items), items=items)

def _ppm_component(total_fails: int, total: int, frac: float) -> int:
    return int((total_fails * frac) / max(1, total) * 1_000_000)

def compute_safety_summary() -> SafetySummary:
    prof = get_port_profile()
    dist = compute_distributions()
    # 取 CVaR 最优策略代表当前在用
    best_id = min(dist.items(), key=lambda kv: cvar95(kv[1]))[0]
    vals = dist[best_id]
    guard = float(prof.get("guard_threshold_kwh", 18.0))
    total = len(vals)
    fails = sum(1 for v in vals if v > guard)
    pass_rate = 1 - (fails / max(1, total))
    rules = [
        SafetyRule(rule="capacity_limit", ppm=_ppm_component(fails, total, 0.40)),
        SafetyRule(rule="ramp_rate",      ppm=_ppm_component(fails, total, 0.35)),
        SafetyRule(rule="thermal_bound",  ppm=_ppm_component(fails, total, 0.25))
    ]
    return SafetySummary(
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cvar95_kwh=cvar95(vals),
        guard_pass_rate=pass_rate,
        violations_ppm=int(fails / max(1, total) * 1_000_000),
        rules=rules
    )

def compute_actions_hist(bins: int = 10) -> ActionsHist:
    # 动作幅度 ~ Beta 分布（偏保守），可改成读取真实策略输出
    N = 400
    vals = []
    for _ in range(N):
        # 近似 Beta(2.5, 3.5) 的取样（用中心极限定性模拟）
        v = sum(_RNG.random() for _ in range(3)) / 3  # 0~1 左右的偏保守
        vals.append(max(0.0, min(1.0, v)))
    counts = [0] * bins
    for v in vals:
        idx = min(bins - 1, int(v * bins))
        counts[idx] += 1
    hist = [{"bin": f"{i/bins:.1f}~{(i+1)/bins:.1f}", "count": counts[i]} for i in range(bins)]
    return ActionsHist(hist=hist)
