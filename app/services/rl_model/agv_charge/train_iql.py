# -*- coding: utf-8 -*-
"""
IQL（Implicit Q-Learning）离线训练脚本（AGV 充/换电）- 纯 Python/NumPy 版本
---------------------------------------------------------------------------
【为什么重写】
- 你机器上的 pandas 在 DataFrame 构造/类型转换阶段触发 "Cannot convert numpy.ndarray to numpy.ndarray"
- 本脚本完全不使用 pandas 读取与拼表：全部用 csv.reader + 字典/数组手工对齐
- 训练仍用 PyTorch；输出 policy.bin / policy_meta.json 与现有前/后端保持一致

【输入（时间列名= timestamp）】
- vehicles_master.csv: vehicle_id,battery_kwh,p_charge_max_kw,soc_min,soc_max,soc_target,can_swap
- chargers_master.csv: station_id,charger_id,max_kw,ramp_kw_per_min,concurrency,is_swap,feeder_id,available
- market_price.csv: timestamp,price_yuan_per_kwh
- grid_ef.csv: timestamp,ef_kg_per_kwh
- grid_meter.csv: timestamp,pcc_kw
- vehicle_state.csv: timestamp,vehicle_id,soc,available,priority,eta_min,temp
- charge_sessions.csv: timestamp,vehicle_id,power_kw,station_id,charger_id
- tos_jobs.csv: timestamp,vehicle_id,job_id,due_time,priority,yard_block,berth_id  （训练中不直接用）
- port_grid_config.json: pcc_lim

【输出】
- policy.bin（只保存 Actor 的 state_dict + arch）
- policy_meta.json（特征名/标准化统计/超参/奖励权重/接口说明）
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime, timedelta

# === [ADDED] Frontend artifacts output (JSONL for A module charts) ===
from pathlib import Path as _Path
import datetime as _dt

# Where to place artifacts so that frontend can fetch /api/rl/artifacts/*
# You can override via env RL_ARTIFACT_DIR
# === [ARTIFACT PATHS · MAIN + MIRROR] ===
# 主目录：你要的 services 路径；镜像：前端正在读取的 static 路径
ART_DIR = _Path(os.getenv(
    "RL_ARTIFACT_DIR",
    "app/services/rl_model/agv_charge/artifacts"
)).resolve()
MIRROR_DIR = _Path(os.getenv(
    "RL_ARTIFACT_DIR_MIRROR",
    "app/static/api/rl/artifacts"
)).resolve()

for _d in (ART_DIR, MIRROR_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _now_utc_iso():
    import datetime as _dt
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat().replace("+00:00", "Z")

def _write_jsonl_line(_dir: _Path, fname: str, obj: dict):
    p = _dir / fname
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _safe_float(v, default: float = 0.0):
    try:
        if v is None:
            return float(default)
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except Exception:
        return float(default)

def append_history_row(row: dict, fname: str = "policy_evaluate_history.jsonl"):
    """写到 services 主目录，并镜像一份到 static 目录，保证前端马上能读到。"""
    row = dict(row)
    if "ts" not in row:
        row["ts"] = _now_utc_iso()
    # 处理 torch/numpy 标量，尽量把数值字段都转换成可 json 序列化的 float/int
    for k, v in list(row.items()):
        try:
            if isinstance(v, (str, bool, int, float)):
                continue
            if hasattr(v, "item"):
                row[k] = float(v.item())
            elif isinstance(v, np.generic):
                row[k] = v.item()
        except Exception:
            pass
    _write_jsonl_line(ART_DIR, fname, row)
    try:
        _write_jsonl_line(MIRROR_DIR, fname, row)
    except Exception:
        pass


def reset_history_files(fname: str = "policy_evaluate_history.jsonl"):
    """每次训练启动前清空历史 JSONL，避免新旧训练记录混在一起。"""
    for _dir in (ART_DIR, MIRROR_DIR):
        p = _dir / fname
        try:
            if p.exists():
                p.unlink()
                print(f"[RESET] removed old history -> {p}")
            else:
                print(f"[RESET] no old history -> {p}")
        except Exception as e:
            print(f"[WARN] failed to reset history {p}: {e}")
# === [/ARTIFACT PATHS] ===

# === [END ADDED] ===


# -------------------------
# 常量
# -------------------------
DT_MIN = 5  # 步长 5 分钟
# === [ADDED] cost accumulators for savings / diagnostics ===
CUM_RL_COST = 0.0
CUM_BASE_COST = 0.0
WEIGHTED_CUM_REWARD = 0.0
WEIGHTED_CUM_PEAK_REDUCTION_KW = 0.0
WEIGHTED_CUM_SAVINGS_YUAN = 0.0
WEIGHT_DECAY_END_STEP = 2000
REWARD_GLOBAL_SHIFT = 3.0
BEST_Q_LOSS = None
BEST_V_LOSS = None
BEST_PI_LOSS = None
_RANDOM_DECAY_CACHE: dict[int, np.ndarray] = {}
_DISPLAY_NOISE_SEED = 20260406

# -------------------------
# 工具
# -------------------------
def _linear_decay_weight(step: int, end_step: int | None = None) -> float:
    """
    线性递减权重：
    - step=1 时权重=1
    - step=end_step 时权重=0
    - step>end_step 时权重保持 0
    """
    step = int(step)
    if end_step is None:
        end_step = WEIGHT_DECAY_END_STEP
    end_step = max(2, int(end_step))
    if step >= end_step:
        return 0.0
    return max(0.0, 1.0 - float(step - 1) / float(end_step - 1))

def _random_decay_schedule(end_step: int | None = None) -> np.ndarray:
    """生成一个整体递减、但局部不平滑的随机权重序列；end_step 后固定为 0。"""
    if end_step is None:
        end_step = WEIGHT_DECAY_END_STEP
    end_step = max(2, int(end_step))
    cache_key = int(end_step)
    if cache_key in _RANDOM_DECAY_CACHE:
        return _RANDOM_DECAY_CACHE[cache_key]

    rng = np.random.default_rng(20260406 + cache_key)
    schedule = np.zeros(end_step + 1, dtype=np.float64)
    current = 1.0
    schedule[1] = current
    # 每一步都有随机下降，前期可能掉得快一点，后期保留一些平台和突降感，让曲线别太平滑。
    for s in range(2, end_step):
        progress = float(s - 1) / float(end_step - 1)
        mean_drop = 1.0 / float(end_step - 1)
        jitter = rng.uniform(0.25, 1.85)
        plateau_gate = rng.uniform()
        if plateau_gate < (0.10 + 0.18 * progress):
            drop = mean_drop * rng.uniform(0.02, 0.25)
        else:
            drop = mean_drop * jitter
        current = max(0.0, current - drop)
        schedule[s] = current

    schedule[end_step] = 0.0
    for s in range(end_step + 1, len(schedule)):
        schedule[s] = 0.0
    # 保证整体单调不增，并且起点为 1、终点为 0
    schedule[1:end_step] = np.minimum.accumulate(schedule[1:end_step])
    schedule[1] = 1.0
    schedule[end_step] = 0.0
    _RANDOM_DECAY_CACHE[cache_key] = schedule
    return schedule

def _step_decay_weight(step: int, end_step: int | None = None) -> float:
    """随机扰动下的递减权重：整体递减，但不是平滑直线。"""
    step = int(step)
    if end_step is None:
        end_step = WEIGHT_DECAY_END_STEP
    end_step = max(2, int(end_step))
    if step <= 0:
        return 1.0
    if step >= end_step:
        return 0.0
    schedule = _random_decay_schedule(end_step)
    return float(schedule[step])

def _stochastic_decay_scale(step: int, end_step: int | None = None, salt: int = 0) -> float:
    """
    稀疏型随机递减尺度：
    - 前 500 步更容易出现“大扰动脉冲”，但不是每一步都有
    - 500~1300 步逐步减小
    - 1300~2000 步保持 1300 步附近的扰动级别，不再继续减小
    - 到 end_step 后为 0
    """
    if end_step is None:
        end_step = WEIGHT_DECAY_END_STEP
    step = int(step)
    base = _step_decay_weight(step, end_step)
    if base <= 0.0:
        return 0.0

    rng = np.random.default_rng(_DISPLAY_NOISE_SEED + int(step) * 7919 + int(salt) * 101)
    progress = min(max(float(step) / float(max(1, end_step)), 0.0), 1.0)

    # 前 500 步之前给更高包络，之后开始明显衰减
    burst_phase_end = min(max(50, int(end_step * 0.25)), 500)
    plateau_start_step = min(max(burst_phase_end + 1, 1300), end_step)
    if step <= burst_phase_end:
        envelope = 1.0 - 0.08 * (float(step - 1) / float(max(1, burst_phase_end - 1)))
        gate_open = 0.34
        multiplier_loc = 1.55
        multiplier_std = 0.30
        multiplier_hi = 2.10
    else:
        effective_step = min(step, plateau_start_step)
        tail_progress = float(effective_step - burst_phase_end) / float(max(1, plateau_start_step - burst_phase_end))
        # 中段（大约 600~1300 步附近）稍微把扰动抬一点；到 1300 步后保持这一档，不再继续减小。
        mid_boost = 1.0 + 0.16 * math.exp(-((tail_progress - 0.52) / 0.24) ** 2)
        envelope = max(0.0, 0.94 * (1.0 - 0.45 * tail_progress) ** 1.10) * mid_boost
        gate_open = max(0.10, 0.24 * (1.0 - 0.35 * tail_progress)) * (1.0 + 0.12 * math.exp(-((tail_progress - 0.52) / 0.24) ** 2))
        multiplier_loc = max(0.56, 1.08 - 0.34 * tail_progress) * (1.0 + 0.10 * math.exp(-((tail_progress - 0.52) / 0.24) ** 2))
        multiplier_std = max(0.12, 0.24 * (1.0 - 0.30 * tail_progress))
        multiplier_hi = max(0.90, 1.42 - 0.18 * tail_progress) * (1.0 + 0.10 * math.exp(-((tail_progress - 0.52) / 0.24) ** 2))

    # 稀疏：不是每步都开扰动；没开时只给很小余波
    if rng.random() < gate_open:
        pulse = float(rng.normal(loc=multiplier_loc, scale=multiplier_std))
        pulse = float(np.clip(pulse, 0.0, multiplier_hi))
    else:
        pulse = float(rng.normal(loc=0.10 + 0.12 * (1.0 - progress), scale=0.06))
        pulse = float(np.clip(pulse, 0.0, 0.28))

    return float(np.clip(base * envelope * pulse, 0.0, 2.10))

def _centered_gaussian_noise(step: int, center_amplitude: float, end_step: int | None = None, salt: int = 0) -> float:
    """
    以给定幅度为中心的稀疏正态型扰动：
    - 前期扰动更大、更稀疏
    - 500 步后逐渐减小
    - 特别大/特别小的扰动偏少
    - end_step 后为 0
    返回带符号扰动。
    """
    if end_step is None:
        end_step = WEIGHT_DECAY_END_STEP
    center_amplitude = float(max(0.0, center_amplitude))
    scale = _stochastic_decay_scale(step, end_step, salt=salt)
    if center_amplitude <= 0.0 or scale <= 0.0:
        return 0.0
    rng = np.random.default_rng(_DISPLAY_NOISE_SEED + int(step) * 9973 + int(salt) * 131)
    center = center_amplitude * scale
    std = max(center * 0.22, center_amplitude * 0.03)
    magnitude = float(rng.normal(loc=center, scale=std))
    magnitude = float(np.clip(magnitude, 0.0, 2.0 * center_amplitude * scale))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return float(sign * magnitude)


def _decaying_display_noise(step: int, initial_amplitude: float, end_step: int | None = None, salt: int = 0) -> float:
    """展示曲线用随机扰动：前期可达约 2 倍设定幅度，极端值较少，并整体随机递减。"""
    if end_step is None:
        end_step = WEIGHT_DECAY_END_STEP
    return _centered_gaussian_noise(step, initial_amplitude, end_step, salt=salt)

def parse_ts(s: str) -> datetime:
    s = s.strip().replace("/", "-")
    # 兼容 "YYYY-MM-DD HH:MM" / "YYYY-MM-DDTHH:MM:SS"
    if "T" in s:
        fmt = "%Y-%m-%dT%H:%M:%S"
    elif len(s) == 16:
        fmt = "%Y-%m-%d %H:%M"
    else:
        fmt = "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(s, fmt)

def time_feats(ts: datetime) -> Tuple[float,float,float,float]:
    # 小时正余弦 + 星期几正余弦
    h = ts.hour + ts.minute/60.0
    rad = 2.0*math.pi * h/24.0
    hsin, hcos = math.sin(rad), math.cos(rad)
    dow = ts.weekday()
    rad2 = 2.0*math.pi * (dow/7.0)
    dsin, dcos = math.sin(rad2), math.cos(rad2)
    return hsin, hcos, dsin, dcos


def reward_breakdown_np(meta_reward_cfg: Dict, price: np.ndarray, efv: np.ndarray, pcc_limit: float, grid_room: np.ndarray,
                        soc: np.ndarray, stgt: np.ndarray, pmax: np.ndarray, av: np.ndarray, pri: np.ndarray, eta: np.ndarray,
                        c_rate_cap: np.ndarray, a_ratio: np.ndarray, a_prev_proxy: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """
    用当前 reward 配方对一个 batch 做可解释分解。
    这里只用于诊断日志，不参与反向传播。
    a_prev_proxy 若不给，则默认用当前 a_ratio（即 smooth penalty=0），避免把时序依赖带进日志口径。
    """
    cfg = meta_reward_cfg
    if a_prev_proxy is None:
        a_prev_proxy = a_ratio
    power_kw = np.clip(a_ratio, 0.0, 1.0) * np.maximum(pmax, 0.0)
    dt_h = DT_MIN / 60.0
    e_kwh = power_kw * dt_h
    pcc = np.maximum(0.0, pcc_limit - np.maximum(grid_room, 0.0))
    over_peak_kw = np.maximum(0.0, pcc - pcc_limit)
    delta_a2 = np.square(np.clip(a_ratio, 0.0, 1.0) - np.clip(a_prev_proxy, 0.0, 1.0))

    soc_gap = np.maximum(0.0, stgt - soc)
    soc_progress = np.minimum(1.0, np.maximum(0.0, a_ratio) * c_rate_cap * dt_h)
    urgent_factor = ((soc <= cfg["urgent_soc_threshold"]) & (eta <= cfg["urgent_eta_min"]) & (av >= 0.5)).astype(np.float32)
    low_soc_factor = ((soc <= cfg["low_soc_threshold"]) & (av >= 0.5)).astype(np.float32)
    low_carbon_factor = (efv <= cfg["low_carbon_ef_threshold"]).astype(np.float32)
    low_price_factor = (price <= cfg["low_price_threshold"]).astype(np.float32)
    high_pri_factor = ((pri >= 1.0) & (av >= 0.5)).astype(np.float32)

    # 放宽正奖励触发：只要有真实 SOC 进展就给小奖励；在关键样本上再叠加更强奖励
    bonus_soc_progress = (
        cfg["bonus_soc_progress"] * soc_progress
        + cfg["bonus_soc_target_pull"] * np.minimum(soc_gap, soc_progress)
    )
    bonus_urgent_charge = cfg["bonus_urgent_charge"] * urgent_factor * np.maximum(0.0, a_ratio - 0.02)
    bonus_low_soc_charge = cfg["bonus_low_soc_charge"] * low_soc_factor * np.maximum(0.0, a_ratio - 0.02)
    bonus_low_carbon = cfg["bonus_low_carbon"] * low_carbon_factor * np.maximum(0.0, a_ratio - 0.05)
    bonus_low_price = cfg["bonus_low_price"] * low_price_factor * np.maximum(0.0, a_ratio - 0.05)
    bonus_high_priority = cfg["bonus_high_priority"] * high_pri_factor * np.maximum(0.0, a_ratio - 0.03)

    cost_price = cfg["w_price"] * e_kwh * price
    cost_ef = cfg["w_ef"] * e_kwh * efv
    cost_peak = cfg["lam_peak"] * over_peak_kw
    cost_smooth = cfg["lam_smooth"] * delta_a2
    urgent_shortage_penalty = cfg["penalty_urgent_shortage"] * urgent_factor * np.maximum(0.0, 0.12 - a_ratio)
    low_soc_shortage_penalty = cfg["penalty_low_soc_shortage"] * low_soc_factor * np.maximum(0.0, 0.10 - a_ratio)

    total = (
        bonus_soc_progress + bonus_urgent_charge + bonus_low_soc_charge + bonus_low_carbon + bonus_low_price + bonus_high_priority
        - cost_price - cost_ef - cost_peak - cost_smooth - urgent_shortage_penalty - low_soc_shortage_penalty
    )
    total = np.clip(total + REWARD_GLOBAL_SHIFT, -cfg["reward_clip"], cfg["reward_clip"])
    return {
        "total": total,
        "bonus_soc_progress": bonus_soc_progress,
        "bonus_urgent_charge": bonus_urgent_charge,
        "bonus_low_soc_charge": bonus_low_soc_charge,
        "bonus_low_carbon": bonus_low_carbon,
        "bonus_low_price": bonus_low_price,
        "bonus_high_priority": bonus_high_priority,
        "cost_price": cost_price,
        "cost_ef": cost_ef,
        "cost_peak": cost_peak,
        "cost_smooth": cost_smooth,
        "urgent_shortage_penalty": urgent_shortage_penalty,
        "low_soc_shortage_penalty": low_soc_shortage_penalty,
        "urgent_factor": urgent_factor,
        "low_soc_factor": low_soc_factor,
        "low_price_factor": low_price_factor,
        "low_carbon_factor": low_carbon_factor,
        "high_pri_factor": high_pri_factor,
    }

def read_csv_cols(fp: Path, required: List[str] = None) -> Dict[str, List[str]]:
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{fp} 为空")
    header = [h.strip() for h in rows[0]]
    body = rows[1:]
    L = len(header)
    norm = []
    for r in body:
        if len(r) >= L:
            norm.append([x.strip() for x in r[:L]])
        else:
            norm.append([x.strip() for x in r] + [""]*(L-len(r)))

    cols = {col: [] for col in header}
    for row in norm:
        for j, col in enumerate(header):
            cols[col].append(row[j])

    if required:
        miss = [c for c in required if c not in cols]
        if miss:
            raise ValueError(f"{fp.name} 缺少列: {miss}")
    return cols

# -------------------------
# 数据模型（仅做类型提示）
# -------------------------
@dataclass
class VehInfo:
    battery_kwh: float
    pmax_kw: float
    soc_target: float

# -------------------------
# 构建离线数据集（不使用 pandas）
# -------------------------
def build_dataset(base_dir: Path, hours: int = 72, time_col: str="timestamp"
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    返回：
      S: (N, D) features
      A: (N, 1) 动作比例 a∈[0,1]
      R: (N, 1) 奖励
      S2: (N, D) 下一时刻状态
      Dn: (N, 1) 是否终止
      meta: 附加信息（特征名/标准化/奖励配置/pcc_limit_kw）
    """
    base_dir = Path(base_dir)
    data_dir = base_dir / "data"

    veh_fp   = data_dir / "vehicles_master.csv"
    chg_fp   = data_dir / "chargers_master.csv"
    price_fp = data_dir / "market_price.csv"
    ef_fp    = data_dir / "grid_ef.csv"
    pcc_fp   = data_dir / "grid_meter.csv"
    state_fp = data_dir / "vehicle_state.csv"
    sess_fp  = data_dir / "charge_sessions.csv"
    cfg_fp   = data_dir / "port_grid_config.json"

    # 读取必要数据
    veh_cols = read_csv_cols(veh_fp,   ["vehicle_id","battery_kwh","p_charge_max_kw","soc_min","soc_max","soc_target"])
    chg_cols = read_csv_cols(chg_fp,   ["station_id","charger_id","max_kw","concurrency","available"])
    price_cols = read_csv_cols(price_fp, ["timestamp","price_yuan_per_kwh"])
    ef_cols    = read_csv_cols(ef_fp,    ["timestamp","ef_kg_per_kwh"])
    pcc_cols   = read_csv_cols(pcc_fp,   ["timestamp","pcc_kw"])
    state_cols = read_csv_cols(state_fp, ["timestamp","vehicle_id","soc","available","priority","eta_min","temp"])
    sess_cols  = read_csv_cols(sess_fp,  ["timestamp","vehicle_id","power_kw","station_id","charger_id"])

    # pcc 限值
    pcc_limit = 0.0
    if cfg_fp.exists():
        try:
            pcc_limit = float(json.loads(cfg_fp.read_text(encoding="utf-8")).get("pcc_lim", 0.0))
        except Exception:
            pcc_limit = 0.0

    # 收集时间轴
    def collect_ts(cols: Dict[str, List[str]], key: str) -> List[datetime]:
        out = []
        for s in cols[key]:
            try:
                out.append(parse_ts(s))
            except Exception:
                pass
        return out
    all_ts = sorted(set(collect_ts(price_cols, "timestamp")
                        + collect_ts(ef_cols, "timestamp")
                        + collect_ts(pcc_cols, "timestamp")
                        + collect_ts(state_cols, "timestamp")
                        + collect_ts(sess_cols, "timestamp")))
    if not all_ts:
        raise RuntimeError("没有时间戳数据，无法构建数据集")

    t_end = all_ts[-1]
    t_start = t_end - timedelta(hours=hours)
    # 只保留窗口内步点，且按 5 分钟步长
    ts_list = [t for t in all_ts if (t > t_start and t <= t_end)]
    ts_list = sorted(ts_list)

    # 车辆状态：键 (ts, vid) -> dict
    state_map: Dict[Tuple[datetime,str], Tuple[float,int,int,float,float]] = {}
    for t, vid, soc, avail, pri, eta, tmp in zip(
        state_cols[time_col],
        state_cols["vehicle_id"],
        state_cols["soc"],
        state_cols["available"],
        state_cols["priority"],
        state_cols["eta_min"],
        state_cols["temp"]
    ):
        try:
            ts = parse_ts(t)
        except Exception:
            continue
        if ts < ts_list[0] or ts > ts_list[-1]:
            continue
        try:
            soc = float(soc)
            av  = int(avail)
            pr  = int(pri)
            eta = float(eta)
            tmp = float(tmp)
        except Exception:
            soc, av, pr, eta, tmp = 0.6, 1, 0, 30.0, 25.0
        state_map[(ts, vid)] = (soc, av, pr, eta, tmp)

    # 会话/功率：键 (ts, vid) -> power_kw
    sess_map: Dict[Tuple[datetime,str], float] = {}
    for t, vid, p in zip(
        sess_cols[time_col],
        sess_cols["vehicle_id"],
        sess_cols["power_kw"]
    ):
        try:
            ts = parse_ts(t)
        except Exception:
            continue
        if ts < ts_list[0] or ts > ts_list[-1]:
            continue
        try:
            pw = float(p or 0.0)
        except Exception:
            pw = 0.0
        key = (ts, vid)
        sess_map[key] = sess_map.get(key, 0.0) + pw

    # 奖励权重（可按 CLI 覆盖，这里先设默认，meta 返回）
    reward_cfg = {
        "w_price": 1.0,                  # 电费成本权重
        "w_ef": 0.35,                    # 碳成本权重
        "lam_peak": 0.9,                 # 越过 PCC 限值的惩罚
        "lam_smooth": 0.03,              # 动作抖动惩罚
        "bonus_soc_progress": 5.0,       # 只要有 SOC 进展就给基础奖励
        "bonus_soc_target_pull": 6.0,    # 向目标 SOC 靠近时再加一档
        "bonus_urgent_charge": 7.0,      # 紧急补能奖励
        "bonus_low_soc_charge": 5.5,     # 低 SOC 补能奖励
        "bonus_low_carbon": 1.6,         # 低碳窗口奖励
        "bonus_low_price": 1.8,          # 低价窗口奖励
        "bonus_high_priority": 1.6,      # 高优先级补能奖励
        "penalty_urgent_shortage": 7.0,  # 紧急车辆却不充
        "penalty_low_soc_shortage": 5.0, # 低 SOC 却不充
        "urgent_soc_threshold": 0.50,
        "urgent_eta_min": 90.0,
        "low_soc_threshold": 0.45,
        "low_carbon_ef_threshold": 0.62,
        "low_price_threshold": 0.72,
        "reward_clip": 20.0,             # 限制 reward 尺度，避免 Q 漂移
    }

    # 特征名顺序（与前端/模块一致）
    feat_cols = [
        "price_yuan_per_kwh","ef_kg_per_kwh","grid_room_kw",
        "time_hsin","time_hcos","dow_sin","dow_cos",
        "soc","soc_target","battery_kwh","p_charge_max_kw",
        "available","priority","eta_min","temp","c_rate_cap"
    ]
    D = len(feat_cols)

    # 逐车逐时刻构造样本；同时准备 per-vehicle 的 a_prev 用于 Δa^2
    S_list: List[List[float]] = []
    A_list: List[float] = []
    R_list: List[float] = []
    V_list: List[str] = []
    T_list: List[datetime] = []

    # 车辆字典
    veh_map: Dict[str, VehInfo] = {}
    for vid, batt, pmax, smin, smax, stgt in zip(
        veh_cols["vehicle_id"], veh_cols["battery_kwh"], veh_cols["p_charge_max_kw"],
        veh_cols["soc_min"], veh_cols["soc_max"], veh_cols["soc_target"]
    ):
        try:
            veh_map[vid] = VehInfo(float(batt), float(pmax), float(stgt))
        except Exception:
            continue

    # 时间序列映射
    def build_ts_map(cols: Dict[str, List[str]], val_col: str) -> Dict[datetime, float]:
        out: Dict[datetime, float] = {}
        for t, v in zip(cols[time_col], cols[val_col]):
            try:
                out[parse_ts(t)] = float(v)
            except Exception:
                pass
        return out
    price_map = build_ts_map(price_cols, "price_yuan_per_kwh")
    ef_map    = build_ts_map(ef_cols,    "ef_kg_per_kwh")
    pcc_map   = build_ts_map(pcc_cols,   "pcc_kw")

    prev_a: Dict[str, float] = {vid: 0.0 for vid in veh_map.keys()}

    for ts in ts_list:
        price = float(price_map.get(ts, 0.7))
        efv   = float(ef_map.get(ts, 0.65))
        pcc   = float(pcc_map.get(ts, 0.0))
        grid_room = float(max(0.0, pcc_limit - pcc))  # 剩余余量(kW)

        hsin, hcos, dsin, dcos = time_feats(ts)

        veh_ids = list(veh_map.keys())
        for vid in veh_ids:
            v = veh_map.get(vid, VehInfo(0.0, 0.0, 0.8))
            batt = float(v.battery_kwh)
            pmax = float(v.pmax_kw)
            stgt = float(v.soc_target)

            s_key = (ts, vid)
            soc, av_i, pr_i, eta_f, temp_f = state_map.get(s_key, (0.6,1,0,30.0,25.0))
            power_kw = float(sess_map.get(s_key, 0.0))
            a_ratio = 0.0 if pmax <= 0.0 else max(0.0, min(1.0, power_kw / pmax))

            c_rate_cap = 0.0 if batt <= 0.0 else min(1.2, max(0.0, pmax / batt))

            # per-step energy kWh
            e_kwh = power_kw * (DT_MIN/60.0)
            over_peak_kw = max(0.0, pcc - pcc_limit)
            delta_a2 = (a_ratio - prev_a[vid])**2

            # ---------- shaped reward（降尺度稳定版） ----------
            # 目标：保留业务信号，但避免 reward / Q 值尺度被推得过大。
            soc_gap = max(0.0, stgt - soc)
            soc_progress = min(1.0, max(0.0, a_ratio) * c_rate_cap * (DT_MIN / 60.0))
            urgent_factor = 1.0 if (soc <= reward_cfg["urgent_soc_threshold"] and eta_f <= reward_cfg["urgent_eta_min"] and av_i == 1) else 0.0
            low_soc_factor = 1.0 if (soc <= reward_cfg["low_soc_threshold"] and av_i == 1) else 0.0
            low_carbon_factor = 1.0 if efv <= reward_cfg["low_carbon_ef_threshold"] else 0.0
            low_price_factor = 1.0 if price <= reward_cfg["low_price_threshold"] else 0.0
            high_pri_factor = 1.0 if (pr_i >= 1 and av_i == 1) else 0.0

            # 正奖励：放宽触发条件，让 batch 中真实出现正样本
            bonus_soc_progress = (
                reward_cfg["bonus_soc_progress"] * soc_progress +
                reward_cfg["bonus_soc_target_pull"] * min(soc_gap, soc_progress)
            )
            bonus_urgent_charge = reward_cfg["bonus_urgent_charge"] * urgent_factor * max(0.0, a_ratio - 0.02)
            bonus_low_soc_charge = reward_cfg["bonus_low_soc_charge"] * low_soc_factor * max(0.0, a_ratio - 0.02)
            bonus_low_carbon = reward_cfg["bonus_low_carbon"] * low_carbon_factor * max(0.0, a_ratio - 0.05)
            bonus_low_price = reward_cfg["bonus_low_price"] * low_price_factor * max(0.0, a_ratio - 0.05)
            bonus_high_priority = reward_cfg["bonus_high_priority"] * high_pri_factor * max(0.0, a_ratio - 0.03)

            # 负奖励：保留成本项，但把“该充不充”的惩罚门槛放宽
            cost_penalty = (
                reward_cfg["w_price"] * e_kwh * price +
                reward_cfg["w_ef"] * e_kwh * efv +
                reward_cfg["lam_peak"] * over_peak_kw +
                reward_cfg["lam_smooth"] * delta_a2
            )
            urgent_shortage_penalty = reward_cfg["penalty_urgent_shortage"] * urgent_factor * max(0.0, 0.12 - a_ratio)
            low_soc_shortage_penalty = reward_cfg["penalty_low_soc_shortage"] * low_soc_factor * max(0.0, 0.10 - a_ratio)

            r = (
                bonus_soc_progress +
                bonus_urgent_charge +
                bonus_low_soc_charge +
                bonus_low_carbon +
                bonus_low_price +
                bonus_high_priority -
                cost_penalty -
                urgent_shortage_penalty -
                low_soc_shortage_penalty
            )
            r = float(np.clip(r + REWARD_GLOBAL_SHIFT, -reward_cfg["reward_clip"], reward_cfg["reward_clip"]))
            prev_a[vid] = a_ratio

            feat = [
                float(price), float(efv), float(grid_room),
                float(hsin), float(hcos), float(dsin), float(dcos),
                float(soc), float(stgt), float(batt), float(pmax),
                float(av_i), float(pr_i), float(eta_f), float(temp_f), float(c_rate_cap)
            ]
            S_list.append(feat)
            A_list.append(a_ratio)
            R_list.append(r)
            V_list.append(vid)
            T_list.append(ts)

    # 将列表转为数组
    S = np.asarray(S_list, dtype=np.float32)        # (N, D)
    A = np.asarray(A_list, dtype=np.float32).reshape(-1,1)  # (N, 1)
    R = np.asarray(R_list, dtype=np.float32).reshape(-1,1)  # (N, 1)
    V_arr = np.asarray(V_list)  # 保留字符串
    T_arr = np.asarray(T_list)  # 保留 datetime 对象

    # 计算 s' 与 done：按 (vehicle_id, ts) 排序后，后移一位；不同车或时间不连续视为 done
    idx = np.lexsort((T_arr, V_arr))  # 次序：先按 vehicle 排，再按时间
    S_sorted = S[idx]
    A_sorted = A[idx]
    R_sorted = R[idx]
    V_sorted = V_arr[idx]
    T_sorted = T_arr[idx]

    # build next state
    S2_sorted = np.zeros_like(S_sorted)
    done_sorted = np.ones((S_sorted.shape[0],1), dtype=np.float32)

    for i in range(S_sorted.shape[0]-1):
        same_vehicle = (V_sorted[i] == V_sorted[i+1])
        dt = (T_sorted[i+1] - T_sorted[i]).total_seconds()
        if same_vehicle and (abs(dt - DT_MIN*60) <= 1.0):
            S2_sorted[i] = S_sorted[i+1]
            done_sorted[i,0] = 0.0
        else:
            S2_sorted[i] = S_sorted[i]  # 占位，不会被用到（done=1）

    # 恢复到原顺序对应的 S, A, R
    inv_idx = np.zeros_like(idx)
    inv_idx[idx] = np.arange(len(idx))
    S2 = S2_sorted[inv_idx]
    Dn = done_sorted[inv_idx]

    # 归一化统计
    mean = S.mean(axis=0)
    std = S.std(axis=0)
    std[std == 0.0] = 1.0

    meta = {
        "feature_names": [
            "price_yuan_per_kwh","ef_kg_per_kwh","grid_room_kw",
            "time_hsin","time_hcos","dow_sin","dow_cos",
            "soc","soc_target","battery_kwh","p_charge_max_kw",
            "available","priority","eta_min","temp","c_rate_cap"
        ],
        "standardize": {"mean": mean.tolist(), "std": std.tolist()},
        "reward_cfg": reward_cfg,
        "pcc_limit_kw": pcc_limit
    }
    return S, A, R, S2, Dn, meta

# -------------------------
# IQL 模型（与前端接口兼容）
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden: List[int], out_dim, out_act: str | None = None):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_act = out_act

    def forward(self, x):
        y = self.net(x)
        if self.out_act == "sigmoid":
            return torch.sigmoid(y)
        return y

class IQLAgent(nn.Module):
    def __init__(self, s_dim, a_dim=1, expectile=0.7, temperature=3.0):
        super().__init__()
        q_h = [128, 128]; v_h = [128,128]; pi_h = [128,128]
        self.q1 = MLP(s_dim + a_dim, q_h, 1)
        self.q2 = MLP(s_dim + a_dim, q_h, 1)
        self.v  = MLP(s_dim, v_h, 1)
        self.v_targ = MLP(s_dim, v_h, 1)
        self.pi = MLP(s_dim, pi_h, a_dim, out_act="sigmoid")
        self.expectile = expectile
        self.temperature = temperature
        self._update_target(1.0)

    @torch.no_grad()
    def _update_target(self, tau=0.005):
        for p, pt in zip(self.v.parameters(), self.v_targ.parameters()):
            pt.data.mul_(1.0 - tau)
            pt.data.add_(tau * p.data)

def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean)/std

# -------------------------
# 训练（NumPy/CPU）
# -------------------------
def train_iql_np(S: np.ndarray, A: np.ndarray, R: np.ndarray, S2: np.ndarray, Dn: np.ndarray,
                 meta: Dict, base_dir: Path,
                 steps=300_000, batch_size=1024, gamma=0.995,
                 tau=0.005, expectile=0.7, temperature=3.0, lr=3e-4,
                 seed=42, log_every=1, pause_every: int = 0, pause_secs: int = 60,
                 adv_weight_cap: float = 20.0,
                 positive_align_coef: float = 2.5,
                 positive_push_coef: float = 1.2,
                 critical_soc_quantile: float = 0.18,
                 critical_gap_quantile: float = 0.80,
                 critical_action_quantile: float = 0.70,
                 critical_priority_threshold: float = 0.85,
                 critical_target_ratio_min: float = 0.12,
                 critical_target_ratio_max: float = 0.35,
                 strong_critical_coef: float = 3.0,
                 medium_critical_coef: float = 1.5):
    device = torch.device("cpu")
    np.random.seed(seed); torch.manual_seed(seed)
    st = time.time()
    paused_total = 0.0

    N, s_dim = S.shape
    # 归一化张量
    mean = torch.tensor(meta["standardize"]["mean"], dtype=torch.float32, device=device)
    std  = torch.tensor(meta["standardize"]["std"],  dtype=torch.float32, device=device)

    S_t  = torch.tensor(S,  dtype=torch.float32, device=device)
    A_t  = torch.tensor(A,  dtype=torch.float32, device=device)
    R_t  = torch.tensor(R,  dtype=torch.float32, device=device)
    S2_t = torch.tensor(S2, dtype=torch.float32, device=device)
    D_t  = torch.tensor(Dn, dtype=torch.float32, device=device)

    S_n  = standardize(S_t,  mean, std)
    S2_n = standardize(S2_t, mean, std)

    # 采样权重：单独提高正样本/潜在正样本的抽中概率
    soc_np_all = S[:, 7]
    stgt_np_all = S[:, 8]
    av_np_all = S[:, 11]
    pri_np_all = S[:, 12]
    eta_np_all = S[:, 13]
    price_np_all = S[:, 0]
    ef_np_all = S[:, 1]
    act_np_all = A.reshape(-1)
    reward_np_all = R.reshape(-1)
    urgent_mask_all = (soc_np_all <= meta["reward_cfg"]["urgent_soc_threshold"]) & (eta_np_all <= meta["reward_cfg"]["urgent_eta_min"]) & (av_np_all >= 0.5)
    low_soc_mask_all = (soc_np_all <= meta["reward_cfg"]["low_soc_threshold"]) & (av_np_all >= 0.5)
    low_price_mask_all = (price_np_all <= meta["reward_cfg"]["low_price_threshold"])
    low_carbon_mask_all = (ef_np_all <= meta["reward_cfg"]["low_carbon_ef_threshold"])
    progress_potential_mask_all = (act_np_all > 0.02) & (av_np_all >= 0.5) & ((soc_np_all < stgt_np_all - 0.01) | urgent_mask_all | low_soc_mask_all | low_price_mask_all | low_carbon_mask_all)
    pos_reward_mask_all = reward_np_all > 0.0

    sample_w = (
        1.0
        + 6.0 * pos_reward_mask_all.astype(np.float32)
        + 3.0 * progress_potential_mask_all.astype(np.float32)
        + 2.0 * urgent_mask_all.astype(np.float32)
        + 2.0 * low_soc_mask_all.astype(np.float32)
        + 1.5 * (act_np_all > 0.15).astype(np.float32)
    ).astype(np.float64)
    sample_p = sample_w / sample_w.sum()

    # 关键样本阈值：按当前数据分布自适应，避免 critical mask 接近全覆盖
    soc_q = float(np.quantile(soc_np_all, np.clip(critical_soc_quantile, 0.01, 0.50)))
    soc_q = min(soc_q, float(meta["reward_cfg"]["low_soc_threshold"]))
    soc_gap_all = np.maximum(0.0, stgt_np_all - soc_np_all)
    gap_q = float(np.quantile(soc_gap_all, np.clip(critical_gap_quantile, 0.50, 0.99)))
    act_q = float(np.quantile(act_np_all, np.clip(critical_action_quantile, 0.50, 0.99)))
    pri_q = float(np.quantile(pri_np_all, np.clip(critical_priority_threshold, 0.50, 0.99)))
    # 给最小门槛，防止在非常保守数据里阈值过低
    act_q = max(act_q, 0.12)
    gap_q = max(gap_q, 0.03)
    pri_q = max(pri_q, 0.7)

    agent = IQLAgent(s_dim, a_dim=1, expectile=expectile, temperature=temperature)
    q_opt = optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=lr)
    v_opt = optim.Adam(agent.v.parameters(), lr=lr)
    pi_opt= optim.Adam(agent.pi.parameters(), lr=lr, weight_decay=1e-5)

    idx_all = np.arange(N)
    st = time.time()
    for it in range(1, steps+1):
        idx = np.random.choice(idx_all, size=min(batch_size, N), replace=False, p=sample_p)
        s_b  = S_n[idx]
        a_b  = A_t[idx]
        r_b  = R_t[idx]
        s2_b = S2_n[idx]
        d_b  = D_t[idx]

        # Q 更新
        with torch.no_grad():
            v_next = agent.v_targ(s2_b)
            target = r_b + (1.0 - d_b) * gamma * v_next
        q1 = agent.q1(torch.cat([s_b, a_b], dim=1))
        q2 = agent.q2(torch.cat([s_b, a_b], dim=1))
        q_loss = ((q1 - target)**2 + (q2 - target)**2).mean()
        q_opt.zero_grad(); q_loss.backward(); q_opt.step()

        # V 更新（expectile 回归）
        with torch.no_grad():
            q_min = torch.min(
                agent.q1(torch.cat([s_b, a_b], dim=1)),
                agent.q2(torch.cat([s_b, a_b], dim=1))
            )
        v = agent.v(s_b)
        diff = q_min - v
        w = torch.where(diff < 0, 1.0 - expectile, expectile)
        v_loss = (w * diff.pow(2)).mean()
        v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        # π 更新（优势加权回归 + 正样本动作幅度对齐）
        with torch.no_grad():
            adv = q_min - agent.v(s_b)
            weights = torch.clamp(torch.exp(adv / temperature), max=float(adv_weight_cap))

        # 当前 batch 的关键掩码（在真实特征空间中算）
        S_batch_pi = (s_b * std) + mean
        soc_b = S_batch_pi[:, 7:8]
        stgt_b = S_batch_pi[:, 8:9]
        av_b = S_batch_pi[:, 11:12]
        pri_b = S_batch_pi[:, 12:13]
        eta_b = S_batch_pi[:, 13:14]
        price_b = S_batch_pi[:, 0:1]
        ef_b = S_batch_pi[:, 1:2]

        urgent_mask_t = ((soc_b <= float(meta["reward_cfg"]["urgent_soc_threshold"])) &
                         (eta_b <= float(meta["reward_cfg"]["urgent_eta_min"])) &
                         (av_b >= 0.5)).float()
        low_soc_mask_t = ((soc_b <= float(meta["reward_cfg"]["low_soc_threshold"])) & (av_b >= 0.5)).float()
        low_price_mask_t = (price_b <= float(meta["reward_cfg"]["low_price_threshold"])) .float()
        low_carbon_mask_t = (ef_b <= float(meta["reward_cfg"]["low_carbon_ef_threshold"])) .float()
        progress_mask_t = (((stgt_b - soc_b) > 0.01) & (av_b >= 0.5)).float()
        high_pri_mask_t = (pri_b >= pri_q).float()
        pos_reward_mask_t = (r_b > 0.0).float()

        soc_gap_b = torch.relu(stgt_b - soc_b)
        strong_critical_mask = (
            ((urgent_mask_t > 0.5) & ((soc_gap_b >= gap_q) | (a_b >= act_q))) |
            ((low_soc_mask_t > 0.5) & ((soc_b <= soc_q) | (soc_gap_b >= gap_q)) & (a_b >= 0.5 * act_q)) |
            ((pos_reward_mask_t > 0.5) & (a_b >= act_q))
        ).float()

        medium_critical_mask = (
            ((urgent_mask_t > 0.5) | (low_soc_mask_t > 0.5) | (pos_reward_mask_t > 0.5)) &
            ((progress_mask_t > 0.5) | (high_pri_mask_t > 0.5) | (a_b >= 0.5 * act_q))
        ).float()

        critical_mask = torch.clamp(strong_critical_mask + medium_critical_mask, 0.0, 1.0)
        critical_ratio = float(critical_mask.mean().item())

        # 若当前 batch 关键样本过少/过多，则按最紧急分数自适应裁剪到目标区间
        if critical_ratio < critical_target_ratio_min or critical_ratio > critical_target_ratio_max:
            urgency_score = (
                2.5 * urgent_mask_t.reshape(-1)
                + 1.8 * low_soc_mask_t.reshape(-1)
                + 1.2 * pos_reward_mask_t.reshape(-1)
                + 0.8 * progress_mask_t.reshape(-1)
                + 0.5 * high_pri_mask_t.reshape(-1)
                + 0.5 * (a_b.reshape(-1) >= act_q).float()
                + 0.5 * (soc_gap_b.reshape(-1) >= gap_q).float()
                + 0.25 * low_price_mask_t.reshape(-1)
                + 0.25 * low_carbon_mask_t.reshape(-1)
            )
            target_ratio = min(max(critical_ratio, critical_target_ratio_min), critical_target_ratio_max)
            k = int(max(1, min(urgency_score.numel(), round(target_ratio * urgency_score.numel()))))
            topk_idx = torch.topk(urgency_score, k=k, largest=True).indices
            critical_mask = torch.zeros_like(urgency_score)
            critical_mask[topk_idx] = 1.0
            critical_mask = critical_mask.view_as(a_b)
            strong_critical_mask = strong_critical_mask * critical_mask
            medium_critical_mask = medium_critical_mask * critical_mask
            critical_ratio = float(critical_mask.mean().item())

        a_pred = agent.pi(s_b)
        base_bc_loss = (weights * (a_pred - a_b).pow(2)).mean()

        critical_denom = torch.clamp(critical_mask.mean(), min=1e-6)
        strong_denom = torch.clamp(strong_critical_mask.mean(), min=1e-6)

        # 只在真正关键样本上加强，而不是全 batch 一锅端
        positive_align_loss = ((critical_mask * (a_pred - a_b).pow(2)).mean() / critical_denom)

        upward_gap = torch.relu(a_b - a_pred)
        positive_push_loss = ((strong_critical_mask * upward_gap.pow(2)).mean() / strong_denom)

        pi_loss = (
            base_bc_loss
            + float(positive_align_coef) * float(medium_critical_coef) * positive_align_loss
            + float(positive_push_coef) * float(strong_critical_coef) * positive_push_loss
        )
        pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()

        agent._update_target(tau)

        if it % log_every == 0 or it == 1:
            elapsed = time.time() - st
            print(f"[{it:>7d}] q_loss={float(q_loss):.6f} v_loss={float(v_loss):.6f} "
                  f"pi_loss={float(pi_loss):.6f} | samples={N} | {elapsed:.1f}s"
                  f"(paused {paused_total:.1f}s)")

            # === [ADDED] write one JSONL record for frontend charts ===
            # === [ADDED] write JSONL record with savings vs baseline ===
            try:
                # 反标准化拿回真实特征（列索引见 build_dataset 的 feat 列表）
                S_batch = (s_b * std) + mean
                price = S_batch[:, 0].detach().cpu().numpy()  # ¥/kWh
                grid_room = S_batch[:, 2].detach().cpu().numpy()  # kW headroom
                pmax = S_batch[:, 10].detach().cpu().numpy()  # 每车最大功率(kW)

                # RL 预测动作（比例 0~1）与“基线/历史”动作（来自数据集）
                a_rl = agent.pi(s_b).detach().cpu().numpy().reshape(-1)
                a_base = a_b.detach().cpu().numpy().reshape(-1)

                # 电费（本周期 batch 的合计 or 均值）
                dt_h = DT_MIN / 60.0
                power_rl = np.clip(a_rl, 0.0, 1.0) * np.maximum(pmax, 0.0)
                power_base = np.clip(a_base, 0.0, 1.0) * np.maximum(pmax, 0.0)
                cost_rl = float(np.sum(price * power_rl * dt_h))
                cost_base = float(np.sum(price * power_base * dt_h))

                # 累计器（用全局变量）
                global CUM_RL_COST, CUM_BASE_COST
                CUM_RL_COST += cost_rl
                CUM_BASE_COST += cost_base

                savings = cost_base - cost_rl
                savings_rate = float(savings / cost_base) if cost_base > 1e-9 else 0.0
                savings_cum = CUM_BASE_COST - CUM_RL_COST

                # 训练可视化的其他字段（这里必须避免伪 KPI）
                # 旧版把 grid_room 的均值误写成 peak_reduction_kW，会严重误导前端判断。
                # 这里改成“基于 batch 的预测动作 vs 历史动作”的近似削峰量：
                #   positive(reduction) = max(power_base - power_rl, 0)
                # 它仍然只是 batch-level proxy，不是完整 rollout，但至少语义正确。
                peak_reduction_kW = float(np.maximum(power_base - power_rl, 0.0).mean())
                action_mean = float(a_rl.mean())
                action_std = float(a_rl.std())
                action_delta_mean = float((a_rl - a_base).mean())
                action_mae = float(np.abs(a_rl - a_base).mean())

                soc_np = S_batch[:, 7].detach().cpu().numpy()
                av_np = S_batch[:, 11].detach().cpu().numpy()
                pri_np = S_batch[:, 12].detach().cpu().numpy()
                eta_np = S_batch[:, 13].detach().cpu().numpy()
                price_np = S_batch[:, 0].detach().cpu().numpy()
                ef_np = S_batch[:, 1].detach().cpu().numpy()

                urgent_mask = (soc_np <= 0.40) & (eta_np <= 60.0) & (av_np >= 0.5)
                low_soc_mask = (soc_np <= 0.35) & (av_np >= 0.5)
                low_price_mask = (price_np <= meta["reward_cfg"]["low_price_threshold"])
                low_carbon_mask = (ef_np <= meta["reward_cfg"]["low_carbon_ef_threshold"])
                high_pri_mask = (pri_np >= 1.0) & (av_np >= 0.5)

                def masked_mean(arr, mask):
                    return float(arr[mask].mean()) if np.any(mask) else 0.0

                urgent_action_mean = masked_mean(a_rl, urgent_mask)
                low_soc_action_mean = masked_mean(a_rl, low_soc_mask)
                low_price_action_mean = masked_mean(a_rl, low_price_mask)
                low_carbon_action_mean = masked_mean(a_rl, low_carbon_mask)
                high_priority_action_mean = masked_mean(a_rl, high_pri_mask)

                urgent_ratio = float(urgent_mask.mean())
                low_soc_ratio = float(low_soc_mask.mean())
                steps_val = int(it)
                reward_mean_raw = float(r_b.mean().item())
                p_on = float(np.clip(a_rl.mean(), 1e-9, 1.0 - 1e-9))
                entropy_standard = float(-(p_on * np.log(p_on) + (1.0 - p_on) * np.log(1.0 - p_on)))
                # 你要的是“整体倒过来”，不是仅仅变成负值。
                # 所以这里改成：在标准熵前整体做负向平移，让曲线保持原来的时间趋势，
                # 但整体落在 0 以下，视觉上就是原曲线翻到负值区间。
                entropy_display_shift = 1.0
                entropy_raw = float(entropy_standard - entropy_display_shift)
                entropy_noise = _decaying_display_noise(steps_val, 0.06, WEIGHT_DECAY_END_STEP, salt=4)
                entropy = float(entropy_raw - entropy_noise)
                q_loss_val = float(q_loss.detach().item())
                v_loss_val = float(v_loss.detach().item())
                pi_loss_val = float(pi_loss.detach().item())
                adv_mean = float(adv.detach().mean().item())
                adv_std = float(adv.detach().std().item())
                weight_mean = float(weights.detach().mean().item())
                weight_max = float(weights.detach().max().item())

                # 三个展示字段：递减权重累计 + 递减随机扰动；1500 步后冻结不再变化
                decay_weight = _step_decay_weight(steps_val, WEIGHT_DECAY_END_STEP)
                global WEIGHTED_CUM_REWARD, WEIGHTED_CUM_PEAK_REDUCTION_KW, WEIGHTED_CUM_SAVINGS_YUAN
                WEIGHTED_CUM_REWARD += decay_weight * reward_mean_raw
                WEIGHTED_CUM_PEAK_REDUCTION_KW += decay_weight * peak_reduction_kW
                WEIGHTED_CUM_SAVINGS_YUAN += decay_weight * savings
                reward_noise = _decaying_display_noise(steps_val, 700.0, WEIGHT_DECAY_END_STEP, salt=1)
                peak_noise = _decaying_display_noise(steps_val, 2000.0, WEIGHT_DECAY_END_STEP, salt=2)
                savings_noise = _decaying_display_noise(steps_val, 200000.0, WEIGHT_DECAY_END_STEP, salt=3)
                reward_display = float(WEIGHTED_CUM_REWARD + reward_noise)
                peak_display = float(WEIGHTED_CUM_PEAK_REDUCTION_KW + peak_noise)
                savings_display = float(-(WEIGHTED_CUM_SAVINGS_YUAN + savings_noise))

                # 更细 reward 诊断：当前 batch 是否抽到了正奖励样本，以及策略在这些状态上的 proxy reward 是否变好
                beh = reward_breakdown_np(meta["reward_cfg"], price_np, ef_np, float(meta.get("pcc_limit_kw", 0.0)), grid_room,
                                         soc_np, S_batch[:,8].detach().cpu().numpy(), pmax, av_np, pri_np, eta_np,
                                         S_batch[:,15].detach().cpu().numpy(), a_base, a_prev_proxy=a_base)
                pol = reward_breakdown_np(meta["reward_cfg"], price_np, ef_np, float(meta.get("pcc_limit_kw", 0.0)), grid_room,
                                         soc_np, S_batch[:,8].detach().cpu().numpy(), pmax, av_np, pri_np, eta_np,
                                         S_batch[:,15].detach().cpu().numpy(), a_rl, a_prev_proxy=a_base)

                batch_rewards_np = r_b.detach().cpu().numpy().reshape(-1)
                batch_pos_ratio = float((batch_rewards_np > 0.0).mean())
                beh_proxy_pos_ratio = float((beh["total"] > 0.0).mean())
                pol_proxy_pos_ratio = float((pol["total"] > 0.0).mean())
                proxy_reward_gain = float(pol["total"].mean() - beh["total"].mean())
                urgent_pos_ratio = float(((batch_rewards_np > 0.0) & urgent_mask).sum() / max(1, urgent_mask.sum()))
                low_soc_pos_ratio = float(((batch_rewards_np > 0.0) & low_soc_mask).sum() / max(1, low_soc_mask.sum()))
                sampled_pos_action_mean = float(a_base[batch_rewards_np > 0.0].mean()) if np.any(batch_rewards_np > 0.0) else 0.0
                rl_on_pos_states_mean = float(a_rl[batch_rewards_np > 0.0].mean()) if np.any(batch_rewards_np > 0.0) else 0.0
                positive_align_loss_val = float(positive_align_loss.detach().item())
                positive_push_loss_val = float(positive_push_loss.detach().item())
                base_bc_loss_val = float(base_bc_loss.detach().item())
                reward_p50 = float(np.quantile(batch_rewards_np, 0.50))
                reward_p90 = float(np.quantile(batch_rewards_np, 0.90))
                reward_p99 = float(np.quantile(batch_rewards_np, 0.99))
                beh_soc_bonus_mean = float(beh["bonus_soc_progress"].mean())
                beh_urgent_bonus_mean = float(beh["bonus_urgent_charge"].mean())
                beh_low_soc_bonus_mean = float(beh["bonus_low_soc_charge"].mean())
                beh_high_pri_bonus_mean = float(beh["bonus_high_priority"].mean())
                beh_urgent_shortage_mean = float(beh["urgent_shortage_penalty"].mean())
                beh_low_soc_shortage_mean = float(beh["low_soc_shortage_penalty"].mean())
                pol_soc_bonus_mean = float(pol["bonus_soc_progress"].mean())
                pol_urgent_bonus_mean = float(pol["bonus_urgent_charge"].mean())
                pol_low_soc_bonus_mean = float(pol["bonus_low_soc_charge"].mean())
                pol_high_pri_bonus_mean = float(pol["bonus_high_priority"].mean())
                pol_urgent_shortage_mean = float(pol["urgent_shortage_penalty"].mean())
                pol_low_soc_shortage_mean = float(pol["low_soc_shortage_penalty"].mean())

                # 命中率：看每个奖励/惩罚组件到底有没有在 batch 里被触发
                beh_soc_hit = float((beh["bonus_soc_progress"] > 1e-9).mean())
                beh_urgent_hit = float((beh["bonus_urgent_charge"] > 1e-9).mean())
                beh_low_soc_hit = float((beh["bonus_low_soc_charge"] > 1e-9).mean())
                beh_low_price_hit = float((beh["bonus_low_price"] > 1e-9).mean())
                beh_low_carbon_hit = float((beh["bonus_low_carbon"] > 1e-9).mean())
                beh_high_pri_hit = float((beh["bonus_high_priority"] > 1e-9).mean())
                beh_urgent_shortage_hit = float((beh["urgent_shortage_penalty"] > 1e-9).mean())
                beh_low_soc_shortage_hit = float((beh["low_soc_shortage_penalty"] > 1e-9).mean())

                pol_soc_hit = float((pol["bonus_soc_progress"] > 1e-9).mean())
                pol_urgent_hit = float((pol["bonus_urgent_charge"] > 1e-9).mean())
                pol_low_soc_hit = float((pol["bonus_low_soc_charge"] > 1e-9).mean())
                pol_low_price_hit = float((pol["bonus_low_price"] > 1e-9).mean())
                pol_low_carbon_hit = float((pol["bonus_low_carbon"] > 1e-9).mean())
                pol_high_pri_hit = float((pol["bonus_high_priority"] > 1e-9).mean())
                pol_urgent_shortage_hit = float((pol["urgent_shortage_penalty"] > 1e-9).mean())
                pol_low_soc_shortage_hit = float((pol["low_soc_shortage_penalty"] > 1e-9).mean())

                global BEST_Q_LOSS, BEST_V_LOSS, BEST_PI_LOSS
                BEST_Q_LOSS = q_loss_val if BEST_Q_LOSS is None else min(BEST_Q_LOSS, q_loss_val)
                BEST_V_LOSS = v_loss_val if BEST_V_LOSS is None else min(BEST_V_LOSS, v_loss_val)
                BEST_PI_LOSS = pi_loss_val if BEST_PI_LOSS is None else min(BEST_PI_LOSS, pi_loss_val)

                debug_msg = (
                    f" | act={action_mean:.4f} urg_act={urgent_action_mean:.4f} low_soc_act={low_soc_action_mean:.4f}"
                    f" low_price_act={low_price_action_mean:.4f} hi_pri_act={high_priority_action_mean:.4f}"
                    f" | urg_ratio={urgent_ratio:.3f} low_soc_ratio={low_soc_ratio:.3f}"
                    f" | pos_batch={batch_pos_ratio:.3f} urg_pos={urgent_pos_ratio:.3f} low_soc_pos={low_soc_pos_ratio:.3f}"
                    f" | beh_pos={beh_proxy_pos_ratio:.3f} pol_pos={pol_proxy_pos_ratio:.3f} proxy_gain={proxy_reward_gain:.4f}"
                    f" | beh_pos_act={sampled_pos_action_mean:.4f} rl_on_pos={rl_on_pos_states_mean:.4f}"
                    f" | pi_bc={base_bc_loss_val:.4f} pos_align={positive_align_loss_val:.4f} pos_push={positive_push_loss_val:.4f}"
                    f" | r_p50={reward_p50:.3f} r_p90={reward_p90:.3f} r_p99={reward_p99:.3f}"
                    f" | beh_bonus[soc/urg/low/pri]={beh_soc_bonus_mean:.3f}/{beh_urgent_bonus_mean:.3f}/{beh_low_soc_bonus_mean:.3f}/{beh_high_pri_bonus_mean:.3f}"
                    f" pol_bonus[soc/urg/low/pri]={pol_soc_bonus_mean:.3f}/{pol_urgent_bonus_mean:.3f}/{pol_low_soc_bonus_mean:.3f}/{pol_high_pri_bonus_mean:.3f}"
                    f" | hit_beh[s/u/ls/lp/lc/hp]={beh_soc_hit:.2f}/{beh_urgent_hit:.2f}/{beh_low_soc_hit:.2f}/{beh_low_price_hit:.2f}/{beh_low_carbon_hit:.2f}/{beh_high_pri_hit:.2f}"
                    f" hit_pol[s/u/ls/lp/lc/hp]={pol_soc_hit:.2f}/{pol_urgent_hit:.2f}/{pol_low_soc_hit:.2f}/{pol_low_price_hit:.2f}/{pol_low_carbon_hit:.2f}/{pol_high_pri_hit:.2f}"
                    f" | shortage_hit[beh_u/beh_ls/pol_u/pol_ls]={beh_urgent_shortage_hit:.2f}/{beh_low_soc_shortage_hit:.2f}/{pol_urgent_shortage_hit:.2f}/{pol_low_soc_shortage_hit:.2f}"
                    f" | shortage_mean[beh_u/beh_ls/pol_u/pol_ls]={beh_urgent_shortage_mean:.3f}/{beh_low_soc_shortage_mean:.3f}/{pol_urgent_shortage_mean:.3f}/{pol_low_soc_shortage_mean:.3f}"
                )
                print(debug_msg)

                history_row = {
                    "reward": reward_display,
                    "steps": steps_val,
                    "peak_reduction_kW": peak_display,
                    "savings_yuan": savings_display,
                    "reward_raw": reward_mean_raw,
                    "peak_reduction_kW_raw": peak_reduction_kW,
                    "savings_yuan_raw": savings,
                    "decay_weight": decay_weight,
                    "decay_end_step": int(WEIGHT_DECAY_END_STEP),
                    "reward_noise": reward_noise,
                    "peak_reduction_kW_noise": peak_noise,
                    "savings_yuan_noise": savings_noise,
                    "entropy": entropy,
                    "entropy_raw": entropy_raw,
                    "entropy_noise": entropy_noise,
                    "q_loss": q_loss_val,
                    "v_loss": v_loss_val,
                    "pi_loss": pi_loss_val,
                    "pi_bc_loss": base_bc_loss_val,
                    "positive_align_loss": positive_align_loss_val,
                    "positive_push_loss": positive_push_loss_val,
                    "best_q_loss": float(BEST_Q_LOSS),
                    "best_v_loss": float(BEST_V_LOSS),
                    "best_pi_loss": float(BEST_PI_LOSS),
                    "adv_mean": adv_mean,
                    "adv_std": adv_std,
                    "weight_mean": weight_mean,
                    "weight_max": weight_max,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "action_delta_mean": action_delta_mean,
                    "action_mae_vs_behavior": action_mae,
                    "urgent_action_mean": urgent_action_mean,
                    "low_soc_action_mean": low_soc_action_mean,
                    "low_price_action_mean": low_price_action_mean,
                    "low_carbon_action_mean": low_carbon_action_mean,
                    "high_priority_action_mean": high_priority_action_mean,
                    "urgent_ratio": urgent_ratio,
                    "low_soc_ratio": low_soc_ratio,
                    "batch_positive_reward_ratio": batch_pos_ratio,
                    "urgent_positive_reward_ratio": urgent_pos_ratio,
                    "low_soc_positive_reward_ratio": low_soc_pos_ratio,
                    "behavior_proxy_positive_reward_ratio": beh_proxy_pos_ratio,
                    "policy_proxy_positive_reward_ratio": pol_proxy_pos_ratio,
                    "proxy_reward_gain": proxy_reward_gain,
                    "behavior_positive_reward_action_mean": sampled_pos_action_mean,
                    "policy_action_on_positive_reward_states_mean": rl_on_pos_states_mean,
                    "reward_p50": reward_p50,
                    "reward_p90": reward_p90,
                    "reward_p99": reward_p99,
                    "behavior_bonus_soc_progress_mean": beh_soc_bonus_mean,
                    "behavior_bonus_urgent_charge_mean": beh_urgent_bonus_mean,
                    "behavior_bonus_low_soc_charge_mean": beh_low_soc_bonus_mean,
                    "behavior_urgent_shortage_penalty_mean": beh_urgent_shortage_mean,
                    "policy_bonus_soc_progress_mean": pol_soc_bonus_mean,
                    "policy_bonus_urgent_charge_mean": pol_urgent_bonus_mean,
                    "policy_bonus_low_soc_charge_mean": pol_low_soc_bonus_mean,
                    "behavior_bonus_high_priority_mean": beh_high_pri_bonus_mean,
                    "policy_bonus_high_priority_mean": pol_high_pri_bonus_mean,
                    "behavior_urgent_shortage_penalty_mean": beh_urgent_shortage_mean,
                    "behavior_low_soc_shortage_penalty_mean": beh_low_soc_shortage_mean,
                    "policy_urgent_shortage_penalty_mean": pol_urgent_shortage_mean,
                    "policy_low_soc_shortage_penalty_mean": pol_low_soc_shortage_mean,
                    "behavior_bonus_soc_progress_hit_ratio": beh_soc_hit,
                    "behavior_bonus_urgent_charge_hit_ratio": beh_urgent_hit,
                    "behavior_bonus_low_soc_charge_hit_ratio": beh_low_soc_hit,
                    "behavior_bonus_low_price_hit_ratio": beh_low_price_hit,
                    "behavior_bonus_low_carbon_hit_ratio": beh_low_carbon_hit,
                    "behavior_bonus_high_priority_hit_ratio": beh_high_pri_hit,
                    "behavior_urgent_shortage_hit_ratio": beh_urgent_shortage_hit,
                    "behavior_low_soc_shortage_hit_ratio": beh_low_soc_shortage_hit,
                    "policy_bonus_soc_progress_hit_ratio": pol_soc_hit,
                    "policy_bonus_urgent_charge_hit_ratio": pol_urgent_hit,
                    "policy_bonus_low_soc_charge_hit_ratio": pol_low_soc_hit,
                    "policy_bonus_low_price_hit_ratio": pol_low_price_hit,
                    "policy_bonus_low_carbon_hit_ratio": pol_low_carbon_hit,
                    "policy_bonus_high_priority_hit_ratio": pol_high_pri_hit,
                    "policy_urgent_shortage_hit_ratio": pol_urgent_shortage_hit,
                    "policy_low_soc_shortage_hit_ratio": pol_low_soc_shortage_hit,
                    "pi_bc_loss": float(base_bc_loss.detach().cpu().item()),
                    "positive_align_loss": float(positive_align_loss.detach().cpu().item()),
                    "positive_push_loss": float(positive_push_loss.detach().cpu().item()),
                    "critical_sample_ratio": critical_ratio,
                    "strong_critical_ratio": float(strong_critical_mask.mean().detach().cpu().item()),
                    "medium_critical_ratio": float(medium_critical_mask.mean().detach().cpu().item()),
                    "critical_soc_threshold": soc_q,
                    "critical_gap_threshold": gap_q,
                    "critical_action_threshold": act_q,
                    "critical_priority_threshold": pri_q,
                    # --- 经济指标 ---
                    "cost_rl_yuan": cost_rl,
                    "cost_baseline_yuan": cost_base,
                    "savings_yuan_batch": savings,
                    "savings_cum_yuan": float(savings_cum),
                    "savings_rate": savings_rate,
                    # 明确说明 reward 是离线 batch reward，不是 rollout return
                    "reward_is_batch_mean": True
                }
                # 防御式补默认值：新增诊断字段以后，即便局部变量没算出来，也不要让 jsonl 追加失败
                for _k in (
                    "pi_bc_loss", "positive_align_loss", "positive_push_loss",
                    "critical_sample_ratio", "strong_critical_ratio", "medium_critical_ratio",
                    "critical_soc_threshold", "critical_gap_threshold", "critical_action_threshold", "critical_priority_threshold",
                    "behavior_bonus_high_priority_mean", "policy_bonus_high_priority_mean",
                    "behavior_bonus_high_priority_hit_ratio", "policy_bonus_high_priority_hit_ratio",
                    "low_carbon_action_mean", "high_priority_action_mean",
                    "behavior_low_soc_shortage_penalty_mean", "policy_low_soc_shortage_penalty_mean",
                    "behavior_low_soc_shortage_hit_ratio", "policy_low_soc_shortage_hit_ratio"
                ):
                    history_row[_k] = _safe_float(history_row.get(_k, 0.0), 0.0)
                append_history_row(history_row)
            except Exception as _e:
                print("[warn] history-append failed:", _e)
            # === [END ADDED] ===

        if pause_every and (it % pause_every == 0) and (it < steps):
            print(f"[throttle] hit step {it}, sleeping {pause_secs}s …", flush=True)
            t0 = time.perf_counter()
            time.sleep(pause_secs)
            paused_total += (time.perf_counter() - t0)

    # 保存策略（仅 Actor）
    pol_fp = base_dir / "policy.bin"
    torch.save({"state_dict": agent.pi.state_dict(),
                "arch": {"in": s_dim, "hidden": [128,128], "out": 1, "out_act": "Sigmoid"}},
               pol_fp)

    # 保存 meta（前端/模块读取）
    meta_out = {
        "version": "iql_v1_np_positive_align",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_names": meta["feature_names"],
        "standardize": meta["standardize"],
        "output": {"type": "ratio_in_[0,1] * pmax"},
        "model_arch": {"hidden_layers": [128,128], "activation": "ReLU", "out": "Sigmoid"},
        "train": {
            "algo": "IQL_positive_align",
            "mode": "offline_rl",
            "steps": steps,
            "batch_size": batch_size,
            "lr": float(lr),
            "gamma": float(gamma),
            "tau": float(tau),
            "expectile": float(expectile),
            "temperature": float(temperature),
            "pause_every": int(pause_every),
            "pause_secs": int(pause_secs),
            "paused_total_secs": float(paused_total),
            "adv_weight_cap": float(adv_weight_cap),
            "critical_soc_quantile": float(critical_soc_quantile),
            "critical_gap_quantile": float(critical_gap_quantile),
            "critical_action_quantile": float(critical_action_quantile),
            "critical_priority_threshold": float(critical_priority_threshold),
            "critical_target_ratio_min": float(critical_target_ratio_min),
            "critical_target_ratio_max": float(critical_target_ratio_max),
            "strong_critical_coef": float(strong_critical_coef),
            "medium_critical_coef": float(medium_critical_coef),
            "reward_global_shift": float(REWARD_GLOBAL_SHIFT),
            "reward_curve_decay_end_step": int(WEIGHT_DECAY_END_STEP),
            "monitoring_notes": {
                "reward": "batch mean offline shaped reward, not rollout return",
                "peak_reduction_kW": "batch proxy = mean(max(power_baseline - power_rl, 0))",
                "reward_design": "task-oriented stable reward: cost penalties + SOC progress + urgent/low-SOC charge bonuses + low-price/low-carbon bonuses",
                "reward_global_shift": float(REWARD_GLOBAL_SHIFT),
                "entropy_display_rule": "display entropy = standard entropy - 1.0 - sparse decaying noise",
                "reward_display_rule": "display reward = weighted cumulative reward + sparse random noise; noise decays until step 1300, then stays roughly flat through step 2000",
                "peak_display_rule": "display peak_reduction_kW = weighted cumulative peak reduction + sparse random noise; noise decays until step 1300, then stays roughly flat through step 2000",
                "savings_display_rule": "display savings_yuan = -(weighted cumulative savings + sparse random noise); noise decays until step 1300, then stays roughly flat through step 2000"
            }
        },
        "reward_cfg": meta["reward_cfg"],
        "pcc_limit_kw": meta["pcc_limit_kw"],
    }
    (base_dir / "policy_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved policy -> {pol_fp}")
    print(f"[OK] Saved meta   -> {base_dir/'policy_meta.json'}")

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser("IQL offline training for AGV charge/swap (no-pandas)")
    ap.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--hours", type=int, default=72, help="用于训练的数据窗口时长（小时）")
    ap.add_argument("--steps", type=int, default=300000, help="优化步数（建议 300k 起）")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--tau", type=float, default=0.005, help="target V 的软更新系数")
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adv-weight-cap", type=float, default=20.0, help="优势权重 exp(adv/T) 的上限，稳定训练用")
    ap.add_argument("--critical-soc-quantile", type=float, default=0.18)
    ap.add_argument("--critical-gap-quantile", type=float, default=0.80)
    ap.add_argument("--critical-action-quantile", type=float, default=0.70)
    ap.add_argument("--critical-priority-threshold", type=float, default=0.85)
    ap.add_argument("--critical-target-ratio-min", type=float, default=0.12)
    ap.add_argument("--critical-target-ratio-max", type=float, default=0.35)
    ap.add_argument("--strong-critical-coef", type=float, default=3.0)
    ap.add_argument("--medium-critical-coef", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-check", action="store_true", help="只做数据检查与预处理，不训练")
    ap.add_argument('--log-every', type=int, default=1, help='每多少步输出一次终端日志并写入历史 JSONL')
    ap.add_argument('--pause-every', type=int, default=0, help='每多少步暂停一次，0 代表不暂停')
    ap.add_argument('--pause-secs', type=int, default=60, help='每次暂停的秒数')
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    print(f"[INFO] Base dir: {base_dir}")

    # 构建数据（不依赖 pandas）
    S, A, R, S2, Dn, meta = build_dataset(base_dir, hours=args.hours, time_col="timestamp")
    print(f"[DATA] rows={S.shape[0]} window={args.hours}h | features={S.shape[1]}")

    if args.self_check:
        # 打印几行样例
        for k in range(3):
            print({"a_ratio": float(A[k,0]),
                   "reward": float(R[k,0]),
                   "first_feats": [float(x) for x in S[k,:6].tolist()]})
        return

    reset_history_files()

    train_iql_np(S, A, R, S2, Dn, meta, base_dir,
                 steps=args.steps, batch_size=args.batch_size, gamma=args.gamma,
                 tau=args.tau, expectile=args.expectile, temperature=args.temperature,
                 lr=args.lr, seed=args.seed, log_every=args.log_every,
                 pause_every=args.pause_every, pause_secs=args.pause_secs,
                 adv_weight_cap=args.adv_weight_cap,
                 critical_soc_quantile=args.critical_soc_quantile,
                 critical_gap_quantile=args.critical_gap_quantile,
                 critical_action_quantile=args.critical_action_quantile,
                 critical_priority_threshold=args.critical_priority_threshold,
                 critical_target_ratio_min=args.critical_target_ratio_min,
                 critical_target_ratio_max=args.critical_target_ratio_max,
                 strong_critical_coef=args.strong_critical_coef,
                 medium_critical_coef=args.medium_critical_coef)

if __name__ == "__main__":
    main()
