# app/services/rl_model/agv_charge/train_bc_iql.py
# -*- coding: utf-8 -*-
"""
AGV/无人集卡充/换电：成本感知的加权 BC（Cost-Aware Weighted BC）
------------------------------------------------------------------
【本版要点 v2.2】
- 在“行为克隆(BC)”损失上加入“成本/碳/峰值塑形”与“谷段鼓励”，无需改动数据/评估接口；
- 样本重加权：在高电价/高碳/低网荷余量时段，放大“减小功率”的学习信号；
- 峰/谷塑形：对预测比例 r∈(0,1) 施加正则：峰段惩罚 r·1{peak}，谷段鼓励 (1−r)·1{valley}；
- 自适应阈值：电价/因子阈值用数据的分位数（默认 q75 当峰、q25 当谷）自动标定；
- 训练与落盘接口兼容：policy.bin / policy_meta.json；评估仍用 adapter.evaluate_policy()；
- 新增训练摘要与评估摘要 artifacts，便于首页/API 直接解释，不再依赖前端自行拼装。
- 额外写出 IQL 兼容别名文件：policy_evaluate_history.jsonl / policy_train_summary.json。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .adapter import AGVChargeAdapter


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_cpu() -> torch.device:
    return torch.device("cpu")


@dataclass
class FeatureSpec:
    names: List[str]
    mean: List[float]
    std: List[float]
    stats_price: Dict[str, float]
    stats_ef: Dict[str, float]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return float(v) if math.isfinite(v) else float(default)


def _time_feats(t: datetime) -> Tuple[float, float, float, float]:
    h = t.hour + t.minute / 60.0
    hod = 2 * math.pi * h / 24.0
    dow = 2 * math.pi * (t.weekday() / 7.0)
    return math.sin(hod), math.cos(hod), math.sin(dow), math.cos(dow)


def _normalize_dt_key(x) -> str:
    try:
        ts = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except Exception:
        try:
            ts = np.datetime64(x).astype("datetime64[s]").astype(object)
        except Exception:
            return str(x)
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _load_charge_session_fallback(base_dir: Path) -> Dict[str, Dict[str, float]]:
    candidates = [
        base_dir / "data" / "charge_sessions.csv",
        base_dir / "charge_sessions.csv",
    ]
    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        return {}

    try:
        import pandas as pd  # local import to avoid hard dependency at import time
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] charge_sessions fallback unavailable: {e}")
        return {}

    cols = {str(c).strip(): c for c in df.columns}
    t_col = cols.get("timestamp") or cols.get("time")
    vid_col = cols.get("vehicle_id") or cols.get("agv_id") or cols.get("vehicle")
    p_col = cols.get("power_kw") or cols.get("avg_power_kw") or cols.get("kw") or cols.get("power")
    if t_col is None or vid_col is None or p_col is None:
        print(f"[WARN] charge_sessions fallback columns missing: columns={list(df.columns)}")
        return {}

    out: Dict[str, Dict[str, float]] = {}
    bad_rows = 0
    for _, row in df[[t_col, vid_col, p_col]].iterrows():
        try:
            key = _normalize_dt_key(row[t_col])
            vid = str(row[vid_col]).strip()
            p = _safe_float(row[p_col], 0.0)
            if not vid:
                bad_rows += 1
                continue
            out.setdefault(key, {})
            out[key][vid] = max(0.0, out[key].get(vid, 0.0) + max(0.0, p))
        except Exception:
            bad_rows += 1

    nz_steps = sum(1 for m in out.values() if any(v > 1e-6 for v in m.values()))
    nz_points = sum(sum(1 for v in m.values() if v > 1e-6) for m in out.values())
    print(
        "[SESSIONS FALLBACK] path=%s steps=%d nz_steps=%d nz_points=%d bad_rows=%d"
        % (str(csv_path), len(out), nz_steps, nz_points, bad_rows)
    )
    return out


def _summarize_label_vector(name: str, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        print(f"[{name}] empty")
        return
    finite = np.isfinite(arr)
    x = arr[finite] if finite.any() else np.zeros(1, dtype=np.float32)
    nz = x > 1e-6
    print(
        "[%s] n=%d finite=%.2f%% nz=%.2f%% mean=%.4f p50=%.4f p90=%.4f max=%.4f"
        % (
            name,
            arr.size,
            100.0 * float(finite.mean()),
            100.0 * float(nz.mean()),
            _safe_float(float(np.mean(x))),
            _safe_float(float(np.quantile(x, 0.50))),
            _safe_float(float(np.quantile(x, 0.90))),
            _safe_float(float(np.max(x))),
        )
    )


def build_sample_from_obs(
    obs: Dict, max_c_rate: float
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    把一次观测 obs 展开为“每辆车一条样本”。
    返回: (feature_names, X[n,d], y[n], P[n], price[n], ef[n], gridroom[n])
    """
    price = float(obs.get("price_yuan_per_kwh", 0.0))
    ef = float(obs.get("ef_kg_per_kwh", 0.0))
    grid_room = float(obs.get("grid_room_kw", 0.0))
    s_hsin, s_hcos, s_wsin, s_wcos = _time_feats(obs["time"])

    names = [
        "price_yuan_per_kwh", "ef_kg_per_kwh", "grid_room_kw", "time_hsin", "time_hcos", "dow_sin", "dow_cos",
        "soc", "soc_target", "battery_kwh", "p_charge_max_kw", "available", "priority", "eta_min", "temp",
        "c_rate_cap",
    ]

    hist_act = obs.get("_hist_action_for_bc", {})
    feats, labels, pmaxs = [], [], []
    prices, efs, rooms = [], [], []
    for v in obs["vehicles"]:
        soc = float(v.get("soc", 0.0))
        soc_target = float(v.get("soc_target", 0.8))
        batt = float(v.get("battery_kwh", 1.0))
        pmax_hw = float(v.get("p_charge_max_kw", 0.0))
        c_cap = batt * float(max_c_rate)
        pmax = float(min(pmax_hw, c_cap))
        av = float(v.get("available", 1))
        prio = float(v.get("priority", 0))
        eta = float(v.get("eta_min", 0.0))
        temp = float(v.get("temp", 25.0))

        x = [
            price, ef, grid_room, s_hsin, s_hcos, s_wsin, s_wcos,
            soc, soc_target, batt, pmax_hw, av, prio, eta, temp,
            min(1.0, (pmax / max(batt, 1e-6))),
        ]
        feats.append(x)
        labels.append(float(hist_act.get(v["vehicle_id"], 0.0)))
        pmaxs.append(pmax)
        prices.append(price)
        efs.append(ef)
        rooms.append(grid_room)

    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    P = np.array(pmaxs, dtype=np.float32)
    return names, X, y, P, np.array(prices, dtype=np.float32), np.array(efs, dtype=np.float32), np.array(rooms, dtype=np.float32)


@dataclass
class DataAudit:
    n_rows: int
    n_cols: int
    frac_any_nan: float
    frac_any_inf: float
    all_nan_cols: List[int]


def audit_impute_and_standardize(
    X: np.ndarray, y: np.ndarray, P: np.ndarray, feat_names: List[str], raw_price: np.ndarray, raw_ef: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FeatureSpec, DataAudit]:
    n, d = X.shape
    mask_nan = ~np.isfinite(X)
    frac_any_nan = float(mask_nan.any(axis=1).mean())
    frac_any_inf = float(np.isinf(X).any(axis=1).mean())
    all_nan_cols = [i for i in range(d) if (~np.isfinite(X[:, i])).all()]

    X_clean = X.copy()
    X_clean[~np.isfinite(X_clean)] = np.nan
    col_mean = np.nanmean(X_clean, axis=0)
    col_std = np.nanstd(X_clean, axis=0)
    for i in all_nan_cols:
        col_mean[i] = 0.0
        col_std[i] = 1.0
    col_std = np.where(col_std < 1e-6, 1.0, col_std)

    mean_row = np.tile(col_mean.reshape(1, -1), (n, 1))
    X_imp = np.where(np.isfinite(X), X, mean_row)

    y_imp = y.copy()
    y_imp[~np.isfinite(y_imp)] = 0.0
    P_imp = P.copy()
    P_imp[~np.isfinite(P_imp)] = 1.0
    P_imp = np.where(P_imp <= 0.0, 1e-6, P_imp)

    Xn = (X_imp - col_mean) / col_std

    pr = raw_price[np.isfinite(raw_price)]
    ef = raw_ef[np.isfinite(raw_ef)]
    stats_price = {
        "mean": float(np.mean(pr)) if pr.size > 0 else 0.0,
        "std": float(np.std(pr)) if pr.size > 0 else 1.0,
        "q25": float(np.quantile(pr, 0.25)) if pr.size > 0 else 0.0,
        "q75": float(np.quantile(pr, 0.75)) if pr.size > 0 else 0.0,
        "min": float(np.min(pr)) if pr.size > 0 else 0.0,
        "max": float(np.max(pr)) if pr.size > 0 else 0.0,
    }
    stats_ef = {
        "mean": float(np.mean(ef)) if ef.size > 0 else 0.0,
        "std": float(np.std(ef)) if ef.size > 0 else 1.0,
        "q25": float(np.quantile(ef, 0.25)) if ef.size > 0 else 0.0,
        "q75": float(np.quantile(ef, 0.75)) if ef.size > 0 else 0.0,
        "min": float(np.min(ef)) if ef.size > 0 else 0.0,
        "max": float(np.max(ef)) if ef.size > 0 else 0.0,
    }

    spec = FeatureSpec(
        names=feat_names,
        mean=col_mean.astype(np.float32).tolist(),
        std=col_std.astype(np.float32).tolist(),
        stats_price=stats_price,
        stats_ef=stats_ef,
    )
    audit = DataAudit(n_rows=n, n_cols=d, frac_any_nan=frac_any_nan, frac_any_inf=frac_any_inf, all_nan_cols=all_nan_cols)
    return Xn.astype(np.float32), y_imp.astype(np.float32), P_imp.astype(np.float32), spec, audit


class BCDataset(torch.utils.data.Dataset):
    def __init__(self, Xn: np.ndarray, y: np.ndarray, P: np.ndarray,
                 raw_price: np.ndarray, raw_ef: np.ndarray, raw_room: np.ndarray,
                 pos_weight: float = 3.0):
        self.X = torch.from_numpy(Xn).float()
        self.y = torch.from_numpy(y).float()
        self.P = torch.from_numpy(P).float()
        self.price = torch.from_numpy(raw_price).float()
        self.ef = torch.from_numpy(raw_ef).float()
        self.room = torch.from_numpy(raw_room).float()
        self.y_ratio = torch.clamp(self.y / torch.clamp(self.P, min=1e-6), 0.0, 1.0)
        self.w = torch.ones_like(self.y_ratio)
        self.w[self.y > 1e-3] = float(pos_weight)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_ratio[idx], self.P[idx], self.w[idx], self.price[idx], self.ef[idx], self.room[idx]


class PolicyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128, 64, 32]):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class TrainConfig:
    hours: int = 6
    epochs: int = 15
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    pos_weight: float = 3.0
    val_ratio: float = 0.1
    seed: int = 42
    w_price: float = 1.0
    w_ef: float = 0.5
    w_room: float = 0.5
    lam_peak: float = 0.5
    lam_valley: float = 0.3
    eval_every: int = 5
    eval_horizon_hours: int = 6


def build_training_data(adapter: AGVChargeAdapter, hours: int, cfg: TrainConfig):
    set_seed(cfg.seed)
    t0 = adapter._time_index[0]
    t1 = adapter._index[0] + timedelta(hours=hours) if hasattr(adapter, "_index") else t0 + timedelta(hours=hours)

    X_list, y_list, P_list = [], [], []
    price_list, ef_list, room_list = [], [], []
    feat_names_ref = None

    fallback_sessions = _load_charge_session_fallback(adapter.base_dir if hasattr(adapter, "base_dir") else Path(__file__).resolve().parent)
    source_stats = {
        "steps_total": 0,
        "adapter_nonempty_steps": 0,
        "fallback_nonempty_steps": 0,
        "fallback_used_steps": 0,
        "both_empty_steps": 0,
        "adapter_points": 0,
        "fallback_points": 0,
        "chosen_points": 0,
    }
    adapter_vals: List[float] = []
    fallback_vals: List[float] = []
    chosen_vals: List[float] = []

    for item in adapter.sample_transitions(start=t0, end=t1, include_actions_from_history=True):
        obs = item["obs"]
        source_stats["steps_total"] += 1
        ts_key = _normalize_dt_key(obs["time"])

        adapter_action = item.get("action", {}) or {}
        adapter_action = {str(k): max(0.0, _safe_float(v, 0.0)) for k, v in adapter_action.items() if _safe_float(v, 0.0) > 1e-9}
        fallback_action = fallback_sessions.get(ts_key, {}) if fallback_sessions else {}
        fallback_action = {str(k): max(0.0, _safe_float(v, 0.0)) for k, v in fallback_action.items() if _safe_float(v, 0.0) > 1e-9}

        if adapter_action:
            source_stats["adapter_nonempty_steps"] += 1
            source_stats["adapter_points"] += len(adapter_action)
            adapter_vals.extend(adapter_action.values())
        if fallback_action:
            source_stats["fallback_nonempty_steps"] += 1
            source_stats["fallback_points"] += len(fallback_action)
            fallback_vals.extend(fallback_action.values())

        chosen_action = dict(adapter_action)
        used_fallback_here = False
        for vid, p in fallback_action.items():
            if chosen_action.get(vid, 0.0) <= 1e-9:
                chosen_action[vid] = p
                used_fallback_here = True
        if used_fallback_here:
            source_stats["fallback_used_steps"] += 1
        if not chosen_action:
            source_stats["both_empty_steps"] += 1
        else:
            source_stats["chosen_points"] += len(chosen_action)
            chosen_vals.extend(chosen_action.values())

        obs["_hist_action_for_bc"] = chosen_action
        names, X, y, P, pr, ef, rm = build_sample_from_obs(obs, adapter.cfg.max_c_rate)
        if feat_names_ref is None:
            feat_names_ref = names
        X_list.append(X)
        y_list.append(y)
        P_list.append(P)
        price_list.append(pr)
        ef_list.append(ef)
        room_list.append(rm)

    if not X_list:
        raise RuntimeError("No training samples collected. Check data availability.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    P = np.concatenate(P_list)
    raw_price = np.concatenate(price_list)
    raw_ef = np.concatenate(ef_list)
    raw_room = np.concatenate(room_list)

    print(
        "[ACTION SOURCE] steps=%d adapter_nonempty=%d fallback_nonempty=%d fallback_used=%d both_empty=%d adapter_points=%d fallback_points=%d chosen_points=%d"
        % (
            source_stats["steps_total"],
            source_stats["adapter_nonempty_steps"],
            source_stats["fallback_nonempty_steps"],
            source_stats["fallback_used_steps"],
            source_stats["both_empty_steps"],
            source_stats["adapter_points"],
            source_stats["fallback_points"],
            source_stats["chosen_points"],
        )
    )
    _summarize_label_vector("LABEL ADAPTER_RAW_KW", np.array(adapter_vals, dtype=np.float32))
    _summarize_label_vector("LABEL FALLBACK_RAW_KW", np.array(fallback_vals, dtype=np.float32))
    _summarize_label_vector("LABEL CHOSEN_RAW_KW", np.array(chosen_vals, dtype=np.float32))
    _summarize_label_vector("LABEL RAW_ALL_SAMPLES_KW", y)
    _summarize_label_vector("PMAX_ALL_SAMPLES_KW", P)

    Xn, y_imp, P_imp, spec, audit = audit_impute_and_standardize(X, y, P, feat_names_ref, raw_price, raw_ef)

    nan_cols_readable = [feat_names_ref[i] if 0 <= i < len(feat_names_ref) else f"col_{i}" for i in audit.all_nan_cols]
    print(
        "[DATA AUDIT] rows=%d cols=%d nan-row%%=%.2f%% inf-row%%=%.2f%% all-NaN-cols=%s"
        % (audit.n_rows, audit.n_cols, 100 * audit.frac_any_nan, 100 * audit.frac_any_inf, nan_cols_readable)
    )

    n = Xn.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(cfg.val_ratio * n)) if n > 1 else 0
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:] if n_val < n else idx[: max(1, n - 1)]
    if tr_idx.size == 0:
        tr_idx = idx
    if val_idx.size == 0:
        val_idx = tr_idx[: min(1, tr_idx.size)]

    ds_tr = BCDataset(Xn[tr_idx], y_imp[tr_idx], P_imp[tr_idx], raw_price[tr_idx], raw_ef[tr_idx], raw_room[tr_idx], pos_weight=cfg.pos_weight)
    ds_va = BCDataset(Xn[val_idx], y_imp[val_idx], P_imp[val_idx], raw_price[val_idx], raw_ef[val_idx], raw_room[val_idx], pos_weight=cfg.pos_weight)

    y_ratio_preclip = y_imp / np.clip(P_imp, 1e-6, None)
    _summarize_label_vector("LABEL FINAL_RATIO_PRECLIP", y_ratio_preclip)
    y_ratio_all = np.clip(y_ratio_preclip, 0.0, 1.0)
    _summarize_label_vector("LABEL FINAL_RATIO_CLIPPED", y_ratio_all)
    peak_mask_all = ((raw_price >= spec.stats_price.get("q75", 0.0)) | (raw_ef >= spec.stats_ef.get("q75", 0.0)))
    valley_mask_all = ((raw_price <= spec.stats_price.get("q25", 0.0)) & (raw_ef <= spec.stats_ef.get("q25", 0.0)))
    low_room_mask_all = raw_room <= np.quantile(raw_room, 0.25) if raw_room.size else np.zeros_like(raw_room, dtype=bool)

    dataset_profile = {
        "n_samples": int(n),
        "n_features": int(Xn.shape[1]),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "positive_action_ratio": _safe_float(float((y_imp > 1e-3).mean())),
        "target_mean_ratio": _safe_float(float(np.mean(y_ratio_all))),
        "target_p50_ratio": _safe_float(float(np.quantile(y_ratio_all, 0.50))),
        "target_p90_ratio": _safe_float(float(np.quantile(y_ratio_all, 0.90))),
        "target_near_zero_ratio": _safe_float(float((y_ratio_all <= 1e-3).mean())),
        "target_high_ratio_share": _safe_float(float((y_ratio_all >= 0.8).mean())),
        "mean_pmax_kw": _safe_float(float(np.mean(P_imp))),
        "max_pmax_kw": _safe_float(float(np.max(P_imp))),
        "price_mean": _safe_float(spec.stats_price.get("mean", 0.0)),
        "price_q25": _safe_float(spec.stats_price.get("q25", 0.0)),
        "price_q75": _safe_float(spec.stats_price.get("q75", 0.0)),
        "ef_mean": _safe_float(spec.stats_ef.get("mean", 0.0)),
        "ef_q25": _safe_float(spec.stats_ef.get("q25", 0.0)),
        "ef_q75": _safe_float(spec.stats_ef.get("q75", 0.0)),
        "grid_room_min": _safe_float(float(np.min(raw_room))) if raw_room.size else 0.0,
        "grid_room_mean": _safe_float(float(np.mean(raw_room))) if raw_room.size else 0.0,
        "grid_room_q25": _safe_float(float(np.quantile(raw_room, 0.25))) if raw_room.size else 0.0,
        "peak_sample_ratio": _safe_float(float(np.mean(peak_mask_all))),
        "valley_sample_ratio": _safe_float(float(np.mean(valley_mask_all))),
        "low_room_sample_ratio": _safe_float(float(np.mean(low_room_mask_all))),
        "action_source_audit": {
            "steps_total": int(source_stats["steps_total"]),
            "adapter_nonempty_steps": int(source_stats["adapter_nonempty_steps"]),
            "fallback_nonempty_steps": int(source_stats["fallback_nonempty_steps"]),
            "fallback_used_steps": int(source_stats["fallback_used_steps"]),
            "both_empty_steps": int(source_stats["both_empty_steps"]),
            "adapter_points": int(source_stats["adapter_points"]),
            "fallback_points": int(source_stats["fallback_points"]),
            "chosen_points": int(source_stats["chosen_points"]),
            "adapter_raw_nonzero_ratio": _safe_float(float((np.array(adapter_vals, dtype=np.float32) > 1e-6).mean())) if adapter_vals else 0.0,
            "fallback_raw_nonzero_ratio": _safe_float(float((np.array(fallback_vals, dtype=np.float32) > 1e-6).mean())) if fallback_vals else 0.0,
            "chosen_raw_nonzero_ratio": _safe_float(float((np.array(chosen_vals, dtype=np.float32) > 1e-6).mean())) if chosen_vals else 0.0,
        },
        "data_audit": {
            "rows": int(audit.n_rows),
            "cols": int(audit.n_cols),
            "frac_any_nan": _safe_float(audit.frac_any_nan),
            "frac_any_inf": _safe_float(audit.frac_any_inf),
            "all_nan_cols": nan_cols_readable,
        },
    }
    print(
        "[DATA PROFILE] pos=%.2f%% target_mean=%.4f target_p50=%.4f target_p90=%.4f near_zero=%.2f%% high>=0.8=%.2f%% peak=%.2f%% valley=%.2f%% low_room=%.2f%%"
        % (
            100 * dataset_profile["positive_action_ratio"],
            dataset_profile["target_mean_ratio"],
            dataset_profile["target_p50_ratio"],
            dataset_profile["target_p90_ratio"],
            100 * dataset_profile["target_near_zero_ratio"],
            100 * dataset_profile["target_high_ratio_share"],
            100 * dataset_profile["peak_sample_ratio"],
            100 * dataset_profile["valley_sample_ratio"],
            100 * dataset_profile["low_room_sample_ratio"],
        )
    )
    return ds_tr, ds_va, spec, dataset_profile



def evaluate_policy_brief(adapter: AGVChargeAdapter, model: nn.Module, spec: FeatureSpec, device: torch.device, horizon_hours: int = 6, max_c_rate: float | None = None) -> Dict[str, float]:
    """轻量评估：训练期只输出业务指标，不频繁写大文件。"""
    if max_c_rate is None:
        max_c_rate = float(adapter.cfg.max_c_rate)

    policy_fn = make_policy_fn(model, spec, max_c_rate=max_c_rate)

    def constrained(obs):
        raw = policy_fn(obs)
        return adapter._project_to_feasible(obs["time"], raw)

    steps = max(1, int(np.ceil(horizon_hours * 60 / adapter.dt_min)))
    tidx = list(adapter._time_index[:steps])
    total_reward = 0.0
    total_energy = 0.0
    total_cost = 0.0
    total_peak_penalty = 0.0
    total_delay = 0.0
    peak_with_charging = 0.0
    peak_grid_only = 0.0
    charge_steps = 0

    for t in tidx:
        obs = adapter._build_observation_at(t)
        action = constrained(obs) or {}
        reward, info = adapter._calc_reward_at(t, action_power_map=action)
        total_reward += float(reward)
        total_energy += float(info.get("energy_kwh", 0.0))
        total_cost += float(info.get("elec_cost", 0.0)) + float(info.get("carbon_cost", 0.0))
        total_peak_penalty += float(info.get("peak_penalty", 0.0))
        total_delay += float(info.get("delay_penalty", 0.0))
        total_power = float(info.get("total_power_kw", 0.0))
        grid_kw = float(info.get("grid_kw", 0.0))
        peak_with_charging = max(peak_with_charging, grid_kw + total_power)
        peak_grid_only = max(peak_grid_only, grid_kw)
        if total_power > 1e-6:
            charge_steps += 1

    demand_limit = float(adapter.grid.demand_limit_kw) if (adapter.grid and getattr(adapter.grid, 'demand_limit_kw', None) is not None) else None
    if demand_limit is not None:
        peak_headroom_kw = max(0.0, demand_limit - peak_with_charging)
        over_limit_kw = max(0.0, peak_with_charging - demand_limit)
    else:
        peak_headroom_kw = 0.0
        over_limit_kw = 0.0

    return {
        "reward_sum": _safe_float(total_reward),
        "reward_mean": _safe_float(total_reward / max(1, len(tidx))),
        "energy_kwh": _safe_float(total_energy),
        "cost_yuan": _safe_float(total_cost),
        "peak_with_charging_kw": _safe_float(peak_with_charging),
        "peak_grid_only_kw": _safe_float(peak_grid_only),
        "peak_delta_vs_grid_kw": _safe_float(peak_with_charging - peak_grid_only),
        "peak_headroom_kw": _safe_float(peak_headroom_kw),
        "over_limit_kw": _safe_float(over_limit_kw),
        "peak_penalty_yuan": _safe_float(total_peak_penalty),
        "delay_hours": _safe_float(total_delay),
        "charge_step_ratio": _safe_float(charge_steps / max(1, len(tidx))),
        "dispatch_ready": bool(over_limit_kw <= 1e-6),
        "horizon_hours": int(horizon_hours),
    }


def train_bc(ds_tr: BCDataset, ds_va: BCDataset, spec: FeatureSpec, cfg: TrainConfig, device: torch.device, adapter: AGVChargeAdapter | None = None, artifact_dir: Path | None = None):
    model = PolicyMLP(in_dim=len(spec.mean)).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.MSELoss(reduction="none")

    p_q25, p_q75 = spec.stats_price.get("q25", 0.0), spec.stats_price.get("q75", 0.0)
    ef_q25, ef_q75 = spec.stats_ef.get("q25", 0.0), spec.stats_ef.get("q75", 0.0)

    def run_epoch(ds, train: bool = True):
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=train)
        total, denom = 0.0, 0.0
        mae, mape_denom = 0.0, 1e-6
        peak_hits, valley_hits = 0.0, 0.0
        mean_ratio = 0.0
        mean_power = 0.0
        mean_target_ratio = 0.0
        mean_target_kw = 0.0
        pos_target_ratio = 0.0
        near_zero_pred_ratio = 0.0
        high_pred_ratio = 0.0
        mean_cost_weight = 0.0
        mean_base_loss = 0.0
        mean_reg_peak = 0.0
        mean_reg_valley = 0.0
        grad_norm_acc = 0.0
        grad_norm_steps = 0.0
        if train:
            model.train()
        else:
            model.eval()

        for Xb, yr, Pb, Wb, price, ef, room in loader:
            Xb = Xb.to(device)
            yr = yr.to(device)
            Pb = Pb.to(device)
            Wb = Wb.to(device)
            price = price.to(device)
            ef = ef.to(device)
            room = room.to(device)

            p_mu, p_sd = spec.stats_price["mean"], max(spec.stats_price["std"], 1e-6)
            e_mu, e_sd = spec.stats_ef["mean"], max(spec.stats_ef["std"], 1e-6)
            p_z = (price - p_mu) / p_sd
            e_z = (ef - e_mu) / e_sd
            room_pos = torch.clamp(room, min=0.0)
            room_w = 1.0 / torch.clamp(1.0 + room_pos, min=1.0)
            w_cost = 1.0 + cfg.w_price * torch.relu(p_z) + cfg.w_ef * torch.relu(e_z) + cfg.w_room * (1.0 - room_w)
            w_cost = torch.clamp(w_cost, 0.1, 10.0)

            with torch.set_grad_enabled(train):
                r = model(Xb)
                yhat = r * Pb
                y_true = yr * Pb
                base = crit(yhat, y_true).squeeze(-1)

                peak_mask = ((price >= p_q75) | (ef >= ef_q75)).float()
                valley_mask = ((price <= p_q25) & (ef <= ef_q25)).float()
                reg_peak = cfg.lam_peak * (r * peak_mask)
                reg_valley = cfg.lam_valley * ((1.0 - r) * valley_mask)
                weighted_base = base * Wb * w_cost
                loss_vec = weighted_base + reg_peak + reg_valley

                if not torch.isfinite(loss_vec).all():
                    valid = torch.isfinite(loss_vec)
                    loss = loss_vec[valid].mean() if valid.any() else torch.tensor(0.0, device=device)
                else:
                    loss = loss_vec.mean()

                if train:
                    opt.zero_grad()
                    loss.backward()
                    grad_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            gn = float(p.grad.detach().data.norm(2).item())
                            grad_sq += gn * gn
                    grad_norm_acc += math.sqrt(max(grad_sq, 0.0))
                    grad_norm_steps += 1.0
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    opt.step()

            batch_n = Xb.shape[0]
            total += float(loss.item()) * batch_n
            denom += batch_n
            with torch.no_grad():
                e = (yhat - y_true).abs()
                e = torch.where(torch.isfinite(e), e, torch.zeros_like(e))
                mae += float(e.sum().item())
                mape_denom += float(torch.clamp(y_true.abs(), min=1e-3).sum().item())
                peak_hits += float(peak_mask.sum().item())
                valley_hits += float(valley_mask.sum().item())
                mean_ratio += float(r.mean().item()) * batch_n
                mean_power += float(yhat.mean().item()) * batch_n
                mean_target_ratio += float(yr.mean().item()) * batch_n
                mean_target_kw += float(y_true.mean().item()) * batch_n
                pos_target_ratio += float((yr > 1e-3).float().mean().item()) * batch_n
                near_zero_pred_ratio += float((r <= 1e-3).float().mean().item()) * batch_n
                high_pred_ratio += float((r >= 0.8).float().mean().item()) * batch_n
                mean_cost_weight += float(w_cost.mean().item()) * batch_n
                mean_base_loss += float(weighted_base.mean().item()) * batch_n
                mean_reg_peak += float(reg_peak.mean().item()) * batch_n
                mean_reg_valley += float(reg_valley.mean().item()) * batch_n

        return {
            "loss": total / max(1, denom),
            "mae": mae / max(1.0, denom),
            "mape_proxy": (mae / mape_denom),
            "peak_ratio": peak_hits / max(1.0, denom),
            "valley_ratio": valley_hits / max(1.0, denom),
            "mean_pred_ratio": mean_ratio / max(1.0, denom),
            "mean_pred_kw": mean_power / max(1.0, denom),
            "mean_target_ratio": mean_target_ratio / max(1.0, denom),
            "mean_target_kw": mean_target_kw / max(1.0, denom),
            "positive_target_ratio": pos_target_ratio / max(1.0, denom),
            "pred_near_zero_ratio": near_zero_pred_ratio / max(1.0, denom),
            "pred_high_ratio": high_pred_ratio / max(1.0, denom),
            "cost_weight_mean": mean_cost_weight / max(1.0, denom),
            "base_loss_mean": mean_base_loss / max(1.0, denom),
            "reg_peak_mean": mean_reg_peak / max(1.0, denom),
            "reg_valley_mean": mean_reg_valley / max(1.0, denom),
            "grad_norm_mean": grad_norm_acc / max(1.0, grad_norm_steps),
        }

    hist: List[Dict] = []
    metrics_jsonl_path = None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path = artifact_dir / "policy_train_metrics.jsonl"
        try:
            metrics_jsonl_path.write_text("", encoding="utf-8")
        except Exception:
            metrics_jsonl_path = None

    for ep in range(cfg.epochs):
        tr = run_epoch(ds_tr, train=True)
        va = run_epoch(ds_va, train=False)
        row = {
            "epoch": ep + 1,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_mae": tr["mae"],
            "val_mae": va["mae"],
            "train_mape": tr["mape_proxy"],
            "val_mape": va["mape_proxy"],
            "train_peak_ratio": tr["peak_ratio"],
            "val_peak_ratio": va["peak_ratio"],
            "train_valley_ratio": tr["valley_ratio"],
            "val_valley_ratio": va["valley_ratio"],
            "train_mean_pred_ratio": tr["mean_pred_ratio"],
            "val_mean_pred_ratio": va["mean_pred_ratio"],
            "train_mean_pred_kw": tr["mean_pred_kw"],
            "val_mean_pred_kw": va["mean_pred_kw"],
            "train_mean_target_ratio": tr["mean_target_ratio"],
            "val_mean_target_ratio": va["mean_target_ratio"],
            "train_mean_target_kw": tr["mean_target_kw"],
            "val_mean_target_kw": va["mean_target_kw"],
            "train_positive_target_ratio": tr["positive_target_ratio"],
            "val_positive_target_ratio": va["positive_target_ratio"],
            "train_pred_near_zero_ratio": tr["pred_near_zero_ratio"],
            "val_pred_near_zero_ratio": va["pred_near_zero_ratio"],
            "train_pred_high_ratio": tr["pred_high_ratio"],
            "val_pred_high_ratio": va["pred_high_ratio"],
            "train_cost_weight_mean": tr["cost_weight_mean"],
            "val_cost_weight_mean": va["cost_weight_mean"],
            "train_base_loss_mean": tr["base_loss_mean"],
            "val_base_loss_mean": va["base_loss_mean"],
            "train_reg_peak_mean": tr["reg_peak_mean"],
            "val_reg_peak_mean": va["reg_peak_mean"],
            "train_reg_valley_mean": tr["reg_valley_mean"],
            "val_reg_valley_mean": va["reg_valley_mean"],
            "train_grad_norm_mean": tr["grad_norm_mean"],
        }
        if adapter is not None and int(cfg.eval_every) > 0 and (((ep + 1) % int(cfg.eval_every) == 0) or (ep + 1 == cfg.epochs)):
            biz = evaluate_policy_brief(adapter, model, spec, device, horizon_hours=int(cfg.eval_horizon_hours), max_c_rate=adapter.cfg.max_c_rate)
            row.update({
                "reward_sum": biz.get("reward_sum", 0.0),
                "reward_mean": biz.get("reward_mean", 0.0),
                "cost_yuan": biz.get("cost_yuan", 0.0),
                "peak_with_charging_kw": biz.get("peak_with_charging_kw", 0.0),
                "peak_grid_only_kw": biz.get("peak_grid_only_kw", 0.0),
                "peak_delta_vs_grid_kw": biz.get("peak_delta_vs_grid_kw", 0.0),
                "peak_headroom_kw": biz.get("peak_headroom_kw", 0.0),
                "over_limit_kw": biz.get("over_limit_kw", 0.0),
                "energy_kwh": biz.get("energy_kwh", 0.0),
                "charge_step_ratio": biz.get("charge_step_ratio", 0.0),
                "dispatch_ready": bool(biz.get("dispatch_ready", False)),
                "eval_horizon_hours": biz.get("horizon_hours", int(cfg.eval_horizon_hours)),
            })

        row = {k: (_safe_float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()}
        hist.append(row)
        if metrics_jsonl_path is not None:
            try:
                with metrics_jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                pass

        print(
            f"[E{ep+1:02d}] tr_loss={row['train_loss']:.6f} va_loss={row['val_loss']:.6f} "
            f"tr_MAE={row['train_mae']:.4f} va_MAE={row['val_mae']:.4f} "
            f"tr_ratio={row['train_mean_pred_ratio']:.4f}/{row['train_mean_target_ratio']:.4f} "
            f"va_ratio={row['val_mean_pred_ratio']:.4f}/{row['val_mean_target_ratio']:.4f} "
            f"tr_kw={row['train_mean_pred_kw']:.2f}/{row['train_mean_target_kw']:.2f} "
            f"va_kw={row['val_mean_pred_kw']:.2f}/{row['val_mean_target_kw']:.2f}"
        )
        print(
            f"      near0 tr/va={100*row['train_pred_near_zero_ratio']:.2f}%/{100*row['val_pred_near_zero_ratio']:.2f}% "
            f"high>=0.8 tr/va={100*row['train_pred_high_ratio']:.2f}%/{100*row['val_pred_high_ratio']:.2f}% "
            f"pos_target tr/va={100*row['train_positive_target_ratio']:.2f}%/{100*row['val_positive_target_ratio']:.2f}%"
        )
        print(
            f"      peak tr/va={100*row['train_peak_ratio']:.2f}%/{100*row['val_peak_ratio']:.2f}% "
            f"valley tr/va={100*row['train_valley_ratio']:.2f}%/{100*row['val_valley_ratio']:.2f}% "
            f"cost_w tr/va={row['train_cost_weight_mean']:.3f}/{row['val_cost_weight_mean']:.3f} grad={row['train_grad_norm_mean']:.3f}"
        )
        print(
            f"      loss_parts tr(base/peak/valley)={row['train_base_loss_mean']:.6f}/{row['train_reg_peak_mean']:.6f}/{row['train_reg_valley_mean']:.6f} "
            f"va(base/peak/valley)={row['val_base_loss_mean']:.6f}/{row['val_reg_peak_mean']:.6f}/{row['val_reg_valley_mean']:.6f}"
        )
        if 'reward_sum' in row:
            print(
                f"      biz reward={row['reward_sum']:.2f} reward_mean={row['reward_mean']:.4f} cost={row['cost_yuan']:.2f} "
                f"peak={row['peak_with_charging_kw']:.2f} base_peak={row['peak_grid_only_kw']:.2f} dpeak={row['peak_delta_vs_grid_kw']:.2f} "
                f"headroom={row['peak_headroom_kw']:.2f} over={row['over_limit_kw']:.2f} energy={row['energy_kwh']:.2f} "
                f"charge_steps={100*row['charge_step_ratio']:.2f}% ready={row['dispatch_ready']}"
            )
    return model, hist


def summarize_train_history(train_hist: List[Dict]) -> Dict:
    if not train_hist:
        return {
            "epochs": 0,
            "status": "no_history",
            "best_epoch_by_val_loss": None,
            "final_epoch": {},
            "improvement": {},
        }

    best = min(train_hist, key=lambda x: x.get("val_loss", float("inf")))
    first = train_hist[0]
    last = train_hist[-1]
    overfit_gap = _safe_float(last.get("val_loss", 0.0) - last.get("train_loss", 0.0))
    return {
        "epochs": len(train_hist),
        "status": "ok",
        "best_epoch_by_val_loss": int(best.get("epoch", 0) or 0),
        "best_val_loss": _safe_float(best.get("val_loss", 0.0)),
        "final_epoch": last,
        "improvement": {
            "train_loss_drop": _safe_float(first.get("train_loss", 0.0) - last.get("train_loss", 0.0)),
            "val_loss_drop": _safe_float(first.get("val_loss", 0.0) - last.get("val_loss", 0.0)),
            "train_mae_drop": _safe_float(first.get("train_mae", 0.0) - last.get("train_mae", 0.0)),
            "val_mae_drop": _safe_float(first.get("val_mae", 0.0) - last.get("val_mae", 0.0)),
        },
        "final_gap": {
            "loss_gap": overfit_gap,
            "mae_gap": _safe_float(last.get("val_mae", 0.0) - last.get("train_mae", 0.0)),
        },
        "behavior": {
            "final_train_mean_pred_ratio": _safe_float(last.get("train_mean_pred_ratio", 0.0)),
            "final_val_mean_pred_ratio": _safe_float(last.get("val_mean_pred_ratio", 0.0)),
            "final_train_mean_pred_kw": _safe_float(last.get("train_mean_pred_kw", 0.0)),
            "final_val_mean_pred_kw": _safe_float(last.get("val_mean_pred_kw", 0.0)),
            "final_train_mean_target_ratio": _safe_float(last.get("train_mean_target_ratio", 0.0)),
            "final_val_mean_target_ratio": _safe_float(last.get("val_mean_target_ratio", 0.0)),
            "final_train_pred_near_zero_ratio": _safe_float(last.get("train_pred_near_zero_ratio", 0.0)),
            "final_val_pred_near_zero_ratio": _safe_float(last.get("val_pred_near_zero_ratio", 0.0)),
            "final_train_pred_high_ratio": _safe_float(last.get("train_pred_high_ratio", 0.0)),
            "final_val_pred_high_ratio": _safe_float(last.get("val_pred_high_ratio", 0.0)),
        },
        "recent_epochs": train_hist[-5:],
    }


def save_policy(base_dir: Path, model: nn.Module, spec: FeatureSpec, cfg: TrainConfig, train_hist: List[Dict]):
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), base_dir / "policy.bin")
    training_summary = summarize_train_history(train_hist)
    meta = {
        "version": "bc_costaware_v2_2",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "feature_names": spec.names,
        "standardize": {"mean": spec.mean, "std": spec.std},
        "price_stats": spec.stats_price,
        "ef_stats": spec.stats_ef,
        "output": {"type": "ratio_in_[0,1] * pmax"},
        "train": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "pos_weight": cfg.pos_weight,
            "val_ratio": cfg.val_ratio,
            "w_price": cfg.w_price,
            "w_ef": cfg.w_ef,
            "w_room": cfg.w_room,
            "lam_peak": cfg.lam_peak,
            "lam_valley": cfg.lam_valley,
            "history": train_hist[-5:],
            "summary": training_summary,
        },
        "thresholds": {
            "price_peak_q75": _safe_float(spec.stats_price.get("q75", 0.0)),
            "price_valley_q25": _safe_float(spec.stats_price.get("q25", 0.0)),
            "ef_peak_q75": _safe_float(spec.stats_ef.get("q75", 0.0)),
            "ef_valley_q25": _safe_float(spec.stats_ef.get("q25", 0.0)),
        },
        "model_arch": {"hidden_layers": [128, 64, 32], "activation": "ReLU", "out": "Sigmoid"},
    }
    (base_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Saved policy -> {base_dir / 'policy.bin'}")


def make_policy_fn(model: nn.Module, spec: FeatureSpec, max_c_rate: float):
    device = device_cpu()
    model = model.to(device).eval()

    def _fn(obs: Dict) -> Dict[str, float]:
        price = float(obs.get("price_yuan_per_kwh", 0.0))
        ef = float(obs.get("ef_kg_per_kwh", 0.0))
        grid_room = float(obs.get("grid_room_kw", 0.0))
        s_hsin, s_hcos, s_wsin, s_wcos = _time_feats(obs["time"])

        feats, pmaxs, vids, avs = [], [], [], []
        for v in obs["vehicles"]:
            soc = float(v.get("soc", 0.0))
            soc_target = float(v.get("soc_target", 0.8))
            batt = float(v.get("battery_kwh", 1.0))
            pmax_hw = float(v.get("p_charge_max_kw", 0.0))
            c_cap = batt * float(max_c_rate)
            pmax = float(min(pmax_hw, c_cap))
            av = float(v.get("available", 1))
            prio = float(v.get("priority", 0))
            eta = float(v.get("eta_min", 0.0))
            temp = float(v.get("temp", 25.0))

            x = [
                price, ef, grid_room, s_hsin, s_hcos, s_wsin, s_wcos,
                soc, soc_target, batt, pmax_hw, av, prio, eta, temp,
                min(1.0, (pmax / max(batt, 1e-6))),
            ]
            feats.append(x)
            pmaxs.append(pmax)
            vids.append(v["vehicle_id"])
            avs.append(av)

        if not feats:
            return {}
        X = np.array(feats, dtype=np.float32)
        X_imp = np.where(
            np.isfinite(X),
            X,
            np.tile(np.array(spec.mean, dtype=np.float32).reshape(1, -1), (X.shape[0], 1)),
        )
        mu = np.array(spec.mean, dtype=np.float32)
        sd = np.array([s if s > 1e-6 else 1.0 for s in spec.std], dtype=np.float32)
        Xn = (X_imp - mu) / sd

        with torch.no_grad():
            r = model(torch.from_numpy(Xn).float().to(device))
        p = r.cpu().numpy() * np.array(pmaxs, dtype=np.float32)
        out = {}
        for vid, pv, av in zip(vids, p.tolist(), avs):
            if av > 0:
                out[vid] = float(max(0.0, pv))
        return out

    return _fn


def write_training_artifacts(
    adapter: AGVChargeAdapter,
    cfg: TrainConfig,
    spec: FeatureSpec,
    train_hist: List[Dict],
    dataset_profile: Dict,
    eval_res: Dict,
):
    artifact_dir = adapter.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    training_summary = summarize_train_history(train_hist)
    rollout_kpi = eval_res.get("kpi", {}) if isinstance(eval_res, dict) else {}
    rollout_summary = {
        "history_path": eval_res.get("history_path") if isinstance(eval_res, dict) else None,
        "kpi": rollout_kpi,
        "dispatch_ready": bool(rollout_kpi.get("dispatch_ready", False)) if isinstance(rollout_kpi, dict) else False,
        "peak_reduction_kw": _safe_float(rollout_kpi.get("peak_reduction_kw", 0.0)) if isinstance(rollout_kpi, dict) else 0.0,
        "cost_saving_yuan": _safe_float(rollout_kpi.get("cost_saving_yuan", 0.0)) if isinstance(rollout_kpi, dict) else 0.0,
        "carbon_reduction_kg": _safe_float(rollout_kpi.get("carbon_reduction_kg", 0.0)) if isinstance(rollout_kpi, dict) else 0.0,
    }

    adapter_profile = adapter.module_a_data_profile() if hasattr(adapter, "module_a_data_profile") else {}
    adapter_latest = adapter.summarize_latest_window(12) if hasattr(adapter, "summarize_latest_window") else {}

    training_run_summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "module": "agv_charge",
        "model_version": "bc_costaware_v2_2",
        "training_window_hours": int(cfg.hours),
        "training_config": {
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": _safe_float(cfg.lr),
            "weight_decay": _safe_float(cfg.weight_decay),
            "pos_weight": _safe_float(cfg.pos_weight),
            "val_ratio": _safe_float(cfg.val_ratio),
            "seed": int(cfg.seed),
            "w_price": _safe_float(cfg.w_price),
            "w_ef": _safe_float(cfg.w_ef),
            "w_room": _safe_float(cfg.w_room),
            "lam_peak": _safe_float(cfg.lam_peak),
            "lam_valley": _safe_float(cfg.lam_valley),
        },
        "feature_spec": {
            "n_features": len(spec.names),
            "feature_names": spec.names,
            "price_stats": spec.stats_price,
            "ef_stats": spec.stats_ef,
        },
        "dataset_profile": dataset_profile,
        "training_summary": training_summary,
        "rollout_summary": rollout_summary,
        "adapter_profile": adapter_profile,
        "adapter_latest_window": adapter_latest,
        "explainability": {
            "objective": "在高价/高碳/低余量时更保守，在低价/低碳谷段更积极",
            "peak_rule": "price>=q75 或 ef>=q75 时抑制更高功率",
            "valley_rule": "price<=q25 且 ef<=q25 时鼓励补能",
            "operator_readout": [
                "看 peak_reduction_kw 判断是否真的削峰",
                "看 cost_saving_yuan / carbon_reduction_kg 判断经济与碳效益",
                "看 final_gap.loss_gap 判断训练/验证是否偏离过大",
            ],
        },
    }

    (artifact_dir / "training_history.json").write_text(json.dumps(train_hist, indent=2, ensure_ascii=False), encoding="utf-8")
    (artifact_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (artifact_dir / "module_a_training_summary.json").write_text(json.dumps(training_run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    bc_eval_report = {
        "offline_last_epoch": train_hist[-1] if train_hist else {},
        "training_summary": training_summary,
        "dataset_profile": dataset_profile,
        "rollout_kpi": rollout_kpi,
        "history_path": eval_res.get("history_path") if isinstance(eval_res, dict) else None,
    }
    (artifact_dir / "bc_eval_report.json").write_text(json.dumps(bc_eval_report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 与 IQL 口径兼容的别名文件 ----
    compat_summary = {
        "created_at": training_run_summary.get("created_at"),
        "module": "agv_charge",
        "trainer": "bc_costaware_label_audit",
        "reward_is_batch_mean": False,
        "steps": int(cfg.epochs),
        "epochs": int(cfg.epochs),
        "peak_reduction_kW": _safe_float(rollout_kpi.get("peak_reduction_kw", 0.0)) if isinstance(rollout_kpi, dict) else 0.0,
        "savings_yuan": _safe_float(rollout_kpi.get("cost_saving_yuan", 0.0)) if isinstance(rollout_kpi, dict) else 0.0,
        "carbon_reduction_kg": _safe_float(rollout_kpi.get("carbon_reduction_kg", 0.0)) if isinstance(rollout_kpi, dict) else 0.0,
        "dispatch_ready": bool(rollout_kpi.get("dispatch_ready", False)) if isinstance(rollout_kpi, dict) else False,
        "train_loss": _safe_float(training_summary.get("final_epoch", {}).get("train_loss", 0.0)),
        "val_loss": _safe_float(training_summary.get("final_epoch", {}).get("val_loss", 0.0)),
        "train_mae": _safe_float(training_summary.get("final_epoch", {}).get("train_mae", 0.0)),
        "val_mae": _safe_float(training_summary.get("final_epoch", {}).get("val_mae", 0.0)),
        "train_mean_pred_ratio": _safe_float(training_summary.get("final_epoch", {}).get("train_mean_pred_ratio", 0.0)),
        "val_mean_pred_ratio": _safe_float(training_summary.get("final_epoch", {}).get("val_mean_pred_ratio", 0.0)),
        "history_path": eval_res.get("history_path") if isinstance(eval_res, dict) else None,
        "ts": datetime.utcnow().isoformat() + "Z",
    }
    (artifact_dir / "policy_train_summary.json").write_text(
        json.dumps(compat_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    src_history = Path(str(eval_res.get("history_path"))) if isinstance(eval_res, dict) and eval_res.get("history_path") else None
    dst_history = artifact_dir / "policy_evaluate_history.jsonl"
    if src_history and src_history.exists():
        try:
            if src_history.resolve() != dst_history.resolve():
                shutil.copyfile(src_history, dst_history)
        except Exception:
            try:
                dst_history.write_text(src_history.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

    print(f"[OK] Wrote training artifacts -> {artifact_dir}")


def quick_train(base_dir: Path, tcfg: TrainConfig):
    set_seed(tcfg.seed)
    dev = device_cpu()

    adapter = AGVChargeAdapter(base_dir=base_dir)
    adapter.self_check()

    ds_tr, ds_va, spec, dataset_profile = build_training_data(adapter, hours=tcfg.hours, cfg=tcfg)
    model, hist = train_bc(ds_tr, ds_va, spec, tcfg, device=dev, adapter=adapter, artifact_dir=adapter.artifact_dir)
    save_policy(base_dir, model, spec, tcfg, hist)

    policy_fn = make_policy_fn(model, spec, max_c_rate=adapter.cfg.max_c_rate)

    def policy_with_constraints(obs):
        raw = policy_fn(obs)
        return adapter._project_to_feasible(obs["time"], raw)

    eval_res = adapter.evaluate_policy(policy_with_constraints, horizon_hours=min(6, tcfg.hours))
    write_training_artifacts(
        adapter=adapter,
        cfg=tcfg,
        spec=spec,
        train_hist=hist,
        dataset_profile=dataset_profile,
        eval_res=eval_res,
    )


def main():
    p = argparse.ArgumentParser(description="AGV charge/swap - Cost-Aware Weighted BC")
    p.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parent), help="directory that contains config.yaml and data/")
    p.add_argument("--quick-train", action="store_true", help="run a quick cost-aware BC training + evaluation")
    p.add_argument("--hours", type=int, default=6, help="hours of data window for training")
    p.add_argument("--epochs", type=int, default=15, help="training epochs")
    p.add_argument("--batch-size", type=int, default=512, help="mini-batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization")
    p.add_argument("--pos-weight", type=float, default=3.0, help="weight for positive (charging) samples")
    p.add_argument("--val-ratio", type=float, default=0.1, help="validation split ratio")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--w-price", type=float, default=1.0, help="price reweight strength")
    p.add_argument("--w-ef", type=float, default=0.5, help="emission factor reweight strength")
    p.add_argument("--w-room", type=float, default=0.5, help="grid room reweight strength")
    p.add_argument("--lam-peak", type=float, default=0.5, help="peak penalty coefficient")
    p.add_argument("--lam-valley", type=float, default=0.3, help="valley encouragement coefficient")
    p.add_argument("--eval-every", type=int, default=5, help="print rollout business KPI every N epochs")
    p.add_argument("--eval-horizon-hours", type=int, default=6, help="rollout horizon used for per-epoch business KPI")
    args = p.parse_args()

    cfg = TrainConfig(
        hours=args.hours,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        val_ratio=args.val_ratio,
        seed=args.seed,
        w_price=args.w_price,
        w_ef=args.w_ef,
        w_room=args.w_room,
        lam_peak=args.lam_peak,
        lam_valley=args.lam_valley,
        eval_every=args.eval_every,
        eval_horizon_hours=args.eval_horizon_hours,
    )

    quick_train(base_dir=Path(args.base_dir), tcfg=cfg)


if __name__ == "__main__":
    main()
