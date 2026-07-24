# -*- coding: utf-8 -*-
"""
AGV充/换电策略回放评估与出图模块（无 pandas 依赖）
------------------------------------------------
功能：
1) 读取已训练策略（policy.bin + policy_meta.json）
2) 在最近 N 小时的窗口内，用策略生成“预测充电功率”，与历史真实充电对比
3) 构造“假设新策略”下的 PCC 曲线（pcc_adj = pcc_baseline - hist_charge + pred_charge）
4) 计算成本/碳/峰值/平滑等指标，输出 PNG 图和评估 JSON

输入数据目录：{BASE}/data 下 9 个文件（与训练同名）
输出目录：    {BASE}/artifacts 下生成若干 PNG/JSON 文件

如何自检（只评估&出图）：
    python -m app.services.rl_model.agv_charge.module --self-check --eval-hours 6
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn

# 使用非交互后端以便服务器/无显示环境
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT_MIN = 5  # 时间步 5 分钟


# -----------------------------
# 通用：时间解析/时间特征
# -----------------------------
def parse_ts(s: str) -> datetime:
    s = s.strip()
    fmts = ["%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S"]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            pass
    if "." in s:
        s2 = s.split(".")[0]
        for f in fmts:
            try:
                return datetime.strptime(s2, f)
            except Exception:
                pass
    raise ValueError(f"无法解析时间: {s}")


def time_feats(ts: datetime) -> Tuple[float, float, float, float]:
    hod = ts.hour + ts.minute / 60.0
    dow = ts.weekday()
    hsin = np.sin(2 * np.pi * hod / 24.0)
    hcos = np.cos(2 * np.pi * hod / 24.0)
    dsin = np.sin(2 * np.pi * (dow / 7.0))
    dcos = np.cos(2 * np.pi * (dow / 7.0))
    return float(hsin), float(hcos), float(dsin), float(dcos)


# -----------------------------
# 纯 Python 读 CSV -> 列字典
# -----------------------------
def read_csv_cols(fp: Path, required: List[str] = None) -> Dict[str, List[str]]:
    if not fp.exists():
        raise FileNotFoundError(fp)
    rows = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(fp, "r", encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            break
        except Exception:
            rows = None
    if not rows:
        raise RuntimeError(f"{fp} 为空或无法解析")
    header = [str(h).strip() for h in rows[0]]
    body = rows[1:]
    L = len(header)
    cols = {c: [] for c in header}
    for r in body:
        rr = [x.strip() for x in r[:L]] + [""] * max(0, L - len(r))
        for j, c in enumerate(header):
            cols[c].append(rr[j])
    if required:
        miss = [c for c in required if c not in cols]
        if miss:
            raise ValueError(f"{fp.name} 缺少列: {miss}")
    return cols


# -----------------------------
# MLP 定义（与策略导出兼容）
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, out_act: str | None = None):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_act = out_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.out_act == "Sigmoid" or self.out_act == "sigmoid":
            return torch.sigmoid(y)
        return y


# -----------------------------
# 载入策略 & Meta
# -----------------------------
def load_policy(base_dir: Path):
    meta_fp = base_dir / "policy_meta.json"
    pol_fp = base_dir / "policy.bin"
    if not meta_fp.exists() or not pol_fp.exists():
        raise SystemExit("[STOP] 未找到 policy_meta.json 或 policy.bin，请先训练策略。")

    meta = json.loads(meta_fp.read_text(encoding="utf-8"))
    feat_names: List[str] = meta["feature_names"]
    mean = np.asarray(meta["standardize"]["mean"], dtype=np.float32)
    std = np.asarray(meta["standardize"]["std"], dtype=np.float32)
    std[std == 0.0] = 1.0

    arch = meta.get("model_arch", {"hidden_layers": [128, 128], "out": "Sigmoid"})
    hidden = arch.get("hidden_layers", [128, 128])
    out_act = arch.get("out", "Sigmoid")

    s_dim = len(feat_names)
    pi = MLP(in_dim=s_dim, hidden=hidden, out_dim=1, out_act=out_act)
    state = torch.load(pol_fp, map_location="cpu")
    # 兼容两种保存格式（train_iql.py 保存为 {"state_dict":...}）
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    pi.load_state_dict(sd)
    pi.eval()

    # 奖励配置（用于成本/罚金计算）
    rw = meta.get("reward_cfg", {"w_price": 1.2, "w_ef": 0.6, "lam_peak": 0.8, "lam_smooth": 0.05})
    pcc_limit = float(meta.get("pcc_limit_kw", 14000.0))

    return pi, feat_names, mean, std, rw, pcc_limit


# -----------------------------
# 构建评估窗口数据（不依赖 pandas）
# -----------------------------
def build_eval_window(base_dir: Path, hours: int = 6, time_col: str = "timestamp"):
    data_dir = base_dir / "data"
    price_cols = read_csv_cols(data_dir / "market_price.csv", [time_col, "price_yuan_per_kwh"])
    all_ts = sorted(parse_ts(t) for t in price_cols[time_col])
    if not all_ts:
        raise SystemExit("[STOP] market_price.csv 无有效时间点")
    t_end = all_ts[-1]
    t_start = t_end - timedelta(hours=hours)
    ts_list = [t for t in all_ts if (t > t_start and t <= t_end)]
    ts_list = sorted(ts_list)

    ef_cols = read_csv_cols(data_dir / "grid_ef.csv", [time_col, "ef_kg_per_kwh"])
    grid_cols = read_csv_cols(data_dir / "grid_meter.csv", [time_col, "pcc_kw"])
    vs_cols = read_csv_cols(
        data_dir / "vehicle_state.csv",
        [time_col, "vehicle_id", "soc", "available", "priority", "eta_min", "temp"],
    )
    sess_cols = read_csv_cols(
        data_dir / "charge_sessions.csv",
        [time_col, "vehicle_id", "power_kw", "station_id", "charger_id"],
    )
    veh_cols = read_csv_cols(
        data_dir / "vehicles_master.csv",
        ["vehicle_id", "battery_kwh", "p_charge_max_kw", "soc_min", "soc_max", "soc_target", "can_swap"],
    )

    # 时间映射
    def map_ts(cols: Dict[str, List[str]], val_col: str) -> Dict[datetime, float]:
        m = {}
        for t, v in zip(cols[time_col], cols[val_col]):
            try:
                m[parse_ts(t)] = float(v or 0.0)
            except Exception:
                continue
        return m

    price_map = map_ts(price_cols, "price_yuan_per_kwh")
    ef_map = map_ts(ef_cols, "ef_kg_per_kwh")
    pcc_map = map_ts(grid_cols, "pcc_kw")

    # 车辆静态参数
    veh_ids = veh_cols["vehicle_id"]
    batt_map = {}
    pmax_map = {}
    stgt_map = {}
    for i, vid in enumerate(veh_ids):
        try:
            batt_map[vid] = float(veh_cols["battery_kwh"][i] or 0.0)
            pmax_map[vid] = float(veh_cols["p_charge_max_kw"][i] or 0.0)
            stgt_map[vid] = float(veh_cols["soc_target"][i] or 0.8)
        except Exception:
            batt_map[vid] = 0.0
            pmax_map[vid] = 0.0
            stgt_map[vid] = 0.8

    # 车辆状态 (ts,vid) -> (soc,avail,priority,eta,temp)
    state_map: Dict[Tuple[datetime, str], Tuple[float, int, int, float, float]] = {}
    for t, vid, soc, av, pri, eta, tmp in zip(
        vs_cols[time_col], vs_cols["vehicle_id"], vs_cols["soc"], vs_cols["available"], vs_cols["priority"], vs_cols["eta_min"], vs_cols["temp"]
    ):
        try:
            ts = parse_ts(t)
        except Exception:
            continue
        try:
            soc_f = float(soc or 0.6)
            av_i = int(float(av or 1))
            pr_i = int(float(pr or 0))
            eta_f = float(eta or 30.0)
            tmp_f = float(tmp or 25.0)
        except Exception:
            soc_f, av_i, pr_i, eta_f, tmp_f = 0.6, 1, 0, 30.0, 25.0
        state_map[(ts, vid)] = (soc_f, av_i, pr_i, eta_f, tmp_f)

    # 历史充电会话，聚合到 (ts) -> 合计功率（用于对比）
    hist_charge_map: Dict[datetime, float] = {ts: 0.0 for ts in ts_list}
    for t, vid, p in zip(sess_cols[time_col], sess_cols["vehicle_id"], sess_cols["power_kw"]):
        try:
            ts = parse_ts(t)
        except Exception:
            continue
        if ts in hist_charge_map:
            try:
                hist_charge_map[ts] += float(p or 0.0)
            except Exception:
                continue

    return {
        "ts_list": ts_list,
        "price_map": price_map,
        "ef_map": ef_map,
        "pcc_map": pcc_map,
        "veh_ids": veh_ids,
        "batt_map": batt_map,
        "pmax_map": pmax_map,
        "stgt_map": stgt_map,
        "state_map": state_map,
        "hist_charge_map": hist_charge_map,
    }


# -----------------------------
# 评估与出图
# -----------------------------
def evaluate_and_plot(base_dir: Path, eval_hours: int, save_plots: bool = True):
    # 载入策略/标准化/奖励权重/限值
    pi, feat_names, mean, std, rw, pcc_limit = load_policy(base_dir)
    meta = {"feature_names": feat_names, "mean": mean, "std": std}
    ed = build_eval_window(base_dir, hours=eval_hours)

    ts_list = ed["ts_list"]
    veh_ids = ed["veh_ids"]

    artifacts = base_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # 逐时刻计算“预测充电功率”
    pred_charge = []  # 每个 ts 的合计预测充电功率（kW）
    hist_charge = []  # 历史充电功率合计（kW）
    pcc_baseline = []  # 原始 PCC
    pcc_adjusted = []  # 新策略假设下的 PCC（pcc - hist + pred）

    # 成本/奖励分解（每步）
    step_energy_cost = []
    step_carbon_cost = []
    step_peak_penalty = []
    step_smooth_penalty = []
    step_reward = []

    # 记录每车上一时刻 a_ratio 以计算平滑罚（Δa^2）
    prev_a = {vid: 0.0 for vid in veh_ids}

    # 特征顺序按 meta["feature_names"] 构造
    fn = feat_names
    # 名称到索引（便于 sanity check）
    name_idx = {n: i for i, n in enumerate(fn)}

    for ts in ts_list:
        price = float(ed["price_map"].get(ts, 0.0))
        ef = float(ed["ef_map"].get(ts, 0.0))
        pcc = float(ed["pcc_map"].get(ts, 0.0))
        hist = float(ed["hist_charge_map"].get(ts, 0.0))
        grid_room = float(pcc_limit - pcc)
        hsin, hcos, dsin, dcos = time_feats(ts)

        # 逐车构造特征 -> 策略 -> 比例 -> 功率
        total_pred_kw = 0.0
        total_smooth = 0.0
        E_kwh_sum = 0.0  # 用电量（kWh）汇总

        for vid in veh_ids:
            batt = float(ed["batt_map"].get(vid, 0.0))
            pmax = float(ed["pmax_map"].get(vid, 0.0))
            stgt = float(ed["stgt_map"].get(vid, 0.8))
            soc, av_i, pr_i, eta_f, tmp_f = ed["state_map"].get((ts, vid), (0.6, 1, 0, 30.0, 25.0))
            c_rate_cap = 0.0 if batt <= 0 else min(1.2, max(0.0, pmax / batt))

            # 与训练一致的特征顺序（缺项用0）
            feat_vec = np.zeros(len(fn), dtype=np.float32)
            def setv(name: str, val: float):
                if name in name_idx:
                    feat_vec[name_idx[name]] = float(val)

            setv("price_yuan_per_kwh", price)
            setv("ef_kg_per_kwh", ef)
            setv("grid_room_kw", grid_room)
            setv("time_hsin", hsin)
            setv("time_hcos", hcos)
            setv("dow_sin", dsin)
            setv("dow_cos", dcos)
            setv("soc", soc)
            setv("soc_target", stgt)
            setv("battery_kwh", batt)
            setv("p_charge_max_kw", pmax)
            setv("available", av_i)
            setv("priority", pr_i)
            setv("eta_min", eta_f)
            setv("temp", tmp_f)
            setv("c_rate_cap", c_rate_cap)

            # 标准化 -> 策略前向
            x = (feat_vec - mean) / (std + 1e-8)
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            a_ratio = float(pi(x_t).detach().cpu().numpy().reshape(-1)[0])
            # 约束：不可用时=0
            if av_i <= 0:
                a_ratio = 0.0
            a_ratio = max(0.0, min(1.0, a_ratio))
            p_kw = a_ratio * max(0.0, pmax)

            total_pred_kw += p_kw
            E_kwh_sum += p_kw * (DT_MIN / 60.0)
            total_smooth += (a_ratio - prev_a[vid]) ** 2
            prev_a[vid] = a_ratio

        # 假设新策略后 PCC
        pcc_adj = pcc - hist + total_pred_kw

        # 成本/奖励分解（步级聚合）
        price_cost = rw["w_price"] * E_kwh_sum * price
        carbon_cost = rw["w_ef"] * E_kwh_sum * ef
        peak_pen = rw["lam_peak"] * max(0.0, pcc_adj - pcc_limit)
        smooth_pen = rw["lam_smooth"] * total_smooth
        reward = -(price_cost + carbon_cost + peak_pen + smooth_pen)

        pred_charge.append(total_pred_kw)
        hist_charge.append(hist)
        pcc_baseline.append(pcc)
        pcc_adjusted.append(pcc_adj)

        step_energy_cost.append(price_cost)
        step_carbon_cost.append(carbon_cost)
        step_peak_penalty.append(peak_pen)
        step_smooth_penalty.append(smooth_pen)
        step_reward.append(reward)

    pred_charge = np.asarray(pred_charge, dtype=np.float32)
    hist_charge = np.asarray(hist_charge, dtype=np.float32)
    pcc_baseline = np.asarray(pcc_baseline, dtype=np.float32)
    pcc_adjusted = np.asarray(pcc_adjusted, dtype=np.float32)

    step_energy_cost = np.asarray(step_energy_cost, dtype=np.float32)
    step_carbon_cost = np.asarray(step_carbon_cost, dtype=np.float32)
    step_peak_penalty = np.asarray(step_peak_penalty, dtype=np.float32)
    step_smooth_penalty = np.asarray(step_smooth_penalty, dtype=np.float32)
    step_reward = np.asarray(step_reward, dtype=np.float32)

    # KPI 计算
    peak_before = float(np.max(pcc_baseline)) if pcc_baseline.size else 0.0
    peak_after = float(np.max(pcc_adjusted)) if pcc_adjusted.size else 0.0
    peak_reduction_kw = float(peak_before - peak_after)
    peak_reduction = (peak_reduction_kw / peak_before) * 100.0 if peak_before > 1e-6 else 0.0

    # 能源与碳成本（策略 vs 历史）
    # 历史成本：按 hist_charge 估算
    # 价格/EF 序列
    price_seq = np.array([float(ed["price_map"].get(ts, 0.0)) for ts in ts_list], dtype=np.float32)
    ef_seq = np.array([float(ed["ef_map"].get(ts, 0.0)) for ts in ts_list], dtype=np.float32)
    e_hist = hist_charge * (DT_MIN / 60.0)
    e_pred = pred_charge * (DT_MIN / 60.0)

    energy_cost_hist = float(np.sum(rw["w_price"] * e_hist * price_seq))
    energy_cost_pred = float(np.sum(rw["w_price"] * e_pred * price_seq))
    carbon_cost_hist = float(np.sum(rw["w_ef"] * e_hist * ef_seq))
    carbon_cost_pred = float(np.sum(rw["w_ef"] * e_pred * ef_seq))

    report = {
        "window_hours": eval_hours,
        "samples": len(ts_list),
        "pcc_limit_kw": float(pcc_limit),
        "peak_before_kw": peak_before,
        "peak_after_kw": peak_after,
        "peak_reduction_kw": peak_reduction_kw,
        "peak_reduction_pct": peak_reduction,
        "energy_cost_hist_yuan": energy_cost_hist,
        "energy_cost_pred_yuan": energy_cost_pred,
        "carbon_cost_hist_equiv": carbon_cost_hist,
        "carbon_cost_pred_equiv": carbon_cost_pred,
        "reward_sum": float(np.sum(step_reward)),
        "reward_components_sum": {
            "price_cost": float(np.sum(step_energy_cost)),
            "carbon_cost": float(np.sum(step_carbon_cost)),
            "peak_penalty": float(np.sum(step_peak_penalty)),
            "smooth_penalty": float(np.sum(step_smooth_penalty)),
        },
    }
    (artifacts / "iql_eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 出图 ----------
    # 1) 充电功率 + PCC
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(range(len(ts_list)), hist_charge, label="历史充电功率(kW)")
    ax1.plot(range(len(ts_list)), pred_charge, label="策略-预测充电功率(kW)")
    ax1.set_xlabel("时间步（5分钟）")
    ax1.set_ylabel("充电功率 (kW)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(range(len(ts_list)), pcc_baseline, label="PCC基线(kW)", linestyle="--")
    ax2.plot(range(len(ts_list)), pcc_adjusted, label="PCC(策略假设)(kW)", linestyle=":")
    ax2.set_ylabel("PCC 功率 (kW)")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(artifacts / "rollout_loads.png", dpi=150)
    plt.close(fig)

    # 2) 价格/碳因子与充电功率
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(range(len(ts_list)), pred_charge, label="策略-预测充电功率(kW)")
    ax1.plot(range(len(ts_list)), hist_charge, label="历史充电功率(kW)", linestyle="--")
    ax1.set_xlabel("时间步（5分钟）")
    ax1.set_ylabel("充电功率 (kW)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(range(len(ts_list)), price_seq, label="电价(元/kWh)")
    ax2.plot(range(len(ts_list)), ef_seq, label="碳因子(kg/kWh)", linestyle=":")
    ax2.set_ylabel("电价 / 碳因子")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(artifacts / "rollout_price_ef.png", dpi=150)
    plt.close(fig)

    # 3) 成本/奖励分解（时间序列）
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(step_energy_cost, label="电费(步)")
    ax.plot(step_carbon_cost, label="碳费(步)")
    ax.plot(step_peak_penalty, label="峰值罚金(步)")
    ax.plot(step_smooth_penalty, label="平滑罚(步)")
    ax.plot(step_reward, label="奖励 r_t")
    ax.set_xlabel("时间步（5分钟）")
    ax.set_ylabel("成本 / 奖励（单位：权重后）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifacts / "reward_costs.png", dpi=150)
    plt.close(fig)

    print(f"[OK] 报告: {artifacts/'iql_eval_report.json'}")
    print(f"[OK] 图像: {artifacts/'rollout_loads.png'}")
    print(f"[OK] 图像: {artifacts/'rollout_price_ef.png'}")
    print(f"[OK] 图像: {artifacts/'reward_costs.png'}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser("AGV Charge Module - policy evaluation & plotting")
    ap.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--eval-hours", type=int, default=6, help="评估的窗口时长（小时）")
    ap.add_argument("--self-check", action="store_true", help="只做评估与出图，不涉及训练")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    if args.self_check:
        evaluate_and_plot(base_dir, eval_hours=args.eval_hours, save_plots=True)
    else:
        # 默认行为也做一次评估&出图（对接前端按钮时可替换为其他入口）
        evaluate_and_plot(base_dir, eval_hours=args.eval_hours, save_plots=True)


if __name__ == "__main__":
    main()
