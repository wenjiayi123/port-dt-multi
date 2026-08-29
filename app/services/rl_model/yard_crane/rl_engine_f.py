# app/services/rl_model/yard_crane/rl_engine_f.py
# -*- coding: utf-8 -*-
"""
F 模块｜场桥（RTG/RMG）训练与执行引擎
====================================
大白话：
- 这是场桥模块（F）的训练/执行主程序。仅用标准库 + numpy（不依赖深度库）。
- 离线阶段：支持 **TD3+BC**（默认）或 **IQL** 两种脱敏预训练；数据由 module_f.prepare_offline_dataset() + 安全噪声扩增获得；
- 在线阶段：**Safe‑SAC（拉格朗日约束）** 残差微调，带 **CVaR/QR 风险** 与 **KL 信任域**，动作经 **module_f.YardCraneEnv** 的硬屏蔽后下发；
- 训练/评估输出：统一 JSONL（与前端/审计一致），并保存策略参数到 policy.bin / policy_meta.json。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .module_f import (
    make_env, rollout_and_log, prepare_offline_dataset, baseline_policy_fn,
    YardCraneEnv, DEFAULT_JSONL, STATIC_JSONL
)
# 站端口径（电力/DR/并网）来源文件：demand_window_config.json / dr_events.json / bess_master.json
# 字段含义与路径同名对齐，真实落地仅需替换同名文件。  :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

# -------------------------
# JSONL Logger
# -------------------------
class JsonlLogger:
    def __init__(self, path: str = DEFAULT_JSONL):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, d: Dict[str, Any]):
        self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

def _mirror_to_static(src: str, dst: str = STATIC_JSONL):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
            for line in fsrc:
                fdst.write(line)
    except Exception:
        pass

# -------------------------
# 特征工程（线性可解释）
# -------------------------
class FeatureMaker:
    """
    把 env.obs(dict) 转为 φ(s)：
    - 以参考轨迹与未来分位、历史残差摘要、队列/价格/EF/温度等构成；
    - 功率/待机残差分别以残差带归一化（防尺度漂移）。
    """
    def __init__(self, res_band_power: float = 0.10, res_band_idle_min: float = 3.0):
        self.res_band_power = max(1e-6, float(res_band_power))  # 相对百分比（0~1）
        self.res_band_idle = max(1e-6, float(res_band_idle_min))  # 绝对分钟

    def obs_to_phi(self, obs: Dict[str, Any]) -> np.ndarray:
        # 数值稳健：缺失→0 / nan→0
        def nz(x, dv=0.0):
            try:
                if x is None: return dv
                xx = float(x)
                if math.isnan(xx): return dv
                return xx
            except Exception:
                return dv
        # 模式 one-hot
        mode_ref = str(obs.get("mode_ref", "normal")).lower()
        mode_onehot = [1.0 if m == mode_ref else 0.0 for m in ["normal", "ecol1", "ecol2", "ecol3"]]
        hist = obs.get("hist_actions", [])
        if not hist: hist = [{"pwr_pct": obs.get("pwr_ref_pct", 1.0), "idle_min": obs.get("idle_ref_min", 8.0)}]
        ha_last = hist[-1] if hist else {"pwr_pct": 1.0, "idle_min": 8.0}
        ha_mean_p = float(np.mean([nz(h.get("pwr_pct"), 1.0) for h in hist]))
        ha_mean_i = float(np.mean([nz(h.get("idle_min"), 8.0) for h in hist]))

        phi = np.array([
            1.0,
            nz(obs.get("pwr_ref_pct"), 1.0), nz(obs.get("idle_ref_min"), 8.0),
            nz(obs.get("queue_p50"), 1.0),
            nz(obs.get("price"), 0.6), nz(obs.get("ef"), 0.65),
            nz(obs.get("tmotor"), 70.0), nz(obs.get("tinv"), 70.0),
            nz(obs.get("pcc_kw"), 10000.0),
            nz(obs.get("fut_price_p50"), 0.6), nz(obs.get("fut_price_p90"), 0.8),
            nz(obs.get("fut_ef_p50"), 0.65), nz(obs.get("fut_ef_p90"), 0.7),
            nz(ha_last.get("pwr_pct"), 1.0), nz(ha_last.get("idle_min"), 8.0),
            ha_mean_p, ha_mean_i
        ] + mode_onehot, dtype=np.float64)
        return phi

    def sa_features(self, obs: Dict[str, Any], a: np.ndarray) -> np.ndarray:
        # a = [d_power_pct, d_idle_min]
        phi_s = self.obs_to_phi(obs)
        # 归一化：功率残差按比例、待机残差按分钟带
        a0 = float(a[0]) / max(1e-6, self.res_band_power)
        a1 = float(a[1]) / max(1e-6, self.res_band_idle)
        return np.concatenate([phi_s, np.array([a0, a1], dtype=np.float64)], axis=0)

# -------------------------
# 策略与价值近似器（线性）
# -------------------------
class GaussianPolicy:
    """
    连续残差策略：π(a|s)=N(μ(s), diag(σ^2))，a=[dP_pct, dIdle_min]
    μ(s)=W·φ(s)+b，经 tanh 映射到残差带；σ 设最小下限 + 噪声日历，防“夹边重复轨迹”。
    """
    def __init__(self, feat_dim: int, res_band_power: float, res_band_idle: float, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.D = feat_dim
        self.W = self.rng.normal(scale=0.01, size=(2, feat_dim))
        self.b = np.zeros(2, dtype=np.float64)
        self.log_std = np.log(np.array([0.15, 0.10], dtype=np.float64))
        self.min_std_ratio = 0.08
        self.noise_calendar = [1.0, 1.3, 0.9, 1.5, 1.0, 0.8]
        self.res_band_power = float(res_band_power)
        self.res_band_idle = float(res_band_idle)

    def forward_mean_raw(self, phi: np.ndarray) -> np.ndarray:
        return self.W.dot(phi) + self.b  # 未tanh

    def _scale(self, x: np.ndarray) -> np.ndarray:
        # tanh 映射 + 残差带缩放
        y = np.tanh(x)
        return np.array([y[0]*self.res_band_power, y[1]*self.res_band_idle], dtype=np.float64)

    def sample(self, phi: np.ndarray, step_idx: int, kl_boost: float = 1.0, force_min_std: bool = True) -> Tuple[np.ndarray, float]:
        mu = self.forward_mean_raw(phi)
        std = np.exp(self.log_std)
        if force_min_std: std = np.maximum(std, self.min_std_ratio)
        std = std * self.noise_calendar[step_idx % len(self.noise_calendar)] * kl_boost
        eps = self.rng.normal(size=2)
        a_raw = mu + std * eps
        a = self._scale(a_raw)
        # 近似 log_prob（仅用于相对 PG 指标，不用于精确概率）
        logp = -0.5 * float(np.sum(((a_raw-mu)/std)**2 + 2*np.log(std) + np.log(2*np.pi)))
        return a, logp

    def mean_action(self, phi: np.ndarray) -> np.ndarray:
        return self._scale(self.forward_mean_raw(phi))

    def kl_with(self, other: "GaussianPolicy", phi_batch: np.ndarray) -> float:
        mu1 = (self.W.dot(phi_batch.T).T + self.b)
        mu0 = (other.W.dot(phi_batch.T).T + other.b)
        std1 = np.maximum(np.exp(self.log_std), self.min_std_ratio)
        std0 = np.maximum(np.exp(other.log_std), other.min_std_ratio)
        term1 = np.sum((std0**2 + (mu0-mu1)**2)/(std1**2), axis=1)
        kl = 0.5 * np.mean(term1 - 2 + 2*(np.log(std1)-np.log(std0)).sum())
        return float(max(0.0, kl))

    def copy(self) -> "GaussianPolicy":
        g = GaussianPolicy(self.D, self.res_band_power, self.res_band_idle)
        g.W = self.W.copy(); g.b = self.b.copy(); g.log_std = self.log_std.copy()
        g.min_std_ratio = self.min_std_ratio; g.noise_calendar = list(self.noise_calendar)
        return g

# IQL 轻量组件（与 E 模块思路一致）
class IQLAgent:
    def __init__(self, feat_dim: int, fm: FeatureMaker, tau: float = 0.7, adv_temp: float = 0.5, lr: float = 3e-4, seed: int = 42):
        self.fm = fm
        self.tau = float(tau)
        self.adv_temp = float(adv_temp)
        self.lr = float(lr)
        self.rng = np.random.RandomState(seed)
        self.v = np.zeros(feat_dim, dtype=np.float64)
        self.policy = GaussianPolicy(feat_dim, fm.res_band_power, fm.res_band_idle, seed=seed)

    def v_value(self, phi: np.ndarray) -> float:
        return float(self.v.dot(phi))

    def v_update(self, phi: np.ndarray, y: float):
        v = self.v_value(phi); delta = y - v
        w = self.tau if delta > 0 else (1.0 - self.tau)
        grad = -2.0 * w * delta * phi
        self.v -= self.lr * grad

    def policy_update_awr(self, phi: np.ndarray, a: np.ndarray, y: float):
        v = self.v_value(phi)
        adv = y - v
        w = math.exp(max(-10.0, min(10.0, adv / max(1e-6, self.adv_temp))))
        mu = self.policy.forward_mean_raw(phi)
        std = np.maximum(np.exp(self.policy.log_std), self.policy.min_std_ratio)
        # 反向映射：把实值动作 a 映射回“无带”域用于梯度近似
        a_raw = np.array([
            np.arctanh(np.clip(a[0]/max(1e-6,self.policy.res_band_power), -0.999, 0.999)),
            np.arctanh(np.clip(a[1]/max(1e-6,self.policy.res_band_idle), -0.999, 0.999)),
        ], dtype=np.float64)
        grad_mu = (a_raw - mu) / (std**2)  # 近似 ∇logπ
        self.policy.W += self.lr * np.outer(grad_mu, phi) * w
        self.policy.b += self.lr * grad_mu * w

# TD3+BC（线性近似版）
class TD3BC:
    """
    线性 TD3+BC：
    - Q1/Q2(s,a)=w·[φ(s), a_norm]；目标网络软更新；
    - 策略为确定性线性 μ(s)（经 tanh + 残差带缩放）；
    - actor 损失：-E[Q1(s,π(s))] + α_bc * ||π(s)-a_bc||^2；
    - 数据：来自基线 + 安全噪声扩增的离线数据集。
    """
    def __init__(self, feat_dim: int, fm: FeatureMaker, lr_q: float = 3e-4, lr_actor: float = 3e-4, tau: float = 0.005, bc_alpha: float = 2.5, seed: int = 42):
        self.fm = fm
        self.rng = np.random.RandomState(seed)
        self.Ds = feat_dim
        self.Dsa = feat_dim + 2
        # Q 网络
        self.Q1 = self.rng.normal(scale=0.01, size=(self.Dsa,))
        self.Q2 = self.rng.normal(scale=0.01, size=(self.Dsa,))
        self.Q1_t = self.Q1.copy()
        self.Q2_t = self.Q2.copy()
        # 策略（确定性）
        self.pi = GaussianPolicy(feat_dim, fm.res_band_power, fm.res_band_idle, seed=seed)  # 复用高斯结构，仅用 mean_action
        self.lr_q = lr_q; self.lr_actor = lr_actor; self.tau = tau; self.bc_alpha = bc_alpha
        self.gamma = 0.995

    def q_value(self, w: np.ndarray, phi_sa: np.ndarray) -> float:
        return float(np.dot(w, phi_sa))

    def _target_q(self, phi_s_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 目标动作（加入小噪声）
        a_next = self.pi.mean_action(phi_s_next)
        an0 = a_next[0]/max(1e-6,self.fm.res_band_power)
        an1 = a_next[1]/max(1e-6,self.fm.res_band_idle)
        return np.array([an0, an1], dtype=np.float64), a_next

    def train_step(self, batch: List[Dict[str, Any]], policy_delay: int, step_idx: int) -> Dict[str, float]:
        # 更新 Q
        loss_q = 0.0
        for it, item in enumerate(batch):
            obs = item["obs"]; next_obs = item["next_obs"]; r = float(item["reward"])
            done = bool(item.get("done", False))
            a_bc = np.array([float(item["action"].get("d_power_pct", 0.0)),
                             float(item["action"].get("d_idle_min", 0.0))], dtype=np.float64)
            phi_s = self.fm.obs_to_phi(obs)
            phi_sn = self.fm.obs_to_phi(next_obs) if next_obs else phi_s
            # 目标
            a_next_norm, a_next = self._target_q(phi_sn)
            phi_sa_next = np.concatenate([phi_sn, a_next_norm], axis=0)
            q1_t = self.q_value(self.Q1_t, phi_sa_next)
            q2_t = self.q_value(self.Q2_t, phi_sa_next)
            y = r + (0.0 if done else self.gamma * min(q1_t, q2_t))

            # 当前
            a_norm = np.array([a_bc[0]/max(1e-6,self.fm.res_band_power), a_bc[1]/max(1e-6,self.fm.res_band_idle)], dtype=np.float64)
            phi_sa = np.concatenate([phi_s, a_norm], axis=0)
            # L2 损失梯度
            err1 = y - self.q_value(self.Q1, phi_sa)
            err2 = y - self.q_value(self.Q2, phi_sa)
            self.Q1 += self.lr_q * err1 * phi_sa
            self.Q2 += self.lr_q * err2 * phi_sa
            loss_q += 0.5*(err1**2 + err2**2)

            # actor 延迟更新
            if (it + step_idx) % max(2, policy_delay) == 0:
                a_pi = self.pi.mean_action(phi_s)
                an0 = a_pi[0]/max(1e-6,self.fm.res_band_power)
                an1 = a_pi[1]/max(1e-6,self.fm.res_band_idle)
                phi_sa_pi = np.concatenate([phi_s, np.array([an0, an1])], axis=0)
                q1_pi = self.q_value(self.Q1, phi_sa_pi)
                # TD3 部分：最大化 Q1 → 最小化 -Q1；BC 正则：靠近 a_bc
                bc = ( (a_pi[0]-a_bc[0])**2 + (a_pi[1]-a_bc[1])**2 )
                # 近似对 μ 的梯度：对均值 raw 做一阶近似（线性层 → 直接按残差方向更新）
                grad_mu = np.array([
                    -self.lr_actor * (q1_pi) * 0.5,  # 缩放系数仅做稳定性，不影响方向性
                    -self.lr_actor * (q1_pi) * 0.5
                ]) + self.lr_actor * self.bc_alpha * np.array([
                    (a_bc[0]-a_pi[0]),
                    (a_bc[1]-a_pi[1])
                ])
                # 把“对动作的梯度”投影回 W,b（线性层）
                mu_raw = self.pi.forward_mean_raw(phi_s)  # 2
                # 用 ∂μ/∂θ = φ(s)
                self.pi.W += np.outer(grad_mu, phi_s)
                self.pi.b += grad_mu

                # 软更新目标网络
                self.Q1_t = (1-self.tau)*self.Q1_t + self.tau*self.Q1
                self.Q2_t = (1-self.tau)*self.Q2_t + self.tau*self.Q2

        return {"loss_q": float(loss_q/ max(1,len(batch)))}

# Quantile Critic for CVaR/QR
class QuantileCritic:
    def __init__(self, feat_dim_sa: int, taus: List[float], lr: float = 3e-4, gamma: float = 0.995):
        self.taus = np.array(taus, dtype=np.float64)
        self.W = np.zeros((len(taus), feat_dim_sa), dtype=np.float64)
        self.lr = float(lr); self.gamma = float(gamma)

    def q_values(self, phi_sa: np.ndarray) -> np.ndarray:
        return self.W.dot(phi_sa)

    def td_update(self, phi_sa: np.ndarray, td_target: float):
        y = self.q_values(phi_sa)
        for k, tau in enumerate(self.taus):
            e = td_target - y[k]
            g = (tau - (1.0 if e < 0 else 0.0)) * phi_sa
            self.W[k, :] += self.lr * g

    def cvar(self, phi_sa: np.ndarray, alpha: float = 0.1) -> float:
        qs = self.q_values(phi_sa)
        mask = self.taus <= alpha
        return float(np.mean(qs[mask])) if np.any(mask) else float(np.min(qs))

# Safe-SAC 在线微调器
class SafeSACTrainer:
    def __init__(self, fm: FeatureMaker, policy: GaussianPolicy, critic: QuantileCritic, kl_max: float = 0.05, alpha_ent: float = 0.2, lr_actor: float = 3e-4, lr_lambda: float = 1e-3):
        self.fm = fm; self.policy = policy; self.critic = critic
        self.kl_max = float(kl_max); self.alpha_ent = float(alpha_ent); self.lr_actor = float(lr_actor)
        self.lmb = {"mask_rate": 1.0, "sla_pen": 0.0, "thermal": 0.0}
        self.targets = {"mask_rate": 0.12, "sla_pen": 0.0, "thermal": 0.0}
        self.lr_lambda = float(lr_lambda)

    def actor_step(self, batch: List[Dict[str, Any]], old_policy: GaussianPolicy) -> Dict[str, float]:
        grads_W = np.zeros_like(self.policy.W); grads_b = np.zeros_like(self.policy.b)
        phi_batch = []
        for item in batch:
            obs = item["obs"]; a = np.array([item["action"].get("d_power_pct",0.0), item["action"].get("d_idle_min",0.0)], dtype=np.float64)
            phi = self.fm.obs_to_phi(obs); phi_sa = self.fm.sa_features(obs, a)
            q_cvar = self.critic.cvar(phi_sa, alpha=0.1)
            g = item.get("g_costs", {"mask":0.0, "sla":0.0, "thermal":0.0})
            penalty = self.lmb["mask_rate"]*g.get("mask",0.0) + self.lmb["sla_pen"]*g.get("sla",0.0) + self.lmb["thermal"]*g.get("thermal",0.0)
            weight = q_cvar - penalty
            mu = self.policy.forward_mean_raw(phi)
            std = np.maximum(np.exp(self.policy.log_std), self.policy.min_std_ratio)
            # 反映射
            a_raw = np.array([
                np.arctanh(np.clip(a[0]/max(1e-6,self.policy.res_band_power), -0.999, 0.999)),
                np.arctanh(np.clip(a[1]/max(1e-6,self.policy.res_band_idle), -0.999, 0.999))
            ], dtype=np.float64)
            grad_mu = (a_raw - mu)/(std**2)
            grads_W += np.outer(grad_mu, phi) * weight
            grads_b += grad_mu * weight
            phi_batch.append(phi)

        step_scale = self.lr_actor / max(1,len(batch))
        # KL trust-region
        pol_tmp = self.policy.copy()
        pol_tmp.W = self.policy.W + step_scale*grads_W
        pol_tmp.b = self.policy.b + step_scale*grads_b
        kl = pol_tmp.kl_with(old_policy, np.stack(phi_batch, axis=0))
        if kl > self.kl_max:
            step_scale *= max(0.05, self.kl_max / max(1e-6, kl))
        self.policy.W += step_scale*grads_W
        self.policy.b += step_scale*grads_b
        return {"kl": float(kl), "step_scale": float(step_scale)}

    def lambda_step(self, roll_metrics: Dict[str, float]):
        gap_mask = roll_metrics.get("mask_rate",0.0) - self.targets["mask_rate"]
        gap_sla = roll_metrics.get("sla_penalty",0.0) - self.targets["sla_pen"]
        gap_th = roll_metrics.get("thermal_penalty",0.0) - self.targets["thermal"]
        self.lmb["mask_rate"] = max(0.0, self.lmb["mask_rate"] + self.lr_lambda * gap_mask)
        self.lmb["sla_pen"]   = max(0.0, self.lmb["sla_pen"]   + self.lr_lambda * gap_sla)
        self.lmb["thermal"]   = max(0.0, self.lmb["thermal"]   + self.lr_lambda * gap_th)
        return self.lmb.copy()

# -------------------------
# 训练主控
# -------------------------
class Trainer:
    def __init__(self, dt_min: int = 5, horizon: int = 144, seed: int = 42):
        self.dt_min = dt_min; self.horizon = horizon; self.seed = seed
        random.seed(seed); np.random.seed(seed)
        self.env, self.planner, self.ctx = make_env(dt_min=dt_min, horizon_steps=horizon, jsonl_path=DEFAULT_JSONL)
        # 残差带（从 env 经验参数读取）
        self.fm = FeatureMaker(res_band_power=0.10, res_band_idle_min=3.0)
        # IQL 组件
        feat_dim = len(self.fm.obs_to_phi(self.env.reset(0)))
        self.iql = IQLAgent(feat_dim, self.fm, tau=0.7, adv_temp=0.5, lr=3e-4, seed=seed)
        # TD3+BC 组件
        self.td3bc = TD3BC(feat_dim, self.fm, lr_q=3e-4, lr_actor=3e-4, tau=0.005, bc_alpha=2.5, seed=seed)
        # Safe-SAC 组件
        feat_sa_dim = feat_dim + 2
        self.critic = QuantileCritic(feat_sa_dim, taus=[0.05,0.25,0.5,0.75,0.95], lr=3e-4, gamma=0.995)
        self.sac   = SafeSACTrainer(self.fm, self.iql.policy, self.critic, kl_max=0.05, alpha_ent=0.2, lr_actor=3e-4)

        self.log = JsonlLogger(DEFAULT_JSONL)

    # ---------- 数据集 ----------
    def _dataset_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "offline_dataset_crane.jsonl")

    def _dataset_path_aug(self) -> str:
        return os.path.join(os.path.dirname(__file__), "offline_dataset_crane_aug.jsonl")

    def build_or_load_offline(self, augment_noise: bool = True, noise_scale: float = 0.5) -> str:
        path = self._dataset_path()
        if not os.path.exists(path):
            # 基线数据（零残差）
            prepare_offline_dataset(*make_env(dt_min=self.dt_min, horizon_steps=self.horizon, jsonl_path=os.path.join(os.path.dirname(__file__), "_tmp.jsonl"))[:1], path)
        if augment_noise:
            # 生成“安全噪声”增强数据（动作在残差带内，屏蔽后落地）
            env2, _, _ = make_env(dt_min=self.dt_min, horizon_steps=self.horizon, jsonl_path=os.path.join(os.path.dirname(__file__), "_tmp_aug.jsonl"))
            aug = self._dataset_path_aug()
            with open(aug, "w", encoding="utf-8") as f:
                obs = env2.reset(0)
                for t in range(self.horizon):
                    # 在残差带内随机采样，保证可行性由环境屏蔽
                    dP = float(np.clip(np.random.normal(scale=noise_scale*0.10), -0.10, 0.10))
                    dI = float(np.clip(np.random.normal(scale=noise_scale*3.0),  -3.0,   3.0))
                    act = {"d_power_pct": dP, "d_idle_min": dI, "mode": ""}
                    next_obs, r, done, info = env2.step(act)
                    f.write(json.dumps({"key":"transition", "obs":obs, "action":act, "reward":float(r), "next_obs":next_obs, "done":bool(done)}, ensure_ascii=False)+"\n")
                    obs = next_obs
                    if done: break
        return path

    def _iter_jsonl(self, paths: List[str], batch: int):
        buf = []
        for p in paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    buf += f.readlines()
        idxs = np.arange(len(buf)); np.random.shuffle(idxs)
        cur = 0
        while cur < len(buf):
            batch_items = []
            for _ in range(batch):
                if cur >= len(buf): break
                try:
                    item = json.loads(buf[idxs[cur]])
                except Exception:
                    cur += 1; continue
                if item.get("key") != "transition":
                    cur += 1; continue
                batch_items.append(item); cur += 1
            if batch_items: yield batch_items

    # ---------- IQL 预训练 ----------
    def train_offline_iql(self, steps: int = 30000, batch: int = 512, tau: float = 0.7, lr: float = 3e-4, log_every: int = 200, sleep_every: int = 1000, sleep_sec: int = 60) -> Dict[str, Any]:
        ds_base = self.build_or_load_offline(augment_noise=True, noise_scale=0.5)
        ds_aug  = self._dataset_path_aug()
        paths = [ds_base] + ([ds_aug] if os.path.exists(ds_aug) else [])
        self.iql.tau = tau; self.iql.lr = lr
        it = 0
        for epoch in range(max(1, steps//max(1,batch))+1):
            for batch_items in self._iter_jsonl(paths, batch):
                it += 1
                for item in batch_items:
                    obs = item["obs"]; next_obs = item["next_obs"]; r = float(item["reward"])
                    if not next_obs: continue
                    phi = self.fm.obs_to_phi(obs); phi_next = self.fm.obs_to_phi(next_obs)
                    y = r + 0.995*self.iql.v_value(phi_next)
                    self.iql.v_update(phi, y)
                    a = np.array([float(item["action"].get("d_power_pct",0.0)), float(item["action"].get("d_idle_min",0.0))], dtype=np.float64)
                    self.iql.policy_update_awr(phi, a, y)
                if it % log_every == 0:
                    self.log.write({"key":"rl_train","stage":"iql_offline","iter":int(it),"tau":float(self.iql.tau),"lr":float(self.iql.lr)})
                if it % max(1,sleep_every) == 0: time.sleep(max(0, sleep_sec))
                if it >= steps: break
            if it >= steps: break
        policy_path = self._save_policy(stage="iql_offline")
        return {"dataset": paths, "policy_path": policy_path}

    # ---------- TD3+BC 预训练 ----------
    def train_offline_td3bc(self, steps: int = 30000, batch: int = 512, policy_delay: int = 2, log_every: int = 200, sleep_every: int = 1000, sleep_sec: int = 60) -> Dict[str, Any]:
        ds_base = self.build_or_load_offline(augment_noise=True, noise_scale=0.8)
        ds_aug  = self._dataset_path_aug()
        paths = [ds_base] + ([ds_aug] if os.path.exists(ds_aug) else [])
        it = 0
        for epoch in range(max(1, steps//max(1,batch))+1):
            for batch_items in self._iter_jsonl(paths, batch):
                it += 1
                stats = self.td3bc.train_step(batch_items, policy_delay=policy_delay, step_idx=it)
                if it % log_every == 0:
                    self.log.write({"key":"rl_train","stage":"td3bc_offline","iter":int(it),"loss_q":float(stats["loss_q"])})
                if it % max(1,sleep_every) == 0: time.sleep(max(0, sleep_sec))
                if it >= steps: break
            if it >= steps: break
        self.iql.policy.W = self.td3bc.pi.W.copy()
        self.iql.policy.b = self.td3bc.pi.b.copy()
        self.iql.policy.log_std = self.td3bc.pi.log_std.copy()
        policy_path = self._save_policy(stage="td3bc_offline")
        return {"dataset": paths, "policy_path": policy_path}

    # ---------- 在线 Safe-SAC 微调 ----------
    def online_finetune(self, algo_from: str = "iql", env_steps: int = 2000, batch_size: int = 64, log_every: int = 100, sleep_every: int = 1000, sleep_sec: int = 60) -> Dict[str, Any]:
        # 选择策略来源
        if algo_from.lower().startswith("td3"):
            # 把 TD3 策略权重拷贝给高斯策略的均值（协方差保持）
            self.iql.policy.W = self.td3bc.pi.W.copy()
            self.iql.policy.b = self.td3bc.pi.b.copy()

        replay: List[Dict[str, Any]] = []
        old_policy = self.iql.policy.copy()
        obs = self.env.reset(0)
        mask_cnt = 0
        for t in range(env_steps):
            phi = self.fm.obs_to_phi(obs)
            a_vec, logp = self.iql.policy.sample(phi, step_idx=t, kl_boost=1.0, force_min_std=True)
            act = {"d_power_pct": float(a_vec[0]), "d_idle_min": float(a_vec[1]), "mode": ""}
            next_obs, r, done, info = self.env.step(act)
            g_costs = {
                "mask": float(info.get("masked",0)),
                "sla":  float(max(0.0, self.env.metrics.get("sla_viol", 0.0))),
                "thermal": float(max(0.0, self.env.metrics.get("thermal_pen", 0.0)))
            }
            replay.append({"obs":obs, "action":act, "reward":float(r), "next_obs":next_obs, "done":bool(done), "g_costs":g_costs})
            mask_cnt += int(info.get("masked",0))
            # critic update
            phi_sa = self.fm.sa_features(obs, np.array([act["d_power_pct"], act["d_idle_min"]], dtype=np.float64))
            y = float(r);
            if not done and next_obs: y += 0.995 * self.iql.v_value(self.fm.obs_to_phi(next_obs))
            self.critic.td_update(phi_sa, td_target=y)
            # actor step
            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                stats = self.sac.actor_step(batch, old_policy=old_policy)
                if stats["kl"] <= self.sac.kl_max:
                    old_policy = self.iql.policy.copy()
            # log & λ update
            if (t+1) % log_every == 0:
                roll = {
                    "mask_rate": mask_cnt/max(1,(t+1)),
                    "sla_penalty": float(self.env.metrics.get("sla_viol",0.0)/max(1,(t+1))),
                    "thermal_penalty": float(self.env.metrics.get("thermal_pen",0.0)/max(1,(t+1)))
                }
                lambdas = self.sac.lambda_step(roll)
                self._log({"key":"policy_update","stage":"safe_sac_online","step":int(t+1),"roll_metrics":roll,"lambdas":lambdas,"kl_last":stats.get("kl",0.0) if 'stats' in locals() else 0.0})
            # anti-stagnation
            if (t+1) % 200 == 0 and len(replay) >= 200:
                last = replay[-200:]; var_dp = float(np.var([x["action"]["d_power_pct"] for x in last]))
                mask_rate = float(np.mean([x["g_costs"]["mask"] for x in last]))
                if var_dp < (0.02*self.fm.res_band_power)**2 and mask_rate > 0.2:
                    self.iql.policy.min_std_ratio = min(0.20, self.iql.policy.min_std_ratio * 1.4)
                    self.iql.policy.noise_calendar = list(reversed(self.iql.policy.noise_calendar))
                    self._log({"key":"anti_stagnation","when":int(t+1),"var_200":var_dp,"mask_rate_200":mask_rate,"min_std_ratio":float(self.iql.policy.min_std_ratio)})
            obs = next_obs if not done else self.env.reset(0)
            if (t+1) % max(1,sleep_every) == 0: time.sleep(max(0, sleep_sec))

        pol_path = self._save_policy(stage="safe_sac_online")
        return {"policy_path": pol_path, "env_steps": int(env_steps)}

    # ---------- 评估 ----------
    def evaluate_policy(self, steps: Optional[int] = None) -> Dict[str, Any]:
        self._load_policy()
        steps = steps or self.horizon
        def policy_fn(obs: Dict[str, Any]) -> Dict[str, Any]:
            phi = self.fm.obs_to_phi(obs); a = self.iql.policy.mean_action(phi)
            return {"d_power_pct": float(a[0]), "d_idle_min": float(a[1]), "mode": ""}
        summary = rollout_and_log(self.env, policy_fn, max_steps=steps)
        self._log({"key":"eval_summary","summary":summary})
        return summary

    # ---------- 日志 / 存取 ----------
    def _log(self, d: Dict[str, Any]):
        self.log.write(d)
        _mirror_to_static(DEFAULT_JSONL, STATIC_JSONL)

    def _save_policy(self, stage: str) -> str:
        base = os.path.dirname(__file__)
        pol_bin = os.path.join(base, "policy.bin")
        pol_meta = os.path.join(base, "policy_meta.json")
        obj = {
            "W": self.iql.policy.W.tolist(),
            "b": self.iql.policy.b.tolist(),
            "log_std": self.iql.policy.log_std.tolist(),
            "min_std_ratio": float(self.iql.policy.min_std_ratio),
            "res_band_power": float(self.iql.policy.res_band_power),
            "res_band_idle": float(self.iql.policy.res_band_idle)
        }
        with open(pol_bin, "w", encoding="utf-8") as f: f.write(json.dumps(obj))
        meta = {"stage":stage, "dt_min":self.dt_min, "horizon":self.horizon, "saved_at":int(time.time()), "ctx_keys": list(self.ctx.keys())}
        with open(pol_meta, "w", encoding="utf-8") as f: f.write(json.dumps(meta, ensure_ascii=False))
        _mirror_to_static(DEFAULT_JSONL, STATIC_JSONL)
        return pol_bin

    def _load_policy(self, path: Optional[str] = None) -> str:
        pol_bin = path or os.path.join(os.path.dirname(__file__), "policy.bin")
        if not os.path.exists(pol_bin) or os.path.getsize(pol_bin) == 0:
            raise RuntimeError("yard crane policy artifact is missing or empty")
        with open(pol_bin, "r", encoding="utf-8") as f:
            obj = json.load(f)
        W = np.asarray(obj["W"], dtype=np.float64)
        b = np.asarray(obj["b"], dtype=np.float64)
        log_std = np.asarray(obj["log_std"], dtype=np.float64)
        if W.shape != self.iql.policy.W.shape:
            raise ValueError(f"policy W shape mismatch: {W.shape} != {self.iql.policy.W.shape}")
        if b.shape != self.iql.policy.b.shape or log_std.shape != self.iql.policy.log_std.shape:
            raise ValueError("policy vector shape mismatch")
        self.iql.policy.W = W
        self.iql.policy.b = b
        self.iql.policy.log_std = log_std
        self.iql.policy.min_std_ratio = float(obj.get("min_std_ratio", self.iql.policy.min_std_ratio))
        return pol_bin

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Yard Crane RL Engine: TD3+BC/IQL offline + Safe-SAC online")
    p.add_argument("--train-offline", action="store_true", help="运行离线预训练（TD3+BC 默认，或 --algo iql）")
    p.add_argument("--online-finetune", action="store_true", help="运行 Safe-SAC 在线微调（孪生环境）")
    p.add_argument("--evaluate", action="store_true", help="评估当前策略")
    p.add_argument("--algo", type=str, default="td3bc", help="td3bc 或 iql")
    p.add_argument("--steps", type=int, default=30000, help="训练步数（离线）或环境步数（在线）")
    p.add_argument("--batch", type=int, default=512, help="批大小（离线）")
    p.add_argument("--policy-delay", type=int, default=2, help="TD3 策略延迟更新步数")
    p.add_argument("--tau", type=float, default=0.7, help="IQL expectile τ")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率（IQL/Actor/Q）")
    p.add_argument("--log-every", type=int, default=200, help="日志步间隔")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--sleep-every", type=int, default=1000, help="每隔 N 步休眠（固定回合休息）")
    p.add_argument("--sleep-sec", type=int, default=60, help="休眠秒数")
    p.add_argument("--dt-min", type=int, default=5, help="环境步长（分钟）")
    p.add_argument("--horizon", type=int, default=144, help="回放窗口步数（5min*144=12h）")
    args = p.parse_args()

    tr = Trainer(dt_min=args.dt_min, horizon=args.horizon, seed=args.seed)
    if args.train_offline:
        if args.algo.lower().startswith("iql"):
            out = tr.train_offline_iql(steps=args.steps, batch=args.batch, tau=args.tau, lr=args.lr,
                                       log_every=args.log_every, sleep_every=args.sleep_every, sleep_sec=args.sleep_sec)
        else:
            out = tr.train_offline_td3bc(steps=args.steps, batch=args.batch, policy_delay=args.policy_delay,
                                         log_every=args.log_every, sleep_every=args.sleep_every, sleep_sec=args.sleep_sec)
        print(json.dumps({"offline_train_done": True, **out}, ensure_ascii=False)); return
    if args.online_finetune:
        out = tr.online_finetune(algo_from=args.algo, env_steps=args.steps, batch_size=max(32, args.batch//8),
                                 log_every=args.log_every, sleep_every=args.sleep_every, sleep_sec=args.sleep_sec)
        print(json.dumps({"online_finetune_done": True, **out}, ensure_ascii=False)); return
    if args.evaluate:
        out = tr.evaluate_policy(steps=args.horizon)
        print(json.dumps({"evaluate_done": True, **out}, ensure_ascii=False)); return
    print("Nothing to do. Use --train-offline / --online-finetune / --evaluate")

if __name__ == "__main__":
    main()
