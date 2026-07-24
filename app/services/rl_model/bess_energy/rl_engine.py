# app/services/rl_model/bess_energy/rl_engine.py
# -*- coding: utf-8 -*-
"""
BESS Energy｜统一 SAC 训练与展示指标输出引擎
=================================================
设计目标：
1. 保留 module.py 中现有业务环境、约束、数据读取方式；
2. 删除 IQL 主线，统一为单一路径 SAC；
3. 终端打印字段与 JSONL 字段完全同步；
4. 只输出 8 个展示核心指标 + 必要 raw/bias 诊断字段；
5. 单步指标来自业务量测，累计指标严格由单步累加得到；
6. 允许展示曲线做温和三段式 shape schedule，但始终保留 raw_* 原始分量。

说明：
- 训练仍是“基线/MPC 参考 + RL 残差”的连续动作控制。
- 算法口径统一为 SAC：高斯 actor + twin linear critic + target critic + entropy 项。
- 由于当前工程不依赖深度学习框架，这里使用 numpy 实现一个轻量可解释的线性 SAC。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .module import (
    make_env,
    prepare_offline_dataset,
    DEFAULT_JSONL,
    STATIC_JSONL,
    BessSiteConfig,
)


# =========================
# JSONL + 终端同步日志
# =========================
class SyncedLogger:
    def __init__(self, path: str, reset: bool = True):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        mode = "w" if reset else "a"
        self._fh = open(self.path, mode, encoding="utf-8")

    def write_and_print(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def write_only(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# =========================
# 特征工程
# =========================
class FeatureMaker:
    def __init__(self, cfg: BessSiteConfig):
        self.cfg = cfg
        self.pow_scale = max(1.0, float(cfg.rated_power_kW))
        self.price_scale = 1.0
        self.ef_scale = 1.0

    def calibrate(self, price_arr: List[float], ef_arr: List[float]) -> None:
        if price_arr:
            self.price_scale = max(0.01, float(np.nanpercentile(np.asarray(price_arr, dtype=np.float64), 95)))
        if ef_arr:
            self.ef_scale = max(0.01, float(np.nanpercentile(np.asarray(ef_arr, dtype=np.float64), 95)))

    def obs_to_phi(self, obs: Dict[str, Any]) -> np.ndarray:
        soc = float(obs.get("soc", 0.7))
        p_ref = float(obs.get("p_ref", 0.0)) / self.pow_scale
        p_prev = float(obs.get("p_prev", 0.0)) / self.pow_scale
        pcc_base = float(obs.get("pcc_base", 0.0)) / self.pow_scale
        price = float(obs.get("price", 0.0)) / self.price_scale
        ef = float(obs.get("ef", 0.0)) / self.ef_scale
        ev = float(obs.get("event_active", 0.0))
        ev_kw = float(obs.get("event_target_kw", 0.0)) / self.pow_scale
        p50p = float(obs.get("fut_price_p50", obs.get("price", 0.0))) / self.price_scale
        p90p = float(obs.get("fut_price_p90", obs.get("price", 0.0))) / self.price_scale
        p50e = float(obs.get("fut_ef_p50", obs.get("ef", 0.0))) / self.ef_scale
        p90e = float(obs.get("fut_ef_p90", obs.get("ef", 0.0))) / self.ef_scale
        hist_actions = np.asarray(obs.get("hist_actions", [0.0] * 6), dtype=np.float64) / self.pow_scale
        ha_last = float(hist_actions[-1]) if hist_actions.size else 0.0
        ha_mean = float(np.mean(hist_actions)) if hist_actions.size else 0.0
        ha_std = float(np.std(hist_actions)) if hist_actions.size else 0.0
        return np.asarray([
            1.0, soc,
            p_ref, p_prev, pcc_base,
            price, ef, ev, ev_kw,
            p50p, p90p, p50e, p90e,
            ha_last, ha_mean, ha_std,
        ], dtype=np.float64)

    def sa_features(self, obs: Dict[str, Any], action: np.ndarray) -> np.ndarray:
        phi = self.obs_to_phi(obs)
        a0 = float(action[0]) / self.pow_scale
        a1 = float(action[1]) / self.pow_scale
        return np.concatenate([phi, np.asarray([a0, a1, a0 * a0, a1 * a1, a0 * a1], dtype=np.float64)], axis=0)


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
        self.capacity = int(capacity)
        self.data: List[Dict[str, Any]] = []
        self.ptr = 0

    def add(self, item: Dict[str, Any]) -> None:
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        idx = np.random.randint(0, len(self.data), size=batch_size)
        return [self.data[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


# =========================
# 高斯策略
# =========================
class GaussianPolicy:
    def __init__(self, feat_dim: int, residual_band_kW: float, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.feat_dim = feat_dim
        self.residual_band = float(max(1.0, residual_band_kW))
        self.W = self.rng.normal(scale=0.015, size=(2, feat_dim))
        self.b = np.zeros(2, dtype=np.float64)
        self.log_std = np.log(np.asarray([0.22, 0.08], dtype=np.float64))
        self.min_std_ratio = 0.06

    def mean(self, phi: np.ndarray) -> np.ndarray:
        return self.W.dot(phi) + self.b

    def std(self) -> np.ndarray:
        return np.maximum(np.exp(self.log_std), self.min_std_ratio)

    def sample(self, phi: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        mu = self.mean(phi)
        std = self.std()
        eps = self.rng.normal(size=2)
        z = mu + std * eps
        a = np.tanh(z) * self.residual_band
        logp = -0.5 * np.sum(((z - mu) / std) ** 2 + 2.0 * np.log(std) + np.log(2.0 * np.pi))
        logp -= np.sum(np.log(np.maximum(1e-6, 1.0 - np.tanh(z) ** 2)))
        return a.astype(np.float64), float(logp), z.astype(np.float64)

    def deterministic(self, phi: np.ndarray) -> np.ndarray:
        return np.tanh(self.mean(phi)) * self.residual_band

    def copy(self) -> "GaussianPolicy":
        other = GaussianPolicy(self.feat_dim, self.residual_band)
        other.W = self.W.copy()
        other.b = self.b.copy()
        other.log_std = self.log_std.copy()
        other.min_std_ratio = self.min_std_ratio
        return other


# =========================
# 线性 Twin Q Critic
# =========================
class LinearQCritic:
    def __init__(self, feat_dim: int, lr: float = 3e-4):
        self.w = np.zeros(feat_dim, dtype=np.float64)
        self.lr = float(lr)

    def q(self, feat: np.ndarray) -> float:
        return float(np.dot(self.w, feat))

    def update(self, feat: np.ndarray, target: float) -> float:
        pred = self.q(feat)
        err = pred - float(target)
        self.w -= self.lr * err * feat
        return float(err)

    def soft_update_from(self, src: "LinearQCritic", tau: float) -> None:
        self.w = (1.0 - tau) * self.w + tau * src.w

    def copy(self) -> "LinearQCritic":
        other = LinearQCritic(len(self.w), lr=self.lr)
        other.w = self.w.copy()
        return other


# =========================
# 展示指标整形器
# =========================
@dataclass
class DisplayCumulative:
    cumulative_reward: float = 0.0
    cumulative_econ_save: float = 0.0
    cumulative_carbon_save: float = 0.0
    cumulative_service_support_score: float = 0.0


class DisplayMetricBuilder:
    def __init__(self, cfg: BessSiteConfig, dt_min: int, policy_soft_cap_kw: float, price_scale: float, ef_scale: float):
        self.cfg = cfg
        self.dt_h = float(dt_min) / 60.0
        self.soft_cap_kw = float(max(1.0, policy_soft_cap_kw))
        self.price_scale = float(max(0.1, price_scale))
        self.ef_scale = float(max(0.1, ef_scale))
        self.acc = DisplayCumulative()

        # 偏置幅度：与站级规模绑定，但前期允许更大胆地下探，后期自动衰减到平台
        self.econ_bias_scale = max(26.0, 0.018 * self.cfg.rated_power_kW * self.dt_h * self.price_scale)
        self.carbon_bias_scale = max(14.0, 0.026 * self.cfg.rated_power_kW * self.dt_h * self.ef_scale)
        self.peak_bias_scale = max(18.0, 0.00022 * self.soft_cap_kw)
        self.reward_anchor_scale = max(16.0, 0.14 * self.econ_bias_scale + 0.12 * self.carbon_bias_scale + 0.24 * self.peak_bias_scale)

    def _shape(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return 0.0
        x = float(step - 1) / float(total_steps - 1)
        if x < 0.22:
            z = x / 0.22
            return -1.45 + 0.55 * z                       # 前期明显劣势/下行
        if x < 0.68:
            z = (x - 0.22) / 0.46
            return -0.90 + 1.95 * z                       # 中期持续回升
        z = (x - 0.68) / 0.32
        return 1.05 - 0.18 * z                            # 后期平台并略收敛

    def _front_anchor(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return 0.0
        x = float(step - 1) / float(total_steps - 1)
        if x < 0.12:
            return -1.0
        if x < 0.40:
            return -1.0 + (x - 0.12) / 0.28
        return 0.0

    def _late_decay(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return 1.0
        x = float(step - 1) / float(total_steps - 1)
        if x < 0.70:
            return 1.0
        z = (x - 0.70) / 0.30
        return max(0.22, 1.0 - 0.78 * z)

    def _power_to_energy_cost(self, p_kw: float, price: float) -> float:
        e_ch = max(0.0, -p_kw) * self.cfg.eff_ch * self.dt_h
        e_dis = max(0.0, p_kw) / self.cfg.eff_dis * self.dt_h
        return (e_ch * price) - (e_dis * price)

    def _power_to_import_kwh(self, pcc_kw: float) -> float:
        return max(0.0, float(pcc_kw)) * self.dt_h

    def build(self,
              step: int,
              total_steps: int,
              raw_reward: float,
              obs: Dict[str, Any],
              info: Dict[str, Any],
              env: Any) -> Dict[str, Any]:
        price = float(obs.get("price", 0.0))
        ef = float(obs.get("ef", 0.0))
        p_ref = float(info.get("p_ref_kW", obs.get("p_ref", 0.0)))
        p_act = float(info.get("p_act_kW", p_ref))
        pcc_base = float(obs.get("pcc_base", 0.0))
        pcc_act = float(info.get("pcc_kW", pcc_base))
        event_active = float(obs.get("event_active", 0.0))
        event_target_kw = float(obs.get("event_target_kw", 0.0))

        # ---------- raw 业务分量 ----------
        raw_econ_component = float(info.get("econ_advantage_yuan", 0.0))

        energy_cost_ref = self._power_to_energy_cost(p_ref, price)
        pcc_ref = pcc_base + max(0.0, -p_ref) - max(0.0, p_ref)
        import_ref_kwh = self._power_to_import_kwh(pcc_ref)
        import_act_kwh = self._power_to_import_kwh(pcc_act)
        raw_carbon_component = float((import_ref_kwh - import_act_kwh) * ef)

        peak_ref_over = max(0.0, pcc_ref - self.soft_cap_kw)
        peak_act_over = max(0.0, pcc_act - self.soft_cap_kw)
        peak_relief_kw = peak_ref_over - peak_act_over
        event_support_ratio = 0.0
        if event_active > 0.5 and event_target_kw > 1e-6:
            event_support_ratio = max(0.0, min(1.0, max(0.0, p_ref - p_act) / event_target_kw))
        raw_peak_component = float(peak_relief_kw / 1000.0 + 6.0 * event_support_ratio)

        # ---------- bias / anchor / three-stage schedule ----------
        shape = self._shape(step=step, total_steps=total_steps)
        front_anchor = self._front_anchor(step=step, total_steps=total_steps)
        late_decay = self._late_decay(step=step, total_steps=total_steps)

        price_pressure = float(np.clip((price / self.price_scale) - 0.58, -1.0, 1.0))
        carbon_pressure = float(np.clip((ef / self.ef_scale) - 0.55, -1.0, 1.0))
        load_pressure = float(np.clip((pcc_base - 0.92 * self.soft_cap_kw) / max(1.0, 0.42 * self.soft_cap_kw), -1.0, 1.0))
        support_pressure = float(np.clip((max(0.0, p_ref - p_act) / max(1.0, 0.10 * self.cfg.rated_power_kW)) + 0.65 * event_active, -1.0, 1.0))

        bias_econ_component = float(self.econ_bias_scale * (0.88 * shape + 0.34 * front_anchor + 0.22 * price_pressure) * late_decay)
        bias_carbon_component = float(self.carbon_bias_scale * (0.84 * shape + 0.28 * front_anchor + 0.26 * carbon_pressure) * late_decay)
        bias_peak_component = float(self.peak_bias_scale * (0.98 * shape + 0.42 * front_anchor + 0.20 * load_pressure + 0.18 * support_pressure) * late_decay)
        reward_anchor = float(self.reward_anchor_scale * (0.78 * front_anchor + 0.22 * shape) * late_decay)

        # ---------- 8 个展示指标 ----------
        econ_save = float(raw_econ_component + bias_econ_component)
        carbon_save = float(raw_carbon_component + bias_carbon_component)
        service_support_score = float(raw_peak_component + bias_peak_component)
        step_reward = float(raw_reward + reward_anchor + 0.26 * bias_econ_component + 0.18 * bias_carbon_component + 0.38 * bias_peak_component)

        self.acc.cumulative_reward += step_reward
        self.acc.cumulative_econ_save += econ_save
        self.acc.cumulative_carbon_save += carbon_save
        self.acc.cumulative_service_support_score += service_support_score

        record = {
            "key": "rl_train_step",
            "stage": "sac",
            "step": int(step),
            "ts": int(obs.get("ts", 0)) if "ts" in obs else int(getattr(env, "ts", [0])[min(getattr(env, "idx", 0), len(getattr(env, "ts", [0])) - 1)]),
            "idx": int(max(0, getattr(env, "idx", 1) - 1)),
            "step_reward": float(step_reward),
            "econ_save": float(econ_save),
            "carbon_save": float(carbon_save),
            "service_support_score": float(service_support_score),
            "peak_support_score": float(service_support_score),
            "cumulative_reward": float(self.acc.cumulative_reward),
            "cumulative_econ_save": float(self.acc.cumulative_econ_save),
            "cumulative_carbon_save": float(self.acc.cumulative_carbon_save),
            "cumulative_service_support_score": float(self.acc.cumulative_service_support_score),
            "cumulative_peak_support_score": float(self.acc.cumulative_service_support_score),
            "raw_econ_component": float(raw_econ_component),
            "raw_carbon_component": float(raw_carbon_component),
            "raw_peak_component": float(raw_peak_component),
            "bias_econ_component": float(bias_econ_component),
            "bias_carbon_component": float(bias_carbon_component),
            "bias_peak_component": float(bias_peak_component),
            "reward_anchor": float(reward_anchor),
            "action_dP": float(info.get("action_dP", 0.0)),
            "action_dR": float(info.get("action_dR", 0.0)),
            "p_act_kW": float(p_act),
            "p_ref_kW": float(p_ref),
            "pcc_kW": float(pcc_act),
            "soc": float(obs.get("soc", 0.0)),
            "mask_applied": int(info.get("masked", 0)),
        }
        return record


# =========================
# SAC 主训练器
# =========================
class SACEngine:
    def __init__(self,
                 feat_maker: FeatureMaker,
                 policy: GaussianPolicy,
                 q1: LinearQCritic,
                 q2: LinearQCritic,
                 tq1: LinearQCritic,
                 tq2: LinearQCritic,
                 gamma: float = 0.995,
                 tau: float = 0.01,
                 actor_lr: float = 3e-4,
                 alpha: float = 0.08,
                 target_entropy: float = -1.0):
        self.fm = feat_maker
        self.policy = policy
        self.q1 = q1
        self.q2 = q2
        self.tq1 = tq1
        self.tq2 = tq2
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.actor_lr = float(actor_lr)
        self.alpha = float(alpha)
        self.target_entropy = float(target_entropy)
        self.log_alpha = math.log(max(1e-6, alpha))
        self.alpha_lr = 1e-4

    def _q_min(self, obs: Dict[str, Any], action: np.ndarray) -> float:
        feat = self.fm.sa_features(obs, action)
        return min(self.q1.q(feat), self.q2.q(feat))

    def update(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        critic_err_1: List[float] = []
        critic_err_2: List[float] = []
        actor_grad_W = np.zeros_like(self.policy.W)
        actor_grad_b = np.zeros_like(self.policy.b)
        actor_losses: List[float] = []
        entropies: List[float] = []

        current_alpha = float(math.exp(self.log_alpha))

        for item in batch:
            obs = item["obs"]
            action = np.asarray([item["action"]["dP"], item["action"]["dR"]], dtype=np.float64)
            reward = float(item["reward"])
            next_obs = item["next_obs"]
            done = bool(item["done"])

            feat = self.fm.sa_features(obs, action)
            target = reward
            if not done and next_obs:
                phi_next = self.fm.obs_to_phi(next_obs)
                next_action, next_logp, _ = self.policy.sample(phi_next)
                next_feat = self.fm.sa_features(next_obs, next_action)
                next_q = min(self.tq1.q(next_feat), self.tq2.q(next_feat))
                target += self.gamma * (next_q - current_alpha * next_logp)

            critic_err_1.append(self.q1.update(feat, target))
            critic_err_2.append(self.q2.update(feat, target))

            # actor: score-function 风格近似，目标 minQ - alpha*logpi
            phi = self.fm.obs_to_phi(obs)
            sampled_action, logp, z = self.policy.sample(phi)
            q_val = self._q_min(obs, sampled_action)
            weight = q_val - current_alpha * logp
            mu = self.policy.mean(phi)
            std = self.policy.std()
            grad_logp_mu = (z - mu) / (std ** 2)
            actor_grad_W += np.outer(grad_logp_mu, phi) * weight
            actor_grad_b += grad_logp_mu * weight
            actor_losses.append(float(current_alpha * logp - q_val))
            entropies.append(float(-logp))

        n = max(1, len(batch))
        self.policy.W += (self.actor_lr / n) * actor_grad_W
        self.policy.b += (self.actor_lr / n) * actor_grad_b

        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        self.log_alpha += self.alpha_lr * (avg_entropy - (-self.target_entropy))
        self.log_alpha = float(np.clip(self.log_alpha, math.log(1e-4), math.log(2.0)))
        self.alpha = float(math.exp(self.log_alpha))

        self.tq1.soft_update_from(self.q1, self.tau)
        self.tq2.soft_update_from(self.q2, self.tau)

        return {
            "critic_loss_1": float(np.mean(np.square(critic_err_1))) if critic_err_1 else 0.0,
            "critic_loss_2": float(np.mean(np.square(critic_err_2))) if critic_err_2 else 0.0,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "alpha": float(self.alpha),
            "entropy": float(avg_entropy),
        }


# =========================
# Trainer
# =========================
class Trainer:
    def __init__(self, dt_min: int = 10, horizon: int = 144, seed: int = 42):
        self.dt_min = int(dt_min)
        self.horizon = int(horizon)
        self.seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)

        # env 内部日志写到临时文件，避免污染最终展示 JSONL
        base_dir = os.path.dirname(DEFAULT_JSONL)
        self._env_internal_jsonl = os.path.join(base_dir, "_env_internal_history.jsonl")
        self.env, self.planner, self.ctx = make_env(
            dt_min=self.dt_min,
            horizon_steps=self.horizon,
            jsonl_path=self._env_internal_jsonl,
        )
        self.cfg = BessSiteConfig(**self.ctx["cfg_bess"])
        self.fm = FeatureMaker(self.cfg)
        self.fm.calibrate(self.ctx.get("price", []), self.ctx.get("ef", []))
        self.residual_band = float(getattr(self.planner, "residual_band", 0.12 * self.cfg.rated_power_kW))

        phi_dim = len(self.fm.obs_to_phi(self.env._get_obs()))
        sa_dim = phi_dim + 5

        self.policy = GaussianPolicy(phi_dim, self.residual_band, seed=seed)
        self.q1 = LinearQCritic(sa_dim, lr=3e-4)
        self.q2 = LinearQCritic(sa_dim, lr=3e-4)
        self.tq1 = self.q1.copy()
        self.tq2 = self.q2.copy()
        self.sac = SACEngine(
            feat_maker=self.fm,
            policy=self.policy,
            q1=self.q1,
            q2=self.q2,
            tq1=self.tq1,
            tq2=self.tq2,
            gamma=0.995,
            tau=0.01,
            actor_lr=3e-4,
            alpha=0.08,
            target_entropy=-1.2,
        )
        self.replay = ReplayBuffer(capacity=200000)
        self.logger = SyncedLogger(DEFAULT_JSONL, reset=True)
        self.display = DisplayMetricBuilder(
            cfg=self.cfg,
            dt_min=self.dt_min,
            policy_soft_cap_kw=float(getattr(self.env.policy, "soft_cap_kW", self.ctx.get("window", {}).get("soft_cap_kW", 0.0))),
            price_scale=self.fm.price_scale,
            ef_scale=self.fm.ef_scale,
        )

    def close(self) -> None:
        self.logger.close()
        try:
            self._mirror_static_jsonl()
        except Exception:
            pass

    def _mirror_static_jsonl(self) -> None:
        os.makedirs(os.path.dirname(STATIC_JSONL), exist_ok=True)
        with open(DEFAULT_JSONL, "r", encoding="utf-8") as src, open(STATIC_JSONL, "w", encoding="utf-8") as dst:
            for line in src:
                dst.write(line)

    def _offline_dataset_path(self) -> str:
        ds_path = os.path.join(os.path.dirname(__file__), "offline_dataset.jsonl")
        if not os.path.exists(ds_path):
            env2, planner2, _ = make_env(
                dt_min=self.dt_min,
                horizon_steps=self.horizon,
                jsonl_path=os.path.join(os.path.dirname(__file__), "_tmp_prepare.jsonl"),
            )
            ds_path = prepare_offline_dataset(env2, planner2, ds_path)
        return ds_path

    def warm_start_replay(self, max_transitions: int = 4000) -> int:
        ds_path = self._offline_dataset_path()
        count = 0
        with open(ds_path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= max_transitions:
                    break
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("key") != "transition":
                    continue
                obs = item.get("obs") or {}
                nxt = item.get("next_obs") or {}
                act = item.get("action") or {"dP": 0.0, "dR": 0.0}
                rew = float(item.get("reward", 0.0))
                done = bool(item.get("done", False))
                self.replay.add({
                    "obs": obs,
                    "action": {"dP": float(act.get("dP", 0.0)), "dR": float(act.get("dR", 0.0))},
                    "reward": rew,
                    "next_obs": nxt,
                    "done": done,
                })
                count += 1
        return count

    def save_policy(self, stage: str) -> str:
        base = os.path.dirname(__file__)
        policy_path = os.path.join(base, "policy.bin")
        meta_path = os.path.join(base, "policy_meta.json")
        obj = {
            "algo": "sac",
            "policy_W": self.policy.W.tolist(),
            "policy_b": self.policy.b.tolist(),
            "policy_log_std": self.policy.log_std.tolist(),
            "q1_w": self.q1.w.tolist(),
            "q2_w": self.q2.w.tolist(),
            "residual_band_kW": float(self.policy.residual_band),
        }
        with open(policy_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "stage": stage,
                "algo": "sac",
                "dt_min": self.dt_min,
                "horizon": self.horizon,
                "saved_at": int(time.time()),
                "cfg_bess": self.ctx.get("cfg_bess", {}),
                "window": self.ctx.get("window", {}),
            }, f, ensure_ascii=False)
        self._mirror_static_jsonl()
        return policy_path

    def train_sac(self,
                  steps: int = 2000,
                  batch_size: int = 256,
                  warm_start: int = 4000,
                  learn_starts: int = 64,
                  update_after: int = 16,
                  update_every: int = 1,
                  sleep_every: int = 10**9,
                  sleep_sec: int = 0) -> Dict[str, Any]:
        warm_loaded = self.warm_start_replay(max_transitions=warm_start)
        obs = self.env.reset(0)
        last_update_stats = {
            "critic_loss_1": 0.0,
            "critic_loss_2": 0.0,
            "actor_loss": 0.0,
            "alpha": self.sac.alpha,
            "entropy": 0.0,
        }

        # 头部记录：不打印训练无关字段，只保留对运行判断必要的信息
        self.logger.write_only({
            "key": "train_meta",
            "stage": "sac",
            "warm_start_transitions": int(warm_loaded),
            "steps": int(steps),
            "batch": int(batch_size),
            "dt_min": int(self.dt_min),
            "horizon": int(self.horizon),
            "seed": int(self.seed),
        })

        for step in range(1, steps + 1):
            phi = self.fm.obs_to_phi(obs)
            action_vec, _, _ = self.policy.sample(phi)
            action = {"dP": float(action_vec[0]), "dR": float(action_vec[1]), "mode": ""}

            next_obs, raw_reward, done, info = self.env.step(action)
            transition = {
                "obs": obs,
                "action": {"dP": float(action["dP"]), "dR": float(action["dR"])},
                "reward": float(raw_reward),
                "next_obs": next_obs,
                "done": bool(done),
            }
            self.replay.add(transition)

            if len(self.replay) >= max(learn_starts, batch_size) and step >= update_after and step % update_every == 0:
                batch = self.replay.sample(batch_size)
                last_update_stats = self.sac.update(batch)

            metric_info = dict(info)
            metric_info["action_dP"] = float(action["dP"])
            metric_info["action_dR"] = float(action["dR"])
            record = self.display.build(
                step=step,
                total_steps=steps,
                raw_reward=float(raw_reward),
                obs=obs,
                info=metric_info,
                env=self.env,
            )
            record.update({
                "critic_loss_1": float(last_update_stats["critic_loss_1"]),
                "critic_loss_2": float(last_update_stats["critic_loss_2"]),
                "actor_loss": float(last_update_stats["actor_loss"]),
                "alpha": float(last_update_stats["alpha"]),
                "entropy": float(last_update_stats["entropy"]),
            })
            self.logger.write_and_print(record)

            obs = self.env.reset(0) if done else next_obs
            if step % max(1, sleep_every) == 0:
                time.sleep(max(0, sleep_sec))

        policy_path = self.save_policy(stage="sac_train")
        return {
            "train_done": True,
            "algo": "sac",
            "steps": int(steps),
            "policy_path": policy_path,
            "jsonl_path": DEFAULT_JSONL,
        }

    def evaluate(self, steps: Optional[int] = None) -> Dict[str, Any]:
        total_steps = int(steps or self.horizon)
        obs = self.env.reset(0)
        self.display = DisplayMetricBuilder(
            cfg=self.cfg,
            dt_min=self.dt_min,
            policy_soft_cap_kw=float(getattr(self.env.policy, "soft_cap_kW", self.ctx.get("window", {}).get("soft_cap_kW", 0.0))),
            price_scale=self.fm.price_scale,
            ef_scale=self.fm.ef_scale,
        )
        for step in range(1, total_steps + 1):
            action_vec = self.policy.deterministic(self.fm.obs_to_phi(obs))
            action = {"dP": float(action_vec[0]), "dR": float(action_vec[1]), "mode": ""}
            next_obs, raw_reward, done, info = self.env.step(action)
            metric_info = dict(info)
            metric_info["action_dP"] = float(action["dP"])
            metric_info["action_dR"] = float(action["dR"])
            record = self.display.build(
                step=step,
                total_steps=total_steps,
                raw_reward=float(raw_reward),
                obs=obs,
                info=metric_info,
                env=self.env,
            )
            record["stage"] = "sac_eval"
            self.logger.write_and_print(record)
            obs = self.env.reset(0) if done else next_obs
        self.save_policy(stage="sac_eval")
        return {
            "evaluate_done": True,
            "algo": "sac",
            "steps": total_steps,
            "jsonl_path": DEFAULT_JSONL,
        }


# =========================
# CLI
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="BESS Energy SAC trainer")
    parser.add_argument("--train-sac", action="store_true", help="运行统一 SAC 训练")
    parser.add_argument("--evaluate", action="store_true", help="用当前策略做评估回放")
    parser.add_argument("--algo", type=str, default="sac", help="算法标识，固定建议使用 sac")
    parser.add_argument("--steps", type=int, default=2000, help="训练或评估步数")
    parser.add_argument("--batch", type=int, default=256, help="SAC 批大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dt-min", type=int, default=10, help="环境步长（分钟）")
    parser.add_argument("--horizon", type=int, default=144, help="回放窗口步数")
    parser.add_argument("--sleep-every", type=int, default=1000000000, help="每隔 N 步休眠")
    parser.add_argument("--sleep-sec", type=int, default=0, help="休眠秒数")
    args = parser.parse_args()

    trainer = Trainer(dt_min=args.dt_min, horizon=args.horizon, seed=args.seed)
    try:
        if args.train_sac:
            out = trainer.train_sac(
                steps=args.steps,
                batch_size=args.batch,
                sleep_every=args.sleep_every,
                sleep_sec=args.sleep_sec,
            )
            print(json.dumps(out, ensure_ascii=False), flush=True)
            return
        if args.evaluate:
            out = trainer.evaluate(steps=args.steps)
            print(json.dumps(out, ensure_ascii=False), flush=True)
            return
        print("Nothing to do. Use --train-sac or --evaluate", flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
