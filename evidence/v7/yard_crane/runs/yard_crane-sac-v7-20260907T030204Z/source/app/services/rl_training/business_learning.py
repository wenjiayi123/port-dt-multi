"""Training-only credit assignment; physical transitions and KPI accounting stay shared."""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class FiniteDispatchActions(gym.ActionWrapper):
    """Q-learning lattice including an exact idle command for lossy storage."""

    def __init__(self, env, lattice):
        super().__init__(env)
        self.lattice = np.asarray(lattice, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(lattice))

    def action(self, action):
        index = int(np.asarray(action).item())
        if not 0 <= index < len(self.lattice):
            raise ValueError("dispatch action outside lattice")
        return self.lattice[index].copy()


class BusinessDeltaReward(gym.Wrapper):
    """Remove action-independent load variation from the specialist reward.

    No teacher actions, future rows or evaluation observations are consumed.
    The policy keeps the original observation/action interface. All physical
    service and safety constraints remain in the underlying environment.
    """

    def __init__(self, env, module: str, gamma: float, carbon_multiplier: float = 1.0, peak_multiplier: float = 1.0):
        super().__init__(env)
        self.module = module
        self.gamma = gamma
        self.carbon_multiplier = carbon_multiplier
        self.peak_multiplier = peak_multiplier
        self.previous_potential = 0.0

    def _potential(self):
        env = self.unwrapped
        if self.module not in {"shore_bess", "bess_energy"}:
            return 0.0
        # Fixed inventory value, independent of future prices.
        # Potential-based shaping cancels at terminal and reduces the delay
        # between paying for charge and realizing discharge value.
        return 0.4 * (env._soc - env.soc_initial) * env.energy_kwh / env.power_kw

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.previous_potential = self._potential()
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        original_reward = reward
        if self.module in {"shore_bess", "bess_energy"}:
            reward -= (self.carbon_multiplier - 1.0) * self.unwrapped.weights["carbon"] * info["reward_components"]["carbon"]
            reward -= (self.peak_multiplier - 1.0) * self.unwrapped.weights["demand_peak"] * info["reward_components"]["demand_peak"]
            potential = 0.0 if terminated else self._potential()
            reward += self.gamma * potential - self.previous_potential
            self.previous_potential = potential
            reward *= 10.0
        else:
            b = info["business_step"]
            cost_gain = (b["baseline_energy_cost_cny"] - b["energy_cost_cny"]) / max(b["baseline_energy_cost_cny"], 1.0)
            carbon_gain = (b["baseline_carbon_kg"] - b["carbon_kg"]) / max(b["baseline_carbon_kg"], 1.0)
            power_gain = (b["baseline_power_kw"] - b["power_kw"]) / max(b["baseline_power_kw"], 1.0)
            failures = sum(max(0.0, 1.0 - b[key]) for key in (
                "cooling_satisfaction", "moves_retention", "job_sla_non_degradation",
                "minimum_lux_compliance_rate", "critical_lux_compliance_rate",
            ) if key in b)
            reward = 10.0 * (cost_gain + 0.25 * carbon_gain + 0.25 * self.peak_multiplier * power_gain)
            reward -= 10.0 * (failures + float(info.get("guardrail_violation", False)))
            reward -= 0.005 * len(info.get("projection", []))
        if not np.isfinite(reward):
            raise ValueError("non-finite business reward")
        return obs, float(reward), terminated, truncated, {**info, "original_reward": original_reward}
