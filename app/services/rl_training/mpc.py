from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


class MPCPolicy:
    """Receding-horizon control baseline over the same environment controls."""

    def __init__(
        self,
        horizon: int = 6,
        action_dim: int = 3,
        *,
        episode_steps: int = 48,
        soc_min: float = 0.15,
        soc_max: float = 0.90,
        initial_soc: float = 0.55,
        bess_capacity_kwh: float = 100_000.0,
        bess_power_kw: float = 10_000.0,
        step_hours: float = 1.0,
        charge_efficiency: float = 0.96,
    ) -> None:
        self.horizon = max(2, int(horizon))
        self.action_dim = 7 if int(action_dim) >= 7 else 5 if int(action_dim) >= 5 else 3
        self.episode_steps = max(4, int(episode_steps))
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.initial_soc = float(initial_soc)
        self.bess_capacity_kwh = max(1.0, float(bess_capacity_kwh))
        self.bess_power_kw = max(1.0, float(bess_power_kw))
        self.step_hours = max(1e-6, float(step_hours))
        self.charge_efficiency = min(1.0, max(1e-6, float(charge_efficiency)))

    def parameters(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "action_dimensions": self.action_dim,
            "terminal_soc_aware": True,
            "soc_limits": [self.soc_min, self.soc_max],
            "episode_steps": self.episode_steps,
        }

    def _terminal_feasible_bess_action(
        self,
        requested: float,
        *,
        soc: float,
        progress: float,
    ) -> float:
        local_step = int(round(np.clip(progress, 0.0, 1.0) * (self.episode_steps - 1)))
        remaining_steps = max(0, self.episode_steps - local_step - 1)
        remaining_fraction = remaining_steps / self.episode_steps
        reachable_low = self.initial_soc - (self.initial_soc - self.soc_min) * remaining_fraction
        reachable_high = self.initial_soc + (self.soc_max - self.initial_soc) * remaining_fraction
        if requested >= 0:
            requested_delta = (
                requested
                * self.bess_power_kw
                * self.step_hours
                * self.charge_efficiency
                / self.bess_capacity_kwh
            )
        else:
            requested_delta = (
                requested
                * self.bess_power_kw
                * self.step_hours
                / (self.charge_efficiency * self.bess_capacity_kwh)
            )
        target_soc = float(np.clip(soc + requested_delta, reachable_low, reachable_high))
        delta = target_soc - soc
        if delta >= 0:
            feasible = (
                delta
                * self.bess_capacity_kwh
                / (self.bess_power_kw * self.step_hours * self.charge_efficiency)
            )
        else:
            feasible = (
                delta
                * self.bess_capacity_kwh
                * self.charge_efficiency
                / (self.bess_power_kw * self.step_hours)
            )
        return float(np.clip(feasible, -1.0, 1.0))

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=float)
        demand_signal = float(obs[2])
        price_signal = float(obs[6])
        carbon_signal = float(obs[7])
        state_offset = max(0, obs.size - 13)
        soc = float((obs[9 + state_offset] + 1.0) / 2.0)
        queue = float(obs[10 + state_offset])
        progress = float(obs[-1])

        def objective(vector: np.ndarray) -> float:
            controls = vector.reshape(self.horizon, 3)
            score = 0.0
            predicted_soc = soc
            for step, (bess, service, flex) in enumerate(controls):
                predicted_soc = np.clip(predicted_soc + 0.035 * bess, 0.0, 1.0)
                grid_pressure = demand_signal + 0.55 * bess + 0.18 * flex
                queue_cost = max(0.0, queue - 0.22 * service)
                price_cost = (0.55 + 0.45 * price_signal) * max(0.0, grid_pressure + 1.0)
                carbon_cost = (0.55 + 0.45 * carbon_signal) * max(0.0, grid_pressure + 1.0)
                safety = max(0.0, 0.12 - predicted_soc) ** 2 + max(0.0, predicted_soc - 0.88) ** 2
                smooth = 0.0 if step == 0 else np.sum((controls[step] - controls[step - 1]) ** 2)
                score += price_cost + 0.7 * carbon_cost + 1.6 * queue_cost + 20.0 * safety + 0.04 * smooth
            return float(score)

        bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)] * self.horizon
        result = minimize(objective, np.zeros(self.horizon * 3), method="L-BFGS-B", bounds=bounds)
        action = np.asarray(result.x[:3] if result.success else np.zeros(3), dtype=np.float32)
        action[0] = self._terminal_feasible_bess_action(
            float(action[0]),
            soc=soc,
            progress=progress,
        )
        if self.action_dim > 3:
            action = np.concatenate(
                [action, np.zeros(self.action_dim - 3, dtype=np.float32)]
            )
        return action, None
