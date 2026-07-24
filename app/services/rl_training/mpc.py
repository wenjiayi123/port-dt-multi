from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


class MPCPolicy:
    """Receding-horizon control baseline over the same environment controls."""

    def __init__(self, horizon: int = 6) -> None:
        self.horizon = max(2, int(horizon))

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=float)
        demand_signal = float(obs[2])
        price_signal = float(obs[6])
        carbon_signal = float(obs[7])
        soc = float((obs[9] + 1.0) / 2.0)
        queue = float(obs[10])

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
        return action, None
