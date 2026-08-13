from __future__ import annotations

from typing import Any

import numpy as np

from .mpc import MPCPolicy


class FCFSNeutralPolicy:
    """Deterministic do-nothing/FCFS comparator for causal advantage claims.

    The controller does not optimize against the holdout. It keeps storage and
    flexible load neutral, uses the environment's natural first-come-first-
    served queue, and applies neither berth priority nor yard-flow preference.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = max(1, int(action_dim))

    def predict(self, _observation: Any, deterministic: bool = True):
        del deterministic
        return np.zeros(self.action_dim, dtype=np.float32), None

    def parameters(self) -> dict[str, Any]:
        return {
            "queue_discipline": "first_come_first_served",
            "bess_command": "neutral",
            "flexible_load_command": "neutral",
            "berth_priority": "neutral",
            "yard_flow": "neutral",
            "holdout_tuning": False,
        }


class EngineeringCurrentOpsRulePolicy(MPCPolicy):
    """Transparent SOP-style proxy used when terminal operating logs are absent.

    This is deliberately named an engineering proxy rather than a measured
    current-policy baseline. It combines time-of-use BESS dispatch, queue-based
    service escalation, flexible-load curtailment and bounded allocation
    priorities. Every threshold is fixed before blind-test evaluation.
    """

    def predict(self, observation: Any, deterministic: bool = True):
        del deterministic
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.size < 13:
            raise ValueError("current-operations rule requires a canonical port observation")
        price_signal = float(obs[6])
        carbon_signal = float(obs[7])
        soc = float(np.clip((obs[-4] + 1.0) / 2.0, 0.0, 1.0))
        queue_pressure = float(np.clip(obs[-3], 0.0, 1.5))
        progress = float(np.clip(obs[-1], 0.0, 1.0))

        stress = max(price_signal, 0.65 * price_signal + 0.35 * carbon_signal)
        if stress >= 0.25 and soc > 0.30:
            bess = -min(0.55, max(0.0, (soc - 0.30) * 1.8))
        elif stress <= -0.25 and soc < 0.75:
            bess = min(0.45, max(0.0, (0.75 - soc) * 1.5))
        else:
            bess = 0.0
        bess = self._terminal_feasible_bess_action(bess, soc=soc, progress=progress)

        if queue_pressure >= 0.45:
            service, berth, yard = 0.70, 0.65, 0.60
        elif queue_pressure >= 0.20:
            service, berth, yard = 0.35, 0.30, 0.25
        else:
            service, berth, yard = 0.05, 0.0, 0.0
        flexible = -0.55 if stress >= 0.25 else 0.30 if stress <= -0.25 else 0.0
        action = np.asarray([bess, service, flexible], dtype=np.float32)
        if self.action_dim == 5:
            action = np.concatenate(
                [action, np.asarray([berth, yard], dtype=np.float32)]
            )
        return action, None

    def parameters(self) -> dict[str, Any]:
        return {
            "baseline_kind": "engineering_current_operations_rule_proxy",
            "measured_operator_policy": False,
            "holdout_tuning": False,
            "fixed_thresholds": {
                "high_price_or_carbon_signal": 0.25,
                "low_price_or_carbon_signal": -0.25,
                "queue_escalation": [0.20, 0.45],
            },
            "controls": [
                "time_of_use_bess",
                "queue_based_service",
                "flexible_load_curtailment",
                "berth_priority",
                "yard_flow",
            ],
            "replacement_required": "site SOP replay and operator dispatch logs",
        }
