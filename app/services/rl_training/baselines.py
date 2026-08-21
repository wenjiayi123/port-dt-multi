from __future__ import annotations

from typing import Any

import numpy as np

from .datasets import FACTOR_COLUMNS, REGULATORY_COLUMNS
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
        if self.action_dim >= 5:
            action = np.concatenate(
                [action, np.asarray([berth, yard], dtype=np.float32)]
            )
        if self.action_dim >= 7:
            action = np.concatenate([action, np.zeros(2, dtype=np.float32)])
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


class LegacyV3PolicyAdapter:
    """Run an unchanged five-action V3 policy inside V4 with neutral new actions.

    The adapter removes V4-only regulatory observations, forwards the exact V3
    observation contract to the legacy policy, and appends two zero commands.
    It is a compatibility comparator, not a claim that the legacy policy was
    trained for inspections.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.v3_prefix = 2 + 7 + 2 * len(FACTOR_COLUMNS)
        self.v4_regulatory_width = 2 * len(REGULATORY_COLUMNS)

    def predict(self, observation: Any, deterministic: bool = True):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        expected = self.v3_prefix + self.v4_regulatory_width + 4 + 4
        if obs.size != expected:
            raise ValueError(
                f"legacy V3 adapter requires V4 observation size {expected}; got {obs.size}"
            )
        base_state_start = self.v3_prefix + self.v4_regulatory_width
        v3_observation = np.concatenate(
            [obs[: self.v3_prefix], obs[base_state_start : base_state_start + 4]]
        )
        action, state = self.policy.predict(
            v3_observation, deterministic=deterministic
        )
        legacy_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if legacy_action.size != 5:
            raise ValueError("legacy V3 policy must produce exactly five actions")
        return np.concatenate([legacy_action, np.zeros(2, dtype=np.float32)]), state

    def parameters(self) -> dict[str, Any]:
        base = (
            self.policy.parameters()
            if hasattr(self.policy, "parameters")
            else {"implementation": type(self.policy).__name__}
        )
        return {
            "adapter": "legacy_v3_observation_slice_plus_neutral_v4_actions",
            "legacy_policy": base,
            "inspection_buffer": 0.0,
            "recovery_priority": 0.0,
            "legacy_artifact_modified": False,
        }
