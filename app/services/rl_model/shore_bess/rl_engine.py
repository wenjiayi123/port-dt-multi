"""Compatibility entry point for the checked-in Shore+BESS SAC model.

The original trainer was accidentally committed as ``rl_engine_副本.py`` while
the inference API imports ``rl_engine``.  Keep the historical trainer untouched
and expose only the model classes needed by the runtime.
"""

from importlib import import_module

import numpy as np

_legacy_engine = import_module(
    "app.services.rl_model.shore_bess.rl_engine_副本"
)

MLP = _legacy_engine.MLP


class GaussianPolicy(_legacy_engine.GaussianPolicy):
    """Add the loader expected by the runtime to the historical trainer class."""

    @staticmethod
    def load(data):
        net_data = data["net"]
        policy = GaussianPolicy(
            state_dim=int(net_data["input_dim"]),
            act_dim=int(net_data["output_dim"]),
            hidden=list(net_data["hidden"]),
            seed=42,
            init_std=1.0,
        )
        policy.net = MLP.load(net_data)
        policy.log_std = np.asarray(data["log_std"], dtype=np.float32)
        policy.log_std_m = np.zeros_like(policy.log_std)
        policy.log_std_v = np.zeros_like(policy.log_std)
        policy.opt_t = 0
        return policy

__all__ = ["GaussianPolicy", "MLP"]
