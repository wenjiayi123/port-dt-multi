"""Clone-safe Shore+BESS policy classes used by the inference runtime.

The historical trainer remains outside the release package, but runtime model
loading must never depend on a workstation-only backup file.  These NumPy
classes implement the exact serialized MLP/Gaussian actor contract required by
``api.PolicyRunner`` and the checked-in ``policy.bin``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class MLP:
    """Small tanh MLP compatible with the historical JSON serialization."""

    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        output_dim: int,
        seed: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden = [int(width) for width in hidden]
        self.output_dim = int(output_dim)
        rng = np.random.RandomState(seed)
        dims = [self.input_dim, *self.hidden, self.output_dim]
        self.params: List[Dict[str, np.ndarray]] = []
        for left, right in zip(dims, dims[1:]):
            weights = (
                rng.randn(left, right).astype(np.float32)
                / np.sqrt(max(1.0, float(left)))
            )
            bias = np.zeros((right,), dtype=np.float32)
            self.params.append({"W": weights, "B": bias})

    def forward_with_hidden(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        activations = np.asarray(x, dtype=np.float32)
        caches: List[Dict[str, np.ndarray]] = []
        for index, parameters in enumerate(self.params):
            logits = activations @ parameters["W"] + parameters["B"]
            caches.append({"A": activations, "Z": logits})
            activations = (
                np.tanh(logits)
                if index < len(self.params) - 1
                else logits
            )
        return activations, caches

    def save(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "output_dim": self.output_dim,
            "params": [
                {"W": item["W"].tolist(), "B": item["B"].tolist()}
                for item in self.params
            ],
        }

    @staticmethod
    def load(data: Dict[str, Any]) -> "MLP":
        model = MLP(
            input_dim=int(data["input_dim"]),
            hidden=[int(width) for width in data["hidden"]],
            output_dim=int(data["output_dim"]),
        )
        serialized = data.get("params") or []
        if len(serialized) != len(model.params):
            raise ValueError("serialized MLP layer count does not match architecture")
        for index, parameters in enumerate(serialized):
            weights = np.asarray(parameters["W"], dtype=np.float32)
            bias = np.asarray(parameters["B"], dtype=np.float32)
            if weights.shape != model.params[index]["W"].shape:
                raise ValueError(f"serialized MLP weight shape mismatch at layer {index}")
            if bias.shape != model.params[index]["B"].shape:
                raise ValueError(f"serialized MLP bias shape mismatch at layer {index}")
            model.params[index] = {"W": weights, "B": bias}
        return model


class GaussianPolicy:
    """Gaussian actor with deterministic mean inference and JSON loading."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden: List[int],
        seed: int,
        init_std: float,
    ) -> None:
        self.net = MLP(state_dim, hidden, act_dim, seed=seed)
        initial = np.log(max(1e-6, float(init_std)))
        self.log_std = np.full((act_dim,), initial, dtype=np.float32)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, _ = self.net.forward_with_hidden(state)
        std = np.exp(self.log_std)[None, :]
        return mean, std

    def save(self) -> Dict[str, Any]:
        return {"net": self.net.save(), "log_std": self.log_std.tolist()}

    @staticmethod
    def load(data: Dict[str, Any]) -> "GaussianPolicy":
        network = data["net"]
        policy = GaussianPolicy(
            state_dim=int(network["input_dim"]),
            act_dim=int(network["output_dim"]),
            hidden=[int(width) for width in network["hidden"]],
            seed=42,
            init_std=1.0,
        )
        policy.net = MLP.load(network)
        log_std = np.asarray(data["log_std"], dtype=np.float32)
        if log_std.shape != (policy.net.output_dim,):
            raise ValueError("serialized policy log_std shape does not match action dimension")
        policy.log_std = log_std
        return policy


__all__ = ["GaussianPolicy", "MLP"]
