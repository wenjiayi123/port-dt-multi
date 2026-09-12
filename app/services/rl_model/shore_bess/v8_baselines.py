"""Causal, non-neural comparators for the frozen V8 31D observation contract.

The carbon-aware rule uses current observations and TRAIN-ONLY hourly price /
carbon means. It never reads a future row, does not train a neural network, and
must never be reported as an RL teacher, learned policy or optimizer update.
Both comparators return the same discrete action lattice used by the V8 DQN.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from app.services.rl_model.shore_bess.v8_environment import LATTICE, ShoreBESSV8Env
from app.services.rl_training.datasets import PortDataset


@dataclass(frozen=True)
class TrainingHourlyProfile:
    """Only complete training rows may contribute to these frozen statistics."""

    hourly_price: tuple[float, ...]
    hourly_carbon: tuple[float, ...]
    price_center: float
    carbon_center: float
    carbon_span: float
    load_center: float
    load_std: float
    soft_cap_kw: float
    soc_initial: float
    rated_power_kw: float
    flexible_aux_limit: float
    defer_limit_kw: float
    source_dataset_sha256: str
    training_first_timestamp: str
    training_last_timestamp: str
    training_rows: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_training_hourly_profile(
    dataset: PortDataset, train_slice: slice, *, config: dict[str, Any] | None = None,
) -> TrainingHourlyProfile:
    """Estimate transparent calendar profiles; reject any post-April 2025 rows."""
    if train_slice.start not in (None, 0) or train_slice.stop is None or train_slice.step not in (None, 1):
        raise ValueError("baseline statistics require an explicit chronological training prefix")
    timestamps = dataset.timestamps[train_slice]
    if not timestamps or timestamps[-1] >= "2025-05-01T00:00:00Z":
        raise ValueError("May 2025 onward is reserved for validation/evaluation")
    if timestamps[-1] != "2025-04-30T23:00:00Z":
        raise ValueError("V8 comparator expects complete source months through April 2025")
    values = dataset.values[train_slice].astype(np.float64)
    hours = np.asarray([int(stamp[11:13]) for stamp in timestamps], dtype=np.int64)
    if set(hours) != set(range(24)):
        raise ValueError("training profile must cover all 24 UTC hours")
    kwargs = {"config": config} if config is not None else {}
    env = ShoreBESSV8Env(dataset, train_slice, normalization_slice=train_slice,
                        training=False, discrete=True, **kwargs)
    try:
        return TrainingHourlyProfile(
            hourly_price=tuple(float(values[hours == hour, 4].mean()) for hour in range(24)),
            hourly_carbon=tuple(float(values[hours == hour, 5].mean()) for hour in range(24)),
            price_center=env.price_center, carbon_center=env.carbon_center,
            carbon_span=env.carbon_span, load_center=env.load_center, load_std=env.load_std,
            soft_cap_kw=env.soft_cap_kw, soc_initial=env.soc_initial,
            rated_power_kw=env.power_kw,
            flexible_aux_limit=env.flex_limit, defer_limit_kw=env.defer_limit_kw,
            source_dataset_sha256=dataset.fingerprint,
            training_first_timestamp=timestamps[0], training_last_timestamp=timestamps[-1],
            training_rows=len(timestamps),
        )
    finally:
        env.close()


def lattice_index(action: np.ndarray) -> int:
    """Use an explicit nearest-neighbor mapping for historical continuous rules."""
    return int(np.argmin(np.sum((LATTICE - action) ** 2, axis=1)))


def decode_current(observation: np.ndarray, profile: TrainingHourlyProfile) -> dict[str, float]:
    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (31,) or not np.isfinite(obs).all():
        raise ValueError("the causal comparator requires one finite V8 31D observation")
    hour = int(round(float(np.arctan2(obs[0], obs[1])) * 24 / (2 * np.pi))) % 24
    return {
        "hour": hour,
        "base_kw": float(obs[2] * profile.load_std + profile.load_center),
        "auxiliary_kw": float(obs[3] * 400.0),
        "price": float(obs[4] * .35 + profile.price_center),
        "carbon": float(obs[6] * profile.carbon_span + profile.carbon_center),
        "soc": float(obs[7] * .18 + profile.soc_initial),
        "backlog_kwh": max(0.0, float(obs[11] * 2000.0)),
        "baseline_peak_kw": float(obs[16] * profile.load_std + profile.load_center),
    }


def legacy_peak_valley_action(observation: np.ndarray, profile: TrainingHourlyProfile) -> int:
    """Historical economic rule, transparently quantized to the V8 action space."""
    current = decode_current(observation, profile)
    base, price, soc = current["base_kw"], current["price"], current["soc"]
    if base > profile.soft_cap_kw + 500.0 and soc > profile.soc_initial - .12:
        bess = min(.55, max(0.0, (base - profile.soft_cap_kw) / profile.rated_power_kw))
    elif price >= .95 and soc > profile.soc_initial - .10:
        bess = .55
    elif price <= .50 and soc < profile.soc_initial + .10 and base + 500.0 < profile.soft_cap_kw:
        bess = -.70
    else:
        bess = 0.0
    return lattice_index(np.asarray([bess, 0.0], dtype=np.float32))


def causal_carbon_aware_action(
    observation: np.ndarray,
    profile: TrainingHourlyProfile,
    *,
    carbon_weight: float = 8.0,
    future_quantile: float = .5,
    horizon_hours: int = 12,
    max_backlog_hours: float = 3.0,
) -> int:
    """A calendar-threshold flex rule; fixed zero BESS avoids lossy arbitrage.

    Hour profiles are fitted only to training data. Current carbon/price/load,
    backlog and running baseline demand come solely from the observation.
    Positive-flex requests respect observed demand headroom; the SAME frozen
    V8 FIFO projection still has final authority over overdue work and limits.
    This is a conventional rule and does not establish neural performance.
    """
    if not 0.0 <= future_quantile <= 1.0 or not 1 <= horizon_hours <= 12:
        raise ValueError("invalid rule quantile or horizon")
    if (not np.isfinite(carbon_weight) or carbon_weight < 0
            or not np.isfinite(max_backlog_hours) or max_backlog_hours <= 0):
        raise ValueError("rule weights and backlog limit must be finite and positive")
    current = decode_current(observation, profile)
    hour = int(current["hour"])
    future_score = np.asarray([
        profile.hourly_price[(hour + offset) % 24]
        + carbon_weight * (profile.hourly_carbon[(hour + offset) % 24] - profile.carbon_center)
        for offset in range(1, horizon_hours + 1)
    ])
    threshold = float(np.quantile(future_score, future_quantile))
    score = current["price"] + carbon_weight * (current["carbon"] - profile.carbon_center)
    backlog = current["backlog_kwh"]
    capacity = current["auxiliary_kw"] * profile.flexible_aux_limit
    # A small tolerance removes float32 calendar ties; it is not an admission tolerance.
    repay = backlog > 1e-6 and score <= threshold + 1e-5
    if current["base_kw"] + min(backlog, capacity) > max(current["baseline_peak_kw"], current["base_kw"]) + 1e-5:
        repay = False
    if repay:
        flex = 1.0
    elif score > threshold + 1e-5 and backlog < profile.defer_limit_kw * max_backlog_hours:
        flex = -1.0
    else:
        flex = 0.0
    return lattice_index(np.asarray([0.0, flex], dtype=np.float32))


class RuleBaseline:
    """SB3-compatible predict interface; contains no neural weights or training."""

    def __init__(self, profile: TrainingHourlyProfile, *, kind: str = "carbon_aware", **parameters: Any):
        if kind not in {"carbon_aware", "legacy_peak_valley", "idle"}:
            raise ValueError("unknown V8 baseline kind")
        self.profile = profile
        self.kind = kind
        defaults = {"carbon_weight": 8.0, "future_quantile": .5,
                    "horizon_hours": 12, "max_backlog_hours": 3.0}
        self.parameters = {**defaults, **parameters} if kind == "carbon_aware" else {}

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        del deterministic
        observations = np.asarray(observation)
        if observations.ndim == 2:
            return np.asarray([self.predict(row)[0] for row in observations], dtype=np.int64), None
        if self.kind == "idle":
            action = lattice_index(np.zeros(2, dtype=np.float32))
        elif self.kind == "legacy_peak_valley":
            action = legacy_peak_valley_action(observations, self.profile)
        else:
            action = causal_carbon_aware_action(observations, self.profile, **self.parameters)
        return np.int64(action), None

    def provenance(self) -> dict[str, Any]:
        return {"kind": self.kind, "is_neural_policy": False, "optimizer_updates": 0,
                "source": "current observation plus train-only hourly profile",
                "historical_rule_quantized_to_v8_lattice": self.kind == "legacy_peak_valley",
                "parameters": self.parameters, "training_profile": self.profile.as_dict()}
