"""Explicit undiscounted n-step replay for Shore+BESS credit experiments.

This is an off-policy n-step approximation, not an unchanged vanilla SAC/TD3
algorithm. Intermediate rewards come from the recorded behavior trajectory;
there is no importance correction. SAC's intermediate entropy terms are not
included, so SAC use is additionally an approximate soft target. These limits
must accompany an experiment report. No teacher or future exogenous row is
consulted: only already observed transitions enter each return.

Only gamma=1 is supported. SB3 2.3 algorithms apply their own single gamma to
the endpoint bootstrap; gamma=1 makes that correct for every stored horizon,
including partial terminal windows. A different discount requires an algorithm
whose target explicitly consumes each sampled transition's horizon.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class _Transition:
    observation: np.ndarray
    next_observation: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    timeout: bool


class NStepReplayBuffer(ReplayBuffer):
    """One-environment replay with exact reward sums and episode-local queues.

    SB3's off-policy collector already replaces VecEnv autoreset observations
    with the real terminal observation before calling add(). This buffer uses
    that supplied next_obs verbatim and preserves TimeLimit timeout semantics.
    A done signal flushes every pending origin, including short terminal tails.
    An explicit nonterminal collection boundary can call flush_pending(); an
    unannounced reset is rejected if it breaks observed-state continuity.
    """

    approximation = {
        "return": "sum of observed behavior rewards with undiscounted endpoint bootstrap",
        "off_policy_importance_correction": False,
        "sac_intermediate_entropy_terms_included": False,
        "vanilla_algorithm_equivalence_claimed": False,
    }

    def __init__(self, buffer_size, observation_space, action_space, device="auto", n_envs=1,
                 optimize_memory_usage=False, handle_timeout_termination=True, *, n_step=24, gamma=1.0):
        if not isinstance(n_step, (int, np.integer)) or isinstance(n_step, bool) or n_step < 1:
            raise ValueError("n_step must be a positive integer")
        if not np.isfinite(gamma) or gamma != 1.0:
            raise ValueError("NStepReplayBuffer requires gamma=1 and an undiscounted SB3 learner")
        if n_envs != 1:
            raise ValueError("NStepReplayBuffer currently supports exactly one environment")
        if optimize_memory_usage:
            raise ValueError("n-step endpoint observations require optimize_memory_usage=False")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._pending: deque[_Transition] = deque()
        self.observed_transition_count = 0
        self.emitted_transition_count = 0
        self.discarded_pending_count = 0
        self.reset_count = 0
        self._reset_statistics()
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs,
                         optimize_memory_usage=False, handle_timeout_termination=handle_timeout_termination)
        self.n_step_horizons = np.zeros((self.buffer_size, 1), dtype=np.int32)

    def _reset_statistics(self) -> None:
        self.observed_terminal_count = self.observed_timeout_count = 0
        self.emitted_terminal_count = self.emitted_timeout_count = 0
        self.sampled_terminal_count = self.sampled_timeout_count = 0
        self.sampled_batch_count = 0
        self._emitted_horizon_counts = np.zeros(self.n_step + 1, dtype=np.int64)
        self._sampled_horizon_counts = np.zeros(self.n_step + 1, dtype=np.int64)
        self.flush_counts: dict[str, int] = {}
        self.last_flush: dict[str, Any] | None = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _emit_oldest(self) -> None:
        transitions = list(self._pending)[:self.n_step]
        if not transitions:
            return
        first, last = transitions[0], transitions[-1]
        reward_sum = float(sum(transition.reward for transition in transitions))
        if not np.isfinite(reward_sum):
            raise ValueError("non-finite n-step reward sum")
        position = self.pos
        super().add(first.observation, last.next_observation, first.action,
                    np.asarray([reward_sum], dtype=np.float32),
                    np.asarray([last.done], dtype=np.float32),
                    [{"TimeLimit.truncated": last.timeout}])
        self.n_step_horizons[position, 0] = len(transitions)
        self.emitted_transition_count += 1
        self._emitted_horizon_counts[len(transitions)] += 1
        self.emitted_terminal_count += int(last.done and not last.timeout)
        self.emitted_timeout_count += int(last.timeout)
        self._pending.popleft()

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        rewards, dones = np.asarray(reward), np.asarray(done)
        if rewards.size != 1 or dones.size != 1 or len(infos) != 1:
            raise ValueError("n-step add requires exactly one vectorized transition")
        observation = np.asarray(obs).copy()
        next_observation = np.asarray(next_obs).copy()
        copied_action = np.asarray(action).copy()
        reward_value = float(rewards.item())
        done_value = float(dones.item())
        if (not np.isfinite(reward_value) or not np.isfinite(done_value)
                or done_value not in (0.0, 1.0)
                or not np.isfinite(observation).all() or not np.isfinite(next_observation).all()
                or not np.isfinite(copied_action).all()):
            raise ValueError("finite observations, action, reward and binary done are required")
        timeout = bool(infos[0].get("TimeLimit.truncated", False))
        if timeout and not done_value:
            raise ValueError("a timeout must also end the collected episode")
        if self._pending and not np.array_equal(self._pending[-1].next_observation, observation):
            raise ValueError("transition discontinuity: reset/episode boundary lacks done or an explicit flush")
        self._pending.append(_Transition(observation, next_observation, copied_action,
                                         reward_value, bool(done_value), timeout))
        self.observed_transition_count += 1
        self.observed_terminal_count += int(bool(done_value) and not timeout)
        self.observed_timeout_count += int(timeout)
        if done_value:
            self.flush_pending(reason="episode_end")
        elif len(self._pending) >= self.n_step:
            self._emit_oldest()

    def flush_pending(self, *, reason: str = "collection_boundary") -> int:
        """Store short tails using the last *observed* endpoint and done flag.

        At an ordinary budget stop, done stays false and the learner bootstraps
        from that observed state. This never invents a terminal signal.
        Flushing only stores transitions; it never calls an optimizer. A final
        budget flush does not establish that those new rows were sampled.
        """
        if reason not in {"episode_end", "collection_boundary", "training_budget_end"}:
            raise ValueError("unregistered n-step flush reason")
        emitted = len(self._pending)
        endpoint = self._pending[-1] if self._pending else None
        while self._pending:
            self._emit_oldest()
        self.flush_counts[reason] = self.flush_counts.get(reason, 0) + 1
        self.last_flush = {"reason": reason, "observed_transitions": self.observed_transition_count,
                           "newly_stored_transitions": emitted,
                           "endpoint_done": endpoint.done if endpoint else None,
                           "endpoint_timeout": endpoint.timeout if endpoint else None,
                           "storage_only_no_optimizer_update_performed": True}
        return emitted

    def _get_samples(self, batch_inds: np.ndarray, env=None):
        """Count actual replay draws, including repeats; this is not update count."""
        samples = super()._get_samples(batch_inds, env=env)
        horizons = self.n_step_horizons[batch_inds, 0]
        self._sampled_horizon_counts += np.bincount(horizons, minlength=self.n_step + 1)
        self.sampled_batch_count += 1
        dones = self.dones[batch_inds, 0].astype(bool)
        timeouts = self.timeouts[batch_inds, 0].astype(bool)
        self.sampled_terminal_count += int(np.count_nonzero(dones & ~timeouts))
        self.sampled_timeout_count += int(np.count_nonzero(timeouts))
        return samples

    def discard_pending(self) -> int:
        """Explicitly drop unfinished origins when their endpoint is unusable."""
        discarded = len(self._pending)
        self._pending.clear()
        self.discarded_pending_count += discarded
        return discarded

    def reset(self) -> None:
        self.reset_count += 1
        self._pending.clear()
        self.observed_transition_count = self.emitted_transition_count = self.discarded_pending_count = 0
        self._reset_statistics()
        super().reset()
        if hasattr(self, "n_step_horizons"):
            self.n_step_horizons.fill(0)

    def describe(self) -> dict[str, Any]:
        def histogram(counts) -> dict[str, int]:
            return {str(horizon): int(count) for horizon, count in enumerate(counts) if count}

        size = self.size()
        resident_horizons = np.bincount(self.n_step_horizons[:size, 0], minlength=self.n_step + 1)
        resident_dones = self.dones[:size, 0].astype(bool)
        resident_timeouts = self.timeouts[:size, 0].astype(bool)
        return {"class": type(self).__name__, "n_step": self.n_step, "gamma": self.gamma,
                "statistics_scope": "since_last_replay_buffer_reset", "reset_count": self.reset_count,
                "n_envs": self.n_envs, "handle_timeout_termination": self.handle_timeout_termination,
                "optimize_memory_usage": self.optimize_memory_usage,
                "observed_transitions": self.observed_transition_count,
                "emitted_transitions": self.emitted_transition_count,
                "pending_transitions": self.pending_count,
                "discarded_pending_transitions": self.discarded_pending_count,
                "transition_accounting_balanced": self.observed_transition_count == (
                    self.emitted_transition_count + self.pending_count + self.discarded_pending_count),
                "emitted_horizon_counts": histogram(self._emitted_horizon_counts),
                "emitted_horizon_accounting_balanced": int(self._emitted_horizon_counts.sum()) == self.emitted_transition_count,
                "resident_transitions": size, "resident_horizon_counts": histogram(resident_horizons),
                "resident_position": self.pos, "resident_full": self.full,
                "observed_terminal_events": self.observed_terminal_count,
                "observed_timeout_events": self.observed_timeout_count,
                "emitted_terminal_endpoints": self.emitted_terminal_count,
                "emitted_timeout_endpoints": self.emitted_timeout_count,
                "resident_terminal_endpoints": int(np.count_nonzero(resident_dones & ~resident_timeouts)),
                "resident_timeout_endpoints": int(np.count_nonzero(resident_timeouts)),
                "sampled_batches": self.sampled_batch_count,
                "sampled_transition_draws": int(self._sampled_horizon_counts.sum()),
                "sampled_horizon_counts": histogram(self._sampled_horizon_counts),
                "sampled_terminal_endpoints": self.sampled_terminal_count,
                "sampled_timeout_endpoints": self.sampled_timeout_count,
                "sampling_count_basis": "replay draws with replacement, not unique transitions or optimizer updates",
                "flush_counts": dict(self.flush_counts), "last_flush": dict(self.last_flush) if self.last_flush else None,
                "flush_claim_boundary": "Storage only. Final emitted tails are not asserted to have been sampled or learned.",
                "approximation": dict(self.approximation)}
