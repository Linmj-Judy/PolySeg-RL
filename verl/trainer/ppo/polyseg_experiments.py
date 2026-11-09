"""Small, dependency-free validation suite for the PolySeg methods.

This is intentionally a CPU reference implementation.  It is used to test the
statistical contracts (smooth features, macro compression, Bellman residuals,
feedback efficiency and calibration) without requiring a language model or a
rollout service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from verl.trainer.ppo.polyseg import (
    PolySegRegretTracker,
    PolynomialSegmentCritic,
    bellman_closure_residual,
    segment_comparator_regret,
)


@dataclass(frozen=True)
class SmoothMDPConfig:
    horizon: int = 8
    state_dim: int = 2
    action_count: int = 2
    transition_noise: float = 0.01
    seed: int = 0


class SmoothPolynomialSegmentMDP:
    """Bounded smooth continuous-state MDP used by the Methods experiment plan."""

    def __init__(self, config: SmoothMDPConfig = SmoothMDPConfig()) -> None:
        if config.horizon < 1 or config.state_dim < 1 or config.action_count < 2:
            raise ValueError("invalid synthetic MDP dimensions")
        if config.transition_noise < 0:
            raise ValueError("transition_noise must be non-negative")
        self.config = config
        generator = torch.Generator().manual_seed(int(config.seed))
        self.transition_matrix = torch.randn((config.state_dim, config.state_dim), generator=generator) * 0.25
        self.action_matrix = torch.randn((config.state_dim, config.action_count), generator=generator) * 0.25

    def reset(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.empty(self.config.state_dim).uniform_(-0.5, 0.5, generator=generator)

    def reward(self, state: torch.Tensor, action: int) -> torch.Tensor:
        action_value = (float(action) / max(self.config.action_count - 1, 1)) * 2.0 - 1.0
        smooth = 0.35 * torch.cos(state).mean() + 0.25 * torch.sin(state.sum() + action_value)
        return (0.5 + 0.25 * smooth).clamp(0.0, 1.0)

    def transition(self, state: torch.Tensor, action: int, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(state.shape, generator=generator) * self.config.transition_noise
        return torch.tanh(self.transition_matrix @ state + self.action_matrix[:, int(action)] + noise)

    def rollout(
        self,
        policy: Callable[[torch.Tensor, int], int],
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.reset(generator)
        states, actions, rewards = [], [], []
        for step in range(self.config.horizon):
            action = int(policy(state, step)) % self.config.action_count
            states.append(state)
            actions.append(action)
            rewards.append(self.reward(state, action))
            state = self.transition(state, action, generator)
        return torch.stack(states), torch.tensor(actions, dtype=torch.long), torch.stack(rewards)


def calibration_metrics(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> dict[str, float]:
    """Return Brier score and expected calibration error for binary feedback."""
    p = torch.as_tensor(probabilities, dtype=torch.float32).flatten().clamp(0.0, 1.0)
    y = torch.as_tensor(labels, dtype=torch.float32, device=p.device).flatten()
    if p.shape != y.shape or p.numel() == 0:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    edges = torch.linspace(0.0, 1.0, max(int(bins), 1) + 1, device=p.device)
    ece = p.new_zeros(())
    for i in range(len(edges) - 1):
        mask = (p >= edges[i]) & (p <= edges[i + 1] if i == len(edges) - 2 else p < edges[i + 1])
        if torch.any(mask):
            ece = ece + mask.float().mean() * (p[mask].mean() - y[mask].mean()).abs()
    return {"brier": float((p - y).square().mean()), "ece": float(ece)}


def run_synthetic_polyseg_experiment(
    *,
    episodes: int = 32,
    q: int = 2,
    m: int = 4,
    config: SmoothMDPConfig | None = None,
) -> dict[str, float]:
    """Run the documented synthetic smooth-MDP checks and return scalar metrics."""
    mdp = SmoothPolynomialSegmentMDP(config or SmoothMDPConfig())
    if episodes < 2 or not 1 <= m <= mdp.config.horizon:
        raise ValueError("episodes must be at least two and m must fit the horizon")
    generator = torch.Generator().manual_seed(int(mdp.config.seed) + 17)
    boundaries = torch.tensor(np.linspace(0, mdp.config.horizon, m + 1, dtype=np.int64))
    features, targets, rewards, labels, probs = [], [], [], [], []

    def random_policy(_state: torch.Tensor, _step: int) -> int:
        return int(torch.randint(mdp.config.action_count, (), generator=generator))

    for _ in range(episodes):
        states, actions, step_rewards = mdp.rollout(random_policy, generator=generator)
        segment_features = []
        segment_targets = []
        for j in range(m):
            start, end = int(boundaries[j]), int(boundaries[j + 1])
            state = states[start:end].mean(0).clamp(-1.0, 1.0)
            segment_features.append(state)
            segment_targets.append(step_rewards[start:end].flip(0).cumsum(0).flip(0)[0])
        features.append(torch.stack(segment_features))
        targets.append(torch.stack(segment_targets))
        rewards.append(step_rewards.sum())
        labels.extend((step_rewards > 0.5).float().tolist())
        probs.extend(step_rewards.tolist())

    feature_tensor = torch.stack(features)
    target_tensor = torch.stack(targets)
    critic = PolynomialSegmentCritic(q=q, feature_dim=mdp.config.state_dim, feedback="sum", ridge_lambda=1e-2)
    critic.fit(feature_tensor.reshape(-1, mdp.config.state_dim), target_tensor.reshape(-1))
    prediction = critic.predict(feature_tensor.reshape(-1, mdp.config.state_dim)).reshape_as(target_tensor)
    next_values = torch.cat((prediction[:, 1:], torch.zeros_like(prediction[:, :1])), dim=1)
    closure = bellman_closure_residual(target_tensor, next_values, prediction)
    # The max-action reward is a transparent proxy comparator for this smoke
    # experiment; a real task can supply an exact oracle trajectory instead.
    oracle_rewards = []
    for states in feature_tensor:
        total = 0.0
        for state in states:
            action_values = torch.stack([mdp.reward(state, a) for a in range(mdp.config.action_count)])
            total += float(action_values.max())
        oracle_rewards.append(total)
    oracle = torch.tensor(oracle_rewards)
    observed = torch.stack(rewards)
    regret = segment_comparator_regret(oracle, observed)
    tracker = PolySegRegretTracker()
    for expected, actual in zip(oracle.tolist(), observed.tolist(), strict=True):
        tracker.update(comparator_value=expected, policy_value=actual)
    calibration = calibration_metrics(torch.tensor(probs), torch.tensor(labels))
    return {
        "critic_mse": float((prediction - target_tensor).square().mean()),
        "bellman_closure_residual": float(closure),
        "segment_regret": float(regret),
        "queries": float(episodes * m),
        "queries_per_episode": float(m),
        "accuracy_per_query": float(observed.mean() / max(episodes * m, 1)),
        "brier": calibration["brier"],
        "ece": calibration["ece"],
        "tracker_episodes": float(tracker.episodes),
    }


__all__ = [
    "SmoothMDPConfig",
    "SmoothPolynomialSegmentMDP",
    "calibration_metrics",
    "run_synthetic_polyseg_experiment",
]
