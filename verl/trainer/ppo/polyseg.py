"""Polynomial Segment Reinforcement Learning primitives.

This module contains the parts of PolySeg-RL that are independent of a
particular rollout backend: Legendre features, segment construction,
budgeted feedback, ridge/logistic feedback estimators, macro GAE and the
budget-aware ``(q, m)`` selector.  The tensor APIs intentionally mirror verl's
batch-first convention, so the functions can be used from both the legacy and
v1 trainers.

The implementation follows ``paper_methods.md``.  In particular, a segment
reward is one macro reward; it is never copied to every token as an additive
reward.  Token values returned by :func:`compute_segment_gae_advantage` are a
scatter view of macro advantages and are suitable for the existing token PPO
loss.
"""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import fcntl
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence
from numbers import Real
from uuid import uuid4

import numpy as np
import torch


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_real(value: Any, name: str, *, minimum: float = 0.0, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    numeric = float(value)
    if (strictly_positive and numeric <= minimum) or (not strictly_positive and numeric < minimum):
        raise ValueError(f"{name} must be {'>' if strictly_positive else '>='} {minimum}")
    return numeric


def _strict_nonnegative_int(value: Any, name: str) -> int:
    """Parse a checkpoint counter without truncating fractional values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


__all__ = [
    "POLYSEG_FEATURE_MAP_VERSION",
    "PolySegConfig",
    "PolySegChoice",
    "SegmentRewardBudget",
    "FileBackedSegmentRewardBudget",
    "PolynomialSegmentCritic",
    "PolynomialEstimatorState",
    "PolySegDependenceState",
    "RidgeEstimatorState",
    "reset_polyseg_estimator_state",
    "polyseg_estimator_state_dict",
    "load_polyseg_estimator_state_dict",
    "reset_polyseg_dependence_state",
    "polyseg_dependence_state_dict",
    "load_polyseg_dependence_state_dict",
    "MLPSegmentCritic",
    "PolynomialTransitionEstimator",
    "MacroLSVIPlan",
    "MacroLSVIPlanner",
    "SegmentTrace",
    "SuffixFeedbackCollector",
    "enumerate_multi_indices",
    "legendre_features",
    "make_segment_slices",
    "segment_ids_from_lengths",
    "segment_ids",
    "pool_token_values",
    "scatter_segment_values",
    "suffix_potential_differences",
    "compute_segment_gae_advantage",
    "compute_polyseg_advantage",
    "choose_polynomial_segment_config",
    "BudgetedPolySegController",
    "segment_feedback_bound",
    "segment_log_prob_ratio",
    "segment_ppo_loss",
    "compute_segment_policy_loss",
    "macro_bellman_backup",
    "POLYSEG_PROTOCOLS",
    "normalize_feedback_protocol",
    "resolve_polyseg_option",
    "feedback_protocol_contract",
    "decode_token_ids",
    "extract_segment_metadata",
    "build_segment_features",
    "feature_map_contract",
    "cross_fitted_segment_residuals",
    "cross_fitted_logistic_segment_residuals",
    "estimate_dependence_diagnostics",
    "compute_segment_rewards",
    "macro_return_targets",
    "ridge_feature_norms",
    "confidence_bonus",
    "transition_confidence_bonus",
    "FrozenLowDimRepresentation",
    "load_frozen_representation",
    "aggregate_step_features",
    "SharedStepRewardEstimator",
    "recover_step_rewards_from_segment_feedback",
    "dependence_confidence_radius",
    "bellman_closure_residual",
    "segment_comparator_regret",
    "budget_comparator_regret",
    "compression_gap",
    "full_step_comparator_regret",
    "budget_feasible_segments",
    "unified_regret_bound",
    "PolySegRegretTracker",
]


# Feature coordinates are part of the statistical object being fit.  A
# checkpoint or result is not reproducible if it only records ``q`` and the
# basis while silently changing the boundary normalizer.  Keep a stable
# version string and expose a content hash for every run that builds these
# summaries.
POLYSEG_FEATURE_MAP_VERSION = "polyseg-fixed-boundary-summary-v1"


def feature_map_contract(
    *,
    response_length: int,
    feature_dim: int,
    feature_horizon: float | None = None,
    max_duration: float | None = None,
    max_action_tokens: float | None = None,
    max_tool_calls: float = 16.0,
) -> dict[str, Any]:
    """Return the immutable feature-map specification and its content hash.

    The defaults intentionally match :func:`build_segment_features`.  The
    hash includes effective scales rather than current-trajectory maxima, so
    the same contract can be checked after padding, replay, or checkpoint
    restore.  It is metadata only and never changes feature values.
    """

    if int(response_length) < 1 or int(feature_dim) < 1:
        raise ValueError("response_length and feature_dim must be positive")
    horizon = max(float(feature_horizon if feature_horizon is not None else response_length), 1.0)
    duration = max(float(max_duration if max_duration is not None else horizon), 1.0)
    action = max(float(max_action_tokens if max_action_tokens is not None else horizon), 1.0)
    tools = max(float(max_tool_calls), 1.0)
    base_coordinates = ["segment_start", "segment_end", "action_fraction", "duration_fraction"]
    coordinate_order = base_coordinates[: min(int(feature_dim), len(base_coordinates))]
    if int(feature_dim) >= 5:
        coordinate_order.append("tool_call_fraction")
    if int(feature_dim) > len(coordinate_order):
        coordinate_order.extend(["zero_pad"] * (int(feature_dim) - len(coordinate_order)))
    spec = {
        "version": POLYSEG_FEATURE_MAP_VERSION,
        "semantic_role": "rollout_observable_boundary_state_action_summary",
        "causal_availability": "available_at_segment_boundary",
        "markov_sufficiency": "not_established",
        "response_length": int(response_length),
        "feature_dim": int(feature_dim),
        "feature_horizon": horizon,
        "max_duration": duration,
        "max_action_tokens": action,
        "max_tool_calls": tools,
        "coordinate_order": coordinate_order,
        "out_of_range_policy": "clip_to_unit_interval",
        "padding_policy": "zero_pad_trailing_coordinates",
    }
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "version": POLYSEG_FEATURE_MAP_VERSION,
        "semantic_role": spec["semantic_role"],
        "causal_availability": spec["causal_availability"],
        "markov_sufficiency": spec["markov_sufficiency"],
        "hash": hashlib.sha256(payload).hexdigest(),
        "spec": spec,
    }


class PolynomialEstimatorState:
    """Serializable sufficient statistics for an accumulating macro critic.

    A rollout batch is usually too small to identify a high-order basis.  When
    ``PolySegConfig.accumulate_estimator`` is enabled, each batch contributes
    ``X.T @ X`` and ``X.T @ y`` to this object and the critic is solved from the
    aggregate.  Keeping sufficient statistics rather than all old rollouts
    makes the state cheap to checkpoint and avoids retaining token data.

    The same state also owns the vector-valued transition regression statistics;
    this is intentional because reward and transition fits must advance at the
    same trainer step and be restored atomically.
    """

    VERSION = 2

    def __init__(
        self,
        *,
        q: int,
        feature_dim: int,
        basis: str = "total",
        normalized_basis: bool = False,
        ridge_lambda: float = 1.0,
        transition_ridge_lambda: float | None = None,
        dependence_history_limit: int = 128,
        feedback: str = "local_sum",
    ):
        if q < 0 or feature_dim < 1:
            raise ValueError("q must be non-negative and feature_dim must be positive")
        if basis not in {"total", "additive"}:
            raise ValueError("basis must be 'total' or 'additive'")
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        if transition_ridge_lambda is None:
            transition_ridge_lambda = ridge_lambda
        if transition_ridge_lambda <= 0:
            raise ValueError("transition_ridge_lambda must be positive")
        if int(dependence_history_limit) < 0:
            raise ValueError("dependence_history_limit must be non-negative")
        self.q = int(q)
        self.feature_dim = int(feature_dim)
        self.basis = basis
        self.normalized_basis = bool(normalized_basis)
        self.ridge_lambda = float(ridge_lambda)
        self.transition_ridge_lambda = float(transition_ridge_lambda)
        self.dependence_history_limit = int(dependence_history_limit)
        # The sufficient statistics are protocol-specific.  Persisting the
        # canonical feedback name prevents an accumulating state created for
        # one reward semantics from being silently reused by another.
        self.feedback = normalize_feedback_protocol(feedback)
        self.sample_count = 0
        self.transition_sample_count = 0
        self.gram: torch.Tensor | None = None
        self.rhs: torch.Tensor | None = None
        self.transition_gram: torch.Tensor | None = None
        self.transition_rhs: torch.Tensor | None = None
        # Compact, bounded history for prequential cross-batch OOF
        # dependence diagnostics.  Entries contain polynomial design rows,
        # labels and observation weights, never token-level rollout data.
        self.dependence_history: list[dict[str, torch.Tensor]] = []

    @property
    def dimension(self) -> int:
        return (
            1 + self.feature_dim * self.q
            if self.basis == "additive"
            else math.comb(self.feature_dim + self.q, self.feature_dim)
        )

    def _check_features(self, features: torch.Tensor) -> torch.Tensor:
        x = _as_tensor(features).reshape(-1, self.dimension)
        if x.shape[1] != self.dimension:
            raise ValueError("feature dimension does not match estimator state")
        return x

    @staticmethod
    def _accumulate_matrix(
        current: torch.Tensor | None,
        update: torch.Tensor,
    ) -> torch.Tensor:
        return update if current is None else current.to(update) + update

    def update(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Add reward/value observations to the ridge sufficient statistics."""

        x = self._check_features(features)
        y = _as_tensor(targets, device=x.device).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and targets have incompatible sample counts")
        if sample_weight is not None:
            weight = _as_tensor(sample_weight, device=x.device).reshape(-1).clamp_min(0)
            if weight.shape != y.shape:
                raise ValueError("sample_weight shape mismatch")
            x = x * weight.sqrt().unsqueeze(-1)
            y = y * weight.sqrt()
        self.gram = self._accumulate_matrix(self.gram, x.T @ x)
        self.rhs = self._accumulate_matrix(self.rhs, x.T @ y)
        self.sample_count += int(x.shape[0])

    def solve(self) -> torch.Tensor:
        if self.gram is None or self.rhs is None:
            raise RuntimeError("estimator state has no reward observations")
        eye = torch.eye(self.dimension, dtype=self.gram.dtype, device=self.gram.device)
        return torch.linalg.solve(self.gram + self.ridge_lambda * eye, self.rhs)

    def update_transition(self, features: torch.Tensor, next_features: torch.Tensor) -> None:
        """Add adjacent-boundary features to the transition ridge fit."""

        x = self._check_features(features)
        y = _as_tensor(next_features, device=x.device)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and next_features have incompatible sample counts")
        self.transition_gram = self._accumulate_matrix(self.transition_gram, x.T @ x)
        self.transition_rhs = self._accumulate_matrix(self.transition_rhs, x.T @ y)
        self.transition_sample_count += int(x.shape[0])

    def solve_transition(self) -> torch.Tensor:
        if self.transition_gram is None or self.transition_rhs is None:
            raise RuntimeError("estimator state has no transition observations")
        eye = torch.eye(self.dimension, dtype=self.transition_gram.dtype, device=self.transition_gram.device)
        return torch.linalg.solve(self.transition_gram + self.transition_ridge_lambda * eye, self.transition_rhs)

    def append_dependence_history(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        score_residuals: torch.Tensor | None = None,
    ) -> None:
        """Store complete valid trajectories for future cross-batch OOF fits.

        The history is detached to CPU and bounded by
        ``dependence_history_limit``. It is used only by dependence
        diagnostics, so it cannot alter the policy-gradient target.
        """
        if self.dependence_history_limit == 0:
            return
        x = _as_tensor(features).detach()
        y = _as_tensor(targets, device=x.device).detach()
        mask = _as_tensor(valid_mask, device=x.device).to(torch.bool)
        if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape or mask.shape != y.shape:
            raise ValueError("dependence history tensors have incompatible shapes")
        if weights is None:
            w = torch.ones_like(y, dtype=torch.float32)
        else:
            w = _as_tensor(weights, device=x.device).detach().to(torch.float32)
            if w.shape != y.shape:
                raise ValueError("dependence history weights must match targets")
        if score_residuals is None:
            raise ValueError("score_residuals are required for persistent OOF history")
        scores = _as_tensor(score_residuals, device=x.device).detach()
        if scores.ndim != 3 or scores.shape[:2] != y.shape or scores.shape[-1] != self.dimension:
            raise ValueError("score_residuals must have shape [batch, segments, dimension]")
        for row in range(x.shape[0]):
            keep = mask[row]
            if not bool(keep.any()):
                continue
            self.dependence_history.append(
                {
                    "features": x[row][keep].cpu(),
                    "targets": y[row][keep].cpu(),
                    "weights": w[row][keep].cpu(),
                    "scores": scores[row][keep].cpu(),
                }
            )
        if len(self.dependence_history) > self.dependence_history_limit:
            self.dependence_history = self.dependence_history[-self.dependence_history_limit :]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "q": self.q,
            "feature_dim": self.feature_dim,
            "basis": self.basis,
            "normalized_basis": self.normalized_basis,
            "ridge_lambda": self.ridge_lambda,
            "transition_ridge_lambda": self.transition_ridge_lambda,
            "dependence_history_limit": self.dependence_history_limit,
            "feedback": self.feedback,
            "sample_count": self.sample_count,
            "transition_sample_count": self.transition_sample_count,
            "gram": None if self.gram is None else self.gram.detach().cpu(),
            "rhs": None if self.rhs is None else self.rhs.detach().cpu(),
            "transition_gram": None if self.transition_gram is None else self.transition_gram.detach().cpu(),
            "transition_rhs": None if self.transition_rhs is None else self.transition_rhs.detach().cpu(),
            "dependence_history": [
                {key: value.detach().cpu() for key, value in item.items()}
                for item in self.dependence_history
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "PolynomialEstimatorState":
        for name in ("q", "feature_dim", "basis"):
            if name in state and getattr(self, name) != state[name]:
                raise ValueError(f"estimator state {name} does not match configured estimator")
        if "normalized_basis" in state and bool(state["normalized_basis"]) != self.normalized_basis:
            raise ValueError("estimator normalized-basis setting does not match configured estimator")
        for name in ("ridge_lambda", "transition_ridge_lambda"):
            if name in state:
                value = float(state[name])
                expected = getattr(self, name)
                if not math.isfinite(value) or value <= 0 or not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"estimator state {name} does not match configured estimator")
        if "feedback" not in state:
            # Version-1 checkpoints predate the protocol field and are only
            # unambiguous for the historical local-sum default.  Refuse to
            # load them into any other feedback semantics instead of silently
            # reusing reward statistics under a different target.
            if self.feedback != "local_sum":
                raise ValueError("estimator state is missing feedback semantics")
        elif normalize_feedback_protocol(state["feedback"]) != self.feedback:
            raise ValueError("estimator state feedback does not match configured estimator")
        self.sample_count = _strict_nonnegative_int(state.get("sample_count", 0), "sample_count")
        self.transition_sample_count = _strict_nonnegative_int(
            state.get("transition_sample_count", 0), "transition_sample_count"
        )
        if "dependence_history_limit" in state:
            history_limit = _strict_nonnegative_int(state["dependence_history_limit"], "dependence_history_limit")
            if history_limit != self.dependence_history_limit:
                raise ValueError("estimator dependence history limit does not match configured state")
        for name in ("gram", "rhs", "transition_gram", "transition_rhs"):
            value = state.get(name)
            setattr(self, name, None if value is None else _as_tensor(value))
            if value is not None and not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"invalid non-finite values in estimator state {name}")
        if self.gram is not None and self.gram.shape != (self.dimension, self.dimension):
            raise ValueError("invalid reward Gram shape in estimator state")
        if self.rhs is not None and self.rhs.shape != (self.dimension,):
            raise ValueError("invalid reward target shape in estimator state")
        if self.transition_gram is not None and self.transition_gram.shape != (self.dimension, self.dimension):
            raise ValueError("invalid transition Gram shape in estimator state")
        if self.transition_rhs is not None and (self.transition_rhs.ndim != 2 or self.transition_rhs.shape[0] != self.dimension):
            raise ValueError("invalid transition target shape in estimator state")
        if self.sample_count < 0 or self.transition_sample_count < 0:
            raise ValueError("estimator sample counts must be non-negative")
        history = state.get("dependence_history", [])
        if not isinstance(history, (list, tuple)):
            raise ValueError("invalid dependence history")
        restored: list[dict[str, torch.Tensor]] = []
        for item in history:
            if not isinstance(item, Mapping) or not {"features", "targets", "weights", "scores"}.issubset(item):
                raise ValueError("invalid dependence history entry")
            x = _as_tensor(item["features"]).detach().cpu()
            y = _as_tensor(item["targets"]).detach().cpu().reshape(-1)
            w = _as_tensor(item["weights"]).detach().cpu().reshape(-1)
            scores = _as_tensor(item["scores"]).detach().cpu()
            if x.ndim != 2 or x.shape[1] != self.dimension or y.shape != w.shape or x.shape[0] != y.shape[0] or scores.shape != x.shape:
                raise ValueError("invalid dependence history entry shape")
            if (
                x.shape[0] == 0
                or not torch.isfinite(x).all()
                or not torch.isfinite(y).all()
                or not torch.isfinite(w).all()
                or not torch.isfinite(scores).all()
            ):
                raise ValueError("invalid dependence history values")
            restored.append({"features": x, "targets": y, "weights": w, "scores": scores})
        self.dependence_history = restored[-self.dependence_history_limit :] if self.dependence_history_limit else []
        return self

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PolynomialEstimatorState":
        obj = cls(
            q=int(state["q"]),
            feature_dim=int(state["feature_dim"]),
            basis=state.get("basis", "total"),
            normalized_basis=bool(state.get("normalized_basis", False)),
            ridge_lambda=float(state.get("ridge_lambda", 1.0)),
            transition_ridge_lambda=float(state.get("transition_ridge_lambda", state.get("ridge_lambda", 1.0))),
            dependence_history_limit=_strict_nonnegative_int(
                state.get("dependence_history_limit", 128), "dependence_history_limit"
            ),
            feedback=state.get("feedback", "local_sum"),
        )
        return obj.load_state_dict(state)


class PolySegDependenceState(PolynomialEstimatorState):
    """Persistent dependence diagnostics and online-radius state.

    This state is intentionally separate from the accumulating critic state.
    A confidence bonus must use the radius observed *before* the current
    batch; the current batch can update this object only after its advantage
    computation has completed successfully.  The inherited bounded history
    stores compact design/target/score rows for strict prequential OOF fits.
    """

    DEPENDENCE_VERSION = 1

    def __init__(self, *, feedback: str = "local_sum", **kwargs: Any):
        super().__init__(feedback=feedback, **kwargs)
        self.dependence_radius_multiplier = 1.0
        self.dependence_diagnostic_batches = 0

    def update_dependence_radius(self, multiplier: float) -> None:
        value = float(multiplier)
        if not math.isfinite(value) or value < 1.0:
            raise ValueError("dependence radius multiplier must be finite and at least one")
        self.dependence_radius_multiplier = value
        self.dependence_diagnostic_batches += 1

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "dependence_state_version": self.DEPENDENCE_VERSION,
                "dependence_radius_multiplier": self.dependence_radius_multiplier,
                "dependence_diagnostic_batches": self.dependence_diagnostic_batches,
                "feedback": self.feedback,
            }
        )
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PolySegDependenceState":
        obj = cls(
            q=int(state["q"]),
            feature_dim=int(state["feature_dim"]),
            basis=state.get("basis", "total"),
            normalized_basis=bool(state.get("normalized_basis", False)),
            ridge_lambda=float(state.get("ridge_lambda", 1.0)),
            transition_ridge_lambda=float(
                state.get("transition_ridge_lambda", state.get("ridge_lambda", 1.0))
            ),
            dependence_history_limit=_strict_nonnegative_int(
                state.get("dependence_history_limit", 128), "dependence_history_limit"
            ),
            feedback=state.get("feedback", "local_sum"),
        )
        PolynomialEstimatorState.load_state_dict(obj, state)
        obj.dependence_radius_multiplier = float(state.get("dependence_radius_multiplier", 1.0))
        obj.dependence_diagnostic_batches = _strict_nonnegative_int(
            state.get("dependence_diagnostic_batches", 0), "dependence_diagnostic_batches"
        )
        if not math.isfinite(obj.dependence_radius_multiplier) or obj.dependence_radius_multiplier < 1.0:
            raise ValueError("invalid dependence radius multiplier")
        if obj.dependence_diagnostic_batches < 0:
            raise ValueError("invalid dependence diagnostic batch count")
        return obj


# A short alias makes the intent clear to callers that use this as a generic
# ridge-regression accumulator.
RidgeEstimatorState = PolynomialEstimatorState


_POLYSEG_ESTIMATOR_STATES: dict[str, PolynomialEstimatorState] = {}
_POLYSEG_DEPENDENCE_STATES: dict[str, PolySegDependenceState] = {}


def _resolve_estimator_state(
    config: "PolySegConfig", state: PolynomialEstimatorState | None
) -> PolynomialEstimatorState | None:
    if not config.accumulate_estimator:
        return None
    if state is not None:
        return state
    state_id = config.estimator_state_id or "__polyseg_default__"
    expected = dict(
        q=config.q,
        feature_dim=config.feature_dim,
        basis=config.basis,
        normalized_basis=config.normalized_basis,
        ridge_lambda=config.ridge_lambda,
        transition_ridge_lambda=config.transition_ridge_lambda,
        dependence_history_limit=config.dependence_history_limit,
        feedback=normalize_feedback_protocol(config.feedback),
    )
    current = _POLYSEG_ESTIMATOR_STATES.get(state_id)
    if current is None:
        current = _POLYSEG_ESTIMATOR_STATES[state_id] = PolynomialEstimatorState(**expected)
    elif any(getattr(current, key) != value for key, value in expected.items()):
        raise ValueError(f"estimator_state_id={state_id!r} is already bound to a different PolySeg configuration")
    return current


def _resolve_dependence_state(config: "PolySegConfig") -> PolySegDependenceState | None:
    """Resolve the independent persistent state used by confidence bonuses."""

    if not config.use_confidence_bonus:
        return None
    expected = dict(
        q=config.q,
        feature_dim=config.feature_dim,
        basis=config.basis,
        normalized_basis=config.normalized_basis,
        ridge_lambda=config.ridge_lambda,
        transition_ridge_lambda=config.transition_ridge_lambda,
        dependence_history_limit=config.dependence_history_limit,
        feedback=normalize_feedback_protocol(config.feedback),
    )
    if config.dependence_state_id:
        state_id = config.dependence_state_id
    elif config.estimator_state_id:
        state_id = f"{config.estimator_state_id}:dependence"
    else:
        # Keep independent defaults for genuinely different estimator
        # geometries while allowing repeated calls in one run to share state.
        state_id = "__polyseg_dependence_default__:" + json.dumps(
            expected, sort_keys=True, separators=(",", ":")
        )
    current = _POLYSEG_DEPENDENCE_STATES.get(state_id)
    if current is None:
        current = _POLYSEG_DEPENDENCE_STATES[state_id] = PolySegDependenceState(**expected)
    elif any(getattr(current, key) != value for key, value in expected.items()):
        raise ValueError(f"dependence_state_id={state_id!r} is already bound to a different PolySeg configuration")
    return current


def reset_polyseg_estimator_state(state_id: str | None = None) -> None:
    """Reset one accumulating estimator, or all estimators when omitted."""

    if state_id is None:
        _POLYSEG_ESTIMATOR_STATES.clear()
    else:
        _POLYSEG_ESTIMATOR_STATES.pop(str(state_id), None)


def reset_polyseg_dependence_state(state_id: str | None = None) -> None:
    """Reset one confidence/dependence state, or all such states."""

    if state_id is None:
        _POLYSEG_DEPENDENCE_STATES.clear()
    else:
        _POLYSEG_DEPENDENCE_STATES.pop(str(state_id), None)


def polyseg_estimator_state_dict(state_id: str | None = None) -> dict[str, Any]:
    """Return a CPU-serializable snapshot of the accumulating estimator registry."""

    if state_id is not None:
        state = _POLYSEG_ESTIMATOR_STATES.get(str(state_id))
        return {} if state is None else {str(state_id): state.state_dict()}
    return {key: value.state_dict() for key, value in _POLYSEG_ESTIMATOR_STATES.items()}


def load_polyseg_estimator_state_dict(state: Mapping[str, Any], *, clear: bool = False) -> None:
    """Restore estimator states produced by :func:`polyseg_estimator_state_dict`."""

    if clear:
        _POLYSEG_ESTIMATOR_STATES.clear()
    for state_id, payload in state.items():
        _POLYSEG_ESTIMATOR_STATES[str(state_id)] = PolynomialEstimatorState.from_state_dict(payload)


def polyseg_dependence_state_dict(state_id: str | None = None) -> dict[str, Any]:
    """Return CPU-serializable persistent confidence/dependence snapshots."""

    if state_id is not None:
        state = _POLYSEG_DEPENDENCE_STATES.get(str(state_id))
        return {} if state is None else {str(state_id): state.state_dict()}
    return {key: value.state_dict() for key, value in _POLYSEG_DEPENDENCE_STATES.items()}


def load_polyseg_dependence_state_dict(state: Mapping[str, Any], *, clear: bool = False) -> None:
    """Restore persistent confidence/dependence states from a checkpoint."""

    if clear:
        _POLYSEG_DEPENDENCE_STATES.clear()
    for state_id, payload in state.items():
        _POLYSEG_DEPENDENCE_STATES[str(state_id)] = PolySegDependenceState.from_state_dict(payload)


# ``sum``/``binary`` are retained as compatibility aliases for the original
# smoke-test configuration.  The canonical names make the scorer contract
# explicit: a scorer either returns a local macro reward, a utility/potential,
# or a terminal outcome.
POLYSEG_PROTOCOLS = frozenset(
    {
        "local_sum", "utility_delta", "prm_potential", "terminal", "binary_local", "mean",
        "sum", "binary", "suffix", "potential", "prm",
    }
)


# These contracts are deliberately more specific than the canonical protocol
# name.  In particular, ``binary_local`` is an exact deterministic verifier
# label; it is not a Bernoulli-logistic observation and it is not a PRM
# pseudo-label.  Keeping the semantic object next to protocol normalization
# lets reward producers, synthetic controls, and audits share one vocabulary
# instead of inferring a data-generating process from the word "binary".
_FEEDBACK_PROTOCOL_CONTRACTS: dict[str, dict[str, str]] = {
    "local_sum": {
        "family": "aggregate_reward",
        "label_semantics": "observed_segment_reward",
        "estimator": "ridge_or_direct_macro",
        "comparator": "macro_return",
        "reward_aggregation": "segment_additive_sum",
        "reward_sign": "caller_defined",
        "raw_scale": "caller_declared_macro_units",
        "normalization": "none_by_default",
    },
    "utility_delta": {
        "family": "potential_difference",
        "label_semantics": "provider_utility_potential",
        "estimator": "potential_difference",
        "comparator": "macro_return",
        "reward_aggregation": "adjacent_potential_difference",
        "reward_sign": "signed",
        "raw_scale": "unit_potential_difference",
        "normalization": "potential_difference_preserves_telescoping",
    },
    "prm_potential": {
        "family": "potential_difference",
        "label_semantics": "prm_model_derived_pseudo_label",
        "estimator": "potential_difference",
        "comparator": "exact_terminal_verifier_diagnostic",
        "reward_aggregation": "adjacent_potential_difference",
        "reward_sign": "signed",
        "raw_scale": "unit_prm_potential_difference",
        "normalization": "potential_difference_preserves_telescoping",
    },
    "terminal": {
        "family": "sparse_outcome",
        "label_semantics": "deterministic_terminal_verifier",
        "estimator": "terminal_outcome",
        "comparator": "terminal_verifier",
        "reward_aggregation": "terminal_sparse_anchor",
        "reward_sign": "verifier_defined",
        "raw_scale": "verifier_score_units",
        "normalization": "no_segment_count_rescaling",
    },
    "binary_local": {
        "family": "binary_feedback",
        "label_semantics": "deterministic_prefix_verifier",
        "estimator": "logistic_fisher_or_binary_critic",
        "comparator": "exact_prefix_verifier",
        "reward_aggregation": "segment_binary_label",
        "reward_sign": "nonnegative_unit_interval",
        "raw_scale": "bernoulli_label_units",
        "normalization": "logistic_latent_and_probability_are_separate",
    },
    "mean": {
        "family": "aggregate_reward",
        "label_semantics": "observed_segment_mean",
        "estimator": "ridge_or_direct_macro",
        "comparator": "macro_return",
        "reward_aggregation": "segment_mean",
        "reward_sign": "caller_defined",
        "raw_scale": "caller_declared_macro_units",
        "normalization": "segment_mean_explicit",
        "training_scale": "one_over_realized_segment_count",
    },
}


def normalize_feedback_protocol(protocol: str) -> str:
    """Return the canonical reward-feedback protocol name.

    ``utility_delta`` is the implementation of the suffix/potential feedback
    in ``paper_methods.md`` Eq. (24)/(26): values returned by the scorer are
    utilities ``U_j`` and adjacent differences become macro rewards.
    """

    aliases = {
        "sum": "local_sum",
        "binary": "binary_local",
        "suffix": "utility_delta",
        "potential": "utility_delta",
        "prm": "prm_potential",
    }
    protocol = str(protocol).strip().lower()
    if protocol not in POLYSEG_PROTOCOLS:
        raise ValueError(
            f"Unknown PolySeg feedback protocol {protocol!r}; "
            "expected one of local_sum, utility_delta, prm_potential, terminal, binary_local, mean"
        )
    return aliases.get(protocol, protocol)


_POLYSEG_ROLLOUT_ALIASES = {
    "feedback": "polyseg_feedback",
    "feature_dim": "polyseg_feature_dim",
    "segment_cost": "polyseg_segment_cost",
    "budget": "polyseg_total_budget",
    "budget_fallback": "polyseg_budget_fallback",
    "critic_compute_cost": "polyseg_critic_compute_cost",
    "compute_budget": "polyseg_compute_budget",
    "q_candidates": "polyseg_q_candidates",
    "m_candidates": "polyseg_m_candidates",
    "selector_norm": "polyseg_selector_norm",
    "adaptive": "polyseg_adaptive",
}


def _config_value(mapping: Any, key: str) -> Any:
    """Read a config mapping/object without turning missing into a default."""
    if mapping is None:
        return None
    try:
        value = mapping.get(key, None) if hasattr(mapping, "get") else getattr(mapping, key, None)
    except (AttributeError, KeyError, TypeError):
        return None
    return value


def resolve_polyseg_option(
    config: Any,
    key: str,
    *,
    overrides: Mapping[str, Any] | None = None,
    rollout_alias: str | None = None,
    default: Any = None,
) -> Any:
    """Resolve one PolySeg option using one precedence contract.

    Explicit reward-manager kwargs win over ``algorithm.polyseg``.  Rollout
    fields are compatibility aliases used only when neither training-side
    source supplies a value.  ``None`` means unspecified, which lets a
    launcher use the algorithm value without a hidden YAML default masking it.
    """
    reward_kwargs = _config_value(_config_value(config, "reward"), "reward_kwargs")
    algorithm = _config_value(config, "algorithm")
    algorithm_polyseg = _config_value(algorithm, "polyseg")
    # Read the rollout object separately because both DictConfig and plain
    # namespaces are used by launchers and unit tests.
    actor_rollout_ref = _config_value(config, "actor_rollout_ref")
    rollout = _config_value(actor_rollout_ref, "rollout")
    if rollout is None:
        rollout = getattr(actor_rollout_ref, "rollout", None) if actor_rollout_ref is not None else None
    candidates = [
        overrides,
        reward_kwargs,
        algorithm_polyseg,
    ]
    alias = rollout_alias or _POLYSEG_ROLLOUT_ALIASES.get(key)
    if alias:
        candidates.append({alias: _config_value(rollout, alias)})
    candidates.append(rollout)
    for mapping in candidates:
        value = _config_value(mapping, key)
        if value is not None:
            return value
        if alias and mapping is not None and _config_value(mapping, alias) is not None:
            return _config_value(mapping, alias)
    return default


def feedback_protocol_contract(protocol: str) -> dict[str, str]:
    """Return the explicit statistical semantics for a feedback protocol.

    The returned mapping is a copy so callers may add provenance fields to a
    per-trajectory record without mutating the global contract.  This helper
    is intentionally descriptive: it does not turn deterministic verifier
    labels into Bernoulli samples or make PRM probabilities into correctness
    labels.  Synthetic Bernoulli-logistic experiments therefore retain their
    own ``feedback_protocol=bernoulli_logistic`` artifact value.
    """

    canonical = normalize_feedback_protocol(protocol)
    try:
        contract = _FEEDBACK_PROTOCOL_CONTRACTS[canonical]
    except KeyError as exc:  # pragma: no cover - normalize already guards this
        raise ValueError(f"no semantic contract for PolySeg protocol {canonical!r}") from exc
    return {"protocol": canonical, **contract}


def decode_token_ids(tokenizer, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
    """Decode CPU/Python token ids safely for Hugging Face tokenizers."""

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    elif isinstance(token_ids, np.ndarray):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _object_value(value: Any) -> Any:
    """Unwrap the object scalar wrappers used by DataProto/TransferQueue."""

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return value.reshape(()).item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _extra_fields_for_item(item: Any) -> dict[str, Any]:
    fields = item.non_tensor_batch.get("extra_fields")
    if fields is None:
        # AgentLoopManager (non-TransferQueue path) historically called this
        # field ``tool_extra_fields``.  Accept it while the rollout contract is
        # migrated, without changing the segment semantics.
        fields = item.non_tensor_batch.get("tool_extra_fields", {})
    if fields is None:
        fields = {}
    fields = _object_value(fields)
    if isinstance(fields, (list, tuple)) and len(fields) == 1 and isinstance(fields[0], dict):
        fields = fields[0]
    if isinstance(fields, dict):
        return dict(fields)
    return {}


def extract_segment_metadata(
    item: Any,
    *,
    response_length: int,
    valid_response_mask: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    fallback_m: int | None = None,
    require_rollout_segments: bool = False,
) -> dict[str, Any]:
    """Read rollout-owned segments, with an explicit smoke-only fallback.

    Agent loops write ``segment_ids`` aligned with the response and optional
    boundary/duration metadata into ``extra_fields``.  This helper also accepts
    direct non-tensor fields for legacy callers.  It never silently falls back
    when ``require_rollout_segments`` is enabled.
    """

    extra = _extra_fields_for_item(item)
    direct = item.non_tensor_batch
    raw_ids = extra.get("segment_ids", direct.get("segment_ids"))
    raw_boundaries = extra.get("segment_boundaries", direct.get("segment_boundaries"))
    raw_durations = extra.get("segment_durations", direct.get("segment_durations"))
    if raw_durations is None:
        raw_durations = direct.get("segment_durations")

    if raw_ids is None:
        if require_rollout_segments:
            raise ValueError(
                "PolySeg agentic reward requires segment_ids produced by AgentLoop; "
                "fixed token slicing is only available for explicit smoke tests."
            )
        if fallback_m is None:
            raise ValueError("fallback_m is required when rollout segment metadata is absent")
        valid_positions = torch.nonzero(valid_response_mask.to(torch.bool), as_tuple=False).flatten()
        valid_len = int(valid_positions.numel())
        if valid_len < 1:
            valid_len = 1
        slices = make_segment_slices(valid_len, min(int(fallback_m), valid_len))
        ids = torch.full((response_length,), -1, dtype=torch.long)
        for segment_id, (start, end) in enumerate(slices):
            positions = valid_positions[start:end]
            ids[positions] = segment_id
        durations = torch.as_tensor([end - start for start, end in slices], dtype=torch.float32)
        boundaries = [(int(start), int(end)) for start, end in slices]
        return {"segment_ids": ids, "segment_durations": durations, "segment_boundaries": boundaries, "from_rollout": False}

    ids = torch.as_tensor(raw_ids, dtype=torch.long).flatten().cpu()
    if ids.numel() < response_length:
        padded = torch.full((response_length,), -1, dtype=torch.long)
        padded[: ids.numel()] = ids
        ids = padded
    elif ids.numel() > response_length:
        raise ValueError(
            "rollout segment_ids must be token-aligned with response_length; "
            f"got {ids.numel()} IDs for {response_length} response positions"
        )
    ids[~valid_response_mask.to(torch.bool).cpu()] = -1
    active_ids = ids[valid_response_mask.to(torch.bool).cpu()]
    if active_ids.numel() == 0:
        raise ValueError("rollout segment_ids contain no valid response positions")
    if torch.any(active_ids < 0):
        raise ValueError("rollout segment_ids must be non-negative on valid response positions")

    # Segment ids must be contiguous for the macro tensors and estimator.  A
    # rollout may use arbitrary ids, so remap in first-occurrence order.
    ordered = []
    for value in active_ids.tolist():
        if value not in ordered:
            ordered.append(value)
    remap = {value: index for index, value in enumerate(ordered)}
    ids = torch.tensor([remap.get(value, -1) for value in ids.tolist()], dtype=torch.long)
    count = len(ordered)

    if raw_boundaries is None:
        boundaries = []
        for segment_id in range(count):
            positions = torch.nonzero(ids == segment_id, as_tuple=False).flatten()
            boundaries.append((int(positions.min()), int(positions.max()) + 1))
    else:
        boundaries = []
        for boundary in list(raw_boundaries):
            if isinstance(boundary, dict):
                boundaries.append((int(boundary["start"]), int(boundary["end"])))
            else:
                boundaries.append((int(boundary[0]), int(boundary[1])))
        if len(boundaries) != count:
            raise ValueError(
                "rollout segment_boundaries must contain one span per segment ID"
            )
        # Rollout-owned boundaries are causal metadata, not a display hint.
        # Reject malformed spans instead of allowing a reward manager to fit
        # features on empty, overlapping, or out-of-range intervals that
        # disagree with the token-aligned segment IDs.
        previous_end = 0
        for segment_id, (start, end) in enumerate(boundaries):
            if start < 0 or end <= start or end > response_length:
                raise ValueError(
                    f"segment boundary {segment_id} is outside response bounds: {(start, end)}"
                )
            if segment_id and start < previous_end:
                raise ValueError("rollout segment boundaries must be non-overlapping and ordered")
            positions = torch.nonzero(ids == segment_id, as_tuple=False).flatten()
            if positions.numel() == 0 or int(positions.min()) < start or int(positions.max()) >= end:
                raise ValueError(
                    f"segment boundary {segment_id} does not cover its token-aligned segment IDs"
                )
            previous_end = end

    if raw_durations is None:
        # AgentLoop segments represent one environment/tool transition.  The
        # token span is only the fallback for fixed-token smoke data.
        durations = torch.ones(count, dtype=torch.float32)
    else:
        durations = torch.as_tensor(raw_durations, dtype=torch.float32).flatten().cpu()
        if durations.numel() != count or torch.any(durations <= 0):
            raise ValueError("segment_durations must contain one positive value per rollout segment")

    return {"segment_ids": ids, "segment_durations": durations, "segment_boundaries": boundaries, "from_rollout": True}


def compute_segment_rewards(values: Sequence[float], protocol: str) -> list[float]:
    """Convert scorer outputs to one macro reward per segment."""

    canonical = normalize_feedback_protocol(protocol)
    values = [float(value) for value in values]
    if not values:
        return []
    if canonical == "local_sum":
        return values
    if canonical == "binary_local":
        return [float(value > 0.5) for value in values]
    if canonical in {"utility_delta", "prm_potential"}:
        previous = 0.0
        rewards = []
        for value in values:
            rewards.append(value - previous)
            previous = value
        return rewards
    if canonical == "terminal":
        return [0.0] * (len(values) - 1) + [values[-1]]
    if canonical == "mean":
        # Mean feedback is an episode-level protocol.  PPO still consumes an
        # additive macro reward, so distribute the observed episode mean over
        # the realized segments.  This keeps the training target and
        # ``episode_reward`` on the same scale for variable segment counts.
        scale = 1.0 / float(len(values))
        return [value * scale for value in values]
    raise AssertionError(f"unhandled protocol {canonical}")


def build_segment_features(
    boundaries: Sequence[tuple[int, int]],
    durations: Sequence[float] | torch.Tensor,
    *,
    response_length: int,
    action_lengths: Sequence[int] | None = None,
    tool_call_counts: Sequence[int] | None = None,
    feature_dim: int = 4,
    feature_horizon: float | None = None,
    max_duration: float | None = None,
    max_action_tokens: float | None = None,
    max_tool_calls: float = 16.0,
    state_features: torch.Tensor | np.ndarray | None = None,
    representation: FrozenLowDimRepresentation | None = None,
    state_mask: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Build the frozen low-dimensional feature map for macro states.

    The first coordinates are rollout-observable summaries: segment position,
    boundary position, action-token fraction, and normalized duration. A fifth
    coordinate records tool-call density when requested; remaining dimensions
    are zero padded so the feature width is explicit and stable across runs.
    """

    if feature_dim < 1 or response_length < 1:
        raise ValueError("feature_dim and response_length must be positive")
    count = len(boundaries)
    if count == 0:
        return torch.zeros((0, feature_dim), dtype=torch.float32)
    duration_values = torch.as_tensor(durations, dtype=torch.float32).flatten()
    if duration_values.numel() != count or torch.any(duration_values <= 0):
        raise ValueError("durations must contain one positive value per segment")
    lengths = [max(0, int(end) - int(start)) for start, end in boundaries]
    action_lengths = list(action_lengths) if action_lengths is not None else lengths
    tool_call_counts = list(tool_call_counts) if tool_call_counts is not None else [0] * count
    if len(action_lengths) != count or len(tool_call_counts) != count:
        raise ValueError("segment metadata lengths must agree")
    horizon_scale = max(float(feature_horizon if feature_horizon is not None else response_length), 1.0)
    duration_scale = max(float(max_duration if max_duration is not None else horizon_scale), 1.0)
    action_scale = max(float(max_action_tokens if max_action_tokens is not None else horizon_scale), 1.0)
    tool_scale = max(float(max_tool_calls), 1.0)
    if state_features is not None:
        state = _as_tensor(state_features, dtype=torch.float32)
        if state.ndim == 2 and state.shape[0] == count:
            segment_state = state
        elif state.ndim == 2 and state.shape[0] >= response_length:
            mask = None if state_mask is None else _as_tensor(state_mask, device=state.device).to(torch.bool).flatten()
            if mask is not None and mask.numel() < response_length:
                raise ValueError("state_mask must cover response_length rows")
            pooled = []
            for start, end in boundaries:
                rows = state[int(start): int(end)]
                if mask is not None:
                    rows = rows[mask[int(start): int(end)]]
                if rows.numel() == 0:
                    raise ValueError("each segment must contain at least one state feature row")
                pooled.append(rows.mean(0))
            segment_state = torch.stack(pooled, dim=0)
        else:
            raise ValueError("state_features must be [segments,input_dim] or [tokens,input_dim]")
        # Callers may provide either raw frozen hidden states together with a
        # serialized representation, or already mapped ``[-1, 1]`` features.
        # The latter is useful when the representation was applied by a
        # rollout service.  Raw features still fail closed unless an explicit
        # representation is supplied, so a changing hidden dimension cannot
        # silently become a new statistical feature map.
        if representation is None:
            if segment_state.shape[1] != feature_dim:
                raise ValueError(
                    "state_features require a frozen representation when their width "
                    "does not equal feature_dim"
                )
            mapped = segment_state
        else:
            mapped = representation.transform(segment_state)
        if mapped.shape[1] != feature_dim:
            raise ValueError("representation output dimension must equal feature_dim")
        return mapped.to(dtype=torch.float32)
    rows = []
    for index, ((start, end), duration) in enumerate(zip(boundaries, duration_values.tolist(), strict=True)):
        # These scales are caller/configuration supplied, never maxima of the
        # current trajectory.  This keeps the feature map fixed when sibling
        # segments, padding, or batch composition change.
        raw = [
            -1.0 + 2.0 * (float(start) / max(horizon_scale - 1.0, 1.0)),
            -1.0 + 2.0 * ((float(end) - 1.0) / max(horizon_scale - 1.0, 1.0)),
            -1.0 + 2.0 * min(float(action_lengths[index]) / action_scale, 1.0),
            -1.0 + 2.0 * min(float(duration) / duration_scale, 1.0),
        ]
        if feature_dim >= 5:
            raw.append(-1.0 + 2.0 * min(float(tool_call_counts[index]) / tool_scale, 1.0))
        rows.append(raw[:feature_dim] + [0.0] * max(0, feature_dim - len(raw)))
    return torch.tensor(rows, dtype=torch.float32)


class FrozenLowDimRepresentation:
    """A fitted, immutable low-dimensional map for PolySeg state features.

    The original PolySeg path uses rollout summaries as a safe fallback.  This
    class provides the missing representation contract for callers that have
    frozen LM hidden states or other continuous state/action features.  Mean,
    scale, projection and output scale are all serialized and fingerprinted;
    ``transform`` never refits them.
    """

    VERSION = "polyseg-frozen-lowdim-v1"

    def __init__(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        projection: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.mean = _as_tensor(mean, dtype=torch.float32).flatten().detach().cpu()
        self.scale = _as_tensor(scale, dtype=torch.float32).flatten().detach().cpu()
        self.projection = _as_tensor(projection, dtype=torch.float32).detach().cpu()
        self.output_scale = _as_tensor(output_scale, dtype=torch.float32).flatten().detach().cpu()
        if self.mean.numel() < 1 or self.scale.shape != self.mean.shape:
            raise ValueError("representation mean and scale must be non-empty and aligned")
        if self.projection.ndim != 2 or self.projection.shape[0] != self.mean.numel():
            raise ValueError("representation projection has an invalid input dimension")
        if self.output_scale.shape != (self.projection.shape[1],):
            raise ValueError("representation output_scale must match projection output dimension")
        if not all(torch.isfinite(x).all() for x in (self.mean, self.scale, self.projection, self.output_scale)):
            raise ValueError("representation parameters must be finite")
        if torch.any(self.scale <= 0) or torch.any(self.output_scale <= 0):
            raise ValueError("representation scales must be positive")
        self.feature_names = tuple(feature_names or [f"z{i}" for i in range(self.dimension)])
        if len(self.feature_names) != self.dimension or len(set(self.feature_names)) != self.dimension:
            raise ValueError("representation feature_names must match the output dimension and be unique")

    @property
    def input_dim(self) -> int:
        return int(self.mean.numel())

    @property
    def dimension(self) -> int:
        return int(self.projection.shape[1])

    @classmethod
    def fit(
        cls,
        samples: torch.Tensor | np.ndarray | Sequence[Sequence[float]],
        output_dim: int,
        *,
        seed: int = 0,
        feature_names: Sequence[str] | None = None,
    ) -> "FrozenLowDimRepresentation":
        """Fit normalization and a deterministic random orthogonal projection."""
        x = _as_tensor(samples, dtype=torch.float32)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("samples must be a non-empty [N,input_dim] matrix")
        if output_dim < 1 or output_dim > x.shape[1]:
            raise ValueError("output_dim must lie in [1, input_dim]")
        if not torch.isfinite(x).all():
            raise ValueError("representation samples must be finite")
        mean = x.mean(0)
        scale = x.std(0, unbiased=False).clamp_min(1e-6)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        raw = torch.randn((x.shape[1], output_dim), generator=generator, dtype=torch.float32)
        projection, _ = torch.linalg.qr(raw, mode="reduced")
        normalized = (x - mean) / scale
        projected = normalized @ projection
        output_scale = projected.abs().amax(0).clamp_min(1.0)
        return cls(mean, scale, projection, output_scale, feature_names=feature_names)

    @property
    def fingerprint(self) -> str:
        payload = self.state_dict()
        serializable = {
            key: (value.tolist() if torch.is_tensor(value) else value)
            for key, value in payload.items()
        }
        encoded = json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def transform(self, samples: torch.Tensor | np.ndarray | Sequence[Sequence[float]]) -> torch.Tensor:
        x = _as_tensor(samples, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("representation samples have the wrong input dimension")
        z = ((x - self.mean.to(x.device)) / self.scale.to(x.device)) @ self.projection.to(x.device)
        return (z / self.output_scale.to(z.device)).clamp(-1.0, 1.0)

    def drift(
        self,
        reference: torch.Tensor | np.ndarray | Sequence[Sequence[float]],
        current: torch.Tensor | np.ndarray | Sequence[Sequence[float]],
    ) -> dict[str, float]:
        """Measure representation drift between two checkpoints/batches."""
        before = self.transform(reference)
        after = self.transform(current)
        if before.shape != after.shape:
            raise ValueError("reference and current representation samples must align")
        norms = (after - before).norm(dim=-1)
        return {
            "mean_l2": float(norms.mean()),
            "max_l2": float(norms.max()),
            "p95_l2": float(torch.quantile(norms, 0.95)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "mean": self.mean,
            "scale": self.scale,
            "projection": self.projection,
            "output_scale": self.output_scale,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FrozenLowDimRepresentation":
        if state.get("version", cls.VERSION) != cls.VERSION:
            raise ValueError("unsupported frozen representation version")
        return cls(
            state["mean"], state["scale"], state["projection"], state["output_scale"],
            feature_names=state.get("feature_names"),
        )


def load_frozen_representation(path: str | os.PathLike[str]) -> FrozenLowDimRepresentation:
    """Load a serialized representation without permitting an implicit refit."""
    location = Path(path).expanduser()
    if not location.is_file():
        raise FileNotFoundError(f"frozen PolySeg representation does not exist: {location}")
    if location.suffix.lower() == ".json":
        payload = json.loads(location.read_text())
    else:
        payload = torch.load(location, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("frozen PolySeg representation checkpoint must be a mapping")
    return FrozenLowDimRepresentation.from_state_dict(payload)


def aggregate_step_features(
    step_features: torch.Tensor,
    segment_ids_tensor: torch.Tensor,
    *,
    num_segments: int | None = None,
    step_mask: torch.Tensor | None = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Construct the shared-parameter design Φ_j = Σ_{h in B_j} φ(z_h)."""
    x = _as_tensor(step_features)
    ids = _as_tensor(segment_ids_tensor, device=x.device, dtype=torch.long)
    if x.ndim != 3 or ids.ndim != 2 or x.shape[:2] != ids.shape:
        raise ValueError("step_features must be [batch,tokens,dim] and segment_ids [batch,tokens]")
    if num_segments is None:
        active = ids[ids >= 0]
        num_segments = int(active.max().item()) + 1 if active.numel() else 0
    if num_segments < 1:
        return x.new_zeros((x.shape[0], 0, x.shape[-1]))
    safe = ids.clamp_min(0)
    mask = (ids >= 0) if step_mask is None else _as_tensor(step_mask, device=x.device).to(torch.bool)
    if mask.shape != ids.shape:
        raise ValueError("step_mask must match segment_ids")
    if torch.any(ids[mask] < 0):
        raise ValueError("active step features must have non-negative segment ids")
    if torch.any(ids[mask] >= num_segments):
        raise ValueError("segment_ids contain an id outside num_segments")
    out = x.new_zeros((x.shape[0], num_segments, x.shape[-1]))
    out.scatter_add_(1, safe.unsqueeze(-1).expand_as(x), torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x)))
    if reduction == "mean":
        counts = x.new_zeros((x.shape[0], num_segments, 1))
        counts.scatter_add_(1, safe.unsqueeze(-1), mask.to(x.dtype).unsqueeze(-1))
        out = out / counts.clamp_min(1.0)
    elif reduction != "sum":
        raise ValueError("reduction must be 'sum' or 'mean'")
    return out


class SharedStepRewardEstimator:
    """Ridge estimator for the shared step-level reward route."""

    def __init__(self, feature_dim: int, ridge_lambda: float = 1.0, *, q: int | None = None, basis: str = "total", normalized_basis: bool = False) -> None:
        if feature_dim < 1 or ridge_lambda <= 0 or (q is not None and q < 0):
            raise ValueError("feature_dim must be positive and ridge_lambda must be positive")
        if basis not in {"total", "additive"}:
            raise ValueError("basis must be 'total' or 'additive'")
        self.input_dim = int(feature_dim)
        self.q = q
        self.basis = basis
        self.normalized_basis = bool(normalized_basis)
        self.feature_dim = (
            int(legendre_features(torch.zeros(1, feature_dim), q, basis=basis, normalized=normalized_basis).shape[-1])
            if q is not None else int(feature_dim)
        )
        self.ridge_lambda = float(ridge_lambda)
        self.gram: torch.Tensor | None = None
        self.rhs: torch.Tensor | None = None
        self.theta: torch.Tensor | None = None

    def fit_segment_feedback(
        self,
        step_features: torch.Tensor,
        segment_ids_tensor: torch.Tensor,
        segment_feedback: torch.Tensor,
        *,
        segment_mask: torch.Tensor | None = None,
        step_mask: torch.Tensor | None = None,
        accumulate: bool = False,
    ) -> "SharedStepRewardEstimator":
        raw = _as_tensor(step_features)
        if raw.ndim != 3 or raw.shape[-1] != self.input_dim:
            raise ValueError("step_features do not match the estimator input dimension")
        design = raw if self.q is None else legendre_features(
            raw.reshape(-1, raw.shape[-1]), self.q, basis=self.basis, normalized=self.normalized_basis
        ).reshape(raw.shape[0], raw.shape[1], -1)
        phi = aggregate_step_features(
            design,
            segment_ids_tensor,
            num_segments=segment_feedback.shape[-1],
            step_mask=step_mask,
            reduction="sum",
        )
        y = _as_tensor(segment_feedback, device=phi.device)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        mask = torch.ones_like(y, dtype=torch.bool) if segment_mask is None else _as_tensor(segment_mask, device=phi.device).to(torch.bool)
        if mask.shape != y.shape or phi.shape[:2] != y.shape:
            raise ValueError("segment feedback and mask must align with aggregated features")
        x = phi[mask].reshape(-1, self.feature_dim)
        target = y[mask].reshape(-1)
        if x.numel() == 0:
            raise ValueError("cannot fit shared reward estimator without observed feedback")
        gram = x.T @ x
        rhs = x.T @ target
        if accumulate and self.gram is not None:
            gram, rhs = self.gram.to(x) + gram, self.rhs.to(x) + rhs
        self.gram, self.rhs = gram.detach(), rhs.detach()
        eye = torch.eye(self.feature_dim, dtype=x.dtype, device=x.device)
        self.theta = torch.linalg.solve(gram + self.ridge_lambda * eye, rhs)
        return self

    def predict_step_rewards(self, step_features: torch.Tensor) -> torch.Tensor:
        if self.theta is None:
            raise RuntimeError("shared reward estimator must be fitted before prediction")
        raw = _as_tensor(step_features)
        if raw.ndim < 2 or raw.shape[-1] != self.input_dim:
            raise ValueError("step_features do not match the estimator input dimension")
        design = raw if self.q is None else legendre_features(
            raw.reshape(-1, raw.shape[-1]), self.q, basis=self.basis, normalized=self.normalized_basis
        ).reshape(*raw.shape[:-1], -1)
        return design @ self.theta.to(design.device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim, "feature_dim": self.feature_dim, "q": self.q,
            "basis": self.basis, "normalized_basis": self.normalized_basis,
            "ridge_lambda": self.ridge_lambda, "gram": self.gram, "rhs": self.rhs, "theta": self.theta,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "SharedStepRewardEstimator":
        if state.get("feature_dim", self.feature_dim) != self.feature_dim or state.get("input_dim", self.input_dim) != self.input_dim:
            raise ValueError("shared reward feature dimension does not match")
        if "q" in state and state["q"] != self.q:
            raise ValueError("shared reward polynomial order does not match")
        if "basis" in state and state["basis"] != self.basis:
            raise ValueError("shared reward basis does not match")
        if "normalized_basis" in state and bool(state["normalized_basis"]) != self.normalized_basis:
            raise ValueError("shared reward normalized-basis setting does not match")
        if "ridge_lambda" in state:
            value = float(state["ridge_lambda"])
            if not math.isfinite(value) or value <= 0 or not math.isclose(value, self.ridge_lambda, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("shared reward ridge regularization does not match")
        self.gram = None if state.get("gram") is None else _as_tensor(state["gram"])
        self.rhs = None if state.get("rhs") is None else _as_tensor(state["rhs"])
        self.theta = None if state.get("theta") is None else _as_tensor(state["theta"])
        if self.gram is not None and (self.gram.shape != (self.feature_dim, self.feature_dim) or not torch.isfinite(self.gram).all()):
            raise ValueError("invalid shared reward Gram matrix")
        if self.rhs is not None and (self.rhs.shape != (self.feature_dim,) or not torch.isfinite(self.rhs).all()):
            raise ValueError("invalid shared reward right-hand side")
        if self.theta is not None and not torch.isfinite(self.theta).all():
            raise ValueError("non-finite shared reward parameter vector")
        if self.theta is not None and self.theta.shape != (self.feature_dim,):
            raise ValueError("invalid shared reward parameter vector")
        return self


def recover_step_rewards_from_segment_feedback(
    step_features: torch.Tensor,
    segment_ids_tensor: torch.Tensor,
    segment_feedback: torch.Tensor,
    *,
    ridge_lambda: float = 1.0,
    segment_mask: torch.Tensor | None = None,
    step_mask: torch.Tensor | None = None,
    q: int | None = None,
    basis: str = "total",
    normalized_basis: bool = False,
) -> tuple[torch.Tensor, SharedStepRewardEstimator]:
    """Fit shared reward parameters and return one recovered reward per step."""
    estimator = SharedStepRewardEstimator(
        step_features.shape[-1], ridge_lambda, q=q, basis=basis, normalized_basis=normalized_basis
    )
    estimator.fit_segment_feedback(
        step_features,
        segment_ids_tensor,
        segment_feedback,
        segment_mask=segment_mask,
        step_mask=step_mask,
    )
    return estimator.predict_step_rewards(step_features), estimator


def dependence_confidence_radius(
    *,
    dimension: int,
    effective_samples: float,
    delta: float = 0.05,
    kappa_segment: float = 1.0,
    kappa_trajectory: float = 1.0,
    sigma: float = 1.0,
    bound: float = 1.0,
) -> float:
    """Compute an explicit finite-sample dependence-aware radius."""
    if dimension < 1 or effective_samples <= 0 or not 0 < delta < 1:
        raise ValueError("dimension/effective_samples must be positive and delta must lie in (0,1)")
    if min(kappa_segment, kappa_trajectory, sigma, bound) < 0:
        raise ValueError("dependence radius parameters must be non-negative")
    neff = effective_samples / max(kappa_segment * kappa_trajectory, 1.0)
    log_term = math.log(max(2.0 * dimension / delta, 1.0000001))
    return float(sigma * math.sqrt(2.0 * log_term / neff) + 3.0 * bound * log_term / neff)


def bellman_closure_residual(
    macro_rewards: torch.Tensor,
    successor_values: torch.Tensor,
    fitted_bellman_values: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    gamma: float = 1.0,
    durations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the empirical ζ residual for a fitted macro Bellman class."""
    r = _as_tensor(macro_rewards)
    nxt = _as_tensor(successor_values, device=r.device)
    fit = _as_tensor(fitted_bellman_values, device=r.device)
    if r.shape != nxt.shape or r.shape != fit.shape:
        raise ValueError("Bellman inputs must have identical shapes")
    d = torch.ones_like(r) if durations is None else _as_tensor(durations, device=r.device)
    if d.shape != r.shape:
        raise ValueError("durations must match Bellman inputs")
    residual = (r + torch.pow(torch.as_tensor(gamma, dtype=r.dtype, device=r.device), d) * nxt - fit).abs()
    if mask is not None:
        valid = _as_tensor(mask, device=r.device).to(torch.bool)
        if valid.shape != r.shape:
            raise ValueError("mask must match Bellman inputs")
        residual = residual[valid]
    return residual.max() if residual.numel() else r.new_zeros(())


def segment_comparator_regret(comparator_values: torch.Tensor | Sequence[float], policy_values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Compute empirical regret against a supplied macro comparator."""
    comparator = _as_tensor(comparator_values)
    policy = _as_tensor(policy_values, device=comparator.device)
    if comparator.shape != policy.shape:
        raise ValueError("comparator and policy values must have the same shape")
    return (comparator - policy).clamp_min(0).sum()


def budget_comparator_regret(comparator_values: torch.Tensor | Sequence[float], policy_values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    return segment_comparator_regret(comparator_values, policy_values)


def compression_gap(step_optimal_values: torch.Tensor | Sequence[float], segment_optimal_values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Compute the observed step-vs-segment comparator restriction gap."""
    step = _as_tensor(step_optimal_values)
    segment = _as_tensor(segment_optimal_values, device=step.device)
    if step.shape != segment.shape:
        raise ValueError("step and segment comparator values must have the same shape")
    return (step - segment).clamp_min(0).sum()


def full_step_comparator_regret(
    segment_regret: float | torch.Tensor,
    step_optimal_values: torch.Tensor | Sequence[float],
    segment_optimal_values: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Decompose full-step regret into macro regret plus compression gap."""
    macro = _as_tensor(segment_regret)
    return macro + compression_gap(step_optimal_values, segment_optimal_values)


def budget_feasible_segments(
    *,
    episodes: int,
    horizon: int,
    budget: float | None,
    segment_cost: float = 1.0,
) -> int:
    """Return m_B(K) for a fixed per-segment query cost."""
    if episodes < 1 or horizon < 1 or segment_cost <= 0:
        raise ValueError("episodes/horizon must be positive and segment_cost must be positive")
    if budget is None:
        return int(horizon)
    if budget < 0 or not math.isfinite(float(budget)):
        raise ValueError("budget must be finite and non-negative")
    return min(int(horizon), int(math.floor(float(budget) / (episodes * segment_cost))))


def unified_regret_bound(
    *,
    episodes: int,
    horizon: int,
    q: int,
    m: int,
    feature_dim: int,
    dependence_kappa_segment: float = 1.0,
    dependence_kappa_trajectory: float = 1.0,
    feedback_factor: float = 1.0,
    bellman_error: float = 0.0,
    compression_error: float = 0.0,
    feedback_error: float = 0.0,
    comparator_gap: float = 0.0,
    stat_constant: float = 1.0,
) -> dict[str, float]:
    """Evaluate the unified empirical/theoretical regret decomposition.

    This is a bound calculator, not a proof: users must supply valid smoothness,
    coverage and dependence constants for the returned value to have theorem
    semantics.
    """
    if episodes < 1 or horizon < 1 or q < 0 or not 1 <= m <= horizon or feature_dim < 1:
        raise ValueError("invalid regret-bound dimensions")
    if min(dependence_kappa_segment, dependence_kappa_trajectory, feedback_factor, stat_constant) < 0:
        raise ValueError("regret-bound constants must be non-negative")
    dimension = q + 1 if feature_dim == 1 else math.comb(feature_dim + q, feature_dim)
    statistical = stat_constant * feedback_factor * dimension * math.sqrt(
        episodes * m * max(dependence_kappa_segment, 1.0) * max(dependence_kappa_trajectory, 1.0)
    )
    total = statistical + episodes * (
        max(float(bellman_error), 0.0)
        + max(float(compression_error), 0.0)
        + max(float(feedback_error), 0.0)
        + max(float(comparator_gap), 0.0)
    )
    return {
        "statistical": float(statistical),
        "bellman": float(episodes * max(float(bellman_error), 0.0)),
        "compression": float(episodes * max(float(compression_error), 0.0)),
        "feedback": float(episodes * max(float(feedback_error), 0.0)),
        "comparator": float(episodes * max(float(comparator_gap), 0.0)),
        "total": float(total),
        "feature_dimension": float(dimension),
    }


class PolySegRegretTracker:
    """Serializable empirical comparator and budget accounting tracker."""

    def __init__(self) -> None:
        self.segment_regret = 0.0
        self.budget_regret = 0.0
        self.compression_gap = 0.0
        self.episodes = 0

    def update(self, *, comparator_value: float, policy_value: float, budget_comparator_value: float | None = None, step_optimal_value: float | None = None, segment_optimal_value: float | None = None) -> None:
        self.segment_regret += max(float(comparator_value) - float(policy_value), 0.0)
        self.budget_regret += max(float(budget_comparator_value if budget_comparator_value is not None else comparator_value) - float(policy_value), 0.0)
        if step_optimal_value is not None and segment_optimal_value is not None:
            self.compression_gap += max(float(step_optimal_value) - float(segment_optimal_value), 0.0)
        self.episodes += 1

    def state_dict(self) -> dict[str, Any]:
        return {"segment_regret": self.segment_regret, "budget_regret": self.budget_regret, "compression_gap": self.compression_gap, "episodes": self.episodes}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PolySegRegretTracker":
        obj = cls()
        for name in ("segment_regret", "budget_regret", "compression_gap"):
            value = float(state.get(name, 0.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid regret tracker {name}")
            setattr(obj, name, value)
        obj.episodes = _strict_nonnegative_int(state.get("episodes", 0), "regret tracker episodes")
        return obj


def cross_fitted_segment_residuals(
    features: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    *,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    ridge_lambda: float = 1.0,
) -> np.ndarray:
    """Compute trajectory-level out-of-fold residuals for dependence diagnostics.

    ``features`` has shape ``[trajectories, segments, dimension]`` and
    ``targets`` has shape ``[trajectories, segments]``.  Each trajectory is
    held out while a ridge model is fit on all other trajectories, so the
    returned residuals are not in-sample critic errors. A one-trajectory
    batch uses the zero prior because no independent fold exists; callers
    must not label that fallback as an out-of-fold diagnostic.
    """
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    x = np.asarray(features.detach().cpu() if torch.is_tensor(features) else features, dtype=np.float64)
    y = np.asarray(targets.detach().cpu() if torch.is_tensor(targets) else targets, dtype=np.float64)
    if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape:
        raise ValueError("features must be [trajectories, segments, dimension] and targets [trajectories, segments]")
    mask = np.ones(y.shape, dtype=bool) if valid_mask is None else np.asarray(
        valid_mask.detach().cpu() if torch.is_tensor(valid_mask) else valid_mask, dtype=bool
    )
    if mask.shape != y.shape:
        raise ValueError("valid_mask shape must match targets")
    residuals = np.zeros_like(y, dtype=np.float64)
    for held_out in range(x.shape[0]):
        train = mask.copy()
        train[held_out] = False
        train_x = x[train]
        train_y = y[train]
        test_x = x[held_out]
        if train_x.shape[0] == 0:
            prediction = np.zeros(test_x.shape[0])
        else:
            gram = train_x.T @ train_x + ridge_lambda * np.eye(train_x.shape[1])
            rhs = train_x.T @ train_y
            theta = np.linalg.solve(gram, rhs)
            prediction = test_x @ theta
        residuals[held_out] = y[held_out] - prediction
        residuals[held_out][~mask[held_out]] = 0.0
    return residuals


def cross_fitted_logistic_segment_residuals(
    features: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    *,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    logistic_l2: float = 1.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Return trajectory-held-out Bernoulli residuals for dependence scores.

    Binary feedback is an observation from a logistic mean model.  Using a
    linear ridge residual for its dependence estimate confounds sigmoid
    curvature with correlation and is not the score/Fisher quantity in the
    Methods.  This helper fits a regularized logistic model on all other
    trajectories and returns ``y - sigmoid(phi theta)`` for the held-out
    trajectory.  With only one trajectory there is no independent fold, so the
    zero-logit prior (probability 1/2) is used and callers can mark the result
    as non-OOF through the existing diagnostic flag.
    """
    if logistic_l2 <= 0 or max_iter < 1:
        raise ValueError("logistic_l2 must be positive and max_iter must be positive")
    x = np.asarray(features.detach().cpu() if torch.is_tensor(features) else features, dtype=np.float64)
    y = np.asarray(targets.detach().cpu() if torch.is_tensor(targets) else targets, dtype=np.float64)
    if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape:
        raise ValueError("features must be [trajectories, segments, dimension] and targets [trajectories, segments]")
    mask = np.ones(y.shape, dtype=bool) if valid_mask is None else np.asarray(
        valid_mask.detach().cpu() if torch.is_tensor(valid_mask) else valid_mask, dtype=bool
    )
    if mask.shape != y.shape:
        raise ValueError("valid_mask shape must match targets")
    if np.any((y[mask] < 0) | (y[mask] > 1)):
        raise ValueError("logistic targets must lie in [0, 1]")

    def sigmoid(logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(logits, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    residuals = np.zeros_like(y, dtype=np.float64)
    for held_out in range(x.shape[0]):
        train_mask = mask.copy()
        train_mask[held_out] = False
        train_x = x[train_mask]
        train_y = y[train_mask]
        test_x = x[held_out]
        theta = np.zeros(x.shape[-1], dtype=np.float64)
        if train_x.shape[0] > 0:
            for _ in range(int(max_iter)):
                probs = sigmoid(train_x @ theta)
                curvature = probs * (1.0 - probs)
                hessian = train_x.T @ (train_x * curvature[:, None]) + logistic_l2 * np.eye(x.shape[-1])
                gradient = train_x.T @ (probs - train_y) + logistic_l2 * theta
                step = np.linalg.solve(hessian, gradient)
                theta -= step
                if float(np.max(np.abs(step))) < 1e-7:
                    break
        residuals[held_out] = y[held_out] - sigmoid(test_x @ theta)
        residuals[held_out][~mask[held_out]] = 0.0
    return residuals


def _cross_fitted_ragged_residuals(
    feature_rows: Sequence[np.ndarray],
    target_rows: Sequence[np.ndarray],
    *,
    binary: bool,
    ridge_lambda: float = 1.0,
    logistic_l2: float = 1.0,
    max_iter: int = 50,
) -> list[np.ndarray]:
    """Fit trajectory-held-out residuals for variable-length histories.

    A persistent history can contain batches with different ``m`` values, so
    the rectangular public helpers cannot represent it without padding. This
    ragged implementation keeps complete trajectory clusters and is used only
    by the cross-batch diagnostic path.
    """
    if not feature_rows or len(feature_rows) != len(target_rows):
        raise ValueError("ragged OOF inputs must be nonempty and aligned")
    dimension = int(feature_rows[0].shape[-1])
    if dimension < 1:
        raise ValueError("ragged OOF feature dimension must be positive")
    for x, y in zip(feature_rows, target_rows, strict=True):
        if x.ndim != 2 or x.shape[1] != dimension or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
            raise ValueError("ragged OOF rows have incompatible shapes")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("ragged OOF rows must be finite")
    if binary and (
        np.any(np.concatenate(target_rows) < 0)
        or np.any(np.concatenate(target_rows) > 1)
    ):
        raise ValueError("logistic targets must lie in [0, 1]")
    if ridge_lambda <= 0 or logistic_l2 <= 0 or max_iter < 1:
        raise ValueError("regularization must be positive and max_iter must be positive")

    def sigmoid(logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(logits, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    residual_rows: list[np.ndarray] = []
    for held_out, (test_x, test_y) in enumerate(zip(feature_rows, target_rows, strict=True)):
        train_x = np.concatenate(
            [row for index, row in enumerate(feature_rows) if index != held_out], axis=0
        )
        train_y = np.concatenate(
            [row for index, row in enumerate(target_rows) if index != held_out], axis=0
        )
        if train_x.shape[0] == 0:
            prediction = np.full(test_y.shape, 0.5 if binary else 0.0, dtype=np.float64)
        elif binary:
            theta = np.zeros(dimension, dtype=np.float64)
            for _ in range(int(max_iter)):
                probs = sigmoid(train_x @ theta)
                curvature = probs * (1.0 - probs)
                hessian = train_x.T @ (train_x * curvature[:, None]) + logistic_l2 * np.eye(dimension)
                gradient = train_x.T @ (probs - train_y) + logistic_l2 * theta
                step = np.linalg.solve(hessian, gradient)
                theta -= step
                if float(np.max(np.abs(step))) < 1e-7:
                    break
            prediction = sigmoid(test_x @ theta)
        else:
            gram = train_x.T @ train_x + ridge_lambda * np.eye(dimension)
            theta = np.linalg.solve(gram, train_x.T @ train_y)
            prediction = test_x @ theta
        residual_rows.append(test_y - prediction)
    return residual_rows


def _dependence_covariance(x: np.ndarray) -> np.ndarray:
    """Stable covariance for a nonempty ``[observations, dimension]`` array."""
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((x.shape[-1] if x.ndim == 2 else 1,) * 2, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / max(x.shape[0], 1)


def _largest_regularized_eigen(base: np.ndarray, long_run: np.ndarray, ridge: float) -> float:
    dimension = base.shape[0]
    regularized = (base + base.T) / 2.0 + ridge * np.eye(dimension)
    eigvals, eigvecs = np.linalg.eigh(regularized)
    inverse_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, ridge))) @ eigvecs.T
    normalized = (inverse_sqrt @ ((long_run + long_run.T) / 2.0) @ inverse_sqrt)
    return float(np.max(np.linalg.eigvalsh((normalized + normalized.T) / 2.0)))


def _missing_dependence_diagnostics(
    *,
    reason: str,
    dimension: int,
    hac_lag: int = 0,
) -> dict[str, Any]:
    """Return an explicit finite fallback when no feedback rows are usable.

    A budget-exhausted batch can carry structural segment metadata while all
    of its labels are missing.  Such rows must not be represented as observed
    zero residuals, and they cannot support a dependence estimate.  The
    numerical values below keep downstream logging/checkpoint schemas finite;
    callers must use ``dependence_available`` before treating them as an
    estimate.
    """

    return {
        "dependence_available": False,
        "dependence_missing_reason": str(reason),
        "kappa_segment_by_m": {},
        "kappa_segment": 1.0,
        "kappa_trajectory": 1.0,
        "nominal_observations": 0,
        "cluster_count": 0,
        "total_observation_weight": 0.0,
        "weighted_ess_ratio": 0.0,
        "effective_samples": 0.0,
        "weighted_effective_samples": 0.0,
        "segment_acf": [],
        "trajectory_acf": [],
        "kappa_segment_upper": 1.0,
        "kappa_segment_lower": 1.0,
        "kappa_segment_upper_by_m": {},
        "kappa_trajectory_upper": 1.0,
        "kappa_trajectory_lower": 1.0,
        "chi_hat": 0.0,
        "hac_lag": int(max(hac_lag, 0)),
        "iid_cluster_radius_ratio": 1.0,
        "iid_cluster_radius_ratio_upper": 1.0,
        "trajectory_count": 0,
        "segment_count": 0,
        "bootstrap_samples": 0,
        "bootstrap_ci": {"segment": [1.0, 1.0], "trajectory": [1.0, 1.0]},
        "dimension": int(dimension),
    }


def _estimate_dependence_once(
    score_trajectories: list[np.ndarray],
    weights: list[np.ndarray],
    *,
    hac_lag: int,
    ridge: float,
) -> dict[str, Any]:
    # Callers normally remove fully masked trajectories before reaching this
    # helper, but bootstrap/history paths can supply ragged rows directly.
    # Normalize that input here as well so an empty row cannot make
    # ``max_m == 0`` and later produce a misleading ``kappa_by_m[0]`` lookup.
    if len(score_trajectories) != len(weights):
        raise ValueError("score trajectories and weights must have the same length")
    nonempty = [
        (row, weight)
        for row, weight in zip(score_trajectories, weights, strict=True)
        if row.shape[0] > 0
    ]
    if not nonempty:
        dimension = (
            int(score_trajectories[0].shape[1])
            if score_trajectories and score_trajectories[0].ndim == 2
            else 1
        )
        return _missing_dependence_diagnostics(
            reason="no_valid_observations",
            dimension=dimension,
            hac_lag=hac_lag,
        )
    score_trajectories = [row for row, _ in nonempty]
    weights = [weight for _, weight in nonempty]
    dimension = score_trajectories[0].shape[1]
    max_m = max((row.shape[0] for row in score_trajectories), default=1)
    observations = np.concatenate(score_trajectories, axis=0) if score_trajectories else np.zeros((1, dimension))
    gamma0 = _dependence_covariance(observations)
    kappa_by_m: dict[int, float] = {}
    acf: list[float] = []
    for lag in range(1, min(hac_lag, max_m - 1) + 1):
        pairs = [
            (row[:-lag], row[lag:])
            for row in score_trajectories
            if row.shape[0] > lag
        ]
        if not pairs:
            gamma_l = np.zeros_like(gamma0)
        else:
            left = np.concatenate([pair[0] for pair in pairs], axis=0)
            right = np.concatenate([pair[1] for pair in pairs], axis=0)
            center = observations.mean(axis=0)
            gamma_l = ((left - center).T @ (right - center)) / max(left.shape[0], 1)
        acf.append(float(np.trace(gamma_l) / max(np.trace(gamma0), ridge)))
    for m in range(1, max_m + 1):
        long_run = gamma0.copy()
        segment_lag = min(hac_lag, m - 1)
        for lag in range(1, segment_lag + 1):
            pairs = [(row[:-lag], row[lag:]) for row in score_trajectories if row.shape[0] > lag]
            if pairs:
                left = np.concatenate([pair[0] for pair in pairs], axis=0)
                right = np.concatenate([pair[1] for pair in pairs], axis=0)
                center = observations.mean(axis=0)
                gamma_l = ((left - center).T @ (right - center)) / max(left.shape[0], 1)
                # Bartlett/Newey-West taper uses the truncated HAC window,
                # not the nominal trajectory length (Methods 14.2).
                long_run += (1.0 - lag / (segment_lag + 1)) * (gamma_l + gamma_l.T)
        kappa_by_m[m] = float(np.clip(_largest_regularized_eigen(gamma0, long_run, ridge), 1.0, float(m)))

    cluster_sums = np.stack([row.sum(axis=0) for row in score_trajectories], axis=0)
    cluster_base = _dependence_covariance(cluster_sums)
    cluster_long = cluster_base.copy()
    cluster_lag = min(hac_lag, max(len(cluster_sums) - 1, 0))
    cluster_acf: list[float] = []
    centered_clusters = cluster_sums - cluster_sums.mean(axis=0, keepdims=True)
    for lag in range(1, cluster_lag + 1):
        left, right = centered_clusters[:-lag], centered_clusters[lag:]
        gamma_l = (left.T @ right) / max(left.shape[0], 1)
        cluster_long += (1.0 - lag / (cluster_lag + 1)) * (gamma_l + gamma_l.T)
        cluster_acf.append(float(np.trace(gamma_l) / max(np.trace(cluster_base), ridge)))
    kappa_tr = float(np.clip(_largest_regularized_eigen(cluster_base, cluster_long, ridge), 1.0, float(max(len(cluster_sums), 1))))
    flat_weights = np.concatenate(weights, axis=0) if weights else np.ones(1)
    total_weight = float(flat_weights.sum())
    weighted_clusters = np.stack([row_w @ row_s for row_w, row_s in zip(weights, score_trajectories, strict=True)], axis=0)
    weighted_variance = float(np.square(weighted_clusters - weighted_clusters.mean(axis=0)).sum())
    sigma0 = float(np.trace(gamma0))
    nominal_observations = int(sum(row.shape[0] for row in score_trajectories))
    # The estimand is a weighted observation mean.  Kish-style scaling must
    # cancel under ``w -> c w``; using an absolute ridge or clipping by total
    # weight would make the reported ESS depend on the unit of duration.
    if sigma0 <= 0.0 or weighted_variance <= 0.0 or nominal_observations <= 0:
        weighted_ess = float(max(nominal_observations, 1))
    else:
        weighted_ess = float(total_weight**2 * sigma0 / weighted_variance)
        weighted_ess = float(np.clip(weighted_ess, 1.0, float(nominal_observations)))
    return {
        "kappa_segment_by_m": kappa_by_m,
        "kappa_segment": kappa_by_m[max_m],
        "kappa_trajectory": kappa_tr,
        # Keep the observation measure explicit.  ``effective_samples`` is
        # the dependence-adjusted nominal segment count; weighted ESS is a
        # separate estimand based on the supplied segment weights.
        "nominal_observations": nominal_observations,
        "cluster_count": int(len(score_trajectories)),
        "total_observation_weight": total_weight,
        # Report the effective weighted observations as a fraction of the
        # nominal observation rows.  This ratio is also invariant to the
        # unit used for duration weights.
        "weighted_ess_ratio": float(weighted_ess / nominal_observations) if nominal_observations > 0 else 0.0,
        "effective_samples": float(sum(row.shape[0] for row in score_trajectories) / max(kappa_by_m[max_m] * kappa_tr, 1e-12)),
        "weighted_effective_samples": weighted_ess,
        "segment_acf": acf,
        "trajectory_acf": cluster_acf,
    }


def estimate_dependence_diagnostics(
    score_residuals: torch.Tensor | np.ndarray | Sequence[np.ndarray],
    *,
    valid_mask: torch.Tensor | np.ndarray | Sequence[np.ndarray] | None = None,
    weights: torch.Tensor | np.ndarray | Sequence[np.ndarray] | None = None,
    hac_lag: int | None = None,
    ridge: float = 1e-6,
    bootstrap_samples: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate HAC/cluster dependence, ESS, conservative bounds and ``chi``.

    The input is a sequence of per-trajectory score-residual matrices with
    shape ``[segments, dimension]``. It must be formed from cross-fitted
    residuals (use :func:`cross_fitted_segment_residuals` for scalar targets).
    Bootstrap resamples complete trajectories and therefore preserves the
    within-trajectory ordering required by the theory.
    """
    raw = score_residuals.detach().cpu().numpy() if torch.is_tensor(score_residuals) else score_residuals
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        trajectories = [np.asarray(row, dtype=np.float64) for row in raw]
    elif isinstance(raw, np.ndarray) and raw.ndim == 2:
        trajectories = [np.asarray(row[:, None], dtype=np.float64) for row in raw]
    elif isinstance(raw, np.ndarray) and raw.ndim == 1:
        trajectories = [np.asarray(raw[:, None], dtype=np.float64)]
    else:
        trajectories = [np.asarray(row, dtype=np.float64) for row in raw]
    if not trajectories or any(row.ndim != 2 or row.shape[0] == 0 for row in trajectories):
        raise ValueError("score_residuals must contain nonempty [segments, dimension] trajectories")
    dimension = trajectories[0].shape[1]
    if any(row.shape[1] != dimension for row in trajectories):
        raise ValueError("all trajectory score residuals must have the same dimension")

    def _rows(value, default):
        if value is None:
            return [np.full(row.shape[0], default, dtype=np.float64) for row in trajectories]
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else value
        if isinstance(array, np.ndarray) and array.ndim == 2:
            return [np.asarray(row, dtype=np.float64) for row in array]
        return [np.asarray(row, dtype=np.float64) for row in array]

    masks = _rows(valid_mask, 1.0)
    weight_rows = _rows(weights, 1.0)
    filtered_scores, filtered_weights = [], []
    for row, mask, weight in zip(trajectories, masks, weight_rows, strict=True):
        if mask.shape[0] != row.shape[0] or weight.shape[0] != row.shape[0]:
            raise ValueError("valid_mask and weights must match each trajectory")
        keep = mask.astype(bool)
        filtered_scores.append(row[keep])
        filtered_weights.append(np.maximum(weight[keep], 0.0))
    # Fully masked trajectories are absent observations, rather than
    # zero-valued trajectories.  Remove them before estimating HAC or
    # resampling complete trajectory clusters.  In particular, this prevents
    # budget exhaustion from producing max_m=0 and a fabricated dependence
    # estimate.
    nonempty = [
        (score_row, weight_row)
        for score_row, weight_row in zip(filtered_scores, filtered_weights, strict=True)
        if score_row.shape[0] > 0
    ]
    if not nonempty:
        return _missing_dependence_diagnostics(
            reason="no_valid_observations",
            dimension=dimension,
            hac_lag=0 if hac_lag is None else int(hac_lag),
        )
    filtered_scores = [pair[0] for pair in nonempty]
    filtered_weights = [pair[1] for pair in nonempty]
    if hac_lag is None:
        hac_lag = max(1, int(math.sqrt(max(row.shape[0] for row in filtered_scores))))
    hac_lag = max(0, int(hac_lag))
    estimate = _estimate_dependence_once(filtered_scores, filtered_weights, hac_lag=hac_lag, ridge=ridge)
    # ``_estimate_dependence_once`` is defensive, but keep the public result
    # explicit if a future change reintroduces an empty bootstrap input.
    if not estimate.get("dependence_available", True):
        estimate.update(
            {
                "hac_lag": int(hac_lag),
                "trajectory_count": len(filtered_scores),
                "segment_count": int(sum(row.shape[0] for row in filtered_scores)),
            }
        )
        return estimate
    kappa_values, kappa_tr_values = [], []
    kappa_grid_values: dict[int, list[float]] = {int(m): [] for m in estimate["kappa_segment_by_m"]}
    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(bootstrap_samples))):
        indices = rng.integers(0, len(filtered_scores), size=len(filtered_scores))
        boot_scores = [filtered_scores[index] for index in indices]
        boot_weights = [filtered_weights[index] for index in indices]
        boot = _estimate_dependence_once(boot_scores, boot_weights, hac_lag=hac_lag, ridge=ridge)
        kappa_values.append(boot["kappa_segment"])
        kappa_tr_values.append(boot["kappa_trajectory"])
        for m, value in boot["kappa_segment_by_m"].items():
            kappa_grid_values.setdefault(int(m), []).append(float(value))
    if kappa_values:
        seg_upper = float(np.quantile(kappa_values, 0.95))
        tr_upper = float(np.quantile(kappa_tr_values, 0.95))
        seg_lower = float(np.quantile(kappa_values, 0.05))
        tr_lower = float(np.quantile(kappa_tr_values, 0.05))
    else:
        seg_upper = seg_lower = estimate["kappa_segment"]
        tr_upper = tr_lower = estimate["kappa_trajectory"]
    kappa_grid = estimate["kappa_segment_by_m"]
    kappa_grid_upper = {}
    for m, value in kappa_grid.items():
        values = kappa_grid_values.get(int(m), [])
        kappa_grid_upper[int(m)] = max(float(value), float(np.quantile(values, 0.95))) if values else float(value)
    # Keep raw bootstrap intervals for diagnosis, while the operational
    # conservative bound must never shrink below the observed point value.
    # This envelope is an empirical safeguard, not a coverage guarantee.
    raw_bootstrap_ci = {"segment": [seg_lower, seg_upper], "trajectory": [tr_lower, tr_upper]}
    seg_upper = max(seg_upper, estimate["kappa_segment"])
    tr_upper = max(tr_upper, estimate["kappa_trajectory"])
    grid = [(math.log(m), math.log(max(value, 1.0))) for m, value in kappa_grid.items() if m >= 2]
    beta = np.polyfit(*zip(*grid), 1)[0] if len(grid) >= 2 else 0.0
    estimate.update(
        {
            "dependence_available": True,
            "kappa_segment_upper": seg_upper,
            "kappa_segment_lower": seg_lower,
            "kappa_segment_upper_by_m": kappa_grid_upper,
            "kappa_trajectory_upper": tr_upper,
            "kappa_trajectory_lower": tr_lower,
            "chi_hat": float(np.clip(1.0 - beta, 0.0, 1.0)),
            "hac_lag": hac_lag,
            "iid_cluster_radius_ratio": float(math.sqrt(max(estimate["kappa_segment"] * estimate["kappa_trajectory"], 1.0))),
            "iid_cluster_radius_ratio_upper": float(math.sqrt(max(seg_upper * tr_upper, 1.0))),
            "trajectory_count": len(filtered_scores),
            "segment_count": int(sum(row.shape[0] for row in filtered_scores)),
            "bootstrap_samples": int(max(0, bootstrap_samples)),
            "bootstrap_ci": raw_bootstrap_ci,
        }
    )
    return estimate


def _polyseg_dependence_metrics(
    features: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    durations: torch.Tensor | None,
    q: int,
    basis: str,
    ridge_lambda: float,
    binary: bool = False,
    logistic_l2: float = 1.0,
    weight_policy: str = "unit",
    history_state: PolynomialEstimatorState | None = None,
    persistent_history: bool = False,
) -> dict[str, Any]:
    """Build dependence diagnostics from cross-fitted polynomial score residuals."""
    weight_policy = str(weight_policy).strip().lower()
    if weight_policy not in {"unit", "segment_duration"}:
        raise ValueError("weight_policy must be 'unit' or 'segment_duration'")
    design = legendre_features(features.reshape(-1, features.shape[-1]), q, basis=basis).reshape(
        features.shape[0], features.shape[1], -1
    )
    duration_rows = (
        np.ones(targets.shape, dtype=np.float64)
        if weight_policy == "unit" or durations is None
        else np.asarray(durations.detach().cpu() if torch.is_tensor(durations) else durations, dtype=np.float64)
    )
    if duration_rows.ndim == 1:
        duration_rows = np.broadcast_to(duration_rows[None, :], targets.shape)
    design_np = design.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    if persistent_history and history_state is not None and history_state.dependence_history:
        history_features = [item["features"].numpy() for item in history_state.dependence_history]
        history_targets = [item["targets"].numpy() for item in history_state.dependence_history]
        current_features = [design_np[row][valid_np[row]] for row in range(design_np.shape[0]) if valid_np[row].any()]
        current_targets = [targets_np[row][valid_np[row]] for row in range(targets_np.shape[0]) if valid_np[row].any()]
        # Strictly prequential semantics: each current trajectory is held out
        # against the previous history only. Other trajectories from the same
        # batch must not become training rows before the policy update.
        residual_rows = []
        for current_x, current_y in zip(current_features, current_targets, strict=True):
            residual_rows.append(
                _cross_fitted_ragged_residuals(
                    history_features + [current_x],
                    history_targets + [current_y],
                    binary=binary,
                    ridge_lambda=ridge_lambda,
                    logistic_l2=logistic_l2,
                )[-1]
            )
        residuals = np.zeros_like(targets_np, dtype=np.float64)
        current_index = 0
        for row in range(design_np.shape[0]):
            if valid_np[row].any():
                residuals[row, valid_np[row]] = residual_rows[current_index]
                current_index += 1
        prior_scores = [item["scores"].numpy() for item in history_state.dependence_history]
        prior_weights = [item["weights"].numpy() for item in history_state.dependence_history]
        current_scores = [design_np[row][valid_np[row]] * residuals[row, valid_np[row], None] for row in range(design_np.shape[0]) if valid_np[row].any()]
        current_weights = [duration_rows[row][valid_np[row]] for row in range(design_np.shape[0]) if valid_np[row].any()]
        score_rows = prior_scores + current_scores
        weight_rows = prior_weights + current_weights
        diagnostics = estimate_dependence_diagnostics(score_rows, weights=weight_rows)
        diagnostics["diagnostic_oof"] = 1.0
        diagnostics["dependence_oof_mode"] = "persistent_prequential_history"
    else:
        residuals = (
            cross_fitted_logistic_segment_residuals(
                design,
                targets,
                valid_mask=valid,
                logistic_l2=logistic_l2,
            )
            if binary
            else cross_fitted_segment_residuals(
                design,
                targets,
                valid_mask=valid,
                ridge_lambda=ridge_lambda,
            )
        )
        score = design_np * residuals[..., None]
        diagnostics = estimate_dependence_diagnostics(score, valid_mask=valid, weights=duration_rows)
        valid_trajectories = valid_np.any(axis=1)
        diagnostics["diagnostic_oof"] = float(valid_trajectories.sum() >= 2)
        diagnostics["dependence_oof_mode"] = "batch_cross_fit"
    diagnostics["dependence_score_model"] = "logistic_fisher_score" if binary else "ridge_residual_score"
    diagnostics["weight_policy"] = weight_policy
    diagnostics["residual_rms"] = float(
        np.sqrt(np.mean(residuals[valid_np] ** 2)) if valid_np.any() else 0.0
    )
    diagnostics["oof_residuals"] = residuals
    return diagnostics


@dataclass(frozen=True)
class PolySegConfig:
    """Runtime options for the PolySeg advantage estimator and selector."""

    q: int = 2
    m: int = 4
    feature_dim: int = 4
    feature_horizon: int | None = None
    basis: str = "total"
    normalized_basis: bool = False
    feedback: str = "sum"
    # ``macro`` is the direct macro-MDP estimator.  ``shared_step`` enables
    # the identifiable shared-parameter route from detailed_methods.md and
    # requires rollout-provided frozen per-step features.
    reward_route: str = "macro"
    ridge_lambda: float = 1.0
    logistic_l2: float = 1.0
    gamma: float = 1.0
    lam: float = 1.0
    use_kl_in_reward: bool = False
    normalize_advantages: bool = True
    r_max: float = 1.0
    horizon: int = 0
    segment_cost: float = 1.0
    budget: int | None = None
    # Optional declared critic-compute cost per fitted polynomial feature.
    # This is separate from verifier/suffix feedback cost: q changes the
    # feature dimension and therefore the amount of critic work even when m
    # and the number of scored trajectories are unchanged.
    critic_compute_cost: float = 0.0
    compute_budget: float | None = None
    smoothness_s: float = 2.0
    stat_exponent_u: float = 1.0
    approx_exponent_v: float = 1.0
    segment_exponent_gamma: float = 1.0
    dependence_chi: float = 1.0
    stat_constant: float = 1.0
    approx_constant: float = 1.0
    segment_constant: float = 1.0
    cost_penalty: float = 0.0
    use_polynomial_critic: bool = False
    critic_type: str = "polynomial"
    mlp_hidden_dim: int = 32
    mlp_learning_rate: float = 0.05
    mlp_steps: int = 100
    accumulate_estimator: bool = False
    estimator_state_id: str = ""
    polynomial_target: str = "return"
    use_confidence_bonus: bool = False
    reward_confidence_beta: float = 1.0
    transition_confidence_beta: float = 1.0
    confidence_delta: float = 0.05
    use_formal_confidence_radius: bool = False
    bellman_closure_tolerance: float = 0.0
    # Fit a macro transition operator from adjacent segment-boundary
    # features and use it for the Bellman target.  This is disabled by
    # default because ordinary LLM rollouts do not expose an environment
    # state transition; enabling it makes the approximation explicit rather
    # than silently treating a return smoother as a transition model.
    use_transition_estimator: bool = False
    transition_ridge_lambda: float = 1.0
    transition_backup_iterations: int = 1
    # Weight macro-critic observations by rollout-observable segment duration
    # (or use unit weights for the equal-segment ablation).  Duration weighting
    # is the default for variable-length agentic trajectories and leaves the
    # fixed single-turn protocol unchanged because its durations are one.
    critic_weighting: str = "duration"
    # Prior trajectory clusters retained for prequential OOF dependence
    # diagnostics when accumulating estimator state.
    dependence_history_limit: int = 128
    # Independent checkpointable identity for confidence/dependence state.
    # Empty derives a run-local identity from estimator_state_id.
    dependence_state_id: str = ""

    def __post_init__(self):
        for name in ("normalized_basis", "use_kl_in_reward", "normalize_advantages", "use_polynomial_critic", "accumulate_estimator", "use_confidence_bonus", "use_formal_confidence_radius", "use_transition_estimator"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        _strict_int(self.q, "q", minimum=0)
        _strict_int(self.m, "m", minimum=1)
        _strict_int(self.feature_dim, "feature_dim", minimum=1)
        if self.feature_horizon is not None:
            _strict_int(self.feature_horizon, "feature_horizon", minimum=1)
        _strict_int(self.horizon, "horizon", minimum=0)
        _strict_int(self.mlp_hidden_dim, "mlp_hidden_dim", minimum=1)
        _strict_int(self.mlp_steps, "mlp_steps", minimum=1)
        _strict_int(self.transition_backup_iterations, "transition_backup_iterations", minimum=1)
        _strict_int(self.dependence_history_limit, "dependence_history_limit", minimum=0)
        if self.q < 0:
            raise ValueError("q must be non-negative")
        if self.m < 1:
            raise ValueError("m must be positive")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self.feature_horizon is not None and self.feature_horizon < 1:
            raise ValueError("feature_horizon must be positive when provided")
        if self.basis not in {"total", "additive"}:
            raise ValueError("basis must be 'total' or 'additive'")
        if normalize_feedback_protocol(self.feedback) not in {
            "local_sum", "binary_local", "utility_delta", "prm_potential", "terminal", "mean"
        }:
            raise ValueError("unsupported PolySeg feedback protocol")
        if self.reward_route not in {"macro", "shared_step"}:
            raise ValueError("reward_route must be 'macro' or 'shared_step'")
        if not 0.0 < float(self.confidence_delta) < 1.0:
            raise ValueError("confidence_delta must lie in (0, 1)")
        if self.bellman_closure_tolerance < 0 or not math.isfinite(float(self.bellman_closure_tolerance)):
            raise ValueError("bellman_closure_tolerance must be finite and non-negative")
        if not 0 <= self.dependence_chi <= 1:
            raise ValueError("dependence_chi must be in [0, 1]")
        if self.segment_cost <= 0:
            raise ValueError("segment_cost must be positive")
        if not math.isfinite(float(self.critic_compute_cost)) or self.critic_compute_cost < 0:
            raise ValueError("critic_compute_cost must be finite and non-negative")
        if self.compute_budget is not None and (
            not math.isfinite(float(self.compute_budget)) or self.compute_budget < 0
        ):
            raise ValueError("compute_budget must be finite and non-negative")
        if self.r_max < 0 or not math.isfinite(float(self.r_max)):
            raise ValueError("r_max must be finite and non-negative")
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative")
        if self.polynomial_target not in {"return", "value"}:
            raise ValueError("polynomial_target must be 'return' or 'value'")
        if self.critic_type not in {"polynomial", "mlp"}:
            raise ValueError("critic_type must be 'polynomial' or 'mlp'")
        if self.mlp_hidden_dim < 1 or self.mlp_learning_rate <= 0 or self.mlp_steps < 1:
            raise ValueError("invalid MLP critic configuration")
        if not isinstance(self.estimator_state_id, str):
            raise ValueError("estimator_state_id must be a string")
        if not isinstance(self.dependence_state_id, str):
            raise ValueError("dependence_state_id must be a string")
        if self.accumulate_estimator and self.critic_type != "polynomial":
            raise ValueError("accumulate_estimator currently requires critic_type='polynomial'")
        if self.accumulate_estimator and normalize_feedback_protocol(self.feedback) == "binary_local":
            raise ValueError("accumulate_estimator currently supports sum/local feedback only")
        if self.accumulate_estimator and self.use_confidence_bonus:
            raise ValueError("accumulate_estimator cannot combine with use_confidence_bonus")
        if self.reward_confidence_beta < 0 or self.transition_confidence_beta < 0:
            raise ValueError("confidence betas must be non-negative")
        if self.transition_ridge_lambda <= 0:
            raise ValueError("transition_ridge_lambda must be positive")
        if self.transition_backup_iterations < 1:
            raise ValueError("transition_backup_iterations must be positive")
        if self.critic_weighting not in {"uniform", "duration"}:
            raise ValueError("critic_weighting must be 'uniform' or 'duration'")
        if int(self.dependence_history_limit) < 0:
            raise ValueError("dependence_history_limit must be non-negative")
        if self.use_transition_estimator and not self.use_polynomial_critic:
            raise ValueError("use_transition_estimator requires use_polynomial_critic")

    @classmethod
    def from_any(cls, config: Any = None, **overrides) -> "PolySegConfig":
        """Build a config from a dataclass, OmegaConf object, mapping or None."""

        if config is None:
            values: dict[str, Any] = {}
        elif isinstance(config, cls):
            values = asdict(config)
        elif isinstance(config, Mapping):
            values = dict(config)
        else:
            values = {key: getattr(config, key) for key in cls.__dataclass_fields__ if hasattr(config, key)}
            nested_config = getattr(config, "polyseg", None)
            if nested_config is not None:
                values["polyseg"] = nested_config
            for key in ("gamma", "lam"):
                if hasattr(config, key):
                    values[key] = getattr(config, key)
        # The trainer commonly passes the top-level AlgoConfig.  Prefer its
        # nested polyseg section, then apply explicit overrides last so adaptive
        # batch metadata can intentionally replace static q/m values.
        nested = values.get("polyseg")
        if nested is not None and not isinstance(config, cls):
            nested_values = dict(nested) if isinstance(nested, Mapping) else {
                key: getattr(nested, key) for key in cls.__dataclass_fields__ if hasattr(nested, key)
            }
            values = {**values, **nested_values}
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


@dataclass(frozen=True)
class PolySegChoice:
    """A feasible candidate selected by the finite ``(q, m)`` search."""

    q: int
    m: int
    objective: float | None
    statistical_term: float | None
    approximation_term: float | None
    segment_term: float | None
    feedback_factor: float | None
    query_cost: float
    feature_dim: int
    # Declared critic-side compute cost, kept separate from verifier query
    # cost so a budget curve cannot silently conflate the two resources.
    compute_cost: float = 0.0
    # ``skip`` is an explicit, auditable controller-level fallback.  It is
    # kept on the choice so the rollout and reward paths cannot accidentally
    # turn an exhausted budget into an ordinary zero-valued label.
    fallback: str = "none"
    fallback_reason: str = ""


class PolySegSelectionInfeasible(ValueError):
    """The candidate grid has no feasible choice under the supplied constraints."""


class BudgetedPolySegController:
    """Stateful budget controller for the theory's episode-level selector.

    The rollout backend may choose to keep a fixed segmentation for a fair
    ablation.  When it supports per-episode metadata, this controller can be
    called before generation to choose a feasible ``(q, m)`` pair and then be
    updated with the actual verifier cost.  Keeping the ledger here makes the
    selection policy serializable and prevents a planning choice from
    silently overspending the feedback budget.
    """

    def __init__(
        self,
        *,
        q_candidates: Iterable[int],
        m_candidates: Iterable[int],
        total_budget: float | None,
        segment_cost: float = 1.0,
        **bound_kwargs: Any,
    ):
        _strict_real(segment_cost, "segment_cost", strictly_positive=True)
        if not math.isfinite(segment_cost) or segment_cost <= 0:
            raise ValueError("segment_cost must be finite and positive")
        parsed_q = [_strict_nonnegative_int(q, "q candidate") for q in q_candidates]
        parsed_m = [_strict_nonnegative_int(m, "m candidate") for m in m_candidates]
        self.q_candidates = tuple(sorted(set(parsed_q)))
        self.m_candidates = tuple(sorted(set(parsed_m)))
        if not self.q_candidates or not self.m_candidates:
            raise ValueError("q_candidates and m_candidates must be non-empty")
        if any(q < 0 for q in self.q_candidates) or any(m < 1 for m in self.m_candidates):
            raise ValueError("q candidates must be non-negative and m candidates positive")
        self.total_budget = None if total_budget is None else _strict_real(total_budget, "total_budget")
        if self.total_budget is not None and (not math.isfinite(self.total_budget) or self.total_budget < 0):
            raise ValueError("total_budget must be finite and non-negative")
        self.segment_cost = float(segment_cost)
        self.bound_kwargs = dict(bound_kwargs)
        # Checkpointed selector bounds are trusted configuration, not data to
        # be silently repaired.  Reject non-finite numeric values before they
        # can reach the candidate search.
        for name, value in self.bound_kwargs.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError(f"controller bound {name} must be finite")
        self.critic_compute_cost = float(self.bound_kwargs.pop("critic_compute_cost", 0.0))
        if not math.isfinite(self.critic_compute_cost) or self.critic_compute_cost < 0:
            raise ValueError("critic_compute_cost must be finite and non-negative")
        self.compute_budget = self.bound_kwargs.pop("compute_budget", None)
        if self.compute_budget is not None:
            self.compute_budget = float(self.compute_budget)
            if not math.isfinite(self.compute_budget) or self.compute_budget < 0:
                raise ValueError("compute_budget must be finite and non-negative")
        self.consumed_compute_budget = 0.0
        self.consumed_budget = 0.0
        self.episodes = 0
        self.last_choice: PolySegChoice | None = None
        self.budget_fallback = str(self.bound_kwargs.pop("budget_fallback", "error")).strip().lower()
        if self.budget_fallback not in {"error", "skip"}:
            raise ValueError("budget_fallback must be 'error' or 'skip'")
        # Dependence-aware selector state.  The controller starts without
        # observations; unseen m values are handled conservatively by the
        # finite-horizon bound in choose_polynomial_segment_config.
        self.selector_norm = str(self.bound_kwargs.pop("selector_norm", "effective_round")).strip().lower()
        self.kappa_segment_upper_by_m: dict[int, float] = {}
        self.kappa_segment_upper_by_qm: dict[tuple[int, int], float] = {}
        self.kappa_trajectory_upper = 1.0

    @property
    def remaining_budget(self) -> float | None:
        if self.total_budget is None:
            return None
        return self.total_budget - self.consumed_budget

    @property
    def remaining_compute_budget(self) -> float | None:
        if self.compute_budget is None:
            return None
        return self.compute_budget - self.consumed_compute_budget

    def choose(self, *, episodes: int, horizon: int, feature_dim: int, feedback: str = "sum") -> PolySegChoice:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        choice_args = dict(
            q_candidates=self.q_candidates,
            m_candidates=self.m_candidates,
            episodes=episodes,
            horizon=horizon,
            feature_dim=feature_dim,
            feedback=feedback,
            selector_norm=self.selector_norm,
            kappa_segment_upper_by_m=self.kappa_segment_upper_by_m,
            kappa_segment_upper_by_qm=self.kappa_segment_upper_by_qm,
            kappa_trajectory_upper=self.kappa_trajectory_upper,
            segment_cost=self.segment_cost,
            critic_compute_cost=self.critic_compute_cost,
            compute_budget=self.remaining_compute_budget,
            **self.bound_kwargs,
        )
        try:
            choice = choose_polynomial_segment_config(budget=self.remaining_budget, **choice_args)
        except PolySegSelectionInfeasible:
            if self.budget_fallback != "skip" or (
                self.remaining_budget is None and self.remaining_compute_budget is None
            ):
                raise
            # Check that removing ONLY the budget constraint restores
            # feasibility.  Never swallow invalid parameters or an
            # insufficient effective sample size as a budget fallback.
            unconstrained_args = dict(choice_args)
            unconstrained_args.update(budget=None, compute_budget=None)
            unconstrained = choose_polynomial_segment_config(**unconstrained_args)
            feedback_limited = (
                self.remaining_budget is not None
                and unconstrained.query_cost > self.remaining_budget + 1e-8
            )
            compute_limited = (
                self.remaining_compute_budget is not None
                and unconstrained.compute_cost > self.remaining_compute_budget + 1e-8
            )
            if feedback_limited and compute_limited:
                fallback_reason = "feedback_and_compute_budget_exhausted_before_batch"
            elif compute_limited:
                fallback_reason = "critic_compute_budget_exhausted_before_batch"
            else:
                fallback_reason = "feedback_budget_exhausted_before_batch"
            choice = PolySegChoice(
                q=unconstrained.q,
                m=min(m for m in self.m_candidates if m <= horizon),
                objective=None,
                statistical_term=None,
                approximation_term=None,
                segment_term=None,
                feedback_factor=None,
                query_cost=0.0,
                feature_dim=unconstrained.feature_dim,
                compute_cost=0.0,
                fallback="skip",
                fallback_reason=fallback_reason,
            )
        self.last_choice = choice
        return choice

    def observe(self, query_cost: float, *, compute_cost: float = 0.0, episodes: int = 1) -> float | None:
        if not math.isfinite(query_cost) or query_cost < 0 or episodes < 1:
            raise ValueError("query_cost must be finite and non-negative and episodes positive")
        if not math.isfinite(compute_cost) or compute_cost < 0:
            raise ValueError("compute_cost must be finite and non-negative")
        amount = float(query_cost)
        if self.total_budget is not None and self.consumed_budget + amount > self.total_budget + 1e-8:
            raise RuntimeError(
                f"controller budget exhausted: requested={amount}, remaining={self.remaining_budget}"
            )
        if self.compute_budget is not None and self.consumed_compute_budget + float(compute_cost) > self.compute_budget + 1e-8:
            raise RuntimeError(
                f"critic compute budget exhausted: requested={compute_cost}, remaining={self.remaining_compute_budget}"
            )
        # Check both ledgers before mutating either one. A rejected compute
        # charge must not leave the feedback ledger partially consumed.
        self.consumed_budget += amount
        self.consumed_compute_budget += float(compute_cost)
        self.episodes += int(episodes)
        return self.remaining_budget

    def update_dependence_diagnostics(
        self,
        *,
        m: int,
        kappa_segment_upper: float,
        kappa_trajectory_upper: float,
        q: int | None = None,
    ) -> None:
        """Update the conservative effective-round selector state.

        Diagnostics are produced after a batch has been scored, so they are
        intentionally used only for the next rollout batch.  Invalid or
        non-finite observations are ignored rather than weakening the
        conservative fallback for an unseen segment count.
        """
        m = int(m)
        seg = float(kappa_segment_upper)
        traj = float(kappa_trajectory_upper)
        if m < 1 or not math.isfinite(seg) or not math.isfinite(traj):
            return
        if q is not None and int(q) >= 0:
            self.kappa_segment_upper_by_qm[(int(q), m)] = min(max(seg, 1.0), float(m))
        elif q is None:
            # A residual-based estimate at one q is not a uniform bound for
            # other polynomial orders. Only explicitly q-independent input
            # may populate the legacy m-only cache.
            self.kappa_segment_upper_by_m[m] = min(max(seg, 1.0), float(m))
        self.kappa_trajectory_upper = max(self.kappa_trajectory_upper, traj, 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_candidates": list(self.q_candidates),
            "m_candidates": list(self.m_candidates),
            "total_budget": self.total_budget,
            "segment_cost": self.segment_cost,
            "critic_compute_cost": self.critic_compute_cost,
            "compute_budget": self.compute_budget,
            "consumed_compute_budget": self.consumed_compute_budget,
            "budget_fallback": self.budget_fallback,
            "bound_kwargs": dict(self.bound_kwargs),
            "consumed_budget": self.consumed_budget,
            "episodes": self.episodes,
            "selector_norm": self.selector_norm,
            "kappa_segment_upper_by_m": dict(self.kappa_segment_upper_by_m),
            "kappa_segment_upper_by_qm": [
                {"q": q, "m": m, "value": value}
                for (q, m), value in self.kappa_segment_upper_by_qm.items()
            ],
            "kappa_trajectory_upper": self.kappa_trajectory_upper,
            "last_choice": None if self.last_choice is None else asdict(self.last_choice),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BudgetedPolySegController":
        controller = cls(
            q_candidates=state["q_candidates"],
            m_candidates=state["m_candidates"],
            total_budget=state.get("total_budget"),
            segment_cost=state.get("segment_cost", 1.0),
            critic_compute_cost=state.get("critic_compute_cost", 0.0),
            compute_budget=state.get("compute_budget"),
            **dict(state.get("bound_kwargs", {})),
        )
        controller.consumed_budget = float(state.get("consumed_budget", 0.0))
        controller.consumed_compute_budget = float(state.get("consumed_compute_budget", 0.0))
        controller.episodes = _strict_nonnegative_int(state.get("episodes", 0), "controller episodes")
        controller.selector_norm = str(state.get("selector_norm", controller.selector_norm)).strip().lower()
        if (
            not math.isfinite(controller.consumed_budget)
            or controller.consumed_budget < 0
            or not math.isfinite(controller.consumed_compute_budget)
            or controller.consumed_compute_budget < 0
            or controller.episodes < 0
        ):
            raise ValueError("invalid controller counters")
        if controller.selector_norm not in {"power", "effective_round", "segment_update"}:
            raise ValueError("invalid controller selector_norm")
        controller.budget_fallback = str(state.get("budget_fallback", controller.budget_fallback)).strip().lower()
        if controller.budget_fallback not in {"error", "skip"}:
            raise ValueError("invalid controller budget_fallback")
        raw_m_bounds = state.get("kappa_segment_upper_by_m", {})
        if not isinstance(raw_m_bounds, Mapping):
            raise ValueError("invalid controller m dependence bounds")
        controller.kappa_segment_upper_by_m = {}
        for key, raw_value in raw_m_bounds.items():
            m_key = _strict_nonnegative_int(key, "controller m dependence key")
            value = float(raw_value)
            if m_key < 1 or not math.isfinite(value) or value < 1.0 or value > float(m_key):
                raise ValueError("invalid controller m dependence bound")
            controller.kappa_segment_upper_by_m[m_key] = value
        raw_qm = state.get("kappa_segment_upper_by_qm", [])
        if isinstance(raw_qm, Mapping):
            # Accept the transient tuple-key representation used by early
            # checkpoints in this development branch.
            raw_qm = [
                {"q": key[0], "m": key[1], "value": value}
                for key, value in raw_qm.items()
            ]
        if not isinstance(raw_qm, (list, tuple)):
            raise ValueError("invalid controller q/m dependence bounds")
        controller.kappa_segment_upper_by_qm = {}
        for item in raw_qm:
            if not isinstance(item, Mapping):
                raise ValueError("invalid controller q/m dependence entry")
            q_key = _strict_nonnegative_int(item["q"], "controller q dependence key")
            m_key = _strict_nonnegative_int(item["m"], "controller m dependence key")
            value = float(item["value"])
            if m_key < 1 or not math.isfinite(value) or value < 1.0 or value > float(m_key):
                raise ValueError("invalid controller q/m dependence bound")
            controller.kappa_segment_upper_by_qm[(q_key, m_key)] = value
        trajectory_upper = float(state.get("kappa_trajectory_upper", 1.0))
        if not math.isfinite(trajectory_upper) or trajectory_upper < 1.0:
            raise ValueError("invalid controller trajectory dependence bound")
        controller.kappa_trajectory_upper = trajectory_upper
        choice = state.get("last_choice")
        controller.last_choice = None if choice is None else PolySegChoice(**choice)
        if controller.total_budget is not None and controller.consumed_budget > controller.total_budget + 1e-8:
            raise ValueError("invalid controller state: consumed budget exceeds total")
        if controller.compute_budget is not None and controller.consumed_compute_budget > controller.compute_budget + 1e-8:
            raise ValueError("invalid controller state: consumed compute budget exceeds total")
        return controller


class SegmentRewardBudget:
    """Exact accounting for segment/suffix/verifier feedback queries.

    ``consume`` is deliberately transactional: a failed budget check never
    changes the remaining budget.  This makes it safe to call from concurrent
    rollout schedulers before submitting a query.
    """

    def __init__(self, total: float, *, cost_per_query: float = 1.0):
        _strict_real(total, "total budget")
        _strict_real(cost_per_query, "cost_per_query", strictly_positive=True)
        if not math.isfinite(float(total)) or total < 0:
            raise ValueError("total budget must be non-negative")
        if not math.isfinite(float(cost_per_query)) or cost_per_query <= 0:
            raise ValueError("cost_per_query must be positive")
        self.total = float(total)
        self.cost_per_query = float(cost_per_query)
        self.consumed = 0.0
        # Reservations are held while an external verifier is in flight.
        # ``consumed`` contains only settled, billed cost.  Keeping the two
        # quantities separate prevents concurrent workers from spending the
        # same remaining budget and makes exception/refund paths explicit.
        self.reserved = 0.0
        self._reservations: dict[str, float] = {}
        self._settled: dict[str, float] = {}

    @property
    def remaining(self) -> float:
        return self.total - self.consumed - self.reserved

    def can_afford(self, cost: float | int = 1) -> bool:
        if not math.isfinite(float(cost)) or cost < 0:
            raise ValueError("cost must be non-negative")
        return self.consumed + self.reserved + float(cost) <= self.total + 1e-8

    def reserve(self, cost: float | int = 1, *, reservation_id: str | None = None) -> str:
        """Atomically hold ``cost`` until an external query is settled.

        A caller may supply a stable id so retry/reissue callbacks are
        idempotent.  Repeating an active reservation with the same amount
        returns the existing token without charging twice.
        """
        amount = float(cost)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("reservation cost must be finite and non-negative")
        token = str(reservation_id or uuid4().hex)
        if token in self._reservations:
            if not math.isclose(self._reservations[token], amount, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("reservation id is already active with a different cost")
            return token
        if token in self._settled:
            return token
        if not self.can_afford(amount):
            raise RuntimeError(f"feedback budget exhausted: requested={amount}, remaining={self.remaining}")
        self._reservations[token] = amount
        self.reserved += amount
        return token

    def commit(self, reservation_id: str, *, cost: float | None = None) -> float:
        """Settle a reservation at the measured billed cost.

        The actual cost may be lower than the conservative reservation.  A
        repeated commit is a no-op and returns the same remaining budget,
        which makes delayed/retried reward callbacks safe.
        """
        token = str(reservation_id)
        if token in self._settled:
            return self.remaining
        if token not in self._reservations:
            raise KeyError(f"unknown budget reservation: {token}")
        reserved = self._reservations.pop(token)
        actual = reserved if cost is None else float(cost)
        if not math.isfinite(actual) or actual < 0 or actual > reserved + 1e-8:
            # Keep the reservation intact so the caller can explicitly refund
            # or retry after a malformed billing response.
            self._reservations[token] = reserved
            raise ValueError("settled cost must be finite, non-negative, and no larger than reserved cost")
        self.reserved -= reserved
        self.consumed += actual
        self._settled[token] = actual
        return self.remaining

    def refund_reservation(self, reservation_id: str) -> float:
        """Release an unbilled reservation; repeated release is idempotent."""
        token = str(reservation_id)
        reserved = self._reservations.pop(token, None)
        if reserved is not None:
            self.reserved -= reserved
        return self.remaining

    def consume(self, queries: float | int = 1, *, cost: float | None = None) -> float:
        amount = float(queries) * self.cost_per_query if cost is None else float(cost)
        token = self.reserve(amount)
        return self.commit(token)

    def refund(self, cost: float) -> float:
        """Legacy adjustment for already-settled cost.

        New query paths should use :meth:`refund_reservation` so a callback
        cannot accidentally refund another request's billed amount.
        """
        amount = float(cost)
        if not math.isfinite(amount) or amount < 0 or amount > self.consumed + 1e-8:
            raise ValueError("refund must be non-negative and no larger than consumed cost")
        self.consumed -= amount
        return self.remaining

    def state_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "cost_per_query": self.cost_per_query,
            "consumed": self.consumed,
            "reserved": self.reserved,
            "reservations": dict(self._reservations),
            "settled": dict(self._settled),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, float]) -> "SegmentRewardBudget":
        obj = cls(state["total"], cost_per_query=state.get("cost_per_query", 1.0))
        obj.consumed = float(state.get("consumed", 0.0))
        reservations = state.get("reservations", {})
        settled = state.get("settled", {})
        if not isinstance(reservations, Mapping) or not isinstance(settled, Mapping):
            raise ValueError("invalid budget state: reservations and settled must be mappings")
        obj._reservations = {str(k): float(v) for k, v in reservations.items()}
        obj.reserved = float(state.get("reserved", sum(obj._reservations.values())))
        obj._settled = {str(k): float(v) for k, v in settled.items()}
        values = [obj.consumed, obj.reserved, *obj._reservations.values(), *obj._settled.values()]
        if any(not math.isfinite(value) or value < -1e-8 for value in values):
            raise ValueError("invalid budget state: values must be finite and non-negative")
        if set(obj._reservations).intersection(obj._settled):
            raise ValueError("invalid budget state: reservation is both active and settled")
        if state.get("reservations") is not None and not math.isclose(
            obj.reserved, sum(obj._reservations.values()), rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("invalid budget state: reserved total does not match reservations")
        if state.get("settled") is not None and not math.isclose(
            obj.consumed, sum(obj._settled.values()), rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("invalid budget state: consumed total does not match settled costs")
        if obj.consumed + obj.reserved > obj.total + 1e-8:
            raise ValueError("invalid budget state: consumed plus reserved exceeds total")
        return obj


class FileBackedSegmentRewardBudget:
    """Process-safe shared ledger for a training-wide feedback budget.

    Ray reward workers are separate processes (and may be separate hosts in a
    distributed deployment), so an in-memory :class:`SegmentRewardBudget`
    cannot enforce a global ``B_K``.  This adapter stores the same state in a
    small JSON file and takes an advisory ``flock`` around every read-modify-
    write operation.  The lock is held only while touching the ledger; the
    external verifier call remains outside the critical section, with an
    explicit reservation preventing concurrent overspending.

    The file is initialized exactly once.  A second worker opening an existing
    ledger must agree on ``total`` and ``cost_per_query``; it cannot silently
    reset a running experiment.  ``ledger_id`` is persisted in the file so
    journal rows from independent workers share one auditable identity.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike[str], total: float, *, cost_per_query: float = 1.0):
        if not path:
            raise ValueError("a ledger path is required")
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._configured_total = float(total)
        self._configured_cost = float(cost_per_query)
        if not math.isfinite(self._configured_total) or self._configured_total < 0:
            raise ValueError("total budget must be finite and non-negative")
        if not math.isfinite(self._configured_cost) or self._configured_cost <= 0:
            raise ValueError("cost_per_query must be finite and positive")
        with self._locked():
            state = self._read_unlocked()
            if state is None:
                state = self._new_state()
                self._write_unlocked(state)
            else:
                self._validate_config(state)

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": self._SCHEMA_VERSION,
            "ledger_id": uuid4().hex,
            "total": self._configured_total,
            "cost_per_query": self._configured_cost,
            "consumed": 0.0,
            "reserved": 0.0,
            "reservations": {},
            "settled": {},
        }

    def _validate_config(self, state: Mapping[str, Any]) -> None:
        schema_version = state.get("schema_version", 0)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != self._SCHEMA_VERSION:
            raise ValueError("unsupported PolySeg shared-ledger schema")
        total = float(state["total"])
        cost = float(state.get("cost_per_query", 1.0))
        if not math.isclose(total, self._configured_total, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                f"shared feedback ledger total mismatch: existing={total}, requested={self._configured_total}"
            )
        if not math.isclose(cost, self._configured_cost, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                f"shared feedback ledger cost mismatch: existing={cost}, requested={self._configured_cost}"
            )
        # Round-trip through the in-memory validator to reject malformed or
        # overdrawn state before a worker can use it.
        SegmentRewardBudget.from_state_dict(state)

    def _read_unlocked(self) -> dict[str, Any] | None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise ValueError("shared feedback ledger must contain a JSON object")
        return value

    def _write_unlocked(self, state: Mapping[str, Any]) -> None:
        # Replace the file atomically while the lock is held.  A crash leaves
        # either the old complete JSON document or the new complete document.
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(state), handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    from contextlib import contextmanager

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _transaction(self, operation):
        with self._locked():
            state = self._read_unlocked()
            if state is None:
                raise RuntimeError("shared feedback ledger disappeared while in use")
            self._validate_config(state)
            budget = SegmentRewardBudget.from_state_dict(state)
            result = operation(budget)
            next_state = budget.state_dict()
            next_state.update(
                {
                    "schema_version": self._SCHEMA_VERSION,
                    "ledger_id": state["ledger_id"],
                }
            )
            self._write_unlocked(next_state)
            return result

    @property
    def ledger_id(self) -> str:
        with self._locked():
            state = self._read_unlocked()
            if state is None:
                raise RuntimeError("shared feedback ledger disappeared while in use")
            self._validate_config(state)
            return str(state["ledger_id"])

    @property
    def total(self) -> float:
        return float(self._snapshot()["total"])

    @property
    def cost_per_query(self) -> float:
        return float(self._snapshot().get("cost_per_query", 1.0))

    @property
    def consumed(self) -> float:
        return float(self._snapshot()["consumed"])

    @property
    def reserved(self) -> float:
        return float(self._snapshot()["reserved"])

    @property
    def remaining(self) -> float:
        state = self._snapshot()
        return float(state["total"] - state["consumed"] - state["reserved"])

    def _snapshot(self) -> dict[str, Any]:
        with self._locked():
            state = self._read_unlocked()
            if state is None:
                raise RuntimeError("shared feedback ledger disappeared while in use")
            self._validate_config(state)
            return state

    def can_afford(self, cost: float | int = 1) -> bool:
        return self._transaction(lambda budget: budget.can_afford(cost))

    def reserve(self, cost: float | int = 1, *, reservation_id: str | None = None) -> str:
        return self._transaction(lambda budget: budget.reserve(cost, reservation_id=reservation_id))

    def commit(self, reservation_id: str, *, cost: float | None = None) -> float:
        return self._transaction(lambda budget: budget.commit(reservation_id, cost=cost))

    def refund_reservation(self, reservation_id: str) -> float:
        return self._transaction(lambda budget: budget.refund_reservation(reservation_id))

    def consume(self, queries: float | int = 1, *, cost: float | None = None) -> float:
        def _consume(budget):
            return budget.consume(queries, cost=cost)

        return self._transaction(_consume)

    def state_dict(self) -> dict[str, Any]:
        state = self._snapshot()
        state["schema_version"] = self._SCHEMA_VERSION
        return state


@dataclass
class SegmentTrace:
    """Validated segment metadata for one trajectory.

    ``segment_ids`` uses ``-1`` for padding/observation positions.  The class
    is deliberately backend-neutral and can be serialized into a DataProto
    non-tensor field before being padded by the reward loop.
    """

    segment_ids: torch.Tensor
    response_mask: torch.Tensor
    segment_rewards: torch.Tensor | None = None
    segment_durations: torch.Tensor | None = None
    segment_mask: torch.Tensor | None = None

    def validate(self) -> "SegmentTrace":
        if self.segment_ids.ndim != 1 or self.response_mask.ndim != 1:
            raise ValueError("SegmentTrace fields must be one-dimensional")
        if self.segment_ids.shape != self.response_mask.shape:
            raise ValueError("segment_ids and response_mask must have the same length")
        active = self.response_mask.to(torch.bool)
        if torch.any(self.segment_ids[active] < 0):
            raise ValueError("active response positions must have non-negative segment ids")
        count = int(self.segment_ids[active].max().item()) + 1 if torch.any(active) else 0
        if self.segment_rewards is not None and self.segment_rewards.numel() != count:
            raise ValueError("segment_rewards length does not match segment ids")
        if self.segment_mask is not None and self.segment_mask.numel() != count:
            raise ValueError("segment_mask length does not match segment ids")
        if self.segment_durations is not None:
            if self.segment_durations.numel() != count:
                raise ValueError("segment_durations must contain one value per segment")
            duration_mask = (
                torch.ones(count, dtype=torch.bool, device=self.segment_durations.device)
                if self.segment_mask is None
                else self.segment_mask.to(device=self.segment_durations.device, dtype=torch.bool).reshape(-1)
            )
            valid_durations = self.segment_durations.reshape(-1)[duration_mask]
            if torch.any(~torch.isfinite(valid_durations)) or torch.any(valid_durations <= 0):
                raise ValueError("segment_durations must be finite and positive on valid segments")
        return self


class PolynomialTransitionEstimator:
    """Finite-dimensional macro transition regression.

    For a chosen next-state feature map ``next_features``, this estimates
    ``E[next_features(X_{j+1}) | z_j]`` with one ridge solve per output.  It is
    the implementable vector-valued form of Eq. (30)-(32); callers can apply
    the estimated operator to any value model expressed in that feature map.
    """

    def __init__(self, q: int, feature_dim: int, *, basis: str = "total", ridge_lambda: float = 1.0):
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        self.q, self.feature_dim, self.basis, self.ridge_lambda = q, feature_dim, basis, ridge_lambda
        self.theta: torch.Tensor | None = None
        self.design: torch.Tensor | None = None
        self._gram: torch.Tensor | None = None
        self._sample_count = 0
        self.target_semantics = "next_features"

    @property
    def dimension(self) -> int:
        return 1 + self.feature_dim * self.q if self.basis == "additive" else math.comb(self.feature_dim + self.q, self.feature_dim)

    def fit(
        self,
        z: torch.Tensor,
        next_features: torch.Tensor,
        *,
        accumulate: bool = False,
        state: PolynomialEstimatorState | None = None,
    ) -> "PolynomialTransitionEstimator":
        x = legendre_features(z, self.q, basis=self.basis).reshape(-1, self.dimension)
        y = _as_tensor(next_features, device=x.device)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("z and next_features have incompatible sample counts")
        if accumulate:
            if state is None:
                raise ValueError("state is required when accumulate=True")
            state.update_transition(x, y)
            if state.transition_ridge_lambda != self.ridge_lambda:
                raise ValueError("transition state regularization does not match estimator")
            self.theta = state.solve_transition()
            self._gram = state.transition_gram
            self._sample_count = state.transition_sample_count
            self.design = None
        else:
            gram = x.T @ x + self.ridge_lambda * torch.eye(self.dimension, dtype=x.dtype, device=x.device)
            self.theta = torch.linalg.solve(gram, x.T @ y)
            self._gram = x.T @ x
            self._sample_count = int(x.shape[0])
            self.design = x.detach()
        self.target_semantics = "next_features"
        return self

    def fit_test_function(
        self,
        z: torch.Tensor,
        next_features: torch.Tensor,
        test_function,
        *,
        accumulate: bool = False,
        state: PolynomialEstimatorState | None = None,
    ) -> "PolynomialTransitionEstimator":
        """Fit ``P f(z) = E[f(X') | z]`` from observed successor states.

        A vector fit of ``E[X' | z]`` can only be plugged into a value model
        without changing the estimand when that value model is affine in the
        successor features.  Polynomial critics with ``q > 1`` are nonlinear,
        so ``V(E[X'])`` is generally different from ``E[V(X')]``.  This helper
        evaluates the declared test function on each observed successor and
        regresses that scalar/vector target directly, preserving the
        action-conditioned ``P V`` object required by the macro Bellman
        backup.
        """

        successor = _as_tensor(next_features)
        targets = test_function(successor)
        targets = _as_tensor(targets, device=successor.device)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        if targets.ndim != 2 or targets.shape[0] != successor.shape[0]:
            raise ValueError("test_function must return [samples] or [samples, outputs]")
        fitted = self.fit(z, targets, accumulate=accumulate, state=state)
        fitted.target_semantics = "test_function"
        return fitted

    def predict_test_function(self, z: torch.Tensor) -> torch.Tensor:
        """Predict the fitted ``P f`` test function value(s)."""

        if self.theta is None:
            raise RuntimeError("transition estimator must be fit before predict")
        x = legendre_features(z, self.q, basis=self.basis)
        prediction = x @ self.theta
        return prediction.squeeze(-1) if prediction.shape[-1] == 1 else prediction

    def features(self, z: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        """Return the action-conditioned transition design features."""

        return legendre_features(_as_tensor(z), self.q, basis=self.basis)

    def predict_next_features(self, z: torch.Tensor) -> torch.Tensor:
        if self.theta is None:
            raise RuntimeError("transition estimator must be fit before predict")
        if self.target_semantics != "next_features":
            raise RuntimeError(
                "transition estimator was fit on a test function; use predict_test_function"
            )
        x = legendre_features(z, self.q, basis=self.basis)
        return x @ self.theta

    def apply_value(self, z: torch.Tensor, next_value_weights: torch.Tensor) -> torch.Tensor:
        """Apply the learned transition operator to a linear next value."""
        return self.predict_next_features(z) @ _as_tensor(next_value_weights, device=z.device)

    def gram_matrix(self) -> torch.Tensor:
        """Return the regularized design Gram matrix used by the ridge solve."""
        if self.design is None:
            if self._gram is None:
                raise RuntimeError("transition estimator must be fit before requesting its Gram matrix")
            return self._gram + self.ridge_lambda * torch.eye(
                self.dimension, dtype=self._gram.dtype, device=self._gram.device
            )
        return self.design.T @ self.design + self.ridge_lambda * torch.eye(
            self.dimension, dtype=self.design.dtype, device=self.design.device
        )

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def feature_norms(self, z: torch.Tensor) -> torch.Tensor:
        """Return elliptical norms ``||phi(z)||_{V^{-1}}`` for confidence bounds."""
        x = legendre_features(z, self.q, basis=self.basis)
        return ridge_feature_norms(x, self.gram_matrix())

    def confidence_bonus(self, z: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Return a transition uncertainty bonus for each macro sample."""
        if beta < 0:
            raise ValueError("beta must be non-negative")
        return float(beta) * self.feature_norms(z)


@dataclass
class MacroLSVIPlan:
    """Optimistic finite-action macro plan returned by ``MacroLSVIPlanner``.

    Arrays are indexed as ``[stage, planning_state, action]``.  Keeping the
    optimistic Q table and both confidence terms in the return object makes
    exploration decisions auditable in a synthetic Macro-MDP experiment.
    """

    q_values: torch.Tensor
    values: torch.Tensor
    actions: torch.Tensor
    reward_bonus: torch.Tensor
    transition_bonus: torch.Tensor


class MacroLSVIPlanner:
    """Finite-action optimistic planner for the independent Macro-MDP branch.

    ``fit`` consumes observed macro transitions with frozen state-action
    features.  ``plan`` performs a stage-wise backward recursion over a
    finite candidate action set.  Reward regression and direct ``P V``
    regression use the same Legendre feature contract as
    :class:`PolynomialTransitionEstimator`; uncertainty is added with
    elliptical bonuses and the result is clipped to the duration-aware
    ``r_max`` envelope.

    The planner requires actual boundary successor features and candidate
    action features.  It fails closed when those are absent, so a return
    critic cannot be mistaken for an unknown-transition LSVI implementation.
    It is an implementable finite-action Macro-MDP component, not a claim of a
    general LLM or theorem-level regret guarantee.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        q: int,
        feature_dim: int,
        stages: int,
        actions: int,
        basis: str = "total",
        normalized_basis: bool = False,
        ridge_lambda: float = 1.0,
        transition_ridge_lambda: float | None = None,
        gamma: float = 1.0,
        r_max: float = 1.0,
        reward_confidence_beta: float = 1.0,
        transition_confidence_beta: float = 1.0,
    ) -> None:
        if isinstance(q, bool) or int(q) != q or q < 0:
            raise ValueError("q must be a non-negative integer")
        if isinstance(feature_dim, bool) or int(feature_dim) != feature_dim or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if isinstance(stages, bool) or int(stages) != stages or stages < 1:
            raise ValueError("stages must be a positive integer")
        if isinstance(actions, bool) or int(actions) != actions or actions < 1:
            raise ValueError("actions must be a positive integer")
        if basis not in {"total", "additive"}:
            raise ValueError("basis must be 'total' or 'additive'")
        transition_lambda = ridge_lambda if transition_ridge_lambda is None else transition_ridge_lambda
        for name, value in (
            ("ridge_lambda", ridge_lambda),
            ("transition_ridge_lambda", transition_lambda),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(gamma)) or not 0 < float(gamma) <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        for name, value in (
            ("r_max", r_max),
            ("reward_confidence_beta", reward_confidence_beta),
            ("transition_confidence_beta", transition_confidence_beta),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.q = int(q)
        self.feature_dim = int(feature_dim)
        self.stages = int(stages)
        self.actions = int(actions)
        self.basis = basis
        self.normalized_basis = bool(normalized_basis)
        self.ridge_lambda = float(ridge_lambda)
        self.transition_ridge_lambda = float(transition_lambda)
        self.gamma = float(gamma)
        self.r_max = float(r_max)
        self.reward_confidence_beta = float(reward_confidence_beta)
        self.transition_confidence_beta = float(transition_confidence_beta)
        self.dimension = int(
            legendre_features(
                torch.zeros(1, self.feature_dim),
                self.q,
                basis=self.basis,
                normalized=self.normalized_basis,
            ).shape[-1]
        )
        self.current_features: torch.Tensor | None = None
        self.rewards: torch.Tensor | None = None
        self.next_features: torch.Tensor | None = None
        self.next_action_features: torch.Tensor | None = None
        self.observation_mask: torch.Tensor | None = None
        self.transition_mask: torch.Tensor | None = None
        self.reward_theta: list[torch.Tensor | None] = [None] * self.stages
        self.reward_gram: list[torch.Tensor | None] = [None] * self.stages
        self.reward_counts: list[int] = [0] * self.stages
        self.transition_models: list[PolynomialTransitionEstimator | None] = [None] * self.stages
        self.last_plan: MacroLSVIPlan | None = None

    def _design(self, features: torch.Tensor) -> torch.Tensor:
        values = _as_tensor(features)
        if values.shape[-1] != self.feature_dim:
            raise ValueError("features do not match planner feature_dim")
        return legendre_features(
            values.reshape(-1, self.feature_dim),
            self.q,
            basis=self.basis,
            normalized=self.normalized_basis,
            strict_range=True,
        ).reshape(*values.shape[:-1], self.dimension)

    def _fit_reward_models(self) -> None:
        assert self.current_features is not None
        assert self.rewards is not None and self.observation_mask is not None
        design = self._design(self.current_features)
        self.reward_theta = [None] * self.stages
        self.reward_gram = [None] * self.stages
        self.reward_counts = [0] * self.stages
        for stage in range(self.stages):
            valid = self.observation_mask[:, stage]
            if not bool(valid.any()):
                continue
            x = design[valid, stage]
            y = self.rewards[valid, stage].to(dtype=x.dtype)
            gram = x.T @ x
            eye = torch.eye(self.dimension, dtype=x.dtype, device=x.device)
            self.reward_theta[stage] = torch.linalg.solve(gram + self.ridge_lambda * eye, x.T @ y)
            self.reward_gram[stage] = gram
            self.reward_counts[stage] = int(valid.sum().item())

    def fit(
        self,
        current_features: torch.Tensor,
        rewards: torch.Tensor,
        next_features: torch.Tensor,
        next_action_features: torch.Tensor,
        *,
        observation_mask: torch.Tensor | None = None,
        transition_mask: torch.Tensor | None = None,
    ) -> "MacroLSVIPlanner":
        """Fit the planner on observed macro rows.

        Shapes are ``[N, stages, feature_dim]`` for current and successor
        state-action features, ``[N, stages]`` for cumulative rewards, and
        ``[N, stages, actions, feature_dim]`` for candidate actions at each
        observed successor boundary.  The current feature is the frozen
        state-action representation used by the reward and transition model;
        callers must keep this representation fixed across planner updates.
        """

        current = _as_tensor(current_features)
        reward = _as_tensor(rewards, device=current.device)
        successor = _as_tensor(next_features, device=current.device)
        next_candidates = _as_tensor(next_action_features, device=current.device)
        expected = (current.shape[0], self.stages, self.feature_dim)
        if current.ndim != 3 or tuple(current.shape) != expected:
            raise ValueError("current_features must have shape [N, stages, feature_dim]")
        if reward.shape != current.shape[:2]:
            raise ValueError("rewards must have shape [N, stages]")
        if successor.shape != current.shape:
            raise ValueError("next_features must match current_features")
        if next_candidates.shape != (current.shape[0], self.stages, self.actions, self.feature_dim):
            raise ValueError("next_action_features must have shape [N, stages, actions, feature_dim]")
        for name, value in (
            ("current_features", current),
            ("rewards", reward),
            ("next_features", successor),
            ("next_action_features", next_candidates),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite values")
        # Validate the frozen feature range before any state is mutated.
        self._design(current.reshape(-1, self.feature_dim))
        self._design(successor.reshape(-1, self.feature_dim))
        self._design(next_candidates.reshape(-1, self.feature_dim))
        observed = torch.ones(current.shape[:2], dtype=torch.bool, device=current.device)
        if observation_mask is not None:
            observed = _as_tensor(observation_mask, device=current.device).to(torch.bool)
            if observed.shape != current.shape[:2]:
                raise ValueError("observation_mask must have shape [N, stages]")
        transitions = observed.clone()
        if transition_mask is not None:
            transitions = _as_tensor(transition_mask, device=current.device).to(torch.bool)
            if transitions.shape != current.shape[:2]:
                raise ValueError("transition_mask must have shape [N, stages]")
            transitions = transitions & observed
        self.current_features = current.detach().clone()
        self.rewards = reward.detach().clone()
        self.next_features = successor.detach().clone()
        self.next_action_features = next_candidates.detach().clone()
        self.observation_mask = observed.detach().clone()
        self.transition_mask = transitions.detach().clone()
        self.transition_models = [None] * self.stages
        self._fit_reward_models()
        self.last_plan = None
        return self

    def update(
        self,
        current_features: torch.Tensor,
        rewards: torch.Tensor,
        next_features: torch.Tensor,
        next_action_features: torch.Tensor,
        *,
        observation_mask: torch.Tensor | None = None,
        transition_mask: torch.Tensor | None = None,
    ) -> "MacroLSVIPlanner":
        """Append one batch of macro rows and refit sufficient statistics."""

        if self.current_features is None:
            return self.fit(
                current_features,
                rewards,
                next_features,
                next_action_features,
                observation_mask=observation_mask,
                transition_mask=transition_mask,
            )
        current, rewards, successor, next_candidates = (
            current_features,
            rewards,
            next_features,
            next_action_features,
        )
        current = _as_tensor(current, device=self.current_features.device)
        rewards = _as_tensor(rewards, device=self.current_features.device)
        successor = _as_tensor(successor, device=self.current_features.device)
        next_candidates = _as_tensor(next_candidates, device=self.current_features.device)
        old_observed = self.observation_mask
        old_transitions = self.transition_mask
        assert old_observed is not None and old_transitions is not None
        new_observed = torch.ones(rewards.shape, dtype=torch.bool, device=current.device)
        if observation_mask is not None:
            new_observed = _as_tensor(observation_mask, device=current.device).to(torch.bool)
        new_transitions = new_observed.clone()
        if transition_mask is not None:
            new_transitions = _as_tensor(transition_mask, device=current.device).to(torch.bool) & new_observed
        return self.fit(
            torch.cat((self.current_features, current), dim=0),
            torch.cat((self.rewards, rewards), dim=0),
            torch.cat((self.next_features, successor), dim=0),
            torch.cat((self.next_action_features, next_candidates), dim=0),
            observation_mask=torch.cat((old_observed, new_observed), dim=0),
            transition_mask=torch.cat((old_transitions, new_transitions), dim=0),
        )

    def _predict_reward(self, stage: int, candidates: torch.Tensor) -> torch.Tensor:
        design = self._design(candidates)
        theta = self.reward_theta[stage]
        if theta is None:
            return torch.zeros(candidates.shape[:-1], dtype=candidates.dtype, device=candidates.device)
        return design @ theta.to(device=design.device, dtype=design.dtype)

    def _evaluate_stage(
        self,
        stage: int,
        candidates: torch.Tensor,
        stage_models: Sequence[PolynomialTransitionEstimator | None],
        durations: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return optimistic Q, reward bonus, and transition bonus."""

        if candidates.ndim < 2 or candidates.shape[-1] != self.feature_dim:
            raise ValueError("candidate features must end in feature_dim")
        flat = candidates.reshape(-1, self.feature_dim)
        reward = self._predict_reward(stage, flat).reshape(candidates.shape[:-1])
        transition = stage_models[stage]
        future_bound = durations[stage + 1 :].sum() * self.r_max
        if transition is None:
            successor = torch.full_like(reward, future_bound)
            transition_bonus = torch.zeros_like(reward)
        else:
            successor = transition.predict_test_function(flat).reshape_as(reward)
            transition_bonus = transition_confidence_bonus(
                transition.features(flat),
                gram=transition.gram_matrix(),
                beta=self.transition_confidence_beta,
            ).reshape_as(reward)
            transition_bonus = transition_bonus * future_bound.to(transition_bonus)
        reward_bonus = torch.zeros_like(reward)
        if self.reward_theta[stage] is not None and self.reward_gram[stage] is not None:
            gram = self.reward_gram[stage].to(device=flat.device, dtype=flat.dtype)
            gram = gram + self.ridge_lambda * torch.eye(self.dimension, dtype=flat.dtype, device=flat.device)
            reward_bonus = confidence_bonus(
                self._design(flat),
                gram=gram,
                beta=self.reward_confidence_beta,
            ).reshape_as(reward)
        discount = torch.pow(
            torch.as_tensor(self.gamma, dtype=reward.dtype, device=reward.device), durations[stage]
        )
        q_values = reward + discount * successor + reward_bonus + transition_bonus
        upper = durations[stage:].sum() * self.r_max
        if self.reward_theta[stage] is None:
            q_values = torch.full_like(q_values, upper)
        q_values = q_values.clamp_min(0).minimum(upper.to(q_values))
        return q_values.masked_fill(~action_mask, -torch.inf), reward_bonus, transition_bonus

    def plan(
        self,
        candidate_features: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
    ) -> MacroLSVIPlan:
        """Run optimistic backward recursion over candidate macro actions.

        ``candidate_features`` may be ``[stages, actions, feature_dim]`` for
        one planning state or ``[stages, states, actions, feature_dim]`` for
        a batch of planning states.
        """

        if self.current_features is None or self.observation_mask is None or self.transition_mask is None:
            raise RuntimeError("fit must be called before plan")
        candidates = _as_tensor(candidate_features)
        if candidates.ndim == 3:
            candidates = candidates.unsqueeze(1)
        if candidates.ndim != 4 or candidates.shape[0] != self.stages or candidates.shape[2:] != (
            self.actions,
            self.feature_dim,
        ):
            raise ValueError("candidate_features must have shape [stages, states, actions, feature_dim]")
        self._design(candidates.reshape(-1, self.feature_dim))
        states = int(candidates.shape[1])
        if action_mask is None:
            valid_actions = torch.ones(
                (self.stages, states, self.actions), dtype=torch.bool, device=candidates.device
            )
        else:
            valid_actions = _as_tensor(action_mask, device=candidates.device).to(torch.bool)
            if valid_actions.ndim == 2:
                valid_actions = valid_actions.unsqueeze(1).expand(-1, states, -1)
            if valid_actions.shape != (self.stages, states, self.actions):
                raise ValueError("action_mask must have shape [stages, states, actions]")
        if not bool(valid_actions.any(dim=-1).all()):
            raise ValueError("each planning state must have at least one valid action")
        if durations is None:
            duration_values = torch.ones(self.stages, dtype=candidates.dtype, device=candidates.device)
        else:
            duration_values = _as_tensor(durations, device=candidates.device).flatten()
            if duration_values.shape != (self.stages,):
                raise ValueError("durations must have shape [stages]")
            if torch.any(~torch.isfinite(duration_values)) or torch.any(duration_values <= 0):
                raise ValueError("durations must be finite and positive")
        q_values = torch.zeros(
            (self.stages, states, self.actions), dtype=candidates.dtype, device=candidates.device
        )
        values = torch.zeros((self.stages, states), dtype=candidates.dtype, device=candidates.device)
        actions = torch.zeros((self.stages, states), dtype=torch.long, device=candidates.device)
        reward_bonuses = torch.zeros_like(q_values)
        transition_bonuses = torch.zeros_like(q_values)
        stage_models: list[PolynomialTransitionEstimator | None] = [None] * self.stages
        assert self.current_features is not None and self.next_features is not None
        assert self.next_action_features is not None and self.transition_mask is not None
        for stage in reversed(range(self.stages)):
            if stage < self.stages - 1:
                transition_rows = self.transition_mask[:, stage]
                if bool(transition_rows.any()):
                    next_candidates = self.next_action_features[transition_rows, stage + 1]
                    next_mask = torch.ones(
                        (next_candidates.shape[0], self.actions), dtype=torch.bool, device=next_candidates.device
                    )
                    next_q, _, _ = self._evaluate_stage(
                        stage + 1,
                        next_candidates,
                        stage_models,
                        duration_values,
                        next_mask,
                    )
                    next_values = next_q.max(dim=-1).values
                    transition = PolynomialTransitionEstimator(
                        self.q,
                        self.feature_dim,
                        basis=self.basis,
                        ridge_lambda=self.transition_ridge_lambda,
                    )
                    targets = next_values.detach()
                    transition.fit_test_function(
                        self.current_features[transition_rows, stage],
                        self.next_features[transition_rows, stage],
                        lambda successor, targets=targets: targets.to(
                            device=successor.device, dtype=successor.dtype
                        ),
                    )
                    stage_models[stage] = transition
            q_stage, reward_bonus, transition_bonus = self._evaluate_stage(
                stage,
                candidates[stage],
                stage_models,
                duration_values,
                valid_actions[stage],
            )
            q_values[stage] = q_stage
            reward_bonuses[stage] = reward_bonus
            transition_bonuses[stage] = transition_bonus
            actions[stage] = torch.argmax(q_stage, dim=-1)
            values[stage] = q_stage.gather(-1, actions[stage].unsqueeze(-1)).squeeze(-1)
        self.transition_models = stage_models
        plan = MacroLSVIPlan(
            q_values=q_values,
            values=values,
            actions=actions,
            reward_bonus=reward_bonuses,
            transition_bonus=transition_bonuses,
        )
        self.last_plan = plan
        return plan

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "q": self.q,
            "feature_dim": self.feature_dim,
            "stages": self.stages,
            "actions": self.actions,
            "basis": self.basis,
            "normalized_basis": self.normalized_basis,
            "ridge_lambda": self.ridge_lambda,
            "transition_ridge_lambda": self.transition_ridge_lambda,
            "gamma": self.gamma,
            "r_max": self.r_max,
            "reward_confidence_beta": self.reward_confidence_beta,
            "transition_confidence_beta": self.transition_confidence_beta,
            "current_features": None if self.current_features is None else self.current_features.detach().cpu(),
            "rewards": None if self.rewards is None else self.rewards.detach().cpu(),
            "next_features": None if self.next_features is None else self.next_features.detach().cpu(),
            "next_action_features": None
            if self.next_action_features is None
            else self.next_action_features.detach().cpu(),
            "observation_mask": None if self.observation_mask is None else self.observation_mask.cpu(),
            "transition_mask": None if self.transition_mask is None else self.transition_mask.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "MacroLSVIPlanner":
        if not isinstance(state, Mapping):
            raise ValueError("planner checkpoint must be a mapping")
        if int(state.get("version", cls.VERSION)) != cls.VERSION:
            raise ValueError("unsupported MacroLSVIPlanner checkpoint version")
        planner = cls(
            q=int(state["q"]),
            feature_dim=int(state["feature_dim"]),
            stages=int(state["stages"]),
            actions=int(state["actions"]),
            basis=state.get("basis", "total"),
            normalized_basis=bool(state.get("normalized_basis", False)),
            ridge_lambda=float(state.get("ridge_lambda", 1.0)),
            transition_ridge_lambda=float(
                state.get("transition_ridge_lambda", state.get("ridge_lambda", 1.0))
            ),
            gamma=float(state.get("gamma", 1.0)),
            r_max=float(state.get("r_max", 1.0)),
            reward_confidence_beta=float(state.get("reward_confidence_beta", 1.0)),
            transition_confidence_beta=float(state.get("transition_confidence_beta", 1.0)),
        )
        if state.get("current_features") is not None:
            required = ("rewards", "next_features", "next_action_features")
            if any(state.get(key) is None for key in required):
                raise ValueError("planner checkpoint is missing fitted data")
            planner.fit(
                state["current_features"],
                state["rewards"],
                state["next_features"],
                state["next_action_features"],
                observation_mask=state.get("observation_mask"),
                transition_mask=state.get("transition_mask"),
            )
        return planner


class SuffixFeedbackCollector:
    """Budget-aware Monte Carlo suffix verifier (Eq. 23-25)."""

    def __init__(self, verifier, *, budget: SegmentRewardBudget, samples_per_boundary: int = 1):
        if samples_per_boundary < 1:
            raise ValueError("samples_per_boundary must be positive")
        self.verifier = verifier
        self.budget = budget
        self.samples_per_boundary = samples_per_boundary

    def collect(self, prefix_at_boundary: Sequence[Any]) -> torch.Tensor:
        utilities = []
        for prefix in prefix_at_boundary:
            if not self.budget.can_afford(self.samples_per_boundary):
                raise RuntimeError("feedback budget exhausted during suffix collection")
            outcomes = [float(self.verifier(prefix)) for _ in range(self.samples_per_boundary)]
            self.budget.consume(self.samples_per_boundary)
            utilities.append(sum(outcomes) / len(outcomes))
        return suffix_potential_differences(torch.as_tensor(utilities, dtype=torch.float32).unsqueeze(0)).squeeze(0)


def enumerate_multi_indices(feature_dim: int, q: int) -> list[tuple[int, ...]]:
    """Enumerate total-degree multi-indices in deterministic graded order."""

    if feature_dim < 1 or q < 0:
        raise ValueError("feature_dim must be positive and q must be non-negative")
    indices: list[tuple[int, ...]] = []
    for degree in range(q + 1):
        for values in itertools.product(range(degree + 1), repeat=feature_dim):
            if sum(values) == degree:
                indices.append(values)
    return indices


def _as_tensor(value: Any, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _prepare_segment_durations(
    segment_durations: torch.Tensor | None,
    reference: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Normalize duration metadata, ignoring values in padded segment slots.

    Rollout transports commonly pad ragged segment arrays with zero durations.
    A padded duration is not an environment transition and therefore must not be
    validated or contribute to a reverse discount/remaining-horizon recursion.
    Valid slots still require finite, strictly positive durations. Padding is
    replaced by one as a neutral value so callers can safely exponentiate before
    applying their validity mask.
    """

    if valid.shape != reference.shape:
        raise ValueError("segment_mask shape must match segment data")
    valid = valid.to(device=reference.device, dtype=torch.bool)
    if segment_durations is None:
        durations = torch.ones_like(reference)
    else:
        durations = _as_tensor(segment_durations, device=reference.device)
        if durations.ndim == reference.ndim - 1:
            durations = durations.unsqueeze(0)
        if durations.shape != reference.shape:
            raise ValueError("segment_durations must match segment data")
        valid_durations = durations[valid]
        if torch.any(~torch.isfinite(valid_durations)) or torch.any(valid_durations <= 0):
            raise ValueError("segment_durations must be finite and positive on valid segments")
        durations = torch.where(valid, durations, torch.ones_like(durations))
    return durations


def _remaining_segment_duration(
    segment_durations: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return the undiscounted duration remaining from every segment.

    Padded slots are excluded before the reverse cumulative sum.  Keeping this
    operation in one helper makes the value-range contract identical for the
    return critic, the transition backup, and the public Bellman helper.
    """

    durations = _as_tensor(segment_durations)
    mask = _as_tensor(valid, dtype=torch.bool, device=durations.device)
    if durations.ndim != 2 or mask.shape != durations.shape:
        raise ValueError("segment durations and mask must have shape [batch, segments]")
    return torch.flip(
        torch.cumsum(torch.flip(durations * mask.to(durations.dtype), dims=[1]), dim=1),
        dims=[1],
    )


def _clip_segment_rewards_to_r_max(
    rewards: torch.Tensor,
    segment_durations: torch.Tensor,
    valid: torch.Tensor,
    r_max: float,
) -> torch.Tensor:
    """Apply the per-segment optimistic reward range from the Methods.

    Rewards may be signed (for example potential differences), so this is an
    upper bound only.  A lower clamp would destroy the exact telescoping
    semantics of signed utility feedback.
    """

    if r_max < 0 or not math.isfinite(float(r_max)):
        raise ValueError("r_max must be finite and non-negative")
    durations = _prepare_segment_durations(segment_durations, rewards, valid)
    mask = valid.to(dtype=torch.bool, device=rewards.device)
    upper = durations.to(dtype=rewards.dtype) * float(r_max)
    return torch.minimum(rewards, upper) * mask.to(dtype=rewards.dtype)


def _clip_segment_values_to_r_max(
    values: torch.Tensor,
    segment_durations: torch.Tensor,
    valid: torch.Tensor,
    r_max: float,
) -> torch.Tensor:
    """Clip macro value/return estimates to remaining-duration range."""

    if r_max < 0 or not math.isfinite(float(r_max)):
        raise ValueError("r_max must be finite and non-negative")
    mask = valid.to(dtype=torch.bool, device=values.device)
    remaining = _remaining_segment_duration(segment_durations, mask).to(dtype=values.dtype)
    upper = remaining * float(r_max)
    return torch.minimum(values, upper) * mask.to(dtype=values.dtype)


def _pre_update_regularized_gram(
    dimension: int,
    regularizer: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    state_gram: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the design Gram matrix available before the current batch.

    Online confidence bonuses must not use the samples whose policy update they
    are driving.  With no persistent estimator state, the prior is the ridge
    regularizer; with a state, only its already-settled sufficient statistics
    are used.
    """

    eye = torch.eye(dimension, dtype=dtype, device=device)
    if state_gram is None:
        return float(regularizer) * eye
    return _as_tensor(state_gram, dtype=dtype, device=device) + float(regularizer) * eye


def ridge_feature_norms(features: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """Compute elliptical feature norms used by reward/transition confidence bounds.

    ``gram`` is a positive-definite ridge-regularized design matrix.  Solving
    the linear system instead of explicitly inverting it keeps the estimator
    numerically stable for the high-order Legendre ablations.
    """
    x = _as_tensor(features)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    v = _as_tensor(gram, device=x.device)
    if v.ndim != 2 or v.shape[0] != v.shape[1] or v.shape[0] != x.shape[-1]:
        raise ValueError("gram must be square with one row per feature")
    solved = torch.linalg.solve(v, x.T).T
    return torch.sqrt((x * solved).sum(dim=-1).clamp_min(0))


def confidence_bonus(
    features: torch.Tensor,
    *,
    gram: torch.Tensor | None = None,
    ridge_lambda: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute a reward confidence bonus ``beta ||phi||_{V^{-1}}``.

    When no prior design matrix is supplied, the current batch is used to
    form ``V = X^T X + lambda I``.  This fallback is useful for diagnostics;
    online callers should pass the accumulated design Gram matrix.
    """
    if ridge_lambda <= 0 or beta < 0:
        raise ValueError("ridge_lambda must be positive and beta must be non-negative")
    x = _as_tensor(features)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if gram is None:
        gram = x.T @ x + ridge_lambda * torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return float(beta) * ridge_feature_norms(x, gram)


def transition_confidence_bonus(
    features: torch.Tensor,
    *,
    gram: torch.Tensor | None = None,
    ridge_lambda: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Alias with transition semantics for the macro Bellman uncertainty term."""
    return confidence_bonus(features, gram=gram, ridge_lambda=ridge_lambda, beta=beta)


def _legendre_values(x: torch.Tensor, q: int, normalized: bool) -> torch.Tensor:
    """Return ``[..., q + 1]`` one-dimensional Legendre values."""

    values = [torch.ones_like(x)]
    if q >= 1:
        values.append(x)
    for degree in range(1, q):
        values.append(((2 * degree + 1) * x * values[-1] - degree * values[-2]) / (degree + 1))
    result = torch.stack(values, dim=-1)
    if normalized:
        scales = torch.sqrt((torch.arange(0, q + 1, device=x.device, dtype=x.dtype) * 2 + 1) / 2)
        result = result * scales
    return result


def legendre_features(
    z: torch.Tensor | np.ndarray | Sequence[float],
    q: int,
    *,
    basis: str = "total",
    normalized: bool = False,
    strict_range: bool = False,
) -> torch.Tensor:
    """Compute total-degree or additive Legendre features.

    ``z`` has shape ``(..., p)`` and values in ``[-1, 1]``.  The total-degree
    output has ``binom(p + q, p)`` columns.  The additive output is
    ``1 + p*q`` columns and avoids tensor-product dimension growth.
    """

    if q < 0:
        raise ValueError("q must be non-negative")
    if basis not in {"total", "additive"}:
        raise ValueError("basis must be 'total' or 'additive'")
    z = _as_tensor(z)
    if z.ndim == 0:
        z = z.reshape(1, 1)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if strict_range and (torch.any(z < -1) or torch.any(z > 1)):
        raise ValueError("Legendre inputs must lie in [-1, 1]")
    one_dim = _legendre_values(z, q, normalized)
    if basis == "additive":
        columns = [torch.ones_like(z[..., 0])]
        for dim in range(z.shape[-1]):
            columns.extend(one_dim[..., dim, degree] for degree in range(1, q + 1))
        return torch.stack(columns, dim=-1)
    columns = []
    for alpha in enumerate_multi_indices(z.shape[-1], q):
        term = torch.ones_like(z[..., 0])
        for dim, degree in enumerate(alpha):
            term = term * one_dim[..., dim, degree]
        columns.append(term)
    return torch.stack(columns, dim=-1)


def make_segment_slices(length: int, m: int) -> list[tuple[int, int]]:
    """Partition ``length`` steps into ``m`` balanced, half-open intervals."""

    if length < 1 or m < 1 or m > length:
        raise ValueError(f"require 1 <= m <= length, got m={m}, length={length}")
    base, remainder = divmod(length, m)
    slices = []
    start = 0
    for index in range(m):
        end = start + base + (index < remainder)
        slices.append((start, end))
        start = end
    return slices


def segment_ids_from_lengths(lengths: Sequence[int], *, device=None) -> torch.Tensor:
    if not lengths or any(int(length) <= 0 for length in lengths):
        raise ValueError("segment lengths must be a non-empty sequence of positive integers")
    return torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.long, device=device),
        torch.as_tensor(lengths, dtype=torch.long, device=device),
    )


def segment_ids(length: int, m: int, *, device=None) -> torch.Tensor:
    return segment_ids_from_lengths([end - start for start, end in make_segment_slices(length, m)], device=device)


def pool_token_values(
    token_values: torch.Tensor,
    ids: torch.Tensor,
    *,
    num_segments: int | None = None,
    mask: torch.Tensor | None = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Pool token values into segments while respecting the response mask."""

    if token_values.ndim == 1:
        token_values = token_values.unsqueeze(0)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0).expand(token_values.shape[0], -1)
    if token_values.shape != ids.shape:
        raise ValueError(f"token_values and ids must have the same shape, got {token_values.shape} vs {ids.shape}")
    if mask is None:
        mask = torch.ones_like(token_values, dtype=torch.bool)
    else:
        mask = mask.to(dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
    if num_segments is None:
        num_segments = int(ids.max().item()) + 1 if ids.numel() else 0
    if num_segments < 1:
        return token_values.new_zeros((token_values.shape[0], 0))
    if torch.any(ids < 0) or torch.any(ids >= num_segments):
        raise ValueError("segment ids must be in [0, num_segments)")
    out = token_values.new_zeros((token_values.shape[0], num_segments))
    out.scatter_add_(1, ids, torch.where(mask, token_values, torch.zeros_like(token_values)))
    if reduction == "mean":
        counts = token_values.new_zeros((token_values.shape[0], num_segments))
        counts.scatter_add_(1, ids, mask.to(token_values.dtype))
        out = out / counts.clamp_min(1)
    elif reduction != "sum":
        raise ValueError("reduction must be 'sum' or 'mean'")
    return out


def scatter_segment_values(
    values: torch.Tensor,
    ids: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map ``[batch, segments]`` values to ``[batch, tokens]``."""

    if values.ndim == 1:
        values = values.unsqueeze(0)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0).expand(values.shape[0], -1)
    if torch.any(ids < 0) or torch.any(ids >= values.shape[1]):
        raise ValueError("segment ids point outside values")
    result = values.gather(1, ids)
    if mask is not None:
        result = result * mask.to(dtype=result.dtype)
    return result


def suffix_potential_differences(potentials: torch.Tensor, *, initial: float = 0.0) -> torch.Tensor:
    """Convert suffix utility/value estimates to potential-difference rewards."""

    if potentials.ndim == 1:
        potentials = potentials.unsqueeze(0)
    previous = torch.cat([potentials.new_full((potentials.shape[0], 1), initial), potentials[:, :-1]], dim=1)
    return potentials - previous


def macro_bellman_backup(
    segment_rewards: torch.Tensor,
    successor_values: torch.Tensor,
    *,
    reward_bonus: torch.Tensor | None = None,
    transition_bonus: torch.Tensor | None = None,
    segment_mask: torch.Tensor | None = None,
    segment_durations: torch.Tensor | None = None,
    gamma: float = 1.0,
    r_max: float | None = None,
    horizon: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform the optimistic macro Bellman backup from Eq. (92)-(94)."""

    rewards = _as_tensor(segment_rewards)
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
    successor = _as_tensor(successor_values, device=rewards.device)
    if successor.ndim == 1:
        successor = successor.unsqueeze(0)
    if successor.shape != rewards.shape:
        raise ValueError("successor_values must match segment_rewards")
    base_shape = rewards.shape[:2] if rewards.ndim == 3 else rewards.shape
    if segment_mask is None:
        valid = torch.ones(base_shape, dtype=torch.bool, device=rewards.device)
    else:
        valid = _as_tensor(segment_mask, device=rewards.device).to(torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != base_shape:
            raise ValueError("segment_mask must match the batch and segment dimensions")
    durations = _prepare_segment_durations(segment_durations, rewards.new_zeros(base_shape), valid)
    discount = torch.pow(torch.as_tensor(gamma, dtype=rewards.dtype, device=rewards.device), durations)
    if rewards.ndim == 3 and discount.ndim == 2:
        discount = discount.unsqueeze(-1)
    bonus = torch.zeros_like(rewards)
    if reward_bonus is not None:
        bonus = bonus + _as_tensor(reward_bonus, device=rewards.device)
    if transition_bonus is not None:
        bonus = bonus + _as_tensor(transition_bonus, device=rewards.device)
    raw_q_values = rewards + discount * successor
    if r_max is not None:
        if horizon is None or horizon < 1:
            raise ValueError("horizon is required when r_max is provided")
        if r_max < 0 or not math.isfinite(float(r_max)):
            raise ValueError("r_max must be finite and non-negative")
        # Bound the raw signed return, then add optimism.  In particular, do
        # not clamp potential-difference rewards to zero: utility deltas may be
        # negative and their exact telescoping sum must remain unchanged.
        remaining = torch.flip(
            torch.cumsum(torch.flip(durations * valid.to(durations.dtype), dims=[1]), dim=1), dims=[1]
        )
        if raw_q_values.ndim == 3:
            remaining = remaining.unsqueeze(-1)
        raw_q_values = raw_q_values.minimum(remaining * r_max)
    q_values = raw_q_values + bonus
    q_values = q_values * (valid.to(q_values.dtype).unsqueeze(-1) if q_values.ndim == 3 else valid.to(q_values.dtype))
    # The final dimension, when present, is the macro-action dimension.  A
    # batch of one observed action has no action axis and already represents
    # its Q value.  Temporal maxima would be a different operator and would
    # break the macro Bellman recursion.
    values = q_values.max(dim=-1).values if q_values.ndim == 3 else q_values
    return q_values, values


def segment_log_prob_ratio(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    segment_ids_tensor: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
    num_segments: int | None = None,
    max_log_ratio: float = 20.0,
) -> torch.Tensor:
    """Compute one PPO ratio per macro action (Eq. 100)."""

    if log_probs.shape != old_log_probs.shape:
        raise ValueError("log_probs and old_log_probs must have the same shape")
    if segment_ids_tensor.shape != log_probs.shape:
        raise ValueError("segment_ids_tensor must match log_probs")
    if response_mask is None:
        response_mask = torch.ones_like(log_probs, dtype=torch.bool)
    safe_ids = segment_ids_tensor.clamp_min(0)
    if num_segments is None:
        num_segments = int(safe_ids.max().item()) + 1
    log_ratio = pool_token_values(log_probs - old_log_probs, safe_ids, num_segments=num_segments, mask=response_mask)
    return torch.exp(log_ratio.clamp(-max_log_ratio, max_log_ratio))


def segment_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    segment_advantages: torch.Tensor,
    segment_ids_tensor: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
    segment_mask: torch.Tensor | None = None,
    clip_ratio: float = 0.2,
    num_segments: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Clipped PPO objective over macro action ratios (Eq. 101)."""

    if response_mask is None:
        response_mask = torch.ones_like(log_probs, dtype=torch.bool)
    ratios = segment_log_prob_ratio(
        log_probs,
        old_log_probs,
        segment_ids_tensor,
        response_mask=response_mask,
        num_segments=num_segments or segment_advantages.shape[1],
    )
    advantages = _as_tensor(segment_advantages, device=ratios.device)
    unclipped = ratios * advantages
    clipped_ratios = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    clipped = clipped_ratios * advantages
    objective = torch.minimum(unclipped, clipped)
    if segment_mask is None:
        valid = torch.ones_like(objective, dtype=torch.bool)
    else:
        valid = segment_mask.to(dtype=torch.bool, device=objective.device)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != objective.shape:
            raise ValueError("segment_mask must match segment objective shape")
    if num_segments is not None and segment_advantages.shape[1] > num_segments:
        valid = valid & (torch.arange(segment_advantages.shape[1], device=objective.device)[None, :] < num_segments)
    loss = -(objective[valid]).mean() if torch.any(valid) else objective.sum() * 0.0
    metrics = {
        "segment/ratio_mean": ratios[valid].mean().detach() if torch.any(valid) else ratios.new_zeros(()),
        "segment/clip_fraction": (((ratios - clipped_ratios).abs() > 0).float()[valid].mean().detach()
                                   if torch.any(valid) else ratios.new_zeros(())),
        "segment/advantage_mean": advantages[valid].mean().detach() if torch.any(valid) else advantages.new_zeros(()),
    }
    return loss, metrics


def compute_segment_policy_loss(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    token_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    segment_ids_tensor: torch.Tensor | None = None,
    segment_feedback_mask: torch.Tensor | None = None,
    num_segments: int = 4,
    clip_ratio: float = 0.2,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply PPO to macro actions using a product of token probabilities.

    The existing verl PPO loss treats every token as an independent action.
    PolySeg's macro action is the whole segment, so its importance ratio is
    ``exp(sum_token(log pi - log pi_old))`` and its advantage is pooled once
    per segment. Production callers supply rollout-owned ids; inference from
    the response mask is retained only for explicit compatibility callers.
    Feedback metadata defines the macro axis when present. A configured
    segment count is a cap, so short responses may have fewer actual segments.
    """

    if segment_ids_tensor is None:
        segment_ids_tensor = _infer_ids_from_mask(response_mask, num_segments)
    ids = segment_ids_tensor.to(device=log_prob.device, dtype=torch.long)
    safe_ids = ids.clamp_min(0)
    if safe_ids.shape != log_prob.shape:
        raise ValueError("segment_ids_tensor must match log-probability shape")
    observed_segment_count = int(safe_ids.max().item()) + 1 if safe_ids.numel() else 0
    segment_count = max(num_segments, observed_segment_count)
    feedback = None
    if segment_feedback_mask is not None:
        feedback = segment_feedback_mask.to(device=log_prob.device, dtype=torch.bool)
        if feedback.ndim == 1:
            feedback = feedback.unsqueeze(0)
        if feedback.ndim != 2 or feedback.shape[0] != log_prob.shape[0]:
            raise ValueError("segment_feedback_mask batch axis must match log probabilities")
        # TransferQueue pads the macro axis independently in each actor
        # microbatch. A microbatch can therefore carry a mask narrower than
        # the token-aligned segment ids. Missing entries mean that no feedback
        # was transported for that segment and must be excluded from PPO;
        # never broadcast a shorter mask or mark those entries as observed.
        feedback_width = feedback.shape[1]
        # A configured macro axis larger than the observed response is a
        # batch-level contract: a narrow mask there means metadata was lost.
        # When the configured axis equals the observed short rollout, a
        # narrower mask is a legitimate ragged microbatch and is padded with
        # unqueried entries below.
        if feedback_width < observed_segment_count and num_segments > observed_segment_count:
            raise ValueError(
                "segment_feedback_mask must cover every observed segment id"
            )
        segment_count = max(segment_count, feedback_width)
        if feedback_width == 0:
            if torch.any(response_mask):
                raise ValueError("segment_feedback_mask must cover every observed segment")
            zero = log_prob.sum() * 0.0
            return zero, {
                "segment/ratio_mean": zero.detach(),
                "segment/clip_fraction": zero.detach(),
                "segment/advantage_mean": zero.detach(),
            }
        if feedback_width != segment_count:
            aligned = torch.zeros(
                (feedback.shape[0], segment_count), dtype=torch.bool, device=feedback.device
            )
            aligned[:, : min(feedback_width, segment_count)] = feedback[:, :segment_count]
            feedback = aligned
    segment_advantages = pool_token_values(
        token_advantages,
        safe_ids,
        num_segments=segment_count,
        mask=response_mask,
        reduction="mean",
    )
    segment_valid = pool_token_values(
        torch.ones_like(token_advantages),
        safe_ids,
        num_segments=segment_count,
        mask=response_mask,
        reduction="sum",
    ) > 0
    if feedback is not None:
        segment_valid = segment_valid & feedback
    if rollout_is_weights is not None:
        # Sequence-level rollout correction is represented at macro granularity.
        segment_weights = pool_token_values(
            rollout_is_weights,
            safe_ids,
            num_segments=segment_count,
            mask=response_mask,
            reduction="mean",
        )
        segment_advantages = segment_advantages * segment_weights
    loss, metrics = segment_ppo_loss(
        log_prob,
        old_log_prob,
        segment_advantages,
        safe_ids,
        response_mask=response_mask,
        segment_mask=segment_valid,
        clip_ratio=clip_ratio,
        num_segments=segment_count,
    )
    return loss, metrics


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    valid = values[mask]
    if valid.numel() <= 1:
        return values * mask.to(values.dtype)
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((values - mean) / std) * mask.to(values.dtype)


def compute_segment_gae_advantage(
    segment_rewards: torch.Tensor,
    segment_values: torch.Tensor | None = None,
    *,
    segment_mask: torch.Tensor | None = None,
    segment_durations: torch.Tensor | None = None,
    gamma: float = 1.0,
    lam: float = 1.0,
    normalize: bool = False,
    token_segment_ids: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute duration-aware macro GAE and optionally scatter it to tokens.

    ``segment_values`` may contain ``m`` current values or ``m + 1`` values;
    the latter explicitly supplies the terminal successor value.  For padded
    segments, ``segment_mask`` prevents both their reward and their discount
    from leaking into preceding segments.
    """

    rewards = _as_tensor(segment_rewards)
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
    batch, count = rewards.shape
    if segment_mask is None:
        valid = torch.ones_like(rewards, dtype=torch.bool)
    else:
        valid = segment_mask.to(dtype=torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != rewards.shape:
            raise ValueError("segment_mask shape must match segment_rewards")
    durations = _prepare_segment_durations(segment_durations, rewards, valid)
    if segment_values is None:
        values = torch.zeros_like(rewards)
    else:
        values = _as_tensor(segment_values, device=rewards.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[0] != batch or values.shape[1] not in {count, count + 1}:
            raise ValueError("segment_values must have shape [batch, m] or [batch, m+1]")
    current_values = values[:, :count]
    advantages = torch.zeros_like(rewards)
    next_value = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
    next_gae = torch.zeros_like(next_value)
    for j in reversed(range(count)):
        active = valid[:, j].to(rewards.dtype)
        if values.shape[1] == count + 1:
            successor = values[:, j + 1]
        else:
            successor = next_value
        discount = torch.pow(torch.as_tensor(gamma, dtype=rewards.dtype, device=rewards.device), durations[:, j])
        delta = rewards[:, j] + discount * successor - current_values[:, j]
        next_gae = (delta + discount * lam * next_gae) * active
        advantages[:, j] = next_gae
        # A masked slot breaks the chain completely.  Preserving the old
        # next_value here let a valid segment before a missing one use the
        # value of a later segment as its successor, which disagreed with
        # macro_return_targets' reset semantics.
        next_value = current_values[:, j] * active
    returns = (advantages + current_values) * valid.to(rewards.dtype)
    if normalize:
        advantages = _masked_whiten(advantages, valid)
    if token_segment_ids is None:
        return advantages, returns
    token_advantages = scatter_segment_values(advantages, token_segment_ids, mask=token_mask)
    token_returns = scatter_segment_values(returns, token_segment_ids, mask=token_mask)
    return advantages, returns, token_advantages, token_returns


def macro_return_targets(
    segment_rewards: torch.Tensor,
    *,
    segment_mask: torch.Tensor | None = None,
    segment_durations: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute the macro Monte-Carlo reward-to-go target.

    This is the target used by the frozen polynomial critic.  A segment
    reward is consumed once at its macro boundary; duration only changes the
    discount to the next boundary.  Padding is masked before the reverse
    recursion so it cannot leak into a preceding segment.
    """

    rewards = _as_tensor(segment_rewards)
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
    if rewards.ndim != 2:
        raise ValueError("segment_rewards must have shape [batch, segments]")
    batch, count = rewards.shape
    if segment_mask is None:
        valid = torch.ones_like(rewards, dtype=torch.bool)
    else:
        valid = _as_tensor(segment_mask, device=rewards.device).to(torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != rewards.shape:
            raise ValueError("segment_mask shape must match segment_rewards")
    durations = _prepare_segment_durations(segment_durations, rewards, valid)
    targets = torch.zeros_like(rewards)
    next_return = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
    gamma_tensor = torch.as_tensor(gamma, dtype=rewards.dtype, device=rewards.device)
    for index in reversed(range(count)):
        active = valid[:, index].to(rewards.dtype)
        discount = torch.pow(gamma_tensor, durations[:, index])
        next_return = (rewards[:, index] + discount * next_return) * active
        targets[:, index] = next_return
    return targets


def _extract_polyseg_field(batch: Any, name: str, default=None):
    if batch is None:
        return default
    try:
        if name in batch:
            return batch[name]
    except (TypeError, KeyError):
        pass
    if hasattr(batch, "get"):
        return batch.get(name, default)
    return default


def _uniform_polyseg_scalar(value: Any, name: str) -> int | None:
    """Return a scalar adaptive selector value, rejecting mixed-q batches."""
    if value is None:
        return None
    tensor = torch.as_tensor(value).reshape(-1)
    if tensor.numel() == 0:
        return None
    first = int(tensor[0].item())
    if torch.any(tensor != tensor[0]):
        raise ValueError(f"PolySeg adaptive {name} must be uniform within one advantage batch")
    return first


def _infer_ids_from_mask(response_mask: torch.Tensor, m: int) -> torch.Tensor:
    ids = []
    for row in response_mask:
        length = int(row.sum().item())
        row_ids = (
            torch.empty(0, dtype=torch.long, device=response_mask.device)
            if length == 0
            else segment_ids(length, min(m, length), device=response_mask.device)
        )
        padded = torch.full((response_mask.shape[1],), -1, dtype=torch.long, device=response_mask.device)
        padded[:length] = row_ids
        ids.append(padded)
    return torch.stack(ids, dim=0)


def _critic_observation_weights(
    segment_durations: torch.Tensor,
    valid: torch.Tensor,
    weighting: str,
) -> torch.Tensor | None:
    """Return normalized macro-critic weights for the configured protocol.

    Agentic segments can represent different amounts of elapsed environment
    time.  Duration weighting gives the frozen critic the same weighted
    observation measure used by the dependence/ESS diagnostic.  We normalize
    the positive weights to mean one over the current fit so the target scale
    and confidence radii remain comparable to the unit-weight protocol.
    """

    mode = str(weighting).strip().lower()
    if mode == "uniform":
        return None
    if mode != "duration":
        raise ValueError("critic_weighting must be 'uniform' or 'duration'")
    durations = _as_tensor(segment_durations, device=valid.device).to(torch.float32)
    mask = valid.to(torch.bool)
    if durations.shape != mask.shape:
        raise ValueError("segment_durations and valid mask must have the same shape")
    weights = durations.clamp_min(0.0)
    if not bool(mask.any()):
        return weights
    positive = weights[mask]
    mean = positive.mean().clamp_min(torch.finfo(weights.dtype).eps)
    weights = weights / mean
    # Padded observations are never passed to fit, but keeping their weight at
    # zero makes accidental future flattening fail closed.
    return weights * mask.to(weights.dtype)


def compute_polyseg_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    values: torch.Tensor | None = None,
    config: Any = None,
    batch: Any = None,
    non_tensor_batch: Any = None,
    segment_rewards: torch.Tensor | None = None,
    segment_values: torch.Tensor | None = None,
    segment_ids_tensor: torch.Tensor | None = None,
    segment_mask: torch.Tensor | None = None,
    segment_feedback_mask: torch.Tensor | None = None,
    segment_durations: torch.Tensor | None = None,
    segment_features: torch.Tensor | None = None,
    segment_targets: torch.Tensor | None = None,
    segment_next_features: torch.Tensor | None = None,
    segment_transition_mask: torch.Tensor | None = None,
    step_features: torch.Tensor | None = None,
    shared_step_reward_estimator: SharedStepRewardEstimator | None = None,
    estimator_state: PolynomialEstimatorState | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """verl-compatible entry point for macro GAE.

    Segment tensors can be carried in ``DataProto.batch`` under the names
    ``segment_rewards``, ``segment_values``, ``segment_ids``,
    ``segment_mask`` and ``segment_durations``.  The explicit keyword names
    are useful for unit tests and direct integration with custom trainers.
    """

    # The rollout controller chooses q/m at batch boundaries.  Carrying the
    # selected values in non-tensor metadata keeps the estimator synchronized
    # with the actual segment layout while retaining fixed-config behavior for
    # older callers.
    selected_q = _extract_polyseg_field(batch, "polyseg_adaptive_q")
    if selected_q is None:
        selected_q = _extract_polyseg_field(non_tensor_batch, "polyseg_adaptive_q")
    selected_m = _extract_polyseg_field(batch, "polyseg_adaptive_m")
    if selected_m is None:
        selected_m = _extract_polyseg_field(non_tensor_batch, "polyseg_adaptive_m")
    q_override = _uniform_polyseg_scalar(selected_q, "q")
    m_override = _uniform_polyseg_scalar(selected_m, "m")
    cfg = PolySegConfig.from_any(config, q=q_override, m=m_override)
    if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"):
        if q_override is not None:
            non_tensor_batch["polyseg_adaptive_q"] = np.asarray(selected_q)
        if m_override is not None:
            non_tensor_batch["polyseg_adaptive_m"] = np.asarray(selected_m)
    estimator_state = _resolve_estimator_state(cfg, estimator_state)
    dependence_state = _resolve_dependence_state(cfg)
    pending_dependence_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    pending_dependence_radius: float | None = None

    def _commit_dependence_state() -> None:
        """Commit diagnostics only after the batch has produced an advantage."""

        if pending_dependence_history is not None:
            design, targets_for_history, valid_for_history, weights_for_history, scores = pending_dependence_history
            seen: set[int] = set()
            for state in (estimator_state, dependence_state):
                if state is None or id(state) in seen:
                    continue
                seen.add(id(state))
                state.append_dependence_history(
                    design,
                    targets_for_history,
                    valid_for_history,
                    weights=weights_for_history,
                    score_residuals=scores,
                )
        if dependence_state is not None and pending_dependence_radius is not None:
            dependence_state.update_dependence_radius(pending_dependence_radius)

    token_level_rewards = _as_tensor(token_level_rewards)
    response_mask = response_mask.to(dtype=torch.bool)
    if segment_rewards is None:
        segment_rewards = _extract_polyseg_field(batch, "segment_rewards")
    if segment_rewards is None:
        segment_rewards = _extract_polyseg_field(non_tensor_batch, "segment_rewards")
    if segment_values is None:
        segment_values = _extract_polyseg_field(batch, "segment_values")
    if segment_values is None:
        segment_values = _extract_polyseg_field(non_tensor_batch, "segment_values")
    if segment_ids_tensor is None:
        segment_ids_tensor = _extract_polyseg_field(batch, "segment_ids")
    if segment_ids_tensor is None:
        segment_ids_tensor = _extract_polyseg_field(non_tensor_batch, "segment_ids")
    if segment_mask is None:
        segment_mask = _extract_polyseg_field(batch, "segment_mask")
    if segment_mask is None:
        segment_mask = _extract_polyseg_field(non_tensor_batch, "segment_mask")
    if segment_feedback_mask is None:
        segment_feedback_mask = _extract_polyseg_field(batch, "segment_feedback_mask")
    if segment_feedback_mask is None:
        segment_feedback_mask = _extract_polyseg_field(non_tensor_batch, "segment_feedback_mask")
    if segment_durations is None:
        segment_durations = _extract_polyseg_field(batch, "segment_durations")
    if segment_durations is None:
        segment_durations = _extract_polyseg_field(non_tensor_batch, "segment_durations")
    if segment_features is None:
        segment_features = _extract_polyseg_field(batch, "segment_features")
    if segment_features is None:
        segment_features = _extract_polyseg_field(non_tensor_batch, "segment_features")
    if segment_targets is None:
        segment_targets = _extract_polyseg_field(batch, "segment_targets")
    if segment_targets is None:
        segment_targets = _extract_polyseg_field(batch, "segment_returns")
    if segment_targets is None:
        segment_targets = _extract_polyseg_field(non_tensor_batch, "segment_targets")
    if segment_targets is None:
        segment_targets = _extract_polyseg_field(non_tensor_batch, "segment_returns")
    if segment_next_features is None:
        segment_next_features = _extract_polyseg_field(batch, "segment_next_features")
    if segment_next_features is None:
        segment_next_features = _extract_polyseg_field(non_tensor_batch, "segment_next_features")
    if segment_transition_mask is None:
        segment_transition_mask = _extract_polyseg_field(batch, "segment_transition_mask")
    if segment_transition_mask is None:
        segment_transition_mask = _extract_polyseg_field(non_tensor_batch, "segment_transition_mask")
    if step_features is None:
        step_features = _extract_polyseg_field(batch, "step_features")
    if step_features is None:
        step_features = _extract_polyseg_field(non_tensor_batch, "step_features")
    segment_ids_source = "rollout"
    if segment_ids_tensor is None:
        if cfg.use_polynomial_critic or cfg.reward_route == "shared_step":
            # Equal-width inference here can disagree with the event boundaries
            # used later by the PPO ratio, which silently misassigns macro
            # advantages.  Research paths must supply rollout-owned IDs.
            raise ValueError(
                "PolySeg research advantages require rollout-owned segment_ids; "
                "equal-width inference is only available for explicit smoke callers"
            )
        segment_ids_tensor = _infer_ids_from_mask(response_mask, cfg.m)
        segment_ids_source = "inferred_equal_width"
    else:
        segment_ids_tensor = segment_ids_tensor.to(device=token_level_rewards.device, dtype=torch.long)
    if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"):
        non_tensor_batch["polyseg_segment_ids_source"] = np.full(
            (token_level_rewards.shape[0],), 0.0 if segment_ids_source == "rollout" else 1.0,
            dtype=np.float32,
        )
    # ``-1`` denotes response padding.  Pooling cannot accept it as an index,
    # so replace it temporarily and restore the mask through response_mask.
    safe_ids = segment_ids_tensor.clamp_min(0)
    had_explicit_segment_rewards = segment_rewards is not None
    if segment_rewards is None:
        segment_rewards = pool_token_values(
            token_level_rewards, safe_ids, num_segments=cfg.m, mask=response_mask, reduction="sum"
        )
    else:
        segment_rewards = _as_tensor(segment_rewards, device=token_level_rewards.device)
    if segment_mask is None:
        segment_mask = torch.arange(segment_rewards.shape[1], device=token_level_rewards.device)[None, :] < (
            safe_ids.max(dim=1).values[:, None] + 1
        )
    else:
        segment_mask = _as_tensor(segment_mask, device=token_level_rewards.device).to(torch.bool)
        if segment_mask.ndim == 1:
            segment_mask = segment_mask.unsqueeze(0)
    if segment_feedback_mask is not None:
        segment_feedback_mask = _as_tensor(segment_feedback_mask, device=token_level_rewards.device).to(torch.bool)
        if segment_feedback_mask.ndim == 1:
            segment_feedback_mask = segment_feedback_mask.unsqueeze(0)
        if segment_feedback_mask.shape != segment_mask.shape:
            raise ValueError("segment_feedback_mask must match segment_rewards and segment_mask")
        # Keep rollout structure for audit, but exclude unqueried fallback
        # segments from critic fitting and macro advantage propagation.
        segment_mask = segment_mask & segment_feedback_mask
    if cfg.reward_route == "shared_step":
        if step_features is None:
            raise ValueError(
                "reward_route='shared_step' requires rollout-provided frozen per-step features"
            )
        recovered, fitted_shared = recover_step_rewards_from_segment_feedback(
            _as_tensor(step_features, device=token_level_rewards.device),
            segment_ids_tensor,
            segment_rewards,
            ridge_lambda=cfg.ridge_lambda,
            segment_mask=segment_mask,
            step_mask=response_mask,
            q=cfg.q,
            basis=cfg.basis,
            normalized_basis=cfg.normalized_basis,
        )
        segment_rewards = pool_token_values(
            recovered,
            safe_ids,
            num_segments=segment_rewards.shape[1],
            mask=response_mask,
            reduction="sum",
        )
        if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"):
            non_tensor_batch["polyseg_shared_step_reward_recovered"] = recovered.detach().cpu().numpy()
            non_tensor_batch["polyseg_shared_step_reward_theta"] = fitted_shared.theta.detach().cpu().numpy()
    # Legacy callers may omit durations; retain their historical unit-duration
    # semantics unless the caller supplied the metadata or an optimistic
    # transition/confidence path explicitly requires the theorem range.
    explicit_segment_durations = segment_durations is not None
    segment_durations = _prepare_segment_durations(segment_durations, segment_rewards, segment_mask)
    enforce_value_range = bool(
        explicit_segment_durations or cfg.use_confidence_bonus or cfg.use_transition_estimator
    )
    if cfg.use_kl_in_reward and had_explicit_segment_rewards:
        # Reward managers expose raw macro verifier feedback separately from
        # token_level_rewards.  Recover only the auxiliary token component
        # (normally KL shaping) and add it once per action segment; the sparse
        # verifier reward itself must not be double-counted.
        sparse_macro = torch.zeros_like(token_level_rewards)
        for row in range(segment_rewards.shape[0]):
            for segment_id in range(segment_rewards.shape[1]):
                positions = torch.nonzero(
                    (safe_ids[row] == segment_id) & response_mask[row], as_tuple=False
                ).flatten()
                if positions.numel():
                    sparse_macro[row, int(positions[-1])] = segment_rewards[row, segment_id]
        auxiliary = token_level_rewards - sparse_macro
        segment_rewards = segment_rewards + pool_token_values(
            auxiliary, safe_ids, num_segments=segment_rewards.shape[1], mask=response_mask, reduction="sum"
        )
    values = None if values is None else _as_tensor(values, device=token_level_rewards.device)
    pooled_values = None
    if values is not None:
        pooled_values = pool_token_values(
            values,
            safe_ids,
            num_segments=segment_rewards.shape[1],
            mask=response_mask,
            reduction="mean",
        )
    if cfg.use_polynomial_critic and segment_features is None:
        # Older/remote reward-loop transports may carry only segment ids and
        # durations.  Reconstruct the same frozen observable feature map from
        # those boundaries rather than silently disabling the critic.  Tool
        # counts are unavailable in this fallback and therefore remain zero.
        derived_features = []
        response_width = response_mask.shape[1]
        for row in range(segment_rewards.shape[0]):
            row_ids = safe_ids[row]
            row_mask = segment_mask[row]
            boundaries = []
            for segment_id in range(segment_rewards.shape[1]):
                positions = torch.nonzero((row_ids == segment_id) & response_mask[row], as_tuple=False).flatten()
                if positions.numel():
                    boundaries.append((int(positions.min()), int(positions.max()) + 1))
                else:
                    boundaries.append((0, 1))
            durations_row = (
                torch.ones(segment_rewards.shape[1], dtype=torch.float32, device=segment_rewards.device)
                if segment_durations is None
                else _as_tensor(segment_durations[row], device=segment_rewards.device)
            )
            derived_features.append(
                build_segment_features(
                    boundaries,
                    durations_row,
                    response_length=cfg.feature_horizon or response_width,
                    feature_dim=cfg.feature_dim,
                ).to(device=segment_rewards.device)
            )
        segment_features = torch.stack(derived_features, dim=0)
    effective_segment_rewards = segment_rewards
    transition_successor_values = None
    transition_bonus = torch.zeros_like(segment_rewards)
    # Confidence bonuses use a conservative dependence correction.  The
    # multiplier remains one for the ordinary fixed-control path and when no
    # dependence diagnostics are available (for example a tiny compatibility
    # caller without a non-tensor side channel).
    pre_update_dependence_radius_multiplier = (
        1.0
        if dependence_state is None
        else max(float(dependence_state.dependence_radius_multiplier), 1.0)
    )
    # This is the multiplier actually used by the online bonus.  A stateful
    # confidence path must use the prior batch's value; the current batch's
    # diagnostic is committed only after the advantage is successfully built.
    dependence_radius_multiplier = pre_update_dependence_radius_multiplier
    if cfg.use_polynomial_critic and segment_features is not None:
        features = _as_tensor(segment_features, device=token_level_rewards.device)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[:2] != segment_rewards.shape:
            raise ValueError("segment_features must have shape [batch, segments, feature_dim]")
        valid = segment_mask.to(dtype=torch.bool, device=features.device)
        if not bool(valid.any()):
            # A controller-level budget fallback can leave an entire batch
            # without queried labels.  Preserve the structural metadata but
            # return a zero training signal instead of fitting a critic on
            # fabricated zeros or calling a dependence estimator with no
            # observations.
            if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"):
                non_tensor_batch["polyseg_feedback_available"] = np.zeros(
                    (segment_rewards.shape[0],), dtype=np.float32
                )
            return compute_segment_gae_advantage(
                torch.zeros_like(segment_rewards),
                segment_values,
                segment_mask=valid,
                segment_durations=segment_durations,
                gamma=cfg.gamma,
                lam=cfg.lam,
                normalize=cfg.normalize_advantages,
                token_segment_ids=safe_ids,
                token_mask=response_mask,
            )[-2:]
        # Enforce the per-segment reward range before any critic or transition
        # fit.  This is a no-op for already bounded verifier rewards, while it
        # prevents optimistic bonuses and malformed external scores from
        # escaping the theorem's duration-aware value range.
        if enforce_value_range:
            effective_segment_rewards = _clip_segment_rewards_to_r_max(
                effective_segment_rewards,
                segment_durations,
                valid,
                cfg.r_max,
            )
        # Dependence diagnostics operate on cross-fitted polynomial score
        # residuals. They are computed before the in-sample critic fit so the
        # confidence correction cannot be made artificially optimistic.
        if (
            (non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"))
            or estimator_state is not None
            or dependence_state is not None
        ):
            diagnostic_sink = (
                non_tensor_batch
                if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__")
                else {}
            )
            dependence = _polyseg_dependence_metrics(
                features,
                segment_rewards.to(dtype=features.dtype),
                valid,
                durations=segment_durations,
                q=cfg.q,
                basis=cfg.basis,
                ridge_lambda=cfg.ridge_lambda,
                binary=normalize_feedback_protocol(cfg.feedback) == "binary_local",
                logistic_l2=cfg.logistic_l2,
                weight_policy="segment_duration" if cfg.critic_weighting == "duration" else "unit",
                history_state=estimator_state if estimator_state is not None else dependence_state,
                persistent_history=estimator_state is not None or dependence_state is not None,
            )
            observed_dependence_radius_multiplier = float(
                max(dependence["iid_cluster_radius_ratio_upper"], 1.0)
            )
            design_dimension = int(
                legendre_features(
                    features.reshape(-1, features.shape[-1]), cfg.q, basis=cfg.basis
                ).shape[-1]
            )
            explicit_radius = dependence_confidence_radius(
                dimension=design_dimension,
                effective_samples=max(float(valid.sum().item()), 1.0),
                delta=cfg.confidence_delta,
                kappa_segment=float(dependence["kappa_segment_upper"]),
                kappa_trajectory=float(dependence["kappa_trajectory_upper"]),
            )
            # Keep the historical empirical multiplier by default.  Users can
            # opt into the explicit delta-dependent radius for theorem-aligned
            # optimism without changing old checkpoints' numerical behavior.
            if cfg.use_formal_confidence_radius:
                observed_dependence_radius_multiplier *= max(1.0, explicit_radius)
            if dependence_state is None:
                # Preserve the legacy non-stateful diagnostic behavior.
                dependence_radius_multiplier = observed_dependence_radius_multiplier
            else:
                pending_dependence_radius = observed_dependence_radius_multiplier
            diagnostic_fields = {
                "polyseg_dependence_available": float(
                    bool(dependence.get("dependence_available", True))
                ),
                "polyseg_dependence_missing": float(
                    not bool(dependence.get("dependence_available", True))
                ),
                "polyseg_kappa_segment": dependence["kappa_segment"],
                "polyseg_kappa_segment_upper": dependence["kappa_segment_upper"],
                "polyseg_kappa_segment_selected_upper": dependence.get(
                    "kappa_segment_upper_by_m", {}
                ).get(cfg.m, dependence["kappa_segment_upper"]),
                "polyseg_kappa_trajectory": dependence["kappa_trajectory"],
                "polyseg_kappa_trajectory_upper": dependence["kappa_trajectory_upper"],
                "polyseg_nominal_observations": dependence["nominal_observations"],
                "polyseg_dependence_cluster_count": dependence["cluster_count"],
                "polyseg_total_observation_weight": dependence["total_observation_weight"],
                "polyseg_weighted_ess_ratio": dependence["weighted_ess_ratio"],
                "polyseg_dependence_weighted_by_duration": float(
                    dependence["weight_policy"] == "segment_duration"
                ),
                "polyseg_effective_samples": dependence["effective_samples"],
                "polyseg_weighted_effective_samples": dependence["weighted_effective_samples"],
                "polyseg_iid_cluster_radius_ratio": dependence["iid_cluster_radius_ratio"],
                "polyseg_iid_cluster_radius_ratio_upper": dependence["iid_cluster_radius_ratio_upper"],
                "polyseg_chi_hat": dependence["chi_hat"],
                "polyseg_hac_lag": dependence["hac_lag"],
                "polyseg_dependence_diagnostic_oof": dependence["diagnostic_oof"],
                "polyseg_dependence_residual_rms": dependence["residual_rms"],
                # This historical field reports the observed post-update
                # diagnostic.  The separate fields make the online semantics
                # explicit for confidence-enabled runs.
                "polyseg_dependence_radius_multiplier": observed_dependence_radius_multiplier,
                "polyseg_dependence_radius_multiplier_pre_update": pre_update_dependence_radius_multiplier,
                "polyseg_dependence_radius_multiplier_post_update": observed_dependence_radius_multiplier,
                "polyseg_confidence_radius": explicit_radius,
                "polyseg_confidence_radius_pre_update": pre_update_dependence_radius_multiplier,
                "polyseg_confidence_radius_post_update": observed_dependence_radius_multiplier,
                "polyseg_dependence_state_batches": float(
                    dependence_state.dependence_diagnostic_batches + 1
                    if dependence_state is not None
                    else 0
                ),
                "polyseg_confidence_delta": cfg.confidence_delta,
                "polyseg_dependence_logistic_score": float(
                    dependence.get("dependence_score_model") == "logistic_fisher_score"
                ),
                "polyseg_dependence_persistent_oof": float(
                    dependence.get("dependence_oof_mode") == "persistent_prequential_history"
                ),
                "polyseg_dependence_history_trajectories": float(
                    len(estimator_state.dependence_history) if estimator_state is not None else 0
                ),
            }
            for field, value in diagnostic_fields.items():
                diagnostic_sink[field] = np.full(
                    (segment_rewards.shape[0],), float(value), dtype=np.float32
                )
            if estimator_state is not None or dependence_state is not None:
                # Defer the mutation until the complete advantage path returns;
                # a failed batch must not advance either history or radius.
                design = legendre_features(
                    features.reshape(-1, features.shape[-1]), cfg.q, basis=cfg.basis
                ).reshape(features.shape[0], features.shape[1], -1)
                score_rows = design * _as_tensor(dependence["oof_residuals"], device=design.device).unsqueeze(-1)
                history_weights = (
                    segment_durations
                    if cfg.critic_weighting == "duration"
                    else torch.ones_like(segment_rewards)
                )
                pending_dependence_history = (
                    design,
                    segment_rewards.to(dtype=design.dtype),
                    valid,
                    history_weights,
                    score_rows,
                )
        binary_feedback = normalize_feedback_protocol(cfg.feedback) == "binary_local"
        if binary_feedback:
            # Binary feedback is a Bernoulli observation of latent segment
            # utility.  Fitting a ridge regressor to 0/1 labels would violate
            # Eq. (22) and makes sigmoid saturation look like reward noise.
            # Use the regularized logistic estimator, then feed its clipped
            # latent logit into the macro Bellman recursion.  The exact
            # observed labels remain available in ``segment_rewards`` and the
            # probability-space MSE is reported as the estimator diagnostic.
            if cfg.critic_type != "polynomial":
                raise ValueError("binary feedback requires critic_type='polynomial'")
            if estimator_state is not None:
                raise ValueError("accumulate_estimator is not supported for binary logistic feedback")
            binary_labels = segment_rewards.to(dtype=features.dtype).clamp(0.0, 1.0)
            critic = PolynomialSegmentCritic(
                q=cfg.q,
                feature_dim=features.shape[-1],
                basis=cfg.basis,
                feedback="binary",
                logistic_l2=cfg.logistic_l2,
                ridge_lambda=cfg.ridge_lambda,
                normalized_basis=cfg.normalized_basis,
            )
            pre_update_gram = _pre_update_regularized_gram(
                critic.dimension,
                cfg.logistic_l2,
                device=features.device,
                dtype=features.dtype,
            )
            critic.fit(features[valid], binary_labels[valid])
            latent_values = critic.predict(
                features.reshape(-1, features.shape[-1]), latent=True
            ).reshape_as(segment_rewards)
            # Keep the latent logit as the binary reward signal, but use the
            # probability prediction as a distinct value baseline. Reusing the
            # same latent tensor for both makes the terminal TD error exactly
            # zero (r_T - V_T), cancelling the only terminal signal.
            latent_values = latent_values.clamp(-60.0, 60.0)
            probabilities = torch.sigmoid(latent_values)
            binary_value_baseline = probabilities
            confidence = torch.zeros_like(segment_rewards)
            if cfg.use_confidence_bonus:
                confidence[valid] = (
                    dependence_radius_multiplier
                    * confidence_bonus(
                        critic.features(features[valid]),
                        gram=pre_update_gram,
                        beta=cfg.reward_confidence_beta,
                    )
                ).to(dtype=confidence.dtype)
            effective_segment_rewards = (latent_values + confidence) * valid.to(latent_values.dtype)
            # The confidence term is optimistic, but the Methods still bound
            # each macro reward by its remaining-duration range.  Keep signed
            # potential-difference feedback intact by applying an upper-only
            # clip rather than a nonnegative clamp.
            effective_segment_rewards = _clip_segment_rewards_to_r_max(
                effective_segment_rewards,
                segment_durations,
                valid,
                cfg.r_max,
            )
            binary_value_baseline = _clip_segment_values_to_r_max(
                binary_value_baseline,
                segment_durations,
                valid,
                cfg.r_max,
            )
            next_values = torch.cat(
                [binary_value_baseline[:, 1:], torch.zeros_like(binary_value_baseline[:, :1])], dim=1
            )
            bellman_residual = effective_segment_rewards + torch.pow(
                torch.as_tensor(cfg.gamma, dtype=segment_rewards.dtype, device=segment_rewards.device),
                segment_durations,
            ) * next_values - binary_value_baseline
            critic_mse = (probabilities.detach() - binary_labels.detach()).square()[valid].mean()
            bellman_mse = bellman_residual.detach().square()[valid].mean()
            if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"):
                gram = critic.gram_matrix()
                eigvals = torch.linalg.eigvalsh(gram)
                non_tensor_batch["polyseg_critic_mse"] = np.full(
                    (segment_rewards.shape[0],), float(critic_mse.detach().cpu()), dtype=np.float32
                )
                non_tensor_batch["polyseg_bellman_residual_mse"] = np.full(
                    (segment_rewards.shape[0],), float(bellman_mse.detach().cpu()), dtype=np.float32
                )
                non_tensor_batch["polyseg_confidence_bonus"] = np.full(
                    (segment_rewards.shape[0],), float(confidence.detach().mean().cpu()), dtype=np.float32
                )
                non_tensor_batch["polyseg_confidence_pre_update"] = np.full(
                    (segment_rewards.shape[0],),
                    float(bool(cfg.use_confidence_bonus)),
                    dtype=np.float32,
                )
                non_tensor_batch["polyseg_feature_condition"] = np.full(
                    (segment_rewards.shape[0],), float(torch.linalg.cond(gram).detach().cpu()), dtype=np.float32
                )
                non_tensor_batch["polyseg_feature_min_eigenvalue"] = np.full(
                    (segment_rewards.shape[0],), float(eigvals.min().detach().cpu()), dtype=np.float32
                )
                non_tensor_batch["polyseg_feature_batch_samples"] = np.full(
                    (segment_rewards.shape[0],), float(valid.sum().detach().cpu()), dtype=np.float32
                )
                non_tensor_batch["polyseg_estimator_sample_count"] = np.full(
                    (segment_rewards.shape[0],), float(valid.sum().detach().cpu()), dtype=np.float32
                )
                non_tensor_batch["polyseg_feature_dimension"] = np.full(
                    (segment_rewards.shape[0],), float(critic.dimension), dtype=np.float32
                )
                non_tensor_batch["polyseg_binary_mean_probability"] = np.full(
                    (segment_rewards.shape[0],), float(probabilities[valid].detach().mean().cpu()), dtype=np.float32
                )
            result = compute_segment_gae_advantage(
                effective_segment_rewards,
                binary_value_baseline,
                segment_mask=valid,
                segment_durations=segment_durations,
                gamma=cfg.gamma,
                lam=cfg.lam,
                normalize=cfg.normalize_advantages,
                token_segment_ids=safe_ids,
                token_mask=response_mask,
            )
            _, _, token_advantages, token_returns = result
            _commit_dependence_state()
            return token_advantages, token_returns
        if cfg.polynomial_target == "return":
            if segment_targets is None:
                targets = macro_return_targets(
                    effective_segment_rewards,
                    segment_mask=valid,
                    segment_durations=segment_durations,
                    gamma=cfg.gamma,
                )
            else:
                targets = _as_tensor(segment_targets, device=features.device)
            if targets.ndim == 1:
                targets = targets.unsqueeze(0)
            if targets.shape != segment_rewards.shape:
                raise ValueError("segment_targets must have shape [batch, segments]")
        else:
            if segment_targets is not None:
                targets = _as_tensor(segment_targets, device=features.device)
                if targets.ndim == 1:
                    targets = targets.unsqueeze(0)
            elif pooled_values is not None:
                targets = pooled_values
            else:
                raise ValueError("polynomial_target='value' requires values or segment_targets")
            if targets.shape != segment_rewards.shape:
                raise ValueError("segment_targets must have shape [batch, segments]")
        # Critic targets are macro returns/value estimates.  Apply the same
        # remaining-duration upper envelope used by the Bellman helper so the
        # main advantage path cannot fit values outside the declared range.
        if enforce_value_range:
            targets = _clip_segment_values_to_r_max(
                targets,
                segment_durations,
                valid,
                cfg.r_max,
            )
        raw_targets = targets.detach().clone()
        if cfg.critic_type == "mlp":
            if cfg.use_transition_estimator or cfg.use_confidence_bonus:
                raise ValueError("MLP critic baseline does not support transition/confidence bonuses")
            critic = MLPSegmentCritic(
                feature_dim=features.shape[-1],
                hidden_dim=cfg.mlp_hidden_dim,
                learning_rate=cfg.mlp_learning_rate,
                steps=cfg.mlp_steps,
            )
        else:
            critic = PolynomialSegmentCritic(
                q=cfg.q,
                feature_dim=features.shape[-1],
                basis=cfg.basis,
                feedback="sum",
                ridge_lambda=cfg.ridge_lambda,
                normalized_basis=cfg.normalized_basis,
            )
        critic_weights = _critic_observation_weights(segment_durations, valid, cfg.critic_weighting)
        pre_update_gram = None
        if cfg.critic_type == "polynomial":
            pre_update_gram = _pre_update_regularized_gram(
                critic.dimension,
                cfg.ridge_lambda,
                device=features.device,
                dtype=features.dtype,
                state_gram=None if estimator_state is None else estimator_state.gram,
            )
        if estimator_state is not None and cfg.critic_type != "polynomial":
            raise ValueError("accumulating estimator state requires critic_type='polynomial'")
        confidence = torch.zeros_like(segment_rewards)
        # Fit once on the observed target to obtain the design matrix, then
        # add the optional optimism bonus and refit on the optimistic target.
        # This keeps the confidence term part of the same macro return target
        # used by the critic and the subsequent GAE recursion.
        critic.fit(
            features[valid],
            targets[valid],
            sample_weight=None if critic_weights is None else critic_weights[valid],
            accumulate=estimator_state is not None,
            state=estimator_state,
        )
        if estimator_state is not None and cfg.use_confidence_bonus:
            raise ValueError("confidence bonuses are not supported with accumulate_estimator")
        # Optional explicit macro transition route.  LLM rollouts generally
        # do not expose a state representation at every boundary, so this is
        # opt-in and requires caller-provided next-boundary features.  A
        # missing field is an error rather than a silent return-critic
        # fallback, which keeps the theoretical transition claim honest.
        if cfg.use_transition_estimator:
            if estimator_state is not None:
                raise ValueError(
                    "use_transition_estimator requires a non-accumulating critic: "
                    "the P V test function changes after each Bellman backup"
                )
            if segment_next_features is None:
                raise ValueError(
                    "use_transition_estimator=True requires segment_next_features "
                    "with one next-state feature vector per segment"
                )
            next_features = _as_tensor(segment_next_features, device=features.device)
            if next_features.ndim == 2:
                next_features = next_features.unsqueeze(0)
            if next_features.shape != features.shape:
                raise ValueError("segment_next_features must match segment_features shape")
            if segment_transition_mask is None:
                transition_valid = torch.zeros_like(valid)
                if valid.shape[1] > 1:
                    transition_valid[:, :-1] = valid[:, :-1] & valid[:, 1:]
            else:
                transition_valid = _as_tensor(segment_transition_mask, device=features.device).to(torch.bool)
                if transition_valid.ndim == 1:
                    transition_valid = transition_valid.unsqueeze(0)
                if transition_valid.shape != valid.shape:
                    raise ValueError("segment_transition_mask must match segment reward shape")
                transition_valid = transition_valid & valid
            if not torch.any(transition_valid):
                raise ValueError("segment_transition_mask contains no valid macro transitions")
            transition = PolynomialTransitionEstimator(
                cfg.q,
                features.shape[-1],
                basis=cfg.basis,
                ridge_lambda=cfg.transition_ridge_lambda,
            )
            transition_pre_update_gram = _pre_update_regularized_gram(
                transition.dimension,
                cfg.transition_ridge_lambda,
                device=features.device,
                dtype=features.dtype,
                state_gram=(
                    None
                    if estimator_state is None
                    else estimator_state.transition_gram
                ),
            )
            # Fit the transition estimator on the observed successor value
            # directly.  Fitting E[X'|X,U] and then evaluating V(E[X'])
            # changes the Bellman estimand for nonlinear critics; the required
            # object is E[V(X')|X,U] = P V.
            transition.fit_test_function(
                features[transition_valid],
                next_features[transition_valid],
                lambda successor: critic.predict(successor),
                accumulate=estimator_state is not None,
                state=estimator_state,
            )
            transition_successor_values = transition.predict_test_function(
                features.reshape(-1, features.shape[-1])
            ).reshape_as(segment_rewards)
            if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"):
                non_tensor_batch["polyseg_transition_target_semantics"] = np.full(
                    (segment_rewards.shape[0],), "p_v_test_function", dtype="<U32"
                )
            transition_successor_values = _clip_segment_values_to_r_max(
                transition_successor_values,
                segment_durations,
                valid,
                cfg.r_max,
            )
            transition_successor_values = transition_successor_values * transition_valid.to(transition_successor_values.dtype)
            if cfg.use_confidence_bonus:
                transition_bonus[transition_valid] = (
                    dependence_radius_multiplier
                    * confidence_bonus(
                        transition.features(features[transition_valid]),
                        gram=transition_pre_update_gram,
                        beta=cfg.transition_confidence_beta,
                    )
                ).to(dtype=transition_bonus.dtype)
                effective_segment_rewards = effective_segment_rewards + transition_bonus
            if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"):
                transition_gram = transition.gram_matrix()
                transition_eigvals = torch.linalg.eigvalsh(transition_gram)
                non_tensor_batch["polyseg_transition_confidence_bonus"] = np.full(
                    (segment_rewards.shape[0],),
                    float(transition_bonus.detach().mean().cpu()),
                    dtype=np.float32,
                )
                non_tensor_batch["polyseg_transition_feature_condition"] = np.full(
                    (segment_rewards.shape[0],),
                    float(torch.linalg.cond(transition_gram).detach().cpu()),
                    dtype=np.float32,
                )
                non_tensor_batch["polyseg_transition_feature_min_eigenvalue"] = np.full(
                    (segment_rewards.shape[0],),
                    float(transition_eigvals.min().detach().cpu()),
                    dtype=np.float32,
                )
        if cfg.use_confidence_bonus:
            confidence[valid] = (
                dependence_radius_multiplier
                * confidence_bonus(
                    critic.features(features[valid]),
                    gram=pre_update_gram,
                    beta=cfg.reward_confidence_beta,
                )
            ).to(dtype=confidence.dtype)
            effective_segment_rewards = _clip_segment_rewards_to_r_max(
                segment_rewards + confidence + transition_bonus,
                segment_durations,
                valid,
                cfg.r_max,
            )
            if cfg.polynomial_target == "return":
                targets = macro_return_targets(
                    effective_segment_rewards,
                    segment_mask=valid,
                    segment_durations=segment_durations,
                    gamma=cfg.gamma,
                )
            else:
                targets = targets + confidence
            targets = _clip_segment_values_to_r_max(
                targets,
                segment_durations,
                valid,
                cfg.r_max,
            )
            critic.fit(
                features[valid],
                targets[valid],
                sample_weight=None if critic_weights is None else critic_weights[valid],
            )
        if cfg.use_transition_estimator:
            # Fitted macro Bellman backup.  Recompute the successor under the
            # updated critic and refit the polynomial value once (or the
            # configured number of times) on r + gamma^duration P_hat V.
            # This is deliberately separate from the return-target path above
            # so the two estimators remain auditable in ablations.
            discounts = torch.pow(
                torch.as_tensor(cfg.gamma, dtype=segment_rewards.dtype, device=segment_rewards.device),
                (
                    torch.ones_like(segment_rewards)
                    if segment_durations is None
                    else _as_tensor(segment_durations, device=segment_rewards.device)
                ),
            )
            # A persistent sufficient-statistics critic cannot replay old
            # feature rows to change all historical Bellman targets.  Keep its
            # solved return fit stable and use the current transition model
            # for successor values; non-accumulating runs retain the explicit
            # iterative Bellman refit below.
            backup_iterations = 0 if estimator_state is not None else cfg.transition_backup_iterations
            for _ in range(backup_iterations):
                transition.fit_test_function(
                    features[transition_valid],
                    next_features[transition_valid],
                    lambda successor: critic.predict(successor),
                )
                transition_successor_values = transition.predict_test_function(
                    features.reshape(-1, features.shape[-1])
                ).reshape_as(segment_rewards)
                transition_successor_values = _clip_segment_values_to_r_max(
                    transition_successor_values,
                    segment_durations,
                    valid,
                    cfg.r_max,
                )
                transition_successor_values = transition_successor_values * transition_valid.to(
                    transition_successor_values.dtype
                )
                targets = effective_segment_rewards + discounts * transition_successor_values
                targets = _clip_segment_values_to_r_max(
                    targets,
                    segment_durations,
                    valid,
                    cfg.r_max,
                )
                critic.fit(
                    features[valid],
                    targets[valid],
                    sample_weight=None if critic_weights is None else critic_weights[valid],
                )
            if backup_iterations:
                transition.fit_test_function(
                    features[transition_valid],
                    next_features[transition_valid],
                    lambda successor: critic.predict(successor),
                )
            transition_successor_values = transition.predict_test_function(
                features.reshape(-1, features.shape[-1])
            ).reshape_as(segment_rewards)
            transition_successor_values = _clip_segment_values_to_r_max(
                transition_successor_values,
                segment_durations,
                valid,
                cfg.r_max,
            )
            transition_successor_values = transition_successor_values * transition_valid.to(
                transition_successor_values.dtype
            )
        segment_values = critic.predict(features.reshape(-1, features.shape[-1])).reshape(segment_rewards.shape)
        if enforce_value_range:
            segment_values = _clip_segment_values_to_r_max(
                segment_values,
                segment_durations,
                valid,
                cfg.r_max,
            )
        residual = (segment_values.detach() - targets.detach())
        critic_mse = (residual.square()[valid]).mean() if torch.any(valid) else residual.new_zeros(())
        next_values = (
            transition_successor_values
            if transition_successor_values is not None
            else torch.cat([segment_values[:, 1:], torch.zeros_like(segment_values[:, :1])], dim=1)
        )
        durations_for_residual = (
            torch.ones_like(segment_rewards)
            if segment_durations is None
            else _as_tensor(segment_durations, device=segment_rewards.device)
        )
        bellman_residual = effective_segment_rewards + torch.pow(
            torch.as_tensor(cfg.gamma, dtype=segment_rewards.dtype, device=segment_rewards.device),
            durations_for_residual,
        ) * next_values - segment_values
        bellman_mse = (
            bellman_residual.detach().square()[valid]
        ).mean() if torch.any(valid) else bellman_residual.new_zeros(())
        closure_residual = bellman_closure_residual(
            effective_segment_rewards,
            next_values,
            segment_values,
            mask=valid,
            gamma=cfg.gamma,
            durations=durations_for_residual,
        )
        if cfg.bellman_closure_tolerance > 0 and float(closure_residual.detach().cpu()) > cfg.bellman_closure_tolerance:
            raise ValueError(
                "empirical Bellman closure residual exceeds configured tolerance: "
                f"{float(closure_residual):.6g} > {cfg.bellman_closure_tolerance:.6g}"
            )
        # Keep diagnostics in the non-tensor side channel.  Putting scalar
        # fields in the training TensorDict would make padding-free actor and
        # critic workers treat them as sequence fields.
        if non_tensor_batch is not None and hasattr(non_tensor_batch, "__setitem__"):
            if cfg.critic_type == "mlp":
                condition = 0.0
                min_eigenvalue = 0.0
            else:
                gram = critic.gram_matrix()
                eigvals = torch.linalg.eigvalsh(gram)
                condition = float(torch.linalg.cond(gram).detach().cpu())
                min_eigenvalue = float(eigvals.min().detach().cpu())
            non_tensor_batch["polyseg_critic_mse"] = np.full(
                (segment_rewards.shape[0],), float(critic_mse.detach().cpu()), dtype=np.float32
            )
            non_tensor_batch["polyseg_bellman_residual_mse"] = np.full(
                (segment_rewards.shape[0],), float(bellman_mse.detach().cpu()), dtype=np.float32
            )
            non_tensor_batch["polyseg_bellman_closure_residual"] = np.full(
                (segment_rewards.shape[0],), float(closure_residual.detach().cpu()), dtype=np.float32
            )
            non_tensor_batch["polyseg_confidence_bonus"] = np.full(
                (segment_rewards.shape[0],), float(confidence.detach().mean().cpu()), dtype=np.float32
            )
            non_tensor_batch["polyseg_confidence_pre_update"] = np.full(
                (segment_rewards.shape[0],),
                float(bool(cfg.use_confidence_bonus)),
                dtype=np.float32,
            )
            non_tensor_batch["polyseg_feature_condition"] = np.full(
                (segment_rewards.shape[0],), condition, dtype=np.float32
            )
            non_tensor_batch["polyseg_feature_min_eigenvalue"] = np.full(
                (segment_rewards.shape[0],), min_eigenvalue, dtype=np.float32
            )
            non_tensor_batch["polyseg_feature_effective_samples"] = np.full(
                (segment_rewards.shape[0],),
                float(estimator_state.sample_count if estimator_state is not None else valid.sum().detach().cpu()),
                dtype=np.float32,
            )
            non_tensor_batch["polyseg_feature_batch_samples"] = np.full(
                (segment_rewards.shape[0],), float(valid.sum().detach().cpu()), dtype=np.float32
            )
            non_tensor_batch["polyseg_estimator_sample_count"] = np.full(
                (segment_rewards.shape[0],),
                float(estimator_state.sample_count if estimator_state is not None else valid.sum().detach().cpu()),
                dtype=np.float32,
            )
            non_tensor_batch["polyseg_transition_effective_samples"] = np.full(
                (segment_rewards.shape[0],),
                float(
                    estimator_state.transition_sample_count
                    if estimator_state is not None
                    else (transition.sample_count if cfg.use_transition_estimator else 0)
                ),
                dtype=np.float32,
            )
            non_tensor_batch["polyseg_feature_dimension"] = np.full(
                (segment_rewards.shape[0],), float(critic.dimension), dtype=np.float32
            )
            # Diagnostics required by the theory/experiment plan.  These are
            # computed on detached rollout features and never affect the
            # policy update: spectral tail measures high-degree coefficient
            # energy, theta stability compares q with q//2 on the same batch,
            # and held-out residual uses trajectory-level splitting so one
            # trajectory cannot leak adjacent segment targets into evaluation.
            if cfg.critic_type == "polynomial":
                design_features = critic.features(features[valid]).reshape(-1, critic.dimension)
                theta = critic.theta.detach()
                total_energy = theta.square().sum().clamp_min(1e-12)
                if cfg.basis == "additive":
                    degree_labels = [0] + [degree for _ in range(features.shape[-1]) for degree in range(1, cfg.q + 1)]
                else:
                    degree_labels = [sum(alpha) for alpha in enumerate_multi_indices(features.shape[-1], cfg.q)]
                tail_mask = torch.as_tensor(
                    [degree > cfg.q // 2 for degree in degree_labels], dtype=torch.bool, device=theta.device
                )
                spectral_tail = theta[tail_mask].square().sum() / total_energy
                theta_stability = torch.zeros((), dtype=theta.dtype, device=theta.device)
                prediction_stability = torch.zeros((), dtype=theta.dtype, device=theta.device)
                theta_stability_available = 0.0
                if cfg.q >= 1:
                    low_q = cfg.q // 2
                    low_critic = PolynomialSegmentCritic(
                        q=low_q,
                        feature_dim=features.shape[-1],
                        basis=cfg.basis,
                        feedback="sum",
                        ridge_lambda=cfg.ridge_lambda,
                        normalized_basis=cfg.normalized_basis,
                    )
                    low_critic.fit(features[valid], raw_targets[valid])
                    # Coefficient stability on the shared low-degree basis.
                    full_theta = critic.theta.detach()[: low_critic.dimension]
                    low_theta = low_critic.theta.detach()
                    theta_stability = (full_theta - low_theta).norm() / low_theta.norm().clamp_min(1e-12)
                    # Prediction stability is a different quantity and must not
                    # be conflated with coefficient stability.
                    full_pred = critic.predict(features[valid]).detach()
                    low_pred = low_critic.predict(features[valid]).detach()
                    prediction_stability = (full_pred - low_pred).norm() / full_pred.norm().clamp_min(1e-12)
                    theta_stability_available = 1.0
                # Unavailable diagnostics are NaN with an explicit availability
                # flag; a missing held-out split must not look like zero error.
                heldout_residual = torch.full((), float("nan"), dtype=theta.dtype, device=theta.device)
                heldout_bellman = torch.full((), float("nan"), dtype=theta.dtype, device=theta.device)
                heldout_available = 0.0
                if features.shape[0] >= 2:
                    heldout_rows = torch.arange(features.shape[0], device=features.device) % 2 == 1
                    fit_rows = ~heldout_rows
                    fit_valid = valid[fit_rows]
                    eval_valid = valid[heldout_rows]
                    if torch.any(fit_valid) and torch.any(eval_valid):
                        heldout_critic = PolynomialSegmentCritic(
                            q=cfg.q,
                            feature_dim=features.shape[-1],
                            basis=cfg.basis,
                            feedback="sum",
                            ridge_lambda=cfg.ridge_lambda,
                            normalized_basis=cfg.normalized_basis,
                        )
                        heldout_critic.fit(features[fit_rows][fit_valid], raw_targets[fit_rows][fit_valid])
                        eval_features = features[heldout_rows]
                        eval_targets = raw_targets[heldout_rows]
                        eval_values = heldout_critic.predict(eval_features.reshape(-1, features.shape[-1])).reshape_as(eval_targets)
                        heldout_residual = (eval_values[eval_valid] - eval_targets[eval_valid]).square().mean()
                        eval_rewards = effective_segment_rewards[heldout_rows]
                        eval_next = torch.cat([eval_values[:, 1:], torch.zeros_like(eval_values[:, :1])], dim=1)
                        eval_discounts = torch.pow(
                            torch.as_tensor(cfg.gamma, dtype=eval_rewards.dtype, device=eval_rewards.device),
                            torch.ones_like(eval_rewards)
                            if segment_durations is None
                            else _as_tensor(segment_durations[heldout_rows], device=eval_rewards.device),
                        )
                        eval_bellman = eval_rewards + eval_discounts * eval_next - eval_values
                        heldout_bellman = eval_bellman[eval_valid].square().mean()
                        heldout_available = 1.0
                for key, value in (
                    ("polyseg_empirical_spectral_tail", spectral_tail),
                    ("polyseg_theta_q_stability", theta_stability),
                    ("polyseg_prediction_q_stability", prediction_stability),
                    ("polyseg_q_stability_available", torch.as_tensor(theta_stability_available, dtype=theta.dtype, device=theta.device)),
                    ("polyseg_heldout_return_residual_mse", heldout_residual),
                    ("polyseg_heldout_bellman_residual_mse", heldout_bellman),
                    ("polyseg_heldout_diagnostics_available", torch.as_tensor(heldout_available, dtype=theta.dtype, device=theta.device)),
                ):
                    non_tensor_batch[key] = np.full(
                        (segment_rewards.shape[0],), float(value.detach().cpu()), dtype=np.float32
                    )

    elif segment_values is None:
        segment_values = pooled_values
    if transition_successor_values is not None:
        # Use the learned successor for every macro TD error.  The generic
        # helper supports only a shifted value chain, while a fitted transition
        # operator predicts each boundary independently, so recurse explicitly.
        current = segment_values if segment_values is not None else torch.zeros_like(segment_rewards)
        rewards = effective_segment_rewards
        valid = segment_mask.to(torch.bool)
        durations = (
            torch.ones_like(rewards)
            if segment_durations is None
            else _as_tensor(segment_durations, device=rewards.device)
        )
        current = current[:, : rewards.shape[1]]
        transition_advantages = torch.zeros_like(rewards)
        next_gae = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)
        for index in reversed(range(rewards.shape[1])):
            active = valid[:, index].to(rewards.dtype)
            discount = torch.pow(torch.as_tensor(cfg.gamma, dtype=rewards.dtype, device=rewards.device), durations[:, index])
            delta = rewards[:, index] + discount * transition_successor_values[:, index] - current[:, index]
            next_gae = (delta + discount * cfg.lam * next_gae) * active
            transition_advantages[:, index] = next_gae
        transition_returns = (transition_advantages + current) * valid.to(rewards.dtype)
        transition_returns = _clip_segment_values_to_r_max(
            transition_returns,
            durations,
            valid,
            cfg.r_max,
        )
        if cfg.normalize_advantages:
            transition_advantages = _masked_whiten(transition_advantages, valid)
        token_advantages = scatter_segment_values(transition_advantages, safe_ids, mask=response_mask)
        token_returns = scatter_segment_values(transition_returns, safe_ids, mask=response_mask)
        _commit_dependence_state()
        return token_advantages, token_returns
    result = compute_segment_gae_advantage(
        effective_segment_rewards,
        segment_values,
        segment_mask=segment_mask,
        segment_durations=segment_durations,
        gamma=cfg.gamma,
        lam=cfg.lam,
        normalize=cfg.normalize_advantages,
        token_segment_ids=safe_ids,
        token_mask=response_mask,
    )
    _, _, token_advantages, token_returns = result
    _commit_dependence_state()
    return token_advantages, token_returns


def _saturating_product(*terms: float) -> float:
    """Multiply nonnegative bounds without overflowing to infinity."""

    result = 1.0
    limit = sys.float_info.max
    for term in terms:
        value = float(term)
        if value < 0 or not math.isfinite(value):
            raise ValueError("bound terms must be finite and non-negative")
        if value == 0.0:
            return 0.0
        if result > limit / value:
            return limit
        result *= value
    return result


def _feedback_factor(m: int, horizon: int, r_max: float, feedback: str) -> float:
    feedback = normalize_feedback_protocol(feedback)
    if feedback in {"local_sum", "mean", "terminal", "utility_delta", "prm_potential"}:
        return 1.0
    if feedback != "binary_local":
        raise ValueError("unsupported PolySeg feedback protocol")
    if m < 1:
        raise ValueError("m must be positive")
    if horizon < 1 or r_max < 0 or not math.isfinite(float(r_max)):
        raise ValueError("horizon must be positive and r_max must be finite and non-negative")
    # Work in log space first.  Direct exp(log-factor) overflowed for large
    # horizons even though the selector only needs a conservative finite
    # multiplier.  Saturating at the largest representable float preserves the
    # ordering without producing an exception or NaN downstream.
    log_factor = float(horizon) * float(r_max) / (2.0 * float(m))
    return math.exp(min(log_factor, math.log(sys.float_info.max)))


def _statistical_round_counts(
    *,
    episodes: int,
    m: int,
    q: int,
    dependence_chi: float,
    selector_norm: str,
    kappa_segment_upper_by_m: Mapping[int, float] | None,
    kappa_segment_upper_by_qm: Mapping[tuple[int, int], float] | None,
    kappa_trajectory_upper: float,
    kappa_fallback: str = "auto",
) -> tuple[float, float]:
    """Return ``(information_samples, statistical_rounds)`` for one candidate.

    ``information_samples`` controls the identifiability constraint
    ``D_q <= information_samples``; it is always the conservative
    dependence-discounted count for the dependence-aware norms.
    ``statistical_rounds`` is the term that enters the statistical regret:
    effective-round normalization counts effective macro rounds, while the
    segment-update normalization counts actual segment decisions inflated by
    dependence.  Keeping both in one helper prevents the selector and the
    reported bound from drifting apart.
    """

    if selector_norm == "power":
        information = episodes * (m**dependence_chi)
        return information, information
    qm_grid = {
        (int(key[0]), int(key[1])): float(value)
        for key, value in (kappa_segment_upper_by_qm or {}).items()
        if isinstance(key, tuple) and len(key) == 2 and math.isfinite(float(value))
    }
    kappa_grid = {
        int(key): float(value)
        for key, value in (kappa_segment_upper_by_m or {}).items()
        if int(key) >= 1 and math.isfinite(float(value))
    }
    # ``kappa`` is q-dependent.  ``auto`` (default) uses a supplied m-only
    # estimate only while no q-specific cell exists for that m; once any order
    # has been observed at m, an unseen order falls back to the conservative
    # nominal bound m instead of importing a different order's estimate.
    # ``nominal`` and ``per_m`` are explicit overrides for audits.
    q_specific_m = {mm for _, mm in qm_grid}
    if kappa_fallback == "per_m":
        fallback = kappa_grid.get(m, float(m))
    elif kappa_fallback == "nominal":
        fallback = float(m)
    elif kappa_fallback == "auto":
        fallback = float(m) if m in q_specific_m else kappa_grid.get(m, float(m))
    else:
        raise ValueError("kappa_fallback must be 'auto', 'nominal' or 'per_m'")
    kappa_seg = max(qm_grid.get((q, m), fallback), 1.0)
    kappa_seg = min(kappa_seg, float(m))
    information = episodes * m / (kappa_seg * kappa_trajectory_upper)
    if selector_norm == "effective_round":
        return information, information
    return information, episodes * m * kappa_seg * kappa_trajectory_upper



def _protocol_query_count(feedback: str, m: int) -> int:
    """Number of verifier queries one trajectory needs for a protocol.

    Terminal feedback observes only the final segment, so its actual query
    count is one; every other protocol queries each segment.  Using ``m`` for
    terminal overstates cost and rejects feasible high-m candidates.
    """
    canonical = normalize_feedback_protocol(feedback)
    if canonical == "terminal":
        return 1
    return int(m)


def _objective_terms(
    *,
    q: int,
    m: int,
    episodes: int,
    horizon: int,
    feature_dim: int,
    feedback: str,
    dependence_chi: float,
    stat_exponent_u: float,
    approx_exponent_v: float,
    segment_exponent_gamma: float,
    stat_constant: float,
    approx_constant: float,
    segment_constant: float,
    selector_norm: str,
    kappa_segment_upper_by_m: Mapping[int, float] | None,
    kappa_segment_upper_by_qm: Mapping[tuple[int, int], float] | None,
    kappa_trajectory_upper: float,
    r_max: float,
    segment_cost: float,
    critic_compute_cost: float = 0.0,
    cost_penalty: float = 0.0,
    approx_dimension_exponent: float | None = None,
) -> dict[str, float]:
    """Single decomposition shared by the selector and the reported bound."""
    dimension = q + 1 if feature_dim == 1 else math.comb(feature_dim + q, feature_dim)
    information, statistical_rounds = _statistical_round_counts(
        episodes=episodes,
        m=m,
        q=q,
        dependence_chi=dependence_chi,
        selector_norm=selector_norm,
        kappa_segment_upper_by_m=kappa_segment_upper_by_m,
        kappa_segment_upper_by_qm=kappa_segment_upper_by_qm,
        kappa_trajectory_upper=kappa_trajectory_upper,
    )
    factor = _feedback_factor(m, horizon, r_max, feedback)
    statistical = _saturating_product(
        stat_constant, factor, dimension**stat_exponent_u, math.sqrt(max(statistical_rounds, 1e-12))
    )
    exponent = (
        max(0.0, stat_exponent_u - 0.5)
        if approx_dimension_exponent is None
        else float(approx_dimension_exponent)
    )
    approximation = approx_constant * episodes * (dimension**exponent) * (q + 1) ** (-approx_exponent_v)
    segment_term = 0.0 if normalize_feedback_protocol(feedback) == "local_sum" else (
        segment_constant * episodes * (horizon / m) ** segment_exponent_gamma
    )
    query_count = _protocol_query_count(feedback, m)
    query_cost = episodes * query_count * segment_cost
    compute_cost = episodes * m * dimension * float(critic_compute_cost)
    objective = statistical + approximation + segment_term + cost_penalty * (query_cost + compute_cost)
    return {
        "dimension": float(dimension),
        "information_samples": float(information),
        "statistical_rounds": float(statistical_rounds),
        "feedback_factor": float(factor),
        "statistical": float(statistical),
        "approximation": float(approximation),
        "segment": float(segment_term),
        "query_count": float(query_count),
        "query_cost": float(query_cost),
        "compute_cost": float(compute_cost),
        "objective": float(objective),
    }

def choose_polynomial_segment_config(
    *,
    q_candidates: Iterable[int],
    m_candidates: Iterable[int],
    episodes: int,
    horizon: int,
    budget: float | None,
    feature_dim: int,
    feedback: str = "sum",
    segment_cost: float = 1.0,
    critic_compute_cost: float = 0.0,
    compute_budget: float | None = None,
    dependence_chi: float = 1.0,
    stat_exponent_u: float = 1.0,
    approx_exponent_v: float = 1.0,
    segment_exponent_gamma: float = 1.0,
    stat_constant: float = 1.0,
    approx_constant: float = 1.0,
    segment_constant: float = 1.0,
    r_max: float = 1.0,
    cost_penalty: float = 0.0,
    selector_norm: str = "power",
    kappa_segment_upper_by_m: Mapping[int, float] | None = None,
    kappa_segment_upper_by_qm: Mapping[tuple[int, int], float] | None = None,
    kappa_trajectory_upper: float = 1.0,
) -> PolySegChoice:
    """Finite constrained search for the theory's joint ``(q, m)`` choice.

    ``selector_norm='power'`` preserves the historical ``K m**chi``
    parameterization used by the synthetic utilities.  The adaptive rollout
    controller uses ``selector_norm='effective_round'``: its statistical
    sample count is the conservative lower estimate

    ``K*m / (kappa_segment_upper(m) * kappa_trajectory_upper)``.

    Keeping the convention explicit prevents an empirical effective-round
    diagnostic from being silently mixed with the segment-update regret
    normalization used by the main implementation.
    """

    if episodes < 1 or horizon < 1 or feature_dim < 1:
        raise ValueError("episodes, horizon and feature_dim must be positive")
    if segment_cost <= 0:
        raise ValueError("segment_cost must be positive")
    if not math.isfinite(float(critic_compute_cost)) or critic_compute_cost < 0:
        raise ValueError("critic_compute_cost must be finite and non-negative")
    if compute_budget is not None and (
        not math.isfinite(float(compute_budget)) or compute_budget < 0
    ):
        raise ValueError("compute_budget must be finite and non-negative")
    if approx_exponent_v <= 0 or stat_exponent_u < 0 or segment_exponent_gamma < 0:
        raise ValueError("invalid approximation/statistical exponents")
    selector_norm = str(selector_norm).strip().lower()
    if selector_norm not in {"power", "effective_round", "segment_update"}:
        raise ValueError("selector_norm must be 'power', 'effective_round', or 'segment_update'")
    if kappa_trajectory_upper < 1 or not math.isfinite(float(kappa_trajectory_upper)):
        raise ValueError("kappa_trajectory_upper must be finite and at least one")
    kappa_grid = {
        int(key): float(value)
        for key, value in (kappa_segment_upper_by_m or {}).items()
        if int(key) >= 1 and math.isfinite(float(value))
    }
    kappa_qm_grid = {
        (int(key[0]), int(key[1])): float(value)
        for key, value in (kappa_segment_upper_by_qm or {}).items()
        if len(key) == 2
        and int(key[0]) >= 0
        and int(key[1]) >= 1
        and math.isfinite(float(value))
    }
    candidates: list[PolySegChoice] = []
    for m in sorted(set(int(x) for x in m_candidates)):
        if m < 1 or m > horizon:
            continue
        # Budget feasibility must use the protocol's actual query count:
        # terminal feedback queries one segment, not m.
        if episodes * _protocol_query_count(feedback, m) * segment_cost > (
            float("inf") if budget is None else budget
        ) + 1e-8:
            continue
        for q in sorted(set(int(x) for x in q_candidates)):
            if q < 0:
                continue
            terms = _objective_terms(
                q=q, m=m, episodes=episodes, horizon=horizon, feature_dim=feature_dim,
                feedback=feedback, dependence_chi=dependence_chi,
                stat_exponent_u=stat_exponent_u, approx_exponent_v=approx_exponent_v,
                segment_exponent_gamma=segment_exponent_gamma,
                stat_constant=stat_constant, approx_constant=approx_constant,
                segment_constant=segment_constant, selector_norm=selector_norm,
                kappa_segment_upper_by_m=kappa_grid, kappa_segment_upper_by_qm=kappa_qm_grid,
                kappa_trajectory_upper=kappa_trajectory_upper, r_max=r_max,
                segment_cost=segment_cost, critic_compute_cost=critic_compute_cost,
                cost_penalty=cost_penalty,
            )
            # Identifiability and declared compute limits are constraints, not
            # objective terms, so they are checked before ranking candidates.
            if terms["dimension"] > max(terms["information_samples"], 1e-12) + 1e-8:
                continue
            if compute_budget is not None and terms["compute_cost"] > float(compute_budget) + 1e-8:
                continue
            candidates.append(
                PolySegChoice(
                    q, m, terms["objective"], terms["statistical"], terms["approximation"],
                    terms["segment"], terms["feedback_factor"], terms["query_cost"],
                    int(terms["dimension"]), terms["compute_cost"],
                )
            )
    if not candidates:
        raise PolySegSelectionInfeasible("no feasible (q, m) candidate under the supplied budget and sample constraints")
    return min(candidates, key=lambda candidate: (candidate.objective, candidate.m, candidate.q))


def segment_feedback_bound(
    *,
    q: int,
    m: int,
    episodes: int,
    horizon: int,
    feature_dim: int,
    feedback: str = "sum",
    dependence_chi: float = 1.0,
    r_max: float = 1.0,
    smoothness_s: float = 2.0,
    stat_exponent_u: float = 1.0,
    approx_constant: float = 1.0,
    stat_constant: float = 1.0,
    segment_constant: float = 1.0,
    segment_exponent_gamma: float = 1.0,
    selector_norm: str = "power",
    kappa_segment_upper_by_m: Mapping[int, float] | None = None,
    kappa_segment_upper_by_qm: Mapping[tuple[int, int], float] | None = None,
    kappa_trajectory_upper: float = 1.0,
    segment_cost: float = 1.0,
    critic_compute_cost: float = 0.0,
    cost_penalty: float = 0.0,
    approx_exponent_v: float | None = None,
    approx_dimension_exponent: float | None = None,
) -> dict[str, float]:
    """Return the same objective decomposition the selector optimizes.

    ``approx_exponent_v`` defaults to ``smoothness_s`` so existing callers
    keep their meaning; passing both lets a selector and its reported bound use
    one identical approximation term.  ``query_count``/``query_cost`` are
    protocol-aware (terminal feedback queries one segment).
    """

    if q < 0 or m < 1 or episodes < 1 or horizon < 1 or feature_dim < 1:
        raise ValueError("q, m, episodes, horizon and feature_dim must be valid")
    if approx_exponent_v is None:
        approx_exponent_v = float(smoothness_s)
    terms = _objective_terms(
        q=q, m=m, episodes=episodes, horizon=horizon, feature_dim=feature_dim,
        feedback=feedback, dependence_chi=dependence_chi,
        stat_exponent_u=stat_exponent_u, approx_exponent_v=float(approx_exponent_v),
        segment_exponent_gamma=segment_exponent_gamma,
        stat_constant=stat_constant, approx_constant=approx_constant,
        segment_constant=segment_constant, selector_norm=selector_norm,
        kappa_segment_upper_by_m=kappa_segment_upper_by_m,
        kappa_segment_upper_by_qm=kappa_segment_upper_by_qm,
        kappa_trajectory_upper=kappa_trajectory_upper, r_max=r_max,
        segment_cost=segment_cost, critic_compute_cost=critic_compute_cost,
        cost_penalty=cost_penalty,
        approx_dimension_exponent=approx_dimension_exponent,
    )
    return {
        "statistical": terms["statistical"],
        "approximation": terms["approximation"],
        "segment": terms["segment"],
        "feedback_factor": terms["feedback_factor"],
        "statistical_rounds": terms["statistical_rounds"],
        "selector_norm": selector_norm,
        "query_count": terms["query_count"],
        "query_cost": terms["query_cost"],
        "compute_cost": terms["compute_cost"],
        "objective": terms["objective"],
        "total": terms["objective"],
        "feature_dim": terms["dimension"],
    }


class PolynomialSegmentCritic:
    """Ridge or logistic regression over frozen Legendre segment features."""

    def __init__(
        self,
        q: int,
        feature_dim: int,
        *,
        basis: str = "total",
        feedback: str = "sum",
        ridge_lambda: float = 1.0,
        logistic_l2: float = 1.0,
        normalized_basis: bool = False,
        max_logistic_iter: int = 50,
    ):
        if ridge_lambda <= 0 or logistic_l2 <= 0:
            raise ValueError("regularization must be positive")
        if feedback not in {"sum", "binary"}:
            raise ValueError("feedback must be 'sum' or 'binary'")
        self.q = q
        self.feature_dim = feature_dim
        self.basis = basis
        self.feedback = feedback
        self.ridge_lambda = ridge_lambda
        self.logistic_l2 = logistic_l2
        self.normalized_basis = normalized_basis
        self.max_logistic_iter = max_logistic_iter
        self.theta: torch.Tensor | None = None
        self.design: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self._sufficient_state: PolynomialEstimatorState | None = None

    @property
    def dimension(self) -> int:
        return 1 + self.feature_dim * self.q if self.basis == "additive" else math.comb(self.feature_dim + self.q, self.feature_dim)

    def features(self, z: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        return legendre_features(z, self.q, basis=self.basis, normalized=self.normalized_basis)

    def fit(
        self,
        z: torch.Tensor,
        targets: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        accumulate: bool = False,
        state: PolynomialEstimatorState | None = None,
    ) -> "PolynomialSegmentCritic":
        if accumulate:
            if self.feedback != "sum":
                raise ValueError("accumulating estimator state currently supports sum feedback only")
            if state is None:
                raise ValueError("state is required when accumulate=True")
            x = self.features(z).reshape(-1, self.dimension)
            y = _as_tensor(targets, device=x.device).reshape(-1)
            if x.shape[0] != y.shape[0]:
                raise ValueError("z and targets have incompatible sample counts")
            state.update(x, y, sample_weight=sample_weight)
            if state.ridge_lambda != self.ridge_lambda:
                raise ValueError("critic state regularization does not match estimator")
            self.theta = state.solve()
            self._sufficient_state = state
            self.design = None
            self.targets = None
            return self
        self._sufficient_state = None
        x = self.features(z).reshape(-1, self.dimension)
        y = _as_tensor(targets, device=x.device).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("z and targets have incompatible sample counts")
        weight = None
        if sample_weight is not None:
            weight = _as_tensor(sample_weight, device=x.device).reshape(-1).clamp_min(0)
            if weight.shape != y.shape:
                raise ValueError("sample_weight shape mismatch")
        if self.feedback == "sum":
            if weight is not None:
                # Weighted least squares is exactly sqrt-weight scaling.
                x_fit = x * weight.sqrt().unsqueeze(-1)
                y_fit = y * weight.sqrt()
            else:
                x_fit, y_fit = x, y
            gram = x_fit.T @ x_fit + self.ridge_lambda * torch.eye(self.dimension, dtype=x.dtype, device=x.device)
            self.theta = torch.linalg.solve(gram, x_fit.T @ y_fit)
        else:
            if weight is not None:
                if torch.any((y < 0) | (y > 1)):
                    raise ValueError("binary logistic targets must lie in [0, 1]")
                weight = weight / weight.mean().clamp_min(torch.finfo(weight.dtype).eps)
            theta = torch.zeros(self.dimension, dtype=x.dtype, device=x.device)
            # Newton updates are stable for the small feature spaces used by
            # PolySeg; damping keeps near-separable binary feedback finite.
            # Sample weights multiply each observation's gradient and Hessian
            # contribution; they must NOT rescale the binary label.
            for _ in range(self.max_logistic_iter):
                logits = x @ theta
                probs = torch.sigmoid(logits)
                curvature = probs * (1.0 - probs)
                if weight is not None:
                    curvature = curvature * weight
                    residual = (probs - y) * weight
                else:
                    residual = probs - y
                hessian = x.T @ (x * curvature.unsqueeze(-1))
                hessian = hessian + self.logistic_l2 * torch.eye(self.dimension, dtype=x.dtype, device=x.device)
                gradient = x.T @ residual + self.logistic_l2 * theta
                step = torch.linalg.solve(hessian, gradient)
                theta = theta - step
                if torch.max(torch.abs(step)).item() < 1e-6:
                    break
            self.theta = theta
        self.design = x.detach()
        self.targets = y.detach()
        return self

    def predict(self, z: torch.Tensor | np.ndarray | Sequence[float], *, latent: bool = False) -> torch.Tensor:
        if self.theta is None:
            raise RuntimeError("critic must be fit before predict")
        logits = self.features(z) @ self.theta
        if self.feedback == "binary" and not latent:
            return torch.sigmoid(logits)
        return logits

    def gram_matrix(self) -> torch.Tensor:
        """Return the ridge/logistic design matrix used by the fitted critic."""
        if self._sufficient_state is not None:
            gram = self._sufficient_state.gram
            if gram is None:
                raise RuntimeError("critic must be fit before requesting its Gram matrix")
            return gram + self.ridge_lambda * torch.eye(
                self.dimension, dtype=gram.dtype, device=gram.device
            )
        if self.design is None:
            raise RuntimeError("critic must be fit before requesting its Gram matrix")
        if self.feedback == "binary":
            # Logistic uncertainty is governed by the observed Fisher/Hessian,
            # not the unweighted X^T X design.  Saturated probabilities
            # therefore correctly reduce information and enlarge the bonus.
            logits = self.design @ self.theta
            probabilities = torch.sigmoid(logits)
            weights = probabilities * (1.0 - probabilities)
            gram = self.design.T @ (self.design * weights.unsqueeze(-1))
            regularizer = self.logistic_l2
        else:
            gram = self.design.T @ self.design
            regularizer = self.ridge_lambda
        return gram + regularizer * torch.eye(
            self.dimension, dtype=self.design.dtype, device=self.design.device
        )

    def confidence_bonus(self, z: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Return ``beta ||phi(z)||_{V^{-1}}`` for fitted critic features."""
        return confidence_bonus(self.features(z), gram=self.gram_matrix(), beta=beta)

    def state_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "feature_dim": self.feature_dim,
            "basis": self.basis,
            "feedback": self.feedback,
            "ridge_lambda": self.ridge_lambda,
            "logistic_l2": self.logistic_l2,
            "normalized_basis": self.normalized_basis,
            "theta": None if self.theta is None else self.theta.detach().cpu(),
            "sufficient_state": None if self._sufficient_state is None else self._sufficient_state.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "PolynomialSegmentCritic":
        for name in ("q", "feature_dim", "basis", "feedback"):
            if name in state and getattr(self, name) != state[name]:
                raise ValueError(f"critic state {name} does not match configured critic")
        for name in ("ridge_lambda", "logistic_l2"):
            if name in state:
                value = float(state[name])
                expected = getattr(self, name)
                if not math.isfinite(value) or value <= 0 or not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"critic state {name} does not match configured critic")
        if "normalized_basis" in state and bool(state["normalized_basis"]) != self.normalized_basis:
            raise ValueError("critic normalized-basis setting does not match configured critic")
        self.theta = None if state.get("theta") is None else _as_tensor(state["theta"])
        if self.theta is not None and (self.theta.shape != (self.dimension,) or not torch.isfinite(self.theta).all()):
            raise ValueError("invalid critic parameter vector")
        sufficient_state = state.get("sufficient_state")
        self._sufficient_state = (
            None if sufficient_state is None else PolynomialEstimatorState.from_state_dict(sufficient_state)
        )
        if self._sufficient_state is not None and (
            self._sufficient_state.q != self.q
            or self._sufficient_state.feature_dim != self.feature_dim
            or self._sufficient_state.basis != self.basis
            or self._sufficient_state.normalized_basis != self.normalized_basis
            or not math.isclose(self._sufficient_state.ridge_lambda, self.ridge_lambda, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("critic sufficient state configuration does not match")
        expected_feedback = "local_sum" if self.feedback == "sum" else "binary_local"
        if self._sufficient_state is not None and self._sufficient_state.feedback != expected_feedback:
            raise ValueError("critic sufficient state feedback does not match configured critic")
        self.design = None
        self.targets = None
        return self


class MLPSegmentCritic:
    """Small nonlinear regression baseline over frozen segment features.

    The MLP is deliberately a separate baseline from :class:`PolynomialSegmentCritic`:
    it receives the same rollout-observable features and the same macro return
    targets, but has no Legendre basis or closed-form ridge solve.  Fitting is
    performed on the detached batch only, so this critic is a diagnostic
    estimator and never backpropagates through the policy model.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 32,
        learning_rate: float = 0.05,
        steps: int = 100,
    ):
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if learning_rate <= 0 or steps < 1:
            raise ValueError("learning_rate must be positive and steps must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.steps = int(steps)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        self.design: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None

    @property
    def dimension(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def fit(
        self,
        z: torch.Tensor,
        targets: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        accumulate: bool = False,
        state: PolynomialEstimatorState | None = None,
    ) -> "MLPSegmentCritic":
        """Fit on one macro batch using the common critic interface.

        The polynomial critic accepts ``accumulate``/``state`` for persistent
        sufficient statistics.  MLP is intentionally a batch-only baseline,
        but it must still accept the public keyword contract so selecting
        ``critic_type=mlp`` cannot fail before the first PPO update.
        """

        if accumulate or state is not None:
            raise ValueError("MLP critic does not support accumulate_estimator")
        x = _as_tensor(z).reshape(-1, self.feature_dim)
        y = _as_tensor(targets, device=x.device).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("z and targets have incompatible sample counts")
        self.model = self.model.to(device=x.device, dtype=x.dtype)
        x = x.detach()
        y = y.detach()
        weight = None
        if sample_weight is not None:
            weight = _as_tensor(sample_weight, device=x.device).reshape(-1).clamp_min(0)
            if weight.shape != y.shape:
                raise ValueError("sample_weight shape mismatch")
            weight = weight / weight[weight > 0].mean().clamp_min(torch.finfo(weight.dtype).eps) if torch.any(weight > 0) else weight
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        for _ in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            residual = self.model(x).squeeze(-1) - y
            loss = (
                (residual.square() * weight).sum() / weight.sum().clamp_min(torch.finfo(weight.dtype).eps)
                if weight is not None
                else residual.square().mean()
            )
            loss.backward()
            optimizer.step()
        self.model.eval()
        self.design = x
        self.targets = y
        return self

    def predict(self, z: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        if self.design is None:
            raise RuntimeError("critic must be fit before predict")
        x = _as_tensor(z, device=self.design.device).reshape(-1, self.feature_dim)
        with torch.no_grad():
            return self.model(x).squeeze(-1)


# Keep registration at the end of this module.  ``core_algos`` imports this
# module only on a PolySeg lookup, so the normal verl import path stays free of
# optional PolySeg work while the configured estimator is always discoverable.
from verl.trainer.ppo.core_algos import AdvantageEstimator, register_adv_est  # noqa: E402


@register_adv_est(AdvantageEstimator.POLYSEG)
def compute_polyseg_advantage_registered(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Any = None,
    batch: Any = None,
    non_tensor_batch: Any = None,
    values: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Registered adapter used by verl's legacy and v1 PPO trainers."""

    if values is None:
        values = _extract_polyseg_field(batch, "values")
    return compute_polyseg_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        values=values,
        config=config,
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        segment_features=_extract_polyseg_field(batch, "segment_features")
        if _extract_polyseg_field(batch, "segment_features") is not None
        else _extract_polyseg_field(non_tensor_batch, "segment_features"),
        step_features=_extract_polyseg_field(batch, "step_features")
        if _extract_polyseg_field(batch, "step_features") is not None
        else _extract_polyseg_field(non_tensor_batch, "step_features"),
    )


@register_adv_est(AdvantageEstimator.POLYSEG_GAE)
def compute_polyseg_gae_advantage_registered(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Any = None,
    batch: Any = None,
    non_tensor_batch: Any = None,
    values: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias with an explicit name for configs that emphasize macro GAE."""

    return compute_polyseg_advantage_registered(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        config=config,
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        values=values,
        **kwargs,
    )
