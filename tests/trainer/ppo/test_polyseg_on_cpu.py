import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo import core_algos
from verl.experimental.agent_loop.single_turn_agent_loop import build_single_turn_segment_metadata
from verl.trainer.ppo.polyseg import (
    FileBackedSegmentRewardBudget,
    MLPSegmentCritic,
    PolynomialSegmentCritic,
    PolynomialTransitionEstimator,
    BudgetedPolySegController,
    macro_bellman_backup,
    macro_return_targets,
    SegmentRewardBudget,
    choose_polynomial_segment_config,
    compute_polyseg_advantage,
    compute_segment_policy_loss,
    compute_segment_gae_advantage,
    enumerate_multi_indices,
    legendre_features,
    make_segment_slices,
    segment_ids,
    suffix_potential_differences,
    segment_log_prob_ratio,
    segment_ppo_loss,
    SuffixFeedbackCollector,
    build_segment_features,
    feature_map_contract,
    confidence_bonus,
    transition_confidence_bonus,
    PolynomialEstimatorState,
    PolySegDependenceState,
    PolySegConfig,
    load_polyseg_estimator_state_dict,
    polyseg_estimator_state_dict,
    reset_polyseg_estimator_state,
    reset_polyseg_dependence_state,
    polyseg_dependence_state_dict,
    load_polyseg_dependence_state_dict,
    compute_segment_rewards,
    extract_segment_metadata,
    normalize_feedback_protocol,
    feedback_protocol_contract,
    cross_fitted_segment_residuals,
    cross_fitted_logistic_segment_residuals,
    estimate_dependence_diagnostics,
    segment_feedback_bound,
)


def test_total_degree_legendre_dimension_and_recurrence():
    assert len(enumerate_multi_indices(3, 4)) == math.comb(3 + 4, 3)
    x = torch.tensor([[-0.4, 0.2]])
    features = legendre_features(x, 3)
    # P_2(x) = (3x^2 - 1) / 2 appears in the one-dimensional terms.
    assert features.shape == (1, 10)
    assert torch.allclose(features[0, 3], (3 * x[0, 1] ** 2 - 1) / 2)
    assert legendre_features(x, 2, basis="additive").shape[-1] == 1 + 2 * 2


def test_segment_partition_and_scatter():
    assert make_segment_slices(10, 3) == [(0, 4), (4, 7), (7, 10)]
    ids = segment_ids(10, 3)
    assert ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_rollout_boundaries_are_checked_against_token_segment_ids():
    item = SimpleNamespace(
        non_tensor_batch={
            "extra_fields": {
                "segment_ids": [0, 0, 1, 1],
                "segment_boundaries": [(0, 3), (2, 4)],
            }
        },
    )
    with pytest.raises(ValueError, match="non-overlapping"):
        extract_segment_metadata(
            item,
            response_length=4,
            valid_response_mask=torch.ones(4, dtype=torch.bool),
            require_rollout_segments=True,
        )

    item.non_tensor_batch["extra_fields"]["segment_ids"] = [0, 0, 1, 1, 1]
    item.non_tensor_batch["extra_fields"]["segment_boundaries"] = [(0, 2), (2, 5)]
    with pytest.raises(ValueError, match="token-aligned"):
        extract_segment_metadata(
            item,
            response_length=4,
            valid_response_mask=torch.ones(4, dtype=torch.bool),
            require_rollout_segments=True,
        )

    item.non_tensor_batch["extra_fields"]["segment_boundaries"] = [(0, 2), (2, 4)]
    item.non_tensor_batch["extra_fields"]["segment_ids"] = [-1, 0, 1, 1]
    with pytest.raises(ValueError, match="non-negative"):
        extract_segment_metadata(
            item,
            response_length=4,
            valid_response_mask=torch.ones(4, dtype=torch.bool),
            require_rollout_segments=True,
        )

    item.non_tensor_batch["extra_fields"]["segment_ids"] = [0, 0, 1, 1]
    item.non_tensor_batch["extra_fields"]["segment_boundaries"] = [(0, 2), (2, 5)]
    with pytest.raises(ValueError, match="response bounds"):
        extract_segment_metadata(
            item,
            response_length=4,
            valid_response_mask=torch.ones(4, dtype=torch.bool),
            require_rollout_segments=True,
        )

    item.non_tensor_batch["extra_fields"]["segment_boundaries"] = [(0, 1), (1, 4)]
    with pytest.raises(ValueError, match="does not cover"):
        extract_segment_metadata(
            item,
            response_length=4,
            valid_response_mask=torch.ones(4, dtype=torch.bool),
            require_rollout_segments=True,
        )


def test_prm_potential_protocol_is_explicit_and_telescopes():
    assert normalize_feedback_protocol("prm") == "prm_potential"
    rewards = compute_segment_rewards([0.2, 0.6, 0.9], "prm_potential")
    assert rewards == pytest.approx([0.2, 0.4, 0.3])
    assert sum(rewards) == pytest.approx(0.9)


def test_file_backed_budget_is_shared_and_idempotent(tmp_path):
    path = tmp_path / "feedback.ledger.json"
    first = FileBackedSegmentRewardBudget(path, 2.0)
    second = FileBackedSegmentRewardBudget(path, 2.0)
    assert first.ledger_id == second.ledger_id

    token = first.reserve(1.0, reservation_id="trajectory-0:segment:0")
    assert second.remaining == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        second.reserve(2.0, reservation_id="trajectory-1:segment:0")

    # A retry/duplicate callback sees the same active hold rather than
    # charging the global budget a second time.
    assert second.reserve(1.0, reservation_id=token) == token
    assert second.commit(token, cost=0.5) == pytest.approx(1.5)
    assert first.consumed == pytest.approx(0.5)
    assert first.reserved == pytest.approx(0.0)
    assert second.state_dict()["ledger_id"] == first.ledger_id


def test_binary_feedback_is_a_thresholded_macro_protocol():
    assert normalize_feedback_protocol("binary") == "binary_local"
    assert compute_segment_rewards([0.2, 0.8], "binary_local") == [0.0, 1.0]


def test_feedback_protocol_contract_separates_binary_and_prm_semantics():
    binary = feedback_protocol_contract("binary")
    prm = feedback_protocol_contract("prm")
    synthetic = feedback_protocol_contract("terminal")
    assert binary["protocol"] == "binary_local"
    assert binary["label_semantics"] == "deterministic_prefix_verifier"
    assert binary["label_semantics"] != "stochastic_bernoulli_observation"
    assert prm["protocol"] == "prm_potential"
    assert prm["label_semantics"] == "prm_model_derived_pseudo_label"
    assert prm["reward_aggregation"] == "adjacent_potential_difference"
    assert prm["reward_sign"] == "signed"
    assert prm["normalization"] == "potential_difference_preserves_telescoping"
    assert synthetic["label_semantics"] == "deterministic_terminal_verifier"
    assert synthetic["reward_aggregation"] == "terminal_sparse_anchor"
    assert synthetic["normalization"] == "no_segment_count_rescaling"


def test_frozen_segment_feature_map_is_bounded_and_stable():
    features = build_segment_features(
        [(0, 2), (2, 5)],
        [1.0, 2.0],
        response_length=5,
        action_lengths=[2, 2],
        tool_call_counts=[0, 1],
        feature_dim=5,
    )
    assert features.shape == (2, 5)
    assert torch.all(features <= 1) and torch.all(features >= -1)
    assert torch.equal(features, build_segment_features(
        [(0, 2), (2, 5)], [1.0, 2.0], response_length=5,
        action_lengths=[2, 2], tool_call_counts=[0, 1], feature_dim=5,
    ))
    same_first = build_segment_features(
        [(0, 2), (2, 5)], [1.0, 2.0], response_length=5,
        action_lengths=[2, 8], tool_call_counts=[0, 12], feature_dim=5,
    )[0]
    assert torch.allclose(features[0], same_first)


def test_feature_map_contract_is_content_addressed_and_scale_stable():
    contract = feature_map_contract(response_length=128, feature_dim=5)
    assert contract["version"] == "polyseg-fixed-boundary-summary-v1"
    assert contract["semantic_role"] == "rollout_observable_boundary_state_action_summary"
    assert contract["causal_availability"] == "available_at_segment_boundary"
    assert contract["markov_sufficiency"] == "not_established"
    assert len(contract["hash"]) == 64
    assert contract == feature_map_contract(response_length=128, feature_dim=5)
    assert contract["hash"] != feature_map_contract(response_length=256, feature_dim=5)["hash"]
    assert contract["spec"]["coordinate_order"][-1] == "tool_call_fraction"


def test_dependence_diagnostics_use_cross_fitted_scores_and_cluster_bootstrap():
    torch.manual_seed(0)
    features = torch.tensor(
        [
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    )
    targets = torch.tensor([[0.1, 0.4, 0.9], [0.2, 0.5, 1.0], [0.0, 0.3, 0.8], [0.3, 0.6, 1.1]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    residuals = cross_fitted_segment_residuals(features, targets, valid_mask=valid, ridge_lambda=1e-3)
    assert residuals.shape == (4, 3)
    # The held-out predictions differ from a fit on all observations; this
    # guards against accidentally reporting in-sample critic residuals.
    assert not np.allclose(residuals, targets.numpy())
    design_scores = features.numpy() * residuals[..., None]
    diagnostics = estimate_dependence_diagnostics(
        design_scores,
        valid_mask=valid,
        weights=torch.tensor([[1.0, 2.0, 1.0]] * 4),
        hac_lag=2,
        bootstrap_samples=32,
        seed=17,
    )
    for key in (
        "kappa_segment",
        "kappa_segment_upper",
        "kappa_trajectory",
        "kappa_trajectory_upper",
        "effective_samples",
        "weighted_effective_samples",
        "chi_hat",
        "iid_cluster_radius_ratio_upper",
    ):
        assert np.isfinite(diagnostics[key])
    assert diagnostics["kappa_segment"] >= 1.0
    assert diagnostics["kappa_segment_upper"] >= diagnostics["kappa_segment"]
    assert diagnostics["kappa_trajectory_upper"] >= diagnostics["kappa_trajectory"]
    assert diagnostics["bootstrap_samples"] == 32
    assert diagnostics["segment_count"] == 12
    assert diagnostics["nominal_observations"] == 12
    assert diagnostics["cluster_count"] == 4
    assert diagnostics["total_observation_weight"] == pytest.approx(16.0)
    assert diagnostics["weighted_ess_ratio"] == pytest.approx(
        diagnostics["weighted_effective_samples"] / diagnostics["nominal_observations"]
    )


def test_weighted_ess_is_invariant_to_global_weight_scale():
    scores = np.asarray(
        [
            [[-1.0], [0.0], [1.0]],
            [[-0.5], [0.5], [1.5]],
            [[0.2], [0.8], [1.2]],
            [[-1.2], [0.1], [0.9]],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([[1.0, 2.0, 1.0]] * 4)
    first = estimate_dependence_diagnostics(scores, weights=weights, bootstrap_samples=0, hac_lag=1)
    scaled = estimate_dependence_diagnostics(scores, weights=weights * 7.0, bootstrap_samples=0, hac_lag=1)
    assert first["nominal_observations"] == scaled["nominal_observations"] == 12
    assert first["cluster_count"] == scaled["cluster_count"] == 4
    assert scaled["total_observation_weight"] == pytest.approx(7.0 * first["total_observation_weight"])
    assert scaled["weighted_effective_samples"] == pytest.approx(first["weighted_effective_samples"])
    assert scaled["weighted_ess_ratio"] == pytest.approx(first["weighted_ess_ratio"])


def test_binary_dependence_uses_cross_fitted_logistic_score_residuals():
    features = torch.tensor(
        [[[-1.0], [0.0], [1.0]], [[-1.0], [0.0], [1.0]], [[-1.0], [0.0], [1.0]]]
    )
    targets = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    residuals = cross_fitted_logistic_segment_residuals(
        features, targets, valid_mask=valid, logistic_l2=0.5
    )
    assert residuals.shape == tuple(targets.shape)
    assert np.isfinite(residuals).all()
    assert np.all((residuals > -1.0) & (residuals < 1.0))
    assert not np.allclose(residuals, cross_fitted_segment_residuals(features, targets, valid_mask=valid))


def test_dependence_diagnostics_preserve_variable_length_trajectories():
    scores = [np.array([[1.0], [1.0]]), np.array([[0.0], [0.5], [1.0]]), np.array([[0.2]])]
    result = estimate_dependence_diagnostics(scores, bootstrap_samples=8, seed=3)
    assert result["trajectory_count"] == 3
    assert result["segment_count"] == 6
    assert 0.0 <= result["chi_hat"] <= 1.0


def test_dependence_diagnostics_fail_closed_when_all_feedback_is_masked():
    scores = [
        np.asarray([[9.0], [9.0]], dtype=np.float64),
        np.asarray([[-7.0], [-7.0], [-7.0]], dtype=np.float64),
    ]
    valid = [np.zeros(2, dtype=bool), np.zeros(3, dtype=bool)]
    result = estimate_dependence_diagnostics(scores, valid_mask=valid, bootstrap_samples=8)
    assert result["dependence_available"] is False
    assert result["dependence_missing_reason"] == "no_valid_observations"
    assert result["nominal_observations"] == 0
    assert result["trajectory_count"] == 0
    assert result["segment_count"] == 0
    assert result["kappa_segment_by_m"] == {}
    assert np.isfinite(result["kappa_segment"])
    # Missing labels do not become observed zero scores or a fake m-grid.
    assert result["effective_samples"] == 0.0


def test_dependence_diagnostics_drops_fully_filtered_trajectories():
    scores = [
        np.asarray([[1.0], [2.0]], dtype=np.float64),
        np.asarray([[100.0], [100.0], [100.0]], dtype=np.float64),
        np.asarray([[3.0], [4.0], [5.0]], dtype=np.float64),
    ]
    valid = [
        np.asarray([True, False]),
        np.asarray([False, False, False]),
        np.asarray([True, True, True]),
    ]
    # The second trajectory is entirely unavailable and must not count as a
    # zero cluster in either the HAC estimate or the cluster bootstrap.
    result = estimate_dependence_diagnostics(scores, valid_mask=valid, bootstrap_samples=8)
    assert result["dependence_available"] is True
    assert result["trajectory_count"] == 2
    assert result["cluster_count"] == 2
    assert result["segment_count"] == 4
    assert result["nominal_observations"] == 4


def test_internal_dependence_estimator_handles_ragged_empty_rows():
    # History/bootstrap callers may pass ragged rows directly.  Empty rows
    # are absent observations and must not create a synthetic m=0 lookup.
    from verl.trainer.ppo.polyseg import _estimate_dependence_once

    result = _estimate_dependence_once(
        [np.zeros((0, 1)), np.asarray([[1.0], [2.0]])],
        [np.zeros(0), np.ones(2)],
        hac_lag=1,
        ridge=1e-6,
    )
    assert result["kappa_segment"] >= 1.0
    assert result["nominal_observations"] == 2
    assert result["cluster_count"] == 1

    missing = _estimate_dependence_once(
        [np.zeros((0, 1))], [np.zeros(0)], hac_lag=1, ridge=1e-6
    )
    assert missing["dependence_available"] is False
    assert missing["kappa_segment_by_m"] == {}


def test_hac_uses_bartlett_window_at_truncated_lag():
    # Gamma_0=1 and Gamma_1=1/3. At L=1, Bartlett's weight is 1/2,
    # so the normalized long-run variance is 4/3 (up to ridge).
    scores = np.array([[[-1.0], [-1.0], [1.0], [1.0]]] * 4)
    result = estimate_dependence_diagnostics(scores, hac_lag=1, bootstrap_samples=0, ridge=1e-6)
    assert result["kappa_segment"] == pytest.approx((4 / 3) / (1 + 1e-6))


def test_bootstrap_operational_upper_cannot_shrink_below_point_estimate():
    # Some resamples of this heterogeneous collection have lower covariance
    # inflation than the original. The operational upper takes the envelope.
    rng = np.random.default_rng(123)
    scores = rng.normal(size=(6, 5, 2))
    scores += np.arange(6)[:, None, None]
    for seed in range(5):
        result = estimate_dependence_diagnostics(scores, bootstrap_samples=1, seed=seed)
        assert result["kappa_segment_upper"] >= result["kappa_segment"]
        assert result["kappa_trajectory_upper"] >= result["kappa_trajectory"]
        assert result["iid_cluster_radius_ratio_upper"] >= result["iid_cluster_radius_ratio"]
        for m, point in result["kappa_segment_by_m"].items():
            assert result["kappa_segment_upper_by_m"][m] >= point


def test_single_trajectory_residual_does_not_fit_held_out_targets():
    features = np.ones((1, 2, 1))
    targets = np.array([[1.0, 3.0]])
    # No training fold exists. A mean fit on these held-out targets would
    # leak them into the prediction and spuriously erase cluster variation.
    assert np.array_equal(cross_fitted_segment_residuals(features, targets), targets)


def test_budget_is_transactional_and_exact():
    budget = SegmentRewardBudget(3, cost_per_query=2)
    assert budget.can_afford(1)
    budget.consume(1)
    assert budget.remaining == pytest.approx(1.0)
    with pytest.raises(RuntimeError):
        budget.consume(1)
    assert budget.remaining == pytest.approx(1.0)


def test_budget_reservation_commit_refund_and_idempotence():
    budget = SegmentRewardBudget(5)
    token = budget.reserve(3, reservation_id="query-1")
    assert token == "query-1"
    assert budget.remaining == pytest.approx(2.0)
    assert budget.reserve(3, reservation_id="query-1") == token
    assert budget.remaining == pytest.approx(2.0)
    budget.commit(token, cost=2)
    assert budget.remaining == pytest.approx(3.0)
    budget.commit(token, cost=1)  # settled query IDs are idempotent
    assert budget.remaining == pytest.approx(3.0)
    pending = budget.reserve(2, reservation_id="query-2")
    assert budget.remaining == pytest.approx(1.0)
    budget.refund_reservation(pending)
    budget.refund_reservation(pending)
    assert budget.remaining == pytest.approx(3.0)
    restored = SegmentRewardBudget.from_state_dict(budget.state_dict())
    assert restored.remaining == pytest.approx(3.0)


def test_ridge_and_logistic_segment_critic():
    z = torch.linspace(-1, 1, 32).unsqueeze(-1)
    target = 0.25 + 0.5 * z[:, 0] + 0.2 * (3 * z[:, 0] ** 2 - 1) / 2
    critic = PolynomialSegmentCritic(q=2, feature_dim=1, ridge_lambda=1e-6)
    critic.fit(z, target)
    assert torch.mean((critic.predict(z) - target) ** 2).item() < 1e-4

    binary = PolynomialSegmentCritic(q=1, feature_dim=1, feedback="binary", logistic_l2=1e-3)
    binary.fit(z, (target > target.mean()).float())
    probs = binary.predict(z)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_mlp_segment_critic_baseline_fits_macro_targets():
    z = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    target = z[:, 0].square()
    critic = MLPSegmentCritic(1, hidden_dim=8, learning_rate=0.05, steps=120)
    critic.fit(z, target)
    assert critic.dimension == 25
    assert torch.isfinite(critic.predict(z)).all()
    assert torch.mean((critic.predict(z) - target) ** 2).item() < 0.03


def test_mlp_critic_uses_shared_advantage_fit_contract():
    advantages, returns = compute_polyseg_advantage(
        token_level_rewards=torch.zeros(1, 4),
        response_mask=torch.ones(1, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "critic_type": "mlp",
            "mlp_steps": 3,
            "use_polynomial_critic": True,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]),
        },
        segment_rewards=torch.tensor([[0.5, 1.0]]),
    )
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()


def test_accumulating_ridge_state_matches_concatenated_fit_and_roundtrips():
    reset_polyseg_estimator_state()
    z = torch.tensor([[-1.0], [-0.25], [0.5], [1.0]])
    target = torch.tensor([0.2, 0.8, 1.4, 2.0])
    state = PolynomialEstimatorState(q=2, feature_dim=1, ridge_lambda=0.3)
    critic = PolynomialSegmentCritic(q=2, feature_dim=1, ridge_lambda=0.3)
    critic.fit(z[:2], target[:2], accumulate=True, state=state)
    critic.fit(z[2:], target[2:], accumulate=True, state=state)
    reference = PolynomialSegmentCritic(q=2, feature_dim=1, ridge_lambda=0.3).fit(z, target)
    assert state.sample_count == 4
    assert torch.allclose(critic.predict(z), reference.predict(z), atol=1e-6)
    snapshot = {"run": state.state_dict()}
    load_polyseg_estimator_state_dict(snapshot, clear=True)
    restored = polyseg_estimator_state_dict("run")["run"]
    assert restored["sample_count"] == 4
    assert torch.allclose(restored["gram"], state.gram)


def test_accumulating_polyseg_advantage_reports_cumulative_samples():
    reset_polyseg_estimator_state()


def test_persistent_oof_history_is_prequential_and_checkpointable():
    """Past trajectories provide the next batch's OOF fold and round-trip."""
    reset_polyseg_estimator_state()
    kwargs = dict(
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "ridge_lambda": 1.0,
            "use_polynomial_critic": True,
            "accumulate_estimator": True,
            "estimator_state_id": "persistent-oof-test",
            "dependence_history_limit": 4,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]),
        },
    )
    first = {}
    compute_polyseg_advantage(
        torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
        segment_rewards=torch.tensor([[0.0, 1.0]]), non_tensor_batch=first, **kwargs
    )
    assert first["polyseg_dependence_persistent_oof"].item() == 0.0
    second = {}
    compute_polyseg_advantage(
        torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
        segment_rewards=torch.tensor([[1.0, 0.0]]), non_tensor_batch=second, **kwargs
    )
    assert second["polyseg_dependence_persistent_oof"].item() == 1.0
    assert second["polyseg_dependence_diagnostic_oof"].item() == 1.0
    assert second["polyseg_dependence_history_trajectories"].item() == 1.0
    snapshot = polyseg_estimator_state_dict("persistent-oof-test")
    assert len(snapshot["persistent-oof-test"]["dependence_history"]) == 2
    load_polyseg_estimator_state_dict(snapshot, clear=True)
    restored = polyseg_estimator_state_dict("persistent-oof-test")["persistent-oof-test"]
    assert len(restored["dependence_history"]) == 2
    assert restored["dependence_history_limit"] == 4
    reset_polyseg_estimator_state()
    kwargs = dict(
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "ridge_lambda": 1.0,
            "use_polynomial_critic": True,
            "accumulate_estimator": True,
            "estimator_state_id": "cpu-test",
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]),
        },
    )
    first = {}
    compute_polyseg_advantage(
        torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
        segment_rewards=torch.tensor([[1.0, 2.0]]), non_tensor_batch=first, **kwargs
    )
    second = {}
    compute_polyseg_advantage(
        torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
        segment_rewards=torch.tensor([[2.0, 4.0]]), non_tensor_batch=second, **kwargs
    )
    assert first["polyseg_estimator_sample_count"].item() == 2
    assert second["polyseg_estimator_sample_count"].item() == 4
    reset_polyseg_estimator_state()


def test_accumulating_estimator_rejects_feedback_semantic_reuse():
    """A state bound to one feedback object cannot serve another protocol."""
    reset_polyseg_estimator_state()
    base = {
        "m": 2,
        "q": 1,
        "feature_dim": 1,
        "ridge_lambda": 1.0,
        "use_polynomial_critic": True,
        "accumulate_estimator": True,
        "estimator_state_id": "feedback-semantic-state",
        "normalize_advantages": False,
    }
    batch = {
        "segment_ids": torch.tensor([[0, 0, 1, 1]]),
        "segment_features": torch.tensor([[[-1.0], [1.0]]]),
    }
    compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config=base,
        batch=batch,
        segment_rewards=torch.tensor([[0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="already bound to a different PolySeg configuration"):
        compute_polyseg_advantage(
            torch.zeros(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            config={**base, "feedback": "terminal"},
            batch=batch,
            segment_rewards=torch.tensor([[0.0, 1.0]]),
        )
    reset_polyseg_estimator_state()


def test_duration_aware_macro_gae_and_token_mapping():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.zeros(1, 3)
    durations = torch.tensor([[2.0, 1.0]])
    ids = torch.tensor([[0, 0, 1, 1]])
    segment_adv, segment_returns, token_adv, token_returns = compute_segment_gae_advantage(
        rewards,
        values,
        segment_durations=durations,
        gamma=0.5,
        lam=1.0,
        token_segment_ids=ids,
        token_mask=torch.ones_like(ids, dtype=torch.bool),
    )
    assert torch.allclose(segment_adv, torch.tensor([[1.5, 2.0]]))
    assert torch.equal(token_adv, torch.tensor([[1.5, 1.5, 2.0, 2.0]]))
    assert torch.equal(segment_returns, segment_adv)
    assert torch.equal(token_returns, token_adv)


def test_macro_return_targets_respect_duration_and_padding():
    rewards = torch.tensor([[1.0, 2.0, 99.0]])
    durations = torch.tensor([[2.0, 1.0, 1.0]])
    mask = torch.tensor([[True, True, False]])
    targets = macro_return_targets(rewards, segment_durations=durations, segment_mask=mask, gamma=0.5)
    assert torch.allclose(targets, torch.tensor([[1.5, 2.0, 0.0]]))


def test_padded_duration_is_neutral_but_valid_duration_is_strictly_validated():
    rewards = torch.tensor([[1.0, 2.0, 99.0]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[True, True, False]])
    padded_zero = torch.tensor([[2.0, 1.0, 0.0]])
    padded_nan = torch.tensor([[2.0, 1.0, float("nan")]])
    expected = torch.tensor([[1.5, 2.0, 0.0]])
    for durations in (padded_zero, padded_nan):
        advantages, returns = compute_segment_gae_advantage(
            rewards,
            values,
            segment_mask=mask,
            segment_durations=durations,
            gamma=0.5,
            normalize=False,
        )
        assert torch.allclose(advantages, expected)
        assert torch.allclose(returns, expected)
    with pytest.raises(ValueError, match="valid segments"):
        compute_segment_gae_advantage(
            rewards,
            values,
            segment_mask=mask,
            segment_durations=torch.tensor([[0.0, 1.0, 0.0]]),
            normalize=False,
        )


def test_registered_polyseg_adapter_uses_macro_rewards():
    token_rewards = torch.zeros(1, 6)
    response_mask = torch.ones(1, 6, dtype=torch.bool)
    ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    advantages, returns = compute_polyseg_advantage(
        token_rewards,
        response_mask,
        config={"m": 3, "normalize_advantages": False},
        batch={"segment_ids": ids, "segment_rewards": torch.tensor([[1.0, 2.0, 3.0]])},
    )
    assert torch.equal(advantages, torch.tensor([[6.0, 6.0, 5.0, 5.0, 3.0, 3.0]]))
    assert torch.equal(returns, advantages)
    assert core_algos.get_adv_estimator_fn("polyseg") is not None


def test_registered_polyseg_adapter_can_project_values_to_polynomial_features():
    token_rewards = torch.zeros(1, 6)
    response_mask = torch.ones(1, 6, dtype=torch.bool)
    ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    features = build_segment_features([(0, 2), (2, 4), (4, 6)], [1.0, 1.0, 1.0], response_length=6)
    advantages, returns = compute_polyseg_advantage(
        token_rewards,
        response_mask,
        values=torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]]),
        config={"m": 3, "q": 1, "use_polynomial_critic": True, "normalize_advantages": False},
        batch={
            "segment_ids": ids,
            "segment_rewards": torch.tensor([[1.0, 2.0, 3.0]]),
            "segment_features": features.unsqueeze(0),
        },
    )
    assert advantages.shape == returns.shape == token_rewards.shape
    assert torch.isfinite(advantages).all()


def test_polynomial_critic_defaults_to_macro_return_target():
    token_rewards = torch.zeros(1, 4)
    response_mask = torch.ones(1, 4, dtype=torch.bool)
    ids = torch.tensor([[0, 0, 1, 1]])
    # A linear basis over two distinct boundaries represents the exact
    # reward-to-go targets [3, 2].  Deliberately provide unrelated transformer
    # values: fitting those values would produce non-zero GAE, while fitting
    # the method's macro return target produces zero Bellman advantage.
    features = torch.tensor([[[-1.0], [1.0]]])
    batch = {
        "segment_ids": ids,
        "segment_rewards": torch.tensor([[1.0, 2.0]]),
        "segment_features": features,
    }
    advantages, returns = compute_polyseg_advantage(
        token_rewards,
        response_mask,
        values=torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "ridge_lambda": 1e-8,
            "use_polynomial_critic": True,
            "normalize_advantages": False,
        },
        batch=batch,
    )
    assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-4)
    assert torch.allclose(returns, torch.tensor([[3.0, 3.0, 2.0, 2.0]]), atol=1e-4)


def test_binary_polyseg_uses_logistic_latent_feedback_and_probability_diagnostic():
    non_tensor = {}
    token_rewards = torch.zeros(2, 4)
    response_mask = torch.ones(2, 4, dtype=torch.bool)
    features = torch.tensor([[[-1.0], [1.0]], [[-0.5], [0.5]]])
    ids = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    labels = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    advantages, returns = compute_polyseg_advantage(
        token_rewards,
        response_mask,
        config={
            "m": 2,
            "q": 2,
            "feature_dim": 1,
            "feedback": "binary_local",
            "logistic_l2": 0.1,
            "normalize_advantages": False,
            "use_polynomial_critic": True,
        },
        batch={"segment_ids": ids, "segment_rewards": labels, "segment_features": features},
        non_tensor_batch=non_tensor,
    )
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()
    assert non_tensor["polyseg_binary_mean_probability"].shape == (2,)
    assert 0.0 <= float(non_tensor["polyseg_binary_mean_probability"].mean()) <= 1.0


def test_binary_terminal_latent_reward_is_not_canceled_by_value_baseline():
    token_rewards = torch.zeros(1, 4)
    response_mask = torch.ones(1, 4, dtype=torch.bool)
    advantages, returns = compute_polyseg_advantage(
        token_rewards,
        response_mask,
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "feedback": "binary_local",
            "logistic_l2": 0.1,
            "normalize_advantages": False,
            "use_polynomial_critic": True,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[0.0, 1.0]]),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]),
        },
    )
    # The terminal logit reward and sigmoid value baseline are distinct; a
    # shared latent tensor would make this terminal advantage exactly zero.
    assert abs(float(advantages[0, -1])) > 1e-4
    assert torch.isfinite(returns).all()


def test_polynomial_critic_exports_diagnostics_without_tensor_fields():
    non_tensor = {}
    compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "ridge_lambda": 1e-8,
            "use_polynomial_critic": True,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[1.0, 2.0]]),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]),
        },
        non_tensor_batch=non_tensor,
    )
    assert non_tensor["polyseg_critic_mse"].shape == (1,)
    assert non_tensor["polyseg_bellman_residual_mse"].shape == (1,)


def test_polynomial_critic_exports_spectral_stability_and_heldout_diagnostics():
    non_tensor = {}
    compute_polyseg_advantage(
        torch.zeros(4, 4),
        torch.ones(4, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 2,
            "feature_dim": 1,
            "ridge_lambda": 1.0,
            "use_polynomial_critic": True,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]).repeat(4, 1),
            "segment_rewards": torch.tensor([[1.0, 2.0]]).repeat(4, 1),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]).repeat(4, 1, 1),
        },
        non_tensor_batch=non_tensor,
    )
    for key in (
        "polyseg_empirical_spectral_tail",
        "polyseg_theta_q_stability",
        "polyseg_heldout_return_residual_mse",
        "polyseg_heldout_bellman_residual_mse",
    ):
        assert non_tensor[key].shape == (4,)
        assert np.isfinite(non_tensor[key]).all()


def test_explicit_macro_transition_backup_exports_transition_diagnostics():
    token_rewards = torch.zeros(2, 6)
    response_mask = torch.ones(2, 6, dtype=torch.bool)
    ids = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]])
    features = torch.tensor([[[-1.0], [0.0], [1.0]], [[-0.5], [0.5], [1.0]]])
    next_features = torch.tensor([[[0.0], [1.0], [0.0]], [[0.5], [1.0], [0.0]]])
    transition_mask = torch.tensor([[True, True, False], [True, True, False]])
    non_tensor = {}
    advantages, returns = compute_polyseg_advantage(
        token_rewards,
        response_mask,
        config={
            "m": 3,
            "q": 1,
            "feature_dim": 1,
            "ridge_lambda": 0.1,
            "use_polynomial_critic": True,
            "use_transition_estimator": True,
            "transition_ridge_lambda": 0.1,
            "transition_backup_iterations": 2,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": ids,
            "segment_rewards": torch.tensor([[1.0, 0.5, 0.25], [0.5, 1.0, 0.0]]),
            "segment_features": features,
            "segment_next_features": next_features,
            "segment_transition_mask": transition_mask,
        },
        non_tensor_batch=non_tensor,
    )
    assert advantages.shape == returns.shape == token_rewards.shape
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()
    assert non_tensor["polyseg_transition_feature_condition"].shape == (2,)
    assert non_tensor["polyseg_transition_feature_min_eigenvalue"].shape == (2,)
    assert non_tensor["polyseg_transition_target_semantics"].tolist() == ["p_v_test_function"] * 2


def test_single_turn_rollout_successor_metadata_drives_transition_backup():
    metadata = build_single_turn_segment_metadata([1, 1, 1, 1, 1, 1], configured_m=3, feature_dim=1, feature_horizon=6)
    ids = torch.tensor([metadata["segment_ids"]])
    features = build_segment_features(
        metadata["segment_boundaries"],
        metadata["segment_durations"],
        response_length=6,
        action_lengths=metadata["segment_action_lengths"],
        feature_dim=1,
    ).unsqueeze(0)
    advantages, returns = compute_polyseg_advantage(
        torch.zeros(1, 6),
        torch.ones(1, 6, dtype=torch.bool),
        config={
            "m": 3,
            "q": 1,
            "feature_dim": 1,
            "use_polynomial_critic": True,
            "use_transition_estimator": True,
            "transition_backup_iterations": 1,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": ids,
            "segment_rewards": torch.tensor([[1.0, 0.5, 0.25]]),
            "segment_features": features,
            "segment_next_features": torch.tensor([metadata["segment_next_features"]], dtype=torch.float32),
            "segment_transition_mask": torch.tensor([metadata["segment_transition_mask"]]),
        },
    )
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()


def test_optional_confidence_bonus_is_exported_and_changes_macro_targets():
    non_tensor = {}
    _, returns = compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "ridge_lambda": 1.0,
            "use_polynomial_critic": True,
            "use_confidence_bonus": True,
            "reward_confidence_beta": 1.0,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[1.0, 0.0]]),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]),
        },
        non_tensor_batch=non_tensor,
    )
    assert torch.isfinite(returns).all()
    assert non_tensor["polyseg_confidence_bonus"].item() > 0
    assert non_tensor["polyseg_dependence_diagnostic_oof"].item() == 0
    assert non_tensor["polyseg_dependence_radius_multiplier"].item() >= 1.0
    assert np.isclose(
        non_tensor["polyseg_dependence_radius_multiplier"].item(),
        non_tensor["polyseg_iid_cluster_radius_ratio_upper"].item(),
    )


def test_confidence_bonus_uses_pre_update_ridge_prior():
    """Current-batch design rows must not shrink their own optimism bonus."""
    non_tensor = {}
    compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 0,
            "feature_dim": 1,
            "ridge_lambda": 1.0,
            "use_polynomial_critic": True,
            "use_confidence_bonus": True,
            "reward_confidence_beta": 1.0,
            "r_max": 100.0,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[0.0, 0.0]]),
            "segment_durations": torch.tensor([[10.0, 10.0]]),
            "segment_features": torch.tensor([[[0.0], [1.0]]]),
        },
        non_tensor_batch=non_tensor,
    )
    # q=0 has phi(z)=[1].  With only the ridge prior, ||phi||_{I^-1}=1;
    # using the current two rows would incorrectly reduce it to sqrt(1/3).
    assert non_tensor["polyseg_confidence_bonus"].item() == pytest.approx(1.0, abs=1e-5)


def test_confidence_bonus_uses_pre_update_dependence_radius_and_commits_after_batch():
    reset_polyseg_dependence_state()
    common = dict(
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "ridge_lambda": 1.0,
            "use_polynomial_critic": True,
            "use_confidence_bonus": True,
            "dependence_state_id": "pre-update-radius-test",
            "normalize_advantages": False,
            "r_max": 100.0,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]),
        },
    )
    first = {}
    compute_polyseg_advantage(
        torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
        segment_rewards=torch.tensor([[0.0, 1.0]]), non_tensor_batch=first, **common
    )
    assert first["polyseg_dependence_radius_multiplier_pre_update"].item() == pytest.approx(1.0)
    assert first["polyseg_dependence_state_batches"].item() == pytest.approx(1.0)
    saved_radius = first["polyseg_dependence_radius_multiplier_post_update"].item()
    second = {}
    compute_polyseg_advantage(
        torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
        segment_rewards=torch.tensor([[1.0, 0.0]]), non_tensor_batch=second, **common
    )
    assert second["polyseg_dependence_radius_multiplier_pre_update"].item() == pytest.approx(saved_radius)
    assert second["polyseg_dependence_state_batches"].item() == pytest.approx(2.0)
    reset_polyseg_dependence_state()


def test_current_labels_do_not_change_same_batch_online_confidence_bonus():
    reset_polyseg_dependence_state()
    common = dict(
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 1,
            "use_polynomial_critic": True,
            "use_confidence_bonus": True,
            "normalize_advantages": False,
            "r_max": 100.0,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_features": torch.tensor([[[-1.0], [1.0]]]),
        },
    )
    left, right = {}, {}
    compute_polyseg_advantage(
        torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
        segment_rewards=torch.tensor([[0.0, 0.0]]), non_tensor_batch=left, **common
    )
    reset_polyseg_dependence_state()
    compute_polyseg_advantage(
        torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
        segment_rewards=torch.tensor([[1.0, 1.0]]), non_tensor_batch=right, **common
    )
    assert left["polyseg_dependence_radius_multiplier_pre_update"].item() == pytest.approx(1.0)
    assert right["polyseg_dependence_radius_multiplier_pre_update"].item() == pytest.approx(1.0)
    assert left["polyseg_confidence_bonus"].item() == pytest.approx(
        right["polyseg_confidence_bonus"].item(), abs=1e-6
    )
    reset_polyseg_dependence_state()


def test_dependence_state_checkpoint_roundtrip_preserves_next_pre_update_radius():
    reset_polyseg_dependence_state()
    config = {
        "m": 2, "q": 1, "feature_dim": 1, "use_polynomial_critic": True,
        "use_confidence_bonus": True, "dependence_state_id": "checkpoint-radius-test",
        "normalize_advantages": False, "r_max": 100.0,
    }
    batch = {"segment_ids": torch.tensor([[0, 0, 1, 1]]), "segment_features": torch.tensor([[[-1.0], [1.0]]])}
    first = {}
    compute_polyseg_advantage(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), config=config, batch=batch,
                              segment_rewards=torch.tensor([[0.0, 1.0]]), non_tensor_batch=first)
    snapshot = polyseg_dependence_state_dict("checkpoint-radius-test")
    expected = snapshot["checkpoint-radius-test"]["dependence_radius_multiplier"]
    reset_polyseg_dependence_state()
    load_polyseg_dependence_state_dict(snapshot, clear=True)
    second = {}
    compute_polyseg_advantage(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), config=config, batch=batch,
                              segment_rewards=torch.tensor([[1.0, 0.0]]), non_tensor_batch=second)
    assert second["polyseg_dependence_radius_multiplier_pre_update"].item() == pytest.approx(expected)
    assert second["polyseg_dependence_state_batches"].item() == pytest.approx(2.0)
    reset_polyseg_dependence_state()


def test_failed_batch_does_not_advance_dependence_state():
    reset_polyseg_dependence_state()
    config = {
        "m": 2, "q": 1, "feature_dim": 1, "use_polynomial_critic": True,
        "use_confidence_bonus": True, "dependence_state_id": "failed-radius-test",
        "normalize_advantages": False, "r_max": 100.0, "bellman_closure_tolerance": 0.0,
    }
    batch = {"segment_ids": torch.tensor([[0, 0, 1, 1]]), "segment_features": torch.tensor([[[-1.0], [1.0]]])}
    # An invalid feature shape fails before the commit hook and must not leave
    # a partially updated history/radius in the process registry.
    with pytest.raises(ValueError, match="segment_features"):
        compute_polyseg_advantage(
            torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), config=config, batch=batch,
            segment_rewards=torch.tensor([[0.0, 1.0]]),
            segment_features=torch.zeros(1, 3, 1),
        )
    state = polyseg_dependence_state_dict("failed-radius-test")["failed-radius-test"]
    assert state["dependence_diagnostic_batches"] == 0
    assert state["dependence_history"] == []
    reset_polyseg_dependence_state()


def test_polynomial_critic_reconstructs_features_when_transport_drops_them():
    advantages, returns = compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 1,
            "feature_dim": 4,
            "ridge_lambda": 1.0,
            "use_polynomial_critic": True,
            "normalize_advantages": False,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[1.0, 2.0]]),
        },
    )
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()


def test_mixed_adaptive_q_m_batch_fails_before_advantage_computation():
    with pytest.raises(ValueError, match="uniform"):
        compute_polyseg_advantage(
            torch.zeros(2, 4),
            torch.ones(2, 4, dtype=torch.bool),
            config={"q": 1, "m": 2, "normalize_advantages": False},
            batch={
                "segment_ids": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
                "segment_rewards": torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
            },
            non_tensor_batch={
                "polyseg_adaptive_q": np.asarray([0, 1]),
                "polyseg_adaptive_m": np.asarray([2, 2]),
            },
        )


def test_adaptive_selector_metadata_overrides_runtime_q_and_m():
    non_tensor = {
        "polyseg_adaptive_q": np.asarray([0]),
        "polyseg_adaptive_m": np.asarray([2]),
    }
    advantages, returns = compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={"q": 4, "m": 4, "feature_dim": 4, "normalize_advantages": False},
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[1.0, 2.0]]),
        },
        non_tensor_batch=non_tensor,
    )
    assert torch.equal(advantages, torch.tensor([[3.0, 3.0, 2.0, 2.0]]))
    assert torch.equal(returns, advantages)


def test_feedback_mask_excludes_unqueried_segments_from_macro_advantage():
    advantages, returns = compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={"m": 2, "q": 0, "normalize_advantages": False},
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[1.0, 9.0]]),
            "segment_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "segment_feedback_mask": torch.tensor([[1, 0]], dtype=torch.bool),
        },
    )
    assert torch.equal(advantages, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    assert torch.equal(returns, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_all_feedback_missing_returns_finite_zero_signal_for_polynomial_critic():
    advantages, returns = compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 2,
            "feature_dim": 4,
            "use_polynomial_critic": True,
            "normalize_advantages": True,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.zeros(1, 2),
            "segment_mask": torch.ones(1, 2, dtype=torch.bool),
            "segment_feedback_mask": torch.zeros(1, 2, dtype=torch.bool),
        },
    )
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()
    assert torch.equal(advantages, torch.zeros_like(advantages))
    assert torch.equal(returns, torch.zeros_like(returns))


def test_suffix_difference_and_budgeted_choice():
    assert torch.allclose(suffix_potential_differences(torch.tensor([[0.2, 0.5, 0.4]])), torch.tensor([[0.2, 0.3, -0.1]]))
    choice = choose_polynomial_segment_config(
        q_candidates=[0, 1, 2, 4],
        m_candidates=[1, 2, 4, 8],
        episodes=100,
        horizon=16,
        budget=300,
        feature_dim=2,
        feedback="binary",
    )
    assert choice.m <= 3
    assert choice.query_cost <= 300
    assert choice.feedback_factor > 1


def test_budgeted_polyseg_controller_selects_and_restores_state():
    controller = BudgetedPolySegController(
        q_candidates=[0, 1, 2],
        m_candidates=[1, 2, 4],
        total_budget=16,
        segment_cost=1.0,
        stat_constant=0.1,
        approx_constant=0.1,
        segment_constant=0.1,
    )
    choice = controller.choose(episodes=2, horizon=8, feature_dim=1)
    assert choice.query_cost <= 16
    assert controller.observe(choice.query_cost) == pytest.approx(16 - choice.query_cost)
    restored = BudgetedPolySegController.from_state_dict(controller.state_dict())
    assert restored.remaining_budget == pytest.approx(controller.remaining_budget)
    assert restored.last_choice == choice


def test_effective_round_selector_uses_conservative_dependence_bounds():
    # With one observed segment count, the effective-round statistic is
    # K*m/(kappa_seg^U*kappa_tr^U); the power selector remains available as
    # an explicit compatibility convention.
    effective = choose_polynomial_segment_config(
        q_candidates=[0],
        m_candidates=[2],
        episodes=10,
        horizon=8,
        budget=None,
        feature_dim=1,
        selector_norm="effective_round",
        kappa_segment_upper_by_m={2: 1.0},
        kappa_trajectory_upper=2.0,
        stat_constant=1.0,
        approx_constant=0.0,
        segment_constant=0.0,
    )
    assert effective.statistical_term == pytest.approx(math.sqrt(10.0))
    power = choose_polynomial_segment_config(
        q_candidates=[0],
        m_candidates=[2],
        episodes=10,
        horizon=8,
        budget=None,
        feature_dim=1,
        selector_norm="power",
        dependence_chi=1.0,
        stat_constant=1.0,
        approx_constant=0.0,
        segment_constant=0.0,
    )
    assert power.statistical_term == pytest.approx(math.sqrt(20.0))

    q_specific = choose_polynomial_segment_config(
        q_candidates=[0, 1],
        m_candidates=[2],
        episodes=10,
        horizon=8,
        budget=None,
        feature_dim=1,
        selector_norm="effective_round",
        kappa_segment_upper_by_m={2: 1.0},
        kappa_segment_upper_by_qm={(1, 2): 2.0},
        kappa_trajectory_upper=1.0,
        stat_constant=1.0,
        approx_constant=0.0,
        segment_constant=0.0,
    )
    # Both orders have one effective sample per trajectory: q=1 was
    # measured, while q=0 falls back conservatively instead of borrowing
    # the legacy m-only bound. The lower-dimensional q=0 is selected.
    assert q_specific.q == 0


def test_selector_tracks_q_dependent_critic_compute_and_budget():
    common = dict(
        q_candidates=[0, 2], m_candidates=[2], episodes=4, horizon=8,
        budget=None, feature_dim=2, selector_norm="power",
        stat_constant=0.0, approx_constant=0.0, segment_constant=0.0,
        critic_compute_cost=1.0,
    )
    q0 = choose_polynomial_segment_config(**common, compute_budget=8)
    assert q0.q == 0
    assert q0.compute_cost == pytest.approx(4 * 2 * 1 * 1)
    # feature_dim=2 has C(2+2,2)=6 polynomial features at q=2, so its
    # declared critic work is 6x the q=0 candidate and cannot fit the same
    # compute budget.
    with pytest.raises(ValueError, match="no feasible"):
        choose_polynomial_segment_config(
            **{**common, "q_candidates": [2]}, compute_budget=8
        )
    unrestricted = choose_polynomial_segment_config(
        **{**common, "q_candidates": [2]}, compute_budget=None
    )
    assert unrestricted.compute_cost == pytest.approx(4 * 2 * 6)


def test_controller_compute_budget_roundtrip_and_observation():
    controller = BudgetedPolySegController(
        q_candidates=[0, 2], m_candidates=[2], total_budget=None,
        critic_compute_cost=1.0, compute_budget=8,
        stat_constant=0.0, approx_constant=0.0, segment_constant=0.0,
    )
    choice = controller.choose(episodes=4, horizon=8, feature_dim=2)
    assert choice.q == 0 and choice.compute_cost == pytest.approx(8)
    controller.observe(choice.query_cost, compute_cost=choice.compute_cost, episodes=4)
    assert controller.remaining_compute_budget == pytest.approx(0)
    restored = BudgetedPolySegController.from_state_dict(controller.state_dict())
    assert restored.remaining_compute_budget == pytest.approx(0)


def test_compute_budget_skip_reason_is_distinct_and_transactional():
    controller = BudgetedPolySegController(
        q_candidates=[0], m_candidates=[2], total_budget=None,
        critic_compute_cost=1.0, compute_budget=1.0, budget_fallback="skip",
    )
    choice = controller.choose(episodes=2, horizon=8, feature_dim=1)
    assert choice.fallback == "skip"
    assert choice.fallback_reason == "critic_compute_budget_exhausted_before_batch"
    assert controller.consumed_budget == pytest.approx(0)
    with pytest.raises(RuntimeError, match="critic compute budget exhausted"):
        controller.observe(2.0, compute_cost=2.0, episodes=2)
    assert controller.consumed_budget == pytest.approx(0)


def test_controller_budget_skip_fallback_is_explicit_and_serializable():
    controller = BudgetedPolySegController(
        q_candidates=[0, 2],
        m_candidates=[2, 4],
        total_budget=3,
        segment_cost=1.0,
        budget_fallback="skip",
    )
    choice = controller.choose(episodes=2, horizon=8, feature_dim=1)
    assert choice.fallback == "skip"
    assert choice.fallback_reason == "feedback_budget_exhausted_before_batch"
    assert choice.query_cost == 0.0
    assert controller.remaining_budget == pytest.approx(3.0)
    restored = BudgetedPolySegController.from_state_dict(controller.state_dict())
    assert restored.budget_fallback == "skip"
    assert restored.choose(episodes=2, horizon=8, feature_dim=1).fallback == "skip"


def test_controller_budget_error_remains_fail_closed():
    controller = BudgetedPolySegController(
        q_candidates=[0], m_candidates=[2], total_budget=1, segment_cost=1.0
    )
    with pytest.raises(ValueError, match="no feasible"):
        controller.choose(episodes=2, horizon=8, feature_dim=1)


def test_controller_roundtrips_dependence_selector_state():
    controller = BudgetedPolySegController(
        q_candidates=[0, 1],
        m_candidates=[2, 4],
        total_budget=None,
        selector_norm="effective_round",
    )
    controller.update_dependence_diagnostics(m=2, q=1, kappa_segment_upper=1.25, kappa_trajectory_upper=1.75)
    assert controller.kappa_segment_upper_by_m == {}
    assert controller.kappa_segment_upper_by_qm[(1, 2)] == pytest.approx(1.25)
    assert controller.kappa_trajectory_upper == pytest.approx(1.75)
    restored = BudgetedPolySegController.from_state_dict(controller.state_dict())
    assert restored.selector_norm == "effective_round"
    assert restored.kappa_segment_upper_by_m == {}
    assert restored.kappa_segment_upper_by_qm == {(1, 2): pytest.approx(1.25)}
    assert restored.kappa_trajectory_upper == pytest.approx(1.75)


@pytest.mark.parametrize("selector_norm", ["effective_round", "segment_update"])
def test_unseen_order_does_not_borrow_other_orders_dependence(selector_norm):
    controller = BudgetedPolySegController(
        q_candidates=[2], m_candidates=[4], total_budget=None, selector_norm=selector_norm,
    )
    controller.update_dependence_diagnostics(m=4, q=0, kappa_segment_upper=1, kappa_trajectory_upper=1)
    state = controller.state_dict()
    # Older states duplicated the q=0 observation into the m-only cache.
    state["kappa_segment_upper_by_m"] = {4: 1.0}
    restored = BudgetedPolySegController.from_state_dict(state)
    # D_2=3 > N_eff^L=2 for an unseen q=2. Borrowing q=0's bound would
    # incorrectly admit the candidate with N_eff=8.
    with pytest.raises(ValueError, match="no feasible"):
        restored.choose(episodes=2, horizon=4, feature_dim=1)
    restored.update_dependence_diagnostics(m=4, q=2, kappa_segment_upper=1, kappa_trajectory_upper=1)
    assert restored.choose(episodes=2, horizon=4, feature_dim=1).q == 2


def test_segment_update_feasibility_uses_effective_information_not_regret_penalty():
    kwargs = dict(
        q_candidates=[2], m_candidates=[4], episodes=4, horizon=4, budget=None, feature_dim=1,
        selector_norm="segment_update", kappa_segment_upper_by_qm={(2, 4): 4},
    )
    # D_2=3 fits four independent trajectory clusters, but not two.
    choice = choose_polynomial_segment_config(**kwargs, kappa_trajectory_upper=1)
    assert choice.statistical_term == pytest.approx(3 * math.sqrt(4 * 4 * 4))
    with pytest.raises(ValueError, match="no feasible"):
        choose_polynomial_segment_config(**kwargs, kappa_trajectory_upper=2)


def test_transition_estimator_macro_ratio_and_suffix_budget():
    z = torch.linspace(-1, 1, 32).unsqueeze(-1)
    next_features = torch.cat([z**2, 1 - z], dim=-1)
    transition = PolynomialTransitionEstimator(q=2, feature_dim=1, ridge_lambda=1e-5)
    transition.fit(z, next_features)
    assert torch.mean((transition.predict_next_features(z) - next_features) ** 2).item() < 1e-4

    ids = torch.tensor([[0, 0, 1, 1]])
    old = torch.zeros(1, 4)
    current = torch.tensor([[0.1, 0.2, -0.1, 0.0]])
    ratios = segment_log_prob_ratio(current, old, ids)
    assert torch.allclose(ratios, torch.exp(torch.tensor([[0.3, -0.1]])))
    loss, metrics = segment_ppo_loss(current, old, torch.ones(1, 2), ids)
    assert torch.isfinite(loss) and "segment/clip_fraction" in metrics

    budget = SegmentRewardBudget(4)
    collector = SuffixFeedbackCollector(lambda prefix: prefix == "good", budget=budget, samples_per_boundary=2)
    rewards = collector.collect(["good", "bad"])
    assert torch.allclose(rewards, torch.tensor([1.0, -1.0]))
    assert budget.remaining == pytest.approx(0.0)


def test_transition_estimator_fits_pv_directly_for_nonlinear_value():
    """The Bellman target is E[V(X')|X,U], not V(E[X'|X,U])."""
    z = torch.linspace(-1.0, 1.0, 12).repeat_interleave(8).unsqueeze(-1)
    successor = torch.where(
        torch.arange(z.shape[0]).remainder(2).eq(0).unsqueeze(-1),
        torch.ones_like(z),
        -torch.ones_like(z),
    )
    direct = PolynomialTransitionEstimator(q=2, feature_dim=1, ridge_lambda=1e-6)
    direct.fit_test_function(z, successor, lambda value: value.square().squeeze(-1))
    direct_pv = direct.predict_test_function(torch.tensor([[-0.7], [0.0], [0.8]]))
    assert torch.allclose(direct_pv, torch.ones_like(direct_pv), atol=1e-3, rtol=1e-3)

    vector = PolynomialTransitionEstimator(q=2, feature_dim=1, ridge_lambda=1e-6)
    vector.fit(z, successor)
    plug_in = vector.predict_next_features(torch.tensor([[-0.7], [0.0], [0.8]])).squeeze(-1).square()
    # The conditional mean successor is approximately zero, so plugging it
    # into the nonlinear value function produces a different estimand.
    assert torch.all(plug_in < 1e-2)


def test_reward_and_transition_confidence_bonuses_are_finite_and_nonnegative():
    features = legendre_features(torch.tensor([[-1.0], [0.0], [1.0]]), q=2)
    gram = features.T @ features + 0.1 * torch.eye(features.shape[-1])
    reward_bonus = confidence_bonus(features, gram=gram, beta=2.0)
    transition_bonus = transition_confidence_bonus(features, gram=gram, beta=3.0)
    assert reward_bonus.shape == transition_bonus.shape == (3,)
    assert torch.isfinite(reward_bonus).all() and torch.all(reward_bonus >= 0)
    assert torch.allclose(transition_bonus, reward_bonus * 1.5)

    transition = PolynomialTransitionEstimator(q=2, feature_dim=1, ridge_lambda=0.1)
    z = torch.linspace(-1, 1, 8).unsqueeze(-1)
    transition.fit(z, torch.cat([z, z.square()], dim=-1))
    assert torch.isfinite(transition.confidence_bonus(z, beta=2.0)).all()


def test_macro_policy_loss_uses_one_ratio_per_segment():
    old = torch.zeros(1, 4)
    current = torch.tensor([[0.05, 0.05, -0.05, 0.0]])
    advantages = torch.tensor([[2.0, 2.0, 1.0, 1.0]])
    ids = torch.tensor([[0, 0, 1, 1]])
    response_mask = torch.ones_like(ids, dtype=torch.bool)
    loss, metrics = compute_segment_policy_loss(
        old_log_prob=old,
        log_prob=current,
        token_advantages=advantages,
        response_mask=response_mask,
        segment_ids_tensor=ids,
        num_segments=2,
        clip_ratio=0.2,
    )
    expected = -torch.tensor([math.exp(0.1) * 2.0, math.exp(-0.05) * 1.0]).mean()
    assert torch.allclose(loss, expected)
    assert metrics["segment/ratio_mean"].item() == pytest.approx((math.exp(0.1) + math.exp(-0.05)) / 2)


def test_macro_policy_loss_excludes_missing_feedback_segments():
    old = torch.zeros(1, 4)
    current = torch.tensor([[0.05, 0.05, -0.05, 0.0]])
    advantages = torch.tensor([[2.0, 2.0, 1.0, 1.0]])
    ids = torch.tensor([[0, 0, 1, 1]])
    loss, metrics = compute_segment_policy_loss(
        old_log_prob=old,
        log_prob=current,
        token_advantages=advantages,
        response_mask=torch.ones_like(ids, dtype=torch.bool),
        segment_ids_tensor=ids,
        segment_feedback_mask=torch.tensor([[True, False]]),
        num_segments=2,
        clip_ratio=0.2,
    )
    expected = -torch.tensor(math.exp(0.1) * 2.0)
    assert torch.allclose(loss, expected)
    assert metrics["segment/ratio_mean"].item() == pytest.approx(math.exp(0.1))


@pytest.mark.parametrize("num_segments", [1, 2])
def test_macro_policy_loss_accepts_launcher_m_segment_feedback(num_segments):
    """Actor/reward metadata for every configured q×m cell must align."""
    old = torch.zeros(1, 6)
    current = torch.full((1, 6), 0.01)
    ids = torch.zeros(1, 6, dtype=torch.long)
    if num_segments == 2:
        ids[0, 3:] = 1
    response_mask = torch.ones_like(ids, dtype=torch.bool)
    feedback = torch.ones(1, num_segments, dtype=torch.bool)
    loss, metrics = compute_segment_policy_loss(
        old_log_prob=old,
        log_prob=current,
        token_advantages=torch.ones(1, 6),
        response_mask=response_mask,
        segment_ids_tensor=ids,
        segment_feedback_mask=feedback,
        num_segments=num_segments,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["segment/ratio_mean"])


def test_macro_policy_loss_pads_short_ragged_feedback_mask_fail_closed():
    """Missing ragged transport entries are excluded instead of crashing."""
    old = torch.zeros(1, 6)
    current = torch.full((1, 6), 0.01)
    ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    loss, metrics = compute_segment_policy_loss(
        old_log_prob=old,
        log_prob=current,
        token_advantages=torch.ones(1, 6),
        response_mask=torch.ones_like(ids, dtype=torch.bool),
        segment_ids_tensor=ids,
        # The third segment was observed in token metadata, but its feedback
        # mask was dropped by a narrower ragged TransferQueue microbatch.
        segment_feedback_mask=torch.tensor([[True, True]]),
        num_segments=3,
    )
    assert torch.allclose(loss, -torch.exp(torch.tensor(0.02)))
    assert metrics["segment/ratio_mean"].item() == pytest.approx(math.exp(0.02))


def test_macro_bellman_backup_clips_action_values():
    q_values, values = macro_bellman_backup(
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.zeros(1, 2, 2),
        segment_durations=torch.ones(1, 2),
        r_max=1.0,
        horizon=2,
    )
    assert torch.equal(q_values, torch.tensor([[[1.0, 2.0], [1.0, 1.0]]]))
    assert torch.equal(values, torch.tensor([[2.0, 1.0]]))


def test_main_polyseg_advantage_clips_optimistic_returns_to_duration_bound():
    """The production advantage path must enforce the Methods value range."""
    advantages, returns = compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 0,
            "feature_dim": 1,
            "use_polynomial_critic": True,
            "use_confidence_bonus": True,
            "reward_confidence_beta": 1000.0,
            "normalize_advantages": False,
            "r_max": 1.0,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[1.0, 1.0]]),
            "segment_durations": torch.tensor([[1.0, 1.0]]),
            "segment_features": torch.tensor([[[0.0], [1.0]]]),
        },
    )
    # Returns are scattered to tokens: the first segment has two units of
    # remaining duration and the terminal segment has one.
    assert torch.isfinite(advantages).all()
    assert torch.equal(returns, torch.tensor([[2.0, 2.0, 1.0, 1.0]]))


def test_main_polyseg_advantage_clips_malformed_positive_rewards():
    """A bad external score cannot bypass the same upper value envelope."""
    _, returns = compute_polyseg_advantage(
        torch.zeros(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        config={
            "m": 2,
            "q": 0,
            "feature_dim": 1,
            "use_polynomial_critic": True,
            "normalize_advantages": False,
            "r_max": 1.0,
        },
        batch={
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
            "segment_rewards": torch.tensor([[100.0, 100.0]]),
            "segment_durations": torch.tensor([[1.0, 1.0]]),
            "segment_features": torch.tensor([[[0.0], [1.0]]]),
        },
    )
    assert torch.equal(returns, torch.tensor([[2.0, 2.0, 1.0, 1.0]]))


def test_signed_macro_bounds_and_potential_deltas_are_not_clipped():
    q_values, values = macro_bellman_backup(
        torch.tensor([[[-2.0], [-1.0]]]),
        torch.zeros(1, 2, 1),
        segment_durations=torch.ones(1, 2),
        r_max=1.0,
        horizon=2,
    )
    assert torch.equal(q_values, torch.tensor([[[-2.0], [-1.0]]]))
    assert torch.equal(values, torch.tensor([[-2.0, -1.0]]))
    deltas = suffix_potential_differences(torch.tensor([[0.2, -0.5, -0.1]]))
    assert torch.allclose(deltas.sum(dim=1), torch.tensor([[-0.1]]))


def test_binary_feedback_factor_saturates_in_log_domain():
    choice = choose_polynomial_segment_config(
        q_candidates=[0],
        m_candidates=[1],
        episodes=1,
        horizon=10**12,
        budget=None,
        feature_dim=1,
        feedback="binary_local",
        r_max=1.0,
        approx_constant=0.0,
        segment_constant=0.0,
    )
    assert math.isfinite(choice.feedback_factor)
    assert math.isfinite(choice.statistical_term)
    assert math.isfinite(choice.objective)


def test_custom_reward_preserves_structured_math_dapo_accuracy():
    from examples.polyseg_trainer.reward import compute_score

    correct = compute_score(
        "math_dapo",
        r"\boxed{42}",
        "42",
        {"polyseg_protocol": "terminal"},
    )
    wrong = compute_score(
        "math_dapo",
        r"\boxed{41}",
        "42",
        {"polyseg_protocol": "terminal"},
    )
    assert correct["score"] == 1.0 and correct["acc"] is True
    assert wrong["score"] == -1.0 and wrong["acc"] is False


def test_custom_reward_exposes_scalar_gsm8k_correctness_as_accuracy(monkeypatch):
    from examples.polyseg_trainer import reward

    monkeypatch.setattr(reward, "default_compute_score", lambda *args, **kwargs: 1.0)
    result = reward.compute_score("openai/gsm8k", "answer", "42", {"polyseg_protocol": "terminal"})
    assert result["score"] == 1.0
    assert result["acc"] == 1.0


def test_exact_sum_selector_has_no_compression_bias_term():
    choice = choose_polynomial_segment_config(
        q_candidates=[1], m_candidates=[2], episodes=4, horizon=8, budget=None,
        feature_dim=1, feedback="local_sum", segment_constant=100.0,
    )
    assert choice.segment_term == 0.0


def test_explicit_adaptive_qm_overrides_nested_config():
    cfg = PolySegConfig.from_any({"polyseg": {"q": 2, "m": 4}}, q=0, m=2)
    assert cfg.q == 0 and cfg.m == 2


def test_kl_token_component_is_added_once_to_explicit_macro_reward():
    rewards = torch.zeros(1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0, 1, 1]])
    segment_rewards = torch.tensor([[2.0, 3.0]])
    token_rewards = torch.tensor([[0.5, 2.0, 0.25, 3.0]])
    _, returns = compute_polyseg_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        config={"m": 2, "use_kl_in_reward": True, "normalize_advantages": False},
        batch={"segment_rewards": segment_rewards, "segment_ids": segment_ids},
    )
    assert torch.allclose(returns, torch.tensor([[5.75, 5.75, 3.25, 3.25]]))


def test_bound_and_selector_share_the_statistical_normalization():
    common = dict(q=2, m=4, episodes=32, horizon=16, feature_dim=2, feedback="local_sum")
    power = segment_feedback_bound(**common, selector_norm="power")
    effective = segment_feedback_bound(**common, selector_norm="effective_round")
    segment_update = segment_feedback_bound(**common, selector_norm="segment_update")
    assert power["statistical_rounds"] == pytest.approx(32 * 4)
    assert effective["statistical_rounds"] <= segment_update["statistical_rounds"]
    assert power["segment"] == effective["segment"] == 0.0
    assert effective["selector_norm"] == "effective_round"
    # The selector itself must use the same rounds; a mismatch here was the
    # original "reported bound != optimized objective" defect.
    choice = choose_polynomial_segment_config(
        q_candidates=[2], m_candidates=[4], episodes=32, horizon=16, budget=None,
        feature_dim=2, feedback="local_sum", selector_norm="effective_round",
    )
    assert choice.statistical_term == pytest.approx(effective["statistical"], rel=1e-6)


def test_q_diagnostics_expose_availability_and_prediction_stability():
    features = torch.tensor(
        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.1], [0.2, 0.9]]], dtype=torch.float32
    )
    rewards = torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=torch.float32)
    non_tensor = {}
    compute_polyseg_advantage(
        token_level_rewards=torch.zeros(2, 2),
        response_mask=torch.ones(2, 2, dtype=torch.bool),
        config={"m": 2, "q": 2, "feature_dim": 2, "use_polynomial_critic": True,
                "normalize_advantages": False},
        batch={"segment_rewards": rewards, "segment_features": features,
               "segment_ids": torch.tensor([[0, 1], [0, 1]])},
        non_tensor_batch=non_tensor,
    )
    assert "polyseg_prediction_q_stability" in non_tensor
    assert float(non_tensor["polyseg_q_stability_available"][0]) == 1.0


def test_unavailable_heldout_diagnostics_are_nan_not_zero():
    features = torch.zeros(1, 2, 2)
    rewards = torch.zeros(1, 2)
    non_tensor = {}
    compute_polyseg_advantage(
        token_level_rewards=torch.zeros(1, 2),
        response_mask=torch.ones(1, 2, dtype=torch.bool),
        config={"m": 2, "q": 2, "feature_dim": 2, "use_polynomial_critic": True,
                "normalize_advantages": False},
        batch={"segment_rewards": rewards, "segment_features": features,
               "segment_ids": torch.tensor([[0, 1]])},
        non_tensor_batch=non_tensor,
    )
    assert float(non_tensor["polyseg_heldout_diagnostics_available"][0]) == 0.0
    assert math.isnan(float(non_tensor["polyseg_heldout_return_residual_mse"][0]))


def test_gae_does_not_bridge_across_a_masked_middle_segment():
    rewards = torch.tensor([[1.0, 0.0, 2.0]])
    mask = torch.tensor([[True, False, True]])
    values = torch.tensor([[0.0, 100.0, 0.0]])
    advantages, returns = compute_segment_gae_advantage(
        rewards, values, segment_mask=mask, gamma=1.0, lam=1.0,
    )
    # Segment 0 must not use segment 2's value as successor through the hole.
    assert advantages[0, 0].item() == pytest.approx(1.0)
    assert returns[0, 1].item() == 0.0


def test_polyseg_research_requires_rollout_segment_ids():
    with pytest.raises(ValueError, match="rollout-owned segment_ids"):
        compute_polyseg_advantage(
            torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool),
            config={"m": 2, "q": 1, "feature_dim": 1, "use_polynomial_critic": True},
            batch={"segment_rewards": torch.tensor([[1.0, 2.0]]),
                   "segment_features": torch.zeros(1, 2, 1)},
        )


def test_weighted_logistic_fit_keeps_binary_labels():
    z = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    weight = torch.tensor([10.0, 1.0, 1.0, 10.0])
    critic = PolynomialSegmentCritic(q=1, feature_dim=1, feedback="binary", logistic_l2=1e-3)
    critic.fit(z, y, sample_weight=weight)
    assert torch.all((critic.targets == 0) | (critic.targets == 1))
    assert torch.isfinite(critic.theta).all()


def test_terminal_selector_cost_is_one_query_per_episode():
    choice = choose_polynomial_segment_config(
        q_candidates=[1], m_candidates=[8], episodes=4, horizon=8, budget=4,
        feature_dim=1, feedback="terminal", segment_cost=1.0,
    )
    assert choice.m == 8 and choice.query_cost == 4.0
