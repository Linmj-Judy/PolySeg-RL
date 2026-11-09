import torch
from types import SimpleNamespace

from verl.trainer.ppo.polyseg import (
    FrozenLowDimRepresentation,
    SharedStepRewardEstimator,
    aggregate_step_features,
    bellman_closure_residual,
    dependence_confidence_radius,
    budget_feasible_segments,
    unified_regret_bound,
    recover_step_rewards_from_segment_feedback,
    build_segment_features,
)
from verl.trainer.ppo.polyseg_experiments import (
    SmoothMDPConfig,
    calibration_metrics,
    run_synthetic_polyseg_experiment,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase


def test_frozen_representation_roundtrip_and_fingerprint():
    samples = torch.linspace(-1.0, 1.0, 20).reshape(10, 2)
    representation = FrozenLowDimRepresentation.fit(samples, output_dim=1, seed=7)
    restored = FrozenLowDimRepresentation.from_state_dict(representation.state_dict())
    torch.testing.assert_close(representation.transform(samples), restored.transform(samples))
    assert representation.fingerprint == restored.fingerprint


def test_frozen_representation_drives_rollout_segment_and_step_features():
    hidden = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [-0.5, 0.25]])
    representation = FrozenLowDimRepresentation.fit(hidden, output_dim=2, seed=11)
    features = build_segment_features(
        [(0, 2), (2, 4)], [1.0, 1.0], response_length=4,
        feature_dim=2, state_features=hidden, representation=representation,
    )
    expected = representation.transform(hidden).reshape(2, 2, 2).mean(dim=1)
    torch.testing.assert_close(features, expected)


def test_rollout_representation_loader_checks_hash_and_maps_features(tmp_path):
    samples = torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.5]])
    representation = FrozenLowDimRepresentation.fit(samples, output_dim=2, seed=3)
    path = tmp_path / "representation.pt"
    torch.save(representation.state_dict(), path)
    class _Loop(AgentLoopBase):
        async def run(self, sampling_params, **kwargs):
            raise NotImplementedError

    loop = object.__new__(_Loop)
    loop.rollout_config = SimpleNamespace(
        polyseg_representation_path=str(path),
        polyseg_representation_hash=representation.fingerprint,
        polyseg_feature_dim=2,
    )
    loop._polyseg_representation_cache = None
    loop._polyseg_representation_cache_path = None
    mapped, state, loaded, fingerprint = loop._prepare_polyseg_step_features(
        {"polyseg_step_features": samples}, response_length=3
    )
    assert loaded.fingerprint == representation.fingerprint == fingerprint
    torch.testing.assert_close(mapped, representation.transform(samples))
    torch.testing.assert_close(state, samples)


def test_shared_step_reward_recovery_uses_segment_sum_design():
    step_features = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    segment_ids = torch.tensor([[0, 0, 1, 1]])
    feedback = torch.tensor([[3.0, 7.0]])
    aggregate = aggregate_step_features(step_features, segment_ids, num_segments=2)
    torch.testing.assert_close(aggregate, torch.tensor([[[3.0], [7.0]]]))
    recovered, estimator = recover_step_rewards_from_segment_feedback(
        step_features, segment_ids, feedback, ridge_lambda=1e-6
    )
    assert estimator.theta is not None
    assert recovered.shape == (1, 4)
    torch.testing.assert_close(recovered.sum(1), torch.tensor([10.0]), atol=1e-4, rtol=1e-4)


def test_shared_step_reward_recovery_excludes_observation_padding():
    features = torch.tensor([[[1.0], [100.0], [2.0], [200.0]]])
    ids = torch.tensor([[0, 0, 1, -1]])
    feedback = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[True, False, True, False]])
    recovered, first = recover_step_rewards_from_segment_feedback(
        features, ids, feedback, ridge_lambda=1e-6, step_mask=mask
    )
    _, second = recover_step_rewards_from_segment_feedback(
        torch.tensor([[[1.0], [999.0], [2.0], [777.0]]]),
        ids,
        feedback,
        ridge_lambda=1e-6,
        step_mask=mask,
    )
    assert torch.isfinite(recovered).all()
    # The excluded tool-observation/padding rows cannot affect the fitted
    # shared parameter or the segment sums used for recovery.
    torch.testing.assert_close(first.theta, second.theta)


def test_confidence_and_bellman_audit_are_finite():
    radius = dependence_confidence_radius(
        dimension=4, effective_samples=100, delta=0.05, kappa_segment=2, kappa_trajectory=1.5
    )
    assert radius > 0
    residual = bellman_closure_residual(
        torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 0.0]]), torch.tensor([[3.0, 2.0]])
    )
    assert float(residual) == 0.0


def test_synthetic_suite_reports_required_metrics():
    metrics = run_synthetic_polyseg_experiment(
        episodes=4, q=1, m=2, config=SmoothMDPConfig(horizon=4, state_dim=2, seed=3)
    )
    for key in ("critic_mse", "bellman_closure_residual", "segment_regret", "queries_per_episode", "brier", "ece"):
        assert key in metrics
        assert metrics[key] >= 0
    assert metrics["queries_per_episode"] == 2


def test_calibration_metrics_are_zero_for_perfect_predictions():
    metrics = calibration_metrics(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
    assert metrics["brier"] == 0.0
    assert metrics["ece"] == 0.0


def test_unified_regret_bound_and_budget_comparator_terms():
    assert budget_feasible_segments(episodes=10, horizon=8, budget=25, segment_cost=1.0) == 2
    bound = unified_regret_bound(episodes=10, horizon=8, q=2, m=2, feature_dim=2, bellman_error=0.1)
    assert bound["feature_dimension"] == 6
    assert bound["total"] >= bound["statistical"]
