"""CPU contracts for the PolySeg-Sci v4 multi-fidelity primitives."""

import math
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.config.algorithm import PolySegScienceConfig
from verl.trainer.config import AlgoConfig, PolySegAlgorithmConfig
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.polyseg_sci import (
    ActionRecord,
    BoundarySnapshot,
    BudgetLedger,
    BudgetedQuerySampler,
    BranchResult,
    CausalSegmenter,
    EndpointObservation,
    EndpointSchema,
    EndpointSpec,
    FeatureMapContract,
    FidelityFeatureMapContract,
    load_fidelity_feature_map_contract,
    load_feature_map_contract,
    FidelitySpec,
    FreshBranchRunner,
    RestoreAudit,
    run_science_round,
    PolySegSciVerifier,
    PolynomialVectorPredictor,
    QueryCandidate,
    QueryDecision,
    StateAuditOracle,
    constrained_direction,
    scalarize_boundary_values,
    projected_dual_ascent,
    MultiFidelityCalibrator,
    mean_observations,
    residual_correction_covariance,
    trajectory_bootstrap_ci,
    pareto_min_norm_direction,
    preference_direction,
    residual_correct,
    segment_score_gradient,
    parameter_score_gradients,
    ScoreGradientSketcher,
    compute_polyseg_science_advantage,
    compute_polyseg_science_advantage_registered,
    ScientificEndpointEvaluator,
    ConstraintDualState,
    BoundaryRisk,
    compute_boundary_risk,
    compute_gradient_risk_covariance,
    estimate_boundary_error_covariance,
    AuditDesign,
    BernoulliRiskAllocator,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils import losses as worker_losses
from verl.workers.utils.losses import ppo_loss


def _schema(mode="preference"):
    hf = lambda endpoint: FidelitySpec(f"{endpoint}-hf", "HF", f"evaluator-{endpoint}", "hf-v1", 5.0, kind="gold")
    lf = lambda endpoint: FidelitySpec(f"{endpoint}-lf", "LF", f"evaluator-{endpoint}", "lf-v1", 1.0)
    endpoints = (
        EndpointSpec("activity", "activity", "a.u.", valid_range=(0, 1), hf_fidelity_id="activity-hf", fidelities=(lf("activity"), hf("activity")), priority=2),
        EndpointSpec("stability", "stability", "hours", valid_range=(0, 100), hf_fidelity_id="stability-hf", fidelities=(lf("stability"), hf("stability")), constraint="lower" if mode == "constraint" else None, threshold=40 if mode == "constraint" else None),
    )
    return EndpointSchema(
        endpoints,
        objective_mode=mode,
        weights=(0.6, 0.4) if mode == "preference" else None,
        primary_objective="activity" if mode == "constraint" else None,
    )


def test_endpoint_schema_is_vector_and_hf_is_gold():
    schema = _schema()
    assert schema.dimension == 2
    assert schema.fingerprint
    schema.validate_vector([0.5, 42.0], ["observed", "observed"])
    with pytest.raises(ValueError, match="invalid observed"):
        schema.validate_vector([None, 42.0], ["observed", "observed"])
    with pytest.raises(ValueError, match="outside valid range"):
        schema.validate_vector([2.0, 42.0], ["observed", "observed"])


def test_feature_map_contract_roundtrip_and_hash_guard(tmp_path):
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    path = tmp_path / "feature_map.json"
    path.write_text(json.dumps(feature_map.to_dict()), encoding="utf-8")
    restored = load_feature_map_contract(path, expected_hash=feature_map.fingerprint)
    assert restored.to_dict() == feature_map.to_dict()
    with pytest.raises(ValueError, match="feature_map_hash"):
        load_feature_map_contract(path, expected_hash="0" * 64)
    with pytest.raises(ValueError, match="HF fidelity"):
        FidelitySpec("bad", "HF", "proxy", "v1", 1.0, kind="proxy")


def test_fidelity_feature_map_loader_roundtrip_and_hash_guard(tmp_path):
    slots = ("activity::activity-lf", "stability::stability-lf")
    contract = FidelityFeatureMapContract.fit(
        [[0.1, 20.0], [0.8, 60.0]],
        [[True, True], [True, True]],
        slots,
    )
    path = tmp_path / "fidelity_feature_map.json"
    path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    restored = load_fidelity_feature_map_contract(path, expected_hash=contract.fingerprint)
    assert restored.to_dict() == contract.to_dict()
    with pytest.raises(ValueError, match="fidelity_feature_map_hash"):
        load_fidelity_feature_map_contract(path, expected_hash="0" * 64)


def test_endpoint_schema_json_round_trip_preserves_fidelity_contract():
    schema = _schema()
    restored = EndpointSchema.from_dict(schema.to_dict())
    assert restored.to_dict() == schema.to_dict()
    assert restored.fingerprint == schema.fingerprint
    assert tuple(f.fidelity_id for f in restored.endpoints[0].fidelities) == (
        "activity-lf",
        "activity-hf",
    )


def test_endpoint_schema_supports_detection_limits_and_interval_bounds_without_zero_fill():
    endpoint = EndpointSpec(
        "activity",
        "activity",
        "a.u.",
        valid_range=(0.0, 1.0),
        detection_limit=0.1,
        quantification_limit=0.2,
        hf_fidelity_id="activity-hf",
        missing_policy="interval",
        fidelities=(FidelitySpec("activity-hf", "HF", "evaluator", "v1", 1.0, kind="gold"),),
    )
    schema = EndpointSchema((endpoint,), objective_mode="preference", weights=(1.0,))
    below = EndpointObservation("activity", "activity-hf", None, status="below_detection_limit", detection_limit=0.1)
    schema.validate_observation(below)
    interval = EndpointObservation("activity", "activity-hf", 0.15, status="interval", lower_bound=0.1, upper_bound=0.2)
    schema.validate_observation(interval)
    with pytest.raises(ValueError, match="below_detection_limit"):
        EndpointObservation("activity", "activity-hf", 0.0, status="below_detection_limit", detection_limit=0.1)
    with pytest.raises(ValueError, match="ordered finite bounds"):
        EndpointObservation("activity", "activity-hf", 0.15, status="interval", lower_bound=0.2, upper_bound=0.1)


def test_proxy_only_endpoint_has_no_hf_claim():
    proxy = EndpointSpec(
        "proxy",
        "proxy",
        "score",
        hf_fidelity_id="",
        proxy_only=True,
        fidelities=(FidelitySpec("proxy-lf", "LF", "proxy", "v1", 1.0),),
    )
    schema = EndpointSchema((proxy,), objective_mode="preference", weights=(1.0,))
    assert schema.endpoints[0].proxy_only


def test_endpoint_validation_rejects_missing_policy_mismatch_and_failed_values():
    schema = _schema()
    with pytest.raises(ValueError, match="not registered for censored"):
        schema.validate_observation(EndpointObservation("activity", "activity-hf", 0.5, status="censored"))
    with pytest.raises(ValueError, match="null value"):
        schema.validate_observation(
            EndpointObservation("activity", "activity-hf", 0.5, status="branch_failed", failure_class="branch_failure")
        )
    with pytest.raises(ValueError, match="null value"):
        EndpointObservation("activity", "activity-hf", 0.5, status="branch_failed", failure_class="branch_failure")
    # Vector-level validation uses the same missingness contract as the
    # observation object and must accept an explicitly recorded failure with
    # no scientific value.
    schema.validate_vector([None, 42.0], ["branch_failed", "observed"])
    with pytest.raises(ValueError, match="null value"):
        schema.validate_vector([0.5, 42.0], ["branch_failed", "observed"])


def test_predictor_rejects_an_endpoint_with_no_training_observations():
    features = [[0.0], [1.0]]
    fmap = FeatureMapContract.fit(features, ["progress"])
    predictor = PolynomialVectorPredictor(_schema(), fmap, q=0)
    with pytest.raises(ValueError, match="missing endpoints"):
        predictor.fit(features, [[0.1, None], [0.9, None]])


@pytest.mark.parametrize("q", [1.5, True, "2"])
def test_predictor_rejects_non_integer_polynomial_degree(q):
    fmap = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    with pytest.raises(ValueError, match="non-negative integer"):
        PolynomialVectorPredictor(_schema(), fmap, q=q)


@pytest.mark.parametrize("ridge_lambda", [True, "1.0"])
def test_predictor_rejects_non_numeric_ridge_lambda(ridge_lambda):
    fmap = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    with pytest.raises(ValueError, match="ridge_lambda"):
        PolynomialVectorPredictor(_schema(), fmap, q=1, ridge_lambda=ridge_lambda)


def test_action_record_and_causal_segmenter_exclude_tool_observations():
    actions = [ActionRecord(f"a{i}", "token", i not in {2, 5}, i, i + 1) for i in range(8)]
    mask = [a.is_policy_action for a in actions]
    segments = CausalSegmenter(m=2, horizon=8).segment(mask)
    assert [s.action_indices for s in segments] == [(0, 1, 3), (4, 6, 7)]
    assert all(2 not in s.action_indices and 5 not in s.action_indices for s in segments)
    with pytest.raises(ValueError, match="strictly increasing"):
        CausalSegmenter(rule="event").segment(mask, event_positions=[4, 3])


def test_fixed_horizon_segmenter_clips_final_bin_to_early_terminated_response():
    # The registered horizon is eight tokens, but this response terminates at
    # token five.  The last active bin must not reference token positions 5-7.
    segments = CausalSegmenter(m=2, horizon=8).segment([False, False, False, False, True])
    assert [(segment.start, segment.end, segment.action_indices) for segment in segments] == [(4, 5, (4,))]


def test_frozen_feature_map_and_multioutput_crossfit():
    features = [[0.0, 0.0], [0.5, 0.2], [1.0, 1.0], [0.2, 0.8]]
    fmap = FeatureMapContract.fit(features, ["progress", "evidence"])
    z = fmap.transform([[0.0, 1.0]])
    assert torch.all(z.abs() <= 1)
    schema = _schema()
    predictor = PolynomialVectorPredictor(schema, fmap, q=1, ridge_lambda=1e-3)
    predictor.fit(features, [[0.1, 20], [0.3, 30], [0.9, 80], [0.2, 50]], groups=["a", "a", "b", "b"], cross_fit=True)
    prediction = predictor.predict(features)
    assert prediction.shape == (4, 2)
    assert predictor.oof_predictions is not None
    assert predictor.diagnostics()["endpoint_dimension"] == 2
    state = predictor.state_dict()
    assert state["feature_map"]["version"] == fmap.version


def test_fidelity_features_and_masks_enter_frozen_predictor_and_checkpoint():
    schema = _schema()
    fmap = FeatureMapContract.fit([[0.0], [1.0], [0.5]], ["progress"])
    slots = ("activity::activity-lf", "stability::stability-lf")
    values = [[0.1, 20.0], [0.8, 60.0], [0.4, 40.0]]
    mask = [[True, True], [True, False], [False, True]]
    fidelity_map = FidelityFeatureMapContract.fit(values, mask, slots)
    predictor = PolynomialVectorPredictor(schema, fmap, q=1, fidelity_feature_map=fidelity_map)
    predictor.fit(
        [[0.0], [0.5], [1.0]],
        [[0.2, 20.0], [0.5, 40.0], [0.9, 60.0]],
        fidelity_values=values,
        fidelity_mask=mask,
    )
    full = predictor.predict([[0.25]], [[0.2, 30.0]], [[True, True]])
    missing = predictor.predict([[0.25]], [[0.0, 0.0]], [[False, False]])
    assert full.shape == (1, schema.dimension)
    assert missing.shape == full.shape
    state = predictor.state_dict()
    restored = PolynomialVectorPredictor.from_state_dict(state)
    assert restored.fidelity_feature_map is not None
    torch.testing.assert_close(
        restored.predict([[0.25]], [[0.2, 30.0]], [[True, True]]),
        full,
    )


def test_fidelity_feature_contract_rejects_unknown_slot_order_and_missing_mask():
    schema = _schema()
    fmap = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    with pytest.raises(ValueError, match="slots"):
        PolynomialVectorPredictor(
            schema,
            fmap,
            q=0,
            fidelity_feature_map=FidelityFeatureMapContract.fit(
                [[0.0, 0.0], [1.0, 1.0]],
                [[True, True], [True, True]],
                ["stability::stability-lf", "activity::activity-lf"],
            ),
        )
    with pytest.raises(ValueError, match="non-finite"):
        fmap.transform([[float("inf")]])


def test_cross_fit_fails_closed_when_a_fold_has_no_endpoint_training_data():
    features = [[0.0], [0.5], [1.0]]
    fmap = FeatureMapContract.fit(features, ["progress"])
    predictor = PolynomialVectorPredictor(_schema(), fmap, q=0)
    # The stability endpoint exists only in the group held out for the first
    # fold.  A zero OOF head would be a fabricated scientific prediction.
    with pytest.raises(ValueError, match="every fold"):
        predictor.fit(
            features,
            [[0.1, None], [0.4, None], [0.9, 80.0]],
            groups=["a", "a", "b"],
            cross_fit=True,
        )


def test_predictor_per_endpoint_regularization_and_frozen_output_scaling():
    features = [[0.0], [0.5], [1.0], [0.25]]
    fmap = FeatureMapContract.fit(features, ["progress"])
    predictor = PolynomialVectorPredictor(_schema(), fmap, q=1, ridge_lambda=(1e-3, 10.0))
    predictor.fit(features, [[0.1, 10.0], [0.3, 20.0], [0.9, 80.0], [0.2, 50.0]])
    diagnostics = predictor.diagnostics()
    assert len(diagnostics["condition_numbers"]) == 2
    assert len(diagnostics["minimum_eigenvalues"]) == 2
    assert predictor.state_dict()["target_scale"][1] > 0


def test_predictor_accepts_explicit_missing_endpoint_mask():
    features = [[0.0], [1.0], [0.5]]
    fmap = FeatureMapContract.fit(features, ["progress"])
    predictor = PolynomialVectorPredictor(_schema(), fmap, q=0)
    predictor.fit(features, [[0.1, None], [0.9, 80], [0.4, 40]], groups=["a", "b", "c"])
    assert predictor.predict([[0.25]]).shape == (1, 2)
    with pytest.raises(ValueError, match="observed predictor targets"):
        predictor.fit(features, torch.tensor([[float("nan"), 20.0], [0.9, 80.0], [0.4, 40.0]]), target_mask=[[True, True], [True, True], [True, True]])
    with pytest.raises(ValueError, match="observed predictor targets"):
        predictor.fit(features, [[float("nan"), 20.0], [0.9, 80.0], [0.4, 40.0]])


def test_hard_budget_sampler_records_first_and_second_order_propensity():
    candidates = tuple(QueryCandidate(f"b{i}", "activity", 1, i + 1, 1, 2.0) for i in range(4))
    decision = BudgetedQuerySampler().sample(candidates, 4.0, hard_budget=True, seed=7)
    assert len(decision.selected_ids) == 2
    assert all(math.isclose(p, 0.5) for p in decision.inclusion_probabilities.values())
    assert all(math.isclose(p, 1 / 6) for p in decision.pair_inclusion_probabilities.values())
    assert decision.planned_cost == 4.0
    adaptive_equal = BudgetedQuerySampler().sample(
        candidates, 4.0, mode="adaptive", hard_budget=True, seed=7
    )
    assert all(0 < value <= 1 for value in adaptive_equal.inclusion_probabilities.values())
    assert adaptive_equal.planned_cost <= 4.0
    unequal = (candidates[0], QueryCandidate("b9", "activity", 1, 1, 1, 3.0))
    unequal_decision = BudgetedQuerySampler().sample(unequal, 3, hard_budget=True, seed=7)
    assert unequal_decision.inclusion_probabilities["b0::activity"] == pytest.approx(1 / 3)
    assert unequal_decision.inclusion_probabilities["b9::activity"] == pytest.approx(1 / 3)
    assert unequal_decision.pair_inclusion_probabilities["b0::activity::b9::activity"] == pytest.approx(0.0)
    assert unequal_decision.planned_cost in {0.0, 2.0, 3.0}
    adaptive_unequal = BudgetedQuerySampler().sample(unequal, 3, mode="adaptive", seed=7)
    assert all(value > 0 for value in adaptive_unequal.inclusion_probabilities.values())
    with pytest.raises(ValueError, match="exact-enumeration limit"):
        BudgetedQuerySampler().sample(
            tuple(QueryCandidate(f"b{i}", "activity", 1, 1, 1, 1 + i % 2) for i in range(19)),
            5,
            hard_budget=True,
        )


def test_query_sampler_enforces_minimum_inclusion_probability():
    candidates = (
        QueryCandidate("a", "activity", 0.0, 0.0, 1.0, 1.0),
        QueryCandidate("b", "activity", 4.0, 1.0, 1.0, 1.0),
    )
    decision = BudgetedQuerySampler().sample(
        candidates,
        0.4,
        mode="adaptive",
        min_inclusion_probability=0.2,
        seed=4,
    )
    assert decision.inclusion_probabilities["a::activity"] == pytest.approx(0.2)
    assert decision.inclusion_probabilities["b::activity"] >= 0.2
    assert sum(
        candidate.cost * decision.inclusion_probabilities[BudgetedQuerySampler._key(candidate)]
        for candidate in candidates
    ) <= 0.4 + 1e-8
    with pytest.raises(ValueError, match="minimum inclusion probability"):
        BudgetedQuerySampler().sample(
            candidates,
            0.39,
            min_inclusion_probability=0.2,
        )


def test_adaptive_sampler_uses_clipped_kkt_floor_directly():
    candidates = (
        QueryCandidate("high", "activity", 4.0, 1.0, 1.0, 1.0),
        QueryCandidate("low", "activity", 1.0, 1.0, 1.0, 1.0),
    )
    decision = BudgetedQuerySampler().sample(
        candidates,
        0.6,
        mode="adaptive",
        min_inclusion_probability=0.2,
        seed=3,
    )
    # The solution to p_i = clip(scale * sqrt(risk_i / cost_i), .2, 1)
    # spends the budget at p=(.4, .2).  Adding the floor to every residual
    # allocation would incorrectly produce (.333..., .266...).
    assert decision.inclusion_probabilities == pytest.approx(
        {"high::activity": 0.4, "low::activity": 0.2}
    )
    assert sum(
        candidate.cost * decision.inclusion_probabilities[BudgetedQuerySampler._key(candidate)]
        for candidate in candidates
    ) == pytest.approx(0.6)


def test_hard_budget_minimum_inclusion_probability_is_fail_closed():
    candidates = tuple(QueryCandidate(f"b{i}", "activity", 1.0, 1.0, 1.0, 1.0) for i in range(4))
    decision = BudgetedQuerySampler().sample(
        candidates,
        2.0,
        hard_budget=True,
        min_inclusion_probability=0.5,
        seed=9,
    )
    assert all(value == pytest.approx(0.5) for value in decision.inclusion_probabilities.values())
    with pytest.raises(ValueError, match="minimum inclusion probability"):
        BudgetedQuerySampler().sample(
            candidates,
            2.0,
            hard_budget=True,
            min_inclusion_probability=0.75,
        )


def test_adaptive_sampler_uses_square_root_gradient_risk_allocation():
    candidates = (
        QueryCandidate("a", "activity", 4.0, 1.0, 1.0, 1.0),
        QueryCandidate("b", "activity", 1.0, 1.0, 1.0, 1.0),
    )
    decision = BudgetedQuerySampler().sample(candidates, 1.0, mode="adaptive", seed=3)
    assert decision.inclusion_probabilities["a::activity"] > decision.inclusion_probabilities["b::activity"]
    assert sum(decision.inclusion_probabilities.values()) <= 1.0 + 1e-6


def test_adaptive_sampler_probabilities_are_invariant_to_common_risk_scale():
    base = (
        QueryCandidate("a", "activity", 4.0, 1.0, 1.0, 1.0),
        QueryCandidate("b", "activity", 1.0, 1.0, 1.0, 2.0),
    )
    scaled = (
        QueryCandidate("a", "activity", 400.0, 10.0, 1.0, 1.0),
        QueryCandidate("b", "activity", 100.0, 10.0, 1.0, 2.0),
    )
    first = BudgetedQuerySampler().sample(base, 1.0, mode="adaptive", seed=4)
    second = BudgetedQuerySampler().sample(scaled, 1.0, mode="adaptive", seed=4)
    assert first.inclusion_probabilities == pytest.approx(second.inclusion_probabilities)


def test_adaptive_sampler_expected_budget_and_positive_selected_propensities():
    candidates = (QueryCandidate("a", "activity", 1, 1, 1, 1), QueryCandidate("b", "activity", 3, 2, 1, 1))
    decision = BudgetedQuerySampler().sample(candidates, 1.0, mode="adaptive", seed=1)
    assert decision.planned_cost <= 2
    assert all(0 <= p <= 1 for p in decision.inclusion_probabilities.values())
    assert all(decision.inclusion_probabilities[key] > 0 for key in decision.selected_ids)


def test_adaptive_sampler_does_not_spend_budget_on_zero_risk_candidates():
    candidates = (
        QueryCandidate("zero", "activity", 0.0, 1.0, 1.0, 1.0),
        QueryCandidate("risky", "activity", 1.0, 1.0, 1.0, 1.0),
    )
    decision = BudgetedQuerySampler().sample(candidates, 1.0, mode="adaptive", seed=0)
    assert decision.inclusion_probabilities["zero::activity"] == 0.0
    assert decision.inclusion_probabilities["risky::activity"] == 1.0


def test_adaptive_sampler_handles_all_zero_risk_without_unbounded_bracketing():
    candidate = QueryCandidate("zero", "activity", 0.0, 0.0, 1.0, 1.0)
    decision = BudgetedQuerySampler().sample((candidate,), 10.0, mode="adaptive", seed=0)
    assert decision.selected_ids == ()
    assert decision.inclusion_probabilities["zero::activity"] == 0.0
    assert decision.planned_cost == 0.0


def test_zero_propensity_cannot_be_reported_as_unbiased_correction():
    candidate = QueryCandidate("b0", "activity", 0.0, 1.0, 1.0, 1.0)
    decision = BudgetedQuerySampler().sample((candidate,), 1.0, mode="adaptive", seed=0)
    _, valid, audit = residual_correct(
        torch.tensor([[0.2, 30.0]]),
        decision,
        {},
        endpoint_ids=["activity", "stability"],
        boundary_ids=["b0"],
    )
    assert not valid.any()
    assert not audit["unbiased_under_response_model"]
    assert "b0::activity" in audit["zero_propensity"]


def test_residual_correction_keeps_unqueried_and_failed_values_explicit():
    candidates = (QueryCandidate("b0", "activity", 1, 1, 1, 1), QueryCandidate("b1", "activity", 1, 1, 1, 1))
    decision = BudgetedQuerySampler().sample(candidates, 1, hard_budget=True, seed=0)
    observations = {"b0::activity": EndpointObservation("activity", "activity-hf", 0.8)}
    corrected, valid, audit = residual_correct(torch.tensor([[0.2, 30.0], [0.4, 50.0]]), decision, observations, endpoint_ids=["activity", "stability"], boundary_ids=["b0", "b1"])
    if "b0" in decision.selected_ids[0]:
        assert valid[0, 0]
        assert torch.isclose(corrected[0, 0], torch.tensor(1.4, dtype=torch.float64))
    assert len(audit["failed"]) <= 1
    assert not valid[:, 1].any()
    selected_boundary = decision.selected_ids[0].split("::", 1)[0]
    failed = {f"{selected_boundary}::activity": EndpointObservation("activity", "activity-hf", None, status="branch_failed", failure_class="branch_failure")}
    corrected, valid, failed_audit = residual_correct(
        torch.tensor([[0.2, 30.0]]), decision, failed, endpoint_ids=["activity", "stability"], boundary_ids=[selected_boundary]
    )
    assert torch.allclose(corrected, torch.tensor([[0.2, 30.0]], dtype=torch.float64))
    assert not failed_audit["unbiased_under_response_model"]
    assert not valid.any()
    assert failed_audit["selection_probabilities"][f"{selected_boundary}::activity"] > 0
    assert failed_audit["valid_observation_mask"][f"{selected_boundary}::activity"] is False
    assert failed_audit["selected"][0]["response_probability"] is None
    corrected, valid, response_audit = residual_correct(
        torch.tensor([[0.2, 30.0]]), decision, failed, endpoint_ids=["activity", "stability"], boundary_ids=[selected_boundary], response_probabilities={f"{selected_boundary}::activity": 0.5}, response_model_validated=True
    )
    assert response_audit["unbiased_under_response_model"] and not valid.any()


def test_observed_residual_without_response_probability_is_flagged_as_assumption():
    candidate = QueryCandidate("b0", "activity", 1, 1, 1, 1)
    decision = BudgetedQuerySampler().sample((candidate,), 1, hard_budget=True)
    corrected, valid, audit = residual_correct(
        torch.tensor([[0.2, 30.0]]),
        decision,
        {"b0::activity": EndpointObservation("activity", "activity-hf", 0.8)},
        endpoint_ids=["activity", "stability"],
        boundary_ids=["b0"],
    )
    assert valid[0, 0] and corrected[0, 0] == pytest.approx(0.8)
    assert audit["response_model_assumption_required"]
    assert not audit["unbiased_under_response_model"]
    assert audit["selection_probabilities"]["b0::activity"] == pytest.approx(1.0)
    assert audit["response_probabilities"]["b0::activity"] == pytest.approx(1.0)
    assert audit["valid_observation_mask"]["b0::activity"] is True


def test_numeric_response_probability_still_requires_model_attestation():
    candidate = QueryCandidate("b0", "activity", 1, 1, 1, 1)
    decision = BudgetedQuerySampler().sample((candidate,), 1, hard_budget=True)
    _, valid, audit = residual_correct(
        torch.tensor([[0.2, 30.0]]),
        decision,
        {"b0::activity": EndpointObservation("activity", "activity-hf", 0.8)},
        endpoint_ids=["activity", "stability"],
        boundary_ids=["b0"],
        response_probabilities={"b0::activity": 0.5},
    )
    assert valid[0, 0]
    assert audit["response_model_assumption_required"]
    assert not audit["unbiased_under_response_model"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_model_validated": "false"},
        {"selection_probabilities": {"b0::activity": "1.0"}},
        {"response_probabilities": {"b0::activity": True}},
        {"valid_observation_mask": {"b0::activity": "true"}},
    ],
)
def test_residual_correction_rejects_coercible_audit_fields(kwargs):
    candidate = QueryCandidate("b0", "activity", 1, 1, 1, 1)
    decision = BudgetedQuerySampler().sample((candidate,), 1, hard_budget=True)
    with pytest.raises(ValueError):
        residual_correct(
            torch.tensor([[0.2]]),
            decision,
            {"b0::activity": EndpointObservation("activity", "activity-hf", 0.8)},
            endpoint_ids=["activity"],
            boundary_ids=["b0"],
            **kwargs,
        )


def test_snapshot_oracle_fails_closed_on_state_mismatch():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {"file": "h"}, {"dep": "h"}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    restored = {"boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1", "feature_map_hash": "fm-hash", "segment_rule_version": "seg-v1", "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session", "tokenizer_version": "tok-v1", "dependency_hashes": {"dep": "h"}, "sampling_config": {}, "rng_state": "rng", "execution_budget_remaining": 6, "artifact_hashes": {"file": "h"}, "rng_independent": True}
    passed = StateAuditOracle().audit(snapshot, restored)
    assert passed.passed
    failed = StateAuditOracle().audit(snapshot, {**restored, "policy_version": "old"})
    assert not failed.passed and "policy_version_mismatch" in failed.reasons


def test_snapshot_oracle_fails_closed_on_non_mapping_restore():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    for malformed in ("bad", 3, [], object()):
        result = StateAuditOracle().audit(snapshot, malformed)
        assert result.status == "failed"
        assert "restore_not_mapping" in result.reasons


def test_constraint_dual_restore_rejects_fractional_update_count():
    schema = _schema("constraint")
    state = ConstraintDualState.initialize(schema, primary_objective=0, learning_rate=0.1)
    with pytest.raises(ValueError, match="update_count"):
        ConstraintDualState.from_state_dict(schema, {**state.state_dict(), "update_count": 0.5})
    for invalid in ("1", 1.0, True):
        with pytest.raises(ValueError, match="update_count"):
            ConstraintDualState.from_state_dict(schema, {**state.state_dict(), "update_count": invalid})
    with pytest.raises(ValueError, match="primary_objective"):
        ConstraintDualState.from_state_dict(schema, {**state.state_dict(), "primary_objective": 0.5})


def test_snapshot_restore_does_not_require_future_branch_attestation():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    restored = {"boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1", "feature_map_hash": "fm-hash", "segment_rule_version": "seg-v1", "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session", "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng", "execution_budget_remaining": 6, "artifact_hashes": {}}
    assert StateAuditOracle().audit(snapshot, restored).passed


def test_snapshot_restore_checks_optional_captured_state_projections():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    restored = {"boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1", "feature_map_hash": "fm-hash", "feature_map_version": "fm", "segment_rule_version": "seg-v1", "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session", "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng", "execution_budget_remaining": 6, "artifact_hashes": {}, "action_mask": [False]}
    result = StateAuditOracle().audit(snapshot, restored)
    assert not result.passed and "action_mask_mismatch" in result.reasons


def test_snapshot_restore_invalid_cost_is_a_failed_audit_record():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    result = StateAuditOracle().audit(snapshot, None, cost="not-a-number")
    assert not result.passed
    assert result.cost == 0.0
    assert "invalid_restore_cost" in result.reasons


def test_fresh_branch_runner_attests_independent_rng_and_means_vectors():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    audit = StateAuditOracle().audit(snapshot, {"boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1", "feature_map_hash": "fm-hash", "segment_rule_version": "seg-v1", "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session", "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng", "execution_budget_remaining": 6, "artifact_hashes": {}, "rng_independent": True})
    schema = _schema()
    def branch(_, rng_id, remaining):
        return {"branch_id": f"b0:fresh:{rng_id}", "policy_version": "policy-v1", "rng_id": rng_id, "rng_independent": True, "execution_budget_remaining": remaining - 1, "cost": 1, "observations": (EndpointObservation("activity", "activity-hf", 0.5, cost=1), EndpointObservation("stability", "stability-hf", 50.0, cost=1))}
    results = FreshBranchRunner().run(snapshot, audit, branch_count=2, branch_fn=branch, schema=schema, policy_version="policy-v1")
    assert len({r.rng_id for r in results}) == 2
    assert all(r.status == "completed" for r in results)


def test_fresh_branch_runner_accepts_durable_restored_boundary_continuation():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    audit = RestoreAudit(snapshot.boundary_id, "passed", snapshot.digest, snapshot.digest, 0.0, evidence_kind="durable")
    schema = _schema()

    def branch(_, rng_id, remaining):
        return {
            "branch_id": f"b0:fresh:{rng_id}", "policy_version": "policy-v1", "rng_id": rng_id,
            "rng_independent": True, "execution_budget_remaining": remaining,
            "trajectory_complete": True,
            "continuation_provenance": {
                "continuation_mode": "restored_boundary",
                "prefix": "prefix",
                "sampling_config": {},
                "policy_version": "policy-v1",
            },
            "observations": (
                EndpointObservation("activity", "activity-hf", 0.5, evaluator_id="evaluator-activity", evaluator_version="hf-v1"),
                EndpointObservation("stability", "stability-hf", 50.0, evaluator_id="evaluator-stability", evaluator_version="hf-v1"),
            ),
        }

    results = FreshBranchRunner().run(
        snapshot, audit, branch_count=1, branch_fn=branch, schema=schema,
        policy_version="policy-v1", require_evaluator_provenance=True,
        require_continuation_provenance=True,
    )
    assert results[0].status == "completed"
    assert results[0].continuation_mode == "restored_boundary"
    assert results[0].research_eligible is True


def test_fresh_branch_runner_rejects_original_prompt_restart_as_research():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    audit = RestoreAudit(snapshot.boundary_id, "passed", snapshot.digest, snapshot.digest, 0.0, evidence_kind="durable")

    def branch(_, rng_id, remaining):
        return {
            "branch_id": f"b0:fresh:{rng_id}", "policy_version": "policy-v1", "rng_id": rng_id,
            "rng_independent": True, "execution_budget_remaining": remaining,
            "trajectory_complete": True,
            "continuation_provenance": {
                "continuation_mode": "original_prompt_restart",
                "prefix": "original prompt",
                "sampling_config": {"temperature": 0.7},
            },
            "observations": (
                EndpointObservation("activity", "activity-hf", 0.5),
                EndpointObservation("stability", "stability-hf", 50.0),
            ),
        }

    results = FreshBranchRunner().run(
        snapshot, audit, branch_count=1, branch_fn=branch, schema=_schema(),
        policy_version="policy-v1", require_continuation_provenance=True,
    )
    assert results[0].status == "failed"
    assert "continuation provenance" in (results[0].failure_reason or "")


def test_fresh_branch_runner_rejects_unattested_rng_independence():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    restored = {"boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1", "feature_map_hash": "fm-hash", "segment_rule_version": "seg-v1", "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session", "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng", "execution_budget_remaining": 6, "artifact_hashes": {}, "rng_independent": True}
    audit = StateAuditOracle().audit(snapshot, restored)
    schema = _schema()

    def branch(_, rng_id, remaining):
        return {"branch_id": f"b0:fresh:{rng_id}", "policy_version": "policy-v1", "rng_id": rng_id, "execution_budget_remaining": remaining, "observations": ()}

    result = FreshBranchRunner().run(snapshot, audit, branch_count=1, branch_fn=branch, schema=schema, policy_version="policy-v1")
    assert result[0].status == "failed" and result[0].failure_class == "branch_failure"


def test_fresh_branch_runner_requires_branch_identity_attestation():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    restored = {"boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1", "feature_map_hash": "fm-hash", "segment_rule_version": "seg-v1", "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session", "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng", "execution_budget_remaining": 6, "artifact_hashes": {}}
    audit = StateAuditOracle().audit(snapshot, restored)

    result = FreshBranchRunner().run(
        snapshot,
        audit,
        branch_count=1,
        branch_fn=lambda _snapshot, rng_id, remaining: {
            "policy_version": "policy-v1",
            "rng_id": rng_id,
            "rng_independent": True,
            "execution_budget_remaining": remaining,
            "observations": (),
        },
        schema=_schema(),
        policy_version="policy-v1",
    )
    assert result[0].status == "failed" and "branch identity" in (result[0].failure_reason or "")


def test_fresh_branch_runner_rejects_wrong_branch_identity():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    restored = {"boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1", "feature_map_hash": "fm-hash", "segment_rule_version": "seg-v1", "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session", "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng", "execution_budget_remaining": 6, "artifact_hashes": {}, "rng_independent": True}
    audit = StateAuditOracle().audit(snapshot, restored)

    def branch(_, rng_id, remaining):
        return {"branch_id": "wrong", "policy_version": "policy-v1", "rng_id": rng_id, "rng_independent": True, "execution_budget_remaining": remaining, "observations": ()}

    result = FreshBranchRunner().run(snapshot, audit, branch_count=1, branch_fn=branch, schema=_schema(), policy_version="policy-v1")
    assert result[0].status == "failed" and "branch identity" in (result[0].failure_reason or "")


def test_fresh_branch_runner_rejects_incomplete_endpoint_vector():
    snapshot = BoundarySnapshot("task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap")
    restored = {"boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1", "feature_map_hash": "fm-hash", "segment_rule_version": "seg-v1", "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session", "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng", "execution_budget_remaining": 6, "artifact_hashes": {}, "rng_independent": True}
    audit = StateAuditOracle().audit(snapshot, restored)
    schema = _schema()

    def branch(_, rng_id, remaining):
        return {"branch_id": f"b0:fresh:{rng_id}", "policy_version": "policy-v1", "rng_id": rng_id, "rng_independent": True, "execution_budget_remaining": remaining, "observations": (EndpointObservation("activity", "activity-hf", 0.5),)}

    result = FreshBranchRunner().run(snapshot, audit, branch_count=1, branch_fn=branch, schema=schema, policy_version="policy-v1")
    assert result[0].status == "failed" and result[0].failure_class == "branch_failure"
    assert "incomplete endpoint vector" in (result[0].failure_reason or "")


def test_mean_observations_does_not_shrink_branch_count_after_failure():
    schema = _schema()
    results = (
        # A failed branch has no valid HF sample.  The successful branch must
        # not be treated as n_b=1 when the requested branch count was two.
        BranchResult(
            "b:fresh:0", "rng0", "completed",
            (
                EndpointObservation("activity", "activity-hf", 0.8),
                EndpointObservation("stability", "stability-hf", 50.0),
            ), 5.0, 1.0,
        ),
        BranchResult(
            "b:fresh:1", "rng1", "failed", (), 6.0, 0.0, "branch_failure", "timeout",
        ),
    )
    values, valid, covariance = mean_observations(results, schema)
    assert values == (None, None)
    assert valid == (False, False)
    assert covariance.shape == (2, 2)


def test_mean_observations_keeps_one_endpoint_covariance_matrix_shape():
    schema = EndpointSchema(
        (
            EndpointSpec(
                "score", "score", "unit", hf_fidelity_id="score-hf",
                fidelities=(FidelitySpec("score-hf", "HF", "evaluator", "v1", 1.0, kind="gold"),),
            ),
        ),
        objective_mode="preference", weights=(1.0,),
    )
    results = tuple(
        BranchResult(f"b:fresh:{index}", f"rng{index}", "completed", (EndpointObservation("score", "score-hf", value),), 1.0, 0.0)
        for index, value in enumerate((0.0, 1.0))
    )
    values, valid, covariance = mean_observations(results, schema)
    assert values == (0.5,) and valid == (True,)
    assert covariance.shape == (1, 1)


def test_science_round_orchestrates_restore_query_correction_gradient_and_verifier():
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    actions = {
        "b0": BoundarySnapshot(
            "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
            "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap0"
        ),
        "b1": BoundarySnapshot(
            "task", "traj", "run", "session", "b1", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
            "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap1"
        ),
    }
    candidates = tuple(
        QueryCandidate(boundary_id, endpoint_id, 1.0, 1.0, 1.0, 1.0)
        for boundary_id in actions
        for endpoint_id in schema.endpoint_ids
    )

    def restore(snapshot):
        return {**snapshot.to_dict(), "snapshot_digest": snapshot.digest, "artifact_hashes": snapshot.workspace_hashes}

    def branch(snapshot, rng_id, remaining):
        return {
            "branch_id": f"{snapshot.boundary_id}:fresh:{rng_id}",
            "policy_version": "policy-v1",
            "rng_id": rng_id,
            "rng_independent": True,
            "execution_budget_remaining": remaining - 1.0,
            "cost": 1.0,
            "observations": (
                EndpointObservation("activity", "activity-hf", 0.5, evaluator_id="evaluator-activity", evaluator_version="hf-v1"),
                EndpointObservation("stability", "stability-hf", 50.0, evaluator_id="evaluator-stability", evaluator_version="hf-v1"),
            ),
        }

    result = run_science_round(
        schema=schema,
        feature_map=feature_map,
        policy_version="policy-v1",
        boundary_ids=("b0", "b1"),
        boundary_snapshots=actions,
        boundary_intervals=[(0, 2), (2, 4)],
        boundary_features=[[0.0], [1.0]],
        boundary_predictions=torch.tensor([[0.1, 20.0], [0.2, 30.0], [0.3, 40.0], [0.4, 50.0]]),
        terminal_values=[0.4, 50.0],
        segment_scores=torch.ones((3, 2)),
        candidates=candidates,
        query_budget=4.0,
        branch_fn=branch,
        restore_fn=restore,
        ledger=BudgetLedger(20.0),
        branch_count=2,
        planned_branch_cost=1.0,
        audit_cost=0.5,
        terminal_cost=1.0,
        terminal_evaluator_ids={"activity": "evaluator-activity", "stability": "evaluator-stability"},
        terminal_evaluator_versions={"activity": "hf-v1", "stability": "hf-v1"},
    )
    assert result.verification.passed
    assert result.corrected_boundary_values.shape == (4, 2)
    assert result.segment_gradient.shape == (2, 2)
    assert len(result.boundary_records) == 2
    assert result.boundary_records[0].branch_count["activity"] == 2
    assert result.ledger["spent"] > 0
    assert result.ledger["consumed"]["audit"] == pytest.approx(1.0)
    assert set(result.covariance_by_boundary) == {"b0", "b1"}
    assert result.audit["covariance_by_boundary"] == result.covariance_by_boundary
    assert set(result.audit["gradient_risk_by_query"]) == {
        f"{boundary_id}::{endpoint_id}"
        for boundary_id in actions
        for endpoint_id in schema.endpoint_ids
    }
    assert result.boundary_records[0].gradient_risk["activity"]["estimated_gradient_risk"] == pytest.approx(1.0)


@pytest.mark.parametrize("field,value", [("query_budget", True), ("terminal_cost", "1.0"), ("restore_cost", True), ("audit_cost", "0.1"), ("planned_branch_cost", True), ("branch_count", 1.5)])
def test_science_round_rejects_coercible_budget_and_branch_controls(field, value):
    kwargs = _science_round_contract_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match="invalid science-round"):
        run_science_round(**kwargs)


def _science_round_contract_kwargs():
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap0"
    )
    candidate = QueryCandidate("b0", "activity", 1.0, 1.0, 1.0, 1.0)
    kwargs = dict(
        schema=schema, feature_map=feature_map, policy_version="policy-v1", boundary_ids=("b0",),
        boundary_snapshots={"b0": snapshot}, boundary_intervals=[(0, 2)], boundary_features=[[0.0]],
        boundary_predictions=torch.tensor([[0.1, 20.0], [0.2, 30.0], [0.4, 50.0]]), terminal_values=[0.4, 50.0],
        segment_scores=torch.ones((2, 1)), candidates=(candidate,), query_budget=1.0,
        branch_fn=lambda *_args: {}, restore_fn=lambda value: value.to_dict(), ledger=BudgetLedger(10.0),
        planned_branch_cost=1.0, terminal_evaluator_ids={"activity": "evaluator-activity", "stability": "evaluator-stability"},
        terminal_evaluator_versions={"activity": "hf-v1", "stability": "hf-v1"},
    )
    return kwargs


@pytest.mark.parametrize("invalid_cost", [True, "0.0", None])
def test_science_round_records_failed_restore_without_fabricated_rng_or_observations(invalid_cost):
    kwargs = _science_round_contract_kwargs()
    calls = []
    def branch(*args):
        calls.append(args)
        raise AssertionError("failed restore must not start a branch")
    def restore(snapshot):
        if invalid_cost is None:
            raise RuntimeError("restore unavailable")
        return snapshot.to_dict(), invalid_cost
    kwargs.update(branch_fn=branch, restore_fn=restore)
    result = run_science_round(**kwargs)
    assert calls == []
    assert result.observations == {}
    record = result.boundary_records[0]
    assert record.branch_status["activity"] == "restore_failed"
    assert record.branch_rng_id["activity"] == ""
    assert not record.valid_observation["activity"]
    assert record.oof_residual is None
    assert record.failure_reason["activity"]
    assert result.ledger["reserved"].get("branch", 0.0) == 0.0
    assert not result.verification.passed


def test_science_round_keeps_selected_endpoint_when_other_vector_dimension_is_missing():
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap0"
    )
    # Only activity is a candidate.  The branch still carries the complete
    # vector envelope, but stability is an explicitly failed assay.  The
    # selected activity observation must remain usable for its own propensity
    # correction instead of being discarded by an all-endpoints gate.
    candidate = QueryCandidate("b0", "activity", 1.0, 1.0, 1.0, 1.0)

    def restore(value):
        return {**value.to_dict(), "snapshot_digest": value.digest, "artifact_hashes": value.workspace_hashes}

    def branch(value, rng_id, remaining):
        return {
            "branch_id": f"b0:fresh:{rng_id}",
            "policy_version": "policy-v1",
            "rng_id": rng_id,
            "rng_independent": True,
            "execution_budget_remaining": remaining,
            "cost": 0.1,
            "observations": (
                EndpointObservation("activity", "activity-hf", 0.8, evaluator_id="evaluator-activity", evaluator_version="hf-v1"),
                EndpointObservation(
                    "stability", "stability-hf", None, status="assay_failed",
                    evaluator_id="evaluator-stability", evaluator_version="hf-v1",
                ),
            ),
        }

    result = run_science_round(
        schema=schema,
        feature_map=feature_map,
        policy_version="policy-v1",
        boundary_ids=("b0",),
        boundary_snapshots={"b0": snapshot},
        boundary_intervals=[(0, 2)],
        boundary_features=[[0.0]],
        boundary_predictions=torch.tensor([[0.1, 20.0], [0.2, 30.0], [0.4, 50.0]]),
        terminal_values=[0.4, 50.0],
        segment_scores=torch.ones((2, 1)),
        candidates=(candidate,),
        query_budget=1.0,
        branch_fn=branch,
        restore_fn=restore,
        ledger=BudgetLedger(10.0),
        planned_branch_cost=0.1,
        terminal_evaluator_ids={"activity": "evaluator-activity", "stability": "evaluator-stability"},
        terminal_evaluator_versions={"activity": "hf-v1", "stability": "hf-v1"},
    )
    assert "b0::activity" in result.observations
    assert result.observations["b0::activity"].value == pytest.approx(0.8)
    assert "b0::stability" not in result.observations
    assert result.boundary_records[0].valid_observation["activity"] is True


def test_science_round_requires_complete_candidate_coverage_for_positive_floor():
    kwargs = _science_round_contract_kwargs()
    kwargs["min_inclusion_probability"] = 0.1
    with pytest.raises(ValueError, match="every internal endpoint boundary"):
        run_science_round(**kwargs)


def test_science_round_rejects_mixed_branch_counts_within_boundary():
    kwargs = _science_round_contract_kwargs()
    kwargs["candidates"] = (
        QueryCandidate("b0", "activity", 1.0, 1.0, 1.0, 1.0, branch_count=2),
        QueryCandidate("b0", "stability", 1.0, 1.0, 1.0, 1.0, branch_count=3),
    )
    kwargs["query_budget"] = 2.0
    with pytest.raises(ValueError, match="share branch_count"):
        run_science_round(**kwargs)


def test_science_round_rejects_external_branch_count_contract_mismatch():
    kwargs = _science_round_contract_kwargs()
    kwargs["branch_counts_by_boundary"] = {"b0": 2}
    with pytest.raises(ValueError, match="do not match branch_counts_by_boundary"):
        run_science_round(**kwargs)


def test_science_round_reserves_budget_before_callbacks():
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap0"
    )
    candidate = QueryCandidate("b0", "activity", 1.0, 1.0, 1.0, 1.0)
    restore_calls = []
    branch_calls = []

    def restore(value):
        restore_calls.append(value.boundary_id)
        return {**value.to_dict(), "snapshot_digest": value.digest, "artifact_hashes": value.workspace_hashes}

    def branch(value, rng_id, remaining):
        branch_calls.append(value.boundary_id)
        return {}

    with pytest.raises(ValueError, match="budget exceeded"):
        run_science_round(
            schema=schema,
            feature_map=feature_map,
            policy_version="policy-v1",
            boundary_ids=("b0",),
            boundary_snapshots={"b0": snapshot},
            boundary_intervals=[(0, 2)],
            boundary_features=[[0.0]],
            boundary_predictions=torch.tensor([[0.1, 20.0], [0.2, 30.0], [0.4, 50.0]]),
            terminal_values=[0.4, 50.0],
            segment_scores=torch.ones((2, 1)),
            candidates=(candidate,),
            query_budget=1.0,
            branch_fn=branch,
            restore_fn=restore,
            ledger=BudgetLedger(2.0),
            planned_branch_cost=1.0,
            terminal_cost=1.0,
            terminal_evaluator_ids={"activity": "evaluator-activity", "stability": "evaluator-stability"},
            terminal_evaluator_versions={"activity": "hf-v1", "stability": "hf-v1"},
        )
    assert restore_calls == []
    assert branch_calls == []


def test_science_round_marks_branch_cost_overrun_failed_without_observation():
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap0"
    )
    candidate = QueryCandidate("b0", "activity", 1.0, 1.0, 1.0, 1.0)

    def restore(value):
        return {**value.to_dict(), "snapshot_digest": value.digest, "artifact_hashes": value.workspace_hashes}

    def branch(value, rng_id, remaining):
        return {
            "branch_id": f"b0:fresh:{rng_id}",
            "policy_version": "policy-v1",
            "rng_id": rng_id,
            "rng_independent": True,
            "execution_budget_remaining": remaining,
            "cost": 2.0,
            "observations": (
                EndpointObservation("activity", "activity-hf", 0.8, evaluator_id="evaluator-activity", evaluator_version="hf-v1"),
                EndpointObservation("stability", "stability-hf", 50.0, evaluator_id="evaluator-stability", evaluator_version="hf-v1"),
            ),
        }

    result = run_science_round(
        schema=schema,
        feature_map=feature_map,
        policy_version="policy-v1",
        boundary_ids=("b0",),
        boundary_snapshots={"b0": snapshot},
        boundary_intervals=[(0, 2)],
        boundary_features=[[0.0]],
        boundary_predictions=torch.tensor([[0.1, 20.0], [0.2, 30.0], [0.4, 50.0]]),
        terminal_values=[0.4, 50.0],
        segment_scores=torch.ones((2, 1)),
        candidates=(candidate,),
        query_budget=1.0,
        branch_fn=branch,
        restore_fn=restore,
        ledger=BudgetLedger(10.0),
        planned_branch_cost=1.0,
        terminal_evaluator_ids={"activity": "evaluator-activity", "stability": "evaluator-stability"},
        terminal_evaluator_versions={"activity": "hf-v1", "stability": "hf-v1"},
    )
    assert result.observations == {}
    assert result.boundary_records[0].branch_status["activity"] == "failed"
    assert result.boundary_records[0].failure_reason["activity"] == "branch_cost_exceeds_planned_reservation"
    assert result.verification.passed is False
    assert result.ledger["consumed"]["branch"] == pytest.approx(2.0)


def test_budget_ledger_rejects_overcommit_and_tracks_all_categories():
    ledger = BudgetLedger(10)
    ledger.reserve("query", 4, reference="b0")
    ledger.commit("query", 3, reference="b0", planned=4)
    ledger.record("restore", 2)
    # One unit remains reserved for the planned query, so only four units are
    # currently free even though five have been committed.
    assert ledger.spent == 5 and ledger.remaining == 4
    with pytest.raises(ValueError, match="budget exceeded"):
        ledger.record("branch", 6)
    assert ledger.fingerprint


def test_budget_ledger_does_not_overcommit_after_releasing_partial_reservation():
    ledger = BudgetLedger(10)
    ledger.reserve("query", 4)
    with pytest.raises(ValueError, match="budget exceeded"):
        ledger.commit("query", 11)
    assert ledger.spent == 0 and ledger.reserved_total == 4


def test_segment_score_gradient_detaches_values_and_preserves_endpoint_axis():
    scores = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    values = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [5.0, 6.0]]], requires_grad=True)
    grad = segment_score_gradient(scores, values)
    assert grad.shape == (1, 2, 2)
    torch.testing.assert_close(grad, torch.tensor([[[11.0, 16.0], [11.0, 16.0]]]))
    assert not grad.requires_grad


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=11, hidden_size=5):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return SimpleNamespace(logits=self.proj(self.embedding(input_ids)))


def test_parameter_score_gradients_matches_direct_autograd_and_adds_segments():
    torch.manual_seed(4)
    model = _TinyCausalLM()
    model.train()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    scores = parameter_score_gradients(
        model, ids, response_start=2, segment_ranges=[(0, 2), (2, 4)],
    )
    assert scores.shape == (2, sum(p.numel() for p in model.parameters()))
    assert model.training is True
    logits = model(ids).logits
    token_lp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)[0]
    direct = []
    params = list(model.parameters())
    for positions in ((1, 2), (3, 4)):
        grads = torch.autograd.grad(token_lp[list(positions)].sum(), params, retain_graph=True)
        direct.append(torch.cat([g.reshape(-1) for g in grads]))
    torch.testing.assert_close(scores, torch.stack(direct), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        scores.sum(0),
        parameter_score_gradients(model, ids, response_start=2, segment_ranges=[(0, 4)])[0],
    )


def test_parameter_score_gradients_action_mask_subset_and_parameter_subset():
    torch.manual_seed(8)
    model = _TinyCausalLM()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    selected = parameter_score_gradients(
        model, ids, response_start=1, segment_ranges=[(0, 4)],
        action_mask=torch.tensor([1, 0, 1, 0]), parameter_names=["proj.bias"],
    )
    assert selected.shape == (1, model.proj.bias.numel())
    expected = parameter_score_gradients(
        model, ids, response_start=1, segment_ranges=[(0, 1), (2, 3)],
        parameter_names=["proj.bias"],
    ).sum(0, keepdim=True)
    torch.testing.assert_close(selected, expected)
    with pytest.raises(ValueError, match="only 0/1"):
        parameter_score_gradients(model, ids, response_start=1, segment_ranges=[(0, 4)], action_mask=[1, 0.5, 1, 0])
    with pytest.raises(ValueError, match="only 0/1"):
        parameter_score_gradients(model, ids, response_start=1, segment_ranges=[(0, 4)], attention_mask=[[1, 1, 1, 1, 2]])
    with pytest.raises(ValueError, match="bounds must be integers"):
        parameter_score_gradients(model, ids, response_start=1, segment_ranges=[(0.0, 2)])


def test_parameter_score_gradients_is_not_average_log_probability():
    torch.manual_seed(9)
    model = _TinyCausalLM()
    ids = torch.tensor([[1, 2, 3, 4]])
    score = parameter_score_gradients(model, ids, response_start=1, segment_ranges=[(0, 3)])
    logits = model(ids).logits
    lp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:, None]).squeeze()
    assert score.shape[1] == sum(p.numel() for p in model.parameters())
    assert not torch.allclose(score.flatten()[: lp.numel()], lp.detach() / 3)


def test_parameter_score_gradients_restores_mixed_child_training_modes():
    model = _TinyCausalLM()
    model.train()
    model.embedding.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    parameter_score_gradients(model, ids, response_start=1, segment_ranges=[(0, 3)])
    assert model.training is True
    assert model.embedding.training is False
    assert model.proj.training is True


def test_score_gradient_sketcher_is_deterministic_and_validates_projection():
    torch.manual_seed(12)
    model = _TinyCausalLM()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    exact = parameter_score_gradients(
        model, ids, response_start=2, segment_ranges=[(0, 2), (2, 4)],
        parameter_names=["proj.weight", "proj.bias"],
    )
    first = ScoreGradientSketcher(3, seed=19, parameter_prefix="proj", checkpoint_hash="actor-hash").fit(model)
    second = ScoreGradientSketcher(3, seed=19, parameter_prefix="proj", checkpoint_hash="actor-hash").fit(model)
    projected = first.project(exact)
    torch.testing.assert_close(projected, second.project(exact))
    assert first.parameter_names is None
    assert first._fitted_parameter_names == ("proj.weight", "proj.bias")
    report = first.validate(exact)
    assert report["validated"] is True
    assert report["parameter_dimension"] == exact.shape[1]
    assert report["projection_dimension"] == 3
    scores = first.score(model, ids, response_start=2, segment_ranges=[(0, 2), (2, 4)])
    torch.testing.assert_close(scores, projected)


def test_score_gradient_sketcher_state_roundtrip_and_rejects_invalid_payloads():
    model = _TinyCausalLM()
    sketch = ScoreGradientSketcher(2, seed=5, parameter_names=["proj.bias"]).fit(model)
    restored = ScoreGradientSketcher.from_state_dict(sketch.state_dict())
    assert restored.parameter_names == ("proj.bias",)
    assert restored._fitted_parameter_names == ("proj.bias",)
    torch.testing.assert_close(restored.project(torch.ones(2, model.proj.bias.numel())), sketch.project(torch.ones(2, model.proj.bias.numel())))
    bad = sketch.state_dict()
    bad["projection"] = [[float("nan")]]
    with pytest.raises(ValueError, match="invalid score sketch projection"):
        ScoreGradientSketcher.from_state_dict(bad)
    with pytest.raises(ValueError, match="unsupported score sketch checkpoint"):
        ScoreGradientSketcher.from_state_dict({"version": "old"})


def test_pareto_preference_and_constraint_directions_are_explicit():
    gradients = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    _, alpha = pareto_min_norm_direction(gradients)
    assert torch.all(alpha >= 0) and torch.isclose(alpha.sum(), torch.tensor(1.0))
    torch.testing.assert_close(preference_direction(gradients, [0.25, 0.75]), torch.tensor([0.25, 1.5]))
    constrained = constrained_direction(gradients, _schema("constraint"), [0.2, 30], penalty=2, allow_dual_bootstrap=True)
    assert constrained[1] > 0
    upper_endpoint = EndpointSpec(
        "toxicity", "toxicity", "a.u.", hf_fidelity_id="toxicity-hf", constraint="upper", threshold=1.0,
        fidelities=(FidelitySpec("toxicity-hf", "HF", "tox", "v1", 1.0, kind="gold"),),
    )
    upper_schema = EndpointSchema(
        (_schema("constraint").endpoints[0], upper_endpoint),
        objective_mode="constraint",
        primary_objective="activity",
    )
    direction = constrained_direction(torch.tensor([[0.0], [2.0]]), upper_schema, [0.2, 2.0], penalty=1.0, allow_dual_bootstrap=True)
    assert direction[0] < 0
    dual = projected_dual_ascent(torch.zeros(2), upper_schema, [0.2, 2.0], learning_rate=0.5)
    assert torch.isclose(dual[1], torch.tensor(0.5))
    values = torch.tensor([[1.0, 2.0]])
    scalar = scalarize_boundary_values(values, upper_schema, mode="constraint", estimates=[0.2, 2.0], penalty=1.0, allow_dual_bootstrap=True)
    assert scalar.item() < 1.0
    with pytest.raises(ValueError, match="persistent multipliers"):
        constrained_direction(gradients, _schema("constraint"), [0.2, 30])


def test_constraint_scalarization_does_not_mutate_dual_state():
    schema = _schema("constraint")
    multipliers = torch.tensor([0.0, 2.0])
    scalarize_boundary_values(
        torch.tensor([[1.0, 50.0]]),
        schema,
        mode="constraint",
        multipliers=multipliers,
    )
    torch.testing.assert_close(multipliers, torch.tensor([0.0, 2.0]))


def test_constraint_dual_state_is_persistent_and_schema_bound():
    schema = _schema("constraint")
    state = ConstraintDualState.initialize(schema, primary_objective=0, learning_rate=0.5)
    updated = state.update(schema, [0.2, 30.0])
    assert torch.isclose(updated[1], torch.tensor(5.0, dtype=updated.dtype))
    payload = state.state_dict()
    restored = ConstraintDualState.from_state_dict(schema, payload)
    assert restored.update_count == 1
    assert torch.equal(restored.multipliers, state.multipliers)
    with pytest.raises(ValueError, match="fingerprint"):
        ConstraintDualState.from_state_dict(_schema(), payload)
    with pytest.raises(ValueError, match="dual estimates"):
        state.update(schema, [0.2, float("nan")])


def test_projected_dual_ascent_keeps_float_dtype_for_python_sequence():
    schema = _schema("constraint")
    updated = projected_dual_ascent([0, 0], schema, [0.2, 30.0], learning_rate=0.5)
    assert updated.dtype.is_floating_point
    assert torch.isclose(updated[1], torch.tensor(5.0, dtype=updated.dtype))


def test_constraint_adapter_requires_persistent_duals_by_default():
    from verl.trainer.ppo.polyseg_sci import compute_polyseg_science_advantage_registered

    algo = AlgoConfig(
        adv_estimator="polyseg_sci",
        polyseg=PolySegAlgorithmConfig(
            science=PolySegScienceConfig(objective_mode="constraint", primary_objective=0)
        ),
    )
    batch = {
        "science_boundary_values": torch.zeros(1, 2, 2),
        "segment_ids": torch.zeros(1, 1, dtype=torch.long),
        "science_endpoint_estimates": torch.zeros(1, 2),
    }
    with pytest.raises(ValueError, match="persistent science_constraint_multipliers"):
        compute_polyseg_science_advantage_registered(
            token_level_rewards=None,
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            config=algo,
            batch=batch,
        )


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_science_objective_boolean_controls_are_not_coerced(value):
    schema = _schema("constraint")
    values = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="allow_dual_bootstrap"):
        scalarize_boundary_values(
            values,
            schema,
            mode="constraint",
            estimates=[0.0, 0.0],
            allow_dual_bootstrap=value,
        )

    with pytest.raises(ValueError, match="allow_dual_bootstrap"):
        constrained_direction(
            torch.zeros(2, 1),
            schema,
            [0.0, 0.0],
            allow_dual_bootstrap=value,
        )


def test_registered_science_objective_rejects_fractional_primary_and_string_bootstrap():
    algo = AlgoConfig(
        adv_estimator="polyseg_sci",
        polyseg=PolySegAlgorithmConfig(
            science=PolySegScienceConfig(objective_mode="constraint", primary_objective=0)
        ),
    )
    batch = {
        "science_boundary_values": torch.zeros(1, 2, 2),
        "segment_ids": torch.zeros(1, 1, dtype=torch.long),
        "science_endpoint_estimates": torch.zeros(1, 2),
    }
    with pytest.raises(ValueError, match="allow_dual_bootstrap"):
        compute_polyseg_science_advantage_registered(
            token_level_rewards=None,
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            config={"polyseg": {"science": {"objective_mode": "constraint", "primary_objective": 0, "allow_dual_bootstrap": "false"}}},
            batch=batch,
        )
    with pytest.raises(ValueError, match="primary_objective"):
        compute_polyseg_science_advantage_registered(
            token_level_rewards=None,
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            config={"polyseg": {"science": {"objective_mode": "constraint", "primary_objective": 0.5, "allow_dual_bootstrap": True}}},
            batch=batch,
        )


def test_failed_observation_with_value_is_rejected_at_construction():
    with pytest.raises(ValueError, match="null value"):
        EndpointObservation("activity", "activity-hf", 0.5, status="branch_failed", failure_class="branch_failure")


def test_endpoint_failure_status_has_matching_failure_class():
    with pytest.raises(ValueError, match="failure_class='branch_failure'"):
        EndpointObservation("activity", "activity-hf", None, status="branch_failed", failure_class="timeout")


def test_vector_correction_covariance_uses_joint_endpoint_propensities():
    covariance = residual_correction_covariance(
        torch.tensor([0.2, -0.1]),
        torch.tensor([[1.0, 0.3], [0.3, 2.0]]),
        [0.5, 0.25],
        2,
        pair_inclusion_probabilities=[[0.5, 0.1], [0.1, 0.25]],
    )
    assert covariance.shape == (2, 2)
    expected_cross = (0.1 / (0.5 * 0.25) - 1) * 0.2 * -0.1 + (0.1 / (0.5 * 0.25)) * 0.3 / 2
    assert covariance[0, 1].item() == pytest.approx(expected_cross)
    with pytest.raises(ValueError, match="pair inclusion probabilities"):
        residual_correction_covariance(torch.tensor([0.2, -0.1]), torch.eye(2), [0.5, 0.25], 2)

    with pytest.raises(ValueError, match="diagonal must equal"):
        residual_correction_covariance(
            torch.tensor([0.2, -0.1]),
            torch.eye(2),
            [0.5, 0.25],
            2,
            pair_inclusion_probabilities=[[0.4, 0.1], [0.1, 0.25]],
        )
    with pytest.raises(ValueError, match="symmetric"):
        residual_correction_covariance(
            torch.tensor([0.2, -0.1]),
            torch.eye(2),
            [0.5, 0.25],
            2,
            pair_inclusion_probabilities=[[0.5, 0.1], [0.2, 0.25]],
        )
    with pytest.raises(ValueError, match="must not exceed"):
        residual_correction_covariance(
            torch.tensor([0.2, -0.1]),
            torch.eye(2),
            [0.5, 0.25],
            2,
            pair_inclusion_probabilities=[[0.5, 0.3], [0.3, 0.25]],
        )
    with pytest.raises(ValueError, match="Frechet lower bound"):
        residual_correction_covariance(
            torch.tensor([0.2, -0.1]),
            torch.eye(2),
            [0.8, 0.7],
            2,
            pair_inclusion_probabilities=[[0.8, 0.4], [0.4, 0.7]],
        )

    unequal = residual_correction_covariance(
        torch.tensor([0.2, -0.1]), torch.tensor([[1.0, 0.3], [0.3, 2.0]]),
        [0.5, 0.25], 2,
        pair_inclusion_probabilities=[[0.5, 0.1], [0.1, 0.25]],
        branch_counts=[2, 4], paired_branch_counts=[[2, 2], [2, 4]],
    )
    expected_unequal_cross = (0.1 / (0.5 * 0.25) - 1) * 0.2 * -0.1 + (0.1 / (0.5 * 0.25)) * 0.3 * 2 / (2 * 4)
    assert unequal[0, 1].item() == pytest.approx(expected_unequal_cross)
    mutually_exclusive = residual_correction_covariance(
        torch.tensor([0.2, -0.1]), torch.eye(2), [0.5, 0.5], 2,
        pair_inclusion_probabilities=[[0.5, 0.0], [0.0, 0.5]],
    )
    assert mutually_exclusive[0, 1].item() == pytest.approx(-0.2 * -0.1)


def test_vector_correction_covariance_rejects_missing_overlap_for_unequal_counts():
    with pytest.raises(ValueError, match="unequal branch_counts"):
        residual_correction_covariance(
            torch.tensor([0.2, -0.1]), torch.eye(2), [0.5, 0.25], 2,
            pair_inclusion_probabilities=[[0.5, 0.1], [0.1, 0.25]], branch_counts=[2, 4],
        )


@pytest.mark.parametrize("exclusive", [False, True])
def test_covariance_matches_exact_enumeration_of_shared_branch_noise(exclusive):
    from itertools import product

    error = torch.tensor([0.3, -0.2], dtype=torch.float64)
    selections = [(0, 1), (1, 0)] if exclusive else list(product((0, 1), repeat=2))
    corrected_samples = []
    for selection in selections:
        for z1, z2 in product((-1.0, 1.0), repeat=2):
            # Endpoint 1 has one replicate. Endpoint 2 averages two;
            # only z1 is shared, and each underlying pair has covariance 1.
            means = torch.tensor([z1, (z1 + z2) / 2], dtype=torch.float64)
            selected = torch.tensor(selection, dtype=torch.float64)
            corrected_samples.append(error + selected / 0.5 * (means - error))
    samples = torch.stack(corrected_samples)
    centered = samples - samples.mean(0)
    exact_covariance = centered.T @ centered / len(samples)
    joint = 0.0 if exclusive else 0.25
    actual = residual_correction_covariance(
        error, torch.ones(2, 2), [0.5, 0.5], 1,
        pair_inclusion_probabilities=[[0.5, joint], [joint, 0.5]],
        branch_counts=torch.tensor([1, 2]), paired_branch_counts=[[1, 1], [1, 2]],
    )
    torch.testing.assert_close(actual, exact_covariance, rtol=0, atol=1e-12)


def test_vector_correction_covariance_rejects_fractional_or_boolean_branch_counts():
    with pytest.raises(ValueError, match="branch_count must be a positive integer"):
        residual_correction_covariance(torch.tensor([0.2, -0.1]), torch.eye(2), 0.5, 1.5)
    with pytest.raises(ValueError, match="branch_count must be a positive integer"):
        residual_correction_covariance(torch.tensor([0.2, -0.1]), torch.eye(2), 0.5, True)
    with pytest.raises(ValueError, match="branch_counts must contain positive integers"):
        residual_correction_covariance(
            torch.tensor([0.2, -0.1]), torch.eye(2), [0.5, 0.25], 2,
            pair_inclusion_probabilities=[[0.5, 0.1], [0.1, 0.25]], branch_counts=[2, 1.5],
        )


def test_science_round_validates_response_model_and_propensity_before_callbacks():
    for override in (
        {"response_model_validated": "false"},
        {"response_probabilities": {"b0::activity": "1.0"}},
    ):
        kwargs = _science_round_contract_kwargs()
        calls = []
        kwargs.update(
            override,
            restore_fn=lambda _snapshot: calls.append("restore"),
            branch_fn=lambda *_args: calls.append("branch"),
        )
        with pytest.raises(ValueError):
            run_science_round(**kwargs)
        assert calls == []
        assert kwargs["ledger"].spent == 0
        assert kwargs["ledger"].reserved_total == 0


def test_query_decision_pair_propensities_match_marginals():
    candidates = (
        QueryCandidate("b0", "activity", 1.0, 1.0, 1.0, 1.0),
        QueryCandidate("b1", "activity", 1.0, 1.0, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="consistent with endpoint marginals"):
        QueryDecision(
            candidates,
            ("b0::activity",),
            {"b0::activity": 0.8, "b1::activity": 0.7},
            {"b0::activity::b1::activity": 0.4},
            "uniform", False, 2.0, 1.0, 0,
        )


def test_vector_covariance_and_calibration_report_dependence_instead_of_dropping_it():
    covariance = residual_correction_covariance(torch.tensor([0.2, -0.1]), torch.tensor([[1.0, 0.3], [0.3, 2.0]]), 0.5, 2)
    assert covariance.shape == (2, 2)
    assert covariance[0, 1] != 0
    report = MultiFidelityCalibrator().diagnostics([0.1, 0.2, 0.3], [0.2, 0.4, 0.6])
    assert report["n"] == 3 and report["calibration_slope"] > 0


def test_calibration_reports_heldout_extrapolation_and_domain_shift():
    report = MultiFidelityCalibrator().diagnostics(
        [0.0, 0.5, 1.0],
        [0.1, 0.6, 0.9],
        heldout_low_fidelity=[1.5, 0.25],
        heldout_high_fidelity=[1.3, 0.4],
        domain_labels=["a", "a", "b"],
        heldout_domain_labels=["c", "c"],
    )
    assert report["heldout_n"] == 2
    assert report["extrapolation_fraction"] == pytest.approx(0.5)
    assert report["domain_shift_available"] is True
    assert report["domain_shift_mean_delta"] == pytest.approx(1.0)


def test_vector_calibration_reports_endpoint_metrics_and_residual_dependence():
    report = MultiFidelityCalibrator().diagnostics_vector(
        [[0.0, 1.0], [0.5, 1.5], [1.0, 2.0], [0.25, 1.25]],
        [[0.1, 1.2], [0.6, 1.4], [0.9, 2.1], [0.3, 1.3]],
        endpoint_ids=["activity", "stability"],
    )
    assert report["endpoint_dimension"] == 2
    assert set(report["endpoints"]) == {"activity", "stability"}
    assert report["pair_counts"] == [[4, 4], [4, 4]]
    assert report["residual_covariance"][0][1] is not None
    assert report["residual_correlation"][0][1] is not None


def test_multifidelity_calibrator_rejects_coercible_controls_and_masks():
    with pytest.raises(ValueError, match="ridge_lambda"):
        MultiFidelityCalibrator(ridge_lambda=True)
    calibrator = MultiFidelityCalibrator()
    with pytest.raises(ValueError, match="interval"):
        calibrator.diagnostics([0.1, 0.2], [0.2, 0.4], interval="1.96")
    with pytest.raises(ValueError, match="valid_mask must contain booleans"):
        calibrator.diagnostics([0.1, 0.2], [0.2, 0.4], valid_mask=[1, 0])
    with pytest.raises(ValueError, match="valid_mask must contain booleans"):
        calibrator.diagnostics_vector([[0.1], [0.2]], [[0.2], [0.4]], valid_mask=[[1], [0]])


def test_bootstrap_resamples_complete_trajectory_groups():
    report = trajectory_bootstrap_ci([0.0, 1.0, 0.5, 0.5], ["t0", "t0", "t1", "t1"], repetitions=100, seed=4)
    assert report["groups"] == 2 and report["lower"] <= report["estimate"] <= report["upper"]


def test_ragged_feedback_width_reaches_real_actor_ppo_loss(monkeypatch):
    """The production ppo_loss path accepts a 15-wide microbatch under m=16."""
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    data = TensorDict(
        {
            "response_mask": torch.ones(1, 3, dtype=torch.bool),
            "old_log_probs": torch.zeros(1, 3),
            "advantages": torch.ones(1, 3),
            "segment_ids": torch.tensor([[0, 1, 2]]),
            "segment_feedback_mask": torch.ones(1, 15, dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, global_batch_size=1, global_macro_batch_size=3)
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, polyseg_num_segments=16, policy_loss=PolicyLossConfig(loss_mode="polyseg"))
    log_probs = torch.zeros(1, 3, requires_grad=True)
    loss, _ = ppo_loss(config, {"log_probs": log_probs}, data)
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(torch.autograd.grad(loss, log_probs)[0], -torch.ones_like(log_probs) / 3)


def test_real_actor_loss_uses_global_macro_denominator(monkeypatch):
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    data = TensorDict(
        {
            "response_mask": torch.ones(1, 2, dtype=torch.bool),
            "old_log_probs": torch.zeros(1, 2),
            "advantages": torch.ones(1, 2),
            "segment_ids": torch.tensor([[0, 1]]),
            "segment_feedback_mask": torch.ones(1, 15, dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=2, global_batch_size=2, global_macro_batch_size=4)
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, polyseg_num_segments=16, policy_loss=PolicyLossConfig(loss_mode="polyseg"))
    loss, _ = ppo_loss(config, {"log_probs": torch.zeros(1, 2, requires_grad=True)}, data)
    torch.testing.assert_close(loss, torch.tensor(-0.5))


def test_science_config_is_explicit_and_validated():
    cfg = PolySegScienceConfig(enabled=True, objective_mode="pareto", query_mode="adaptive", hard_budget=True, train_budget=12, branch_count=2)
    assert cfg.enabled and cfg.objective_mode == "pareto" and cfg.branch_count == 2
    assert cfg.representation_source == "frozen_hidden_projection"
    assert cfg.cross_fit_by == "trajectory" and cfg.require_full_state_contract
    with pytest.raises(ValueError, match="budget_mode"):
        PolySegScienceConfig(budget_mode="unbounded")
    with pytest.raises(ValueError, match="action_tokens"):
        PolySegScienceConfig(min_action_tokens=20, max_action_tokens=4)
    with pytest.raises(ValueError, match="objective_mode"):
        PolySegScienceConfig(objective_mode="posthoc")
    assert PolySegScienceConfig(execution_owner="driver").execution_owner == "driver"
    with pytest.raises(ValueError, match="execution_owner"):
        PolySegScienceConfig(execution_owner="ray")


def test_legacy_policy_update_alias_resolves_to_effective_science_mode():
    config = AlgoConfig(
        polyseg=PolySegAlgorithmConfig(
            science=PolySegScienceConfig(policy_update="constraint", primary_objective=0)
        )
    )
    from verl.trainer.ppo.polyseg_sci import _science_objective_mode

    assert _science_objective_mode(config) == "constraint"


def test_agent_loop_manager_initializes_and_restores_schema_bound_duals(tmp_path, monkeypatch):
    schema = _schema("constraint")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema.to_dict()), encoding="utf-8")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {}, "rollout": {"agent": {"num_workers": 0}}},
            "algorithm": {
                "adv_estimator": "polyseg_sci",
                "polyseg": {
                    "science": {
                        "enabled": True,
                        "objective_mode": "constraint",
                        "endpoint_schema_path": str(schema_path),
                        "endpoint_schema_hash": schema.fingerprint,
                        "primary_objective": 0,
                        "dual_learning_rate": 0.5,
                    }
                },
            },
        }
    )
    monkeypatch.setattr(AgentLoopManager, "agent_loop_workers_class", object(), raising=False)
    manager = AgentLoopManager(config, llm_client=None)
    assert manager.science_dual_state is not None
    assert manager.science_endpoint_schema.fingerprint == schema.fingerprint
    prompts = DataProto.from_single_dict({"index": torch.arange(2)})
    manager._attach_science_objective_state(prompts)
    assert prompts.non_tensor_batch["science_constraint_multipliers"].shape == (2, schema.dimension)
    assert torch.allclose(
        torch.as_tensor(prompts.non_tensor_batch["science_constraint_multipliers"]),
        torch.zeros(2, schema.dimension),
    )
    updated = manager.update_science_dual_state([0.4, 20.0])
    assert updated.shape == (schema.dimension,)
    payload = manager.science_dual_state.state_dict()
    restored = manager.set_science_dual_state(payload)
    assert restored.update_count == 1
    torch.testing.assert_close(restored.multipliers, updated)


def test_agent_loop_manager_loads_configured_fidelity_feature_map(tmp_path, monkeypatch):
    schema = _schema("preference")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema.to_dict()), encoding="utf-8")
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    feature_path = tmp_path / "feature_map.json"
    feature_path.write_text(json.dumps(feature_map.to_dict()), encoding="utf-8")
    slots = ("activity::activity-lf", "stability::stability-lf")
    fidelity_map = FidelityFeatureMapContract.fit(
        [[0.1, 20.0], [0.8, 60.0]],
        [[True, True], [True, True]],
        slots,
    )
    fidelity_path = tmp_path / "fidelity_feature_map.json"
    fidelity_path.write_text(json.dumps(fidelity_map.to_dict()), encoding="utf-8")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {}, "rollout": {"agent": {"num_workers": 0}}},
            "algorithm": {
                "adv_estimator": "polyseg_sci",
                "polyseg": {
                    "science": {
                        "enabled": True,
                        "objective_mode": "preference",
                        "endpoint_schema_path": str(schema_path),
                        "endpoint_schema_hash": schema.fingerprint,
                        "feature_map_path": str(feature_path),
                        "feature_map_hash": feature_map.fingerprint,
                        "fidelity_feature_map_path": str(fidelity_path),
                        "fidelity_feature_map_hash": fidelity_map.fingerprint,
                        "train_budget": 10.0,
                    }
                },
            },
        }
    )
    monkeypatch.setattr(AgentLoopManager, "agent_loop_workers_class", object(), raising=False)
    manager = AgentLoopManager(config, llm_client=None)
    assert manager.science_checkpoint_state is not None
    assert manager.science_checkpoint_state.fidelity_feature_map is not None
    assert manager.science_checkpoint_state.fidelity_feature_map.fingerprint == fidelity_map.fingerprint

    bad_config = OmegaConf.create(config)
    bad_config.algorithm.polyseg.science.fidelity_feature_map_hash = "0" * 64
    with pytest.raises(ValueError, match="fidelity_feature_map_hash"):
        AgentLoopManager(bad_config, llm_client=None)


def test_science_advantage_is_a_fail_closed_vector_to_token_adapter():
    values = torch.tensor(
        [[[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]]], dtype=torch.float32
    )
    token_adv, returns = compute_polyseg_science_advantage(
        response_mask=torch.tensor([[True, True, False, True]]),
        segment_ids=torch.tensor([[0, 0, -1, 1]]),
        boundary_values=values,
        objective_mode="preference",
        endpoint_weights=[0.5, 0.5],
        segment_mask=torch.tensor([[True, True]]),
    )
    torch.testing.assert_close(token_adv, torch.tensor([[1.5, 1.5, 0.0, 1.5]]))
    torch.testing.assert_close(returns, token_adv)
    with pytest.raises(ValueError, match="boundary values"):
        compute_polyseg_science_advantage(
            response_mask=torch.ones(1, 2, dtype=torch.bool),
            segment_ids=torch.zeros(1, 2, dtype=torch.long),
            boundary_values=torch.tensor([[[0.0, float("nan")], [1.0, 2.0]]]),
            objective_mode="preference",
            endpoint_weights=[1.0, 1.0],
        )


def test_science_advantage_accepts_explicit_structural_macro_padding():
    algo = AlgoConfig(
        adv_estimator="polyseg_sci",
        polyseg=PolySegAlgorithmConfig(
            science=PolySegScienceConfig(endpoint_weights=(0.5, 0.5), objective_mode="preference")
        ),
    )
    batch = {
        "science_boundary_values": torch.tensor(
            [[[0.0, 0.0], [1.0, 2.0], [1.0, 2.0]]]
        ),
        "science_boundary_mask": torch.tensor([[True, False]]),
        "science_endpoint_mask": torch.tensor(
            [[[True, True], [True, True], [False, False]]]
        ),
        "segment_ids": torch.tensor([[0, 0, -1]]),
    }
    advantages, _ = compute_polyseg_science_advantage_registered(
        token_level_rewards=None,
        response_mask=torch.tensor([[True, True, False]]),
        config=algo,
        batch=batch,
    )
    torch.testing.assert_close(advantages, torch.tensor([[1.5, 1.5, 0.0]]))
    batch["science_endpoint_mask"][0, 1, 0] = False
    with pytest.raises(ValueError, match="incomplete active endpoint"):
        compute_polyseg_science_advantage_registered(
            token_level_rewards=None,
            response_mask=torch.tensor([[True, True, False]]),
            config=algo,
            batch=batch,
        )


def test_science_advantage_allows_nan_only_in_explicitly_inactive_padding():
    algo = AlgoConfig(
        adv_estimator="polyseg_sci",
        polyseg=PolySegAlgorithmConfig(
            science=PolySegScienceConfig(endpoint_weights=(1.0,), objective_mode="preference")
        ),
    )
    batch = {
        "science_boundary_values": torch.tensor([[[0.0], [1.0], [float("nan")]]]),
        "science_boundary_mask": torch.tensor([[True, False]]),
        "science_endpoint_mask": torch.tensor([[[True], [True], [False]]]),
        "segment_ids": torch.tensor([[0, -1]]),
    }
    advantages, _ = compute_polyseg_science_advantage_registered(
        token_level_rewards=None,
        response_mask=torch.tensor([[True, False]]),
        config=algo,
        batch=batch,
    )
    torch.testing.assert_close(advantages, torch.tensor([[1.0, 0.0]]))


def test_science_estimator_reaches_verl_advantage_dispatch_and_actor_loss(monkeypatch):
    algo = AlgoConfig(
        adv_estimator="polyseg_sci",
        polyseg=PolySegAlgorithmConfig(
            science=PolySegScienceConfig(endpoint_weights=(0.5, 0.5), objective_mode="preference")
        ),
    )
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(1, 4),
                "response_mask": torch.tensor([[True, True, False, True]]),
                "segment_ids": torch.tensor([[0, 0, -1, 1]]),
                "segment_mask": torch.tensor([[True, True]]),
                "science_boundary_values": torch.tensor(
                    [[[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]]]
                ),
            },
            batch_size=[1],
        )
    )
    data = compute_advantage(data, adv_estimator="polyseg_sci", config=algo)
    torch.testing.assert_close(data.batch["advantages"], torch.tensor([[1.5, 1.5, 0.0, 1.5]]))
    actor_config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    actor_data = TensorDict(
        {
            "response_mask": data.batch["response_mask"],
            "old_log_probs": torch.zeros(1, 4),
            "advantages": data.batch["advantages"],
        },
        batch_size=[1],
    )
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    log_probs = torch.zeros(1, 4, requires_grad=True)
    loss, _ = ppo_loss(actor_config, {"log_probs": log_probs}, actor_data)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, log_probs)[0]).all()


def test_science_advantage_rejects_active_invalid_segment_ids_and_clamps_padding():
    values = torch.tensor([[[0.0], [1.0], [2.0]]])
    with pytest.raises(ValueError, match="non-negative segment ID"):
        compute_polyseg_science_advantage(
            response_mask=torch.tensor([[True, False]]),
            segment_ids=torch.tensor([[-1, 99]]),
            boundary_values=values,
            objective_mode="preference",
            endpoint_weights=[1.0],
        )
    # Out-of-range IDs in inactive padding are ignored rather than reaching
    # torch.gather and producing an accidental index error.
    advantages, _ = compute_polyseg_science_advantage(
        response_mask=torch.tensor([[True, False]]),
        segment_ids=torch.tensor([[0, 99]]),
        boundary_values=values,
        objective_mode="preference",
        endpoint_weights=[1.0],
    )
    torch.testing.assert_close(advantages, torch.tensor([[1.0, 0.0]]))


def test_constraint_estimator_accepts_broadcast_multiplier_and_preserves_upper_sign():
    values = torch.tensor([[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]]])
    token_adv, _ = compute_polyseg_science_advantage(
        response_mask=torch.tensor([[True, True]]),
        segment_ids=torch.tensor([[0, 1]]),
        boundary_values=values,
        objective_mode="constraint",
        constraint_multipliers=torch.tensor([1.0, 2.0]),
        constraint_signs=torch.tensor([1.0, -1.0]),
    )
    torch.testing.assert_close(token_adv, torch.tensor([[-4.0, -4.0]]))


def test_scientific_evaluator_enforces_fidelity_provenance_and_range():
    schema = _schema()
    evaluator = ScientificEndpointEvaluator(
        schema,
        evaluator_id="evaluator-activity",
        evaluator_version="hf-v1",
        evaluate_fn=lambda _: (EndpointObservation("activity", "activity-hf", 0.5),),
    )
    result = evaluator.evaluate({}, cost=2.0)
    assert result[0].evaluator_id == "evaluator-activity" and result[0].cost == 2.0
    bad = ScientificEndpointEvaluator(
        schema,
        evaluator_id="evaluator-activity",
        evaluator_version="hf-v1",
        evaluate_fn=lambda _: (EndpointObservation("activity", "activity-hf", 1.5),),
    )
    with pytest.raises(ValueError, match="outside valid range"):
        bad.evaluate({}, cost=1.0)


@pytest.mark.parametrize("status", ["interval", "censored", "below_detection_limit"])
def test_scientific_evaluator_preserves_missingness_metadata_on_rebuild(status):
    endpoint = EndpointSpec(
        "activity", "activity", "a.u.", valid_range=(0.0, 1.0),
        detection_limit=0.1, hf_fidelity_id="activity-hf",
        missing_policy="interval" if status == "below_detection_limit" else status,
        fidelities=(FidelitySpec("activity-hf", "HF", "evaluator", "v1", 1.0, kind="gold"),),
    )
    schema = EndpointSchema((endpoint,), objective_mode="preference", weights=(1.0,))
    source = EndpointObservation(
        "activity", "activity-hf", None if status == "below_detection_limit" else 0.15, status=status, cost=0.25,
        lower_bound=0.1, upper_bound=0.2, detection_limit=0.1,
        replicate_id="rep-7",
    )
    evaluator = ScientificEndpointEvaluator(
        schema, evaluator_id="evaluator", evaluator_version="v1", evaluate_fn=lambda _: (source,)
    )
    rebuilt = evaluator.evaluate({}, cost=2.0)[0]
    assert rebuilt.lower_bound == 0.1
    assert rebuilt.upper_bound == 0.2
    assert rebuilt.detection_limit == 0.1
    assert rebuilt.replicate_id == "rep-7"
    assert rebuilt.cost == 0.25
    schema.validate_observation(rebuilt, allow_empty_provenance=False)


def test_boundary_risk_exposes_explicit_alpha_beta_and_design_risk():
    risk = compute_boundary_risk(
        boundary_id="b0",
        W_b=[2.0, -1.0],
        residual_vector=[0.5, -0.25],
        branch_covariance=[[4.0, 1.0], [1.0, 2.0]],
        branch_counts=[1, 2, 4],
        cost_by_n={1: 3.0, 2: 5.0, 4: 9.0},
        inclusion_probability=0.5,
        assumptions=["OOF residual", "conditional branch covariance"],
    )
    assert isinstance(risk, BoundaryRisk)
    assert risk.alpha == pytest.approx(1.5625)
    assert risk.beta_by_n[1] == pytest.approx(30.0)
    assert risk.beta_by_n[2] == pytest.approx(15.0)
    assert risk.risk_by_n[2] == pytest.approx(31.5625)
    assert risk.cost_by_n[4] == pytest.approx(9.0)
    assert risk.valid is True
    assert risk.to_dict()["beta_by_n"]["2"] == pytest.approx(15.0)


def test_boundary_risk_rejects_non_psd_covariance_and_zero_propensity():
    with pytest.raises(ValueError, match="positive semidefinite"):
        compute_boundary_risk(
            W_b=[1.0], residual_vector=[1.0], branch_covariance=[[-1.0]],
        )
    with pytest.raises(ValueError, match=r"in \(0,1\]"):
        compute_boundary_risk(
            W_b=[1.0], residual_vector=[1.0], branch_covariance=[[1.0]],
            inclusion_probability=0.0,
        )


def test_boundary_risk_uses_methods_column_vectorization():
    metric = torch.diag(torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=torch.float64))
    risk = compute_boundary_risk(
        W_b=[[1.0, 2.0], [3.0, 4.0]],
        residual_vector=[3.0, 4.0],
        branch_covariance=torch.zeros((2, 2), dtype=torch.float64),
        gradient_metric=metric,
    )
    # vec(delta G) = [1*3, 3*4, 2*3, 4*4] for endpoint-specific W \otimes I_R.
    assert risk.alpha == pytest.approx(261049.0)


def test_gradient_risk_covariance_preserves_cross_boundary_blocks():
    covariance = compute_gradient_risk_covariance(
        sensitivities=[[1.0, 0.0], [0.0, 2.0]],
        boundary_covariances=[[[1.0]], [[2.0]]],
        cross_boundary_covariances={(1, 0): [[0.5]]},
    )
    assert covariance.shape == (2, 2)
    torch.testing.assert_close(covariance, torch.tensor([[1.0, 1.0], [1.0, 8.0]], dtype=torch.float64))


def test_gradient_risk_covariance_rejects_non_psd_boundary_block():
    with pytest.raises(ValueError, match="positive semidefinite"):
        compute_gradient_risk_covariance(
            sensitivities=[[1.0]],
            boundary_covariances=[[[-1.0]]],
        )


def test_boundary_risk_rejects_mismatched_branch_count_maps():
    risk = compute_boundary_risk(
        boundary_id="b0",
        W_b=[1.0],
        residual_vector=[0.5],
        branch_covariance=[[0.25]],
        branch_counts=[1, 2],
    )
    with pytest.raises(ValueError, match="share the same branch-count keys"):
        BoundaryRisk(
            boundary_id=risk.boundary_id,
            alpha=risk.alpha,
            beta_by_n=risk.beta_by_n,
            risk_by_n={1: risk.risk_by_n[1]},
            cost_by_n=risk.cost_by_n,
            sensitivity_norm=risk.sensitivity_norm,
            residual_norm=risk.residual_norm,
            covariance_trace=risk.covariance_trace,
            valid=True,
        )


def test_query_candidate_round_trips_integer_risk_keys():
    candidate = QueryCandidate(
        "b0", "activity", 1.0, 1.0, 1.0, 2.0, branch_count=2,
        alpha=3.0, beta_by_n={1: 4.0, 2: 2.0},
        risk_by_n={1: 7.0, 2: 5.0}, cost_by_n={1: 2.0, 2: 3.0},
    )
    restored = QueryCandidate.from_dict(candidate.to_dict())
    assert restored == candidate
    assert restored.estimated_gradient_risk == pytest.approx(5.0)


def test_bernoulli_risk_allocator_solves_budget_and_selects_branch_count():
    risks = {
        "b0": compute_boundary_risk(
            boundary_id="b0", W_b=[1.0], residual_vector=[1.0],
            branch_covariance=[[4.0]], branch_counts=[1, 2],
            cost_by_n={1: 1.0, 2: 1.5},
        ),
        "b1": compute_boundary_risk(
            boundary_id="b1", W_b=[2.0], residual_vector=[0.5],
            branch_covariance=[[1.0]], branch_counts=[1, 2],
            cost_by_n={1: 1.0, 2: 1.5},
        ),
    }
    design = BernoulliRiskAllocator.allocate(risks, budget=1.5, p_min=0.25, seed=9)
    assert isinstance(design, AuditDesign)
    assert design.status == "feasible"
    assert design.expected_cost <= 1.5 + 1e-8
    assert all(value >= 0.25 for value in design.probabilities.values())
    assert set(design.pair_probabilities) == {"b0::b1"}
    infeasible = BernoulliRiskAllocator.allocate(risks, budget=0.1, p_min=0.25)
    assert infeasible.status == "infeasible_budget"


def test_bernoulli_allocator_uses_clipped_kkt_floor_directly():
    high = compute_boundary_risk(
        boundary_id="high", W_b=[2.0], residual_vector=[1.0],
        branch_covariance=[[0.0]], branch_counts=[1], cost_by_n={1: 1.0},
    )
    low = compute_boundary_risk(
        boundary_id="low", W_b=[1.0], residual_vector=[1.0],
        branch_covariance=[[0.0]], branch_counts=[1], cost_by_n={1: 1.0},
    )
    design = BernoulliRiskAllocator.allocate(
        {"high": high, "low": low}, budget=0.6, p_min=0.2,
    )
    assert design.probabilities == pytest.approx({"high": 0.4, "low": 0.2})
    assert design.expected_cost == pytest.approx(0.6)


def test_bernoulli_allocator_handles_zero_and_mixed_risk_without_nan():
    zero = compute_boundary_risk(
        boundary_id="zero", W_b=[0.0], residual_vector=[0.0],
        branch_covariance=[[0.0]], branch_counts=[1], cost_by_n={1: 1.0},
    )
    all_zero = BernoulliRiskAllocator.allocate({"zero": zero}, budget=0.6, p_min=0.5)
    assert all_zero.status == "feasible"
    assert all_zero.lambda_value is None
    assert all_zero.probabilities["zero"] == pytest.approx(0.5)
    positive = compute_boundary_risk(
        boundary_id="positive", W_b=[1.0], residual_vector=[0.0],
        branch_covariance=[[1.0]], branch_counts=[1], cost_by_n={1: 1.0},
    )
    mixed = BernoulliRiskAllocator.allocate(
        {"zero": zero, "positive": positive}, budget=1.0, p_min=0.0,
    )
    assert mixed.status == "feasible"
    assert mixed.probabilities["positive"] == pytest.approx(1.0)
    assert mixed.probabilities["zero"] == pytest.approx(0.0)
    assert AuditDesign.from_dict(mixed.to_dict()) == mixed


def test_bernoulli_allocator_zero_risk_slack_budget_stops_at_floor():
    zero = compute_boundary_risk(
        boundary_id="zero-slack", W_b=[0.0], residual_vector=[0.0],
        branch_covariance=[[0.0]], branch_counts=[1], cost_by_n={1: 2.0},
    )
    design = BernoulliRiskAllocator.allocate(
        {"zero-slack": zero}, budget=100.0, p_min=0.25,
    )
    assert design.status == "feasible"
    assert design.probabilities["zero-slack"] == pytest.approx(0.25)
    assert design.expected_cost == pytest.approx(0.5)
    assert design.lambda_value is None


def test_bernoulli_allocator_positive_risk_saturation_leaves_zero_risk_at_floor():
    zero = compute_boundary_risk(
        boundary_id="zero-slack-mixed", W_b=[0.0], residual_vector=[0.0],
        branch_covariance=[[0.0]], branch_counts=[1], cost_by_n={1: 2.0},
    )
    positive = compute_boundary_risk(
        boundary_id="positive-slack-mixed", W_b=[1.0], residual_vector=[1.0],
        branch_covariance=[[0.0]], branch_counts=[1], cost_by_n={1: 1.0},
    )
    design = BernoulliRiskAllocator.allocate(
        {"zero-slack-mixed": zero, "positive-slack-mixed": positive},
        budget=100.0,
        p_min=0.25,
    )
    assert design.probabilities["positive-slack-mixed"] == pytest.approx(1.0)
    assert design.probabilities["zero-slack-mixed"] == pytest.approx(0.25)
    assert design.expected_cost == pytest.approx(1.5)
    assert design.lambda_value == pytest.approx(0.0)


def test_bernoulli_allocator_persists_methods_lambda_not_scale():
    risk = compute_boundary_risk(
        boundary_id="lambda", W_b=[1.0], residual_vector=[1.0],
        branch_covariance=[[0.0]], branch_counts=[1], cost_by_n={1: 1.0},
    )
    design = BernoulliRiskAllocator.allocate({"lambda": risk}, budget=0.25, p_min=0.0)
    assert design.status == "feasible"
    assert design.probabilities["lambda"] == pytest.approx(0.25)
    assert design.lambda_value == pytest.approx(16.0)
