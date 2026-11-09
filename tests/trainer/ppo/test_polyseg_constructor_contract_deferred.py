import pytest

from verl.trainer.ppo.polyseg_sci import (
    ActionRecord,
    CausalSegmenter,
    EndpointObservation,
    FeatureMapContract,
    FidelitySpec,
    BudgetLedger,
    SegmentBoundary,
    EndpointSpec, EndpointSchema, QueryCandidate, QueryDecision, RestoreAudit, BranchResult,
    BoundaryRecord, BoundarySnapshot,
)
from verl.trainer.ppo.polyseg import BudgetedPolySegController, PolySegConfig
from verl.trainer.ppo.polyseg_sci_provider import ScienceRoundRequest, ScienceRoundResponse


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(token_start=4, token_end=4),
        dict(token_start=0.5, token_end=1.5),
        dict(token_start=False, token_end=True),
    ],
)
def test_policy_action_record_requires_positive_integer_span(kwargs):
    with pytest.raises(ValueError, match="invalid action token interval"):
        ActionRecord("policy", "policy_token", True, action_mask=True, **kwargs)


def test_tool_lifecycle_record_can_keep_zero_span():
    record = ActionRecord("tool", "tool_call", False, 0, 0, action_mask=False)
    assert record.token_start == record.token_end == 0


@pytest.mark.parametrize("is_policy_action", [0, 1, 0.0, 1.0, "true", None])
def test_action_record_requires_boolean_policy_flag(is_policy_action):
    with pytest.raises(ValueError, match="is_policy_action"):
        ActionRecord("policy", "policy_token", is_policy_action, 0, 1)


@pytest.mark.parametrize("boundary_type", ["", None, 1])
def test_segment_boundary_requires_nonempty_string_type(boundary_type):
    with pytest.raises(ValueError, match="boundary_type"):
        SegmentBoundary(0, 0, 1, (0,), boundary_type, 1)


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_endpoint_observation_requires_numeric_observed_value(value):
    with pytest.raises(ValueError):
        EndpointObservation("endpoint", "fidelity", value)


@pytest.mark.parametrize("cost", [True, "1.0"])
def test_endpoint_observation_requires_numeric_cost(cost):
    with pytest.raises(ValueError):
        EndpointObservation("endpoint", "fidelity", 0.5, cost=cost)


@pytest.mark.parametrize("clip", [0, 1, "false", "true", None])
def test_feature_map_requires_boolean_clip(clip):
    with pytest.raises(ValueError, match="clip"):
        FeatureMapContract(("x",), (0.0,), (1.0,), clip=clip)
    with pytest.raises(ValueError, match="clip"):
        FeatureMapContract.from_dict({"feature_names": ["x"], "lower": [0.0], "upper": [1.0], "clip": clip})


@pytest.mark.parametrize("calibrated_to_hf", [0, 1, "false", "true", None])
def test_fidelity_requires_boolean_calibration_flag(calibrated_to_hf):
    with pytest.raises(ValueError, match="calibrated_to_hf"):
        FidelitySpec("f", "LF", "e", "v", 1.0, calibrated_to_hf=calibrated_to_hf)


@pytest.mark.parametrize("budget", [True, "2"])
def test_budget_ledger_requires_real_numeric_budget(budget):
    with pytest.raises(ValueError):
        BudgetLedger(budget)


@pytest.mark.parametrize("amount", [True, "1.0"])
def test_budget_ledger_rejects_non_numeric_amounts(amount):
    with pytest.raises(ValueError):
        BudgetLedger(2.0, consumed={"query": amount})
    ledger = BudgetLedger(2.0)
    with pytest.raises(ValueError):
        ledger.reserve("query", amount)


@pytest.mark.parametrize("field", ["q", "m", "feature_dim", "horizon", "mlp_hidden_dim", "mlp_steps", "transition_backup_iterations", "dependence_history_limit"])
def test_polyseg_config_rejects_fractional_or_boolean_integer_controls(field):
    for value in (True, 1.5):
        with pytest.raises(ValueError):
            PolySegConfig(**{field: value})


def test_controller_rejects_non_numeric_total_budget():
    with pytest.raises(ValueError):
        BudgetedPolySegController(q_candidates=[1], m_candidates=[1], total_budget=True)
    with pytest.raises(ValueError):
        BudgetedPolySegController(q_candidates=[1], m_candidates=[1], total_budget="2.0")


def _provider_fields(**overrides):
    fields = {
        "segment_ids": [0],
        "segment_mask": [True],
        "science_boundary_values": [[0.1], [0.2]],
        "science_boundary_mask": [True],
        "science_endpoint_mask": [[True], [True]],
    }
    fields.update(overrides)
    return fields


@pytest.mark.parametrize("eligible", [0, 1, 1.0, "true", None])
def test_science_provider_response_requires_boolean_eligibility(eligible):
    with pytest.raises(ValueError, match="research_eligible"):
        ScienceRoundResponse(fields=_provider_fields(), research_eligible=eligible)


@pytest.mark.parametrize("field,value", [("segment_mask", [1]), ("science_boundary_mask", [1]), ("science_endpoint_mask", [[1], [1]]), ("segment_ids", [0.5]), ("segment_ids", [-0.5])])
def test_science_provider_response_rejects_coercible_metadata(field, value):
    with pytest.raises(ValueError):
        ScienceRoundResponse(fields=_provider_fields(**{field: value}))


@pytest.mark.parametrize("validate", [0, 1, 1.0, "true", None])
def test_science_provider_request_requires_boolean_validation_flag(validate):
    with pytest.raises(ValueError, match="validate"):
        ScienceRoundRequest(None, (), validate, {})


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(segment_id=0.5, start=0, end=2, action_indices=(0, 1), duration=2),
        dict(segment_id=0, start=0, end=2, action_indices=(True, 1), duration=2),
        dict(segment_id=0, start=0, end=2, action_indices=(0, 0), duration=2),
        dict(segment_id=0, start=0, end=2, action_indices=(0, 1), duration=2.0),
    ],
)
def test_segment_boundary_requires_strict_integer_unique_indices(kwargs):
    with pytest.raises(ValueError):
        SegmentBoundary(boundary_type="event", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(m=1.5, horizon=8),
        dict(m=True, horizon=8),
        dict(m=2, horizon=8.5),
        dict(m=2, horizon=True),
    ],
)
def test_causal_segmenter_requires_integer_controls(kwargs):
    with pytest.raises(ValueError):
        CausalSegmenter(**kwargs)


@pytest.mark.parametrize("mask", [[0, 1], [0.0, 1.0], [True, 1]])
def test_causal_segmenter_rejects_non_boolean_action_mask(mask):
    with pytest.raises(ValueError, match="action_mask"):
        CausalSegmenter(m=1, horizon=2).segment(mask)


@pytest.mark.parametrize("positions", [[1.5], [True], [0], [3]])
def test_causal_segmenter_rejects_non_integer_or_out_of_range_event_positions(positions):
    with pytest.raises(ValueError, match="boundary positions"):
        CausalSegmenter(rule="event", m=1).segment([True, True], event_positions=positions)


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, "false", None])
def test_query_decision_requires_boolean_hard_budget(value):
    from verl.trainer.ppo.polyseg_sci import QueryDecision
    with pytest.raises(ValueError, match="hard_budget"):
        QueryDecision((), (), {}, {}, "uniform", value, 0.0, 0.0, 0)


@pytest.mark.parametrize("field", ["trajectory_complete", "research_eligible"])
def test_trajectory_record_requires_boolean_flags(field):
    from verl.trainer.ppo.polyseg_sci import TrajectoryRecord
    data = dict(task_id="t", trajectory_id="tr", prompt_id="p", run_id="r", session_id="s", actor_checkpoint="a", tokenizer_version="tv", policy_version="pv", sampling_config={}, terminal_observations=(), terminal_evaluator_id="e", terminal_evaluator_version="v", terminal_cost=0.0, execution_budget_initial=1.0, execution_budget_remaining=1.0, restore_contract_version="v", restore_status="not_attempted", restore_cost=0.0, feature_map_version="f", feature_map_hash="h", segment_rule_version="s", trajectory_complete=True, research_eligible=False)
    data[field] = "false"
    with pytest.raises(ValueError, match="boolean"):
        TrajectoryRecord(**data)


@pytest.mark.parametrize("field", ["normalized_basis", "use_kl_in_reward", "normalize_advantages", "use_polynomial_critic", "accumulate_estimator", "use_confidence_bonus", "use_formal_confidence_radius", "use_transition_estimator"])
def test_polyseg_config_requires_boolean_flags(field):
    from verl.trainer.ppo.polyseg import PolySegConfig
    with pytest.raises(ValueError, match="boolean"):
        PolySegConfig(**{field: "false"})

@pytest.mark.parametrize("factory", [
    lambda: FidelitySpec("f", "LF", "e", "v", True),
    lambda: EndpointSpec("e", "e", "u", priority=True, proxy_only=True),
    lambda: QueryCandidate("b", "e", True, 1.0, 1.0, 1.0),
    lambda: QueryDecision((), (), {}, {}, "uniform", False, True, 0.0, 0),
    lambda: RestoreAudit("b", "passed", "s", None, True),
    lambda: BranchResult("b", "r", "passed", (), 1.0, True),
])
def test_science_numeric_ledger_fields_reject_boolean(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("value", [True, False, "1"])
def test_endpoint_schema_rejects_non_real_preference_weights(value):
    endpoint = EndpointSpec("e", "e", "u", proxy_only=True)
    with pytest.raises((ValueError, TypeError)):
        EndpointSchema((endpoint,), weights=(value,))


@pytest.mark.parametrize("field", ["execution_cost", "timestamp", "logprob", "old_logprob", "remaining_budget"])
def test_action_record_rejects_boolean_or_string_numeric_fields(field):
    kwargs = dict(action_id="a", action_type="policy", is_policy_action=True, token_start=0, token_end=1)
    kwargs[field] = True if field != "timestamp" else "1.0"
    with pytest.raises((ValueError, TypeError)):
        ActionRecord(**kwargs)


@pytest.mark.parametrize("field", ["lower", "upper"])
def test_feature_map_rejects_boolean_numeric_bounds(field):
    kwargs = {"feature_names": ("x",), "lower": (0.0,), "upper": (2.0,)}
    kwargs[field] = (True,)
    with pytest.raises(ValueError):
        FeatureMapContract(**kwargs)


def test_action_record_requires_boolean_action_mask():
    with pytest.raises(ValueError, match="action_mask"):
        ActionRecord("a", "policy", True, 0, 1, action_mask="true")


def _boundary_record():
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10.0, 6.0,
        "policy", "tokenizer", {}, "feature", "feature-hash", "segment", "snapshot",
    )
    return BoundaryRecord(
        boundary_id="b0", trajectory_id="traj", segment_id=0, start=0, end=1, duration=1,
        boundary_type="science_round", snapshot=snapshot, feature_vector=(0.0,), prediction=(0.1,),
        oof_residual=(None,), query_selected={"endpoint": True},
        query_inclusion_probability={"endpoint": 1.0}, branch_count={"endpoint": 1},
        gold_or_proxy={"endpoint": "gold"}, gold_score={"endpoint": 0.5}, query_cost=1.0,
        restore_cost=0.1, branch_status={"endpoint": "completed"}, failure_reason={"endpoint": None},
        budget_remaining_after=4.0, response_probability={"endpoint": 1.0},
        valid_observation={"endpoint": True}, remaining_execution_budget=6.0,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("segment_id", True), ("start", 0.5), ("end", "1"), ("duration", True),
        ("query_selected", {"endpoint": 1}),
        ("query_inclusion_probability", {"endpoint": "1.0"}),
        ("branch_count", {"endpoint": 1.0}),
        ("valid_observation", {"endpoint": "true"}),
    ],
)
def test_boundary_record_rejects_coercible_ledger_fields(field, value):
    with pytest.raises(ValueError):
        BoundaryRecord(**{**_boundary_record().__dict__, field: value})


def test_boundary_record_accepts_generated_contract():
    record = _boundary_record()
    assert record.duration == 1
    assert record.to_dict()["query_selected"] == {"endpoint": True}
