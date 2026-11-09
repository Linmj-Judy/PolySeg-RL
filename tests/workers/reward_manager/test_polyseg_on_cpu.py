import numpy as np
import pytest
import torch
import json

from verl import DataProto
from verl.trainer.ppo.polyseg import compute_segment_rewards
from verl.trainer.ppo.polyseg import feature_map_contract
from verl.workers.reward_manager.polyseg import PolySegRewardManager


class _DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in token_ids)


def _data(*, with_segments=True, batch_size=2):
    fields = {
        "prompts": torch.zeros((batch_size, 2), dtype=torch.long),
        "responses": torch.arange(batch_size * 4, dtype=torch.long).reshape(batch_size, 4),
        "attention_mask": torch.ones((batch_size, 6), dtype=torch.long),
        "response_mask": torch.ones((batch_size, 4), dtype=torch.long),
        "data_source": np.asarray(["dummy"] * batch_size, dtype=object),
        "reward_model": np.asarray([{"ground_truth": "answer"}] * batch_size, dtype=object),
    }
    if with_segments:
        fields["extra_fields"] = np.asarray(
            [{"segment_ids": [0, 0, 1, 1], "segment_durations": [2.0, 1.0]}] * batch_size,
            dtype=object,
        )
    return DataProto.from_single_dict(fields)


def test_utility_delta_telescopes_and_unknown_protocol_fails():
    assert compute_segment_rewards([0.2, 0.5, 0.9], "utility_delta") == pytest.approx([0.2, 0.3, 0.4])
    assert sum(compute_segment_rewards([0.2, 0.5, 0.9], "utility_delta")) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        compute_segment_rewards([1.0], "typo")


def test_utility_delta_requires_explicit_utility_key():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *args, **kwargs: {"score": 1.0},
        polyseg_config={"m": 2, "feedback": "utility_delta"},
        require_rollout_segments=True,
    )
    with pytest.raises(KeyError, match="utility"):
        manager(_data(batch_size=1), return_dict=True)


def test_scorer_query_cost_is_budgeted_per_trajectory():
    def score(data_source, solution_str, ground_truth, extra_info):
        return {"score": 0.5, "query_cost": 2.0}

    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=score,
        polyseg_config={"m": 2, "feedback": "local_sum", "budget": 4, "segment_cost": 2},
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    assert result["reward_extra_info"]["segment_query_count"] == [2]
    assert result["reward_extra_info"]["segment_query_cost"] == [4.0]


def test_global_budget_passes_pre_reservation_capacity_to_scorer_at_exact_boundary():
    seen = []

    def score(data_source, solution_str, ground_truth, extra_info):
        seen.append(float(extra_info["budget_remaining"]))
        # This is the exact-budget case: the scorer must be allowed to bill
        # the reservation even though the ledger's post-reservation balance is
        # already zero.
        assert float(extra_info["budget_remaining"]) in (pytest.approx(2.0), pytest.approx(1.0))
        return {"score": 1.0, "query_cost": 1.0}

    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=score,
        polyseg_config={
            "m": 2,
            "feedback": "local_sum",
            "budget": 2,
            "budget_scope": "global",
            "segment_cost": 1,
        },
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    assert seen == [2.0, 1.0]
    assert result["reward_extra_info"]["segment_budget_remaining"] == [0.0]


def test_reward_manager_uses_rollout_segments_and_isolates_budget():
    calls = []

    def score(data_source, solution_str, ground_truth, extra_info):
        calls.append(extra_info["segment_id"])
        return {"score": float(extra_info["segment_id"] + 1), "acc": 1.0 if extra_info["segment_id"] == 1 else 0.0}

    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=score,
        polyseg_config={"q": 4, "m": 4, "feedback": "local_sum", "budget": 2, "segment_cost": 1},
        require_rollout_segments=True,
    )
    result = manager(_data(), return_dict=True)
    assert result["reward_extra_info"]["segment_rewards"][0].tolist() == [1.0, 2.0]
    assert result["reward_extra_info"]["segment_rewards"][1].tolist() == [1.0, 2.0]
    assert result["reward_extra_info"]["episode_reward"] == [3.0, 3.0]
    assert result["reward_extra_info"]["acc"] == [1.0, 1.0]
    assert result["reward_extra_info"]["polyseg_adaptive_q"] == [4, 4]
    assert result["reward_extra_info"]["polyseg_feature_map_version"] == ["polyseg-fixed-boundary-summary-v1"] * 2
    assert all(len(value) == 64 for value in result["reward_extra_info"]["polyseg_feature_map_hash"])
    assert calls == [0, 1, 0, 1]
    # The sparse legacy tensor has one reward at each segment's final action.
    assert result["reward_tensor"][0].tolist() == [0.0, 1.0, 0.0, 2.0]


def test_reward_manager_dispatches_configured_reward_fn_key():
    seen = []

    def score(data_source, solution_str, ground_truth, extra_info):
        seen.append(data_source)
        return {"score": 1.0, "acc": 1.0}

    data = _data(batch_size=1)
    data.non_tensor_batch.pop("data_source")
    data.non_tensor_batch["task_source"] = np.asarray(["custom/task"], dtype=object)
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=score,
        reward_fn_key="task_source",
        polyseg_config={"m": 2, "feedback": "local_sum"},
        require_rollout_segments=True,
    )
    result = manager(data, return_dict=True)
    assert result["reward_extra_info"]["final_accuracy"] == [1.0]
    assert seen == ["custom/task", "custom/task"]


def test_reward_manager_missing_configured_reward_fn_key_fails_closed():
    data = _data(batch_size=1)
    data.non_tensor_batch.pop("data_source")
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *args, **kwargs: {"score": 1.0},
        reward_fn_key="task_source",
        polyseg_config={"m": 2, "feedback": "local_sum"},
        require_rollout_segments=True,
    )
    with pytest.raises(KeyError, match="task_source"):
        manager(data, return_dict=True)


def test_reward_manager_preserves_direct_rollout_successor_fields():
    fields = {
        "prompts": torch.zeros((1, 2), dtype=torch.long),
        "responses": torch.arange(4, dtype=torch.long).reshape(1, 4),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "response_mask": torch.ones((1, 4), dtype=torch.long),
        "data_source": np.asarray(["dummy"], dtype=object),
        "reward_model": np.asarray([{"ground_truth": "answer"}], dtype=object),
        "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1], "segment_durations": [1.0, 1.0]}], dtype=object),
        "segment_next_features": np.asarray([[[0.5], [0.0]]], dtype=object),
        "segment_transition_mask": np.asarray([[True, False]], dtype=object),
    }
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *args, **kwargs: {"score": 1.0},
        polyseg_config={"m": 2, "feature_dim": 1},
        require_rollout_segments=True,
    )
    result = manager(DataProto.from_single_dict(fields), return_dict=True)
    info = result["reward_extra_info"]
    assert info["polyseg_transition_source"] == ["rollout"]
    assert np.asarray(info["segment_transition_mask"][0]).tolist() == [True, False]
    assert np.asarray(info["segment_next_features"][0]).tolist() == [[0.5], [0.0]]


def test_reward_manager_uses_fixed_feature_horizon_for_ragged_batches():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *args, **kwargs: {"score": 1.0},
        polyseg_config={"m": 2, "feature_horizon": 2048},
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    info = result["reward_extra_info"]
    expected = feature_map_contract(response_length=2048, feature_dim=4)["hash"]
    assert info["polyseg_feature_map_hash"] == [expected]
    assert info["polyseg_feature_causal_availability"] == ["available_at_segment_boundary"]
    assert info["polyseg_feature_markov_sufficiency"] == ["not_established"]


def test_agentic_mode_rejects_missing_rollout_metadata():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *args, **kwargs: 1.0,
        polyseg_config={"m": 2},
        require_rollout_segments=True,
    )
    with pytest.raises(ValueError, match="segment_ids"):
        manager(_data(with_segments=False), return_dict=True)


def test_tool_observation_is_not_reward_endpoint():
    fields = {
        "prompts": torch.zeros((1, 2), dtype=torch.long),
        "responses": torch.arange(5, dtype=torch.long).reshape(1, 5),
        "attention_mask": torch.ones((1, 7), dtype=torch.long),
        "response_mask": torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.long),
        "data_source": np.asarray(["dummy"], dtype=object),
        "reward_model": np.asarray([{"ground_truth": "answer"}], dtype=object),
        "extra_fields": np.asarray(
            [{"segment_ids": [0, 0, 0, 0, 1], "segment_durations": [1.0, 1.0]}], dtype=object
        ),
    }
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda data_source, solution_str, ground_truth, extra_info: {"score": extra_info["segment_id"] + 1},
        require_rollout_segments=True,
    )
    result = manager(DataProto.from_single_dict(fields), return_dict=True)
    # Segment 0 has observation positions 2/3; its reward goes to action token 1.
    assert result["reward_tensor"][0].tolist() == [0.0, 1.0, 0.0, 0.0, 2.0]


def test_budget_fallback_skip_is_explicit_and_auditable():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scorer must not run")),
        polyseg_config={"m": 2, "feedback": "local_sum", "budget": 0, "segment_cost": 1, "budget_fallback": "skip"},
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    info = result["reward_extra_info"]
    assert info["segment_query_count"] == [0]
    assert info["segment_feedback_fallback_count"] == [2]
    assert info["segment_budget_remaining"] == [0.0]
    assert np.asarray(info["segment_feedback_mask"][0]).tolist() == [0, 0]


def test_controller_skip_marker_prevents_queries_without_reward_budget():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scorer must not run")),
        polyseg_config={"m": 2, "feedback": "local_sum"},
        require_rollout_segments=True,
    )
    data = _data(batch_size=1)
    data.non_tensor_batch["polyseg_adaptive_fallback"] = np.asarray(["skip"], dtype=object)
    data.non_tensor_batch["polyseg_adaptive_fallback_reason"] = np.asarray(
        ["feedback_budget_exhausted_before_batch"], dtype=object
    )
    result = manager(data, return_dict=True)
    info = result["reward_extra_info"]
    assert info["segment_query_count"] == [0]
    assert info["polyseg_adaptive_fallback"] == ["skip"]
    assert info["polyseg_adaptive_fallback_reason"] == ["feedback_budget_exhausted_before_batch"]
    assert info["segment_feedback_fallback_count"] == [2]


def test_verifier_receives_tool_context_for_each_segment():
    seen = []

    def score(data_source, solution_str, ground_truth, extra_info):
        seen.append((extra_info["tool_calls"], extra_info["tool_observations"]))
        return {"score": 0.0}

    data = _data(batch_size=1)
    data.non_tensor_batch["extra_fields"] = np.asarray(
        [
            {
                "segment_ids": [0, 0, 1, 1],
                "segment_tool_calls": [[{"name": "calculator", "arguments": "2+2"}], []],
                "segment_tool_observations": [["4"], []],
            }
        ],
        dtype=object,
    )
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=score,
        polyseg_config={"m": 2, "feedback": "local_sum"},
        require_rollout_segments=True,
    )
    manager(data, return_dict=True)
    assert seen == [([{"name": "calculator", "arguments": "2+2"}], ["4"]), ([], [])]


def test_missing_accuracy_is_not_filled_from_utility_score():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *args, **kwargs: {"score": 0.37},
        polyseg_config={"m": 2, "feedback": "local_sum"},
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    info = result["reward_extra_info"]
    assert info["final_accuracy"] == [None]
    assert info["acc"] == [None]



def test_terminal_feedback_marks_nonfinal_segments_unqueried():
    calls = []

    def score(data_source, solution_str, ground_truth, extra_info):
        calls.append(extra_info["segment_id"])
        return {"score": 1.0, "acc": 1.0}

    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(), num_examine=0, compute_score=score,
        polyseg_config={"m": 2, "feedback": "terminal"}, require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    assert calls == [1]
    assert result["reward_extra_info"]["segment_query_count"] == [1]
    assert [int(x) for x in result["reward_extra_info"]["segment_feedback_mask"][0]] == [0, 1]


def test_zero_reported_cost_cannot_create_free_budget():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(), num_examine=0,
        compute_score=lambda *a, **k: {"score": 1.0, "query_cost": 0.0},
        polyseg_config={"m": 2, "feedback": "local_sum", "budget": 4, "segment_cost": 1.0},
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    # The conservative default charges the reserved upper bound.
    assert result["reward_extra_info"]["segment_query_cost"] == [2.0]


def test_reported_cost_policy_requires_receipt_for_lower_bill():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *a, **k: {
            "score": 1.0,
            "query_cost": 0.25,
            "billing_receipt": {"provider_query_id": "q-1", "billed_cost": 0.25},
        },
        polyseg_config={
            "m": 2,
            "feedback": "local_sum",
            "budget": 2,
            "segment_cost": 1.0,
            "feedback_cost_policy": "reported_with_receipt",
        },
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    assert result["reward_extra_info"]["segment_query_cost"] == [0.5]
    assert result["reward_extra_info"]["segment_budget_remaining"] == [pytest.approx(1.5)]


def test_reported_cost_without_receipt_keeps_conservative_bill():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *a, **k: {"score": 1.0, "query_cost": 0.25},
        polyseg_config={
            "m": 2,
            "feedback": "local_sum",
            "budget": 2,
            "segment_cost": 1.0,
            "feedback_cost_policy": "reported_with_receipt",
        },
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    assert result["reward_extra_info"]["segment_query_cost"] == [2.0]
    assert result["reward_extra_info"]["segment_budget_remaining"] == [pytest.approx(0.0)]


def test_receipt_without_billed_cost_keeps_conservative_bill():
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *a, **k: {
            "score": 1.0,
            "query_cost": 0.25,
            "billing_receipt": {"provider_query_id": "q-missing-cost"},
        },
        polyseg_config={
            "m": 2,
            "feedback": "local_sum",
            "budget": 2,
            "segment_cost": 1.0,
            "feedback_cost_policy": "reported_with_receipt",
        },
        require_rollout_segments=True,
    )
    result = manager(_data(batch_size=1), return_dict=True)
    info = result["reward_extra_info"]
    assert info["segment_query_cost"] == [2.0]
    assert all(item["status"] == "invalid_billed_cost" for item in info["billing_receipts"][0])


def test_sync_reward_manager_writes_realized_resource_ledger(tmp_path):
    journal = tmp_path / "feedback.jsonl"
    manager = PolySegRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=lambda *a, **k: {"score": 1.0},
        polyseg_config={
            "m": 2,
            "feedback": "local_sum",
            "feedback_journal_path": str(journal),
            "feedback_attempt_id": "attempt",
        },
        require_rollout_segments=True,
    )
    manager(_data(batch_size=1), return_dict=True)
    row = json.loads(journal.read_text().strip())
    assert row["resource_accounting"] == {
        "prompt_tokens": 2,
        "response_tokens": 4,
        "action_tokens": 4,
        "tool_observation_tokens": 0,
        "total_tokens": 6,
        "tool_calls": 0,
        "tool_successes": 0,
        "tool_failures": 0,
    }
