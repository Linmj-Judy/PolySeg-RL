import asyncio

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.polyseg import PolySegRewardManager
from verl.trainer.ppo.polyseg import feature_map_contract


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids)


def test_async_polyseg_reward_manager_accepts_rollout_segments():
    config = OmegaConf.create(
        {
            "reward": {"reward_kwargs": {"m": 2, "feedback": "utility_delta", "budget": 2}},
            "algorithm": {"polyseg": {"q": 4}},
        }
    )

    async def score(**kwargs):
        return {"utility": [0.25, 0.75][kwargs["extra_info"]["segment_id"]], "acc": 1.0}

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    data = DataProto.from_single_dict(
        {
            "prompts": torch.zeros((1, 2), dtype=torch.long),
            "responses": torch.arange(4).reshape(1, 4),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "data_source": np.asarray(["dummy"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
            "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1], "segment_durations": [1, 2]}], dtype=object),
        }
    )
    result = asyncio.run(manager.run_single(data))
    assert result["reward_score"] == 0.75
    assert result["reward_extra_info"]["segment_rewards"].tolist() == [0.25, 0.5]
    assert result["reward_extra_info"]["segment_query_cost"] == 2.0
    assert result["reward_extra_info"]["polyseg_adaptive_q"] == 4
    assert result["reward_extra_info"]["polyseg_transition_source"] == "missing"
    assert result["reward_extra_info"]["polyseg_feature_map_version"] == "polyseg-fixed-boundary-summary-v1"
    assert len(result["reward_extra_info"]["polyseg_feature_map_hash"]) == 64
    assert result["reward_extra_info"]["polyseg_feature_causal_availability"] == "available_at_segment_boundary"
    assert result["reward_extra_info"]["polyseg_feature_markov_sufficiency"] == "not_established"
    assert not np.asarray(result["reward_extra_info"]["segment_transition_mask"]).any()
    assert np.asarray(result["reward_extra_info"]["segment_feedback_mask"]).tolist() == [1, 1]


def test_async_reward_manager_uses_fixed_feature_horizon():
    config = OmegaConf.create(
        {
            "reward": {"reward_kwargs": {"m": 2, "feature_horizon": 2048}},
            "algorithm": {"polyseg": {}},
        }
    )

    async def score(**kwargs):
        return {"score": 1.0, "acc": 1.0}

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    data = DataProto.from_single_dict(
        {
            "prompts": torch.zeros((1, 2), dtype=torch.long),
            "responses": torch.arange(4).reshape(1, 4),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "data_source": np.asarray(["dummy"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
            "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1], "segment_durations": [1, 1]}], dtype=object),
        }
    )
    result = asyncio.run(manager.run_single(data))
    expected = feature_map_contract(response_length=2048, feature_dim=4)["hash"]
    assert result["reward_extra_info"]["polyseg_feature_map_hash"] == expected


def test_async_reward_manager_dispatches_configured_reward_fn_key():
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "task_source"},
            "reward": {"reward_kwargs": {"m": 2, "feedback": "local_sum"}},
            "algorithm": {"polyseg": {}},
        }
    )
    seen = []

    async def score(**kwargs):
        seen.append(kwargs["data_source"])
        return {"score": 1.0, "acc": 1.0}

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    data = DataProto.from_single_dict(
        {
            "prompts": torch.zeros((1, 2), dtype=torch.long),
            "responses": torch.arange(4).reshape(1, 4),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "task_source": np.asarray(["custom/task"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
            "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1], "segment_durations": [1, 1]}], dtype=object),
        }
    )
    result = asyncio.run(manager.run_single(data))
    assert result["reward_extra_info"]["final_accuracy"] == 1.0
    assert seen == ["custom/task", "custom/task"]


def test_async_reward_manager_missing_configured_reward_fn_key_fails_closed():
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "task_source"},
            "reward": {"reward_kwargs": {"m": 2, "feedback": "local_sum"}},
            "algorithm": {"polyseg": {}},
        }
    )

    async def score(**kwargs):
        return {"score": 1.0}

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    data = DataProto.from_single_dict(
        {
            "prompts": torch.zeros((1, 2), dtype=torch.long),
            "responses": torch.arange(4).reshape(1, 4),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "data_source": np.asarray(["wrong/source"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
            "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1]}], dtype=object),
        }
    )
    try:
        asyncio.run(manager.run_single(data))
    except KeyError as exc:
        assert "task_source" in str(exc)
    else:
        raise AssertionError("missing configured reward_fn_key did not fail closed")


def test_async_controller_skip_marker_is_auditable_and_does_not_score():
    config = OmegaConf.create(
        {"reward": {"reward_kwargs": {"m": 2, "feedback": "local_sum"}}, "algorithm": {"polyseg": {}}}
    )

    async def score(**kwargs):
        raise AssertionError("scorer must not run for controller fallback")

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    data = DataProto.from_single_dict(
        {
            "prompts": torch.zeros((1, 2), dtype=torch.long),
            "responses": torch.arange(4).reshape(1, 4),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "data_source": np.asarray(["dummy"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
            "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1]}], dtype=object),
            "polyseg_adaptive_fallback": np.asarray(["skip"], dtype=object),
            "polyseg_adaptive_fallback_reason": np.asarray(
                ["feedback_budget_exhausted_before_batch"], dtype=object
            ),
        }
    )
    result = asyncio.run(manager.run_single(data))
    info = result["reward_extra_info"]
    assert info["segment_query_count"] == 0
    assert info["segment_feedback_fallback_count"] == 2
    assert np.asarray(info["segment_feedback_mask"]).tolist() == [0, 0]


def test_async_utility_delta_rejects_score_alias():
    config = OmegaConf.create(
        {"reward": {"reward_kwargs": {"m": 2, "feedback": "utility_delta"}}, "algorithm": {"polyseg": {}}}
    )

    async def score(**kwargs):
        return {"score": 1.0}

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    data = DataProto.from_single_dict(
        {
            "prompts": torch.zeros((1, 2), dtype=torch.long),
            "responses": torch.arange(4).reshape(1, 4),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "data_source": np.asarray(["dummy"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
            "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1]}], dtype=object),
        }
    )
    try:
        asyncio.run(manager.run_single(data))
    except KeyError as exc:
        assert "utility" in str(exc)
    else:
        raise AssertionError("utility_delta accepted an ambiguous score response")


def test_global_budget_scope_is_shared_across_concurrent_trajectories():
    config = OmegaConf.create(
        {
            "reward": {"reward_kwargs": {"m": 1, "feedback": "local_sum", "budget": 2, "budget_scope": "global"}},
            "algorithm": {"polyseg": {}},
        }
    )

    async def score(**kwargs):
        await asyncio.sleep(0)
        return {"score": 1.0, "query_cost": 1.0}

    manager = PolySegRewardManager(config, _Tokenizer(), score)

    def sample():
        return DataProto.from_single_dict(
            {
                "prompts": torch.zeros((1, 2), dtype=torch.long),
                "responses": torch.arange(4).reshape(1, 4),
                "attention_mask": torch.ones((1, 6), dtype=torch.long),
                "response_mask": torch.ones((1, 4), dtype=torch.long),
                "data_source": np.asarray(["dummy"], dtype=object),
                "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
                "extra_fields": np.asarray([{"segment_ids": [0, 0, 0, 0]}], dtype=object),
            }
        )

    async def run():
        return await asyncio.gather(manager.run_single(sample()), manager.run_single(sample()))

    results = asyncio.run(run())
    assert sum(result["reward_extra_info"]["segment_query_cost"] for result in results) == 2.0
    assert results[0]["reward_extra_info"]["segment_budget_remaining"] == 0.0
    assert results[1]["reward_extra_info"]["segment_budget_remaining"] == 0.0


def test_async_missing_accuracy_is_not_filled_from_utility_score():
    config = OmegaConf.create(
        {"reward": {"reward_kwargs": {"m": 2, "feedback": "local_sum"}}, "algorithm": {"polyseg": {}}}
    )

    async def score(**kwargs):
        return {"score": 0.37}

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    data = DataProto.from_single_dict(
        {
            "prompts": torch.zeros((1, 2), dtype=torch.long),
            "responses": torch.arange(4).reshape(1, 4),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "data_source": np.asarray(["dummy"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
            "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1]}], dtype=object),
        }
    )
    result = asyncio.run(manager.run_single(data))
    assert result["reward_extra_info"]["final_accuracy"] is None
    assert result["reward_extra_info"]["acc"] is None


def _async_budget_data():
    return DataProto.from_single_dict(
        {
            "prompts": torch.zeros((1, 2), dtype=torch.long),
            "responses": torch.arange(4).reshape(1, 4),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "data_source": np.asarray(["dummy"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
            "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1]}], dtype=object),
        }
    )


def test_async_reported_cost_policy_requires_receipt_for_lower_bill():
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "m": 2,
                    "feedback": "local_sum",
                    "budget": 2,
                    "segment_cost": 1.0,
                    "feedback_cost_policy": "reported_with_receipt",
                }
            },
            "algorithm": {"polyseg": {}},
        }
    )

    async def score(**kwargs):
        return {
            "score": 1.0,
            "query_cost": 0.25,
            "billing_receipt": {"provider_query_id": "q-1", "billed_cost": 0.25},
        }

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    result = asyncio.run(manager.run_single(_async_budget_data()))
    assert result["reward_extra_info"]["segment_query_cost"] == 0.5
    assert result["reward_extra_info"]["segment_budget_remaining"] == 1.5


def test_async_reported_cost_without_receipt_keeps_conservative_bill():
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "m": 2,
                    "feedback": "local_sum",
                    "budget": 2,
                    "segment_cost": 1.0,
                    "feedback_cost_policy": "reported_with_receipt",
                }
            },
            "algorithm": {"polyseg": {}},
        }
    )

    async def score(**kwargs):
        return {"score": 1.0, "query_cost": 0.25}

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    result = asyncio.run(manager.run_single(_async_budget_data()))
    assert result["reward_extra_info"]["segment_query_cost"] == 2.0
    assert result["reward_extra_info"]["segment_budget_remaining"] == 0.0


def test_async_receipt_without_billed_cost_keeps_conservative_bill():
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "m": 2,
                    "feedback": "local_sum",
                    "budget": 2,
                    "segment_cost": 1.0,
                    "feedback_cost_policy": "reported_with_receipt",
                }
            },
            "algorithm": {"polyseg": {}},
        }
    )

    async def score(**kwargs):
        return {
            "score": 1.0,
            "query_cost": 0.25,
            "billing_receipt": {"provider_query_id": "q-missing-cost"},
        }

    manager = PolySegRewardManager(config, _Tokenizer(), score)
    result = asyncio.run(manager.run_single(_async_budget_data()))
    info = result["reward_extra_info"]
    assert info["segment_query_cost"] == 2.0
    assert all(item["status"] == "invalid_billed_cost" for item in info["billing_receipts"])

