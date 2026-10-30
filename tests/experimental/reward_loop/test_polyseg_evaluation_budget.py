"""Evaluation must still score after B_K is exhausted, without refilling it."""

import asyncio
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.polyseg import PolySegRewardManager as AsyncManager
from verl.workers.reward_manager.polyseg import PolySegRewardManager as SyncManager


class Tokenizer:
    def decode(self, ids, **kwargs):
        return " ".join(str(int(x)) for x in ids)


def sample(*, validation=False, marker="meta", skip=False):
    data = DataProto.from_single_dict({
        "prompts": torch.zeros((1, 2), dtype=torch.long),
        "responses": torch.arange(4).reshape(1, 4),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "response_mask": torch.ones((1, 4), dtype=torch.long),
        "data_source": np.asarray(["dummy"], dtype=object),
        "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
        "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1]}], dtype=object),
        "polyseg_adaptive_fallback": np.asarray(["skip" if skip else "none"], dtype=object),
    })
    if marker == "meta":
        data.meta_info["validate"] = validation
    else:
        data.non_tensor_batch["validate"] = np.asarray([validation])
    return data


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("marker", ["meta", "non_tensor"])
@pytest.mark.parametrize("scope", ["global", "trajectory"])
def test_evaluation_scores_every_segment_without_changing_training_budget(kind, marker, scope, tmp_path):
    calls = []

    def score(data_source, solution_str, ground_truth, extra_info):
        calls.append(extra_info["segment_id"])
        return {"score": 1.0, "acc": 1.0, "query_cost": 1.0}

    journal = tmp_path / "feedback.jsonl"
    kwargs = dict(m=2, feedback="binary_local", budget=2, budget_scope=scope, budget_fallback="skip",
                  feedback_journal_path=str(journal), feedback_attempt_id="test-attempt")
    if kind == "async":
        config = OmegaConf.create({"reward": {"reward_kwargs": kwargs}, "algorithm": {"polyseg": {}}})
        manager = AsyncManager(config, Tokenizer(), score)

        async def evaluate(data):
            return (await manager.run_single(data))["reward_extra_info"]
    else:
        manager = SyncManager(Tokenizer(), 0, compute_score=score, **kwargs)

        async def evaluate(data):
            info = manager(data, return_dict=True)["reward_extra_info"]
            return {key: value[0] for key, value in info.items()}

    async def run():
        training = await evaluate(sample())
        assert training["segment_query_cost"] == 2
        before = None if manager.global_budget is None else manager.global_budget.state_dict()
        # Even a stale training skip marker must not suppress held-out labels.
        evaluations = await asyncio.gather(*[
            evaluate(sample(validation=True, marker=marker, skip=True)) for _ in range(3)
        ])
        for info in evaluations:
            assert info["acc"] == 1.0
            assert info["segment_feedback_fallback_count"] == 0
            assert info["evaluation_query_cost"] == 2
            assert info["evaluation_query_count"] == 2
            assert info["feedback_budget_scope"] == "evaluation"
        after = None if manager.global_budget is None else manager.global_budget.state_dict()
        assert after == before
        training_after = await evaluate(sample())
        if scope == "global":
            assert training_after["segment_query_cost"] == 0
            assert training_after["acc"] is None
            assert len(calls) == 8
        else:
            assert training_after["segment_query_cost"] == 2
            assert len(calls) == 10

    asyncio.run(run())
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(rows) == 5 and len({row["event_id"] for row in rows}) == 5
    assert {row["attempt_id"] for row in rows} == {"test-attempt"}
    training = [row for row in rows if row["phase"] == "training"]
    evaluation = [row for row in rows if row["phase"] == "evaluation"]
    assert sum(row["query_cost"] for row in training) == (2 if scope == "global" else 4)
    assert sum(row["query_cost"] for row in evaluation) == 6
    assert all(row["global_budget"] is None for row in evaluation)
    if scope == "global":
        assert len({row["ledger_id"] for row in training}) == 1
        assert training[-1]["skipped_count"] == 2
        assert training[-1]["global_budget"] == {"total": 2, "consumed": 2, "reserved": 0}


def test_async_global_budget_passes_pre_reservation_capacity_to_scorer():
    seen = []

    def score(data_source, solution_str, ground_truth, extra_info):
        seen.append(float(extra_info["budget_remaining"]))
        assert float(extra_info["budget_remaining"]) in (pytest.approx(2.0), pytest.approx(1.0))
        return {"score": 1.0, "query_cost": 1.0}

    config = OmegaConf.create(
        {
            "reward": {
                "num_workers": 1,
                "reward_kwargs": {
                    "m": 2,
                    "feedback": "local_sum",
                    "budget": 2,
                    "budget_scope": "global",
                    "segment_cost": 1,
                },
            },
            "algorithm": {"polyseg": {}},
        }
    )
    manager = AsyncManager(config, Tokenizer(), score)

    async def run():
        result = await manager.run_single(sample())
        info = result["reward_extra_info"]
        assert info["segment_budget_remaining"] == pytest.approx(0.0)

    asyncio.run(run())
    assert seen == [2.0, 1.0]
