from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.polyseg import PolySegRewardManager as AsyncPolySegRewardManager
from verl.trainer.ppo.polyseg import (
    BudgetedPolySegController,
    PolynomialEstimatorState,
    PolynomialSegmentCritic,
    PolySegRegretTracker,
    SegmentRewardBudget,
    SharedStepRewardEstimator,
    compute_segment_rewards,
    resolve_polyseg_option,
)
from verl.workers.reward_manager.polyseg import PolySegRewardManager


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids)


def _data():
    return DataProto.from_single_dict({
        "prompts": torch.zeros((1, 2), dtype=torch.long),
        "responses": torch.arange(4).reshape(1, 4),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "response_mask": torch.ones((1, 4), dtype=torch.long),
        "data_source": np.asarray(["dummy"], dtype=object),
        "reward_model": np.asarray([{"ground_truth": "x"}], dtype=object),
        "extra_fields": np.asarray([{"segment_ids": [0, 0, 1, 1], "segment_durations": [1, 1]}], dtype=object),
    })


def test_mean_feedback_is_additive_training_scale_and_keeps_raw_mean():
    assert compute_segment_rewards([2.0, 4.0], "mean") == pytest.approx([1.0, 2.0])
    manager = PolySegRewardManager(
        _Tokenizer(), 0, lambda *args: {"score": [2.0, 4.0][args[3]["segment_id"]]},
        polyseg_config={"m": 2, "feedback": "mean"}, require_rollout_segments=True,
    )
    info = manager(_data(), return_dict=True)["reward_extra_info"]
    assert info["episode_reward"] == pytest.approx([3.0])
    assert info["segment_reward_sum"] == pytest.approx([3.0])
    assert info["segment_feedback_mean"] == pytest.approx([3.0])
    assert info["segment_rewards"][0].tolist() == pytest.approx([1.0, 2.0])


def test_async_mean_feedback_matches_sync_contract():
    async def score(**kwargs):
        return {"score": [2.0, 4.0][kwargs["extra_info"]["segment_id"]]}
    cfg = OmegaConf.create({"reward": {"reward_kwargs": {"m": 2, "feedback": "mean"}}, "algorithm": {"polyseg": {}}})
    info = __import__("asyncio").run(AsyncPolySegRewardManager(cfg, _Tokenizer(), score).run_single(_data()))["reward_extra_info"]
    assert info["episode_reward"] == pytest.approx(3.0)
    assert info["segment_feedback_mean"] == pytest.approx(3.0)
    assert np.asarray(info["segment_rewards"]).tolist() == pytest.approx([1.0, 2.0])


def test_algorithm_config_wins_over_rollout_compatibility_alias():
    cfg = OmegaConf.create({
        "algorithm": {"polyseg": {"feedback": "mean", "feature_dim": 7, "segment_cost": 3.0}},
        "actor_rollout_ref": {"rollout": {"polyseg_feedback": "sum", "polyseg_feature_dim": 4, "polyseg_segment_cost": 1.0}},
        "reward": {"reward_kwargs": {"feedback": None, "feature_dim": None, "segment_cost": None}},
    })
    assert resolve_polyseg_option(cfg, "feedback") == "mean"
    assert resolve_polyseg_option(cfg, "feature_dim") == 7
    assert resolve_polyseg_option(cfg, "segment_cost") == 3.0


def test_rollout_alias_remains_compatible_when_algorithm_value_unspecified():
    cfg = OmegaConf.create({
        "algorithm": {"polyseg": {}},
        "actor_rollout_ref": {"rollout": {"polyseg_feedback": "mean", "polyseg_feature_dim": 9, "polyseg_segment_cost": 2.0}},
        "reward": {"reward_kwargs": {"feedback": None, "feature_dim": None, "segment_cost": None}},
    })
    assert resolve_polyseg_option(cfg, "feedback") == "mean"
    assert resolve_polyseg_option(cfg, "feature_dim") == 9
    assert resolve_polyseg_option(cfg, "segment_cost") == 2.0


def test_adaptive_manager_uses_same_effective_runtime_options(monkeypatch):
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager
    monkeypatch.setattr(AgentLoopManager, "agent_loop_workers_class", object(), raising=False)
    cfg = OmegaConf.create({
        "actor_rollout_ref": {"model": {}, "rollout": {
            "polyseg_adaptive": True, "polyseg_q_candidates": [0], "polyseg_m_candidates": [1],
            "polyseg_feedback": "sum", "polyseg_feature_dim": 4, "polyseg_segment_cost": 1.0,
        }},
        "algorithm": {"polyseg": {"feedback": "mean", "feature_dim": 7, "segment_cost": 3.0}},
        "reward": {"reward_kwargs": {"feedback": None, "feature_dim": None, "segment_cost": None}},
    })
    manager = AgentLoopManager(cfg, llm_client=None)
    assert manager.polyseg_controller.segment_cost == pytest.approx(3.0)
    assert manager.polyseg_controller.bound_kwargs["r_max"] == pytest.approx(1.0)


def test_legacy_colocated_reward_commits_measured_cost_after_scoring():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class Controller:
        def __init__(self):
            self.calls = []

        def observe(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    class RewardLoop:
        reward_loop_worker_handles = None

        def compute_rm_score(self, batch):
            n = len(batch)
            return DataProto.from_single_dict({
                "responses": torch.zeros((n, 2), dtype=torch.long),
                "segment_query_cost": np.full(n, 2.0),
                "polyseg_adaptive_actual_cost": np.full(n, 2.0),
                "polyseg_adaptive_compute_cost": np.full(n, 0.5),
            })

    controller = Controller()
    trainer = object.__new__(RayPPOTrainer)
    trainer.reward_loop_manager = RewardLoop()
    trainer.agent_loop_manager = SimpleNamespace(polyseg_controller=controller)
    batch = DataProto.from_single_dict({"responses": torch.zeros((3, 2), dtype=torch.long)})
    trainer._compute_reward_colocate(batch)
    assert controller.calls == [((6.0,), {"compute_cost": 1.5, "episodes": 3})]


def test_checkpoint_restore_rejects_nonfinite_and_incompatible_estimator_state():
    estimator = PolynomialEstimatorState(q=1, feature_dim=1, ridge_lambda=0.5)
    with pytest.raises(ValueError, match="non-finite"):
        estimator.load_state_dict({**estimator.state_dict(), "gram": torch.tensor([[float("nan")]])})
    with pytest.raises(ValueError, match="does not match"):
        PolynomialEstimatorState(q=1, feature_dim=1, ridge_lambda=0.5).load_state_dict(
            {**estimator.state_dict(), "ridge_lambda": 1.0}
        )

    shared = SharedStepRewardEstimator(feature_dim=2, q=None)
    with pytest.raises(ValueError, match="parameter vector"):
        shared.load_state_dict({**shared.state_dict(), "theta": torch.tensor([0.0])})
    with pytest.raises(ValueError, match="non-finite"):
        shared.load_state_dict({**shared.state_dict(), "theta": torch.tensor([0.0, float("nan")])})

    critic = PolynomialSegmentCritic(q=1, feature_dim=1, ridge_lambda=0.5)
    with pytest.raises(ValueError, match="parameter vector"):
        critic.load_state_dict({**critic.state_dict(), "theta": torch.zeros(1)})
    with pytest.raises(ValueError, match="does not match"):
        critic.load_state_dict({**critic.state_dict(), "ridge_lambda": 1.0})
    binary = PolynomialSegmentCritic(q=1, feature_dim=1, feedback="binary")
    local_state = PolynomialEstimatorState(q=1, feature_dim=1, feedback="local_sum")
    with pytest.raises(ValueError, match="feedback"):
        binary.load_state_dict({**binary.state_dict(), "theta": torch.zeros(binary.dimension), "sufficient_state": local_state.state_dict()})

    with pytest.raises(ValueError, match="sample_count"):
        estimator.load_state_dict({**estimator.state_dict(), "sample_count": -0.5})
    with pytest.raises(ValueError, match="dependence_history_limit"):
        PolynomialEstimatorState.from_state_dict({**estimator.state_dict(), "dependence_history_limit": -0.5})
    shared = SharedStepRewardEstimator(feature_dim=2)
    with pytest.raises(ValueError, match="feature dimension"):
        shared.load_state_dict({**shared.state_dict(), "feature_dim": 2.5})


def test_checkpoint_restore_rejects_invalid_budget_controller_and_regret_state():
    controller = BudgetedPolySegController(q_candidates=[0], m_candidates=[1], total_budget=10.0)
    with pytest.raises(ValueError, match="counters"):
        BudgetedPolySegController.from_state_dict({**controller.state_dict(), "consumed_budget": -1.0})
    with pytest.raises(ValueError, match="selector_norm"):
        BudgetedPolySegController.from_state_dict({**controller.state_dict(), "selector_norm": "bad"})
    with pytest.raises(ValueError, match="q candidate"):
        BudgetedPolySegController.from_state_dict({**controller.state_dict(), "q_candidates": [0.5]})
    with pytest.raises(ValueError, match="bound"):
        BudgetedPolySegController.from_state_dict({**controller.state_dict(), "bound_kwargs": {"r_max": float("nan")}})

    budget = SegmentRewardBudget(3.0)
    with pytest.raises(ValueError, match="finite"):
        SegmentRewardBudget.from_state_dict({**budget.state_dict(), "consumed": float("nan")})
    with pytest.raises(ValueError, match="non-negative"):
        SegmentRewardBudget.from_state_dict({**budget.state_dict(), "reservations": {"r": -1.0}, "reserved": -1.0})
    with pytest.raises(ValueError, match="mappings"):
        SegmentRewardBudget.from_state_dict({**budget.state_dict(), "reservations": None})

    with pytest.raises(ValueError, match="regret tracker"):
        PolySegRegretTracker.from_state_dict({"segment_regret": float("nan")})
    with pytest.raises(ValueError, match="episodes"):
        PolySegRegretTracker.from_state_dict({"episodes": -1})
    with pytest.raises(ValueError, match="episodes"):
        PolySegRegretTracker.from_state_dict({"episodes": -0.5})
    with pytest.raises(ValueError, match="diagnostic"):
        from verl.trainer.ppo.polyseg import PolySegDependenceState
        dependence = PolySegDependenceState(q=1, feature_dim=1)
        PolySegDependenceState.from_state_dict({**dependence.state_dict(), "dependence_diagnostic_batches": -0.5})
