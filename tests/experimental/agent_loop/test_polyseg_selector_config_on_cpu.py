"""Verify the rollout manager uses the configured selector objective."""

import math
import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from types import SimpleNamespace

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.trainer.ppo.polyseg import BudgetedPolySegController
from verl.utils.polyseg_feedback import is_validation_batch
from verl.utils import tensordict_utils as tu


@pytest.mark.parametrize("stat_exponent", [1.0, 1.5])
def test_manager_forwards_selector_exponents_and_constants(monkeypatch, stat_exponent):
    # __init__ builds the real driver controller without starting workers.
    monkeypatch.setattr(AgentLoopManager, "agent_loop_workers_class", object(), raising=False)
    config = OmegaConf.create({
        "actor_rollout_ref": {
            "model": {},
            "rollout": {
                "polyseg_adaptive": True,
                "polyseg_selector_norm": "effective_round",
                "polyseg_q_candidates": [2],
                "polyseg_m_candidates": [4],
            },
        },
        "algorithm": {"polyseg": {
            "stat_exponent_u": stat_exponent,
            "approx_exponent_v": 2.5,
            "segment_exponent_gamma": 0.75,
            "dependence_chi": 0.25,
            "stat_constant": 2.0,
            "approx_constant": 3.0,
            "segment_constant": 4.0,
            "r_max": 0.1,
            "cost_penalty": 0.5,
        }},
    })
    manager = AgentLoopManager(config, llm_client=None)
    controller = manager.polyseg_controller
    assert controller.bound_kwargs == dict(config.algorithm.polyseg)
    choice = controller.choose(episodes=16, horizon=8, feature_dim=2, feedback="binary_local")
    dimension = math.comb(2 + 2, 2)
    # Unseen m=4 has kappa_seg=4; the effective round count is 16.
    statistical = 2 * math.exp(8 * 0.1 / (2 * 4)) * dimension**stat_exponent * math.sqrt(16)
    approximation = 3 * 16 * dimension**(stat_exponent - 0.5) * 3**(-2.5)
    segment = 4 * 16 * (8 / 4)**0.75
    assert choice.statistical_term == pytest.approx(statistical)
    assert choice.approximation_term == pytest.approx(approximation)
    assert choice.objective == pytest.approx(statistical + approximation + segment + 0.5 * 16 * 4)
    restored = BudgetedPolySegController.from_state_dict(controller.state_dict())
    assert restored.choose(episodes=16, horizon=8, feature_dim=2, feedback="binary_local") == choice


@pytest.mark.parametrize("transport", ["dataproto", "tensordict"])
@pytest.mark.parametrize("budget", [0, 32])
def test_validation_reuses_training_choice_without_mutating_selector(transport, budget):
    manager = object.__new__(AgentLoopManager)
    manager.rollout_config = OmegaConf.create({"response_length": 8, "polyseg_feature_dim": 1})
    manager.polyseg_controller = BudgetedPolySegController(
        q_candidates=[0, 1], m_candidates=[1, 2], total_budget=budget, budget_fallback="skip"
    )
    previous = manager.polyseg_controller.choose(episodes=8, horizon=8, feature_dim=1)
    before = copy.deepcopy(manager.polyseg_controller.state_dict())
    if transport == "dataproto":
        prompts = DataProto.from_single_dict({"index": np.arange(17)})
        prompts.meta_info["validate"] = True
    else:
        prompts = TensorDict({"index": torch.arange(17)}, batch_size=[17])
        tu.assign_non_tensor_data(prompts, "validate", True)
    choice = manager._prepare_polyseg_adaptive(prompts, validate=is_validation_batch(prompts))
    assert choice.m == previous.m and choice.q == previous.q
    assert manager.polyseg_controller.state_dict() == before
    assert manager.polyseg_controller.last_choice is previous
    values = prompts.non_tensor_batch if transport == "dataproto" else {
        key: list(prompts[key]) for key in ("polyseg_adaptive_m", "polyseg_adaptive_q", "polyseg_adaptive_fallback")
    }
    assert list(values["polyseg_adaptive_m"]) == [previous.m] * 17
    assert list(values["polyseg_adaptive_q"]) == [previous.q] * 17
    assert list(values["polyseg_adaptive_fallback"]) == ["none"] * 17


@pytest.mark.asyncio
async def test_validation_generation_does_not_observe_heldout_cost():
    manager = object.__new__(AgentLoopManager)
    manager.rollout_config = OmegaConf.create({"response_length": 8, "polyseg_feature_dim": 1})
    manager.polyseg_controller = BudgetedPolySegController(
        q_candidates=[0], m_candidates=[2], total_budget=0, budget_fallback="skip"
    )
    manager.polyseg_controller.choose(episodes=8, horizon=8, feature_dim=1)
    before = copy.deepcopy(manager.polyseg_controller.state_dict())
    prompts = DataProto.from_single_dict({"input_ids": torch.ones((17, 1), dtype=torch.long)})
    prompts.meta_info["validate"] = True

    async def remote(chunk):
        # Simulate expensive held-out scoring: observing this against B_K=0
        # would raise and must never be attempted by the real manager method.
        chunk.non_tensor_batch["segment_query_cost"] = np.full(len(chunk), 2.0)
        chunk.meta_info["metrics"] = [{} for _ in range(len(chunk))]
        return chunk

    manager.agent_loop_workers = [SimpleNamespace(generate_sequences=SimpleNamespace(remote=remote))]
    manager._performance_metrics = lambda metrics, output: {}
    result = await manager.generate_sequences(prompts)
    assert result.meta_info["validate"] is True
    assert result.non_tensor_batch["segment_query_cost"].sum() == 34
    assert manager.polyseg_controller.state_dict() == before
