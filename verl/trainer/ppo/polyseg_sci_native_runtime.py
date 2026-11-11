"""Consume OpenAI4S' native RL records at the external trainer boundary.

The harness owns execution and recording. This module preserves each decision's
actual prompt/response tensors, associates externally computed boundary values,
and builds an episode-mean, token-sum raw score-function objective. Host events
remain evidence; they never become additional policy actions.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_reinforce

CONTRACT_VERSION = "polyseg-native-openai4s-decisions-v1"


def native_trajectory_batch(
    trajectories: Sequence[Any],
    snapshots: Sequence[Any],
    boundary_values: Sequence[Any],
    *,
    policy_version: str,
    endpoint_weights: Sequence[float],
    pad_token_id: int,
) -> DataProto:
    """Build one row per native PolicyDecision, normalizing loss per episode.

    ``boundary_values[k]`` has shape [decisions+1, endpoints], in native decision
    order. Values must come from the trainer's estimator and terminal evaluator;
    reward annotations are never guessed or copied to intermediate boundaries.
    A DatasetSnapshot binds terminal execution facts, not restorable runtime
    state. This adapter does not certify continuation or endpoint validity.
    """
    from openai4s.rl.dataset import RolloutDataset

    if not trajectories or len(trajectories) != len(snapshots) or len(trajectories) != len(boundary_values):
        raise ValueError("nonempty trajectories, snapshots and boundary values must align")
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError("explicit behavior policy version is required")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    weights = torch.as_tensor(endpoint_weights, dtype=torch.float64)
    if weights.ndim != 1 or not weights.numel() or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("endpoint weights must be a finite nonnegative nonzero vector")
    weights = weights / weights.sum()
    rows = []
    episodes = set()
    for trajectory, snapshot, supplied_values in zip(trajectories, snapshots, boundary_values, strict=True):
        dataset = RolloutDataset.from_trajectory(trajectory, snapshot)
        episode = trajectory.episode.episode_id
        if episode in episodes:
            raise ValueError("duplicate episode")
        episodes.add(episode)
        if not dataset.samples:
            raise ValueError("episode has no policy decisions")
        values = torch.as_tensor(supplied_values, dtype=torch.float64).detach().cpu().clone()
        if values.shape != (len(dataset.samples) + 1, weights.numel()) or not torch.isfinite(values).all():
            raise ValueError("boundary values must match native decision and endpoint axes")
        turns = [sample.turn for sample in dataset.samples]
        if turns != sorted(set(turns)):
            raise ValueError("native decisions must be in unique chronological order")
        for index, sample in enumerate(dataset.samples):
            decision = sample.decision
            if decision["policy_version"] != policy_version:
                raise ValueError("behavior policy version mismatch")
            prompt, response = decision.get("prompt_token_ids"), decision.get("token_ids")
            logprobs, mask = decision.get("token_logprobs"), decision.get("loss_mask")
            if not prompt or not response or logprobs is None or mask is None:
                raise ValueError("native prompt/response tokens, sampling logprobs and loss mask are required")
            if any(type(token) is not int or token < 0 for token in (*prompt, *response)):
                raise ValueError("invalid native token IDs")
            if len(response) != len(logprobs) or len(response) != len(mask) or any(type(m) is not int or m not in (0, 1) for m in mask) or not any(mask):
                raise ValueError("invalid native policy-token alignment")
            if any(type(x) not in (int, float) or not math.isfinite(x) or x > 1e-6 for x in logprobs):
                raise ValueError("invalid sampling logprob")
            if not decision.get("sampling_config"):
                raise ValueError("sampling configuration must be explicit")
            rows.append((sample, values[index:index + 2], snapshot.to_dict()))

    batch_size = len(rows)
    prompt_length = max(len(sample.decision["prompt_token_ids"]) for sample, _, _ in rows)
    response_length = max(len(sample.decision["token_ids"]) for sample, _, _ in rows)
    prompts = torch.full((batch_size, prompt_length), pad_token_id, dtype=torch.long)
    responses = torch.full((batch_size, response_length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((batch_size, prompt_length + response_length), dtype=torch.long)
    response_mask = torch.zeros((batch_size, response_length), dtype=torch.bool)
    old_logprobs = torch.zeros((batch_size, response_length), dtype=torch.float64)
    advantages = torch.zeros_like(old_logprobs)
    ids = torch.full((batch_size, response_length), -1, dtype=torch.long)
    for row, (sample, values, _) in enumerate(rows):
        decision = sample.decision
        p, r = len(decision["prompt_token_ids"]), len(decision["token_ids"])
        prompts[row, -p:] = torch.tensor(decision["prompt_token_ids"])
        responses[row, :r] = torch.tensor(decision["token_ids"])
        attention[row, prompt_length - p:prompt_length + r] = 1
        response_mask[row, :r] = torch.tensor(decision["loss_mask"], dtype=torch.bool)
        old_logprobs[row, :r] = torch.tensor(decision["token_logprobs"], dtype=torch.float64)
        advantages[row, :r] = ((values[1] - values[0]) @ weights) * response_mask[row, :r]
        ids[row, response_mask[row]] = 0
    tensors = {
        "prompts": prompts, "responses": responses,
        "input_ids": torch.cat((prompts, responses), dim=1),
        "attention_mask": attention, "position_ids": (attention.cumsum(-1) - 1).clamp_min(0),
        "response_mask": response_mask, "old_log_probs": old_logprobs,
        "advantages": advantages.detach(), "segment_ids": ids,
        "segment_mask": torch.ones((batch_size, 1), dtype=torch.bool),
        "science_boundary_values": torch.stack([values for _, values, _ in rows]),
    }
    objects = {
        "episode_id": np.array([s.episode_id for s, _, _ in rows], dtype=object),
        "decision_id": np.array([s.decision_id for s, _, _ in rows], dtype=object),
        "policy_version": np.array([s.decision["policy_version"] for s, _, _ in rows], dtype=object),
        "native_facts": np.array([s.to_dict() for s, _, _ in rows], dtype=object),
        "dataset_snapshot": np.array([snapshot for _, _, snapshot in rows], dtype=object),
    }
    return DataProto(
        batch=TensorDict(tensors, batch_size=[batch_size]), non_tensor_batch=objects,
        meta_info={"contract_version": CONTRACT_VERSION, "episode_count": len(episodes),
                   "decision_count": batch_size, "endpoint_weights": weights.tolist(),
                   "loss_normalization": "episode-mean-token-sum", "continuation_verified": False},
    )


def native_outcome_reinforce_batch(
    trajectories: Sequence[Any], snapshots: Sequence[Any], endpoint_vectors: Sequence[Any],
    *, policy_version: str, endpoint_weights: Sequence[float], pad_token_id: int,
) -> DataProto:
    """Explicit S1 outcome-only baseline over all policy decisions.

    This is standard episodic REINFORCE, not intermediate value estimation.
    No science_boundary_values are emitted, since no such estimates exist.
    """
    if len(trajectories) != len(endpoint_vectors):
        raise ValueError("one observed terminal vector is required per episode")
    endpoints = [torch.as_tensor(v, dtype=torch.float64).detach().cpu() for v in endpoint_vectors]
    if any(v.shape != (len(endpoint_weights),) or not torch.isfinite(v).all() for v in endpoints):
        raise ValueError("observed endpoints must match the registered vector schema")
    # Reuse native token transport/validation; these zero placeholders are
    # removed before returning and are never presented as estimated values.
    placeholders = [torch.zeros((len(t.decisions) + 1, len(endpoint_weights))) for t in trajectories]
    data = native_trajectory_batch(trajectories, snapshots, placeholders, policy_version=policy_version,
                                   endpoint_weights=endpoint_weights, pad_token_id=pad_token_id)
    del data.batch["science_boundary_values"]
    weights = torch.tensor(data.meta_info["endpoint_weights"], dtype=torch.float64)
    values = {t.episode.episode_id: v @ weights for t, v in zip(trajectories, endpoints, strict=True)}
    for row, episode in enumerate(data.non_tensor_batch["episode_id"]):
        data.batch["advantages"][row] = values[episode] * data.batch["response_mask"][row]
    data.meta_info["estimator_variant"] = "outcome_only_reinforce"
    data.meta_info["initial_baseline"] = 0.0
    return data


def raw_native_policy_loss(current_logprobs: torch.Tensor, data: DataProto, *, behavior_logprob_atol: float = 1e-6) -> torch.Tensor:
    """One raw gradient evaluation on an unsplit complete-episode batch.

    No clipping or length normalization; detach estimator outputs and average
    over trajectories, not decisions. Distributed/minibatch callers need a
    separately supplied global episode count and must not use this helper.
    """
    if data.meta_info.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("native runtime contract is missing")
    episodes = len(set(data.non_tensor_batch["episode_id"]))
    if episodes != data.meta_info.get("episode_count") or len(data) != data.meta_info.get("decision_count"):
        raise ValueError("raw loss requires the complete unsplit episode batch")
    if current_logprobs.shape != data.batch["old_log_probs"].shape:
        raise ValueError("current logprobs must match native decision tokens")
    mask = data.batch["response_mask"].to(current_logprobs.device)
    behavior = data.batch["old_log_probs"].to(current_logprobs.device)
    if not torch.isfinite(current_logprobs[mask]).all():
        raise ValueError("current policy logprobs must be finite")
    if not math.isfinite(behavior_logprob_atol) or not 0 <= behavior_logprob_atol <= 0.01:
        raise ValueError("behavior logprob tolerance must be explicit and at most 0.01")
    if not torch.allclose(current_logprobs.detach()[mask].double(), behavior[mask].double(), atol=behavior_logprob_atol, rtol=0):
        raise ValueError("raw native update requires the unchanged behavior policy; policy lag needs an explicit estimator")
    config = SimpleNamespace(global_batch_info={"dp_size": 1, "batch_num_tokens": None,
                                               "global_batch_size": episodes, "loss_scale_factor": None})
    loss, _ = compute_policy_loss_reinforce(
        behavior, current_logprobs,
        data.batch["advantages"].detach().to(current_logprobs.device),
        mask, "seq-mean-token-sum", config,
    )
    return loss
