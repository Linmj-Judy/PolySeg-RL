"""Synchronous PolySeg reward manager for verl's legacy trainer.

The manager owns reward semantics, while the AgentLoop owns segment
boundaries. Fixed token partitions remain available only for the explicit
legacy/smoke path (``require_rollout_segments=False``).
"""

from __future__ import annotations

from collections import defaultdict
import copy
import uuid
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.polyseg_feedback import (
    billing_receipt_journal_entries,
    is_validation_batch,
    settle_feedback_cost,
)
from verl.utils.polyseg_journal import append_feedback_record
from verl.trainer.ppo.polyseg import (
    FileBackedSegmentRewardBudget,
    SegmentRewardBudget,
    build_segment_features,
    compute_segment_rewards,
    decode_token_ids,
    extract_segment_metadata,
    feature_map_contract,
    feedback_protocol_contract,
    normalize_feedback_protocol,
    resolve_polyseg_option,
)
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _scalar(value: Any, default: Any):
    """Normalize a per-sample transport value to a Python scalar."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    return default if value is None else value


def _resource_accounting(item, *, prompt_length: int, valid_mask: torch.Tensor,
                         action_mask: torch.Tensor, extra: dict[str, Any]) -> dict[str, int]:
    """Return a per-trajectory realized token/tool ledger for the journal."""
    attention = item.batch["attention_mask"][:prompt_length].to(torch.bool).cpu()
    prompt_tokens = int(attention.sum().item())
    response_tokens = int(valid_mask.to(torch.bool).sum().item())
    action_tokens = int((action_mask.to(torch.bool) & valid_mask.to(torch.bool)).sum().item())
    tool_calls_value = extra.get("tool_call_count")
    if tool_calls_value is None:
        calls = extra.get("segment_tool_calls", [])
        tool_calls_value = sum(len(value) for value in calls) if isinstance(calls, (list, tuple)) else 0
    tool_calls = int(_scalar(tool_calls_value, 0))
    successes = int(_scalar(extra.get("tool_call_success_count"), 0))
    failures_value = extra.get("tool_call_failure_count")
    failures = int(_scalar(failures_value, max(tool_calls - successes, 0)))
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "action_tokens": action_tokens,
        "tool_observation_tokens": response_tokens - action_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "tool_calls": tool_calls,
        "tool_successes": successes,
        "tool_failures": failures,
    }


@register("polyseg")
class PolySegRewardManager(AbstractRewardManager):
    """Compute one verifier feedback value for each rollout-owned segment."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        cfg = kwargs.get("polyseg_config", {}) or {}

        def get_value(key, default):
            if kwargs.get(key) is not None:
                return kwargs[key]
            if cfg.get(key) is not None:
                return cfg[key]
            return resolve_polyseg_option(config, key, default=default)

        config = kwargs.get("config")
        rollout_config = getattr(getattr(config, "actor_rollout_ref", None), "rollout", None)
        configured_horizon = getattr(rollout_config, "response_length", None)
        if configured_horizon is None and isinstance(rollout_config, dict):
            configured_horizon = rollout_config.get("response_length")
        raw_horizon = get_value("feature_horizon", configured_horizon)
        self.feature_horizon = None if raw_horizon is None else int(raw_horizon)

        self.m = int(get_value("m", 4))
        self.q = int(get_value("q", 2))
        self.feature_dim = int(get_value("feature_dim", 4))
        if self.feature_horizon is not None and self.feature_horizon < 1:
            # Direct unit-test/legacy construction has no trainer config.  In
            # that case the response width is used per call, while real
            # trainer runs always pass the configured rollout horizon.
            self.feature_horizon = None
        self.protocol = normalize_feedback_protocol(get_value("feedback", "local_sum"))
        self.feedback_contract = feedback_protocol_contract(self.protocol)
        self.budget_limit = get_value("budget", None)
        self.budget_scope = str(get_value("budget_scope", "trajectory")).strip().lower()
        self.segment_cost = float(get_value("segment_cost", 1.0))
        self.budget_fallback = str(get_value("budget_fallback", "error")).strip().lower()
        self.require_rollout_segments = bool(get_value("require_rollout_segments", False))
        self.decode_skip_special_tokens = bool(get_value("decode_skip_special_tokens", True))
        self.feedback_cost_policy = str(
            get_value("feedback_cost_policy", "reserve_upper")
        ).strip().lower()
        if self.feedback_cost_policy not in {"reserve_upper", "reported_with_receipt"}:
            raise ValueError(
                "feedback_cost_policy must be 'reserve_upper' or 'reported_with_receipt'"
            )
        self.feedback_journal_path = get_value("feedback_journal_path", None)
        self.feedback_ledger_path = get_value("feedback_ledger_path", None)
        self.feedback_attempt_id = get_value("feedback_attempt_id", None)
        self.feedback_ledger_id = uuid.uuid4().hex
        if self.budget_scope not in {"trajectory", "global"}:
            raise ValueError("PolySeg budget_scope must be 'trajectory' or 'global'")
        # Legacy reward workers are independent processes.  A process-local
        # global budget would silently multiply B by the worker count, so
        # reject that combination until a central atomic ledger is supplied.
        config = kwargs.get("config")
        reward_cfg = getattr(config, "reward", None) if config is not None else None
        reward_workers = int(reward_cfg.get("num_workers", 1)) if reward_cfg is not None and hasattr(reward_cfg, "get") else 1
        if self.budget_scope == "global" and reward_workers > 1 and not self.feedback_ledger_path:
            raise ValueError(
                "PolySeg global budget with multiple legacy workers requires "
                "reward.reward_kwargs.feedback_ledger_path"
            )
        self.global_budget = (
            (
                FileBackedSegmentRewardBudget(
                    self.feedback_ledger_path,
                    float(self.budget_limit),
                    cost_per_query=self.segment_cost,
                )
                if self.feedback_ledger_path
                else SegmentRewardBudget(float(self.budget_limit), cost_per_query=self.segment_cost)
            )
            if self.budget_limit is not None and self.budget_scope == "global"
            else None
        )
        if self.global_budget is not None and hasattr(self.global_budget, "ledger_id"):
            self.feedback_ledger_id = self.global_budget.ledger_id

        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.segment_cost <= 0:
            raise ValueError("segment_cost must be > 0")

    def _billed_cost(self, reported_cost: float, info: dict[str, Any]) -> float:
        """Return the amount to charge for one completed query."""
        return settle_feedback_cost(
            info,
            reported_cost=reported_cost,
            reserved_cost=self.segment_cost,
            policy=self.feedback_cost_policy,
        )

    @staticmethod
    def _value(result: Any, *, protocol: str) -> tuple[float, dict[str, Any]]:
        if isinstance(result, dict):
            if protocol in {"utility_delta", "prm_potential"}:
                key = "utility" if "utility" in result else None
            else:
                key = "score" if "score" in result else "utility" if "utility" in result else None
            if key is None:
                expected = "utility" if protocol == "utility_delta" else "score"
                raise KeyError(f"PolySeg {protocol} scorer must return '{expected}'")
            return float(result[key]), dict(result)
        if protocol in {"utility_delta", "prm_potential"}:
            raise TypeError("PolySeg utility/potential scorer must return {'utility': ...}; a scalar is ambiguous")
        value = float(result)
        return value, {"score": value}

    @staticmethod
    def _valid_masks(item, response_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        attention = item.batch["attention_mask"]
        prompt_length = item.batch["prompts"].shape[-1]
        valid = attention[prompt_length : prompt_length + response_length].to(torch.bool).cpu()
        response_mask = item.batch.get("response_mask")
        if response_mask is None:
            action = valid.clone()
        else:
            action = response_mask[:response_length].to(torch.bool).cpu() & valid
        return valid, action

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if not is_validation_batch(data):
            return self._compute_batch(data, return_dict=return_dict)
        evaluation_manager = copy.copy(self)
        evaluation_manager.budget_limit = None
        evaluation_manager.global_budget = None
        result = evaluation_manager._compute_batch(data, return_dict=True, evaluation=True)
        details = result["reward_extra_info"]
        details["evaluation_query_cost"] = list(details["segment_query_cost"])
        details["evaluation_query_count"] = list(details["segment_query_count"])
        details["feedback_budget_scope"] = ["evaluation"] * len(data)
        details["training_budget_remaining"] = [
            None if self.global_budget is None else self.global_budget.remaining
        ] * len(data)
        return result if return_dict else result["reward_tensor"]

    def _compute_batch(self, data: DataProto, *, return_dict: bool, evaluation: bool = False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extras = defaultdict(list)
        per_sample_segment_data: list[dict[str, Any]] = []

        for i in range(len(data)):
            item = data[i]
            response_ids = item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_mask, action_mask = self._valid_masks(item, response_length)
            metadata = extract_segment_metadata(
                item,
                response_length=response_length,
                valid_response_mask=valid_mask,
                action_mask=action_mask,
                fallback_m=int(_scalar(item.non_tensor_batch.get("polyseg_adaptive_m"), self.m)),
                require_rollout_segments=self.require_rollout_segments,
            )
            segment_ids = metadata["segment_ids"]
            boundaries = metadata["segment_boundaries"]
            segment_durations = metadata["segment_durations"]
            segment_count = len(boundaries)
            metadata_from_rollout = bool(metadata.get("from_rollout", False))

            valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
            valid_response_ids = response_ids.detach().cpu()[valid_positions]
            response_text = decode_token_ids(
                self.tokenizer, valid_response_ids, skip_special_tokens=self.decode_skip_special_tokens
            )
            data_source = item.non_tensor_batch[self.reward_fn_key]
            reward_model = item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model["ground_truth"]
            base_extra = dict(item.non_tensor_batch.get("extra_info", {}) or {})
            raw_prompt = item.non_tensor_batch.get("raw_prompt")
            raw_prompt = raw_prompt.item() if isinstance(raw_prompt, np.ndarray) and raw_prompt.ndim == 0 else raw_prompt
            if raw_prompt is not None:
                base_extra["raw_prompt"] = raw_prompt
            extra_fields = item.non_tensor_batch.get("extra_fields")
            if extra_fields is None:
                extra_fields = item.non_tensor_batch.get("tool_extra_fields", {})
            if isinstance(extra_fields, np.ndarray) and extra_fields.ndim == 0:
                extra_fields = extra_fields.item()
            if isinstance(extra_fields, dict):
                base_extra.update(extra_fields)
            # ``selected_m`` is a planning cap; the measured event count is
            # the actual macro count after termination/merging.
            actual_m = int(segment_count)
            m_semantics = str(base_extra.get("polyseg_m_semantics", "fixed_rollout"))
            # AgentLoop places rollout-owned ragged fields directly in the
            # non-tensor batch after padding.  Merge them explicitly here;
            # otherwise successor features would be silently dropped unless
            # a dataset happened to duplicate them inside ``extra_info``.
            for key in (
                "segment_features",
                "step_features",
                "segment_next_features",
                "segment_transition_mask",
                "segment_action_lengths",
                "segment_tool_calls",
                "segment_tool_observations",
                "tool_observation_mask",
                "tool_observation_spans",
                "segment_tool_observation_spans",
                "segment_action_masks",
                "segment_action_mask",
                "tool_call_count",
                "tool_call_success_count",
                "tool_call_failure_count",
                "trajectory_id",
                "episode_id",
            ):
                value = item.non_tensor_batch.get(key)
                if value is None:
                    try:
                        value = item.batch.get(key)
                    except (AttributeError, KeyError):
                        value = None
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                if key in {"trajectory_id", "episode_id"} and isinstance(value, np.ndarray) and value.size == 1:
                    value = value.reshape(-1)[0]
                    value = value.item() if hasattr(value, "item") else value
                if value is not None:
                    base_extra[key] = value
            selected_m = int(_scalar(item.non_tensor_batch.get("polyseg_adaptive_m"), base_extra.get("polyseg_adaptive_m", self.m)))
            selected_q = int(_scalar(item.non_tensor_batch.get("polyseg_adaptive_q"), base_extra.get("polyseg_adaptive_q", self.q)))
            planned_cost = float(
                _scalar(
                    item.non_tensor_batch.get("polyseg_adaptive_planned_cost"),
                    base_extra.get("polyseg_adaptive_planned_cost", selected_m * self.segment_cost),
                )
            )
            adaptive_fallback = str(
                _scalar(
                    item.non_tensor_batch.get("polyseg_adaptive_fallback"),
                    base_extra.get("polyseg_adaptive_fallback", "none"),
                )
            )
            adaptive_fallback_reason = str(
                _scalar(
                    item.non_tensor_batch.get("polyseg_adaptive_fallback_reason"),
                    base_extra.get("polyseg_adaptive_fallback_reason", ""),
                )
            )
            if evaluation:
                adaptive_fallback, adaptive_fallback_reason = "none", ""
            base_extra.update(
                {
                    "polyseg_protocol": self.protocol,
                    "polyseg_m": selected_m,
                    "polyseg_q": selected_q,
                    "polyseg_adaptive_m": selected_m,
                    "polyseg_adaptive_q": selected_q,
                    "polyseg_adaptive_planned_cost": planned_cost,
                    "polyseg_adaptive_fallback": adaptive_fallback,
                    "polyseg_adaptive_fallback_reason": adaptive_fallback_reason,
                    "polyseg_segment_metadata_source": "rollout" if metadata_from_rollout else "fixed_token_fallback",
                }
            )
            local_budget = (
                SegmentRewardBudget(self.budget_limit, cost_per_query=self.segment_cost)
            if self.budget_limit is not None and self.budget_scope == "trajectory"
                else None
            )

            values: list[float] = []
            infos: list[dict[str, Any]] = []
            query_count = 0
            query_cost_total = 0.0
            fallback_segments: list[int] = []
            if adaptive_fallback not in {"none", "skip"}:
                raise ValueError(f"unsupported PolySeg adaptive fallback: {adaptive_fallback!r}")
            for segment_id, (start, end) in enumerate(boundaries):
                if adaptive_fallback == "skip":
                    values.append(0.0)
                    infos.append(
                        {
                            "feedback_status": "skipped_budget",
                            "fallback_reason": adaptive_fallback_reason,
                            "query_cost": 0.0,
                        }
                    )
                    fallback_segments.append(segment_id)
                    continue
                if self.protocol == "terminal" and segment_id < segment_count - 1:
                    values.append(0.0)
                    infos.append({"feedback_status": "unqueried_terminal", "query_cost": 0.0})
                    continue
                if local_budget is not None and not local_budget.can_afford(cost=self.segment_cost):
                    if self.budget_fallback == "skip":
                        values.append(0.0)
                        infos.append({"feedback_status": "skipped_budget", "query_cost": 0.0})
                        fallback_segments.append(segment_id)
                        continue
                    raise RuntimeError("PolySeg per-trajectory feedback budget exhausted before segment query")
                reservation_token = None
                if self.global_budget is not None:
                    try:
                        trajectory_id = base_extra.get("trajectory_id", base_extra.get("episode_id"))
                        reservation_token = self.global_budget.reserve(
                            self.segment_cost,
                            reservation_id=(
                                None if trajectory_id in (None, "") else f"{trajectory_id}:segment:{segment_id}"
                            ),
                        )
                    except RuntimeError:
                        if self.budget_fallback == "skip":
                            values.append(0.0)
                            infos.append({"feedback_status": "skipped_budget", "query_cost": 0.0})
                            fallback_segments.append(segment_id)
                            continue
                        raise
                # ``reserve`` already removes this query from the ledger's
                # public remaining balance.  A scorer may still need to
                # validate the full cost of the in-flight request (the exact
                # budget boundary is a legal case), so expose the balance
                # available before this reservation rather than the post-hold
                # value.  The reservation itself remains the authoritative
                # concurrency guard and is settled below.
                scorer_budget_remaining = (
                    self.global_budget.remaining + self.segment_cost
                    if self.global_budget is not None
                    else None if local_budget is None else local_budget.remaining
                )
                segment_positions = torch.arange(start, end, dtype=torch.long)
                segment_positions = segment_positions[valid_mask[segment_positions]]
                segment_ids_slice = response_ids.detach().cpu()[segment_positions]
                segment_text = decode_token_ids(
                    self.tokenizer, segment_ids_slice, skip_special_tokens=self.decode_skip_special_tokens
                )
                prefix_positions = valid_positions[valid_positions <= segment_positions[-1]] if len(segment_positions) else valid_positions
                prefix_text = decode_token_ids(
                    self.tokenizer, response_ids.detach().cpu()[prefix_positions], skip_special_tokens=self.decode_skip_special_tokens
                )
                extra = dict(base_extra)
                tool_observations = base_extra.get("segment_tool_observations", [])
                tool_calls = base_extra.get("segment_tool_calls", [])
                if isinstance(tool_observations, (list, tuple)) and segment_id < len(tool_observations):
                    tool_observations = tool_observations[segment_id]
                if isinstance(tool_calls, (list, tuple)) and segment_id < len(tool_calls):
                    tool_calls = tool_calls[segment_id]
                extra.update(
                    {
                        "segment_id": segment_id,
                        "segment_start": int(start),
                        "segment_end": int(end),
                        "segment_boundary": (int(start), int(end)),
                        "segment_duration": float(segment_durations[segment_id]),
                        "segment_count": segment_count,
                        "segment_text": segment_text,
                        "segment_prefix": prefix_text,
                        "segment_action_mask": action_mask[start:end].numpy(),
                        "segment_response_mask": valid_mask[start:end].numpy(),
                        "full_response": response_text,
                        "tool_observations": tool_observations,
                        "tool_calls": tool_calls,
                        "polyseg_protocol": self.protocol,
                        "budget_remaining": scorer_budget_remaining,
                        "budget_reservation_id": reservation_token,
                    }
                )
                try:
                    result = self.compute_score(data_source, segment_text, ground_truth, extra)
                except Exception:
                    # A raised exception is not evidence that the provider did
                    # not bill the request.  The conservative default keeps the
                    # full reservation charged; only an explicit pre-submit
                    # failure may be refunded by the caller.
                    if self.global_budget is not None and reservation_token is not None:
                        self.global_budget.commit(reservation_token, cost=self.segment_cost)
                    raise
                try:
                    value, info = self._value(result, protocol=self.protocol)
                except Exception:
                    if self.global_budget is not None and reservation_token is not None:
                        self.global_budget.commit(reservation_token, cost=self.segment_cost)
                    raise
                info.setdefault("feedback_protocol_contract", dict(self.feedback_contract))
                info.setdefault("budget_reservation_id", reservation_token)
                raw_cost = info.get("query_cost", self.segment_cost)
                try:
                    query_cost = float(raw_cost)
                except (TypeError, ValueError) as exc:
                    if self.global_budget is not None and reservation_token is not None:
                        self.global_budget.commit(reservation_token, cost=self.segment_cost)
                    raise ValueError("PolySeg scorer query_cost must be a finite non-negative number") from exc
                if not np.isfinite(query_cost) or query_cost < 0:
                    if self.global_budget is not None and reservation_token is not None:
                        self.global_budget.commit(reservation_token, cost=self.segment_cost)
                    raise ValueError("PolySeg scorer query_cost must be a finite non-negative number")
                if query_cost > self.segment_cost + 1e-8:
                    if self.global_budget is not None and reservation_token is not None:
                        self.global_budget.commit(reservation_token, cost=self.segment_cost)
                    raise ValueError(
                        "PolySeg scorer query_cost exceeds configured segment_cost; "
                        "set segment_cost to the maximum verifier/suffix query cost"
                    )
                # A scorer-reported lower cost is only an authoritative bill
                # when it carries a provider receipt and the configured policy
                # allows it.  Otherwise the reserved upper bound is charged,
                # which prevents a buggy/hostile scorer from creating budget.
                billed_cost = self._billed_cost(query_cost, info)
                if local_budget is not None and not local_budget.can_afford(cost=billed_cost):
                    if self.global_budget is not None and reservation_token is not None:
                        self.global_budget.commit(reservation_token, cost=billed_cost)
                    raise RuntimeError(
                        "PolySeg scorer exceeded the per-trajectory feedback budget: "
                        f"requested={billed_cost}, remaining={local_budget.remaining}"
                    )
                values.append(value)
                infos.append(info)
                query_count += 1
                query_cost_total += billed_cost
                if local_budget is not None:
                    local_budget.consume(cost=billed_cost)
                if self.global_budget is not None and reservation_token is not None:
                    self.global_budget.commit(reservation_token, cost=billed_cost)

            raw_feedback_mean = float(np.mean(values)) if values else 0.0
            rewards = compute_segment_rewards(values, self.protocol)
            total_reward = float(sum(rewards))
            # Legacy token reward is a compatibility projection. PolySeg's
            # estimator reads segment_rewards; each segment reward is written
            # once at its final valid action token.
            for segment_id, (start, end) in enumerate(boundaries):
                positions = torch.arange(start, end, dtype=torch.long)
                positions = positions[action_mask[positions]]
                if positions.numel():
                    reward_tensor[i, int(positions[-1])] = rewards[segment_id]

            rollout_features = base_extra.get("segment_features")
            if rollout_features is not None:
                segment_features = np.asarray(rollout_features, dtype=np.float32)
                if segment_features.ndim == 3 and segment_features.shape[0] == 1:
                    segment_features = segment_features[0]
                if segment_features.ndim != 2 or segment_features.shape != (segment_count, self.feature_dim):
                    raise ValueError(
                        "rollout segment_features must have shape "
                        f"({segment_count}, {self.feature_dim})"
                    )
                if not np.isfinite(segment_features).all():
                    raise ValueError("rollout segment_features must be finite")
            else:
                segment_features = build_segment_features(
                            boundaries,
                            segment_durations,
                            response_length=response_length,
                            action_lengths=base_extra.get("segment_action_lengths"),
                            tool_call_counts=[
                                len(calls) if isinstance(calls, (list, tuple)) else 0
                                for calls in (
                                    base_extra.get("segment_tool_calls", [])
                                    if isinstance(base_extra.get("segment_tool_calls", []), (list, tuple))
                                    else []
                                )
                            ]
                            or [0] * segment_count,
                            feature_dim=self.feature_dim,
                            feature_horizon=self.feature_horizon,
                        ).numpy().astype(np.float32)
            representation_hash = _scalar(base_extra.get("polyseg_representation_hash"), None)
            feature_contract = (
                {
                    "version": "polyseg-frozen-lowdim-v1",
                    "hash": str(representation_hash),
                    "semantic_role": "frozen_rollout_representation",
                    "causal_availability": "available_at_segment_boundary",
                    "markov_sufficiency": "not_established",
                }
                if representation_hash
                else feature_map_contract(
                    response_length=self.feature_horizon or response_length,
                    feature_dim=self.feature_dim,
                )
            )
            per_sample_segment_data.append(
                {
                    "segment_rewards": np.asarray(rewards, dtype=np.float32),
                    "segment_durations": segment_durations.numpy().astype(np.float32),
                    "segment_mask": np.ones(segment_count, dtype=np.bool_),
                    "segment_feedback_mask": np.asarray(
                        [info.get("feedback_status") not in {"skipped_budget", "unqueried_terminal"} for info in infos], dtype=np.bool_
                    ),
                    "segment_ids": segment_ids.numpy().astype(np.int64),
                    "segment_features": segment_features,
                    **(
                        {"polyseg_representation_hash": str(representation_hash)}
                        if representation_hash
                        else {}
                    ),
                    **(
                        {"step_features": np.asarray(base_extra["step_features"], dtype=np.float32)}
                        if "step_features" in base_extra
                        else {}
                    ),
                    "polyseg_feature_map_version": feature_contract["version"],
                    "polyseg_feature_map_hash": feature_contract["hash"],
                    "polyseg_feature_semantic_role": feature_contract["semantic_role"],
                    "polyseg_feature_causal_availability": feature_contract["causal_availability"],
                    "polyseg_feature_markov_sufficiency": feature_contract["markov_sufficiency"],
                    # A theorem-aligned transition estimator requires
                    # rollout-owned next-boundary features.  Do not synthesize
                    # them by shifting the current segment summaries.
                    "segment_next_features": np.asarray(
                        base_extra.get("segment_next_features", np.zeros((segment_count, self.feature_dim), dtype=np.float32)),
                        dtype=np.float32,
                    ),
                    "segment_transition_mask": np.asarray(
                        base_extra.get("segment_transition_mask", np.zeros(segment_count, dtype=np.bool_)),
                        dtype=np.bool_,
                    ),
                    "polyseg_transition_source": (
                        "rollout" if "segment_next_features" in base_extra else "missing"
                    ),
                }
            )
            final_info = infos[-1] if infos else {}
            extras["episode_reward"].append(total_reward)
            extras["segment_reward_sum"].append(float(sum(rewards)))
            extras["segment_reward_mean"].append(float(np.mean(rewards)) if rewards else 0.0)
            extras["segment_feedback_mean"].append(raw_feedback_mean)
            # ``score``/utility and observed correctness are distinct
            # quantities.  Do not fill an absent correctness field with the
            # scorer's utility or with zero; missing remains auditable.
            final_accuracy_value = final_info.get("acc", final_info.get("accuracy"))
            final_accuracy = None if final_accuracy_value is None else float(final_accuracy_value)
            extras["final_accuracy"].append(final_accuracy)
            extras["acc"].append(final_accuracy)
            extras["segment_count"].append(segment_count)
            extras["polyseg_adaptive_actual_m"].append(actual_m)
            extras["polyseg_m_semantics"].append(m_semantics)
            extras["polyseg_transition_source"].append(
                "rollout" if "segment_next_features" in base_extra else "missing"
            )
            extras["polyseg_feature_map_version"].append(feature_contract["version"])
            extras["polyseg_feature_map_hash"].append(feature_contract["hash"])
            extras["polyseg_segment_metadata_source"].append(
                "rollout" if metadata_from_rollout else "fixed_token_fallback"
            )
            extras["segment_query_count"].append(query_count)
            extras["segment_feedback_fallback_segments"].append(list(fallback_segments))
            extras["segment_feedback_fallback_count"].append(len(fallback_segments))
            extras["segment_query_cost"].append(float(query_cost_total))
            prm_scores = [float(info["prm_score"]) for info in infos if "prm_score" in info]
            extras["prm_score_mean"].append(float(np.mean(prm_scores)) if prm_scores else None)
            extras["prm_score_final"].append(float(prm_scores[-1]) if prm_scores else None)
            extras["model_derived_pseudo_label"].append(bool(prm_scores))
            extras["prm_model_path"].append(str(infos[-1].get("prm_model_path", "")) if prm_scores else "")
            extras["polyseg_adaptive_actual_cost"].append(float(query_cost_total))
            extras["polyseg_adaptive_planned_cost"].append(planned_cost)
            extras["polyseg_adaptive_fallback"].append(adaptive_fallback)
            extras["polyseg_adaptive_fallback_reason"].append(adaptive_fallback_reason)
            extras["polyseg_adaptive_q"].append(selected_q)
            extras["polyseg_adaptive_m"].append(selected_m)
            extras["polyseg_adaptive_remaining_budget"].append(
                _scalar(item.non_tensor_batch.get("polyseg_adaptive_remaining_budget"), None)
            )
            extras["segment_budget_remaining"].append(
                self.global_budget.remaining
                if self.global_budget is not None
                else None if local_budget is None else local_budget.remaining
            )
            extras["segment_info"].append(
                [
                    {"segment_id": j, "reward": rewards[j], "value": values[j], **infos[j]}
                    for j in range(segment_count)
                ]
            )
            extras["feedback_protocol_contract"].append(dict(self.feedback_contract))
            extras.setdefault("segment_budget_reservation_ids", []).append(
                [info.get("budget_reservation_id") for info in infos]
            )
            extras.setdefault("billing_receipts", []).append(billing_receipt_journal_entries(infos))
            resource_accounting = _resource_accounting(
                item,
                prompt_length=item.batch["prompts"].shape[-1],
                valid_mask=valid_mask,
                action_mask=action_mask,
                extra=base_extra,
            )
            append_feedback_record(
                self.feedback_journal_path, attempt_id=self.feedback_attempt_id,
                ledger_id=self.feedback_ledger_id, evaluation=evaluation, protocol=self.protocol,
                query_cost=query_cost_total, query_count=query_count,
                segment_count=segment_count, skipped_count=len(fallback_segments),
                budget_scope=self.budget_scope, global_budget=self.global_budget,
                resource_accounting=resource_accounting,
                billing_receipts=billing_receipt_journal_entries(infos),
                feedback_contract=self.feedback_contract,
            )

        max_segments = max((len(row["segment_rewards"]) for row in per_sample_segment_data), default=0)
        for key, fill, dtype in (
            ("segment_rewards", 0.0, np.float32),
            ("segment_durations", 0.0, np.float32),
            # TransferQueue fields must keep one dtype across all producers.
            # Use int64 for this logical mask because rollout response masks
            # and padding utilities use the same canonical transport dtype.
            ("segment_mask", 0, np.int64),
            ("segment_feedback_mask", 0, np.int64),
        ):
            extras[key] = [
                np.pad(row[key], (0, max_segments - len(row[key])), constant_values=fill).astype(dtype)
                for row in per_sample_segment_data
            ]
        extras["segment_ids"] = [row["segment_ids"] for row in per_sample_segment_data]
        # Preserve the immutable feature-map provenance in the reward-extra
        # envelope as well as the ragged feature tensor.  Validation JSONL and
        # downstream artifact audits can then distinguish a fixed contract
        # from a per-batch/padded-width normalization.
        extras["polyseg_feature_map_version"] = [
            row["polyseg_feature_map_version"] for row in per_sample_segment_data
        ]
        extras["polyseg_feature_map_hash"] = [
            row["polyseg_feature_map_hash"] for row in per_sample_segment_data
        ]
        extras["polyseg_feature_semantic_role"] = [
            row["polyseg_feature_semantic_role"] for row in per_sample_segment_data
        ]
        extras["polyseg_feature_causal_availability"] = [
            row["polyseg_feature_causal_availability"] for row in per_sample_segment_data
        ]
        extras["polyseg_feature_markov_sufficiency"] = [
            row["polyseg_feature_markov_sufficiency"] for row in per_sample_segment_data
        ]
        extras["segment_features"] = [
            np.pad(
                row["segment_features"],
                ((0, max_segments - len(row["segment_features"])), (0, 0)),
                constant_values=0.0,
            ).astype(np.float32)
            for row in per_sample_segment_data
        ]
        if any("polyseg_representation_hash" in row for row in per_sample_segment_data):
            if not all("polyseg_representation_hash" in row for row in per_sample_segment_data):
                raise ValueError("representation hash must be present for every sample with frozen features")
            extras["polyseg_representation_hash"] = [
                row["polyseg_representation_hash"] for row in per_sample_segment_data
            ]
        if any("step_features" in row for row in per_sample_segment_data):
            if not all("step_features" in row for row in per_sample_segment_data):
                raise ValueError("step_features must be present for every sample when shared-step recovery is used")
            max_steps = max(int(row["step_features"].shape[0]) for row in per_sample_segment_data)
            extras["step_features"] = [
                np.pad(
                    row["step_features"],
                    ((0, max_steps - row["step_features"].shape[0]), (0, 0)),
                    constant_values=0.0,
                ).astype(np.float32)
                for row in per_sample_segment_data
            ]
        extras["segment_next_features"] = [
            np.pad(
                row["segment_next_features"],
                ((0, max_segments - len(row["segment_next_features"])), (0, 0)),
                constant_values=0.0,
            ).astype(np.float32)
            for row in per_sample_segment_data
        ]
        extras["segment_transition_mask"] = [
            np.pad(
                row["segment_transition_mask"],
                (0, max_segments - len(row["segment_transition_mask"])),
                constant_values=False,
            ).astype(np.bool_)
            for row in per_sample_segment_data
        ]

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extras}
        return reward_tensor

