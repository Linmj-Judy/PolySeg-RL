"""PolySeg reward manager for verl's experimental reward loop."""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import uuid
from typing import Any

import numpy as np

from verl import DataProto
from verl.utils.polyseg_feedback import is_validation_batch
from verl.utils.polyseg_journal import append_feedback_record

logger = logging.getLogger(__name__)
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.trainer.ppo.polyseg import (
    FileBackedSegmentRewardBudget,
    SegmentRewardBudget,
    build_segment_features,
    compute_segment_rewards,
    decode_token_ids,
    extract_segment_metadata,
    feature_map_contract,
    normalize_feedback_protocol,
)


def _scalar(value: Any, default: Any):
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


@register("polyseg")
class PolySegRewardManager(RewardManagerBase):
    """Query a verifier once per rollout-owned segment and expose macro rewards."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        reward_kwargs = config.reward.get("reward_kwargs", {}) or {}
        polyseg_cfg = config.algorithm.get("polyseg", {}) or {}

        def get_value(key, default):
            return reward_kwargs[key] if reward_kwargs.get(key) is not None else polyseg_cfg.get(key, default)

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
            self.feature_horizon = None
        self.protocol = normalize_feedback_protocol(get_value("feedback", "local_sum"))
        self.max_budget = get_value("budget", None)
        self.budget_scope = str(get_value("budget_scope", "trajectory")).strip().lower()
        self.segment_cost = float(get_value("segment_cost", 1.0))
        self.budget_fallback = str(get_value("budget_fallback", "error")).strip().lower()
        self.require_rollout_segments = bool(get_value("require_rollout_segments", False))
        self.decode_skip_special_tokens = bool(get_value("decode_skip_special_tokens", True))
        self.is_async = inspect.iscoroutinefunction(compute_score)
        data_config = getattr(config, "data", None)
        if data_config is not None and hasattr(data_config, "get"):
            self.reward_fn_key = str(data_config.get("reward_fn_key", "data_source"))
        else:
            self.reward_fn_key = "data_source"
        if self.budget_scope not in {"trajectory", "global"}:
            raise ValueError("PolySeg budget_scope must be 'trajectory' or 'global'")
        reward_config = getattr(config, "reward", None)
        reward_workers = int(reward_config.get("num_workers", 1)) if reward_config is not None and hasattr(reward_config, "get") else 1
        feedback_ledger_path = get_value("feedback_ledger_path", None)
        if self.budget_scope == "global" and reward_workers > 1 and not feedback_ledger_path:
            raise ValueError(
                "PolySeg global budget with multiple reward workers requires "
                "reward.reward_kwargs.feedback_ledger_path"
            )
        self.global_budget = (
            (
                FileBackedSegmentRewardBudget(
                    feedback_ledger_path,
                    float(self.max_budget),
                    cost_per_query=self.segment_cost,
                )
                if feedback_ledger_path
                else SegmentRewardBudget(float(self.max_budget), cost_per_query=self.segment_cost)
            )
            if self.max_budget is not None and self.budget_scope == "global"
            else None
        )
        self._global_budget_lock = asyncio.Lock()
        self.feedback_journal_path = get_value("feedback_journal_path", None)
        self.feedback_ledger_path = feedback_ledger_path
        self.feedback_attempt_id = get_value("feedback_attempt_id", None)
        self.feedback_ledger_id = uuid.uuid4().hex
        if self.global_budget is not None and hasattr(self.global_budget, "ledger_id"):
            self.feedback_ledger_id = self.global_budget.ledger_id
        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.segment_cost <= 0:
            raise ValueError("segment_cost must be > 0")

    async def _score(self, *, data_source, solution_str, ground_truth, extra_info):
        kwargs = {
            "data_source": data_source,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
        }
        if self.is_async:
            return await self.compute_score(**kwargs)
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self.compute_score(**kwargs))

    async def _reserve_global(self, cost: float, *, reservation_id: str | None = None) -> tuple[str, float] | None:
        """Reserve an upper-bound query cost atomically across concurrent samples."""
        if self.global_budget is None:
            return None
        async with self._global_budget_lock:
            token = self.global_budget.reserve(cost, reservation_id=reservation_id)
            return token, self.global_budget.remaining

    async def _refund_global(self, token: str | None) -> float | None:
        if self.global_budget is None or token is None:
            return None if self.global_budget is None else self.global_budget.remaining
        async with self._global_budget_lock:
            self.global_budget.refund_reservation(token)
            return self.global_budget.remaining

    async def _commit_global(self, token: str | None, cost: float) -> float | None:
        if self.global_budget is None or token is None:
            return None if self.global_budget is None else self.global_budget.remaining
        async with self._global_budget_lock:
            return self.global_budget.commit(token, cost=cost)

    @staticmethod
    def _score_value(result: Any, *, protocol: str) -> tuple[float, dict[str, Any]]:
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

    async def run_single(self, data: DataProto) -> dict:
        if not is_validation_batch(data):
            return await self._run_single(data)
        # Do not mutate a shared worker's ledger while concurrent training
        # calls may be in flight. Evaluation uses the same scorer/protocol
        # and reports its real query cost separately from training B_K.
        evaluation_manager = copy.copy(self)
        evaluation_manager.max_budget = None
        evaluation_manager.global_budget = None
        result = await evaluation_manager._run_single(data, evaluation=True)
        details = result["reward_extra_info"]
        details["evaluation_query_cost"] = details["segment_query_cost"]
        details["evaluation_query_count"] = details["segment_query_count"]
        details["feedback_budget_scope"] = "evaluation"
        details["training_budget_remaining"] = None if self.global_budget is None else self.global_budget.remaining
        return result

    async def _run_single(self, data: DataProto, *, evaluation: bool = False) -> dict:
        item = data[0]
        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        prompt_length = item.batch["prompts"].shape[-1]
        attention = item.batch["attention_mask"][prompt_length : prompt_length + response_length].to(bool).cpu()
        response_mask = item.batch.get("response_mask")
        action_mask = attention if response_mask is None else response_mask[:response_length].to(bool).cpu() & attention
        metadata = extract_segment_metadata(
            item,
            response_length=response_length,
            valid_response_mask=attention,
            action_mask=action_mask,
            fallback_m=int(_scalar(item.non_tensor_batch.get("polyseg_adaptive_m"), self.m)),
            require_rollout_segments=self.require_rollout_segments,
        )
        segment_ids = metadata["segment_ids"]
        boundaries = metadata["segment_boundaries"]
        durations = metadata["segment_durations"]
        metadata_from_rollout = bool(metadata.get("from_rollout", False))
        valid_positions = attention.nonzero(as_tuple=False).flatten()
        valid_response_ids = response_ids.detach().cpu()[valid_positions]
        response_text = decode_token_ids(
            self.tokenizer, valid_response_ids, skip_special_tokens=self.decode_skip_special_tokens
        )
        if self.reward_fn_key not in item.non_tensor_batch:
            raise KeyError(f"PolySeg reward input is missing configured reward_fn_key={self.reward_fn_key!r}")
        data_source = item.non_tensor_batch[self.reward_fn_key]
        ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
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
        # ``selected_m`` is a planning cap; retain measured event count and
        # the rollout's boundary semantics for adaptive audits.
        actual_m = int(len(boundaries))
        m_semantics = str(base_extra.get("polyseg_m_semantics", "fixed_rollout"))
        # Rollout-owned ragged fields are also transported as direct
        # non-tensor keys by AgentLoop/TransferQueue.  Preserve them here so
        # successor-state and budget-id metadata cannot disappear between the
        # rollout and asynchronous scorer.
        for key in (
            "segment_next_features",
            "segment_transition_mask",
            "segment_action_lengths",
            "segment_tool_calls",
            "segment_tool_observations",
            "trajectory_id",
            "episode_id",
        ):
            value = item.non_tensor_batch.get(key)
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
        planned_compute_cost = float(
            _scalar(
                item.non_tensor_batch.get("polyseg_adaptive_compute_cost"),
                base_extra.get("polyseg_adaptive_compute_cost", 0.0),
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
                "polyseg_adaptive_compute_cost": planned_compute_cost,
                "polyseg_adaptive_fallback": adaptive_fallback,
                "polyseg_adaptive_fallback_reason": adaptive_fallback_reason,
            }
        )
        local_budget = (
            SegmentRewardBudget(self.max_budget, cost_per_query=self.segment_cost)
            if self.max_budget is not None and self.budget_scope == "trajectory"
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
            if self.protocol == "terminal" and segment_id < len(boundaries) - 1:
                values.append(0.0)
                infos.append({})
                continue
            if local_budget is not None and not local_budget.can_afford(cost=self.segment_cost):
                if self.budget_fallback == "skip":
                    values.append(0.0)
                    infos.append({"feedback_status": "skipped_budget", "query_cost": 0.0})
                    fallback_segments.append(segment_id)
                    continue
                raise RuntimeError("PolySeg per-trajectory feedback budget exhausted before segment query")
            reservation_token = None
            try:
                trajectory_id = base_extra.get("trajectory_id", base_extra.get("episode_id"))
                reservation = await self._reserve_global(
                    self.segment_cost,
                    reservation_id=(
                        None if trajectory_id in (None, "") else f"{trajectory_id}:segment:{segment_id}"
                    ),
                )
                if reservation is None:
                    reserved_global = None
                else:
                    reservation_token, reserved_global = reservation
            except RuntimeError:
                if self.budget_fallback == "skip":
                    values.append(0.0)
                    infos.append({"feedback_status": "skipped_budget", "query_cost": 0.0})
                    fallback_segments.append(segment_id)
                    continue
                raise
            # The reservation is already reflected in ``reserved_global``.
            # Pass the pre-reservation capacity to the scorer so an exact
            # budget boundary (remaining == segment_cost) is accepted.  The
            # ledger reservation still prevents concurrent overspending and
            # is settled after the scorer reports its actual cost.
            scorer_budget_remaining = (
                reserved_global + self.segment_cost
                if reserved_global is not None
                else None if local_budget is None else local_budget.remaining
            )
            positions = np.arange(start, end, dtype=np.int64)
            positions = positions[attention[positions].numpy()]
            segment_tensor_ids = response_ids.detach().cpu()[positions.tolist()]
            segment_text = decode_token_ids(
                self.tokenizer, segment_tensor_ids, skip_special_tokens=self.decode_skip_special_tokens
            )
            prefix_positions = valid_positions[valid_positions <= int(positions[-1])] if len(positions) else valid_positions
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
                    "segment_duration": float(durations[segment_id]),
                    "segment_count": len(boundaries),
                    "segment_text": segment_text,
                    "segment_prefix": prefix_text,
                    "segment_action_mask": action_mask[start:end].numpy(),
                    "segment_response_mask": attention[start:end].numpy(),
                    "full_response": response_text,
                    "tool_observations": tool_observations,
                    "tool_calls": tool_calls,
                    "polyseg_protocol": self.protocol,
                    "budget_remaining": scorer_budget_remaining,
                    "budget_reservation_id": reservation_token,
                }
            )
            try:
                result = await self._score(
                    data_source=data_source,
                    solution_str=segment_text,
                    ground_truth=ground_truth,
                    extra_info=extra,
                )
            except Exception:
                # No billed response was returned, so release the active
                # reservation. A retry can then make an exact new hold.
                await self._refund_global(reservation_token)
                raise
            try:
                value, info = self._score_value(result, protocol=self.protocol)
            except Exception:
                # The verifier returned but could not be normalized. Treat
                # the conservative reservation as billed and leave an
                # auditable ledger state instead of leaking an active hold.
                await self._commit_global(reservation_token, self.segment_cost)
                raise
            info.setdefault("budget_reservation_id", reservation_token)
            raw_cost = info.get("query_cost", self.segment_cost)
            try:
                query_cost = float(raw_cost)
            except (TypeError, ValueError) as exc:
                # A response was received but its billing metadata is
                # malformed. Keep the conservative reservation charged rather
                # than silently refunding an externally billed request.
                await self._commit_global(reservation_token, self.segment_cost)
                raise ValueError("PolySeg scorer query_cost must be a finite non-negative number") from exc
            if not np.isfinite(query_cost) or query_cost < 0:
                await self._commit_global(reservation_token, self.segment_cost)
                raise ValueError("PolySeg scorer query_cost must be a finite non-negative number")
            if query_cost > self.segment_cost + 1e-8:
                await self._commit_global(reservation_token, self.segment_cost)
                raise ValueError(
                    "PolySeg scorer query_cost exceeds configured segment_cost; "
                    "set segment_cost to the maximum verifier/suffix query cost"
                )
            if local_budget is not None and not local_budget.can_afford(cost=query_cost):
                await self._commit_global(reservation_token, query_cost)
                raise RuntimeError(
                    "PolySeg scorer exceeded the per-trajectory feedback budget: "
                    f"requested={query_cost}, remaining={local_budget.remaining}"
                )
            values.append(value)
            infos.append(info)
            query_count += 1
            query_cost_total += query_cost
            if local_budget is not None:
                local_budget.consume(cost=query_cost)
            await self._commit_global(reservation_token, query_cost)

        rewards = compute_segment_rewards(values, self.protocol)
        total = float(np.mean(rewards)) if self.protocol == "mean" else float(sum(rewards))
        segment_features = build_segment_features(
            boundaries,
            durations,
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
            or [0] * len(boundaries),
            feature_dim=self.feature_dim,
            feature_horizon=self.feature_horizon,
        ).numpy().astype(np.float32)
        feature_contract = feature_map_contract(
            response_length=self.feature_horizon or response_length,
            feature_dim=self.feature_dim,
        )
        final_info = infos[-1] if infos else {}
        # Accuracy is an observed verifier metric, not a synonym for utility
        # or reward.  Keep it missing when the scorer did not return it so
        # downstream summaries cannot silently treat a score as correctness.
        final_accuracy_value = final_info.get("acc", final_info.get("accuracy"))
        final_accuracy = None if final_accuracy_value is None else float(final_accuracy_value)
        details: dict[str, Any] = {
            "segment_rewards": np.asarray(rewards, dtype=np.float32),
            "segment_ids": segment_ids.numpy().astype(np.int64),
            "segment_durations": durations.numpy().astype(np.float32),
            "segment_features": segment_features,
            "polyseg_feature_map_version": feature_contract["version"],
            "polyseg_feature_map_hash": feature_contract["hash"],
            # Do not fabricate a transition by shifting current summary
            # features. A transition backup is valid only with rollout-owned
            # next-boundary features and an explicit transition mask.
            "segment_next_features": np.asarray(
                base_extra.get("segment_next_features", np.zeros((len(boundaries), self.feature_dim), dtype=np.float32)),
                dtype=np.float32,
            ),
            "segment_transition_mask": np.asarray(
                base_extra.get("segment_transition_mask", np.zeros(len(boundaries), dtype=np.bool_)),
                dtype=np.bool_,
            ),
            "polyseg_transition_source": (
                "rollout" if "segment_next_features" in base_extra else "missing"
            ),
            # Keep logical masks transport-stable for TransferQueue.  The
            # estimator converts this field to bool at the computation edge.
            "segment_mask": np.ones(len(rewards), dtype=np.int64),
            "segment_feedback_mask": np.asarray(
                [info.get("feedback_status") != "skipped_budget" for info in infos], dtype=np.int64
            ),
            "episode_reward": total,
            "segment_reward_sum": float(sum(rewards)),
            "segment_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "final_accuracy": final_accuracy,
            "segment_count": len(rewards),
            "polyseg_adaptive_actual_m": actual_m,
            "polyseg_m_semantics": m_semantics,
            "segment_query_cost": float(query_cost_total),
            "polyseg_adaptive_actual_cost": float(query_cost_total),
            "polyseg_adaptive_planned_cost": planned_cost,
            "polyseg_adaptive_compute_cost": planned_compute_cost,
            "polyseg_adaptive_fallback": adaptive_fallback,
            "polyseg_adaptive_fallback_reason": adaptive_fallback_reason,
            "polyseg_adaptive_q": selected_q,
            "polyseg_adaptive_m": selected_m,
            "polyseg_adaptive_remaining_budget": _scalar(
                item.non_tensor_batch.get("polyseg_adaptive_remaining_budget"), None
            ),
            "segment_query_count": query_count,
            "polyseg_segment_metadata_source": "rollout" if metadata_from_rollout else "fixed_token_fallback",
            "segment_feedback_fallback_segments": list(fallback_segments),
            "segment_feedback_fallback_count": len(fallback_segments),
            "segment_budget_remaining": (
                self.global_budget.remaining
                if self.global_budget is not None
                else None if local_budget is None else local_budget.remaining
            ),
            "segment_info": [
                {"segment_id": j, "reward": rewards[j], "value": values[j], **infos[j]}
                for j in range(len(rewards))
            ],
            "segment_budget_reservation_ids": [
                info.get("budget_reservation_id") for info in infos
            ],
        }
        prm_scores = [float(info["prm_score"]) for info in infos if "prm_score" in info]
        if prm_scores:
            details["prm_score_mean"] = float(np.mean(prm_scores))
            details["prm_score_final"] = float(prm_scores[-1])
            details["model_derived_pseudo_label"] = True
            details["prm_model_path"] = str(infos[-1].get("prm_model_path", ""))
        # ``acc`` remains a final-verifier metric for existing logging code;
        # episode reward is exposed separately and is never mislabeled.
        details["acc"] = details["final_accuracy"]
        append_feedback_record(
            self.feedback_journal_path, attempt_id=self.feedback_attempt_id,
            ledger_id=self.feedback_ledger_id, evaluation=evaluation, protocol=self.protocol,
            query_cost=query_cost_total, query_count=query_count,
            segment_count=len(boundaries), skipped_count=len(fallback_segments),
            budget_scope=self.budget_scope, global_budget=self.global_budget,
        )
        return {"reward_score": total, "reward_extra_info": details}

