# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import inspect
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.function_tool import FunctionTool, normalize_function_tool_return
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput
from verl.trainer.ppo.polyseg import build_segment_features
from verl.experimental.agent_loop.boundary_state import (
    BoundaryStateContractError,
    validate_boundary_snapshot,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPEC_DECODE_EXTRA_KEYS = (
    "spec_num_draft_tokens",
    "spec_num_accepted_tokens",
    "spec_num_verify_steps",
)


def _contiguous_spans(positions: list[int]) -> list[tuple[int, int]]:
    """Convert sorted token positions to half-open contiguous spans."""

    if not positions:
        return []
    spans: list[tuple[int, int]] = []
    start = previous = int(positions[0])
    for raw_position in positions[1:]:
        position = int(raw_position)
        if position != previous + 1:
            spans.append((start, previous + 1))
            start = position
        previous = position
    spans.append((start, previous + 1))
    return spans


def build_tool_observation_metadata(
    segment_ids: list[int] | tuple[int, ...],
    response_mask: list[int] | tuple[int, ...],
    segment_count: int,
) -> tuple[list[bool], list[list[tuple[int, int]]], list[list[bool]]]:
    """Build explicit token and segment action/observation metadata.

    ``response_mask`` is one for policy-generated action tokens and zero for
    tool-context or padding tokens.  Segment IDs distinguish tool-context
    tokens from right padding, so the exported observation mask is fail-closed
    for padded positions.  Spans are absolute response-token half-open ranges.
    """

    ids = [int(value) for value in segment_ids]
    actions = [bool(value) for value in response_mask]
    if len(ids) != len(actions):
        raise ValueError("segment_ids and response_mask must be token-aligned")
    if segment_count < 0:
        raise ValueError("segment_count must be non-negative")
    if any(segment_id >= segment_count for segment_id in ids if segment_id >= 0):
        raise ValueError("segment_ids exceed the segment count")
    observation_mask = [segment_id >= 0 and not is_action for segment_id, is_action in zip(ids, actions, strict=True)]
    spans = [
        _contiguous_spans(
            [position for position, (segment_id, is_observation) in enumerate(zip(ids, observation_mask, strict=True)) if segment_id == index and is_observation]
        )
        for index in range(segment_count)
    ]
    action_masks: list[list[bool]] = []
    for index in range(segment_count):
        start, end = _segment_span(ids, index)
        action_masks.append(actions[start:end])
    return observation_mask, spans, action_masks


def _segment_span(segment_ids: list[int], segment_id: int) -> tuple[int, int]:
    positions = [position for position, value in enumerate(segment_ids) if value == segment_id]
    if not positions:
        raise ValueError(f"segment {segment_id} has no token positions")
    return positions[0], positions[-1] + 1


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []
        # SGLang video payload for the initial prompt (see AgentLoopBase.build_sglang_video_payload);
        # None for text/image-only inputs or when the vLLM backend is used.
        self.mm_processor_output: Optional[list[dict[str, Any]]] = None
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        # Rollout-owned PolySeg metadata.  A segment contains one assistant
        # action and, when present, its tool call/observation.  Observation
        # tokens remain response_mask=0 but retain the segment id.
        self.segment_ids: list[int] = []
        self.current_segment_id = 0
        self.next_segment_pending = False
        # Per-segment environment context.  These fields stay aligned with
        # ``segment_ids`` at the macro level and are consumed by a verifier or
        # suffix-feedback scorer; they are deliberately kept separate from
        # token text because tool observations are not policy actions.
        self.segment_tool_observations: dict[int, list[str]] = {}
        self.segment_tool_observation_spans: dict[int, list[dict[str, Any]]] = {}
        self.segment_tool_calls: dict[int, list[dict[str, Any]]] = {}
        self.segment_action_texts: dict[int, str] = {}
        self.segment_durations: dict[int, float] = {}
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        # Explicit per-trajectory tool accounting is exported with the
        # rollout so full agentic runs can report valid calls and failures,
        # rather than inferring them from token counts.
        self.tool_call_successes = 0
        self.tool_call_failures = 0

        # Segment state IDs are correlation keys only. A science harness may
        # attach durable boundary/hidden/kernel artifacts through a provider.
        self.boundary_snapshots: dict[int, dict[str, Any]] = {}
        self.boundary_snapshot_errors: dict[int, str] = {}
        self.boundary_state_provider: Any = None
        self.require_boundary_snapshots = False

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    merges_context_messages = True

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        """Initialize the tool agent loop.

        Args:
            tools: Tools to use for the tool agent loop.
        """
        super().__init__(*args, **kwargs)

        self.boundary_state_provider = kwargs.get("boundary_state_provider")
        self.require_boundary_snapshots = bool(
            getattr(self.rollout_config, "polyseg_require_boundary_snapshots", False)
        )

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side

        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        # Some instruction checkpoints ship a Hermes template but do not
        # reliably emit the XML wrapper that the template documents.  The
        # event protocol can opt into vLLM's per-request JSON-schema decoding
        # so the first action is still sampled by the model and then executed
        # by this loop.  Keep this disabled for ordinary rollouts and for
        # backends that do not expose vLLM structured outputs.
        self.guided_first_tool_call = os.getenv("POLYSEG_GUIDED_TOOL_CALL", "0").lower() in {
            "1", "true", "yes"
        }

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @staticmethod
    def _make_request_id(priority: int, full_determinism: bool) -> str:
        return f"det-{priority}" if full_determinism else uuid4().hex

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        metrics = {}
        request_id = self._make_request_id(priority, self.rollout_config.full_determinism)
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )
        # Allow a harness to provide a per-request provider without placing a
        # non-serializable object in the dataset's ordinary extra_info fields.
        agent_data.boundary_state_provider = kwargs.get(
            "boundary_state_provider", getattr(self, "boundary_state_provider", None)
        )
        agent_data.require_boundary_snapshots = bool(
            kwargs.get(
                "polyseg_require_boundary_snapshots",
                getattr(self, "require_boundary_snapshots", False),
            )
        )
        agent_data.sampling_params = dict(sampling_params)

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            mm_processor_output=agent_data.mm_processor_output,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        response_length = len(output.response_ids)
        segment_ids = agent_data.segment_ids[:response_length]
        segment_count = max(segment_ids) + 1 if segment_ids else 0
        # Adaptive m is an upper bound for event-native agent segments.  If a
        # trajectory has more event segments than selected m, merge adjacent
        # complete events rather than splitting a tool call/observation pair.
        # Fixed event protocols pass their cap through the rollout config,
        # while adaptive protocols pass it per request.  Resolve both paths
        # here so the trace records the actual requested cap rather than
        # silently treating fixed-m tool runs as uncapped.
        configured_m = kwargs.get("polyseg_adaptive_m", getattr(self.rollout_config, "polyseg_fixed_m", 0))
        configured_m = int(configured_m.item() if hasattr(configured_m, "item") else configured_m or 0)
        if configured_m > 0 and segment_count > configured_m:
            old_count = segment_count
            remap = {
                old_id: min(configured_m - 1, (old_id * configured_m) // old_count)
                for old_id in range(old_count)
            }
            segment_ids = [remap.get(value, value) for value in segment_ids]
            merged_observations: dict[int, list[str]] = {}
            merged_observation_spans: dict[int, list[dict[str, Any]]] = {}
            merged_calls: dict[int, list[dict[str, Any]]] = {}
            merged_actions: dict[int, list[str]] = {}
            merged_durations: dict[int, float] = {}
            merged_snapshots: dict[int, dict[str, Any]] = {}
            merged_snapshot_errors: dict[int, str] = {}
            for old_id, new_id in remap.items():
                merged_observations.setdefault(new_id, []).extend(agent_data.segment_tool_observations.get(old_id, []))
                merged_observation_spans.setdefault(new_id, []).extend(
                    agent_data.segment_tool_observation_spans.get(old_id, [])
                )
                merged_calls.setdefault(new_id, []).extend(agent_data.segment_tool_calls.get(old_id, []))
                merged_actions.setdefault(new_id, []).append(agent_data.segment_action_texts.get(old_id, ""))
                merged_durations[new_id] = merged_durations.get(new_id, 0.0) + float(
                    agent_data.segment_durations.get(old_id, 1.0)
                )
                if old_id in agent_data.boundary_snapshots:
                    merged_snapshots.setdefault(new_id, agent_data.boundary_snapshots[old_id])
                if old_id in agent_data.boundary_snapshot_errors:
                    merged_snapshot_errors[new_id] = agent_data.boundary_snapshot_errors[old_id]
            agent_data.segment_tool_observations = merged_observations
            agent_data.segment_tool_observation_spans = merged_observation_spans
            agent_data.segment_tool_calls = merged_calls
            agent_data.segment_action_texts = {
                key: "\\n".join(value) for key, value in merged_actions.items()
            }
            agent_data.segment_durations = merged_durations
            agent_data.boundary_snapshots = merged_snapshots
            agent_data.boundary_snapshot_errors = merged_snapshot_errors
            segment_count = configured_m
        segment_durations = [
            float(agent_data.segment_durations.get(segment_id, 1.0))
            for segment_id in range(segment_count)
        ]
        segment_action_lengths = [
            sum(
                1
                for token_segment_id, is_action in zip(segment_ids, output.response_mask)
                if is_action and token_segment_id == segment_id
            )
            for segment_id in range(segment_count)
        ]
        segment_tool_calls = [
            list(agent_data.segment_tool_calls.get(segment_id, []))
            for segment_id in range(segment_count)
        ]
        segment_boundaries = [
            (segment_ids.index(segment_id), len(segment_ids) - segment_ids[::-1].index(segment_id))
            for segment_id in range(segment_count)
        ] if segment_ids else []
        observation_mask, observation_spans, segment_action_masks = build_tool_observation_metadata(
            segment_ids,
            output.response_mask,
            segment_count,
        ) if segment_count else ([], [], [])
        recorded_observation_spans = [
            list(agent_data.segment_tool_observation_spans.get(segment_id, []))
            for segment_id in range(segment_count)
        ]
        mapped_step_features, state_features, representation, representation_hash = self._prepare_polyseg_step_features(
            kwargs, response_length
        )
        # Tool-agent rollouts own the event boundaries and therefore can
        # expose the observed successor boundary feature directly.  The final
        # event is terminal and intentionally has no transition label.
        segment_features = build_segment_features(
            segment_boundaries,
            segment_durations,
            response_length=max(self.response_length, 1),
            action_lengths=segment_action_lengths,
            tool_call_counts=[len(calls) for calls in segment_tool_calls],
            feature_dim=max(1, int(getattr(self.rollout_config, "polyseg_feature_dim", 4))),
            state_features=state_features if state_features is not None else mapped_step_features,
            representation=representation,
            state_mask=torch.as_tensor(output.response_mask, dtype=torch.bool),
        )
        segment_next_features = torch.zeros_like(segment_features)
        segment_transition_mask = [False] * segment_count
        if segment_count > 1:
            segment_next_features[:-1] = segment_features[1:]
            segment_transition_mask[:-1] = [True] * (segment_count - 1)
        output.extra_fields.update(
            {
                "segment_ids": segment_ids,
                "polyseg_adaptive_actual_m": segment_count,
                "polyseg_m_semantics": "event_boundary_cap" if configured_m > 0 else "event_boundary_count",
                # One assistant/tool interaction is one macro transition by
                # default.  Tools may return a measured duration in their
                # result metadata; that value is preserved for duration-aware
                # Bellman/GAE instead of deriving duration from token count.
                "segment_durations": segment_durations,
                "segment_action_lengths": segment_action_lengths,
                "segment_tool_observations": [
                    list(agent_data.segment_tool_observations.get(segment_id, []))
                    for segment_id in range(segment_count)
                ],
                "tool_observation_mask": observation_mask,
                "tool_observation_spans": observation_spans,
                "segment_tool_observation_spans": recorded_observation_spans,
                "segment_action_masks": segment_action_masks,
                "segment_action_mask": [
                    segment_id >= 0 and bool(is_action)
                    for segment_id, is_action in zip(segment_ids, output.response_mask, strict=True)
                ],
                "segment_tool_calls": segment_tool_calls,
                "segment_action_texts": [
                    agent_data.segment_action_texts.get(segment_id, "")
                    for segment_id in range(segment_count)
                ],
                "segment_state_ids": [
                    f"{request_id}:s{segment_id}" for segment_id in range(segment_count)
                ],
                "trajectory_id": request_id,
                "episode_id": request_id,
                "segment_boundaries": segment_boundaries,
                "segment_features": segment_features.tolist(),
                "segment_next_features": segment_next_features.tolist(),
                "segment_transition_mask": segment_transition_mask,
                "boundary_snapshots": [
                    agent_data.boundary_snapshots.get(segment_id)
                    for segment_id in range(segment_count)
                ],
                "boundary_snapshot_errors": {
                    str(segment_id): reason
                    for segment_id, reason in agent_data.boundary_snapshot_errors.items()
                },
                "boundary_snapshot_required": bool(agent_data.require_boundary_snapshots),
            }
        )
        if mapped_step_features is not None:
            output.extra_fields["step_features"] = mapped_step_features.tolist()
            output.extra_fields["polyseg_representation_hash"] = representation_hash
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.update(
            {
                "tool_call_count": int(sum(len(calls) for calls in agent_data.segment_tool_calls.values())),
                "tool_call_success_count": int(agent_data.tool_call_successes),
                "tool_call_failure_count": int(agent_data.tool_call_failures),
            }
        )
        for key in (
            "polyseg_adaptive_q",
            "polyseg_adaptive_m",
            "polyseg_adaptive_planned_cost",
            "polyseg_adaptive_compute_cost",
            "polyseg_adaptive_remaining_budget",
            "polyseg_adaptive_fallback",
            "polyseg_adaptive_fallback_reason",
        ):
            if key in kwargs:
                value = kwargs[key]
                output.extra_fields[key] = value.item() if hasattr(value, "item") else value
        # Preserve an auditable trajectory at validation, including the raw
        # event IDs before any m-cap merging and the action/observation mask.
        # Reward extras alone do not contain these rollout-owned objects.
        output.extra_fields["agentic_trace"] = {
            "schema_version": "polyseg-agentic-trace-v1",
            "trajectory_id": request_id,
            "episode_id": request_id,
            "agent_loop": "tool_agent",
            "m_semantics": output.extra_fields["polyseg_m_semantics"],
            "requested_m_cap": configured_m,
            "actual_m": segment_count,
            "event_segment_ids": agent_data.segment_ids[:response_length],
            "response_token_ids": output.response_ids,
            "response_mask": output.response_mask,
            **{
                key: output.extra_fields[key]
                for key in (
                    "segment_ids",
                    "segment_boundaries",
                    "segment_durations",
                    "segment_action_lengths",
                    "segment_tool_calls",
                    "segment_tool_observations",
                    "tool_observation_mask",
                    "tool_observation_spans",
                    "segment_tool_observation_spans",
                    "segment_action_masks",
                    "segment_action_mask",
                    "segment_action_texts",
                    "segment_state_ids",
                    "tool_call_count",
                    "tool_call_success_count",
                    "tool_call_failure_count",
                    "boundary_snapshots",
                    "boundary_snapshot_errors",
                    "boundary_snapshot_required",
                )
            },
        }
        return output

    async def _capture_boundary_state(
        self,
        agent_data: AgentData,
        *,
        segment_id: int,
        sampling_params: dict[str, Any],
    ) -> None:
        """Capture a provider-owned recoverable boundary, fail-closed when required."""

        provider = getattr(agent_data, "boundary_state_provider", None)
        required = bool(getattr(agent_data, "require_boundary_snapshots", False))
        if segment_id in agent_data.boundary_snapshots:
            return
        if provider is None:
            reason = "boundary_state_provider_missing"
            agent_data.boundary_snapshot_errors[segment_id] = reason
            if required:
                raise RuntimeError(reason)
            return
        capture = getattr(provider, "capture_boundary_state", None)
        if not callable(capture):
            capture = getattr(provider, "capture", None)
        if not callable(capture):
            reason = "boundary_state_provider_capture_missing"
            agent_data.boundary_snapshot_errors[segment_id] = reason
            if required:
                raise RuntimeError(reason)
            return
        context = {
            "request_id": agent_data.request_id,
            "trajectory_id": agent_data.request_id,
            "session_id": agent_data.request_id,
            "segment_id": segment_id,
            "prefix_token_ids": list(agent_data.prompt_ids),
            "action_mask": list(agent_data.response_mask),
            "response_logprobs": list(agent_data.response_logprobs or []),
            "tool_calls": list(agent_data.segment_tool_calls.get(segment_id, [])),
            "tool_observations": list(agent_data.segment_tool_observations.get(segment_id, [])),
            "sampling_config": dict(sampling_params),
            "server_manager": self.server_manager,
        }
        try:
            # Providers in external harnesses commonly expose either a
            # keyword-rich callback (``capture_boundary_state(**context)``)
            # or a single request envelope (``capture(context)``).  Inspect
            # the signature before invoking so a provider-side TypeError is
            # not accidentally executed twice and the two supported shapes
            # have identical validation and fail-closed behavior.
            try:
                signature = inspect.signature(capture)
                signature.bind(**context)
            except (TypeError, ValueError):
                snapshot = capture(context)
            else:
                snapshot = capture(**context)
            if inspect.isawaitable(snapshot):
                snapshot = await snapshot
            agent_data.boundary_snapshots[segment_id] = validate_boundary_snapshot(
                snapshot, require_restore_audit=required
            )
        except (BoundaryStateContractError, TypeError, ValueError, RuntimeError) as exc:
            reason = f"boundary_snapshot_invalid:{type(exc).__name__}:{exc}"
            agent_data.boundary_snapshot_errors[segment_id] = reason
            if required:
                raise RuntimeError(reason) from exc

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        # Continuous Token is the only tokenization path; multimodal prompts require a
        # VL builder + processor, so validate before building any tokens.
        self._assert_mm_supported(bool(agent_data.image_data or agent_data.video_data or agent_data.audio_data))
        mm_inputs = {}
        prompt_ids = await self.ct_build_initial_tokens(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_inputs_out=mm_inputs,
        )
        agent_data.prompt_ids = prompt_ids
        agent_data.mm_processor_output = self.build_sglang_video_payload(agent_data.video_data, mm_inputs)
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        if agent_data.next_segment_pending:
            agent_data.current_segment_id += 1
            agent_data.next_segment_pending = False
        # Inject tool parser stop tokens so generation halts after each tool call
        if self.tool_parser.stop_token_ids:
            stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}

        # The guided request is deliberately limited to the first assistant
        # turn and a single selected function.  vLLM validates this JSON
        # schema and stops at a complete object; the returned tokens are then
        # parsed below and dispatched through ``_call_tool``.  No text or tool
        # result is injected into the trajectory.
        if self.guided_first_tool_call and agent_data.assistant_turns == 0:
            active_tools = getattr(agent_data, "_active_tools", self.tools)
            if len(active_tools) == 1:
                only_tool = next(iter(active_tools.values()))
                function_schema = only_tool.tool_schema.model_dump(
                    exclude_unset=True, exclude_none=True
                ).get("function", {})
                parameters = function_schema.get("parameters")
                if isinstance(parameters, dict):
                    schema = {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": [only_tool.name]},
                            "arguments": parameters,
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                    }
                    try:
                        # Keep the import lazy: CPU/unit-test users and
                        # non-vLLM backends should not need vLLM installed.
                        from vllm.sampling_params import StructuredOutputsParams
                    except ImportError as exc:  # pragma: no cover - backend-only path
                        raise RuntimeError(
                            "POLYSEG_GUIDED_TOOL_CALL requires the vLLM structured-output backend"
                        ) from exc
                    sampling_params = {
                        **sampling_params,
                        "structured_outputs": StructuredOutputsParams(json=schema),
                    }

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                mm_processor_output=agent_data.mm_processor_output,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields and key in agent_data.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            agent_data.prompt_ids,
            agent_data.response_ids,
            agent_data.response_mask,
            agent_data.response_logprobs if (agent_data.response_logprobs or output.log_probs) else None,
            assistant_logprobs=output.log_probs if output.log_probs else None,
        )
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        agent_data.segment_ids = AgentLoopBase.align_segment_ids(
            merge_result, getattr(agent_data, "segment_ids", []), getattr(agent_data, "current_segment_id", 0)
        )
        if response_logprobs is not None:
            agent_data.response_logprobs = response_logprobs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        assistant_content, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids, tools
        )
        # Preserve the exact assistant action text at the segment level for
        # replay/failure analysis.  Tool observations are recorded separately
        # and remain response_mask=0.
        agent_data.segment_action_texts[agent_data.current_segment_id] = self.tokenizer.decode(
            agent_data.response_ids, skip_special_tokens=False
        )
        if agent_data.tool_calls:
            segment_tool_calls = getattr(agent_data, "segment_tool_calls", None)
            if segment_tool_calls is None:
                segment_tool_calls = {}
                agent_data.segment_tool_calls = segment_tool_calls
            segment_tool_calls.setdefault(agent_data.current_segment_id, []).extend(
                [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "tool_call_id": call.tool_call_id,
                    }
                    for call in agent_data.tool_calls[: self.max_parallel_calls]
                ]
            )
        agent_data.messages.append(self._build_assistant_message(assistant_content, agent_data))

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute
        previous_messages = list(agent_data.messages)
        previous_response_length = len(agent_data.response_mask)

        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_index, (tool_response, tool_reward, tool_metadata) in enumerate(responses):
            tool_call = agent_data.tool_calls[tool_index]
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    for img in tool_response.image:
                        if img is not None:
                            content.append({"type": "image", "image": img})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}
            if tool_call.tool_call_id is not None:
                message["tool_call_id"] = tool_call.tool_call_id

            add_messages.append(message)

            segment_id = int(getattr(agent_data, "current_segment_id", 0))
            segment_tool_observations = getattr(agent_data, "segment_tool_observations", None)
            if segment_tool_observations is None:
                segment_tool_observations = {}
                agent_data.segment_tool_observations = segment_tool_observations
            segment_tool_observations.setdefault(segment_id, []).append(tool_response.text or "")
            if isinstance(tool_metadata, dict):
                # A stateful science tool may return its durable checkpoint
                # projection together with the observation.  This is the
                # serializable path used by Ray/dataset rollouts; bare state
                # IDs are rejected by ``validate_boundary_snapshot``.
                snapshot = tool_metadata.get("boundary_snapshot")
                if snapshot is not None:
                    try:
                        agent_data.boundary_snapshots[segment_id] = validate_boundary_snapshot(
                            snapshot,
                            require_restore_audit=bool(getattr(agent_data, "require_boundary_snapshots", False)),
                        )
                    except BoundaryStateContractError as exc:
                        reason = f"boundary_snapshot_invalid:{exc}"
                        agent_data.boundary_snapshot_errors[segment_id] = reason
                        if getattr(agent_data, "require_boundary_snapshots", False):
                            raise RuntimeError(reason) from exc
                duration = tool_metadata.get("duration", tool_metadata.get("segment_duration"))
                if duration is not None:
                    duration = float(duration)
                    if duration <= 0:
                        raise ValueError("tool duration must be positive")
                    # Parallel tool calls form one environment transition;
                    # retain the slowest measured call as its duration.
                    segment_durations = getattr(agent_data, "segment_durations", None)
                    if segment_durations is None:
                        segment_durations = {}
                        agent_data.segment_durations = segment_durations
                    segment_durations[segment_id] = max(
                        segment_durations.get(segment_id, 1.0), duration
                    )

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # Continuous Token is the only tokenization path; tool responses that add new
        # images require a VL builder + processor. Validate before mutating rollout state
        # so a failure never leaves a half-built prompt behind.
        self._assert_mm_supported(bool(new_images_this_turn))

        agent_data.messages.extend(add_messages)

        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        merge_result, response_mask, response_logprobs = await self.ct_merge_context_msg(
            previous_messages,
            agent_data.messages,
            agent_data.prompt_ids,
            agent_data.response_mask,
            agent_data.response_logprobs if agent_data.response_logprobs else None,
            tools=schemas,
        )
        agent_data.segment_ids = AgentLoopBase.align_segment_ids(
            merge_result, getattr(agent_data, "segment_ids", []), getattr(agent_data, "current_segment_id", 0)
        )
        removed = int(getattr(merge_result, "removed_prefix_token_count", 0))
        inserted = len(getattr(merge_result, "inserted_token_ids", ()))
        context_start = max(0, previous_response_length - removed) + inserted
        context_end = context_start + int(getattr(merge_result, "appended_token_count", 0))
        if context_end > context_start:
            segment_id = int(getattr(agent_data, "current_segment_id", 0))
            tool_call_ids = [call.tool_call_id for call in agent_data.tool_calls if call.tool_call_id]
            agent_data.segment_tool_observation_spans.setdefault(segment_id, []).append(
                {
                    "start": context_start,
                    "end": context_end,
                    "tool_call_ids": tool_call_ids,
                }
            )
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        capture_boundary = getattr(self, "_capture_boundary_state", None)
        if callable(capture_boundary):
            await capture_boundary(
                agent_data,
                segment_id=int(getattr(agent_data, "current_segment_id", 0)),
                sampling_params=getattr(agent_data, "sampling_params", {}),
            )
        if len(response_mask) >= self.response_length:
            return AgentState.TERMINATED
        # The next assistant action starts the next macro segment; the tool
        # observation just appended above stays in the current segment.
        agent_data.next_segment_pending = True
        if agent_data.response_logprobs:
            agent_data.response_logprobs = response_logprobs or []

        # Keep original image objects for rollout and final multimodal postprocessing.
        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.user_turns += 1
        return AgentState.GENERATING

    def _build_assistant_message(self, content: str, agent_data: AgentData) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if not agent_data.tool_calls:
            return message

        tool_calls = []
        for index, tool_call in enumerate(agent_data.tool_calls[: self.max_parallel_calls]):
            function_call, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                OpenAIFunctionParsedSchema(name=tool_call.name, arguments=tool_call.arguments)
            )
            if has_decode_error:
                logger.warning(
                    f"Invalid tool call arguments for '{tool_call.name}': expected a JSON object string, "
                    f"got {tool_call.arguments!r}; keeping the raw arguments in the assistant message. "
                    f"Note: re-applying the chat template to this message may render differently from "
                    f"the model's actual generation because the tool call arguments are in invalid format."
                )
                function_dict = {"name": tool_call.name, "arguments": tool_call.arguments}
            else:
                function_dict = function_call.model_dump()
            tool_call_message = {
                "type": "function",
                "function": function_dict,
            }
            if tool_call.tool_call_id is not None:
                tool_call_message["id"] = tool_call.tool_call_id
            tool_calls.append(tool_call_message)
        message["tool_calls"] = tool_calls
        return message

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Dispatches between two contracts:
        - ``FunctionTool``: stateless function-based tool. Invoked directly with
          parsed arguments; no lifecycle.
        - ``BaseTool`` subclass: stateful tool with full lifecycle.
        """
        active_tools = getattr(agent_data, "_active_tools", self.tools)

        # Validate tool name
        tool_name = tool_call.name
        if tool_name not in active_tools:
            available = list(active_tools.keys())
            msg = f"Unknown function '{tool_name}'. Available tools: {available}"
            logger.warning(msg)
            # Keep the error path compatible with lightweight AgentData-like
            # objects used by callers/tests that predate the replay counters.
            agent_data.tool_call_failures = getattr(agent_data, "tool_call_failures", 0) + 1
            return ToolResponse(text=msg), 0.0, {}

        # Validate tool arguments
        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
            logger.warning(msg)
            agent_data.tool_call_failures = getattr(agent_data, "tool_call_failures", 0) + 1
            return ToolResponse(text=msg), 0.0, {}

        # Execute tool
        tool, instance_id = None, None
        try:
            tool = active_tools[tool_name]

            if isinstance(tool, FunctionTool):
                # Function-based tools have no lifecycle; call directly.
                # Note: tools_kwargs (create_kwargs / release_kwargs) is intentionally
                # ignored here. Function tools are stateless and per-trajectory state
                # injection is not supported by design; use a BaseTool subclass instead.
                raw = await tool.call(tool_args)
                tool_execution_response, tool_reward, res = normalize_function_tool_return(raw)
            else:
                # BaseTool subclass
                kwargs = tools_kwargs.get(tool_name, {})
                instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
                tool_execution_response, tool_reward, res = await tool.execute(
                    instance_id, tool_args, agent_data=agent_data
                )
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_name}': {e}")
            agent_data.tool_call_failures = getattr(agent_data, "tool_call_failures", 0) + 1
            return ToolResponse(text=f"Error executing tool '{tool_name}': {e}"), 0.0, {}
        finally:
            # Only BaseTool instances need release (function tools never set instance_id).
            if tool and instance_id and not isinstance(tool, FunctionTool):
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            elif self.tool_response_truncate_side == "right":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        agent_data.tool_call_successes = getattr(agent_data, "tool_call_successes", 0) + 1
        return ToolResponse(**tool_response_kwargs), tool_reward, res
