# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.trainer.ppo.polyseg import build_segment_features, make_segment_slices
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def build_single_turn_segment_metadata(
    response_mask: list[int],
    configured_m: int = 0,
    feature_dim: int = 4,
    feature_horizon: int | None = None,
) -> dict[str, Any]:
    """Build fixed-token PolySeg metadata for one generated response.

    ``response_mask`` is the action mask returned by the backend. Padding or
    other non-policy tokens receive ``-1`` and are excluded from boundaries.
    This pure helper keeps the rollout contract testable without starting a
    vLLM server.
    """

    response_length = len(response_mask)
    action_positions = [index for index, value in enumerate(response_mask) if bool(value)]
    segment_count = (
        min(int(configured_m), len(action_positions))
        if int(configured_m) > 0 and action_positions
        else (1 if action_positions else 0)
    )
    segment_ids = [-1] * response_length
    boundaries: list[tuple[int, int]] = []
    if action_positions:
        for segment_id, (start, end) in enumerate(make_segment_slices(len(action_positions), segment_count)):
            positions = action_positions[start:end]
            for position in positions:
                segment_ids[position] = segment_id
            boundaries.append((positions[0], positions[-1] + 1))
    # Single-turn fixed partitions have a rollout-owned successor feature for
    # every nonterminal segment: the next observed segment boundary.  Expose
    # it explicitly so the transition estimator can be run on this protocol
    # without asking the reward manager to invent a shifted state.  The final
    # segment is terminal and is masked out.
    features = build_segment_features(
        boundaries,
        [1.0] * len(boundaries),
        response_length=max(int(feature_horizon or response_length), 1),
        action_lengths=[
            sum(1 for position in action_positions if segment_ids[position] == segment_id)
            for segment_id in range(len(boundaries))
        ],
        feature_dim=max(1, int(feature_dim)),
    )
    next_features = torch.zeros_like(features)
    transition_mask = [False] * len(boundaries)
    if len(boundaries) > 1:
        next_features[:-1] = features[1:]
        transition_mask[:-1] = [True] * (len(boundaries) - 1)
    return {
        "segment_ids": segment_ids,
        "segment_durations": [1.0] * len(boundaries),
        "segment_boundaries": boundaries,
        "segment_action_lengths": [
            sum(1 for position in action_positions if segment_ids[position] == segment_id)
            for segment_id in range(len(boundaries))
        ],
        "segment_tool_calls": [[] for _ in boundaries],
        "segment_tool_observations": [[] for _ in boundaries],
        "segment_next_features": next_features.tolist(),
        "segment_transition_mask": transition_mask,
    }


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. build the initial prompt with Continuous Token (the only tokenization path).
        # Multimodal inputs require a VL builder + processor; fail loudly otherwise.
        self._assert_mm_supported(bool(multi_modal_data))
        mm_inputs = {}
        prompt_ids = await self.ct_build_initial_tokens(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_inputs_out=mm_inputs,
        )

        # Capture the processor features so SGLang can be handed the video (see
        # AgentLoopBase.build_sglang_video_payload); without this they are computed and dropped.
        mm_processor_output = self.build_sglang_video_payload(videos, mm_inputs)

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_output=mm_processor_output,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            output.token_ids,
            [],
            [] if output.log_probs else None,
            assistant_logprobs=output.log_probs if output.log_probs else None,
        )
        response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            mm_processor_output=mm_processor_output,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        response_length = len(output.response_ids)
        # Single-turn math rollouts have no environment boundary. For an
        # explicit fixed-token PolySeg experiment, construct balanced
        # boundaries over action tokens; otherwise retain one segment.
        configured_m = kwargs.get("polyseg_adaptive_m", getattr(self.rollout_config, "polyseg_fixed_m", 0))
        if hasattr(configured_m, "item"):
            configured_m = configured_m.item()
        configured_m = int(configured_m or 0)
        metadata = build_single_turn_segment_metadata(
            output.response_mask,
            configured_m,
            feature_dim=max(1, int(getattr(self.rollout_config, "polyseg_feature_dim", 4))),
            feature_horizon=self.response_length,
        )
        output.extra_fields.update(
            {
                **metadata,
                "turn_scores": [],
                "tool_rewards": [],
                "trajectory_id": request_id,
                "episode_id": request_id,
                "segment_state_ids": [f"{request_id}:s{segment_id}" for segment_id in range(len(metadata["segment_boundaries"]))],
                "segment_action_texts": [
                    self.tokenizer.decode(output.response_ids[start:end], skip_special_tokens=False)
                    for start, end in metadata["segment_boundaries"]
                ],
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

        return output
