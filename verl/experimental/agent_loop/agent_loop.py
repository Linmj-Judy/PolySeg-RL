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
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.experimental.agent_loop.boundary_state import load_boundary_state_provider
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.polyseg import (
    BudgetedPolySegController,
    FrozenLowDimRepresentation,
    PolySegConfig,
    load_frozen_representation,
    resolve_polyseg_option,
)
from verl.trainer.ppo.polyseg_sci import ConstraintDualState, _science_objective_mode, load_endpoint_schema
from verl.trainer.ppo.polyseg_sci_provider import (
    ScienceRoundRequest,
    attach_science_response,
    load_science_round_provider,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.import_utils import resolve_config_path
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.skip import SkipManager
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
)
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000

# PolySeg metadata has two ragged axes: token-aligned ``segment_ids`` and
# macro-aligned fields.  Keep this schema in one place so legacy, v1, and
# TransferQueue adapters use the same names and dtypes.
POLYSEG_METADATA_DTYPES = {
    "segment_rewards": torch.float32,
    "segment_ids": torch.long,
    "segment_durations": torch.float32,
    "segment_mask": torch.long,
    "segment_feedback_mask": torch.long,
    "segment_features": torch.float32,
    # Optional rollout-provided frozen per-step representation used by the
    # shared-step reward route.  It is ragged along the token axis.
    "step_features": torch.float32,
    "segment_next_features": torch.float32,
    "segment_transition_mask": torch.long,
    # Explicit token-level observation masks are rollout-owned.  The segment
    # action-mask lists remain ragged and are preserved in extra_fields.
    "tool_observation_mask": torch.bool,
}

# PolySeg-Sci keeps vector endpoint values and objective state on the rollout
# side.  These fields are intentionally separate from scalar reward metadata:
# the science estimator must fail closed when an endpoint or its mask is lost
# while crossing the AgentLoop/TransferQueue boundary.
POLYSEG_SCI_METADATA_DTYPES = {
    "science_boundary_values": torch.float32,
    "science_boundary_mask": torch.bool,
    "science_endpoint_mask": torch.bool,
    "science_pareto_alpha": torch.float32,
    "science_constraint_multipliers": torch.float32,
    "science_endpoint_estimates": torch.float32,
}

ROLLOUT_METADATA_DTYPES = {**POLYSEG_METADATA_DTYPES, **POLYSEG_SCI_METADATA_DTYPES}


def _polyseg_metadata_value(extra_fields: dict[str, Any], key: str):
    """Return rollout metadata from direct or nested reward fields."""
    value = extra_fields.get(key)
    if value is not None:
        return value
    reward_info = extra_fields.get("reward_extra_info")
    if isinstance(reward_info, dict):
        return reward_info.get(key)
    return None


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""
    mm_processor_output: Optional[list[dict[str, Any]]] = None
    """SGLang video payload (see AgentLoopBase.build_sglang_video_payload); reused by the teacher
    (OPD) path so an SGLang teacher receives the video instead of raw frames. None otherwise."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            routed_experts = torch.tensor(routed_experts, dtype=torch.int16)
            # Router replay indexes this field by absolute token position, so it must
            # span the whole sequence. The rollout engine records fewer rows than that:
            # it only sees tokens fed through the model, and multi-turn loops stop
            # recording at the last generation. Trailing rows stay zero; replay masks
            # them out instead of consuming them.
            total_length = output["prompts"].size(0) + output["responses"].size(0)
            aligned = routed_experts.new_zeros((total_length, *routed_experts.shape[1:]))
            num_rows = min(routed_experts.size(0), total_length)
            aligned[:num_rows] = routed_experts[:num_rows]
            output["routed_experts"] = aligned

        # Promote rollout-owned PolySeg metadata before reward handling. The
        # actor loss must consume these ids exactly as produced by rollout; it
        # must never infer event boundaries from token counts.
        segment_ids = _polyseg_metadata_value(self.extra_fields, "segment_ids")
        if segment_ids is not None:
            segment_ids = np.array(segment_ids, copy=True).reshape(-1)
            response_length = output["responses"].numel()
            if segment_ids.size != response_length:
                raise ValueError(
                    "rollout segment_ids must be token-aligned with responses: "
                    f"got {segment_ids.size} ids for {response_length} response tokens"
                )
            output["segment_ids"] = torch.as_tensor(segment_ids, dtype=torch.long)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            reward_extra_info = self.extra_fields.get("reward_extra_info", {})
            segment_rewards = _polyseg_metadata_value(self.extra_fields, "segment_rewards")
            segment_ids = _polyseg_metadata_value(self.extra_fields, "segment_ids")
            if segment_rewards is not None and segment_ids is not None:
                segment_rewards = torch.as_tensor(np.array(segment_rewards, copy=True), dtype=rm_scores.dtype)
                segment_ids = torch.as_tensor(np.array(segment_ids, copy=True), dtype=torch.long)
                action_mask = output["response_mask"].to(torch.bool)
                for segment_id in range(segment_rewards.numel()):
                    positions = torch.nonzero((segment_ids == segment_id) & action_mask, as_tuple=False).flatten()
                    if positions.numel():
                        rm_scores[int(positions[-1])] = segment_rewards[segment_id]
            else:
                rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

            # The transfer-queue (v1) trainer receives the reward result through
            # ``extra_fields`` because the asynchronous AgentLoop computes the
            # score after the rollout tensor is created.  Promote PolySeg's
            # ragged macro metadata to tensor fields as well, so downstream
            # advantage computation can request it directly from the queue.
            # Keeping the original nested copy in ``extra_fields`` preserves
            # compatibility with validation/logging consumers.
            if isinstance(reward_extra_info, dict):
                for key, dtype in ROLLOUT_METADATA_DTYPES.items():
                    value = _polyseg_metadata_value(self.extra_fields, key)
                    if value is None:
                        continue
                    # Copy NumPy views before conversion: reward managers often
                    # return arrays backed by a read-only transport buffer.
                    if isinstance(value, np.ndarray):
                        value = np.array(value, copy=True)
                    output[key] = torch.as_tensor(value, dtype=dtype)

        # Promote direct rollout metadata too, including asynchronous reward
        # results that are not present when the output object is first built.
        for key, dtype in ROLLOUT_METADATA_DTYPES.items():
            value = _polyseg_metadata_value(self.extra_fields, key)
            if value is None:
                continue
            value = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            if key == "step_features":
                step_tensor = torch.as_tensor(value, dtype=dtype)
                if step_tensor.ndim != 2 or step_tensor.shape[0] != output["responses"].numel():
                    raise ValueError(
                        "rollout step_features must be [response_tokens, feature_dim] and "
                        "token-aligned with responses"
                    )
            if key == "tool_observation_mask":
                observation_tensor = torch.as_tensor(value, dtype=dtype).reshape(-1)
                if observation_tensor.numel() != output["responses"].numel():
                    raise ValueError(
                        "rollout tool_observation_mask must be token-aligned with responses"
                    )
            output[key] = torch.as_tensor(value, dtype=dtype)

        # Mutating a default dict does not add the field to Pydantic's fields-set,
        # so model_dump(exclude_unset=True) can omit populated extra fields.
        extra_fields = output.setdefault("extra_fields", self.extra_fields.copy())
        teacher_ids, teacher_logprobs = (
            extra_fields.pop("teacher_ids", None),
            extra_fields.pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        response_topk_ids, response_topk_log_probs = (
            extra_fields.pop("response_topk_ids", None),
            extra_fields.pop("response_topk_log_probs", None),
        )
        if response_topk_ids is not None and response_topk_log_probs is not None:
            from verl.trainer.ppo.score_centering import pad_rollout_topk

            if self.num_turns > 2:
                raise ValueError("rollout.topk_log_probs supports the single-turn agent loop only.")
            if len(response_topk_ids) != output["responses"].size(0):
                raise ValueError(
                    f"sampler top-k heads cover {len(response_topk_ids)} tokens, "
                    f"but the response has {output['responses'].size(0)}."
                )
            output["rollout_topk_ids"], output["rollout_topk_log_probs"] = pad_rollout_topk(
                response_topk_ids, response_topk_log_probs, prompt_length=output["prompts"].size(0)
            )
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
        hf_model_type: Root Hugging Face ``model_type`` used for Continuous Token builder selection.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        hf_model_type: str | None = None,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        # Continuous Token is the only rollout tokenization path for agent loops.
        # The model family (boundary handling) is inferred by exact lookup of the
        # root Hugging Face config's model_type. Unrecognized models use the
        # default builder (or default VL builder with a multimodal processor).
        self.continuous_token_builder = create_continuous_token_builder(
            self.tokenizer,
            hf_model_type=hf_model_type,
            chat_template_kwargs=self.apply_chat_template_kwargs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            processor=self.processor,
        )
        self.loop = get_event_loop()
        # A representation is immutable for the lifetime of a rollout worker.
        # Cache it after the first request so every trajectory uses the exact
        # same normalization/projection and fingerprint.
        self._polyseg_representation_cache: FrozenLowDimRepresentation | None = None
        self._polyseg_representation_cache_path: str | None = None

    def _assert_mm_supported(self, has_multi_modal: bool) -> None:
        """Fail loudly when multimodal inputs are present but unsupported.

        Multimodal rollout requires both a Continuous Token builder that supports
        multimodal boundaries and a non-None processor to render placeholder spans.
        Silent fallback is not allowed; callers must invoke this before mutating any
        rollout state so failures never leave a half-built prompt behind.
        """
        if not has_multi_modal:
            return
        if not (self.continuous_token_builder.supports_multimodal() and self.processor is not None):
            raise ValueError(
                "Multimodal inputs require a Continuous Token builder that supports multimodal "
                "AND a non-None processor, but got "
                f"supports_multimodal={self.continuous_token_builder.supports_multimodal()}, "
                f"processor={'set' if self.processor is not None else 'None'}. "
                "Use a VL base model (with its processor) or remove multimodal inputs."
            )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def ct_build_initial_tokens(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_inputs_out: dict[str, Any] = None,
    ) -> list[int]:
        """Build the initial prompt token ids with Continuous Token.

        Multimodal inputs are forwarded to the builder so that VL builders can
        render placeholder spans through the processor. Text-only builders accept
        and ignore these arguments. ``mm_processor_kwargs`` is not threaded here:
        it is a builder-lifetime constant captured at construction and applied by
        the builder itself during rendering.
        """
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.build_initial_tokens(
                messages,
                tools=tools,
                images=images,
                videos=videos,
                audios=audios,
                mm_inputs_out=mm_inputs_out,
            ),
        )

        # Mirror the response-side ``response_ids[:response_length]`` cap on the prompt side:
        # every prompt produced by the agent loop must fit in ``rollout.prompt_length`` so that
        # ``_pad_token_ids`` (and downstream ``torch.cat``) can rely on uniform shapes.
        # Multimodal prompts cannot be sliced here because placeholder tokens must remain
        # aligned 1:1 with ``multi_modal_inputs`` features, so we fail loudly instead.
        prompt_length = self.rollout_config.prompt_length
        if (images or videos or audios) and len(prompt_ids) > prompt_length:
            raise ValueError(
                f"Multimodal prompt produced {len(prompt_ids)} tokens, exceeding "
                f"rollout.prompt_length={prompt_length}. Truncating multimodal token "
                f"sequences corrupts vision/audio feature alignment, so this is treated "
                f"as a configuration error. Reduce the multimodal input size "
                f"(e.g. ``total_pixels`` / ``max_pixels`` / fps / number of frames) or "
                f"increase ``rollout.prompt_length``."
            )
        return self._cap_text_prompt_length(prompt_ids)

    @staticmethod
    def build_sglang_video_payload(videos, mm_inputs: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
        """Wrap already-computed video features for SGLang's ``video_data``.

        SGLang cannot ingest raw frames — its ``video_data`` accepts a path / URL / base64 string or
        a dict. ``mm_inputs`` is the processor output captured during tokenization (via the
        ``mm_inputs_out`` hook). Tagging it ``format="processor_output"`` makes
        ``BaseMultimodalProcessor`` pass the dict through untouched instead of re-decoding the video,
        which would re-sample frames and desync the rollout from the actor. Returns ``None`` when
        there is no video, so callers can hand the result straight to ``generate(...)``.

        Shared by SingleTurnAgentLoop and ToolAgentLoop; vLLM keeps the raw frames and ignores this
        (the per-backend split lives in ``LLMServerClient.generate``).
        """
        if not videos or "pixel_values_videos" not in mm_inputs:
            return None
        payload = {
            "format": "processor_output",
            "pixel_values_videos": mm_inputs["pixel_values_videos"],
            "video_grid_thw": mm_inputs["video_grid_thw"],
        }
        if "second_per_grid_ts" in mm_inputs:
            payload["second_per_grid_ts"] = mm_inputs["second_per_grid_ts"]
        return [payload]

    async def ct_merge_context_msg(
        self,
        previous_messages: list[dict],
        updated_messages: list[dict],
        runtime_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        tools: list[dict] = None,
    ):
        """Merge appended context messages into runtime tokens and metadata."""
        merge_result = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.merge_context_tokens(
                previous_messages,
                updated_messages,
                runtime_token_ids,
                tools=tools,
            ),
        )
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(
            merge_result, response_mask, response_logprobs
        )
        return merge_result, aligned_response_mask, aligned_response_logprobs

    async def ct_merge_assistant_token(
        self,
        runtime_token_ids: list[int],
        assistant_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        assistant_logprobs: Optional[list[float]] = None,
    ):
        """Merge assistant-generated tokens and align response metadata."""
        merge_result = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.merge_assistant_tokens(
                runtime_token_ids,
                assistant_token_ids,
            ),
        )
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(
            merge_result,
            response_mask,
            response_logprobs,
            assistant_logprobs=assistant_logprobs,
        )
        return merge_result, aligned_response_mask, aligned_response_logprobs

    @staticmethod
    def align_segment_ids(merge_result, segment_ids: list[int], segment_id: int) -> list[int]:
        """Keep rollout-owned segment ids aligned with Continuous Token edits.

        Tool/context tokens are assigned to the assistant action segment that
        caused the tool call.  This makes a segment span action, tool call and
        observation while ``response_mask`` still excludes observations from
        the actor loss.
        """

        aligned = list(segment_ids)
        # A few downstream integrations provide a lightweight merge result
        # object in tests or custom agent loops.  The production
        # ``MergeResult`` always has these fields; when an integration omits
        # them, leave its existing metadata untouched instead of fabricating
        # token alignment from the full runtime sequence.
        if not hasattr(merge_result, "appended_token_count"):
            return aligned
        removed = int(getattr(merge_result, "removed_prefix_token_count", 0))
        if removed:
            aligned = aligned[: -removed]
        inserted = getattr(merge_result, "inserted_token_ids", ())
        aligned += [segment_id] * len(inserted)
        aligned += [segment_id] * int(merge_result.appended_token_count)
        return aligned

    def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            logger.warning(
                "Prompt of %d tokens exceeds rollout.prompt_length=%d; left-truncating.",
                len(prompt_ids),
                prompt_length,
            )
            return prompt_ids[-prompt_length:]
        return prompt_ids

    def _load_polyseg_representation(self) -> FrozenLowDimRepresentation | None:
        """Load and validate the rollout-side frozen PolySeg representation.

        Hidden states are only a valid fixed feature map when the checkpoint is
        explicit and its content hash matches the configured value.  An
        already mapped ``feature_dim`` vector can be supplied without a file;
        in that case it is treated as an externally frozen representation.
        """

        path = getattr(self.rollout_config, "polyseg_representation_path", None)
        configured_hash = str(getattr(self.rollout_config, "polyseg_representation_hash", "") or "")
        if path is None or str(path).strip() == "":
            if configured_hash:
                raise ValueError(
                    "polyseg_representation_hash is set but polyseg_representation_path is empty"
                )
            return None
        path = os.path.expanduser(str(path))
        if self._polyseg_representation_cache is None or self._polyseg_representation_cache_path != path:
            representation = load_frozen_representation(path)
            if configured_hash and representation.fingerprint != configured_hash:
                raise ValueError(
                    "frozen PolySeg representation hash does not match "
                    "rollout.polyseg_representation_hash"
                )
            self._polyseg_representation_cache = representation
            self._polyseg_representation_cache_path = path
        elif configured_hash and self._polyseg_representation_cache.fingerprint != configured_hash:
            raise ValueError("cached PolySeg representation hash does not match configured hash")
        return self._polyseg_representation_cache

    def _prepare_polyseg_step_features(
        self, kwargs: dict[str, Any], response_length: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, FrozenLowDimRepresentation | None, str | None]:
        """Validate external per-token state features for a rollout.

        Returns mapped features, the state matrix to use for segment pooling,
        the frozen representation (if raw hidden states were supplied), and
        its fingerprint.  The aliases keep dataset integrations compatible
        while the canonical transport field remains ``step_features``.
        """

        value = None
        source_key = None
        sources = [kwargs]
        nested_extra = kwargs.get("extra_info")
        if isinstance(nested_extra, Mapping):
            sources.append(nested_extra)
        for key in ("polyseg_step_features", "step_features", "polyseg_hidden_states", "hidden_states"):
            for source in sources:
                if key in source and source[key] is not None:
                    value = source[key]
                    source_key = key
                    break
            if value is not None:
                break
        configured_representation = self._load_polyseg_representation()
        if value is None:
            if configured_representation is not None:
                raise ValueError(
                    "rollout PolySeg frozen representation is configured, but no per-token "
                    "step_features/hidden_states were supplied"
                )
            return None, None, None, None
        if response_length < 1:
            raise ValueError("PolySeg step features cannot be attached to an empty response")
        try:
            raw = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source_key} must be a finite numeric feature matrix") from exc
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw.squeeze(0)
        if raw.ndim == 1:
            raw = raw.unsqueeze(-1)
        if raw.ndim != 2 or raw.shape[0] < response_length or raw.shape[1] < 1:
            raise ValueError(
                f"{source_key} must have shape [response_tokens, feature_dim] with at least "
                f"{response_length} rows; got {tuple(raw.shape)}"
            )
        raw = raw[:response_length].contiguous()
        if not torch.isfinite(raw).all():
            raise ValueError(f"{source_key} contains non-finite values")
        representation = configured_representation
        expected_dim = max(1, int(getattr(self.rollout_config, "polyseg_feature_dim", 4)))
        if representation is None:
            if raw.shape[1] != expected_dim:
                raise ValueError(
                    f"{source_key} has width {raw.shape[1]}, but rollout.polyseg_feature_dim="
                    f"{expected_dim}; configure a frozen representation for raw hidden states"
                )
            if torch.any(raw < -1.0) or torch.any(raw > 1.0):
                raise ValueError(
                    f"{source_key} must already be normalized to [-1, 1] when no frozen "
                    "representation checkpoint is configured"
                )
            mapped = raw
            state = mapped
            fingerprint = None
        else:
            if raw.shape[1] != representation.input_dim:
                raise ValueError(
                    f"{source_key} width {raw.shape[1]} does not match frozen representation "
                    f"input_dim={representation.input_dim}"
                )
            mapped = representation.transform(raw)
            if mapped.shape[1] != expected_dim:
                raise ValueError(
                    "frozen representation output dimension does not match "
                    "rollout.polyseg_feature_dim"
                )
            state = raw
            fingerprint = representation.fingerprint
        requested_hash = kwargs.get("polyseg_representation_hash")
        if requested_hash is None and isinstance(nested_extra, Mapping):
            requested_hash = nested_extra.get("polyseg_representation_hash")
        if requested_hash is not None:
            requested_hash = str(requested_hash.item() if hasattr(requested_hash, "item") else requested_hash)
            if not fingerprint or requested_hash != fingerprint:
                raise ValueError("per-sample PolySeg representation hash failed closed validation")
        return mapped, state, representation, fingerprint

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        # PolySeg-Sci must run on both the ordinary colocated AgentLoop path
        # and the TransferQueue adapter.  Keep the provider explicit and
        # fail closed when science is enabled without a configured harness;
        # disabled scalar PolySeg runs still take the zero-cost ``None`` path.
        self.science_round_provider = load_science_round_provider(config)
        # Boundary snapshots are worker-owned runtime artifacts. Keep the
        # provider handle out of dataset metadata and inject it into each
        # tool-agent loop instance alongside the science provider seam.
        self.boundary_state_provider = load_boundary_state_provider(config)

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        hf_model_type = getattr(self.model_config.hf_config, "model_type", None)
        self.hf_model_type: str | None = hf_model_type if isinstance(hf_model_type, str) else None
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        if trace_config.get("token2text", False):
            # rollout_trace_op runs on the LLM client, so provide the tokenizer
            # needed to decode each generate call's prompt and response tokens.
            self.llm_client.tokenizer = self.tokenizer
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        if not isinstance(validate, bool):
            raise ValueError("rollout validate flag must be boolean")
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                hf_model_type=self.hf_model_type,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
                # Some lightweight unit-test workers are constructed with
                # ``object.__new__`` and predate the worker-owned provider
                # attribute.  Preserve the disabled/optional path there;
                # required snapshot runs still fail closed during normal
                # worker construction and ToolAgentLoop capture.
                boundary_state_provider=getattr(self, "boundary_state_provider", None),
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)

    def _pad_token_ids(
        self,
        tokens: list[int],
        *,
        max_length: int,
        padding_side: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        """Right/left pad a flat list of token ids to a ``(1, max_length)`` tensor."""
        # tokenizer.pad() with empty input returns dict with list values
        # instead of tensors, which breaks downstream .dim() calls.
        if not tokens:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            result = {"input_ids": torch.full((1, max_length), pad_id, dtype=torch.long)}
            if return_attention_mask:
                result["attention_mask"] = torch.zeros((1, max_length), dtype=torch.long)
            return result
        self.tokenizer.padding_side = padding_side
        padded = self.tokenizer.pad(
            {"input_ids": tokens},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=return_attention_mask,
        )
        if padded["input_ids"].dim() == 1:
            padded["input_ids"] = padded["input_ids"].unsqueeze(0)
            if return_attention_mask:
                padded["attention_mask"] = padded["attention_mask"].unsqueeze(0)
        return padded

    async def _apply_science_round(self, outputs: list[AgentLoopOutput], *, validate: bool, kwargs: dict[str, Any]) -> None:
        """Run the configured PolySeg-Sci provider for the ordinary worker path.

        The TransferQueue worker has a parallel implementation because it
        persists outputs asynchronously.  Keeping the provider call here as
        a small shared seam makes the colocated/legacy path obey the same
        endpoint-vector and fail-closed contract without putting evaluator or
        restore side effects into advantage computation.
        """

        if not isinstance(validate, bool):
            raise ValueError("science rollout validate flag must be boolean")
        if getattr(self, "science_round_provider", None) is None:
            return
        # Validation batches do not participate in the training advantage
        # update.  A real OpenAI4S provider intentionally refuses validation
        # rounds because they would spend training-audit budget without a
        # frozen-policy correction record; keep that policy consistent on the
        # ordinary colocated path.
        if validate:
            return
        if not outputs:
            raise ValueError("PolySeg-Sci requires at least one AgentLoop output")
        request = ScienceRoundRequest(
            output=outputs[-1],
            outputs=tuple(outputs),
            validate=validate,
            context={**kwargs, "validate": validate},
        )
        response = await self.science_round_provider.execute(request)
        attach_science_response(outputs, response)

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        if not isinstance(validate, bool):
            raise ValueError("rollout validate flag must be boolean")
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
        # Driver-owned science objective state must survive both the colocated
        # reward path and the asynchronous TransferQueue path.  Agent-loop
        # implementations are free to ignore unknown kwargs, so copy only the
        # registered vector contract into the durable extra-field envelope.
        for key in POLYSEG_SCI_METADATA_DTYPES:
            if key in kwargs and key not in output.extra_fields:
                value = kwargs[key]
                output.extra_fields[key] = value.tolist() if isinstance(value, np.ndarray) else value

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        prompt_output = self._pad_token_ids(
            output.prompt_ids,
            max_length=self.rollout_config.prompt_length,
            padding_side="left",
            return_attention_mask=True,
        )

        response_output = self._pad_token_ids(
            output.response_ids,
            max_length=self.rollout_config.response_length,
            padding_side="right",
            return_attention_mask=True,
        )

        response_mask_output = self._pad_token_ids(
            output.response_mask,
            max_length=self.rollout_config.response_length,
            padding_side="right",
            return_attention_mask=False,
        )

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            experts_tensor = experts_tensor.to(torch.int16)
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )
        await self._compute_score([output], kwargs={**kwargs, "validate": validate})
        apply_science_round = getattr(self, "_apply_science_round", None)
        if apply_science_round is not None:
            await apply_science_round([output], validate=validate, kwargs=kwargs)
        await self._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        # Collapse expanded vision placeholder runs back to a single placeholder
        # per media item before decoding, so the processor re-expands cleanly.
        # ``input_ids`` already contains fully expanded image/video pad tokens; if
        # those placeholders are NOT registered as special tokens (e.g. Kimi-VL
        # ``<|media_pad|>``, GLM-4.6V ``<|image|>``), ``skip_special_tokens=True``
        # will not strip them, and re-feeding the already-expanded text to the
        # processor triggers a double-expansion error. Collapsing at the token-id
        # level (using the processor-resolved placeholder ids) is model-agnostic
        # and leaves the well-behaved (special-token) placeholders untouched.
        image_token_id = get_processor_token_id(self.processor, "image")
        video_token_id = get_processor_token_id(self.processor, "video")
        collapse_ids = {t for t in (image_token_id, video_token_id) if t is not None}
        collapsed_ids, prev_id = [], None
        for token_id in input_ids.squeeze(0).tolist():
            if token_id in collapse_ids and token_id == prev_id:
                continue
            collapsed_ids.append(token_id)
            prev_id = token_id
        current_text = self.tokenizer.decode(collapsed_ids, skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        # text-only OR non-M-RoPE multimodal (e.g. Gemma4) -> standard 1D positions
        if self.processor is None or not hasattr(self.processor, "get_rope_index"):
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Allow model-specific processors to contribute additional RoPE inputs.
        get_rope_index_kwargs = getattr(self.processor, "get_rope_index_kwargs", None)
        if get_rope_index_kwargs is not None:
            multi_modal_kwargs.update(get_rope_index_kwargs(multi_modal_inputs))

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "response_mask": torch.nn.utils.rnn.pad_sequence(
                            [torch.as_tensor(output.response_mask, dtype=torch.long) for output in outputs],
                            batch_first=True,
                            padding_value=0,
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and not validate:
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                mm_processor_output=getattr(output, "mm_processor_output", None),
                routing_key=routing_key,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if inputs[0].teacher_logprobs is not None and inputs[0].teacher_ids is not None:
            optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
            optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)

        # Preserve the rollout-owned token boundary schema in the legacy batch.
        # Pad only the right edge; ``-1`` remains the canonical id for response
        # padding/observation positions and is never used to infer boundaries.
        raw_segment_ids = []
        missing_segment_ids = False
        for input_item in inputs:
            value = _polyseg_metadata_value(input_item.extra_fields, "segment_ids")
            if value is None:
                missing_segment_ids = True
                continue
            ids = torch.as_tensor(np.array(value, copy=True), dtype=torch.long).flatten()
            if ids.numel() > response_mask.shape[1]:
                raise ValueError("rollout segment_ids cannot exceed the padded response width")
            padded = torch.full((response_mask.shape[1],), -1, dtype=torch.long)
            padded[: ids.numel()] = ids
            raw_segment_ids.append(padded)
        if missing_segment_ids:
            if raw_segment_ids:
                raise ValueError("PolySeg segment_ids must be present for every sample in a legacy batch")
        elif raw_segment_ids:
            optional_outputs["segment_ids"] = torch.stack(raw_segment_ids, dim=0)

        # Per-token frozen features are ragged in the response axis.  Pad only
        # the right edge to the rollout response width; the actor/reward path
        # uses response_mask and segment_ids to exclude those rows.  Mixed
        # presence would make shared-step recovery silently drop trajectories,
        # so reject it at the transport boundary.
        raw_step_features = [_polyseg_metadata_value(item.extra_fields, "step_features") for item in inputs]
        if raw_step_features and any(value is not None for value in raw_step_features):
            if any(value is None for value in raw_step_features):
                raise ValueError("step_features must be present for every rollout in a shared-step batch")
            feature_tensors = []
            feature_dim = None
            for value in raw_step_features:
                tensor = torch.as_tensor(value, dtype=torch.float32)
                if tensor.ndim == 3 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.ndim != 2:
                    raise ValueError("step_features must be [response_tokens, feature_dim]")
                if tensor.shape[0] > response_mask.shape[1] or tensor.shape[1] < 1:
                    raise ValueError("step_features exceed the padded response width or have an empty feature axis")
                if not torch.isfinite(tensor).all():
                    raise ValueError("step_features must contain finite values")
                if feature_dim is None:
                    feature_dim = int(tensor.shape[1])
                elif int(tensor.shape[1]) != feature_dim:
                    raise ValueError("step_features feature dimensions must agree across the rollout batch")
                padded = torch.zeros((response_mask.shape[1], feature_dim), dtype=torch.float32)
                padded[: tensor.shape[0]] = tensor
                feature_tensors.append(padded)
            optional_outputs["step_features"] = torch.stack(feature_tensors, dim=0)

        # Science vectors use a macro axis that is naturally ragged when
        # causal trajectories terminate at different lengths.  Pad only that
        # structural axis and carry explicit masks; padded values repeat the
        # last observed boundary solely to keep the tensor finite and are
        # excluded by ``science_boundary_mask``/``science_endpoint_mask``.
        raw_values = [_polyseg_metadata_value(item.extra_fields, "science_boundary_values") for item in inputs]
        if raw_values and any(value is not None for value in raw_values):
            if any(value is None for value in raw_values):
                raise ValueError("science_boundary_values must be present for every rollout in the batch")
            value_tensors = [torch.as_tensor(value, dtype=torch.float32) for value in raw_values]
            if any(tensor.ndim != 2 or tensor.shape[0] < 2 for tensor in value_tensors):
                raise ValueError("science_boundary_values must be [segments+1, endpoints]")
            endpoint_dim = value_tensors[0].shape[1]
            if any(tensor.shape[1] != endpoint_dim or not torch.isfinite(tensor).all() for tensor in value_tensors):
                raise ValueError("science boundary endpoint vectors must be finite and aligned")
            max_boundaries = max(tensor.shape[0] for tensor in value_tensors)
            padded_values = []
            padded_segment_masks = []
            padded_endpoint_masks = []
            for item, tensor in zip(inputs, value_tensors, strict=True):
                segments = tensor.shape[0] - 1
                boundary_mask_value = _polyseg_metadata_value(item.extra_fields, "science_boundary_mask")
                segment_mask = torch.ones(segments, dtype=torch.bool) if boundary_mask_value is None else torch.as_tensor(
                    boundary_mask_value, dtype=torch.bool
                ).reshape(-1)
                if segment_mask.numel() != segments:
                    raise ValueError("science_boundary_mask must match the segment axis")
                endpoint_mask_value = _polyseg_metadata_value(item.extra_fields, "science_endpoint_mask")
                endpoint_mask = torch.ones_like(tensor, dtype=torch.bool) if endpoint_mask_value is None else torch.as_tensor(
                    endpoint_mask_value, dtype=torch.bool
                )
                if endpoint_mask.shape != tensor.shape:
                    raise ValueError("science_endpoint_mask must match boundary values")
                pad_rows = max_boundaries - tensor.shape[0]
                if pad_rows:
                    tensor = torch.cat((tensor, tensor[-1:].expand(pad_rows, -1)), dim=0)
                    endpoint_mask = torch.cat(
                        (endpoint_mask, torch.zeros(pad_rows, endpoint_dim, dtype=torch.bool)), dim=0
                    )
                    segment_mask = torch.cat((segment_mask, torch.zeros(pad_rows, dtype=torch.bool)), dim=0)
                padded_values.append(tensor)
                padded_endpoint_masks.append(endpoint_mask)
                padded_segment_masks.append(segment_mask)
            optional_outputs["science_boundary_values"] = torch.stack(padded_values, dim=0)
            optional_outputs["science_boundary_mask"] = torch.stack(padded_segment_masks, dim=0)
            optional_outputs["science_endpoint_mask"] = torch.stack(padded_endpoint_masks, dim=0)

        for key, dtype in POLYSEG_SCI_METADATA_DTYPES.items():
            if key in {"science_boundary_values", "science_boundary_mask", "science_endpoint_mask"}:
                continue
            raw_values = [_polyseg_metadata_value(item.extra_fields, key) for item in inputs]
            if not raw_values or any(value is None for value in raw_values):
                continue
            tensors = [torch.as_tensor(value, dtype=dtype) for value in raw_values]
            if all(tensor.shape == tensors[0].shape for tensor in tensors[1:]):
                optional_outputs[key] = torch.stack(tensors, dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            for row, (score, input_item) in enumerate(zip(scores, inputs, strict=True)):
                reward_info = input_item.extra_fields.get("reward_extra_info", {})
                segment_rewards = reward_info.get("segment_rewards") if isinstance(reward_info, dict) else None
                segment_ids = reward_info.get("segment_ids") if isinstance(reward_info, dict) else None
                if segment_rewards is not None and segment_ids is not None:
                    reward_values = torch.as_tensor(np.array(segment_rewards, copy=True), dtype=rm_scores.dtype)
                    ids = torch.as_tensor(np.array(segment_ids, copy=True), dtype=torch.long)
                    actions = response_mask[row].to(torch.bool)
                    if ids.numel() < actions.numel():
                        padded_ids = torch.full((actions.numel(),), -1, dtype=torch.long)
                        padded_ids[: ids.numel()] = ids
                        ids = padded_ids
                    else:
                        ids = ids[: actions.numel()]
                    for segment_id in range(reward_values.numel()):
                        positions = torch.nonzero((ids == segment_id) & actions, as_tuple=False).flatten()
                        if positions.numel():
                            rm_scores[row, int(positions[-1])] = reward_values[segment_id]
                else:
                    response_length = attention_mask[row, prompt_ids.size(1) :].sum() - 1
                    rm_scores[row, int(response_length)] = float(score)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        # Keep explicit tool accounting as numeric batch fields so the
        # trainer's common metric reducer can report it on every full run.
        non_tensor_batch["tool_call_counts"] = np.asarray(
            [int(input.extra_fields.get("tool_call_count", 0) or 0) for input in inputs], dtype=np.int32
        )
        non_tensor_batch["tool_call_success_counts"] = np.asarray(
            [int(input.extra_fields.get("tool_call_success_count", 0) or 0) for input in inputs], dtype=np.int32
        )
        non_tensor_batch["tool_call_failure_counts"] = np.asarray(
            [int(input.extra_fields.get("tool_call_failure_count", 0) or 0) for input in inputs], dtype=np.int32
        )
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # Add reward extras using a 1-D object array for ragged PolySeg
        # metadata.  ``np.array`` would either collapse equal-shaped nested
        # values or raise on mixed segment counts, producing backend-specific
        # schemas before the actor ever sees the batch.
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = sorted({key for info in reward_extra_infos if isinstance(info, dict) for key in info})
        for key in reward_extra_keys:
            values = [info.get(key) if isinstance(info, dict) else None for info in reward_extra_infos]
            if key in ROLLOUT_METADATA_DTYPES:
                value_array = np.empty(len(values), dtype=object)
                value_array[:] = [np.array(value, copy=True) if isinstance(value, np.ndarray) else value for value in values]
                non_tensor_batch[key] = value_array
            else:
                non_tensor_batch[key] = np.array(values)

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        # PolySeg-Sci constraint multipliers live on the driver alongside the
        # adaptive selector.  Rollout workers only receive serialized values;
        # they must never create worker-local dual states because that would
        # make the primal/dual update depend on worker count and scheduling.
        self.science_endpoint_schema = None
        self.science_dual_state: ConstraintDualState | None = None
        self._initialize_science_dual_state(config)

        # The adaptive selector is deliberately driver-side. Rollout workers
        # process chunks independently, so a worker-local ledger would multiply
        # the configured feedback budget by the worker count.
        self.polyseg_controller: BudgetedPolySegController | None = None
        adaptive = resolve_polyseg_option(config, "adaptive", default=False)
        if not isinstance(adaptive, bool):
            raise ValueError("PolySeg adaptive flag must be boolean")
        if adaptive:
            selector_config = PolySegConfig.from_any(getattr(config, "algorithm", None))
            # Algorithm config is canonical; rollout aliases are accepted for
            # launcher-only overrides when the nested algorithm value keeps
            # its backward-compatible default.
            rollout_compute_cost = getattr(self.rollout_config, "polyseg_critic_compute_cost", None)
            rollout_compute_budget = getattr(self.rollout_config, "polyseg_compute_budget", None)
            critic_compute_cost = selector_config.critic_compute_cost
            if critic_compute_cost == 0.0 and rollout_compute_cost not in (None, 0, 0.0):
                critic_compute_cost = float(rollout_compute_cost)
            compute_budget = selector_config.compute_budget
            if compute_budget is None and rollout_compute_budget is not None:
                compute_budget = float(rollout_compute_budget)
            self.polyseg_controller = BudgetedPolySegController(
                q_candidates=resolve_polyseg_option(
                    config, "q_candidates", default=[0, 1, 2, 4]
                ),
                m_candidates=resolve_polyseg_option(
                    config, "m_candidates", default=[1, 2, 4, 8]
                ),
                total_budget=resolve_polyseg_option(
                    config, "budget", rollout_alias="polyseg_total_budget", default=None
                ),
                segment_cost=float(
                    resolve_polyseg_option(config, "segment_cost", default=1.0)
                ),
                selector_norm=str(
                    resolve_polyseg_option(config, "selector_norm", rollout_alias="polyseg_selector_norm", default="effective_round")
                ),
                budget_fallback=str(
                    resolve_polyseg_option(config, "budget_fallback", default="error")
                ),
                dependence_chi=selector_config.dependence_chi,
                stat_exponent_u=selector_config.stat_exponent_u,
                approx_exponent_v=selector_config.approx_exponent_v,
                segment_exponent_gamma=selector_config.segment_exponent_gamma,
                stat_constant=selector_config.stat_constant,
                approx_constant=selector_config.approx_constant,
                segment_constant=selector_config.segment_constant,
                r_max=selector_config.r_max,
                cost_penalty=selector_config.cost_penalty,
                critic_compute_cost=critic_compute_cost,
                compute_budget=compute_budget,
            )

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @staticmethod
    def _config_value(source: Any, name: str, default: Any = None) -> Any:
        """Read a value from OmegaConf, a mapping, or a dataclass config."""

        if source is None:
            return default
        try:
            value = source.get(name, default)
        except (AttributeError, TypeError):
            value = getattr(source, name, default)
        # OmegaConf's ``None`` is a real missing value for this contract; do
        # not turn it into the string ``"None"`` while resolving paths.
        return default if value is None and default is not None else value

    def _science_config(self, config: Any) -> Any:
        algorithm = self._config_value(config, "algorithm", None)
        return self._config_value(self._config_value(algorithm, "polyseg", None), "science", None)

    def _initialize_science_dual_state(self, config: Any) -> None:
        """Create the persistent dual state for an enabled constraint run.

        A constraint update has no meaningful dimension without the
        pre-registered endpoint schema.  Therefore a constraint science run
        with no schema path fails during manager construction rather than
        silently falling back to a per-batch penalty.
        """

        science = self._science_config(config)
        algorithm = self._config_value(config, "algorithm", None)
        adv_estimator = self._config_value(algorithm, "adv_estimator", None)
        enabled = self._config_value(science, "enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("PolySeg-Sci enabled flag must be boolean")
        mode = _science_objective_mode(config) or "preference"
        if not ((enabled or adv_estimator == "polyseg_sci") and mode == "constraint"):
            return

        schema_path = self._config_value(science, "endpoint_schema_path", "")
        if not schema_path:
            raise ValueError("PolySeg-Sci constraint mode requires endpoint_schema_path")
        schema = load_endpoint_schema(schema_path)
        configured_hash = self._config_value(science, "endpoint_schema_hash", "")
        if configured_hash and str(configured_hash) != schema.fingerprint:
            raise ValueError("PolySeg-Sci endpoint schema hash does not match endpoint_schema_path")
        if schema.objective_mode != str(mode):
            raise ValueError("PolySeg-Sci objective_mode does not match the endpoint schema")

        configured_primary = self._config_value(science, "primary_objective", None)
        if configured_primary is None:
            primary = schema.primary_index()
        elif isinstance(configured_primary, str):
            primary = schema.endpoint_index(configured_primary)
        else:
            if isinstance(configured_primary, bool) or not isinstance(configured_primary, int):
                raise ValueError("PolySeg-Sci primary_objective must be an integer or endpoint id")
            primary = configured_primary
            primary = schema.primary_index(primary)
        learning_rate = self._config_value(science, "dual_learning_rate", 0.01)
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("PolySeg-Sci dual_learning_rate must be positive and finite")
        initial = self._config_value(science, "initial_multipliers", None)
        self.science_endpoint_schema = schema
        self.science_dual_state = ConstraintDualState.initialize(
            schema,
            primary_objective=primary,
            learning_rate=learning_rate,
            initial_multipliers=initial,
        )

    def set_science_dual_state(self, payload: Any) -> ConstraintDualState:
        """Restore a schema-bound dual state from a checkpoint payload."""

        if isinstance(payload, ConstraintDualState):
            payload = payload.state_dict()
        if not isinstance(payload, Mapping):
            raise ValueError("science dual checkpoint must be a state dictionary")
        payload = dict(payload)
        if self.science_endpoint_schema is None:
            # This also validates that the current run is explicitly configured
            # for constraint science before accepting checkpoint state.
            self._initialize_science_dual_state(self.config)
        if self.science_endpoint_schema is None:
            raise RuntimeError("science dual state cannot be restored without an endpoint schema")
        self.science_dual_state = ConstraintDualState.from_state_dict(self.science_endpoint_schema, payload)
        return self.science_dual_state

    def update_science_dual_state(self, estimates: Any) -> torch.Tensor:
        """Apply projected dual ascent to an audited HF estimate vector."""

        if self.science_dual_state is None or self.science_endpoint_schema is None:
            raise RuntimeError("PolySeg-Sci dual state is not enabled")
        return self.science_dual_state.update(self.science_endpoint_schema, estimates)

    def _attach_science_objective_state(self, prompts: Any) -> None:
        """Attach the current driver-side constraint vector to each prompt."""

        if getattr(self, "science_dual_state", None) is None:
            return
        values = self.science_dual_state.multipliers.detach().cpu().to(torch.float32).tolist()
        n = len(prompts)
        if hasattr(prompts, "non_tensor_batch"):
            prompts.non_tensor_batch["science_constraint_multipliers"] = np.asarray(
                [values] * n, dtype=np.float32
            )
        else:
            from verl.utils import tensordict_utils as tu

            tu.assign_non_tensor_stack(prompts, "science_constraint_multipliers", [values] * n)

    def _prepare_polyseg_adaptive(self, prompts, *, validate: bool = False):
        """Choose and attach one adaptive pair to a rollout batch.

        Both the DataProto and TransferQueue managers use this hook.  The
        latter passes a TensorDict, so scalar metadata is wrapped with the
        TensorDict non-tensor helpers rather than assigned as a NumPy array.
        """
        self._attach_science_objective_state(prompts)
        if self.polyseg_controller is None:
            return None
        if validate:
            # Freeze segmentation to the last training choice. Evaluation
            # batch size and labels must not alter selector state or budget.
            from types import SimpleNamespace

            previous = self.polyseg_controller.last_choice
            choice = SimpleNamespace(
                q=previous.q if previous is not None else min(self.polyseg_controller.q_candidates),
                m=previous.m if previous is not None else min(self.polyseg_controller.m_candidates),
                compute_cost=previous.compute_cost if previous is not None else 0.0,
                fallback="none", fallback_reason="",
            )
        else:
            choice = self.polyseg_controller.choose(
                episodes=max(1, len(prompts)),
                horizon=max(1, int(getattr(self.rollout_config, "response_length", 1))),
                feature_dim=max(
                    1, int(resolve_polyseg_option(self.config, "feature_dim", default=4))
                ),
                feedback=str(resolve_polyseg_option(self.config, "feedback", default="sum")),
            )
        episodes = max(1, len(prompts))
        values = {
            "polyseg_adaptive_q": choice.q,
            "polyseg_adaptive_m": choice.m,
            "polyseg_adaptive_planned_cost": (
                0.0
                if choice.fallback == "skip"
                else choice.m * self.polyseg_controller.segment_cost
            ),
            # Selector choices report batch totals; transport the per-
            # trajectory amount so reward-worker accounting can sum rows
            # without multiplying the same batch cost twice.
            "polyseg_adaptive_compute_cost": float(choice.compute_cost) / episodes,
            # This is the pre-observation ledger value. The trainer replaces
            # it with the post-observation value once reward cost is known.
            "polyseg_adaptive_remaining_budget": self.polyseg_controller.remaining_budget,
            "polyseg_adaptive_fallback": choice.fallback,
            "polyseg_adaptive_fallback_reason": choice.fallback_reason,
        }
        if hasattr(prompts, "non_tensor_batch"):
            n = len(prompts)
            prompts.non_tensor_batch.update(
                {
                    "polyseg_adaptive_q": np.full(n, choice.q, dtype=np.int64),
                    "polyseg_adaptive_m": np.full(n, choice.m, dtype=np.int64),
                    "polyseg_adaptive_planned_cost": np.full(
                        n, values["polyseg_adaptive_planned_cost"], dtype=np.float32
                    ),
                    "polyseg_adaptive_compute_cost": np.full(
                        n, values["polyseg_adaptive_compute_cost"], dtype=np.float32
                    ),
                    "polyseg_adaptive_remaining_budget": np.full(
                        n,
                        np.nan if values["polyseg_adaptive_remaining_budget"] is None else values["polyseg_adaptive_remaining_budget"],
                        dtype=np.float32,
                    ),
                    "polyseg_adaptive_fallback": np.full(n, choice.fallback, dtype=object),
                    "polyseg_adaptive_fallback_reason": np.full(n, choice.fallback_reason, dtype=object),
                }
            )
        else:
            from verl.utils import tensordict_utils as tu

            for key, value in values.items():
                tu.assign_non_tensor_stack(prompts, key, [value] * len(prompts))
        # For validation this is a snapshot used only for segmentation;
        # generate_sequences must not observe its held-out query costs.
        return choice

    def observe_polyseg_cost(self, actual_cost: float, *, compute_cost: float = 0.0, episodes: int = 1):
        """Update the shared adaptive ledger from measured verifier cost."""
        if self.polyseg_controller is None:
            return None
        return self.polyseg_controller.observe(float(actual_cost), compute_cost=float(compute_cost), episodes=episodes)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    @SkipManager.annotate(role="rollout")
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        # Attach per-sample priority to the batch (like ``uid``) so each sample gets
        # a globally-unique priority that flows to vLLM request scheduling. Assigned
        # before chunking so chunks own disjoint ranges without per-worker offsets.
        if "priority" not in prompts.non_tensor_batch:
            prompts.non_tensor_batch["priority"] = np.arange(len(prompts), dtype=np.int64)

        validate = prompts.meta_info.get("validate", False)
        if not isinstance(validate, bool):
            raise ValueError("rollout validate flag must be boolean")
        adaptive_choice = self._prepare_polyseg_adaptive(prompts, validate=validate)

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        output.meta_info["validate"] = validate
        if adaptive_choice is not None and not validate and self.reward_loop_worker_handles is not None:
            # Streaming reward workers have already completed verifier scoring
            # before the rollout output reaches the manager.  Legacy colocated
            # reward is accounted by the trainer after its reward call instead.
            actual = output.non_tensor_batch.get("segment_query_cost")
            actual_total = 0.0
            if actual is not None:
                try:
                    values = np.asarray(actual, dtype=np.float64).reshape(-1)
                    actual_total = float(values[np.isfinite(values)].sum())
                except (TypeError, ValueError):
                    actual_total = 0.0
            remaining = self.observe_polyseg_cost(actual_total, episodes=len(output))
            if remaining is not None:
                output.non_tensor_batch["polyseg_adaptive_remaining_budget"] = np.full(
                    len(output), remaining, dtype=np.float32
                )
            output.meta_info["polyseg_adaptive_choice"] = {
                "q": adaptive_choice.q,
                "m": adaptive_choice.m,
                "query_cost": adaptive_choice.query_cost,
                "planned_batch_cost": adaptive_choice.query_cost,
                "actual_batch_cost": actual_total,
                "remaining_budget": remaining,
            }
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
