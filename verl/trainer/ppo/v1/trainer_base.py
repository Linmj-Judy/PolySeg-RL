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

import json
import logging
import math
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.checkpoint_callback import build_checkpoint_callback
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    RolloutMoELoadBalanceMetricsAccumulator,
    compute_data_metrics,
    compute_moe_lb_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    get_metric_data_with_optional_routed_experts,
    process_validation_metrics,
)
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_spec_decode_metrics
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import (
    Role,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    need_teacher_policy,
)
from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY, ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.utils import MetricsAggregator, compute_advantage_for_multi_trajectories
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.import_utils import load_extern_type
from verl.utils.metric import reduce_metrics
from verl.utils.polyseg_journal import (
    configured_feedback_journal_path,
    configured_feedback_ledger_path,
    restore_feedback_ledger_snapshot,
    restore_feedback_journal_snapshot,
    snapshot_feedback_ledger,
    snapshot_feedback_journal,
    verify_feedback_journal_snapshot,
    verify_feedback_ledger_snapshot,
)
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.skip import SkipManager
from verl.utils.tracking import DapoFilteredRewardTableLogger, Tracking, ValidationGenerationsLogger
from verl.workers.config import CriticConfig, DistillationConfig, HFModelConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import response_from_nested, response_to_nested


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params.pop("topk_log_probs", None)
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _count_tq_prompt_groups(partition_id: str = "train") -> int:
    """Number of prompt groups currently registered in a TransferQueue partition."""
    data = tq.kv_list(partition_id)
    if not data:
        return 0
    return sum(1 for tag in data.get(partition_id, {}).values() if tag.get("is_prompt", False))


class PPOTrainer(ABC):
    """Base class for PPO trainer.

    Args:
        config: DictConfig from yaml config file.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.checkpoint_callback = build_checkpoint_callback(config)
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.trainer_mode = self.config.trainer.v1.trainer_mode
        self.parameter_sync_step = self.config.trainer.v1.get(self.trainer_mode, {}).get("parameter_sync_step", 1)
        self.replay_buffer = self._build_replay_buffer()
        self._rollout_moe_lb_metrics_accumulator = RolloutMoELoadBalanceMetricsAccumulator(
            model_config=self.config.actor_rollout_ref.model
        )
        # track mini-batch index within a parameter_sync_step cycle for Decoupled PPO
        self.local_trigger_step = 0
        self._restored_tq_prompt_count = 0

    def _build_replay_buffer(self) -> ReplayBuffer:
        """Instantiate the replay buffer (or a user-provided custom sampler).

        Set ``trainer.v1.sampler.custom_sampler.{path,name}`` to plug in a custom
        ``ReplayBuffer`` subclass; otherwise the built-in implementation is used.
        """
        sampler_config = self.config.trainer.v1.sampler
        custom_sampler = sampler_config.get("custom_sampler", None)
        has_custom_sampler = bool(
            custom_sampler is not None and custom_sampler.get("path") and custom_sampler.get("name")
        )
        if has_custom_sampler:
            sampler_cls = load_extern_type(custom_sampler.path, custom_sampler.name)
        else:
            sampler_cls = ReplayBuffer if self.trainer_mode == "sync" else ReplayBufferAsync

        replay_buffer_kwargs = dict(
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
        )
        # Preserve the existing constructor contract for external samplers; custom implementations own
        # their filtering semantics and can consume algorithm.filter_groups through their own config.
        if not has_custom_sampler:
            filter_groups_metric = self._resolve_filter_groups_metric()
            sync_refill_failed_groups = bool(sampler_config.get("sync_refill_failed_groups", False))
            replay_buffer_kwargs.update(
                filter_groups_metric=filter_groups_metric,
                sync_refill_failed_groups=sync_refill_failed_groups,
            )
            if sampler_cls is ReplayBuffer:
                filter_groups = self.config.algorithm.get("filter_groups", None)
                max_inflight_gen_batches = 1
                if filter_groups_metric is not None:
                    max_inflight_gen_batches = filter_groups.get("max_inflight_gen_batches", 1)
                train_batch_size = self.config.data.train_batch_size
                replay_buffer_kwargs.update(
                    train_batch_size=train_batch_size,
                    gen_batch_size=1
                    if filter_groups_metric is not None or sync_refill_failed_groups
                    else (self.config.data.get("gen_batch_size", None) or train_batch_size),
                    max_inflight_gen_batches=max_inflight_gen_batches,
                )
        return sampler_cls(**replay_buffer_kwargs)

    def _resolve_filter_groups_metric(self) -> str | None:
        """Resolve DAPO's group metric and verify that rollout computes it before sampling."""
        filter_groups = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        if not filter_enabled:
            return None

        filter_metric = filter_groups.get("metric", None)
        if not filter_metric:
            raise ValueError("algorithm.filter_groups.metric must be set when group filtering is enabled")

        reward_model = self.config.reward.reward_model
        streaming_reward_path = not reward_model.enable or reward_model.enable_resource_pool
        assert streaming_reward_path, (
            "algorithm.filter_groups requires the reward metric at sampling time: use rule-based reward or "
            "reward.reward_model.enable_resource_pool=True. A colocated reward model computes rewards only "
            "after replay-buffer sampling."
        )
        max_num_gen_batches = filter_groups.get("max_num_gen_batches", 0)
        if max_num_gen_batches > 0:
            logger.warning(
                "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
                "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
                max_num_gen_batches,
            )
        return str(filter_metric)

    def init(self):
        """Initialize all components of the trainer.

        1. WorkerGroup: actor, critic, reference with model engine: FSDP/Megatron/VeOmni/...
        2. LLMServerManager: launch and manage LLM server replicas for generation.
        3. CheckpointEngineManager: sync weights between worker group and LLM server replicas.
        4. RewardLoopManager: reward workers for rule-based reward, optional LLM server for model-based reward.
        5. [Optional] MultiTeacherModelManager: LLM teacher servers for on-policy distillation.
        """
        self._setup()
        self.on_init_end()

    def _setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # 1. define actor and rollout class
        if Role.Actor in self.role_worker_mapping:
            actor_role = Role.Actor
        elif Role.ActorRolloutRef in self.role_worker_mapping:
            actor_role = Role.ActorRolloutRef
        else:
            actor_role = Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        if actor_role == Role.Actor and self.use_reference_policy and not self.ref_in_actor:
            ref_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[ref_resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu

            # Wire the critic profiler config via the hydra path (real dataclass tool_config), so the
            # standalone critic TrainingWorker gets a working DistProfiler instead of a silent no-op.
            critic_omega_profiler_config = self.config.critic.get("profiler", {})
            critic_profiler_config = (
                omega_conf_to_dataclass(critic_omega_profiler_config) if critic_omega_profiler_config else None
            )

            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
                profiler_config=critic_profiler_config,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initialize critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        logger.info("actor and ref model engine initialized")

        if self.use_reference_policy and not self.ref_in_actor:
            if actor_role == Role.Actor:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                self.ref_policy_wg = all_wg[str(actor_role)]
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize teacher loop manager
        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # 9. initialize agent loop manager
        # Trainers that opt out of colocated rollout replicas on the training
        # GPUs (v1 separate_async with actor_rollout_ref.hybrid_engine=False)
        # set ``self._enable_hybrid_replicas = False`` before super()._setup();
        # they get an empty manager here and serve rollout from their
        # standalone replicas only.
        if getattr(self, "_enable_hybrid_replicas", True):
            self.llm_server_manager: LLMServerManager = LLMServerManager.create(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rollout_resource_pool=actor_rollout_resource_pool,
            )
        else:
            self.llm_server_manager = LLMServerManager.create_empty(config=self.config)

        # 10. initialize checkpoint engine manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        checkpoint_engine_config.backend = "naive"
        self.checkpoint_manager: CheckpointEngineManager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()
        self._load_checkpoint()

        logger.info("all initialize finished, ready to fit")

    def get_llm_client(self) -> LLMServerClient:
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client()

    def get_teacher_client(self) -> Optional[dict[str, LLMServerClient]]:
        """Get the On-Policy Distillation teacher server clients.

        Returns:
            dict[str, LLMServerClient]: The teacher server clients.
        """
        return self.teacher_model_manager.get_client() if self.use_teacher_policy else None

    def get_reward_handles(self) -> list[ray.actor.ActorHandle]:
        """Get the handles of reward loop workers."""
        return self.reward_loop_manager.reward_loop_worker_handles

    def fit(self, agent_loop_manager: AgentLoopManager):
        """Fit the trainer with the agent loop manager.

        Args:
            agent_loop_manager: The agent loop manager to generate sequences.
        """
        self.agent_loop_manager = agent_loop_manager
        self._restore_polyseg_controller_state()

        # initialize SkipManager for V1 rollout skip support
        SkipManager.init(self.config)

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            self.on_validate_begin()
            val_metrics = self._validate()
            self.on_validate_end()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // self.steps_per_epoch
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        SkipManager.set_step(self.global_steps)
        self._reissue_inflight_prompts()
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        self.on_train_begin()
        last_val_metrics = None
        # ``total_training_steps`` is the authoritative stopping condition.
        # When a full epoch is not divisible by the batch size, the previous
        # ``current_epoch < total_epochs`` guard exited immediately after the
        # penultimate update: with N=935, step 934 incremented the counter to
        # 935 and made ``current_epoch == total_epochs`` before the loop could
        # execute the configured final-step save/validation hooks.  This left
        # successful runs without their required checkpoint or held-out
        # artifacts.  The explicit step budget already encodes the requested
        # epoch count (and is also used for resumed runs), so use it directly.
        while self.global_steps <= self.total_training_steps:
            is_last_step = self.global_steps >= self.total_training_steps
            metrics = {}
            self.timing_raw = {}

            # 1. perform rollout and actor/critic training
            with marked_timer("step", self.timing_raw):
                self.on_step_begin()

                self._start_profiling()
                batch = self.step(metrics, self.timing_raw)
                self._stop_profiling()

                # 2. save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", self.timing_raw, color="green"):
                        self._save_checkpoint()

                self.on_step_end()
                metrics.update(self._consume_sync_metrics())

            # 4. validate
            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("testing", self.timing_raw, color="green"):
                    self.on_validate_begin()
                    val_metrics: dict = self._validate()
                    self.on_validate_end()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            # 5. record metrics
            self._compute_metrics(batch, metrics, self.timing_raw, global_steps=self.global_steps, epoch=current_epoch)

            # 6. dump rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, self.timing_raw, rollout_data_dir)

            # 7. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

            dapo_filtered_reward_counts = metrics.pop(DAPO_FILTERED_REWARD_COUNTS_KEY, None)
            self.logger.log(data=metrics, step=self.global_steps)
            if dapo_filtered_reward_counts:
                self.dapo_filtered_reward_logger.log(
                    self.config.trainer.logger, dapo_filtered_reward_counts, self.global_steps
                )
            progress_bar.update(1)
            self.global_steps += 1
            SkipManager.set_step(self.global_steps)
            current_epoch = (self.global_steps - 1) // self.steps_per_epoch
            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                break

        self.on_train_end()
        self._shutdown_dump_executor()
        self._shutdown_train_dataloader()

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        train_batch_size = self.config.data.train_batch_size
        assert train_batch_size % self.parameter_sync_step == 0, (
            f"train_batch_size ({train_batch_size}) must be divisible by "
            f"parameter_sync_step ({self.parameter_sync_step})"
        )
        sample_batch_size = train_batch_size // self.parameter_sync_step

        prepare_metrics = self.prepare_step()

        metrics_aggregator = MetricsAggregator()
        if prepare_metrics:
            metrics_aggregator.add_step_metrics(prepare_metrics)
        combined_keys: list = []
        combined_tags: list = []
        combined_partition_id = "train"
        for trigger_idx in range(self.parameter_sync_step):
            self.local_trigger_step = trigger_idx
            iter_metrics: dict = {}
            batch = self._step_once(iter_metrics, timing_raw, sample_batch_size)
            sample_count = sum(not tag.get("is_padding", False) for tag in batch.tags)
            metrics_aggregator.add_step_metrics(iter_metrics, sample_count=sample_count)
            combined_keys.extend(batch.keys)
            combined_tags.extend(batch.tags)
            combined_partition_id = batch.partition_id

        metrics.update(metrics_aggregator.get_aggregated_metrics())
        return KVBatchMeta(partition_id=combined_partition_id, keys=combined_keys, tags=combined_tags)

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        """Run a single local update: sample one mini-batch and perform the full PPO pipeline once."""
        # 1. sample batch from replay buffer
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()

        # 2. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        # 3. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 4. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 5. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 7. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 8. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 9. update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch

    # ------------------------------ abstract methods ------------------------------

    def on_init_end(self):
        """Called after the initialization ends."""
        return

    def on_train_begin(self):
        """Called before the training loop starts."""
        return

    def _add_async_warmup_batches(self, num_warmup_batches: int | float) -> None:
        """Fill the async prefetch window without duplicating checkpointed prompt groups.

        Fractional values are supported: e.g. 1.5 with train_batch_size=64 adds
        one full batch (64 prompts) plus half a batch (32 prompts); the fractional
        part is rounded down to a whole number of gen_batch_size chunks (prompts
        are fetched per gen_batch_size).
        """
        if self.config.skip.rollout_tq.enable or num_warmup_batches <= 0:
            return

        restored_prompts = self._restored_tq_prompt_count
        gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
        target_chunks = math.floor(num_warmup_batches * self.config.data.train_batch_size / gen_batch_size)
        target_prompts = target_chunks * gen_batch_size
        missing_prompts = max(0, target_prompts - restored_prompts)
        if missing_prompts == 0:
            logger.info(
                f"Skipping async warmup: {restored_prompts} restored prompt groups already fill the "
                f"{target_prompts}-prompt prefetch window"
            )
            return

        self._add_prompts_to_generate(missing_prompts)
        logger.info(
            f"Added {missing_prompts} warmup prompts after restoring {restored_prompts} of "
            f"{target_prompts} target prompt groups"
        )

    def on_train_end(self):
        """Called after the training loop ends."""
        return

    def on_validate_begin(self):
        """Called before the validation loop starts."""
        return

    def on_validate_end(self):
        """Called after the validation loop ends."""
        return

    def on_step_begin(self):
        """Called at the beginning of each training step."""
        return

    @abstractmethod
    def on_step_end(self):
        """Called at the end of each training step."""
        return

    def _consume_sync_metrics(self) -> dict:
        """Weight-sync stats stashed by ``on_step_end`` (e.g. the delta engines'
        changed ratio / wire payload), merged into this step's logged metrics."""
        metrics = getattr(self, "_pending_sync_metrics", None) or {}
        self._pending_sync_metrics = {}
        return metrics

    def on_sample_begin(self):
        """Called at the beginning of sampling batch from replay buffer."""
        return

    @abstractmethod
    def on_sample_end(self):
        """Called after sampling a batch from replay buffer."""
        return

    # ------------------------------ common methods ------------------------------

    def _get_n_gpus_for_throughput(self) -> int:
        """Return the total number of GPUs used for throughput normalization.

        By default this is the trainer-side GPU count from the resource pool
        manager.  Modes that use additional dedicated GPUs (e.g. separate-async
        standalone rollout) should override this to include them.
        """
        return self.resource_pool_manager.get_n_gpus()

    def _init_tokenizer(self):
        """Initialize tokenizer and processor from the model config."""
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model)
        self.tokenizer = model_config.tokenizer
        # Used for multimodal LLM, could be None
        self.processor = model_config.processor

    def _init_dataloader(self):
        """Initialize train and validate dataloader."""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        # Exact refill counts require single-prompt dataloader fetches.
        filter_groups = self.config.algorithm.get("filter_groups", None)
        dapo_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        sync_refill_failed_groups = bool(self.config.trainer.v1.sampler.get("sync_refill_failed_groups", False))
        requires_exact_refill = self.trainer_mode != "sync" or dapo_enabled or sync_refill_failed_groups
        if requires_exact_refill:
            user_gen_batch_size = self.config.data.get("gen_batch_size", None)
            if user_gen_batch_size not in (None, 1):
                logger.warning(f"data.gen_batch_size={user_gen_batch_size} is overridden to 1.")
            elif user_gen_batch_size is None:
                logger.info("data.gen_batch_size defaulted to 1.")
            with open_dict(self.config):
                self.config.data.gen_batch_size = 1

        # use gen_batch_size as the batch size for the dataloader if set, otherwise use train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=gen_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            # Keep the final partial batch so every row in a full-data
            # contract is observed.  ``_fetch_one_gen_batch`` pads that one
            # batch to the configured size for PPO's divisibility invariants.
            drop_last=False,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.train_dataloader_it = None
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            f"train and validate dataloader initialized, train dataset size: "
            f"{len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
        )

        self.steps_per_epoch = math.ceil(len(self.train_dataset) / self.config.data.train_batch_size)

        # adjust total_training_steps
        total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        # The LR scheduler steps once per local update, and each global step performs
        # ``parameter_sync_step`` local updates (see ``PPOTrainer.step``). The optimizer's
        # schedule horizon must therefore count optimizer updates.
        optim_total_training_steps = total_training_steps * self.parameter_sync_step
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = optim_total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = optim_total_training_steps
        except Exception as e:
            logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _init_resource_pool_mgr(self):
        config = self.config
        # role => worker class
        self.role_worker_mapping = {}
        # role => resource pool
        self.mapping = {}

        # Add actor rollout worker to mapping
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        if not getattr(self, "_enable_hybrid_replicas", True):
            role = Role.Actor
        else:
            role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"
        if role == Role.Actor and need_reference_policy(config) and not ref_in_actor:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            self.mapping[Role.RefPolicy] = "global_pool"

        # Add critic worker to mapping.
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

        # Global resource pool is used for actor, rollout, critic, ref
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Add separate resource pool for reward model if enabled
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def _load_checkpoint(self):
        self.global_steps = 0
        self._pending_polyseg_controller_state = None
        self._pending_science_dual_state = None

        # A trainer instance owns the process-local fallback registry. Clear
        # it before either starting fresh or loading a checkpoint; otherwise a
        # second trainer created in the same Python process could inherit old
        # sufficient statistics accidentally.
        # ``algorithm`` is optional in the minimal trainer configurations used by
        # callback/checkpoint integrations (and by older non-PolySeg jobs).  Use
        # mapping-style access so saving a regular checkpoint does not depend on
        # a PolySeg-specific config node being present.
        polyseg_config = self.config.get("algorithm", {}).get("polyseg", {})
        if polyseg_config.get("accumulate_estimator", False) or polyseg_config.get("use_confidence_bonus", False):
            from verl.trainer.ppo.polyseg import reset_polyseg_estimator_state, reset_polyseg_dependence_state

            if polyseg_config.get("accumulate_estimator", False):
                reset_polyseg_estimator_state()
            if polyseg_config.get("use_confidence_bonus", False):
                reset_polyseg_dependence_state()

        # 1. find latest checkpoint folder
        if self.config.trainer.resume_mode == "disable":
            return
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

        controller_state_path = os.path.join(global_step_folder, "polyseg_controller.pt")
        if os.path.exists(controller_state_path):
            self._pending_polyseg_controller_state = torch.load(controller_state_path, weights_only=False)
            logger.info(f"Loaded pending PolySeg controller state from {controller_state_path}")
        science_dual_state_path = os.path.join(global_step_folder, "science_dual_state.pt")
        if os.path.exists(science_dual_state_path):
            self._pending_science_dual_state = torch.load(science_dual_state_path, weights_only=False)
            logger.info(f"Loaded pending PolySeg-Sci dual state from {science_dual_state_path}")

        # PolySeg's accumulating ridge estimator is trainer state in addition
        # to actor/critic weights. Restore it before processing the first
        # resumed batch so effective sample counts remain reproducible.
        # ``algorithm`` is optional in minimal trainer configurations used by
        # checkpoint/callback integrations and older non-PolySeg jobs.
        polyseg_config = self.config.get("algorithm", {}).get("polyseg", {})
        if polyseg_config.get("accumulate_estimator", False):
            polyseg_state_path = os.path.join(global_step_folder, "polyseg_estimator.pt")
            if os.path.exists(polyseg_state_path):
                from verl.trainer.ppo.polyseg import load_polyseg_estimator_state_dict

                load_polyseg_estimator_state_dict(torch.load(polyseg_state_path, weights_only=False), clear=True)
                logger.info(f"Loaded PolySeg estimator state from {polyseg_state_path}")
            else:
                logger.warning(f"No PolySeg estimator state found at {polyseg_state_path}; starting estimator from scratch")

        if polyseg_config.get("use_confidence_bonus", False):
            polyseg_dependence_path = os.path.join(global_step_folder, "polyseg_dependence.pt")
            if os.path.exists(polyseg_dependence_path):
                from verl.trainer.ppo.polyseg import load_polyseg_dependence_state_dict

                load_polyseg_dependence_state_dict(
                    torch.load(polyseg_dependence_path, weights_only=False), clear=True
                )
                logger.info(f"Loaded PolySeg dependence/confidence state from {polyseg_dependence_path}")
            else:
                logger.warning(
                    f"No PolySeg dependence state found at {polyseg_dependence_path}; "
                    "starting confidence radius from its prior"
                )

        ledger_path = configured_feedback_ledger_path(self.config)
        journal_path = configured_feedback_journal_path(self.config)
        journal_snapshot_path = os.path.join(global_step_folder, "polyseg_feedback_journal.json")
        if journal_path is not None:
            # A checkpoint without a journal cursor cannot be replayed safely:
            # an interrupted process may have appended complete trajectories
            # after this checkpoint and those rows would otherwise be charged
            # twice when the dataloader resumes from the saved step.
            if not os.path.isfile(journal_snapshot_path):
                raise RuntimeError(
                    "checkpoint is missing polyseg_feedback_journal.json while a shared feedback journal is configured"
                )
            verify_feedback_journal_snapshot(journal_path, journal_snapshot_path)
        ledger_snapshot_path = os.path.join(global_step_folder, "polyseg_feedback_ledger.json")
        if ledger_path is not None:
            if not os.path.exists(ledger_snapshot_path):
                raise RuntimeError(
                    "checkpoint is missing polyseg_feedback_ledger.json while a shared feedback ledger is configured"
                )

        if journal_path is not None:
            restored = restore_feedback_journal_snapshot(journal_path, journal_snapshot_path)
            if restored["truncated_bytes"]:
                logger.warning(
                    "Removed %s post-checkpoint feedback-journal bytes before resume",
                    restored["truncated_bytes"],
                )

        # A process may have settled feedback after the last checkpoint but
        # before it was interrupted.  The journal cursor above removes those
        # rows; restore the matching ledger snapshot before replaying the
        # dataloader so the same feedback can be charged exactly once.
        if ledger_path is not None:
            restore_feedback_ledger_snapshot(ledger_path, ledger_snapshot_path)
            verify_feedback_ledger_snapshot(ledger_path, ledger_snapshot_path)
            logger.info("Restored and verified PolySeg feedback-ledger snapshot before resume")

        # 2. load actor checkpoint
        self.actor_rollout_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 3. load critic checkpoint
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 4. load dataloader checkpoint
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # 5. restore TransferQueue state (async modes). Re-issuing the restored in-flight prompts is
        # deferred to fit() to use the agent_loop_manager.
        if self.trainer_mode != "sync":
            tq_ckpt_path = os.path.join(global_step_folder, "transfer_queue")
            if os.path.exists(tq_ckpt_path):
                logger.info(f"Loading TransferQueue state from {tq_ckpt_path}")
                tq.load_checkpoint(tq_ckpt_path)
                self._restored_tq_prompt_count = _count_tq_prompt_groups()
                logger.info(f"Restored {self._restored_tq_prompt_count} training prompt groups from TransferQueue")

    def _restore_polyseg_controller_state(self) -> None:
        """Restore driver-side adaptive selector state after manager creation."""
        state = getattr(self, "_pending_polyseg_controller_state", None)
        manager = getattr(self, "agent_loop_manager", None)
        if state is not None:
            controller = getattr(manager, "polyseg_controller", None)
            if controller is None:
                raise RuntimeError("checkpoint contains PolySeg controller state but adaptive manager is disabled")
            from verl.trainer.ppo.polyseg import BudgetedPolySegController

            manager.polyseg_controller = BudgetedPolySegController.from_state_dict(state)
            self._pending_polyseg_controller_state = None
            logger.info("Restored PolySeg adaptive selector state")
        science_state = getattr(self, "_pending_science_dual_state", None)
        if science_state is not None:
            setter = getattr(manager, "set_science_dual_state", None)
            if setter is None:
                raise RuntimeError("checkpoint contains science dual state but manager cannot restore it")
            setter(science_state)
            self._pending_science_dual_state = None

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        """Restart checkpointed pending/running prompt groups from their persisted prompt data."""
        if self.trainer_mode == "sync":
            return 0
        data = tq.kv_list(partition_id)
        if not data:
            return 0
        items = data.get(partition_id, {})
        inflight_uids = [
            key
            for key, tag in items.items()
            if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
        ]
        if not inflight_uids:
            return 0

        batch = tq.kv_batch_get(keys=inflight_uids, partition_id=partition_id)
        inflight_uid_set = set(inflight_uids)
        old_trajectory_keys = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.split("_", 1)[0] in inflight_uid_set
        ]
        if old_trajectory_keys:
            tq.kv_clear(keys=old_trajectory_keys, partition_id=partition_id)

        # Treat this as a new dispatch attempt for the resumed training step.
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in inflight_uids]
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)

        logger.info(
            f"Re-issued {len(inflight_uids)} in-flight prompts for step {self.global_steps}; "
            f"cleared {len(old_trajectory_keys)} old trajectories from partition {partition_id}"
        )
        return len(inflight_uids)

    def _save_checkpoint(self):
        """Save actor, critic, and dataloader checkpoints to local (and optionally remote) storage."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        logger.info(f"Saving checkpoint to {local_global_step_folder}")

        # resolve max checkpoints to keep
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # save actor
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save critic
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        # ``algorithm`` is optional in minimal trainer configurations used by
        # checkpoint/callback integrations and older non-PolySeg jobs.
        polyseg_config = self.config.get("algorithm", {}).get("polyseg", {})
        if polyseg_config.get("accumulate_estimator", False):
            from verl.trainer.ppo.polyseg import polyseg_estimator_state_dict

            polyseg_state_path = os.path.join(local_global_step_folder, "polyseg_estimator.pt")
            torch.save(polyseg_estimator_state_dict(), polyseg_state_path)
            logger.info(f"Saved PolySeg estimator state to {polyseg_state_path}")

        if polyseg_config.get("use_confidence_bonus", False):
            from verl.trainer.ppo.polyseg import polyseg_dependence_state_dict

            polyseg_dependence_path = os.path.join(local_global_step_folder, "polyseg_dependence.pt")
            torch.save(polyseg_dependence_state_dict(), polyseg_dependence_path)
            logger.info(f"Saved PolySeg dependence/confidence state to {polyseg_dependence_path}")

        controller = getattr(getattr(self, "agent_loop_manager", None), "polyseg_controller", None)
        if controller is not None:
            torch.save(controller.state_dict(), os.path.join(local_global_step_folder, "polyseg_controller.pt"))
            logger.info("Saved PolySeg adaptive selector state")
        science_dual_state = getattr(getattr(self, "agent_loop_manager", None), "science_dual_state", None)
        if science_dual_state is not None:
            state_dict = science_dual_state.state_dict() if hasattr(science_dual_state, "state_dict") else science_dual_state
            torch.save(state_dict, os.path.join(local_global_step_folder, "science_dual_state.pt"))
            logger.info("Saved PolySeg-Sci constraint dual state")

        # The selector state is checkpointed by torch; the shared reward ledger
        # is a separate process-safe JSON file.  Store an exact snapshot beside
        # the model so resume can fail closed instead of silently charging a
        # different budget state.
        ledger_path = configured_feedback_ledger_path(self.config)
        if ledger_path is not None:
            ledger_snapshot_path = os.path.join(local_global_step_folder, "polyseg_feedback_ledger.json")
            try:
                snapshot_feedback_ledger(ledger_path, ledger_snapshot_path)
                logger.info("Saved PolySeg feedback-ledger snapshot")
            except FileNotFoundError:
                logger.warning("Configured PolySeg feedback ledger is missing; checkpoint is not ledger-resumable")

        # Save a cursor for the append-only trajectory journal as part of the
        # same checkpoint directory.  On resume, rows written after this
        # checkpoint are truncated before the dataloader replays that batch;
        # otherwise a clean interruption between periodic checkpoints would
        # duplicate feedback cost and trajectory counts.
        journal_path = configured_feedback_journal_path(self.config)
        if journal_path is not None:
            journal_snapshot_path = os.path.join(local_global_step_folder, "polyseg_feedback_journal.json")
            try:
                snapshot_feedback_journal(journal_path, journal_snapshot_path)
                logger.info("Saved PolySeg feedback-journal checkpoint cursor")
            except FileNotFoundError as error:
                raise RuntimeError(
                    "configured PolySeg feedback journal is missing; refusing to create a non-resumable checkpoint"
                ) from error

        # save TransferQueue state for async modes so in-flight prompts (already fetched from the
        # dataloader but not yet trained into this checkpoint's weights) survive a restart:
        # finished trajectories are restored as-is, pending/running prompts are re-issued on resume.
        if self.trainer_mode != "sync":
            tq.save_checkpoint(
                os.path.join(local_global_step_folder, "transfer_queue"),
                metadata={"global_steps": self.global_steps},
            )

        # write latest checkpointed iteration tracker for atomic resume
        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
        if actor_ckpt_cfg.get("async_save", False):
            logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            self.checkpoint_callback.on_save(
                trainer=self,
                global_step=self.global_steps,
                checkpoint_dir=local_global_step_folder,
                async_save=True,
            )
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        self.checkpoint_callback.on_save(
            trainer=self, global_step=self.global_steps, checkpoint_dir=local_global_step_folder, async_save=False
        )

    def _validate(self) -> dict[str, float]:
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}
        # To avoid failure sessions
        expected_acc_counts: dict[tuple[str, str], int] = {}

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch_size = len(batch_dict["uid"])
            batch_data_sources = batch_dict.get("data_source", ["unknown"] * batch_size)
            batch_rollout_ns = batch_dict.get(
                "__rollout_n__", [self.config.actor_rollout_ref.rollout.val_kwargs.n] * batch_size
            )
            for uid, data_source, rollout_n in zip(
                batch_dict["uid"], batch_data_sources, batch_rollout_ns, strict=True
            ):
                rollout_n = int(rollout_n)
                expected_acc_counts[(str(data_source), str(uid))] = rollout_n

            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            # Register each prompt (GRPO group) in TransferQueue as a tag-only status marker.
            # global_steps is required by ReplayBuffer's metadata sync / staleness ordering.
            tags = [
                {"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))
            ]
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer: one prompt (GRPO group) per submitted row.
            batch, _ = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="val", batch_size=len(batch)
            )

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )

            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            sample_uids.extend(data.pop("uid").tolist())
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            sample_turns.extend(data.pop("num_turns").tolist())
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    reward_extra_info = (
                        extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    )
                    # Persist rollout-owned tool/event traces in validation
                    # JSONL.  Scalar reward extras alone cannot prove that a
                    # tool interaction occurred or that m respected complete
                    # event boundaries.
                    if isinstance(extra_field, dict) and "agentic_trace" in extra_field:
                        reward_extra_info = dict(reward_extra_info)
                        reward_extra_info["agentic_trace"] = extra_field["agentic_trace"]
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)

            # 5. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
                for key in dump_all_keys
                for parts in [key.rsplit("_", 2)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys},
                dump_path=val_data_dir,
            )

        return self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
            expected_acc_counts=expected_acc_counts,
        )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _shutdown_train_dataloader(self):
        """Stop the train dataloader's worker processes.

        StatefulDataLoader keeps its last iterator, so the workers would outlive fit(). At process exit Ray kills them
        with the actor, and the trainer then reports "DataLoader worker (pid ...) is killed by signal: Killed".
        """
        iterator, self.train_dataloader_it = self.train_dataloader_it, None
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout data from TransferQueue and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            fields = ["uid", "prompts", "responses", "rm_scores", "reward_model"]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

            uids = data.pop("uid").tolist()
            inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["prompts"]]
            outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["responses"]]
            scores = data["rm_scores"].sum(dim=1).tolist()

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
            else:
                gts = [None] * len(uids)

            # Sort by uid key ({sample}_{rollout}_{output})
            sort_keys = []
            for key in batch.keys:
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
                else:
                    sort_keys.append((key, 0, 0))
            sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

            inputs = [inputs[i] for i in sorted_indices]
            outputs = [outputs[i] for i in sorted_indices]
            gts = [gts[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            reward_extra_infos_dict = {"uid": [batch.keys[i] for i in sorted_indices]}

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _val_metrics_update(
        self,
        data_sources,
        sample_uids,
        reward_extra_infos_dict,
        sample_turns,
        expected_acc_counts: dict[tuple[str, str], int] | None = None,
    ) -> dict[str, float]:
        data_src2var2metric2val = process_validation_metrics(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            expected_acc_counts=expected_acc_counts,
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.array(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _rollout_server_managers(self) -> list:
        """LLM server managers whose inference engines take part in rollout profiling.

        Subclasses owning additional replicas (e.g. the standalone rollout of separate-async
        training) extend this list so those engines are profiled as well.
        """
        managers = [getattr(self, "llm_server_manager", None)]
        return [manager for manager in managers if manager is not None]

    def _start_rollout_profiling(self) -> None:
        """Start rollout profiling."""
        for manager in self._rollout_server_managers():
            manager.start_profile()

    def _stop_rollout_profiling(self) -> None:
        """Stop rollout profiling."""
        for manager in self._rollout_server_managers():
            manager.stop_profile()

    def _start_profiling(self) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )

        if do_profile:
            # "train", not "e2e": this window only holds what the training worker itself runs
            # (log-prob forwards and the actor update). Generation happens in the inference
            # engines below, which write their own traces.
            #
            # In the hybrid engine, actor/rollout and the (colocated) reference -- and sometimes the
            # critic -- share ONE worker group object, so ref_policy_wg / critic_wg can alias
            # actor_rollout_wg. Each start/stop_profile round-trips to every rank and, on stop, runs
            # the finish hook (e.g. the user's trace-upload command); driving the same physical
            # workers more than once would fire that hook once per alias and upload the same trace
            # file multiple times. Drive each distinct worker group exactly once.
            self.actor_rollout_wg.start_profile(role="train", profile_step=self.global_steps)
            seen = {id(self.actor_rollout_wg)}
            if self.use_reference_policy and id(self.ref_policy_wg) not in seen:
                seen.add(id(self.ref_policy_wg))
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic and id(self.critic_wg) not in seen:
                seen.add(id(self.critic_wg))
                self.critic_wg.start_profile(profile_step=self.global_steps)
            self._start_rollout_profiling()

    def _stop_profiling(self) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        this_step_profile = self.curr_step_profile
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile

        if do_profile:
            # Run the finish command (e.g. the trace upload) only once, on the last profiled step, so
            # a command that uploads the whole save_path sends each trace once instead of re-uploading
            # the accumulating directory every step. "Last" is the largest configured step, or the
            # run's final step if it ends earlier on a profiled step.
            profiled_steps = self.config.global_profiler.steps
            is_last_step = self.global_steps >= self.total_training_steps
            run_command = bool(
                this_step_profile and profiled_steps and (self.global_steps == max(profiled_steps) or is_last_step)
            )
            # See _start_profiling: skip aliased worker groups so the finish hook (and any trace
            # upload it triggers) fires exactly once per distinct process, not once per role alias.
            self.actor_rollout_wg.stop_profile(run_command=run_command)
            seen = {id(self.actor_rollout_wg)}
            if self.use_reference_policy and id(self.ref_policy_wg) not in seen:
                seen.add(id(self.ref_policy_wg))
                self.ref_policy_wg.stop_profile(run_command=run_command)
            if self.use_critic and id(self.critic_wg) not in seen:
                seen.add(id(self.critic_wg))
                self.critic_wg.stop_profile(run_command=run_command)

    def _fetch_one_gen_batch(self) -> TensorDict:
        """Fetch one ``gen_batch_size`` chunk from the dataloader."""
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        # StatefulDataLoader keeps the final partial batch when the full-data
        # cardinality is not divisible by the PPO batch size.  Downstream
        # actor/critic dispatch requires a fixed-size global batch, so repeat
        # rows from this same batch only for padding.  The original partial
        # rows are therefore all consumed exactly once; duplicated rows are
        # explicitly synthetic padding and receive fresh UIDs.
        gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
        observed = len(batch_dict["raw_prompt"])
        if observed < gen_batch_size:
            pad_indices = np.arange(gen_batch_size - observed, dtype=np.int64) % max(observed, 1)
            for key, value in list(batch_dict.items()):
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = torch.cat([value, value[torch.as_tensor(pad_indices, device=value.device)]], dim=0)
                elif isinstance(value, np.ndarray):
                    batch_dict[key] = np.concatenate([value, value[pad_indices]], axis=0)
                elif isinstance(value, list):
                    batch_dict[key] = value + [value[int(index)] for index in pad_indices]
            observed = gen_batch_size
        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(observed)], dtype=object)
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int | None = None) -> TensorDict:
        """Fetch and coalesce the requested number of prompts."""
        train_batch_size = self.config.data.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or train_batch_size
        if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
            raise ValueError(
                f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size "
                f"({gen_batch_size}); it is submitted in whole gen_batch_size dataloader fetches."
            )

        chunks = [self._fetch_one_gen_batch() for _ in range(num_prompts // gen_batch_size)]
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    def _submit_batch_to_rollout(self, batch: TensorDict) -> int:
        """Register prompts in TransferQueue and dispatch them for generation."""
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]
        if self.trainer_mode != "sync":
            tq.kv_batch_put(
                keys=list(batch["uid"]),
                partition_id="train",
                tags=tags,
                # Persist prompt data for async checkpoint recovery.
                # TODO: maybe let workers do it?
                fields=batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)]),
            )
        else:
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)

        self.agent_loop_manager.generate_sequences(batch)
        return len(batch)

    def _add_prompts_to_generate(self, num_prompts: int) -> int:
        """Add an exact number of prompts to the AgentLoopManager."""
        batch = self._next_train_batch(num_prompts)
        return self._submit_batch_to_rollout(batch)

    @SkipManager.annotate_tq(role="rollout_tq", phase="submit")
    def _add_batch_to_generate(self):
        """Add one training batch to the AgentLoopManager."""
        batch = self._next_train_batch()
        self._submit_batch_to_rollout(batch)

    def prepare_step(self) -> dict:
        self._add_batch_to_generate()
        return {}

    def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict | None = None) -> KVBatchMeta:
        """Compute the reward score with a colocated reward model."""
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"

        # 1. read the fields required by the reward model from TransferQueue.
        # Reward managers need rollout-owned segment ids and the action mask to
        # distinguish assistant tokens from tool observations.  Omitting these
        # fields silently turns agentic PolySeg back into fixed-token slicing.
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "raw_prompt",
            "extra_fields",
            "data_source",
            "reward_model",
            "extra_info",
            "polyseg_adaptive_q",
            "polyseg_adaptive_m",
            "polyseg_adaptive_planned_cost",
            "polyseg_adaptive_compute_cost",
            "polyseg_adaptive_remaining_budget",
            "polyseg_adaptive_fallback",
            "polyseg_adaptive_fallback_reason",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        prompt_lengths = data["prompts"].offsets().diff()
        response_lengths = data["responses"].offsets().diff()
        prompts = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        responses = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

        # 2. rebuild the attention mask aligned with the [prompts | responses] layout.
        prompt_mask = self._lengths_to_mask(prompt_lengths, prompts.size(1))
        response_attention_mask = self._lengths_to_mask(response_lengths, responses.size(1))
        action_mask = data["response_mask"].to_padded_tensor(padding=0).to(torch.int64)
        attention_mask = torch.cat([prompt_mask, response_attention_mask], dim=1)

        # `raw_prompt` is a non-tensor field; depending on the TransferQueue backend it
        # comes back as a tensordict LinkedList (a `list` subclass), a NonTensorStack or a
        # numpy array. `list(...)` normalizes all of them to a plain list where each element
        # is one sample's chat-message list (whereas `.tolist()` only exists on numpy/tensors).
        raw_prompts = list(data["raw_prompt"])
        raw_prompt_arr = np.empty(len(raw_prompts), dtype=object)
        raw_prompt_arr[:] = raw_prompts
        extra_fields = list(data["extra_fields"])
        extra_fields_arr = np.empty(len(extra_fields), dtype=object)
        extra_fields_arr[:] = extra_fields
        data_sources = list(data["data_source"])
        data_sources_arr = np.empty(len(data_sources), dtype=object)
        data_sources_arr[:] = data_sources
        reward_models = list(data["reward_model"])
        reward_models_arr = np.empty(len(reward_models), dtype=object)
        reward_models_arr[:] = reward_models
        extra_infos = list(data["extra_info"])
        extra_infos_arr = np.empty(len(extra_infos), dtype=object)
        extra_infos_arr[:] = extra_infos

        adaptive_fields = {}
        for key in (
            "polyseg_adaptive_q",
            "polyseg_adaptive_m",
            "polyseg_adaptive_planned_cost",
            "polyseg_adaptive_compute_cost",
            "polyseg_adaptive_remaining_budget",
            "polyseg_adaptive_fallback",
            "polyseg_adaptive_fallback_reason",
        ):
            if key in data:
                adaptive_fields[key] = np.asarray(list(data[key]))

        rm_input = DataProto(
            batch=TensorDict(
                {
                    "prompts": prompts,
                    "responses": responses,
                    "response_mask": action_mask,
                    "attention_mask": attention_mask,
                },
                batch_size=len(batch),
            ),
            non_tensor_batch={
                "raw_prompt": raw_prompt_arr,
                "extra_fields": extra_fields_arr,
                "data_source": data_sources_arr,
                "reward_model": reward_models_arr,
                "extra_info": extra_infos_arr,
                **adaptive_fields,
            },
            meta_info={"validate": batch.partition_id == "val"},
        )

        # 3. run the reward model (wakes/sleeps the reward model internally).
        rm_output = self.reward_loop_manager.compute_rm_score(rm_input)

        # The TransferQueue manager owns the same driver-side controller as
        # AgentLoopManagerTQ.  Update it using measured reward-loop cost after
        # the verifier has run; planned m*cost is only a feasibility estimate.
        if batch.partition_id != "val" and getattr(self.agent_loop_manager, "polyseg_controller", None) is not None:
            actual = rm_output.non_tensor_batch.get("polyseg_adaptive_actual_cost")
            if actual is None:
                actual = rm_output.non_tensor_batch.get("segment_query_cost")
            try:
                actual_total = float(np.asarray(actual, dtype=np.float64).reshape(-1).sum())
            except (TypeError, ValueError):
                actual_total = 0.0
            planned_compute = rm_output.non_tensor_batch.get("polyseg_adaptive_compute_cost")
            try:
                compute_total = float(np.asarray(planned_compute, dtype=np.float64).reshape(-1).sum())
            except (TypeError, ValueError):
                compute_total = 0.0
            self.agent_loop_manager.observe_polyseg_cost(
                actual_total, compute_cost=compute_total, episodes=len(batch)
            )

        # 4. write rm_scores (and reward extra info) back to TransferQueue.
        padded_rm_scores = rm_output.batch["rm_scores"]
        rm_scores = torch.nested.as_nested_tensor(
            [padded_rm_scores[i, : response_lengths[i]] for i in range(len(batch))],
            layout=torch.jagged,
        )
        write_back = {"rm_scores": rm_scores}
        # PolySeg metadata has two ragged axes: token-aligned segment ids and
        # macro-aligned rewards/durations/masks.  Preserve each row's native
        # length in a jagged TensorQueue field; dropping these fields here
        # would silently turn the v1 trainer back into outcome-only PPO.
        polyseg_keys = (
            "segment_rewards",
            "segment_ids",
            "segment_durations",
            "segment_mask",
            "segment_feedback_mask",
            "segment_features",
            "step_features",
            "segment_next_features",
            "segment_transition_mask",
        )
        polyseg_batch = rm_output.batch
        segment_masks = polyseg_batch.get("segment_mask")
        for key in polyseg_keys:
            if key not in polyseg_batch.keys():
                continue
            values = []
            for index in range(len(batch)):
                row = polyseg_batch[key][index]
                if key == "segment_ids":
                    expected = int(response_lengths[index].item())
                    if row.shape[0] != expected:
                        raise ValueError(
                            "segment_ids must be response-token aligned in the v1 reward schema: "
                            f"row {index} has {row.shape[0]} ids for {expected} response tokens"
                        )
                    length = expected
                elif key == "step_features":
                    expected = int(response_lengths[index].item())
                    if row.shape[0] != expected:
                        raise ValueError(
                            "step_features must be response-token aligned in the v1 reward schema: "
                            f"row {index} has {row.shape[0]} rows for {expected} response tokens"
                        )
                    length = expected
                elif key == "segment_mask":
                    length = int(row.shape[0])
                elif segment_masks is not None:
                    length = int(segment_masks[index].to(torch.bool).sum().item())
                else:
                    length = int(row.shape[0])
                if key in {"segment_features", "step_features", "segment_next_features"}:
                    values.append(row[:length, :])
                else:
                    values.append(row[:length])
            write_back[key] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
        for key in rm_output.meta_info.get("reward_extra_keys", []):
            write_back[key] = rm_output.non_tensor_batch[key]
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=tu.get_tensordict(write_back),
        )

        return batch

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        """Build a right-padded mask of shape (len(lengths), width) from per-row valid lengths."""
        positions = torch.arange(width, device=lengths.device).unsqueeze(0)
        return (positions < lengths.unsqueeze(1)).to(torch.int64)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        """Return the global batch multiple required by downstream train steps(e.g. critics, actors)."""
        required_multiple = dp_size

        # If enabled with critic training, the batch should align with critic PPO mini-batches.
        if self.use_critic:
            critic_global_mini_batch_size = self.config.critic.ppo_mini_batch_size
            critic_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, critic_global_mini_batch_size)

        # If there is an actor update, the batch should align with actor PPO mini-batches too.
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_global_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            actor_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, actor_global_mini_batch_size)

        # Notice lcm(a, b, c) == lcm(lcm(a, b), c), so it is optimal.
        return required_multiple

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        # get actor dp size
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        dp_size = max(dp_rank_mapping) + 1

        # Upsampling the batch with padding sequences
        batch_multiple = self._get_required_batch_multiple(dp_size)
        batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
        global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        batch.reorder([j for partition in global_partition_lst for j in partition])
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return batch

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            return batch

        # 1. compute log probs
        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        batch = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch)

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def _compute_values(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        batch.extra_info.update(
            {
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output = self.critic_wg.infer_batch(batch)
        # TODO: DataProtoFuture support KVBatchMeta
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["values", "response_mask"]
        )
        data["values"] = response_from_nested(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("values"))

        return batch

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the advantage of the batch."""
        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob"]
        if self.use_critic and self.config.algorithm.adv_estimator != "polyseg_sci":
            fields.append("values")
        if self.config.algorithm.adv_estimator in ("polyseg", "polyseg_gae"):
            fields.extend(
                [
                    "segment_rewards",
                    "segment_ids",
                    "segment_durations",
                    "segment_mask",
                    "segment_feedback_mask",
                    "segment_features",
                    "step_features",
                    "segment_next_features",
                    "segment_transition_mask",
                    # Scalar reward accounting fields are transported with the
                    # batch so fixed-budget experiments can report their actual
                    # verifier cost instead of inferring it from m.
                    "segment_query_cost",
                    "segment_query_count",
                    "segment_count",
                    "polyseg_adaptive_actual_m",
                    "polyseg_m_semantics",
                    "segment_budget_remaining",
                    "episode_reward",
                    "final_accuracy",
                    # Async reward workers attach scalar accounting to the
                    # rollout's extra_fields envelope rather than to a tensor
                    # queue field.  Keep that envelope available below.
                    "extra_fields",
                ]
            )
        elif self.config.algorithm.adv_estimator == "polyseg_sci":
            # Science batches carry only the explicit vector boundary contract;
            # requesting scalar PolySeg fields here would make a valid science
            # rollout look malformed in the TransferQueue.
            fields.extend(
                [
                    "segment_ids",
                    "segment_mask",
                    "science_boundary_values",
                    "science_boundary_mask",
                    "science_endpoint_mask",
                    "extra_fields",
                ]
            )
            science_cfg = self.config.algorithm.get("polyseg", {}).get("science", {})
            from verl.trainer.ppo.polyseg_sci import _science_objective_mode

            science_mode = _science_objective_mode(self.config.algorithm) or "preference"
            if science_mode == "pareto":
                fields.append("science_pareto_alpha")
            elif science_mode == "constraint":
                fields.append("science_constraint_multipliers")
                # Estimates are needed only by the explicitly enabled dual
                # bootstrap compatibility path; production runs restore a
                # persistent dual state and do not need this field.
                allow_dual_bootstrap = science_cfg.get("allow_dual_bootstrap", False)
                if not isinstance(allow_dual_bootstrap, bool):
                    raise ValueError("science allow_dual_bootstrap must be boolean")
                if allow_dual_bootstrap:
                    fields.append("science_endpoint_estimates")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # In the async AgentLoop path reward extras are nested under
        # ``extra_fields.reward_extra_info``.  Extract scalar accounting before
        # converting the remaining TensorDict to padded tensors.
        polyseg_accounting: dict[str, list[float]] = defaultdict(list)
        if self.config.algorithm.adv_estimator in ("polyseg", "polyseg_gae", "polyseg_sci") and "extra_fields" in data:
            raw_extra_fields = data.pop("extra_fields")
            try:
                extra_items = raw_extra_fields.tolist() if hasattr(raw_extra_fields, "tolist") else list(raw_extra_fields)
            except (TypeError, ValueError):
                extra_items = []
            accounting_keys = (
                "segment_query_cost",
                "segment_query_count",
                "segment_count",
                "polyseg_adaptive_actual_m",
                "polyseg_m_semantics",
                "segment_budget_remaining",
                "episode_reward",
                "final_accuracy",
                "polyseg_adaptive_q",
                "polyseg_adaptive_m",
                "polyseg_adaptive_planned_cost",
                "polyseg_adaptive_compute_cost",
                "polyseg_adaptive_actual_cost",
                "polyseg_adaptive_remaining_budget",
                "polyseg_adaptive_fallback_count",
                "segment_feedback_fallback_count",
                "tool_call_count",
                "tool_call_success_count",
                "tool_call_failure_count",
            )
            for extra_item in extra_items:
                info = extra_item.get("reward_extra_info", {}) if isinstance(extra_item, dict) else {}
                if not isinstance(info, dict):
                    info = {}
                for key in accounting_keys:
                    value = info.get(key)
                    if value is None and isinstance(extra_item, dict):
                        value = extra_item.get(key)
                    if value is not None:
                        try:
                            polyseg_accounting[key].append(float(value))
                        except (TypeError, ValueError):
                            pass

        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)
        for key, values in polyseg_accounting.items():
            if values:
                data.non_tensor_batch[key] = np.asarray(values, dtype=np.float32)
        controller = getattr(self.agent_loop_manager, "polyseg_controller", None)
        if controller is not None and polyseg_accounting.get("polyseg_adaptive_actual_cost"):
            measured = float(np.asarray(polyseg_accounting["polyseg_adaptive_actual_cost"], dtype=np.float32).sum())
            measured_compute = float(np.asarray(
                polyseg_accounting.get("polyseg_adaptive_compute_cost", []), dtype=np.float32
            ).sum())
            metrics["polyseg/adaptive_actual_batch_cost"] = measured
            metrics["polyseg/adaptive_compute_batch_cost"] = measured_compute
            # ``_compute_reward_colocate`` is skipped when distributed reward
            # workers are active. Charge this batch here in that deployment;
            # the colocated path already charged it before this method ran.
            if self.reward_loop_manager.reward_loop_worker_handles is not None:
                controller.observe(
                    measured,
                    compute_cost=measured_compute,
                    episodes=len(polyseg_accounting["polyseg_adaptive_actual_cost"]),
                )
        if controller is not None and controller.remaining_budget is not None:
            data.non_tensor_batch["polyseg_adaptive_remaining_budget"] = np.full(
                len(data), controller.remaining_budget, dtype=np.float32
            )
            # Reward extras carry the pre-verifier ledger value attached at
            # rollout selection time.  Replace that stale value too, because
            # the final metric reduction below also reads the extracted
            # ``polyseg_accounting`` mapping.
            if "polyseg_adaptive_remaining_budget" in polyseg_accounting:
                polyseg_accounting["polyseg_adaptive_remaining_budget"] = [
                    float(controller.remaining_budget)
                ] * len(polyseg_accounting["polyseg_adaptive_remaining_budget"])
        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # Constraint-mode science updates consume the driver-owned projected
        # dual state.  Reattach the current vector after TransferQueue
        # retrieval so a stale or worker-local multiplier can never silently
        # steer the actor.  The state is updated only by the audited science
        # harness through AgentLoopManager.update_science_dual_state().
        if self.config.algorithm.adv_estimator == "polyseg_sci":
            from verl.trainer.ppo.polyseg_sci import _science_objective_mode

            science_mode = _science_objective_mode(self.config.algorithm) or "preference"
            if science_mode == "constraint":
                science_state = getattr(self.agent_loop_manager, "science_dual_state", None)
                if science_state is not None:
                    data.batch["science_constraint_multipliers"] = science_state.multipliers.to(
                        device=data.batch["response_mask"].device, dtype=torch.float32
                    )

        # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )
        # Dependence estimates are available only after scoring this batch.
        # Feed the selected (q, m) batch's conservative upper bounds to the
        # shared driver-side controller for the *next* rollout batch.
        controller = getattr(self.agent_loop_manager, "polyseg_controller", None)
        if controller is not None and self.config.algorithm.adv_estimator in ("polyseg", "polyseg_gae"):
            try:
                selected_m = np.asarray(data.non_tensor_batch.get("polyseg_adaptive_m"), dtype=np.float64)
                selected_q = np.asarray(data.non_tensor_batch.get("polyseg_adaptive_q"), dtype=np.float64)
                seg_upper = np.asarray(data.non_tensor_batch.get("polyseg_kappa_segment_selected_upper"), dtype=np.float64)
                if seg_upper.size == 0:
                    seg_upper = np.asarray(data.non_tensor_batch.get("polyseg_kappa_segment_upper"), dtype=np.float64)
                tr_upper = np.asarray(data.non_tensor_batch.get("polyseg_kappa_trajectory_upper"), dtype=np.float64)
                selected_m = selected_m[np.isfinite(selected_m)]
                selected_q = selected_q[np.isfinite(selected_q)]
                seg_upper = seg_upper[np.isfinite(seg_upper)]
                tr_upper = tr_upper[np.isfinite(tr_upper)]
                if selected_m.size and seg_upper.size and tr_upper.size:
                    controller.update_dependence_diagnostics(
                        m=int(round(float(selected_m.mean()))),
                        kappa_segment_upper=float(seg_upper.mean()),
                        kappa_trajectory_upper=float(tr_upper.mean()),
                        q=None if not selected_q.size else int(round(float(selected_q.mean()))),
                    )
            except (TypeError, ValueError, AttributeError):
                # Missing diagnostics preserve the conservative unseen-m
                # fallback and must not abort an otherwise valid PPO batch.
                pass
        metrics.update(data.meta_info.pop("polyseg_metrics", {}))
        if self.config.algorithm.adv_estimator in ("polyseg", "polyseg_gae"):
            if controller is not None and controller.total_budget is not None:
                if controller.remaining_budget is not None:
                    data.non_tensor_batch["polyseg_adaptive_remaining_budget"] = np.full(
                        len(data), controller.remaining_budget, dtype=np.float32
                    )
            for source_key, metric_key in (
                ("polyseg_critic_mse", "polyseg/critic_mse"),
                ("polyseg_bellman_residual_mse", "polyseg/bellman_residual_mse"),
                ("polyseg_bellman_closure_residual", "polyseg/bellman_closure_residual"),
                ("polyseg_confidence_bonus", "polyseg/confidence_bonus"),
                ("polyseg_confidence_radius", "polyseg/confidence_radius"),
                ("polyseg_confidence_delta", "polyseg/confidence_delta"),
                ("polyseg_confidence_pre_update", "polyseg/confidence_pre_update"),
                ("polyseg_feature_condition", "polyseg/feature_condition"),
                ("polyseg_feature_min_eigenvalue", "polyseg/feature_min_eigenvalue"),
                ("polyseg_feature_effective_samples", "polyseg/effective_samples"),
                ("polyseg_feature_batch_samples", "polyseg/batch_samples"),
                ("polyseg_estimator_sample_count", "polyseg/estimator_samples"),
                ("polyseg_transition_effective_samples", "polyseg/transition_samples"),
                ("polyseg_feature_dimension", "polyseg/feature_dimension"),
                ("polyseg_empirical_spectral_tail", "polyseg/empirical_spectral_tail"),
                ("polyseg_theta_q_stability", "polyseg/theta_q_stability"),
                ("polyseg_heldout_return_residual_mse", "polyseg/heldout_return_residual_mse"),
                ("polyseg_heldout_bellman_residual_mse", "polyseg/heldout_bellman_residual_mse"),
                ("polyseg_transition_confidence_bonus", "polyseg/transition_confidence_bonus"),
                ("polyseg_transition_feature_condition", "polyseg/transition_feature_condition"),
                ("polyseg_transition_feature_min_eigenvalue", "polyseg/transition_min_eigenvalue"),
                ("polyseg_kappa_segment", "polyseg/kappa_segment"),
                ("polyseg_kappa_segment_upper", "polyseg/kappa_segment_upper"),
                ("polyseg_kappa_segment_selected_upper", "polyseg/kappa_segment_selected_upper"),
                ("polyseg_kappa_trajectory", "polyseg/kappa_trajectory"),
                ("polyseg_kappa_trajectory_upper", "polyseg/kappa_trajectory_upper"),
                ("polyseg_nominal_observations", "polyseg/nominal_observations"),
                ("polyseg_dependence_cluster_count", "polyseg/dependence_cluster_count"),
                ("polyseg_total_observation_weight", "polyseg/total_observation_weight"),
                ("polyseg_weighted_ess_ratio", "polyseg/weighted_ess_ratio"),
                ("polyseg_dependence_weighted_by_duration", "polyseg/dependence_weighted_by_duration"),
                ("polyseg_effective_samples", "polyseg/dependence_effective_samples"),
                ("polyseg_weighted_effective_samples", "polyseg/weighted_effective_samples"),
                ("polyseg_iid_cluster_radius_ratio", "polyseg/iid_cluster_radius_ratio"),
                ("polyseg_iid_cluster_radius_ratio_upper", "polyseg/iid_cluster_radius_ratio_upper"),
                ("polyseg_chi_hat", "polyseg/chi_hat"),
                ("polyseg_hac_lag", "polyseg/hac_lag"),
                ("polyseg_dependence_diagnostic_oof", "polyseg/dependence_diagnostic_oof"),
                ("polyseg_dependence_residual_rms", "polyseg/dependence_residual_rms"),
                ("polyseg_dependence_radius_multiplier", "polyseg/dependence_radius_multiplier"),
                ("polyseg_dependence_radius_multiplier_pre_update", "polyseg/dependence_radius_multiplier_pre_update"),
                ("polyseg_dependence_radius_multiplier_post_update", "polyseg/dependence_radius_multiplier_post_update"),
                ("polyseg_confidence_radius_pre_update", "polyseg/confidence_radius_pre_update"),
                ("polyseg_confidence_radius_post_update", "polyseg/confidence_radius_post_update"),
                ("polyseg_dependence_state_batches", "polyseg/dependence_state_batches"),
                ("polyseg_dependence_logistic_score", "polyseg/dependence_logistic_score"),
                ("polyseg_dependence_persistent_oof", "polyseg/dependence_persistent_oof"),
                ("polyseg_dependence_history_trajectories", "polyseg/dependence_history_trajectories"),
            ):
                if source_key in data.non_tensor_batch:
                    metrics[metric_key] = float(np.asarray(data.non_tensor_batch[source_key], dtype=np.float32).mean())
            for source_key, metric_key in (
                ("segment_query_cost", "polyseg/query_cost"),
                ("segment_query_count", "polyseg/query_count"),
                ("segment_count", "polyseg/segment_count"),
                ("polyseg_adaptive_actual_m", "polyseg/adaptive_actual_m"),
                ("segment_budget_remaining", "polyseg/budget_remaining"),
                ("episode_reward", "polyseg/episode_reward"),
                ("final_accuracy", "polyseg/final_accuracy"),
                ("prm_score_mean", "polyseg/prm_score_mean"),
                ("prm_score_final", "polyseg/prm_score_final"),
                ("polyseg_adaptive_q", "polyseg/adaptive_q"),
                ("polyseg_adaptive_m", "polyseg/adaptive_m"),
                ("polyseg_adaptive_planned_cost", "polyseg/adaptive_planned_cost"),
                ("polyseg_adaptive_compute_cost", "polyseg/adaptive_compute_cost"),
                ("polyseg_adaptive_actual_cost", "polyseg/adaptive_actual_cost"),
                ("polyseg_adaptive_remaining_budget", "polyseg/adaptive_remaining_budget"),
                ("polyseg_adaptive_fallback_count", "polyseg/adaptive_fallback_count"),
                ("segment_feedback_fallback_count", "polyseg/feedback_fallback_count"),
                ("tool_call_count", "tool_calls/count"),
                ("tool_call_success_count", "tool_calls/success_count"),
                ("tool_call_failure_count", "tool_calls/failure_count"),
            ):
                value = data.non_tensor_batch.get(source_key)
                if value is None:
                    value = data.batch.get(source_key)
                if value is not None:
                    try:
                        array = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, dtype=np.float32)
                        finite = array[np.isfinite(array)]
                        if finite.size:
                            metrics[metric_key] = float(finite.mean())
                    except (TypeError, ValueError):
                        logger.debug("Unable to reduce PolySeg accounting field %s", source_key, exc_info=True)
            # Cross-checks for the worker journal; never turn missing sample
            # costs into zero or infer totals by multiplying batch means.
            for kind in ("cost", "count"):
                values = data.non_tensor_batch.get(f"segment_query_{kind}")
                if values is None:
                    values = data.batch.get(f"segment_query_{kind}")
                if values is None:
                    continue
                try:
                    array = np.asarray(values.detach().cpu() if torch.is_tensor(values) else values, dtype=np.float64)
                    if array.size == len(data) and np.isfinite(array).all() and (array >= 0).all():
                        metrics[f"polyseg/query_{kind}_batch_total"] = float(array.sum())
                except (TypeError, ValueError):
                    logger.debug("Invalid PolySeg query %s batch accounting", kind, exc_info=True)
            for source_key, metric_key in (
                ("segment_query_cost", "polyseg/query_cost"),
                ("segment_query_count", "polyseg/query_count"),
                ("segment_count", "polyseg/segment_count"),
                ("polyseg_adaptive_actual_m", "polyseg/adaptive_actual_m"),
                ("segment_budget_remaining", "polyseg/budget_remaining"),
                ("episode_reward", "polyseg/episode_reward"),
                ("final_accuracy", "polyseg/final_accuracy"),
                ("prm_score_mean", "polyseg/prm_score_mean"),
                ("prm_score_final", "polyseg/prm_score_final"),
                ("polyseg_adaptive_q", "polyseg/adaptive_q"),
                ("polyseg_adaptive_m", "polyseg/adaptive_m"),
                ("polyseg_adaptive_planned_cost", "polyseg/adaptive_planned_cost"),
                ("polyseg_adaptive_actual_cost", "polyseg/adaptive_actual_cost"),
                ("polyseg_adaptive_remaining_budget", "polyseg/adaptive_remaining_budget"),
                ("polyseg_adaptive_fallback_count", "polyseg/adaptive_fallback_count"),
                ("segment_feedback_fallback_count", "polyseg/feedback_fallback_count"),
            ):
                values = polyseg_accounting.get(source_key)
                if values:
                    metrics[metric_key] = float(np.mean(values))
            planned_values = data.non_tensor_batch.get("polyseg_adaptive_planned_cost")
            if planned_values is not None:
                try:
                    metrics["polyseg/adaptive_planned_batch_cost"] = float(
                        np.asarray(planned_values, dtype=np.float32).sum()
                    )
                except (TypeError, ValueError):
                    logger.debug("Unable to reduce adaptive planned batch cost", exc_info=True)
            compute_values = data.non_tensor_batch.get("polyseg_adaptive_compute_cost")
            if compute_values is not None:
                try:
                    metrics["polyseg/adaptive_compute_batch_cost"] = float(
                        np.asarray(compute_values, dtype=np.float32).sum()
                    )
                except (TypeError, ValueError):
                    logger.debug("Unable to reduce adaptive compute batch cost", exc_info=True)

        # 4. write nested advantages and returns back to TransferQueue
        fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            output[field] = response_to_nested(data.batch[field], response_mask)
        output = TensorDict(output, batch_size=len(batch))

        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

        return batch

    def _update_critic(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the critic network."""
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        extra_info = {
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.critic.ppo_epochs,
            "seed": self.config.critic.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.critic.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)

        output: DataProtoFuture = self.critic_wg.train_mini_batch(batch)
        output: TensorDict = output.get()
        output = rename_dict(output["metrics"], "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_metrics = reduce_metrics(output)
        metrics.update(critic_metrics)

        return batch

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the actor network."""
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        distillation_only = False  # distillation_only flag means we can skip policy loss and reduce mem footprint
        if is_distillation_enabled(self.config.get("distillation")):
            distillation_loss_cfg = self.distillation_config.distillation_loss
            distillation_only = (
                distillation_use_topk
                and not distillation_loss_cfg.use_task_rewards
                and not distillation_loss_cfg.use_policy_gradient
            )
        score_centering = bool(
            self.config.actor_rollout_ref.actor.policy_loss.get("rollout_correction", {}).get("score_centering", False)
        )
        global_macro_batch_size = None
        if self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla") == "polyseg":
            segment_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["segment_ids", "segment_feedback_mask", "response_mask"],
            )
            if "segment_ids" not in segment_data:
                raise ValueError("PolySeg actor PPO requires rollout-owned segment_ids in the TransferQueue batch")
            ids = segment_data["segment_ids"].to_padded_tensor(-1)
            masks = segment_data["response_mask"].to_padded_tensor(0).to(torch.bool)
            feedback_masks = (
                segment_data["segment_feedback_mask"].to_padded_tensor(0).to(torch.bool)
                if "segment_feedback_mask" in segment_data
                else None
            )
            global_macro_batch_size = 0
            for row_index, (row_ids, row_mask) in enumerate(zip(ids, masks, strict=True)):
                valid_ids = row_ids[row_mask & (row_ids >= 0)]
                if feedback_masks is not None and valid_ids.numel():
                    feedback = feedback_masks[row_index]
                    in_range = valid_ids < feedback.shape[0]
                    labels = torch.zeros_like(in_range)
                    if torch.any(in_range):
                        labels[in_range] = feedback[valid_ids[in_range]]
                    valid_ids = valid_ids[labels]
                global_macro_batch_size += int(torch.unique(valid_ids).numel())
        extra_info = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "distillation_only": distillation_only,
            "score_centering": score_centering,
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
            **({"global_macro_batch_size": global_macro_batch_size} if global_macro_batch_size is not None else {}),
        }
        batch.extra_info.update(extra_info)

        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        actor_metrics = reduce_metrics(output)
        metrics.update(actor_metrics)

        return batch

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        # 1. collect necessary fields from TransferQueue for computing metrics
        non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "advantages",
            "returns",
            "rm_scores",
            "token_level_rewards",
            "num_turns",
            "tool_call_count",
            "tool_call_success_count",
            "tool_call_failure_count",
        ]
        if self.use_critic:
            fields.append("values")
        moe_lb_metrics_interval = self.config.actor_rollout_ref.rollout.get("moe_load_balance_metrics_interval", 0)
        data = get_metric_data_with_optional_routed_experts(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=fields,
            moe_lb_metrics_interval=moe_lb_metrics_interval,
            global_steps=global_steps,
            accumulator=self._rollout_moe_lb_metrics_accumulator,
            kv_batch_get=tq.kv_batch_get,
        )

        num_turns = np.array(data.pop("num_turns").tolist())
        prompt_length = data["prompts"].offsets().diff()
        response_length = data["responses"].offsets().diff()
        global_token_num = (prompt_length + response_length).tolist()
        # Persist realized token ledgers for every training step.  Maximum
        # response length is only a cap; these counters are the actual costs
        # consumed by the rollout and distinguish assistant action tokens from
        # tool-observation tokens.  The runner sums these batch totals across
        # the whole run before accepting any budget comparison.
        response_mask = data["response_mask"].to_padded_tensor(padding=0).to(torch.float32)
        action_token_length = response_mask.sum(dim=1)
        valid_rows = torch.as_tensor(non_padding_mask, dtype=torch.bool, device=prompt_length.device)
        metrics.update(
            {
                "resource/prompt_tokens_batch_total": int(prompt_length[valid_rows].sum().item()),
                "resource/response_tokens_batch_total": int(response_length[valid_rows].sum().item()),
                "resource/action_tokens_batch_total": int(action_token_length[valid_rows].sum().item()),
                "resource/tool_observation_tokens_batch_total": int(
                    (response_length[valid_rows] - action_token_length[valid_rows]).sum().item()
                ),
                "resource/total_tokens_batch_total": int(
                    (prompt_length[valid_rows] + response_length[valid_rows]).sum().item()
                ),
            }
        )
        for field, metric_key in (
            ("tool_call_count", "resource/tool_calls_batch_total"),
            ("tool_call_success_count", "resource/tool_successes_batch_total"),
            ("tool_call_failure_count", "resource/tool_failures_batch_total"),
        ):
            if field in data:
                values = data[field].to(torch.float32) if torch.is_tensor(data[field]) else torch.as_tensor(data[field], dtype=torch.float32)
                metrics[metric_key] = int(values[valid_rows].sum().item())
        min_global_steps = np.array([tag["min_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]
        max_global_steps = np.array([tag["max_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]

        # Only fetch speculative decoding stats when rollout writes them.
        spec_drafts = spec_accepts = spec_verifies = None
        mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
        if mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout:
            spec_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields = spec_data.pop("extra_fields").tolist()
            # The rollout omits the spec_* stats when the backend does not report
            # per-request spec-decode stats; leave all three as None in that case.
            if extra_fields and all(
                isinstance(extra_field, dict) and "spec_num_draft_tokens" in extra_field for extra_field in extra_fields
            ):
                spec_drafts = [extra_field["spec_num_draft_tokens"] for extra_field in extra_fields]
                spec_accepts = [extra_field["spec_num_accepted_tokens"] for extra_field in extra_fields]
                spec_verifies = [extra_field["spec_num_verify_steps"] for extra_field in extra_fields]

        data = data.to_padded_tensor()
        data["token_level_scores"] = data["rm_scores"]
        if "token_level_rewards" not in data:
            data["token_level_rewards"] = data["rm_scores"]
        data["prompt_length"] = prompt_length.float()
        data["response_length"] = response_length.float()
        batch = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
        metrics_batch = batch.select_idxs(non_padding_mask) if non_padding_mask.any() else batch
        # Expose per-row turn counts under the V0 schema (agent_loop writes
        # ``__num_turns__`` into the non-tensor batch) so compute_data_metrics
        # emits the V0-compatible ``num_turns/{mean,max,min}`` tags in addition
        # to the ``training/num_turns/*`` names computed below.
        num_turns_for_metrics = num_turns[non_padding_mask] if non_padding_mask.any() else num_turns
        metrics_batch.non_tensor_batch["__num_turns__"] = np.asarray(num_turns_for_metrics, dtype=np.int32)

        # 2. compute metrics
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(
            compute_moe_lb_metrics(
                metrics_batch=metrics_batch,
                moe_lb_metrics_interval=moe_lb_metrics_interval,
                global_steps=global_steps,
                accumulator=self._rollout_moe_lb_metrics_accumulator,
            )
        )
        metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self._get_n_gpus_for_throughput()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

        # 3. other auxiliary metrics
        if non_padding_mask.any():
            num_turns = num_turns[non_padding_mask]
        metrics.update(
            {
                "training/num_turns/mean": num_turns.mean(),
                "training/num_turns/max": num_turns.max(),
                "training/num_turns/min": num_turns.min(),
            }
        )

        # 4. per-request speculative-decoding aggregation (same metrics async PPO logs;
        # see compute_spec_decode_metrics in verl/trainer/ppo/ray_trainer.py).
        metrics.update(compute_spec_decode_metrics(spec_drafts, spec_accepts, spec_verifies, non_padding_mask))

        # 5. off-policy staleness metrics
        #   global_steps is the model weight version (one update_weights per global_step), and
        #   min/max_global_steps are the versions a trajectory was generated across, so all quantities
        #   below are already in model-version units.
        #   - trajectory_spans: how many distinct model versions a single trajectory was
        #     generated across (1 == fully generated on a single version). This captures the
        #     within-trajectory policy inconsistency caused by partial rollout / continuation.
        #   - trajectory_staleness: how many model versions the trajectory lags behind the
        #     *current* policy. A trajectory spans versions [min_global_steps, max_global_steps],
        #     so the lag is a range: the freshest weights used give the lower bound
        #     (global_steps - max_global_steps) and the oldest weights the worst case
        #     (global_steps - min_global_steps). We log the lower bound as the primary metric.
        trajectory_spans = max_global_steps - min_global_steps + 1
        trajectory_staleness = (global_steps - 1) - max_global_steps
        trajectory_staleness_worst = (global_steps - 1) - min_global_steps
        metrics.update(
            {
                "training/off_policy/trajectory_spans/mean": trajectory_spans.mean(),
                "training/off_policy/trajectory_spans/max": trajectory_spans.max(),
                "training/off_policy/trajectory_spans/min": trajectory_spans.min(),
                "training/off_policy/trajectory_staleness/mean": trajectory_staleness.mean(),
                "training/off_policy/trajectory_staleness/max": trajectory_staleness.max(),
                "training/off_policy/trajectory_staleness/min": trajectory_staleness.min(),
                "training/off_policy/trajectory_staleness_worst/mean": trajectory_staleness_worst.mean(),
                "training/off_policy/trajectory_staleness_worst/max": trajectory_staleness_worst.max(),
                "training/off_policy/trajectory_staleness_worst/min": trajectory_staleness_worst.min(),
            }
        )


TRAINER_REGISTRY: dict[str, type[PPOTrainer]] = {}


def register_trainer(name: str):
    """Class decorator that registers a :class:`PPOTrainer` subclass under ``name``."""

    def decorator(cls: type[PPOTrainer]) -> type[PPOTrainer]:
        if not (isinstance(cls, type) and issubclass(cls, PPOTrainer)):
            raise TypeError(f"register_trainer expected a PPOTrainer subclass, got {cls!r}")
        existing = TRAINER_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Trainer name '{name}' is already registered to {existing.__name__}; "
                f"cannot re-register it to {cls.__name__}."
            )
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def get_trainer_cls(name: str) -> type[PPOTrainer]:
    """Return the :class:`PPOTrainer` subclass registered under ``name``."""
    try:
        return TRAINER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(TRAINER_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown trainer '{name}'. Available trainers: {available}.") from None
