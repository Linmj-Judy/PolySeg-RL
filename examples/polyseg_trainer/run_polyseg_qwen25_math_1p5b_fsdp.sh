#!/usr/bin/env bash
# Launch a small PolySeg-RL run from the verl FSDP recipe.
set -xeuo pipefail

ROOT=${ROOT:-/volume/viewers/judy/rl_research}
MODEL_PATH=${MODEL_PATH:-$ROOT/models/base/Qwen/Qwen2.5-Math-1.5B}
TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE to an agentic dataset containing agent_name/tool metadata}
VAL_FILE=${VAL_FILE:?set VAL_FILE to an agentic validation dataset}
FUNCTION_TOOL_PATH=${FUNCTION_TOOL_PATH:?set FUNCTION_TOOL_PATH to the real tool implementation used by the dataset}
POLYSEG_SUFFIX_ENDPOINT=${POLYSEG_SUFFIX_ENDPOINT:?set POLYSEG_SUFFIX_ENDPOINT to the suffix sampler/verifier service}
M=${M:-4}
Q=${Q:-2}
SUFFIX_SAMPLES=${SUFFIX_SAMPLES:-4}
BUDGET=${BUDGET:-$((M * SUFFIX_SAMPLES))}
N_GPUS=${N_GPUS:-4}
export POLYSEG_SUFFIX_ENDPOINT POLYSEG_SUFFIX_SAMPLES=${SUFFIX_SAMPLES}

cd "$(dirname "$0")/../.."
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=polyseg \
  algorithm.polyseg.q=${Q} \
  algorithm.polyseg.m=${M} \
  algorithm.polyseg.feature_dim=4 \
  algorithm.polyseg.basis=total \
  algorithm.polyseg.feedback=utility_delta \
  algorithm.polyseg.normalize_advantages=True \
  algorithm.polyseg.budget=${BUDGET} \
  actor_rollout_ref.actor.policy_loss.loss_mode=polyseg \
  actor_rollout_ref.actor.polyseg_num_segments=${M} \
  actor_rollout_ref.rollout.polyseg_fixed_m=${M} \
  reward.reward_manager.name=polyseg \
  reward.custom_reward_function.path="${PWD}/examples/polyseg_trainer/suffix_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_kwargs.m=${M} \
  reward.reward_kwargs.feedback=utility_delta \
  reward.reward_kwargs.feedback_cost_policy=reported_with_receipt \
  reward.reward_kwargs.segment_cost=${SUFFIX_SAMPLES} \
  reward.reward_kwargs.require_rollout_segments=true \
  reward.reward_kwargs.budget=${BUDGET} \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.function_tool_path="${FUNCTION_TOOL_PATH}" \
  data.train_files="['${TRAIN_FILE}']" \
  data.val_files="['${VAL_FILE}']" \
  data.train_batch_size=32 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  critic.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.logger='["console"]' \
  trainer.project_name=polyseg_rl \
  trainer.experiment_name="polyseg_agentic_q${Q}_m${M}_b${BUDGET}" \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  "$@"
