#!/usr/bin/env bash
# One-step real-data agentic PolySeg verifier smoke (no synthetic reward).
set -xeuo pipefail

ROOT=${ROOT:-/volume/viewers/judy/rl_research}
# The tool-agent contract is instruction-following; callers can set MODEL_PATH
# to a base checkpoint for a controlled base-model comparison.
MODEL_PATH=${MODEL_PATH:-$ROOT/models/instruction/Qwen/Qwen2.5-Math-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$ROOT/datasets/agentic/gsm8k_polyseg/train.parquet}
VAL_FILE=${VAL_FILE:-$ROOT/datasets/agentic/gsm8k_polyseg/test.parquet}
FUNCTION_TOOL_PATH=${FUNCTION_TOOL_PATH:-$ROOT/verl/examples/polyseg_trainer/function_tools.py}
M=${M:-4}
Q=${Q:-2}
# Four H100s are the default for the full 1.5B protocol; callers can override
# this for larger checkpoints or dedicated resource splits.
N_GPUS=${N_GPUS:-4}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-polyseg}
LOSS_MODE=${LOSS_MODE:-polyseg}
REWARD_MANAGER=${REWARD_MANAGER:-polyseg}
USE_POLYNOMIAL_CRITIC=${USE_POLYNOMIAL_CRITIC:-true}
CRITIC_TYPE=${CRITIC_TYPE:-polynomial}
ACCUMULATE_ESTIMATOR=${ACCUMULATE_ESTIMATOR:-false}
CRITIC_ENABLE=${CRITIC_ENABLE:-}
USE_CONFIDENCE_BONUS=${USE_CONFIDENCE_BONUS:-false}
REWARD_CONFIDENCE_BETA=${REWARD_CONFIDENCE_BETA:-1.0}
TRANSITION_CONFIDENCE_BETA=${TRANSITION_CONFIDENCE_BETA:-1.0}
REQUIRE_ROLLOUT_SEGMENTS=${REQUIRE_ROLLOUT_SEGMENTS:-true}
AGENT_LOOP=${AGENT_LOOP:-tool_agent}
POLYSEG_FIXED_M=${POLYSEG_FIXED_M:-0}
POLYSEG_ADAPTIVE=${POLYSEG_ADAPTIVE:-false}
POLYSEG_Q_CANDIDATES=${POLYSEG_Q_CANDIDATES:-0,1,2,4}
POLYSEG_M_CANDIDATES=${POLYSEG_M_CANDIDATES:-1,2,4,8}
POLYSEG_TOTAL_BUDGET=${POLYSEG_TOTAL_BUDGET:-}
FEEDBACK=${FEEDBACK:-terminal}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-$ROOT/verl/examples/polyseg_trainer/reward.py}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-$ROOT/models/reward_models/Qwen/Qwen2.5-Math-7B-PRM800K}
POLYSEG_PRM_DEVICE=${POLYSEG_PRM_DEVICE:-auto}
PRM_MAX_LENGTH=${PRM_MAX_LENGTH:-4096}
SEGMENT_COST=${SEGMENT_COST:-1.0}
BUDGET_SCOPE=${BUDGET_SCOPE:-trajectory}
PYTHON=${PYTHON:-/volume/viewers/judy/miniconda3/envs/polyseg-verl/bin/python}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# null lets the trainer derive one full epoch from the complete dataloader;
# matrix runs pass an explicit resolved step count into their sidecars.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SEED=${SEED:-42}
MODEL_TAG=${MODEL_TAG:-${MODEL_PATH##*/}}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-polyseg_gsm8k_agentic_${MODEL_TAG}_r${MAX_RESPONSE_LENGTH}_${REWARD_MANAGER}_q${Q}_m${M}}
ESTIMATOR_STATE_ID=${ESTIMATOR_STATE_ID:-${EXPERIMENT_NAME}}

# A 128-token budget is legal for generic single-turn PPO, but is usually too
# short for this two-turn GSM8K tool trajectory (assistant action + tool result
# + final answer). Keep low-budget ablations possible while making the failure
# mode explicit in the log instead of silently producing all-zero verifier
# scores after truncation.
if (( MAX_RESPONSE_LENGTH < 256 )); then
  echo "WARNING: MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH} is likely too short for the agentic GSM8K protocol; final verifier markers may be truncated."
fi

# ``data.max_response_length`` is the generated-token budget.  It is not the
# model context length.  Fail early when the requested prompt + response
# budget cannot fit the checkpoint's positional limit instead of letting the
# rollout backend truncate the trajectory silently.
"$PYTHON" - "$MODEL_PATH" "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" <<'PY'
import json
import sys
from pathlib import Path

model_path, prompt_length, response_length = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
config_path = Path(model_path) / "config.json"
if config_path.is_file():
    config = json.loads(config_path.read_text())
    context_length = config.get("max_position_embeddings")
    if context_length is None and isinstance(config.get("text_config"), dict):
        context_length = config["text_config"].get("max_position_embeddings")
    if context_length is not None:
        requested = prompt_length + response_length
        if requested > int(context_length):
            raise SystemExit(
                f"MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH = {requested} exceeds "
                f"{model_path} context limit {context_length}; lower the budgets "
                "or choose a longer-context checkpoint."
            )
        print(f"model_context={context_length} requested_prompt_plus_response={requested}")
PY

cd "$ROOT/verl"
TRAIN_CMD=(
  "$PYTHON" -m verl.trainer.main_ppo \
  trainer.use_v1=true \
  trainer.v1.trainer_mode=sync \
  transfer_queue.enable=true \
  algorithm.adv_estimator=${ADV_ESTIMATOR} \
  algorithm.polyseg.q=${Q} \
  algorithm.polyseg.m=${M} \
  algorithm.polyseg.feature_dim=4 \
  algorithm.polyseg.feedback=${FEEDBACK} \
  algorithm.polyseg.use_polynomial_critic=${USE_POLYNOMIAL_CRITIC} \
  algorithm.polyseg.critic_type=${CRITIC_TYPE} \
  algorithm.polyseg.use_confidence_bonus=${USE_CONFIDENCE_BONUS} \
  algorithm.polyseg.reward_confidence_beta=${REWARD_CONFIDENCE_BETA} \
  algorithm.polyseg.transition_confidence_beta=${TRANSITION_CONFIDENCE_BETA} \
  algorithm.polyseg.accumulate_estimator=${ACCUMULATE_ESTIMATOR} \
  algorithm.polyseg.estimator_state_id="${ESTIMATOR_STATE_ID}" \
  algorithm.polyseg.normalize_advantages=true \
  actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
  actor_rollout_ref.actor.polyseg_num_segments=${M} \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP} \
  actor_rollout_ref.rollout.polyseg_fixed_m=${POLYSEG_FIXED_M} \
  actor_rollout_ref.rollout.polyseg_adaptive=${POLYSEG_ADAPTIVE} \
  actor_rollout_ref.rollout.polyseg_q_candidates="[${POLYSEG_Q_CANDIDATES}]" \
  actor_rollout_ref.rollout.polyseg_m_candidates="[${POLYSEG_M_CANDIDATES}]" \
  actor_rollout_ref.rollout.polyseg_feedback=${FEEDBACK} \
  actor_rollout_ref.rollout.multi_turn.function_tool_path="${FUNCTION_TOOL_PATH}" \
  reward.reward_manager.name=${REWARD_MANAGER} \
  reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_kwargs.m=${M} \
  reward.reward_kwargs.feedback=${FEEDBACK} \
  reward.reward_kwargs.segment_cost=${SEGMENT_COST} \
  reward.reward_kwargs.budget_scope=${BUDGET_SCOPE} \
  reward.reward_kwargs.require_rollout_segments=${REQUIRE_ROLLOUT_SEGMENTS} \
  data.train_files="['${TRAIN_FILE}']" \
  data.val_files="['${VAL_FILE}']" \
  data.train_max_samples=${TRAIN_MAX_SAMPLES} \
  data.val_max_samples=${VAL_MAX_SAMPLES} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.seed=${SEED} \
  data.dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  critic.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.seed=${SEED} \
  actor_rollout_ref.actor.data_loader_seed=${SEED} \
  critic.ppo_mini_batch_size=8 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.data_loader_seed=${SEED} \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.logger='["console"]' \
  trainer.project_name=polyseg_rl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.val_before_train=false \
  trainer.resume_mode=disable \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  critic.enable=${CRITIC_ENABLE:-null}
)
if [[ "${REWARD_FUNCTION_PATH}" == *"prm_reward.py" ]]; then
  TRAIN_CMD+=(
    +reward.custom_reward_function.reward_kwargs.prm_model_path="${PRM_MODEL_PATH}"
    +reward.custom_reward_function.reward_kwargs.prm_device="${POLYSEG_PRM_DEVICE}"
    +reward.custom_reward_function.reward_kwargs.prm_max_length=${PRM_MAX_LENGTH}
  )
fi
if [[ -n "${POLYSEG_TOTAL_BUDGET}" ]]; then
  TRAIN_CMD+=("actor_rollout_ref.rollout.polyseg_total_budget=${POLYSEG_TOTAL_BUDGET}")
else
  TRAIN_CMD+=("actor_rollout_ref.rollout.polyseg_total_budget=null")
fi
TRAIN_CMD+=("$@")
PYTHONPATH="$ROOT/verl" "${TRAIN_CMD[@]}"
