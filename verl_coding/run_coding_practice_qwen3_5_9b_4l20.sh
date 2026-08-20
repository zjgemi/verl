#!/usr/bin/env bash

# GRPO | Qwen3.5-9B | multi-turn coding RL | 4x A100/L20 LoRA training
#
# Scientific coding practice RL training with tool agent loop.
# The model learns to solve scientific computing problems by writing Python code
# and executing it in a Docker sandbox environment.
#
# Default config: LoRA rank32/alpha64, Ulysses SP=4, response 32k, n=8, lr=1e-5
# Stable on 4x A100-80GB (peak ~51GB) and 4x L20-48GB with offload.
#
# Usage:
#   bash verl_coding/run_coding_practice_qwen3_5_9b_4l20.sh

set -xeuo pipefail

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}

# Model
MODEL_PATH=${MODEL_PATH:-/trisol/input/model}

# Data
TRAIN_FILE=${TRAIN_FILE:-/trisol/input/datasets/ds-0/train.parquet}
VAL_FILE=${VAL_FILE:-/trisol/input/datasets/ds-0/validation.parquet}

# Tool config for multi-turn agent loop
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-$(python3 -c "import verl_coding, os; print(os.path.join(os.path.dirname(verl_coding.__file__), 'coding_tools.yaml'))")}
IMAGES_PATH=${IMAGES_PATH:-/trisol/input/datasets/ds-0/images.json}
MAX_CONCURRENT_SANDBOXES=${MAX_CONCURRENT_SANDBOXES:-16}

# Python path additions (lbg_sandbox_manager, coding tools, etc.)
EXTRA_PYTHON_DIR=${EXTRA_PYTHON_DIR:-}

# Hardware
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# LoRA
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}

# Batch & sequence sizing
train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-16384}
max_response_length=${MAX_RESPONSE_LENGTH:-32768}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
max_model_len=${MAX_MODEL_LEN:-49152}

# Multi-turn
max_assistant_turns=${MAX_ASSISTANT_TURNS:-10}
max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-2048}

# Optimization
actor_lr=${ACTOR_LR:-1e-5}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

# Ulysses sequence parallel (reduces per-GPU activation peak)
ulysses_sp=${ULYSSES_SP:-4}

# Rollout
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.35}
rollout_n=${ROLLOUT_N:-8}
rollout_load_format=${ROLLOUT_LOAD_FORMAT:-safetensors}
rollout_layered_summon=${ROLLOUT_LAYERED_SUMMON:-True}

# Training
total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}
val_before_train=${VAL_BEFORE_TRAIN:-False}

# Logging
PROJECT_NAME=${PROJECT_NAME:-verl_coding_practice}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_9b_coding_grpo_${INFER_BACKEND}_4l20_$(date +%Y%m%d_%H%M)}

# Agent loop workers
agent_num_workers=${AGENT_NUM_WORKERS:-8}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
actor_param_offload=True
actor_optimizer_offload=False
n_trainer_devices=${NGPUS_PER_NODE}

# Ensure PYTHONPATH includes the project root and extra dependencies
export PYTHONPATH="${PWD}:${EXTRA_PYTHON_DIR}:${PYTHONPATH:-}"

# Generate tool config with current sandbox concurrency limit
_tool_config_orig="${TOOL_CONFIG_PATH}"
_tool_config_tmp=$(mktemp /tmp/coding_tools_XXXXXX.yaml)
sed "s/max_concurrent_sandboxes: [0-9]*/max_concurrent_sandboxes: ${MAX_CONCURRENT_SANDBOXES}/" "${_tool_config_orig}" > "${_tool_config_tmp}"
TOOL_CONFIG_PATH="${_tool_config_tmp}"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    # reward_model is already in the data (style: rule, ground_truth per sample)
    # data_source="coding_practice" routes to coding_reward.compute_score via __init__.py
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    ++actor_rollout_ref.model.lora_rank=${LORA_RANK}
    ++actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    ++actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${ulysses_sp}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    ++actor_rollout_ref.rollout.max_model_len=${max_model_len}
    ++actor_rollout_ref.rollout.max_num_seqs=128
    ++actor_rollout_ref.rollout.load_format=${rollout_load_format}
    ++actor_rollout_ref.rollout.layered_summon=${rollout_layered_summon}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # Multi-turn tool agent configuration
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}"
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns}
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length}
    # Agent loop
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.agent.num_workers=${agent_num_workers}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    ++actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${ulysses_sp}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_trainer_devices}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir=/trisol/output/checkpoints
    ++trainer.val_before_train=${val_before_train}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"