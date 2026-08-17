#!/usr/bin/env bash

# GRPO | Qwen3.5-9B | multi-turn coding RL | 4x L20 FSDP training
#
# Scientific coding practice RL training with tool agent loop.
# The model learns to solve scientific computing problems by writing Python code
# and executing it in a Docker sandbox environment.
#
# Hardware: 4x NVIDIA L20 (48GB VRAM each, 192GB total)
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

# Python path additions (lbg_sandbox_manager, coding tools, etc.)
EXTRA_PYTHON_DIR=${EXTRA_PYTHON_DIR:-}

# Hardware: 4x NVIDIA L20
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# Batch & sequence sizing (scaled for 4 GPUs)
train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-16384}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

# Multi-turn
max_assistant_turns=${MAX_ASSISTANT_TURNS:-10}
max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-2048}

# Optimization
actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

# Rollout: TP=2 for 4 GPUs (2 GPUs for inference, 2 for actor sharding)
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.35}
rollout_n=${ROLLOUT_N:-4}

# Training
total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

# Logging
PROJECT_NAME=${PROJECT_NAME:-verl_coding_practice}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_9b_coding_grpo_${INFER_BACKEND}_4l20_$(date +%Y%m%d_%H%M)}

# Agent loop workers
agent_num_workers=${AGENT_NUM_WORKERS:-8}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
# GPU settings for L20 (48GB): no offloading needed
actor_param_offload=True
actor_optimizer_offload=True
n_trainer_devices=${NGPUS_PER_NODE}

# Ensure PYTHONPATH includes the project root and extra dependencies
export PYTHONPATH="${PWD}:${EXTRA_PYTHON_DIR}:${PYTHONPATH:-}"

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
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
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
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    ++actor_rollout_ref.rollout.max_model_len=32768
    ++actor_rollout_ref.rollout.max_num_seqs=128
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