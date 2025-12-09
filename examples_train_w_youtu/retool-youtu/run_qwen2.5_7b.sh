#!/bin/bash

set -ex
export VLLM_USE_V1=1
export UVICORN_LIMIT_CONCURRENCY=300   # 根据需要调整
export UVICORN_BACKLOG=4096
export UVICORN_TIMEOUT_KEEP_ALIVE=60
ulimit -n 65535
# ================= Conda Environment =================
rm -rf ~/.cache/huggingface/datasets
source ~/anaconda3/bin/activate youtu
# ================= Model Configuration =================
export BASE_MODEL="YOUR_MODEL_PATH"
echo "Model Path: $BASE_MODEL"
# ================= GPUs =================
NNODES=4
GPUS_PER_NODE=8
infer_tp=1  # vllm tensor parallel
train_sp=1  # train sequence parallel
N_RUNNERS=128

# ================= Experiment =================
PROJECT_DIR="$(pwd)"
experiment_name=qwen2.5-7b_dapo
project_name=agent_lightning_retool
default_local_dir="${PROJECT_DIR}/checkpoints/$project_name/$experiment_name"
mkdir -p ${default_local_dir}

# ================= Test Sandbox Fusion Tool =================
AGENT_LOG_DIR=${default_local_dir}/logs
cd ${PROJECT_DIR}/examples_train_w_youtu/retool_youtu
mkdir -p ${AGENT_LOG_DIR}
echo "Start AGL STORE"
nohup agl store --port 9999 > $AGENT_LOG_DIR/${experiment_name}_store_output.log 2>&1 &
sleep 5
echo "Start AGL RUNNER"
nohup env AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=runner python train_calc_sandbox_agent.py --external-store-address http://localhost:9999 --n-runners $N_RUNNERS > $AGENT_LOG_DIR/${MASTER_ADDR}_output.log 2>&1 &
cd ../../
# ================= Tokens and Login =================
HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
export WANDB_KEY="YOUR_WANDB_TOKEN"
wandb login ${WANDB_KEY}
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
# ================= Data Configuration =================
# Custom dataset class
CUSTOM_DATASET_PATH="${PROJECT_DIR}/examples_train_w_youtu/retool_youtu/retool.py"
CUSTOM_DATASET_NAME="CustomRLHFDataset"

# Training and validation data paths
dapo_math_17k=${PROJECT_DIR}/datasets/BytedTsinghua-SIA/DAPO-Math-17k
aime_2024=${PROJECT_DIR}/datasets/Maxwell-Jia/AIME_2024
aime_2025=${PROJECT_DIR}/datasets/yentinglin/aime_2025

TRAIN_FILE="$dapo_math_17k"
VAL_FILE="$aime_2024"

# ================= Algorithm Parameters =================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# ================= Data Parameters =================
max_turns=8
max_prompt_length=14336
max_response_length=2048
max_num_batched_tokens=$((max_prompt_length + max_response_length))
train_batch_size=128
ppo_mini_batch_size=256
n_resp_per_prompt=8
n_resp_per_prompt_val=30

# ================= Training Parameters =================
actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=1.0
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.6
VAL_N=30

# ================= Performance =================
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))
balance_batch=False
# ================= Trainer configs =================
val_before_train=False

cd ${PROJECT_DIR}/examples_train_w_youtu/retool_youtu
# ================= Run Training with Config Overrides =================
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=algorithm python train_calc_sandbox_agent.py \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --model "$BASE_MODEL" \
    --val-temperature $VAL_TEMPERATURE \
    --val-top-p $VAL_TOP_P \
    --val-n $VAL_N \
    --n-runners $N_RUNNERS \
    --custom-dataset-path "$CUSTOM_DATASET_PATH" \
    --custom-dataset-name "$CUSTOM_DATASET_NAME" \
    --external-store-address http://localhost:9999 \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=$offload \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="./sandbox_fusion_tool_config.yaml" \
    trainer.balance_batch=$balance_batch \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.val_before_train=$val_before_train \
    trainer.val_only=False \
    trainer.log_val_generations=4 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=5 \
    trainer.default_local_dir=$default_local_dir \
    trainer.rollout_data_dir=$default_local_dir/rollout \
    trainer.validation_data_dir=$default_local_dir/validation \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500

