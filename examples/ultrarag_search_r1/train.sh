#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${AGL_ROOT}/.venv/bin/python" ]; then
    PYTHON="${AGL_ROOT}/.venv/bin/python"
    echo "Using uv virtual environment: ${PYTHON}"
else
    PYTHON="python"
    echo "Warning: uv virtual environment not found at ${AGL_ROOT}/.venv/bin/python"
    echo "Using system python. Make sure all dependencies are installed."
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3,4,5}
export N_GPUS=${N_GPUS:-2}
export VLLM_PORT=${VLLM_PORT:-8001}
export VLLM_HOST=${VLLM_HOST:-127.0.0.1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export BASE_MODEL=${BASE_MODEL:-/path/to/llama-3.2-3b-instruct}
export DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/../search_r1/data}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ultrarag_search_r1}
export PROJECT_NAME=${PROJECT_NAME:-AgentLightning-ultrarag}

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $N_GPUS"
echo "Data dir: $DATA_DIR"
echo "Base model: $BASE_MODEL"

cd "${SCRIPT_DIR}"
PYTHONPATH="${SCRIPT_DIR}" ${PYTHON} -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test_100.parquet \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.n_gpus_per_node=${N_GPUS} \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.truncation='error' \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir=checkpoints/ultrarag_search_r1_checkpoints/$EXPERIMENT_NAME \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300
