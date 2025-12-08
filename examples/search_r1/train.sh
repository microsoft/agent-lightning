#!/bin/bash

set -e

export N_GPUS=8
export BASE_MODEL=meta-llama/Llama-3.2-3B
export ROLLOUT_TP_SIZE=1
export DATA_DIR=data
export EXPERIMENT_NAME=searchr1
export PROJECT_NAME=AgentLightning-searchr1
echo "Starting training script..."

python -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    trainer.n_gpus_per_node=${N_GPUS} \
    data.train_batch_size=512 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.truncation='error' \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir=checkpoints/searchr1_checkpoints/$EXPERIMENT_NAME \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300
