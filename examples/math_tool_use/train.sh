#!/bin/bash

# Copyright (c) Microsoft. All rights reserved.

# Training Script for Math Reasoning Agent
# 
# This script configures and starts the GRPO training server that optimizes
# the math agent through reinforcement learning.
#
# The server:
# 1. Receives trajectories from agent workers
# 2. Computes advantages using GRPO algorithm
# 3. Updates the policy model
# 4. Serves the updated model to workers
#
# Usage:
#   bash train.sh                    # Use default settings
#   bash train.sh trainer.total_epochs=5  # Override specific parameters

set -e

# ==============================================================================
# Configuration
# ==============================================================================

# Model settings
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export N_GPUS=1
export ROLLOUT_TP_SIZE=1

# Data settings
export DATA_DIR=data
export TRAIN_FILE=${DATA_DIR}/gsm8k_train.parquet
export TEST_FILE=${DATA_DIR}/gsm8k_test.parquet

# Experiment tracking
export EXPERIMENT_NAME=math_agent_gsm8k
export PROJECT_NAME=AgentLightning

# ==============================================================================
# Pre-flight checks
# ==============================================================================

echo "=================================="
echo "Math Agent Training Configuration"
echo "=================================="
echo ""
echo "Model: ${BASE_MODEL}"
echo "GPUs: ${N_GPUS}"
echo "Train data: ${TRAIN_FILE}"
echo "Test data: ${TEST_FILE}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo ""

# Check if data files exist
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Error: Training file not found: ${TRAIN_FILE}"
    echo "Please run: python prepare_data.py"
    exit 1
fi

if [ ! -f "${TEST_FILE}" ]; then
    echo "Error: Test file not found: ${TEST_FILE}"
    echo "Please run: python prepare_data.py"
    exit 1
fi

echo "✓ Data files found"
echo ""

# Check if Ray is running
if ! ray status &> /dev/null; then
    echo "Warning: Ray cluster not detected"
    echo "Please start Ray: bash ../../scripts/restart_ray.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting training server..."
echo ""

# ==============================================================================
# Launch Training
# ==============================================================================

python -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    \
    `# GPU Configuration` \
    trainer.n_gpus_per_node=${N_GPUS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    \
    `# Batch Sizes` \
    data.train_batch_size=64 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    `# Sequence Lengths` \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.truncation='error' \
    \
    `# Multi-turn Settings` \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    \
    `# Optimization Settings` \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.use_kl_in_reward=False \
    \
    `# Memory Optimization` \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    \
    `# Rollout Settings` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    \
    `# Training Schedule` \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    \
    `# Logging` \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    \
    `# Infrastructure` \
    trainer.nnodes=1 \
    \
    `# Allow parameter overrides from command line` \
    $@

# ==============================================================================
# Post-training
# ==============================================================================

echo ""
echo "=================================="
echo "Training Complete"
echo "=================================="
echo ""
echo "Checkpoints saved to: checkpoints/${EXPERIMENT_NAME}/"
echo ""
echo "To continue training:"
echo "  1. Update BASE_MODEL to point to a checkpoint"
echo "  2. Run: bash train.sh trainer.total_epochs=5"
echo ""
echo "To evaluate a checkpoint:"
echo "  1. Load the checkpoint in math_agent.py"
echo "  2. Run evaluation on the test set"