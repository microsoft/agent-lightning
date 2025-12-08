#!/bin/bash

set -e

# 在你的 训练脚本 开头添加
export UVICORN_LIMIT_CONCURRENCY=300   # 根据需要调整
export UVICORN_BACKLOG=4096
export UVICORN_TIMEOUT_KEEP_ALIVE=60

# 同时建议增加系统文件描述符限制
ulimit -n 65535

export WANDB_KEY="YOUR_WANDB_KEY"
wandb login ${WANDB_KEY}
# ================= Experiment =================
PROJECT_DIR="$(pwd)"
export DATA_ROOT_PATH="${PROJECT_DIR}/datasets/asearcher_data/"
export MODEL_ROOT_PATH="{MODEL_ROOT_PATH}"
export SAVE_ROOT_PATH="${PROJECT_DIR}"
export REWARD_MODEL_URL="REWARD_MODEL_URL(open-ai format)"
export REWARD_MODEL_NAME="REWARD_MODEL_NAME"

# start the agl store service
agl store --port 9999 &
sleep 60
AGL_MANAGED_STORE=0 python3 examples/search_r1_youtu/train_search_agent.py qwen3 --external-store-address http://localhost:9999 --n-runners 2 --youtu 

