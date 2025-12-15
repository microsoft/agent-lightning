#!/bin/bash
# 一键启动 Spider 训练（Qwen3-4B-Instruct 4×24GB 配置）
set -e

ROOT="/home/lthpc/student/LiTengfei/project/myfork/agent-lightning"
SPIDER_DIR="$ROOT/examples/spider"
cd "$SPIDER_DIR"

# 固定使用打好补丁的 venv Python
PYTHON_BIN="/home/lthpc/student/LiTengfei/env/light/bin/python"

# Python 搜索当前源码
export PYTHONPATH="$ROOT:$PYTHONPATH"

# Ray 临时目录（可用大空间磁盘）
mkdir -p /home/lthpc/raytmp
export RAY_TMPDIR=/home/lthpc/raytmp

# Spider 数据目录（绝对路径）
export VERL_SPIDER_DATA_DIR="$SPIDER_DIR/data"

# Lightning store 端口（并行跑多个实例时改成不同端口，比如 AGL_SERVER_PORT=4748）
export AGL_SERVER_PORT="${AGL_SERVER_PORT:-4748}"

# 可选：关闭 wandb 需设 WANDB_DISABLED=true
# export WANDB_DISABLED=true

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" train_sql_agent.py local_qwen05 --config-file configs/singleGPU_qwen05b.json
