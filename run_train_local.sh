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

# 配置路径（直接在此处硬编码，修改更方便）
CONFIG_PATH="/home/lthpc/student/LiTengfei/project/myfork/agent-lightning/examples/spider/configs/singleGPU_qwen05b.json"
CONFIG_STEM="$(basename "$CONFIG_PATH" .json)"

# 日志文件
LOG_DIR="$ROOT/examples/spider/log"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${CONFIG_STEM}_${TS}.log"

echo "========================================" > "$LOG_FILE"
echo "Training started at: $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
echo "Hostname: $(hostname)" >> "$LOG_FILE"
echo "User: $(whoami)" >> "$LOG_FILE"
echo "CWD: $(pwd)" >> "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" >> "$LOG_FILE"
echo "Python: $($PYTHON_BIN --version 2>&1)" >> "$LOG_FILE"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')" >> "$LOG_FILE"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'n/a')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

CUDA_VISIBLE_DEVICES=2 nohup "$PYTHON_BIN" train_sql_agent.py local_qwen05 --config-file "$CONFIG_PATH" \
  >> "$LOG_FILE" 2>&1 &

echo "Training started with PID $!"
echo "Config: $CONFIG_PATH"
echo "Log: $LOG_FILE"
