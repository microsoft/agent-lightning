#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

set -euo pipefail

command -v vllm >/dev/null 2>&1 || {
  echo "vllm is not installed in the active environment." >&2
  exit 2
}

model="${SHAPER_MODEL:-Qwen/Qwen3.6-27B}"
host="${SHAPER_VLLM_HOST:-0.0.0.0}"
port="${SHAPER_VLLM_PORT:-8001}"
tp_size="${SHAPER_VLLM_TP_SIZE:-8}"
max_model_len="${SHAPER_VLLM_MAX_MODEL_LEN:-262144}"
reasoning_parser="${SHAPER_VLLM_REASONING_PARSER:-qwen3}"

args=(
  serve "$model"
  --host "$host"
  --port "$port"
  --served-model-name "$model"
  --tensor-parallel-size "$tp_size"
  --max-model-len "$max_model_len"
)
if [[ -n "$reasoning_parser" ]]; then
  args+=(--reasoning-parser "$reasoning_parser")
fi

exec vllm "${args[@]}" "$@"
