#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES="${SC_GPU:-0}"

exec .venv-int8-int4/bin/vllm serve \
    "${SC_INT8_MODEL:-outputs/precision_validation/Qwen3.5-9B-W8A8-INT8}" \
    --served-model-name Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --quantization compressed-tensors \
    --kv-cache-dtype int4_per_token_head \
    --attention-backend TRITON_ATTN \
    --linear-backend triton \
    --language-model-only \
    --max-model-len 16384 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 1024 \
    --max-logprobs 128 \
    --gpu-memory-utilization 0.35 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --port "${SC_PORT:-8000}" "$@"
