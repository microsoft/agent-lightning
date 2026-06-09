#!/bin/bash
# Install VERL training dependencies into the project .venv.
set -euo pipefail

PYTHON_BIN=".venv/bin/python"
VLLM_VERSION="0.12.0"
VERL_VERSION="0.7.1"
FLASH_ATTN_VERSION="2.8.3"

usage() {
    echo "Usage: scripts/setup_verl.sh <cu129|cu130>"
}

if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

CUDA_VARIANT="$1"
if [ "$CUDA_VARIANT" != "cu129" ] && [ "$CUDA_VARIANT" != "cu130" ]; then
    usage
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: expected Python executable not found: $PYTHON_BIN"
    echo "Run 'uv sync' from the project root first."
    exit 1
fi

echo "Using CUDA wheel variant: $CUDA_VARIANT"

uv pip install --python "$PYTHON_BIN" pip

uv pip install --python "$PYTHON_BIN" \
    "vllm==$VLLM_VERSION" "verl==$VERL_VERSION" \
    --torch-backend="$CUDA_VARIANT" \
    --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/$CUDA_VARIANT" \
    --extra-index-url "https://download.pytorch.org/whl/$CUDA_VARIANT" \
    --index-strategy unsafe-best-match

FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --python "$PYTHON_BIN" \
    "flash-attn==$FLASH_ATTN_VERSION" \
    --force-reinstall \
    --no-cache \
    --no-binary flash-attn \
    --no-build-isolation \
    --no-deps
