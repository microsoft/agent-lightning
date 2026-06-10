#!/bin/bash
# Install VERL training dependencies into a managed Python environment.
set -euo pipefail

FLASH_ATTN_VERSION="2.8.3"

usage() {
    echo "Usage: bash scripts/setup_verl.sh <0.7.1|0.8.0> <cu129|cu130> [venv_path]"
}

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage
    exit 1
fi

VERL_VERSION="$1"
CUDA_VARIANT="$2"
VENV_PATH="${3:-.venv}"
PYTHON_BIN="$VENV_PATH/bin/python"

if [ "$VERL_VERSION" != "0.7.1" ] && [ "$VERL_VERSION" != "0.8.0" ]; then
    usage
    exit 1
fi

if [ "$CUDA_VARIANT" != "cu129" ] && [ "$CUDA_VARIANT" != "cu130" ]; then
    usage
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: expected Python executable not found: $PYTHON_BIN"
    echo "Run 'uv sync' from the project root first, or pass a venv path."
    exit 1
fi

if [ "$VERL_VERSION" = "0.7.1" ]; then
    VLLM_VERSION="0.12.0"
else
    VLLM_VERSION="0.20.2"
fi

echo "Using Python executable: $PYTHON_BIN"
echo "Using VERL version: $VERL_VERSION"
echo "Using CUDA wheel variant: $CUDA_VARIANT"

uv pip install --python "$PYTHON_BIN" pip

if [ "$VERL_VERSION" = "0.7.1" ]; then
    uv pip install --python "$PYTHON_BIN" \
        "vllm==$VLLM_VERSION" "verl==$VERL_VERSION" \
        --torch-backend="$CUDA_VARIANT" \
        --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/$CUDA_VARIANT" \
        --extra-index-url "https://download.pytorch.org/whl/$CUDA_VARIANT" \
        --index-strategy unsafe-best-match
else
    uv pip install --python "$PYTHON_BIN" \
        "vllm==$VLLM_VERSION" \
        --torch-backend="$CUDA_VARIANT" \
        --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/$CUDA_VARIANT" \
        --extra-index-url "https://download.pytorch.org/whl/$CUDA_VARIANT" \
        --index-strategy unsafe-best-match

    uv pip install --python "$PYTHON_BIN" "verl==$VERL_VERSION"
fi

FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --python "$PYTHON_BIN" \
    "flash-attn==$FLASH_ATTN_VERSION" \
    --force-reinstall \
    --no-cache \
    --no-binary flash-attn \
    --no-build-isolation \
    --no-deps
