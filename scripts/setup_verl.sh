#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.

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

# flash-attn is built from source against torch's CUDA runtime. For cu130 the system
# CUDA toolkit (often 12.x) is too old, so install a matching CUDA 13.0 pip toolchain
# and point the build at it. Versions are pinned to 13.0.x so nvcc's version matches
# torch's CUDART (13000); a mismatched minor (e.g. 13.3) trips cccl's compatibility
# check ("CUDA compiler and CUDA toolkit headers are incompatible").
if [ "$CUDA_VARIANT" = "cu130" ]; then
    uv pip install --python "$PYTHON_BIN" \
        "nvidia-cuda-nvcc>=13.0,<13.1" \
        "nvidia-cuda-crt>=13.0,<13.1" \
        "nvidia-nvvm>=13.0,<13.1" \
        "nvidia-cuda-cccl>=13.0,<13.1" \
        "nvidia-cuda-runtime>=13.0,<13.1"

    CUDA_HOME="$("$PYTHON_BIN" -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
    ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0}"
fi

FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --python "$PYTHON_BIN" \
    "flash-attn==$FLASH_ATTN_VERSION" \
    --force-reinstall \
    --no-cache \
    --no-binary flash-attn \
    --no-build-isolation \
    --no-deps
