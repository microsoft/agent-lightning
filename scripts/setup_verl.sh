#!/bin/bash
# Install VERL training dependencies with a pinned GPU stack.
#
# Default stack:
#   torch==2.9.0, torchvision==0.24.0, torchaudio==2.9.0,
#   vllm, verl, transformers,
#   ray==2.49.2, numpy==1.26.4, flash-attn==2.8.3,
#   xformers==0.0.33.post1, triton==3.5.0.
#
# Run from the repo root:
#
#   scripts/setup_verl.sh              # CUDA 13.0 PyTorch wheels (default)
#   scripts/setup_verl.sh cu128        # force CUDA 12.8 PyTorch wheels
#   scripts/setup_verl.sh cu126        # force CUDA 12.6 PyTorch wheels
#   scripts/setup_verl.sh cpu          # CPU-only PyTorch wheels; skips flash-attn
#
# This intentionally does not use `uv sync --extra verl`, because the resolver
# can select newer torch/vllm versions that are binary-incompatible with an
# already-built flash-attn extension.
set -euo pipefail

# --- Pinned versions ---
PYTHON_BIN=".venv/bin/python"
TORCH_VERSION="2.9.0"
TORCHVISION_VERSION="0.24.0"
TORCHAUDIO_VERSION="2.9.0"
VERL_VERSION="0.7.1"
VLLM_VERSION="0.12.0"
FLASH_ATTN_VERSION="2.8.3"
RAY_VERSION="2.49.2"
TRANSFORMERS_VERSION="4.57.1"
TOKENIZERS_VERSION="0.22.2"
NUMPY_VERSION="1.26.4"
TRITON_VERSION="3.5.0"
ACCELERATE_VERSION="1.13.0"
DATASETS_VERSION="4.8.4"
HYDRA_CORE_VERSION="1.3.2"
OMEGACONF_VERSION="2.3.0"
XFORMERS_VERSION="0.0.33.post1"

# --- CUDA variant ---
VARIANT="${1:-cu130}"
echo "Using PyTorch wheel variant: $VARIANT"

if [ "$VARIANT" = "cpu" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    TORCH_INDEX="https://download.pytorch.org/whl/${VARIANT}"
fi

echo ""
echo "=== Installing VERL deps ==="
echo "  PyTorch index: $TORCH_INDEX"
echo "  torch==$TORCH_VERSION"
echo "  torchvision==$TORCHVISION_VERSION"
echo "  torchaudio==$TORCHAUDIO_VERSION"
echo "  verl==$VERL_VERSION"
echo "  vllm==$VLLM_VERSION"
echo "  transformers==$TRANSFORMERS_VERSION"
echo "  ray==$RAY_VERSION"
echo "  numpy==$NUMPY_VERSION"
echo "  flash-attn==$FLASH_ATTN_VERSION"
echo ""

# Verify the index is reachable
if ! curl -sf "${TORCH_INDEX}/torch/" > /dev/null 2>&1; then
    echo "WARNING: PyTorch index $TORCH_INDEX may not exist."
    echo "  Available indexes: https://download.pytorch.org/whl/"
    echo "  Common: cu126, cu128, cu130, cpu"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=== Syncing base project deps (without [verl] extra) ==="
uv sync --extra controller --extra dev

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: expected Python executable not found: $PYTHON_BIN"
    exit 1
fi

echo ""
echo "=== Installing pip into .venv ==="
uv pip install --python "$PYTHON_BIN" pip

echo ""
echo "=== Installing pinned PyTorch stack ==="
uv pip install --python "$PYTHON_BIN" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX"

echo ""
echo "=== Installing pinned VERL runtime stack ==="
RUNTIME_PACKAGES=(
    "numpy==$NUMPY_VERSION"
    "triton==$TRITON_VERSION"
    "ray==$RAY_VERSION"
    "transformers==$TRANSFORMERS_VERSION"
    "tokenizers==$TOKENIZERS_VERSION"
    "accelerate==$ACCELERATE_VERSION"
    "datasets==$DATASETS_VERSION"
    "hydra-core==$HYDRA_CORE_VERSION"
    "omegaconf==$OMEGACONF_VERSION"
    "verl==$VERL_VERSION"
    "vllm==$VLLM_VERSION"
)
if [ "$VARIANT" != "cpu" ]; then
    RUNTIME_PACKAGES+=("xformers==$XFORMERS_VERSION")
fi
uv pip install --python "$PYTHON_BIN" "${RUNTIME_PACKAGES[@]}"

echo ""
if [ "$VARIANT" = "cpu" ]; then
    echo "=== Skipping flash-attn for CPU-only setup ==="
else
    # Build flash-attn locally against the pinned torch ABI. --no-deps keeps uv
    # from upgrading torch while resolving flash-attn's broad dependency range.
    echo "=== Building and installing flash-attn against final torch ==="
    FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --python "$PYTHON_BIN" \
        "flash-attn==$FLASH_ATTN_VERSION" \
        --force-reinstall \
        --no-cache \
        --no-binary flash-attn \
        --no-build-isolation \
        --no-deps
fi

echo ""
echo "=== Verifying installation ==="
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md

import torch

print(f"  torch={torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
for package in (
    "torchvision",
    "torchaudio",
    "vllm",
    "verl",
    "ray",
    "transformers",
    "tokenizers",
    "numpy",
    "triton",
    "accelerate",
    "datasets",
    "hydra-core",
    "omegaconf",
    "xformers",
    "flash-attn",
):
    try:
        print(f"  {package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"  {package}=not installed")

try:
    import flash_attn_2_cuda  # noqa: F401
except ImportError as exc:
    if md.version("torch").endswith("+cpu"):
        print("  flash_attn_2_cuda=skipped for CPU torch")
    else:
        raise SystemExit(f"flash_attn_2_cuda import failed: {exc}") from exc
else:
    print("  flash_attn_2_cuda=ok")
PY

echo ""
echo "=== Done ==="
