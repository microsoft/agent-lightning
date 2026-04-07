#!/bin/bash
# Install VERL training dependencies with GPU support.
#
# Auto-detects CUDA version and uses the matching PyTorch wheel index.
# Run from the repo root:
#
#   scripts/setup_verl.sh              # auto-detect CUDA
#   scripts/setup_verl.sh cu126        # force CUDA 12.6
#   scripts/setup_verl.sh cpu          # CPU-only (no GPU)
#
# This installs the [verl] optional dependency group from pyproject.toml
# with the correct PyTorch CUDA wheels.
set -euo pipefail

# --- Detect or accept CUDA variant ---
if [ -n "${1:-}" ]; then
    VARIANT="$1"
    echo "Using specified variant: $VARIANT"
else
    # Auto-detect from nvcc
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        # Convert "13.1" → "cu130" (major + minor, drop patch)
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
        VARIANT="cu${CUDA_MAJOR}${CUDA_MINOR}0"
        # Check if this index actually exists; fall back to nearest known
        echo "Detected CUDA $CUDA_VERSION → trying PyTorch index: $VARIANT"
    else
        echo "WARNING: nvcc not found. Falling back to CPU-only PyTorch."
        echo "  For GPU support, install CUDA toolkit or specify variant:"
        echo "    $0 cu130"
        VARIANT="cpu"
    fi
fi

if [ "$VARIANT" = "cpu" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    TORCH_INDEX="https://download.pytorch.org/whl/${VARIANT}"
fi

echo ""
echo "=== Installing VERL deps ==="
echo "  PyTorch index: $TORCH_INDEX"
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

uv sync --extra verl --index "pytorch=${TORCH_INDEX}"

echo ""
echo "=== Verifying installation ==="
uv run python -c "
import torch
print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}')
import vllm; print(f'  vllm={vllm.__version__}')
import verl; print(f'  verl={verl.__version__}')
import ray; print(f'  ray={ray.__version__}')
import transformers; print(f'  transformers={transformers.__version__}')
"

echo ""
echo "=== Done ==="
