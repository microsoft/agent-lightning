#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

# Runtime launcher script for WebShop agent training on Azure ML.
#
# This script installs dependencies using uv sync (locked) and runs training,
# avoiding the conda overlay issues that cause flash-attn and vLLM failures.
#
# Usage:
#   ./run_webshop_aml.sh [config] [setup_script]
#
# Arguments:
#   config       - Training config: dev|fast|qwen (default: qwen)
#   setup_script - Dependency lane: legacy|stable|latest (default: stable)

set -euo pipefail

CONFIG="${1:-qwen}"
SETUP_SCRIPT="${SETUP_SCRIPT:-${2:-stable}}"

echo "== WebShop AML Job =="
echo "Config: ${CONFIG}"
echo "Setup script: ${SETUP_SCRIPT}"
echo ""

echo "== Diagnostics =="
nvidia-smi || true
python -V || true
which python || true
df -h || true
echo ""

# Recommended caches (avoid re-downloading HF/W&B repeatedly)
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_DIR="${WANDB_DIR:-$PWD/.cache/wandb}"

# Ray/vLLM often benefit from higher fd limits
ulimit -n 65535 || true

# Install build tools needed for vLLM V1 torch.compile/Triton
echo "== Installing build dependencies =="
apt-get update && apt-get install -y --no-install-recommends \
  gcc g++ python3-dev curl wget gnupg ca-certificates || true

# Install Node.js 20 + pnpm for headless runner
echo "== Installing Node.js 20 =="
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y --no-install-recommends nodejs
npm install -g pnpm

# Install Java 21 for pyserini (WebShop search engine)
echo "== Installing Java 21 (Temurin) =="
# Create man page directories (missing in minimal containers, causes dpkg failures)
mkdir -p /usr/share/man/man1
mkdir -p /etc/apt/keyrings
wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg
echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb noble main" > /etc/apt/sources.list.d/adoptium.list
apt-get update && apt-get install -y --no-install-recommends temurin-21-jdk
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
export PATH="${JAVA_HOME}/bin:${PATH}"

# Ensure pip tooling is sane
echo "== Installing uv =="
python -m pip install -U pip
python -m pip install -U uv

# Install dependencies using uv sync with locked versions.
# This mirrors upstream "Examples - Spider" CI approach.
echo "== Installing Python dependencies with uv sync (${SETUP_SCRIPT}) =="
uv sync --frozen --no-default-groups --extra verl \
  --group dev --group experiment --group agents --group "torch-gpu-${SETUP_SCRIPT}"

# Use explicit venv Python path to avoid conda/system Python conflicts
VENV_PYTHON=".venv/bin/python"

echo "Using Python: $VENV_PYTHON"
$VENV_PYTHON --version

# Install WebShop Python dependencies using uv pip (uv doesn't install pip into venv)
echo "== Installing WebShop server dependencies =="
uv pip install --no-cache -r contrib/recipes/webshop/server/requirements.txt || {
    echo "ERROR: Failed to install server requirements"
    exit 1
}
uv pip install --no-cache -r contrib/recipes/webshop/agl/requirements.txt || {
    echo "ERROR: Failed to install agl requirements"
    exit 1
}

# Verify critical packages are installed
echo "== Verifying critical packages =="
$VENV_PYTHON -c "import cleantext; print(f'cleantext version: {cleantext.__version__}')" || {
    echo "ERROR: cleantext not installed, trying explicit install..."
    uv pip install "cleantext>=1.1.4"
}
$VENV_PYTHON -c "import pyserini; print('pyserini OK')" || echo "WARNING: pyserini not available"

# Download spacy model (required by WebShop's web_agent_site)
echo "== Downloading spaCy model en_core_web_sm =="
$VENV_PYTHON -m spacy download en_core_web_sm || {
    echo "WARNING: spacy download failed, trying uv pip install..."
    uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
}

# Verify spacy model is available
$VENV_PYTHON -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model en_core_web_sm loaded successfully')" || {
    echo "ERROR: Failed to load spaCy model en_core_web_sm"
    echo "Attempting fallback install..."
    uv pip install en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
    $VENV_PYTHON -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded after fallback')" || {
        echo "FATAL: Cannot load spaCy model, WebShop will fail"
        exit 1
    }
}

# Activate for subsequent commands (ray, etc.)
source .venv/bin/activate

# Install Node.js dependencies for headless runner
echo "== Installing Node.js dependencies =="
cd contrib/recipes/webshop
pnpm install --frozen-lockfile || pnpm install
cd ../../..

# Start Ray cluster
echo "== Starting Ray cluster =="
if [[ -f "./scripts/restart_ray.sh" ]]; then
  ./scripts/restart_ray.sh
else
  ray stop --force || true
  env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 ray start --head --disable-usage-stats
fi

sleep 5

# Run training
echo "== Starting WebShop training =="
cd contrib/recipes/webshop

# Debug: list scripts directory
echo "Contents of scripts/:"
ls -la scripts/ || echo "scripts/ directory not found!"

# Make script executable and run
if [[ -f "scripts/run_stack.sh" ]]; then
  chmod +x scripts/run_stack.sh
  PYTHONUNBUFFERED=1 bash scripts/run_stack.sh "${CONFIG}"
else
  echo "ERROR: scripts/run_stack.sh not found!"
  echo "Current directory: $(pwd)"
  echo "Directory listing:"
  ls -la
  exit 1
fi
