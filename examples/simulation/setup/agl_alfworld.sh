#!/usr/bin/env bash
set -e  # Stop the script if any command fails

# ---------- Create and Setup Environment ----------
echo "=== Creating agl-alfworld conda environment ==="
conda create -n agl-alfworld python=3.10 -y

# ---------- Install Core Dependencies (AGL) ----------
echo "=== Installing AGL dependencies ==="
conda run -n agl-alfworld pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# conda run -n agl-alfworld pip install flash_attn==2.8.0.post2
conda run -n agl-alfworld pip install transformers==4.56.1
conda run -n agl-alfworld pip install wandb
conda run -n agl-alfworld pip install vllm==0.10.2
conda run -n agl-alfworld pip install verl==0.5.0
conda run -n agl-alfworld pip install click==8.2.1
# conda run -n agl-alfworld pip install flash_attn==2.8.3
conda run -n agl-alfworld pip install --extra-index-url https://miropsota.github.io/torch_packages_builder flash_attn==2.8.3+pt2.8.0cu126
conda run -n agl-alfworld pip install -e .[dev]

# ---------- Install ALFWorld Dependencies ----------
echo "=== Installing ALFWorld dependencies ==="
conda run -n agl-alfworld pip install gymnasium==0.29.1 stable-baselines3==2.6.0
conda run -n agl-alfworld pip install alfworld pandas pyarrow omegaconf
conda run -n agl-alfworld pip install 'openai-agents[litellm]'==0.2.9
conda run -n agl-alfworld pip install -U "autogen-agentchat" "autogen-ext[openai]"

# ---------- Download ALFWorld Source ----------
echo "=== Downloading ALFWorld source ==="
conda run -n agl-alfworld python examples/simulation/envs/alfworld/download_alfworld_source.py

echo "✅ agl-alfworld environment has been successfully set up!"
