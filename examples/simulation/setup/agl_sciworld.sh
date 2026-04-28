#!/usr/bin/env bash
set -e  # Stop the script if any command fails

# ---------- Create and Setup Environment ----------
echo "=== Creating agl-sciworld conda environment ==="
conda create -n agl-sciworld python=3.10 -y

# ---------- Install Core Dependencies (AGL) ----------
echo "=== Installing AGL dependencies ==="
conda run -n agl-sciworld pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# conda run -n agl-sciworld pip install flash_attn==2.8.0.post2
conda run -n agl-sciworld pip install transformers==4.56.1
conda run -n agl-sciworld pip install wandb
conda run -n agl-sciworld pip install vllm==0.10.2
conda run -n agl-sciworld pip install verl==0.5.0
conda run -n agl-sciworld pip install click==8.2.1
# conda run -n agl-sciworld pip install flash_attn==2.8.3
conda run -n agl-sciworld pip install --extra-index-url https://miropsota.github.io/torch_packages_builder flash_attn==2.8.3+pt2.8.0cu126
conda run -n agl-sciworld pip install -e .[dev]

# ---------- Install ScienceWorld Dependencies ----------
echo "=== Installing ScienceWorld dependencies ==="
sudo apt-get update
sudo apt install -y openjdk-18-jdk

conda run -n agl-sciworld pip install scienceworld omegaconf numpy gym gymnasium sentence_transformers
conda run -n agl-sciworld pip install 'openai-agents[litellm]'==0.2.9
conda run -n agl-sciworld pip install -U "autogen-agentchat" "autogen-ext[openai]"

echo "✅ agl-sciworld environment has been successfully set up!"
