# OpenAI Client Example

This is a minimal example demonstrating how to use the OpenAI client to query an LLM endpoint and train a model with reinforcement learning using `verl`.

The dataset used is **GSM8K**, and the model is **Qwen2.5-1.5B-Instruct**.
The script can be run on a single **A100 80GB GPU**.


## Quick Start

First, start a Ray cluster with the following command.
Replace `XXXXX` with your own Weights & Biases (wandb) API key.

```bash
ray stop
env WANDB_API_KEY=XXXXX RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0
```

Then start the training:

```bash
python train.py
```

All LLM queries made by `gsm8k_agent` will be automatically recorded and used for training with the emitted rewards.
