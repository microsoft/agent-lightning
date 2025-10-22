# Unsloth SFT Example

[![unsloth CI status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-unsloth.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-unsloth.yml)

This example demonstrates Supervised Fine-Tuning (SFT) using the Unsloth library for efficient training with 4-bit quantization and LoRA. The example trains a math-solving agent on the GSM-hard dataset. It's compatible with Agent-lightning v0.2 or later.

## Overview

The SFT workflow iteratively improves the model by collecting rollouts, ranking them by reward, and fine-tuning on the top-performing examples. Unsloth optimizes the training process with memory-efficient techniques including 4-bit quantization, LoRA (Low-Rank Adaptation), and gradient checkpointing.

## Requirements

Follow the [installation guide](../../docs/tutorials/installation.md) to install Agent-Lightning, PyTorch and vLLM. You will not need VERL for this example. Additionally, install Unsloth and related packages.

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.10.2
pip install unsloth==2025.10.1 unsloth_zoo==2025.10.1 bitsandbytes peft datasets transformers trl kernels

pip install openai-agents mcp
```

This example requires a GPU with 16GB memory to load models in 4-bit quantization. The training uses LoRA to reduce memory requirements during fine-tuning.

## Dataset

The example uses the GSM-hard dataset from Hugging Face. The dataset contains mathematical reasoning problems with numeric answers. A convenience function is provided in `math_agent.py` to download the first 64 samples for quick experimentation. The samples have already been included in the repository in `data_gsmhard.jsonl`.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `math_agent.py` | Math agent implementation using the OpenAI Agents library and MCP calculator tool |
| `sft_allinone.py` | All-in-one SFT training script that runs the complete workflow |
| `sft_algorithm.py` | Core SFT algorithm implementation with data collection and training logic |
| `sft_rollout_runners.py` | Rollout runner configuration for parallel agent execution |
| `unsloth_helper.py` | Unsloth training utilities with LoRA configuration and model management |
| `data_gsmhard.jsonl` | Local copy of GSM-hard dataset samples (64 samples) |

## Running Examples

### Training

The all-in-one script handles the complete SFT workflow including store management, rollout execution, and model training:

```bash
python sft_allinone.py
```

See [How to Fine-tune with Unsloth](../../docs/how-to/unsloth-sft.md) for more details.

### Training in Separate Processes

The all-in-one script is recommended for most use cases. However, you can also run the algorithm, runners, and store in separate processes if needed:

```bash
# Terminal 1: Start the store
agl store

# Terminal 2: Run the algorithm
python sft_algorithm.py

# Terminal 3: Run the rollout runners
python sft_rollout_runners.py
```

This approach provides more control for debugging and distributed setups but requires manual coordination between processes.

### Debugging

To test the math agent without training:

```bash
python math_agent.py
```

This runs a dry run with a few problems to verify the agent setup. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables to configure the API endpoint.
