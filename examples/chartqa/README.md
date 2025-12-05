# ChartQA Example

[![chartqa CI status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-chartqa.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-chartqa.yml)

This example demonstrates training a visual reasoning agent on the ChartQA dataset using Agent-Lightning with the VERL algorithm and LangGraph framework. The agent answers questions about charts through a multi-step workflow with self-refinement. It's compatible with Agent-lightning v0.2 or later.

## Requirements

This example requires a single node with at least one 40GB GPU. Follow the [installation guide](../../docs/tutorials/installation.md) to install Agent-Lightning and VERL-related dependencies.

Additionally, install the vision-language model dependencies:

```bash
uv pip install datasets pillow pandas pyarrow
uv pip install "langgraph<1.0" "langchain[openai]<1.0" "langchain-community"
```

## Dataset

Download the ChartQA dataset and prepare it for training:

```bash
cd examples/chartqa
./download_chartqa.sh
```

This downloads the ChartQA dataset from HuggingFace (`HuggingFaceM4/ChartQA`), saves images locally, and creates parquet files for training/testing. No HuggingFace token is required (public dataset).

**Dataset Statistics:**
- Training: ~18,000 chart question-answer pairs
- Test: ~2,000 pairs
- Chart types: Bar, line, pie, scatter, etc.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `chartqa_agent.py` | Chart reasoning agent using LangGraph with multi-step workflow (observe → extract → calculate → check → refine) |
| `train_chartqa_agent.py` | Training script using VERL algorithm with configurable hyperparameters (fast, qwen) |
| `download_chartqa.sh` | Script to download the ChartQA dataset from HuggingFace |
| `prepare_data.py` | Script to prepare parquet files from downloaded images |
| `prompts.py` | Prompt templates for the agent workflow |
| `data/` | Directory containing images and parquet files after download |

## Running Examples

### Debugging with OpenAI API

For quick testing with OpenAI API (no local GPU required):

```bash
export OPENAI_API_KEY=your-api-key
USE_OPENAI=1 OPENAI_MODEL=gpt-4o uv run python chartqa_agent.py
```

### Debugging with Local Model

To test the agent with a local vLLM server:

```bash
# Run the agent on test samples
export OPENAI_API_BASE=http://localhost:8088/v1
export MODEL=Qwen/Qwen2-VL-2B-Instruct
export CHARTQA_DATA_DIR=<path to chartqa data>

# Start a vLLM server (specify image path for VLM)
vllm serve Qwen/Qwen2-VL-2B-Instruct \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --allowed-local-media-path $CHARTQA_DATA_DIR \
    --enable-prefix-caching \
    --port 8088

uv run python chartqa_agent.py
```

### Training with Local Model

Run the training script with VERL reinforcement learning:

```bash
# Fast training (CI/testing, reduced epochs)
uv run python train_chartqa_agent.py fast

# Standard Qwen2-VL-2B training (2 epochs)
uv run python train_chartqa_agent.py qwen
```

If you want to track experiments with Weights & Biases, set the `WANDB_API_KEY` environment variable before training.

The script automatically launches agent workers and the training server. The agent workers execute chart reasoning rollouts using the vision-language model, while the training server applies the VERL algorithm (GRPO) to improve the model based on answer accuracy rewards.


