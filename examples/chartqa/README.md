# ChartQA Example

This example demonstrates training a visual reasoning agent on the ChartQA dataset using Agent-Lightning with the VERL algorithm and LangGraph framework. The agent answers questions about charts through a multi-step workflow with self-refinement. It's compatible with Agent-lightning v0.2 or later.

## Requirements

This example requires a single node with at least two 40GB GPU. Install dependencies with:

```bash
uv sync --frozen \
    --group dev \
    --group torch-gpu-stable \
    --group chartqa \
    --no-default-groups
```

See the [installation guide](../../docs/tutorials/installation.md) for more details on dependency groups.

## Dataset

Download the ChartQA dataset and prepare it for training:

```bash
cd examples/chartqa
python prepare_data.py
```

This downloads the ChartQA dataset from HuggingFace (`HuggingFaceM4/ChartQA`), saves images locally, and creates parquet files for training/testing. No HuggingFace token is required (public dataset).

**Dataset Statistics:**
- Training: ~18,000 chart question-answer pairs
- Test: ~2,500 pairs
- Chart types: Bar, line, pie, scatter, etc.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `chartqa_agent.py` | Chart reasoning agent using LangGraph with multi-step workflow (observe → extract → calculate → check → refine) |
| `train_chartqa_agent.py` | Training script using VERL algorithm with configurable hyperparameters (fast, qwen) |
| `prepare_data.py` | Script to download ChartQA dataset from HuggingFace and prepare parquet files |
| `prompts.py` | Prompt templates for the agent workflow |
| `data/` | Directory containing images and parquet files after download |

## Running Examples

### Debugging with Cloud API (Default)

For quick testing with OpenAI or other cloud APIs (no local GPU required):

```bash
export OPENAI_API_KEY=<your-api-key>
export MODEL=gpt-4o  # or other vision-capable model
python chartqa_agent.py
```

For other providers (Azure, etc.), set `OPENAI_API_BASE`:

```bash
export OPENAI_API_BASE=https://your-resource.openai.azure.com/v1
export MODEL=gpt-4o
python chartqa_agent.py
```

### Debugging with Local Model (LLMProxy)

To test the agent with a local vLLM server and LLMProxy:

```bash
# Start a vLLM server (specify image path for VLM)
export CHARTQA_DATA_DIR=<path to chartqa data>
vllm serve Qwen/Qwen2-VL-2B-Instruct \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --allowed-local-media-path $CHARTQA_DATA_DIR \
    --enable-prefix-caching \
    --port 8088

# Run the agent with LLMProxy
USE_LLM_PROXY=1 \
    OPENAI_API_BASE=http://localhost:8088/v1 \
    MODEL=Qwen/Qwen2-VL-2B-Instruct \
    python chartqa_agent.py
```

### Training with Local Model

Run the training script with VERL reinforcement learning:

```bash
# Fast training for testing (2 steps)
python train_chartqa_agent.py fast

# Standard Qwen2-VL-2B training
python train_chartqa_agent.py qwen
```

If you want to track experiments with Weights & Biases, set the `WANDB_API_KEY` environment variable before training.

The script automatically launches agent workers and the training server. The agent workers execute chart reasoning rollouts using the vision-language model, while the training server applies the VERL algorithm (GRPO) to improve the model based on answer accuracy rewards.
