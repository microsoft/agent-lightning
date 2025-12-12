# ChartQA Example

This example demonstrates training a visual reasoning agent on the ChartQA dataset using Agent-Lightning with the VERL algorithm and LangGraph framework. The agent answers questions about charts through a multi-step workflow with self-refinement. It's compatible with Agent-lightning v0.3.0 or later.

## Requirements

This example requires a single node with at least one 40GB GPU. Install dependencies with:

```bash
uv sync --frozen \
    --group dev \
    --group experiment \
    --group image \
    --group langchain \
    --group vllm-0-10-2 \
    --group torch-gpu-stable
```

**Currently vLLM 0.10.2 is the only tested version. You might see issues like `cu_seqlens_q must be on CUDA` or flash-attn installation failures if you use other versions.** (See https://github.com/vllm-project/vllm/issues/27340)

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
| `train_chartqa_agent.py` | Training script using VERL algorithm with configurable hyperparameters (debug, qwen) |
| `debug_chartqa_agent.py` | Debugging script to test the agent with cloud APIs or a local vLLM proxy |
| `prepare_data.py` | Script to download ChartQA dataset from HuggingFace and prepare parquet files |
| `prompts.py` | Prompt templates for the agent workflow |
| `multimodal_utils.py` | Utility functions for encoding images to base64 |
| `data/` | Directory containing images and parquet files after download |

## Running Examples

### Debugging with Cloud API (Default)

For quick testing with OpenAI or other cloud APIs (no local GPU required):

```bash
export OPENAI_API_KEY=<your-api-key>
python debug_chartqa_agent.py
```

For other providers (Azure, etc.), set `OPENAI_API_BASE`:

```bash
export OPENAI_API_BASE=https://your-resource.openai.azure.com/v1
export OPENAI_MODEL=gpt-4o
python debug_chartqa_agent.py
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
    OPENAI_MODEL=Qwen/Qwen2-VL-2B-Instruct \
    python chartqa_agent.py
```

### Training with Local Model

```bash
python train_chartqa_agent.py debug --n-runners 2
```

You can also use an external store server (recommended for distributed setups), first start the store:

```bash
agl store --port 4747
```

Then run the training script with the external store address:

```bash
AGL_MANAGED_STORE=0 python train_chartqa_agent.py fast --external-store-address http://localhost:4747
```

If you want to track experiments with Weights & Biases, set the `WANDB_API_KEY` environment variable before training.

The script automatically launches agent workers and the training server. The agent workers execute chart reasoning rollouts using the vision-language model, while the training server applies the VERL algorithm (GRPO) to improve the model based on answer accuracy rewards.
