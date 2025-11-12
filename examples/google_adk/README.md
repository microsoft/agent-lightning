# ADK Example

This example demonstrates training a Google ADK (Application Development Kit) agent using Agent-Lightning with reinforcement learning. The agent processes user instructions and generates ADK actions for Google applications. It's compatible with Agent-lightning v0.2 or later.

## Requirements

This example requires a single node with at least one 40GB GPU. Follow the [installation guide](../../docs/tutorials/installation.md) to install Agent-Lightning and VERL-related dependencies.

Additionally, install the ADK dependencies:

```bash
pip install "google-adk>=0.3.0" "fastapi" "uvicorn" "pytest"
```

Or use the local `pyproject.toml` in this directory:

```bash
cd examples/google_adk
uv sync
```

## Dataset

Detailed dataset preparation instructions are available in the [How to Train an ADK Agent](../../docs/how-to/train-adk-agent.md) guide.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `adk_agent.py` | ADK agent implementation using the v0.2 LitAgent API, wrapping ADK's agent orchestration capabilities |
| `train_adk.py` | Training script with support for CI modes and external store debugging |
| `prepare_dataset.py` | Dataset conversion utility for generating Parquet files from various formats |
| `adk_debug.py` | Sanity check script for testing the agent without training |

## Running Examples

### Training

Train the ADK agent using the training script:

```bash
cd examples/google_adk
python train_adk.py --train-file data/train.parquet --val-file data/test.parquet
```

The script supports several CLI options:

- `--train-file`: Path to training Parquet file (default: `data/train.parquet`)
- `--val-file`: Path to validation Parquet file (default: `data/test.parquet`)
- `--model`: Model name for the rollout LLM (default: from `OPENAI_MODEL` env var or `meta-llama/Meta-Llama-3-8B-Instruct`)
- `--endpoint`: OpenAI-compatible base URL (default: from `OPENAI_API_BASE` env var or `http://localhost:8000/v1`)
- `--ci`: Reduce workload for CI (smaller dataset slice, fewer runners)
- `--ci-fast`: Ultra-fast CI mode (tiny dataset slice, 1 runner)
- `--external-store-address`: Use an external LightningStore address for debugging (e.g., `http://localhost:4747`)
- `--wandb-project`: Weights & Biases project name (default: `agent-lightning-adk`)
- `--wandb-run-name`: Weights & Biases run name (optional)

**Example with CI mode:**

```bash
python train_adk.py --ci-fast --wandb-project my-project --wandb-run-name test-run
```

**Example with external store for debugging:**

```bash
python train_adk.py --external-store-address http://localhost:4747
```

### Debugging

To test the agent without training, use the sanity check script:

```bash
python adk_debug.py
```

You can customize the test with CLI options:

- `--file`: Path to Parquet file (default: `data/test.parquet`)
- `--index`: Row index to test (default: `0`)
- `--model`: Model name (default: from `OPENAI_MODEL` env var)
- `--endpoint`: API endpoint (default: from `OPENAI_API_BASE` env var or `http://localhost:8000/v1`)

**Example:**

```bash
python adk_debug.py --file data/train.parquet --index 0 --endpoint http://localhost:8000/v1 --model meta-llama/Meta-Llama-3-8B-Instruct
```

This runs a single rollout and prints the reward, helping you verify that the agent configuration, ADK orchestration, and dataset are correct before starting a full training run.

## Configuration

The agent expects an OpenAI-compatible API service. Configure your service endpoint and credentials using environment variables:

- `OPENAI_API_BASE`: Base URL for the API service
- `OPENAI_API_KEY`: API key (if required)
- `OPENAI_MODEL`: Default model name to use

If you're using vLLM locally, you typically don't need an API key:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

## Agent Implementation Details

The `LitAdkAgent` class implements the v0.2 LitAgent API:

- **Task Type**: `AdkTask` with `question`, `app_id`, `ground_truth`, and optional `meta` fields
- **LLM Resource**: Extracted from `resources["main_llm"]` (VERL convention)
- **Reward**: Returns a float reward based on action matching the ground truth (customize as needed)
- **Orchestration**: Wraps ADK's agent orchestration capabilities, similar to how `LitSQLAgent` wraps LangGraph workflows
- **Observability**: Leverages ADK's Cloud Trace integration for span visualization, showing how operations (LLM calls, tool executions) are organized within rollouts

ADK provides observability features that automatically capture spans for each operation, organizing them for visualization. This makes it easy to understand agent execution patterns, identify optimization opportunities, and debug issues. The spans can be organized as sequential, parallel, or nested structures, providing rich insights into agent behavior.

For a real integration, replace the placeholder action generation with actual ADK agent orchestration calls in the `rollout` method, enabling Cloud Trace integration for comprehensive observability.
