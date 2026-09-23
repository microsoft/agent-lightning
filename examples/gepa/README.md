# GEPA Example

This example demonstrates [GEPA](https://github.com/gepa-ai/gepa) (Generalized Evolutionary Prompt Adaptation) prompt optimization on a HotPotQA question-answering agent. It supports Azure OpenAI (Entra ID or API key) and plain OpenAI as backends.

## Overview

The HotPotQA agent answers factoid questions from the [HotPotQA](https://hotpotqa.github.io/) dataset (loaded via DSPy). GEPA optimizes the prompt template through evolutionary search with reflective mutations, tracking a Pareto frontier of per-example performance. An optional autoresearch outer loop (`gepa_autoresearch.py`) searches over GEPA hyperparameters themselves.

## Choosing an LLM Backend

Set the backend via `LLM_PROVIDER` env var or `--provider` CLI arg:

| Provider | Auth method | Extra dependency | Key env vars |
|----------|-------------|------------------|--------------|
| `azure_entra` (default) | Entra ID / `az login` | `azure-identity` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |
| `azure_key` | API key | — | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` |
| `openai` | OpenAI API key | — | `OPENAI_API_KEY` |

## Requirements

1. Install Agent-Lightning with GEPA extras:

   ```bash
   uv sync --extra gepa
   ```

2. **(Azure Entra ID only)** Install the Azure Identity library and authenticate:

   ```bash
   uv pip install azure-identity
   az login
   ```

3. Copy `.env.example` to `.env` and fill in the variables for your chosen provider. Export them before running:

   ```bash
   export $(grep -v '^#' .env | xargs)
   ```

## Included Files

| File | Description |
|------|-------------|
| `llm_backend.py` | Centralized LLM provider logic (Azure Entra ID, Azure API key, OpenAI) |
| `hotpotqa_agent.py` | HotPotQA question-answering agent with multi-backend LLM support |
| `hotpotqa_gepa.py` | GEPA training script that optimizes the agent's prompt template |
| `gepa_autoresearch.py` | Autoresearch-style outer loop that searches over GEPA hyperparameters |
| `.env.example` | Template for environment variables (all three providers documented) |

## Smoke Test

Run a single task to verify authentication and connectivity:

```bash
# Default (azure_entra):
python hotpotqa_agent.py

# Or with a specific provider:
LLM_PROVIDER=openai python hotpotqa_agent.py
```

## Full Training

Run GEPA optimization over the training dataset:

```bash
python hotpotqa_gepa.py

# Or with a specific provider:
python hotpotqa_gepa.py --provider openai
```

GEPA will evaluate prompt candidates, build reflective datasets from execution traces, and propose improved prompts. The best prompt is logged at the end of training.

## Autoresearch (Hyperparameter Search)

Run the autoresearch outer loop to search over GEPA configurations:

```bash
python gepa_autoresearch.py

# With LLM-guided proposals:
python gepa_autoresearch.py --proposal-policy llm --iterations 8

# With random search:
python gepa_autoresearch.py --proposal-policy random --iterations 16
```

Each trial runs a full GEPA experiment and evaluates the learned prompt on a held-out split. The best configuration and prompt are saved to the run directory.

## W&B Experiment Tracking

Track GEPA optimization progress (scores, budget, candidate acceptance) in [Weights & Biases](https://wandb.ai/):

```bash
pip install wandb
python hotpotqa_gepa.py --wandb
```

Customize the project and run name:

```bash
python hotpotqa_gepa.py --wandb --wandb-project my-project --wandb-name run-1
```

Set `WANDB_API_KEY` or run `wandb login` before launching.

## GEPA vs APO

| Aspect | APO (`examples/apo/`) | GEPA (`examples/gepa/`) |
|--------|----------------------|------------------------|
| Algorithm | Beam search with gradient-based proposals | Evolutionary search with reflective mutations |
| Selection strategy | Single best | Pareto frontier (per-example tracking) |
| Adapter | Requires `TraceToMessages` adapter | Built-in adapter (no explicit adapter needed) |
| Reflection model | Uses `AsyncOpenAI` client directly | Uses `litellm.completion()` via model string |
| Auth in this example | OpenAI API key | Multi-backend (Azure Entra ID, Azure key, OpenAI) |
| Budget control | Beam rounds / beam width | `max_metric_calls` budget |
