# GEPA Example

This example demonstrates [GEPA](https://github.com/gepa-ai/gepa) (Generalized Evolutionary Prompt Adaptation) prompt optimization on a room-booking agent. It supports Azure OpenAI (Entra ID or API key) and plain OpenAI as backends.

## Overview

The room-booking agent selects the best meeting room given constraints (capacity, equipment, accessibility, availability). GEPA optimizes the prompt template through evolutionary search with reflective mutations, tracking a Pareto frontier of per-example performance.

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
| `room_selector.py` | Room booking agent with multi-backend LLM support |
| `room_selector_gepa.py` | GEPA training script that optimizes the agent's prompt template |
| `room_tasks.jsonl` | Dataset with 57 room booking scenarios and expected selections |
| `.env.example` | Template for environment variables (all three providers documented) |

## Smoke Test

Run a single task to verify authentication and connectivity:

```bash
# Default (azure_entra):
python room_selector.py

# Or with a specific provider:
LLM_PROVIDER=openai python room_selector.py
```

## Full Training

Run GEPA optimization over the training dataset:

```bash
python room_selector_gepa.py

# Or with a specific provider:
python room_selector_gepa.py --provider openai
```

GEPA will evaluate prompt candidates, build reflective datasets from execution traces, and propose improved prompts. The best prompt is logged at the end of training.

## W&B Experiment Tracking

Track GEPA optimization progress (scores, budget, candidate acceptance) in [Weights & Biases](https://wandb.ai/):

```bash
pip install wandb
python room_selector_gepa.py --wandb
```

Customize the project and run name:

```bash
python room_selector_gepa.py --wandb --wandb-project my-project --wandb-name run-1
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
