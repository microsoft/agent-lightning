# GEPA Example

This example demonstrates [GEPA](https://github.com/gepa-ai/gepa) (Generalized Evolutionary Prompt Adaptation) prompt optimization on a room-booking agent, using Azure OpenAI with Entra ID authentication.

## Overview

The room-booking agent selects the best meeting room given constraints (capacity, equipment, accessibility, availability). GEPA optimizes the prompt template through evolutionary search with reflective mutations, tracking a Pareto frontier of per-example performance.

## Requirements

1. Install Agent-Lightning with GEPA extras:

   ```bash
   uv sync --extra gepa
   ```

2. Install the Azure Identity library:

   ```bash
   uv pip install azure-identity
   ```

3. Authenticate with Azure:

   ```bash
   az login
   ```

4. Copy `.env.example` to `.env` and fill in your Azure OpenAI resource details. Export the variables before running:

   ```bash
   export $(grep -v '^#' .env | xargs)
   ```

## Included Files

| File | Description |
|------|-------------|
| `room_selector.py` | Room booking agent using Azure OpenAI with Entra ID authentication |
| `room_selector_gepa.py` | GEPA training script that optimizes the agent's prompt template |
| `room_tasks.jsonl` | Dataset with 57 room booking scenarios and expected selections |
| `.env.example` | Template for required Azure environment variables |

## Smoke Test

Run a single task to verify Azure authentication and connectivity:

```bash
python room_selector.py
```

## Full Training

Run GEPA optimization over the training dataset:

```bash
python room_selector_gepa.py
```

GEPA will evaluate prompt candidates, build reflective datasets from execution traces, and propose improved prompts. The best prompt is logged at the end of training.

## GEPA vs APO

| Aspect | APO (`examples/apo/`) | GEPA (`examples/gepa/`) |
|--------|----------------------|------------------------|
| Algorithm | Beam search with gradient-based proposals | Evolutionary search with reflective mutations |
| Selection strategy | Single best | Pareto frontier (per-example tracking) |
| Adapter | Requires `TraceToMessages` adapter | Built-in adapter (no explicit adapter needed) |
| Reflection model | Uses `AsyncOpenAI` client directly | Uses `litellm.completion()` via model string |
| Auth in this example | OpenAI API key | Azure Entra ID (DefaultAzureCredential) |
| Budget control | Beam rounds / beam width | `max_metric_calls` budget |
