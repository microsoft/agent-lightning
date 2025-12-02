# Training Claude Code with Agent-lightning

[![claude-code CI status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-claude-code.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-claude-code.yml)

This example shows how to wrap Anthropic's Claude Code experience with Agent-lightning instrumentation to solve SWE-bench tasks, collect spans/logs, and optionally convert those traces into HuggingFace datasets.

**NOTE:** This example only shows how to integrate Claude Code as an agent in Agent-lightning. The training part is still under development and welcoming contributions!

## Overview

`claude_code_agent.py` spins up a Lightning Store, an LLM proxy, and the Claude Code controller. Each SWE-bench instance is executed inside the official container image so you can either prompt-tune against Anthropic's hosted models or point Claude Code at a self-hosted OpenAI-compatible backend such as vLLM. When a backend surfaces token IDs/logprobs (e.g., vLLM), the traces are turned into triplets that downstream fine-tuning pipelines can consume.

## Requirements

First, install Agent-lightning following the [installation guide](https://microsoft.github.io/agent-lightning/stable/tutorials/installation/). Then install the SWE-bench harness plus utilities used by this example:

```bash
(uv) pip install swebench transformers datasets python-dotenv
```

Docker must be available because each SWE-bench instance is executed in a container via `swebench_utils`.

Finally, set API credentials depending on backend:

- `ANTHROPIC_API_KEY` for the official Claude Code path.
- `OPENAI_API_KEY` (or another OpenAI-compatible key) for the `openai` backend.
- A running OpenAI-compatible server (e.g., vLLM) when using the `vllm` backend.

## Dataset

`swebench_samples.jsonl` contains a handful of SWE-bench issues for smoke testing. For full-scale benchmarks load `princeton-nlp/SWE-bench` via `load_swebench_dataset` or point `--dataset-path` to your own JSONL file.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `claude_code_agent.py` | CLI entry point that launches the Lightning store, LLM proxy, and Claude Code agent |
| `claude_code_controller.py` | Manages the SWE-bench Docker runtime and translates model outputs into git patches |
| `extended_adapter.py` | Adapter that converts LLM proxy spans into triplets with token IDs, logprobs, and chat history |
| `swebench_samples.jsonl` | Mini SWE-bench subset for quick validation |
| `swebench_utils/` | Utilities for running/evaluating SWE-bench instances inside containers |
| `templates/handle_hook.template.sh` | Helper script injected into containers for hook handling |
| `templates/settings.template.json` | Base configuration consumed by Claude Code CLI |

## Running the Example

All commands are issued from `examples/claude_code`. Inspect the module-level docstring in `claude_code_agent.py` for the full CLI reference.

### Hosted vLLM (open-source models)

First, launch your model behind an OpenAI-compatible endpoint, for example:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --max-model-len 131072 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

Run the Agent-lightning harness and point it at the server:

```bash
python claude_code_agent.py vllm \
    --backend-model-high Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --backend-model-low Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --frontend-model-high claude-sonnet-4-5-20250929 \
    --frontend-model-low claude-haiku-4-5-20251001 \
    --base-url http://localhost:8000/v1 \
    --dataset-path swebench_samples.jsonl \
    --output-dir data_debug \
    --max-turns 5 \
    --limit 2
```

The backend model names must match what the server exposes. Because this mode surfaces token IDs/logprobs, the script saves both raw span logs and HuggingFace datasets per instance.

### Official Claude Code (Anthropic API)

```bash
export ANTHROPIC_API_KEY=sk-...
python claude_code_agent.py anthropic \
    --dataset-path swebench_samples.jsonl \
    --output-dir data_anthropic \
    --frontend-model-high claude-sonnet-4-5-20250929 \
    --frontend-model-low claude-haiku-4-5-20251001
```

Backend model flags are optional here because the Anthropic API strings match the frontend names. This path is ideal for validating prompts against the hosted experience (trace outputs do not contain token IDs or logprobs).

### OpenAI-Compatible Providers

```bash
export OPENAI_API_KEY=sk-...
python claude_code_agent.py openai \
    --backend-model-high gpt-4.1 \
    --backend-model-low gpt-4o-mini \
    --dataset-path swebench_samples.jsonl \
    --output-dir data_openai
```

Use this mode whenever Claude Code should talk to Azure OpenAI, OpenAI, or another compatible provider. `--base-url` is optional—pass it if your endpoint differs from the public OpenAI URL.

Adjust `--max-turns`, `--cooldown-seconds`, and `--limit` to control runtime and rate limits regardless of backend.

## Outputs and Trace Collection

- `output_dir/stream_<instance_id>.json` contains the complete span stream captured from the Lightning Store for each rollout.
- When running with `backend_type=vllm`, `output_dir/dataset-<instance_id>/` stores a HuggingFace dataset with token IDs, logprobs, prompts, and metadata produced by `ExtendedLlmProxyTraceToTriplet`.
- `logs/<instance_id>/` is created by the SWE-bench runtime and mirrors the console output from the container.
- Return values from the agent are also evaluated via `swebench_utils.evaluation.evaluate`, so `data_debug` (or your chosen folder) will contain evaluation reports alongside traces.

Use these artifacts to fine-tune models, debug Claude Code behavior, or replay rollouts in downstream Agent-lightning workflows.
