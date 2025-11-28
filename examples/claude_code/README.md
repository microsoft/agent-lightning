# Training Claude Code with Agent-lightning

This example demonstrates how to train a Claude Code agent with Agent-lightning. **The example is still under development.**

It wraps Claude Code as the agent to:

1. collect traces from agent execution on coding tasks;
2. train a hosted LLM with the traces ***🔨 Under development***

## Requirements

1. Install agentlightning following [installation instructions](https://microsoft.github.io/agent-lightning/stable/tutorials/installation/);
2. `(uv) pip install swebench` for evaluation.

## Dataset

We provide a small dataset `swebench_samples.jsonl` which is a subset of [SWE-bench](https://huggingface.co/datasets/SWE-bench/SWE-bench) for sanity check.

The instruction to prepare the full dataset is still underway.

## Included Files

| Filename                        | Description |
|--------------------------------|-------------|
| `cc_agent.py`                   | Main entry point for running Claude Code agent on coding tasks with trace collection capabilities |
| `claude_code_controller.py`     | Controller implementation for managing Claude Code agent interactions and execution |
| `custom_adapter.py`             | Custom adapter for integrating with Claude Code's interface and communication protocols |
| `custom_callbacks.py`           | Callback handlers for customizing agent behavior and responses during execution |
| `handle_hook.template.sh`       | Template script for handling hooks during agent execution |
| `settings.template.json`        | Template configuration file with default settings for Claude Code agent |
| `swe_debug.jsonl`               | Debug dataset containing a subset of SWE-bench samples for testing and verification |
| `swebench_utils/`               | Utility module with helper functions for SWE-bench dataset containerized exeuction and evaluation |

## Trace collection

We support running Claude Code via two ways:

- Hosted LLM servers (i.e., vLLM), useful for fine-tuning the LLM;
- Official Claude Code (i.e., via Anthropic API), useful for prompt tuning.

### From Hosted LLM server

1. Prepare an OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --max-model-len 131072 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

2. Sanity check:

```bash
# Suppose the vllm server is running at localhost:8000
python cc_agent \
    --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --server_address http://localhost:8000/v1 \
    --dataset_path swe_debug.jsonl \
    --max_step 32 \
    --output_dir data_debug
```

The above commands will generate a `data_debug` dir, which contains two targets: (1) a Huggingface Dataset named `dataset-<instance_id>` and (2) a trace file named `stream_<instance_id>.jsonl`, where `instance_id` is a unique key of the SWE-bench samples.
The dataset showcases the versatile customization capability of agent-lightning. In particular, we support extracting **prompt/response ids**, **logprobs** from the vllm server.
The trace file is the conversation logs for claude code to tackle the SWE-bench instance.

In addition, there will be a `logs` dir, which is the output of the docker container executing agent calls.

### From official Claude Code
1. Prepare ANTHROPIC_API_KEY
```bash
export ANTHROPIC_API_KEY=sk-<your private key>
```

2. Sanity check
```bash
cd examples/cc
python cc_agent \
    --official \
    --dataset_path swe_debug.jsonl \
    --max_step 32 \
    --output_dir data_debug
```
As the underlying model is provided by Anthropic, we cannot obtain prompt/response ids and logprobs. However, we can still obtain a trace file named `<instance_id>.json` under `data_debug`.
