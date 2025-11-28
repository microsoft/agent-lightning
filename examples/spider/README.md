# Spider Example

[![spider CI status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-spider.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-spider.yml)

This example demonstrates how to train a text-to-SQL agent on the Spider dataset using Agent-Lightning with reinforcement learning. It's compatible with Agent-lightning v0.2 or later.

## Requirements

This example depends on LangChain v0.x and several SQL-related libraries. Install the required dependencies with:

```bash
pip install "langgraph<1.0" "langchain[openai]<1.0" "langchain-community" "langchain-text-splitters<1.0" "sqlparse" "nltk"
```

Additionally, follow the [installation guide](../../docs/tutorials/installation.md) to install Agent-Lightning and VERL-related dependencies.

## Dataset

Detailed dataset preparation instructions are available in the [How to Train a SQL Agent](../../docs/how-to/train-sql-agent.md) guide.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `train_sql_agent.py` | Training script for SQL agents with support for multiple model configurations (Qwen, LLaMA, fast mode for CI) |
| `sql_agent.py` | SQL agent implementation using LangGraph and LangChain, with debugging capabilities |
| `data/` | Directory containing the Spider dataset files |
| `spider_eval/` | Evaluation utilities for assessing SQL agent performance |

## Running Examples

### Training

Train a SQL agent using the Qwen2.5-Coder-1.5B-Instruct model with the following command. This requires a single node with at least one 40GB GPU:

```bash
python train_sql_agent.py qwen
```

If you want to use an NPU for training, please refer to the **Launch Training with NPUS** section in [How to Train a SQL Agent](../../docs/how-to/train-sql-agent.md).

### Debugging

To test and debug the SQL agent interactively:

```bash
python sql_agent.py
```

This command requires an OpenAI-compatible API service. Configure your service endpoint and credentials using the `OPENAI_API_BASE` and `OPENAI_API_KEY` environment variables.
