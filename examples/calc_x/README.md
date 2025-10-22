# Calc-X Example

[![calc_x CI status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-calc-x.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-calc-x.yml)

This example demonstrates training a mathematical reasoning agent using Agent-Lightning with the VERL algorithm and AutoGen framework. The agent solves math problems using a calculator tool through the Model Context Protocol (MCP). It's compatible with Agent-lightning v0.2 or later.

## Requirements

This example requires a single node with at least one 40GB GPU. Follow the [installation guide](../../docs/tutorials/installation.md) to install Agent-Lightning and VERL-related dependencies.

Additionally, ensure `uv` and the MCP calculator server are properly installed. The agent relies on the MCP protocol to access calculator functionality during problem-solving.

```bash
pip install "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0"
```

## Dataset

Download the Calc-X dataset in parquet format from [here](https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view?usp=sharing) and extract it to the `data` folder:

```bash
unzip calc-x-data.zip -d data
```

The dataset contains mathematical problems with ground truth solutions for training and evaluation.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `calc_agent.py` | Math problem-solving agent using AutoGen and MCP calculator tool |
| `train_calc_agent.py` | Training script using VERL algorithm with configurable hyperparameters |
| `eval_utils.py` | Evaluation utilities for assessing agent accuracy on math problems |
| `data/` | Directory containing training and test datasets in parquet format |
| `tests/` | Test files including MCP calculator verification script |
| `legacy_calc_agent.py` | Legacy agent implementation compatible with Agent-lightning v0.1.x (deprecated) |
| `legacy_calc_agent_debug.py` | Legacy debugging script compatible with Agent-lightning v0.1.x (deprecated) |
| `legacy_train.sh` | Legacy training script compatible with Agent-lightning v0.1.x (deprecated) |

## Running Examples

### Training

The training process uses distributed Ray workers to run agent rollouts in parallel while the training server optimizes the model. Start Ray before launching the training:

```bash
bash ../../scripts/restart_ray.sh
```

If you want to track experiments with Weights & Biases, set the `WANDB_API_KEY` environment variable **before starting Ray**.

Then run the training script:

```bash
python train_calc_agent.py --train-file data/train.parquet --val-file data/test.parquet
```

The script automatically launches agent workers and the training server. The agent workers execute math problem rollouts using the MCP calculator, while the training server applies the VERL algorithm to improve the model based on rewards.

### Debugging

To test the agent interactively without training:

```bash
python calc_agent.py
```

This runs the agent on sample problems to verify that the MCP calculator integration and AutoGen setup work correctly. This test relies on an OpenAI service available. Set `OPENAI_API_KEY` environment variable to the API key of the OpenAI service; and `OPENAI_API_BASE` environment variable to the base URL of the OpenAI service.

A very common issue is that the agent may hang indefinitely if the environment is not properly configured. Verify that `uv` and the MCP calculator server are correctly installed by running:

```bash
python tests/test_mcp_calculator.py
```
