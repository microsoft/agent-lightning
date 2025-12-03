# Math Reasoning Agent with Tool Use

A beginner-friendly example demonstrating how to train an AI agent to solve grade school math problems using reinforcement learning with Agent-Lightning. The agent learns to use a calculator tool effectively and improve its reasoning over time.

## Overview

This example shows how Agent-Lightning can optimize an agent with **minimal code changes**.

<br/>

The agent:

- Solves math word problems from the GSM8K dataset
- Uses a calculator tool for arithmetic operations
- Learns through reinforcement learning (GRPO) to improve accuracy
- Runs on CPU or a single GPU with smaller models

**Key Features:**

- ✅ Simple setup - no external services required
- ✅ Beginner-friendly - clear code structure
- ✅ Educational - demonstrates core RL concepts
- ✅ Scalable - tested with models from 1B to 70B+ parameters

## Quick Start

### 1. Install Dependencies

```bash
pip install agentlightning
pip install datasets  # for GSM8K dataset
```

### 2. Prepare Training Data

```bash
python prepare_data.py
```

This downloads the GSM8K dataset and prepares it in the required format. Creates:

- `data/gsm8k_train.parquet` (~7K examples)
- `data/gsm8k_test.parquet` (~1K examples)

### 3. Start Ray Cluster

```bash
bash ../../scripts/restart_ray.sh
```

Optional: Set `WANDB_API_KEY` environment variable before starting Ray for experiment tracking.

### 4. Run the Agent

```bash
python math_agent.py
```

This launches 8 agent workers by default (configurable via `n_workers` parameter).

### 5. Start Training

In another terminal:

```bash
bash train.sh
```

The training will:

- Run for 3 epochs by default
- Save checkpoints every 50 steps
- Evaluate on test set every 25 steps
- Log metrics to console and Wandb

### Zero Code Change Philosophy

The agent code uses standard Python with minimal Agent-Lightning integration:

```python
class MathAgent(LitAgent):
    async def training_rollout_async(self, task, rollout_id, resources):
        # Your normal agent logic here
        result = solve_problem(task['question'])
        reward = compute_reward(result, task['answer'])
        return reward
```

### Progressive Learning

Watch the agent improve over training:

- **Epoch 0**: ~20-30% accuracy (baseline)
- **Epoch 1**: ~40-50% accuracy (learns tool use)
- **Epoch 2**: ~55-65% accuracy (refined reasoning)
- **Epoch 3**: ~60-70% accuracy (near convergence)

### Reward Shaping

The example demonstrates sophisticated reward design:

- **Correct answer**: +1.0
- **Correct tool use but wrong answer**: +0.3
- **Valid format but wrong answer**: +0.1
- **Invalid format**: -0.1

## Architecture

### Files Structure

```
examples/math_tool_use/
├── README.md              # main documentation
├── CONTRIBUTING.md
├── requirements.txt       #  dependencies
├── math_agent.py         # cpre agent implementation
├── calculator_tool.py    # calc tool definition
├── utils.py              # reward computation & metrics
├── prepare_data.py       # dataset preparation
├── train.sh              # training config
├── evaluate.py           # eval script
└── data/                 # generated data directory
```

### Agent Workflow

1. **Receive Problem**: Agent gets a math word problem
2. **Reasoning**: Agent thinks through the solution step-by-step
3. **Tool Use**: Agent calls calculator for arithmetic operations
4. **Generate Answer**: Agent provides final numerical answer
5. **Reward**: System computes reward based on correctness

## Configuration

### Model Settings

Edit `train.sh` to customize:

```bash
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct  # Model to train
export N_GPUS=1                                # Number of GPUs
export ROLLOUT_TP_SIZE=1                       # Tensor parallelism
```

models i used:

- **CPU/Small GPU**: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`
- **Single GPU (24GB)**: `Qwen/Qwen2.5-7B-Instruct`

to try:

- **Multi-GPU**: `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`

### Training Hyperparameters

Key parameters in `train.sh`:

```bash
data.train_batch_size=64              # Batch size for training
actor_rollout_ref.rollout.n=4         # Samples per prompt
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.optim.lr=5e-6  # Learning rate
trainer.total_epochs=3                 # Number of epochs
```

### Reward Function

Customize reward logic in `utils.py`:

```python
def compute_reward(predicted, ground_truth, used_calculator):
    if is_correct(predicted, ground_truth):
        return 1.0
    elif used_calculator:
        return 0.3  # Partial credit for tool use
    elif has_valid_format(predicted):
        return 0.1
    return -0.1
```

## Dataset: GSM8K

The [GSM8K dataset](https://github.com/openai/grade-school-math) contains:

- **Training**: 7,473 grade school math problems
- **Test**: 1,319 problems
- **Format**: Natural language questions with numerical answers

## Evaluation Results

Results with `Qwen2.5-1.5B-Instruct` on a single RTX 3060:

| Epoch | Train Reward | Test Accuracy | Training Time |
| ----- | ------------ | ------------- | ------------- |
| 0     | 0.15         | 22%           | -             |
| 1     | 0.42         | 48%           | 45 min        |
| 2     | 0.58         | 63%           | 45 min        |
| 3     | 0.65         | 68%           | 45 min        |

Results may vary based on:

- Model size and initialization
- Hyperparameters
- Random seed
- Hardware configuration

## Quick Troubleshooting

### Out of Memory

1. Reduce batch size:

   ```bash
   data.train_batch_size=32
   actor_rollout_ref.actor.ppo_mini_batch_size=16
   ```

2. Enable gradient checkpointing (already enabled in default config)

### Poor Convergence

1. Adjust learning rate:

   ```bash
   actor_rollout_ref.actor.optim.lr=1e-5  # Try 1e-5 to 1e-6
   ```

2. Increase samples per prompt:

   ```bash
   actor_rollout_ref.rollout.n=8  # More exploration
   ```

3. Check reward shaping - ensure positive signals for partial progress
