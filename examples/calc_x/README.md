# Calc-X Example — VERL Training on agl-lite

Train a mathematical reasoning agent using VERL (PPO/GRPO) on agl-lite.
The agent uses AutoGen + MCP calculator tools to solve math problems,
with the agl-lite gateway transparently capturing all LLM interactions
for RL training.

## Architecture

```
run.sh → agl-lite deploy (agl-in-host) → train_calc_agent.py
              │                                    │
              ├── K8s: controller                  ├── load Calc-X dataset
              └── Host: agl-lite serve             └── run_ppo() → VERL trainer
                    │                                        │
                    │                               AglLiteRolloutBridge (HTTP)
                    │                                        │
                    ├── enqueue rollouts ←───────────────────┘
                    ├── controller creates K8s Jobs
                    │     └── agent pod: AutoGen + MCP calculator
                    │           └── LLM calls → gateway → vLLM
                    ├── gateway captures token IDs
                    └── triplets → padded tensors → PPO update
```

## Requirements

- Single node with at least one 40GB GPU
- minikube running with nvidia-container-toolkit
- `uv` installed
- Python 3.12+

## Python / Conda Environment

Use Python 3.12. The run script expects the repo-local `.venv`, so a fresh
conda environment only needs to provide Python and `uv`; `setup_verl.sh` creates
and installs the project `.venv` used by the script.

```bash
conda create -n agl-lite python=3.12 -y
conda activate agl-lite
python -m pip install -U pip uv

# Pick the CUDA wheel index that matches the machine: cu126, cu128, cu130, or cpu.
scripts/setup_verl.sh cu128
```

## Dataset

Download the Calc-X dataset from [Google Drive](https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view?usp=sharing)
and extract to the `data/` folder:

```bash
cd examples/calc_x
# Download calc-x-data.zip from the link above, then:
unzip data/calc-x-data.zip -d data/
```

The dataset contains:
- `train.parquet` — 8192 math problems for training
- `test.parquet` — 500 problems for validation
- `test_mini.parquet` — 20 problems for quick testing
- `sample.jsonl` — 10 rows for smoke testing (checked into git)

## Quick Start

```bash
# Prerequisites
export AGL_KEY=$(openssl rand -hex 32)

# Full training
examples/calc_x/run.sh

# CI smoke test (single PPO step)
examples/calc_x/run.sh --ci-fast
```

## Standalone Training

If agl-lite and vLLM are already running:

```bash
python examples/calc_x/train_calc_agent.py \
    --train-file examples/calc_x/data/train.parquet \
    --val-file examples/calc_x/data/test.parquet
```

## Files

| File | Description |
|------|-------------|
| `calc_agent.py` | Standalone agent container — AutoGen + MCP calculator, no agl-lite imports |
| `train_calc_agent.py` | Training script — loads dataset, builds VERL config, calls `run_ppo()` |
| `run.sh` | E2E entrypoint — verify vLLM, build images, deploy, run training |
| `Dockerfile` | Agent container image |
| `job-template.yaml` | Complete K8s Job template for agent jobs |
| `data/` | Dataset directory (parquet files, gitignored except sample.jsonl) |

### Logs

Each run creates a timestamped log directory under `logs/`:

```
logs/20260410-002043/
  server.log       # agl-lite server (JSON, structlog)
  training.log     # VERL training output (Ray workers, metrics, progress)
  agents/          # Per-agent logs (mounted from minikube via hostPath)
    <attempt-id>/
      agent.log    # Agent stdout + structured logs
```

`run.sh` sets up `minikube mount` so agent pod logs (written to hostPath
`/tmp/agl-lite/logs/` inside the VM) appear on the host filesystem.
