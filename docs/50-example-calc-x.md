# Calc-X

| GPU | Model | Controller Mode | Trainer Mode | Code |
|---|---|---|---|---|
| 1× A100 80GB | `Qwen/Qwen2.5-1.5B-Instruct` | K8s or local | Sync and async | [Source](https://github.com/microsoft/agent-lightning/tree/main/examples/calc_x) |

Calc-X is a proof-of-concept (POC) example that trains a mathematical reasoning agent on the Calc-X dataset with `verl` and Agent Lightning >=v1.0. It is intentionally lightweight and requires only one GPU. The agent uses AutoGen + MCP calculator tools to solve math problems.

The example supports two controller modes:

- **K8s mode:** Minikube provides a minimal Kubernetes environment, and agent rollouts run as Kubernetes Jobs.
- **Local mode:** Agent rollouts run directly as local processes without Kubernetes.

Both synchronous and asynchronous trainer modes are supported.

## Data Preparation

Download the Calc-X dataset from [Google Drive](https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view?usp=sharing), then extract it into `examples/calc_x/data/`:

```bash
cd examples/calc_x
unzip data/calc-x-data.zip -d data/
```

The expected dataset files are:

- `data/train.parquet`
- `data/test.parquet`
- `data/test_mini.parquet`
- `data/sample.jsonl`

## Local Mode

Make sure you have activated the project environment and installed the following package in Python:

```bash
source .venv/bin/activate
uv pip install \
    openai \
    httpx \
    sympy \
    "autogen-agentchat" \
    "autogen-ext[openai]" \
    "mcp>=1.11.0,<2" \
    mcp-server-calculator
```

Then start training:

```bash
source .venv/bin/activate
cd examples/calc_x
bash run_local.sh
```

`run_local.sh` starts `agl-server` and `agl-controller`, and writes their logs under `/tmp/`. The script starts the agent in multi-process mode.
When `run_local.sh` exits, it automatically cleans up the server, controller, and agent it started.

## K8s Mode

This example uses Minikube to demonstrate the minimal Kubernetes workflow. For production deployments, replace Minikube with a production-grade Kubernetes cluster.

Make sure you have installed `docker` and `minikube`, then start training by:

```bash
source .venv/bin/activate
cd examples/calc_x
bash run_minikube.sh
```

`run_minikube.sh` starts `agl-server` and `agl-controller`, and writes their logs under `/tmp/`. The script also starts a new local Minikube single-node K8s cluster, and the agent runs in this cluster as Kubernetes Jobs.
When `run_minikube.sh` exits, it automatically cleans up the server, controller, and Minikube it started.
Minikube needs at least 64 GB of memory; otherwise, it may be killed due to insufficient memory.

## Evaluating a Trained Checkpoint

Training writes checkpoints under `trainer.default_local_dir`, which for this example is
`checkpoints/agentlightning/calc_x/`, one folder per `global_step_N`. To score one of them on
a held-out split without training anything further, pass `--val-only` and point `--resume` at
the checkpoint:

```bash
source .venv/bin/activate
cd examples/calc_x
python train_calc_agent.py \
    --val-only \
    --resume checkpoints/agentlightning/calc_x/global_step_128 \
    --val-file data/test.parquet
```

`--resume` sets `trainer.resume_mode` and `trainer.resume_from_path`, which is how `verl`
restores a checkpoint. `--model` keeps its earlier meaning, the base model, because the
gateway requires its own model name and `actor_rollout_ref.model.path` to be identical
([API gateway configuration](25-api-gateway-configuration.md)); aiming either of them at a
checkpoint makes the proxy answer "model not found".

`--val-only` then runs a single validation pass over `--val-file` and exits without taking an
optimizer step. The metrics are printed as `Initial validation metrics: {...}` and logged to
W&B under the `calc_x_eval` experiment. The one to read is `val/reward`: it is the mean reward
over the split, and the Calc-X reward is 1.0 for an answer matching the reference and 0.0
otherwise, so it doubles as the accuracy.

Under local mode the same flags go through `run_local.sh`, which forwards its arguments:
`bash run_local.sh --val-only --resume checkpoints/agentlightning/calc_x/global_step_128`.

Two details worth knowing:

- `--val-only` still wants a non-empty train split, because `run_ppo` sizes its sampler from
  one. The example hands the held-out split in as both, so make sure it has at least
  `data.train_batch_size` rows (32 by default).
- The checkpoint is only read, never written, so the same split can be scored again later
  and compared across runs.
