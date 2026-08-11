# Calc-X

| GPU | Model | Controller Mode | Trainer Mode |
|---|---|---|---|
| 1× A100 80GB | `Qwen/Qwen2.5-1.5B-Instruct` | K8s or local | Sync and async |

Calc-X is a proof-of-concept (POC) example that trains a mathematical reasoning agent on the Calc-X dataset with VERL and Agent Lightning >=v1.0. It is intentionally lightweight and requires only one GPU. The agent uses AutoGen + MCP calculator tools to solve math problems.

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
    "mcp>=1.10.0" \
    mcp-server-calculator
```

Then start training:

```bash
source .venv/bin/activate
cd examples/calc_x
bash run_local.sh
```

`run_local.sh` starts agl-lite-server and agl-lite-controller, and writes their logs under `/tmp/`. The script starts the agent in multi-process mode.
When `run_local.sh` exits, it automatically cleans up the server, controller, and agent it started.

## K8s Mode

This example uses Minikube to demonstrate the minimal Kubernetes workflow. For production deployments, replace Minikube with a production-grade Kubernetes cluster.

Make sure you have installed `docker` and `minikube`, then start training by:

```bash
source .venv/bin/activate
cd examples/calc_x
bash run_minikube.sh
```

`run_minikube.sh` starts agl-lite-server and agl-lite-controller, and writes their logs under `/tmp/`. The script also starts a new local Minikube single-node K8s cluster, and the agent runs in this cluster as Kubernetes Jobs.
When `run_minikube.sh` exits, it automatically cleans up the server, controller, and Minikube it started.
Minikube needs at least 64 GB of memory; otherwise, it may be killed due to insufficient memory.
