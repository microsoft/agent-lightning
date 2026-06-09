# Calc-X

Calc-X trains a mathematical reasoning agent on the Calc-X dataset with VERL and agl-lite.
The agent uses AutoGen + MCP calculator tools to solve math problems.

The agent can run in two modes:

- Minikube mode: agent rollouts run as Kubernetes Jobs in minikube.
- Local mode: agent rollouts run as local multi-process workers.

This example only needs one GPU.

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

## Minikube Mode

Make sure you have installed `docker` and `minikube`, then start training by:

```bash
source .venv/bin/activate
cd examples/calc_x
bash run_minikube.sh
```

`run_minikube.sh` starts agl-lite-server and agl-lite-controller, and writes their logs under `/tmp/`. The script also starts a new local minikube single-node K8S cluster, and the agent runs in this cluster as Kubernetes jobs.
When `run_minikube.sh` exits, it automatically cleans up the server, controller, and minikube it started.
minikube needs at least 64 GB of memory; otherwise, it may be killed due to insufficient memory.

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
