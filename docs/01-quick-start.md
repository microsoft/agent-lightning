# Quick Start

This quick start requires only one machine with one A100 GPU. It runs Agent Lightning v1.0 with the **local controller** and provides the shortest path from an installed repository to a real rollout-driven training job.

## Before you start

Complete [Installation](00-installation.md), including the `verl` GPU stack.

> AGL v1.0 itself is lightweight, but policy inference and GRPO updates still require the GPU stack used by `verl` and vLLM.
>
> This quick start uses `runner_type=local`, which is not supported on native Windows. On Windows, run it from a Linux environment such as WSL. See [Local runner limits](30-controller-configuration.md#local-runner-limits) for details.

## 1. Prepare the example

Download the Calc-X dataset from [Google Drive](https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view?usp=sharing), then extract it and place these files under `examples/calc_x/data/`:

```text
train.parquet
test.parquet
test_mini.parquet
sample.jsonl
```

Activate the project environment and install the dependencies:

```bash
source .venv/bin/activate
uv pip install openai httpx sympy \
  "autogen-agentchat" "autogen-ext[openai]" \
  "mcp>=1.11.0,<2" mcp-server-calculator \
  googleapis-common-protos google-api-core proto-plus tensorboard
```

The last line is not a new dependency. AutoGen pins `protobuf` to 5.x, and naming these already-installed packages lets `uv` move them to versions that accept it. Otherwise the Ray dashboard agent fails to import and `ray start` times out.

## 2. Start one local run

From `examples/calc_x` (the launcher and the agent use paths relative to that directory):

```bash
cd examples/calc_x
bash run_local.sh
```

The launcher performs four operations:

1. starts Ray and the `verl`/vLLM model backend;
2. starts `agl-server` on port `8181`;
3. starts `agl-controller runner_type=local`;
4. runs the Calc-X training entrypoint.

Service logs are written under `/tmp/`.

Once the task is running, you can view the training results in W&B.

When you want to stop the run, press `Ctrl+C` once and wait for the script to exit. Do not press `Ctrl+C` repeatedly, as the cleanup process takes some time to stop all resources and processes safely.

## What's Next

1. Read [Basics](05-basics.md) to learn the core Agent Lightning >= v1.0 concepts.
2. Read the complete [Calc-X example](50-example-calc-x.md), which also covers the Kubernetes controller mode.
