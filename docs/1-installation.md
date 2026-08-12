# Installation

This guide sets up a single-node environment for Agent Lightning v1.0. After completing it, you can run single-machine training jobs.

Before getting started, install `uv` and NVIDIA CUDA. We support CUDA `12.9` or `13.0`.

#### Step 1: UV Sync

From the project root, run:

```bash
cd <this-repo>
uv sync
```

This installs the base Python environment into `.venv` under the project root.

#### Step 2: Install `verl` and FlashAttention

Agent Lightning uses `verl` as its training backend. The compatible versions of `verl`, `vllm`, and `torch` are tightly coupled, and installing `flash-attn` can also be error-prone. We recommend using `scripts/setup_verl.sh` to install the tested, pinned GPU stack and build `flash-attn` from source.

Pass the `verl` version and CUDA wheel variant explicitly. The script supports `verl==0.7.1` or `verl==0.8.0`, and CUDA wheel variant `cu129` or `cu130`. We recommend CUDA `13.0` with `verl==0.8.0`:

```bash
source .venv/bin/activate
bash scripts/setup_verl.sh 0.8.0 cu130

# or

bash scripts/setup_verl.sh 0.7.1 cu129
```

For `verl==0.7.1`, the script installs `vllm==0.12.0`. For `verl==0.8.0`, it installs `vllm==0.20.2` first, then installs `verl==0.8.0`. Both paths build `flash-attn==2.8.3` locally against the selected environment. Depending on the number of CPU cores available, the script can take 10-30 minutes to complete.

#### Step 3: W&B Login

By default, all tasks upload logs and trajectories to Weights & Biases. Log in to W&B before running a task:

```bash
uv run wandb login
```
