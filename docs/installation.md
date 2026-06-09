# Installation

This guide sets up a single-node environment for AGL Lite. After completing it, you can run single-machine training jobs.

Install the following first:

- `uv`
- NVIDIA CUDA: `12.9` or `13.0`

From the project root, run:

```bash
uv sync
```

This installs the base Python environment into `.venv` under the project root.

To run commands inside this environment, use `uv run`, for example:

```bash
uv run python --version
```

`verl` setup is more involved than the base environment. Its compatible versions are tightly coupled with the installed `torch` and `vllm` versions, so use `scripts/setup_verl.sh` to install the pinned GPU stack.

Pass the CUDA wheel variant explicitly. The script only accepts `cu129` or `cu130`:

```bash
scripts/setup_verl.sh cu130

# or

scripts/setup_verl.sh cu129
```

The script installs `vllm==0.12.0` and `verl==0.7.1` into `.venv` with the selected CUDA wheel variant, then builds `flash-attn==2.8.3` locally against that environment.

Before running experiments that report to Weights & Biases, log in once:

```bash
uv run wandb login
```
