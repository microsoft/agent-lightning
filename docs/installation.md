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

`verl` setup is more involved than the base environment. Its compatible versions are tightly coupled with the installed `torch` and `vllm` versions, so use `scripts/setup_verl.sh` to install the pinned GPU stack.

Pass the VERL version and CUDA wheel variant explicitly. The script supports VERL `0.7.1` or `0.8.0`, and CUDA wheel variant `cu129` or `cu130`:

```bash
source .venv/bin/activate
bash scripts/setup_verl.sh 0.8.0 cu130

# or

bash scripts/setup_verl.sh 0.7.1 cu129
```

For VERL `0.7.1`, the script installs `vllm==0.12.0`. For VERL `0.8.0`, it installs `vllm==0.20.2` first, then installs `verl==0.8.0`. Both paths build `flash-attn==2.8.3` locally against the selected environment. Depending on the number of CPU cores available, the script can take 10-30 minutes to complete.

Before running experiments that report to Weights & Biases, log in once:

```bash
uv run wandb login
```
