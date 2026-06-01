# llm-in-sandbox Example - Local K8s VERL Training

Run the llm-in-sandbox training flow on agl-lite with local Kubernetes or minikube.

## Directory Structure

| Path | Purpose |
|------|---------|
| `README.md` | This guide |
| `.env.example` | Deploy, model, data, and training defaults |
| `Dockerfile.agent` | Agent image with llm-in-sandbox and data baked in |
| `job-template.yaml` | Pod spec fragment for rollout Jobs |
| `gateway-config.yaml` | Gateway route config that injects `return_token_ids` |
| `hooks.py` | agl-lite rollout hook that injects per-sample env vars |
| `agents/runner.py` | Container adapter that runs `llm-in-sandbox run_in_container` and posts events |
| `train_llm_in_sandbox.py` | Host-side VERL training entrypoint |
| `run.sh` | Build images, deploy agl-lite/controller, and run training |
| `data/` | Required train and test datasets |
| `vendor/llm-in-sandbox/` | Vendored llm-in-sandbox package from the reference zip |

## Environment

Use Python 3.12 for agl-lite and VERL, matching the rest of this repository.

```bash
conda create -n agl-lite python=3.12 -y
conda activate agl-lite
python -m pip install -U pip uv

# Pick the CUDA wheel index for the machine: cu126, cu128, cu130, or cpu.
scripts/setup_verl.sh cu128
```

You also need:

- minikube or another local Kubernetes cluster
- `kubectl`
- Docker/minikube image build support
- GPU capacity for VERL's internal vLLM server

## Data

The example uses these datasets copied from `llm-in-sandbox.zip`. The test data
path is generic and defaults to math in `.env.example`; point it at another
compatible `test_verl.json` directory to test a different split.

| Split | Path |
|-------|------|
| train | `data/llm_sandbox_instruct_pretrain/train_verl.json` |
| test default | `data/llm_sandbox_math_mini/test_verl.json` |
| test optional | `data/llm_sandbox_chem_mini/test_verl.json` |

The same paths are explicit in `.env.example`:

```bash
AGL_TRAIN_DATA_DIR=examples/llm-in-sandbox/data/llm_sandbox_instruct_pretrain
AGL_TEST_DATA_DIR=examples/llm-in-sandbox/data/llm_sandbox_math_mini
```

## Local K8s / Minikube

Start minikube, set `AGL_KEY`, and run the example:

```bash
export AGL_KEY=$(openssl rand -hex 32)
examples/llm-in-sandbox/run.sh
```

## How The Flow Works

```text
run.sh
  -> scripts/build_images.sh --include-example llm-in-sandbox
  -> uv run agl-lite deploy --env-file examples/llm-in-sandbox/.env.example
  -> train_llm_in_sandbox.py
       -> agl_lite.verl.entrypoint.run_ppo(...)
       -> agl-lite enqueue rollout
       -> controller creates K8s Job
       -> agent pod runs /app/runner.py
       -> runner calls llm-in-sandbox run_in_container
       -> LLM calls go through OPENAI_BASE_URL to agl-lite gateway
       -> runner posts agent_output and reward events
```

## Logs And Cleanup

Logs are written under `examples/llm-in-sandbox/logs/<timestamp>/`.

Clean up the local deployment with:

```bash
uv run agl-lite deploy --env-file examples/llm-in-sandbox/.env.example --cleanup
```