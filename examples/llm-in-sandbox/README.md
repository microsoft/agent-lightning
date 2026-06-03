# llm-in-sandbox Example - Local K8s VERL Training

Run the llm-in-sandbox training flow on agl-lite with local Kubernetes or minikube.

## Directory Structure

| Path | Purpose |
|------|---------|
| `README.md` | This guide |
| `Dockerfile.agent` | Agent image with llm-in-sandbox and data baked in |
| `job-template.yaml` | Complete K8s Job template for rollout Jobs |
| `agents/runner.py` | Container adapter that runs `llm-in-sandbox run_in_container` and posts events |
| `train_llm_in_sandbox.py` | Host-side VERL training entrypoint |
| `run.sh` | Build images, start agl-lite server/controller, and run training |
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

The example uses these datasets copied from `llm-in-sandbox.zip`. The default
train and validation splits are the sampled mini datasets listed below.

| Split | Path |
|-------|------|
| train default | `data/llm_sandbox_sampled_pretrain_mini/train_verl.json` |
| val default | `data/llm_sandbox_sampled_vali_mini/test_verl.json` |
| other compatible val split | any `test_verl.json` directory under `data/` |


## Local K8s / Minikube

Start minikube, set `AGL_KEY`, and run the example:

```bash
export AGL_KEY=$(openssl rand -hex 32)
examples/llm-in-sandbox/run.sh
```

The launcher now starts `agl-lite-server` and `agl-lite-controller` directly, then runs `train_llm_in_sandbox.py`. Extra VERL settings can be passed as dotlist overrides after the script arguments, for example:

```bash
examples/llm-in-sandbox/run.sh --ci trainer.total_epochs=2 actor_rollout_ref.rollout.n=2
```

## How The Flow Works

```text
run.sh
  -> minikube start + minikube image build
  -> agl-lite-server
  -> agl-lite-controller
  -> train_llm_in_sandbox.py
       -> agl_lite.verl.entrypoint.run_ppo(...)
       -> agl-lite enqueue rollout
       -> controller creates K8s Job from job-template.yaml
       -> agent pod runs /app/runner.py
       -> runner calls llm-in-sandbox run_in_container
       -> LLM calls go through AGL_OPENAI_BASE_URL / OPENAI_BASE_URL to agl-lite gateway
       -> runner posts agent_output and reward events
```

## Logs And Cleanup

The launcher uses the local terminal output for logs. Clean up with `Ctrl+C`; the script stops the server, controller, and ray on exit.