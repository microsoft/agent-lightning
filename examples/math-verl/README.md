# Math VERL — training E2E on agl-lite

This example is the **training** path for math tasks using VERL + agl-lite.

- `examples/math-poc/` remains for **task delegation + data pipeline verification**.
- `examples/math-verl/` is for **training loop integration** (Phase 5c).

## Scope

- vLLM-only training path (no mock mode)
- `run.sh` deploys agl-lite in `agl-in-host` mode and runs VERL training

## Prerequisites

- minikube running
- GPU memory available for VERL's internal vLLM rollout server
- VERL dependencies installed in current Python env

## Python / Conda Environment

Use Python 3.12. Like `calc_x`, this is a VERL training example and the run
script expects the repo-local `.venv`, so create a small conda bootstrap env and
let `setup_verl.sh` install the project training environment.

```bash
conda create -n agl-lite-verl python=3.12 -y
conda activate agl-lite-verl
python -m pip install -U pip uv

# Pick the CUDA wheel index that matches the machine: cu126, cu128, cu130, or cpu.
scripts/setup_verl.sh cu128
```

## Quick Start

```bash
# 1. Configure
export AGL_KEY=$(openssl rand -hex 32)

# 2. Run
examples/math-verl/run.sh
```

This will:
1. build the agl-lite and math-poc agent images
2. stop the repo-managed external `agl-vllm` container if it is still running
3. run `agl-lite deploy --env-file examples/math-verl/.env.example`
4. source `.local/agl-lite.env` and export `AGL_NAMESPACE` for VERL cleanup
5. run `examples/math-verl/train.py` (preflight + VERL `run_ppo`)

## Files

| File | Purpose |
|------|---------|
| `train.py` | VERL training entry using `agl_lite.verl.entrypoint.run_ppo(...)` |
| `run.sh` | Launcher: deploy lifecycle + training command |
| `.env.example` | Deploy and training env defaults |

## Notes

- `train.py` reuses math-poc assets:
  - dataset: `examples/math-poc/data/gsm8k_sample.jsonl`
  - job template: `examples/math-poc/job-template.yaml`
  - hooks + gateway config via `.env.example`
