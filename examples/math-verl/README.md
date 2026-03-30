# Math VERL — training E2E on agl-lite

This example is the **training** path for math tasks using VERL + agl-lite.

- `examples/math-poc/` remains for **task delegation + data pipeline verification**.
- `examples/math-verl/` is for **training loop integration** (Phase 5c).

## Scope

- vLLM-only training path (no mock mode)
- K8s/minikube lifecycle is external
- Script launches `agl-lite serve` on host and runs VERL training

## Prerequisites

- Controller already running in K8s (recommended: `scripts/deploy.sh --agl-in-host`)
- vLLM backend available for VERL rollout workers
- `AGL_KEY` exported
- VERL dependencies installed in current Python env

## Run

```bash
examples/math-verl/run.sh
```

This will:
1. start `agl-lite serve` with math vLLM hooks + gateway config
2. run `examples/math-verl/train.py` (preflight + VERL `run_ppo`)
3. stop `agl-lite serve` on exit

## Files

| File | Purpose |
|------|---------|
| `train.py` | VERL training entry using `agl_lite.verl.entrypoint.run_ppo(...)` |
| `run.sh` | Host-side launcher: serve lifecycle + training command |
| `.env.example` | Training-specific env defaults |

## Notes

- `train.py` reuses math-poc assets:
  - dataset: `examples/math-poc/data/gsm8k_sample.jsonl`
  - job template: `examples/math-poc/vllm/job-template.yaml`
  - hooks + gateway config via `run.sh`
