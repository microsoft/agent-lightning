# Math PoC — GSM8K data pipeline verification

End-to-end demonstration of agl-lite task delegation + data capture for math rollouts.
Two modes: **mock** (CPU-only, deterministic) and **vllm** (real GPU inference).

> Scope: this folder is for rollout orchestration and event/data pipeline verification.
> For VERL training integration, use `examples/math-verl/`.

---

## Architecture

### Mock mode (Phase 4a — CPU-only)

Everything runs on a single machine with minikube. No GPU required.

```
┌─ minikube ────────────────────────────────────────┐
│                                                   │
│  agl-lite serve    (Deployment)  ← HTTP API       │
│  agl-controller    (Deployment)  ← creates Jobs   │
│  mockai            (Deployment)  ← echo server    │
│                                                   │
│  agent pods        (Jobs)        ← created per    │
│                                    rollout        │
└──────────────────────┬────────────────────────────┘
                       │ port-forward :8080
                       ▼
┌─ host ────────────────────────────────────────────┐
│  mock_rl_loop.py   ← algorithm (enqueue, reward)  │
└───────────────────────────────────────────────────┘
```

- **mockai** (echo mode): returns the user message verbatim. The algorithm
  embeds `\boxed{answer}` in the input to produce deterministic correct/wrong
  rewards (alternating pattern, avg reward = 0.60).
- **Weight update**: deregister model → re-register with version+1. No actual
  model reload — version is metadata in agl-lite store.
- **Result**: fully deterministic, assertion-checked. See `reference_output.log`.

### vLLM mode (Phase 4b — GPU)

agl-lite and vLLM run on the host (colocated). Only the controller and agent
pods run in minikube. This minimizes network hops: gateway → vLLM is localhost.

```
┌─ minikube ────────────────────────────────────────┐
│                                                   │
│  agl-controller    (Deployment)  ← creates Jobs   │
│                                                   │
│  agent pods        (Jobs)   ─── OPENAI_BASE_URL ──┼──┐
│                                                   │  │
└───────────────────────────────────────────────────┘  │
                                                       │ host.minikube.internal:8080
                                                       ▼
┌─ host ────────────────────────────────────────────┐
│                                                   │
│  agl-lite serve    (process :8080)                │
│    └─ gateway ──→ vLLM (localhost:8010)            │
│                                                   │
│  rl_loop.py        ← algorithm (localhost:8080)   │
│                                                   │
│  vLLM (Docker)     ← Qwen2.5-1.5B-Instruct       │
│   :8010              (vllm/vllm-openai:latest)    │
└───────────────────────────────────────────────────┘
```

- **agl-lite** runs on host — no port-forward needed for the algorithm.
- **Gateway → vLLM** is `localhost:8010` — no cross-network hop.
- **Agent pods** reach agl-lite via `host.minikube.internal:8080`.
- **Gateway config** injects `return_token_ids: true` into all requests
  (needed for RL training — prompt + response token IDs).
- **Model is frozen** in Phase 4b — no actual weight updates.

---

## Event flow

Each rollout produces three events:

```
model_request (auto, gateway)  →  agent_output (agent)  →  reward (algorithm)
```

| Event | Source | Data |
|-------|--------|------|
| `model_request` | gateway (auto) | `{request, response (SSE chunks), server: {model, version, endpoint}}` |
| `agent_output` | agent pod | `{answer, raw_response}` |
| `reward` | algorithm | `{value, ground_truth, agent_answer, reason}` |

---

## Quick start

### Prerequisites

- minikube running (`minikube status`)
- For vLLM mode: Docker + `nvidia-container-toolkit`

### Python / Conda Environment

Use Python 3.12 with the core agl-lite controller dependencies. This example
does not need the VERL extra.

```bash
conda create -n agl-lite-math-poc python=3.12 -y
conda activate agl-lite-math-poc
python -m pip install -U pip uv
uv sync --extra controller
```

Mock mode runs with only CPU/K8s dependencies. vLLM mode still uses this same
Python environment; the model server is started separately by
`scripts/start_vllm.sh` in Docker.

### 1. Configure

```bash
# API key
export AGL_KEY=$(openssl rand -hex 32)

# Optional: adjust mode-specific configs
$EDITOR examples/math-poc/mock/.env.example
$EDITOR examples/math-poc/vllm/.env.example
```

Each mode keeps deploy and experiment settings in
`examples/math-poc/<mode>/.env.example`. `run.sh` passes that file to
`agl-lite deploy --env-file` and reads `AGL_NAMESPACE` from it.

### 2. Run (mock mode)

```bash
examples/math-poc/run.sh mock
```

Builds images, deploys to minikube, runs 2-iteration RL loop, verifies results.
Logs saved to `examples/math-poc/logs/<timestamp>/`.

### 3. Run (vLLM mode)

```bash
# Start vLLM (once, runs in background)
scripts/start_vllm.sh

# Run experiment (repeatable)
examples/math-poc/run.sh

# Stop vLLM when done
scripts/start_vllm.sh --stop
```

### 4. Verify

Compare against reference outputs:

```bash
# Mock mode:
latest_mock_log=$(ls -td examples/math-poc/logs/*-mock/rl_loop.log | head -n 1)
grep -E '^  (--- Iteration|Rollouts:|Events:|Average reward:|Iterations:|Accuracy:|Iter [0-9]|Checks:|Math PoC)|^    \[PASS\]' "$latest_mock_log" \
  | diff - examples/math-poc/reference_output.log

# vLLM mode (structure matches; LLM reasoning text may vary):
latest_vllm_log=$(ls -td examples/math-poc/logs/*-vllm/rl_loop.log | head -n 1)
grep -E '^  (--- Iteration|Rollouts:|Events:|Average reward:|Iterations:|Accuracy:|Iter [0-9]|Checks:|Math PoC)|^    \[PASS\]' "$latest_vllm_log" \
  | diff - examples/math-poc/reference_output_vllm.log
```

For mock mode the output is fully deterministic. For vLLM mode, the model's
reasoning text varies but the structure (events, checks, rewards) should match.

---

## Files

| File | Purpose |
|------|---------|
| `mock_rl_loop.py` | Mock algorithm — deterministic E2E test (mockai echo mode) |
| `rl_loop.py` | Real algorithm — GSM8K with vLLM (numeric reward) |
| `agents/qa_agent.py` | Agent — streaming LLM call + `\boxed{}` parsing |
| `job-template.yaml` | K8s pod spec for agent Jobs |
| `Dockerfile.agent` | Agent container image |
| `.dockerignore` | Docker build-context ignore rules for the agent image |
| `k8s-mockai.yaml` | Mockai deployment + service (mock mode only) |
| `data/gsm8k_sample.jsonl` | 30 GSM8K problems with ground truth |
| `mock/.env.example` | Complete config for mock mode |
| `vllm/.env.example` | Complete config for vLLM mode |
| `run.sh` | One-command: build → deploy → run → verify → collect logs |
| `reference_output.log` | Expected output — mock mode (redacted IDs) |
| `reference_output_vllm.log` | Expected output — vLLM mode (redacted IDs) |
| `logs/` | Per-run logs (gitignored) |

For training integration (VERL), see `examples/math-verl/`. 

---

## Environment details

Our development/test environment:

| Component | Spec |
|-----------|------|
| **Host** | 128 CPU, 995 GB RAM, 4× NVIDIA RTX A6000 (48 GB each) |
| **Minikube** | Docker driver, single node, no GPU passthrough |
| **vLLM** | Docker container on host, GPU 0, `Qwen/Qwen2.5-1.5B-Instruct` |
| **Docker image** | `vllm/vllm-openai:latest` (bundles CUDA runtime) |
| **K8s namespace** | `agl-test` (configurable via the mode `.env.example`) |
| **Agent pods** | CPU-only in minikube, reach vLLM via `host.minikube.internal:8010` |

Note: pip-installing vLLM on the host failed due to triton/gcc compilation
issues with CUDA 13.0. The Docker image works reliably and is the recommended
approach.
