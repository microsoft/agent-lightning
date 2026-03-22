# Math PoC — GSM8K with agl-lite

End-to-end demonstration of agl-lite running math problem rollouts with GSM8K.
Two modes: **mock** (CPU-only, deterministic) and **vllm** (real GPU inference).

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

vLLM runs on the host as a Docker container with GPU access. agl-lite infra
runs in minikube. Agent pods reach vLLM through the gateway, which forwards
to `host.minikube.internal`.

```
┌─ minikube ────────────────────────────────────────┐
│                                                   │
│  agl-lite serve    (Deployment)  ← HTTP API       │
│  agl-controller    (Deployment)  ← creates Jobs   │
│                                                   │
│  agent pods        (Jobs)   ─── gateway ──────────┼──┐
│                                                   │  │
└──────────────────────┬────────────────────────────┘  │
                       │ port-forward :8080             │
                       ▼                               ▼
┌─ host ────────────────────────────────────────────┐
│  rl_loop.py        ← algorithm (enqueue, reward)  │
│                                                   │
│  vLLM (Docker)     ← Qwen2.5-1.5B-Instruct       │
│   :8010              real inference on GPU         │
│   (vllm/vllm-openai:latest)                      │
└───────────────────────────────────────────────────┘
```

- **vLLM** runs via Docker (`vllm/vllm-openai:latest`) with NVIDIA GPU access.
  Started separately with `scripts/start_vllm.sh`.
- **Agent** sends `stream=True` requests. Gateway captures all SSE chunks as
  `model_request` events, then forwards to the agent.
- **Reward**: parse model's `\boxed{answer}`, compare to ground truth
  numerically (handles `18` vs `18.0`, `$18`, `18,000`, etc.).
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

### 1. Configure

```bash
# Pick one:
cp examples/math-poc/.env.mockai.example deploy/.env   # mock (CPU-only)
cp examples/math-poc/.env.vllm.example deploy/.env     # vLLM (GPU)

# API key
export AGL_KEY=$(openssl rand -hex 32)
```

Each `.env` file is self-contained — infrastructure, model server, and
experiment settings all in one place. Just copy and go.

### 2. Run (mock mode)

```bash
examples/math-poc/run.sh
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
sed -E 's/[0-9a-f]{32}/<rollout-id>/g' examples/math-poc/logs/*/mock_rl_loop.log \
  | diff - examples/math-poc/reference_output.log

# vLLM mode (structure matches; LLM reasoning text may vary):
diff <(grep -E '^\s*(✅|❌|Rollouts|Events|Accuracy|Iter |Checks:)' examples/math-poc/logs/*/rl_loop.log) \
     <(grep -E '^\s*(✅|❌|Rollouts|Events|Accuracy|Iter |Checks:)' examples/math-poc/reference_output_vllm.log)
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
| `k8s-mockai.yaml` | Mockai deployment + service (mock mode only) |
| `data/gsm8k_sample.jsonl` | 30 GSM8K problems with ground truth |
| `.env.mockai.example` | Complete config for mock mode → `cp` to `deploy/.env` |
| `.env.vllm.example` | Complete config for vLLM mode → `cp` to `deploy/.env` |
| `run.sh` | One-command: build → deploy → run → verify → collect logs |
| `reference_output.log` | Expected output — mock mode (redacted IDs) |
| `reference_output_vllm.log` | Expected output — vLLM mode (redacted IDs) |
| `logs/` | Per-run logs (gitignored) |

---

## Environment details

Our development/test environment:

| Component | Spec |
|-----------|------|
| **Host** | 128 CPU, 995 GB RAM, 4× NVIDIA RTX A6000 (48 GB each) |
| **Minikube** | Docker driver, single node, no GPU passthrough |
| **vLLM** | Docker container on host, GPU 0, `Qwen/Qwen2.5-1.5B-Instruct` |
| **Docker image** | `vllm/vllm-openai:latest` (bundles CUDA runtime) |
| **K8s namespace** | `agl-test` (configurable via `deploy/.env`) |
| **Agent pods** | CPU-only in minikube, reach vLLM via `host.minikube.internal:8010` |

Note: pip-installing vLLM on the host failed due to triton/gcc compilation
issues with CUDA 13.0. The Docker image works reliably and is the recommended
approach.
