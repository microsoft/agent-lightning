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
└──────────────────────┬───────────────────────────-┘
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

vLLM runs on the host with GPU access. Minikube pods reach it via
`host.minikube.internal`.

```
┌─ minikube ────────────────────────────────────────┐
│                                                   │
│  agl-lite serve    (Deployment)  ← HTTP API       │
│  agl-controller    (Deployment)  ← creates Jobs   │
│                                                   │
│  agent pods        (Jobs)   ─── gateway ──────────┼──┐
│                                                   │  │
└──────────────────────┬────────────────────────────┘  │
                       │ port-forward :8080            │
                       ▼                               ▼
┌─ host ────────────────────────────────────────────┐
│  rl_loop.py        ← algorithm (enqueue, reward)  │
│  vLLM :8001        ← Qwen2.5-1.5B-Instruct        │
│                      (4× A6000, real inference)   │
└───────────────────────────────────────────────────┘
```

- **vLLM** generates real math reasoning with `\boxed{}` answers.
- **Reward**: parse model's `\boxed{answer}`, compare to ground truth numerically.
- **No mockai** needed — real model server registered directly.
- **Model is frozen** in Phase 4b (no actual weight updates). Version bump is
  metadata-only to validate the tracking path.

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
| `reward` | algorithm | `{value, ground_truth, agent_answer}` |

---

## Quick start

### Prerequisites

- minikube running (`minikube status`)
- For vLLM mode: NVIDIA GPU with vLLM installed

### 1. Configure

```bash
# Mock mode (CPU-only, deterministic):
cp examples/math-poc/.env.mockai.example deploy/.env

# OR vLLM mode (real GPU inference):
cp examples/math-poc/.env.vllm.example deploy/.env

# API key (set once, used by all components)
export AGL_KEY=$(openssl rand -hex 32)
```

Each `.env` file is self-contained — infrastructure settings, model server
config, and experiment parameters all in one place.

### 2. Run (mock mode)

```bash
examples/math-poc/run.sh
```

### 3. Run (vLLM mode)

```bash
# Start vLLM on host (separate terminal)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 --host 0.0.0.0

# Run (uses settings from deploy/.env)
examples/math-poc/run.sh
```

### 4. Verify

Compare your output against the reference:
```bash
# Redact rollout IDs from your log
sed -E 's/[0-9a-f]{32}/<rollout-id>/g' examples/math-poc/logs/*/mock_rl_loop.log \
  | diff - examples/math-poc/reference_output.log
```

---

## Files

| File | Purpose |
|------|---------|
| `mock_rl_loop.py` | Mock algorithm — deterministic E2E test with mockai |
| `rl_loop.py` | Real algorithm — GSM8K with vLLM (Phase 4b) |
| `agents/qa_agent.py` | Agent — LLM call + `\boxed{}` parsing (streaming) |
| `job-template.yaml` | K8s pod spec for agent Jobs |
| `Dockerfile.agent` | Agent container image |
| `k8s-mockai.yaml` | Mockai deployment + service (mock mode only) |
| `data/gsm8k_sample.jsonl` | 30 GSM8K problems with ground truth |
| `.env.mockai.example` | Complete config for mock mode → copy to `deploy/.env` |
| `.env.vllm.example` | Complete config for vLLM mode → copy to `deploy/.env` |
| `run.sh` | One-command: build → deploy → run → verify → collect logs |
| `reference_output.log` | Expected output (mock mode, redacted IDs) |
| `logs/` | Per-run logs (gitignored) |

---

## Environment details

Our development environment:

| Component | Spec |
|-----------|------|
| Host | 128 CPU, 995 GB RAM, 4× NVIDIA RTX A6000 (48 GB each) |
| minikube | Docker driver, single node, no GPU passthrough |
| vLLM | Host-side, one A6000, `Qwen/Qwen2.5-1.5B-Instruct` |
| K8s namespace | `agl-test` (configurable via `deploy/.env`) |
| Agent pods | CPU-only (run inside minikube, reach vLLM via `host.minikube.internal`) |
