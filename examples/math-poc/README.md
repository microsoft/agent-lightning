# Math PoC — Mock RL Loop

End-to-end demonstration of agl-lite with a mock model server (mockai) and GSM8K math problems.

## What it does

1. Deploys agl-lite serve + controller + mockai to minikube
2. Runs a 2-iteration RL loop:
   - **Iteration 1**: Register resources + model (v1) → enqueue 5 rollouts → agents solve problems → collect events → compute rewards
   - **Weight update**: Deregister model → re-register as v2 (same endpoint)
   - **Iteration 2**: Enqueue 5 more rollouts → verify events have version=2
3. Verifies: events captured, version tracking works, rewards computed

## How it works

- **Agent** (`qa_agent.py`): reads `AGL_TASK_INPUT`, calls LLM via `OPENAI_BASE_URL`, prints response
- **Model server** (mockai, echo mode): returns the user message verbatim
- **Reward**: algorithm embeds correct/wrong answers in prompts, parses echoed response, compares to ground truth
- **Dataset**: 30 GSM8K problems in `data/gsm8k_sample.jsonl`

## Quick start

```bash
# 1. Configure
cp deploy/.env.example deploy/.env
# Edit deploy/.env (set AGL_K8S_NAMESPACE)

# 2. Set secret
export AGL_KEY=$(openssl rand -hex 32)

# 3. Run everything
examples/math-poc/run.sh
```

## Files

| File | Purpose |
|------|---------|
| `mock_rl_loop.py` | Algorithm script — full RL loop (runs on host) |
| `job-template.yaml` | K8s pod spec for agent jobs |
| `Dockerfile.agent` | Agent container image |
| `k8s-mockai.yaml` | Mockai deployment + service |
| `data/gsm8k_sample.jsonl` | 30 GSM8K problems |
| `run.sh` | One-command: build + deploy + run + verify |
