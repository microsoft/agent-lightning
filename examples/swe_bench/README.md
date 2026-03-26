# SWE-bench Example

Run coding agents on [SWE-bench](https://github.com/princeton-nlp/SWE-bench) tasks
using agl-lite for orchestration, with official evaluation and grading.

## Architecture

```
Compute backend (host)              K8s cluster
──────────────────────              ──────────────
agl-lite server                     controller
  + SWEBenchHooks                     ↕
  + swebench package                agent pods (per-instance SWE-bench images)
  + gateway → vLLM                    entrypoint.sh:
                                        1. install + run coding agent
vLLM (code model)                       2. git diff → post patch
                                        3. run eval_script → post test_output
                                      ↕
                                    agl-lite server (grades via on_succeeded hook)
```

**Data flow:**
1. Algorithm sends raw SWE-bench JSONL rows as `input`
2. Hook `on_enqueue`: sets per-instance Docker image, generates eval_script
3. Container: runs agent → eval → posts artifacts (patch + test output)
4. Hook `on_succeeded`: reads test output from disk, grades via `get_eval_report()`
5. Algorithm reads reward events (resolved/not resolved)

## Coding Agents

Two agents included:

- **`mini_swe_agent`** — lightweight Python agent using OpenAI tool_use API.
  Good for testing and OSS models. Not production-quality.
- **`claude_code`** — uses the Claude Code CLI. Requires Claude API access.

Set `AGL_CODING_AGENT=mini_swe_agent` or `AGL_CODING_AGENT=claude_code` in your env.

## Prerequisites

1. **K8s cluster** running (minikube for local dev)
2. **SWE-bench Docker images** pre-built for the sample instances:
   ```bash
   pip install swebench
   python -m swebench.harness.docker_build \
     --dataset_path examples/swe_bench/swebench_samples.jsonl \
     --split dev
   ```
3. **vLLM** running with a code-capable model:
   ```bash
   scripts/start_vllm.sh  # or manually start with your preferred model
   ```

## Quick Start

```bash
# 1. Configure
cp examples/swe_bench/.env.example deploy/.env
export AGL_KEY=$(openssl rand -hex 32)

# 2. Run
examples/swe_bench/run.sh
```

The `run.sh` script:
- Creates a ConfigMap with agent scripts
- Deploys the K8s controller (`--controller-only` mode)
- Starts the agl-lite server on the host (with SWE-bench hooks)
- Runs `rl_loop.py` to enqueue instances and poll for results

## Files

```
examples/swe_bench/
├── rl_loop.py              # Algorithm script (task-agnostic)
├── hooks.py                # SWEBenchHooks (on_enqueue + on_succeeded)
├── run.sh                  # One-command E2E runner
├── Dockerfile.server       # agl-lite + swebench package (for in-cluster mode)
├── gateway-config.yaml     # Gateway routing config
├── job-template.yaml       # K8s pod spec (image overridden per rollout by hook)
├── .env.example            # Environment config template
├── swebench_samples.jsonl  # 5 sample instances for testing
└── agents/
    ├── entrypoint.sh       # Shared entrypoint: agent → eval → post artifacts
    ├── claude_code/        # Claude Code agent scripts
    │   ├── install.sh
    │   ├── run.sh
    │   └── CLAUDE.md
    └── mini_swe_agent/     # Mini SWE agent scripts
        ├── install.sh
        ├── run.sh
        └── agent.py
```

## Configuration

Key environment variables (set in `deploy/.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `AGL_CODING_AGENT` | Agent to use (`mini_swe_agent`, `claude_code`) | `mini_swe_agent` |
| `AGL_MODEL_NAME` | Model name for the agent | — |
| `AGL_MODEL_ENDPOINT` | vLLM endpoint URL | — |
| `AGL_BATCH_SIZE` | Instances per batch | `5` |
| `AGL_TIMEOUT` | Timeout per rollout (seconds) | `5400` (90 min) |
