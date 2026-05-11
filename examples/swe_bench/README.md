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

## Coding Agent

Uses **Claude Code** (via the Claude CLI) as the coding agent. The agent receives
the problem statement, explores the repository, and attempts to fix the bug.

Configuration via `AGL_CODING_AGENT=claude_code` (default). To add a new agent,
create `agents/<name>/install.sh` + `agents/<name>/run.sh` and update `run.sh`
to include the files in the ConfigMap.

## Prerequisites

1. **K8s cluster** running (minikube for local dev)
2. **SWE-bench Docker images** pre-built for the sample instances:
   ```bash
  # Build directly inside minikube's Docker daemon so the rollout pods can see them.
  eval "$(minikube -p minikube docker-env)"
   .venv/bin/python examples/swe_bench/build_images.py --limit 1
  eval "$(minikube -p minikube docker-env -u)"
   ```
3. **vLLM** running with a code-capable model:
   ```bash
   eval "$(minikube -p minikube docker-env -u)"  # vLLM needs host Docker + GPU access
   scripts/start_vllm.sh  # or manually start with your preferred model
   ```

## Python / Conda Environment

Use Python 3.12 with agl-lite controller dependencies plus `swebench`. The
server loads `examples/swe_bench/hooks.py`, and those hooks import the
`swebench` package for test-spec generation and grading.

```bash
conda create -n agl-lite-swebench python=3.12 -y
conda activate agl-lite-swebench
python -m pip install -U pip uv
uv sync --extra controller
uv pip install swebench
```

This example does not need the VERL extra. vLLM is normally run separately in
Docker via `scripts/start_vllm.sh`, and the coding agent is installed inside the
agent pod by the scripts under `agents/`.

## Quick Start

```bash
# 1. Configure
export AGL_KEY=$(openssl rand -hex 32)
$EDITOR examples/swe_bench/.env.example

# 2. Run
examples/swe_bench/run.sh
```

The default `.env.example` runs one SWE-bench instance as a smoke test. Increase
`AGL_BATCH_SIZE` after building the corresponding images.

The `run.sh` script:
- Checks the required SWE-bench images and prints the build command if any are missing
- Runs `agl-lite deploy --env-file examples/swe_bench/.env.example`
- Creates a ConfigMap with agent scripts
- Launches agl-lite on host in `agl-in-host` mode with SWE-bench hooks
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
├── .dockerignore           # Docker build-context ignore rules
├── .env.example            # Environment config template
├── swebench_samples.jsonl  # 5 sample instances for testing
└── agents/
    ├── entrypoint.sh       # Shared entrypoint: agent → eval → post artifacts
    └── claude_code/        # Claude Code agent scripts
        ├── install.sh
        ├── run.sh
        ├── handle_hook.sh
        └── CLAUDE.md
```

## Configuration

Key environment variables (set in `examples/swe_bench/.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `AGL_NAMESPACE` | K8s namespace for agl-lite and agent Jobs | `agl-swebench` |
| `AGL_MODE` | Deploy topology | `agl-in-host` |
| `AGL_CODING_AGENT` | Agent to use (`claude_code`) | `claude_code` |
| `AGL_MODEL_NAME` | Model name for the agent | — |
| `AGL_MODEL_ENDPOINT` | vLLM endpoint URL | — |
| `AGL_BATCH_SIZE` | Instances per batch | `5` |
| `AGL_TIMEOUT` | Timeout per rollout (seconds) | `5400` (90 min) |
