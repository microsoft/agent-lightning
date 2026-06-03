# ScienceWorld Example — VERL Training on agl-lite (Local Runner)

Train an LLM with VERL (PPO/GRPO) to solve text-based science tasks from
AllenAI's [ScienceWorld](https://github.com/allenai/ScienceWorld), using
agl-lite's **local runner** — every rollout runs as a short-lived Python
subprocess on this host. **No K8s, no Docker, no minikube.**

This mirrors [examples/calc_x](../calc_x/) `run_local.sh`: the agent is a
standalone script that talks to the agl-lite gateway over OpenAI-compatible
HTTP and posts a single `reward` event; the gateway transparently captures
token IDs for RL training.

## Architecture

```text
run_local.sh ─► agl-lite-server (:8080)                     background
            ├── store + events + gateway → vLLM (booted by VERL)
            │
            ├─► agl-lite-controller runner_type=local        background
            │     └── LocalReconciler: one subprocess per rollout
            │           └── SWAgent.run():
            │                 ├── env vars injected via local.env_map
            │                 │     (TASK_NAME / VARIATION_IDX / SIMPLIFICATION)
            │                 ├── ScienceWorldEnv.load + reset
            │                 ├── loop: prompt(obs + valid actions) → LLM
            │                 │         → action idx → env.step
            │                 └── POST "reward" event = final_score / 100
            │
            └─► train_sw_agent.py                             foreground
                  ├── builds (task_name × variation_idx) dataset
                  ├── run_ppo() → VERL boots Ray + internal vLLM
                  └── AglLiteRolloutBridge enqueues rollouts over HTTP
```

Key facts:

- **No K8s.** The controller runs locally and spawns one subprocess per rollout.
- **vLLM is owned by VERL internally** — don't start an external vLLM.
- **Train vs val temperature** is applied server-side by the proxy from the
  `/mode/{train|val}/` URL segment; the agent does not set temperature.
- **Reward** is posted directly by the agent as a `reward` event and consumed
  by the VERL bridge — no hooks file required.

## Requirements

- Single node with at least one 40GB GPU
- `uv` installed, Python 3.12+
- Java 1.8+ (ScienceWorld runs a JVM per rollout)

## Setup

```bash
# 1. CUDA / VERL stack (skip if already done for another example).
scripts/setup_verl.sh cu128   # match your driver: cu126/cu128/cu130/cpu

# 2. ScienceWorld runtime — Java + the Python wrappers.
sudo apt-get install -y default-jre
uv pip install scienceworld openai
```

## Quick Start

```bash
# Full training
examples/science_world/run_local.sh

# Short CI smoke test
examples/science_world/run_local.sh --ci
```

`run_local.sh` starts `agl-lite-server` and `agl-lite-controller
runner_type=local`, waits for `/healthz`, then runs training in the
foreground. All processes are torn down on exit.

## Standalone Training

If agl-lite server + local controller are already running:

```bash
python examples/science_world/train_sw_agent.py \
    --task-names find-non-living-thing,find-living-thing \
    --variations-per-task 50 \
    --agl-base-url http://localhost:8080 \
    --agl-key dummy
```

## Files

| File | Description |
|------|-------------|
| `agents/sw_agent.py` | Standalone agent — drives ScienceWorldEnv + LLM, posts `reward` |
| `train_sw_agent.py` | Builds the dataset and VERL config, calls `run_ppo()` |
| `run_local.sh` | E2E local entrypoint — server + local controller + training |

## Configuration

`train_sw_agent.py` CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--task-names` | `all` | Comma-separated task names, or `all` |
| `--variations-per-task` | `50` | Max variations/task (auto-capped per task) |
| `--simplification` | `easy` | ScienceWorld simplification preset |
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | HF model id or path |
| `--agl-base-url` | `http://localhost:8080` | agl-lite server URL |
| `--agl-key` | `""` | agl-lite API key |
| `--run-name` | _none_ | Suffix appended to the experiment name |
| `--ci` | off | Short CI-style training loop |

SWAgent runtime knobs (environment variables, read by the rollout subprocess):

| Var | Default | Description |
|-----|---------|-------------|
| `SW_MAX_STEPS` | `30` | Max LLM turns per episode |
| `SW_ENV_STEP_LIMIT` | `100` | `envStepLimit` passed to ScienceWorldEnv |
| `SW_MAX_VALID_ACTIONS_SHOWN` | `50` | Max valid actions shown to the LLM |
| `SW_OBS_SNIPPET_CHARS` | `240` | Observation snippet length in logs |
| `AGL_MAX_TOKENS` | `256` | Max tokens per LLM completion |
