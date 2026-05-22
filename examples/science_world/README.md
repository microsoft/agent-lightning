# ScienceWorld Example — VERL Training on agl-lite (Local Runner)

Train a small LLM with VERL (PPO/GRPO) to solve text-based science tasks
from AllenAI's [ScienceWorld](https://github.com/allenai/ScienceWorld),
using agl-lite's **local runner** — every rollout runs as a Python
subprocess on this host. **No K8s, no Docker, no minikube.**

This is the local-mode counterpart of [examples/async_calc_x](../async_calc_x/),
intended to be the simplest possible "agent + RL training" loop on
agl-lite.

## TL;DR — launch in 4 steps

```bash
# 1. CUDA / VERL stack (skip if already done for another example).
scripts/setup_verl.sh cu130   # match your driver: cu126/cu128/cu130/cpu

# 2. ScienceWorld runtime — Java + the Python wrapper.
sudo apt-get install -y default-jre
uv pip install scienceworld openai

# 3. Set keys.
export AGL_KEY=$(openssl rand -hex 32)
wandb login                   # optional but recommended

# 4. Launch.
examples/science_world/run.sh           # full training
# or
examples/science_world/run.sh --ci-fast # single PPO step smoke test
```

## Architecture

```text
host (single machine):
  run.sh ─► agl-lite serve  (:8080, host)        background
        ├── store + events + SWHooks
        └── gateway → vLLM (registered at runtime by VERL)

  run.sh ─► agl-lite controller (local runner)   background
        └── LocalReconciler tick: reap → enforce → admit
              └── per rollout: subprocess running SWAgent
                     ├── JVM + ScienceWorldEnv.load + reset
                     ├── multi-turn loop:
                     │     prompt(obs + valid_actions) → LLM → action idx
                     │     env.step(action) → POST "step" event
                     ├── POST "episode_result" event with final_score
                     └── exit 0 (SUCCEEDED) or non-zero (TERMINAL_FAILED)

  run.sh ─► train_sw_agent.py                    foreground
        └── builds (task_name × variation_idx) dataset
        └── run_ppo() → VERL boots Ray + internal vLLM
        └── AglLiteRolloutBridge enqueues rollouts via HTTP
        └── async-rollout: pause/drain at end of each PPO step
```

**Topology facts to internalize before you debug:**

- **No K8s.** `AGL_NAMESPACE=local` is passed only because the controller
  CLI requires it; nothing in local mode touches Kubernetes.
- **`agl-lite serve` and the controller both run on the host** as
  background processes started by `run.sh`. `run.sh`'s `EXIT` trap sends
  `SIGTERM` to both on Ctrl-C / failure.
- **Each rollout starts its own JVM (~2–3 s overhead).** Pool size
  defaults to 16 (`AGL_LOCAL_POOL_SIZE`). Tune to your host's RAM —
  each JVM costs ~200 MB.
- **vLLM is owned by VERL internally** (hybrid mode), exactly like
  `async_calc_x`. Don't start an external vLLM.

## Hardware

| Resource | Minimum | Tested |
|---|---|---|
| GPU | 1 × 80 GB (7B + vLLM + FSDP offload) | 1 × A100 80 GB |
| Host RAM | 64 GB (FSDP offloads optimizer + params to CPU) | 220 GB |
| Host CPUs | 8 (1 per JVM + 4 for VERL) | 24 |
| Disk | 80 GB (model + scienceworld jar) | — |

> The 1.5B model fits on a 24 GB GPU; the 7B default needs 80 GB once
> vLLM (~55% of VRAM) and the FSDP-offloaded actor share the device.
> To shrink to 24 GB, drop `AGL_MODEL_NAME` back to
> `Qwen/Qwen2.5-1.5B-Instruct` and raise `gpu_memory_utilization` to ~0.6.

## Prerequisites

- Python 3.12, `uv`
- **Java 1.8+** (ScienceWorld is Scala-on-JVM): `sudo apt-get install default-jre`
- CUDA driver matching `scripts/setup_verl.sh` (`cu126`/`cu128`/`cu130`)
- `wandb login` if you want online metrics

Install the Python deps (on top of the agl-lite env):

```bash
uv pip install scienceworld openai
```

`scripts/setup_verl.sh` already installs torch / vllm / verl / flash-attn.
If you've never run it, run it once:

```bash
scripts/setup_verl.sh cu130    # adjust for your driver
```

## Configuration: `.env.example`

`run.sh` sources this file. The fields that matter on first launch:

| Variable | Meaning | Default |
|---|---|---|
| `AGL_HOST_PORT` | Port for `agl-lite serve` | `8080` |
| `AGL_LOCAL_POOL_SIZE` | Concurrent rollout subprocesses (JVMs) | `16` |
| `AGL_MODEL_NAME` | HF model id served via the gateway | `Qwen/Qwen2.5-7B-Instruct` |
| `AGL_TASK_NAMES` | Comma-separated names, or `all` for every task | `all` (30 tasks) |
| `AGL_VARIATIONS_PER_TASK` | Max variation indices per task (80/20 train/val split, auto-capped per task at `env.get_max_variations`) | `50` |
| `AGL_SIMPLIFICATION` | `easy` / `medium` / `hard` | `easy` |
| `AGL_NAMESPACE` | Required by CLI, unused in local mode | `local` |

### Dataset sizing with `all` tasks

Several tasks have fewer than 50 variations (e.g. `identify-life-stages-2`
has 10, `power-component` has 20). `build_dataset` caps the per-task
budget at `min(AGL_VARIATIONS_PER_TASK, env.get_max_variations(name))`
and applies the 80/20 split per task — so the `all` preset produces
roughly **~1100 train rows + ~290 val rows** across the 30 tasks.

### Validation cost

`trainer.test_freq=32` runs validation every 32 PPO steps. With ~290
val rows × `rollout.n=4` ≈ 1160 rollouts per validation; at
`AGL_LOCAL_POOL_SIZE=16` and ~10 s/rollout that is roughly **12–15 min
per validation pass**. If that dominates wall-clock time, raise
`test_freq` to `64` or `128`.

To browse the full ScienceWorld task list, see
[ScienceWorld README — task table](https://github.com/allenai/ScienceWorld#tasks).

## Auth keys

Same rules as `async_calc_x`:

- `AGL_KEY` must be exported (or come from `.local/agl-lite.env`).
- `AGL_ADMIN_KEY` is auto-generated by `run.sh` if unset and **must
  differ** from `AGL_KEY` — agent subprocesses carry `AGL_KEY` and must
  not be able to reach the `/admin/gateway/*` surface.

## What `run.sh` does, in order

1. Resolves repo root, creates `logs/<timestamp>/`, exports `AGL_LOG_DIR`.
2. Sources `.env.example`; validates `AGL_KEY`; generates `AGL_ADMIN_KEY` if missing.
3. `ray stop --force` to clean up prior runs.
4. Starts `agl-lite serve` (port `$AGL_HOST_PORT`, gateway-config +
   hooks loaded) as a background process; polls `/healthz` for 40 s.
5. Starts `agl-lite controller --runner-type=local
   --local-pool-size=$AGL_LOCAL_POOL_SIZE
   --local-agent-class=examples.science_world.agents.sw_agent:SWAgent`
   as a background process.
6. `exec`s the trainer with all positional args forwarded.
7. `trap cleanup EXIT` sends `SIGTERM` (then `SIGKILL` after 2 s) to the
   server and controller. In-flight rollout subprocesses are killed by
   the controller's `LocalReconciler._shutdown`.

## CI smoke test

```bash
examples/science_world/run.sh --ci-fast
```

`--ci-fast` sets `total_training_steps=1`, `train_batch_size=4`,
`rollout.n=2`. The full loop completes in a few minutes once the model
is cached.

You can also bypass the LLM entirely for plumbing checks:

```bash
SW_STUB_LLM=1 examples/science_world/run.sh --ci-fast
```

`SW_STUB_LLM=1` makes the agent pick `valid_actions[0]` every turn — the
gateway never sees a real request, so this only validates the controller
↔ agent ↔ hooks loop. Useful when iterating on infrastructure without
burning GPU time.

## Reward shape

ScienceWorld gives an incremental reward per `env.step` and a cumulative
`info["score"]` that ends in `[0, 100]`. This example follows two
choices documented in [the design doc](../../docs/superpowers/specs/2026-05-21-science-world-local-rl-design.md):

- Per-step rewards are **posted as `step` events** for inspection (visible
  via `GET /rollout/<id>/events`).
- The trainer-facing reward is a **single scalar per episode**:
  `final_score / 100.0`, written by `SWHooks.on_succeeded` as a `reward`
  event right after the rollout transitions to SUCCEEDED.

True per-step credit assignment would require trainer-side changes
outside the scope of an example; ScienceWorld's cumulative score is
already a reasonable proxy.

## Monitoring a live run

```bash
# Training progress (rollouts completed)
LOG=$(ls -1d examples/science_world/logs/* | tail -1)
grep "Completed " "$LOG/training.log" | tail -5

# Controller decisions (admit / reap / enforce)
tail -f "$LOG/controller.log"

# Server / gateway (JSON, structlog)
tail -f "$LOG/server.log"

# Per-attempt agent logs (subprocess stdout/stderr go to the controller log,
# but the agent also writes /tmp/agl-lite/logs/<attempt_id>/agent.log
# (controlled by AGL_LOG_DIR in the worker env)).
ls /tmp/agl-lite/logs/

# Health snapshot
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
ps -o pid,rss,comm -C java | head     # one JVM per active rollout
```

## Stopping cleanly

`Ctrl-C` on `run.sh` runs the cleanup trap. If `run.sh` exited
abnormally, kill leftover processes manually:

```bash
pkill -TERM -f "agl-lite controller"
pkill -TERM -f "agl-lite serve"
pkill -TERM -f "local_worker"           # rollout subprocesses
pkill -TERM -f java                     # JVMs (careful — kills ALL java)
.venv/bin/ray stop --force
```

## Files

| File | Description |
|---|---|
| [agents/sw_agent.py](agents/sw_agent.py) | `SWAgent` — multi-turn LLM loop against `ScienceWorldEnv`. Loaded by `local_worker`. |
| [hooks.py](hooks.py) | `SWHooks` — passthrough `on_enqueue`; `on_succeeded` writes `reward = final_score / 100`. |
| [train_sw_agent.py](train_sw_agent.py) | Trainer entrypoint — builds dataset, builds VERL config, calls `run_ppo`. |
| [gateway-config.yaml](gateway-config.yaml) | Injects `return_token_ids: true` for RL token capture. |
| [run.sh](run.sh) | E2E launcher — starts serve + controller + trainer. |
| [.env.example](.env.example) | Default config sourced by `run.sh`. |

## Design

See [docs/superpowers/specs/2026-05-21-science-world-local-rl-design.md](../../docs/superpowers/specs/2026-05-21-science-world-local-rl-design.md)
for the full design — components, data flow, error handling, and the
rationale for each decision (per-rollout JVM vs pool, constrained-action
prompt, final-score reward, etc.).

## Troubleshooting

### `ImportError: scienceworld`

```bash
uv pip install scienceworld
```

### `Could not find Java executable`

ScienceWorld needs a JRE.

```bash
sudo apt-get install -y default-jre
java -version    # must report 1.8 or newer
```

### Rollouts immediately TERMINAL_FAILED

Inspect the controller log — the worker's traceback goes to its stderr,
which is inherited by the controller process:

```bash
LOG=$(ls -1d examples/science_world/logs/* | tail -1)
grep -A 20 "Traceback" "$LOG/controller.log"
```

Common causes:

- ScienceWorld task name typo — `env.load("foo", 0, "easy")` raises if
  the task name doesn't match `env.get_task_names()`.
- JVM OOM on a small host — lower `AGL_LOCAL_POOL_SIZE`.

### Trainer stuck at `Completed 0/N unfinished=N`

The agent or controller died. Check `controller.log` and the per-attempt
log directory (`/tmp/agl-lite/logs/local-<rollout_id>/agent.log`).

### `pkill -f java` killed unrelated processes

It kills *all* Java processes on the host. If you have other JVMs you
care about, use `pgrep -f "scienceworld\|py4j"` instead and inspect the
list before killing.
