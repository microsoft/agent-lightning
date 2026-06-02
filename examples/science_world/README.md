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
host (single machine, 8×A100 tested):
  run.sh ─► agl-lite serve  (:$AGL_HOST_PORT)              background
        │   (NUMA-pinned on multi-NUMA hosts)
        ├── store + events + SWHooks
        └── gateway → vLLM (registered at runtime by VERL)

  run.sh ─► agl-lite controller (local runner)             background
        └── LocalReconciler tick: reap → enforce → admit
              └── per rollout: subprocess running SWAgent
                     ├── env injected: AGL_IS_TRAIN (1/0), SW_*, AGL_TEMPERATURE_*
                     ├── JVM + ScienceWorldEnv.load + reset
                     ├── multi-turn loop:
                     │     prompt(obs + valid_actions) → LLM → action idx
                     │     env.step(action) → POST "step" event
                     ├── POST "episode_result" event with final_score
                     └── exit 0 (SUCCEEDED) or non-zero (FAILED)

  run.sh ─► train_sw_agent.py                              foreground
        └── builds (task_name × variation_idx) dataset
        └── run_ppo() → VERL boots Ray + internal vLLM (hybrid, TP=2×DP=4)
        └── AglLiteRolloutBridge enqueues rollouts via HTTP
              ├── train batch: rollout.n=4 (GRPO group), is_train=True
              └── val batch:   rollout.n=1,             is_train=False
        └── async-rollout: pause/drain at end of each PPO step
```

**Topology facts to internalize before you debug:**

- **No K8s.** `AGL_NAMESPACE=local` is passed only because the controller
  CLI requires it; nothing in local mode touches Kubernetes.
- **`agl-lite serve` and the controller both run on the host** as
  background processes started by `run.sh`. `run.sh`'s `EXIT` trap sends
  `SIGTERM` to both on Ctrl-C / failure.
- **Each rollout starts its own JVM (~2–3 s overhead).** Pool size
  defaults to 64 (`AGL_LOCAL_POOL_SIZE`) on the 8×A100 host — each JVM
  costs ~200 MB; shrink on smaller hosts.
- **vLLM is owned by VERL internally** (hybrid mode), exactly like
  `async_calc_x`. Don't start an external vLLM.
- **Train and val rollouts diverge by env var, not code path.** The
  controller injects `AGL_IS_TRAIN=1|0` per rollout from
  `rollout.metadata.is_train`; the agent uses it to pick
  `AGL_TEMPERATURE_TRAIN` vs `AGL_TEMPERATURE_VAL` for sampling.

## Hardware

| Resource | Minimum | Tested |
|---|---|---|
| GPU | 1 × 80 GB (7B + vLLM + FSDP offload) | **8 × A100-SXM4-40 GB** (hybrid: TP=2 × DP=4) |
| Host RAM | 64 GB (FSDP offloads optimizer + params to CPU) | 220 GB |
| Host CPUs | 8 (1 per JVM + 4 for VERL) | 96 (4 NUMA nodes) |
| Disk | 80 GB (model + scienceworld jar) | — |

The default config in [train_sw_agent.py](train_sw_agent.py) (`tensor_model_parallel_size=2`, `n_gpus_per_node=8`, `gpu_memory_utilization=0.75`) targets the 8×A100-40GB host. On a single 80 GB GPU drop `n_gpus_per_node=1`, `tensor_model_parallel_size=1`, and proportionally shrink `train_batch_size` / `async_train_batch_size`.

> To shrink to a 24 GB GPU, drop `AGL_MODEL_NAME` back to
> `Qwen/Qwen2.5-1.5B-Instruct` and raise `gpu_memory_utilization` to ~0.6.

### NUMA pinning (multi-NUMA hosts only)

The tested host has 8 GPUs spread across 4 NUMA nodes (GPU0/1 → NUMA 1,
GPU2/3 → NUMA 0, GPU4/5 → NUMA 3, GPU6/7 → NUMA 2). `run.sh` auto-detects
the NUMA layout via `numactl --hardware`:

- If `available: >=2 nodes`, pin `agl-lite serve` to NUMA node 2 with
  `numactl --cpunodebind=2 --membind=2` so the server stops bouncing
  across nodes.
- The **controller is left un-pinned** so the rollout subprocess pool
  (64 JVMs at `AGL_LOCAL_POOL_SIZE=64`) spreads across all cores instead
  of crowding one node.
- On single-NUMA boxes (e.g. dev laptops) pinning is skipped entirely.

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
| `AGL_HOST_PORT` | Port for `agl-lite serve` | `18080` |
| `AGL_LOCAL_POOL_SIZE` | Concurrent rollout subprocesses (JVMs) | `64` |
| `AGL_N_GPUS_PER_NODE` | GPUs VERL is allowed to use (FSDP + vLLM placement) | `8` |
| `AGL_MODEL_NAME` | HF model id served via the gateway | `Qwen/Qwen2.5-7B-Instruct` |
| `AGL_TASK_NAMES` | Comma-separated names, or `all` for every task | `all` (30 tasks) |
| `AGL_VARIATIONS_PER_TASK` | Max variation indices per task (80/20 train/val split, auto-capped per task at `env.get_max_variations`) | `100` |
| `AGL_SIMPLIFICATION` | ScienceWorld simplification flags (see below) | `easy` |
| `AGL_NAMESPACE` | Required by CLI, unused in local mode | `local` |

### `AGL_SIMPLIFICATION` — ScienceWorld difficulty flags

ScienceWorld's `env.load(task, variation, simplificationStr)` accepts a
**comma-separated string of simplification flags** (no spaces). Valid
flags, per `env.get_possible_simplifications()`:

| Flag | Effect |
|---|---|
| `teleportAction` | Adds actions to teleport directly to any location (skips walking). |
| `selfWateringFlowerPots` | Flower pots water themselves — plants won't die. |
| `openContainers` | Containers (drawers, cupboards, …) start open. |
| `openDoors` | Doors start open. |
| `noElectricalAction` | Drops `connect X to Y` actions — shrinks the action space (incompatible with electrical tasks). |
| `easy` | Alias for all 5 flags above. |

Common values for `AGL_SIMPLIFICATION`:

- `easy` — fastest path, all 5 flags on (the example's default).
- `""` (empty string) — original difficulty, no flags. This is what
  ScienceWorld calls "hard" colloquially; pass the empty string, not
  the word `hard` (`env.load` rejects unknown values).
- Any custom comma-separated subset, e.g.
  `teleportAction,openDoors,openContainers` for a middle-ground setting.

There is no `medium` / `hard` preset in ScienceWorld — passing those
literal strings will raise `Unknown simplification`.

### SWAgent runtime (per-rollout subprocess)

Read by [agents/sw_agent.py](agents/sw_agent.py); each subprocess inherits these from the controller env.

| Variable | Meaning | Default |
|---|---|---|
| `SW_MAX_STEPS` | Max LLM turns per episode (outer agent loop) | `100` |
| `SW_ENV_STEP_LIMIT` | `envStepLimit` passed to `ScienceWorldEnv` | `100` |
| `SW_MAX_VALID_ACTIONS_SHOWN` | Cap on valid-action list shown to the LLM (also caps parsed action idx) | `50` |
| `SW_OBS_SNIPPET_CHARS` | Length of obs snippet stored in each `step` event | `240` |

### LLM sampling — train vs val temperature split

| Variable | Meaning | Default |
|---|---|---|
| `AGL_TEMPERATURE_TRAIN` | Sampling temperature for train rollouts (exploration) | `1.0` |
| `AGL_TEMPERATURE_VAL` | Sampling temperature for val rollouts (deterministic eval) | `0.0` |
| `AGL_MAX_TOKENS` | Max tokens generated per LLM call | `256` |

`SWAgent` picks the right one via `AGL_IS_TRAIN` — injected per-rollout
by the local controller from `rollout.metadata.is_train` (see
[agl_lite/controller/local_reconciler.py](../../agl_lite/controller/local_reconciler.py)
`_build_worker_env`). The bridge sets `is_train=False` only for the val
batch ([rollout_bridge.py](../../agl_lite/verl/rollout_bridge.py)
`_async_register_and_enqueue`), so val rollouts deterministically use
temperature 0 while training rollouts keep exploring at temperature 1.

### Dataset sizing with `all` tasks

Several tasks have fewer than 100 variations (e.g. `identify-life-stages-2`
has 10, `power-component` has 20). `build_dataset` caps the per-task
budget at `min(AGL_VARIATIONS_PER_TASK, env.get_max_variations(name))`
and applies the 80/20 split per task — so the `all` preset with the
default `AGL_VARIATIONS_PER_TASK=100` produces roughly **~2200 train
rows + ~580 val rows** across the 30 tasks (exact counts depend on
per-task variation caps).

### Validation cost & frequency

`trainer.test_freq=32` runs validation every 32 PPO steps **plus once on
the last step**. With the default 8×A100 config the trainer does:

- ~2200 train rows / `train_batch_size=64` ≈ **34 PPO steps per epoch**
- `total_epochs=2` → ~**68 PPO steps total** → val triggers at step 32,
  64, and the final step (≈ **3 val passes**).

Each val pass = ~580 val rows × `rollout.n=1` (val uses 1 rollout per
sample, no GRPO group) ≈ 580 rollouts; with `AGL_LOCAL_POOL_SIZE=64`
running concurrently this is roughly **1–2 min per validation pass**.
Set `trainer.val_before_train=True` (currently `False`) if you want a
baseline measurement before training starts. Raise `test_freq` to
reduce wall-clock spent on validation; set it to `0` to skip val
entirely (the last-step val is also skipped, see
[trainer.py](../../agl_lite/verl/trainer.py)).

To browse the full ScienceWorld task list, see
[ScienceWorld README — task table](https://github.com/allenai/ScienceWorld#tasks).

## Auth keys

Same rules as `async_calc_x`:

- `AGL_KEY` must be exported (or come from `.local/agl-lite.env`).
- `AGL_ADMIN_KEY` is auto-generated by `run.sh` if unset and **must
  differ** from `AGL_KEY` — agent subprocesses carry `AGL_KEY` and must
  not be able to reach the `/proxy/{pause,resume,state}` surface.

## What `run.sh` does, in order

1. Resolves repo root, creates `logs/<timestamp>/`, exports `AGL_LOG_DIR`.
2. Sources `.env.example`; validates `AGL_KEY`; generates `AGL_ADMIN_KEY` if missing.
3. Auto-detects NUMA layout (`numactl --hardware`); on multi-NUMA hosts
   prepares a `numactl --cpunodebind=2 --membind=2` prefix for the
   server (controller stays un-pinned).
4. `ray stop --force` to clean up prior runs.
5. Starts `agl-lite serve` (port `$AGL_HOST_PORT`, gateway-config +
   hooks loaded), with the NUMA prefix if applicable, as a background
   process; polls `/healthz` for 40 s.
6. Starts `agl-lite-controller runner_type=local
   --local-pool-size=$AGL_LOCAL_POOL_SIZE
   --local-agent-class=examples.science_world.agents.sw_agent:SWAgent`
   as a background process.
7. Runs the trainer in the foreground (`python … | tee training.log`),
   forwarding all positional args.
8. `trap cleanup EXIT` sends `SIGTERM` (then `SIGKILL` after 2 s) to the
   server and controller. In-flight rollout subprocesses are killed by
   the controller's `LocalReconciler._shutdown`.

## CI smoke test

```bash
examples/science_world/run.sh --ci-fast
```

`--ci-fast` sets `total_training_steps=1`, `train_batch_size=4`,
`ppo_mini_batch_size=4`, `rollout.n=2`, `async_train_batch_size=6`, and
`gpu_memory_utilization=0.6`. The full loop completes in a few minutes
once the model is cached.

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
pkill -TERM -f "local_reconciler"       # rollout subprocesses
pkill -TERM -f java                     # JVMs (careful — kills ALL java)
.venv/bin/ray stop --force
```

## Files

| File | Description |
|---|---|
| [agents/sw_agent.py](agents/sw_agent.py) | `SWAgent` — multi-turn LLM loop against `ScienceWorldEnv`. Loaded by the local reconciler worker entrypoint. |
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

### Rollouts Immediately FAILED

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
