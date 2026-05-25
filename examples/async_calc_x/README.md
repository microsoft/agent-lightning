# Calc-X Example — VERL Training on agl-lite (async-rollout)

Train a math-reasoning agent with VERL (PPO/GRPO) on agl-lite, using the
**async-rollout** path (`agentlightning.async_rollout.enabled=true`). The
agent uses AutoGen + an MCP calculator tool to solve math problems; the
agl-lite gateway transparently captures every LLM exchange for RL training.



## TL;DR — Launch in 4 steps

```bash
# 1. CUDA / VERL stack (one-time, ~50 min for flash-attn source build)
scripts/setup_verl.sh cu130          # match this to your driver: cu126/cu128/cu130/cpu
# Refresh Python deps once if your env predates local mode.
uv sync --extra verl

# 2. minikube — MUST be large enough; default 8 GiB OOMs the val pass
minikube start --memory=32768 --cpus=16 --driver=docker

# 3. Get the dataset
unzip examples/async_calc_x/data/calc-x-data.zip -d examples/async_calc_x/data/

# 4. Set keys & launch
export AGL_KEY=$(openssl rand -hex 32)
wandb login                           # optional but recommended
examples/async_calc_x/run.sh

# Local runner smoke test, no minikube or Docker agent image required
examples/async_calc_x/run.sh --local --ci-fast
```

## Architecture

```
run.sh ──► agl-lite deploy (agl-in-host) ──► train_calc_agent.py
            │                                       │
            ├── K8s:   controller                    ├── load Calc-X dataset
            └── Host:  agl-lite serve                ├── build VERL config
                  │                                  └── run_ppo() → VERL trainer
                  │                                            │
                  ├── enqueue rollouts ◄───────── AglLiteRolloutBridge (HTTP)
                  ├── controller creates K8s Jobs
                  │     └── agent pod: AutoGen + MCP calculator
                  │           └── LLM calls → gateway → vLLM (started by VERL, hybrid mode)
                  ├── gateway captures token IDs
                  └── triplets → padded tensors → PPO update
```

**Key topology facts** (catch these before debugging):

- **vLLM is started by VERL internally** (hybrid mode), not by `scripts/start_vllm.sh`.
  Don't run an external vLLM — VERL's `AglLiteAgentLoopManager` registers
  its own vLLM server with the gateway at runtime.
- **agl-lite serve runs on the host**, not in K8s (`AGL_MODE=agl-in-host`).
- **The controller runs in K8s as root** (deploy uses sudo) and creates
  agent Jobs in namespace `agl-async-calcx`.
- **Agent pods talk to the host gateway via `host.minikube.internal:8080`**.
- **Local mode** (`run.sh --local`) skips minikube/deploy/image build and
  runs `agl-lite controller --runner-type local`, spawning one Python
  subprocess per rollout on the host.

## Hardware

| Resource | Minimum | Tested |
|----------|---------|--------|
| GPU | 1 × 40 GB | 1 × A100 80 GB |
| Host RAM | 64 GB | 220 GB |
| Host CPUs | 16 | 24 |
| **minikube memory** | **32 GiB** | 32 GiB |
| **minikube CPUs** | **8** | 16 (host has 24) |
| Disk | 200 GB (model + datasets + minikube + flash-attn build) | — |

> **WHY minikube needs 32 GiB:** validation enqueues all 500 rows of
> `test.parquet` concurrently. With the default 8 GiB cap, kube-apiserver +
> etcd + scheduling churn for 500 pods drives the minikube container to
> 99% memory; the apiserver becomes unresponsive and the trainer hangs
> forever at `Completed 0/500 unfinished=500`. **Always start minikube
> with `--memory=32768`** for this example.

## Software prerequisites

- Python 3.12, `uv` installed
- Docker (user in `docker` group, no sudo)
- minikube ≥ 1.38 with the `docker` driver
- kubectl on PATH (installed by minikube)
- CUDA driver matching your `scripts/setup_verl.sh` flavor
  (cu126 / cu128 / cu130 / cpu)
- `wandb login` if you want online metrics

## One-time setup

```bash
# Conda env (or any Python 3.12 env)
conda create -n agl-lite python=3.12 -y
conda activate agl-lite
python -m pip install -U pip uv

# VERL + torch + vllm + flash-attn at pinned versions.
# Picks: cu126 / cu128 / cu130 / cpu. Match your driver.
scripts/setup_verl.sh cu130
```

`setup_verl.sh` installs torch 2.9.0, vLLM 0.12.0, verl 0.7.1,
flash-attn 2.8.3 (source build, A100 only via `FLASH_ATTN_CUDA_ARCHS=80`
needs ~50 min), xformers 0.0.33.post1, triton 3.5.0.

**Gotchas:**
- vLLM 0.12.0 cu130 wheel is **not on PyPI** — `setup_verl.sh` knows to
  pull it from the GitHub release URL.
- If flash-attn build fails partway, killing the install can leave a
  stub `flash_attn_2_cuda` that imports but is broken. Verify with
  `ldd .venv/lib/python3.12/site-packages/flash_attn_2_cuda*.so | grep cudart`
  — it must show `libcudart.so.13` for cu130.

## Dataset

Download `calc-x-data.zip` from
[Google Drive](https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view?usp=sharing)
into `examples/async_calc_x/data/`, then:

```bash
unzip examples/async_calc_x/data/calc-x-data.zip -d examples/async_calc_x/data/
```

Files produced:
- `train.parquet` — 8192 problems (training)
- `test.parquet` — 500 problems (full validation)
- `test_mini.parquet` — 20 problems (`--ci-fast`)
- `sample.jsonl` — 10 rows (checked into git, smoke test only)

## Configuration: `.env.example`

`.env.example` is sourced by `run.sh` and consumed by `agl-lite deploy`.
The variables that actually matter on first launch:

| Variable | Meaning | Default |
|----------|---------|---------|
| `AGL_NAMESPACE` | K8s namespace for controller + agent Jobs | `agl-async-calcx` |
| `AGL_MODE` | Deploy topology (do not change) | `agl-in-host` |
| `AGL_HOST_PORT` | Host port for `agl-lite serve` | `8080` |
| `AGL_LOCAL_POOL_SIZE` | Max concurrent local subprocess rollouts for `--local` | `8` |
| `AGL_MODEL_NAME` | HF model id served via gateway | `Qwen/Qwen2.5-1.5B-Instruct` |
| `AGL_TRAIN_FILE` / `AGL_VAL_FILE` | Parquet paths | `data/{train,test}.parquet` |
| `AGL_VLLM_*` | **Ignored in async path** — VERL manages vLLM | — |

> **Watch out:** `.local/agl-lite.env` is also sourced by `run.sh` after
> `.env.example` and overrides variables. If a previous run left a stale
> `AGL_NAMESPACE` there (e.g. `agl-calcx` from the sync example), it will
> silently take effect. Delete or update that file when switching examples.

## Auth keys

Two keys are required by the async path:

| Key | Used by | Source |
|-----|---------|--------|
| `AGL_KEY` | Agent pods, gateway clients | You export it, or `.local/agl-lite.env` from a previous `agl-lite deploy` |
| `AGL_ADMIN_KEY` | Trainer-only, gates `/admin/gateway/{pause,resume,state}` | `run.sh` auto-generates one if unset |

Rules `run.sh` enforces:
- `AGL_KEY` must be set (it errors out otherwise).
- `AGL_ADMIN_KEY` must **differ** from `AGL_KEY` (agent pods carry
  `AGL_KEY` and must not be able to reach the admin surface).

## Launching

### Full training

```bash
# Foreground
export AGL_KEY=$(openssl rand -hex 32)
examples/async_calc_x/run.sh

# Background (recommended for full runs — they take hours)
export AGL_KEY=$(openssl rand -hex 32)
LOG=.local/setup-logs/async_calc_x_run_$(date +%Y%m%d-%H%M%S).log
nohup env PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" \
    bash examples/async_calc_x/run.sh > "$LOG" 2>&1 &
echo $! > .local/setup-logs/last_train.pid
echo $LOG > .local/setup-logs/last_train.logpath
disown
tail -f "$LOG"
```

### CI smoke test (single PPO step, ~5 min after warm-up)

```bash
examples/async_calc_x/run.sh --ci-fast
```

`--ci-fast` uses `test_mini.parquet`, sets `total_training_steps=1`, and
disables checkpoint saving. Use it to validate the whole stack quickly
before committing to a multi-hour run.

### Local runner mode

```bash
export AGL_KEY=$(openssl rand -hex 32)
export WANDB_MODE=disabled            # optional, useful for smoke tests
examples/async_calc_x/run.sh --local --ci-fast
```

Local mode starts `agl-lite serve` and a local controller in the background,
then runs the same async VERL training script in the foreground. It uses
`examples.async_calc_x.agents.calc_agent:CalcXAgent` through the local worker,
so AutoGen/MCP behavior is the same as the K8s agent script path, but rollout
processes run directly on the host. `AGL_LOCAL_POOL_SIZE` controls local
rollout concurrency. The local Python environment must include the agent
runtime packages from the `verl` extra, including AutoGen and
`mcp-server-calculator`; `run.sh --local` checks this before launching the
controller.

### Standalone training (infra already up)

```bash
python examples/async_calc_x/train_calc_agent.py \
    --train-file examples/async_calc_x/data/train.parquet \
    --val-file examples/async_calc_x/data/test.parquet
```

## What `run.sh` does, in order

1. Creates `logs/<timestamp>/`. In K8s mode, bind-mounts `agents/` into
   minikube via `minikube mount` and kills any stale mount from a previous run.
2. Sources `.env.example`, then `.local/agl-lite.env` (state file from
   previous `agl-lite deploy`, if present).
3. Validates `AGL_KEY`; generates `AGL_ADMIN_KEY` if unset.
4. K8s mode builds Docker images via `scripts/build_images.sh --include-example async_calc_x`
   (loads them straight into minikube's docker, no registry).
5. K8s mode cleans any previous namespace, then runs
   `agl-lite deploy --env-file .env.example` to create the K8s controller and
   start `agl-lite serve` on the host as a systemd-style child.
6. Local mode starts `agl-lite serve` and `agl-lite controller --runner-type local`
   as child processes instead.
7. Polls `GET /healthz` up to 40 s for readiness.
8. Cleans leftover Ray (`ray stop --force`).
9. `exec` into `train_calc_agent.py`. VERL boots Ray, loads the model,
   then starts its own internal vLLM, registers it with the gateway,
   and begins the validation pass before training.

## Monitoring a live run

```bash
# Trainer progress (rollouts completed)
LOG=$(cat .local/setup-logs/last_train.logpath)
grep "Completed " "$LOG" | tail -5

# Agent pods (should cycle Running → Completed)
kubectl get pods -n agl-async-calcx

# Controller logs
kubectl logs -n agl-async-calcx deploy/agl-controller --tail=50

# Gateway / server log (JSON, structlog)
tail -f examples/async_calc_x/logs/<timestamp>/server.log

# wandb URL (printed near top of training.log)
grep "View run at" "$LOG"

# Health snapshot
docker stats minikube --no-stream    # memory % should stay well under 80
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
```

Expected steady-state: model takes ~5 min to load, val pass (500 rollouts)
takes ~10–15 min on A100, then PPO step #1 starts. Total wall clock for
a full run with the default config is several hours.

## Stopping cleanly

```bash
PID=$(cat .local/setup-logs/last_train.pid)
pkill -TERM -P "$PID"; kill -TERM "$PID"
sleep 3
pkill -9 -P "$PID"; kill -9 "$PID"
.venv/bin/ray stop --force
pkill -9 -f "train_calc_agent|agl-lite serve|ray::|VLLM::|wandb-xpu"
sudo pkill -9 -f "agl-lite controller"     # controller runs as root
```

Verify GPU is freed:

```bash
nvidia-smi --query-gpu=memory.used --format=csv
# expect: 0 MiB (or only your shell's overhead)
```

## wandb metrics

`trainer.logger=["console","wandb"]` is set in
[train_calc_agent.py](train_calc_agent.py) at line 99. The wandb run
appears under the `agl-lite` project; the run name comes from
`trainer.experiment_name` (default `async_calc_x_v1`, or
`async_calc_x_<timestamp>_<rand>` in CI mode).

VERL emits **two `logger.log()` calls per training step**, plus one at
step 0 for the initial validation pass:

| When | What gets logged | Source |
|------|------------------|--------|
| `step=0` (before training) | Initial val metrics only | [`trainer.py:294-297`](../../agl_lite/verl/trainer.py#L294-L297) |
| End of every step | Train + (optionally) val + timing + perf | [`trainer.py:325-345`](../../agl_lite/verl/trainer.py#L325-L345), VERL `ray_trainer.py:1603` |
| End of run | Final val metrics (console only) | [`trainer.py:345`](../../agl_lite/verl/trainer.py#L345) |

### Metric families

All scalar; grouped by prefix in the wandb UI:

| Prefix | What it captures | Example keys |
|--------|------------------|--------------|
| `critic/score/*` | Raw reward from hooks before any KL/penalty | `critic/score/mean`, `.../max`, `.../min` |
| `critic/rewards/*` | Score after KL-in-reward (if enabled) | `critic/rewards/mean`, `.../max`, `.../min` |
| `critic/advantages/*` | GRPO/GAE advantages over response tokens | `critic/advantages/{mean,max,min}` |
| `critic/returns/*` | Returns (advantage + value baseline) | `critic/returns/{mean,max,min}` |
| `critic/values/*` | Critic values — **only when `adv_estimator=gae`**. GRPO disables critic, so these are absent in this example. | `critic/values/{mean,max,min}`, `critic/vf_explained_var` |
| `actor/*` | Actor loss, KL, entropy, clip stats | `actor/pg_loss`, `actor/entropy`, `actor/kl_loss`, `actor/clip_ratio` |
| `response_length/*` | All samples including aborted | `response_length/{mean,max,min,clip_ratio}` |
| `response_length_non_aborted/*` | Excludes aborted samples (resp_len=0) | same suffixes |
| `response/aborted_ratio` | Fraction with `response_length==0` | scalar |
| `prompt_length/*` | Prompt token counts | `prompt_length/{mean,max,min,clip_ratio}` |
| `num_turns/*` | Multi-turn conversation depth | `num_turns/{mean,max,min}` |
| `tool_call_counts/*` | Calculator MCP tool invocations per sample | `tool_call_counts/{mean,max,min}` |
| `val-core/<data_source>/<var>/<metric_name>` | Best-of-N / pass@N val metrics (e.g. `mean@4`, `best@4`) | `val-core/calc_x/score/mean@4` |
| `val-aux/<data_source>/<var>/<metric_name>` | Per-sample / std val metrics | `val-aux/calc_x/score/std@4`, `val-aux/num_turns/mean` |
| `timing_s/*` | Wall-clock per stage (seconds) | `timing_s/gen`, `.../ref`, `.../adv`, `.../update_actor`, `.../step`, `.../validate` |
| `timing_per_token_ms/*` | Normalized timing | `timing_per_token_ms/gen`, etc. |
| `perf/*` | Throughput | `perf/total_num_tokens`, `perf/time_per_step`, `perf/throughput` |
| `training/*` | KL coefficient, global step, epoch | `training/global_step`, `training/epoch`, `training/kl_coef` |
| `training/async/*` | **Async-rollout bookkeeping** — see table below | `training/async/n_new_data_ids`, `.../n_carry_over_data_ids_in`, `.../sample_iterator_epoch`, `.../sample_iterator_consumed`, `.../cross_epoch_boundary`, `.../n_carry_over_in`, `.../n_carry_over_out`, `.../n_carry_over_resumed`, `.../carry_over_age_max_steps`, `.../groups_finished_reached`, `.../n_active_data_ids`, `.../n_active_rollouts`, `.../n_selected_groups`, `.../drain_wait_seconds`, `.../drain_timeout`, `.../inflight_at_pause`, `.../group_finish_skew_s` |

The most useful single chart for "is RL working?" is
`val-core/calc_x/score/mean@4` (validation pass-rate of the agent on the
500-row test set) over `training/global_step`.

### `training/async/*` — what each one tells you

These are the async-rollout-specific metrics — they only appear when
`agentlightning.async_rollout.enabled=true` (which this example sets).
Logged every step from [`trainer.py:732-737`](../../agl_lite/verl/trainer.py#L732-L737)
and [`rollout_bridge.py:1655-1660`](../../agl_lite/verl/rollout_bridge.py#L1655-L1660).

| Metric | Meaning | What's healthy |
|--------|---------|----------------|
| `training/async/n_new_data_ids` | Fresh samples this step pulled from the dataloader | Tracks toward `async_train_batch_size - n_carry_over` |
| `training/async/n_carry_over_data_ids_in` | Rollouts still in-flight when this step started (carried from previous step) | Steady-state value, bounded by `async_train_batch_size`; 0 on step 1 |
| `training/async/n_carry_over_in` | Same count, recorded by the bridge before this step's enqueue | Should equal `n_carry_over_data_ids_in` |
| `training/async/n_carry_over_out` | Rollouts still in-flight when this step **ended** (will be carry-over for next step) | Tail length — non-zero is normal in async mode; a steadily climbing value means rollouts are accumulating faster than the GPU drains them |
| `training/async/n_carry_over_resumed` | Carry-over rollouts that completed during this step and got merged into the training batch | The "productive" share of the carry-over pool |
| `training/async/carry_over_age_max_steps` | Max age (in training steps) of any still-pending carry-over rid | Should stay small (≤ a few). High values flag stuck rollouts; trigger a warning if `> max_carry_over_age_steps` |
| `training/async/sample_iterator_epoch` | How many times the train dataloader has fully wrapped around | Bumps once per ~`len(train)/async_train_batch_size` steps |
| `training/async/sample_iterator_consumed` | Total samples pulled from the dataloader so far this run | Monotonically increasing |
| `training/async/cross_epoch_boundary` | 1 if this step's `take()` crossed an epoch boundary, else 0 | Spikes briefly at epoch wraps |
| `training/async/group_finish_skew_s` | Median wall-clock spread (max − min, seconds) of rollout end-times **within each selected group** | Small (a few seconds typical for Calc-X). Large values mean long-tail stragglers inside groups — see subsection below |

#### `group_finish_skew_s` — what the code actually does

Each prompt (a `data_id`) is rolled out `rollout_n` times to form a GRPO
group (this example uses `n=4`). With async rollout, those 4 rollouts
fire concurrently through the gateway but generally **don't finish at
the same instant** — one may stream tool calls and tokens faster than
its siblings.

`group_finish_skew_s` quantifies that intra-group spread. It is
computed by
[`_compute_group_finish_skew`](../../agl_lite/verl/rollout_bridge.py#L1564-L1578)
at the moment the bridge has just picked the `target_groups` finished
groups for this training step
([`rollout_bridge.py:1512-1514`](../../agl_lite/verl/rollout_bridge.py#L1512-L1514)
on the happy path,
[`rollout_bridge.py:1556-1558`](../../agl_lite/verl/rollout_bridge.py#L1556-L1558)
on the timeout-placeholder path):

```python
def _compute_group_finish_skew(self, selected_dids: list[str]) -> float:
    per_group_spread: list[float] = []
    for did in selected_dids:
        rids = self._data_id_to_rids.get(did, set())                   # all rollout ids belonging to this prompt
        ends = [self._rollout_end_time[r] for r in rids if r in self._rollout_end_time]
        if len(ends) >= 2:
            per_group_spread.append(max(ends) - min(ends))             # spread for this one group
    if not per_group_spread:
        return 0.0
    return float(np.median(per_group_spread))                          # median across groups
```

Concretely:

1. `selected_dids` is the list of `data_id`s the bridge just chose to
   ship into the PPO step (length = `target_groups`, i.e. one entry per
   group that finished first).
2. For each `did`, `_data_id_to_rids[did]` is the set of `rollout_n`
   rids that belong to that prompt (populated when each rid is
   enqueued, [`rollout_bridge.py:1334`](../../agl_lite/verl/rollout_bridge.py#L1334)).
3. `_rollout_end_time[r]` is the `time.time()` stamped onto each rid
   the moment it transitions to a terminal state (Succeeded / Failed /
   Timeout — set at
   [`rollout_bridge.py:608`](../../agl_lite/verl/rollout_bridge.py#L608),
   [`:619`](../../agl_lite/verl/rollout_bridge.py#L619),
   [`:1245`](../../agl_lite/verl/rollout_bridge.py#L1245), and
   [`:1385`](../../agl_lite/verl/rollout_bridge.py#L1385)).
4. For each group with ≥ 2 finished rollouts, the spread is
   `max(ends) - min(ends)` — i.e. how long the **last** rollout in the
   group took to finish after the **first** one did.
5. The metric value is the **median** of those per-group spreads.
   Median (not mean) so a single pathological group doesn't dominate
   the signal.

**How to read it:**

- **Small (`< 5s` for Calc-X)** — Healthy. Rollouts in the same group
  are roughly the same length and finish close together.
- **Large and stable (tens of seconds)** — Groups have intrinsic
  long-tail behavior: one rollout in the group is consistently doing
  many more tool calls / longer reasoning than its siblings. This is
  the *normal* failure mode for "the agent occasionally goes down a
  much deeper search tree."
- **Large and growing** — Worse: vLLM is saturated and the tail
  rollouts are sitting in the request queue. Cross-reference with
  `n_carry_over_out` climbing and `inflight_at_pause` non-zero at the
  end of the step. Likely fix: lower `async_train_batch_size` or
  `rollout.n`, or give vLLM more GPU.

**Why it matters operationally:** the bridge pauses the gateway and
calls `wait_until_inflight_drained` once `target_groups` finished
groups are picked. Tail rollouts from those selected groups are *also*
in `inflight_at_pause` and must drain inside `drain_timeout` (default
30s, configured via `gateway_drain_timeout_seconds`). If
`group_finish_skew_s` ever approaches `drain_timeout`, `drain_timeout`
will start flipping to `1` — bump the drain timeout or shrink the
group.

**Common diagnostic patterns:**

- `n_carry_over_out` climbing every step with `n_carry_over_resumed` low
  → vLLM throughput is the bottleneck; the gateway pause-window isn't
  large enough to drain. Increase `gateway_drain_timeout_seconds` or
  lower `async_train_batch_size`.
- `carry_over_age_max_steps` growing without bound → a specific rollout
  is wedged (agent pod stuck, hooks crashing). Check
  `kubectl get pods -n agl-async-calcx` for that rid.
- `cross_epoch_boundary=1` showing up before `sample_iterator_epoch`
  bumps → expected for the step that straddles the wrap.

**Yes, these all save to wandb.** They are written into the same
`metrics` dict that gets pushed by
[`logger.log(data=metrics, step=...)`](../../agl_lite/verl/trainer.py#L773)
at the end of every async step — exactly the same path as
`critic/rewards/*`, `actor/*`, etc.

### "Why is wandb only showing System metrics?"

**The first `logger.log()` only fires *after* the initial validation
pass completes.** For this example that's the full 500 rows of
`test.parquet`, which takes ~10–15 min on one A100. Until then, the only
data wandb has is what its client autocaptures — CPU / GPU / RAM /
network — shown under **System** in the UI.

**Symptoms of being in this window:**

```bash
grep -E "Completed [0-9]+/500" training.log | tail -1
# Completed 311/500 (unfinished=189)   ← still validating

grep -E "Initial validation metrics" training.log
# (nothing — val hasn't returned yet)
```

If you stop the run during this window (Ctrl-C, OOM, anything), wandb
keeps the run with system metrics only and no scalars.

**To get scalars sooner**:

1. **Run `--ci-fast`** — uses `test_mini.parquet` (20 rows), val pass
   finishes in ~30 s and you see metrics immediately.

2. **Skip the initial val pass** — edit
   [train_calc_agent.py:97](train_calc_agent.py#L97) and set
   `"val_before_train": False`. The first metric push then happens at
   the end of training step 1.

3. **Just wait** for the val pass to finish. The first scalars appear
   right after you see this in `training.log`:

   ```text
   Initial validation metrics: {'val-core/calc_x/score/mean@4': ...}
   ```

### Sanity-check what wandb received

```bash
# Locally-cached run state (before sync to cloud)
ls wandb/run-*/files/wandb-summary.json
cat wandb/latest-run/files/wandb-summary.json | python -m json.tool | head -40

# Or query the run via wandb CLI
wandb run <run-id>
```

## Troubleshooting

### Stuck at `Completed 0/500 unfinished=500` indefinitely

**Cause:** minikube apiserver dead, almost always from OOM.

```bash
docker stats minikube --no-stream    # check mem %
kubectl get ns                       # likely hangs
```

Fix: `minikube delete --purge && minikube start --memory=32768 --cpus=16`,
then relaunch.

### `kubectl` hangs / timeout

Apiserver crashed. Same fix as above.

### Agent pods stuck `ErrImageNeverPull`

The image name in [job-template.yaml](job-template.yaml) does not match the
tag produced by `scripts/build_images.sh`. Both must be `async-calc-x-agent:dev`.
You can also retag in minikube: `eval $(minikube docker-env) && docker tag <old> async-calc-x-agent:dev`.

### `vllm._C` ImportError: `libcudart.so.12`

`setup_verl.sh` installed the cu12 vLLM wheel from PyPI instead of the
cu130 wheel from GitHub releases. Reinstall:

```bash
uv pip install --python .venv/bin/python \
    https://github.com/vllm-project/vllm/releases/download/v0.12.0/vllm-0.12.0+cu130-cp38-abi3-manylinux_2_31_x86_64.whl \
    --force-reinstall --no-deps
```

### flash-attn import error after install completes "successfully"

A killed mid-build can leave a stub. Rebuild from source with the right
arch:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 \
  uv pip install --python .venv/bin/python flash-attn==2.8.3 \
      --no-build-isolation --force-reinstall
# Takes ~50 min on A100 host. Verify:
ldd .venv/lib/python3.12/site-packages/flash_attn_2_cuda*.so | grep cudart
```

### vLLM `hermes_tool_parser` JSONDecodeError spam

These are **non-fatal**. vLLM 0.12.0's Hermes parser occasionally fails to
split concurrent `<tool_call>` blocks. The parser falls back to plain text
(`tools_called=False`), the rollout still finishes, and the trainer counts
it as a normal Completed sample. Look for `failed=0 cancelled=0 timeout=0`
in the progress line — if those are all zero, the spam is cosmetic.

### Trainer prints config then exits without launching Ray

Almost always a CUDA / torch / flash-attn version mismatch. Verify in
order:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -c "import flash_attn_2_cuda; print('flash-attn ok')"
.venv/bin/python -c "import vllm; print(vllm.__version__)"
```

## Files

| File | Description |
|------|-------------|
| `agents/calc_agent.py` | Standalone agent container — AutoGen + MCP calculator, no agl-lite imports |
| `eval_utils.py` | Sympy-based numeric answer comparison |
| `train_calc_agent.py` | Training entrypoint — dataset load, VERL config build, `run_ppo()` |
| `run.sh` | E2E launcher — mount, deploy, wait, train |
| `Dockerfile.agent` | Agent container image (Python 3.12-slim + autogen + mcp) |
| `job-template.yaml` | K8s pod spec for agent Jobs |
| `hooks.py` | `CalcXHooks` — `on_enqueue` injects task; `on_succeeded` computes reward |
| `gateway-config.yaml` | Injects `return_token_ids` for RL token tracking |
| `.env.example` | Deploy + experiment config (single file) |
| `data/` | Dataset directory (gitignored except `sample.jsonl`) |
| `logs/<timestamp>/` | Per-run logs: `server.log`, `training.log`, `agents/<id>/agent.log` |
