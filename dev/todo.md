# agl-lite Implementation Plan

> Aligned with the final architecture in `docs/design/0_architecture.md`
> and reviewed architecture decisions.

## Guiding Principles

1. **One repo, one package, two entrypoints**: `agl-lite serve` and `agl-lite controller`
2. **Controller talks to service only over HTTP** — no shared in-memory state
3. **Shared code limited to types/schemas** — both entrypoints import the same models
4. **Test each layer before building the next** — schemas → store → API → gateway → controller
5. **Freeze normative contracts early** — schemas, state transitions, auth matrix, event ordering

---

## Completed

- [x] **Phase 0**: Schemas, state transitions, project skeleton (Pydantic models, dev tooling)
- [x] **Phase 1**: In-memory store — rollouts (batch), events, resources, models, archive. 156 tests.
- [x] **Phase 2**: HTTP API — FastAPI app, auth, all store routes, gateway (config/router/proxy with wildcard + list-based routes), CLI, streaming SSE.
- [x] **Phase 3**: K8s controller — Python client, job builder, reconciler (create/watch/cancel/crash-recovery), CLI. 254 tests total.
- [x] **Phase 4a**: E2E with mock model server (CPU-only minikube). 271 unit tests + 9 e2e/cpu tests.
  - Kr8s adapter, `agl-client` CLI, deploy structure, example agents
  - `job_template` refactor (raw K8s pod spec, name-matched container overrides)
  - Math PoC mock mode: 2-iteration RL loop with weight update (v1→v2), streaming, deterministic rewards via `\boxed{}` embedding, model_request event validation, reference log
  - Scripts: `build_images.sh`, `deploy.sh`, `run.sh` (one-command E2E with log capture)
- [x] **Phase 4b**: E2E with real vLLM (GPU).
  - vLLM via Docker (`vllm/vllm-openai:latest`) on host GPU; `scripts/start_vllm.sh`
  - Colocated topology: agl-lite + vLLM + algorithm on host, only controller + agents in minikube
  - Gateway param injection: `gateway-config.yaml` adds `return_token_ids: true` to all requests; prompt + response token IDs captured in `model_request` events
  - `rl_loop.py` — real algorithm: plain questions, numeric reward, token_id verification
  - `deploy.sh --no-serve` for controller-only K8s deployment
  - Two `.env` examples (`.env.mockai.example`, `.env.vllm.example`), mode-aware `run.sh`
  - Reference logs for both modes; event captures prepared body (with injected params)
  - 271 unit tests passing, both modes E2E verified

---

## Phase 4a.7: Additional E2E scenarios [backlog]

Not on the critical path — the happy-path lifecycle is fully validated. Add when needed.

- [ ] **Cancel test**: enqueue → cancel mid-run → verify cancelled status
- [ ] **Retry test**: agent with `CRASH_ON_FIRST=1` → K8s Job retries → succeeds on second attempt
- [ ] **503 test**: agents hitting gateway during model deregistration window

---

## Phase 4b.3: Performance baseline [backlog]

- [ ] Measure: rollout throughput, gateway proxy latency overhead, event capture overhead
- [ ] Compare direct vLLM vs gateway-proxied vLLM

---

## SWE-bench Example (`examples/swe_bench`) [discuss]

**Goal**: Add an end-to-end SWE-bench example that uses agl-lite to orchestrate coding
agents (Claude Code, mini-swe-agent, etc.) solving SWE-bench tasks inside official SWE-bench
Docker containers, with a reward function that evaluates patches by running golden tests.

### Design Decisions (Resolved)

#### (A) Container image — per-rollout via `RolloutConfig.image`

`RolloutConfig` already has a first-class `image` field (no need for `overrides`):
```python
EnqueueRolloutRequest(
    resources_id=resources_id,
    input=json.dumps({"instance_id": ..., "problem_statement": ..., ...}),
    config={
        "image": f"swebench/sweb.eval.x86_64.{safe_id}:latest",
        "environment_variables": {"AGL_CODING_AGENT": "claude_code"},
    }
)
```
The `job_template` defines the generic pod spec (entrypoint command, resource limits,
volume mounts); `config.image` overrides only the agent container image per-rollout.

Image pull policy: use `imagePullPolicy: IfNotPresent` (naive flow). If image pull
becomes a bottleneck later, options include:
- Pre-pulling with a DaemonSet or init job
- Using smaller SWE-bench images (e.g., Epoch AI's trimmed images)

#### (B) Coding agent scripts (pluggable)

Each coding agent gets `install.sh` + `run.sh` under `agents/<name>/`:
- `agents/claude_code/install.sh` — install claude CLI via `curl`
- `agents/claude_code/run.sh` — launch claude code with problem statement + CLAUDE.md
- `agents/mini_swe_agent/install.sh` — e.g., `pip install ...` or inline Python
- `agents/mini_swe_agent/run.sh` — launch the agent

Shared `agents/entrypoint.sh`:
1. Reads `AGL_CODING_AGENT` env var → dispatches to `agents/<name>/install.sh` then `run.sh`
2. After agent finishes → `git diff HEAD` → captures patch
3. Posts `agent_output` event with `data.patch` to `AGL_EVENT_URL`

#### (C) Mountable files — volume mount via ConfigMap

Files like `CLAUDE.md`, `entrypoint.sh`, and per-agent scripts live in a ConfigMap.
The existing `Mount` schema + job_builder already support ConfigMap mounts:
```python
# Mount spec: source is a ConfigMap name (not starting with "/" or "pvc:")
config={
    "mount": [{"name": "agent-scripts", "mount_path": "/agl/agents", "source": "swe-agent-scripts"}],
}
```
The entrypoint copies relevant files: `cp /agl/agents/claude_code/CLAUDE.md /testbed/`.

ConfigMap is created by `deploy.sh` or `run.sh` from the `agents/` directory:
```bash
kubectl create configmap swe-agent-scripts --from-file=agents/ -n $NS
```
> **Open question for later**: for large files or binary assets, a PVC or init container
> downloading from a URL may be better. ConfigMap is fine for scripts + markdown.

#### (D) Reward function — single-container agent + evaluation

Everything runs in **one container per rollout**. No second rollout needed.

**Why this works** (traced from `swebench.harness`):

1. `make_test_spec(instance)` → `TestSpec` with `eval_script` (~2KB bash script)
2. `eval_script` only touches **test files** (from `test_patch`), not source files:
   - `git checkout {base_commit} {test_files}` — reset only files in `test_patch`
   - `git apply test_patch` — apply golden tests
   - run test command (e.g., `pytest`)
   - `git checkout {base_commit} {test_files}` — revert test files
3. The agent's source code modifications are **untouched** by `eval_script`
4. `get_eval_report()` just parses test output text — no Docker SDK, pure string matching
5. `eval_script` is self-contained (conda env, repo, deps all pre-installed in SWE-bench image)
6. `make_test_spec()` only needs the dataset instance — not the agent's output.
   So `eval_script` can be **pre-generated** before the agent runs.

**Single-container flow** — the entrypoint does everything:

```bash
# entrypoint.sh (runs inside SWE-bench Docker image at /testbed)
# Phase 1: Agent
parse AGL_CODING_AGENT → install agent → run agent (modifies /testbed source)
git diff HEAD → capture patch → post agent_output event

# Phase 2: Evaluate (same container, agent's changes still in working tree)
run eval_script (pre-generated, passed via env var AGL_EVAL_SCRIPT)
  → resets test files → applies golden tests → runs pytest → reverts test files
capture test output → parse for PASS/FAIL → compute reward → post reward event
```

**Data flow**:
```
Algorithm (rl_loop.py)                     K8s container
──────────────────────                     ──────────────
1. make_test_spec(instance) → eval_script  (pre-generate, ~2KB)
2. enqueue rollout with:              →    entrypoint.sh:
     image: swebench/<instance>              1. install + run coding agent
     env: AGL_EVAL_SCRIPT=<script>           2. git diff → post agent_output(patch)
          AGL_EVAL_META=<FAIL_TO_PASS,...>    3. run eval_script → capture test log
     input: problem_statement                4. parse log → post reward event
3. poll until succeeded               ←
4. get_events(reward)                 ←    reward=1.0 or 0.0
```

**Grading inside the container**: The entrypoint includes a small inline grading
function (or a Python script mounted via ConfigMap). The grading logic from
`swebench.harness.grading` is just regex-based log parsing (~30 lines):
- Extract text between `>>>>> Start Test Output` and `>>>>> End Test Output`
- Parse pytest output lines for PASSED/FAILED status
- Check FAIL_TO_PASS tests all pass, PASS_TO_PASS tests still pass → resolved
- Post reward event (1.0 or 0.0) to `AGL_EVENT_URL`

The `FAIL_TO_PASS` and `PASS_TO_PASS` test lists are passed via env var (`AGL_EVAL_META`)
so the grading script knows which tests to check.

**Advantages over two-phase approach**:
- No second rollout — simpler algorithm, fewer K8s jobs, no patch-passing problem
- No need for Docker on algorithm host
- Evaluation runs in the exact environment where the agent made changes
- One timeout covers agent + evaluation (can still be generous, e.g., 60 min)

#### (E) Algorithm script structure

Similar to math-poc `rl_loop.py`, but simpler than two-phase since evaluation
happens inside the agent container:
```
rl_loop.py:
  1. register resources (job_template, ConfigMap setup)
  2. load dataset (swebench_samples.jsonl or full SWE-bench)
  3. register model server (vLLM endpoint)
  4. for each batch:
     a. for each instance: make_test_spec → eval_script + FAIL_TO_PASS/PASS_TO_PASS
     b. enqueue rollouts (per-instance image, eval_script in env, coding agent config)
     c. poll until all done
     d. get_events → collect reward events (posted by container) + agent_output (patches)
     e. aggregate results (resolved rate, patches for training data)
```

### Files to Create

```
examples/swe_bench/
├── README.md                          # setup + usage docs
├── rl_loop.py                         # algorithm script (task-agnostic, enqueue + poll + tensors)
├── hooks.py                           # SWE-bench store hooks (on_enqueue + on_succeeded)
├── Dockerfile.server                  # agl-lite + swebench + hooks.py
├── gateway-config.yaml                # same as math-poc (inject return_token_ids)
├── job-template.yaml                  # K8s pod spec — generic, image overridden by hook
├── swebench_samples.jsonl             # small dataset for smoke testing
├── agents/
│   ├── entrypoint.sh                  # shared entrypoint: agent → eval → write to volume
│   ├── claude_code/
│   │   ├── install.sh                 # install claude CLI
│   │   ├── run.sh                     # run claude code on the problem
│   │   └── CLAUDE.md                  # system instructions for claude code
│   └── mini_swe_agent/
│       ├── install.sh                 # install mini-swe-agent
│       └── run.sh                     # run mini-swe-agent on the problem
├── run.sh                             # one-command E2E runner
└── .env.vllm.example                  # env config for vLLM mode
```

Also needed in `agl_lite/` core:
```
agl_lite/
├── hooks.py                           # RolloutHooks base class (ABC)
├── store/memory.py                    # updated: hook integration points
└── server/app.py                      # updated: --hooks CLI flag, load module
```

### Remaining Open Questions

1. **Timeout budget**: Agent + evaluation share one `activeDeadlineSeconds`. Agent may take
   30-60 min, evaluation 5-30 min. Default to 90 min? Configurable via env var.
2. **Dataset format**: Use full SWE-bench JSONL format (same fields as original example)
   so `make_test_spec` works directly. The `input` field sent to agent is a subset
   (instance_id + problem_statement); eval_script and test metadata passed via env vars.
3. **eval_script size in env var**: Typical eval_script is ~2KB. K8s env var limit is 1MB.
   Safe for all SWE-bench instances. If any edge case exceeds, fall back to ConfigMap.
4. **ConfigMap creation for agent scripts**: `run.sh` creates the ConfigMap from `agents/`
   directory. Need to handle updates (delete + recreate) and namespace.
5. **Grading without swebench package**: The container doesn't have `swebench` installed.
   `grade.py` re-implements the minimal log parsing (~30 lines of regex). The algorithm
   passes `FAIL_TO_PASS` and `PASS_TO_PASS` test lists via `AGL_EVAL_META` env var.

**Update (hooks-in-server)**: Instead of grading inside the container or on a separate
algorithm host, task-specific logic runs as **store hooks** inside the agl-lite server.
The container writes test output + patch to a shared volume. The hook reads them and
grades using official swebench tools. See the Store Hooks architecture below.

---

## Store Hooks — task-specific logic in the server [completed]

**Goal**: Make the algorithm/trainer layer task-agnostic by moving task-specific logic
(dataset parsing, rollout configuration, reward computation) into **hooks** that run
inside the agl-lite server process. Users customize behavior by writing a Python module
and patching it into the agl-lite Docker image.

### Problem: current daemon mixes concerns

The current `AglLiteDaemon` (and original `AgentModeDaemon`) mixes:

| Concern | Task-agnostic? | Current location |
|---------|---------------|-----------------|
| Register model servers | ✅ | `_async_set_up` |
| Parse dataset rows → rollout input | ❌ task-specific | `_async_set_up` (inlines data directly) |
| Build `EnqueueRolloutRequest` (image, env, command) | ❌ task-specific | `_async_set_up` (generic config) |
| Poll rollouts until done | ✅ | `_async_run_until_finished` |
| Fetch events + build triplets | ✅ | `_async_validate_data` |
| Compute reward from events | ❌ task-specific | Currently outside daemon (agent posts reward) |
| Triplets → padded tensors → DataProto | ✅ | `get_train_data_batch` |
| Metrics computation | ✅ | `get_test_metrics` |

### Solution: hooks as store pre-processors

Task-specific logic runs as **synchronous hooks inside the store**, called at two
lifecycle points. Since the store is single-threaded (all methods are sync `def`,
called from `async def` route handlers on one event loop), hooks execute atomically —
no reader can see intermediate state while a hook is running.

#### Hook interface

```python
class RolloutHooks(ABC):
    """Task-specific hooks. Loaded by agl-lite server at startup."""

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        """Pre-processor: transform rollout request before it enters the store.

        Called for each request in enqueue_rollouts(), BEFORE the rollout is persisted.
        If this raises, the rollout is never created and the API returns an error.

        Use cases:
        - Map instance_id → Docker image tag
        - Generate eval_script from dataset
        - Inject task-specific env vars
        """
        return request  # default: passthrough

    def on_succeeded(self, rollout: Rollout, events: dict, store: InMemoryStore) -> None:
        """Post-transition hook: called when rollout transitions to SUCCEEDED.

        Runs synchronously inside update_rollout(), after the transition is committed.
        Since the store is single-threaded, no reader can interleave — the transition
        and this hook are atomic from any external observer's perspective.

        Use cases:
        - Read test output from volume → grade with official tools → post reward event
        - Parse agent output → compute numeric reward → post reward event
        """
        pass  # default: no-op

    def on_failed(self, rollout: Rollout, store: InMemoryStore) -> None:
        """Post-transition hook: called when rollout transitions to TERMINAL_FAILED."""
        pass  # default: no-op
```

#### Store integration

```python
# store/memory.py
class InMemoryStore:
    def __init__(self, hooks: RolloutHooks | None = None):
        self._hooks = hooks
        ...

    def enqueue_rollouts(self, requests):
        results = []
        for req in requests:
            # Pre-processor: transform request before persist
            if self._hooks:
                req = self._hooks.on_enqueue(req)

            rollout = Rollout(rollout_id=..., input=req.input, config=req.config, ...)
            self._rollouts[rollout_id] = rollout
            results.append(rollout)
        return results

    def update_rollout(self, rollout_id, req):
        rollout = self.get_rollout(rollout_id)
        old_status = rollout.status
        # ... validate transition, apply update ...
        self._rollouts[rollout_id] = updated

        # Post-transition hooks — still inside the sync method,
        # no reader can interleave (single-threaded event loop)
        if self._hooks and updated.status != old_status:
            events = self._events.get(rollout_id, {})
            if updated.status == RolloutStatus.SUCCEEDED:
                self._hooks.on_succeeded(updated, events, self)
                # hook may call self.add_event(..., "reward", ...)
                # reward is in the store before this method returns
            elif updated.status == RolloutStatus.TERMINAL_FAILED:
                self._hooks.on_failed(updated, self)

        return updated
```

#### Why atomicity is free

The store is single-threaded by design (see `docs/dev_guidelines.md § Concurrency Model`):

- Store methods are plain `def` (synchronous) — no `await`, no yield points
- Route handlers are `async def` on one event loop — only one can execute store
  code at a time
- Hooks run inside store methods — same synchronous block

So when `on_succeeded` fires and posts a reward event:
1. No other request can read from the store during this time
2. When the method returns, the rollout is SUCCEEDED **and** the reward event exists
3. The daemon never sees SUCCEEDED without a reward

No flags (`reward_pending`), no intermediate states, no race conditions. If we ever
move to async hooks or multi-worker, we can add a `reward_pending` flag then —
it's a backward-compatible addition. For the sync single-threaded store, pre-processors
are sufficient.

#### Constraints on hooks

Hooks must be **fast and synchronous** (no `await`, no blocking network calls):
- Volume reads: local disk, ~μs for KB files ✅
- `make_test_spec()`: pure Python, ~1ms ✅
- `get_eval_report()`: regex parsing of test log, ~1-5ms ✅
- Network calls to external APIs: ❌ (would block event loop)

If a hook needs async I/O in the future, we'd run it via `run_in_executor()` and
add the `reward_pending` flag. But the volume-based pattern avoids this entirely.

### User workflow: custom Docker image

```dockerfile
# Dockerfile.swebench
FROM agl-lite:latest
RUN pip install swebench
COPY hooks.py /app/hooks/
```

```bash
# Launch:
agl-lite serve --hooks /app/hooks/hooks.py
# or via env var:
AGL_HOOKS_MODULE=/app/hooks/hooks.py agl-lite serve
```

Server loads the module at startup, instantiates the hooks class, passes it to the store.

### Example: SWE-bench hooks

```python
# hooks.py — mounted into agl-lite container
from agl_lite.hooks import RolloutHooks

class SWEBenchHooks(RolloutHooks):
    def __init__(self, dataset_path, volume_path, namespace="swebench"):
        self.instances = {inst["instance_id"]: inst for inst in load_dataset(dataset_path)}
        self.volume_path = Path(volume_path)
        self.namespace = namespace

    def on_enqueue(self, request):
        """Map instance_id → Docker image + eval_script."""
        instance_id = request.input.get("instance_id") if isinstance(request.input, dict) else None
        if not instance_id or instance_id not in self.instances:
            return request  # passthrough for non-SWE-bench rollouts

        instance = self.instances[instance_id]
        spec = make_test_spec(instance, namespace=self.namespace)
        safe_id = instance_id.lower().replace("__", "_1776_")

        config = request.config or {}
        config["image"] = f"{self.namespace}/sweb.eval.x86_64.{safe_id}:latest"
        config.setdefault("environment_variables", {}).update({
            "AGL_EVAL_SCRIPT": spec.eval_script,
        })
        request.config = config
        return request

    def on_succeeded(self, rollout, events, store):
        """Grade using official swebench harness."""
        instance_id = rollout.input.get("instance_id") if isinstance(rollout.input, dict) else None
        if not instance_id or instance_id not in self.instances:
            return

        instance = self.instances[instance_id]
        spec = make_test_spec(instance, namespace=self.namespace)

        log_path = self.volume_path / rollout.rollout_id / "test_output.txt"
        patch_path = self.volume_path / rollout.rollout_id / "patch.diff"

        if not log_path.exists():
            reward = 0.0
        else:
            prediction = {
                "instance_id": instance_id,
                "model_patch": patch_path.read_text() if patch_path.exists() else "",
                "model_name_or_path": "agl-lite",
            }
            report = get_eval_report(spec, prediction, str(log_path),
                                     include_tests_status=True)
            reward = 1.0 if report[instance_id]["resolved"] else 0.0

        # Post reward event — happens atomically before update_rollout returns
        attempt_id = rollout.succeeded_attempt_id or "unknown"
        store.add_event(rollout.rollout_id, attempt_id, "reward", {"value": reward})
```

### Example: Math hooks (trivial)

```python
class MathHooks(RolloutHooks):
    def on_succeeded(self, rollout, events, store):
        """Agent already posted reward — no-op needed."""
        pass  # math agent posts its own reward via AGL_EVENT_URL
```

### Impact on daemon and architecture

The daemon becomes fully task-agnostic:

```
Trainer (VERL PPO)           agl-lite server (with hooks)
─────────────────           ─────────────────────────────
set_up_data_and_server()
  → POST /api/rollouts       → on_enqueue hook transforms each request
                                (image, eval_script, env vars)
                              → rollouts persisted to store

run_until_all_finished()
  → poll GET /api/rollouts    → controller creates K8s jobs
                              → jobs complete
                              → PATCH {status: succeeded}
                              → on_succeeded hook grades + posts reward
                                (atomic — reward is in store before response)

get_train_data_batch()
  → GET /api/events           → returns events (with reward already there)
  → triplets → tensors → DataProto
```

| Current | With hooks |
|---------|-----------|
| Daemon receives raw data, builds rollout config | Daemon sends raw data, hook builds config |
| Reward posted by container or algorithm script | Hook posts reward atomically on completion |
| New task = modify daemon + algorithm script | New task = write hooks.py + build Docker image |
| Task deps (swebench) in algorithm process | Task deps in server container (user-built image) |
| Daemon is task-aware (~500 lines) | Daemon is task-agnostic (~300 lines) |

### Why this is a differentiation point

1. **Simplest possible user interface**: write one Python file with 2 methods, build a
   Docker image. No separate processes, no webhook infra, no polling for rewards.
2. **Atomicity for free**: single-threaded sync store means hooks + transitions are
   indivisible. No race conditions, no flags, no intermediate states.
3. **Official grading on the hot path**: `get_eval_report()` runs inside the hook with
   full access to the store. Results are immediately available — zero extra round trips.
4. **Composable**: same daemon, same trainer config for any task. Just swap the Docker image.
5. **Compared to Agent Lightning**: their 1154-line monolithic daemon becomes:
   - hooks.py (~50-100 lines per task, user-written)
   - daemon (~300 lines, task-agnostic, maintained by infra)
   - agl-lite server (HTTP API + hook integration)

---

## Math-poc restructuring with hooks [completed]

**Goal**: Restructure `examples/math-poc` to use Store Hooks, unify the two rl_loop
scripts into one, and organize mode-specific files into subfolders.

### Changes

1. **Unified `rl_loop.py`** (~200 lines): task-agnostic orchestration. Registers
   resources + model, enqueues raw dataset rows as `input`, polls, fetches events
   (rewards already posted by hooks), logs results. No `build_tasks`, no `compute_reward`.

2. **Mode subfolders** (`mock/`, `vllm/`): each contains mode-specific files:
   - `hooks.py` — `MathMockHooks` / `MathVllmHooks`
   - `.env.example` — mode config
   - `gateway-config.yaml` — param injection (vllm only needs `return_token_ids`)
   - `job-template.yaml` — pod spec (could differ per mode)

3. **`run.sh`** takes mode argument: `run.sh [mock|vllm]` (default: `vllm`).
   Reads `.env.example` from the mode subfolder. Passes `--hooks` to `agl-lite serve`.

4. **Hook responsibilities**:
   - `on_enqueue`: set `config.image`, `config.environment_variables.AGL_TASK_INPUT`
     (question text for mock with `\boxed{}` embedding, plain question for vllm),
     `AGL_MODEL_NAME`, stash `ground_truth` in `metadata`.
   - `on_succeeded`: extract answer from `agent_output` event, compare with
     `ground_truth` (from `metadata` or `rollout.input`), post reward event.

5. **Shared files** stay at top level: `agents/`, `data/`, `Dockerfile.agent`, `README.md`.

### Files

```
examples/math-poc/
├── rl_loop.py                 # unified (replaces mock_rl_loop.py + rl_loop.py)
├── run.sh                     # run.sh [mock|vllm]
├── README.md
├── Dockerfile.agent
├── agents/
│   ├── qa_agent.py
│   └── README.md
├── data/
│   └── gsm8k_sample.jsonl
├── mock/
│   ├── hooks.py               # MathMockHooks
│   ├── .env.example
│   ├── gateway-config.yaml
│   ├── job-template.yaml
│   └── k8s-mockai.yaml
└── vllm/
    ├── hooks.py               # MathVllmHooks
    ├── .env.example
    ├── gateway-config.yaml
    └── job-template.yaml
```

---

## Phase 5: VERL Algorithm Integration

**Goal**: Connect VERL's PPO training loop to agl-lite as the data pipe. agl-lite handles
rollout orchestration, event collection, and triplet extraction; VERL handles tensor
construction and weight updates.

### Background: Agent Lightning daemon structure (1154 lines)

The `AgentModeDaemon` in Agent Lightning is the bridge between the store and the VERL trainer.
The trainer calls 4 daemon methods: `set_up_data_and_server`, `run_until_all_finished`,
`get_train_data_batch`, `clear_data_and_server`. The daemon internally calls 4 store methods
(`add_resources`, `enqueue_many_rollouts`, `wait_for_rollouts`, `query_spans`) plus an adapter
(`spans → triplets`), then builds padded tensors from triplets.

Breakdown of what to replace vs reuse:

| Category | Lines | % | Action |
|---|---|---|---|
| Store interaction | 209 | 18% | **Replace** (agl-lite HTTP API) |
| Proxy server | 141 | 12% | **Drop** (gateway replaces) |
| Init (mixed) | 72 | 6% | **Simplify** |
| `get_train_data_batch` | 328 | 28% | **Reuse** |
| Multimodal (mrope) | 63 | 5% | **Reuse** (not tested in MVP) |
| Validation/metrics | 106 | 9% | **Reuse** |
| Utilities (padding) | 157 | 14% | **Reuse** |

### Key design: Triplet API in agl-lite

Instead of fetching raw events and converting on the VERL side, agl-lite itself provides
a triplet endpoint. This keeps event→triplet conversion in the data pipe where it belongs.

```
Agent Lightning original:
  daemon → store.query_spans() → adapter.adapt(spans) → List[Triplet]
           ~~~~~~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~~~~~~~
           complex Span objects   600-line tree adapter

agl-lite:
  daemon → GET /api/rollouts/{id}/triplets → List[Triplet]
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           server does events→triplets internally (~50 lines)
```

The triplet extraction logic in agl-lite is simple because events are flat:
- `model_request` events have `response.prompt_token_ids` and `response.choices[].token_ids`
- `reward` events have `data.value`
- Match by order (each model_request pairs with the following reward)
- Streaming responses: gather `token_ids` across SSE chunks, `prompt_token_ids` from first chunk

### Phase 5a: Triplet API (server-side, in agl-lite) ✅

- [x] **5a.1**: `format=triplet` on `GET /api/events` — trims model_request to
  `prompt_token_ids` + `response_token_ids`, reward to scalar value.
  Handles streaming (gather token_ids across SSE chunks) and non-streaming.
  4 API tests.

- [x] **5a.2**: `AglLiteClient.get_events(format="triplet")` — passes format param.

### Phase 5b: AglLiteDaemon (VERL-side bridge) ✅

Implemented as standalone class (not subclass) in `agl_lite/verl/daemon.py`.
Uses `AglLiteClient` for all HTTP calls. Same 4 methods the trainer expects.

**Dependency direction**: Agent Lightning trainer → agl-lite HTTP API.
agl-lite is a standalone HTTP service. AglLiteDaemon copies the tensor math
from Agent Lightning so that agl-lite can eventually be a full replacement.

```
agl-lite (this repo)                VERL trainer
─────────────────────               ─────────────
HTTP API:                           AglLiteDaemon:
  POST /api/rollouts         ←───    set_up_data_and_server()
  GET  /api/rollouts/{id}    ←───    run_until_all_finished()
  GET  /api/events?format=triplet ←  _async_validate_data()
  POST /api/resources/models ←───    _async_set_up()
                                    get_train_data_batch() → DataProto
```

`agl_lite/verl/daemon.py` (851 lines):
- **NEW** (187 lines): store interaction via AglLiteClient
  - `_async_set_up`: `client.register_models()` + `client.enqueue_rollouts()`
  - `_async_validate_data`: `client.get_events(format="triplet")` → Triplet/RolloutLegacy
  - `_async_run_until_finished`: poll `client.get_rollout()` for succeeded status
  - No proxy server, no adapter, no LightningStore
- **COPIED** (510 lines): from agent-lightning `AgentModeDaemon`
  - `get_train_data_batch`: triplets → padded tensors → DataProto (transition + trajectory levels)
  - Multimodal (mrope, image grid for Qwen2-VL)
  - Utilities (left/right padding, token matching, native conversion)
  - Validation/metrics

9 tests (5 utility, 4 daemon with real agl-lite server via ASGI transport).

### Phase 5c: Full training loop E2E

- [ ] Training script using agl-lite + VERL on Qwen2.5-1.5B-Instruct
- [ ] Weight update protocol: after PPO step, update vLLM model weights
- [ ] Multi-iteration training with measurable reward improvement

---

## Phase 6: Polish

- [ ] Structured logging (JSON, with rollout_id/attempt_id context)
- [ ] Prometheus metrics (optional)
- [ ] Docker images for agl-lite serve and controller
- [ ] CI/CD pipeline
- [ ] User documentation beyond get_started.md

---

## Pre-Implementation Decisions (Frozen)

These are settled and should not be revisited during implementation:

| Decision | Resolution |
|----------|-----------|
| Package layout | One package, two entrypoints |
| Controller-to-service communication | HTTP only, no shared memory |
| Store backend (MVP) | In-memory, single instance |
| Event ordering | Append order in per-rollout list, no sequence counter |
| `event_id` | Removed — events identified by position in list |
| `timestamp` | Assigned by store at write time |
| `job_template` | Raw K8s pod spec dict, loaded from YAML file. No typed schema — any valid K8s field works. `overrides` on `RolloutConfig` for per-rollout overrides. |
| Auth | Single `AGL_KEY` for all components, no role-based access for MVP. `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` trick for agents. |
| Health endpoint | `GET /healthz`, no auth |
| Error codes | 401 missing/invalid key, 404 rollout not found, 409 invalid transition |
| Archive format | JSONL, user-specified file path (`*.jsonl`). Append if file exists, create if not. Includes rollout + events + resources per archive call. |
| Gateway config | Static YAML at startup. List-based routes: `[{model_in, model_out, params}]`, first match wins. Wildcard support. |
| Model routing | Per-model round-robin. `(model, endpoint)` composite key. Version per server. Optional `token`. |
| Namespace | Single namespace per controller instance. Manifests omit namespace, applied via `-n`. |
| `timeout` / `max_retries` | Map to K8s Job `activeDeadlineSeconds` / `backoffLimit` |
| Agent auth injection | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` env vars via `secretKeyRef` to `agl-lite-keys/AGL_KEY`. |
| vLLM deployment | Docker container (`vllm/vllm-openai:latest`) on host, separate from agl-lite lifecycle. `scripts/start_vllm.sh` for convenience. |
| vLLM topology | agl-lite colocated with algorithm + vLLM on host. Controller + agents in minikube. Gateway → vLLM is localhost. Agents reach host via `host.minikube.internal`. |
| Gateway param injection | `gateway-config.yaml` with `params.add` injects RL-specific params (e.g., `return_token_ids: true`). Event captures prepared body (post-injection). |
| Event request body | Captures the prepared body (after gateway param injection), not the original agent request. RL algorithm sees exactly what was sent to the model server. |
