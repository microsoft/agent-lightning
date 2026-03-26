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
- [x] **Store Hooks**: `RolloutHooks` base class with `on_enqueue`/`on_succeeded`/`on_failed`.
  Hooks run as sync pre-processors inside store (atomic, single-threaded). `--hooks` CLI flag.
  `load_hooks()` dynamic loader. `metadata` field on Rollout for algorithm control indexes.
  `AGL_TASK_INPUT` removed from job_builder (hook sets it explicitly). 12 hook tests.
- [x] **Math-poc restructuring**: Unified `rl_loop_v2.py` (~200 lines) replaces two ~480-line
  scripts. Mode subfolders (`mock/`, `vllm/`) with per-mode `hooks.py`, `.env.example`,
  `gateway-config.yaml`, `job-template.yaml`. Algorithm only sets `input`, `resources_id`,
  `metadata`. E2E verified with vLLM (3/3 rollouts, hooks computed rewards). 7 hook tests.

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

### Architecture Overview

```
Compute backend (host)              K8s cluster
──────────────────────              ──────────────
agl-lite server                     controller
  + SWEBenchHooks                     ↕
  + swebench package                agent pods (per-instance SWE-bench images)
  + gateway → vLLM (internal)         ↕
                                    agl-lite server (via cluster DNS or host.minikube.internal)
vLLM instance(s)
  (internal endpoints)
```

**Deployment**: agl-lite runs on the compute backend (`--controller-only` mode) because
the gateway needs direct access to internal model server endpoints. On minikube,
`deploy.sh` auto-patches CoreDNS so pods resolve `host.minikube.internal`.

**Data flow**:
```
Algorithm (rl_loop.py)         Server hooks                   K8s container
─────────────────────          ────────────                   ──────────────
sends raw dataset rows    →    on_enqueue:                →   entrypoint.sh:
  input = full instance          set image per instance        1. install + run agent
  metadata = {agent, ...}        generate eval_script          2. git diff → post patch
                                  set env vars                  3. run eval_script
                                                                4. post test_output artifact

polls for completion      ←    on_succeeded:              ←   container exits 0
gets events (reward)             read artifact (disk)
                                 grade via swebench
                                 post reward event
```

**Key principle**: The algorithm (`rl_loop.py`) is task-agnostic — it only sends raw
dataset rows as `input` and polls for results. SWE-bench-specific logic is split between
the hook (image selection, eval setup, grading) and the container (agent execution, eval).

### Design Decisions

#### (A) Hook: `on_enqueue` — per-rollout image + eval setup

The `SWEBenchHooks.on_enqueue` hook transforms each raw dataset row into a runnable
rollout configuration:

```python
class SWEBenchHooks(RolloutHooks):
    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        instance = request.input  # full SWE-bench JSONL row
        instance_id = instance["instance_id"]

        # 1. Set per-instance Docker image
        safe_id = instance_id.lower().replace("__", "_1776_")
        request.config.image = f"sweb.eval.x86_64.{safe_id}:latest"

        # 2. Generate eval_script via swebench
        test_spec = make_test_spec(instance)
        request.config.environment_variables.update({
            "AGL_TASK_INPUT": instance["problem_statement"],
            "AGL_EVAL_SCRIPT": test_spec.eval_script,
            "AGL_EVAL_META": json.dumps({
                "FAIL_TO_PASS": test_spec.FAIL_TO_PASS,
                "PASS_TO_PASS": test_spec.PASS_TO_PASS,
                "instance_id": instance_id,
            }),
            "AGL_CODING_AGENT": os.environ.get("AGL_CODING_AGENT", "mini_swe_agent"),
        })
        return request
```

This requires the `swebench` package installed on the server (via custom Dockerfile).
`make_test_spec()` is CPU-only, ~ms per call — safe for sync hooks.

#### (B) Container: agent execution + evaluation (no grading)

The container runs the coding agent and evaluation, then posts raw artifacts.
**Grading happens in the hook**, not in the container.

```bash
# entrypoint.sh (runs inside SWE-bench Docker image at /testbed)

# Phase 1: Agent
source /agl/agents/${AGL_CODING_AGENT}/install.sh
source /agl/agents/${AGL_CODING_AGENT}/run.sh

# Phase 2: Capture patch
PATCH=$(git diff HEAD)
curl -X POST "$AGL_EVENT_URL" \
  -d '{"event_type":"agent_output","data":{"patch":"...","instance_id":"..."}}'

# Phase 3: Evaluate (run tests, capture output)
echo "$AGL_EVAL_SCRIPT" > /tmp/eval.sh
bash /tmp/eval.sh > /tmp/test_output.txt 2>&1

# Phase 4: Post test output as artifact (large file, stored on disk by server)
curl -X POST "$AGL_EVENT_URL" \
  -d '{"event_type":"artifact","data":{"filename":"test_output.txt","content":"..."}}'
```

The container does NOT grade — it posts the raw test output as an `artifact` event.

**Why single-container works** (traced from `swebench.harness`):
- `eval_script` only touches **test files** (from `test_patch`), not source files
- The agent's source modifications are untouched by evaluation
- `eval_script` is self-contained (conda env, repo, deps pre-installed in image)
- `make_test_spec()` only needs the dataset instance, not agent output — pre-generated in hook

#### (C) Hook: `on_succeeded` — grading via official swebench tools

The `on_succeeded` hook reads artifacts from disk and grades using official swebench:

```python
def on_succeeded(self, rollout, events, store):
    # 1. Read test output from artifact (written to disk by store)
    artifact_path = self._find_artifact(events, "test_output.txt")
    test_log = Path(artifact_path).read_text()   # ~μs, local disk

    # 2. Reconstruct TestSpec from rollout.input
    test_spec = make_test_spec(rollout.input)

    # 3. Extract patch from agent_output event
    patch = self._extract_patch(events)

    # 4. Grade using official swebench
    report = get_eval_report(
        test_spec=test_spec,
        prediction={"instance_id": ..., "model_patch": patch, ...},
        test_log_path=artifact_path,
        include_tests_status=True,
    )
    resolved = report[instance_id]["resolved"]

    # 5. Post reward event
    store.add_event(rollout.rollout_id, attempt_id, "reward", {
        "value": 1.0 if resolved else 0.0,
        "resolved": resolved,
        "instance_id": instance_id,
    })
```

**Why grade in the hook** (not in the container):
- Uses official `get_eval_report()` — credible, maintained, no reimplementation
- Grading logic is testable Python, not bash
- Container stays simple: just run agent + eval + post artifacts
- `swebench` package only needed on server, not baked into every SWE-bench image

#### (D) Artifact events — large file handling

Test output logs can be large (100KB–10MB). Storing them in-memory as regular events
would bloat the store. Instead, `artifact` is a special event type:

- **Store handling**: When `event_type == "artifact"`, the store writes
  `data["content"]` to disk (`<artifact_dir>/<rollout_id>/<filename>`) and replaces
  the event data with a lightweight reference (`{filename, path, size}`).
- **Hook access**: `on_succeeded` reads artifacts from disk (fits sync constraint).
- **Archiving**: Artifact content is skipped when archiving to JSONL. The files
  persist on disk alongside the archive. (Details deferred to backlog.)

Implementation: ~15 lines in `InMemoryStore.add_event()`, configurable via
`--artifact-dir` (default: `/data/agl-artifacts/`).

#### (E) Coding agent scripts (pluggable)

Each coding agent has `install.sh` + `run.sh` under `agents/<name>/`:
- `agents/claude_code/` — install claude CLI, run with problem statement + CLAUDE.md
- `agents/mini_swe_agent/` — lightweight Python agent for testing/OSS models

Shared `agents/entrypoint.sh` dispatches based on `AGL_CODING_AGENT` env var.

Agent scripts are mounted into the container via ConfigMap:
```bash
kubectl create configmap swe-agent-scripts --from-file=agents/ -n $NS
```
Job template mounts at `/agl/agents/`:
```yaml
containers:
  - name: agent
    command: ["bash", "/agl/agents/entrypoint.sh"]
    volumeMounts:
      - name: agent-scripts
        mountPath: /agl/agents
volumes:
  - name: agent-scripts
    configMap:
      name: swe-agent-scripts
      defaultMode: 0755
```

#### (F) Container image — per-rollout via hook

Each SWE-bench instance needs its own Docker image (`sweb.eval.x86_64.<id>:latest`).
The hook sets `config.image` per-rollout; the job template provides everything else.

Image pull policy: `imagePullPolicy: IfNotPresent` (naive). Images must be pre-built
and available in the cluster's registry. For minikube, build into minikube's Docker
daemon. For production, push to a container registry.

Future optimization: Epoch AI's trimmed images, DaemonSet pre-pull, or init containers.

#### (G) Algorithm script (`rl_loop.py`)

Task-agnostic, same structure as math-poc:

```python
# rl_loop.py (simplified)
dataset = load_jsonl("swebench_samples.jsonl")  # full SWE-bench rows

for batch in batches(dataset, batch_size):
    rollouts = [
        EnqueueRolloutRequest(
            resources_id=resources_id,
            input=instance,         # raw dataset row — hook handles everything
            metadata={"sample_idx_in_batch": i},
        )
        for i, instance in enumerate(batch)
    ]
    enqueue(rollouts)
    poll_until_done()
    events = get_events()  # reward events posted by on_succeeded hook
    # aggregate: resolved rate, patches for training data
```

The algorithm does NOT call `make_test_spec()` or know about SWE-bench internals.
It just sends raw dataset rows and reads reward events.

### Files to Create

```
examples/swe_bench/
├── README.md                          # setup + usage docs
├── rl_loop.py                         # algorithm script (task-agnostic)
├── hooks.py                           # SWEBenchHooks (on_enqueue + on_succeeded)
├── Dockerfile.server                  # agl-lite + swebench package + hooks.py
├── gateway-config.yaml                # inject return_token_ids for vLLM
├── job-template.yaml                  # K8s pod spec — generic, image overridden by hook
├── .env.example                       # env config
├── run.sh                             # one-command E2E runner
├── swebench_samples.jsonl             # small dataset for smoke testing
├── agents/
│   ├── entrypoint.sh                  # shared entrypoint: agent → eval → post artifacts
│   ├── claude_code/
│   │   ├── install.sh                 # install claude CLI
│   │   ├── run.sh                     # run claude code
│   │   └── CLAUDE.md                  # system instructions
│   └── mini_swe_agent/
│       ├── install.sh                 # install mini-swe-agent
│       └── run.sh                     # run mini-swe-agent
```

Core changes needed:
```
agl_lite/store/memory.py               # artifact event handling (~15 lines)
agl_lite/server/config.py              # --artifact-dir setting
scripts/deploy.sh                      # --controller-only, CoreDNS auto-patch (done)
```

### Open Questions

1. **Timeout budget**: Agent + evaluation share one `activeDeadlineSeconds`. Agent may take
   30–60 min, evaluation 5–30 min. Default 90 min? Configurable via `AGL_TIMEOUT` env var.

2. **eval_script size in env var**: Typical ~2KB, K8s limit 1MB. Safe for all instances.
   If edge cases exceed, fall back to ConfigMap mount.

3. **ConfigMap vs PVC for agent scripts**: ConfigMap is fine for shell scripts + markdown.
   For large files or binary assets, switch to PVC or init container download.

4. **Image availability**: SWE-bench images must be pre-built. Need documentation on
   how to build them (`python -m swebench.harness.docker_build ...`). For minikube,
   build inside minikube's Docker daemon.

5. **`swebench` package on server**: The hook calls `make_test_spec()` and
   `get_eval_report()`, requiring the `swebench` pip package. Custom `Dockerfile.server`
   extends base agl-lite image. The package is pure Python (~10MB), no GPU dependencies.

6. **Artifact archiving**: Artifact files persist on disk but are skipped in JSONL archive.
   Future work: configurable retention, cleanup policy, S3/GCS upload. (Backlog.)

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
