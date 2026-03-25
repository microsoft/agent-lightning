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

#### (D) Reward function — evaluation inside K8s containers

All computation runs on K8s (not the algorithm host). This is the principled approach:
the algorithm host only orchestrates via HTTP; Docker access is not required.

**How swebench evaluation works** (traced from `swebench.harness`):

1. `make_test_spec(instance)` → `TestSpec` with `eval_script` (a bash script)
2. `eval_script` does: reset tests to base_commit → apply `test_patch` (golden tests
   from the dataset) → run test command → revert tests
3. The eval script runs inside the **same SWE-bench Docker image** (it has the repo,
   conda env, dependencies pre-installed)
4. Test output is captured → `get_eval_report()` parses the log → resolves pass/fail

**Key insight**: `eval_script` is a self-contained bash script. `get_eval_report()` just
parses a text log. Both can run anywhere — no Docker SDK needed for grading.

**Approach — two-phase per rollout**:

Phase 1 (agent job, already in K8s):
- Coding agent solves the problem → posts `agent_output` with `data.patch`

Phase 2 (evaluation, algorithm-side):
- Algorithm retrieves the patch from `agent_output` event
- Algorithm generates `eval_script` using `make_test_spec(instance)` (Python call, no Docker)
- Algorithm generates a self-contained evaluator shell script that:
  (a) applies the model patch (`git apply`)
  (b) runs the eval_script
  (c) captures test output
  (d) posts a `reward` event (parse test output inline or use a small Python grading script)
- Algorithm enqueues a **second rollout** (evaluator job) using the same SWE-bench image,
  with the evaluator script passed via env var or ConfigMap

```
Algorithm                          K8s
─────────                          ───
1. enqueue agent rollout     →     agent job runs, posts agent_output(patch)
2. poll until succeeded      ←
3. get_events(agent_output)  ←
4. generate eval_script      (local Python: make_test_spec → eval_script)
5. enqueue evaluator rollout →     evaluator job: git apply + eval_script + post reward
6. poll until succeeded      ←
7. get_events(reward)        ←     reward=1.0 or 0.0
```

The evaluator job reuses the same K8s machinery. No Docker on algorithm host.

**Evaluator script strategy**: The algorithm creates a wrapper script that:
- Takes the patch (passed as `AGL_TASK_INPUT` or env var)
- Applies it with `git apply` (with fallbacks like `git apply --reject`, `patch -p1`)
- Runs the eval_script (generated by `make_test_spec`)
- Parses test output for PASS/FAIL markers
- Posts reward event to `AGL_EVENT_URL`

This can be a Python script (`evaluator.py`) mounted via ConfigMap that imports
a minimal grading function (re-implemented from `get_eval_report` without the Docker
SDK dependency, just log parsing).

#### (E) Algorithm script structure

Similar to math-poc `rl_loop.py`:
```
rl_loop.py:
  1. register resources (job_template, ConfigMap setup)
  2. load dataset (swebench_samples.jsonl or full SWE-bench)
  3. register model server (vLLM endpoint)
  4. for each batch:
     a. map instances → agent rollouts (per-instance image, coding agent config)
     b. enqueue agent rollouts → poll → collect agent_output events (patches)
     c. for each patch: generate eval_script → enqueue evaluator rollout
     d. poll evaluator rollouts → collect reward events
     e. aggregate results (resolved rate, etc.)
```

### Files to Create

```
examples/swe_bench/
├── README.md                          # setup + usage docs
├── rl_loop.py                         # algorithm script (enqueue, poll, evaluate, reward)
├── gateway-config.yaml                # same as math-poc (inject return_token_ids)
├── job-template.yaml                  # K8s pod spec — generic, image overridden per rollout
├── eval-job-template.yaml             # K8s pod spec for evaluator jobs (or reuse job-template)
├── swebench_samples.jsonl             # small dataset for smoke testing
├── evaluation/
│   ├── __init__.py
│   ├── eval_script_gen.py             # wraps make_test_spec → eval_script generation
│   └── grading.py                     # standalone log parser (from swebench.harness.grading,
│                                      #   no Docker SDK dep — just parses test output text)
├── agents/
│   ├── entrypoint.sh                  # shared entrypoint: dispatch to agent, capture patch
│   ├── evaluator.py                   # evaluator script: apply patch + run eval + post reward
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

### Remaining Open Questions

1. **Evaluation timeout**: SWE-bench tests can take 5-30 min. Need `activeDeadlineSeconds`
   tuning for evaluator jobs (separate from agent job timeout).
2. **Dataset format**: Use full SWE-bench JSONL format (same fields as original example)
   so `make_test_spec` works directly. The `input` field sent to agent can be a subset
   (instance_id + problem_statement); the full instance data stays in the algorithm.
3. **Evaluator passing mechanism**: The eval_script + model patch need to reach the evaluator
   container. Options:
   - (a) `AGL_TASK_INPUT` env var (JSON with patch + eval_script) — simple but eval_script can
     be large (multi-KB bash). K8s env var limit is 1MB, so usually fine.
   - (b) ConfigMap per evaluation batch — more robust for large scripts but adds K8s API calls.
   - (c) Evaluator fetches patch from agl-lite API (knows rollout_id of agent run) — cleanest
     but requires evaluator to call agl-lite API.
4. **ConfigMap creation for agent scripts**: `run.sh` creates the ConfigMap from `agents/`
   directory. Need to handle updates (delete + recreate) and namespace.
5. **Evaluator as second rollout vs sidecar**: Current design uses a second rollout (separate
   K8s Job). Alternative: evaluator runs as a sidecar or post-hook in the agent job, but
   this couples agent and evaluation lifecycle. Keep them separate for now.

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
