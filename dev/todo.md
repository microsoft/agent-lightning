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

### Key Design Decisions to Discuss

#### (A) Container image strategy
SWE-bench provides per-instance Docker images (`swebench/sweb.eval.x86_64.<instance_id>:latest`).
In the math-poc, we use a single `math-agent:dev` image. For SWE-bench, the `job_template`
needs per-rollout image overrides via the existing `config.overrides` mechanism on `EnqueueRolloutRequest`.

**Proposal**: The algorithm script maps `instance_id → docker image tag` and passes the image
as a per-rollout override:
```python
EnqueueRolloutRequest(
    resources_id=resources_id,
    input=json.dumps({"instance_id": ..., "problem_statement": ...}),
    config={
        "overrides": {
            "containers": [{"name": "agent", "image": f"swebench/sweb.eval.x86_64.{instance_id}:latest"}]
        },
        "environment_variables": {"AGL_CODING_AGENT": "claude_code"}
    }
)
```

#### (B) Coding agent scripts (pluggable)
Each coding agent has an install + run script under `examples/swe_bench/agents/`:
- `agents/claude_code/install.sh` — install claude CLI
- `agents/claude_code/run.sh` — launch claude code with the problem statement
- `agents/mini_swe_agent/install.sh` — `pip install mini-swe-agent` (or inline)
- `agents/mini_swe_agent/run.sh` — launch the agent

A shared entrypoint script `agents/entrypoint.sh` reads `AGL_CODING_AGENT` env var
and dispatches to the right agent's install + run scripts. After the agent finishes,
the entrypoint does `git diff HEAD` to capture the patch and posts it as an `agent_output` event.

#### (C) Mountable files (e.g., `CLAUDE.md`)
Files like `CLAUDE.md` (system prompt / instructions for claude code) live in
`examples/swe_bench/agents/claude_code/CLAUDE.md`. These are baked into the agent
image via a ConfigMap or directly into the entrypoint script (copy to `/testbed/CLAUDE.md`).

**Proposal**: Use a ConfigMap mounted into the job container. The `job_template` references it,
and the entrypoint copies relevant files to `/testbed/` before running the agent.

#### (D) Reward function — separate evaluation container
After the agent produces a patch (`agent_output` event with `data.patch`), the reward function in
the algorithm script:
1. Retrieves the patch from the `agent_output` event
2. Starts a **new** container with the **same** SWE-bench Docker image
3. Applies the patch (`git apply`)
4. Runs the golden test script (from `swebench.harness.test_spec.make_test_spec`)
5. Grades pass/fail using `swebench.harness.grading.get_eval_report`
6. Posts a `reward` event (1.0 = resolved, 0.0 = not resolved)

This reuses the evaluation logic from the original `swebench_utils/evaluation.py` but runs
on the **algorithm host** (not in K8s). The algorithm host needs Docker access.

**Alternative**: Run evaluation as a second K8s job. But this adds complexity and the algorithm
host already has Docker (for vLLM). Keep it simple — evaluate on host.

#### (E) Algorithm script structure
Similar to `rl_loop.py` in math-poc, but adapted for SWE-bench:
- `rl_loop.py`: register resources → load dataset → enqueue batch → poll → retrieve patches → evaluate → post rewards
- Uses vLLM as backend (gateway proxies LLM calls with `return_token_ids`)
- Configurable coding agent via env var
- Dataset: `swebench_samples.jsonl` (small subset) or full SWE-bench via `--dataset-path`

### Files to Create

```
examples/swe_bench/
├── README.md                          # setup + usage docs
├── rl_loop.py                         # algorithm script (enqueue, poll, evaluate, reward)
├── gateway-config.yaml                # same as math-poc (inject return_token_ids)
├── job-template.yaml                  # K8s pod spec for SWE-bench agent jobs
├── swebench_samples.jsonl             # small dataset for smoke testing
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py                    # reward function: apply patch → run tests → grade
├── agents/
│   ├── entrypoint.sh                  # shared entrypoint: dispatch to agent, capture patch
│   ├── claude_code/
│   │   ├── install.sh                 # install claude CLI
│   │   ├── run.sh                     # run claude code on the problem
│   │   └── CLAUDE.md                  # system instructions for claude code
│   └── mini_swe_agent/
│       ├── install.sh                 # install mini-swe-agent
│       └── run.sh                     # run mini-swe-agent on the problem
├── run.sh                             # one-command E2E runner
├── .env.vllm.example                  # env config for vLLM mode
└── Dockerfile.agent                   # NOT needed — uses SWE-bench images directly
```

### Open Questions

1. **Image pull policy**: SWE-bench images are large (~2-10GB). Should we pre-pull them?
   minikube can't easily pull from Docker Hub in CI. For local dev, `imagePullPolicy: IfNotPresent`.
2. **ConfigMap vs bake-in**: For `CLAUDE.md` and agent scripts — ConfigMap is more flexible but
   adds deploy complexity. Alternative: the entrypoint downloads them from a URL or they're
   passed as env vars (for small files).
3. **Evaluation timeout**: SWE-bench tests can take 5-30 min. Need `activeDeadlineSeconds` tuning.
4. **Dataset format**: Should we use the same JSONL format as the original example, or simplify
   to just `instance_id` + `problem_statement`?

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
