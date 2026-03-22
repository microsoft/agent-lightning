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

### Phase 5a: Triplet API (server-side, in agl-lite)

- [ ] **5a.1**: Triplet schema + extraction logic
  - `Triplet` model: `prompt: {"token_ids": [...], "image_urls": [...]}`,
    `response: {"token_ids": [...]}`, `reward: float | None`, `metadata: {}`
  - Extraction function: `events_to_triplets(events: list[Event]) → list[Triplet]`
  - Handle streaming (gather token_ids across chunks) and non-streaming responses
  - Unit tests with synthetic events

- [ ] **5a.2**: HTTP endpoint
  - `GET /api/rollouts/{rollout_id}/triplets` → `List[Triplet]`
  - Calls store to fetch events, runs extraction, returns triplets
  - Optional `?attempt_id=` filter (default: latest attempt)
  - Tests with mock store

- [ ] **5a.3**: E2E verification
  - Extend `rl_loop.py` to call triplet API after rollout completion
  - Verify: prompt_token_ids match, response token_ids match, reward matches
  - Log triplet summary (n_triplets, token lengths)

### Phase 5b: VERL-side integration (OPEN — two options, choose one)

**Dependency direction**: Agent Lightning trainer → agl-lite HTTP API.
agl-lite is a standalone HTTP service. The VERL-side code lives in
agent-lightning (or a bridge contrib), NOT in agl-lite.

```
agl-lite (this repo)                Agent Lightning (VERL trainer)
─────────────────────               ──────────────────────────────
HTTP API only:                      RayPPOTrainer
  POST /api/rollouts/enqueue  ←───    daemon.set_up_data_and_server()
  GET  /api/rollouts?status=  ←───    daemon.run_until_all_finished()
  GET  /api/rollouts/{id}/triplets ←  daemon._validate_data_v1()
  POST /api/resources/models  ←───    daemon._async_set_up()

No VERL/torch dependency.          Needs daemon that talks HTTP.
No knowledge of tensors/PPO.        get_train_data_batch() unchanged.
```

Both options reuse `get_train_data_batch()` (328 lines of tensor math) unchanged.
Both talk to agl-lite over HTTP to enqueue rollouts, poll, and fetch triplets.
The difference is how they integrate with Agent Lightning's class hierarchy.

#### Option A: Daemon subclass (in agent-lightning)

Subclass `AgentModeDaemon`, override the store-interaction methods (~210 lines).
Inherit `get_train_data_batch`, multimodal, validation, utilities (730 lines).
Lives in agent-lightning repo (e.g., `contrib/agentlightning/contrib/agl_lite/daemon.py`).

```python
# In agent-lightning repo — NOT in agl-lite
class AglLiteDaemon(AgentModeDaemon):
    """Daemon that talks to agl-lite HTTP API instead of LightningStore."""

    def __init__(self, agl_lite_url, agl_key, ...):
        # Skip proxy server setup, just init HTTP client + inherited fields
        ...

    async def _async_set_up(self, data, server_addresses, is_train=True):
        # POST /api/resources/models (register vLLM endpoint from server_addresses)
        # POST /api/rollouts/enqueue (batch enqueue with config)
        ...

    async def _validate_data_v1(self, rollout) -> RolloutLegacy:
        # GET /api/rollouts/{id}/triplets (agl-lite does events→triplets)
        # No adapter needed — server already converted
        ...

    async def _async_run_until_finished(self, verbose=True):
        # GET /api/rollouts?status=succeeded (poll until all done)
        ...
```

| Pro | Con |
|-----|-----|
| Minimal new code (~150 lines) | Inherits proxy/v0 baggage in class |
| Proven tensor math, no copy | `__init__` needs careful super() avoidance |
| Multimodal inherited for free | Fragile if upstream daemon changes |
| Trainer class (`AgentLightningTrainer`) unchanged | N/A |

#### Option B: Standalone interface (in agent-lightning or bridge package)

Clean class with same 4 methods the trainer expects. Copy `get_train_data_batch`
and utilities (~500 lines) from the daemon.

| Pro | Con |
|-----|-----|
| No inheritance baggage | Must copy ~500 lines of tensor math |
| Clean, fully self-contained | Multimodal must be copied explicitly |
| Easy to test independently | Tensor math may diverge from upstream |

#### Recommendation: Option A

Option A is the natural choice because:
1. **agl-lite has no new dependency** — it's just HTTP endpoints
2. **Agent Lightning owns the daemon** — subclassing in its own repo is clean
3. **Tensor math stays upstream** — no copy, no divergence
4. **Trainer unchanged** — just pass `daemon_cls=AglLiteDaemon` to the trainer
5. **~150 lines** vs ~650 lines for Option B

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
