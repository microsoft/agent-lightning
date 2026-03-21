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

- [x] **Phase 0: Schemas and Project Skeleton** — frozen data models, project structure, dev tooling
- [x] **Phase 0.4: State transition rules** — valid transitions table, cancel_requested rules, exhaustive tests
- [x] **Phase 1: In-Memory Store** — `InMemoryStore` with all operations, 156 tests passing
  - Rollout: enqueue_rollouts (batch), PATCH-style partial update (transition validation), cancel, query
  - Events: add_event (individual), query (smart attempt_id resolution), list_attempts
  - Resources: add, get_by_id, get_latest
  - Models: register_models (batch, upsert by model+endpoint), list, get_model_pool, remove by model/endpoints, remove_all
  - Archive: validate terminal, write JSONL (append), purge
  - Key refactors: `PatchRolloutRequest` (true PATCH semantics, `exclude_unset`), store methods accept request objects,
    batch interface for rollouts and models (`enqueue_rollouts`, `register_models`)

---

## Phase 2: HTTP API (agl-lite serve)

**Goal**: Full API surface, serving over HTTP, with auth middleware.

### Phase 2 Decisions

| # | Decision |
|---|---|
| 1 | Store sharing: `app.state.store` set in lifespan |
| 3 | `GET /api/rollouts/{rid}` returns `RolloutDetail` with `attempts: List[str]` |
| 4 | `DELETE /api/models/{model}`: optional body `{endpoints: [...]}` to remove specific servers; no body = remove pool. `DELETE /api/models` removes all. |
| 5 | Auth: single `AGL_KEY` for all components, no roles. Gateway checks `Authorization: Bearer` and `x-api-key`. Warning logged when unset. |
| 6 | Error format: `{"detail": "message"}` (FastAPI default `HTTPException`) |
| 7 | Gateway → model server: optional `token` field on `ModelServer`. If set, gateway sends `Authorization: Bearer <token>` |
| 8 | Event data: full request body + full response body, **no headers** |
| 12 | Batch-only for rollouts and models at both HTTP and store level. Store owns batch semantics (`enqueue_rollouts`, `register_models`). Events stay individual. |

### Phase 2 Open Questions

| # | Topic | Status |
|---|-------|--------|
| 5 | Auth: single `AGL_KEY`, no roles for MVP | **Resolved** |
| 7 | Gateway → model server auth: optional `token` on `ModelServer` | **Resolved** |
| 9 | Gateway route config: `model_in → model_out` mapping + per-route param adjustments | **Resolved** |
| 10 | Event ingestion: gateway-side `POST /rollout/{rid}/attempt/{aid}/events` | **Resolved** |
| 11 | Streaming: tee + buffer + event-write. Detail during implementation of `gateway/proxy.py`. | **Deferred to impl** |
| 12b | Batch-only for rollouts and models; individual for events. No shared config at batch level (client-side sugar). | **Resolved** |

### 2.1 FastAPI app (`agl_lite/server/app.py`) ✅
- [x] Lifespan: create `InMemoryStore`, set on `app.state.store`
- [x] Mount all route modules
- [x] Health endpoint: `GET /healthz` (no auth)
- [x] Warn when `AGL_KEY` not set

### 2.2 Auth (`agl_lite/server/auth.py`) ✅
- [x] Extract key from `Authorization: Bearer <key>` or `x-api-key: <key>` header
- [x] Validate key against `AGL_KEY`; empty key = auth disabled
- [x] 401 for missing/invalid key
- [x] `/healthz` exempt

### 2.3 Store API routes (`agl_lite/server/routes/`) ✅
- [x] `rollouts.py` — POST (batch), GET (query), GET/{rid} (RolloutDetail + attempts), PATCH/{rid}, POST/{rid}/cancel
- [x] `events.py` — GET /api/events (read-only; writes via gateway)
- [x] `models.py` — POST (batch register), GET (list), DELETE/{model} (optional body), DELETE (all)
- [x] `resources.py` — POST, GET /latest, GET/{id}
- [x] `archive.py` — POST /api/rollouts/archive

All routes delegate to Store methods. Thin HTTP layer.

### 2.4 Gateway module (`agl_lite/gateway/`) ✅
- [x] `config.py` — load YAML route config (`model_in → model_out` + `params.add`/`params.drop`)
- [x] `router.py` — model routing (resolve model_in → model_out), server selection (round-robin per model pool), param adjustment
- [x] `proxy.py` — HTTP forwarding via httpx (non-streaming + streaming), event capture (request body + response body, no headers)

### 2.5 Gateway routes (`agl_lite/server/routes/gateway.py`) ✅
- [x] Path parsing: extract `rollout_id`, `attempt_id` from `/rollout/{rid}/attempt/{aid}/...`
- [x] Event ingestion endpoint: `POST /rollout/{rid}/attempt/{aid}/events`
- [x] Rollout existence check (in-process dict lookup) for LLM proxy
- [x] Wire gateway module: parse model → route → select server → proxy → capture event
- [x] Edge cases: no servers for model (503), missing model field (400), rollout not found (404)

### 2.6 CLI (`agl_lite/cli.py`) ✅
- [x] `agl-lite serve --host --port --gateway-config` entrypoint
- [x] Reads `AGL_KEY` from env var

### 2.7 Integration tests ✅
- [x] Full API round-trip tests (enqueue → query → update → events → archive)
- [x] Auth tests (valid key, invalid key, auth disabled, healthz)
- [x] Gateway proxy tests (mock model server, routing, param adjustment, non-streaming, 503)
- [x] Gateway routing tests (model_in → model_out, passthrough, round-robin)
- [x] Streaming proxy test (SSE tee+buffer+capture)

**Deliverables**: Working HTTP service, all endpoints functional, auth enforced.

---

## Phase 3: K8s Controller ✅

**Goal**: Controller that reconciles rollouts into K8s Jobs.

### 3.0 Python client (`agl_lite/client.py`) ✅
- [x] `AglLiteClient` — thin typed HTTP client wrapping agl-lite API
- [x] Shared by controller and algorithm (no duplication)
- [x] Methods: rollouts (enqueue, query, get, patch, cancel, archive), events, models, resources
- [x] Uses httpx.AsyncClient with connection pooling
- [x] Tests: 15 unit tests against real FastAPI app (httpx ASGITransport)

### 3.1 Controller config (`agl_lite/controller/config.py`) ✅
- [x] `ControllerSettings` — namespace, poll_interval, max_queue_time, agl_lite_url, secret_name
- [x] Reads from env vars (pydantic-settings, AGL_ prefix)

### 3.2 Job spec builder (`agl_lite/controller/job_builder.py`) ✅
- [x] Pure function: `build_job_spec(rollout, job_defaults, controller_settings) -> dict`
- [x] `job_defaults` (from resources) + `rollout.config` merge
- [x] Env var injection: OPENAI_BASE_URL, OPENAI_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, AGL_TASK_INPUT, AGL_EVENT_URL
- [x] Secret ref injection for API keys (from `secret_name` setting)
- [x] `timeout` → `activeDeadlineSeconds`, `max_retries` → `backoffLimit`
- [x] `overrides` escape hatch merged into Job spec (deep merge)
- [x] Deterministic job name: `agl-rollout-{rollout_id}`
- [x] `ttlSecondsAfterFinished: 3600` for pod GC safety
- [x] 31 unit tests

### 3.3 Reconciler (`agl_lite/controller/reconciler.py`) ✅
- [x] K8sClient protocol for testability (mock in tests, kr8s in prod)
- [x] Resources cache: `dict[resources_id, ResourcesUpdate]`, persistent, no invalidation
- [x] Reconcile loop (`asyncio.gather`):
  1. `periodic_reconcile()` — query `queuing` rollouts → create Jobs; crash recovery
  2. `watch_jobs()` — watch Job events (Complete/Failed) → update rollout status
  3. Handle `cancel_requested` — delete Job, update status to `cancelled`
- [x] Job creation failure → stay in `queuing`, retry next cycle
- [x] Max queue time (default 1h) → `terminal_failed` with error_message
- [x] `find_succeeded_pod_uid` — list pods by job-name label
- [x] Uses `AglLiteClient` for store access, K8sClient protocol for K8s API
- [x] 19 unit tests with fully mocked K8s + API clients

### 3.4 CLI ✅
- [x] `agl-lite controller --agl-lite-url --namespace --secret-name`
- [x] Reads `AGL_KEY` from env
- [ ] kr8s adapter (`Kr8sClient`) — deferred to Phase 4 (needs real cluster)

### 3.5 Tests ✅
- [x] 31 unit tests for Job spec builder (merge logic, env var injection, overrides, mounts)
- [x] 19 unit tests for reconciler (create, cancel, crash recovery, watch events, caching)
- [x] 15 unit tests for Python client against FastAPI app

**Deliverables**: Controller creates/watches/deletes Jobs, updates rollout status correctly. 254 tests total.

---

## Phase 4a: E2E with Mock Model Server (CPU-only minikube) [discuss]

**Goal**: Prove the full agl-lite lifecycle works end-to-end, fast and deterministic, using a mock OpenAI-compatible server inside minikube.

### Phase 4a Decisions

| # | Decision |
|---|----------|
| 1 | **Mock server**: Use existing `~/mockai` (polly3d/mockai) — Node.js OpenAI-compatible mock with echo/random/fixed modes, streaming SSE, configurable delay. No custom mock needed. |
| 2 | **Failure modes**: 503 tested via model deregistration in agl-lite (gateway returns 503 when no servers). Agent retry tested via agent crash (env var `CRASH_ON_FIRST=1`). Mock itself stays healthy. |
| 3 | **Client CLI**: Add `agl-lite client` subcommand group wrapping `AglLiteClient` — query rollouts, events, models, etc. from command line. Useful for debugging, demos, E2E scripts. |
| 4 | **Examples folder**: `examples/agents/` (qa_agent, react_agent, shared Dockerfile), `examples/algorithm/` (run_batch.py). Agents use only `openai` SDK — no agl-lite import. |
| 5 | **Docker builds**: `minikube image build` with local context + `imagePullPolicy: Never`. agl-lite image: `COPY . /src && pip install /src[controller]`. Agent/mock: self-contained dirs. |
| 6 | **E2E cleanup**: Delete + recreate namespace at test start. Single `scripts/e2e_test.sh` wraps full lifecycle. |
| 7 | **Deployments**: agl-lite serve and controller as separate K8s Deployments (matches production topology). Algorithm script runs on host via `kubectl port-forward`. |

### 4a.1 Kr8s adapter (`agl_lite/controller/kr8s_adapter.py`) [discuss]
- [ ] Implement `Kr8sClient` satisfying the `K8sClient` protocol in reconciler
- [ ] Methods: create_job, delete_job, get_job, list_jobs, list_pods, watch_jobs
- [ ] Wire into `agl-lite controller` CLI entrypoint

### 4a.2 Client CLI (`agl-lite client`) [discuss]
- [ ] Typer subcommand group wrapping `AglLiteClient`
- [ ] Subcommands: `rollouts list`, `rollouts get <rid>`, `events list`, `models list`, `models register`, `resources get-latest`, etc.
- [ ] Reads `--url` and `AGL_KEY` from env/options
- [ ] Useful for debugging, demos, and E2E test scripts

### 4a.3 Mock OpenAI server (mockai) [discuss]
- [ ] Build `~/mockai` into minikube: `minikube image build -t mockai:dev ~/mockai`
- [ ] K8s Deployment + Service manifest (`examples/mock-openai/k8s.yaml`)
- [ ] Config: `MOCK_TYPE=echo` (returns last user message — deterministic, verifiable)
- [ ] No modifications to mockai needed

### 4a.4 Example agents (`examples/agents/`) [discuss]
- [ ] `qa_agent.py` — simplest: read `AGL_TASK_INPUT`, one LLM call via `OPENAI_BASE_URL`, print result
- [ ] `react_agent.py` — multi-turn: tool-use loop with multiple LLM calls (tests multi-event capture)
- [ ] Shared `Dockerfile` (build arg to select agent script)
- [ ] `CRASH_ON_FIRST=1` env var support in qa_agent (for retry test)
- [ ] Does NOT import agl-lite — proves language-agnostic contract

### 4a.5 Example algorithm script (`examples/algorithm/`) [discuss]
- [ ] `run_batch.py` — register resources, register mock model server, enqueue batch, poll, retrieve events
- [ ] Demonstrates full lifecycle using `AglLiteClient` or `agl-lite client` CLI

### 4a.6 K8s manifests and setup script [discuss]
- [ ] `deploy/namespace.yaml` — namespace + RBAC + secret
- [ ] `deploy/agl-lite.yaml` — agl-lite serve Deployment + Service
- [ ] `deploy/controller.yaml` — controller Deployment
- [ ] `deploy/mock-openai.yaml` — mockai Deployment + Service
- [ ] `scripts/e2e_setup.sh` — nuke namespace, rebuild images, apply all manifests, wait for ready
- [ ] Dockerfile for agl-lite (serve + controller share one image, different CMD)

### 4a.7 End-to-end tests [discuss]
- [ ] `scripts/e2e_test.sh` or `tests/e2e/test_minikube.py` — orchestrates full lifecycle
- [ ] Happy path: enqueue 5 rollouts → Jobs created → agents run → events captured → retrieve trajectories
- [ ] Cancel test: enqueue → cancel mid-run → verify cancelled status
- [ ] Retry test: agent crashes on first attempt (`CRASH_ON_FIRST=1`) → K8s retries → succeeds
- [ ] Weight update test: deregister model → agents get 503 → re-register → agents succeed

**Deliverables**: Working E2E demo on minikube (CPU-only), client CLI, validated get_started.md. All tests fast and deterministic.

---

## Phase 4b: E2E with Real vLLM (GPU) [backlog]

**Goal**: Validate agl-lite with a real vLLM inference server on GPU. Bridge to VERL integration.

### 4b.1 vLLM deployment
- [ ] Deploy vLLM on host GPUs (4× A6000), expose to minikube (NodePort or host network)
- [ ] Model selection (small model for fast iteration, e.g., Qwen-2.5-1.5B or similar)
- [ ] Register real vLLM endpoint as model server via agl-lite API

### 4b.2 Real inference E2E
- [ ] Reuse Phase 4a agent + algorithm script against real vLLM
- [ ] Verify event capture contains real model responses
- [ ] Weight update protocol: vLLM model reload + agl-lite model server re-registration

### 4b.3 Performance baseline
- [ ] Measure: rollout throughput, gateway proxy latency overhead, event capture overhead
- [ ] Compare direct vLLM vs gateway-proxied vLLM

**Deliverables**: Proven real-inference path, performance baseline. Prerequisite for Phase 5.

---

## Phase 5: Algorithm Integration (VERL)

**Goal**: Port VERL RL algorithm to consume agl-lite events. Assumes Phase 4b (real vLLM) is complete.

- [ ] Adapter: events → triplets (prompt, response, reward)
- [ ] VERL training loop using agl-lite API
- [ ] vLLM model server registration + weight update protocol
- [ ] Full RL training loop on minikube

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
| `job_defaults` schema | Typed `JobDefaults` model, validated at POST time. Known fields validated; `overrides` dict for unknown K8s fields. |
| Auth | Single `AGL_KEY` for all components, no role-based access for MVP. `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` trick for agents. |
| Health endpoint | `GET /healthz`, no auth |
| Error codes | 401 missing/invalid key, 404 rollout not found, 409 invalid transition |
| Archive format | JSONL, user-specified file path (`*.jsonl`). Append if file exists, create if not. Includes rollout + events + resources per archive call. |
| Gateway config | Static YAML at startup. Routes: `model_in → model_out` + per-route `params.add`/`params.drop`. No route = passthrough. |
| Model routing | Per-model round-robin. Model name = grouping key. Store: `Dict[model, Dict[endpoint, ModelServer]]`. |
| Model server identity | `(model, endpoint)` composite key. Version per server (supports online RL rolling updates). Optional `token` for gateway → model server auth. |
| Rollout existence check | On both LLM proxy and event ingestion (in-process, ~100ns) |
| Namespace | Single namespace per controller instance |
| `timeout` | Maps to K8s Job `activeDeadlineSeconds` |
| `max_retries` | Maps to K8s Job `backoffLimit` |
| Model routing | Per-model round-robin for MVP |
| Agent auth injection | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` env vars in Job spec, both via `secretKeyRef` to same `agl-lite-keys/AGL_KEY`. Gateway checks both `Authorization: Bearer` and `x-api-key` headers. |
