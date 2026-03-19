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
- [x] **Phase 1: In-Memory Store** — `InMemoryStore` with all operations, 114 tests passing
  - Rollout: enqueue, PATCH-style partial update (transition validation), cancel, query
  - Events: add, query (smart attempt_id resolution), list_attempts
  - Resources: add, get_by_id, get_latest
  - Models: register (upsert by endpoint), list, remove by endpoint, remove_all
  - Archive: validate terminal, write JSONL (append), purge
  - Key refactors: `UpdateRolloutRequest` → `PatchRolloutRequest` (true PATCH semantics, `exclude_unset`),
    dropped `model_id` (endpoint is natural key), store methods accept request objects directly

---

## Phase 2: HTTP API (agl-lite serve)

**Goal**: Full API surface, serving over HTTP, with auth middleware.

### Phase 2 Decisions

| # | Decision |
|---|---|
| 1 | Store sharing: `app.state.store` set in lifespan |
| 2 | `POST /api/rollouts`: single vs batch distinguished by `rollouts` key presence |
| 3 | `GET /api/rollouts/{rid}` returns `RolloutDetail` with `attempts: List[str]` |
| 4 | `DELETE /api/models`: optional `?endpoint=` query param (present=remove one, absent=remove all) |
| 5 | Auth: single `AGL_KEY` for all components, no roles. Gateway checks `Authorization: Bearer` and `x-api-key` |
| 6 | Error format: `{"detail": "message"}` (FastAPI default `HTTPException`) |
| 7 | Gateway → model server: optional `token` field on `ModelServer`. If set, gateway sends `Authorization: Bearer <token>` |
| 8 | Event data: full request body + full response body, **no headers** |
| 12 | Batch config merging in route handler; store sees individual `EnqueueRolloutRequest`s |

### Phase 2 Open Questions

| # | Topic | Status |
|---|-------|--------|
| 5 | Auth: single `AGL_KEY`, no roles for MVP | **Resolved** |
| 7 | Gateway → model server auth: optional `token` on `ModelServer` | **Resolved** |
| 9 | Parameter adjustment YAML format | TBD |
| 10 | Event ingestion endpoint location (agent-side vs API-side) | TBD |
| 11 | Streaming: buffer/tee/event-write flow details | TBD |
| 12b | Batch-only APIs? Drop individual `add_event`/`register_model` in favor of list-of-one? | TBD |

### 2.1 FastAPI app (`agl_lite/server/app.py`)
- [ ] Lifespan: create `InMemoryStore`, set on `app.state.store`; load gateway config, load API keys from env
- [ ] Mount all route modules
- [ ] Health endpoint: `GET /healthz` (no auth)

### 2.2 Auth middleware (`agl_lite/server/auth.py`)
- [ ] Extract key from `Authorization: Bearer <key>` or `x-api-key: <key>` header
- [ ] Validate key against `AGL_KEY`
- [ ] 401 for missing/invalid key
- [ ] `/healthz` exempt

### 2.3 Store API routes (`agl_lite/server/routes/`)
- [ ] `rollouts.py` — POST (single/batch via `rollouts` key), GET, GET/{rid} (returns `RolloutDetail` with attempts), PATCH/{rid}, POST/{rid}/cancel
- [ ] `events.py` — GET /api/events
- [ ] `models.py` — POST (list), GET, DELETE (optional `?endpoint=`)
- [ ] `resources.py` — POST, GET /latest, GET/{id}
- [ ] `archive.py` — POST /api/rollouts/archive

All routes delegate to Store methods. Thin HTTP layer — validate request, call store, return response. Batch config merging for `POST /api/rollouts` happens in route handler.

### 2.4 Gateway proxy routes (`agl_lite/server/gateway.py`)
- [ ] Path parsing: extract `rollout_id`, `attempt_id` from `/rollout/{rid}/attempt/{aid}/v1/...`
- [ ] Rollout existence check (in-process dict lookup)
- [ ] Model server selection (round-robin from store.models)
- [ ] Parameter adjustment (load from YAML config at startup)
- [ ] Non-streaming proxy: forward request, capture response (body only, no headers), write event, return
- [ ] Streaming proxy: tee stream to client + buffer, assemble on completion, write event
- [ ] Edge cases: client disconnect, backend error, no model servers (503)
- [ ] Event ingestion endpoint: `/rollout/{rid}/attempt/{aid}/events`

### 2.5 CLI (`agl_lite/cli.py`)
- [ ] `agl-lite serve --host --port --gateway-config` entrypoint
- [ ] Reads `AGL_KEY` from env var

### 2.6 Integration tests
- [ ] Full API round-trip tests (enqueue → query → update → events → archive)
- [ ] Auth tests (valid key, wrong role, missing key, healthz)
- [ ] Gateway proxy tests (mock model server, streaming, non-streaming, 503)

**Deliverables**: Working HTTP service, all endpoints functional, auth enforced.

---

## Phase 3: K8s Controller

**Goal**: Controller that reconciles rollouts into K8s Jobs.

### 3.1 Controller core (`agl_lite/controller/reconciler.py`)
- [ ] HTTP client for agl-lite API (using controller key)
- [ ] Reconcile loop:
  1. Query `queuing` rollouts → create Jobs
  2. Watch Job events (Complete/Failed) → update rollout status
  3. Handle `cancel_requested` → delete Job, update status
  4. Periodic reconcile for crash recovery
- [ ] Job spec builder:
  - Fetch resources snapshot (deduplicate per resources_id)
  - Extract `job_defaults`
  - Merge with `rollout.config`
  - Inject env vars (OPENAI_BASE_URL, OPENAI_API_KEY, AGL_TASK_INPUT, AGL_EVENT_URL)
  - Deterministic job name: `agl-rollout-{rollout_id}`
- [ ] `find_succeeded_pod_uid` — per `docs/design/1_k8s_controller.md`

### 3.2 Job template rendering
- [ ] `job_defaults` (from resources) + `rollout.config` → K8s Job YAML
- [ ] Secret ref injection for OPENAI_API_KEY
- [ ] Mount rendering (PVC, hostPath, ConfigMap)
- [ ] `timeout` → `activeDeadlineSeconds`
- [ ] `max_retries` → `backoffLimit`

### 3.3 CLI
- [ ] `agl-lite controller --agl-lite-url --namespace --secret-name`
- [ ] Reads `AGL_KEY` from env

### 3.4 Tests
- [ ] Unit tests for Job spec builder (merge logic, env var injection)
- [ ] Unit tests for state transition mapping (Job conditions → rollout status)
- [ ] Integration tests with mock K8s API (or kind/minikube)

**Deliverables**: Controller creates/watches/deletes Jobs, updates rollout status correctly.

---

## Phase 4: End-to-End Validation

**Goal**: Prove the system works with a real agent on minikube.

### 4.1 Example agent
- [ ] Simple Python agent that reads `AGL_TASK_INPUT`, calls LLM via `OPENAI_BASE_URL`, prints result
- [ ] Dockerfile
- [ ] Does NOT import agl-lite — proves language-agnostic contract

### 4.2 Example algorithm script
- [ ] Python script: register resources, register model server, enqueue batch, poll, retrieve events
- [ ] Demonstrates full lifecycle

### 4.3 Minikube setup
- [ ] Script or Makefile: start minikube, create namespace, create secret, deploy agl-lite, deploy controller
- [ ] Matches `docs/get_started.md`

### 4.4 End-to-end test
- [ ] Algorithm enqueues 5 rollouts → controller creates Jobs → agents run → events captured → algorithm retrieves trajectories
- [ ] Cancel test
- [ ] Retry test (agent crashes on first attempt)
- [ ] Weight update test (503 → retry → success)

**Deliverables**: Working E2E demo on minikube, validated get_started.md.

---

## Phase 5: Algorithm Integration (VERL)

**Goal**: Port VERL RL algorithm to consume agl-lite events.

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
| Gateway config | Static YAML at startup (param adjustment), no runtime changes |
| Rollout existence check | On both LLM proxy and event ingestion (in-process, ~100ns) |
| Namespace | Single namespace per controller instance |
| `timeout` | Maps to K8s Job `activeDeadlineSeconds` |
| `max_retries` | Maps to K8s Job `backoffLimit` |
| Model routing | Round-robin for MVP |
| Agent auth injection | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` env vars in Job spec, both via `secretKeyRef` to same `agl-lite-keys/AGL_KEY`. Gateway checks both `Authorization: Bearer` and `x-api-key` headers. |
