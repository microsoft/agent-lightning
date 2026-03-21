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
- [x] **Phase 2**: HTTP API — FastAPI app, auth, all store routes, gateway (config/router/proxy), CLI, streaming SSE. 
- [x] **Phase 3**: K8s controller — Python client, job builder, reconciler (create/watch/cancel/crash-recovery), CLI. 254 tests total.
- [ ] **Deferred**: kr8s adapter (`Kr8sClient`) — moved to Phase 4a.

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
