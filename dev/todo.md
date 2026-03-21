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
| 4 | **Examples folder**: `examples/agents/python/` (source + Dockerfile, openai SDK only), `examples/math-poc/` (full PoC scenario with algorithm script + PoC-specific K8s manifests like mockai). Agents are task-specific, not infra. |
| 5 | **Docker builds**: `minikube image build` with local context + `imagePullPolicy: Never`. Infra images built from `deploy/`, agent images built from `examples/agents/python/`. |
| 6 | **E2E cleanup**: Delete + recreate namespace at test start. Single `scripts/e2e_test.sh` wraps full lifecycle. |
| 7 | **Deployments**: agl-lite serve and controller as separate K8s Deployments (matches production topology). Controller reuses agl-lite image with different CMD (no separate Dockerfile). |
| 8 | **Algorithm script**: Python, runs on host via `kubectl port-forward`. No Docker image needed — target audience is RL researchers using Python. Uses `AglLiteClient`. |
| 9 | **Deploy layout**: `deploy/` = infra only (agl-lite, controller, common K8s resources). Task-specific things (agents, mockai, algorithm scripts) live in `examples/`. |
| 10 | **Mock algorithm**: Full RL loop sim — 2 iterations with weight update (deregister → 503 window → re-register with bumped version). Verifies version tracking in events. No mockai restart needed (version is store metadata). |

### 4a.1 Kr8s adapter (`agl_lite/controller/kr8s_adapter.py`) [discuss]
- [ ] Implement `Kr8sClient` satisfying the `K8sClient` protocol in reconciler
- [ ] Methods: create_job, delete_job, get_job, list_jobs, list_pods, watch_jobs
- [ ] Wire into `agl-lite controller` CLI entrypoint

### 4a.2 Client CLI (`agl-lite client`) [discuss]
- [ ] Typer subcommand group wrapping `AglLiteClient`
- [ ] Subcommands: `rollouts list`, `rollouts get <rid>`, `events list`, `models list`, `models register`, `resources get-latest`, etc.
- [ ] Reads `--url` and `AGL_KEY` from env/options
- [ ] Useful for debugging, demos, and E2E test scripts

### 4a.3 Deploy and examples structure [discuss]

`deploy/` = infrastructure (any agl-lite setup). `examples/` = task-specific (agents, PoC scenarios).

```
deploy/
├── agl-lite/                    # HTTP service (store + gateway)
│   ├── Dockerfile               # python:3.12-slim + pip install .[controller]
│   ├── k8s.yaml                 # Deployment + Service
│   ├── gateway-config.yaml      # Example route config (passthrough for mock)
│   └── README.md
├── controller/                  # K8s reconciler (reuses agl-lite image)
│   ├── k8s.yaml                 # Deployment (image: agl-lite:dev, cmd: agl-lite controller)
│   ├── rbac.yaml                # ServiceAccount + Role + RoleBinding
│   └── README.md
├── common/                      # Shared K8s resources
│   ├── namespace.yaml           # Namespace
│   └── secret.yaml              # AGL_KEY secret template
└── README.md                    # Deploy overview + ordering guide

examples/
├── agents/
│   └── python/                  # Python agent source (reusable templates)
│       ├── qa_agent.py          # simplest: 1 LLM call
│       ├── react_agent.py       # multi-turn: tool loop
│       └── README.md
├── math-poc/                    # Full PoC: mock RL iterations on CPU
│   ├── mock_rl_loop.py          # algorithm script (runs on host)
│   ├── Dockerfile.agent         # agent image for this PoC (copies agents, adds data/tools)
│   ├── k8s-mockai.yaml          # mockai Deployment+Service (PoC-specific)
│   ├── run.sh                   # one-command: setup + run + verify
│   └── README.md                # how to run this PoC end-to-end
└── README.md
```

### 4a.4 Example agents (`examples/agents/python/`) [discuss]
- [ ] `qa_agent.py` — simplest: read `AGL_TASK_INPUT`, one LLM call via `OPENAI_BASE_URL`, print result
- [ ] `react_agent.py` — multi-turn: tool-use loop with multiple LLM calls (tests multi-event capture)
- [ ] `CRASH_ON_FIRST=1` env var support in qa_agent (for retry test)
- [ ] Does NOT import agl-lite — proves language-agnostic contract
- [ ] Pure source code — no Dockerfile here (Dockerfile is PoC-specific, lives in `examples/math-poc/`)

### 4a.5 Math PoC — mock RL loop (`examples/math-poc/`) [discuss]
- [ ] `mock_rl_loop.py` — Python script, runs on host, uses `AglLiteClient`
- [ ] `Dockerfile.agent` — agent image for this PoC (COPY agents from `../agents/python/`, add any data/tools)
- [ ] `k8s-mockai.yaml` — mockai Deployment + Service (PoC-specific, not infra)
- [ ] `run.sh` — one-command E2E: setup infra + deploy mockai + run algorithm + verify
- [ ] Full 2-iteration RL loop:
  - Iter 1: register resources + model (v1) → enqueue batch → poll → retrieve trajectories → "compute rewards"
  - Weight update: DELETE model → (simulated delay) → re-register model (v2, same endpoint)
  - Iter 2: enqueue batch → poll → retrieve trajectories → verify events have version=2
- [ ] Print summary: iterations completed, rollouts, events, versions seen
- [ ] Serves as both E2E test driver and user-facing example

### 4a.6 E2E scripts [discuss]
- [ ] `scripts/e2e_setup.sh` — nuke namespace → build images (agl-lite from `deploy/agl-lite/`, agents from `examples/agents/python/`) → apply infra manifests → wait for pods ready
- [ ] `scripts/e2e_teardown.sh` — delete namespace (optional cleanup)
- [ ] PoC-specific orchestration lives in `examples/math-poc/run.sh` (calls e2e_setup.sh, then deploys mockai, runs algorithm)

### 4a.7 End-to-end test scenarios [discuss]
- [ ] **Happy path**: 2-iteration RL loop (4a.5) — full lifecycle with weight update
- [ ] **Cancel test**: enqueue → cancel mid-run → verify cancelled status
- [ ] **Retry test**: agent with `CRASH_ON_FIRST=1` → K8s Job retries → succeeds on second attempt
- [ ] **503 test**: part of weight update in RL loop — agents hitting gateway during model deregistration window

**Deliverables**: Working E2E demo on minikube (CPU-only), client CLI, per-module deploy structure, validated get_started.md. All tests fast and deterministic.

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
