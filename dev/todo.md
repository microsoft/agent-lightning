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
| 5 | **Docker builds**: Unified `scripts/build_images.sh` (bash). Uses `minikube image build` + `imagePullPolicy: Never`. Infra from `deploy/`, PoC images selective via args. `.dockerignore` for fast builds. agl-lite Dockerfile builds from repo root context. |
| 6 | **E2E cleanup**: Delete + recreate namespace at test start. Single `scripts/e2e_test.sh` wraps full lifecycle. |
| 7 | **Deployments**: agl-lite serve and controller as separate K8s Deployments (matches production topology). Controller reuses agl-lite image with different CMD (no separate Dockerfile). |
| 8 | **Algorithm script**: Python, runs on host via `kubectl port-forward`. No Docker image needed — target audience is RL researchers using Python. Uses `AglLiteClient`. |
| 9 | **Deploy layout**: `deploy/` = infra only (agl-lite, controller). No `deploy/common/` — setup script creates K8s resources from `.env`. Task-specific things (agents, mockai, algorithm scripts) live in `examples/`. |
| 10 | **Mock algorithm**: Full RL loop sim — 2 iterations with weight update (deregister → 503 window → re-register with bumped version). Verifies version tracking in events. No mockai restart needed (version is store metadata). |
| 11 | **Dataset**: GSM8K 30-problem subset (`examples/math-poc/data/gsm8k_sample.jsonl`). Algorithm embeds correct or randomly wrong answers in the prompt. Mockai (echo mode) echoes the prompt back. Reward function extracts embedded answer and compares to ground truth — real parsing, mix of reward 1.0/0.0, fully verifiable. No mockai modification needed. |
| 12 | **Agent image**: Contains all agent scripts at `/app/`. Rollout `config.command` selects which to run (e.g. `["python", "/app/qa_agent.py"]`). Image = environment, command = task. |
| 13 | **Agent Dockerfile context**: `examples/` as build context, `Dockerfile.agent` COPYs from `agents/python/` and `math-poc/data/`. |
| 14 | **Configuration**: Split into `deploy/.env` (secrets + bootstrap: `AGL_KEY`, `AGL_K8S_NAMESPACE`) and `deploy/config.yaml` (structured non-secret config: serve host/port, agl_lite_url, controller settings). Setup script reads `.env` for namespace/secret creation, loads `config.yaml` into K8s ConfigMap (`--from-file`). Pods mount ConfigMap as `/etc/agl-lite/config.yaml`. CLI supports `--config` flag. Precedence: config file → env vars → CLI args. |
| 15 | **Dockerfile**: Use `uv` for fast installs (`curl -LsSf https://astral.sh/uv/install.sh`). `.dockerignore` excludes `.venv/`, `.git/`, `tests/`, `docs/`, `dev/`, `examples/`, `node_modules/`, `tmp/`, `.local/`, `__pycache__/`. |
| 16 | **Namespace**: Manifests omit `metadata.namespace`. Setup script applies with `-n $AGL_K8S_NAMESPACE` from `.env`. Works for any namespace. |
| 17 | **Phase 4a topology**: All-in-K8s (agl-lite serve, controller, mockai as Deployments). Only algorithm script runs on host (via port-forward). Avoids host↔K8s bridging complexity. |
| 18 | **Deploy scripts**: Python for orchestration (`scripts/deploy.py` — namespace, secrets, configmap, manifests, wait, health check). Bash for image builds (`scripts/build_images.sh` — thin wrapper around `minikube image build`). PoC orchestration in `examples/math-poc/run.py` (Python). |

### 4a.1 Kr8s adapter (`agl_lite/controller/kr8s_adapter.py`) [completed]
- [x] Implement `Kr8sClient` satisfying the `K8sClient` protocol in reconciler
- [x] Methods: create_job, delete_job, get_job, list_jobs, list_pods, watch_jobs
- [x] Wire into `agl-lite controller` CLI entrypoint
- [x] 9 integration tests against real minikube (create, get, delete, list, watch, complete, fail)

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
│   ├── Dockerfile               # python:3.12-slim + uv + pip install .[controller]
│   ├── k8s.yaml                 # Deployment + Service (mounts ConfigMap, env from Secret)
│   └── README.md
├── controller/                  # K8s reconciler (reuses agl-lite image)
│   ├── k8s.yaml                 # Deployment (mounts ConfigMap, env from Secret)
│   ├── rbac.yaml                # ServiceAccount + Role + RoleBinding
│   └── README.md
├── .env.example                 # secrets + bootstrap: AGL_KEY, AGL_K8S_NAMESPACE
├── config.example.yaml          # structured non-secret config: serve, controller, agl_lite_url
└── README.md                    # deploy overview + ordering guide
```

Setup script creates K8s resources from `.env` + `config.yaml`:
- `source deploy/.env`
- `kubectl create namespace $AGL_K8S_NAMESPACE`
- `kubectl -n $AGL_K8S_NAMESPACE create secret generic agl-lite-keys --from-literal=AGL_KEY="$AGL_KEY"` (never on disk)
- `kubectl -n $AGL_K8S_NAMESPACE create configmap agl-lite-config --from-file=config.yaml=deploy/config.yaml`
- `kubectl apply -n $AGL_K8S_NAMESPACE -f deploy/...` (manifests omit namespace, mount ConfigMap as volume)

```
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
- [ ] `scripts/build_images.sh` — bash, thin wrapper around `minikube image build`. Always builds agl-lite; selectively builds PoC images via args (e.g., `--math-poc`).
- [ ] `scripts/deploy.py` — Python orchestration: read `.env` + `config.yaml`, create namespace/secret/configmap, apply manifests, wait for pods ready, health check agl-lite.
- [ ] `scripts/teardown.sh` — bash, `kubectl delete namespace` (simple).
- [ ] `.dockerignore` — exclude `.venv/`, `.git/`, `tests/`, `docs/`, `dev/`, `examples/`, `node_modules/`, `tmp/`, `.local/`, `__pycache__/`.
- [ ] `examples/math-poc/run.py` — Python: calls deploy.py for infra, deploys mockai, starts port-forward, runs mock_rl_loop.py, verifies results, cleanup.

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
