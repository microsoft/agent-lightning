# agl-lite TODO

## Phase 0: Foundation — Project Setup [ready]

Set up the Python project skeleton and define core data types.

**Tasks:**
- Create `pyproject.toml` with project metadata, dependencies (pydantic, fastapi, aiohttp, etc.)
- Create package layout: `agl_lite/` with `__init__.py`
- Define core data types in `agl_lite/types/`:
  - `RequestRecord` — single LLM request-response pair
  - `Trajectory` — ordered list of `RequestRecord` + reward
  - `Rollout` — unit of work with simplified states (`queuing`, `running`, `succeeded`, `failed`)
  - `Resource` / `ResourcesUpdate` — versioned named resource bundles
  - `Triplet` — `(prompt, response, reward)` for RL algorithms
- Define base `Store` interface in `agl_lite/store/base.py`

**Files to create:**
- `pyproject.toml`
- `agl_lite/__init__.py`
- `agl_lite/types/__init__.py`
- `agl_lite/types/core.py`
- `agl_lite/store/__init__.py`
- `agl_lite/store/base.py`

## Phase 1: Store Implementation [backlog]

Implement the in-memory store and HTTP API.

**Tasks:**
- `InMemoryStore` implementing the base `Store` interface
- Store HTTP API (FastAPI) for K8s service exposure
- Store client library for runner/algorithm pods

**Files to create:**
- `agl_lite/store/memory.py`
- `agl_lite/store/server.py`
- `agl_lite/store/client.py`

## Phase 2: Gateway [backlog]

Build the self-owned request gateway replacing LiteLLM proxy and OTEL tracer.

**Tasks:**
- Reverse proxy forwarding OpenAI-compatible requests to LLM backends
- Request-response recording as `RequestRecord` → Store
- Path-based routing: `/rollout/{rid}/attempt/{aid}/v1/chat/completions`
- Backend management: read current model endpoint from Store resources

**Files to create:**
- `agl_lite/gateway/__init__.py`
- `agl_lite/gateway/proxy.py`
- `agl_lite/gateway/recorder.py`
- `agl_lite/gateway/server.py`

## Phase 3: Runner [backlog]

Define the agent abstraction and runner loop.

**Tasks:**
- `Agent` base class with `rollout()` method
- Runner loop: dequeue → execute → update status
- Dockerfile for containerized runner

**Files to create:**
- `agl_lite/agent.py`
- `agl_lite/runner/__init__.py`
- `agl_lite/runner/base.py`
- `agl_lite/runner/loop.py`
- `docker/runner/Dockerfile`

## Phase 4: Algorithm Framework [backlog]

Define the algorithm abstraction and trajectory adapter.

**Tasks:**
- `Algorithm` base class with `run()` method
- `TrajectoryAdapter` converting trajectories → triplets
- Algorithm loop: enqueue → wait → query → learn → update resources

**Files to create:**
- `agl_lite/algorithm/__init__.py`
- `agl_lite/algorithm/base.py`
- `agl_lite/adapter/__init__.py`
- `agl_lite/adapter/base.py`
- `agl_lite/adapter/triplet.py`

## Phase 5: Kubernetes Integration [backlog]

K8s manifests and controller for rollout lifecycle.

**Tasks:**
- Store + Gateway as K8s Deployments + Services
- Runner as K8s Job template
- K8s controller/watcher syncing Job status → Store rollout status
- Minikube dev setup instructions

**Files to create:**
- `k8s/store.yaml`
- `k8s/gateway.yaml`
- `k8s/runner-job-template.yaml`
- `k8s/controller/` (TBD)
- `docs/setup/minikube.md`

## Phase 6: VERL Integration [backlog]

Port the VERL RL algorithm to agl-lite's simplified format.

**Tasks:**
- Adapt VERL to consume `Trajectory` instead of OTEL spans
- vLLM backend registration through Gateway
- End-to-end RL training on K8s

## Phase 7: Polish [backlog]

CLI, docs, CI/CD.

**Tasks:**
- `agl-lite` CLI (store, gateway, runner subcommands)
- User documentation and examples
- CI/CD pipeline

## Architecture Document [completed]

Created `docs/refactor/0_architecture.md` capturing full understanding of Agent Lightning and the agl-lite refactoring plan.
