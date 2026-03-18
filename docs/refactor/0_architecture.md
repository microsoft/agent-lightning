# agl-lite Architecture: From Agent Lightning to Minimal Workable Version

This document captures a comprehensive understanding of the original Agent Lightning architecture and maps out how each component is simplified, replaced, or removed in agl-lite.

---

## 1. Original Agent Lightning Architecture

### 1.1 Core Loop

Agent Lightning is built on three main components in a coordinated loop:

```
Algorithm ──enqueue_rollout──▶ LightningStore ──dequeue_rollout──▶ Runner
    ▲                              │                                  │
    │                         (spans, resources,                      │
    │                          attempts, workers)                     │
    │                              │                                  │
    └──query_spans + learn─────────┘◀──────add_span + update_attempt──┘
```

- **Algorithm**: The "brain" — decides tasks, learns from results, updates resources (model weights, prompts).
- **Runner**: The "worker" — executes rollouts, runs the agent, records spans.
- **LightningStore**: Central database + message queue — single source of truth.

### 1.2 Data Types

| Type | Description |
|------|-------------|
| **Rollout** | Unit of work. Lifecycle: `queuing → preparing → running → succeeded/failed/cancelled/requeuing` |
| **Attempt** | Single execution of a rollout. Supports retries: `preparing → running → succeeded/failed/timeout/unresponsive` |
| **Span** | Structured trace event (LLM call, tool invocation, reward). Ordered by monotonic `sequence_id` per attempt. Based on OpenTelemetry. |
| **Resource** | Versioned named bundles (prompt templates, model checkpoints, proxy URLs) |
| **Triplet** | `(prompt, response, reward)` — fundamental RL learning unit extracted from spans |
| **Worker** | Runner instance metadata (heartbeat, status, current assignment) |
| **Dataset** | Collection of incomplete rollouts (tasks) for the agent to process |

### 1.3 Supporting Components

| Component | Role | Key Dependencies |
|-----------|------|------------------|
| **Tracer** | Instruments agent code, captures OpenTelemetry spans, ships to Store | `opentelemetry-sdk`, `agentops` |
| **Adapter** | Transforms raw spans → algorithm-consumable formats (e.g., `TracerTraceToTriplet` → RL triplets) | OpenTelemetry span format |
| **LLM Proxy** | LiteLLM-based reverse proxy between agent and LLM backends; instruments server-side, manages model swaps, URL routing | `litellm`, `fastapi`, OpenTelemetry |
| **Trainer** | High-level orchestrator wiring all components | All of the above |
| **Hook** | User callbacks at lifecycle points (`on_rollout_start/end`, `on_trace_start/end`) | — |
| **ExecutionStrategy** | Controls how algorithm/runner bundles are placed (shared-memory vs. client-server) | `multiprocessing`, `asyncio` |
| **LitAgent** | Base class for user-defined agents. `rollout(task, resources, rollout) → RolloutRawResult` | — |

### 1.4 Store Architecture

Three layers:

1. **Collections Layer** — Low-level CRUD primitives (`Collection`, `Queue`, `KeyValue`). Backends: InMemory, MongoDB.
2. **Store Layer** — `CollectionBasedLightningStore` builds on collections with business logic (status transitions, watchdog health checks, retry policies).
3. **Wrappers** — `LightningStoreThreaded` (mutex thread safety), `LightningStoreServer/Client` (HTTP multi-process).

Key store responsibilities:
- Task queue (`enqueue_rollout` / `dequeue_rollout`)
- Rollout + attempt lifecycle management (status transitions, retries via watchdog)
- Span ingest + ordering (monotonic `sequence_id`)
- Resource versioning
- Worker telemetry

### 1.5 LLM Proxy Internals

The proxy is a LiteLLM-based FastAPI server with:
- **RolloutAttemptMiddleware**: Rewrites `/rollout/{rid}/attempt/{aid}/v1/chat/completions` → `/v1/chat/completions`, injects `x-rollout-id`, `x-attempt-id`, `x-sequence-id` headers.
- **StreamConversionMiddleware**: Converts streaming → non-streaming for better OTEL capture.
- **LightningSpanExporter**: Buffers OTEL spans, flushes subtrees to the store.
- **LightningOpenTelemetry**: LiteLLM callback that wires OTEL export.

### 1.6 Execution Strategies

- **SharedMemoryExecutionStrategy**: Algorithm + runners as threads in one process. Good for debugging.
- **ClientServerExecutionStrategy**: Algorithm process hosts `LightningStoreServer` (HTTP API). Runners connect via `LightningStoreClient`. Supports multi-process scaling.

### 1.7 VERL Integration (RL Example)

The VERL algorithm:
1. Launches a vLLM chat completion endpoint
2. Registers it in the LLM Proxy → Store as resource
3. Enqueues rollouts from dataset
4. Runners dequeue, execute agents against the proxy endpoint
5. Proxy + tracer capture spans → Store
6. Algorithm queries spans, adapter converts to triplets → FSDP training loop
7. Model weights updated → repeat

---

## 2. agl-lite Simplification Plan

### 2.1 What Changes

| # | Original | agl-lite Replacement | Rationale |
|---|----------|---------------------|-----------|
| 1 | **LiteLLM** proxy for LLM routing | **Self-owned request gateway** | Remove heavy dependency; simpler proxy that records traffic |
| 2 | **OpenTelemetry** stack (spans, tracers, exporters, instrumentation) | **Gateway records request-response pairs** during transfer | Eliminate OTEL complexity; the gateway *is* the instrumentation |
| 3 | **Span-based** trajectory format | **Sequence of requests (with responses)** | Much simpler data model; no span trees, no sequence_id allocation |
| 4 | **In-process** execution strategies + watchdog retry | **Kubernetes** as default runner (`minikube` for single machine) | Offload scheduling, retry, timeout to K8s controller |

### 2.2 What Stays (Conceptually)

| Concept | agl-lite Form |
|---------|---------------|
| Algorithm ↔ Store ↔ Runner loop | Same decoupled architecture |
| Rollout / Attempt lifecycle | Simplified states (K8s manages retry/timeout) |
| Resource versioning | Same concept (prompt templates, model endpoints) |
| Adapter pattern | Simplified — transforms request-response sequences instead of OTEL spans |
| Agent abstraction | Same `rollout()` interface |
| Store API | Simplified subset (no span sequence_id, no watchdog, no OTEL conversion) |

### 2.3 What Gets Removed

| Component | Reason |
|-----------|--------|
| `agentlightning.tracer.*` (AgentOps, OTEL, Weave tracers) | Gateway replaces all tracing |
| `agentlightning.instrumentation.*` (LiteLLM, vLLM, AgentOps hooks) | No longer needed |
| `agentlightning.llm_proxy` (LiteLLM-based proxy) | Replaced by self-owned gateway |
| `agentlightning.semconv` (OTEL semantic conventions) | No OTEL |
| `agentlightning.utils.otel`, `agentlightning.utils.otlp` | No OTEL |
| `LightningSpanExporter`, `LightningOpenTelemetry` | No OTEL |
| `SharedMemoryExecutionStrategy`, `ClientServerExecutionStrategy` | K8s replaces execution strategies |
| `LightningStoreServer` / `LightningStoreClient` | Store communication redesigned for K8s |
| `LightningStoreThreaded` | K8s pods are isolated; no shared-memory threading model |
| Watchdog (timeout/unresponsive detection in Store) | K8s liveness/readiness probes + controller |
| Span `sequence_id` allocation | No OTEL spans to order |
| `RolloutAttemptMiddleware` URL rewriting | Gateway handles routing natively |
| Legacy/compat code (`TrainerLegacy`, `RolloutLegacy`, `fit_v0`) | Clean slate |

---

## 3. agl-lite Target Architecture

### 3.1 High-Level Overview

![agl-lite Target Architecture](../images/lite_arch.excalidraw.svg)

The architecture is organized into three columns:

- **Compute Backend** (green) — Inference Servers (vLLMs) and Training Engine (Megatron/PyTorch). Training engine pushes updated weights to inference servers.
- **AGL-Lite** (blue) — The Gateway (agl-router) sits between inference servers and agent runners, recording all request-response traffic into the Data Store. The Data Store feeds trajectory data back to the training engine.
- **Agent Runner** (red) — Kubernetes-based. A K8S Controller manages agent Pods. Pods make LLM calls through the Gateway.

### 3.2 Component Mapping

| agl-lite Component | Responsibility |
|--------------------|----------------|
| **Store** | Rollout queue, attempt tracking, resource versioning, trajectory storage. Exposed as a K8s Service (HTTP API). |
| **Gateway** | Reverse proxy between agents and LLM backends. Records every request-response pair as trajectory data and writes to Store. Replaces both LLM Proxy and Tracer. |
| **Runner** | K8s Job or Deployment. Each pod runs one agent. Dequeues rollouts from Store, executes agent, sends LLM calls through Gateway. |
| **Algorithm** | K8s Pod. Enqueues rollouts, queries trajectories from Store, runs learning (RL, prompt tuning, etc.), updates resources. |
| **Agent** | User-defined Python class with `rollout()` method. Packaged into a container image. |
| **K8s Controller** | Custom controller or operator managing rollout lifecycle: retry on pod failure, timeout via `activeDeadlineSeconds`, scaling runner pods. |

### 3.3 Simplified Data Model

#### Trajectory (replaces Span tree)

```python
class RequestRecord:
    """Single LLM request-response pair captured by the Gateway."""
    request_id: str
    rollout_id: str
    attempt_id: str
    sequence: int           # auto-incrementing within the attempt
    timestamp: float
    
    # Request
    model: str
    messages: List[Dict]    # OpenAI chat format
    parameters: Dict        # temperature, max_tokens, etc.
    
    # Response
    response: Dict          # full OpenAI-format response
    usage: Dict             # token counts
    latency_ms: float
    
    metadata: Dict          # extra headers, annotations

class Trajectory:
    """Complete trajectory for one rollout attempt."""
    rollout_id: str
    attempt_id: str
    records: List[RequestRecord]  # ordered by sequence
    reward: Optional[float]
```

#### Simplified Rollout States

```
              K8s creates pod
queuing ─────────────────────▶ running ──────▶ succeeded
                                  │               
                                  ├──────▶ failed ──▶ (K8s retry or) terminal_failed
                                  │               
                                  └──────▶ timeout (K8s activeDeadlineSeconds)
                                                └──▶ (K8s retry or) terminal_failed
```

- **No `preparing` state** — pod creation is atomic from the rollout's perspective
- **No `unresponsive` state** — K8s liveness probes handle this
- **No `requeuing` state** — K8s Job `backoffLimit` handles retries
- **No `cancelled`** — delete the K8s Job

#### Simplified Store API

```python
class Store:
    # Rollout management
    async def enqueue_rollout(input, mode, resources_id, config) -> Rollout
    async def dequeue_rollout(worker_id) -> Optional[Rollout]
    async def update_rollout(rollout_id, status, ...) -> Rollout
    async def query_rollouts(status_in, ...) -> List[Rollout]
    async def wait_for_rollouts(rollout_ids, timeout) -> List[Rollout]
    
    # Trajectory storage (replaces span APIs)
    async def add_request_record(record: RequestRecord) -> RequestRecord
    async def query_trajectory(rollout_id, attempt_id) -> Trajectory
    
    # Resource management
    async def add_resources(resources) -> ResourcesUpdate
    async def get_latest_resources() -> Optional[ResourcesUpdate]
    
    # No attempt management (K8s handles this)
    # No span sequence_id allocation (gateway auto-increments)
    # No watchdog (K8s probes)
    # No worker telemetry (K8s pod status)
```

### 3.4 Gateway Design

The Gateway is the central innovation replacing both LiteLLM Proxy and OTEL Tracer:

```
Agent ──▶ Gateway ──▶ LLM Backend
              │
              ▼
           Store (trajectory records)
```

Key responsibilities:
1. **Reverse proxy**: Forward OpenAI-compatible requests to LLM backends
2. **Recording**: Capture every request-response pair as a `RequestRecord`
3. **Routing**: Map rollout/attempt context to the correct backend (via URL path or headers)
4. **Resource awareness**: Read current model endpoint from Store resources

Routing options (choose one):
- **Path-based**: `/rollout/{rid}/attempt/{aid}/v1/chat/completions` (similar to original, but gateway handles natively)
- **Header-based**: Standard `/v1/chat/completions` with `X-Rollout-Id`, `X-Attempt-Id` headers

The gateway is a simple Python HTTP server (e.g., `aiohttp` or `fastapi`) — no LiteLLM dependency.

### 3.5 K8s Integration

| K8s Resource | agl-lite Role |
|-------------|---------------|
| **Deployment** | Store service, Gateway service |
| **Job** | Individual rollout execution (one pod per rollout, or batched) |
| **Service** | Expose Store API and Gateway to pods |
| **ConfigMap/Secret** | Algorithm resources (prompts, model endpoints) |
| **CRD + Controller** (optional) | `RolloutBatch` custom resource for advanced lifecycle management |

Retry control:
- `Job.spec.backoffLimit` for retry count
- `Job.spec.activeDeadlineSeconds` for timeout
- K8s controller watches Job status and updates Store accordingly

### 3.6 Adapter Simplification

```python
class TrajectoryAdapter:
    """Convert trajectory records into algorithm-consumable format."""
    
    def adapt(self, trajectory: Trajectory) -> List[Triplet]:
        """Extract (prompt, response, reward) triplets from a trajectory."""
        triplets = []
        for record in trajectory.records:
            triplets.append(Triplet(
                prompt=record.messages,
                response=record.response,
                reward=trajectory.reward,  # or per-step reward if available
                metadata=record.metadata,
            ))
        return triplets
```

No OTEL span parsing, no parent-child tree reconstruction, no attribute unflattening.

---

## 4. Refactoring Phases (Coarse Plan)

### Phase 0: Foundation
- [ ] Set up Python project structure (`pyproject.toml`, package layout)
- [ ] Define core data types (`Rollout`, `RequestRecord`, `Trajectory`, `Resource`, `Triplet`)
- [ ] Implement base `Store` interface

### Phase 1: Store
- [ ] Implement `InMemoryStore` for development/testing
- [ ] Implement Store HTTP API (FastAPI service)
- [ ] Store client library for pods to call

### Phase 2: Gateway
- [ ] Implement request gateway (reverse proxy + recording)
- [ ] Path-based or header-based rollout/attempt routing
- [ ] Write `RequestRecord` to Store on each request-response

### Phase 3: Runner
- [ ] Define `Agent` base class (equivalent to `LitAgent`)
- [ ] Implement runner loop: dequeue rollout → execute agent → update status
- [ ] Containerize runner (Dockerfile)

### Phase 4: Algorithm Framework
- [ ] Define `Algorithm` base class
- [ ] Implement `TrajectoryAdapter` (trajectory → triplets)
- [ ] Wire algorithm loop: enqueue rollouts → wait → query trajectories → learn

### Phase 5: Kubernetes Integration
- [ ] K8s manifests for Store, Gateway as Services
- [ ] Runner as K8s Job template
- [ ] K8s controller or simple watcher for rollout lifecycle
- [ ] Minikube dev setup instructions

### Phase 6: VERL Integration
- [ ] Port the VERL algorithm to work with simplified trajectory format
- [ ] vLLM backend registration via Gateway
- [ ] End-to-end RL training loop on K8s

### Phase 7: Polish
- [ ] CLI tooling (`agl-lite` command)
- [ ] Documentation and examples
- [ ] CI/CD pipeline
