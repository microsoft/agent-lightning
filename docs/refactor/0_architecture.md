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
| 4 | **In-process** execution strategies + watchdog retry | **Kubernetes** as default runner (`minikube` for single machine) | Offload scheduling, retry, timeout to K8s controller; deployment topology is flexible |

### 2.2 What Stays (Conceptually)

| Concept | agl-lite Form |
|---------|---------------|
| Algorithm ↔ Store ↔ Runner loop | Same decoupled architecture |
| Rollout / Attempt lifecycle | Simplified states (K8s manages retry/timeout) |
| Resource versioning | Same concept (prompt templates, model endpoints) |
| Adapter pattern | Simplified — transforms request-response sequences instead of OTEL spans |
| Agent abstraction | Language-agnostic: any program that consumes Gateway endpoint via environment variables (OAI-compatible `base_url`) |
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
| `LitAgent` base class, Python agent SDK | Agents are now language-agnostic containers; no base class needed |

---

## 3. agl-lite Target Architecture

### 3.1 High-Level Overview

![agl-lite Target Architecture](../images/lite_arch.excalidraw.svg)

The architecture is organized into three logical groups. **No strong assumption is made about their co-location** — they communicate only through well-defined APIs (HTTP/gRPC), so each group can live in the same K8s cluster, in separate clusters, or even across cloud boundaries.

- **Compute Backend** (green) — Inference Servers (vLLMs) and Training Engine (Megatron/PyTorch). This is a **prerequisite managed by the user**; agl-lite does not own or deploy it. The compute backend may be in the same K8s cluster as agent runner, in a separate but network-accessible cluster, or provided by a remote fine-tuning service. Training engine pushes updated weights to inference servers.
- **AGL-Lite** (blue) — The Gateway (agl-router) sits between inference servers and agent runners, recording all request-response traffic into the Data Store. The Data Store feeds trajectory data back to the training engine. AGL-Lite can be deployed in the same K8s cluster as the Agent Runner, or co-located with the Compute Backend — in either case it only needs to **expose its API** (Store + Gateway endpoints) to the Agent Runner.
- **Agent Runner** (red) — Kubernetes-based. A K8S Controller manages agent Pods. Pods make LLM calls through the Gateway. The runner only needs network access to the AGL-Lite API (Gateway + Store endpoints); it does not need direct access to the Compute Backend.

### 3.2 Component Mapping

| agl-lite Component | Responsibility |
|--------------------|----------------|
| **Store** | Rollout queue, attempt tracking, resource versioning, trajectory storage. Exposed as an HTTP API. Can run as a K8s Service, a standalone process, or be co-located with the Compute Backend. |
| **Gateway** | Reverse proxy between agents and LLM backends. Records every request-response pair as trajectory data and writes to Store. Replaces both LLM Proxy and Tracer. Typically co-located with or near the inference servers for low latency. |
| **Runner** | K8s Job or Deployment. Each pod runs one agent container. Dequeues rollouts from Store, launches the agent process, sends LLM calls through Gateway. Only requires network access to Store and Gateway endpoints. |
| **Algorithm** | The learning loop. Enqueues rollouts, queries trajectories from Store, runs learning (RL, prompt tuning, etc.), updates resources. Typically co-located with the Compute Backend (training engine). |
| **Agent** | Any LLM-consuming program — written in any language or framework. The only contract is that it reads the Gateway endpoint from environment variables (e.g., `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`) and makes standard API calls. Packaged into a container image. No base class or SDK required. |
| **K8s Controller** | Custom controller or operator managing rollout lifecycle: retry on pod failure, timeout via `activeDeadlineSeconds`, scaling runner pods. Lives in the same K8s cluster as the runner. |

### 3.3 Simplified Data Model

#### ID Generation and Flow

| ID | Generated by | Mechanism |
|----|-------------|-----------|
| `rollout_id` | **Store** | UUID, created when Algorithm calls `enqueue_rollout()`. Passed to the K8s controller, which injects it as an env var into the agent pod. |
| `attempt_id` | **K8s** (implicitly) | Every pod K8s creates has a unique `metadata.uid` (UUID). Exposed to the container via the [Downward API](https://kubernetes.io/docs/concepts/workloads/pods/downward-api/). On retry, K8s creates a new pod with a new UID — no custom ID generation needed. |

The K8s controller composes the Gateway URL from these IDs and injects it as the agent's `OPENAI_BASE_URL`:

```yaml
# Job template (simplified)
env:
  - name: ROLLOUT_ID
    value: "R1"                      # set by K8s controller from Store
  - name: POD_UID
    valueFrom:
      fieldRef:
        fieldPath: metadata.uid      # K8s generates a unique UID per pod
  - name: OPENAI_BASE_URL
    value: "http://gateway:8080/rollout/$(ROLLOUT_ID)/attempt/$(POD_UID)/v1"
```

The agent sees a normal OpenAI-compatible base URL and has **zero awareness of agl-lite**:
```
OPENAI_BASE_URL=http://gateway:8080/rollout/R1/attempt/a1b2c3d4-e5f6-7890/v1
```

#### Attempt as a data tag, not an entity

In the original Agent Lightning, `Attempt` was a full entity with its own status lifecycle (`preparing → running → succeeded/failed/timeout/unresponsive`), health checks, and watchdog management. In agl-lite, **attempt is not an entity in the Store** — it is purely a **partitioning tag** on request records, derived from the K8s pod UID. The Store does not track attempt status; K8s owns the pod lifecycle.

This means:
- No attempt table in the Store
- No attempt status transitions
- No attempt health checks or watchdog
- Records are simply tagged with `(rollout_id, attempt_id)` for clean separation

On retry, the data stays clean because each pod has a distinct UID:
```
Pod #1 (uid=aaa): rollout=R1, attempt=aaa → [req1, req2, req3] → pod crashes
Pod #2 (uid=bbb): rollout=R1, attempt=bbb → [req1', req2', req3', req4'] → succeeds
```

Store contents — no mixing, no ambiguity:
```
(R1, aaa, seq=1), (R1, aaa, seq=2), (R1, aaa, seq=3)         ← failed run
(R1, bbb, seq=1), (R1, bbb, seq=2), (R1, bbb, seq=3), (R1, bbb, seq=4)  ← success
```

Even in rare node-partition scenarios (two pods briefly running for the same rollout), each pod writes to its own `attempt_id` partition — data never collides.

The Algorithm queries the successful attempt's records for training. Failed attempt data remains available for debugging and observability.

#### Trajectory (replaces Span tree)

```python
class RequestRecord:
    """Single LLM request-response pair captured by the Gateway."""
    request_id: str
    rollout_id: str
    attempt_id: str         # = K8s pod UID, used as data partitioning tag
    sequence: int           # auto-incrementing within the attempt (assigned by Gateway)
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
    attempt_id: str         # = K8s pod UID
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
    async def list_attempts(rollout_id) -> List[str]  # list attempt_ids for a rollout
    
    # Resource management
    async def add_resources(resources) -> ResourcesUpdate
    async def get_latest_resources() -> Optional[ResourcesUpdate]
    
    # No attempt lifecycle management (K8s owns pod lifecycle)
    # No span sequence_id allocation (Gateway auto-increments per attempt)
    # No watchdog (K8s probes handle liveness on the runner side)
    # No worker telemetry (K8s pod status on the runner side)
```

> **Deployment note**: The Store is a standalone HTTP service. It does not assume it runs inside the same K8s cluster as the runner — it only needs to be network-reachable from both the Agent Runner (for rollout queue + trajectory writes) and the Algorithm / Compute Backend (for trajectory reads + resource updates).

### 3.4 Gateway Design

The Gateway is the central innovation replacing both LiteLLM Proxy and OTEL Tracer:

```
Agent (any language) ──▶ Gateway ──▶ LLM Backend
                            │
                            ▼
                         Store (trajectory records)
```

The agent connects to the Gateway the same way it would connect to any OpenAI-compatible endpoint — via `OPENAI_BASE_URL` (or similar) environment variable. The agent does not need to know about agl-lite at all.

#### Request flow

```
1. Agent sends (using OPENAI_BASE_URL):
   POST http://gateway:8080/rollout/R1/attempt/a1b2c3d4/v1/chat/completions

2. Gateway parses path:
   → rollout_id = "R1"
   → attempt_id = "a1b2c3d4"
   → downstream path = /v1/chat/completions

3. Gateway forwards to LLM backend:
   POST http://vllm:8000/v1/chat/completions

4. Gateway captures response, writes to Store:
   RequestRecord(rollout_id="R1", attempt_id="a1b2c3d4", sequence=<auto>, ...)
```

The `rollout_id` and `attempt_id` are embedded in the URL path by the K8s Job template (see Section 3.3). The Gateway extracts them purely from the path prefix — no special headers or agent-side logic needed.

#### Key responsibilities

1. **Reverse proxy**: Forward OpenAI-compatible requests to LLM backends
2. **Path parsing**: Extract `rollout_id` and `attempt_id` from the URL prefix, strip it, forward the rest
3. **Recording**: Capture every request-response pair as a `RequestRecord`, auto-incrementing `sequence` per `(rollout_id, attempt_id)`
4. **Resource awareness**: Read current model endpoint from Store resources

The gateway is a simple Python HTTP server (e.g., `aiohttp` or `fastapi`) — no LiteLLM dependency.

### 3.5 K8s Integration (Agent Runner Side)

The K8s resources below describe the **Agent Runner** cluster. The Store and Gateway may or may not live here — they only need to be reachable via network.

| K8s Resource | agl-lite Role |
|-------------|---------------|
| **Deployment** | (Optional) Store service, Gateway service — if co-located with runner |
| **Job** | Individual rollout execution (one pod per rollout, or batched) |
| **Service** | Expose Store API and Gateway to pods (or ExternalName/Ingress if Store/Gateway are remote) |
| **ConfigMap/Secret** | Endpoint URLs (Store, Gateway), algorithm resources (prompts, model endpoints) |
| **CRD + Controller** (optional) | `RolloutBatch` custom resource for advanced lifecycle management |

#### Job template example

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: rollout-R1
spec:
  backoffLimit: 3                    # K8s retries up to 3 times
  activeDeadlineSeconds: 600         # timeout after 10 minutes
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: agent
          image: user/my-agent:latest   # any language, any framework
          env:
            - name: ROLLOUT_ID
              value: "R1"               # set by controller from Store
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid   # unique per pod, changes on retry
            - name: OPENAI_BASE_URL
              value: "http://gateway:8080/rollout/$(ROLLOUT_ID)/attempt/$(POD_UID)/v1"
```

On each retry, K8s creates a new pod with a new `metadata.uid`, so `OPENAI_BASE_URL` automatically points to a fresh attempt partition in the Gateway.

#### Retry control

- `Job.spec.backoffLimit` for retry count
- `Job.spec.activeDeadlineSeconds` for timeout
- K8s controller watches Job status and updates Store rollout status accordingly

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
