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
| 3 | **Span-based** trajectory format | **Sequence of events** (`model_request`, `reward`, + open user-defined types) | Much simpler data model; no span trees, no sequence_id allocation. Only two reserved types; everything else is opaque pass-through. |
| 4 | **In-process** execution strategies + watchdog retry | **Kubernetes** as default runner (`minikube` for single machine) | Offload scheduling, retry, timeout to K8s controller; deployment topology is flexible |

### 2.2 What Stays (Conceptually)

| Concept | agl-lite Form |
|---------|---------------|
| Algorithm ↔ Store ↔ Runner loop | Same decoupled architecture |
| Rollout / Attempt lifecycle | Simplified states (K8s manages retry/timeout) |
| Resource versioning | Same concept (prompt templates, model endpoints) |
| Adapter pattern | Simplified — filters events by type (`model_request`, `reward`) instead of parsing OTEL spans |
| Agent abstraction | Language-agnostic: any program that consumes Gateway endpoint via environment variables (OAI-compatible `base_url`) |
| Store API | Simplified subset (event-based, no span sequence_id, no watchdog, no OTEL conversion) |

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
- **AGL-Lite** (blue) — A single service combining the Gateway (agl-router) and the Data Store. The Gateway sits between inference servers and agent runners, recording all request-response traffic into the Store. The Store feeds trajectory data back to the training engine. One endpoint, one deployment. AGL-Lite can be deployed in the same K8s cluster as the Agent Runner, or co-located with the Compute Backend — in either case it only needs to **expose its API** to the Agent Runner.
- **Agent Runner** (red) — Kubernetes-based. A K8S Controller manages agent Pods. Pods make LLM calls through the agl-lite Gateway. The runner only needs network access to the agl-lite Service endpoint; it does not need direct access to the Compute Backend.

### 3.2 Component Mapping

| agl-lite Component | Responsibility |
|--------------------|----------------|
| **agl-lite Service** | Single HTTP service combining Gateway (LLM reverse proxy, event auto-capture) and Store (rollout queue, event storage, resource versioning). One deployment, one endpoint. Can run as a K8s Service, a standalone process, or be co-located with the Compute Backend. |
| **Runner** | K8s Job or Deployment. Each pod runs one agent container. Only requires network access to the agl-lite Service endpoint. |
| **Algorithm** | The learning loop. Enqueues rollouts, queries trajectories, runs learning (RL, prompt tuning, etc.), updates resources. Typically co-located with the Compute Backend (training engine). Talks to the agl-lite Service API. |
| **Agent** | Any LLM-consuming program — written in any language or framework. The only contract is that it reads the Gateway endpoint from environment variables (e.g., `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`) and makes standard API calls. Packaged into a container image. No base class or SDK required. |
| **K8s Controller** | Custom controller or operator managing rollout lifecycle: retry on pod failure, timeout via `activeDeadlineSeconds`, scaling runner pods. Lives in the same K8s cluster as the runner. Talks to the agl-lite Service API. |

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
  - name: AGL_LITE_URL
    value: "http://agl-lite:8080"    # single service endpoint
  - name: OPENAI_BASE_URL
    value: "$(AGL_LITE_URL)/rollout/$(ROLLOUT_ID)/attempt/$(POD_UID)/v1"
```

The agent sees a normal OpenAI-compatible base URL and has **zero awareness of agl-lite**:
```
OPENAI_BASE_URL=http://agl-lite:8080/rollout/R1/attempt/a1b2c3d4-e5f6-7890/v1
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

#### Event-based Trajectory

agl-lite is a **data pipe**, not a schema enforcer. The trajectory is a sequence of **events**. Only two event types have well-known structure — `model_request` (which the Gateway must create) and `reward` (which the Algorithm must consume for training). Everything else is opaque pass-through: a dict with a `type` field, stored and delivered as-is. Users define their own event types and consume them in their own algorithms.

```python
class Event:
    """Single event in a trajectory. The universal unit of data in agl-lite."""
    event_id: str
    event_type: str             # "model_request", "reward", or any user-defined string
    rollout_id: str
    attempt_id: str             # = K8s pod UID
    sequence: int               # global ordering within the attempt
    timestamp: float
    data: Dict                  # event-type-specific payload (see below)
```

**Reserved event types** (agl-lite understands these):

```python
# event_type = "model_request"
# Created automatically by the Gateway on every LLM call.
{
    "model": "gpt-4",
    "request": {
        "messages": [...],          # OpenAI chat format
        "temperature": 0.7,
        # ... other parameters
    },
    "response": {                   # full OpenAI-format response
        "choices": [...],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, ...},
    },
    "latency_ms": 1234.5,
}

# event_type = "reward"
# Reported by the environment, evaluator, or runner.
{
    "value": 0.85,                  # scalar reward (required)
    "message": "all tests passed",  # optional human-readable explanation
}
```

**User-defined event types** (agl-lite stores and delivers, but does not interpret):

```python
# event_type = "tool_result" (user-defined example)
{"tool_name": "execute_code", "output": "hello\n", "exit_code": 0}

# event_type = "observation" (user-defined example)
{"content": "Task: Write a function that...", "source": "environment"}

# event_type = "my_custom_metric" (user-defined example)
{"score": 42, "details": {...}}
```

The Store stores all events identically — it does not validate `data` payloads beyond the common fields (`event_id`, `event_type`, `rollout_id`, `attempt_id`, `sequence`, `timestamp`, `data`).

#### Trajectory

```python
class Trajectory:
    """Complete trajectory for one rollout attempt — a sequence of events."""
    rollout_id: str
    attempt_id: str             # = K8s pod UID
    events: List[Event]         # ordered by sequence, mixed event types
```

All event types share a single monotonically increasing `sequence` counter per `(rollout_id, attempt_id)`, preserving true temporal ordering:

```
seq=1  model_request   (agent calls LLM)
seq=2  tool_result     (runner reports tool output)     ← user-defined type
seq=3  model_request   (agent sends tool result to LLM)
seq=4  action          (agent submits answer)           ← user-defined type
seq=5  reward          (environment scores: 0.85)
```

#### How events are produced

| Source | Event types | Mechanism |
|--------|------------|-----------|
| **Gateway** | `model_request` | Auto-captured on every proxied LLM call. Agent is unaware. |
| **Runner / Environment** | `reward`, plus any user-defined types | Explicit HTTP POST to Gateway event endpoint (see Section 3.4). These are agl-lite-aware components. |
| **Agent** (optional) | Any user-defined types | If the agent *chooses* to report events, it can POST to an optional event URL (`AGL_EVENT_URL` env var). But this is never required. |

#### Rollout Record

```python
class RolloutStatus(str, Enum):
    QUEUING = "queuing"                 # in Store queue, no Job yet
    RUNNING = "running"                 # Job exists, execution in progress (including between retries)
    SUCCEEDED = "succeeded"             # terminal — one attempt completed successfully
    TERMINAL_FAILED = "terminal_failed" # terminal — all retries exhausted or deadline exceeded
    CANCELLED = "cancelled"             # terminal — user requested cancellation

class Rollout:
    rollout_id: str
    status: RolloutStatus
    cancel_requested: bool              # flag set by user, read by controller
    
    input: Dict                         # task description
    resources_id: Optional[str]         # resource version to use
    config: Dict                        # backoff limit, timeout, etc.
    
    # Set by controller during lifecycle
    job_name: Optional[str]             # K8s Job name (set on Job creation)
    succeeded_attempt_id: Optional[str] # pod UID of successful attempt (set on success)
    error_message: Optional[str]        # error info (set on terminal_failed)
    
    # Concurrency control
    version: int                        # optimistic locking — incremented on every update
    
    created_at: float
    updated_at: float
```

`cancel_requested` is a separate flag rather than a status because:
- User expresses **intent** (set flag) without knowing the current execution state
- Controller **executes** (deletes Job, updates status) in its own reconciliation loop
- No invalid status transition — the flag can be set whenever status is non-terminal

#### Rollout State Machine

```
queuing ──[controller creates Job]──────────→ running
   │                                            │  │
   │                                            │  ├──[Job Complete]──→ succeeded
   │                                            │  │                    (final)
   ├──[Job creation failed]──→ terminal_failed  │  │
   │                           (final)          │  ├──[Job Failed]───→ terminal_failed
   │                                            │  │                    (final)
   │                                            │  │
   ├──[cancel + no Job]─────→ cancelled         │  └──[cancel]───────→ cancelled
   │                          (final)           │                       (final)
   │                                            │
   └────────────────────────────────────────────┘
          (cancel_requested can be set while queuing or running)
```

**Valid transitions (Store-enforced):**

| From | To | Trigger |
|------|----|---------|
| `queuing` | `running` | Controller: Job created successfully |
| `queuing` | `terminal_failed` | Controller: Job creation failed (quota, invalid image, etc.) |
| `queuing` | `cancelled` | Controller: `cancel_requested` is true, no Job exists |
| `running` | `succeeded` | Controller: Job has `Complete` condition |
| `running` | `terminal_failed` | Controller: Job has `Failed` condition (BackoffLimitExceeded, DeadlineExceeded) |
| `running` | `cancelled` | Controller: `cancel_requested` is true, Job deleted |

**Store-enforced invariants:**
- `succeeded`, `terminal_failed`, `cancelled` are **final** — no transitions out, Store rejects any attempt
- `running → queuing` is **rejected** — no going backwards
- `cancel_requested` can only be set to `true` when status is `queuing` or `running`; setting it on a terminal rollout returns an error

#### Store API

```python
class Store:
    # Rollout management
    async def enqueue_rollout(input, resources_id, config) -> Rollout
    async def update_rollout(rollout_id, status, expected_version,
                             job_name=None, succeeded_attempt_id=None,
                             error_message=None) -> Rollout
        # Enforces: valid transition + optimistic locking (version check)
        # Raises: ConflictError (version mismatch), InvalidTransitionError
    async def cancel_rollout(rollout_id) -> Rollout
        # Sets cancel_requested=True. Rejects if already terminal.
    async def query_rollouts(status_in=None, cancel_requested=None) -> List[Rollout]
    async def wait_for_rollouts(rollout_ids, timeout) -> List[Rollout]
        # Long-polls until all specified rollouts reach a terminal state,
        # or timeout expires. Returns current state of all requested rollouts.
    
    # Event storage
    async def add_event(event: Event) -> Event
    async def add_events(events: List[Event]) -> List[Event]
    async def query_events(rollout_id, attempt_id=None,
                           event_type=None, limit=None, offset=None) -> List[Event]
    async def list_attempts(rollout_id) -> List[str]
    
    # Resource management
    async def add_resources(resources) -> ResourcesUpdate
    async def get_latest_resources() -> Optional[ResourcesUpdate]
```

> **Deployment note**: The agl-lite Service is a single HTTP server. It does not assume it runs inside the same K8s cluster as the runner — it only needs to be network-reachable from the Agent Runner, the K8s Controller, and the Algorithm.

### 3.4 Unified API Spec

The Gateway (LLM proxy) and Store (data management) are combined into a **single HTTP service**. All paths are served by one endpoint. This eliminates the network hop between Gateway and Store on the hot path (every LLM request), and simplifies deployment to one service.

#### Path layout

| Path pattern | Function | Consumer |
|---|---|---|
| `/rollout/{rid}/attempt/{aid}/v1/...` | **LLM reverse proxy** — forwards to LLM backend, auto-captures `model_request` events | Agent pods |
| `/rollout/{rid}/attempt/{aid}/events` | **Event ingestion** — accepts reward and user-defined events | Agent pods, runner, environment |
| `/api/rollouts` | **Rollout management** — enqueue, query, cancel, wait | Algorithm, K8s controller |
| `/api/rollouts/{rid}` | **Single rollout** — get, update | K8s controller |
| `/api/rollouts/{rid}/cancel` | **Cancel rollout** — set cancel_requested flag | Algorithm, user |
| `/api/rollouts/wait` | **Wait for completion** — long-poll until terminal | Algorithm |
| `/api/events` | **Event query** — query events by rollout/attempt/type | Algorithm |
| `/api/attempts/{rid}` | **List attempts** — list attempt_ids for a rollout | Algorithm |
| `/api/resources` | **Resource management** — add, get latest | Algorithm |

#### LLM proxy paths (agent-facing, transparent)

**`POST /rollout/{rollout_id}/attempt/{attempt_id}/v1/chat/completions`**

The agent calls this as a normal OpenAI endpoint (via `OPENAI_BASE_URL`). The service:
1. Parses `rollout_id` and `attempt_id` from the path prefix
2. Strips the prefix, forwards `POST /v1/chat/completions` to the LLM backend
3. Captures request + response as a `model_request` event (auto-assigned `sequence`, `timestamp`)
4. Returns the LLM response to the agent

Any path under `/rollout/{rid}/attempt/{aid}/v1/...` is proxied. The agent is unaware of agl-lite.

**`POST /rollout/{rollout_id}/attempt/{attempt_id}/events`**

Accepts explicit events (reward, user-defined types). Body:
```json
{"event_type": "reward", "data": {"value": 0.85, "message": "all tests passed"}}
```
The service assigns `event_id`, `sequence`, `timestamp` and stores the event. Used by runners, environments, evaluators, and optionally by agents (via `AGL_EVENT_URL`).

#### Store paths (management API)

**Rollout management:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/rollouts` | Enqueue a new rollout. Body: `{input, resources_id?, config?}`. Returns `Rollout` with status `queuing`. |
| `GET` | `/api/rollouts` | Query rollouts. Params: `status_in`, `cancel_requested`, `limit`, `offset`. Returns `List[Rollout]`. |
| `GET` | `/api/rollouts/{rollout_id}` | Get a single rollout by ID. Returns `Rollout`. |
| `PATCH` | `/api/rollouts/{rollout_id}` | Update rollout status. Body: `{status, expected_version, job_name?, succeeded_attempt_id?, error_message?}`. Enforces valid transitions + optimistic locking. Used by K8s controller. |
| `POST` | `/api/rollouts/{rollout_id}/cancel` | Set `cancel_requested=true`. Rejects if already terminal. Used by Algorithm or user. |
| `POST` | `/api/rollouts/wait` | Long-poll until rollouts complete. Body: `{rollout_ids, timeout?}`. Returns when all reach terminal state or timeout. |

**Event / trajectory access:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/events` | Query events. Params: `rollout_id`, `attempt_id?`, `event_type?`, `limit?`, `offset?`. Default order: by sequence. A full trajectory is just `GET /api/events?rollout_id={rid}&attempt_id={aid}`. |
| `GET` | `/api/attempts/{rollout_id}` | List all attempt_ids for a rollout. |

**Resource management:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/resources` | Add a new resource snapshot. Body: `{resources}`. Returns `ResourcesUpdate` with generated ID. |
| `GET` | `/api/resources/latest` | Get the latest resource snapshot. |
| `GET` | `/api/resources/{resources_id}` | Get a specific resource snapshot by ID. |

### 3.5 K8s Controller

The K8s controller bridges the Store and K8s. It watches K8s Job status, creates Jobs for queued rollouts, handles cancellation, and syncs terminal status back to the Store. It is the **only component that writes rollout status transitions** (aside from `enqueue_rollout` which creates the initial `queuing` state and `cancel_rollout` which sets the `cancel_requested` flag).

#### K8s resources

| K8s Resource | agl-lite Role |
|-------------|---------------|
| **Deployment** | (Optional) agl-lite Service — if co-located with runner |
| **Job** | Individual rollout execution (one pod per rollout, or batched) |
| **Service** | Expose agl-lite Service to pods (or ExternalName/Ingress if remote) |
| **ConfigMap/Secret** | agl-lite Service endpoint URL, algorithm resources (prompts, model endpoints) |

#### Job naming and labeling

Jobs use deterministic names to prevent duplicates and enable crash recovery:

```yaml
metadata:
  name: agl-rollout-{rollout_id}       # deterministic — K8s rejects duplicates
  labels:
    agl-lite/rollout-id: {rollout_id}   # for label-based lookups
```

On creation failure due to `AlreadyExists`, the controller fetches the existing Job and proceeds.

#### Job template

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: agl-rollout-R1
  labels:
    agl-lite/rollout-id: R1
spec:
  backoffLimit: 3                    # K8s retries up to 3 times
  activeDeadlineSeconds: 600         # timeout after 10 minutes
  ttlSecondsAfterFinished: 3600     # auto-cleanup 1h after completion (for debugging)
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: agent
          image: user/my-agent:latest
          env:
            - name: ROLLOUT_ID
              value: "R1"
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
            - name: AGL_LITE_URL                 # single endpoint for everything
              value: "http://agl-lite:8080"
            - name: OPENAI_BASE_URL
              value: "$(AGL_LITE_URL)/rollout/$(ROLLOUT_ID)/attempt/$(POD_UID)/v1"
            - name: AGL_EVENT_URL
              value: "$(AGL_LITE_URL)/rollout/$(ROLLOUT_ID)/attempt/$(POD_UID)/events"
```

On each retry, K8s creates a new pod with a new `metadata.uid`, so `OPENAI_BASE_URL` automatically points to a fresh attempt partition in the Gateway.

#### Controller main loop

The controller uses the standard K8s controller pattern — **watch + periodic reconciliation**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Controller Main Loop                               │
│                                                                       │
│  1. WATCH: K8s Job events (label: agl-lite/rollout-id)               │
│     → on any Job status change, reconcile that rollout                │
│                                                                       │
│  2. POLL Store: query rollouts in "queuing" or with cancel_requested  │
│     → create Jobs for new queuing rollouts                            │
│     → process cancellations                                           │
│                                                                       │
│  3. PERIODIC FULL RECONCILE (every N seconds):                        │
│     → for each "queuing" rollout: ensure Job exists or create it      │
│     → for each "running" rollout: check Job status, sync if needed    │
│     → catches anything missed by watch/poll (crash recovery)          │
└──────────────────────────────────────────────────────────────────────┘
```

#### Reconcile logic (per rollout)

```python
def reconcile(rollout: Rollout):
    if rollout.status in (SUCCEEDED, TERMINAL_FAILED, CANCELLED):
        # Final state. Ensure K8s Job is cleaned up (idempotent).
        if rollout.job_name:
            k8s.delete_job_if_exists(rollout.job_name)
        return
    
    # ── Handle cancel first (takes priority) ──
    if rollout.cancel_requested:
        handle_cancel(rollout)
        return
    
    # ── Normal lifecycle ──
    if rollout.status == QUEUING:
        handle_queuing(rollout)
    elif rollout.status == RUNNING:
        handle_running(rollout)


def handle_cancel(rollout):
    """Process cancel_requested flag."""
    if rollout.status == QUEUING and rollout.job_name is None:
        # No Job exists. Straight to cancelled.
        store.update_rollout(rollout.rollout_id,
            status=CANCELLED,
            expected_version=rollout.version)
        return
    
    if rollout.job_name is None:
        # Running but no job_name? Shouldn't happen, but be safe.
        store.update_rollout(rollout.rollout_id,
            status=CANCELLED,
            expected_version=rollout.version)
        return
    
    job = k8s.get_job(rollout.job_name)
    
    if job is None:
        # Job already gone. Mark cancelled.
        store.update_rollout(rollout.rollout_id,
            status=CANCELLED,
            expected_version=rollout.version)
        return
    
    # Job exists. Check if it already succeeded before we delete.
    if job_has_condition(job, "Complete"):
        # Success already happened. Honor success over cancel.
        succeeded_pod_uid = find_succeeded_pod_uid(job)
        store.update_rollout(rollout.rollout_id,
            status=SUCCEEDED,
            succeeded_attempt_id=succeeded_pod_uid,
            expected_version=rollout.version)
        k8s.delete_job(rollout.job_name)
        return
    
    if job_has_condition(job, "Failed"):
        # Job already failed on its own. User wanted cancel — mark cancelled,
        # not terminal_failed. The intent was cancellation.
        store.update_rollout(rollout.rollout_id,
            status=CANCELLED,
            expected_version=rollout.version)
        k8s.delete_job(rollout.job_name)
        return
    
    # Job is still active. Delete it.
    # Foreground propagation: K8s deletes pods first, then Job.
    k8s.delete_job(rollout.job_name, propagation="Foreground")
    # Don't mark cancelled yet — Job is still terminating.
    # On next reconciliation, get_job returns None → mark cancelled.
    # This prevents a window where Store says "cancelled" but pods are
    # still running and writing events.


def handle_queuing(rollout):
    """Create K8s Job for a queuing rollout."""
    if rollout.job_name is not None:
        # Job was already created (controller crashed after creation
        # but before status update). Check its status.
        job = k8s.get_job(rollout.job_name)
        if job is not None:
            store.update_rollout(rollout.rollout_id,
                status=RUNNING, job_name=rollout.job_name,
                expected_version=rollout.version)
            return
        # Job name set but Job gone? Something went wrong.
        store.update_rollout(rollout.rollout_id,
            status=TERMINAL_FAILED,
            error_message="Job not found during recovery",
            expected_version=rollout.version)
        return
    
    job_name = f"agl-rollout-{rollout.rollout_id}"
    try:
        k8s.create_job(make_job_spec(rollout, job_name))
        store.update_rollout(rollout.rollout_id,
            status=RUNNING, job_name=job_name,
            expected_version=rollout.version)
    except K8sAlreadyExistsError:
        # Job exists (duplicate from previous attempt). Fetch and proceed.
        store.update_rollout(rollout.rollout_id,
            status=RUNNING, job_name=job_name,
            expected_version=rollout.version)
    except K8sError as e:
        store.update_rollout(rollout.rollout_id,
            status=TERMINAL_FAILED,
            error_message=f"Job creation failed: {e}",
            expected_version=rollout.version)


def handle_running(rollout):
    """Sync K8s Job status to Store for a running rollout."""
    job = k8s.get_job(rollout.job_name)
    
    if job is None:
        # Job disappeared (manually deleted, namespace cleanup).
        store.update_rollout(rollout.rollout_id,
            status=TERMINAL_FAILED,
            error_message="K8s Job not found",
            expected_version=rollout.version)
        return
    
    conditions = {c.type: c for c in (job.status.conditions or [])}
    
    if "Complete" in conditions:
        succeeded_pod_uid = find_succeeded_pod_uid(job)
        store.update_rollout(rollout.rollout_id,
            status=SUCCEEDED,
            succeeded_attempt_id=succeeded_pod_uid,
            expected_version=rollout.version)
    elif "Failed" in conditions:
        reason = conditions["Failed"].reason     # BackoffLimitExceeded, DeadlineExceeded
        message = conditions["Failed"].message
        store.update_rollout(rollout.rollout_id,
            status=TERMINAL_FAILED,
            error_message=f"{reason}: {message}",
            expected_version=rollout.version)
    # else: Job still active (running or between retries). No update needed.
```

All `update_rollout` calls use optimistic locking (`expected_version`). On `ConflictError`, the controller re-fetches the rollout and re-evaluates — another instance may have already handled it.

#### Edge cases

**Controller crash and recovery:**
On restart, periodic full reconciliation scans all non-terminal rollouts and syncs them with K8s Job status. This is idempotent — if a Job exists, its status is checked; if it's gone, the rollout is marked `terminal_failed`. The deterministic Job name (`agl-rollout-{rollout_id}`) ensures the controller can always find the Job for a rollout.

**Two controller instances (leader election gap):**
Both read `version=N`, both try to update the same rollout. One succeeds (`version→N+1`), the other gets `ConflictError`, re-fetches, sees the update was already done. Optimistic locking is the single serialization point.

**Store unavailable:**
The K8s controller pattern naturally handles this: if `update_rollout` fails due to Store being unreachable, the event is requeued with exponential backoff. The Job keeps running regardless of Store availability. When the Store comes back, the controller retries. No data loss — K8s Job status is the durable record.

**Job creation race (controller crash mid-creation):**
Controller creates Job, crashes before calling `update_rollout`. On restart, it finds a `queuing` rollout. It tries to create `agl-rollout-{rollout_id}` — K8s returns `AlreadyExists`. Controller catches this, sets status to `running`. Idempotent.

**Job deleted externally (`kubectl delete job`):**
Periodic reconciliation finds a `running` rollout whose Job no longer exists. Marks `terminal_failed` with error "K8s Job not found".

**Cancel + success race:**
User sets `cancel_requested=true`. Controller reconciles and checks Job status before deleting. If Job already has `Complete` condition, **success wins** — the work was done and trajectory data is captured. Controller marks `succeeded` and cleans up the Job. Cancel is effectively a no-op in this case.

**Cancel + failure race:**
If `cancel_requested=true` and Job has `Failed` condition, controller marks `cancelled` (not `terminal_failed`) — user's intent was to cancel, and the failure is consistent with that intent.

**Cancel during Job termination:**
After the controller calls `k8s.delete_job(propagation="Foreground")`, the Job enters a terminating state. Pods receive SIGTERM, then SIGKILL after grace period (default 30s). The controller does **not** mark `cancelled` until the Job is fully gone (`get_job` returns None). This prevents a window where the Store says `cancelled` but pods are still running and writing events to the Gateway.

**Node partition (two pods running):**
Data stays clean — each pod writes to its own `attempt_id` partition (see Section 3.3). The Job stays `active` throughout, so rollout remains `running`. Only when the Job reaches a terminal condition does the controller update the Store. If both pods succeed, K8s Job (`completions=1`) terminates after the first; `succeeded_attempt_id` records whichever pod K8s considers the completion.

**Algorithm queries stale status:**
Inherent in async systems. The controller syncs within seconds in normal operation. `wait_for_rollouts(ids, timeout)` long-polls the Store, returning when status changes or timeout fires.

**Rollout enqueued but controller is down:**
Rollouts stay `queuing`. When the controller comes back, periodic reconciliation picks them up and creates Jobs. No data loss, just delay.

### 3.6 Adapter Simplification

```python
class TrajectoryAdapter:
    """Convert trajectory events into algorithm-consumable format."""
    
    def adapt(self, trajectory: Trajectory) -> List[Triplet]:
        """Extract (prompt, response, reward) triplets from a trajectory."""
        model_events = [e for e in trajectory.events if e.event_type == "model_request"]
        reward_events = [e for e in trajectory.events if e.event_type == "reward"]
        total_reward = sum(r.data["value"] for r in reward_events) if reward_events else None
        
        return [Triplet(
            prompt=e.data["request"]["messages"],
            response=e.data["response"],
            reward=total_reward,
        ) for e in model_events]
```

The adapter only needs to understand `model_request` and `reward` — the two reserved types. User-defined event types (tool results, observations, custom metrics, etc.) are consumed by user-defined algorithm code, not the adapter.

No OTEL span parsing, no parent-child tree reconstruction, no attribute unflattening.

---

## 4. API Change Summary: Agent Lightning → agl-lite

### 4.1 Original Agent Lightning API Surface

The original `LightningStore` has **25+ methods** across 6 domains:

**Rollout management (8 methods):**
`start_rollout`, `enqueue_rollout`, `enqueue_many_rollouts`, `dequeue_rollout`, `dequeue_many_rollouts`, `update_rollout`, `query_rollouts`, `get_rollout_by_id`, `wait_for_rollouts`

**Attempt management (4 methods):**
`start_attempt`, `update_attempt`, `query_attempts`, `get_latest_attempt`

**Span management (5 methods):**
`add_span`, `add_many_spans`, `add_otel_span`, `query_spans`, `get_next_span_sequence_id`, `get_many_span_sequence_ids`

**Resource management (4 methods):**
`add_resources`, `update_resources`, `query_resources`, `get_resources_by_id`, `get_latest_resources`

**Worker management (3 methods):**
`query_workers`, `get_worker_by_id`, `update_worker`

**Meta (3):**
`capabilities`, `statistics`, `otlp_traces_endpoint`

**LLM Proxy** is a separate FastAPI server with:
- `RolloutAttemptMiddleware` (URL rewriting `/rollout/{rid}/attempt/{aid}/v1/...` → `/v1/...` + header injection)
- `StreamConversionMiddleware` (stream → non-stream for OTEL capture)
- `MessageInspectionMiddleware`
- `LightningSpanExporter` (OTEL span batching + flush to Store)
- `LightningOpenTelemetry` (LiteLLM callback wiring)

### 4.2 agl-lite API Surface

A single HTTP service with **~15 endpoints** across 4 domains:

**LLM proxy (2 paths, agent-facing):**

| Path | Replaces |
|------|----------|
| `POST /rollout/{rid}/attempt/{aid}/v1/...` | `RolloutAttemptMiddleware` + `StreamConversionMiddleware` + `LightningSpanExporter` + LiteLLM proxy. One path does it all: proxy + auto-capture as event. |
| `POST /rollout/{rid}/attempt/{aid}/events` | *New.* Explicit event reporting (reward, user-defined). No original equivalent — rewards were extracted from OTEL spans. |

**Rollout management (6 endpoints):**

| Endpoint | Replaces |
|----------|----------|
| `POST /api/rollouts` | `enqueue_rollout`, `enqueue_many_rollouts` (batch via JSON array body) |
| `GET /api/rollouts` | `query_rollouts` (simplified params: `status_in`, `cancel_requested`, `limit`, `offset`) |
| `GET /api/rollouts/{rid}` | `get_rollout_by_id` |
| `PATCH /api/rollouts/{rid}` | `update_rollout` (with optimistic locking via `expected_version`) |
| `POST /api/rollouts/{rid}/cancel` | *New.* Sets `cancel_requested` flag. Original used `update_rollout(status="cancelled")`. |
| `POST /api/rollouts/wait` | `wait_for_rollouts` |

**Event / trajectory (2 endpoints):**

| Endpoint | Replaces |
|----------|----------|
| `GET /api/events` | `query_spans` (events replace spans; filterable by `event_type`, `attempt_id`). A trajectory is `GET /api/events?rollout_id={rid}&attempt_id={aid}`. |
| `GET /api/attempts/{rid}` | `query_attempts` (returns only attempt IDs, not full Attempt objects) |

**Resource management (3 endpoints):**

| Endpoint | Replaces |
|----------|----------|
| `POST /api/resources` | `add_resources` |
| `GET /api/resources/latest` | `get_latest_resources` |
| `GET /api/resources/{id}` | `get_resources_by_id` |

### 4.3 What's Removed and Why

| Removed | Reason |
|---------|--------|
| `start_rollout` | No "start + first attempt" combo. K8s controller creates Jobs, pod UIDs are attempts. |
| `dequeue_rollout`, `dequeue_many_rollouts` | No work queue pull. K8s controller polls Store for `queuing` rollouts and creates Jobs. |
| `start_attempt`, `update_attempt`, `query_attempts` (as entity), `get_latest_attempt` | Attempt is a data tag (pod UID), not a managed entity. No attempt status machine. |
| `add_span`, `add_many_spans`, `add_otel_span` | Replaced by `model_request` events (auto-captured) and `/events` endpoint. |
| `get_next_span_sequence_id`, `get_many_span_sequence_ids` | Sequence auto-assigned by the service per `(rollout_id, attempt_id)`. No client-side allocation. |
| `query_workers`, `get_worker_by_id`, `update_worker` | K8s manages pod/worker lifecycle. No worker telemetry in agl-lite. |
| `update_resources` (in-place mutation) | Resources are immutable snapshots. Post a new one instead. |
| `query_resources` (paginated search) | Simplified to `get latest` and `get by ID`. |
| `capabilities`, `statistics`, `otlp_traces_endpoint` | No OTEL, no capability negotiation. Stats can be added later if needed. |
| `RolloutAttemptMiddleware` | Path parsing is built into the unified service. |
| `StreamConversionMiddleware` | No stream→non-stream conversion needed. Events capture the final response. |
| `LightningSpanExporter` | No OTEL span batching. Events written directly to in-process Store. |
| `LightningOpenTelemetry` callback | No LiteLLM, no OTEL callbacks. |

### 4.4 What's New

| New | Reason |
|-----|--------|
| `POST /rollout/{rid}/attempt/{aid}/events` | Explicit event ingestion (reward, user-defined types). Original had no direct event API — everything went through OTEL spans. |
| `POST /api/rollouts/{rid}/cancel` | Explicit cancel with `cancel_requested` flag. Cleaner than overloading `update_rollout(status="cancelled")`. |
| `cancel_requested` flag on Rollout | Separate intent from execution. Original used status directly. |
| `expected_version` on `PATCH /api/rollouts/{rid}` | Optimistic locking for safe concurrent updates. Original relied on in-process thread locks or single-writer patterns. |
| `succeeded_attempt_id` on Rollout | Directly links successful rollout to its trajectory data. Original required querying attempts to find the successful one. |
| Open event types (`event_type: str`) | Extensible without schema changes. Original was locked to OTEL span format. |
| Unified service (proxy + store) | Single deployment, in-process event capture on hot path. Original had separate LLM Proxy and Store Server. |
