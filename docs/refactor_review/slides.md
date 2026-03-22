---
theme: default
title: "agl-lite: Refactoring Agent Lightning"
info: |
  Design review for Agent Lightning developers and users.
  agl-lite is a minimal, workable replacement for Agent Lightning —
  same RL loop, simpler infrastructure, fewer dependencies.
author: Yuqi Yang
keywords: agl-lite, agent-lightning, agentic-rl, kubernetes
exportFilename: agl-lite-refactor-review
drawings:
  persist: false
transition: slide-left
mdc: true
---

# agl-lite

## A Minimal Workable Version of Agent Lightning

Design Review for Agent Lightning Developers

<!--
Opening: frame the goal — this is not a new project, it's a simplification
of Agent Lightning that preserves what matters and removes what doesn't.
-->

---

# Agenda

<v-clicks>

### 1. Why — the problem with current Agent Lightning

### 2. What changed — four key simplifications

### 3. Architecture — how agl-lite works

### 4. VERL integration — what the trainer sees

### 5. For developers — what lives where, how to collaborate

### 6. Status and next steps

</v-clicks>

<!--
Walk through at high level, then go deeper on each.
The key message: your training code barely changes; the infrastructure gets much simpler.
-->

---
layout: section
---

# Part 1
## Why agl-lite?

*What's wrong with the current stack?*

---

# The Dependency Problem

Agent Lightning pulls in **heavy, entangled dependencies** that are hard to maintain and debug.

<div class="grid grid-cols-3 gap-4 mt-6">
<div class="border-2 border-red-300 rounded-lg p-4 bg-red-50">
  <div class="text-lg font-bold text-red-700 mb-2">LiteLLM</div>
  <div class="compact-text">

  - Entire LLM gateway framework as a dependency
  - Frequent breaking changes upstream
  - Forces `stream=false` to backend for OTEL capture

  </div>
</div>
<div class="border-2 border-red-300 rounded-lg p-4 bg-red-50">
  <div class="text-lg font-bold text-red-700 mb-2">OpenTelemetry</div>
  <div class="compact-text">

  - 5+ OTEL packages (SDK, API, exporters, semconv)
  - Span trees, sequence IDs, parent-child linking
  - Custom `LightningSpanExporter` with batching

  </div>
</div>
<div class="border-2 border-red-300 rounded-lg p-4 bg-red-50">
  <div class="text-lg font-bold text-red-700 mb-2">Complex Store</div>
  <div class="compact-text">

  - 25+ methods across 6 domains
  - 3 abstraction layers (Collections → Store → Wrappers)
  - Attempt entity with its own state machine + watchdog

  </div>
</div>
</div>

<div class="mt-6 p-3 bg-yellow-50 rounded border border-yellow-200">

⚡ **Result:** 30K+ lines of Python. Debugging an LLM call failure means tracing through LiteLLM → OTEL → SpanExporter → Store → Adapter → Trainer.

</div>

<!--
The key insight: most of this complexity exists to work around limitations
of LiteLLM and OTEL, not because the RL loop needs it.
-->

---

# What We Actually Need

The RL training loop only requires **four operations**:

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

<v-clicks>

1. **Register** model servers (vLLM endpoints)
2. **Enqueue** rollouts (tasks for agents)
3. **Wait** for rollouts to complete
4. **Fetch** trajectories as `(prompt, response, reward)` triplets

</v-clicks>

</div>
<div>

<div class="compact-code-block">

```python
# What the trainer actually calls:
daemon.set_up_data_and_server(data, servers)
daemon.run_until_all_finished()
batch = daemon.get_train_data_batch()
# → DataProto with padded tensors
#   ready for PPO / GRPO / REINFORCE
```

</div>

</div>
</div>

<div class="mt-6 p-3 bg-blue-50 rounded border border-blue-200">

💡 **Principle:** agl-lite is the data pipe between agents and the trainer. Everything else — LiteLLM, OTEL, span trees, attempt entities, watchdogs — is accidental complexity.

</div>

<!--
This is the core insight that drives every design decision in agl-lite.
If the trainer only needs these 4 operations, we should build exactly that.
-->

---
layout: section
---

# Part 2
## What Changed

*Four simplifications, each removing a dependency*

---

# Four Key Simplifications

<div class="compact-table">

| # | Original | agl-lite | What's removed |
|---|----------|----------|----------------|
| 1 | **LiteLLM** proxy | Built-in gateway | LiteLLM package, OTEL callbacks, stream conversion middleware |
| 2 | **OpenTelemetry** spans | Event sequence | 5+ OTEL packages, span exporter, sequence ID allocation, span trees |
| 3 | **Custom store** (25+ methods) | Simplified store (12 endpoints) | Attempt entity, watchdog, worker telemetry, dequeue pattern |
| 4 | **In-process** execution | **Kubernetes** Jobs | Execution strategies, thread wrappers, process management |

</div>

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="border-2 border-green-300 rounded-lg p-4 bg-green-50">
  <div class="text-lg font-bold text-green-700 mb-2">✅ agl-lite</div>
  <div class="compact-text">

  - **3.7K lines** of Python (source)
  - **8 dependencies** (FastAPI, httpx, pydantic, etc.)
  - **284 tests**, all passing

  </div>
</div>
<div class="border-2 border-red-300 rounded-lg p-4 bg-red-50">
  <div class="text-lg font-bold text-red-700 mb-2">❌ Agent Lightning</div>
  <div class="compact-text">

  - **30K+ lines** of Python
  - **13+ heavy deps** (LiteLLM, OTEL, AgentOps...)
  - Complex multi-layer store architecture

  </div>
</div>
</div>

<!--
~8x reduction in code. The 3.7K includes everything: gateway, store,
controller, schemas, CLI, client library, and the VERL daemon.
-->

---

# Simplification 1: LiteLLM → Built-in Gateway

<div class="grid grid-cols-2 gap-6">
<div>

#### Before (Agent Lightning)

<div class="compact-text">

- LiteLLM as reverse proxy
- `RolloutAttemptMiddleware` rewrites URLs
- `StreamConversionMiddleware` forces `stream=false` to backend, re-streams as fake SSE
- `LightningOpenTelemetry` callback captures spans
- `LightningSpanExporter` batches + flushes to Store
- Agent → LiteLLM Proxy → Store *(separate services)*

</div>

</div>
<div>

#### After (agl-lite)

<div class="compact-text">

- Simple httpx reverse proxy (181 lines)
- Path encodes `rollout_id` + `attempt_id`
- **Real streaming** — tees chunks to agent, buffers for capture
- Events written to in-process store (dict lookup, ~100ns)
- Gateway route config: model mapping + parameter adjustment
- Single service — no network hop on hot path

</div>

</div>
</div>

<div class="compact-code-block mt-4">

```
Agent ◄──chunk──chunk──[DONE]──◄ Gateway ◄──chunk──chunk──[DONE]──◄ vLLM
                                    │
                               (buffer + capture)
                                    │
                                    ▼
                              write model_request event (in-process, ~100ns)
```

</div>

<!--
Key improvement: real streaming. Agent Lightning forced stream=false to backend
because OTEL couldn't handle SSE. agl-lite tees the real stream.
-->

---

# Simplification 2: OTEL Spans → Events

<div class="grid grid-cols-2 gap-6">
<div>

#### Before: Span Trees

<div class="compact-text">

- OpenTelemetry span with parent-child relationships
- `sequence_id` allocation per attempt
- `LightningSpanExporter` with buffering + flush
- Adapter reconstructs span tree → extracts triplets
- 5+ OTEL packages

</div>

<div class="compact-code-block">

```
Span(trace_id, span_id, parent_span_id,
     sequence_id, attributes={...},
     events=[...], links=[...])
```

</div>

</div>
<div>

#### After: Flat Event List

<div class="compact-text">

- Ordered list per `(rollout_id, attempt_id)`
- Two reserved types: `model_request`, `reward`
- Everything else: opaque pass-through
- No span trees, no sequence IDs
- Insertion order = temporal order

</div>

<div class="compact-code-block">

```
Event(event_type="model_request",
      rollout_id, attempt_id,
      timestamp, data={request, response,
                       server})
```

</div>

</div>
</div>

<div class="mt-4 p-3 bg-blue-50 rounded border border-blue-200">

💡 **agl-lite is a data pipe** — it stores and delivers events without restricting schema. Only `model_request` and `reward` have well-known structure. Users define their own event types freely.

</div>

---

# Simplification 3: Store — 25+ Methods → 12 Endpoints

<div class="grid grid-cols-2 gap-6">
<div>

#### Removed

<div class="compact-text">

- **Attempt entity** — pod UID is the attempt ID (K8s Downward API)
- **Watchdog** — K8s liveness probes + controller
- **Worker telemetry** — K8s manages pods
- **Dequeue pattern** — controller creates Jobs directly
- **Span sequence IDs** — insertion order suffices
- **`wait_for_rollouts`** — client-side polling

</div>

</div>
<div>

#### What Remains

<div class="compact-table">

| Domain | Endpoints |
|--------|-----------|
| LLM proxy | 2 (proxy + events) |
| Rollouts | 5 (CRUD + cancel) |
| Models | 3 (register, list, remove) |
| Events | 1 (query with filters) |
| Resources | 3 (add, get latest, get by ID) |

</div>

</div>
</div>

<div class="mt-4 p-3 bg-green-50 rounded border border-green-200">

✅ **Single in-memory store** — Python dict, single-threaded asyncio, no locks needed. Partition key `(rollout_id, attempt_id)` eliminates cross-agent contention entirely.

</div>

---

# Simplification 4: In-Process Execution → K8s Jobs

<div class="grid grid-cols-2 gap-6">
<div>

#### Before

<div class="compact-text">

- `SharedMemoryExecutionStrategy` — threads in one process
- `ClientServerExecutionStrategy` — multi-process with Store HTTP server
- Custom retry logic, watchdog health checks
- `LightningStoreThreaded` mutex wrapper
- `LitAgent` base class required (Python only)

</div>

</div>
<div>

#### After

<div class="compact-text">

- K8s Job per rollout (`backoffLimit` for retries)
- Pod UID = attempt ID (zero allocation)
- `activeDeadlineSeconds` for timeout
- Controller syncs Job status → Store
- **Language-agnostic**: any container that reads env vars

</div>

</div>
</div>

<div class="compact-code-block mt-4">

```yaml
# Controller injects into every agent pod:
env:
  - name: OPENAI_BASE_URL                    # agent uses any OpenAI-compatible SDK
    value: "http://agl-lite:8080/rollout/$(ROLLOUT_ID)/attempt/$(POD_UID)/v1"
  - name: AGL_TASK_INPUT                     # task payload (JSON)
    value: '{"prompt": "Solve: 2+2=?", ...}'
  - name: AGL_EVENT_URL                      # for posting rewards
    value: "http://agl-lite:8080/rollout/$(ROLLOUT_ID)/attempt/$(POD_UID)/events"
```

</div>

<!--
Agents don't import agl-lite. They're just containers that call an OpenAI endpoint.
The gateway transparently captures all LLM traffic.
-->

---
layout: section
---

# Part 3
## Architecture

*How the pieces fit together*

---

# High-Level Architecture

<div class="grid grid-cols-3 gap-4 mt-4">
<div class="arch-box arch-green">

#### Compute Backend

<div class="compact-text">

- vLLM inference servers (GPU)
- Training engine (VERL/Megatron)
- Managed by user, not agl-lite
- Pushes weights → inference servers

</div>
</div>
<div class="arch-box arch-blue">

#### agl-lite Service

<div class="compact-text">

- **Gateway**: LLM reverse proxy, auto-captures events
- **Store**: rollout queue, event storage, model registry
- Single process, single endpoint
- In-process store — no network hop

</div>
</div>
<div class="arch-box arch-red">

#### Agent Runner (K8s)

<div class="compact-text">

- **Controller**: creates Jobs, syncs status
- **Agent pods**: any container, any language
- Only needs network access to agl-lite
- No direct access to compute backend

</div>
</div>
</div>

<div class="compact-code-block mt-4">

```
Trainer ──POST /api/rollouts──▶ agl-lite ◀──K8s Job──── Controller
  ▲                               │  ▲                      │
  │                          (events)  │                     │
  │                               │    │              (create/watch Jobs)
  │                               ▼    │                     │
  └──GET /api/events────────── Store   Gateway ◀─── Agent Pods
                                         │                   │
                                         └───── vLLM ────────┘
                                          (proxy + capture)
```

</div>

<!--
Three groups, connected only by HTTP. No shared memory, no in-process coupling.
Each can be deployed independently.
-->

---

# Data Flow: From Task to Training Tensor

<div class="compact-text">

<v-clicks>

**1. Trainer enqueues rollouts** → `POST /api/rollouts` with task inputs

**2. Controller creates K8s Jobs** → injects `OPENAI_BASE_URL`, `AGL_TASK_INPUT` as env vars

**3. Agent runs** → calls `OPENAI_BASE_URL` (unaware of agl-lite) → gateway proxies to vLLM

**4. Gateway auto-captures** → `model_request` event with `{request, response, server}` + token IDs

**5. Agent posts reward** → `POST .../events` with `{event_type: "reward", data: {value: 0.85}}`

**6. Agent exits** → K8s marks Job complete → controller updates rollout → `succeeded`

**7. Trainer fetches triplets** → `GET /api/events?format=triplet` → trimmed to `{prompt_token_ids, response_token_ids, value}`

**8. Daemon builds tensors** → `get_train_data_batch()` → padded `DataProto` → PPO/GRPO step

</v-clicks>

</div>

<div class="mt-4 p-3 bg-yellow-50 rounded border border-yellow-200">

⚡ **Key insight:** The gateway is the instrumentation. No tracer, no OTEL, no agent SDK. The agent just calls an OpenAI endpoint. Everything is captured transparently.

</div>

---

# Weight Update Protocol

Clean coordination between training and inference — no special "update mode" API.

<div class="compact-code-block">

```
Trainer                                    agl-lite Gateway
───────                                    ────────────────
1. Training step complete

2. DELETE /api/models/qwen-7b        ──→   Model pool empty
                                           New LLM requests → 503 + Retry-After
                                           (OpenAI SDK auto-retries, agent stays alive)

3. Kill old vLLM servers
4. Launch new servers with updated weights
5. Wait for servers to be ready

6. POST /api/models                  ──→   Servers registered (version: 43)
   [{model: "qwen-7b",                     Routing resumes
     endpoint: "http://vllm-0:8000/v1",    Retrying agents succeed on next attempt
     version: 43}]
```

</div>

<div class="mt-4 p-3 bg-green-50 rounded border border-green-200">

✅ **No agent crashes** during weight updates. 503 + Retry-After is handled by standard OpenAI SDK. No K8s Job retry consumed. Emergent from CRUD — not a special API.

</div>

---
layout: section
---

# Part 4
## VERL Integration

*What changes for the trainer — and what doesn't*

---

# AglLiteDaemon: The Bridge

`AglLiteDaemon` replaces `AgentModeDaemon` — same interface, simpler internals.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### New Code (187 lines)

<div class="compact-text">

Store interaction via `AglLiteClient`:

- `set_up_data_and_server` → `client.register_models()` + `client.enqueue_rollouts()`
- `run_until_all_finished` → poll `client.get_rollout()` for `succeeded`
- `_validate_data` → `client.get_events(format="triplet")`
- `clear_data_and_server` → reset internal state

</div>

</div>
<div>

#### Copied Unchanged (510 lines)

<div class="compact-text">

Tensor construction from Agent Lightning:

- `get_train_data_batch()` — triplets → padded tensors → `DataProto` (290 lines)
- Multimodal support — mrope position IDs, image grid (53 lines)
- Utilities — left/right padding, token matching (127 lines)
- Validation and metrics (40 lines)

</div>

</div>
</div>

<div class="mt-4 p-3 bg-blue-50 rounded border border-blue-200">

💡 **The tensor math is identical.** `get_train_data_batch()` is copied verbatim from `AgentModeDaemon`. If it works in Agent Lightning, it works in agl-lite.

</div>

<!--
This is the most important slide for VERL developers.
The training math doesn't change. Only the data transport layer changes.
-->

---

# Side-by-Side: What the Daemon Calls

<div class="compact-table">

| Operation | AgentModeDaemon (original) | AglLiteDaemon (new) |
|-----------|---------------------------|---------------------|
| Register model | `store.add_resources({model_endpoints: ...})` | `client.register_models([{model, endpoint}])` |
| Enqueue work | `store.enqueue_many_rollouts(rollouts)` | `client.enqueue_rollouts([{input, config}])` |
| Wait for completion | `store.wait_for_rollouts(ids)` | poll `client.get_rollout(id)` until `succeeded` |
| Fetch trajectories | `store.query_spans(rid)` → `adapter.adapt(spans)` | `client.get_events(rid, format="triplet")` |
| Build tensors | `get_train_data_batch()` → `DataProto` | **Same** — copied unchanged |

</div>

<div class="mt-6 p-3 bg-green-50 rounded border border-green-200">

✅ **The adapter is now server-side.** `format=triplet` on `GET /api/events` trims events to `{prompt_token_ids, response_token_ids}` + `{reward_value}`. No OTEL span parsing. No client-side adapter.

</div>

---

# Triplet Format: Server-Side Extraction

`GET /api/events?rollout_id=R1&format=triplet` returns pre-trimmed events:

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### Full Event (default)

<div class="compact-code-block-xs">

```json
{
  "event_type": "model_request",
  "data": {
    "request": {"model": "qwen-7b",
      "messages": [...],
      "return_token_ids": true},
    "response": [
      {"choices": [{"delta": {"content": "The"},
        "token_ids": [464]}],
       "prompt_token_ids": [1, 2, 3]},
      {"choices": [{"delta": {"content": " answer"},
        "token_ids": [1234]}]},
      {"choices": [{"delta": {"content": " is 4"},
        "token_ids": [374, 220, 19]}]}
    ],
    "server": {"model": "qwen-7b",
      "endpoint": "http://vllm:8000/v1",
      "version": 42},
    "latency_ms": 1234.5
  }
}
```

</div>
</div>
<div>

#### Triplet Format

<div class="compact-code-block-xs">

```json
{
  "event_type": "model_request",
  "data": {
    "prompt_token_ids": [1, 2, 3],
    "response_token_ids": [464, 1234, 374, 220, 19],
    "server": {"model": "qwen-7b",
      "endpoint": "http://vllm:8000/v1",
      "version": 42}
  }
}
```

</div>

<div class="compact-code-block-xs mt-4">

```json
{
  "event_type": "reward",
  "data": {
    "value": 0.85
  }
}
```

</div>

</div>
</div>

<!--
The server gathers token_ids across SSE chunks (streaming responses),
extracts prompt_token_ids from the first chunk, and merges response_token_ids.
This is what the daemon needs for tensor construction.
-->

---

# What the Trainer Code Looks Like

<div class="compact-code-block">

```python
# Before (Agent Lightning) — in the VERL trainer
daemon = AgentModeDaemon(
    store=lightning_store,           # LightningStore client
    proxy=llm_proxy,                 # LiteLLM proxy reference
    adapter=TracerTraceToTriplet(),  # OTEL span → triplet adapter
    train_rollout_n=4, tokenizer=tokenizer, ...
)

# After (agl-lite) — drop-in replacement
daemon = AglLiteDaemon(
    agl_lite_url="http://agl-lite:8080",  # single HTTP endpoint
    agl_key="agl_xxx...",                  # shared API key
    train_rollout_n=4, tokenizer=tokenizer, ...
)

# Training loop — IDENTICAL in both cases
for batch in dataset:
    daemon.set_up_data_and_server(batch, vllm_addresses)
    daemon.run_until_all_finished()
    data_proto = daemon.get_train_data_batch()
    # → PPO / GRPO / REINFORCE step with data_proto
    daemon.clear_data_and_server()
```

</div>

<div class="mt-4 p-3 bg-green-50 rounded border border-green-200">

✅ **Two lines change** in the trainer: the daemon constructor. The training loop is untouched.

</div>

---
layout: section
---

# Part 5
## For Developers

*How to collaborate on agl-lite*

---

# What Lives Where

<div class="grid grid-cols-2 gap-6">
<div>

#### agl-lite (this repo)

<div class="compact-text">

- **Gateway** — LLM reverse proxy + event capture
- **Store** — rollout queue, event storage, model registry
- **Controller** — K8s Job management
- **Client** — `AglLiteClient` Python library + `agl-client` CLI
- **VERL Daemon** — `AglLiteDaemon` (bridge to trainer)
- **Schemas** — Pydantic models for API
- No torch, no VERL, no training code

</div>

</div>
<div>

#### Agent Lightning (upstream)

<div class="compact-text">

- **Trainer** — `AgentLightningTrainer` / `RayPPOTrainer`
- **Dataset** — data loading, batching
- **Entrypoint** — CLI for launching training
- **Agent implementations** — task-specific agents
- These files will eventually be copied to agl-lite once validated
- Training code has no agl-lite dependency — only HTTP

</div>

</div>
</div>

<div class="mt-4 p-3 bg-yellow-50 rounded border border-yellow-200">

⚡ **Dependency direction:** Trainer → agl-lite HTTP API. agl-lite never imports torch or VERL. The daemon lives in agl-lite but its tensor code only runs when torch is available.

</div>

---

# Agent Contract: Language-Agnostic

Agents don't import agl-lite. They're just containers that:

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

<div class="compact-text">

1. Read `AGL_TASK_INPUT` env var (JSON)
2. Call `OPENAI_BASE_URL` with any OpenAI-compatible SDK
3. (Optional) Post reward to `AGL_EVENT_URL`
4. Exit with code 0 on success

</div>

<div class="compact-code-block mt-4">

```python
# Python agent — no agl-lite import
import os, json, openai

task = json.loads(os.environ["AGL_TASK_INPUT"])
client = openai.OpenAI()  # uses OPENAI_BASE_URL

response = client.chat.completions.create(
    model="gpt-4.1",  # gateway routes to vLLM
    messages=[{"role": "user",
               "content": task["prompt"]}],
)
# Gateway captures this call automatically
```

</div>
</div>
<div>

<div class="compact-code-block">

```javascript
// JavaScript agent — same contract
const task = JSON.parse(process.env.AGL_TASK_INPUT);

const response = await fetch(
  process.env.OPENAI_BASE_URL +
    "/chat/completions",
  {
    method: "POST",
    headers: {
      Authorization:
        `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: "gpt-4.1",
      messages: [{ role: "user",
                   content: task.prompt }],
    }),
  }
);
```

</div>
</div>
</div>

<!--
This is a huge win for the research team. Any agent framework, any language.
No base class, no SDK, no instrumentation code.
-->

---

# Gateway Route Config

The gateway maps agent-facing model names to actual backend models:

<div class="grid grid-cols-2 gap-6">
<div>

<div class="compact-code-block">

```yaml
# gateway-config.yaml
routes:
  - model_in: "gpt-4.1"       # agent sends
    model_out: "qwen-7b"      # gateway routes to
    params:
      add:
        temperature: 0.7
        max_tokens: 4096
        return_token_ids: true # for RL training
      drop:
        - stream_options      # vLLM unsupported
        - logprobs            # save compute

  - model_in: "*"             # catch-all
    model_out: "*"            # passthrough
    params:
      add:
        return_token_ids: true
```

</div>
</div>
<div>

#### What This Enables

<div class="compact-text">

- **Model aliasing** — agents use familiar names (`gpt-4.1`), gateway routes to local models
- **Parameter normalization** — enforce `return_token_ids: true` for training without agent changes
- **Backend compatibility** — drop unsupported params for vLLM/TGI
- **Wildcard catch-all** — any model not matched gets passthrough with token ID injection
- Events record both original and adjusted params

</div>

</div>
</div>

---
layout: section
---

# Part 6
## Status & Next Steps

---

# Current Status

<div class="grid grid-cols-2 gap-6">
<div>

#### Implemented ✅

<div class="compact-text">

- Gateway with route config + real streaming
- In-memory store (rollouts, events, models, resources)
- K8s controller (kr8s-based, Job lifecycle)
- `AglLiteClient` + `agl-client` CLI
- `AglLiteDaemon` for VERL integration
- `format=triplet` on events API
- Auth (shared API key)
- Math PoC E2E (mock + real vLLM)
- **284 tests** passing

</div>
</div>
<div>

#### Next Steps 🔜

<div class="compact-text">

- **Phase 5c**: Full training loop E2E
  - Training script: agl-lite + VERL on Qwen2.5-1.5B-Instruct
  - Weight update protocol validation
  - Multi-iteration training with reward improvement

- **Phase 6**: Copy remaining VERL files
  - `trainer.py`, `dataset.py`, `entrypoint.py` from Agent Lightning
  - Validate end-to-end training parity

- **Phase 7**: Production polish
  - Structured logging + metrics
  - CI/CD pipeline
  - User documentation

</div>
</div>
</div>

---

# How You Can Help

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="border-2 border-blue-300 rounded-lg p-4 bg-blue-50">
  <div class="text-lg font-bold text-blue-700 mb-2">🔬 Researchers</div>
  <div class="compact-text">

  - Try the Math PoC with your own tasks
  - Test agent implementations against the gateway
  - Validate triplet format covers your training needs
  - Report edge cases in multimodal / multi-turn workflows

  </div>
</div>
<div class="border-2 border-purple-300 rounded-lg p-4 bg-purple-50">
  <div class="text-lg font-bold text-purple-700 mb-2">🛠️ Engineers</div>
  <div class="compact-text">

  - Review the architecture doc (`docs/design/0_architecture.md`)
  - Test K8s controller with your cluster setup
  - Contribute persistent store backends (SQLite, PostgreSQL)
  - Help with CI/CD and deployment automation

  </div>
</div>
</div>

<div class="mt-6 p-3 bg-green-50 rounded border border-green-200">

✅ **Getting started:** `git clone` → `uv sync` → `uv run pytest` (284 tests, ~10s). The Math PoC runs on minikube with mock LLM or real vLLM. See `examples/math-poc/README.md`.

</div>

---
layout: center
---

# Questions?

<div class="mt-8 text-lg">

**Repo:** `agl-lite` · **Arch doc:** `docs/design/0_architecture.md`

**Design principle:** agl-lite is the data pipe. Simple HTTP API in, training tensors out.

</div>

<!--
Open for questions. Key points to reiterate:
1. Your training loop barely changes (2 lines in constructor)
2. Agents are language-agnostic containers
3. 8x less code, 0 heavy dependencies
4. Same tensor math, same DataProto output
-->
