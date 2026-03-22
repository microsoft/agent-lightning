---
theme: default
title: "agl-lite: A Minimal Workable Version of Agent Lightning"
info: |
  Design review for Agent Lightning developers and users.
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

Design Review · March 2026

---

# Agenda

<v-clicks>

### 1. What is agl-lite — three design choices

### 2. Architecture overview

### 3. Deep dive — Gateway

### 4. Deep dive — Store + K8s Controller

### 5. VERL integration

### 6. Status & collaboration

</v-clicks>

---
layout: section
---

# Part 1
## What is agl-lite

*Three design choices*

---

# Three Design Choices

<v-clicks>

**1. Self-owned request gateway** — a purpose-built LLM reverse proxy that captures all request-response data transparently as it flows through

**2. Gateway-level data capture** — instead of instrumenting agents, the gateway records request-response pairs during transfer and stores them as events — the proxy *is* the instrumentation

**3. Kubernetes-native agent runner** — K8s Jobs as the execution unit, pod UIDs as attempt IDs, Job lifecycle as the retry mechanism

</v-clicks>

<div class="mt-6 p-3 bg-blue-50 rounded border border-blue-200" v-click>

💡 **A natural consequence of (3):** since K8s owns retry, timeout, and scheduling, the data store and rollout state machine become much simpler — the store focuses purely on data, not execution control.

</div>

<!--
These three choices drive everything in agl-lite.
They're independent — each can be understood on its own.
The store simplification falls out of choice 3.
-->

---
layout: section
---

# Part 2
## Architecture Overview

---

# High-Level Architecture

<img src="../images/lite_arch.excalidraw.svg" class="mx-auto" style="max-height: 360px;" alt="agl-lite architecture: Compute Backend (green), AGL-Lite service (blue), Agent Runner on K8s (red)" />

<!--
Three groups connected only by HTTP.
Walk through the arrows: tasks flow right, trajectories flow left.
Agents call through the gateway to reach vLLMs — they never talk to compute backend directly.
-->

---

# Three Groups, One Protocol

<div class="grid grid-cols-3 gap-4 mt-2">
<div class="arch-box arch-green">

#### Compute Backend

<div class="compact-text">

- vLLM inference servers
- Training engine (VERL / Megatron)
- **Managed by user** — agl-lite doesn't deploy this
- Pushes updated weights → vLLMs

</div>
</div>
<div class="arch-box arch-blue">

#### agl-lite Service

<div class="compact-text">

- **Gateway**: LLM reverse proxy, auto event capture
- **Data Store**: rollout queue, events, model registry
- Single process, single endpoint
- In-process store — no network hop on hot path

</div>
</div>
<div class="arch-box arch-red">

#### Agent Runner (K8s)

<div class="compact-text">

- **Controller**: creates/watches K8s Jobs
- **Agent pods**: any container, any language
- Only needs network access to agl-lite
- No direct access to compute backend

</div>
</div>
</div>

<div class="mt-4 p-3 bg-yellow-50 rounded border border-yellow-200">

⚡ **No co-location assumption.** The three groups communicate only via HTTP. They can live in the same cluster, separate clusters, or across cloud boundaries.

</div>

---
layout: section
---

# Part 3
## Deep Dive — Gateway

*Transparent proxy, model routing, weight updates*

---

# Gateway: Transparent LLM Proxy

Agents call the gateway as a normal OpenAI endpoint — they don't know agl-lite exists.

<div class="compact-code-block">

```
Agent ◄──chunk──chunk──chunk──[DONE]──◄ Gateway ◄──chunk──chunk──chunk──[DONE]──◄ vLLM
                                           │
                                      buffer chunks in memory
                                           │
                                      on stream complete:
                                           ▼
                                     write model_request event
                                     (in-process dict write, ~100ns)
```

</div>

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### How it works

<div class="compact-text">

- Path encodes context: `/rollout/{rid}/attempt/{aid}/v1/...`
- **Real streaming** — tees SSE chunks to agent immediately while buffering
- Validates rollout exists in-process (~100ns dict lookup) before forwarding
- At stream end, writes complete `model_request` event with request, response, server metadata

</div>
</div>
<div>

#### What gets captured

<div class="compact-code-block-xs">

```json
{
  "event_type": "model_request",
  "data": {
    "request": { "model": "gpt-4.1", "messages": [...] },
    "response": [... SSE chunks ...],
    "server": { "model": "qwen-7b",
                "endpoint": "http://vllm-0:8000/v1",
                "version": 42 },
    "latency_ms": 1234.5
  }
}
```

</div>
</div>
</div>

---

# Gateway: Model Routing + Parameter Injection

<div class="grid grid-cols-2 gap-6">
<div>

#### Route config (YAML, loaded at startup)

<div class="compact-code-block">

```yaml
routes:
  - model_in: "gpt-4.1"       # what agent sends
    model_out: "qwen-7b"      # rewritten to this
    params:
      add:
        temperature: 0.7
        max_tokens: 4096
        return_token_ids: true
      drop:
        - stream_options
        - logprobs

  - model_in: "*"             # catch-all wildcard
    model_out: "*"            # passthrough
    params:
      add:
        return_token_ids: true
```

</div>
</div>
<div>

#### What this enables

<div class="compact-text">

- **Model aliasing** — agents use familiar names (`gpt-4.1`), gateway routes to local vLLMs
- **Parameter enforcement** — inject `return_token_ids: true` for training without touching agent code
- **Backend compatibility** — strip params that vLLM/TGI don't support
- **Wildcard catch-all** — unmatched models pass through with param adjustments
- Events record **both** original and rewritten request

</div>

<div class="mt-4 p-3 bg-green-50 rounded border border-green-200">

✅ First match wins. Specific rules before wildcards. `model_out: "*"` keeps the original model name.

</div>

</div>
</div>

---

# Gateway: Weight Update Protocol

Training-inference coordination via standard CRUD — no special "update mode" API.

<div class="compact-code-block">

```
Trainer                                    agl-lite Gateway
───────                                    ────────────────
1. Training step complete

2. DELETE /api/models/qwen-7b        ──→   Pool empty → new LLM requests get 503 + Retry-After
                                           (OpenAI SDK auto-retries; agent pod stays alive)

3. Kill old vLLM, launch with new weights, wait for ready

4. POST /api/models                  ──→   Servers registered (version: 43), routing resumes
   [{model: "qwen-7b",                     Retrying agents succeed transparently
     endpoint: "...", version: 43}]
```

</div>

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="compact-text">

#### During the unavailable window

- Gateway returns **503 + Retry-After**
- Standard OpenAI SDKs auto-retry with backoff
- Agent pod stays alive — **no K8s Job retry consumed**
- Next retry after registration succeeds transparently

</div>
<div class="compact-text">

#### Per-request version tracking

Every `model_request` event records `server.version`:
- Importance sampling across policy versions
- Off-policy correction for stale data
- Training data filtering by version
- Online RL: rolling update, one server at a time

</div>
</div>

---
layout: section
---

# Part 4
## Deep Dive — Store + K8s Controller

*Events, rollout lifecycle, reconciliation*

---

# Event Model: Flat Sequence of Events

<div class="grid grid-cols-2 gap-6">
<div>

#### Design

<div class="compact-text">

- Ordered list per `(rollout_id, attempt_id)`
- **Two reserved types**: `model_request` (auto-captured by gateway), `reward` (training signal)
- **Everything else**: opaque pass-through — users define their own event types freely
- Insertion order = temporal order (single-threaded asyncio)

</div>

<div class="compact-code-block mt-2">

```python
class Event:
    event_type: str       # "model_request", "reward",
                          # or any user-defined string
    rollout_id: str
    attempt_id: str       # = K8s pod UID
    timestamp: float
    data: Dict            # type-specific payload
```

</div>

</div>
<div>

#### Example trajectory

<div class="compact-code-block">

```
[0]  model_request  (auto-captured by gateway)
[1]  tool_result    (user-defined — runner reports)
[2]  model_request  (auto-captured by gateway)
[3]  action         (user-defined — agent submits)
[4]  reward         (environment scores: 0.85)
```

</div>

<div class="mt-4 p-3 bg-blue-50 rounded border border-blue-200">

💡 **agl-lite is a data pipe.** It stores and delivers events without restricting schema. Only `model_request` and `reward` have well-known structure. Users extend freely.

</div>

</div>
</div>

---

# K8s-Native: Attempt = Pod UID

Every K8s pod has a unique `metadata.uid` — agl-lite uses this as the attempt ID. Zero allocation, zero coordination.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### Data partitioning on retry

<div class="compact-code-block">

```
Pod #1 (uid=aaa):  rollout=R1, attempt=aaa
  → [req1, req2, req3] → pod crashes

Pod #2 (uid=bbb):  rollout=R1, attempt=bbb
  → [req1', req2', req3', req4'] → succeeds
```

</div>

<div class="compact-text mt-2">

- Each pod writes to its own `(rid, aid)` partition
- Data never collides — even with node partitions
- Algorithm queries the succeeded attempt

</div>

</div>
<div>

#### Simplified rollout states

Since K8s owns retry (`backoffLimit`) and timeout (`activeDeadlineSeconds`):

<div class="compact-code-block">

```
queuing ──→ running ──→ succeeded
   │           │         (terminal)
   │           └──────→ terminal_failed
   │                     (terminal)
   └──────────────────→ cancelled
                         (terminal)
```

</div>

<div class="compact-text mt-2">

- No attempt status machine
- No watchdog or health checks
- Controller is the sole writer of transitions
- Store enforces valid transitions, rejects invalid

</div>

</div>
</div>

---

# Controller: Reconciliation Pattern

The controller is the bridge between the store and K8s. It's the **only component that writes rollout status transitions**.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### Main loop

<div class="compact-text">

1. **Poll store** for `queuing` rollouts → create K8s Jobs
2. **Watch Jobs** for status changes → sync to store
3. **Handle cancellation**: `cancel_requested` flag → delete Job → `cancelled`
4. **Periodic full reconcile** for crash recovery

</div>

#### Job construction

<div class="compact-text">

- `job_template` (raw K8s pod spec from resources) + `rollout.config` (per-rollout overrides)
- Controller injects env vars into container named `agent`
- Deterministic name `agl-rollout-{rid}` → idempotent on crash recovery

</div>

</div>
<div>

#### Controller-injected env vars

<div class="compact-code-block-xs">

```yaml
env:
  - name: OPENAI_BASE_URL
    value: "$(AGL_LITE_URL)/rollout/$(ROLLOUT_ID)
           /attempt/$(POD_UID)/v1"
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: agl-lite-keys
        key: AGL_KEY
  - name: AGL_TASK_INPUT
    value: '{"prompt": "Solve: 2+2=?"}'
  - name: AGL_EVENT_URL
    value: "$(AGL_LITE_URL)/rollout/$(ROLLOUT_ID)
           /attempt/$(POD_UID)/events"
```

</div>

<div class="compact-text mt-2">

On retry, K8s creates a new pod → new `POD_UID` → `OPENAI_BASE_URL` and `AGL_EVENT_URL` automatically point to a fresh attempt partition.

</div>

</div>
</div>

---
layout: section
---

# Part 5
## VERL Integration

*AglLiteDaemon, trainer code, triplet format*

---

# AglLiteDaemon: The Trainer Bridge

`AglLiteDaemon` provides the same interface as `AgentModeDaemon` — the trainer barely changes.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### New code (187 lines)

<div class="compact-text">

Store interaction via `AglLiteClient`:

- `set_up_data_and_server` → register models + enqueue rollouts
- `run_until_all_finished` → poll rollout status until `succeeded`
- `_validate_data` → fetch events with `format=triplet`
- `clear_data_and_server` → reset state

</div>

</div>
<div>

#### Copied unchanged (510 lines)

<div class="compact-text">

Tensor construction from Agent Lightning `AgentModeDaemon`:

- `get_train_data_batch()` — triplets → padded tensors → `DataProto` (290 lines)
- Multimodal — mrope position IDs, image grid for Qwen2-VL (53 lines)
- Utilities — left/right padding, token matching (127 lines)
- Validation and metrics (40 lines)

</div>

</div>
</div>

<div class="mt-4 p-3 bg-green-50 rounded border border-green-200">

✅ **The tensor math is identical.** `get_train_data_batch()` is copied verbatim. The change is only in how data reaches it — HTTP instead of in-process store calls.

</div>

---

# What the Trainer Code Looks Like

<div class="compact-code-block">

```python
# Using agl-lite
daemon = AglLiteDaemon(
    agl_lite_url="http://agl-lite:8080",  # single HTTP endpoint
    agl_key="agl_xxx...",                  # shared API key
    train_rollout_n=4, tokenizer=tokenizer, mini_batch_size=64, pad_token_id=0,
)

# Training loop
for batch in dataset:
    daemon.set_up_data_and_server(batch, vllm_addresses)
    daemon.run_until_all_finished()
    data_proto = daemon.get_train_data_batch()
    # → DataProto with padded tensors, ready for PPO / GRPO / REINFORCE
    daemon.clear_data_and_server()
```

</div>

<div class="mt-4 p-3 bg-blue-50 rounded border border-blue-200">

💡 **Same four methods, same `DataProto` output.** The daemon constructor is the only difference — an HTTP URL instead of store/proxy/adapter objects.

</div>

---

# Triplet Format: Server-Side Extraction

`GET /api/events?rollout_id=R1&format=triplet` trims events to what tensor construction needs.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### Full event (default)

<div class="compact-code-block-xs">

```json
{
  "event_type": "model_request",
  "data": {
    "request": {"model": "qwen-7b", "messages": [...]},
    "response": [
      {"choices": [...], "token_ids": [464],
       "prompt_token_ids": [1, 2, 3]},
      {"choices": [...], "token_ids": [1234]},
      {"choices": [...], "token_ids": [374, 220, 19]}
    ],
    "server": {"model": "qwen-7b", "version": 42, ...},
    "latency_ms": 1234.5
  }
}
```

</div>
</div>
<div>

#### Triplet format

<div class="compact-code-block-xs">

```json
{
  "event_type": "model_request",
  "data": {
    "prompt_token_ids": [1, 2, 3],
    "response_token_ids": [464, 1234, 374, 220, 19],
    "server": {"model": "qwen-7b", "version": 42, ...}
  }
}
```

</div>

<div class="compact-code-block-xs mt-4">

```json
{ "event_type": "reward", "data": { "value": 0.85 } }
```

</div>
</div>
</div>

<div class="mt-2 p-2 bg-yellow-50 rounded border border-yellow-200">
<div class="compact-text">

⚡ Server gathers `token_ids` across SSE chunks, extracts `prompt_token_ids` from first chunk, merges `response_token_ids` — the daemon receives ready-to-use IDs.

</div>
</div>

---

# Agent Contract: Language-Agnostic

Agents don't import agl-lite. They're containers that read env vars and call an OpenAI endpoint.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

<div class="compact-code-block">

```python
# Python agent — no agl-lite import
import os, json, openai

task = json.loads(os.environ["AGL_TASK_INPUT"])
client = openai.OpenAI()  # reads OPENAI_BASE_URL

resp = client.chat.completions.create(
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
const task = JSON.parse(
  process.env.AGL_TASK_INPUT);

const resp = await fetch(
  `${process.env.OPENAI_BASE_URL}` +
    `/chat/completions`,
  { method: "POST",
    headers: { Authorization:
      `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: "gpt-4.1",
      messages: [{ role: "user",
                   content: task.prompt }],
    })
  }
);
```

</div>
</div>
</div>

<div class="mt-2 p-3 bg-green-50 rounded border border-green-200">

✅ Any language, any framework, any OpenAI-compatible SDK. No base class, no agl-lite dependency.

</div>

---
layout: section
---

# Part 6
## Status & Collaboration

---

# Current Status

<div class="grid grid-cols-2 gap-6">
<div>

#### What's built ✅

<div class="compact-text">

- Gateway with route config + real streaming
- In-memory store (rollouts, events, models, resources)
- K8s controller (kr8s-based, Job lifecycle)
- `AglLiteClient` library + `agl-client` CLI
- `AglLiteDaemon` for VERL integration
- `format=triplet` on events API
- Auth (shared API key)
- Math PoC end-to-end (mock + real vLLM)
- **3.7K lines source, 284 tests** (~10s)

</div>
</div>
<div>

#### What's next 🔜

<div class="compact-text">

- **Phase 5c**: Full training loop end-to-end
  - agl-lite + VERL + Qwen2.5-1.5B-Instruct
  - Weight update protocol validation
  - Multi-iteration training with reward improvement

- **Phase 6**: Copy remaining VERL files
  - `trainer.py`, `dataset.py`, `entrypoint.py`
  - Validate training parity

- **Phase 7**: Production readiness
  - Structured logging, metrics, CI/CD
  - Persistent store backends (SQLite, PostgreSQL)
  - Documentation

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
  - Validate triplet format for your training needs
  - Report edge cases in multimodal / multi-turn agents

  </div>
</div>
<div class="border-2 border-purple-300 rounded-lg p-4 bg-purple-50">
  <div class="text-lg font-bold text-purple-700 mb-2">🛠️ Engineers</div>
  <div class="compact-text">

  - Review the architecture doc (`docs/design/0_architecture.md`)
  - Test K8s controller with your cluster setup
  - Contribute persistent store backends
  - Help with CI/CD and deployment automation

  </div>
</div>
</div>

<div class="mt-6 p-3 bg-green-50 rounded border border-green-200">

✅ **Getting started:** `git clone` → `uv sync` → `uv run pytest` (284 tests, ~10s). Math PoC runs on minikube. See `examples/math-poc/README.md`.

</div>

---
layout: center
---

# Questions?

<div class="mt-8 text-lg">

**Architecture doc:** `docs/design/0_architecture.md`

**Design principle:** agl-lite is the data pipe — simple HTTP API in, training tensors out.

</div>

<!--
Key points to reiterate if asked:
1. Three choices: own gateway, gateway-level capture, K8s-native
2. Trainer code: same 4 methods, same DataProto output
3. Agents: language-agnostic, just env vars + OpenAI SDK
4. Tensor math: copied verbatim from AgentModeDaemon
-->
