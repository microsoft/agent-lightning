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

### 1. What is agl-lite — three design choices

### 2. Architecture overview

### 3. Deep dive — Gateway

### 4. Deep dive — Store + K8s Controller

### 5. VERL integration

### 6. Deployment & demo

### 7. Status & next steps

---
layout: section
---

# Part 1
## What is agl-lite

*Three design choices*

---

# Three Design Choices

*Self-owned gateway · Gateway-level data capture · K8s-native runner*

**1. Self-owned request gateway** — a purpose-built LLM reverse proxy that captures all request-response data transparently as it flows through

**2. Gateway-level data capture** — instead of instrumenting agents, the gateway records request-response pairs during transfer and stores them as events — the proxy *is* the instrumentation

**3. Kubernetes-native agent runner** — K8s Jobs as the execution unit, pod UIDs as attempt IDs, Job lifecycle as the retry mechanism

<div class="mt-6 p-3 bg-blue-50 rounded border border-blue-200">

💡 **A natural consequence of (3):** since K8s owns retry, timeout, and scheduling, the data store and rollout state machine become much simpler — the store focuses purely on data, not execution control.

</div>

<!--
These three choices drive everything in agl-lite.
They're independent — each can be understood on its own.
The store simplification falls out of choice 3.
-->

---

# Interface Summary

<img src="/lite-api.jpg" alt="summary of agl-lite api" class="shadow rounded">

---
layout: section
---

# Part 2
## Architecture Overview

---

# High-Level Architecture

<img src="/lite_arch.excalidraw.svg" alt="Architecture diagram" class="w-full border rounded">

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

*Transparent proxy, agent contract, model routing, weight updates*

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
- **Three reserved event types:**
  - `model_request` — auto-captured by gateway
  - `agent_output` — reported by the agent (no enforced schema)
  - `reward` — training signal from the algorithm
- Everything else: opaque pass-through — users define freely
- Insertion order = temporal order (single-threaded asyncio)

</div>

<div class="compact-code-block mt-2">

```python
class Event:
    event_type: str     # reserved or user-defined
    rollout_id: str
    attempt_id: str     # = K8s pod UID
    timestamp: float
    data: Dict          # type-specific payload
```

</div>

</div>
<div>

#### Example trajectory

<div class="compact-code-block">

```
[0]  model_request  (auto — gateway captured)
[1]  agent_output   (agent reports its answer)
[2]  reward         (algorithm scores: 0.85)
```

</div>

A multi-turn agent might produce:

<div class="compact-code-block">

```
[0]  model_request  (turn 1 — auto)
[1]  tool_result    (user-defined — runner)
[2]  model_request  (turn 2 — auto)
[3]  agent_output   (agent's final answer)
[4]  reward         (score: 0.85)
```

</div>

<div class="mt-2 p-3 bg-blue-50 rounded border border-blue-200">

💡 **agl-lite is a data pipe.** It stores and delivers events without restricting schema. Users extend freely with their own event types.

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

The controller bridges the store and K8s — the **only component that writes rollout status transitions**.

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
    value: "$(AGL_BASE_URL)/rollout/$(ROLLOUT_ID)
           /attempt/$(POD_UID)/v1"
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: agl-lite-keys
        key: AGL_KEY
  - name: AGL_TASK_INPUT
    value: '{"prompt": "Solve: 2+2=?"}'
  - name: AGL_EVENT_URL
    value: "$(AGL_BASE_URL)/rollout/$(ROLLOUT_ID)
           /attempt/$(POD_UID)/events"
```

</div>

<div class="compact-text mt-2">

On retry, K8s creates a new pod → new `POD_UID` → URLs automatically point to a fresh attempt partition.

</div>

</div>
</div>

---

# Job Template: Examples

The `job_template` is a raw K8s pod spec — maintained per experiment, stored as an immutable resource snapshot.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### Simple: Math PoC (single container)

<div class="compact-code-block">

```yaml
# job-template.yaml
containers:
  - name: agent
    image: math-agent:dev
    command: ["python", "/app/qa_agent.py"]
    imagePullPolicy: Never
    resources:
      requests:
        cpu: "100m"
        memory: "128Mi"
```

</div>

<div class="mt-4 px-3 bg-blue-50 rounded border border-blue-200">

💡 Any valid K8s pod spec fields work — nodeSelector, tolerations, volumes, init containers. The store doesn't validate it — K8s does at Job creation.

</div>

</div>
<div>

#### Multi-container: Coding tasks 
<div class="compact-code-block">

```yaml
# job-template.yaml
containers:
  - name: agent
    imagePullPolicy: Never
    resources:
      requests: {cpu: "1", memory: "2Gi"}
    volumeMounts:
      - name: workspace
        mountPath: /workspace
  - name: scorer
    image: scorer:latest
    command: ["python", "run_tests.py"]
    volumeMounts:
      - name: workspace
        mountPath: /workspace
volumes:
  - name: workspace
    emptyDir: {}
```

</div>

</div>
</div>

---

# Job Template: Merge Flow

The controller merges `job_template` with per-rollout config at Job creation time.

<div class="grid grid-cols-3 gap-6 mt-2">
<div class="col-span-2">

#### How the merge works

<div class="compact-text">

1. **`job_template`** provides the base pod spec (from resources snapshot)
2. **Controller injects** into the container named `agent`: env vars (`OPENAI_BASE_URL`, `AGL_TASK_INPUT`, etc.)
3. **`rollout.config`** can override per-rollout: image, command, extra env vars
4. **`rollout.config.overrides`** can patch other containers by name (e.g., swap scorer image per task) (*Backlog*)

</div>

</div>
<div>

<div class="compact-code-block">

```
job_template (raw pod spec, from YAML)
  │
  ├── controller injects into "agent" container:
  │     ├── OPENAI_BASE_URL, OPENAI_API_KEY
  │     ├── AGL_TASK_INPUT, AGL_EVENT_URL
  │     └── extra env from rollout.config
  │
  ├── rollout.config.overrides (if any):
  │     └── name-matched container merge
  │
  └── wrap in Job metadata:
        ├── name: agl-rollout-{rid}
        ├── backoffLimit (retries)
        └── activeDeadlineSeconds (timeout)
```

</div>

</div>
</div>

<div class="mt-4 p-3 bg-green-50 rounded border border-green-200">

✅ **Researcher** can focus on what changes per rollout (image, command, env) in `rollout.config`. **Controller** handles the merge and injects the rest. Separation of concerns.

</div>

---
layout: section
---

# Part 5
## VERL Integration

*AglLiteRolloutBridge, trainer code, triplet format*

---

# AglLiteRolloutBridge: The Trainer Bridge

`AglLiteRolloutBridge` provides the same trainer-facing methods as `AgentModeDaemon` — the trainer barely changes.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

#### New code (187 lines)

<div class="compact-text">

Store interaction via `AglLiteClient`:

- `set_up_data_and_server` → register models + enqueue rollouts
- `run_until_all_finished` → poll rollout status until `succeeded`
- `_async_fetch_rollout_result` → fetch events with `format=triplet`
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
rollout_bridge = AglLiteRolloutBridge(
    agl_base_url="http://agl-lite:8080",  # single HTTP endpoint
    agl_key="agl_xxx...",                  # shared API key
    train_rollout_n=4, tokenizer=tokenizer, mini_batch_size=64, pad_token_id=0,
)

# Training loop
for batch in dataset:
    rollout_bridge.set_up_data_and_server(batch, vllm_addresses)
    rollout_bridge.run_until_all_finished()
    data_proto = rollout_bridge.get_train_data_batch()
    # → DataProto with padded tensors, ready for PPO / GRPO / REINFORCE
    rollout_bridge.clear_data_and_server()
```

</div>

<div class="mt-4 p-3 bg-blue-50 rounded border border-blue-200">

💡 **Same four methods, same `DataProto` output.** The rollout bridge constructor is the only difference — an HTTP URL instead of store/proxy/adapter objects.

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

⚡ Server gathers `token_ids` across SSE chunks, extracts `prompt_token_ids` from first chunk, merges `response_token_ids` — the rollout bridge receives ready-to-use IDs.

</div>
</div>

---
layout: section
---

# Part 6
## Deployment & Demo

*Math PoC on minikube with real vLLM*

---

# Math PoC: GSM8K with Qwen2.5-1.5B-Instruct

The `examples/math-poc/` demonstrates: enqueue tasks → agents solve via vLLM → algorithm scores.

<div class="compact-code-block">

```
┌─ minikube ────────────────────────────────────────┐
│                                                   │
│  agl-controller    (Deployment)  ← creates Jobs   │
│                                                   │
│  agent pods        (Jobs)   ─── OPENAI_BASE_URL ──┼──┐
│                                                   │  │
└───────────────────────────────────────────────────┘  │
                                                       │ host.minikube.internal:8080
                                                       ▼
┌─ host ────────────────────────────────────────────┐
│  agl-lite serve    (process :8080)                │
│    └─ gateway ──→ vLLM (localhost:8010)           │
│                                                   │
│  rl_loop.py        ← algorithm (localhost:8080)   │
│  vLLM (Docker, GPU 0)  ← Qwen2.5-1.5B-Instruct    │
└───────────────────────────────────────────────────┘
```

</div>

<div class="compact-text mt-2">

- **Agent pods** in minikube reach agl-lite on host via `host.minikube.internal:8080`
- **Gateway → vLLM** is `localhost:8010` — no cross-network hop
- Gateway config injects `return_token_ids: true` into all requests
- Each rollout produces 3 events: `model_request` (auto) → `agent_output` (agent) → `reward` (algorithm)

</div>

---

# Math PoC: Results

<div class="grid grid-cols-2 gap-6">
<div>

#### Event flow per rollout

<div class="compact-table">

| Event | Source | Content |
|-------|--------|---------|
| `model_request` | gateway (auto) | request, SSE response, token IDs, server version |
| `agent_output` | agent pod | parsed answer, raw LLM response |
| `reward` | algorithm | score, ground truth, comparison |

</div>
</div>
<div>

<div class="compact-code-block-xs">

```
ITERATION 1 (model version=1)
  ── First model_request event (sample) ──
    server:   model=Qwen/Qwen2.5-1.5B-Instruct, version=1
    request.model:    Qwen/Qwen2.5-1.5B-Instruct
    request.stream:   True
    request.return_token_ids: True
    request.messages: 2 messages
    response: 362 SSE chunks (streaming)
      content (1285 chars):
        To determine how much Janet makes every day at the farmers' market, we need to follow these steps:
        1. Calculate the total number of eggs laid by the ducks each day.
        2. Determine how many eggs are eaten for breakfast.
        3. Subtract the number of eggs eaten from the total number of eggs laid to find ou
        ...
      prompt_token_ids: 102 tokens
      response token_ids: 361 tokens
        first 10: [151644, 8948, 198, 2610, 2299, 264, 10950, 6888, 17847, 13]
        first 10: [1249, 8253, 1246, 1753, 53665, 3643, 1449, 1899, 518, 279]
  ── end sample ──
    <rollout-id>: [✓] answer='18', gt='18' → correct
    <rollout-id>: [✓] answer='3', gt='3' → correct
    <rollout-id>: [✗] answer='0', gt='70000' → wrong: 0.0 != 70000.0
    <rollout-id>: [✓] answer='540', gt='540' → correct
    <rollout-id>: [✓] answer='20', gt='20' → correct
```

</div>
</div>
</div>

---
layout: section
---

# Part 7
## Status & Next Steps

---

# Current Status & Next Steps

<div class="grid grid-cols-2 gap-6">
<div>

#### What's built ✅

<div class="compact-text">

- Gateway with route config + real streaming
- In-memory store (rollouts, events, models, resources)
- K8s controller (kr8s-based, Job lifecycle)
- `AglLiteClient` library + `agl-client` CLI
- `AglLiteRolloutBridge` for VERL integration
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
