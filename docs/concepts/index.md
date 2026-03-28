# Concepts

This section explains the core ideas behind agl-lite. Read these before diving into the [User Guide](../user-guide/deployment.md) or [Reference](../reference/api.md).

## Architecture at a Glance

![agl-lite architecture](../images/lite_arch.excalidraw.svg)

agl-lite is organized into **three groups** connected only by HTTP — no co-location assumption:

| Group | Components | Managed by |
|-------|-----------|------------|
| **Compute Backend** (green) | Training engine (VERL/Megatron), inference servers (vLLM) | User — agl-lite does not deploy this |
| **agl-lite Service** (blue) | [Gateway](gateway.md) (LLM reverse proxy) + [Data Store](store.md) (rollouts, events, models) | agl-lite — single process, single endpoint |
| **Agent Runner** (red) | [K8s Controller](controller.md) + agent pods (any container, any language) | agl-lite controller + K8s |

The three groups can live in the same K8s cluster, separate clusters, or across cloud boundaries. The only requirement is HTTP reachability.

## Three Design Choices

Everything in agl-lite follows from three independent design choices:

### 1. Self-owned LLM gateway

A purpose-built reverse proxy replaces litellm. It handles model routing, parameter injection, and — critically — transparent request-response capture.

→ [Gateway deep dive](gateway.md)

### 2. Gateway-level data capture

Instead of instrumenting agents with OpenTelemetry spans, the gateway records request-response pairs as they flow through. The proxy *is* the instrumentation. Agents need zero awareness of agl-lite.

→ [Data Model](data-model.md) · [Agent Contract](agent-contract.md)

### 3. K8s-native agent runner

K8s Jobs are the execution unit. Pod UIDs serve as attempt IDs. The Job lifecycle handles retry and timeout. This makes the data store much simpler — it stores data, not execution state.

→ [Controller](controller.md)

## How the pieces connect

A typical RL iteration flows like this:

```
Algorithm                      agl-lite Service               K8s Cluster
─────────                      ────────────────               ───────────
1. POST /api/rollouts          Store: rollouts → queuing
   (enqueue tasks)

2.                             Controller: poll queuing        Create K8s Jobs
                                                              Agent pods start

3.                             Gateway: proxy LLM calls ←──── Agent calls OPENAI_BASE_URL
                               Store: auto-capture events     (standard OpenAI SDK)

4.                             Controller: watch Jobs          Pod exits 0 → Complete
                               Store: rollout → succeeded

5. GET /api/events             Store: return trajectory
   (fetch trajectories)        (sequence of events)

6. Train, update weights
   DELETE/POST /api/models     Gateway: weight update          Agents retry transparently
                               protocol (503 → re-register)
```

## What to read next

| Topic | What you'll learn |
|-------|-------------------|
| [Gateway](gateway.md) | How the LLM proxy works — routing, param injection, streaming, event capture |
| [Data Store](store.md) | Rollouts, events, resources, model servers — the data layer |
| [Controller](controller.md) | K8s reconciliation, Job lifecycle, retry, crash recovery |
| [Agent Contract](agent-contract.md) | What agents need to know (env vars only — no SDK) |
| [Data Model](data-model.md) | Event types, trajectory format, triplet extraction |
| [Weight Updates](weight-updates.md) | How training-inference coordination works |
