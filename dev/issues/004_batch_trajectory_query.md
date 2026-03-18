# 004 — 🔴 Batch Trajectory Query for Training

## Problem

The Algorithm fetches trajectories for an entire training batch. A typical RL training step processes 256–4096 rollouts. Currently `GET /api/events` takes one `rollout_id`. That means 256–4096 sequential HTTP calls per training step — a major bottleneck.

## Why this is architectural

- Training throughput is directly gated by trajectory fetch time
- Sequential HTTP calls add latency proportional to batch size
- The original Agent Lightning had `query_rollouts` returning multiple rollouts with their data, and `query_spans` per rollout — same N+1 problem, but in-process (shared memory) it was fast

## Options

**A. Batch endpoint** — `POST /api/events/batch` with body `{rollout_ids: [...], attempt_id_resolution: "auto"}`. Returns `{rollout_id: [events]}` map. Single HTTP call.

**B. Streaming endpoint** — `GET /api/events/stream?rollout_ids=R1,R2,...` returns NDJSON stream. Good for very large batches.

**C. Client-side parallelism** — Algorithm fires N concurrent `GET /api/events` calls. Simple but puts load on the service and doesn't reduce total bytes transferred.

**D. Rollout-embedded events** — `GET /api/rollouts?status_in=succeeded&include_events=true` returns rollouts with events inline. One call but potentially huge response.

## Recommendation

**A (batch endpoint) as primary.** Simple, single request-response, easy to implement. Add response streaming for very large batches later if needed.

```
POST /api/events/batch
{
  "rollout_ids": ["R1", "R2", ...],
  "event_type": "model_request",     // optional filter
  "attempt_id": "auto"               // auto-resolve per rollout (succeeded or latest)
}

Response:
{
  "R1": [event, event, ...],
  "R2": [event, event, ...],
  ...
}
```

## Changes needed

- Add `POST /api/events/batch` to Section 3.4 path layout and Store paths
- Add to Section 4.2 comparison table
- Update Store API pseudocode
