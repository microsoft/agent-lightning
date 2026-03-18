# 003 — 🔴 LLM Backend Routing and Resource Mapping

## Problem

The gateway proxies LLM requests but the doc never specifies **where it forwards them**. During RL training, the model endpoint changes between iterations:

1. Algorithm trains new weights → pushes to inference server
2. Algorithm updates resources: `{model_endpoint: "http://vllm-v2:8000"}`
3. New rollouts should route to the new endpoint

The old gateway section mentioned "Resource awareness: Read current model endpoint from Store resources" but the unified API spec doesn't address this.

## Key questions

1. **Per-rollout routing**: Each rollout has a `resources_id`. Do different rollouts route to different backends? (Yes — iteration N rollouts use model vN, iteration N+1 rollouts use model vN+1.)

2. **Where is the backend URL stored?** In the resource snapshot under a well-known key? E.g., `resources["model_endpoint"] = "http://vllm:8000/v1"`?

3. **When is routing resolved?** At Job creation time (baked into env var)? At request time (gateway looks up per-request)?

## Options

**A. Static per-rollout (env var)** — Controller resolves `resources_id → model_endpoint` at Job creation time and injects as `LLM_BACKEND_URL` env var. Gateway reads it from a header or path parameter. Simple but can't handle mid-rollout model swaps.

**B. Gateway resolves at request time** — Gateway reads `resources_id` from the rollout record (cached), looks up the resource snapshot, extracts the backend URL. Supports dynamic routing but adds latency per request.

**C. Static gateway config** — One backend URL configured at gateway startup. All requests go to the same backend. Simplest but doesn't support model versioning during training.

**D. Resource-to-endpoint mapping in gateway config** — Gateway maintains a map of `resources_id → backend_url`, updated when Algorithm posts new resources. No per-request DB lookup.

## Recommendation

**A (static per-rollout) for MVP.** The controller already reads the rollout's `resources_id` to build the Job spec. It can resolve the backend URL at that point and inject it. The gateway then reads a well-known header or config to determine the upstream.

This works because in RL training, all rollouts in one batch share the same `resources_id` (same model version). The model changes between batches, not within a rollout.

## Changes needed

- Define well-known resource keys: `{"model_endpoint": "http://..."}` or similar
- Add `LLM_BACKEND_URL` env var to Job template
- Specify how gateway determines upstream (from request metadata or global config)
- Document the per-rollout-batch routing model
