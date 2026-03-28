# Gateway

The gateway is a transparent LLM reverse proxy that sits between agents and inference servers. It does three things simultaneously: **routes requests**, **adjusts parameters**, and **captures data** — all without the agent knowing agl-lite exists.

## How it works

```
Agent ◄──chunk──chunk──chunk──[DONE]──◄ Gateway ◄──chunk──chunk──chunk──[DONE]──◄ vLLM
                                          │
                                     (buffer chunks in memory)
                                          │
                                     on stream complete:
                                          ▼
                                    write model_request event
                                    (in-process dict write, ~100ns)
```

The gateway is part of the agl-lite service process — no network hop between the proxy and the data store. Every LLM request goes through these steps:

1. **Parse context** from the URL path: `/rollout/{rid}/attempt/{aid}/v1/...`
2. **Validate** that `rollout_id` exists in the store (~100ns dict lookup). Returns 404 for orphan requests — avoids wasting GPU on invalid calls.
3. **Route** the request: look up the model name in the route config, rewrite `model` field, select a server from the pool (round-robin)
4. **Adjust parameters**: add/drop fields per the route config
5. **Forward** to the selected model server
6. **Stream back** to the agent in real-time (SSE chunks forwarded immediately)
7. **Capture**: on stream completion, write a `model_request` event with the full request, response, server metadata, and latency

## Model routing

The gateway maps agent-facing model names to actual model servers via a YAML config loaded at startup:

```yaml
# gateway-config.yaml
routes:
  - model_in: "gpt-4.1"           # what the agent sends
    model_out: "qwen-7b"          # rewritten to this
    params:
      add:
        temperature: 0.7
        max_tokens: 4096
        return_token_ids: true     # needed for RL training
      drop:
        - stream_options           # vLLM doesn't support this
        - logprobs

  - model_in: "*"                  # wildcard catch-all
    model_out: "*"                 # keep original model name
    params:
      add:
        return_token_ids: true
```

**Routing rules:**

- Routes are evaluated in list order — **first match wins**
- `model_in` with no matching route → passthrough (no rewrite, no param adjustment)
- `model_in: "*"` → catch-all, matches any model not matched by earlier rules
- `model_out: "*"` → keep the original model name, but still apply param adjustments
- `add` fields are merged into the request body (override if key exists)
- `drop` fields are removed from the request body

**Why this matters for RL:**

- **Model aliasing** — agents use familiar names (`gpt-4.1`), gateway routes to local vLLMs running fine-tuned checkpoints
- **Parameter enforcement** — inject `return_token_ids: true` for training without modifying agent code
- **Backend compatibility** — strip params that vLLM/TGI don't support

## Server selection

Model servers are registered dynamically via the API (`POST /api/models`). Each server has:

- `model` — grouping key (e.g., `"qwen-7b"`)
- `endpoint` — URL (e.g., `"http://vllm-0:8000/v1"`)
- `version` — training step number (for per-request version tracking)
- `token` — optional auth token for the model server

The gateway selects servers using **round-robin** within each model pool. When multiple servers serve the same model, requests are distributed evenly.

When no servers are registered for a model (e.g., during a [weight update](weight-updates.md)), the gateway returns **503 Service Unavailable** with a `Retry-After` header. Standard OpenAI SDKs auto-retry on 503 — the agent stays alive, no K8s Job retry is consumed.

## Streaming

The gateway handles both non-streaming and streaming requests:

**Non-streaming** (`stream: false`): Forward request, receive full JSON response, write event, return response.

**Streaming** (`stream: true`): The gateway **tees** the SSE stream — each chunk is forwarded to the agent immediately while simultaneously buffered. When the stream completes (`data: [DONE]`), the gateway assembles the full response and writes one `model_request` event.

**Edge cases:**

| Scenario | Behavior |
|----------|----------|
| Client disconnects mid-stream | Gateway continues reading from backend to capture complete data. Event status: `"client_disconnected"` |
| Backend error mid-stream | Event written with partial response. Event status: `"stream_error"` |
| Concurrent streams | Each completes independently. Insertion order = completion order (single-threaded asyncio) |

**Memory**: Each concurrent stream buffers one response. 128K-context response ≈ 500KB. 100 concurrent streams ≈ 50MB.

## What gets captured

Every proxied LLM call produces a `model_request` event:

```json
{
  "event_type": "model_request",
  "rollout_id": "abc123",
  "attempt_id": "pod-uid-xyz",
  "timestamp": 1711234567.89,
  "data": {
    "request": {
      "model": "gpt-4.1",
      "messages": [{"role": "user", "content": "..."}],
      "temperature": 0.7
    },
    "response": {
      "choices": [{"message": {"content": "..."}}],
      "usage": {"prompt_tokens": 100, "completion_tokens": 50}
    },
    "server": {
      "model": "qwen-7b",
      "endpoint": "http://vllm-0:8000/v1",
      "version": 42
    },
    "latency_ms": 1234.5,
    "status": "ok"
  }
}
```

The event records **both** the original request (what the agent sent, including original model name) and the server metadata (actual model, endpoint, training version). This is essential for RL: the `server.version` field tracks which policy generated each response.

## Performance

The gateway is a single Python async process. The hot path is I/O-bound — each request waits 2–20 seconds for LLM inference while the event loop serves other requests. Store writes are in-process dict operations (~100ns).

| Scale | Concurrent agents | Feasibility |
|-------|-------------------|-------------|
| Small | 50–100 | Trivial |
| Medium | 500–1,000 | Comfortable |
| Large | 2,000–5,000 | Fine with tuning |

The bottleneck is always the LLM servers, never the gateway.
