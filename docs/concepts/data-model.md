# Data Model

agl-lite is a **data pipe** — it stores and delivers events without restricting schema. The trajectory is a flat sequence of events, not a tree of spans.

## Events

An event is the basic unit of data:

```python
class Event:
    event_type: str        # "model_request", "reward", or any user-defined string
    rollout_id: str
    attempt_id: str        # = K8s pod UID
    timestamp: float       # assigned by store at write time
    data: Dict             # event-type-specific payload
```

Events are identified by their **position in the list** (insertion order), not by a separate ID. Single-threaded asyncio guarantees temporal ordering.

## Reserved event types

agl-lite understands two event types. Everything else is opaque pass-through.

### `model_request`

Auto-captured by the [gateway](gateway.md) on every proxied LLM call. The agent is unaware.

```json
{
  "event_type": "model_request",
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

Key fields:
- `request.model` — what the agent sent (before routing)
- `server.model` — what the model server received (after routing)
- `server.version` — training step of the model, essential for importance sampling in RL

For streaming responses, the gateway buffers all SSE chunks and assembles the complete response. Token IDs (when `return_token_ids: true` is injected via gateway config) are gathered across chunks.

### `reward`

Reported by the algorithm, evaluator, or environment. This is the training signal.

```json
{
  "event_type": "reward",
  "data": {
    "value": 0.85,
    "message": "all tests passed"
  }
}
```

## User-defined event types

Any other `event_type` is stored and delivered as-is. agl-lite doesn't interpret the `data` payload.

```json
{"event_type": "tool_result", "data": {"tool_name": "execute_code", "output": "hello\n"}}
{"event_type": "agent_output", "data": {"answer": "42", "raw_response": "..."}}
{"event_type": "observation", "data": {"content": "Task description...", "source": "env"}}
```

## Trajectory

A trajectory is the complete sequence of events for one rollout attempt:

```
[0]  model_request   (agent calls LLM — auto-captured)
[1]  tool_result     (runner reports tool output — user-defined)
[2]  model_request   (agent sends tool result to LLM — auto-captured)
[3]  agent_output    (agent reports answer — user-defined)
[4]  reward          (algorithm scores: 0.85)
```

All event types are interleaved in a single ordered list, preserving temporal ordering.

### Event sources

| Source | Event types | Mechanism |
|--------|------------|-----------|
| **Gateway** | `model_request` | Auto-captured on every proxied LLM call |
| **Algorithm / Environment** | `reward`, user-defined | HTTP POST to event endpoint |
| **Agent** (optional) | User-defined | POST to `AGL_EVENT_URL` (never required) |

### Concurrent requests

Tool-use agents may fire multiple LLM calls in parallel. Each concurrent request completes independently. Insertion order for concurrent completions is arbitrary (whichever asyncio coroutine resumes first). This is storage ordering, not causal ordering — use `timestamp` for approximate causal information.

## Triplet format

For RL training, the algorithm needs token IDs, not full text. The `format=triplet` query parameter on the events API extracts just what tensor construction needs:

```
GET /api/events?rollout_id=R1&format=triplet
```

**Standard format:**
```json
{
  "event_type": "model_request",
  "data": {
    "request": {"model": "qwen-7b", "messages": [...]},
    "response": [
      {"choices": [...], "token_ids": [464], "prompt_token_ids": [1, 2, 3]},
      {"choices": [...], "token_ids": [1234]},
      {"choices": [...], "token_ids": [374, 220, 19]}
    ],
    "server": {"model": "qwen-7b", "version": 42}
  }
}
```

**Triplet format:**
```json
{
  "event_type": "model_request",
  "data": {
    "prompt_token_ids": [1, 2, 3],
    "response_token_ids": [464, 1234, 374, 220, 19],
    "server": {"model": "qwen-7b", "version": 42}
  }
}
```

The server gathers `token_ids` across SSE chunks, extracts `prompt_token_ids` from the first chunk, and merges `response_token_ids`. The [VERL integration](../user-guide/verl-integration.md) uses this format to build padded tensors for PPO/GRPO training.

## Data partitioning

Events are stored in a nested dict: `rollout_id → attempt_id → list[Event]`.

On retry, K8s creates a new pod (new UID), so each attempt writes to its own partition:

```
(R1, aaa) → [req1, req2, req3]              ← failed run
(R1, bbb) → [req1', req2', req3', req4']    ← successful run
```

The algorithm queries the succeeded attempt's trajectory by default. Failed attempt data remains available for debugging.
