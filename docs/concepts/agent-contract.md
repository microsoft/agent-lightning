# Agent Contract

Agents in agl-lite are **plain containers**. No base class, no SDK, no agl-lite dependency. Any language, any framework — if it can read environment variables and call an HTTP endpoint, it's an agent.

## The contract: 4 environment variables

The [controller](controller.md) injects these into every agent pod:

| Env var | Purpose | Example |
|---------|---------|---------|
| `OPENAI_BASE_URL` | LLM endpoint (points to agl-lite gateway with rollout/attempt context) | `http://agl-lite:8080/rollout/R1/attempt/abc123/v1` |
| `OPENAI_API_KEY` | Auth key (from K8s Secret) | `ak_xxx...` |
| `AGL_TASK_INPUT` | Task payload (JSON string) | `{"prompt": "Write a sort function"}` |
| `AGL_EVENT_URL` | Optional: endpoint for posting custom events | `http://agl-lite:8080/rollout/R1/attempt/abc123/events` |

For Anthropic SDK compatibility, `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` are also set (same gateway URL, same key).

## Minimal Python agent

```python
import os, json, openai

# Read task
task = json.loads(os.environ["AGL_TASK_INPUT"])

# Call LLM — uses OPENAI_BASE_URL automatically
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1",  # gateway routes to your vLLM
    messages=[{"role": "user", "content": task["prompt"]}],
)

print(response.choices[0].message.content)
# Gateway captures this call automatically — no instrumentation needed
```

## Minimal JavaScript agent

```javascript
const task = JSON.parse(process.env.AGL_TASK_INPUT);

const resp = await fetch(
  `${process.env.OPENAI_BASE_URL}/chat/completions`,
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: "gpt-4.1",
      messages: [{ role: "user", content: task.prompt }],
    }),
  }
);
```

## What agents don't need to know

- **No agl-lite import** — agents don't depend on agl-lite at all
- **No event reporting** — the [gateway](gateway.md) captures LLM calls automatically
- **No retry logic** — K8s handles retry via `backoffLimit`
- **No status reporting** — exit code 0 = success, non-zero = failure (K8s convention)

## Optional: posting custom events

Agents *can* post events if they want to report additional data (e.g., parsed answers, intermediate results):

```python
import os, json, httpx

# Post a custom event
httpx.post(
    os.environ["AGL_EVENT_URL"],
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    json={
        "event_type": "agent_output",
        "data": {"answer": "42", "raw_response": "The answer is 42."},
    },
)
```

This is never required — it's for agents that want to enrich the trajectory with structured data beyond raw LLM calls.

## Container packaging

Agents are packaged as Docker images. The `job_template` in the [resources snapshot](store.md#resources) defines the pod spec. Per-rollout config can override image, command, and environment variables.

```dockerfile
# Example: minimal Python agent
FROM python:3.12-slim
RUN pip install openai
COPY agent.py /app/agent.py
CMD ["python", "/app/agent.py"]
```

The controller sets `image` and `command` from `rollout.config` — the Dockerfile provides defaults, but they can be overridden per experiment.

## How the URL encodes context

The `OPENAI_BASE_URL` embeds the rollout and attempt context:

```
http://agl-lite:8080/rollout/{rollout_id}/attempt/{pod_uid}/v1
```

When the agent calls `/v1/chat/completions`, the gateway extracts `rollout_id` and `attempt_id` from the path and tags the captured event accordingly. On retry (new pod, new UID), the URL changes automatically — each attempt writes to its own data partition.
