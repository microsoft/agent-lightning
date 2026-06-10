# agl-lite

**Minimal agentic RL infrastructure — a streamlined [Agent Lightning](https://github.com/microsoft/agent-lightning).**

agl-lite provides transparent LLM request capture, a rollout data store, and Kubernetes-native agent execution — all behind a single HTTP endpoint. Agents use standard OpenAI SDKs with zero instrumentation; the gateway captures everything automatically.

## Architecture

<p align="center">
  <img src="docs/images/lite_arch.excalidraw.svg" alt="agl-lite architecture" width="800">
</p>

Three groups connected only by HTTP:

| Group | What it does | Managed by |
|-------|-------------|------------|
| **Compute Backend** | Model training (VERL/Megatron) + inference servers (vLLM) | User |
| **agl-lite Service** | Gateway (LLM proxy + event capture) + Data Store (rollouts, events, models) | agl-lite |
| **Agent Runner** | K8s controller + agent pods (any container, any language) | agl-lite + K8s |

## Key Design Choices

1. **Self-owned LLM gateway** — a purpose-built reverse proxy replaces litellm, capturing all request-response data transparently as it flows through
2. **Gateway-level data capture** — instead of instrumenting agents with OpenTelemetry, the gateway records request-response pairs during transfer — the proxy *is* the instrumentation
3. **K8s-native agent runner** — K8s Jobs as the execution unit, rollout-scoped attempt IDs for trace grouping, Job lifecycle as the retry mechanism — the store focuses purely on data, not execution control

## Quick Start

Install the project and run the remaining test suite:

```bash
git clone https://github.com/<org>/agl-lite && cd agl-lite
uv sync --extra dev
uv run pytest
```

See the [Getting Started guide](docs/get_started.md) for the full setup walkthrough.

## How Agents Work

Agents are plain containers that read env vars and call an OpenAI-compatible endpoint. No agl-lite import, no base class — any language, any framework:

```python
import os, json, openai

task = json.loads(os.environ["AGL_TASK_INPUT"])
client = openai.OpenAI()  # reads OPENAI_BASE_URL automatically

response = client.chat.completions.create(
    model="gpt-4.1",  # gateway routes to your vLLM
    messages=[{"role": "user", "content": task["prompt"]}],
)
# Gateway captures this call automatically — no instrumentation needed
```

The controller injects 4 env vars into every agent pod: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `AGL_TASK_INPUT`, and `AGL_EVENT_URL`. See [What Happens Next](docs/get_started.md#what-happens-next) for details.

## Documentation

| Section | Content |
|---------|---------|
| [Getting Started](docs/get_started.md) | Prerequisites, setup flow, first run |
| [Architecture](docs/design/0_architecture.md) | Full system design — data models, API spec, components |
| [K8s Controller](docs/design/1_k8s_controller.md) | Controller design and implementation details |
| [Dev Guidelines](docs/dev_guidelines.md) | Code conventions, tooling, concurrency model |

## Project Status

- **~3.5K lines** of source code, endpoint test coverage
- Gateway with route config, streaming proxy, and automatic event capture
- In-memory store (rollouts, events, models, resources)
- K8s controller with Job lifecycle management
- Python client library (`AglLiteClient`)
- VERL integration (`AglLiteRolloutBridge`) with triplet format

## License

TBD
