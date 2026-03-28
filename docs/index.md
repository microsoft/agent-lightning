# agl-lite

**Minimal agentic RL infrastructure — a streamlined [Agent Lightning](https://github.com/microsoft/agent-lightning).**

agl-lite provides transparent LLM request capture, a rollout data store, and Kubernetes-native agent execution — all behind a single HTTP endpoint. Agents use standard OpenAI SDKs with zero instrumentation; the gateway captures everything automatically.

<p align="center">
  <img src="images/lite_arch.excalidraw.svg" alt="agl-lite architecture" width="800">
</p>

## Three Design Choices

1. **Self-owned LLM gateway** — a purpose-built reverse proxy captures all request-response data transparently as it flows through
2. **Gateway-level data capture** — instead of instrumenting agents with OpenTelemetry, the gateway records request-response pairs during transfer — the proxy *is* the instrumentation
3. **K8s-native agent runner** — K8s Jobs as the execution unit, pod UIDs as attempt IDs, Job lifecycle as the retry mechanism — the store focuses purely on data, not execution control

## Get started

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Run GSM8K math problems with vLLM in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Concepts**

    ---

    Understand the gateway, store, controller, and agent contract

    [:octicons-arrow-right-24: Concepts](concepts/index.md)

-   :material-cog:{ .lg .middle } **User Guide**

    ---

    Deploy, configure, write agents, and run experiments

    [:octicons-arrow-right-24: User Guide](user-guide/deployment.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    REST API, CLI, schemas, and client library

    [:octicons-arrow-right-24: Reference](reference/api.md)

</div>

## How agents work

Agents are plain containers that read env vars and call an OpenAI-compatible endpoint — no agl-lite import, any language:

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

## Examples

| Example | Description |
|---------|-------------|
| [Math PoC](examples/math-poc.md) | GSM8K problems with Qwen2.5-1.5B-Instruct (vLLM) |
| [SWE-bench](examples/swe-bench.md) | Coding tasks with Claude Code agent |
