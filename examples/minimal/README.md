# Minimal Component Showcase

[![minimal CI status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unit.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unit.yml)

`examples/minimal` provides bite-sized programs that demonstrate how individual Agent-lightning building blocks behave in isolation.

Each module have been documented with its own CLI usage in the module-level docstring. Use this directory as a reference when wiring the same pieces into a larger system.

## What’s Included?

| Component | Demonstrated In | Highlights |
| --- | --- | --- |
| LightningStore + OTLP ingestion | `write_traces.py` | Shows how `OtelTracer` and `AgentOpsTracer` open rollouts, emit spans, and optionally forward them to a remote store client. |
| LLM proxying | `llm_proxy.py` | Guards either OpenAI or a local vLLM deployment with `LLMProxy`, proving how requests are routed through `/rollout/<id>/attempt/<id>` namespaces and captured in the store. |
| vLLM lifecycle | `vllm_server.py` | Minimal context manager that shells out to `vllm serve`, monitors readiness, and tears down the process safely. |

All runtime instructions (CLI arguments, required environment variables, etc.) are embedded directly in each script’s top-level docstring so the source stays self-documenting.

For full-fledged training workflows or multi-component experiments, browse the other subdirectories under `examples/`. This `minimal` folder deliberately keeps each demonstration focused on a single component so you can understand and test them independently.
