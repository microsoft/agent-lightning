# Working with Traces

Tracer is the secret sauce that enables Agent-lightning to train ANY AI agents. Tracing was originally an observability concept within LLMOps before Agent-lightning came out, which was largely used to monitor the behavior of LLM Agents. Agent-lightning builds on top of that concept and extends tracing to be a first-class citizen within the agent training loop. Traces are used not only for observability, but also for learning signals (e.g., rewards) that drive the training of agents.

In Agent-lightning, traces are stored as [`Span`][agentlightning.Span] objects within a [`LightningStore`][agentlightning.LightningStore]. Each span represents a single operation within the agent's workflow, such as an LLM call, a tool invocation, or an agent decision. Spans are organized into a tree structure, where parent spans represent higher-level operations and child spans represent lower-level operations. We will discuss how to write and read [`Span`][agentlightning.Span] objects in the following sections.

## Writing Spans

Most [`Runner`][agentlightning.Runner]s use a [`Tracer`][agentlightning.Tracer] instance to instrument the agent's workflow and capture spans. That's why Agent-lightning requires almost zero modifications to existing agent codebases to train agents.

There are typically two built-in [`Tracer`][agentlightning.Tracer] implementations you can use in the [`Runner`][agentlightning.Runner]. The two implementations are both based on [OpenTelemetry](https://opentelemetry.io/) standard. It's also important to note that we only use [Traces](https://opentelemetry.io/docs/concepts/signals/traces/) from OpenTelemetry, and we don't use other signals like Metrics or Logs.

### AgentOps Tracer

[`AgentOpsTracer`][agentlightning.AgentOpsTracer] is the default [`Tracer`][agentlightning.Tracer] used by Agent-lightning. It uses [AgentOps SDK](https://www.agentops.ai/) to instrument your agent's source code and send them into a [OpenTelemetry TracerProvider](https://opentelemetry.io/docs/specs/otel/trace/api/). Within [`AgentOpsTracer`][agentlightning.AgentOpsTracer] implementation, we purposely refrain AgentOps SDK connecting to their services (so as to avoid overwhelming their services and your bank account with too many requests). Instead, we use [`SpanProcessor`](https://opentelemetry-python.readthedocs.io/en/latest/sdk/trace.html) to capture and store OpenTelemetry spans produced by AgentOps SDK locally and send them to [`LightningStore`][agentlightning.LightningStore] for persistence.

Within [AgentOps documentation](https://docs.agentops.ai/v2/introduction), they listed the frameworks they have integrated with. Since we use AgentOps SDK under the hood, Agent-lightning can automatically trace agents built with those frameworks.

The OpenTelemetry spans from AgentOps mostly comply with [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/concepts/semantic-conventions/). However, we notice several issues with their current implementation:

1. Some OpenAI LLM calls return extra information in the response (e.g., [Token IDs](../deep-dive/serving-llm.md)), but AgentOps SDK does not capture them in the spans.
2. AgentOps does a good job at grouping the spans into a hierarchy, but it's not always reliable as spans may produced by different instrumentation methods. For example, there might be no parent-child relationship between an OpenAI call span and a LangChain agent step span.
3. Some important operations might be missed from the spans. For example, with newer versions of LangGraph, AgentOps SDK fails to capture spans that contains the information of graph node names.

Therefore, we added another instrumentation layer inside [`AgentOpsTracer`][agentlightning.AgentOpsTracer] to address the above issues. We expect there can be still more issues with other frameworks, and we suggest users debug with [Hooks](./debug.md) or implement a customized tracer if necessary.

### OpenTelemetry Tracer

### LLM Proxy

### Custom Tracer

### Distributed Tracing

## Reading Traces

**Why traces are difficult to read?**

### Adapter

### Reading Rewards
