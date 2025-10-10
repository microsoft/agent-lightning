# The Bird's Eye View of Agent Lightning

This article summarizes how agent-lightning (as of v0.2) wires algorithms, runners, and stores together and shows where auxiliary components (tracer, adapters, proxies) plug into the loop. Each section provides a diagram for a different perspective of the system.

## Algorithm ↔ Runner ↔ Store data flow

At the very high level, Agent-lightning bundles the configured algorithm and runner and asks the "execution strategy" (details below) to execute them against the same `LightningStore` instance. The algorithm (in an automatic-interactive setting) typically enqueues work (new rollouts and resource updates) while the runner dequeues and executes those tasks, streaming traces and status updates back into the store. Once rollouts finish, the algorithm can query the completed data and apply adapters to convert the data for learning signals. The diagram below highlights the steady-state flow. We consider a very simple setup without any optional components and parallelism.

```mermaid
sequenceDiagram
    autonumber
    participant Algo as Algorithm
    participant Store as LightningStore
    participant Runner
    participant Agent

    loop Over the dataset
        Algo-->>Store: add_resources + enqueue_rollout
        Store-->>Runner: dequeue_rollout → AttemptedRollout
        Store-->>Runner: get_latest_resources
        Runner-->>Store: update_attempt("running", worker_id)
        Runner->>Agent: rollout + resources
        Agent->>Runner: reward / spans
        Runner-->>Store: add_span or add_otel_span
        Runner-->>Store: update_attempt("finished", status)
        Store-->>Algo: query_rollouts + spans
        Algo-->>Algo: Update resources (optional)
    end
```

*Solid lines are instantaneous calls, dashed lines are async / long-running.*

### Key Terms on the Arrows

We define the following terms which may be helpful for understanding the diagram above.

- **Resources:** A collection of assets to be tuned or trained. Agents perform rollouts against resources and collect span data. Algorithms use those data to update the resources. In case of RL training, the resources are the tunable model. In case of prompt tuning, the resources are the prompt templates.
- **Rollout:** A unit of work that an agent performs against a resource. A rollout (noun) can be incomplete, in which case it's also known as **"task"**, **"sample"** or **"job"** (these terms are used interchangeably). The agent executes its own defined workflow against the rollout -- this process is also called "rollout" (verb). After running is complete, the rollout (noun) is completed.
- **Attempt:** A single execution of a rollout. One rollout can have multiple attempts in case of failures or timeouts.
- **Span:** During the rollout, the agent can generate multiple spans (also known as "traces" or "events"). The recorded spans are collected in the store, which serves as the crucial part for understanding the agents' behavior and optimizing the agents.
- **Reward:** Reward is a special span that is semantically defined as a number judging the quality of the rollout for a period of time during the rollout.
- **Dataset:** A collection of incomplete rollouts (i.e., tasks) for the agent to play with. The three datasets (i.e., train, val, dev) serve as the initial input for the algorithm to enqueue the first batch of rollouts.

## Supporting Components in the Loop

Although the diagram above is simple and clear, it doesn't show many supporting components that Agent-lightning offers to make writing agents, runners, and algorithms easier. Here we introduce the key components and how they fit into the loop.

### Tracer

Tracer is a component that serves as a member variable of runners to record spans during the agents' rollout and send it to the store.

```mermaid
sequenceDiagram
    autonumber
    participant Store
    participant Runner
    participant Tracer
    participant Agent

    Note over Runner,Tracer: Runner manages tracer as member

    Tracer->>Agent: Apply instrumentation
    loop Until no more rollouts
        Store-->>Runner: dequeue_rollout → AttemptedRollout
        Store-->>Runner: get_latest_resources
        Runner->>Agent: training_rollout / validation_rollout
        loop For each finished span
            Agent-->>Tracer: openai.chat.completion invoked<br>agent.execute invoked<br>...
            Agent->>Tracer: emit intermediate reward
            Tracer-->>Store: add_otel_span(rollout_id, attempt_id, span)
        end
        Agent->>Runner: final reward + extra spans (if any)
        Runner-->>Store: add_span(rollout_id, attempt_id, span)
        Runner-->>Store: update_attempt(status)
    end
    Tracer->>Agent: Unapply instrumentation
```

The above diagram shows the overall data flow between store, tracer and agent. In realistic, it's a bit more complicated than that. The spans are not actually emitted actively by the agent, instead they are "caught" by the tracer by hooking and instrumenting key methods used in the agents. The tracers use a special callback (exporter) to monitor those events and logs to the store. Before the rollout starts, the runner enters a `trace_context` before invoking the agent, which wires the store identifiers into the tracer (illustrated in the following figure). Every span completion then streams back to the store through `LightningSpanProcessor.on_end`, so the agent's instrumentation lands in `add_otel_span`. If the agent's rollout method returns a numeric reward, the runner emits one more OpenTelemetry span before finalizing the attempt.

### Hooks

Hooks are user-defined callbacks to augment an existing runner's behavior. Currently, hooks can be called at the beginning and the end of trace and rollout.

```mermaid
sequenceDiagram
    autonumber
    participant Store
    participant Runner
    participant Tracer
    participant Hook
    participant Agent

    loop Until no more rollouts
        Store-->>Runner: dequeue_rollout → AttemptedRollout
        Store-->>Runner: get_latest_resources

        Runner->>Agent: training_rollout / validation_rollout
        Tracer->>Agent: enter_trace_context
        loop For each finished span
            Agent-->>Tracer: openai.chat.completion invoked<br>agent.execute invoked<br>...
            Agent->>Tracer: emit intermediate reward
            Tracer-->>Store: add_otel_span(rollout_id, attempt_id, span)
        end
        Agent->>Runner: final reward + extra spans (if any)
        Tracer->>Agent: exit_trace_context
        Runner-->>Store: add_span(rollout_id, attempt_id, span)
        Runner-->>Store: update_attempt(status)
    end
```

### Adapter

The adapter is the algorithm's bridge between raw traces and learning signals. Users can configure an adapter in the algorithm before `algorithm.run` starts, so that the algorithm instance can later call `adapter.adapt(...)` on spans fetched from the store to conveniently converts the spans into a format suitable for learning.

The runner streams spans into the store as it executes rollouts, and algorithms query those spans to construct data needed for learning. For example, the VERL algorithm collects spans for each completed rollout, converts them with `TraceTripletAdapter` (by default), which implements `adapt` by traversing OpenTelemetry spans, aligning prompts, responses, and reward spans into `Triplet` records (details below) that downstream RL fine-tuning code can consume. The figure below summarizes the relationship.

```mermaid
flowchart LR
    Runner -- (1) add_otel_span --> Store
    Store -- (2) query_spans --> Algorithm
    Algorithm -- (3) spans --> Adapter
    Adapter -- (4) transformed data --> Algorithm
```

### LLM Proxy

The LLM proxy is an optional bridge between the runner's LLM calls and the algorithm's resource management. `LLMProxy` is associated with a store and it's handed over to the algorithm. LLM Proxy is added as a special resource that redirects to LLM endpoint. This endpoint can be:

1. **Endpoint served by the algorithm:** If the algorithm is internally updating the LLM weights (e.g., RL), it can launch an LLM inference engine (i.e., a model server) and register the endpoint URL with the proxy. The proxy then forwards all LLM calls to that endpoint.
2. **Third-party LLM endpoint:** If the algorithm is not updating the LLM weights (e.g., prompt tuning), it can register a third-party LLM endpoint into the proxy.

During rollouts, the runner invokes the proxy's HTTP endpoint instead of calling a model backend directly. The proxy augments each request with rollout/attempt metadata, tracks which rollout is initiating the current request, and then records OpenTelemetry spans via `LightningSpanExporter`.

Functionally, the proxy acts as a "shield" in front of LLM calls. It's also a convenient way to integrate diverse LLM backends (e.g., OpenAI, Azure, Anthropic, local models) without changing the agent code. It can also be used to support diverse LLM clients (e.g., Anthropic API), add retry logic, rate limiting and caching.

```mermaid
sequenceDiagram
    autonumber
    participant Algo as Algorithm
    participant LLMProxy as LLM Proxy
    participant Store
    participant Runner
    participant Agent

    Note over Algo,LLMProxy: Algorithm manages LLMProxy as member

    loop Over the Dataset
        Algo->>Algo: Launch LLM Inference Engine<br>(optional)
        Algo->>LLMProxy: Register Inference Engine<br>(optional)
        Algo-->>Store: enqueue_rollout
        LLMProxy->>Store: Proxy URL added as Resource
        Store-->>Runner: dequeue_rollout → AttemptedRollout
        Store-->>Runner: get_latest_resources
        Runner->>Agent: rollout + resources<br>(LLM Proxy URL as resource)
        loop Defined by Agent
            Agent-->>LLMProxy: LLM calls
            LLMProxy-->>Store: add_span or add_otel_span
            LLMProxy-->>Agent: LLM responses
            Agent-->>Runner: rewards
            Runner-->>Store: add_span or add_otel_span
        end
        Runner-->>Store: update_attempt("finished", status)
        Store-->>Algo: query_rollouts + spans
        Algo-->>Algo: Update LLM Weights<br>(optional)
    end
```

In this diagram, the store receives spans from both the proxy and the runner. We will see a problem later with parallelism where the proxy and runner are in different machines.

### Trainer

Trainer is a high-level interface for users to initiate the whole learning process. Trainer manages almost all the components mentioned above, including algorithm, store, runner, agent, tracer, adapter and LLM proxy. All the components can be configured via the trainer interface and lives as long as the trainer lives.

Some components are injected into other components as member variables or weak references. For example, the store is injected into the algorithm and runner. The tracer and agent are injected into the runner. The adapter and LLM proxy are injected into the algorithm. The store is further injected into the tracer, adapter and LLM proxy by the runner and algorithm respectively.

```mermaid
flowchart LR
    %% Ownership (constructed/held by Trainer)
    Trainer --has--> Tracer["Tracer (AgentOpsTracer*)"]
    Trainer --has--> Adapter["Adapter (TraceTripletAdapter*)"]
    Trainer -.has.-> Store["LightningStore (InMemory* default)"]
    Trainer -.has.-> Runner["Runner (AgentRunner* default)"]
    Trainer -.has.-> Algorithm["Algorithm (no default)"]
    Trainer -.has.-> LLMProxy["LLM Proxy (no default)"]

    %% Trainer passes references
    Algorithm -.uses.-> Trainer
    Algorithm -.injects.-> Store
    Algorithm -.injects.-> Adapter
    Algorithm -.injects.-> LLMProxy

    Runner -.injects.-> Agent["Agent (LitAgent)"]
    Runner -.injects.-> Store
    Runner -.has.-> Tracer

    %% Cross-wiring done at runtime
    LLMProxy -.set_store().-> Store

    %% Agent awareness
    Agent -.set_trainer().-> Trainer
```

## Inside an RL Algorithm (VERL Example)

TODO: talk about LLM inference engine (and the proxy shield), resource (LLM endpoint), triplets (adapter output).

## Execution Strategies and Parallelism

A key observation from the diagram above is that there is absolutely no communication between (1) runner and agents and (2) algorithm. The only overlap of them is the store.

TODO: talk about bundles: algorithm bundle and runner bundle. Runner can be forked and parallelized. Bundles communicate with each other through the store.

### Shared-memory Strategy

TODO: sharing the same memory between all bundles. The store needs to be thread safe. Figure.

### Client-server Strategy

TODO: the store is splitted into a server handle and a client handle. The algorithm bundle holds the source of truth of the store (i.e., the server). The runner bundle uses the client handle to communicate with the server. Inside the algorithm bundle, if there are multiple processes (e.g., llm proxy needs a standalone process), the subprocesses also hold the client bundle.

## Online/Continuous Learning

TODO: using an continuous `algorithm.run` with continuous `runner.step()`. Explain where resources and samples come from.
