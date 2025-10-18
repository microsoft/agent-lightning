# Scaling out Algorithms and Rollouts

Agent-lightning splits training into an **algorithm bundle** and a **runner bundle** that exchange work through the [`LightningStore`][agentlightning.LightningStore]. This tutorial shows how to increase rollout throughput, place bundles across processes or machines, and keep the algorithm side scalable with external frameworks.

## Parallelizing Rollouts with [`Trainer`][agentlightning.Trainer]

[`Trainer`][agentlightning.Trainer] is the quickest way to dial up rollout parallelism. When you call [`Trainer.fit`][agentlightning.Trainer.fit]:

- The algorithm enqueues rollouts into the store.
- Each runner subprocess dequeues work, executes your [`LitAgent`][agentlightning.LitAgent], emits spans through the tracer, and reports rewards.
- The algorithm consumes spans via its [`Adapter`][agentlightning.Adapter], updates resources, and schedules the next batch.

Increase throughput by setting `n_runners` when constructing the trainer:

```python
import agentlightning as agl
from calc_agent import calc_agent
from datasets import Dataset as HFDataset

train_dataset = HFDataset.from_parquet("data/train.parquet").to_list()
val_dataset = HFDataset.from_parquet("data/test.parquet").to_list()

algorithm = agl.VERL(verl_config)

trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=8,  # launch eight rollout workers
    tracer=agl.OtelTracer(),
    adapter=agl.LlmProxyTraceToTriplet(),
)

trainer.fit(calc_agent, train_dataset=train_dataset, val_dataset=val_dataset)
```

A few practical guidelines:

- Use the `max_rollouts` argument to cap work per runner when smoke testing.
- Pass an explicit [`LightningStore`][agentlightning.LightningStore] if you need durability or want to share the queue with other processes.
- Seed `initial_resources` so every runner pulls the same prompt/model bundle on startup.

!!! tip
    Before scaling out, run [`Trainer.dev()`][agentlightning.Trainer.dev] with `n_runners=1` to verify the rollout logic and spans without burning GPU hours.

## Client-server Architecture

The default [`ClientServerExecutionStrategy`][agentlightning.ClientServerExecutionStrategy] starts a [`LightningStoreServer`][agentlightning.LightningStoreServer] alongside the algorithm and spawns runner processes that talk to it through [`LightningStoreClient`][agentlightning.LightningStoreClient]. All runners share the HTTP endpoint, so the queue and spans stay consistent across processes or machines.

### Run everything from one entry point

If you simply instantiate `Trainer` (as above), it will:

1. Create the store (by default [`InMemoryLightningStore`][agentlightning.InMemoryLightningStore]).
2. Wrap it in `LightningStoreServer`.
3. Fork `n_runners` subprocesses that connect through `LightningStoreClient`.

You can override server placement or ports through constructor arguments or environment variables:

```python
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=4,
    strategy={
        "type": "cs",
        "server_host": "0.0.0.0",
        "server_port": 4747,
        "managed_store": True,
    },
)
```

Set `AGL_SERVER_HOST` and `AGL_SERVER_PORT` if you prefer environment-based configuration.

### Split algorithm and runners across processes

For advanced debugging or remote execution, reuse the same training script but select which bundle to run:

```bash
# Start a long-lived store (optionally on another machine)
agl store --port 4747

# Algorithm side: runs the algorithm bundle and the store server
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=algorithm \
python train_calc_agent.py --external-store-address http://localhost:4747

# Runner side: connects only as runners
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=runner \
python train_calc_agent.py --external-store-address http://localhost:4747 --n-runners 6
```

Setting `AGL_MANAGED_STORE=0` skips the automatic wrapping so the script respects the external store. Inside the script, pass a [`LightningStoreClient`][agentlightning.LightningStoreClient] when `--external-store-address` is provided:

```python
store = agl.LightningStoreClient(external_store_address) if external_store_address else None
trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=store)
```

You can run multiple runner commands (even on different hosts) against the same endpoint to grow capacity.

!!! note
    The HTTP API is considered internal; interact with it through `LightningStoreClient` instead of hand-rolled requests so retries, health checks, and serialization stay consistent.

## Execution Strategy

Execution strategies control where bundles live, how the store is wrapped, and how shutdown is coordinated. Pass a strategy instance or a registry entry to [`Trainer(strategy=...)`][agentlightning.Trainer]:

```python
trainer = agl.Trainer(
    algorithm=algorithm,
    strategy={"type": "shm", "n_runners": 2, "main_thread": "algorithm"},
)
```

If you omit the strategy, the trainer defaults to `ClientServerExecutionStrategy(n_runners=n_runners)`.

### Shared-memory Strategy

[`SharedMemoryExecutionStrategy`][agentlightning.SharedMemoryExecutionStrategy] keeps everything inside one process. The algorithm runs on the main thread while each runner lives on a Python thread guarded by [`LightningStoreThreaded`][agentlightning.LightningStoreThreaded].

Use it when you want:

- Easier debugging with shared breakpoints and no serialization overhead.
- Minimal startup time for unit tests.

Sample configuration:

```python
trainer = agl.Trainer(
    algorithm=algorithm,
    strategy={
        "type": "shm",
        "n_runners": 3,
        "main_thread": "algorithm",  # keep the algorithm on the main thread
        "managed_store": True,
    },
)
```

When `main_thread="runner"`, the runner occupies the main thread and `n_runners` must be `1`. The strategy respects `AGL_MANAGED_STORE`; set it to `0` to opt out of the `LightningStoreThreaded` wrapper.

### Client-server Strategy

Stick with [`ClientServerExecutionStrategy`][agentlightning.ClientServerExecutionStrategy] when you need process isolation or want to fan out across machines. Useful knobs:

- `role`: `"algorithm"`, `"runner"`, or `"both"` (defaults to the `AGL_CURRENT_ROLE` env var or `"both"`). Combining it with CLI flags lets you reuse the same entry point for each bundle.
- `main_process`: choose which bundle stays in the original process when `role="both"`.
- `managed_store=False`: supply your own `LightningStoreServer` or a fully managed external store.

```python
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=4,
    strategy={
        "type": "cs",
        "role": "both",
        "main_process": "algorithm",
        "graceful_timeout": 10.0,
    },
)
```

The strategy automatically escalates shutdown (cooperative stop → `SIGINT` → `terminate()` → `kill()`) so long-running runners do not linger.

## Parallelizing Algorithms

Runner parallelism scales rollout throughput, but the algorithm loop remains single-process inside the execution strategy. To scale model updates:

- Choose algorithms that already integrate with distributed training frameworks. [`VERL`][agentlightning.algorithm.verl.VERL], for example, launches FSDP and vLLM components internally while the trainer keeps the store and runners fed.
- Externalize heavy services (LLM proxies, evaluators) from within the algorithm process. They can still report spans via [`LightningStoreClient`][agentlightning.LightningStoreClient].
- Use the role-based launch pattern above to place the algorithm on a dedicated machine with more GPU memory while runners stay closer to data sources.
- Keep the store authoritative. Whether you enqueue rollouts in batches or stream them, the algorithm can poll [`LightningStore.wait_for_rollouts`][agentlightning.LightningStore.wait_for_rollouts] or [`LightningStore.query_spans`][agentlightning.LightningStore.query_spans] without competing with runner threads.

!!! tip
    `docs/deep-dive/birds-eye-view.md` illustrates how adapters, proxies, and stores interact when the algorithm spawns additional workers. Use that diagram as a checklist when introducing new distributed components.

With these patterns, you can independently scale rollout workers, keep the algorithm responsive, and plug in specialized distributed trainers without rewriting the agent loop.
