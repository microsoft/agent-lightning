# Debugging and Troubleshooting

Debugging an agent workflow is much easier when you can peel back the layers: run the rollout logic in isolation, dry-run the trainer loop, then exercise the full algorithm and runner stack. The [`examples/apo/apo_debug.py`]({{ src("examples/apo/apo_debug.py") }}) script showcases these techniques in a compact form. This guide breaks them down and explains when to reach for each tool.

## Using [`Runner`][agentlightning.Runner] in Isolation

[`Runner`][agentlightning.Runner] is a core building block within the Agent-lightning architecture. It's a long-lived worker that wraps your [`LitAgent`][agentlightning.Runner] that is capable of tracing and communicating with the [`LightningStore`][agentlightning.LightningStore]. Normally, users will not need to mind using [`Runner`][agentlightning.Runner] directly as it's auto-managed. This tutorial tells you how to manually create a [`Runner`][agentlightning.Runner] and use it for debugging.

If you are using [`@rollout`][agentlightning.rollout] or [`LitAgent`][agentlightning.LitAgent] to define your agent logic, both will produce a [`LitAgent`][agentlightning.LitAgent] instance. You can use [`LitAgentRunner`][agentlightning.LitAgentRunner] to run that agent in isolation. Firstly, to create a [`LitAgentRunner`][agentlightning.LitAgentRunner], you need to provide a [`Tracer`][agentlightning.Tracer] instance. [`LitAgentRunner`][agentlightning.LitAgentRunner] will not automatically create a [`Tracer`][agentlightning.Tracer] for you. For tutorials on how to create and use a [`Tracer`][agentlightning.Tracer], please refer to [Working with Traces](./traces.md) tutorial.

[`Runner.run_context`][agentlightning.Runner.run_context] will initialize the [`Runner`][agentlightning.Runner] to a state that it's ready to run a particular agent function. To make this happen, other than the [`Tracer`][agentlightning.Tracer] and the agent, you also need to provide a [`LightningStore`][agentlightning.LightningStore] which is used to collected all the spans generated during the rollout. You can use [`InMemoryLightningStore`][agentlightning.InMemoryLightningStore] for debugging purpose, which keeps all the data in memory without any network calls.

Putting it all together, we have:

```python
import agentlightning as agl

tracer = agl.OtelTracer()
runner = agl.LitAgentRunner(tracer)
store = agl.InMemoryLightningStore()

with runner.run_context(agent=apo_rollout, store=store):
    ...
```

Now let's talk about what can be done within the `run_context` block. The most important method is `runner.step(...)`, which executes a single rollout of the agent logic. You need to provide the input payload for the rollout as the argument of `runner.step(...)`. The input payload should match what your agent function expects. For example, if your agent function is defined as:

```python
with runner.run_context(agent=apo_rollout, store=store):
    resource = agl.PromptTemplate(template="You are a helpful assistant. {any_question}", engine="f-string")
    rollout = await runner.step(
        "Explain why the sky appears blue using principles of light scattering in 100 words.",
        resources={"main_prompt": resource},
    )

- `run_context` guards the runner lifecycle. Under the hood it calls `init`, `init_worker`, and the matching teardown hooks so you get the same instrumentation you would inside a trainer-managed process.
- `InMemoryLightningStore` keeps rollouts and spans local to the process. Swap in `LightningStoreClient` when you want to share state across processes.
- `runner.step(...)` executes a single rollout and returns the full `Rollout` object, including status and metadata. That makes it ideal for tight edit-run loops when you are fixing prompt/templates or agent logic.

!!! note

    You still get spans while debugging this way. After the call to `runner.step`, query the store for spans if you want to inspect prompts or rewards:

    ```python
    spans = await store.query_spans(rollout.rollout_id)
    ```

    The `log_llm_span` helper in [`examples/apo/apo_custom_algorithm.py`]({{ config.repo_url }}/tree/{{ config.extra.source_commit }}/examples/apo/apo_custom_algorithm.py) shows one way to surface those events with Rich logging.

### When to use `runner.step`

- Validate that resources (prompt templates, model handles, etc.) are wired correctly before you loop over a dataset.
- Capture quick repros for bugs reported in production by re-running a single rollout with the same input payload.
- Benchmark updated agent logic in isolation—`runner.step` bypasses the store queue and executes immediately.
- Send the current rollout to the algorithm for online/continuous learning.

## Dry-Run the Trainer Loop

Once single rollouts behave, switch to the trainer’s dry-run mode. `Trainer.dev` spins up a lightweight fast algorithm so you can exercise the same infrastructure as `Trainer.fit` without standing up your full RL stack.

```python
from agentlightning import Trainer
from agentlightning.types import Dataset, PromptTemplate

dataset: Dataset[str] = [
    "Explain why the sky appears blue using principles of light scattering in 100 words.",
    "What's the capital of France?",
]
resource = PromptTemplate(template="You are a helpful assistant. {any_question}", engine="f-string")

trainer = Trainer(
    n_workers=1,
    initial_resources={"main_prompt": resource},
)
trainer.dev(apo_rollout, dataset)
```

- `trainer.dev` streams up to ten tasks through the runner infrastructure and prints span data to the console. It is the fastest way to verify that your agent can read resources, talk to external services, and emit a final reward.
- Provide the same `LitAgent` you will use in production (`apo_rollout` in the example). The trainer wraps it in the runner stack so the behaviour matches `fit`.
- Seed `initial_resources` with any named resources your agent expects. When you plug in a real algorithm later, those fields will be overwritten by whatever the algorithm writes into the store.

!!! tip

    Keep your logging setup identical between debug and training runs. The call to `configure_logger()` at the bottom of `apo_debug.py` ensures runner and trainer traces share the same formatting and verbosity.

## Debug the Algorithm/Runner Boundary

Many issues surface once the optimisation algorithm and runners have to coordinate through the store. The `apo_custom_algorithm.py` example demonstrates how to debug that interaction in three separate processes: store, algorithm, and runner.

- **Use a real store implementation.** Instantiate `LightningStoreClient("http://localhost:4747")` so the algorithm and runners talk to the same persistence layer that `Trainer` would manage.
- **Add and consume resources explicitly.** The algorithm updates resources with `store.add_resources(...)` before enqueuing a rollout. This mirrors what `Trainer` does on your behalf during `fit`, so bugs here usually point to mismatched resource names or payload types.
- **Poll for rollout completion.** `store.wait_for_rollouts(...)` gives you a simple loop that surfaces when runners fail to pick up tasks. If the wait times out, inspect runner logs or run `runner.step` manually to reproduce the failure case.
- **Inspect spans and rewards.** Helpers like `log_llm_span` and `find_final_reward` illustrate how to introspect the traces that flow through the store. Keeping these utilities in your algorithm process makes it easier to label “bad” runs for replay in `runner.step`.

!!! warning

    Avoid mixing debug modes in the same Python process unless you are sure the tracer backend can handle multiple initialisations. The CLI in `apo_debug.py` runs either the runner workflow or the trainer workflow, not both sequentially.
