# Debugging and Troubleshooting

When you train your own agent with Agent-lightning, most failures surface because the agent logic is brittle or simply incorrect. Debugging becomes easier when you peel back the stack: start by driving the rollout logic on its own, dry-run the trainer loop, and only then bring the full algorithm and runner topology online. The [`examples/apo/apo_debug.py`]({{ src("examples/apo/apo_debug.py") }}) script demonstrates these techniques; this guide expands on each approach and helps you decide when to reach for them.

## Using [`Runner`][agentlightning.Runner] in Isolation

[`Runner`][agentlightning.Runner] is a long-lived worker that wraps your [`LitAgent`][agentlightning.LitAgent], coordinates tracing, and talks to the [`LightningStore`][agentlightning.LightningStore]. In typical training flows the trainer manages runners for you, but being able to spin one up manually is invaluable while debugging.

If you define rollout logic with [`@rollout`][agentlightning.rollout] or implement a [`LitAgent`][agentlightning.LitAgent] directly, you will get a [`LitAgent`][agentlightning.LitAgent] instance and you should be able to execute it with [`LitAgentRunner`][agentlightning.LitAgentRunner], which is a subclass of [`Runner`][agentlightning.Runner]. The runner needs but does not instantiate a [`Tracer`][agentlightning.Tracer], so supply one yourself. See [Working with Traces](./traces.md) for a walkthrough of tracer options.

[`Runner.run_context`][agentlightning.Runner.run_context] prepares the runner to execute a particular agent. Besides the agent and tracer you must provide a store that will collect spans and rollouts. [`InMemoryLightningStore`][agentlightning.InMemoryLightningStore] keeps everything in-process, which is perfect for debugging sessions.

```python
import agentlightning as agl

tracer = agl.OtelTracer()
runner = agl.LitAgentRunner(tracer)
store = agl.InMemoryLightningStore()

with runner.run_context(agent=apo_rollout, store=store):
    ...
```

Inside the [`run_context`][agentlightning.Runner.run_context] block you can call [`runner.step(...)`][agentlightning.Runner.step] to execute a single rollout. The payload includes the task input and any [`NamedResources`][agentlightning.NamedResources] the agent expects. Read [introduction to Resources][introduction-to-resources] and [NamedResources][introduction-to-named-resources] for more details. For example, if your agent references a [`PromptTemplate`][agentlightning.PromptTemplate], pass it through the `resources` argument:

```python
with runner.run_context(agent=apo_rollout, store=store):
    resource = agl.PromptTemplate(template="You are a helpful assistant. {any_question}", engine="f-string")
    rollout = await runner.step(
        "Explain why the sky appears blue using principles of light scattering in 100 words.",
        resources={"main_prompt": resource},
    )
```

You can do as many things as you want within the [`Runner.run_context`][agentlightning.Runner.run_context] block. After the rollout finishes you can query the store to inspect what happened:

```python
print(await store.query_rollouts())
print(await store.query_spans(rollout.rollout_id))
```

Example output (with a reward span captured):

```python
[Rollout(rollout_id='ro-519769241af8', input='Explain why the sky appears blue using principles of light scattering in 100 words.', start_time=1760706315.6996238, ..., status='succeeded')]
[Span(rollout_id='ro-519769241af8', attempt_id='at-a6b62caf', sequence_id=1, ..., name='agentlightning.reward', attributes={'reward': 0.95}, ...)]
```

Swap in an [`AgentOpsTracer`][agentlightning.AgentOpsTracer] instead of [`OtelTracer`][agentlightning.OtelTracer] to see the underlying LLM spans alongside reward information:

```python
[
    Span(rollout_id='ro-519769241af8', attempt_id='at-a6b62caf', sequence_id=1, ..., name='openai.chat.completion', attributes={..., 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'You are a helpful assistant. Explain why the sky appears blue using principles of light scattering in 100 words.', ...}),
    Span(rollout_id='ro-519769241af8', attempt_id='at-a6b62caf', sequence_id=2, ..., name='openai.chat.completion', attributes={..., 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'Evaluate how well the output fulfills the task...', ...}),
    Span(rollout_id='ro-519769241af8', attempt_id='at-a6b62caf', sequence_id=3, ..., name='agentlightning.reward', attributes={'reward': 0.95}, ...)
]
```

!!! tip

    Spans too difficult to read? Try using [`Adapter`][agentlightning.Adapter] to convert them into a [more readable format](./traces.md).

[`Runner.step`][agentlightning.Runner.step] executes a full rollout even though it is named "step". The companion method [`Runner.iter`][agentlightning.Runner.iter] executes multiple "steps" by continuously pulling new rollout inputs from the store until a stop event is set. Use `iter` once you are confident the single-step path works and you have another worker [`enqueue_rollout`][agentlightning.LightningStore.enqueue_rollout] to the store.

!!! tip

    You can also call [`Runner.step`][agentlightning.Runner.step] to inject ad-hoc rollouts into a running store being used by another algorithm, so that the rollouts can be consumed by the algorithms. This is very recently known as the paradigm of ["online RL"](https://cursor.com/blog/tab-rl). At the moment, no algorithm in the [algorithm zoo](../algorithm-zoo/index.md) consumes externally generated rollouts, but the data flow is available there if you need it.

## Debug with LLM Proxy

If you are dealing with LLM optimization like Reinforcement Learning, we generally recommend using an online stable LLM service for your debugging purposes, like `openai/gpt-4.1-nano`. After the debugging is done, you can switch to a local training endpoint.

However, if you want to use a local LLM features like [getting the token IDs](../deep-dive/serving-llm.md), you can also manually start a local vLLM server by:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8080
```

Then start the LLM proxy via the following script:

```python
import asyncio
import aiohttp
import agentlightning as agl

async def serve_llm_proxy():
    store = agl.InMemoryLightningStore()
    store_server = agl.LightningStoreServer(store, "127.0.0.1", 8081)
    await store_server.start()

    llm_proxy = agl.LLMProxy(
        port=8082,
        model_list=[
            {
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "litellm_params": {
                    "model": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct",
                    "api_base": "http://localhost:8080/v1",
                },
            }
        ],
        store=store_server,
    )

    await llm_proxy.start()
    await asyncio.sleep(1000000)
```

Test the served LLM proxy with a client like:

```python
async def test_llm_proxy():
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8082/v1/chat/completions", json={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Hello, world!"}],
        }) as response:
            print(await response.json())
```

You can now use the LLM proxy by specifying environment variables:

```bash
export OPENAI_API_BASE=http://localhost:8081/v1
export OPENAI_API_KEY=dummy
```

You might see warnings about `Missing or invalid rollout_id, attempt_id, or sequence_id` in the LLM proxy logs. This is fine because you don't have a rollout and attempt yet when you are debugging. When you started the training, the algorithm will create the rollouts for you and the warnings will go away.

## Hook into Runner's Lifecycle

[`Runner.run_context`][agentlightning.Runner.run_context] accepts a `hooks` argument so you can observe or augment lifecycle events without editing your agent. Hooks subclass [`Hook`][agentlightning.Hook] and can respond to four asynchronous callbacks: [`on_trace_start`][agentlightning.Hook.on_trace_start], [`on_rollout_start`][agentlightning.Hook.on_rollout_start], [`on_rollout_end`][agentlightning.Hook.on_rollout_end], and [`on_trace_end`][agentlightning.Hook.on_trace_end]. This is useful for:

- Capturing raw OpenTelemetry spans before they hit the store and before the [`LitAgentRunner`][agentlightning.LitAgentRunner] do postprocessing on the rollout
- Inspecting the tracer instance after they are activated
- Logging rollout inputs before they are processed by the agent

The `hook` mode in [`examples/apo/apo_debug.py`]({{ src("examples/apo/apo_debug.py") }}) prints every span collected during a rollout:

```python
import agentlightning as agl

# ... Same as previous example

class DebugHook(agl.Hook):
    async def on_trace_end(self, *, agent, runner, tracer, rollout):
        trace = tracer.get_last_trace()
        print("Trace spans collected during the rollout:")
        for span in trace:
            print(f"- {span.name} (status: {span.status}):\n  {span.attributes}")

with runner.run_context(
    agent=apo_rollout,
    store=store,
    hooks=[DebugHook()],
):
    await runner.step(
        "Explain why the sky appears blue using principles of light scattering in 100 words.",
        resources={"main_prompt": resource},
    )
```

Because hooks run inside the runner process you can also attach debuggers or breakpoints directly in the callback implementations.

!!! note

    For a better understanding of where hooks are called, we show a pseudo code of Runner's working flow below:

    ```python
    resources = await store.get_latest_resources()
    rollout = ...
    try:
        # <-- on_rollout_start
        with tracer.trace_context(...):
            # <--- on_trace_start
            result = await agent.rollout(...)
            # <--- on_trace_end
        post_process_result(result)
    except Exception:
        # <-- on_rollout_end
        await store.update_attempt(status=...)
    ```

## Dry-Run the Trainer Loop

Once single rollouts behave, switch to the trainer’s dry-run mode. [`Trainer.dev`][agentlightning.Trainer.dev] spins up a lightweight fast algorithm — [`agentlightning.Baseline`][agentlightning.Baseline] by default — so you can exercise the same infrastructure as [`Trainer.fit`][agentlightning.Trainer.fit] without standing up complex stacks like RL or SFT.

!!! warning
    When you enable multiple runners via `n_runners`, the trainer may execute them in separate worker processes. Attaching a debugger such as `pdb` is only practical when `n_runners=1`, and even then the runner might not live in the main process.

```python
import agentlightning as agl

dataset: agl.Dataset[str] = [
    "Explain why the sky appears blue using principles of light scattering in 100 words.",
    "What's the capital of France?",
]
resource = agl.PromptTemplate(template="You are a helpful assistant. {any_question}", engine="f-string")

trainer = agl.Trainer(
    n_runners=1,
    initial_resources={"main_prompt": resource},
)
trainer.dev(apo_rollout, dataset)
```

Just like [`Runner.run_context`][agentlightning.Runner.run_context], [`Trainer.dev`][agentlightning.Trainer.dev] requires the [`NamedResources`][agentlightning.NamedResources] your agent expects. The key difference is that resources are attached to the trainer rather than the runner.

[`Trainer.dev`][agentlightning.Trainer.dev] uses an almost switchable interface from [`Trainer.fit`][agentlightning.Trainer.fit]. It also needs a dataset to iterate over, similar to [`fit`][agentlightning.Trainer.fit]. Under the hood [`dev`][agentlightning.Trainer.dev] uses the same implementation as [`fit`][agentlightning.Trainer.fit], which means you can spin up multiple runners, observe scheduler behavior, and validate how algorithms adapt rollouts. The default [`Baseline`][agentlightning.Baseline] logs detailed traces so you can see each rollout as the algorithm perceives it:

```text
21:20:30 Initial resources set: {'main_prompt': PromptTemplate(resource_type='prompt_template', template='You are a helpful assistant. {any_question}', engine='f-string')}
21:20:30 Proceeding epoch 1/1.
21:20:30 Enqueued rollout ro-302fb202bd85 in train mode with sample: Explain why the sky appears blue using principles of light scattering in 100 words.
21:20:30 Enqueued rollout ro-e65a3ffaa540 in train mode with sample: What's the capital of France?
21:20:30 Waiting for 2 harvest tasks to complete...
21:20:30 [Rollout ro-302fb202bd85] Status is initialized to queuing.
21:20:30 [Rollout ro-e65a3ffaa540] Status is initialized to queuing.
21:20:35 [Rollout ro-302fb202bd85] Finished with status succeeded in 3.80 seconds.
21:20:35 [Rollout ro-302fb202bd85 | Attempt 1] ID: at-f84ad21c. Status: succeeded. Worker: Worker-0
21:20:35 [Rollout ro-302fb202bd85 | Attempt at-f84ad21c | Span 3a286a856af6bea8] #1 (openai.chat.completion) ... 1.95 seconds. Attribute keys: ['gen_ai.request.type', 'gen_ai.system', ...]
21:20:35 [Rollout ro-302fb202bd85 | Attempt at-f84ad21c | Span e2f44b775e058dd6] #2 (openai.chat.completion) ... 1.24 seconds. Attribute keys: ['gen_ai.request.type', 'gen_ai.system', ...]
21:20:35 [Rollout ro-302fb202bd85 | Attempt at-f84ad21c | Span 45ee3c94fa1070ec] #3 (agentlightning.reward) ... 0.00 seconds. Attribute keys: ['reward']
21:20:35 [Rollout ro-302fb202bd85] Adapted data: [Triplet(prompt={'token_ids': []}, response={'token_ids': []}, reward=None, metadata={'response_id': '...', 'agent_name': ''}), Triplet(prompt={'token_ids': []}, response={'token_ids': []}, reward=0.95, metadata={'response_id': '...', 'agent_name': ''})]
21:20:35 Finished 1 rollouts.
21:20:35 [Rollout ro-e65a3ffaa540] Status changed to preparing.
21:20:40 [Rollout ro-e65a3ffaa540] Finished with status succeeded in 6.39 seconds.
21:20:40 [Rollout ro-e65a3ffaa540 | Attempt 1] ID: at-eaefa5d4. Status: succeeded. Worker: Worker-0
21:20:40 [Rollout ro-e65a3ffaa540 | Attempt at-eaefa5d4 | Span 901dd6acc0f50147] #1 (openai.chat.completion) ... 1.30 seconds. Attribute keys: ['gen_ai.request.type', 'gen_ai.system', ...]
21:20:40 [Rollout ro-e65a3ffaa540 | Attempt at-eaefa5d4 | Span 52e0aa63e02be611] #2 (openai.chat.completion) ... 1.26 seconds. Attribute keys: ['gen_ai.request.type', 'gen_ai.system', ...]
21:20:40 [Rollout ro-e65a3ffaa540 | Attempt at-eaefa5d4 | Span 6c452de193fbffd3] #3 (agentlightning.reward) ... 0.00 seconds. Attribute keys: ['reward']
21:20:40 [Rollout ro-e65a3ffaa540] Adapted data: [Triplet(prompt={'token_ids': []}, response={'token_ids': []}, reward=None, metadata={'response_id': '...', 'agent_name': ''}), Triplet(prompt={'token_ids': []}, response={'token_ids': []}, reward=1.0, metadata={'response_id': '...', 'agent_name': ''})]
21:20:40 Finished 2 rollouts.
```

The only limitation is that resources remain static and components like [`LLMProxy`][agentlightning.LLMProxy] are not wired in. For richer dry runs you can subclass [`FastAlgorithm`][agentlightning.FastAlgorithm] and override the pieces you care about.

## Debug the Algorithm-Runner Boundary

[](){ #debug-with-external-store }

Debugging algorithms in Agent-Lightning is often more challenging than debugging agents. Algorithms are typically **stateful** and depend on several moving parts — runners, stores, and trainers — which makes it difficult to isolate and inspect their behavior. Even mocking an agent to cooperate with an algorithm can be costly and error-prone. To simplify this, Agent-Lightning provides a way to run algorithms in isolation so you can attach a debugger and inspect internal state without interference from other components.

By default, [`Trainer.fit`][agentlightning.Trainer.fit] runs the algorithm in the main process and thread, but its logs are interleaved with those from the store and runners, making it hard to follow what’s happening inside the algorithm itself. In [*Write Your First Algorithm*](../how-to/write-first-algorithm.md), we covered how to stand up a store, algorithm, and runner in isolation for your own implementations. This section extends that approach to cover two common questions:

1. How can I run built-in or class-based algorithms (inheriting from [`Algorithm`][agentlightning.Algorithm]) in isolation?
2. How can I still use [`Trainer`][agentlightning.Trainer] features like `n_runners`, `adapter`, or `llm_proxy` while debugging?

The solution is to keep using a [`Trainer`][agentlightning.Trainer] instance but **manage the store yourself**, running the algorithm and runner roles separately. This approach mirrors the internal process orchestration of [`Trainer.fit`][agentlightning.Trainer.fit], but with more visibility and control. Below, we show a step-by-step guide to achieve this with the [`calc_agent` example]({{ src("examples/calc_x/train_calc_agent.py") }}).

**1. Launch the store manually.**
In a separate terminal, start the store:

```bash
agl store --port 4747
```

Then, in your training script, create a [`LightningStoreClient`][agentlightning.LightningStoreClient] and pass it to the trainer:

```python
client = agl.LightningStoreClient("http://localhost:4747")
trainer = agl.Trainer(store=client, ...)
```

Set the environment variable `AGL_MANAGED_STORE=0` so the trainer doesn't attempt to manage the store automatically.

**2. Start the runner and algorithm processes separately.**
Each process should run the same training script, but with different environment variables specifying the current role.
This setup faithfully mirrors how [`Trainer.fit`][agentlightning.Trainer.fit] orchestrates these components behind the scenes.

```bash
# Terminal 2 – Runner process
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=runner \
    python train_calc_agent.py --external-store-address http://localhost:4747 --val-file data/test_mini.parquet

# Terminal 3 – Algorithm process
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=algorithm \
    python train_calc_agent.py --external-store-address http://localhost:4747 --val-file data/test_mini.parquet
```

**3. Reuse your existing trainer configuration.**
You can continue using the same datasets, adapters, and proxies as usual. Because the store is now external, you can:

* Attach debuggers to either the algorithm or runner process
* Add fine-grained logging or tracing
* Simulate partial failures or latency in individual components

This setup provides a faithful reproduction of the algorithm–runner interaction while keeping the store visible for inspection. Once you’ve resolved the issue, simply set `AGL_MANAGED_STORE=1` (or omit it) to return to the standard managed training workflow.
