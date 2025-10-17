# Debugging and Troubleshooting

If you are training your own agent with Agent-lightning, a majority of the bugs come from the agent is not robust enough or simply buggy. Debugging an agent workflow is much easier when you can peel back the layers: run the rollout logic in isolation, dry-run the trainer loop, then exercise the full algorithm and runner stack. The [`examples/apo/apo_debug.py`]({{ src("examples/apo/apo_debug.py") }}) script showcases these techniques in a compact form. This guide breaks them down and explains when to reach for each tool.

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

Now let's talk about what can be done within the `run_context` block. The most important method is [`runner.step(...)`](agentlightning.Runner.step), which executes a single rollout of the agent logic. You need to provide the input payload for the rollout as the argument of `runner.step(...)`. The input payload consists of the task input and [`NamedResources`][agentlightning.NamedResources] for the rollout. For example, if you are using a [`PromptTemplate`][agentlightning.PromptTemplate] resource in the agent, you can pass a dictionary with an arbitrary resource key to [`PromptTempalte`][agentlightning.PromptTemplate] as the value. Read [introduction to Resources][introduction-to-resources] and [NamedResources][introduction-to-named-resources] for more details.

```python
with runner.run_context(agent=apo_rollout, store=store):
    resource = agl.PromptTemplate(template="You are a helpful assistant. {any_question}", engine="f-string")
    rollout = await runner.step(
        "Explain why the sky appears blue using principles of light scattering in 100 words.",
        resources={"main_prompt": resource},
    )
```

You can do as many things as you want within the [`Runner.run_context`][agentlightning.Runner.run_context] block. Afterwards, you can interact with the store to get what has happened during this rollout. For example, printing:

```python
print(await store.query_rollouts())
print(await store.query_spans(rollout.rollout_id))
```

gives you a reward span captured:

```python
[Rollout(rollout_id='ro-519769241af8', input='Explain why the sky appears blue using principles of light scattering in 100 words.', start_time=1760706315.6996238, ..., status='succeeded')]
[Span(rollout_id='ro-519769241af8', attempt_id='at-a6b62caf', sequence_id=1, ..., name='agentlightning.reward', attributes={'reward': 0.95}, ...)]
```

If [`AgentOpsTracer`][agentlightning.AgentOpsTracer] replaces the [`OtelTracer`][agentlightning.OtelTracer] used above, you can also get the LLM spans captured:

```python
[
    Span(rollout_id='ro-519769241af8', attempt_id='at-a6b62caf', sequence_id=1, ..., name='openai.chat.completion', attributes={..., 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'You are a helpful assistant. Explain why the sky appears blue using principles of light scattering in 100 words.', ...}),
    Span(rollout_id='ro-519769241af8', attempt_id='at-a6b62caf', sequence_id=2, ..., name='openai.chat.completion', attributes={..., 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'Evaluate how well the output fulfills the task...', ...}),
    Span(rollout_id='ro-519769241af8', attempt_id='at-a6b62caf', sequence_id=3, ..., name='agentlightning.reward', attributes={'reward': 0.95}, ...)
]
```

!!! note

    You might wonder why [`Runner.step`][agentlightning.Runner.step] actually carries out one rollout, although it's called a step. The reason is that [`Runner.step`][agentlightning.Runner.step] is named as opposed to [`Runner.iter`][agentlightning.Runner.iter]. [`Runner.iter`][agentlightning.Runner.iter] on the other hand, runs a loop that executes multiple "steps". It continuously fetches new rollout inputs from the store and executes them until some stop event is set.

    You can also use [`Runner.step`][agentlightning.Runner.step] to connecting an existing algorithm, and add customized arbitrary rollouts to the store. But at the moment, no algorithm in the [algorithm zoo](../algorithm-zoo/index.md) supports leveraging rollouts generated by external [`Runner.step`] calls.

## Hook into Runner's Lifecycle

TBD: write this section based on apo_debug.py with hooks.

## Dry-Run the Trainer Loop

Once single rollouts behave, switch to the trainer’s dry-run mode. `Trainer.dev` spins up a lightweight fast algorithm ([`agentlightning.Baseline`][agentlightning.Baseline] by default) so you can exercise the same infrastructure as `Trainer.fit` without standing up complex algorithm stacks like RL or SFT.

!!! note

    A major difference between this method and the previous approaches is that with the `n_runners` parameter being set up, it's not easy to attach a debugger like `pdb` to the agent. You might be able to do so in some cases when `n_runners` is set to 1, but it's not guaranteed that one runner will be executed in the main process and main thread.

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

Similar to [`Runner.run_context`][agentlightning.Runner.run_context], [`Trainer.dev`][agentlightning.Trainer.dev] approach only requires a [`NamedResources`][agentlightning.NamedResources] being provided. The difference is that for this time, the [`NamedResources`][agentlightning.NamedResources] is provided to the [`Trainer`][agentlightning.Trainer] itself, instead of [`Runner`][agentlightning.Runner].

[`Trainer.dev`][agentlightning.Trainer.dev] is actually very similar to what the agent will behave in a real [`Trainer.fit`][agentlightning.Trainer.fit]. They actually share the same implementation underneath. You can spinning up `n_runners` and run multiple agents in parallel. [`Baseline`][agentlightning.Baseline] will also log many outputs on the console, to help you understand what your agent looks like from the angle of the algorithm:

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

The only differences might be that the resources are still static, and features like [`LLMProxy`][agentlightning.LLMProxy] are not available. You can subclass [`FastAlgorithm`][agentlightning.FastAlgorithm] to implement your own algorithm for dry run for this kind of testing.

## Debug the Algorithm/Runner Boundary

TBD: complete this section based on the draft.

We've seen how to run store, algorithm and runner in isolation with customized algorithm in ../how-to/write-first-algorithm.md. but in reality, it's more difficult to doing so when you are using a built-in algorithm based on [`Algorithm`][agentlightning.Algorithm] class, especially when you still want to use the configuration options like `n_runners`, `adapter`, `llm_proxy` in Trainer. Initializing the algorithm and runner on your own and manages the process is possible but it's non trivial and not recommended.

Run store in isolation: `agl store` in another terminal and initialize a `LightningStoreClient` before initializing the Trainer. put the store as `store=client` in Trainer init. Set `AGL_MANAGED_STORE=0` in environment variables.

Then in two terminals, launch the same script with different environment variables: `AGL_CURRENT_ROLE=algorithm` and `AGL_CURRENT_ROLE=runner`

The following is an example:

AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=runner python train_calc_agent.py --external-store-address http://localhost:4747 --val-file data/test_mini.parquet
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=algorithm python train_calc_agent.py --external-store-address http://localhost:4747 --val-file data/test_mini.parquet
