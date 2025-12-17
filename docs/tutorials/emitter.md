## Using the Emitter

[](){ #using-emitter }

While returning a single float for the final reward is sufficient for many algorithm-agent combinations, some advanced scenarios require richer feedback. For instance, an algorithm might learn more effectively if it receives intermediate rewards throughout a multi-step task, or if the agent needs to emit additional spans for debugging or analysis.

Agent-lightning provides an **emitter** module that allows you to record custom spans from within your agent's logic. Like many common operations (like LLM calls) that are automatically instrumented by [Tracer][agentlightning.Tracer], the emitter will also send a [Span][agentlightning.Span] that records an Agent-lightning-specific operation. Then algorithms can query and read those spans later. See [Working with Traces](./traces.md) for more details.

For multi-step routines (function calls, tools, or adapters) you can wrap code with [`operation`][agentlightning.operation], either as a decorator or a context manager,to capture inputs, outputs, and metadata on a dedicated [`operation`][agentlightning.operation] span. This makes it easier to correlate downstream annotations (like rewards or messages) with the higher-level work that produced them.

You can find the emitter functions from [`agentlightning.emitter`](../reference/agent.md).

### Emitting Rewards, Messages, and More

Here are the primary emitter functions:

* [`emit_reward(value: float)`][agentlightning.emit_reward]: Records an intermediate reward.
* [`emit_message(message: str)`][agentlightning.emit_message]: Records a simple log message as a span.
* [`emit_exception(exception: BaseException)`][agentlightning.emit_exception]: Records a Python exception, including its type, message, and stack trace.
* [`emit_object(obj: Any)`][agentlightning.emit_object]: Records any JSON-serializable object, perfect for structured data.

Each helper accepts nested `attributes` (or keyword arguments, in the case of [`operation`][agentlightning.operation]) and automatically flattens/sanitizes them into dotted OpenTelemetry keys. That means you can pass ordinary dictionaries/lists without pre-processing and still get consistent attribute names such as `meta.tag` across [`emit_annotation`][agentlightning.emit_annotation], [`emit_message`][agentlightning.emit_message], [`emit_object`][agentlightning.emit_object], [`emit_exception`][agentlightning.emit_exception], [`emit_reward`][agentlightning.emit_reward], and [`operation`][agentlightning.operation]. All emitter helpers also support a `propagate` flag; setting `propagate=False` keeps the span local—useful for offline tests—while the default `True` streams spans through the active tracer/exporters.

Let's see an example of an agent using these emitters to provide detailed feedback.

```python
import agentlightning as agl

@agl.rollout
def multi_step_agent(task: dict, prompt_template: PromptTemplate) -> float:
    try:
        # Step 1: Initial planning
        agl.emit_message("Starting planning phase.")
        plan = generate_plan(task, prompt_template)
        agl.emit_object({"plan_steps": len(plan), "first_step": plan[0]})

        # Award a small reward for a valid plan
        plan_reward = grade_plan(plan)
        agl.emit_reward(plan_reward)

        # Step 2: Execute the plan
        agl.emit_message(f"Executing {len(plan)}-step plan.")
        execution_result = execute_plan(plan)

        # Step 3: Final evaluation
        final_reward = custom_grade_final_result(execution_result, task["expected_output"])

        # The return value is treated as the final reward for the rollout
        return final_reward

    except ValueError as e:
        # Record the specific error and return a failure reward
        agl.emit_exception(e)
        return 0.0
```

By using the emitter, you create a rich, detailed trace of your agent's execution. This data can be invaluable for debugging and is essential for advanced algorithms that can learn from more than just a single final score.

### Linking to Other Spans

Sometimes a span should explicitly point back to another span that produced the input it is working on (for example, linking a reward annotation to the `"agentlightning.operation"` span that generated a response). Agent-lightning encodes these relationships through flattened link attributes. The helper [`make_link_attributes`][agentlightning.utils.otel.make_link_attributes] converts a dictionary of keys—such as `trace_id`, `span_id`, or any custom attribute—into the `"agentlightning.link.*"` fields expected by the backend. Later on, [`query_linked_spans`][agentlightning.utils.otel.query_linked_spans] can be used to recover the original span(s) from those link descriptors.

```python
import opentelemetry.trace as trace_api
from agentlightning import emit_annotation, operation
from agentlightning.utils.otel import make_link_attributes, make_tag_attributes

with operation(conversation_id="chat-42") as op:
    # ... perform the work ...
    span_ctx = op.span.get_span_context()
    link_attrs = make_link_attributes({
        "conversation_id": "chat-42",
    })

    emit_annotation(
        {
            **link_attrs,
            **make_tag_attributes(["reward", "good"]),
        }
    )
```

When analyzing in adapters, pass the extracted link models to [`query_linked_spans`][agentlightning.utils.otel.query_linked_spans] to retrieve the matching span(s):

```python
from agentlightning.utils.otel import extract_links_from_attributes, query_linked_spans

annotation_span = ...  # Span from your trace store
operation_spans = [...]  # list of spans you want to search

link_models = extract_links_from_attributes(annotation_span.attributes)
matches = query_linked_spans(operation_spans, link_models)
assert matches  # Contains the original operation span
```

!!! tip "Correlating Rewards with LLM Requests"

    [Tracer](./traces.md) instruments each request/response as its own span. You can link to the [`gen_ai.response.id`](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) attribute, which comes from the LLM response ID.

    ```python
    from agentlightning import emit_reward
    from agentlightning.utils.otel import make_link_attributes

    result = call_llm(prompt)
    reward_links = make_link_attributes({"gen_ai.response.id": result.id})
    emit_reward(0.9, attributes=reward_links)
    ```

    Later, use the same `gen_ai.response.id` key inside `query_linked_spans` to find the reward(s) that reference that specific LLM request span.
