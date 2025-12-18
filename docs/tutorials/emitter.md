# Using the Emitter

[](){ #using-emitter }

While returning a single float for the final reward is sufficient for many algorithm-agent combinations, some advanced scenarios require richer feedback. For instance, an algorithm might learn more effectively if it receives intermediate rewards throughout a multi-step task, or if the agent needs to emit additional spans for debugging or analysis.

Agent-lightning provides an **emitter** module for recording custom spans inside your agent logic. Just as [Tracer][agentlightning.Tracer] automatically instruments common operations (for example, LLM calls), each emitter helper sends a [Span][agentlightning.Span] that captures Agent-lightning-specific work so downstream algorithms can query it later. See [Working with Traces](./traces.md) for more details.

For multi-step routines such as function calls, tools, or adapters, wrap code with [`operation`][agentlightning.operation] — either as a decorator or a context manager — to capture inputs, outputs, and metadata on a dedicated [`operation`][agentlightning.operation] span. This makes it easier to correlate downstream annotations (like rewards or messages) with the higher-level work that produced them.

You can find the emitter functions in [`agentlightning.emitter`](../reference/agent.md).

## Emitting Rewards, Messages, and More

Here are the primary emitter functions:

* [`emit_reward(value: float)`][agentlightning.emit_reward]: Records an intermediate/final reward, which is a convenient wrapper of [`emit_annotation`][agentlightning.emit_annotation].
* [`emit_annotation(attributes: Dict[str, Any])`][agentlightning.emit_annotation]: Records arbitrary metadata as a span.
* [`emit_message(message: str)`][agentlightning.emit_message]: Records a simple log message as a span.
* [`emit_exception(exception: BaseException)`][agentlightning.emit_exception]: Records a Python exception, including its type, message, and stack trace.
* [`emit_object(obj: Any)`][agentlightning.emit_object]: Records any JSON-serializable object, perfect for structured data.

Let's first see an example of an agent using these emitters to provide detailed feedback.

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

Each helper accepts nested `attributes` (or keyword arguments for [`operation`][agentlightning.operation]) and automatically flattens/sanitizes them into dotted OpenTelemetry keys. This means you can pass ordinary dictionaries/lists without pre-processing and still get consistent attribute names such as `meta.any_attribute` across all emitter operations. Agent-lightning does not restrict the attributes you supply, but it is best to consult [OpenTelemetry's semantic conventions](https://opentelemetry.io/docs/specs/semconv/) for recommended names. Agent-lightning also defines [specific semconv](../reference/semconv.md) for its own use cases. The pattern looks like this:

```python
from opentelemetry.semconv.attributes import server_attributes
from agentlightning import emit_object

emit_object({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
}, attributes={
    server_attributes.SERVER_ADDRESS: "127.0.0.1",
    server_attributes.SERVER_PORT: 8080,
})
```

Running the above code sends the following span to the backend if you have a tracer active:

```text
Span(
    name='agentlightning.object',
    attributes={
        'agentlightning.object.type': 'dict',
        'agentlightning.object.json': '{"name": "John Doe", "age": 30, "email": "john.doe@example.com"}',
        'server.address': '127.0.0.1',
        'server.port': 8080
    }
)
```

!!! tip

    If you don't have a tracer active, the above code will raise the following error:

    ```text
    RuntimeError: No active tracer found. Cannot emit object span.
    ```

    By default, emitter helpers delegate to the active tracer to create and export spans (specifically via [`Tracer.create_span`][agentlightning.Tracer.create_span]). If you want to emit spans without an active tracer, set `propagate=False` to keep the span local — a useful option for offline tests. The default `True` streams spans through the active tracer/exporters.

When working with [agentlightning.semconv](../reference/semconv.md), you typically use utilities such as [`make_tag_attributes`][agentlightning.utils.otel.make_tag_attributes] and [`make_link_attributes`][agentlightning.utils.otel.make_link_attributes] to build the attributes dictionary. For example:

```python
from agentlightning.utils.otel import make_tag_attributes

emit_annotation(make_tag_attributes(["tool", "calculator", "fast", "good"]))
```

The above code will send a span with the following attributes to the backend:

```json
{
    "agentlightning.tag.0": "tool",
    "agentlightning.tag.1": "calculator",
    "agentlightning.tag.2": "fast",
    "agentlightning.tag.3": "good"
}
```

A counterpart utility function [`extract_tags_from_attributes`][agentlightning.utils.otel.extract_tags_from_attributes] is also available to extract the tags from the attributes dictionary.

## Operations

The [`operation`][agentlightning.operation] helper tracks logical units of work within your agent, capturing inputs, outputs, timing, and success/failure status. Unlike point-in-time emitters, operations create a span representing a time interval. Use operations for tool calls, multi-step workflows, debugging, and performance monitoring. [`operation`][agentlightning.operation] works as either a decorator or a context manager.

The decorator automatically captures function arguments as inputs and the return value as output:

```python
import agentlightning as agl

@agl.operation
def search_documents(query: str, max_results: int = 10) -> list[dict]:
    results = perform_search(query, max_results)
    return results

@agl.operation(category="tool", priority="high")
def execute_calculation(expression: str) -> float:
    return eval_safely(expression)
```

The example above emits a span with `{"category": "tool", "priority": "high"}` attributes. It also records the function input and output via [OPERATION_INPUT][agentlightning.semconv.LightningSpanAttributes.OPERATION_INPUT] and [OPERATION_OUTPUT][agentlightning.semconv.LightningSpanAttributes.OPERATION_OUTPUT]. It works with async functions too:

```python
@agl.operation
async def async_api_call(endpoint: str, payload: dict) -> dict:
    response = await http_client.post(endpoint, json=payload)
    return response.json()
```

Override the operation name if needed:

```python
@agl.operation(name="custom-name")
def any_weird_name_i_dont_want():
    pass
```

For more control, [`operation`][agentlightning.operation] can also be used as a context manager to explicitly record inputs and outputs:

```python
with agl.operation(tool_name="web_search") as op:
    op.set_input(query="latest AI research", filters={"date": "2024"})
    results = search_web("latest AI research", {"date": "2024"})
    op.set_output({"result_count": len(results), "top_result": results[0]})
```

The `propagate=False` flag also applies to [`operation`][agentlightning.operation] when you want to keep operations local without requiring an active tracer:

```python
@agl.operation(propagate=False)
def local_test():
    return "Not sent to backend"
```

## Linking to Other Spans

Sometimes a span should explicitly point back to another span that produced the input it is working on (for example, linking a reward annotation to the [`agentlightning.operation`][agentlightning.operation] span that generated a response). Agent-lightning encodes these relationships through flattened link attributes. The helper [`make_link_attributes`][agentlightning.utils.otel.make_link_attributes] converts a dictionary of keys such as `trace_id`, `span_id`, or any custom attribute into the `"agentlightning.link.*"` ([LightningSpanAttributes.LINK][agentlightning.semconv.LightningSpanAttributes.LINK]) fields expected by the backend. Later, [`query_linked_spans`][agentlightning.utils.otel.query_linked_spans] can recover the original span(s) from those link descriptors.

```python
import opentelemetry.trace as trace_api
from agentlightning import emit_annotation, operation
from agentlightning.utils.otel import make_link_attributes, make_tag_attributes

with operation(conversation_id="chat-42") as op:
    # ... perform the work ...
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
