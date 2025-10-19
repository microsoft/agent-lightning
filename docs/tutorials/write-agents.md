# Writing Agents

This tutorial will focus on the heart of the system: the agent itself, guiding you through the different ways to define an agent's logic in Agent-lightning.

The basic requirements for any agent are:

1.  It must accept a single **task** as input.
2.  It must accept a set of tunable **resources** (like a [PromptTemplate][agentlightning.PromptTemplate] or [LLM][agentlightning.LLM]).
3.  It must **emit** trace span data so that algorithms can understand its behavior and learn from it. *The simplest way to do this is by returning a final reward.*

In practice, please also bear in mind that tasks, resources, and spans have extra requirements, in order to make it *trainable* within Agent-lightning:

1.  You will need a training dataset containing a set of tasks, of the same type that your agent expects as input.
2.  The tunable resources are related to the algorithm. For example, the APO algorithm we've seen tunes a [PromptTemplate][agentlightning.PromptTemplate]. Other algorithms might tune model weights or other configurations.
3.  The type of spans an algorithm can use varies. Almost all algorithms support a single, final reward span at the end of a rollout. However, not all algorithms support rewards emitted mid-rollout, let alone other kinds of spans like exceptions or log messages.

This tutorial will show you how to write an agent that can handle various tasks and resources and emit all kinds of spans. However, you should understand that agents and algorithms are often co-designed. Supporting new types of resources or spans in an algorithm is often much more complex than just adding them to an agent.

## [`@rollout`][agentlightning.rollout] Decorator

The simplest way to create an agent is by writing a standard Python function and marking it with the [@rollout][agentlightning.rollout] decorator. This approach is perfect for agents with straightforward logic that doesn't require complex state management.

Agent-lightning automatically inspects your function's signature and injects the required resources. For example, if your function has a parameter named `prompt_template`, Agent-lightning will find the [PromptTemplate][agentlightning.PromptTemplate] resource for the current rollout and pass it in.

Let's revisit the `room_selector` agent from the first tutorial:

```python
from typing import TypedDict
from agentlightning import PromptTemplate, rollout

# Define a data structure for the task input
class RoomSelectionTask(TypedDict):
    # ... fields for the task ...
    pass

@rollout
def room_selector(task: RoomSelectionTask, prompt_template: PromptTemplate) -> float:
    # 1. Use the injected prompt_template to format the input for the LLM
    prompt = prompt_template.format(**task)

    # 2. Execute the agent's logic (e.g., call an LLM, use tools)
    # ...

    # 3. Grade the final choice to get a reward
    reward = room_selection_grader(final_message, task["expected_choice"])

    # 4. Return the final reward as a float
    return reward
```

When you train this agent, the dataset is expected to be a list of `RoomSelectionTask` objects:

```python
from agentlightning import Dataset, Trainer

dataset: Dataset[RoomSelectionTask] = [
    RoomSelectionTask(date="2025-10-15", time="10:00", duration_min=60, attendees=10),
    RoomSelectionTask(date="2025-10-16", time="10:00", duration_min=60, attendees=10),
]

Trainer().fit(agent=room_selector, train_dataset=dataset)
```

Behind the scenes, the [`@rollout`][agentlightning.rollout] decorator wraps your function in a `FunctionalLitAgent` object, which is a subclass of [LitAgent][agentlightning.LitAgent] introduced below, making it compatible with the [Trainer][agentlightning.Trainer] and [Runner][agentlightning.Runner]. It supports parameters like `task`, `prompt_template`, `llm`, and `rollout`, giving you flexible access to the execution context.

Here is another example with more advanced usage with `llm` and `rollout` as parameters. The `llm` parameter gives you an OpenAI-compatible LLM endpoint to interact with, which can be tuned under the hood by algorithms. The `rollout` parameter gives you the full [Rollout][agentlightning.Rollout] object, which contains the rollout ID, rollout mode (training or validation), etc.

```python
from openai import OpenAI
from agentlightning import LLM, Rollout

class FlightBookingTask(TypedDict):
    request: str
    expected_booking: dict

@rollout
def flight_assistant(task: FlightBookingTask, llm: LLM, rollout: Rollout) -> float:
    print(f"Rollout ID: {rollout.rollout_id}")
    print(f"Rollout Mode: {rollout.mode}")

    # Use the tuned LLM resource to create an OpenAI client
    client = OpenAI(
        # This endpoint could be a proxy to a proxy to a proxy ...
        # It could be different every time `flight_assistant` is called
        # But it should be OpenAI-API compatible
        base_url=llm.endpoint,

        # Use a dummy key if not provided
        # Usually this does not matter because the training LLM is often not guarded by an API key
        # But you can use `or os.environ["OPENAI_API_KEY"]` to make the function compatible with 3rd-party LLMs
        api_key=llm.api_key or "dummy-key",
    )

    # Make an API call with the specified model
    response = client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "user", "content": task["request"]}],
    )
    # Whether the API supports features like streaming, tool calls, etc. depends on
    # the endpoint that algorithms are serving to you.
    final_message = response.choices[0].message.content

    # Grade the result and return a reward
    reward = grade_flight_booking(final_message, task["expected_booking"])
    return reward
```

## Return Values from Agents

The value your agent function returns (i.e., the return value of the function decorated by [`@rollout`][agentlightning.rollout]) is crucial, as it's the primary way to report the outcome of a rollout. Agent-lightning supports several return types to accommodate different scenarios, from simple rewards to detailed, custom traces.

* **`float`**: This is the simplest and most common return type. The `float` is treated as the **final reward** for the entire rollout. Agent-lightning automatically creates a final reward span based on this value.

* **`None`**: Returning `None` tells the runner that trace collection is being handled entirely by the [Tracer][agentlightning.Tracer] through auto-instrumentation (e.g., via AgentOps). In this case, the runner will simply retrieve the spans that the tracer has already captured.

!!! important "Emitting the Final Reward"

    When returning `None`, you must still ensure a final reward is logged. You can do this by using the [`emit_reward`][agentlightning.emit_reward] function (covered in the [Emitter section][using-emitter] below) or by wrapping your reward calculation function with the [`@reward`][agentlightning.reward.reward] decorator.

* **`list[ReadableSpan]`** or **`list[Span]`**: For advanced use cases, you can manually construct and return a complete list of all spans for the rollout. This gives you full control over the trace data. You can return either a list of OpenTelemetry `ReadableSpan` objects or Agent-lightning's native `Span` objects.

For most users, returning a **`float`** for simple agents or returning **`None`** and using the emitter for more complex ones are the recommended approaches.

## Class-based Agents

For more complex agents that require state, helper methods, or distinct logic for training versus validation, you can create a class that inherits from `LitAgent`. This object-oriented approach provides more structure and control over the agent's lifecycle.

To create a class-based agent, you subclass [agentlightning.LitAgent][] and implement its `rollout` method.

[](){ #introduction-to-named-resources }

Here's how the `room_selector` could be implemented as a class. The rollout method has a slightly different signature than the function-based agent, mainly in how it handles the resources. Putting it simply, algorithms do not just send a [PromptTemplate][agentlightning.PromptTemplate] to the agents, they instead send [NamedResources][agentlightning.NamedResources], which is a mapping from resource key to [Resource][agentlightning.Resource]. This design is to allow for more advanced features like multi-resource tuning.

With [`@rollout`][agentlightning.rollout] decorator, the resource with correctly matched type will be automatically injected into the rollout method. However, when you use a class-based agent, you need to manually access the resource from the `resources` dictionary. Built-in algorithms listed their resource key naming conventions [here](../algorithm-zoo/index.md).

```python
import agentlightning as agl

class RoomSelectorAgent(agl.LitAgent[RoomSelectionTask]):
    def rollout(self, task: RoomSelectionTask, resources: agl.NamedResources, rollout: agl.Rollout) -> float:
        # 1. Access the prompt_template from the resources dictionary
        prompt_template = resources["prompt_template"]

        # 2. Execute the agent's logic
        prompt = prompt_template.format(**task)
        # ...

        # 3. Grade the final choice
        reward = room_selection_grader(final_message, task["expected_choice"])

        # 4. Return the final reward
        return reward

# To use it with the trainer:
# agent = RoomSelectorAgent()
# trainer.fit(agent=agent, ...)
```

The `LitAgent` class provides several methods you can override for more fine-grained control:

* `rollout()`: The primary method for the agent's logic. It's called for both training and validation by default.
* `training_rollout()` / `validation_rollout()`: Implement these if you need different behavior during training (e.g., with exploration) and validation (e.g., with deterministic choices).
* `rollout_async()` / `training_rollout_async()` / `validation_rollout_async()`: Implement the asynchronous versions of these methods if your agent uses `asyncio`.

!!! note

    Rollout is always executed in an asynchronous context no matter whether the agent is asynchronous or synchronous. If your synchronous agent contains some `asyncio.run()` calls, it might raise an error that there is already an event loop running. To avoid blocking the event loop, it's recommended to offload the inner async operations to a separate thread. Here is a sample code:

    ```python
    import asyncio
    import queue
    import threading

    def run_sync_ephemeral(coro) -> Any:
        """
        Run an async coroutine from sync code.
        - If no loop in this thread: use asyncio.run() directly.
        - If already in an event loop: spawn a worker thread that calls asyncio.run()
        (which creates and closes a brand-new event loop per call).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread; safe to use asyncio.run
            return asyncio.run(coro)

        # Already in a running loop -> execute in a worker thread
        q = queue.Queue[Any]()

        def worker():
            try:
                result = asyncio.run(coro)  # creates & closes its own loop
                q.put((True, result))
            except BaseException as e:
                q.put((False, e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        ok, payload = q.get()
        t.join()
        if ok:
            return payload
        raise payload
    ```

## Using the Emitter

[](){ #using-emitter }

While returning a single float for the final reward is sufficient for many algorithms, some advanced scenarios require richer feedback. For instance, an algorithm might learn more effectively if it receives intermediate rewards throughout a multi-step task.

Agent-lightning provides an **emitter** module that allows you to record custom spans from within your agent's logic. Like many common operations (like LLM calls) that are automatically instrumented by [Tracer][agentlightning.Tracer], the emitter will also send a [Span][agentlightning.Span] that records an Agent-lightning-specific operation. Then algorithms can query and read those spans later. See [Working with Traces](./traces.md) for more details.

You can find the emitter functions from [agentlightning.emitter](../reference/agent.md).

### Emitting Rewards, Messages, and More

Here are the primary emitter functions:

* [`emit_reward(value: float)`][agentlightning.emit_reward]: Records an intermediate reward.
* [`emit_message(message: str)`][agentlightning.emit_message]: Records a simple log message as a span.
* [`emit_exception(exception: BaseException)`][agentlightning.emit_exception]: Records a Python exception, including its type, message, and stack trace.
* [`emit_object(obj: Any)`][agentlightning.emit_object]: Records any JSON-serializable object, perfect for structured data.

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
