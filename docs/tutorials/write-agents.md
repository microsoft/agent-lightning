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

## `@rollout` Decorator

The simplest way to create an agent is by writing a standard Python function and marking it with the [@rollout][agentlightning.rollout] decorator. This approach is perfect for agents with straightforward logic that doesn't require complex state management.

Agent-lightning automatically inspects your function's signature and injects the required resources. For example, if your function has a parameter named `prompt_template`, Agent-lightning will find the `PromptTemplate` resource for the current rollout and pass it in.

Let's revisit the `room_selector` agent from the first tutorial:

```python
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

When train this agent with algorithms, the dataset is expected to be a list of `RoomSelectionTask` objects.

```python
from agentlightning import Dataset

dataset: Dataset[RoomSelectionTask] = [
    RoomSelectionTask(date="2025-10-15", time="10:00", duration_min=60, attendees=10),
    RoomSelectionTask(date="2025-10-16", time="10:00", duration_min=60, attendees=10),
]

Trainer().fit(agent=room_selector, train_dataset=dataset)
```

Behind the scenes, the [`@rollout`][agentlightning.rollout] decorator wraps your function in a `FunctionalLitAgent` object, which is a subclass of [LitAgent][agentlightning.LitAgent] introduced below, making it compatible with the [Trainer][agentlightning.Trainer] and [Runner][agentlightning.BaseRunner]. It supports parameters like `task`, `prompt_template`, `llm`, and `rollout`, giving you flexible access to the execution context. The return value is automatically converted into a final reward span.

Here is another example with more advanced usages with `llm` and `rollout` as parameters. The `llm` parameter gives you an OpenAI-compatible LLM endpoint to interact with, which can be tuned under the hood by algorithms. The `rollout` parameter gives you a full rollout object, which contains the rollout ID, rollout mode (training or validation), etc.

```python
from agentlightning.types import LLM, Rollouts

@rollout
def flight_assistant(task: FlightBookingTask, llm: LLM, rollout: Rollout) -> float:
    print(f"Rollout ID: {rollout.rollout_id}")
    print(f"Rollout Mode: {rollout.mode}")

    # TODO: show how to use the LLM
```

## Return Values from Agents

TODO: explain the return values from agents.


TODO: The rest of the tutorial has not been reviewed yet. Please review on your own.

## Class-based Agents

For more complex agents that require state, helper methods, or distinct logic for training versus validation, you can create a class that inherits from `LitAgent`. This object-oriented approach provides more structure and control over the agent's lifecycle.

To create a class-based agent, you subclass `agentlightning.agent.LitAgent` and implement the `rollout` method.

Here's how the `room_selector` could be implemented as a class:

```python
from agentlightning.agent import LitAgent
from agentlightning.types import NamedResources, Rollout

class RoomSelectorAgent(LitAgent[RoomSelectionTask]):
    def rollout(self, task: RoomSelectionTask, resources: NamedResources, rollout: Rollout) -> float:
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

  - `rollout()`: The primary method for the agent's logic. It's called for both training and validation by default.
  - `training_rollout()` / `validation_rollout()`: Implement these if you need different behavior during training (e.g., with exploration) and validation (e.g., with deterministic choices).
  - `rollout_async()` / `training_rollout_async()` / `validation_rollout_async()`: Implement the asynchronous versions of these methods if your agent uses `asyncio`.

-----

## Using the Emitter

While returning a single float for the final reward is sufficient for many algorithms, some advanced scenarios require richer feedback. For instance, an algorithm might learn more effectively if it receives intermediate rewards throughout a multi-step task.

Agent-lightning provides an **emitter** module that allows you to record custom spans from within your agent's logic. Remember, the `Tracer` automatically instruments many common operations (like LLM calls), but the emitter is for your own, domain-specific events.

You can import the emitter functions from `agentlightning.emitter`.

### Emitting Rewards, Messages, and More

Here are the primary emitter functions:

  - `emit_reward(value: float)`: Records an intermediate reward.
  - `emit_message(message: str)`: Records a simple log message as a span.
  - `emit_exception(exception: BaseException)`: Records a Python exception, including its type, message, and stack trace.
  - `emit_object(obj: Any)`: Records any JSON-serializable object, perfect for structured data.

Let's see an example of an agent using these emitters to provide detailed feedback.

```python
from agentlightning import rollout
from agentlightning.emitter import emit_reward, emit_message, emit_exception, emit_object

@rollout
def multi_step_agent(task: dict, prompt_template: PromptTemplate) -> float:
    try:
        # Step 1: Initial planning
        emit_message("Starting planning phase.")
        plan = generate_plan(task, prompt_template)
        emit_object({"plan_steps": len(plan), "first_step": plan[0]})

        # Award a small reward for a valid plan
        plan_reward = grade_plan(plan)
        emit_reward(plan_reward)

        # Step 2: Execute the plan
        emit_message(f"Executing {len(plan)}-step plan.")
        execution_result = execute_plan(plan)

        # Step 3: Final evaluation
        final_reward = grade_final_result(execution_result, task["expected_output"])

        # The return value is treated as the final reward for the rollout
        return final_reward

    except ValueError as e:
        # Record the specific error and return a failure reward
        emit_exception(e)
        return 0.0
```

By using the emitter, you create a rich, detailed trace of your agent's execution. This data can be invaluable for debugging and is essential for advanced algorithms that can learn from more than just a single final score.
