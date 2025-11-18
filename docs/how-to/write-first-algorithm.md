# Write the First Algorithm with Agent-lightning

In the [first tutorial](./train-first-agent.md), "Train the First Agent," we introduced the [Trainer][agentlightning.Trainer] and showed how to use a pre-built algorithm like **Automatic Prompt Optimization (APO)** to improve an agent's performance. The [Trainer][agentlightning.Trainer] handled all the complex interactions, letting us focus on the agent's logic.

Now, we'll go a step deeper. What if you have a unique training idea that doesn't fit a standard algorithm? This tutorial will show you how to write your own custom algorithm from scratch. We'll build a simple algorithm that systematically tests a list of prompt templates and identifies the one with the highest reward.

By the end, you'll understand the core mechanics of how the **Algorithm**, **Runner**, and a new component, the **Store**, work together to create the powerful training loop at the heart of Agent-lightning.

!!! tip
    This tutorial helps you build a basic understanding of how to interact with Agent-lightning's core components. It's recommended that all users customizing algorithms should read this tutorial, even for those who are not planning to do prompt optimization.

## Core Concepts for Training

Before diving into the [LightningStore][agentlightning.LightningStore], let's define two key concepts that are central to any training process in Agent-lightning: **Resources** and the **Tracer**.

### Resources: The Tunable Assets

[](){ #introduction-to-resources }

**Resources** are the assets your algorithm is trying to improve. Think of them as the "recipe" an agent uses to perform its task. This recipe can be:

  * A **prompt template** that guides an LLM.
  * The **weights** of a machine learning model.
  * Any other configuration or data your agent needs.

The algorithm's job is to run experiments and iteratively update these resources to find the best-performing version.

### Tracer: The Data Collector

How does the algorithm know if a change was an improvement? It needs data. This is where the **Tracer** comes in.

The Tracer automatically **instruments** (aka modifies / patches) the agent's code. This means it watches for important events, like an LLM call, a tool being used, or **reward signals**, and records a detailed log of what happened. Each of these logs is called a **Span** (which has already been introduced in the [last tutorial](./train-first-agent.md)).

A collection of spans from a single task execution gives the algorithm a complete, step-by-step trace of the agent's behavior, which is essential for learning and making improvements. Our default tracer is built on the AgentOps SDK to support instrumenting code written in various Agent/non-agent frameworks.

## The Central Hub: The LightningStore

Now, where do all these resources, tasks, and spans live? They are all managed by the **LightningStore**.

The LightningStore acts as the central database and message queue for the entire system. It's the single source of truth that decouples the Algorithm from the Runners.

!!! note

    In the [last tutorial](./train-first-agent.md) we simplified the training loop, saying the Algorithm and Agent communicate "via the Trainer." That's true at a high level, but the component that makes it all possible is actually the **LightningStore**.

* The **Algorithm** connects to the Store to `enqueue_rollout` (tasks) and `update_resources` (like new prompt templates). It also queries the Store to retrieve the resulting spans and rewards from completed rollouts.
* The **Runners** connect to the Store to `dequeue_rollout` (polling for available tasks). After executing a task, they use the `Tracer` to write the resulting spans and status updates back to the Store.

This architecture is key to Agent-lightning's scalability. Since the Algorithm and Runners only talk to the Store, they can run in different processes or even on different machines.

![Store Architecture](../assets/store-api-visualized.svg){ .center }

!!! tip "A Mental Model of What the Store Contains"

    The [LightningStore][agentlightning.LightningStore] isn't just a simple database; it's an organized system for managing the entire training lifecycle. Here's what it keeps track of:

    * **Task Queue**: A queue of pending **Rollouts** waiting for a Runner to pick them up, interactable via `enqueue_rollout` and `dequeue_rollout`.
    * **Rollouts**: The record of a single task. A rollout contains metadata about the task and tracks all **Attempts** to complete it, interactable via `query_rollouts` and `wait_for_rollouts`.
    * **Attempts**: A single execution of a rollout. If an attempt fails (e.g., due to a network error), the Store can automatically schedules a retry if it's configured. Each attempt is linked to its parent rollout and contains the status and timing information. The rollout status is [synced](../deep-dive/store.md) with its children's status. **For beginners, you can assume each rollout has only one attempt unless you have explicitly configure the retry.**
    * **Spans**: The detailed, structured logs generated by the `Tracer` during an attempt. Each span is linked to its parent attempt and rollout.
    * **Resources**: A versioned collection of the assets (like prompt templates) that the algorithm creates. Each rollout is linked to the specific version of the resources it should use.

## Building a Custom Algorithm

Let's build an algorithm that finds the best system prompt from a predefined list. The logic is straightforward:

1.  Start with a list of candidate prompt templates.
2.  For each template, create a "resource" bundle in the Store.
3.  Enqueue a rollout (a task), telling the Runner to use this specific resource.
4.  Wait for a Runner to pick up the task and complete it.
5.  Query the Store to get the final reward from the rollout's spans.
6.  After testing all templates, compare the rewards and declare the best one.

We can implement this as a simple Python function that interacts directly with the [LightningStore][agentlightning.LightningStore].

```python
async def find_best_prompt(store, prompts_to_test, task_input):
    """A simple algorithm to find the best prompt from a list."""
    results = []

    # Iterate through each prompt to test it
    for prompt in prompts_to_test:
        print(f"[Algo] Updating prompt template to: '{prompt}'")

        # 1. Update the resources in the store with the new prompt
        resources_update = await store.add_resources(
            resources={"prompt_template": prompt}
        )

        # 2. Enqueue a rollout task for a runner to execute
        print("[Algo] Queuing task for clients...")
        rollout = await store.enqueue_rollout(
            input=task_input,
            resources_id=resources_update.resources_id,
        )
        print(f"[Algo] Task '{rollout.rollout_id}' is now available for clients.")

        # 3. Wait for the rollout to be completed by a runner
        await store.wait_for_rollouts([rollout.rollout_id])

        # 4. Query the completed rollout and its spans
        completed_rollout = await store.get_rollout_by_id(rollout.rollout_id)
        print(f"[Algo] Received Result: {completed_rollout.model_dump_json(indent=None)}")

        spans = await store.query_spans(rollout.rollout_id)
        # We expect at least two spans: one for the LLM call and one for the final reward
        print(f"[Algo] Queried Spans:\n  - " + "\n  - ".join(str(span) for span in spans))
        # find_final_reward is a helper function to extract the reward span
        final_reward = find_final_reward(spans)
        print(f"[Algo] Final reward: {final_reward}\n")

        results.append((prompt, final_reward))

    # 5. Find and print the best prompt based on the collected rewards
    print(f"[Algo] All prompts and their rewards: {results}")
    best_prompt, best_reward = max(results, key=lambda item: item[1])
    print(f"[Algo] Best prompt found: '{best_prompt}' with reward {best_reward}")
```

!!! note "Asynchronous Operations"

    You'll notice the `async` and `await` keywords. Agent-lightning is built on asyncio to handle concurrent operations efficiently. All interactions with the store are asynchronous network calls, so they must be awaited.

## The Agent and Runner

Our algorithm needs an **agent** to execute the tasks and a **runner** to manage the process.

The runner is a long-lived worker process. Its job is simple:

1.  Connect to the [LightningStore][agentlightning.LightningStore] via a [LightningStoreClient][agentlightning.LightningStoreClient].
2.  Enter a loop, constantly asking the [LightningStore][agentlightning.LightningStore] for new tasks (`dequeue_rollout`).
3.  When it gets a task, it runs the `simple_agent` function.
4.  Crucially, the runner wraps the agent execution with a **Tracer**. The tracer automatically captures all the important events (like the LLM call and the final reward) as spans and sends them back to the [LightningStore][agentlightning.LightningStore].

```python
# Connecting to Store
store = agl.LightningStoreClient("http://localhost:4747")  # or some other address
runner = LitAgentRunner[str](tracer=AgentOpsTracer())
with runner.run_context(agent=simple_agent, store=store):  # <-- where the wrapping and instrumentation happens
    await runner.iter()  # polling for new tasks forever
```

For this example, the agent's job is to take the prompt from the resources, use it to ask an LLM a question, and return a score.

```python
def simple_agent(task: str, prompt_template: PromptTemplate) -> float:
    """An agent that answers a question and gets judged by an LLM."""
    client = OpenAI()

    # Generate a response using the provided prompt template
    prompt = prompt_template.format(any_question=task)
    response = client.chat.completions.create(
        model="gpt-4.1-nano", messages=[{"role": "user", "content": prompt}]
    )
    llm_output = response.choices[0].message.content
    print(f"[Rollout] LLM returned: {llm_output}")

    # This llm_output and the final score are automatically logged as spans by the Tracer
    score = random.uniform(0, 1)  # Replace with actual scoring logic if needed
    return score
```

## Running the Example

To see everything in action, you'll need three separate terminal windows.

!!! tip

    If you want to follow along, you can find the complete code for this example in the [apo_custom_algorithm.py]({{ src("examples/apo/apo_custom_algorithm.py") }}) file.

**1. Start the Store:**
In the first terminal, start the LightningStore server. This component will wait for connections from the algorithm and the runner. The store will be listening on port `4747` ⚡ by default.

```bash
agl store
```

**2. Start the Runner:**
In the second terminal, start the runner process. It will connect to the store and wait for tasks.

The code to start the runner looks like the following:

```bash
export OPENAI_API_KEY=sk-... # Your OpenAI API key
python apo_custom_algorithm.py runner
```

You will see output indicating the runner has started and is waiting for rollouts.

```text
2025-10-14 22:23:41,339 [INFO] ... [Worker 0] Setting up tracer...
2025-10-14 22:23:41,343 [INFO] ... [Worker 0] Instrumentation applied.
2025-10-14 22:23:41,494 [INFO] ... [Worker 0] AgentOps client initialized.
2025-10-14 22:23:41,494 [INFO] ... [Worker 0] Started async rollouts (max: unlimited).
```

**3. Start the Algorithm:**
In the third terminal, run the algorithm. This will kick off the entire process.

For example, we run the algorithm code shown above with the following parameters:

```python
prompts_to_test = [
    "You are a helpful assistant. {any_question}",
    "You are a knowledgeable AI. {any_question}",
    "You are a friendly chatbot. {any_question}",
]
task_input = "Why is the sky blue?"
store = agl.LightningStoreClient("http://localhost:4747")
find_best_prompt(store, prompts_to_test, task_input)
```

Or you can simply use our pre-written script to try out:

```bash
python apo_custom_algorithm.py algo
```

### Understanding the Output

As the algorithm runs, you'll see logs appear across all three terminals, showing the components interacting in real-time.

**Algorithm Output:**
The algorithm terminal shows the main control flow: updating prompts, queuing tasks, and receiving the final results. You can also see the raw span data it retrieves from the store.

```text
[Algo] Updating prompt template to: 'You are a helpful assistant. {any_question}'
[Algo] Queuing task for clients...
[Algo] Task 'ro-1d18988581cd' is now available for clients.
[Algo] Received Result: rollout_id='ro-1d18988581cd' ... status='succeeded' ...
[Algo] Queried Spans:
  - Span(name='openai.chat.completion', attributes={'gen_ai.prompt.0.content': 'You are a helpful assistant...', 'gen_ai.completion.0.content': 'The sky appears blue...'})
  - Span(name='reward', attributes={'value': 0.95})
[Algo] Final reward: 0.95

[Algo] Updating prompt template to: 'You are a knowledgeable AI. {any_question}'
...
[Algo] Final reward: 0.95

[Algo] Updating prompt template to: 'You are a friendly chatbot. {any_question}'
...
[Algo] Final reward: 1.0

[Algo] All prompts and their rewards: [('You are a helpful assistant. {any_question}', 0.95), ('You are a knowledgeable AI. {any_question}', 0.95), ('You are a friendly chatbot. {any_question}', 1.0)]
[Algo] Best prompt found: 'You are a friendly chatbot. {any_question}' with reward 1.0
```

**Runner Output:**
The runner terminal shows it picking up each task, executing the agent logic, and reporting the completion.

```text
[Rollout] LLM returned: The sky appears blue due to Rayleigh scattering...
2025-10-14 22:25:50,803 [INFO] ... [Worker 0 | Rollout ro-a9f54ac19af5] Completed in 4.24s. ...

[Rollout] LLM returned: The sky looks blue because of a process called Rayleigh scattering...
2025-10-14 22:25:59,863 [INFO] ... [Worker 0 | Rollout ro-c67eaa9016b6] Completed in 4.06s. ...
```

**Store Server Output:**
The store terminal shows a detailed log of every interaction, confirming its role as the central hub. You can see requests to enqueue and dequeue rollouts, add spans, and update statuses.

```text
... "POST /enqueue_rollout HTTP/1.1" 200 ...
... "GET /dequeue_rollout HTTP/1.1" 200 ...
... "POST /add_span HTTP/1.1" 200 ...
... "POST /update_attempt HTTP/1.1" 200 ...
... "POST /wait_for_rollouts HTTP/1.1" 200 ...
... "GET /query_spans/ro-c67eaa9016b6 HTTP/1.1" 200 ...
```

!!! info "So Where is Trainer?"

    You might be wondering why the [last tutorial](./train-first-agent.md) focused on the [Trainer][agentlightning.Trainer] class, but we haven't used it here.

    Think of the [Trainer][agentlightning.Trainer] as a convenient wrapper that manages the entire training process for you. It's perfect when you want to apply a pre-built algorithm to your agent without worrying about the underlying mechanics. The [Trainer][agentlightning.Trainer] handles starting the [LightningStore][agentlightning.LightningStore], coordinating the [Runners][agentlightning.Runner], managing their lifecycles, and handling errors.

    In this tutorial, however, our goal is to *build a new algorithm*. To do that, we need to interact directly with the core components: the [Store][agentlightning.LightningStore], the [Runner][agentlightning.Runner], and the algorithm logic itself. Running them separately gives you more control and clearer, isolated logs, which is ideal for development and debugging.

    Once your custom algorithm is mature, you can package it to comply with our standard interface ([@algo][agentlightning.algo] or [Algorithm][agentlightning.Algorithm]). This allows you to use it with the [Trainer][agentlightning.Trainer] again, getting all the benefits of automated lifecycle management while using your own custom logic. A sample code doing this is available in [apo_custom_algorithm_trainer.py]({{ src("examples/apo/apo_custom_algorithm_trainer.py") }}).
