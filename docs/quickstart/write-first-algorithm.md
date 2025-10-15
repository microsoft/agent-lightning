# Write the First Algorithm with Agent-lightning

In the [first tutorial](./train-first-agent.md), "Train the First Agent," we introduced the Trainer and showed how to use a pre-built algorithm like **Automatic Prompt Optimization (APO)** to improve an agent's performance. The Trainer handled all the complex interactions, letting us focus on the agent's logic.

Now, we'll go a step deeper. What if you have a unique training idea that doesn't fit a standard algorithm? This tutorial will show you how to write your own custom algorithm from scratch. We'll build a simple algorithm that systematically tests a list of prompt templates and identifies the one with the highest reward.

By the end, you'll understand the core mechanics of how the Algorithm, Runner, and a new component (**Store**), working together to create the powerful training loop at the heart of Agent-lightning.

!!! tip

    This tutorial helps you build a basic understanding of how to interact with Agent-lightning's core components. It's recommended that all users customizing algorithms should read this tutorial, even for those who are not planning to do prompt optimization.

## The Central Hub: The LightningStore

In the last tutorial, we simplified the training loop, saying the Algorithm and Agent communicate "via the Trainer." That's true at a high level, but the component that makes it all possible is the **LightningStore**.

The LightningStore acts as the central database and message queue for the entire system. It's the single source of truth that decouples the Algorithm from the Runners.

<!-- You should explain what's resources before introducing the store. I believe that's not covered in the last tutorial. -->

The Algorithm connects to the Store to `enqueue_rollout` (tasks) and `update_resources` (like prompt templates). It also queries the Store to retrieve the results (spans and rewards) from completed rollouts.

The Runners connect to the Store to `dequeue_rollout` (polling for available tasks). After executing a task, they write the resulting spans and status updates back to the Store.

This architecture is key to Agent-lightning's scalability. Since the Algorithm and Runners only talk to the Store, they can run in different processes or even on different machines without knowing about each other.

![Store Architecture](../)

<!-- talk about that the store auto handles attempts, retries, and maintain task queue, rollouts, spans, resources. --->
<!-- we should also talk about what is tracer -- it instruments the calls to llm and log spans. Our default tracer is based on AgentOps SDK. -->
!!! tip "A Mental Model on What Store has Stored"

    TBD

## Building a Custom Algorithm

Let's build an algorithm that finds the best system prompt from a predefined list. The logic is straightforward:

1. Start with a list of candidate prompt templates.
2. For each template, create a "resource" bundle.
3. Enqueue a rollout (a task) in the Store, telling the Runner to use this specific resource.
4. Wait for a Runner to pick up the task and complete it.
5. Query the Store to get the final reward for that rollout.
6. After testing all templates, compare the rewards and declare the best one.

<!-- the implementation should be a pure python function that interacts with the store, not class based. -->

!!! note "Asynchronous Operations"

    You'll notice the `async` and `await` keywords. Agent-lightning is built on asyncio to handle concurrent operations efficiently. All interactions with the store are asynchronous network calls, so they must be awaited.

## Agent and Runner

Our algorithm needs an agent to execute the tasks. For this example, the agent's job is simple: take the prompt from the resources, use it to ask an LLM a question, and return a random score from 0-1 (which can be optionally replaced with another LLM call (a "judge") to score the response, as shown in the [full sample code]({{ config.repo_url }}/tree/{{ config.extra.source_commit }}/examples/apo/apo_custom_algorithm.py)).

The code is very similar to what we have seen in the [last tutorial](./train-first-agent.md):

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
    # This llm_output is automatically logged as a span by the runner
    score = random.uniform(0, 1)  # Replace with actual scoring logic if needed

    return score
```

<!-- Use a runner and tracer to kickoff the iteration. Listen to more tasks, carry out the tasks, and report back to the store. -->

## Running the Example

To see everything in action, you'll need three separate terminal windows.

!!! tip

    If you want to follow along, you can find the complete code for this example in the [apo_custom_algorithm.py]({{ config.repo_url }}/tree/{{ config.extra.source_commit }}/examples/apo/apo_custom_algorithm.py) file.

1. **Start the Store:** In the first terminal, start the LightningStore server. This component will wait for connections from the algorithm and the runner. The store will be listening to `4747`⚡ by default.

```bash
agl store
```

2. **Start the Runner:** In the second terminal, start the runner process. It will connect to the store and wait for tasks.

```bash
export OPENAI_API_KEY=sk-...  # Pointing to an valid OpenAI API key
python apo_custom_algorithm.py runner
```

You will see output indicating the runner has started and is waiting for rollouts.

```text
2025-10-14 22:23:41,339 [INFO] ... [Worker 0] Setting up tracer...
2025-10-14 22:23:41,343 [INFO] ... [Worker 0] Instrumentation applied.
2025-10-14 22:23:41,494 [INFO] ... [Worker 0] AgentOps client initialized.
2025-10-14 22:23:41,494 [INFO] ... [Worker 0] Started async rollouts (max: unlimited).
```

3. **Start the Algorithm:** In the third terminal, run the algorithm. This will kick off the entire process.

```bash
python apo_custom_algorithm.py algo
```

### Understanding the Output

As the algorithm runs, you'll see logs appear across all three terminals, showing the components interacting in real-time.

**Algorithm Output:** The algorithm terminal shows the main control flow: updating prompts, queuing tasks, and receiving the final results.

```text
[Algo] Updating prompt template to: 'You are a helpful assistant. {any_question}'
[Algo] Queuing task for clients...
[Algo] Task 'ro-1d18988581cd' is now available for clients.
[Algo] Received Result: rollout_id='ro-1d18988581cd' ... status='succeeded' ...
[Algo] Final reward: 0.95
<!-- show at least spans for one rollout here -->
[Algo] Updating prompt template to: 'You are a knowledgeable AI. {any_question}'
...
[Algo] Final reward: 0.95

[Algo] Updating prompt template to: 'You are a friendly chatbot. {any_question}'
...
[Algo] Final reward: 1.0

[Algo] All prompts and their rewards: [('You are a helpful assistant. {any_question}', 0.95), ('You are a knowledgeable AI. {any_question}', 0.95), ('You are a friendly chatbot. {any_question}', 1.0)]
[Algo] Best prompt found: 'You are a friendly chatbot. {any_question}' with reward 1.0
```

**Runner Output:** The runner terminal shows it picking up each task, executing the agent logic, and reporting the completion.

```text
[Rollout] LLM returned: The sky appears blue due to Rayleigh scattering...
[Judge] Judge returned score: 0.95
2025-10-14 22:25:50,803 [INFO] ... [Worker 0 | Rollout ro-a9f54ac19af5] Completed in 4.24s. ...

[Rollout] LLM returned: The sky looks blue because of a process called Rayleigh scattering...
[Judge] Judge returned score: 1.0
2025-10-14 22:25:59,863 [INFO] ... [Worker 0 | Rollout ro-c67eaa9016b6] Completed in 4.06s. ...
```

**Store Server Output:** The store terminal shows a detailed log of every interaction, confirming its role as the central hub. You can see requests to enqueue and dequeue rollouts, add spans, and update statuses.

```text
... "POST /enqueue_rollout HTTP/1.1" 200 ...
... "GET /dequeue_rollout HTTP/1.1" 200 ...
... "POST /add_span HTTP/1.1" 200 ...
... "POST /update_attempt HTTP/1.1" 200 ...
... "POST /wait_for_rollouts HTTP/1.1" 200 ...
... "GET /query_spans/ro-c67eaa9016b6 HTTP/1.1" 200 ...
```
