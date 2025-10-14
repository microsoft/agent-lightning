# Getting Started

This guide walks you through building your first Agent Lightning application - a simple prompt optimization system that finds the best system prompt for an AI agent.

## What You'll Build

You'll create a distributed training system with a server that manages optimization algorithms and tasks, a client with multiple workers that execute tasks in parallel, and built-in telemetry for monitoring and debugging.

Before starting, ensure you have Python 3.10 or later, Agent Lightning installed (`pip install agentlightning`), and an OpenAI API key. The complete code is available in the [examples/apo]({{ config.repo_url }}/tree/{{ config.extra.source_commit }}/examples/apo) directory.

## Part 1: Building Your Agent

Let's start by creating a simple agent that can answer questions using OpenAI's API. Your agent needs to inherit from `LitAgent` and implement a `training_rollout` method.

### Step 1: Create Your Agent Class

First, import the necessary dependencies and create your agent class:

```python
from agentlightning.litagent import LitAgent

class SimpleAgent(LitAgent):
    def training_rollout(self, task, rollout_id, resources):
        """Execute a single training rollout."""
```

The `training_rollout` method is the heart of your agent. It receives three parameters: a `task` dictionary containing the work to do (like "What is the capital of France?"), a unique `rollout_id` for tracking this execution, and `resources` from the server - in our case, the system prompt we're testing.

### Step 2: Execute the Task

Inside the training_rollout method, extract the system prompt from resources and use it to complete the task:

```python
# Extract the system prompt being tested
system_prompt = resources["system_prompt"].template

# Call OpenAI with this prompt
result = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["prompt"]},
    ],
)
```

The server sends different prompts to test, and your agent uses each one to answer the same question. This lets us compare which prompt works best.

### Step 3: Return a Reward Score

After executing the task, return a reward score between 0 and 1:

```python
# In real scenarios, calculate based on response quality
return random.uniform(0, 1)
```

Higher rewards mean better performance. In a real system, you'd calculate this with rules, or even an LLM as a judge. For now, we're using random values to demonstrate the flow.

### Step 4: Set Up the Trainer

To run your agent with multiple workers in parallel:

```python
from agentlightning.trainer import Trainer

agent = SimpleAgent()
trainer = Trainer(n_workers=2)  # Create 2 parallel workers
trainer.fit(agent, "http://127.0.0.1:9997")
```

The trainer creates separate processes for each worker, allowing them to execute tasks independently. This parallelization significantly speeds up the optimization process - with 2 workers, you can test prompts twice as fast.

## Part 2: Building the Optimization Server

The server coordinates the training process and implements your optimization algorithm. It manages resources, distributes tasks, and collects results.

### Step 1: Initialize the Server

Create an async function to run your optimization:

```python
import asyncio
from agentlightning.server import AgentLightningServer
from agentlightning.types import PromptTemplate

async def prompt_optimization():
    server = AgentLightningServer(host="127.0.0.1", port=9997)
    await server.start()
```

We use async/await because the server handles multiple clients simultaneously. This allows it to queue tasks without blocking and process results as they arrive from different workers.

### Step 2: Test Different Prompts

Define the prompts you want to test and iterate through them:

```python
prompt_candidates = [
    "You are a helpful assistant.",
    "You are a knowledgeable AI.",
    "You are a friendly chatbot.",
]

for prompt in prompt_candidates:
    # Send this prompt to all connected clients
    resources = {
        "system_prompt": PromptTemplate(template=prompt, engine="f-string")
    }
    await server.update_resources(resources)
```

When you update resources, all connected clients immediately receive the new system prompt. The format of resources can be arbitrary. We use the key `"system_prompt"` here as an example. The resources here are exactly you would expect at the client side, who will use this prompt for the next task they process.

### Step 3: Queue Tasks and Collect Results

For each prompt, queue a task and wait for results. The `{"prompt": ...}` format here is exact what you would expect from the client side code.

```python
# Queue a task for clients to process
task_id = await server.queue_task(
    sample={"prompt": "What is the capital of France?"},
    mode="train"
)

# Wait for a client to complete it (30 second timeout)
rollout = await server.poll_completed_rollout(task_id, timeout=30)

# Extract and store the reward (this comes from the return value of the client side)
reward = rollout.final_reward
```

The server queues the same question for each prompt. The rollout object contains not just the reward, but also detailed telemetry and trace information for debugging and optimization.

### Step 4: Find the Best Prompt

After testing all candidates, identify the winner:

```python
best_prompt = max(prompt_and_rewards, key=lambda x: x[1])
print(f"Best prompt: '{best_prompt[0]}' with reward {best_prompt[1]:.3f}")
```

## Running Your System

The [Complete example code]({{ config.repo_url }}/tree/{{ config.extra.source_commit }}/examples/apo) can be found on the GitHub repository. To run it:

Create a `.env` file with your API credentials:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional
```

Start the server first in one terminal:
```bash
python server.py
```

Then start the client in another terminal:
```bash
python client.py
```

## Understanding the Output

When you run the system, you'll see detailed logs from both the client and server. Understanding these logs helps you debug issues and optimize performance.

### Client Output Explained

```
2025-08-10 12:59:38,224 [INFO] Initializing Trainer...
```
The trainer is starting up and preparing to create workers.

```
[INFO] Starting AgentOps local server on port 52081...
```
A local telemetry server starts to collect metrics and traces. You can access this at `http://localhost:52081` to see detailed execution traces.

```
[INFO] Starting worker process 0...
[INFO] Starting worker process 1...
```
Two separate processes are created. Each can execute tasks independently, doubling your throughput.

```
[INFO] [Task 1 Received] ID: rollout-c1eb987b...
Resources: {'system_prompt': PromptTemplate(...)}
```
A worker receives a task from the server along with the current prompt to test.

```
[INFO] [Worker 0 | Rollout] Completed in 1.09s. Reward: 0.631
```
Worker 0 finished executing the task in 1.09 seconds and calculated a reward of 0.631. This tells you both performance (execution time) and quality (reward score).

### Server Output Explained

```
[Algo] Testing prompt: 'You are a helpful assistant.'
```
The optimization algorithm selects the next prompt to test.

```
[Algo] Task 'rollout-c1eb987b...' is now available for clients.
```
The task is queued and waiting for an available worker to pick it up.

```
[Algo] Received reward: 0.631
```
A client completed the task and returned a performance score. The server uses this to compare prompts.

```
[Algo] Best prompt: 'You are a knowledgeable AI.' (reward: 0.925)
```
After testing all prompts, the server identifies which one performed best.

## What's Happening Behind the Scenes

Agent Lightning handles several complex operations automatically. Multiple workers process tasks simultaneously. Every API call and execution is tracked through the telemetry tracing system, providing detailed traces for debugging and optimization. If a worker fails, others continue processing, ensuring your optimization doesn't stop.

The AgentOps tracer, enabled by default, collects comprehensive data about each execution, including API calls, timing information, token usage, and error traces. The data is sent to the server and can be accessed via `rollout.triplets` and `rollout.traces` at the server side to build more advanced automatic optimization algorithms.

## Next Steps

Now that you have a working system, how about:

- Replacing the random reward with actual quality metrics based on real response accuracy?
- Testing the system prompt on a batch of different questions to see how it performs across various tasks?
- Making the algorithm automatically improve the system prompt based on the best-performing ones?
- Setting up a real agent system that consists of multiple prompts, and optimizing them together?
