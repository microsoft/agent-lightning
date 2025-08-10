# Basic Usage

## Creating an Agent

```python
from agentlightning import Agent

agent = Agent(
    name="my_agent",
    config={
        "model": "gpt-4",
        "temperature": 0.7
    }
)
```

## Training

```python
# Train with default settings
agent.train()

# Train with custom parameters
agent.train(
    epochs=10,
    batch_size=32
)
```

## Running Tasks

```python
# Simple task
result = agent.run("Analyze this text")

# Task with context
result = agent.run(
    task="Analyze this text",
    context={"additional_info": "value"}
)
```