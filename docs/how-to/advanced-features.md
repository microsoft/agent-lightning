# Advanced Features

## Custom Agents

Create custom agents by extending the base Agent class:

```python
from agentlightning import Agent

class CustomAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
    
    def custom_method(self):
        # Your custom logic
        pass
```

## Monitoring

Use built-in monitoring tools:

```python
from agentlightning import Monitor

monitor = Monitor(agent)
monitor.start()

# Run your agent tasks
agent.run("Task")

# Get metrics
metrics = monitor.get_metrics()
```

## Debugging

Enable debug mode for detailed logging:

```python
agent = Agent(debug=True)
```