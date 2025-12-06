# Minimal Example: Custom Tracer & Adapter

A minimal  Agent-Lightning example demonstrating **custom observability patterns** through tracer and adapter interfaces.


 this example shows you how to:

1. **Create Custom Tracers** - Integrate Agent-Lightning with your observability platform (DataDog, New Relic, internal systems)
2. **Build Custom Adapters** - Transform traces into rewards using your own logic
3. **Minimal Setup** - Complete working example in a single file

## When To Use This Pattern

Use custom tracers/adapters when you need to:

- ✅ Integrate with existing observability infrastructure
- ✅ Compute rewards from multiple signals (latency, tokens, cost, correctness)
- ✅ Export traces to custom analytics pipelines
- ✅ Debug agent behavior with fine-grained instrumentation

## Quick Start

### Debug Mode (No Training)

Test the agent and see custom traces:

```bash
python app.py 
```

This runs the agent on sample tasks and prints captured spans in your custom format.

### Training Mode

Run full RL training with custom observability:

```bash
# Start Ray cluster
bash ../../scripts/restart_ray.sh

# Train
python app.py --train
```

## Code Structure (~180 lines)

```python
# 1. Custom Tracer 
class CustomTracer(agl.Tracer):
    def start_span(self, name: str, **attrs): ...
    def end_span(self, span): ...
    def add_event(self, name: str, **attrs): ...

# 2. Custom Adapter 
class CustomAdapter(agl.Adapter):
    def extract(self, trace) -> agl.Triplet: ...
    def _compute_reward(self, span) -> float: ...

# 3. Agent 
@agl.rollout
async def simple_math_agent(task, llm): ...

# 4. Training/Debug 
def train_mode(): ...
def debug_mode(): ...
```

## Custom Tracer Pattern

```python
class CustomTracer(agl.Tracer):
    """Captures spans in your preferred format."""
    
    def start_span(self, name: str, **attributes):
        # Your instrumentation logic
        self.current_span = CustomSpan(name, attributes)
        return self.current_span
    
    def add_event(self, name: str, **attributes):
        # Log events during execution
        self.current_span.events.append({...})
```

**Use cases:**
- Send spans to DataDog: `datadog.tracer.start_span()`
- Export to Prometheus: `prometheus_client.Counter(...).inc()`
- Custom logging: Write to your internal systems

## Custom Adapter Pattern

```python
class CustomAdapter(agl.Adapter):
    """Transforms traces into rewards."""
    
    def extract(self, trace) -> agl.Triplet:
        prompt = trace.attributes["prompt"]
        response = trace.events[-1]["content"]
        
        # Multi-signal reward
        reward = self._compute_reward(trace)
        
        return agl.Triplet(prompt, response, reward)
    
    def _compute_reward(self, span):
        # Combine multiple signals
        correctness = span.attributes["correct"]
        latency = span.attributes["latency"]
        tokens = span.attributes["tokens"]
        
        return correctness - 0.1 * (latency > 10) + 0.05 * (tokens < 500)
```

**Use cases:**

- Aggregate metrics from multiple sources
- Apply domain-specific reward shaping
- Incorporate business metrics (cost, user satisfaction)

## Extending This Example

### 1. Add Real Observability Platform

```python
import datadog

class DataDogTracer(agl.Tracer):
    def start_span(self, name: str, **attrs):
        return datadog.tracer.trace(name, **attrs)
```

### 2. Multi-Signal Rewards

```python
class BusinessAdapter(agl.Adapter):
    def _compute_reward(self, span):
        correctness = span.attributes["correct"]
        cost = span.attributes["api_cost"]
        latency = span.attributes["latency"]
        
        # Business objective: correct, cheap, fast
        return correctness - 0.5 * cost - 0.1 * latency
```

### 3. Async Event Streaming

```python
class StreamingTracer(agl.Tracer):
    async def add_event(self, name: str, **attrs):
        await kafka_producer.send("agent-events", {...})
```
