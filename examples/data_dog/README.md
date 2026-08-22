# DataDog Integration Example

Complete end-to-end example showing how to integrate Agent-Lightning with DataDog APM for  observability.

## What This Demonstrates

1. **Custom DataDog Tracer** - Sends all agent spans to DataDog APM
2. **Multi-Signal Rewards** - Optimizes for correctness + latency + token efficiency
3. **Production Monitoring** - Real-time visibility into agent training

## Prerequisites

```bash
# Install dependencies
pip install agentlightning ddtrace openai

# Setup DataDog
export DD_API_KEY=your_datadog_api_key
export DD_SITE=datadoghq.com  # or datadoghq.eu for EU
export DD_SERVICE=agent-lightning-demo
```

Get your DataDog API key from: https://app.datadoghq.com/organization-settings/api-keys

## Quick Start

### 1. Debug Mode (Test Without Training)

```bash
# Start a local LLM endpoint first (e.g., vLLM)
# Then run:
python datadog_agent.py --debug
```

This runs 3 sample tasks and sends spans to DataDog. Check your DataDog APM dashboard to see the traces.

### 2. Training Mode (Full RL Training)

```bash
# Start Ray cluster
bash ../../scripts/restart_ray.sh

# Run training
python datadog_agent.py --train
```

This trains the agent with VERL and sends all spans to DataDog for monitoring.

## What You'll See in DataDog

### APM Traces
Navigate to: https://app.datadoghq.com/apm/traces

You'll see traces for each agent rollout with:
- **Service**: `agent-lightning-demo` or `agent-lightning-training`
- **Resource**: Individual task IDs
- **Tags**: Custom attributes like `task_id`, `reward`, `latency`, `tokens`

### Custom Metrics
- `agent.reward` - Reward value for each rollout
- `duration_seconds` - Time taken for each rollout

### Event Tags
- `event.reward` - Reward emission events
- `event.error` - Error events with messages

## Code Structure

```python
# 1. DataDog Tracer (sends spans to DataDog)
class DataDogTracer(agl.Tracer):
    def start_span(self, name, **attrs):
        return dd_tracer.trace(name, service=self.service)
    
    def end_span(self, span):
        span.finish()  # Sends to DataDog
    
    def add_event(self, name, **attrs):
        span.set_tag(f"event.{name}", attrs)

# 2. Multi-Signal Adapter (reward shaping)
class MultiSignalAdapter(agl.TraceAdapter[List[agl.Triplet]]):
    def adapt(self, spans: List[ReadableSpan]) -> List[agl.Triplet]:
        # Extract prompt, response, reward
        # Apply multi-signal reward:
        reward = correctness - 0.1*(latency>5) + 0.05*(tokens<300)
        return [agl.Triplet(prompt, response, reward)]

# 3. Simple Math Agent
@agl.rollout
async def math_agent(task, llm):
    # Solve math problem
    agl.emit_reward(reward)  # Triggers tracer.add_event()

# 4. Training with DataDog
trainer = agl.Trainer(
    algorithm=agl.VERL(config),
    tracer=DataDogTracer(),
    adapter=MultiSignalAdapter(),
)
trainer.fit(math_agent, train_data)
```

## Multi-Signal Reward Shaping

The adapter computes rewards from three signals:

| Signal | Weight | Purpose |
|--------|--------|---------|
| **Correctness** | Base (0.0 or 1.0) | Did the agent get the right answer? |
| **Latency** | -0.1 if >5s | Penalty for slow responses |
| **Tokens** | +0.05 if <300 | Bonus for concise answers |

**Formula**: `reward = correctness - 0.1*(latency>5s) + 0.05*(tokens<300)`

This encourages the agent to be accurate, fast, AND efficient.



## Some troubleshooting

### No traces in DataDog

1. Check API key: `echo $DD_API_KEY`
2. Check site: `echo $DD_SITE` (should be `datadoghq.com` or `datadoghq.eu`)
3. Verify ddtrace installed: `pip show ddtrace`



###  test locally without DataDog

Comment out the DataDog tracer and use default:

```python
# tracer = DataDogTracer()  # Comment this
tracer = agl.OtelTracer()  # Use default instead
```

## Production Deployment

For production use:

1. **Set DD_ENV**: `export DD_ENV=production`
2. **Set DD_VERSION**: `export DD_VERSION=1.0.0`
3. **Enable profiling**: Add `DD_PROFILING_ENABLED=true`
4. **Use tags**: Add `DD_TAGS=team:ml,project:agents`

Example  setup:

```bash
export DD_API_KEY=your_key
export DD_SITE=datadoghq.com
export DD_SERVICE=agent-lightning
export DD_ENV=production
export DD_VERSION=1.2.3
export DD_TAGS=team:ml-platform,project:math-agents
export DD_PROFILING_ENABLED=true

python datadog_agent.py --train
```

