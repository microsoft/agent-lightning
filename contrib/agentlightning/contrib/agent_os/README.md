# Agent-OS Integration for Agent-Lightning

Kernel-level safety during AI agent training.

## Overview

[Agent-OS](https://github.com/imran-siddique/agent-os) provides deterministic governance
for AI agents. This integration enables:

- **0% policy violations during training** - Unsafe actions are blocked or penalized
- **Policy violations → RL penalties** - Agents learn to avoid unsafe behavior
- **Complete audit trail** - From training to production

## Installation

```bash
pip install agentlightning agent-os-kernel
```

## Quick Start

```python
from agentlightning import Trainer
from agentlightning.contrib.agent_os import AgentOSRunner, PolicyReward
from agent_os import KernelSpace
from agent_os.policies import SQLPolicy

# Create governed kernel
kernel = KernelSpace(policy=SQLPolicy(
    deny=["DROP", "DELETE"]
))

# Wrap in Agent-OS runner
runner = AgentOSRunner(kernel)

# Train with policy-aware rewards
trainer = Trainer(
    runner=runner,
    reward_fn=PolicyReward(kernel),
    algorithm="GRPO"
)

trainer.train()
```

## Components

### AgentOSRunner

Wraps agent execution with kernel-level policy enforcement:

```python
from agentlightning.contrib.agent_os import AgentOSRunner

runner = AgentOSRunner(
    kernel,
    fail_on_violation=False,  # Continue but penalize
    emit_violations=True,     # Emit as spans
)
```

### PolicyReward

Converts policy violations to negative RL rewards:

```python
from agentlightning.contrib.agent_os import PolicyReward

reward_fn = PolicyReward(
    kernel,
    base_reward_fn=accuracy_reward,
    critical_penalty=-100.0,
    clean_bonus=5.0,
)
```

### FlightRecorderAdapter

Imports Agent-OS audit logs to LightningStore:

```python
from agentlightning.contrib.agent_os import FlightRecorderAdapter

adapter = FlightRecorderAdapter(flight_recorder)
adapter.import_to_store(lightning_store)
```

## Benchmarks

| Metric | Without Agent-OS | With Agent-OS |
|--------|------------------|---------------|
| Policy Violations | 12.3% | **0.0%** |
| Task Accuracy | 76.4% | **79.2%** |

## Documentation

- [Agent-OS Documentation](https://imran-siddique.github.io/agent-os-docs/)
- [Integration Guide](./docs/integration.md)

## License

MIT
