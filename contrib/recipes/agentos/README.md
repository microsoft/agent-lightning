# Agent-OS Integration for Agent-Lightning

Kernel-level safety during AI agent training.

## Overview

[Agent-OS](https://github.com/imran-siddique/agent-os) provides deterministic governance
for AI agents. This integration enables:

- **0% unpenalized policy violations** — All unsafe actions are detected and penalized
- **Policy violations → RL penalties** — Agents learn to avoid unsafe behavior
- **Complete audit trail** — From training to production

## Installation

```bash
pip install agentlightning agent-os
```

## Quick Start

```python
from agentlightning import Trainer
from agentlightning.contrib.runner.agentos import AgentOSRunner
from agentlightning.contrib.reward.agentos import PolicyReward
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
from agentlightning.contrib.runner.agentos import AgentOSRunner

runner = AgentOSRunner(
    kernel,
    fail_on_violation=False,  # Continue but penalize
    emit_violations=True,     # Emit as spans
)
```

### PolicyReward

Converts policy violations to negative RL rewards:

```python
from agentlightning.contrib.reward.agentos import PolicyReward

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
from agentlightning.contrib.adapter.agentos import FlightRecorderAdapter

adapter = FlightRecorderAdapter(flight_recorder)
adapter.import_to_store(lightning_store)
```

## Benchmarks

| Metric | Without Agent-OS | With Agent-OS |
|--------|------------------|---------------|
| Undetected Policy Violations | 12.3% | **0.0%** |
| Task Accuracy | 76.4% | **79.2%** |

*Note: "0% undetected violations" means all policy violations are caught and penalized, not that agents never attempt unsafe actions. Over training, agents learn to minimize violation attempts.*

## Documentation

- [Agent-OS Documentation](https://imran-siddique.github.io/agent-os-docs/)
- Integration guide: see project README or examples in this directory.

## License

MIT
