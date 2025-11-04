# APO Example

[![apo CI status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-apo.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-apo.yml)

This example folder contains three complementary tutorials that demonstrate different aspects of Agent-Lightning. It's compatible with Agent-lightning v0.2 or later.

## Overview

The folder showcases three distinct use cases: using the built-in APO algorithm to train a room selection agent, creating custom training algorithms from scratch, and debugging agents effectively. Each tutorial is self-contained and demonstrates a specific workflow.

## Requirements

Follow the [installation guide](../../docs/tutorials/installation.md) to install Agent-Lightning and APO-extra dependencies. All examples also require an OpenAI-compatible API service.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `room_selector.py` | Room booking agent implementation using function calling |
| `room_selector_apo.py` | Training script using the built-in APO algorithm to optimize prompts |
| `room_tasks.jsonl` | Dataset with room booking scenarios and expected selections |
| `apo_custom_algorithm.py` | Tutorial on creating custom algorithms (runnable as algo or runner) |
| `apo_custom_algorithm_trainer.py` | Shows how to integrate custom algorithms into the Trainer |
| `apo_debug.py` | Tutorial demonstrating various agent debugging techniques |
| `legacy_apo_client.py` | Deprecated APO client implementation compatible with Agent-lightning v0.1.x |
| `legacy_apo_server.py` | Deprecated APO server implementation compatible with Agent-lightning v0.1.x |

## Sample 1: Using Built-in APO Algorithm

The `room_selector_apo.py` script demonstrates how to use Agent-Lightning's built-in APO (Asynchronous Prompt Optimization) algorithm to train a room booking agent. The agent learns to select meeting rooms based on duration, attendee count, equipment needs, accessibility requirements, and availability.

Run the training with:

```bash
python room_selector_apo.py
```

This script initializes the APO algorithm with beam search parameters, loads the room booking dataset, and optimizes the agent's prompt template through iterative training. The algorithm automatically manages the training loop, gradient computation, and prompt updates. Read more about this example in [Train the First Agent with APO](../../docs/how-to/train-first-agent.md).

## Sample 2: Creating Custom Algorithms

The `apo_custom_algorithm.py` and `apo_custom_algorithm_trainer.py` files teach you how to implement custom training algorithms from scratch. This is useful when the built-in algorithms don't fit your specific needs. See [Custom Algorithm tutorial](../../docs/how-to/write-first-algorithm.md) for more details.

### Option A: Run algorithm and runner separately

Start the store, algorithm, and runner in three separate terminals:

```bash
# Terminal 1: Start the store
agl store

# Terminal 2: Run the algorithm
python apo_custom_algorithm.py algo

# Terminal 3: Run the rollout runner
python apo_custom_algorithm.py runner
```

### Option B: Run integrated version

Use the integrated trainer that handles all components:

```bash
python apo_custom_algorithm_trainer.py
```

## Sample 3: Debugging Agents

The `apo_debug.py` script demonstrates multiple approaches to debugging agents in Agent-Lightning:

```bash
python apo_debug.py
```

Read more about this example in [Debugging Agents](../../docs/tutorials/debug.md).

## Appendix: Dataset Format

The `room_tasks.jsonl` file contains meeting scenarios with the following structure:

```json
{
  "id": "s01",
  "task_input": {
    "date": "2025-10-13",
    "time": "16:30",
    "duration_min": 30,
    "attendees": 12,
    "needs": ["projector", "confphone"],
    "accessible_required": true
  },
  "expected_choice": "Nova"
}
```
