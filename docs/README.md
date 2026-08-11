# Agent Lightning Documentation

Welcome to the Agent Lightning v1.0 documentation. Start with the installation and quick-start guides, then use the configuration guides and examples below to build and train your own agents.

## Getting Started

| Guide | Description |
|---|---|
| [Installation](1-installation.md) | Set up the base environment and the tested VERL GPU stack. |
| [Quick Start](2-quick-start.md) | Run a local end-to-end rollout-driven training job. |
| [Basics](3-basics.md) | Learn the core components, rollouts, events, and trajectories. |

## Configuration

| Guide | Description |
|---|---|
| [Trainer Configuration](4-trainer-configuration.md) | Configure VERL integration, rollout collection, and trace aggregation. |
| [Server Configuration](5-server-configuration.md) | Configure the API gateway and model proxy. |
| [Controller Configuration](6-controller-configuration.md) | Configure local and Kubernetes rollout runners. |
| [Asynchronous Training](7-asynchronous-training.md) | Configure collocated asynchronous collection and pause/drain behavior. |

## Examples

| Example | Description |
|---|---|
| [Calc-X](8-example-calc-x.md) | Train a math reasoning agent with AutoGen and MCP calculator tools. |
| [GSM8K](9-example-gsm8k.md) | Train an agent on grade-school math reasoning tasks. |
| [ScienceWorld](10-example-science-world.md) | Train an agent on interactive science tasks in a text environment. |
| [Search-R1](11-example-search-r1.md) | Train a multi-turn retrieval and reasoning agent. |
| [LLM-in-Sandbox](12-example-llm-in-sandbox.md) | Train a general agent with computer and code execution tools. |
| [Coding Agent](13-example-coding-agent.md) | Train a coding agent using repository tests as feedback. |

Return to the [project overview](../README.md).
