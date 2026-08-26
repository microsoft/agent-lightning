# Agent Lightning Documentation

<p align="center">
	<img src="images/agl-v1.0.svg" alt="Agent Lightning v1.0" width="500">
</p>

Welcome to the Agent Lightning v1.0 documentation. Start with the installation and quick-start guides, then use the configuration guides and examples below to build and train your own agents.

Agent Lightning v1.0 is a completely redesigned and reimplemented version with the following key features:

- 🪶 **~3,500 lines of core Python:** Simplicity is the first principle.
- 🧩 **Training with real agent harnesses:** Agents interact with the model through the Agent Lightning v1.0 proxy with zero changes while keeping tools, context, control flow, and environments in the loop.
- ☸️ **Native Kubernetes support:** Agents run directly as Kubernetes Jobs without relying on external sandbox services.
- 💻 **A complete coding-agent training example:** The released pipeline covers data cleaning, reward-hacking prevention, and training scripts.

For the legacy Agent Lightning releases earlier than v1.0, see the [`v0.x` code branch](https://github.com/microsoft/agent-lightning/tree/v0.x) and the [v0.3.0 documentation](https://microsoft.github.io/agent-lightning/0.3.0/).

## Getting Started

| Guide | Description |
|---|---|
| [Installation](00-installation.md) | Set up the base environment and the tested `verl` GPU stack. |
| [Quick Start](01-quick-start.md) | Run a local end-to-end rollout-driven training job. |
| [Basics](05-basics.md) | Learn the core components, rollouts, events, and trajectories. |

## Configuration

| Guide | Description |
|---|---|
| [Trainer Configuration](20-trainer-configuration.md) | Configure `verl` integration, rollout collection, and trace aggregation. |
| [API Gateway Configuration](25-api-gateway-configuration.md) | Configure the API Gateway and model proxy. |
| [Controller Configuration](30-controller-configuration.md) | Configure local and Kubernetes rollout runners. |
| [Asynchronous Training](35-asynchronous-training.md) | Configure collocated asynchronous collection and pause/drain behavior. |

## Examples

| Example | Description |
|---|---|
| [Calc-X](50-example-calc-x.md) | Train a math reasoning agent with AutoGen and MCP calculator tools. |
| [GSM8K](55-example-gsm8k.md) | Train an agent on grade-school math reasoning tasks. |
| [ScienceWorld](60-example-science-world.md) | Train an agent on interactive science tasks in a text environment. |
| [Search-R1](65-example-search-r1.md) | Train a multi-turn retrieval and reasoning agent. |
| [LLM-in-Sandbox](70-example-llm-in-sandbox.md) | Train a general agent with computer and code execution tools. |
| [Coding Agent](75-example-coding-agent.md) | Train a coding agent using repository tests as feedback. |
| [Multimodal QA](80-example-multimodal-qa.md) | Train a vision-language model on synthetic image QA. |
