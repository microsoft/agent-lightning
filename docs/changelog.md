# Changelog

## Agent-lightning v0.2.2 (11/12/2025)

Agent-lightning v0.2.2 is a stabilization release for v0.2.1. It introduces several bug fixes.

* Fix compatibility issues with VERL 0.6.0.
* Fix model name for pre-downloaded models in VERL.
* Fix preparing status transition on rollout when creating attempts.
* Fix OpenAI Agents SDK compatibility issues.

**Full Changelog**: https://github.com/microsoft/agent-lightning/compare/v0.2.1...v0.2.2

---

## Agent-lightning v0.2.1 (10/30/2025)

Agent-lightning v0.2.1 is a stabilization release for v0.2.0. It introduces several bug fixes and new features, plus a number of unlisted CI improvements.

### Bug fixes

* Fix LiteLLM issues when restarting the proxy multiple times in the same process (#174 #206)
* Fix LiteLLM model name selection when multiple servers use the same model (#197)
* Fix store port conflict handling (#227)

### New Features

* Add trainer port option for client-server strategies (#198)

### Documentation

* Add tutorial for launching workers on separate machines (#213)
* Add link to VERL framework (#210)
* Add link to vLLM blog (#215)
* Fix a couple of typos and avoid emacs backup files (#237)

### New Contributors

A warm welcome to our first-time contributors: @scott-vsi, @ddsfda99, @jeis4wpi 🎉

**Full Changelog**: https://github.com/microsoft/agent-lightning/compare/v0.2.0...v0.2.1

---

## Agent-lightning v0.2.0 (10/22/2025)

Agent-Lightning v0.2.0 introduces major framework improvements, new execution strategies, expanded documentation, and enhanced reliability across the agent training and deployment workflow. This release includes **78 pull requests** since v0.1.2.

### Core Enhancements

* **Lightning Store**: Added unified interface and implementation for Agent-lightning's core storage.
* **Emitter**: Emitting any objects as spans to the store.
* **Adapter** and **Tracer**: Adapting to OpenAI-like messages, and OpenTelemetry dummy tracer.
* **LLM Proxy**: Added LLM Proxy as the first-class citizen in Agent-lightning.
* **Agent Runner**: New version providing a more modular and robust runner design.
* **Embedded Algorithms**: Algorithms are now embedded directly into trainers for simplicity.
* **New Execution Strategies**: Introduced *Client-Server* and *Shared Memory* execution models.
* **Trainer Updates**: Integrated v0.2 interfaces and FastAlgorithm validation.

### Documentation & Examples

* Revamped documentation with new guides for **agent creation**, **training**, **debugging**, and **store concepts**.
* Improved quickstart tutorials, clarified installation and new deep-dive articles.
* Added and updated examples: *SQL Agent*, *Calc-X*, *Local SFT*, *Search-R1*, and *APO algorithm*.

### Developer Experience

* Migrated build and CI pipelines to **1ES**, split workflows and aggregate badges for clarity.
* Adopted **uv** as the dependency manager.
* Added GPU-based pytest workflows for full test coverage.
* Enhanced debugging UX, pre-commit configs, and linting (Pyright fixes, import sorting).

### Ecosystem & Integrations

* Added support for agents built with [**Agent-framework**](https://github.com/microsoft/agent-framework).
* Added new community listings: [*DeepWerewolf*](https://github.com/af-74413592/DeepWerewolf) and [*AgentFlow*](https://agentflow.stanford.edu/).

### New Contributors

A warm welcome to our first-time contributors:
@hzy46, @lunaqiu, @syeehyn, @linhx1999, @SiyunZhao, and @acured 🎉

**Full changelog:** [v0.1.2 → v0.2.0](https://github.com/microsoft/agent-lightning/compare/v0.1.2...v0.2.0)

---

## Agent-lightning v0.1.2 (08/12/2025)

### What's Changed
* Add basic documentation in https://github.com/microsoft/agent-lightning/pull/33
* RAG example by @wizardlancet in https://github.com/microsoft/agent-lightning/pull/21

### New Contributors
* @wizardlancet made their first contribution in https://github.com/microsoft/agent-lightning/pull/21

**Full Changelog**: https://github.com/microsoft/agent-lightning/compare/v0.1.1...v0.1.2

---

## Agent-lightning v0.1.1 (08/06/2025)

### What's Changed
* Disable HTTP tracer tests and bump to 0.1.1 in https://github.com/microsoft/agent-lightning/pull/26
* Fix trainer bugs in v0.1 in https://github.com/microsoft/agent-lightning/pull/24

**Full Changelog**: https://github.com/microsoft/agent-lightning/compare/v0.1...v0.1.1

---

## Agent-lightning v0.1.0 (08/04/2025)

The first release of Agent-lightning!

- Turn your agent into an optimizable beast with **ZERO CODE CHANGE** (almost)! 💤
- Build with **ANY** agent framework (LangChain, OpenAI Agent SDK, AutoGen, CrewAI, ...); or even WITHOUT agent framework (Python OpenAI). You name it! 🤖
- **Selectively** optimize one or more agents in a multi-agent system. 🎯
- Embraces Reinforcement Learning, Automatic Prompt Optimization and more **algorithms**. 🤗

Install via `pip install agentlightning`.
