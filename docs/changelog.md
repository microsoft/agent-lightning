# Changelog

## Agent-lightning v0.3.0 (12/24/2025)

Agent-lightning v0.3.0 is a major release that introduces several new features and bug fixes. The release is a collaboration effort between Agent-lightning core teams and the community. Thanks to all the contributors who made this release possible.

### Highlights

* **Tinker integration**: Support Tinker as an alternative backend for Reinforcement Learning (#226 #245 #264 #269 #327). See [example code](https://github.com/microsoft/agent-lightning/tree/v0.3.0/examples/tinker), [blog 1](https://medium.com/@yugez/tuning-any-ai-agent-with-tinker-agent-lightning-part-1-1d8c9a397f0e) and [blog 2](https://medium.com/@yugez/tuning-any-ai-agent-with-tinker-agent-lightning-part-2-332c5437f0dc).
* **Azure OpenAI integration**: Support Azure OpenAI as a backend for LLM inference and supervised fine-tuning (#256 #327). [Example code](https://github.com/microsoft/agent-lightning/tree/v0.3.0/examples/azure).
* **MongoDB based Lightning Store** is added as an alternative backend for Lightning Store (#323). [Example code](https://github.com/microsoft/agent-lightning/tree/v0.3.0/examples/mongo).
* **Contrib package**: Add contrib package for community projects. Search-R1 is integrated as a contrib recipe. More coming. (#239 #396 #410 #412 #417).
* **RESTful API**: Stabilize and document RESTful API for Lightning Store (#241 #275). [Documentation](https://microsoft.github.io/agent-lightning/v0.3.0/reference/restful/).
* **OTel Semantic Convention** that are specifically designed for Agent-optimization areas (#340). [Documentation](http://microsoft.github.io/agent-lightning/v0.3.0/reference/semconv/).
* *[Preview]* **Agent-lightning Dashboard** is now available (#288 #289 #291 #296 #371 #375). It's the official Web application for inspecting and debugging Agent-lightning experiments. See details [here](https://microsoft.github.io/agent-lightning/v0.3.0/tutorials/debug/).
* *[Preview]* **Multi-modality example** featuring VERL and LangGraph agent on ChartQA dataset (#379). [Example code](https://github.com/microsoft/agent-lightning/tree/v0.3.0/examples/chartqa).
* *[Preview]* Integrate **Claude Code** as a LitAgent and support training on SWE-Bench (#332 #346 #348). [Example code](https://github.com/microsoft/agent-lightning/tree/v0.3.0/examples/claude_code).
* *[Preview]* **Weave tracer** as a substitute for AgentOps tracer (#277 #411 #420 #423). [Documentation](https://microsoft.github.io/agent-lightning/v0.3.0/tutorials/traces/#weave-tracer-experimental).
* *[Preview]* **Trajectory Level Aggregation** for more efficient training with VERL. See [blog](https://agent-lightning.github.io/posts/trajectory_level_aggregation/) and [documentation](https://microsoft.github.io/agent-lightning/v0.3.0/algorithm-zoo/verl/).

### Store Benchmark

During this iteration, the core Lightning Store is rewritten to be more efficient and scalable (#315 #318 #328 #342 #344 #356 #380 #388 #418 #421). We show the benchmark results in the following table and compare it with the previous release (v0.2.2).

| Throughput (\#rollout/sec) | v0.2.2 | v0.3.0 (in-memory) | v0.3.0 (Mongo) |
| :---- | :---- | :---- | :---- |
| Minimal (batch, #runner=32, #turns=6) | 8.53 | 9.10 | 8.73 |
| Medium (batch, #runners=100, #turns=10) | 11.97 | 21.46 | 28.01 |
| Mid-high (batch, #runners=256, #turns=8) | 7.26 | 22.37 | 29.50 |
| Large (batch, #runners=256, #turns=4) | timeout (\<7.5) | 28.67 | 53.59 |
| Long queue (FIFO queue, #runners=256, #turns=4) | timeout (\<7.5) | 29.17 | 42.16 |
| Heavy trace (FIFO queue, #runners=512, #turns=20) | 5.79 | 12.90 | 31.95 |

### Maintenance and Bug fixes

#### Core (Store, Interfaces, etc.)

* Add Trainer port option for client-server strategies (#198)
* Fix store port conflict handling (#227)
* Unified PythonServerLauncher (#286 #292 #303)
* Make health timeout configurable (#305)
* Refactor logging (#306)
* Support OTLP in LightningStore (313)
* Centralized metrics helper (#368)
* Fix redundant cancel tracebacks on Ctrl+C (#370)

#### Proxy, Adapters and Algorithms

* Fix training metrics before and after processing in VERL (#145)
* Forward streaming requests for Anthropic and OpenAI APIs (as non-streaming requests) (#299)
* Check traces with reward for VERL (#317)
* Patch LiteLLM root span (#341)
* Handle ref_in_actor flag for LoRA compatibility (#386)
* Support `with_llm_proxy` and `with_store` in algorithms (#398)
* Support image urls export in TracerTraceToTriplets (#400)
* Fix match_rewards assign_to elements in TraceTree (#403)
* Support customizing trainer and daemon in VERL (#407)

#### Runners, Tracers and Agents

* Refactor tracer initialization (#321)
* Fix OpenAI Agents 0.6 compatibility (#322)
* Operation emitter (#359)
* Sunset HTTP tracer (#402)

#### Examples

* Fix typos in train-first-agent.md (#263)
* Fix room_selector example which always run the first task (#270)
* Fix typo in SQL agent example (#285)
* Add the README and script files for training SQL agent on NPU (#272)
* Examples Catalog and Refine Contribution Guide (#331)
* Upgrade LangChain to 1.x (#364)
* Update RAG example to Agent-lightning v0.2.x (#349)

#### Miscellaneous

* DeepWiki Badge (#263)
* Add AGENTS.md (#374)

### New Contributors

Warm welcome to our first-time contributors: @cptnm3, @TerryChan, @genji970, @zxgx, @xiaochulaoban, @lspinheiro, @Kwanghoon-Choi, @Vasuk12, @totoluo, @jinghuan-Chen 🎉

**Full Changelog**: https://github.com/microsoft/agent-lightning/compare/v0.2.0...v0.3.0

---

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
