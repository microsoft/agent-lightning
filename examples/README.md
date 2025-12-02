# ⚡ Examples Catalog

This catalog highlights the examples shipped with Agent-lightning.

| Example | Description | CI Maintenance |
|---------|-------------|----------------|
| [apo](./apo) | Automatic Prompt Optimization tutorials covering built-in, custom, and debugging workflows. | [![apo workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-apo.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-apo.yml) |
| [azure](./azure) | Supervised fine-tuning with Azure OpenAI. | [![azure workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-azure.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-azure.yml) |
| [calc_x](./calc_x) | VERL-powered math reasoning agent training that uses AutoGen with an MCP calculator tool. | [![calc_x workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-calc-x.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-calc-x.yml) |
| [claude_code](./claude_code) | Claude Code SWE-bench harness that records Agent-lightning traces across Anthropic, vLLM, and OpenAI-compatible backends. | [![claude_code workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-claude-code.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-claude-code.yml) |
| [minimal](./minimal) | Bite-sized programs that demonstrate how individual Agent-lightning building blocks behave in isolation. | [![minimal workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unit.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unit.yml) |
| [rag](./rag) | Retrieval-Augmented Generation pipeline targeting the MuSiQue dataset with Wikipedia retrieval. | **Unmaintained** — last verified with Agent-lightning v0.1.1 |
| [search_r1](./search_r1) | Framework-free Search-R1 reinforcement learning training workflow with a retrieval backend. | **Unmaintained** — last verified with Agent-lightning v0.1.2 |
| [spider](./spider) | Text-to-SQL reinforcement learning training on the Spider dataset using LangGraph. | [![spider workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-spider.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-spider.yml) |
| [tinker](./tinker) | Reinforcement learning with Tinker as the backend training service. | [![tinker workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-tinker.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-tinker.yml) |
| [unsloth](./unsloth) | Supervised fine-tuning example powered by Unsloth with 4-bit quantization and LoRA. | [![unsloth workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unsloth.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-unsloth.yml) |

*NOTE: CI status avoids taking any workflow running with latest dependencies into account. That's why we reference the corresponding `badge-*` workflows instead. Each example's own README also displays its `examples-*` workflow status whenever the project is maintained by CI.*
