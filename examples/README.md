# ⚡ Examples Catalog

This catalog highlights the examples shipped with Agent Lightning.

| Example | Description | CI Maintenance |
|---------|-------------|----------------|
| [apo](./apo) | Automatic Prompt Optimization tutorials covering built-in, custom, and debugging workflows. | [![apo workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-apo.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/badge-apo.yml) |
| [calc_x](./calc_x) | VERL-powered math reasoning agent training that uses AutoGen with an MCP calculator tool. | [![calc_x workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-calc-x.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/badge-calc-x.yml) |
| [rag](./rag) | Retrieval-Augmented Generation pipeline targeting the MuSiQue dataset with Wikipedia retrieval. | **Unmaintained** — last verified with Agent Lightning v0.1.1 |
| [search_r1](./search_r1) | Framework-free Search-R1 reinforcement learning training workflow with a retrieval backend. | **Unmaintained** — last verified with Agent Lightning v0.1.2 |
| [spider](./spider) | Text-to-SQL reinforcement learning training on the Spider dataset using LangGraph. | [![spider workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-spider.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/badge-spider.yml) |
| [unsloth](./unsloth) | Supervised fine-tuning example powered by Unsloth with 4-bit quantization and LoRA. | [![unsloth workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unsloth.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unsloth.yml) |

*NOTE: CI status avoid taking any workflow running with latest dependencies into account. That's why we reference the corresponding `badge-*` workflows instead. Each example's own README also displays its `examples-*` workflow status whenever the project is maintained by CI.*
