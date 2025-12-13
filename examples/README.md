# ⚡ Examples Catalog

This catalog highlights the examples shipped with Agent-lightning.

| Example | Description | CI Maintenance |
|---------|-------------|----------------|
| [apo](./apo) | Automatic Prompt Optimization tutorials covering built-in, custom, and debugging workflows. | [![apo workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-apo.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-apo.yml) |
| [azure](./azure) | Supervised fine-tuning with Azure OpenAI. | [![azure workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-azure.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-azure.yml) |
| [calc_x](./calc_x) | VERL-powered math reasoning agent training that uses AutoGen with an MCP calculator tool. | [![calc_x workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-calc-x.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-calc-x.yml) |
| [chartqa](./chartqa) | Vision-language ChartQA agent that reasons over charts with LangGraph and VERL plus multi-step self-refinement. | [![chartqa workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-chartqa.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-chartqa.yml) |
| [claude_code](./claude_code) | Claude Code SWE-bench harness that records Agent-lightning traces across Anthropic, vLLM, and OpenAI-compatible backends. | [![claude_code workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-claude-code.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-claude-code.yml) |
| [minimal](./minimal) | Bite-sized programs that demonstrate how individual Agent-lightning building blocks behave in isolation. | [![minimal workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unit.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unit.yml) |
| [rag](./rag) | Retrieval-Augmented Generation pipeline targeting the MuSiQue dataset with Wikipedia retrieval. | [![rag workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-rag.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-rag.yml) |
| [search_r1](./search_r1) | Framework-free Search-R1 reinforcement learning training workflow with a retrieval backend. | **Last verified with Agent-lightning v0.1.2** |
| [spider](./spider) | Text-to-SQL reinforcement learning training on the Spider dataset using LangGraph. | [![spider workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-spider.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-spider.yml) |
| [tinker](./tinker) | Reinforcement learning with Tinker as the backend training service. | [![tinker workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-tinker.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-tinker.yml) |
| [unsloth](./unsloth) | Supervised fine-tuning example powered by Unsloth with 4-bit quantization and LoRA. | [![unsloth workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/badge-unsloth.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-unsloth.yml) |

## `examples-*` workflow status

CI status above avoids taking any workflow running with latest dependencies into account. That's why we reference the corresponding `badge-*` workflows instead. The following table displays the raw `examples-*` workflow status whenever the project is maintained by CI.*

| Workflow | Status |
|----------|--------|
| `examples-apo.yml` | [![examples-apo workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-apo.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-apo.yml) |
| `examples-azure.yml` | [![examples-azure workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-azure.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-azure.yml) |
| `examples-calc-x.yml` | [![examples-calc-x workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-calc-x.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-calc-x.yml) |
| `examples-chartqa.yml` | [![examples-chartqa workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-chartqa.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-chartqa.yml) |
| `examples-claude-code.yml` | [![examples-claude-code workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-claude-code.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-claude-code.yml) |
| `examples-compat.yml` | [![examples-compat workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-compat.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-compat.yml) |
| `examples-rag.yml` | [![examples-rag workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-rag.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-rag.yml) |
| `examples-spider.yml` | [![examples-spider workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-spider.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-spider.yml) |
| `examples-tinker.yml` | [![examples-tinker workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-tinker.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-tinker.yml) |
| `examples-unsloth.yml` | [![examples-unsloth workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-unsloth.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-unsloth.yml) |
