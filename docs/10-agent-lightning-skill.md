# Optimize Agents with the Agent Lightning Skill

Agent Lightning includes the [Agent Lightning Skill](https://github.com/microsoft/agent-lightning/tree/main/skills/agent-lightning) for Claude Code, Codex, and GitHub Copilot. Give a coding agent the source for an editable AI agent and a benchmark, and it can search for measured improvements to quality, cost, latency, and reliability without breaking the agent's deployment contract.

The skill complements the Agent Lightning training framework. Use the skill when a coding agent can edit and evaluate the agent's implementation or configuration. Use the framework when you want to collect rollouts and train model weights with reinforcement learning.

Install it by following the [Agent Lightning Skill installation instructions](00-installation.md#agent-lightning-skill).

## Start an Optimization Run

Open the workspace that contains the agent and its benchmark, then ask the coding agent to improve it. For example:

> I've got an agent in this workspace, and it's underperforming on our benchmark. Raise its benchmark score while keeping any increase in per-run cost minimal. Buy score cheaply, and only pay more when it clearly earns its keep.

The coding agent will use the skill to:

- inspect the agent, benchmark, and deployment-visible inputs;
- identify likely prompt, tool, workflow, model, routing, or recovery changes;
- run controlled comparisons and account for noisy or stochastic results;
- track both the one-time optimization budget and the final agent's per-run cost; and
- leave a coherent, measured checkpoint that preserves the external interface.

The workflow is most useful when the agent is editable, the benchmark is runnable, and the deployment contract is explicit. Held-out labels or training-only metadata should remain outside the deployed path.

## Measured Results

We evaluated the skill with Claude Code, Codex, and GitHub Copilot optimizing agents for SpreadsheetBench, OfficeQA, and ALFWorld. The results below average all three coding agents, the tested optimization budgets, and repeated runs. Parentheses show the percentage-point improvement over the original agent on held-out data. The starting-agent row is not shown in the table.

| Method | SpreadsheetBench accuracy (%) | OfficeQA correctness (%) | ALFWorld success (%) |
| :--- | ---: | ---: | ---: |
| Coding agents without Agent Lightning | 62.9 (+37.3) | 54.1 (+22.3) | 88.6 (+31.6) |
| **Coding agents with Agent Lightning** | **66.7 (+41.1)** | **54.5 (+22.7)** | **94.9 (+37.9)** |

See the [full benchmark methodology and cost breakdowns](https://github.com/microsoft/agent-lightning/tree/main/skills#performance-breakdowns) for the tested budgets, baselines, repeated-run setup, and charts.
