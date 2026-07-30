# Agent Skills

Skills in the [Agent Skills](https://agentskills.io) format (`<name>/SKILL.md`), installable into any compatible agent.

## agl-optimizer

Turns your coding agent into an **agent optimizer**: given an editable agent and a benchmark to hillclimb on, it improves the agent's accuracy, cost, and latency through focused, individually-measured edits — keeping only what moves the frontier. Developed in the [agl-skill harness](https://github.com/agent-lightning/agl-skill) and measured there against a no-skill control under a fair, leakage-free protocol.

You provide the environment; the skill does the optimizing. Before invoking it, have ready: a working copy of the agent (keep the original pristine), labeled examples, a frozen eval command, and an objective + budget. See [here](https://github.com/agent-lightning/agl-skill/tree/main/targets) for examples.

### Installation

Claude Code users can alternatively install the packaged plugin, published from [agent-lightning/agl-skill](https://github.com/agent-lightning/agl-skill) to the community marketplace: `/plugin install agl-skill@claude-community` (the skill then appears as `/agl-skill:agl-optimizer`).

Alternatively, install via GitHub:
```bash
gh skill install microsoft/agent-lightning agl-optimizer --agent claude-code
gh skill install microsoft/agent-lightning agl-optimizer --agent codex
gh skill install microsoft/agent-lightning agl-optimizer --agent github-copilot
```

or copy `agl-optimizer/` into your agent's skills folder (`~/.claude/skills/` for Claude Code, `~/.agents/skills/` for Codex, `~/.copilot/skills/` for Copilot).

### Results

**Main finding:** Coding-agent harnesses are already strong optimizers. The clearest opportunity is improving consistency while preserving their high average performance, rather than expecting large score gains.

For context, these are the published GPT-5.4-mini direct-chat results from [SkillOpt](https://github.com/microsoft/SkillOpt) (Table 1):

| Method | Spreadsheet (%) | OfficeQA (%) | ALFWorld (%) |
| :--- | ---: | ---: | ---: |
| No skill | 36.1 | 22.1 | 73.1 |
| Human skill | 42.9 | 45.9 | 56.7 |
| LLM skill | 36.8 | 36.6 | 65.7 |
| Trace2Skill | 40.7 | 20.9 | 82.8 |
| TextGrad | 38.2 | 30.0 | 70.9 |
| GEPA | 42.5 | 45.3 | 81.3 |
| **SkillOpt** | **47.5** | **48.8** | **85.8** |

Our evaluation uses a different protocol: Claude Code, Codex, and Copilot act as agentic optimizers that can edit the whole agent. Results are held-out test means pooled across \$5, \$10, and \$25 budgets (`n = 9`) and are shown as mean ± standard deviation. Higher is better; bold marks the better mean within each row.

| Benchmark (train/test) | Optimizer | agl-skill v9.7 (%) | No-skill control (%) | Change (pp) |
| :--- | :--- | ---: | ---: | ---: |
| SpreadsheetBench (120/280) | Claude Code | 67.8 ± 5.7 | **68.9** ± 2.5 | −1.1 |
| SpreadsheetBench (120/280) | Codex | **64.4** ± 2.9 | 58.4 ± 17.7 | +5.9 |
| SpreadsheetBench (120/280) | Copilot | **68.1** ± 3.2 | 61.5 ± 13.1 | +6.6 |
| OfficeQA (50/172) | Claude Code | **59.9** ± 4.4 | 58.7 ± 3.2 | +1.2 |
| OfficeQA (50/172) | Codex | **51.6** ± 2.9 | 50.6 ± 1.3 | +1.0 |
| OfficeQA (50/172) | Copilot | 52.1 ± 2.4 | **53.0** ± 2.7 | −0.8 |
| ALFWorld (3553/134) | Claude Code | **89.1** ± 26.4 | 88.2 ± 33.1 | +0.9 |
| ALFWorld (3553/134) | Codex | 97.5 ± 4.7 | **97.8** ± 5.9 | −0.3 |
| ALFWorld (3553/134) | Copilot | 97.5 ± 7.2 | **99.9** ± 0.3 | −2.4 |

Because the two tables use different protocols and data splits, compare results within a table, not across tables. Full per-budget scores, cost breakdowns, and caveats are available in [RESULTS_V9_7.md](https://github.com/agent-lightning/agl-skill/blob/verifier-v8/RESULTS_V9_7.md).
