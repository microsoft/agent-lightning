# Agent Skills

Skills in the [Agent Skills](https://agentskills.io) format (`<name>/SKILL.md`), installable into any compatible agent.

## Agent Lightning

Turns your coding agent into an **agent optimizer**: given an editable agent and a benchmark to hillclimb on, it improves the agent's accuracy, cost, and latency through focused, individually-measured edits — keeping only what moves the frontier. It was measured against a no-skill control under a fair, leakage-free protocol.

You provide the environment; the skill does the optimizing. Before invoking it, have ready: a working copy of the agent (keep the original pristine), labeled examples, a frozen eval command, and an objective + budget.

### Installation

Install the skill from this repository for Claude Code, Codex, or GitHub Copilot:

```bash
gh skill install microsoft/agent-lightning agent-lightning --agent claude-code
gh skill install microsoft/agent-lightning agent-lightning --agent codex
gh skill install microsoft/agent-lightning agent-lightning --agent github-copilot
```

The `skills/agent-lightning/` directory is both the canonical Agent Skills package and the Claude Code plugin root, so both publication paths use the same `SKILL.md` without a copied or symlinked wrapper.

### Results

**Main finding:** Coding-agent harnesses are already strong optimizers. The clearest opportunity is improving consistency while preserving their high average performance, rather than expecting large score gains.

SkillOpt and the other non-agentic results are taken from the [SkillOpt paper](https://github.com/microsoft/SkillOpt) (Table 1); our agentic rows use the same splits and average all optimizers, budgets, and replicates.

| Method | SpreadsheetBench accuracy (%) | OfficeQA correctness (%) | ALFWorld success (%) |
| :--- | ---: | ---: | ---: |
| No skill | 36.1 | 22.1 | 73.1 |
| Human skill | 42.9 | 45.9 | 56.7 |
| LLM skill | 36.8 | 36.6 | 65.7 |
| Trace2Skill | 40.7 | 20.9 | 82.8 |
| TextGrad | 38.2 | 30.0 | 70.9 |
| GEPA | 42.5 | 45.3 | 81.3 |
| SkillOpt | 47.5 | 48.8 | 85.8 |
| Agentic optimizer average, no skill | 62.9 | 54.1 | 88.6 |
| **Agentic optimizer average, Agent Lightning** | **66.7** | **54.5** | **94.9** |

#### Performance versus overall cost

Each benchmark includes the \$5, \$10, and \$25 nominal-budget groups with three runs per treatment cell. Every point averages the three held-out finale runs for one harness, treatment, and budget: the x-axis is average overall cost on a log scale, and the y-axis is average SpreadsheetBench accuracy, OfficeQA correctness, or ALFWorld success. Color and shape identify the optimizer; filled markers use Agent Lightning and hollow markers are no-skill controls. Budget is not encoded in the legend. Overall cost includes optimizer LLM calls, train/self-evaluation, and held-out finale deployment; it excludes the pristine-baseline evaluations.

Claude Code uses Claude Opus 4.8; Codex and GitHub Copilot use GPT 5.6 Sol as their optimizer models.

![SpreadsheetBench accuracy versus overall cost](assets/agent-lightning-spreadsheetbench-accuracy-overall-cost.svg)

![OfficeQA correctness versus overall cost](assets/agent-lightning-officeqa-correctness-overall-cost.svg)

![ALFWorld success versus overall cost](assets/agent-lightning-alfworld-success-overall-cost.svg)

#### Performance versus finale cost

The selected-budget views use the groups with the strongest aggregate skill-over-control lift: \$5 for SpreadsheetBench and \$10 for OfficeQA and ALFWorld. Every harness/treatment point is one of three runs; the x-axis is that run's finale cost, and the y-axis is held-out SpreadsheetBench accuracy, OfficeQA correctness, or ALFWorld success. Finale cost measures LLM gateway spend, so an ALFWorld deterministic controller can have exactly \$0 finale cost while still executing and scoring real environment steps; coincident zero-cost ALFWorld results are offset slightly along the x-axis so each replicate remains visible. SpreadsheetBench and OfficeQA show their aggregate pristine-baseline results as single reference points. The dotted ALFWorld baseline is a score-only reference: the corrected records do not include baseline deployment cost, so assigning it an x-coordinate would invent data.

![SpreadsheetBench accuracy versus finale cost](assets/agent-lightning-spreadsheetbench-accuracy-finale-cost.svg)

![OfficeQA correctness versus finale cost](assets/agent-lightning-officeqa-correctness-finale-cost.svg)

![ALFWorld success versus finale cost](assets/agent-lightning-alfworld-success-finale-cost.svg)
