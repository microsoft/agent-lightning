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

Each benchmark includes the \$5, \$10, and \$25 nominal-budget groups with three runs per treatment cell. Every point is one held-out finale result: the x-axis is that run's overall cost on a log scale, and the y-axis is SpreadsheetBench accuracy, OfficeQA correctness, or ALFWorld success. Color and shape identify the optimizer; filled markers use Agent Lightning and hollow markers are no-skill controls. Budget is not encoded in the legend. Overall cost includes optimizer LLM calls, train/self-evaluation, and held-out finale deployment; it excludes the pristine-baseline evaluations.

Claude Code uses Claude Opus 4.8; Codex and GitHub Copilot use GPT 5.6 Sol as their optimizer models.

![SpreadsheetBench accuracy versus overall cost](assets/agent-lightning-spreadsheetbench-accuracy-overall-cost.svg)

![OfficeQA correctness versus overall cost](assets/agent-lightning-officeqa-correctness-overall-cost.svg)

![ALFWorld success versus overall cost](assets/agent-lightning-alfworld-success-overall-cost.svg)

#### Performance versus finale cost

The selected-budget views use the groups with the strongest aggregate skill-over-control lift: \$5 for SpreadsheetBench and \$10 for OfficeQA and ALFWorld. Every harness/treatment point is one of three runs; the x-axis is that run's finale cost, and the y-axis is held-out SpreadsheetBench accuracy, OfficeQA correctness, or ALFWorld success. Finale cost measures LLM gateway spend, so an ALFWorld deterministic controller can have exactly \$0 finale cost while still executing and scoring real environment steps; coincident zero-cost ALFWorld results are offset slightly along the x-axis so each replicate remains visible. SpreadsheetBench and OfficeQA show their aggregate pristine-baseline results as single reference points. The corrected ALFWorld records do not include baseline deployment cost, so its aggregate measured success is shown as a horizontal reference instead of assigning it an x-coordinate.

![SpreadsheetBench accuracy versus finale cost](assets/agent-lightning-spreadsheetbench-accuracy-finale-cost.svg)

![OfficeQA correctness versus finale cost](assets/agent-lightning-officeqa-correctness-finale-cost.svg)

![ALFWorld success versus finale cost](assets/agent-lightning-alfworld-success-finale-cost.svg)

#### \$5 budget snapshot

| Benchmark metric (train/test) | Result | Score (%) | Finale cost |
| :--- | :--- | ---: | ---: |
| SpreadsheetBench accuracy (120/280) | Baseline | 25.66 ± 2.65 | \$1.51 ± 0.04 |
|  | Claude Code with skill | 63.79 ± 5.24 | **\$2.45 ± 0.45** |
|  | Claude Code without skill | **68.23 ± 0.55** | \$2.52 ± 0.93 |
|  | Codex with skill | **65.47 ± 4.59** | **\$1.66 ± 0.19** |
|  | Codex without skill | 41.49 ± 24.28 | \$1.73 ± 0.15 |
|  | Copilot with skill | **66.31 ± 2.05** | **\$1.65 ± 0.13** |
|  | Copilot without skill | 51.68 ± 20.82 | \$1.66 ± 0.09 |
| OfficeQA correctness (50/172) | Baseline | 31.78 ± 1.21 | \$2.78 ± 0.06 |
|  | Claude Code with skill | 56.78 ± 3.87 | \$5.35 ± 1.60 |
|  | Claude Code without skill | **59.69 ± 4.88** | **\$4.69 ± 1.60** |
|  | Codex with skill | **49.81 ± 2.98** | **\$3.38 ± 0.54** |
|  | Codex without skill | 49.61 ± 0.67 | \$3.77 ± 0.22 |
|  | Copilot with skill | 51.55 ± 3.74 | **\$3.69 ± 0.45** |
|  | Copilot without skill | **54.65 ± 2.01** | \$4.20 ± 0.58 |
| ALFWorld success (3553/134) | Baseline | 56.97 ± 0.43 | — |
|  | Claude Code with skill | **95.02 ± 1.14** | \$3.83 ± 0.78 |
|  | Claude Code without skill | 93.53 ± 4.11 | **\$3.21 ± 0.38** |
|  | Codex with skill | 87.31 ± 21.97 | \$1.69 ± 2.92 |
|  | Codex without skill | **96.52 ± 3.02** | **\$0.97 ± 1.68** |
|  | Copilot with skill | **99.75 ± 0.43** | \$0.01 ± 0.02 |
|  | Copilot without skill | 95.52 ± 7.12 | **\$0.00 ± 0.00** |

#### \$10 budget snapshot

| Benchmark metric (train/test) | Result | Score (%) | Finale cost |
| :--- | :--- | ---: | ---: |
| SpreadsheetBench accuracy (120/280) | Baseline | 25.66 ± 2.65 | \$1.51 ± 0.04 |
|  | Claude Code with skill | 67.75 ± 2.40 | **\$2.01 ± 0.26** |
|  | Claude Code without skill | **69.42 ± 4.32** | \$2.11 ± 0.21 |
|  | Codex with skill | 64.63 ± 0.75 | **\$1.59 ± 0.06** |
|  | Codex without skill | **68.59 ± 1.16** | \$1.72 ± 0.08 |
|  | Copilot with skill | **69.30 ± 4.32** | \$2.08 ± 0.74 |
|  | Copilot without skill | 64.39 ± 2.88 | **\$1.63 ± 0.05** |
| OfficeQA correctness (50/172) | Baseline | 31.78 ± 1.21 | \$2.78 ± 0.06 |
|  | Claude Code with skill | **62.60 ± 3.74** | \$5.88 ± 0.82 |
|  | Claude Code without skill | 59.30 ± 1.74 | **\$5.68 ± 0.74** |
|  | Codex with skill | **54.07 ± 1.16** | \$3.95 ± 0.43 |
|  | Codex without skill | 50.00 ± 0.58 | **\$3.77 ± 0.27** |
|  | Copilot with skill | **53.68 ± 0.89** | \$3.46 ± 0.03 |
|  | Copilot without skill | 51.16 ± 4.07 | **\$3.07 ± 1.46** |
| ALFWorld success (3553/134) | Baseline | 56.97 ± 0.43 | — |
|  | Claude Code with skill | 93.78 ± 0.43 | **\$3.62 ± 0.51** |
|  | Claude Code without skill | **94.28 ± 3.02** | \$3.67 ± 0.90 |
|  | Codex with skill | **99.00 ± 0.86** | \$0.75 ± 1.28 |
|  | Codex without skill | 89.55 ± 18.10 | **\$0.00 ± 0.00** |
|  | Copilot with skill | **100.00 ± 0.00** | \$0.00 ± 0.00 |
|  | Copilot without skill | 66.92 ± 57.30 | \$0.00 ± 0.00 |

#### \$25 budget snapshot

| Benchmark metric (train/test) | Result | Score (%) | Finale cost |
| :--- | :--- | ---: | ---: |
| SpreadsheetBench accuracy (120/280) | Baseline | 25.66 ± 2.65 | \$1.51 ± 0.04 |
|  | Claude Code with skill | **71.70 ± 7.11** | \$8.12 ± 5.26 |
|  | Claude Code without skill | 68.94 ± 1.98 | **\$5.31 ± 5.71** |
|  | Codex with skill | 62.95 ± 2.52 | **\$1.76 ± 0.24** |
|  | Codex without skill | **65.23 ± 2.40** | \$1.84 ± 0.24 |
|  | Copilot with skill | **68.71 ± 3.12** | \$3.48 ± 3.17 |
|  | Copilot without skill | 68.47 ± 3.60 | **\$1.69 ± 0.01** |
| OfficeQA correctness (50/172) | Baseline | 31.78 ± 1.21 | \$2.78 ± 0.06 |
|  | Claude Code with skill | **60.27 ± 4.70** | **\$5.32 ± 0.62** |
|  | Claude Code without skill | 57.17 ± 2.98 | \$11.81 ± 7.02 |
|  | Codex with skill | 50.78 ± 2.87 | **\$3.28 ± 0.22** |
|  | Codex without skill | **52.13 ± 0.34** | \$3.83 ± 0.41 |
|  | Copilot with skill | 51.16 ± 1.74 | **\$4.13 ± 0.25** |
|  | Copilot without skill | **53.10 ± 0.34** | \$4.38 ± 0.40 |
| ALFWorld success (3553/134) | Baseline | 56.97 ± 0.43 | — |
|  | Claude Code with skill | 82.59 ± 21.84 | \$4.28 ± 1.27 |
|  | Claude Code without skill | **94.78 ± 4.48** | **\$3.02 ± 0.55** |
|  | Codex with skill | **96.77 ± 5.60** | \$1.08 ± 1.87 |
|  | Codex without skill | 66.67 ± 57.74 | **\$0.01 ± 0.02** |
|  | Copilot with skill | 100.00 ± 0.00 | \$0.12 ± 0.20 |
|  | Copilot without skill | 100.00 ± 0.00 | **\$0.00 ± 0.00** |
