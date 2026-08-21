## Agent Lightning Skill

**Teaches your coding agent how to write better agent code.** Given an editable agent and a benchmark to hillclimb on, it improves the agent's accuracy, cost, and latency through edits of prompts, skills, tools, workflows, configurations and pre-/post-processings.

### Installation

Install the skill from this repository for Claude Code, Codex, or GitHub Copilot:

```bash
gh skill install microsoft/agent-lightning agent-lightning --agent claude-code
gh skill install microsoft/agent-lightning agent-lightning --agent codex
gh skill install microsoft/agent-lightning agent-lightning --agent github-copilot
```

The core files of the skill is in [`agent-lightning`][agent-lightning] directory.

### How It Works

A large portion of the skill's power comes from the model and the coding-agent harness itself. As a matter of fact, coding-agent harnesses (like Claude Code, Codex, GitHub Copilot) are already strong optimizers. They can improve an agent's performance simply by using the following prompt:

> I've got an agent in this workspace — and it's underperforming on our benchmark. Can you raise its benchmark score while keeping any increase in per-run cost minimal — buy score cheaply, and only pay more when it clearly earns its keep?

The improvement can be further boosted when the coding agent is armed with our skill, which makes the optimization more powerful and robust.

We've challenged Claude Code, Codex, GitHub Copilot to optimize three poorly-written agents on three benchmarks. The model to to drive these agents being optimized are GPT-5.4-mini; The models that are used by the optimizer coding agents are Opus 4.8 for Claude Code, and GPT-5.6-Sol for Codex and GitHub Copilot respectively. We compared against other methods that are non-coding-agent-based (all other results are taken from the [SkillOpt paper](https://github.com/microsoft/SkillOpt)). The results are shown below.

| Method | SpreadsheetBench accuracy (%) | OfficeQA correctness (%) | ALFWorld success (%) |
| :--- | ---: | ---: | ---: |
| No skill | 36.1 | 22.1 | 73.1 |
| Human skill | 42.9 | 45.9 | 56.7 |
| LLM skill | 36.8 | 36.6 | 65.7 |
| Trace2Skill | 40.7 | 20.9 | 82.8 |
| TextGrad | 38.2 | 30.0 | 70.9 |
| GEPA | 42.5 | 45.3 | 81.3 |
| SkillOpt | 47.5 | 48.8 | 85.8 |
| Coding Agent (Avg. of CC+Codex+GHCP) | 62.9 | 54.1 | 88.6 |
| **Coding Agent (with Agent Lightning Skill)** | **66.7** | **54.5** | **94.9** |

### Performance Breakdowns

To further measure how the coding agent responds to a limited API budget, we control the API credit balance they can use during the optimization. Note that every API call, including those calls made by the coding agent itself, and those calls made by the agent being optimized, are billed into the credit. We experimented with three groups, each with \$5, \$10, and \$25 budget, and we performed three runs per group, per coding-agent harness.

The results are shown below, every point on the chart averages the three held-out finale runs for one harness, treatment, and budget: the x-axis is average overall cost on a log scale, and the y-axis is average SpreadsheetBench accuracy, OfficeQA correctness, or ALFWorld success. Color and shape identify the optimizer; filled markers use Agent Lightning and hollow markers run without. Budget is not encoded in the legend. Overall cost includes optimizer LLM calls, train/self-evaluation, and held-out finale deployment; it excludes the pristine-baseline evaluations.

![SpreadsheetBench accuracy versus overall cost](assets/agent-lightning-spreadsheetbench-accuracy-overall-cost.svg)

![OfficeQA correctness versus overall cost](assets/agent-lightning-officeqa-correctness-overall-cost.svg)

![ALFWorld success versus overall cost](assets/agent-lightning-alfworld-success-overall-cost.svg)

As the coding agent can change anything in the agent code, it can sometimes change some hard-coded settings in the code being optimized (e.g., tweaking the reasoning effort, or using an more expensive model). It's valuable to see whether the performance improvements are actually bought with a more expensive API cost, which we call "finale evaluation cost".

We chose a slice from the pervious experiment, \$5 optimizer budget for SpreadsheetBench, and \$10 budget for OfficeQA and ALFWorld as the datasets are larger. Every point in the chart is an average of three runs; the x-axis is that run's finale cost, and the y-axis is held-out SpreadsheetBench accuracy, OfficeQA correctness, or ALFWorld success. As shown in the chart, the overall performance increases much more compared to the smaller increase in API cost.

![SpreadsheetBench accuracy versus finale cost](assets/agent-lightning-spreadsheetbench-accuracy-finale-cost.svg)

![OfficeQA correctness versus finale cost](assets/agent-lightning-officeqa-correctness-finale-cost.svg)

![ALFWorld success versus finale cost](assets/agent-lightning-alfworld-success-finale-cost.svg)
