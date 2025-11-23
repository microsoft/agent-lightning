# Contributing Guide

Agent Lightning gets better every time someone files a clear bug, polishes docs, improves tests, or lands a new feature. This guide collects the expectations, checklists, and tips that help you go from “I have an idea” to “my pull request just merged.”

## Before You Start

Agent-lightning is built by a small Microsoft Research team with limited reviewer hours and GPU budget. For any sizeable change (new algorithm, example, or API surface) please first discuss scope with us in [Discord](https://discord.gg/RYk7CdvDR7). Early alignment keeps your effort from being blocked late in the process.

## Where You Can Help

Pick a lane, or combine several. Just keep the discussion-first principle in mind for anything non-trivial.

### Documentation Improvements

Documentation improvements are the easiest way to get started. You can find more about how to write good documentations and organize documentations in the following sections. Here are some general contribution points we can think of:

- Tighten language, fix typos, clarify confusing sections, or add missing links. Fresh eyes catch docs gaps best.
- Organize content using the directories listed below so readers can actually find it.
- Avoid duplicate prose, unrelated “how-to” guides, or translations (we cannot maintain them today).

!!! note "Changes that are usually rejected"

    - Copy/pasting existing docs with shallow edits.
    - Adding a `how-to` guide that is not tied to a new example.
    - Adding doc translations to other languages (no capacity to review/maintain yet).

### Bug Fixes

Bug fixes are the fastest way to get familiar with the codebase. To get started, you can:

- Browse the ["good first issue"](https://github.com/microsoft/agent-lightning/labels/good%20first%20issue) and ["bug"](https://github.com/microsoft/agent-lightning/labels/bug) labels; drop a comment before you start so we can mark it as taken.
- For fresh bugs, open an issue with reproduction steps, logs, and expected behavior before submitting a fix.
- Keep each pull request focused, ideally avoiding breaking API changes. Larger refactors should be discussed via RFC or maintainer sync.

### New Examples

Examples must be curated so that we can maintain them. We generally merge only those that meet at least one (ideally several) of these criteria:

- Demonstrates an agent framework or workflow that is materially different from what already exists. ([LangChain](https://www.langchain.com/) vs. [LlamaIndex](https://www.llamaindex.ai/) is not different enough; [LangChain](https://www.langchain.com/) vs. [n8n](https://n8n.io/) or [Vercel AI SDK](https://ai-sdk.dev/) is, because they either have different orchestration paradigms or differ in programming languages.)
- Shows measurable performance gains on a **real-world** problem with a **real-world** dataset, such as tuning a search agent with Google Search API or improving a coding agent’s (e.g., Claude Code) SWE-Bench score.
- Integrates a new algorithm, training backend, or serving stack (see “New Algorithms” below).
- Validates scenarios that are rarely tested, such as multi-modality agents or long-lived memory/workflow agents.

Bonus points for examples that:

- Ship CI or self-test coverage so we know they still work as the core evolves.  **Otherwise, we would have to mark the example as unmaintained because we won't be able to test the examples manually before each release.**
- Include a [`docs/how-to/`]({{ src("docs/how-to/") }}) guide (or a detailed README if no how-to exists) without duplicating content in multiple places.
- Favor simple, dependency-light code over heavy abstractions.

!!! warning "Please discuss first"

    Examples tend to be the most time-consuming contributions for both you and reviewers. Sync with us on Discord or through an issue before diving into a new one.

### Fresh Implementations of Core Modules

If you are looking to extend [`Runner`][agentlightning.Runner], [`Tracer`][agentlightning.Tracer], [`Adapter`][agentlightning.Adapter], [`LightningStore`][agentlightning.LightningStore], or another core interface, here are the steps:

1. File an issue or proposal first.
2. Explain which interface you are extending, why existing implementations are insufficient, and how you intend to test compatibility with the rest of the stack (unit tests, documentation updates, example refreshes, etc.).
3. Any API changes must be reviewed up front. DO NOT begin coding large changes before the discussion lands!

### New Algorithms

If you are integrating a new training/serving backend, check whether it already lives in the [Algorithm Zoo](../algorithm-zoo/index.md) or is covered in the [Examples Catalog](../how-to/examples-catalog.md). We especially welcome:

- Currently unsupported or under-tested algorithms such as Supervised Fine-tuning (SFT), Direct Policy Optimization (DPO), or Monte Carlo Tree Search (MCTS).
- Tuning [Resource][agentlightning.Resource]s that are not supported yet, such as workflows or memory.
- Expansions of supported stacks, e.g., adding multi-modality to APO or multi-agent prompt tuning.
- Reinforcement-learning integrations beyond our current stack of [VERL](https://github.com/volcengine/verl), [vLLM](https://vllm.ai/), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-foundry/models/openai), and [Tinker](https://tinker-docs.thinkingmachines.ai/). Contributions using [SGLang](https://github.com/sgl-project/sglang), [TRL](https://github.com/huggingface/trl), [SkyRL](https://github.com/NovaSky-AI/SkyRL), [RLinf](https://github.com/RLinf/RLinf), [litgpt](https://github.com/Lightning-AI/litgpt), or similar are welcome.

Most brand-new algorithms ultimately land as “new examples,” so read that section too. Post an issue or design doc to scope the work, reuse existing utilities, and avoid duplicating efforts. Mature, battle-tested examples graduate into the [Algorithm Zoo](../algorithm-zoo/index.md).

### Ecosystem Projects

Have a project that builds on Agent-lightning but does not belong in the main repo? Fork it or depend on it externally, then let us know. We can showcase notable projects in [Community Projects](../index.md) and the main [README]({{ src("README.md") }}).

### Other Contribution Ideas

- **Tests.** Add or improve cases in [`tests/`]({{ src("tests") }}) (unit, integration, or end-to-end).
- **Benchmarks.** Expand [`tests/benchmark`]({{ src("tests/benchmark") }}) to stress large-scale training or rollouts.
- **Issue triage.** Reproduce bugs, confirm whether they reproduce on `main`, or suggest short-term mitigations so maintainers can prioritize.

## Contribution Workflow

The steps below keep changes reviewable and CI-friendly. Follow them in order; rerun the relevant pieces if you revisit a branch later.

### 1. Prepare Your Environment

Minimum tooling:

- **Python** 3.10+ (3.12 recommended).
- **uv** for dependency and virtual-environment management. Install it using the [official uv docs](https://docs.astral.sh/uv/getting-started/installation/).
- **Git** configured with your GitHub credentials.

Clone your fork and point `upstream` at the official repo:

```bash
git clone git@github.com:<your-username>/agent-lightning.git
cd agent-lightning
git remote add upstream https://github.com/microsoft/agent-lightning.git
```

Install the default development stack:

```bash
uv sync --group dev
```

Need GPU extras or specific optional dependencies? Lock them in with one command:

```bash
uv sync --frozen \
    --extra apo \
    --extra verl \
    --group dev \
    --group torch-cpu \
    --group torch-stable \
    --group agents \
    --no-default-groups
```

After `uv sync`, run commands via `uv run ...` (add `--no-sync` once the environment is locked) or activate `.venv/`.

### 2. Install and Run Pre-commit

Formatting and linting are enforced through [pre-commit](https://pre-commit.com/). Install once, then run before each push:

```bash
uv run pre-commit install
uv run pre-commit run --all-files --show-diff-on-failure --color=always
```

Once installed, the hooks run automatically on every `git commit`. Running the pre-commit hooks locally keeps CI green and diffs manageable.

### 3. Branch From a Fresh `main`

Start all work from the latest upstream state:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

Branch naming convention:

- `feature/<short-description>` for new features.
- `fix/<short-description>` for bug fixes.
- `docs/<short-description>` for documentation-only updates.
- `chore/<short-description>` for tooling or maintenance.

Use lowercase with hyphens, e.g., `feature/async-runner-hooks`.

!!! note "Where should docs or examples live?"

    Many new contributors get confused about what to put in the `docs/how-to/` directory and what to put in the `examples/` directory (particularly README files). Here is a quick reference you can refer to:

    | Location | Description |
    | --- | --- |
    | `docs/algorithm-zoo/` | Documentation for **built-in algorithms** shipped with Agent-lightning. |
    | `docs/how-to/` | Step-by-step **how-to guides**, usually tied to an example in `examples/`. |
    | `docs/tutorials/` | Conceptual walkthroughs for components or workflows. See [debugging](../tutorials/debug.md) or [parallelization](../tutorials/parallelize.md) for examples. |
    | `docs/deep-dive/` | Advanced explanations and in-depth concepts. |
    | `examples/<name>/README.md` | Example-specific README. If any related how-to if that exists, link to it avoid duplicating the same instructions twice; write only brief instructions on how to install and run the example. Otherwise, you can make the README more detailed and self-explanatory. |

    Remember to register new docs in [`mkdocs.yml`]({{ src("mkdocs.yml") }}), add examples to [examples/README]({{ src("examples/README.md") }}), and update the [Examples Catalog](../how-to/examples-catalog.md).

### 4. Test and Validate

Most contributions require automated checks. Prefix commands with `uv run` so they use the project environment.

**Full test suite**

```bash
uv run pytest -v
```

**Targeted tests**

```bash
uv run pytest tests/path/to/test_file.py -k test_name
```

**Optional/gated tests:** GPU-specific suites or API-dependent tests run automatically when the required hardware or environment variables (such as `OPENAI_API_KEY`) are present.

**Static analysis:**

```bash
uv run pyright
```

If you have touched code under `examples/`, you should run the example-specific smoke tests. Each directory includes a README with example-specific smoke tests—run those too.

!!! note "Build documentation when needed"

    Keep API references under [docs/reference]({{ src("docs/reference/") }}) up to date. Doc-only changes should still build cleanly:

    ```bash
    uv run mkdocs serve --strict   # live reload
    uv run mkdocs build --strict   # CI-equivalent
    ```

    `--strict` elevates warnings to errors so you catch issues before CI.

Before opening a PR, double-check the basics:

- Run `uv lock` if you changed dependencies.
- Run `uv run pre-commit run --all-files` (hooks installed via `pre-commit install` run automatically on `git commit`, but rerun them if you amended history).
- Execute the relevant commands from the test list above.
- Validate each affected example via its README instructions.

### 5. Open a Pull Request

1. Push your branch:
   ```bash
   git push origin <branch-name>
   ```
2. Open a PR against `microsoft/agent-lightning:main`.
3. Fill out the template with a concise summary, the commands/tests you ran, and linked issues (use `Fixes #123` syntax to auto-close).
4. Include screenshots or logs if they clarify behavior.
5. Address review feedback promptly. Follow-up tweaks work best as focused commits; `git commit --fixup` is handy for reviewer-suggested edits.

Thanks for contributing! every improvement strengthens the Agent Lightning community!
