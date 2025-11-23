# Contributing Guide

Agent Lightning thrives on community improvements, whether you are polishing docs, fixing bugs, or building new features. This guide shows the shortest path from cloning the repository to shipping a polished pull request.

## What to Contribute

Agent-lightning is maintained by a small team at Microsoft Research. We seek for contributions from the community to help us improve the project. However, due to the limited human and computation resources of the core maintainence team, we kindly ask you to discuss with us first before you start working on a large-scale change. Contact the team via [Discord](https://discord.gg/RYk7CdvDR7).

The following guide might help you if you are interested in contributing to the project but do not know where to start.

### Documentation Improvements

Start with trivial documentation improvements, such as fixing typos, clarifying unclear descriptions, or adding missing links.

### Bug Fixes

Bug fixes are the fastest way to get familiar with the codebase:

- Look through the ["good first issue"](https://github.com/microsoft/agent-lightning/labels/good%20first%20issue) and ["bug"](https://github.com/microsoft/agent-lightning/labels/bug) labels. Comment on the issue to claim it so we know you are working on it.
- When you find an unreported bug, open an issue that includes repro steps, logs, and expected behavior before sending a pull request.
- Focus on independent fixes that do not introduce breaking API changes. Larger refactors should go through an RFC or maintainer sync first.

### New Examples

Due to limited resources, we have a high bar for new examples. We only accept examples that satisfy at least one of conditions of below. For the compactness of examples, we prefer two or more conditions met.

- Illustrates how to use an agent framework that is significantly different from the examples we have already provided. For example, [LangChain](https://www.langchain.com/) and [LlamaIndex](https://www.llamaindex.ai/) is not considered as significantly different; but [LangChain](https://www.langchain.com/) is significantly different from [n8n](https://n8n.io/) because they have different orchestration paradigms, and also from [Vercel AI SDK](https://ai-sdk.dev/) because they are written in different programming languages.
- Illustrates a strong performance on a benchmark dataset.
- Demonstrates a training or serving backend that is otherwise hard to reproduce (e.g., Tinker integration, MCP-based tools, proprietary deployment pipelines).
- Ships with CI or self-test coverage that keeps the example runnable over time.
- Includes a full walkthrough in the `examples/<name>/README.md` plus cross-links to the relevant `docs/how-to/*` article.

### Fresh Implementations of Core Modules

This covers new implementations of components like [`Runner`][agentlightning.Runner], [`Tracer`][agentlightning.Tracer], [`Adapter`][agentlightning.Adapter], [`LightningStore`][agentlightning.LightningStore].

Before starting, file an issue or proposal that explains:

- Which interface you want to extend (e.g., a new `Tracer` that exports to another telemetry sink).
- Why existing implementations are insufficient.
- How you plan to test interoperability with the rest of the stack (unit tests, example updates, docs).

### New Algorithms

This covers integrations of other algorithm backends. First check whether the algorithm you want to support is already supported in [Algorithm Zoo](../algorithm-zoo/index.md); or it's available as an example in [Examples Catalog](../how-to/examples-catalog.md).

We especially welcome:

- Connectors to reinforcement-learning frameworks (e.g., VERL, Tinker, bespoke schedulers) that can be toggled via `agl.Trainer`.
- Prompt or policy optimization loops that expose clean configuration surfaces and include reproducible training/eval scripts.
- Recipes that demonstrate migration paths from raw scripts to integrated Agent-lightning components.

Open an issue with your design doc so we can help scope the work, point you to existing utilities, and avoid overlapping efforts.

### Ecosystem Projects

We are looking for projects that build on top of Agent-lightning. If you think your project can benefit the community, while not wanting to go through the review and discussion process of adding a new example or enhancement, please create a fork of Agent-lightning or a new project that uses Agent-lightning as a dependency. We will be happy to list your project in [Community Projects](../index.md) and [README]({{ src("README.md") }}).

### Other Contribution Ideas

- **Test cases.** Additions to `tests/` or other integration checks that make regressions easier to catch.
- **Benchmarks.** Additions to `tests/benchmark` to perform pressure tests on Agent-lightning, particularly for large-scale training and rollouts.
- **Issue triage.** Reproducing reported bugs, confirming whether they still happen on `main`, or suggesting short-term mitigations helps maintainers prioritize fixes.

## How to Contribute

### Step 1. Prepare Your Environment

You should at least have:

- **Python** 3.10 or newer (we recommend 3.12).
- **uv** for dependency and virtual environment management. Install it from the [official uv docs](https://docs.astral.sh/uv/getting-started/installation/).
- **Git** configured with your GitHub credentials.

Then fork the repo, then clone your fork and register the upstream remote so you can stay current:

```bash
git clone git@github.com:<your-username>/agent-lightning.git
cd agent-lightning
git remote add upstream https://github.com/microsoft/agent-lightning.git
```

Install the standard development toolchain:

```bash
uv sync --group dev
```

Want GPU extras, example dependencies, or other optional features? Pin everything in one pass:

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

After `uv sync`, run commands with `uv run ...` (or `uv run --no-sync` once the environment is locked), or activate the virtual environment in `.venv/`.

### Step 2. Install and Run Pre-commit

We enforce formatting and linting with [pre-commit](https://pre-commit.com/). Install the hooks once, then run them before every push:

```bash
uv run pre-commit install

# The following will auto-run if you have set up the pre-commit hooks to run automatically on commit.
uv run pre-commit run --all-files --show-diff-on-failure --color=always
```

Running them locally saves a CI round-trip and keeps diffs tidy.

### Step 3. Branching Workflow

Start from a fresh `main`, then branch for your change:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

Create a topic branch with one of these prefixes:

- `feature/<short-description>` for new features
- `fix/<short-description>` for bug fixes
- `docs/<short-description>` for documentation-only work
- `chore/<short-description>` for tooling or maintenance

Stick to lowercase words separated by hyphens, e.g. `feature/async-runner-hooks`.

### Step 4. Test Your Changes

Most updates should ship with automated checks. Preface commands with `uv run` so they use the project environment.

**Full test suite**

```bash
uv run pytest -v
```

**Targeted tests**

```bash
uv run pytest tests/path/to/test_file.py -k test_name
```

**Optional/gated tests**

GPU-specific suites or API-dependent tests run automatically when the required hardware or environment variables (such as `OPENAI_API_KEY`) are present.

**Static analysis**

```bash
uv run pyright
```

Touching code under `examples/`? Each directory includes a README with example-specific smoke tests—run those too.

### Step 5. Build Documentation (When Applicable)

Doc changes should build cleanly before you push:

```bash
uv run mkdocs serve --strict  # live reload while editing
uv run mkdocs build --strict  # CI-equivalent validation
```

`--strict` matches CI and promotes warnings to errors so you catch them early.

### Step 6. Final Local Checks

- Run `uv run pre-commit run --all-files` (hooks installed via `pre-commit install` run automatically on `git commit`, but rerun them if you amended history).
- Execute the relevant test commands from Step 4.
- Validate any affected examples by following the instructions in `examples/<name>/README`.

### Step 7. Open a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin <branch-name>
   ```
2. Open a PR against `microsoft/agent-lightning:main`.
3. Complete the PR template with:
    - A concise summary of the change.
    - The tests or commands you ran (copy from Step 4/6).
    - Linked issues (use `Fixes #123` to auto-close).
4. Attach screenshots or terminal output when it clarifies behavior.
5. Address review feedback promptly. Use focused commits, and consider `git commit --fixup` for follow-up adjustments.

Thanks for contributing—every improvement grows the Agent Lightning community!
