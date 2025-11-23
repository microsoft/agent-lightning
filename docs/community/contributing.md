# Contributing Guide

Agent Lightning thrives on community improvements, whether you are polishing docs, fixing bugs, or building new features. This guide shows the shortest path from cloning the repository to shipping a polished pull request.

## What to Contribute

Agent-lightning is maintained by a small team at Microsoft Research. We seek for contributions from the community to help us improve the project. However, due to the limited human and computation resources of the core maintainence team, we kindly ask you to discuss with us first before you start working on a large-scale change. Contact the team via [Discord](https://discord.gg/RYk7CdvDR7).

The following guide might help you if you are interested in contributing to the project but do not know where to start.

### Documentation Improvements

Start with trivial documentation improvements, such as fixing typos, clarifying unclear descriptions, or adding missing links. Beginners are the best readers of documentation, so rewrite the documentation to make it more understandable and accessible to you will also benefit others.

You can find more about how to write good documentations and organize documentations in the following sections.

!!! note "Changes that will likely to be rejected"

    - Copying the same content from the existing documentation to the new documentation, and modifying it slightly.
    - Adding new `how-to` guides that are not related to any new example.
    - Adding translations of the documentation to other languages.

### Bug Fixes

Bug fixes are the fastest way to get familiar with the codebase:

- Look through the ["good first issue"](https://github.com/microsoft/agent-lightning/labels/good%20first%20issue) and ["bug"](https://github.com/microsoft/agent-lightning/labels/bug) labels. Comment on the issue to claim it so we know you are working on it.
- When you find an unreported bug, open an issue that includes repro steps, logs, and expected behavior before sending a pull request.
- Focus on independent fixes that do not introduce breaking API changes. Larger refactors should go through an RFC or maintainer sync first.

### New Examples

Due to limited computation and human resources of the maintainence team, we have a high bar for examples to be merged into the official repository. Right now, we only accept examples that satisfy at least one of conditions of below. For the compactness of examples, we prefer two or more conditions met.

- Illustrates how to use an agent framework that is significantly different from the examples we have already provided. For example, [LangChain](https://www.langchain.com/) and [LlamaIndex](https://www.llamaindex.ai/) is not considered as significantly different; but [LangChain](https://www.langchain.com/) is significantly different from [n8n](https://n8n.io/) because they have different orchestration paradigms, and also from [Vercel AI SDK](https://ai-sdk.dev/) because they are written in different programming languages.
- Illustrates a strong performance improvement on a real-world problem. We welcome contributions that have tuned a **real-world** agent on a **real-world** dataset, demonstrating the effectiveness of such tuning. Examples are like: tuning a search agent to perform better with Google search API; tuning the system prompt of a coding agent (e.g., Claude Code) to make it better on SWE-Bench.
- Illustrates how to integrate a new algorithm, particularly with new training or serving backends. More details are in the later "New Algorithms" section.
- Non-trivial cases that have otherwise never been experimentally verified. For example, training a multi-modality agent that supports images; training the memory / workflow of an agent.

We would appreciate examples with the following characteristics. They would be a bonus.

- Ships with CI or self-test coverage that keeps the example runnable over time. This will guarantee that the example won't break as the Agent-lightning codebase evolves. **Otherwise, we would have to mark the example as unmaintained because we won't be able to test the examples manually before each release.**
- Includes a how-to guide related to the example in the `docs/how-to/` directory. You can also write a more detailed README if there is no related how-to guide. Refrain from repeating the same content twice as it brings extra maintenance burden.
- Keep the example code simple and self-explanatory, avoiding complex abstractions and dependencies.

!!! warning "Important: Discussion first!"

    Make sure you have discussed with us first before you start working on a new example! The effort can be large and time-consuming, so we would like to make sure it's worth it.

### Fresh Implementations of Core Modules

This covers new implementations of components like [`Runner`][agentlightning.Runner], [`Tracer`][agentlightning.Tracer], [`Adapter`][agentlightning.Adapter], [`LightningStore`][agentlightning.LightningStore].

Before starting, file an issue or proposal that explains:

- Which interface you want to extend (e.g., a new [`Runner`][agentlightning.Runner] that uses another telemetry SDK).
- Why existing implementations are insufficient, and why the new implementation is useful.
- How you plan to test interoperability with the rest of the stack (unit tests, example updates, docs).

If you need to touch the APIs, please always discuss with us first before you start working on it.

### New Algorithms

This covers integrations of other algorithm backends. First check whether the algorithm you want to support is already supported in [Algorithm Zoo](../algorithm-zoo/index.md); or it's available as an example in [Examples Catalog](../how-to/examples-catalog.md).

We especially welcome:

- Currently unsupported or badly-tested algorithms like Supervised Fine-tuning (SFT), Direct Policy Optimization (DPO), Monte Carlo Tree Search (MCTS).
- Expanding the capabilities of currently supported algorithms. For example, adding multi-modality support to APO; adding multi-agent multi-prompt tuning to APO.
- For the Reinforcement Learning algorithms in particular, the most common tech stacks of reinforcement learning shown in Agent-lightning currently involves [VERL](https://github.com/volcengine/verl); [vLLM](https://vllm.ai/); [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-foundry/models/openai); [Tinker](https://tinker-docs.thinkingmachines.ai/). We welcome contributions that integrate with other backends like [SGLang](https://github.com/sgl-project/sglang); [TRL](https://github.com/huggingface/trl); [SkyRL](https://github.com/NovaSky-AI/SkyRL); [RLinf](https://github.com/RLinf/RLinf); [litgpt](https://github.com/Lightning-AI/litgpt).

Most new algorithms (unless existing algorithms improvments), will fall into the "new examples" category. Make sure that you have also read that section. Open an issue with your design doc so we can help scope the work, point you to existing utilities, and avoid overlapping efforts. Examples will be promoted to the "Algorithm Zoo" when they are ready enough.

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

### Step 3. Branching Workflow and Making Changes

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

!!! note "Putting the New Documentation and Examples in the Right Place"

    Many new contributors get confused about what to put in the `docs/how-to/` directory and what to put in the `examples/` directory (particularly README files). Here is a quick reference you can refer to:

    | Location | Description |
    | --- | --- |
    | `docs/algorithm-zoo/` | Put all documentation for **built-in algorithms** shipped with Agent-lightning here. |
    | `docs/how-to/` | Put all documentation for **how-to recipes** here. Usually, it's a self-contained guide that explains how an **example** is implemented. Accompanying code in `examples/` folder is encouraged. |
    | `docs/tutorials/` | Tutorials are explanations of how a specific component in Agent-lightning works. They can also explain a general workflow to accomplish something (e.g., [debugging](../tutorials/debug.md), [parallelization](../tutorials/parallelize.md)). |
    | `docs/deep-dive` | More advanced tutorials and concepts' explanations. |
    | `examples/<xxx>/README.md` | The README file for an example. If a related how-to guides exist, link it and only explain how to install the dependencies and run the example in the README. If there is no related how-to guide, you can make it more detailed and self-explained. |

    Remember to add the new documentation to the [`mkdocs.yml`]({{ src("mkdocs.yml") }}) file, and new examples to [Example README]({{ src("examples/README.md") }}) and [Examples Catalog](../how-to/examples-catalog.md).

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

**Optional/gated tests:** GPU-specific suites or API-dependent tests run automatically when the required hardware or environment variables (such as `OPENAI_API_KEY`) are present.

**Static analysis:**

```bash
uv run pyright
```

If you have touched code under `examples/`, you should run the example-specific smoke tests. Each directory includes a README with example-specific smoke tests—run those too.

!!! note "Build Documentation (When Applicable)"

    Make sure you have updated [references]({{ src("docs/reference/") }}) if you have changed the API.

    Doc changes should build cleanly before you push:

    ```bash
    uv run mkdocs serve --strict  # live reload while editing
    uv run mkdocs build --strict  # CI-equivalent validation
    ```

    `--strict` matches CI and promotes warnings to errors so you catch them early.

Before finalizing your pull request, you should again run the following checks:

- Run `uv lock` to update the lock file if you have changed `pyproject.toml` or other dependencies.
- Run `uv run pre-commit run --all-files` (hooks installed via `pre-commit install` run automatically on `git commit`, but rerun them if you amended history).
- Execute the relevant test commands from Step 4.
- Validate any affected examples by following the instructions in `examples/<name>/README`.

### Step 5. Open a Pull Request

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
