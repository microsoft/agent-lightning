# Contributing Guide

Agent Lightning thrives on community improvements, whether you are polishing docs, fixing bugs, or building new features. This guide shows the shortest path from cloning the repository to shipping a polished pull request.

## Step 1. Prepare Your Environment

### Prerequisites

- **Python** 3.10 or newer (we test on 3.10–3.13).
- **uv** for dependency and virtual environment management. Install it from the [official uv docs](https://docs.astral.sh/uv/getting-started/installation/).
- **Git** configured with your GitHub credentials.

### Clone the Repository

Fork the repo, then clone your fork and register the upstream remote so you can stay current:

```bash
git clone git@github.com:<your-username>/agent-lightning.git
cd agent-lightning
git remote add upstream https://github.com/microsoft/agent-lightning.git
```

### Install Dependencies

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

---

## Step 2. Install and Run Pre-commit

We enforce formatting and linting with [pre-commit](https://pre-commit.com/). Install the hooks once, then run them before every push:

```bash
uv run pre-commit install

# The following will auto-run if you have set up the pre-commit hooks to run automatically on commit.
uv run pre-commit run --all-files --show-diff-on-failure --color=always
```

Running them locally saves a CI round-trip and keeps diffs tidy.

---

## Step 3. Branching Workflow

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

---

## Step 4. Test Your Changes

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

---

## Step 5. Build Documentation (When Applicable)

Doc changes should build cleanly before you push:

```bash
uv run mkdocs serve --strict  # live reload while editing
uv run mkdocs build --strict  # CI-equivalent validation
```

`--strict` matches CI and promotes warnings to errors so you catch them early.

---

## Step 6. Final Local Checks

- Run `uv run pre-commit run --all-files` (hooks installed via `pre-commit install` run automatically on `git commit`, but rerun them if you amended history).
- Execute the relevant test commands from Step 4.
- Validate any affected examples by following the instructions in `examples/<name>/README`.

---

## Step 7. Open a Pull Request

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
