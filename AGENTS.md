# Repository Guidelines for Agent-lightning

## Project Structure & Module Organization

The `agentlightning/` package holds adapters, runners, tracers, reward logic, and the `agl` CLI entry point. Docs, tutorials, and the algorithm zoo live under `docs/`, while runnable workflows and benchmarking recipes sit in `examples/`. Dashboards and UI assets live in `dashboard/`; automation scripts live in `scripts/`; regression coverage mirrors runtime modules inside `tests/`. Register new documentation in `mkdocs.yml` and keep example READMEs brief by pointing to the relevant how-to guide.

## Build, Test, and Development Commands

Run `uv sync --group dev` to install tooling, then execute everything through `uv run --no-sync <cmd>`. Core commands:

- `uv run --no-sync pytest tests/path/to/test_file.py` — run unit-test (use `-k` for targeted debugging and `-m` for skipping tests).
- `uv run --no-sync pyright /path/to/file.py` — static type validation aligned with CI.
- `uv run --no-sync pre-commit run --all-files --show-diff-on-failure` — applies Black, isort, Flake8, and lint hooks.
- `uv run --no-sync mkdocs build --strict` — validates docs.

Record extra dependencies with `uv lock` so reviewers can replay the environment.

## Coding Style & Naming Conventions

Python 3.12 features are allowed, but stay compatible with `requires-python = ">=3.10"`. Use 4-space indentation, 120-character lines, and Black + isort (profile `black`) formatting; do not hand-edit generated diffs. Modules, functions, and variables use `snake_case`; classes use `PascalCase`; CLI and branch names use lowercase hyphenated tokens. Keep type hints up to date (pyright enforces them; especially for new tests) and favor dataclasses/Pydantic models already defined in `agentlightning.types`. Use Google-style docstrings for new modules, classes, and functions; use `[][]` syntax for symbol references; refrain from adding type hints to parameters in docstrings.

## Testing Guidelines

Add or update `tests/` cases whenever touching logic. Prefer colocated modules (e.g., `tests/execution/` for trainer code) and mirror filenames. Use parametrized tests plus `pytest.mark` gates (`openai`, `gpu`, `agentops`, `mongo`, `llmproxy`) so hardware/API-dependent suites can be skipped locally via `-m "not gpu"`. Favor fixtures or fake spans instead of live services, and call out required environment variables (like `OPENAI_API_KEY`) near the test. When testing, target specific files or units; refrain from running the entire test suite.

## Commit & Pull Request Guidelines

Work from fresh `main` and branch using `feature/<slug>`, `fix/<slug>`, `docs/<slug>`, or `chore/<slug>`. Write imperative, scoped commits and reference issues with `Fixes #123` where applicable. Before pushing, rerun hooks and the relevant tests; attach logs or screenshots for UI/dashboard tweaks. PRs should summarize intent, list the commands you ran, and surface dependency or doc-navigation updates so reviewers can trace coverage quickly.
