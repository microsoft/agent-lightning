# Repository Guidelines

## Architecture Overview
Agent Lightning loops through stages: runner and tracer emit spans, LightningStore (`agentlightning/store/`) synchronizes them, and algorithms in `agentlightning/algorithm/` learn from the traces.

## Project Structure & Module Organization
- `agentlightning/`: adapters, runner/execution stack, trainer, tracer, reward logic, `agl` CLI.
- `docs/` & `examples/`: documentation (assets in `docs/assets/`, nav in `mkdocs.yml`) plus runnable workflows whose READMEs link to their how-to guides.
- `dashboard/`, `scripts/`, `tests/`: UI bundles, automation for releases/datasets/CI, and coverage mirrors of the runtime tree—document download steps instead of committing binaries.

## Build, Test, and Development Commands
- `uv sync --group dev` — install tooling once.
- `uv run --no-sync pytest -v` — run suites; add a path or `-k expr` for targeted loops.
- `uv run --no-sync pyright` — static analysis aligned with CI.
- `uv run --no-sync pre-commit run --all-files --show-diff-on-failure` and `uv run --no-sync mkdocs build --strict` — formatting/lint hooks plus doc validation.
Commit the refreshed `uv.lock` whenever dependencies move and note optional groups (VERL, APO, GPU) in PRs.

## Coding Style & Naming Conventions
- Stay compatible with `requires-python >= 3.10`, use 4-space indentation, 120-character lines, and formatter-managed diffs (Black + isort, profile `black`).
- Adopt `snake_case` for modules/functions/variables, `PascalCase` for classes, and lowercase-hyphenated CLI flags or branch names.
- Keep type hints exhaustive (pyright enforced) and reuse dataclasses/Pydantic models from `agentlightning.types`.
- Use Google-style docstrings for new modules or public methods, keeping descriptions short and avoiding redundant type annotations. Use `[][]` syntax for cross-references.

## Testing Guidelines
- Mirror runtime directories under `tests/` and align filenames for quick lookup.
- Parametrize pytest cases and apply markers (`openai`, `gpu`, `agentops`, `mongo`, `llmproxy`) so optional suites can be skipped with selectors like `-m "not mongo"` yet still run in CI.
- Favor fixtures, fake spans, and local LightningStore instances; gate unavoidable external calls with the relevant marker and mention required environment variables when present.

## Commit & Pull Request Guidelines
- Branch from fresh `main` using `feature/<slug>`, `fix/<slug>`, `docs/<slug>`, or `chore/<slug>`.
- Write imperative, scoped commits, reference issues with `Fixes #123`, and rerun pre-commit plus relevant pytest/doc builds before pushing.
- PR descriptions should summarize intent, list verification commands, highlight dependency or docs-navigation updates, and link new docs/examples via `mkdocs.yml` or `examples/README.md`; include logs for dashboard tweaks.
