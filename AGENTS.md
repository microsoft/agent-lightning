# Repository Guidelines

## Architecture Overview
Agent Lightning loops through stages: runner and tracer emit spans, LightningStore (`agentlightning/store/`) synchronizes them, and algorithms in `agentlightning/algorithm/` learn from the traces.

## Project Structure & Module Organization
- `agentlightning/`: adapters, runner/execution stack, trainer, tracer, reward logic, `agl` CLI.
- `docs/` & `examples/`: documentation (assets in `docs/assets/`, nav in `mkdocs.yml`) plus runnable workflows whose READMEs link to their how-to guides. `docs/how-to` contains step-by-step guides for accomplishing a specific task; `docs/tutorials` contains conceptual walkthroughs for components or features.
- `dashboard/`, `scripts/`, `tests/`: UI bundles, automation for releases/datasets/CI, and coverage mirrors of the runtime tree—document download steps instead of committing binaries.

## Build, Test, and Development Commands
- `uv sync --group dev` — install tooling once.
- `uv run --no-sync pytest -v` — run suites; add a path or `-k expr` for targeted loops.
- `uv run --no-sync pyright` — static analysis aligned with CI.
- `uv run --no-sync pre-commit run --all-files --show-diff-on-failure` and `uv run --no-sync mkdocs build --strict` — formatting/lint hooks plus doc validation.
Commit the refreshed `uv.lock` whenever dependencies move and note optional groups (VERL, APO, GPU) in PRs.

## Coding Style & Naming Conventions
- Stay compatible with `requires-python >= 3.10`, use 4-space indentation, 120-character lines, and formatter-managed diffs (Black + isort, profile `black`). `snake_case` for modules/functions/variables, `PascalCase` for classes and React components, and lowercase-hyphenated CLI flags, branch names and TypeScript files.
- Keep type hints exhaustive (pyright enforced) and reuse dataclasses/Pydantic models from `agentlightning.types`.
- Use Google-style docstrings for new modules or public methods, keeping descriptions short and avoiding redundant type annotations. Use `[][]` syntax for cross-references.

## Testing Guidelines
- Mirror runtime directories under `tests/` and align filenames for quick lookup.
- Parametrize pytest cases and apply markers (`openai`, `gpu`, `agentops`, `mongo`, `llmproxy`) so optional suites can be skipped with selectors like `-m "not mongo"` yet still run in CI.
- Favor fixtures; prefer real stores/spans/agents over fakes. Make sure most branches are covered by tests.

## Example Contributions
- Examples should include a README with smoke instructions so maintainers can validate them quickly. The README should contain a "Included Files" section with a list of the files in the example directory and a description of what each file does.
- The major example files that are meant to be run should be self-contained and have a module-level docstring with CLI usage instructions. Important/complex classes/functions that are meant to be read and learned for users should be documented with their own docstrings and inline comments.
- New examples should include a CI workflow that runs the example and verifies that it runs successfully. The workflow should be named `examples-<name>.yml` and should be placed in the `.github/workflows/` directory. The workflow should be registered in `badge-<name>.yml`, `badge-examples.yml`, and `badge-latest.yml` if necessary.

## Commit & Pull Request Guidelines
- Branch from fresh `main` using `feature/<slug>`, `fix/<slug>`, `docs/<slug>`, or `chore/<slug>`.
- Write imperative, scoped commits, reference issues with `Fixes #123`, and rerun pre-commit plus relevant pytest/doc builds before pushing.
- PR descriptions should summarize intent, list verification commands, highlight dependency or docs-navigation updates, and link new docs/examples via `mkdocs.yml` or `examples/README.md`; include logs for dashboard tweaks.
