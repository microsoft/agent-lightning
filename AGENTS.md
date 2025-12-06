# Repository Guidelines

## Architecture Overview
Agent Lightning runs through a continuous loop: runners and tracers emit spans, `LightningStore` (`agentlightning/store/`) keeps them synchronized, and algorithms in `agentlightning/algorithm/` consume those traces to improve behavior.

## Project Structure & Module Organization
- `agentlightning/`: adapters, execution stack, training loop, tracer, reward logic, and the `agl` CLI.
- `docs/` & `examples/`: narrative and procedural docs (assets in `docs/assets/`, navigation in `mkdocs.yml`) plus runnable workflows whose READMEs point to their companion how-to guides. `docs/how-to` covers task-focused instructions, while `docs/tutorials` explains concepts and subsystems.
- `dashboard/`, `scripts/`, `tests/`: UI bundles, release/dataset/CI automation, and mirrored coverage of the runtime tree. Record download steps rather than committing binaries.

## Build, Test, and Development Commands
- `uv sync --group dev` — provision tooling once per environment.
- `uv run --no-sync pytest -v` — execute the full suite; add a path or `-k expr` to narrow the run.
- `uv run --no-sync pyright` — enforce static typing parity with CI.
- `uv run --no-sync pre-commit run --all-files --show-diff-on-failure` and `uv run --no-sync mkdocs build --strict` — keep formatting tidy and documentation valid.
Always commit the refreshed `uv.lock` when dependencies shift, and mention optional groups (VERL, APO, GPU) in PR notes.

## Coding Style & Naming Conventions
- Target `requires-python >= 3.10`, four-space indentation, 120-character lines (though docstrings may run longer), and formatter-owned diffs (Black + isort, `black` profile). Use `snake_case` for modules, functions, and variables; `PascalCase` for classes and React components; lowercase hyphenation for CLI flags, branch names, and TypeScript filenames.
- Maintain exhaustive type hints (pyright enforces them) and prefer shared dataclasses or Pydantic models from `agentlightning.types`.
- Author Google-style docstrings for new modules or public methods—succinct descriptions, no redundant type info, no redundant `Key features/components` bullet points, and `[][]` syntax for cross-references.
- Writing logs is encouraged, especially for long functions with multiple steps and try-except blocks that catch all exceptions. Use `logging.getLogger(__name__)` to get loggers. Distinguish between DEBUG, INFO, WARNING, and ERROR logs.

## Testing Guidelines
- Mirror runtime directories under `tests/` and match filenames for quick traceability.
- Parametrize pytest cases and apply markers (`openai`, `gpu`, `agentops`, `mongo`, `llmproxy`) so optional suites can be skipped via selectors like `-m "not mongo"` yet still exercised in CI.
- Lean on fixtures, favor real stores/spans/agents over mocks, and drive coverage across the majority of branches.
- If an imported module is missing from the environment, check whether `uv sync` has been run with the right groups. Do not make stubs for external dependencies unless necessary.

## Example Contributions
- Ship each example with a README that includes smoke-test instructions so maintainers can validate quickly. The README must contain an "Included Files" section summarizing every file and its role.
- Keep runnable example modules self-contained with a module-level docstring describing CLI usage. Document important or educational classes/functions with targeted docstrings and inline comments where clarity matters.
- Add a CI workflow per example named `examples-<name>.yml` in `.github/workflows/`. Register it in `badge-<name>.yml`, `badge-examples.yml`, and `badge-latest.yml` when applicable so badges stay accurate.

## Commit & Pull Request Guidelines
- Branch from a fresh `main` using `feature/<slug>`, `fix/<slug>`, `docs/<slug>`, or `chore/<slug>`.
- Write imperative, scoped commits, reference issues with `Fixes #123`, and rerun pre-commit plus the relevant pytest/doc builds before pushing.
- Use PR descriptions to summarize intent, list verification commands, call out dependency or docs-navigation updates, and link new docs/examples via `mkdocs.yml` or `examples/README.md`. Include logs for dashboard changes.
