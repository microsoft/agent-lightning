# Contributing to Agent Lightning

Thanks for your interest in Agent Lightning! This page covers the recommended contribution points,
how to set up a development environment, and what reviewers expect from a pull request.

## Ways to contribute

- **Bug reports.** A clear report is a contribution in itself. Include what you ran, what you
  expected, what happened, and a minimal reproduction. For agentic RL issues, the rollout logs and
  the shapes of the tensors involved are usually what makes a report actionable.
- **Examples.** `examples/` holds runnable harnesses for different agents and backends. New examples
  that show a supported workflow end to end are welcome.
- **Documentation.** `docs/` is built with MkDocs; corrections and clarifications to configuration
  semantics are especially useful, because the training behaviour is easy to misread.
- **Tests.** `tests/` runs on CPU with the `verl-cpu` dependency group (see below), so a regression
  test for a bug you hit is cheap to add and keeps the bug fixed.
- **Code.** `agentlightning/` is deliberately small: the server, the rollout controller and the VERL
  integration. Changes that keep that surface small are easier to review.

Look for issues labelled
[`good first issue`](https://github.com/microsoft/agent-lightning/labels/good%20first%20issue) or
[`help wanted`](https://github.com/microsoft/agent-lightning/labels/help%20wanted)
for tasks the maintainers have already scoped.

## Development environment

Agent Lightning requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for environments and
locking. The commands below are the ones CI runs, so running them locally reproduces CI exactly.

```bash
git clone https://github.com/microsoft/agent-lightning.git
cd agent-lightning

# Lint, formatting and type-checking
uv sync --frozen --no-default-groups --extra dev --group dev
uv run --locked --no-sync pre-commit run --all-files --show-diff-on-failure
uv run --locked --no-sync ruff check .
uv run --locked --no-sync ruff format --check .
uv run --locked --no-sync python scripts/check_headers.py

# Type checking and tests (verl-cpu pulls a CPU build of torch and VERL)
uv sync --frozen --no-default-groups --extra dev --group dev --group verl-cpu
uv run --locked --no-sync pyright
uv run --locked --no-sync pytest -v --durations=20 tests
```

The whole test suite runs on CPU — no GPU is needed for `pytest tests`. That is only true of the
tests: any real rollout-driven training needs the `verl` GPU stack, although no cluster is required
for the simplest path — the [Quick Start](https://github.com/microsoft/agent-lightning/blob/main/docs/01-quick-start.md)
trains on a single machine with one A100 using the local controller.

If you changed `docs/`, build them the way CI does:

```bash
uv sync --frozen --no-default-groups --group docs
uv run --locked --no-sync mkdocs build --strict
```

## Repository layout

| Path | What lives there |
| --- | --- |
| `agentlightning/server/` | The Agent Lightning server: rollout lifecycle, event storage, the LLM proxy and its routes. |
| `agentlightning/controller/` | Rollout controllers. In `k8s` mode each rollout runs as a Kubernetes Job; in `local` mode each rollout runs as a subprocess on the Controller machine, so no cluster is needed. |
| `agentlightning/verl/` | The VERL integration: rollout management, trace-to-training-row adapters, advantage and loss normalization. |
| `agentlightning/client.py`, `agentlightning/schemas.py` | Client used by the trainer and the shared request/response schemas. |
| `examples/` | Runnable harnesses (Calc-X, GSM8K, ScienceWorld, Search-R1, Coding Agent, Multimodal QA, ...). |
| `tests/` | The CPU test suite, including the adapter and normalization unit tests. |
| `docs/` | This documentation, built with MkDocs. |
| `skills/`, `.agents/skills/` | Agent-facing skills shipped with the repository. |

## Pull requests

- **Branch off `main`** with a short kebab-case name that says what the branch does, for example
  `fix/per-rollout-mean-normalization` or `docs/contributing-guide`, and keep it rebased on `main`
  while it is open. Do not open a pull request from your own `main`.
- **Keep it focused.** One problem per pull request; split unrelated cleanups out. Small diffs get
  reviewed much faster than large ones.
- **Link the issue.** Reference the issue your change addresses in the description, using
  `Fixes #123` when the pull request resolves it.
- **Say what, why and how you verified it.** What changed, why it is needed, and the exact commands
  you ran with their results. When you cannot run something (for example a multi-GPU path), say so
  explicitly instead of leaving it implied.
- **Follow the existing style.** Titles in the history look like `fix(verl): ...`,
  `fix(controller): ...` or `feat(server): ...`. Match the conventions of the module you touch, and
  keep comments and docstrings in English.
- **Add or update tests** for behaviour changes, and update the documentation when user-visible
  behaviour changes.
- **Keep CI green.** A pull request is ready for review when the lint, type-check, test, package and
  docs jobs pass.

## Using AI assistants

This repository ships agent-facing material under `.agents/` and `skills/`, and AI-assisted work is
fine here. Two expectations keep it reviewable: review the diff yourself so you can explain and
defend every line, and say in the pull request description that an assistant was involved, so
reviewers know how to read it. A patch whose author cannot explain it is not ready to be reviewed,
no matter how it was produced.

## Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you
have the right to, and actually do, grant us the rights to use your contribution. For details, visit
<https://cla.opensource.microsoft.com>.

When you open a pull request, a bot determines whether you need to provide a CLA and decorates the
pull request accordingly (for example with a status check or a comment). Follow the instructions the
bot provides. You only need to do this once across all repositories using our CLA.

## Reporting security issues

Please do not report security vulnerabilities through public GitHub issues. See
[SECURITY.md](https://github.com/microsoft/agent-lightning/blob/main/SECURITY.md) for the
reporting process.

## Code of conduct

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more
information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
