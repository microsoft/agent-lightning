# Instructions for Coding Agents

## Project Overview

agl-lite is a minimal workable version of [Agent Lightning](https://github.com/microsoft/agent-lightning) for agentic RL infrastructure. It provides an LLM gateway, data store, and K8s-based agent runner — all behind a single HTTP API.

## Key Documents

| Document | Path | Read When |
|----------|------|-----------|
| Architecture (full) | `docs/design/0_architecture.md` | Understanding the system design — **start here** |
| K8s Controller | `docs/design/1_k8s_controller.md` | Working on the controller or Job builder |
| Dev Guidelines | `docs/dev_guidelines.md` | Code style, conventions, concurrency model |
| Getting Started | `docs/get_started.md` | Setup flow and first run |

## Reading the Architecture Doc

The architecture doc (`docs/design/0_architecture.md`) is ~80KB. Read the TOC first:

```bash
grep "^##" docs/design/0_architecture.md
```

Then read only the sections relevant to your task.

## Project Layout

```
agl_lite/
  schemas/          # Pydantic v2 data models (shared across all components)
  store/            # In-memory data store
  gateway/          # LLM reverse proxy (model routing, param injection, event capture)
  server/           # FastAPI HTTP service (thin wrapper over store + gateway)
  controller/       # K8s controller (reconcile rollouts → Jobs)
  client.py         # Python client library (AglLiteClient)
  client_cli.py     # CLI tool (agl-client)
  cli.py            # Server/controller CLI (agl-lite serve, agl-lite controller)
  hooks.py          # RolloutHooks base class for customization
  verl/             # VERL training framework integration
tests/              # Unit, integration, and E2E tests
deploy/             # Dockerfiles, K8s manifests, .env config
examples/           # Math PoC (GSM8K), SWE-bench
```

## Local Environment

Local environment setup and secrets are in `.local/` (gitignored). See `.local/README.md`.

## Key Conventions

- **Python 3.12**, managed with `uv`
- **All route handlers must be `async def`** — never sync `def` (breaks thread safety)
- **Store methods are plain `def`** — synchronous, in-memory, no I/O
- **Ruff** for linting/formatting, **pyright** for type checking
- **Conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Run tests: `uv run pytest`
