# Development Guidelines

Tooling, conventions, and implementation standards for agl-lite.

## Tooling

| Tool | Purpose | Version / Notes |
|------|---------|-----------------|
| **Python** | Runtime | 3.12 |
| **uv** | Package manager, build tool | Manages deps, venv, lockfile |
| **ruff** | Linter + formatter | Replaces black, isort, flake8 |
| **pyright** | Type checker | Better Pydantic v2 support than mypy |
| **pytest** | Test framework | + `pytest-asyncio` for async tests |

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP framework (agl-lite serve) |
| `uvicorn` | ASGI server |
| `pydantic` (v2) | Schema definitions, validation |
| `pydantic-settings` | Config from env vars (typed) |
| `httpx` | Async HTTP client (gateway proxy, controller client) |
| `kr8s` | Async K8s client (controller) |
| `typer` | CLI framework |
| `structlog` | Structured JSON logging |

## Project Layout

```
agl_lite/
  __init__.py
  schemas/                # shared data models (Pydantic v2)
    __init__.py
    rollout.py            # Rollout, RolloutStatus, RolloutConfig, Mount
    event.py              # Event, ModelRequestData, RewardData
    resources.py          # ResourcesUpdate, JobDefaults
    model_server.py       # ModelServer
    errors.py             # ConflictError, InvalidTransitionError
    api.py                # request/response body models
  store/                  # in-memory store
    __init__.py
    memory.py             # InMemoryStore
  server/                 # FastAPI app (serve entrypoint)
    __init__.py
    app.py                # FastAPI lifespan, mount routes
    auth.py               # API key middleware
    config.py             # ServerSettings (pydantic-settings)
    routes/
      __init__.py
      rollouts.py
      events.py
      models.py
      resources.py
      archive.py
    gateway.py            # LLM reverse proxy + event auto-capture
  controller/             # K8s controller (controller entrypoint)
    __init__.py
    reconciler.py         # reconcile loop
    job_builder.py        # Job spec rendering
    config.py             # ControllerSettings
  cli.py                  # typer CLI (serve, controller subcommands)
tests/
  __init__.py
  schemas/                # schema validation tests
  store/                  # store logic tests (state transitions, events, archive)
  server/                 # API integration tests (auth, routes, gateway)
  controller/             # controller tests (job builder, reconcile)
  e2e/                    # end-to-end tests
```

## Code Conventions

### Style
- **Line length**: 120
- **Formatter**: ruff format (applied on save / pre-commit)
- **Linter**: ruff check (all default rules + pyright type checking)
- **Imports**: sorted by ruff (isort-compatible)

### Ruff config (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]

[tool.ruff.format]
quote-style = "double"
```

### Pyright config (`pyproject.toml`)

```toml
[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "standard"
```

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions / variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Pydantic models: `PascalCase`, fields `snake_case`

### Async
- All store methods are `async def` (even if in-memory impl is synchronous) — keeps interface consistent for future DB backends
- All HTTP handlers are `async def`
- Use `httpx.AsyncClient` for outbound requests (gateway proxy, controller → agl-lite)

### Error handling
- `ConflictError` → HTTP 409 (version mismatch, invalid transition)
- `NotFoundError` → HTTP 404 (rollout, resource, model not found)
- `ForbiddenError` → HTTP 403 (valid key, wrong role)
- Pydantic `ValidationError` → HTTP 422 (automatic via FastAPI)
- Unexpected errors → HTTP 500 (log with structlog, include request context)

### Logging
- Use `structlog` with JSON output
- Bind context per request: `rollout_id`, `attempt_id`, `method`, `path`
- Log levels:
  - `info`: request handled, rollout state change, job created/deleted
  - `warning`: version conflict (retry expected), client disconnect
  - `error`: unexpected failures, K8s API errors

## Git Conventions

- **Commits**: conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- **Branch**: `main` only for MVP
- **Lockfile**: `uv.lock` committed to repo
- **No generated files**: `.gitignore` excludes `__pycache__`, `.venv`, `.pytest_cache`, etc.

## Testing

- **Unit tests**: schemas, store logic, job builder — no I/O
- **Integration tests**: FastAPI `TestClient` for HTTP layer, mock model servers for gateway
- **Controller tests**: mock K8s API (or kind/minikube for CI)
- **E2E tests**: full stack on minikube
- **Naming**: `test_{module}_{behavior}.py` or `test_{feature}.py`
- **Async tests**: use `@pytest.mark.asyncio` with `pytest-asyncio`

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=agl_lite

# Run specific test file
uv run pytest tests/store/test_rollouts.py
```

## Key References

| Document | Path | Content |
|----------|------|---------|
| Architecture | `docs/design/0_architecture.md` | Ground truth — data models, API spec, component design |
| K8s Controller | `docs/design/1_k8s_controller.md` | Controller implementation details |
| Getting Started | `docs/get_started.md` | Setup and first run guide |
| Implementation Plan | `dev/todo.md` | Phased task list |
| Issues | `dev/issues/README.md` | Architecture issue tracker (all resolved) |
