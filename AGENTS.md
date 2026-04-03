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

## Config Classes Are Pure Data — CLI Owns Env Var Mapping

All config/settings classes are plain `pydantic.BaseModel` — pure data carriers with no ambient environment access. **The CLI (`cli.py`) is the only place that reads `os.environ`** and constructs config objects explicitly.

**Never do this:**
```python
# BAD — BaseSettings reads os.environ implicitly
class ServerSettings(BaseSettings):
    gateway_config: str | None = None  # silently reads GATEWAY_CONFIG from env
    model_config = {"env_prefix": "AGL_"}
```

**Do this instead:**
```python
# GOOD — plain model, constructed explicitly in cli.py
class ServerSettings(BaseModel):
    gateway_config: str | None = None

# In cli.py:
settings = ServerSettings(
    gateway_config=os.environ.get("AGL_GATEWAY_CONFIG"),
)
```

**Why:**
- Tests construct settings with zero `monkeypatch.setenv` noise
- All env var → field mappings are visible in one file (`cli.py`), not scattered across classes with different `env_prefix` rules
- No silent ambient reads — if a field is missing, the failure is explicit at the CLI boundary
- `BaseSettings` `env_prefix` rules are subtle and error-prone (e.g. `env_prefix=""` + field `agl_key` accidentally reads `AGL_KEY` while `gateway_config` reads `GATEWAY_CONFIG`, not `AGL_GATEWAY_CONFIG`)

**The one exception:** `DeploySettings` parses an explicit `.env` file given by the user via `--env-file`. This is not ambient env reading — it's "parse this specific file into a struct." It may use `python-dotenv` for parsing, but the result is still constructed explicitly.

**The rule:** if a class reads `os.environ` without being explicitly told to, it violates this principle.



agl-lite is system-level infrastructure code that is called through deep stacks — CLI → controller → reconciler → job builder → K8s. At this level, default values are a liability, not a convenience.

**Default values are appropriate for:**
- Genuinely optional tuning knobs (`ttl_after_finished`, `poll_interval`, `timeout`)
- Feature flags that are off by default
- Collection fields that are empty when absent (`env: list = []`)

**Default values are wrong for:**
- Inputs that represent a real decision the caller must make consciously — especially across module or layer boundaries
- Paths, URLs, or external resources that may not exist in all environments
- Any value that silently changes system behavior when the caller forgets to pass it

**The rule:** if omitting an argument would mask a bug or produce silently wrong behavior, it must be required — no default. Fail loudly at startup rather than fail mysteriously at runtime.

**Concrete example from this codebase:** `build_job_spec(..., manifest_template: str)` has no default. An earlier version had `_DEFAULT_MANIFEST_TEMPLATE_PATH` that resolved to a path valid in the source tree but not in an installed Docker image — a bug that would have been invisible until deployment. Making it required forces every caller to be explicit about where the template comes from.
