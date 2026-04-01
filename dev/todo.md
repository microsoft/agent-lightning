# agl-lite Implementation Plan

> Aligned with the final architecture in `docs/design/0_architecture.md`
> and reviewed architecture decisions.

## Guiding Principles

1. **One repo, one package, two entrypoints**: `agl-lite serve` and `agl-lite controller`
2. **Controller talks to service only over HTTP** — no shared in-memory state
3. **Shared code limited to types/schemas** — both entrypoints import the same models
4. **Test each layer before building the next** — schemas → store → API → gateway → controller
5. **Freeze normative contracts early** — schemas, state transitions, auth matrix, event ordering

---

## Completed

- [x] **Phase 0**: Schemas, state transitions, project skeleton (Pydantic models, dev tooling)
- [x] **Phase 1**: In-memory store — rollouts (batch), events, resources, models, archive. 156 tests.
- [x] **Phase 2**: HTTP API — FastAPI app, auth, all store routes, gateway (config/router/proxy with wildcard + list-based routes), CLI, streaming SSE.
- [x] **Phase 3**: K8s controller — Python client, job builder, reconciler (create/watch/cancel/crash-recovery), CLI. 254 tests total.
- [x] **Phase 4a**: E2E with mock model server (CPU-only minikube). 271 unit tests + 9 e2e/cpu tests.
  - Kr8s adapter, `agl-client` CLI, deploy structure, example agents
  - `job_template` refactor (raw K8s pod spec, name-matched container overrides)
  - Math PoC mock mode: 2-iteration RL loop with weight update (v1→v2), streaming, deterministic rewards via `\boxed{}` embedding, model_request event validation, reference log
  - Scripts: `build_images.sh`, `deploy.sh`, `run.sh` (one-command E2E with log capture)
- [x] **Phase 4b**: E2E with real vLLM (GPU).
  - vLLM via Docker (`vllm/vllm-openai:latest`) on host GPU; `scripts/start_vllm.sh`
  - Colocated topology: agl-lite + vLLM + algorithm on host, only controller + agents in minikube
  - Gateway param injection: `gateway-config.yaml` adds `return_token_ids: true` to all requests; prompt + response token IDs captured in `model_request` events
  - `rl_loop.py` — real algorithm: plain questions, numeric reward, token_id verification
  - `deploy.sh --no-serve` for controller-only K8s deployment
  - Two `.env` examples (`.env.mockai.example`, `.env.vllm.example`), mode-aware `run.sh`
  - Reference logs for both modes; event captures prepared body (with injected params)
  - 271 unit tests passing, both modes E2E verified
- [x] **Store Hooks**: `RolloutHooks` base class with `on_enqueue`/`on_succeeded`/`on_failed`.
  Hooks run as sync pre-processors inside store (atomic, single-threaded). `--hooks` CLI flag.
  `load_hooks()` dynamic loader. `metadata` field on Rollout for algorithm control indexes.
  `AGL_TASK_INPUT` removed from job_builder (hook sets it explicitly). 12 hook tests.
- [x] **Math-poc restructuring**: Unified `rl_loop_v2.py` (~200 lines) replaces two ~480-line
  scripts. Mode subfolders (`mock/`, `vllm/`) with per-mode `hooks.py`, `.env.example`,
  `gateway-config.yaml`, `job-template.yaml`. Algorithm only sets `input`, `resources_id`,
  `metadata`. E2E verified with vLLM (3/3 rollouts, hooks computed rewards). 7 hook tests.

---

## Phase 4a.7: Additional E2E scenarios [backlog]

Not on the critical path — the happy-path lifecycle is fully validated. Add when needed.

- [ ] **Cancel test**: enqueue → cancel mid-run → verify cancelled status
- [ ] **Retry test**: agent with `CRASH_ON_FIRST=1` → K8s Job retries → succeeds on second attempt
- [ ] **503 test**: agents hitting gateway during model deregistration window

---

## Phase 4b.3: Performance baseline [backlog]

- [ ] Measure: rollout throughput, gateway proxy latency overhead, event capture overhead
- [ ] Compare direct vLLM vs gateway-proxied vLLM

---

## SWE-bench follow-ups [backlog]

- [ ] Remove artifact event support from store if no remaining consumers (separate cleanup task)

---

## Phase 5: VERL Algorithm Integration

**Goal**: Connect VERL's PPO training loop to agl-lite as the data pipe. agl-lite handles
rollout orchestration, event collection, and triplet extraction; VERL handles tensor
construction and weight updates.

### Background: Agent Lightning daemon structure (1154 lines)

The `AgentModeDaemon` in Agent Lightning is the bridge between the store and the VERL trainer.
The trainer calls 4 daemon methods: `set_up_data_and_server`, `run_until_all_finished`,
`get_train_data_batch`, `clear_data_and_server`. The daemon internally calls 4 store methods
(`add_resources`, `enqueue_many_rollouts`, `wait_for_rollouts`, `query_spans`) plus an adapter
(`spans → triplets`), then builds padded tensors from triplets.

Breakdown of what to replace vs reuse:

| Category | Lines | % | Action |
|---|---|---|---|
| Store interaction | 209 | 18% | **Replace** (agl-lite HTTP API) |
| Proxy server | 141 | 12% | **Drop** (gateway replaces) |
| Init (mixed) | 72 | 6% | **Simplify** |
| `get_train_data_batch` | 328 | 28% | **Reuse** |
| Multimodal (mrope) | 63 | 5% | **Reuse** (not tested in MVP) |
| Validation/metrics | 106 | 9% | **Reuse** |
| Utilities (padding) | 157 | 14% | **Reuse** |

### Key design: Triplet API in agl-lite

Instead of fetching raw events and converting on the VERL side, agl-lite itself provides
a triplet endpoint. This keeps event→triplet conversion in the data pipe where it belongs.

```
Agent Lightning original:
  daemon → store.query_spans() → adapter.adapt(spans) → List[Triplet]
           ~~~~~~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~~~~~~~
           complex Span objects   600-line tree adapter

agl-lite:
  daemon → GET /api/rollouts/{id}/triplets → List[Triplet]
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           server does events→triplets internally (~50 lines)
```

The triplet extraction logic in agl-lite is simple because events are flat:
- `model_request` events have `response.prompt_token_ids` and `response.choices[].token_ids`
- `reward` events have `data.value`
- Match by order (each model_request pairs with the following reward)
- Streaming responses: gather `token_ids` across SSE chunks, `prompt_token_ids` from first chunk

### Phase 5a: Triplet API (server-side, in agl-lite) ✅

- [x] **5a.1**: `format=triplet` on `GET /api/events` — trims model_request to
  `prompt_token_ids` + `response_token_ids`, reward to scalar value.
  Handles streaming (gather token_ids across SSE chunks) and non-streaming.
  4 API tests.

- [x] **5a.2**: `AglLiteClient.get_events(format="triplet")` — passes format param.

### Phase 5b: AglLiteDaemon (VERL-side bridge) ✅

Implemented as standalone class (not subclass) in `agl_lite/verl/daemon.py`.
Uses `AglLiteClient` for all HTTP calls. Same 4 methods the trainer expects.

**Dependency direction**: Agent Lightning trainer → agl-lite HTTP API.
agl-lite is a standalone HTTP service. AglLiteDaemon copies the tensor math
from Agent Lightning so that agl-lite can eventually be a full replacement.

```
agl-lite (this repo)                VERL trainer
─────────────────────               ─────────────
HTTP API:                           AglLiteDaemon:
  POST /api/rollouts         ←───    set_up_data_and_server()
  GET  /api/rollouts/{id}    ←───    run_until_all_finished()
  GET  /api/events?format=triplet ←  _async_validate_data()
  POST /api/resources/models ←───    _async_set_up()
                                    get_train_data_batch() → DataProto
```

`agl_lite/verl/daemon.py` (851 lines):
- **NEW** (187 lines): store interaction via AglLiteClient
  - `_async_set_up`: `client.register_models()` + `client.enqueue_rollouts()`
  - `_async_validate_data`: `client.get_events(format="triplet")` → Triplet/RolloutLegacy
  - `_async_run_until_finished`: poll `client.get_rollout()` for succeeded status
  - No proxy server, no adapter, no LightningStore
- **COPIED** (510 lines): from agent-lightning `AgentModeDaemon`
  - `get_train_data_batch`: triplets → padded tensors → DataProto (transition + trajectory levels)
  - Multimodal (mrope, image grid for Qwen2-VL)
  - Utilities (left/right padding, token matching, native conversion)
  - Validation/metrics

9 tests (5 utility, 4 daemon with real agl-lite server via ASGI transport).

### Phase 5c: Full training loop E2E [ongoing]

- [ ] Training script using agl-lite + VERL on Qwen2.5-1.5B-Instruct
- [ ] Weight update protocol: after PPO step, update vLLM model weights
- [ ] Multi-iteration training with measurable reward improvement
- [ ] Align rollout config between Agent Lightning and agl-lite: `RolloutConfig(unresponsive_seconds, timeout_seconds)` vs agl-lite's `config.timeout` — field names, semantics (active deadline vs unresponsive timeout), and K8s Job mapping need reconciliation

#### Next actions

- [x] Add a minimal executable training script under `examples/math-verl/` (`train.py`) that builds VERL config + dataset and calls `run_ppo()`.
- [ ] Migrate `examples/calc_x/` from Agent Lightning as the primary apple-to-apple training example (ignore `legacy_*` files):
  - adapt `train_calc_agent.py` to agl-lite VERL entrypoint (`run_ppo`) and agl-lite service auth/url flow
  - replace `agentlightning` runtime dependencies in `calc_agent.py`/`eval_utils.py` with agl-lite-compatible agent + reward/event flow
  - provide a self-contained run script with explicit external lifecycle boundary (`deploy.sh --controller-only` outside, host `agl-lite serve` inside run script)
- [ ] Add pre-flight checks in calc_x script/docs: healthz, auth, model registration, rollout completion, triplet extraction, non-empty PPO batch.

---

## Phase 6: Documentation [ongoing]

**Goal**: Create comprehensive user-facing documentation using mkdocs, organized for
different audiences (new users, researchers, operators, contributors).

### Structure

```
docs/
├── index.md                          # Home — what is agl-lite, key design choices, quick links
├── getting-started/
│   ├── index.md                      # Overview of setup flow (the 7-step diagram)
│   ├── prerequisites.md              # Docker, uv, Node.js, kubectl
│   ├── minikube.md                   # Minikube setup
│   └── quickstart.md                 # First run — math-poc vLLM mode
├── concepts/
│   ├── index.md                      # How agl-lite works (high-level, 3 groups diagram)
│   ├── gateway.md                    # Transparent LLM proxy, model routing, param injection, event capture
│   ├── store.md                      # Data model: rollouts, events, resources, model servers
│   ├── controller.md                 # K8s reconciliation, Job lifecycle, attempt = pod UID
│   ├── agent-contract.md             # Env vars, language-agnostic, no SDK dependency
│   ├── data-model.md                 # Event types, trajectory format, triplet format
│   └── weight-updates.md             # Weight update protocol (DELETE → re-POST cycle)
├── user-guide/
│   ├── deployment.md                 # deploy/ structure, .env config, build & deploy scripts
│   ├── configuration.md              # Server settings, gateway YAML config, controller settings
│   ├── writing-agents.md             # How to write an agent (Python, JS, any language)
│   ├── running-experiments.md        # Enqueue rollouts, poll status, retrieve trajectories
│   ├── hooks.md                      # RolloutHooks: on_enqueue, on_succeeded customization
│   └── verl-integration.md           # AglLiteDaemon, training loop, triplet format
├── examples/
│   ├── math-poc.md                   # GSM8K end-to-end (mock + vLLM modes)
│   └── swe-bench.md                  # SWE-bench with Claude Code agent
├── reference/
│   ├── api.md                        # Full REST API spec (all endpoints, request/response schemas)
│   ├── cli.md                        # agl-lite serve, agl-lite controller, agl-client commands
│   ├── schemas.md                    # Pydantic models: Rollout, Event, ModelServer, Resources
│   ├── client-library.md             # AglLiteClient Python API
│   └── gateway-config.md             # Route config YAML format (model_in/out, params add/drop)
├── development/
│   ├── guidelines.md                 # Code conventions, tooling, concurrency model
│   ├── testing.md                    # Test structure, how to run, async conventions
│   └── project-layout.md            # Source tree walkthrough
└── design/
    ├── architecture.md               # Full architecture doc (from 0_architecture.md)
    └── k8s-controller.md             # Controller design details (from 1_k8s_controller.md)
```

### Source material mapping

| Target | Source | Action |
|--------|--------|--------|
| `getting-started/prerequisites.md` | `docs/how-to/install_prerequisites.md` | Clean up, polish |
| `getting-started/minikube.md` | `docs/how-to/install_minikube.md` | Clean up, polish |
| `getting-started/quickstart.md` | `docs/get_started.md` | Rewrite for vLLM math-poc |
| `concepts/*` | `docs/design/0_architecture.md` §1-3, slides | Extract user-facing concepts |
| `user-guide/deployment.md` | `deploy/README.md`, `deploy/*/README.md` | Consolidate |
| `user-guide/configuration.md` | `docs/dev_guidelines.md`, source code | Extract config reference |
| `user-guide/hooks.md` | `dev/todo.md` Store Hooks section | Write from design notes |
| `user-guide/verl-integration.md` | `dev/todo.md` Phase 5, slides | Write from design notes |
| `examples/math-poc.md` | `examples/math-poc/README.md` | Polish, add to mkdocs |
| `examples/swe-bench.md` | `examples/swe_bench/README.md` | Polish, add to mkdocs |
| `reference/api.md` | `docs/design/0_architecture.md` §3.4 | Extract API spec |
| `reference/cli.md` | `agl_lite/cli.py`, `agl_lite/client_cli.py` | Document from source |
| `reference/schemas.md` | `agl_lite/schemas/*.py` | Document from source |
| `reference/client-library.md` | `agl_lite/client.py` | Document from source |
| `reference/gateway-config.md` | `agl_lite/gateway/config.py`, examples | Document from source |
| `development/guidelines.md` | `docs/dev_guidelines.md` | Move as-is |
| `development/testing.md` | `docs/dev_guidelines.md` Testing section | Extract |
| `development/project-layout.md` | `docs/dev_guidelines.md` Project Layout | Extract |
| `design/*` | `docs/design/0_architecture.md`, `1_k8s_controller.md` | Keep as-is |

### Tasks

- [x] **6.0**: Repo README rewrite — architecture SVG, vLLM quick start, agent snippet, examples table
- [x] **6.0.1**: Move coding agent instructions to `AGENTS.md`
- [x] **6.1**: mkdocs setup — `mkdocs.yml`, theme, nav structure, placeholder stubs, build verified [completed]
- [ ] **6.2**: Getting Started section — prerequisites, minikube, quickstart [ready]
- [x] **6.3**: Concepts section — gateway, store, controller, agent contract, data model, weight updates [completed]
- [ ] **6.4**: User Guide section — deployment, configuration, writing agents, running experiments, hooks, VERL [backlog]
- [ ] **6.5**: Examples section — math-poc, swe-bench [backlog]
- [ ] **6.6**: Reference section — API, CLI, schemas, client library, gateway config [backlog]
- [ ] **6.7**: Development section — guidelines, testing, project layout [backlog]
- [ ] **6.8**: Design section — move existing design docs [backlog]
- [ ] **6.9**: Clean up old docs structure (`docs/how-to/`, `docs/refactor_review/`) [backlog]

---

## job_builder: Jinja2 template + PodPatcher [completed]

Refactor `agl_lite/controller/job_builder.py` to move the hardcoded Job manifest
structure and controller env vars into a Jinja2 template, with a `PodPatcher`
schema for per-container env/volume injection.

**Design:**
- `deploy/controller/job-template.yaml.j2`: two YAML documents separated by `---`
  - Doc 0: Job manifest scaffold (apiVersion, kind, metadata, spec shell with
    `restartPolicy: Never`, empty `containers: []`, empty `volumes: []`)
  - Doc 1: `PodPatcher` — controller env vars (with Jinja2 variables) + default volumes
- `PodPatcher(BaseModel)` in `job_builder.py`: validates doc 1 (`env`, `volumes`)
- `build_job_spec` renders the template, parses both docs, then:
  1. Merges user's `job_template` pod spec fragment into the scaffold
     (containers replace `[]`, volumes merged by name, pod-level fields deep-merged)
  2. Injects `patcher.env` into ALL containers (user wins on name conflict)
  3. Injects `patcher.volumes` into pod spec (user wins on name conflict)
  4. Applies `rollout.config` named fields to "agent" container
  5. Applies `rollout.config.overrides`
- `ControllerConfig` in `deploy.py`: add `job_manifest_template: str | None`
- `ControllerSettings` in `controller/config.py`: add `job_manifest_template: str | None`
- Template loaded once at controller startup (or lazily), path resolved relative
  to config file or falling back to packaged default
- Update `controller/reconciler.py` to load/pass template to `build_job_spec`
- Update tests in `tests/controller/test_job_builder.py`

**Files changed:**
- `deploy/controller/job-template.yaml.j2` (new)
- `agl_lite/controller/job_builder.py`
- `agl_lite/controller/config.py`
- `agl_lite/controller/reconciler.py`
- `agl_lite/deploy.py`
- `tests/controller/test_job_builder.py`

---

## Simplify job construction: RolloutConfig.pod_spec + hook on_startup [ready]

Refactor the job construction flow to remove the three-layer merge and make
the hook the single point of pod spec assembly.

**Design decisions agreed:**
- `RolloutConfig` shrinks to `{pod_spec, timeout, max_retries}` — remove `image`,
  `command`, `environment_variables`, `mount`, `overrides`; hook writes directly
  into `pod_spec` per-sample
- Hook loads pod spec template once via `on_startup(self, store)`, stores as
  `self._pod_spec`; deep-copies and modifies per request in `on_enqueue`
- Hook-specific config (e.g. template path) communicated via env vars — hook
  author's responsibility to document
- `RolloutHooks` base class gains: `on_startup(store)` lifecycle method,
  `copy_pod_spec()` helper, `get_container(pod_spec, name)` static helper
- `build_job_spec(rollout, settings, manifest_template)` — `user_pod_spec` param
  removed; controller merges `rollout.config.pod_spec` instead
- `_get_job_template`, `_resources_cache` removed from reconciler — controller
  no longer fetches resources

**Files to change:**
- `agl_lite/schemas/rollout.py` — simplify `RolloutConfig`
- `agl_lite/hooks.py` — add `on_startup`, `copy_pod_spec`, `get_container`
- `agl_lite/controller/job_builder.py` — remove `user_pod_spec` param
- `agl_lite/controller/reconciler.py` — remove resources fetch + cache
- `agl_lite/server/` — call `hook.on_startup(store)` in FastAPI lifespan
- `examples/swe_bench/hooks.py` — rewrite using new pattern
- `tests/` — update all affected tests

---

## Convert deploy config from YAML to .env format [ready]

Replace `agl-lite deploy --config agl-lite.yaml` with
`agl-lite deploy --env-file agl-lite.env` for consistency with the
env-var-based config pattern used everywhere else in the system.

**Design decisions agreed:**
- `.env` file is the single project config for a deployment — `DeploySettings`
  only reads its declared fields, extra vars (hook config, model endpoints,
  experiment params) are ignored by pydantic-settings and consumed by other
  components via `os.environ`; `source deploy/agl-lite.env` in shell populates
  the environment for all components at once
- Explicit `--env-file` flag (standard name, used by docker-compose/podman) —
  keeps safety of `--config`, makes format self-evident, no ambient env reading
- `DeployConfig(BaseModel)` —> `DeploySettings(BaseSettings)` with
  `env_prefix="AGL_"`; pydantic-settings loads the file via `_env_file` kwarg
- `ServerRuntimeConfig` wrapper class removed — `gateway_config`, `hooks`,
  `artifact_dir` promote to top-level `AGL_GATEWAY_CONFIG`, `AGL_HOOKS`,
  `AGL_ARTIFACT_DIR`
- `deploy/agl-lite.yaml.example` —> `deploy/agl-lite.env.example`
- `docs/deploy.md` updated accordingly

**Files to change:**
- `agl_lite/deploy.py` — `DeployConfig` → `DeploySettings(BaseSettings)`,
  flatten `ServerRuntimeConfig`, rename CLI arg to `--env-file`
- `agl_lite/cli.py` — update deploy entrypoint arg
- `deploy/agl-lite.yaml.example` — replace with `deploy/agl-lite.env.example`
- `deploy/README.md`, `docs/deploy.md` — update docs

---

## Phase 7: Polish [backlog]

- [ ] Structured logging (JSON, with rollout_id/attempt_id context)
- [ ] Prometheus metrics (optional)
- [ ] Docker images for agl-lite serve and controller
- [ ] CI/CD pipeline

---

## Pre-Implementation Decisions (Frozen)

These are settled and should not be revisited during implementation:

| Decision | Resolution |
|----------|-----------|
| Package layout | One package, two entrypoints |
| Controller-to-service communication | HTTP only, no shared memory |
| Store backend (MVP) | In-memory, single instance |
| Event ordering | Append order in per-rollout list, no sequence counter |
| `event_id` | Removed — events identified by position in list |
| `timestamp` | Assigned by store at write time |
| `job_template` | Raw K8s pod spec dict, loaded from YAML file. No typed schema — any valid K8s field works. `overrides` on `RolloutConfig` for per-rollout overrides. |
| Auth | Single `AGL_KEY` for all components, no role-based access for MVP. `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` trick for agents. |
| Health endpoint | `GET /healthz`, no auth |
| Error codes | 401 missing/invalid key, 404 rollout not found, 409 invalid transition |
| Archive format | JSONL, user-specified file path (`*.jsonl`). Append if file exists, create if not. Includes rollout + events + resources per archive call. |
| Gateway config | Static YAML at startup. List-based routes: `[{model_in, model_out, params}]`, first match wins. Wildcard support. |
| Model routing | Per-model round-robin. `(model, endpoint)` composite key. Version per server. Optional `token`. |
| Namespace | Single namespace per controller instance. Manifests omit namespace, applied via `-n`. |
| `timeout` / `max_retries` | Map to K8s Job `activeDeadlineSeconds` / `backoffLimit` |
| Agent auth injection | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` env vars via `secretKeyRef` to `agl-lite-keys/AGL_KEY`. |
| vLLM deployment | Docker container (`vllm/vllm-openai:latest`) on host, separate from agl-lite lifecycle. `scripts/start_vllm.sh` for convenience. |
| vLLM topology | agl-lite colocated with algorithm + vLLM on host. Controller + agents in minikube. Gateway → vLLM is localhost. Agents reach host via `host.minikube.internal`. |
| Gateway param injection | `gateway-config.yaml` with `params.add` injects RL-specific params (e.g., `return_token_ids: true`). Event captures prepared body (post-injection). |
| Event request body | Captures the prepared body (after gateway param injection), not the original agent request. RL algorithm sees exactly what was sent to the model server. |
