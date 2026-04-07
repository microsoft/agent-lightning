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

- [x] **Phase 0–4b**: Schemas, store, HTTP API, gateway, K8s controller, E2E (mock + vLLM)
- [x] **Store Hooks**: `RolloutHooks` base class — `on_startup`, `on_enqueue`, `on_succeeded`, `on_failed`; base auto-loads `AGL_POD_SPEC_TEMPLATE`; `copy_pod_spec()` / `get_container()` helpers
- [x] **Job builder**: Jinja2 template + `PodPatcher`; `RolloutConfig` simplified to `{pod_spec, timeout, max_retries}`
- [x] **Deploy config**: YAML → `.env` format; `DeploySettings(BaseSettings)` with `AGL_*` prefix
- [x] **Phase 5a–5b**: Triplet API + `AglLiteDaemon` (VERL bridge)
- [x] **Config hygiene**: no defaults in CLI or settings for pod-side values; `AGL_SECRET_NAME` removed (hardcoded `agl-lite-keys`); `AGL_K8S_NAMESPACE` → `AGL_NAMESPACE`; `lite_url` → `base_url`
- [x] **Math-poc**: updated hooks, deploy.env, run.sh, rl_loop.py for new API
- [x] **Settings refactor**: `ServerSettings` + `ControllerSettings` → plain `BaseModel`; `cli.py` is the sole env boundary via `typer.Option(default, envvar="AGL_*")`; `agl_key` → `key`
- [x] **Logging (7a)**: structured logging via structlog; dual output (ConsoleRenderer stdout + JSONRenderer file); `AGL_LOG_DIR` / `AGL_LOG_LEVEL`; default archive path; `deploy.py` stdout redirect removed
- [x] **Logging (7b)**: per-pod hostPath volume via PodPatcher; `AGL_LOG_DIR=/agl/logs/$(AGL_ATTEMPT_ID)`; `volume_mounts` field added to `PodPatcher`; ordering invariant enforced + tested

---

## Anthropic `/v1/messages` + vLLM token IDs [discuss]

vLLM's `/v1/messages` (Anthropic-compatible) interface does not appear to support `return_token_ids` yet — unlike `/v1/chat/completions` which returns `prompt_token_ids` and per-choice `token_ids` when enabled. If we want to train with vLLM through the Anthropic interface, we need token IDs for triplet extraction.

Questions to resolve:
- Does vLLM plan to add `return_token_ids` support to the `/v1/messages` endpoint?
- If not, should agents using the Anthropic SDK fall back to `/v1/chat/completions` for training runs?
- Should `assemble_anthropic_message` include placeholder logic for token IDs in anticipation, or stay clean until vLLM adds support?

---

- [ ] **Cancel test**: enqueue → cancel mid-run → verify cancelled status
- [ ] **Retry test**: agent with `CRASH_ON_FIRST=1` → K8s Job retries → succeeds on second attempt
- [ ] **503 test**: agents hitting gateway during model deregistration window

---

## Phase 4b.3: Performance baseline [backlog]

- [ ] Measure: rollout throughput, gateway proxy latency overhead, event capture overhead
- [ ] Compare direct vLLM vs gateway-proxied vLLM

---

## SWE-bench follow-ups [backlog]

- [ ] Remove `resources_id` field from rollout schemas and archive JSONL output (dead field — controller no longer fetches resources)

---

## Phase 5c: Full training loop E2E [ongoing]

Migrate `examples/calc_x/` from Agent Lightning as the primary VERL training example.
Mode: `agl-in-host` (agl-lite serve + vLLM on host, controller + agent pods in minikube).

### Architecture

```
run.sh                          ← E2E entrypoint
  ├── verify vLLM reachable
  ├── minikube image build (agent Docker image)
  ├── agl-lite deploy --env-file vllm/.env.example  (agl-in-host)
  │     ├── K8s: namespace, secret, configmap
  │     ├── K8s: controller deployment
  │     └── Host: agl-lite serve (with hooks)
  ├── wait for healthz
  └── exec python train_calc_agent.py "$@"

train_calc_agent.py             ← VERL training (assumes infra is up)
  ├── load Calc-X parquet dataset
  ├── build VERL config dict
  │     └── agentlightning.agl_base_url / agl_key from env
  └── run_ppo(config, train_dataset, val_dataset)
        └── Ray → AgentLightningTrainer.fit()
              └── AglLiteDaemon (HTTP → agl-lite server)
                    ├── register model servers
                    ├── enqueue rollouts → controller creates K8s Jobs
                    │     └── agent pod: AutoGen + MCP calculator → gateway → vLLM
                    ├── poll until all complete
                    ├── fetch triplets (format=triplet, token IDs from gateway)
                    └── build padded tensors → PPO update
```

For iterative development, run `train_calc_agent.py` directly (infra already up).

### Implementation plan

Each sub-item is one commit. Implement in order.

#### 5c.1: Dataset and eval utils [completed]

- [ ] `data/` — parquet files already downloaded; keep download instructions in README
      (Google Drive link, manual download). Add `data/` to `.gitignore`.
- [ ] `data/sample.jsonl` — 5-10 rows extracted from train.parquet for smoke testing
- [ ] `eval_utils.py` — remove `from agentlightning.reward import reward` decorator;
      keep `scalar_are_results_same` and `evaluate` as pure functions;
      `evaluate` becomes sync (drop `async`, not needed in hooks)

#### 5c.2: Agent container [completed]

- [ ] `agents/calc_agent.py` — standalone container agent (no agl-lite imports):
      reads `AGL_TASK_INPUT` (JSON: `{question, id}`), starts MCP calculator
      via `uvx mcp-server-calculator`, runs AutoGen `AssistantAgent` with
      `reflect_on_tool_use=True`, extracts answer via `### ANSWER: <answer> ###`
      regex, posts `agent_output` event to `AGL_EVENT_URL` with `{answer, raw_response}`.
      Uses `OPENAI_BASE_URL` / `OPENAI_API_KEY` (injected by controller).
      Timeout: 5 min per problem.
- [ ] `Dockerfile.agent` — `python:3.12-slim` + `pip install openai autogen-agentchat
      autogen-ext[openai] mcp` + copy agent script. Needs `uvx` for MCP server.
- [ ] `job-template.yaml` — pod spec with `agent` container, `imagePullPolicy: Never`
      (minikube), resource requests (CPU-only, MCP tools don't need GPU)

#### 5c.3: Hooks and config [completed]

- [ ] `vllm/hooks.py` — `CalcXHooks(RolloutHooks)`:
      `on_enqueue`: inject `AGL_TASK_INPUT` (question + id) and `AGL_MODEL_NAME` into pod env.
      `on_succeeded`: extract answer from `agent_output` event, compare with ground truth
      using `eval_utils.scalar_are_results_same`, post `reward` event.
- [ ] `vllm/gateway-config.yaml` — same as math-poc: `return_token_ids: true` for all models
- [ ] `vllm/.env.example` — deploy config: `AGL_NAMESPACE`, `AGL_MODE=agl-in-host`,
      `AGL_GATEWAY_CONFIG`, `AGL_HOOKS`, `AGL_POD_SPEC_TEMPLATE`,
      `AGL_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct`, `AGL_MODEL_ENDPOINT=http://localhost:8010/v1`,
      vLLM params, VERL params

#### 5c.4: Training script [completed]

- [ ] `train_calc_agent.py` — rewrite:
      - Load parquet dataset (train + val) via `datasets.Dataset.from_parquet`
      - Build VERL config dict (reuse `verl_default_config()` structure)
      - Add `agentlightning` section: `agl_base_url` from `AGL_BASE_URL` env,
        `agl_key` from `AGL_KEY` env
      - Call `run_ppo(config, train_dataset, val_dataset)` from
        `agl_lite.verl.entrypoint`
      - CLI args: `--train-file`, `--val-file`, `--model` (optional override),
        `--ci` / `--ci-fast` (for testing)
      - Drop all Agent Lightning deps: `agl.VERL`, `agl.Trainer`, `agl.OtelTracer`,
        `LlmProxyTraceToTriplet`, `LightningStoreClient`, `MongoLightningStore`,
        `WeaveTracer`, `n_runners`, `external_store_address`, `mongo_uri`, `weave`,
        `lora` (can add back later), `trajectory_level`

#### 5c.5: Run script and cleanup [completed]

- [ ] `run.sh` — E2E entrypoint:
      1. Source `vllm/.env.example`
      2. Verify vLLM reachable at `AGL_VLLM_PORT`
      3. `scripts/build_images.sh --include-example calc-x` (build agent image into minikube)
      4. `agl-lite deploy --env-file vllm/.env.example`
      5. Wait for healthz
      6. `exec python train_calc_agent.py "$@"`
- [ ] Delete old files: `calc_agent.py` (old), `tests/` (Agent Lightning tests)
- [ ] Update `README.md` with new usage instructions

#### 5c.6: Smoke test and validation [blocked]

Blocked on VERL 0.7.1 API migration (see 5c.7 below).

Environment issues resolved:
- [x] Ray cluster conflict on shared machine (RAY_GCS_SERVER_PORT=0, RAY_tmpdir)
- [x] Ray worker venv isolation (working_dir=None)
- [x] flash-attn two-phase install (uv pip install --no-build-isolation after uv sync)
- [x] CUDA auto-detect in setup_verl.sh (cu128 fallback)
- [x] verl/vllm version alignment (verl 0.7.1 + vllm 0.12.0)

#### 5c.7: Migrate VERL integration to 0.7.1 AgentLoopManager API [ongoing]

Implementation done. Remaining: hardware-specific VERL config tuning.

Completed:
- [x] `agl_lite/verl/agent_loop.py` — `AglLiteAgentLoopManager(AgentLoopManager)`
- [x] Delete `agl_lite/verl/trainer.py` — use standard `RayPPOTrainer`
- [x] Simplify `agl_lite/verl/entrypoint.py` — use verl's `TaskRunner` pattern
- [x] `config.yaml` — add `model_endpoint`, `timeout_seconds`
- [x] Fix `dataset.py` — `LoadedDataset` for verl 0.7.1 serialization
- [x] Fix Ray isolation — `RAY_GCS_SERVER_PORT=0`, `RAY_tmpdir`, `working_dir=None`
- [x] Fix import paths — `verl.utils.ray_utils.auto_await`

E2E progress:
- [x] Ray init (separate cluster on shared machine)
- [x] FSDP model loaded (Qwen2.5-1.5B-Instruct)
- [x] vLLMHttpServer launched by VERL
- [ ] vLLM engine core startup — fails with `Engine core initialization failed`.
      Likely GPU memory contention: FSDP + vLLM on same GPU in hybrid mode.
      Need to tune `gpu_memory_utilization`, or use more GPUs (`n_gpus_per_node: 2+`).
- [ ] Agent rollout execution via agl-lite
- [ ] PPO update completion

Environment note: requires `python3.12-dev` for triton JIT compilation.

VERL 0.7.1 introduced `AgentLoopManager` — a built-in agent orchestration system
that replaces the pattern of subclassing `RayPPOTrainer`. Our current code
(`AgentLightningTrainer` + custom `_train_step` + custom `fit`) overrides half
the trainer internals, all of which changed in 0.7.1.

##### Architecture change

```
Before (verl 0.6.0 pattern):
  AgentLightningTrainer(RayPPOTrainer)
    ├── __init__(reward_fn, val_reward_fn, ...)     ← removed in 0.7.1
    ├── _train_step() override                      ← calls AglLiteDaemon directly
    ├── _validate() override                        ← calls AglLiteDaemon directly
    └── fit() override                              ← heavily customized loop

After (verl 0.7.1 pattern):
  Standard RayPPOTrainer (no subclass needed)
    └── AgentLoopManager.generate_sequences()
          └── custom AglLiteAgentLoopManager
                └── AglLiteDaemon (HTTP → agl-lite server)
```

The integration point moves from "override the trainer" to "provide a custom
`AgentLoopManager`" — a designed extension point in verl 0.7.1.

##### Specific breaking changes

| Change | Old (0.6.0) | New (0.7.1) |
|--------|-------------|-------------|
| Trainer init | `RayPPOTrainer(reward_fn=..., val_reward_fn=...)` | No reward_fn args |
| Reward flow | `reward_fn()` called in `_train_step` | `rm_scores` populated by `AgentLoopWorker`, extracted via `extract_reward()` |
| Rollout orchestration | Custom `_train_step` calls daemon | `AgentLoopManager.generate_sequences()` returns `DataProto` with `rm_scores` |
| Custom manager config | N/A | `rollout.agent.agent_loop_manager_class` FQN |
| Validation | Custom `_validate` calls daemon | Standard `_validate` uses `generate_sequences` + `extract_reward` |

##### Implementation plan

- [ ] **Create `agl_lite/verl/agent_loop.py`** — `AglLiteAgentLoopManager(AgentLoopManager)`:
      - `generate_sequences(prompts: DataProto) -> DataProto`:
        1. Register model servers (from `self.config` / server addresses)
        2. Enqueue rollouts via `AglLiteDaemon`
        3. Poll until all complete
        4. Fetch triplets, build padded tensors
        5. Populate `rm_scores` from reward events
        6. Return `DataProto` in the format `RayPPOTrainer.fit()` expects
      - Reuse `AglLiteDaemon` internally for HTTP communication + tensor construction
      - Key: `generate_sequences` must return data matching what the standard
        training loop expects (`input_ids`, `attention_mask`, `position_ids`,
        `responses`, `rm_scores`, `response_mask`, etc.)

- [ ] **Delete `agl_lite/verl/trainer.py`** — `AgentLightningTrainer` no longer needed;
      standard `RayPPOTrainer` handles the training loop

- [ ] **Simplify `agl_lite/verl/entrypoint.py`** — use verl's `main_ppo.TaskRunner`
      or minimal wrapper; remove reward_fn/val_reward_fn plumbing, worker setup
      duplication. Key addition: set `agent_loop_manager_class` in config.

- [ ] **Update `train_calc_agent.py`** — add to VERL config:
      ```python
      "actor_rollout_ref": {
          "rollout": {
              "agent": {
                  "agent_loop_manager_class": "agl_lite.verl.agent_loop.AglLiteAgentLoopManager",
              },
          },
      }
      ```

- [ ] **Update `agl_lite/verl/config.yaml`** — add agent loop defaults

- [ ] **Update tests** — `tests/verl/test_daemon.py` may need adjustment
      if daemon interface changes

### Future (separate items)

- [ ] Weight update protocol: after PPO step, update vLLM model weights
- [ ] Multi-iteration training with measurable reward improvement
- [ ] Pre-flight checks: healthz, auth, model registration, rollout completion,
      triplet extraction, non-empty PPO batch

---

## Phase 6: Documentation [ongoing]

- [ ] **6.2**: Getting Started — prerequisites, minikube, quickstart [ready]
- [ ] **6.4**: User Guide — deployment, configuration, writing agents, running experiments, hooks, VERL [backlog]
- [ ] **6.5**: Examples — math-poc, swe-bench [backlog]
- [ ] **6.6**: Reference — API, CLI, schemas, client library, gateway config [backlog]
- [ ] **6.7**: Development — guidelines, testing, project layout [backlog]
- [ ] **6.8**: Design — move existing design docs [backlog]
- [ ] **6.9**: Clean up old docs structure (`docs/how-to/`, `docs/refactor_review/`) [backlog]

---

## Phase 7: Polish [backlog]

- [ ] Prometheus metrics (optional)
- [ ] Docker images for agl-lite serve and controller
- [ ] CI/CD pipeline
- [ ] `build_images.sh`: support build backends beyond minikube (e.g., `docker build` + push to a registry) for real cluster deployments. Currently hardcoded to `minikube image build` which only works for local dev.

---

## Logging persistence [backlog]

Goal: logs survive pod deletion and are easy to find without a log aggregation stack.

---

## Pre-Implementation Decisions (Frozen)

| Decision | Resolution |
|----------|-----------|
| Package layout | One package, two entrypoints |
| Controller-to-service communication | HTTP only, no shared memory |
| Store backend (MVP) | In-memory, single instance |
| Event ordering | Append order in per-rollout list, no sequence counter |
| `event_id` | Removed — events identified by position in list |
| `timestamp` | Assigned by store at write time |
| Auth | Single `AGL_KEY` for all components. `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` trick for agents. |
| Health endpoint | `GET /healthz`, no auth |
| Error codes | 401 missing/invalid key, 404 not found, 409 invalid transition |
| Archive format | JSONL, user-specified path. Append if exists. Includes rollout + events per call. |
| Gateway config | Static YAML at startup. List-based routes, first match wins. Wildcard support. |
| Model routing | Per-model round-robin. `(model, endpoint)` composite key. Version per server. |
| Namespace | Single namespace per controller instance. |
| `timeout` / `max_retries` | Map to K8s Job `activeDeadlineSeconds` / `backoffLimit` |
| Agent secret | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` via `secretKeyRef` → `agl-lite-keys/AGL_KEY` |
| K8s secret name | `agl-lite-keys` — hardcoded in template and deploy.py |
| K8s ConfigMap name | `agl-lite-config` — hardcoded in k8s.yaml and deploy.py |
| Pod spec assembly | Hook owns it entirely via `on_enqueue`; `RolloutConfig.pod_spec` is the transport |
| Job manifest template | `AGL_JOB_MANIFEST_TEMPLATE` — default in `DeploySettings`; required in controller pod |
| User pod spec template | `AGL_POD_SPEC_TEMPLATE` — loaded by base `RolloutHooks.on_startup` |
