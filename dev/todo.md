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

## Gateway streaming assembly: multi-format support [discuss]

### Problem

The gateway proxy captures streaming responses as events. Currently `_forward_streaming` in `proxy.py` does:
1. `_parse_sse_chunks(raw)` — format-agnostic SSE→JSON extraction
2. `_assemble_chat_completion(chunks)` — OpenAI chat-completion-specific assembly (`choices[i].delta.content` → `choices[i].message.content`)
3. Falls back to `{"chunks": [...]}` for unknown paths

This is path-dependent (`path.endswith("chat/completions")`) and only supports one format. But the gateway already documents support for multiple LLM API formats (architecture doc §LLM proxy paths), and real agents use different SDKs:

| Endpoint path | Provider | Streaming delta location | Assembled shape |
|---|---|---|---|
| `chat/completions` | OpenAI / vLLM | `choices[i].delta.content` | `ChatCompletion` dict |
| `completions` | OpenAI / vLLM (legacy) | `choices[i].text` (string) | `Completion` dict |
| `messages` | Anthropic / Claude | event-typed SSE: `content_block_delta → delta.text` | `Message` dict |
| `responses` | OpenAI Responses API | multi-event-type SSE | `Response` dict (complex) |

Key differences:
- **OpenAI chat/completions**: `data: {"choices":[{"delta":{"content":"..."}}]}` — each line is `data: <json>`
- **OpenAI completions**: `data: {"choices":[{"text":"..."}]}` — simpler, no delta wrapper
- **Anthropic messages**: uses SSE `event:` lines (`message_start`, `content_block_delta`, `message_delta`, `message_stop`) with typed data payloads — NOT just `data: <json>` lines. `_parse_sse_chunks` ignores the `event:` line and only grabs `data:` payloads, but the assembly logic needs to dispatch on event type.
- **OpenAI responses**: similar multi-event-type structure, not yet widely adopted by vLLM

### Design goals

1. **Uniform stored shape per format** — streaming and non-streaming events for the same endpoint should produce the same response shape (already achieved for chat/completions)
2. **No false assembly** — unknown paths fall back to raw chunks, never silently corrupt data
3. **SSE parsing stays generic** — `_parse_sse_chunks` remains format-agnostic (extract `data:` lines as JSON)
4. **Extensible without combinatorial explosion** — adding a new format should be one assembler function + one path match, not cross-cutting changes

### Proposed solution: assembler registry

A dict mapping path suffixes to assembler functions. Each assembler has the same signature:

```python
# Type alias
Assembler = Callable[[list[dict[str, Any]]], dict[str, Any]]

# Registry — evaluated in order, first suffix match wins
_ASSEMBLERS: list[tuple[str, Assembler]] = [
    ("chat/completions", _assemble_chat_completion),
    ("completions",      _assemble_completion),
    ("messages",         _assemble_anthropic_message),
    # Future: ("responses", _assemble_responses),
]

def _select_assembler(path: str) -> Assembler | None:
    """Return the assembler for the given path, or None for raw fallback."""
    normalized = path.rstrip("/")
    for suffix, assembler in _ASSEMBLERS:
        if normalized.endswith(suffix):
            return assembler
    return None
```

In `_forward_streaming`:
```python
chunks = _parse_sse_chunks(raw)
assembler = _select_assembler(path)
response_body = assembler(chunks) if assembler else {"chunks": chunks}
```

### Assembler functions (scope: 3 formats)

**1. `_assemble_chat_completion`** — already exists, handles `choices[i].delta.content`

**2. `_assemble_completion`** — new, for legacy `/v1/completions`:
```python
def _assemble_completion(chunks: list[dict]) -> dict:
    # choices[i].text (string, no delta wrapper)
    # Assembled shape: {id, object: "text_completion", choices: [{text: "full text", finish_reason}], usage}
```

**3. `_assemble_anthropic_message`** — new, for `/v1/messages`:
```python
def _assemble_anthropic_message(chunks: list[dict]) -> dict:
    # SSE events: message_start (has message shell), content_block_delta (has delta.text),
    #             message_delta (has stop_reason + usage), message_stop
    # Note: _parse_sse_chunks only captures the `data:` JSON, not the `event:` type line.
    #       The event type is detectable from the data payload structure:
    #       - has "type": "message_start" / "content_block_delta" / etc.
    # Assembled shape mirrors Anthropic non-streaming Message:
    #   {id, type: "message", role: "assistant", content: [{type: "text", text: "..."}],
    #    stop_reason, usage: {input_tokens, output_tokens}}
```

### SSE parsing for Anthropic

`_parse_sse_chunks` currently works for Anthropic because it extracts all `data: <json>` lines. Anthropic SSE looks like:
```
event: message_start
data: {"type": "message_start", "message": {...}}

event: content_block_delta  
data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}
```

The `event:` lines are ignored (not `data:` prefix), and the `data:` lines are valid JSON with a `type` field that identifies the event. So `_parse_sse_chunks` works as-is — the assembler dispatches on `chunk["type"]`.

### Non-streaming response: also needs format awareness

Currently `_forward_non_streaming` stores `resp.json()` as-is. This is fine — non-streaming responses are already in their final shape regardless of format. No changes needed there.

### Ordering in `_ASSEMBLERS`

`chat/completions` must come before `completions` (since `completions` is a suffix of `chat/completions`). The list order guarantees first-match semantics.

### Scope

- Phase 1: `chat/completions` (done), `completions`, `messages` — covers vLLM + Anthropic
- Phase 2 (future): `responses` (OpenAI Responses API) when vLLM or agents adopt it

### Files to change

- `agl_lite/gateway/proxy.py` — add `_assemble_completion`, `_assemble_anthropic_message`, `_ASSEMBLERS` registry, `_select_assembler`; update `_forward_streaming`
- `tests/gateway/test_proxy.py` — add tests for legacy completions streaming assembly, Anthropic messages streaming assembly, unknown path fallback

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

- [ ] Migrate `examples/calc_x/` from Agent Lightning as the primary training example
  - Adapt `train_calc_agent.py` to agl-lite VERL entrypoint (`run_ppo`) and agl-lite auth/url flow
  - Replace `agentlightning` runtime dependencies in `calc_agent.py` / `eval_utils.py`
  - Provide a self-contained run script
- [ ] Weight update protocol: after PPO step, update vLLM model weights
- [ ] Multi-iteration training with measurable reward improvement
- [ ] Pre-flight checks: healthz, auth, model registration, rollout completion, triplet extraction, non-empty PPO batch

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

---

## Settings refactor: BaseModel + CLI owns env mapping [completed]

(Done — see Completed section above.)

---

## Logging persistence [backlog]

Goal: logs survive pod deletion and are easy to find without a log aggregation stack.

### 7a: Structured logging for server [completed]

(Done — see Completed section above.)

### 7b: Per-pod log volume for agents [completed]

(Done — see Completed section above.)

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
