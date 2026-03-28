# K8s Controller Design

Detailed design for the K8s controller component described in [0_architecture.md](0_architecture.md) Section 3.5.

This document covers implementation-level decisions that are too detailed for the architecture doc but important to record.

---

## 1. Design Considerations

### 1.1 Resolving succeeded pod UID

**Context**: When a Job completes, the controller needs the pod UID of the successful pod to set `succeeded_attempt_id` on the Rollout. K8s Job `status.conditions` only contain `Complete`/`Failed` — not pod UIDs.

**Approach**: List pods by Job label and find the succeeded one:

```python
def find_succeeded_pod_uid(job) -> Optional[str]:
    pods = k8s.list_pods(label_selector=f"job-name={job.metadata.name}")
    for pod in pods.items:
        if pod.status.phase == "Succeeded":
            return str(pod.metadata.uid)
    return None
```

**Pod GC race**: K8s may garbage-collect completed pods before the controller queries them. This is mitigated by `ttlSecondsAfterFinished` on the Job spec (set to 3600s = 1 hour in the Job template). The controller's watch callback fires within seconds of Job completion — well within the TTL window.

**Edge cases**:
- **Multiple succeeded pods** (node partition): pick any one. Both have valid event data partitioned by their own `attempt_id`.
- **No succeeded pod found** (unexpected): set `succeeded_attempt_id = None`, log a warning. Rollout is still marked `succeeded` but the user must call `GET /api/rollouts/{rid}` (which includes `attempts` list) to find the right attempt.

**Timing**: Query pods in the **watch callback** (reactive, immediate), not deferred to periodic reconciliation. This minimizes the window between Job completion and pod query.

### 1.2 Job creation failure handling

**Context**: Job creation can fail for transient (quota, scheduling pressure) or permanent (invalid image, bad spec, RBAC) reasons. Need to handle both without losing rollouts.

**Decision**: Distinguish _creation failure_ from _execution failure_:

| Failure mode | Example | K8s signal | Controller action |
|---|---|---|---|
| **Transient creation failure** | Quota exceeded, admission webhook timeout | API 4xx/5xx on Job create | Stay in `queuing`, retry next reconcile cycle |
| **Permanent creation failure** | Invalid image, bad spec | API 4xx on Job create (repeated) | Stay in `queuing` until max queue time exceeded → `terminal_failed` |
| **Agent/task failure** | Bad code, crashes, wrong logic | Job condition `Failed` (BackoffLimitExceeded) | `terminal_failed` with error_message |
| **Infra timeout** | Pod stuck Pending, slow startup | Job condition `Failed` (DeadlineExceeded) | `terminal_failed` with error_message |

**Key rules**:
- `queuing → running` only on **successful** Job creation (API returns 2xx)
- Job creation failure → log error, leave rollout in `queuing`, retry next cycle
- **Max queue time** (configurable, default 1 hour): if a rollout stays in `queuing` beyond this, move to `terminal_failed` with `error_message: "Exceeded max queue time — Job creation repeatedly failed: <last_error>"`
- No new states needed — `error_message` on the rollout carries the failure distinction
- The algorithm reads `error_message` to decide: re-enqueue with different config vs fix the agent

### 1.3 Resources caching

**Context**: Controller fetches resources by `resources_id` for each rollout. Batches of rollouts often share the same `resources_id`.

**Decision**: Simple persistent `dict[resources_id, ResourcesUpdate]` cache in the controller process. Resources are immutable (write-once, never updated) — no invalidation needed. Only fetches from API on cache miss. Grows slowly (one entry per unique `resources_id`, typically < 100).

## 2. Module Structure

### 2.1 Python client (`agl_lite/client.py`)

Thin typed HTTP client wrapping the agl-lite API. Shared by both controller and algorithm.

```python
class AglLiteClient:
    def __init__(self, base_url: str, agl_key: str | None = None): ...

    # Rollouts
    async def enqueue_rollouts(self, rollouts: list[dict]) -> list[Rollout]: ...
    async def query_rollouts(self, status_in: list[str] | None = None, ...) -> list[Rollout]: ...
    async def get_rollout(self, rollout_id: str) -> Rollout: ...
    async def patch_rollout(self, rollout_id: str, **fields) -> Rollout: ...
    async def cancel_rollout(self, rollout_id: str) -> Rollout: ...
    async def archive_rollouts(self, rollout_ids: list[str], backend: dict | None = None) -> dict: ...

    # Events
    async def get_events(self, rollout_id: str, attempt_id: str | None = None, ...) -> list[Event]: ...

    # Models
    async def register_models(self, models: list[dict]) -> list[ModelServer]: ...
    async def list_models(self) -> list[ModelServer]: ...
    async def delete_models(self, model: str | None = None, endpoints: list[str] | None = None) -> None: ...

    # Resources
    async def add_resources(self, resources: dict) -> ResourcesUpdate: ...
    async def get_resources(self, resources_id: str) -> ResourcesUpdate: ...
    async def get_latest_resources(self) -> ResourcesUpdate | None: ...
```

Uses `httpx.AsyncClient` internally with connection pooling. Shares Pydantic schema types for request/response — no duplication. Controller mocks this in tests instead of httpx.

### 2.2 Controller modules

```
agl_lite/controller/
├── __init__.py
├── reconciler.py    # Main reconcile loop (watch + periodic)
├── job_builder.py   # Pure function: (rollout, resources, settings) → K8s Job dict
└── config.py        # Controller settings (namespace, poll interval, max_queue_time)
```

- **`job_builder.py`**: Pure data transformation, no I/O. Takes rollout config + job_defaults + controller settings → K8s Job manifest dict. Easy to unit test.
- **`reconciler.py`**: Async loop using kr8s. Uses `AglLiteClient` for store access, kr8s for K8s API. Two tasks: `watch_jobs()` (reactive) + `periodic_reconcile()` (crash recovery).
- **`config.py`**: `ControllerSettings` with namespace, poll interval, max queue time, agl-lite URL.
