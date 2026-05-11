# K8s Controller

The controller bridges the [data store](store.md) and Kubernetes. It is the **only component that writes rollout status transitions** (aside from initial enqueue and cancel flag).

## What it does

1. **Creates Jobs** for `queuing` rollouts
2. **Watches Jobs** for status changes and syncs to the store
3. **Handles cancellation**: reads `cancel_requested` flag → deletes Job → marks `cancelled`
4. **Crash recovery**: periodic full reconciliation catches anything missed

## Attempt = Pod UID

Every K8s pod has a unique `metadata.uid`. agl-lite uses this as the attempt ID — zero allocation, zero coordination.

```
Pod #1 (uid=aaa):  rollout=R1, attempt=aaa
  → [req1, req2, req3] → pod crashes

Pod #2 (uid=bbb):  rollout=R1, attempt=bbb
  → [req1', req2', req3', req4'] → succeeds
```

Each pod writes to its own `(rollout_id, attempt_id)` partition in the store. Data never collides — even during node partitions where two pods briefly run for the same rollout.

**Why this simplifies everything:**

- No attempt status machine (K8s owns pod lifecycle)
- No attempt health checks or watchdog
- No custom ID generation
- Retry is just K8s creating a new pod (new UID → new attempt partition)

## Job construction

The controller builds each Job by merging two layers:

```
job_template (raw K8s pod spec, from resources snapshot)
  │
  ├── Inject into container named "agent":
  │     ├── OPENAI_BASE_URL  (gateway path with rollout + pod UID)
  │     ├── OPENAI_API_KEY   (from K8s Secret)
  │     ├── ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY (same gateway)
  │     ├── AGL_TASK_INPUT   (task payload as JSON)
  │     ├── AGL_EVENT_URL    (for posting custom events)
  │     ├── image, command, extra env vars (from rollout.config)
  │     └── volume mounts (from rollout.config)
  │
  └── Wrap in Job metadata:
        ├── name: agl-rollout-{rollout_id}  (deterministic)
        ├── backoffLimit (from config.max_retries)
        └── activeDeadlineSeconds (from config.timeout)
```

The `job_template` is a **raw K8s pod spec** — any valid K8s fields work (nodeSelector, tolerations, volumes, init containers). The store doesn't validate it; K8s does at Job creation.

### Simple example

```yaml
# job-template.yaml — math PoC
containers:
  - name: agent
    image: math-agent:dev
    command: ["python", "/app/qa_agent.py"]
    imagePullPolicy: Never
    resources:
      requests:
        cpu: "100m"
        memory: "128Mi"
```

### Multi-container example

```yaml
# job-template.yaml — coding tasks with scorer sidecar
containers:
  - name: agent
    imagePullPolicy: Never
    resources:
      requests: {cpu: "1", memory: "2Gi"}
    volumeMounts:
      - name: workspace
        mountPath: /workspace
  - name: scorer
    image: scorer:latest
    command: ["python", "run_tests.py"]
    volumeMounts:
      - name: workspace
        mountPath: /workspace
volumes:
  - name: workspace
    emptyDir: {}
```

## Reconciliation loop

The controller uses the standard K8s controller pattern — watch + periodic reconciliation:

```
1. WATCH: K8s Job events (label: agl-lite/rollout-id)
   → on any Job status change, reconcile that rollout

2. POLL Store: query rollouts in "queuing" or with cancel_requested
  → create Jobs for new queuing rollouts, subject to Pod creation rate limits
   → process cancellations

3. PERIODIC FULL RECONCILE (every N seconds):
   → for each non-terminal rollout: check Job status, sync if needed
   → catches anything missed by watch/poll (crash recovery)
```

### Per-rollout reconciliation

```python
# Simplified reconcile logic
def reconcile(rollout):
    if rollout.status is terminal:
        cleanup_job_if_exists(rollout.job_name)
        return

    if rollout.cancel_requested:
        handle_cancel(rollout)       # delete Job → cancelled
        return

    if rollout.status == QUEUING:
        create_job(rollout)          # → running (or terminal_failed)

    elif rollout.status == RUNNING:
        sync_job_status(rollout)     # → succeeded / terminal_failed / no-op
```

### Pod creation rate limit

The controller keeps an in-memory sliding window of successful Job creation timestamps. By default, it creates at most 100 agent Jobs every 10 seconds (`AGL_MAX_PODS_PER_WINDOW=100`, `AGL_RATE_LIMIT_WINDOW_SECONDS=10`). A timestamp is recorded after the Kubernetes `create_job` call succeeds. Failed create attempts do not consume capacity, and rollouts that hit the limit remain `queuing` until a later reconcile cycle.

This limiter is per controller process, matching the default single-replica controller deployment.

## Edge cases

The controller handles these scenarios gracefully:

| Scenario | Behavior |
|----------|----------|
| **Controller crash** | On restart, full reconcile scans all non-terminal rollouts and syncs with K8s |
| **Job already exists** (crash during creation) | Deterministic name `agl-rollout-{rid}` → K8s returns `AlreadyExists` → controller proceeds |
| **Job deleted externally** | Running rollout whose Job is gone → `terminal_failed` with "Job not found" |
| **Cancel + success race** | Job already completed before cancel is processed → **success wins** (data is captured) |
| **Cancel + failure race** | Job already failed → marked `cancelled` (user's intent was cancellation) |
| **Cancel during termination** | Controller waits for Job to fully delete before marking `cancelled` (prevents stale event writes) |
| **Store unavailable** | K8s Jobs keep running. Controller retries on next reconcile cycle |

## Deployment

The controller runs as a single-replica K8s Deployment (no leader election needed for MVP). It needs:

- **Network access to agl-lite** — to query rollouts and update status
- **K8s API access** — via a ServiceAccount with permissions for Jobs and Pods
- **API key** — same `AGL_KEY` as other components (from K8s Secret)
