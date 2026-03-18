# 010 — 🟢 find_succeeded_pod_uid Implementation

## Problem

The controller calls `find_succeeded_pod_uid(job)` to set `succeeded_attempt_id`. But K8s Job `status.conditions` only contain `Complete`/`Failed` — not the pod UID that succeeded.

## Implementation

Need to list pods by Job label and find the succeeded one:

```python
def find_succeeded_pod_uid(job) -> Optional[str]:
    pods = k8s.list_pods(label_selector=f"job-name={job.metadata.name}")
    for pod in pods.items:
        if pod.status.phase == "Succeeded":
            return str(pod.metadata.uid)
    return None
```

## Edge cases

1. **Pod GC'd before query** — K8s may garbage-collect completed pods (controlled by `ttlSecondsAfterFinished` on the Job and pod GC settings). If the pod is gone, we can't find the UID. Mitigation: query pods immediately when Job condition changes (in the watch handler), not in periodic reconciliation.

2. **Multiple succeeded pods** (node partition edge case) — pick any one. Both have valid data partitioned by their own `attempt_id`.

3. **No succeeded pod found** — Job says Complete but no pod with phase=Succeeded. Defensive: set `succeeded_attempt_id = None` and log a warning. The rollout is still marked succeeded but the user must query `list_attempts` to find the right one.

## Recommendation

This is an implementation detail, not an architecture change. Document the pod query approach and the edge cases. The controller should query pods as part of the watch callback (not deferred to periodic reconcile) to minimize the GC race window.

## Changes needed

- Add implementation note to Section 3.5 edge cases or pseudocode
