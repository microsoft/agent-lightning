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
