# VERL Integration

agl-lite's VERL bridge enqueues rollout Jobs through the agl-lite HTTP API and
builds the `DataProto` batches consumed by PPO/GRPO training.

## Agent Job Cleanup

By default, completed agent Jobs are left for Kubernetes TTL cleanup through the
Job manifest's `ttlSecondsAfterFinished` setting. For training environments that
need faster cleanup between rollout batches, the bridge can explicitly delete
only the agent Jobs created for the current batch:

```yaml
agentlightning:
    cleanup_agent_jobs: true
    cleanup_namespace: agl-lite
```

The cleanup is intentionally narrow:

- It deletes Kubernetes Jobs, not Pods, so Kubernetes handles owned Pod cleanup.
- It lists only Jobs labeled `app.kubernetes.io/managed-by=agl-lite`.
- It deletes only Jobs whose `agl-lite/rollout-id` label matches rollouts tracked
    by the current `AglLiteRolloutBridge` batch.
- `clear_data_and_server()` remains a local in-memory reset and does not touch
    the Kubernetes cluster.

The trainer process must have Kubernetes access that can list and delete Jobs in
`cleanup_namespace`. If that RBAC or kubeconfig is not available, keep
`cleanup_agent_jobs` disabled and rely on Job TTL cleanup.
