# 007 — 🟡 Rollout Config Schema

## Problem

`Rollout.config: Dict` is unspecified. The K8s controller must extract fields from this dict to build the Job spec. Without a schema, the controller can't be implemented.

## Required fields (controller needs these)

```python
class RolloutConfig:
    image: str                          # agent container image
    backoff_limit: int = 3              # K8s Job backoffLimit (retry count)
    timeout_seconds: int = 600          # K8s Job activeDeadlineSeconds
    # Optional
    cpu_request: Optional[str]          # e.g., "500m"
    memory_request: Optional[str]       # e.g., "1Gi"
    cpu_limit: Optional[str]
    memory_limit: Optional[str]
    env: Optional[Dict[str, str]]       # additional env vars for agent
    node_selector: Optional[Dict]       # K8s node selector
```

## Questions

1. Should `image` be per-rollout or per-batch? In most RL setups, all rollouts use the same agent image. Could be a default in the controller config, overridable per-rollout.

2. Should config include K8s-specific fields (nodeSelector, tolerations) or stay abstract? For agl-lite (K8s-native), K8s-specific is fine.

3. Should sensitive env vars (API keys) go here or in K8s Secrets? Secrets should be referenced by name, not stored in the Store.

## Recommendation

Define a `RolloutConfig` schema with required fields (`image`, `backoff_limit`, `timeout_seconds`) and optional K8s-specific fields. Controller config provides defaults; per-rollout config overrides.

## Changes needed

- Define `RolloutConfig` in Section 3.3
- Update Job template in Section 3.5 to show how config maps to Job spec fields
