# Architecture Issues

Issues discovered during critical review of `docs/refactor/0_architecture.md`.

Each issue is a separate file: `{number}_{slug}.md`.
Severity: 🔴 architectural gap, 🟡 important design issue, 🟢 minor/deferrable.

## Index

| # | Severity | Title | Status |
|---|----------|-------|--------|
| 001 | 🔴 | [Streaming LLM response handling](001_streaming_response.md) | open |
| 002 | 🔴 | [Task input delivery to agent](002_task_input_delivery.md) | open |
| 003 | 🔴 | [LLM backend routing and resource mapping](003_llm_backend_routing.md) | open |
| 004 | 🔴 | [Batch trajectory query for training](004_batch_trajectory_query.md) | open |
| 005 | 🟡 | [Gateway scaling and sequence counters](005_gateway_scaling.md) | open |
| 006 | 🟡 | [Concurrent LLM calls and adapter design](006_concurrent_calls_adapter.md) | open |
| 007 | 🟡 | [Rollout config schema](007_rollout_config_schema.md) | open |
| 008 | 🟡 | [Data retention and eviction](008_data_retention.md) | open |
| 009 | 🟢 | [Event ingestion validation](009_event_validation.md) | open |
| 010 | 🟢 | [find_succeeded_pod_uid implementation](010_find_succeeded_pod_uid.md) | open |
| 011 | 🟡 | [Cross-boundary authentication and transport security](011_cross_boundary_security.md) | open |
