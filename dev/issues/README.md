# Architecture Issues

Issues discovered during critical review of `docs/refactor/0_architecture.md`.

Each issue is a separate file: `{number}_{slug}.md`.
Severity: 🔴 architectural gap, 🟡 important design issue, 🟢 minor/deferrable.

## Index

| # | Severity | Title | Status |
|---|----------|-------|--------|
| 001 | 🔴 | Streaming LLM response handling | **resolved** → `docs/refactor/0_architecture.md` §3.4 LLM proxy paths |
| 002 | 🔴 | [Task input delivery to agent](002_task_input_delivery.md) | open |
| 003 | 🔴 | LLM backend routing and resource mapping | **resolved** → `docs/refactor/0_architecture.md` §3.4 Model Server Management |
| 004 | 🔴 | Batch trajectory query for training | **resolved** → `docs/refactor/0_architecture.md` §3.4 Bulk data transfer. No batch endpoint needed; concurrent queries + archive-to-shared-storage for remote case. |
| 005 | 🟡 | Gateway scaling and sequence counters | **resolved** → `docs/refactor/0_architecture.md` §3.4 Concurrency and scaling |
| 006 | 🟡 | Concurrent LLM calls and adapter design | **resolved** → concurrency note in §3.3, adapter relabeled as example in §3.6 |
| 007 | 🟡 | [Rollout config schema](007_rollout_config_schema.md) | open |
| 008 | 🟡 | Data retention and eviction | **resolved** → `docs/refactor/0_architecture.md` §3.4 Data lifecycle |
| 009 | 🟢 | [Event ingestion validation](009_event_validation.md) | open |
| 010 | 🟢 | [find_succeeded_pod_uid implementation](010_find_succeeded_pod_uid.md) | **resolved** → `docs/refactor/1_k8s_controller.md` |
| 011 | 🟡 | [Cross-boundary authentication and transport security](011_cross_boundary_security.md) | open |
