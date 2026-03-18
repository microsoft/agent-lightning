# 009 — 🟢 Event Ingestion Validation

## Problem

`POST /rollout/{rid}/attempt/{aid}/events` and the auto-captured `model_request` events — does the service validate that `rollout_id` exists in the Store?

**If yes**: adds a DB lookup to every LLM call on the hot path. Could add latency.

**If no**: orphan events accumulate from stale pods (e.g., pod running after rollout cancelled, race between Job deletion and pod shutdown).

## Recommendation

**No validation on write (hot path stays fast).** Orphan events are harmless — they're never queried (no rollout references them for training). Add optional periodic garbage collection:

```sql
DELETE FROM events
WHERE rollout_id NOT IN (SELECT rollout_id FROM rollouts)
AND timestamp < NOW() - INTERVAL '1 hour'
```

Or simply rely on the data retention mechanism (issue #008) to clean them up naturally.

## Changes needed

- Add a note in Section 3.4 that event ingestion is fire-and-forget (no rollout validation)
