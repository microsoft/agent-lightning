# 008 — 🟡 Data Retention and Eviction

## Problem

No mention of data lifecycle. The original Agent Lightning had `eviction_threshold_bytes` and `safe_threshold_bytes` for memory management. For long training runs:

- Thousands of rollouts × multiple attempts × dozens of events each
- `model_request` events contain full request + response payloads (potentially large with long contexts)
- Events table grows unbounded

## Scale estimate

- 1000 rollouts/iteration × 10 events/rollout × 5KB/event = 50MB/iteration
- 100 iterations = 5GB of event data
- With 128K context windows, individual events can be 500KB+ → 500GB+

## Options

**A. TTL-based** — events older than N hours auto-deleted. Simple. Risk: algorithm hasn't consumed them yet.

**B. Iteration-based** — Algorithm calls `DELETE /api/events?before_iteration=N` after consuming. Explicit, safe.

**C. Reference counting** — events marked "consumed" by algorithm, GC'd after. Complex.

**D. External storage** — events written to object storage (S3) after consumption, purged from DB. Production-grade but complex.

**E. Configurable at startup** — `max_event_storage_bytes` with LRU eviction on old rollouts.

## Recommendation

**B (explicit purge) for MVP.** Add `DELETE /api/events` with rollout-based filtering. The Algorithm knows when it's done with a batch. Keep it simple — no automatic eviction. Document the growth rate so users can plan.

Add a note about payload size considerations (truncation options for very large contexts).

## Changes needed

- Add `DELETE /api/events` endpoint to Section 3.4
- Add a "Data lifecycle" note in Section 3.3 or 3.4
- Document storage growth estimates
