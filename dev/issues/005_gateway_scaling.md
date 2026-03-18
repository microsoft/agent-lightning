# 005 — 🟡 Gateway Scaling and Sequence Counters

## Problem

All LLM traffic flows through one service instance. At scale (hundreds of concurrent agents), this is a throughput bottleneck and single point of failure. The doc's main argument for the unified service is "in-process event capture on hot path" — but this only works with a single instance.

## Tension

- **Single instance**: in-process sequence counters are trivial (atomic int per `(rid, aid)`), event writes are fast (no network hop). But limited throughput.
- **Multiple instances**: need distributed sequence counter (DB-backed), lose the in-process advantage.

## Analysis

For agl-lite MVP, single instance is likely fine:
- Even a modest server can proxy thousands of concurrent LLM requests
- LLM inference is the bottleneck (seconds per request), not the proxy (milliseconds of overhead)
- A single FastAPI/aiohttp process with async I/O can handle high concurrency

## Recommendation

**State position explicitly: single instance for v1.** Document the scaling path for later:
1. Sequence counters move to DB (atomic increment, e.g., `RETURNING` in Postgres)
2. Multiple stateless gateway instances behind a load balancer
3. Events written via async batch insert (buffered in-process, flushed periodically)

This is not blocking for MVP but should be acknowledged in the architecture doc.

## Changes needed

- Add a "Scaling considerations" note in Section 3.4
