# 006 — 🟡 Concurrent LLM Calls and Adapter Design

## Problem

Two related issues:

### 1. Concurrent LLM calls

Tool-use agents commonly fan out parallel LLM calls (e.g., multiple tool invocations simultaneously). The `sequence` counter imposes a total order on what is inherently concurrent. The assigned order is arbitrary (depends on which request completes first).

This is **fine for storage** — we need some ordering for pagination and replay. But consumers must understand that `sequence` is not necessarily causal order.

### 2. Adapter is over-simplified

The example `TrajectoryAdapter` sums all rewards and assigns the same total to every `model_request` triplet. This is a specific RL strategy (episode-level reward). Other strategies exist:
- Per-step rewards (each model call gets its own reward)
- Discounted rewards (later calls get less credit)
- Advantage-based (relative to baseline)
- Token-level rewards (sub-response granularity)

Presenting this as THE adapter, rather than AN example, could mislead users.

## Recommendation

1. Add a note that `sequence` is total ordering for storage, not causal ordering. Concurrent events may have adjacent sequence numbers in arbitrary order. Timestamp provides approximate causal information.

2. Rename `TrajectoryAdapter` section to explicitly say "Example Adapter" and note that users should implement their own for their RL algorithm's reward assignment strategy.

## Changes needed

- Add concurrency note to Section 3.3 (Event-based Trajectory)
- Relabel Section 3.6 as example adapter
