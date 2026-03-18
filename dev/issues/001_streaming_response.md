# 001 — 🔴 Streaming LLM Response Handling

## Problem

The doc removes `StreamConversionMiddleware` saying "Events capture the final response." But most production LLM usage is **streaming** (SSE). The gateway must handle this — it's the hardest part of the proxy implementation and is completely unaddressed.

## What the gateway must do for streaming

1. Agent sends `POST /v1/chat/completions` with `stream: true`
2. Gateway forwards to LLM backend, receives SSE stream
3. Gateway must **simultaneously**:
   - Stream chunks back to the agent (low latency — agent sees tokens as they arrive)
   - Buffer all chunks to reconstruct the complete response
4. After stream ends, gateway assembles the full response and writes `model_request` event
5. If stream errors mid-way, gateway must still record a partial event (for debugging)

## Why this is architectural

- Affects gateway memory model (buffering per concurrent stream)
- Affects event write timing (event written after stream completes, not at request time)
- Affects error handling (partial streams, client disconnects mid-stream)
- Affects sequence counter semantics (sequence assigned at start or end of stream?)

## Options

**A. Tee the stream** — gateway reads from backend, writes to both client and buffer. Event written on stream completion. Sequence assigned when event is written.

**B. Record request-only, response-async** — write a `model_request_start` event immediately, update with response later. More complex, two writes per call.

**C. Non-streaming internally** — force non-streaming to backend, stream the response to client ourselves. Simpler capture but higher latency and loses real streaming benefits.

## Recommendation

Option A (tee the stream). It's what production proxies do. Sequence assigned at stream completion maintains correct temporal ordering. Partial stream errors produce an event with `error` field in `data`.

## Affected sections

- Section 3.4 (Unified API Spec — LLM proxy paths)
- Section 4.3 (removal of `StreamConversionMiddleware`)
