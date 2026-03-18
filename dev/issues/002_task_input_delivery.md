# 002 — 🔴 Task Input Delivery to Agent

## Problem

Rollout has `input: Dict` — the task description (e.g., "write a function that sorts a list"). The agent needs this to know *what to do*. But the doc only shows `OPENAI_BASE_URL` and `AGL_EVENT_URL` as env vars injected into the pod. The task delivery mechanism is completely missing.

## Why this is architectural

Without a way to deliver the task, the entire rollout loop is broken. The agent container starts but has no instructions.

## Options

**A. Env var `AGL_TASK_INPUT`** — serialize `input` dict as JSON and inject as env var. Simple. Limited by env var size (~128KB in K8s). Fine for text prompts, not for large payloads.

**B. File mount via ConfigMap/Secret** — controller creates a ConfigMap with the task, mounts as `/task/input.json`. More complex (creates extra K8s resources), but handles larger payloads.

**C. Agent fetches from API** — inject `AGL_TASK_URL` env var pointing to `GET /api/rollouts/{rid}/input`. Agent makes one HTTP call on startup. No size limit. Requires agent to be minimally agl-lite-aware (one HTTP GET).

**D. Baked into container image** — task is part of the image. Only works for fixed-task scenarios (benchmarks), not general RL.

## Recommendation

**A (env var) as primary, with C (API fetch) as documented alternative for large payloads.**

Most RL tasks are small text prompts (well under 128KB). Env var is zero-effort for the agent — it just reads `AGL_TASK_INPUT`. For the rare case of large payloads, document the API fetch pattern.

This keeps agents language-agnostic (every language can read env vars) and requires zero agl-lite awareness for the common case.

## Changes needed

- Add `AGL_TASK_INPUT` env var to Job template in Section 3.3 and 3.5
- Add `GET /api/rollouts/{rid}/input` endpoint (optional, for large payloads)
- Document size limits and the API fetch fallback
