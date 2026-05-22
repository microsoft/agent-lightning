"""Event schemas — the universal unit of data in agl-lite."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Event(BaseModel):
    """Single event in a trajectory.

    Events are stored in insertion order per rollout. Position in the list
    is the identity — no separate event ID needed. Only two event types
    have well-known structure (model_request, reward). Everything else is
    opaque pass-through.
    """

    event_type: str  # "model_request", "reward", or any user-defined string
    rollout_id: str
    attempt_id: str  # = K8s pod UID
    timestamp: float  # assigned by store at write time
    data: dict[str, Any]  # event-type-specific payload


class ModelRequestData(BaseModel):
    """Well-known structure for event_type='model_request'.

    Created automatically by the Gateway on every proxied LLM call.
    Not enforced by the Store — this is a documentation/validation helper.
    """

    model: str
    model_version: int | None = None  # training step of the serving model
    request: dict[str, Any]  # original request body (messages, temperature, etc.)
    adjusted_params: dict[str, Any] | None = None  # only if param adjustment changed anything
    response: dict[str, Any]  # full response body
    latency_ms: float | None = None
    http_status: int | None = None
    status: str = "ok"  # "ok", "error", "client_disconnected", "stream_error"
    retry_count: int = 0
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


class RewardData(BaseModel):
    """Well-known structure for event_type='reward'.

    Reported by the environment, evaluator, or runner.
    Not enforced by the Store — this is a documentation/validation helper.
    """

    value: float  # scalar reward (required)
    message: str | None = None  # optional human-readable explanation
    source: str | None = None  # e.g. "agent" for explicit evaluator output, "fallback" for system fill-in
    reason: str | None = None  # optional machine-readable explanation
