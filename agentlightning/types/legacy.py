# Copyright (c) Microsoft. All rights reserved.

"""Deprecated compatibility models for legacy HTTP protocol paths.

These types are retained to support historical client/server integrations while
core runtime now uses store-based payloads and new execution contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, Field

from .core import Triplet

TaskInput = Any
"""Legacy task payload alias kept for API compatibility."""


class Task(BaseModel):
    """Rollout request served to legacy protocol clients."""

    rollout_id: str
    input: TaskInput

    mode: Optional[str] = None
    resources_id: Optional[str] = None

    create_time: Optional[float] = None
    last_claim_time: Optional[float] = None
    num_claims: Optional[int] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskIfAny(BaseModel):
    """Compatibility envelope for polling legacy task endpoints."""

    is_available: bool
    task: Optional[Task] = None


class RolloutLegacy(BaseModel):
    """Legacy reporting payload exchanged with the deprecated HTTP server."""

    rollout_id: str
    task: Optional[Task] = None
    final_reward: Optional[float] = None
    triplets: Optional[List[Triplet]] = None
    trace: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "A list of spans that conform to the OpenTelemetry JSON format. "
            "Users of the opentelemetry-sdk can generate this by calling "
            "json.loads(readable_span.to_json())."
        ),
    )
    logs: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


RolloutRawResultLegacy = Union[None, float, List[Triplet], List[Dict[str, Any]], List[ReadableSpan], RolloutLegacy]
"""Legacy rollout result type."""

__all__ = [
    "TaskInput",
    "Task",
    "TaskIfAny",
    "RolloutLegacy",
    "RolloutRawResultLegacy",
]
