# Copyright (c) Microsoft. All rights reserved.

"""Rollout result dataclasses used as GEPA's Trajectory and RolloutOutput types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentlightning.types import RolloutStatus, Span


@dataclass
class RolloutTrajectory:
    """Execution trace for a single rollout, used as GEPA's ``Trajectory`` type.

    Captures the full span history and metadata needed by
    `LightningGEPAAdapter.make_reflective_dataset` to build per-component
    evidence records for GEPA's reflection step.
    """

    rollout_id: str
    status: RolloutStatus
    spans: List[Span]
    final_reward: Optional[float]
    input: Any
    messages: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: {})  # pyright: ignore[reportUnknownVariableType]


@dataclass
class RolloutOutput:
    """Summarized output for a single rollout, used as GEPA's ``RolloutOutput`` type.

    A lightweight summary carrying only the identifiers and scalar reward.
    GEPA forwards these through ``EvaluationBatch.outputs`` but does not
    interpret them directly.
    """

    rollout_id: str
    status: RolloutStatus
    final_reward: Optional[float]
