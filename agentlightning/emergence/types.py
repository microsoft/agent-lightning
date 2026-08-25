# Copyright (c) Microsoft. All rights reserved.

"""Shared types for the open emergence module."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    """Record produced by RewardAuditAdapter for staleness analysis."""

    rollout_id: str
    attempt_id: str
    reward_value: Optional[float] = None
    reward_dimensions: Dict[str, float] = Field(default_factory=dict)
    task_input_hash: str = ""
    timestamp: Optional[float] = None


class StalenessReport(BaseModel):
    """Report from reward staleness detection."""

    rank_correlation: float
    """Spearman rank correlation between reward and independent success."""
    window_size: int
    severity: str = "advisory"
    """'advisory' | 'warning' | 'critical'"""
    description: str = ""


class DistributionShiftReport(BaseModel):
    """Report from reward distribution shift detection."""

    kl_divergence: float
    window_size: int
    description: str = ""


class ConditionResult(BaseModel):
    """Result of evaluating a single validity condition."""

    condition_name: str
    passed: bool
    value: Optional[float] = None
    threshold: Optional[float] = None
    description: str = ""


class DissolutionSignal(BaseModel):
    """Signal emitted when a dissolution condition fires."""

    trigger: str
    severity: str = "advisory"
    """'advisory' | 'warning' | 'critical'"""
    recommendation: str = ""


class DissolutionAction(BaseModel):
    """Action taken when dissolution is executed."""

    resources_id: str
    policy: str
    action_taken: str
    description: str = ""
