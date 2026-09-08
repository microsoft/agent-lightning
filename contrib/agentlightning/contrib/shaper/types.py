# Copyright (c) Microsoft. All rights reserved.

"""Typed records shared by the SHAPER algorithm, agents, and trace adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentlightning.types import PromptTemplate, RolloutStatus

ArtifactStage = Literal["seed", "skill", "harness"]
"""Artifact currently being evolved by SHAPER."""


class RoundRecord(BaseModel):
    """Observable record for one planner/executor interaction round.

    Agents should emit one record after every executed command. Observations use
    OpenAI-compatible content parts so a diagnostic model can consume text,
    image URLs, or base64-encoded images without a benchmark-specific adapter.
    Hidden simulator state and ground-truth annotations must not be included.
    """

    model_config = ConfigDict(extra="forbid")

    record_type: Literal["shaper_round"] = "shaper_round"
    round_index: int = Field(ge=0)
    task_instruction: str
    planner_response: str
    command: str
    observation_before: List[Dict[str, Any]] = Field(default_factory=lambda: list[Dict[str, Any]]())
    observation_after: List[Dict[str, Any]] = Field(default_factory=lambda: list[Dict[str, Any]]())
    context_payload: Any = None
    harness_input: Any = None
    execution_steps: int = Field(default=0, ge=0)
    action_result: Dict[str, Any] = Field(default_factory=dict)
    runtime_errors: List[str] = Field(default_factory=list)


class EpisodeMetadata(BaseModel):
    """Optional non-visual metadata emitted once near the end of an episode."""

    model_config = ConfigDict(extra="forbid")

    record_type: Literal["shaper_episode"] = "shaper_episode"
    environment_invalid: bool = False
    termination_reason: str = ""
    runtime_errors: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class EpisodeTrace(BaseModel):
    """Structured trajectory extracted from Agent Lightning spans."""

    model_config = ConfigDict(extra="forbid")

    rollout_id: str = ""
    task: Any = None
    status: RolloutStatus = "failed"
    final_reward: Optional[float] = None
    rounds: List[RoundRecord] = Field(default_factory=lambda: list[RoundRecord]())
    metadata: EpisodeMetadata = Field(default_factory=EpisodeMetadata)
    adapter_errors: List[str] = Field(default_factory=list)


class RoundCritique(BaseModel):
    """Judger output grounded in one before/after execution transition."""

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=0)
    progress: Literal["success", "partial", "failed", "unclear"]
    progress_score: float = Field(ge=0.0, le=1.0)
    observable_change: str
    command_assessment: str
    reasoning_assessment: str
    context_assessment: str
    likely_cause: str
    suggested_fix: str


class EpisodeSummary(BaseModel):
    """Compact episode-level textual gradient input."""

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    reward: float
    environment_invalid: bool
    instruction_fidelity: str
    progress_and_outcome: str
    repetition_or_recovery: str
    decomposition_quality: str
    context_effectiveness: str
    root_cause: str
    actionable_change: str


class ArtifactCandidate(BaseModel):
    """Versioned pair of model-external artifacts optimized by SHAPER."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    version: str
    skill: PromptTemplate
    harness: PromptTemplate
    stage: ArtifactStage
    parent_version: Optional[str] = None
    rationale: str = ""
    validation_score: Optional[float] = None

    def artifact_key(self) -> tuple[str, str]:
        """Return a stable value key used to remove duplicate proposals."""

        return self.skill.template, self.harness.template


class CandidateEvaluation(BaseModel):
    """Reward and trace bundle from evaluating one candidate on one batch."""

    model_config = ConfigDict(extra="forbid")

    candidate_version: str
    mode: Literal["train", "val"]
    requested_rollouts: int = Field(ge=0)
    finished_rollouts: int = Field(ge=0)
    valid_rollouts: int = Field(ge=0)
    score: float
    traces: List[EpisodeTrace] = Field(default_factory=lambda: list[EpisodeTrace]())


class OptimizationEvent(BaseModel):
    """Serializable optimization-history entry supplied to later optimizer calls."""

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=0)
    stage: Literal["skill", "harness"]
    parent_version: str
    candidate_version: Optional[str] = None
    rationale: str
    validation_score: Optional[float] = None
    validation_error: Optional[str] = None
