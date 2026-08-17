# Copyright (c) Microsoft. All rights reserved.

"""SHAPER: two-stage skill and context-harness evolution for frozen agents."""

from .algorithm import (
    DEFAULT_HARNESS_CONTRACT,
    SHAPER,
    IncomparableCandidateError,
    RolloutInfrastructureError,
    SkillValidator,
    validate_nonempty_skill,
)
from .prompting import parse_json_object
from .roles import (
    ArtifactProposal,
    OptimizationStage,
    OptimizerRequestContext,
    RoleCompleter,
    RoleRequest,
    SHAPERRoleProtocol,
)
from .sandbox import (
    HarnessOutputValidator,
    HarnessRuntimeError,
    HarnessValidationResult,
    PythonHarnessRuntime,
    PythonHarnessValidator,
)
from .trace import SHAPERTraceAdapter, emit_episode_metadata, emit_round_record
from .types import (
    ArtifactCandidate,
    ArtifactStage,
    CandidateEvaluation,
    EpisodeMetadata,
    EpisodeSummary,
    EpisodeTrace,
    OptimizationEvent,
    RoundCritique,
    RoundRecord,
)

__all__ = [
    "SHAPER",
    "DEFAULT_HARNESS_CONTRACT",
    "IncomparableCandidateError",
    "RolloutInfrastructureError",
    "SkillValidator",
    "validate_nonempty_skill",
    "ArtifactCandidate",
    "ArtifactStage",
    "CandidateEvaluation",
    "EpisodeMetadata",
    "EpisodeSummary",
    "EpisodeTrace",
    "HarnessValidationResult",
    "HarnessOutputValidator",
    "HarnessRuntimeError",
    "OptimizationEvent",
    "PythonHarnessValidator",
    "PythonHarnessRuntime",
    "RoundCritique",
    "RoundRecord",
    "SHAPERTraceAdapter",
    "ArtifactProposal",
    "OptimizationStage",
    "OptimizerRequestContext",
    "RoleCompleter",
    "RoleRequest",
    "SHAPERRoleProtocol",
    "parse_json_object",
    "emit_episode_metadata",
    "emit_round_record",
]
