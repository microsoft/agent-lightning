# Copyright (c) Microsoft. All rights reserved.

"""Extension points for benchmark-faithful SHAPER role protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Protocol, Sequence

from .types import ArtifactCandidate, CandidateEvaluation, OptimizationEvent

RoleContent = str | list[dict[str, Any]]
RoleResponseFormat = Literal["json_object", "text"]
OptimizationStage = Literal["skill", "harness"]


@dataclass(frozen=True)
class RoleRequest:
    """One model call made by a benchmark role protocol."""

    system_prompt: str
    user_content: RoleContent
    temperature: float = 0.0
    response_format: RoleResponseFormat = "text"


@dataclass(frozen=True)
class ArtifactProposal:
    """Parsed replacement artifact returned by an optimizer role."""

    rationale: str
    artifact: str


@dataclass(frozen=True)
class OptimizerRequestContext:
    """State supplied when a benchmark builds one optimizer request."""

    parent: ArtifactCandidate
    stage: OptimizationStage
    textual_gradient: Any
    round_index: int
    branch_index: int
    optimization_history: Sequence[OptimizationEvent]
    harness_contract: str
    harness_function_name: str
    harness_smoke_args: Sequence[Any]
    validation_feedback: str = ""


RoleCompleter = Callable[[RoleRequest], Awaitable[str]]


class SHAPERRoleProtocol(Protocol):
    """Benchmark-owned formatting and parsing for SHAPER's model roles."""

    async def build_textual_gradient(
        self,
        evaluation: CandidateEvaluation,
        complete: RoleCompleter,
    ) -> Any:
        """Diagnose one development batch using benchmark-specific roles."""

        ...

    def build_optimizer_request(self, context: OptimizerRequestContext) -> RoleRequest:
        """Build the skill- or harness-optimizer request for one proposal."""

        ...

    def parse_optimizer_response(
        self,
        stage: OptimizationStage,
        response: str,
    ) -> ArtifactProposal:
        """Extract a complete replacement artifact from the role response."""

        ...
