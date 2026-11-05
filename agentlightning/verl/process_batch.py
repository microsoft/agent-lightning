# Copyright (c) Microsoft. All rights reserved.

"""Complete original-action projection without truncation or reward-dependent filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentlightning.credit.artifacts import replay_credit
from agentlightning.credit.state import validate_snapshot
from agentlightning.credit.types import IncompletePolicyTraceError, validate_sampling_parameters

if TYPE_CHECKING:
    from agentlightning.verl.agl_rollout_manager import CompletedRollout


@dataclass(frozen=True)
class TokenOrigin:
    """Identity/offset of one complete generated token, including action EOS."""

    action_event_id: str
    action_index: int
    offset: int


@dataclass(frozen=True)
class ProjectionLimits:
    """Predeclared row limits; exceeding them aborts instead of training a prefix."""

    max_prompt_length: int
    max_response_length: int
    max_total_length: int | None = None
    eos_token_id: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for value in (self.max_prompt_length, self.max_response_length, self.max_total_length):
            if value is not None and (type(value) is not int or value < 1):
                raise IncompletePolicyTraceError("projection limits must be positive integers")


@dataclass(frozen=True)
class ProjectedCreditRow:
    """One native row with masks derived exclusively from original token provenance."""

    rollout_id: str
    data_id: str
    snapshot_hash: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    origins: tuple[TokenOrigin | None, ...]
    image_urls: tuple[str, ...] | None = None

    @property
    def response_mask(self) -> tuple[int, ...]:
        return tuple(int(origin is not None) for origin in self.origins)


def _tokens(value, description: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or any(type(token) is not int or token < 0 for token in value):
        raise IncompletePolicyTraceError(f"invalid original {description} token geometry")
    return tuple(value)


def _check_limits(row: ProjectedCreditRow, limits: ProjectionLimits) -> None:
    if len(row.prompt_ids) > limits.max_prompt_length or len(row.response_ids) > limits.max_response_length:
        raise IncompletePolicyTraceError(
            "complete process action exceeds row limit; bound generation before sampling or increase limits"
        )
    if limits.max_total_length is not None and len(row.prompt_ids) + len(row.response_ids) > limits.max_total_length:
        raise IncompletePolicyTraceError("complete process context exceeds total row limit")


def project_process_rollout(
    completed: CompletedRollout, *, level: str, limits: ProjectionLimits
) -> list[ProjectedCreditRow]:
    """Project sealed successful actions into native transition or merged text rows.

    Tool/prompt/pad positions receive no origin. Prefix mismatch starts a new
    complete row; merging never changes action time, evidence cutoff or token
    identity. No action is inferred from matching token strings.
    """
    artifact = completed.process_credit
    if artifact is None:
        raise IncompletePolicyTraceError("process projection requires sealed original evidence")
    resolved = replay_credit(artifact)
    actions, _ = validate_snapshot(artifact.snapshot)
    triplets = completed.triplets or []
    if level not in {"transition", "trajectory"} or len(triplets) != len(actions):
        raise IncompletePolicyTraceError("process action count or aggregation level is inconsistent")
    rows = []
    expected_origins = set()
    for index, (triplet, action) in enumerate(zip(triplets, actions, strict=True)):
        envelope = action.envelope
        metadata = triplet.metadata.get("process_credit", {})
        if metadata.get("action_event_id") != envelope.event_id or metadata.get("action_index") != index:
            raise IncompletePolicyTraceError("native triplet lost its original action identity")
        validate_sampling_parameters(action.data.get("adjusted_params") or action.data.get("request", {}))
        source = action.data["response"]
        choice = source["choices"][0]
        prompt = _tokens(triplet.prompt.get("token_ids"), "prompt")
        response = _tokens(triplet.response.get("token_ids"), "response")
        original_prompt = source.get("prompt_token_ids") or choice.get("prompt_token_ids", [])
        if prompt != _tokens(original_prompt, "captured prompt") or response != _tokens(
            choice.get("token_ids"), "captured response"
        ):
            raise IncompletePolicyTraceError("native token projection differs from original sampled response")
        log_probs = triplet.response.get("log_probs")
        if not response or not isinstance(log_probs, list | tuple) or len(log_probs) != len(response):
            raise IncompletePolicyTraceError("complete action requires aligned captured token logprobs")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value > 0
            for value in log_probs
        ):
            raise IncompletePolicyTraceError("nonfinite/invalid captured action logprob")
        finish = choice.get("finish_reason")
        eos = limits.eos_token_id
        eos_ids = (eos,) if isinstance(eos, int) else (eos or ())
        if finish != "length" and (finish != "stop" or response[-1] not in eos_ids):
            raise IncompletePolicyTraceError("complete action EOS is absent or unverifiable")
        usage = source.get("usage", {})
        if "completion_tokens" in usage and usage["completion_tokens"] != len(response):
            raise IncompletePolicyTraceError("captured response omitted sampled tokens including EOS")
        origins = tuple(TokenOrigin(envelope.event_id, index, offset) for offset in range(len(response)))
        expected_origins.update(origins)
        row = ProjectedCreditRow(
            completed.rollout_id,
            completed.data_id,
            resolved.snapshot_hash,
            prompt,
            response,
            tuple(log_probs),
            origins,
            tuple(triplet.image_urls) if triplet.image_urls else None,
        )
        _check_limits(row, limits)
        if level == "trajectory" and row.image_urls:
            raise IncompletePolicyTraceError("trajectory-level images remain unsupported; use transition mode")
        if (
            level == "trajectory"
            and rows
            and prompt[: len(rows[-1].prompt_ids + rows[-1].response_ids)]
            == rows[-1].prompt_ids + rows[-1].response_ids
        ):
            previous = rows.pop()
            observation = prompt[len(previous.prompt_ids) + len(previous.response_ids) :]
            row = ProjectedCreditRow(
                row.rollout_id,
                row.data_id,
                row.snapshot_hash,
                previous.prompt_ids,
                previous.response_ids + observation + response,
                previous.log_probs + (0.0,) * len(observation) + tuple(log_probs),
                previous.origins + (None,) * len(observation) + origins,
            )
            _check_limits(row, limits)
        rows.append(row)
    actual = [origin for row in rows for origin in row.origins if origin is not None]
    if len(actual) != len(set(actual)) or set(actual) != expected_origins:
        raise IncompletePolicyTraceError("projection must retain every sampled action token exactly once")
    return rows
