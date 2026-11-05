# Copyright (c) Microsoft. All rights reserved.

"""Versioned process-credit contracts without torch or VERL dependencies."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL_VERSION = 1
ALGORITHM_VERSION = "sealed-process-reinforce.v1"
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")


class CreditProtocolError(ValueError):
    """Malformed or incompatible process evidence; never an implicit zero reward."""


class CreditConflictError(CreditProtocolError):
    """A logical identity was reused inconsistently or evidence was already sealed."""


class IncompletePolicyTraceError(CreditProtocolError):
    """The complete sampled policy trajectory cannot be established."""


class CreditFinalizationTimeout(TimeoutError):
    """A terminal collection did not supply sealed evidence before its deadline."""


class FrozenCreditModel(BaseModel):
    """Reject unknown protocol fields and nonfinite numbers; freeze field assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("version", mode="before", check_fields=False)
    @classmethod
    def exact_version(cls, value: Any) -> Any:
        if type(value) is not int or value != PROTOCOL_VERSION:
            raise CreditProtocolError("unknown process credit protocol version")
        return value


def canonical_json(value: Any) -> bytes:
    """Encode finite semantic JSON independently of dictionary insertion order."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise CreditProtocolError("credit evidence must be finite primitive JSON") from exc


def semantic_digest(value: Any) -> str:
    """Content identity, not source authentication or an executable serialization."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def nonempty(value: str) -> str:
    """Retain caller identity exactly, rejecting empty/whitespace identifiers."""
    if not isinstance(value, str) or not value.strip():
        raise CreditProtocolError("credit identity/source must be nonempty")
    return value


def event_identifier(value: str) -> str:
    """Canonicalize a stable UUID so alternate textual forms cannot duplicate it."""
    try:
        return UUID(value).hex
    except (ValueError, TypeError, AttributeError) as exc:
        raise CreditProtocolError("event identity must be a UUID") from exc


def finite_float32(value: float, context: str) -> float:
    """Guard accumulation and casts without importing a numerical framework."""
    if isinstance(value, bool) or not math.isfinite(value) or abs(value) > FLOAT32_MAX:
        raise CreditProtocolError(f"{context} must be finite and representable as float32")
    return value


class ProcessCreditConfig(FrozenCreditModel):
    """Fixed-before-sampling objective, evidence limits and output policy."""

    enabled: bool = Field(default=False, strict=True)
    gamma: float = Field(default=1.0, ge=0, le=1)
    baseline: Literal["zero", "task_loo"] = "zero"
    normalization_tokens: float = Field(default=1024.0, gt=0)
    terminal_weight: float = 1.0
    process_weight: float = 1.0
    finalization_timeout_seconds: float = Field(default=300.0, gt=0)
    max_events_per_rollout: int = Field(default=100000, gt=0, strict=True)
    max_actions_per_rollout: int = Field(default=1024, gt=0, strict=True)
    max_artifact_bytes: int = Field(default=67108864, gt=0, strict=True)
    artifact_dir: str | None = None

    @field_validator(
        "gamma",
        "normalization_tokens",
        "terminal_weight",
        "process_weight",
        "finalization_timeout_seconds",
        mode="before",
    )
    @classmethod
    def numeric(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CreditProtocolError("process configuration requires finite numeric values")
        return value

    @property
    def digest(self) -> str:
        """Include estimator settings and fixed weights, excluding the output path."""
        return semantic_digest(self.model_dump(mode="json", exclude={"artifact_dir"}))


class CreditEnvelope(FrozenCreditModel):
    """Original action identity or explicit target/evidence reward identity."""

    version: Literal[1] = 1
    event_id: str
    policy_version: int = Field(ge=0, strict=True)
    sequence: int | None = Field(default=None, ge=0, strict=True)
    action_index: int | None = Field(default=None, ge=0, strict=True)
    request_id: str | None = None
    kind: Literal["process", "terminal"] | None = None
    target_event_id: str | None = None
    evidence_through_event_id: str | None = None

    @field_validator("event_id", "target_event_id", "evidence_through_event_id")
    @classmethod
    def uuid_identity(cls, value: str | None) -> str | None:
        return event_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def event_kind(self) -> CreditEnvelope:
        if self.kind is None:
            if self.action_index is None or self.request_id is None:
                raise CreditProtocolError("action envelope requires request ID and original action index")
            nonempty(self.request_id)
            if self.target_event_id is not None or self.evidence_through_event_id is not None:
                raise CreditProtocolError("action envelope cannot carry reward targets")
        else:
            if self.action_index is not None or self.request_id is not None:
                raise CreditProtocolError("reward envelope cannot masquerade as a model request")
            if self.kind == "process" and (self.target_event_id is None or self.evidence_through_event_id is None):
                raise CreditProtocolError("process reward requires target and evidence cutoff")
            if self.kind == "terminal" and self.target_event_id is not None:
                raise CreditProtocolError("terminal reward has no attribution target")
        return self


class CreditRecord(FrozenCreditModel):
    """Semantic native event without transport timestamps; data is copied on ownership."""

    event_type: Literal["model_request", "reward"]
    rollout_id: str
    attempt_id: str
    data: dict[str, Any]

    @property
    def envelope(self) -> CreditEnvelope:
        return CreditEnvelope.model_validate(self.data.get("_agl_credit"))


class FinalizeCreditRequest(FrozenCreditModel):
    """Complete exact action manifest and reward set supplied by the rollout owner."""

    version: Literal[1] = 1
    action_event_ids: tuple[str, ...]
    reward_event_ids: tuple[str, ...]
    terminal_reward_id: str
    policy_version: int = Field(ge=0, strict=True)

    @field_validator("action_event_ids", "reward_event_ids")
    @classmethod
    def identities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        result = tuple(event_identifier(value) for value in values)
        if len(result) != len(set(result)):
            raise CreditProtocolError("duplicate manifest event identity")
        return result

    @field_validator("terminal_reward_id")
    @classmethod
    def terminal_identity(cls, value: str) -> str:
        return event_identifier(value)

    @field_validator("reward_event_ids")
    @classmethod
    def canonical_reward_order(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(values))


class CreditSnapshot(FrozenCreditModel):
    """Sealed original attempt evidence with a semantic watermark and checksum."""

    version: Literal[1] = 1
    rollout_id: str
    attempt_id: str
    policy_version: int = Field(ge=0, strict=True)
    group_id: str
    group_size: int = Field(gt=0, strict=True)
    action_event_ids: tuple[str, ...]
    reward_event_ids: tuple[str, ...]
    terminal_reward_id: str
    watermark: int = Field(ge=-1, strict=True)
    events: tuple[CreditRecord, ...]
    snapshot_hash: str

    @model_validator(mode="after")
    def checksum(self) -> CreditSnapshot:
        if self.snapshot_hash != semantic_digest(self.model_dump(mode="json", exclude={"snapshot_hash"})):
            raise CreditProtocolError("snapshot semantic checksum mismatch")
        return self


class CreditReceipt(FrozenCreditModel):
    """Stable acknowledgement of the immutable evidence boundary."""

    version: Literal[1] = 1
    rollout_id: str
    attempt_id: str
    policy_version: int
    snapshot_hash: str
    watermark: int
    action_count: int
    reward_count: int


class CreditCompletion(FrozenCreditModel):
    """Transport-neutral captured model reply and its original logical identity."""

    body: dict[str, Any]
    status_code: int
    event_id: str
    sequence: int
    action_index: int
    request_id: str
    policy_version: int


class RewardAttribution(FrozenCreditModel):
    """Keep judged action separate from the information cutoff used for credit."""

    reward_id: str
    kind: Literal["process", "terminal"]
    target_index: int | None
    cutoff_index: int | None
    raw_value: float
    weighted_value: float
    source: str


class ResolvedCredit(FrozenCreditModel):
    """Complete action-space return resolution, before any training-row projection."""

    version: Literal[1] = 1
    algorithm_version: Literal["sealed-process-reinforce.v1"] = ALGORITHM_VERSION
    rollout_id: str
    attempt_id: str
    policy_version: int
    group_id: str
    group_size: int
    config_digest: str
    snapshot_hash: str
    action_event_ids: tuple[str, ...]
    rewards: tuple[float, ...]
    returns: tuple[float, ...]
    discounts: tuple[float, ...]
    attributions: tuple[RewardAttribution, ...]
    terminal_outcome: float
    episode_return: float


class CreditArtifact(FrozenCreditModel):
    """Primitive replay evidence, projection provenance and predeclared scheduling."""

    version: Literal[1] = 1
    algorithm_version: Literal["sealed-process-reinforce.v1"] = ALGORITHM_VERSION
    snapshot: CreditSnapshot
    config: ProcessCreditConfig
    resolved: ResolvedCredit
    projection: tuple[dict[str, Any], ...] = ()
    scheduled_groups: tuple[dict[str, Any], ...] = ()
    artifact_hash: str


def validate_sampling_parameters(parameters: Mapping[str, Any], *, prefix: str = "sampling") -> None:
    """Reject sampling changes not present in the trainable policy likelihood."""
    expected = {
        "temperature": 1,
        "top_p": 1,
        "min_p": 0,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "n": 1,
    }
    for key, value in expected.items():
        if parameters.get(key, value) != value:
            raise CreditProtocolError(f"process credit requires {prefix}.{key}={value}")
    if parameters.get("top_k", -1) not in (-1, 0, None):
        raise CreditProtocolError(f"process credit requires {prefix}.top_k disabled")
    for key in (
        "guided_decoding",
        "structured_outputs",
        "allowed_token_ids",
        "logit_bias",
        "stop",
        "ignore_eos",
        "return_score_centering",
        "return_routed_experts",
    ):
        if parameters.get(key):
            raise CreditProtocolError(f"process credit does not support {prefix}.{key}")
