# Copyright (c) Microsoft. All rights reserved.

"""Trace emission and adaptation helpers for SHAPER rollouts."""

from __future__ import annotations

from typing import Sequence, cast

from agentlightning.adapter import TraceAdapter
from agentlightning.emitter import emit_object, get_object_value
from agentlightning.reward import find_final_reward
from agentlightning.semconv import AGL_OBJECT
from agentlightning.types import Span

from .types import EpisodeMetadata, EpisodeTrace, RoundRecord


def emit_round_record(record: RoundRecord) -> None:
    """Emit one observable planner/executor transition into the active trace."""

    emit_object(record.model_dump(mode="json"), attributes={"shaper.record_type": "round"})


def emit_episode_metadata(metadata: EpisodeMetadata) -> None:
    """Emit optional episode validity and termination metadata."""

    emit_object(metadata.model_dump(mode="json"), attributes={"shaper.record_type": "episode"})


class SHAPERTraceAdapter(TraceAdapter[EpisodeTrace]):
    """Extract SHAPER records and final reward from an Agent Lightning trace."""

    def adapt(self, source: Sequence[Span], /) -> EpisodeTrace:
        rounds: list[RoundRecord] = []
        metadata = EpisodeMetadata()
        errors: list[str] = []

        for span in sorted(source, key=lambda item: item.sequence_id):
            if span.name != AGL_OBJECT:
                continue
            try:
                payload: object = get_object_value(span)
            except (RuntimeError, TypeError, ValueError) as exc:
                errors.append(f"object span {span.span_id}: {exc}")
                continue
            if not isinstance(payload, dict):
                continue

            object_payload = cast(dict[str, object], payload)
            record_type = object_payload.get("record_type")
            try:
                if record_type == "shaper_round":
                    rounds.append(RoundRecord.model_validate(object_payload))
                elif record_type == "shaper_episode":
                    metadata = EpisodeMetadata.model_validate(object_payload)
            except ValueError as exc:
                errors.append(f"invalid {record_type!r} record in span {span.span_id}: {exc}")

        rounds.sort(key=lambda item: item.round_index)
        return EpisodeTrace(
            final_reward=find_final_reward(source),
            rounds=rounds,
            metadata=metadata,
            adapter_errors=errors,
        )
