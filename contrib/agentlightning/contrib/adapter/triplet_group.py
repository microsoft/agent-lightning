# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Dict, List, Optional

from agentlightning.types import Span, Triplet
from agentlightning.adapter.triplet import TracerTraceToTriplet

class TracerTraceToTripletGroup(TracerTraceToTriplet):
    """Convert tracer-emitted spans into triplet trajectories.

    Attributes:
        repair_hierarchy: When `True`, repair the span tree using
            [`TraceTree.repair_hierarchy()`][agentlightning.adapter.triplet.TraceTree.repair_hierarchy]
            before matching calls and rewards.
        llm_call_match: Regular expression pattern that selects LLM call span names.
        agent_match: Optional regular expression pattern for agent span names. When omitted, spans
            from any agent are considered.
        exclude_llm_call_in_reward: When `True`, ignore matches under reward spans while searching
            for rewards.
        reward_match: Strategy used to associate rewards with LLM calls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_span_groups(self, spans):
        def resolve_step_count(span, next_span, spans, index):
            """
            Determine step_count for a given span using next_span or fallback search.
            """
            # CASE A: If next_span exists and parent_id matches
            if next_span and span.parent_id == next_span.span_id:
                return next_span.attributes.get("step_count")

            # CASE B: Fallback — search forward for agentlightning.operation
            for s in spans[index + 1 :]:
                if s.name == "agentlightning.operation" and span.parent_id == s.span_id:
                    return s.attributes.get("step_count")

            return None

        def extract_step_count_from_links(span):
            """
            Extract step_count from agentlightning.link.* attributes.
            """
            key = span.attributes.get("agentlightning.link.0.key_match")
            if key == "step_count":
                return span.attributes.get("agentlightning.link.0.value_match")
            return None

        span_groups = {}

        for i, span in enumerate(spans):
            next_span = spans[i + 1] if i + 1 < len(spans) else None
            step_count = None

            if span.name == "openai.chat.completion":
                step_count = resolve_step_count(span, next_span, spans, i)
                if step_count is None:
                    continue

                step_count = str(step_count)
                span_groups.setdefault(step_count, {})
                span_groups[step_count]["call_span"] = span

            elif span.name == "agentlightning.object":
                step_count = extract_step_count_from_links(span)
                if step_count is None:
                    continue

                step_count = str(step_count)
                span_groups.setdefault(step_count, {})
                span_groups[step_count]["object_span"] = span

            elif span.name == "agentlightning.annotation":
                step_count = extract_step_count_from_links(span)
                if step_count is None:
                    continue

                step_count = str(step_count)
                span_groups.setdefault(step_count, {})
                span_groups[step_count]["annotation_span"] = span

        return span_groups

    def adapt_group(self, source: Sequence[Span], /) -> List[Triplet]:
        span_groups = self._extract_span_groups(source)

        def token_ids(span: Optional[Span], key: str) -> list:
            return span.attributes.get(key, []) if span else []

        def reward0(span: Optional[Span]) -> float:
            if not span:
                return 0.0
            return float(span.attributes.get("agentlightning.reward.0.value", 0.0))

        def reward1(span: Optional[Span]) -> Optional[float]:
            if not span:
                return None
            val = span.attributes.get("agentlightning.reward.1.value")
            return float(val) if val is not None else None

        def message(span: Optional[Span]) -> Optional[str]:
            if not span:
                return None
            return span.attributes.get("agentlightning.object.literal")

        triplets: List[Triplet] = []

        for group in span_groups.values():
            call_span = group.get("call_span")
            if not token_ids(call_span, "prompt_token_ids"):
                continue

            object_span = group.get("object_span")
            annotation_span = group.get("annotation_span")
            request_id = group.get("request_id")

            triplets.append(
                Triplet(
                    prompt={"token_ids": token_ids(call_span, "prompt_token_ids")},
                    response={"token_ids": token_ids(call_span, "response_token_ids")},
                    reward=reward0(annotation_span),
                    metadata={
                        "response_id": request_id,
                        "intrinsic_reward": reward1(annotation_span),
                        "message": message(object_span),
                    },
                )
            )

        return triplets