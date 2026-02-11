# Copyright (c) Microsoft. All rights reserved.

"""Shared fixtures for Vercel AI WebShop tests."""

import itertools
import json
from typing import Any, Dict, List, Optional

import pytest

from agentlightning.adapter.triplet import TracerTraceToTriplet
from agentlightning.types import Span

# Counter for generating unique sequence IDs
_SEQ = itertools.count()

# Pattern to match Vercel AI SDK span names
VERCEL_AI_SDK_LLM_CALL_PATTERN = r"ai\.(generateText|streamText|generateObject)(\.do(Generate|Stream))?"


@pytest.fixture
def adapter() -> TracerTraceToTriplet:
    """Create adapter configured for Vercel AI SDK spans."""
    return TracerTraceToTriplet(llm_call_match=VERCEL_AI_SDK_LLM_CALL_PATTERN)


@pytest.fixture
def make_span():
    """Factory for creating generic test spans."""

    def _make(
        name: str,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        start_time: float = 0.0,
        end_time: float = 1.0,
        span_id: Optional[str] = None,
    ) -> Span:
        return Span.from_attributes(
            rollout_id="rollout-test-001",
            attempt_id="attempt-test-001",
            sequence_id=next(_SEQ),
            trace_id="trace-test-001",
            span_id=span_id or f"span-{next(_SEQ):04d}",
            parent_id=parent_id,
            name=name,
            attributes=attributes or {},
            start_time=start_time,
            end_time=end_time,
        )

    return _make


@pytest.fixture
def make_vercel_ai_span(make_span):
    """Factory for creating Vercel AI SDK style spans.

    Creates spans that match the ai.generateText pattern with optional token IDs.
    """

    def _make(
        name: str = "ai.generateText",
        *,
        prompt_text: str = "Hello",
        response_text: str = "World",
        token_ids: Optional[Dict[str, List[int]]] = None,
        parent_id: Optional[str] = None,
        start_time: float = 0.0,
        end_time: float = 1.0,
    ) -> Span:
        attrs: Dict[str, Any] = {
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": prompt_text,
            "gen_ai.completion.0.content": response_text,
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.response.id": f"resp-{next(_SEQ):04d}",
        }

        # Add token IDs if provided
        if token_ids:
            if "prompt" in token_ids:
                attrs["prompt_token_ids"] = token_ids["prompt"]
            if "response" in token_ids:
                attrs["response_token_ids"] = token_ids["response"]

        return make_span(
            name,
            attributes=attrs,
            parent_id=parent_id,
            start_time=start_time,
            end_time=end_time,
        )

    return _make


@pytest.fixture
def make_reward_span(make_span):
    """Factory for creating reward spans in AgentOps format."""

    def _make(
        reward: float,
        *,
        parent_id: Optional[str] = None,
        start_time: float = 0.0,
        end_time: float = 1.0,
    ) -> Span:
        attrs = {
            "agentops.task.output": json.dumps({"type": "reward", "value": reward}),
        }
        return make_span(
            "reward",
            attributes=attrs,
            parent_id=parent_id,
            start_time=start_time,
            end_time=end_time,
        )

    return _make


@pytest.fixture
def make_session_span(make_span):
    """Factory for creating agent session spans (root spans)."""

    def _make(
        *,
        start_time: float = 0.0,
        end_time: float = 10.0,
    ) -> Span:
        return make_span(
            "agent.session",
            attributes={"agent.name": "webshop-agent"},
            parent_id=None,
            start_time=start_time,
            end_time=end_time,
            span_id="root-session",
        )

    return _make
