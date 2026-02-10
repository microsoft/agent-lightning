# Copyright (c) Microsoft. All rights reserved.

"""End-to-end integration tests for OTLP span ingestion and query.

These tests validate the complete flow:
1. Create rollout in LightningStore
2. Dequeue to get attempt
3. Add spans (simulating TypeScript runner)
4. Query spans back
5. Convert to triplets using adapter

This tests the core issue: spans being sent but not found when queried.
"""

import pytest

from agentlightning.adapter.triplet import TracerTraceToTriplet
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import Span

# Pattern to match Vercel AI SDK span names
VERCEL_AI_SDK_LLM_CALL_PATTERN = r"ai\.(generateText|streamText|generateObject)(\.do(Generate|Stream))?"


@pytest.fixture
def store():
    """Create an in-memory LightningStore."""
    return InMemoryLightningStore()


@pytest.fixture
def integration_adapter():
    """Adapter configured for Vercel AI SDK spans."""
    return TracerTraceToTriplet(llm_call_match=VERCEL_AI_SDK_LLM_CALL_PATTERN)


@pytest.mark.asyncio
async def test_full_flow_rollout_to_triplet(store, integration_adapter):
    """Test complete flow: Create rollout -> Add spans -> Query spans -> Convert to triplets.

    This test validates that:
    1. Spans can be added to a valid rollout/attempt
    2. Spans can be queried back with attempt_id="latest"
    3. Adapter produces triplets with token IDs
    """
    # 1. Create rollout
    rollout = await store.enqueue_rollout(
        input={"instruction": "Buy shoes"},
        mode="train",
    )
    rollout_id = rollout.rollout_id

    # 2. Dequeue to get attempt
    dequeued = await store.dequeue_rollout(worker_id="test-worker")
    assert dequeued is not None, "Should dequeue a rollout"
    attempt_id = dequeued.attempt.attempt_id

    # 3. Add spans (simulating TypeScript runner)
    llm_span = Span.from_attributes(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=0,
        trace_id="trace-001",
        span_id="span-001",
        parent_id=None,
        name="ai.generateText",
        attributes={
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Buy shoes",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "search[shoes]",
            "prompt_token_ids": [1, 2, 3],
            "response_token_ids": [4, 5],
        },
        start_time=0.0,
        end_time=1.0,
    )
    reward_span = Span.from_attributes(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=1,
        trace_id="trace-001",
        span_id="span-002",
        parent_id=None,
        name="reward",
        attributes={
            "agentops.task.output": '{"type": "reward", "value": 0.75}',
        },
        start_time=1.0,
        end_time=1.1,
    )

    added = await store.add_many_spans([llm_span, reward_span])
    assert len(added) == 2, f"Expected 2 spans added, got {len(added)}"

    # 4. Query spans back (using "latest" like the daemon does)
    spans = await store.query_spans(rollout_id, attempt_id="latest")
    assert len(spans) == 2, f"Expected 2 spans from query, got {len(spans)}"

    # Verify span attributes are preserved
    span_names = {s.name for s in spans}
    assert "ai.generateText" in span_names, "Should have LLM span"
    assert "reward" in span_names, "Should have reward span"

    # 5. Convert to triplets
    triplets = integration_adapter.adapt(list(spans))
    assert len(triplets) >= 1, f"Expected at least 1 triplet, got {len(triplets)}"

    # 6. Verify triplet has token IDs
    llm_triplet = triplets[0]
    assert llm_triplet.prompt.get("token_ids") == [
        1,
        2,
        3,
    ], f"Expected prompt token_ids [1,2,3], got {llm_triplet.prompt.get('token_ids')}"
    assert llm_triplet.response.get("token_ids") == [
        4,
        5,
    ], f"Expected response token_ids [4,5], got {llm_triplet.response.get('token_ids')}"


@pytest.mark.asyncio
async def test_span_ingestion_requires_valid_attempt(store):
    """Test that spans can only be added to existing rollouts/attempts.

    This validates the span rejection behavior when rollout_id/attempt_id is invalid.
    """
    # Create rollout and dequeue
    rollout = await store.enqueue_rollout(input={}, mode="train")
    dequeued = await store.dequeue_rollout(worker_id="test")
    assert dequeued is not None
    rollout_id = rollout.rollout_id
    attempt_id = dequeued.attempt.attempt_id

    # Adding span to valid rollout/attempt should succeed
    valid_span = Span.from_attributes(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=0,
        trace_id="t1",
        span_id="s1",
        parent_id=None,
        name="test.span",
        attributes={"key": "value"},
        start_time=0.0,
        end_time=1.0,
    )

    result = await store.add_many_spans([valid_span])
    assert len(result) == 1, "Should successfully add span with valid IDs"

    # Adding span to INVALID rollout should fail
    invalid_span = Span.from_attributes(
        rollout_id="invalid-rollout-id",
        attempt_id=attempt_id,
        sequence_id=0,
        trace_id="t2",
        span_id="s2",
        parent_id=None,
        name="test.span",
        attributes={},
        start_time=0.0,
        end_time=1.0,
    )

    with pytest.raises(ValueError, match="Rollout.*not found"):
        await store.add_many_spans([invalid_span])


@pytest.mark.asyncio
async def test_query_spans_by_explicit_attempt_id(store):
    """Test querying spans by explicit attempt_id (not 'latest')."""
    # Create rollout and dequeue
    rollout = await store.enqueue_rollout(input={}, mode="train")
    dequeued = await store.dequeue_rollout(worker_id="test")
    assert dequeued is not None
    rollout_id = rollout.rollout_id
    attempt_id = dequeued.attempt.attempt_id

    # Add span
    span = Span.from_attributes(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=0,
        trace_id="t1",
        span_id="s1",
        parent_id=None,
        name="test.span",
        attributes={"marker": "test-value"},
        start_time=0.0,
        end_time=1.0,
    )
    await store.add_many_spans([span])

    # Query by explicit attempt_id
    queried = await store.query_spans(rollout_id, attempt_id=attempt_id)
    assert len(queried) == 1, f"Expected 1 span, got {len(queried)}"
    assert queried[0].attributes["marker"] == "test-value"

    # Query by "latest" should give same result
    queried_latest = await store.query_spans(rollout_id, attempt_id="latest")
    assert len(queried_latest) == 1, "Query with 'latest' should return same span"


@pytest.mark.asyncio
async def test_empty_spans_query_returns_empty_list(store):
    """Test that querying spans for a rollout with no spans returns empty list."""
    # Create rollout and dequeue (but don't add spans)
    rollout = await store.enqueue_rollout(input={}, mode="train")
    await store.dequeue_rollout(worker_id="test")
    rollout_id = rollout.rollout_id

    # Query should return empty
    spans = await store.query_spans(rollout_id, attempt_id="latest")
    assert len(spans) == 0, "Should return empty list when no spans exist"


@pytest.mark.asyncio
async def test_adapter_produces_empty_triplets_without_token_ids(store, integration_adapter):
    """Test that adapter handles spans without token_ids (current Vercel AI SDK behavior).

    This documents the limitation: Vercel AI SDK doesn't emit token_ids by default.
    """
    # Create rollout and dequeue
    rollout = await store.enqueue_rollout(input={}, mode="train")
    dequeued = await store.dequeue_rollout(worker_id="test")
    assert dequeued is not None
    rollout_id = rollout.rollout_id
    attempt_id = dequeued.attempt.attempt_id

    # Add span WITHOUT token_ids (like Vercel AI SDK would emit)
    llm_span = Span.from_attributes(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=0,
        trace_id="trace-001",
        span_id="span-001",
        parent_id=None,
        name="ai.generateText",
        attributes={
            "gen_ai.prompt.0.content": "Hello",
            "gen_ai.completion.0.content": "World",
            # No prompt_token_ids or response_token_ids!
        },
        start_time=0.0,
        end_time=1.0,
    )
    await store.add_many_spans([llm_span])

    # Query and convert
    spans = await store.query_spans(rollout_id, attempt_id="latest")
    triplets = integration_adapter.adapt(list(spans))

    # Should still produce triplet, but with empty token_ids
    assert len(triplets) == 1, "Should produce 1 triplet"
    assert triplets[0].prompt.get("token_ids", []) == [], "Should have empty prompt token_ids"
    assert triplets[0].response.get("token_ids", []) == [], "Should have empty response token_ids"
