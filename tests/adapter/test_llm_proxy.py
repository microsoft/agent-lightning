# Copyright (c) Microsoft. All rights reserved.

import json
from typing import Any, Dict, List

from agentlightning.adapter import LlmProxyTraceToTriplet
from agentlightning.types import Span


def _mk_span(
    *,
    span_id: str,
    name: str,
    seq: int,
    start: int,
    end: int,
    attrs: Dict[str, Any],
    parent_id: str | None = None,
) -> Span:
    return Span.from_attributes(
        rollout_id="rollout-X",
        attempt_id="attempt-Y",
        sequence_id=seq,
        trace_id="trace-Z",
        span_id=span_id,
        parent_id=parent_id,
        name=name,
        attributes=attrs,
        start_time=start,
        end_time=end,
    )


def _raw_attrs_with_tokens(
    prompt_ids: List[int], resp_ids: List[int], *, response_id: str, model: str = "my/own-model"
):
    """
    Build attributes for a 'raw_gen_ai_request' span mirroring the sample.
    Token id fields are stringified lists, as in the provided trace dump.
    """
    return {
        "llm.hosted_vllm.prompt_token_ids": str(prompt_ids),
        # vLLM sometimes sends response_token_ids as List[List[int]]
        "llm.hosted_vllm.response_token_ids": str([resp_ids]),
        "llm.hosted_vllm.choices": str([{"token_ids": resp_ids}]),
        "llm.hosted_vllm.id": response_id,
        "llm.hosted_vllm.model": model,
    }


def _agentops_reward_attrs(value: float):
    # As in the sample: agentops.task.output is often a JSON-encoded dict
    return {
        "agentops.task.output": json.dumps({"type": "reward", "value": value}),
        "agentops.span.kind": "task",
        "operation.name": "agentops_reward_operation",
    }


def test_sequence_matching_assigns_reward_to_latest_prior_llm():
    """
    Grounded in the provided sample:
    - One raw_gen_ai_request at sequence 1
    - One raw_gen_ai_request at sequence 4
    - A reward emitted at sequence 6
    First-occurrence by sequence should assign the reward to the seq=4 LLM call.
    """
    # Use short token lists for readability, but structure matches the sample.
    p1 = [151644, 8948, 198]
    r1 = [151657, 198, 4913]

    p2 = [151644, 872, 198, 151657, 198]
    r2 = [15, 13, 15, 22, 24, 3245]

    spans = [
        _mk_span(
            span_id="s-raw-1",
            name="raw_gen_ai_request",
            seq=1,
            start=1000,
            end=1010,
            attrs=_raw_attrs_with_tokens(p1, r1, response_id="chatcmpl-AAA"),
        ),
        _mk_span(
            span_id="s-raw-4",
            name="raw_gen_ai_request",
            seq=4,
            start=2000,
            end=2020,
            attrs=_raw_attrs_with_tokens(p2, r2, response_id="chatcmpl-BBB"),
        ),
        _mk_span(
            span_id="s-reward-6",
            name="agentops_reward_operation.task",
            seq=6,
            start=3000,
            end=3001,
            attrs=_agentops_reward_attrs(0.0),
        ),
    ]

    adapter = LlmProxyTraceToTriplet()
    trips = adapter.adapt(spans)

    # Two LLM calls → two triplets
    assert len(trips) == 2

    # Ordered by LLM sequence only
    t1, t2 = trips

    assert t1.prompt["token_ids"] == p1
    assert t1.response["token_ids"] == r1
    # Reward appears at seq 6, so the *latest* prior LLM (seq 4) gets it, not seq 1.
    assert t1.reward is None

    assert t2.prompt["token_ids"] == p2
    assert t2.response["token_ids"] == r2
    assert t2.reward == 0.0


def test_deduplicates_same_response_id_from_raw_spans():
    """
    If two raw_gen_ai_request spans share the same response id,
    only the first should be kept as an LLM call.
    """
    ids = [1, 2, 3]
    spans = [
        _mk_span(
            span_id="dup-a",
            name="raw_gen_ai_request",
            seq=1,
            start=100,
            end=101,
            attrs=_raw_attrs_with_tokens(ids, ids, response_id="chatcmpl-DUP"),
        ),
        _mk_span(
            span_id="dup-b",
            name="raw_gen_ai_request",
            seq=2,
            start=110,
            end=111,
            attrs=_raw_attrs_with_tokens(ids, ids, response_id="chatcmpl-DUP"),
        ),
        _mk_span(
            span_id="reward-3",
            name="agentops_reward_operation.task",
            seq=3,
            start=200,
            end=201,
            attrs=_agentops_reward_attrs(1.0),
        ),
    ]

    adapter = LlmProxyTraceToTriplet()
    trips = adapter.adapt(spans)

    assert len(trips) == 1
    assert trips[0].prompt["token_ids"] == ids
    assert trips[0].response["token_ids"] == ids
    # Reward at seq 3 should attach to the only LLM call (seq 1).
    assert trips[0].reward == 1.0


def test_ignores_litellm_request_without_token_ids():
    """
    litellm_request spans often carry usage and prompt/response *text*,
    but not token id arrays. Adapter should ignore these unless token ids exist.
    """
    spans = [
        _mk_span(
            span_id="litellm-no-tids",
            name="litellm_request",
            seq=1,
            start=10,
            end=20,
            attrs={
                "gen_ai.response.id": "chatcmpl-XYZ",
                # no prompt_token_ids / response_token_ids here
            },
        ),
        _mk_span(
            span_id="reward-2",
            name="agentops_reward_operation.task",
            seq=2,
            start=30,
            end=31,
            attrs=_agentops_reward_attrs(0.5),
        ),
    ]

    adapter = LlmProxyTraceToTriplet()
    trips = adapter.adapt(spans)

    # No token ids → no triplets
    assert trips == []


def test_reward_none():
    prompt_ids = [1, 2, 3]
    resp_ids = [4, 5, 6]
    spans = [
        _mk_span(
            span_id="s-raw-1",
            name="raw_gen_ai_request",
            seq=1,
            start=100,
            end=101,
            attrs=_raw_attrs_with_tokens(prompt_ids, resp_ids, response_id="chatcmpl-BUG"),
        ),
    ]

    adapter = LlmProxyTraceToTriplet()

    triplets = adapter.adapt(spans)
    assert len(triplets) == 1
    assert triplets[0].prompt["token_ids"] == prompt_ids
    assert triplets[0].response["token_ids"] == resp_ids
    assert triplets[0].reward is None
    assert triplets[0].metadata["response_id"] == "chatcmpl-BUG"


def test_rewards_before_or_equal_sequence_are_skipped():
    """
    Rewards that appear before an LLM call (or share its sequence id) should be ignored.
    Only the first reward strictly after the LLM call should apply.
    """
    prompt_ids = [9, 9, 9]
    resp_ids = [8, 8, 8]
    spans = [
        _mk_span(
            span_id="reward-early",
            name="agentops_reward_operation.task",
            seq=1,
            start=10,
            end=11,
            attrs=_agentops_reward_attrs(1.0),
        ),
        _mk_span(
            span_id="llm-call",
            name="raw_gen_ai_request",
            seq=2,
            start=20,
            end=21,
            attrs=_raw_attrs_with_tokens(prompt_ids, resp_ids, response_id="chatcmpl-SKIP"),
        ),
        _mk_span(
            span_id="reward-same-seq",
            name="agentops_reward_operation.task",
            seq=2,
            start=22,
            end=23,
            attrs=_agentops_reward_attrs(2.0),
        ),
        _mk_span(
            span_id="reward-late",
            name="agentops_reward_operation.task",
            seq=3,
            start=30,
            end=31,
            attrs=_agentops_reward_attrs(3.5),
        ),
    ]

    adapter = LlmProxyTraceToTriplet()
    triplets = adapter.adapt(spans)

    assert len(triplets) == 1
    triplet = triplets[0]
    assert triplet.prompt["token_ids"] == prompt_ids
    assert triplet.response["token_ids"] == resp_ids
    # Only the reward after the LLM call should attach.
    assert triplet.reward == 3.5


def test_multiple_rewards_attach_to_latest_unmatched_llm_calls():
    """
    Rewards should attach to the most recent unmatched LLM call whose sequence id is smaller.
    Later rewards backfill older unmatched LLM calls.
    """
    p1, r1 = [1, 1], [2, 2]
    p2, r2 = [3, 3], [4, 4]
    p3, r3 = [5, 5], [6, 6]
    spans = [
        _mk_span(
            span_id="llm-1",
            name="raw_gen_ai_request",
            seq=2,
            start=100,
            end=110,
            attrs=_raw_attrs_with_tokens(p1, r1, response_id="chatcmpl-A"),
        ),
        _mk_span(
            span_id="llm-2",
            name="raw_gen_ai_request",
            seq=4,
            start=120,
            end=130,
            attrs=_raw_attrs_with_tokens(p2, r2, response_id="chatcmpl-B"),
        ),
        _mk_span(
            span_id="llm-3",
            name="raw_gen_ai_request",
            seq=6,
            start=140,
            end=150,
            attrs=_raw_attrs_with_tokens(p3, r3, response_id="chatcmpl-C"),
        ),
        _mk_span(
            span_id="reward-1",
            name="agentops_reward_operation.task",
            seq=5,
            start=200,
            end=201,
            attrs=_agentops_reward_attrs(0.1),
        ),
        _mk_span(
            span_id="reward-2",
            name="agentops_reward_operation.task",
            seq=7,
            start=210,
            end=211,
            attrs=_agentops_reward_attrs(0.2),
        ),
        _mk_span(
            span_id="reward-3",
            name="agentops_reward_operation.task",
            seq=8,
            start=220,
            end=221,
            attrs=_agentops_reward_attrs(0.3),
        ),
    ]

    adapter = LlmProxyTraceToTriplet()
    triplets = adapter.adapt(spans)

    assert len(triplets) == 3
    # Triplets are emitted in LLM sequence order.
    assert triplets[0].reward == 0.3  # backfilled by the last reward
    assert triplets[1].reward == 0.1  # first reward targets the latest prior call (seq=4)
    assert triplets[2].reward == 0.2  # second reward picks up the remaining unmatched call (seq=6)
