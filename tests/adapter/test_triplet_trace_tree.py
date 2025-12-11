# Copyright (c) Microsoft. All rights reserved.

import itertools
import json
from typing import Any, Dict, List, Optional

import pytest

from agentlightning.adapter.triplet import TracerTraceToTriplet, TraceTree
from agentlightning.types import Span
from agentlightning.types.tracer import SpanNames

_SEQ = itertools.count()


def make_span(
    span_id: str,
    name: str,
    *,
    parent_id: Optional[str],
    start_time: float,
    end_time: float,
    attributes: Optional[Dict[str, Any]] = None,
) -> Span:
    return Span.from_attributes(
        rollout_id="rollout-1",
        attempt_id="attempt-1",
        sequence_id=next(_SEQ),
        trace_id="trace-1",
        span_id=span_id,
        parent_id=parent_id,
        name=name,
        attributes=attributes or {},
        start_time=start_time,
        end_time=end_time,
    )


def make_llm_span(
    span_id: str,
    *,
    parent_id: str,
    start: float,
    end: float,
    prompt_ids: Optional[List[int]] = None,
    response_ids: Optional[List[int]] = None,
    response_id: Optional[str] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> Span:
    attrs: Dict[str, Any] = {
        "prompt_token_ids": prompt_ids or [],
        "response_token_ids": response_ids or [],
    }
    if response_id is not None:
        attrs["gen_ai.response.id"] = response_id
    if extra_attrs:
        attrs.update(extra_attrs)
    return make_span(
        span_id,
        "openai.chat.completion",
        parent_id=parent_id,
        start_time=start,
        end_time=end,
        attributes=attrs,
    )


def reward_attributes(value: float) -> Dict[str, Any]:
    return {
        "agentops.task.output": json.dumps({"type": "reward", "value": value}),
    }


def test_trace_tree_from_spans_orders_children_and_agent_names():
    root = make_span(
        "root",
        "agent.session",
        parent_id=None,
        start_time=0.0,
        end_time=10.0,
        attributes={"agent.name": "primary-agent"},
    )
    llm = make_llm_span(
        "llm",
        parent_id="root",
        start=1.0,
        end=2.0,
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_id="resp-1",
    )

    tree = TraceTree.from_spans([llm, root])

    assert tree.id == "root"
    assert [child.id for child in tree.children] == ["llm"]
    assert tree.find_id("llm") is tree.children[0]
    assert tree.names_tuple() == ("agent.session [primary-agent]", [("openai.chat.completion", [])])
    as_json = tree.to_json()
    assert as_json["children"][0]["span"]["name"] == "openai.chat.completion"


def test_trace_tree_virtual_root_for_multiple_roots():
    first_root = make_span("root-a", "agent.first", parent_id=None, start_time=0.0, end_time=5.0)
    second_root = make_span("root-b", "agent.second", parent_id=None, start_time=5.0, end_time=9.0)

    tree = TraceTree.from_spans([first_root, second_root])

    assert tree.id == "virtual-root"
    assert tree.span.name == "virtual-root"
    assert tree.start_time == first_root.start_time
    assert tree.end_time == second_root.end_time
    assert {child.id for child in tree.children} == {"root-a", "root-b"}


def test_trace_tree_handles_missing_parent_and_empty_input():
    with pytest.raises(ValueError):
        TraceTree.from_spans([])

    orphan_child = make_span(
        "child",
        "agent.child",
        parent_id="ghost-parent",
        start_time=2.0,
        end_time=4.0,
        attributes={"agent.name": "nested"},
    )
    llm = make_llm_span(
        "grandchild",
        parent_id="child",
        start=3.0,
        end=3.5,
        prompt_ids=[1],
        response_ids=[2],
        response_id="resp-nested",
    )

    tree = TraceTree.from_spans([llm, orphan_child])

    assert tree.id == "ghost-parent"
    assert tree.span.name == SpanNames.VIRTUAL.value
    assert tree.span.rollout_id == orphan_child.rollout_id
    assert [child.id for child in tree.children] == ["child"]
    assert tree.children[0].children[0].id == "grandchild"


def test_trace_tree_repair_hierarchy_moves_llm_span_under_agent():
    root = make_span("root", "session", parent_id=None, start_time=0.0, end_time=10.0)
    agent = make_span(
        "agent",
        "agent.node",
        parent_id="root",
        start_time=1.0,
        end_time=9.0,
        attributes={"agent.name": "planner"},
    )
    llm = make_llm_span(
        "llm",
        parent_id="root",
        start=2.0,
        end=3.0,
        prompt_ids=[42],
        response_ids=[7],
        response_id="resp-planner",
    )

    tree = TraceTree.from_spans([root, agent, llm])
    assert any(child.id == "llm" for child in tree.children)

    tree.repair_hierarchy()

    assert not any(child.id == "llm" for child in tree.children)
    agent_node = tree.find_id("agent")
    assert agent_node is not None
    assert [child.id for child in agent_node.children] == ["llm"]


def test_trace_tree_to_trajectory_skips_empty_and_dedupes_llm_calls():
    root = make_span("root", "session", parent_id=None, start_time=0.0, end_time=10.0)
    agent = make_span(
        "agent",
        "agent.node",
        parent_id="root",
        start_time=1.0,
        end_time=9.0,
        attributes={"agent.name": "primary-agent"},
    )
    first = make_llm_span(
        "llm-1",
        parent_id="agent",
        start=2.0,
        end=3.0,
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_id="resp-1",
    )
    duplicate = make_llm_span(
        "llm-2",
        parent_id="agent",
        start=3.2,
        end=3.8,
        prompt_ids=[9],
        response_ids=[8],
        response_id="resp-1",
    )
    empty_tokens = make_llm_span(
        "llm-3",
        parent_id="agent",
        start=4.0,
        end=5.0,
        prompt_ids=[],
        response_ids=[],
        response_id="resp-2",
    )
    reward = make_span(
        "reward",
        "agent.reward",
        parent_id="agent",
        start_time=6.0,
        end_time=6.1,
        attributes=reward_attributes(0.5),
    )

    tree = TraceTree.from_spans([root, agent, first, duplicate, empty_tokens, reward])

    trajectory = tree.to_trajectory(
        agent_match="primary-agent",
        dedup_llm_call=True,
        _skip_empty_token_spans=True,
    )
    assert len(trajectory) == 1
    triplet = trajectory[0]
    assert triplet.prompt["token_ids"] == [1, 2]
    assert triplet.response["token_ids"] == [3, 4]
    assert triplet.metadata["response_id"] == "resp-1"
    assert triplet.metadata["agent_name"] == "primary-agent"
    assert triplet.reward == 0.5

    with_final_reward = tree.to_trajectory(
        agent_match="primary-agent",
        dedup_llm_call=True,
        _skip_empty_token_spans=True,
        final_reward=1.0,
    )
    assert len(with_final_reward) == 1
    assert with_final_reward[0].reward == 1.0


def test_tracer_trace_to_triplet_repair_required_for_agent_filter():
    root = make_span("root", "session", parent_id=None, start_time=0.0, end_time=10.0)
    agent = make_span(
        "agent",
        "agent.node",
        parent_id="root",
        start_time=1.0,
        end_time=9.0,
        attributes={"agent.name": "planner"},
    )
    llm_outside_agent = make_llm_span(
        "llm",
        parent_id="root",
        start=2.0,
        end=3.0,
        prompt_ids=[7],
        response_ids=[8],
        response_id="resp-planner",
    )
    reward = make_span(
        "reward",
        "agent.reward",
        parent_id="agent",
        start_time=4.0,
        end_time=4.5,
        attributes=reward_attributes(0.3),
    )
    spans = [root, agent, llm_outside_agent, reward]

    adapter = TracerTraceToTriplet(agent_match="planner")
    triplets = adapter.adapt(spans)
    assert len(triplets) == 1
    assert triplets[0].metadata["agent_name"] == "planner"
    assert triplets[0].reward == 0.3

    adapter_without_repair = TracerTraceToTriplet(repair_hierarchy=False, agent_match="planner")
    assert adapter_without_repair.adapt(spans) == []


def test_tracer_trace_to_triplet_dedup_and_skip_empty_token_spans():
    root = make_span("root", "session", parent_id=None, start_time=0.0, end_time=10.0)
    agent = make_span(
        "agent",
        "agent.node",
        parent_id="root",
        start_time=1.0,
        end_time=9.0,
        attributes={"agent.name": "collector"},
    )
    kept_llm = make_llm_span(
        "llm-1",
        parent_id="agent",
        start=2.0,
        end=3.0,
        prompt_ids=[10],
        response_ids=[20],
        response_id="resp-shared",
    )
    duplicate_llm = make_llm_span(
        "llm-2",
        parent_id="agent",
        start=3.5,
        end=4.2,
        prompt_ids=[99],
        response_ids=[98],
        response_id="resp-shared",
    )
    missing_tokens = make_llm_span(
        "llm-3",
        parent_id="agent",
        start=5.0,
        end=5.5,
        prompt_ids=[],
        response_ids=[],
        response_id="resp-3",
    )
    reward = make_span(
        "reward",
        "agent.reward",
        parent_id="agent",
        start_time=6.0,
        end_time=6.5,
        attributes=reward_attributes(0.25),
    )
    spans = [root, agent, kept_llm, duplicate_llm, missing_tokens, reward]

    adapter = TracerTraceToTriplet(_skip_empty_token_spans=True)
    triplets = adapter.adapt(spans)

    assert len(triplets) == 1
    assert triplets[0].prompt["token_ids"] == [10]
    assert triplets[0].response["token_ids"] == [20]
    assert triplets[0].metadata["response_id"] == "resp-shared"
    assert triplets[0].reward == 0.25
