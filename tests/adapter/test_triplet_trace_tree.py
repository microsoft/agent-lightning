# Copyright (c) Microsoft. All rights reserved.

import itertools
import json
from typing import Any, Dict, List, Optional

import pytest

from agentlightning.adapter.triplet import TracerTraceToTriplet, TraceTree
from agentlightning.types import Span
from agentlightning.types.tracer import SpanNames
from agentlightning.utils.otel import filter_and_unflatten_attributes

_SEQ = itertools.count()


def qwen_multimodal_attrs(response_id: str) -> Dict[str, Any]:
    """Simplified attributes derived from the provided Qwen trace dump."""
    prompt_content = json.dumps(
        [
            {"type": "text", "text": "Question: How many food item is shown in the bar graph?"},
            {"type": "image_url", "image_url": {"url": "file:///home/kiki/Projects/chartqa/test.png"}},
        ]
    )
    return {
        "gen_ai.request.type": "chat",
        "gen_ai.system": "OpenAI",
        "gen_ai.request.model": "Qwen/Qwen2-VL-2B-Instruct",
        "gen_ai.request.temperature": 0.0,
        "gen_ai.request.streaming": False,
        "gen_ai.request.headers": "{'X-Stainless-Raw-Response': 'true'}",
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": prompt_content,
        "gen_ai.response.id": response_id,
        "gen_ai.response.model": "Qwen/Qwen2-VL-2B-Instruct",
        "gen_ai.usage.total_tokens": 12,
        "gen_ai.usage.prompt_tokens": 10,
        "gen_ai.usage.completion_tokens": 2,
        "gen_ai.completion.0.content": "The bar graph shows 10 food items.",
        "gen_ai.completion.0.finish_reason": "stop",
        "gen_ai.completion.0.role": "assistant",
        # Shortened token arrays to keep the fixture readable.
        "prompt_token_ids": (151644, 8948, 198, 2610),
        "response_token_ids": (785, 3619, 4771),
    }


def gpt_multimodal_attrs(response_id: str) -> Dict[str, Any]:
    """Simplified attributes derived from the provided GPT-4o trace dump."""
    prompt_content = json.dumps(
        [
            {"type": "text", "text": "Question: How many food item is shown in the bar graph?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA..."}},
        ]
    )
    prompt_filter_results = json.dumps(
        [
            {
                "prompt_index": 1,
                "content_filter_result": {"sexual": {"filtered": False, "severity": "safe"}},
            },
            {"prompt_index": 0, "content_filter_result": {}},
        ]
    )
    completion_filter_results = json.dumps(
        {
            "hate": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
        }
    )
    return {
        "gen_ai.request.type": "chat",
        "gen_ai.system": "OpenAI",
        "gen_ai.request.model": "gpt-4.1-mini",
        "gen_ai.request.temperature": 0.0,
        "gen_ai.request.streaming": False,
        "gen_ai.request.headers": "{'X-Stainless-Raw-Response': 'true'}",
        "gen_ai.openai.system_fingerprint": "fp_3dcd5944f5",
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": prompt_content,
        "gen_ai.prompt.prompt_filter_results": prompt_filter_results,
        "gen_ai.response.id": response_id,
        "gen_ai.response.model": "gpt-4.1-mini-2025-04-14",
        "gen_ai.usage.total_tokens": 9,
        "gen_ai.usage.prompt_tokens": 7,
        "gen_ai.usage.completion_tokens": 2,
        "gen_ai.completion.0.finish_reason": "stop",
        "gen_ai.completion.0.role": "assistant",
        "gen_ai.completion.0.content": "The bar graph shows 13 food items.",
        "gen_ai.completion.0.content_filter_results": completion_filter_results,
    }


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


def test_trace_tree_find_llm_calls_dedupes_across_agents():
    root = make_span("root", "session", parent_id=None, start_time=0.0, end_time=10.0)
    agent_a = make_span(
        "agent-a",
        "agent.node",
        parent_id="root",
        start_time=0.5,
        end_time=5.0,
        attributes={"agent.name": "vision-a"},
    )
    agent_b = make_span(
        "agent-b",
        "agent.node",
        parent_id="root",
        start_time=5.1,
        end_time=9.5,
        attributes={"agent.name": "vision-b"},
    )
    shared_response_id = "chatcmpl-shared"
    llm_a = make_span(
        "llm-a",
        "openai.chat.completion",
        parent_id="agent-a",
        start_time=1.0,
        end_time=2.0,
        attributes=qwen_multimodal_attrs(shared_response_id),
    )
    llm_b = make_span(
        "llm-b",
        "openai.chat.completion",
        parent_id="agent-b",
        start_time=6.0,
        end_time=7.0,
        attributes=gpt_multimodal_attrs(shared_response_id),
    )

    tree = TraceTree.from_spans([root, agent_a, agent_b, llm_a, llm_b])
    matches = tree.find_llm_calls(
        llm_call_match=r"openai\.chat\.completion",
        agent_match=None,
        within_matching_subtree="*",
        within_reward=False,
        within_llm_call=False,
        existing_llm_call_response_ids=set(),
    )

    assert len(matches) == 1
    assert matches[0][0].id == "llm-a"


def test_tracer_trace_to_triplet_handles_multimodal_payloads():
    root = make_span("root", "session", parent_id=None, start_time=0.0, end_time=15.0)
    agent = make_span(
        "agent",
        "agent.node",
        parent_id="root",
        start_time=0.5,
        end_time=14.5,
        attributes={"agent.name": "vision-agent"},
    )
    llm_first = make_span(
        "llm-qwen",
        "openai.chat.completion",
        parent_id="agent",
        start_time=1.0,
        end_time=2.0,
        attributes=qwen_multimodal_attrs("chatcmpl-qwen"),
    )
    llm_second = make_span(
        "llm-gpt",
        "openai.chat.completion",
        parent_id="agent",
        start_time=10.0,
        end_time=11.0,
        attributes=gpt_multimodal_attrs("chatcmpl-gpt"),
    )
    reward = make_span(
        "reward",
        "agent.reward",
        parent_id="agent",
        start_time=12.0,
        end_time=12.5,
        attributes=reward_attributes(0.7),
    )

    assert llm_first.attributes["gen_ai.request.headers"] == "{'X-Stainless-Raw-Response': 'true'}"
    assert llm_second.attributes["gen_ai.openai.system_fingerprint"] == "fp_3dcd5944f5"
    assert "gen_ai.prompt.prompt_filter_results" in llm_second.attributes
    qwen_prompt = json.loads(llm_first.attributes["gen_ai.prompt.0.content"])  # type: ignore
    assert qwen_prompt[0]["type"] == "text"
    assert qwen_prompt[1]["image_url"]["url"].startswith("file://")
    gpt_prompt = json.loads(llm_second.attributes["gen_ai.prompt.0.content"])  # type: ignore
    assert gpt_prompt[1]["image_url"]["url"].startswith("data:image/jpeg")
    assert llm_first.attributes["gen_ai.completion.0.content"] == "The bar graph shows 10 food items."
    assert llm_second.attributes["gen_ai.completion.0.content"] == "The bar graph shows 13 food items."

    adapter = TracerTraceToTriplet(agent_match="vision-agent")
    triplets = adapter.adapt([root, agent, llm_first, llm_second, reward])

    assert len(triplets) == 2
    first, second = triplets
    assert list(first.prompt["token_ids"]) == [151644, 8948, 198, 2610]
    assert list(first.response["token_ids"]) == [785, 3619, 4771]
    assert first.metadata["response_id"] == "chatcmpl-qwen"
    assert first.metadata["agent_name"] == "vision-agent"
    assert first.reward is None

    assert second.prompt["token_ids"] == []
    assert second.response["token_ids"] == []
    assert second.metadata["response_id"] == "chatcmpl-gpt"
    assert triplets[0].metadata["agent_name"] == "vision-agent"
    assert triplets[1].metadata["agent_name"] == "vision-agent"
    qwen_prompt_raw = triplets[0].prompt["raw_content"]
    assert qwen_prompt_raw == filter_and_unflatten_attributes(llm_first.attributes, "gen_ai.prompt")
    assert triplets[0].prompt["image_urls"] == ["file:///home/kiki/Projects/chartqa/test.png"]
    qwen_content = json.loads(qwen_prompt_raw[0]["content"])
    assert qwen_content[1]["image_url"]["url"] == "file:///home/kiki/Projects/chartqa/test.png"
    qwen_request = filter_and_unflatten_attributes(llm_first.attributes, "gen_ai.request")
    qwen_response = filter_and_unflatten_attributes(llm_first.attributes, "gen_ai.response")
    assert triplets[0].metadata["request"] == qwen_request
    assert triplets[0].metadata["response"] == qwen_response
    assert triplets[0].response["raw_content"] == qwen_response

    gpt_prompt_raw = triplets[1].prompt["raw_content"]
    assert gpt_prompt_raw == filter_and_unflatten_attributes(llm_second.attributes, "gen_ai.prompt")
    gpt_content = json.loads(gpt_prompt_raw["0"]["content"])
    assert gpt_content[1]["image_url"]["url"].startswith("data:image/jpeg")
    assert triplets[1].prompt["image_urls"] == ["data:image/jpeg;base64,AAA..."]
    gpt_request = filter_and_unflatten_attributes(llm_second.attributes, "gen_ai.request")
    gpt_response = filter_and_unflatten_attributes(llm_second.attributes, "gen_ai.response")
    assert triplets[1].metadata["request"] == gpt_request
    assert triplets[1].metadata["response"] == gpt_response
    assert triplets[1].response["raw_content"] == gpt_response
    assert second.reward == 0.7
