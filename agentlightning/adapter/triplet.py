# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel

from agentlightning.types import SpanNames, Triplet
from agentlightning.types.tracer import Span

from .base import TraceAdapter


class Transition(BaseModel):
    """
    Transition class representing one transition in a trajectory.
    State and action are a list of token IDs.
    """

    state: List[int]
    action: List[int]
    response_id: Optional[str]
    # action_logprobs: List[float]
    agent_name: str
    reward: Optional[float]


class RewardMatchPolicy(str, Enum):
    """How to find the reward for each transition from the trace.
    In all cases, the reward must have data `{"type": "reward", "value": <float>|None}`,
    as defined in `reward.py`.
    """

    FIRST_SIBLING = "first_sibling"
    """Use the first sibling in the current trace subtree as the reward, except another LLM call match is found."""

    FIRST_OCCURRENCE = "first_occurrence"
    """Use the first occurrence of the reward (in start time order) that occur after the current LLM call match.
    """


class TraceTree:
    """
    A trace item, along with its span and children.
    """

    def __init__(
        self,
        id: str,
        span: Span,
        children: Optional[List["TraceTree"]] = None,
    ):
        self.id = id
        self.span = span
        self.children = children or []

    @property
    def start_time(self):
        return self.span.start_time

    @property
    def end_time(self):
        return self.span.end_time

    def find_id(self, id: str) -> "TraceTree | None":
        if self.id == id:
            return self
        for child in self.children:
            found = child.find_id(id)
            if found:
                return found
        return None

    def add_child(self, child: "TraceTree") -> None:
        self.children.append(child)

    def visualize(self, filename: str, interested_span_match: str | None = None) -> None:
        """
        Visualize the trace tree using graphviz.
        For debugging purposes only.
        Use `interested_span_match` to filter the spans (and its ancesters) to be visualized.
        """
        import graphviz

        dot = graphviz.Digraph(comment="Trace Tree")

        should_visit_cache: Dict[str, bool] = {}

        def should_visit(node: "TraceTree") -> bool:
            if node.id in should_visit_cache:
                return should_visit_cache[node.id]
            if interested_span_match is not None:
                if re.search(interested_span_match, node.span.name):
                    should_visit_cache[node.id] = True
                    return True
                else:
                    should_visit_cache[node.id] = False
                    for child in node.children:
                        if should_visit(child):
                            should_visit_cache[node.id] = True

                    return should_visit_cache[node.id]
            else:
                return True

        def visit(node: "TraceTree") -> bool:
            if not should_visit(node):
                return False
            agent_name = node.agent_name()
            vis_name = node.id[:8] + " (" + node.span.name + ")"
            if agent_name is not None:
                vis_name += " [" + agent_name + "]"
            dot.node(node.id, vis_name)  # type: ignore
            for child in node.children:
                if visit(child):
                    dot.edge(node.id, child.id)  # type: ignore
            return True

        visit(self)
        dot.render(filename, format="png", cleanup=True)  # type: ignore

    def names_tuple(self) -> Tuple[str, List[Any]]:
        """Return the span name, and a list of children.
        Each child is also a tuple of span name and a list of children.
        Useful for debugging and testing.
        """
        name = self.span.name
        agent_name = self.agent_name()
        if agent_name is not None:
            name += " [" + agent_name + "]"
        children_names: List[Tuple[str, List[Any]]] = []
        for child in self.children:
            child_name, child_children = child.names_tuple()
            children_names.append((child_name, child_children))
        return name, children_names

    def traverse(self) -> List["TraceTree"]:
        """
        Traverse the trace tree and return a list of all spans.
        """
        spans: List["TraceTree"] = [self]
        for child in self.children:
            spans.extend(child.traverse())
        return spans

    def to_json(self) -> dict[str, Any]:
        if isinstance(self.span, ReadableSpan):
            span_data = json.loads(self.span.to_json())
        else:
            span_data = self.span.model_dump()
        return {
            "id": self.id,
            "span": span_data,
            "children": [child.to_json() for child in self.children],
        }

    @classmethod
    def from_spans(cls, spans: List[Span]) -> "TraceTree":
        """
        Create a TraceTree from a list of spans.
        All spans without parents found will be considered as candidate root spans.
        If multiple root spans are found, a virtual root span will be created as the parent of all root spans.
        """

        if not spans:
            raise ValueError("No spans provided to create TraceTree.")

        # Process trace items in topological order
        id_to_span = {span.span_id: span for span in spans}

        forward_graph: dict[str, list[str]] = {}
        root_ids: list[str] = []
        for span in spans:
            span_id = span.span_id
            if span.parent_id is None:
                root_ids.append(span.span_id)
            else:
                if span.parent_id not in forward_graph:
                    forward_graph[span.parent_id] = []
                forward_graph[span.parent_id].append(span_id)

        # Diff between span with data and forward_graph keys
        # Sometimes the top-level session span is lost.
        unfound_roots = set(forward_graph.keys()) - set(id_to_span.keys())
        for unfound_root in unfound_roots:
            root_ids.append(unfound_root)

        def visit(node_id: str) -> "TraceTree":
            children: list[TraceTree] = []
            if node_id in forward_graph:
                for child_id in forward_graph[node_id]:
                    children.append(visit(child_id))

            if node_id not in id_to_span:
                assert len(children) > 0
                virtual_span = Span.from_attributes(
                    rollout_id=children[0].span.rollout_id,
                    attempt_id=children[0].span.attempt_id,
                    sequence_id=children[0].span.sequence_id,
                    trace_id=children[0].span.trace_id,
                    span_id=node_id,
                    parent_id=None,
                    attributes={},
                    start_time=min(child.start_time for child in children if child.start_time is not None),
                    end_time=max(child.end_time for child in children if child.end_time is not None),
                )
                return cls(node_id, virtual_span, children=children)
            else:
                return cls(
                    node_id,
                    id_to_span[node_id],
                    children=children,
                )

        # Create a virtual root span if multiple root spans are found
        if len(root_ids) > 1:
            root_spans = [visit(root_id) for root_id in root_ids]
            virtual_root = TraceTree(
                id="virtual-root",
                span=Span.from_attributes(
                    rollout_id=root_spans[0].span.rollout_id,
                    attempt_id=root_spans[0].span.attempt_id,
                    sequence_id=root_spans[0].span.sequence_id,
                    trace_id=root_spans[0].span.trace_id,
                    span_id=None,  # Generate one
                    parent_id=None,
                    name="virtual-root",
                    attributes={},
                    start_time=root_spans[0].start_time,
                    end_time=root_spans[-1].end_time,
                ),
                children=root_spans,
            )
            return virtual_root
        elif len(root_ids) == 0:
            # No root spans found
            raise ValueError("No root spans found in the trace.")
        else:
            root_span = visit(root_ids[0])
            return root_span

    def agent_name(self) -> Optional[str]:
        """Return the name of agent span. Return the agent or None (not an agent at all).
        Extend this function to support more agent frameworks."""
        attributes = self.span.attributes
        if attributes is None:  # type: ignore
            return None

        # Case 1: OpenAI Agent SDK
        agent_name = cast(Optional[str], attributes.get("agent.name"))
        if agent_name is not None:
            return agent_name

        # Case 2: Agentops decorator @agent
        is_agent = attributes.get("agentops.span.kind") == "agent"
        if is_agent:
            agent_name = cast(Optional[str], attributes.get("operation.name"))
            if agent_name is not None:
                return agent_name

        # Case 3: Autogen team
        agent_name = cast(Optional[str], attributes.get("recipient_agent_type"))
        if agent_name is not None:
            return agent_name

        # Case 4: LangGraph
        agent_name = cast(Optional[str], attributes.get("langchain.chain.type"))
        if agent_name is not None:
            return agent_name

        # Case 5: agent-framework
        agent_name = cast(Optional[str], attributes.get("executor.id"))
        if agent_name is not None:
            return agent_name

    def maybe_reward_dict(self) -> dict[str, Any]:
        for key in [
            "agentops.task.output",  # newer versions of agentops
            "agentops.entity.output",
        ]:
            output = self.span.attributes.get(key)  # type: ignore
            if output:
                if isinstance(output, dict):
                    return output
                elif isinstance(output, str):
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        return {}

        # Latest emit reward format
        if self.span.name == SpanNames.REWARD.value and self.span.attributes:
            return {"type": "reward", "value": self.span.attributes.get("reward", None)}
        return {}

    def is_reward_span(self) -> bool:
        maybe_reward = self.maybe_reward_dict()
        return maybe_reward and maybe_reward.get("type") == "reward"  # type: ignore

    def find_llm_calls(
        self,
        *,
        llm_call_match: str,
        agent_match: Optional[str],
        within_matching_subtree: str | None = None,
        within_reward: Optional[bool] = None,
        within_llm_call: Optional[bool] = None,
        existing_llm_call_response_ids: Optional[set[str]] = None,
    ) -> List[Tuple["TraceTree", str]]:
        """Find all LLM calls in the trace tree.

        The LLM call is defined as a span with type = request and name matching `llm_call_match`.
        If `agent_match` is not None, it must also reside in an agent span (type = agent) with name matched.

        Return a list of traces and the agent names (why it's selected).
        """
        llm_calls: List[Tuple[TraceTree, str]] = []

        is_llm_call = True
        if within_matching_subtree is None or within_reward is True:
            # We must be in an interesting agent subtree, and not in a reward span.
            is_llm_call = False
        if re.search(llm_call_match, self.span.name) is None:
            # The span name does not match the LLM call match.
            is_llm_call = False
        if is_llm_call:
            # Check the response id
            response_id: Optional[str] = self.span.attributes.get("gen_ai.response.id")  # type: ignore
            if response_id is None and within_llm_call is True:
                is_llm_call = False
            if (
                response_id is not None
                and existing_llm_call_response_ids is not None
                and response_id in existing_llm_call_response_ids
            ):
                is_llm_call = False

            if is_llm_call:
                llm_calls.append((self, within_matching_subtree))  # type: ignore
                existing_llm_call_response_ids = existing_llm_call_response_ids or set()
                if response_id is not None:
                    existing_llm_call_response_ids.add(response_id)
                if within_llm_call is not None:
                    within_llm_call = True

        agent_name = self.agent_name()
        if agent_name is not None:
            if agent_match is None or re.search(agent_match, agent_name):
                within_matching_subtree = agent_name
            else:
                within_matching_subtree = None

        if within_reward is not None and self.is_reward_span():
            within_reward = True

        for child in self.children:
            llm_calls.extend(
                child.find_llm_calls(
                    llm_call_match=llm_call_match,
                    agent_match=agent_match,
                    within_matching_subtree=within_matching_subtree,
                    within_reward=within_reward,
                    within_llm_call=within_llm_call,
                    existing_llm_call_response_ids=existing_llm_call_response_ids,
                )
            )

        return llm_calls

    def repair_hierarchy(self) -> None:
        """
        We find that sometimes the hierarchy is not correct, due to the way the spans are created.
        The spans within the agent frameworks (e.g., OpenAI Agent SDK) and spans within the LLM frameworks
        (e.g., Anthropic) are created in two systems.
        So the inner LLM completion span does not necessarily have an agent span as a parent.
        Rather they sometimes directly become children of the root span.
        This becomes a problem when we want to select the LLM completion span with agent as filter.
        To repair the hierarchy, for each children of the root span, we find a span over the whole tree,
        with duration covering the current span and being closest to the current span.

        This function modifies the tree in place.
        """
        nodes_to_repair = list(self.children)
        for repair_node in nodes_to_repair:
            if len(self.children) == 1:
                # If there is only one child, we don't need to repair the hierarchy.
                break
            # Find the closest parent span (but not the root itself)
            closest_parent = None
            closest_duration = float("inf")
            for node in self.traverse():
                if node.id == repair_node.id:
                    continue
                if node is self:
                    continue
                if node.start_time <= repair_node.start_time and node.end_time >= repair_node.end_time:  # type: ignore
                    duration_delta = node.end_time - repair_node.end_time + repair_node.start_time - node.start_time  # type: ignore
                    if duration_delta > 0 and duration_delta < closest_duration:
                        closest_duration = duration_delta  # type: ignore
                        closest_parent = node

            # Repair the hierarchy
            if closest_parent is not None:
                self.children.remove(repair_node)
                closest_parent.children.append(repair_node)

    def match_rewards(self, reward_match: str, llm_calls: List["TraceTree"]) -> dict[str, Optional[float]]:
        """Match the rewards to the LLM calls."""
        llm_call_ids = set([llm_call.id for llm_call in llm_calls])
        rewards: dict[str, Optional[float]] = {}

        if reward_match == RewardMatchPolicy.FIRST_OCCURRENCE:
            time_sorted: List[TraceTree] = cast(List[TraceTree], sorted(self.traverse(), key=lambda x: x.start_time))  # type: ignore
            assign_to: List[Tuple[str, int]] = []  # type: ignore
            for item in time_sorted:
                if item.id in llm_call_ids:
                    assign_to.append((item.id, item.end_time))  # type: ignore

                # get reward
                agentops_output = item.maybe_reward_dict()
                if agentops_output and agentops_output.get("type") == "reward":
                    for assign_to_id, assign_to_end_time in reversed(assign_to):
                        # This reward happens before the end of the LLM call.
                        if assign_to_end_time > item.start_time:  # type: ignore
                            continue
                        # Ok, we found someone to assign to
                        if assign_to_id in rewards:
                            # If the reward is already set, skip
                            continue
                        rewards[assign_to_id] = agentops_output.get("value", None)
                        break

        elif reward_match == RewardMatchPolicy.FIRST_SIBLING:
            for item in self.traverse():
                assign_to: List[Tuple[str, int]] = []
                for child in item.children:
                    if child.id in llm_call_ids:
                        assign_to.append(child.id)  # type: ignore

                    agentops_output = item.maybe_reward_dict()
                    if agentops_output and agentops_output.get("type") == "reward":
                        for assign_to_id, assign_to_end_time in reversed(assign_to):
                            if assign_to_end_time > item.start_time:  # type: ignore
                                # This reward happens before the end of the LLM call.
                                continue
                            if assign_to_id in rewards:
                                continue
                            rewards[assign_to_id] = agentops_output.get("value", None)
                            break

        return rewards

    def to_trajectory(
        self,
        llm_call_match: str = r"openai\.chat\.completion",
        agent_match: Optional[str] = None,
        exclude_llm_call_in_reward: bool = True,
        dedup_llm_call: bool = True,
        reward_match: RewardMatchPolicy = RewardMatchPolicy.FIRST_OCCURRENCE,
        final_reward: Optional[float] = None,
    ) -> List[Triplet]:
        """Convert the trace tree to a trajectory.

        First, we find all the LLM calls (span type = request, `llm_call_match` matching the span name).
        If the agent match is set, we check, for each LLM call,
        if it resides in an agent (span type = agent, `agent_match` matching the span name).
        The above sets the basis for the trajectory, as we use the prompt token IDs and response token IDs for each LLM call,
        as the state and action of each transition.

        Then, we find the reward for each transition.
        The reward is searched on the trace tree, after the LLM call,
        until the next LLM call or the end of the tree depending on the policy.
        It can be enforced to a sibling or the first occurrence in the time order, depending on the policy.
        If a reward is never found for a transition, it is set to None.
        """
        # Find all LLM calls
        llm_calls = self.find_llm_calls(
            llm_call_match=llm_call_match,
            agent_match=agent_match,
            within_matching_subtree="*" if agent_match is None else None,
            within_reward=False if exclude_llm_call_in_reward else None,
            within_llm_call=False if dedup_llm_call else None,
            existing_llm_call_response_ids=set(),
        )
        id_transitions = [
            (
                llm_call.id,
                Triplet(
                    prompt={"token_ids": llm_call.span.attributes.get("prompt_token_ids", [])},  # type: ignore
                    response={"token_ids": llm_call.span.attributes.get("response_token_ids", [])},  # type: ignore
                    reward=None,
                    metadata=dict(
                        response_id=llm_call.span.attributes.get(  # type: ignore
                            "gen_ai.response.id", None
                        ),  # it works at least for OpenAI
                        agent_name=agent_name,
                    ),
                ),
            )
            for llm_call, agent_name in llm_calls
        ]

        rewards = self.match_rewards(reward_match, [call for call, _ in llm_calls])
        transitions = [
            transition.model_copy(update={"reward": rewards.get(id, None)}) for id, transition in id_transitions
        ]
        if final_reward is not None and len(transitions) > 0:
            # Add the final reward to the last transition
            transitions[-1] = transitions[-1].model_copy(update={"reward": final_reward})
        return transitions

    def __repr__(self):
        return (
            f"TraceTree(id={self.id}, span={self.span}, start_time={self.start_time}, "
            + f"end_time={self.end_time}, children={self.children})"
        )


class BaseTraceTripletAdapter(TraceAdapter[List[Triplet]]):
    """
    Base class for trace triplet adapters.
    """


class TraceTripletAdapter(BaseTraceTripletAdapter):
    """
    An adapter to convert OpenTelemetry spans to triplet data.

    Attributes:
        repair_hierarchy: When `repair_hierarchy` is set to True, the trace will be repaired with the time information.
            See `TraceTree.repair_hierarchy` for more details.
        llm_call_match: Regular expression pattern to match LLM call span names.
        agent_match: Optional regular expression pattern to match agent span names. If None, all agents are matched.
        exclude_llm_call_in_reward: Whether to exclude LLM calls that occur within reward spans.
        reward_match: Policy for matching rewards to LLM calls.
    """

    def __init__(
        self,
        repair_hierarchy: bool = True,
        llm_call_match: str = r"openai\.chat\.completion",
        agent_match: Optional[str] = None,
        exclude_llm_call_in_reward: bool = True,
        reward_match: RewardMatchPolicy = RewardMatchPolicy.FIRST_OCCURRENCE,
    ):
        self.repair_hierarchy = repair_hierarchy
        self.llm_call_match = llm_call_match
        self.agent_match = agent_match
        self.exclude_llm_call_in_reward = exclude_llm_call_in_reward
        self.reward_match = reward_match

    def visualize(
        self,
        source: Union[List[Span], List[ReadableSpan]],
        /,
        filename: str = "trace_tree",
        interested_span_match: str | None = None,
    ) -> TraceTree:
        """
        Visualize the trace tree.

        Args:
            source (List[Span]): The list of OpenTelemetry spans to visualize.
            filename (str): The base filename for the output visualization (default: "trace_tree").
            interested_span_match (str | None): Optional regular expression pattern to highlight or focus on specific spans in the visualization.

        Returns:
            TraceTree: The constructed trace tree object.
        """
        source_normalized = [
            Span.from_opentelemetry(span, "dummy", "dummy", 0) if isinstance(span, ReadableSpan) else span
            for span in source
        ]
        trace_tree = TraceTree.from_spans(source_normalized)
        if self.repair_hierarchy:
            trace_tree.repair_hierarchy()
        trace_tree.visualize(filename, interested_span_match=interested_span_match)
        return trace_tree

    def adapt(self, source: Union[List[Span], List[ReadableSpan]], /) -> List[Triplet]:  # type: ignore
        """Convert OpenTelemetry spans to a list of Triplet objects."""
        source_normalized = [
            Span.from_opentelemetry(span, "dummy", "dummy", 0) if isinstance(span, ReadableSpan) else span
            for span in source
        ]
        trace_tree = TraceTree.from_spans(source_normalized)
        if self.repair_hierarchy:
            trace_tree.repair_hierarchy()
        trajectory = trace_tree.to_trajectory(
            llm_call_match=self.llm_call_match,
            agent_match=self.agent_match,
            exclude_llm_call_in_reward=self.exclude_llm_call_in_reward,
            reward_match=self.reward_match,
        )
        return trajectory


class LlmProxyTripletAdapter(BaseTraceTripletAdapter):
    """
    Converting telemetry data emitted by the LLM Proxy to triplet data.
    This adapter is very experimental. Should only be used when the TraceTripletAdapter does not work at all.

    IMPORTANT: Do NOT rely on timestamps here. Proxy spans can be emitted from different
    machines with unsynchronized clocks. We therefore treat `sequence_id` as the only
    reliable ordering primitive and perform "first occurrence" reward matching using
    sequence order only.

    Strategy:

    1) Sort spans by (sequence_id, start_time).
    2) Extract LLM calls that expose prompt/response token IDs from either:
       - litellm_request         (sometimes only metadata, ignore if no token ids)
       - raw_gen_ai_request      (llm.hosted_vllm.* stringified fields)
    3) Extract rewards from spans whose attributes contain an AgentOps-style
       reward payload or explicit REWARD span.
    4) For each reward with sequence R, assign it to the most recent *unmatched* LLM call
       with sequence < R. Ignore timestamps completely.
    """

    def _literal_eval_maybe(self, v: Any) -> Any:
        import ast

        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except Exception:
                return v
        return v

    def _extract_tokens_from_raw(self, attrs: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """Extract token ids from raw_gen_ai_request attributes.

        - llm.hosted_vllm.prompt_token_ids: string -> List[int]
        - llm.hosted_vllm.response_token_ids: string -> List[List[int]] -> take first
        - llm.hosted_vllm.choices: string -> [{'token_ids': [...]}] -> take first
        """
        prompt_ids: List[int] = []
        resp_ids: List[int] = []

        # prompt
        p = attrs.get("llm.hosted_vllm.prompt_token_ids")
        p = self._literal_eval_maybe(p)
        if isinstance(p, list) and all(isinstance(x, int) for x in p):  # type: ignore
            prompt_ids = cast(List[int], p)

        # response preferred path
        r = attrs.get("llm.hosted_vllm.response_token_ids")
        r = self._literal_eval_maybe(r)
        if isinstance(r, list) and len(r) > 0 and isinstance(r[0], list):  # type: ignore
            first = cast(List[Any], r[0])
            if all(isinstance(x, int) for x in first):
                resp_ids = cast(List[int], first)

        # fallback via choices
        if not resp_ids:
            choices = attrs.get("llm.hosted_vllm.choices")
            choices = self._literal_eval_maybe(choices)
            if isinstance(choices, list) and choices:
                cand = cast(Any, choices[0])
                if isinstance(cand, dict):
                    tids = cast(Dict[str, Any], cand).get("token_ids")
                    if isinstance(tids, list) and all(isinstance(x, int) for x in tids):  # type: ignore
                        resp_ids = cast(List[int], tids)

        return prompt_ids, resp_ids

    def _extract_tokens_from_openai(self, attrs: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        prompt_ids = cast(Any, attrs.get("prompt_token_ids") or [])
        resp_ids = cast(Any, attrs.get("response_token_ids") or [])
        prompt_ids = self._literal_eval_maybe(prompt_ids)
        resp_ids = self._literal_eval_maybe(resp_ids)
        if not (isinstance(prompt_ids, list) and all(isinstance(x, int) for x in prompt_ids)):  # type: ignore
            prompt_ids = []
        if not (isinstance(resp_ids, list) and all(isinstance(x, int) for x in resp_ids)):  # type: ignore
            resp_ids = []
        return cast(List[int], prompt_ids), cast(List[int], resp_ids)

    def _maybe_reward_value(self, span: Span) -> Optional[float]:
        """
        Parse reward from typical AgentOps payload or explicit REWARD span.
        """
        attrs = span.attributes or {}

        # AgentOps new/old keys
        for k in ("agentops.task.output", "agentops.entity.output"):
            v = attrs.get(k)
            v = self._literal_eval_maybe(v)
            if isinstance(v, dict) and cast(Dict[str, Any], v).get("type") == "reward":
                rv = cast(Dict[str, Any], v).get("value", None)
                if rv is None or isinstance(rv, (int, float)):
                    return None if rv is None else float(rv)

        # Explicit reward span
        if span.name == SpanNames.REWARD.value:
            rv = attrs.get("reward", None)
            if rv is None or isinstance(rv, (int, float)):
                return None if rv is None else float(rv)

        return None

    def _request_id_from_attrs(self, attrs: Dict[str, Any]) -> Optional[str]:
        # Prefer OpenAI-like id if present, else proxy raw id.
        rid = attrs.get("gen_ai.response.id") or attrs.get("llm.hosted_vllm.id")
        return str(rid) if isinstance(rid, str) and rid else None

    def adapt(self, source: List[Span], /) -> List[Triplet]:  # type: ignore
        # 1) Sort deterministically by (sequence_id, start_time).
        spans = sorted(
            source,
            key=lambda s: (s.sequence_id, s.start_time),
        )

        # 2) Collect LLM calls with token IDs.
        llm_items: List[Dict[str, Any]] = []
        seen_request_ids: set[str] = set()
        for s in spans:
            attrs = s.attributes or {}
            prompt_ids: List[int] = []
            resp_ids: List[int] = []

            if s.name == "raw_gen_ai_request":
                prompt_ids, resp_ids = self._extract_tokens_from_raw(attrs)
            elif s.name == "litellm_request":
                # Some proxies never include token ids here. Ignore unless present.
                prompt_ids, resp_ids = self._extract_tokens_from_openai(attrs)

            if prompt_ids and resp_ids:
                rid = self._request_id_from_attrs(attrs)
                if rid:
                    # Duplicated request ID. This request is already handled.
                    if rid in seen_request_ids:
                        continue
                    seen_request_ids.add(rid)
                llm_items.append(
                    dict(
                        span=s,
                        seq=s.sequence_id,
                        response_ids=resp_ids,
                        prompt_ids=prompt_ids,
                        request_id=rid,
                    )
                )

        # Order LLM items by sequence only.
        llm_items.sort(key=lambda x: x["seq"])

        # Collect rewards by sequence only.
        rewards: List[Tuple[int, Optional[float]]] = []
        for s in spans:
            val = self._maybe_reward_value(s)
            if val is not None:
                rewards.append((s.sequence_id, val))

        # First-occurrence matching by sequence_id only:
        # For reward at sequence R, assign to the most recent unmatched LLM with seq < R.
        assigned: Dict[str, Optional[float]] = {}
        for r_seq, r_val in sorted(rewards, key=lambda x: x[0]):
            for item in reversed(llm_items):
                sid = item["span"].span_id
                if sid in assigned:
                    continue
                if item["seq"] < r_seq:
                    assigned[sid] = r_val
                    break

        # Build triplets in LLM sequence order.
        triplets: List[Triplet] = []
        for item in llm_items:
            s = item["span"]
            triplets.append(
                Triplet(
                    prompt={"token_ids": item["prompt_ids"]},
                    response={"token_ids": item["response_ids"]},
                    reward=assigned.get(s.span_id, None),
                    metadata=dict(
                        # This is called response_id to align with the other adapters.
                        response_id=item["request_id"],
                    ),
                )
            )

        return triplets
