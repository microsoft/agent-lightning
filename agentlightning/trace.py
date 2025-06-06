import json
import re
from enum import Enum
from typing import Any, List, Optional, Tuple
from pydantic import BaseModel

from agentops.sdk import TracingCore
from agentops.sdk.processors import SpanProcessor, Context
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import Span, ReadableSpan


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


class LightningTrace:
    """
    A trace item, along with its span and children.
    """

    def __init__(
        self,
        id: str,
        span: ReadableSpan,
        children: Optional[List["LightningTrace"]] = None,
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

    def find_id(self, id: str) -> "LightningTrace | None":
        if self.id == id:
            return self
        for child in self.children:
            found = child.find_id(id)
            if found:
                return found
        return None

    def add_child(self, child: "LightningTrace") -> None:
        self.children.append(child)

    def _tree_visualize(self, filename: str, interested_span_match: str | None = None) -> None:
        """
        Visualize the trace tree using graphviz.
        For debugging purposes only.
        Use `interested_span_match` to filter the spans (and its ancesters) to be visualized.
        """
        import graphviz

        dot = graphviz.Digraph(comment="Trace Tree")

        should_visit_cache = {}

        def should_visit(node: "LightningTrace") -> bool:
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

        def visit(node: "LightningTrace") -> bool:
            if not should_visit(node):
                return False
            agent_name = node.agent_name()
            vis_name = node.id[:8] + " (" + node.span.name + ")"
            if agent_name is not None:
                vis_name += " [" + agent_name + "]"
            dot.node(node.id, vis_name)
            for child in node.children:
                if visit(child):
                    dot.edge(node.id, child.id)
            return True

        visit(self)
        dot.render(filename, format="png", cleanup=True)

    def traverse(self) -> List["LightningTrace"]:
        """
        Traverse the trace tree and return a list of all spans.
        """
        spans = [self]
        for child in self.children:
            spans.extend(child.traverse())
        return spans

    def to_json(self) -> dict[str, ReadableSpan]:
        return {
            "id": self.id,
            "span": self.span.to_json(),
            "children": [child.to_json() for child in self.children],
        }

    @classmethod
    def from_spans(cls, spans: List[ReadableSpan]) -> "LightningTrace":
        """
        Create a LightningTrace from a list of spans.
        All spans without parents found will be considered as candidate root spans.
        If multiple root spans are found, a virtual root span will be created as the parent of all root spans.
        """

        # Process trace items in topological order
        id_to_span = {span.get_span_context().span_id: span for span in spans}

        forward_graph: dict[int, list[int]] = {}
        root_ids: list[int] = []
        for span in spans:
            if span.parent is None:
                root_ids.append(span.get_span_context().span_id)
            else:
                if span.parent.span_id not in forward_graph:
                    forward_graph[span.parent.span_id] = []
                forward_graph[span.parent.span_id].append(
                    span.get_span_context().span_id
                )

        # Diff between span with data and forward_graph keys
        # Sometimes the top-level session span is lost.
        unfound_roots = set(forward_graph.keys()) - set(id_to_span.keys())
        for unfound_root in unfound_roots:
            root_ids.append(unfound_root)

        def visit(node_id):
            children: list[LightningTrace] = []
            if node_id in forward_graph:
                for child_id in forward_graph[node_id]:
                    children.append(visit(child_id))

            if node_id not in id_to_span:
                assert len(children) > 0
                virtual_span = ReadableSpan(
                    context=trace_api.SpanContext(
                        trace_id=children[0].span.get_span_context().trace_id,
                        span_id=node_id,
                        is_remote=False,
                    ),
                    name="virtual-node",
                    kind=trace_api.SpanKind.INTERNAL,
                    attributes={},
                    start_time=children[0].start_time,
                    end_time=children[-1].end_time,
                )
                return cls(
                    trace_api.format_span_id(node_id), virtual_span, children=children
                )
            else:
                return cls(
                    trace_api.format_span_id(node_id),
                    id_to_span[node_id],
                    children=children,
                )

        # Create a virtual root span if multiple root spans are found
        if len(root_ids) > 1:
            root_spans = [visit(root_id) for root_id in root_ids]
            virtual_root = LightningTrace(
                id="virtual-root",
                span=ReadableSpan(
                    context=trace_api.SpanContext(
                        trace_id=root_spans[0].span.get_span_context().trace_id,
                        span_id=0,
                        is_remote=False,
                    ),
                    name="virtual-root",
                    kind=trace_api.SpanKind.INTERNAL,
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

        # Case 1: OpenAI Agent SDK
        agent_name = self.span.attributes.get("agent.name")
        if agent_name is not None:
            return agent_name

        # Case 2: Agentops decorator @agent
        is_agent = self.span.attributes.get("agentops.span.kind") == "agent"
        if is_agent:
            agent_name = self.span.attributes.get("operation.name")
            if agent_name is not None:
                return agent_name

        # Case 3: Autogen team
        agent_name = self.span.attributes.get("recipient_agent_type")
        if agent_name is not None:
            return agent_name

        # Case 4: LangGraph
        agent_name = self.span.attributes.get("langchain.chain.type")
        if agent_name is not None:
            return agent_name

    def maybe_reward_dict(self) -> dict[str, Any]:
        output = self.span.attributes.get("agentops.entity.output")
        if output:
            if isinstance(output, dict):
                return output
            elif isinstance(output, str):
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return {}
        return {}

    def is_reward_span(self) -> bool:
        maybe_reward = self.maybe_reward_dict()
        return maybe_reward and maybe_reward.get("type") == "reward"

    def find_llm_calls(
        self,
        llm_call_match: str,
        agent_match: Optional[str],
        within_matching_subtree: str | None = None,
        within_reward: Optional[bool] = None,
    ) -> List[Tuple["LightningTrace", str]]:
        """Find all LLM calls in the trace tree.

        The LLM call is defined as a span with type = request and name matching `llm_call_match`.
        If `agent_match` is not None, it must also reside in an agent span (type = agent) with name matched.

        Return a list of traces and the agent names (why it's selected).
        """
        llm_calls: List[Tuple[LightningTrace, str]] = []

        if within_matching_subtree is not None and (within_reward is None or not within_reward):
            # We are in an interesting agent subtree, and not in a reward span.
            if re.search(llm_call_match, self.span.name):
                llm_calls.append((self, within_matching_subtree))

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
                    llm_call_match, agent_match, within_matching_subtree, within_reward
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
            # Find the closest parent span
            closest_parent = None
            closest_duration = float("inf")
            for node in self.traverse():
                if node.id == repair_node.id:
                    continue
                if (
                    node.start_time <= repair_node.start_time
                    and node.end_time >= repair_node.end_time
                ):
                    duration_delta = (
                        node.end_time
                        - repair_node.end_time
                        + repair_node.start_time
                        - node.start_time
                    )
                    if duration_delta < closest_duration:
                        closest_duration = duration_delta
                        closest_parent = node

            # Repair the hierarchy
            if closest_parent is not None:
                self.children.remove(repair_node)
                closest_parent.children.append(repair_node)

    def match_rewards(
        self, reward_match: str, llm_calls: List["LightningTrace"]
    ) -> dict[str, Optional[float]]:
        """Match the rewards to the LLM calls."""
        llm_call_ids = set([llm_call.id for llm_call in llm_calls])
        rewards: dict[str, Optional[float]] = {}

        if reward_match == RewardMatchPolicy.FIRST_OCCURRENCE:
            time_sorted: List[LightningTrace] = sorted(
                self.traverse(), key=lambda x: x.start_time
            )
            assign_to: int | None = None
            for item in time_sorted:
                if item.id in llm_call_ids:
                    assign_to = item.id

                # get reward
                agentops_output = item.maybe_reward_dict()
                if agentops_output and agentops_output.get("type") == "reward":
                    if assign_to is not None:
                        # Ok, we found someone to assign to
                        if assign_to in rewards:
                            # If the reward is already set, skip
                            continue
                        rewards[assign_to] = agentops_output.get("value", None)

        elif reward_match == RewardMatchPolicy.FIRST_SIBLING:
            for item in self.traverse():
                assign_to: int | None = None
                for child in item.children:
                    if child.id in llm_call_ids:
                        assign_to = child.id

                    agentops_output = item.maybe_reward_dict()
                    if agentops_output and agentops_output.get("type") == "reward":
                        if assign_to is not None:
                            if assign_to in rewards:
                                continue
                            rewards[assign_to] = agentops_output.get("value", None)

        return rewards

    def to_trajectory(
        self,
        llm_call_match: str = r"openai\.chat\.completion",
        agent_match: Optional[str] = None,
        exclude_llm_call_in_reward: bool = True,
        reward_match: RewardMatchPolicy = RewardMatchPolicy.FIRST_OCCURRENCE,
        final_reward: Optional[float] = None,
    ) -> List[Transition]:
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
            llm_call_match, agent_match, '*' if agent_match is None else None, False if exclude_llm_call_in_reward else None
        )
        id_transitions = [
            (
                llm_call.id,
                Transition(
                    state=llm_call.span.attributes.get("prompt_token_ids", []),
                    action=llm_call.span.attributes.get("response_token_ids", []),
                    response_id=llm_call.span.attributes.get(
                        "gen_ai.response.id", None
                    ),  # it works at least for OpenAI
                    agent_name=agent_name,
                    reward=None,
                ),
            )
            for llm_call, agent_name in llm_calls
        ]

        rewards = self.match_rewards(reward_match, [call for call, _ in llm_calls])
        transitions = [
            transition.model_copy(update={"reward": rewards.get(id, None)})
            for id, transition in id_transitions
        ]
        if final_reward is not None and len(transitions) > 0:
            # Add the final reward to the last transition
            transitions[-1] = transitions[-1].model_copy(update={"reward": final_reward})
        return transitions

    def __repr__(self):
        return (
            f"LightningTrace(id={self.id}, span={self.span}, start_time={self.start_time}, "
            + f"end_time={self.end_time}, children={self.children})"
        )


class LightningSpanProcessor(SpanProcessor):

    _last_trace: Optional[LightningTrace] = None
    _spans: List[ReadableSpan] = []

    def __enter__(self):
        self._last_trace = None
        self._spans = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def last_trace(self, repair_hierarchy: bool = True) -> LightningTrace:
        """
        Get and convert the last trace into a tree.

        When `repair_hierarchy` is set to True, the trace will be repaired with the time information.
        See `LightningTrace.repair_hierarchy` for more details.
        """
        if self._last_trace is None:
            self._last_trace = LightningTrace.from_spans(self._spans)
            if repair_hierarchy:
                self._last_trace.repair_hierarchy()
        return self._last_trace

    def on_end(self, span: ReadableSpan) -> None:
        # Skip if span is not sampled
        if not span.context or not span.context.trace_flags.sampled:
            return

        self._spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


_global_lightning_span_processor: Optional[LightningSpanProcessor] = None


def lightning_span_processor():
    global _global_lightning_span_processor
    if _global_lightning_span_processor is None:
        _global_lightning_span_processor = LightningSpanProcessor()
        TracingCore.get_instance()._provider.add_span_processor(
            _global_lightning_span_processor
        )
    return _global_lightning_span_processor
