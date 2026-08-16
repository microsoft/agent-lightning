# Copyright (c) Microsoft. All rights reserved.

"""Trace a LangGraph agent with Agent Lightning.

This example runs a tiny two-node [LangGraph](https://github.com/langchain-ai/langgraph)
workflow — one deterministic chat-model node and one plain Python node —
under Agent Lightning's AgentOps tracer, then reads the captured spans
back from the store.

The whole run is offline: the chat model is LangChain's deterministic
``FakeMessagesListChatModel``, so no API keys, GPU, or network calls are
required.

Run it with:

```bash
uv sync --frozen --group dev --group langchain --no-default-groups
python examples/langgraph/trace_langgraph.py
```
"""

import asyncio
from typing import Any, Dict

try:
    # langchain-core >= 1.0 moved the fake chat models here.
    from langchain_core.fakes import FakeMessagesListChatModel
except ImportError:  # pragma: no cover - older langchain-core
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console

from agentlightning import AgentOpsTracer, setup_logging
from agentlightning.store import InMemoryLightningStore

console = Console()


def build_graph() -> CompiledStateGraph:
    """Build a two-node LangGraph workflow.

    ``say_hello`` runs a deterministic fake chat model; ``reverse_text``
    is a plain Python node. After the run, the workflow execution and the
    model call must both show up as captured spans.
    """
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="Hello from the fake model!")])

    def say_hello(state: MessagesState) -> Dict[str, Any]:
        """Ask the (fake) chat model for a greeting."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def reverse_text(state: MessagesState) -> Dict[str, Any]:
        """Reverse the last message's content without any model."""
        last = state["messages"][-1]
        assert isinstance(last.content, str), f"Expected text content, got {type(last.content)}"
        return {"messages": [AIMessage(content=last.content[::-1])]}

    graph = StateGraph(MessagesState)
    graph.add_node("say_hello", say_hello)
    graph.add_node("reverse_text", reverse_text)
    graph.add_edge(START, "say_hello")
    graph.add_edge("say_hello", "reverse_text")
    graph.add_edge("reverse_text", END)
    return graph.compile()


async def main() -> None:
    """Run the graph under the AgentOps tracer and verify the spans."""
    setup_logging()

    tracer = AgentOpsTracer(agentops_managed=True, instrument_managed=False)
    store = InMemoryLightningStore()
    rollout = await store.start_rollout(input={"origin": "langgraph_tracing_example"})
    graph = build_graph()

    with tracer.lifespan(store):
        async with tracer.trace_context(
            "langgraph-run",
            rollout_id=rollout.rollout_id,
            attempt_id=rollout.attempt.attempt_id,
        ):
            handler = tracer.get_langchain_handler()
            result = graph.invoke(
                {"messages": [HumanMessage(content="Hello!")]},
                {"callbacks": [handler]} if handler else None,
            )
            console.print(result)

    spans = await store.query_spans(rollout_id=rollout.rollout_id)
    console.print(spans)

    span_names = [span.name for span in spans]
    # The exact span names are owned by the AgentOps instrumentation and
    # may change between versions, so assert on structural guarantees
    # instead: at least one span carries the langgraph workflow marker
    # and at least one model call was traced.
    assert any("langgraph" in name for name in span_names), span_names
    assert any("llm" in name or "model" in name for name in span_names), span_names
    console.print("[green]The LangGraph workflow and its model call were captured as spans.[/green]")


if __name__ == "__main__":
    asyncio.run(main())
