# LangGraph Tracing Example

This example traces a tiny [LangGraph](https://github.com/langchain-ai/langgraph)
workflow with Agent Lightning's AgentOps tracer and verifies that every
graph node is captured as a span in the store — the first step toward
training agents built with LangChain/LangGraph.

The run is fully offline: the chat model is LangChain's deterministic
`FakeMessagesListChatModel`, so no API keys, GPU, or network access are
required.

## Run

```bash
uv sync --frozen --group dev --group langchain --no-default-groups
python examples/langgraph/trace_langgraph.py
```

Expected output: the final graph state plus the captured span list. The
script exits `0` after asserting that the LangGraph workflow execution
and at least one model call were captured in the store.

## Included Files

- `trace_langgraph.py` — builds the two-node graph, runs it under the
  AgentOps tracer inside a rollout, then reads the spans back from the
  in-memory store and asserts the workflow and model call were captured.
