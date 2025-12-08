# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Dict

from agentops import instrumentation
from agentops.integration.callbacks.langchain import LangchainCallbackHandler

original_on_chain_start = LangchainCallbackHandler.on_chain_start
langgraph_entry = None

__all__ = [
    "instrument_agentops_langchain",
    "uninstrument_agentops_langchain",
]


def on_chain_start(self: Any, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
    if "name" in kwargs:
        if serialized is None:  # type: ignore
            serialized = {}
        serialized = serialized.copy()
        serialized["name"] = kwargs["name"]
    if "run_id" in kwargs:
        if serialized is None:  # type: ignore
            serialized = {}
        serialized = serialized.copy()
        if "id" not in serialized:
            serialized["id"] = kwargs["run_id"]
    return original_on_chain_start(self, serialized, inputs, **kwargs)


def instrument_agentops_langchain():
    """Bypass AgentOp's native support for Langchain."""
    global langgraph_entry
    langgraph_entry = instrumentation.AGENTIC_LIBRARIES.pop("langgraph", None)
    LangchainCallbackHandler.on_chain_start = on_chain_start


def uninstrument_agentops_langchain():
    """Restore AgentOp's native support for Langchain."""
    global langgraph_entry
    if langgraph_entry is not None:
        instrumentation.AGENTIC_LIBRARIES["langgraph"] = langgraph_entry
        langgraph_entry = None
    LangchainCallbackHandler.on_chain_start = original_on_chain_start
