from typing import Dict, Any
from agentops.integration.callbacks.langchain import LangchainCallbackHandler
from agentops import instrumentation


original_on_chain_start = LangchainCallbackHandler.on_chain_start
langgraph_entry = None


def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
    if "name" in kwargs:
        if serialized is None:
            serialized = {}
        serialized = serialized.copy()
        serialized["name"] = kwargs["name"]
    if "run_id" in kwargs:
        if serialized is None:
            serialized = {}
        serialized = serialized.copy()
        if "id" not in serialized:
            serialized["id"] = kwargs["run_id"]
    return original_on_chain_start(self, serialized, inputs, **kwargs)


def instrument_agentops_langchain():
    global langgraph_entry
    langgraph_entry = instrumentation.AGENTIC_LIBRARIES.pop("langgraph", None)
    LangchainCallbackHandler.on_chain_start = on_chain_start


def uninstrument_agentops_langchain():
    global langgraph_entry
    if langgraph_entry is not None:
        instrumentation.AGENTIC_LIBRARIES["langgraph"] = langgraph_entry
        langgraph_entry = None
    LangchainCallbackHandler.on_chain_start = original_on_chain_start
