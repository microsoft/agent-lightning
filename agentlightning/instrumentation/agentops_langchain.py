from typing import Dict, Any
from agentops.integration.callbacks.langchain import LangchainCallbackHandler


original_on_chain_start = LangchainCallbackHandler.on_chain_start


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
    LangchainCallbackHandler.on_chain_start = on_chain_start
