# Copyright (c) Microsoft. All rights reserved.

__version__ = "0.3.0"

from .adapter import *
from .algorithm import *
from .client import AgentLightningClient, DevTaskLoader  # deprecated  # type: ignore
from .config import *
from .emitter import *
from .env_var import *
from .execution import *
from .litagent import *
from .llm_proxy import *
from .logging import configure_logger  # deprecated  # type: ignore
from .logging import setup as setup_logging  # type: ignore
from .logging import setup_module as setup_module_logging  # type: ignore
from .runner import *
from .server import AgentLightningServer  # deprecated  # type: ignore
from .store import *
from .tracer import *
from .trainer import *
from .types import *

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, cast

def wrap_message(content: str) -> str:
    start_token = '<AGL_MESSAGE_START>'
    end_token = '<AGL_MESSAGE_END>'
    if not content.startswith(start_token):
        content = start_token + content
    if not content.endswith(end_token):
        content += end_token
    return content


def add_message(message1: str | List[Dict[str, Any]], message2: str | Dict[str, Any]) -> str | List[Dict[str, Any]]:
    
    if isinstance(message1, list):
        new_message = []
        for msg in message1:
            new_message.append({"role": msg["role"], "content": wrap_message(msg["content"])})
        new_message.append({"role": message2["role"], "content": wrap_message(message2["content"])})
        return new_message
    else:
        return wrap_message(message1) + wrap_message(message2)
