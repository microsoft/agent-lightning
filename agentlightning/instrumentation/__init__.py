# Copyright (c) Microsoft. All rights reserved.

import warnings

AGENTOPS_INSTALLED: bool = False
AGENTOPS_LANGCHAIN_INSTALLED: bool = False
LITELLM_INSTALLED: bool = False
VLLM_INSTALLED: bool = False

try:
    from . import agentops  # type: ignore

    AGENTOPS_INSTALLED = True  # type: ignore
except ImportError:
    pass

try:
    from . import litellm  # type: ignore

    LITELLM_INSTALLED = True  # type: ignore
except ImportError:
    pass

# MAGIC! DO NOT TOUCH THIS!
# vllm import will cause reward tracing function to fail and produce nothing.
# try:
#     from . import vllm

#     VLLM_INSTALLED = True
# except ImportError:
#     pass


try:
    from . import agentops_langchain  # type: ignore

    AGENTOPS_LANGCHAIN_INSTALLED = True  # type: ignore
except ImportError:
    pass


def instrument_all():
    """Instrument all the instrumentation libraries."""
    if AGENTOPS_INSTALLED:
        from .agentops import instrument_agentops

        instrument_agentops()
    else:
        warnings.warn("agentops is not installed. It's therefore not instrumented.")

    if LITELLM_INSTALLED:
        from .litellm import instrument_litellm

        instrument_litellm()
    else:
        warnings.warn("litellm is not installed. It's therefore not instrumented.")

    if VLLM_INSTALLED:
        from .vllm import instrument_vllm

        instrument_vllm()
    else:
        warnings.warn("vllm is not installed. It's therefore not instrumented.")

    if AGENTOPS_LANGCHAIN_INSTALLED:
        from .agentops_langchain import instrument_agentops_langchain

        instrument_agentops_langchain()
    else:
        warnings.warn("Agentops-langchain integration is not installed. It's therefore not instrumented.")


def uninstrument_all():
    """Uninstrument all the instrumentation libraries."""
    if AGENTOPS_INSTALLED:
        try:
            from .agentops import uninstrument_agentops

            uninstrument_agentops()
        except ImportError:
            warnings.warn("agentops is installed but uninstrument_agentops could not be imported.")
    else:
        warnings.warn("agentops is not installed. It's therefore not uninstrumented.")

    if LITELLM_INSTALLED:
        try:
            from .litellm import uninstrument_litellm

            uninstrument_litellm()
        except ImportError:
            warnings.warn("litellm is installed but uninstrument_litellm could not be imported.")
    else:
        warnings.warn("litellm is not installed. It's therefore not uninstrumented.")

    if VLLM_INSTALLED:
        try:
            from .vllm import uninstrument_vllm

            uninstrument_vllm()
        except ImportError:
            warnings.warn("vllm is installed but uninstrument_vllm could not be imported.")
    else:
        warnings.warn("vllm is not installed. It's therefore not uninstrumented.")

    if AGENTOPS_LANGCHAIN_INSTALLED:
        try:
            from .agentops_langchain import uninstrument_agentops_langchain

            uninstrument_agentops_langchain()
        except ImportError:
            warnings.warn("agentops_langchain is installed but uninstrument_agentops_langchain could not be imported.")
    else:
        warnings.warn("Agentops-langchain integration is not installed. It's therefore not uninstrumented.")
