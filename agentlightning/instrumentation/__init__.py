import warnings

AGENTOPS_INSTALLED = False
AGENTOPS_LANGCHAIN_INSTALLED = False
LITELLM_INSTALLED = False
VLLM_INSTALLED = False

try:
    from . import agentops

    AGENTOPS_INSTALLED = True
except ImportError:
    pass

try:
    from . import litellm

    LITELLM_INSTALLED = True
except ImportError:
    pass

try:
    from . import vllm

    VLLM_INSTALLED = True
except ImportError:
    pass


try:
    from . import agentops_langchain

    AGENTOPS_LANGCHAIN_INSTALLED = True
except ImportError:
    pass


def instrument_all():
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
