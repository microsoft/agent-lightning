"""
Multi-Provider Client (LiteLLM Version)
========================================
Async client that routes to different LLM providers using LiteLLM.

Usage:
  - gemini/gemini-2.0-flash  → Google API
  - groq/llama-3.3-70b       → Groq API
  - ollama/llama3            → Local Ollama
  - openai/<model_name>      → OpenAI or Custom Base URL

Model name should follow the standard LiteLLM "provider/model" format.

## Usage
```python
from agentlightning.utils.multi_provider_client import MultiProviderClient
client = MultiProviderClient()

```
"""

from litellm import acompletion

class MultiProviderClient:
    """Async client that routes to different providers using LiteLLM."""
    
    def __init__(self, **kwargs):
        print("--- Multi Provider Client (LiteLLM) Initialized ---")

    @property
    def chat(self):
        return self._ChatProxy(self)
    
    class _ChatProxy:
        def __init__(self, parent):
            self.parent = parent
        
        @property
        def completions(self):
            return self.parent._CompletionsProxy(self.parent)
    
    class _CompletionsProxy:
        def __init__(self, parent):
            self.parent = parent
        
        async def create(self, model: str, **kwargs):
            return await acompletion(model=model, **kwargs)