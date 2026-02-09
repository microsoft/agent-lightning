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
from agentlightning.utils.multi_provider_ai_client import MultiProviderClient
client = MultiProviderClient()

# Use with APO
algo = agl.APO(
    client,
    gradient_model="gemini/gemini-2.0-flash",
    apply_edit_model="groq/llama-3.3-70b-versatile",
)

"""

from litellm import acompletion

class MultiProviderClient:
    """Async client that routes to different providers using LiteLLM.
    Uses standard LiteLLM 'provider/model' format.
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the client. LiteLLM automatically picks up API keys 
        from environment variables (e.g., GOOGLE_API_KEY, GROQ_API_KEY).
        """
        pass

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
            """
            Passes the request directly to LiteLLM for routing.
            
            Args:
            model: String in "provider/model_name" format.
            **kwargs: Additional arguments for the completion call.
            """
            print("--- Multi Provider Client (LiteLLM) ---")
            print(f"Routing to: {model}")

            return await acompletion(model=model, **kwargs)