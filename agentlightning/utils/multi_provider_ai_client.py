"""
Multi-Provider Client
========================
Async client that routes to different LLM providers based on model name.

Usage:
  - google-gemini-2.0-flash  → Google API
  - groq-llama-3.3-70b       → Groq API
  
Model name must be in "provider-model" format.

## Usage
```python
from agentlightning.utils.multi_provider_ai_client import MultiProviderClient
client = MultiProviderClient()
# Use with APO
algo = agl.APO(
    client,
    gradient_model="google-gemini-2.0-flash",
    apply_edit_model="groq-llama-3.3-70b-versatile",
)


"""

import os
from openai import AsyncOpenAI


class MultiProviderClient:
    """Async client that routes to different providers based on model name.
    Model format: "provider-model_name"
    Examples:
        - google-gemini-2.0-flash
        - groq-meta-llama/llama-4-maverick-17b-128e-instruct
        - etc.
    """
    
    def __init__(self, custom_providers: dict[str, dict] | None = None):
        """
        Args:
            custom_providers: Additional providers. Format:
                {
                    "provider_name": {
                        "api_key": "...",  # or env var name
                        "base_url": "https://..."
                    }
                }
        """

        self.clients = {}
        
        # Only create clients for providers with API keys
        if os.getenv("GOOGLE_API_KEY"):
            self.clients["google"] = AsyncOpenAI(
                api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        
        if os.getenv("GROQ_API_KEY"):
            self.clients["groq"] = AsyncOpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1")
        
        if os.getenv("OPENAI_API_KEY"):
            self.clients["openai"] = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"))
        
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.clients["azure"] = AsyncOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                base_url=os.getenv("AZURE_OPENAI_ENDPOINT"))
        
        if os.getenv("OPENROUTER_API_KEY"):
            self.clients["openrouter"] = AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1")
        
        # Add custom providers
        if custom_providers:
            for name, config in custom_providers.items():
                api_key = config.get("api_key") or os.getenv(config.get("api_key_env", ""))
                base_url = config.get("base_url")
                self.clients[name] = AsyncOpenAI(api_key=api_key, base_url=base_url)


    def _parse_model(self, model: str) -> tuple[str, str]:
        """Parse model name into provider and actual model name.
        
        Args:
            model: String in "provider-model_name" format
            
        Returns:
            (provider, actual_model_name) tuple
        """
        if "-" not in model:
            raise ValueError(f"Model format must be 'provider-model_name': {model}")
        
        idx = model.find("-")
        provider = model[:idx]
        actual_model = model[idx + 1:]
        
        if provider not in self.clients:
            for name in self.clients:
                if model.startswith(name + "-"):
                    provider = name
                    actual_model = model[len(name) + 1:]
                    break
            else:
                raise ValueError(f"Unknown provider: {provider}. Supported: {list(self.clients.keys())}")
        
        return provider, actual_model


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
            provider, actual_model = self.parent._parse_model(model)
            client = self.parent.clients[provider]
            print("--- Multi Provider Client ---")
            print(f"{provider.upper()}: {actual_model}")
            return await client.chat.completions.create(model=actual_model, **kwargs)
