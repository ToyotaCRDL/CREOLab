from .base_client import BaseLLMClient
from .openai_client import OpenAIVisionClient
from .anthropic_client import AnthropicVisionClient
from .gemini_client import GeminiVisionClient
from .llama_vertex_client import LlamaVertexClient
from .base_models import *

try:
    from ..utils.config_loader import config_loader
except ImportError:
    from utils.config_loader import config_loader


def create_llm_client(model=None):
    """Factory: return the appropriate LLM client based on config provider."""
    provider = config_loader.get_provider(default="openai")

    if provider == "openai":
        client = OpenAIVisionClient(model=model)
    elif provider in ("gemini", "gemini3flash"):
        client = GeminiVisionClient(model=model)
    elif provider == "llama":
        client = LlamaVertexClient(model=model)
    elif provider in ("claude_sonnet", "claude_opus", "claude_opus_thinking", "claude_haiku"):
        client = AnthropicVisionClient(model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    print(f"[LLM] provider={provider}, model={client.model}")
    return client


__all__ = [
    'BaseLLMClient',
    'OpenAIVisionClient',
    'AnthropicVisionClient',
    'GeminiVisionClient',
    'LlamaVertexClient',
    'create_llm_client',
]
