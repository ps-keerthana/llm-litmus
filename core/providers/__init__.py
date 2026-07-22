"""
Core Providers Module (core/providers)
Export base contracts, provider implementations, and factory method.
"""

from core.providers.base import ProviderClient, CompletionResult
from core.providers.factory import get_provider_client
from core.providers.groq import GroqProviderClient
from core.providers.ollama import OllamaProviderClient
from core.providers.openai import OpenAIProviderClient
from core.providers.anthropic import AnthropicProviderClient

__all__ = [
    "ProviderClient",
    "CompletionResult",
    "get_provider_client",
    "GroqProviderClient",
    "OllamaProviderClient",
    "OpenAIProviderClient",
    "AnthropicProviderClient",
]
