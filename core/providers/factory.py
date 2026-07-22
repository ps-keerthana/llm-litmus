"""
Provider Factory (core/providers/factory.py)
Creates and manages ProviderClient instances cleanly without spreading
if-else branching across the evaluation codebase.
"""

import logging
from typing import Dict, Optional
from config import LLM_PROVIDER
from core.providers.base import ProviderClient
from core.providers.groq import GroqProviderClient
from core.providers.ollama import OllamaProviderClient
from core.providers.openai import OpenAIProviderClient
from core.providers.anthropic import AnthropicProviderClient

logger = logging.getLogger(__name__)

_PROVIDER_CACHE: Dict[str, ProviderClient] = {}


def get_provider_client(provider_name: Optional[str] = None) -> ProviderClient:
    """
    Factory function to retrieve or instantiate a ProviderClient.
    Defaults to config.LLM_PROVIDER if provider_name is not provided.
    Caches provider instances per provider_name for efficiency.
    """
    target_provider = (provider_name or LLM_PROVIDER).strip().lower()

    if target_provider in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[target_provider]

    if target_provider == "groq":
        client = GroqProviderClient()
    elif target_provider == "ollama":
        client = OllamaProviderClient()
    elif target_provider == "openai":
        client = OpenAIProviderClient()
    elif target_provider == "anthropic":
        client = AnthropicProviderClient()
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{target_provider}'. "
            f"Supported options are: 'groq', 'ollama', 'openai', 'anthropic'."
        )

    _PROVIDER_CACHE[target_provider] = client
    logger.info("[Provider Factory] Initialized provider client: '%s' (model: %s)", target_provider, client.get_model_name())
    return client
