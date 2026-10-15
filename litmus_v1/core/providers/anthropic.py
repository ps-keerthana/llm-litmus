"""
Anthropic Provider Client Placeholder (core/providers/anthropic.py)
Lightweight stub for Anthropic API expansion.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from config import ANTHROPIC_MODEL_NAME
from core.providers.base import ProviderClient, CompletionResult

logger = logging.getLogger(__name__)


class AnthropicProviderClient(ProviderClient):
    """
    Anthropic API provider implementation placeholder.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or ANTHROPIC_MODEL_NAME

    def get_model_name(self) -> str:
        return self.model_name

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        raise NotImplementedError("Anthropic provider client is a stub. Configure ANTHROPIC_API_KEY to enable.")
