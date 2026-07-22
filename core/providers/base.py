"""
Base Provider Client Interface (core/providers/base.py)
Defines the standard contract and result data structures for LLM inference providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: Optional[Any] = None


class ProviderClient(ABC):
    """
    Abstract interface for LLM inference providers.
    All providers (Groq, Ollama, OpenAI, Anthropic) must implement this contract.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Executes a chat completion request and returns a normalized CompletionResult.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Returns the active model name for this provider.
        """
        pass
