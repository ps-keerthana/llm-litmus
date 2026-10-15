"""
Ollama Provider Client (core/providers/ollama.py)
Implements ProviderClient for local or remote Ollama endpoints using the
OpenAI-compatible client interface (http://localhost:11434/v1).
Bypasses SQLite rate limit scheduler since Ollama has no quota limits.
"""

import logging
from typing import List, Dict, Any, Optional
from config import OLLAMA_API_URL, OLLAMA_MODEL_NAME
from core.providers.base import ProviderClient, CompletionResult

logger = logging.getLogger(__name__)


class OllamaProviderClient(ProviderClient):
    """
    Ollama LLM provider using OpenAI-compatible API protocol.
    No rate-limiting or scheduler required.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.base_url = base_url or OLLAMA_API_URL
        self.model_name = model_name or OLLAMA_MODEL_NAME
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "The 'openai' Python package is required to use the Ollama provider. "
                    "Please run 'pip install openai'."
                ) from e

            self._client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",  # Dummy key required by OpenAI client protocol
                timeout=120.0,
            )
        return self._client

    def get_model_name(self) -> str:
        return self.model_name

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        req_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            req_kwargs["response_format"] = response_format

        try:
            response = self._get_client().chat.completions.create(**req_kwargs)
            if response is None or not response.choices:
                raise ValueError("Ollama endpoint returned an empty response choices list.")

            text = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            p_tokens = usage.prompt_tokens if usage else 0
            c_tokens = usage.completion_tokens if usage else 0
            t_tokens = usage.total_tokens if usage else (p_tokens + c_tokens)

            return CompletionResult(
                text=text,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                raw_response=response,
            )
        except Exception as e:
            logger.error("[Ollama Provider Error] Generation failed: %s", e)
            raise
