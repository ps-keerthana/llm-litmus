"""
Core Generator Module (core/generator.py)
Thin orchestration layer that interfaces with LLM providers via the
ProviderClient abstraction. Handles caching, cost calculation, and latency tracking.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from core.providers import get_provider_client, CompletionResult

logger = logging.getLogger(__name__)

# Module-level globals to track wait latency and cache status
LAST_API_SLEEP_TIME: float = 0.0
WAS_LAST_CALL_CACHED: bool = False


def call_groq_with_retry(
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, str]] = None,
    temperature: float = 0.0,
    max_retries: int = 8,
    _estimated_tokens: int = 0,
) -> Any:
    """
    Backward-compatible wrapper for Groq provider execution.
    Delegates directly to GroqProviderClient.
    """
    from core.providers.groq import GroqProviderClient
    client = GroqProviderClient()
    result = client.complete(
        messages=messages,
        temperature=temperature,
        response_format=response_format,
        max_retries=max_retries,
        _estimated_tokens=_estimated_tokens,
    )
    return result.raw_response


def generate_answer(
    question: str,
    context_chunks: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    provider_name: Optional[str] = None,
) -> Tuple[str, int, int]:
    """
    Generates an answer using retrieved document context via configured ProviderClient.
    Handles content-addressable caching and latency tracking.

    Returns:
        (generated_answer, prompt_tokens, completion_tokens)
    """
    global LAST_API_SLEEP_TIME, WAS_LAST_CALL_CACHED
    LAST_API_SLEEP_TIME = 0.0
    WAS_LAST_CALL_CACHED = False

    context = "\n\n".join(context_chunks)

    if system_prompt is None:
        system_prompt = (
            "You are an expert Indian income tax assistant. Answer the question using ONLY the provided context.\n"
            "1. Multi-Step Deduction: Combine facts, limits, and eligibility criteria from different parts of the context "
            "(e.g., Section 80C limits, Section 80D medical insurance, or age-based slab rules) to calculate totals or deduce complete answers.\n"
            "2. Adversarial & False Premises: If a question assumes a false tax rule or asks about a provision not supported by the context, "
            "explicitly state the facts according to the context or say \"I don't have information about that.\"\n"
            "3. Refusal Rule: Do not refuse to answer if the conclusion can be directly deduced from the context. "
            "However, if the answer cannot be deduced or found in the context, say \"I don't have information about that.\""
        )

    prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""

    provider = get_provider_client(provider_name)
    model_name = provider.get_model_name()

    # Check cache first to bypass provider calls and reduce query overhead
    from core.cache import get_cache_key, lookup_cache, update_cache
    cache_key = get_cache_key(model_name, prompt, temperature)
    cached = lookup_cache(cache_key)
    if cached is not None:
        WAS_LAST_CALL_CACHED = True
        return (
            cached["answer"],
            cached["prompt_tokens"],
            cached["completion_tokens"]
        )

    try:
        result: CompletionResult = provider.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        answer = result.text.strip()
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens

        # Save to cache
        update_cache(cache_key, {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        })

        return answer, prompt_tokens, completion_tokens

    except Exception as e:
        logger.error("[Generator] Failed to generate answer via provider '%s': %s", provider.get_model_name(), e)
        return "I don't have information about that. (Error: LLM generation failed)", 0, 0


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculates the approximate USD cost of an API call.
    Delegates to core.utils.calculate_cost which reads pricing from config.
    """
    from core.utils import calculate_cost as _calculate_cost
    return _calculate_cost(prompt_tokens, completion_tokens)
