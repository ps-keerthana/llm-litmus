"""
Core Generator Module (core/generator.py)
Thin orchestration layer that interfaces with LLM providers via the
ProviderClient abstraction. Handles caching, cost calculation, and latency tracking.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from core.providers import get_provider_client, CompletionResult

logger = logging.getLogger(__name__)

# Internal scheduler side-channel: groq.py writes the cumulative sleep time
# accumulated during rate-limit waits for a single generate_answer() call.
# Callers should read .sleep_time from the returned GenerationResult instead
# of accessing these globals directly — the globals are an implementation detail.
LAST_API_SLEEP_TIME: float = 0.0
WAS_LAST_CALL_CACHED: bool = False


@dataclass
class GenerationResult:
    """
    Structured return value from generate_answer().

    Prefer reading sleep_time and was_cached from this object rather than
    from the module-level globals (LAST_API_SLEEP_TIME / WAS_LAST_CALL_CACHED),
    which are maintained only for backward compatibility with groq.py's scheduler.
    """
    answer: str
    prompt_tokens: int
    completion_tokens: int
    sleep_time: float = 0.0   # seconds spent sleeping due to rate limits
    was_cached: bool = False   # True if the result came from the cache

    def as_tuple(self) -> Tuple[str, int, int]:
        """Backward-compatible (answer, prompt_tokens, completion_tokens) tuple."""
        return self.answer, self.prompt_tokens, self.completion_tokens


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
) -> GenerationResult:
    """
    Generates an answer using retrieved document context via configured ProviderClient.
    Handles content-addressable caching and latency tracking.

    Returns:
        GenerationResult with answer, token counts, sleep_time, and was_cached flag.
        Call .as_tuple() for the legacy (answer, prompt_tokens, completion_tokens) form.
    """
    global LAST_API_SLEEP_TIME, WAS_LAST_CALL_CACHED
    LAST_API_SLEEP_TIME = 0.0
    WAS_LAST_CALL_CACHED = False

    context = "\n\n".join(context_chunks)

    if system_prompt is None:
        import config as _cfg
        system_prompt = (
            f"You are an expert assistant for a {_cfg.DOMAIN_DESCRIPTION}. "
            "Answer the question using ONLY the provided context.\n"
            "1. Multi-Step Reasoning: Combine facts and criteria from different parts of the context "
            "to calculate totals, compare options, or deduce complete answers.\n"
            "2. False Premises: If the question assumes something not supported by the context, "
            "explicitly correct it using the context or say 'I don't have information about that.'\n"
            "3. Refusal Rule: Do not refuse if the conclusion can be deduced from the context. "
            "If the answer cannot be found or deduced, say 'I don't have information about that.'"
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
        return GenerationResult(
            answer=cached["answer"],
            prompt_tokens=cached["prompt_tokens"],
            completion_tokens=cached["completion_tokens"],
            sleep_time=0.0,
            was_cached=True,
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
            "completion_tokens": completion_tokens,
        })

        # Read sleep time written by groq.py's scheduler (side-channel global)
        sleep_time = LAST_API_SLEEP_TIME

        return GenerationResult(
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sleep_time=sleep_time,
            was_cached=False,
        )

    except Exception as e:
        logger.error(
            "[Generator] Failed to generate answer via provider '%s': %s",
            provider.get_model_name(), e,
        )
        return GenerationResult(
            answer="I don't have information about that. (Error: LLM generation failed)",
            prompt_tokens=0,
            completion_tokens=0,
            sleep_time=0.0,
            was_cached=False,
        )


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculates the approximate USD cost of an API call.
    Delegates to core.utils.calculate_cost which reads pricing from config.
    """
    from core.utils import calculate_cost as _calculate_cost
    return _calculate_cost(prompt_tokens, completion_tokens)
