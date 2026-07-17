"""
Core Generator Module
Interfaces with the Groq API to generate answers based on context,
implementing robust retries, token usage logging, and cost analysis.

Rate limiting is handled externally by core/scheduler.py which uses a
SQLite-backed dual token bucket shared across all processes.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from groq import Groq
from config import MODEL_NAME
from core import scheduler

logger = logging.getLogger(__name__)

# Module-level globals to track wait latency and cache status
LAST_API_SLEEP_TIME: float = 0.0
WAS_LAST_CALL_CACHED: bool = False

# Lazy-initialized Groq client — only created on first actual API call.
# max_retries=0: we disable the SDK's internal retry so OUR wrapper owns
# the full retry policy with proper Retry-After header handling.
_groq_client: Optional[Groq] = None



def _get_groq_client() -> Groq:
    """Returns the singleton Groq client, initializing it on first call."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Export it or add it to a .env file."
            )
        # max_retries=0: disable SDK-level retries entirely.
        # Our call_groq_with_retry() handles all retry logic, including
        # reading the Retry-After header for the correct sleep duration.
        _groq_client = Groq(api_key=api_key, timeout=30.0, max_retries=0)
    return _groq_client


def _parse_retry_after(exc: Exception) -> float:
    """
    Extracts the server-specified retry delay from a RateLimitError.
    Groq sends 'retry-after-ms' (milliseconds) or 'retry-after' (seconds).
    Falls back to a safe default of 15 seconds if the header is absent.
    """
    default_wait = 15.0
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {})
            # Prefer high-precision millisecond header
            if "retry-after-ms" in headers:
                return float(headers["retry-after-ms"]) / 1000.0
            if "retry-after" in headers:
                return float(headers["retry-after"])
    except Exception:
        pass
    return default_wait


def call_groq_with_retry(
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, str]] = None,
    temperature: float = 0.0,
    max_retries: int = 8,
    _estimated_tokens: int = 0,
) -> Any:
    """
    Executes Groq API chat completions with proactive rate scheduling and
    robust fallback retry handling:
    - Blocks proactively via scheduler.acquire() before the first attempt.
    - Honors the Retry-After header for the exact sleep duration on 429s.
    - Exponential backoff with jitter for non-rate-limit errors.
    - Tracks cumulative sleep time in LAST_API_SLEEP_TIME for latency correction.

    Parameters
    ----------
    _estimated_tokens : int
        Caller-supplied prompt token estimate for the scheduler. If 0 the
        scheduler uses its own conservative default.
    """
    global LAST_API_SLEEP_TIME
    from groq import RateLimitError
    from config import SCHEDULER_ESTIMATED_OUTPUT_TOKENS

    backoff = 2.0
    response = None
    for attempt in range(max_retries):
        try:
            # Proactively acquire capacity before each attempt.
            # The scheduler blocks here if the bucket is empty or spacing window is active.
            wait_s = scheduler.acquire(_estimated_tokens)
            LAST_API_SLEEP_TIME += wait_s

            kwargs: Dict[str, Any] = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = _get_groq_client().chat.completions.create(**kwargs)
            return response

        except RateLimitError as e:
            # Scheduler should prevent this; treat as a last-resort safety net.
            # Drain the bucket to prevent subsequent calls from firing while we are blocked.
            scheduler.drain()
            wait = _parse_retry_after(e)
            jitter = wait * 0.1 * (0.5 - (attempt % 2) * 0.5)
            sleep_s = round(wait + jitter, 2)
            logger.info(
                "[Rate Limit] 429 received (attempt %d/%d). "
                "Groq says wait %.0fs — sleeping %.1fs then retrying...",
                attempt + 1, max_retries, wait, sleep_s,
            )
            time.sleep(sleep_s)
            LAST_API_SLEEP_TIME += sleep_s

        except Exception as e:
            logger.warning(
                "[API Error] Groq call failed (attempt %d/%d): %s",
                attempt + 1, max_retries, e,
            )
            if attempt == max_retries - 1:
                raise
            jitter = backoff * 0.2 * (0.5 - (attempt % 2))
            sleep_s = round(backoff + jitter, 2)
            time.sleep(sleep_s)
            LAST_API_SLEEP_TIME += sleep_s
            backoff = min(backoff * 2.0, 60.0)

    return None


def generate_answer(
    question: str,
    context_chunks: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> Tuple[str, int, int]:
    """
    Generates an answer using the retrieved document context.
    Resets LAST_API_SLEEP_TIME at the start of each call so callers can
    accurately subtract sleep overhead from wall-clock latency.

    Returns:
        (generated_answer, prompt_tokens, completion_tokens)
    """
    global LAST_API_SLEEP_TIME, WAS_LAST_CALL_CACHED
    LAST_API_SLEEP_TIME = 0.0
    WAS_LAST_CALL_CACHED = False

    context = "\n\n".join(context_chunks)

    if system_prompt is None:
        system_prompt = (
            "You are a helpful Indian income tax assistant. Answer the question using ONLY the context below.\n"
            "If the context contains a rule, limit, or eligibility criteria that directly implies the answer to the question "
            "(e.g. minimum requirements, numerical comparisons, or exclusions), apply that rule to deduce the answer. "
            "Do not refuse to answer simply because the final conclusion is not written verbatim, provided the logic can be directly deduced from the context.\n"
            "If the answer cannot be deduced or found in the context, say \"I don't have information about that.\""
        )


    prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""

    # Check cache first to bypass Groq limits and reduce query overhead
    from core.cache import get_cache_key, lookup_cache, update_cache
    from config import SCHEDULER_ESTIMATED_OUTPUT_TOKENS
    cache_key = get_cache_key(MODEL_NAME, prompt, temperature)
    cached = lookup_cache(cache_key)
    if cached is not None:
        WAS_LAST_CALL_CACHED = True
        return (
            cached["answer"],
            cached["prompt_tokens"],
            cached["completion_tokens"]
        )

    # Conservative token estimate: 1 token ≈ 3 bytes of UTF-8 text.
    # The scheduler will refund the difference after the actual count arrives.
    estimated_prompt_tokens = max(1, len(prompt.encode("utf-8")) // 3)
    estimated_total = estimated_prompt_tokens + SCHEDULER_ESTIMATED_OUTPUT_TOKENS

    try:
        response = call_groq_with_retry(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            _estimated_tokens=estimated_prompt_tokens,
        )
        if response is None:
            raise ValueError("Groq API returned None after all retries.")

        answer = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        actual_total = prompt_tokens + completion_tokens

        # Return over-estimated tokens to the bucket.
        scheduler.refund(estimated_total, actual_total)
        logger.info(
            "[SCHEDULER] actual_tokens=%d  estimated_tokens=%d  refund=%d",
            actual_total, estimated_total, max(0, estimated_total - actual_total),
        )

        # Save to cache
        update_cache(cache_key, {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        })

        return answer, prompt_tokens, completion_tokens

    except Exception as e:
        logger.error("[Generator] Failed to generate answer: %s", e)
        return "I don't have information about that. (Error: LLM generation failed)", 0, 0


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculates the approximate USD cost of a Groq API call.
    Delegates to core.utils.calculate_cost which reads pricing from config.
    """
    from core.utils import calculate_cost as _calculate_cost
    return _calculate_cost(prompt_tokens, completion_tokens)
