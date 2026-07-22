"""
Groq Provider Client (core/providers/groq.py)
Implements ProviderClient for Groq API with robust retries,
rate limit handling, and SQLite-backed token bucket scheduler.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from groq import Groq, RateLimitError
from config import MODEL_NAME, SCHEDULER_ESTIMATED_OUTPUT_TOKENS
from core import scheduler
from core.providers.base import ProviderClient, CompletionResult

logger = logging.getLogger(__name__)


def _parse_retry_after(exc: Exception) -> float:
    default_wait = 15.0
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {})
            if "retry-after-ms" in headers:
                return float(headers["retry-after-ms"]) / 1000.0
            if "retry-after" in headers:
                return float(headers["retry-after"])
    except Exception:
        pass
    return default_wait


class GroqProviderClient(ProviderClient):
    """
    Groq API provider client integrated with the SQLite rate-limit scheduler.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or MODEL_NAME
        self._client: Optional[Groq] = None

    def _get_client(self) -> Groq:
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY is not set. Export it or add it to a .env file.")
            self._client = Groq(api_key=api_key, timeout=30.0, max_retries=0)
        return self._client

    def get_model_name(self) -> str:
        return self.model_name

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 8,
        _estimated_tokens: int = 0,
        **kwargs: Any,
    ) -> CompletionResult:
        # Estimate prompt tokens for scheduler debit
        prompt_text = "".join(m.get("content", "") for m in messages)
        est_tokens = _estimated_tokens or max(1, len(prompt_text.encode("utf-8")) // 3)

        backoff = 2.0
        for attempt in range(max_retries):
            try:
                wait_s = scheduler.acquire(est_tokens)
                import core.generator as _gen_mod
                _gen_mod.LAST_API_SLEEP_TIME += wait_s

                req_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                }
                if response_format:
                    req_kwargs["response_format"] = response_format

                response = self._get_client().chat.completions.create(**req_kwargs)
                if response is None:
                    raise ValueError("Groq API returned None after call.")

                text = response.choices[0].message.content.strip()
                usage = getattr(response, "usage", None)
                p_tokens = usage.prompt_tokens if usage else 0
                c_tokens = usage.completion_tokens if usage else 0
                t_tokens = usage.total_tokens if usage else (p_tokens + c_tokens)

                # Refund over-estimated reservation
                est_total = est_tokens + SCHEDULER_ESTIMATED_OUTPUT_TOKENS
                scheduler.refund(est_total, t_tokens)

                return CompletionResult(
                    text=text,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    raw_response=response,
                )

            except RateLimitError as e:
                scheduler.drain()
                wait = _parse_retry_after(e)
                jitter = wait * 0.1 * (0.5 - (attempt % 2) * 0.5)
                sleep_s = round(wait + jitter, 2)
                logger.info(
                    "[Groq Provider Rate Limit] 429 received (attempt %d/%d). "
                    "Groq says wait %.0fs — sleeping %.1fs then retrying...",
                    attempt + 1, max_retries, wait, sleep_s,
                )
                time.sleep(sleep_s)
                import core.generator as _gen_mod
                _gen_mod.LAST_API_SLEEP_TIME += sleep_s

            except Exception as e:
                logger.warning(
                    "[Groq Provider Error] Call failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                if attempt == max_retries - 1:
                    raise
                jitter = backoff * 0.2 * (0.5 - (attempt % 2))
                sleep_s = round(backoff + jitter, 2)
                time.sleep(sleep_s)
                import core.generator as _gen_mod
                _gen_mod.LAST_API_SLEEP_TIME += sleep_s
                backoff = min(backoff * 2.0, 60.0)

        raise RuntimeError("Groq API calls failed after max retries.")
